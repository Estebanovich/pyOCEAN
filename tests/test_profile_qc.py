"""
Tests of ``diagnostics.profile_qc``: quality control of in-situ T/S profiles and fixed-depth records (gliders,
wire-following profilers, moored CTDs, bottom-temperature loggers) before they are used as truth, and the profile
metrics (mixed-layer and thermocline depth) shared by observations and models.
"""
from __future__ import annotations

import numpy as np
import pytest

from diagnostics import profile_qc as pq


def test_provider_ok_qartod_and_dac_scales():
    q = np.array([1, 2, 3, 4, 9, np.nan])
    np.testing.assert_array_equal(pq.provider_ok(q, "qartod"), [True, True, False, False, False, True])
    d = np.array([0, 1, 2, 3, 4, 5, 8, 9, np.nan])
    np.testing.assert_array_equal(pq.provider_ok(d, "dac"), [True, True, True, False, False, True, False, False, True])
    with pytest.raises(ValueError):
        pq.provider_ok(q, "argo3")


def test_vertical_spike_argo_test():
    p = np.arange(20.0)
    t = 10.0 - 0.1 * p
    t[7] += 3.0                                   # 3 °C single-point spike above 500 dbar (threshold 6 → kept)
    t[12] += 7.0                                  # 7 °C spike → flagged
    pid = np.zeros(20, int)
    bad = pq.vertical_spike(t, p, pid, thr_shallow=6.0, thr_deep=2.0)
    assert bad[12] and not bad[7]
    assert bad.sum() == 1
    # the neighbours of a profile boundary are never compared across profiles
    pid2 = np.r_[np.zeros(10, int), np.ones(10, int)]
    t2 = np.r_[np.full(10, 5.0), np.full(10, 20.0)]
    assert not pq.vertical_spike(t2, np.r_[p[:10], p[:10]], pid2, 6.0, 2.0).any()


def test_split_legs_wire_following_profiler():
    # one-minute bins: down 10 → 60 m in 10 min, park 5 min, up 60 → 10 m in 10 min, then a 2-h gap and a short leg
    t = np.r_[np.arange(10), np.arange(10, 15), np.arange(15, 25), np.arange(145, 148)].astype("timedelta64[m]")
    p = np.r_[np.linspace(10, 60, 10), np.full(5, 60.0), np.linspace(60, 10, 10), [10.0, 11.0, 12.0]]
    leg = pq.split_legs(np.datetime64("2017-05-01T00:00") + t, p, gap=np.timedelta64(30, "m"), min_span=10.0)
    assert set(leg[:10]) == {0} and set(leg[15:25]) == {1}
    assert (leg[10:15] <= 1).all()                      # the park joins a leg or is dropped (−1)
    assert (leg[25:] == -1).all()                       # 2 m span < min_span


def test_bin_profiles_means_per_bin():
    pid = np.array([0, 0, 0, 1, 1])
    z = np.array([0.2, 0.8, 1.5, 0.1, 3.9])
    x = np.array([1.0, 3.0, 5.0, 7.0, 9.0])
    edges = np.arange(0.0, 5.0, 1.0)
    b = pq.bin_profiles(pid, z, x, edges, n_prof=2)
    np.testing.assert_allclose(b[0], [2.0, 5.0, np.nan, np.nan])
    np.testing.assert_allclose(b[1], [7.0, np.nan, np.nan, 9.0])


def test_density_inversion_flags_lighter_water_below():
    sig = np.array([[24.0, 24.5, 25.0, 24.9, 26.0, np.nan, 26.5]])
    bad = pq.density_inversion(sig, thr=0.03)
    np.testing.assert_array_equal(bad[0], [False, False, False, True, False, False, False])
    assert not pq.density_inversion(sig, thr=0.2).any()


def test_stuck_profiles():
    x = np.array([[5.0] * 12 + [np.nan] * 3, np.linspace(5, 6, 15)])
    np.testing.assert_array_equal(pq.stuck_profiles(x, min_n=10), [True, False])


def test_speed_and_duplicates():
    t = np.datetime64("2017-06-01T00:00") + np.array([0, 60, 120, 120, 180], "timedelta64[m]")
    lat = np.array([40.0, 40.01, 40.02, 40.02, 41.0])       # last jump ~110 km in 1 h
    lon = np.full(5, -70.0)
    dup = pq.duplicate_profiles(t, lat, lon)
    np.testing.assert_array_equal(dup, [False, False, False, True, False])
    fast = pq.speed_flags(t, lat, lon, vmax=3.0)
    assert fast[4] and not fast[:4].any()


def test_haversine_km():
    assert pq.haversine_km(40.0, -70.0, 41.0, -70.0) == pytest.approx(111.2, rel=2e-3)


def test_mixed_layer_depth_threshold():
    z = np.arange(0.5, 60.0, 1.0)
    sig = np.where(z < 15.0, 24.0, 24.0 + 0.01 * (z - 15.0) + 1.0)      # step of 1 kg m⁻³ at 15 m
    mld = pq.mixed_layer_depth(z, sig[None, :], z_ref=5.0, dsig=0.03, z_top_max=7.0)
    assert mld[0] == pytest.approx(14.53, abs=0.05)
    # no data shallower than z_top_max → undefined
    sig2 = sig.copy()
    sig2[z < 8.0] = np.nan
    assert np.isnan(pq.mixed_layer_depth(z, sig2[None, :], 5.0, 0.03, 7.0)[0])
    # mixed to the bottom of the profile → NaN (not a defined MLD)
    assert np.isnan(pq.mixed_layer_depth(z, np.full((1, z.size), 24.0), 5.0, 0.03, 7.0)[0])


def test_thermocline_depth_max_gradient():
    z = np.arange(0.5, 100.0, 1.0)
    t = 20.0 - 10.0 / (1.0 + np.exp(-(z - 30.0) / 3.0))                  # tanh-like thermocline at 30 m
    assert pq.thermocline_depth(z, t[None, :], smooth=5)[0] == pytest.approx(30.0, abs=1.0)


def test_pair_profiles_and_bias_drift():
    t0 = np.datetime64("2017-06-01T00:00")
    ta = t0 + np.arange(0, 30 * 24, 6).astype("timedelta64[h]")
    tb = ta + np.timedelta64(30, "m")
    la, lo = np.full(ta.size, 40.0), np.full(ta.size, -70.8)
    lb = la + 0.02                                                     # 2.2 km apart
    ia, ib = pq.pair_profiles(ta, la, lo, tb, lb, lo, max_km=5.0, max_dt=np.timedelta64(2, "h"))
    assert ia.size == ta.size and (ia == ib).all()
    days = (ta - t0) / np.timedelta64(1, "D")
    d = 0.05 + 0.001 * days + np.where(np.arange(ta.size) % 7 == 0, 0.5, 0.0)   # bias, drift, outliers
    r = pq.bias_drift(ta, d)
    assert r["median"] == pytest.approx(0.05 + 0.001 * 15, abs=0.01)
    assert r["drift_per_30d"] == pytest.approx(0.03, abs=0.01)
    assert r["n"] == ta.size


def test_classify_record():
    assert pq.classify(accepted=0.99, problems=[]) == "usable"
    assert pq.classify(accepted=0.90, problems=[]) == "partial"
    assert pq.classify(accepted=0.99, problems=["gap"]) == "partial"
    assert pq.classify(accepted=0.30, problems=[]) == "corrupted"
    assert pq.classify(accepted=0.99, problems=[], fatal=["biased S"]) == "corrupted"


def test_out_of_water_from_conductivity():
    c = np.array([4.5, 0.01, 3.9, np.nan])
    np.testing.assert_array_equal(pq.out_of_water(c, c_min=2.0), [False, True, False, False])


def test_fixed_depth_qc_pressure_jumps():
    # a moored CTD: deployment ends at hour 50 (pressure drops to the surface), bad pressure outliers
    t = np.datetime64("2017-06-01T00:00") + np.arange(60).astype("timedelta64[h]")
    p = np.full(60, 133.0)
    p[50:] = 2.0
    bad = pq.pressure_off_station(p, tol=5.0)
    assert bad[50:].all() and not bad[:50].any()
    assert t.size == bad.size


def test_salinity_on_theta_surface():
    z = np.arange(0.5, 1000.0, 1.0)
    th = 20.0 - 15.0 * z / 1000.0                    # θ = 6 °C at z = 933.8 m
    S = 35.0 + 0.0001 * z
    s6 = pq.salinity_on_theta(z, th[None, :], S[None, :], theta=6.0, z_min=150.0)
    assert s6[0] == pytest.approx(35.0 + 0.0001 * 933.83, abs=1e-4)
    # profile that never reaches θ = 6 °C below z_min → NaN
    assert np.isnan(pq.salinity_on_theta(z, th[None, :] + 5.0, S[None, :], 6.0, 150.0)[0])


def test_pressure_off_station_running_median_keeps_redeployment():
    # two deployments at 452 and 464 dbar (turnaround), with a 2-sample recovery transient at the surface
    p = np.r_[np.full(200, 452.0), [3.0, 5.0], np.full(200, 464.0)]
    bad = pq.pressure_off_station(p, tol=3.0, window=97)
    assert bad[200:202].all()
    assert bad.sum() <= 4                      # only the transient (± a sample at the step)
    assert pq.pressure_off_station(p, tol=3.0).sum() > 150     # the record-median form loses a deployment
