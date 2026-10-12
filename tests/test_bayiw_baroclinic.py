"""
Tests of the BayIW baroclinic benchmark tools (spec docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md).
Pure functions always; the ones reading the MITgcm inputs skip when the external volumes are not mounted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from validation.bayiw_baroclinic import bayiw

HAVE_2LAYER = (bayiw.CASES["2layer"]["dir"] / "input").exists()
HAVE_REF = bayiw.REF.exists()


def _layers(rng, n=12, H=150.0, vanished=(0, 1)):
    h = rng.uniform(1.0, 20.0, n)
    h[list(vanished)] = 0.0                                              # vanished bed layers (z* in the bay)
    h *= H / h.sum()
    zw = -H + np.concatenate([[0.0], np.cumsum(h)])
    return h, 0.5 * (zw[1:] + zw[:-1])


def test_sample_at_depth_is_exact_for_a_linear_profile_with_vanished_layers():
    # review focus 2
    rng = np.random.default_rng(1)
    h, zc = _layers(rng)
    f = 3.0 + 0.1 * zc
    ok = h > 0
    for z0 in np.linspace(zc[ok].min() + 1e-6, zc.max() - 1e-6, 17):
        out = bayiw.sample_at_depth(zc[:, None], f[:, None], h[:, None], z0, np.array([-150.0]))
        np.testing.assert_allclose(out, 3.0 + 0.1 * z0, rtol=1e-12)


def test_sample_at_depth_is_nan_below_the_bed():
    # review focus 1: 250 m under a 150 m bay
    rng = np.random.default_rng(2)
    h, zc = _layers(rng)
    out = bayiw.sample_at_depth(zc[:, None], zc[:, None], h[:, None], -250.0, np.array([-150.0]))
    assert np.isnan(out).all()


def test_sample_at_depth_is_constant_between_the_bed_and_the_deepest_centre():
    rng = np.random.default_rng(3)
    h, zc = _layers(rng)
    f = 3.0 + 0.1 * zc
    k0 = np.flatnonzero(h > 0)[0]
    out = bayiw.sample_at_depth(zc[:, None], f[:, None], h[:, None], -149.9, np.array([-150.0]))
    np.testing.assert_allclose(out, f[k0], rtol=1e-14)


def test_mitgcm_profile_at_takes_the_level_value():
    T = np.arange(50.0)[:, None, None] * np.ones((1, 2, 2))
    z = np.array([-0.5, -bayiw.Z_C[29], -599.0])[:, None, None] * np.ones((1, 2, 2))
    z = -np.abs(z)
    out = bayiw.mitgcm_profile_at(z, T)
    np.testing.assert_array_equal(out[:, 0, 0], [0.0, 29.0, 49.0])


def test_centred_targets_put_both_water_masses_at_bracket_centres():
    # review focus 3: the tie of the 21-layer lock exchange
    rho_a, rho_b, n = 997.526, 998.376, 25
    t = bayiw.centred_targets(rho_a, rho_b, n)
    assert t.size == n + 1 and np.all(np.diff(t) > 0)
    for rho in (rho_a, rho_b):
        k = np.searchsorted(t, rho) - 1
        np.testing.assert_allclose((rho - t[k]) / (t[k + 1] - t[k]), 0.5, atol=1e-9)


def test_profile_targets_leave_a_margin_beyond_the_extremes():
    rho = np.linspace(1000.0, 1006.0, 50)
    t = bayiw.profile_targets(rho, 25)
    assert t.size == 26 and t[0] < rho.min() and t[-1] > rho.max() and np.all(np.diff(t) > 0)


def test_reference_depths_match_the_mitgcm_series():
    np.testing.assert_allclose([bayiw.Z110, bayiw.Z250], [-111.4, -257.75], atol=1e-9)
    assert abs(bayiw.Z_F[-1] + 599.9) < 1e-9


@pytest.mark.skipif(not HAVE_2LAYER, reason="BayIW MITgcm inputs not mounted")
def test_two_layer_inputs_and_model_shapes():
    dx, dy, H, T, wind = bayiw.load_inputs("2layer", "bay")
    assert dx.shape == (bayiw.NX,) and dy.shape == (bayiw.NY,) and T.shape == (50, bayiw.NY, bayiw.NX)
    assert np.nanmax(H) <= 599.9 + 1e-9 and np.isnan(H).any()
    assert 120.0 < H[345, 280] < 170.0                                   # station Seiche is in the bay
    np.testing.assert_allclose(T[0, 200, 280], 22.77, atol=0.01)
    np.testing.assert_allclose(T[-1, 200, 280], 18.52, atol=0.01)
    assert wind.ndim == 3 and wind.shape[1:] == (bayiw.NY, bayiw.NX)


def test_light_targets_keep_the_top_layers_in_zstar():
    # OM4 style (Adcroft et al. 2019): n_light targets lighter than any water keep the top layers on their z* floor;
    # the hybrid rule only makes z layers where the target is lighter than the water
    rho_a, rho_b, n, nl = 997.526, 998.376, 25, 10
    t = bayiw.centred_targets(rho_a, rho_b, n, n_light=nl)
    assert t.size == n + 1 and np.all(np.diff(t) > 0)
    assert (t[:nl + 1] < rho_a).all()                                    # brackets 0 … nl − 1 lighter than the water
    for rho in (rho_a, rho_b):
        k = np.searchsorted(t, rho) - 1
        np.testing.assert_allclose((rho - t[k]) / (t[k + 1] - t[k]), 0.5, atol=1e-9)
    p = bayiw.profile_targets(np.linspace(1000.0, 1006.0, 50), n, n_light=nl)
    assert p.size == n + 1 and np.all(np.diff(p) > 0) and (p[:nl + 1] < 1000.0).all() and p[-1] > 1006.0


from validation.bayiw_baroclinic import compare  # noqa: E402


def test_dominant_period_finds_a_sine():
    t = np.arange(0.0, 60.0, 0.25)
    x = 0.3 * np.sin(2 * np.pi * t / 7.3) + 0.05 * t
    assert abs(compare.dominant_period(t, x) / 7.3 - 1) < 0.01


def test_corr_lag_finds_a_one_hour_delay():
    t = np.arange(0.0, 48.0, 0.25)
    a = np.sin(2 * np.pi * t / 12.0)
    b = np.sin(2 * np.pi * (t - 1.0) / 12.0)
    c0, lag, c = compare.corr_lag(t, a, b)
    assert abs(lag - 1.0) < 1e-9 and c > 0.999 and c0 < c


def test_window_selects_the_requested_hours():
    # review focus 5
    t = np.arange(27.0, 120.25, 0.25)
    tt, xx = compare.window(t, t * 2, 0.0, 60.0)
    assert tt.min() == 27.0 and tt.max() < 60.0 and np.array_equal(xx, tt * 2)


def test_relative_field_error_ignores_land():
    # review focus 4: NaN land in pyOCEAN, masked (NaN after loading) in MITgcm
    a = np.array([[1.0, np.nan], [2.0, 3.0]])
    b = np.array([[1.0, 5.0], [np.nan, 3.0]])
    assert compare.relative_field_error(a, b) == 0.0


def test_interface_thickness_of_a_tanh_profile():
    z = -np.linspace(1.0, 600.0, 600)
    T = 20.65 + 2.125 * np.tanh((z + 116.0) / 10.0)
    # 10 % → 90 % of ΔT of a tanh: 2·atanh(0.8)·L = 21.97 m for L = 10 m
    assert abs(compare.interface_thickness(T[:, None], z, 18.525, 22.775) - 21.97) < 0.5


@pytest.mark.skipif(not HAVE_REF, reason="MITgcm reference series not mounted")
def test_mitgcm_zoom_stations_are_ocean_values():
    # mask_zoom is 1 on land and 0 in the ocean (8112 land cells in the zoom window)
    t, d = compare.mitgcm_stations("2layer", "T110")
    assert t.size == 481 and t[0] == 0.0
    for name, (bay, nobay) in d.items():
        assert np.isfinite(bay).all(), name
    np.testing.assert_allclose(d["Seiche"][0][0], 22.767, atol=1e-3)


def test_mitgcm_source_levels_fill_the_column_bed_first():
    # MITgcm z-levels as the ALE initial source grid: delR clipped at the bed, bed first, T flipped with them
    H = np.array([[599.9, 120.0, np.nan]])
    T = np.broadcast_to(np.arange(50.0)[:, None, None], (50, 1, 3))           # surface level = 0
    h, Ts = bayiw.mitgcm_source(H, T)
    np.testing.assert_allclose(h.sum(axis=0), [[599.9, 120.0, 0.0]], rtol=1e-12)
    assert h[-1, 0, 0] == bayiw.DELR[0] and Ts[-1, 0, 0] == 0.0 and Ts[0, 0, 0] == 49.0
    k = np.flatnonzero(h[:, 0, 1] > 0)[0]                                       # deepest wet level of the 120 m column
    assert Ts[k, 0, 1] == 49 - k and np.isclose(h[k, 0, 1], 120.0 + bayiw.Z_F[49 - k])  # partial level, top at Z_F[49 − k]


def test_mitgcm_depth_rounds_to_the_nearest_level_interface():
    # MITgcm hFacMin = 1 (set_defaults.F:180): a bottom cell with hFac < ½ is dropped, else made full
    # (ini_masks_etc.F:108-120), so the effective depth is a level interface: 164 → 163.3, 129.02 → 126.6 m
    H = np.array([164.0, 129.0212746016822, 599.9, 700.0, np.nan, 0.3, 0.6])
    D = bayiw.mitgcm_depth(H)
    np.testing.assert_allclose(D[:4], [163.3, 126.6, 599.9, 599.9], rtol=1e-12)
    assert np.isnan(D[4]) and np.isnan(D[5]) and D[6] == 1.0          # < ½ of the 1 m top level: land


@pytest.mark.skipif(not HAVE_2LAYER, reason="BayIW MITgcm inputs not mounted")
def test_load_inputs_uses_the_mitgcm_effective_depth():
    H = bayiw.load_inputs("2layer", "bay")[2]
    wet = np.isfinite(H)
    assert np.abs(H[wet][:, None] + bayiw.Z_F[None, :]).min(axis=1).max() < 1e-9
    assert abs(H[345, 280] - 163.3) < 1e-9
