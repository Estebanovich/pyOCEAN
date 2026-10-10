"""
Tests for diagnostics.sealevel_qc — quality control of raw (unverified)
sea-level records, e.g. the 1-min IOC SLSMF / SMN-UNAM feeds.

Raw gauge data carry instrumental defects that must be told apart from real
ocean signals (storm surge, seiches, tsunamis):

    defect                      real signal it must not be confused with
    --------------------------  ----------------------------------------
    spike (1–3 samples)         tsunami / harbour seiche (coherent, minutes)
    stuck sensor (flat line)    slack water quantised at 1 cm resolution
    datum shift (sharp step)    storm surge (rises over hours)
    clock offset / drift        lagoon or harbour lag (constant, physical)
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics.sealevel_qc import (
    clock_changes,
    clock_offsets,
    compare_twins,
    detect_steps,
    flag_flatlines,
    flag_spikes,
    harmonic_analysis,
    regularize,
)
from forcing.tides import CONSTITUENTS, predict

CONS = list(CONSTITUENTS)
rng = np.random.default_rng(11)
HC = np.array([0.30 * np.exp(-1j * 1.0), 0.08 * np.exp(-1j * 1.5), 0.15 * np.exp(-1j * 2.0),
               0.12 * np.exp(-1j * 0.5), 0.06 * np.exp(-1j * 0.8), 0.05 * np.exp(-1j * 2.1),
               0.022 * np.exp(-1j * 1.6), 0.025 * np.exp(-1j * 0.3)])
T0 = np.datetime64("2026-07-01T00:00", "ns")
MIN = np.timedelta64(60, "s")


def _times(days: float, step=MIN):
    n = int(days * 86400 / (step / np.timedelta64(1, "s")))
    return T0 + np.arange(n) * step


# --------------------------------------------------------------------------
class TestHarmonicAnalysis:

    def test_recovers_all_constituents_from_long_record(self):
        t = _times(400, np.timedelta64(1, "h"))
        h = 1.2 + predict(t, HC, CONS)
        out = harmonic_analysis(t, h, CONS)
        np.testing.assert_allclose(out["hc"], HC, atol=1e-10)
        assert out["mean"] == pytest.approx(1.2, abs=1e-10)
        assert np.max(np.abs(out["residual"])) < 1e-10

    def test_gaps_are_ignored(self):
        t = _times(400, np.timedelta64(1, "h"))
        h = predict(t, HC, CONS) + 0.01 * rng.normal(size=t.size)
        h[rng.random(t.size) < 0.3] = np.nan
        h[2000:3000] = np.nan
        out = harmonic_analysis(t, h, CONS)
        np.testing.assert_allclose(out["hc"], HC, atol=2e-3)
        assert np.isnan(out["residual"][2500])

    def test_short_record_unresolved_pairs_raise(self):
        # 30 days cannot separate K1–P1 or S2–K2 (Rayleigh period ≈ 182.6 d)
        t = _times(30, np.timedelta64(10, "m"))
        with pytest.raises(ValueError, match="p1|k2"):
            harmonic_analysis(t, predict(t, HC, CONS), CONS)

    def test_inference_from_reference_ratios(self):
        # P1 and K2 inferred from K1 and S2 with complex ratios (e.g. from TPXO)
        t = _times(30, np.timedelta64(10, "m"))
        r_p1, r_k2 = HC[5] / HC[2], HC[6] / HC[1]
        out = harmonic_analysis(t, predict(t, HC, CONS), CONS,
                                inference={"p1": ("k1", r_p1), "k2": ("s2", r_k2)})
        np.testing.assert_allclose(out["hc"], HC, atol=1e-10)
        assert out["inferred"] == ["p1", "k2"]


# --------------------------------------------------------------------------
class TestRegularize:

    def test_sorts_and_bins_to_regular_grid(self):
        t = T0 + np.array([0, 120, 60, 185, 240], "timedelta64[s]")
        h = np.array([1.0, 1.2, 1.1, 1.3, 1.4])
        tr, hr, info = regularize(t, h, MIN)
        assert tr[0] == T0 and np.all(np.diff(tr) == MIN)
        np.testing.assert_allclose(hr, [1.0, 1.1, 1.2, 1.3, 1.4])
        assert info["n_unsorted"] == 1

    def test_grid_anchored_to_multiples_of_step(self):
        # Two sensors starting at 00:07 and 00:00 must share grid times
        step = np.timedelta64(10, "m")
        t = T0 + np.timedelta64(7, "m") + np.arange(5) * step
        tr, _, _ = regularize(t, np.arange(5.0), step)
        assert np.all((tr - T0) % step == np.timedelta64(0, "ns"))

    def test_equal_duplicates_merge_conflicting_ones_become_nan(self):
        # IOC feeds repeat samples at hh:mm:00 and hh:mm:45 (same bin)
        t = T0 + np.array([0, 60, 60, 105, 120, 120], "timedelta64[s]")
        h = np.array([1.0, 1.1, 1.1, 1.1, 1.2, 1.9])
        tr, hr, info = regularize(t, h, MIN, tol=0.02)
        np.testing.assert_allclose(hr[:2], [1.0, 1.1])
        assert np.isnan(hr[2])
        assert info["n_duplicate"] == 3 and info["n_conflict"] == 1

    def test_gaps_reported(self):
        t = np.concatenate([_times(0.1), _times(0.1) + np.timedelta64(5, "h")])
        tr, hr, info = regularize(t, np.ones(t.size), MIN, min_gap=np.timedelta64(30, "m"))
        assert len(info["gaps"]) == 1
        start, length = info["gaps"][0]
        assert start == t[t.size // 2 - 1] + MIN
        assert length == np.timedelta64(5, "h") - np.timedelta64(144, "m")
        assert np.isnan(hr).sum() == length // MIN


# --------------------------------------------------------------------------
class TestSpikes:

    def _signal(self):
        t = _times(5)
        x = (t - T0) / np.timedelta64(1, "h")
        surge = 1.0 * np.exp(-(((x - 60.0) / 6.0) ** 2))            # 1 m storm surge, hours wide
        return t, predict(t, HC, CONS) + surge + 0.005 * rng.normal(size=t.size)

    def test_isolated_spikes_flagged_surge_kept(self):
        t, h = self._signal()
        idx = np.array([500, 1501, 1502, 3000, 6000])
        h[idx] += np.array([0.5, -0.4, -0.4, 2.0, 0.3])
        bad = flag_spikes(h)
        assert set(np.flatnonzero(bad)) == set(idx)

    def test_tsunami_like_oscillation_not_flagged(self):
        t, h = self._signal()
        tt = np.arange(180)                                           # 3 h, 15-min period
        h[3000:3180] += 0.3 * np.sin(2 * np.pi * tt / 15.0) * np.exp(-tt / 90.0)
        assert not flag_spikes(h).any()

    def test_quantised_smooth_data_not_flagged(self):
        t = _times(3)
        h = np.round(predict(t, HC, CONS), 2)                         # 1-cm resolution, MAD ≈ 0
        assert not flag_spikes(h).any()


# --------------------------------------------------------------------------
class TestFlatlines:

    def test_stuck_sensor_flagged_slack_water_not(self):
        t = _times(3)
        tide = predict(t, HC, CONS)
        h = np.round(tide, 2)
        h[1000:1360] = h[1000]                                        # sensor stuck 6 h
        bad = flag_flatlines(h, resolution=0.01, expected=tide)
        assert bad[1000:1360].all()
        assert 360 <= bad.sum() <= 365                              # quantised neighbours may join

    def test_without_reference_only_long_runs(self):
        h = np.round(predict(_times(3), HC, CONS), 2)
        h[2000:2800] = 0.123
        bad = flag_flatlines(h, resolution=0.01, max_samples=600)
        assert bad[2000:2800].all() and bad.sum() == 800

    def test_fine_resolution_needs_absolute_change(self):
        # 1-mm gauge: a 10-min plateau while the tide moves ~1 cm is within the
        # reference-tide error; only changes above min_abs (2 cm) count
        t = _times(3)
        tide = predict(t, HC, CONS)
        h = np.round(tide, 3)
        i = int(np.argmax(np.abs(np.gradient(tide)))) // 2 + 100
        h[i:i + 10] = h[i]
        assert np.ptp(tide[i:i + 10]) > 0.005
        assert not flag_flatlines(h, resolution=0.001, expected=tide)[i:i + 10].any()

    def test_nan_breaks_runs(self):
        h = np.full(100, 0.5)
        h[50] = np.nan
        assert not flag_flatlines(h, resolution=0.01, max_samples=60).any()


# --------------------------------------------------------------------------
class TestSteps:

    def _resid(self, days=6):
        t = _times(days)
        x = (t - T0) / np.timedelta64(1, "h")
        return t, 0.6 * np.exp(-(((x - 40.0) / 8.0) ** 2)) + 0.01 * rng.normal(size=t.size)

    def test_datum_shift_found_surge_ignored(self):
        t, r = self._resid()
        r[5000:] += 0.10
        steps = detect_steps(r)
        assert len(steps) == 1
        assert abs(steps[0]["index"] - 5000) <= 2
        assert steps[0]["size"] == pytest.approx(0.10, abs=0.01)

    def test_shift_across_gap_reported_with_gap_length(self):
        t, r = self._resid()
        r[6000:] -= 0.15
        r[4000:6000] = np.nan                                        # maintenance visit
        steps = detect_steps(r)
        assert len(steps) == 1 and steps[0]["index"] == 6000
        assert steps[0]["gap_samples"] == 2000
        assert steps[0]["size"] == pytest.approx(-0.15, abs=0.01)

    def test_residual_tide_and_seiche_no_steps(self):
        # Overtides / well filtering leave tidal residuals, harbours seiche:
        # neither is a datum shift
        t = _times(6)
        sec = (t - T0) / np.timedelta64(1, "s")
        r = (0.05 * np.sin(2 * np.pi * sec / 44714.0) + 0.04 * np.sin(2 * np.pi * sec / 1200.0)
             + 0.01 * rng.normal(size=t.size))
        assert detect_steps(r) == []

    def test_surge_with_strong_harbour_seiche_no_steps(self):
        # Seiche makes D_S swing; during a surge D_S can match D_L by chance
        t, r = self._resid()
        sec = (t - T0) / np.timedelta64(1, "s")
        r += 0.10 * np.sin(2 * np.pi * sec / 1200.0)
        assert detect_steps(r) == []

    def test_subtidal_variability_with_seiche_no_steps(self):
        # Coastal-trapped waves / wind setup (days) + seiche (minutes)
        t = _times(12)
        sec = (t - T0) / np.timedelta64(1, "s")
        r = (0.15 * np.sin(2 * np.pi * sec / (3 * 86400.0)) + 0.08 * np.sin(2 * np.pi * sec / 900.0)
             + 0.01 * rng.normal(size=t.size))
        assert detect_steps(r) == []

    def test_shelf_oscillations_with_sharp_fronts_no_steps(self):
        # Progreso-like: 1.5-h oscillations with abrupt fronts on a moving mean
        t = _times(12)
        sec = (t - T0) / np.timedelta64(1, "s")
        saw = 0.12 * (2 * ((sec / 5400.0) % 1.0) - 1.0)
        r = saw + 0.2 * np.sin(2 * np.pi * sec / (5 * 86400.0)) + 0.01 * rng.normal(size=t.size)
        assert detect_steps(r) == []

    def test_noise_only_no_steps(self):
        assert detect_steps(self._resid()[1]) == []


# --------------------------------------------------------------------------
class TestClock:

    def _ref(self, tt):
        return predict(tt, HC, CONS)

    def test_time_zone_change_mid_record(self):
        t = _times(12)
        tau = np.where(t >= T0 + np.timedelta64(6, "D"), 6 * 3600, 0).astype("timedelta64[s]")
        h = 0.8 + self._ref(t - tau) + 0.005 * rng.normal(size=t.size)
        out = clock_offsets(t, h, self._ref, window=np.timedelta64(2, "D"))
        np.testing.assert_allclose(out["tau_minutes"], [0, 0, 0, 360, 360, 360], atol=1)
        np.testing.assert_allclose(out["gain"], 1.0, atol=0.02)

    def test_clock_drift(self):
        t = _times(12)
        drift = ((t - T0) / np.timedelta64(1, "D") * 2 * 60).astype("timedelta64[s]")   # 2 min/day
        out = clock_offsets(t, self._ref(t - drift), self._ref, window=np.timedelta64(1, "D"))
        days = (out["t_center"] - T0) / np.timedelta64(1, "D")
        np.testing.assert_allclose(out["tau_minutes"], 2 * days, atol=1.0)

    def test_skill_separates_tide_from_noise(self):
        t = _times(6)
        good = clock_offsets(t, self._ref(t) + 0.01 * rng.normal(size=t.size), self._ref,
                             window=np.timedelta64(3, "D"))
        bad = clock_offsets(t, rng.normal(size=t.size), self._ref, window=np.timedelta64(3, "D"))
        assert np.all(good["skill"] > 0.99) and np.all(bad["skill"] < 0.05)

    def test_inverted_sensor_negative_gain(self):
        t = _times(4)
        out = clock_offsets(t, -self._ref(t), self._ref, window=np.timedelta64(2, "D"))
        np.testing.assert_allclose(out["gain"], -1.0, atol=1e-6)
        np.testing.assert_allclose(out["tau_minutes"], 0.0, atol=0.01)

    def test_window_with_too_few_samples_is_nan(self):
        t = _times(4)
        h = self._ref(t)
        h[: int(0.75 * t.size // 2)] = np.nan                       # first window 25 % valid
        out = clock_offsets(t, h, self._ref, window=np.timedelta64(2, "D"))
        assert np.isnan(out["tau_minutes"][0]) and out["tau_minutes"][1] == pytest.approx(0.0, abs=0.01)


    def test_positive_gain_removes_half_period_ambiguity(self):
        # Semidiurnal tide: p(t − 6.21 h) = −p(t); with κ > 0 only τ = 0 fits
        hc = np.zeros(len(CONS), complex)
        hc[0] = 0.07
        ref = lambda tt: predict(tt, hc, CONS)                          # noqa: E731
        t = _times(3)
        kw = dict(window=np.timedelta64(3, "D"), max_lag=np.timedelta64(13, "h"), positive_gain=True)
        out = clock_offsets(t, ref(t) + 0.002 * rng.normal(size=t.size), ref, **kw)
        assert abs(out["tau_minutes"][0]) < 2 and out["gain"][0] > 0
        flipped = clock_offsets(t, -ref(t), ref, **kw)                  # only a half-period shift fits κ > 0
        assert flipped["gain"][0] > 0 and abs(abs(flipped["tau_minutes"][0]) - 372.6) < 3


class TestClockChanges:

    T = T0 + np.arange(30) * np.timedelta64(3, "D")

    def test_scatter_alone_is_not_a_change(self):
        out = clock_changes(self.T, 12.0 * rng.normal(size=30))
        assert not out["jump"] and not out["drift"]

    def test_persistent_jump_found(self):
        tau = 12.0 * rng.normal(size=30)
        tau[18:] += 60.0                                               # e.g. DST hour after a reset
        out = clock_changes(self.T, tau)
        assert out["jump"] and out["jump_minutes"] == pytest.approx(60, abs=15)
        assert out["jump_time"] in (self.T[17], self.T[18])
        assert not out["drift"]

    def test_drift_found(self):
        days = np.arange(30) * 3.0
        out = clock_changes(self.T, 0.5 * days + 3.0 * rng.normal(size=30))
        assert out["drift"] and out["drift_min_per_day"] == pytest.approx(0.5, abs=0.1)
        assert not out["jump"]

    def test_nan_windows_ignored(self):
        tau = 12.0 * rng.normal(size=30)
        tau[::4] = np.nan
        assert not clock_changes(self.T, tau)["jump"]


# --------------------------------------------------------------------------
class TestTwins:

    def test_offset_gain_lag_and_steps_between_colocated_sensors(self):
        # Radar vs float at one SMN site: tide and surge cancel in the difference
        t = _times(10)
        x = (t - T0) / np.timedelta64(1, "h")
        sea = predict(t, HC, CONS) + 0.5 * np.exp(-(((x - 100) / 10) ** 2))
        ext = np.concatenate([np.full(3, sea[0]), sea])
        radar = sea + 0.002 * rng.normal(size=t.size)
        flt = 0.97 * ext[:-3] + 0.25 + 0.002 * rng.normal(size=t.size)     # float lags 3 min
        flt[9000:] += 0.08                                                    # datum shift on float
        out = compare_twins(radar, flt, step=MIN)
        assert out["lag_minutes"] == pytest.approx(3.0, abs=0.5)
        assert out["gain"] == pytest.approx(0.97, abs=0.01)
        assert out["offset"] == pytest.approx(-0.25, abs=0.02)
        assert len(out["steps"]) == 1 and abs(out["steps"][0]["index"] - 9000) <= 2
        assert out["steps"][0]["size"] == pytest.approx(-0.08, abs=0.01)

    def test_gain_not_diluted_by_seiche_on_reference_sensor(self):
        # Radar sees the harbour seiche, the float well filters it: regressing
        # on the noisy sensor must not bias κ (errors-in-variables)
        t = _times(10)
        sec = (t - T0) / np.timedelta64(1, "s")
        sea = predict(t, 0.27 * HC, CONS)                               # M2 ≈ 8 cm (Zihuatanejo)
        radar = sea + 0.06 * np.sin(2 * np.pi * sec / 900.0) + 0.02 * rng.normal(size=t.size)
        out = compare_twins(radar, sea, step=MIN)
        assert out["gain"] == pytest.approx(1.0, abs=0.02)

    def test_scattered_gaps_tolerated(self):
        t = _times(10)
        sea = predict(t, HC, CONS)
        h2 = 0.98 * sea + 0.1
        h2[rng.random(t.size) < 0.2] = np.nan                             # 20 % random dropouts
        out = compare_twins(sea, h2, step=MIN)
        assert out["gain"] == pytest.approx(0.98, abs=0.01) and abs(out["lag_minutes"]) < 0.5

    def test_sub_sample_lag(self):
        # Lag between grid points (parabolic refinement of the misfit minimum)
        t = _times(5)
        out = compare_twins(predict(t, HC, CONS), predict(t - np.timedelta64(150, "s"), HC, CONS), step=MIN)
        assert out["lag_minutes"] == pytest.approx(2.5, abs=0.1)

    def test_drift_between_sensors(self):
        t = _times(20)
        sea = predict(t, HC, CONS)
        days = (t - T0) / np.timedelta64(1, "D")
        out = compare_twins(sea, sea - 0.002 * days, step=MIN)
        assert out["drift_m_per_day"] == pytest.approx(0.002, abs=2e-4)
