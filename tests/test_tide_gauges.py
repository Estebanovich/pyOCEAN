"""
Tests for diagnostics.tide_gauges — comparison of harmonic constants with tide
gauge databases (TICON-3) and detection of instrumental defects.

Instrumental defects leave systematic signatures across constituents that a
model error or a local harbour effect does not:

    time-zone offset τ (whole hours) :  h_obs = h e^{−iω_c τ}   (phase shift ∝ frequency)
    unit error κ = 10ⁿ               :  h_obs = κ h
    inverted sensor                  :  κ ≈ −1
whereas lagoon/harbour filtering gives κ < 1 with a lag of minutes to an hour
(physical, flagged ``lag_damping``), and other residuals are ``misfit``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from diagnostics.tide_gauges import (
    complex_from_amp_phase,
    diagnose_station,
    load_ticon,
    vector_difference,
)
from forcing.tides import CONSTITUENTS, OMEGA

rng = np.random.default_rng(7)
MODEL = rng.normal(0, 0.2, 8) + 1j * rng.normal(0, 0.2, 8)
MODEL[0] = 0.5 * np.exp(-1j * 1.0)                         # dominant M2


def _obs(tau_s=0.0, kappa=1.0, noise=0.0):
    w = np.array([OMEGA[c] for c in CONSTITUENTS])
    n = noise * (rng.normal(size=8) + 1j * rng.normal(size=8))
    return kappa * MODEL * np.exp(-1j * w * tau_s) + n


class TestConversions:

    def test_amp_phase_convention(self):
        # TPXO / TICON: h = A e^{−iG}, G = Greenwich phase lag
        h = complex_from_amp_phase(np.array([2.0]), np.array([90.0]))
        assert h[0] == pytest.approx(-2.0j)

    def test_vector_difference(self):
        d = vector_difference(np.array([1.0 + 0j]), np.array([1.0 * np.exp(-1j * np.deg2rad(10))]))
        assert d[0] == pytest.approx(2 * np.sin(np.deg2rad(5)), rel=1e-12)


class TestDiagnosis:

    def test_clean_station_not_flagged(self):
        d = diagnose_station(_obs(noise=0.003), MODEL, list(CONSTITUENTS))
        assert d["flags"] == []
        assert abs(d["tau_minutes"]) < 10 and d["gain"] == pytest.approx(1.0, abs=0.05)

    @pytest.mark.parametrize("hours", [-7.0, -6.0, 1.0, 5.0])
    def test_time_zone_offset_detected(self, hours):
        # Whole-hour offsets: local time (Mexico UTC−6/−7) or DST logged as UTC
        d = diagnose_station(_obs(tau_s=hours * 3600, noise=0.002), MODEL, list(CONSTITUENTS))
        assert "timezone" in d["flags"]
        assert d["tau_minutes"] == pytest.approx(hours * 60, abs=5)

    @pytest.mark.parametrize("kappa", [0.1, 10.0, 0.01])
    def test_unit_error_detected(self, kappa):
        d = diagnose_station(_obs(kappa=kappa), MODEL, list(CONSTITUENTS))
        assert "units" in d["flags"] and d["gain"] == pytest.approx(kappa, rel=0.02)

    def test_lagoon_damping_and_lag_is_physical_not_instrumental(self):
        # Coastal lagoon / harbour filtering: amplitude halved and ~40 min lag
        d = diagnose_station(_obs(tau_s=40 * 60, kappa=0.5), MODEL, list(CONSTITUENTS))
        assert "lag_damping" in d["flags"]
        assert not {"timezone", "units", "inverted"} & set(d["flags"])

    def test_inverted_sensor_detected(self):
        d = diagnose_station(_obs(kappa=-1.0), MODEL, list(CONSTITUENTS))
        assert "inverted" in d["flags"]

    def test_local_misfit_is_not_called_instrumental(self):
        # Random, frequency-independent misfit (harbour resonance, model error)
        bad = MODEL + 0.15 * (rng.normal(size=8) + 1j * rng.normal(size=8))
        d = diagnose_station(bad, MODEL, list(CONSTITUENTS))
        assert not {"timezone", "units", "inverted", "lag_damping"} & set(d["flags"])
        assert "misfit" in d["flags"]

    def test_corrected_rss_reported(self):
        d = diagnose_station(_obs(tau_s=-6 * 3600), MODEL, list(CONSTITUENTS))
        assert d["rss_corrected"] < 0.05 * d["rss"]


class TestLoadTicon:

    def test_reads_region_and_converts(self, tmp_path: Path):
        lines = []
        for lat, lon, typ in [(23.2, 253.58, "Coastal"), (50.0, 10.0, "Coastal"), (23.3, 253.6, "River")]:
            for c, a, g in [("M2", 34.5, 99.6), ("K1", 20.9, 176.2)]:
                lines.append(f"{lat}\t{lon}\t{c}\t{a}\t{g}\t0.1\t0.2\t2.4\t192311\t5.0\t02/01/1953\t28/06/1975\tgesla.x\t{typ}")
        p = tmp_path / "TICON_3.txt"
        p.write_text("\n".join(lines) + "\n")
        st = load_ticon(p, lon_range=(-110, -100), lat_range=(20, 26), constituents=["m2", "k1"])
        assert len(st) == 1                                        # river and far station excluded
        row = st.iloc[0]
        assert row["lon"] == pytest.approx(253.58 - 360.0)
        assert row["hc"][0] == pytest.approx(0.345 * np.exp(-1j * np.deg2rad(99.6)))
        assert row["start"].year == 1953

    def test_include_river_gauges_on_request(self, tmp_path: Path):
        p = tmp_path / "TICON_3.txt"
        p.write_text("23.3\t253.6\tM2\t30\t100\t0\t0\t0\t1000\t1\t01/01/2000\t01/01/2002\tg\tRiver\n")
        assert len(load_ticon(p, (-110, -100), (20, 26), ["m2"], types=("Coastal", "River"))) == 1


class TestSiteConsistency:

    def test_record_disagreeing_with_its_siblings_is_isolated(self):
        from diagnostics.tide_gauges import compare_with_siblings
        good = [_obs(noise=0.002) for _ in range(3)]
        bad = _obs(kappa=0.1)                                  # e.g. mm stored as cm
        out = compare_with_siblings([*good, bad], list(CONSTITUENTS))
        assert "units" in out[3]["flags"]
        assert all(o["flags"] == [] for o in out[:3])

    def test_single_record_has_no_sibling_check(self):
        from diagnostics.tide_gauges import compare_with_siblings
        assert compare_with_siblings([_obs()], list(CONSTITUENTS)) == [None]


def test_duplicated_constituent_rows_tolerated(tmp_path: Path):
    p = tmp_path / "TICON_3.txt"
    row = "23.2\t253.6\tM2\t34.5\t99.6\t0.1\t0.2\t2.4\t1000\t5\t02/01/1953\t28/06/1975\tg\tCoastal\n"
    p.write_text(row + row)
    st = load_ticon(p, (-110, -100), (20, 26), ["m2"])
    assert len(st) == 1 and st.iloc[0]["hc"][0] == pytest.approx(0.345 * np.exp(-1j * np.deg2rad(99.6)))
