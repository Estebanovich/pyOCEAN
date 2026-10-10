"""
Tests for forcing.tides — harmonic tide prediction with OTPS/TPXO conventions.

Reference: tests/data/tide_otps_reference.npz, produced with pyTMD 2.1.7
(validation/tides/make_reference.py), which reproduces the OSU Tidal
Prediction Software (OTPS) used with TPXO (Egbert & Erofeeva 2002).
pyOCEAN must match it to round-off so that TPXO constants are used exactly
as their authors intend.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forcing.tides import (
    CONSTITUENTS,
    MJD_1992,
    nodal_corrections,
    predict,
    to_mjd,
)

REF = Path(__file__).parent / "data" / "tide_otps_reference.npz"


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


class TestConstituents:

    def test_hycom_order(self):
        # HYCOM tidcon order and the TPXO subset used by HYCOM-IOA
        assert CONSTITUENTS == ("m2", "s2", "k1", "o1", "n2", "p1", "k2", "q1")

    def test_periods(self):
        from forcing.tides import OMEGA
        hours = {c: 2 * np.pi / OMEGA[c] / 3600 for c in CONSTITUENTS}
        assert hours["m2"] == pytest.approx(12.4206, abs=1e-3)
        assert hours["s2"] == pytest.approx(12.0000, abs=1e-3)
        assert hours["k1"] == pytest.approx(23.9345, abs=1e-3)
        assert hours["o1"] == pytest.approx(25.8193, abs=1e-3)


class TestTime:

    def test_mjd_of_otps_epoch(self):
        assert to_mjd(np.datetime64("1992-01-01T00:00")) == pytest.approx(MJD_1992)

    def test_mjd_of_j2000(self):
        assert to_mjd(np.datetime64("2000-01-01T12:00")) == pytest.approx(51544.5)


class TestNodalCorrections:

    def test_against_otps(self, ref):
        f, u = nodal_corrections(ref["mjd"], list(ref["constituents"]))
        np.testing.assert_allclose(f, ref["f"], rtol=1e-12, atol=1e-14)
        np.testing.assert_allclose(u, ref["u"], rtol=0, atol=1e-12)

    def test_solar_constituents_uncorrected(self):
        f, u = nodal_corrections(np.array([50000.0, 60000.0]), ["s2", "p1"])
        np.testing.assert_allclose(f, 1.0)
        np.testing.assert_allclose(u, 0.0)

    def test_m2_nodal_factor_range(self):
        # 18.61-year modulation of M2: f ≈ 1 ± 0.037
        mjd = np.linspace(50000.0, 50000.0 + 18.61 * 365.25, 2000)
        f, _ = nodal_corrections(mjd, ["m2"])
        assert f.min() == pytest.approx(1 - 0.0373, abs=1e-3)
        assert f.max() == pytest.approx(1 + 0.0373, abs=1e-3)


class TestPredict:

    def test_against_otps_time_series(self, ref):
        t = np.datetime64("1992-01-01T00:00", "ns") + (ref["t_days_1992"] * 86400e9).astype("timedelta64[ns]")
        for hc, eta in zip(ref["hc"], ref["eta"]):
            np.testing.assert_allclose(predict(t, hc, list(ref["constituents"])), eta, atol=1e-11)

    def test_spatial_fields_broadcast(self, ref):
        t = np.array(["2026-10-10T00", "2026-10-10T06"], dtype="datetime64[ns]")
        hc = np.stack([ref["hc"][0], ref["hc"][1]])[:, None, :] * np.ones((1, 3, 1))   # (2, 3, nc)
        out = predict(t, hc, list(ref["constituents"]))
        assert out.shape == (2, 2, 3)                    # (time, *space)
        np.testing.assert_allclose(out[:, 1, 2], predict(t, ref["hc"][1], list(ref["constituents"])))

    def test_pure_m2_amplitude(self):
        # A single constituent with |hc| = A oscillates between ±f·A
        t = np.datetime64("2026-10-01") + np.arange(0, 48 * 3600, 600).astype("timedelta64[s]")
        eta = predict(t, np.array([0.5 * np.exp(-1j * 0.3)]), ["m2"])
        f, _ = nodal_corrections(to_mjd(t), ["m2"])
        assert np.abs(eta).max() == pytest.approx(0.5 * f.mean(), rel=2e-3)

    def test_unknown_constituent(self):
        with pytest.raises(KeyError, match="m3"):
            predict(np.array(["2026-10-10"], dtype="datetime64[ns]"), np.array([1.0 + 0j]), ["m3"])

    def test_shape_mismatch(self):
        with pytest.raises(ValueError, match="constituents"):
            predict(np.array(["2026-10-10"], dtype="datetime64[ns]"), np.ones(3, complex), ["m2", "s2"])
