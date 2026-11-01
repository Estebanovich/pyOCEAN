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


# ---------------------------------------------------------------------------
# Long-period MF and MM (regional chain Task 5)
# ---------------------------------------------------------------------------

REF_MM_MF = Path(__file__).parent / "data" / "tide_mm_mf_reference.npz"


class TestLongPeriod:
    """
    MF and MM against OTPS: constit.h ω and astronomical argument, which equal the Doodson arguments 2s and
    s − p of Cartwright's mean longitudes; f, u of the OTPS ``nodal`` long-period branch.  Reference arrays
    from validation/tides/make_mm_mf_reference.py (pyTMD is not installed here).
    """

    def test_mm_mf_frequencies_and_nodal(self):
        from forcing.tides import ALL_CONSTITUENTS, OMEGA, PHASE_1992
        r = np.load(REF_MM_MF)
        names = list(r["constituents"])
        assert set(names) <= set(ALL_CONSTITUENTS)
        om = np.array([OMEGA[c] for c in names])
        ph = np.array([PHASE_1992[c] for c in names])
        np.testing.assert_array_equal(om, r["omega_otps"])
        np.testing.assert_array_equal(ph, r["phase_otps"])
        np.testing.assert_allclose(om, r["omega_doodson"], rtol=1e-5)        # constit.h rounds to 5 digits
        np.testing.assert_allclose(ph, r["phase_doodson"], rtol=0, atol=1e-8)
        f, u = nodal_corrections(r["mjd"], names)
        np.testing.assert_allclose(f, r["f"], rtol=1e-14, atol=0)
        np.testing.assert_allclose(u, r["u"], rtol=0, atol=1e-14)
        # MOM6's first-order factors (class c): same f, u within the 2N, 3N terms (≤ 3.1°)
        np.testing.assert_allclose(f, r["f_mom6"], rtol=0, atol=1e-12)
        assert np.max(np.abs(u - r["u_mom6"])) < np.deg2rad(3.1)

    def test_periods(self):
        from forcing.tides import OMEGA
        days = {c: 2 * np.pi / OMEGA[c] / 86400 for c in ("mf", "mm")}
        assert days["mf"] == pytest.approx(13.661, abs=2e-3)
        assert days["mm"] == pytest.approx(27.555, abs=2e-3)

    def test_existing_eight_bit_for_bit(self, ref):
        """
        Adding MF and MM changes nothing of the eight HYCOM constituents: f, u, ω, astronomical arguments and a
        30-day prediction equal, bit for bit, those of forcing/tides.py before Task 5 (commit 71e848c,
        tests/data/task5_base_reference.npz from validation/tides/make_task5_base_reference.py).
        """
        from forcing.tides import OMEGA, PHASE_1992
        base = np.load(Path(__file__).parent / "data" / "task5_base_reference.npz")
        names = list(base["eight_names"])
        assert CONSTITUENTS == tuple(names)
        f, u = nodal_corrections(ref["mjd"], names)
        np.testing.assert_array_equal(f, base["eight_f"])
        np.testing.assert_array_equal(u, base["eight_u"])
        np.testing.assert_array_equal([OMEGA[c] for c in names], base["eight_omega"])
        np.testing.assert_array_equal([PHASE_1992[c] for c in names], base["eight_phase"])
        t = np.datetime64("1992-01-01T00:00", "ns") + (ref["t_days_1992"] * 86400e9).astype("timedelta64[ns]")
        eta = np.array([predict(t, hc, list(ref["constituents"])) for hc in ref["hc"]])
        np.testing.assert_array_equal(eta, base["eight_eta"])
        # ten constituents = the eight plus MF, MM (the matrix product order changes the rounding only)
        hc2 = np.array([0.01 + 0.02j, -0.03 + 0.01j])
        ten = predict(t, np.concatenate([ref["hc"][0], hc2]), names + ["mf", "mm"])
        np.testing.assert_allclose(ten, predict(t, ref["hc"][0], names) + predict(t, hc2, ["mf", "mm"]),
                                   rtol=0, atol=1e-15)

    def test_nodal_mjd_fixes_corrections(self):
        """predict(nodal_mjd=…) uses f, u of that date at every time (MOM6 TIDE_NODAL_REF_DATE)."""
        from forcing.tides import predict_mjd
        mjd = 61345.0 + np.arange(0, 400, 50.0)
        hc = np.array([0.1 - 0.05j, 0.02 + 0.01j])
        names = ["m2", "mf"]
        fixed = predict_mjd(mjd, hc, names, nodal_mjd=49170.0)
        f, u = nodal_corrections(np.array([49170.0]), names)
        from forcing.tides import MJD_1992, OMEGA, PHASE_1992
        th = (np.array([OMEGA[c] for c in names])[None] * ((mjd - MJD_1992) * 86400.0)[:, None]
              + np.array([PHASE_1992[c] for c in names])[None] + u)
        np.testing.assert_allclose(fixed, (f * np.cos(th)) @ hc.real - (f * np.sin(th)) @ hc.imag, atol=1e-15)
        np.testing.assert_array_equal(predict_mjd(mjd, hc, names),
                                      predict(np.datetime64("1858-11-17", "ns")
                                              + (mjd * 86400e9).astype("timedelta64[ns]"), hc, names))

    def test_long_period_equilibrium_tide(self):
        from forcing.tides import EQ_AMPLITUDE, LOVE_ALPHA, equilibrium_constants
        lat = np.array([0.0, 35.26438968, 90.0])                         # ½ − 3/2 sin²φ = ½, 0, −1
        hc = equilibrium_constants(np.array([0.0, 120.0, -60.0]), lat, ["mf", "mm"])
        assert LOVE_ALPHA["mf"] == pytest.approx(0.693)
        np.testing.assert_allclose(hc[:, 0], 0.693 * EQ_AMPLITUDE["mf"] * np.array([0.5, 0.0, -1.0]), atol=1e-10)
        np.testing.assert_allclose(hc.imag, 0.0, atol=0)
