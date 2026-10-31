"""
Tests for forcing.bulk_coare.bulk_coare_croco — COARE 3.0 as CROCO's bulk_flux.F computes
it (~/pyOCEAN_refs/croco git ab63b58, default options BULK_GUSTINESS and BULK_LW).

Reference: tests/data/bulk_croco_coare_reference.npz, written by
validation/bulk_points/build_reference.sh (``coare``) from the compiled scalar block of
bulk_flux.F (:198-659, :758-974) on validation/bulk_points/points.py::COARE_POINTS (the 250
states of POINTS in CROCO's units, absolute wind, p = 1e5 Pa, plus a downwelling longwave).
The table holds CROCO's kinematic outputs; they are turned into W m⁻² and N m⁻² here with
the same ρ₀ = 1025 kg m⁻³ (croco.in:61) and C_p = 3985 J kg⁻¹ K⁻¹ (scalars.h:640).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forcing import bulk_coare as bc
from forcing.bulk import TurbulentFluxes, _SCHEMES
from forcing.bulk_coare import CoareFluxes, bulk_coare_croco
from validation.bulk_points import points as bp

REF = Path(__file__).parent / "data" / "bulk_croco_coare_reference.npz"
RHO0 = 1025.0        # kg m⁻³, croco.in:61
CP = 3985.0          # J kg⁻¹ K⁻¹, scalars.h:640
RTOL = 1e-10


def _close(a: np.ndarray, b: np.ndarray) -> float:
    """Largest |a − b| relative to max(|b|, tiny) point by point."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


def _call(**kw) -> CoareFluxes:
    P = bp.COARE_POINTS
    return bulk_coare_croco(P["u10"], P["v10"], P["t_air_c"], P["q_air"], P["sst_c"],
                            P["lw_down"], **kw)


def test_matches_croco_reference() -> None:
    ref = np.load(REF)
    for k in bp.COARE_INPUT_FIELDS:                 # inputs unchanged since the table was made
        np.testing.assert_array_equal(ref[f"in_{k}"], bp.COARE_POINTS[k])
    f = _call()
    # CROCO kinematic, positive into the ocean (bulk_flux.F:630-659, :706-712)
    expected = {
        "tau_x": RHO0 * ref["sustr"],
        "tau_y": RHO0 * ref["svstr"],
        "q_sens": RHO0 * CP * ref["hfsen"],
        "q_lat": RHO0 * CP * ref["hflat"],
        "q_lw": RHO0 * CP * ref["hflw"],
        "evap": ref["evap"],
        "cd": ref["cd"],
        "rho_air": ref["rho_air"],
        "ustar": ref["wstar"],
    }
    for name, want in expected.items():
        err = _close(getattr(f, name), want)
        assert err < RTOL, f"{name}: max relative error {err:.2e}"


def test_intermediates_match_croco() -> None:
    """Scales, final gusty wind, q_sat, Charnock and p at 2 m of the compiled CROCO."""
    ref = np.load(REF)
    P = bp.COARE_POINTS
    d = bc._coare_core(np.sqrt(P["u10"] * P["u10"] + P["v10"] * P["v10"]), P["t_air_c"],
                       P["q_air"], P["sst_c"])
    for name in ("wstar", "tstar", "qstar", "delw", "qsea", "charn", "patm"):
        err = _close(d[name], ref[name])
        assert err < RTOL, f"{name}: max relative error {err:.2e}"


def test_croco_quirks() -> None:
    # ψ_t(0) = −((1 + 0)^1.5 + 0.6667 (0 − 14.28) e⁰ + 8.525) = −0.004524, not 0
    # (stable branch includes 0, bulk_flux.F:952, 964-966); ψ_u(0) = 0 (unstable branch, :922-929)
    assert bc._psit_coare(np.array([0.0]))[0] == pytest.approx(-0.004524, abs=1e-14)   # cancellation in 1 − 9.52 + 8.525
    assert abs(bc._psiu_coare(np.array([0.0]))[0]) < 1e-15

    # q_sat at the sea surface with psurf = 1e5 Pa and the 0.98 factor (bulk_flux.F:235,
    # 816-830), CtoK = 273.16 (params_bulk.h:1); 20 °C sea, liquid branch
    t_k = 20.0 + 273.16
    psat = 0.98 * np.exp(60.2227554 - 6822.40088 / t_k - 5.13926744 * np.log(t_k))
    mvoma = 18.0153e-3 / 28.9644e-3
    q_expected = mvoma * psat / (1.0e5 + (mvoma - 1.0) * psat)
    d = bc._coare_core(np.array([5.0]), np.array([18.0]), np.array([0.01]), np.array([20.0]))
    assert d["qsea"][0] == pytest.approx(q_expected, rel=1e-14)
    # p_sfc is CROCO's psurf: q_sat at the sea surface uses it directly, falling as it rises
    d2 = bc._coare_core(np.array([5.0]), np.array([18.0]), np.array([0.01]), np.array([20.0]),
                        p_sfc=1.01e5)
    assert d2["qsea"][0] < d["qsea"][0]

    # Charnock from the initial gusty wind √(w² + 0.5²) only (bulk_flux.F:247, 498-503), not
    # the gust-updated wind of the iterations: strongly unstable 11 m/s
    w = np.array([11.0])
    d = bc._coare_core(w, np.array([5.0]), np.array([0.002]), np.array([25.0]))
    dw0 = np.sqrt(w * w + 0.25)
    np.testing.assert_array_equal(d["charn"], bc._charnock(dw0))
    assert d["delw"][0] != dw0[0]
    assert bc._charnock(d["delw"])[0] != d["charn"][0]
    assert bc._charnock(np.array([9.0, 14.0, 20.0])) == pytest.approx(
        [0.011, 0.011 + 0.125 * 0.007 * 4.0, 0.018], rel=1e-14)


def test_calm_and_extreme_stability_finite() -> None:
    """Zero wind and ΔT = ±15 K: the 1 m/s floor and the gustiness keep u* > 0 (Review Focus 1)."""
    sst = np.full(4, 17.0)
    t_air = sst + np.array([15.0, -15.0, 15.0, -15.0])
    q = np.array([0.002, 0.02, 0.0, 0.001])
    u = np.array([0.0, 0.0, 0.0, 1e-3])
    for gust in (True, False):
        f = bulk_coare_croco(u, np.zeros(4), t_air, q, sst, np.full(4, 300.0), gustiness=gust)
        for name in ("tau_x", "tau_y", "q_sens", "q_lat", "q_lw", "evap", "cd", "rho_air",
                     "ustar", "ch"):
            assert np.all(np.isfinite(getattr(f, name))), (gust, name)
        assert np.all(f.ustar > 0.0)
        # stress uses the component of the absolute wind (bulk_flux.F:708): none here
        np.testing.assert_array_equal(f.tau_x[:3], 0.0)
        assert np.all(f.q_sens[[0, 2]] > 0.0) and np.all(f.q_sens[[1, 3]] < 0.0)


def test_scheme_adapter() -> None:
    """_SCHEMES['coare']: canonical inputs (T in K, wind relative to the current) → TurbulentFluxes."""
    P = bp.COARE_POINTS
    t_k = P["t_air_c"] + 273.15
    tf = _SCHEMES["coare"](P["u10"], P["v10"], t_k, P["q_air"], P["sst_c"], p=np.full(250, 1.02e5))
    assert isinstance(tf, TurbulentFluxes)
    f = bulk_coare_croco(P["u10"], P["v10"], t_k - 273.15, P["q_air"], P["sst_c"], P["lw_down"])
    for name in ("tau_x", "tau_y", "q_sens", "q_lat", "evap", "cd", "rho_air", "ustar", "ch"):
        np.testing.assert_array_equal(getattr(tf, name), getattr(f, name))
    np.testing.assert_array_equal(tf.ce, f.ch)
    # relative wind: the current is subtracted from the wind
    uo = np.full(250, 0.4)
    tr = _SCHEMES["coare"](P["u10"] + uo, P["v10"], t_k, P["q_air"], P["sst_c"],
                           p=np.full(250, 1.02e5), u_ocean=uo)
    np.testing.assert_allclose(tr.tau_x, tf.tau_x, rtol=1e-12, atol=1e-15)
