"""
Tests for forcing.bulk.bulk_ncar_fms — the NCAR bulk formulae in the form of the FMS
coupler that MOM6 uses (FMScoupler 2025.02, shared/surface_flux.F90).

Reference: tests/data/bulk_fms_ncar_reference.npz, written by
validation/bulk_points/build_reference.sh from the compiled, unchanged surface_flux.F90 on
validation/bulk_points/points.py::POINTS (250 points: stable and unstable, wind 0.1–30 m/s,
SST −1.8–32 °C, with and without a surface current), for z_u = z_t = z_q = 10 m
(ncar_ocean_fluxes) and z_t = z_q = 2 m (ncar_ocean_fluxes_multilevel).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forcing.bulk import _SCHEMES, bulk_ncar, bulk_ncar_fms
from validation.bulk_points import points as bp

REF = Path(__file__).parent / "data" / "bulk_fms_ncar_reference.npz"
HLV = 2.5e6          # J kg⁻¹, FMS gfdl_constants.fh HLV
RHO_FW = 1000.0      # kg m⁻³, evap in m s⁻¹ (forcing.bulk.RHO_FW)
RTOL = 1e-10


def _fms_call(tag: str):
    zu, zt, zq = bp.HEIGHTS[tag]
    P = bp.POINTS
    return bulk_ncar_fms(P["u10"], P["v10"], P["t_air"], P["q_air"], P["sst"], P["p"],
                         u_ocean=P["u_ocean"], v_ocean=P["v_ocean"], z_u=zu, z_t=zt, z_q=zq)


def _close(a: np.ndarray, b: np.ndarray) -> float:
    """Largest |a − b| relative to max(|b|, tiny) point by point."""
    return float(np.max(np.abs(a - b) / np.maximum(np.abs(b), 1e-300)))


@pytest.mark.parametrize("tag", ["z10", "z2"])
def test_matches_fms_reference(tag: str) -> None:
    ref = np.load(REF)
    for k in bp.INPUT_FIELDS:                       # POINTS unchanged since the table was made
        np.testing.assert_array_equal(ref[f"in_{k}"], bp.POINTS[k])
    f = _fms_call(tag)
    # FMS signs: stress on the atmosphere and upward fluxes (surface_flux.F90:441-476)
    expected = {
        "tau_x": -ref[f"{tag}_flux_u"],
        "tau_y": -ref[f"{tag}_flux_v"],
        "q_sens": -ref[f"{tag}_flux_t"],
        "q_lat": -HLV * ref[f"{tag}_flux_q"],
        "evap": ref[f"{tag}_flux_q"] / RHO_FW,
        "cd": ref[f"{tag}_cd_m"],
        "ch": ref[f"{tag}_cd_t"],
        "ce": ref[f"{tag}_cd_q"],
        "ustar": ref[f"{tag}_u_star"],
    }
    for name, want in expected.items():
        got = getattr(f, name)
        err = _close(got, want)
        assert err < RTOL, f"{tag} {name}: max relative error {err:.2e}"


def test_calm_and_extreme_stability_finite() -> None:
    """Zero wind and ΔT = ±15 K: the 0.5 m/s floor keeps u* > 0; everything finite."""
    sst = np.full(4, 290.0)
    t_air = sst + np.array([15.0, -15.0, 15.0, -15.0])
    q = np.array([0.002, 0.02, 0.0, 0.01])
    u = np.array([0.0, 0.0, 0.0, 1e-3])
    f = bulk_ncar_fms(u, np.zeros(4), t_air, q, sst, np.full(4, 1.0e5))
    for name in ("tau_x", "tau_y", "q_sens", "q_lat", "evap", "cd", "ch", "ce", "ustar",
                 "rho_air", "u10n", "zeta"):
        assert np.all(np.isfinite(getattr(f, name))), name
    # w_atm = 0 → no flux (surface_flux.F90:420-423 multiply by the unfloored speed)
    np.testing.assert_array_equal(f.tau_x[:3], 0.0)
    np.testing.assert_array_equal(f.q_sens[:3], 0.0)
    assert np.all(f.ustar > 0.0)


def test_ly04_differences_documented() -> None:
    """bulk_ncar (LY04 / MITgcm form) vs bulk_ncar_fms on POINTS, everything at 10 m.

    The measured differences (and their causes) are in points.LY04_VS_FMS_MAX_REL and in
    validation/bulk_points/README.md; this test fails if either scheme changes.
    """
    P = bp.POINTS
    fms = _fms_call("z10")
    ly = bulk_ncar(P["u10"], P["v10"], P["t_air"], P["q_air"], P["sst"] - bp.T0_K, P["p"],
                   u_ocean=P["u_ocean"], v_ocean=P["v_ocean"], z_u=10.0, z_t=10.0)
    assert set(bp.LY04_VS_FMS_MAX_REL) == {"tau", "q_sens", "q_lat", "evap"}
    tau_ly, tau_fms = np.hypot(ly.tau_x, ly.tau_y), np.hypot(fms.tau_x, fms.tau_y)
    pairs = {"tau": (tau_ly, tau_fms), "q_sens": (ly.q_sens, fms.q_sens),
             "q_lat": (ly.q_lat, fms.q_lat), "evap": (ly.evap, fms.evap)}
    for name, (a, b) in pairs.items():
        rel = float(np.max(np.abs(a - b)) / np.max(np.abs(b)))
        bound = bp.LY04_VS_FMS_MAX_REL[name]
        assert rel <= bound, f"{name}: {rel:.4g} > documented {bound:.4g}"
        assert rel >= 0.5 * bound, f"{name}: {rel:.4g} far below documented {bound:.4g}; update"


def test_scheme_registry() -> None:
    assert _SCHEMES["ncar"] is bulk_ncar
    assert _SCHEMES["ncar_fms"] is bulk_ncar_fms


def test_compute_surface_fluxes_ncar_fms_takes_celsius() -> None:
    """compute_surface_fluxes passes the model SST [°C] to 'ncar_fms' in K."""
    import xarray as xr

    from forcing.bulk import compute_surface_fluxes

    P = bp.POINTS
    sl = slice(0, 20)
    atm = xr.Dataset({
        "u10": ("x", P["u10"][sl]), "v10": ("x", P["v10"][sl]),
        "t2m": ("x", P["t_air"][sl]), "q2m": ("x", P["q_air"][sl]), "msl": ("x", P["p"][sl]),
        "sw_down": ("x", np.zeros(20)), "lw_down": ("x", np.zeros(20)),
    })
    sst_c = P["sst"][sl] - bp.T0_K
    out = compute_surface_fluxes(atm, sst_c, scheme="ncar_fms", albedo=0.0, emissivity=0.0)
    f = bulk_ncar_fms(P["u10"][sl], P["v10"][sl], P["t_air"][sl], P["q_air"][sl],
                      sst_c + bp.T0_K, P["p"][sl])
    np.testing.assert_array_equal(out["tau_x"], f.tau_x)
    np.testing.assert_array_equal(out["Q_net"], f.q_sens + f.q_lat)
    np.testing.assert_array_equal(out["E_P"], f.evap)
