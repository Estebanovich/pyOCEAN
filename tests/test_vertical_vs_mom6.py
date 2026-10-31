"""
Level-1 validation (spec §8): pyOCEAN column routines against the original MOM6
Fortran (compiled by validation/mom6_column/build_reference.sh).  Skipped when the
reference build is absent.
"""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "validation" / "mom6_column"))
import columns  # noqa: E402

from core.vertical import remap as rm  # noqa: E402

BUILD = Path.home() / "pyOCEAN_refs" / "build" / "column"
needs = lambda exe: pytest.mark.skipif(not (BUILD / exe).exists(), reason=f"MOM6 reference {exe} not built")


@needs("remap_driver")
def test_remap_matches_mom6(tmp_path):
    cases = columns.remap_cases()
    columns.write_remap_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "remap_driver"), str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_remap_results(tmp_path / "out.txt")
    for c, r in zip(cases, ref):
        u1 = np.empty(len(c["h1"]))
        rm.remap_column(c["h0"], c["u0"], c["h1"], c["scheme"], u1)
        np.testing.assert_allclose(u1, r, rtol=1e-11, atol=1e-11,
                                   err_msg=f"{columns.SCHEME_NAMES[c['scheme']]} n0={len(c['h0'])}")


from core.vertical import regrid as rg  # noqa: E402


@needs("regrid_driver")
def test_regrid_matches_mom6(tmp_path):
    cases = columns.regrid_cases()
    columns.write_regrid_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "regrid_driver"), str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_regrid_results(tmp_path / "out.txt")
    for c, r in zip(cases, ref):
        if c["kind"] == "zstar":
            zw = np.empty(len(c["dz"]) + 1)
            rg.zstar_column(c["H"], c["eta"], c["dz"], 1e-3, zw)
            np.testing.assert_allclose(zw[::-1], r, atol=1e-10, err_msg=f"zstar H={c['H']}")
        elif c["kind"] == "sigma":
            zw = np.empty(len(c["dz"]) + 1)
            rg.sigma_column(c["H"], c["eta"], c["dz"][::-1], 1e-3, zw)
            np.testing.assert_allclose(zw[::-1], r, atol=1e-10, err_msg=f"sigma H={c['H']}")
        else:
            x = np.empty(len(c["h"]) + 1)
            rg._p1m_h2_positions(c["h"], c["rho"], c["targets"], x)
            np.testing.assert_allclose(x, r, atol=1e-9, err_msg="P1M_H2 interpolation")


from physics.mixing.kappa_shear import JacksonParams, kappa_shear_column  # noqa: E402


@needs("kappa_shear_driver")
def test_kappa_shear_matches_mom6(tmp_path):
    """
    Jackson et al. (2008) shear mixing: the NumPy oracle against MOM6's private
    kappa_shear_column (MOM_kappa_shear.F90:866-2069, extracted by build_reference.sh) with
    nzc = nz, linear EOS in T only (T = −b, dbuoy_dT = −1, dbuoy_dS = 0), Boussinesq, MOM6
    defaults = JacksonParams(), KAPPA_SHEAR_ITER_BUG = False.

    Tolerance |Δκ| ≤ 1e-10 max κ and |ΔQ| ≤ 1e-10 max Q per column.  The scheme iterates to
    KAPPA_SHEAR_TOL_ERR = 0.1 and has discrete branches (truncation at κ_trunc, sub-step
    halving, Newton switch); a port error or a branch taken differently shows up at
    ≥ 1e-3 of κ.  Observed: bit for bit (since task 3, squares as x*x) with the reference built
    with -ffp-contract=off, ≤ 2.1e-13 relative with FMA contraction on (README).  1e-10 is
    far above round-off and nine orders below the scheme's own tolerance.  The MOM6
    counters also check that the comparison covers the Newton branch and the truncation ramp.
    """
    cases = columns.kappa_shear_cases()
    columns.write_kappa_shear_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "kappa_shear_driver"), str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_kappa_shear_results(tmp_path / "out.txt")
    assert len(ref) == len(cases)
    p = JacksonParams()
    for c, r in zip(cases, ref):
        kappa, tke = kappa_shear_column(c["u"], c["v"], c["b"], c["dz"], c["f2"], c["dt"], c["kappa0"], p)
        np.testing.assert_allclose(kappa, r["kappa"], rtol=0.0, atol=1e-10 * r["kappa"].max() + 1e-300,
                                   err_msg=f"kappa, {c['name']}")
        np.testing.assert_allclose(tke, r["tke"], rtol=0.0, atol=1e-10 * r["tke"].max(),
                                   err_msg=f"TKE, {c['name']}")
        if c["name"] == "subcritical":
            assert np.all(r["kappa"] == 0.0)
        if "(Newton)" in c["name"]:
            assert r["n_newton"] > 0, f"MOM6 did not enter Newton in {c['name']}"
    assert sum(r["n_ramp"] for r in ref) > 0                 # Picard truncation ramp exercised


from physics.mixing.bbl_mixing import BBLParams, lotw_bbl_column  # noqa: E402


@needs("bbl_driver")
def test_lotw_bbl_matches_mom6(tmp_path):
    """
    Drag-driven BBL diffusivity: the NumPy oracle against MOM6 add_LOTW_BBL_diffusivity
    (MOM_set_diffusivity.F90:1606-1789, extracted verbatim by build_reference.sh) with the same
    inputs: visc%ustar_BBL = u*, visc%BBL_meanKE_loss = tke / BBL_EFFIC (so MOM6's
    TKE_column = pyOCEAN's tke, SD:1709), Boussinesq, unit scaling 1, all four corner f equal,
    no tidal u*/dissipation, no Rayleigh drag, LOTW_BBL_ANSWER_DATE 99991231; no thin-layer
    merge (a 1 mm layer goes in as it is).

    Tolerance |ΔKd| ≤ 1e-10 max Kd per column (spec criterion 1).  The kernel is one sweep with
    no iteration; the only discrete branches are the energy cut (SD:1765) and the KD_MAX branch
    (SD:1769), and a port error moves Kd by O(1).  Observed with the driver built with
    -ffp-contract=off: bit for bit in 14 of 15 columns, 1.5e-17 in the other (the ulp of
    tke / BBL_EFFIC × BBL_EFFIC; README).
    """
    cases = columns.bbl_cases()
    columns.write_bbl_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "bbl_driver"), "lotw", str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_bbl_results(tmp_path / "out.txt")
    assert len(ref) == len(cases)
    n_kd_max = 0
    for c, r in zip(cases, ref):
        p = BBLParams(**c["p"])
        kd = lotw_bbl_column(c["dz"], c["n2"], c["ustar"], c["absf"], c["tke"], p)
        assert kd.shape == r["kd"].shape
        assert r["kd"].max() > 0.0, c["name"]
        np.testing.assert_allclose(kd, r["kd"], rtol=0.0, atol=1e-10 * r["kd"].max(),
                                   err_msg=f"Kd_BBL, {c['name']}")
        n_kd_max += int(np.sum(r["kd"] == p.kd_max))
        if c["name"] == "ample energy (uncut)":
            # energy never limits: MOM6 gives the rotation-limited law of the wall itself
            z = np.cumsum(c["dz"])[:-1]
            d = c["dz"].sum()
            kd_wall = p.von_karman * c["ustar"] ** 2 * z * (d - z) / (c["ustar"] * d + c["absf"] * z * (d - z))
            np.testing.assert_allclose(r["kd"][1:-1], kd_wall, rtol=1e-12)
    # the comparison covers the KD_MAX branch (case 3: two interfaces)
    assert n_kd_max == 2


@needs("bbl_driver")
def test_set_bbl_tke_factor(tmp_path):
    """
    The energy and u* that MOM6 hands to the BBL mixing for a uniform flow U in x
    (v = 0), from MOM6 set_BBL_TKE (MOM_set_diffusivity.F90:1947-2152, extracted verbatim and
    compiled; 3 × 3 cells, equal face areas).  Its inputs are set as set_viscous_BBL does:
    per-face u*_f = √C_d √(u² + v_at_u² + U_bg²) (MOM_set_viscosity.F90:687-689, 711; at the v
    faces u_at_v = U, so they carry the same u*_f), kv_bbl = (√C_d u*_f) bbl_thick (SV:1115),
    max(KV_BBL_MIN, ·) (SV:1142) with CORRECT_BBL_BOUNDS = False (MOM6 default).

    Measured (KV_BBL_MIN not active): ustar_BBL = √2 √C_d |U_b| and BBL_meanKE_loss = 2 C_d |U_b| U² (|U_b| =
    √(U² + U_bg²)), because SD:2132 takes ½ Σ over the four faces of u*_f² (all four carry
    u*_f) and SD:2134-2137 sums the two u faces without ½ (the v faces carry no normal
    velocity).  Class (d) of the reference; pyOCEAN uses the physical energy C_d|U|³ and
    u* = √(|τ_b|/ρ₀) (spec decision), so γ = 0.2 in pyOCEAN equals 0.1 of MOM6/OM4.  When
    C_d|U_b| h_bbl < KV_BBL_MIN both grow by KV_BBL_MIN/(C_d|U_b| h_bbl) (last case).
    """
    cases = columns.bbl_tke_cases()
    columns.write_bbl_tke_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "bbl_driver"), "tke", str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_bbl_tke_results(tmp_path / "out.txt")
    assert len(ref) == len(cases)
    for c, r in zip(cases, ref):
        ub = np.hypot(c["u"], c["ubg"])
        # KV_BBL_MIN clip without CORRECT_BBL_BOUNDS: set_BBL_TKE recovers an inflated face u*
        infl = max(1.0, c["kv_min"] / (c["cd"] * ub * c["bbl_thick"]))
        np.testing.assert_allclose(r["ustar_bbl"] / (np.sqrt(c["cd"]) * ub), np.sqrt(2.0) * infl, rtol=1e-12,
                                   err_msg=f"ustar_BBL factor, {c['name']}")
        np.testing.assert_allclose(r["mean_ke_loss"] / (c["cd"] * ub * c["u"] ** 2), 2.0 * infl, rtol=1e-12,
                                   err_msg=f"BBL_meanKE_loss factor, {c['name']}")
    assert any(c["kv_min"] / (c["cd"] * np.hypot(c["u"], c["ubg"]) * c["bbl_thick"]) > 1.0 for c in cases)
