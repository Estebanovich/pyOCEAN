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


@needs("kappa_shear_driver")
def test_kappa_shear_nonlinear_eos_matches_mom6(tmp_path, monkeypatch):
    """
    Jackson et al. (2008) shear mixing with a nonlinear EOS: the NumPy oracle's two-tracer path
    (``kappa_shear_column`` with T, S, eos) against MOM6's kappa_shear_column with tv%T associated
    and tv%eqn_of_state = WRIGHT_FULL or ROQUET_RHO (``kappa_shear_driver eos``): MOM6 itself
    carries T and Sal through calculate_projected_state and builds dbuoy_dT, dbuoy_dS once per
    column (MOM_kappa_shear.F90:1126-1136, 1493-1503).  14 columns × 2 EOS
    (``columns.kappa_shear_eos_cases``), ρ₀ 1035, g 9.81, Boussinesq, JacksonParams() defaults.

    Tolerance as the linear test, |Δκ| ≤ 1e-10 max κ and |ΔQ| ≤ 1e-10 max Q per column (observed:
    bit for bit in every column with the driver built with -ffp-contract=off).  The comparison
    covers MOM6's Newton branch (uniform shear Ri 0.22, 3 iterations), the Picard truncation ramp
    (26 of 28 columns), mixing in every column, and several sub-steps per Δt in at least half of
    them (counted here as calls of ``_find_kappa_tke``: 1 + 2 per sub-step that mixes; measured
    ≥ 2 sub-steps in 22 of 28, up to 13).
    """
    from physics.mixing import kappa_shear as ks
    cases = columns.kappa_shear_eos_cases()
    columns.write_kappa_shear_eos_cases(tmp_path / "in.txt", cases)
    subprocess.run([str(BUILD / "kappa_shear_driver"), "eos", str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_kappa_shear_results(tmp_path / "out.txt")
    assert len(ref) == len(cases)
    calls = [0]
    real = ks._find_kappa_tke

    def counting(*a, **k):
        calls[0] += 1
        return real(*a, **k)
    monkeypatch.setattr(ks, "_find_kappa_tke", counting)
    p = JacksonParams()
    n_multi = 0
    for c, r in zip(cases, ref):
        calls[0] = 0
        kappa, tke = kappa_shear_column(c["u"], c["v"], None, c["dz"], c["f2"], c["dt"], c["kappa0"], p,
                                        T=c["T"], S=c["S"], eos=c["eos"], rho0=c["rho0"], g=c["g"])
        assert r["kappa"].max() > 0.0, c["name"]
        np.testing.assert_allclose(kappa, r["kappa"], rtol=0.0, atol=1e-10 * r["kappa"].max(),
                                   err_msg=f"kappa, {c['name']}")
        np.testing.assert_allclose(tke, r["tke"], rtol=0.0, atol=1e-10 * r["tke"].max(),
                                   err_msg=f"TKE, {c['name']}")
        if "(Newton)" in c["name"]:
            assert r["n_newton"] > 0, f"MOM6 did not enter Newton in {c['name']}"
        n_multi += calls[0] >= 5                              # ≥ 2 sub-steps
    assert sum(r["n_ramp"] for r in ref) > 0
    assert n_multi >= len(cases) // 2


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


def _bbl_n2_mom6(tmp_path):
    cases = columns.bbl_n2_cases()
    columns.write_bbl_n2_cases(tmp_path / "n2in.txt", cases)
    subprocess.run([str(BUILD / "bbl_driver"), "n2", str(tmp_path / "n2in.txt"), str(tmp_path / "n2out.txt")],
                   check=True, capture_output=True)
    ref = columns.read_bbl_n2_results(tmp_path / "n2out.txt")
    assert len(ref) == len(cases)
    return cases, ref


def _bbl_n2_pyocean(c):
    """N² of the BBL path for an unmerged column (every layer its own merged layer), bed first."""
    from physics.mixing.bbl_mixing import _merged_n2, _merged_n2_local, n2_bottom_bulk
    from physics.eos.stratification import eos_code
    n = len(c["dz"])
    lay = np.arange(n)
    if c["eos"] == "linear":
        n2 = _merged_n2(columns.bbl_n2_density(c), c["dz"], lay, c["dz"].copy(), n, c["rho0"], c["g"])
    else:
        n2 = _merged_n2_local(c["T"], c["S"], c["dz"], lay, c["dz"].copy(), n, c["rho0"], c["g"],
                              eos_code(c["eos"]))
    n2_bottom_bulk(n2, c["dz"], n)
    return n2


@needs("bbl_driver")
def test_bbl_bottom_n2_matches_mom6_find_n2(tmp_path):
    """
    N² of the BBL path against MOM6 ``find_N2`` (MOM_set_diffusivity.F90:1095-1270, extracted verbatim), including
    the bulk N²_bot that overwrites the two lowest interior interfaces (:1202-1258; class (a) fix of 2026-11-01,
    ``docs/design/teos10_refs/mom6_eos.md`` §5).  MOM6 clips the density jumps at 0 (:1173-1176); pyOCEAN keeps the
    signed local N² above the bottom (the BBL kernel only uses max(N², N²_min)), so the comparison is on
    max(N², 0).  Tolerance 1e-9 relative to the column's largest N²: the linear cases take N² from ρ differences
    (ρ ≈ 1030 against Δρ ≈ 0.01, ~1e-11 cancellation); the nonlinear ones agree to ~1e-15.
    """
    cases, ref = _bbl_n2_mom6(tmp_path)
    for c, r in zip(cases, ref):
        n2 = np.maximum(_bbl_n2_pyocean(c), 0.0)
        assert n2.shape == r.shape, c["name"]
        np.testing.assert_allclose(n2, r, rtol=0.0, atol=1e-9 * max(r.max(), 1e-30), err_msg=c["name"])
        if len(c["dz"]) >= 3:
            assert r[1] == r[2], c["name"]                     # MOM6 writes N²_bot at both


@needs("bbl_driver")
def test_bbl_kd_with_bottom_n2_matches_mom6(tmp_path):
    """
    End to end: Kd of ``lotw_bbl_columns`` (Numba) and ``lotw_bbl_columns_numpy`` on the T, S (or ρ) columns against
    MOM6 add_LOTW_BBL_diffusivity fed with MOM6 find_N2's N2_int (the two drivers chained), with no thin layer to
    merge (h_thin 1e-6 m).  u* 1 cm/s, |f| 1e-4 s⁻¹, TKE 2e-7 m³ s⁻³ (energy-limited), OM4 parameters.
    Tolerance 1e-9 max Kd per column (the N² tolerance of the test above carried through).
    """
    from core.components.eos import make_eos
    from physics.mixing.bbl_mixing import BBLParams, lotw_bbl_columns, lotw_bbl_columns_numpy
    cases, ref = _bbl_n2_mom6(tmp_path)
    lotw = [dict(name=c["name"], dz=c["dz"], n2=r, ustar=0.01, absf=1e-4, tke=2e-7, p={})
            for c, r in zip(cases, ref)]
    columns.write_bbl_cases(tmp_path / "in.txt", lotw)
    subprocess.run([str(BUILD / "bbl_driver"), "lotw", str(tmp_path / "in.txt"), str(tmp_path / "out.txt")],
                   check=True, capture_output=True)
    kd_ref = columns.read_bbl_results(tmp_path / "out.txt")
    for c, r in zip(cases, kd_ref):
        dz = c["dz"][:, None]
        kw = dict(p=BBLParams(), rho0=c["rho0"], g=c["g"])
        if c["eos"] == "linear":
            args = (dz, columns.bbl_n2_density(c)[:, None], 0.01, 1e-4, 2e-7, 1e-6)
        else:
            kw.update(T=c["T"][:, None], S=c["S"][:, None], eos=make_eos(c["eos"]))
            args = (dz, np.zeros_like(dz), 0.01, 1e-4, 2e-7, 1e-6)
        kd = lotw_bbl_columns(*args, **kw)[:, 0]
        assert np.array_equal(kd, lotw_bbl_columns_numpy(*args, **kw)[:, 0]), c["name"]
        np.testing.assert_allclose(kd, r["kd"], rtol=0.0, atol=1e-9 * max(r["kd"].max(), 1e-30),
                                   err_msg=c["name"])


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
