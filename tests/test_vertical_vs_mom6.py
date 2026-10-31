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
