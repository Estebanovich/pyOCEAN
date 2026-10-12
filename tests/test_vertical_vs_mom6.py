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
