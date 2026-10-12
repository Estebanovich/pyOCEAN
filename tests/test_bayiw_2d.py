"""Tests of the BayIW 2-D section tools (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.bayiw_baroclinic import bayiw

HAVE = (bayiw.CASES["2layer"]["dir"] / "input").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="BayIW MITgcm inputs not mounted")


def test_section_column_is_i280_with_the_bay():
    from validation.bayiw_2d import section
    d = section.section_inputs("2layer", "bay")
    assert d["dx"].shape == (1,) and d["dy"].shape == (352,) and d["H"].shape == (352, 1)
    assert 120.0 < d["H"][345, 0] < 170.0 and abs(d["H"][200, 0] - 599.9) < 1e-9
    assert d["T"].shape == (50, 352, 1) and d["wind"].shape == (180, 352, 1)
    assert section.SECTION_STATIONS == {"Gamma": 215, "C": 265, "Omega": 315, "Seiche2": 325, "Seiche": 345}
    full = bayiw.load_inputs("2layer", "bay")
    np.testing.assert_array_equal(d["T"][:, :, 0], full[3][:, :, 280])
    assert d["x0"] == bayiw.X0 + float(full[0][:280].sum())


def test_mitgcm_inputs_round_trip_to_the_column(tmp_path):
    # review focus 1: big-endian float64, (Nr, Ny, Nx) and (nrec, Ny, Nx); Nx = 3 with land walls at i = 0, 2
    from validation.bayiw_2d import section
    section.write_mitgcm_inputs("2layer", "bay", tmp_path)
    d = section.section_inputs("2layer", "bay")
    T = np.fromfile(tmp_path / "input" / "theta.bin", ">f8").reshape(50, 352, 3)
    W = np.fromfile(tmp_path / "input" / "wind.bin", ">f8").reshape(180, 352, 3)
    B = np.fromfile(tmp_path / "input" / "bathy.bin", ">f8").reshape(352, 3)
    for i in range(3):
        np.testing.assert_array_equal(T[:, :, i], d["T"][:, :, 0])
        np.testing.assert_array_equal(W[:, :, i], d["wind"][:, :, 0])
    np.testing.assert_array_equal(B[:, 1], np.where(np.isfinite(d["H"][:, 0]), -d["H"][:, 0], 0.0))
    assert (B[:, 0] == 0).all() and (B[:, 2] == 0).all()
    data = (tmp_path / "data").read_text()
    assert "endTime=216000." in data and "'input/theta.bin'" in data and "'input/wind.bin'" in data
    assert "../input" not in data and "hydrogSaltFile" not in data and "deltaT=30." in data
    obcs = (tmp_path / "data.obcs").read_text()
    assert "OB_Jsouth = 0, 1, 0," in obcs and "OB_Iwest" not in obcs and "OBCS_balanceFacS=-1." in obcs


def test_mitgcm_periodic_variant_has_three_wet_columns(tmp_path):
    from validation.bayiw_2d import section
    section.write_mitgcm_inputs("linear", "nobay", tmp_path, periodic=True)
    B = np.fromfile(tmp_path / "input" / "bathy.bin", ">f8").reshape(352, 3)
    np.testing.assert_array_equal(B[:, 0], B[:, 1])
    np.testing.assert_array_equal(B[:, 2], B[:, 1])
    assert "OB_Jsouth = 3*1," in (tmp_path / "data.obcs").read_text()
    assert "hydrogSaltFile" not in (tmp_path / "data").read_text()
