"""Tests of the BayIW cut tools (spec docs/superpowers/specs/2026-10-12-bayiw-cut-design.md)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.bayiw_baroclinic import bayiw

HAVE = (bayiw.CASES["2layer"]["dir"] / "input").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="BayIW MITgcm inputs not mounted")


def test_cut_indices_and_stations():
    from validation.bayiw_cut import cut
    m, p = cut.cut_inputs("2layer", "bay", "mit"), cut.cut_inputs("2layer", "bay", "py")
    assert m["H"].shape == (160, 200) and p["H"].shape == (159, 198) and m["T"].shape == (50, 160, 200)
    full = bayiw.load_inputs("2layer", "bay")
    np.testing.assert_array_equal(m["H"], full[2][192:352, 180:380])
    np.testing.assert_array_equal(p["T"], full[3][:, 193:352, 181:379])
    assert m["x0"] == bayiw.X0 + float(full[0][:180].sum()) and p["y0"] == bayiw.Y0 + float(full[1][:193].sum())
    for name, (i, j) in bayiw.STATIONS.items():
        li, lj = cut.local_stations("py")[name]
        assert (li, lj) == (i - 181, j - 193) and np.isfinite(p["H"][lj, li])


def test_mitgcm_inputs_and_radiating_namelists(tmp_path):
    from validation.bayiw_cut import cut
    cut.write_mitgcm_inputs("2layer", "bay", tmp_path, "radiate")
    d = cut.cut_inputs("2layer", "bay", "mit")
    T = np.fromfile(tmp_path / "input" / "theta.bin", ">f8").reshape(50, 160, 200)
    np.testing.assert_array_equal(T, d["T"])
    W = np.fromfile(tmp_path / "input" / "wind.bin", ">f8").reshape(-1, 160, 200)
    np.testing.assert_array_equal(W, d["wind"])
    obcs = (tmp_path / "data.obcs").read_text()
    for s in ("OB_Jsouth = 200*1", "OB_Iwest = 160*1", "OB_Ieast = 160*-1", "useOBCSbalance=.FALSE.",
              "useOBCSsponge=.FALSE.", "useOrlanskiWest=.TRUE."):
        assert s in obcs, s
    data = (tmp_path / "data").read_text()
    assert "dumpFreq=21600." in data and "endTime=216000." in data and "../input" not in data
    diag = (tmp_path / "data.diagnostics").read_text()
    assert "'ETAN'" in diag and "'THETA'" in diag and "frequency(2) = 900." in diag
