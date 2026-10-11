"""
Tests for forcing.sources.hycom_ab — native HYCOM .a/.b file reader.

HYCOM stores each 2-D field as a big-endian float32 record of idm·jdm
words padded to a multiple of 4096 words; land/void points hold 2^100.
The companion .b text file carries a header and one line per record with
its name and min/max.  Three .b dialects are covered: archive (archv),
grid/depth ("name: min,max = …") and forcing ("name: day,span,range = …").
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from forcing.sources.hycom_ab import hycom_npad, read_b, read_field

IDM, JDM = 7, 5
SPVAL = 2.0 ** 100


def _write_a(path: Path, fields: list[np.ndarray]) -> None:
    npad = (-IDM * JDM) % 4096
    with open(path, "wb") as f:
        for fld in fields:
            rec = np.where(np.isnan(fld), SPVAL, fld).astype(">f4").ravel()
            rec.tofile(f)
            np.zeros(npad, dtype=">f4").tofile(f)


def _fields() -> dict[str, np.ndarray]:
    j, i = np.mgrid[0:JDM, 0:IDM]
    ssh = 0.01 * i - 0.02 * j
    t1 = 25.0 + 0.1 * i
    t2 = 20.0 + 0.1 * j
    t1[0, 0] = t2[0, 0] = ssh[0, 0] = np.nan          # land
    return {"ssh": ssh, "t1": t1, "t2": t2, "dp1": np.full((JDM, IDM), 9810.0),
            "dp2": np.full((JDM, IDM), 98100.0)}


def _archive(tmp: Path, corrupt_minmax: bool = False) -> Path:
    f = _fields()
    order = [("srfhgt", 0, f["ssh"]), ("thknss", 1, f["dp1"]), ("temp", 1, f["t1"]),
             ("thknss", 2, f["dp2"]), ("temp", 2, f["t2"])]
    _write_a(tmp / "archv.2026_100_00.a", [x for *_, x in order])
    lines = [
        "HYCOM archive, synthetic test", "line 2", "line 3", "line 4",
        "  23    'iversn' = hycom version number x10",
        "  10    'iexpt ' = experiment number x10",
        "   3    'yrflag' = days in year flag",
        f"   {IDM}    'idm   ' = longitudinal array size",
        f"   {JDM}    'jdm   ' = latitudinal  array size",
        "field       time step  model day  k  dens        min              max",
    ]
    for name, k, x in order:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if corrupt_minmax and name == "temp" and k == 2:
            hi += 5.0
        lines.append(f"{name:<8} = {52272:10d} {45006.0:11.3f} {k:2d} {34.0:6.3f}   {lo:14.7E}  {hi:14.7E}")
    (tmp / "archv.2026_100_00.b").write_text("\n".join(lines) + "\n")
    return tmp / "archv.2026_100_00.b"


def _grid(tmp: Path) -> Path:
    j, i = np.mgrid[0:JDM, 0:IDM]
    plon, plat = -140.0 + 0.04 * i, -5.0 + 0.04 * j
    _write_a(tmp / "regional.grid.a", [plon, plat])
    (tmp / "regional.grid.b").write_text(
        f" {IDM}    'idm   ' = longitudinal array size\n"
        f" {JDM}    'jdm   ' = latitudinal  array size\n"
        "    2    'mapflg' = map flag (-1=unknown,0=mercator,2=uniform,4=f-plane)\n"
        f"plon:  min,max =  {plon.min():14.5f}  {plon.max():14.5f}\n"
        f"plat:  min,max =  {plat.min():14.5f}  {plat.max():14.5f}\n")
    return tmp / "regional.grid.b"


def _forcing(tmp: Path) -> Path:
    recs = [np.full((JDM, IDM), 24.0 + h) for h in range(3)]
    _write_a(tmp / "airtmp_20260829.a", recs)
    lines = ["WRF-PACIFICO airtmp", "line 2", "line 3", "line 4", f"i/jdm = {IDM} {JDM}"]
    for h, r in enumerate(recs):
        lines.append(f"  airtmp: day,span,range = {46263.0 + h / 24:10.4f} 0.041667   "
                     f"{r.min():9.6E}   {r.max():9.6E}")
    (tmp / "airtmp_20260829.b").write_text("\n".join(lines) + "\n")
    return tmp / "airtmp_20260829.b"


# ===========================================================================

class TestPadding:

    def test_pac_mex_padding(self):
        # force2ab_PAC_MEX.py: 1600 × 1125 → npad = 2240
        assert hycom_npad(1600, 1125) == 2240

    def test_exact_multiple_needs_no_padding(self):
        assert hycom_npad(4096, 2) == 0


class TestReadB:

    def test_archive_header_and_records(self, tmp_path):
        b = read_b(_archive(tmp_path))
        assert b.header["idm"] == IDM and b.header["jdm"] == JDM and b.header["iversn"] == 23
        assert [(r.name, r.k) for r in b.records] == [
            ("srfhgt", 0), ("thknss", 1), ("temp", 1), ("thknss", 2), ("temp", 2)]
        assert b.records[2].min == pytest.approx(25.0, rel=1e-6)

    def test_grid_dialect(self, tmp_path):
        b = read_b(_grid(tmp_path))
        assert b.header["mapflg"] == 2
        assert [r.name for r in b.records] == ["plon", "plat"]
        assert b.records[0].min == pytest.approx(-140.0)

    def test_forcing_dialect(self, tmp_path):
        b = read_b(_forcing(tmp_path))
        assert [r.name for r in b.records] == ["airtmp"] * 3
        assert b.records[1].day == pytest.approx(46263.0 + 1 / 24, abs=1e-4)


class TestReadField:

    def test_field_by_name_and_layer(self, tmp_path):
        x = read_field(_archive(tmp_path), "temp", k=2)
        np.testing.assert_allclose(x, _fields()["t2"], rtol=1e-6)
        assert x.shape == (JDM, IDM)

    def test_land_is_nan(self, tmp_path):
        x = read_field(_archive(tmp_path), "srfhgt")
        assert np.isnan(x[0, 0]) and np.isfinite(x[1:, 1:]).all()

    def test_grid_field_uses_header_dimensions(self, tmp_path):
        x = read_field(_grid(tmp_path), "plat")
        np.testing.assert_allclose(x[:, 0], -5.0 + 0.04 * np.arange(JDM), rtol=1e-6)

    def test_nth_record_of_repeated_name(self, tmp_path):
        x = read_field(_forcing(tmp_path), "airtmp", index=2)
        np.testing.assert_allclose(x, 26.0)

    def test_explicit_dimensions_accepted(self, tmp_path):
        np.testing.assert_allclose(read_field(_forcing(tmp_path), "airtmp", idm=IDM, jdm=JDM), 24.0)

    def test_minmax_mismatch_detected(self, tmp_path):
        with pytest.raises(ValueError, match="min/max"):
            read_field(_archive(tmp_path, corrupt_minmax=True), "temp", k=2)

    def test_unknown_field(self, tmp_path):
        with pytest.raises(KeyError, match="salin"):
            read_field(_archive(tmp_path), "salin", k=1)

    def test_truncated_a_file(self, tmp_path):
        b = _archive(tmp_path)
        a = b.with_suffix(".a")
        a.write_bytes(a.read_bytes()[: 4096 * 4 * 2])
        with pytest.raises(ValueError, match="short"):
            read_field(b, "temp", k=2)


class TestOtherDialects:
    """Dialects seen in the HYCOM-IOA PAC_MEX files (regional.depth, rmu, chl)."""

    def _write(self, tmp: Path, name: str, lines: list[str], recs: list[np.ndarray]) -> Path:
        _write_a(tmp / f"{name}.a", recs)
        (tmp / f"{name}.b").write_text("\n".join(lines) + "\n")
        return tmp / f"{name}.b"

    def test_depth_dialect(self, tmp_path):
        d = np.linspace(5.0, 6106.4, IDM * JDM).reshape(JDM, IDM)
        b = self._write(tmp_path, "regional.depth", ["title"] * 5 + [
            f"min,max depth = {d.min():11.7f} {d.max():11.5f}"], [d])
        assert read_b(b).records[0].name == "depth"
        np.testing.assert_allclose(read_field(b, "depth", idm=IDM, jdm=JDM), d, rtol=1e-6)

    def test_range_dialect(self, tmp_path):
        r = np.full((JDM, IDM), 1.1574074e-4)
        b = self._write(tmp_path, "rmu", ["Relaxation Mask", "", "", "", f"i/jdm = {IDM} {JDM}",
                                          f"     rmu: range = {0.0:16.7E} {r.max():16.7E}"], [r])
        r[0, 0] = 0.0
        _write_a(tmp_path / "rmu.a", [r])
        np.testing.assert_allclose(read_field(b, "rmu"), r, rtol=1e-6)

    def test_month_dialect(self, tmp_path):
        recs = [np.full((JDM, IDM), 0.05 * (m + 1)) for m in range(3)]
        lines = ["SeaWifs monthly clim", "", "", "", f"i/jdm = {IDM} {JDM}"] + [
            f"     chl: month,range = {m + 1:02d}   {x.min():14.7E}   {x.max():14.7E}"
            for m, x in enumerate(recs)]
        b = self._write(tmp_path, "chl", lines, recs)
        assert [r.name for r in read_b(b).records] == ["chl"] * 3
        np.testing.assert_allclose(read_field(b, "chl", index=2), 0.15, rtol=1e-6)

    def test_sal_dialect_without_header(self, tmp_path):
        # tidal.salReIm_tpxo10.b (HYCOM-IOA, LUSTRE): no title, no i/jdm, "min, max" with a space
        recs = [np.full((JDM, IDM), -6.4304397e-02), np.full((JDM, IDM), 5.7942215e-02)]
        b = self._write(tmp_path, "tidal.salReIm", [
            f"SAL-m2R : min, max =  {recs[0].min():15.7E}  {recs[0].max():15.7E}",
            f"SAL-m2I : min, max =  {recs[1].min():15.7E}  {recs[1].max():15.7E}"], recs)
        assert [r.name for r in read_b(b).records] == ["SAL-m2R", "SAL-m2I"]
        np.testing.assert_allclose(read_field(b, "SAL-m2I", idm=IDM, jdm=JDM), 5.7942215e-02, rtol=1e-6)

    def test_unnamed_minmax_dialect(self, tmp_path):
        # JSL.b (HYCOM-examples GLBp0.08) and tidal.rh.b: a single "min, max =" line, no name
        x = np.full((JDM, IDM), 2.2973286e-03)
        x[0, 0] = 0.0
        b = self._write(tmp_path, "tidal.rh", [f"min, max =   0.0000000E+00   {x.max():14.7E}"], [x])
        r = read_b(b).records
        assert len(r) == 1 and r[0].name == "tidal.rh"
        np.testing.assert_allclose(read_field(b, "tidal.rh", idm=IDM, jdm=JDM), x, rtol=1e-6)

    def test_dash_named_minmax_dialect(self, tmp_path):
        # relax/DRAG/cb.b (HYCOM-IOA spatial bottom drag): "CB - min, max = a b"
        x = np.linspace(2.5e-3, 5.2e-3, IDM * JDM).reshape(JDM, IDM)
        b = self._write(tmp_path, "cb", [f"CB - min, max = {x.min():15.8E} {x.max():15.8E}"], [x])
        assert read_b(b).records[0].name == "CB"
        np.testing.assert_allclose(read_field(b, "CB", idm=IDM, jdm=JDM), x, rtol=1e-6)

    def test_name_with_space(self, tmp_path):
        recs = [np.full((JDM, IDM), 8e-6)]
        b = self._write(tmp_path, "rivers", ["Monthly river inflow", "", "", "",
                                             f"   5 smoothing passes.  i/jdm =  {IDM}  {JDM}",
                                             "  river precip: month,range =  1    8.0000000E-06   8.0000000E-06"],
                        recs)
        np.testing.assert_allclose(read_field(b, "river precip"), 8e-6, rtol=1e-6)

    def test_relaxation_layers(self, tmp_path):
        # relax tem/sal/int files: month, layer k, target density, min, max
        recs = [np.full((JDM, IDM), 28.0 - 4.0 * k) for k in range(3)]
        lines = ["relax", "", "", "", f"i/jdm = {IDM} {JDM}"] + [
            f" tem: month,layer,dens,range = 01  {k + 1:02d} {17.0 + k:6.3f}  {x.min():14.7E}  {x.max():14.7E}"
            for k, x in enumerate(recs)]
        b = self._write(tmp_path, "relax_tem", lines, recs)
        assert [r.k for r in read_b(b).records] == [1, 2, 3]
        np.testing.assert_allclose(read_field(b, "tem", k=2), 24.0)

    def test_title_line_dimensions_ignored(self, tmp_path):
        # topo_NUEVA_25/depth_GLB*.b: a title line quotes the *source* global grid
        d = np.full((JDM, IDM), 100.0)
        b = self._write(tmp_path, "depth_sub", [
            "depth_GLBb0.08 subregioned to PAC_MEX", "i/jdm = 4500 3298; plon,plat range = 74 434",
            "", "", "", "min,max depth =  100.0  100.0"], [d])
        assert "idm" not in read_b(b).header
        with pytest.raises(ValueError, match="idm"):
            read_field(b, "depth")
        np.testing.assert_allclose(read_field(b, "depth", idm=IDM, jdm=JDM), 100.0)

    def test_a_file_size_must_match_dimensions(self, tmp_path):
        d = np.full((JDM, IDM), 100.0)
        b = self._write(tmp_path, "depth_wrong", ["t"] * 5 + ["min,max depth =  100.0  100.0"], [d])
        with pytest.raises(ValueError, match="size"):
            read_field(b, "depth", idm=100, jdm=50)   # 5000 words → 2 blocks per record
