"""
Test of validation/hycom_ioa/extract_archs_ssh.py (standalone extractor run on
the cluster): srfhgt record located through the archive .b, η = srfhgt / 9.806,
subset to the lon/lat box, data void → NaN.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "hycom_ioa"))
from extract_archs_ssh import main  # noqa: E402

IDM, JDM = 30, 20


def _write_a(path: Path, recs: list[np.ndarray]) -> None:
    npad = (4096 - (IDM * JDM) % 4096) % 4096
    with open(path, "wb") as f:
        for r in recs:
            r.astype(">f4").tofile(f)
            np.zeros(npad, ">f4").tofile(f)


def _grid(tmp: Path) -> Path:
    lon = -120.0 + 1.0 * np.arange(IDM)
    lat = 10.0 + 1.0 * np.arange(JDM)
    plon, plat = np.meshgrid(lon, lat)
    _write_a(tmp / "regional.grid.a", [plon, plat])
    (tmp / "regional.grid.b").write_text(
        f" {IDM}    'idm   ' = longitudinal array size\n {JDM}    'jdm   ' = latitudinal  array size\n"
        "    2    'mapflg' = map flag\n"
        f"plon:  min,max = {plon.min():15.5f} {plon.max():15.5f}\n"
        f"plat:  min,max = {plat.min():15.5f} {plat.max():15.5f}\n")
    return tmp / "regional.grid.b"


def _archs(tmp: Path, name: str, day: float, eta: np.ndarray) -> Path:
    montg = np.full((JDM, IDM), 3.0)
    srf = eta * 9.806
    srf_a = np.where(np.isnan(srf), 2.0 ** 100, srf)
    _write_a(tmp / f"{name}.a", [montg, srf_a])
    hdr = ["Archive", "", "", "", f"  23    'iversn' = version", "  10    'iexpt ' = expt", "   3    'yrflag' = flag",
           f"  {IDM}    'idm   ' = longitudinal", f"  {JDM}    'jdm   ' = latitudinal",
           "field       time step  model day  k  dens        min              max"]
    lines = [f"montg1   =       10 {day:11.3f}   0  0.000 {montg.min():16.7E} {montg.max():16.7E}",
             f"srfhgt   =       10 {day:11.3f}   0  0.000 {np.nanmin(srf):16.7E} {np.nanmax(srf):16.7E}"]
    (tmp / f"{name}.b").write_text("\n".join(hdr + lines) + "\n")
    return tmp / f"{name}.b"


def test_extracts_box_in_metres(tmp_path, monkeypatch):
    g = _grid(tmp_path)
    eta1 = np.linspace(-1.0, 1.0, IDM * JDM).reshape(JDM, IDM)
    eta1[0, 0] = np.nan                                           # land
    eta2 = eta1 + 0.5
    files = [_archs(tmp_path, "000_archs.2026_271_13", 45927.5417, eta1),
             _archs(tmp_path, "000_archs.2026_271_14", 45927.5833, eta2)]
    monkeypatch.setenv("LON0", "-115")
    monkeypatch.setenv("LON1", "-105")
    monkeypatch.setenv("LAT0", "12")
    monkeypatch.setenv("LAT1", "20")
    main([str(tmp_path / "out.npz"), str(g), *map(str, files)])
    d = np.load(tmp_path / "out.npz")
    np.testing.assert_allclose(d["model_day"], [45927.5417, 45927.5833])
    assert d["lon"][0] == -115.0 and d["lon"][-1] == -106.0 and d["lat"][0] == 12.0 and d["lat"][-1] == 19.0
    np.testing.assert_allclose(d["ssh"][1], eta2[2:10, 5:15], rtol=1e-6)


def test_minmax_mismatch_rejected(tmp_path):
    import re
    g = _grid(tmp_path)
    b = _archs(tmp_path, "000_archs.bad", 1.0, np.zeros((JDM, IDM)))
    lines = b.read_text().splitlines()
    lines[-1] = re.sub(r"(\S+)\s+(\S+)$", "9.0000000E+00 9.5000000E+00", lines[-1])   # wrong srfhgt min/max
    b.write_text("\n".join(lines) + "\n")
    with pytest.raises(ValueError, match="min/max differ"):
        main([str(tmp_path / "o.npz"), str(g), str(b)])


def test_directory_searched_recursively(tmp_path, monkeypatch):
    g = _grid(tmp_path)
    sub = tmp_path / "tar" / "tars_0260271012" / "data"
    sub.mkdir(parents=True)
    for k, day in enumerate((2.0, 1.0)):                       # written out of order
        _archs(sub, f"archs.2026_27{k}_00", day, np.full((JDM, IDM), float(k)))
    _archs(sub, "000_archs.dummy", 9.0, np.zeros((JDM, IDM)))
    main([str(tmp_path / "o.npz"), str(g), str(tmp_path / "tar" / "tars_0260271012")])
    d = np.load(tmp_path / "o.npz")
    assert list(d["files"]) == ["archs.2026_270_00.b", "archs.2026_271_00.b"]


def test_empty_directory_reports_contents(tmp_path):
    g = _grid(tmp_path)
    (tmp_path / "empty").mkdir()
    (tmp_path / "empty" / "archs.tar").write_text("x")
    with pytest.raises(SystemExit, match="archs.tar"):
        main([str(tmp_path / "o.npz"), str(g), str(tmp_path / "empty")])
