"""
Tests for forcing.sources.hycom_archive — native HYCOM archives (archv/archm)
on hybrid layers → canonical ocean on z levels.

Conventions under test (HYCOM-src mod_archiv.F90; HYCOM-tools archv2data3z.f):
  * srfhgt is geopotential [m² s⁻²]:  ssh = srfhgt / (thref·9806) = srfhgt / 9.806
  * thknss is layer thickness in pressure units [Pa]:  h = dp / 9806
  * archv (artype 1): u-vel./v-vel. are baroclinic; total = u-vel. + u_btrop
  * archm (artype 2): u-vel./v-vel. are already total
  * u at the western face, v at the southern face of each p-cell (C-grid)
  * model day with yrflag=3 counts days since 1900-12-31 00:00
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest

from forcing.sources.base import BBox, get_source
from forcing.sources.hycom_archive import HYCOMArchiveSource, remap_layers_to_z

IDM, JDM, KDM = 8, 6, 3
ONEM = 9806.0
SPVAL = 2.0 ** 100
DAY = 45006.25                                     # 2024-03-22T06 for yrflag=3


def _write_ab(tmp: Path, stem: str, header: list[str], records: list[tuple[str, int, np.ndarray]],
              archive_style: bool = True) -> Path:
    npad = (-IDM * JDM) % 4096
    with open(tmp / f"{stem}.a", "wb") as f:
        for _, _, x in records:
            np.where(np.isnan(x), SPVAL, x).astype(">f4").ravel().tofile(f)
            np.zeros(npad, ">f4").tofile(f)
    lines = list(header)
    for name, k, x in records:
        lo, hi = np.nanmin(x), np.nanmax(x)
        if archive_style:
            lines.append(f"{name:<8} = {52272:11d}{DAY:11.3f}{k:4d}{34.0:7.3f}{lo:16.7E}{hi:16.7E}")
        else:
            lines.append(f"{name}:  min,max = {lo:16.7E} {hi:16.7E}")
    (tmp / f"{stem}.b").write_text("\n".join(lines) + "\n")
    return tmp / f"{stem}.b"


def _grid(tmp: Path, skew: float = 0.0) -> Path:
    j, i = np.mgrid[0:JDM, 0:IDM].astype(float)
    dl = 0.5
    plon, plat = -100.0 + dl * i, 15.0 + dl * j + skew * i
    recs = [("plon", 0, plon), ("plat", 0, plat), ("qlon", 0, plon - dl / 2), ("qlat", 0, plat - dl / 2),
            ("ulon", 0, plon - dl / 2), ("ulat", 0, plat), ("vlon", 0, plon), ("vlat", 0, plat - dl / 2)]
    hdr = [f" {IDM}    'idm   ' = longitudinal array size", f" {JDM}    'jdm   ' = latitudinal  array size",
           "    2    'mapflg' = map flag (-1=unknown,0=mercator,2=uniform,4=f-plane)"]
    return _write_ab(tmp, "regional.grid", hdr, recs, archive_style=False)


def _fields():
    j, i = np.mgrid[0:JDM, 0:IDM].astype(float)
    h = [np.full((JDM, IDM), 10.0), np.full((JDM, IDM), 40.0), np.full((JDM, IDM), 950.0)]
    h[1][:, -1] = 0.0                               # vanished layer in the east column
    h[2][:, 0] = 50.0                               # shallow west column: bottom at 100 m
    temp = [np.full((JDM, IDM), 28.0), np.full((JDM, IDM), 20.0), np.full((JDM, IDM), 6.0)]
    salt = [np.full((JDM, IDM), 36.0), np.full((JDM, IDM), 36.3), np.full((JDM, IDM), 34.9)]
    ubc = [np.full((JDM, IDM), 0.2), np.full((JDM, IDM), 0.05), np.full((JDM, IDM), -0.01)]
    vbc = [np.zeros((JDM, IDM)) for _ in range(KDM)]
    ubt = 0.1 * i                                   # barotropic u varies eastward
    vbt = np.full((JDM, IDM), -0.05)
    ssh = 0.3 + 0.01 * j
    for x in (*h, *temp, *salt, *ubc, *vbc):
        x[0, 0] = np.nan                            # land corner
    ssh[0, 0] = ubt[0, 0] = vbt[0, 0] = np.nan
    return dict(h=h, temp=temp, salt=salt, ubc=ubc, vbc=vbc, ubt=ubt, vbt=vbt, ssh=ssh)


def _archive(tmp: Path, stem: str = "archv.2024_082_06") -> Path:
    f = _fields()
    recs = [("montg1", 0, f["ssh"] * 9.806), ("srfhgt", 0, f["ssh"] * 9.806),
            ("u_btrop", 0, f["ubt"]), ("v_btrop", 0, f["vbt"])]
    for k in range(KDM):
        recs += [("u-vel.", k + 1, f["ubc"][k] if stem.startswith("archv") else f["ubc"][k] + f["ubt"]),
                 ("v-vel.", k + 1, f["vbc"][k] if stem.startswith("archv") else f["vbc"][k] + f["vbt"]),
                 ("thknss", k + 1, f["h"][k] * ONEM),
                 ("temp", k + 1, f["temp"][k]), ("salin", k + 1, f["salt"][k])]
    hdr = ["HYCOM archive", "t2", "t3", "t4",
           "  23    'iversn' = hycom version number x10", "  10    'iexpt ' = experiment number x10",
           "   3    'yrflag' = days in year flag", f"   {IDM}    'idm   ' = longitudinal array size",
           f"   {JDM}    'jdm   ' = latitudinal  array size",
           "field       time step  model day  k  dens        min              max"]
    return _write_ab(tmp, stem, hdr, recs)


DEPTHS = np.array([0.0, 5.0, 10.0, 30.0, 49.0, 50.0, 99.0, 150.0, 999.0, 1500.0])
BB = BBox(-99.0, -97.0, 16.0, 17.0)
T_ARCH = datetime(1900, 12, 31) + timedelta(days=DAY)


# ===========================================================================
# Layer → z remapping
# ===========================================================================

class TestRemapLayersToZ:

    def test_piecewise_constant_sampling(self):
        h = np.array([10.0, 40.0, 950.0])[:, None]
        v = np.array([28.0, 20.0, 6.0])[:, None]
        out = remap_layers_to_z(v, h, np.array([0.0, 9.99, 10.0, 49.0, 50.0, 999.0]))
        np.testing.assert_allclose(out[:, 0], [28.0, 28.0, 20.0, 20.0, 6.0, 6.0])

    def test_below_bottom_is_nan(self):
        h = np.array([10.0, 40.0, 50.0])[:, None]
        out = remap_layers_to_z(np.array([1.0, 2.0, 3.0])[:, None], h, np.array([99.0, 100.0, 150.0]))
        assert out[0, 0] == 3.0 and np.isnan(out[1:, 0]).all()

    def test_vanished_layers_skipped(self):
        h = np.array([10.0, 0.0, 0.0, 90.0])[:, None]
        out = remap_layers_to_z(np.array([1.0, 2.0, 3.0, 4.0])[:, None], h, np.array([10.0, 50.0]))
        np.testing.assert_allclose(out[:, 0], [4.0, 4.0])

    def test_land_column_stays_nan(self):
        h = np.full((3, 1), np.nan)
        assert np.isnan(remap_layers_to_z(np.ones((3, 1)), h, np.array([0.0, 10.0]))).all()


# ===========================================================================
# Archive adapter
# ===========================================================================

class TestHYCOMArchiveSource:
    """``open_all()`` = canonical dataset over the whole archive domain (no subsetting)."""

    def _src(self, tmp, stem="archv.2024_082_06", **kw):
        return HYCOMArchiveSource([_archive(tmp, stem)], grid=_grid(tmp), depths=DEPTHS, **kw)

    def test_registered(self, tmp_path):
        src = get_source("hycom_archive", paths=[_archive(tmp_path)], grid=_grid(tmp_path))
        assert isinstance(src, HYCOMArchiveSource)

    def test_canonical_output(self, tmp_path):
        ds = self._src(tmp_path).load(T_ARCH - timedelta(hours=1), T_ARCH + timedelta(hours=1), BB)
        assert set(ds.data_vars) == {"temp", "salt", "u", "v", "ssh"}
        assert ds.temp.dims == ("time", "depth", "lat", "lon")
        np.testing.assert_array_equal(ds.depth, DEPTHS)

    def test_model_day_to_time(self, tmp_path):
        b = _archive(tmp_path)
        assert HYCOMArchiveSource.archive_time(b) == np.datetime64(T_ARCH, "ns")

    def test_ssh_from_srfhgt(self, tmp_path):
        ds = self._src(tmp_path).open_all()
        np.testing.assert_allclose(ds.ssh.isel(time=0).values[1:, 1:], _fields()["ssh"][1:, 1:], rtol=1e-6)

    def test_temperature_profile_from_layers(self, tmp_path):
        ds = self._src(tmp_path).open_all()
        col = ds.temp.isel(time=0, lat=3, lon=3).values
        np.testing.assert_allclose(col[:8], [28, 28, 20, 20, 20, 6, 6, 6], rtol=1e-6)

    def test_shallow_column_nan_below_bottom(self, tmp_path):
        ds = self._src(tmp_path).open_all()
        col = ds.temp.isel(time=0, lat=3, lon=0).values      # bottom at 100 m
        assert col[6] == pytest.approx(6.0) and np.isnan(col[7:]).all()

    def test_vanished_layer_column(self, tmp_path):
        ds = self._src(tmp_path).open_all()
        col = ds.temp.isel(time=0, lat=3, lon=-1).values     # layer 2 has zero thickness
        np.testing.assert_allclose(col[2:4], [6.0, 6.0], rtol=1e-6)

    def test_archv_velocity_is_baroclinic_plus_barotropic_at_p_points(self, tmp_path):
        ds = self._src(tmp_path).open_all()
        f = _fields()
        i, j = 3, 3
        ubt_p = 0.5 * (f["ubt"][j, i] + f["ubt"][j, i + 1])  # u-faces i and i+1 → p-cell i
        assert ds.u.isel(time=0, depth=0, lat=j, lon=i).item() == pytest.approx(0.2 + ubt_p, rel=1e-6)
        assert ds.v.isel(time=0, depth=0, lat=j, lon=i).item() == pytest.approx(-0.05, rel=1e-6)

    def test_archm_velocity_already_total(self, tmp_path):
        ds = self._src(tmp_path, stem="archm.2024_082_06").open_all()
        f = _fields()
        ubt_p = 0.5 * (f["ubt"][3, 3] + f["ubt"][3, 4])
        assert ds.u.isel(time=0, depth=0, lat=3, lon=3).item() == pytest.approx(0.2 + ubt_p, rel=1e-6)

    def test_coordinates_from_regional_grid(self, tmp_path):
        ds = self._src(tmp_path).open_all()
        np.testing.assert_allclose(ds.lon, -100.0 + 0.5 * np.arange(IDM))
        np.testing.assert_allclose(ds.lat, 15.0 + 0.5 * np.arange(JDM))

    def test_curvilinear_grid_rejected(self, tmp_path):
        with pytest.raises(NotImplementedError, match="rectilinear"):
            HYCOMArchiveSource([_archive(tmp_path)], grid=_grid(tmp_path, skew=0.01)).open_all()

    def test_temperature_is_potential(self, tmp_path):
        assert "potential" in self._src(tmp_path).open_all().temp.attrs["comment"]
