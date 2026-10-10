"""
Tests for forcing.sources.wrf — raw ``wrfout`` adapter.

Synthetic files follow the WRF-ARW output conventions used by WRF-PACIFICO
(the atmosphere of HYCOM-IOA, see ~/HYCOM/force2ab_PAC_MEX.py): dimensions
Time × south_north × west_east, character ``Times``, 3-D ``XLAT``/``XLONG``
on a Mercator grid, ``Q2`` as mixing ratio, ``RAINC + RAINNC`` accumulated
since the start of each forecast cycle, and grid-relative winds rotated with
``COSALPHA``/``SINALPHA``.
"""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from forcing.interp import fill_land
from forcing.sources.base import BBox, get_source
from forcing.sources.wrf import WRFSource

NY, NX = 30, 40
START = datetime(2015, 4, 1, 0)


def _write_wrfout(
    path: Path,
    start: datetime = START,
    nt: int = 25,
    u10: float = 6.0,
    v10: float = 0.0,
    rain_mm_per_h: float = 3.6,
    alpha_deg: float = 0.0,
    mercator: bool = True,
    land_cols: int = 0,
) -> Path:
    times = [start + timedelta(hours=h) for h in range(nt)]
    tchar = np.array([list(t.strftime("%Y-%m-%d_%H:%M:%S")) for t in times], dtype="S1")
    lon1 = np.linspace(-100.0, -90.0, NX)
    # Mercator: latitude spacing grows poleward, but lat depends only on south_north
    y = np.linspace(np.log(np.tan(np.pi / 4 + np.deg2rad(15.0) / 2)),
                    np.log(np.tan(np.pi / 4 + np.deg2rad(25.0) / 2)), NY)
    lat1 = np.rad2deg(2 * np.arctan(np.exp(y)) - np.pi / 2)
    LON, LAT = np.meshgrid(lon1, lat1)
    if not mercator:   # Lambert-like: latitude varies along west_east too
        LAT = LAT + 0.05 * (LON - LON.mean())
    shape = (nt, NY, NX)
    dims = ("Time", "south_north", "west_east")
    hours = np.arange(nt, dtype=float)[:, None, None]
    landmask = np.zeros((NY, NX))
    if land_cols:
        landmask[:, :land_cols] = 1.0
    t2 = np.full(shape, 298.0)
    t2[:, landmask.astype(bool)] = 310.0           # hot land
    ds = xr.Dataset(
        {
            "Times": (("Time", "DateStrLen"), tchar),
            "XLAT": (dims, np.broadcast_to(LAT, shape).copy()),
            "XLONG": (dims, np.broadcast_to(LON, shape).copy()),
            "U10": (dims, np.full(shape, u10)),
            "V10": (dims, np.full(shape, v10)),
            "T2": (dims, t2),
            "Q2": (dims, np.full(shape, 0.015)),
            "PSFC": (dims, np.full(shape, 101000.0)),
            "SWDOWN": (dims, np.full(shape, 700.0)),
            "GLW": (dims, np.full(shape, 410.0)),
            "RAINC": (dims, np.broadcast_to(0.25 * rain_mm_per_h * hours, shape).copy()),
            "RAINNC": (dims, np.broadcast_to(0.75 * rain_mm_per_h * hours, shape).copy()),
            "COSALPHA": (dims, np.full(shape, np.cos(np.deg2rad(alpha_deg)))),
            "SINALPHA": (dims, np.full(shape, np.sin(np.deg2rad(alpha_deg)))),
            "LANDMASK": (dims, np.broadcast_to(landmask, shape).copy()),
        },
        attrs={"MAP_PROJ": 3 if mercator else 1, "TITLE": " OUTPUT FROM WRF V4.2 MODEL"},
    )
    ds.to_netcdf(path)
    return path


BB = BBox(lon_min=-97.0, lon_max=-93.0, lat_min=18.0, lat_max=22.0)
T0 = datetime(2015, 4, 1, 2)
T1 = datetime(2015, 4, 1, 20)


class TestWRFLoad:

    def test_registered_as_wrf(self, tmp_path):
        src = get_source("wrf", paths=[_write_wrfout(tmp_path / "wrfout_d01_2015-04-01_00:00:00")])
        assert isinstance(src, WRFSource)

    def test_canonical_output(self, tmp_path):
        ds = WRFSource([_write_wrfout(tmp_path / "w.nc")]).load(T0, T1, BB)
        assert set(ds.data_vars) == {"u10", "v10", "t2m", "q2m", "msl", "sw_down", "lw_down", "precip"}
        assert ds.lat.ndim == 1 and ds.lon.ndim == 1
        assert ds.time.values[0] == np.datetime64(T0, "ns") - np.timedelta64(1, "h")

    def test_q2_mixing_ratio_to_specific_humidity(self, tmp_path):
        ds = WRFSource([_write_wrfout(tmp_path / "w.nc")]).load(T0, T1, BB)
        np.testing.assert_allclose(ds.q2m, 0.015 / 1.015, rtol=1e-12)

    def test_surface_pressure_and_radiation_pass_through(self, tmp_path):
        ds = WRFSource([_write_wrfout(tmp_path / "w.nc")]).load(T0, T1, BB)
        np.testing.assert_allclose(ds.msl, 101000.0)
        np.testing.assert_allclose(ds.sw_down, 700.0)
        np.testing.assert_allclose(ds.lw_down, 410.0)

    def test_grid_relative_winds_rotated_to_earth(self, tmp_path):
        # WRF: u_earth = U cosα − V sinα,  v_earth = V cosα + U sinα
        alpha = 30.0
        ds = WRFSource([_write_wrfout(tmp_path / "w.nc", u10=6.0, v10=0.0, alpha_deg=alpha)]).load(T0, T1, BB)
        np.testing.assert_allclose(ds.u10, 6.0 * np.cos(np.deg2rad(alpha)), rtol=1e-12)
        np.testing.assert_allclose(ds.v10, 6.0 * np.sin(np.deg2rad(alpha)), rtol=1e-12)

    def test_precipitation_deaccumulated(self, tmp_path):
        # 3.6 mm/h = 1e-6 m/s
        ds = WRFSource([_write_wrfout(tmp_path / "w.nc", rain_mm_per_h=3.6)]).load(T0, T1, BB)
        np.testing.assert_allclose(ds.precip, 1e-6, rtol=1e-12)

    def test_decreasing_accumulation_gives_no_negative_rain(self, tmp_path):
        # Round-off or a bucket reset without I_RAINNC can make RAINNC drop slightly
        path = _write_wrfout(tmp_path / "w.nc", rain_mm_per_h=0.0)
        ds = xr.load_dataset(path)
        ds["RAINNC"][5:] = ds["RAINNC"][5:] - 0.01
        ds.to_netcdf(tmp_path / "w2.nc")
        out = WRFSource([tmp_path / "w2.nc"]).load(T0, T1, BB)
        assert float(out.precip.min()) >= 0.0

    def test_consecutive_cycles_restart_accumulation(self, tmp_path):
        # Two daily forecast cycles; each restarts RAINC/RAINNC at zero.
        p1 = _write_wrfout(tmp_path / "day1.nc", start=START, nt=25, rain_mm_per_h=3.6)
        p2 = _write_wrfout(tmp_path / "day2.nc", start=START + timedelta(days=1), nt=25, rain_mm_per_h=7.2)
        ds = WRFSource([p2, p1]).load(datetime(2015, 4, 1, 2), datetime(2015, 4, 2, 20), BB)
        rate = ds.precip.isel(lat=0, lon=0).to_series()
        assert float(rate.min()) >= 0.0
        assert rate[np.datetime64("2015-04-01T12")] == pytest.approx(1e-6, rel=1e-9)
        assert rate[np.datetime64("2015-04-02T12")] == pytest.approx(2e-6, rel=1e-9)

    def test_overlapping_times_taken_from_newest_cycle(self, tmp_path):
        p1 = _write_wrfout(tmp_path / "day1.nc", start=START, nt=25, u10=5.0)    # ends 2015-04-02T00
        p2 = _write_wrfout(tmp_path / "day2.nc", start=START + timedelta(days=1), nt=25, u10=9.0)
        ds = WRFSource([p1, p2]).load(datetime(2015, 4, 1, 2), datetime(2015, 4, 2, 20), BB)
        assert np.all(np.diff(ds.time.values) > np.timedelta64(0))
        assert float(ds.u10.sel(time="2015-04-02T00").isel(lat=0, lon=0)) == 9.0

    def test_land_masked_for_sea_over_land_fill(self, tmp_path):
        ds = WRFSource([_write_wrfout(tmp_path / "w.nc", land_cols=12)]).load(T0, T1, BB)
        assert np.isnan(ds.t2m.isel(time=0, lon=0)).all()        # westernmost columns are land
        filled = fill_land(ds.t2m)
        np.testing.assert_allclose(filled, 298.0)                  # hot land air does not leak

    def test_land_kept_when_requested(self, tmp_path):
        ds = WRFSource([_write_wrfout(tmp_path / "w.nc", land_cols=12)], mask_land=False).load(T0, T1, BB)
        assert not np.isnan(ds.t2m).any()

    def test_non_rectilinear_grid_rejected(self, tmp_path):
        with pytest.raises(NotImplementedError, match="rectilinear"):
            WRFSource([_write_wrfout(tmp_path / "w.nc", mercator=False)]).load(T0, T1, BB)

    def test_no_files_rejected(self):
        with pytest.raises(ValueError, match="paths"):
            WRFSource([])
