"""
Tests for forcing.sources.hycom_z — HYCOM z-level NetCDF (ESPC-D-V02, GOFS 3.1)
and for the vertical interpolation forcing.interp.to_model_levels.

Synthetic files reproduce the metadata of the real ESPC-D-V02 forecast files
on data.hycom.org (checked 2026-10-10 from the header of
US058GCOM-OPSnce.espc-d-031-hycom_fcst_glby008_2026100112_t0000_t3z.nc):
one variable per file (t3z: water_temp, s3z: salinity, u3z: water_u,
v3z: water_v, ssh: surf_el), NetCDF-3 classic, packed shorts with
scale_factor/add_offset/_FillValue = −30000, 0–360 °E longitudes, depth
positive down, time in hours since 2000-01-01, an extra ``tau`` variable,
and water_temp with standard_name = sea_water_temperature (in situ).
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gsw
import numpy as np
import pytest
import xarray as xr

from core.grid.vertical import create_z_grid
from forcing.interp import to_model_levels
from forcing.sources.base import BBox, get_source
from forcing.sources.hycom_z import HYCOMZSource

DEPTH = np.array([0.0, 10.0, 50.0, 100.0, 500.0, 1000.0, 2000.0, 4000.0])
LON = np.arange(260.0, 270.01, 0.5)            # 0–360 °E, as glby0.08
LAT = np.arange(15.0, 25.01, 0.5)
HOURS = np.array([231_396.0, 231_399.0, 231_402.0])   # 2026-05-25T12 + 0, 3, 6 h
T_REF = datetime(2026, 5, 25, 12)

_VARS = {
    "t3z": ("water_temp", "sea_water_temperature", "degC", 0.001, 20.0, True),
    "s3z": ("salinity", "sea_water_salinity", "psu", 0.001, 20.0, True),
    "u3z": ("water_u", "eastward_sea_water_velocity", "m/s", 0.001, 0.0, True),
    "v3z": ("water_v", "northward_sea_water_velocity", "m/s", 0.001, 0.0, True),
    "ssh": ("surf_el", "sea_surface_elevation", "m", 0.001, 0.0, False),
}


def _field(code: str) -> np.ndarray:
    nz, ny, nx = DEPTH.size, LAT.size, LON.size
    z = DEPTH[:, None, None]
    if code == "t3z":
        f = 28.0 - 24.0 * (1.0 - np.exp(-z / 600.0)) + 0.0 * LON[None, None, :]
    elif code == "s3z":
        f = 36.0 - 1.2 * (1.0 - np.exp(-z / 800.0)) + 0.0 * LON[None, None, :]
    elif code == "u3z":
        f = 0.3 * np.exp(-z / 200.0) + 0.0 * LON[None, None, :]
    elif code == "v3z":
        f = -0.1 * np.exp(-z / 200.0) + 0.0 * LON[None, None, :]
    else:
        return np.full((HOURS.size, ny, nx), 0.25)
    f = np.broadcast_to(f, (nz, ny, nx)).copy()
    f[-1, :, LON < 265.0] = np.nan              # below the sea floor west of 95°W
    return np.broadcast_to(f, (HOURS.size, nz, ny, nx)).copy()


def _write(tmp: Path, codes=("t3z", "s3z", "u3z", "v3z", "ssh")) -> list[Path]:
    paths = []
    for code in codes:
        name, std, units, scale, offset, is3d = _VARS[code]
        dims = ("time", "depth", "lat", "lon") if is3d else ("time", "lat", "lon")
        coords = {"time": ("time", HOURS, {"units": "hours since 2000-01-01 00:00:00", "calendar": "standard"}),
                  "lat": LAT, "lon": ("lon", LON, {"modulo": "360 degrees"})}
        if is3d:
            coords["depth"] = ("depth", DEPTH, {"positive": "down", "units": "m"})
        ds = xr.Dataset({name: (dims, _field(code), {"standard_name": std, "units": units})},
                        coords=coords)
        ds["tau"] = ("time", HOURS - HOURS[0], {"units": "hours since analysis"})
        enc = {name: {"dtype": "int16", "scale_factor": scale, "add_offset": offset, "_FillValue": -30000}}
        path = tmp / f"US058GCOM-OPSnce.espc-d-031-hycom_fcst_glby008_2026052512_t0000_{code}.nc"
        ds.to_netcdf(path, format="NETCDF3_CLASSIC", encoding=enc)
        paths.append(path)
    return paths


BB = BBox(lon_min=-97.0, lon_max=-92.0, lat_min=17.0, lat_max=22.0)
T0, T1 = T_REF, datetime(2026, 5, 25, 18)


# ===========================================================================
# Adapter
# ===========================================================================

class TestHYCOMZSource:

    @pytest.mark.parametrize("name", ["espc", "gofs"])
    def test_registered(self, tmp_path, name):
        assert isinstance(get_source(name, paths=_write(tmp_path)), HYCOMZSource)

    def test_canonical_ocean(self, tmp_path):
        ds = HYCOMZSource(_write(tmp_path)).load(T0, T1, BB)
        assert set(ds.data_vars) == {"temp", "salt", "u", "v", "ssh"}
        assert ds["temp"].dims == ("time", "depth", "lat", "lon")
        np.testing.assert_array_equal(ds.depth.values, DEPTH)
        assert ds.time.values[0] == np.datetime64("2026-05-25T12", "ns")

    def test_packed_values_decoded(self, tmp_path):
        ds = HYCOMZSource(_write(tmp_path), temperature="potential").load(T0, T1, BB)
        np.testing.assert_allclose(ds.salt.isel(time=0, lat=0, lon=-1).values,
                                   _field("s3z")[0, :, 0, -1], atol=6e-4)
        np.testing.assert_allclose(ds.ssh, 0.25, atol=6e-4)

    def test_in_situ_converted_to_potential_temperature(self, tmp_path):
        # TEOS-10: θ = pt0_from_t(SA(SP, p, λ, φ), t, p), p from depth and latitude
        ds = HYCOMZSource(_write(tmp_path)).load(T0, T1, BB)
        col = dict(time=0, lat=0, lon=-1)
        lat, lon = float(ds.lat[0]), float(ds.lon[-1])
        t_raw = ds.temp.isel(**col)  # converted
        t_insitu = _field("t3z")[0, :, 0, -1]
        sp = _field("s3z")[0, :, 0, -1]
        p = gsw.p_from_z(-DEPTH, lat)
        expected = gsw.pt0_from_t(gsw.SA_from_SP(sp, p, lon, lat), t_insitu, p)
        np.testing.assert_allclose(t_raw.values, expected, atol=2e-3)
        assert t_raw.values[-1] < t_insitu[-1] - 0.2          # deep water: θ < t

    def test_potential_temperature_passthrough(self, tmp_path):
        ds = HYCOMZSource(_write(tmp_path), temperature="potential").load(T0, T1, BB)
        np.testing.assert_allclose(ds.temp.isel(time=0, lat=0, lon=-1).values,
                                   _field("t3z")[0, :, 0, -1], atol=6e-4)

    def test_records_temperature_kind(self, tmp_path):
        ds = HYCOMZSource(_write(tmp_path)).load(T0, T1, BB)
        assert "in situ" in ds.temp.attrs["comment"]

    def test_below_bottom_is_nan(self, tmp_path):
        ds = HYCOMZSource(_write(tmp_path)).load(T0, T1, BB)
        west = ds.lon.values < -95.0
        assert west.any()
        assert np.isnan(ds.temp.isel(time=0, depth=-1).values[:, west]).all()

    def test_longitudes_converted_to_minus180_180(self, tmp_path):
        ds = HYCOMZSource(_write(tmp_path)).load(T0, T1, BB)
        assert float(ds.lon.min()) <= BB.lon_min and float(ds.lon.max()) >= BB.lon_max
        assert float(ds.lon.max()) < 0.0

    def test_in_situ_requires_salinity(self, tmp_path):
        with pytest.raises(ValueError, match="salinity"):
            HYCOMZSource(_write(tmp_path, codes=("t3z",))).load(T0, T1, BB)

    def test_invalid_temperature_kind(self, tmp_path):
        with pytest.raises(ValueError, match="temperature"):
            HYCOMZSource(_write(tmp_path), temperature="conservative")

    def test_no_files_rejected(self):
        with pytest.raises(ValueError, match="paths"):
            HYCOMZSource([])


# ===========================================================================
# Vertical interpolation to model levels
# ===========================================================================

def _zgrid(dz: np.ndarray) -> xr.Dataset:
    return create_z_grid(len(dz), dz)


def _profile_da(values: np.ndarray, depth=DEPTH) -> xr.DataArray:
    return xr.DataArray(values[:, None, None] * np.ones((1, 2, 3)),
                        dims=("depth", "j_t", "i_t"), coords={"depth": depth})


class TestToModelLevels:

    def test_linear_profile_exact(self):
        vg = _zgrid(np.array([5.0, 10.0, 20.0, 50.0, 100.0]))
        out = to_model_levels(xr.Dataset({"t": _profile_da(30.0 - 0.01 * DEPTH)}), vg)
        np.testing.assert_allclose(out["t"].isel(j_t=0, i_t=0).values, 30.0 + 0.01 * vg.z_t.values)
        assert out["t"].dims == ("k", "j_t", "i_t")

    def test_above_shallowest_source_level_constant(self):
        depth = np.array([2.0, 10.0, 50.0])
        vg = _zgrid(np.array([1.0, 1.0, 10.0]))      # z_t = −0.5, −1.5, …
        out = to_model_levels(xr.Dataset({"t": _profile_da(np.array([25.0, 24.0, 20.0]), depth)}), vg)
        np.testing.assert_allclose(out["t"].isel(j_t=0, i_t=0).values[:2], 25.0)

    def test_below_bottom_filled_with_deepest_valid_value(self):
        vals = 30.0 - 0.01 * DEPTH
        vals[-2:] = np.nan                                   # sea floor between 1000 and 2000 m
        vg = _zgrid(np.full(30, 100.0))              # down to 3000 m
        out = to_model_levels(xr.Dataset({"t": _profile_da(vals)}), vg)
        col = out["t"].isel(j_t=0, i_t=0).values
        assert not np.isnan(col).any()
        np.testing.assert_allclose(col[-1], 30.0 - 0.01 * 1000.0)

    def test_two_dimensional_fields_untouched(self):
        ds = xr.Dataset({"t": _profile_da(30.0 - 0.01 * DEPTH),
                         "ssh": (("j_t", "i_t"), np.full((2, 3), 0.1))})
        out = to_model_levels(ds, _zgrid(np.array([5.0, 10.0])))
        np.testing.assert_allclose(out["ssh"], 0.1)
        assert "depth" not in out.dims
