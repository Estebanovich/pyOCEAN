"""
Tests for forcing.sources.tpxo (TPXO atlas reader) and forcing.tides.boundary_tides.

Synthetic files follow the TPXO10-atlas-v2 NetCDF layout as read by the
HYCOM-IOA tooling (~/HYCOM/sal_tpxo10/tpxo2amph.py):
  h_<c>_tpxo10_atlas_30_v2.nc : hRe, hIm [mm], dims (nx, ny) = (lon, lat)
  u_<c>_tpxo10_atlas_30_v2.nc : uRe, uIm, vRe, vIm transports [cm² s⁻¹]
                                (names checked on the real TPXO10-atlas-v2 files)
  grid_tpxo10atlas_v2.nc      : lon_z, lat_z, lon_u, lat_u, lon_v, lat_v, hz
with longitudes 0–360 °E on an Arakawa C grid (u half a cell west, v half a
cell south of the z nodes).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.dynamics.free_surface import open_boundary_hfac
from core.grid.horizontal import create_spherical_grid
from core.grid.vertical import compute_hfac, create_z_grid
from forcing.sources.tpxo import read_tpxo
from forcing.tides import boundary_tide_constants, boundary_tides, boundary_tides_at, predict

D = 1.0 / 3.0
LON_Z = np.arange(250.0, 270.0 + 1e-9, D)
LAT_Z = np.arange(10.0, 25.0 + 1e-9, D)


def _lin(lon, lat, a, b, c):
    return a * (lon - 260.0) + b * (lat - 17.0) + c


def _write_tpxo(tmp: Path, constituents=("m2", "k1")) -> tuple[list[Path], list[Path], Path]:
    LZ, TZ = np.meshgrid(LON_Z, LAT_Z, indexing="ij")             # (nx, ny) storage
    LU, TU = LZ - D / 2, TZ
    LV, TV = LZ, TZ - D / 2
    grid = xr.Dataset({
        "lon_z": ("nx", LON_Z), "lat_z": ("ny", LAT_Z),
        "lon_u": ("nx", LON_Z - D / 2), "lat_u": ("ny", LAT_Z),
        "lon_v": ("nx", LON_Z), "lat_v": ("ny", LAT_Z - D / 2),
        "hz": (("nx", "ny"), np.full(LZ.shape, 3000.0), {"units": "meter"}),
    })
    gpath = tmp / "grid_tpxo10atlas_v2.nc"
    grid.to_netcdf(gpath)
    hp, up = [], []
    for n, c in enumerate(constituents):
        s = n + 1.0
        h = xr.Dataset({
            "hRe": (("nx", "ny"), np.round(1000 * _lin(LZ, TZ, 0.01 * s, 0.02, 0.4 * s)).astype("int32"),
                    {"units": "millimeter"}),
            "hIm": (("nx", "ny"), np.round(1000 * _lin(LZ, TZ, -0.01, 0.01 * s, -0.1 * s)).astype("int32"),
                    {"units": "millimeter"}),
        })
        hp.append(tmp / f"h_{c}_tpxo10_atlas_30_v2.nc")
        h.to_netcdf(hp[-1])
        u = xr.Dataset({
            "uRe": (("nx", "ny"), 1e4 * _lin(LU, TU, 1.0, 0.5, 30.0 * s), {"units": "centimeter^2/sec"}),
            "uIm": (("nx", "ny"), 1e4 * _lin(LU, TU, 0.2, -0.3, 5.0), {"units": "centimeter^2/sec"}),
            "vRe": (("nx", "ny"), 1e4 * _lin(LV, TV, -0.4, 0.1, 10.0), {"units": "centimeter^2/sec"}),
            "vIm": (("nx", "ny"), 1e4 * _lin(LV, TV, 0.3, 0.2, -8.0 * s), {"units": "centimeter^2/sec"}),
        })
        up.append(tmp / f"u_{c}_tpxo10_atlas_30_v2.nc")
        u.to_netcdf(up[-1])
    return hp, up, gpath


# ===========================================================================

class TestReadTPXO:

    def test_constituents_and_units(self, tmp_path):
        hp, up, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, up, g, lon_range=(-105.0, -95.0), lat_range=(14.0, 20.0))
        assert list(tc.constituent.values) == ["m2", "k1"]
        lon, lat = -100.0, 17.0                                     # = 260 °E, on a node
        h = tc.h.sel(lon=lon, lat=lat, method="nearest")
        assert h.sel(constituent="m2").item() == pytest.approx(0.4 + 0.0j - 0.1j * 1, abs=1e-3)
        assert tc.h.attrs["units"] == "m" and tc.U.attrs["units"] == "m2 s-1"

    def test_longitudes_minus180_180_and_subset(self, tmp_path):
        hp, up, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, up, g, lon_range=(-105.0, -95.0), lat_range=(14.0, 20.0))
        assert float(tc.lon.min()) <= -105.0 and float(tc.lon.max()) >= -95.0
        assert float(tc.lon.max()) < 0.0
        assert tc.lon.size < LON_Z.size

    def test_transports_on_their_own_c_grid_points(self, tmp_path):
        hp, up, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, up, g, lon_range=(-105.0, -95.0), lat_range=(14.0, 20.0))
        # U is stored at lon_u = lon_z − Δ/2; value there equals the linear field
        lon_u = float(tc.U.lon_u.values[5])
        lat_u = float(tc.U.lat.values[4])
        expected = _lin(lon_u + 360.0, lat_u, 1.0, 0.5, 30.0) + 1j * _lin(lon_u + 360.0, lat_u, 0.2, -0.3, 5.0)
        assert tc.U.sel(constituent="m2").isel(lat=4, lon_u=5).item() == pytest.approx(expected, rel=1e-9)

    def test_elevation_only(self, tmp_path):
        hp, _, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, None, g, lon_range=(-105.0, -95.0), lat_range=(14.0, 20.0))
        assert "U" not in tc and "h" in tc

    def test_unknown_units_rejected(self, tmp_path):
        hp, up, g = _write_tpxo(tmp_path, constituents=("m2",))
        ds = xr.load_dataset(hp[0])
        ds["hRe"].attrs["units"] = "furlongs"
        ds.to_netcdf(tmp_path / "bad_h_m2.nc")
        with pytest.raises(ValueError, match="units"):
            read_tpxo([tmp_path / "bad_h_m2.nc"], None, g, lon_range=(-105, -95), lat_range=(14, 20))


class TestBoundaryTides:

    def _model(self):
        hg = create_spherical_grid(nx=8, ny=6, dlon=0.5, dlat=0.5, lon0=-102.0, lat0=15.0)
        vg = create_z_grid(1, [2000.0])
        hf = open_boundary_hfac(compute_hfac(vg, np.full((6, 8), 2000.0)), ["west", "south"])
        fu = np.zeros((6, 9), bool)
        fu[:, 0] = True
        fv = np.zeros((7, 8), bool)
        fv[0, :] = True
        return hg, vg, hf, fu, fv

    def test_elevation_at_faces_matches_prediction(self, tmp_path):
        hp, up, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, up, g, lon_range=(-104.0, -96.0), lat_range=(13.0, 20.0))
        hg, vg, hf, fu, fv = self._model()
        t = np.datetime64("2026-10-10T06:00")
        bt = boundary_tides(tc, hg, vg, hf, fu, fv, t)
        j = 2
        lon, lat = float(hg.lon_u.values[j, 0]) + 360.0, float(hg.lat_u.values[j, 0])
        hc = np.array([_lin(lon, lat, 0.01 * s, 0.02, 0.4 * s) + 1j * _lin(lon, lat, -0.01, 0.01 * s, -0.1 * s)
                       for s in (1.0, 2.0)])
        assert bt["eta_u"][j, 0] == pytest.approx(predict(np.array([t]), hc, ["m2", "k1"])[0], abs=2e-3)
        assert np.all(bt["eta_u"][:, 1:] == 0.0)

    def test_velocity_is_transport_over_model_depth(self, tmp_path):
        hp, up, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, up, g, lon_range=(-104.0, -96.0), lat_range=(13.0, 20.0))
        hg, vg, hf, fu, fv = self._model()
        t = np.datetime64("2026-10-10T06:00")
        bt = boundary_tides(tc, hg, vg, hf, fu, fv, t)
        i = 3
        lon, lat = float(hg.lon_v.values[0, i]) + 360.0, float(hg.lat_v.values[0, i])
        V = np.array([_lin(lon, lat, -0.4, 0.1, 10.0) + 1j * _lin(lon, lat, 0.3, 0.2, -8.0 * s) for s in (1.0, 2.0)])
        expected = predict(np.array([t]), V, ["m2", "k1"])[0] / 2000.0
        assert bt["vbar_v"][0, i] == pytest.approx(expected, rel=1e-6)

    def test_constituent_subset_and_ramp(self, tmp_path):
        hp, up, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, up, g, lon_range=(-104.0, -96.0), lat_range=(13.0, 20.0))
        hg, vg, hf, fu, fv = self._model()
        t = np.datetime64("2026-10-10T06:00")
        full = boundary_tides(tc, hg, vg, hf, fu, fv, t)
        half = boundary_tides(tc, hg, vg, hf, fu, fv, t, ramp=0.5)
        np.testing.assert_allclose(half["eta_u"], 0.5 * full["eta_u"])
        only_m2 = boundary_tides(tc, hg, vg, hf, fu, fv, t, constituents=["m2"])
        assert not np.allclose(only_m2["eta_u"], full["eta_u"])

    def test_precomputed_face_constants_identical(self, tmp_path):
        # Interpolate once, predict every step: same numbers as boundary_tides
        hp, up, g = _write_tpxo(tmp_path)
        tc = read_tpxo(hp, up, g, lon_range=(-104.0, -96.0), lat_range=(13.0, 20.0))
        hg, vg, hf, fu, fv = self._model()
        bc = boundary_tide_constants(tc, hg, vg, hf, fu, fv)
        for t, ramp in ((np.datetime64("2026-10-10T06:00"), 1.0), (np.datetime64("2026-11-01T17:30"), 0.3)):
            ref = boundary_tides(tc, hg, vg, hf, fu, fv, t, ramp=ramp)
            new = boundary_tides_at(bc, t, ramp=ramp)
            for k in ref:
                np.testing.assert_array_equal(new[k], ref[k])

    def test_legacy_uppercase_transport_names(self, tmp_path):
        # Older OTIS atlases use URe/UIm/VRe/VIm
        hp, up, g = _write_tpxo(tmp_path, constituents=("m2",))
        ds = xr.load_dataset(up[0]).rename({"uRe": "URe", "uIm": "UIm", "vRe": "VRe", "vIm": "VIm"})
        ds.to_netcdf(tmp_path / "u_m2_legacy.nc")
        a = read_tpxo(hp, up, g, lon_range=(-105, -95), lat_range=(14, 20))
        b = read_tpxo(hp, [tmp_path / "u_m2_legacy.nc"], g, lon_range=(-105, -95), lat_range=(14, 20))
        np.testing.assert_array_equal(a.U.values, b.U.values)


class TestCoastalInterpolation:
    """TPXO marks land with exactly 0 (``option_0 = "land"``): it must not dilute coastal values."""

    def test_land_nodes_excluded_from_bilinear_weights(self, tmp_path):
        from forcing.tides import _interp_complex
        hp, up, g = _write_tpxo(tmp_path, constituents=("m2",))
        ds = xr.load_dataset(hp[0])
        i_land = int(np.argmin(np.abs(LON_Z - 259.0)))           # land column at 259 °E
        ds["hRe"][i_land, :] = 0
        ds["hIm"][i_land, :] = 0
        ds.to_netcdf(tmp_path / "h_m2_coast.nc")
        tc = read_tpxo([tmp_path / "h_m2_coast.nc"], None, g, lon_range=(-105, -95), lat_range=(14, 20))
        assert np.isnan(tc.h.sel(lon=-101.0, lat=17.0, method="nearest").item())
        # midway between the land node (259 °E) and the ocean node east of it, on a node row:
        lon_mid = LON_Z[i_land] + D / 2 - 360.0
        lat_row = 17.0
        got = _interp_complex(tc.h, "lat", "lon", np.array([lat_row]), np.array([lon_mid]))[0, 0]
        ocean = tc.h.sel(lon=LON_Z[i_land + 1] - 360.0, lat=lat_row, method="nearest").item()
        assert got == pytest.approx(ocean, rel=1e-12)

    def test_point_surrounded_by_land_is_nan(self, tmp_path):
        from forcing.tides import _interp_complex
        hp, up, g = _write_tpxo(tmp_path, constituents=("m2",))
        ds = xr.load_dataset(hp[0])
        ds["hRe"][:] = 0
        ds["hIm"][:] = 0
        ds.to_netcdf(tmp_path / "h_m2_land.nc")
        tc = read_tpxo([tmp_path / "h_m2_land.nc"], None, g, lon_range=(-105, -95), lat_range=(14, 20))
        assert np.isnan(_interp_complex(tc.h, "lat", "lon", np.array([17.1]), np.array([-100.1]))[0, 0])
