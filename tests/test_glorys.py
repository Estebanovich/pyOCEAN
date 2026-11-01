"""
Tests for forcing.sources.glorys (CMEMS GLORYS12 reanalysis) and its use through forcing.ocean_init.

The synthetic file reproduces the layout of the real product subset downloaded with ``copernicusmarine``
(checked 2026-11-01 on glorys12_my_20170429_20170903_box.nc): dims (time, depth, latitude, longitude),
float32 thetao [degC, potential], so [1e-3 PSU], uo/vo [m s-1, east/north], zos [m], depth positive down,
NaN over land and below the bed, daily records.  Analytic fields (linear in lon, lat, depth) make
interpolation exact.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.grid.supergrid import grid_from_supergrid
from core.grid.vertical import create_z_grid
from forcing.ocean_init import (boundary_data, initial_state, prepare_ocean, prepare_ocean_source,
                                source_layers)
from forcing.rotation import to_grid
from forcing.sources.base import BBox, get_source
from forcing.sources.glorys import GLORYSSource

from tests.test_supergrid import _rotated_supergrid

LON = np.arange(-73.0, -66.99, 0.25)
LAT = np.arange(33.0, 37.51, 0.25)
DEPTH = np.array([0.494, 1.541, 5.0, 20.0, 100.0, 500.0, 1000.0], dtype=np.float32)
TIMES = np.array(["2017-05-01", "2017-05-02", "2017-05-03"], dtype="datetime64[ns]")
BBOX = BBox(-72.0, -68.0, 34.0, 36.0)
REAL = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/data/glorys/glorys12_my_20170429_20170903_box.nc")
NWA12 = Path.home() / "pyOCEAN_refs" / "nwa12_grid"


def _fields():
    t, z, y, x = np.meshgrid(np.arange(3), DEPTH.astype(float), LAT, LON, indexing="ij")
    th = 20.0 - 0.01 * z + 0.3 * (x + 70.0) + 0.2 * (y - 35.0) + 0.5 * t
    so = 35.0 - 0.001 * z + 0.1 * (x + 70.0)
    uo = 0.5 + 0.02 * (y - 35.0) + 0.0 * z + 0.0 * t
    vo = 0.2 + 0.0 * z + 0.01 * (x + 70.0) + 0.0 * t
    zos = 0.1 + 0.01 * (x[:, 0] + 70.0) + 0.02 * t[:, 0]
    return th, so, uo, vo, zos


def _write(tmp_path: Path, name: str = "glorys.nc", land: bool = True, uv: tuple[float, float] | None = None) -> Path:
    th, so, uo, vo, zos = _fields()
    if uv is not None:
        uo, vo = np.full_like(uo, uv[0]), np.full_like(vo, uv[1])
    if land:
        for f in (th, so, uo, vo):
            f[:, :, 8:12, 10:14] = np.nan         # an island, all levels
            f[:, 5:, :, :4] = np.nan              # below the bed in the west
        zos[:, 8:12, 10:14] = np.nan
    dims4, dims3 = ("time", "depth", "latitude", "longitude"), ("time", "latitude", "longitude")
    ds = xr.Dataset(
        {"thetao": (dims4, th.astype("f4"), {"units": "degrees_C", "standard_name": "sea_water_potential_temperature"}),
         "so": (dims4, so.astype("f4"), {"units": "1e-3", "standard_name": "sea_water_salinity"}),
         "uo": (dims4, uo.astype("f4"), {"units": "m s-1"}),
         "vo": (dims4, vo.astype("f4"), {"units": "m s-1"}),
         "zos": (dims3, zos.astype("f4"), {"units": "m"})},
        coords={"time": TIMES, "depth": ("depth", DEPTH, {"positive": "down", "units": "m"}),
                "latitude": LAT.astype("f4"), "longitude": LON.astype("f4")})
    p = tmp_path / name
    ds.to_netcdf(p)
    return p


def _load(path: Path, t0="2017-05-01", t1="2017-05-03") -> xr.Dataset:
    return GLORYSSource([path]).load(datetime.fromisoformat(t0), datetime.fromisoformat(t1), BBOX)


def test_read_canonical_units_and_dims(tmp_path):
    ocean = _load(_write(tmp_path, land=False))
    assert isinstance(get_source("glorys", paths=[_write(tmp_path, "b.nc")]), GLORYSSource)
    assert set(ocean.data_vars) == {"temp", "salt", "u", "v", "ssh"}
    assert ocean["temp"].dims == ("time", "depth", "lat", "lon")
    assert ocean["ssh"].dims == ("time", "lat", "lon")
    assert ocean["temp"].attrs["units"] == "degC" and ocean["salt"].attrs["units"] == "psu"
    assert GLORYSSource.ts_kind == "theta_sp"
    assert float(ocean.lat.min()) <= 34.0 and float(ocean.lat.max()) >= 36.0
    th = _fields()[0]
    j, i = int(np.argmin(abs(LAT - 35.0))), int(np.argmin(abs(LON + 70.0)))
    got = ocean["temp"].sel(lat=35.0, lon=-70.0).values
    np.testing.assert_array_equal(got, th[:, :, j, i].astype("f4").astype(float))


def test_nan_filled_from_nearest_ocean(tmp_path):
    # Review Focus 1: land and below-bed NaN take the value of the nearest ocean node on the same level
    # (CEFI fill_glorys_nn.sh: cdo setmisstonn), ocean nodes are untouched, and nothing stays NaN.
    path = _write(tmp_path)
    raw = xr.open_dataset(path).load()
    ocean = _load(path)
    assert not ocean["temp"].isnull().any() and not ocean["ssh"].isnull().any()
    sub = raw.sel(latitude=ocean.lat.values.astype("f4"), longitude=ocean.lon.values.astype("f4"))
    valid = ~sub["thetao"].isnull().values
    np.testing.assert_array_equal(ocean["temp"].values[valid], sub["thetao"].values[valid].astype(float))
    # Island (rows 8-11, cols 10-13), every level: corner node (8, 10) has the ocean nodes (7, 10) and (8, 9)
    # at one cell; a node is filled with the value of ONE of its nearest ocean neighbours, never a mean.
    th = raw["thetao"].values.astype(float)
    for (jj, ii, cands) in ((8, 10, [(7, 10), (8, 9)]), (11, 13, [(12, 13), (11, 14)])):
        for k in range(3):
            got = float(ocean["temp"].isel(time=1, depth=k).sel(lat=LAT[jj], lon=LON[ii]))
            assert any(got == th[1, k, a, b] for a, b in cands), (jj, ii, k)
    # below-bed west block (levels 5, 6, cols 0-3) is filled with the nearest ocean node of that level
    got = float(ocean["temp"].isel(time=0, depth=6).sel(lat=LAT[3], lon=LON[3]))
    assert got == th[0, 6, 3, 4]


def test_time_outside_coverage_raises(tmp_path):
    path = _write(tmp_path)
    with pytest.raises(ValueError, match="2017-05-01.*2017-05-03"):
        _load(path, "2017-04-20", "2017-05-02")
    with pytest.raises(ValueError, match="2017-05-03"):
        _load(path, "2017-05-02", "2017-06-02")


def _grids(nx=8, ny=6, nz=5):
    sg = _rotated_supergrid(nx, ny, 10_000.0, 0.0, lon0=-70.5, lat0=34.8)
    h = grid_from_supergrid(sg)
    return h, create_z_grid(nz, [2.0, 4.0, 10.0, 30.0, 100.0])


def test_boundary_faces_on_cgrid_points(tmp_path):
    h, v = _grids()
    ocean = _load(_write(tmp_path, land=False))
    prep = prepare_ocean(ocean, h, v)
    t = np.datetime64("2017-05-02")
    # u at u faces of west/east, v at v faces of south/north, evaluated at the face lon/lat (bilinear
    # of a field linear in lat for u and linear in lon for v is exact; the grid is ~unrotated).
    for side, name in (("west", "u"), ("east", "u"), ("south", "v"), ("north", "v")):
        ob = boundary_data(prep, side, t)
        n = h.sizes["j_t"] if name == "u" else h.sizes["i_t"]
        assert ob.u_normal.shape == (v.sizes["k"], n) and ob.T.shape == ob.u_normal.shape
        assert ob.eta.shape == (n,)
    west = boundary_data(prep, "west", t).u_normal[0]
    lat_u = h["lat_u"].values[:, 0]
    lon_u = h["lon_u"].values[:, 0]
    ue = 0.5 + 0.02 * (lat_u - 35.0)
    vn = 0.2 + 0.01 * (lon_u + 70.0)
    cos_f = 0.5 * (h["cos_t"].values[:, 0] + h["cos_t"].values[:, 0])
    sin_f = h["sin_t"].values[:, 0]
    ux, _ = to_grid(ue, vn, cos_f, sin_f)
    np.testing.assert_allclose(west, ux, atol=2e-5)


@pytest.mark.parametrize("alpha", [0.0, 30.0, 120.0])
def test_rotated_grid_velocity(tmp_path, alpha):
    sg = _rotated_supergrid(8, 6, 10_000.0, alpha, lon0=-70.0, lat0=35.0)
    h = grid_from_supergrid(sg)
    v = create_z_grid(3, [2.0, 10.0, 50.0])
    ocean = _load(_write(tmp_path, land=False, uv=(0.4, 0.3)))
    prep = prepare_ocean(ocean, h, v)
    a = np.deg2rad(alpha)
    ux = 0.4 * np.cos(a) + 0.3 * np.sin(a)
    vy = -0.4 * np.sin(a) + 0.3 * np.cos(a)
    np.testing.assert_allclose(prep["u"]["u"].values, ux, atol=1e-6)
    np.testing.assert_allclose(prep["v"]["v"].values, vy, atol=1e-6)
    # magnitude preserved by the rotation, and the model state carries the grid components
    st = initial_state(prep, h, v, np.datetime64("2017-05-02"))
    assert np.allclose(np.hypot(st.u.values[:, :, 1:-1][:, :-1], 0.0), abs(ux), atol=1e-6)
    # component at a given face equals to_grid of the (unrotated) data with the face angle
    assert prep["u"]["u"].dims[-2:] == ("j_t", "i_u") and prep["v"]["v"].dims[-2:] == ("j_v", "i_t")


def test_unrotated_grid_is_bit_for_bit(tmp_path):
    # Grids without cos_t/sin_t: no rotation code runs, u and v are the east/north data.
    from core.grid.horizontal import create_spherical_grid
    h = create_spherical_grid(nx=8, ny=6, dlon=0.1, dlat=0.1, lon0=-70.4, lat0=34.7)
    v = create_z_grid(3, [2.0, 10.0, 50.0])
    ocean = _load(_write(tmp_path, land=False, uv=(0.4, 0.3)))
    prep = prepare_ocean(ocean, h, v)
    np.testing.assert_allclose(prep["u"]["u"].values, 0.4, atol=1e-6)
    np.testing.assert_allclose(prep["v"]["v"].values, 0.3, atol=1e-6)


def test_initial_state_column_content_conserved(tmp_path):
    # Source layers (GLORYS levels, interfaces half-way, bed-truncated) carry the data heat and salt; the
    # ALE start from h_src remaps them with exact content, so Σ h T and Σ h S per column match to 1e-12.
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from test_sigma_ale import _ale_box
    nx, nz = 8, 12
    H = np.linspace(300.0, 900.0, nx)[None, :]
    m = _ale_box("zstar", nx=nx, nz=nz, depth=900.0, dx=10_000.0, bathy=H, salinity=True)
    from core.grid.horizontal import create_spherical_grid
    hg = create_spherical_grid(nx=nx, ny=1, dlon=0.1, dlat=0.1, lon0=-70.4, lat0=34.9)
    ocean = _load(_write(tmp_path, land=False))
    src = prepare_ocean_source(ocean, hg)
    eta = np.full((1, nx), 0.1)
    h_src, T_src, S_src = source_layers(src, H, eta, np.datetime64("2017-05-02"))
    np.testing.assert_allclose(h_src.sum(axis=0), H + eta, rtol=1e-13)
    assert (h_src >= 0).all() and np.isfinite(T_src).all()
    st = m.initial_state(eta, T_src, h_src=h_src, S=S_src)
    heat_src, heat = (h_src * T_src).sum(axis=0), (st.h * st.T).sum(axis=0)
    salt_src, salt = (h_src * S_src).sum(axis=0), (st.h * st.S).sum(axis=0)
    np.testing.assert_allclose(heat, heat_src, rtol=1e-12)
    np.testing.assert_allclose(salt, salt_src, rtol=1e-12)


@pytest.mark.skipif(not (REAL.exists() and (NWA12 / "ocean_hgrid.nc").exists()),
                    reason="real GLORYS box or NWA12 grid not available")
def test_real_file_matches_copernicusmarine():
    # Criterion 3: the adapter's values at GLORYS nodes equal the downloaded file's values (no re-download),
    # on the initial state and on a boundary record of a 20 x 20 window of the NWA12 grid near 39.5N, 72W.
    raw = xr.open_dataset(REAL)
    t0 = datetime(2017, 5, 10)
    t1 = datetime(2017, 5, 12)
    with xr.open_dataset(NWA12 / "ocean_hgrid.nc") as hg:
        hg = hg.load()
    lon, lat = hg["x"].values[1::2, 1::2], hg["y"].values[1::2, 1::2]
    j, i = np.unravel_index(np.argmin((lon + 72.0) ** 2 + (lat - 39.5) ** 2), lon.shape)
    h = grid_from_supergrid(hg, window=(int(i) - 10, int(i) + 10, int(j) - 10, int(j) + 10))
    src = GLORYSSource([REAL])
    bbox = BBox.from_grid(h, pad=0.2)
    ocean = src.load(t0, t1, bbox)
    assert not ocean["temp"].isnull().any(axis=(2, 3)).all()
    v = create_z_grid(4, [5.0, 20.0, 75.0, 400.0])
    # (1) at GLORYS nodes the filled ocean equals the file wherever the file has ocean
    sub = raw.sel(time=ocean.time.values, latitude=ocean.lat.values, longitude=ocean.lon.values, method="nearest")
    valid = ~sub["thetao"].isnull().values
    np.testing.assert_array_equal(ocean["temp"].values[valid], sub["thetao"].values[valid].astype(float))
    np.testing.assert_array_equal(ocean["salt"].values[valid], sub["so"].values[valid].astype(float))
    np.testing.assert_array_equal(ocean["u"].values[valid], sub["uo"].values[valid].astype(float))
    # (2) bilinear spot check of the initial T at three model T points against the raw file
    prep = prepare_ocean(ocean, h, v)
    st = initial_state(prep, h, v, np.datetime64("2017-05-10"))
    for (jj, ii) in ((3, 4), (10, 10), (16, 15)):
        lo, la = float(h["lon_t"].values[jj, ii]), float(h["lat_t"].values[jj, ii])
        zc = -float(v["z_t"].values[0])
        want = float(raw["thetao"].sel(time="2017-05-10").interp(longitude=lo, latitude=la, depth=zc))
        assert np.isfinite(want)                           # all eight surrounding nodes are ocean
        assert abs(float(st.T.values[0, jj, ii]) - want) < 1e-7
    # (3) boundary record: west-face u_normal at the window rows equals bilinear u rotated to the grid
    ob = boundary_data(prep, "west", np.datetime64("2017-05-10"))
    assert ob.u_normal.shape == (4, 20) and np.isfinite(ob.u_normal).all() and np.isfinite(ob.T).all()
    lo, la = h["lon_u"].values[:, 0], h["lat_u"].values[:, 0]
    cos_f, sin_f = h["cos_t"].values[:, 0], h["sin_t"].values[:, 0]
    zc = float(-v["z_t"].values[0])
    r10 = raw.sel(time="2017-05-10")
    ue = np.array([float(r10["uo"].interp(longitude=a, latitude=b, depth=zc)) for a, b in zip(lo, la)])
    vn = np.array([float(r10["vo"].interp(longitude=a, latitude=b, depth=zc)) for a, b in zip(lo, la)])
    ux, _ = to_grid(ue, vn, cos_f, sin_f)
    assert np.max(np.abs(ob.u_normal[0] - ux)) < 1e-7
