"""Tests for forcing.rotation and the rotated-grid wind of AtmosphereProvider (regional chain, Task 4)."""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from forcing.atmosphere import AtmosphereProvider
from forcing.bulk import bulk_ncar
from forcing.rotation import to_earth, to_grid

ORIGIN = np.datetime64("2020-01-01T00:00:00", "ns")


def test_round_trip():
    rng = np.random.default_rng(0)
    a = rng.uniform(-np.pi, np.pi, (3, 4, 5))
    u, v = rng.normal(size=(2, 3, 4, 5))
    ue, vn = to_earth(*to_grid(u, v, np.cos(a), np.sin(a)), np.cos(a), np.sin(a))
    assert np.max(np.abs(ue - u)) < 1e-15 and np.max(np.abs(vn - v)) < 1e-15


def test_rotation_30deg():
    a = np.deg2rad(30.0)
    c, s = np.cos(a), np.sin(a)
    # an eastward unit vector points along (cos 30, -sin 30) in a grid rotated 30 deg CCW
    ux, vy = to_grid(np.array([1.0]), np.array([0.0]), c, s)
    assert np.allclose([ux[0], vy[0]], [np.sqrt(3) / 2, -0.5], atol=1e-15)
    # the grid-x unit vector points 30 deg north of east
    ue, vn = to_earth(np.array([1.0]), np.array([0.0]), c, s)
    assert np.allclose([ue[0], vn[0]], [np.sqrt(3) / 2, 0.5], atol=1e-15)


def _grid(angle: float | None) -> xr.Dataset:
    g = create_cartesian_grid(6, 5, 1000.0, 1000.0)
    g["lon_t"] = -70.0 + g["x_t"] / 1e5
    g["lat_t"] = 40.0 + g["y_t"] / 1e5
    if angle is not None:
        shape = g["lon_t"].shape
        g["cos_t"] = (g["lon_t"].dims, np.full(shape, np.cos(angle)))
        g["sin_t"] = (g["lon_t"].dims, np.full(shape, np.sin(angle)))
    return g


def _source(u: float, v: float) -> xr.Dataset:
    lat = np.arange(39.8, 40.41, 0.05)[::-1]
    lon = np.arange(-70.2, -69.79, 0.05)
    time = ORIGIN + (np.array([0.0, 3600.0]) * 1e9).astype("timedelta64[ns]")
    one = np.ones((2, lat.size, lon.size))
    f = {"u10": u, "v10": v, "t2m": 293.0, "q2m": 0.01, "msl": 101300.0, "sw_down": 300.0,
         "lw_down": 350.0, "precip": 0.0}
    return xr.Dataset({k: (("time", "lat", "lon"), x * one) for k, x in f.items()},
                      coords={"time": time, "lat": lat, "lon": lon})


def test_bulk_stress_rotated_grid_equals_unrotated():
    ang = np.deg2rad(37.0)
    c, s = np.cos(ang), np.sin(ang)
    uw, vw, ue, ve = 8.0, -3.0, 0.4, 0.2                      # earth wind and current
    src = _source(uw, vw)
    ref = AtmosphereProvider(src, _grid(None), ORIGIN)(1800.0)
    rot = AtmosphereProvider(src, _grid(ang), ORIGIN)(1800.0)
    assert np.allclose(rot["u10"], c * uw + s * vw, atol=1e-13)
    cur_x, cur_y = to_grid(np.full_like(ref["u10"], ue), np.full_like(ref["u10"], ve), c, s)

    def stress(atm, ux, vy):
        return bulk_ncar(atm["u10"], atm["v10"], atm["t2m"], atm["q2m"], 15.0, p=atm["msl"],
                         u_ocean=ux, v_ocean=vy)
    f0 = stress(ref, np.full_like(ref["u10"], ue), np.full_like(ref["u10"], ve))
    f1 = stress(rot, cur_x, cur_y)
    te, tn = to_earth(f1.tau_x, f1.tau_y, c, s)
    assert np.max(np.abs(te - f0.tau_x)) < 1e-12 and np.max(np.abs(tn - f0.tau_y)) < 1e-12
    assert np.max(np.abs(f1.q_sens - f0.q_sens)) < 1e-10


def test_no_angle_bit_for_bit():
    src = _source(8.0, -3.0)
    a = AtmosphereProvider(src, _grid(None), ORIGIN)(1234.0)
    p = AtmosphereProvider(src, _grid(None), ORIGIN)
    assert p._rot is None
    from forcing.interp import to_model_grid, fill_land
    rec = src.isel(time=0, drop=True).load()
    ref = to_model_grid(xr.Dataset({k: fill_land(rec[k]) for k in rec.data_vars}), _grid(None), point="t")
    assert np.array_equal(p(0.0)["u10"], np.ascontiguousarray(ref["u10"].values, dtype=np.float64))
    assert a["u10"].dtype == np.float64
