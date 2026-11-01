"""
Tests for core.grid.supergrid — C-grid metrics from a MOM6 supergrid.

Reference: docs/design/regional_refs/mom6_grid.md (MOM6 git 0e9a561,
MOM_grid_initialize.F90:170-305, MOM_shared_initialization.F90:571-642).

Validation strategy:
  1. A Cartesian supergrid (metres) gives create_cartesian_grid bit for bit.
  2. A regular lon/lat supergrid gives create_spherical_grid to 1e-12.
  3. A window = MOM6 run on the cut supergrid; equal to slicing the full
     result except the extrapolated outermost faces / corners.
  4. The rotation angle is MOM6's corner formula (not angle_dx).
  5. A skewed supergrid raises ValueError naming (j, i).
  6. The real NWA12 supergrid (skipped when absent).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import (
    R_EARTH,
    create_cartesian_grid,
    create_spherical_grid,
)
from core.grid.supergrid import grid_from_supergrid, supergrid_from_arrays

NWA12_DIR = Path.home() / "pyOCEAN_refs" / "nwa12_grid"


# ---------------------------------------------------------------------------
# Synthetic supergrid builders
# ---------------------------------------------------------------------------

def _cartesian_supergrid(nx: int, ny: int, d: float) -> xr.Dataset:
    """Uniform Cartesian supergrid in metres (half-segments d/2)."""
    h = d / 2.0
    X, Y = np.meshgrid(np.arange(2 * nx + 1) * h, np.arange(2 * ny + 1) * h)
    dx = np.full((2 * ny + 1, 2 * nx), h)
    dy = np.full((2 * ny, 2 * nx + 1), h)
    area = np.full((2 * ny, 2 * nx), h * h)
    return supergrid_from_arrays(X, Y, dx, dy, area, np.zeros_like(X), units="m")


def _spherical_supergrid(nx: int, ny: int, dlon: float, dlat: float,
                         lon0: float, lat0: float) -> xr.Dataset:
    """Regular lon/lat supergrid matching create_spherical_grid's metrics.

    dx half-segments are R cos(lat_node) Δλ/2 and dy half-segments R Δφ/2.
    create_spherical_grid's area is the product dx_t·dy_t, so each sub-cell
    here gets the area (R cos φ_T Δλ/2)(R Δφ/2) of its parent T row: the
    four-sub-cell sum then reproduces that product (a test construction,
    a real supergrid has exact spherical sub-cell areas).
    """
    lon = lon0 + np.arange(2 * nx + 1) * (dlon / 2.0)
    lat = lat0 + np.arange(2 * ny + 1) * (dlat / 2.0)
    X, Y = np.meshgrid(lon, lat)
    hx = R_EARTH * np.cos(np.deg2rad(lat)) * np.deg2rad(dlon) / 2.0  # per node row
    dx = np.repeat(hx[:, None], 2 * nx, axis=1)
    dy = np.full((2 * ny, 2 * nx + 1), R_EARTH * np.deg2rad(dlat) / 2.0)
    lat_parent_t = lat[1::2]                           # T row of each sub-row pair
    hx_t = R_EARTH * np.cos(np.deg2rad(lat_parent_t)) * np.deg2rad(dlon) / 2.0
    sub = hx_t * (R_EARTH * np.deg2rad(dlat) / 2.0)
    area = np.repeat(np.repeat(sub, 2)[:, None], 2 * nx, axis=1)
    return supergrid_from_arrays(X, Y, dx, dy, area, np.zeros_like(X))


def _haversine(lon1, lat1, lon2, lat2):
    p1, p2 = np.deg2rad(lat1), np.deg2rad(lat2)
    dl = np.deg2rad(lon2 - lon1)
    a = np.sin((p2 - p1) / 2) ** 2 + np.cos(p1) * np.cos(p2) * np.sin(dl / 2) ** 2
    return 2 * R_EARTH * np.arcsin(np.sqrt(a))


def _rotated_supergrid(nx: int, ny: int, d: float, alpha_deg: float,
                       lon0: float = -70.0, lat0: float = 35.0) -> xr.Dataset:
    """Supergrid whose grid-x points alpha_deg counter-clockwise from east."""
    a = np.deg2rad(alpha_deg)
    Xl, Yl = np.meshgrid(np.arange(2 * nx + 1) * d / 2, np.arange(2 * ny + 1) * d / 2)
    E = Xl * np.cos(a) - Yl * np.sin(a)
    N = Xl * np.sin(a) + Yl * np.cos(a)
    # Rotate in the Mercator plane (conformal), so the cells stay orthogonal.
    k = R_EARTH * np.cos(np.deg2rad(lat0))
    y0 = np.log(np.tan(np.pi / 4 + np.deg2rad(lat0) / 2))
    lon = lon0 + np.rad2deg(E / k)
    lat = np.rad2deg(2 * np.arctan(np.exp(y0 + N / k)) - np.pi / 2)
    dx = _haversine(lon[:, :-1], lat[:, :-1], lon[:, 1:], lat[:, 1:])
    dy = _haversine(lon[:-1], lat[:-1], lon[1:], lat[1:])
    area = 0.25 * (dx[:-1] + dx[1:]) * (dy[:, :-1] + dy[:, 1:])
    return supergrid_from_arrays(lon, lat, dx, dy, area, np.full_like(lon, alpha_deg))


def _mom6_angle(lon_q: np.ndarray, lat_q: np.ndarray, lon_t: np.ndarray) -> np.ndarray:
    """MOM6 initialize_grid_rotation_angle (bugs=False), written out
    independently (MOM_shared_initialization.F90:604-622, 629-642)."""
    nj, ni = lon_t.shape
    out = np.empty((nj, ni))
    pi_720 = np.arctan(1.0) / 180.0
    for j in range(nj):
        for i in range(ni):
            def lb(J, I):
                xc = lon_t[j, i]
                return np.mod(lon_q[J, I] - (xc - 180.0), 360.0) + (xc - 180.0)
            # lonB(m, n) = B(I+m-2, J+n-2): Fortran (I-1, J-1) = Python (j, i)
            l11, l21, l12, l22 = lb(j, i), lb(j, i + 1), lb(j + 1, i), lb(j + 1, i + 1)
            la = lat_q
            lon_scale = np.cos(pi_720 * ((la[j, i] + la[j + 1, i + 1]) + (la[j, i + 1] + la[j + 1, i])))
            out[j, i] = np.arctan2(lon_scale * ((l12 - l21) + (l22 - l11)),
                                   (la[j + 1, i] - la[j, i + 1]) + (la[j + 1, i + 1] - la[j, i]))
    return out


SHARED = ["x_t", "y_t", "x_u", "y_u", "x_v", "y_v", "x_q", "y_q",
          "dx_t", "dy_t", "area_t", "dx_u", "dy_u", "area_u",
          "dx_v", "dy_v", "area_v", "f_t", "f_q", "mask_t"]


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

def test_cartesian_supergrid_equals_create_cartesian_grid() -> None:
    nx, ny, d = 12, 9, 5_000.0
    g = grid_from_supergrid(_cartesian_supergrid(nx, ny, d))
    ref = create_cartesian_grid(nx, ny, d, d)
    assert g.attrs["grid_type"] == "mom6_supergrid"
    # f is excluded: a planar (metre) supergrid has no latitude, so f is NaN
    # by design (MOM6 assumes degrees, MOM_grid_initialize.F90:256-259).
    for name in SHARED:
        if name in ("f_t", "f_q"):
            continue
        assert g[name].dims == ref[name].dims, name
        assert np.array_equal(g[name].values, ref[name].values), name
    assert np.isnan(g["f_q"].values).all()
    for name in ("dx_q", "dy_q", "area_q"):
        assert g[name].shape == (ny + 1, nx + 1)
    assert np.array_equal(g["dx_q"].values, np.full((ny + 1, nx + 1), d))
    assert np.array_equal(g["area_q"].values, np.full((ny + 1, nx + 1), d * d))
    assert np.array_equal(g["cos_t"].values, np.ones((ny, nx)))
    assert np.array_equal(g["sin_t"].values, np.zeros((ny, nx)))


def test_spherical_supergrid_equals_create_spherical_grid() -> None:
    nx, ny, dlon, dlat, lon0, lat0 = 12, 9, 0.25, 0.2, -75.0, 30.0
    g = grid_from_supergrid(_spherical_supergrid(nx, ny, dlon, dlat, lon0, lat0))
    ref = create_spherical_grid(nx, ny, dlon, dlat, lon0, lat0)
    for name in SHARED + ["lon_t", "lat_t", "lon_u", "lat_u",
                          "lon_v", "lat_v", "lon_q", "lat_q"]:
        assert g[name].dims == ref[name].dims, name
        np.testing.assert_allclose(g[name].values, ref[name].values,
                                   rtol=1e-12, atol=0.0, err_msg=name)
    # Regular lon/lat grid: MOM6 angle is exactly 0 (lat differences along j).
    np.testing.assert_allclose(g["cos_t"].values, 1.0, rtol=0, atol=1e-15)
    np.testing.assert_allclose(g["sin_t"].values, 0.0, rtol=0, atol=1e-15)


def test_window_matches_full_then_slice() -> None:
    # Non-uniform grid (latitude-dependent dx, stretched dy) so that MOM6's
    # linear edge extrapolation differs from the parent grid's true values.
    sg = _spherical_supergrid(14, 11, 0.3, 0.25, -80.0, 20.0)
    rng = np.random.default_rng(3)
    dy = sg["dy"].values * (1.0 + 0.05 * rng.random(sg["dy"].shape[0]))[:, None]
    sg["dy"] = (sg["dy"].dims, dy)
    full = grid_from_supergrid(sg)
    i0, i1, j0, j1 = 3, 10, 2, 8
    win = grid_from_supergrid(sg, window=(i0, i1, j0, j1))
    assert win.sizes["i_t"] == i1 - i0 and win.sizes["j_t"] == j1 - j0

    sl = {"t": (slice(j0, j1), slice(i0, i1)), "u": (slice(j0, j1), slice(i0, i1 + 1)),
          "v": (slice(j0, j1 + 1), slice(i0, i1)), "q": (slice(j0, j1 + 1), slice(i0, i1 + 1))}
    # x_*, y_* are excluded: they are distances from the grid's own SW corner.
    for name, loc in [("dx_t", "t"), ("dy_t", "t"), ("area_t", "t"), ("f_t", "t"),
                      ("cos_t", "t"), ("sin_t", "t"), ("lon_t", "t"), ("lat_t", "t"),
                      ("dx_u", "u"), ("dy_u", "u"), ("area_u", "u"), ("lon_u", "u"),
                      ("dx_v", "v"), ("dy_v", "v"), ("area_v", "v"), ("lat_v", "v"),
                      ("dx_q", "q"), ("dy_q", "q"), ("area_q", "q"), ("f_q", "q")]:
        a = win[name].values
        b = full[name].values[sl[loc]]
        # The cut supergrid is a grid of its own: MOM6 extrapolates the
        # half-segments beyond its edge (MOM_grid_initialize.F90:1139-1169),
        # so dx_u at I = 0, ni, dy_v at J = 0, nj and the q metrics on the
        # window edge differ from the parent; interior points are equal.
        inner = (slice(None), slice(1, -1)) if loc == "u" else \
                (slice(1, -1), slice(None)) if loc == "v" else \
                (slice(1, -1), slice(1, -1)) if loc == "q" else (slice(None), slice(None))
        assert np.array_equal(a[inner], b[inner]), name
    # The edge rule is active in this test (otherwise it would not test it).
    assert not np.array_equal(win["dy_v"].values[0], full["dy_v"].values[j0, i0:i1])


def test_window_depth_full_or_window_shape() -> None:
    sg = _spherical_supergrid(8, 6, 0.3, 0.25, -80.0, 20.0)
    depth = np.arange(48, dtype=float).reshape(6, 8) - 5.0
    w = (2, 6, 1, 4)
    a = grid_from_supergrid(sg, depth=depth, window=w)
    b = grid_from_supergrid(sg, depth=depth[1:4, 2:6], window=w)
    assert np.array_equal(a["mask_t"].values, b["mask_t"].values)
    assert np.array_equal(a["mask_t"].values, (depth[1:4, 2:6] > 0).astype(np.int8))
    with pytest.raises(ValueError, match="depth"):
        grid_from_supergrid(sg, depth=np.ones((5, 5)))


def test_rotated_grid_angle() -> None:
    sg = _rotated_supergrid(8, 6, 5_000.0, 30.0)
    g = grid_from_supergrid(sg)
    lon_q, lat_q = g["lon_q"].values, g["lat_q"].values
    ang = _mom6_angle(lon_q, lat_q, g["lon_t"].values)   # MOM6 clockwise angle
    # cos_t = cos_rot, sin_t = -sin_rot (grid-x counter-clockwise from east).
    np.testing.assert_allclose(g["cos_t"].values, np.cos(ang), rtol=0, atol=1e-14)
    np.testing.assert_allclose(g["sin_t"].values, -np.sin(ang), rtol=0, atol=1e-14)
    got = np.rad2deg(np.arctan2(g["sin_t"].values, g["cos_t"].values))
    assert np.max(np.abs(got - 30.0)) < 0.05
    # u_east = cos u_x - sin v_y: a pure grid-x unit vector points 30° north of east.
    ue = g["cos_t"].values * 1.0
    vn = g["sin_t"].values * 1.0
    assert np.max(np.abs(np.rad2deg(np.arctan2(vn, ue)) - 30.0)) < 0.05


def test_non_orthogonal_raises() -> None:
    nx, ny, d = 10, 8, 5_000.0
    sg = _cartesian_supergrid(nx, ny, d)
    y = sg["y"].values.copy()
    # Move the u node between cells (3, 4) and (3, 5) north by 2 % of a cell:
    # grid-x of both cells tilts by ~0.02 rad while grid-y stays north.
    y[2 * 3 + 1, 2 * 5] += 0.02 * d
    sg["y"] = (sg["y"].dims, y)
    with pytest.raises(ValueError, match=r"\(j, i\) = \(3, [45]\)"):
        grid_from_supergrid(sg)
    # Only wet cells are checked when a depth is given.
    depth = np.ones((ny, nx))
    depth[3, 4:6] = 0.0
    g = grid_from_supergrid(sg, depth=depth)
    assert g["mask_t"].values[3, 4] == 0


def test_mask_from_depth() -> None:
    sg = _cartesian_supergrid(4, 3, 1000.0)
    depth = np.array([[0.0, 5.0, -1.0, 10.0]] * 3)
    g = grid_from_supergrid(sg, depth=depth)
    assert g["mask_t"].dtype == np.int8
    assert np.array_equal(g["mask_t"].values, (depth > 0).astype(np.int8))


@pytest.mark.skipif(not (NWA12_DIR / "ocean_hgrid.nc").exists(),
                    reason="NWA12 supergrid not available")
def test_nwa12_full_grid() -> None:
    with xr.open_dataset(NWA12_DIR / "ocean_hgrid.nc") as hg:
        hg = hg.load()
    with xr.open_dataset(NWA12_DIR / "ocean_topog.nc") as tp:
        depth = tp["depth"].values.astype(float)
    g = grid_from_supergrid(hg, depth=depth)
    assert g.sizes["j_t"] == 845 and g.sizes["i_t"] == 775
    assert g["dx_u"].shape == (845, 776) and g["dy_v"].shape == (846, 775)
    total = hg["area"].values.sum()
    assert abs(g["area_t"].values.sum() - total) <= 1e-12 * total
    assert 0 < g["mask_t"].values.sum() < 845 * 775
    # MOM6 angle vs the generator's angle_dx: angle_MOM6 ≈ -angle_dx (§6).
    ang = np.rad2deg(np.arctan2(g["sin_t"].values, g["cos_t"].values))
    adx = hg["angle_dx"].values[1::2, 1::2]
    assert np.max(np.abs(ang - adx)) < 0.5
