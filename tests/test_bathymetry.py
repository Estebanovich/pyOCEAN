"""
Tests for core.grid.bathymetry — Phase 2 bathymetry preprocessing.

Coverage
--------
  load_gebco          — 6 tests
  regrid_to_model     — 7 tests
  smooth_bathymetry   — 11 tests  (hann + shapiro + n_skip)
  fill_isolated_cells — 5 tests
  apply_min_depth     — 4 tests
  build_bathymetry    — 5 tests
  Total               — 38 tests
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.grid.bathymetry import (
    apply_min_depth,
    build_bathymetry,
    fill_isolated_cells,
    load_gebco,
    regrid_to_model,
    smooth_bathymetry,
)
from core.grid.horizontal import create_spherical_grid, create_cartesian_grid


# ---------------------------------------------------------------------------
# Fixtures — synthetic GEBCO-style NetCDF files
# ---------------------------------------------------------------------------

def _make_gebco_nc(
    path: Path,
    lon_range: tuple[float, float] = (-100.0, -70.0),
    lat_range: tuple[float, float] = (15.0, 35.0),
    resolution: float = 1.0,
    elev_value: float = -1000.0,
    var_name: str = "elevation",
) -> Path:
    """Write a minimal GEBCO-style NetCDF4 file for testing."""
    lons = np.arange(lon_range[0], lon_range[1] + resolution, resolution)
    lats = np.arange(lat_range[0], lat_range[1] + resolution, resolution)
    elev = np.full((len(lats), len(lons)), elev_value, dtype=np.float32)

    ds = xr.Dataset(
        {var_name: (["lat", "lon"], elev, {"units": "m"})},
        coords={"lon": lons, "lat": lats},
    )
    ds.to_netcdf(path)
    return path


@pytest.fixture
def gebco_file(tmp_path):
    """Synthetic GEBCO file: all-ocean (-1000 m), 'elevation' variable."""
    return _make_gebco_nc(tmp_path / "gebco.nc")


@pytest.fixture
def gebco_z_file(tmp_path):
    """Synthetic GEBCO file using 'z' variable name."""
    return _make_gebco_nc(tmp_path / "gebco_z.nc", var_name="z")


@pytest.fixture
def gebco_mixed_file(tmp_path):
    """Synthetic file with land (positive) and ocean (negative) cells."""
    lons = np.arange(-100.0, -69.0, 1.0)
    lats = np.arange(15.0, 36.0, 1.0)
    # West half ocean (-500 m), east half land (+100 m)
    elev = np.full((len(lats), len(lons)), -500.0, dtype=np.float32)
    mid = len(lons) // 2
    elev[:, mid:] = 100.0
    ds = xr.Dataset(
        {"elevation": (["lat", "lon"], elev, {"units": "m"})},
        coords={"lon": lons, "lat": lats},
    )
    path = tmp_path / "gebco_mixed.nc"
    ds.to_netcdf(path)
    return path


@pytest.fixture
def spherical_grid():
    """Small spherical grid centred on the Gulf of Mexico."""
    return create_spherical_grid(nx=10, ny=8, dlon=1.0, dlat=1.0,
                                 lon0=-97.0, lat0=18.0)


# ---------------------------------------------------------------------------
# load_gebco
# ---------------------------------------------------------------------------

class TestLoadGebco:

    def test_returns_dataarray(self, gebco_file):
        raw = load_gebco(gebco_file, (-95.0, -75.0), (18.0, 30.0))
        assert isinstance(raw, xr.DataArray)

    def test_clips_to_bounds_with_margin(self, gebco_file):
        raw = load_gebco(gebco_file, (-90.0, -80.0), (20.0, 28.0))
        # With 0.5° margin: lon ∈ [-90.5, -79.5], lat ∈ [19.5, 28.5]
        assert float(raw.lon.min()) <= -90.0
        assert float(raw.lon.max()) >= -80.0
        assert float(raw.lat.min()) <= 20.0
        assert float(raw.lat.max()) >= 28.0

    def test_reads_elevation_variable(self, gebco_file):
        raw = load_gebco(gebco_file, (-95.0, -75.0), (18.0, 30.0))
        # All-ocean file: all values should be -1000
        assert np.allclose(raw.values, -1000.0)

    def test_reads_z_variable_fallback(self, gebco_z_file):
        raw = load_gebco(gebco_z_file, (-95.0, -75.0), (18.0, 30.0))
        assert np.allclose(raw.values, -1000.0)

    def test_file_not_found_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_gebco(tmp_path / "nonexistent.nc", (-95.0, -75.0), (18.0, 30.0))

    def test_missing_variable_raises(self, tmp_path):
        # File with neither 'elevation' nor 'z'
        ds = xr.Dataset(
            {"temperature": (["lat", "lon"], np.zeros((5, 5)))},
            coords={"lon": np.linspace(-90, -80, 5),
                    "lat": np.linspace(20, 28, 5)},
        )
        bad_path = tmp_path / "bad.nc"
        ds.to_netcdf(bad_path)
        with pytest.raises(KeyError):
            load_gebco(bad_path, (-90.0, -80.0), (20.0, 28.0))


# ---------------------------------------------------------------------------
# regrid_to_model
# ---------------------------------------------------------------------------

class TestRegridToModel:

    def _flat_raw(self, elev: float = -1000.0) -> xr.DataArray:
        """1°-resolution DataArray, uniform elevation."""
        lons = np.arange(-100.0, -69.0, 1.0)
        lats = np.arange(15.0, 36.0, 1.0)
        data = np.full((len(lats), len(lons)), elev, dtype=float)
        return xr.DataArray(data, coords={"lat": lats, "lon": lons},
                            dims=["lat", "lon"])

    def test_flat_ocean_returns_uniform_depth(self):
        raw = self._flat_raw(-1000.0)
        lon_t = np.array([-95.0, -90.0, -85.0])
        lat_t = np.array([20.0, 25.0])
        depth = regrid_to_model(raw, lon_t, lat_t)
        assert depth.shape == (2, 3)
        assert np.allclose(depth, 1000.0)

    def test_ocean_elevation_becomes_positive_depth(self):
        raw = self._flat_raw(-500.0)
        depth = regrid_to_model(raw, np.array([-90.0]), np.array([22.0]))
        assert np.allclose(depth, 500.0)

    def test_land_elevation_becomes_zero(self):
        raw = self._flat_raw(200.0)   # all land
        depth = regrid_to_model(raw, np.array([-90.0]), np.array([22.0]))
        assert np.allclose(depth, 0.0)

    def test_out_of_bounds_points_are_land(self):
        raw = self._flat_raw(-1000.0)
        # lon = 0° is far outside the raw domain (-100 to -69)
        depth = regrid_to_model(raw, np.array([0.0]), np.array([25.0]))
        assert np.allclose(depth, 0.0)

    def test_2d_input_arrays(self):
        raw = self._flat_raw(-800.0)
        LON, LAT = np.meshgrid(
            np.array([-95.0, -90.0, -85.0]),
            np.array([20.0, 25.0]),
        )
        depth = regrid_to_model(raw, LON, LAT)
        assert depth.shape == (2, 3)
        assert np.allclose(depth, 800.0)

    def test_nearest_method(self):
        raw = self._flat_raw(-300.0)
        depth = regrid_to_model(raw,
                                np.array([-90.0]),
                                np.array([22.0]),
                                method="nearest")
        assert np.allclose(depth, 300.0)

    def test_mixed_land_ocean(self, gebco_mixed_file):
        raw = load_gebco(gebco_mixed_file, (-100.0, -69.0), (15.0, 35.0))
        # Far west point (ocean) and far east point (land)
        depth_ocean = regrid_to_model(raw,
                                      np.array([-97.0]),
                                      np.array([22.0]))
        depth_land  = regrid_to_model(raw,
                                      np.array([-73.0]),
                                      np.array([22.0]))
        assert depth_ocean[0, 0] > 0
        assert depth_land[0, 0] == 0.0


# ---------------------------------------------------------------------------
# smooth_bathymetry
# ---------------------------------------------------------------------------

class TestSmoothBathymetry:

    def test_zero_passes_returns_copy(self):
        d = np.array([[0.0, 100.0, 200.0],
                      [0.0, 150.0, 250.0]])
        out = smooth_bathymetry(d, n_passes=0)
        assert np.array_equal(out, d)
        assert out is not d    # it is a copy

    def test_flat_field_unchanged_hann(self):
        d = np.full((5, 5), 1000.0)
        out = smooth_bathymetry(d, n_passes=3, kernel="hann")
        assert np.allclose(out, 1000.0)

    def test_flat_field_unchanged_shapiro(self):
        d = np.full((5, 5), 1000.0)
        out = smooth_bathymetry(d, n_passes=3, kernel="shapiro")
        assert np.allclose(out, 1000.0)

    def test_land_cells_stay_zero(self):
        # Land border around a flat ocean interior
        d = np.zeros((5, 5))
        d[1:4, 1:4] = 500.0   # ocean interior
        out = smooth_bathymetry(d, n_passes=2)
        # Land border must remain 0
        assert np.all(out[0, :] == 0.0)
        assert np.all(out[:, 0] == 0.0)

    def test_gradient_decreases_after_one_pass_hann(self):
        d = np.zeros((5, 5))
        d[1:4, 1:4] = np.array([[100.0, 500.0, 100.0],
                                 [100.0, 500.0, 100.0],
                                 [100.0, 500.0, 100.0]])
        out = smooth_bathymetry(d, n_passes=1, kernel="hann")
        grad_before = abs(d[2, 2] - d[2, 1])
        grad_after  = abs(out[2, 2] - out[2, 1])
        assert grad_after < grad_before

    def test_multiple_passes_smoother_than_one(self):
        d = np.zeros((7, 7))
        d[1:6, 1:6] = np.random.RandomState(42).uniform(100, 1000, (5, 5))
        out1 = smooth_bathymetry(d, n_passes=1)
        out5 = smooth_bathymetry(d, n_passes=5)
        var1 = np.var(out1[1:6, 1:6])
        var5 = np.var(out5[1:6, 1:6])
        assert var5 < var1

    def test_hann_smoother_than_shapiro(self):
        # 9-point kernel should produce a smoother result than 5-point
        d = np.zeros((7, 7))
        d[1:6, 1:6] = np.random.RandomState(0).uniform(100, 2000, (5, 5))
        out_hann    = smooth_bathymetry(d, n_passes=3, kernel="hann")
        out_shapiro = smooth_bathymetry(d, n_passes=3, kernel="shapiro")
        # Hann spreads information via diagonals → lower variance
        assert np.var(out_hann[1:6, 1:6]) <= np.var(out_shapiro[1:6, 1:6])

    def test_invalid_kernel_raises(self):
        d = np.ones((4, 4)) * 100.0
        with pytest.raises(ValueError):
            smooth_bathymetry(d, kernel="gaussian")

    def test_n_skip_preserves_coastal_cells(self):
        # Ocean strip adjacent to land walls: n_skip=1 → those cells unchanged
        d = np.zeros((5, 5))
        d[1:4, 1:4] = np.array([[100.0, 500.0, 100.0],
                                 [100.0, 500.0, 100.0],
                                 [100.0, 500.0, 100.0]])
        # Cells at d[1:4,1] and d[1:4,3] are 1 cell from land → should not change
        out = smooth_bathymetry(d, n_passes=5, n_skip=1)
        assert np.allclose(out[1:4, 1], 100.0)
        assert np.allclose(out[1:4, 3], 100.0)

    def test_n_skip_zero_smooths_all_ocean(self):
        # n_skip=0 → all ocean cells are eligible
        d = np.zeros((5, 5))
        d[1:4, 1:4] = np.array([[100.0, 900.0, 100.0],
                                 [100.0, 900.0, 100.0],
                                 [100.0, 900.0, 100.0]])
        out = smooth_bathymetry(d, n_passes=1, n_skip=0, kernel="shapiro")
        # The 100.0 coastal cells should change (they have a 900.0 neighbor)
        assert not np.allclose(out[1:4, 1], 100.0)

    def test_output_shape_preserved(self):
        d = np.ones((12, 15)) * 200.0
        out = smooth_bathymetry(d, n_passes=2)
        assert out.shape == (12, 15)


# ---------------------------------------------------------------------------
# fill_isolated_cells
# ---------------------------------------------------------------------------

class TestFillIsolatedCells:

    def test_isolated_single_cell_removed(self):
        d = np.zeros((5, 5))
        d[2, 2] = 1000.0   # surrounded by land on all 4 sides
        out = fill_isolated_cells(d)
        assert out[2, 2] == 0.0

    def test_connected_ocean_unchanged(self):
        d = np.zeros((5, 7))
        d[1:4, 1:6] = 500.0    # 3×5 ocean block — each cell has ≥ 2 ocean nbrs
        out = fill_isolated_cells(d)
        assert np.all(out[2, 2:5] == 500.0)   # interior rows definitely kept

    def test_narrow_one_cell_inlet_removed(self):
        # Single-cell peninsular connector: cell (2,3) has only 1 ocean nbr
        d = np.zeros((5, 5))
        d[2, 1:4] = 300.0   # row of ocean 3 cells wide
        d[2, 0] = 0.0       # west wall
        d[2, 4] = 0.0       # east wall
        # Force a single-connected cell at top: (1,2) only touches d[2,2]
        d[1, 2] = 300.0
        out = fill_isolated_cells(out := d.copy())
        # (1,2) has only 1 ocean neighbour → should be removed
        assert out[1, 2] == 0.0

    def test_land_unaffected(self):
        d = np.zeros((4, 4))
        out = fill_isolated_cells(d)
        assert np.all(out == 0.0)

    def test_iterates_until_convergence(self):
        # Chain: 3 cells in a line, the ends are isolated after removing tails
        d = np.zeros((3, 5))
        d[1, 1] = 100.0   # cell with only 1 ocean nbr (right)
        d[1, 2] = 100.0   # centre
        d[1, 3] = 100.0   # cell with only 1 ocean nbr (left)
        # All three form a line with end-cells having exactly 1 ocean nbr
        # → both ends removed → centre isolated → centre removed
        out = fill_isolated_cells(d)
        assert np.all(out == 0.0)


# ---------------------------------------------------------------------------
# apply_min_depth
# ---------------------------------------------------------------------------

class TestApplyMinDepth:

    def test_shallow_ocean_raised(self):
        d = np.array([[5.0, 8.0, 15.0]])
        out = apply_min_depth(d, min_depth=10.0)
        assert out[0, 0] == 10.0
        assert out[0, 1] == 10.0
        assert out[0, 2] == 15.0    # already deep enough

    def test_deep_ocean_unchanged(self):
        d = np.array([[1000.0, 2000.0, 3500.0]])
        out = apply_min_depth(d, min_depth=10.0)
        assert np.array_equal(out, d)

    def test_land_cells_unchanged(self):
        d = np.array([[0.0, 0.0, 5.0]])
        out = apply_min_depth(d, min_depth=10.0)
        assert out[0, 0] == 0.0
        assert out[0, 1] == 0.0
        assert out[0, 2] == 10.0   # ocean cell raised

    def test_exactly_min_depth_unchanged(self):
        d = np.array([[10.0]])
        out = apply_min_depth(d, min_depth=10.0)
        assert out[0, 0] == 10.0


# ---------------------------------------------------------------------------
# build_bathymetry (integration)
# ---------------------------------------------------------------------------

class TestBuildBathymetry:

    def test_output_shape_matches_grid(self, gebco_file, spherical_grid):
        ny = spherical_grid.attrs["ny"]
        nx = spherical_grid.attrs["nx"]
        depth = build_bathymetry(spherical_grid, gebco_file,
                                 min_depth=10.0, n_smooth_passes=0)
        assert depth.shape == (ny, nx)

    def test_all_values_nonnegative(self, gebco_file, spherical_grid):
        depth = build_bathymetry(spherical_grid, gebco_file,
                                 min_depth=10.0, n_smooth_passes=1)
        assert np.all(depth >= 0.0)

    def test_min_depth_honored(self, gebco_file, spherical_grid):
        min_d = 50.0
        depth = build_bathymetry(spherical_grid, gebco_file,
                                 min_depth=min_d, n_smooth_passes=0)
        ocean = depth > 0
        if ocean.any():
            assert np.all(depth[ocean] >= min_d)

    def test_no_smoothing_passes(self, gebco_file, spherical_grid):
        depth = build_bathymetry(spherical_grid, gebco_file,
                                 min_depth=10.0, n_smooth_passes=0)
        assert np.all(depth >= 0.0)

    def test_cartesian_grid_raises_key_error(self, gebco_file):
        cart = create_cartesian_grid(nx=10, ny=8, dx=10e3, dy=10e3)
        with pytest.raises(KeyError):
            build_bathymetry(cart, gebco_file)
