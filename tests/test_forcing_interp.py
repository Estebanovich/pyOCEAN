"""
Tests for forcing.interp — source grid → model C-grid, and time interpolation.

Shared by every forcing adapter, so the tests use analytic fields whose
interpolated values are known exactly (bilinear interpolation of a linear
field is exact).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_spherical_grid
from forcing.interp import fill_land, interp_time, to_model_grid


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

def _linear_source(
    lon: np.ndarray | None = None,
    lat: np.ndarray | None = None,
    a: float = 2.0, b: float = -3.0, c: float = 10.0,
) -> xr.Dataset:
    """phi(lon, lat) = a·lon + b·lat + c on a regular 0.5° source grid."""
    if lon is None:
        lon = np.arange(-100.0, -79.5, 0.5)
    if lat is None:
        lat = np.arange(15.0, 32.5, 0.5)
    LON, LAT = np.meshgrid(lon, lat)
    time = np.array(["2015-04-01T00", "2015-04-01T06"], dtype="datetime64[ns]")
    phi = np.stack([a * LON + b * LAT + c, a * LON + b * LAT + c + 1.0])
    return xr.Dataset(
        {"phi": (("time", "lat", "lon"), phi, {"units": "1"})},
        coords={"time": time, "lat": lat, "lon": lon},
    )


@pytest.fixture
def grid() -> xr.Dataset:
    # 10 × 8 cells of 1° starting at 95°W, 20°N
    return create_spherical_grid(nx=10, ny=8, dlon=1.0, dlat=1.0, lon0=-95.0, lat0=20.0)


def _expected(grid: xr.Dataset, point: str, a=2.0, b=-3.0, c=10.0) -> np.ndarray:
    return a * grid[f"lon_{point}"].values + b * grid[f"lat_{point}"].values + c


# ---------------------------------------------------------------------------
# to_model_grid
# ---------------------------------------------------------------------------

class TestToModelGrid:

    @pytest.mark.parametrize("point,dims", [
        ("t", ("j_t", "i_t")),
        ("u", ("j_t", "i_u")),
        ("v", ("j_v", "i_t")),
    ])
    def test_output_dims_match_c_grid_point(self, grid, point, dims):
        out = to_model_grid(_linear_source(), grid, point=point)
        assert out["phi"].dims == ("time",) + dims

    @pytest.mark.parametrize("point", ["t", "u", "v"])
    def test_linear_field_is_reproduced_exactly(self, grid, point):
        out = to_model_grid(_linear_source(), grid, point=point)
        np.testing.assert_allclose(out["phi"].isel(time=0).values, _expected(grid, point), atol=1e-10)

    def test_time_axis_preserved(self, grid):
        src = _linear_source()
        out = to_model_grid(src, grid)
        np.testing.assert_array_equal(out.time.values, src.time.values)
        np.testing.assert_allclose(
            (out["phi"].isel(time=1) - out["phi"].isel(time=0)).values, 1.0, atol=1e-10
        )

    def test_attributes_preserved(self, grid):
        out = to_model_grid(_linear_source(), grid)
        assert out["phi"].attrs["units"] == "1"

    def test_descending_latitude_source(self, grid):
        src = _linear_source().isel(lat=slice(None, None, -1))
        out = to_model_grid(src, grid)
        np.testing.assert_allclose(out["phi"].isel(time=0).values, _expected(grid, "t"), atol=1e-10)

    def test_source_longitudes_0_360(self, grid):
        # ERA5/GFS/TPXO often use 0–360°E; model grid uses −180–180°E.
        lon360 = np.arange(260.0, 280.5, 0.5)
        src = _linear_source(lon=lon360, a=0.0)   # a=0: field independent of the lon convention
        out = to_model_grid(src, grid)
        np.testing.assert_allclose(out["phi"].isel(time=0).values, _expected(grid, "t", a=0.0), atol=1e-10)

    def test_extra_vertical_dimension_carried_through(self, grid):
        src = _linear_source()
        src = xr.concat([src, src + 5.0], dim=xr.DataArray([0.5, 10.0], dims="depth", name="depth"))
        src = src.transpose("time", "depth", "lat", "lon")
        out = to_model_grid(src, grid)
        assert out["phi"].dims == ("time", "depth", "j_t", "i_t")
        np.testing.assert_allclose(
            out["phi"].isel(time=0, depth=1).values, _expected(grid, "t") + 5.0, atol=1e-10
        )

    def test_domain_outside_source_raises(self, grid):
        src = _linear_source(lon=np.arange(-90.0, -79.5, 0.5))   # does not cover 95°W
        with pytest.raises(ValueError, match="does not cover"):
            to_model_grid(src, grid)

    @pytest.mark.xfail(strict=True, reason="known pre-existing failure (CLAUDE.md, also on 6f6a2ec): "
                       "to_model_grid does not raise for an invalid point")
    def test_invalid_point_raises(self, grid):
        with pytest.raises(ValueError, match="point"):
            to_model_grid(_linear_source(), grid, point="q")


# ---------------------------------------------------------------------------
# fill_land (sea-over-land extrapolation)
# ---------------------------------------------------------------------------

class TestFillLand:

    def test_no_nans_is_identity(self):
        da = _linear_source()["phi"]
        xr.testing.assert_identical(fill_land(da), da)

    def test_fills_all_nans_in_each_slice(self):
        da = _linear_source()["phi"].copy()
        da[:, :5, :5] = np.nan
        out = fill_land(da)
        assert not np.isnan(out.values).any()

    def test_ocean_values_untouched(self):
        da = _linear_source()["phi"].copy()
        da[:, :5, :5] = np.nan
        out = fill_land(da)
        valid = ~np.isnan(da.values)
        np.testing.assert_array_equal(out.values[valid], da.values[valid])

    def test_single_hole_gets_neighbour_mean(self):
        da = xr.DataArray(
            np.array([[1.0, 2.0, 3.0], [4.0, np.nan, 6.0], [7.0, 8.0, 9.0]]),
            dims=("lat", "lon"),
        )
        out = fill_land(da)
        assert out.values[1, 1] == pytest.approx((2.0 + 4.0 + 6.0 + 8.0) / 4.0)

    def test_constant_field_stays_constant(self):
        da = xr.DataArray(np.full((6, 7), 3.5), dims=("lat", "lon"))
        da[2:5, 1:4] = np.nan
        np.testing.assert_allclose(fill_land(da).values, 3.5)

    def test_all_nan_slice_left_as_nan(self):
        # A deep level that is entirely below the sea floor: nothing to extrapolate from
        da = xr.DataArray(np.ones((2, 3, 4)), dims=("depth", "lat", "lon"))
        da[1] = np.nan
        out = fill_land(da)
        assert np.isnan(out.values[1]).all()
        np.testing.assert_array_equal(out.values[0], 1.0)

    def test_input_not_modified(self):
        da = _linear_source()["phi"].copy()
        da[:, 0, 0] = np.nan
        fill_land(da)
        assert np.isnan(da.values[:, 0, 0]).all()


# ---------------------------------------------------------------------------
# interp_time
# ---------------------------------------------------------------------------

class TestInterpTime:

    def test_exact_record_returned(self):
        src = _linear_source()
        out = interp_time(src, np.datetime64("2015-04-01T06"))
        xr.testing.assert_allclose(out["phi"], src["phi"].isel(time=1, drop=True))

    def test_linear_midpoint(self):
        src = _linear_source()
        out = interp_time(src, np.datetime64("2015-04-01T03"))
        np.testing.assert_allclose(
            out["phi"].values, src["phi"].isel(time=0).values + 0.5, atol=1e-12
        )

    def test_no_time_dimension_in_result(self):
        out = interp_time(_linear_source(), np.datetime64("2015-04-01T01"))
        assert "time" not in out["phi"].dims

    def test_outside_range_raises(self):
        with pytest.raises(ValueError, match="outside"):
            interp_time(_linear_source(), np.datetime64("2015-04-01T07"))
