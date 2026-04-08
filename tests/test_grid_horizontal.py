"""
Tests for core.grid.horizontal — Arakawa C-grid generation.

Validation strategy:
  1. Dimension sizes match the expected nx / ny geometry.
  2. C-grid staggering: U-points lie halfway between T-points in x;
     V-points lie halfway between T-points in y.
  3. Conservation: total T-cell area equals the analytical domain area Lx·Ly.
  4. Coriolis — f-plane: constant everywhere.
  5. Coriolis — beta-plane: linear gradient ∂f/∂y = β (finite-difference check).
  6. Coriolis — spherical: f = 2Ω·sin(φ) at equator, poles, and mid-latitudes.
  7. Spherical metric: dx_t ≈ R·cos(φ)·Δλ (compare against analytical formula).
  8. set_land_mask applies the mask correctly and raises on bad shape.
"""

import numpy as np
import pytest

from core.grid.horizontal import (
    OMEGA,
    R_EARTH,
    create_cartesian_grid,
    create_spherical_grid,
    set_land_mask,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY = 20, 10
DX, DY = 5_000.0, 5_000.0     # 5 km cells
F0 = 1e-4                      # s⁻¹ — typical mid-latitude f
BETA = 2e-11                   # m⁻¹ s⁻¹ — typical beta


@pytest.fixture
def cart_fplane() -> object:
    """Uniform Cartesian f-plane grid."""
    return create_cartesian_grid(NX, NY, DX, DY, f0=F0, beta=0.0)


@pytest.fixture
def cart_beta() -> object:
    """Uniform Cartesian beta-plane grid."""
    return create_cartesian_grid(NX, NY, DX, DY, f0=F0, beta=BETA)


@pytest.fixture
def sph_grid() -> object:
    """1° × 1° spherical grid, 20×10 cells starting at (0°E, 20°N)."""
    return create_spherical_grid(NX, NY, dlon=1.0, dlat=1.0, lon0=0.0, lat0=20.0)


# ---------------------------------------------------------------------------
# 1. Dimension sizes
# ---------------------------------------------------------------------------

class TestDimensions:
    def test_t_shape(self, cart_fplane):
        assert cart_fplane["x_t"].shape == (NY, NX)
        assert cart_fplane["y_t"].shape == (NY, NX)

    def test_u_shape(self, cart_fplane):
        # U-points: nx+1 faces in x, ny rows in y
        assert cart_fplane["x_u"].shape == (NY, NX + 1)

    def test_v_shape(self, cart_fplane):
        # V-points: nx columns in x, ny+1 faces in y
        assert cart_fplane["x_v"].shape == (NY + 1, NX)

    def test_q_shape(self, cart_fplane):
        assert cart_fplane["x_q"].shape == (NY + 1, NX + 1)

    def test_metric_shapes(self, cart_fplane):
        assert cart_fplane["dx_t"].shape  == (NY, NX)
        assert cart_fplane["area_u"].shape == (NY, NX + 1)
        assert cart_fplane["dy_v"].shape  == (NY + 1, NX)

    def test_f_shapes(self, cart_fplane):
        assert cart_fplane["f_t"].shape == (NY, NX)
        assert cart_fplane["f_q"].shape == (NY + 1, NX + 1)

    def test_dataset_attrs(self, cart_fplane):
        assert cart_fplane.attrs["nx"] == NX
        assert cart_fplane.attrs["ny"] == NY
        assert cart_fplane.attrs["grid_type"] == "cartesian"


# ---------------------------------------------------------------------------
# 2. C-grid staggering geometry
# ---------------------------------------------------------------------------

class TestStaggering:
    def test_u_midway_between_t_in_x(self, cart_fplane):
        """x_U[j, i] should be the average of x_T[j, i-1] and x_T[j, i]."""
        x_t = cart_fplane["x_t"].values      # (NY, NX)
        x_u = cart_fplane["x_u"].values      # (NY, NX+1)
        # Interior U-faces: i=1 … NX-1
        midpoints = 0.5 * (x_t[:, :-1] + x_t[:, 1:])   # (NY, NX-1)
        np.testing.assert_allclose(x_u[:, 1:-1], midpoints, rtol=1e-12)

    def test_v_midway_between_t_in_y(self, cart_fplane):
        """y_V[j, i] should be the average of y_T[j-1, i] and y_T[j, i]."""
        y_t = cart_fplane["y_t"].values      # (NY, NX)
        y_v = cart_fplane["y_v"].values      # (NY+1, NX)
        # Interior V-faces: j=1 … NY-1
        midpoints = 0.5 * (y_t[:-1, :] + y_t[1:, :])   # (NY-1, NX)
        np.testing.assert_allclose(y_v[1:-1, :], midpoints, rtol=1e-12)

    def test_t_centred_in_u(self, cart_fplane):
        """T-point x should be the average of its west and east U-faces."""
        x_t = cart_fplane["x_t"].values      # (NY, NX)
        x_u = cart_fplane["x_u"].values      # (NY, NX+1)
        np.testing.assert_allclose(
            x_t, 0.5 * (x_u[:, :-1] + x_u[:, 1:]), rtol=1e-12
        )

    def test_t_centred_in_v(self, cart_fplane):
        """T-point y should be the average of its south and north V-faces."""
        y_t = cart_fplane["y_t"].values      # (NY, NX)
        y_v = cart_fplane["y_v"].values      # (NY+1, NX)
        np.testing.assert_allclose(
            y_t, 0.5 * (y_v[:-1, :] + y_v[1:, :]), rtol=1e-12
        )


# ---------------------------------------------------------------------------
# 3. Area conservation
# ---------------------------------------------------------------------------

class TestAreaConservation:
    def test_total_t_area_equals_domain_area(self, cart_fplane):
        """Sum of all T-cell areas must equal Lx × Ly."""
        expected = NX * DX * NY * DY
        actual = float(cart_fplane["area_t"].values.sum())
        np.testing.assert_allclose(actual, expected, rtol=1e-12)

    def test_cartesian_area_uniform(self, cart_fplane):
        """All T-cell areas should be identical for a uniform Cartesian grid."""
        area = cart_fplane["area_t"].values
        np.testing.assert_allclose(area, DX * DY, rtol=1e-12)


# ---------------------------------------------------------------------------
# 4 & 5. Coriolis — Cartesian grids
# ---------------------------------------------------------------------------

class TestCoriolisCartesian:
    def test_fplane_constant(self, cart_fplane):
        """f-plane: Coriolis should be F0 everywhere."""
        np.testing.assert_allclose(cart_fplane["f_t"].values, F0, rtol=1e-12)
        np.testing.assert_allclose(cart_fplane["f_q"].values, F0, rtol=1e-12)

    def test_beta_plane_gradient(self, cart_beta):
        """Beta-plane: finite difference ∂f/∂y ≈ β within numerical precision."""
        f_t = cart_beta["f_t"].values        # (NY, NX)
        y_t = cart_beta["y_t"].values        # (NY, NX)
        # Use centred differences for interior rows
        df_dy = (f_t[2:, :] - f_t[:-2, :]) / (y_t[2:, :] - y_t[:-2, :])
        np.testing.assert_allclose(df_dy, BETA, rtol=1e-10)

    def test_beta_increases_northward(self, cart_beta):
        """f should increase monotonically from south to north."""
        f_col = cart_beta["f_t"].values[:, 0]
        assert np.all(np.diff(f_col) > 0)

    def test_f0_at_southern_boundary(self, cart_beta):
        """First T-row is at y = y0 + dy/2, so f[0] = f0 + β*(y0 + dy/2)."""
        y_first = 0.0 + DY / 2.0                       # x0=0, y0=0
        expected = F0 + BETA * y_first
        np.testing.assert_allclose(
            cart_beta["f_t"].values[0, 0], expected, rtol=1e-12
        )


# ---------------------------------------------------------------------------
# 6. Coriolis — Spherical grid
# ---------------------------------------------------------------------------

class TestCoriolisSpherical:
    def test_coriolis_formula(self, sph_grid):
        """f = 2Ω sin(φ) at all T-points."""
        lat = sph_grid["lat_t"].values
        f_t = sph_grid["f_t"].values
        expected = 2.0 * OMEGA * np.sin(np.deg2rad(lat))
        np.testing.assert_allclose(f_t, expected, rtol=1e-12)

    def test_coriolis_positive_nh(self, sph_grid):
        """All points are in the Northern Hemisphere → f > 0."""
        assert np.all(sph_grid["f_t"].values > 0)

    def test_coriolis_southern_hemisphere(self):
        """Points in SH have f < 0."""
        g = create_spherical_grid(10, 10, dlon=1.0, dlat=1.0, lon0=0.0, lat0=-30.0)
        assert np.all(g["f_t"].values < 0)

    def test_coriolis_equator_near_zero(self):
        """f ≈ 0 at the equator (lat0 chosen so T-points straddle equator)."""
        g = create_spherical_grid(4, 4, dlon=1.0, dlat=1.0, lon0=0.0, lat0=-2.0)
        # T-point at j=1: lat = -2 + 1.5 = -0.5°  → small but non-zero
        f_min = np.abs(g["f_t"].values).min()
        assert f_min < 2.0 * OMEGA * np.sin(np.deg2rad(1.0))


# ---------------------------------------------------------------------------
# 7. Spherical metric: dx_t ≈ R·cos(φ)·Δλ
# ---------------------------------------------------------------------------

class TestSphericalMetrics:
    def test_dx_t_formula(self, sph_grid):
        """dx_t should equal R·cos(φ_T)·Δλ (arc-length formula)."""
        lat = sph_grid["lat_t"].values
        dx_t = sph_grid["dx_t"].values
        expected = R_EARTH * np.cos(np.deg2rad(lat)) * np.deg2rad(1.0)
        np.testing.assert_allclose(dx_t, expected, rtol=1e-10)

    def test_dy_t_constant(self, sph_grid):
        """dy_t = R·Δφ is constant for a uniform latitude grid."""
        dy_t = sph_grid["dy_t"].values
        expected = R_EARTH * np.deg2rad(1.0)
        np.testing.assert_allclose(dy_t, expected, rtol=1e-10)

    def test_dx_decreases_poleward(self, sph_grid):
        """dx_t should decrease toward higher latitudes (cos shrinks)."""
        dx_col = sph_grid["dx_t"].values[:, 0]    # single longitude column
        assert np.all(np.diff(dx_col) < 0)

    def test_lon_lat_variables_present(self, sph_grid):
        """Spherical grid must carry lon_t / lat_t coordinate variables."""
        for var in ("lon_t", "lat_t", "lon_u", "lat_u", "lon_v", "lat_v"):
            assert var in sph_grid.data_vars, f"Missing variable: {var}"


# ---------------------------------------------------------------------------
# 8. set_land_mask
# ---------------------------------------------------------------------------

class TestLandMask:
    def test_mask_applied(self, cart_fplane):
        mask = np.zeros((NY, NX), dtype=np.int8)
        mask[2:5, 3:8] = 1                           # only a rectangle is ocean
        g = set_land_mask(cart_fplane, mask)
        np.testing.assert_array_equal(g["mask_t"].values, mask)

    def test_wrong_shape_raises(self, cart_fplane):
        bad_mask = np.ones((NY + 1, NX), dtype=np.int8)
        with pytest.raises(ValueError, match="mask shape"):
            set_land_mask(cart_fplane, bad_mask)

    def test_default_mask_all_ocean(self, cart_fplane):
        np.testing.assert_array_equal(
            cart_fplane["mask_t"].values, np.ones((NY, NX), dtype=np.int8)
        )
