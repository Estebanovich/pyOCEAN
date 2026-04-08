"""
Tests for core.grid.vertical — z-level grid and partial-cell hFac.

Validation strategy:
  1. Interface z_f and layer-centre z_t positions are geometrically correct.
  2. sum(dz) = total depth (conservation).
  3. Uniform grid: all dz identical, all z_t equidistant.
  4. Stretched grid: sum(dz) = total_depth and dz increases monotonically.
  5. hFac = 1 for fully wet cells, 0 for fully dry, fraction for partial.
  6. Partial cells thinner than hfac_min are collapsed to 0.
  7. hFac_u = 0 at domain walls (closed-boundary no-slip).
  8. hFac_u = min of two adjacent T-cell hFac values (staggering).
  9. dz_t = dz * hfac_t (effective thickness definition).
  10. Volume consistency: sum(area_t * dz_t) = total wet volume.
  11. Input validation: bad dz, negative depth, wrong ndim raise ValueError.
"""

import numpy as np
import pytest

from core.grid.vertical import (
    compute_hfac,
    create_z_grid,
    stretched_z_grid,
    uniform_z_grid,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NZ = 10
DZ_UNIFORM = 50.0          # m  → total depth 500 m
TOTAL_DEPTH = 500.0        # m

NY, NX = 8, 6              # small domain for hFac tests


@pytest.fixture
def uniform_grid():
    return uniform_z_grid(NZ, DZ_UNIFORM)


@pytest.fixture
def flat_depth():
    """Constant depth = TOTAL_DEPTH (all T-cells fully wet)."""
    return np.full((NY, NX), TOTAL_DEPTH)


@pytest.fixture
def shallow_depth():
    """Depth = 75 m — first layer fully wet, second partially wet."""
    return np.full((NY, NX), 75.0)


@pytest.fixture
def mixed_depth():
    """Domain with a ridge: left half 500 m, right half 125 m."""
    depth = np.full((NY, NX), TOTAL_DEPTH)
    depth[:, NX // 2 :] = 125.0
    return depth


# ---------------------------------------------------------------------------
# 1-3. create_z_grid / uniform_z_grid geometry
# ---------------------------------------------------------------------------

class TestZGridGeometry:
    def test_z_f_surface_is_zero(self, uniform_grid):
        assert uniform_grid["z_f"].values[0] == 0.0

    def test_z_f_bottom_is_minus_total_depth(self, uniform_grid):
        np.testing.assert_allclose(
            uniform_grid["z_f"].values[-1], -TOTAL_DEPTH, rtol=1e-12
        )

    def test_z_f_monotonically_decreasing(self, uniform_grid):
        z_f = uniform_grid["z_f"].values
        assert np.all(np.diff(z_f) < 0)

    def test_z_t_is_midpoint_of_interfaces(self, uniform_grid):
        z_f = uniform_grid["z_f"].values
        z_t = uniform_grid["z_t"].values
        expected = 0.5 * (z_f[:-1] + z_f[1:])
        np.testing.assert_allclose(z_t, expected, rtol=1e-12)

    def test_dz_sum_equals_total_depth(self, uniform_grid):
        np.testing.assert_allclose(
            uniform_grid["dz"].values.sum(), TOTAL_DEPTH, rtol=1e-12
        )

    def test_uniform_dz_all_equal(self, uniform_grid):
        dz = uniform_grid["dz"].values
        np.testing.assert_allclose(dz, DZ_UNIFORM, rtol=1e-12)

    def test_shape_z_f(self, uniform_grid):
        assert uniform_grid["z_f"].shape == (NZ + 1,)

    def test_shape_z_t(self, uniform_grid):
        assert uniform_grid["z_t"].shape == (NZ,)

    def test_attrs_nz(self, uniform_grid):
        assert uniform_grid.attrs["nz"] == NZ

    def test_attrs_total_depth(self, uniform_grid):
        np.testing.assert_allclose(
            uniform_grid.attrs["total_depth"], TOTAL_DEPTH, rtol=1e-12
        )

    def test_explicit_dz_array(self):
        dz = np.array([5.0, 10.0, 20.0, 40.0, 80.0])
        g = create_z_grid(5, dz)
        np.testing.assert_allclose(g["dz"].values, dz, rtol=1e-12)
        np.testing.assert_allclose(g.attrs["total_depth"], dz.sum(), rtol=1e-12)


# ---------------------------------------------------------------------------
# 4. Stretched grid
# ---------------------------------------------------------------------------

class TestStretchedGrid:
    def test_sum_equals_total_depth(self):
        g = stretched_z_grid(nz=20, dz_surface=5.0, total_depth=1000.0)
        np.testing.assert_allclose(g["dz"].values.sum(), 1000.0, atol=1e-8)

    def test_dz_increases_downward(self):
        g = stretched_z_grid(nz=20, dz_surface=5.0, total_depth=1000.0)
        dz = g["dz"].values
        assert np.all(np.diff(dz) > 0)

    def test_surface_layer_thickness(self):
        g = stretched_z_grid(nz=20, dz_surface=5.0, total_depth=1000.0)
        np.testing.assert_allclose(g["dz"].values[0], 5.0, rtol=1e-3)

    def test_uniform_when_exact_fit(self):
        """If uniform spacing exactly fills total_depth, r = 1."""
        g = stretched_z_grid(nz=10, dz_surface=50.0, total_depth=500.0)
        np.testing.assert_allclose(g["dz"].values, 50.0, rtol=1e-10)

    def test_raises_if_uniform_overshoots(self):
        with pytest.raises(ValueError, match="already exceeds"):
            stretched_z_grid(nz=10, dz_surface=60.0, total_depth=500.0)


# ---------------------------------------------------------------------------
# 5. hFac = 1 for fully wet cells
# ---------------------------------------------------------------------------

class TestHFacFullyWet:
    def test_hfac_t_all_one_for_flat_deep_bottom(self, uniform_grid, flat_depth):
        topo = compute_hfac(uniform_grid, flat_depth)
        np.testing.assert_allclose(topo["hfac_t"].values, 1.0, rtol=1e-12)

    def test_hfac_u_interior_all_one_for_flat_deep(self, uniform_grid, flat_depth):
        topo = compute_hfac(uniform_grid, flat_depth)
        # Interior U-faces (i=1..NX-1): both adjacent T-cells are fully wet
        hfac_u_interior = topo["hfac_u"].values[:, :, 1:-1]
        np.testing.assert_allclose(hfac_u_interior, 1.0, rtol=1e-12)

    def test_dz_t_equals_dz_when_fully_wet(self, uniform_grid, flat_depth):
        topo = compute_hfac(uniform_grid, flat_depth)
        dz_1d = uniform_grid["dz"].values                      # (NZ,)
        expected = np.broadcast_to(dz_1d[:, None, None], topo["dz_t"].shape)
        np.testing.assert_allclose(topo["dz_t"].values, expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 6. Partial cells and hfac_min collapse
# ---------------------------------------------------------------------------

class TestHFacPartial:
    def test_bottom_layer_partial_for_shallow_depth(self, uniform_grid, shallow_depth):
        """depth=75 m with dz=50 m/layer: layer 0 fully wet, layer 1 half wet."""
        topo = compute_hfac(uniform_grid, shallow_depth)
        hfac_t = topo["hfac_t"].values    # (NZ, NY, NX)

        # Layer 0: 0 to -50 m, depth = 75 m → fully wet
        np.testing.assert_allclose(hfac_t[0], 1.0, rtol=1e-12)

        # Layer 1: -50 to -100 m, depth = 75 m
        # wet_thickness = min(50, max(0, 75 + (-50))) = min(50, 25) = 25 → hFac = 0.5
        np.testing.assert_allclose(hfac_t[1], 0.5, rtol=1e-12)

        # Layers 2 and below: fully dry
        np.testing.assert_allclose(hfac_t[2:], 0.0, rtol=1e-12)

    def test_hfac_zero_for_land(self, uniform_grid):
        depth = np.zeros((NY, NX))
        topo = compute_hfac(uniform_grid, depth)
        np.testing.assert_allclose(topo["hfac_t"].values, 0.0, rtol=1e-12)

    def test_hfac_min_collapses_thin_cells(self, uniform_grid):
        """A cell with hFac just below hfac_min should become 0."""
        # dz=50 m, hfac_min=0.1 → minimum wet thickness = 5 m
        # Set depth to 4 m (hFac = 4/50 = 0.08 < 0.1) → should be 0
        depth = np.full((NY, NX), 4.0)
        topo = compute_hfac(uniform_grid, depth, hfac_min=0.1)
        np.testing.assert_allclose(topo["hfac_t"].values, 0.0, rtol=1e-12)

    def test_hfac_above_min_preserved(self, uniform_grid):
        """A cell with hFac above hfac_min should be preserved."""
        # dz=50 m, hfac_min=0.1 → 5 m threshold
        # Set depth to 10 m (hFac = 10/50 = 0.2 > 0.1) → preserved
        depth = np.full((NY, NX), 10.0)
        topo = compute_hfac(uniform_grid, depth, hfac_min=0.1)
        expected = 10.0 / 50.0
        np.testing.assert_allclose(topo["hfac_t"].values[0], expected, rtol=1e-12)


# ---------------------------------------------------------------------------
# 7-8. hFac_u staggering and wall conditions
# ---------------------------------------------------------------------------

class TestHFacStaggering:
    def test_hfac_u_zero_at_west_wall(self, uniform_grid, flat_depth):
        topo = compute_hfac(uniform_grid, flat_depth)
        np.testing.assert_allclose(topo["hfac_u"].values[:, :, 0], 0.0, rtol=1e-12)

    def test_hfac_u_zero_at_east_wall(self, uniform_grid, flat_depth):
        topo = compute_hfac(uniform_grid, flat_depth)
        np.testing.assert_allclose(topo["hfac_u"].values[:, :, -1], 0.0, rtol=1e-12)

    def test_hfac_v_zero_at_south_wall(self, uniform_grid, flat_depth):
        topo = compute_hfac(uniform_grid, flat_depth)
        np.testing.assert_allclose(topo["hfac_v"].values[:, 0, :], 0.0, rtol=1e-12)

    def test_hfac_v_zero_at_north_wall(self, uniform_grid, flat_depth):
        topo = compute_hfac(uniform_grid, flat_depth)
        np.testing.assert_allclose(topo["hfac_v"].values[:, -1, :], 0.0, rtol=1e-12)

    def test_hfac_u_zero_at_ridge_face(self, uniform_grid, mixed_depth):
        """U-face between deep and shallow columns: lower layers should be 0."""
        topo = compute_hfac(uniform_grid, mixed_depth)
        # Ridge at i = NX//2: depth goes from 500 m to 125 m
        # depth_u at that face = min(500, 125) = 125 m
        # Layers 3+ (below 150 m) should have hfac_u = 0
        ridge_face = topo["hfac_u"].values[:, :, NX // 2]
        # Layer 0: 0–50m → wet (125 m > 50 m)
        np.testing.assert_allclose(ridge_face[0], 1.0, rtol=1e-12)
        # Layers 3+ (below 150 m) → dry (125 m < 150 m)
        np.testing.assert_allclose(ridge_face[3:], 0.0, rtol=1e-12)

    def test_depth_u_is_min_of_adjacent_t_depths(self, uniform_grid, mixed_depth):
        topo = compute_hfac(uniform_grid, mixed_depth)
        depth_t = topo["depth_t"].values            # (NY, NX)
        depth_u = topo["depth_u"].values            # (NY, NX+1)
        # Interior U-faces
        expected_interior = np.minimum(depth_t[:, :-1], depth_t[:, 1:])
        np.testing.assert_allclose(depth_u[:, 1:-1], expected_interior, rtol=1e-12)


# ---------------------------------------------------------------------------
# 9. dz_t = dz * hfac_t
# ---------------------------------------------------------------------------

class TestEffectiveThickness:
    def test_dz_t_definition(self, uniform_grid, mixed_depth):
        topo = compute_hfac(uniform_grid, mixed_depth)
        dz_1d = uniform_grid["dz"].values
        hfac_t = topo["hfac_t"].values
        expected = dz_1d[:, np.newaxis, np.newaxis] * hfac_t
        np.testing.assert_allclose(topo["dz_t"].values, expected, rtol=1e-12)

    def test_dz_t_zero_on_land(self, uniform_grid):
        depth = np.zeros((NY, NX))
        topo = compute_hfac(uniform_grid, depth)
        np.testing.assert_allclose(topo["dz_t"].values, 0.0, rtol=1e-12)


# ---------------------------------------------------------------------------
# 10. Volume consistency
# ---------------------------------------------------------------------------

class TestVolumeConsistency:
    def test_wet_volume_flat_bottom(self, uniform_grid, flat_depth):
        """Total wet volume = Lx * Ly * total_depth for a flat-bottom domain."""
        from core.grid.horizontal import create_cartesian_grid

        horiz = create_cartesian_grid(NX, NY, dx=1000.0, dy=1000.0)
        topo = compute_hfac(uniform_grid, flat_depth)

        area_t = horiz["area_t"].values                 # (NY, NX)
        dz_t   = topo["dz_t"].values                   # (NZ, NY, NX)
        vol    = (area_t[np.newaxis, :, :] * dz_t).sum()

        expected = NX * 1000.0 * NY * 1000.0 * TOTAL_DEPTH
        np.testing.assert_allclose(vol, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 11. Input validation
# ---------------------------------------------------------------------------

class TestInputValidation:
    def test_wrong_dz_length_raises(self):
        with pytest.raises(ValueError, match="length nz"):
            create_z_grid(5, [10.0, 20.0])   # len=2, nz=5

    def test_zero_dz_raises(self):
        with pytest.raises(ValueError, match="positive"):
            create_z_grid(3, [10.0, 0.0, 20.0])

    def test_negative_dz_raises(self):
        with pytest.raises(ValueError, match="positive"):
            create_z_grid(3, [10.0, -5.0, 20.0])

    def test_negative_depth_raises(self, uniform_grid):
        depth = np.full((NY, NX), -10.0)
        with pytest.raises(ValueError, match="≥ 0"):
            compute_hfac(uniform_grid, depth)

    def test_1d_depth_raises(self, uniform_grid):
        with pytest.raises(ValueError, match="2-D"):
            compute_hfac(uniform_grid, np.ones(NX))
