"""
Tests for core/state.py — ModelState prognostic variable container.

Covers:
  - Construction via from_grids (Cartesian + z-level)
  - Correct dimensions and shapes for all 6 variables
  - Coordinate consistency with horizontal and vertical grids
  - Default initial values (u=v=w=eta=0, T and S uniform)
  - Kinematic BCs: w=0 at surface and bottom
  - zeros_like factory
  - copy() produces independent deep copy
  - Arithmetic: __add__, __mul__, __rmul__
  - to_dataset() structure and attributes
  - shape property and __repr__
  - time tracking
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import create_z_grid, uniform_z_grid
from core.state import ModelState


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hgrid():
    """Small Cartesian grid: 6 × 4 T-cells."""
    return create_cartesian_grid(nx=6, ny=4, dx=1e3, dy=1e3)


@pytest.fixture
def vgrid():
    """Uniform z-level grid: 5 layers of 10 m each (50 m total)."""
    return uniform_z_grid(nz=5, dz_uniform=10.0)


@pytest.fixture
def state(hgrid, vgrid):
    """Default ModelState (zero velocities, T=10, S=35)."""
    return ModelState.from_grids(hgrid, vgrid)


# ---------------------------------------------------------------------------
# 1. Construction
# ---------------------------------------------------------------------------

class TestConstruction:

    def test_from_grids_returns_model_state(self, state):
        assert isinstance(state, ModelState)

    def test_from_grids_default_time(self, state):
        assert state.time == 0.0

    def test_from_grids_custom_time(self, hgrid, vgrid):
        st = ModelState.from_grids(hgrid, vgrid, time=3600.0)
        assert st.time == 3600.0

    def test_from_grids_custom_T_S(self, hgrid, vgrid):
        st = ModelState.from_grids(hgrid, vgrid, T_init=20.0, S_init=36.5)
        assert float(st.T.mean()) == pytest.approx(20.0)
        assert float(st.S.mean()) == pytest.approx(36.5)


# ---------------------------------------------------------------------------
# 2. Dimensions and shapes
# ---------------------------------------------------------------------------

class TestDimsAndShapes:
    """All variables must have the correct C-grid dims and matching shapes."""

    def test_u_dims(self, state):
        assert state.u.dims == ("k", "j_t", "i_u")

    def test_v_dims(self, state):
        assert state.v.dims == ("k", "j_v", "i_t")

    def test_w_dims(self, state):
        assert state.w.dims == ("k_f", "j_t", "i_t")

    def test_T_dims(self, state):
        assert state.T.dims == ("k", "j_t", "i_t")

    def test_S_dims(self, state):
        assert state.S.dims == ("k", "j_t", "i_t")

    def test_eta_dims(self, state):
        assert state.eta.dims == ("j_t", "i_t")

    # ---- Shapes (nx=6, ny=4, nz=5) ----------------------------------------
    def test_u_shape(self, state):
        assert state.u.shape == (5, 4, 7)   # (nz, ny, nx+1)

    def test_v_shape(self, state):
        assert state.v.shape == (5, 5, 6)   # (nz, ny+1, nx)

    def test_w_shape(self, state):
        assert state.w.shape == (6, 4, 6)   # (nz+1, ny, nx)

    def test_T_shape(self, state):
        assert state.T.shape == (5, 4, 6)   # (nz, ny, nx)

    def test_S_shape(self, state):
        assert state.S.shape == (5, 4, 6)

    def test_eta_shape(self, state):
        assert state.eta.shape == (4, 6)    # (ny, nx)

    def test_shape_property(self, state):
        s = state.shape
        assert s["u"]   == (5, 4, 7)
        assert s["v"]   == (5, 5, 6)
        assert s["w"]   == (6, 4, 6)
        assert s["T"]   == (5, 4, 6)
        assert s["S"]   == (5, 4, 6)
        assert s["eta"] == (4, 6)


# ---------------------------------------------------------------------------
# 3. Coordinate consistency
# ---------------------------------------------------------------------------

class TestCoordinates:

    def test_u_coords_match_grid(self, state, hgrid, vgrid):
        """u must share k, j_t, i_u coordinates with the parent grids."""
        np.testing.assert_array_equal(state.u.coords["i_u"].values,
                                      hgrid.coords["i_u"].values)
        np.testing.assert_array_equal(state.u.coords["j_t"].values,
                                      hgrid.coords["j_t"].values)
        np.testing.assert_array_equal(state.u.coords["k"].values,
                                      vgrid.coords["k"].values)

    def test_v_coords_match_grid(self, state, hgrid, vgrid):
        np.testing.assert_array_equal(state.v.coords["i_t"].values,
                                      hgrid.coords["i_t"].values)
        np.testing.assert_array_equal(state.v.coords["j_v"].values,
                                      hgrid.coords["j_v"].values)
        np.testing.assert_array_equal(state.v.coords["k"].values,
                                      vgrid.coords["k"].values)

    def test_w_kf_coords_match_grid(self, state, vgrid):
        np.testing.assert_array_equal(state.w.coords["k_f"].values,
                                      vgrid.coords["k_f"].values)

    def test_T_S_eta_share_jt_it(self, state, hgrid):
        for var in (state.T, state.S, state.eta):
            np.testing.assert_array_equal(var.coords["j_t"].values,
                                          hgrid.coords["j_t"].values)
            np.testing.assert_array_equal(var.coords["i_t"].values,
                                          hgrid.coords["i_t"].values)


# ---------------------------------------------------------------------------
# 4. Initial values
# ---------------------------------------------------------------------------

class TestInitialValues:

    def test_u_is_zero(self, state):
        assert float(np.abs(state.u.values).max()) == 0.0

    def test_v_is_zero(self, state):
        assert float(np.abs(state.v.values).max()) == 0.0

    def test_w_is_zero(self, state):
        assert float(np.abs(state.w.values).max()) == 0.0

    def test_eta_is_zero(self, state):
        assert float(np.abs(state.eta.values).max()) == 0.0

    def test_T_uniform_default(self, state):
        assert float(state.T.values.min()) == pytest.approx(10.0)
        assert float(state.T.values.max()) == pytest.approx(10.0)

    def test_S_uniform_default(self, state):
        assert float(state.S.values.min()) == pytest.approx(35.0)
        assert float(state.S.values.max()) == pytest.approx(35.0)

    def test_w_surface_bc(self, state):
        """w at k_f=0 (surface interface) must be zero."""
        assert float(np.abs(state.w.isel(k_f=0).values).max()) == 0.0

    def test_w_bottom_bc(self, state):
        """w at k_f=nz (bottom interface) must be zero."""
        assert float(np.abs(state.w.isel(k_f=-1).values).max()) == 0.0


# ---------------------------------------------------------------------------
# 5. zeros_like
# ---------------------------------------------------------------------------

class TestZerosLike:

    def test_zeros_like_shapes_match(self, state):
        z = ModelState.zeros_like(state)
        for key in ("u", "v", "w", "T", "S", "eta"):
            assert getattr(z, key).shape == getattr(state, key).shape

    def test_zeros_like_all_zero(self, state):
        z = ModelState.zeros_like(state)
        for key in ("u", "v", "w", "T", "S", "eta"):
            assert float(np.abs(getattr(z, key).values).max()) == 0.0

    def test_zeros_like_time_is_zero(self, state):
        # time of the reference is non-zero but zeros_like resets it
        state.time = 1000.0
        z = ModelState.zeros_like(state)
        assert z.time == 0.0

    def test_zeros_like_is_independent(self, state):
        z = ModelState.zeros_like(state)
        z.u.values[:] = 99.0
        assert float(state.u.values.max()) == 0.0


# ---------------------------------------------------------------------------
# 6. copy()
# ---------------------------------------------------------------------------

class TestCopy:

    def test_copy_preserves_values(self, state):
        state.u.values[0, 0, 0] = 1.5
        state.T.values[2, 1, 3] = 25.0
        c = state.copy()
        assert c.u.values[0, 0, 0] == pytest.approx(1.5)
        assert c.T.values[2, 1, 3] == pytest.approx(25.0)

    def test_copy_is_independent_u(self, state):
        c = state.copy()
        state.u.values[:] = 99.0
        assert float(c.u.values.max()) == 0.0

    def test_copy_is_independent_T(self, state):
        c = state.copy()
        state.T.values[:] = -99.0
        assert float(c.T.values.min()) == pytest.approx(10.0)

    def test_copy_preserves_time(self, state):
        state.time = 7200.0
        c = state.copy()
        assert c.time == 7200.0


# ---------------------------------------------------------------------------
# 7. Arithmetic
# ---------------------------------------------------------------------------

class TestArithmetic:

    def test_add_states(self, state):
        """state + zeros_like gives same values."""
        z = ModelState.zeros_like(state)
        result = state + z
        np.testing.assert_allclose(result.T.values, state.T.values)
        np.testing.assert_allclose(result.S.values, state.S.values)

    def test_add_time_from_left(self, state):
        """__add__ takes time from left operand."""
        state.time = 100.0
        other = state.copy()
        other.time = 200.0
        result = state + other
        assert result.time == 100.0

    def test_mul_scalar(self, state):
        """state * 2.0 doubles every field."""
        r = state * 2.0
        np.testing.assert_allclose(r.T.values, state.T.values * 2.0)
        np.testing.assert_allclose(r.S.values, state.S.values * 2.0)
        np.testing.assert_allclose(r.u.values, state.u.values * 2.0)

    def test_rmul_scalar(self, state):
        """2.0 * state == state * 2.0."""
        r1 = state * 3.0
        r2 = 3.0 * state
        np.testing.assert_allclose(r1.T.values, r2.T.values)
        np.testing.assert_allclose(r1.u.values, r2.u.values)

    def test_add_tendency_pattern(self, state):
        """
        Simulate one Euler step: state_new = state_old + dt * tendency.

        Checks that velocity updates correctly propagate dimensions.
        """
        dt = 600.0  # 10 minutes
        tend = ModelState.zeros_like(state)
        tend.u.values[0, :, 1:-1] = 1e-4  # add small zonal acceleration
        tend.T.values[:] = -0.001          # slight cooling

        new = state + dt * tend
        expected_u = dt * 1e-4
        assert float(new.u.isel(k=0).values[:, 1:-1].mean()) == pytest.approx(expected_u)
        expected_T = 10.0 + dt * (-0.001)
        assert float(new.T.values.mean()) == pytest.approx(expected_T)

    def test_arithmetic_preserves_dims(self, state):
        """Dims must not change after arithmetic."""
        result = 2.0 * state + ModelState.zeros_like(state)
        assert result.u.dims   == ("k", "j_t", "i_u")
        assert result.v.dims   == ("k", "j_v", "i_t")
        assert result.w.dims   == ("k_f", "j_t", "i_t")
        assert result.T.dims   == ("k", "j_t", "i_t")
        assert result.eta.dims == ("j_t", "i_t")


# ---------------------------------------------------------------------------
# 8. to_dataset()
# ---------------------------------------------------------------------------

class TestToDataset:

    def test_returns_xarray_dataset(self, state):
        ds = state.to_dataset()
        assert isinstance(ds, xr.Dataset)

    def test_all_variables_present(self, state):
        ds = state.to_dataset()
        for var in ("u", "v", "w", "T", "S", "eta"):
            assert var in ds

    def test_time_attribute(self, state):
        state.time = 1234.5
        ds = state.to_dataset()
        assert ds.attrs["time_s"] == pytest.approx(1234.5)

    def test_variable_dims_preserved(self, state):
        ds = state.to_dataset()
        assert ds["u"].dims   == ("k", "j_t", "i_u")
        assert ds["v"].dims   == ("k", "j_v", "i_t")
        assert ds["w"].dims   == ("k_f", "j_t", "i_t")
        assert ds["T"].dims   == ("k", "j_t", "i_t")
        assert ds["S"].dims   == ("k", "j_t", "i_t")
        assert ds["eta"].dims == ("j_t", "i_t")

    def test_variable_units(self, state):
        ds = state.to_dataset()
        assert ds["u"].attrs["units"] == "m/s"
        assert ds["T"].attrs["units"] == "degC"
        assert ds["S"].attrs["units"] == "PSU"
        assert ds["eta"].attrs["units"] == "m"


# ---------------------------------------------------------------------------
# 9. repr
# ---------------------------------------------------------------------------

class TestRepr:

    def test_repr_contains_nx_ny_nz(self, state):
        r = repr(state)
        assert "nx=6"  in r
        assert "ny=4"  in r
        assert "nz=5"  in r

    def test_repr_contains_variable_names(self, state):
        r = repr(state)
        for var in ("u", "v", "w", "T", "S", "eta"):
            assert var in r


# ---------------------------------------------------------------------------
# 10. Non-square / non-uniform grids
# ---------------------------------------------------------------------------

class TestNonUniformGrids:

    def test_non_square_domain(self):
        hg = create_cartesian_grid(nx=12, ny=7, dx=5e3, dy=2e3)
        vg = create_z_grid(nz=3, dz=[5.0, 10.0, 20.0])
        st = ModelState.from_grids(hg, vg)
        assert st.u.shape   == (3, 7, 13)
        assert st.v.shape   == (3, 8, 12)
        assert st.w.shape   == (4, 7, 12)
        assert st.T.shape   == (3, 7, 12)
        assert st.eta.shape == (7, 12)

    def test_single_layer(self):
        hg = create_cartesian_grid(nx=4, ny=3, dx=1e3, dy=1e3)
        vg = uniform_z_grid(nz=1, dz_uniform=50.0)
        st = ModelState.from_grids(hg, vg)
        assert st.u.shape   == (1, 3, 5)
        assert st.w.shape   == (2, 3, 4)  # nz+1 = 2 interfaces
        assert st.w.isel(k_f=0).values.max() == 0.0
        assert st.w.isel(k_f=1).values.max() == 0.0
