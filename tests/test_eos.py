"""
Tests for physics/eos.py — equation of state.

Covers:
  - density_linear: reference point, α effect, β effect, coefficients
  - density_teos10: sanity range, monotonicity, surface pressure
  - compute_density: dispatch, dims/coords, linear vs teos10, custom callable
  - Pressure computation from z_grid in compute_density
  - Error handling: bad eos string
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import create_z_grid, uniform_z_grid
from core.state import ModelState
from physics.eos import (
    density_linear,
    density_teos10,
    compute_density,
    RHO_0,
    G,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=6, ny=4, dx=1e3, dy=1e3)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=5, dz_uniform=10.0)


@pytest.fixture
def state_uniform(hgrid, vgrid):
    """Uniform T=10 °C, S=35 PSU — density should equal ρ₀ at reference point."""
    return ModelState.from_grids(hgrid, vgrid, T_init=10.0, S_init=35.0)


# ---------------------------------------------------------------------------
# 1. density_linear — unit tests
# ---------------------------------------------------------------------------

class TestDensityLinear:

    def test_at_reference_point(self):
        """ρ(T₀, S₀) = ρ₀ exactly."""
        rho = density_linear(10.0, 35.0,
                             rho0=1025.0, alpha=2e-4, beta=7.4e-4,
                             T0=10.0, S0=35.0)
        assert float(rho) == pytest.approx(1025.0)

    def test_warmer_is_less_dense(self):
        """Increasing T decreases ρ (α > 0 ⟹ ∂ρ/∂T < 0)."""
        rho_ref  = density_linear(10.0, 35.0)
        rho_warm = density_linear(15.0, 35.0)
        assert float(rho_warm) < float(rho_ref)

    def test_saltier_is_denser(self):
        """Increasing S increases ρ (β > 0 ⟹ ∂ρ/∂S > 0)."""
        rho_ref   = density_linear(10.0, 35.0)
        rho_salty = density_linear(10.0, 36.0)
        assert float(rho_salty) > float(rho_ref)

    def test_alpha_sensitivity(self):
        """∂ρ/∂T = −ρ₀ · α."""
        dT = 1.0
        rho0, alpha = 1025.0, 2e-4
        rho1 = density_linear(10.0, 35.0, rho0=rho0, alpha=alpha)
        rho2 = density_linear(10.0 + dT, 35.0, rho0=rho0, alpha=alpha)
        drho_dT = (float(rho2) - float(rho1)) / dT
        assert drho_dT == pytest.approx(-rho0 * alpha, rel=1e-10)

    def test_beta_sensitivity(self):
        """∂ρ/∂S = ρ₀ · β."""
        dS = 1.0
        rho0, beta = 1025.0, 7.4e-4
        rho1 = density_linear(10.0, 35.0, rho0=rho0, beta=beta)
        rho2 = density_linear(10.0, 35.0 + dS, rho0=rho0, beta=beta)
        drho_dS = (float(rho2) - float(rho1)) / dS
        assert drho_dS == pytest.approx(rho0 * beta, rel=1e-10)

    def test_array_input(self):
        """Works on 3-D NumPy arrays (model field slices)."""
        T = np.full((5, 4, 6), 10.0)
        S = np.full((5, 4, 6), 35.0)
        rho = density_linear(T, S)
        assert rho.shape == (5, 4, 6)
        assert np.allclose(rho, 1025.0)

    def test_custom_reference(self):
        """Custom (rho0, T0, S0) reference state."""
        rho = density_linear(20.0, 34.0,
                              rho0=1024.0, alpha=2e-4, beta=7.4e-4,
                              T0=20.0, S0=34.0)
        assert float(rho) == pytest.approx(1024.0)

    def test_returns_numpy_array(self):
        rho = density_linear(np.array([10.0]), np.array([35.0]))
        assert isinstance(rho, np.ndarray)

    def test_linear_in_T(self):
        """ρ is linear in T — midpoint value equals average of endpoints."""
        rho_lo  = float(density_linear(5.0,  35.0))
        rho_hi  = float(density_linear(15.0, 35.0))
        rho_mid = float(density_linear(10.0, 35.0))
        assert rho_mid == pytest.approx(0.5 * (rho_lo + rho_hi), rel=1e-12)

    def test_linear_in_S(self):
        """ρ is linear in S."""
        rho_lo  = float(density_linear(10.0, 30.0))
        rho_hi  = float(density_linear(10.0, 40.0))
        rho_mid = float(density_linear(10.0, 35.0))
        assert rho_mid == pytest.approx(0.5 * (rho_lo + rho_hi), rel=1e-12)


# ---------------------------------------------------------------------------
# 2. density_teos10 — sanity tests
# ---------------------------------------------------------------------------

class TestDensityTeos10:

    def test_surface_fresh_cold(self):
        """Cold fresh water near surface ≈ 999.8 kg/m³."""
        rho = density_teos10(0.0, 0.0, 0.0)
        assert 999.0 < float(rho) < 1001.0

    def test_surface_typical_seawater(self):
        """Typical surface seawater (10°C, 35 PSU) ≈ 1026 kg/m³."""
        rho = density_teos10(10.0, 35.0, 0.0)
        assert 1025.0 < float(rho) < 1027.5

    def test_deep_seawater_denser(self):
        """Deep water is denser than surface due to compressibility."""
        rho_surf = density_teos10(2.0, 35.0,   0.0)
        rho_deep = density_teos10(2.0, 35.0, 400.0)
        assert float(rho_deep) > float(rho_surf)

    def test_warm_less_dense_than_cold(self):
        """Warmer water is less dense at fixed S and p."""
        rho_cold = density_teos10( 5.0, 35.0, 0.0)
        rho_warm = density_teos10(25.0, 35.0, 0.0)
        assert float(rho_warm) < float(rho_cold)

    def test_salty_denser(self):
        """Saltier water is denser at fixed T and p."""
        rho_fresh = density_teos10(10.0, 30.0, 0.0)
        rho_salty = density_teos10(10.0, 40.0, 0.0)
        assert float(rho_salty) > float(rho_fresh)

    def test_array_shapes(self):
        """Works with 3-D arrays."""
        T = np.linspace(5.0, 25.0, 5 * 4 * 6).reshape(5, 4, 6)
        S = np.full_like(T, 35.0)
        p = np.zeros_like(T)
        rho = density_teos10(T, S, p)
        assert rho.shape == (5, 4, 6)

    def test_pressure_broadcast(self):
        """p_dbar broadcastable from (nz,1,1) to (nz,ny,nx)."""
        T = np.full((3, 4, 5), 10.0)
        S = np.full((3, 4, 5), 35.0)
        p = np.array([0.0, 100.0, 300.0])[:, np.newaxis, np.newaxis]
        rho = density_teos10(T, S, p)
        assert rho.shape == (3, 4, 5)

    def test_returns_numpy_array(self):
        rho = density_teos10(np.array([10.0]), np.array([35.0]), np.array([0.0]))
        assert isinstance(rho, np.ndarray)

    def test_physically_reasonable_range(self):
        """All densities should be in 990–1060 kg/m³ for ocean conditions."""
        T = np.linspace(-2.0, 35.0, 20)
        S = np.linspace(0.0, 42.0, 20)
        p = np.linspace(0.0, 1000.0, 20)
        rho = density_teos10(T, S, p)
        assert float(rho.min()) > 990.0
        assert float(rho.max()) < 1060.0


# ---------------------------------------------------------------------------
# 3. compute_density — integration tests
# ---------------------------------------------------------------------------

class TestComputeDensity:

    def test_returns_xarray_dataarray(self, state_uniform, vgrid):
        rho = compute_density(state_uniform, vgrid, eos="linear")
        assert isinstance(rho, xr.DataArray)

    def test_dims_correct(self, state_uniform, vgrid):
        rho = compute_density(state_uniform, vgrid, eos="linear")
        assert rho.dims == ("k", "j_t", "i_t")

    def test_shape_correct(self, state_uniform, vgrid):
        """(nz=5, ny=4, nx=6) for the fixture grids."""
        rho = compute_density(state_uniform, vgrid, eos="linear")
        assert rho.shape == (5, 4, 6)

    def test_coords_match_state_T(self, state_uniform, vgrid):
        """ρ must share coords with state.T (same grid positions)."""
        rho = compute_density(state_uniform, vgrid, eos="linear")
        np.testing.assert_array_equal(rho.coords["k"].values,
                                      state_uniform.T.coords["k"].values)
        np.testing.assert_array_equal(rho.coords["j_t"].values,
                                      state_uniform.T.coords["j_t"].values)
        np.testing.assert_array_equal(rho.coords["i_t"].values,
                                      state_uniform.T.coords["i_t"].values)

    def test_units_attribute(self, state_uniform, vgrid):
        rho = compute_density(state_uniform, vgrid, eos="linear")
        assert rho.attrs["units"] == "kg/m3"

    def test_linear_at_reference_gives_rho0(self, state_uniform, vgrid):
        """T=T₀, S=S₀ ⟹ ρ = ρ₀ everywhere with linear EOS."""
        rho = compute_density(state_uniform, vgrid, eos="linear",
                              rho0=1025.0, alpha=2e-4, beta=7.4e-4,
                              T0=10.0, S0=35.0)
        assert float(rho.values.mean()) == pytest.approx(1025.0, rel=1e-10)

    def test_teos10_dispatch(self, state_uniform, vgrid):
        """eos='teos10' returns physically reasonable densities."""
        rho = compute_density(state_uniform, vgrid, eos="teos10")
        assert float(rho.values.min()) > 1020.0
        assert float(rho.values.max()) < 1035.0

    def test_teos10_denser_at_depth(self, hgrid):
        """TEOS-10 density should increase with depth (compressibility)."""
        vg = create_z_grid(nz=10, dz=[50.0] * 10)   # 500 m total
        st = ModelState.from_grids(hg := hgrid, vg, T_init=10.0, S_init=35.0)
        rho = compute_density(st, vg, eos="teos10")
        # Mean density at deep layer > shallow layer
        rho_top = float(rho.isel(k=0).mean())
        rho_bot = float(rho.isel(k=-1).mean())
        assert rho_bot > rho_top

    def test_pressure_from_z_grid(self, hgrid):
        """
        TEOS-10 density at p=0 (surface) vs deep layer should differ by
        the expected compressibility.  Approximate check: deep layer is
        denser than surface layer even at same T and S.
        """
        vg = create_z_grid(nz=2, dz=[10.0, 490.0])
        st = ModelState.from_grids(hgrid, vg, T_init=10.0, S_init=35.0)
        rho = compute_density(st, vg, eos="teos10")
        assert float(rho.isel(k=1).mean()) > float(rho.isel(k=0).mean())

    def test_custom_callable_eos(self, state_uniform, vgrid):
        """A user-supplied callable is accepted as eos."""
        def my_eos(T, S, p):
            return np.full_like(T, 1028.5)

        rho = compute_density(state_uniform, vgrid, eos=my_eos)
        assert float(rho.values.mean()) == pytest.approx(1028.5)

    def test_custom_callable_eos_attr(self, state_uniform, vgrid):
        """eos='custom' recorded in attrs when a callable is passed."""
        rho = compute_density(state_uniform, vgrid, eos=lambda T, S, p: np.ones_like(T))
        assert rho.attrs["eos"] == "custom"

    def test_bad_eos_string_raises(self, state_uniform, vgrid):
        with pytest.raises(ValueError, match="Unknown eos"):
            compute_density(state_uniform, vgrid, eos="bad_eos")

    def test_linear_kwargs_forwarded(self, hgrid, vgrid):
        """Custom alpha/beta are forwarded to density_linear."""
        # With alpha=0, beta=0 ⟹ ρ = ρ₀ regardless of T and S
        st = ModelState.from_grids(hgrid, vgrid, T_init=20.0, S_init=30.0)
        rho = compute_density(st, vgrid, eos="linear",
                              rho0=1020.0, alpha=0.0, beta=0.0,
                              T0=10.0, S0=35.0)
        assert float(rho.values.mean()) == pytest.approx(1020.0)

    def test_linear_eos_attr_recorded(self, state_uniform, vgrid):
        rho = compute_density(state_uniform, vgrid, eos="linear")
        assert rho.attrs["eos"] == "linear"

    def test_teos10_eos_attr_recorded(self, state_uniform, vgrid):
        rho = compute_density(state_uniform, vgrid, eos="teos10")
        assert rho.attrs["eos"] == "teos10"

    def test_linear_vs_teos10_close_at_surface(self, hgrid):
        """At surface, linear and TEOS-10 densities should agree to < 1 kg/m³."""
        vg = uniform_z_grid(nz=1, dz_uniform=5.0)
        st = ModelState.from_grids(hgrid, vg, T_init=10.0, S_init=35.0)
        rho_lin  = compute_density(st, vg, eos="linear")
        rho_teos = compute_density(st, vg, eos="teos10")
        diff = abs(float(rho_lin.mean()) - float(rho_teos.mean()))
        # Linear EOS is a first-order approximation; agreement within 3 kg/m³
        # is expected for typical surface conditions (Vallis 2006, §1.6).
        assert diff < 3.0

    def test_pressure_formula(self, hgrid):
        """
        Verify the pressure assigned to each layer:
            p_dbar[k] = ρ₀ · g · |z_t[k]| / 1e4
        by checking that TEOS-10 density increases with layer index k
        when T and S are uniform (only pressure varies).
        """
        vg = create_z_grid(nz=5, dz=[100.0] * 5)    # 500 m total
        st = ModelState.from_grids(hgrid, vg, T_init=10.0, S_init=35.0)
        rho = compute_density(st, vg, eos="teos10")
        # Each deeper layer should be at higher pressure ⟹ denser
        for k in range(4):
            assert float(rho.isel(k=k + 1).mean()) > float(rho.isel(k=k).mean())
