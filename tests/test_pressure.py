"""
Tests for core/dynamics/pressure.py.

Physical validation strategy (CLAUDE.md §Validation):
  1. Analytical solutions where available.
  2. Monotonicity and sign checks from known physics.
  3. Boundary condition enforcement.

Covers:
  - hydrostatic_pressure: dims, shape, coords, flat-ocean resting state,
    barotropic term, baroclinic stratification, increases with depth.
  - pressure_gradient_x / _y: dims, shape, coords, zero for uniform p,
    sign convention, magnitude, boundary zeros, antisymmetry.
  - surface_pressure_cg: convergence, Dirichlet BCs, flat-bottom Poisson.
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import create_z_grid, uniform_z_grid
from core.state import ModelState
from physics.eos import compute_density
from core.dynamics.pressure import (
    hydrostatic_pressure,
    pressure_gradient_x,
    pressure_gradient_y,
    surface_pressure_cg,
    G,
    RHO_0,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=8, ny=6, dx=2e3, dy=2e3)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=4, dz_uniform=25.0)


@pytest.fixture
def state_rest(hgrid, vgrid):
    """Uniform T=10, S=35, η=0 — ocean at rest."""
    return ModelState.from_grids(hgrid, vgrid, T_init=10.0, S_init=35.0)


@pytest.fixture
def rho_uniform(state_rest, vgrid):
    return compute_density(state_rest, vgrid, eos="linear")


@pytest.fixture
def p_rest(state_rest, rho_uniform, vgrid):
    return hydrostatic_pressure(rho_uniform, state_rest.eta, vgrid)


# ---------------------------------------------------------------------------
# 1. hydrostatic_pressure
# ---------------------------------------------------------------------------

class TestHydrostaticPressure:

    def test_returns_dataarray(self, p_rest):
        assert isinstance(p_rest, xr.DataArray)

    def test_dims(self, p_rest):
        assert p_rest.dims == ("k", "j_t", "i_t")

    def test_shape(self, p_rest):
        assert p_rest.shape == (4, 6, 8)    # (nz, ny, nx)

    def test_units_attr(self, p_rest):
        assert p_rest.attrs["units"] == "Pa"

    def test_coords_match_rho(self, p_rest, rho_uniform):
        """Pressure must share coordinates with density."""
        np.testing.assert_array_equal(p_rest.coords["k"].values,
                                      rho_uniform.coords["k"].values)
        np.testing.assert_array_equal(p_rest.coords["j_t"].values,
                                      rho_uniform.coords["j_t"].values)
        np.testing.assert_array_equal(p_rest.coords["i_t"].values,
                                      rho_uniform.coords["i_t"].values)

    def test_surface_layer_greater_than_zero(self, p_rest):
        """Pressure at k=0 must be positive (p_bar + half-layer weight)."""
        assert float(p_rest.isel(k=0).min()) > 0.0

    def test_increases_with_depth(self, p_rest):
        """Hydrostatic pressure must increase monotonically downward."""
        for k in range(3):
            p_upper = float(p_rest.isel(k=k).mean())
            p_lower = float(p_rest.isel(k=k + 1).mean())
            assert p_lower > p_upper

    def test_barotropic_term_eta_zero(self, rho_uniform, state_rest, vgrid):
        """With η=0, p_bar=0 — only baroclinic integral contributes."""
        p = hydrostatic_pressure(rho_uniform, state_rest.eta, vgrid)
        # Surface layer: p[0] = 0 + 0.5 * ρ * g * dz[0]
        dz0 = float(vgrid["dz"].values[0])
        rho0_val = float(rho_uniform.isel(k=0).mean())
        expected = 0.5 * rho0_val * G * dz0
        assert float(p.isel(k=0).mean()) == pytest.approx(expected, rel=1e-6)

    def test_barotropic_term_eta_positive(self, hgrid, vgrid):
        """Positive η adds ρ₀ g η to all layers uniformly."""
        st = ModelState.from_grids(hgrid, vgrid)
        st.eta.values[:] = 1.0   # 1 m SSH everywhere
        rho = compute_density(st, vgrid, eos="linear")

        p_zero = hydrostatic_pressure(
            rho,
            xr.DataArray(np.zeros((6, 8)), dims=["j_t", "i_t"],
                         coords={"j_t": st.eta.coords["j_t"],
                                 "i_t": st.eta.coords["i_t"]}),
            vgrid,
        )
        p_eta1 = hydrostatic_pressure(rho, st.eta, vgrid)

        delta = (p_eta1 - p_zero).values
        expected = RHO_0 * G * 1.0   # 1025 * 9.81 * 1 ≈ 10055 Pa
        np.testing.assert_allclose(delta, expected, rtol=1e-6)

    def test_stratified_baroclinic(self, hgrid, vgrid):
        """
        Warm surface layer (lower ρ) must produce smaller baroclinic pressure
        increment for the top layer than a cold uniform water column.
        """
        # Warm uniform column (low ρ)
        st_warm = ModelState.from_grids(hgrid, vgrid, T_init=25.0, S_init=35.0)
        # Cold uniform column (high ρ)
        st_cold = ModelState.from_grids(hgrid, vgrid, T_init=5.0, S_init=35.0)

        rho_warm = compute_density(st_warm, vgrid, eos="linear")
        rho_cold = compute_density(st_cold, vgrid, eos="linear")

        p_warm = hydrostatic_pressure(rho_warm, st_warm.eta, vgrid)
        p_cold = hydrostatic_pressure(rho_cold, st_cold.eta, vgrid)

        # Warm column has lighter layers → less pressure at every depth
        for k in range(4):
            assert float(p_warm.isel(k=k).mean()) < float(p_cold.isel(k=k).mean())

    def test_analytical_two_layer(self, hgrid):
        """
        Analytical check: two equal layers of thickness h = 50 m,
        ρ₁ = 1025 kg/m³ (top), ρ₂ = 1027 kg/m³ (bottom), η = 0.

        Expected:
            p[0] = ½ ρ₁ g h = ½·1025·9.81·50 ≈ 251,531 Pa
            p[1] = ρ₁ g h + ½ ρ₂ g h
                 = 1025·9.81·50 + ½·1027·9.81·50 ≈ 754,640 Pa
        """
        h = 50.0
        vg = create_z_grid(nz=2, dz=[h, h])
        ny, nx = 6, 8
        st = ModelState.from_grids(hgrid, vg)

        rho1, rho2 = 1025.0, 1027.0
        rho_np = np.empty((2, ny, nx))
        rho_np[0] = rho1
        rho_np[1] = rho2

        rho = xr.DataArray(
            rho_np,
            dims=["k", "j_t", "i_t"],
            coords={"k": vg.coords["k"],
                    "j_t": st.T.coords["j_t"],
                    "i_t": st.T.coords["i_t"]},
        )

        p = hydrostatic_pressure(rho, st.eta, vg)

        p0_expected = 0.5 * rho1 * G * h
        p1_expected = rho1 * G * h + 0.5 * rho2 * G * h

        assert float(p.isel(k=0).mean()) == pytest.approx(p0_expected, rel=1e-8)
        assert float(p.isel(k=1).mean()) == pytest.approx(p1_expected, rel=1e-8)


# ---------------------------------------------------------------------------
# 2. pressure_gradient_x
# ---------------------------------------------------------------------------

class TestPressureGradientX:

    def test_dims(self, p_rest, hgrid):
        pgf = pressure_gradient_x(p_rest, hgrid)
        assert pgf.dims == ("k", "j_t", "i_u")

    def test_shape(self, p_rest, hgrid):
        pgf = pressure_gradient_x(p_rest, hgrid)
        assert pgf.shape == (4, 6, 9)   # (nz, ny, nx+1)

    def test_units_attr(self, p_rest, hgrid):
        pgf = pressure_gradient_x(p_rest, hgrid)
        assert pgf.attrs["units"] == "m/s2"

    def test_coords_i_u(self, p_rest, hgrid):
        pgf = pressure_gradient_x(p_rest, hgrid)
        np.testing.assert_array_equal(pgf.coords["i_u"].values,
                                      hgrid.coords["i_u"].values)

    def test_uniform_pressure_gives_zero(self, p_rest, hgrid):
        """Uniform pressure field → zero gradient everywhere."""
        p_uniform = xr.DataArray(
            np.ones(p_rest.shape) * 1e5,
            dims=p_rest.dims, coords=p_rest.coords,
        )
        pgf = pressure_gradient_x(p_uniform, hgrid)
        np.testing.assert_allclose(pgf.values, 0.0, atol=1e-15)

    def test_boundary_faces_zero(self, p_rest, hgrid):
        """Western (i_u=0) and eastern (i_u=nx) boundary faces must be zero."""
        pgf = pressure_gradient_x(p_rest, hgrid)
        np.testing.assert_allclose(pgf.isel(i_u=0).values,  0.0, atol=1e-15)
        np.testing.assert_allclose(pgf.isel(i_u=-1).values, 0.0, atol=1e-15)

    def test_sign_convention(self, p_rest, hgrid):
        """
        High pressure to the east → westward PGF (negative u tendency).
        p increases eastward (i direction): PGF_x must be negative.
        """
        ny, nx = 6, 8
        nz = 4
        p_np = np.zeros((nz, ny, nx))
        # p[i] = i * 1e3 Pa (increases eastward)
        for i in range(nx):
            p_np[:, :, i] = i * 1e3

        p_east = xr.DataArray(
            p_np, dims=p_rest.dims, coords=p_rest.coords,
        )
        pgf = pressure_gradient_x(p_east, hgrid)
        # Interior faces should all be negative
        assert float(pgf.isel(i_u=slice(1, -1)).values.max()) < 0.0

    def test_magnitude(self, hgrid, vgrid):
        """
        For Δp = 1000 Pa over Δx = 2000 m (dx from hgrid):
            PGF_x = −(1/ρ₀) · Δp/Δx = −(1000 / (1025 × 2000)) ≈ −4.878 × 10⁻⁴ m/s²
        """
        nz, ny, nx = 4, 6, 8
        dx = 2e3

        p_np = np.zeros((nz, ny, nx))
        # p[:,  :, i+1] - p[:, :, i] = 1000 Pa for all i
        for i in range(nx):
            p_np[:, :, i] = i * 1000.0

        st = ModelState.from_grids(hgrid, vgrid)
        p = xr.DataArray(p_np, dims=["k", "j_t", "i_t"],
                         coords={"k": st.T.coords["k"],
                                 "j_t": st.T.coords["j_t"],
                                 "i_t": st.T.coords["i_t"]})

        pgf = pressure_gradient_x(p, hgrid)
        expected = -1000.0 / (RHO_0 * dx)
        np.testing.assert_allclose(
            pgf.isel(i_u=slice(1, -1)).values,
            expected, rtol=1e-8,
        )


# ---------------------------------------------------------------------------
# 3. pressure_gradient_y
# ---------------------------------------------------------------------------

class TestPressureGradientY:

    def test_dims(self, p_rest, hgrid):
        pgf = pressure_gradient_y(p_rest, hgrid)
        assert pgf.dims == ("k", "j_v", "i_t")

    def test_shape(self, p_rest, hgrid):
        pgf = pressure_gradient_y(p_rest, hgrid)
        assert pgf.shape == (4, 7, 8)   # (nz, ny+1, nx)

    def test_units_attr(self, p_rest, hgrid):
        pgf = pressure_gradient_y(p_rest, hgrid)
        assert pgf.attrs["units"] == "m/s2"

    def test_coords_j_v(self, p_rest, hgrid):
        pgf = pressure_gradient_y(p_rest, hgrid)
        np.testing.assert_array_equal(pgf.coords["j_v"].values,
                                      hgrid.coords["j_v"].values)

    def test_uniform_pressure_gives_zero(self, p_rest, hgrid):
        p_uniform = xr.DataArray(
            np.ones(p_rest.shape) * 1e5,
            dims=p_rest.dims, coords=p_rest.coords,
        )
        pgf = pressure_gradient_y(p_uniform, hgrid)
        np.testing.assert_allclose(pgf.values, 0.0, atol=1e-15)

    def test_boundary_faces_zero(self, p_rest, hgrid):
        """Southern (j_v=0) and northern (j_v=ny) boundary faces must be zero."""
        pgf = pressure_gradient_y(p_rest, hgrid)
        np.testing.assert_allclose(pgf.isel(j_v=0).values,  0.0, atol=1e-15)
        np.testing.assert_allclose(pgf.isel(j_v=-1).values, 0.0, atol=1e-15)

    def test_sign_convention(self, p_rest, hgrid):
        """High pressure to the north → southward PGF (negative v tendency)."""
        nz, ny, nx = 4, 6, 8
        p_np = np.zeros((nz, ny, nx))
        for j in range(ny):
            p_np[:, j, :] = j * 1e3   # increases northward

        p_north = xr.DataArray(p_np, dims=p_rest.dims, coords=p_rest.coords)
        pgf = pressure_gradient_y(p_north, hgrid)
        assert float(pgf.isel(j_v=slice(1, -1)).values.max()) < 0.0

    def test_magnitude(self, hgrid, vgrid):
        """Δp = 500 Pa, Δy = 2000 m → PGF_y = −500/(1025·2000) ≈ −2.44×10⁻⁴ m/s²."""
        nz, ny, nx = 4, 6, 8
        dy = 2e3
        p_np = np.zeros((nz, ny, nx))
        for j in range(ny):
            p_np[:, j, :] = j * 500.0

        st = ModelState.from_grids(hgrid, vgrid)
        p = xr.DataArray(p_np, dims=["k", "j_t", "i_t"],
                         coords={"k": st.T.coords["k"],
                                 "j_t": st.T.coords["j_t"],
                                 "i_t": st.T.coords["i_t"]})

        pgf = pressure_gradient_y(p, hgrid)
        expected = -500.0 / (RHO_0 * dy)
        np.testing.assert_allclose(
            pgf.isel(j_v=slice(1, -1)).values,
            expected, rtol=1e-8,
        )

    def test_x_and_y_pgf_independent(self, hgrid, vgrid):
        """
        A purely zonal pressure gradient must give zero PGF_y,
        and a purely meridional gradient must give zero PGF_x.
        """
        nz, ny, nx = 4, 6, 8
        st = ModelState.from_grids(hgrid, vgrid)

        # Purely zonal: p[i] = i * 1000
        p_zonal_np = np.zeros((nz, ny, nx))
        for i in range(nx):
            p_zonal_np[:, :, i] = i * 1000.0
        p_zon = xr.DataArray(p_zonal_np, dims=["k", "j_t", "i_t"],
                             coords=st.T.coords)

        # Purely meridional: p[j] = j * 1000
        p_merid_np = np.zeros((nz, ny, nx))
        for j in range(ny):
            p_merid_np[:, j, :] = j * 1000.0
        p_mer = xr.DataArray(p_merid_np, dims=["k", "j_t", "i_t"],
                             coords=st.T.coords)

        pgf_x_zonal  = pressure_gradient_x(p_zon, hgrid)
        pgf_y_zonal  = pressure_gradient_y(p_zon, hgrid)
        pgf_x_merid  = pressure_gradient_x(p_mer, hgrid)
        pgf_y_merid  = pressure_gradient_y(p_mer, hgrid)

        np.testing.assert_allclose(pgf_y_zonal.values,  0.0, atol=1e-15)
        np.testing.assert_allclose(pgf_x_merid.values,  0.0, atol=1e-15)
        # PGF_x for zonal and PGF_y for meridional are non-zero (interior faces)
        assert float(np.abs(pgf_x_zonal.isel(i_u=slice(1, -1)).values).max()) > 0.0
        assert float(np.abs(pgf_y_merid.isel(j_v=slice(1, -1)).values).max()) > 0.0


# ---------------------------------------------------------------------------
# 4. surface_pressure_cg
# ---------------------------------------------------------------------------

class TestSurfacePressureCG:

    def _flat_H(self, ny, nx, depth=100.0):
        return np.full((ny, nx), depth)

    def test_returns_ndarray(self):
        ny, nx = 10, 10
        div = np.zeros((ny, nx))
        eta = surface_pressure_cg(div, self._flat_H(ny, nx), dx=1e3, dy=1e3, dt=600.0)
        assert isinstance(eta, np.ndarray)

    def test_output_shape(self):
        ny, nx = 8, 12
        div = np.zeros((ny, nx))
        eta = surface_pressure_cg(div, self._flat_H(ny, nx), dx=1e3, dy=1e3, dt=600.0)
        assert eta.shape == (ny, nx)

    def test_zero_rhs_gives_zero_eta(self):
        """With zero divergence, the solution is η = 0 (Dirichlet BCs)."""
        ny, nx = 10, 10
        div = np.zeros((ny, nx))
        eta = surface_pressure_cg(div, self._flat_H(ny, nx), dx=1e3, dy=1e3, dt=600.0)
        np.testing.assert_allclose(eta, 0.0, atol=1e-10)

    def test_dirichlet_boundary_zero(self):
        """η = 0 on all boundaries regardless of interior divergence."""
        ny, nx = 12, 10
        rng = np.random.default_rng(42)
        div = np.zeros((ny, nx))
        div[2:ny-2, 2:nx-2] = rng.uniform(-1e-4, 1e-4, (ny - 4, nx - 4))
        eta = surface_pressure_cg(div, self._flat_H(ny, nx), dx=2e3, dy=2e3, dt=300.0)

        np.testing.assert_allclose(eta[0, :],  0.0, atol=1e-10)
        np.testing.assert_allclose(eta[-1, :], 0.0, atol=1e-10)
        np.testing.assert_allclose(eta[:, 0],  0.0, atol=1e-10)
        np.testing.assert_allclose(eta[:, -1], 0.0, atol=1e-10)

    def test_symmetric_forcing_gives_symmetric_solution(self):
        """
        A divergence field symmetric about the domain centre must produce a
        symmetric η.  This checks that the matrix assembly has no asymmetric bugs.
        """
        ny, nx = 11, 11  # odd sizes so there is a true centre cell
        div = np.zeros((ny, nx))
        jc, ic = ny // 2, nx // 2
        div[jc, ic] = -1e-4   # single source at centre

        eta = surface_pressure_cg(div, self._flat_H(ny, nx),
                                  dx=1e3, dy=1e3, dt=600.0)

        # Left-right symmetry (flip along x)
        np.testing.assert_allclose(eta, eta[:, ::-1], atol=1e-10)
        # Top-bottom symmetry (flip along y)
        np.testing.assert_allclose(eta, eta[::-1, :], atol=1e-10)

    def test_poisson_residual(self):
        """
        For a flat bottom, verify that A·η ≈ b by checking the residual is
        small.  This indirectly validates the sparse matrix assembly.
        """
        ny, nx = 10, 10
        rng = np.random.default_rng(7)
        div = np.zeros((ny, nx))
        div[3:7, 3:7] = rng.uniform(-1e-5, 1e-5, (4, 4))

        H = self._flat_H(ny, nx, depth=200.0)
        dx = dy = 1e3
        dt = 600.0
        eta = surface_pressure_cg(div, H, dx=dx, dy=dy, dt=dt, tol=1e-12)

        # Reconstruct A·eta at interior points and compare to div
        inv_g_dt2 = 1.0 / (G * dt ** 2)
        for j in range(1, ny - 1):
            for i in range(1, nx - 1):
                H_e = 0.5 * (H[j, i] + H[j, i + 1])
                H_w = 0.5 * (H[j, i] + H[j, i - 1])
                H_n = 0.5 * (H[j, i] + H[j + 1, i])
                H_s = 0.5 * (H[j, i] + H[j - 1, i])

                lhs = (
                    H_e * (eta[j, i + 1] - eta[j, i]) / dx ** 2
                    - H_w * (eta[j, i] - eta[j, i - 1]) / dx ** 2
                    + H_n * (eta[j + 1, i] - eta[j, i]) / dy ** 2
                    - H_s * (eta[j, i] - eta[j - 1, i]) / dy ** 2
                    - eta[j, i] * inv_g_dt2
                )
                assert lhs == pytest.approx(div[j, i], abs=1e-10)
