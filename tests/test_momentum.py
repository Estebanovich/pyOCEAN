"""
Tests for core/dynamics/momentum.py.

Validation strategy (CLAUDE.md §Validation):
  1. Analytical solutions where available (inertial oscillation, pure PGF acceleration).
  2. Sign and symmetry checks from known physics.
  3. Conservation / continuity checks.
  4. Dimensions, shapes, and coordinate consistency.

Covers
------
  _v_to_u / _u_to_v        : symmetry, zero interpolation from zero input
  coriolis_u / coriolis_v  : sign, dims, coords, f-plane analytical value
  eta_tendency             : dims, zero for uniform flow, sign for divergent flow
  diagnose_w               : surface/bottom BCs, continuity check
  momentum_tendency        : dims, resting-state = 0, Coriolis sign,
                             PGF acceleration, eta coupling
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import create_z_grid, uniform_z_grid
from core.state import ModelState
from core.dynamics.pressure import (
    hydrostatic_pressure,
    pressure_gradient_x,
    pressure_gradient_y,
)
from core.dynamics.momentum import (
    _v_to_u,
    _u_to_v,
    coriolis_u,
    coriolis_v,
    eta_tendency,
    diagnose_w,
    momentum_tendency,
)
from physics.eos import compute_density


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY, NZ = 8, 6, 4
DX = DY = 2000.0     # 2 km
DZ = 25.0            # m
F0 = 1e-4            # s⁻¹ (f-plane)


@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DY, f0=F0, beta=0.0)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=NZ, dz_uniform=DZ)


@pytest.fixture
def state_rest(hgrid, vgrid):
    """Uniform T=10, S=35, η=0, all velocities zero — ocean at rest."""
    return ModelState.from_grids(hgrid, vgrid, T_init=10.0, S_init=35.0)


@pytest.fixture
def rho_uniform(state_rest, vgrid):
    return compute_density(state_rest, vgrid, eos="linear")


@pytest.fixture
def p_hydrostatic(rho_uniform, state_rest, vgrid):
    return hydrostatic_pressure(rho_uniform, state_rest.eta, vgrid)


@pytest.fixture
def pgf_x_flat(p_hydrostatic, hgrid):
    return pressure_gradient_x(p_hydrostatic, hgrid)


@pytest.fixture
def pgf_y_flat(p_hydrostatic, hgrid):
    return pressure_gradient_y(p_hydrostatic, hgrid)


# ---------------------------------------------------------------------------
# 1.  Interpolation helpers
# ---------------------------------------------------------------------------

class TestInterpolation:

    def test_v_to_u_zero(self):
        """Zero v → zero v_at_u."""
        v_np = np.zeros((NZ, NY + 1, NX))
        result = _v_to_u(v_np)
        assert result.shape == (NZ, NY, NX + 1)
        assert np.all(result == 0.0)

    def test_u_to_v_zero(self):
        """Zero u → zero u_at_v."""
        u_np = np.zeros((NZ, NY, NX + 1))
        result = _u_to_v(u_np)
        assert result.shape == (NZ, NY + 1, NX)
        assert np.all(result == 0.0)

    def test_v_to_u_uniform(self):
        """Uniform v=1 → interior v_at_u = 1 (boundary = 0)."""
        v_np = np.ones((NZ, NY + 1, NX))
        result = _v_to_u(v_np)
        # Interior i_u=1..NX-1 should be 1
        assert np.allclose(result[:, :, 1:NX], 1.0)
        # Boundary i_u=0 and i_u=NX should be 0
        assert np.all(result[:, :, 0] == 0.0)
        assert np.all(result[:, :, NX] == 0.0)

    def test_u_to_v_uniform(self):
        """Uniform u=1 → interior u_at_v = 1 (boundary = 0)."""
        u_np = np.ones((NZ, NY, NX + 1))
        result = _u_to_v(u_np)
        assert np.allclose(result[:, 1:NY, :], 1.0)
        assert np.all(result[:, 0, :] == 0.0)
        assert np.all(result[:, NY, :] == 0.0)

    def test_v_to_u_average(self):
        """Check 4-point average numerics for a known pattern."""
        v_np = np.zeros((1, NY + 1, NX))
        # Set a known patch to verify the average formula
        v_np[0, 2, 3] = 4.0    # south-west V-face of U-face (j=2, i_u=4)
        v_np[0, 3, 3] = 4.0    # north-west
        v_np[0, 2, 4] = 4.0    # south-east
        v_np[0, 3, 4] = 4.0    # north-east
        result = _v_to_u(v_np)
        # U-face (j=2, i_u=4): average of v[2,3], v[3,3], v[2,4], v[3,4] = 4
        assert np.isclose(result[0, 2, 4], 4.0)

    def test_u_to_v_average(self):
        """Check 4-point average numerics for a known pattern."""
        u_np = np.zeros((1, NY, NX + 1))
        # Set known patch to verify average at V-face (j_v=2, i=3)
        u_np[0, 1, 3] = 4.0    # SW U-face
        u_np[0, 2, 3] = 4.0    # NW U-face
        u_np[0, 1, 4] = 4.0    # SE U-face
        u_np[0, 2, 4] = 4.0    # NE U-face
        result = _u_to_v(u_np)
        assert np.isclose(result[0, 2, 3], 4.0)

    def test_v_to_u_shape(self):
        v_np = np.random.rand(NZ, NY + 1, NX)
        assert _v_to_u(v_np).shape == (NZ, NY, NX + 1)

    def test_u_to_v_shape(self):
        u_np = np.random.rand(NZ, NY, NX + 1)
        assert _u_to_v(u_np).shape == (NZ, NY + 1, NX)


# ---------------------------------------------------------------------------
# 2.  Coriolis functions
# ---------------------------------------------------------------------------

class TestCoriolis:

    def test_coriolis_u_dims(self, state_rest, hgrid):
        f_q = hgrid["f_q"].values
        result = coriolis_u(state_rest.v, f_q)
        assert result.dims == ("k", "j_t", "i_u")

    def test_coriolis_u_shape(self, state_rest, hgrid):
        f_q = hgrid["f_q"].values
        result = coriolis_u(state_rest.v, f_q)
        assert result.shape == (NZ, NY, NX + 1)

    def test_coriolis_v_dims(self, state_rest, hgrid):
        f_q = hgrid["f_q"].values
        result = coriolis_v(state_rest.u, f_q)
        assert result.dims == ("k", "j_v", "i_t")

    def test_coriolis_v_shape(self, state_rest, hgrid):
        f_q = hgrid["f_q"].values
        result = coriolis_v(state_rest.u, f_q)
        assert result.shape == (NZ, NY + 1, NX)

    def test_coriolis_u_zero_v(self, state_rest, hgrid):
        """Zero v → zero Coriolis zonal acceleration."""
        f_q = hgrid["f_q"].values
        result = coriolis_u(state_rest.v, f_q)
        assert np.allclose(result.values, 0.0)

    def test_coriolis_v_zero_u(self, state_rest, hgrid):
        """Zero u → zero Coriolis meridional acceleration."""
        f_q = hgrid["f_q"].values
        result = coriolis_v(state_rest.u, f_q)
        assert np.allclose(result.values, 0.0)

    def test_coriolis_u_sign_NH(self, state_rest, hgrid):
        """
        f > 0 (NH), v > 0 (northward) → Coriolis deflects flow rightward
        → du/dt = +f*v > 0 (eastward acceleration).
        """
        state = state_rest.copy()
        v_val = np.ones_like(state.v.values)
        state.v.values[:] = v_val
        f_q = hgrid["f_q"].values
        result = coriolis_u(state.v, f_q)
        # Interior U-faces should have +f*v > 0
        assert np.all(result.values[:, :, 1:NX] > 0.0)

    def test_coriolis_v_sign_NH(self, state_rest, hgrid):
        """
        f > 0 (NH), u > 0 (eastward) → Coriolis deflects flow rightward
        → dv/dt = -f*u < 0 (southward acceleration).
        """
        state = state_rest.copy()
        state.u.values[:] = 1.0
        f_q = hgrid["f_q"].values
        result = coriolis_v(state.u, f_q)
        # Interior V-faces should have -f*u < 0
        assert np.all(result.values[:, 1:NY, :] < 0.0)

    def test_coriolis_u_analytical(self, state_rest, hgrid):
        """
        f-plane: uniform v=V₀ → interior du/dt = F₀·V₀  (exact).
        """
        V0 = 0.5   # m/s
        state = state_rest.copy()
        state.v.values[:] = V0
        f_q = hgrid["f_q"].values   # all = F0 for f-plane
        result = coriolis_u(state.v, f_q)
        # Interior i_u=1..NX-1: each f_at_u = F0, v_at_u = V0
        expected = F0 * V0
        assert np.allclose(result.values[:, :, 1:NX], expected)

    def test_coriolis_v_analytical(self, state_rest, hgrid):
        """
        f-plane: uniform u=U₀ → interior dv/dt = -F₀·U₀  (exact).

        Check only the strictly interior columns i=1..NX-2, because the
        two outermost i columns of u_at_v average a wall face (i_u=0 or
        i_u=NX, which is 0) with interior faces, giving values ≠ U₀.
        """
        U0 = 0.3
        state = state_rest.copy()
        state.u.values[:] = U0
        # NOTE: wall columns (i_u=0 and i_u=NX) are already kept at U0 here;
        # for the analytical test we only check i_t = 1..NX-2 where all four
        # contributing U-faces are interior and equal to U0.
        f_q = hgrid["f_q"].values
        result = coriolis_v(state.u, f_q)
        # Interior j_v=1..NY-1, strictly interior i=1..NX-2:
        # f_at_v = F0, u_at_v = U0 (all four averaging neighbours = U0)
        expected = -F0 * U0
        assert np.allclose(result.values[:, 1:NY, 1:NX - 1], expected)


# ---------------------------------------------------------------------------
# 3.  Sea-surface-height tendency
# ---------------------------------------------------------------------------

class TestEtaTendency:

    def test_dims(self, state_rest, hgrid, vgrid):
        result = eta_tendency(state_rest.u, state_rest.v, hgrid, vgrid)
        assert result.dims == ("j_t", "i_t")

    def test_shape(self, state_rest, hgrid, vgrid):
        result = eta_tendency(state_rest.u, state_rest.v, hgrid, vgrid)
        assert result.shape == (NY, NX)

    def test_zero_for_zero_velocity(self, state_rest, hgrid, vgrid):
        """Zero velocity → zero eta tendency."""
        result = eta_tendency(state_rest.u, state_rest.v, hgrid, vgrid)
        assert np.allclose(result.values, 0.0)

    def test_sign_convergence(self, state_rest, hgrid, vgrid):
        """
        Convergent flow (u > 0 toward domain centre) → dη/dt > 0 (sea level rises).
        Set u[k, j, i_u] > 0 for i_u < NX//2, < 0 for i_u > NX//2 (convergence).
        """
        state = state_rest.copy()
        # Build simple convergent u: u > 0 on left half, < 0 on right half
        u_np = np.zeros_like(state.u.values)
        # Create convergence: negative divergence (positive dη/dt)
        # div_u = (u[i+1] - u[i])/dx. For convergence we want div_u < 0.
        # Set u[i_u=1..NX//2] = +1 (flowing right), u[i_u=NX//2..NX-1] = -1 (flowing left)
        # → at i_t=NX//2-1 (interior): u[i_u=NX//2]-u[i_u=NX//2-1] = -1-1 = -2 → convergence
        for iu in range(1, NX // 2):
            u_np[:, :, iu] = 1.0
        for iu in range(NX // 2, NX):
            u_np[:, :, iu] = -1.0
        state.u.values[:] = u_np
        result = eta_tendency(state.u, state.v, hgrid, vgrid)
        # Sea level should rise near the centre
        centre_i = NX // 2 - 1
        assert np.all(result.values[:, centre_i] > 0.0), \
            f"Expected dη/dt > 0 near convergence, got {result.values[:, centre_i]}"

    def test_sign_divergence(self, state_rest, hgrid, vgrid):
        """
        Divergent flow → dη/dt < 0 (sea level drops).
        """
        state = state_rest.copy()
        u_np = np.zeros_like(state.u.values)
        # divergence: u > 0 on right half, u < 0 on left half
        for iu in range(1, NX // 2):
            u_np[:, :, iu] = -1.0
        for iu in range(NX // 2, NX):
            u_np[:, :, iu] = 1.0
        state.u.values[:] = u_np
        result = eta_tendency(state.u, state.v, hgrid, vgrid)
        centre_i = NX // 2 - 1
        assert np.all(result.values[:, centre_i] < 0.0)

    def test_integral_conservation(self, state_rest, hgrid, vgrid):
        """
        For closed boundaries (u=0 at walls), total dη/dt summed over domain
        must be zero: no volume change in a closed basin.
        """
        state = state_rest.copy()
        # Random interior velocities, but keep walls zero
        rng = np.random.default_rng(42)
        u_np = rng.standard_normal(state.u.shape)
        u_np[:, :, 0] = 0.0    # west wall
        u_np[:, :, -1] = 0.0   # east wall
        v_np = rng.standard_normal(state.v.shape)
        v_np[:, 0, :] = 0.0    # south wall
        v_np[:, -1, :] = 0.0   # north wall
        state.u.values[:] = u_np
        state.v.values[:] = v_np
        result = eta_tendency(state.u, state.v, hgrid, vgrid)
        # Sum of dη/dt * dx * dy should be zero (volume conservation)
        area = hgrid["area_t"].values
        total = np.sum(result.values * area)
        assert np.isclose(total, 0.0, atol=1e-8), f"Volume not conserved: Σ(dη*A) = {total}"


# ---------------------------------------------------------------------------
# 4.  Diagnostic vertical velocity
# ---------------------------------------------------------------------------

class TestDiagnoseW:

    def test_dims(self, state_rest, hgrid, vgrid):
        result = diagnose_w(state_rest.u, state_rest.v, hgrid, vgrid)
        assert result.dims == ("k_f", "j_t", "i_t")

    def test_shape(self, state_rest, hgrid, vgrid):
        result = diagnose_w(state_rest.u, state_rest.v, hgrid, vgrid)
        assert result.shape == (NZ + 1, NY, NX)

    def test_zero_velocity_zero_w(self, state_rest, hgrid, vgrid):
        """Zero u, v → zero w everywhere."""
        result = diagnose_w(state_rest.u, state_rest.v, hgrid, vgrid)
        assert np.allclose(result.values, 0.0)

    def test_surface_bc(self, state_rest, hgrid, vgrid):
        """w[k_f=0] = 0 (rigid-lid surface BC) always."""
        state = state_rest.copy()
        state.u.values[:] = np.random.rand(*state.u.shape)
        state.v.values[:] = np.random.rand(*state.v.shape)
        result = diagnose_w(state.u, state.v, hgrid, vgrid)
        assert np.allclose(result.values[0], 0.0), "Surface w must be 0"

    def test_bottom_bc_divergence_free(self, state_rest, hgrid, vgrid):
        """
        For a depth-uniform, horizontally uniform flow (zero 2-D divergence
        at every cell), the depth-integrated continuity gives w=0 everywhere.

        Physically: if u[k,j,i_u] = U (same at every U-face) and v=0, then
        (u[k,j,i+1] - u[k,j,i]) = 0 at every interior cell → div2d = 0 →
        w[k_f] = 0 for all k_f.
        """
        state = state_rest.copy()
        # Uniform u (same value at every U-face) → div_u = 0 everywhere
        state.u.values[:] = 0.5
        # Leave v = 0 → div_v = 0
        result = diagnose_w(state.u, state.v, hgrid, vgrid)
        assert np.allclose(result.values, 0.0, atol=1e-14), \
            f"w should be 0 for divergence-free flow, max={np.abs(result.values).max()}"

    def test_w_sign_convergence(self, state_rest, hgrid, vgrid):
        """
        Horizontal convergence (div2d < 0) must produce downwelling (w < 0
        at depth).  With rigid-lid w[0]=0 and div2d < 0:
            w[1] = w[0] + dz*div2d < 0.
        """
        state = state_rest.copy()
        # uniform convergence: u decreasing to the east at all layers
        u_np = np.zeros_like(state.u.values)
        # u[k,j,i_u=1]=1, u[k,j,i_u=2]=0.5, ... decreasing → div_u < 0 (wrong)
        # div_u = u[i+1]-u[i]/dx. We need u[i+1] < u[i] → u decreasing eastward.
        for iu in range(NX + 1):
            u_np[:, :, iu] = 1.0 - iu * 0.1   # decreasing
        u_np[:, :, 0] = 0.0   # wall
        u_np[:, :, -1] = 0.0  # wall
        state.u.values[:] = u_np
        result = diagnose_w(state.u, state.v, hgrid, vgrid)
        # At an interior cell, div_u < 0 → w[1] < 0 (downwelling)
        # Check a central column
        j_mid, i_mid = NY // 2, NX // 2
        assert result.values[1, j_mid, i_mid] < 0.0


# ---------------------------------------------------------------------------
# 5.  Full momentum_tendency
# ---------------------------------------------------------------------------

class TestMomentumTendency:

    def test_dims_u(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert tend.u.dims == ("k", "j_t", "i_u")

    def test_dims_v(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert tend.v.dims == ("k", "j_v", "i_t")

    def test_dims_eta(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert tend.eta.dims == ("j_t", "i_t")

    def test_shape_u(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert tend.u.shape == (NZ, NY, NX + 1)

    def test_shape_v(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert tend.v.shape == (NZ, NY + 1, NX)

    def test_shape_eta(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert tend.eta.shape == (NY, NX)

    def test_resting_state_zero_tendency(self, state_rest, pgf_x_flat, pgf_y_flat,
                                         hgrid, vgrid):
        """
        Ocean at rest with uniform stratification:
          u=v=0 → Coriolis = 0
          uniform pressure → PGF = 0
          zero velocity → dη/dt = 0
        → all tendencies must be zero.
        """
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert np.allclose(tend.u.values, 0.0), "du/dt should be 0 at rest"
        assert np.allclose(tend.v.values, 0.0), "dv/dt should be 0 at rest"
        assert np.allclose(tend.eta.values, 0.0), "deta/dt should be 0 at rest"

    def test_T_S_w_tendencies_zero(self, state_rest, pgf_x_flat, pgf_y_flat,
                                    hgrid, vgrid):
        """T, S, w tendencies must always be zero (no advection yet)."""
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert np.allclose(tend.T.values, 0.0)
        assert np.allclose(tend.S.values, 0.0)
        assert np.allclose(tend.w.values, 0.0)

    def test_coriolis_sign_in_tendency(self, state_rest, hgrid, vgrid,
                                        pgf_x_flat, pgf_y_flat):
        """
        NH f-plane (f > 0), v > 0 → du/dt > 0 (interior).
        """
        state = state_rest.copy()
        state.v.values[:] = 1.0
        tend = momentum_tendency(state, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert np.all(tend.u.values[:, :, 1:NX] > 0.0), \
            "du/dt must be > 0 when f>0, v>0"

    def test_pgf_acceleration(self, state_rest, hgrid, vgrid):
        """
        Impose a uniform zonal pressure gradient → only du/dt from PGF;
        dv/dt should be zero (no meridional PGF).
        """
        rho = compute_density(state_rest, vgrid, eos="linear")
        p0 = hydrostatic_pressure(rho, state_rest.eta, vgrid)

        # Add a linear pressure tilt: p = p0 + p_tilt where p_tilt increases eastward
        p_tilt_np = p0.values.copy()
        p_tilt = 500.0   # Pa / cell (large to see signal clearly)
        for i in range(NX):
            p_tilt_np[:, :, i] += p_tilt * i

        p_tilted = xr.DataArray(p_tilt_np, dims=p0.dims, coords=p0.coords)
        pgf_x_tilted = pressure_gradient_x(p_tilted, hgrid)
        pgf_y_zero = pressure_gradient_y(p0, hgrid)  # uniform p in y → PGF_y = 0

        tend = momentum_tendency(state_rest, pgf_x_tilted, pgf_y_zero, hgrid, vgrid)
        # du/dt = PGF_x < 0 (pressure higher to east → westward PGF)
        assert np.all(tend.u.values[:, :, 1:NX] < 0.0), \
            "PGF_x should accelerate u westward when p increases eastward"
        # dv/dt ≈ 0 (no meridional PGF, no meridional velocity)
        assert np.allclose(tend.v.values, 0.0, atol=1e-12)

    def test_viscosity_damps_gradient(self, state_rest, pgf_x_flat, pgf_y_flat,
                                       hgrid, vgrid):
        """
        With nu_h > 0 and a non-uniform velocity, the viscous tendency
        should have opposite sign to the Laplacian (it damps extrema).
        Set u to a Gaussian bump → Laplacian is negative at the peak
        → du/dt from viscosity < 0 at the peak.
        """
        state = state_rest.copy()
        u_np = np.zeros_like(state.u.values)
        # Gaussian bump at centre of U-face array
        i_mid = NX // 2
        for iu in range(1, NX):
            u_np[:, :, iu] = np.exp(-((iu - i_mid) / 2.0) ** 2)
        state.u.values[:] = u_np

        tend_inviscid = momentum_tendency(state, pgf_x_flat, pgf_y_flat,
                                          hgrid, vgrid, nu_h=0.0)
        tend_viscous = momentum_tendency(state, pgf_x_flat, pgf_y_flat,
                                         hgrid, vgrid, nu_h=1e3)

        # Viscous tendency at peak should be less than inviscid (peak is damped)
        assert np.any(
            tend_viscous.u.values[:, :, i_mid] < tend_inviscid.u.values[:, :, i_mid]
        ), "Viscosity should damp the Gaussian peak"

    def test_eta_tendency_from_full_momentum(self, state_rest, hgrid, vgrid,
                                              pgf_x_flat, pgf_y_flat):
        """
        eta tendency in the returned ModelState should equal the standalone
        eta_tendency() call for the same state.
        """
        from core.dynamics.momentum import eta_tendency as eta_tend_fn
        expected = eta_tend_fn(state_rest.u, state_rest.v, hgrid, vgrid)
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert np.allclose(tend.eta.values, expected.values)

    def test_coords_u(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        """Tendency u should share coordinates with state.u."""
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert set(tend.u.coords.keys()) == set(state_rest.u.coords.keys())

    def test_coords_v(self, state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid):
        tend = momentum_tendency(state_rest, pgf_x_flat, pgf_y_flat, hgrid, vgrid)
        assert set(tend.v.coords.keys()) == set(state_rest.v.coords.keys())
