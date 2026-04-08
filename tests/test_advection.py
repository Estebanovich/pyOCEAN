"""
Tests for numerics/advection/__init__.py — first-order upwind tracer advection.

Validation strategy (CLAUDE.md §Validation):
  1. Analytical solutions: 1-D pure advection with known initial condition.
  2. Conservation: domain-integral of tracer conserved in closed basin.
  3. Monotonicity: upwind never creates new extrema.
  4. Sign / physics: warm water advected east → T increases downstream.
  5. Dims, shapes, coords.

Covers
------
  upwind_flux_x / y / z   : shape, coords, zero from zero velocity,
                             boundary faces zero, analytical value check
  advect_tracer            : shape, coords, conservation, sign, 1-D advection
  tracer_tendency          : ModelState structure, T/S populated, u/v/w/η zero
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import uniform_z_grid
from core.state import ModelState
from numerics.advection import (
    upwind_flux_x,
    upwind_flux_y,
    upwind_flux_z,
    advect_tracer,
    tracer_tendency,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY, NZ = 10, 8, 4
DX = DY = 1000.0    # 1 km
DZ = 10.0           # m
F0 = 1e-4           # s⁻¹


@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DY, f0=F0)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=NZ, dz_uniform=DZ)


@pytest.fixture
def state_uniform(hgrid, vgrid):
    """Uniform T=20, S=35; u=v=w=0 everywhere."""
    return ModelState.from_grids(hgrid, vgrid, T_init=20.0, S_init=35.0)


def _make_state(hgrid, vgrid, T=20.0, S=35.0, u=0.0, v=0.0):
    """Helper: uniform state with prescribed scalar velocities."""
    st = ModelState.from_grids(hgrid, vgrid, T_init=T, S_init=S)
    st.u.values[:] = u
    st.v.values[:] = v
    return st


# ---------------------------------------------------------------------------
# 1.  upwind_flux_x
# ---------------------------------------------------------------------------

class TestUpwindFluxX:

    def test_shape(self, state_uniform):
        F = upwind_flux_x(state_uniform.T, state_uniform.u)
        assert F.shape == (NZ, NY, NX + 1)

    def test_dims(self, state_uniform):
        F = upwind_flux_x(state_uniform.T, state_uniform.u)
        assert F.dims == ("k", "j_t", "i_u")

    def test_zero_u_zero_flux(self, state_uniform):
        """Zero velocity → zero flux everywhere."""
        F = upwind_flux_x(state_uniform.T, state_uniform.u)
        assert np.allclose(F.values, 0.0)

    def test_boundary_faces_zero(self, state_uniform):
        """West (i_u=0) and east (i_u=NX) boundary fluxes always zero."""
        st = state_uniform.copy()
        st.u.values[:] = 1.0
        F = upwind_flux_x(st.T, st.u)
        assert np.allclose(F.values[:, :, 0], 0.0), "West boundary flux not zero"
        assert np.allclose(F.values[:, :, -1], 0.0), "East boundary flux not zero"

    def test_positive_u_uses_west_cell(self, state_uniform):
        """
        With u > 0 (eastward), the upwind cell is to the west.
        Set T[i] = i so that T_west = T[i_u-1] = i_u-1.
        Flux at i_u = u * (i_u-1).
        """
        st = state_uniform.copy()
        U = 2.0
        st.u.values[:] = U
        # T[k,j,i] = float(i)
        for i in range(NX):
            st.T.values[:, :, i] = float(i)
        F = upwind_flux_x(st.T, st.u)
        # Interior i_u=1..NX-1: flux = U * T_west = U * (i_u-1)
        for iu in range(1, NX):
            expected = U * float(iu - 1)
            assert np.allclose(F.values[:, :, iu], expected), \
                f"flux at i_u={iu}: expected {expected}, got {F.values[0,0,iu]}"

    def test_negative_u_uses_east_cell(self, state_uniform):
        """
        With u < 0 (westward), the upwind cell is to the east.
        Set T[i] = i; flux at i_u = u * T[i_u].
        """
        st = state_uniform.copy()
        U = -2.0
        st.u.values[:] = U
        for i in range(NX):
            st.T.values[:, :, i] = float(i)
        F = upwind_flux_x(st.T, st.u)
        # Interior i_u=1..NX-1: flux = U * T_east = U * i_u
        for iu in range(1, NX):
            expected = U * float(iu)
            assert np.allclose(F.values[:, :, iu], expected), \
                f"flux at i_u={iu}: expected {expected}, got {F.values[0,0,iu]}"

    def test_uniform_tracer_flux_uniform(self, state_uniform):
        """
        Uniform T=c with any u → interior flux = u * c (constant).
        This verifies that upwind correctly returns the value of the
        constant field regardless of flow direction.
        """
        T_const = 20.0
        U = 3.0
        st = state_uniform.copy()
        st.u.values[:] = U
        F = upwind_flux_x(st.T, st.u)
        # Interior faces: upwind T = T_const always
        assert np.allclose(F.values[:, :, 1:NX], U * T_const)


# ---------------------------------------------------------------------------
# 2.  upwind_flux_y
# ---------------------------------------------------------------------------

class TestUpwindFluxY:

    def test_shape(self, state_uniform):
        F = upwind_flux_y(state_uniform.T, state_uniform.v)
        assert F.shape == (NZ, NY + 1, NX)

    def test_dims(self, state_uniform):
        F = upwind_flux_y(state_uniform.T, state_uniform.v)
        assert F.dims == ("k", "j_v", "i_t")

    def test_zero_v_zero_flux(self, state_uniform):
        F = upwind_flux_y(state_uniform.T, state_uniform.v)
        assert np.allclose(F.values, 0.0)

    def test_boundary_faces_zero(self, state_uniform):
        st = state_uniform.copy()
        st.v.values[:] = 1.0
        F = upwind_flux_y(st.T, st.v)
        assert np.allclose(F.values[:, 0, :], 0.0), "South boundary flux not zero"
        assert np.allclose(F.values[:, -1, :], 0.0), "North boundary flux not zero"

    def test_positive_v_uses_south_cell(self, state_uniform):
        """v > 0 (northward): upwind = south T-cell (T[j_v-1])."""
        st = state_uniform.copy()
        V = 1.5
        st.v.values[:] = V
        for j in range(NY):
            st.T.values[:, j, :] = float(j)
        F = upwind_flux_y(st.T, st.v)
        for jv in range(1, NY):
            expected = V * float(jv - 1)
            assert np.allclose(F.values[:, jv, :], expected)

    def test_negative_v_uses_north_cell(self, state_uniform):
        """v < 0 (southward): upwind = north T-cell (T[j_v])."""
        st = state_uniform.copy()
        V = -1.5
        st.v.values[:] = V
        for j in range(NY):
            st.T.values[:, j, :] = float(j)
        F = upwind_flux_y(st.T, st.v)
        for jv in range(1, NY):
            expected = V * float(jv)
            assert np.allclose(F.values[:, jv, :], expected)


# ---------------------------------------------------------------------------
# 3.  upwind_flux_z
# ---------------------------------------------------------------------------

class TestUpwindFluxZ:

    def test_shape(self, state_uniform):
        F = upwind_flux_z(state_uniform.T, state_uniform.w)
        assert F.shape == (NZ + 1, NY, NX)

    def test_dims(self, state_uniform):
        F = upwind_flux_z(state_uniform.T, state_uniform.w)
        assert F.dims == ("k_f", "j_t", "i_t")

    def test_zero_w_zero_flux(self, state_uniform):
        F = upwind_flux_z(state_uniform.T, state_uniform.w)
        assert np.allclose(F.values, 0.0)

    def test_surface_bottom_faces_zero(self, state_uniform):
        """k_f=0 (surface) and k_f=NZ (bottom) always 0."""
        # Even if we force w ≠ 0 at boundaries (physically wrong but test robustness)
        st = state_uniform.copy()
        st.w.values[:] = 1.0
        F = upwind_flux_z(st.T, st.w)
        assert np.allclose(F.values[0], 0.0), "Surface flux not zero"
        assert np.allclose(F.values[-1], 0.0), "Bottom flux not zero"

    def test_upward_w_uses_lower_layer(self, state_uniform):
        """
        w > 0 (upward): upwind value = T in layer below (T[k_f]).
        Set T[k] = k.  Interface k_f=1: T below = T[k=1] = 1.
        """
        st = state_uniform.copy()
        W = 0.1
        # Set w at interior interfaces only (leave 0 at boundaries)
        st.w.values[1:NZ, :, :] = W
        for k in range(NZ):
            st.T.values[k, :, :] = float(k)
        F = upwind_flux_z(st.T, st.w)
        # Interface k_f=1: T_below = T[k=1] = 1
        assert np.allclose(F.values[1], W * 1.0)
        # Interface k_f=2: T_below = T[k=2] = 2
        assert np.allclose(F.values[2], W * 2.0)

    def test_downward_w_uses_upper_layer(self, state_uniform):
        """
        w < 0 (downward): upwind value = T in layer above (T[k_f-1]).
        Set T[k] = k.  Interface k_f=1: T above = T[k=0] = 0.
        """
        st = state_uniform.copy()
        W = -0.1
        st.w.values[1:NZ, :, :] = W
        for k in range(NZ):
            st.T.values[k, :, :] = float(k)
        F = upwind_flux_z(st.T, st.w)
        # Interface k_f=1: T_above = T[k=0] = 0
        assert np.allclose(F.values[1], W * 0.0)
        # Interface k_f=2: T_above = T[k=1] = 1
        assert np.allclose(F.values[2], W * 1.0)

    def test_uniform_tracer_flux_uniform(self, state_uniform):
        """
        Uniform T=c with any w → flux = w * c (regardless of direction).
        """
        T_const = 20.0
        W = 0.05
        st = state_uniform.copy()
        st.w.values[1:NZ] = W
        F = upwind_flux_z(st.T, st.w)
        assert np.allclose(F.values[1:NZ], W * T_const)


# ---------------------------------------------------------------------------
# 4.  advect_tracer
# ---------------------------------------------------------------------------

class TestAdvectTracer:

    def test_shape(self, state_uniform, hgrid, vgrid):
        dτ = advect_tracer(state_uniform.T, state_uniform.u, state_uniform.v,
                           state_uniform.w, hgrid, vgrid)
        assert dτ.shape == (NZ, NY, NX)

    def test_dims(self, state_uniform, hgrid, vgrid):
        dτ = advect_tracer(state_uniform.T, state_uniform.u, state_uniform.v,
                           state_uniform.w, hgrid, vgrid)
        assert dτ.dims == ("k", "j_t", "i_t")

    def test_zero_velocity_zero_tendency(self, state_uniform, hgrid, vgrid):
        """Zero velocity → zero advective tendency."""
        dτ = advect_tracer(state_uniform.T, state_uniform.u, state_uniform.v,
                           state_uniform.w, hgrid, vgrid)
        assert np.allclose(dτ.values, 0.0)

    def test_uniform_tracer_zero_tendency_interior(self, state_uniform, hgrid, vgrid):
        """
        Uniform tracer with uniform interior flow → zero tendency for INTERIOR cells.

        For a uniform τ = c, dτ/dt = −c · div(u).  Interior cells (away from
        walls) have div(u) = (u[i+1] − u[i])/dx = 0 when u is uniform there,
        so dτ/dt = 0.  Wall-adjacent cells are excluded: div(u) ≠ 0 there
        because the wall forces flux = 0 at the boundary face (u = 0 physically)
        while the adjacent interior face carries non-zero flux.
        """
        st = state_uniform.copy()
        # Set u uniform at interior U-faces; walls stay 0
        st.u.values[:, :, 1:-1] = 2.0   # i_u = 1..NX-1 (includes NX-1 = east side)
        st.u.values[:, :, 0] = 0.0      # west wall BC (redundant but explicit)
        st.u.values[:, :, -1] = 0.0     # east wall BC
        st.v.values[:, 1:-1, :] = -1.0  # j_v = 1..NY-1
        st.v.values[:, 0, :] = 0.0
        st.v.values[:, -1, :] = 0.0
        dτ = advect_tracer(st.T, st.u, st.v, st.w, hgrid, vgrid)
        # Interior cells: i = 1..NX-2, j = 1..NY-2 (away from wall-adjacent rows/cols)
        assert np.allclose(dτ.values[:, 1:-1, 1:-1], 0.0, atol=1e-12), \
            "Uniform tracer should give zero tendency for interior cells"

    def test_tracer_conservation_closed_basin(self, state_uniform, hgrid, vgrid):
        """
        In a closed basin (u=0 at east/west walls, v=0 at north/south walls),
        the domain-integrated tracer tendency must be zero:
            Σ_{k,j,i} dτ/dt · dz · dy · dx = 0
        """
        rng = np.random.default_rng(42)
        st = state_uniform.copy()

        # Non-uniform T for interesting divergence
        st.T.values[:] = rng.standard_normal(st.T.shape) + 20.0

        # Non-zero u,v but with wall BCs
        u_np = rng.standard_normal(st.u.shape)
        u_np[:, :, 0] = 0.0    # west wall
        u_np[:, :, -1] = 0.0   # east wall
        st.u.values[:] = u_np

        v_np = rng.standard_normal(st.v.shape)
        v_np[:, 0, :] = 0.0    # south wall
        v_np[:, -1, :] = 0.0   # north wall
        st.v.values[:] = v_np

        dτ = advect_tracer(st.T, st.u, st.v, st.w, hgrid, vgrid)

        dz   = vgrid["dz"].values            # (NZ,)
        area = hgrid["area_t"].values         # (NY, NX)
        vol  = dz[:, np.newaxis, np.newaxis] * area[np.newaxis, :, :]
        total = np.sum(dτ.values * vol)
        assert np.isclose(total, 0.0, atol=1e-8), \
            f"Tracer conservation violated: Σ(dT/dt · dV) = {total:.3e}"

    def test_1d_zonal_advection_positive_u(self, hgrid, vgrid):
        """
        1-D pure zonal advection test (analytical).

        Setup: u > 0 (eastward), T = step function at i=NX//2.
        After one time step Δt: the step should shift one cell to the east
        (upwind scheme = pure donor-cell).

        Specifically:
            T[i < NX//2] = 0,  T[i ≥ NX//2] = 1,  u = U everywhere (walls = 0)
        Flux at i_u = 1..NX-1 with u > 0: F[i_u] = U * T[i_u-1]
        Divergence at i: -(F[i+1] - F[i])/dx

        For i = NX//2-1 (just west of the step):
            F[i_u=i] = U*T[i-1] = 0,  F[i_u=i+1] = U*T[i] = 0  → dT/dt = 0 ✓
        For i = NX//2 (first cell of the step):
            F[i_u=i] = U*T[i-1] = 0,  F[i_u=i+1] = U*T[i] = U
            → dT/dt = -(U - 0)/dx = -U/dx  (step erodes from the west) ✓

        We verify: dT/dt[k, j, NX//2] = -U/dx  (erosion of the leading edge).
        """
        st = ModelState.from_grids(hgrid, vgrid, T_init=0.0)
        U = 1.0
        i_step = NX // 2
        st.T.values[:, :, i_step:] = 1.0   # step at i=NX//2
        st.u.values[:] = U
        # Keep walls zero (physically required; u already 0 at walls in array)
        st.u.values[:, :, 0] = 0.0
        st.u.values[:, :, -1] = 0.0

        dτ = advect_tracer(st.T, st.u, st.v, st.w, hgrid, vgrid)

        # At i=NX//2 (leading edge): dT/dt = -U/dx (erosion)
        expected = -U / DX
        assert np.allclose(dτ.values[:, :, i_step], expected, rtol=1e-10), \
            f"Expected dT/dt={expected} at step edge, got {dτ.values[0,0,i_step]}"

        # Just west of step (i=i_step-1): dT/dt = 0
        assert np.allclose(dτ.values[:, :, i_step - 1], 0.0, atol=1e-14)

        # Deep inside step (i=i_step+1..NX-2): dT/dt = 0 (uniform → no divergence).
        # Exclude the east wall cell (i=NX-1): its boundary flux is forced to 0
        # while the adjacent interior face carries non-zero flux, giving a
        # wall-boundary artefact that is not part of the scheme's interior stencil.
        if i_step + 1 < NX - 1:
            assert np.allclose(dτ.values[:, :, i_step + 1:NX - 1], 0.0, atol=1e-14)

    def test_1d_zonal_advection_negative_u(self, hgrid, vgrid):
        """
        u < 0 (westward advection).  Step at i=NX//2.
        The western face of cell i_step-1 sees the step recede:
            dT/dt[i_step-1] = +|U|/dx  (accretion from the east).
        """
        st = ModelState.from_grids(hgrid, vgrid, T_init=0.0)
        U = -1.0
        i_step = NX // 2
        st.T.values[:, :, i_step:] = 1.0
        st.u.values[:] = U
        st.u.values[:, :, 0] = 0.0
        st.u.values[:, :, -1] = 0.0

        dτ = advect_tracer(st.T, st.u, st.v, st.w, hgrid, vgrid)

        # At i=i_step-1 (just west of step): flux converges → dT/dt > 0
        expected = abs(U) / DX
        assert np.allclose(dτ.values[:, :, i_step - 1], expected, rtol=1e-10)

        # At i=i_step (leading edge of step for westward flow): dT/dt = 0
        # (both cells to east and west of i_u=i_step+1 are 1 for i_step+1 < NX)
        assert np.allclose(dτ.values[:, :, i_step + 1:NX - 1], 0.0, atol=1e-14)

    def test_1d_meridional_advection(self, hgrid, vgrid):
        """
        Pure meridional advection: v > 0 northward, step at j=NY//2.
        dT/dt[j=NY//2] = -V/dy  (erosion of north-facing edge).
        """
        st = ModelState.from_grids(hgrid, vgrid, T_init=0.0)
        V = 0.5
        j_step = NY // 2
        st.T.values[:, j_step:, :] = 1.0
        st.v.values[:] = V
        st.v.values[:, 0, :] = 0.0
        st.v.values[:, -1, :] = 0.0

        dτ = advect_tracer(st.T, st.u, st.v, st.w, hgrid, vgrid)

        expected = -V / DY
        assert np.allclose(dτ.values[:, j_step, :], expected, rtol=1e-10)
        if j_step > 0:
            assert np.allclose(dτ.values[:, j_step - 1, :], 0.0, atol=1e-14)

    def test_monotonicity(self, hgrid, vgrid):
        """
        Upwind must not create new extrema (monotone scheme).
        After several advection time steps, min and max of T should stay
        within [T_min_initial, T_max_initial].
        """
        rng = np.random.default_rng(99)
        st = ModelState.from_grids(hgrid, vgrid)
        st.T.values[:] = rng.uniform(10.0, 30.0, st.T.shape)
        u_np = np.zeros(st.u.shape)
        u_np[:, :, 1:-1] = 0.5   # interior U-faces only; walls stay 0
        st.u.values[:] = u_np

        # Record initial extrema for INTERIOR cells only.
        # Wall-adjacent cells (i=0, i=NX-1, j=0, j=NY-1) experience boundary-
        # flux artefacts (forced zero flux at the wall face) that can cause
        # values outside the interior [T_min, T_max] range; this is a known
        # consequence of the zero-flux wall BC, not a scheme defect.
        T_int0 = st.T.values[:, 1:-1, 1:-1]
        T_min0 = T_int0.min()
        T_max0 = T_int0.max()

        dt = 100.0   # s
        for _ in range(10):
            dτ = advect_tracer(st.T, st.u, st.v, st.w, hgrid, vgrid)
            new_T = st.T + dt * dτ
            st.T.values[:] = new_T.values

        T_int_final = st.T.values[:, 1:-1, 1:-1]
        assert T_int_final.min() >= T_min0 - 1e-12, "Upwind created new minimum in interior"
        assert T_int_final.max() <= T_max0 + 1e-12, "Upwind created new maximum in interior"

    def test_sign_warm_advection(self, hgrid, vgrid):
        """
        Eastward flow (u > 0) with warm water on the west side.
        T = 30 for i < NX//2, T = 10 for i ≥ NX//2.
        At the interface i=NX//2 (step edge), dT/dt > 0 (warming).
        """
        st = ModelState.from_grids(hgrid, vgrid)
        i_step = NX // 2
        st.T.values[:, :, :i_step] = 30.0
        st.T.values[:, :, i_step:] = 10.0
        st.u.values[:] = 1.0
        st.u.values[:, :, 0] = 0.0
        st.u.values[:, :, -1] = 0.0

        dτ = advect_tracer(st.T, st.u, st.v, st.w, hgrid, vgrid)
        # At i=NX//2: F[i_u=i_step] = u*T_west = 1*30, F[i_u=i_step+1] = u*T[i_step] = 1*10
        # dT/dt = -(10-30)/dx = 20/dx > 0 (warming)
        assert np.all(dτ.values[:, :, i_step] > 0.0), \
            "Warm advection should increase T downstream"


# ---------------------------------------------------------------------------
# 5.  tracer_tendency (ModelState wrapper)
# ---------------------------------------------------------------------------

class TestTracerTendency:

    def test_returns_modelstate(self, state_uniform, hgrid, vgrid):
        tend = tracer_tendency(state_uniform, hgrid, vgrid)
        assert isinstance(tend, ModelState)

    def test_T_populated(self, state_uniform, hgrid, vgrid):
        tend = tracer_tendency(state_uniform, hgrid, vgrid)
        assert tend.T.dims == ("k", "j_t", "i_t")
        assert tend.T.shape == (NZ, NY, NX)

    def test_S_populated(self, state_uniform, hgrid, vgrid):
        tend = tracer_tendency(state_uniform, hgrid, vgrid)
        assert tend.S.dims == ("k", "j_t", "i_t")
        assert tend.S.shape == (NZ, NY, NX)

    def test_u_v_w_eta_zero(self, state_uniform, hgrid, vgrid):
        """Momentum and η tendencies must be zero."""
        tend = tracer_tendency(state_uniform, hgrid, vgrid)
        assert np.allclose(tend.u.values, 0.0)
        assert np.allclose(tend.v.values, 0.0)
        assert np.allclose(tend.w.values, 0.0)
        assert np.allclose(tend.eta.values, 0.0)

    def test_uniform_tracer_zero_tendency(self, state_uniform, hgrid, vgrid):
        """Interior cells only: uniform tracer with uniform interior flow → zero."""
        st = state_uniform.copy()
        st.u.values[:, :, 1:-1] = 1.0
        st.u.values[:, :, 0] = 0.0
        st.u.values[:, :, -1] = 0.0
        st.v.values[:, 1:-1, :] = 0.5
        st.v.values[:, 0, :] = 0.0
        st.v.values[:, -1, :] = 0.0
        tend = tracer_tendency(st, hgrid, vgrid)
        assert np.allclose(tend.T.values[:, 1:-1, 1:-1], 0.0)
        assert np.allclose(tend.S.values[:, 1:-1, 1:-1], 0.0)

    def test_T_and_S_independent(self, state_uniform, hgrid, vgrid):
        """T and S advection must be computed independently."""
        st = state_uniform.copy()
        # Give T and S very different values
        st.T.values[:] = 25.0
        st.T.values[:, :, NX // 2:] = 15.0
        st.S.values[:] = 36.0
        st.S.values[:, :, NX // 2:] = 34.0
        st.u.values[:] = 1.0
        st.u.values[:, :, 0] = 0.0
        st.u.values[:, :, -1] = 0.0

        tend = tracer_tendency(st, hgrid, vgrid)
        # T step amplitude = 25 - 15 = 10; S step amplitude = 36 - 34 = 2
        # ratio should be ~5 in the tendency at the step edge
        T_edge = tend.T.values[0, 0, NX // 2]
        S_edge = tend.S.values[0, 0, NX // 2]
        assert not np.isclose(T_edge, S_edge), \
            "T and S tendencies should differ when T and S have different gradients"
        assert np.isclose(T_edge / S_edge, (25 - 15) / (36 - 34), rtol=1e-6)

    def test_integration_with_ab2(self, hgrid, vgrid):
        """
        Verify that tracer_tendency integrates into the AB2 loop without error.
        Run 3 steps and check that T changes and is still finite.
        """
        from core.timestepping.ab2 import ab2_step

        st = ModelState.from_grids(hgrid, vgrid, T_init=20.0)
        i_step = NX // 2
        st.T.values[:, :, i_step:] = 25.0
        st.u.values[:] = 1.0
        st.u.values[:, :, 0] = 0.0
        st.u.values[:, :, -1] = 0.0

        def tend_fn(s):
            return tracer_tendency(s, hgrid, vgrid)

        DT = 50.0
        tend_prev = None
        for _ in range(3):
            st, tend_prev = ab2_step(st, tend_fn, DT, tend_prev)

        assert np.all(np.isfinite(st.T.values)), "T has NaN/Inf after 3 AB2 steps"
        # T should differ from initial (advection has occurred)
        assert not np.allclose(st.T.values[:, :, i_step], 25.0), \
            "T at step edge should have evolved"
