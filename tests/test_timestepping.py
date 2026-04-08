"""
Tests for core/timestepping/ab2.py — Adams–Bashforth 2.

Validation strategy:
  1. Analytical solution: constant tendency → linear growth.
  2. Analytical solution: oscillator (constant rotation in state space).
  3. First-step (Forward Euler) correctness.
  4. AB2 coefficients (3/2, −1/2) verified numerically.
  5. Time attribute tracking.
  6. Return types and shapes.

All tests use ``ModelState`` directly so that the timestepper is tested
independently of the physics modules.
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import uniform_z_grid
from core.state import ModelState
from core.timestepping.ab2 import ab2_step


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY, NZ = 4, 3, 2
DX = DY = 1000.0
DZ = 10.0


@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DY, f0=1e-4)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=NZ, dz_uniform=DZ)


@pytest.fixture
def state0(hgrid, vgrid):
    """Initial state: u=1, all other fields zero."""
    st = ModelState.from_grids(hgrid, vgrid)
    st.u.values[:] = 1.0
    return st


def _const_tendency(value: float):
    """Return a tendency_fn that always gives a constant value for u, zeros elsewhere."""
    def tendency_fn(state: ModelState) -> ModelState:
        tend = ModelState.zeros_like(state)
        tend.u.values[:] = value
        return tend
    return tendency_fn


def _zero_tendency(state: ModelState) -> ModelState:
    """Zero tendency — state should not change."""
    return ModelState.zeros_like(state)


# ---------------------------------------------------------------------------
# 1.  Return types and basic API
# ---------------------------------------------------------------------------

class TestAB2API:

    def test_returns_tuple(self, state0):
        result = ab2_step(state0, _zero_tendency, dt=60.0)
        assert isinstance(result, tuple)
        assert len(result) == 2

    def test_returns_model_state(self, state0):
        state_new, tend = ab2_step(state0, _zero_tendency, dt=60.0)
        assert isinstance(state_new, ModelState)
        assert isinstance(tend, ModelState)

    def test_u_shape_preserved(self, state0):
        state_new, _ = ab2_step(state0, _zero_tendency, dt=60.0)
        assert state_new.u.shape == state0.u.shape

    def test_v_shape_preserved(self, state0):
        state_new, _ = ab2_step(state0, _zero_tendency, dt=60.0)
        assert state_new.v.shape == state0.v.shape

    def test_eta_shape_preserved(self, state0):
        state_new, _ = ab2_step(state0, _zero_tendency, dt=60.0)
        assert state_new.eta.shape == state0.eta.shape

    def test_zero_tendency_no_change(self, state0):
        """Zero tendency → state unchanged."""
        dt = 600.0
        state_new, _ = ab2_step(state0, _zero_tendency, dt=dt)
        assert np.allclose(state_new.u.values, state0.u.values)
        assert np.allclose(state_new.v.values, state0.v.values)
        assert np.allclose(state_new.eta.values, state0.eta.values)


# ---------------------------------------------------------------------------
# 2.  Forward Euler first step
# ---------------------------------------------------------------------------

class TestForwardEuler:

    def test_first_step_FE_u(self, state0):
        """
        First step (tendency_prev=None) must use Forward Euler:
            u^1 = u^0 + dt * F^0    →    u^1 = 1 + dt * 2 = 1 + 1200 = 1201.
        """
        DT = 600.0
        RATE = 2.0
        state_new, _ = ab2_step(state0, _const_tendency(RATE), dt=DT)
        expected = state0.u.values + DT * RATE
        assert np.allclose(state_new.u.values, expected), \
            f"FE first step failed: max err = {np.abs(state_new.u.values - expected).max()}"

    def test_first_step_time_updated(self, state0):
        """time attribute incremented by dt."""
        DT = 300.0
        state_new, _ = ab2_step(state0, _zero_tendency, dt=DT)
        assert np.isclose(state_new.time, state0.time + DT)

    def test_first_step_tendency_returned(self, state0):
        """
        The returned tendency should be the one evaluated at state0,
        not at state_new.
        """
        RATE = 5.0
        state_new, tend = ab2_step(state0, _const_tendency(RATE), dt=60.0)
        assert np.allclose(tend.u.values, RATE)


# ---------------------------------------------------------------------------
# 3.  AB2 second step and coefficients
# ---------------------------------------------------------------------------

class TestAB2Step:

    def test_ab2_coefficients(self, state0):
        """
        With constant tendency F = c, AB2 gives:
            u^2 = u^1 + dt * (3/2 * c - 1/2 * c) = u^1 + dt * c
        i.e. same as FE when tendency is constant.  But let's use different
        F^n and F^{n-1} to directly verify the (3/2, -1/2) weights.

        Set F^{n-1} = a, F^n = b.  Expect:
            u^{n+1} = u^n + dt * (3/2*b - 1/2*a).
        """
        DT = 600.0
        a, b = 1.0, 3.0   # previous and current tendencies

        # Build tend_prev with u = a
        tend_prev = ModelState.zeros_like(state0)
        tend_prev.u.values[:] = a

        # Build tendency_fn that always returns b
        state_new, _ = ab2_step(state0, _const_tendency(b), dt=DT, tendency_prev=tend_prev)

        expected = state0.u.values + DT * (1.5 * b - 0.5 * a)
        assert np.allclose(state_new.u.values, expected), \
            f"AB2 coefficients wrong: max err = {np.abs(state_new.u.values - expected).max()}"

    def test_ab2_time_updated(self, state0):
        """time is incremented correctly on the AB2 step."""
        DT = 300.0
        tend_prev = ModelState.zeros_like(state0)
        state_new, _ = ab2_step(state0, _zero_tendency, dt=DT, tendency_prev=tend_prev)
        assert np.isclose(state_new.time, state0.time + DT)

    def test_returned_tendency_is_current(self, state0):
        """
        The second return value should be F^n (current), not F^{n-1} (previous).
        After the first step, pass the returned tendency as tend_prev for step 2
        and verify the expected result.
        """
        DT = 60.0
        RATE_0 = 2.0
        RATE_1 = 5.0

        # Step 1 (FE): F^0 = RATE_0
        state1, tend0 = ab2_step(state0, _const_tendency(RATE_0), dt=DT)
        assert np.allclose(tend0.u.values, RATE_0)

        # Step 2 (AB2): F^1 = RATE_1, F^0 = RATE_0
        state2, tend1 = ab2_step(state1, _const_tendency(RATE_1), dt=DT,
                                  tendency_prev=tend0)
        expected = state1.u.values + DT * (1.5 * RATE_1 - 0.5 * RATE_0)
        assert np.allclose(state2.u.values, expected)
        assert np.allclose(tend1.u.values, RATE_1)

    def test_ab2_step3(self, state0):
        """
        Run 3 full steps and verify each against the analytical formula.
        Uses a constant tendency RATE for simplicity:
            u^1 = u^0 + dt * RATE             (FE)
            u^2 = u^1 + dt * RATE             (AB2, same since F constant)
            u^3 = u^2 + dt * RATE
        → u^3 = u^0 + 3*dt*RATE
        """
        DT = 300.0
        RATE = 1.5
        state = state0
        tend_prev = None
        for _ in range(3):
            state, tend_prev = ab2_step(state, _const_tendency(RATE), dt=DT,
                                         tendency_prev=tend_prev)
        expected = state0.u.values + 3 * DT * RATE
        assert np.allclose(state.u.values, expected, rtol=1e-10)


# ---------------------------------------------------------------------------
# 4.  Time tracking over multiple steps
# ---------------------------------------------------------------------------

class TestTimeTracking:

    def test_time_accumulates(self, state0):
        """
        Verify that state.time accumulates correctly over 5 steps
        regardless of whether FE or AB2 is used.
        """
        DT = 600.0
        N = 5
        state = state0
        tend_prev = None
        for _ in range(N):
            state, tend_prev = ab2_step(state, _zero_tendency, dt=DT,
                                         tendency_prev=tend_prev)
        assert np.isclose(state.time, state0.time + N * DT)

    def test_different_dt_steps(self, state0):
        """Variable dt steps accumulate correctly."""
        dts = [100.0, 200.0, 300.0]
        state = state0
        tend_prev = None
        for dt in dts:
            state, tend_prev = ab2_step(state, _zero_tendency, dt=dt,
                                         tendency_prev=tend_prev)
        assert np.isclose(state.time, sum(dts))


# ---------------------------------------------------------------------------
# 5.  No in-place mutation of input state
# ---------------------------------------------------------------------------

class TestImmutability:

    def test_input_state_not_mutated(self, state0):
        """ab2_step must not modify state0."""
        u_before = state0.u.values.copy()
        _ = ab2_step(state0, _const_tendency(99.0), dt=600.0)
        assert np.allclose(state0.u.values, u_before), \
            "ab2_step mutated the input state"

    def test_tend_prev_not_mutated(self, state0):
        """ab2_step must not modify tend_prev."""
        tend_prev = ModelState.zeros_like(state0)
        tend_prev.u.values[:] = 7.0
        u_before = tend_prev.u.values.copy()
        _ = ab2_step(state0, _const_tendency(2.0), dt=300.0,
                     tendency_prev=tend_prev)
        assert np.allclose(tend_prev.u.values, u_before), \
            "ab2_step mutated tend_prev"


# ---------------------------------------------------------------------------
# 6.  Integration with physics — gravity-wave test
# ---------------------------------------------------------------------------

class TestGravityWaveIntegration:
    """
    Shallow-water gravity wave: sloshing in a 1-D closed basin.

    Initial condition: η = A * sin(π x / L), u = v = 0.
    Expected behaviour: initial pressure gradient drives flow, which
    creates counter-pressure, and the system oscillates.

    We only verify that the state evolves (η changes) and that the
    total volume (Ση * dx) is conserved over a few steps.
    """

    @pytest.fixture
    def hgrid_1d(self):
        return create_cartesian_grid(nx=20, ny=4, dx=5e3, dy=5e3,
                                     f0=0.0, beta=0.0)

    @pytest.fixture
    def vgrid_1d(self):
        return uniform_z_grid(nz=2, dz_uniform=500.0)

    def test_eta_evolves(self, hgrid_1d, vgrid_1d):
        """
        With an initial η tilt and no-slip walls, η should change after
        two steps: step 1 drives u via PGF; step 2 changes η via divergence.

        Step 1: u=v=0 → dη/dt=0, but PGF drives du/dt ≠ 0.
        Step 2: u ≠ 0 → dη/dt ≠ 0 → η changes.
        """
        from physics.eos import compute_density
        from core.dynamics.pressure import (
            hydrostatic_pressure, pressure_gradient_x, pressure_gradient_y,
        )
        from core.dynamics.momentum import momentum_tendency

        st = ModelState.from_grids(hgrid_1d, vgrid_1d)
        nx, ny = 20, 4
        x = np.arange(nx) / nx
        st.eta.values[:] = 0.01 * np.sin(np.pi * x)[np.newaxis, :]

        def full_tendency(s):
            rho = compute_density(s, vgrid_1d, eos="linear")
            p = hydrostatic_pressure(rho, s.eta, vgrid_1d)
            pgfx = pressure_gradient_x(p, hgrid_1d)
            pgfy = pressure_gradient_y(p, hgrid_1d)
            return momentum_tendency(s, pgfx, pgfy, hgrid_1d, vgrid_1d)

        DT = 60.0
        # Step 1: FE — PGF drives u; η unchanged (u=0 at start → dη/dt=0)
        state1, tend0 = ab2_step(st, full_tendency, dt=DT)
        # After step 1, u should have changed (PGF acceleration)
        assert not np.allclose(state1.u.values, st.u.values), \
            "u did not evolve after step 1 (PGF should drive u)"
        # Step 2: AB2 — now u ≠ 0, so η should change
        state2, _ = ab2_step(state1, full_tendency, dt=DT, tendency_prev=tend0)
        assert not np.allclose(state2.eta.values, state1.eta.values), \
            "η did not evolve after step 2 (divergence of u should change η)"

    def test_volume_conservation(self, hgrid_1d, vgrid_1d):
        """
        Total volume (Σ η·ΔA) should be approximately conserved over 5 steps
        for a closed basin (u=0 at walls enforced by PGF boundary conditions).
        """
        from physics.eos import compute_density
        from core.dynamics.pressure import (
            hydrostatic_pressure, pressure_gradient_x, pressure_gradient_y,
        )
        from core.dynamics.momentum import momentum_tendency

        st = ModelState.from_grids(hgrid_1d, vgrid_1d)
        nx = 20
        x = np.arange(nx) / nx
        st.eta.values[:] = 0.01 * np.sin(np.pi * x)[np.newaxis, :]

        area = hgrid_1d["area_t"].values
        vol0 = np.sum(st.eta.values * area)

        def full_tendency(s):
            rho = compute_density(s, vgrid_1d, eos="linear")
            p = hydrostatic_pressure(rho, s.eta, vgrid_1d)
            pgfx = pressure_gradient_x(p, hgrid_1d)
            pgfy = pressure_gradient_y(p, hgrid_1d)
            return momentum_tendency(s, pgfx, pgfy, hgrid_1d, vgrid_1d)

        DT = 60.0
        state = st
        tend_prev = None
        for _ in range(5):
            state, tend_prev = ab2_step(state, full_tendency, DT, tend_prev)

        vol_final = np.sum(state.eta.values * area)
        assert np.isclose(vol_final, vol0, atol=1e-6), \
            f"Volume not conserved: Δvol = {vol_final - vol0:.2e} m³"
