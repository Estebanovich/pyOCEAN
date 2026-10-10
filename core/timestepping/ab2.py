"""
Adams–Bashforth 2 (AB2) time stepping for pyOCEAN — Phase 1.

Implements the second-order Adams–Bashforth explicit scheme for advancing the
model state one time step Δt forward:

    state^{n+1} = state^n + Δt · [3/2 · F^n − 1/2 · F^{n-1}]    (AB2, n ≥ 1)
    state^1     = state^0 + Δt · F^0                               (Forward Euler, n = 0)

where F^n = tendency_fn(state^n) is the model RHS evaluated at time n.

The first step (n=0 → n=1) uses Forward Euler because no previous tendency
F^{n-1} is available.  Subsequent steps use the full AB2 formula.  This
start-up procedure is standard in ocean and atmosphere models (Durran 2010,
§2.7.1).

Stability
---------
AB2 is stable for linear problems when the time step satisfies the condition:

    Δt < Δt_CFL / (stability_factor)

For the shallow-water gravity-wave CFL:

    Δt_CFL = Δx / (g H)^{1/2}

where H is the total depth and Δx is the grid spacing.  In practice, for a
regional ocean model AB2 is stable with a Courant number ≲ 0.3–0.5 when
applied to the baroclinic modes (Durran 2010, §6.4).

Usage
-----
    from core.timestepping.ab2 import ab2_step

    # Build the tendency function (wraps eos → pressure → momentum)
    def full_tendency(st):
        rho  = compute_density(st, vgrid)
        p    = hydrostatic_pressure(rho, st.eta, vgrid)
        pgfx = pressure_gradient_x(p, hgrid)
        pgfy = pressure_gradient_y(p, hgrid)
        tend = momentum_tendency(st, pgfx, pgfy, hgrid, vgrid, nu_h=nu_h)
        return tend

    # Time loop
    tend_prev = None
    for n in range(N_steps):
        state, tend_prev = ab2_step(state, full_tendency, dt, tend_prev)
        state.w = diagnose_w(state.u, state.v, hgrid, vgrid)

References
----------
Durran (2010) — Numerical Methods for Fluid Dynamics, §2.7.1 (AB2 derivation
    and stability), §6.4 (ocean model applications).
Durran (2010) — Table 2.1 (AB2 coefficients).
Marshall et al. (1997) — MITgcm time-stepping, §4.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable, Optional

from core.state import ModelState

if TYPE_CHECKING:
    from core.dynamics.free_surface import ImplicitFreeSurface


def ab2_step(
    state: ModelState,
    tendency_fn: Callable[[ModelState], ModelState],
    dt: float,
    tendency_prev: Optional[ModelState] = None,
) -> tuple[ModelState, ModelState]:
    """
    Advance the model state by one time step using Adams–Bashforth 2.

    First step (n=0, ``tendency_prev=None``)
        Uses Forward Euler, which is first-order accurate but self-starting:

            state^1 = state^0 + Δt · F^0

    Subsequent steps (n ≥ 1, ``tendency_prev`` provided)
        Uses the full AB2 formula with coefficients (3/2, −1/2)
        (Durran 2010, eq. 2.132):

            state^{n+1} = state^n + Δt · (3/2 · F^n − 1/2 · F^{n-1})

    The state time attribute is updated: ``state_new.time = state.time + dt``.

    Parameters
    ----------
    state : ModelState
        Current model state at time n (``state.time = n · Δt``).
    tendency_fn : Callable[[ModelState], ModelState]
        RHS function.  Called as ``tendency_fn(state)`` and must return a
        ModelState whose fields represent d(state)/dt.
    dt : float
        Time step [s].
    tendency_prev : ModelState or None, optional
        Tendency F^{n-1} from the previous step.  ``None`` on the first step
        (triggers Forward Euler start-up).

    Returns
    -------
    state_new : ModelState
        Model state at time n+1.  ``state_new.time = state.time + dt``.
    tendency_curr : ModelState
        Tendency F^n just evaluated.  Pass this as ``tendency_prev`` on the
        next call.

    Examples
    --------
    One-step invocation:

    >>> state1, tend0 = ab2_step(state0, my_tendency_fn, dt=600.0)
    >>> state2, tend1 = ab2_step(state1, my_tendency_fn, dt=600.0, tendency_prev=tend0)

    Compact time loop:

    >>> tend_prev = None
    >>> for n in range(N_steps):
    ...     state, tend_prev = ab2_step(state, full_tendency, dt, tend_prev)

    Notes
    -----
    * ``tendency_fn`` must accept exactly one positional argument (the state)
      plus any additional context captured via closure.  Wrap the full physics
      chain in a closure before passing it here.

    * The `ModelState.__add__` and `__rmul__` operators are used for state
      arithmetic; no in-place modification of ``state`` occurs.

    * For diagnostics computed outside the tendency (e.g. vertical velocity w
      from continuity), update them after the call:
          state = state._replace_w(diagnose_w(state.u, state.v, hgrid, vgrid))
      or more simply assign state.w directly (ModelState attributes are mutable).
    """
    # Evaluate current tendency F^n
    tendency_curr: ModelState = tendency_fn(state)

    if tendency_prev is None:
        # ---- Forward Euler start-up (first step) --------------------------
        # state^1 = state^0 + Δt · F^0       (Durran 2010, eq. 2.105)
        state_new = state + dt * tendency_curr
    else:
        # ---- Adams–Bashforth 2 (n ≥ 1) ------------------------------------
        # state^{n+1} = state^n + Δt · (3/2 · F^n − 1/2 · F^{n-1})
        #             = state^n + (3/2 · Δt) · F^n + (−1/2 · Δt) · F^{n-1}
        #                                              (Durran 2010, eq. 2.132)
        state_new = state + (1.5 * dt) * tendency_curr + (-0.5 * dt) * tendency_prev

    state_new.time = state.time + dt

    return state_new, tendency_curr


def ab2_step_implicit_fs(
    state: ModelState,
    tendency_fn: Callable[[ModelState], ModelState],
    dt: float,
    free_surface: "ImplicitFreeSurface",
    tendency_prev: Optional[ModelState] = None,
) -> tuple[ModelState, ModelState]:
    """
    One AB2 step with an implicit free surface (MITgcm-style splitting).

    All tendencies except the surface-pressure gradient are extrapolated with
    AB2 to n + ½ (Forward Euler on the first step), then the free surface is
    solved implicitly (``core.dynamics.free_surface``):

        G^{n+½} = 3/2 G^n − 1/2 G^{n−1}                    (Durran 2010, eq. 2.132)
        û       = u^n + Δt G_u^{n+½},   T^{n+1} = T^n + Δt G_T^{n+½}
        (η^{n+1}, u^{n+1}) = θ-method free-surface step from (η^n, u^n, û)
                                                          (Marshall et al. 1997, §2)

    ``tendency_fn`` must therefore **exclude** the barotropic pressure
    gradient −g∇η: compute the hydrostatic pressure with η = 0 (baroclinic
    part only).  Its η tendency is ignored — continuity is enforced by the
    free-surface solver.  The diagnostic w is carried over unchanged; update
    it with ``core.dynamics.momentum.diagnose_w`` if needed.

    Parameters
    ----------
    state : ModelState
        State at time n.
    tendency_fn : callable
        ModelState → ModelState tendency (without −g∇η).
    dt : float
        Time step [s]; must equal ``free_surface.dt``.
    free_surface : ImplicitFreeSurface
        Pre-assembled operator for this grid and Δt.
    tendency_prev : ModelState or None
        G^{n−1}; None on the first step.

    Returns
    -------
    state_new, tendency_curr
    """
    if abs(dt - free_surface.dt) > 1e-12 * dt:
        raise ValueError(f"dt={dt} differs from the free-surface operator dt={free_surface.dt}")

    tendency_curr: ModelState = tendency_fn(state)
    if tendency_prev is None:
        g_ab = tendency_curr
    else:
        g_ab = 1.5 * tendency_curr + (-0.5) * tendency_prev

    u_hat = state.u.values + dt * g_ab.u.values
    v_hat = state.v.values + dt * g_ab.v.values
    eta_new, u_new, v_new = free_surface.step(
        state.eta.values, state.u.values, state.v.values, u_hat, v_hat)

    state_new = ModelState(
        u=state.u.copy(data=u_new),
        v=state.v.copy(data=v_new),
        w=state.w.copy(),
        T=state.T.copy(data=state.T.values + dt * g_ab.T.values),
        S=state.S.copy(data=state.S.values + dt * g_ab.S.values),
        eta=state.eta.copy(data=eta_new),
        time=state.time + dt,
    )
    return state_new, tendency_curr
