"""
Near-inertial stability of the ALE step with the time-centred baroclinic transport θ_bc = ½.

Root cause (docs/design/theta_bc_inertial/root_cause.md, 2026-11-02, class a): the vector-invariant
Coriolis PV flux q·(vh, −uh) used the layer transports stored by the previous step's continuity,
which carry θ_bc uⁿ + (1−θ_bc) uⁿ⁻¹.  AB3 then places Coriolis at n + θ_bc − ½ (forward Euler at
θ_bc = ½) and a near-inertial mode grows at σ = (1−θ_bc) f² Δt.  The fix builds the PV-flux transport
from the current state, Q = L uⁿ h̄ⁿ, every step (MOM6 recomputes Coriolis with the predicted
transports in its corrector, MOM_dynamics_split_RK2.F90:793, :903, :968, git 0e9a561).

Tests:
* a doubly periodic f-plane box with a baroclinic near-inertial oscillation must not grow;
* the vector-invariant terms must not depend on the stored (lagged) transports ``st.Qx, st.Qy``;
* the linear amplification matrix of the AB3 + θ_bc continuity reduced-gravity f-plane mode with
  Coriolis on uⁿ has spectral radius ≤ 1 (the lagged variant does not: documents the bug);
* the ALE flux-form momentum path (Coriolis on uⁿ, unaffected by the fix) stays stable.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid

F0 = 1e-4          # Coriolis parameter f [s⁻¹]
NX, NZ = 16, 4     # horizontal cells per side, layers
DX = 20e3          # grid spacing Δx [m]
H0 = 1000.0        # depth H [m]
U0 = 0.05          # initial baroclinic velocity amplitude [m s⁻¹]


def _box(theta_bc: float, f_dt: float, momentum: str = "vector_invariant") -> SigmaBaroclinic:
    """Doubly periodic f-plane box, ALE z*, linear EOS, no closure, kv = kh = ah = 0, FV PGF."""
    dt = f_dt / F0
    hg = create_cartesian_grid(nx=NX, ny=NX, dx=DX, dy=DX, f0=F0)
    return SigmaBaroclinic(hg, np.full((NX, NX), H0), dt=dt, n_layers=NZ, periodic=("x", "y"), rho0=1000.0,
                           t_coef=0.2, z0=None, closure="none", theta=0.6, d_min=0.05,
                           vertical_coordinate="ale", regrid="zstar",
                           regrid_params=dict(dz_top=np.full(NZ, H0 / NZ)), pgf="fv", theta_bc=theta_bc,
                           kh=0.0, kv=0.0, ah=0.0, mix_diagnostic=False, momentum=momentum)


def _initial(m: SigmaBaroclinic):
    """Stable stratification, u = U₀ cos(2πy/L) p(z) with p zero-mean and top-intensified, v = 0."""
    T = (10.0 + 10.0 * (np.arange(NZ) + 0.5) / NZ)[:, None, None] * np.ones((NZ, NX, NX))
    st = m.initial_state(np.zeros((NX, NX)), T)
    prof = np.full(NZ, -1.0 / (NZ - 1))
    prof[-1] = 1.0                                               # k = NZ−1 is the top layer
    yc = (np.arange(NX) + 0.5) * DX
    st.u[:] = U0 * prof[:, None, None] * np.cos(2 * np.pi * yc / (NX * DX))[None, :, None]
    return st


def _growth_per_day(theta_bc: float, f_dt: float, days: float = 8.0, momentum: str = "vector_invariant") -> float:
    """Growth rate [d⁻¹] of the top-layer rms speed: straight-line fit of its log over t > 0.3 t_end."""
    m = _box(theta_bc, f_dt, momentum)
    st = _initial(m)
    nsteps = int(round(days * 86400 * F0 / f_dt))
    t, amp = [], []
    for n in range(nsteps):
        st = m.step(st)
        if n % max(1, nsteps // 400) == 0:
            a = np.sqrt(np.mean(st.u[-1, :, :-1] ** 2) + np.mean(st.v[-1, :-1] ** 2))
            assert np.isfinite(a)
            t.append(st.time)
            amp.append(a)
    t, amp = np.array(t), np.array(amp)
    sel = t > 0.3 * t[-1]
    return float(np.polyfit(t[sel], np.log(amp[sel]), 1)[0] * 86400.0)


@pytest.mark.parametrize("f_dt", [0.12, 0.054])
def test_near_inertial_mode_does_not_grow_at_theta_bc_half(f_dt: float) -> None:
    """Before the fix: +0.18 d⁻¹ (fΔt 0.12) and +0.03 d⁻¹ (0.054) over 8 days."""
    assert _growth_per_day(0.5, f_dt) <= 0.0


def test_flux_form_ale_box_is_stable() -> None:
    """The flux-form path (Coriolis on uⁿ, not touched by the fix) stays non-growing."""
    assert _growth_per_day(0.5, 0.12, momentum="flux") <= 0.0


def test_vector_invariant_terms_ignore_stored_transports() -> None:
    """The PV-flux transport is built from the current (u, v, h): the lagged st.Qx, st.Qy do not enter."""
    m = _box(0.5, 0.12)
    st = _initial(m)
    st = m.step(st)
    st = m.step(st)                                              # st.Qx, st.Qy now hold the lagged transports
    mom = m._momentum
    cu0, cv0 = mom.vector_invariant(st)
    st.Qx, st.Qy = 7.0 * st.Qx + 1.0, -3.0 * st.Qy + 2.0        # arbitrary stored transports
    cu1, cv1 = mom.vector_invariant(st)
    assert np.array_equal(cu0, cu1) and np.array_equal(cv0, cv1)


# --------------------------------------------------------------------------------------------------
# Linear stability: one Fourier mode of the reduced-gravity f-plane problem, non-dimensional
# F = f Δt, a = c k Δt, h scaled by H/c (root_cause.md, "Evidencia"):
#     u^{n+1} = u^n + AB3[ F v^c − i a h ],   v^{n+1} = v^n + AB3[ −F u^c ]
#     h^{n+1} = h^n − i a [ θ_bc u^{n+1} + (1 − θ_bc) u^n ]
# with the Coriolis velocity u^c = uⁿ (fixed scheme) or θ_bc uⁿ + (1−θ_bc) uⁿ⁻¹ (lagged transport).
# AB3 weights (23, −16, 5)/12 (Durran 2010 §2.4).  a ≤ 1 keeps the gravity mode inside the
# forward–backward CFL limit (≈ 1.05–1.3 for θ_bc ∈ [½, 1]); the instability under test is inertial.
# --------------------------------------------------------------------------------------------------
_AB3 = np.array([23.0, -16.0, 5.0]) / 12.0
_NL = 5                                                          # time levels kept: n … n−4


def _amplification_matrix(F: float, a: float, theta_bc: float, lagged: bool) -> np.ndarray:
    """Companion matrix of the state (u, v, h) at levels n … n−4 → n+1 … n−3."""
    nv = 3 * _NL
    M = np.zeros((nv, nv), complex)

    def idx(lvl: int, var: int) -> int:
        return 3 * lvl + var                                      # var 0 u, 1 v, 2 h

    for j, wj in enumerate(_AB3):                                # tendency G^{n−j}
        cw = [(j, theta_bc), (j + 1, 1.0 - theta_bc)] if lagged else [(j, 1.0)]
        for lvl, c in cw:
            M[0, idx(lvl, 1)] += wj * F * c
            M[1, idx(lvl, 0)] += -wj * F * c
        M[0, idx(j, 2)] += -1j * a * wj
    M[0, idx(0, 0)] += 1.0
    M[1, idx(0, 1)] += 1.0
    M[2, :] = -1j * a * theta_bc * M[0, :]
    M[2, idx(0, 2)] += 1.0
    M[2, idx(0, 0)] += -1j * a * (1.0 - theta_bc)
    M[3:, :-3] = np.eye(nv - 3)
    return M


def _max_radius(theta_bc: float, lagged: bool) -> float:
    return max(max(abs(np.linalg.eigvals(_amplification_matrix(F, a, theta_bc, lagged))))
               for F in np.linspace(0.0, 0.3, 16) for a in np.linspace(0.0, 1.0, 21))


@pytest.mark.parametrize("theta_bc", [0.5, 0.6, 1.0])
def test_linear_amplification_with_current_coriolis_is_stable(theta_bc: float) -> None:
    assert _max_radius(theta_bc, lagged=False) <= 1.0 + 1e-12


@pytest.mark.parametrize("theta_bc", [0.5, 0.6])
def test_linear_amplification_with_lagged_transport_grows(theta_bc: float) -> None:
    """The bug as a property of the scheme: the lagged transport makes θ_bc < 1 unstable."""
    assert _max_radius(theta_bc, lagged=True) > 1.0 + 1e-6
