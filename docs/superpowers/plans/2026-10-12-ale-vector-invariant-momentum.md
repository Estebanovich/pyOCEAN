# Vector-Invariant Momentum for the ALE Mode — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the flux-form momentum advection and the separate Coriolis term by MOM6's vector-invariant form (Sadourny 1975 energy PV flux with BOUND_CORIOLIS, minus the kinetic-energy gradient) in the ALE mode only, add a CFL velocity truncation backstop, and show the hybrid lock exchange forms a Benjamin head like MOM6.

**Architecture:** A new module `core/dynamics/vector_invariant.py` holds pure array functions (vorticity at corners, PV, Sadourny PV flux with the clip, kinetic energy and its gradient, CFL truncation). `SigmaBaroclinic` gains `momentum`, `ke_scheme`, `max_truncations`; its `_tendencies` calls the new terms when `vertical_coordinate="ale"`, and `_step_ale` truncates at the end of the step. The σ Eulerian path is untouched.

**Tech Stack:** Python 3, NumPy 1.26.4, pytest. No Numba needed in this plan (vectorised NumPy); optimisation only if the measured cost demands it later.

**Spec:** `docs/superpowers/specs/2026-10-12-ale-vector-invariant-momentum-design.md` (parent: `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md`).

## Global Constraints

- Never use the 6 efficiency cores: run with `NUMBA_NUM_THREADS=5`; one heavy run at a time.
- Do not install packages; numpy stays 1.26.4.
- Commit only with Esteban's authorisation; never stage `pyOCEAN-literature-review-guide.docx`, `log_*.txt`, `*.npz`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn`.
- The σ Eulerian path must stay bit-identical (whole `tests/test_sigma_baroclinic.py`, `tests/test_restart.py`, `tests/test_sigma_wd.py` unchanged and green).
- Every formula cites MOM6 `file:line` (commit 0e9a561) in its docstring.
- Do not overwrite `validation/seamount/out/summary_*.txt` with exploratory runs (use the scratchpad).
- Spec acceptance (§6): hybrid lock exchange with MOM6 viscosity (KV = 1e-4, KH = 0.01): fronts at 1 m and 5 m within ±10 % of Benjamin; max|u| ≤ 0.84 m/s; RPE gain < 0.2 × ALE-σ; zero truncations; Σh = H + η and heat conserved. No regression: ALE-σ lock at Benjamin ±10 %, internal wave within 10 % of σ Eulerian, z\* step at rest ≤ 1e-10 m/s, z\* seamount (FV + ρ_ref + Coriolis) ≤ 0.15 cm/s at day 10.
- Plan B (spec §6) is NOT implemented in this plan: if the acceptance fails, stop and report to Esteban.

## Review Focus

1. Coastlines in a 2-D domain (land cells inside, ny > 1): q finite, ζ = 0 at coast corners, zero tendency on land faces.
2. Stretched grids (non-uniform Δx, Δy): the PV flux must still do no work and the KE gradient must use the local metrics.
3. Columns that dry in ALE-σ (h = 0): q uses the negligible volume, no NaN (the existing beach test runs with the new default).
4. The first step, when no transports exist yet (`st.Qx is None`) and the initial velocity is not zero: the fallback L·u·h̄ must be used.
5. Truncation of v (applied on swapped axes): the donor-cell width must be Δy of the right cell.

---

### Task 1: Vector-invariant terms (`core/dynamics/vector_invariant.py`)

**Files:**
- Create: `core/dynamics/vector_invariant.py`
- Test: `tests/test_vector_invariant.py`

**Interfaces:**
- Produces:
  - `V_NEGLECT: float` (= 1e-28 m³)
  - `corner_mask(active: ndarray (ny,nx) bool) -> ndarray (ny+1,nx+1) bool`
  - `corner_area(dx_u: (ny,nx+1), dy_v: (ny+1,nx)) -> (ny+1,nx+1)`
  - `vorticity_q(u (N,ny,nx+1), v (N,ny+1,nx), dx_u, dy_v, area_q, mask_q) -> (N,ny+1,nx+1)`
  - `pv_q(zeta (N,ny+1,nx+1), f_q (ny+1,nx+1), h (N,ny,nx), area_t (ny,nx), active (ny,nx)) -> (N,ny+1,nx+1)`
  - `coriolis_pv(q, abs_vort, Qx (N,ny,nx+1), Qy (N,ny+1,nx), u, v, dx_u, dy_v, bound: bool = True) -> (cu (N,ny,nx+1), cv (N,ny+1,nx))`
  - `kinetic_energy(u, v, area_u (ny,nx+1), area_v (ny+1,nx), area_t, scheme: str = "arakawa") -> (N,ny,nx)`
  - `ke_gradient(K (N,ny,nx), dx_u, dy_v) -> (−∂K/∂x (N,ny,nx+1), −∂K/∂y (N,ny+1,nx))`, zero on domain-edge faces.

- [ ] **Step 1: Write the failing tests**

Create `tests/test_vector_invariant.py`:

```python
"""
Tests for core/dynamics/vector_invariant.py — vector-invariant momentum terms of the ALE mode
(spec docs/superpowers/specs/2026-10-12-ale-vector-invariant-momentum-design.md §3, §6).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics import vector_invariant as vi
from core.grid.horizontal import create_cartesian_grid


def _grid(nx=6, ny=5, dx=1000.0, dy=800.0, f0=1e-4):
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dy, f0=f0)
    g = {k: np.asarray(hg[k].values, float) for k in ("dx_u", "dy_v", "dx_v", "dy_u", "area_t", "area_u",
                                                       "area_v", "f_q", "x_u", "y_u", "x_v", "y_v")}
    return hg, g


def test_f_plane_uniform_flow():
    # spec §6.1: uniform u, uniform h, ζ = 0 → CAu = 0, CAv = −f u, ∇K = 0
    _, g = _grid()
    n, ny, nx = 2, 5, 6
    act = np.ones((ny, nx), bool)
    u = np.full((n, ny, nx + 1), 0.3)
    v = np.zeros((n, ny + 1, nx))
    h = np.full((n, ny, nx), 10.0)
    zeta = vi.vorticity_q(u, v, g["dx_u"], g["dy_v"], vi.corner_area(g["dx_u"], g["dy_v"]), vi.corner_mask(act))
    np.testing.assert_array_equal(zeta, 0.0)
    q = vi.pv_q(zeta, g["f_q"], h, g["area_t"], act)
    Qx, Qy = g["dy_u"][None] * u * 10.0, g["dx_v"][None] * v * 10.0
    cu, cv = vi.coriolis_pv(q, g["f_q"][None] + zeta, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound=False)
    np.testing.assert_allclose(cu, 0.0, atol=1e-18)
    np.testing.assert_allclose(cv[:, 1:-1, :], -1e-4 * 0.3, rtol=1e-12)
    K = vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"])
    kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
    np.testing.assert_allclose(K, 0.5 * 0.3**2, rtol=1e-12)
    np.testing.assert_allclose(kx, 0.0, atol=1e-18)
    np.testing.assert_allclose(ky, 0.0, atol=1e-18)


def test_pv_flux_equals_the_energy_conserving_coriolis_of_the_sigma_model():
    # spec §6.2: h uniform, ζ = 0, transports L u h → identical to SigmaBaroclinic._coriolis
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    hg, g = _grid(dx=1000.0, dy=1000.0)
    n, ny, nx = 3, 5, 6
    m = SigmaBaroclinic(hg, np.full((ny, nx), 20.0), dt=10.0, n_layers=n)
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(n, ny, nx + 1)), rng.normal(size=(n, ny + 1, nx))
    u[..., [0, -1]] = 0.0
    v[:, [0, -1], :] = 0.0
    h = np.full((n, ny, nx), 7.0)
    act = np.ones((ny, nx), bool)
    q = vi.pv_q(np.zeros((n, ny + 1, nx + 1)), g["f_q"], h, g["area_t"], act)
    cu, cv = vi.coriolis_pv(q, np.broadcast_to(g["f_q"], q.shape), g["dy_u"][None] * u * 7.0,
                            g["dx_v"][None] * v * 7.0, u, v, g["dx_u"], g["dy_v"], bound=False)
    cu0, cv0 = m._coriolis(u, v)
    np.testing.assert_allclose(cu, cu0, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(cv, cv0, rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize("stretched", [False, True])
def test_sadourny_energy_pv_flux_does_no_work(stretched):
    # spec §6.3 + review focus 2: Σ (Qx CAu Δx_u + Qy CAv Δy_v) = 0 for any q, closed domain
    n, ny, nx = 2, 5, 7
    _, g = _grid(nx=nx, ny=ny)
    rng = np.random.default_rng(1)
    if stretched:
        g["dx_u"] = g["dx_u"] * rng.uniform(0.5, 1.5, g["dx_u"].shape)
        g["dy_v"] = g["dy_v"] * rng.uniform(0.5, 1.5, g["dy_v"].shape)
    q = rng.normal(size=(n, ny + 1, nx + 1))
    Qx, Qy = rng.normal(size=(n, ny, nx + 1)), rng.normal(size=(n, ny + 1, nx))
    Qx[..., [0, -1]] = 0.0
    Qy[:, [0, -1], :] = 0.0
    cu, cv = vi.coriolis_pv(q, q, Qx, Qy, Qx, Qy, g["dx_u"], g["dy_v"], bound=False)
    work = (Qx * cu * g["dx_u"][None]).sum() + (Qy * cv * g["dy_v"][None]).sum()
    assert abs(work) < 1e-12 * np.abs(Qx * cu * g["dx_u"][None]).sum()


def test_solid_body_rotation_vorticity_and_free_slip_coast():
    # spec §6.4 + review focus 1
    _, g = _grid(nx=8, ny=7, f0=0.0)
    n, ny, nx = 1, 7, 8
    om = 1e-5
    u = (-om * g["y_u"])[None]
    v = (om * g["x_v"])[None]
    act = np.ones((ny, nx), bool)
    act[3, 4] = False                                                   # an island
    mq = vi.corner_mask(act)
    zeta = vi.vorticity_q(u, v, g["dx_u"], g["dy_v"], vi.corner_area(g["dx_u"], g["dy_v"]), mq)
    np.testing.assert_allclose(zeta[0][mq], 2 * om, rtol=1e-10)
    np.testing.assert_array_equal(zeta[0][~mq], 0.0)
    assert not mq[3:5, 4:6].any() and not mq[0].any() and not mq[:, -1].any()


def test_kinetic_energy_schemes():
    # spec §6.5: K = u²/2 for uniform u; Godunov uses only inflow velocities
    _, g = _grid()
    n, ny, nx = 1, 5, 6
    u = np.full((n, ny, nx + 1), -0.4)
    v = np.zeros((n, ny + 1, nx))
    for s in ("arakawa", "gudonov"):
        np.testing.assert_allclose(vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"], s), 0.08, rtol=1e-12)
    u = np.zeros((n, ny, nx + 1))
    u[0, 2, 3] = 1.0                                                    # leaves cell (2, 2) east, enters (2, 3) west
    K = vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"], "gudonov")
    assert K[0, 2, 2] == 0.0 and K[0, 2, 3] > 0.0


def test_bound_coriolis_caps_the_pv_flux_of_a_vanished_layer():
    # spec §6.6: q·vh ≈ (f+ζ) v h_face/h_q explodes when h_q ≪ h_face; the clip caps it at max (f+ζ) v
    _, g = _grid(nx=4, ny=4)
    n, ny, nx = 1, 4, 4
    act = np.ones((ny, nx), bool)
    h = np.full((n, ny, nx), 1e-3)                                      # vanished layer
    u = np.zeros((n, ny, nx + 1))
    v = np.full((n, ny + 1, nx), 0.1)
    v[:, [0, -1], :] = 0.0
    zeta = np.zeros((n, ny + 1, nx + 1))
    q = vi.pv_q(zeta, g["f_q"], h, g["area_t"], act)
    Qy = g["dx_v"][None] * v * 10.0                                     # transport of the thick upwind layer
    Qx = np.zeros_like(u)
    cu0, _ = vi.coriolis_pv(q, g["f_q"][None] + zeta, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound=False)
    cu, _ = vi.coriolis_pv(q, g["f_q"][None] + zeta, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound=True)
    assert np.abs(cu0).max() > 100 * 1e-4 * 0.1
    assert np.abs(cu).max() <= 1e-4 * 0.1 * (1 + 1e-12)
    h0 = np.zeros_like(h)                                               # empty cells: no NaN
    assert np.isfinite(vi.pv_q(zeta, g["f_q"], h0, g["area_t"], act)).all()
```

Note on the stretched case: `create_cartesian_grid` is uniform; the test perturbs `dx_u`, `dy_v` directly (the identity is purely algebraic).

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_vector_invariant.py -q`
Expected: collection error `ModuleNotFoundError: No module named 'core.dynamics.vector_invariant'`.

- [ ] **Step 3: Implement**

Create `core/dynamics/vector_invariant.py`:

```python
"""
Vector-invariant momentum terms for the ALE mode of the stratified model (spec
docs/superpowers/specs/2026-10-12-ale-vector-invariant-momentum-design.md §3).

    ∂u/∂t = q (vh)‾ − ∂K/∂x + …,    ∂v/∂t = −q (uh)‾ − ∂K/∂y + …,    q = (f + ζ)/h_q,  K = ½|u|²

on the Arakawa C grid, as MOM6 (src/core/MOM_CoriolisAdv.F90, commit 0e9a561).  The flux form
divides by the volume of the momentum cell and accelerates the nose of a gravity current in a
layer that thins (hybrid lock exchange, validation/mom6_lock_exchange); Bleck & Smith (1990,
app. B) and Higdon (2005) describe the problem; MOM6 and HYCOM use this form.

Shapes: u (N, ny, nx+1), v (N, ny+1, nx), h (N, ny, nx), corners q (N, ny+1, nx+1); corner
[J, I] is shared by cells (J−1, J) × (I−1, I).

References: Sadourny, R. (1975). J. Atmos. Sci. 32, 680–689.  Arakawa, A. & Lamb, V. R. (1981).
Mon. Wea. Rev. 109, 18–36.  Bleck, R. & Smith, L. T. (1990). JGR 95, 3273–3285.
"""

from __future__ import annotations

import numpy as np

V_NEGLECT = 1e-20 * 1e-8   # [m³] negligible volume in q: H_subroundoff (1e-4 m)² (MOM_CoriolisAdv.F90:337)


def corner_mask(active: np.ndarray) -> np.ndarray:
    """Corners (ny+1, nx+1) surrounded by four wet cells (MOM6 mask2dBu); outside the domain is land."""
    a = np.pad(np.asarray(active, bool), 1, constant_values=False)
    return a[:-1, :-1] & a[:-1, 1:] & a[1:, :-1] & a[1:, 1:]


def corner_area(dx_u: np.ndarray, dy_v: np.ndarray) -> np.ndarray:
    """Area of the corner cells [m²]: Δx from dx_u averaged in y, Δy from dy_v averaged in x (edges: one side)."""
    dxp = np.concatenate([dx_u[:1], dx_u, dx_u[-1:]], axis=0)
    dyp = np.concatenate([dy_v[:, :1], dy_v, dy_v[:, -1:]], axis=1)
    return 0.5 * (dxp[:-1] + dxp[1:]) * 0.5 * (dyp[:, :-1] + dyp[:, 1:])


def vorticity_q(u: np.ndarray, v: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray, area_q: np.ndarray,
                mask_q: np.ndarray) -> np.ndarray:
    """
    Relative vorticity at the corners [s⁻¹] in circulation form, ζ = (δx(v Δy) − δy(u Δx))/A_q,
    free slip: 0 at corners touching land or the domain edge (MOM_CoriolisAdv.F90:595-610).
    """
    n, ny, nx1 = u.shape
    vdy = v * dy_v[None]
    udx = u * dx_u[None]
    dvdx = np.zeros((n, ny + 1, nx1))
    dudy = np.zeros((n, ny + 1, nx1))
    dvdx[:, :, 1:-1] = vdy[:, :, 1:] - vdy[:, :, :-1]
    dudy[:, 1:-1, :] = udx[:, 1:, :] - udx[:, :-1, :]
    return np.where(mask_q[None], (dvdx - dudy) / area_q[None], 0.0)


def _corner_sum(x: np.ndarray) -> np.ndarray:
    pad = [(0, 0)] * (x.ndim - 2) + [(1, 1), (1, 1)]
    p = np.pad(x, pad)
    return p[..., :-1, :-1] + p[..., :-1, 1:] + p[..., 1:, :-1] + p[..., 1:, 1:]


def pv_q(zeta: np.ndarray, f_q: np.ndarray, h: np.ndarray, area_t: np.ndarray, active: np.ndarray) -> np.ndarray:
    """
    Potential vorticity at the corners [m⁻¹ s⁻¹], q = (f + ζ) Σ A/(Σ A h + V_neglect) over the wet
    cells around the corner (MOM_CoriolisAdv.F90:337, 364-367, 429-437, 625-629).
    """
    A = np.where(active, area_t, 0.0)
    Ah = np.where(active[None], np.nan_to_num(h), 0.0) * A[None]
    return (f_q[None] + zeta) * _corner_sum(A)[None] / (_corner_sum(Ah) + V_NEGLECT)


def coriolis_pv(q: np.ndarray, abs_vort: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, u: np.ndarray,
                v: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray, bound: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Sadourny (1975) energy-conserving PV flux [m s⁻²] (MOM_CoriolisAdv.F90:818-823, 1071-1076):

        CAu[j,I] =  ¼/Δx_u (q[j+1,I](Qy[j+1,I−1] + Qy[j+1,I]) + q[j,I](Qy[j,I−1] + Qy[j,I]))
        CAv[J,i] = −¼/Δy_v (q[J,i](Qx[J−1,i] + Qx[J,i]) + q[J,i+1](Qx[J−1,i+1] + Qx[J,i+1]))

    Qx, Qy: layer volume transports [m³ s⁻¹].  With ``bound`` (BOUND_CORIOLIS, :1008-1023,
    1260-1275) CAu is clipped to the range of (f+ζ)·v at the four neighbouring v-faces (its two
    corners) and CAv to that of −(f+ζ)·u: q·vh ≈ (f+ζ) v h_face/h_q grows when h_q ≪ h_face.
    Domain-edge faces: 0.
    """
    cu = np.zeros_like(Qx)
    cv = np.zeros_like(Qy)
    qN, qS = q[:, 1:, 1:-1], q[:, :-1, 1:-1]
    cu[..., 1:-1] = 0.25 / dx_u[None, :, 1:-1] * (qN * (Qy[:, 1:, :-1] + Qy[:, 1:, 1:])
                                                   + qS * (Qy[:, :-1, :-1] + Qy[:, :-1, 1:]))
    qW, qE = q[:, 1:-1, :-1], q[:, 1:-1, 1:]
    cv[:, 1:-1, :] = -0.25 / dy_v[None, 1:-1, :] * (qW * (Qx[:, :-1, :-1] + Qx[:, 1:, :-1])
                                                    + qE * (Qx[:, :-1, 1:] + Qx[:, 1:, 1:]))
    if bound:
        aN, aS = abs_vort[:, 1:, 1:-1], abs_vort[:, :-1, 1:-1]
        c = np.stack([aN * v[:, 1:, 1:], aN * v[:, 1:, :-1], aS * v[:, :-1, 1:], aS * v[:, :-1, :-1]])
        cu[..., 1:-1] = np.clip(cu[..., 1:-1], c.min(axis=0), c.max(axis=0))
        aW, aE = abs_vort[:, 1:-1, :-1], abs_vort[:, 1:-1, 1:]
        c = np.stack([-aW * u[:, :-1, :-1], -aW * u[:, 1:, :-1], -aE * u[:, :-1, 1:], -aE * u[:, 1:, 1:]])
        cv[:, 1:-1, :] = np.clip(cv[:, 1:-1, :], c.min(axis=0), c.max(axis=0))
    return cu, cv


def kinetic_energy(u: np.ndarray, v: np.ndarray, area_u: np.ndarray, area_v: np.ndarray, area_t: np.ndarray,
                   scheme: str = "arakawa") -> np.ndarray:
    """
    Kinetic energy at cell centres [m² s⁻²] (MOM6 gradKE, MOM_CoriolisAdv.F90:1377-1535).
    'arakawa': K = ¼(A_u u_E² + A_u u_W² + A_v v_N² + A_v v_S²)/A_T.
    'gudonov': K = ½(max(u_m² A_uE, u_p² A_uW) + max(v_m² A_vN, v_p² A_vS))/A_T with u_p = max(u_W, 0),
    u_m = min(u_E, 0) (only the velocities that enter the cell).
    """
    aE, aW = area_u[None, :, 1:], area_u[None, :, :-1]
    aN, aS = area_v[None, 1:], area_v[None, :-1]
    if scheme == "arakawa":
        return 0.25 * (aE * u[..., 1:] ** 2 + aW * u[..., :-1] ** 2 + aN * v[:, 1:] ** 2 + aS * v[:, :-1] ** 2) / area_t[None]
    if scheme == "gudonov":
        up, um = np.maximum(u[..., :-1], 0.0), np.minimum(u[..., 1:], 0.0)
        vp, vm = np.maximum(v[:, :-1], 0.0), np.minimum(v[:, 1:], 0.0)
        return 0.5 * (np.maximum(um**2 * aE, up**2 * aW) + np.maximum(vm**2 * aN, vp**2 * aS)) / area_t[None]
    raise ValueError(f"unknown ke_scheme {scheme!r}; use 'arakawa' or 'gudonov'")


def ke_gradient(K: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """−∂K/∂x at the u-faces and −∂K/∂y at the v-faces [m s⁻²]; domain-edge faces 0."""
    n, ny, nx = K.shape
    gx = np.zeros((n, ny, nx + 1))
    gy = np.zeros((n, ny + 1, nx))
    gx[..., 1:-1] = -(K[..., 1:] - K[..., :-1]) / dx_u[None, :, 1:-1]
    gy[:, 1:-1, :] = -(K[:, 1:] - K[:, :-1]) / dy_v[None, 1:-1, :]
    return gx, gy
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_vector_invariant.py -q`
Expected: 7 passed. If `test_solid_body_rotation_vorticity_and_free_slip_coast` fails on the corner-area of a stretched edge, check `corner_area` against `dx_u`/`dy_v` of `create_cartesian_grid` (uniform here, so it must be exact).

- [ ] **Step 5: Commit (with Esteban's authorisation)**

```bash
git add core/dynamics/vector_invariant.py tests/test_vector_invariant.py
git commit -m "Vector-invariant momentum terms (MOM6 Sadourny PV flux, BOUND_CORIOLIS, KE gradient)"
```

---

### Task 2: CFL velocity truncation and the viscous-coupling guard

**Files:**
- Modify: `core/dynamics/vector_invariant.py` (add `truncate_cfl`)
- Test: `tests/test_vector_invariant.py`

**Interfaces:**
- Produces: `truncate_cfl(u (N,ny,nx+1), dt: float, dx_t (ny,nx), h_face (N,ny,nx+1), h_count: float, cfl: float = 0.5) -> (u_new, n_counted: int, first: tuple[int,int,int] | None)`. For v, call on swapped axes with `dy_t.T`.

- [ ] **Step 1: Write the tests**

Append to `tests/test_vector_invariant.py`:

```python
def test_cfl_truncation_caps_the_velocity_and_counts_only_thick_faces():
    # spec §6.7 (MOM6 vertvisc_limit_vel, MOM_vert_friction.F90:2555-2715)
    dt, dx = 10.0, 100.0
    u = np.zeros((2, 1, 4))
    u[0, 0, 1] = 7.0                                                     # Courant 0.7 → 0.45 Δx/Δt = 4.5 m/s
    u[1, 0, 2] = -7.0
    h_face = np.ones_like(u)
    h_face[1] = 1e-3                                                     # thin layer: truncated, not counted
    dx_t = np.full((1, 3), dx)
    out, n, first = vi.truncate_cfl(u, dt, dx_t, h_face, 0.01)
    assert out[0, 0, 1] == pytest.approx(4.5) and out[1, 0, 2] == pytest.approx(-4.5)
    assert n == 1 and first == (0, 0, 1)
    same, n0, f0 = vi.truncate_cfl(np.full_like(u, 4.0), dt, dx_t, h_face, 0.01)
    np.testing.assert_array_equal(same, 4.0)
    assert n0 == 0 and f0 is None


def test_cfl_truncation_uses_the_donor_cell_width():
    # review focus 5: u > 0 takes the west cell, u < 0 the east cell
    dt = 10.0
    dx_t = np.array([[100.0, 1000.0]])
    u = np.array([[[0.0, 7.0, 0.0]]])                                   # west cell 100 m: Courant 0.7
    out, _, _ = vi.truncate_cfl(u, dt, dx_t, np.ones_like(u), 0.01)
    assert out[0, 0, 1] == pytest.approx(4.5)
    u = np.array([[[0.0, -7.0, 0.0]]])                                  # east cell 1000 m: Courant 0.07
    out, _, _ = vi.truncate_cfl(u, dt, dx_t, np.ones_like(u), 0.01)
    assert out[0, 0, 1] == -7.0


def test_implicit_viscosity_slaves_a_vanishing_layer_to_its_neighbours():
    # spec §6.8 (guard of existing behaviour: passes before any change).  MOM6 form
    # (MOM_vert_friction.F90:711-783): dz_k u_k + Δt ν/(½(dz_k+dz_k±1)) coupling; with ν = 1e-4,
    # Δt = 20 s and 1 m neighbours a layer thinner than ~4Δtν/H ≈ 8 mm loses its own tendency.
    # Ruling (plan): the spec's "1 mm" is at the threshold (the layer only moves half way); 0.1 mm
    # is well below it.
    from core.dynamics.sigma_wd import vertical_implicit
    u = np.array([[0.5], [5.0], [0.5]])                                 # thin middle layer with a large tendency
    dz = np.array([[1.0], [1e-4], [1.0]])
    out = vertical_implicit(u, dz, np.full((4, 1), 1e-4), np.zeros(1), 20.0)
    assert abs(out[1, 0] - 0.5) < 0.1
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_vector_invariant.py -q`
Expected: the two truncation tests FAIL with `AttributeError: ... has no attribute 'truncate_cfl'`; the viscosity guard PASSES (existing behaviour).

- [ ] **Step 3: Implement**

Append to `core/dynamics/vector_invariant.py`:

```python
def truncate_cfl(u: np.ndarray, dt: float, dx_t: np.ndarray, h_face: np.ndarray, h_count: float,
                 cfl: float = 0.5) -> tuple[np.ndarray, int, tuple[int, int, int] | None]:
    """
    CFL truncation of the layer velocities, a backstop that should not act (MOM6
    vertvisc_limit_vel, MOM_vert_friction.F90:2555-2715): where |u| Δt/Δx_donor > cfl,
    u = ±0.9 cfl Δx_donor/Δt, Δx_donor the width of the cell the water comes from (west for
    u ≥ 0).  u (N, ny, nx+1), dx_t (ny, nx) [m], h_face (N, ny, nx+1) [m].  Truncations at faces
    thicker than h_count are counted (MOM6 counts h_u > 3 Å, :2584, 2619).  Returns (u, count,
    first counted (k, j, i) or None).  For v call it on swapped axes with dy_t.T.
    """
    dxW = np.concatenate([dx_t[:, :1], dx_t], axis=1)[None]
    dxE = np.concatenate([dx_t, dx_t[:, -1:]], axis=1)[None]
    dx_d = np.where(u >= 0, dxW, dxE)
    over = np.abs(u) * dt / dx_d > cfl
    out = np.where(over, np.sign(u) * 0.9 * cfl * dx_d / dt, u)
    counted = over & (h_face > h_count)
    first = tuple(int(a[0]) for a in np.nonzero(counted)) if counted.any() else None
    return out, int(counted.sum()), first
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_vector_invariant.py -q`
Expected: 10 passed.

- [ ] **Step 5: Commit (with Esteban's authorisation)**

```bash
git add core/dynamics/vector_invariant.py tests/test_vector_invariant.py
git commit -m "CFL velocity truncation backstop (MOM6 vertvisc_limit_vel)"
```

---

### Task 3: Vector-invariant momentum in the ALE model

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (import; `SigmaState` `n_trunc`; constructor and docstring; `_setup_ale`; new `_vector_invariant`; `_tendencies`; end of `_step_ale`)
- Modify: `tests/test_sigma_ale.py` (`_ale_box` gains `nu=1e-6`; new tests)

**Interfaces:**
- Consumes: Task 1 and Task 2 functions; `self._mean_face_thickness(h) -> (hu (N,ny,nx+1), hv (N,ny+1,nx))` (existing).
- Produces: `SigmaBaroclinic(..., momentum: str | None = None, ke_scheme: str = "arakawa", max_truncations: int = 1000)`; attributes `self.momentum`, `self.ke_scheme`, `self.max_truncations`; `SigmaState.n_trunc: int = 0`; `_ale_box(..., nu=1e-6, ...)`.

- [ ] **Step 1: Write the failing tests**

In `tests/test_sigma_ale.py` change `_ale_box` to take `nu=1e-6` and pass `nu_const=nu` (instead of the literal `1e-6`):

```python
def _ale_box(rule="zstar", nx=32, nz=10, depth=20.0, dx=2000.0, dt=30.0, bathy=None, params=None, nu=1e-6, **kw):
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    H = np.full((1, nx), depth) if bathy is None else bathy
    if params is None:
        params = dict(dz_top=np.full(nz, np.nanmax(H) / nz))
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                           nu_const=nu, theta=0.6, d_min=0.05, vertical_coordinate="ale", regrid=rule,
                           regrid_params=params, **kw)
```

Add to the `test_ale_configuration_errors` parametrize list:

```python
    (dict(momentum="weno"), "momentum"),
    (dict(ke_scheme="weno"), "ke_scheme"),
```

and append:

```python
# ---------------------------------------------------------------- vector-invariant momentum
def test_momentum_defaults_vector_invariant_in_ale_and_flux_in_sigma():
    assert _ale_box("zstar", nx=8).momentum == "vector_invariant"
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=10.0, n_layers=4)
    assert m.momentum == "flux"
    with pytest.raises(ValueError, match="vector_invariant"):
        SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=10.0, n_layers=4, momentum="vector_invariant")


def test_ale_first_step_with_initial_velocity_uses_the_fallback_transports():
    # review focus 4: no transports yet (st.Qx is None) but u ≠ 0 — the PV flux and KE use L u h̄
    m = _ale_box("zstar", nx=16)
    st = m.initial_state(np.zeros((1, 16)), np.full((10, 1, 16), 7.0))
    st.u[:, 0, 1:-1] = 0.05
    gu, gv = m._tendencies(st)
    assert np.isfinite(gu).all() and np.isfinite(gv).all()
    st2 = m.step(st)
    assert st2.Qx is not None and np.isfinite(st2.u).all()


def test_ale_truncates_a_runaway_velocity_and_stops_beyond_max_truncations():
    m = _ale_box("zstar", nx=16, dt=30.0)
    st = m.initial_state(np.zeros((1, 16)), np.full((10, 1, 16), 7.0))
    st.u[5, 0, 8] = 0.95 * 2000.0 / 30.0                                 # Courant 0.95 in a 2 m layer (K symmetric
    st2 = m.step(st)                                                      # around this face: it survives the step)
    assert np.abs(st2.u).max() <= 0.5 * 2000.0 / 30.0 + 1e-12
    assert st2.n_trunc >= 1
    m0 = _ale_box("zstar", nx=16, dt=30.0, max_truncations=0)
    st = m0.initial_state(np.zeros((1, 16)), np.full((10, 1, 16), 7.0))
    st.u[5, 0, 8] = 0.95 * 2000.0 / 30.0
    with pytest.raises(RuntimeError, match="truncation"):
        m0.step(st)


def test_vector_invariant_coriolis_in_a_rotating_ale_basin_keeps_rest_and_stays_finite():
    # f-plane 2-D z* basin with a coast (review focus 1): rest stays rest; a sloshing start stays finite
    nx, ny, nz = 10, 8, 4
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=2000.0, dy=2000.0, f0=1e-4)
    H = np.full((ny, nx), 20.0)
    H[3, 4] = np.nan                                                    # an island
    m = SigmaBaroclinic(hg, H, dt=30.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 5.0)))
    st = m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 7.0))
    for _ in range(20):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-12 and np.abs(st.v).max() < 1e-12
    x = (np.arange(nx) + 0.5) * 2000.0
    st = m.initial_state(0.2 * np.cos(np.pi * x / (nx * 2000.0))[None, :].repeat(ny, 0), np.full((nz, ny, nx), 7.0))
    for _ in range(100):
        st = m.step(st)
    assert np.isfinite(st.u).all() and np.isfinite(st.v).all() and np.abs(st.v).max() > 0
```

- [ ] **Step 2: Run them to see them fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k "momentum or fallback or truncat or rotating_ale or configuration" -q`
Expected: FAIL — `TypeError ... unexpected keyword argument 'momentum'` / `'max_truncations'`, `AttributeError: ... 'momentum'`, `n_trunc`.

- [ ] **Step 3: Implement**

In `core/dynamics/sigma_baroclinic.py`:

1. Import: `from core.dynamics import vector_invariant as vi` next to the other `core` imports.
2. `SigmaState`: add after `time: float = 0.0`:
   `n_trunc: int = 0                                         # CFL velocity truncations so far (ALE, counted faces)`
3. Constructor: add `momentum: str | None = None, ke_scheme: str = "arakawa", max_truncations: int = 1000` after `init_iterations: int = 10` in the signature; document them in the class docstring:
   ```
       momentum : 'vector_invariant' (default in ALE: PV flux + KE gradient, MOM6) or 'flux' (default in
             σ: flux-form WENO5 + energy-conserving Coriolis); 'vector_invariant' needs vertical_coordinate='ale'
       ke_scheme : 'arakawa' (default) or 'gudonov' (upwind) kinetic energy of the vector-invariant form
       max_truncations : ALE: stop after this many CFL velocity truncations at faces thicker than h_thin_bed
   ```
   and pass them to `_setup_ale(..., act, momentum, ke_scheme, max_truncations)`.
4. `_setup_ale`: add the three parameters (defaults `None, "arakawa", 1000`) and, right after the `vertical_coordinate` check and `self.ale = ...` (before `if not self.ale: return`):
   ```python
        momentum = momentum or ("vector_invariant" if self.ale else "flux")
        if momentum not in ("flux", "vector_invariant"):
            raise ValueError(f"unknown momentum {momentum!r}; use 'vector_invariant' or 'flux'")
        if momentum == "vector_invariant" and not self.ale:
            raise ValueError("momentum='vector_invariant' needs vertical_coordinate='ale' (the σ model keeps the "
                             "flux form, which matches CROCO)")
        if ke_scheme not in ("arakawa", "gudonov"):
            raise ValueError(f"unknown ke_scheme {ke_scheme!r}; use 'arakawa' or 'gudonov'")
        self.momentum, self.ke_scheme, self.max_truncations = momentum, ke_scheme, int(max_truncations)
   ```
   and at the end of `_setup_ale` (ALE only) precompute the geometry:
   ```python
        g = {k: np.asarray(self.hg[k].values, float) for k in ("dx_u", "dy_v", "area_t", "area_u", "area_v",
                                                               "dx_t", "dy_t")}
        g["area_q"] = vi.corner_area(g["dx_u"], g["dy_v"])
        g["mask_q"] = vi.corner_mask(act)
        self._vi = g
   ```
   (`self.hg` is set before `_setup_ale` is called; check the order in `__init__` and move the call after `self.hg = h_grid` if needed.)
5. New method after `_coriolis`:
   ```python
    def _vector_invariant(self, st: SigmaState) -> tuple[np.ndarray, np.ndarray]:
        """
        Vector-invariant momentum terms of the ALE step (spec 2026-10-12 §3): Sadourny (1975) energy
        PV flux q·(vh, −uh) with BOUND_CORIOLIS, minus ∇K (core.dynamics.vector_invariant).  The
        transports are those of the last Lagrangian step (as MOM6 uses the uh of its last continuity);
        the first step uses L u h̄.
        """
        g = self._vi
        zeta = vi.vorticity_q(st.u, st.v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"])
        q = vi.pv_q(zeta, self._fq, st.h, g["area_t"], self.fs.active)
        Qx, Qy = st.Qx, st.Qy
        if Qx is None or Qy is None:
            hu, hv = self._mean_face_thickness(st.h)
            Qx, Qy = self.fs.Lu[None] * st.u * hu, self.fs.Lv[None] * st.v * hv
        cu, cv = vi.coriolis_pv(q, self._fq[None] + zeta, Qx, Qy, st.u, st.v, g["dx_u"], g["dy_v"], bound=True)
        K = vi.kinetic_energy(st.u, st.v, g["area_u"], g["area_v"], g["area_t"], self.ke_scheme)
        kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
        return cu + kx, cv + ky
   ```
6. `_tendencies`: replace
   ```python
        cu, cv = self._coriolis(st.u, st.v)
   ```
   by
   ```python
        vinv = self.ale and self.momentum == "vector_invariant"
        cu, cv = self._vector_invariant(st) if vinv else self._coriolis(st.u, st.v)
   ```
   change `if self.adv == "flux":` to `if self.adv == "flux" and not vinv:` and, inside the layer loop, `if self.adv != "flux":` to `if self.adv != "flux" and not vinv:` (the Laplacian `ah` term stays for both).
7. End of `_step_ale`, after `u_new`/`v_new` are remapped and thin-bed-zeroed (they use `hu1`, `hv1`), before building the returned state:
   ```python
        g = self._vi
        u_before, v_before = u_new, v_new
        u_new, nu_, fu = vi.truncate_cfl(u_new, dt, g["dx_t"], hu1, self.h_thin_bed)
        v_sw, nv_, fv = vi.truncate_cfl(v_new.swapaxes(1, 2), dt, g["dy_t"].T, hv1.swapaxes(1, 2), self.h_thin_bed)
        v_new = v_sw.swapaxes(1, 2)
        n_trunc = st.n_trunc + nu_ + nv_
        if n_trunc > self.max_truncations:
            if fu is not None:
                k, j, i = fu
                where = f"layer {k} at u-face (j={j}, i={i}): u = {u_before[k, j, i]:.3g} m/s, face thickness {hu1[k, j, i]:.3g} m"
            else:
                k, i, j = fv
                where = f"layer {k} at v-face (j={j}, i={i}): v = {v_before[k, j, i]:.3g} m/s, face thickness {hv1[k, j, i]:.3g} m"
            raise RuntimeError(f"{n_trunc} CFL velocity truncations exceed max_truncations={self.max_truncations}; "
                               f"first this step: {where}")
   ```
   and pass `n_trunc=n_trunc` to the returned `SigmaState(...)`.

- [ ] **Step 4: Run the ALE tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_vector_invariant.py -q`
Expected: the new tests pass. Existing ALE tests now run with the vector-invariant default; read every failure. The ALE-σ lock exchange (Benjamin ±10 %), the internal wave (within 10 % of σ), the beach (review focus 3) and the bay-mouth step must pass unchanged — a failure there is a finding to debug (superpowers:systematic-debugging), not a tolerance to loosen. The strict xfails of the hybrid lock may now XPASS (that is Task 4).

- [ ] **Step 5: Run the σ suites (bit-identical guard)**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_baroclinic.py tests/test_sigma_wd.py tests/test_restart.py -q`
Expected: all pass, unchanged counts (41 + 14 + 8).

- [ ] **Step 6: Commit (with Esteban's authorisation)**

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py
git commit -m "ALE: vector-invariant momentum by default and CFL truncation backstop"
```

---

### Task 4: Acceptance — hybrid lock exchange against MOM6

**Files:**
- Modify: `tests/test_sigma_ale.py` (`_hybrid_lock`, `_lock_run`, hybrid lock tests and their markers)

**Interfaces:**
- Consumes: `momentum` default from Task 3; `SigmaState.n_trunc`.
- Produces: `_hybrid_lock(...)` builds with `nu=1e-4, ah=0.01` (MOM6 KV, KH); `_lock_run` also returns `umax` (max |u| over the run) and `n_trunc`.

- [ ] **Step 1: MOM6 viscosity in the hybrid lock and new run outputs**

In `_hybrid_lock`, build the hybrid model with `_ale_box("hybrid", ..., params=params, nu=1e-4, ah=0.01)` (the σ model used for the densities stays as is). In `_lock_run`, track `umax = max(umax, np.abs(st.u).max())` every step and return `umax=umax, n_trunc=st.n_trunc` in the dict.

- [ ] **Step 2: Write the acceptance assertions**

Replace the three hybrid lock tests by (remove both strict `xfail` markers):

```python
def test_hybrid_lock_exchange_fronts_leave_the_lock_at_the_benjamin_speed():
    # spec 2026-10-12 §6: body (5 m of the 10 m current) at Benjamin ±10 % (MOM6: 0.98–1.05 U)
    r = _lock_run("hybrid")
    assert r["one_thick_layer"] and r["consistent"]
    for k in range(2):
        assert abs(r["speed"][5.0][k] - 1) < 0.10, r["speed"]
    assert abs(r["heat"]) < 1e-12


def test_hybrid_lock_exchange_mixes_much_less_than_sigma():
    r, s = _lock_run("hybrid"), _lock_run("sigma")
    assert r["rpe_gain"] < 0.2 * s["rpe_gain"], (r["rpe_gain"], s["rpe_gain"])
    assert r["one_thick_layer"] and r["consistent"]
    assert abs(r["heat"]) < 1e-12


def test_hybrid_lock_exchange_nose_moves_at_the_benjamin_speed_without_runaway_velocities():
    # spec §6: 1 m fronts within ±10 %, max|u| ≤ 1.5 × MOM6 (0.56 m/s), no truncation
    r = _lock_run("hybrid")
    for k in range(2):
        assert abs(r["speed"][1.0][k] - 1) < 0.10, r["speed"]
    assert r["umax"] <= 0.84, r["umax"]
    assert r["n_trunc"] == 0
```

- [ ] **Step 3: Run and read the numbers**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k hybrid_lock -q` and
`NUMBA_NUM_THREADS=5 python -c "import tests.test_sigma_ale as t; r = t._lock_run('hybrid'); print(r['speed'], r['rpe_gain'], r['umax'], r['n_trunc'])"`.
Expected: all three pass. Record the numbers in the ledger.

- [ ] **Step 4: Decision gate**

If any criterion fails: do not loosen it and do not start plan B. Mark the failing test `xfail(strict=True)` with the measured numbers in the reason, ledger the result, finish Task 5 (regressions and docs), and stop: report to Esteban with the numbers (plan B — `CORIOLIS_EN_DIS`, then `hvel` — needs his decision).

- [ ] **Step 5: Commit (with Esteban's authorisation)**

```bash
git add tests/test_sigma_ale.py
git commit -m "Hybrid lock exchange with vector-invariant momentum: acceptance against MOM6"
```

---

### Task 5: Regressions, validation table and docs

**Files:**
- Modify: `validation/mom6_lock_exchange/README.md` (results table, diagnosis outcome)
- Modify: `CLAUDE.md` (rows of `sigma_baroclinic` ALE mode, new `vector_invariant.py` row, total count)
- Modify: `docs/design/vision_roadmap.md` §7 (hybrid lock row)
- Modify: memory `~/.claude/projects/-Users-estebancruz-pyOCEAN/memory/project_status.md`

- [ ] **Step 1: z\* step and seamount**

Run (scratchpad scripts; do not write `validation/seamount/out/`):
- bay-mouth step: `_step_section("zstar")` 720 steps, print `np.abs(st.u).max()`. Expected ≤ 1e-10 m/s.
- z\* seamount with FV + ρ_ref and Coriolis: `setup(6000.0, 20, 600.0, "fv", "zstar")` from `validation/seamount/run_seamount.py`, `m.rho_ref = reference_profile(...)`, 10 days; print max|u| per day. Expected day 10 ≤ 0.15 cm/s.
If either fails: debug before continuing (systematic-debugging); it is a regression of the new momentum form.

- [ ] **Step 2: Cost per step**

Scratchpad script: an ALE z\* model on a 160 × 100 × 30 grid (as the Plan B timing), 20 steps after 3 warm-up steps, once with `momentum="vector_invariant"` and once with `momentum="flux"`; print s/step. Expected: reported, no threshold.

- [ ] **Step 3: Validation table**

Run: `NUMBA_NUM_THREADS=5 python validation/mom6_lock_exchange/compare.py` and replace the results table in `validation/mom6_lock_exchange/README.md` with its output (the pyOCEAN rows now use the vector-invariant default and MOM6 viscosity). Add one paragraph "Resultado (vectorial invariante)" with the before/after numbers.

- [ ] **Step 4: Full suite and docs**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider | tail -1` and note the counts. Update:
- `CLAUDE.md`: new row "Vector-invariant momentum terms (MOM6 Sadourny energy PV flux + BOUND_CORIOLIS, KE Arakawa/Godunov, CFL truncation) | `core/dynamics/vector_invariant.py` | `tests/test_vector_invariant.py` | ✅ 10/10"; the ALE-mode row (default momentum, hybrid lock numbers, xfails if any); total line.
- Roadmap §7 hybrid-lock row with the new numbers and status.
- Memory: replace the B2/MOM6 paragraph with the outcome.

- [ ] **Step 5: Commit (with Esteban's authorisation)**

```bash
git add validation/mom6_lock_exchange/README.md CLAUDE.md docs/design/vision_roadmap.md
git commit -m "Docs: vector-invariant momentum results (hybrid lock vs MOM6)"
```
