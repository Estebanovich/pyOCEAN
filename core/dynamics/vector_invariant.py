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
