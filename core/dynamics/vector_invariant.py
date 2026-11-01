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

import numba
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


def grid_corner_area(hg) -> np.ndarray:
    """
    Area of the corner (q) cells [m²] used by the vorticity ζ = circulation/A_q: the grid's ``area_q`` when it has
    one (MOM6 areaBu, four supergrid sub-cells, ``core.grid.supergrid``; MOM_CoriolisAdv.F90:605-608 divides by
    G%IareaBu), else ``corner_area`` of Δx_u and Δy_v (grids of ``core.grid.horizontal``, which carry no corner
    metrics; equal to areaBu in the interior of a stretched Cartesian grid, and bit for bit as before).
    """
    if "area_q" in hg:
        return np.asarray(hg["area_q"].values, float)
    return corner_area(np.asarray(hg["dx_u"].values, float), np.asarray(hg["dy_v"].values, float))


def vorticity_q_numpy(u: np.ndarray, v: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray, area_q: np.ndarray,
                mask_q: np.ndarray) -> np.ndarray:
    """
    NumPy reference of vorticity_q (test oracle).
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


def pv_q_numpy(zeta: np.ndarray, f_q: np.ndarray, h: np.ndarray, area_t: np.ndarray, active: np.ndarray) -> np.ndarray:
    """
    NumPy reference of pv_q (test oracle).
    Potential vorticity at the corners [m⁻¹ s⁻¹], q = (f + ζ) Σ A/(Σ A h + V_neglect) over the wet
    cells around the corner (MOM_CoriolisAdv.F90:337, 364-367, 429-437, 625-629).
    """
    A = np.where(active, area_t, 0.0)
    Ah = np.where(active[None], np.nan_to_num(h), 0.0) * A[None]
    return (f_q[None] + zeta) * _corner_sum(A)[None] / (_corner_sum(Ah) + V_NEGLECT)


def coriolis_pv_numpy(q: np.ndarray, abs_vort: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, u: np.ndarray,
                v: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray, bound: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    NumPy reference of coriolis_pv (test oracle).
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


def kinetic_energy_numpy(u: np.ndarray, v: np.ndarray, area_u: np.ndarray, area_v: np.ndarray, area_t: np.ndarray,
                   scheme: str = "arakawa") -> np.ndarray:
    """
    NumPy reference of kinetic_energy (test oracle).
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


def ke_gradient_numpy(K: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """NumPy reference of ke_gradient (test oracle).  −∂K/∂x at the u-faces and −∂K/∂y at the v-faces [m s⁻²]; domain-edge faces 0."""
    n, ny, nx = K.shape
    gx = np.zeros((n, ny, nx + 1))
    gy = np.zeros((n, ny + 1, nx))
    gx[..., 1:-1] = -(K[..., 1:] - K[..., :-1]) / dx_u[None, :, 1:-1]
    gy[:, 1:-1, :] = -(K[:, 1:] - K[:, :-1]) / dy_v[None, 1:-1, :]
    return gx, gy


def truncate_cfl_numpy(u: np.ndarray, dt: float, dx_t: np.ndarray, h_face: np.ndarray, h_count: float,
                       cfl: float = 0.5) -> tuple[np.ndarray, int, tuple[int, int, int] | None]:
    """
    NumPy reference of truncate_cfl (test oracle).
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


@numba.njit(parallel=True, cache=True)
def _truncate_kernel(u, dt, dx_t, h_face, h_count, cfl, out, counted, periodic):  # pragma: no cover
    """truncate_cfl_numpy with the same arithmetic, parallel over (layer, row); ``periodic``: the edge faces take
    the donor across the wrap (cell nx−1 west of face 0, cell 0 east of face nx)."""
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        for i in range(nx1):
            x = u[k, j, i]
            if periodic:
                dxd = dx_t[j, (i - 1) % nx] if x >= 0 else dx_t[j, i % nx]
            else:
                dxd = dx_t[j, max(i - 1, 0)] if x >= 0 else dx_t[j, min(i, nx - 1)]
            over = abs(x) * dt / dxd > cfl
            out[k, j, i] = np.sign(x) * 0.9 * cfl * dxd / dt if over else x
            counted[k, j, i] = over and h_face[k, j, i] > h_count


def truncate_cfl(u: np.ndarray, dt: float, dx_t: np.ndarray, h_face: np.ndarray, h_count: float,
                 cfl: float = 0.5, periodic: bool = False) -> tuple[np.ndarray, int, tuple[int, int, int] | None]:
    """
    CFL truncation of the layer velocities (MOM6 vertvisc_limit_vel; see truncate_cfl_numpy, the reference).
    Returns (u, count of truncations at faces thicker than h_count, first counted (k, j, i) or None).
    Numba kernel; threads set by the caller (the ALE step).  For v call it on swapped axes with dy_t.T.
    ``periodic`` (N3 task 3): the axis wraps; faces 0 and nx (one wrap face) are truncated alike and counted once.
    """
    out = np.empty(np.shape(u))
    counted = np.empty(np.shape(u), np.bool_)
    _truncate_kernel(np.asarray(u, float), float(dt), np.asarray(dx_t, float), np.asarray(h_face, float),
                     float(h_count), float(cfl), out, counted, bool(periodic))
    if periodic:
        counted[..., -1] = False                                       # face nx is face 0
    c = int(counted.sum())
    first = tuple(int(a[0]) for a in np.nonzero(counted)) if c else None
    return out, c, first


@numba.njit(parallel=True, cache=True)
def _ke_grad_kernel(K, dx_u, dy_v, gx, gy):  # pragma: no cover
    """ke_gradient_numpy with the same arithmetic."""
    n, ny, nx = K.shape
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        gx[k, j, 0] = 0.0
        gx[k, j, nx] = 0.0
        for i in range(1, nx):
            gx[k, j, i] = -(K[k, j, i] - K[k, j, i - 1]) / dx_u[j, i]
    for p in numba.prange(n * (ny + 1)):
        k, J = p // (ny + 1), p % (ny + 1)
        for i in range(nx):
            gy[k, J, i] = 0.0 if (J == 0 or J == ny) else -(K[k, J, i] - K[k, J - 1, i]) / dy_v[J, i]


def ke_gradient(K: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """−∂K/∂x at the u-faces and −∂K/∂y at the v-faces [m s⁻²]; domain-edge faces 0.  Numba kernel;
    ke_gradient_numpy is the reference."""
    n, ny, nx = np.shape(K)
    gx, gy = np.empty((n, ny, nx + 1)), np.empty((n, ny + 1, nx))
    _ke_grad_kernel(np.asarray(K, float), np.asarray(dx_u, float), np.asarray(dy_v, float), gx, gy)
    return gx, gy


# ---------------------------------------------------------------- Numba kernels (ALE performance task 5.3)
@numba.njit(parallel=True, cache=True)
def _vort_pv_kernel(u, v, dx_u, dy_v, area_q, mask_q, f_q, h, area_t, active, zeta, q):  # pragma: no cover
    """ζ (vorticity_q_numpy) and q (pv_q_numpy) at the corners with the same per-element arithmetic."""
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    for p in numba.prange(n * (ny + 1)):
        k, J = p // (ny + 1), p % (ny + 1)
        for I in range(nx1):
            dvdx = 0.0
            dudy = 0.0
            if 0 < I < nx:
                dvdx = v[k, J, I] * dy_v[J, I] - v[k, J, I - 1] * dy_v[J, I - 1]
            if 0 < J < ny:
                dudy = u[k, J, I] * dx_u[J, I] - u[k, J - 1, I] * dx_u[J - 1, I]
            z = (dvdx - dudy) / area_q[J, I] if mask_q[J, I] else 0.0
            zeta[k, J, I] = z
            sa = 0.0
            sah = 0.0
            for jj, ii in ((J - 1, I - 1), (J - 1, I), (J, I - 1), (J, I)):
                a = 0.0
                ah = 0.0
                if 0 <= jj < ny and 0 <= ii < nx and active[jj, ii]:
                    a = area_t[jj, ii]
                    hv = h[k, jj, ii]
                    ah = (hv if np.isfinite(hv) else 0.0) * a
                sa += a
                sah += ah
            q[k, J, I] = (f_q[J, I] + z) * sa / (sah + V_NEGLECT)


def vorticity_q(u: np.ndarray, v: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray, area_q: np.ndarray,
                mask_q: np.ndarray) -> np.ndarray:
    """
    Relative vorticity at the corners [s⁻¹] in circulation form, ζ = (δx(v Δy) − δy(u Δx))/A_q,
    free slip: 0 at corners touching land or the domain edge (MOM_CoriolisAdv.F90:595-610).
    Numba kernel; vorticity_q_numpy is the reference.
    """
    ny, nx1 = u.shape[1], u.shape[2]
    return vorticity_and_pv(u, v, dx_u, dy_v, area_q, mask_q, np.zeros((ny + 1, nx1)),
                            np.zeros((u.shape[0], ny, nx1 - 1)), np.ones((ny, nx1 - 1)),
                            np.zeros((ny, nx1 - 1), bool))[0]


def vorticity_and_pv(u: np.ndarray, v: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray, area_q: np.ndarray,
                     mask_q: np.ndarray, f_q: np.ndarray, h: np.ndarray, area_t: np.ndarray,
                     active: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """(ζ, q) at the corners in one pass: vorticity_q and pv_q fused (Numba kernel, same arithmetic)."""
    n, ny, nx1 = u.shape
    zeta, q = np.empty((n, ny + 1, nx1)), np.empty((n, ny + 1, nx1))
    c = np.asarray                                                     # views pass as they are (perf part 2)
    _vort_pv_kernel(c(u, dtype=float), c(v, dtype=float), c(dx_u, dtype=float), c(dy_v, dtype=float),
                    c(area_q, dtype=float), c(mask_q, dtype=np.bool_), c(f_q, dtype=float), c(h, dtype=float),
                    c(area_t, dtype=float), c(active, dtype=np.bool_), zeta, q)
    return zeta, q


@numba.njit(parallel=True, cache=True)
def _pv_kernel(zeta, f_q, h, area_t, active, q):  # pragma: no cover
    """q of pv_q_numpy with the same per-element arithmetic."""
    n, ny1, nx1 = zeta.shape
    ny, nx = ny1 - 1, nx1 - 1
    for p in numba.prange(n * ny1):
        k, J = p // ny1, p % ny1
        for I in range(nx1):
            sa = 0.0
            sah = 0.0
            for jj, ii in ((J - 1, I - 1), (J - 1, I), (J, I - 1), (J, I)):
                a = 0.0
                ah = 0.0
                if 0 <= jj < ny and 0 <= ii < nx and active[jj, ii]:
                    a = area_t[jj, ii]
                    hv = h[k, jj, ii]
                    ah = (hv if np.isfinite(hv) else 0.0) * a
                sa += a
                sah += ah
            q[k, J, I] = (f_q[J, I] + zeta[k, J, I]) * sa / (sah + V_NEGLECT)


def pv_q(zeta: np.ndarray, f_q: np.ndarray, h: np.ndarray, area_t: np.ndarray, active: np.ndarray) -> np.ndarray:
    """
    Potential vorticity at the corners [m⁻¹ s⁻¹], q = (f + ζ) Σ A/(Σ A h + V_neglect) over the wet
    cells around the corner (MOM_CoriolisAdv.F90:337, 364-367, 429-437, 625-629).  Numba kernel;
    pv_q_numpy is the reference.
    """
    q = np.empty(np.shape(zeta))
    c = np.asarray                                                     # views pass as they are (perf part 2)
    _pv_kernel(c(zeta, dtype=float), c(f_q, dtype=float), c(h, dtype=float), c(area_t, dtype=float),
               c(active, dtype=np.bool_), q)
    return q


@numba.njit(parallel=True, cache=True)
def _coriolis_kernel(q, av, Qx, Qy, u, v, dx_u, dy_v, bound, cu, cv):  # pragma: no cover
    """coriolis_pv_numpy with the same per-element arithmetic (Sadourny 1975; BOUND_CORIOLIS)."""
    n, ny, nx1 = Qx.shape
    nx = nx1 - 1
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        cu[k, j, 0] = 0.0
        cu[k, j, nx] = 0.0
        for I in range(1, nx):
            x = 0.25 / dx_u[j, I] * (q[k, j + 1, I] * (Qy[k, j + 1, I - 1] + Qy[k, j + 1, I])
                                     + q[k, j, I] * (Qy[k, j, I - 1] + Qy[k, j, I]))
            if bound:
                aN, aS = av[k, j + 1, I], av[k, j, I]
                c1, c2, c3, c4 = aN * v[k, j + 1, I], aN * v[k, j + 1, I - 1], aS * v[k, j, I], aS * v[k, j, I - 1]
                x = min(max(x, min(min(c1, c2), min(c3, c4))), max(max(c1, c2), max(c3, c4)))
            cu[k, j, I] = x
    for p in numba.prange(n * (ny + 1)):
        k, J = p // (ny + 1), p % (ny + 1)
        for i in range(nx):
            if J == 0 or J == ny:
                cv[k, J, i] = 0.0
                continue
            x = -0.25 / dy_v[J, i] * (q[k, J, i] * (Qx[k, J - 1, i] + Qx[k, J, i])
                                      + q[k, J, i + 1] * (Qx[k, J - 1, i + 1] + Qx[k, J, i + 1]))
            if bound:
                aW, aE = av[k, J, i], av[k, J, i + 1]
                c1, c2 = -aW * u[k, J - 1, i], -aW * u[k, J, i]
                c3, c4 = -aE * u[k, J - 1, i + 1], -aE * u[k, J, i + 1]
                x = min(max(x, min(min(c1, c2), min(c3, c4))), max(max(c1, c2), max(c3, c4)))
            cv[k, J, i] = x


def coriolis_pv(q: np.ndarray, abs_vort: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, u: np.ndarray,
                v: np.ndarray, dx_u: np.ndarray, dy_v: np.ndarray, bound: bool = True) -> tuple[np.ndarray, np.ndarray]:
    """
    Sadourny (1975) energy-conserving PV flux [m s⁻²] (MOM_CoriolisAdv.F90:818-823, 1071-1076) with
    BOUND_CORIOLIS (:1008-1023, 1260-1275); see coriolis_pv_numpy for the formulae.  Domain-edge faces 0.
    Numba kernel; coriolis_pv_numpy is the reference.
    """
    cu, cv = np.empty(np.shape(Qx)), np.empty(np.shape(Qy))
    c = np.asarray                                                     # views pass as they are (perf part 2)
    _coriolis_kernel(c(q, dtype=float), c(abs_vort, dtype=float), c(Qx, dtype=float), c(Qy, dtype=float),
                     c(u, dtype=float), c(v, dtype=float), c(dx_u, dtype=float), c(dy_v, dtype=float), bool(bound),
                     cu, cv)
    return cu, cv


@numba.njit(parallel=True, cache=True)
def _ke_kernel(u, v, area_u, area_v, area_t, godunov, K):  # pragma: no cover
    """kinetic_energy_numpy with the same per-element arithmetic."""
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        for i in range(nx):
            aE, aW, aN, aS = area_u[j, i + 1], area_u[j, i], area_v[j + 1, i], area_v[j, i]
            if godunov:
                up, um = max(u[k, j, i], 0.0), min(u[k, j, i + 1], 0.0)
                vp, vm = max(v[k, j, i], 0.0), min(v[k, j + 1, i], 0.0)
                K[k, j, i] = 0.5 * (max(um ** 2 * aE, up ** 2 * aW) + max(vm ** 2 * aN, vp ** 2 * aS)) / area_t[j, i]
            else:
                K[k, j, i] = 0.25 * (aE * u[k, j, i + 1] ** 2 + aW * u[k, j, i] ** 2 + aN * v[k, j + 1, i] ** 2
                                     + aS * v[k, j, i] ** 2) / area_t[j, i]


def kinetic_energy(u: np.ndarray, v: np.ndarray, area_u: np.ndarray, area_v: np.ndarray, area_t: np.ndarray,
                   scheme: str = "arakawa") -> np.ndarray:
    """
    Kinetic energy at cell centres [m² s⁻²] (MOM6 gradKE, MOM_CoriolisAdv.F90:1377-1535), 'arakawa' or
    'gudonov' (see kinetic_energy_numpy).  Numba kernel; kinetic_energy_numpy is the reference.
    """
    if scheme not in ("arakawa", "gudonov"):
        raise ValueError(f"unknown ke_scheme {scheme!r}; use 'arakawa' or 'gudonov'")
    n, ny, nx1 = u.shape
    K = np.empty((n, ny, nx1 - 1))
    c = np.asarray                                                     # views pass as they are (perf part 2)
    _ke_kernel(c(u, dtype=float), c(v, dtype=float), c(area_u, dtype=float), c(area_v, dtype=float),
               c(area_t, dtype=float), scheme == "gudonov", K)
    return K
