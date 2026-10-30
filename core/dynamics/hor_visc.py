"""
Laplacian Smagorinsky horizontal viscosity in thickness-weighted stress-tensor form, per layer.

Spec docs/superpowers/specs/2026-10-30-smagorinsky-viscosity-design.md ("Definiciones"); reference
implementation MOM6 ``src/parameterizations/lateral/MOM_hor_visc.F90`` (local checkout
``~/pyOCEAN_refs/MOM6``; reading notes docs/design/smag_refs/mom6_hor_visc.md).  Every layer k is
treated on its own, along the layer (σ, z*, hybrid alike), as MOM6.

Strain rates (Smagorinsky 1963, 1993; Griffies & Hallberg 2000), orthogonal curvilinear C grid:

    D_T = ∂u/∂x − ∂v/∂y        (tension, at h points;   MOM_hor_visc.F90:707-715)
    D_S = ∂v/∂x + ∂u/∂y        (shearing, at q corners; MOM_hor_visc.F90:718-722, 927)

with the metric forms of MOM6, e.g. ∂u/∂x|_h = (Δy_T/Δx_T)(u_{i+½}/Δy_{u,i+½} − u_{i−½}/Δy_{u,i−½}).
Free slip: D_S = 0 at corners that are not surrounded by four wet faces (land, dry faces of
wetting–drying) and at the corners of open-boundary faces (MOM6 ``OBC_STRAIN_ZERO``).

Shear magnitude without the divergence (MOM6, MITgcm; MOM_hor_visc.F90:1018-1022, 1355-1359):

    |S|_h = √( D_T² + ¼ Σ₄ D_S² ),     |S|_q = √( D_S² + ¼ Σ₄ D_T² )

Viscosity, grid scale and background (MOM_hor_visc.F90:3382-3389, 3403-3414, 1060-1070):

    Δ² = 2 Δx² Δy² / (Δx² + Δy²),      ν = max( A_h, C Δ² |S| )

Stability bound on the final ν, at h and q separately (MOM6 ``BOUND_KH``, MOM_hor_visc.F90:3507-3529
and 1121-1133, 1500-1512), rescaled for Adams–Bashforth 3 (real-axis stability interval λΔt ≤ 6/11
instead of MOM6's forward-Euler-like 0.25 factor; spec "Definiciones"):

    ν ≤ 0.8 · (6/11) / (4 Δt (1/Δx² + 1/Δy²)) · hrat_min,
    hrat_min = min(1, h_min / (h_point + h_neglect))

where h_min is the thinnest of the four faces around the point (MOM_hor_visc.F90:1033-1037,
1368-1371).  At h points only wet faces enter h_min (MOM6 uses the land-masked half thickness there;
a land face receives no tendency, so it does not constrain the explicit step).

Thickness-weighted stresses and their divergence (MOM_hor_visc.F90:1362-1367, 1727-1737, 1757-1781):

    σ_xx = ν_h h D_T,   σ_xy = ν_q h_q D_S,
    h_q = 2 h2uq h2vq / ( h_neglect³ + (h2uq + h2vq) (h_u,s + h_u,n + h_v,w + h_v,e) ),
    h2uq = 4 h_u,s h_u,n,  h2vq = 4 h_v,w h_v,e          (0 when any of the four faces vanishes)

    F_u = [ (Δy_T,e² σ_xx,e − Δy_T,w² σ_xx,w)/Δy_u + (Δx_q,n² σ_xy,n − Δx_q,s² σ_xy,s)/Δx_u ] / (A_u (h_u + h_neglect))
    F_v = [ (Δy_q,e² σ_xy,e − Δy_q,w² σ_xy,w)/Δy_v − (Δx_T,n² σ_xx,n − Δx_T,s² σ_xx,s)/Δx_v ] / (A_v (h_v + h_neglect))

Summation by parts gives Σ h_u u F_u A_u + Σ h_v v F_v A_v = −Σ ν h D_T² A_T − Σ ν h_q D_S² A_q ≤ 0
(the operator only removes kinetic energy).  F_u = 0 on non-wet faces, on open-boundary faces
(MOM_hor_visc.F90:1761-1771, 1783-1793) and on the domain-edge faces.

Outputs at points without stress are 0: ν_h where the cell has no wet face, ν_q at corners that are
not active.  ``n_limited`` counts active h and q points where the bound strictly reduced ν.

Shapes: u (N, ny, nx+1), v (N, ny+1, nx), h_t (N, ny, nx), q corners (N, ny+1, nx+1); corner [J, I]
is shared by cells (J−1, J) × (I−1, I).  Masks are 2-D (ny, nx+1), (ny+1, nx), (ny+1, nx+1).

References: Smagorinsky, J. (1963). Mon. Wea. Rev. 91, 99–164.  Smagorinsky, J. (1993). In Large Eddy
Simulation of Complex Engineering and Geophysical Flows, CUP, 3–36.  Griffies, S. M. & Hallberg, R. W.
(2000). Mon. Wea. Rev. 128, 2935–2946.
"""

from __future__ import annotations

import numba
import numpy as np
import xarray as xr

from core.model.threads import _THREADS
from core.vertical.remap import H_NEGLECT

BOUND_COEF = 0.8          # MOM6 HORVISC_BOUND_COEF default (MOM_hor_visc.F90:3109)
AB3_STABILITY = 6.0 / 11.0  # AB3 stability interval on the negative real axis, λΔt ≤ 6/11 (Durran 2010)


def smag_geometry(hg: xr.Dataset) -> dict[str, np.ndarray]:
    """
    Static metric arrays of the Smagorinsky operator from a horizontal grid.

    Keys: dx_t, dy_t, area_t (ny, nx); dx_u, dy_u, area_u (ny, nx+1); dx_v, dy_v, area_v (ny+1, nx);
    dx_q, dy_q, area_q (ny+1, nx+1) [m, m²]; delta2_t, delta2_q = 2 Δx² Δy²/(Δx² + Δy²) [m²]
    (MOM_hor_visc.F90:3382, 3403); inv_lap_t, inv_lap_q = 1/(1/Δx² + 1/Δy²) [m²].  The grid has no corner
    spacings, so Δx_q is the mean of the two Δx_v next to the corner in i and Δy_q the mean of the two Δy_u
    next to it in j (the single neighbour at the domain edge).  Extra keys hold precomputed ratios and
    reciprocals used by the kernels (dy_dx_t, dx_dy_t, dy_dx_q, dx_dy_q, idx_u, idy_u, idx_v, idy_v,
    dx2_t, dy2_t, dx2_q, dy2_q, iarea_u, iarea_v), as MOM6 does (MOM_hor_visc.F90:3319-3341).
    """
    g = {k: np.ascontiguousarray(hg[k].values, dtype=float)
         for k in ("dx_t", "dy_t", "area_t", "dx_u", "dy_u", "area_u", "dx_v", "dy_v", "area_v")}
    dx_v, dy_u = g["dx_v"], g["dy_u"]
    dx_q = np.empty((dx_v.shape[0], dx_v.shape[1] + 1))
    dx_q[:, 1:-1] = 0.5 * (dx_v[:, :-1] + dx_v[:, 1:])
    dx_q[:, 0], dx_q[:, -1] = dx_v[:, 0], dx_v[:, -1]
    dy_q = np.empty((dy_u.shape[0] + 1, dy_u.shape[1]))
    dy_q[1:-1] = 0.5 * (dy_u[:-1] + dy_u[1:])
    dy_q[0], dy_q[-1] = dy_u[0], dy_u[-1]
    g["dx_q"], g["dy_q"], g["area_q"] = dx_q, dy_q, dx_q * dy_q
    for p in ("t", "q"):
        dx2, dy2 = g[f"dx_{p}"] ** 2, g[f"dy_{p}"] ** 2
        g[f"dx2_{p}"], g[f"dy2_{p}"] = dx2, dy2
        g[f"delta2_{p}"] = (2.0 * dx2 * dy2) / (dx2 + dy2)
        g[f"inv_lap_{p}"] = 1.0 / (1.0 / dx2 + 1.0 / dy2)
        g[f"dy_dx_{p}"] = g[f"dy_{p}"] / g[f"dx_{p}"]
        g[f"dx_dy_{p}"] = g[f"dx_{p}"] / g[f"dy_{p}"]
    for p in ("u", "v"):
        g[f"idx_{p}"] = 1.0 / g[f"dx_{p}"]
        g[f"idy_{p}"] = 1.0 / g[f"dy_{p}"]
        g[f"iarea_{p}"] = 1.0 / g[f"area_{p}"]
    return g


def _bound_factor(dt: float) -> float:
    """0.8 · (6/11) / (4 Δt) [s⁻¹]; times inv_lap and hrat_min gives the ν bound [m² s⁻¹]."""
    return (BOUND_COEF * AB3_STABILITY) / (4.0 * dt)


# ---------------------------------------------------------------- NumPy oracle
def smagorinsky_numpy(u: np.ndarray, v: np.ndarray, h_t: np.ndarray, h_u: np.ndarray, h_v: np.ndarray,
                      wet_u: np.ndarray, wet_v: np.ndarray, mask_q: np.ndarray,
                      open_u: np.ndarray, open_v: np.ndarray, geom: dict[str, np.ndarray],
                      C: float, ah: float, dt: float, h_neglect: float = H_NEGLECT,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Smagorinsky viscous tendencies (reference implementation; the module docstring has the equations).

    Parameters
    ----------
    u, v : layer velocities [m s⁻¹], (N, ny, nx+1) and (N, ny+1, nx).
    h_t, h_u, h_v : layer thickness at cells and faces [m].
    wet_u, wet_v : bool face masks (False = land or dry face: u treated as 0, no tendency).
    mask_q : bool (ny+1, nx+1), True where the corner has its four faces wet.
    open_u, open_v : bool, open-boundary faces (no tendency, D_S = 0 at their corners).
    geom : ``smag_geometry`` output.
    C : Smagorinsky coefficient (dimensionless, MOM6 SMAG_LAP_CONST; 0.15 in OM4).
    ah : background viscosity A_h [m² s⁻¹].
    dt : time step of the explicit (AB3) momentum step [s], for the stability bound.
    h_neglect : negligible thickness [m] (``core.vertical.remap.H_NEGLECT``).

    Returns
    -------
    fu, fv : tendencies [m s⁻²]; nu_t (N, ny, nx), nu_q (N, ny+1, nx+1) [m² s⁻¹]; n_limited (int).
    """
    g = geom
    n, ny, nx1 = np.shape(u)
    nx = nx1 - 1
    hn = float(h_neglect)
    hn3 = hn * hn * hn
    um = np.where(wet_u[None], u, 0.0)
    vm = np.where(wet_v[None], v, 0.0)

    # tension at h (MOM_hor_visc.F90:707-715)
    dudx = g["dy_dx_t"] * ((um[..., 1:] * g["idy_u"][:, 1:]) - (um[..., :-1] * g["idy_u"][:, :-1]))
    dvdy = g["dx_dy_t"] * ((vm[:, 1:] * g["idx_v"][1:]) - (vm[:, :-1] * g["idx_v"][:-1]))
    DT = dudx - dvdy

    # shearing strain at interior corners (MOM_hor_visc.F90:718-722), free slip (:927)
    act_q = np.zeros((ny + 1, nx + 1), bool)
    act_q[1:-1, 1:-1] = (mask_q[1:-1, 1:-1] & ~open_u[:-1, 1:-1] & ~open_u[1:, 1:-1]
                         & ~open_v[1:-1, :-1] & ~open_v[1:-1, 1:])
    DS = np.zeros((n, ny + 1, nx + 1))
    idy_v, idx_u = g["idy_v"], g["idx_u"]
    dvdx = g["dy_dx_q"][1:-1, 1:-1] * ((vm[:, 1:-1, 1:] * idy_v[1:-1, 1:]) - (vm[:, 1:-1, :-1] * idy_v[1:-1, :-1]))
    dudy = g["dx_dy_q"][1:-1, 1:-1] * ((um[:, 1:, 1:-1] * idx_u[1:, 1:-1]) - (um[:, :-1, 1:-1] * idx_u[:-1, 1:-1]))
    DS[:, 1:-1, 1:-1] = np.where(act_q[None, 1:-1, 1:-1], dvdx + dudy, 0.0)

    bf = _bound_factor(dt)
    # ---- h points
    DS2 = DS * DS
    S_t = np.sqrt((DT * DT) + 0.25 * ((DS2[:, :-1, :-1] + DS2[:, 1:, 1:]) + (DS2[:, 1:, :-1] + DS2[:, :-1, 1:])))
    act_t = wet_u[:, :-1] | wet_u[:, 1:] | wet_v[:-1] | wet_v[1:]
    big = np.inf
    hw = np.where(wet_u[None, :, :-1], h_u[..., :-1], big)
    he = np.where(wet_u[None, :, 1:], h_u[..., 1:], big)
    hs = np.where(wet_v[None, :-1], h_v[:, :-1], big)
    hnn = np.where(wet_v[None, 1:], h_v[:, 1:], big)
    hmin_t = np.minimum(np.minimum(hw, he), np.minimum(hs, hnn))
    hrat_t = np.minimum(1.0, hmin_t / (h_t + hn))
    raw_t = np.maximum(ah, (C * g["delta2_t"]) * S_t)
    bnd_t = (bf * g["inv_lap_t"]) * hrat_t
    lim_t = (raw_t > bnd_t) & act_t[None]
    nu_t = np.where(act_t[None], np.where(raw_t > bnd_t, bnd_t, raw_t), 0.0)
    sxx = (nu_t * DT) * h_t

    # ---- q points (interior)
    DT2 = DT * DT
    S_q = np.sqrt((DS2[:, 1:-1, 1:-1])
                  + 0.25 * ((DT2[:, :-1, :-1] + DT2[:, 1:, 1:]) + (DT2[:, 1:, :-1] + DT2[:, :-1, 1:])))
    hus, hun = h_u[:, :-1, 1:-1], h_u[:, 1:, 1:-1]
    hvw, hve = h_v[:, 1:-1, :-1], h_v[:, 1:-1, 1:]
    h2uq = 4.0 * (hus * hun)
    h2vq = 4.0 * (hvw * hve)
    hq = (2.0 * (h2uq * h2vq)) / (hn3 + (h2uq + h2vq) * ((hus + hun) + (hvw + hve)))
    hmin_q = np.minimum(np.minimum(hus, hun), np.minimum(hvw, hve))
    hrat_q = np.minimum(1.0, hmin_q / (hq + hn))
    raw_q = np.maximum(ah, (C * g["delta2_q"][1:-1, 1:-1]) * S_q)
    bnd_q = (bf * g["inv_lap_q"][1:-1, 1:-1]) * hrat_q
    aq = act_q[None, 1:-1, 1:-1]
    lim_q = (raw_q > bnd_q) & aq
    nu_q = np.zeros((n, ny + 1, nx + 1))
    nu_q[:, 1:-1, 1:-1] = np.where(aq, np.where(raw_q > bnd_q, bnd_q, raw_q), 0.0)
    sxy = np.zeros((n, ny + 1, nx + 1))
    sxy[:, 1:-1, 1:-1] = (nu_q[:, 1:-1, 1:-1] * DS[:, 1:-1, 1:-1]) * hq

    # ---- divergence of the stress (MOM_hor_visc.F90:1757-1760, 1779-1781)
    fu = np.zeros((n, ny, nx + 1))
    xx = (g["dy2_t"][:, 1:] * sxx[..., 1:]) - (g["dy2_t"][:, :-1] * sxx[..., :-1])
    xy = (g["dx2_q"][1:, 1:-1] * sxy[:, 1:, 1:-1]) - (g["dx2_q"][:-1, 1:-1] * sxy[:, :-1, 1:-1])
    fu[..., 1:-1] = (((idx_u[:, 1:-1] * xy) + (g["idy_u"][:, 1:-1] * xx)) * g["iarea_u"][:, 1:-1]) \
        / (h_u[..., 1:-1] + hn)
    fu = np.where((wet_u & ~open_u)[None], fu, 0.0)
    fv = np.zeros((n, ny + 1, nx))
    xy = (g["dy2_q"][1:-1, 1:] * sxy[:, 1:-1, 1:]) - (g["dy2_q"][1:-1, :-1] * sxy[:, 1:-1, :-1])
    xx = (g["dx2_t"][1:] * sxx[:, 1:]) - (g["dx2_t"][:-1] * sxx[:, :-1])
    fv[:, 1:-1] = (((idy_v[1:-1] * xy) - (g["idx_v"][1:-1] * xx)) * g["iarea_v"][1:-1]) / (h_v[:, 1:-1] + hn)
    fv = np.where((wet_v & ~open_v)[None], fv, 0.0)
    return fu, fv, nu_t, nu_q, int(lim_t.sum() + lim_q.sum())


# ---------------------------------------------------------------- Numba kernel
@numba.njit(parallel=True, cache=True)
def _smag_kernel(u, v, h_t, h_u, h_v, wet_u, wet_v, act_q, fuok, fvok,
                 dy_dx_t, dx_dy_t, dy_dx_q, dx_dy_q, idx_u, idy_u, idx_v, idy_v,
                 dx2_t, dy2_t, dx2_q, dy2_q, iarea_u, iarea_v, delta2_t, delta2_q, inv_lap_t, inv_lap_q,
                 C, ah, bf, hn, DT, DS, nu_t, nu_q, sxx, sxy, fu, fv, lim_t, lim_q):  # pragma: no cover
    """smagorinsky_numpy with the same per-element arithmetic, parallel over (layer, row).
    Pass 1: D_T and D_S; pass 2: ν and σ at h and at q (needs neighbour strains); pass 3: F_u, F_v."""
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    hn3 = hn * hn * hn
    # pass 1a: tension
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        for i in range(nx):
            ue = u[k, j, i + 1] if wet_u[j, i + 1] else 0.0
            uw = u[k, j, i] if wet_u[j, i] else 0.0
            vn = v[k, j + 1, i] if wet_v[j + 1, i] else 0.0
            vs = v[k, j, i] if wet_v[j, i] else 0.0
            dudx = dy_dx_t[j, i] * ((ue * idy_u[j, i + 1]) - (uw * idy_u[j, i]))
            dvdy = dx_dy_t[j, i] * ((vn * idx_v[j + 1, i]) - (vs * idx_v[j, i]))
            DT[k, j, i] = dudx - dvdy
    # pass 1b: shearing strain
    for p in numba.prange(n * (ny + 1)):
        k, J = p // (ny + 1), p % (ny + 1)
        for I in range(nx1):
            if act_q[J, I]:
                ve = v[k, J, I] if wet_v[J, I] else 0.0
                vw = v[k, J, I - 1] if wet_v[J, I - 1] else 0.0
                un = u[k, J, I] if wet_u[J, I] else 0.0
                us = u[k, J - 1, I] if wet_u[J - 1, I] else 0.0
                dvdx = dy_dx_q[J, I] * ((ve * idy_v[J, I]) - (vw * idy_v[J, I - 1]))
                dudy = dx_dy_q[J, I] * ((un * idx_u[J, I]) - (us * idx_u[J - 1, I]))
                DS[k, J, I] = dvdx + dudy
            else:
                DS[k, J, I] = 0.0
    # pass 2a: ν and σ_xx at h
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        cnt = 0
        for i in range(nx):
            a = DS[k, j, i] * DS[k, j, i]
            b = DS[k, j + 1, i + 1] * DS[k, j + 1, i + 1]
            c = DS[k, j + 1, i] * DS[k, j + 1, i]
            d = DS[k, j, i + 1] * DS[k, j, i + 1]
            dt_ = DT[k, j, i]
            S = np.sqrt((dt_ * dt_) + 0.25 * ((a + b) + (c + d)))
            if wet_u[j, i] or wet_u[j, i + 1] or wet_v[j, i] or wet_v[j + 1, i]:
                hmin = np.inf
                if wet_u[j, i]:
                    hmin = min(hmin, h_u[k, j, i])
                if wet_u[j, i + 1]:
                    hmin = min(hmin, h_u[k, j, i + 1])
                if wet_v[j, i]:
                    hmin = min(hmin, h_v[k, j, i])
                if wet_v[j + 1, i]:
                    hmin = min(hmin, h_v[k, j + 1, i])
                hrat = min(1.0, hmin / (h_t[k, j, i] + hn))
                raw = max(ah, (C * delta2_t[j, i]) * S)
                bnd = (bf * inv_lap_t[j, i]) * hrat
                if raw > bnd:
                    nu = bnd
                    cnt += 1
                else:
                    nu = raw
            else:
                nu = 0.0
            nu_t[k, j, i] = nu
            sxx[k, j, i] = (nu * dt_) * h_t[k, j, i]
        lim_t[p] = cnt
    # pass 2b: ν and σ_xy at q
    for p in numba.prange(n * (ny + 1)):
        k, J = p // (ny + 1), p % (ny + 1)
        cnt = 0
        for I in range(nx1):
            if act_q[J, I]:
                a = DT[k, J - 1, I - 1] * DT[k, J - 1, I - 1]
                b = DT[k, J, I] * DT[k, J, I]
                c = DT[k, J, I - 1] * DT[k, J, I - 1]
                d = DT[k, J - 1, I] * DT[k, J - 1, I]
                ds = DS[k, J, I]
                S = np.sqrt((ds * ds) + 0.25 * ((a + b) + (c + d)))
                hus, hun = h_u[k, J - 1, I], h_u[k, J, I]
                hvw, hve = h_v[k, J, I - 1], h_v[k, J, I]
                h2uq = 4.0 * (hus * hun)
                h2vq = 4.0 * (hvw * hve)
                hq = (2.0 * (h2uq * h2vq)) / (hn3 + (h2uq + h2vq) * ((hus + hun) + (hvw + hve)))
                hmin = min(min(hus, hun), min(hvw, hve))
                hrat = min(1.0, hmin / (hq + hn))
                raw = max(ah, (C * delta2_q[J, I]) * S)
                bnd = (bf * inv_lap_q[J, I]) * hrat
                if raw > bnd:
                    nu = bnd
                    cnt += 1
                else:
                    nu = raw
                nu_q[k, J, I] = nu
                sxy[k, J, I] = (nu * ds) * hq
            else:
                nu_q[k, J, I] = 0.0
                sxy[k, J, I] = 0.0
        lim_q[p] = cnt
    # pass 3a: F_u
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        for I in range(nx1):
            if 0 < I < nx and fuok[j, I]:
                xx = (dy2_t[j, I] * sxx[k, j, I]) - (dy2_t[j, I - 1] * sxx[k, j, I - 1])
                xy = (dx2_q[j + 1, I] * sxy[k, j + 1, I]) - (dx2_q[j, I] * sxy[k, j, I])
                fu[k, j, I] = (((idx_u[j, I] * xy) + (idy_u[j, I] * xx)) * iarea_u[j, I]) / (h_u[k, j, I] + hn)
            else:
                fu[k, j, I] = 0.0
    # pass 3b: F_v
    for p in numba.prange(n * (ny + 1)):
        k, J = p // (ny + 1), p % (ny + 1)
        for i in range(nx):
            if 0 < J < ny and fvok[J, i]:
                xy = (dy2_q[J, i + 1] * sxy[k, J, i + 1]) - (dy2_q[J, i] * sxy[k, J, i])
                xx = (dx2_t[J, i] * sxx[k, J, i]) - (dx2_t[J - 1, i] * sxx[k, J - 1, i])
                fv[k, J, i] = (((idy_v[J, i] * xy) - (idx_v[J, i] * xx)) * iarea_v[J, i]) / (h_v[k, J, i] + hn)
            else:
                fv[k, J, i] = 0.0


def smagorinsky(u: np.ndarray, v: np.ndarray, h_t: np.ndarray, h_u: np.ndarray, h_v: np.ndarray,
                wet_u: np.ndarray, wet_v: np.ndarray, mask_q: np.ndarray,
                open_u: np.ndarray, open_v: np.ndarray, geom: dict[str, np.ndarray],
                C: float, ah: float, dt: float, h_neglect: float = H_NEGLECT,
                ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Smagorinsky viscous tendencies, Numba kernel (same signature, outputs and bits as
    ``smagorinsky_numpy``, the reference; any thread count gives the same result).  Threads from
    ``core.model.threads`` (``set_threads``).
    """
    numba.set_num_threads(_THREADS[0])
    g = geom
    u = np.ascontiguousarray(u, dtype=float)
    v = np.ascontiguousarray(v, dtype=float)
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    wet_u = np.ascontiguousarray(wet_u, dtype=np.bool_)
    wet_v = np.ascontiguousarray(wet_v, dtype=np.bool_)
    open_u = np.asarray(open_u, dtype=np.bool_)
    open_v = np.asarray(open_v, dtype=np.bool_)
    act_q = np.zeros((ny + 1, nx + 1), np.bool_)
    act_q[1:-1, 1:-1] = (np.asarray(mask_q, bool)[1:-1, 1:-1] & ~open_u[:-1, 1:-1] & ~open_u[1:, 1:-1]
                         & ~open_v[1:-1, :-1] & ~open_v[1:-1, 1:])
    fuok = np.ascontiguousarray(wet_u & ~open_u)
    fvok = np.ascontiguousarray(wet_v & ~open_v)
    DT, sxx, nu_t = (np.empty((n, ny, nx)) for _ in range(3))
    DS, sxy, nu_q = (np.empty((n, ny + 1, nx + 1)) for _ in range(3))
    fu, fv = np.empty((n, ny, nx + 1)), np.empty((n, ny + 1, nx))
    lim_t = np.zeros(n * ny, np.int64)
    lim_q = np.zeros(n * (ny + 1), np.int64)
    _smag_kernel(u, v, np.ascontiguousarray(h_t, dtype=float), np.ascontiguousarray(h_u, dtype=float),
                 np.ascontiguousarray(h_v, dtype=float), wet_u, wet_v, act_q, fuok, fvok,
                 g["dy_dx_t"], g["dx_dy_t"], g["dy_dx_q"], g["dx_dy_q"], g["idx_u"], g["idy_u"],
                 g["idx_v"], g["idy_v"], g["dx2_t"], g["dy2_t"], g["dx2_q"], g["dy2_q"],
                 g["iarea_u"], g["iarea_v"], g["delta2_t"], g["delta2_q"], g["inv_lap_t"], g["inv_lap_q"],
                 float(C), float(ah), _bound_factor(float(dt)), float(h_neglect),
                 DT, DS, nu_t, nu_q, sxx, sxy, fu, fv, lim_t, lim_q)
    return fu, fv, nu_t, nu_q, int(lim_t.sum() + lim_q.sum())
