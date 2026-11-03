"""
Laplacian Smagorinsky horizontal viscosity in thickness-weighted stress-tensor form, per layer, and (second half of
the module, ``biharmonic``) MOM6's biharmonic Smagorinsky viscosity in the same form (regional chain, Gulf Stream fix
step 4; reading notes docs/design/regional_refs/mom6_biharmonic.md).

Spec docs/superpowers/specs/2026-10-30-smagorinsky-viscosity-design.md ("Definiciones"); reference
implementation MOM6 ``src/parameterizations/lateral/MOM_hor_visc.F90`` (local checkout
``~/pyOCEAN_refs/MOM6``; reading notes docs/design/smag_refs/mom6_hor_visc.md).  Every layer k is
treated on its own, along the layer (σ, z*, hybrid alike), as MOM6.

Strain rates (Smagorinsky 1963, 1993; Griffies & Hallberg 2000), orthogonal curvilinear C grid:

    D_T = ∂u/∂x − ∂v/∂y        (tension, at h points;   MOM_hor_visc.F90:707-715)
    D_S = ∂v/∂x + ∂u/∂y        (shearing, at q corners; MOM_hor_visc.F90:718-722, 927)

with the metric forms of MOM6, e.g. ∂u/∂x|_h = (Δy_T/Δx_T)(u_{i+½}/Δy_{u,i+½} − u_{i−½}/Δy_{u,i−½}).
Free slip: D_S = 0 at corners that are not surrounded by four wet faces (land, dry faces of
wetting–drying) and at the corners of open-boundary faces (MOM6 ``OBC_STRAIN_ZERO``).  The normal velocity of an
open-boundary face enters D_T like any wet face (the caller marks open faces wet; MOM_hor_visc.F90:708-715).

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
    (MOM_hor_visc.F90:3382, 3403); inv_lap_t, inv_lap_q = 1/(1/Δx² + 1/Δy²) [m²].  The corner spacings are the
    grid's ``dx_q``, ``dy_q``, ``area_q`` when it has them (MOM6 dxBu, dyBu, areaBu from a supergrid,
    MOM_hor_visc.F90:3319, 3327; ``core.grid.supergrid``); a grid without them (``core.grid.horizontal``) gets
    Δx_q as the mean of the two Δx_v next to the corner in i and Δy_q the mean of the two Δy_u next to it in j (the
    single neighbour at the domain edge), equal to dxBu, dyBu in the interior of a stretched Cartesian grid.
    Extra keys hold precomputed ratios and reciprocals used by the kernels (dy_dx_t, dx_dy_t, dy_dx_q, dx_dy_q,
    idx_u, idy_u, idx_v, idy_v, dx2_t, dy2_t, dx2_q, dy2_q, iarea_u, iarea_v), as MOM6 does
    (MOM_hor_visc.F90:3319-3341).
    """
    g = {k: np.ascontiguousarray(hg[k].values, dtype=float)
         for k in ("dx_t", "dy_t", "area_t", "dx_u", "dy_u", "area_u", "dx_v", "dy_v", "area_v")}
    if all(k in hg for k in ("dx_q", "dy_q", "area_q")):              # supergrid: MOM6 dxBu, dyBu, areaBu
        for k in ("dx_q", "dy_q", "area_q"):
            g[k] = np.ascontiguousarray(hg[k].values, dtype=float)
    else:                                                              # grids without corner metrics
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


def open_corner_strain(u: np.ndarray, v: np.ndarray, vt: np.ndarray, flags: np.ndarray, h_t: np.ndarray,
                       h_u: np.ndarray, h_v: np.ndarray, geom: dict[str, np.ndarray], h_neglect: float = H_NEGLECT,
                       ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Shearing strain D_S [s⁻¹], corner thickness h_q [m] and thinnest face h_min [m] at the open-edge corners
    ``flags`` (N, ny+1, nx+1; 0 elsewhere), MOM6 ``OBC_STRAIN_COMPUTED`` (MOM_hor_visc.F90:795-807, 824-836):

        west  ∂v/∂x = 2 (Δy/Δx)_q (v[J, 0]/Δy_v − vt/Δy_v)      east  2 (Δy/Δx)_q (vt − v[J, nx−1])/Δy_v
        south ∂u/∂y = 2 (Δx/Δy)_q (u[0, I] − vt)/Δx_u            north 2 (Δx/Δy)_q (vt − u[ny−1, I])/Δx_u

    plus the ordinary along-boundary derivative of the normal velocity at the two boundary faces (their own u, v:
    MOM6 keeps mask2dCu = 1 on OBC faces).  Thicknesses projected from the interior (MOM_hor_visc.F90:857-880): the
    open boundary faces take the thickness of their boundary cell ``h_t``, the face beyond the edge mirrors the
    interior one; h_q is the harmonic form of the interior corners (:1362-1367).  ``vt``:
    ``core.dynamics.open_boundary.tangential_velocity``.
    """
    g = geom
    flags = np.asarray(flags, bool)
    N, ny, nx1 = u.shape
    nx = nx1 - 1
    hn3 = float(h_neglect) ** 3
    dvdx, dudy = np.zeros((N, ny + 1, nx + 1)), np.zeros((N, ny + 1, nx + 1))
    hus, hun, hvw, hve = (np.ones((N, ny + 1, nx + 1)) for _ in range(4))
    J = np.s_[1:-1]
    for I, col, sgn in ((0, 0, 1.0), (nx, nx - 1, -1.0)):                      # west, east
        dvdx[:, J, I] = 2.0 * g["dy_dx_q"][None, J, I] * (sgn * (v[:, J, col] - vt[:, J, I])) * g["idy_v"][None, J, col]
        dudy[:, J, I] = g["dx_dy_q"][None, J, I] * ((u[:, 1:, I] * g["idx_u"][None, 1:, I])
                                                    - (u[:, :-1, I] * g["idx_u"][None, :-1, I]))
        hus[:, J, I], hun[:, J, I] = h_t[:, :-1, col], h_t[:, 1:, col]
        hvw[:, J, I] = hve[:, J, I] = h_v[:, J, col]
    I = np.s_[1:-1]
    for Jr, row, sgn in ((0, 0, 1.0), (ny, ny - 1, -1.0)):                    # south, north
        dudy[:, Jr, I] = 2.0 * g["dx_dy_q"][None, Jr, I] * (sgn * (u[:, row, I] - vt[:, Jr, I])) * g["idx_u"][None, row, I]
        dvdx[:, Jr, I] = g["dy_dx_q"][None, Jr, I] * ((v[:, Jr, 1:] * g["idy_v"][None, Jr, 1:])
                                                      - (v[:, Jr, :-1] * g["idy_v"][None, Jr, :-1]))
        hvw[:, Jr, I], hve[:, Jr, I] = h_t[:, row, :-1], h_t[:, row, 1:]
        hus[:, Jr, I] = hun[:, Jr, I] = h_u[:, row, I]
    h2uq = 4.0 * (hus * hun)
    h2vq = 4.0 * (hvw * hve)
    hq = (2.0 * (h2uq * h2vq)) / (hn3 + (h2uq + h2vq) * ((hus + hun) + (hvw + hve)))
    hmin = np.minimum(np.minimum(hus, hun), np.minimum(hvw, hve))
    z = flags[None]
    return np.where(z, dvdx + dudy, 0.0), np.where(z, hq, 0.0), np.where(z, hmin, 0.0)


# ---------------------------------------------------------------- NumPy oracle
def smagorinsky_numpy(u: np.ndarray, v: np.ndarray, h_t: np.ndarray, h_u: np.ndarray, h_v: np.ndarray,
                      wet_u: np.ndarray, wet_v: np.ndarray, mask_q: np.ndarray,
                      open_u: np.ndarray, open_v: np.ndarray, geom: dict[str, np.ndarray],
                      C: float, ah: float, dt: float, h_neglect: float = H_NEGLECT,
                      obc_q: np.ndarray | None = None, obc_strain: tuple | None = None,
                      ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Smagorinsky viscous tendencies (reference implementation; the module docstring has the equations).

    Parameters
    ----------
    u, v : layer velocities [m s⁻¹], (N, ny, nx+1) and (N, ny+1, nx).
    h_t, h_u, h_v : layer thickness at cells and faces [m].
    wet_u, wet_v : bool face masks (False = land or dry face: u treated as 0, no tendency).  Open-boundary faces
        must be included (True): their normal velocity is a real boundary value and enters D_T and |S| of the
        boundary cells, as MOM6's mask2dCu = 1 on OBC faces (MOM_grid_initialize.F90:1270-1288; the tension loop
        MOM_hor_visc.F90:708-715 has no OBC branch); ``Momentum.smagorinsky_terms`` does so (fix of 2026-11-03).
    mask_q : bool (ny+1, nx+1), True where the corner has its four faces wet.
    open_u, open_v : bool, open-boundary faces (no tendency, D_S = 0 at their corners = MOM6 OBC_STRAIN_ZERO unless
        ``obc_q`` gives them OBC_STRAIN_COMPUTED).
    geom : ``smag_geometry`` output.
    C : Smagorinsky coefficient (dimensionless, MOM6 SMAG_LAP_CONST; 0.15 in OM4).
    ah : background viscosity A_h [m² s⁻¹].
    dt : time step of the explicit (AB3) momentum step [s], for the stability bound.
    h_neglect : negligible thickness [m] (``core.vertical.remap.H_NEGLECT``).
    obc_q, obc_strain : open-edge corners with MOM6 ``OBC_STRAIN_COMPUTED`` (``obc_tangential='mom6'``) and their
        (D_S, h_q, h_min) from ``open_corner_strain``; there ν_q and σ_xy are computed as at an interior corner, with
        the two tensions beyond the edge mirrored from the interior cells.  None: no such corners (as before).

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
    if obc_q is not None and np.any(obc_q):
        ob = np.asarray(obc_q, bool)
        DS = np.where(ob[None], obc_strain[0], DS)

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
    n_obc = 0
    if obc_q is not None and np.any(obc_q):                       # open-edge corners: tensions mirrored outside
        for J, I in zip(*np.nonzero(ob)):
            jm, jp = min(max(J - 1, 0), ny - 1), min(J, ny - 1)
            im, ip = min(max(I - 1, 0), nx - 1), min(I, nx - 1)
            a, b, c, d = DT2[:, jm, im], DT2[:, jp, ip], DT2[:, jp, im], DT2[:, jm, ip]
            ds = DS[:, J, I]
            Sq = np.sqrt((ds * ds) + 0.25 * ((a + b) + (c + d)))
            hqo, hmo = obc_strain[1][:, J, I], obc_strain[2][:, J, I]
            hrat = np.minimum(1.0, hmo / (hqo + hn))
            raw = np.maximum(ah, (C * g["delta2_q"][J, I]) * Sq)
            bnd = (bf * g["inv_lap_q"][J, I]) * hrat
            n_obc += int((raw > bnd).sum())
            nu_q[:, J, I] = np.where(raw > bnd, bnd, raw)
            sxy[:, J, I] = (nu_q[:, J, I] * ds) * hqo

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
    return fu, fv, nu_t, nu_q, int(lim_t.sum() + lim_q.sum()) + n_obc


# ---------------------------------------------------------------- Numba kernel
@numba.njit(parallel=True, cache=True)
def _smag_kernel(u, v, h_t, h_u, h_v, wet_u, wet_v, act_q, obc_q, ds_obc, hq_obc, hmin_obc, fuok, fvok,
                 dy_dx_t, dx_dy_t, dy_dx_q, dx_dy_q, idx_u, idy_u, idx_v, idy_v,
                 dx2_t, dy2_t, dx2_q, dy2_q, iarea_u, iarea_v, delta2_t, delta2_q, inv_lap_t, inv_lap_q,
                 C, ah, bf, hn, DT, DS, nu_t, nu_q, sxx, sxy, fu, fv, lim_t, lim_q):  # pragma: no cover
    """smagorinsky_numpy with the same per-element arithmetic, parallel over (layer, row).
    Pass 1: D_T and D_S; pass 2: ν and σ at h and at q (needs neighbour strains); pass 3: F_u, F_v.
    ``obc_q`` corners (open-edge, MOM6 OBC_STRAIN_COMPUTED) take D_S, h_q, h_min from the precomputed arrays and
    their tensions at clamped (mirrored) cell indices."""
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
            if obc_q[J, I]:
                DS[k, J, I] = ds_obc[k, J, I]
            elif act_q[J, I]:
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
            elif obc_q[J, I]:
                jm, jp = min(max(J - 1, 0), ny - 1), min(J, ny - 1)
                im, ip = min(max(I - 1, 0), nx - 1), min(I, nx - 1)
                a = DT[k, jm, im] * DT[k, jm, im]
                b = DT[k, jp, ip] * DT[k, jp, ip]
                c = DT[k, jp, im] * DT[k, jp, im]
                d = DT[k, jm, ip] * DT[k, jm, ip]
                ds = DS[k, J, I]
                S = np.sqrt((ds * ds) + 0.25 * ((a + b) + (c + d)))
                hq = hq_obc[k, J, I]
                hrat = min(1.0, hmin_obc[k, J, I] / (hq + hn))
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
                obc_q: np.ndarray | None = None, obc_strain: tuple | None = None,
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
    if obc_q is None:
        obc_q = np.zeros((ny + 1, nx + 1), np.bool_)
        ds_obc = hq_obc = hmin_obc = np.zeros((1, 1, 1))
    else:
        obc_q = np.ascontiguousarray(obc_q, dtype=np.bool_)
        ds_obc, hq_obc, hmin_obc = (np.ascontiguousarray(a, dtype=float) for a in obc_strain)
    fuok = np.ascontiguousarray(wet_u & ~open_u)
    fvok = np.ascontiguousarray(wet_v & ~open_v)
    DT, sxx, nu_t = (np.empty((n, ny, nx)) for _ in range(3))
    DS, sxy, nu_q = (np.empty((n, ny + 1, nx + 1)) for _ in range(3))
    fu, fv = np.empty((n, ny, nx + 1)), np.empty((n, ny + 1, nx))
    lim_t = np.zeros(n * ny, np.int64)
    lim_q = np.zeros(n * (ny + 1), np.int64)
    _smag_kernel(u, v, np.ascontiguousarray(h_t, dtype=float), np.ascontiguousarray(h_u, dtype=float),
                 np.ascontiguousarray(h_v, dtype=float), wet_u, wet_v, act_q, obc_q, ds_obc, hq_obc, hmin_obc,
                 fuok, fvok,
                 g["dy_dx_t"], g["dx_dy_t"], g["dy_dx_q"], g["dx_dy_q"], g["idx_u"], g["idy_u"],
                 g["idx_v"], g["idy_v"], g["dx2_t"], g["dy2_t"], g["dx2_q"], g["dy2_q"],
                 g["iarea_u"], g["iarea_v"], g["delta2_t"], g["delta2_q"], g["inv_lap_t"], g["inv_lap_q"],
                 float(C), float(ah), _bound_factor(float(dt)), float(h_neglect),
                 DT, DS, nu_t, nu_q, sxx, sxy, fu, fv, lim_t, lim_q)
    return fu, fv, nu_t, nu_q, int(lim_t.sum() + lim_q.sum())


# ================================================================ biharmonic Smagorinsky
# MOM6 BIHARMONIC + SMAGORINSKY_AH + BOUND_AH (src/parameterizations/lateral/MOM_hor_visc.F90, git 0e9a561; reading
# notes docs/design/regional_refs/mom6_biharmonic.md).  Per layer, along the layer:
#
#   Del2u = Idx2dyCu (dx2q_n D_S,n − dx2q_s D_S,s) + Idxdy2u (dy2h_e D_T,e − dy2h_w D_T,w)        (:948-950)
#   Del2v = Idxdy2v (dy2q_e D_S,e − dy2q_w D_S,w) − Idx2dyCv (dx2h_n D_T,n − dx2h_s D_T,s)        (:952-954)
#   str_xx = A_h,h h [ (Δy/Δx)_T δ_x(Del2u/Δy_u) − (Δx/Δy)_T δ_y(Del2v/Δx_v) ]                    (:1300-1306, 1745)
#   str_xy = A_h,q h_q [ (Δy/Δx)_q δ_x(Del2v/Δy_v) + (Δx/Δy)_q δ_y(Del2u/Δx_u) ]                  (:1316-1318, 1669-1672)
#   F_u = [ (Δx_q,s² str_xy,s − Δx_q,n² str_xy,n)/Δx_u + (Δy_T,w² str_xx,w − Δy_T,e² str_xx,e)/Δy_u ] / (A_u (h_u + h_neglect))
#                                                                                                    (:1755-1760, 1777-1781)
# so F = −∇·(A_h h ∇(∇²u))/h; for constant A_h h it is −A_h ∇⁴u.  Coefficient (:1192-1206, 3446-3468, 3489-3501):
#
#   A_h = max( AH, AH_VEL_SCALE Δ³, SMAG_BI_CONST Δ⁴ |S| ),   Δ² = 2Δx²Δy²/(Δx²+Δy²),  |S| as the Laplacian
#
# bounded (BOUND_AH, :1248-1256, 3536-3590) by hrat_min · Ah_Max, Ah_Max = 0.8 · ½ /(Δt denom) with MOM6's
# metric ``denom`` (u0u, u0v, v0u, v0v), rescaled for AB3 by 6/11 as the Laplacian (``_bound_factor``): on a
# uniform grid Ah_Max = 0.8 (6/11) Δ⁴/(64 Δt), i.e. λΔt ≤ 0.8 · 6/11 for the 2Δ checkerboard (λ = 64 A_h/Δ⁴).
# Del2 is not masked at land faces (MOM6): a land face takes ±Idxdy2u dy2h D_T of its wet cell (u = 0 on the land
# face, D_T = 0 in the land cell).  Free slip at land corners (D_S = 0, str_xy = 0).  Open boundaries:
# OBC_ZERO_BIHARMONIC (Del2u = 0 on open u-faces, Del2v = 0 on open v-faces, :956-970; NWA12 sets it), no tendency on
# open faces (:1762-1773, 1784-1795), and str_xy = 0 at the open-edge corners: with ``obc_tangential='none'`` that
# is MOM6's OBC_STRAIN_ZERO (dDel2vdx = dDel2udy = 0, :1328-1349); with 'mom6' (OBC_STRAIN_COMPUTED) MOM6 takes the
# cross-boundary δ(Del2) from its halo, here mirrored (zero gradient) so that term is 0 and the along-edge one is a
# difference of two open-face Del2 = 0, i.e. MOM6's OBC_STRAIN_FREESLIP for the biharmonic part (class c); the
# computed D_S at those corners still enters |S| and the Del2v (Del2u) of the column (row) next to the edge.


def biharmonic_geometry(geom: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """
    Static metric arrays of the biharmonic operator from ``smag_geometry`` output (MOM_hor_visc.F90:3430-3435,
    3446-3448, 3489-3490, 3536-3590), cached in ``geom`` under ``"bih"``.

    Keys: idx2dy_u = 1/(Δx_u² Δy_u), idxdy2_u = 1/(Δx_u Δy_u²) (ny, nx+1) and the v analogues [m⁻³];
    delta3_t, delta4_t, delta3_q, delta4_q = Δ³, Δ⁴ from Δ² = 2Δx²Δy²/(Δx²+Δy²) [m³, m⁴]; inv_denom_t, inv_denom_q
    = 1/denom of BOUND_AH [m⁴] (MOM6 Ah_Max = coef ½/(Δt denom)).  ``denom`` reaches one point beyond the
    domain edge (MOM6 halo metrics); here the edge metrics are repeated outward (affects the bound of the edge
    cells and corners only, class c).
    """
    if "bih" in geom:
        return geom["bih"]
    g = geom
    b: dict[str, np.ndarray] = {}
    b["idx2dy_u"] = (g["idx_u"] * g["idx_u"]) * g["idy_u"]
    b["idxdy2_u"] = g["idx_u"] * (g["idy_u"] * g["idy_u"])
    b["idx2dy_v"] = (g["idx_v"] * g["idx_v"]) * g["idy_v"]
    b["idxdy2_v"] = g["idx_v"] * (g["idy_v"] * g["idy_v"])
    for p in ("t", "q"):
        d2 = g[f"delta2_{p}"]
        b[f"delta3_{p}"] = d2 * np.sqrt(d2)
        b[f"delta4_{p}"] = d2 * d2

    P = {k: np.pad(v, 1, mode="edge") for k, v in dict(
        dy2t=g["dy2_t"], dx2t=g["dx2_t"], DYdxT=g["dy_dx_t"], DXdyT=g["dx_dy_t"], dx2q=g["dx2_q"], dy2q=g["dy2_q"],
        DYdxq=g["dy_dx_q"], DXdyq=g["dx_dy_q"], Idxu=g["idx_u"], Idyu=g["idy_u"], Idxv=g["idx_v"], Idyv=g["idy_v"],
        Iareau=g["iarea_u"], Iareav=g["iarea_v"], idx2dy_u=b["idx2dy_u"], idxdy2_u=b["idxdy2_u"],
        idx2dy_v=b["idx2dy_v"], idxdy2_v=b["idxdy2_v"]).items()}
    ny, nx = g["dx_t"].shape

    def at(name, dj, di, nr, nc):
        """Padded array ``name`` at (target index + (dj, di)) over a target grid of nr × nc."""
        return P[name][1 + dj:1 + dj + nr, 1 + di:1 + di + nc]

    # u0u, u0v at u faces and v0u, v0v at v faces (MOM_hor_visc.F90:3537-3556)
    nu_r, nu_c = ny, nx + 1

    def ufield(name, dj, di):
        return at(name, dj, di, nu_r, nu_c)

    u0u = (ufield("idxdy2_u", 0, 0) * ((ufield("dy2t", 0, 0) * ufield("DYdxT", 0, 0)
                                        * (ufield("Idyu", 0, 1) + ufield("Idyu", 0, 0)))
                                       + (ufield("dy2t", 0, -1) * ufield("DYdxT", 0, -1)
                                          * (ufield("Idyu", 0, 0) + ufield("Idyu", 0, -1))))) \
        + (ufield("idx2dy_u", 0, 0) * ((ufield("dx2q", 1, 0) * ufield("DXdyq", 1, 0)
                                        * (ufield("Idxu", 1, 0) + ufield("Idxu", 0, 0)))
                                       + (ufield("dx2q", 0, 0) * ufield("DXdyq", 0, 0)
                                          * (ufield("Idxu", 0, 0) + ufield("Idxu", -1, 0)))))
    u0v = (ufield("idxdy2_u", 0, 0) * ((ufield("dy2t", 0, 0) * ufield("DXdyT", 0, 0)
                                        * (ufield("Idxv", 1, 0) + ufield("Idxv", 0, 0)))
                                       + (ufield("dy2t", 0, -1) * ufield("DXdyT", 0, -1)
                                          * (ufield("Idxv", 1, -1) + ufield("Idxv", 0, -1))))) \
        + (ufield("idx2dy_u", 0, 0) * ((ufield("dx2q", 1, 0) * ufield("DYdxq", 1, 0)
                                        * (ufield("Idyv", 1, 0) + ufield("Idyv", 1, -1)))
                                       + (ufield("dx2q", 0, 0) * ufield("DYdxq", 0, 0)
                                          * (ufield("Idyv", 0, 0) + ufield("Idyv", 0, -1)))))
    nv_r, nv_c = ny + 1, nx

    def vfield(name, dj, di):
        return at(name, dj, di, nv_r, nv_c)

    v0u = (vfield("idxdy2_v", 0, 0) * ((vfield("dy2q", 0, 1) * vfield("DXdyq", 0, 1)
                                        * (vfield("Idxu", 0, 1) + vfield("Idxu", -1, 1)))
                                       + (vfield("dy2q", 0, 0) * vfield("DXdyq", 0, 0)
                                          * (vfield("Idxu", 0, 0) + vfield("Idxu", -1, 0))))) \
        + (vfield("idx2dy_v", 0, 0) * ((vfield("dx2t", 0, 0) * vfield("DYdxT", 0, 0)
                                        * (vfield("Idyu", 0, 1) + vfield("Idyu", 0, 0)))
                                       + (vfield("dx2t", -1, 0) * vfield("DYdxT", -1, 0)
                                          * (vfield("Idyu", -1, 1) + vfield("Idyu", -1, 0)))))
    v0v = (vfield("idxdy2_v", 0, 0) * ((vfield("dy2q", 0, 1) * vfield("DYdxq", 0, 1)
                                        * (vfield("Idyv", 0, 1) + vfield("Idyv", 0, 0)))
                                       + (vfield("dy2q", 0, 0) * vfield("DYdxq", 0, 0)
                                          * (vfield("Idyv", 0, 0) + vfield("Idyv", 0, -1))))) \
        + (vfield("idx2dy_v", 0, 0) * ((vfield("dx2t", 0, 0) * vfield("DXdyT", 0, 0)
                                        * (vfield("Idxv", 1, 0) + vfield("Idxv", 0, 0)))
                                       + (vfield("dx2t", -1, 0) * vfield("DXdyT", -1, 0)
                                          * (vfield("Idxv", 0, 0) + vfield("Idxv", -1, 0)))))
    g_ = g
    # denominators at h (:3557-3566) and q (:3576-3585)
    u0u_p, u0v_p = np.pad(u0u, 1, mode="edge"), np.pad(u0v, 1, mode="edge")
    v0u_p, v0v_p = np.pad(v0u, 1, mode="edge"), np.pad(v0v, 1, mode="edge")

    def sl(a, dj, di, nr, nc):
        return a[1 + dj:1 + dj + nr, 1 + di:1 + di + nc]

    idy_u, idx_v, idx_u, idy_v = g_["idy_u"], g_["idx_v"], g_["idx_u"], g_["idy_v"]
    ia_u, ia_v = g_["iarea_u"], g_["iarea_v"]
    # h point (j, i): u faces i+1 (east), i (west); v faces j+1 (north), j (south)
    ue, uw = np.s_[:, 1:], np.s_[:, :-1]
    vn, vs = np.s_[1:, :], np.s_[:-1, :]
    den1 = g_["dy2_t"] * (((g_["dy_dx_t"] * ((idy_u[ue] * u0u[ue]) + (idy_u[uw] * u0u[uw])))
                           + (g_["dx_dy_t"] * ((idx_v[vn] * v0u[vn]) + (idx_v[vs] * v0u[vs]))))
                          * np.maximum(idy_u[ue] * ia_u[ue], idy_u[uw] * ia_u[uw]))
    den2 = g_["dx2_t"] * (((g_["dy_dx_t"] * ((idy_u[ue] * u0v[ue]) + (idy_u[uw] * u0v[uw])))
                           + (g_["dx_dy_t"] * ((idx_v[vn] * v0v[vn]) + (idx_v[vs] * v0v[vs]))))
                          * np.maximum(idx_v[vn] * ia_v[vn], idx_v[vs] * ia_v[vs]))
    b["inv_denom_t"] = 1.0 / np.maximum(den1, den2)
    # q point (J, I): u faces (J, I) north, (J−1, I) south; v faces (J, I) east, (J, I−1) west — beyond the domain
    # edge the padded (repeated) values
    nqr, nqc = ny + 1, nx + 1
    pidx_u, pia_u = np.pad(idx_u, 1, mode="edge"), np.pad(ia_u, 1, mode="edge")
    pidy_v, pia_v = np.pad(idy_v, 1, mode="edge"), np.pad(ia_v, 1, mode="edge")
    un_ = lambda a: sl(a, 0, 0, nqr, nqc)               # noqa: E731  u face (J, I)
    us_ = lambda a: sl(a, -1, 0, nqr, nqc)              # noqa: E731  u face (J−1, I)
    ve_ = lambda a: sl(a, 0, 0, nqr, nqc)               # noqa: E731  v face (J, I)
    vw_ = lambda a: sl(a, 0, -1, nqr, nqc)              # noqa: E731  v face (J, I−1)
    den1 = g_["dx2_q"] * (((g_["dx_dy_q"] * ((un_(u0u_p) * un_(pidx_u)) + (us_(u0u_p) * us_(pidx_u))))
                           + (g_["dy_dx_q"] * ((ve_(v0u_p) * ve_(pidy_v)) + (vw_(v0u_p) * vw_(pidy_v)))))
                          * np.maximum(us_(pidx_u) * us_(pia_u), un_(pidx_u) * un_(pia_u)))
    den2 = g_["dy2_q"] * (((g_["dx_dy_q"] * ((un_(u0v_p) * un_(pidx_u)) + (us_(u0v_p) * us_(pidx_u))))
                           + (g_["dy_dx_q"] * ((ve_(v0v_p) * ve_(pidy_v)) + (vw_(v0v_p) * vw_(pidy_v)))))
                          * np.maximum(vw_(pidy_v) * vw_(pia_v), ve_(pidy_v) * ve_(pia_v)))
    b["inv_denom_q"] = 1.0 / np.maximum(den1, den2)
    for k in list(b):
        b[k] = np.ascontiguousarray(b[k], dtype=float)
    geom["bih"] = b
    return b


def _bih_bound_factor(dt: float) -> float:
    """0.8 · ½ · (6/11) / Δt [s⁻¹]; times inv_denom and hrat_min gives the A_h bound [m⁴ s⁻¹] (BOUND_AH, AB3)."""
    return (BOUND_COEF * 0.5 * AB3_STABILITY) / dt


def _bih_prepare(u, v, wet_u, wet_v, mask_q, open_u, open_v, obc_q):
    """Masked velocities and the active-corner set shared by the oracle and the kernel driver."""
    ny, nx = wet_u.shape[0], wet_v.shape[1]
    open_u = np.asarray(open_u, bool)
    open_v = np.asarray(open_v, bool)
    act_q = np.zeros((ny + 1, nx + 1), bool)
    act_q[1:-1, 1:-1] = (np.asarray(mask_q, bool)[1:-1, 1:-1] & ~open_u[:-1, 1:-1] & ~open_u[1:, 1:-1]
                         & ~open_v[1:-1, :-1] & ~open_v[1:-1, 1:])
    ob = np.zeros((ny + 1, nx + 1), bool) if obc_q is None else np.asarray(obc_q, bool)
    return open_u, open_v, act_q, ob


# ---------------------------------------------------------------- NumPy oracle
def biharmonic_numpy(u: np.ndarray, v: np.ndarray, h_t: np.ndarray, h_u: np.ndarray, h_v: np.ndarray,
                     wet_u: np.ndarray, wet_v: np.ndarray, mask_q: np.ndarray,
                     open_u: np.ndarray, open_v: np.ndarray, geom: dict[str, np.ndarray],
                     C: float, ah_vel_scale: float, ah: float, dt: float, h_neglect: float = H_NEGLECT,
                     obc_q: np.ndarray | None = None, obc_strain: tuple | None = None,
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Biharmonic Smagorinsky viscous tendencies (reference implementation; equations in the block comment above
    ``biharmonic_geometry``; MOM_hor_visc.F90 BIHARMONIC + SMAGORINSKY_AH + BOUND_AH + OBC_ZERO_BIHARMONIC).

    Parameters
    ----------
    u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom, h_neglect, obc_q, obc_strain :
        as ``smagorinsky_numpy`` (open faces must be marked wet; ``obc_q``/``obc_strain`` the open-edge corners of
        ``obc_tangential='mom6'`` with their computed D_S, used in |S| and in Del2).
    C : SMAG_BI_CONST (dimensionless; NWA12 0.03).
    ah_vel_scale : AH_VEL_SCALE [m s⁻¹], background A_h = ah_vel_scale Δ³ (NWA12 0.01).
    ah : AH [m⁴ s⁻¹], constant background.
    dt : time step of the explicit (AB3) momentum step [s], for the bound.

    Returns
    -------
    fu, fv : tendencies [m s⁻²]; ah_t (N, ny, nx), ah_q (N, ny+1, nx+1) [m⁴ s⁻¹] (0 where there is no stress);
    n_limited (int): active h and q points where the bound strictly reduced A_h.
    """
    g, bg = geom, biharmonic_geometry(geom)
    n, ny, nx1 = np.shape(u)
    nx = nx1 - 1
    hn = float(h_neglect)
    hn3 = hn * hn * hn
    open_u, open_v, act_q, ob = _bih_prepare(u, v, wet_u, wet_v, mask_q, open_u, open_v, obc_q)
    um = np.where(wet_u[None], u, 0.0)
    vm = np.where(wet_v[None], v, 0.0)
    # strains (MOM_hor_visc.F90:707-722, 927), as the Laplacian
    dudx = g["dy_dx_t"] * ((um[..., 1:] * g["idy_u"][:, 1:]) - (um[..., :-1] * g["idy_u"][:, :-1]))
    dvdy = g["dx_dy_t"] * ((vm[:, 1:] * g["idx_v"][1:]) - (vm[:, :-1] * g["idx_v"][:-1]))
    DT = dudx - dvdy
    DS = np.zeros((n, ny + 1, nx + 1))
    idy_v, idx_u, idx_v, idy_u = g["idy_v"], g["idx_u"], g["idx_v"], g["idy_u"]
    dvdx = g["dy_dx_q"][1:-1, 1:-1] * ((vm[:, 1:-1, 1:] * idy_v[1:-1, 1:]) - (vm[:, 1:-1, :-1] * idy_v[1:-1, :-1]))
    dudy = g["dx_dy_q"][1:-1, 1:-1] * ((um[:, 1:, 1:-1] * idx_u[1:, 1:-1]) - (um[:, :-1, 1:-1] * idx_u[:-1, 1:-1]))
    DS[:, 1:-1, 1:-1] = np.where(act_q[None, 1:-1, 1:-1], dvdx + dudy, 0.0)
    if ob.any():
        DS = np.where(ob[None], obc_strain[0], DS)

    # Del2u, Del2v at every face (MOM_hor_visc.F90:948-954; D_T = 0 outside the domain), zero on open faces (:956-970)
    zc = np.zeros((n, ny, 1))
    eT = np.concatenate([g["dy2_t"][None] * DT, zc], axis=2)          # east cell of face I
    wT = np.concatenate([zc, g["dy2_t"][None] * DT], axis=2)          # west cell
    xy = (g["dx2_q"][1:] * DS[:, 1:]) - (g["dx2_q"][:-1] * DS[:, :-1])
    d2u = (bg["idx2dy_u"] * xy) + (bg["idxdy2_u"] * (eT - wT))
    d2u = np.where(open_u[None], 0.0, d2u)
    zr = np.zeros((n, 1, nx))
    nT = np.concatenate([g["dx2_t"][None] * DT, zr], axis=1)
    sT = np.concatenate([zr, g["dx2_t"][None] * DT], axis=1)
    xy = (g["dy2_q"][:, 1:] * DS[..., 1:]) - (g["dy2_q"][:, :-1] * DS[..., :-1])
    d2v = (bg["idxdy2_v"] * xy) - (bg["idx2dy_v"] * (nT - sT))
    d2v = np.where(open_v[None], 0.0, d2v)

    bf = _bih_bound_factor(dt)
    # ---- h points (MOM_hor_visc.F90:1018-1037, 1187-1306)
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
    bgr_t = max(float(ah), 0.0)
    ahbg_t = np.maximum(bgr_t, float(ah_vel_scale) * bg["delta3_t"])
    raw_t = np.maximum(ahbg_t, (float(C) * bg["delta4_t"]) * S_t)
    bnd_t = hrat_t * (bf * bg["inv_denom_t"])
    lim_t = (raw_t > bnd_t) & act_t[None]
    ah_t = np.where(act_t[None], np.where(raw_t > bnd_t, bnd_t, raw_t), 0.0)
    d_del2u = (idy_u[:, 1:] * d2u[..., 1:]) - (idy_u[:, :-1] * d2u[..., :-1])
    d_del2v = (idx_v[1:] * d2v[:, 1:]) - (idx_v[:-1] * d2v[:, :-1])
    sxx = (ah_t * ((g["dy_dx_t"] * d_del2u) - (g["dx_dy_t"] * d_del2v))) * h_t

    # ---- q points (interior active corners; MOM_hor_visc.F90:1313-1318, 1353-1371, 1581-1672)
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
    ahbg_q = np.maximum(bgr_t, float(ah_vel_scale) * bg["delta3_q"][1:-1, 1:-1])
    raw_q = np.maximum(ahbg_q, (float(C) * bg["delta4_q"][1:-1, 1:-1]) * S_q)
    bnd_q = hrat_q * (bf * bg["inv_denom_q"][1:-1, 1:-1])
    aq = act_q[None, 1:-1, 1:-1]
    lim_q = (raw_q > bnd_q) & aq
    ah_q = np.zeros((n, ny + 1, nx + 1))
    ah_q[:, 1:-1, 1:-1] = np.where(aq, np.where(raw_q > bnd_q, bnd_q, raw_q), 0.0)
    ddvdx = g["dy_dx_q"][1:-1, 1:-1] * ((d2v[:, 1:-1, 1:] * idy_v[1:-1, 1:]) - (d2v[:, 1:-1, :-1] * idy_v[1:-1, :-1]))
    ddudy = g["dx_dy_q"][1:-1, 1:-1] * ((d2u[:, 1:, 1:-1] * idx_u[1:, 1:-1]) - (d2u[:, :-1, 1:-1] * idx_u[:-1, 1:-1]))
    sxy = np.zeros((n, ny + 1, nx + 1))
    sxy[:, 1:-1, 1:-1] = (ah_q[:, 1:-1, 1:-1] * (ddvdx + ddudy)) * hq

    # ---- divergence (MOM_hor_visc.F90:1755-1760, 1777-1781): F = −∇·(str)/h
    fu = np.zeros((n, ny, nx + 1))
    xy = (g["dx2_q"][:-1, 1:-1] * sxy[:, :-1, 1:-1]) - (g["dx2_q"][1:, 1:-1] * sxy[:, 1:, 1:-1])
    xx = (g["dy2_t"][:, :-1] * sxx[..., :-1]) - (g["dy2_t"][:, 1:] * sxx[..., 1:])
    fu[..., 1:-1] = (((idx_u[:, 1:-1] * xy) + (idy_u[:, 1:-1] * xx)) * g["iarea_u"][:, 1:-1]) / (h_u[..., 1:-1] + hn)
    fu = np.where((wet_u & ~open_u)[None], fu, 0.0)
    fv = np.zeros((n, ny + 1, nx))
    xy = (g["dy2_q"][1:-1, :-1] * sxy[:, 1:-1, :-1]) - (g["dy2_q"][1:-1, 1:] * sxy[:, 1:-1, 1:])
    xx = (g["dx2_t"][:-1] * sxx[:, :-1]) - (g["dx2_t"][1:] * sxx[:, 1:])
    fv[:, 1:-1] = (((idy_v[1:-1] * xy) - (idx_v[1:-1] * xx)) * g["iarea_v"][1:-1]) / (h_v[:, 1:-1] + hn)
    fv = np.where((wet_v & ~open_v)[None], fv, 0.0)
    return fu, fv, ah_t, ah_q, int(lim_t.sum() + lim_q.sum())


# ---------------------------------------------------------------- Numba kernel
@numba.njit(parallel=True, cache=True)
def _bih_kernel(u, v, h_t, h_u, h_v, wet_u, wet_v, open_u, open_v, act_q, obc_q, ds_obc, fuok, fvok,
                dy_dx_t, dx_dy_t, dy_dx_q, dx_dy_q, idx_u, idy_u, idx_v, idy_v,
                dx2_t, dy2_t, dx2_q, dy2_q, iarea_u, iarea_v, idx2dy_u, idxdy2_u, idx2dy_v, idxdy2_v,
                delta3_t, delta4_t, delta3_q, delta4_q, inv_denom_t, inv_denom_q,
                C, vel, bgr, bf, hn, DT, DS, d2u, d2v, ah_t, ah_q, sxx, sxy, fu, fv, lim_t, lim_q):  # pragma: no cover
    """biharmonic_numpy with the same per-element arithmetic, parallel over (layer, row).  Pass 1: D_T, D_S;
    pass 2: Del2u, Del2v; pass 3: A_h and the stresses at h and q; pass 4: F_u, F_v."""
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    hn3 = hn * hn * hn
    for p in numba.prange(n * ny):                                   # 1a tension
        k, j = p // ny, p % ny
        for i in range(nx):
            ue = u[k, j, i + 1] if wet_u[j, i + 1] else 0.0
            uw = u[k, j, i] if wet_u[j, i] else 0.0
            vn = v[k, j + 1, i] if wet_v[j + 1, i] else 0.0
            vs = v[k, j, i] if wet_v[j, i] else 0.0
            dudx = dy_dx_t[j, i] * ((ue * idy_u[j, i + 1]) - (uw * idy_u[j, i]))
            dvdy = dx_dy_t[j, i] * ((vn * idx_v[j + 1, i]) - (vs * idx_v[j, i]))
            DT[k, j, i] = dudx - dvdy
    for p in numba.prange(n * (ny + 1)):                             # 1b shearing strain
        k, J = p // (ny + 1), p % (ny + 1)
        for I in range(nx1):
            if obc_q[J, I]:
                DS[k, J, I] = ds_obc[k, J, I]
            elif act_q[J, I]:
                ve = v[k, J, I] if wet_v[J, I] else 0.0
                vw = v[k, J, I - 1] if wet_v[J, I - 1] else 0.0
                un = u[k, J, I] if wet_u[J, I] else 0.0
                us = u[k, J - 1, I] if wet_u[J - 1, I] else 0.0
                dvdx = dy_dx_q[J, I] * ((ve * idy_v[J, I]) - (vw * idy_v[J, I - 1]))
                dudy = dx_dy_q[J, I] * ((un * idx_u[J, I]) - (us * idx_u[J - 1, I]))
                DS[k, J, I] = dvdx + dudy
            else:
                DS[k, J, I] = 0.0
    for p in numba.prange(n * ny):                                   # 2a Del2u
        k, j = p // ny, p % ny
        for I in range(nx1):
            if open_u[j, I]:
                d2u[k, j, I] = 0.0
            else:
                xy = (dx2_q[j + 1, I] * DS[k, j + 1, I]) - (dx2_q[j, I] * DS[k, j, I])
                e = dy2_t[j, I] * DT[k, j, I] if I < nx else 0.0
                w = dy2_t[j, I - 1] * DT[k, j, I - 1] if I > 0 else 0.0
                d2u[k, j, I] = (idx2dy_u[j, I] * xy) + (idxdy2_u[j, I] * (e - w))
    for p in numba.prange(n * (ny + 1)):                             # 2b Del2v
        k, J = p // (ny + 1), p % (ny + 1)
        for i in range(nx):
            if open_v[J, i]:
                d2v[k, J, i] = 0.0
            else:
                xy = (dy2_q[J, i + 1] * DS[k, J, i + 1]) - (dy2_q[J, i] * DS[k, J, i])
                nt = dx2_t[J, i] * DT[k, J, i] if J < ny else 0.0
                st = dx2_t[J - 1, i] * DT[k, J - 1, i] if J > 0 else 0.0
                d2v[k, J, i] = (idxdy2_v[J, i] * xy) - (idx2dy_v[J, i] * (nt - st))
    for p in numba.prange(n * ny):                                   # 3a A_h and str_xx at h
        k, j = p // ny, p % ny
        cnt = 0
        for i in range(nx):
            if wet_u[j, i] or wet_u[j, i + 1] or wet_v[j, i] or wet_v[j + 1, i]:
                a = DS[k, j, i] * DS[k, j, i]
                b = DS[k, j + 1, i + 1] * DS[k, j + 1, i + 1]
                c = DS[k, j + 1, i] * DS[k, j + 1, i]
                d = DS[k, j, i + 1] * DS[k, j, i + 1]
                dt_ = DT[k, j, i]
                S = np.sqrt((dt_ * dt_) + 0.25 * ((a + b) + (c + d)))
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
                raw = max(max(bgr, vel * delta3_t[j, i]), (C * delta4_t[j, i]) * S)
                bnd = hrat * (bf * inv_denom_t[j, i])
                if raw > bnd:
                    ah = bnd
                    cnt += 1
                else:
                    ah = raw
            else:
                ah = 0.0
            ddu = (idy_u[j, i + 1] * d2u[k, j, i + 1]) - (idy_u[j, i] * d2u[k, j, i])
            ddv = (idx_v[j + 1, i] * d2v[k, j + 1, i]) - (idx_v[j, i] * d2v[k, j, i])
            ah_t[k, j, i] = ah
            sxx[k, j, i] = (ah * ((dy_dx_t[j, i] * ddu) - (dx_dy_t[j, i] * ddv))) * h_t[k, j, i]
        lim_t[p] = cnt
    for p in numba.prange(n * (ny + 1)):                             # 3b A_h and str_xy at q
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
                raw = max(max(bgr, vel * delta3_q[J, I]), (C * delta4_q[J, I]) * S)
                bnd = hrat * (bf * inv_denom_q[J, I])
                if raw > bnd:
                    ah = bnd
                    cnt += 1
                else:
                    ah = raw
                ddvdx = dy_dx_q[J, I] * ((d2v[k, J, I] * idy_v[J, I]) - (d2v[k, J, I - 1] * idy_v[J, I - 1]))
                ddudy = dx_dy_q[J, I] * ((d2u[k, J, I] * idx_u[J, I]) - (d2u[k, J - 1, I] * idx_u[J - 1, I]))
                ah_q[k, J, I] = ah
                sxy[k, J, I] = (ah * (ddvdx + ddudy)) * hq
            else:
                ah_q[k, J, I] = 0.0
                sxy[k, J, I] = 0.0
        lim_q[p] = cnt
    for p in numba.prange(n * ny):                                   # 4a F_u
        k, j = p // ny, p % ny
        for I in range(nx1):
            if 0 < I < nx and fuok[j, I]:
                xy = (dx2_q[j, I] * sxy[k, j, I]) - (dx2_q[j + 1, I] * sxy[k, j + 1, I])
                xx = (dy2_t[j, I - 1] * sxx[k, j, I - 1]) - (dy2_t[j, I] * sxx[k, j, I])
                fu[k, j, I] = (((idx_u[j, I] * xy) + (idy_u[j, I] * xx)) * iarea_u[j, I]) / (h_u[k, j, I] + hn)
            else:
                fu[k, j, I] = 0.0
    for p in numba.prange(n * (ny + 1)):                             # 4b F_v
        k, J = p // (ny + 1), p % (ny + 1)
        for i in range(nx):
            if 0 < J < ny and fvok[J, i]:
                xy = (dy2_q[J, i] * sxy[k, J, i]) - (dy2_q[J, i + 1] * sxy[k, J, i + 1])
                xx = (dx2_t[J - 1, i] * sxx[k, J - 1, i]) - (dx2_t[J, i] * sxx[k, J, i])
                fv[k, J, i] = (((idy_v[J, i] * xy) - (idx_v[J, i] * xx)) * iarea_v[J, i]) / (h_v[k, J, i] + hn)
            else:
                fv[k, J, i] = 0.0


def biharmonic(u: np.ndarray, v: np.ndarray, h_t: np.ndarray, h_u: np.ndarray, h_v: np.ndarray,
               wet_u: np.ndarray, wet_v: np.ndarray, mask_q: np.ndarray,
               open_u: np.ndarray, open_v: np.ndarray, geom: dict[str, np.ndarray],
               C: float, ah_vel_scale: float, ah: float, dt: float, h_neglect: float = H_NEGLECT,
               obc_q: np.ndarray | None = None, obc_strain: tuple | None = None,
               ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Biharmonic Smagorinsky viscous tendencies, Numba kernel (same signature, outputs and bits as
    ``biharmonic_numpy``, the reference; any thread count gives the same result).
    """
    numba.set_num_threads(_THREADS[0])
    g, bg = geom, biharmonic_geometry(geom)
    u = np.ascontiguousarray(u, dtype=float)
    v = np.ascontiguousarray(v, dtype=float)
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    wet_u = np.ascontiguousarray(wet_u, dtype=np.bool_)
    wet_v = np.ascontiguousarray(wet_v, dtype=np.bool_)
    open_u, open_v, act_q, ob = _bih_prepare(u, v, wet_u, wet_v, mask_q, open_u, open_v, obc_q)
    ds_obc = (np.ascontiguousarray(obc_strain[0], dtype=float) if ob.any() else np.zeros((1, 1, 1)))
    fuok = np.ascontiguousarray(wet_u & ~open_u)
    fvok = np.ascontiguousarray(wet_v & ~open_v)
    DT, sxx, ah_t = (np.empty((n, ny, nx)) for _ in range(3))
    DS, sxy, ah_q = (np.empty((n, ny + 1, nx + 1)) for _ in range(3))
    d2u, fu = np.empty((n, ny, nx + 1)), np.empty((n, ny, nx + 1))
    d2v, fv = np.empty((n, ny + 1, nx)), np.empty((n, ny + 1, nx))
    lim_t = np.zeros(n * ny, np.int64)
    lim_q = np.zeros(n * (ny + 1), np.int64)
    _bih_kernel(u, v, np.ascontiguousarray(h_t, dtype=float), np.ascontiguousarray(h_u, dtype=float),
                np.ascontiguousarray(h_v, dtype=float), wet_u, wet_v, np.ascontiguousarray(open_u),
                np.ascontiguousarray(open_v), np.ascontiguousarray(act_q), np.ascontiguousarray(ob), ds_obc, fuok, fvok,
                g["dy_dx_t"], g["dx_dy_t"], g["dy_dx_q"], g["dx_dy_q"], g["idx_u"], g["idy_u"], g["idx_v"], g["idy_v"],
                g["dx2_t"], g["dy2_t"], g["dx2_q"], g["dy2_q"], g["iarea_u"], g["iarea_v"],
                bg["idx2dy_u"], bg["idxdy2_u"], bg["idx2dy_v"], bg["idxdy2_v"],
                bg["delta3_t"], bg["delta4_t"], bg["delta3_q"], bg["delta4_q"], bg["inv_denom_t"], bg["inv_denom_q"],
                float(C), float(ah_vel_scale), max(float(ah), 0.0), _bih_bound_factor(float(dt)), float(h_neglect),
                DT, DS, d2u, d2v, ah_t, ah_q, sxx, sxy, fu, fv, lim_t, lim_q)
    return fu, fv, ah_t, ah_q, int(lim_t.sum() + lim_q.sum())
