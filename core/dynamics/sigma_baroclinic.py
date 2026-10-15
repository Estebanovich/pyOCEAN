"""
Stratified σ-layer model with wetting and drying.

Extends the homogeneous ``SigmaWetDry`` (Casulli & Cheng 1992 fractional step)
with an active temperature tracer, a linear equation of state, the baroclinic
pressure gradient and momentum advection on σ-surfaces.  Each step:

1. Explicit momentum tendencies G_k, combined with AB3 (Durran 2010, §2.4;
   MITgcm's AB3, Adcroft et al. 2004 [verificar]):
       G = −(u·∇_σ)u − (ω/h) ∂u/∂k + f k×u (energy conserving) + A_h ∇²u + PGF_bc
2. ``SigmaWetDry.step``: implicit vertical viscosity + bed stress, θ-implicit
   free surface on the depth-mean, uniform barotropic correction.
3. Layer volume transports Q_k = L D_f Δσ_k (θ u_k^{n+1} + (1−θ) u_k^n), whose
   vertical sum is exactly the free-surface flux, so
       ω_{k+½} = ω_{k−½} − [ (V_k^{n+1} − V_k^n)/Δt + ∇_h·Q_k ] / A
   vanishes at the surface and a uniform tracer stays uniform (constancy).
4. Tracers in flux form with WENO5 face values (Jiang & Shu 1996; the CROCO
   GRAV_ADJ choice) — or UP3 (Leonard 1979), which overshoots by ~50 % at a
   lock-exchange nose, or first-order upwind — lower order next to walls, advanced
   with SSP-RK3 (Shu & Osher 1988) — or a predictor (centred 6th order, half step)
   / corrector (WENO at mid-step, full step), ``tracer_time="pc"`` — on frozen
   transports, layer volumes linear in
   time.  T and T² are advected together, giving the online numerical mixing of
   Burchard & Rennau (2008): M_num = V (A(T²) − (A T)²)/Δt.

Baroclinic pressure gradient on σ-surfaces (hydrostatic, Boussinesq; default
density-Jacobian form of Song 1998, see ``_baroclinic_pgf``):

    PGF_x = −(1/ρ₀) [ ∂P′/∂x|_σ + g ρ′ ∂z/∂x|_σ ],   P′(z) = g ∫_z^η ρ′ dz′,   ρ′ = ρ − ρ₀

(second-order two-term form; exact for horizontally uniform ρ over a flat bed.
Over sloping beds use the finite-volume PGF of Adcroft et al. 2008 — roadmap
stage B, seamount test).  Equation of state: ρ = ρ₀ − α (T − T₀), as CROCO
``lin_EOS_cff``.

Surface (wind) stress τ enters as the top flux of the implicit vertical viscosity
(``step(st, tau_x, tau_y)``; Ekman transport τ/ρ₀f to 0.05 %).

Open boundaries (``open_boundaries``, any of west/east/south/north), after Chapman
(1985) and Marsaleix et al. (2006): Flather (1976) radiation of the depth mean to a
resting exterior; Orlanski (1976) radiation of the baroclinic normal velocity;
upstream tracer values at the open faces (external T on inflow); layer transports
there equal to Δσ × the free-surface flux + the baroclinic part, so ω and tracer
constancy are exact; optional flow-relaxation sponge (Martinsen & Engedahl 1987)
with the MITgcm OBCS profile on the baroclinic velocity and the tracer anomaly.
Mode-1 internal-wave test (16 h, 60 km channel vs a 300 km truth): interior error
0.82 closed, 0.16 Orlanski only, 0.015 with a 10-cell sponge.

Vertical grid: σ-layers with any fixed fractions Δσ_k (``dsigma``; uniform by
default, Song & Haidvogel 1994 stretching or MITgcm ``delR`` via
core.grid.sigma_levels); the PGF, ω and transports use the actual layer
heights, the vertical WENO/UP3 reconstruction works in layer-index space (as in
CROCO).

Limitations (documented, tested where it matters): Cartesian uniform spacing for
the UP3/WENO face interpolation; fractions are the same in every column (no
ROMS-type h_c / Vtransform 2 yet); salinity not included; the sponge relaxes towards
a resting exterior (no prescribed external baroclinic state yet).

References
----------
Burchard, H. & Rennau, H. (2008). Ocean Modell. 20, 293–311.
Casulli, V. & Cheng, R.T. (1992). Int. J. Numer. Meth. Fluids 15, 629–648.
Shchepetkin, A.F. & McWilliams, J.C. (2005). Ocean Modell. 9, 347–404.
Shu, C.-W. & Osher, S. (1988). J. Comput. Phys. 77, 439–471.
Jiang, G.-S. & Shu, C.-W. (1996). Efficient implementation of weighted ENO
    schemes. J. Comput. Phys. 126, 202–228.
Chapman, D.C. (1985). J. Phys. Oceanogr. 15, 1060–1075.
Flather, R.A. (1976). Mém. Soc. R. Sci. Liège 6(10), 141–164.
Marsaleix, P., Auclair, F. & Estournel, C. (2006). J. Atmos. Oceanic Technol. 23, 1604–1613.
Martinsen, E.A. & Engedahl, H. (1987). Coastal Eng. 11, 603–627.
Orlanski, I. (1976). J. Comput. Phys. 21, 251–269.
Borges, R., Carmona, M., Costa, B. & Don, W.S. (2008). An improved weighted
    essentially non-oscillatory scheme for hyperbolic conservation laws.
    J. Comput. Phys. 227, 3191–3211.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field

import numba
import numpy as np
import xarray as xr
from numba import njit, prange

from core.dynamics import vector_invariant as vi
from core.dynamics.nonlinear import _spacings_u, _spacings_v, advection_u, advection_v, laplacian_3d
from core.dynamics.sigma_wd import SigmaWetDry, vertical_implicit
from diagnostics.mixing import reference_potential_energy
from core.vertical import regrid as rg
from core.vertical import remap as rm
from core.dynamics import open_boundary as obc
from core.vertical.ale import (DZ_FLOOR, baroclinic_split, blend_thin_layers, exterior_on_layers, face_fractions,
                               face_thickness, face_transport_thickness, lagrangian_thickness, limit_outflow,
                               open_face_transport, ppm_transport_x, transport_velocity, upwind_transport_x,
                               viscous_face_thickness, zero_thin_bed_velocity)
from core.vertical.pgf import fv_pgf

G = 9.81
AB3 = (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0)
REMAP_SCHEMES = {"pcm": rm.PCM, "plm": rm.PLM, "ppm_h4": rm.PPM_H4}


@dataclass
class SigmaState:
    """Prognostic state: η (ny,nx), u (N,ny,nx+1), v (N,ny+1,nx), T (N,ny,nx); diagnostics."""
    eta: np.ndarray
    u: np.ndarray
    v: np.ndarray
    T: np.ndarray
    h: np.ndarray | None = None                              # ALE layer thickness (N,ny,nx) [m], k = 0 at the bed
    g_hist: list = field(default_factory=list)              # past explicit tendencies (AB3)
    omega: np.ndarray | None = None                          # ω at interfaces (N+1,ny,nx) [m s⁻¹]
    mix_num: np.ndarray | None = None                        # volume-weighted numerical mixing [T² m³ s⁻¹]
    Qx: np.ndarray | None = None                             # layer volume transports of the last step [m³ s⁻¹]
    Qy: np.ndarray | None = None
    Fz: np.ndarray | None = None
    time: float = 0.0
    n_trunc: int = 0                                         # CFL velocity truncations so far (ALE, counted faces)
    n_limited: int = 0                                       # cell-layers scaled by the outflow limiter so far (ALE)


def _up3_faces(c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int, order: int = 3) -> np.ndarray:
    """
    Face values of c on the interior faces along ``axis`` (between i−1 and i) for
    volume flux ``flux`` (sign gives direction): UP3 where the 4-point stencil is
    valid (``ok`` cells), first-order upwind otherwise.  Uniform spacing assumed.
    """
    c = np.moveaxis(c, axis, -1)
    ok = np.moveaxis(ok, axis, -1)
    f = np.moveaxis(flux, axis, -1)
    n = c.shape[-1]
    lo, hi = c[..., :-1], c[..., 1:]                        # cells i−1, i for faces 1..n−1
    olo, ohi = ok[..., :-1], ok[..., 1:]
    llo = np.concatenate([c[..., :1], c[..., :-2]], axis=-1)            # i−2
    hhi = np.concatenate([c[..., 2:], c[..., -1:]], axis=-1)            # i+1
    ollo = np.concatenate([np.zeros_like(ok[..., :1]), ok[..., :-2]], axis=-1)
    ohhi = np.concatenate([ok[..., 2:], np.zeros_like(ok[..., :1])], axis=-1)
    if order == 1:
        pos, neg = lo, hi
    else:
        pos = np.where(ollo & olo & ohi, (-llo + 5.0 * lo + 2.0 * hi) / 6.0, lo)
        neg = np.where(olo & ohi & ohhi, (2.0 * lo + 5.0 * hi - hhi) / 6.0, hi)
    fi = f[..., 1:n]
    face = np.zeros(f.shape)
    face[..., 1:n] = np.where(fi >= 0, pos, neg)
    return np.moveaxis(face, -1, axis)


def _shift(c: np.ndarray, ok: np.ndarray, s: int) -> tuple[np.ndarray, np.ndarray]:
    """c[..., m + s] and its validity for m = 0 … n−1 (out of range → invalid)."""
    n = c.shape[-1]
    out = np.zeros_like(c)
    val = np.zeros(ok.shape, bool)
    lo, hi = max(0, -s), min(n, n - s)
    out[..., lo:hi] = c[..., lo + s:hi + s]
    val[..., lo:hi] = ok[..., lo + s:hi + s]
    return out, val


_THREADS = [1]


def set_threads(n: int) -> None:
    """
    Threads for the compiled kernels (default 1).  Rows are independent, so results are
    bit-identical for any thread count.  On a laptop keep the total over concurrent runs at
    or below the number of performance cores.
    """
    _THREADS[0] = max(1, min(int(n), numba.config.NUMBA_NUM_THREADS))


@njit(inline="always")
def _wval(v1, v2, v3, v4, v5, k1, k2, k3, k4, k5, z_variant):  # pragma: no cover
    """
    WENO5 value at a face from the upwind-ordered stencil v1 … v5 (v3 = upwind cell) and the
    validity of each cell: WENO5-Z or -JS when all five are valid, WENO3 when v2 … v4 are,
    else first-order upwind (see ``_weno5_faces_numpy`` for the formulas).
    """
    if k1 and k2 and k3 and k4 and k5:
        q0 = (2 * v1 - 7 * v2 + 11 * v3) / 6
        q1 = (-v2 + 5 * v3 + 2 * v4) / 6
        q2 = (2 * v3 + 5 * v4 - v5) / 6
        b0 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
        b1 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
        b2 = 13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2
        if z_variant:
            tau, ez = abs(b2 - b0), 1e-40
            a0 = 0.1 * (1 + tau / (ez + b0))
            a1 = 0.6 * (1 + tau / (ez + b1))
            a2 = 0.3 * (1 + tau / (ez + b2))
        else:
            a0 = 0.1 / (1e-6 + b0) ** 2
            a1 = 0.6 / (1e-6 + b1) ** 2
            a2 = 0.3 / (1e-6 + b2) ** 2
        return (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)
    if k2 and k3 and k4:
        c0, c1 = (-v2 + 3 * v3) / 2, (v3 + v4) / 2
        e0 = (1 / 3) / (1e-6 + (v3 - v2) ** 2) ** 2
        e1 = (2 / 3) / (1e-6 + (v4 - v3) ** 2) ** 2
        return (e0 * c0 + e1 * c1) / (e0 + e1)
    return v3


@njit(cache=True, parallel=True)
def _weno5_rows(c: np.ndarray, flux: np.ndarray, ok: np.ndarray, z_variant: bool) -> np.ndarray:  # pragma: no cover
    """
    Compiled WENO5 kernel on rows (M, n) of cells and (M, n_f) of face fluxes: the same
    reconstruction as ``_weno5_faces_numpy`` but only on the upwind side of each face
    (half the work) and without temporaries.  Faces 1 … n−1; the others stay 0.
    """
    M, n = c.shape
    out = np.zeros(flux.shape)
    for m in prange(M):
        for j in range(1, n):
            if flux[m, j] >= 0.0:
                i1, i2, i3, i4, i5 = j - 3, j - 2, j - 1, j, j + 1
            else:
                i1, i2, i3, i4, i5 = j + 2, j + 1, j, j - 1, j - 2
            k1 = 0 <= i1 < n and ok[m, i1]
            k2 = 0 <= i2 < n and ok[m, i2]
            k3 = 0 <= i3 < n and ok[m, i3]
            k4 = 0 <= i4 < n and ok[m, i4]
            k5 = 0 <= i5 < n and ok[m, i5]
            v1 = c[m, i1] if 0 <= i1 < n else 0.0
            v2 = c[m, i2] if 0 <= i2 < n else 0.0
            v3 = c[m, i3] if 0 <= i3 < n else 0.0
            v4 = c[m, i4] if 0 <= i4 < n else 0.0
            v5 = c[m, i5] if 0 <= i5 < n else 0.0
            out[m, j] = _wval(v1, v2, v3, v4, v5, k1, k2, k3, k4, k5, z_variant)
    return out


@njit(cache=True, parallel=True)
def _advect_faces_h(c, Qx, Qy, ok, ext, open_w, open_e, open_s, open_n, z_variant):  # pragma: no cover
    """
    Horizontal face fluxes Q·c_face of ``_advect_div`` (WENO5 upwind faces, upstream values on open edges),
    returned instead of differenced — also the high-order flux of the FCT (spec 2026-10-14 §4).
    """
    nt, n, ny, nx = c.shape
    fx = np.zeros((nt, n, ny, nx + 1))
    fy = np.zeros((nt, n, ny + 1, nx))
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for i in range(1, nx):
                q = Qx[k, j, i]
                if q == 0.0:                                     # no flux (land, closed): fx stays 0
                    continue
                if q >= 0.0:
                    i1, i2, i3, i4, i5 = i - 3, i - 2, i - 1, i, i + 1
                else:
                    i1, i2, i3, i4, i5 = i + 2, i + 1, i, i - 1, i - 2
                b1, b2, b3, b4, b5 = 0 <= i1 < nx, 0 <= i2 < nx, 0 <= i3 < nx, 0 <= i4 < nx, 0 <= i5 < nx
                v = _wval(c[t, k, j, i1] if b1 else 0.0, c[t, k, j, i2] if b2 else 0.0,
                          c[t, k, j, i3] if b3 else 0.0, c[t, k, j, i4] if b4 else 0.0,
                          c[t, k, j, i5] if b5 else 0.0,
                          b1 and ok[k, j, i1], b2 and ok[k, j, i2], b3 and ok[k, j, i3],
                          b4 and ok[k, j, i4], b5 and ok[k, j, i5], z_variant)
                fx[t, k, j, i] = q * v
            q = Qx[k, j, 0]
            if open_w:
                fx[t, k, j, 0] = q * (ext[t, k, j, 0] if q > 0.0 else c[t, k, j, 0])
            q = Qx[k, j, nx]
            if open_e:
                fx[t, k, j, nx] = q * (ext[t, k, j, nx - 1] if q < 0.0 else c[t, k, j, nx - 1])
        for j in range(1, ny):
            for i in range(nx):
                q = Qy[k, j, i]
                if q == 0.0:
                    continue
                if q >= 0.0:
                    j1, j2, j3, j4, j5 = j - 3, j - 2, j - 1, j, j + 1
                else:
                    j1, j2, j3, j4, j5 = j + 2, j + 1, j, j - 1, j - 2
                b1, b2, b3, b4, b5 = 0 <= j1 < ny, 0 <= j2 < ny, 0 <= j3 < ny, 0 <= j4 < ny, 0 <= j5 < ny
                v = _wval(c[t, k, j1, i] if b1 else 0.0, c[t, k, j2, i] if b2 else 0.0,
                          c[t, k, j3, i] if b3 else 0.0, c[t, k, j4, i] if b4 else 0.0,
                          c[t, k, j5, i] if b5 else 0.0,
                          b1 and ok[k, j1, i], b2 and ok[k, j2, i], b3 and ok[k, j3, i],
                          b4 and ok[k, j4, i], b5 and ok[k, j5, i], z_variant)
                fy[t, k, j, i] = q * v
        for i in range(nx):
            q = Qy[k, 0, i]
            if open_s:
                fy[t, k, 0, i] = q * (ext[t, k, 0, i] if q > 0.0 else c[t, k, 0, i])
            q = Qy[k, ny, i]
            if open_n:
                fy[t, k, ny, i] = q * (ext[t, k, ny - 1, i] if q < 0.0 else c[t, k, ny - 1, i])
    return fx, fy


@njit(cache=True, parallel=True)
def _advect_div(c, Qx, Qy, Fz, ok, ext, open_w, open_e, open_s, open_n, z_variant, vert):  # pragma: no cover
    """
    Fused flux-form advection RHS −∇·(Q c) of the tracer stack c (ntr, N, ny, nx) with WENO5 face
    values in x, y and z (same rules as ``_weno5_rows``) and upstream values on the open edges
    (``ext`` on inflow, the boundary cell on outflow; ``SigmaBaroclinic._open_faces``).  Face
    fluxes are computed once, in parallel over (tracer, layer) or (tracer, row), then differenced.
    """
    nt, n, ny, nx = c.shape
    fx, fy = _advect_faces_h(c, Qx, Qy, ok, ext, open_w, open_e, open_s, open_n, z_variant)
    fz = np.zeros((nt, n + 1, ny, nx)) if vert else np.zeros((nt, 1, 1, 1))   # ALE (Fz ≡ 0): no vertical pass
    if vert:
        for p in prange(nt * ny):
            t, j = p // ny, p % ny
            for k in range(1, n):
                for i in range(nx):
                    q = Fz[k, j, i]
                    if q == 0.0:                                     # ALE: no flux through the interfaces
                        continue
                    if q >= 0.0:
                        k1, k2, k3, k4, k5 = k - 3, k - 2, k - 1, k, k + 1
                    else:
                        k1, k2, k3, k4, k5 = k + 2, k + 1, k, k - 1, k - 2
                    b1, b2, b3, b4, b5 = 0 <= k1 < n, 0 <= k2 < n, 0 <= k3 < n, 0 <= k4 < n, 0 <= k5 < n
                    v = _wval(c[t, k1, j, i] if b1 else 0.0, c[t, k2, j, i] if b2 else 0.0,
                              c[t, k3, j, i] if b3 else 0.0, c[t, k4, j, i] if b4 else 0.0,
                              c[t, k5, j, i] if b5 else 0.0,
                              b1 and ok[k1, j, i], b2 and ok[k2, j, i], b3 and ok[k3, j, i],
                              b4 and ok[k4, j, i], b5 and ok[k5, j, i], z_variant)
                    fz[t, k, j, i] = q * v
    out = np.empty_like(c)
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for i in range(nx):
                if vert:
                    out[t, k, j, i] = -(fx[t, k, j, i + 1] - fx[t, k, j, i] + fy[t, k, j + 1, i] - fy[t, k, j, i]
                                        + fz[t, k + 1, j, i] - fz[t, k, j, i])
                else:
                    out[t, k, j, i] = -(fx[t, k, j, i + 1] - fx[t, k, j, i] + fy[t, k, j + 1, i] - fy[t, k, j, i])
    return out


def fct_coefficients_numpy(c0: np.ndarray, cL: np.ndarray, V1: np.ndarray, Ax: np.ndarray, Ay: np.ndarray,
                           ok: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Zalesak (1979, eqs. 7–13) flux-limiting coefficients C ∈ [0, 1] of the antidiffusive fluxes A = F_H − F_L
    (stack (nt, N, ny, nx); Ax (nt, N, ny, nx+1), Ay (nt, N, ny+1, nx) [tracer m³ s⁻¹]) so that
    V1 c = V1 c_L − Δt ∇·(C A) stays inside [c_min, c_max] = the extremes of c0 and c_L over the cell, its x, y
    neighbours and the layers k ± 1 of its column (eq. 15 with the multidimensional stencil, ``ok`` cells only).
    Bounds from the layer alone clipped legitimate along-layer extrema to donor cell (Zalesak's "clipping": σ layers
    bent over a seamount summit, 0.22 → 0.71 cm/s; hybrid internal wave 0.50 → 0.31 of the oscillation).  Cells not ``ok`` (dry, V1 = 0) get R± = 0, so the
    faces of a vanishing cell are upwind.  NumPy oracle of _fct_coeff_kernel.
    """
    big = np.inf
    okb = np.broadcast_to(ok, c0.shape)
    hi = np.where(okb, np.maximum(c0, cL), -big)
    lo = np.where(okb, np.minimum(c0, cL), big)

    def neigh(a, f):
        out = a.copy()
        out[..., 1:] = f(out[..., 1:], a[..., :-1]); out[..., :-1] = f(out[..., :-1], a[..., 1:])
        out[..., 1:, :] = f(out[..., 1:, :], a[..., :-1, :]); out[..., :-1, :] = f(out[..., :-1, :], a[..., 1:, :])
        out[:, 1:] = f(out[:, 1:], a[:, :-1]); out[:, :-1] = f(out[:, :-1], a[:, 1:])      # layers k ± 1
        return out

    cmax, cmin = neigh(hi, np.maximum), neigh(lo, np.minimum)
    p_in = dt * (np.maximum(Ax[..., :-1], 0) - np.minimum(Ax[..., 1:], 0)
                 + np.maximum(Ay[..., :-1, :], 0) - np.minimum(Ay[..., 1:, :], 0))
    p_out = dt * (np.maximum(Ax[..., 1:], 0) - np.minimum(Ax[..., :-1], 0)
                  + np.maximum(Ay[..., 1:, :], 0) - np.minimum(Ay[..., :-1, :], 0))
    q_in = np.maximum(V1[None] * (cmax - cL), 0.0)
    q_out = np.maximum(V1[None] * (cL - cmin), 0.0)
    r_in = np.where(okb & (p_in > 0), np.minimum(1.0, q_in / np.where(p_in > 0, p_in, 1.0)), 0.0)
    r_out = np.where(okb & (p_out > 0), np.minimum(1.0, q_out / np.where(p_out > 0, p_out, 1.0)), 0.0)
    Cx = np.zeros_like(Ax); Cy = np.zeros_like(Ay)
    Cx[..., 1:-1] = np.where(Ax[..., 1:-1] >= 0, np.minimum(r_in[..., 1:], r_out[..., :-1]),
                             np.minimum(r_in[..., :-1], r_out[..., 1:]))
    Cy[..., 1:-1, :] = np.where(Ay[..., 1:-1, :] >= 0, np.minimum(r_in[..., 1:, :], r_out[..., :-1, :]),
                                np.minimum(r_in[..., :-1, :], r_out[..., 1:, :]))
    return Cx, Cy


@njit(cache=True, parallel=True)
def _fct_coeff_kernel(c0, cL, V1, Ax, Ay, ok, dt):  # pragma: no cover
    """fct_coefficients_numpy with the same arithmetic, parallel over (tracer, layer)."""
    nt, n, ny, nx = c0.shape
    r_in = np.zeros((nt, n, ny, nx))
    r_out = np.zeros((nt, n, ny, nx))
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for i in range(nx):
                if not ok[k, j, i]:
                    continue
                cmax, cmin = -np.inf, np.inf
                for kk, jj, ii in ((k, j, i), (k, j, i - 1), (k, j, i + 1), (k, j - 1, i), (k, j + 1, i),
                                   (k - 1, j, i), (k + 1, j, i)):
                    if 0 <= kk < n and 0 <= jj < ny and 0 <= ii < nx and ok[kk, jj, ii]:
                        cmax = max(cmax, c0[t, kk, jj, ii], cL[t, kk, jj, ii])
                        cmin = min(cmin, c0[t, kk, jj, ii], cL[t, kk, jj, ii])
                p_in = dt * (max(Ax[t, k, j, i], 0.0) - min(Ax[t, k, j, i + 1], 0.0)
                             + max(Ay[t, k, j, i], 0.0) - min(Ay[t, k, j + 1, i], 0.0))
                p_out = dt * (max(Ax[t, k, j, i + 1], 0.0) - min(Ax[t, k, j, i], 0.0)
                              + max(Ay[t, k, j + 1, i], 0.0) - min(Ay[t, k, j, i], 0.0))
                q_in = max(V1[k, j, i] * (cmax - cL[t, k, j, i]), 0.0)
                q_out = max(V1[k, j, i] * (cL[t, k, j, i] - cmin), 0.0)
                r_in[t, k, j, i] = min(1.0, q_in / p_in) if p_in > 0.0 else 0.0
                r_out[t, k, j, i] = min(1.0, q_out / p_out) if p_out > 0.0 else 0.0
    Cx = np.zeros(Ax.shape)
    Cy = np.zeros(Ay.shape)
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for f in range(1, nx):
                if Ax[t, k, j, f] >= 0.0:
                    Cx[t, k, j, f] = min(r_in[t, k, j, f], r_out[t, k, j, f - 1])
                else:
                    Cx[t, k, j, f] = min(r_in[t, k, j, f - 1], r_out[t, k, j, f])
        for f in range(1, ny):
            for i in range(nx):
                if Ay[t, k, f, i] >= 0.0:
                    Cy[t, k, f, i] = min(r_in[t, k, f, i], r_out[t, k, f - 1, i])
                else:
                    Cy[t, k, f, i] = min(r_in[t, k, f - 1, i], r_out[t, k, f, i])
    return Cx, Cy


@njit(cache=True, parallel=True)
def _fct_stage(q0, qp, fx, fy, a, b, dt, V, C, q_out, c_out):  # pragma: no cover
    """
    One update of SigmaBaroclinic._advect_fct_numpy with its exact expressions: q = a q0 + b (qp − Δt ∇·F) with the
    face fluxes F = (fx, fy) (a = 0, b = 1: q0 − Δt ∇·F; a = ¾, b = ¼: the second RK3 stage), then the ``safe``
    rule c = q / V where V > 0, else the step's initial C.  Stacks (nt, N, ny, nx), V (N, ny, nx).
    """
    nt, n, ny, nx = q0.shape
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for i in range(nx):
                d = fx[t, k, j, i + 1] - fx[t, k, j, i] + fy[t, k, j + 1, i] - fy[t, k, j, i]
                q = a * q0[t, k, j, i] + b * (qp[t, k, j, i] - dt * d)
                q_out[t, k, j, i] = q
                v = V[k, j, i]
                c_out[t, k, j, i] = q / v if v > 0 else C[t, k, j, i]


@njit(cache=True, parallel=True)
def _fct_fluxes(c, Qx, Qy, ext, open_w, open_e, open_s, open_n, fx0, fy0, fx1, fy1, fx2, fy2):  # pragma: no cover
    """
    Donor-cell flux F_L = Q c_up (open edges: ``ext`` on inflow, the boundary cell on outflow, as _advect_faces_h)
    and antidiffusive flux A = F_H − F_L with the RK3 flux F_H = (F0 + F1)/6 + 2 F2/3 (Zalesak 1979, eq. 6), the
    arithmetic of SigmaBaroclinic._upwind_faces_h and _advect_fct_numpy.  Returns (FLx, FLy, Ax, Ay).
    """
    nt, n, ny, nx = c.shape
    FLx = np.zeros((nt, n, ny, nx + 1))
    FLy = np.zeros((nt, n, ny + 1, nx))
    Ax = np.empty((nt, n, ny, nx + 1))
    Ay = np.empty((nt, n, ny + 1, nx))
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for f in range(1, nx):
                q = Qx[k, j, f]
                FLx[t, k, j, f] = q * (c[t, k, j, f - 1] if q >= 0 else c[t, k, j, f])
            q = Qx[k, j, 0]
            if open_w:
                FLx[t, k, j, 0] = q * (ext[t, k, j, 0] if q > 0 else c[t, k, j, 0])
            q = Qx[k, j, nx]
            if open_e:
                FLx[t, k, j, nx] = q * (ext[t, k, j, nx - 1] if q < 0 else c[t, k, j, nx - 1])
            for f in range(nx + 1):
                fh = (fx0[t, k, j, f] + fx1[t, k, j, f]) / 6.0 + 2.0 * fx2[t, k, j, f] / 3.0
                Ax[t, k, j, f] = fh - FLx[t, k, j, f]
        for i in range(nx):
            for f in range(1, ny):
                q = Qy[k, f, i]
                FLy[t, k, f, i] = q * (c[t, k, f - 1, i] if q >= 0 else c[t, k, f, i])
            q = Qy[k, 0, i]
            if open_s:
                FLy[t, k, 0, i] = q * (ext[t, k, 0, i] if q > 0 else c[t, k, 0, i])
            q = Qy[k, ny, i]
            if open_n:
                FLy[t, k, ny, i] = q * (ext[t, k, ny - 1, i] if q < 0 else c[t, k, ny - 1, i])
            for f in range(ny + 1):
                fh = (fy0[t, k, f, i] + fy1[t, k, f, i]) / 6.0 + 2.0 * fy2[t, k, f, i] / 3.0
                Ay[t, k, f, i] = fh - FLy[t, k, f, i]
    return FLx, FLy, Ax, Ay


@njit(cache=True, parallel=True)
def _fct_final(q0, FLx, FLy, Ax, Ay, Cx, Cy, dt, V1, C):  # pragma: no cover
    """V1 c = V0 c0 − Δt ∇·(F_L + C A) (Zalesak 1979, eq. 14) with the ``safe`` rule, as _advect_fct_numpy."""
    nt, n, ny, nx = q0.shape
    out = np.empty((nt, n, ny, nx))
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for i in range(nx):
                d = (FLx[t, k, j, i + 1] + Cx[t, k, j, i + 1] * Ax[t, k, j, i + 1]
                     - (FLx[t, k, j, i] + Cx[t, k, j, i] * Ax[t, k, j, i])
                     + (FLy[t, k, j + 1, i] + Cy[t, k, j + 1, i] * Ay[t, k, j + 1, i])
                     - (FLy[t, k, j, i] + Cy[t, k, j, i] * Ay[t, k, j, i]))
                q = q0[t, k, j, i] - dt * d
                v = V1[k, j, i]
                out[t, k, j, i] = q / v if v > 0 else C[t, k, j, i]
    return out


@njit(cache=True, parallel=True)
def _diffuse_ale_kernel(T, Tw, h, act, cx, cy, okx, oky, V, kh, dt, out):  # pragma: no cover
    """
    ALE horizontal diffusion (SigmaBaroclinic._diffuse_h_numpy with h given) with the same per-element
    arithmetic: F_x = ((−K_h·L_u/Δx_u)·min(h_L, h_R))·(Tw_i − Tw_{i−1}) on interior faces between active
    cells and with okx, then T − Δt·div/V where V > 0.
    """
    n, ny, nx = T.shape
    for p in prange(n * ny):
        k, j = p // ny, p % ny
        for i in range(nx):
            fw = 0.0
            fe = 0.0
            fs_ = 0.0
            fn = 0.0
            if 0 < i and okx[j, i]:
                hu = min(h[k, j, i - 1], h[k, j, i]) if act[j, i - 1] and act[j, i] else 0.0
                fw = -kh * cx[j, i] * hu * (Tw[k, j, i] - Tw[k, j, i - 1])
            if i < nx - 1 and okx[j, i + 1]:
                hu = min(h[k, j, i], h[k, j, i + 1]) if act[j, i] and act[j, i + 1] else 0.0
                fe = -kh * cx[j, i + 1] * hu * (Tw[k, j, i + 1] - Tw[k, j, i])
            if 0 < j and oky[j, i]:
                hv = min(h[k, j - 1, i], h[k, j, i]) if act[j - 1, i] and act[j, i] else 0.0
                fs_ = -kh * cy[j, i] * hv * (Tw[k, j, i] - Tw[k, j - 1, i])
            if j < ny - 1 and oky[j + 1, i]:
                hv = min(h[k, j, i], h[k, j + 1, i]) if act[j, i] and act[j + 1, i] else 0.0
                fn = -kh * cy[j + 1, i] * hv * (Tw[k, j + 1, i] - Tw[k, j, i])
            div = fe - fw + fn - fs_
            out[k, j, i] = T[k, j, i] - dt * div / V[k, j, i] if V[k, j, i] > 0.0 else T[k, j, i]


def _weno5_faces(c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int, variant: str = "z") -> np.ndarray:
    """WENO5 face values (see ``_weno5_faces_numpy`` for the scheme), compiled with Numba."""
    if variant not in ("z", "js"):
        raise ValueError(f"unknown WENO variant {variant!r}")
    cm = np.moveaxis(np.asarray(c, float), axis, -1)
    fm = np.moveaxis(np.asarray(flux, float), axis, -1)
    om = np.moveaxis(np.asarray(ok, bool), axis, -1)
    lead = np.broadcast_shapes(cm.shape[:-1], fm.shape[:-1], om.shape[:-1])
    n, nf = cm.shape[-1], fm.shape[-1]
    c2 = np.ascontiguousarray(np.broadcast_to(cm, lead + (n,))).reshape(-1, n)
    f2 = np.ascontiguousarray(np.broadcast_to(fm, lead + (nf,))).reshape(-1, nf)
    o2 = np.ascontiguousarray(np.broadcast_to(om, lead + (n,))).reshape(-1, n)
    numba.set_num_threads(_THREADS[0])
    out = _weno5_rows(c2, f2, o2, variant == "z").reshape(lead + (nf,))
    return np.moveaxis(out, -1, axis)


def _weno5_faces_numpy(c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int, variant: str = "z") -> np.ndarray:
    """
    WENO5 face values (Jiang & Shu 1996) on interior faces along ``axis``, upwind
    with respect to ``flux``; WENO3 where only 3 upwind cells are valid, first-order
    upwind next to walls.  For face j (between cells j−1 and j) and flux > 0 the
    stencil is v₁…v₅ = c_{j−3} … c_{j+1}:
        q₀ = (2v₁ − 7v₂ + 11v₃)/6,  q₁ = (−v₂ + 5v₃ + 2v₄)/6,  q₂ = (2v₃ + 5v₄ − v₅)/6
        β₀ = 13/12 (v₁−2v₂+v₃)² + ¼ (v₁−4v₂+3v₃)²,  β₁ = 13/12 (v₂−2v₃+v₄)² + ¼ (v₂−v₄)²
        β₂ = 13/12 (v₃−2v₄+v₅)² + ¼ (3v₃−4v₄+v₅)²,   d = (0.1, 0.6, 0.3)
    variant 'z' (default; Borges et al. 2008, as CROCO step3d_t.F):  α_r = d_r (1 + τ₅/(ε+β_r)),
        τ₅ = |β₂ − β₀|, ε = 1e-40 — less dissipative, no tunable ε
    variant 'js' (Jiang & Shu 1996):  α_r = d_r/(ε+β_r)², ε = 1e-6
    """
    if variant not in ("z", "js"):
        raise ValueError(f"unknown WENO variant {variant!r}")
    c = np.moveaxis(c, axis, -1)
    ok = np.moveaxis(ok, axis, -1)
    f = np.moveaxis(flux, axis, -1)
    n = c.shape[-1]

    def recon(v1, v2, v3, v4, v5, m5, m3):
        q0 = (2 * v1 - 7 * v2 + 11 * v3) / 6
        q1 = (-v2 + 5 * v3 + 2 * v4) / 6
        q2 = (2 * v3 + 5 * v4 - v5) / 6
        b0 = 13 / 12 * (v1 - 2 * v2 + v3) ** 2 + 0.25 * (v1 - 4 * v2 + 3 * v3) ** 2
        b1 = 13 / 12 * (v2 - 2 * v3 + v4) ** 2 + 0.25 * (v2 - v4) ** 2
        b2 = 13 / 12 * (v3 - 2 * v4 + v5) ** 2 + 0.25 * (3 * v3 - 4 * v4 + v5) ** 2
        if variant == "z":
            tau, ez = np.abs(b2 - b0), 1e-40
            a0, a1, a2 = 0.1 * (1 + tau / (ez + b0)), 0.6 * (1 + tau / (ez + b1)), 0.3 * (1 + tau / (ez + b2))
        else:
            a0, a1, a2 = 0.1 / (1e-6 + b0) ** 2, 0.6 / (1e-6 + b1) ** 2, 0.3 / (1e-6 + b2) ** 2
        w5 = (a0 * q0 + a1 * q1 + a2 * q2) / (a0 + a1 + a2)
        # WENO3: candidates (−v₂ + 3v₃)/2 and (v₃ + v₄)/2, d = (1/3, 2/3)
        c0, c1 = (-v2 + 3 * v3) / 2, (v3 + v4) / 2
        e0, e1 = (1 / 3) / (1e-6 + (v3 - v2) ** 2) ** 2, (2 / 3) / (1e-6 + (v4 - v3) ** 2) ** 2
        w3 = (e0 * c0 + e1 * c1) / (e0 + e1)
        return np.where(m5, w5, np.where(m3, w3, v3))

    # cells relative to face j, expressed at the index of cell j (m = j): shifts −3 … +2
    sh = {s: _shift(c, ok, s) for s in (-3, -2, -1, 0, 1, 2)}
    v, k = {s: sh[s][0] for s in sh}, {s: sh[s][1] for s in sh}
    pos = recon(v[-3], v[-2], v[-1], v[0], v[1],
                k[-3] & k[-2] & k[-1] & k[0] & k[1], k[-2] & k[-1] & k[0])
    neg = recon(v[2], v[1], v[0], v[-1], v[-2],
                k[2] & k[1] & k[0] & k[-1] & k[-2], k[1] & k[0] & k[-1])
    face = np.zeros(f.shape)
    face[..., 1:n] = np.where(f[..., 1:n] >= 0, pos[..., 1:n], neg[..., 1:n])
    return np.moveaxis(face, -1, axis)


def _centred_faces(c: np.ndarray, ok: np.ndarray, axis: int, n_out: int | None = None) -> np.ndarray:
    """
    Centred (non-dissipative) face values on interior faces along ``axis``:
    6th order (37(c₋₁+c₀) − 8(c₋₂+c₁) + (c₋₃+c₂))/60 where the stencil is valid,
    4th order (7(c₋₁+c₀) − (c₋₂+c₁))/12 next to it, 2nd order at walls
    (CROCO flux6/flux4/flux2, used in the predictor stage).  ``n_out``: length of the
    returned face axis (default n+1; the vertical uses N, faces 0…N−1).
    """
    c = np.moveaxis(c, axis, -1)
    ok = np.moveaxis(ok, axis, -1)
    n = c.shape[-1]
    sh = {s: _shift(c, ok, s) for s in (-3, -2, -1, 0, 1, 2)}
    v, k = {s: sh[s][0] for s in sh}, {s: sh[s][1] for s in sh}
    f6 = (37 * (v[-1] + v[0]) - 8 * (v[-2] + v[1]) + (v[-3] + v[2])) / 60
    f4 = (7 * (v[-1] + v[0]) - (v[-2] + v[1])) / 12
    f2 = 0.5 * (v[-1] + v[0])
    m6 = k[-3] & k[-2] & k[-1] & k[0] & k[1] & k[2]
    m4 = k[-2] & k[-1] & k[0] & k[1]
    face = np.zeros(c.shape[:-1] + (n + 1 if n_out is None else n_out,))
    face[..., 1:n] = np.where(m6, f6, np.where(m4, f4, f2))[..., 1:n]
    return np.moveaxis(face, -1, axis)


@njit(cache=True, parallel=True)
def _flux_form_u_kernel(u, Qx, Qy, Fz, V, ok, dt):  # pragma: no cover
    """Compiled ``_flux_form_u_numpy`` (same transports, WENO5-Z values, face rules and limiter)."""
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    Fx = np.zeros((n, ny, nx))
    Fy = np.zeros((n, ny + 1, nx1))
    Fzz = np.zeros((n + 1, ny, nx1))
    out = np.zeros_like(u)
    for k in prange(n):
        for j in range(ny):
            # x: through cell centres c (between u-faces c and c+1), transport Uc = mean of the face transports
            for c in range(nx):
                q = 0.5 * (Qx[k, j, c] + Qx[k, j, c + 1])
                f = c + 1
                if q >= 0.0:
                    i1, i2, i3, i4, i5 = f - 3, f - 2, f - 1, f, f + 1
                else:
                    i1, i2, i3, i4, i5 = f + 2, f + 1, f, f - 1, f - 2
                b1, b2, b3, b4, b5 = 0 <= i1 < nx1, 0 <= i2 < nx1, 0 <= i3 < nx1, 0 <= i4 < nx1, 0 <= i5 < nx1
                v = _wval(u[k, j, i1] if b1 else 0.0, u[k, j, i2] if b2 else 0.0, u[k, j, i3] if b3 else 0.0,
                          u[k, j, i4] if b4 else 0.0, u[k, j, i5] if b5 else 0.0,
                          b1 and ok[k, j, i1], b2 and ok[k, j, i2], b3 and ok[k, j, i3],
                          b4 and ok[k, j, i4], b5 and ok[k, j, i5], True)
                Fx[k, j, c] = q * v
        # y: through the corners q = 0 … ny at u-face i (rows outside the domain: 0 and invalid)
        for qq in range(ny + 1):
            for i in range(1, nx):
                q = 0.5 * (Qy[k, qq, i - 1] + Qy[k, qq, i])
                if q >= 0.0:
                    r1, r2, r3, r4, r5 = qq - 3, qq - 2, qq - 1, qq, qq + 1
                else:
                    r1, r2, r3, r4, r5 = qq + 2, qq + 1, qq, qq - 1, qq - 2
                b1, b2, b3, b4, b5 = 0 <= r1 < ny, 0 <= r2 < ny, 0 <= r3 < ny, 0 <= r4 < ny, 0 <= r5 < ny
                v = _wval(u[k, r1, i] if b1 else 0.0, u[k, r2, i] if b2 else 0.0, u[k, r3, i] if b3 else 0.0,
                          u[k, r4, i] if b4 else 0.0, u[k, r5, i] if b5 else 0.0,
                          b1 and ok[k, r1, i], b2 and ok[k, r2, i], b3 and ok[k, r3, i],
                          b4 and ok[k, r4, i], b5 and ok[k, r5, i], True)
                Fy[k, qq, i] = q * v
    for j in prange(ny):
        # z: through the interfaces kk = 1 … N−1 at the u-face, transport = mean of the two cells' Fz
        for kk in range(1, n):
            for i in range(1, nx):
                q = 0.5 * (Fz[kk, j, i - 1] + Fz[kk, j, i])
                if q >= 0.0:
                    k1, k2, k3, k4, k5 = kk - 3, kk - 2, kk - 1, kk, kk + 1
                else:
                    k1, k2, k3, k4, k5 = kk + 2, kk + 1, kk, kk - 1, kk - 2
                b1, b2, b3, b4, b5 = 0 <= k1 < n, 0 <= k2 < n, 0 <= k3 < n, 0 <= k4 < n, 0 <= k5 < n
                v = _wval(u[k1, j, i] if b1 else 0.0, u[k2, j, i] if b2 else 0.0, u[k3, j, i] if b3 else 0.0,
                          u[k4, j, i] if b4 else 0.0, u[k5, j, i] if b5 else 0.0,
                          b1 and ok[k1, j, i], b2 and ok[k2, j, i], b3 and ok[k3, j, i],
                          b4 and ok[k4, j, i], b5 and ok[k5, j, i], True)
                Fzz[kk, j, i] = q * v
    for k in prange(n):
        for j in range(ny):
            for i in range(1, nx):
                if not ok[k, j, i]:
                    continue
                vu = 0.5 * (V[k, j, i - 1] + V[k, j, i])
                if vu <= 0.0:
                    continue
                div = Fx[k, j, i] - Fx[k, j, i - 1] + Fy[k, j + 1, i] - Fy[k, j, i] + Fzz[k + 1, j, i] - Fzz[k, j, i]
                uc_hi = 0.5 * (Qx[k, j, i] + Qx[k, j, i + 1])
                uc_lo = 0.5 * (Qx[k, j, i - 1] + Qx[k, j, i])
                vq_hi = 0.5 * (Qy[k, j + 1, i - 1] + Qy[k, j + 1, i])
                vq_lo = 0.5 * (Qy[k, j, i - 1] + Qy[k, j, i])
                w_hi = 0.5 * (Fz[k + 1, j, i - 1] + Fz[k + 1, j, i])
                w_lo = 0.5 * (Fz[k, j, i - 1] + Fz[k, j, i])
                mass = uc_hi - uc_lo + vq_hi - vq_lo + w_hi - w_lo
                if dt > 0.0:                                         # vanished-cell limiter (ALE)
                    q_in = max(uc_lo, 0.0) + max(-uc_hi, 0.0) + max(vq_lo, 0.0) + max(-vq_hi, 0.0) \
                        + max(w_lo, 0.0) + max(-w_hi, 0.0)
                    q_out = max(-uc_lo, 0.0) + max(uc_hi, 0.0) + max(-vq_lo, 0.0) + max(vq_hi, 0.0) \
                        + max(-w_lo, 0.0) + max(w_hi, 0.0)
                    vu = max(vu, 2.0 * dt * max(q_in, q_out))
                out[k, j, i] = -(div - u[k, j, i] * mass) / vu
    return out


def _flux_form_u(u: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, V: np.ndarray,
                 ok_u: np.ndarray, dt: float = 0.0) -> np.ndarray:
    """Flux-form momentum advection (see ``_flux_form_u_numpy``), compiled."""
    c = lambda a, t=float: np.ascontiguousarray(a, dtype=t)                       # noqa: E731
    numba.set_num_threads(_THREADS[0])
    return _flux_form_u_kernel(c(u), c(Qx), c(Qy), c(Fz), c(V), c(np.broadcast_to(ok_u, u.shape), bool), float(dt))


def _flux_form_u_numpy(u: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, V: np.ndarray,
                       ok_u: np.ndarray, dt: float = 0.0) -> np.ndarray:
    """
    Momentum-conserving advection of u (N,ny,nx+1) by the layer transports
    (flux form, as ROMS/CROCO; Shchepetkin & McWilliams 2005):

        −(1/V_u) [ δ_x(U ũ) + δ_y(V ũ) + δ_k(W ũ) − u (δ_x U + δ_y V + δ_k W) ]

    U = transport through the cell centres (mean of the two u-face transports),
    V through the corners, W through the layer interfaces at the u-face; ũ are
    WENO5 values upwind of each transport.  Subtracting u × (mass divergence)
    gives the velocity tendency while the momentum flux stays conservative, which
    sets the speed of fronts and jumps (Borden & Meiburg 2013; Ungarish et al. 2018).
    With dt > 0 (ALE: layers that vanish) V_u is replaced by max(V_u, 2 Δt max(Q_in, Q_out)),
    Q_in and Q_out the transports into and out of the u-cell: a vanished cell flooded by (or
    draining) a current relaxes to the current's velocity at a rate ~1/(2Δt) at most (exactly ½
    max|ũ − u| per step when the flow only enters or only leaves), instead of Q Δt/V_u ≫ 1, inside
    the AB3 stability limit (λΔt < 6/11).  Counting only the inflow left the outflow term
    Q_out (ũ − u)/V_u unbounded (hybrid lock exchange, 3.4 m s⁻² in a 1 mm layer); counting
    Σ|Q| = Q_in + Q_out engaged it in thick cells from u Δt/Δx > ¼.  Cells with u Δt/Δx ≤ ½
    are unchanged.  Same purpose as the
    CFL truncation of MOM6 (MOM_vert_friction.F90:2555, vertvisc_limit_vel), but continuous.
    Returns the tendency on interior u-faces [m s⁻²].
    """
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    # x: through cell centres c = 0 … nx−1 (between u-faces c and c+1)
    Uc = 0.5 * (Qx[..., :-1] + Qx[..., 1:])
    fl = np.zeros((n, ny, nx + 2))
    fl[..., 1:nx + 1] = Uc
    okx = np.broadcast_to(ok_u, u.shape)
    Fx = Uc * _weno5_faces(u, fl, okx, axis=2)[..., 1:nx + 1]
    # y: through corners (j = 0 … ny, at u-face i), transport = mean of the v-transports of cells i−1, i
    Vq = np.zeros((n, ny + 1, nx1))
    Vq[..., 1:-1] = 0.5 * (Qy[..., :-1] + Qy[..., 1:])
    uy = np.concatenate([np.zeros((n, 1, nx1)), u, np.zeros((n, 1, nx1))], axis=1)
    oky = np.concatenate([np.zeros((n, 1, nx1), bool), okx, np.zeros((n, 1, nx1), bool)], axis=1)
    fly = np.concatenate([np.zeros((n, 1, nx1)), Vq, np.zeros((n, 1, nx1))], axis=1)
    Fy = Vq * _weno5_faces(uy, fly, oky, axis=1)[:, 1:ny + 2]
    # z: through interfaces k = 0 … N at the u-face
    Wz = np.zeros((n + 1, ny, nx1))
    Wz[..., 1:-1] = 0.5 * (Fz[..., :-1] + Fz[..., 1:])
    flz = np.zeros((n, ny, nx1))
    flz[1:] = Wz[1:-1]
    Fzz = np.zeros((n + 1, ny, nx1))
    Fzz[1:-1] = Wz[1:-1] * _weno5_faces(u, flz, okx, axis=0)[1:]
    div = np.zeros_like(u)
    div[..., 1:-1] = (Fx[..., 1:] - Fx[..., :-1])
    div += Fy[:, 1:] - Fy[:, :-1] + Fzz[1:] - Fzz[:-1]
    mass = np.zeros_like(u)
    mass[..., 1:-1] = Uc[..., 1:] - Uc[..., :-1]
    mass += Vq[:, 1:] - Vq[:, :-1] + Wz[1:] - Wz[:-1]
    Vu = np.zeros_like(u)
    Vu[..., 1:-1] = 0.5 * (V[..., :-1] + V[..., 1:])
    if dt > 0:
        q_in, q_out = np.zeros_like(u), np.zeros_like(u)
        q_in[..., 1:-1] = np.maximum(Uc[..., :-1], 0) + np.maximum(-Uc[..., 1:], 0)
        q_out[..., 1:-1] = np.maximum(-Uc[..., :-1], 0) + np.maximum(Uc[..., 1:], 0)
        q_in += np.maximum(Vq[:, :-1], 0) + np.maximum(-Vq[:, 1:], 0) + np.maximum(Wz[:-1], 0) + np.maximum(-Wz[1:], 0)
        q_out += np.maximum(-Vq[:, :-1], 0) + np.maximum(Vq[:, 1:], 0) + np.maximum(-Wz[:-1], 0) + np.maximum(Wz[1:], 0)
        Vu = np.where(Vu > 0, np.maximum(Vu, 2.0 * dt * np.maximum(q_in, q_out)), Vu)
    out = np.where(Vu > 0, -(div - u * mass) / np.where(Vu > 0, Vu, 1.0), 0.0)
    out[..., [0, -1]] = 0.0
    return np.where(okx, out, 0.0)


def reference_profile(z: np.ndarray, rho: np.ndarray, n_bins: int = 400):
    """
    Horizontally averaged density as a function of depth, ρ_ref(z), from cell
    values (z, ρ): ρ is averaged in ``n_bins`` depth bins and interpolated
    linearly (constant beyond the data).  Used as ``rho_ref`` to remove the
    resting stratification from the pressure-gradient computation (Haney 1991;
    Berntsen et al. 2015).  Returns a callable z → ρ_ref.
    """
    zf, rf = np.asarray(z, float).ravel(), np.asarray(rho, float).ravel()
    ok = np.isfinite(zf) & np.isfinite(rf)
    zf, rf = zf[ok], rf[ok]
    edges = np.linspace(zf.min(), zf.max(), n_bins + 1)
    idx = np.clip(np.digitize(zf, edges) - 1, 0, n_bins - 1)
    cnt = np.bincount(idx, minlength=n_bins)
    mean = np.bincount(idx, weights=rf, minlength=n_bins) / np.maximum(cnt, 1)
    zc = np.bincount(idx, weights=zf, minlength=n_bins) / np.maximum(cnt, 1)
    keep = cnt > 0
    return ReferenceProfile(zc[keep], mean[keep])


@numba.njit(parallel=True, cache=True, fastmath={"contract"})
def _interp_kernel(x, xp, fp, out):  # pragma: no cover
    """np.interp element by element with NumPy's arithmetic (numpy/core/src/multiarray/compiled_base.c,
    arr_interp): constant beyond the ends, the node value on a node, else slope·(x − xp_j) + fp_j.  Only FMA
    contraction is allowed (no other fastmath flag): NumPy's arm64 build fuses slope·(x − xp_j) + fp_j, and
    without it 1 value in ~3e4 differed by one ulp."""
    n = xp.size
    for p in numba.prange(x.size):
        v = x[p]
        if v != v:                                            # NaN height: NaN, as NumPy (before any search)
            out[p] = v
        elif v > xp[n - 1]:
            out[p] = fp[n - 1]
        elif v < xp[0]:
            out[p] = fp[0]
        else:
            lo, hi = 0, n                                     # largest j with xp[j] <= v
            while hi - lo > 1:
                mid = (lo + hi) // 2
                if xp[mid] <= v:
                    lo = mid
                else:
                    hi = mid
            j = lo
            if j == n - 1 or xp[j] == v:
                out[p] = fp[j]
            else:
                slope = (fp[j + 1] - fp[j]) / (xp[j + 1] - xp[j])
                r = slope * (v - xp[j]) + fp[j]
                if np.isnan(r):
                    r = slope * (v - xp[j + 1]) + fp[j + 1]
                    if np.isnan(r) and fp[j] == fp[j + 1]:
                        r = fp[j]
                out[p] = r


@numba.njit(parallel=True, cache=True)
def _rk_stage(qa, qb, r, dt, V, C, stage, q_out, c_out):  # pragma: no cover
    """
    One SSP-RK3 stage of SigmaBaroclinic._advect_numpy with its exact expressions: stage 1 q = q0 + dt r,
    2: 0.75 q0 + 0.25 (q1 + dt r), 3: q0/3 + 2/3 (q2 + dt r) (qa = q0, qb the previous stage); then the
    ``safe`` rule c = q / V where V > 0, else the step's initial C.  Arrays (ntr, N, ny, nx) C-contiguous;
    V (N, ny, nx) broadcast over the tracers.  1-D arrays, flattened by the caller (ravel inside a parallel
    kernel may copy, losing the writes).
    """
    nv = V.size
    for p in numba.prange(q_out.size):
        if stage == 1:
            q = qa[p] + dt * r[p]
        elif stage == 2:
            q = 0.75 * qa[p] + 0.25 * (qb[p] + dt * r[p])
        else:
            q = qa[p] / 3.0 + 2.0 / 3.0 * (qb[p] + dt * r[p])
        q_out[p] = q
        v = V[p % nv]
        c_out[p] = q / v if v > 0 else C[p]


@numba.njit(parallel=True, cache=True)
def _ab3_kernel(g0, g1, g2, three, out):  # pragma: no cover
    """AB3[0] g0 + AB3[1] g1 + AB3[2] g2 (three) or 1.5 g0 − 0.5 g1 (AB2 start-up), NumPy's evaluation order.
    1-D arrays (flattened by the caller: ravel inside a parallel kernel may copy, losing the writes)."""
    a0, a1, a2 = 23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0
    for p in numba.prange(out.size):
        if three:
            out[p] = a0 * g0[p] + a1 * g1[p] + a2 * g2[p]
        else:
            out[p] = 1.5 * g0[p] - 0.5 * g1[p]


def ab3_combine(g0: np.ndarray, g1: np.ndarray, g2: np.ndarray | None) -> np.ndarray:
    """Adams–Bashforth extrapolation of the explicit tendencies (newest g0): AB3 with g2, AB2 without
    (step's start-up).  Fused kernel with the same arithmetic as the NumPy expressions."""
    if np.shape(g1) != np.shape(g0) or (g2 is not None and np.shape(g2) != np.shape(g0)):
        raise ValueError(f"ab3_combine: tendency shapes differ {np.shape(g0)}, {np.shape(g1)}, "
                         f"{None if g2 is None else np.shape(g2)}")       # the kernel does not check bounds
    out = np.empty(np.shape(g0))
    g0, g1 = np.ascontiguousarray(g0, float), np.ascontiguousarray(g1, float)
    numba.set_num_threads(_THREADS[0])
    g2 = g0 if g2 is None else np.ascontiguousarray(g2, float)
    _ab3_kernel(g0.ravel(), g1.ravel(), g2.ravel(), g2 is not g0, out.ravel())
    return out


class ReferenceProfile:
    """ρ_ref(z) tabulated at heights z [m] (increasing) with values rho [kg m⁻³]; linear in between."""

    _interp_kernel = staticmethod(_interp_kernel)

    def __init__(self, z: np.ndarray, rho: np.ndarray) -> None:
        self.z, self.rho = np.asarray(z, float), np.asarray(rho, float)

    def __call__(self, zz: np.ndarray) -> np.ndarray:
        """np.interp(zz, z, rho); a parallel kernel with the same arithmetic for ≥ 1000 heights."""
        zz = np.asarray(zz, float)
        if zz.size < 1000:
            return np.interp(zz, self.z, self.rho)
        out = np.empty(zz.shape)
        numba.set_num_threads(_THREADS[0])
        _interp_kernel(np.ascontiguousarray(zz).ravel(), self.z, self.rho, out.ravel())
        return out


EDGES = ("west", "east", "south", "north")


def _sponge_rates(shape: tuple[int, int], edges: tuple[str, ...], n_cells: int, tau_inner: float,
                  tau_bound: float):
    """
    Relaxation rates [s⁻¹] of the open-boundary sponge at cells, u- and v-faces, with the MITgcm
    OBCS profile (obcs_sponge.F, useLinearSponge = .FALSE.): at distance d = 1 … N cells from the
    boundary the time scale is τ(d) = [(N − d) τ_bound + d τ_inner]/N and the relaxation acts on
    (N − d)/N of the departure, i.e. rate(d) = (N − d)/(N τ(d)) — strong at the edge, zero at d = N.
    Returns (None, None, None) without a sponge.
    """
    if not n_cells:
        return None, None, None
    ny, nx = shape
    j, i = np.mgrid[0:ny, 0:nx]
    dist = {"west": i + 1, "east": nx - i, "south": j + 1, "north": ny - j}
    d = np.min([dist[e] for e in edges], axis=0).astype(float)
    N = float(n_cells)
    tau = ((N - d) * tau_bound + d * tau_inner) / N
    rc = np.where(d < N, (N - d) / (N * tau), 0.0)
    ru = np.empty((ny, nx + 1))
    ru[:, 1:-1] = 0.5 * (rc[:, :-1] + rc[:, 1:])
    ru[:, 0], ru[:, -1] = rc[:, 0], rc[:, -1]
    rv = np.empty((ny + 1, nx))
    rv[1:-1] = 0.5 * (rc[:-1] + rc[1:])
    rv[0], rv[-1] = rc[0], rc[-1]
    return rc, ru, rv


class SigmaBaroclinic:
    """
    Stratified σ-layer model (see module docstring).

    Parameters
    ----------
    h_grid, bathy, dt, n_layers, z0, closure, nu_const, theta, d_min, g : as ``SigmaWetDry``
    rho0 : reference density [kg m⁻³];  t_coef : α [kg m⁻³ °C⁻¹];  t_ref : T₀ [°C]
    ah : horizontal Laplacian viscosity [m² s⁻¹] (free slip)
    adv : momentum advection: 'flux' (default; momentum-conserving flux form with WENO5,
          as CROCO) or the advective form 'upwind3' / 'centred' of core.dynamics.nonlinear
    vertical_coordinate : 'sigma_eulerian' (default; σ-layers with vertical advection) or 'ale'
          (Lagrangian step + regrid + remap each step; spec docs/superpowers/specs/2026-10-11-ale-...)
    regrid : ALE grid rule 'sigma' | 'zstar' | 'hybrid' (core.vertical.regrid)
    regrid_params : rule parameters (dz_top, rho_target, max_h_top, d_sigma, d_hybrid, h_min; dsigma
          defaults to the model's); hybrid: ``keep_isopycnal`` (default True) keeps interfaces between
          isopycnally consistent layers (spec §3, B2c)
    remap_scheme : 'pcm' | 'plm' | 'ppm_h4' (default)
    h_thin_bed : bed face layers thinner than this [m] carry no velocity (HYCOM)
    init_iterations : regrid–remap passes of the initial state with the hybrid rule
    momentum : 'vector_invariant' (default in ALE: PV flux + KE gradient, MOM6) or 'flux' (default in
          σ: flux-form WENO5 + energy-conserving Coriolis); 'vector_invariant' needs vertical_coordinate='ale'
    ke_scheme : kinetic energy of the vector-invariant form: 'arakawa' (default, energy conserving, MOM6
          production; Esteban 2026-10-12: 'gudonov' damps progressive internal waves ~9 % per 5 km at dx = 100 m,
          validation/internal_wave/README.md) or 'gudonov' (upwind: only the velocities entering a cell; MOM6
          lock_exchange — gravity-current fronts and bores lose energy, ALE-σ lock exchange 0.98 U vs 0.91/0.84 U
          with 'arakawa'; use it for density currents)
    max_truncations : ALE: stop after this many CFL velocity truncations at faces thicker than h_thin_bed
    thickness_flux : ALE layer thickness at the faces: 'ppm' (default) = MOM6 continuity_PPM with the
          barotropic adjustment (B2b; core.vertical.ale.ppm_transport_x; hybrid lock exchange with 21
          layers = MOM6, needs ~20 layers in density currents) or 'upwind' = donor cell (B2a, 28 % cheaper)
    pgf : 'jacobian' (default), 'twoterm', or 'fv', finite volume (Adcroft et al. 2008; core.vertical.pgf)
    mix_diagnostic : ALE: advect and remap T² for the spurious-mixing diagnostic ``mix_num`` (default True; False
          halves the tracer work and leaves T unchanged; mix_num is then None)
    solver : free-surface solver: 'direct' (σ default) or 'cg' (ALE default: CG preconditioned with the LU of a
          lagged matrix, core.dynamics.wetting_drying; ≈1 s → ≈0.1 s per step on BayIW 560 × 352)
    open_boundaries, sponge_cells, sponge_tau, orlanski_cmax, sponge_tracer : also with vertical_coordinate='ale'
          (Plan C scope A: exterior at rest, T_ext/h_ext kept in z from the initial state; no H ≤ 0 cells at an edge)
    open_boundary_state : t [s] → {edge: boundaries.obcs.OBData with zw}: time-varying exterior (spec
          2026-10-12-baroclinic-obc-internal-wave), evaluated at tⁿ⁺¹ and remapped from its own z grid onto the
          layers (core.dynamics.open_boundary); None = the resting exterior above, unchanged
    open_boundary_mode : {edge: 'radiating' (default: Flather toward (η, ū)ₑₓₜ, Orlanski on u′ then nudging toward
          u′ₑₓₜ with Δt/τ_in when flowing in, Δt/τ_out when out, as CROCO u3dbc.F:111-155; sponge toward the
          exterior) | 'clamped' (u′ = u′ₑₓₜ, Dirichlet, as MITgcm OBCS prescribe / CROCO M3_FRC_BRY)}
    obc_tau_in, obc_tau_out : nudging time scales [s] of 'radiating' (CROCO typical 1 and 10 days)
    theta_bc : time weight of the baroclinic part u′ of the layer transport (None = ½ in every coordinate:
        time-centred, as the AB3 momentum tendency is centred at n + ½); the depth mean keeps the free-surface
        ``theta`` so the layer fluxes add up to the free-surface flux.  Before 2026-10-13 u′ took ``theta`` too,
        which damps internal waves at ~(θ − ½) ω² Δt (Ilicak internal wave: 0.45 of the oscillation left after
        200 days at ν = 0.01 against 0.65–0.72 in MOM6, CROCO and MITgcm; 0.74 with theta_bc = ½;
        validation/ilicak_benchmark/README.md).  The hybrid kept ``theta`` until 2026-10-14: with ½ its
        near-vanished layers blew up (Ilicak IW ν = 0.01 at day ~113); the outflow limiter, the thin-layer
        coupling and blending below remove that (spec 2026-10-14-hybrid-robustness)
    thin_layer_coupling : ALE: thickness of the layers at the faces for the implicit vertical viscosity —
        'none' (the mean (q_E + q_W)/2) or 'mom6' (MOM6's upwind-biased thickness through the whole column:
        harmonic where the water leaves the thinner side, arithmetic where thick water advances over thin;
        core.vertical.ale.viscous_face_thickness): a layer draining from where it has vanished is tied to its
        neighbours instead of accelerating alone (spec 2026-10-14-hybrid-robustness §3); the bed drag keeps the
        mean thickness.  None = 'mom6' in ALE, 'none' otherwise
    thin_layer_cutoff : ALE: layers thinner than this at a face [m] move partly (vanished: wholly) with the layer
        above, HYCOM's blending (mod_momtum.F90:3055-3065, cutoff 0.5 m; core.vertical.ale.blend_thin_layers)
        on the upwind (donor) thickness instead of HYCOM's mean, so an advancing nose is not held back;
        0 = off.  None = 0: blending the vanished layer ahead of a gravity-current nose gives it the counter-flowing
        velocity of the layer above, which then keeps the nose from entering (hybrid lock exchange front 1.05 →
        0.70 U for any cutoff ≥ 0.01 m, 2026-10-14); in z*/σ thin layers mean shallow water, where blending would
        erase the vertical shear
    tracer_limiter : 'fct': Zalesak (1979) flux-corrected transport around the SSP-RK3 WENO5 advection of the ALE
        step — tracers stay inside the local extremes of the old and the donor-cell values (spec 2026-10-14 §4;
        Ilicak overflow z* ν = 1000: new T extrema 4.74 °C → 1e-8); 'none'.  None = 'fct' in ALE with SSP-RK3
        WENO5, 'none' otherwise
    """

    def __init__(self, h_grid: xr.Dataset, bathy: np.ndarray, dt: float, n_layers: int = 10,
                 rho0: float = 1000.0, t_coef: float = 0.2, t_ref: float = 0.0, z0: float | None = None,
                 closure: str = "constant", nu_const: float = 1e-4, ah: float = 0.0, adv: str = "flux",
                 theta: float = 0.6, d_min: float = 0.05, g: float = G, tracer_scheme: str = "weno5",
                 tracer_time: str = "ssprk3", pgf: str = "jacobian", rho_ref=None,
                 open_boundaries: tuple[str, ...] = (), sponge_cells: int = 0,
                 sponge_tau: tuple[float, float] = (1200.0, 60.0), orlanski_cmax: float = 1.0,
                 sponge_tracer: bool = True, dsigma: np.ndarray | None = None, kh: float = 0.0,
                 kv: float = 0.0, solver: str | None = None, vertical_coordinate: str = "sigma_eulerian",
                 regrid: str = "sigma", regrid_params: dict | None = None, remap_scheme: str = "ppm_h4",
                 h_thin_bed: float = 0.01, init_iterations: int = 10, momentum: str | None = None,
                 ke_scheme: str = "arakawa", max_truncations: int = 1000, thickness_flux: str = "ppm",
                 mix_diagnostic: bool = True, open_boundary_state=None, open_boundary_mode: dict | None = None,
                 obc_tau_in: float = 86400.0, obc_tau_out: float = 864000.0, theta_bc: float | None = None,
                 thin_layer_coupling: str | None = None, thin_layer_cutoff: float | None = None,
                 tracer_limiter: str | None = None) -> None:
        if pgf not in ("twoterm", "jacobian", "fv"):
            raise ValueError(f"unknown pressure-gradient scheme {pgf!r}")
        self.pgf, self.rho_ref = pgf, rho_ref
        if tracer_time not in ("ssprk3", "pc"):
            raise ValueError(f"unknown tracer time stepping {tracer_time!r}")
        self.tracer_time = tracer_time
        if tracer_scheme not in ("weno5", "weno5js", "upwind3", "upwind1"):
            raise ValueError(f"unknown tracer scheme {tracer_scheme!r}")
        self.tracer_scheme = tracer_scheme
        bad = set(open_boundaries) - set(EDGES)
        if bad:
            raise ValueError(f"unknown open boundaries {sorted(bad)}; use {EDGES}")
        if sponge_cells and not open_boundaries:
            raise ValueError("a sponge needs open boundaries")
        self.open = tuple(open_boundaries)
        if open_boundary_state is not None and not self.open:
            raise ValueError("open_boundary_state needs open_boundaries")
        self.obc_state, self.obc_mode = open_boundary_state, {e: "radiating" for e in self.open}
        for e, mode in (open_boundary_mode or {}).items():
            if e not in self.open or mode not in ("radiating", "clamped"):
                raise ValueError(f"open_boundary_mode {{{e!r}: {mode!r}}}: edges must be open and modes "
                                 "'radiating' or 'clamped'")
            self.obc_mode[e] = mode
        if open_boundary_state is None and "clamped" in self.obc_mode.values():
            raise ValueError("a 'clamped' open_boundary_mode needs open_boundary_state (the exterior to impose)")
        self.obc_tau_in, self.obc_tau_out = float(obc_tau_in), float(obc_tau_out)
        if theta_bc is None:    # automatic: time-centred u′ in every coordinate (spec 2026-10-14 §5)
            theta_bc = 0.5
        if thin_layer_coupling is None:
            thin_layer_coupling = "mom6" if vertical_coordinate == "ale" else "none"
        if thin_layer_cutoff is None:    # HYCOM blending off: it locks advancing fronts (docstring)
            thin_layer_cutoff = 0.0
        self.theta_bc = float(theta_bc)
        if thin_layer_coupling not in ("none", "mom6"):
            raise ValueError(f"unknown thin_layer_coupling {thin_layer_coupling!r}; use 'none' or 'mom6'")
        self.thin_layer_coupling = thin_layer_coupling
        if not thin_layer_cutoff >= 0.0:
            raise ValueError(f"thin_layer_cutoff must be ≥ 0 m, got {thin_layer_cutoff!r}")
        self.thin_layer_cutoff = float(thin_layer_cutoff)
        self._ext = None
        self.orlanski_cmax, self.sponge_tracer = orlanski_cmax, sponge_tracer
        self.sponge_cells, self.sponge_tau = int(sponge_cells), tuple(float(t) for t in sponge_tau)
        self.kh, self.kv = float(kh), float(kv)
        self.mix_diagnostic = bool(mix_diagnostic)                      # ALE: advect/remap T² for mix_num
        act = np.isfinite(np.asarray(bathy, float))
        fl_u = np.zeros((act.shape[0], act.shape[1] + 1), bool)
        fl_v = np.zeros((act.shape[0] + 1, act.shape[1]), bool)
        if "west" in self.open:
            fl_u[:, 0] = act[:, 0]
        if "east" in self.open:
            fl_u[:, -1] = act[:, -1]
        if "south" in self.open:
            fl_v[0, :] = act[0, :]
        if "north" in self.open:
            fl_v[-1, :] = act[-1, :]
        if solver is None:                     # ALE: CG with the lagged-LU preconditioner (1–3 iterations); the
            solver = "cg" if vertical_coordinate == "ale" else "direct"   # σ mode keeps the direct solve, bit-identical
        self.core = SigmaWetDry(h_grid, bathy, dt=dt, n_layers=n_layers, z0=z0, closure=closure,
                                nu_const=nu_const, theta=theta, g=g, d_min=d_min, dsigma=dsigma, solver=solver,
                                flather_u=fl_u if self.open else None, flather_v=fl_v if self.open else None)
        sponged = tuple(e for e in self.open if self.obc_mode[e] != "clamped")  # a Dirichlet edge needs no sponge
        self._sp_c, self._sp_u, self._sp_v = _sponge_rates(act.shape, sponged, sponge_cells if sponged else 0,
                                                           *sponge_tau)
        self.T_ext = None
        bc = np.zeros(act.shape, bool)                                   # columns that need the exterior profile
        for edge, sl in (("west", np.s_[:, 0]), ("east", np.s_[:, -1]), ("south", np.s_[0, :]), ("north", np.s_[-1, :])):
            if edge in self.open:
                bc[sl] = True
        if self._sp_c is not None:
            bc |= self._sp_c > 0
        self._bc_mask = bc & act
        edge_cells = np.zeros(act.shape, bool)                           # boundary cells: the inflow T needs only
        for edge, sl in (("west", np.s_[:, 0]), ("east", np.s_[:, -1]), ("south", np.s_[0, :]), ("north", np.s_[-1, :])):
            if edge in self.open:                                        # these; the band only for the tracer sponge
                edge_cells[sl] = True
        self._ext_cols = self._bc_mask.copy() if self.sponge_tracer else edge_cells & act
        self._bc_u, self._bc_v = fl_u.copy(), fl_v.copy()               # open faces + sponge band (exterior u′)
        if self._sp_u is not None:
            self._bc_u |= self._sp_u > 0
            self._bc_v |= self._sp_v > 0
        self.h_ext, self._T_in = None, None
        self.hg, self.dt, self.n, self.g = h_grid, dt, n_layers, g
        self.rho0, self.t_coef, self.t_ref, self.ah, self.adv = rho0, t_coef, t_ref, ah, adv
        self.fs = self.core.fs
        self._fq = np.asarray(h_grid["f_q"].values, float)           # Coriolis at cell corners [s⁻¹]
        self.H = self.fs.H
        self.ds = self.core.dsigma
        self.sigma_c = 0.5 * (self.core.sigma_w[1:, 0] + self.core.sigma_w[:-1, 0])   # layer centres (bed = 0)
        self.area = self.fs.area
        self._setup_ale(vertical_coordinate, regrid, regrid_params, remap_scheme, z0, closure, h_thin_bed,
                        init_iterations, act, momentum, ke_scheme, max_truncations, thickness_flux)
        if tracer_limiter is None:
            tracer_limiter = ("fct" if self.ale and self.tracer_time == "ssprk3"
                              and self.tracer_scheme in ("weno5", "weno5js") else "none")
        if tracer_limiter not in ("none", "fct"):
            raise ValueError(f"unknown tracer_limiter {tracer_limiter!r}; use 'none' or 'fct'")
        if tracer_limiter == "fct" and not (self.ale and self.tracer_time == "ssprk3"
                                            and self.tracer_scheme in ("weno5", "weno5js")):
            raise ValueError("tracer_limiter='fct' needs vertical_coordinate='ale', tracer_time='ssprk3' and a "
                             "WENO5 tracer_scheme")
        self.tracer_limiter = tracer_limiter

    def _setup_ale(self, vertical_coordinate, regrid, regrid_params, remap_scheme, z0, closure, h_thin_bed,
                   init_iterations, act, momentum=None, ke_scheme="arakawa", max_truncations=1000,
                   thickness_flux="ppm") -> None:
        """Check and store the ALE configuration (spec §2, §6.2); scope limits of ALE Plan B raise."""
        if vertical_coordinate not in ("sigma_eulerian", "ale"):
            raise ValueError(f"unknown vertical_coordinate {vertical_coordinate!r}; use 'sigma_eulerian' or 'ale'")
        self.ale = vertical_coordinate == "ale"
        momentum = momentum or ("vector_invariant" if self.ale else "flux")
        if momentum not in ("flux", "vector_invariant"):
            raise ValueError(f"unknown momentum {momentum!r}; use 'vector_invariant' or 'flux'")
        if momentum == "vector_invariant" and not self.ale:
            raise ValueError("momentum='vector_invariant' needs vertical_coordinate='ale' (the σ model keeps the "
                             "flux form, which matches CROCO)")
        if ke_scheme not in ("arakawa", "gudonov"):
            raise ValueError(f"unknown ke_scheme {ke_scheme!r}; use 'arakawa' or 'gudonov'")
        self.momentum, self.ke_scheme, self.max_truncations = momentum, ke_scheme, int(max_truncations)
        if thickness_flux not in ("upwind", "ppm"):
            raise ValueError(f"unknown thickness_flux {thickness_flux!r}; use 'upwind' or 'ppm'")
        self.thickness_flux = thickness_flux
        if not self.ale:
            return
        if regrid not in ("sigma", "zstar", "hybrid"):
            raise ValueError(f"unknown regrid rule {regrid!r}; use 'sigma', 'zstar' or 'hybrid'")
        if remap_scheme not in REMAP_SCHEMES:
            raise ValueError(f"unknown remap_scheme {remap_scheme!r}; use one of {sorted(REMAP_SCHEMES)}")
        if z0 is not None:
            raise ValueError("vertical_coordinate='ale' needs z0=None for now: the law-of-the-wall drag uses "
                             "the bed layer, which can vanish under z*")
        if closure not in ("constant", "none"):
            raise ValueError("vertical_coordinate='ale' needs closure 'constant' or 'none' for now "
                             "(GLS and the parabolic ν assume σ-layers)")
        params = dict(regrid_params or {})
        params.setdefault("dsigma", self.ds)
        for key in {"sigma": (), "zstar": ("dz_top",), "hybrid": ("dz_top", "rho_target")}[regrid]:
            if key not in params:
                raise ValueError(f"regrid={regrid!r} needs regrid_params[{key!r}]")
        if regrid == "hybrid":
            # the z* floor is the only thing that gives absent density targets thickness (MOM6 coord_hycom.F90:253-267);
            # one that stops above the bed leaves a homogeneous deep water mass in one slab layer with one velocity,
            # blind to topographic steps (BayIW two-layer section 2026-10-13: interface rise 0.06 of MITgcm's)
            floor, h_max = float(np.sum(params["dz_top"])), float(np.nanmax(np.asarray(self.H, float)))
            if floor < h_max * (1.0 - 1e-6):
                warnings.warn(f"hybrid z* floor (Σ dz_top = {floor:.1f} m) stops above the deepest bed ({h_max:.1f} m): "
                              "below it, layers whose target density is absent stay massless and a homogeneous water "
                              "mass is one slab layer; give dz_top the full depth (as OM4's hycom1_75_800m)",
                              UserWarning, stacklevel=3)
        floodable = act & (np.nan_to_num(np.asarray(self.H, float), nan=1.0) <= 0)
        if floodable.any() and (regrid == "zstar" or (regrid == "hybrid" and float(params.get("d_hybrid", 0.0)) <= 0)):
            raise ValueError("cells with H ≤ 0 can dry: use regrid='sigma', or 'hybrid' with a shallow σ zone "
                             "(regrid_params d_sigma, d_hybrid > 0) covering them")
        edge_cells = np.zeros_like(act)
        for edge, sl in (("west", np.s_[:, 0]), ("east", np.s_[:, -1]), ("south", np.s_[0, :]), ("north", np.s_[-1, :])):
            if edge in self.open:
                edge_cells[sl] = True
        if (edge_cells & floodable).any():
            raise ValueError("open boundaries next to cells with H ≤ 0 (which can dry) are not supported with "
                             "vertical_coordinate='ale' (spec 2026-10-12-ale-open-boundaries §4)")
        self.regrid_rule, self.regrid_params = regrid, params
        self.remap_code = REMAP_SCHEMES[remap_scheme]
        self.h_thin_bed, self.init_iterations = float(h_thin_bed), int(init_iterations)
        g = {k: np.asarray(self.hg[k].values, float) for k in ("dx_u", "dy_v", "area_t", "area_u", "area_v",
                                                               "dx_t", "dy_t")}
        g["area_q"] = vi.corner_area(g["dx_u"], g["dy_v"])
        g["mask_q"] = vi.corner_mask(act)
        self._vi = g                                                    # geometry of the vector-invariant terms

    # ------------------------------------------------------------------ helpers
    def initial_state(self, eta: np.ndarray, T: np.ndarray, h_src: np.ndarray | None = None) -> SigmaState:
        """
        Resting state; T on the layers of ``layer_heights(eta)``.  With ``rho_ref="auto"`` the reference
        profile is taken from this T.  ALE: h is the rule's grid (σ-layers for 'hybrid', followed by
        ``init_iterations`` regrid–remap passes towards the isopycnal grid).

        ALE with ``h_src`` (N_src, ny, nx) [m], bed first, Σ h_src = H + η in wet columns: T (N_src, ny, nx)
        holds the layer means on that source grid (e.g. the z-levels of the data); the hybrid grid is regridded
        on the source levels and T remapped from them, as MOM6 Z_INIT_ALE_REMAPPING
        (MOM_state_initialization.F90:2928-2935).  Without it the
        hybrid grid starts from T on σ-layers, which are ~H/N thick in deep water and lose the structure
        of the top metres (BayIW bay mouth: 0.5 °C spurious step across the 600 → 164 m shelf edge).
        """
        ny, nx = self.H.shape
        eta = np.asarray(eta, float)
        if h_src is not None:
            return self._initial_state_from_source(eta, np.asarray(T, float), np.asarray(h_src, float))
        if isinstance(self.rho_ref, str) and self.rho_ref == "auto":
            z = self.layer_heights(eta)
            act = np.broadcast_to(self.fs.active[None], z.shape)
            self.rho_ref = reference_profile(z[act], self.density(np.asarray(T, float))[act])
        self.T_ext = np.array(T, float)                                # inflow / sponge target at open boundaries
        st = SigmaState(eta=np.array(eta, float), u=np.zeros((self.n, ny, nx + 1)),
                        v=np.zeros((self.n, ny + 1, nx)), T=np.array(T, float),
                        h=self._initial_h(eta) if self.ale else None)
        if self.ale and self.regrid_rule == "hybrid":
            for _ in range(self.init_iterations):
                st.h, (st.T,) = self._regrid_remap(st.eta, st.h, [st.T], st.T)
        if self.ale:                                                    # exterior profile in z (spec §3.5)
            self.T_ext, self.h_ext = st.T.copy(), st.h.copy()
        return st

    def _initial_state_from_source(self, eta: np.ndarray, T_src: np.ndarray, h_src: np.ndarray) -> SigmaState:
        """
        ALE resting state from layer means T_src on a source grid h_src (both (N_src, ny, nx), bed first):
        the rule's grid regridded once on the source levels themselves (MOM6 MOM_state_initialization.F90:2928),
        for 'hybrid' with the model interfaces moved onto the density jumps of the source that would otherwise
        fall inside a model layer (core.vertical.regrid.snap_to_jumps, beyond MOM6), then a conservative remap
        of T from the source grid (:2933-2935).  Dry columns get T = t_ref.
        """
        if not self.ale:
            raise ValueError("initial_state(h_src=...) needs vertical_coordinate='ale'")
        ny, nx = self.H.shape
        if h_src.ndim != 3 or h_src.shape != T_src.shape or h_src.shape[1:] != (ny, nx):
            raise ValueError(f"h_src and T must both be (N_src, {ny}, {nx}); got {h_src.shape} and {T_src.shape}")
        H = np.nan_to_num(np.asarray(self.H, float))
        D = H + eta
        wet = self.fs.active & (D > 0)
        if (h_src < 0).any() or not np.allclose(h_src.sum(axis=0)[wet], D[wet], rtol=1e-10, atol=1e-9):
            raise ValueError("h_src must be ≥ 0 and fill every wet column: Σ h_src = H + η")
        if isinstance(self.rho_ref, str) and self.rho_ref == "auto":
            zc = -H[None] + np.cumsum(h_src, axis=0) - 0.5 * h_src
            act = wet[None] & (h_src > 0)
            self.rho_ref = reference_profile(zc[act], self.density(T_src)[act])
        numba.set_num_threads(_THREADS[0])

        def from_source(h_tgt: np.ndarray) -> np.ndarray:
            T = rm.remap_columns(h_src, T_src, h_tgt, self.remap_code, wet)
            return np.where(wet[None], T, self.t_ref)

        h = self._initial_h(eta)
        if self.regrid_rule == "hybrid":
            # one regrid on the source levels themselves, as MOM6 (regridding_main on the data, :2928); iterating
            # on the model grid (regrid there, remap from the source) oscillates around a sharp jump (BayIW
            # two-layer with a full-depth z* floor: 24–348 mixed layers depending on the pass count)
            rho_src = self.density(np.where(wet[None], T_src, self.t_ref))
            zw = rg.regrid_columns("hybrid", H, eta, h_src, rho_src, self.regrid_params, wet, n_out=self.n)
            zw_src = -H[None] + np.concatenate([np.zeros((1,) + H.shape), np.cumsum(h_src, axis=0)])
            tgt = np.asarray(self.regrid_params["rho_target"], float)
            rg.snap_to_jumps(zw, zw_src, rho_src, float(np.min(np.abs(np.diff(tgt)))))
            h = np.where(wet[None], np.diff(zw, axis=0), h)
        T = from_source(h)
        st = SigmaState(eta=np.array(eta, float), u=np.zeros((self.n, ny, nx + 1)),
                        v=np.zeros((self.n, ny + 1, nx)), T=T, h=h)
        self.T_ext, self.h_ext = T.copy(), h.copy()                    # exterior profile in z (spec §3.5)
        return st

    def layer_heights(self, eta: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """
        Height of the layer centres [m] (z = 0 at the rest surface), (N, ny, nx): σ-layers
        z_k = −H + D σ_c,k, or the centres of thicknesses h (ALE; default the initial grid).
        """
        if h is None and self.ale:
            h = self._initial_h(np.asarray(eta, float))
        if h is None:
            D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
            return -np.nan_to_num(self.H)[None] + D[None] * self.sigma_c[:, None, None]
        for ref, val in getattr(self, "_zc_cache", ()):          # cached by identity of h (performance task 5.6)
            if ref is h:
                return val
        zw = -np.nan_to_num(self.H)[None] + np.concatenate([np.zeros((1,) + h.shape[1:]), np.cumsum(h, axis=0)])
        val = 0.5 * (zw[1:] + zw[:-1])
        self._zc_cache = ((h, val),) + tuple(getattr(self, "_zc_cache", ()))[:1]
        return val

    def density(self, T: np.ndarray) -> np.ndarray:
        """Linear EOS ρ = ρ₀ − α (T − T₀) [kg m⁻³]."""
        return self.rho0 - self.t_coef * (T - self.t_ref)

    def layer_volumes(self, eta: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Layer volumes [m³]: D Δσ_k A, or h_k A (ALE)."""
        if h is not None:
            return h * self.area[None]
        D = np.maximum(np.nan_to_num(self.H + eta, nan=0.0), 0.0)
        return D[None] * self.ds[:, None, None] * self.area[None]

    def tracer_content(self, st: SigmaState, power: int = 1) -> float:
        V = self.layer_volumes(st.eta, st.h)
        return float(np.sum(V * st.T**power))

    def rpe(self, st: SigmaState) -> float:
        """Reference potential energy [J] (flat-bottom basin: area = wet area)."""
        V = self.layer_volumes(st.eta, st.h)
        act = np.broadcast_to(self.fs.active[None], V.shape)
        return reference_potential_energy(self.density(st.T)[act], V[act], area=float(self.area[self.fs.active].sum()))

    # ------------------------------------------------------------------ ALE helpers
    def _initial_h(self, eta: np.ndarray) -> np.ndarray:
        """
        ALE layer thicknesses of the initial grid at η [m]; 0 on dry/land.  'sigma' and 'zstar': the rule's
        grid (σ where H ≤ 0).  'hybrid': σ-layers, on which T is sampled before ``init_iterations``
        regrid–remap passes build the isopycnal grid, as MOM6 (THICKNESS_CONFIG = "uniform",
        MOM_state_initialization.F90:271; REGRID_ACCELERATE_INIT, :488-507).  The z* floor of dz_top
        (e.g. 19 × 0.5 m + one 190 m bed layer) would lose a continuous stratification.
        """
        H = np.nan_to_num(np.asarray(self.H, float))
        wet = self.fs.active & (H + eta > 0)
        h0 = np.zeros((self.n,) + H.shape)
        rule = "sigma" if self.regrid_rule == "hybrid" else self.regrid_rule
        sig = wet & (H <= 0) if rule == "zstar" else np.zeros_like(wet)
        numba.set_num_threads(_THREADS[0])                                 # parallel regrid kernel
        for r, mask in ((rule, wet & ~sig), ("sigma", sig)):
            if mask.any():
                zw = rg.regrid_columns(r, H, eta, h0, None, self.regrid_params, mask)
                h0 = np.where(mask[None], np.diff(zw, axis=0), h0)
        return h0

    def _regrid_remap(self, eta: np.ndarray, h_src: np.ndarray, fields: list, T: np.ndarray):
        """Regrid with the rule (spec §3) at surface η and remap ``fields`` (spec §4): (h_new, remapped)."""
        H = np.nan_to_num(np.asarray(self.H, float))
        wet = self.fs.active & (H + eta > 0) & (h_src.sum(axis=0) > 0)
        h_new = self._regrid(eta, h_src, T, wet)
        return h_new, [rm.remap_columns(h_src, f, h_new, self.remap_code, wet) for f in fields]

    def _regrid(self, eta: np.ndarray, h: np.ndarray, T: np.ndarray, wet: np.ndarray | None = None) -> np.ndarray:
        """New layer thicknesses of the rule (spec §3) from layers h with temperatures T; h kept where dry."""
        H = np.nan_to_num(np.asarray(self.H, float))
        if wet is None:
            wet = self.fs.active & (H + eta > 0) & (h.sum(axis=0) > 0)
        rho = self.density(T) if self.regrid_rule == "hybrid" else None
        numba.set_num_threads(_THREADS[0])                                 # parallel regrid kernel
        zw = rg.regrid_columns(self.regrid_rule, H, eta, h, rho, self.regrid_params, wet)
        return np.where(wet[None], np.diff(zw, axis=0), h)

    def _transport_thickness(self, h: np.ndarray):
        """
        Upwind face thicknesses ((q_E, q_W) at u-faces, (q_N, q_S) at v-faces swapped to (N, nx, ny+1)):
        clipped at the shallower bed for z* and hybrid, not for σ (core.vertical.ale).  Cached by array
        identity for the last two thickness arrays (hⁿ is used by the tendencies and the step; h_new of one
        step is hⁿ of the next); thickness arrays are never written in place after creation.
        """
        for ref, val in getattr(self, "_tt_cache", ()):
            if ref is h:
                return val
        clip = self.regrid_rule != "sigma"
        H, act = np.nan_to_num(np.asarray(self.H, float)), self.fs.active
        x = face_transport_thickness(h, H, act, clip)
        y = face_transport_thickness(h.swapaxes(1, 2), H.T, act.T, clip)
        val = (x, y)
        self._tt_cache = ((h, val),) + tuple(getattr(self, "_tt_cache", ()))[:1]
        return val

    def _mean_face_thickness(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Direction-independent face thickness (q_E + q_W)/2 at u- and v-faces [m]."""
        (qe, qw), (qn, qs) = self._transport_thickness(h)
        return 0.5 * (qe + qw), (0.5 * (qn + qs)).swapaxes(1, 2)

    def _remap_faces(self, vel: np.ndarray, hf_src: np.ndarray, hf_tgt: np.ndarray,
                     w_tgt: np.ndarray | None = None) -> np.ndarray:
        """
        Remap face velocities between two face-layer geometries in fraction space: thicknesses
        w_src S → w_tgt S with the same column total S, so Σ w u (the depth mean of the free
        surface) is conserved exactly (spec §2 step 4).  ``w_tgt``: face_fractions(hf_tgt) if already known.
        """
        tot = hf_src.sum(axis=0)
        wet = tot > 0
        S = np.where(wet, tot, 1.0)[None]
        w_tgt = face_fractions(hf_tgt) if w_tgt is None else w_tgt
        return rm.remap_columns(face_fractions(hf_src) * S, vel, w_tgt * S, self.remap_code, wet)

    # ------------------------------------------------------------------ explicit tendencies
    def _baroclinic_pgf(self, eta: np.ndarray, T: np.ndarray, h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Baroclinic pressure-gradient force at u- and v-faces [m s⁻²].

        ρ′ = ρ − ρ_ref(z) with ρ_ref = ρ₀, or a reference profile ``rho_ref(z)``
        (Haney 1991; the "balanced" method of Berntsen et al. 2015): the pressure of
        a horizontally uniform ρ_ref(z) has no gradient at fixed z, so subtracting it
        removes the large cancelling terms of a resting stratification.

        'twoterm':  PGF = −(1/ρ₀)[∂P′/∂x|_σ + g ρ′ ∂z/∂x|_σ]   (second order; large errors
                     over steep slopes — Haney 1991, seamount max|u| ~50 cm/s)
        'jacobian': density-Jacobian form (Song 1998), integrated down from the surface
                     between layer centres k and k+1:
            ∂P′/∂x|_z(z_k) = ∂P′/∂x|_z(z_{k+1}) + (g/Δx)[δ_x ρ̄ · (δ_k z)‾ − δ_x z̄ · (δ_k ρ)‾]
                     starting at the surface z = η with ρ′ extrapolated linearly from the two
                     top layers.  Exact for ρ linear in z over any topography.
        """
        if self.pgf == "fv":
            return self._fv_pgf(eta, T, h)
        n = self.n
        if h is None:
            D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
            h = D[None] * self.ds[:, None, None]
            z = -np.nan_to_num(self.H)[None] + D[None] * self.sigma_c[:, None, None]   # cell-centre height (z = 0 at rest)
        else:
            z = self.layer_heights(eta, h)
        rho = self.density(T)
        rp = rho - (self.rho_ref(z) if self.rho_ref is not None else self.rho0)
        g = self.g
        # surface values for the Jacobian: z_s = η, ρ′ linearly extrapolated from the two top layers
        z_s = np.nan_to_num(eta)[None]
        if n > 1:
            r_s = (rp[-1] + (rp[-1] - rp[-2]) * (z_s[0] - z[-1]) / np.where(z[-1] - z[-2] != 0, z[-1] - z[-2], 1.0))[None]
        else:
            r_s = rp[-1:]
        if self.pgf == "twoterm":
            above = np.cumsum((rp * h)[::-1], axis=0)[::-1] - rp * h       # ∫ over the layers above
            P = g * (above + 0.5 * rp * h)

        def along(a_lo, a_hi, z_lo, z_hi, P_lo=None, P_hi=None, s_lo=None, s_hi=None):
            """∂P′/∂x|_z × Δx at faces between the 'lo' and 'hi' columns, all layers."""
            if self.pgf == "twoterm":
                return (P_hi - P_lo) + g * 0.5 * (a_hi + a_lo) * (z_hi - z_lo)
            # Jacobian from the surface (z = η, P′ = 0 there; surface term g ρ̄′_s δ_x η) down to the
            # top layer centre, then between consecutive layer centres
            (rs_lo, zs_lo), (rs_hi, zs_hi) = s_lo, s_hi
            a_lo, a_hi = np.concatenate([a_lo, rs_lo]), np.concatenate([a_hi, rs_hi])
            z_lo, z_hi = np.concatenate([z_lo, zs_lo]), np.concatenate([z_hi, zs_hi])
            out = np.empty(a_lo.shape)
            out[-1] = g * 0.5 * (rs_hi[0] + rs_lo[0]) * (zs_hi[0] - zs_lo[0])
            rbar_lo, rbar_hi = 0.5 * (a_lo[1:] + a_lo[:-1]), 0.5 * (a_hi[1:] + a_hi[:-1])
            zbar_lo, zbar_hi = 0.5 * (z_lo[1:] + z_lo[:-1]), 0.5 * (z_hi[1:] + z_hi[:-1])
            dz = 0.5 * ((z_lo[1:] - z_lo[:-1]) + (z_hi[1:] - z_hi[:-1]))
            dr = 0.5 * ((a_lo[1:] - a_lo[:-1]) + (a_hi[1:] - a_hi[:-1]))
            inc = g * ((rbar_hi - rbar_lo) * dz - (zbar_hi - zbar_lo) * dr)   # pair (k, k+1), k = 0 … N−1 (N = surface)
            out[:-1] = out[-1] + np.cumsum(inc[::-1], axis=0)[::-1]
            return out[:-1]

        gx = np.zeros((n,) + self.fs.H_u.shape)
        gy = np.zeros((n,) + self.fs.H_v.shape)
        okx = np.isfinite(self.fs.H_u[:, 1:-1])
        oky = np.isfinite(self.fs.H_v[1:-1, :])
        if self.pgf == "twoterm":
            ax = along(rp[..., :-1], rp[..., 1:], z[..., :-1], z[..., 1:], P[..., :-1], P[..., 1:])
            ay = along(rp[:, :-1], rp[:, 1:], z[:, :-1], z[:, 1:], P[:, :-1], P[:, 1:])
        else:
            ax = along(rp[..., :-1], rp[..., 1:], z[..., :-1], z[..., 1:],
                       s_lo=(r_s[..., :-1], z_s[..., :-1]), s_hi=(r_s[..., 1:], z_s[..., 1:]))
            ay = along(rp[:, :-1], rp[:, 1:], z[:, :-1], z[:, 1:],
                       s_lo=(r_s[:, :-1], z_s[:, :-1]), s_hi=(r_s[:, 1:], z_s[:, 1:]))
        gx[:, :, 1:-1] = np.where(okx, -ax / self.fs.du[:, 1:-1] / self.rho0, 0.0)
        gy[:, 1:-1, :] = np.where(oky, -ay / self.fs.dv[1:-1, :] / self.rho0, 0.0)
        return gx, gy

    def _fv_pgf(self, eta: np.ndarray, T: np.ndarray, h: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Finite-volume PGF (core.vertical.pgf) on σ-layers (h=None) or ALE layers, with ρ_ref(z) if set."""
        if h is None:
            D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
            h = D[None] * self.ds[:, None, None]
        ref = None if isinstance(self.rho_ref, str) else self.rho_ref        # "auto" before initial_state: ρ₀
        numba.set_num_threads(_THREADS[0])
        gx, gy = fv_pgf(h, np.nan_to_num(self.H), np.nan_to_num(eta), np.nan_to_num(T), self.fs.du, self.fs.dv,
                        self.rho0, self.t_coef, self.t_ref, ref, self.g)
        return (np.where(np.isfinite(self.fs.H_u)[None], gx, 0.0),
                np.where(np.isfinite(self.fs.H_v)[None], gy, 0.0))

    def _coriolis(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Energy-conserving C-grid Coriolis terms (+f v̄ at u-faces, −f ū at v-faces):
        every adjacent (u, v) pair shares one cell corner q and is weighted by ¼ f_q,
        so the operator is antisymmetric and does no work (Sadourny 1975; Arakawa &
        Lamb 1977).  u (N, ny, nx+1), v (N, ny+1, nx).
        """
        fq = self._fq
        cu = np.zeros_like(u)
        cv = np.zeros_like(v)
        cu[..., 1:-1] = 0.25 * (fq[:-1, 1:-1] * (v[:, :-1, :-1] + v[:, :-1, 1:])
                                + fq[1:, 1:-1] * (v[:, 1:, :-1] + v[:, 1:, 1:]))
        cv[:, 1:-1, :] = -0.25 * (fq[1:-1, :-1] * (u[:, :-1, :-1] + u[:, 1:, :-1])
                                  + fq[1:-1, 1:] * (u[:, :-1, 1:] + u[:, 1:, 1:]))
        return cu, cv

    def _vector_invariant(self, st: SigmaState) -> tuple[np.ndarray, np.ndarray]:
        """
        Vector-invariant momentum terms of the ALE step (spec 2026-10-12 §3): Sadourny (1975) energy
        PV flux q·(vh, −uh) with BOUND_CORIOLIS, minus ∇K (core.dynamics.vector_invariant).  The
        transports are those of the last Lagrangian step (as MOM6 uses the uh of its last continuity);
        the first step uses L u h̄.
        """
        g = self._vi
        numba.set_num_threads(_THREADS[0])                                 # vector-invariant kernels
        zeta, q = vi.vorticity_and_pv(st.u, st.v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"], self._fq, st.h,
                                      g["area_t"], self.fs.active)
        Qx, Qy = st.Qx, st.Qy
        if Qx is None or Qy is None:
            hu, hv = self._mean_face_thickness(st.h)
            Qx, Qy = self.fs.Lu[None] * st.u * hu, self.fs.Lv[None] * st.v * hv
        cu, cv = vi.coriolis_pv(q, self._fq[None] + zeta, Qx, Qy, st.u, st.v, g["dx_u"], g["dy_v"], bound=True)
        K = vi.kinetic_energy(st.u, st.v, g["area_u"], g["area_v"], g["area_t"], self.ke_scheme)
        kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
        return cu + kx, cv + ky

    def _vertical_advection(self, vel: np.ndarray, omega_f: np.ndarray, h_f: np.ndarray) -> np.ndarray:
        """(ω/h) ∂u/∂k, centred: [ω_{k+½}(u_{k+1}−u_k) + ω_{k−½}(u_k−u_{k−1})] / (2h)."""
        out = np.zeros_like(vel)
        du = vel[1:] - vel[:-1]                                             # at interior interfaces
        out[:-1] += omega_f[1:-1] * du
        out[1:] += omega_f[1:-1] * du
        return out / (2.0 * np.maximum(h_f, 1e-6))

    def _tendencies(self, st: SigmaState) -> tuple[np.ndarray, np.ndarray]:
        n = self.n
        Du, Dv = self.core.face_depths(st.eta)
        wet_u = np.isfinite(self.fs.H_u) & np.isfinite(Du) & (Du > self.fs.d_min)
        wet_v = np.isfinite(self.fs.H_v) & np.isfinite(Dv) & (Dv > self.fs.d_min)
        gu, gv = self._baroclinic_pgf(st.eta, st.T, st.h)
        vinv = self.ale and self.momentum == "vector_invariant"
        cu, cv = self._vector_invariant(st) if vinv else self._coriolis(st.u, st.v)
        gu += cu
        gv += cv
        if self.adv == "flux" and not vinv:
            if st.Qx is not None:
                V = self.layer_volumes(st.eta, st.h)
                lim = self.dt if self.ale else 0.0                                      # vanished-cell limiter
                gu += _flux_form_u(st.u, st.Qx, st.Qy, st.Fz, V, wet_u[None], lim)
                swap = lambda a: np.swapaxes(a, 1, 2)                                   # noqa: E731
                gv += swap(_flux_form_u(swap(st.v), swap(st.Qy), swap(st.Qx), swap(st.Fz), swap(V),
                                        swap(wet_v[None]), lim))
        if self.adv != "flux" and not vinv:
            for k in range(n):
                gu[k] -= advection_u(st.u[k], st.v[k], self.hg, wet_u, scheme=self.adv)
                gv[k] -= advection_v(st.u[k], st.v[k], self.hg, wet_v, scheme=self.adv)
        if self.ah > 0:                                                 # spacings built once (performance task 5.2)
            if getattr(self, "_lap", None) is None:
                self._lap = (_spacings_u(self.hg), _spacings_v(self.hg))
            numba.set_num_threads(_THREADS[0])
            gu += self.ah * laplacian_3d(st.u, wet_u, *self._lap[0])
            gv += self.ah * laplacian_3d(st.v, wet_v, *self._lap[1])
        if st.omega is not None and self.adv != "flux":
            om_u = np.zeros((n + 1,) + Du.shape)
            om_u[:, :, 1:-1] = 0.5 * (st.omega[:, :, 1:] + st.omega[:, :, :-1])
            om_v = np.zeros((n + 1,) + Dv.shape)
            om_v[:, 1:-1, :] = 0.5 * (st.omega[:, 1:, :] + st.omega[:, :-1, :])
            gu -= self._vertical_advection(st.u, om_u, np.nan_to_num(Du)[None] * self.ds[:, None, None])
            gv -= self._vertical_advection(st.v, om_v, np.nan_to_num(Dv)[None] * self.ds[:, None, None])
        return np.where(wet_u[None], gu, 0.0), np.where(wet_v[None], gv, 0.0)

    # ------------------------------------------------------------------ tracers
    def _faces(self, c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int, centred: bool = False) -> np.ndarray:
        if centred:
            return _centred_faces(c, ok, axis, n_out=flux.shape[axis])
        if self.tracer_scheme.startswith("weno5"):
            return _weno5_faces(c, flux, ok, axis, variant="js" if self.tracer_scheme == "weno5js" else "z")
        return _up3_faces(c, flux, ok, axis, order=1 if self.tracer_scheme == "upwind1" else 3)

    def _open_faces(self, face: np.ndarray, c: np.ndarray, Q: np.ndarray, axis: int) -> np.ndarray:
        """
        Tracer values on the open-boundary faces: upstream (Marsaleix et al. 2006, advection
        equation for tracers) — the boundary cell when flowing out, the external value
        ``T_ext`` (and its square for the T² stack) when flowing in.
        """
        if not self.open:
            return face
        T_in = self._inflow_T()
        ext = np.stack([T_in, T_in**2])[: c.shape[0]]
        face = np.array(face, copy=True)
        lo, hi = ("west", "east") if axis == 3 else ("south", "north")
        for edge, f_i, c_i, inflow_sign in ((lo, 0, 0, 1.0), (hi, -1, -1, -1.0)):
            if edge not in self.open:
                continue
            sl_f = [slice(None)] * face.ndim
            sl_c = [slice(None)] * c.ndim
            sl_f[axis], sl_c[axis] = f_i, c_i
            q = np.take(Q, f_i, axis=axis - 1)
            face[tuple(sl_f)] = np.where(inflow_sign * q > 0, ext[tuple(sl_c)], c[tuple(sl_c)])
        return face

    def _diffuse_h(self, T: np.ndarray, V: np.ndarray, eta: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """
        Explicit Laplacian diffusion along the layers (see _diffuse_h_numpy).  ALE: the Numba kernel
        _diffuse_ale_kernel with the same per-element arithmetic (performance task 5.6); σ: the NumPy version.
        """
        if h is None:
            return self._diffuse_h_numpy(T, V, eta, h)
        Tw = T
        if self.rho_ref is not None and not isinstance(self.rho_ref, str) and self.t_coef != 0:
            Tw = T - (self.t_ref + (self.rho0 - self.rho_ref(self.layer_heights(eta, h))) / self.t_coef)
        Du, Dv = self.core.face_depths(eta)
        okx = np.isfinite(self.fs.H_u) & np.isfinite(Du) & (Du > self.fs.d_min)
        oky = np.isfinite(self.fs.H_v) & np.isfinite(Dv) & (Dv > self.fs.d_min)
        act = self.fs.active
        c = np.ascontiguousarray
        out = np.empty(np.shape(T))
        numba.set_num_threads(_THREADS[0])
        _diffuse_ale_kernel(c(T, dtype=float), c(Tw, dtype=float), c(np.where(act[None], np.nan_to_num(h), 0.0)),
                            c(act), c(self.fs.Lu / self.fs.du), c(self.fs.Lv / self.fs.dv), c(okx), c(oky),
                            c(V, dtype=float), float(self.kh), float(self.dt), out)
        return out

    def _diffuse_h_numpy(self, T: np.ndarray, V: np.ndarray, eta: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """
        NumPy reference of _diffuse_h (test oracle; used for σ).
        Explicit Laplacian diffusion along σ-surfaces in flux form (MITgcm diffKhT):
            V_k ∂T/∂t = −Σ_faces F,   F_x = −K_h L_u h_{u,k} (T_i − T_{i−1}) / Δx_u
        with h_{u,k} = D_u Δσ_k (σ) or the face layer thickness min(h_L, h_R) (ALE); conservative, and a uniform
        T stays uniform.  With a reference profile the anomaly T − T_ref(z) is diffused instead
        (Mellor & Blumberg 1985 [verificar]), so a resting stratification is not mixed across
        sloping σ-surfaces (e.g. a shelf step).
        """
        Tw = T
        if self.rho_ref is not None and not isinstance(self.rho_ref, str) and self.t_coef != 0:
            Tw = T - (self.t_ref + (self.rho0 - self.rho_ref(self.layer_heights(eta, h))) / self.t_coef)
        Du, Dv = self.core.face_depths(eta)
        okx = np.isfinite(self.fs.H_u) & np.isfinite(Du) & (Du > self.fs.d_min)
        oky = np.isfinite(self.fs.H_v) & np.isfinite(Dv) & (Dv > self.fs.d_min)
        ds = self.ds[:, None, None]
        Fx = np.zeros((self.n,) + Du.shape)
        Fy = np.zeros((self.n,) + Dv.shape)
        if h is None:
            Fx[..., 1:-1] = -self.kh * (self.fs.Lu * np.nan_to_num(Du) / self.fs.du)[None, :, 1:-1] * ds \
                * (Tw[..., 1:] - Tw[..., :-1])
            Fy[:, 1:-1, :] = -self.kh * (self.fs.Lv * np.nan_to_num(Dv) / self.fs.dv)[None, 1:-1, :] * ds \
                * (Tw[:, 1:, :] - Tw[:, :-1, :])
        else:                                                        # ALE: face layer thickness (min of the cells)
            hu, hv = face_thickness(h, self.fs.active)
            Fx[..., 1:-1] = -self.kh * (self.fs.Lu / self.fs.du)[None, :, 1:-1] * hu[..., 1:-1] \
                * (Tw[..., 1:] - Tw[..., :-1])
            Fy[:, 1:-1, :] = -self.kh * (self.fs.Lv / self.fs.dv)[None, 1:-1, :] * hv[:, 1:-1, :] \
                * (Tw[:, 1:, :] - Tw[:, :-1, :])
        Fx = np.where(okx[None], Fx, 0.0)
        Fy = np.where(oky[None], Fy, 0.0)
        div = Fx[..., 1:] - Fx[..., :-1] + Fy[:, 1:, :] - Fy[:, :-1, :]
        return np.where(V > 0, T - self.dt * div / np.where(V > 0, V, 1.0), T)

    def _baroclinic_part(self, vel: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Depth mean and baroclinic remainder u_k − ū: Σ Δσ_k u_k (σ, ``w`` None — kept as tensordot so
        σ runs stay bit-identical) or Σ w_k u_k with the ALE face fractions (core.vertical.ale.baroclinic_split).
        """
        if w is None:
            mean = np.tensordot(self.ds, vel, axes=(0, 0))
            return mean, vel - mean[None]
        return baroclinic_split(vel, w)

    def _radiate(self, u_old: np.ndarray, u_new: np.ndarray, axis: int, edge: int,
                 w: np.ndarray | None = None, ext: np.ndarray | None = None, mode: str = "radiating") -> None:
        """
        Orlanski (1976) radiation of the baroclinic normal velocity at one open edge, implicit
        upstream form as ROMS u3dbc (Marchesiello et al. 2001, normal part only):

            r = c Δt/Δx = (φ₁ⁿ − φ₁ⁿ⁺¹)(φ₁ⁿ − φ₂ⁿ) / (φ₁ⁿ − φ₂ⁿ)²,  clipped to [0, r_max]
            φ_Bⁿ⁺¹ = (φ_Bⁿ + r φ₁ⁿ⁺¹) / (1 + r)

        with B the boundary face and 1, 2 the next faces inwards (r = 0: incoming → clamped).
        The depth-mean part (Flather) is kept; u_new is modified in place.  ``w``: depth-mean weights
        (Δσ when None, the ALE face fractions of hⁿ for both levels; spec 2026-10-12-ale-open-boundaries §3.2).
        ``ext``: exterior u′ on the face layers (spec 2026-10-12-baroclinic-obc-internal-wave §2): 'radiating'
        nudges the radiated value toward it, u′_B ← (1 − α) u′_B + α u′ₑₓₜ with α = Δt/τ_in when flowing in
        (r = 0) and Δt/τ_out when out (CROCO u3dbc.F:119-127, 144-152); 'clamped' sets u′_B = u′ₑₓₜ.
        """
        _, po = self._baroclinic_part(u_old, w)
        bar, pn = self._baroclinic_part(u_new, w)
        i1, i2 = (1, 2) if edge == 0 else (-2, -3)
        take = lambda a, i: np.take(a, i, axis=axis)                                  # noqa: E731
        grad = take(po, i1) - take(po, i2)
        cx = (take(po, i1) - take(pn, i1)) * grad
        r = np.clip(cx / np.maximum(grad**2, 1e-30), 0.0, self.orlanski_cmax)
        pb = (take(po, edge) + r * take(pn, i1)) / (1.0 + r)
        if ext is not None:
            pe = take(ext, edge)
            if mode == "clamped":
                pb = pe.copy()
            else:
                alpha = np.where(cx <= 0.0, self.dt / self.obc_tau_in, self.dt / self.obc_tau_out)
                pb = (1.0 - alpha) * pb + alpha * pe
        if w is None:
            pb -= np.tensordot(self.ds, pb, axes=(0, 0))[None]                        # keep Σ Δσ u′ = 0
        else:
            pb -= (take(w, edge) * pb).sum(axis=0)[None]                              # keep Σ w u′ = 0
        open_ = take((self.fs.last_open_Du if axis == 2 else self.fs.last_open_Dv)[None], edge) > 0
        sl = [slice(None)] * u_new.ndim
        sl[axis] = edge
        u_new[tuple(sl)] = np.where(open_, take(bar[None], edge) + pb, take(u_new, edge))

    def _flather_args(self) -> dict:
        """Flather radiation to a resting exterior (η = 0, ū = 0) for the free-surface step; {} if closed."""
        if not self.open:
            return {}
        if self._ext is not None:
            return self._ext["flather"]
        zu, zv = np.zeros(self.fs.H_u.shape), np.zeros(self.fs.H_v.shape)
        return dict(flather_eta=(zu, zv), flather_ubar=zu, flather_vbar=zv)

    def _open_velocity(self, st: SigmaState, u_new: np.ndarray, v_new: np.ndarray,
                       wu: np.ndarray | None = None, wv: np.ndarray | None = None) -> None:
        """
        In place: flow-relaxation sponge on the baroclinic velocity (implicit; the depth mean, and so the
        free-surface fluxes, are untouched — barotropic waves leave through Flather), then Orlanski
        radiation at the open edges.  Weights wu, wv: Δσ (None) or the ALE face fractions (spec §3.2-3.3).
        """
        ext_u, ext_v = (None, None) if self._ext is None else (self._ext["up_u"], self._ext["up_v"])
        if self._sp_u is not None:
            for vel, rate, w, pe in ((u_new, self._sp_u, wu, ext_u), (v_new, self._sp_v, wv, ext_v)):
                bar, prime = self._baroclinic_part(vel, w)
                if pe is None:
                    vel[...] = bar[None] + prime / (1.0 + self.dt * rate[None])
                else:                                                     # toward the exterior u′
                    r = self.dt * rate[None]
                    vel[...] = bar[None] + (prime + r * pe) / (1.0 + r)
        for name, vel, old, axis, edge, w, pe in (("west", u_new, st.u, 2, 0, wu, ext_u),
                                                  ("east", u_new, st.u, 2, -1, wu, ext_u),
                                                  ("south", v_new, st.v, 1, 0, wv, ext_v),
                                                  ("north", v_new, st.v, 1, -1, wv, ext_v)):
            if name in self.open:
                self._radiate(old, vel, axis, edge, w, ext=pe, mode=self.obc_mode[name])

    def _evaluate_exterior(self, t: float, hu: np.ndarray, hv: np.ndarray, wu, wv) -> None:
        """Exterior state at t (spec 2026-10-12-baroclinic-obc-internal-wave §1.4): Flather targets and u′ₑₓₜ on
        the face layers of the open faces and sponge band; ``self._ext`` None without a provider."""
        if self.obc_state is None:
            self._ext = None
            return
        state = self.obc_state(t)
        obc.validate_state(state, self.open, self.H.shape)
        Hu, Hv = self._edge_face_depths()
        wu_ = self.ds[:, None, None] * np.ones(hu.shape) if wu is None else wu
        wv_ = self.ds[:, None, None] * np.ones(hv.shape) if wv is None else wv
        code = getattr(self, "remap_code", rm.PPM_H4)
        self._ext = dict(t=t, state=state, flather=obc.flather_targets(state, Hu.shape, Hv.shape),
                         up_u=obc.exterior_velocity(state, "u", Hu, hu, self._bc_u, wu_, code, self.open),
                         up_v=obc.exterior_velocity(state, "v", Hv, hv, self._bc_v, wv_, code, self.open),
                         own=obc.owned_by_state(state, self.open, self.H.shape) & self._bc_mask)

    def _edge_face_depths(self) -> tuple[np.ndarray, np.ndarray]:
        """Rest depth at u- and v-faces [m], 0 on land; the open edge faces (NaN in H_u/H_v, outside the
        domain) take the depth of their boundary cell."""
        H = np.nan_to_num(np.asarray(self.H, float))
        Hu, Hv = np.nan_to_num(self.fs.H_u), np.nan_to_num(self.fs.H_v)
        for edge, (A, sl_f, sl_c) in (("west", (Hu, np.s_[:, 0], np.s_[:, 0])), ("east", (Hu, np.s_[:, -1], np.s_[:, -1])),
                                      ("south", (Hv, np.s_[0, :], np.s_[0, :])), ("north", (Hv, np.s_[-1, :], np.s_[-1, :]))):
            if edge in self.open:
                A[sl_f] = H[sl_c]
        return Hu, Hv

    def _exterior_T(self, h: np.ndarray) -> np.ndarray:
        """Tₑₓₜ on layers h in the boundary/sponge columns: from the provider when set, else the z profile."""
        rest = self._exterior(h) if self.ale else self.T_ext
        if self._ext is None:
            return rest
        T = obc.exterior_tracer(self._ext["state"], self.H, h, self._ext_cols, getattr(self, "remap_code", rm.PPM_H4),
                                self.open)
        return np.where(self._ext["own"][None], T, rest)              # edges without data: resting exterior

    def _inflow_T(self) -> np.ndarray:
        """Exterior T on the layers of this step: σ, ``T_ext``; ALE, the z profile remapped to hⁿ (spec §3.5)."""
        return self.T_ext if self._T_in is None else self._T_in

    def _exterior(self, h: np.ndarray) -> np.ndarray:
        """
        Exterior T profile remapped onto thicknesses h in the boundary/sponge columns (core.vertical.ale).
        ``T_ext`` and ``h_ext`` go together (both set by ``initial_state``); replace them together.
        """
        if self.h_ext is None or self.T_ext is None or np.shape(self.h_ext) != np.shape(self.T_ext) \
                or np.shape(self.h_ext) != h.shape:
            raise ValueError("ALE open boundaries need the exterior profile T_ext and h_ext, set together and "
                             f"with the layer shape {h.shape} (call initial_state, or set both)")
        numba.set_num_threads(_THREADS[0])                                 # parallel remap kernel
        return exterior_on_layers(self.h_ext, self.T_ext, h, self._ext_cols, self.remap_code)

    def _open_face_transports(self, Qx: np.ndarray, Qy: np.ndarray, h: np.ndarray, un: np.ndarray, vn: np.ndarray,
                              u_new: np.ndarray, v_new: np.ndarray) -> None:
        """
        In place: layer fluxes through the open faces with the interior thickness, adding up to the
        Flather flux of the free surface (spec §3.4; core.vertical.ale.open_face_transport).
        """
        th = self.fs.theta
        for edge, Q, D, Qo, L, hc, vo, vn1 in (
                ("west", Qx[..., 0], self.fs.last_open_Du[:, 0], self.fs.last_open_Qu[:, 0], self.fs.Lu[:, 0],
                 h[..., 0], un[..., 0], u_new[..., 0]),
                ("east", Qx[..., -1], self.fs.last_open_Du[:, -1], self.fs.last_open_Qu[:, -1], self.fs.Lu[:, -1],
                 h[..., -1], un[..., -1], u_new[..., -1]),
                ("south", Qy[:, 0, :], self.fs.last_open_Dv[0, :], self.fs.last_open_Qv[0, :], self.fs.Lv[0, :],
                 h[:, 0, :], vn[:, 0, :], v_new[:, 0, :]),
                ("north", Qy[:, -1, :], self.fs.last_open_Dv[-1, :], self.fs.last_open_Qv[-1, :], self.fs.Lv[-1, :],
                 h[:, -1, :], vn[:, -1, :], v_new[:, -1, :])):
            if edge in self.open:
                Q[...] = np.where((D > 0)[None], open_face_transport(hc, vn1, vo, Qo, L, th, self.theta_bc), Q)

    def _rhs_numpy(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, ok: np.ndarray,
                   centred: bool = False) -> np.ndarray:
        """Advection RHS −∇·(Q c) of the stack c (ntr, N, ny, nx), vectorised reference path."""
        okb = np.broadcast_to(ok, c.shape)
        fx = Qx * self._open_faces(self._faces(c, np.broadcast_to(Qx, c.shape[:1] + Qx.shape), okb, 3, centred),
                                   c, Qx, 3)
        fy = Qy * self._open_faces(self._faces(c, np.broadcast_to(Qy, c.shape[:1] + Qy.shape), okb, 2, centred),
                                   c, Qy, 2)
        # vertical: Fz[j] is the volume flux through interface j (bottom of layer j; 0 = bed, N = surface)
        fzc = np.broadcast_to(Fz[:-1], c.shape)
        fz = np.zeros(c.shape[:1] + Fz.shape)
        fz[:, :-1] = Fz[:-1] * self._faces(c, fzc, okb, 1, centred)
        return -(fx[..., 1:] - fx[..., :-1] + fy[:, :, 1:, :] - fy[:, :, :-1, :] + fz[:, 1:] - fz[:, :-1])

    def _rhs(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, ok: np.ndarray,
             centred: bool = False) -> np.ndarray:
        """Advection RHS: the fused compiled kernel for WENO5 upwind faces, else the reference path."""
        if centred or not self.tracer_scheme.startswith("weno5"):
            return self._rhs_numpy(c, Qx, Qy, Fz, ok, centred)
        c = np.ascontiguousarray(c, dtype=float)
        T_in = self._inflow_T()
        ext = c if T_in is None or not self.open else np.ascontiguousarray(np.stack([T_in, T_in**2])[: c.shape[0]])
        numba.set_num_threads(_THREADS[0])
        return _advect_div(c, np.ascontiguousarray(Qx, dtype=float), np.ascontiguousarray(Qy, dtype=float),
                           np.ascontiguousarray(Fz, dtype=float), np.ascontiguousarray(np.broadcast_to(ok, c.shape[1:])),
                           ext, "west" in self.open, "east" in self.open, "south" in self.open,
                           "north" in self.open, self.tracer_scheme == "weno5", bool(np.any(Fz != 0.0)))

    def _ext_stack(self, c: np.ndarray) -> np.ndarray:
        """Inflow values of the stack on open edges (T_in, T_in² for the mixing diagnostic), as _rhs."""
        T_in = self._inflow_T()
        return c if T_in is None or not self.open else np.ascontiguousarray(np.stack([T_in, T_in**2])[: c.shape[0]])

    def _faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, ok: np.ndarray, ext: np.ndarray | None = None):
        """WENO5 horizontal face fluxes (fx, fy) of the stack — the terms _advect_div differences (no ω in ALE).
        ``ext``: the inflow values of _ext_stack, when the caller already has them."""
        c = np.ascontiguousarray(c, dtype=float)
        numba.set_num_threads(_THREADS[0])
        return _advect_faces_h(c, np.ascontiguousarray(Qx, dtype=float), np.ascontiguousarray(Qy, dtype=float),
                               np.ascontiguousarray(np.broadcast_to(ok, c.shape[1:])),
                               self._ext_stack(c) if ext is None else ext,
                               "west" in self.open, "east" in self.open, "south" in self.open, "north" in self.open,
                               self.tracer_scheme == "weno5")

    def _upwind_faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray):
        """Donor-cell face fluxes Q c_up (the FCT low-order flux); open edges as _advect_div (ext on inflow)."""
        ext = self._ext_stack(c)
        fx = np.zeros(c.shape[:3] + (c.shape[3] + 1,))
        fy = np.zeros(c.shape[:2] + (c.shape[2] + 1, c.shape[3]))
        fx[..., 1:-1] = Qx[None, ..., 1:-1] * np.where(Qx[None, ..., 1:-1] >= 0, c[..., :-1], c[..., 1:])
        fy[..., 1:-1, :] = Qy[None, :, 1:-1, :] * np.where(Qy[None, :, 1:-1, :] >= 0, c[..., :-1, :], c[..., 1:, :])
        if "west" in self.open:
            fx[..., 0] = Qx[None, ..., 0] * np.where(Qx[None, ..., 0] > 0, ext[..., 0], c[..., 0])
        if "east" in self.open:
            fx[..., -1] = Qx[None, ..., -1] * np.where(Qx[None, ..., -1] < 0, ext[..., -1], c[..., -1])
        if "south" in self.open:
            fy[..., 0, :] = Qy[None, :, 0, :] * np.where(Qy[None, :, 0, :] > 0, ext[..., 0, :], c[..., 0, :])
        if "north" in self.open:
            fy[..., -1, :] = Qy[None, :, -1, :] * np.where(Qy[None, :, -1, :] < 0, ext[..., -1, :], c[..., -1, :])
        return fx, fy

    def _advect_fct(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                    ok: np.ndarray, unlimited: bool = False) -> np.ndarray:
        """
        _advect_fct_numpy with its arithmetic in fused Numba kernels (_fct_stage, _fct_fluxes, _fct_coeff_kernel,
        _fct_final): the NumPy glue cost 23–36 % of the BayIW step (2026-10-14).  ``unlimited``: the NumPy path.
        """
        if unlimited:
            return self._advect_fct_numpy(C, V0, V1, Qx, Qy, ok, unlimited=True)
        dt = float(self.dt)
        C = np.ascontiguousarray(C, dtype=float)
        Qx, Qy = np.ascontiguousarray(Qx, dtype=float), np.ascontiguousarray(Qy, dtype=float)
        V0, V1 = np.ascontiguousarray(V0, dtype=float), np.ascontiguousarray(V1, dtype=float)
        dV = V1 - V0
        Va, Vb = np.ascontiguousarray(V0 + dV), np.ascontiguousarray(V0 + 0.5 * dV)
        okc = np.ascontiguousarray(np.broadcast_to(ok, C.shape[1:]))
        ext = self._ext_stack(C)
        ext_s = None if ext is C else ext                                # no inflow data: each stage's own values
        ext = np.ascontiguousarray(ext, dtype=float)
        q0 = V0 * C
        q1, c1 = np.empty_like(C), np.empty_like(C)
        q2, c2 = np.empty_like(C), np.empty_like(C)
        fx0, fy0 = self._faces_h(C, Qx, Qy, okc, ext)
        numba.set_num_threads(_THREADS[0])
        _fct_stage(q0, q0, fx0, fy0, 0.0, 1.0, dt, Va, C, q1, c1)
        fx1, fy1 = self._faces_h(c1, Qx, Qy, okc, ext_s)
        _fct_stage(q0, q1, fx1, fy1, 0.75, 0.25, dt, Vb, C, q2, c2)
        fx2, fy2 = self._faces_h(c2, Qx, Qy, okc, ext_s)
        opn = self.open
        FLx, FLy, Ax, Ay = _fct_fluxes(C, Qx, Qy, ext, "west" in opn, "east" in opn, "south" in opn, "north" in opn,
                                       fx0, fy0, fx1, fy1, fx2, fy2)
        qL, cL = q1, c1                                                  # stage buffers reused
        _fct_stage(q0, q0, FLx, FLy, 0.0, 1.0, dt, V1, C, qL, cL)
        Cx, Cy = _fct_coeff_kernel(C, cL, V1, Ax, Ay, okc, dt)
        return _fct_final(q0, FLx, FLy, Ax, Ay, Cx, Cy, dt, V1, C)

    def _advect_fct_numpy(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                          ok: np.ndarray, unlimited: bool = False) -> np.ndarray:
        """
        NumPy reference of _advect_fct (test oracle).
        SSP-RK3 WENO5 advection of the stack C with Zalesak (1979) FCT (spec 2026-10-14 §4).  The high-order flux
        of the step is the RK3 combination F_H = (F(c0) + F(c1))/6 + 2F(c2)/3, so V1 c_RK3 = V0 c0 − Δt ∇·F_H;
        the low-order flux is donor cell, V1 c_L = V0 c0 − Δt ∇·F_L, bounded because limit_outflow keeps every
        layer's outflow ≤ its content; the antidiffusive flux A = F_H − F_L is limited by
        fct_coefficients (_fct_coeff_kernel).  ``unlimited``: C = 1 (the RK3 result; test of the identity).
        HYCOM mod_tsadvc.F90:877-905 does the same around its own scheme.
        """
        dt = self.dt
        dV = V1 - V0
        div = lambda fx, fy: fx[..., 1:] - fx[..., :-1] + fy[..., 1:, :] - fy[..., :-1, :]   # noqa: E731
        safe = lambda q, v: np.where(v > 0, q / np.where(v > 0, v, 1.0), C)                  # noqa: E731
        q0 = V0 * C
        fx0, fy0 = self._faces_h(C, Qx, Qy, ok)
        q1 = q0 - dt * div(fx0, fy0)
        c1 = safe(q1, V0 + dV)
        fx1, fy1 = self._faces_h(c1, Qx, Qy, ok)
        q2 = 0.75 * q0 + 0.25 * (q1 - dt * div(fx1, fy1))
        c2 = safe(q2, V0 + 0.5 * dV)
        fx2, fy2 = self._faces_h(c2, Qx, Qy, ok)
        FHx = (fx0 + fx1) / 6.0 + 2.0 * fx2 / 3.0
        FHy = (fy0 + fy1) / 6.0 + 2.0 * fy2 / 3.0
        FLx, FLy = self._upwind_faces_h(C, Qx, Qy)
        Ax, Ay = FHx - FLx, FHy - FLy
        if unlimited:
            return safe(q0 - dt * div(FHx, FHy), V1)
        cL = safe(q0 - dt * div(FLx, FLy), V1)
        okc = np.ascontiguousarray(np.broadcast_to(ok, C.shape[1:]))
        numba.set_num_threads(_THREADS[0])
        Cx, Cy = _fct_coeff_kernel(np.ascontiguousarray(C, dtype=float), np.ascontiguousarray(cL),
                                   np.ascontiguousarray(V1, dtype=float), np.ascontiguousarray(Ax),
                                   np.ascontiguousarray(Ay), okc, float(dt))
        return safe(q0 - dt * div(FLx + Cx * Ax, FLy + Cy * Ay), V1)

    def _advect_numpy(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                Fz: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """
        NumPy reference of _advect (test oracle).
        SSP-RK3 flux-form advection of the tracer stack C (ntr,N,ny,nx) with frozen
        transports Qx, Qy (layer volume fluxes at faces) and Fz (N+1,ny,nx, through
        the interfaces); layer volumes go linearly from V0 to V1.
        """
        dt = self.dt
        rhs = lambda c, centred=False: self._rhs(c, Qx, Qy, Fz, ok, centred)          # noqa: E731

        dV = V1 - V0
        safe = lambda q, v: np.where(v > 0, q / np.where(v > 0, v, 1.0), C)     # noqa: E731
        q0 = V0 * C
        if self.tracer_time == "pc":
            # predictor: half step with centred 6th-order faces; corrector: full step from n with the
            # upwind (WENO) faces of the mid-step tracer — upwinding applied once (CROCO step3d_t.F)
            Vh = V0 + 0.5 * dV
            ch = safe(q0 + 0.5 * dt * rhs(C, centred=True), Vh)
            return safe(q0 + dt * rhs(ch), V1)
        q1 = q0 + dt * rhs(C)
        Va = V0 + dV
        c1 = safe(q1, Va)
        q2 = 0.75 * q0 + 0.25 * (q1 + dt * rhs(c1))
        Vb = V0 + 0.5 * dV
        c2 = safe(q2, Vb)
        q3 = q0 / 3.0 + 2.0 / 3.0 * (q2 + dt * rhs(c2))
        return safe(q3, V1)

    def _advect(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                Fz: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """
        SSP-RK3 flux-form advection of the tracer stack C (ntr,N,ny,nx) with frozen transports Qx, Qy and Fz;
        layer volumes go linearly from V0 to V1.  The stage combinations and the ``safe`` division run in the
        fused kernel _rk_stage with _advect_numpy's arithmetic (the reference); 'pc' keeps the NumPy path.
        """
        if self.tracer_time == "pc":
            return self._advect_numpy(C, V0, V1, Qx, Qy, Fz, ok)
        dt = self.dt
        C = np.ascontiguousarray(C, dtype=float)
        if np.shape(V0) != C.shape[1:] or np.shape(V1) != C.shape[1:]:     # the kernel does not check bounds
            raise ValueError(f"_advect: volume shape {np.shape(V0)}/{np.shape(V1)} vs tracer layers {C.shape[1:]}")
        dV = V1 - V0
        Va, Vb = np.ascontiguousarray(V0 + dV), np.ascontiguousarray(V0 + 0.5 * dV)
        q0 = V0 * C
        numba.set_num_threads(_THREADS[0])
        f = lambda a: np.ascontiguousarray(a, dtype=float).ravel()            # noqa: E731  1-D views
        q1, c1 = np.empty_like(C), np.empty_like(C)
        _rk_stage(f(q0), f(q0), f(self._rhs(C, Qx, Qy, Fz, ok)), dt, f(Va), f(C), 1, q1.ravel(), c1.ravel())
        q2, c2 = np.empty_like(C), np.empty_like(C)
        _rk_stage(f(q0), f(q1), f(self._rhs(c1, Qx, Qy, Fz, ok)), dt, f(Vb), f(C), 2, q2.ravel(), c2.ravel())
        q3, c3 = np.empty_like(C), np.empty_like(C)
        _rk_stage(f(q0), f(q2), f(self._rhs(c2, Qx, Qy, Fz, ok)), dt, f(V1), f(C), 3, q3.ravel(), c3.ravel())
        return c3

    def _step_ale(self, st: SigmaState, gu_ab: np.ndarray, gv_ab: np.ndarray, tu, tv, hist: list) -> SigmaState:
        """
        ALE step (spec §2): (1) Lagrangian step with no flux through the layer interfaces —
        free surface with the depth mean ū = Σ w_k u_k on the mean face thicknesses of hⁿ, donor-cell
        layer transports of u_k^θ = θ u_k^{n+1} + (1−θ) u_k^n corrected to the free-surface flux
        (core.vertical.ale.upwind_transport_x), layer continuity h† = hⁿ − Δt ∇·Q_k / A, flux-form
        tracer advection with Fz = 0; (2) vertical physics on h†; (3) regrid with the rule; (4)
        conservative remap of T (and T² for the mixing diagnostic) and of u, v in face-fraction
        space.  Velocity is zero in bed layers thin on both sides of a face (HYCOM hybgen.F90:1757-1764).
        Open boundaries (spec 2026-10-12-ale-open-boundaries): Flather in the free surface, sponge on u′ and
        Orlanski with the face fractions, open-face fluxes on the interior thickness, inflow T and tracer
        sponge from the exterior z profile remapped to hⁿ and h†.
        """
        dt, th, n = self.dt, self.fs.theta, self.n
        numba.set_num_threads(_THREADS[0])                                 # parallel ALE kernels (≤ 5 P-cores)
        act = self.fs.active
        (qe, qw), (qn, qs) = self._transport_thickness(st.h)
        hu, hv = 0.5 * (qe + qw), (0.5 * (qn + qs)).swapaxes(1, 2)
        wu, wv = face_fractions(hu), face_fractions(hv)
        self._evaluate_exterior(st.time + dt, hu, hv, wu, wv)
        if self.thin_layer_coupling == "mom6":                         # viscous tie of vanished layers (spec §3)
            hu_v = viscous_face_thickness(qe, qw, st.u)                 # upwind-biased (MOM6, Hallberg & Rhines)
            hv_v = viscous_face_thickness(qn, qs, st.v.swapaxes(1, 2)).swapaxes(1, 2)
            visc = dict(hu=np.maximum(hu_v, DZ_FLOOR), hv=np.maximum(hv_v, DZ_FLOOR),
                        hu_drag=np.maximum(hu[0], DZ_FLOOR), hv_drag=np.maximum(hv[0], DZ_FLOOR),
                        hu_mean=hu, hv_mean=hv)                         # ū on the clipped face column
        else:
            visc = dict(hu=np.maximum(hu, DZ_FLOOR), hv=np.maximum(hv, DZ_FLOOR))
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv,
                                               wu=wu, wv=wv, **visc, **self._flather_args())
        self._open_velocity(st, u_new, v_new, wu, wv)                   # sponge on u′, Orlanski (spec §3.2-3.3)
        u_new = zero_thin_bed_velocity(u_new, hu, wu, self.h_thin_bed)
        v_new = zero_thin_bed_velocity(v_new, hv, wv, self.h_thin_bed)
        if self.thin_layer_cutoff > 0.0:                                # HYCOM blending of thin layers (spec §3, 1B)
            # on the upwind (donor) thickness, not the mean (Hallberg & Rhines 1996): a nose advancing from a thick
            # cell keeps its speed, a layer draining from where it has vanished moves with the layer above
            u_up = np.where(u_new >= 0, qe, qw)
            v_up = np.where(v_new.swapaxes(1, 2) >= 0, qn, qs).swapaxes(1, 2)
            u_new = blend_thin_layers(u_new, u_up, wu, self.thin_layer_cutoff)
            v_new = blend_thin_layers(v_new, v_up, wv, self.thin_layer_cutoff)
        # transport velocities and free-surface volume fluxes (core.vertical.ale.transport_velocity)
        tb = self.theta_bc
        utx, Qbx = transport_velocity(u_new, st.u, self.fs.last_wet_u, wu, self.fs.Lu, self.fs.last_Du, th, tb)
        uty, Qby = transport_velocity(v_new, st.v, self.fs.last_wet_v, wv, self.fs.Lv, self.fs.last_Dv, th, tb)
        if self.thickness_flux == "ppm":
            dx_t, dy_t = self._vi["dx_t"], self._vi["dy_t"]
            Qx = ppm_transport_x(st.h, utx, self.fs.Lu, Qbx, qe, qw, act, dt, dx_t)
            Qy = ppm_transport_x(st.h.swapaxes(1, 2), uty.swapaxes(1, 2), self.fs.Lv.T, Qby.T, qn, qs, act.T,
                                 dt, dy_t.T).swapaxes(1, 2)
        else:
            Qx = upwind_transport_x(qe, qw, utx, self.fs.Lu, Qbx)
            Qy = upwind_transport_x(qn, qs, uty.swapaxes(1, 2), self.fs.Lv.T, Qby.T).swapaxes(1, 2)
        if self.open:                                                   # old velocity unmasked: the wet flag is False at edges
            self._open_face_transports(Qx, Qy, st.h, st.u, st.v, u_new, v_new)
        # outflow limiter: no layer loses more than it holds; Σ_k Q_k unchanged (core.vertical.ale.limit_outflow)
        Qx, Qy, n_lim = limit_outflow(st.h, Qx, Qy, self.area, dt, act)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        h_lag = lagrangian_thickness(st.h, divh, dt, self.area)
        V0, V1 = self.layer_volumes(st.eta, st.h), self.layer_volumes(eta_new, h_lag)
        Fz = np.zeros((n + 1,) + self.H.shape)                              # Lagrangian: no flux through interfaces
        ok = np.broadcast_to(act[None], V1.shape) & (V1 > 0)
        self._T_in = self._exterior_T(st.h) if self.open else None        # inflow T on hⁿ (spec §3.5)
        mixd = self.mix_diagnostic
        stack = np.stack([st.T, st.T**2]) if mixd else st.T[None]
        C = (self._advect_fct(stack, V0, V1, Qx, Qy, ok) if self.tracer_limiter == "fct"
             else self._advect(stack, V0, V1, Qx, Qy, Fz, ok))
        mix = V1 * (C[1] - C[0] ** 2) / dt if mixd else None                # advection part
        T_lag = C[0]
        if self.kh > 0:
            T_lag = self._diffuse_h(T_lag, V1, eta_new, h_lag)
        if self.kv > 0:
            dz = np.maximum(h_lag, DZ_FLOOR).reshape(n, -1)
            T_lag = vertical_implicit(T_lag.reshape(n, -1), dz, np.full((n + 1, dz.shape[1]), self.kv),
                                      np.zeros(dz.shape[1]), dt).reshape(T_lag.shape)
        if self._sp_c is not None and self.sponge_tracer:                 # tracer sponge toward the exterior on h†
            r = dt * self._sp_c[None]
            T_lag = (T_lag + r * self._exterior_T(h_lag)) / (1.0 + r)
        h_new, rem = self._regrid_remap(eta_new, h_lag, [T_lag, T_lag**2] if mixd else [T_lag], T_lag)
        T_new = rem[0]
        if mixd:
            mix = mix + self.layer_volumes(eta_new, h_new) * (rem[1] - T_new**2) / dt   # remap part
        hu_lag, hv_lag = self._mean_face_thickness(h_lag)
        hu1, hv1 = self._mean_face_thickness(h_new)
        wu1, wv1 = face_fractions(hu1), face_fractions(hv1)                # once: remap target and thin-bed rule
        u_new = zero_thin_bed_velocity(self._remap_faces(u_new, hu_lag, hu1, wu1), hu1, wu1, self.h_thin_bed)
        v_new = zero_thin_bed_velocity(self._remap_faces(v_new, hv_lag, hv1, wv1), hv1, wv1, self.h_thin_bed)
        g = self._vi                                                    # CFL truncation backstop (MOM6 vertvisc_limit_vel)
        u_before, v_before = u_new, v_new
        u_new, nu_, fu = vi.truncate_cfl(u_new, dt, g["dx_t"], hu1, self.h_thin_bed)
        v_sw, nv_, fv = vi.truncate_cfl(v_new.swapaxes(1, 2), dt, g["dy_t"].T, hv1.swapaxes(1, 2), self.h_thin_bed)
        v_new = v_sw.swapaxes(1, 2)
        n_trunc = st.n_trunc + nu_ + nv_
        if n_trunc > self.max_truncations:
            if fu is not None:
                k, j, i = fu
                where = (f"layer {k} at u-face (j={j}, i={i}): u = {u_before[k, j, i]:.3g} m/s, "
                         f"face thickness {hu1[k, j, i]:.3g} m")
            else:
                k, i, j = fv
                where = (f"layer {k} at v-face (j={j}, i={i}): v = {v_before[k, j, i]:.3g} m/s, "
                         f"face thickness {hv1[k, j, i]:.3g} m")
            raise RuntimeError(f"{n_trunc} CFL velocity truncations exceed max_truncations={self.max_truncations}; "
                               f"first this step: {where}")
        return SigmaState(eta=eta_new, u=u_new, v=v_new, T=T_new, h=h_new, g_hist=hist, omega=None, mix_num=mix,
                          Qx=Qx, Qy=Qy, Fz=Fz, time=st.time + dt, n_trunc=n_trunc, n_limited=st.n_limited + n_lim)

    # ------------------------------------------------------------------ step
    def step(self, st: SigmaState, tau_x: np.ndarray | None = None, tau_y: np.ndarray | None = None) -> SigmaState:
        """
        Advance one step.  tau_x (ny, nx+1), tau_y (ny+1, nx): surface (wind) stress at the u- and
        v-faces [N m⁻²]; τ/ρ₀ enters as the top boundary flux of the implicit vertical viscosity
        (as ROMS/CROCO sustr/svstr), not through the AB3 extrapolation.
        """
        dt, th, n = self.dt, self.fs.theta, self.n
        gu, gv = self._tendencies(st)
        hist = [(gu, gv)] + st.g_hist[:2]
        if len(hist) == 1:
            gu_ab, gv_ab = gu, gv
        elif len(hist) == 2:
            gu_ab, gv_ab = ab3_combine(gu, hist[1][0], None), ab3_combine(gv, hist[1][1], None)
        else:
            gu_ab = ab3_combine(gu, hist[1][0], hist[2][0])
            gv_ab = ab3_combine(gv, hist[1][1], hist[2][1])
        tu = None if tau_x is None else np.asarray(tau_x, float) / self.rho0
        tv = None if tau_y is None else np.asarray(tau_y, float) / self.rho0
        if self.ale:
            return self._step_ale(st, gu_ab, gv_ab, tu, tv, hist)
        if self.obc_state is not None:
            ds = self.ds[:, None, None]
            Hu_e, Hv_e = self._edge_face_depths()
            self._evaluate_exterior(st.time + dt, Hu_e[None] * ds, Hv_e[None] * ds, None, None)
            D0 = np.maximum(np.nan_to_num(self.H + st.eta, nan=self.fs.d_min), self.fs.d_min)
            self._T_in = self._exterior_T(D0[None] * ds)
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv,
                                               **self._flather_args())
        self._open_velocity(st, u_new, v_new)

        # layer transports consistent with the free-surface flux (fs.last_Du / last_Dv: upwind face depth)
        un = np.where(self.fs.last_wet_u[None], st.u, 0.0)
        vn = np.where(self.fs.last_wet_v[None], st.v, 0.0)
        # depth mean with the free-surface θ, baroclinic part with θ_bc (see transport_velocity)
        tb, ds = self.theta_bc, self.ds[:, None, None]
        ux = tb * u_new + (1 - tb) * un + ((th - tb) * ((ds * u_new).sum(0) - (ds * un).sum(0)))[None]
        vy = tb * v_new + (1 - tb) * vn + ((th - tb) * ((ds * v_new).sum(0) - (ds * vn).sum(0)))[None]
        Qx = (self.fs.Lu * self.fs.last_Du)[None] * ds * ux
        Qy = (self.fs.Lv * self.fs.last_Dv)[None] * ds * vy
        if self.open:
            # open faces: Δσ_k × the free-surface (Flather) flux + the baroclinic part, so Σ_k Q_k is exactly
            # the volume flux of the continuity equation
            for Q, D, Qo, L, vo, vn1 in ((Qx, self.fs.last_open_Du, self.fs.last_open_Qu, self.fs.Lu, st.u, u_new),
                                         (Qy, self.fs.last_open_Dv, self.fs.last_open_Qv, self.fs.Lv, st.v, v_new)):
                po, pn = self._baroclinic_part(vo)[1], self._baroclinic_part(vn1)[1]
                ds = self.ds[:, None, None]
                Q[...] = np.where((D > 0)[None], ds * Qo[None] + (L * D)[None] * ds * (tb * pn + (1 - tb) * po), Q)
        V0, V1 = self.layer_volumes(st.eta), self.layer_volumes(eta_new)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        Fz = np.zeros((n + 1,) + self.H.shape)                              # volume flux through interface k (k=0 bed)
        Fz[1:] = -np.cumsum((V1 - V0) / dt + divh, axis=0)
        Fz[-1] = 0.0                                                        # surface: round-off residual only
        ok = (np.broadcast_to(self.fs.active[None], V1.shape)) & (V1 > 0)
        C = self._advect(np.stack([st.T, st.T**2]), V0, V1, Qx, Qy, Fz, ok)
        T_new = C[0]
        mix = V1 * (C[1] - C[0] ** 2) / dt
        if self.kh > 0:
            T_new = self._diffuse_h(T_new, V1, eta_new)
        if self.kv > 0:
            D1 = np.maximum(np.nan_to_num(self.H + eta_new, nan=self.fs.d_min), self.fs.d_min)
            dz = (D1[None] * self.ds[:, None, None]).reshape(n, -1)
            T_new = vertical_implicit(T_new.reshape(n, -1), dz, np.full((n + 1, dz.shape[1]), self.kv),
                                      np.zeros(dz.shape[1]), dt).reshape(T_new.shape)
        if self._sp_c is not None and self.sponge_tracer:
            r = dt * self._sp_c[None]
            T_new = (T_new + r * (self.T_ext if self._ext is None else self._T_in)) / (1.0 + r)
        omega = Fz / np.where(self.area > 0, self.area, 1.0)[None]
        return SigmaState(eta=eta_new, u=u_new, v=v_new, T=T_new, g_hist=hist, omega=omega, mix_num=mix,
                          Qx=Qx, Qy=Qy, Fz=Fz, time=st.time + dt)
