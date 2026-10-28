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
from core.components.eos import LinearEOS
from core.components.geometry import LayerGeometry
from core.components.pgf import PressureGradient
from core.components.momentum import Momentum, sum_terms
from core.components.open_boundary import CONFIG as _OB_CONFIG, OpenBoundary
from core.components.bottom_drag import QuadraticBedDrag
from core.components.free_surface import FreeSurfaceVerticalViscosity
from core.components.layer_transport import ALELayerTransport, SigmaLayerTransport
from core.components.tracers import TracerAdvection, TracerDiffusion
from core.components.sponge import InteriorSponge
from core.components.vertical_coordinate import ALECoordinate
from core.model.driver import Driver
from core.model.contract import raise_issues
from core.model.preflight import preflight
from core.components.time_stepper import AB3, AB3Momentum, _ab3_kernel, ab3_combine  # noqa: F401  (re-exported)
from core.model.threads import _THREADS, set_threads  # noqa: F401  (one shared list; re-exported)

G = 9.81
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
    tr: np.ndarray | None = None                             # passive tracers (ntr, N, ny, nx), ALE only
    closure: dict | None = None                              # turbulence-closure state (GLS: tke, eps, akv), N1


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
          production; Esteban 2026-10-12: Godunov damps progressive internal waves ~9 % per 5 km at dx = 100 m,
          validation/internal_wave/README.md) or 'godunov' (upwind: only the velocities entering a cell; MOM6
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
    n_tracers : ALE: number of passive tracers carried with T in the same advection stack (WENO5 + FCT with their own
        3-D bounds), remap, diffusion, open boundaries and sponges; no effect on density (spec 2026-10-14-dome §1)
    sponge_rate, sponge_T, sponge_zw, sponge_tr : ALE interior sponge (spec 2026-10-14-dome §2), independent of the open
        boundaries: after the regrid–remap of each step T ← (T + Δt r T*)/(1 + Δt r), implicit as MOM6 apply_sponge,
        with T* the profile sponge_T on the interfaces sponge_zw (surface first, [m] ≤ 0) clipped at each bed,
        stretched to the column and remapped onto the current layers; passive tracers toward sponge_tr (default 0);
        r = sponge_rate (ny, nx) [s⁻¹], ignored on land; momentum is not damped (as MOM6 DOME_initialize_sponges).
        It acts after the remap, so the mixing diagnostic counts advection and remap only
    bottom_drag, cd, drag_bg_vel, hbbl : ALE bed stress (spec 2026-10-14-dome §3): None (default, free slip) or
        'quadratic': τ_b/ρ₀ = −r u_b, r = C_d √(ū_bbl² + v̄_bbl² + u_bg²), with ū_bbl the thickness-weighted mean of the
        face column over the bottom ``hbbl`` metres (MOM6 HBBL, MOM_set_viscosity.F90) — a vanished bed layer does
        not set the drag alone; applied implicitly at the bed of the coupled viscous solve (the depth-mean correction
        δ of SigmaWetDry.column_step includes it).  Legg et al. 2006: C_d = 2e-3
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
                 tracer_limiter: str | None = None, n_tracers: int = 0, sponge_rate: np.ndarray | None = None,
                 sponge_T: np.ndarray | None = None, sponge_zw: np.ndarray | None = None,
                 sponge_tr: np.ndarray | None = None, bottom_drag: str | None = None, cd: float = 2.5e-3,
                 drag_bg_vel: float = 0.0, hbbl: float = 10.0) -> None:
        _args = {k: v for k, v in locals().items() if k != "self"}
        raise_issues(preflight(_args, _given(_args)))                  # N1 pre-flight: every error at once
        if ke_scheme == "godunov":                                      # correct spelling; the kernels use 'gudonov'
            ke_scheme = "gudonov"
        self._ob = OpenBoundary()                                       # N1: open-boundary component
        if pgf not in ("twoterm", "jacobian", "fv"):
            raise ValueError(f"unknown pressure-gradient scheme {pgf!r}")
        self.pgf = pgf
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
        self._eos = LinearEOS(rho0=rho0, t_coef=t_coef, t_ref=t_ref)
        self._time_stepper = AB3Momentum()
        self._pgf = PressureGradient(pgf, self._eos, g, rho_ref)
        self.fs = self.core.fs
        self._free_surface = FreeSurfaceVerticalViscosity(self.core)
        self._fluxes = None                                             # BarotropicFluxes of the last step
        self._fq = np.asarray(h_grid["f_q"].values, float)           # Coriolis at cell corners [s⁻¹]
        self.H = self.fs.H
        self.ds = self.core.dsigma
        self.sigma_c = 0.5 * (self.core.sigma_w[1:, 0] + self.core.sigma_w[:-1, 0])   # layer centres (bed = 0)
        self.area = self.fs.area
        self._geometry = LayerGeometry(self.H, self.ds, self.sigma_c, self.fs.d_min, self.fs.H_u, self.fs.H_v,
                                       self.fs.du, self.fs.dv, self.layer_heights)
        self._setup_ale(vertical_coordinate, regrid, regrid_params, remap_scheme, z0, closure, h_thin_bed,
                        init_iterations, act, momentum, ke_scheme, max_truncations, thickness_flux)
        self._vertical = (ALECoordinate(H=self.H, active=self.fs.active, n_layers=n_layers, rule=self.regrid_rule,
                                        params=self.regrid_params, remap_code=self.remap_code,
                                        h_thin_bed=self.h_thin_bed, max_truncations=self.max_truncations,
                                        vi_geom=self._vi, eos=self._eos) if self.ale else None)
        self._momentum = Momentum(form="vector_invariant" if self.ale and self.momentum == "vector_invariant" else "flux",
                                  adv=adv, ah=ah, ke_scheme=self.ke_scheme, ale=self.ale, dt=dt, hg=h_grid, fq=self._fq,
                                  fs=self.fs, ds=self.ds, vi_geom=getattr(self, "_vi", None),
                                  mean_face_thickness=self._mean_face_thickness, layer_volumes=self.layer_volumes)
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
        self.n_tracers = int(n_tracers)
        if self.n_tracers < 0 or (self.n_tracers and not self.ale):
            raise ValueError(f"n_tracers={n_tracers!r}: passive tracers need vertical_coordinate='ale' and n ≥ 0")
        self.tr_ext, self._tr_in = None, None
        self._ob.bind(dt=dt, ds=self.ds, fs=self.fs, H=self.H, n_tracers=self.n_tracers, ale=self.ale,
                      remap_code=getattr(self, "remap_code", rm.PPM_H4))
        if self.ale:
            self._layer_transport = ALELayerTransport(fs=self.fs, H=self.H, area=self.area, dt=dt,
                                                      thickness_flux=self.thickness_flux, vi_geom=self._vi,
                                                      layer_volumes=self.layer_volumes, ob=self._ob)
        else:
            self._layer_transport = SigmaLayerTransport(fs=self.fs, ds=self.ds, H=self.H, dt=dt,
                                                        layer_volumes=self.layer_volumes, ob=self._ob)
        self._last_layer_fluxes = None
        self._driver = Driver(self)
        self._tracer_advection = TracerAdvection(scheme=self.tracer_scheme, time=self.tracer_time, dt=dt, ob=self._ob)
        self._tracer_diffusion = TracerDiffusion(kh=self.kh, dt=dt, ds=self.ds, fs=self.fs, eos=self._eos, pgf=self._pgf,
                                                 face_depths=self.core.face_depths, layer_heights=self.layer_heights)
        self.sponge_rate = None
        if sponge_rate is not None:
            if not self.ale:
                raise ValueError("sponge_rate (interior sponge) needs vertical_coordinate='ale'")
            r = np.asarray(sponge_rate, float)
            if r.shape != self.H.shape:
                raise ValueError(f"sponge_rate shape {r.shape}, expected {self.H.shape}")
            zw, Ts = np.asarray(sponge_zw, float), np.asarray(sponge_T, float)
            if zw.ndim != 1 or Ts.shape != (zw.size - 1,) or not (np.diff(zw) < 0).all():
                raise ValueError("sponge_T (nz_s,) needs sponge_zw (nz_s + 1,) decreasing from the surface")
            self.sponge_rate = np.where(self.fs.active, np.nan_to_num(r), 0.0)
            self._sp_zw, self._sp_T = zw, Ts
            self._sp_tr = (np.zeros((self.n_tracers, Ts.size)) if sponge_tr is None
                           else np.asarray(sponge_tr, float).reshape(self.n_tracers, Ts.size))
            self._sp_cols = self.sponge_rate > 0
        self._interior_sponge_c = (None if self.sponge_rate is None else
                                   InteriorSponge(rate=self.sponge_rate, T=self._sp_T, zw=self._sp_zw, tr=self._sp_tr,
                                                  cols=self._sp_cols, H=self.H, dt=dt, remap_code=self.remap_code))
        if bottom_drag not in (None, "quadratic"):
            raise ValueError(f"unknown bottom_drag {bottom_drag!r}; use None or 'quadratic'")
        if bottom_drag and not self.ale:
            raise ValueError("bottom_drag='quadratic' needs vertical_coordinate='ale' (σ uses z0, the log-law)")
        self.bottom_drag, self.cd, self.drag_bg_vel, self.hbbl = bottom_drag, float(cd), float(drag_bg_vel), float(hbbl)
        self._bed_drag = QuadraticBedDrag(cd, drag_bg_vel, hbbl)

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
    def _sync_closure_in(self, st: SigmaState) -> None:
        """Load the closure state carried by ``st`` into the GLS object (by reference, no copies)."""
        self._free_surface.sync_closure_in(st.closure)

    def _sync_closure_out(self) -> dict | None:
        """The closure state to carry to the next step (references to the GLS arrays), or None."""
        return self._free_surface.closure_out()

    def initial_state(self, eta: np.ndarray, T: np.ndarray, h_src: np.ndarray | None = None,
                      tr: np.ndarray | None = None) -> SigmaState:
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
        ``tr``: passive tracers (n_tracers, ...) on the grid of T (default 0; ALE ``n_tracers``).
        """
        ny, nx = self.H.shape
        eta = np.asarray(eta, float)
        if h_src is not None:
            return self._initial_state_from_source(eta, np.asarray(T, float), np.asarray(h_src, float), tr)
        tr = self._check_tracers(tr, np.shape(T))
        if isinstance(self.rho_ref, str) and self.rho_ref == "auto":
            z = self.layer_heights(eta)
            act = np.broadcast_to(self.fs.active[None], z.shape)
            self.rho_ref = reference_profile(z[act], self.density(np.asarray(T, float))[act])
        self.T_ext = np.array(T, float)                                # inflow / sponge target at open boundaries
        st = SigmaState(eta=np.array(eta, float), u=np.zeros((self.n, ny, nx + 1)),
                        v=np.zeros((self.n, ny + 1, nx)), T=np.array(T, float),
                        h=self._initial_h(eta) if self.ale else None, tr=tr, closure=self._sync_closure_out())
        if self.ale and self.regrid_rule == "hybrid":
            for _ in range(self.init_iterations):
                fields = [st.T] + ([] if st.tr is None else list(st.tr))
                st.h, rem = self._regrid_remap(st.eta, st.h, fields, st.T)
                st.T = rem[0]
                st.tr = None if st.tr is None else np.stack(rem[1:])
        if self.ale:                                                    # exterior profile in z (spec §3.5)
            self.T_ext, self.h_ext = st.T.copy(), st.h.copy()
            self.tr_ext = None if st.tr is None else st.tr.copy()
        return st

    def _check_tracers(self, tr, shape: tuple) -> np.ndarray | None:
        """Passive tracers (n_tracers,) + shape (zeros when None), or None without tracers; ValueError otherwise."""
        if self.n_tracers == 0:
            if tr is not None:
                raise ValueError("tr given but the model has n_tracers=0")
            return None
        if tr is None:
            return np.zeros((self.n_tracers,) + tuple(shape))
        tr = np.array(tr, float)
        if tr.shape != (self.n_tracers,) + tuple(shape):
            raise ValueError(f"tr shape {tr.shape}, expected {(self.n_tracers,) + tuple(shape)}")
        return tr

    def _initial_state_from_source(self, eta: np.ndarray, T_src: np.ndarray, h_src: np.ndarray,
                                   tr_src: np.ndarray | None = None) -> SigmaState:
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
        tr_src = self._check_tracers(tr_src, T_src.shape)               # passive tracers: the same remap, 0 where dry
        tr = None if tr_src is None else np.stack(
            [np.where(wet[None], rm.remap_columns(h_src, f, h, self.remap_code, wet), 0.0) for f in tr_src])
        st = SigmaState(eta=np.array(eta, float), u=np.zeros((self.n, ny, nx + 1)),
                        v=np.zeros((self.n, ny + 1, nx)), T=T, h=h, tr=tr, closure=self._sync_closure_out())
        self.T_ext, self.h_ext = T.copy(), h.copy()                    # exterior profile in z (spec §3.5)
        self.tr_ext = None if tr is None else tr.copy()
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
        """Linear EOS ρ = ρ₀ − α (T − T₀) [kg m⁻³] (``core.components.eos``)."""
        return self._eos.density(T)

    def layer_volumes(self, eta: np.ndarray, h: np.ndarray | None = None) -> np.ndarray:
        """Layer volumes [m³]: D Δσ_k A, or h_k A (ALE)."""
        if h is not None:
            return h * self.area[None]
        D = np.maximum(np.nan_to_num(self.H + eta, nan=0.0), 0.0)
        return D[None] * self.ds[:, None, None] * self.area[None]

    def tracer_content(self, st: SigmaState, power: int = 1, tracer: int | None = None) -> float:
        """Σ V c^power [unit^power m³] of T (``tracer`` None) or of passive tracer ``tracer``."""
        V = self.layer_volumes(st.eta, st.h)
        c = st.T if tracer is None else st.tr[tracer]
        return float(np.sum(V * c**power))

    def rpe(self, st: SigmaState) -> float:
        """Reference potential energy [J] (flat-bottom basin: area = wet area)."""
        V = self.layer_volumes(st.eta, st.h)
        act = np.broadcast_to(self.fs.active[None], V.shape)
        return reference_potential_energy(self.density(st.T)[act], V[act], area=float(self.area[self.fs.active].sum()))

    # ------------------------------------------------------------------ ALE helpers
    def _initial_h(self, eta: np.ndarray) -> np.ndarray:
        """ALE layer thicknesses of the initial grid at η [m] (``core.components.vertical_coordinate``)."""
        return self._vertical.initial_h(eta)

    def _regrid_remap(self, eta: np.ndarray, h_src: np.ndarray, fields: list, T: np.ndarray):
        """Regrid with the rule and remap ``fields``: (h_new, remapped) (``core.components.vertical_coordinate``)."""
        return self._vertical.regrid_remap(eta, h_src, fields, T)

    def _regrid(self, eta: np.ndarray, h: np.ndarray, T: np.ndarray, wet: np.ndarray | None = None) -> np.ndarray:
        """New layer thicknesses of the rule (``core.components.vertical_coordinate``)."""
        return self._vertical.regrid(eta, h, T, wet)

    def _transport_thickness(self, h: np.ndarray):
        """Upwind face thicknesses, cached by identity of h (``core.components.vertical_coordinate``)."""
        return self._vertical.transport_thickness(h)

    def _mean_face_thickness(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Direction-independent face thickness (q_E + q_W)/2 at u- and v-faces [m]."""
        return self._vertical.mean_face_thickness(h)

    def face_thickness(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """ALE layer thickness at u- and v-faces [m]: the mean of the two upwind (bed-clipped) thicknesses."""
        return self._vertical.mean_face_thickness(h)

    def _remap_faces(self, vel: np.ndarray, hf_src: np.ndarray, hf_tgt: np.ndarray,
                     w_tgt: np.ndarray | None = None) -> np.ndarray:
        """Remap face velocities in fraction space (``core.components.vertical_coordinate``)."""
        return self._vertical.remap_faces(vel, hf_src, hf_tgt, w_tgt)

    # ------------------------------------------------------------------ explicit tendencies
    def _baroclinic_pgf(self, eta: np.ndarray, T: np.ndarray, h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Baroclinic pressure-gradient force at u- and v-faces [m s⁻²] (``core.components.pgf``)."""
        return self._pgf(self._geometry, eta, T, h)

    def _fv_pgf(self, eta: np.ndarray, T: np.ndarray, h: np.ndarray | None) -> tuple[np.ndarray, np.ndarray]:
        """Finite-volume PGF (``core.components.pgf``), whatever ``pgf`` is set to."""
        return self._pgf._fv(self._geometry, eta, T, h)

    def pressure_gradient(self, eta: np.ndarray, T: np.ndarray,
                          h: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Baroclinic pressure-gradient force at u- and v-faces [m s⁻²] of the configured scheme."""
        return self._pgf(self._geometry, eta, T, h)

    @property
    def rho_ref(self):
        """Reference density profile of the PGF: None, "auto" (until ``initial_state``) or z → ρ_ref(z)."""
        return self._pgf.rho_ref

    @rho_ref.setter
    def rho_ref(self, value) -> None:
        self._pgf.rho_ref = value

    def _coriolis(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Energy-conserving C-grid Coriolis terms (``core.components.momentum``)."""
        return self._momentum.coriolis(u, v)

    def _vector_invariant(self, st: SigmaState) -> tuple[np.ndarray, np.ndarray]:
        """Vector-invariant momentum terms of the ALE step (``core.components.momentum``)."""
        return self._momentum.vector_invariant(st)

    def _vertical_advection(self, vel: np.ndarray, omega_f: np.ndarray, h_f: np.ndarray) -> np.ndarray:
        """(ω/h) ∂u/∂k, centred (``core.components.momentum``)."""
        return self._momentum.vertical_advection(vel, omega_f, h_f)

    def _momentum_terms(self, st: SigmaState) -> tuple[list, np.ndarray, np.ndarray]:
        """Named explicit momentum terms of ``st`` in summation order, and the wet u- and v-faces."""
        Du, Dv = self.core.face_depths(st.eta)
        wet_u = np.isfinite(self.fs.H_u) & np.isfinite(Du) & (Du > self.fs.d_min)
        wet_v = np.isfinite(self.fs.H_v) & np.isfinite(Dv) & (Dv > self.fs.d_min)
        return self._momentum.terms(st, self._baroclinic_pgf(st.eta, st.T, st.h), Du, Dv, wet_u, wet_v), wet_u, wet_v

    def _tendencies(self, st: SigmaState) -> tuple[np.ndarray, np.ndarray]:
        """Explicit momentum tendencies at u- and v-faces [m s⁻²]: Σ of the named terms (fixed order)."""
        terms, wet_u, wet_v = self._momentum_terms(st)
        return sum_terms(terms, wet_u, wet_v)

    # ------------------------------------------------------------------ tracers
    def _faces(self, c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int, centred: bool = False) -> np.ndarray:
        return self._tracer_advection.faces(c, flux, ok, axis, centred)

    def _inflow(self, c: np.ndarray) -> np.ndarray | None:
        """This step's inflow stack for the advected stack ``c`` (None: no open edges or no exterior)."""
        ext = self._ext_stack(c)
        return None if ext is c else ext

    def _open_faces(self, face: np.ndarray, c: np.ndarray, Q: np.ndarray, axis: int) -> np.ndarray:
        return self._tracer_advection.open_faces(face, c, Q, axis, self._inflow(c))

    def _diffuse_h(self, T: np.ndarray, V: np.ndarray, eta: np.ndarray, h: np.ndarray | None = None,
                   anomaly: bool = True) -> np.ndarray:
        """Explicit Laplacian diffusion along the layers (``core.components.tracers``)."""
        return self._tracer_diffusion.diffuse_h(T, V, eta, h, anomaly)

    def _diffuse_h_numpy(self, T: np.ndarray, V: np.ndarray, eta: np.ndarray, h: np.ndarray | None = None,
                         anomaly: bool = True) -> np.ndarray:
        """NumPy reference of _diffuse_h (``core.components.tracers``)."""
        return self._tracer_diffusion.diffuse_h_numpy(T, V, eta, h, anomaly)

    def _baroclinic_part(self, vel: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Depth mean and baroclinic remainder (``core.components.open_boundary``)."""
        return self._ob.baroclinic_part(vel, w)

    def _radiate(self, u_old: np.ndarray, u_new: np.ndarray, axis: int, edge: int,
                 w: np.ndarray | None = None, ext: np.ndarray | None = None, mode: str = "radiating") -> None:
        """Orlanski radiation of u′ at one open edge, in place (``core.components.open_boundary``)."""
        self._ob.radiate(u_old, u_new, axis, edge, w, ext=ext, mode=mode, fluxes=self._fluxes)

    def _flather_args(self) -> dict:
        """Flather targets for the free-surface step of this step's exterior; {} if closed."""
        return self._ob.flather_args(self._ext)

    def _open_velocity(self, st: SigmaState, u_new: np.ndarray, v_new: np.ndarray,
                       wu: np.ndarray | None = None, wv: np.ndarray | None = None) -> None:
        """In place: sponge on u′, then Orlanski at the open edges (``core.components.open_boundary``)."""
        self._ob.relax_velocity(st.u, st.v, u_new, v_new, self._ext, self._fluxes, wu, wv)

    def _evaluate_exterior(self, t: float, hu: np.ndarray, hv: np.ndarray, wu, wv) -> None:
        """Exterior state at t, kept for this step in ``self._ext`` (None without a provider)."""
        self._ext = self._ob.evaluate(t, hu, hv, wu, wv)

    def _edge_face_depths(self) -> tuple[np.ndarray, np.ndarray]:
        """Rest depth at u- and v-faces [m], open edge faces from their boundary cell."""
        return self._ob.edge_face_depths()

    def edge_face_depths(self) -> tuple[np.ndarray, np.ndarray]:
        """Rest depth at u- and v-faces [m], 0 on land; open edge faces take the depth of their boundary cell."""
        return self._ob.edge_face_depths()

    def _exterior_T(self, h: np.ndarray) -> np.ndarray:
        """Tₑₓₜ on layers h in the boundary/sponge columns for this step's exterior."""
        return self._ob.exterior_T(h, self._ext)

    def _inflow_T(self) -> np.ndarray:
        """Exterior T on the layers of this step: σ, ``T_ext``; ALE, the z profile remapped to hⁿ (spec §3.5)."""
        return self.T_ext if self._T_in is None else self._T_in

    def _exterior(self, h: np.ndarray) -> np.ndarray:
        """Exterior T profile remapped onto thicknesses h (``core.components.open_boundary``)."""
        return self._ob.exterior(h)

    def _exterior_tr(self, h: np.ndarray) -> np.ndarray:
        """Exterior passive tracers on layers h for this step's exterior."""
        return self._ob.exterior_tr(h, self._ext)

    def _bed_drag_rates(self, u: np.ndarray, v: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """Quadratic bed rates and bbl fractions (``core.components.bottom_drag``)."""
        return self._bed_drag.rates(u, v, hu, hv)

    def _interior_sponge(self, T: np.ndarray, tr: np.ndarray | None, eta: np.ndarray, h: np.ndarray):
        """Implicit relaxation toward the sponge profile (``core.components.sponge``)."""
        return self._interior_sponge_c.apply(T, tr, eta, h)

    def _open_face_transports(self, Qx: np.ndarray, Qy: np.ndarray, h: np.ndarray, un: np.ndarray, vn: np.ndarray,
                              u_new: np.ndarray, v_new: np.ndarray) -> None:
        """In place: layer fluxes through the open faces (``core.components.open_boundary``)."""
        self._ob.open_face_transports(Qx, Qy, h, un, vn, u_new, v_new, self._fluxes)

    def _rhs_numpy(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, ok: np.ndarray,
                   centred: bool = False) -> np.ndarray:
        return self._tracer_advection.rhs_numpy(c, Qx, Qy, Fz, ok, centred, self._inflow(c))

    def _rhs(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, ok: np.ndarray,
             centred: bool = False) -> np.ndarray:
        return self._tracer_advection.rhs(c, Qx, Qy, Fz, ok, centred, self._inflow(c))

    def _ext_stack(self, c: np.ndarray) -> np.ndarray:
        """Inflow values of the advected stack on open edges (``core.components.open_boundary``)."""
        return self._ob.inflow_stack(c, self._inflow_T(), self._tr_in)

    def _faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, ok: np.ndarray, ext: np.ndarray | None = None):
        return self._tracer_advection.faces_h(c, Qx, Qy, ok, ext, self._inflow(c))

    def _upwind_faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray):
        return self._tracer_advection.upwind_faces_h(c, Qx, Qy, self._inflow(c))

    def _advect_fct(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                    ok: np.ndarray, unlimited: bool = False) -> np.ndarray:
        """SSP-RK3 WENO5 with Zalesak FCT (``core.components.tracers``), with this step's inflow values."""
        return self._tracer_advection.advect_fct(C, V0, V1, Qx, Qy, ok, unlimited, self._inflow(C))

    def _advect_fct_numpy(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                          ok: np.ndarray, unlimited: bool = False) -> np.ndarray:
        return self._tracer_advection.advect_fct_numpy(C, V0, V1, Qx, Qy, ok, unlimited, self._inflow(C))

    def _advect_numpy(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                      Fz: np.ndarray, ok: np.ndarray) -> np.ndarray:
        return self._tracer_advection.advect_numpy(C, V0, V1, Qx, Qy, Fz, ok, self._inflow(C))

    def _advect(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                Fz: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """SSP-RK3 (or 'pc') flux-form advection (``core.components.tracers``), with this step's inflow values."""
        return self._tracer_advection.advect(C, V0, V1, Qx, Qy, Fz, ok, self._inflow(C))

    # ------------------------------------------------------------------ step
    def step(self, st: SigmaState, tau_x: np.ndarray | None = None, tau_y: np.ndarray | None = None) -> SigmaState:
        """
        Advance one step.  tau_x (ny, nx+1), tau_y (ny+1, nx): surface (wind) stress at the u- and
        v-faces [N m⁻²]; τ/ρ₀ enters as the top boundary flux of the implicit vertical viscosity
        (as ROMS/CROCO sustr/svstr), not through the AB3 extrapolation.  The ten phases of the step, the same
        in σ and ALE, are run by ``core.model.driver.Driver`` (N1).
        """
        return self._driver.step(st, tau_x, tau_y)


def _delegate(name: str) -> property:
    return property(lambda self: getattr(self._ob, name), lambda self, value: setattr(self._ob, name, value),
                    doc=f"open-boundary configuration ``{name}`` (``core.components.open_boundary``)")


for _name in _OB_CONFIG:
    setattr(SigmaBaroclinic, _name, _delegate(_name))


def _component_param(name: str, targets: tuple[tuple[str, str], ...], to_component=None,
                     normalise=None) -> property:
    """
    A model parameter kept in sync with the components that use it (N1).  The model stores the value; setting it
    after construction (scripts change ``ke_scheme``, tests ``kh``, sensitivity runs ``cd`` …) also sets it on
    every built component in ``targets`` ((component attribute of the model, attribute of the component)), as
    ``SigmaBaroclinic`` read ``self.<name>`` every step before N1.  ``to_component(model, value)``: the value the
    component keeps, when it differs; ``normalise(value)``: the stored spelling.
    """
    key = f"_pre_{name}"

    def get(self):
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(name) from None

    def put(self, value) -> None:
        if normalise is not None:
            value = normalise(value)
        self.__dict__[key] = value
        for comp_name, attr in targets:
            comp = self.__dict__.get(comp_name)
            if comp is not None and hasattr(comp, attr):
                setattr(comp, attr, value if to_component is None else to_component(self, value))
    return property(get, put, doc=f"``{name}``, kept in sync with {', '.join(c for c, _ in targets)}")


def _fixed_after_construction(name: str) -> property:
    """EOS constants: shared by the EOS, PGF, tracer diffusion and regrid; changing them later is refused."""
    key = f"_pre_{name}"

    def get(self):
        try:
            return self.__dict__[key]
        except KeyError:
            raise AttributeError(name) from None

    def put(self, value) -> None:
        if "_eos" in self.__dict__:
            raise AttributeError(f"{name} is fixed after construction (the equation of state, the pressure gradient "
                                 "and the tracer diffusion share it); build a new SigmaBaroclinic instead")
        self.__dict__[key] = value
    return property(get, put, doc=f"``{name}`` of the equation of state (fixed after construction)")


_ke_spelling = lambda v: "gudonov" if v == "godunov" else v                         # noqa: E731  kernel spelling
_PARAMS = {
    "ke_scheme": ((("_momentum", "ke_scheme"),), None, _ke_spelling),
    "ah": ((("_momentum", "ah"),), None, None),
    "adv": ((("_momentum", "adv"),), None, None),
    "momentum": ((("_momentum", "form"),),
                 lambda m, v: "vector_invariant" if m.ale and v == "vector_invariant" else "flux", None),
    "kh": ((("_tracer_diffusion", "kh"),), None, None),
    "tracer_scheme": ((("_tracer_advection", "tracer_scheme"),), None, None),
    "tracer_time": ((("_tracer_advection", "tracer_time"),), None, None),
    "pgf": ((("_pgf", "scheme"),), None, None),
    "thickness_flux": ((("_layer_transport", "thickness_flux"),), None, None),
    "cd": ((("_bed_drag", "cd"),), lambda m, v: float(v), None),
    "hbbl": ((("_bed_drag", "hbbl"),), lambda m, v: float(v), None),
    "drag_bg_vel": ((("_bed_drag", "drag_bg_vel"),), lambda m, v: float(v), None),
    "max_truncations": ((("_vertical", "max_truncations"),), None, None),
    "h_thin_bed": ((("_vertical", "h_thin_bed"),), None, None),
    "regrid_rule": ((("_vertical", "regrid_rule"),), None, None),
    "remap_code": ((("_vertical", "remap_code"), ("_ob", "remap_code"), ("_interior_sponge_c", "remap_code")),
                   None, None),
}
for _name, (_targets, _to, _norm) in _PARAMS.items():
    setattr(SigmaBaroclinic, _name, _component_param(_name, _targets, _to, _norm))
for _name in ("rho0", "t_coef", "t_ref"):
    setattr(SigmaBaroclinic, _name, _fixed_after_construction(_name))


def _given(args: dict) -> frozenset[str]:
    """Constructor arguments passed with a value other than their default (arrays: any array given)."""
    import inspect
    out = set()
    for name, p in inspect.signature(SigmaBaroclinic.__init__).parameters.items():
        if name == "self" or name not in args:
            continue
        v, d = args[name], p.default
        if d is inspect.Parameter.empty or v is d:
            continue
        if isinstance(v, np.ndarray) or isinstance(d, np.ndarray):
            out.add(name)
            continue
        try:
            if v != d:
                out.add(name)
        except (TypeError, ValueError):
            out.add(name)
    return frozenset(out)
