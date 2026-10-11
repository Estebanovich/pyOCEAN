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

from dataclasses import dataclass, field

import numpy as np
import xarray as xr

from core.dynamics.nonlinear import advection_u, advection_v, laplacian_u, laplacian_v
from core.dynamics.sigma_wd import SigmaWetDry
from diagnostics.mixing import reference_potential_energy

G = 9.81
AB3 = (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0)


@dataclass
class SigmaState:
    """Prognostic state: η (ny,nx), u (N,ny,nx+1), v (N,ny+1,nx), T (N,ny,nx); diagnostics."""
    eta: np.ndarray
    u: np.ndarray
    v: np.ndarray
    T: np.ndarray
    g_hist: list = field(default_factory=list)              # past explicit tendencies (AB3)
    omega: np.ndarray | None = None                          # ω at interfaces (N+1,ny,nx) [m s⁻¹]
    mix_num: np.ndarray | None = None                        # volume-weighted numerical mixing [T² m³ s⁻¹]
    Qx: np.ndarray | None = None                             # layer volume transports of the last step [m³ s⁻¹]
    Qy: np.ndarray | None = None
    Fz: np.ndarray | None = None
    time: float = 0.0


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


def _weno5_faces(c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int, variant: str = "z") -> np.ndarray:
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


def _flux_form_u(u: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, V: np.ndarray,
                 ok_u: np.ndarray) -> np.ndarray:
    """
    Momentum-conserving advection of u (N,ny,nx+1) by the layer transports
    (flux form, as ROMS/CROCO; Shchepetkin & McWilliams 2005):

        −(1/V_u) [ δ_x(U ũ) + δ_y(V ũ) + δ_k(W ũ) − u (δ_x U + δ_y V + δ_k W) ]

    U = transport through the cell centres (mean of the two u-face transports),
    V through the corners, W through the layer interfaces at the u-face; ũ are
    WENO5 values upwind of each transport.  Subtracting u × (mass divergence)
    gives the velocity tendency while the momentum flux stays conservative, which
    sets the speed of fronts and jumps (Borden & Meiburg 2013; Ungarish et al. 2018).
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


class ReferenceProfile:
    """ρ_ref(z) tabulated at heights z [m] (increasing) with values rho [kg m⁻³]; linear in between."""

    def __init__(self, z: np.ndarray, rho: np.ndarray) -> None:
        self.z, self.rho = np.asarray(z, float), np.asarray(rho, float)

    def __call__(self, zz: np.ndarray) -> np.ndarray:
        return np.interp(zz, self.z, self.rho)


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
    """

    def __init__(self, h_grid: xr.Dataset, bathy: np.ndarray, dt: float, n_layers: int = 10,
                 rho0: float = 1000.0, t_coef: float = 0.2, t_ref: float = 0.0, z0: float | None = None,
                 closure: str = "constant", nu_const: float = 1e-4, ah: float = 0.0, adv: str = "flux",
                 theta: float = 0.6, d_min: float = 0.05, g: float = G, tracer_scheme: str = "weno5",
                 tracer_time: str = "ssprk3", pgf: str = "jacobian", rho_ref=None,
                 open_boundaries: tuple[str, ...] = (), sponge_cells: int = 0,
                 sponge_tau: tuple[float, float] = (1200.0, 60.0), orlanski_cmax: float = 1.0,
                 sponge_tracer: bool = True, dsigma: np.ndarray | None = None) -> None:
        if pgf not in ("twoterm", "jacobian"):
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
        self.orlanski_cmax, self.sponge_tracer = orlanski_cmax, sponge_tracer
        self.sponge_cells, self.sponge_tau = int(sponge_cells), tuple(float(t) for t in sponge_tau)
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
        self.core = SigmaWetDry(h_grid, bathy, dt=dt, n_layers=n_layers, z0=z0, closure=closure,
                                nu_const=nu_const, theta=theta, g=g, d_min=d_min, dsigma=dsigma,
                                flather_u=fl_u if self.open else None, flather_v=fl_v if self.open else None)
        self._sp_c, self._sp_u, self._sp_v = _sponge_rates(act.shape, self.open, sponge_cells, *sponge_tau)
        self.T_ext = None
        self.hg, self.dt, self.n, self.g = h_grid, dt, n_layers, g
        self.rho0, self.t_coef, self.t_ref, self.ah, self.adv = rho0, t_coef, t_ref, ah, adv
        self.fs = self.core.fs
        self._fq = np.asarray(h_grid["f_q"].values, float)           # Coriolis at cell corners [s⁻¹]
        self.H = self.fs.H
        self.ds = self.core.dsigma
        self.sigma_c = 0.5 * (self.core.sigma_w[1:, 0] + self.core.sigma_w[:-1, 0])   # layer centres (bed = 0)
        self.area = self.fs.area

    # ------------------------------------------------------------------ helpers
    def initial_state(self, eta: np.ndarray, T: np.ndarray) -> SigmaState:
        """Resting state; with ``rho_ref="auto"`` the reference profile is taken from this T."""
        ny, nx = self.H.shape
        if isinstance(self.rho_ref, str) and self.rho_ref == "auto":
            z = self.layer_heights(eta)
            act = np.broadcast_to(self.fs.active[None], z.shape)
            self.rho_ref = reference_profile(z[act], self.density(np.asarray(T, float))[act])
        self.T_ext = np.array(T, float)                                # inflow / sponge target at open boundaries
        return SigmaState(eta=np.array(eta, float), u=np.zeros((self.n, ny, nx + 1)),
                          v=np.zeros((self.n, ny + 1, nx)), T=np.array(T, float))

    def layer_heights(self, eta: np.ndarray) -> np.ndarray:
        """Height of the layer centres z_k = −H + D σ_c,k [m] (z = 0 at the rest surface), (N, ny, nx)."""
        D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
        return -np.nan_to_num(self.H)[None] + D[None] * self.sigma_c[:, None, None]

    def density(self, T: np.ndarray) -> np.ndarray:
        """Linear EOS ρ = ρ₀ − α (T − T₀) [kg m⁻³]."""
        return self.rho0 - self.t_coef * (T - self.t_ref)

    def layer_volumes(self, eta: np.ndarray) -> np.ndarray:
        D = np.maximum(np.nan_to_num(self.H + eta, nan=0.0), 0.0)
        return D[None] * self.ds[:, None, None] * self.area[None]

    def tracer_content(self, st: SigmaState, power: int = 1) -> float:
        V = self.layer_volumes(st.eta)
        return float(np.sum(V * st.T**power))

    def rpe(self, st: SigmaState) -> float:
        """Reference potential energy [J] (flat-bottom basin: area = wet area)."""
        V = self.layer_volumes(st.eta)
        act = np.broadcast_to(self.fs.active[None], V.shape)
        return reference_potential_energy(self.density(st.T)[act], V[act], area=float(self.area[self.fs.active].sum()))

    # ------------------------------------------------------------------ explicit tendencies
    def _baroclinic_pgf(self, eta: np.ndarray, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
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
        n = self.n
        D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
        h = D[None] * self.ds[:, None, None]
        z = -np.nan_to_num(self.H)[None] + D[None] * self.sigma_c[:, None, None]   # cell-centre height (z = 0 at rest)
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
        gu, gv = self._baroclinic_pgf(st.eta, st.T)
        cu, cv = self._coriolis(st.u, st.v)
        gu += cu
        gv += cv
        if self.adv == "flux":
            if st.Qx is not None:
                V = self.layer_volumes(st.eta)
                gu += _flux_form_u(st.u, st.Qx, st.Qy, st.Fz, V, wet_u[None])
                swap = lambda a: np.swapaxes(a, 1, 2)                                   # noqa: E731
                gv += swap(_flux_form_u(swap(st.v), swap(st.Qy), swap(st.Qx), swap(st.Fz), swap(V),
                                        swap(wet_v[None])))
        for k in range(n):
            if self.adv != "flux":
                gu[k] -= advection_u(st.u[k], st.v[k], self.hg, wet_u, scheme=self.adv)
                gv[k] -= advection_v(st.u[k], st.v[k], self.hg, wet_v, scheme=self.adv)
            if self.ah > 0:
                gu[k] += self.ah * laplacian_u(st.u[k], self.hg, wet_u)
                gv[k] += self.ah * laplacian_v(st.v[k], self.hg, wet_v)
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
        ext = np.stack([self.T_ext, self.T_ext**2])[: c.shape[0]]
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

    def _baroclinic_part(self, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Depth mean Σ Δσ_k u_k and the baroclinic remainder u_k − ū (Σ Δσ_k u′_k = 0)."""
        mean = np.tensordot(self.ds, vel, axes=(0, 0))
        return mean, vel - mean[None]

    def _radiate(self, u_old: np.ndarray, u_new: np.ndarray, axis: int, edge: int) -> None:
        """
        Orlanski (1976) radiation of the baroclinic normal velocity at one open edge, implicit
        upstream form as ROMS u3dbc (Marchesiello et al. 2001, normal part only):

            r = c Δt/Δx = (φ₁ⁿ − φ₁ⁿ⁺¹)(φ₁ⁿ − φ₂ⁿ) / (φ₁ⁿ − φ₂ⁿ)²,  clipped to [0, r_max]
            φ_Bⁿ⁺¹ = (φ_Bⁿ + r φ₁ⁿ⁺¹) / (1 + r)

        with B the boundary face and 1, 2 the next faces inwards (r = 0: incoming → clamped).
        The depth-mean part (Flather) is kept; u_new is modified in place.
        """
        _, po = self._baroclinic_part(u_old)
        bar, pn = self._baroclinic_part(u_new)
        i1, i2 = (1, 2) if edge == 0 else (-2, -3)
        take = lambda a, i: np.take(a, i, axis=axis)                                  # noqa: E731
        grad = take(po, i1) - take(po, i2)
        cx = (take(po, i1) - take(pn, i1)) * grad
        r = np.clip(cx / np.maximum(grad**2, 1e-30), 0.0, self.orlanski_cmax)
        pb = (take(po, edge) + r * take(pn, i1)) / (1.0 + r)
        pb -= np.tensordot(self.ds, pb, axes=(0, 0))[None]                            # keep Σ Δσ u′ = 0
        open_ = take((self.fs.last_open_Du if axis == 2 else self.fs.last_open_Dv)[None], edge) > 0
        sl = [slice(None)] * u_new.ndim
        sl[axis] = edge
        u_new[tuple(sl)] = np.where(open_, take(bar[None], edge) + pb, take(u_new, edge))

    def _advect(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                Fz: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """
        SSP-RK3 flux-form advection of the tracer stack C (ntr,N,ny,nx) with frozen
        transports Qx, Qy (layer volume fluxes at faces) and Fz (N+1,ny,nx, through
        the interfaces); layer volumes go linearly from V0 to V1.
        """
        dt = self.dt

        def rhs(c, centred=False):
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
            gu_ab, gv_ab = 1.5 * gu - 0.5 * hist[1][0], 1.5 * gv - 0.5 * hist[1][1]
        else:
            gu_ab = AB3[0] * gu + AB3[1] * hist[1][0] + AB3[2] * hist[2][0]
            gv_ab = AB3[0] * gv + AB3[1] * hist[1][1] + AB3[2] * hist[2][1]
        tu = None if tau_x is None else np.asarray(tau_x, float) / self.rho0
        tv = None if tau_y is None else np.asarray(tau_y, float) / self.rho0
        fl = {}
        if self.open:                                                   # radiate to a resting exterior (η = 0, ū = 0)
            zu, zv = np.zeros(self.fs.H_u.shape), np.zeros(self.fs.H_v.shape)
            fl = dict(flather_eta=(zu, zv), flather_ubar=zu, flather_vbar=zv)
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv, **fl)
        if self._sp_u is not None:
            # flow-relaxation sponge on the baroclinic velocity (implicit; the depth mean, and so the
            # free-surface volume fluxes, are untouched — barotropic waves leave through Flather)
            for vel, rate in ((u_new, self._sp_u), (v_new, self._sp_v)):
                bar, prime = self._baroclinic_part(vel)
                vel[...] = bar[None] + prime / (1.0 + dt * rate[None])
        if self.open:
            for name, vel, old, axis, edge in (("west", u_new, st.u, 2, 0), ("east", u_new, st.u, 2, -1),
                                                ("south", v_new, st.v, 1, 0), ("north", v_new, st.v, 1, -1)):
                if name in self.open:
                    self._radiate(old, vel, axis, edge)

        # layer transports consistent with the free-surface flux (fs.last_Du / last_Dv: upwind face depth)
        un = np.where(self.fs.last_wet_u[None], st.u, 0.0)
        vn = np.where(self.fs.last_wet_v[None], st.v, 0.0)
        Qx = (self.fs.Lu * self.fs.last_Du)[None] * self.ds[:, None, None] * (th * u_new + (1 - th) * un)
        Qy = (self.fs.Lv * self.fs.last_Dv)[None] * self.ds[:, None, None] * (th * v_new + (1 - th) * vn)
        if self.open:
            # open faces: Δσ_k × the free-surface (Flather) flux + the baroclinic part, so Σ_k Q_k is exactly
            # the volume flux of the continuity equation
            for Q, D, Qo, L, vo, vn1 in ((Qx, self.fs.last_open_Du, self.fs.last_open_Qu, self.fs.Lu, st.u, u_new),
                                         (Qy, self.fs.last_open_Dv, self.fs.last_open_Qv, self.fs.Lv, st.v, v_new)):
                po, pn = self._baroclinic_part(vo)[1], self._baroclinic_part(vn1)[1]
                ds = self.ds[:, None, None]
                Q[...] = np.where((D > 0)[None], ds * Qo[None] + (L * D)[None] * ds * (th * pn + (1 - th) * po), Q)
        V0, V1 = self.layer_volumes(st.eta), self.layer_volumes(eta_new)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        Fz = np.zeros((n + 1,) + self.H.shape)                              # volume flux through interface k (k=0 bed)
        Fz[1:] = -np.cumsum((V1 - V0) / dt + divh, axis=0)
        Fz[-1] = 0.0                                                        # surface: round-off residual only
        ok = (np.broadcast_to(self.fs.active[None], V1.shape)) & (V1 > 0)
        C = self._advect(np.stack([st.T, st.T**2]), V0, V1, Qx, Qy, Fz, ok)
        T_new = C[0]
        mix = V1 * (C[1] - C[0] ** 2) / dt
        if self._sp_c is not None and self.sponge_tracer:
            r = dt * self._sp_c[None]
            T_new = (T_new + r * self.T_ext) / (1.0 + r)
        omega = Fz / np.where(self.area > 0, self.area, 1.0)[None]
        return SigmaState(eta=eta_new, u=u_new, v=v_new, T=T_new, g_hist=hist, omega=omega, mix_num=mix,
                          Qx=Qx, Qy=Qy, Fz=Fz, time=st.time + dt)
