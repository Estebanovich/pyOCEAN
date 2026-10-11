"""
Stratified σ-layer model with wetting and drying.

Extends the homogeneous ``SigmaWetDry`` (Casulli & Cheng 1992 fractional step)
with an active temperature tracer, a linear equation of state, the baroclinic
pressure gradient and momentum advection on σ-surfaces.  Each step:

1. Explicit momentum tendencies G_k, combined with AB3 (Durran 2010, §2.4;
   MITgcm's AB3, Adcroft et al. 2004 [verificar]):
       G = −(u·∇_σ)u − (ω/h) ∂u/∂k + A_h ∇²u + PGF_bc
2. ``SigmaWetDry.step``: implicit vertical viscosity + bed stress, θ-implicit
   free surface on the depth-mean, uniform barotropic correction.
3. Layer volume transports Q_k = L D_f Δσ_k (θ u_k^{n+1} + (1−θ) u_k^n), whose
   vertical sum is exactly the free-surface flux, so
       ω_{k+½} = ω_{k−½} − [ (V_k^{n+1} − V_k^n)/Δt + ∇_h·Q_k ] / A
   vanishes at the surface and a uniform tracer stays uniform (constancy).
4. Tracers in flux form with WENO5 face values (Jiang & Shu 1996; the CROCO
   GRAV_ADJ choice) — or UP3 (Leonard 1979), which overshoots by ~50 % at a
   lock-exchange nose, or first-order upwind — lower order next to walls, advanced
   with SSP-RK3 (Shu & Osher 1988) on frozen transports, layer volumes linear in
   time.  T and T² are advected together, giving the online numerical mixing of
   Burchard & Rennau (2008): M_num = V (A(T²) − (A T)²)/Δt.

Baroclinic pressure gradient on σ-surfaces (hydrostatic, Boussinesq):

    PGF_x = −(1/ρ₀) [ ∂P′/∂x|_σ + g ρ′ ∂z/∂x|_σ ],   P′(z) = g ∫_z^η ρ′ dz′,   ρ′ = ρ − ρ₀

(second-order two-term form; exact for horizontally uniform ρ over a flat bed.
Over sloping beds use the finite-volume PGF of Adcroft et al. 2008 — roadmap
stage B, seamount test).  Equation of state: ρ = ρ₀ − α (T − T₀), as CROCO
``lin_EOS_cff``.

Limitations (documented, tested where it matters): Cartesian uniform spacing for
the UP3 face interpolation; closed domains for tracers (Flather faces are not yet
tracer-consistent); salinity not included.

References
----------
Burchard, H. & Rennau, H. (2008). Ocean Modell. 20, 293–311.
Casulli, V. & Cheng, R.T. (1992). Int. J. Numer. Meth. Fluids 15, 629–648.
Shchepetkin, A.F. & McWilliams, J.C. (2005). Ocean Modell. 9, 347–404.
Shu, C.-W. & Osher, S. (1988). J. Comput. Phys. 77, 439–471.
Jiang, G.-S. & Shu, C.-W. (1996). Efficient implementation of weighted ENO
    schemes. J. Comput. Phys. 126, 202–228.
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
                 theta: float = 0.6, d_min: float = 0.05, g: float = G, tracer_scheme: str = "weno5") -> None:
        if tracer_scheme not in ("weno5", "weno5js", "upwind3", "upwind1"):
            raise ValueError(f"unknown tracer scheme {tracer_scheme!r}")
        self.tracer_scheme = tracer_scheme
        self.core = SigmaWetDry(h_grid, bathy, dt=dt, n_layers=n_layers, z0=z0, closure=closure,
                                nu_const=nu_const, theta=theta, g=g, d_min=d_min)
        self.hg, self.dt, self.n, self.g = h_grid, dt, n_layers, g
        self.rho0, self.t_coef, self.t_ref, self.ah, self.adv = rho0, t_coef, t_ref, ah, adv
        self.fs = self.core.fs
        self.H = self.fs.H
        self.ds = self.core.dsigma
        self.area = self.fs.area

    # ------------------------------------------------------------------ helpers
    def initial_state(self, eta: np.ndarray, T: np.ndarray) -> SigmaState:
        ny, nx = self.H.shape
        return SigmaState(eta=np.array(eta, float), u=np.zeros((self.n, ny, nx + 1)),
                          v=np.zeros((self.n, ny + 1, nx)), T=np.array(T, float))

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
        n = self.n
        D = np.maximum(np.nan_to_num(self.H + eta, nan=self.fs.d_min), self.fs.d_min)
        h = D[None] * self.ds[:, None, None]
        rp = self.density(T) - self.rho0
        sig_c = (np.arange(n) + 0.5)[:, None, None] / n
        z = -np.nan_to_num(self.H)[None] + D[None] * sig_c                # cell-centre height (z = 0 at rest surface)
        above = np.cumsum((rp * h)[::-1], axis=0)[::-1] - rp * h          # ∫ over the layers above
        P = self.g * (above + 0.5 * rp * h)                                # P′ at cell centres
        gx = np.zeros((n,) + self.fs.H_u.shape)
        gy = np.zeros((n,) + self.fs.H_v.shape)
        okx = np.isfinite(self.fs.H_u[:, 1:-1])
        dxu = self.fs.du[:, 1:-1]
        gx[:, :, 1:-1] = np.where(okx, -((P[:, :, 1:] - P[:, :, :-1])
                                         + self.g * 0.5 * (rp[:, :, 1:] + rp[:, :, :-1]) * (z[:, :, 1:] - z[:, :, :-1]))
                                  / dxu / self.rho0, 0.0)
        oky = np.isfinite(self.fs.H_v[1:-1, :])
        dyv = self.fs.dv[1:-1, :]
        gy[:, 1:-1, :] = np.where(oky, -((P[:, 1:, :] - P[:, :-1, :])
                                         + self.g * 0.5 * (rp[:, 1:, :] + rp[:, :-1, :]) * (z[:, 1:, :] - z[:, :-1, :]))
                                  / dyv / self.rho0, 0.0)
        return gx, gy

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
    def _faces(self, c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int) -> np.ndarray:
        if self.tracer_scheme.startswith("weno5"):
            return _weno5_faces(c, flux, ok, axis, variant="js" if self.tracer_scheme == "weno5js" else "z")
        return _up3_faces(c, flux, ok, axis, order=1 if self.tracer_scheme == "upwind1" else 3)

    def _advect(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                Fz: np.ndarray, ok: np.ndarray) -> np.ndarray:
        """
        SSP-RK3 flux-form advection of the tracer stack C (ntr,N,ny,nx) with frozen
        transports Qx, Qy (layer volume fluxes at faces) and Fz (N+1,ny,nx, through
        the interfaces); layer volumes go linearly from V0 to V1.
        """
        dt = self.dt

        def rhs(c):
            okb = np.broadcast_to(ok, c.shape)
            fx = Qx * self._faces(c, np.broadcast_to(Qx, c.shape[:1] + Qx.shape), okb, 3)
            fy = Qy * self._faces(c, np.broadcast_to(Qy, c.shape[:1] + Qy.shape), okb, 2)
            # vertical: Fz[j] is the volume flux through interface j (bottom of layer j; 0 = bed, N = surface)
            fzc = np.broadcast_to(Fz[:-1], c.shape)
            fz = np.zeros(c.shape[:1] + Fz.shape)
            fz[:, :-1] = Fz[:-1] * self._faces(c, fzc, okb, 1)
            return -(fx[..., 1:] - fx[..., :-1] + fy[:, :, 1:, :] - fy[:, :, :-1, :] + fz[:, 1:] - fz[:, :-1])

        dV = V1 - V0
        safe = lambda q, v: np.where(v > 0, q / np.where(v > 0, v, 1.0), C)     # noqa: E731
        q0 = V0 * C
        q1 = q0 + dt * rhs(C)
        Va = V0 + dV
        c1 = safe(q1, Va)
        q2 = 0.75 * q0 + 0.25 * (q1 + dt * rhs(c1))
        Vb = V0 + 0.5 * dV
        c2 = safe(q2, Vb)
        q3 = q0 / 3.0 + 2.0 / 3.0 * (q2 + dt * rhs(c2))
        return safe(q3, V1)

    # ------------------------------------------------------------------ step
    def step(self, st: SigmaState) -> SigmaState:
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
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab)

        # layer transports consistent with the free-surface flux (fs.last_Du / last_Dv: upwind face depth)
        un = np.where(self.fs.last_wet_u[None], st.u, 0.0)
        vn = np.where(self.fs.last_wet_v[None], st.v, 0.0)
        Qx = (self.fs.Lu * self.fs.last_Du)[None] * self.ds[:, None, None] * (th * u_new + (1 - th) * un)
        Qy = (self.fs.Lv * self.fs.last_Dv)[None] * self.ds[:, None, None] * (th * v_new + (1 - th) * vn)
        V0, V1 = self.layer_volumes(st.eta), self.layer_volumes(eta_new)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        Fz = np.zeros((n + 1,) + self.H.shape)                              # volume flux through interface k (k=0 bed)
        Fz[1:] = -np.cumsum((V1 - V0) / dt + divh, axis=0)
        Fz[-1] = 0.0                                                        # surface: round-off residual only
        ok = (np.broadcast_to(self.fs.active[None], V1.shape)) & (V1 > 0)
        C = self._advect(np.stack([st.T, st.T**2]), V0, V1, Qx, Qy, Fz, ok)
        T_new = C[0]
        mix = V1 * (C[1] - C[0] ** 2) / dt
        omega = Fz / np.where(self.area > 0, self.area, 1.0)[None]
        return SigmaState(eta=eta_new, u=u_new, v=v_new, T=T_new, g_hist=hist, omega=omega, mix_num=mix,
                          Qx=Qx, Qy=Qy, Fz=Fz, time=st.time + dt)
