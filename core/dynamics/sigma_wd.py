"""
Homogeneous 3-D shallow-water model on σ-layers with wetting and drying.

The water column of total depth D = H + η is split into N layers of equal
thickness Δz = D Δσ (σ = z/D from the bed), so layers thin out but never
vanish as a cell dries.  Each layer carries its own horizontal velocity; the
density is uniform, so the pressure gradient is the barotropic −g∇η in all
layers.  Fractional step (Casulli & Cheng 1992; Casulli 1999):

1. Explicit tendencies G_k (advection, Coriolis, body force — from the caller)
   and implicit vertical viscosity with the log-law bed stress:

       Δz_k (û_k − u_k) / Δt = ∂/∂z (ν ∂u/∂z)|_k − δ_k1 C_d |u₁| û₁ + Δz_k G_k
       C_d = [κ / ln(Δz₁ / 2 z₀)]²                  (first-layer law of the wall)

2. The depth-mean ū̂ = Σ Δσ û_k goes to the wet–dry free-surface solver
   (``WetDryFreeSurface``), which returns η^{n+1} and ū^{n+1} with the
   θ-weighted barotropic pressure gradient and the upwind total face depth.

3. The barotropic correction is uniform in the vertical:
       u_k^{n+1} = û_k + (ū^{n+1} − ū̂);   u_k = 0 on dry faces.

Splitting the pressure correction from the implicit friction lags the
friction by one step on the pressure-driven increment (first order in Δt);
it keeps the free-surface system identical to the 2-D one, so mass
conservation and positivity are inherited from it.  Vertical advection
(ω ∂u/∂σ) is not included yet.

References
----------
Casulli, V. & Cheng, R.T. (1992). Semi-implicit finite difference methods for
    three-dimensional shallow water flow. Int. J. Numer. Meth. Fluids 15, 629–648.
Casulli, V. (1999). A semi-implicit finite difference method for
    non-hydrostatic, free-surface flows. Int. J. Numer. Meth. Fluids 30, 425–440.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from numba import njit, prange

from core.components.closure import GLSClosure
from core.dynamics.wetting_drying import WetDryFreeSurface
from physics.bottom_friction import first_layer_cd
from physics.mixing.gls import KT_MIN
from physics.mixing.parabolic import parabolic_viscosity

G = 9.81
CLOSURES = ("gls", "parabolic", "constant", "none")


@njit(cache=True, parallel=True)
def _thomas_columns(u, dz, nu, r_bed, tau_top, dt):  # pragma: no cover
    """Compiled column solver of ``vertical_implicit`` (same algebra), parallel over columns."""
    n, m = u.shape
    out = np.empty_like(u)
    for col in prange(m):
        cp = np.empty(n)
        dp = np.empty(n)
        for k in range(n):
            lower = 0.0
            upper = 0.0
            diag = dz[k, col]
            if k > 0:
                lower = -dt * nu[k, col] / (0.5 * (dz[k, col] + dz[k - 1, col]))
                diag -= lower
            if k < n - 1:
                upper = -dt * nu[k + 1, col] / (0.5 * (dz[k, col] + dz[k + 1, col]))
                diag -= upper
            diag += dt * r_bed[k, col]
            rhs = dz[k, col] * u[k, col]
            if k == n - 1:
                rhs += dt * tau_top[col]
            if k == 0:
                cp[0] = upper / diag
                dp[0] = rhs / diag
            else:
                mm = diag - lower * cp[k - 1]
                cp[k] = upper / mm
                dp[k] = (rhs - lower * dp[k - 1]) / mm
        out[n - 1, col] = dp[n - 1]
        for k in range(n - 2, -1, -1):
            out[k, col] = dp[k] - cp[k] * out[k + 1, col]
    return out


def vertical_implicit(u: np.ndarray, dz: np.ndarray, nu: np.ndarray, r_bed: np.ndarray, dt: float,
                      tau_top: np.ndarray | None = None) -> np.ndarray:
    """Backward-Euler vertical diffusion with bed stress (see ``_vertical_implicit_numpy``), compiled."""
    u = np.ascontiguousarray(u, dtype=float)
    m = u.shape[1]
    tau = np.zeros(m) if tau_top is None else np.ascontiguousarray(np.broadcast_to(tau_top, (m,)), dtype=float)
    return _thomas_columns(u, np.ascontiguousarray(np.broadcast_to(dz, u.shape), dtype=float),
                           np.ascontiguousarray(nu, dtype=float), _layer_rates(r_bed, u.shape), tau, float(dt))


def _layer_rates(r_bed: np.ndarray, shape: tuple) -> np.ndarray:
    """(N, M) Rayleigh rates per layer [m s⁻¹]: a (M,) bed rate acts on layer 0 only, a (N, M) one as given."""
    r = np.asarray(r_bed, dtype=float)
    if r.ndim == 2:
        return np.ascontiguousarray(np.broadcast_to(r, shape), dtype=float)
    out = np.zeros(shape)
    out[0] = np.broadcast_to(r, (shape[1],))
    return out


def _vertical_implicit_numpy(u: np.ndarray, dz: np.ndarray, nu: np.ndarray, r_bed: np.ndarray, dt: float,
                             tau_top: np.ndarray | None = None) -> np.ndarray:
    """
    Backward-Euler vertical diffusion of momentum with a linearised bed stress
    (tridiagonal, Thomas algorithm; columns are independent).

        Δz_k (u_k' − u_k)/Δt = ν_{k+½}(u'_{k+1} − u'_k)/h_{k+½} − ν_{k−½}(u'_k − u'_{k−1})/h_{k−½}
                               − r_k u'_k + δ_{k,N−1} τ_s/ρ₀

    u, dz : (N, M) layer velocity [m s⁻¹] and thickness [m], k = 0 at the bed
    nu : (N+1, M) viscosity at the interfaces [m² s⁻¹] (bed and surface values unused)
    r_bed : (M,) bed stress rate r = C_d |u_b| [m s⁻¹] on layer 0 (0 = free slip), or (N, M) rates r_k of a bed
        stress spread over the bottom layers (Σ_k r_k u'_k is the stress; ALE quadratic drag over hbbl)
    tau_top : (M,) kinematic surface stress τ_s/ρ₀ [m² s⁻²], the momentum flux into the top layer
    """
    n = u.shape[0]
    h = 0.5 * (dz[1:] + dz[:-1])                          # interface spacing, (N−1, M)
    c = dt * nu[1:-1] / h                                  # coupling at interior interfaces
    lower = np.zeros_like(u)
    upper = np.zeros_like(u)
    lower[1:] = -c
    upper[:-1] = -c
    diag = dz.copy()
    diag[1:] += c
    diag[:-1] += c
    diag += dt * _layer_rates(r_bed, u.shape)
    rhs = dz * u
    if tau_top is not None:
        rhs[-1] = rhs[-1] + dt * tau_top
    # Thomas forward sweep
    cp = np.zeros_like(u)
    dp = np.zeros_like(u)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, n):
        m = diag[k] - lower[k] * cp[k - 1]
        cp[k] = upper[k] / m
        dp[k] = (rhs[k] - lower[k] * dp[k - 1]) / m
    out = np.empty_like(u)
    out[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        out[k] = dp[k] - cp[k] * out[k + 1]
    return out


class SigmaWetDry:
    """
    σ-layer homogeneous 3-D wet–dry model (see module docstring).

    Parameters
    ----------
    h_grid : horizontal grid
    bathy : (ny, nx) bed depth [m] (H > 0 water, H < 0 floodable land, NaN wall)
    dt : time step [s]
    n_layers : number of σ-layers
    dsigma : (n_layers,) layer fractions Δσ_k, k = 0 at the bed (default uniform; see
             core.grid.sigma_levels for stretched σ); must be positive and sum to 1
    z0 : bed roughness length [m] for the first-layer law of the wall (None: no bed stress)
    closure : 'gls' (k-ε, Canuto A, as CROCO; state at cell centres), 'parabolic'
              (ν = κ u* z(1 − z/D)), 'constant' (ν = nu_const) or 'none'
    theta, g, d_min, flather_u, flather_v, solver, clamped_u, clamped_v : as ``WetDryFreeSurface``
    gls_stratified : closure 'gls' with buoyancy (N2a task 8c, physics/mixing/gls.py): ``step`` then needs the
        layer density ``rho`` and the closure also gives the tracer diffusivity ``gls.akt``; False (default) is the
        homogeneous closure of this module, unchanged
    rho0 : reference density ρ₀ of N² = −(g/ρ₀) ∂ρ/∂z [kg m⁻³] (gls_stratified only)
    gls_kt_min : floor (background) of the GLS tracer diffusivity κ_t [m² s⁻¹] (CROCO nuws = 1e-6)
    external_closure : True — the closure component is owned and stepped by the caller (``SigmaBaroclinic``, N3
        task 1, ``core.components.closure``): ``step`` takes ν from its ``nu_u``, ``nu_v`` arguments and the caller
        updates the closure with ``closure_forcing``.  False (default, standalone use): closure 'gls' builds its
        own ``GLSClosure`` (``turbulence``), used by ``step`` as before N3.
    """

    def __init__(self, h_grid: xr.Dataset, bathy: np.ndarray, dt: float, n_layers: int = 10,
                 z0: float | None = 1e-4, closure: str = "parabolic", nu_const: float = 1e-2,
                 nu_min: float = 1e-5, theta: float = 0.6, g: float = G, d_min: float = 0.05,
                 flather_u: np.ndarray | None = None, flather_v: np.ndarray | None = None,
                 solver: str = "direct", cd_max: float = 0.1, dsigma: np.ndarray | None = None,
                 clamped_u: np.ndarray | None = None, clamped_v: np.ndarray | None = None,
                 gls_stratified: bool = False, rho0: float = 1000.0, gls_kt_min: float = KT_MIN,
                 external_closure: bool = False) -> None:
        if closure not in CLOSURES:
            raise ValueError(f"unknown closure {closure!r}; use one of {CLOSURES}")
        self.fs = WetDryFreeSurface(h_grid, bathy, dt=dt, theta=theta, g=g, d_min=d_min,
                                    flather_u=flather_u, flather_v=flather_v, solver=solver,
                                    clamped_u=clamped_u, clamped_v=clamped_v)
        self.dt, self.n_layers, self.z0, self.closure = dt, n_layers, z0, closure
        self.nu_const, self.nu_min, self.d_min, self.cd_max = nu_const, nu_min, d_min, cd_max
        if dsigma is None:
            dsigma = np.full(n_layers, 1.0 / n_layers)
        dsigma = np.asarray(dsigma, float)
        if dsigma.shape != (n_layers,) or np.any(dsigma <= 0) or abs(dsigma.sum() - 1.0) > 1e-12:
            raise ValueError(f"dsigma must be {n_layers} positive fractions summing to 1")
        self.dsigma = dsigma
        self.sigma_w = np.concatenate([[0.0], np.cumsum(dsigma)])[:, None]          # interfaces, bed = 0
        self.sigma_w[-1] = 1.0
        self.H = self.fs.H
        ny, nx = self.H.shape
        self._fl_u = np.zeros((ny, nx + 1), bool) if flather_u is None else np.asarray(flather_u, bool)
        self._fl_v = np.zeros((ny + 1, nx), bool) if flather_v is None else np.asarray(flather_v, bool)
        self.gls_stratified, self.rho0, self.g = bool(gls_stratified) and closure == "gls", float(rho0), g
        self.external_closure = bool(external_closure)
        # the closure component (core.components.closure); with external_closure the caller attaches its own
        # (SigmaBaroclinic) and steps it, the core only holds the reference (``gls``)
        self.turbulence = None
        if closure == "gls" and not self.external_closure:
            self.turbulence = GLSClosure(n_layers, (ny, nx), dt, z0=z0, stratified=self.gls_stratified,
                                         rho0=self.rho0, g=g, kt_min=gls_kt_min)

    @property
    def gls(self):
        """The GLS k-ε object of the closure component (``physics.mixing.gls.GLSKEpsilon``), or None."""
        return getattr(self.turbulence, "gls", None)

    # ------------------------------------------------------------------ columns
    def column_step(self, u: np.ndarray, G_k: np.ndarray | None, depth: np.ndarray,
                    nu: np.ndarray | None = None, tau: np.ndarray | None = None,
                    dz: np.ndarray | None = None, dz_drag: np.ndarray | None = None,
                    dz_mean: np.ndarray | None = None, r_bed: np.ndarray | None = None) -> np.ndarray:
        """
        Step 1 for the columns of one velocity component: û = (vertical viscosity
        + bed stress)⁻¹ (u + Δt G).  u, G_k : (N, ...); depth : (...) total depth [m];
        nu : (N+1, ...) viscosity at the interfaces (closure 'gls': supplied by ``step``).
        tau : (...) kinematic surface stress τ_s/ρ₀ [m² s⁻²] (wind), a flux into the top layer.
        dz : (N, ...) layer thickness at the faces [m] (default D Δσ_k).
        dz_drag : (...) bed-layer thickness for the log-law drag C_d(dz_drag) [m] (default dz[0]); the ALE step
            passes the mean bed-layer thickness when dz is the viscous coupling thickness (spec 2026-10-14 §3).
        dz_mean : (N, ...) layer thickness of the depth mean ū = Σ w_k u_k, w_k = dz_mean_k / D_f, D_f = Σ dz_mean
            (the face column of the free surface; in ALE clipped at the shallower bed), given when dz is not that
            thickness (the ALE thin-layer coupling).  The solve conserves Σ dz_k u_k, so internal friction would
            change ū and the wind would enter it as Δt τ / dz_top; a uniform correction δ then restores the
            depth-mean balance Σ w û = Σ w (u + Δt G) + Δt (τ_s − r û₀) / D_f (implicit bed stress included),
            keeping the shear.
        r_bed : (...) linear bed-stress rate r = C_d |u_b| [m s⁻¹] on layer 0, or (N, ...) per-layer rates r_k of a
            stress spread over the bed layers (ALE quadratic drag over hbbl), given by the caller; overrides the
            log-law of ``z0``.
        """
        shape = u.shape
        n = shape[0]
        uu = (u if G_k is None else u + self.dt * G_k).reshape(n, -1)
        D = np.maximum(np.nan_to_num(np.asarray(depth, float).reshape(-1), nan=self.d_min), self.d_min)
        if dz is None:
            dz = np.broadcast_to(D * self.dsigma[:, None], uu.shape)
        else:
            dz = np.asarray(dz, float).reshape(n, -1)
        if r_bed is not None:
            r = np.nan_to_num(np.asarray(r_bed, float))
            r = r.reshape(n, -1) if r.ndim == len(shape) else r.reshape(-1)
            ustar = np.zeros(D.size)
        elif self.z0 is None:
            r = np.zeros(D.size)
            ustar = np.zeros(D.size)
        else:
            d0 = dz[0] if dz_drag is None else np.asarray(dz_drag, float).reshape(-1)
            cd = first_layer_cd(d0, self.z0, cd_max=self.cd_max)
            r = cd * np.abs(uu[0])
            ustar = np.sqrt(cd) * np.abs(uu[0])
        if nu is not None:
            nu = np.asarray(nu, float).reshape(n + 1, -1)
        elif self.closure == "parabolic":
            nu = parabolic_viscosity(ustar, D, self.sigma_w, nu_min=self.nu_min)
        elif self.closure in ("constant", "gls"):
            nu = np.full((n + 1, D.size), self.nu_const)
        else:
            nu = np.zeros((n + 1, D.size))
        tau = None if tau is None else np.asarray(tau, float).reshape(-1)
        if self.closure == "none" and self.z0 is None and r_bed is None:
            out = uu.copy()
            if tau is not None:
                out[-1] += self.dt * tau / dz[-1]
        else:
            out = vertical_implicit(uu, dz, nu, r, self.dt, tau_top=tau)
        if dz_mean is not None:
            hm = np.asarray(dz_mean, float).reshape(n, -1)
            Df = np.maximum(hm.sum(0), self.d_min)
            w = hm / Df
            stress, r_tot = (r * out[0], r) if r.ndim == 1 else ((r * out).sum(0), r.sum(0))   # Σ_k r_k û_k
            rhs = (w * uu).sum(0) + self.dt * (0.0 if tau is None else tau) / Df - (w * out).sum(0) \
                - self.dt * stress / Df
            out = out + rhs / (w.sum(0) + self.dt * r_tot / Df)          # δ with the bed stress at û + δ
        return out.reshape(shape)

    def face_depths(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Total depth at u- and v-faces: mean of the two cells (edge faces: the inner cell) [m]."""
        D = self.H + eta
        Du = np.full((D.shape[0], D.shape[1] + 1), np.nan)
        Du[:, 1:-1] = 0.5 * (D[:, :-1] + D[:, 1:])
        Du[:, 0], Du[:, -1] = D[:, 0], D[:, -1]
        Dv = np.full((D.shape[0] + 1, D.shape[1]), np.nan)
        Dv[1:-1, :] = 0.5 * (D[:-1, :] + D[1:, :])
        Dv[0, :], Dv[-1, :] = D[0, :], D[-1, :]
        return Du, Dv

    # ------------------------------------------------------------------ full step
    def step(self, eta: np.ndarray, u: np.ndarray, v: np.ndarray, G_u: np.ndarray | None = None,
             G_v: np.ndarray | None = None, flather_eta=None, flather_ubar: np.ndarray | None = None,
             flather_vbar: np.ndarray | None = None, tau_u: np.ndarray | None = None,
             tau_v: np.ndarray | None = None, wu: np.ndarray | None = None, wv: np.ndarray | None = None,
             hu: np.ndarray | None = None, hv: np.ndarray | None = None, hu_drag: np.ndarray | None = None,
             hv_drag: np.ndarray | None = None, hu_mean: np.ndarray | None = None,
             hv_mean: np.ndarray | None = None, r_u: np.ndarray | None = None,
             r_v: np.ndarray | None = None, flather_q=None,
             fw: np.ndarray | None = None, rho: np.ndarray | None = None, nu_u: np.ndarray | None = None,
             nu_v: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance η and the layer velocities u (N, ny, nx+1), v (N, ny+1, nx) one step.
        G_u, G_v : explicit tendencies per layer [m s⁻²] (advection, Coriolis, body force).
        tau_u, tau_v : kinematic surface stress τ/ρ₀ at u- and v-faces [m² s⁻²] (wind).
        wu, wv : layer fractions at the faces for the depth mean ū = Σ w_k u_k (default Δσ_k;
            ALE: core.vertical.ale.face_fractions).
        hu, hv : layer thickness at the faces for the implicit vertical viscosity [m] (default D Δσ_k).
        hu_drag, hv_drag : bed-layer thickness at the faces for the log-law drag [m] (default hu[0], hv[0]); given
            when hu, hv are the ALE coupling thickness.
        hu_mean, hv_mean : layer thickness at the faces whose fractions are wu, wv [m] (ALE: clipped at the
            shallower bed, so Σ is the free-surface face column); given when hu, hv are the coupling thickness,
            so column_step restores the depth-mean balance (``dz_mean``).
        r_u, r_v : linear bed rates at u- and v-faces [m s⁻¹] (ALE quadratic drag; column_step ``r_bed``).
        flather_q : prescribed volume fluxes at Flather faces (WetDryFreeSurface.step).
        fw : (ny, nx) surface freshwater flux P − E [m s⁻¹] as a real volume source of the free surface
            (WetDryFreeSurface.step; N2a); None: none.
        rho : (N, ny, nx) layer density [kg m⁻³] for N² of the stratified GLS (``gls_stratified``; the model EOS at
            the old tracers, as CROCO's rho_eos before step3d_uv2 and gls_mixing, step.F:954-1094).
        nu_u, nu_v : (N+1, ny, nx+1), (N+1, ny+1, nx) viscosity at the face interfaces [m² s⁻¹] from the caller's
            closure (``external_closure``; ``Closure.viscosity``); None: the column solver's own ν.  Ignored when
            the core steps its own closure (standalone 'gls').
        Returns η^{n+1}, u^{n+1}, v^{n+1}.
        """
        ds = self.dsigma[:, None, None]
        Du, Dv = self.face_depths(eta)
        own = self.turbulence if not self.external_closure else None
        if own is not None:                                            # standalone GLS: ν from the current state
            nu_u, nu_v = own.viscosity(None)
        wu_ = ds if wu is None else wu
        wv_ = ds if wv is None else wv
        u_hat = self.column_step(u, G_u, Du, nu_u, tau=tau_u, dz=hu, dz_drag=hu_drag,
                                 dz_mean=hu_mean, r_bed=r_u)                   # coupling thickness ≠ the weights'
        v_hat = self.column_step(v, G_v, Dv, nu_v, tau=tau_v, dz=hv, dz_drag=hv_drag, dz_mean=hv_mean, r_bed=r_v)
        ubar_n, vbar_n = (wu_ * u).sum(0), (wv_ * v).sum(0)
        ubar_hat, vbar_hat = (wu_ * u_hat).sum(0), (wv_ * v_hat).sum(0)
        eta_new, ubar_new, vbar_new = self.fs.step(eta, ubar_n, vbar_n, ubar_hat, vbar_hat, flather_eta=flather_eta,
                                                   flather_ubar=flather_ubar, flather_vbar=flather_vbar,
                                                   flather_q=flather_q, fw=fw)
        wet_u = self.fs.last_wet_u | (self._fl_u & (ubar_new != 0))
        wet_v = self.fs.last_wet_v | (self._fl_v & (vbar_new != 0))
        u_new = np.where(wet_u, u_hat + (ubar_new - ubar_hat), 0.0)
        v_new = np.where(wet_v, v_hat + (vbar_new - vbar_hat), 0.0)
        if own is not None:                                            # CROCO order: ν above, then update
            own.update(rho=rho, **self.closure_forcing(eta_new, u_new, v_new, tau_u, tau_v))
        return eta_new, u_new, v_new

    # ------------------------------------------------------------------ closure coupling
    def closure_forcing(self, eta: np.ndarray, u: np.ndarray, v: np.ndarray, tau_u: np.ndarray | None = None,
                        tau_v: np.ndarray | None = None) -> dict[str, np.ndarray | None]:
        """
        Inputs of ``Closure.update`` after the step (N3 task 1; before N3 inside ``_gls_update`` and
        ``_gls_update_stratified``, same arithmetic): the new layer velocities averaged to the cell centres
        u, v (N, ny, nx), the layer thicknesses h = D Δσ_k (N, ny, nx) at η^{n+1}, the bed stress
        u*²_b = |τ_b|/ρ₀ with τ_b/ρ₀ = C_d |u₁| u₁ of the first-layer log law averaged from the faces (CROCO
        gls_mixing.F), and — stratified GLS only — the surface stress u*²_s = |τ_s|/ρ₀ averaged from the faces
        to the centres (CROCO gls_mixing.F:556-562); homogeneous GLS: u*²_s None (zero).  [m s⁻¹, m, m² s⁻²]
        """
        D = np.maximum(np.nan_to_num(self.H + eta, nan=self.d_min), self.d_min)
        h = D[None] * self.dsigma[:, None, None]
        uc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        Du, Dv = self.face_depths(eta)
        cdu = first_layer_cd(np.maximum(np.nan_to_num(Du, nan=self.d_min), self.d_min) * self.dsigma[0],
                             self.z0 or 1e-4, cd_max=self.cd_max)
        cdv = first_layer_cd(np.maximum(np.nan_to_num(Dv, nan=self.d_min), self.d_min) * self.dsigma[0],
                             self.z0 or 1e-4, cd_max=self.cd_max)
        ny, nx = self.H.shape
        if not self.gls_stratified:                                    # homogeneous GLS (pre-N2a 8c arithmetic)
            speed_u = np.hypot(u[0], 0.0)
            speed_v = np.hypot(v[0], 0.0)
            tbu = cdu * speed_u * u[0] if self.z0 is not None else 0.0 * u[0]
            tbv = cdv * speed_v * v[0] if self.z0 is not None else 0.0 * v[0]
            tx = 0.5 * (tbu[:, :-1] + tbu[:, 1:])
            ty = 0.5 * (tbv[:-1, :] + tbv[1:, :])
            return dict(u=uc, v=vc, h=h, ustar2_sfc=None, ustar2_bot=np.hypot(tx, ty))
        tau_bu = cdu * np.abs(u[0]) * u[0] if self.z0 is not None else np.zeros_like(u[0])
        tau_bv = cdv * np.abs(v[0]) * v[0] if self.z0 is not None else np.zeros_like(v[0])
        bx = 0.5 * (tau_bu[:, :-1] + tau_bu[:, 1:])
        by = 0.5 * (tau_bv[:-1, :] + tau_bv[1:, :])
        tsu = np.zeros((ny, nx + 1)) if tau_u is None else np.broadcast_to(np.nan_to_num(tau_u), (ny, nx + 1))
        tsv = np.zeros((ny + 1, nx)) if tau_v is None else np.broadcast_to(np.nan_to_num(tau_v), (ny + 1, nx))
        sx = 0.5 * (tsu[:, :-1] + tsu[:, 1:])
        sy = 0.5 * (tsv[:-1, :] + tsv[1:, :])
        return dict(u=uc, v=vc, h=h, ustar2_sfc=np.hypot(sx, sy), ustar2_bot=np.hypot(bx, by))
