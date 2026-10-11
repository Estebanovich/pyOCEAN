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

from core.dynamics.wetting_drying import WetDryFreeSurface
from physics.bottom_friction import first_layer_cd
from physics.mixing.gls import GLSKEpsilon
from physics.mixing.parabolic import parabolic_viscosity

G = 9.81
CLOSURES = ("gls", "parabolic", "constant", "none")


def vertical_implicit(u: np.ndarray, dz: np.ndarray, nu: np.ndarray, r_bed: np.ndarray, dt: float) -> np.ndarray:
    """
    Backward-Euler vertical diffusion of momentum with a linearised bed stress
    (tridiagonal, Thomas algorithm; columns are independent).

        Δz_k (u_k' − u_k)/Δt = ν_{k+½}(u'_{k+1} − u'_k)/h_{k+½} − ν_{k−½}(u'_k − u'_{k−1})/h_{k−½}
                               − δ_{k,0} r u'_0

    u, dz : (N, M) layer velocity [m s⁻¹] and thickness [m], k = 0 at the bed
    nu : (N+1, M) viscosity at the interfaces [m² s⁻¹] (bed and surface values unused)
    r_bed : (M,) bed stress rate r = C_d |u_b| [m s⁻¹] (0 = free slip)
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
    diag[0] += dt * r_bed
    rhs = dz * u
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
    n_layers : number of equal-thickness σ-layers
    z0 : bed roughness length [m] for the first-layer law of the wall (None: no bed stress)
    closure : 'gls' (k-ε, Canuto A, as CROCO; state at cell centres), 'parabolic'
              (ν = κ u* z(1 − z/D)), 'constant' (ν = nu_const) or 'none'
    theta, g, d_min, flather_u, flather_v, solver : as ``WetDryFreeSurface``
    """

    def __init__(self, h_grid: xr.Dataset, bathy: np.ndarray, dt: float, n_layers: int = 10,
                 z0: float | None = 1e-4, closure: str = "parabolic", nu_const: float = 1e-2,
                 nu_min: float = 1e-5, theta: float = 0.6, g: float = G, d_min: float = 0.05,
                 flather_u: np.ndarray | None = None, flather_v: np.ndarray | None = None,
                 solver: str = "direct", cd_max: float = 0.1) -> None:
        if closure not in CLOSURES:
            raise ValueError(f"unknown closure {closure!r}; use one of {CLOSURES}")
        self.fs = WetDryFreeSurface(h_grid, bathy, dt=dt, theta=theta, g=g, d_min=d_min,
                                    flather_u=flather_u, flather_v=flather_v, solver=solver)
        self.dt, self.n_layers, self.z0, self.closure = dt, n_layers, z0, closure
        self.nu_const, self.nu_min, self.d_min, self.cd_max = nu_const, nu_min, d_min, cd_max
        self.dsigma = np.full(n_layers, 1.0 / n_layers)
        self.sigma_w = np.linspace(0.0, 1.0, n_layers + 1)[:, None]
        self.H = self.fs.H
        ny, nx = self.H.shape
        self._fl_u = np.zeros((ny, nx + 1), bool) if flather_u is None else np.asarray(flather_u, bool)
        self._fl_v = np.zeros((ny + 1, nx), bool) if flather_v is None else np.asarray(flather_v, bool)
        self.gls = GLSKEpsilon(n_layers, ny * nx, z0_bed=z0 or 1e-4) if closure == "gls" else None

    # ------------------------------------------------------------------ columns
    def column_step(self, u: np.ndarray, G_k: np.ndarray | None, depth: np.ndarray,
                    nu: np.ndarray | None = None) -> np.ndarray:
        """
        Step 1 for the columns of one velocity component: û = (vertical viscosity
        + bed stress)⁻¹ (u + Δt G).  u, G_k : (N, ...); depth : (...) total depth [m];
        nu : (N+1, ...) viscosity at the interfaces (closure 'gls': supplied by ``step``).
        """
        shape = u.shape
        n = shape[0]
        uu = (u if G_k is None else u + self.dt * G_k).reshape(n, -1)
        D = np.maximum(np.nan_to_num(np.asarray(depth, float).reshape(-1), nan=self.d_min), self.d_min)
        dz = np.broadcast_to(D * self.dsigma[:, None], uu.shape)
        if self.z0 is None:
            r = np.zeros(D.size)
            ustar = np.zeros(D.size)
        else:
            cd = first_layer_cd(dz[0], self.z0, cd_max=self.cd_max)
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
        if self.closure == "none" and self.z0 is None:
            return uu.reshape(shape)
        return vertical_implicit(uu, dz, nu, r, self.dt).reshape(shape)

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
             flather_vbar: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance η and the layer velocities u (N, ny, nx+1), v (N, ny+1, nx) one step.
        G_u, G_v : explicit tendencies per layer [m s⁻²] (advection, Coriolis, body force).
        Returns η^{n+1}, u^{n+1}, v^{n+1}.
        """
        ds = self.dsigma[:, None, None]
        Du, Dv = self.face_depths(eta)
        nu_u = nu_v = None
        if self.gls is not None:
            nu_u, nu_v = self._gls_at_faces()
        u_hat = self.column_step(u, G_u, Du, nu_u)
        v_hat = self.column_step(v, G_v, Dv, nu_v)
        ubar_n, vbar_n = (ds * u).sum(0), (ds * v).sum(0)
        ubar_hat, vbar_hat = (ds * u_hat).sum(0), (ds * v_hat).sum(0)
        eta_new, ubar_new, vbar_new = self.fs.step(eta, ubar_n, vbar_n, ubar_hat, vbar_hat, flather_eta=flather_eta,
                                                   flather_ubar=flather_ubar, flather_vbar=flather_vbar)
        wet_u = self.fs.last_wet_u | (self._fl_u & (ubar_new != 0))
        wet_v = self.fs.last_wet_v | (self._fl_v & (vbar_new != 0))
        u_new = np.where(wet_u, u_hat + (ubar_new - ubar_hat), 0.0)
        v_new = np.where(wet_v, v_hat + (vbar_new - vbar_hat), 0.0)
        if self.gls is not None:
            self._gls_update(eta_new, u_new, v_new)
        return eta_new, u_new, v_new

    # ------------------------------------------------------------------ GLS coupling
    def _gls_at_faces(self) -> tuple[np.ndarray, np.ndarray]:
        """Cell-centre ν (N+1, ny, nx) averaged to u- and v-faces (edge faces: the inner cell)."""
        ny, nx = self.H.shape
        nu = self.gls.akv.reshape(self.n_layers + 1, ny, nx)
        nu_u = np.empty((self.n_layers + 1, ny, nx + 1))
        nu_u[:, :, 1:-1] = 0.5 * (nu[:, :, :-1] + nu[:, :, 1:])
        nu_u[:, :, 0], nu_u[:, :, -1] = nu[:, :, 0], nu[:, :, -1]
        nu_v = np.empty((self.n_layers + 1, ny + 1, nx))
        nu_v[:, 1:-1] = 0.5 * (nu[:, :-1] + nu[:, 1:])
        nu_v[:, 0], nu_v[:, -1] = nu[:, 0], nu[:, -1]
        return nu_u, nu_v

    def _gls_update(self, eta: np.ndarray, u: np.ndarray, v: np.ndarray) -> None:
        """
        Advance k-ε with the new velocities: cell-centre layer velocities and bed
        stress |τ_b|/ρ₀ = C_d |u₁| u₁ averaged from the faces (CROCO gls_mixing.F).
        """
        n = self.n_layers
        D = np.maximum(np.nan_to_num(self.H + eta, nan=self.d_min), self.d_min)
        dz = (D[None] * self.dsigma[:, None, None]).reshape(n, -1)
        uc = 0.5 * (u[:, :, :-1] + u[:, :, 1:])
        vc = 0.5 * (v[:, :-1, :] + v[:, 1:, :])
        Du, Dv = self.face_depths(eta)
        cdu = first_layer_cd(np.maximum(np.nan_to_num(Du, nan=self.d_min), self.d_min) * self.dsigma[0], self.z0 or 1e-4,
                             cd_max=self.cd_max)
        cdv = first_layer_cd(np.maximum(np.nan_to_num(Dv, nan=self.d_min), self.d_min) * self.dsigma[0], self.z0 or 1e-4,
                             cd_max=self.cd_max)
        speed_u = np.hypot(u[0], 0.0)
        speed_v = np.hypot(v[0], 0.0)
        tau_u = cdu * speed_u * u[0] if self.z0 is not None else 0.0 * u[0]
        tau_v = cdv * speed_v * v[0] if self.z0 is not None else 0.0 * v[0]
        tx = 0.5 * (tau_u[:, :-1] + tau_u[:, 1:])
        ty = 0.5 * (tau_v[:-1, :] + tau_v[1:, :])
        self.gls.step(uc.reshape(n, -1), vc.reshape(n, -1), dz, np.hypot(tx, ty).ravel(), self.dt)
