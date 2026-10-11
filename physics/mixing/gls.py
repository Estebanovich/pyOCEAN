"""
Generic length scale (GLS) turbulence closure, k-ε variant, with the
Canuto et al. (2001) "A" stability functions — a column-by-column port of
CROCO ``gls_mixing.F`` (default GLS_KEPSILON + CANUTO_A, no horizontal or
vertical advection of the turbulent quantities, vertical shear only).

Equations at the layer interfaces (Umlauf & Burchard 2003, eqs. 11–12;
Warner et al. 2005):

    ∂k/∂t = ∂/∂z (ν/σ_k ∂k/∂z) + P + B − ε
    ∂ψ/∂t = ∂/∂z (ν/σ_ψ ∂ψ/∂z) + (ψ/k)(β₁ P + β₃ B − β₂ ε)

with ψ = (c_μ⁰)^p k^m l^n; for k-ε p = 3, m = 3/2, n = −1 so ψ = ε.
P = ν S², S² = (∂u/∂z)² + (∂v/∂z)²; B = −ν' N² (= 0 here: homogeneous water).
Viscosity ν = c_μ(α_M, α_N) k²/ε, α_M = (k/ε)² S², α_N = (k/ε)² N².

Discretisation (as CROCO): backward Euler, production explicit, dissipation
linearised implicitly (Patankar 1980) so k, ψ stay positive; tridiagonal in
the interior interfaces 1 … N−1 with no-flux for k and the log-layer ψ flux
at the bed and surface (Burchard et al. 1998):

    F_ψ = −n (c_μ⁰)^{p+1} κ σ_ψ⁻¹ k^{m+½} l^n,   l = κ (Δz/2 + z₀)

References
----------
Umlauf, L. & Burchard, H. (2003). A generic length-scale equation for
    geophysical turbulence models. J. Mar. Res. 61, 235–265.
Canuto, V.M., Howard, A., Cheng, Y. & Dubovikov, M.S. (2001). Ocean turbulence.
    Part I. J. Phys. Oceanogr. 31, 1413–1426.
Warner, J.C., Sherwood, C.R., Arango, H.G. & Signell, R.P. (2005). Performance
    of four turbulence closure models implemented using a generic length scale
    method. Ocean Modell. 8, 81–113.
"""

from __future__ import annotations

import numpy as np

KAPPA = 0.41
G = 9.81

# k-ε exponents and coefficients (CROCO GLS_KEPSILON)
RP, RM, RN = 3.0, 1.5, -1.0
BETA1, BETA2 = 1.44, 1.92
SIG_K_INV, SIG_PSI_INV = 1.0, 0.7692
E1, E2, E3 = 3.0 + RP / RN, 1.5 + RM / RN, -1.0 / RN

# Canuto et al. (2001) A — stability-function constants (CROCO gls_stab_func.h)
C1, C2, C3, C4, C5, C6 = 5.0, 0.8, 1.968, 1.136, 0.0, 0.4
CB1, CB2, CB3, CB4, CB5, CBB = 5.95, 0.6, 1.0, 0.0, 0.3333, 0.72
A1, A2, A3, A5 = 2.0 / 3.0 - 0.5 * C2, 1.0 - 0.5 * C3, 1.0 - 0.5 * C4, 0.5 - 0.5 * C6
AB1, AB2, AB3, AB5 = 1.0 - CB2, 1.0 - CB3, 2.0 * (1.0 - CB4), 2.0 * CBB * (1.0 - CB5)
NN, NB = 0.5 * C1, CB1
SF_D0 = 36.0 * NN**3 * NB**2
SF_D1 = 84.0 * A5 * AB3 * NN**2 * NB + 36.0 * AB5 * NN**3 * NB
SF_D2 = 9.0 * (AB2**2 - AB1**2) * NN**3 - 12.0 * (A2**2 - 3.0 * A3**2) * NN * NB**2
SF_D3 = (12.0 * A5 * AB3 * (A2 * AB1 - 3.0 * A3 * AB2) * NN + 12.0 * A5 * AB3 * (A3**2 - A2**2) * NB
         + 12.0 * AB5 * (3.0 * A3**2 - A2**2) * NN * NB)
SF_D4 = 48.0 * A5**2 * AB3**2 * NN + 36.0 * A5 * AB3 * AB5 * NN**2
SF_D5 = 3.0 * (A2**2 - 3.0 * A3**2) * (AB1**2 - AB2**2) * NN
SF_N0 = 36.0 * A1 * NN**2 * NB**2
SF_N1 = (-12.0 * A5 * AB3 * (AB1 + AB2) * NN**2 + 8.0 * A5 * AB3 * (6.0 * A1 - A2 - 3.0 * A3) * NN * NB
         + 36.0 * A1 * AB5 * NN**2 * NB)
SF_N2 = 9.0 * A1 * (AB2**2 - AB1**2) * NN**2
SF_NB0 = 12.0 * AB3 * NN**3 * NB
SF_NB1 = 12.0 * A5 * AB3**2 * NN**2
LIM_AM0, LIM_AM1 = SF_D0 * SF_N0, SF_D0 * SF_N1 + SF_D1 * SF_N0
LIM_AM2, LIM_AM3 = SF_D1 * SF_N1 + SF_D4 * SF_N0, SF_D4 * SF_N1
LIM_AM4, LIM_AM5, LIM_AM6 = SF_D2 * SF_N0, SF_D2 * SF_N1 + SF_D3 * SF_N0, SF_D3 * SF_N1


def _tridiag(lower: np.ndarray, diag: np.ndarray, upper: np.ndarray, rhs: np.ndarray) -> np.ndarray:
    """Thomas algorithm along axis 0 (columns independent)."""
    n = diag.shape[0]
    cp = np.empty_like(diag)
    dp = np.empty_like(diag)
    cp[0] = upper[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, n):
        m = diag[k] - lower[k] * cp[k - 1]
        cp[k] = upper[k] / m
        dp[k] = (rhs[k] - lower[k] * dp[k - 1]) / m
    x = np.empty_like(diag)
    x[-1] = dp[-1]
    for k in range(n - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    return x


class GLSKEpsilon:
    """
    k-ε GLS closure for M independent water columns of N layers.

    State at the N+1 interfaces (index 0 = bed, N = surface), shape (N+1, M):
    ``tke`` k [m² s⁻²], ``eps`` ε [m² s⁻³], ``akv`` ν [m² s⁻¹].
    """

    tke_min, eps_min, nu_min = 1e-10, 1e-12, 1e-5          # CROCO gls_mixing.F (non-3D values)
    galp, z0s_min, z0b_min, chk = 0.53, 1e-2, 1e-4, 1400.0 / G

    def __init__(self, n_layers: int, n_columns: int, z0_bed: float = 1e-4) -> None:
        self.n = n_layers
        self.z0_bed = max(z0_bed, self.z0b_min)
        self.cm0 = ((A2**2 - 3.0 * A3**2 + 3.0 * A1 * NN) / (3.0 * NN**2)) ** 0.25
        self.alpha_n_min = (0.5 * (-(SF_D1 + SF_NB0) + np.sqrt((SF_D1 + SF_NB0) ** 2
                                                                 - 4.0 * SF_D0 * (SF_D4 + SF_NB1)))
                            / (SF_D4 + SF_NB1))
        cff = self.cm0**3 * self.tke_min**1.5 / self.eps_min
        self.psi_min = self.cm0**RP * self.tke_min**RM * cff**RN
        shape = (n_layers + 1, n_columns)
        self.tke = np.full(shape, self.tke_min)
        self.eps = np.full(shape, self.psi_min)             # ψ = ε for k-ε
        self.akv = np.full(shape, self.nu_min)

    def _solve(self, var: np.ndarray, sprod: np.ndarray, dcoef: np.ndarray, sig_inv: float, akv_old: np.ndarray,
               dz: np.ndarray, dt: float, flux_bot: np.ndarray, flux_top: np.ndarray, floor: float) -> np.ndarray:
        """Implicit diffusion + Patankar sources at interfaces 1…N−1; no coupling to 0 and N."""
        n = self.n
        h = 0.5 * (dz[:-1] + dz[1:])                         # interface cell height, (N−1, M)
        # coupling through layer j (between interfaces j and j+1), zero for the bed and surface layers
        fc = -0.5 * dt * sig_inv * (akv_old[:-1] + akv_old[1:]) / dz      # (N, M)
        fc[0] = 0.0
        fc[-1] = 0.0
        lower = fc[:-1]                                       # couples interface i with i−1 (layer i−1)
        upper = fc[1:]                                        # couples interface i with i+1 (layer i)
        diag = h * (1.0 + dt * dcoef) - lower - upper
        rhs = h * (var[1:-1] + dt * sprod)
        rhs[0] += dt * flux_bot
        rhs[-1] += dt * flux_top
        lo = lower.copy()
        lo[0] = 0.0
        up = upper.copy()
        up[-1] = 0.0
        out = _tridiag(lo, diag, up, rhs)
        return np.maximum(out, floor) if n > 1 else out

    def step(self, u: np.ndarray, v: np.ndarray, dz: np.ndarray, ustar2_bot: np.ndarray, dt: float,
             ustar2_sfc: np.ndarray | None = None) -> np.ndarray:
        """
        Advance k and ε one step and return the new viscosity ν (N+1, M).

        u, v : (N, M) layer velocities at the column [m s⁻¹]; dz : (N, M) layer thickness [m]
        ustar2_bot, ustar2_sfc : |τ|/ρ₀ at the bed and surface [m² s⁻²]
        """
        cm0 = self.cm0
        us2s = np.zeros(dz.shape[1]) if ustar2_sfc is None else ustar2_sfc
        h = 0.5 * (dz[:-1] + dz[1:])
        shear2 = ((u[1:] - u[:-1]) ** 2 + (v[1:] - v[:-1]) ** 2) / h**2      # (N−1, M)
        k_old, psi_old, akv_old = self.tke[1:-1], self.eps[1:-1], self.akv
        diss = np.maximum(cm0**E1 * k_old**E2 * psi_old**E3, self.eps_min)
        prod = akv_old[1:-1] * shear2

        # --- k: P explicit, ε/k · k implicit; zero flux, Dirichlet k = u*²/c_μ⁰² stored at 0 and N
        tke_new = self.tke.copy()
        tke_new[0] = np.maximum(self.tke_min, us2s * 0 + ustar2_bot / cm0**2)
        tke_new[-1] = np.maximum(self.tke_min, us2s / cm0**2)
        zero = np.zeros(dz.shape[1])
        tke_new[1:-1] = self._solve(self.tke, prod, diss / k_old, SIG_K_INV, akv_old, dz, dt, zero, zero,
                                    self.tke_min)

        # --- ψ: β₁ (ψ/k) P explicit, β₂ (ε/k) ψ implicit; log-layer flux at the bed and surface
        z0s = np.maximum(self.z0s_min, self.chk * us2s)
        ks = 0.5 * (tke_new[-2] + tke_new[-1])
        ls = KAPPA * (0.5 * dz[-1] + z0s)
        flux_top = -RN * cm0 ** (RP + 1) * KAPPA * SIG_PSI_INV * ks ** (RM + 0.5) * ls**RN
        kb = 0.5 * (tke_new[1] + tke_new[0])
        lb = KAPPA * (0.5 * dz[0] + self.z0_bed)
        flux_bot = -RN * cm0 ** (RP + 1) * KAPPA * SIG_PSI_INV * kb ** (RM + 0.5) * lb**RN
        psi_new = self.eps.copy()
        psi_new[-1] = np.maximum(self.psi_min, cm0**RP * ls**RN * ks**RM)
        psi_new[0] = np.maximum(self.psi_min, cm0**RP * lb**RN * tke_new[0] ** RM)
        psi_new[1:-1] = self._solve(self.eps, BETA1 * psi_old / k_old * prod, BETA2 * diss / k_old, SIG_PSI_INV,
                                    akv_old, dz, dt, flux_bot, flux_top, self.psi_min)

        # --- viscosity from the stability functions (N² = 0)
        k = tke_new[1:-1]
        l_lim = self.galp * np.sqrt(2.0 * k) / np.sqrt(1e-14)                # Galperin limit (no stratification)
        psi = np.maximum(psi_new[1:-1], cm0**RP * l_lim**RN * k**RM)
        psi_new[1:-1] = psi
        eps = np.maximum(cm0**E1 * k**E2 * psi**E3, self.eps_min)
        cff = (k / eps) ** 2
        alpha_m = cff * shear2
        alpha_n = np.minimum(np.maximum(0.73 * self.alpha_n_min, 0.0), 1e10) + 0 * alpha_m
        am_max = ((LIM_AM0 + LIM_AM1 * alpha_n + LIM_AM2 * alpha_n**2 + LIM_AM3 * alpha_n**3)
                  / (LIM_AM4 + LIM_AM5 * alpha_n + LIM_AM6 * alpha_n**2))
        alpha_m = np.minimum(alpha_m, am_max)
        denom = (SF_D0 + SF_D1 * alpha_n + SF_D2 * alpha_m + SF_D3 * alpha_n * alpha_m + SF_D4 * alpha_n**2
                 + SF_D5 * alpha_m**2)
        c_mu = (SF_N0 + SF_N1 * alpha_n + SF_N2 * alpha_m) / denom
        akv = np.empty_like(self.akv)
        akv[1:-1] = np.maximum(c_mu * k**2 / eps, self.nu_min)
        akv[-1] = np.maximum(1.5 * akv[-2] - 0.5 * akv[-3], self.nu_min) if self.n > 2 else akv[-2]
        akv[0] = np.maximum(1.5 * akv[1] - 0.5 * akv[2], self.nu_min) if self.n > 2 else akv[1]
        self.tke, self.eps, self.akv = tke_new, psi_new, akv
        return akv
