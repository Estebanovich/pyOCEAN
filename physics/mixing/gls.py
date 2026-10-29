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
P = ν S², S² = (∂u/∂z)² + (∂v/∂z)²; B = −κ_t N², N² = −(g/ρ₀) ∂ρ/∂z.
Viscosity ν = c_μ(α_M, α_N) k²/ε and tracer diffusivity κ_t = c_μ′(α_M, α_N) k²/ε, α_M = (k/ε)² S²,
α_N = (k/ε)² N².

Two branches (N2a task 8c):

* ``stratified=False`` (default of the class, the pre-8c port, kept unchanged bit for bit): homogeneous water,
  B = 0, α_N = 0 and the Galperin limit with N² = 1e-14 (CROCO's floor ``eps`` of an unstratified column).
* ``stratified=True``: the full stratified k-ε of CROCO ``gls_mixing.F`` (GLS_KEPSILON + CANUTO_A, croco
  OCEAN/gls_mixing.F; line numbers of the local copy ~/pyOCEAN_refs/croco):
    - N² at the interior interfaces from the model density, ``buoyancy_frequency2`` (rho_eos.F:404-411);
    - B = −κ_t′ N² with κ_t′ the diffusivity of the previous step (Akt_old, pre_step3d.F:1091-1099;
      gls_mixing.F:639-640); in the ψ equation (ψ/k) β₃ B with β₃ = β₃⁻ = −0.4 for N² > 0 (stable) and
      β₃⁺ = 1 for N² < 0 (gls_mixing.F:179, 618-619, 639-640);
    - Patankar split (gls_mixing.F:657-665): if P + B > 0 both are explicit sources, otherwise P is explicit and
      B is linearised implicitly, −B/k · k (k equation) or −B/ψ · ψ (ψ equation), so k, ψ stay positive;
    - Galperin l ≤ 0.53 √(2k)/N with N² ≥ 1e-14, applied as a floor on ψ (gls_mixing.F:829-834);
    - α_N = (k/ε)² N² clipped to [0.73 α_N,min, 1e10] and α_M ≤ α_M,max(α_N) (gls_mixing.F:849-860);
    - c_μ, c_μ′ of Canuto et al. (2001) A (gls_mixing.F:863-869, gls_stab_func.h:100-148);
    - ν = max(c_μ k²/ε, ν_min = 1e-5), κ_t = max(c_μ′ k²/ε, κ_min) with κ_min = 1e-6 (CROCO ``nuwm``, ``nuws``
      of a non-3D GLS, gls_mixing.F:151-153, 872-880); bed and surface values extrapolated linearly
      (gls_mixing.F:891-905).  CROCO's Akt_bak (read_inp.F:2463-2471) is not read with GLS_MIXING: the background
      is the floor nuws.  CROCO also uses one Akt for every tracer (T, S and passive: t3dmix_tridiagonal.h:73-75).

  Not ported: horizontal and vertical advection of k and ψ (GLS_HADV_UP1, GLS_VADV_UP1), the 9-point smoothing
  (PSI_SMOOTH, TKE_SMOOTH), 3-D shear, wave forcing — as in the unstratified branch.

The steady-state Richardson number of this k-ε + Canuto A combination, ``steady_state_richardson`` (Umlauf &
Burchard 2003, eq. 28), is 0.302 with CROCO's β₃⁻ = −0.4; UB03 Table 8 gives β₃⁻ = −0.629 for Ri_st = 0.25.

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
BETA3M, BETA3P = -0.4, 1.0                      # stable / unstable N² (gls_mixing.F:179)
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
SF_NB2 = (9.0 * A1 * AB3 * (AB1 - AB2) * NN**2
          + (6.0 * A1 * (A2 - 3.0 * A3) - 4.0 * (A2**2 - 3.0 * A3**2)) * AB3 * NN * NB)   # gls_stab_func.h:141
KT_MIN = 1e-6                                    # CROCO nuws, background (floor) diffusivity (gls_mixing.F:152)
N2_FLOOR = 1e-14                                 # CROCO eps, min stratification in the Galperin limit (l. 151)
LIM_AM0, LIM_AM1 = SF_D0 * SF_N0, SF_D0 * SF_N1 + SF_D1 * SF_N0
LIM_AM2, LIM_AM3 = SF_D1 * SF_N1 + SF_D4 * SF_N0, SF_D4 * SF_N1
LIM_AM4, LIM_AM5, LIM_AM6 = SF_D2 * SF_N0, SF_D2 * SF_N1 + SF_D3 * SF_N0, SF_D3 * SF_N1


def buoyancy_frequency2(rho: np.ndarray, dz: np.ndarray, rho0: float, g: float = G) -> np.ndarray:
    """
    Squared buoyancy frequency at the interior layer interfaces [s⁻²]:

        N²_{k} = −(g/ρ₀) ∂ρ/∂z = (g/ρ₀) (ρ_{k−1} − ρ_k) / h_k,   h_k = ½ (Δz_{k−1} + Δz_k)

    (CROCO rho_eos.F:404-411, linear EOS branch; interface k between layers k−1 and k, k = 0 at the bed).
    rho : (N, M) in-situ density of the layers [kg m⁻³] (the model EOS, ``LinearEOS.density(T, S)``);
    dz : (N, M) layer thickness [m]; rho0 : reference density ρ₀ [kg m⁻³].  Returns (N − 1, M).
    """
    h = 0.5 * (dz[:-1] + dz[1:])
    return (g / rho0) * (rho[:-1] - rho[1:]) / h


def stability_functions(alpha_m: np.ndarray, alpha_n: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Canuto et al. (2001) A stability functions c_μ(α_M, α_N), c_μ′(α_M, α_N) of the momentum and tracer
    diffusivities (CROCO gls_mixing.F:863-869, gls_stab_func.h:100-148; Umlauf & Burchard 2005), without the
    α limits of the closure.
    """
    denom = (SF_D0 + SF_D1 * alpha_n + SF_D2 * alpha_m + SF_D3 * alpha_n * alpha_m + SF_D4 * alpha_n**2
             + SF_D5 * alpha_m**2)
    return (SF_N0 + SF_N1 * alpha_n + SF_N2 * alpha_m) / denom, (SF_NB0 + SF_NB1 * alpha_n + SF_NB2 * alpha_m) / denom


def steady_state_richardson(beta3: float = BETA3M) -> float:
    """
    Steady-state Richardson number Ri_st = N²/S² of homogeneous stratified shear flow for this closure (Umlauf &
    Burchard 2003, §3d, eq. 28):

        Ri_st = (c_μ / c_μ′) (c_ψ2 − c_ψ1) / (c_ψ2 − c_ψ3)

    with c_μ, c_μ′ the Canuto A stability functions at the equilibrium P + B = ε, i.e. c_μ α_M − c_μ′ α_N = 1
    with α_N = Ri α_M (UB03 eqs. 8, 10, 19).  k-ε: c_ψ1 = β₁ = 1.44, c_ψ2 = β₂ = 1.92, c_ψ3 = ``beta3``.
    UB03 Table 8 gives c_ψ3 = −0.629 for Ri_st = 0.25 with the k-ε (R87) model and Canuto A; CROCO's
    β₃⁻ = −0.4 (gls_mixing.F:179) gives a larger Ri_st.
    """
    from scipy.optimize import brentq

    ratio = (BETA2 - BETA1) / (BETA2 - beta3)

    def alpha_m_eq(ri: float) -> float:
        def f(am: float) -> float:
            cm, cmp = stability_functions(am, ri * am)
            return cm * am - cmp * ri * am - 1.0
        return brentq(f, 1e-6, 1e4, xtol=1e-14)

    def g(ri: float) -> float:
        am = alpha_m_eq(ri)
        cm, cmp = stability_functions(am, ri * am)
        return ri - cm / cmp * ratio

    return float(brentq(g, 1e-3, 0.5, xtol=1e-14))     # equilibrium exists for Ri ≲ 0.6


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
    ``tke`` k [m² s⁻²], ``eps`` ε [m² s⁻³], ``akv`` ν [m² s⁻¹]; with ``stratified=True`` also ``akt`` κ_t
    [m² s⁻¹], the tracer diffusivity, which is part of the state (B of the next step uses it, CROCO Akt_old).

    stratified : False (default) — the homogeneous pre-8c port, bit for bit; True — buoyancy (module docstring),
        ``step`` then needs ``n2``.
    kt_min : floor (background) of κ_t [m² s⁻¹], default CROCO's nuws = 1e-6 (gls_mixing.F:152).
    """

    tke_min, eps_min, nu_min = 1e-10, 1e-12, 1e-5          # CROCO gls_mixing.F (non-3D values)
    galp, z0s_min, z0b_min, chk = 0.53, 1e-2, 1e-4, 1400.0 / G

    def __init__(self, n_layers: int, n_columns: int, z0_bed: float = 1e-4, stratified: bool = False,
                 kt_min: float = KT_MIN) -> None:
        self.n = n_layers
        self.stratified, self.kt_min = bool(stratified), float(kt_min)
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
        if self.stratified:
            self.akt = np.full(shape, self.kt_min)

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
             ustar2_sfc: np.ndarray | None = None, n2: np.ndarray | None = None) -> np.ndarray:
        """
        Advance k and ε one step and return the new viscosity ν (N+1, M).

        u, v : (N, M) layer velocities at the column [m s⁻¹]; dz : (N, M) layer thickness [m]
        ustar2_bot, ustar2_sfc : |τ|/ρ₀ at the bed and surface [m² s⁻²]
        n2 : (N−1, M) N² at the interior interfaces [s⁻²] (``buoyancy_frequency2``); used, and required, only
            with ``stratified=True`` (then also ``akt`` is updated)
        """
        if self.stratified:
            if n2 is None:
                raise ValueError("GLSKEpsilon(stratified=True).step needs n2, the N² at the interior interfaces")
            return self._step_stratified(u, v, dz, ustar2_bot, dt, ustar2_sfc, n2)
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

    def _step_stratified(self, u: np.ndarray, v: np.ndarray, dz: np.ndarray, ustar2_bot: np.ndarray, dt: float,
                         ustar2_sfc: np.ndarray | None, n2: np.ndarray) -> np.ndarray:
        """
        One step of the stratified k-ε (CROCO gls_mixing.F, module docstring); the same sequence and, where B = 0,
        the same arithmetic as the homogeneous branch, so N² = 0 reproduces it.

        k:  (k′ − k)/Δt = ∂z(ν/σ_k ∂z k′) + P + B − ε′          (UB03 eq. 11)
        ψ:  (ψ′ − ψ)/Δt = ∂z(ν/σ_ψ ∂z ψ′) + (ψ/k)(β₁ P + β₃ B − β₂ ε′)   (UB03 eq. 12)

        P = ν S², B = −κ_t N² (ν, κ_t of the previous step; gls_mixing.F:621, 639-640).  If P + B > 0 the sources
        are explicit; otherwise P is explicit and B enters the diagonal as −B/k (k) or −B/ψ (ψ), Patankar (1980)
        (gls_mixing.F:657-665).  β₃ = β₃⁻ = −0.4 where N² > 0, β₃⁺ = 1 where N² < 0 (gls_mixing.F:618-619).
        """
        cm0 = self.cm0
        us2s = np.zeros(dz.shape[1]) if ustar2_sfc is None else ustar2_sfc
        h = 0.5 * (dz[:-1] + dz[1:])
        shear2 = ((u[1:] - u[:-1]) ** 2 + (v[1:] - v[:-1]) ** 2) / h**2      # (N−1, M)
        n2 = np.nan_to_num(np.asarray(n2, float))                           # dry/wall columns: no stratification
        k_old, psi_old, akv_old, akt_old = self.tke[1:-1], self.eps[1:-1], self.akv, self.akt
        diss = np.maximum(cm0**E1 * k_old**E2 * psi_old**E3, self.eps_min)
        prod = akv_old[1:-1] * shear2
        n2_pos, n2_neg = np.maximum(n2, 0.0), np.minimum(n2, 0.0)

        # --- k: B = −κ_t N² (cff3m = cff3p = 1 for k), gls_mixing.F:639-640
        buoy = -akt_old[1:-1] * (n2_pos + n2_neg)
        pos = (buoy + prod) > 0.0
        sprod = np.where(pos, prod + buoy, prod)
        dcoef = np.where(pos, diss / k_old, diss / k_old - buoy / k_old)
        tke_new = self.tke.copy()
        tke_new[0] = np.maximum(self.tke_min, ustar2_bot / cm0**2)
        tke_new[-1] = np.maximum(self.tke_min, us2s / cm0**2)
        zero = np.zeros(dz.shape[1])
        tke_new[1:-1] = self._solve(self.tke, sprod, dcoef, SIG_K_INV, akv_old, dz, dt, zero, zero, self.tke_min)

        # --- ψ: (ψ/k)(β₁ P + β₃ B), the B part implicit as −B_ψ/ψ when P + B ≤ 0 (gls_mixing.F:616-665)
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
        sprod_psi = BETA1 * psi_old / k_old * prod
        buoy_psi = -akt_old[1:-1] * (psi_old / k_old) * (BETA3M * n2_pos + BETA3P * n2_neg)
        pos = (buoy_psi + sprod_psi) > 0.0
        src_psi = np.where(pos, sprod_psi + buoy_psi, sprod_psi)
        dco_psi = np.where(pos, BETA2 * diss / k_old, BETA2 * diss / k_old - buoy_psi / psi_old)
        psi_new[1:-1] = self._solve(self.eps, src_psi, dco_psi, SIG_PSI_INV, akv_old, dz, dt, flux_bot, flux_top,
                                    self.psi_min)

        # --- Galperin limit, α_N, α_M, stability functions, ν and κ_t (gls_mixing.F:825-905)
        k = tke_new[1:-1]
        l_lim = self.galp * np.sqrt(2.0 * k) / np.sqrt(np.maximum(N2_FLOOR, n2))
        psi = np.maximum(psi_new[1:-1], cm0**RP * l_lim**RN * k**RM)
        psi_new[1:-1] = psi
        eps = np.maximum(cm0**E1 * k**E2 * psi**E3, self.eps_min)
        cff = (k / eps) ** 2
        alpha_m = cff * shear2
        alpha_n = np.minimum(np.maximum(0.73 * self.alpha_n_min, cff * n2), 1e10)
        am_max = ((LIM_AM0 + LIM_AM1 * alpha_n + LIM_AM2 * alpha_n**2 + LIM_AM3 * alpha_n**3)
                  / (LIM_AM4 + LIM_AM5 * alpha_n + LIM_AM6 * alpha_n**2))
        alpha_m = np.minimum(alpha_m, am_max)
        denom = (SF_D0 + SF_D1 * alpha_n + SF_D2 * alpha_m + SF_D3 * alpha_n * alpha_m + SF_D4 * alpha_n**2
                 + SF_D5 * alpha_m**2)
        c_mu = (SF_N0 + SF_N1 * alpha_n + SF_N2 * alpha_m) / denom
        c_mu_prim = (SF_NB0 + SF_NB1 * alpha_n + SF_NB2 * alpha_m) / denom
        akv = np.empty_like(self.akv)
        akv[1:-1] = np.maximum(c_mu * k**2 / eps, self.nu_min)
        akv[-1] = np.maximum(1.5 * akv[-2] - 0.5 * akv[-3], self.nu_min) if self.n > 2 else akv[-2]
        akv[0] = np.maximum(1.5 * akv[1] - 0.5 * akv[2], self.nu_min) if self.n > 2 else akv[1]
        akt = np.empty_like(self.akt)
        akt[1:-1] = np.maximum(c_mu_prim * k**2 / eps, self.kt_min)
        akt[-1] = np.maximum(1.5 * akt[-2] - 0.5 * akt[-3], self.kt_min) if self.n > 2 else akt[-2]
        akt[0] = np.maximum(1.5 * akt[1] - 0.5 * akt[2], self.kt_min) if self.n > 2 else akt[1]
        self.tke, self.eps, self.akv, self.akt = tke_new, psi_new, akv, akt
        return akv
