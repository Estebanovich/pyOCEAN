"""
KPP (K-Profile Parameterization) vertical mixing for pyOCEAN — Phase 1.

Implements the Large, McWilliams & Doney (1994) KPP scheme.  The scheme
has two physically distinct components:

Ocean Boundary Layer (OBL)
--------------------------
The depth h_bl of the actively mixing surface boundary layer is diagnosed
from a bulk Richardson number:

    Ri_b(d) = [B_r − B(d)] × d / [ΔU²(d) + Vt²(d)]     (L94, eq. 21)

where d = |z| is depth below the surface, B = −(g/ρ₀)(ρ − ρ_r) is
buoyancy relative to the surface reference, ΔU² is the resolved
velocity-shear squared, and Vt² = C_v × d × N × w_s / Ri_cr is the
unresolved turbulent shear contribution (L94, eq. 23).

h_bl is the depth where Ri_b first equals the critical value Ri_cr = 0.3,
found by linear interpolation between grid levels (MITgcm BLDEPTH approach).

Inside the OBL the diffusivity follows a smooth cubic profile matched to
the interior at the OBL base:

    K_x(σ) = h_bl × w_x × G(σ)                           (L94, eq. 14)
    G(σ)   = a₁ σ² + a₂ σ³

with σ = d/h_bl ∈ [0, 1].  The coefficients a₁, a₂ enforce G(0) = 0,
G'(0) = 0 (surface), and G(1) = K_x_int/(h_bl×w_x) ≡ gat1 (matching
interior diffusivity at the OBL base; HYCOM mxkprf.F90 / MITgcm BLMIX).

Phase 1 simplification: the slope-matching condition G'(1) = dat1 is set
to zero (smooth transition), equivalent to a₁ = 3×gat1, a₂ = −2×gat1.
Full derivative matching is a Phase 2 enhancement.

Turbulent velocity scales w_m and w_s (momentum / scalar) are computed
from Monin-Obukhov similarity theory (L94, Appendix B):

    ζ ≥ 0 (stable):    w_x = κ u* / (1 + 5ζ)
    ζ < 0 (unstable):  w_m = κ u* (1 − 16ζ)^(1/4)
                        w_s = κ u* (1 − 16ζ)^(1/2)

where ζ = −κ d b_f / u*³ and u* is the surface friction velocity.

Interior mixing (below h_bl)
-----------------------------
Gradient Richardson number at each interface:

    Ri_g(k+½) = N²(k+½) / S²(k+½)                        (L94, eq. 25)

Richardson-number mixing function:

    f(Ri_g) = (1 − min(Ri_g / Ri₀, 1)²)³                  (L94, eq. 28)

Diffusivities:

    K_m = ν₀ × f(Ri_g) + K_bg   [+ K_conv if N² < 0]
    K_s = κ₀ × f(Ri_g) + K_bg   [+ K_conv if N² < 0]

Non-local transport
-------------------
A counter-gradient scalar flux (L94, eq. 21–22) is included for T and S
in unstable conditions (cooling; the helpers take b_flux = −B_f > 0):

    γ_s = c_g × <w'θ'>_sfc / (w_s × h_bl)

It is zero when surface fluxes are not supplied (quiescent Phase 1 runs).

Sign conventions
----------------
z increases upward, z = 0 at the free surface, z < 0 in the ocean interior.
Layer index k increases downward: k = 0 is the surface layer.
Interface index k_f runs from 0 (surface interface) to nz (bottom interface).

References
----------
Large, W.G., McWilliams, J.C. & Doney, S.C. (1994).
    Oceanic vertical mixing: a review and a model with a nonlocal
    boundary layer parameterization.  Rev. Geophys. 32(4), 363–403.
    https://doi.org/10.1029/94RG01872  [cited as L94]

MITgcm Team (2023). MITgcm manual: KPP package.
    Source: MITgcm/pkg/kpp/kpp_routines.F  (BLDEPTH, BLMIX, Ri_iwmix)

HYCOM Consortium. HYCOM source: HYCOM-src/mxkprf.F90, inikpp.F90.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.state import ModelState

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G: float = 9.81       # m/s²   gravitational acceleration
RHO_0: float = 1025.0 # kg/m³  Boussinesq reference density
CP: float = 3996.0    # J/(kg·K) specific heat of seawater

# ---------------------------------------------------------------------------
# KPP scheme parameters (Large et al. 1994, Table 1)
# ---------------------------------------------------------------------------

KAPPA: float = 0.4       # —      von Kármán constant
RI_CR: float = 0.3       # —      critical bulk Richardson number         (eq. 21)
RI_0: float = 0.7        # —      interior Ri limit for shear mixing       (eq. 28)
C_V: float = 1.8         # —      unresolved turbulent shear coefficient   (eq. 23)
C_G: float = 0.416       # —      non-local counter-gradient coefficient   (eq. 22)
EPSILON: float = 0.1     # —      surface layer fraction                   (eq. 19)
CONC1: float = 5.0       # —      stable boundary correction               (App B)
NU_0: float = 1.0e-3     # m²/s   shear-instability viscosity              (eq. 28)
KAPPA_S0: float = 1.0e-4 # m²/s   shear-instability scalar diffusivity     (eq. 30)
K_BG: float = 1.0e-5     # m²/s   background internal-wave mixing
K_CONV: float = 1.0e-2   # m²/s   convective adjustment (N² < 0)
EPS: float = 1.0e-10     # —      small number to prevent division by zero


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _u_to_t(u_np: np.ndarray) -> np.ndarray:
    """
    Interpolate u (nz, ny, nx+1) from U-faces to T-cell centres (nz, ny, nx).

    Simple 2-point average of the two bounding U-faces:

        u_t[k, j, i] = ½ (u[k, j, i] + u[k, j, i+1])
    """
    return 0.5 * (u_np[:, :, :-1] + u_np[:, :, 1:])


def _v_to_t(v_np: np.ndarray) -> np.ndarray:
    """
    Interpolate v (nz, ny+1, nx) from V-faces to T-cell centres (nz, ny, nx).

        v_t[k, j, i] = ½ (v[k, j, i] + v[k, j+1, i])
    """
    return 0.5 * (v_np[:, :-1, :] + v_np[:, 1:, :])


def _velocity_scales(
    u_star: np.ndarray,
    b_flux: np.ndarray,
    depth: np.ndarray | float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turbulent velocity scales w_m (momentum) and w_s (scalar) from
    Monin-Obukhov similarity theory (Large et al. 1994, Appendix B).

    Stability parameter:
        ζ = −κ d b_f / u*³                          (L94 App B, eq. B1)

    Stable (ζ ≥ 0):
        φ_m = φ_s = 1 + 5ζ    →  w = κ u* / φ

    Weakly–strongly unstable (ζ < 0):
        φ_m = (1 − 16ζ)^(−1/4)   →  w_m = κ u* (1 − 16ζ)^(1/4)
        φ_s = (1 − 16ζ)^(−1/2)   →  w_s = κ u* (1 − 16ζ)^(1/2)

    This analytic form captures the MO profiles without the lookup tables
    used in HYCOM/MITgcm (which extend into the strongly convective regime).
    For the strongly convective regime (ζ ≪ −1) the expressions are bounded
    above by max_w = 0.5 m/s to prevent numerical blow-up.

    Parameters
    ----------
    u_star : np.ndarray, shape (ny, nx) or scalar
        Friction velocity [m/s], u* = sqrt(|τ|/ρ₀).
    b_flux : np.ndarray, shape (ny, nx) or scalar
        Surface buoyancy flux [m²/s³].  Positive → destabilising (cooling; kpp_tendency passes −B_f).
    depth : np.ndarray (ny, nx) or float
        Depth(s) below surface [m] at which to evaluate the scales.

    Returns
    -------
    w_m, w_s : np.ndarray, shape (ny, nx)
        Turbulent velocity scales [m/s].
    """
    # ζ = −κ d b_f / u*³   (negative = unstable, positive = stable)
    zeta = -KAPPA * depth * b_flux / (u_star ** 3 + EPS)

    # Stable branch: φ = 1 + 5ζ  →  w = κ u* / (1 + 5ζ)
    phi_stable = 1.0 + 5.0 * np.maximum(zeta, 0.0)

    # Unstable branch: φ_m = (1 − 16ζ)^(−1/4), φ_s = (1 − 16ζ)^(−1/2)
    zeta_neg = np.minimum(zeta, 0.0)          # clip to ≤ 0
    factor = np.maximum(1.0 - 16.0 * zeta_neg, 1.0)   # ≥ 1

    w_m_unstable = KAPPA * u_star * factor ** 0.25
    w_s_unstable = KAPPA * u_star * factor ** 0.5

    w_m_stable = KAPPA * u_star / np.maximum(phi_stable, EPS)
    w_s_stable = w_m_stable   # same for stable stratification

    stable = zeta >= 0.0
    w_m = np.where(stable, w_m_stable, w_m_unstable)
    w_s = np.where(stable, w_s_stable, w_s_unstable)

    # Bound to prevent numerical blow-up in strongly convective conditions
    w_m = np.clip(w_m, EPS, 0.5)
    w_s = np.clip(w_s, EPS, 0.5)

    return w_m, w_s


# ---------------------------------------------------------------------------
# Public functions
# ---------------------------------------------------------------------------

def obl_depth(
    rho: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
    z_t: np.ndarray,
    u_star: np.ndarray,
    b_flux: np.ndarray,
    *,
    ri_cr: float = RI_CR,
) -> np.ndarray:
    """
    Diagnose the Ocean Boundary Layer depth h_bl via the bulk Richardson
    number criterion (Large et al. 1994, §3.1 and eq. 21).

    The bulk Richardson number at depth d = |z| is:

        Ri_b(d) = [B_r − B(d)] × d / [ΔU²(d) + Vt²(d)]    (L94, eq. 21)

    where
        B(d)   = −(g/ρ₀)(ρ(d) − ρ_r)         buoyancy (relative to surface)
        ΔU²(d) = (u(d)−u_r)² + (v(d)−v_r)²   resolved velocity shear
        Vt²(d) = C_v × d × N(d) × w_s(d) / Ri_cr  unresolved turbulent shear
        subscript r denotes the surface-layer reference (k = 0).

    h_bl is the depth where Ri_b first equals ri_cr, found by linear
    interpolation between adjacent grid-level centres.

    Parameters
    ----------
    rho : np.ndarray, shape (nz, ny, nx)
        In-situ density [kg/m³] at T-cell centres.
    u_t : np.ndarray, shape (nz, ny, nx)
        Zonal velocity at T-cell centres [m/s].
    v_t : np.ndarray, shape (nz, ny, nx)
        Meridional velocity at T-cell centres [m/s].
    z_t : np.ndarray, shape (nz,)
        Layer-centre depths [m], z_t[0] < 0.  (z < 0 in ocean interior.)
    u_star : np.ndarray, shape (ny, nx)
        Surface friction velocity [m/s].  Use zeros for quiescent runs.
    b_flux : np.ndarray, shape (ny, nx)
        Surface buoyancy flux [m²/s³].  Positive = destabilising.
    ri_cr : float, optional
        Critical bulk Richardson number. Default 0.3.

    Returns
    -------
    h_bl : np.ndarray, shape (ny, nx)
        OBL depth [m], positive.  Equals |z_t[-1]| when Ri_b < ri_cr
        throughout the full column (deeply mixed state).
    """
    nz, ny, nx = rho.shape
    d = -z_t   # depth below surface [m], d > 0, shape (nz,)

    # --- Surface-layer reference (k = 0) ------------------------------------
    rho_r = rho[0]   # (ny, nx)
    u_r   = u_t[0]   # (ny, nx)
    v_r   = v_t[0]   # (ny, nx)

    # --- Buoyancy difference: B_r − B(k) = (g/ρ₀)(ρ(k) − ρ_r) ------------
    # For stable stratification: ρ(k) > ρ_r → delta_B > 0 → Ri_b > 0 ✓
    delta_B = (G / RHO_0) * (rho - rho_r[np.newaxis, :, :])   # (nz, ny, nx)

    # --- Resolved velocity shear squared ΔU²(k) ----------------------------
    du = u_t - u_r[np.newaxis, :, :]   # (nz, ny, nx)
    dv = v_t - v_r[np.newaxis, :, :]   # (nz, ny, nx)
    shear2 = du ** 2 + dv ** 2          # (nz, ny, nx)

    # --- Brunt-Väisälä frequency N(k) at cell centres ----------------------
    # N²(k) estimated from finite difference with neighbour below.
    # N²(k) = −(g/ρ₀) × (ρ(k+1) − ρ(k)) / (z_t(k+1) − z_t(k))
    # z_t(k+1) < z_t(k), so denominator is negative → N² > 0 for stable column.
    # N²(k) = (g/ρ₀)(ρ[k+1] − ρ[k]) / dz_c[k]
    # dz_c[k] = z_t[k] − z_t[k+1] > 0 (distance between centres)
    # N² > 0 for stable (ρ[k+1] > ρ[k], denser below)
    N2 = np.zeros((nz, ny, nx))
    dz_c = z_t[:-1] - z_t[1:]   # > 0, shape (nz-1,)
    N2[:-1] = (G / RHO_0) * (rho[1:] - rho[:-1]) / dz_c[:, np.newaxis, np.newaxis]
    N2[-1] = N2[-2]              # extrapolate at bottom layer
    N = np.sqrt(np.maximum(N2, 0.0))   # (nz, ny, nx)

    # --- Unresolved turbulent shear Vt²(k) (L94, eq. 23) ------------------
    # Vt²(d) = C_v × d × N(d) × w_s(d) / Ri_cr
    # w_s evaluated at each depth d[k] (varies per column via h_bl, but we
    # have not yet found h_bl, so we use the current depth as the reference).
    Vt2 = np.zeros((nz, ny, nx))
    for k in range(nz):
        dk = d[k]
        if dk <= 0.0:
            continue
        _, w_s_k = _velocity_scales(u_star, b_flux, dk)   # (ny, nx)
        Vt2[k] = C_V * dk * N[k] * w_s_k / ri_cr          # (ny, nx)

    # --- Bulk Richardson number Ri_b(k) ------------------------------------
    # Ri_b(k) = delta_B(k) × d(k) / (shear2(k) + Vt2(k) + ε)
    Ri_b = (
        delta_B * d[:, np.newaxis, np.newaxis]
    ) / (shear2 + Vt2 + EPS)   # (nz, ny, nx)
    # k = 0: delta_B = 0, shear = 0  →  Ri_b = 0 (surface, always below ri_cr)

    # --- Find h_bl by linear interpolation ---------------------------------
    # Scan downward; take the first level where Ri_b crosses ri_cr.
    h_bl = np.full((ny, nx), d[-1])           # default: full column
    found = np.zeros((ny, nx), dtype=bool)

    for k in range(1, nz):
        # Columns where Ri_b first exceeds ri_cr at layer k (not yet found)
        crossed = (Ri_b[k] >= ri_cr) & (~found)
        if not np.any(crossed):
            continue
        found |= crossed
        # Linear interpolation:  h_bl = d[k-1] + (ri_cr − Ri_b[k-1]) /
        #                                         (Ri_b[k] − Ri_b[k-1]) × (d[k] − d[k-1])
        dRi = Ri_b[k] - Ri_b[k - 1]
        frac = (ri_cr - Ri_b[k - 1]) / np.where(np.abs(dRi) > EPS, dRi, EPS)
        frac = np.clip(frac, 0.0, 1.0)
        h_interp = d[k - 1] + frac * (d[k] - d[k - 1])
        h_bl = np.where(crossed, h_interp, h_bl)

    h_bl = np.clip(h_bl, d[0], d[-1])
    return h_bl   # (ny, nx)


def interior_diffusivity(
    rho: np.ndarray,
    u_t: np.ndarray,
    v_t: np.ndarray,
    dz: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute vertical diffusivities from local gradient Richardson number
    and background internal-wave mixing (Large et al. 1994, §3.2).

    At each interior interface k_f (between layers k_f−1 and k_f), the
    gradient Richardson number is:

        Ri_g = N²(k_f) / S²(k_f)                          (L94, eq. 25)

    where
        N²(k_f) = −(g/ρ₀) × (ρ_below − ρ_above) / Δz_f   (Brunt-Väisälä ²)
        S²(k_f) = [(Δu/Δz)² + (Δv/Δz)²]                  (shear ²)
        Δz_f    = ½(dz[k_f−1] + dz[k_f])                  (interface spacing)

    Richardson-number mixing function (L94, eq. 28):

        f(Ri_g) = (1 − min(Ri_g/Ri₀, 1)²)³

    Diffusivities at each interface:

        K_m = ν₀  × f(Ri_g) + K_bg   [+ K_conv if N² < 0]  (L94, eq. 28)
        K_s = κ₀  × f(Ri_g) + K_bg   [+ K_conv if N² < 0]  (L94, eq. 30)

    K_conv = 10⁻² m²/s is applied as a convective adjustment when N² < 0
    (gravitationally unstable layer), following MITgcm's convAdjFreq
    approach (kpp_routines.F, Ri_iwmix).

    Parameters
    ----------
    rho : np.ndarray, shape (nz, ny, nx)
        In-situ density [kg/m³].
    u_t : np.ndarray, shape (nz, ny, nx)
        Zonal velocity at T-cell centres [m/s].
    v_t : np.ndarray, shape (nz, ny, nx)
        Meridional velocity at T-cell centres [m/s].
    dz : np.ndarray, shape (nz,)
        Layer thicknesses [m].

    Returns
    -------
    K_m, K_s : np.ndarray, shape (nz+1, ny, nx)
        Momentum viscosity and scalar diffusivity at layer interfaces [m²/s].
        K[0] = K[nz] = 0 (surface and bottom BCs; flux specified externally).
    """
    nz, ny, nx = rho.shape

    K_m = np.zeros((nz + 1, ny, nx))
    K_s = np.zeros((nz + 1, ny, nx))

    # Interior interfaces k_f = 1 .. nz-1
    for k_f in range(1, nz):
        k_a = k_f - 1       # layer above
        k_b = k_f           # layer below
        dz_f = 0.5 * (dz[k_a] + dz[k_b])   # distance between centres [m]

        # N² at interface (L94, eq. 25)
        # N² = −(g/ρ₀) ∂ρ/∂z,  ∂ρ/∂z = (ρ_above − ρ_below)/(z_above − z_below)
        # z_above > z_below  →  z_above − z_below ≈ +dz_f
        # ∂ρ/∂z = (ρ[k_a] − ρ[k_b]) / dz_f  →  N² = (g/ρ₀)(ρ[k_b] − ρ[k_a]) / dz_f
        # N² > 0 → stably stratified (dense below); N² < 0 → unstable
        drho = rho[k_b] - rho[k_a]            # positive if denser below (stable)
        N2 = (G / RHO_0) * drho / dz_f        # (ny, nx)

        # Shear squared at interface
        du = (u_t[k_b] - u_t[k_a]) / dz_f    # (ny, nx)
        dv = (v_t[k_b] - v_t[k_a]) / dz_f    # (ny, nx)
        S2 = du ** 2 + dv ** 2                 # (ny, nx)

        # Gradient Richardson number (large where shear → 0)
        Ri_g = N2 / (S2 + EPS)                 # (ny, nx)

        # Richardson-number mixing function f(Ri_g) = (1 − (Ri_g/Ri₀)²)³
        ratio = np.clip(Ri_g / RI_0, 0.0, 1.0)
        f_Ri = (1.0 - ratio ** 2) ** 3          # (ny, nx), in [0, 1]

        # Convective adjustment: large K where N² < 0
        K_c = np.where(N2 < 0.0, K_CONV, 0.0)  # (ny, nx)

        K_m[k_f] = NU_0 * f_Ri + K_BG + K_c
        K_s[k_f] = KAPPA_S0 * f_Ri + K_BG + K_c

    # k_f = 0 (surface) and k_f = nz (bottom): fluxes from BL / no-flux BC
    # already zero from np.zeros initialisation.
    return K_m, K_s   # (nz+1, ny, nx)


def _blmix(
    h_bl: np.ndarray,
    K_m_int: np.ndarray,
    K_s_int: np.ndarray,
    z_f: np.ndarray,
    dz: np.ndarray,
    u_star: np.ndarray,
    b_flux: np.ndarray,
    wT_sfc: np.ndarray,
    wS_sfc: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply the KPP boundary-layer mixing profile inside the OBL.

    Overwrites the interior diffusivities K_m_int, K_s_int with the BL
    profile K_x(σ) = h_bl × w_x × G(σ) for all interfaces with d ≤ h_bl.
    Returns the non-local transport coefficients γ_T and γ_S.

    Shape function (MITgcm BLMIX, HYCOM mxkprf.F90)
    -------------------------------------------------
    G(σ) = a₁ σ² + a₂ σ³

    Boundary conditions:
        G(0) = G'(0) = 0    (zero flux, zero slope at surface)
        G(1) = gat1         (match interior K at σ = 1)
        G'(1) = 0           (Phase 1: zero-slope matching; see module docstring)

    This gives:  a₁ = 3 × gat1,  a₂ = −2 × gat1.

    Non-local transport (L94, eq. 22):
        γ_T = c_g × <w'T'>_sfc / (w_s × h_bl)   [K/m]
        γ_S = c_g × <w'S'>_sfc / (w_s × h_bl)   [PSU/m]
    Active only in unstable conditions (b_flux > 0).

    Parameters
    ----------
    h_bl : (ny, nx) — OBL depth [m]
    K_m_int, K_s_int : (nz+1, ny, nx) — interior diffusivities [m²/s]
    z_f : (nz+1,) — interface depths [m], z_f[0] = 0
    dz  : (nz,)   — layer thicknesses [m]
    u_star : (ny, nx) — friction velocity [m/s]
    b_flux : (ny, nx) — surface buoyancy flux [m²/s³]
    wT_sfc : (ny, nx) — surface kinematic heat flux [K·m/s], positive upward into ocean
    wS_sfc : (ny, nx) — surface kinematic salt flux [PSU·m/s]

    Returns
    -------
    K_m_bl, K_s_bl : (nz+1, ny, nx) — combined diffusivities [m²/s]
    gamma_T, gamma_S : (nz+1, ny, nx) — non-local transport [1/m]
    """
    nz = len(dz)
    ny, nx = h_bl.shape
    d_f = -z_f   # interface depths below surface (positive), shape (nz+1,)

    # Start from interior values; override inside OBL
    K_m_bl = K_m_int.copy()
    K_s_bl = K_s_int.copy()
    gamma_T = np.zeros((nz + 1, ny, nx))
    gamma_S = np.zeros((nz + 1, ny, nx))

    # --- Turbulent velocity scales at the surface layer depth d_sl = ε × h_bl
    # (L94, eq. 19: the velocity scale is evaluated at the base of the
    # surface layer, not at every interface; one value per column.)
    d_sl = EPSILON * np.maximum(h_bl, EPS)   # (ny, nx)
    w_m_bl, w_s_bl = _velocity_scales(u_star, b_flux, d_sl)   # (ny, nx)

    # --- Interior K at the OBL base -----------------------------------------
    # For each column find k_bl: the deepest interface with d_f ≤ h_bl.
    # Using broadcasting: in_bl_3d[k_f, j, i] = True if d_f[k_f] ≤ h_bl[j,i].
    d_f_3d = d_f[:, np.newaxis, np.newaxis]                  # (nf, 1, 1)
    in_bl_3d = d_f_3d <= h_bl[np.newaxis, :, :]              # (nf, ny, nx)
    # k_bl[j,i] = index of deepest interface inside BL (count True − 1)
    k_bl = in_bl_3d.sum(axis=0) - 1                           # (ny, nx), 0 … nz
    k_bl = np.clip(k_bl, 1, nz - 1).astype(int)

    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    K_m_base = K_m_int[k_bl, jj, ii]   # K_m at OBL base interface (ny, nx)
    K_s_base = K_s_int[k_bl, jj, ii]   # (ny, nx)

    # Shape function coefficients (Phase 1: G'(1) = dat1 = 0)
    # gat1 = K_int / (h_bl × w_x)   →   a1 = 3 gat1,  a2 = −2 gat1
    gat1_m = K_m_base / (h_bl * w_m_bl + EPS)   # (ny, nx)
    gat1_s = K_s_base / (h_bl * w_s_bl + EPS)   # (ny, nx)
    a1_m = 3.0 * gat1_m
    a2_m = -2.0 * gat1_m
    a1_s = 3.0 * gat1_s
    a2_s = -2.0 * gat1_s

    # --- Non-local coefficient γ (L94, eq. 22) ------------------------------
    # γ_x = c_g × <w'x'>_sfc / (w_s × h_bl)   (active only when unstable)
    unstable = b_flux > 0.0                              # (ny, nx)
    gamma_factor = np.where(
        unstable,
        C_G / (w_s_bl * h_bl + EPS),
        0.0,
    )   # (ny, nx)

    # --- Apply BL profile to each interface inside the OBL -----------------
    for k_f in range(1, nz):   # skip surface (k_f=0) and bottom (k_f=nz)
        d_kf = d_f[k_f]        # depth of this interface [m] (scalar)
        sigma = d_kf / np.maximum(h_bl, EPS)   # σ = d/h_bl, (ny, nx)
        mask = in_bl_3d[k_f]                    # True where this iface is in BL (ny, nx)

        if not np.any(mask):
            continue

        # G(σ) = a₁ σ² + a₂ σ³
        G_m = a1_m * sigma ** 2 + a2_m * sigma ** 3   # (ny, nx)
        G_s = a1_s * sigma ** 2 + a2_s * sigma ** 3   # (ny, nx)
        G_m = np.maximum(G_m, 0.0)   # enforce non-negative diffusivity
        G_s = np.maximum(G_s, 0.0)

        K_bl_m = h_bl * w_m_bl * G_m   # K_m inside BL (ny, nx)
        K_bl_s = h_bl * w_s_bl * G_s   # K_s inside BL (ny, nx)

        # Use max of BL and interior K (MITgcm convention; HYCOM mxkprf.F90)
        K_m_bl[k_f] = np.where(mask, np.maximum(K_bl_m, K_m_int[k_f]), K_m_bl[k_f])
        K_s_bl[k_f] = np.where(mask, np.maximum(K_bl_s, K_s_int[k_f]), K_s_bl[k_f])

        # Non-local transport γ_T, γ_S at this interface
        # F_nonlocal = K_s × γ × <w'x'>_sfc  →  the tendency then uses ∂F/∂z
        gamma_T[k_f] = np.where(mask, gamma_factor * wT_sfc, 0.0)
        gamma_S[k_f] = np.where(mask, gamma_factor * wS_sfc, 0.0)

    return K_m_bl, K_s_bl, gamma_T, gamma_S


def kpp_tendency(
    state: ModelState,
    v_grid: xr.Dataset,
    *,
    tau_x: np.ndarray | None = None,
    tau_y: np.ndarray | None = None,
    Q_net: np.ndarray | None = None,
    E_P: np.ndarray | None = None,
    eos: str = "linear",
) -> ModelState:
    """
    Compute the KPP vertical mixing tendency for all prognostic variables.

    Evaluates:

        ∂T/∂t|_mix = (1/dz_k) × [F_T(k_f) − F_T(k_f+1)]      (L94, eq. 1)
        F_T(k_f)   = K_s(k_f) × [∂T/∂z + γ_T × <w'T'>_sfc]  (L94, eq. 4)

    with an identical expression for S, and analogously for u, v (with K_m,
    no non-local term).  The divergence is discretised with an explicit
    finite-difference stencil compatible with the Adams–Bashforth 2
    timestepper.

    Surface forcing workflow
    ------------------------
    Wind stress τ → friction velocity:    u* = sqrt(|τ| / ρ₀)
    Net heat flux Q_net → kinematic flux:  <w'T'>_sfc = Q_net / (ρ₀ C_p)
    Evaporation−precipitation → salt flux: <w'S'>_sfc = S_sfc × (E−P)

    In quiescent Phase 1 runs (no surface forcing), all surface fluxes are
    zero: the BL is shallow, and only interior Ri-based mixing is active.

    Parameters
    ----------
    state : ModelState
        Current model state (u, v, w, T, S, η).
    v_grid : xr.Dataset
        Vertical grid providing ``z_t``, ``z_f``, ``dz`` [m].
    tau_x : np.ndarray (ny, nx) or None
        Zonal wind stress [N/m²].  Default: zeros.
    tau_y : np.ndarray (ny, nx) or None
        Meridional wind stress [N/m²].  Default: zeros.
    Q_net : np.ndarray (ny, nx) or None
        Net surface heat flux [W/m²], positive into ocean.  Default: zeros.
    E_P : np.ndarray (ny, nx) or None
        Evaporation minus precipitation [m/s].  Default: zeros.
    eos : str, optional
        EOS used to compute density: ``"linear"`` (default) or ``"teos10"``.
        Passed to ``physics.eos.compute_density``.

    Returns
    -------
    ModelState
        Tendency dstate/dt from KPP mixing [same units per second].
        u, v → [m/s²];  T → [°C/s];  S → [PSU/s];  η, w → 0.

    Notes
    -----
    * The tendency is explicit — use a timestep Δt ≤ h²/(2K) for stability
      (typically satisfied by the CFL constraint of the advection scheme).
    * ``w`` in the returned tendency is zero; ``w`` is diagnosed from
      continuity after the full momentum step.

    References
    ----------
    Large et al. (1994), §2–3 and Appendix B.
    MITgcm kpp_routines.F: KPPMIX, BLDEPTH, BLMIX, Ri_iwmix.
    HYCOM mxkprf.F90: main KPP driver.
    """
    from physics.eos import compute_density

    # ---- Unpack grids and state --------------------------------------------
    dz  = v_grid["dz"].values    # (nz,)
    z_t = v_grid["z_t"].values   # (nz,), negative
    z_f = v_grid["z_f"].values   # (nz+1,), negative, z_f[0] = 0

    u_np  = state.u.values       # (nz, ny, nx+1)
    v_np  = state.v.values       # (nz, ny+1, nx)
    T_np  = state.T.values       # (nz, ny, nx)
    S_np  = state.S.values       # (nz, ny, nx)
    nz, ny, nx = T_np.shape

    # Velocity at T-cell centres (for Richardson numbers)
    u_t = _u_to_t(u_np)   # (nz, ny, nx)
    v_t = _v_to_t(v_np)   # (nz, ny, nx)

    # ---- In-situ density (needed for Ri_b and N²) -------------------------
    rho_da = compute_density(state, v_grid, eos=eos)   # xr.DataArray (nz, ny, nx)
    rho = rho_da.values

    # ---- Surface forcing ---------------------------------------------------
    zeros_2d = np.zeros((ny, nx))

    if tau_x is None:
        tau_x = zeros_2d
    if tau_y is None:
        tau_y = zeros_2d
    if Q_net is None:
        Q_net = zeros_2d
    if E_P is None:
        E_P = zeros_2d

    # Friction velocity: u* = sqrt(|τ| / ρ₀)     [m/s]
    tau_mag = np.sqrt(tau_x ** 2 + tau_y ** 2)
    u_star = np.sqrt(tau_mag / RHO_0)            # (ny, nx)

    # Surface kinematic heat flux: <w'T'>_sfc = Q_net / (ρ₀ Cp)  [K·m/s]
    wT_sfc = Q_net / (RHO_0 * CP)               # (ny, nx)

    # Surface kinematic salt flux: <w'S'>_sfc = S_sfc × (E−P)    [PSU·m/s]
    # Use surface salinity S(k=0) as the reference salinity.
    wS_sfc = S_np[0] * E_P                      # (ny, nx)

    # Surface buoyancy flux: B_f = g(α_T <w'T'> − β_S <w'S'>)   [m²/s³]
    # with the fluxes positive into the ocean, so B_f > 0 is a buoyancy gain
    # (heating) = STABILISING forcing (L94 §2; MOM6/CVMix convention,
    # cvmix_kpp.F90:865 lstable = surf_buoy > 0).
    # Use linear-EOS coefficients (appropriate for Phase 1).
    alpha_T = 2.0e-4   # K⁻¹  thermal expansion (Vallis 2006, §1.6)
    beta_S  = 7.4e-4   # PSU⁻¹ haline contraction
    B_f = G * (alpha_T * wT_sfc - beta_S * wS_sfc)   # (ny, nx), + = stabilising
    # The helpers below (_velocity_scales, obl_depth, _blmix) take their
    # ``b_flux`` argument as positive = DESTABILISING (ζ = −κ d b_flux/u*³,
    # nonlocal term for b_flux > 0), so they receive −B_f.  Before N3 they
    # received +B_f and treated heating as convection (sign fix, N3 task 6;
    # docs/design/n2c_refs/kpp_mom6_cvmix.md §8 A1).
    b_flux = -B_f

    # ---- Step 1: OBL depth -------------------------------------------------
    h_bl = obl_depth(rho, u_t, v_t, z_t, u_star, b_flux)   # (ny, nx)

    # ---- Step 2: Interior diffusivities at all interfaces ------------------
    K_m_int, K_s_int = interior_diffusivity(rho, u_t, v_t, dz)  # (nz+1, ny, nx)

    # ---- Step 3: Boundary-layer mixing (overrides interior inside OBL) ----
    K_m, K_s, gamma_T, gamma_S = _blmix(
        h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux, wT_sfc, wS_sfc
    )   # K: (nz+1, ny, nx),  gamma: (nz+1, ny, nx)

    # ---- Step 4: Mixing tendency from flux divergence ----------------------
    # Flux at interface k_f (between layer k_f−1 and k_f):
    #   F_T[k_f] = K_s[k_f] × (T[k_f-1] − T[k_f]) / Δz_f[k_f]
    # with non-local component:
    #   F_T[k_f] += K_s[k_f] × γ_T[k_f] × <w'T'>_sfc
    # Tendency:
    #   ∂T[k]/∂t = (F_T[k] − F_T[k+1]) / dz[k]
    # where k_f = k is the upper interface and k_f = k+1 is the lower interface.

    # Distance between layer centres for each interior interface
    # dz_f[k_f] = ½(dz[k_f−1] + dz[k_f])  for k_f = 1..nz-1
    dz_f = np.zeros(nz + 1)
    dz_f[1:nz] = 0.5 * (dz[:nz - 1] + dz[1:nz])
    dz_f[0]  = dz[0]     # not used (K[0] = 0 at surface)
    dz_f[nz] = dz[nz-1]  # not used (K[nz] = 0 at bottom)

    # ---- Tracer (T and S) tendency -----------------------------------------
    def _tracer_tendency(
        phi: np.ndarray,        # (nz, ny, nx)
        K: np.ndarray,          # (nz+1, ny, nx)
        gamma: np.ndarray,      # (nz+1, ny, nx)
        wphi_sfc: np.ndarray,   # (ny, nx)
    ) -> np.ndarray:
        """Explicit vertical diffusion + non-local flux divergence."""
        # Diffusive flux at each interface [var·m/s]
        F = np.zeros((nz + 1, ny, nx))
        for k_f in range(1, nz):
            dphi = (phi[k_f - 1] - phi[k_f]) / dz_f[k_f]   # ∂φ/∂z ≈ Δφ/Δz_f
            F[k_f] = K[k_f] * dphi + K[k_f] * gamma[k_f] * wphi_sfc[np.newaxis, ...]
        # Flux divergence → tendency: ∂φ/∂t = (F[k] − F[k+1]) / dz[k]
        dphi_dt = (F[:nz] - F[1:nz + 1]) / dz[:, np.newaxis, np.newaxis]
        return dphi_dt   # (nz, ny, nx)

    dT_dt = _tracer_tendency(T_np, K_s, gamma_T, wT_sfc)   # (nz, ny, nx)
    dS_dt = _tracer_tendency(S_np, K_s, gamma_S, wS_sfc)   # (nz, ny, nx)

    # ---- Momentum (u, v) tendency from vertical viscosity ------------------
    # For u at U-faces, we mix along k at the T-cell position and broadcast.
    # This is a C-grid approximation; full C-grid viscosity staggering is
    # a Phase 2 enhancement.  Here we apply K_m (T-cell values) to u_t/v_t
    # and project back.

    def _momentum_tendency(
        phi_t: np.ndarray,   # (nz, ny, nx) velocity at T-cells
        K: np.ndarray,       # (nz+1, ny, nx)
    ) -> np.ndarray:
        """Explicit vertical viscosity for velocity at T-cell centres."""
        F = np.zeros((nz + 1, ny, nx))
        for k_f in range(1, nz):
            dphi = (phi_t[k_f - 1] - phi_t[k_f]) / dz_f[k_f]
            F[k_f] = K[k_f] * dphi
        dphi_dt = (F[:nz] - F[1:nz + 1]) / dz[:, np.newaxis, np.newaxis]
        return dphi_dt   # (nz, ny, nx), tendency at T-cell centres

    du_t_dt = _momentum_tendency(u_t, K_m)   # (nz, ny, nx) at T-cells
    dv_t_dt = _momentum_tendency(v_t, K_m)   # (nz, ny, nx) at T-cells

    # Project T-cell tendency back to U/V staggered positions
    # du[k, j, i_u] ≈ ½(du_t[k, j, i_u-1] + du_t[k, j, i_u])
    du_dt = np.zeros_like(u_np)   # (nz, ny, nx+1)
    du_dt[:, :, 1:-1] = 0.5 * (du_t_dt[:, :, :-1] + du_t_dt[:, :, 1:])
    # Boundary U-faces remain zero (solid walls or no wind stress projected there)

    dv_dt = np.zeros_like(v_np)   # (nz, ny+1, nx)
    dv_dt[:, 1:-1, :] = 0.5 * (dv_t_dt[:, :-1, :] + dv_t_dt[:, 1:, :])

    # ---- Assemble tendency ModelState -------------------------------------
    du = xr.DataArray(du_dt, dims=state.u.dims, coords=state.u.coords,
                      attrs={"units": "m/s2", "long_name": "KPP du/dt (vertical viscosity)"})
    dv = xr.DataArray(dv_dt, dims=state.v.dims, coords=state.v.coords,
                      attrs={"units": "m/s2", "long_name": "KPP dv/dt (vertical viscosity)"})
    dT = xr.DataArray(dT_dt, dims=state.T.dims, coords=state.T.coords,
                      attrs={"units": "degC/s", "long_name": "KPP dT/dt (diffusion + non-local)"})
    dS = xr.DataArray(dS_dt, dims=state.S.dims, coords=state.S.coords,
                      attrs={"units": "PSU/s", "long_name": "KPP dS/dt (diffusion + non-local)"})

    return ModelState(
        u=du,
        v=dv,
        w=xr.zeros_like(state.w),
        T=dT,
        S=dS,
        eta=xr.zeros_like(state.eta),
        time=0.0,
    )
