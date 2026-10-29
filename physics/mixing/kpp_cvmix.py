"""
K-profile parameterisation (KPP) of the ocean surface boundary layer, column
version written after CVMix as MOM6 calls it (N3 task 6).

Reference implementation (cited as ``file:line``, local copies, MOM6 git
0e9a561, CVMix-src fce4221):

* ``cvmix_kpp.F90`` = ``~/pyOCEAN_refs/MOM6/pkg/CVMix-src/src/shared/cvmix_kpp.F90``
* ``cvmix_math.F90`` = ``~/pyOCEAN_refs/MOM6/pkg/CVMix-src/src/shared/cvmix_math.F90``
* ``MOM_CVMix_KPP.F90`` = ``~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_CVMix_KPP.F90``

Physics (Large, McWilliams & Doney 1994, "L94"; Van Roekel et al. 2018 for
the CVMix choices):

1. **Boundary-layer depth h** (L94 eq. 21): for every layer k taken as a
   trial depth d = |z_c(k)|, the bulk Richardson number

       Ri_b(d) = (1 − ε/2) d (B_r − B(d)) / (|V_r − V(d)|² + V_t²(d))

   with B_r, V_r averaged over the surface layer 0 < z < ε max(d, h₁)
   (h₁ = thickness of the top layer, MOM_CVMix_KPP.F90:1250-1258, 1331-1352);
   the factor (1 − ε/2) is CVMix's (cvmix_kpp.F90:2117-2129).  The
   unresolved shear (L94 eq. 23; Danabasoglu et al. 2006 eq. A3 for C_v):

       V_t²(d) = C_v V_tc d N w_s(εd) / Ri_c,
       V_tc = (0.2 / (c_s ε))^{1/2} / κ²,
       C_v = 2.1 − 200 N  (N < 0.002 s⁻¹),  1.7 otherwise,

   floored at 1e-10 m² s⁻² (cvmix_kpp.F90:2871-2896, DEFAULT branch), N at the
   interface *below* the layer (cvmix_kpp.F90:2694, 2703).  h is the depth where Ri_b
   first exceeds Ri_c = 0.3, interpolated between the two layer centres that
   bracket the crossing with CVMix's linear / quadratic / cubic polynomial
   (third point = the layer above) and a Newton root
   (cvmix_kpp.F90:1734-1809, cvmix_math.F90:60-294).  MOM6 then keeps h at
   least as deep as the top layer and no deeper than the bed
   (MOM_CVMix_KPP.F90:1456-1457, 1551, 1556).
2. **Velocity scales** (L94 eq. 13 and App. B): ζ = σ h B_f κ / u*³ with
   σ = min(ε, σ) (Danabasoglu et al. construction, ``l_LMD_ws = False``);
   stable 1/(1+5ζ); unstable (1−16ζ)^{1/4} (momentum) and (1−16ζ)^{1/2}
   (scalars) down to ζ_m = −0.2, ζ_s = −1, then the convective cubic laws
   (a − cζ)^{1/3}; u* = 0 gives w = 0 for B_f ≥ 0 and
   w = κ (−c σ h κ B_f)^{1/3} for B_f < 0 (cvmix_kpp.F90:2330-2405,
   2523-2603, 2974-3004).
3. **Diffusivities** (MATCH_TECHNIQUE = SimpleShapes, the CVMix and MOM6
   default, cvmix_kpp.F90:989-998, 1200-1238): K_x(σ) = h w_x(σ) G(σ),
   G = σ(1 − σ)², at interfaces 2 … kwup (CVMix numbering), no matching to
   the interior (MOM6 passes zero interior K, MOM_CVMix_KPP.F90:846-855).
   The interface straddling h gets L94 Appendix D's enhanced value
   (cvmix_kpp.F90:1240-1293, 1874-1979).
4. **Nonlocal transport** (L94 eq. 19–20 with the SimpleShapes shape):
   γ(σ) = C_s G(σ), C_s = C* κ (κ ε c_s)^{1/3} = 6.327 (cvmix_kpp.F90:576-584),
   only for B_f ≤ 0 (CVMix: ``lstable = surf_buoy > 0``, cvmix_kpp.F90:821,
   so B_f = 0 counts as unstable); zero at the surface and below h.  MOM6
   applies it as ∂T_k/∂t = F_T (γ_top − γ_bottom)/h_k with F_T the non-solar
   surface flux, positive into the ocean [K m s⁻¹] (MOM_CVMix_KPP.F90:1899-1907);
   in the model orientation γ_top = nonlocal_[k+1], γ_bottom = nonlocal_[k].
   The column integral is zero because γ = 0 at both ends.

Conventions of this module
--------------------------
* ``kpp_column`` takes the model's arrays: columns (N, M), **k = 0 at the
  bed**, interfaces k = 0 … N with the free surface at k = N.
* The lower-level functions (``turbulent_scales``, ``compute_obl_depth``,
  ``kpp_coefficients`` …) follow CVMix: index 0 at the **surface**, heights
  z ≤ 0 (negative downward), arrays (n, M).  ``kpp_column`` flips the arrays.
* Surface buoyancy flux B_f positive = **stabilising** (buoyancy gain, e.g.
  heating), MOM6/CVMix convention (cvmix_kpp.F90:821).

Not implemented (all off in MOM6's defaults and in the CVMix SCM tests):
Langmuir and Stokes-MOST options, MatchGradient / MatchBoth /
ParabolicNonLocal, the NLT_SHAPE overrides, horizontal smoothing of h
(N_SMOOTH), the interior shear mixing of L94 (§4; MOM6 takes it from its own
interior schemes).  ``kappa_bg``/``nu_bg`` add an optional constant
background (default 0: the M.Sc. thesis showed it damps internal-wave
spectra).

Cost: NumPy, vectorised over the M columns with O(N) Python loops; about
0.2 µs per column-layer on one M3 Pro P-core (N = 50, M = 20 000), i.e. ~2 s
for the 560×352×50 BayIW grid (see the task-6 report).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# Newton iteration of cvmix_math_cubic_root_find (cvmix_math.F90:46-47)
NEWTON_TOL: float = 1.0e-12
NEWTON_MAX_ITERS: int = 100

# GV%H_subroundoff = 1e-20 * max(Angstrom_H = 1e-10 m, 1e-17 m) (MOM_verticalGrid.F90:219) [m]
H_SUBROUNDOFF: float = 1.0e-30

_INTERP_TYPES = ("linear", "quadratic", "cubic")
_SW_METHODS = ("MXL_SW", "ALL_SW", "LV1_SW")


@dataclass(frozen=True)
class KPPParams:
    """
    KPP parameters; the defaults are CVMix's as MOM6 sets them.

    Attributes
    ----------
    ri_crit : float
        Critical bulk Richardson number Ri_c [nondim]; 0.3
        (MOM_CVMix_KPP.F90:312-315 RI_CRIT; cvmix_kpp.F90:290).
    vonkarman : float
        von Kármán constant κ [nondim]; 0.4 (MOM_CVMix_KPP.F90:316-318;
        cvmix_kpp.F90:330).
    surf_layer_ext : float
        Surface-layer fraction ε of h [nondim]; 0.1 (MOM_CVMix_KPP.F90:360-362
        SURF_LAYER_EXTENT; cvmix_kpp.F90:389).
    cstar : float
        C* of the nonlocal coefficient [nondim]; 10 (cvmix_kpp.F90:337).
    zeta_m, zeta_s : float
        Matching ζ of the convective cubic laws for momentum and scalars
        [nondim]; −0.2 and −1 (cvmix_kpp.F90:349, 361; L94 App. B).
    cv : float or None
        Scalar C_v of V_t² [nondim].  None (MOM6: CVMix_init_kpp is called
        without Cv) uses Danabasoglu et al. 2006 eq. A3: C_v = 2.1 − 200 N for
        N < 0.002 s⁻¹, else 1.7 (cvmix_kpp.F90:400, 2880-2887).  L94 Table 1
        gives 1.5–1.6 for β_T = −0.2.
    min_vt2 : float
        Floor on V_t² [m² s⁻²]; 1e-10 (MOM_CVMix_KPP.F90:367-370 MINIMUM_VT2).
    min_obl_depth : float
        Minimum h [m]; 0 (MOM_CVMix_KPP.F90:363-366 MINIMUM_OBL_DEPTH).
    max_obl_depth : float
        Maximum h [m]; 0 = no limit (cvmix_kpp.F90:310, 1807-1808; MOM6
        does not set it).
    interp_type : str
        Interpolation of Ri_b for h: 'linear', 'quadratic' (MOM6 default,
        MOM_CVMix_KPP.F90:322-325 INTERP_TYPE) or 'cubic' (the CVMix SCM tests).
    enhance_diffusion : bool
        L94 App. D enhanced K at the interface straddling h; True
        (MOM_CVMix_KPP.F90:319-321 ENHANCE_DIFFUSION).
    compute_ekman, compute_monin_obukhov : bool
        Limit h by 0.7 u*/|f| or u*³/(κ B_f) in stable forcing (L94 eq. 24);
        False (MOM_CVMix_KPP.F90:333-339).
    deep_obl_offset : float
        Distance above the bed to which h is clipped [m]; 0
        (MOM_CVMix_KPP.F90:346-349 DEEP_OBL_OFFSET).
    sw_method : str
        Shortwave in the B_f used for K and γ: 'MXL_SW' (default, only the SW
        absorbed above the bottom of the layer containing h — L94 eq. A3 with
        h_B = h), 'ALL_SW' or 'LV1_SW' (MOM_CVMix_KPP.F90:423-435, 835-845).
    min_thickness : float
        CVMix_ZERO_H_WORK_AROUND [m]; 0 (MOM_CVMix_KPP.F90:437-440): layers are
        inflated to this thickness with the excess carried down so the column
        depth is kept (MOM_CVMix_KPP.F90:1240-1248).
    l_lmd_ws : bool
        True: LMD94's σ in stable forcing; False (CVMix default) caps σ at ε
        (cvmix_kpp.F90:592).
    kappa_bg, nu_bg : float
        pyOCEAN option, constant background diffusivity / viscosity added at
        the interior interfaces [m² s⁻¹]; 0 (N3 spec: no background by default).
    """

    ri_crit: float = 0.3
    vonkarman: float = 0.4
    surf_layer_ext: float = 0.1
    cstar: float = 10.0
    zeta_m: float = -0.2
    zeta_s: float = -1.0
    cv: float | None = None
    min_vt2: float = 1.0e-10
    min_obl_depth: float = 0.0
    max_obl_depth: float = 0.0
    interp_type: str = "quadratic"
    enhance_diffusion: bool = True
    compute_ekman: bool = False
    compute_monin_obukhov: bool = False
    deep_obl_offset: float = 0.0
    sw_method: str = "MXL_SW"
    min_thickness: float = 0.0
    l_lmd_ws: bool = False
    kappa_bg: float = 0.0
    nu_bg: float = 0.0

    def __post_init__(self) -> None:
        errors = []
        if self.ri_crit < 0:
            errors.append("ri_crit must be >= 0")
        if not 0.0 <= self.surf_layer_ext <= 1.0:
            errors.append("surf_layer_ext must be in [0, 1]")
        if self.zeta_m >= 0 or self.zeta_s >= 0:
            errors.append("zeta_m and zeta_s must be negative")
        if self.interp_type not in _INTERP_TYPES:
            errors.append(f"interp_type must be one of {_INTERP_TYPES}")
        if self.sw_method not in _SW_METHODS:
            errors.append(f"sw_method must be one of {_SW_METHODS}")
        for name in ("vonkarman", "min_vt2", "min_obl_depth", "max_obl_depth",
                     "min_thickness", "kappa_bg", "nu_bg"):
            if getattr(self, name) < 0:
                errors.append(f"{name} must be >= 0")
        if errors:
            raise ValueError("KPPParams: " + "; ".join(errors))

    # Derived constants (cvmix_kpp.F90:368-378, 576-578, 2874-2876)
    @property
    def a_m(self) -> float:
        """a_m = (1 − 16 ζ_m)^{−1/4} (1 − 4 ζ_m)  (L94 eq. B1b matching)."""
        return (1.0 - 16.0 * self.zeta_m) ** (-0.25) * (1.0 - 4.0 * self.zeta_m)

    @property
    def c_m(self) -> float:
        """c_m = 12 (1 − 16 ζ_m)^{−1/4}  (8.38 for ζ_m = −0.2)."""
        return (1.0 - 16.0 * self.zeta_m) ** (-0.25) * 12.0

    @property
    def a_s(self) -> float:
        """a_s = (1 − 16 ζ_s)^{1/2} (1 + 8 ζ_s)  (−28.86 for ζ_s = −1)."""
        return np.sqrt(1.0 - 16.0 * self.zeta_s) * (1.0 + 8.0 * self.zeta_s)

    @property
    def c_s(self) -> float:
        """c_s = 24 (1 − 16 ζ_s)^{1/2}  (98.95 for ζ_s = −1)."""
        return 24.0 * np.sqrt(1.0 - 16.0 * self.zeta_s)

    @property
    def nonlocal_coeff(self) -> float:
        """C_s = C* κ (κ ε c_s)^{1/3} = 6.327 (L94 eq. 20; cvmix_kpp.F90:576-578)."""
        k = self.vonkarman
        return self.cstar * k * (k * self.surf_layer_ext * self.c_s) ** (1.0 / 3.0)

    @property
    def vtc(self) -> float:
        """V_tc = (0.2 / (c_s ε))^{1/2} / κ² (L94 eq. 23 with β_T = −0.2; cvmix_kpp.F90:2874-2876)."""
        return np.sqrt(0.2 / (self.c_s * self.surf_layer_ext)) / self.vonkarman ** 2


@dataclass
class KPPResult:
    """
    Output of :func:`kpp_column` (model orientation, k = 0 at the bed).

    Attributes
    ----------
    obl_depth : (M,) boundary-layer depth h below the free surface [m].
    kappa_m : (N+1, M) KPP viscosity at interfaces [m² s⁻¹] (+ ``nu_bg`` inside).
    kappa_s : (N+1, M) KPP diffusivity of heat and salt at interfaces [m² s⁻¹]
        (+ ``kappa_bg`` inside).  CVMix's Tdiff and Sdiff are equal with SimpleShapes.
    nonlocal_ : (N+1, M) nonlocal coefficient γ = C_s σ(1−σ)² at interfaces [nondim];
        zero at the surface (k = N), below h and when B_f > 0.
    k_obl : (M,) CVMix's fractional level of h (surface-down, 1-based:
        layer kw = floor(k_obl) contains h; .25 upper half, .75 lower half;
        cvmix_kpp.F90:1821-1867).  Model layer index = N − floor(k_obl).
    bulk_ri : (N, M) bulk Richardson number at the layer centres [nondim].
    bflux_obl : (M,) surface buoyancy flux used for K and γ [m² s⁻³]
        (positive = stabilising; ``sw_method``).
    """

    obl_depth: np.ndarray
    kappa_m: np.ndarray
    kappa_s: np.ndarray
    nonlocal_: np.ndarray
    k_obl: np.ndarray
    bulk_ri: np.ndarray
    bflux_obl: np.ndarray


# ---------------------------------------------------------------------------
# Velocity scales
# ---------------------------------------------------------------------------

def _phi_inv_m(zeta: np.ndarray, p: KPPParams) -> np.ndarray:
    """
    1/φ_m(ζ) (L94 eqs. B1a–c; cvmix_kpp.F90:2976-2986 compute_phi_inv):
    stable 1/(1+5ζ); −0.2 ≤ ζ < 0: (1−16ζ)^{1/4}; ζ < ζ_m: (a_m − c_m ζ)^{1/3}.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        stable = 1.0 / (1.0 + 5.0 * zeta)
        weak = np.power(1.0 - 16.0 * zeta, 0.25)
        conv = np.power(p.a_m - p.c_m * zeta, 1.0 / 3.0)
    return np.where(zeta >= 0.0, stable, np.where(zeta >= p.zeta_m, weak, conv))


def _phi_inv_s(zeta: np.ndarray, p: KPPParams) -> np.ndarray:
    """
    1/φ_s(ζ) (L94 eqs. B1a, d, e; cvmix_kpp.F90:2989-2999):
    stable 1/(1+5ζ); −1 ≤ ζ < 0: (1−16ζ)^{1/2}; ζ < ζ_s: (a_s − c_s ζ)^{1/3}.
    """
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        stable = 1.0 / (1.0 + 5.0 * zeta)
        weak = np.power(1.0 - 16.0 * zeta, 0.5)
        conv = np.power(p.a_s - p.c_s * zeta, 1.0 / 3.0)
    return np.where(zeta >= 0.0, stable, np.where(zeta >= p.zeta_s, weak, conv))


def turbulent_scales(
    sigma: np.ndarray | float,
    obl_depth: np.ndarray | float,
    bflux: np.ndarray | float,
    ustar: np.ndarray | float,
    params: KPPParams,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Turbulent velocity scales w_m, w_s [m s⁻¹] (L94 eq. 13, App. B).

        w_x = κ u* / φ_x(ζ),   ζ = σ_loc h B_f κ / u*³,   σ_loc = min(ε, σ)

    (cvmix_kpp.F90:2330-2405 / 2523-2603; ``l_LMD_ws`` restores σ_loc = σ when
    B_f ≥ 0).  For u* = 0: w_x = 0 if B_f ≥ 0, else the free-convection limit
    w_x = κ (−c_x min(ε, σ) h κ B_f)^{1/3} (c_m, c_s; cvmix_kpp.F90:2367-2402,
    2561-2600).
    u* whose cube underflows to 0 is treated as u* = 0 (CVMix tests u* ≠ 0).

    All arguments broadcast elementwise; B_f positive = stabilising [m² s⁻³];
    h [m]; u* [m s⁻¹]; σ [nondim].
    """
    p = params
    s, h, B, us = np.broadcast_arrays(
        np.asarray(sigma, dtype=float), np.asarray(obl_depth, dtype=float),
        np.asarray(bflux, dtype=float), np.asarray(ustar, dtype=float))
    kap = p.vonkarman
    sig_eps = np.minimum(p.surf_layer_ext, s)
    if p.l_lmd_ws:
        sig_loc = np.where(B >= 0.0, s, sig_eps)
    else:
        sig_loc = sig_eps
    u3 = us ** 3
    has_u = u3 > 0.0
    with np.errstate(invalid="ignore", divide="ignore", over="ignore"):
        zeta = np.where(has_u, sig_loc * h * B * kap / np.where(has_u, u3, 1.0), 0.0)
    w_m = _phi_inv_m(zeta, p) * kap * us
    w_s = _phi_inv_s(zeta, p) * kap * us
    # u* = 0: free convection or no turbulence (cvmix_kpp.F90:2367-2402)
    conv = (~has_u) & (B < 0.0)
    base = np.where(conv, -sig_eps * h * kap * np.minimum(B, 0.0), 0.0)
    w_m0 = kap * np.power(p.c_m * base, 1.0 / 3.0)
    w_s0 = kap * np.power(p.c_s * base, 1.0 / 3.0)
    w_m = np.where(has_u, w_m, np.where(conv, w_m0, 0.0))
    w_s = np.where(has_u, w_s, np.where(conv, w_s0, 0.0))
    return w_m, w_s


# ---------------------------------------------------------------------------
# Bulk Richardson number
# ---------------------------------------------------------------------------

def unresolved_shear(
    zt: np.ndarray,
    ws: np.ndarray,
    n_iface: np.ndarray,
    params: KPPParams,
) -> np.ndarray:
    """
    Unresolved turbulent shear V_t² at the layer centres [m² s⁻²] (L94 eq. 23):

        V_t²(d) = −C_v V_tc z_t N w_s / Ri_c,  ≥ min_vt2

    CVMix surface-down arrays: ``zt`` (n, M) heights ≤ 0 [m], ``ws`` (n, M)
    [m s⁻¹], ``n_iface`` (n+1, M) buoyancy frequency ≥ 0 at interfaces [s⁻¹];
    N of layer k is the one of the interface below it (cvmix_kpp.F90:2694, 2703;
    DEFAULT branch 2871-2896).
    """
    p = params
    n_c = n_iface[1:]
    if p.cv is None:
        cv = np.where(n_c < 0.002, 2.1 - 200.0 * n_c, 1.7)
    else:
        cv = p.cv
    vt2 = -cv * p.vtc * zt * n_c * ws / p.ri_crit
    return np.maximum(vt2, p.min_vt2)


def bulk_richardson(
    zt: np.ndarray,
    delta_buoy: np.ndarray,
    delta_v2: np.ndarray,
    vt2: np.ndarray,
    params: KPPParams,
) -> np.ndarray:
    """
    Bulk Richardson number at the layer centres (L94 eq. 21; cvmix_kpp.F90:2117-2129):

        Ri_b = −(1 − ε/2) z_t (B_r − B(z_t)) / (|V_r − V|² + V_t²)

    ``delta_buoy`` = B_r − B [m s⁻²] (positive when the layer is denser than the
    surface layer), ``delta_v2`` [m² s⁻²], ``vt2`` [m² s⁻²], ``zt`` ≤ 0 [m].
    A zero denominator gives num × 1e10, as CVMix.
    """
    scaling = 1.0 - 0.5 * params.surf_layer_ext
    num = -scaling * zt * delta_buoy
    den = delta_v2 + vt2
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(den != 0.0, num / np.where(den != 0.0, den, 1.0), num * 1.0e10)


# ---------------------------------------------------------------------------
# Interpolation and root finding (cvmix_math.F90)
# ---------------------------------------------------------------------------

def _poly_interp(
    interp_type: str,
    x1: np.ndarray, x2: np.ndarray, y1: np.ndarray, y2: np.ndarray,
    slope0: np.ndarray,
) -> np.ndarray:
    """
    Coefficients c₀ … c₃ (4, M) of y = Σ c_i x^i through (x1, y1), (x2, y2)
    (cvmix_math.F90:60-170, linear_through_cubic_poly_interp, port of the
    explicit inverse matrices and their summation order):

    * linear: the straight line;
    * quadratic: also y′(x1) = slope0 (the secant from the point above, or 0);
    * cubic: also y′(x1) = slope0 and y′(x2) = (y2 − y1)/(x2 − x1).
    """
    c = np.zeros((4,) + np.shape(x1))
    if interp_type == "linear":
        c[1] = (y2 - y1) / (x2 - x1)
        c[0] = y1 - c[1] * x1
    elif interp_type == "quadratic":
        det = -((x2 - x1) ** 2)
        rhs = (y2, y1, slope0)
        minv = (
            (-1.0 / det, 1.0 / det, -1.0 / (x2 - x1)),
            (2.0 * x1 / det, -2.0 * x1 / det, (x2 + x1) / (x2 - x1)),
            (-(x1 ** 2) / det, x2 * (2.0 * x1 - x2) / det, -x2 * x1 / (x2 - x1)),
        )
        for k in range(3):
            for k2 in range(3):
                c[k2] = c[k2] + minv[2 - k2][k] * rhs[k]
    else:  # cubic
        det = -((x2 - x1) ** 3)
        rhs = (y2, y1, slope0, (y2 - y1) / (x2 - x1))
        minv = (
            (2.0 / det, -2.0 / det, (x1 - x2) / det, (x1 - x2) / det),
            (-3.0 * (x2 + x1) / det, 3.0 * (x2 + x1) / det,
             (x2 - x1) * (2.0 * x2 + x1) / det, (x2 - x1) * (2.0 * x1 + x2) / det),
            (6.0 * x2 * x1 / det, -6.0 * x2 * x1 / det,
             -x2 * (x2 - x1) * (2.0 * x1 + x2) / det, -x1 * (x2 - x1) * (2.0 * x2 + x1) / det),
            (-(x1 ** 2) * (3.0 * x2 - x1) / det, -(x2 ** 2) * (-3.0 * x1 + x2) / det,
             x1 * (x2 ** 2) * (x2 - x1) / det, x2 * (x1 ** 2) * (x2 - x1) / det),
        )
        for k in range(4):
            for k2 in range(4):
                c[k2] = c[k2] + minv[3 - k2][k] * rhs[k]
    return c


def _eval_cubic(c: np.ndarray, x: np.ndarray) -> np.ndarray:
    """c₀ + c₁x + c₂x² + c₃x³ (cvmix_math.F90:301-340)."""
    return c[0] + c[1] * x + c[2] * x ** 2 + c[3] * x ** 3


def _cubic_root_find(c: np.ndarray, x0: np.ndarray) -> np.ndarray:
    """
    Newton root of the cubic from x0 until |f| < 1e-12, at most 100 steps
    (cvmix_math.F90:260-294), vectorised with a per-column stop.
    """
    root = np.array(x0, dtype=float, copy=True)
    f = c[3] * root ** 3 + c[2] * root ** 2 + c[1] * root + c[0]
    for _ in range(NEWTON_MAX_ITERS):
        active = np.abs(f) >= NEWTON_TOL
        if not np.any(active):
            break
        slope = 3.0 * c[3] * root ** 2 + 2.0 * c[2] * root + c[1]
        with np.errstate(invalid="ignore", divide="ignore"):
            step = np.where(active, f / np.where(active, slope, 1.0), 0.0)
        root = root - step
        f = c[3] * root ** 3 + c[2] * root ** 2 + c[1] * root + c[0]
    return root


# ---------------------------------------------------------------------------
# Boundary-layer depth
# ---------------------------------------------------------------------------

def compute_kobl_depth(zw: np.ndarray, zt: np.ndarray, obl_depth: np.ndarray) -> np.ndarray:
    """
    CVMix's fractional level of h (cvmix_kpp.F90:1821-1867): kw + 0.25 if h is in
    the upper half of layer kw (1-based, surface-down), kw + 0.75 in the lower
    half, n + 0.75 below the last interface.  ``zw`` (n+1, M), ``zt`` (n, M) ≤ 0.
    """
    n = zt.shape[0]
    h = np.asarray(obl_depth, dtype=float)
    below_top = h[None, :] < np.abs(zw[1:])           # h above the bottom of layer kw
    found = below_top.any(axis=0)
    kw = np.argmax(below_top, axis=0)                  # 0-based layer containing h
    zt_kw = np.take_along_axis(zt, kw[None, :], axis=0)[0]
    frac = np.where(h < np.abs(zt_kw), 0.25, 0.75)
    return np.where(found, kw + 1 + frac, n + 0.75)


def compute_obl_depth(
    ri: np.ndarray,
    zw: np.ndarray,
    zt: np.ndarray,
    params: KPPParams,
    *,
    ustar: np.ndarray | None = None,
    bflux: np.ndarray | None = None,
    coriolis: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Boundary-layer depth h from Ri_b at the layer centres
    (cvmix_kpp_compute_OBL_depth_low, non-Stokes branch, cvmix_kpp.F90:1734-1813).

    Let k be the first layer (0-based, surface-down) with Ri_b > Ri_c:
    none → h = h_lim; k = 0 → h = |z_t,0|; otherwise the polynomial of
    ``interp_type`` through (z_{k−1}, Ri_{k−1}), (z_k, Ri_k) (slope from
    (z_{k−2}, Ri_{k−2}) when k ≥ 2) is solved for Ri_b = Ri_c by Newton from the
    mid point; k = 1 keeps h ≥ |z_t,0|; h ≤ h_lim.  h_lim = |z_t| of the bottom
    layer, reduced by 0.7 u*/|f| (Ekman) and u*³/(κ B_f) (Monin–Obukhov) in
    stable forcing when switched on (L94 eq. 24).  Finally h ≥ min_obl_depth and
    h ≤ max_obl_depth if > 0.

    Arrays (CVMix orientation): ``ri`` (n, M), ``zw`` (n+1, M), ``zt`` (n, M)
    heights ≤ 0 [m]; ``bflux`` (n, M) or (M,) — its bottom value decides
    stability, as CVMix's ``surf_buoy(nlev)``.  Returns (h (M,) [m], k_obl (M,)).
    """
    p = params
    n, m = ri.shape
    cols = np.arange(m)
    limit = np.abs(zt[n - 1]).copy()
    if p.compute_ekman or p.compute_monin_obukhov:
        if ustar is None or bflux is None:
            raise ValueError("compute_obl_depth: Ekman/Monin-Obukhov limits need ustar and bflux")
        b_bot = np.asarray(bflux, dtype=float)
        b_bot = b_bot[-1] if b_bot.ndim == 2 else b_bot
        us = np.broadcast_to(np.asarray(ustar, dtype=float), (m,))
        stable = b_bot > 0.0
        if p.compute_ekman:
            if coriolis is None:
                raise ValueError("compute_obl_depth: the Ekman limit needs coriolis")
            fabs = np.abs(np.broadcast_to(np.asarray(coriolis, dtype=float), (m,)))
            ok = stable & (fabs != 0.0)
            with np.errstate(divide="ignore", invalid="ignore"):
                ekman = np.where(ok, 0.7 * us / np.where(ok, fabs, 1.0), limit)
            limit = np.minimum(limit, ekman)
        if p.compute_monin_obukhov:
            with np.errstate(divide="ignore", invalid="ignore"):
                mo = np.where(stable, us ** 3 / (np.where(stable, b_bot, 1.0) * p.vonkarman), limit)
            limit = np.minimum(limit, mo)

    exceed = ri > p.ri_crit
    any_ex = exceed.any(axis=0)
    k = np.where(any_ex, np.argmax(exceed, axis=0), n)

    # Interpolation for 1 <= k <= n-1 (Fortran depth(k:k+1), depth(k-1))
    kk = np.clip(k, 1, max(n - 1, 1))
    if n >= 2:
        x1 = zt[kk - 1, cols]
        x2 = zt[kk, cols]
        y1 = ri[kk - 1, cols]
        y2 = ri[kk, cols]
        k0 = np.maximum(kk - 2, 0)
        x0 = zt[k0, cols]
        y0 = ri[k0, cols]
        has3 = kk >= 2
        with np.errstate(invalid="ignore", divide="ignore"):
            slope0 = np.where(has3, (y1 - y0) / np.where(has3, x1 - x0, 1.0), 0.0)
        coeffs = _poly_interp(p.interp_type, x1, x2, y1, y2, slope0)
        coeffs[0] = coeffs[0] - p.ri_crit
        h_int = -_cubic_root_find(coeffs, 0.5 * (x1 + x2))
        h_int = np.where(kk == 1, np.maximum(h_int, -zt[0]), h_int)
        h_int = np.minimum(h_int, limit)
    else:
        h_int = limit
    h = np.where(k == n, np.abs(limit), np.where(k == 0, np.abs(zt[0]), h_int))
    h = np.maximum(h, p.min_obl_depth)
    if p.max_obl_depth > 0.0:
        h = np.minimum(h, p.max_obl_depth)
    return h, compute_kobl_depth(zw, zt, h)


# ---------------------------------------------------------------------------
# Diffusivities and nonlocal term
# ---------------------------------------------------------------------------

def kpp_coefficients(
    zw: np.ndarray,
    zt: np.ndarray,
    obl_depth: np.ndarray,
    k_obl: np.ndarray,
    ustar: np.ndarray,
    bflux: np.ndarray,
    params: KPPParams,
    *,
    kappa_m_in: np.ndarray | None = None,
    kappa_s_in: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    KPP viscosity, diffusivity and nonlocal coefficient at interfaces
    (cvmix_coeffs_kpp_low, SimpleShapes, non-Stokes; cvmix_kpp.F90:701-1297).

    With kwup = floor(k_obl), ktup = nint(k_obl) − 1 (1-based):

    * interfaces 2 … kwup: K_x = h w_x(σ) G(σ), σ = −z_w/h, G = σ(1−σ)²;
      γ = C_s G(σ) if B_f ≤ 0;
    * interface ktup+1 (L94 App. D): with δ = (h + z_t,ktup)/(z_t,ktup − z_t,ktup+1)
      and K_ktup = h w_x(σ_ktup) G(σ_ktup) at the centre above h,
      K_enh = (1−δ)² K_ktup + δ² K_old,  K = (1−δ) K_in + δ K_enh,
      where K_old is the incoming K (ktup = kwup, interface below h) or the
      OBL value (ktup = kwup − 1, interface inside h; γ rescaled by K/K_old)
      (cvmix_kpp.F90:1874-1979);
    * other interfaces keep the incoming K (``kappa_*_in``, default 0 = MOM6:
      no matching to the interior, MOM_CVMix_KPP.F90:846-855).

    Arrays in CVMix orientation: ``zw`` (n+1, M), ``zt`` (n, M) ≤ 0 [m]; ``obl_depth``,
    ``k_obl``, ``ustar`` [m s⁻¹], ``bflux`` [m² s⁻³, + stabilising] (M,).
    Returns (K_m, K_s, γ), each (n+1, M).  Columns with h ≤ 0 return the
    incoming K and γ = 0.
    """
    p = params
    n = zt.shape[0]
    nw = n + 1
    m = zt.shape[1]
    cols = np.arange(m)
    h = np.asarray(obl_depth, dtype=float)
    B = np.broadcast_to(np.asarray(bflux, dtype=float), (m,))
    us = np.broadcast_to(np.asarray(ustar, dtype=float), (m,))
    km_in = np.zeros((nw, m)) if kappa_m_in is None else np.array(kappa_m_in, dtype=float)
    ks_in = np.zeros((nw, m)) if kappa_s_in is None else np.array(kappa_s_in, dtype=float)
    ok = h > 0.0
    hs = np.where(ok, h, 1.0)
    stable = B > 0.0

    kwup = np.floor(k_obl).astype(int)                 # 1-based interface above h
    ktup = (np.floor(k_obl + 0.5) - 1).astype(int)     # 1-based centre above h (nint)
    ktup = np.clip(ktup, 1, n)
    kwup = np.clip(kwup, 1, n)

    zt_up = zt[ktup - 1, cols]
    below = np.where(ktup < n, zt[np.minimum(ktup, n - 1), cols], zw[n, cols])
    with np.errstate(invalid="ignore", divide="ignore"):
        delta = (h + zt_up) / (zt_up - below)

    sigma = -zw / hs[None, :]
    w_m, w_s = turbulent_scales(sigma, hs[None, :], B[None, :], us[None, :], p)
    G = sigma * (1.0 - sigma) ** 2
    kw1 = np.arange(1, nw + 1)[:, None]                 # 1-based interface number
    in_obl = (kw1 >= 2) & (kw1 <= kwup[None, :])
    obl_m = np.where(in_obl, hs * w_m * G, 0.0)
    obl_s = np.where(in_obl, hs * w_s * G, 0.0)
    nonloc = np.where(in_obl & ~stable[None, :], p.nonlocal_coeff * G, 0.0)

    sig_k = -zt_up / hs
    wm_k, ws_k = turbulent_scales(sig_k, hs, B, us, p)
    g_k = sig_k * (1.0 - sig_k) ** 2
    km_k = hs * wm_k * g_k
    ks_k = hs * ws_k * g_k

    idx = ktup                                          # 0-based row of interface ktup+1
    km_out = km_in.copy()
    ks_out = ks_in.copy()
    if p.enhance_diffusion:
        omd = 1.0 - delta
        lkteqkw = ktup == kwup
        old_m = km_out[idx, cols]
        old_s = ks_out[idx, cols]
        obl_old_m = obl_m[idx, cols]
        obl_old_s = obl_s[idx, cols]
        # ktup = kwup: interface ktup+1 below h, blend with the incoming K
        enh_m_a = omd ** 2 * km_k + delta ** 2 * old_m
        enh_s_a = omd ** 2 * ks_k + delta ** 2 * old_s
        new_m_a = omd * old_m + delta * enh_m_a
        new_s_a = omd * old_s + delta * enh_s_a
        # ktup = kwup - 1: interface ktup+1 inside h, blend with the OBL value
        enh_m_b = omd ** 2 * km_k + delta ** 2 * obl_old_m
        enh_s_b = omd ** 2 * ks_k + delta ** 2 * obl_old_s
        new_m_b = omd * old_m + delta * enh_m_b
        new_s_b = omd * old_s + delta * enh_s_b
        new_m = np.where(lkteqkw, new_m_a, new_m_b)
        new_s = np.where(lkteqkw, new_s_a, new_s_b)
        km_out[idx, cols] = np.where(lkteqkw, new_m_a, old_m)
        ks_out[idx, cols] = np.where(lkteqkw, new_s_a, old_s)
        obl_m[idx, cols] = new_m
        obl_s[idx, cols] = new_s
        with np.errstate(invalid="ignore", divide="ignore"):
            ratio = np.where(obl_old_s != 0.0, new_s_b / np.where(obl_old_s != 0.0, obl_old_s, 1.0), 0.0)
        nl_idx = nonloc[idx, cols]
        nonloc[idx, cols] = np.where(lkteqkw, nl_idx, nl_idx * ratio)
    else:
        same = kwup == ktup
        obl_m[idx, cols] = np.where(same, km_in[idx, cols], obl_m[idx, cols])
        obl_s[idx, cols] = np.where(same, ks_in[idx, cols], obl_s[idx, cols])

    kw0 = np.arange(nw)[:, None]
    take = (kw0 >= 1) & (kw0 <= ktup[None, :])
    km = np.where(take, obl_m, km_out)
    ks = np.where(take, obl_s, ks_out)
    km = np.where(ok[None, :], km, km_in)
    ks = np.where(ok[None, :], ks, ks_in)
    nonloc = np.where(ok[None, :], nonloc, 0.0)
    return km, ks, nonloc


# ---------------------------------------------------------------------------
# Column driver in the model orientation
# ---------------------------------------------------------------------------

def _surface_layer_mean(
    x: np.ndarray, dz: np.ndarray, d_top: np.ndarray, sl_depth: np.ndarray,
) -> np.ndarray:
    """
    Thickness-weighted mean of x over 0 < depth < SL(k) for every trial layer k
    (MOM_CVMix_KPP.F90:1331-1352: δH = min(max(0, SL − H_above), Δz), line 1338).

    Surface-down arrays: ``x``, ``dz`` (n, M); ``d_top`` (n, M) depth of each
    layer top; ``sl_depth`` (n, M), non-decreasing in k.  A pointer per column
    walks down to the layer that holds SL(k), and the integral is the running
    sum of the full layers above plus the partial one, done on x − x₀ to keep
    the round-off of the differences B_r − B(d) small.
    """
    n, m = x.shape
    cols = np.arange(m)
    x0 = x[0]
    xa = x - x0[None, :]
    cum = np.zeros((n + 1, m))
    np.cumsum(xa * dz, axis=0, out=cum[1:])
    ptr = np.zeros(m, dtype=int)
    out = np.empty((n, m))
    for k in range(n):
        sl = sl_depth[k]
        while True:
            nxt = np.minimum(ptr + 1, n - 1)
            adv = (ptr < n - 1) & (d_top[nxt, cols] < sl)
            if not np.any(adv):
                break
            ptr = np.where(adv, ptr + 1, ptr)
        dtp = d_top[ptr, cols]
        part = np.clip(sl - dtp, 0.0, dz[ptr, cols])
        htot = dtp + part
        integ = cum[ptr, cols] + xa[ptr, cols] * part
        with np.errstate(invalid="ignore", divide="ignore"):
            mean = np.where(htot > 0.0, integ / np.where(htot > 0.0, htot, 1.0), 0.0)
        out[k] = mean + x0
    return out


def kpp_column(
    h: np.ndarray,
    z_iface: np.ndarray,
    u: np.ndarray,
    v: np.ndarray,
    b: np.ndarray,
    bflux_sfc: np.ndarray,
    ustar: np.ndarray,
    sw_frac_iface: np.ndarray | None,
    f: np.ndarray | None,
    params: KPPParams | None = None,
    *,
    bflux_sw: np.ndarray | None = None,
) -> KPPResult:
    """
    KPP on M independent columns in the model orientation (MOM6's
    KPP_compute_BLD + KPP_calculate, MOM_CVMix_KPP.F90:731-1058, 1062-1700).

    Parameters
    ----------
    h : (N, M) layer thicknesses [m], k = 0 at the bed.  The geometry is built
        from h, accumulated from the free surface (MOM_CVMix_KPP.F90:1240-1248),
        with the ``min_thickness`` work-around.
    z_iface : (N+1, M) interface heights [m], k = 0 at the bed, k = N the free
        surface.  Only checked for the orientation (z_iface[N] ≥ z_iface[0]).
    u, v : (N, M) velocity at the layer centres [m s⁻¹] (C-grid faces averaged
        to the tracer point by the caller, MOM_CVMix_KPP.F90:1344-1345, 1371-1372).
    b : (N, M) buoyancy −g (ρ − ρ₀)/ρ₀ of each layer [m s⁻²], from the model's
        EOS; B_r − B(d) and N² use its differences only.  N² at an interface =
        (b_above − b_below) / (½(h_above + h_below) + 1e-30)
        (MOM_CVMix_KPP.F90:1432-1442).
    bflux_sfc : (M,) **non-solar** surface buoyancy flux (heat without
        shortwave, salt / freshwater) [m² s⁻³], positive = stabilising.
    ustar : (M,) friction velocity u* [m s⁻¹].
    sw_frac_iface : (N+1, M) or None — fraction of the surface shortwave that
        reaches each interface (1 at k = N).  Ignored when ``bflux_sw`` is None.
    f : (M,) or None — Coriolis parameter [s⁻¹], used only by the Ekman limit.
    params : KPPParams (default: CVMix/MOM6 values).
    bflux_sw : (M,) or None — shortwave buoyancy flux at the surface [m² s⁻³]
        (≥ 0, stabilising).  The buoyancy flux felt by a trial depth d is
        B_f(d) = B_ns + B_sw (1 − F_sw(d)), F_sw(d) at the bottom of the layer
        (MOM6 ``buoyFlux(1) − buoyFlux(k+1)``, MOM_CVMix_KPP.F90:1383;
        MOM_forcing_type.F90:1130-1148).

    Returns
    -------
    KPPResult (model orientation).
    """
    p = KPPParams() if params is None else params
    h = np.asarray(h, dtype=float)
    if h.ndim != 2:
        raise ValueError("kpp_column: h must be (N, M)")
    n, m = h.shape
    z_iface = np.asarray(z_iface, dtype=float)
    for name, arr, shp in (("z_iface", z_iface, (n + 1, m)), ("u", u, (n, m)),
                           ("v", v, (n, m)), ("b", b, (n, m))):
        if np.shape(arr) != shp:
            raise ValueError(f"kpp_column: {name} must have shape {shp}, got {np.shape(arr)}")
    if np.any(z_iface[n] < z_iface[0] - 1e-9 * (1.0 + np.abs(z_iface[0]))):
        raise ValueError("kpp_column: z_iface must have the bed at k = 0 and the surface at k = N")
    bns = np.broadcast_to(np.asarray(bflux_sfc, dtype=float), (m,))
    us = np.broadcast_to(np.asarray(ustar, dtype=float), (m,))

    # Surface-down copies (CVMix orientation)
    dz = h[::-1].astype(float)
    us_c = np.asarray(u, dtype=float)[::-1]
    vs_c = np.asarray(v, dtype=float)[::-1]
    bs_c = np.asarray(b, dtype=float)[::-1]

    # Shortwave part of buoyFlux(k): B_sw F(z_k) at interfaces, surface first
    if bflux_sw is None:
        bsw_iface = np.zeros((n + 1, m))
    else:
        if sw_frac_iface is None:
            raise ValueError("kpp_column: bflux_sw needs sw_frac_iface")
        frac = np.asarray(sw_frac_iface, dtype=float)
        if frac.shape != (n + 1, m):
            raise ValueError(f"kpp_column: sw_frac_iface must have shape {(n + 1, m)}")
        bsw = np.broadcast_to(np.asarray(bflux_sw, dtype=float), (m,))
        bsw_iface = bsw[None, :] * frac[::-1]
    b_total = bns + bsw_iface[0]                       # buoyFlux(1): non-solar + all SW

    # Geometry from h, accumulated from the surface (MOM_CVMix_KPP.F90:1240-1248)
    zw = np.zeros((n + 1, m))
    zt = np.zeros((n, m))
    hcorr = np.zeros(m)
    for k in range(n):
        dh = dz[k] + hcorr
        hcorr = np.minimum(dh - p.min_thickness, 0.0)
        dh = np.maximum(dh, p.min_thickness)
        zt[k] = zw[k] - 0.5 * dh
        zw[k + 1] = zw[k] - dh
    d_c = -zt
    d_top_raw = np.zeros((n, m))
    if n > 1:
        np.cumsum(dz[:-1], axis=0, out=d_top_raw[1:])

    # Surface layer of every trial depth: ε max(|z_t,k|, h₁, h_min) (MOM_CVMix_KPP.F90:1250)
    sl = p.surf_layer_ext * np.maximum(np.maximum(d_c, -zw[1][None, :]), p.min_obl_depth)
    sl = np.maximum.accumulate(sl, axis=0)            # non-decreasing already; guard round-off
    b_sl = _surface_layer_mean(bs_c, dz, d_top_raw, sl)
    u_sl = _surface_layer_mean(us_c, dz, d_top_raw, sl)
    v_sl = _surface_layer_mean(vs_c, dz, d_top_raw, sl)
    delta_v2 = (us_c - u_sl) ** 2 + (vs_c - v_sl) ** 2
    delta_b = b_sl - bs_c

    # N at interfaces, surface-down: 0 at the surface and the bed (MOM_CVMix_KPP.F90:1432-1442)
    n_if = np.zeros((n + 1, m))
    if n > 1:
        n2 = (bs_c[:-1] - bs_c[1:]) / (0.5 * (dz[:-1] + dz[1:]) + H_SUBROUNDOFF)
        n_if[1:n] = np.sqrt(np.maximum(n2, 0.0))

    # B_f(d) of each trial depth: surface flux minus SW still below the layer (MOM_CVMix_KPP.F90:1383)
    bf_trial = b_total[None, :] - bsw_iface[1:]
    _, ws_trial = turbulent_scales(1.0, d_c, bf_trial, us[None, :], p)  # σ = 1 → min(ε, 1) = ε
    vt2 = unresolved_shear(zt, ws_trial, n_if, p)
    ri = bulk_richardson(zt, delta_b, delta_v2, vt2, p)

    obl, _ = compute_obl_depth(ri, zw, zt, p, ustar=us, bflux=bf_trial,
                               coriolis=None if f is None else np.broadcast_to(np.asarray(f, float), (m,)))
    # MOM6 clamps (MOM_CVMix_KPP.F90:1456-1457, 1551, 1556)
    obl = np.maximum(obl, -zw[1])
    z_bot = zw[n] + np.minimum(max(p.deep_obl_offset, 0.0), -0.1 * zw[n])
    z_bot = np.minimum(z_bot, zw[1])
    obl = np.minimum(obl, -z_bot)
    kobl = compute_kobl_depth(zw, zt, obl)

    # B_f for K and γ (KPP_SHORTWAVE_METHOD, MOM_CVMix_KPP.F90:837-845)
    if p.sw_method == "ALL_SW":
        bf_obl = b_total.copy()
    elif p.sw_method == "LV1_SW":
        bf_obl = b_total - bsw_iface[min(1, n)]
    else:  # MXL_SW: buoyFlux(1) - buoyFlux(int(kOBL)+1)
        kint = np.clip(np.floor(kobl).astype(int), 0, n)
        bf_obl = b_total - bsw_iface[kint, np.arange(m)]

    km, ks, nl = kpp_coefficients(zw, zt, obl, kobl, us, bf_obl, p)
    if n > 1:
        km[1:n] += p.nu_bg
        ks[1:n] += p.kappa_bg

    return KPPResult(
        obl_depth=obl,
        kappa_m=km[::-1].copy(),
        kappa_s=ks[::-1].copy(),
        nonlocal_=nl[::-1].copy(),
        k_obl=kobl,
        bulk_ri=ri[::-1].copy(),
        bflux_obl=bf_obl,
    )
