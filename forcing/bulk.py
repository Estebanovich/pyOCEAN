"""
Air–sea bulk flux algorithms for pyOCEAN.

Turns canonical near-surface atmospheric state (``forcing/canonical.py``)
plus model SST and surface current into the fluxes consumed by
``forcing.surface.surface_tendency``: wind stress, net and shortwave heat
flux, and evaporation minus precipitation.

Implemented schemes
-------------------
``bulk_kara`` — polynomial fits to COARE 3.0 (Kara et al. 2005), identical
to HYCOM flxflg=6 / wndflg=4, the setting of HYCOM-IOA.

``bulk_ncar_fms`` — the same NCAR equations in the form of the FMS coupler
that MOM6 uses (FMScoupler ``surface_flux.F90``, OM4 settings); equal to the
compiled Fortran bit for bit (validation/bulk_points/).

``bulk_ncar`` — NCAR/CORE bulk formulae (Large & Yeager 2004, §2.1; with
the optional high-wind drag cap of Large & Yeager 2009).  This is the
algorithm of MITgcm ``pkg/exf/exf_bulkformulae.F`` (``ALLOW_BULK_LARGEYEAGER04``)
and HYCOM ``thermf.F90`` flxflg = 5, so pyOCEAN can be validated against
both.  A comparison of the MITgcm, CROCO, HYCOM and MOM6 implementations is
in ``docs/design/era5_bulk_comparison.md``.

Turbulent fluxes (LY04 eq 4), positive **into the ocean**:

    τ  = ρ_a C_d |ΔU| ΔU                              [N m⁻²]
    Q_S = ρ_a c_pa C_h |ΔU| (θ_a − SST)               [W m⁻²]
    F_E = ρ_a C_e |ΔU| (q_a − q_sat(SST))             [kg m⁻² s⁻¹]
    Q_L = Λ_v F_E                                     [W m⁻²]

with ΔU = U_10 − u_ocean the wind relative to the surface current.

Differences from the reference codes (all deliberate, see design doc)
---------------------------------------------------------------------
* ρ_a is computed from pressure, temperature and humidity unless given,
  instead of MITgcm's constant 1.2 kg m⁻³ or LY04's 1.22 kg m⁻³.
* The longwave emissivity multiplies both incoming and outgoing terms,
  ε (LW↓ − σ SST⁴), so that the reflected part of LW↓ is accounted for.
* No magic values are used as land masks.
* The stability parameter ζ uses θ_v and q at the wind height z_u (updated
  each iteration), so 2 m and profile-consistent 10 m inputs give identical
  fluxes (tested in ``tests/test_bulk.py``).

References
----------
Large, W.G. & Yeager, S.G. (2004). Diurnal to decadal global forcing for
    ocean and sea-ice models: the data sets and flux climatologies.
    NCAR Tech. Note NCAR/TN-460+STR.
Large, W.G. & Yeager, S.G. (2009). The global climatology of an
    interannually varying air–sea flux data set. Clim. Dyn. 33, 341–364.
Paulson, C.A. (1970). The mathematical representation of wind speed and
    temperature profiles in the unstable atmospheric surface layer.
    J. Appl. Meteor. 9, 857–861.
Wallace, J.M. & Hobbs, P.V. (2006). Atmospheric Science, 2nd ed., §3.1.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

KAPPA: float = 0.4               # –          von Kármán constant (LY04)
G: float = 9.81                  # m s⁻²      gravitational acceleration
CP_AIR: float = 1000.5           # J kg⁻¹ K⁻¹ specific heat of air (LY04)
LV: float = 2.5e6                # J kg⁻¹     latent heat of vaporisation (LY04)
RHO_FW: float = 1000.0           # kg m⁻³     fresh-water density (E in m s⁻¹)
R_DRY: float = 287.04            # J kg⁻¹ K⁻¹ gas constant of dry air
SIGMA_SB: float = 5.670374419e-8 # W m⁻² K⁻⁴  Stefan–Boltzmann (CODATA 2018)
T0_K: float = 273.15             # K          0 °C
GAMMA_DRY: float = 0.0098        # K m⁻¹      dry adiabatic lapse rate (g/c_pa)
Z_REF: float = 10.0              # m          reference height of neutral coefficients

_ZETA_MAX: float = 10.0          # |ζ| clip, as MITgcm (LY04 option) and HYCOM flxflg=5


# ---------------------------------------------------------------------------
# Thermodynamics and similarity functions
# ---------------------------------------------------------------------------

def qsat_ly04(t_k: np.ndarray, rho_air: float | np.ndarray) -> np.ndarray:
    """
    Saturation specific humidity over fresh water.

        q_sat = (640380 kg m⁻³ / ρ_a) · exp(−5107.4 K / T)      (LY04 eq 5)

    Multiply by 0.98 for the reduction of vapour pressure over sea water
    (LY04 §2.1; Kraus 1972).

    Parameters
    ----------
    t_k : np.ndarray
        Temperature [K].
    rho_air : float or np.ndarray
        Air density [kg m⁻³].

    Returns
    -------
    np.ndarray
        Saturation specific humidity [kg kg⁻¹].
    """
    return 640380.0 / rho_air * np.exp(-5107.4 / t_k)


def air_density(t_k: np.ndarray, q: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Density of moist air from the ideal gas law with virtual temperature.

        ρ_a = p / (R_d T_v),   T_v = T (1 + 0.608 q)    (Wallace & Hobbs 2006, eq. 3.16)

    Parameters
    ----------
    t_k : np.ndarray
        Air temperature [K].
    q : np.ndarray
        Specific humidity [kg kg⁻¹].
    p : np.ndarray
        Pressure [Pa].

    Returns
    -------
    np.ndarray
        Air density [kg m⁻³].
    """
    return p / (R_DRY * t_k * (1.0 + 0.608 * q))


def psi_m_ly04(zeta: np.ndarray) -> np.ndarray:
    """
    Integrated stability function for momentum (LY04 eq 8b, 8c).

        ζ ≥ 0 :  ψ_m = −5ζ
        ζ < 0 :  ψ_m = 2 ln[(1+X)/2] + ln[(1+X²)/2] − 2 arctan X + π/2,
                 X = (1 − 16ζ)^¼                        (Paulson 1970)

    Parameters
    ----------
    zeta : np.ndarray
        Monin–Obukhov stability parameter ζ = z/L [–].
    """
    zeta = np.asarray(zeta, dtype=float)
    x = (1.0 - 16.0 * np.minimum(zeta, 0.0)) ** 0.25
    unstable = (2.0 * np.log((1.0 + x) / 2.0) + np.log((1.0 + x * x) / 2.0)
                - 2.0 * np.arctan(x) + np.pi / 2.0)
    return np.where(zeta >= 0.0, -5.0 * zeta, unstable)


def psi_h_ly04(zeta: np.ndarray) -> np.ndarray:
    """
    Integrated stability function for heat and moisture (LY04 eq 8b, 8c).

        ζ ≥ 0 :  ψ_h = −5ζ
        ζ < 0 :  ψ_h = 2 ln[(1+X²)/2],   X = (1 − 16ζ)^¼   (Paulson 1970)

    Parameters
    ----------
    zeta : np.ndarray
        Monin–Obukhov stability parameter ζ = z/L [–].
    """
    zeta = np.asarray(zeta, dtype=float)
    x = (1.0 - 16.0 * np.minimum(zeta, 0.0)) ** 0.25
    return np.where(zeta >= 0.0, -5.0 * zeta, 2.0 * np.log((1.0 + x * x) / 2.0))


def cd_neutral_ly04(u10n: np.ndarray, high_wind: bool = False) -> np.ndarray:
    """
    Neutral 10 m drag coefficient.

        C_dN = 10⁻³ (2.7/U₁₀N + 0.142 + U₁₀N/13.09)                 (LY04 eq 6a)

    With ``high_wind=True`` the LY09 form is used, which levels off in
    tropical-cyclone winds:

        C_dN = 10⁻³ (2.7/U + 0.142 + U/13.09 − 3.14807×10⁻¹⁰ U⁶),  U < 33 m s⁻¹
        C_dN = 2.34 × 10⁻³,                                          U ≥ 33 m s⁻¹
                                                                     (LY09 eq 11a, 11b)

    Parameters
    ----------
    u10n : np.ndarray
        Neutral 10 m wind speed [m s⁻¹], > 0.
    high_wind : bool
        Apply the LY09 high-wind form.

    Returns
    -------
    np.ndarray
        C_dN [–].
    """
    u = np.asarray(u10n, dtype=float)
    cdn = 1e-3 * (2.7 / u + 0.142 + u / 13.09)
    if high_wind:
        cdn = np.where(u < 33.0, cdn - 1e-3 * 3.14807e-10 * u ** 6, 2.34e-3)
    return cdn


# ---------------------------------------------------------------------------
# NCAR / CORE bulk algorithm
# ---------------------------------------------------------------------------

@dataclass
class TurbulentFluxes:
    """
    Output of a bulk algorithm, all at T-points, fluxes positive into the ocean.

    Attributes
    ----------
    tau_x, tau_y : wind stress [N m⁻²]
    q_sens : sensible heat flux [W m⁻²]
    q_lat : latent heat flux [W m⁻²]
    evap : evaporation [m s⁻¹], positive when water leaves the ocean
    cd, ch, ce : transfer coefficients for momentum, sensible and latent heat [–]
    rho_air : air density used [kg m⁻³]
    ustar : friction velocity √C_d |ΔU| [m s⁻¹]
    u10n : neutral 10 m wind speed [m s⁻¹] (None for non-iterative schemes)
    zeta : Monin–Obukhov stability parameter z_u/L at the wind height [–]
        (None for non-iterative schemes)
    """

    tau_x: np.ndarray
    tau_y: np.ndarray
    q_sens: np.ndarray
    q_lat: np.ndarray
    evap: np.ndarray
    cd: np.ndarray
    ch: np.ndarray
    ce: np.ndarray
    rho_air: np.ndarray
    ustar: np.ndarray
    u10n: np.ndarray | None = None
    zeta: np.ndarray | None = None


def bulk_ncar(
    u10: np.ndarray,
    v10: np.ndarray,
    t_air: np.ndarray,
    q_air: np.ndarray,
    sst: np.ndarray,
    p: np.ndarray | None = None,
    *,
    rho_air: float | np.ndarray | None = None,
    u_ocean: np.ndarray | float = 0.0,
    v_ocean: np.ndarray | float = 0.0,
    z_u: float = 10.0,
    z_t: float = 2.0,
    n_iter: int = 2,
    gamma: float = GAMMA_DRY,
    u_min: float = 0.5,
    high_wind: bool = False,
) -> TurbulentFluxes:
    """
    NCAR/CORE bulk turbulent fluxes (Large & Yeager 2004, §2.1).

    Algorithm (LY04 eqs 6–10):

      1. Relative wind ΔU = U(z_u) − u_ocean, speed U = max(|ΔU|, u_min).
         Potential temperature θ = T + Γ z_t; Δθ = θ − SST,
         Δq = q − 0.98 q_sat(SST).
      2. First guess with neutral coefficients at U₁₀N = U:
             C_dN (eq 6a), C_eN = 34.6×10⁻³ √C_dN (6b),
             C_hN = 32.7×10⁻³ √C_dN (ζ ≤ 0) or 18.0×10⁻³ √C_dN (ζ > 0) (6c);
             u* = √C_d U,  θ* = (C_h/√C_d) Δθ,  q* = (C_e/√C_d) Δq      (7)
      3. Repeat ``n_iter`` times:
             ζ = κ g z_u / u*² · [θ*/θ_v + q*/(q + 1/0.608)]             (8a)
                 with θ_v and q taken at z_u (from the previous pass), so
                 the result does not depend on the height at which T and q
                 are supplied
             ψ_m, ψ_h at ζ(z_u) and ζ(z_t) = ζ z_t/z_u                    (8b, 8c)
             U₁₀N = U [1 + √C_dN/κ (ln(z_u/10) − ψ_m)]⁻¹                  (9a)
             θ, q shifted from z_t to z_u:
                 θ(z_u) = θ(z_t) − θ*/κ [ln(z_t/z_u) + ψ_h(ζ_u) − ψ_h(ζ_t)]  (9b, 9c)
             neutral coefficients at U₁₀N, then shifted to z_u and ζ:
                 C_d = C_dN [1 + √C_dN/κ (ln(z_u/10) − ψ_m)]⁻²
                 C_h = C_hN √(C_d/C_dN) [1 + C_hN/(κ√C_dN) (ln(z_u/10) − ψ_h)]⁻¹
                 C_e = C_eN √(C_d/C_dN) [1 + C_eN/(κ√C_dN) (ln(z_u/10) − ψ_h)]⁻¹   (10a–c)
             u*, θ*, q* updated with eq 7.
      4. Fluxes with LY04 eq 4 (see module docstring).

    LY04 state that two iterations are sufficient; the default reproduces
    MITgcm (``niter_bulk = 2``) and CESM ``shr_flux_atmOcn``, i.e. the CORE /
    OMIP protocol.  ζ is clipped to |ζ| ≤ 10 as in MITgcm, CESM and HYCOM.

    Known limitations (docs/design/era5_bulk_comparison.md §7)
    ----------------------------------------------------------
    * Weak wind (U ≲ 4 m s⁻¹) with strongly stable air: 2 iterations are far
      from the fixed point (latent flux off by up to ~47 W m⁻² in tests); the
      fixed point itself sits on the |ζ| = 10 clip, because the linear stable
      functions ψ = −5ζ shut turbulence down too fast (Cheng & Brutsaert 2005;
      Yagüe et al. 2006).  More iterations do not make this regime more
      realistic — use a scheme with levelling-off stable functions (COARE 3.0)
      for it.
    * Warm but dry air (Δθ > 0, Δθ_v ≈ 0): the Stanton number jumps between
      18.0 and 32.7 ×10⁻³ √C_dN at ζ = 0 (eq 6c) and the iteration has no fixed
      point; it alternates between the two branches at any wind speed, so the
      sensible heat flux depends on the parity of ``n_iter`` (up to a factor
      ≈ 1.8).  A convergence tolerance cannot fix this; keep ``n_iter`` fixed.

    Parameters
    ----------
    u10, v10 : np.ndarray
        Wind components at height ``z_u`` [m s⁻¹].
    t_air : np.ndarray
        Air temperature at height ``z_t`` [K].
    q_air : np.ndarray
        Specific humidity at height ``z_t`` [kg kg⁻¹].
    sst : np.ndarray
        Sea surface temperature [°C].
    p : np.ndarray or None
        Sea-level pressure [Pa]; used for ρ_a when ``rho_air`` is None.
    rho_air : float, np.ndarray or None
        Air density [kg m⁻³]. Pass 1.22 to reproduce LY04 or 1.2 for MITgcm.
    u_ocean, v_ocean : np.ndarray or float
        Surface current at T-points [m s⁻¹] for relative wind (0 = absolute).
    z_u, z_t : float
        Measurement heights of wind and of temperature/humidity [m].
    n_iter : int
        Number of stability iterations.
    gamma : float
        Lapse rate converting T to potential temperature [K m⁻¹].
    u_min : float
        Minimum wind speed [m s⁻¹] (LY04: 0.5).
    high_wind : bool
        Use the LY09 high-wind drag coefficient.

    Returns
    -------
    TurbulentFluxes

    Raises
    ------
    ValueError
        If neither ``rho_air`` nor ``p`` is given.
    """
    u10 = np.asarray(u10, dtype=float)
    v10 = np.asarray(v10, dtype=float)
    t_air = np.asarray(t_air, dtype=float)
    q_air = np.asarray(q_air, dtype=float)
    sst_k = np.asarray(sst, dtype=float) + T0_K

    theta = t_air + gamma * z_t
    if rho_air is None:
        if p is None:
            raise ValueError("bulk_ncar needs either rho_air or p (sea-level pressure)")
        rho_air = air_density(theta, q_air, np.asarray(p, dtype=float))

    # 1. Relative wind and air–sea contrasts
    du = u10 - u_ocean
    dv = v10 - v_ocean
    speed = np.maximum(np.hypot(du, dv), u_min)
    q_sfc = 0.98 * qsat_ly04(sst_k, rho_air)

    # 2. Neutral first guess (U₁₀N = U, heights not yet shifted)
    log_u = np.log(z_u / Z_REF)
    log_tu = np.log(z_t / z_u)
    d_theta = theta - sst_k
    d_q = q_air - q_sfc
    cdn = cd_neutral_ly04(speed, high_wind)
    sq_cdn = np.sqrt(cdn)
    chn = np.where(d_theta > 0.0, 18.0e-3, 32.7e-3) * sq_cdn
    cen = 34.6e-3 * sq_cdn
    cd, ch, ce = cdn, chn, cen
    sq_cd = sq_cdn
    u10n = speed
    ustar = sq_cd * speed
    tstar = ch / sq_cd * d_theta
    qstar = ce / sq_cd * d_q
    zeta_u = np.zeros_like(speed)
    theta_zu, q_zu = theta, q_air      # air values at z_u, updated by eq 9b, 9c

    # 3. Stability iterations
    for _ in range(n_iter):
        theta_v = theta_zu * (1.0 + 0.608 * q_zu)
        zeta_u = KAPPA * G * z_u / ustar ** 2 * (tstar / theta_v + qstar / (q_zu + 1.0 / 0.608))
        zeta_u = np.clip(zeta_u, -_ZETA_MAX, _ZETA_MAX)
        zeta_t = zeta_u * z_t / z_u
        psim_u = psi_m_ly04(zeta_u)
        psih_u = psi_h_ly04(zeta_u)
        psih_t = psi_h_ly04(zeta_t)

        # 9a: neutral 10 m wind (uses C_dN of the previous pass)
        u10n = np.maximum(speed / (1.0 + sq_cdn / KAPPA * (log_u - psim_u)), u_min)
        # 9b, 9c: θ and q from z_t to z_u
        shift = log_tu + psih_u - psih_t
        theta_zu = theta - tstar / KAPPA * shift
        q_zu = q_air - qstar / KAPPA * shift
        d_theta = theta_zu - sst_k
        d_q = q_zu - q_sfc

        # 6: neutral coefficients at the updated U₁₀N
        cdn = cd_neutral_ly04(u10n, high_wind)
        sq_cdn = np.sqrt(cdn)
        chn = np.where(zeta_u > 0.0, 18.0e-3, 32.7e-3) * sq_cdn
        cen = 34.6e-3 * sq_cdn

        # 10a–c: shift to z_u and stability
        sq_cd = sq_cdn / (1.0 + sq_cdn / KAPPA * (log_u - psim_u))
        cd = sq_cd ** 2
        ch = chn * (sq_cd / sq_cdn) / (1.0 + chn / (KAPPA * sq_cdn) * (log_u - psih_u))
        ce = cen * (sq_cd / sq_cdn) / (1.0 + cen / (KAPPA * sq_cdn) * (log_u - psih_u))

        # 7: scales
        ustar = sq_cd * speed
        tstar = ch / sq_cd * d_theta
        qstar = ce / sq_cd * d_q

    # 4. Fluxes (LY04 eq 4), positive into the ocean
    tau_x = rho_air * cd * speed * du
    tau_y = rho_air * cd * speed * dv
    q_sens = rho_air * CP_AIR * ch * speed * d_theta
    water_flux = rho_air * ce * speed * d_q          # kg m⁻² s⁻¹, > 0 = condensation
    q_lat = LV * water_flux
    evap = -water_flux / RHO_FW

    return TurbulentFluxes(
        tau_x=tau_x, tau_y=tau_y, q_sens=q_sens, q_lat=q_lat, evap=evap,
        cd=cd, ch=ch, ce=ce, rho_air=np.broadcast_to(rho_air, speed.shape).copy(),
        ustar=ustar, u10n=u10n, zeta=zeta_u,
    )


# ---------------------------------------------------------------------------
# NCAR bulk in the form of the FMS coupler (MOM6 / OM4)
# ---------------------------------------------------------------------------
#
# Port of FMScoupler 2025.02 (git 5b700a4) shared/surface_flux.F90: surface_flux_1d
# (:216-500) with ncar_ocean_flux = .true. and raoult_sat_vap = .true. (OM4_025/input.nml:
# 113-116), calling ncar_ocean_fluxes (:774-897) or, for z_t, z_q ≠ z_u,
# ncar_ocean_fluxes_multilevel (:905-1044).  Constants are those of FMS
# (constants/gfdl_constants.fh), not of LY04.  Every expression keeps the Fortran
# evaluation order so the port matches the compiled routine to round-off
# (tests/test_bulk_fms.py, validation/bulk_points/).

FMS_RDGAS: float = 287.04                 # J kg⁻¹ K⁻¹  dry-air gas constant (gfdl_constants.fh)
FMS_RVGAS: float = 461.50                 # J kg⁻¹ K⁻¹  water-vapour gas constant
FMS_KAPPA: float = 2.0 / 7.0              # –           R_d / c_p
FMS_CP_AIR: float = FMS_RDGAS / FMS_KAPPA  # J kg⁻¹ K⁻¹  1004.64, c_p of dry air
FMS_HLV: float = 2.500e6                  # J kg⁻¹      latent heat of evaporation
FMS_GRAV: float = 9.80                    # m s⁻²       gravity
FMS_VONKARM: float = 0.40                 # –           von Kármán constant
FMS_TFREEZE: float = 273.16               # K           base of the saturation tables
_FMS_D622: float = FMS_RDGAS / FMS_RVGAS  # surface_flux.F90:149
_FMS_D378: float = 1.0 - _FMS_D622        # :150
_FMS_D608: float = _FMS_D378 / _FMS_D622  # :151-152 (use_virtual_temp = .true.)

_FMS_ES_TABLE: tuple[np.ndarray, np.ndarray, np.ndarray] | None = None


def _fms_compute_es(tem: np.ndarray) -> np.ndarray:
    """
    Saturation vapour pressure [Pa] at the table nodes, FMS ``compute_es_k``
    (FMS 2025.02.01, sat_vapor_pres/include/sat_vapor_pres_k.inc:216-289).

    Goff–Gratch over water above −20 °C, over ice below 0.01 °C (T < 273.16 K), and the
    linear blend of the two between −20 °C and 0.01 °C (Smithsonian Met. Tables p. 350):

        log₁₀ e_i = −9.09718 (T_i/T − 1) − 3.56654 log₁₀(T_i/T) + 0.876793 (1 − T/T_i)
                    + log₁₀ 610.71,                                      T_i = 273.16 K
        log₁₀ e_w = −7.90298 (T_w/T − 1) + 5.02808 log₁₀(T_w/T)
                    − 1.3816e-7 (10^{11.344 (1 − T/T_w)} − 1)
                    + 8.1328e-3 (10^{−3.49149 (T_w/T − 1)} − 1) + log₁₀ 101324.6,  T_w = 373.16 K
        e = 0.05 [(T_i − T) e_i + (T − T_i + 20) e_w]   for T_i − 20 < T < T_i

    Parameters
    ----------
    tem : np.ndarray
        Temperature [K].
    """
    tbasw = FMS_TFREEZE + 100.0
    tbasi = FMS_TFREEZE
    with np.errstate(over="ignore"):
        x_i = (-9.09718 * (tbasi / tem - 1.0) - 3.56654 * np.log10(tbasi / tem)
               + 0.876793 * (1.0 - tem / tbasi) + np.log10(610.71))
        es_ice = np.where(tem < tbasi, 10.0 ** x_i, 0.0)
        x_w = (-7.90298 * (tbasw / tem - 1.0) + 5.02808 * np.log10(tbasw / tem)
               - 1.3816e-07 * (10.0 ** ((1.0 - tem / tbasw) * 11.344) - 1.0)
               + 8.1328e-03 * (10.0 ** ((tbasw / tem - 1.0) * (-3.49149)) - 1.0)
               + np.log10(101324.60))
        es_h2o = np.where(tem > -20.0 + tbasi, 10.0 ** x_w, 0.0)
    blend = 0.05 * ((tbasi - tem) * es_ice + (tem - tbasi + 20.0) * es_h2o)
    return np.where(tem <= -20.0 + tbasi, es_ice, np.where(tem >= tbasi, es_h2o, blend))


def _fms_es_table() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    FMS saturation vapour pressure table TABLE, DTABLE, D2TABLE (built once).

    −160 … 100 °C every 0.1 K (sat_vapor_pres.F90:759-761, 823), derivative by centred
    differences ±0.01 K and one half of the second derivative by centred differences of
    DTABLE (sat_vapor_pres_k.inc:98-137).  This is the table ``escomp`` (= ``lookup_es``,
    sat_vapor_pres.F90:273-278) reads; ``construct_table_wrt_liq`` of OM4 only adds TABLE2
    for ``lookup_es2`` and does not change it.
    """
    global _FMS_ES_TABLE
    if _FMS_ES_TABLE is None:
        tcmin, tcmax, nsize = -160.0, 100.0, (100 - (-160)) * 10 + 1
        dtres = (tcmax - tcmin) / float(nsize - 1)
        tminl = tcmin + FMS_TFREEZE
        dtinvl = 1.0 / dtres
        tinrc = 0.1 * dtres
        tfact = 5.0 * dtinvl
        tem = tminl + dtres * np.arange(nsize, dtype=float)
        table = _fms_compute_es(tem)
        dtable = (_fms_compute_es(tem + tinrc) - _fms_compute_es(tem - tinrc)) * tfact
        d2table = np.empty(nsize)
        d2table[1:-1] = 0.25 * dtinvl * (dtable[2:] - dtable[:-2])
        d2table[0] = 0.50 * dtinvl * (dtable[1] - dtable[0])
        d2table[-1] = 0.50 * dtinvl * (dtable[-1] - dtable[-2])
        _FMS_ES_TABLE = (table, dtable, d2table)
    return _FMS_ES_TABLE


def esat_fms(t_k: np.ndarray) -> np.ndarray:
    """
    Saturation vapour pressure [Pa] by the FMS table lookup ``lookup_es_k``
    (sat_vapor_pres_k.inc:1518-1554), quadratic in the offset δ from the node below:

        e_s = TABLE + δ (DTABLE + δ D2TABLE),   δ = T − T_node

    Below 273.16 K the table blends towards saturation over ice (``_fms_compute_es``), so
    for sea water at −1.8 °C FMS's e_s is 0.16 % below the value over water (Goff–Gratch).

    Parameters
    ----------
    t_k : np.ndarray
        Temperature [K], 113.16–373.16 K.

    Raises
    ------
    ValueError
        Outside the table, where FMS stops with "saturation vapor pressure table overflow".
    """
    table, dtable, d2table = _fms_es_table()
    dtres = 260.0 / float(table.size - 1)
    tminl = -160.0 + FMS_TFREEZE
    tmp = np.asarray(t_k, dtype=float) - tminl
    ind = np.trunc((1.0 / dtres) * (tmp + 0.5 * dtres))
    if np.any(~np.isfinite(ind)) or np.any(ind < 0) or np.any(ind >= table.size):
        raise ValueError("esat_fms: temperature outside the FMS table (113.16–373.16 K)")
    ind = ind.astype(np.int64)
    delta = tmp - dtres * ind
    return table[ind] + delta * (dtable[ind] + delta * d2table[ind])


def bulk_ncar_fms(
    u10: np.ndarray,
    v10: np.ndarray,
    t_air: np.ndarray,
    q_air: np.ndarray,
    sst: np.ndarray,
    p: np.ndarray,
    *,
    u_ocean: np.ndarray | float = 0.0,
    v_ocean: np.ndarray | float = 0.0,
    z_u: float = 10.0,
    z_t: float = 10.0,
    z_q: float = 10.0,
) -> TurbulentFluxes:
    """
    NCAR bulk turbulent fluxes in the form of the FMS coupler that MOM6 uses (OM4).

    Line-by-line port of FMScoupler 2025.02 ``surface_flux_1d`` with
    ``ncar_ocean_flux = .true.``, ``raoult_sat_vap = .true.`` (OM4_025/input.nml:113-116),
    over sea water, no gustiness.  Same Large & Yeager (2004) equations as ``bulk_ncar``;
    the differences are listed in validation/bulk_points/README.md.

    Surface state (surface_flux.F90):

        e_s   = FMS table lookup at SST (esat_fms)                                  :308
        q_s   = 0.98 · ε e_s / (p − (1 − ε) e_s),  ε = R_d/R_v                       :320, :331
        T_v   = T_a (1 + d608 q_a),  d608 = (1 − ε)/ε                                :343
        θ_a   = T_a (p_surf/p_atm)^κ = T_a  (p_atm = p_surf = p, OM4 data_table)     :341, :344
        ΔU    = u_ocean − U_a;  W = |ΔU| (no floor, gust = 0)                        :349, :378

    Coefficients, ``ncar_ocean_fluxes`` (:843-891) or ``_multilevel`` (:939-1040),
    with U = max(W, 0.5 m s⁻¹) and two iterations:

        C_dN = (2.7/U₁₀ + 0.142 + 0.0764 U₁₀)/10³                       (LY04 6a; :849)
        C_eN = 34.6 √C_dN/10³,  C_hN = (18.0 stable | 32.7 unstable) √C_dN/10³  (6b, 6c)
            first guess: stable ⇔ T_a > SST, U₁₀ = U (:847-853)
        u* = √C_d U,  T* = C_h/√C_d (T_a(z_u) − SST),  q* = C_e/√C_d (q_a(z_u) − q_s)  (7)
        B* = g [T*/T_v + q*/(q_a(z_u) + 1/0.608)],  T_v = T_a(z_u)(1 + 0.608 q_a(z_u))  (:960-968)
        ζ_z = κ B* z / u*², clipped to |ζ| ≤ 10, for z = z_u, z_t, z_q          (8a; :970-977)
        X² = max(√|1 − 16ζ|, 1);  ψ_m = ψ_h = −5ζ (ζ ≥ 0), Paulson (ζ < 0)       (8b-e; :979-1016)
        U₁₀ = U / (1 + √C_dN (ln(z_u/10) − ψ_m)/κ)                              (9a; :1020)
        T_a(z_u) = T_a − T*/κ (ln(z_t/z_u) + ψ_h(ζ_u) − ψ_h(ζ_t)), same for q     (9b, 9c; :1021-1022)
        neutral coefficients again at U₁₀, stable ⇔ ζ_u ≥ 0                       (:1025-1029)
        C_d = C_dN / (1 + √C_dN x_m)²,  x_m = (ln(z_u/10) − ψ_m)/κ              (10a; :1031-1032)
        C_h = C_hN / (1 + C_hN x_h/√C_dN) · √(C_d/C_dN),  C_e likewise         (10b, c; :1033-1035)

    With z_t = z_q = z_u the multilevel routine is the single-level one bit for bit (the
    height shift is exactly 0 and ψ(0) = 0 on either branch), so one code serves both.

    Fluxes (surface_flux.F90:423-461), returned positive into the ocean:

        ρ_a = p / (R_d T_v)                                                    :428
        τ   = −ρ_a C_d W ΔU          (FMS flux_u, flux_v are the stress on the air)  :459-461
        Q_S = −c_p ρ_a C_h W (SST − θ_a(z_u))                                   :431-432
        F_E =  ρ_a C_e W (q_s − q_a(z_u))  [kg m⁻² s⁻¹],  Q_L = −L_v F_E          :437-438
        E   =  F_E / ρ_fw  [m s⁻¹]
    (MOM6 takes sens = −flux_t, evap = −flux_q, latent = −L_v flux_q,
    MOM_surface_forcing_gfdl.F90:496-497, 560-561, 576-578.)

    Parameters
    ----------
    u10, v10 : np.ndarray
        Wind at height ``z_u`` [m s⁻¹].
    t_air : np.ndarray
        Air temperature at ``z_t`` [K].
    q_air : np.ndarray
        Specific humidity at ``z_q`` [kg kg⁻¹].
    sst : np.ndarray
        Surface temperature [K] (MOM6 passes potential temperature + 273.15).
    p : np.ndarray
        Sea-level pressure [Pa], used as both p_atm and p_surf as OM4's data_table.
    u_ocean, v_ocean : np.ndarray or float
        Surface current [m s⁻¹].
    z_u, z_t, z_q : float
        Heights of wind, temperature and humidity [m] (FMS ``bulk_zu/zt/zq``).

    Returns
    -------
    TurbulentFluxes
        ``ustar`` is FMS's u_star, √C_d U with the C_d of the first pass of the last
        iteration (:962-963); ``u10n`` the last neutral 10 m wind; ``zeta`` ζ at z_u.
    """
    u10, v10, t_air, q_air, sst, p = np.broadcast_arrays(
        *(np.asarray(a, dtype=float) for a in (u10, v10, t_air, q_air, sst, p)))
    vk = FMS_VONKARM

    # surface humidity at saturation, Raoult's 0.98 (:308, :320, :331)
    e_sat = esat_fms(sst)
    q_s = 0.98 * (_FMS_D622 * e_sat / (p - _FMS_D378 * e_sat))
    tv_atm = t_air * (1.0 + _FMS_D608 * q_air)                  # :343
    u_dif = u_ocean - u10                                       # :349-350
    v_dif = v_ocean - v10
    w_atm = np.sqrt(u_dif * u_dif + v_dif * v_dif + 0.0 * 0.0)  # :378, gust = 0

    # ncar_ocean_fluxes_multilevel (:939-1040)
    u = np.maximum(w_atm, 0.5)
    u10n = u
    t10, q10 = t_air, q_air

    def neutral(u10n: np.ndarray, stab_arg: np.ndarray):
        cd_n10 = (2.7 / u10n + 0.142 + 0.0764 * u10n) / 1e3
        cd_n10_rt = np.sqrt(cd_n10)
        ce_n10 = 34.6 * cd_n10_rt / 1e3
        stab = 0.5 + np.copysign(0.5, stab_arg)
        ch_n10 = (18.0 * stab + 32.7 * (1 - stab)) * cd_n10_rt / 1e3
        return cd_n10, cd_n10_rt, ce_n10, ch_n10

    def psi(zeta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        x2 = np.maximum(np.sqrt(np.abs(1 - 16 * zeta)), 1.0)
        x = np.sqrt(x2)
        psi_m = np.where(zeta >= 0, -5 * zeta,
                         np.log((1 + 2 * x + x2) * (1 + x2) / 8) - 2 * (np.arctan(x) - np.arctan(1.0)))
        psi_h = np.where(zeta >= 0, -5 * zeta, 2 * np.log((1 + x2) / 2))
        return psi_m, psi_h

    def clip(zeta: np.ndarray) -> np.ndarray:
        return np.copysign(np.minimum(np.abs(zeta), 10.0), zeta)

    cd_n10, cd_n10_rt, ce_n10, ch_n10 = neutral(u10n, t10 - sst)       # :947-951
    cd, ch, ce = cd_n10, ch_n10, ce_n10
    log_u = np.log(z_u / 10.0)
    for _ in range(2):                                                  # n_itts = 2
        tv = t10 * (1 + 0.608 * q10)                                    # :960
        cd_rt = np.sqrt(cd)
        ustar = cd_rt * u                                               # 7a
        tstar = (ch / cd_rt) * (t10 - sst)                              # 7b
        qstar = (ce / cd_rt) * (q10 - q_s)                              # 7c
        bstar = FMS_GRAV * (tstar / tv + qstar / (q10 + 1 / 0.608))     # :968
        zeta_u = clip(vk * bstar * z_u / (ustar * ustar))               # 8a
        zeta_t = clip(vk * bstar * z_t / (ustar * ustar))
        zeta_q = clip(vk * bstar * z_q / (ustar * ustar))
        psi_m_u, psi_h_u = psi(zeta_u)
        _, psi_h_t = psi(zeta_t)
        _, psi_h_q = psi(zeta_q)
        u10n = u / (1 + cd_n10_rt * (log_u - psi_m_u) / vk)                        # 9a
        t10 = t_air - (tstar / vk) * (np.log(z_t / z_u) + psi_h_u - psi_h_t)       # 9b
        q10 = q_air - (qstar / vk) * (np.log(z_q / z_u) + psi_h_u - psi_h_q)       # 9c
        cd_n10, cd_n10_rt, ce_n10, ch_n10 = neutral(u10n, zeta_u)                 # :1025-1029
        xx = (log_u - psi_m_u) / vk
        cd = cd_n10 / (1 + cd_n10_rt * xx) ** 2                                    # 10a
        xx = (log_u - psi_h_u) / vk
        ch = ch_n10 / (1 + ch_n10 * xx / cd_n10_rt) * np.sqrt(cd / cd_n10)         # 10b
        ce = ce_n10 / (1 + ce_n10 * xx / cd_n10_rt) * np.sqrt(cd / cd_n10)         # 10c

    # fluxes (:423-461); θ_a and q_a are the values moved to z_u (:409-410)
    rho = p / (FMS_RDGAS * tv_atm)
    flux_t = FMS_CP_AIR * (ch * w_atm) * rho * (sst - t10)
    flux_q = (ce * w_atm) * rho * (q_s - q10)
    rho_drag = (cd * w_atm) * rho
    return TurbulentFluxes(
        tau_x=-(rho_drag * u_dif), tau_y=-(rho_drag * v_dif),
        q_sens=-flux_t, q_lat=-FMS_HLV * flux_q, evap=flux_q / RHO_FW,
        cd=cd, ch=ch, ce=ce, rho_air=rho, ustar=ustar, u10n=u10n, zeta=zeta_u,
    )


# ---------------------------------------------------------------------------
# Kara et al. (2005) polynomial fits to COARE 3.0 — HYCOM flxflg=6
# ---------------------------------------------------------------------------
#
# Coefficient tables of Kara, Hurlburt & Wallcraft (2005), as used in HYCOM
# (thermf.F90, flxflg=4/6 parameter block; mod_momtum.F90, cd_coarep).
# Key "s"/"u"/"n" = stable / unstable / neutral, "0"/"5" = low (Va ≤ 5 m/s) /
# high wind.  Entry [i][j] multiplies τⁱ·xʲ, with τ = virtual Ta − Ts and
# x = Va or 1/Va as described in _kara_poly.

#: Latent-heat exchange coefficient C_l (dimensionless), 3 × 3 per regime.
_CL = {
    "s0": [[-2.925e-4, 5.498e-4, -5.544e-5],
           [7.272e-5, -1.740e-4, 2.489e-5],
           [-6.948e-6, 1.637e-5, -2.618e-6]],
    "s5": [[1.023e-3, 9.657e-6, -2.281e-8],
           [-2.672e-6, 2.103e-4, -5.329e-3],
           [1.546e-6, -6.228e-5, 5.094e-4]],
    "u0": [[2.077e-3, -3.933e-4, 3.971e-5],
           [-2.899e-4, 7.350e-5, -6.267e-6],
           [-1.954e-5, 5.483e-6, -4.867e-7]],
    "u5": [[1.074e-3, 5.579e-6, 5.263e-8],
           [6.912e-6, -2.244e-4, -1.027e-3],
           [1.849e-7, -2.167e-6, -1.010e-4]],
    "n0": [1.14086e-3, -3.120e-6, -9.300e-7],
    "n5": [1.073e-3, 5.531e-6, 5.433e-8],
}

#: Drag coefficient 10³·C_d, 3 × 4 per regime (cubic in Va or 1/Va).
_CD = {
    "s0": [[-0.06695, 0.3133, -0.001473, -0.004056],
           [0.09966, -2.116, 4.626, -2.680],
           [-0.02477, 0.2726, -0.5558, 0.3139]],
    "s5": [[0.55815, 0.08174, -0.0004472, 2.666e-6],
           [-0.005593, 0.2096, -8.634, 18.63],
           [0.0006024, -0.02629, 0.2121, 0.7755]],
    "u0": [[1.891, -0.7182, 0.1975, -0.01790],
           [-0.006304, -0.3028, 0.3120, -0.1210],
           [0.0004406, -0.01769, 0.01303, -0.003394]],
    "u5": [[0.6497, 0.06993, 3.541e-5, -3.428e-6],
           [0.003827, -0.2756, -1.091, 4.946],
           [-4.83e-5, 0.007710, -0.2555, 0.7654]],
    "n0": [1.057, -0.06949, 0.01271],
    "n5": [0.6825, 0.06945, -0.0001029],
}

# HYCOM constants (thermf.F90; mod_momtum.F90)
_HY_TZERO: float = 273.16    # K        Celsius → Kelvin offset used inside HYCOM
_HY_RGAS: float = 287.1      # J kg⁻¹ K⁻¹
_HY_CSUBP: float = 1005.7    # J kg⁻¹ K⁻¹ specific heat of air
_HY_EVAPLH: float = 2.47e6   # J kg⁻¹   latent heat of evaporation
_HY_RHOREF: float = 1000.0   # kg m⁻³   reference density (svref = 1/rhoref)


def _kara_poly(c: list[list[float]], tau: np.ndarray, va: np.ndarray,
               tau_groups_in_inverse_va: bool) -> np.ndarray:
    """
    Evaluate Σᵢ (Σⱼ c[i][j] xᵢʲ) τⁱ, i = 0, 1, 2.

    The τ⁰ group always uses x₀ = Va.  The τ¹ and τ² groups use x = 1/Va
    when ``tau_groups_in_inverse_va`` (high-wind C_l and all C_d fits),
    otherwise x = Va (low-wind C_l fit)            (Kara et al. 2005, eqs 9–12).
    """
    x = 1.0 / va if tau_groups_in_inverse_va else va
    out = np.polynomial.polynomial.polyval(va, c[0])
    out = out + np.polynomial.polynomial.polyval(x, c[1]) * tau
    out = out + np.polynomial.polynomial.polyval(x, c[2]) * tau ** 2
    return out


def _kara_blend(tau: np.ndarray, stable: np.ndarray, unstable: np.ndarray,
                p_stable_075: np.ndarray, p_unstable_075: np.ndarray,
                neutral: np.ndarray, squared: bool) -> np.ndarray:
    """
    Combine regime fits across Ta−Ts (Kara et al. 2005; HYCOM thermf/momtum).

        τ ≥ 0.75            : stable fit
        τ ≤ −0.75           : unstable fit
        −0.098 ≤ τ < 0.75   : blend of stable fit at τ = 0.75 and neutral fit
        −0.75 < τ < −0.098  : blend of unstable fit at τ = −0.75 and neutral fit

    For C_l (``squared=True``) the blend weight of the neutral fit is
    q = ((τ − 0.75)/0.848)² or ((τ + 0.75)/0.652)²; for C_d it is linear,
    1 − q with q = (τ + 0.098)/0.848 or (−τ − 0.098)/0.652.  Both versions
    are continuous at τ = ±0.75.
    """
    if squared:
        w_pos = ((tau - 0.75) / 0.848) ** 2          # weight of neutral fit
        w_neg = ((tau + 0.75) / 0.652) ** 2
    else:
        w_pos = 1.0 - (tau + 0.098) / 0.848
        w_neg = 1.0 - (-tau - 0.098) / 0.652
    blend_pos = (1.0 - w_pos) * p_stable_075 + w_pos * neutral
    blend_neg = (1.0 - w_neg) * p_unstable_075 + w_neg * neutral
    return np.select(
        [tau >= 0.75, tau <= -0.75, tau >= -0.098],
        [stable, unstable, blend_pos],
        default=blend_neg,
    )


def _kara_coefficient(table: dict, tau: np.ndarray, va: np.ndarray,
                      low_wind_inverse: bool, squared: bool) -> np.ndarray:
    """Exchange coefficient from a Kara et al. (2005) table, both wind ranges."""
    out = {}
    for band, inverse in (("0", low_wind_inverse), ("5", True)):
        s, u = table["s" + band], table["u" + band]
        out[band] = _kara_blend(
            tau,
            stable=_kara_poly(s, tau, va, inverse),
            unstable=_kara_poly(u, tau, va, inverse),
            p_stable_075=_kara_poly(s, np.full_like(tau, 0.75), va, inverse),
            p_unstable_075=_kara_poly(u, np.full_like(tau, -0.75), va, inverse),
            neutral=np.polynomial.polynomial.polyval(va, table["n" + band]),
            squared=squared,
        )
    return np.where(va <= 5.0, out["0"], out["5"])


def _hycom_satvpr(t_c: np.ndarray) -> np.ndarray:
    """
    Saturation vapour pressure over water [Pa], t in °C.

        e_s = 100 Σₖ aₖ tᵏ,  k = 0…6                 (Lowe 1977, Table 1)
    """
    a = [6.107799961, 4.436518521e-1, 1.428945805e-2, 2.650648471e-4,
         3.031240396e-6, 2.034080948e-8, 6.136820929e-11]
    return 100.0 * np.polynomial.polynomial.polyval(t_c, a)


def _hycom_qsat_mixing(t_c: np.ndarray, p: np.ndarray, f: np.ndarray | float = 1.0) -> np.ndarray:
    """Saturation mixing ratio 0.622 f e_s / (p − f e_s) [kg kg⁻¹] (HYCOM qsatur6)."""
    es = f * _hycom_satvpr(t_c)
    return 0.622 * es / (p - es)


def bulk_kara(
    u10: np.ndarray,
    v10: np.ndarray,
    t_air: np.ndarray,
    q_air: np.ndarray,
    sst: np.ndarray,
    p: np.ndarray | None = None,
    *,
    sss: np.ndarray | float | None = None,
    u_ocean: np.ndarray | float = 0.0,
    v_ocean: np.ndarray | float = 0.0,
) -> TurbulentFluxes:
    """
    HYCOM flxflg=6 / wndflg=4 bulk fluxes (Kara, Hurlburt & Wallcraft 2005).

    Exchange coefficients are closed-form polynomial fits to COARE 3.0
    (Fairall et al. 2003), quadratic in a virtual air–sea temperature
    difference and in Va or 1/Va; there is no stability iteration.  This
    reproduces HYCOM thermf.F90 (flxflg=6) and mod_momtum.F90 (cd_coarep)
    to round-off — see ``tests/test_bulk_kara.py``, which compares against
    the compiled HYCOM routines.

        r    = q / (1 − q)                     mixing ratio (HYCOM ``vapmix``)
        ρ_a  = p / [R (T_a + 273.16)(1 + 0.608 r)]
        τ_v  = (T_a − T_s) − 0.608 (T_a + 273.16)(r_sat(T_a, p) − r),  τ_v ∈ [−8, 7] °C
        C_l  = fit(τ_v, Va∈[1.2, 34]),  C_s = 0.9554 C_l,  C_d = fit(τ_v, Va∈[1, 34])
        f_S  = 1 − (0.001/1.805) max(S − 0.03, 0)      (Sud & Walker 1997)
        Q_L  = −L_v ρ_a C_l |ΔU| (r_sat(T_s, p, f_S) − r)
        Q_S  = −c_pa ρ_a C_s |ΔU| (T_s − T_a)
        τ    = ρ_a C_d |ΔU| ΔU
        E    = −Q_L / (L_v ρ_ref)

    with ΔU = U₁₀ − u_ocean (HYCOM amoflg with ocnscl=1), T in °C.  The
    temperature and humidity are used at their measurement height (2 m)
    without adjustment, as in HYCOM.  There is no minimum wind speed in the
    fluxes: calm air gives zero turbulent fluxes.

    Parameters
    ----------
    u10, v10 : np.ndarray
        10 m wind [m s⁻¹].
    t_air : np.ndarray
        Air temperature [K]; converted to °C with 273.15, as in the HYCOM-IOA
        forcing preprocessing (HYCOM then uses 273.16 internally).
    q_air : np.ndarray
        Specific humidity [kg kg⁻¹].
    sst : np.ndarray
        Model surface temperature [°C].
    p : np.ndarray
        Sea-level pressure [Pa] (required, HYCOM mslprf=1).
    sss : np.ndarray or float
        Model surface salinity [psu] (required).
    u_ocean, v_ocean : np.ndarray or float
        Surface current at T-points [m s⁻¹]; HYCOM averages the top
        ``thkcdw`` = 3 m plus the barotropic velocity.

    Returns
    -------
    TurbulentFluxes
        ``ce`` holds C_l and ``ch`` holds C_s; ``u10n`` and ``zeta`` are None.

    References
    ----------
    Kara, A.B., Hurlburt, H.E. & Wallcraft, A.J. (2005). Stability-dependent
        exchange coefficients for air–sea fluxes. J. Atmos. Oceanic Technol.
        22, 1080–1094.
    Lowe, P.R. (1977). An approximating polynomial for the computation of
        saturation vapor pressure. J. Appl. Meteor. 16, 100–103.
    Sud, Y.C. & Walker, G.K. (1997). Influence of ocean salinity on the
        saturation vapour pressure (via Witting 1908); as cited in HYCOM thermf.F90.
    """
    if p is None:
        raise ValueError("bulk_kara needs p (sea-level pressure) — HYCOM flxflg=6 uses mslprs")
    if sss is None:
        raise ValueError("bulk_kara needs sss (surface salinity) for the vapour-pressure depression")

    airt = np.asarray(t_air, dtype=float) - T0_K
    q_air = np.asarray(q_air, dtype=float)
    vpmx = q_air / (1.0 - q_air)
    sst = np.asarray(sst, dtype=float)
    p = np.asarray(p, dtype=float)

    du = np.asarray(u10, dtype=float) - u_ocean
    dv = np.asarray(v10, dtype=float) - v_ocean
    speed = np.hypot(du, dv)

    rho_air = p / (_HY_RGAS * (_HY_TZERO + airt) * (1.0 + 0.608 * vpmx))
    tau_v = (airt - sst) - 0.608 * (airt + _HY_TZERO) * (_hycom_qsat_mixing(airt, p) - vpmx)
    tau_v = np.clip(tau_v, -8.0, 7.0)

    cl = _kara_coefficient(_CL, tau_v, np.clip(speed, 1.2, 34.0),
                           low_wind_inverse=False, squared=True)
    cs = 0.9554 * cl
    cd = 1e-3 * _kara_coefficient(_CD, tau_v, np.clip(speed, 1.0, 34.0),
                                  low_wind_inverse=True, squared=False)

    f_sss = 1.0 - (0.001 / 1.805) * np.maximum(np.asarray(sss, dtype=float) - 0.03, 0.0)
    q_lat = -_HY_EVAPLH * rho_air * cl * speed * (_hycom_qsat_mixing(sst, p, f_sss) - vpmx)
    q_sens = -_HY_CSUBP * rho_air * cs * speed * (sst - airt)

    return TurbulentFluxes(
        tau_x=rho_air * cd * speed * du,
        tau_y=rho_air * cd * speed * dv,
        q_sens=q_sens,
        q_lat=q_lat,
        evap=-q_lat / (_HY_EVAPLH * _HY_RHOREF),
        cd=cd, ch=cs, ce=cl, rho_air=rho_air,
        ustar=np.sqrt(cd) * speed,
    )


# ---------------------------------------------------------------------------
# Radiation
# ---------------------------------------------------------------------------

def albedo_large_yeager(lat: np.ndarray) -> np.ndarray:
    """
    Latitude-dependent ocean albedo, 5.8 % at the equator and 8 % at the poles.

        α = 0.069 − 0.011 cos(2φ)            (Large & Yeager 2009; HYCOM albflg=2)

    Parameters
    ----------
    lat : np.ndarray
        Latitude [°N].
    """
    return 0.069 - 0.011 * np.cos(2.0 * np.deg2rad(lat))

def radiative_fluxes(
    sw_down: np.ndarray,
    lw_down: np.ndarray,
    sst: np.ndarray,
    albedo: float | np.ndarray = 0.066,
    emissivity: float = 0.97,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Net shortwave and longwave radiation at the sea surface, positive down.

        Q_SW = (1 − α) SW↓                               (LY04 eq 2; α = 0.066)
        Q_LW = ε (LW↓ − σ SST⁴)

    The emissivity multiplies both terms: the ocean absorbs ε LW↓ and
    reflects (1 − ε) LW↓ (Kirchhoff's law), as in MITgcm ``exf_radiation.F``.
    Default ε = 0.97 (MITgcm ``ocean_emissivity``).

    Parameters
    ----------
    sw_down, lw_down : np.ndarray
        Downwelling shortwave and longwave radiation [W m⁻²].
    sst : np.ndarray
        Sea surface temperature [°C].
    albedo : float or np.ndarray
        Ocean shortwave albedo [–].
    emissivity : float
        Ocean longwave emissivity [–].

    Returns
    -------
    q_sw, q_lw : np.ndarray
        Net shortwave and net longwave flux into the ocean [W m⁻²].
    """
    sst_k = np.asarray(sst, dtype=float) + T0_K
    q_sw = (1.0 - albedo) * np.asarray(sw_down, dtype=float)
    q_lw = emissivity * (np.asarray(lw_down, dtype=float) - SIGMA_SB * sst_k ** 4)
    return q_sw, q_lw


# ---------------------------------------------------------------------------
# Canonical atmosphere → surface_tendency arguments
# ---------------------------------------------------------------------------

_SCHEMES = {"ncar": bulk_ncar, "ncar_fms": bulk_ncar_fms, "kara": bulk_kara}
_SST_KELVIN = {"ncar_fms"}       # schemes that take the SST in K, as FMS


def compute_surface_fluxes(
    atm: xr.Dataset,
    sst: np.ndarray,
    u_sfc: np.ndarray | float = 0.0,
    v_sfc: np.ndarray | float = 0.0,
    *,
    scheme: str = "ncar",
    albedo: float | np.ndarray = 0.066,
    emissivity: float = 0.97,
    **scheme_kwargs,
) -> dict[str, np.ndarray]:
    """
    Surface fluxes for ``forcing.surface.surface_tendency`` from canonical atmosphere.

        Q_net = Q_SW + Q_LW + Q_S + Q_L,   E − P = E − P_rate

    Parameters
    ----------
    atm : xr.Dataset
        Canonical atmosphere already interpolated to model T-points and to
        the model time (``forcing.interp``): u10, v10, t2m, q2m, msl,
        sw_down, lw_down and optionally precip.
    sst : np.ndarray
        Model surface temperature θ[k=0] [°C].
    u_sfc, v_sfc : np.ndarray or float
        Surface current at T-points [m s⁻¹] (0 for absolute wind).
    scheme : {'ncar', 'ncar_fms', 'kara'}
        Bulk algorithm: 'ncar' (Large & Yeager 2004, MITgcm), 'ncar_fms' (the
        same equations in the form of the FMS coupler of MOM6; SST passed in K,
        T and q heights through ``z_t``/``z_q`` in scheme_kwargs) or 'kara'
        (Kara et al. 2005, HYCOM flxflg=6; needs ``sss=`` in scheme_kwargs).
    albedo, emissivity : see ``radiative_fluxes``.
    **scheme_kwargs
        Passed to the bulk algorithm (e.g. ``n_iter``, ``high_wind``).

    Returns
    -------
    dict
        ``tau_x``, ``tau_y`` [N m⁻²], ``Q_net``, ``Q_sw`` [W m⁻²] and
        ``E_P`` [m s⁻¹], matching the keyword arguments of
        ``surface_tendency``.
    """
    if scheme not in _SCHEMES:
        raise ValueError(f"unknown bulk scheme {scheme!r}; available: {sorted(_SCHEMES)}")

    sst_in = np.asarray(sst, dtype=float) + T0_K if scheme in _SST_KELVIN else sst
    turb = _SCHEMES[scheme](
        atm["u10"].values, atm["v10"].values, atm["t2m"].values, atm["q2m"].values,
        sst_in, p=atm["msl"].values, u_ocean=u_sfc, v_ocean=v_sfc, **scheme_kwargs,
    )
    q_sw, q_lw = radiative_fluxes(
        atm["sw_down"].values, atm["lw_down"].values, sst, albedo=albedo, emissivity=emissivity,
    )
    precip = atm["precip"].values if "precip" in atm else 0.0

    return {
        "tau_x": turb.tau_x,
        "tau_y": turb.tau_y,
        "Q_net": q_sw + q_lw + turb.q_sens + turb.q_lat,
        "Q_sw": q_sw,
        "E_P": turb.evap - precip,
    }
