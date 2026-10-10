"""
Air–sea bulk flux algorithms for pyOCEAN.

Turns canonical near-surface atmospheric state (``forcing/canonical.py``)
plus model SST and surface current into the fluxes consumed by
``forcing.surface.surface_tendency``: wind stress, net and shortwave heat
flux, and evaporation minus precipitation.

Implemented scheme
------------------
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
    cd, ch, ce : transfer coefficients at the wind height [–]
    u10n : neutral 10 m wind speed [m s⁻¹]
    ustar : friction velocity [m s⁻¹]
    zeta : Monin–Obukhov stability parameter z_u/L at the wind height [–]
    """

    tau_x: np.ndarray
    tau_y: np.ndarray
    q_sens: np.ndarray
    q_lat: np.ndarray
    evap: np.ndarray
    cd: np.ndarray
    ch: np.ndarray
    ce: np.ndarray
    u10n: np.ndarray
    ustar: np.ndarray
    zeta: np.ndarray


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
    MITgcm (``niter_bulk = 2``).  ζ is clipped to |ζ| ≤ 10 as in MITgcm and
    HYCOM.

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
        cd=cd, ch=ch, ce=ce, u10n=u10n, ustar=ustar, zeta=zeta_u,
    )


# ---------------------------------------------------------------------------
# Radiation
# ---------------------------------------------------------------------------

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

_SCHEMES = {"ncar": bulk_ncar}


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
    scheme : {'ncar'}
        Bulk algorithm.
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

    turb = _SCHEMES[scheme](
        atm["u10"].values, atm["v10"].values, atm["t2m"].values, atm["q2m"].values,
        sst, p=atm["msl"].values, u_ocean=u_sfc, v_ocean=v_sfc, **scheme_kwargs,
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
