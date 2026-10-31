"""
COARE 3.0 bulk air–sea fluxes as CROCO computes them (port of ``OCEAN/bulk_flux.F``).

Line-by-line port of the default path of CROCO's ``bulk_flux_tile``
(~/pyOCEAN_refs/croco, git ab63b58): COARE 3.0 (Fairall et al. 2003) with
``BULK_GUSTINESS`` and ``BULK_LW`` (cppdefs.h:178-185), three iterations, double
precision.  It exists so that pyOCEAN can be forced exactly as CROCO when the two
models are compared (spec N2b, decision: CROCO is the third model); it is a reference
implementation, not the physics of Fairall et al. where the two differ.  The reference
read with every quirk is ``docs/design/n2b_refs/croco_bulk.md`` (§2, §5); the port
equals the compiled ``bulk_flux.F`` on 250 points to ≤ 1e-10 relative
(validation/bulk_points/, tests/test_bulk_coare.py).

CROCO quirks reproduced on purpose
----------------------------------
* ``CtoK = 273.16`` K, not 273.15 (params_bulk.h:1).
* Sea-surface q_sat and the surface Exner function use the constant ``psurf`` = 1e5 Pa
  (params_bulk.h:30-33, bulk_flux.F:177, 235); here ``p_sfc``, default 1e5.
* The Charnock coefficient is set once from the initial gusty wind √(w² + 0.5²)
  and stays fixed through the iterations (bulk_flux.F:247, 497-503).
* ψ_t(0) = −0.004524 ≠ 0: the stable branch includes ζ = 0 and its constants are
  rounded (0.6667, 8.525; bulk_flux.F:952, 964-966).
* After the third iteration u* is not recomputed with the last gusty wind, so
  C_d = (u*/ΔU)² mixes iteration 3's u* (made with iteration 2's ΔU) with iteration 3's ΔU
  (bulk_flux.F:536, 556, 577, 658).
* Webb correction added to the latent heat flux (bulk_flux.F:621-624).
* Outgoing longwave 0.985 σ SST⁴ with σ = 5.6697e-8, no reflected (1 − ε) LW↓ term
  (bulk_flux.F:617-618, params_bulk.h:27-28).
* Evaporation from the kinematic latent flux, E = −C_p hflat/L_v, i.e. a volume flux at
  the seawater reference density ρ₀ (bulk_flux.F:636), not at 1000 kg m⁻³.
* Absolute wind (no surface current, SFLUX_CFB undefined); the stress is
  τ = ρ_a C_d ΔU u with ΔU the gusty wind and u the wind component (bulk_flux.F:657-659,
  706-712).
* ζ is not clipped (unlike the ECUME branch, bulk_flux.F:365).

Units: temperatures in °C, humidity kg kg⁻¹, wind m s⁻¹ at z_u, radiation W m⁻².
Outputs are positive **into the ocean**; ``evap`` is positive when water leaves.

References
----------
Fairall, C.W., Bradley, E.F., Hare, J.E., Grachev, A.A. & Edson, J.B. (2003). Bulk
    parameterization of air–sea fluxes: updates and verification for the COARE
    algorithm. J. Climate 16, 571–591.
Webb, E.K., Pearman, G.I. & Leuning, R. (1980). Correction of flux measurements for
    density effects due to heat and water vapour transfer. Q. J. R. Meteorol. Soc. 106,
    85–100.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

# ---------------------------------------------------------------------------
# Constants: params_bulk.h (CROCO git ab63b58) and scalars.h
# ---------------------------------------------------------------------------
CTOK: float = 273.16                       # K            °C → K (params_bulk.h:1)
BLK_RGAS: float = 287.0596736665907        # J kg⁻¹ K⁻¹   dry-air gas constant (:2)
BLK_RVAP: float = 461.5249933083879        # J kg⁻¹ K⁻¹   water-vapour gas constant (:3)
BLK_CPA: float = 1004.708857833067         # J kg⁻¹ K⁻¹   c_p of dry air (:4)
IP00: float = 1.0e-5                       # Pa⁻¹         1/p₀ (:5)
RDOCPD: float = BLK_RGAS / BLK_CPA         # –            R_d/c_pd (:7)
R_GAS: float = 8.314510                    # J mol⁻¹ K⁻¹  universal gas constant (:9)
MM_DRYAIR: float = 28.9644e-3              # kg mol⁻¹     molar mass of dry air (:10)
MM_WATER: float = 18.0153e-3               # kg mol⁻¹     molar mass of water (:11)
CPVIR: float = BLK_RVAP / BLK_RGAS - 1.0   # –            virtual-temperature factor (:12)
MVOMA: float = MM_WATER / MM_DRYAIR        # –            M_v/M_a (:13)
EPS: float = 1.0e-8                        # –            guard in z/L (:14)
R3: float = 1.0 / 3.0                      # –            (:15)
PIS2: float = 2.0 * np.arctan(1.0)         # –            π/2 (:16)
SQR3: float = np.sqrt(3.0)                 # –            √3 (:17)
PIS2OSQR3: float = PIS2 / SQR3             # –            (:18)
BLK_BETA: float = 1.2                      # –            gustiness β (:21)
BLK_ZABL: float = 600.0                    # m            boundary-layer height z_i (:22)
EMISS_LW: float = 0.985                    # –            sea-surface emissivity (:27)
SIGMA_SB_CROCO: float = 5.6697e-8          # W m⁻² K⁻⁴    Stefan–Boltzmann, CROCO value (:28)
PSURF: float = 1.0e5                       # Pa           constant surface pressure (:30-33)
G_CROCO: float = 9.81                      # m s⁻²        scalars.h:633
CP_CROCO: float = 3985.0                   # J kg⁻¹ K⁻¹   seawater c_p, scalars.h:640
VONKAR: float = 0.41                       # –            von Kármán, scalars.h:643
RHO0_CROCO: float = 1025.0                 # kg m⁻³       croco.in:61
G_BARO: float = 9.80665                    # m s⁻²        local g of exner_patm (bulk_flux.F:787)
ITER_FL: int = 3                           # –            iterations (bulk_flux.F:48)


@dataclass
class CoareFluxes:
    """
    Output of ``bulk_coare_croco``, all at T-points, fluxes positive into the ocean.

    Attributes
    ----------
    tau_x, tau_y : wind stress [N m⁻²] (ρ₀ × CROCO's sustr, svstr)
    q_sens : sensible heat flux [W m⁻²]
    q_lat : latent heat flux with the Webb correction [W m⁻²]
    q_lw : net longwave LW↓ − 0.985 σ SST⁴ [W m⁻²]
    evap : evaporation [m s⁻¹], positive when water leaves the ocean (volume at ρ₀)
    cd : drag coefficient (u*/ΔU)² [–]
    rho_air : air density at z_t [kg m⁻³]
    ustar : friction velocity u* [m s⁻¹]
    ch : transfer coefficient of heat and moisture, Q_S = ρ_a c_pa C_h ΔU Δθ [–]
        (CROCO uses one coefficient for both, bulk_flux.F:545-547)
    """

    tau_x: np.ndarray
    tau_y: np.ndarray
    q_sens: np.ndarray
    q_lat: np.ndarray
    q_lw: np.ndarray
    evap: np.ndarray
    cd: np.ndarray
    rho_air: np.ndarray
    ustar: np.ndarray
    ch: np.ndarray


# ---------------------------------------------------------------------------
# Functions of bulk_flux.F:758-972
# ---------------------------------------------------------------------------

def _qsat(t_k: np.ndarray, patm: np.ndarray | float, coeff: float) -> np.ndarray:
    """
    Saturation specific humidity [kg kg⁻¹] (``qsat``, bulk_flux.F:808-835).

        p_sat = coeff · exp(α − β/T − γ ln T)    [Pa], water for T > CtoK, ice otherwise
        q_sat = (M_v/M_a) p_sat / (p + (M_v/M_a − 1) p_sat)

    t_k : temperature [K]; patm : pressure [Pa]; coeff : 0.98 at the sea surface, 1 in air.
    """
    ice = t_k <= CTOK                                                          # :823
    psat = np.where(
        ice,
        np.exp(32.62117980819471 - 6295.421338904806 / t_k
               - 0.5631331575423155 * np.log(t_k)),                            # :824
        np.exp(60.2227554 - 6822.40088 / t_k - 5.13926744 * np.log(t_k)),     # :826
    )
    psat = coeff * psat                                                        # :828
    return (MVOMA * psat) / (patm + (MVOMA - 1.0) * psat)                      # :830


def _exner_patm(q: np.ndarray, t_k: np.ndarray, z: float,
                psfc: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Inverse Exner function and pressure [Pa] at height z [m] (bulk_flux.F:779-804).

    Three fixed-point iterations of the hypsometric equation with the moist molar mass:

        x_m = M_a + (q/q_sat(T, p)) (M_w − M_a),   p = p_sfc exp(−g x_m z/(R T)),
        1/Π = (p/p₀)^(−R_d/c_pd);   g = 9.80665 here (:787), not the model's 9.81.
    """
    pair = np.full_like(t_k, psfc)                                             # :791
    for _ in range(3):                                                         # :789, 793
        q_sat = _qsat(t_k, pair, 1.0)                                          # :794
        xm = MM_DRYAIR + (q / q_sat) * (MM_WATER - MM_DRYAIR)                  # :795
        pair = psfc * np.exp(-(G_BARO * xm * z / (R_GAS * t_k)))               # :796
    iexn = (pair * IP00) ** (-RDOCPD)                                          # :801
    return iexn, pair


def _air_visc(t_c: np.ndarray) -> np.ndarray:
    """Kinematic viscosity of air [m² s⁻¹] at t_c [°C] (``air_visc``, bulk_flux.F:898-908)."""
    cff = t_c * t_c                                                            # :905
    return 1.326e-5 * (1.0 + 6.542e-3 * t_c + 8.301e-6 * cff - 4.84e-9 * cff * t_c)  # :906


def _psiu_coare(zol: np.ndarray) -> np.ndarray:
    """
    COARE stability function for momentum ψ_u(ζ) (``bulk_psiu_coare``, bulk_flux.F:912-938).

    ζ ≤ 0: Kansas form (χ = (1 − 15ζ)^¼) blended with the free-convection form
    (χ_c = (1 − 10.15ζ)^⅓) as ψ_c + (ψ_k − ψ_c)/(1 + ζ²) (Fairall et al. 1996);
    ζ > 0: −((1 + ζ) + 0.6667 (ζ − 14.28) e^{−min(50, 0.35ζ)} + 8.525) (Beljaars & Holtslag 1991).
    """
    zu = np.minimum(zol, 0.0)                     # each branch on its own domain, no NaN
    chik = (1.0 - 15.0 * zu) ** 0.25                                           # :923
    psik = (2.0 * np.log(0.5 * (1.0 + chik)) + np.log(0.5 * (1.0 + chik ** 2))
            - 2.0 * np.arctan(chik) + PIS2)                                    # :924-925
    chic = (1.0 - 10.15 * zu) ** R3                                            # :926
    psic = (1.5 * np.log(R3 * (chic ** 2 + chic + 1.0))
            - SQR3 * np.arctan((2.0 * chic + 1.0) / SQR3) + 2.0 * PIS2OSQR3)   # :927-928
    unstable = psic + (psik - psic) / (1.0 + zu ** 2)                          # :929
    zs = np.maximum(zol, 0.0)
    chis = -np.minimum(50.0, 0.35 * zs)                                        # :931
    stable = -((1.0 + zs) + 0.6667 * (zs - 14.28) * np.exp(chis) + 8.525)      # :932
    return np.where(zol <= 0.0, unstable, stable)                              # :922


def _psit_coare(zol: np.ndarray) -> np.ndarray:
    """
    COARE stability function for heat and moisture ψ_t(ζ) (``bulk_psit_coare``, :942-972).

    ζ < 0: ψ_k = 2 ln(½(1 + χ²)), χ = (1 − 15ζ)^¼, blended with χ_c = (1 − 34.15ζ)^⅓;
    ζ ≥ 0 (includes 0, so ψ_t(0) = −0.004524):
    −((1 + 2ζ/3)^1.5 + 0.6667 (ζ − 14.28) e^{−min(50, 0.35ζ)} + 8.525).
    """
    zu = np.minimum(zol, 0.0)
    chik = (1.0 - 15.0 * zu) ** 0.25                                           # :953
    psik = 2.0 * np.log(0.5 * (1.0 + chik ** 2))                               # :954
    chic = (1.0 - 34.15 * zu) ** R3                                            # :955
    psic = (1.5 * np.log((chic ** 2 + chic + 1.0) * R3)
            - SQR3 * np.arctan((2.0 * chic + 1.0) / SQR3) + 2.0 * PIS2OSQR3)   # :956-958
    unstable = psic + (psik - psic) / (1.0 + zu ** 2)                          # :959
    zs = np.maximum(zol, 0.0)
    chis = -np.minimum(50.0, 0.35 * zs)                                        # :964
    stable = -((1.0 + 2.0 * zs / 3.0) ** 1.5
               + 0.6667 * (zs - 14.28) * np.exp(chis) + 8.525)                 # :965-966
    return np.where(zol < 0.0, unstable, stable)                               # :952


def _charnock(delw: np.ndarray) -> np.ndarray:
    """
    Charnock coefficient of COARE 3.0 from the wind ΔU [m s⁻¹] (bulk_flux.F:498-503):
    0.011 up to 10 m s⁻¹, linear to 0.018 at 18 m s⁻¹, 0.018 above.
    """
    return np.where(delw > 18.0, 0.018,
                    np.where(delw > 10.0, 0.011 + 0.125 * (0.018 - 0.011) * (delw - 10.0),
                             0.011))


# ---------------------------------------------------------------------------
# Scalar block of bulk_flux_tile, bulk_flux.F:199-659
# ---------------------------------------------------------------------------

def _coare_core(wspd: np.ndarray, t_air_c: np.ndarray, q_air: np.ndarray,
                sst_c: np.ndarray, *, p_sfc: float = PSURF, z_u: float = 10.0,
                z_t: float = 2.0, gustiness: bool = True) -> dict[str, np.ndarray]:
    """
    COARE 3.0 iteration of CROCO for the scales u*, θ*, q* (bulk_flux.F:199-579).

    Returns a dict with ``wstar, tstar, qstar`` (u* [m s⁻¹], θ* [K], q* [kg kg⁻¹]),
    ``delw`` (final gusty wind ΔU [m s⁻¹]), ``cff`` (κ/(ln(z_t/z₀t) − ψ_t), last iteration),
    ``qsea`` [kg kg⁻¹], ``charn``, ``patm`` (pressure at z_t [Pa]), ``rho_air`` [kg m⁻³]
    and the input-derived ``t_air_k``, ``sst_k``, ``q``.

    Steps (croco_bulk.md §5):

        w₀ = max(w, 0.1 min(10, z_u))                                            :222
        ρ_a = p (1 + q)/(R_d T (1 + (M_v/M_a) q)),  p at z_t (exner_patm)        :231-233
        q_s = q_sat(SST, p_sfc, 0.98)                                            :235
        ΔU = √(w₀² + 0.5²) (gustiness) | w₀;  Δq = q − q_s                       :247, 260
        Δθ = T_a/Π_a − SST/Π_s + CtoK (1/Π_a − 1/Π_s)                            :262-263
        first guess from a bulk Richardson number (Fairall et al. 2003)          :429-472
        3 × [z₀ = (α u*²/g + 0.11 ν/u*), z₀t = min(1.15e-4, 5.5e-5 Re*^−0.6),
             ζ = κ g z_u (θ* (1 + ε q) + ε T q*)/(T u*² (1 + ε q) + 1e-8),
             u* = κ ΔU/(ln(z_u/z₀) − ψ_u(ζ)),  θ*, q* = κ (Δθ, Δq)/(ln(z_t/z₀t) − ψ_t(ζ z_t/z_u)),
             B_f = −g/T u* (θ* + ε T q*),  ΔU = √(w₀² + (β (B_f z_i)^⅓ | 0.2)²)]   :511-579
    """
    t_air_c = np.asarray(t_air_c, dtype=float)
    shape = np.broadcast(wspd, t_air_c, q_air, sst_c).shape
    wspd, t_air_c, q_air, sst_c = (np.broadcast_to(np.asarray(a, dtype=float), shape)
                                   for a in (wspd, t_air_c, q_air, sst_c))
    z_t_over_z_u = z_t / z_u                                                   # :97
    log10_o_logzw = np.log(10.0 * 10000.0) / np.log(z_u * 10000.0)            # :110-111
    iexns = (p_sfc * IP00) ** (-RDOCPD)                                        # :177

    wspd0 = np.maximum(wspd, 0.1 * min(10.0, z_u))                             # :199, 222
    tair_c = t_air_c                                                           # :204
    tair_k = tair_c + CTOK                                                     # :205
    q = q_air                                     # given q, instead of spec_hum(RH) (:206-207)
    tsea_c = sst_c                                                             # :227
    tsea_k = tsea_c + CTOK                                                     # :229
    iexna, patm = _exner_patm(q, tair_k, z_t, p_sfc)                           # :231
    rho_air = patm * (1.0 + q) / (BLK_RGAS * tair_k * (1.0 + MVOMA * q))       # :233
    qsea = _qsat(tsea_k, p_sfc, 0.98)                                          # :235
    delw = np.sqrt(wspd0 * wspd0 + 0.25) if gustiness else wspd0               # :247, 252
    delq = q - qsea                                                            # :260
    cff = CTOK * (iexna - iexns)                                               # :262
    delt = tair_c * iexna - tsea_c * iexns + cff                               # :263

    # initial guess (:429-472)
    wstar = 0.035 * delw * log10_o_logzw                                       # :429
    visair = _air_visc(tair_c)                                                 # :430
    charn = 0.011                                                              # :431
    ch10 = 0.00115                                                             # :432
    ribcu = -(z_u / (BLK_ZABL * 0.004 * (BLK_BETA * BLK_BETA * BLK_BETA)))     # :433
    izo10 = G_CROCO * wstar / (charn * wstar * wstar * wstar
                               + 0.11 * G_CROCO * visair)                      # :437-438
    izot10 = 0.1 * np.exp(VONKAR * VONKAR / (ch10 * np.log(10.0 * izo10)))   # :439-440
    cc = np.log(z_u * izo10) * np.log(z_u * izo10) / np.log(z_t * izot10)     # :442-443
    ri = G_CROCO * z_u * (delt + CPVIR * tair_k * delq) / (tair_k * delw * delw)  # :448-449
    zolu = np.where(ri < 0.0,
                    cc * ri / (1.0 + ri / ribcu),                              # :451 unstable
                    cc * ri / (1.0 + 3.0 * ri / cc))                           # :453 stable
    psi_u = _psiu_coare(zolu)                                                  # :455
    logus10 = np.log(z_u * izo10)                                              # :456
    wstar = delw * VONKAR / (logus10 - psi_u)                                  # :460
    zolt = zolu * z_t_over_z_u                                                 # :462
    psi_t = _psit_coare(zolt)                                                  # :464
    logts10 = np.log(z_t * izot10)                                             # :466
    cff = VONKAR / (logts10 - psi_t)                                           # :470
    tstar = delt * cff                                                         # :471
    qstar = delq * cff                                                         # :472
    charn = _charnock(delw)                       # once, from the initial ΔU (:498-503)

    for _ in range(ITER_FL):                                                   # :511
        izow = G_CROCO * wstar / (charn * wstar * wstar * wstar
                                  + 0.11 * G_CROCO * visair)                   # :516-517
        rr = wstar / (izow * visair)                                           # :523
        izot = np.maximum(8695.65, 18181.8 * (rr ** 0.6))                      # :524
        zolu = (VONKAR * G_CROCO * z_u
                * (tstar * (1.0 + CPVIR * q) + CPVIR * tair_k * qstar)
                / (tair_k * wstar * wstar * (1.0 + CPVIR * q) + EPS))          # :527-529
        psi_u = _psiu_coare(zolu)                                              # :531
        logus10 = np.log(z_u * izow)                                           # :532
        wstar = delw * VONKAR / (logus10 - psi_u)                              # :536
        zolt = zolu * z_t_over_z_u                                             # :539
        psi_t = _psit_coare(zolt)                                              # :540
        logts10 = np.log(z_t * izot)                                           # :544
        cff = VONKAR / (logts10 - psi_t)                                       # :545
        tstar = delt * cff                                                     # :546
        qstar = delq * cff                                                     # :547
        if gustiness:
            bf = -(G_CROCO / tair_k * wstar * (tstar + CPVIR * tair_k * qstar))   # :550
            pos = bf > 0.0
            gust = np.where(pos, BLK_BETA * np.where(pos, bf * BLK_ZABL, 0.0) ** R3,
                            0.2)                                               # :551-555
            delw = np.sqrt(wspd0 * wspd0 + gust * gust)                        # :556
        else:
            delw = wspd0                                                       # :572
    return {"wstar": wstar, "tstar": tstar, "qstar": qstar, "delw": delw, "cff": cff,
            "qsea": qsea, "charn": charn, "patm": patm, "rho_air": rho_air,
            "t_air_k": tair_k, "sst_k": tsea_k, "q": q}


def bulk_coare_croco(
    u10: np.ndarray,
    v10: np.ndarray,
    t_air_c: np.ndarray,
    q_air: np.ndarray,
    sst_c: np.ndarray,
    lw_down: np.ndarray,
    *,
    p_sfc: float = PSURF,
    z_u: float = 10.0,
    z_t: float = 2.0,
    gustiness: bool = True,
    rho0: float = RHO0_CROCO,
) -> CoareFluxes:
    """
    COARE 3.0 air–sea fluxes exactly as CROCO's ``bulk_flux.F`` (git ab63b58, defaults).

    Fluxes (bulk_flux.F:605-659), returned positive into the ocean:

        Q_S = ρ_a c_pa u* θ*                                          :606
        L_v = (2.5008 − 0.0023719 SST) 10⁶                            :608
        Q_L = ρ_a L_v u* q* − ρ_a L_v w_Webb q,                         :610, 621-624
              w_Webb = −1.61 u* q* − (1 + 1.61 q) u* θ*/T_a
        Q_LW = LW↓ − 0.985 σ (SST + 273.16)⁴                           :617-618
        E = −C_p hflat/L_v,  hflat = Q_L/(ρ₀ C_p)  ⇒  E = −Q_L/(ρ₀ L_v)     :630, 636
        τ = ρ_a C_d ΔU (u, v),  C_d = (u*/ΔU)²                         :657-659, 706-712

    Parameters
    ----------
    u10, v10 : np.ndarray
        Wind at z_u [m s⁻¹]; absolute (CROCO), or relative to the current if the caller
        subtracts it.  The speed is √(u² + v²) at the point (get_bulk.F:888).
    t_air_c : np.ndarray
        Air temperature at z_t [°C].
    q_air : np.ndarray
        Specific humidity at z_t [kg kg⁻¹] (CROCO's spec_hum is bypassed).
    sst_c : np.ndarray
        Sea surface temperature [°C] (CROCO: top-layer T, no skin).
    lw_down : np.ndarray
        Downwelling longwave radiation [W m⁻²] (``BULK_LW``).
    p_sfc : float
        Surface pressure [Pa]; CROCO's constant psurf = 1e5 (no READ_PATM).
    z_u, z_t : float
        Heights of the wind and of T, q [m] (blk_ZW = 10, blk_ZT = 2, bulk_flux.F:95-96).
    gustiness : bool
        ``BULK_GUSTINESS`` (default on, as CROCO).
    rho0 : float
        Seawater reference density [kg m⁻³] (croco.in:61); enters only E, through the
        kinematic latent flux.

    Returns
    -------
    CoareFluxes
    """
    u10 = np.asarray(u10, dtype=float)
    v10 = np.asarray(v10, dtype=float)
    wspd = np.sqrt(u10 * u10 + v10 * v10)                                      # get_bulk.F:888
    d = _coare_core(wspd, t_air_c, q_air, sst_c, p_sfc=p_sfc, z_u=z_u, z_t=z_t,
                    gustiness=gustiness)
    wstar, tstar, qstar, delw = d["wstar"], d["tstar"], d["qstar"], d["delw"]
    rho_air, tair_k, q = d["rho_air"], d["t_air_k"], d["q"]
    tsea_c = np.asarray(sst_c, dtype=float)
    tsea_k = d["sst_k"]

    hfsen = -(BLK_CPA * rho_air * wstar * tstar)                               # :606, upward
    hlv = (2.5008 - 0.0023719 * tsea_c) * 1.0e6                                # :608
    hflat = -(hlv * rho_air * wstar * qstar)                                   # :610, upward
    q_lw = (np.asarray(lw_down, dtype=float)
            - EMISS_LW * SIGMA_SB_CROCO * tsea_k * tsea_k * tsea_k * tsea_k)   # :617-618
    upvel = -(1.61 * wstar * qstar) - (1.0 + 1.61 * q) * wstar * tstar / tair_k  # :621-622
    hflat = hflat + rho_air * hlv * upvel * q                                  # :624
    hflat_kin = -hflat * (1.0 / rho0) * (1.0 / CP_CROCO)                       # :630, K m s⁻¹
    evap = -CP_CROCO * hflat_kin / hlv                                         # :636
    aer = rho_air * delw                                                       # :657
    cd = (wstar / delw) ** 2                                                   # :658
    return CoareFluxes(
        tau_x=cd * aer * u10,                                                  # :706-708
        tau_y=cd * aer * v10,                                                  # :728-730
        q_sens=-hfsen,
        q_lat=-hflat,
        q_lw=q_lw,
        evap=evap,
        cd=cd,
        rho_air=rho_air,
        ustar=wstar,
        ch=wstar * d["cff"] / delw,
    )
