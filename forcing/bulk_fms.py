# SPDX-License-Identifier: LGPL-3.0-or-later
#
# This file is a line-by-line Python port of parts of the GFDL Flexible Modeling
# System (FMS 2025.02.01, sat_vapor_pres) and FMScoupler 2025.02 (git 5b700a4,
# shared/surface_flux.F90), Copyright (C) NOAA-GFDL, distributed under the GNU Lesser
# General Public License v3.0 or later.  As a derivative work, this file is distributed
# under the same license (see LICENSES/LGPL-3.0.txt and LICENSES/GPL-3.0.txt); the rest
# of pyOCEAN is MIT.  Changes from the original: translated from Fortran 90 to
# NumPy, vectorised over points, Fortran evaluation order kept so the result equals
# the compiled routine bit for bit (tests/test_bulk_fms.py, validation/bulk_points/).
"""
NCAR bulk formulae in the form of the FMS coupler used by MOM6 / OM4.

``bulk_ncar_fms`` and the FMS saturation vapour pressure table (``esat_fms``).
Re-exported by ``forcing.bulk``, so ``from forcing.bulk import bulk_ncar_fms`` keeps
working.  Kept in its own file because it is a port of LGPL-licensed code (see the
header and THIRD_PARTY_NOTICES.md).
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from forcing.bulk import TurbulentFluxes

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
    from forcing.bulk import RHO_FW, TurbulentFluxes  # deferred: forcing.bulk imports this module

    return TurbulentFluxes(
        tau_x=-(rho_drag * u_dif), tau_y=-(rho_drag * v_dif),
        q_sens=-flux_t, q_lat=-FMS_HLV * flux_q, evap=flux_q / RHO_FW,
        cd=cd, ch=ch, ce=ce, rho_air=rho, ustar=ustar, u10n=u10n, zeta=zeta_u,
    )
