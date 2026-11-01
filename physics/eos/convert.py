"""
Conversions between the two temperature/salinity conventions of the 3-D core, and freezing point
(TEOS-10 in the 3-D core, task 2).

Two conventions exist (``ts_kind``):

* ``"theta_sp"``: potential temperature θ [°C] and practical salinity S_P [PSU]; what data sets,
  the linear EOS and MOM6's WRIGHT_FULL take.
* ``"ct_sa"``: conservative temperature Θ [°C] and absolute salinity S_A [g kg⁻¹]; what the
  TEOS-10 polynomial EOS (Roquet et al. 2015) takes.

Salinity: S_A = (35.16504/35) S_P, the reference salinity S_R.  No spatially varying anomaly
δS_A is used, as in MOM6 (``MOM_EOS.F90:1948-2022``; ``docs/design/teos10_refs/mom6_eos.md``).
Temperature: Θ = CT_from_pt(S_A, θ), θ = pt_from_CT(S_A, Θ), with ``gsw`` (MOM6 ports the same
polynomial and a Newton inversion, ``MOM_temperature_convert.F90``).

Freezing point: MOM6 ``TFREEZE_FORM`` LINEAR (``MOM_TFreeze.F90:62``,
``T_Fr = (S0_P0 + dT_dS S) + dT_dp p``), with the NWA12 coefficients 0, −0.054 °C PSU⁻¹,
−7.75e-8 °C Pa⁻¹ (``MOM_EOS.F90:1683-1696``), and TEOS10 = ``gsw_ct_freezing_exact`` with
saturation fraction 0 (``MOM_TFreeze.F90:231-250``).

``gsw`` is an optional dependency (as in ``physics/eos/__init__.py``): it is imported inside the functions
that need it (Θ ↔ θ and the TEOS-10 freezing point), so the 3-D core imports and runs a linear or
``wright_full`` model without it.  ``require_gsw`` gives the clear error ``eos="teos10"`` raises at construction.
"""
from __future__ import annotations

import numpy as np

SR_OVER_SP = 35.16504 / 35.0  # S_A / S_P  [-]  (reference salinity, TEOS-10)

# MOM6 TFREEZE_FORM = "LINEAR", NWA12 values (MOM_EOS.F90:1683-1696)
TFREEZE_S0_P0 = 0.0  # [degC]
DTFREEZE_DS = -0.054  # [degC PSU-1]
DTFREEZE_DP = -7.75e-8  # [degC Pa-1]


def sp_to_sa(SP):
    """Absolute salinity [g kg⁻¹] from practical salinity [PSU]: S_A = (35.16504/35) S_P."""
    return SR_OVER_SP * np.asarray(SP)


def sa_to_sp(SA):
    """Practical salinity [PSU] from absolute salinity [g kg⁻¹]: S_P = (35/35.16504) S_A."""
    return (35.0 / 35.16504) * np.asarray(SA)


def require_gsw(purpose: str = "TEOS-10 conversions"):
    """The ``gsw`` module (optional dependency, imported on first use); ImportError naming ``purpose`` if missing."""
    try:
        import gsw  # optional import: not needed with eos="linear" or "wright_full" and θ/S_P data
    except ImportError as err:
        raise ImportError(f"{purpose} need the 'gsw' package (TEOS-10 GSW toolbox), which is not installed; "
                          "it is optional only for eos='linear' and 'wright_full' with θ/S_P data") from err
    return gsw


def theta_to_ct(SA, theta):
    """Conservative temperature Θ [°C] from potential temperature θ [°C] at S_A [g kg⁻¹] (gsw)."""
    return require_gsw("θ → Θ conversions").CT_from_pt(SA, theta)


def ct_to_theta(SA, CT):
    """Potential temperature θ [°C] from conservative temperature Θ [°C] at S_A [g kg⁻¹] (gsw)."""
    return require_gsw("Θ → θ conversions").pt_from_CT(SA, CT)


def to_model_ts(T, S, eos_ts_kind: str, ts_kind: str):
    """
    Convert data (T, S) given in convention ``ts_kind`` to the convention the EOS needs
    (``eos_ts_kind``).  Returns (T, S); the inputs unchanged when the two agree.
    """
    _check_kind(eos_ts_kind), _check_kind(ts_kind)
    if eos_ts_kind == ts_kind:
        return T, S
    if ts_kind == "theta_sp":
        SA = sp_to_sa(S)
        return theta_to_ct(SA, T), SA
    SP = sa_to_sp(S)
    return ct_to_theta(S, T), SP


def from_model_ts(T, S, eos_ts_kind: str):
    """Model (T, S) in the EOS convention back to (θ, S_P), the convention of data and output."""
    _check_kind(eos_ts_kind)
    if eos_ts_kind == "theta_sp":
        return T, S
    return ct_to_theta(S, T), sa_to_sp(S)


def _check_kind(kind: str) -> None:
    if kind not in ("theta_sp", "ct_sa"):
        raise ValueError(f"ts_kind must be 'theta_sp' or 'ct_sa', got {kind!r}")


def pressure_from_z(z, rho0: float, g: float):
    """Boussinesq sea pressure [Pa] at depth z [m, negative down]: p = −g ρ₀ z."""
    return -g * rho0 * np.asarray(z)


def freezing_linear(S, p):
    """
    MOM6 linear freezing point [°C] of salinity S [PSU] at sea pressure p [Pa]:
    T_f = (S0_P0 + dT/dS S) + dT/dp p  (``MOM_TFreeze.F90:62``; NWA12 coefficients).
    """
    return (TFREEZE_S0_P0 + DTFREEZE_DS * np.asarray(S)) + DTFREEZE_DP * np.asarray(p)


def freezing_teos10(SA, p):
    """
    TEOS-10 freezing conservative temperature Θ_f [°C] of S_A [g kg⁻¹] at sea pressure p [Pa],
    saturation fraction 0 (``gsw.CT_freezing``; MOM6 ``MOM_TFreeze.F90:231-250``).
    """
    return require_gsw("the TEOS-10 freezing point").CT_freezing(SA, np.asarray(p) * 1.0e-4, 0.0)
