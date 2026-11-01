"""
Roquet et al. (2015) polynomial equation of state for in situ density, port of MOM6
``ROQUET_RHO`` (= ``NEMO``) (TEOS-10 in the 3-D core, task 1).

Physics
-------
TEOS-10 defines density from the Gibbs function of sea water (IOC et al. 2010).  Roquet et
al. (2015) fit in situ density, as a function of Conservative Temperature Θ [°C], Absolute
Salinity S_A [g kg⁻¹] and sea pressure p, with a polynomial cheap enough for an ocean
model.  It is split into a pressure-only reference profile and a (Θ, S_A, p) part::

    ρ(Θ, S_A, p) = ρ₀₀(p) + ρ_TS(ζ_S, Θ, p)                               [kg m⁻³]
    ρ₀₀(p)       = Σ_{c=1..6} R0(c−1) p^c                     (6 coefficients)
    ρ_TS         = Σ EOSabc ζ_S^a Θ^b p^c                     (52 coefficients)
    ζ_S          = √(|S_A + δS| / S_u),   δS = 32 g kg⁻¹,  S_u = 40·35.16504/35 g kg⁻¹

The square root of salinity (Fofonoff–Millard) makes the fit regular at S_A → 0; the
offset δS keeps the root and ∂ρ/∂S_A = (∂ρ/∂ζ_S)/(2 S_u ζ_S) finite in fresh water
(review focus 1: rivers).  MOM6 folds the scale factors 1/T_u = 0.025 °C⁻¹ and
1e-8 kbar/Pa into the coefficients, so Θ is used in °C and p in Pa.  Fit range
(MOM_EOS_Roquet_rho.F90:611-616): Θ −6…40 °C, S_A 0…42 g kg⁻¹, p 0…1e8 Pa.  This is a
different fit from the 75-term specific-volume polynomial of GSW (``gsw.rho``), so it
agrees with ``gsw`` only to the fit error (measured in ``tests/test_eos_nonlinear.py``).

References: Roquet, F., Madec, G., McDougall, T. J. and Barker, P. M., 2015: Accurate
polynomial expressions for the density and specific volume of seawater using the TEOS-10
standard.  Ocean Modelling, 90, 29–43, https://doi.org/10.1016/j.ocemod.2015.04.002
(the polyTEOS10-bsq form, their Section 3.1 and Appendix A); IOC, SCOR and IAPSO, 2010: The international
thermodynamic equation of seawater – 2010 (TEOS-10).

Reference implementation, MOM6 git 0e9a561 (``~/pyOCEAN_refs/MOM6/src/equation_of_state/``):
coefficients ``MOM_EOS_Roquet_rho.F90:14-156``, density ``:227-246``, density anomaly
``:287-308``, ∂ρ/∂Θ and ∂ρ/∂S_A ``:384-411``; reference read in
``docs/design/teos10_refs/mom6_eos.md`` §2.  The Horner nesting is MOM6's, term for term,
and the coefficients are built from the same factors in the same order as the Fortran
``parameter`` expressions (integer powers rounded once, as the compiler folds them), so the
result equals compiled MOM6 to round-off (``tests/test_eos_nonlinear.py``).

Anomaly form
------------
ρ_ref is subtracted from the salinity-only surface term ρ₀S₀(ζ_S) (≈ 1000–1030 kg m⁻³)
**before** the Θ and p terms are added (``:305``), so the anomaly keeps its precision.

API (T = Θ [°C], S = S_A [g kg⁻¹], p [Pa]; T, S arrays of equal shape, p of that shape or scalar)
------------------------------------------------------------------------------------------------
``density(T, S, p)``, ``density_anomaly(T, S, p, rho_ref)``, ``density_derivs(T, S, p)``
(Numba), the same with suffix ``_numpy`` (NumPy oracles, bit for bit equal), and the scalar
``@njit`` kernels ``density_elem``, ``rho_anom_elem``, ``drho_elem`` for other kernels.
"""

from __future__ import annotations

import math
from fractions import Fraction

import numpy as np
from numba import njit


def _pw(x: float, n: int) -> float:
    """x**n rounded once to double, as gfortran folds an integer power in a ``parameter``
    expression (MPFR, correctly rounded); repeated multiplication could differ in the last bit."""
    return float(Fraction(x) ** n)


# Coefficients, MOM_EOS_Roquet_rho.F90:14-156: same names, same factors, same order
# (comments as in MOM6).
Pa2kb = 1.e-8                                     # Conversion factor between Pa and kbar [kbar Pa-1]
rdeltaS = 32.                                     # An offset to salinity before taking its square root [g kg-1]
r1_S0 = 0.875/35.16504                            # The inverse of a plausible range of oceanic salinities [kg g-1]
I_Ts = 0.025                                      # The inverse of a plausible range of oceanic temperatures [degC-1]
R00 = 4.6494977072e+01*Pa2kb                      # rho00p P coef.    [kg m-3 Pa-1]
R01 = -5.2099962525*_pw(Pa2kb, 2)                 # rho00p P**2 coef. [kg m-3 Pa-2]
R02 = 2.2601900708e-01*_pw(Pa2kb, 3)              # rho00p P**3 coef. [kg m-3 Pa-3]
R03 = 6.4326772569e-02*_pw(Pa2kb, 4)              # rho00p P**4 coef. [kg m-3 Pa-4]
R04 = 1.5616995503e-02*_pw(Pa2kb, 5)              # rho00p P**5 coef. [kg m-3 Pa-5]
R05 = -1.7243708991e-03*_pw(Pa2kb, 6)             # rho00p P**6 coef. [kg m-3 Pa-6]
EOS000 = 8.0189615746e+02                         # A constant density contribution [kg m-3]
EOS100 = 8.6672408165e+02                         # EoS zs coef.                [kg m-3]
EOS200 = -1.7864682637e+03                        # EoS zs**2 coef.             [kg m-3]
EOS300 = 2.0375295546e+03                         # EoS zs**3 coef.             [kg m-3]
EOS400 = -1.2849161071e+03                        # EoS zs**4 coef.             [kg m-3]
EOS500 = 4.3227585684e+02                         # EoS zs**5 coef.             [kg m-3]
EOS600 = -6.0579916612e+01                        # EoS zs**6 coef.             [kg m-3]
EOS010 = 2.6010145068e+01*I_Ts                    # EoS T coef.          [kg m-3 degC-1]
EOS110 = -6.5281885265e+01*I_Ts                   # EoS zs * T coef.     [kg m-3 degC-1]
EOS210 = 8.1770425108e+01*I_Ts                    # EoS zs**2 * T coef.  [kg m-3 degC-1]
EOS310 = -5.6888046321e+01*I_Ts                   # EoS zs**3 * T coef.  [kg m-3 degC-1]
EOS410 = 1.7681814114e+01*I_Ts                    # EoS zs**2 * T coef.  [kg m-3 degC-1]
EOS510 = -1.9193502195*I_Ts                       # EoS zs**5 * T coef.  [kg m-3 degC-1]
EOS020 = -3.7074170417e+01*_pw(I_Ts, 2)           # EoS T**2 coef.       [kg m-3 degC-2]
EOS120 = 6.1548258127e+01*_pw(I_Ts, 2)            # EoS zs * T**2 coef.  [kg m-3 degC-2]
EOS220 = -6.0362551501e+01*_pw(I_Ts, 2)           # EoS zs**2 * T**2 coef. [kg m-3 degC-2]
EOS320 = 2.9130021253e+01*_pw(I_Ts, 2)            # EoS zs**3 * T**2 coef. [kg m-3 degC-2]
EOS420 = -5.4723692739*_pw(I_Ts, 2)               # EoS zs**4 * T**2 coef. [kg m-3 degC-2]
EOS030 = 2.1661789529e+01*_pw(I_Ts, 3)            # EoS T**3 coef.       [kg m-3 degC-3]
EOS130 = -3.3449108469e+01*_pw(I_Ts, 3)           # EoS zs * T**3 coef.  [kg m-3 degC-3]
EOS230 = 1.9717078466e+01*_pw(I_Ts, 3)            # EoS zs**2 * T**3 coef. [kg m-3 degC-3]
EOS330 = -3.1742946532*_pw(I_Ts, 3)               # EoS zs**3 * T**3 coef. [kg m-3 degC-3]
EOS040 = -8.3627885467*_pw(I_Ts, 4)               # EoS T**4 coef.       [kg m-3 degC-4]
EOS140 = 1.1311538584e+01*_pw(I_Ts, 4)            # EoS zs * T**4 coef.  [kg m-3 degC-4]
EOS240 = -5.3563304045*_pw(I_Ts, 4)               # EoS zs**2 * T**4 coef. [kg m-3 degC-4]
EOS050 = 5.4048723791e-01*_pw(I_Ts, 5)            # EoS T**5 coef.       [kg m-3 degC-5]
EOS150 = 4.8169980163e-01*_pw(I_Ts, 5)            # EoS zs * T**5 coef.  [kg m-3 degC-5]
EOS060 = -1.9083568888e-01*_pw(I_Ts, 6)           # EoS T**6             [kg m-3 degC-6]
EOS001 = 1.9681925209e+01*Pa2kb                   # EoS P coef.            [kg m-3 Pa-1]
EOS101 = -4.2549998214e+01*Pa2kb                  # EoS zs * P coef.       [kg m-3 Pa-1]
EOS201 = 5.0774768218e+01*Pa2kb                   # EoS zs**2 * P coef.    [kg m-3 Pa-1]
EOS301 = -3.0938076334e+01*Pa2kb                  # EoS zs**3 * P coef.    [kg m-3 Pa-1]
EOS401 = 6.6051753097*Pa2kb                       # EoS zs**4 * P coef.    [kg m-3 Pa-1]
EOS011 = -1.3336301113e+01*(I_Ts*Pa2kb)           # EoS T * P coef. [kg m-3 degC-1 Pa-1]
EOS111 = -4.4870114575*(I_Ts*Pa2kb)               # EoS zs * T * P coef. [kg m-3 degC-1 Pa-1]
EOS211 = 5.0042598061*(I_Ts*Pa2kb)                # EoS zs**2 * T * P coef. [kg m-3 degC-1 Pa-1]
EOS311 = -6.5399043664e-01*(I_Ts*Pa2kb)           # EoS zs**3 * T * P coef. [kg m-3 degC-1 Pa-1]
EOS021 = 6.7080479603*(_pw(I_Ts, 2)*Pa2kb)        # EoS T**2 * P coef. [kg m-3 degC-2 Pa-1]
EOS121 = 3.5063081279*(_pw(I_Ts, 2)*Pa2kb)        # EoS zs * T**2 * P coef. [kg m-3 degC-2 Pa-1]
EOS221 = -1.8795372996*(_pw(I_Ts, 2)*Pa2kb)       # EoS zs**2 * T**2 * P coef. [kg m-3 degC-2 Pa-1]
EOS031 = -2.4649669534*(_pw(I_Ts, 3)*Pa2kb)       # EoS T**3 * P coef. [kg m-3 degC-3 Pa-1]
EOS131 = -5.5077101279e-01*(_pw(I_Ts, 3)*Pa2kb)   # EoS zs * T**3 * P coef. [kg m-3 degC-3 Pa-1]
EOS041 = 5.5927935970e-01*(_pw(I_Ts, 4)*Pa2kb)    # EoS T**4 * P coef. [kg m-3 degC-4 Pa-1]
EOS002 = 2.0660924175*_pw(Pa2kb, 2)               # EoS P**2 coef.         [kg m-3 Pa-2]
EOS102 = -4.9527603989*_pw(Pa2kb, 2)              # EoS zs * P**2 coef.    [kg m-3 Pa-2]
EOS202 = 2.5019633244*_pw(Pa2kb, 2)               # EoS zs**2 * P**2 coef. [kg m-3 Pa-2]
EOS012 = 2.0564311499*(I_Ts*_pw(Pa2kb, 2))        # EoS T * P**2 coef. [kg m-3 degC-1 Pa-2]
EOS112 = -2.1311365518e-01*(I_Ts*_pw(Pa2kb, 2))   # EoS zs * T * P**2 coef. [kg m-3 degC-1 Pa-2]
EOS022 = -1.2419983026*(_pw(I_Ts, 2)*_pw(Pa2kb, 2))  # EoS T**2 * P**2 coef. [kg m-3 degC-2 Pa-2]
EOS003 = -2.3342758797e-02*_pw(Pa2kb, 3)          # EoS P**3 coef.         [kg m-3 Pa-3]
EOS103 = -1.8507636718e-02*_pw(Pa2kb, 3)          # EoS zs * P**3 coef.    [kg m-3 Pa-3]
EOS013 = 3.7969820455e-01*(I_Ts*_pw(Pa2kb, 3))    # EoS T * P**3 coef. [kg m-3 degC-1 Pa-3]
ALP000 = EOS010                                   # Constant in the drho_dT fit                [kg m-3 degC-1]
ALP100 = EOS110                                   # drho_dT fit zs coef.                       [kg m-3 degC-1]
ALP200 = EOS210                                   # drho_dT fit zs**2 coef.                    [kg m-3 degC-1]
ALP300 = EOS310                                   # drho_dT fit zs**3 coef.                    [kg m-3 degC-1]
ALP400 = EOS410                                   # drho_dT fit zs**4 coef.                    [kg m-3 degC-1]
ALP500 = EOS510                                   # drho_dT fit zs**5 coef.                    [kg m-3 degC-1]
ALP010 = 2.*EOS020                                # drho_dT fit T coef.                        [kg m-3 degC-2]
ALP110 = 2.*EOS120                                # drho_dT fit zs * T coef.                   [kg m-3 degC-2]
ALP210 = 2.*EOS220                                # drho_dT fit zs**2 * T coef.                [kg m-3 degC-2]
ALP310 = 2.*EOS320                                # drho_dT fit zs**3 * T coef.                [kg m-3 degC-2]
ALP410 = 2.*EOS420                                # drho_dT fit zs**4 * T coef.                [kg m-3 degC-2]
ALP020 = 3.*EOS030                                # drho_dT fit T**2 coef.                     [kg m-3 degC-3]
ALP120 = 3.*EOS130                                # drho_dT fit zs * T**2 coef.                [kg m-3 degC-3]
ALP220 = 3.*EOS230                                # drho_dT fit zs**2 * T**2 coef.             [kg m-3 degC-3]
ALP320 = 3.*EOS330                                # drho_dT fit zs**3 * T**2 coef.             [kg m-3 degC-3]
ALP030 = 4.*EOS040                                # drho_dT fit T**3 coef.                     [kg m-3 degC-4]
ALP130 = 4.*EOS140                                # drho_dT fit zs * T**3 coef.                [kg m-3 degC-4]
ALP230 = 4.*EOS240                                # drho_dT fit zs**2 * T**3 coef.             [kg m-3 degC-4]
ALP040 = 5.*EOS050                                # drho_dT fit T**4 coef.                     [kg m-3 degC-5]
ALP140 = 5.*EOS150                                # drho_dT fit zs* * T**4 coef.               [kg m-3 degC-5]
ALP050 = 6.*EOS060                                # drho_dT fit T**5 coef.                     [kg m-3 degC-6]
ALP001 = EOS011                                   # drho_dT fit P coef.                   [kg m-3 degC-1 Pa-1]
ALP101 = EOS111                                   # drho_dT fit zs * P coef.              [kg m-3 degC-1 Pa-1]
ALP201 = EOS211                                   # drho_dT fit zs**2 * P coef.           [kg m-3 degC-1 Pa-1]
ALP301 = EOS311                                   # drho_dT fit zs**3 * P coef.           [kg m-3 degC-1 Pa-1]
ALP011 = 2.*EOS021                                # drho_dT fit T * P coef.               [kg m-3 degC-2 Pa-1]
ALP111 = 2.*EOS121                                # drho_dT fit zs * T * P coef.          [kg m-3 degC-2 Pa-1]
ALP211 = 2.*EOS221                                # drho_dT fit zs**2 * T * P coef.       [kg m-3 degC-2 Pa-1]
ALP021 = 3.*EOS031                                # drho_dT fit T**2 * P coef.            [kg m-3 degC-3 Pa-1]
ALP121 = 3.*EOS131                                # drho_dT fit zs * T**2 * P coef.       [kg m-3 degC-3 Pa-1]
ALP031 = 4.*EOS041                                # drho_dT fit T**3 * P coef.            [kg m-3 degC-4 Pa-1]
ALP002 = EOS012                                   # drho_dT fit P**2 coef.                [kg m-3 degC-1 Pa-2]
ALP102 = EOS112                                   # drho_dT fit zs * P**2 coef.           [kg m-3 degC-1 Pa-2]
ALP012 = 2.*EOS022                                # drho_dT fit T * P**2 coef.            [kg m-3 degC-2 Pa-2]
ALP003 = EOS013                                   # drho_dT fit P**3 coef.                [kg m-3 degC-1 Pa-3]
BET000 = 0.5*EOS100*r1_S0                         # Constant in the drho_dS fit           [kg m-3 ppt-1]
BET100 = EOS200*r1_S0                             # drho_dS fit zs coef.                  [kg m-3 ppt-1]
BET200 = 1.5*EOS300*r1_S0                         # drho_dS fit zs**2 coef.               [kg m-3 ppt-1]
BET300 = 2.0*EOS400*r1_S0                         # drho_dS fit zs**3 coef.               [kg m-3 ppt-1]
BET400 = 2.5*EOS500*r1_S0                         # drho_dS fit zs**4 coef.               [kg m-3 ppt-1]
BET500 = 3.0*EOS600*r1_S0                         # drho_dS fit zs**5 coef.               [kg m-3 ppt-1]
BET010 = 0.5*EOS110*r1_S0                         # drho_dS fit T coef.            [kg m-3 ppt-1 degC-1]
BET110 = EOS210*r1_S0                             # drho_dS fit zs * T coef.       [kg m-3 ppt-1 degC-1]
BET210 = 1.5*EOS310*r1_S0                         # drho_dS fit zs**2 * T coef.    [kg m-3 ppt-1 degC-1]
BET310 = 2.0*EOS410*r1_S0                         # drho_dS fit zs**3 * T coef.    [kg m-3 ppt-1 degC-1]
BET410 = 2.5*EOS510*r1_S0                         # drho_dS fit zs**4 * T coef.    [kg m-3 ppt-1 degC-1]
BET020 = 0.5*EOS120*r1_S0                         # drho_dS fit T**2 coef.         [kg m-3 ppt-1 degC-2]
BET120 = EOS220*r1_S0                             # drho_dS fit zs * T**2 coef.    [kg m-3 ppt-1 degC-2]
BET220 = 1.5*EOS320*r1_S0                         # drho_dS fit zs**2 * T**2 coef. [kg m-3 ppt-1 degC-2]
BET320 = 2.0*EOS420*r1_S0                         # drho_dS fit zs**3 * T**2 coef. [kg m-3 ppt-1 degC-2]
BET030 = 0.5*EOS130*r1_S0                         # drho_dS fit T**3 coef.         [kg m-3 ppt-1 degC-3]
BET130 = EOS230*r1_S0                             # drho_dS fit zs * T**3 coef.    [kg m-3 ppt-1 degC-3]
BET230 = 1.5*EOS330*r1_S0                         # drho_dS fit zs**2 * T**3 coef. [kg m-3 ppt-1 degC-3]
BET040 = 0.5*EOS140*r1_S0                         # drho_dS fit T**4 coef.         [kg m-3 ppt-1 degC-4]
BET140 = EOS240*r1_S0                             # drho_dS fit zs * T**4 coef.    [kg m-3 ppt-1 degC-4]
BET050 = 0.5*EOS150*r1_S0                         # drho_dS fit T**5 coef.         [kg m-3 ppt-1 degC-5]
BET001 = 0.5*EOS101*r1_S0                         # drho_dS fit P coef.              [kg m-3 ppt-1 Pa-1]
BET101 = EOS201*r1_S0                             # drho_dS fit zs * P coef.         [kg m-3 ppt-1 Pa-1]
BET201 = 1.5*EOS301*r1_S0                         # drho_dS fit zs**2 * P coef.      [kg m-3 ppt-1 Pa-1]
BET301 = 2.0*EOS401*r1_S0                         # drho_dS fit zs**3 * P coef.      [kg m-3 ppt-1 Pa-1]
BET011 = 0.5*EOS111*r1_S0                         # drho_dS fit T * P coef.   [kg m-3 ppt-1 degC-1 Pa-1]
BET111 = EOS211*r1_S0                             # drho_dS fit zs * T * P coef. [kg m-3 ppt-1 degC-1 Pa-1]
BET211 = 1.5*EOS311*r1_S0                         # drho_dS fit zs**2 * T * P coef. [kg m-3 ppt-1 degC-1 Pa-1]
BET021 = 0.5*EOS121*r1_S0                         # drho_dS fit T**2 * P coef. [kg m-3 ppt-1 degC-2 Pa-1]
BET121 = EOS221*r1_S0                             # drho_dS fit zs * T**2 * P coef. [kg m-3 ppt-1 degC-2 Pa-1]
BET031 = 0.5*EOS131*r1_S0                         # drho_dS fit T**3 * P coef. [kg m-3 ppt-1 degC-3 Pa-1]
BET002 = 0.5*EOS102*r1_S0                         # drho_dS fit P**2 coef.           [kg m-3 ppt-1 Pa-2]
BET102 = EOS202*r1_S0                             # drho_dS fit zs * P**2 coef.      [kg m-3 ppt-1 Pa-2]
BET012 = 0.5*EOS112*r1_S0                         # drho_dS fit T * P**2 coef. [kg m-3 ppt-1 degC-1 Pa-2]
BET003 = 0.5*EOS103*r1_S0                         # drho_dS fit P**3 coef.           [kg m-3 ppt-1 Pa-3]


# --------------------------------------------------------------------------- scalar kernels
@njit(cache=True, inline="always")
def _zs(S: float) -> float:
    """ζ_S = √(|S_A + δS| / S_u) [nondim] (MOM_EOS_Roquet_rho.F90:219)."""
    return math.sqrt(abs(S + rdeltaS) * r1_S0)


@njit(cache=True, inline="always")
def _rho_parts(zt: float, zs: float, zp: float) -> tuple[float, float, float, float, float, float]:
    """(ρ_TS0, ρ_TS1, ρ_TS2, ρ_TS3, ρ₀S₀, ρ₀₀p) of MOM_EOS_Roquet_rho.F90:227-243
    [kg m⁻³, kg m⁻³ Pa⁻¹, kg m⁻³ Pa⁻², kg m⁻³ Pa⁻³, kg m⁻³, kg m⁻³]."""
    rhoTS3 = EOS003 + (zs * EOS103 + zt * EOS013)
    rhoTS2 = EOS002 + (zs * (EOS102 + zs * EOS202)
                       + zt * (EOS012 + (zs * EOS112 + zt * EOS022)))
    rhoTS1 = EOS001 + (zs * (EOS101 + zs * (EOS201 + zs * (EOS301 + zs * EOS401)))
                       + zt * (EOS011 + (zs * (EOS111 + zs * (EOS211 + zs * EOS311))
                                         + zt * (EOS021 + (zs * (EOS121 + zs * EOS221)
                                                           + zt * (EOS031 + (zs * EOS131 + zt * EOS041)))))))
    rhoTS0 = zt * (EOS010
                   + (zs * (EOS110 + zs * (EOS210 + zs * (EOS310 + zs * (EOS410 + zs * EOS510))))
                      + zt * (EOS020 + (zs * (EOS120 + zs * (EOS220 + zs * (EOS320 + zs * EOS420)))
                                        + zt * (EOS030 + (zs * (EOS130 + zs * (EOS230 + zs * EOS330))
                                                          + zt * (EOS040 + (zs * (EOS140 + zs * EOS240)
                                                                            + zt * (EOS050 + (zs * EOS150
                                                                                              + zt * EOS060))))))))))
    rho0S0 = EOS000 + zs * (EOS100 + zs * (EOS200 + zs * (EOS300 + zs * (EOS400 + zs * (EOS500 + zs * EOS600)))))
    rho00p = zp * (R00 + zp * (R01 + zp * (R02 + zp * (R03 + zp * (R04 + zp * R05)))))
    return rhoTS0, rhoTS1, rhoTS2, rhoTS3, rho0S0, rho00p


@njit(cache=True, inline="always")
def density_elem(T: float, S: float, p: float) -> float:
    """In situ density ρ [kg m⁻³] of Θ [°C], S_A [g kg⁻¹], p [Pa] (MOM_EOS_Roquet_rho.F90:227-246)."""
    rhoTS0, rhoTS1, rhoTS2, rhoTS3, rho0S0, rho00p = _rho_parts(T, _zs(S), p)
    rhoTS = (rhoTS0 + rho0S0) + p * (rhoTS1 + p * (rhoTS2 + p * rhoTS3))
    return rhoTS + rho00p


@njit(cache=True, inline="always")
def rho_anom_elem(T: float, S: float, p: float, rho_ref: float) -> float:
    """ρ − ρ_ref [kg m⁻³], ρ_ref subtracted from ρ₀S₀ before the Θ and p terms
    (MOM_EOS_Roquet_rho.F90:287-308).  T = Θ [°C], S = S_A [g kg⁻¹], p [Pa]."""
    rhoTS0, rhoTS1, rhoTS2, rhoTS3, rho0S0, rho00p = _rho_parts(T, _zs(S), p)
    rho0S0 = rho0S0 - rho_ref
    rhoTS = (rhoTS0 + rho0S0) + p * (rhoTS1 + p * (rhoTS2 + p * rhoTS3))
    return rhoTS + rho00p


@njit(cache=True, inline="always")
def drho_elem(T: float, S: float, p: float) -> tuple[float, float]:
    """(∂ρ/∂Θ [kg m⁻³ °C⁻¹], ∂ρ/∂S_A [kg m⁻³ (g/kg)⁻¹]) (MOM_EOS_Roquet_rho.F90:384-411).

    The ALP coefficients are b·EOSab'c of the Θ^b terms; the BET ones carry the factor
    a/(2 S_u) of dζ_S/dS_A = 1/(2 S_u ζ_S), and the remaining 1/ζ_S is the final division.
    """
    zt = T
    zs = _zs(S)
    zp = p
    dRdzt3 = ALP003
    dRdzt2 = ALP002 + (zs * ALP102 + zt * ALP012)
    dRdzt1 = ALP001 + (zs * (ALP101 + zs * (ALP201 + zs * ALP301))
                       + zt * (ALP011 + (zs * (ALP111 + zs * ALP211)
                                         + zt * (ALP021 + (zs * ALP121 + zt * ALP031)))))
    dRdzt0 = ALP000 + (zs * (ALP100 + zs * (ALP200 + zs * (ALP300 + zs * (ALP400 + zs * ALP500))))
                       + zt * (ALP010 + (zs * (ALP110 + zs * (ALP210 + zs * (ALP310 + zs * ALP410)))
                                         + zt * (ALP020 + (zs * (ALP120 + zs * (ALP220 + zs * ALP320))
                                                           + zt * (ALP030 + (zt * (ALP040 + (zs * ALP140 + zt * ALP050))
                                                                             + zs * (ALP130 + zs * ALP230))))))))
    drho_dT = dRdzt0 + zp * (dRdzt1 + zp * (dRdzt2 + zp * dRdzt3))

    dRdzs3 = BET003
    dRdzs2 = BET002 + (zs * BET102 + zt * BET012)
    dRdzs1 = BET001 + (zs * (BET101 + zs * (BET201 + zs * BET301))
                       + zt * (BET011 + (zs * (BET111 + zs * BET211)
                                         + zt * (BET021 + (zs * BET121 + zt * BET031)))))
    dRdzs0 = BET000 + (zs * (BET100 + zs * (BET200 + zs * (BET300 + zs * (BET400 + zs * BET500))))
                       + zt * (BET010 + (zs * (BET110 + zs * (BET210 + zs * (BET310 + zs * BET410)))
                                         + zt * (BET020 + (zs * (BET120 + zs * (BET220 + zs * BET320))
                                                           + zt * (BET030 + (zt * (BET040 + (zs * BET140 + zt * BET050))
                                                                             + zs * (BET130 + zs * BET230))))))))
    drho_dS = (dRdzs0 + zp * (dRdzs1 + zp * (dRdzs2 + zp * dRdzs3))) / zs
    return drho_dT, drho_dS


# --------------------------------------------------------------------------- array loops
@njit(cache=True)
def _density_loop(T, S, p, out):
    for i in range(out.size):
        out[i] = density_elem(T[i], S[i], p[i])


@njit(cache=True)
def _anomaly_loop(T, S, p, rho_ref, out):
    for i in range(out.size):
        out[i] = rho_anom_elem(T[i], S[i], p[i], rho_ref)


@njit(cache=True)
def _derivs_loop(T, S, p, dT, dS):
    for i in range(dT.size):
        dT[i], dS[i] = drho_elem(T[i], S[i], p[i])


def _flat(T, S, p) -> tuple[tuple[int, ...], np.ndarray, np.ndarray, np.ndarray]:
    """Broadcast T, S, p to one shape and return contiguous float64 1-D views."""
    T, S, p = np.broadcast_arrays(np.asarray(T, np.float64), np.asarray(S, np.float64),
                                  np.asarray(p, np.float64))
    shape = T.shape
    return shape, *(np.ascontiguousarray(a).ravel() for a in (T, S, p))


def density(T, S, p) -> np.ndarray:
    """In situ density ρ [kg m⁻³] of Θ [°C], S_A [g kg⁻¹], p [Pa] (Numba)."""
    shape, Tf, Sf, pf = _flat(T, S, p)
    out = np.empty(Tf.size)
    _density_loop(Tf, Sf, pf, out)
    return out.reshape(shape)


def density_anomaly(T, S, p, rho_ref: float) -> np.ndarray:
    """ρ − ρ_ref [kg m⁻³] in MOM6's anomaly form (Numba); rho_ref [kg m⁻³]."""
    shape, Tf, Sf, pf = _flat(T, S, p)
    out = np.empty(Tf.size)
    _anomaly_loop(Tf, Sf, pf, float(rho_ref), out)
    return out.reshape(shape)


def density_derivs(T, S, p) -> tuple[np.ndarray, np.ndarray]:
    """(∂ρ/∂Θ [kg m⁻³ °C⁻¹], ∂ρ/∂S_A [kg m⁻³ (g/kg)⁻¹]) (Numba)."""
    shape, Tf, Sf, pf = _flat(T, S, p)
    dT = np.empty(Tf.size)
    dS = np.empty(Tf.size)
    _derivs_loop(Tf, Sf, pf, dT, dS)
    return dT.reshape(shape), dS.reshape(shape)


# --------------------------------------------------------------------------- NumPy oracles
def _arrays(T, S, p):
    return np.broadcast_arrays(np.asarray(T, np.float64), np.asarray(S, np.float64),
                               np.asarray(p, np.float64))


def _rho_parts_numpy(zt, zs, zp):
    rhoTS3 = EOS003 + (zs * EOS103 + zt * EOS013)
    rhoTS2 = EOS002 + (zs * (EOS102 + zs * EOS202)
                       + zt * (EOS012 + (zs * EOS112 + zt * EOS022)))
    rhoTS1 = EOS001 + (zs * (EOS101 + zs * (EOS201 + zs * (EOS301 + zs * EOS401)))
                       + zt * (EOS011 + (zs * (EOS111 + zs * (EOS211 + zs * EOS311))
                                         + zt * (EOS021 + (zs * (EOS121 + zs * EOS221)
                                                           + zt * (EOS031 + (zs * EOS131 + zt * EOS041)))))))
    rhoTS0 = zt * (EOS010
                   + (zs * (EOS110 + zs * (EOS210 + zs * (EOS310 + zs * (EOS410 + zs * EOS510))))
                      + zt * (EOS020 + (zs * (EOS120 + zs * (EOS220 + zs * (EOS320 + zs * EOS420)))
                                        + zt * (EOS030 + (zs * (EOS130 + zs * (EOS230 + zs * EOS330))
                                                          + zt * (EOS040 + (zs * (EOS140 + zs * EOS240)
                                                                            + zt * (EOS050 + (zs * EOS150
                                                                                              + zt * EOS060))))))))))
    rho0S0 = EOS000 + zs * (EOS100 + zs * (EOS200 + zs * (EOS300 + zs * (EOS400 + zs * (EOS500 + zs * EOS600)))))
    rho00p = zp * (R00 + zp * (R01 + zp * (R02 + zp * (R03 + zp * (R04 + zp * R05)))))
    return rhoTS0, rhoTS1, rhoTS2, rhoTS3, rho0S0, rho00p


def density_numpy(T, S, p) -> np.ndarray:
    """NumPy oracle of :func:`density` (same operations, same order)."""
    T, S, p = _arrays(T, S, p)
    rhoTS0, rhoTS1, rhoTS2, rhoTS3, rho0S0, rho00p = _rho_parts_numpy(T, np.sqrt(np.abs(S + rdeltaS) * r1_S0), p)
    rhoTS = (rhoTS0 + rho0S0) + p * (rhoTS1 + p * (rhoTS2 + p * rhoTS3))
    return rhoTS + rho00p


def density_anomaly_numpy(T, S, p, rho_ref: float) -> np.ndarray:
    """NumPy oracle of :func:`density_anomaly`."""
    T, S, p = _arrays(T, S, p)
    rhoTS0, rhoTS1, rhoTS2, rhoTS3, rho0S0, rho00p = _rho_parts_numpy(T, np.sqrt(np.abs(S + rdeltaS) * r1_S0), p)
    rho0S0 = rho0S0 - float(rho_ref)
    rhoTS = (rhoTS0 + rho0S0) + p * (rhoTS1 + p * (rhoTS2 + p * rhoTS3))
    return rhoTS + rho00p


def density_derivs_numpy(T, S, p) -> tuple[np.ndarray, np.ndarray]:
    """NumPy oracle of :func:`density_derivs`."""
    T, S, p = _arrays(T, S, p)
    zt = T
    zs = np.sqrt(np.abs(S + rdeltaS) * r1_S0)
    zp = p
    dRdzt3 = ALP003
    dRdzt2 = ALP002 + (zs * ALP102 + zt * ALP012)
    dRdzt1 = ALP001 + (zs * (ALP101 + zs * (ALP201 + zs * ALP301))
                       + zt * (ALP011 + (zs * (ALP111 + zs * ALP211)
                                         + zt * (ALP021 + (zs * ALP121 + zt * ALP031)))))
    dRdzt0 = ALP000 + (zs * (ALP100 + zs * (ALP200 + zs * (ALP300 + zs * (ALP400 + zs * ALP500))))
                       + zt * (ALP010 + (zs * (ALP110 + zs * (ALP210 + zs * (ALP310 + zs * ALP410)))
                                         + zt * (ALP020 + (zs * (ALP120 + zs * (ALP220 + zs * ALP320))
                                                           + zt * (ALP030 + (zt * (ALP040 + (zs * ALP140 + zt * ALP050))
                                                                             + zs * (ALP130 + zs * ALP230))))))))
    drho_dT = dRdzt0 + zp * (dRdzt1 + zp * (dRdzt2 + zp * dRdzt3))
    dRdzs3 = BET003
    dRdzs2 = BET002 + (zs * BET102 + zt * BET012)
    dRdzs1 = BET001 + (zs * (BET101 + zs * (BET201 + zs * BET301))
                       + zt * (BET011 + (zs * (BET111 + zs * BET211)
                                         + zt * (BET021 + (zs * BET121 + zt * BET031)))))
    dRdzs0 = BET000 + (zs * (BET100 + zs * (BET200 + zs * (BET300 + zs * (BET400 + zs * BET500))))
                       + zt * (BET010 + (zs * (BET110 + zs * (BET210 + zs * (BET310 + zs * BET410)))
                                         + zt * (BET020 + (zs * (BET120 + zs * (BET220 + zs * BET320))
                                                           + zt * (BET030 + (zt * (BET040 + (zs * BET140 + zt * BET050))
                                                                             + zs * (BET130 + zs * BET230))))))))
    drho_dS = (dRdzs0 + zp * (dRdzs1 + zp * (dRdzs2 + zp * dRdzs3))) / zs
    return drho_dT, drho_dS
