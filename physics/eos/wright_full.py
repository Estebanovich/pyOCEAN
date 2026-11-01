"""
Wright (1997) full-range equation of state, port of MOM6 ``WRIGHT_FULL``
(TEOS-10 in the 3-D core, task 1).

Physics
-------
Wright (1997) writes the specific volume of sea water in the Tumlirz–Eckart form

    α(θ, S, p) = α₀(θ, S) + λ(θ, S) / (p + p₀(θ, S))                 [m³ kg⁻¹]

with α₀, p₀ and λ low-order polynomials in potential temperature θ [°C] and
practical salinity S [PSU] (Wright 1997, eq. 1 and Table 1, the "full range" fit to
UNESCO (1981): −2 < θ < 40 °C, 0 < S < 40 PSU, 0 < p < 1e8 Pa)::

    α₀ = a0 + (a1 θ + a2 S)                                             [m³ kg⁻¹]
    p₀ = b0 + (b4 S + θ (b1 + (θ (b2 + b3 θ) + b5 S)))                  [Pa]
    λ  = c0 + (c4 S + θ (c1 + (θ (c2 + c3 θ) + c5 S)))                  [m² s⁻²]
    ρ  = (p + p₀) / (λ + α₀ (p + p₀))                                   [kg m⁻³]

λ/α₀² is the inverse compressibility scale and 1/∂ρ/∂p the squared sound speed.
p is the sea pressure in Pa (0 at the surface).  The inputs are θ and S_P: when the
model carries Θ and S_A this EOS is fed them unconverted, as MOM6 does.

Reference: Wright, D. G., 1997: An equation of state for use in ocean models:
Eckart's formula revisited.  J. Atmos. Oceanic Technol., 14, 735–740,
https://doi.org/10.1175/1520-0426(1997)014<0735:AEOSFU>2.0.CO;2 [verificar: not in the
local library; the coefficients below are MOM6's, in Pa and m² s⁻², and Wright tabulates
them in other units].

Reference implementation, MOM6 git 0e9a561 (``~/pyOCEAN_refs/MOM6/src/equation_of_state/``):
coefficients ``MOM_EOS_Wright_full.F90:23-37``, density ``:86-89``, density anomaly
``:110-120``, ∂ρ/∂θ and ∂ρ/∂S ``:194-202``; reference read in
``docs/design/teos10_refs/mom6_eos.md`` §1.  Every expression keeps MOM6's parenthesization,
so the result equals compiled MOM6 to round-off (``tests/test_eos_nonlinear.py``:
measured bit for bit on 2048 points, ``validation/mom6_column/eos_driver.F90``).

Anomaly form
------------
ρ − ρ_ref is not computed as ρ minus ρ_ref (that would lose ~3 digits of a ~1030 kg m⁻³
number).  MOM6 folds ρ_ref into the numerator (``:110-120``)::

    pa_000 = b0 (1 − a0 ρ_ref) − ρ_ref c0
    ρ − ρ_ref = (pa_000 + (p_TSp − ρ_ref (p_TSp α₀ + (b0 α_TS + λ_TS))))
                / ((c0 + λ_TS) + α₀ (b0 + p_TSp))

with α_TS = a1 θ + a2 S, p_TSp = p + p₀ − b0 and λ_TS = λ − c0.

API (T, S arrays of equal shape, p an array of that shape or a scalar)
---------------------------------------------------------------------
``density(T, S, p)``, ``density_anomaly(T, S, p, rho_ref)``, ``density_derivs(T, S, p)``
(Numba), the same with suffix ``_numpy`` (NumPy oracles, bit for bit equal), and the scalar
``@njit`` kernels ``density_elem``, ``rho_anom_elem``, ``drho_elem`` for other kernels.
"""

from __future__ import annotations

import numpy as np
from numba import njit

# Coefficients, MOM_EOS_Wright_full.F90:23-37 (exact strings).
a0 = 7.133718e-4    # Wright alpha_0 fit [m3 kg-1]
a1 = 2.724670e-7    # Wright alpha_0 fit [m3 kg-1 degC-1]
a2 = -1.646582e-7   # Wright alpha_0 fit [m3 kg-1 PSU-1]
b0 = 5.613770e8     # Wright p_0 fit [Pa]
b1 = 3.600337e6     # Wright p_0 fit [Pa degC-1]
b2 = -3.727194e4    # Wright p_0 fit [Pa degC-2]
b3 = 1.660557e2     # Wright p_0 fit [Pa degC-3]
b4 = 6.844158e5     # Wright p_0 fit [Pa PSU-1]
b5 = -8.389457e3    # Wright p_0 fit [Pa degC-1 PSU-1]
c0 = 1.609893e5     # Wright lambda fit [m2 s-2]
c1 = 8.427815e2     # Wright lambda fit [m2 s-2 degC-1]
c2 = -6.931554      # Wright lambda fit [m2 s-2 degC-2]
c3 = 3.869318e-2    # Wright lambda fit [m2 s-2 degC-3]
c4 = -1.664201e2    # Wright lambda fit [m2 s-2 PSU-1]
c5 = -2.765195      # Wright lambda fit [m2 s-2 degC-1 PSU-1]


# --------------------------------------------------------------------------- scalar kernels
@njit(cache=True)
def density_elem(T: float, S: float, p: float) -> float:
    """In situ density ρ(θ, S, p) [kg m⁻³] (MOM_EOS_Wright_full.F90:86-89).

    T: potential temperature θ [°C]; S: practical salinity [PSU]; p: sea pressure [Pa].
    """
    al0 = a0 + (a1 * T + a2 * S)                                         # [m3 kg-1]
    p0 = b0 + (b4 * S + T * (b1 + (T * (b2 + b3 * T) + b5 * S)))         # [Pa]
    lam = c0 + (c4 * S + T * (c1 + (T * (c2 + c3 * T) + c5 * S)))        # [m2 s-2]
    return (p + p0) / (lam + al0 * (p + p0))


@njit(cache=True)
def rho_anom_elem(T: float, S: float, p: float, rho_ref: float) -> float:
    """Density anomaly ρ − ρ_ref [kg m⁻³] in MOM6's anomaly form (MOM_EOS_Wright_full.F90:110-120).

    T [°C] θ, S [PSU], p [Pa], rho_ref [kg m⁻³].
    """
    pa_000 = b0 * (1.0 - a0 * rho_ref) - rho_ref * c0                    # [Pa]
    al_TS = a1 * T + a2 * S                                              # [m3 kg-1]
    al0 = a0 + al_TS                                                     # [m3 kg-1]
    p_TSp = p + (b4 * S + T * (b1 + (T * (b2 + b3 * T) + b5 * S)))       # [Pa]
    lam_TS = c4 * S + T * (c1 + (T * (c2 + c3 * T) + c5 * S))            # [m2 s-2]
    return ((pa_000 + (p_TSp - rho_ref * (p_TSp * al0 + (b0 * al_TS + lam_TS))))
            / ((c0 + lam_TS) + al0 * (b0 + p_TSp)))


@njit(cache=True)
def drho_elem(T: float, S: float, p: float) -> tuple[float, float]:
    """(∂ρ/∂θ [kg m⁻³ °C⁻¹], ∂ρ/∂S [kg m⁻³ PSU⁻¹]) at (θ, S, p) (MOM_EOS_Wright_full.F90:194-202).

    From ρ = (p + p₀)/(λ + α₀(p + p₀)):
    ∂ρ/∂X = [λ ∂p₀/∂X − (p + p₀)((p + p₀) ∂α₀/∂X + ∂λ/∂X)] / (λ + α₀(p + p₀))².
    """
    al0 = a0 + (a1 * T + a2 * S)
    p0 = b0 + (b4 * S + T * (b1 + (T * (b2 + b3 * T) + b5 * S)))
    lam = c0 + (c4 * S + T * (c1 + (T * (c2 + c3 * T) + c5 * S)))
    den = lam + al0 * (p + p0)
    I_denom2 = 1.0 / (den * den)                                         # [s4 m-4]
    drho_dT = I_denom2 * (lam * (b1 + (T * (2.0 * b2 + 3.0 * b3 * T) + b5 * S))
                          - (p + p0) * ((p + p0) * a1
                                        + (c1 + (T * (c2 * 2.0 + c3 * 3.0 * T) + c5 * S))))
    drho_dS = I_denom2 * (lam * (b4 + b5 * T) - (p + p0) * ((p + p0) * a2 + (c4 + c5 * T)))
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
    """In situ density ρ [kg m⁻³] of θ [°C], S [PSU], p [Pa] (Numba)."""
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
    """(∂ρ/∂θ [kg m⁻³ °C⁻¹], ∂ρ/∂S [kg m⁻³ PSU⁻¹]) (Numba)."""
    shape, Tf, Sf, pf = _flat(T, S, p)
    dT = np.empty(Tf.size)
    dS = np.empty(Tf.size)
    _derivs_loop(Tf, Sf, pf, dT, dS)
    return dT.reshape(shape), dS.reshape(shape)


# --------------------------------------------------------------------------- NumPy oracles
def _arrays(T, S, p):
    return np.broadcast_arrays(np.asarray(T, np.float64), np.asarray(S, np.float64),
                               np.asarray(p, np.float64))


def density_numpy(T, S, p) -> np.ndarray:
    """NumPy oracle of :func:`density` (same operations, same order)."""
    T, S, p = _arrays(T, S, p)
    al0 = a0 + (a1 * T + a2 * S)
    p0 = b0 + (b4 * S + T * (b1 + (T * (b2 + b3 * T) + b5 * S)))
    lam = c0 + (c4 * S + T * (c1 + (T * (c2 + c3 * T) + c5 * S)))
    return (p + p0) / (lam + al0 * (p + p0))


def density_anomaly_numpy(T, S, p, rho_ref: float) -> np.ndarray:
    """NumPy oracle of :func:`density_anomaly`."""
    T, S, p = _arrays(T, S, p)
    rho_ref = float(rho_ref)
    pa_000 = b0 * (1.0 - a0 * rho_ref) - rho_ref * c0
    al_TS = a1 * T + a2 * S
    al0 = a0 + al_TS
    p_TSp = p + (b4 * S + T * (b1 + (T * (b2 + b3 * T) + b5 * S)))
    lam_TS = c4 * S + T * (c1 + (T * (c2 + c3 * T) + c5 * S))
    return ((pa_000 + (p_TSp - rho_ref * (p_TSp * al0 + (b0 * al_TS + lam_TS))))
            / ((c0 + lam_TS) + al0 * (b0 + p_TSp)))


def density_derivs_numpy(T, S, p) -> tuple[np.ndarray, np.ndarray]:
    """NumPy oracle of :func:`density_derivs`."""
    T, S, p = _arrays(T, S, p)
    al0 = a0 + (a1 * T + a2 * S)
    p0 = b0 + (b4 * S + T * (b1 + (T * (b2 + b3 * T) + b5 * S)))
    lam = c0 + (c4 * S + T * (c1 + (T * (c2 + c3 * T) + c5 * S)))
    den = lam + al0 * (p + p0)
    I_denom2 = 1.0 / (den * den)
    drho_dT = I_denom2 * (lam * (b1 + (T * (2.0 * b2 + 3.0 * b3 * T) + b5 * S))
                          - (p + p0) * ((p + p0) * a1
                                        + (c1 + (T * (c2 * 2.0 + c3 * 3.0 * T) + c5 * S))))
    drho_dS = I_denom2 * (lam * (b4 + b5 * T) - (p + p0) * ((p + p0) * a2 + (c4 + c5 * T)))
    return drho_dT, drho_dS
