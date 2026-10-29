"""
Penetrating shortwave radiation: Paulson & Simpson (1977) two-band absorption in Jerlov water types (N3 task 2).

Fraction of the surface shortwave flux that reaches depth z <= 0 [m]

    swfrac(z) = R exp(z / zeta1) + (1 - R) exp(z / zeta2)                 (Paulson & Simpson 1977, eq. 3-4)

with the Jerlov-type coefficients of Large et al. (1994, table) as coded in CROCO ``lmd_swfrac.F`` (version of
``~/pyOCEAN_refs/croco``): ``lmd_mu1`` (zeta1) line 47, ``lmd_mu2`` (zeta2) line 49, ``lmd_r1`` (R) line 51, and the formula
at lines 76-88.  Type I: R = 0.58, zeta1 = 0.35 m, zeta2 = 23 m.

Deposition rule at the bed.  What reaches the bottom interface (1 - sum of layer absorptions = swfrac(-D)) is
deposited in the bottom layer, as the two production models do:
  * CROCO ``step3d_t.F:567`` sets the bottom flux ``FC(0) = -dt btflx`` (no shortwave term), and the interior
    interface flux ``FC(k) = dt srflx swdk(k)`` (``step3d_t.F:578-586``, the term at line 584): the shortwave that crosses the lowest
    interface stays in the lowest layer;
  * MOM6 ``MOM_opacity.F90:824-842`` (``absorbAllSW``; same rule in ``sumSWoverBands``, lines 994-1011): the radiation left over at the bottom
    is given to the bottom layer (``T_chg = Pen_SW_rem / h_heat``); a bottom-mud model is mentioned there as an
    alternative that is not implemented.
So the full Q_sw is always absorbed in the column.

Layer heating (Boussinesq, potential temperature [°C], k = 0 at the bed, interface index i = 0 … N, i = N the
surface, f_i = swfrac at interface i, f_N = 1):

    rho0 cp h_k dT_k/dt = Q_ns delta_{k,N-1} + Q_sw (f_{k+1} - f_k)     k >= 1
    rho0 cp h_0 dT_0/dt =                      Q_sw  f_1                 (f_0 is deposited in layer 0)
"""
from __future__ import annotations

import numpy as np

# Jerlov water types I, IA, IB, II, III: R = lmd_r1, zeta1 = lmd_mu1 [m], zeta2 = lmd_mu2 [m]
# (CROCO lmd_swfrac.F:47-51; Large et al. 1994 / Paulson & Simpson 1977 table 2)
JERLOV = {
    "I": (0.58, 0.35, 23.0),
    "IA": (0.62, 0.6, 20.0),
    "IB": (0.67, 1.0, 17.0),
    "II": (0.77, 1.5, 14.0),
    "III": (0.78, 1.4, 7.9),
}


def swfrac(z: np.ndarray, jerlov: str = "I") -> np.ndarray:
    """
    Fraction of the surface shortwave flux that reaches depth ``z`` (Paulson & Simpson 1977, two bands).

    swfrac(z) = R exp(z/zeta1) + (1 - R) exp(z/zeta2), z <= 0 [m] (negative downward); 1 at z = 0.
    No cutoff of the exponents (CROCO lmd_swfrac.F:77-85 sets a band to 0 below exp(-20), an error of 2e-13 of
    the flux at 10 m for the 0.35 m band); exp underflows to 0 by itself at ~ -745.
    """
    try:
        r1, mu1, mu2 = JERLOV[jerlov]
    except KeyError:
        raise ValueError(f"unknown Jerlov type {jerlov!r} (valid: {', '.join(JERLOV)})") from None
    z = np.asarray(z, float)
    a, b = z / mu1, z / mu2
    return r1 * np.exp(a) + (1.0 - r1) * np.exp(b)


def layer_heating(h: np.ndarray, Q_ns: np.ndarray, Q_sw: np.ndarray | None, jerlov: str = "I"):
    """
    Heat absorbed by each layer [W m⁻², per unit area], and the interface fractions.

    h : (N, ny, nx) layer thicknesses [m], k = 0 at the bed (dry/land columns: zeros).
    Q_ns, Q_sw : (ny, nx) fluxes into the ocean [W m⁻²]; Q_sw None = no shortwave.
    Returns (q, f): q (N, ny, nx) with Σ_k q = Q_ns + Q_sw (rule in the module docstring) and
    f (N+1, ny, nx) = swfrac at the interfaces (None without Q_sw).
    """
    N = h.shape[0]
    q = np.zeros(h.shape)
    q[-1] += Q_ns
    if Q_sw is None:
        return q, None
    depth = np.zeros((N + 1,) + h.shape[1:])                    # depth below the surface of interface i
    depth[:-1] = np.cumsum(h[::-1], axis=0)[::-1]
    f = swfrac(-depth, jerlov)
    q += Q_sw * (f[1:] - f[:-1])
    q[0] += Q_sw * f[0]                                         # what reaches the bed stays in the bottom layer
    return q, f
