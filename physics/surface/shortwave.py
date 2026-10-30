"""
Penetrating shortwave radiation: Paulson & Simpson (1977) two-band absorption in Jerlov water types (N3 task 2).

Fraction of the surface shortwave flux that reaches depth z <= 0 [m]

    swfrac(z) = R exp(z / zeta1) + (1 - R) exp(z / zeta2)                 (Paulson & Simpson 1977, eq. 3-4)

with the Jerlov-type coefficients of Large et al. (1994, table) as coded in CROCO ``lmd_swfrac.F`` (version of
``~/pyOCEAN_refs/croco``): ``lmd_mu1`` (zeta1) line 47, ``lmd_mu2`` (zeta2) line 49, ``lmd_r1`` (R) line 51, and the formula
at lines 76-88.  Type I: R = 0.58, zeta1 = 0.35 m, zeta2 = 23 m.

Deposition of what reaches the bed, and thin layers (final-review fix C1, MOM6 ``absorbRemainingSW``,
``~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_opacity.F90``, git 0e9a561).  The shortwave that crosses the
bottom interface, Q_sw f_0 = Q_sw swfrac(-D), is not kept in the bed layer: it is spread through the water column in
proportion to the layer thickness, as MOM6 does with ``absorbAllSW`` (lines 813-816 accumulate ``h_heat``, 824-842
compute ``T_chg = Pen_SW_rem / h_heat``, 846-860 add it to every layer):

    w_k = h_k                          h_k >= 2 h_min_heat
    w_k = 2 h_k - 2 h_min_heat         h_min_heat < h_k < 2 h_min_heat
    w_k = 0                            h_k <= h_min_heat,          share_k = Q_sw f_0 w_k / sum_j w_j.

The bed layer therefore warms at the column-mean rate of the remainder, whatever its thickness (the earlier rule put
all of Q_sw f_0 into the lowest layer with mass, CROCO ``step3d_t.F:567-586``, which is fine for σ layers of D/N but
heated a 1 mm z*/hybrid bed layer to ~200 °C in 4 h; MOM6 says the remainder "could go into the bottom (with a bottom
mud model), or it could be redistributed back through the water column", line 826-828, and does the latter).  This
rule is used in every vertical coordinate, so a σ column with Q_sw differs from CROCO by the bed remainder only
(Q_sw f_0, 5 % of Q_sw at 50 m in type I water, 6e-4 at 150 m).

Thin layers.  A layer with h <= h_min_heat gets no heat at all (its band absorption joins the bed remainder), so no
heated layer is thinner than the floor of the implicit vertical solvers (``core.vertical.ale.DZ_FLOOR`` = 1e-6 m),
and the heat content stays exact.  MOM6 uses h_min_heat = 2 Angstrom + H_subroundoff (line 709, 2e-10 m); the larger
value here is that diffusion floor (deviation stated on purpose).  The non-solar flux Q_ns enters from the top as
MOM6 ``applyBoundaryFluxesInOut`` (``MOM_diabatic_aux.F90:1101-1118``): a layer takes the fraction
min(1, h_k / D_f) of what is left, D_f = MINIMUM_FORCING_DEPTH = 1 mm (``MOM_diabatic_driver.F90:3447-3452``), and
passes the rest to the layer below; the deepest heatable layer takes what is left, so Q_ns is always absorbed.  For
layers thicker than 1 mm (every σ layer of a wet column with D/N >= 1 mm) this is Q_ns into the top layer, bit for bit.
So the full Q_ns + Q_sw is always absorbed in the column.  (MOM6 also limits the absorption when the whole column is
thinner than H_limit_fluxes = 1 Angstrom; wet columns here are thicker than d_min.)

Layer heating (Boussinesq, potential temperature [°C], k = 0 at the bed, interface index i = 0 … N, i = N the
surface, f_i = swfrac at interface i, f_N = 1, a_k = 1 if h_k > h_min_heat else 0, c_k the Q_ns fractions above):

    rho0 cp h_k dT_k/dt = c_k Q_ns + a_k Q_sw (f_{k+1} - f_k) + w_k / sum(w) Q_sw (f_0 + sum_j (1 - a_j)(f_{j+1} - f_j))
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


# Thinnest layer that is heated [m]: the floor of the implicit vertical solvers (core.vertical.ale.DZ_FLOOR; MOM6
# uses 2 Angstrom, MOM_opacity.F90:709).  Depth over which Q_ns is spread when the top layers are thin [m] (MOM6
# MINIMUM_FORCING_DEPTH default, MOM_diabatic_driver.F90:3447-3452).
H_MIN_HEAT = 1e-6
FORCING_DEPTH = 1e-3


def remainder_weights(h: np.ndarray, h_min_heat: float = H_MIN_HEAT) -> np.ndarray:
    """
    Weights w_k [m] of the bed shortwave remainder (MOM6 ``MOM_opacity.F90:813-816``): h_k for h_k >= 2 h_min_heat,
    2 h_k − 2 h_min_heat between h_min_heat and 2 h_min_heat (a layer in that band warms by T_chg (2 − 2 h_min/h),
    lines 853-858, the same heat), 0 at or below h_min_heat.  h (N, ny, nx), k = 0 at the bed.
    """
    return np.where(h >= 2.0 * h_min_heat, h, np.where(h > h_min_heat, 2.0 * h - 2.0 * h_min_heat, 0.0))


def nonsolar_fractions(h: np.ndarray, h_min_heat: float = H_MIN_HEAT,
                       forcing_depth: float = FORCING_DEPTH) -> np.ndarray:
    """
    Fractions c_k (sum 1 in a column with a heatable layer) of the non-solar flux Q_ns per layer, from the top down
    (MOM6 ``applyBoundaryFluxesInOut``, ``MOM_diabatic_aux.F90:1101-1118``: fractionOfForcing = min(1, h / D_f) of what
    is left, D_f = 1 mm): layers at or below h_min_heat take nothing, the deepest heatable layer takes what is left.
    Columns with no heatable layer get c = 0 (dry/land: h = 0, no flux).  h (N, ny, nx), k = 0 at the bed.
    """
    N = h.shape[0]
    ok = h > h_min_heat
    k_last = np.argmax(ok, axis=0)                             # deepest heatable layer (k = 0 at the bed)
    c = np.zeros_like(h, dtype=float)
    left = np.where(ok.any(axis=0), 1.0, 0.0)
    for k in range(N - 1, -1, -1):
        frac = np.where(ok[k], np.where(k == k_last, 1.0, np.minimum(1.0, h[k] / forcing_depth)), 0.0)
        c[k] = frac * left
        left = left - c[k]
    return c


def layer_heating(h: np.ndarray, Q_ns: np.ndarray, Q_sw: np.ndarray | None, jerlov: str = "I"):
    """
    Heat absorbed by each layer [W m⁻², per unit area], and the interface fractions.

    h : (N, ny, nx) layer thicknesses [m], k = 0 at the bed (dry/land columns: zeros).
    Q_ns, Q_sw : (ny, nx) fluxes into the ocean [W m⁻²]; Q_sw None = no shortwave.
    Returns (q, f): q (N, ny, nx) with Σ_k q = Q_ns + Q_sw in every column with a layer thicker than H_MIN_HEAT
    (rules in the module docstring: Q_ns from the top over the 1 mm forcing depth, the band absorption in each
    heatable layer, the bed remainder and the band absorption of the layers <= H_MIN_HEAT spread over the column with
    ``remainder_weights``) and f (N+1, ny, nx) = swfrac at the interfaces (always, also without Q_sw).
    """
    q = nonsolar_fractions(h) * Q_ns
    depth = np.zeros((h.shape[0] + 1,) + h.shape[1:])          # depth below the surface of interface i
    depth[:-1] = np.cumsum(h[::-1], axis=0)[::-1]
    f = swfrac(-depth, jerlov)
    if Q_sw is None:
        return q, f
    band = Q_sw * (f[1:] - f[:-1])                             # absorbed between the interfaces of layer k
    ok = h > H_MIN_HEAT
    rem = Q_sw * f[0] + np.sum(np.where(ok, 0.0, band), axis=0)   # bed remainder + band of the no-heat layers
    w = remainder_weights(h)
    sw = w.sum(axis=0)
    share = np.where(sw > 0, rem / np.where(sw > 0, sw, 1.0), 0.0)
    q = q + np.where(ok, band, 0.0) + w * share                # MOM6 absorbRemainingSW, MOM_opacity.F90:824-860
    return q, f
