"""
Stratification seen by the turbulence closures with a nonlinear equation of state (TEOS-10 in the 3-D core,
task 6; spec ``docs/superpowers/specs/2026-10-31-teos10-3d-design.md`` goal C, criterion 6).

Why not the in-situ density difference
--------------------------------------
With a nonlinear EOS the density of two neighbouring layers evaluated at their own (different) pressures differs
by the compressibility, ≈ 4–5 kg m⁻³ per km, which says nothing about the static stability.  MOM6 therefore uses
**locally referenced** quantities in every mixing scheme (``docs/design/teos10_refs/mom6_eos.md`` §5):

* Jackson shear mixing, ``find_N2`` (background, BBL, tidal) and double diffusion take the density derivatives at
  the interface mean (T, S) and the interface pressure,

      N²_K = −(g/ρ₀) [∂ρ/∂T (T_up − T_dn) + ∂ρ/∂S (S_up − S_dn)] / ½(Δz_up + Δz_dn)          [s⁻²]

  with ∂ρ/∂T, ∂ρ/∂S at (½(T_up + T_dn), ½(S_up + S_dn), p_K) (``MOM_kappa_shear.F90:1128-1136, 1496-1500``;
  ``MOM_set_diffusivity.F90:1160-1200``).  ``n2_local`` follows the kappa_shear arithmetic:
  dbuoy_dT = (−g/ρ₀) ∂ρ/∂T (``calculate_density_derivs(..., scale=-g_R0)``, ``MOM_EOS.F90:823-829``) and
  N² = (2/(Δz_up + Δz_dn)) (dbuoy_dT ΔT + dbuoy_dS ΔS), **signed** (MOM6's kappa_shear clips it with max(0, ·)
  inside its own iteration; pyOCEAN's GLS uses the signed value as it does with the linear EOS).
* KPP evaluates the full density of both parts at one pressure, the pressure of the top interface of layer k
  (``MOM_CVMix_KPP.F90:1396-1436``): ΔB(k) = (g/ρ₀)[ρ(T_k, S_k, p_k) − ρ(T_sl, S_sl, p_k)] for the bulk Richardson
  number and N²(k) = (g/ρ₀)[ρ(T_k, S_k, p_k) − ρ(T_k+1, S_k+1, p_k)] / (½(h_k + h_k+1)) (``kpp_delta_rho``).
* The KPP surface buoyancy flux uses ∂ρ/∂T, ∂ρ/∂S of the top layer at p = 0 (no atmospheric pressure)
  (``MOM_forcing_type.F90:1116-1148``; ``surface_alpha_beta``).

Pressure in the closures (spec, units): p = p_surf + Σ g ρ₀ h from the surface, p_surf = 0, accumulated layer by
layer as MOM6 does (``pressure(K) = pressure(K-1) + gR0*h(k-1)``, ``MOM_kappa_shear.F90:1128-1130``;
``MOM_set_diffusivity.F90:1166-1167``; ``MOM_CVMix_KPP.F90:1408``) (``interface_pressure``).

With the linear EOS none of this is used: every closure keeps its own expression (bit for bit as before).

Index order: (k, column), k = 0 at the bed; interfaces N+1, interface k between layers k−1 and k.  The Numba
scalars ``buoyancy_jump_elem`` and ``n2_iface_elem`` are shared by the vectorised functions here and by the
compiled closure kernels (``physics.mixing.gls_merged``, ``bbl_mixing``, ``kappa_shear``), so the kernels and
their NumPy oracles agree bit for bit.  EOS codes: ``EOS_CODES`` (0 linear, 1 WRIGHT_FULL, 2 ROQUET_RHO = teos10).

References: Wright, D. G. (1997), J. Atmos. Oceanic Technol. 14, 735–740 [verificar]; Roquet et al. (2015), Ocean
Modell. 90, 29–43 [verificar]; IOC, SCOR and IAPSO (2010), TEOS-10 manual; McDougall, T. J. (1987), Thermobaricity,
cabbeling, and water-mass conversion, J. Geophys. Res. 92, 5448–5464 [verificar].  MOM6 git 0e9a561.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from numba import njit

from physics.eos import roquet_rho, wright_full

EOS_CODES = {"linear": 0, "wright_full": 1, "teos10": 2}


@dataclass
class DensityContext:
    """
    What a closure needs for MOM6's locally referenced stratification with a nonlinear EOS, built by the driver
    (``core.model.driver.Driver._closure_density``) and passed to ``Closure.update(..., dens=...)``.

    eos : the model's EOS component (``core.components.eos``); ``eos.linear`` True → the closures ignore the context
    T, S : (N, ny, nx) layer temperature and salinity in the EOS convention (``eos.ts_kind``) [°C, PSU or g kg⁻¹]
    p_layer : (N, ny, nx) pressure at the layer centres [Pa], g ρ₀ (Σ_{above} h + ½ h_k) from the surface
    rho : (N, ny, nx) in-situ layer density ρ(T, S, p_layer) [kg m⁻³] (the ``rho`` argument of ``update``)
    """
    eos: Any
    T: np.ndarray
    S: np.ndarray
    p_layer: np.ndarray
    rho: np.ndarray

    @property
    def nonlinear(self) -> bool:
        return not self.eos.linear


def nonlinear(dens: DensityContext | None) -> bool:
    """True when ``dens`` asks for the nonlinear path (given and with a nonlinear EOS)."""
    return dens is not None and not dens.eos.linear


def eos_code(eos) -> int:
    """Integer code of an EOS component or name for the Numba kernels (``EOS_CODES``)."""
    name = eos if isinstance(eos, str) else eos.name
    try:
        return EOS_CODES[name]
    except KeyError:
        raise ValueError(f"unknown EOS {name!r}: expected one of {sorted(EOS_CODES)}") from None


# --------------------------------------------------------------------------- scalar kernels
@njit(cache=True)
def drho_code(T: float, S: float, p: float, code: int) -> tuple[float, float]:
    """(∂ρ/∂T, ∂ρ/∂S) [kg m⁻³ °C⁻¹, kg m⁻³ (PSU or g kg⁻¹)⁻¹] of the nonlinear EOS ``code`` (1 WRIGHT_FULL,
    2 ROQUET_RHO) at (T, S, p [Pa]) (``MOM_EOS_Wright_full.F90:194-202``, ``MOM_EOS_Roquet_rho.F90:384-411``)."""
    if code == 1:
        return wright_full.drho_elem(T, S, p)
    return roquet_rho.drho_elem(T, S, p)


@njit(cache=True)
def buoyancy_jump_elem(T_up: float, S_up: float, T_dn: float, S_dn: float, p: float, code: int,
                       g_rho0: float) -> float:
    """
    Locally referenced buoyancy jump across one interface [m s⁻²] (``MOM_kappa_shear.F90:1131-1136, 1496-1500``):

        Δb = dbuoy_dT (T_up − T_dn) + dbuoy_dS (S_up − S_dn),
        dbuoy_dX = (−g/ρ₀) ∂ρ/∂X(½(T_up + T_dn), ½(S_up + S_dn), p)

    p : interface pressure [Pa];  g_rho0 : g/ρ₀ [m⁴ s⁻² kg⁻¹] (MOM6 ``g_R0``).  Δb > 0 is stable.
    """
    T_int = 0.5 * (T_up + T_dn)
    S_int = 0.5 * (S_up + S_dn)
    dT, dS = drho_code(T_int, S_int, p, code)
    sc = -g_rho0                                                       # MOM_EOS.F90:823-826 (scale = −g_R0)
    return (sc * dT) * (T_up - T_dn) + (sc * dS) * (S_up - S_dn)


@njit(cache=True)
def n2_iface_elem(T_up: float, S_up: float, T_dn: float, S_dn: float, dz_up: float, dz_dn: float, p: float,
                  code: int, g_rho0: float) -> float:
    """Signed locally referenced N² [s⁻²] at one interface: (2/(Δz_up + Δz_dn)) Δb (``buoyancy_jump_elem``;
    I_dz_int of ``MOM_kappa_shear.F90:1033``)."""
    return (2.0 / (dz_up + dz_dn)) * buoyancy_jump_elem(T_up, S_up, T_dn, S_dn, p, code, g_rho0)


@njit(cache=True)
def _n2_loop(T, S, dz, p, code, g_rho0, out):
    n, m = T.shape
    for j in range(m):
        for k in range(n - 1):
            out[k, j] = n2_iface_elem(T[k + 1, j], S[k + 1, j], T[k, j], S[k, j], dz[k + 1, j], dz[k, j],
                                      p[k + 1, j], code, g_rho0)


@njit(cache=True)
def _jump_loop(T, S, p, code, g_rho0, out):
    n, m = T.shape
    for j in range(m):
        for k in range(n - 1):
            out[k, j] = buoyancy_jump_elem(T[k + 1, j], S[k + 1, j], T[k, j], S[k, j], p[k + 1, j], code, g_rho0)


# --------------------------------------------------------------------------- array functions
def _cols(x: np.ndarray, n: int) -> np.ndarray:
    return np.ascontiguousarray(np.asarray(x, np.float64).reshape(n, -1))


def interface_pressure(h: np.ndarray, g: float, rho0: float, p_surf: float = 0.0) -> np.ndarray:
    """
    Hydrostatic pressure at the interfaces [Pa] from the layer thicknesses h (N, …) [m], k = 0 at the bed:
    p_N = p_surf, p_k = p_k+1 + (g ρ₀) h_k (MOM6 running sum from the surface, ``MOM_kappa_shear.F90:1128-1130``).
    Returns (N+1, …).
    """
    h = np.asarray(h, np.float64)
    gr0 = g * rho0
    p = np.empty((h.shape[0] + 1,) + h.shape[1:])
    p[-1] = p_surf
    for k in range(h.shape[0] - 1, -1, -1):
        p[k] = p[k + 1] + gr0 * h[k]
    return p


def _g_rho0(eos, rho0: float | None, g: float) -> float:
    return g / (eos.rho0 if rho0 is None else rho0)


def n2_local(T: np.ndarray, S: np.ndarray, dz: np.ndarray, p_iface: np.ndarray, eos, rho0: float | None = None,
             g: float = 9.81) -> np.ndarray:
    """
    Signed locally referenced N² [s⁻²] at the interior interfaces (module docstring; ``MOM_kappa_shear.F90:1128-1136,
    1496-1500``).

    T, S : (N, …) layer T [°C] and S [PSU or g kg⁻¹] in the EOS convention;  dz : (N, …) layer thickness [m];
    p_iface : (N+1, …) interface pressure [Pa] (``interface_pressure``);  eos : nonlinear EOS component;
    rho0 : ρ₀ [kg m⁻³] (None: ``eos.rho0``);  g [m s⁻²].  Returns (N−1, …), entry k = interface between layers k
    and k+1.  A zero Δz_up + Δz_dn gives ±inf or NaN, as the linear ``buoyancy_frequency2``.
    """
    T = np.asarray(T, np.float64)
    n = T.shape[0]
    out = np.empty((n - 1, T[0].size))
    _n2_loop(_cols(T, n), _cols(S, n), _cols(dz, n), _cols(p_iface, n + 1), eos_code(eos), _g_rho0(eos, rho0, g),
             out)
    return out.reshape((n - 1,) + T.shape[1:])


def buoyancy_jumps(T: np.ndarray, S: np.ndarray, p_iface: np.ndarray, eos, rho0: float | None = None,
                   g: float = 9.81) -> np.ndarray:
    """Locally referenced buoyancy jumps Δb (N−1, …) [m s⁻²] at the interior interfaces (``buoyancy_jump_elem``;
    N² × ½(Δz_up + Δz_dn)).  Same arguments as ``n2_local``."""
    T = np.asarray(T, np.float64)
    n = T.shape[0]
    out = np.empty((n - 1, T[0].size))
    _jump_loop(_cols(T, n), _cols(S, n), _cols(p_iface, n + 1), eos_code(eos), _g_rho0(eos, rho0, g), out)
    return out.reshape((n - 1,) + T.shape[1:])


def kpp_delta_rho(T: np.ndarray, S: np.ndarray, T_sl: np.ndarray, S_sl: np.ndarray, p_top: np.ndarray,
                  eos) -> tuple[np.ndarray, np.ndarray]:
    """
    KPP density differences at a common pressure (``MOM_CVMix_KPP.F90:1396-1436``), model orientation.

    T, S : (N, …) layer T and S;  T_sl, S_sl : (N, …) surface-layer means of the trial depth of each layer;
    p_top : (N, …) pressure at the top interface of each layer [Pa] (``interface_pressure(h)[1:]``);
    eos : nonlinear EOS component.

    Returns
    -------
    drho_sl : (N, …) ρ(T_k, S_k, p_top_k) − ρ(T_sl_k, S_sl_k, p_top_k) [kg m⁻³] (MOM6 deltaRho, ``:1428``);
        ΔB = (g/ρ₀) drho_sl.
    drho_up : (N−1, …) ρ(T_k, S_k, p_top_k) − ρ(T_k+1, S_k+1, p_top_k) [kg m⁻³], entry k = the top interface of
        layer k (MOM6 ``rho_1D(kk+2) − rho_1D(kk+3)``, ``:1435``); N² = (g/ρ₀) drho_up / (½(h_k + h_k+1) + h_sub).
    """
    T, S, T_sl, S_sl, p_top = (np.asarray(x, np.float64) for x in (T, S, T_sl, S_sl, p_top))
    rho_k = eos.density(T, S, p_top)
    drho_sl = rho_k - eos.density(T_sl, S_sl, p_top)
    drho_up = rho_k[:-1] - eos.density(T[1:], S[1:], p_top[:-1])
    return drho_sl, drho_up


def surface_alpha_beta(T_top: np.ndarray, S_top: np.ndarray, eos) -> tuple[np.ndarray, np.ndarray]:
    """(∂ρ/∂T, ∂ρ/∂S) of the top layer at p = 0 [kg m⁻³ °C⁻¹, kg m⁻³ (PSU or g kg⁻¹)⁻¹] for the KPP surface
    buoyancy flux B = −(g/ρ₀)(∂ρ/∂S F_S + ∂ρ/∂T F_T) (``MOM_forcing_type.F90:1116-1148``)."""
    T_top = np.asarray(T_top, np.float64)
    return eos.density_derivs(T_top, np.asarray(S_top, np.float64), np.zeros_like(T_top))
