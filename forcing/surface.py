"""
Atmospheric surface forcing for pyOCEAN — Phase 1.

Translates atmospheric boundary conditions (wind stress, heat fluxes,
freshwater flux) into model tendencies applied to the top layer of the
ocean column.

Physics summary
---------------
**Wind stress → momentum** (top layer k = 0):

    du/dt|_wind [j, i_u] = τ_x_u[j, i_u] / (ρ₀ × dz[0])     [m/s²]
    dv/dt|_wind [j_v, i] = τ_y_v[j_v, i] / (ρ₀ × dz[0])     [m/s²]

τ is defined at T-cell centres and interpolated to U/V faces before
dividing by the surface layer thickness (same approach as MITgcm
pkg/exf/exf_apply_forcing.F and HYCOM thermf.F90).

**Heat flux → temperature**:

The net downward heat flux is split into a *non-penetrating* component
(longwave + sensible + latent) applied entirely to the top layer, and a
*penetrating shortwave* component distributed over depth:

    dT/dt|_nonpen [k=0]  = (Q_net − Q_sw) / (ρ₀ Cp dz[0])    [°C/s]

    dT/dt|_sw [k]        = Q_sw × ΔF(k) / (ρ₀ Cp dz[k])      [°C/s]

where ΔF(k) = F(z_f[k]) − F(z_f[k+1]) is the fraction of shortwave
absorbed in layer k, using the Paulson & Simpson (1977) Jerlov Type I
double-exponential profile:

    F(z) = R exp(z / μ₁) + (1 − R) exp(z / μ₂)               (PS77, Table 2)
    R = 0.58,  μ₁ = 0.35 m,  μ₂ = 23.0 m   (Jerlov Type I)

If Q_sw is not supplied, all of Q_net is applied to the top layer only.

**Freshwater → salinity** (virtual salt flux):

    dS/dt|_fw [k=0] = −S_sfc × (E − P) / dz[0]               [PSU/s]

where S_sfc = S[k=0] is the surface salinity and E−P is the net
freshwater flux [m/s] (positive = net evaporation, increases salinity).
This is the standard virtual salt flux formulation used in HYCOM (thermf.F90)
and MITgcm (pkg/exf, pkg/cal).

**Bulk aerodynamic formula** (optional):

    τ = ρ_a × C_D × |U₁₀| × U₁₀                              [N/m²]

where ρ_a = 1.225 kg/m³ and C_D = 1.3 × 10⁻³ (neutral, 10 m reference).

Sign conventions
----------------
All fluxes positive downward (into the ocean):
    Q_net > 0  → ocean warming
    E_P   > 0  → net evaporation → salinity increase → dS/dt > 0
    τ_x   > 0  → eastward wind stress → u acceleration in +x direction

References
----------
Paulson, C.A. & Simpson, J.J. (1977).
    Irradiance measurements in the upper ocean.
    J. Phys. Oceanogr. 7, 952–956.

Large, W.G. & Yeager, S.G. (2009).
    The global climatology of an interannually varying air-sea flux data set.
    Clim. Dyn. 33, 341–364.  (bulk formula review)

HYCOM Consortium. HYCOM source: HYCOM-src/thermf.F90.
MITgcm Team (2023). MITgcm manual: pkg/exf, exf_apply_forcing.F.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.state import ModelState

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

RHO_0: float = 1025.0   # kg/m³  Boussinesq reference density
CP: float = 3996.0       # J/(kg·K)  specific heat of seawater

# ---------------------------------------------------------------------------
# Jerlov Type I shortwave attenuation parameters (Paulson & Simpson 1977)
# ---------------------------------------------------------------------------

_R_PS77: float = 0.58    # fraction in red/near-infrared band
_MU1: float = 0.35       # m   e-folding depth, red band
_MU2: float = 23.0       # m   e-folding depth, blue-green band


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _t_to_u(phi_t: np.ndarray) -> np.ndarray:
    """
    Interpolate a 2-D (ny, nx) field from T-cell centres to U-faces (ny, nx+1).

    Interior faces: arithmetic mean of the two bounding T-cells.
    Boundary faces (i_u = 0, nx): zero (consistent with solid-wall u = 0).
    """
    ny, nx = phi_t.shape
    phi_u = np.zeros((ny, nx + 1))
    phi_u[:, 1:nx] = 0.5 * (phi_t[:, :-1] + phi_t[:, 1:])
    # i_u = 0 and i_u = nx stay zero (solid walls)
    return phi_u


def _t_to_v(phi_t: np.ndarray) -> np.ndarray:
    """
    Interpolate a 2-D (ny, nx) field from T-cell centres to V-faces (ny+1, nx).

    Interior faces: arithmetic mean of the two bounding T-cells.
    Boundary faces (j_v = 0, ny): zero (consistent with solid-wall v = 0).
    """
    ny, nx = phi_t.shape
    phi_v = np.zeros((ny + 1, nx))
    phi_v[1:ny, :] = 0.5 * (phi_t[:-1, :] + phi_t[1:, :])
    # j_v = 0 and j_v = ny stay zero (solid walls)
    return phi_v


def _sw_profile(z_f: np.ndarray) -> np.ndarray:
    """
    Shortwave penetration profile F(z) at layer interfaces.

    F(z) = R exp(z/μ₁) + (1−R) exp(z/μ₂)              (Paulson & Simpson 1977)

    Parameters
    ----------
    z_f : np.ndarray, shape (nz+1,)
        Interface depths [m], z_f[0] = 0 (surface), z_f[k] < 0 below.

    Returns
    -------
    F : np.ndarray, shape (nz+1,)
        Fraction of incident shortwave remaining at each interface.
        F[0] = 1.0 (all shortwave at surface), F[-1] ≈ 0.
    """
    return _R_PS77 * np.exp(z_f / _MU1) + (1.0 - _R_PS77) * np.exp(z_f / _MU2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def bulk_wind_stress(
    u10: np.ndarray,
    v10: np.ndarray,
    rho_a: float = 1.225,
    Cd: float = 1.3e-3,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Compute surface wind stress from 10-m wind speed using the bulk
    aerodynamic formula (Large & Yeager 2009, eq. 1):

        τ_x = ρ_a × C_D × |U₁₀| × u₁₀                [N/m²]
        τ_y = ρ_a × C_D × |U₁₀| × v₁₀                [N/m²]

    Parameters
    ----------
    u10 : np.ndarray, shape (ny, nx)
        Zonal 10-m wind speed [m/s].
    v10 : np.ndarray, shape (ny, nx)
        Meridional 10-m wind speed [m/s].
    rho_a : float, optional
        Air density [kg/m³]. Default 1.225.
    Cd : float, optional
        Neutral drag coefficient at 10 m. Default 1.3 × 10⁻³.

    Returns
    -------
    tau_x, tau_y : np.ndarray, shape (ny, nx)
        Wind stress components [N/m²].
    """
    speed = np.sqrt(u10 ** 2 + v10 ** 2)
    tau_x = rho_a * Cd * speed * u10
    tau_y = rho_a * Cd * speed * v10
    return tau_x, tau_y


def surface_tendency(
    state: ModelState,
    v_grid: xr.Dataset,
    *,
    tau_x: np.ndarray | None = None,
    tau_y: np.ndarray | None = None,
    Q_net: np.ndarray | None = None,
    Q_sw: np.ndarray | None = None,
    E_P: np.ndarray | None = None,
) -> ModelState:
    """
    Compute the ModelState tendency from atmospheric surface forcing.

    Applies wind stress to the top-layer momentum, heat flux to
    temperature, and freshwater flux to salinity.  All fluxes are
    applied only to the **top layer** (k = 0) except for the penetrating
    shortwave, which is distributed over depth following the Paulson &
    Simpson (1977) attenuation profile.

    Parameters
    ----------
    state : ModelState
        Current model state.  Used only for surface salinity S[k=0]
        (virtual salt flux) and for coordinate information.
    v_grid : xr.Dataset
        Vertical grid providing ``z_f`` [m] and ``dz`` [m].
    tau_x : np.ndarray (ny, nx) or None
        Zonal wind stress [N/m²].  Default: zeros.
    tau_y : np.ndarray (ny, nx) or None
        Meridional wind stress [N/m²].  Default: zeros.
    Q_net : np.ndarray (ny, nx) or None
        Net downward heat flux [W/m²], positive into ocean.  Default: zeros.
    Q_sw : np.ndarray (ny, nx) or None
        Penetrating shortwave radiation [W/m²], positive into ocean.
        If None, all of Q_net is treated as non-penetrating (applied to
        top layer only).  Must satisfy 0 ≤ Q_sw ≤ Q_net.
    E_P : np.ndarray (ny, nx) or None
        Evaporation minus precipitation [m/s], positive = net evaporation.
        Default: zeros.

    Returns
    -------
    ModelState
        Tendency dstate/dt from surface forcing [per second].
        u, v → [m/s²]; T → [°C/s]; S → [PSU/s]; η, w → 0.

    Notes
    -----
    * The wind stress tendency is applied only at the top layer and
      interpolated to the staggered U/V faces (Section "Wind stress"
      above).
    * For the shortwave profile to be physical, ``Q_sw`` should be ≤
      ``Q_net``; no clipping is applied here.

    Examples
    --------
    >>> tau_x = np.full((ny, nx), 0.1)   # 0.1 N/m² eastward wind
    >>> tend = surface_tendency(state, vgrid, tau_x=tau_x)
    >>> tend.u[0]  # top-layer u tendency from wind [m/s²]
    """
    dz  = v_grid["dz"].values   # (nz,)
    z_f = v_grid["z_f"].values  # (nz+1,)

    nz = dz.shape[0]
    ny, nx = state.T.shape[1], state.T.shape[2]

    zeros_2d = np.zeros((ny, nx))

    if tau_x is None:
        tau_x = zeros_2d
    if tau_y is None:
        tau_y = zeros_2d
    if Q_net is None:
        Q_net = zeros_2d
    if Q_sw is None:
        Q_sw = zeros_2d
    if E_P is None:
        E_P = zeros_2d

    # ---- Wind stress → momentum (top layer k = 0) -------------------------
    # Interpolate τ from T-cell centres to staggered U/V faces
    tau_x_u = _t_to_u(tau_x)   # (ny, nx+1)
    tau_y_v = _t_to_v(tau_y)   # (ny+1, nx)

    # du/dt = τ_x / (ρ₀ dz[0])  — only top layer
    du_dt = np.zeros_like(state.u.values)   # (nz, ny, nx+1)
    dv_dt = np.zeros_like(state.v.values)   # (nz, ny+1, nx)
    du_dt[0] = tau_x_u / (RHO_0 * dz[0])
    dv_dt[0] = tau_y_v / (RHO_0 * dz[0])

    # ---- Heat flux → temperature ------------------------------------------
    dT_dt = np.zeros((nz, ny, nx))

    # Non-penetrating component: applied entirely to surface layer
    Q_nonsw = Q_net - Q_sw
    dT_dt[0] += Q_nonsw / (RHO_0 * CP * dz[0])   # (ny, nx)

    # Penetrating shortwave: distributed by Jerlov Type I profile
    if np.any(Q_sw != 0.0):
        F = _sw_profile(z_f)   # (nz+1,), F[0]=1, F[nz]≈0
        for k in range(nz):
            dF_k = F[k] - F[k + 1]   # fraction absorbed in layer k
            dT_dt[k] += Q_sw * dF_k / (RHO_0 * CP * dz[k])

    # ---- Freshwater flux → salinity (virtual salt flux) -------------------
    # Virtual salt flux: dS/dt = +S_sfc × (E−P) / dz[0]
    # (MITgcm pkg/exf; HYCOM thermf.F90)
    # Positive E−P → net evaporation → less water, same salt → S increases
    S_sfc = state.S.values[0]   # (ny, nx) surface salinity
    dS_dt = np.zeros((nz, ny, nx))
    dS_dt[0] = S_sfc * E_P / dz[0]

    # ---- Assemble tendency ModelState -------------------------------------
    du = xr.DataArray(
        du_dt, dims=state.u.dims, coords=state.u.coords,
        attrs={"units": "m/s2", "long_name": "Surface wind stress du/dt"},
    )
    dv = xr.DataArray(
        dv_dt, dims=state.v.dims, coords=state.v.coords,
        attrs={"units": "m/s2", "long_name": "Surface wind stress dv/dt"},
    )
    dT = xr.DataArray(
        dT_dt, dims=state.T.dims, coords=state.T.coords,
        attrs={"units": "degC/s", "long_name": "Surface heat flux dT/dt"},
    )
    dS = xr.DataArray(
        dS_dt, dims=state.S.dims, coords=state.S.coords,
        attrs={"units": "PSU/s", "long_name": "Freshwater flux dS/dt"},
    )

    return ModelState(
        u=du,
        v=dv,
        w=xr.zeros_like(state.w),
        T=dT,
        S=dS,
        eta=xr.zeros_like(state.eta),
        time=0.0,
    )
