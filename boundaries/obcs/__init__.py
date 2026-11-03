"""
Open Boundary Conditions (OBCS) for pyOCEAN — Phase 1.

Implements two complementary strategies for open lateral boundaries:

1. **Orlanski radiation condition** (Orlanski 1976)
   Allows outgoing waves to leave the domain without reflection by
   estimating the outward phase speed from the interior solution and
   advecting the boundary value accordingly:

       ∂φ/∂t + c_n ∂φ/∂n = 0                           (Orlanski 1976)

   Phase speed estimate (interior, one cell from boundary):

       c_n = max(0, Δt_φ_int × dn / (dt × Δn_φ))       (Orlanski 1976)

   where Δt_φ_int = φ_int(t) − φ_int(t−dt) and Δn_φ = φ_int − φ_bdy.
   This is equivalent to the MITgcm OBCS package formulation.

   where Δt_φ_int = φ_int(t) − φ_int(t−dt) is the time tendency of the
   interior point and Δn_φ / dn is the normal gradient at the boundary.
   c_n is clipped to [0, dn/dt] (causality + CFL constraint).

   For incoming characteristics (c_n ≤ 0 or specified inflow), the
   boundary value is set to the prescribed exterior value φ_ext.

   Direct-update form (not a tendency; applied after timestepping):

       φ_bdy(t+dt) = φ_bdy(t) + c_n × (dt/dn) × (φ_int − φ_bdy)    outflow
       φ_bdy(t+dt) = φ_ext                                             inflow

   This matches the MITgcm OBCS package approach (obcs_calc.F,
   obcs_apply.F) and HYCOM latbdy.F90 (radiation variant).

2. **Sponge layer nudging** (Modelling strategy from Döscher et al. 2010)
   Smoothly damps all fields toward prescribed exterior values within a
   buffer zone of width L_sponge grid cells adjacent to each open boundary:

       ∂φ/∂t|_sponge = −γ(n) × (φ − φ_ext)             [s⁻¹]

   with a quadratic decay of the relaxation rate away from the boundary:

       γ(n) = (1/τ_sponge) × max(0, 1 − n/L_sponge)²   (Modelling guide)

   where n is the distance in grid cells from the open boundary and τ_sponge
   is the relaxation time at the boundary face (typically 1–10 days).

   Returned as a ModelState tendency, compatible with the AB2 timestepper.

Boundary naming convention
--------------------------
    'west'  → i = 0 face   (u normal, outward direction = −x)
    'east'  → i = nx face  (u normal, outward direction = +x)
    'south' → j = 0 face   (v normal, outward direction = −y)
    'north' → j = ny face  (v normal, outward direction = +y)

Data structure
--------------
Prescribed exterior values at each open boundary are passed as an
``OBData`` dataclass.  Fields not needed for a given boundary may be None.

Supported fields for nudging / prescribed inflow:
    T   shape (nz, n_bdy)    potential temperature [°C]
    S   shape (nz, n_bdy)    salinity [PSU]
    u_normal  shape (nz, n_bdy)  normal velocity [m/s]:
                              u-component for west/east,
                              v-component for south/north.
    eta shape (n_bdy,)       sea surface height [m]

where n_bdy is the number of grid points along the boundary:
    west/east  → n_bdy = ny   (j = 0..ny−1)
    south/north → n_bdy = nx  (i = 0..nx−1)

Phase 1 usage
-------------
For the rectangular bay seiche validation target (CLAUDE.md §Validation),
all boundaries are solid walls (no OBCS needed).  These functions return
zero tendencies or leave state unchanged when no open boundaries are
configured, so they integrate cleanly into the Phase 1 timeloop.

Phase 2+ usage
--------------
Open boundaries with prescribed T, S, u, v from HYCOM operational output
for the Gulf of Mexico subdomain.  Orlanski radiation handles outgoing
mesoscale eddies and tidal harmonics.

References
----------
Orlanski, I. (1976).
    A simple boundary condition for unbounded hyperbolic flows.
    J. Comput. Phys. 21, 251–269.

Marshall, J., Adcroft, A., Hill, C., Perelman, L. & Heisey, C. (1997).
    A finite-volume, incompressible Navier–Stokes model for studies of the
    ocean on parallel computers. J. Geophys. Res. 102(C3), 5753–5766.

MITgcm Team (2023). MITgcm manual: OBCS package.
    Source: MITgcm/pkg/obcs/obcs_calc.F, obcs_apply.F.

HYCOM Consortium. HYCOM source: HYCOM-src/latbdy.F90.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np
import xarray as xr

from core.state import ModelState

# ---------------------------------------------------------------------------
# Small constant for safe division
# ---------------------------------------------------------------------------
_EPS: float = 1.0e-10


# ---------------------------------------------------------------------------
# Data structure for prescribed boundary values
# ---------------------------------------------------------------------------

@dataclass
class OBData:
    """
    Prescribed exterior values for one open boundary.

    Parameters
    ----------
    side : {'west', 'east', 'south', 'north'}
        Which boundary this data applies to.
    T : np.ndarray (nz, n_bdy) or None
        Prescribed potential temperature [°C].
    S : np.ndarray (nz, n_bdy) or None
        Prescribed salinity [PSU].
    u_normal : np.ndarray (nz, n_bdy) or None
        Prescribed normal velocity [m/s].
        u-component for west/east, v-component for south/north.
    eta : np.ndarray (n_bdy,) or None
        Prescribed sea surface height [m].
    zw : np.ndarray (nz+1, n_bdy) or None
        Interface heights of the data's own vertical grid [m], surface first, decreasing (needed by the ALE/σ
        model with a time-varying exterior; spec 2026-10-12-baroclinic-obc-internal-wave §1.1).
    ubar : np.ndarray (n_bdy,) or None
        Exterior depth-mean normal velocity for Flather [m/s]; None = 0.
    tracers : np.ndarray (ntr, nz, n_bdy) or None
        Passive tracers on the same ``zw`` grid (ALE ``n_tracers``); None = 0 in the exterior.
    transport : np.ndarray (n_bdy,) or None
        Prescribed normal volume flux per face [m³ s⁻¹], model sign; replaces Flather there (MOM6 SIMPLE segment): the
        free surface takes exactly this flux, with no radiation term.
    u_tangential : np.ndarray (nz, n_bdy + 1) or None
        Exterior velocity **parallel** to the edge [m/s] on the ``zw`` grid, at the n_bdy + 1 corners of the edge
        line (v for west/east, u for south/north, grid components): the nudging target of MOM6 NUDGED_TAN
        (``obc_tangential='mom6'``, docs/design/regional_refs/mom6_obc_tangential.md).  None = 0 (resting exterior).
    """
    side: Literal["west", "east", "south", "north"]
    T: np.ndarray | None = field(default=None)
    S: np.ndarray | None = field(default=None)
    u_normal: np.ndarray | None = field(default=None)
    eta: np.ndarray | None = field(default=None)
    zw: np.ndarray | None = field(default=None)
    ubar: np.ndarray | None = field(default=None)
    tracers: np.ndarray | None = field(default=None)
    transport: np.ndarray | None = field(default=None)
    u_tangential: np.ndarray | None = field(default=None)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _sponge_mask_1d(n_total: int, sponge_width: int) -> np.ndarray:
    """
    Build a 1-D sponge coefficient profile γ(n) ∈ [0, 1] for one boundary.

    γ(n) = max(0, 1 − n / L)²   where L = sponge_width.

    Returns shape (n_total,): 1 at the boundary (n=0), 0 beyond L.
    """
    n = np.arange(n_total, dtype=float)
    return np.maximum(0.0, 1.0 - n / max(sponge_width, 1)) ** 2


def _orlanski_phase_speed(
    phi_bdy: np.ndarray,
    phi_int: np.ndarray,
    phi_int_prev: np.ndarray,
    dt: float,
    dn: float,
    sign: int = 1,
) -> np.ndarray:
    """
    Estimate outward phase speed for Orlanski radiation.

    For outward-pointing boundaries (west: sign=1, south: sign=1):
        c_n = max(0, −(φ_int − φ_int_prev) / (dt × (φ_int − φ_bdy) / dn))

    For inward-pointing (east, north): sign = −1, numerator is reversed.

    Clipped to [0, dn/dt] for causality.
    """
    c_max = dn / dt
    Dt_phi = phi_int - phi_int_prev          # Δt at interior
    Dn_phi = phi_int - phi_bdy               # normal gradient (interior − bdy)
    # For east/north: wave moves in +direction, reverse sign convention
    c_est = -sign * Dt_phi / (dt * Dn_phi / dn + np.where(np.abs(Dn_phi) > _EPS, 0.0, _EPS))
    c_est = np.where(np.abs(Dn_phi) > _EPS, c_est, 0.0)
    return np.clip(c_est, 0.0, c_max)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def sponge_tendency(
    state: ModelState,
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
    boundaries: list[OBData],
    sponge_width: int = 5,
    tau_sponge: float = 86400.0,
) -> ModelState:
    """
    Compute the sponge-layer nudging tendency toward prescribed boundary values.

    For each configured open boundary, adds a relaxation tendency in a
    buffer zone of width ``sponge_width`` grid cells:

        ∂φ/∂t|_sponge [n] = −γ(n) / τ_sponge × (φ[n] − φ_ext)

    where

        γ(n) = max(0, 1 − n / L_sponge)²                 [dimensionless]

    decays from 1 at the boundary face to 0 at n = sponge_width.

    Supported fields: T, S (3-D at T-cell centres), and η (2-D).
    Velocity sponge is not applied (avoid spurious damping at walls).

    Parameters
    ----------
    state : ModelState
        Current model state.
    h_grid : xr.Dataset
        Horizontal grid (provides attrs ``nx``, ``ny``).
    v_grid : xr.Dataset
        Vertical grid (provides attr ``nz``).
    boundaries : list[OBData]
        Open boundary specifications.  Empty list → zero tendency.
    sponge_width : int, optional
        Sponge buffer width [grid cells].  Default 5.
    tau_sponge : float, optional
        Relaxation timescale [s] at the boundary face.  Default 86400 (1 day).

    Returns
    -------
    ModelState
        Nudging tendency dstate/dt [per second].
        T → [°C/s]; S → [PSU/s]; η → [m/s]; u, v, w → 0.
    """
    nx = int(h_grid.attrs["nx"])
    ny = int(h_grid.attrs["ny"])
    nz = int(v_grid.attrs["nz"])

    dT_dt   = np.zeros((nz, ny, nx))
    dS_dt   = np.zeros((nz, ny, nx))
    deta_dt = np.zeros((ny, nx))
    inv_tau = 1.0 / max(tau_sponge, _EPS)

    for obd in boundaries:
        side = obd.side

        if side in ("west", "east"):
            gamma = _sponge_mask_1d(nx, sponge_width)   # (nx,)
            idx_range = range(min(sponge_width, nx))
            # west: cells 0, 1, …; east: cells nx-1, nx-2, …
            cell_of = (lambda ii: ii) if side == "west" else (lambda ii: nx - 1 - ii)

            for ii in idx_range:
                i = cell_of(ii)
                g = gamma[ii] * inv_tau   # s⁻¹
                if obd.T is not None:
                    dT_dt[:, :, i] = -g * (state.T.values[:, :, i] - obd.T)
                if obd.S is not None:
                    dS_dt[:, :, i] = -g * (state.S.values[:, :, i] - obd.S)
                if obd.eta is not None:
                    deta_dt[:, i] = -g * (state.eta.values[:, i] - obd.eta)

        elif side in ("south", "north"):
            gamma = _sponge_mask_1d(ny, sponge_width)   # (ny,)
            idx_range = range(min(sponge_width, ny))
            cell_of = (lambda jj: jj) if side == "south" else (lambda jj: ny - 1 - jj)

            for jj in idx_range:
                j = cell_of(jj)
                g = gamma[jj] * inv_tau
                if obd.T is not None:
                    dT_dt[:, j, :] = -g * (state.T.values[:, j, :] - obd.T)
                if obd.S is not None:
                    dS_dt[:, j, :] = -g * (state.S.values[:, j, :] - obd.S)
                if obd.eta is not None:
                    deta_dt[j, :] = -g * (state.eta.values[j, :] - obd.eta)

    return ModelState(
        u=xr.zeros_like(state.u),
        v=xr.zeros_like(state.v),
        w=xr.zeros_like(state.w),
        T=xr.DataArray(dT_dt, dims=state.T.dims, coords=state.T.coords,
                       attrs={"units": "degC/s", "long_name": "Sponge dT/dt"}),
        S=xr.DataArray(dS_dt, dims=state.S.dims, coords=state.S.coords,
                       attrs={"units": "PSU/s", "long_name": "Sponge dS/dt"}),
        eta=xr.DataArray(deta_dt, dims=state.eta.dims, coords=state.eta.coords,
                         attrs={"units": "m/s", "long_name": "Sponge deta/dt"}),
        time=0.0,
    )


def apply_orlanski(
    state: ModelState,
    state_prev: ModelState,
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
    dt: float,
    boundaries: list[OBData],
) -> ModelState:
    """
    Update open boundary values using the Orlanski (1976) radiation condition.

    For each configured open boundary and each field (T, S, η, u_normal),
    the outward phase speed c_n is estimated from the interior time tendency
    and normal gradient (Orlanski 1976, eq. 3; MITgcm obcs_calc.F):

        c_n = max(0, −Δt_φ_int / (dt × Δn_φ_int / dn))       outgoing

    then the boundary value is updated:

        φ_bdy = φ_bdy + c_n × (dt/dn) × (φ_int − φ_bdy)      outgoing
        φ_bdy = φ_ext                                           incoming (c_n≤0)

    If no exterior value is prescribed, pure radiation (zero-gradient
    extrapolation) is used for outgoing; incoming characteristic is
    extrapolated from the interior.

    This is a **direct state update** (not a tendency).  Call *after* the
    AB2 timestepper, passing the two most recent states.

    Parameters
    ----------
    state : ModelState
        Current state (after the AB2 step).  Boundary values overwritten.
    state_prev : ModelState
        Previous state (before the AB2 step).
    h_grid : xr.Dataset
        Horizontal grid (provides attrs ``nx``, ``ny``, ``dx``, ``dy``).
    v_grid : xr.Dataset
        Vertical grid (provides attr ``nz``).
    dt : float
        Timestep [s].
    boundaries : list[OBData]
        Open boundary specifications.  Empty list → state unchanged.

    Returns
    -------
    ModelState
        Copy of ``state`` with boundary values updated.
    """
    if not boundaries:
        return state.copy()

    dx = float(h_grid.attrs["dx"])
    dy = float(h_grid.attrs["dy"])

    T_new   = state.T.values.copy()
    S_new   = state.S.values.copy()
    eta_new = state.eta.values.copy()
    u_new   = state.u.values.copy()
    v_new   = state.v.values.copy()

    T_p   = state_prev.T.values
    S_p   = state_prev.S.values
    eta_p = state_prev.eta.values
    u_p   = state_prev.u.values
    v_p   = state_prev.v.values

    for obd in boundaries:
        side = obd.side

        # ------------------------------------------------------------------ west
        if side == "west":
            dn = dx
            c_max = dn / dt

            def _rad_w(phi, phi_prev, phi_ext):
                # phi: (nz, ny, nx) or (ny, nx)
                # Orlanski: c_n = Dt*dn / (dt*Dn), Dn = interior − bdy
                bdy = phi[..., 0]
                interior = phi[..., 1]
                int_prev = phi_prev[..., 1]
                Dt = interior - int_prev   # time change at interior
                Dn = interior - bdy        # normal gradient (outward n = −x, so Dn>0 means bdy<int)
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                outgoing = c > 0.0
                bdy_rad = bdy + c * (dt / dn) * (interior - bdy)
                if phi_ext is not None:
                    return np.where(outgoing, bdy_rad, phi_ext)
                return np.where(outgoing, bdy_rad, interior)

            if obd.T is not None:
                T_new[..., 0] = _rad_w(T_new, T_p, obd.T)
            if obd.S is not None:
                S_new[..., 0] = _rad_w(S_new, S_p, obd.S)
            if obd.eta is not None:
                eta_new[:, 0] = _rad_w(eta_new, eta_p, obd.eta)
            if obd.u_normal is not None:
                # West u-face at i_u=0
                bdy = u_new[:, :, 0]
                interior = u_new[:, :, 1]
                int_prev = u_p[:, :, 1]
                Dt = interior - int_prev
                Dn = interior - bdy
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                u_new[:, :, 0] = np.where(c > 0,
                    bdy + c * (dt / dn) * (interior - bdy),
                    obd.u_normal)

        # ------------------------------------------------------------------ east
        elif side == "east":
            dn = dx
            c_max = dn / dt

            def _rad_e(phi, phi_prev, phi_ext):
                # Same formula as west: c = Dt*dn/(dt*Dn), Dn = interior − bdy
                bdy = phi[..., -1]
                interior = phi[..., -2]
                int_prev = phi_prev[..., -2]
                Dt = interior - int_prev
                Dn = interior - bdy
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                outgoing = c > 0.0
                bdy_rad = bdy + c * (dt / dn) * (interior - bdy)
                if phi_ext is not None:
                    return np.where(outgoing, bdy_rad, phi_ext)
                return np.where(outgoing, bdy_rad, interior)

            if obd.T is not None:
                T_new[..., -1] = _rad_e(T_new, T_p, obd.T)
            if obd.S is not None:
                S_new[..., -1] = _rad_e(S_new, S_p, obd.S)
            if obd.eta is not None:
                eta_new[:, -1] = _rad_e(eta_new, eta_p, obd.eta)
            if obd.u_normal is not None:
                bdy = u_new[:, :, -1]
                interior = u_new[:, :, -2]
                int_prev = u_p[:, :, -2]
                Dt = interior - int_prev
                Dn = interior - bdy
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                u_new[:, :, -1] = np.where(c > 0,
                    bdy + c * (dt / dn) * (interior - bdy),
                    obd.u_normal)

        # ------------------------------------------------------------------ south
        elif side == "south":
            dn = dy
            c_max = dn / dt

            def _rad_s(phi, phi_prev, phi_ext):
                # c = Dt*dn/(dt*Dn), Dn = interior − bdy
                bdy = phi[..., 0, :]    # (..., nx)
                interior = phi[..., 1, :]
                int_prev = phi_prev[..., 1, :]
                Dt = interior - int_prev
                Dn = interior - bdy
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                outgoing = c > 0.0
                bdy_rad = bdy + c * (dt / dn) * (interior - bdy)
                if phi_ext is not None:
                    return np.where(outgoing, bdy_rad, phi_ext)
                return np.where(outgoing, bdy_rad, interior)

            if obd.T is not None:
                T_new[:, 0, :] = _rad_s(T_new, T_p, obd.T)
            if obd.S is not None:
                S_new[:, 0, :] = _rad_s(S_new, S_p, obd.S)
            if obd.eta is not None:
                eta_new[0, :] = _rad_s(eta_new, eta_p, obd.eta)
            if obd.u_normal is not None:
                bdy = v_new[:, 0, :]
                interior = v_new[:, 1, :]
                int_prev = v_p[:, 1, :]
                Dt = interior - int_prev
                Dn = interior - bdy
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                v_new[:, 0, :] = np.where(c > 0,
                    bdy + c * (dt / dn) * (interior - bdy),
                    obd.u_normal)

        # ------------------------------------------------------------------ north
        elif side == "north":
            dn = dy
            c_max = dn / dt

            def _rad_n(phi, phi_prev, phi_ext):
                # c = Dt*dn/(dt*Dn), Dn = interior − bdy
                bdy = phi[..., -1, :]
                interior = phi[..., -2, :]
                int_prev = phi_prev[..., -2, :]
                Dt = interior - int_prev
                Dn = interior - bdy
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                outgoing = c > 0.0
                bdy_rad = bdy + c * (dt / dn) * (interior - bdy)
                if phi_ext is not None:
                    return np.where(outgoing, bdy_rad, phi_ext)
                return np.where(outgoing, bdy_rad, interior)

            if obd.T is not None:
                T_new[:, -1, :] = _rad_n(T_new, T_p, obd.T)
            if obd.S is not None:
                S_new[:, -1, :] = _rad_n(S_new, S_p, obd.S)
            if obd.eta is not None:
                eta_new[-1, :] = _rad_n(eta_new, eta_p, obd.eta)
            if obd.u_normal is not None:
                bdy = v_new[:, -1, :]
                interior = v_new[:, -2, :]
                int_prev = v_p[:, -2, :]
                Dt = interior - int_prev
                Dn = interior - bdy
                c = np.where(np.abs(Dn) > _EPS,
                             np.clip(Dt * dn / (dt * np.where(np.abs(Dn) > _EPS, Dn, _EPS)), 0.0, c_max), 0.0)
                v_new[:, -1, :] = np.where(c > 0,
                    bdy + c * (dt / dn) * (interior - bdy),
                    obd.u_normal)

    return ModelState(
        u=xr.DataArray(u_new, dims=state.u.dims, coords=state.u.coords,
                       attrs=state.u.attrs),
        v=xr.DataArray(v_new, dims=state.v.dims, coords=state.v.coords,
                       attrs=state.v.attrs),
        w=state.w.copy(deep=True),
        T=xr.DataArray(T_new, dims=state.T.dims, coords=state.T.coords,
                       attrs=state.T.attrs),
        S=xr.DataArray(S_new, dims=state.S.dims, coords=state.S.coords,
                       attrs=state.S.attrs),
        eta=xr.DataArray(eta_new, dims=state.eta.dims, coords=state.eta.coords,
                         attrs=state.eta.attrs),
        time=state.time,
    )
