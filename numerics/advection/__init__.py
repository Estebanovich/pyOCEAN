"""
Tracer advection schemes for pyOCEAN.

Phase 1: first-order upwind (donor-cell) scheme.
Phase 2 upgrade: piecewise-parabolic method (PPM) for lower numerical diffusion.

Tracer conservation equation
----------------------------
For a passive tracer τ (potential temperature T or salinity S) in a
Boussinesq fluid with ∇·u = 0:

    ∂τ/∂t + ∇·(u τ) = 0                           (flux / conservative form)

Discretised on the Arakawa C-grid in flux form:

    ∂τ/∂t [k,j,i] = −(F_x[k,j,i+1] − F_x[k,j,i]) / Δx
                    −(F_y[k,j+1,i]  − F_y[k,j,i])  / Δy
                    −(F_z[k_f=k,j,i] − F_z[k_f=k+1,j,i]) / Δz[k]

where the face fluxes are computed by the upwind rule:

    F_x[k,j,i_u] = u[k,j,i_u] · τ_upwind_x

and analogously for y and z.

First-order upwind (donor-cell) rule
-------------------------------------
The upwind value is chosen from the cell on the upstream side of each face:

    τ_upwind_x[k,j,i_u] = τ[k,j,i_u−1]  if u ≥ 0   (west T-cell)
                          = τ[k,j,i_u  ]  if u < 0   (east T-cell)

    τ_upwind_y[k,j_v,i] = τ[k,j_v−1,i]  if v ≥ 0   (south T-cell)
                         = τ[k,j_v,  i]  if v < 0   (north T-cell)

    τ_upwind_z[k_f,j,i] = τ[k_f,  j,i]  if w ≥ 0   (layer below interface)
                         = τ[k_f−1,j,i]  if w < 0   (layer above interface)

The upwind scheme is first-order accurate in space, unconditionally monotone
(no new extrema created), and mass-conservative in flux form.
Its main drawback is significant numerical diffusion ∝ (u Δx / 2) |∂²τ/∂x²|.
PPM (Colella & Woodward 1984) will replace it in Phase 2 to reduce diffusion
while retaining monotonicity.

Boundary conditions
-------------------
*  Horizontal walls: u = 0 at east/west faces (i_u = 0, nx) and v = 0 at
   north/south faces (j_v = 0, ny).  The flux is zero at these faces by
   construction (flux = u · τ = 0).
*  Surface / bottom: w = 0 at k_f = 0 and k_f = nz.  Vertical flux = 0.

hFac (partial cells)
---------------------
Phase 1 uses a flat-bottom, full-cell grid (hFac = 1 everywhere).  The
hFac weighting needed for partial-cell topography is deferred to Phase 2.

References
----------
Durran (2010) — Numerical Methods for Fluid Dynamics.  §3.4 (upwind),
    §3.5.3 (PPM).
Colella & Woodward (1984) — The piecewise-parabolic method (PPM) for
    gas-dynamical simulations.  J. Comput. Phys. 54(1), 174–201.
Smolarkiewicz & Margolin (1998) — MPDATA: A finite-difference solver for
    geophysical flows.  J. Comput. Phys. 140(2), 459–480.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.state import ModelState


# ---------------------------------------------------------------------------
# 1. Face-flux computation  (upwind)
# ---------------------------------------------------------------------------

def upwind_flux_x(
    tracer: xr.DataArray,
    u: xr.DataArray,
) -> xr.DataArray:
    """
    Zonal (x) upwind tracer flux at U-faces.

    Computes F_x = u · τ_upwind at every U-face using the donor-cell rule:

        F_x[k, j, i_u] = u[k,j,i_u] · τ[k,j,i_u−1]   if u ≥ 0
                        = u[k,j,i_u] · τ[k,j,i_u  ]   if u < 0

    Units: [tracer units · m/s]  e.g. °C·m/s for temperature.

    Parameters
    ----------
    tracer : xr.DataArray, dims (k, j_t, i_t)
        Tracer field τ [tracer units] at T-cell centres.
    u : xr.DataArray, dims (k, j_t, i_u)
        Zonal velocity [m/s] at U-faces.

    Returns
    -------
    xr.DataArray, dims (k, j_t, i_u)
        Zonal tracer flux [tracer units · m/s].
        Boundary faces i_u = 0 (west) and i_u = nx (east) are 0 (u = 0 there).

    Notes
    -----
    The flux at boundary U-faces is zero because u = 0 at solid walls.
    The upwind interpolation for interior faces i_u = 1 … nx−1 uses
    τ[k,j,i_u−1] (west T-cell) when u ≥ 0 and τ[k,j,i_u] (east T-cell)
    when u < 0.
    """
    τ_np = tracer.values     # (nz, ny, nx)
    u_np = u.values          # (nz, ny, nx+1)

    nz, ny, nx = τ_np.shape

    # Interior U-faces: i_u = 1 … nx-1
    u_int = u_np[:, :, 1:nx]         # (nz, ny, nx-1)  interior u
    τ_west = τ_np[:, :, 0:nx - 1]   # (nz, ny, nx-1)  upwind for u ≥ 0
    τ_east = τ_np[:, :, 1:nx]       # (nz, ny, nx-1)  upwind for u < 0

    flux_int = u_int * np.where(u_int >= 0.0, τ_west, τ_east)

    # Full flux array (nz, ny, nx+1); boundary faces stay 0
    flux = np.zeros((nz, ny, nx + 1))
    flux[:, :, 1:nx] = flux_int

    return xr.DataArray(
        flux,
        dims=["k", "j_t", "i_u"],
        coords={"k": tracer.coords["k"],
                "j_t": tracer.coords["j_t"],
                "i_u": u.coords["i_u"]},
        attrs={"units": f"{tracer.attrs.get('units','?')} m/s",
               "long_name": "Upwind zonal tracer flux"},
    )


def upwind_flux_y(
    tracer: xr.DataArray,
    v: xr.DataArray,
) -> xr.DataArray:
    """
    Meridional (y) upwind tracer flux at V-faces.

    Donor-cell rule:

        F_y[k, j_v, i] = v[k,j_v,i] · τ[k,j_v−1,i]   if v ≥ 0
                        = v[k,j_v,i] · τ[k,j_v,  i]   if v < 0

    Parameters
    ----------
    tracer : xr.DataArray, dims (k, j_t, i_t)
        Tracer field τ at T-cell centres.
    v : xr.DataArray, dims (k, j_v, i_t)
        Meridional velocity [m/s] at V-faces.

    Returns
    -------
    xr.DataArray, dims (k, j_v, i_t)
        Meridional tracer flux [tracer units · m/s].
        Boundary faces j_v = 0 (south) and j_v = ny (north) are 0.
    """
    τ_np = tracer.values     # (nz, ny, nx)
    v_np = v.values          # (nz, ny+1, nx)

    nz, ny, nx = τ_np.shape

    # Interior V-faces: j_v = 1 … ny-1
    v_int   = v_np[:, 1:ny, :]         # (nz, ny-1, nx)
    τ_south = τ_np[:, 0:ny - 1, :]    # upwind for v ≥ 0
    τ_north = τ_np[:, 1:ny, :]        # upwind for v < 0

    flux_int = v_int * np.where(v_int >= 0.0, τ_south, τ_north)

    flux = np.zeros((nz, ny + 1, nx))
    flux[:, 1:ny, :] = flux_int

    return xr.DataArray(
        flux,
        dims=["k", "j_v", "i_t"],
        coords={"k": tracer.coords["k"],
                "j_v": v.coords["j_v"],
                "i_t": tracer.coords["i_t"]},
        attrs={"units": f"{tracer.attrs.get('units','?')} m/s",
               "long_name": "Upwind meridional tracer flux"},
    )


def upwind_flux_z(
    tracer: xr.DataArray,
    w: xr.DataArray,
) -> xr.DataArray:
    """
    Vertical (z) upwind tracer flux at layer interfaces (k_f faces).

    Donor-cell rule at interface k_f (between layer k_f−1 above and k_f below):

        F_z[k_f, j, i] = w[k_f,j,i] · τ[k_f,   j, i]   if w ≥ 0  (upward; upwind = below)
                        = w[k_f,j,i] · τ[k_f−1, j, i]   if w < 0  (downward; upwind = above)

    Sign convention: w > 0 is upward (toward smaller k_f index, toward surface).

    Parameters
    ----------
    tracer : xr.DataArray, dims (k, j_t, i_t)
        Tracer field τ [tracer units] at layer centres.
    w : xr.DataArray, dims (k_f, j_t, i_t)
        Vertical velocity [m/s] at layer interfaces.
        w[k_f=0] = 0 (surface), w[k_f=nz] = 0 (bottom) are enforced.

    Returns
    -------
    xr.DataArray, dims (k_f, j_t, i_t)
        Vertical tracer flux [tracer units · m/s] at layer interfaces.
        Surface (k_f=0) and bottom (k_f=nz) fluxes are 0.

    Notes
    -----
    The upwind assignment is based on the sign of w at each interface:
      - w > 0 (upward): flow carries water from the layer below → upwind = τ[k_f]
      - w < 0 (downward): flow carries water from the layer above → upwind = τ[k_f−1]
    where "layer below interface k_f" has index k = k_f (0-based; k increases downward).
    """
    τ_np = tracer.values     # (nz, ny, nx)
    w_np = w.values          # (nz+1, ny, nx)

    nz = τ_np.shape[0]

    # Interior interfaces: k_f = 1 … nz-1
    w_int     = w_np[1:nz, :, :]       # (nz-1, ny, nx)
    τ_above   = τ_np[0:nz - 1, :, :]  # τ[k_f-1]: layer above interface k_f
    τ_below   = τ_np[1:nz, :, :]      # τ[k_f]:   layer below interface k_f

    # w > 0 upward → upstream = τ_below; w < 0 downward → upstream = τ_above
    flux_int = w_int * np.where(w_int >= 0.0, τ_below, τ_above)

    flux = np.zeros_like(w_np)     # (nz+1, ny, nx); BC: flux[0]=flux[nz]=0
    flux[1:nz, :, :] = flux_int

    return xr.DataArray(
        flux,
        dims=["k_f", "j_t", "i_t"],
        coords={"k_f": w.coords["k_f"],
                "j_t": tracer.coords["j_t"],
                "i_t": tracer.coords["i_t"]},
        attrs={"units": f"{tracer.attrs.get('units','?')} m/s",
               "long_name": "Upwind vertical tracer flux"},
    )


# ---------------------------------------------------------------------------
# 2. Advective tendency  (flux divergence)
# ---------------------------------------------------------------------------

def advect_tracer(
    tracer: xr.DataArray,
    u: xr.DataArray,
    v: xr.DataArray,
    w: xr.DataArray,
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
) -> xr.DataArray:
    """
    Tracer advective tendency using the first-order upwind (donor-cell) scheme.

    Computes the conservative flux-divergence form:

        ∂τ/∂t|_adv = −∂(uτ)/∂x − ∂(vτ)/∂y − ∂(wτ)/∂z
                   = −(F_x[k,j,i+1]−F_x[k,j,i])/Δx
                    −(F_y[k,j+1,i] −F_y[k,j,i])/Δy
                    −(F_z[k_f=k,j,i]−F_z[k_f=k+1,j,i])/Δz[k]

                                                (Durran 2010, §3.4, eq. 3.63)

    Parameters
    ----------
    tracer : xr.DataArray, dims (k, j_t, i_t)
        Tracer field τ at T-cell centres [tracer units].
    u : xr.DataArray, dims (k, j_t, i_u)
        Zonal velocity [m/s].
    v : xr.DataArray, dims (k, j_v, i_t)
        Meridional velocity [m/s].
    w : xr.DataArray, dims (k_f, j_t, i_t)
        Vertical velocity [m/s] at layer interfaces.
    h_grid : xr.Dataset
        Horizontal grid providing ``dx_t`` [m] and ``dy_t`` [m].
    v_grid : xr.Dataset
        Vertical grid providing ``dz`` [m].

    Returns
    -------
    xr.DataArray, dims (k, j_t, i_t)
        Advective tendency ∂τ/∂t [tracer units / s].

    Notes
    -----
    *  The scheme is mass-conservative and monotone (no spurious extrema for
       non-negative tracers) but introduces first-order numerical diffusion
       ∝ |u| Δx / 2 (Durran 2010, §3.4.1).
    *  The flux divergence is evaluated cell-by-cell in the standard
       finite-volume sense; no explicit use of ``hFac`` (Phase 1 flat bottom).
    *  The returned tendency can be added directly to a ``ModelState``
       tendency field in the time stepper.

    Examples
    --------
    >>> dT_dt = advect_tracer(state.T, state.u, state.v, state.w, hgrid, vgrid)
    >>> dS_dt = advect_tracer(state.S, state.u, state.v, state.w, hgrid, vgrid)
    """
    dx_t = h_grid["dx_t"].values    # (ny, nx)
    dy_t = h_grid["dy_t"].values    # (ny, nx)
    dz   = v_grid["dz"].values      # (nz,)

    nz = dz.shape[0]

    # ---- zonal flux divergence  -------------------------------------------
    F_x  = upwind_flux_x(tracer, u).values         # (nz, ny, nx+1)
    nx   = F_x.shape[2] - 1
    div_x = (F_x[:, :, 1:nx + 1] - F_x[:, :, 0:nx]) / dx_t[np.newaxis, :, :]

    # ---- meridional flux divergence  --------------------------------------
    F_y  = upwind_flux_y(tracer, v).values         # (nz, ny+1, nx)
    ny   = F_y.shape[1] - 1
    div_y = (F_y[:, 1:ny + 1, :] - F_y[:, 0:ny, :]) / dy_t[np.newaxis, :, :]

    # ---- vertical flux divergence  ----------------------------------------
    # ∂(wτ)/∂z = (F_z[k_f=k] − F_z[k_f=k+1]) / dz[k]
    # z increases upward; k_f=k is the TOP (higher z) of layer k.
    F_z   = upwind_flux_z(tracer, w).values        # (nz+1, ny, nx)
    div_z = (F_z[0:nz, :, :] - F_z[1:nz + 1, :, :]) / dz[:, np.newaxis, np.newaxis]

    # ---- total advective tendency  ----------------------------------------
    dτ_dt = -(div_x + div_y + div_z)

    return xr.DataArray(
        dτ_dt,
        dims=tracer.dims,
        coords=tracer.coords,
        attrs={
            "units": f"{tracer.attrs.get('units','?')} s-1",
            "long_name": f"Advective tendency of {tracer.attrs.get('long_name', 'tracer')}",
            "scheme": "first-order upwind (donor-cell)",
        },
    )


# ---------------------------------------------------------------------------
# 3. Convenience wrapper: full-state tracer tendency
# ---------------------------------------------------------------------------

def tracer_tendency(
    state: ModelState,
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
) -> ModelState:
    """
    Compute the advective tendency for all tracers (T and S) in a ModelState.

    Evaluates the upwind advection scheme for both potential temperature T
    and salinity S.  Momentum (u, v, w) and sea-surface-height (η) tendencies
    in the returned state are zero — call ``momentum_tendency`` separately.

    Parameters
    ----------
    state : ModelState
        Current model state (u, v, w, T, S, η).
    h_grid : xr.Dataset
        Horizontal grid providing metric terms.
    v_grid : xr.Dataset
        Vertical grid providing layer thicknesses.

    Returns
    -------
    ModelState
        Tendency dstate/dt.  Units: T → [°C/s]; S → [PSU/s];
        u, v, w, η → 0.

    Examples
    --------
    >>> adv_tend = tracer_tendency(state, hgrid, vgrid)
    >>> # Combine with momentum tendency in the time stepper:
    >>> total_tend = mom_tend + adv_tend   # ModelState addition
    """
    dT_dt = advect_tracer(state.T, state.u, state.v, state.w, h_grid, v_grid)
    dS_dt = advect_tracer(state.S, state.u, state.v, state.w, h_grid, v_grid)

    return ModelState(
        u=xr.zeros_like(state.u),
        v=xr.zeros_like(state.v),
        w=xr.zeros_like(state.w),
        T=dT_dt,
        S=dS_dt,
        eta=xr.zeros_like(state.eta),
        time=0.0,
    )
