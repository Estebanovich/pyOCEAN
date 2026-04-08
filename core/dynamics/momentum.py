"""
Momentum equations for pyOCEAN — Phase 1.

Implements the horizontal momentum tendencies and related diagnostics for the
hydrostatic Boussinesq primitive equations on an Arakawa C-grid:

    ∂u/∂t = +f v  − (1/ρ₀) ∂p/∂x  + ν_h ∇²u      (Vallis 2006, eq. 14.12a)
    ∂v/∂t = −f u  − (1/ρ₀) ∂p/∂y  + ν_h ∇²v      (Vallis 2006, eq. 14.12b)
    ∂η/∂t = −∇·(H ū)                               (free-surface continuity)

Phase 1 terms included
----------------------
* Coriolis acceleration (f-plane and beta-plane via ``h_grid.f_q``).
* Pressure gradient force (pre-computed by ``core.dynamics.pressure``).
* Horizontal Laplacian viscosity ν_h ∇²u (optional; default ν_h = 0).
* Free-surface tendency from depth-integrated divergence.
* Diagnostic vertical velocity w from continuity.

Not included in Phase 1
-----------------------
* Horizontal momentum advection (``numerics.advection``, Phase 1 module 7).
* Vertical mixing (``physics.mixing.kpp``, Phase 1 module 8).
* Lateral boundary forcing (``boundaries.obcs``, Phase 1 module 9).

C-grid Coriolis interpolation
------------------------------
On the Arakawa C-grid the velocity components are staggered:
    u lives at U-faces (j_t, i_u)
    v lives at V-faces (j_v, i_t)
    f lives at q-corners (j_v, i_u)  ← stored in h_grid["f_q"]

For the zonal momentum at U-face (j, i_u), the term +f·v requires v
interpolated to the U-face (4-point arithmetic average of surrounding V-faces)
and f interpolated to the U-face (2-point average of bounding q-corners):

    v_at_u[k, j, i_u] = ¼ (v[k, j,   i_u-1] + v[k, j+1, i_u-1]
                           + v[k, j,   i_u  ] + v[k, j+1, i_u  ])
    f_at_u[j, i_u]    = ½ (f_q[j, i_u] + f_q[j+1, i_u])

And analogously for the meridional momentum (−f·u at V-faces).

Free-surface tendency
---------------------
From the barotropic continuity equation (Marshall et al. 1997, eq. 6):

    ∂η/∂t [j, i] = − Σ_k dz[k] · [  (u[k,j,i+1] − u[k,j,i]) / Δx
                                     + (v[k,j+1,i] − v[k,j,i]) / Δy ]

where the sum over k gives the depth-integrated horizontal divergence.

Diagnostic vertical velocity
-----------------------------
w is not time-stepped; it is diagnosed from the discrete continuity equation
integrated downward from the free surface:

    w[k_f+1] = w[k_f] + dz[k] · div2d[k]   (w[0] = 0, surface BC)

References
----------
Vallis (2006) — Atmospheric and Oceanic Fluid Dynamics, §14.1–14.2.
Marshall et al. (1997) — A finite-volume, incompressible Navier–Stokes model.
    J. Geophys. Res. 102(C3), 5753–5766.
Durran (2010) — Numerical Methods for Fluid Dynamics, §4.2.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.state import ModelState

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

RHO_0: float = 1025.0   # kg/m³  Boussinesq reference density


# ---------------------------------------------------------------------------
# Internal helpers — Coriolis interpolation
# ---------------------------------------------------------------------------

def _v_to_u(v_np: np.ndarray) -> np.ndarray:
    """
    Interpolate v (nz, ny+1, nx) to U-face positions (nz, ny, nx+1).

    4-point arithmetic average on the C-grid:

        v_at_u[k, j, i_u] = ¼ (v[k, j,   i_u-1] + v[k, j+1, i_u-1]
                               + v[k, j,   i_u  ] + v[k, j+1, i_u  ])

    Boundary U-faces (i_u=0 west, i_u=nx east) are set to zero because
    u = 0 at solid walls; the Coriolis term there is also forced to zero.
    """
    # v_np shape: (nz, ny+1, nx)  →  j_v = 0..ny, i_t = 0..nx-1
    nz, nyp1, nx = v_np.shape
    ny = nyp1 - 1

    v_at_u = np.zeros((nz, ny, nx + 1))

    # Interior U-faces i_u = 1..nx-1  →  i_t neighbours i_u-1 and i_u
    v_at_u[:, :, 1:nx] = 0.25 * (
        v_np[:, 0:ny,  0:nx - 1]   # south-west V-face
      + v_np[:, 1:ny + 1, 0:nx - 1]   # north-west V-face
      + v_np[:, 0:ny,  1:nx]       # south-east V-face
      + v_np[:, 1:ny + 1, 1:nx]    # north-east V-face
    )
    # Boundary i_u=0 and i_u=nx stay zero (solid-wall u=0).
    return v_at_u


def _u_to_v(u_np: np.ndarray) -> np.ndarray:
    """
    Interpolate u (nz, ny, nx+1) to V-face positions (nz, ny+1, nx).

    4-point arithmetic average on the C-grid:

        u_at_v[k, j_v, i] = ¼ (u[k, j_v-1, i  ] + u[k, j_v, i  ]
                               + u[k, j_v-1, i+1] + u[k, j_v, i+1])

    Boundary V-faces (j_v=0 south, j_v=ny north) are set to zero because
    v = 0 at solid walls.
    """
    # u_np shape: (nz, ny, nx+1)  →  j_t = 0..ny-1, i_u = 0..nx
    nz, ny, nxp1 = u_np.shape
    nx = nxp1 - 1

    u_at_v = np.zeros((nz, ny + 1, nx))

    # Interior V-faces j_v = 1..ny-1  →  j_t neighbours j_v-1 and j_v
    u_at_v[:, 1:ny, :] = 0.25 * (
        u_np[:, 0:ny - 1, 0:nx]    # south-west U-face
      + u_np[:, 1:ny,     0:nx]    # north-west U-face
      + u_np[:, 0:ny - 1, 1:nx + 1]  # south-east U-face
      + u_np[:, 1:ny,     1:nx + 1]  # north-east U-face
    )
    # Boundary j_v=0 and j_v=ny stay zero (solid-wall v=0).
    return u_at_v


# ---------------------------------------------------------------------------
# Internal helpers — Laplacian viscosity
# ---------------------------------------------------------------------------

def _laplacian_u(u_np: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Laplacian ∇²u at U-face positions (nz, ny, nx+1).

    Uniform Cartesian grid assumed (dx, dy scalars).
    Zonal BC: u=0 at east/west walls (already in array).
    Meridional BC: free-slip, ∂u/∂y=0 at north/south walls (edge padding).
    """
    nz, ny, nxp1 = u_np.shape

    # ---- zonal part: uses u at i_u±1 --------------------------------
    # u at i_u=0 and i_u=nx are 0 (walls), so no extra padding needed.
    d2u_dx2 = np.zeros_like(u_np)
    # interior i_u = 1..nx-1 has valid i_u-1 and i_u+1 in the array
    d2u_dx2[:, :, 1:-1] = (
        (u_np[:, :, 2:] - 2.0 * u_np[:, :, 1:-1] + u_np[:, :, :-2]) / dx**2
    )

    # ---- meridional part: free-slip (edge) padding in j ----------------
    u_pad = np.pad(u_np, ((0, 0), (1, 1), (0, 0)), mode="edge")
    d2u_dy2 = (u_pad[:, 2:, :] - 2.0 * u_pad[:, 1:-1, :] + u_pad[:, :-2, :]) / dy**2

    return d2u_dx2 + d2u_dy2


def _laplacian_v(v_np: np.ndarray, dx: float, dy: float) -> np.ndarray:
    """
    Laplacian ∇²v at V-face positions (nz, ny+1, nx).

    Uniform Cartesian grid assumed.
    Meridional BC: v=0 at north/south walls (already in array).
    Zonal BC: free-slip, ∂v/∂x=0 at east/west walls (edge padding).
    """
    nz, nyp1, nx = v_np.shape

    # ---- meridional part: v at j_v=0 and j_v=ny are 0 (walls) --------
    d2v_dy2 = np.zeros_like(v_np)
    d2v_dy2[:, 1:-1, :] = (
        (v_np[:, 2:, :] - 2.0 * v_np[:, 1:-1, :] + v_np[:, :-2, :]) / dy**2
    )

    # ---- zonal part: free-slip (edge) padding in i ---------------------
    v_pad = np.pad(v_np, ((0, 0), (0, 0), (1, 1)), mode="edge")
    d2v_dx2 = (v_pad[:, :, 2:] - 2.0 * v_pad[:, :, 1:-1] + v_pad[:, :, :-2]) / dx**2

    return d2v_dy2 + d2v_dx2


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def coriolis_u(
    v: xr.DataArray,
    f_q: np.ndarray,
) -> xr.DataArray:
    """
    Coriolis acceleration in the zonal momentum equation at U-faces.

    Evaluates the term +f·v at U-face positions using C-grid interpolation:

        du/dt|_Coriolis [k, j, i_u] = f_at_u[j, i_u] · v_at_u[k, j, i_u]

    where

        f_at_u[j, i_u] = ½ (f_q[j, i_u] + f_q[j+1, i_u])      (Vallis 2006 §14.1)
        v_at_u          = 4-point average of surrounding V-faces (see _v_to_u)

    Parameters
    ----------
    v : xr.DataArray, dims (k, j_v, i_t)
        Meridional velocity [m/s].
    f_q : np.ndarray, shape (ny+1, nx+1)
        Coriolis parameter at q-corners [s⁻¹] from ``h_grid["f_q"]``.

    Returns
    -------
    xr.DataArray, dims (k, j_t, i_u)
        Zonal Coriolis acceleration [m/s²].
    """
    v_np = v.values                    # (nz, ny+1, nx)
    nz, nyp1, nx = v_np.shape
    ny = nyp1 - 1

    # f at U-face: average the two bounding q-corners (north and south)
    # f_q shape: (ny+1, nx+1);  f_at_u shape: (ny, nx+1)
    f_at_u = 0.5 * (f_q[0:ny, :] + f_q[1:ny + 1, :])  # (ny, nx+1)

    v_at_u = _v_to_u(v_np)                            # (nz, ny, nx+1)

    cor_u = f_at_u[np.newaxis, :, :] * v_at_u         # (nz, ny, nx+1)

    return xr.DataArray(
        cor_u,
        dims=["k", "j_t", "i_u"],
        coords={
            "k": v.coords["k"],
            "j_t": v.coords["j_v"].values[0:ny],
            "i_u": np.arange(nx + 1, dtype=float),
        },
        attrs={"units": "m/s2", "long_name": "Coriolis zonal acceleration +f*v"},
    )


def coriolis_v(
    u: xr.DataArray,
    f_q: np.ndarray,
) -> xr.DataArray:
    """
    Coriolis acceleration in the meridional momentum equation at V-faces.

    Evaluates the term −f·u at V-face positions:

        dv/dt|_Coriolis [k, j_v, i] = −f_at_v[j_v, i] · u_at_v[k, j_v, i]

    where

        f_at_v[j_v, i] = ½ (f_q[j_v, i] + f_q[j_v, i+1])
        u_at_v         = 4-point average of surrounding U-faces (see _u_to_v)

    Parameters
    ----------
    u : xr.DataArray, dims (k, j_t, i_u)
        Zonal velocity [m/s].
    f_q : np.ndarray, shape (ny+1, nx+1)
        Coriolis parameter at q-corners [s⁻¹].

    Returns
    -------
    xr.DataArray, dims (k, j_v, i_t)
        Meridional Coriolis acceleration [m/s²].
    """
    u_np = u.values                    # (nz, ny, nx+1)
    nz, ny, nxp1 = u_np.shape
    nx = nxp1 - 1

    # f at V-face: average the two bounding q-corners (west and east)
    # f_q shape: (ny+1, nx+1);  f_at_v shape: (ny+1, nx)
    f_at_v = 0.5 * (f_q[:, 0:nx] + f_q[:, 1:nx + 1])  # (ny+1, nx)

    u_at_v = _u_to_v(u_np)                            # (nz, ny+1, nx)

    cor_v = -f_at_v[np.newaxis, :, :] * u_at_v        # (nz, ny+1, nx)

    return xr.DataArray(
        cor_v,
        dims=["k", "j_v", "i_t"],
        coords={
            "k": u.coords["k"],
            "j_v": np.arange(ny + 1, dtype=float),
            "i_t": u.coords["i_u"].values[0:nx],
        },
        attrs={"units": "m/s2", "long_name": "Coriolis meridional acceleration -f*u"},
    )


def eta_tendency(
    u: xr.DataArray,
    v: xr.DataArray,
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
) -> xr.DataArray:
    """
    Sea-surface-height tendency from the depth-integrated continuity equation.

    Integrates the 3-D divergence of (u, v) over depth:

        ∂η/∂t [j, i] = −Σ_k dz[k] · [(u[k,j,i+1]−u[k,j,i]) / Δx
                                       + (v[k,j+1,i]−v[k,j,i]) / Δy]
                                                    (Marshall et al. 1997, eq. 6)

    where Δx = ``dx_t`` and Δy = ``dy_t`` from the horizontal grid.

    Parameters
    ----------
    u : xr.DataArray, dims (k, j_t, i_u)
        Zonal velocity [m/s].
    v : xr.DataArray, dims (k, j_v, i_t)
        Meridional velocity [m/s].
    h_grid : xr.Dataset
        Horizontal grid providing ``dx_t`` [m] and ``dy_t`` [m].
    v_grid : xr.Dataset
        Vertical grid providing ``dz`` [m].

    Returns
    -------
    xr.DataArray, dims (j_t, i_t)
        ∂η/∂t [m/s].
    """
    u_np  = u.values                    # (nz, ny, nx+1)
    v_np  = v.values                    # (nz, ny+1, nx)
    dx_t  = h_grid["dx_t"].values       # (ny, nx)
    dy_t  = h_grid["dy_t"].values       # (ny, nx)
    dz    = v_grid["dz"].values         # (nz,)

    nz, ny, nx = u_np.shape[0], u_np.shape[1], u_np.shape[2] - 1

    # horizontal divergence of u and v at each layer
    div_u = (u_np[:, :, 1:] - u_np[:, :, :-1]) / dx_t[np.newaxis, :, :]
    div_v = (v_np[:, 1:, :] - v_np[:, :-1, :]) / dy_t[np.newaxis, :, :]

    # depth-integrate: Σ_k dz[k] * (div_u + div_v)
    deta_dt = -np.sum(dz[:, np.newaxis, np.newaxis] * (div_u + div_v), axis=0)

    return xr.DataArray(
        deta_dt,
        dims=["j_t", "i_t"],
        coords={
            "j_t": h_grid.coords["j_t"],
            "i_t": h_grid.coords["i_t"],
        },
        attrs={"units": "m/s", "long_name": "Sea-surface-height tendency deta/dt"},
    )


def diagnose_w(
    u: xr.DataArray,
    v: xr.DataArray,
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
) -> xr.DataArray:
    """
    Diagnose vertical velocity w from the continuity equation.

    Integrates the discrete 3-D continuity equation downward from the
    surface interface (w[k_f=0] = 0, rigid-lid approximation for Phase 1):

        w[k_f+1] = w[k_f] + dz[k] · div2d[k]

    where div2d[k, j, i] = (u[k,j,i+1]−u[k,j,i])/Δx + (v[k,j+1,i]−v[k,j,i])/Δy.

    This follows from ∂u/∂x + ∂v/∂y + ∂w/∂z = 0 (Vallis 2006, §2.4),
    with z increasing upward so ∂w/∂z → (w_top − w_bottom)/dz.

    Parameters
    ----------
    u : xr.DataArray, dims (k, j_t, i_u)
        Zonal velocity [m/s].
    v : xr.DataArray, dims (k, j_v, i_t)
        Meridional velocity [m/s].
    h_grid : xr.Dataset
        Horizontal grid providing ``dx_t``, ``dy_t`` [m].
    v_grid : xr.Dataset
        Vertical grid providing ``dz`` [m] and ``k_f`` coordinate.

    Returns
    -------
    xr.DataArray, dims (k_f, j_t, i_t)
        Vertical velocity w [m/s] at layer interfaces.
        w[k_f=0] = 0 (surface), w[k_f=nz] ≈ 0 (bottom, by continuity).
    """
    u_np = u.values                     # (nz, ny, nx+1)
    v_np = v.values                     # (nz, ny+1, nx)
    dx_t = h_grid["dx_t"].values        # (ny, nx)
    dy_t = h_grid["dy_t"].values        # (ny, nx)
    dz   = v_grid["dz"].values          # (nz,)
    k_f  = v_grid.coords["k_f"].values  # (nz+1,)

    nz = dz.shape[0]
    ny, nx = dx_t.shape

    # 2-D horizontal divergence per layer: (nz, ny, nx)
    div2d = (
        (u_np[:, :, 1:] - u_np[:, :, :-1]) / dx_t[np.newaxis, :, :]
      + (v_np[:, 1:, :] - v_np[:, :-1, :]) / dy_t[np.newaxis, :, :]
    )

    # Integrate downward from surface (w[0] = 0)
    w_np = np.zeros((nz + 1, ny, nx))
    for k in range(nz):
        w_np[k + 1] = w_np[k] + dz[k] * div2d[k]

    return xr.DataArray(
        w_np,
        dims=["k_f", "j_t", "i_t"],
        coords={
            "k_f": k_f,
            "j_t": h_grid.coords["j_t"],
            "i_t": h_grid.coords["i_t"],
        },
        attrs={"units": "m/s", "long_name": "Vertical velocity (diagnosed from continuity)"},
    )


def momentum_tendency(
    state: ModelState,
    pgf_x: xr.DataArray,
    pgf_y: xr.DataArray,
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
    nu_h: float = 0.0,
) -> ModelState:
    """
    Compute the full horizontal momentum tendency d(state)/dt.

    Evaluates the right-hand side of the hydrostatic Boussinesq momentum
    equations (Phase 1: Coriolis + PGF + optional viscosity; no advection):

        du/dt = +f v  − (1/ρ₀)∂p/∂x  + ν_h ∇²u    (Vallis 2006, eq. 14.12a)
        dv/dt = −f u  − (1/ρ₀)∂p/∂y  + ν_h ∇²v    (Vallis 2006, eq. 14.12b)
        dη/dt = −∇·(H ū)                             (continuity, free surface)

    T and S tendencies are zero (advection added in Phase 1 module 7).
    w is diagnosed from continuity (not time-stepped).

    Parameters
    ----------
    state : ModelState
        Current model state (u, v, w, T, S, η).
    pgf_x : xr.DataArray, dims (k, j_t, i_u)
        Zonal pressure gradient force [m/s²] from
        ``core.dynamics.pressure.pressure_gradient_x``.
    pgf_y : xr.DataArray, dims (k, j_v, i_t)
        Meridional pressure gradient force [m/s²] from
        ``core.dynamics.pressure.pressure_gradient_y``.
    h_grid : xr.Dataset
        Horizontal grid (provides ``f_q``, ``dx_t``, ``dy_t``).
    v_grid : xr.Dataset
        Vertical grid (provides ``dz``).
    nu_h : float, optional
        Horizontal Laplacian viscosity [m²/s]. Default 0 (inviscid).

    Returns
    -------
    ModelState
        Tendency dstate/dt.  Units: u, v → [m/s²]; η → [m/s]; T, S, w → 0.

    Notes
    -----
    * The PGF must be computed before calling this function:
        ``pgf_x, pgf_y = pressure_gradient_x(p, h_grid), pressure_gradient_y(p, h_grid)``
    * ``w`` in the returned tendency is zero; call ``diagnose_w`` separately to
      update the diagnostic w after each time step.
    * To obtain a fully Coriolis-only tendency, pass zero arrays for pgf_x/y
      and set nu_h = 0.

    Examples
    --------
    >>> from core.dynamics.pressure import hydrostatic_pressure, pressure_gradient_x, pressure_gradient_y
    >>> from physics.eos import compute_density
    >>> rho  = compute_density(state, vgrid)
    >>> p    = hydrostatic_pressure(rho, state.eta, vgrid)
    >>> pgfx = pressure_gradient_x(p, hgrid)
    >>> pgfy = pressure_gradient_y(p, hgrid)
    >>> tend = momentum_tendency(state, pgfx, pgfy, hgrid, vgrid, nu_h=100.0)
    """
    f_q = h_grid["f_q"].values         # (ny+1, nx+1)
    dx  = float(h_grid.attrs["dx"])    # uniform Cartesian spacing [m]
    dy  = float(h_grid.attrs["dy"])

    u_np = state.u.values              # (nz, ny, nx+1)
    v_np = state.v.values              # (nz, ny+1, nx)

    # ---- du/dt = +f*v + PGF_x + ν_h*∇²u ----------------------------------
    # Coriolis: +f·v at U-faces
    v_at_u = _v_to_u(v_np)                        # (nz, ny, nx+1)
    ny = u_np.shape[1]
    f_at_u = 0.5 * (f_q[0:ny, :] + f_q[1:ny + 1, :])  # (ny, nx+1)
    du_dt = f_at_u[np.newaxis, :, :] * v_at_u     # Coriolis

    # Pressure gradient force (already computed, same shape as u)
    du_dt = du_dt + pgf_x.values

    # Laplacian viscosity
    if nu_h != 0.0:
        du_dt = du_dt + nu_h * _laplacian_u(u_np, dx, dy)

    # ---- dv/dt = -f*u + PGF_y + ν_h*∇²v ----------------------------------
    u_at_v = _u_to_v(u_np)                        # (nz, ny+1, nx)
    nx = v_np.shape[2]
    f_at_v = 0.5 * (f_q[:, 0:nx] + f_q[:, 1:nx + 1])  # (ny+1, nx)
    dv_dt = -f_at_v[np.newaxis, :, :] * u_at_v    # Coriolis

    # Pressure gradient force
    dv_dt = dv_dt + pgf_y.values

    # Laplacian viscosity
    if nu_h != 0.0:
        dv_dt = dv_dt + nu_h * _laplacian_v(v_np, dx, dy)

    # ---- dη/dt from continuity --------------------------------------------
    deta = eta_tendency(state.u, state.v, h_grid, v_grid)

    # ---- Assemble tendency ModelState -------------------------------------
    # T, S, w tendencies = 0 (no advection or mixing in Phase 1)
    du = xr.DataArray(du_dt, dims=state.u.dims, coords=state.u.coords,
                      attrs={"units": "m/s2", "long_name": "du/dt tendency"})
    dv = xr.DataArray(dv_dt, dims=state.v.dims, coords=state.v.coords,
                      attrs={"units": "m/s2", "long_name": "dv/dt tendency"})

    return ModelState(
        u=du,
        v=dv,
        w=xr.zeros_like(state.w),
        T=xr.zeros_like(state.T),
        S=xr.zeros_like(state.S),
        eta=deta,
        time=0.0,
    )
