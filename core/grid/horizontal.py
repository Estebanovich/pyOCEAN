"""
Horizontal grid generation for pyOCEAN.

Implements Arakawa C-grid staggering on Cartesian and spherical-polar
coordinates. All grid information is stored as an xarray Dataset with
named dimensions and CF-compliant variable attributes.

Staggering convention (matches MITgcm / MOM6 / ROMS):

    q---v---q---v---q
    |       |       |
    u   T   u   T   u
    |       |       |
    q---v---q---v---q

  T-points (tracers, pressure, density):  cell centres      → dims (j_t, i_t)
  U-points (zonal velocity u):            western cell face → dims (j_t, i_u)
  V-points (meridional velocity v):       southern face     → dims (j_v, i_t)
  q-points (vorticity, f at corners):     cell corners      → dims (j_v, i_u)

Index sizes for a domain with nx × ny T-cells:
  i_t: 0 … nx-1       j_t: 0 … ny-1
  i_u: 0 … nx         j_v: 0 … ny      (one extra face on each side)

References
----------
Arakawa & Lamb (1977) — Computational design of the basic dynamical
    processes of the UCLA general circulation model. Methods Comput. Phys.
Vallis (2006) — Atmospheric and Oceanic Fluid Dynamics, §1.5 & §2.4.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------
R_EARTH: float = 6.371e6   # m       Earth's mean radius (Vallis 2006, §1.2)
OMEGA: float = 7.2921e-5   # rad/s   Earth's sidereal rotation rate


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def create_cartesian_grid(
    nx: int,
    ny: int,
    dx: float,
    dy: float,
    x0: float = 0.0,
    y0: float = 0.0,
    f0: float = 1e-4,
    beta: float = 0.0,
) -> xr.Dataset:
    """
    Create a uniform Cartesian (flat-Earth) Arakawa C horizontal grid.

    T-point centres are placed at:
        x_T[i] = x0 + (i + ½)·dx,   i = 0 … nx-1
        y_T[j] = y0 + (j + ½)·dy,   j = 0 … ny-1

    The Coriolis parameter follows the f-plane / beta-plane approximation:
        f = f0 + β·y                           (Vallis 2006, eq. 2.34)

    Parameters
    ----------
    nx : int
        Number of T-cells in the x (zonal) direction.
    ny : int
        Number of T-cells in the y (meridional) direction.
    dx : float
        Uniform zonal cell width [m].
    dy : float
        Uniform meridional cell height [m].
    x0 : float, optional
        x-coordinate of the western domain boundary [m]. Default 0.
    y0 : float, optional
        y-coordinate of the southern domain boundary [m]. Default 0.
    f0 : float, optional
        Coriolis parameter at the southern boundary y0 [s⁻¹].
        Default 1e-4 (typical mid-latitude value).
    beta : float, optional
        Beta-plane meridional gradient ∂f/∂y [m⁻¹ s⁻¹]. Default 0 (f-plane).

    Returns
    -------
    xr.Dataset
        Grid dataset. Dimensions: ``i_t``, ``j_t``, ``i_u``, ``j_v``.
        See module docstring for variable naming conventions.

    Examples
    --------
    >>> g = create_cartesian_grid(nx=100, ny=50, dx=2e3, dy=2e3, f0=1e-4)
    >>> g.area_t.values.sum()   # total domain area
    1e10  # 200 km × 100 km
    """
    # ---- 1D position vectors for each point type -------------------------
    # T-point centres
    x_t_1d = x0 + (np.arange(nx) + 0.5) * dx          # length nx
    y_t_1d = y0 + (np.arange(ny) + 0.5) * dy          # length ny

    # U-point western faces (nx+1 values: includes east wall = west wall
    # of the ghost cell just beyond the eastern boundary)
    x_u_1d = x0 + np.arange(nx + 1) * dx              # length nx+1

    # V-point southern faces
    y_v_1d = y0 + np.arange(ny + 1) * dy              # length ny+1

    # ---- 2D position arrays (shape (ny, nx) or similar) ------------------
    # meshgrid convention: X[j, i] — j varies in y, i in x
    X_t, Y_t = np.meshgrid(x_t_1d, y_t_1d)            # (ny,   nx  )
    X_u, Y_u = np.meshgrid(x_u_1d, y_t_1d)            # (ny,   nx+1)
    X_v, Y_v = np.meshgrid(x_t_1d, y_v_1d)            # (ny+1, nx  )
    X_q, Y_q = np.meshgrid(x_u_1d, y_v_1d)            # (ny+1, nx+1)

    # ---- Metric terms (uniform for Cartesian) ----------------------------
    # dx_t[j,i] = zonal width of T-cell  [m]
    # dy_t[j,i] = meridional height of T-cell  [m]
    dx_t = np.full((ny, nx), dx, dtype=float)
    dy_t = np.full((ny, nx), dy, dtype=float)
    area_t = dx_t * dy_t                                # [m²]

    dx_u = np.full((ny, nx + 1), dx, dtype=float)
    dy_u = np.full((ny, nx + 1), dy, dtype=float)
    area_u = dx_u * dy_u

    dx_v = np.full((ny + 1, nx), dx, dtype=float)
    dy_v = np.full((ny + 1, nx), dy, dtype=float)
    area_v = dx_v * dy_v

    # ---- Coriolis parameter ----------------------------------------------
    # f = f0 + β·y  (Vallis 2006, eq. 2.34)
    f_t = f0 + beta * Y_t       # at T-points  (ny,   nx  )
    f_q = f0 + beta * Y_q       # at q-corners (ny+1, nx+1)

    # ---- Ocean mask (all-ocean for idealised Cartesian grid) -------------
    mask_t = np.ones((ny, nx), dtype=np.int8)

    return _assemble_dataset(
        X_t=X_t, Y_t=Y_t,
        X_u=X_u, Y_u=Y_u,
        X_v=X_v, Y_v=Y_v,
        X_q=X_q, Y_q=Y_q,
        dx_t=dx_t, dy_t=dy_t, area_t=area_t,
        dx_u=dx_u, dy_u=dy_u, area_u=area_u,
        dx_v=dx_v, dy_v=dy_v, area_v=area_v,
        f_t=f_t, f_q=f_q,
        mask_t=mask_t,
        attrs={
            "grid_type": "cartesian",
            "nx": nx, "ny": ny,
            "dx": dx, "dy": dy,
            "x0": x0, "y0": y0,
            "Lx": nx * dx, "Ly": ny * dy,
            "f0": f0, "beta": beta,
        },
    )


def create_cartesian_grid_stretched(
    dx: np.ndarray,
    dy: np.ndarray,
    x0: float = 0.0,
    y0: float = 0.0,
    f0: float = 1e-4,
    beta: float = 0.0,
) -> xr.Dataset:
    """
    Create a Cartesian Arakawa C grid with variable (telescopic) spacing.

    Equivalent to MITgcm ``delXFile``/``delYFile``: ``dx[i]`` and ``dy[j]`` are
    the T-cell widths (MITgcm dxF, dyF).  Faces and centres are

        x_u[i] = x0 + Σ_{m<i} dx[m],        x_t[i] = x_u[i] + dx[i]/2

    and the spacing used for gradients at U-faces is the distance between the
    neighbouring T-points (MITgcm dxC, Marshall et al. 1997):

        dx_u[i] = x_t[i] − x_t[i−1] = (dx[i−1] + dx[i]) / 2,   1 ≤ i ≤ nx−1

    with dx_u[0] = dx[0] and dx_u[nx] = dx[nx−1] at the walls (no flow there).
    Likewise dy_v at V-faces.  Face lengths are dy_u[j] = dy[j], dx_v[i] = dx[i].

    The returned grid has no scalar ``dx``/``dy`` attribute, so code that
    assumes uniform spacing fails explicitly instead of silently.

    Parameters
    ----------
    dx : np.ndarray, shape (nx,)
        Zonal cell widths [m], > 0.
    dy : np.ndarray, shape (ny,)
        Meridional cell heights [m], > 0.
    x0, y0 : float
        Position of the south-west corner [m].
    f0, beta : float
        Coriolis parameter f = f0 + β y [s⁻¹, m⁻¹ s⁻¹].

    Returns
    -------
    xr.Dataset
        Same variables and dimensions as ``create_cartesian_grid``.
    """
    dx = np.asarray(dx, dtype=float)
    dy = np.asarray(dy, dtype=float)
    if np.any(dx <= 0) or np.any(dy <= 0):
        raise ValueError("grid spacings dx and dy must be positive")
    nx, ny = dx.size, dy.size

    x_u_1d = x0 + np.concatenate([[0.0], np.cumsum(dx)])
    y_v_1d = y0 + np.concatenate([[0.0], np.cumsum(dy)])
    x_t_1d = x_u_1d[:-1] + 0.5 * dx
    y_t_1d = y_v_1d[:-1] + 0.5 * dy

    X_t, Y_t = np.meshgrid(x_t_1d, y_t_1d)
    X_u, Y_u = np.meshgrid(x_u_1d, y_t_1d)
    X_v, Y_v = np.meshgrid(x_t_1d, y_v_1d)
    X_q, Y_q = np.meshgrid(x_u_1d, y_v_1d)

    # Distance between T-points across each face (dxC, dyC); one-sided at walls
    dxc = np.concatenate([[dx[0]], 0.5 * (dx[:-1] + dx[1:]), [dx[-1]]])
    dyc = np.concatenate([[dy[0]], 0.5 * (dy[:-1] + dy[1:]), [dy[-1]]])

    dx_t, dy_t = np.meshgrid(dx, dy)
    dx_u, dy_u = np.meshgrid(dxc, dy)
    dx_v, dy_v = np.meshgrid(dx, dyc)

    return _assemble_dataset(
        X_t=X_t, Y_t=Y_t, X_u=X_u, Y_u=Y_u, X_v=X_v, Y_v=Y_v, X_q=X_q, Y_q=Y_q,
        dx_t=dx_t, dy_t=dy_t, area_t=dx_t * dy_t,
        dx_u=dx_u, dy_u=dy_u, area_u=dx_u * dy_u,
        dx_v=dx_v, dy_v=dy_v, area_v=dx_v * dy_v,
        f_t=f0 + beta * Y_t, f_q=f0 + beta * Y_q,
        mask_t=np.ones((ny, nx), dtype=np.int8),
        attrs={
            "grid_type": "cartesian_stretched",
            "nx": nx, "ny": ny, "x0": x0, "y0": y0,
            "Lx": float(dx.sum()), "Ly": float(dy.sum()),
            "f0": f0, "beta": beta,
        },
    )


def create_spherical_grid(
    nx: int,
    ny: int,
    dlon: float,
    dlat: float,
    lon0: float = 0.0,
    lat0: float = 0.0,
) -> xr.Dataset:
    """
    Create a uniform spherical-polar Arakawa C horizontal grid.

    T-point centres are placed at:
        lon_T[i] = lon0 + (i + ½)·dlon,  i = 0 … nx-1
        lat_T[j] = lat0 + (j + ½)·dlat,  j = 0 … ny-1

    Physical distances are computed from the arc-length formulae:
        dx = R·cos(φ)·Δλ           (Vallis 2006, §1.2)
        dy = R·Δφ

    The Coriolis parameter uses the full spherical expression:
        f = 2·Ω·sin(φ)             (Vallis 2006, eq. 2.24)

    Parameters
    ----------
    nx : int
        Number of T-cells in the zonal direction.
    ny : int
        Number of T-cells in the meridional direction.
    dlon : float
        Uniform zonal grid spacing [degrees].
    dlat : float
        Uniform meridional grid spacing [degrees].
    lon0 : float, optional
        Longitude of the western boundary [degrees]. Default 0.
    lat0 : float, optional
        Latitude of the southern boundary [degrees]. Default 0.

    Returns
    -------
    xr.Dataset
        Grid dataset with the same structure as ``create_cartesian_grid``.
        Positions are stored as lon/lat [°] AND as x/y [m] (arc-length
        distances from the south-west corner).
    """
    # ---- 1D angular positions -------------------------------------------
    lon_t_1d = lon0 + (np.arange(nx) + 0.5) * dlon    # (nx,)
    lat_t_1d = lat0 + (np.arange(ny) + 0.5) * dlat    # (ny,)
    lon_u_1d = lon0 + np.arange(nx + 1) * dlon         # (nx+1,)
    lat_v_1d = lat0 + np.arange(ny + 1) * dlat         # (ny+1,)

    # ---- 2D angular arrays -----------------------------------------------
    LON_t, LAT_t = np.meshgrid(lon_t_1d, lat_t_1d)    # (ny,   nx  )
    LON_u, LAT_u = np.meshgrid(lon_u_1d, lat_t_1d)    # (ny,   nx+1)
    LON_v, LAT_v = np.meshgrid(lon_t_1d, lat_v_1d)    # (ny+1, nx  )
    LON_q, LAT_q = np.meshgrid(lon_u_1d, lat_v_1d)    # (ny+1, nx+1)

    lat_t_rad = np.deg2rad(LAT_t)
    lat_u_rad = np.deg2rad(LAT_u)
    lat_v_rad = np.deg2rad(LAT_v)
    lat_q_rad = np.deg2rad(LAT_q)

    dlon_rad = np.deg2rad(dlon)
    dlat_rad = np.deg2rad(dlat)

    # ---- Metric terms ---------------------------------------------------
    # T-cell: dx_t = R·cos(φ_T)·Δλ,  dy_t = R·Δφ
    dx_t = R_EARTH * np.cos(lat_t_rad) * dlon_rad      # (ny,   nx  ) [m]
    dy_t = np.full_like(dx_t, R_EARTH * dlat_rad)
    area_t = dx_t * dy_t                                # [m²]

    # U-cell (at U-point latitude = T-point latitude, but U-face has T-cell dy)
    dx_u = R_EARTH * np.cos(lat_u_rad) * dlon_rad      # (ny,   nx+1) [m]
    dy_u = np.full_like(dx_u, R_EARTH * dlat_rad)
    area_u = dx_u * dy_u

    # V-cell (at V-point latitude)
    dx_v = R_EARTH * np.cos(lat_v_rad) * dlon_rad      # (ny+1, nx  ) [m]
    dy_v = np.full_like(dx_v, R_EARTH * dlat_rad)
    area_v = dx_v * dy_v

    # Cartesian arc-length positions from SW corner (for compatibility)
    X_t = R_EARTH * np.deg2rad(LON_t - lon0) * np.cos(np.deg2rad(lat0))
    Y_t = R_EARTH * np.deg2rad(LAT_t - lat0)
    X_u = R_EARTH * np.deg2rad(LON_u - lon0) * np.cos(np.deg2rad(lat0))
    Y_u = R_EARTH * np.deg2rad(LAT_u - lat0)
    X_v = R_EARTH * np.deg2rad(LON_v - lon0) * np.cos(np.deg2rad(lat0))
    Y_v = R_EARTH * np.deg2rad(LAT_v - lat0)
    X_q = R_EARTH * np.deg2rad(LON_q - lon0) * np.cos(np.deg2rad(lat0))
    Y_q = R_EARTH * np.deg2rad(LAT_q - lat0)

    # ---- Coriolis: f = 2·Ω·sin(φ) ---------------------------------------
    f_t = 2.0 * OMEGA * np.sin(lat_t_rad)
    f_q = 2.0 * OMEGA * np.sin(lat_q_rad)

    # ---- Mask ------------------------------------------------------------
    mask_t = np.ones((ny, nx), dtype=np.int8)

    ds = _assemble_dataset(
        X_t=X_t, Y_t=Y_t,
        X_u=X_u, Y_u=Y_u,
        X_v=X_v, Y_v=Y_v,
        X_q=X_q, Y_q=Y_q,
        dx_t=dx_t, dy_t=dy_t, area_t=area_t,
        dx_u=dx_u, dy_u=dy_u, area_u=area_u,
        dx_v=dx_v, dy_v=dy_v, area_v=area_v,
        f_t=f_t, f_q=f_q,
        mask_t=mask_t,
        attrs={
            "grid_type": "spherical_polar",
            "nx": nx, "ny": ny,
            "dlon": dlon, "dlat": dlat,
            "lon0": lon0, "lat0": lat0,
        },
    )

    # Attach lon/lat as additional coordinate variables
    dims_t = ["j_t", "i_t"]
    dims_u = ["j_t", "i_u"]
    dims_v = ["j_v", "i_t"]
    dims_q = ["j_v", "i_u"]
    ds["lon_t"] = xr.DataArray(LON_t, dims=dims_t, attrs={"units": "degrees_east",  "long_name": "T-point longitude"})
    ds["lat_t"] = xr.DataArray(LAT_t, dims=dims_t, attrs={"units": "degrees_north", "long_name": "T-point latitude"})
    ds["lon_u"] = xr.DataArray(LON_u, dims=dims_u, attrs={"units": "degrees_east",  "long_name": "U-point longitude"})
    ds["lat_u"] = xr.DataArray(LAT_u, dims=dims_u, attrs={"units": "degrees_north", "long_name": "U-point latitude"})
    ds["lon_v"] = xr.DataArray(LON_v, dims=dims_v, attrs={"units": "degrees_east",  "long_name": "V-point longitude"})
    ds["lat_v"] = xr.DataArray(LAT_v, dims=dims_v, attrs={"units": "degrees_north", "long_name": "V-point latitude"})
    ds["lon_q"] = xr.DataArray(LON_q, dims=dims_q, attrs={"units": "degrees_east",  "long_name": "q-point longitude"})
    ds["lat_q"] = xr.DataArray(LAT_q, dims=dims_q, attrs={"units": "degrees_north", "long_name": "q-point latitude"})

    return ds


def set_land_mask(grid: xr.Dataset, mask: np.ndarray) -> xr.Dataset:
    """
    Replace the ocean mask in a grid Dataset.

    Parameters
    ----------
    grid : xr.Dataset
        Grid dataset returned by ``create_cartesian_grid`` or
        ``create_spherical_grid``.
    mask : np.ndarray, shape (ny, nx)
        Integer mask array: 1 = ocean cell, 0 = land cell.

    Returns
    -------
    xr.Dataset
        New dataset with ``mask_t`` replaced.
    """
    ny = grid.attrs["ny"]
    nx = grid.attrs["nx"]
    if mask.shape != (ny, nx):
        raise ValueError(
            f"mask shape {mask.shape} does not match grid (ny={ny}, nx={nx})"
        )
    return grid.assign(
        mask_t=xr.DataArray(
            mask.astype(np.int8),
            dims=["j_t", "i_t"],
            attrs={"units": "1", "long_name": "Ocean mask at T-points (1=ocean, 0=land)"},
        )
    )


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _assemble_dataset(
    *,
    X_t: np.ndarray, Y_t: np.ndarray,
    X_u: np.ndarray, Y_u: np.ndarray,
    X_v: np.ndarray, Y_v: np.ndarray,
    X_q: np.ndarray, Y_q: np.ndarray,
    dx_t: np.ndarray, dy_t: np.ndarray, area_t: np.ndarray,
    dx_u: np.ndarray, dy_u: np.ndarray, area_u: np.ndarray,
    dx_v: np.ndarray, dy_v: np.ndarray, area_v: np.ndarray,
    f_t: np.ndarray, f_q: np.ndarray,
    mask_t: np.ndarray,
    attrs: dict,
) -> xr.Dataset:
    """
    Assemble a grid xr.Dataset from pre-computed metric arrays.

    All shape conventions:
        T / metric arrays: (ny,   nx  ) → dims [j_t, i_t]
        U arrays:          (ny,   nx+1) → dims [j_t, i_u]
        V arrays:          (ny+1, nx  ) → dims [j_v, i_t]
        q arrays:          (ny+1, nx+1) → dims [j_v, i_u]
    """
    ny, nx = X_t.shape

    dims_t = ["j_t", "i_t"]
    dims_u = ["j_t", "i_u"]
    dims_v = ["j_v", "i_t"]
    dims_q = ["j_v", "i_u"]

    ds = xr.Dataset(
        data_vars={
            # ---- Grid-point positions ------------------------------------
            "x_t":    (dims_t, X_t,    {"units": "m", "long_name": "T-point x-position"}),
            "y_t":    (dims_t, Y_t,    {"units": "m", "long_name": "T-point y-position"}),
            "x_u":    (dims_u, X_u,    {"units": "m", "long_name": "U-point x-position"}),
            "y_u":    (dims_u, Y_u,    {"units": "m", "long_name": "U-point y-position"}),
            "x_v":    (dims_v, X_v,    {"units": "m", "long_name": "V-point x-position"}),
            "y_v":    (dims_v, Y_v,    {"units": "m", "long_name": "V-point y-position"}),
            "x_q":    (dims_q, X_q,    {"units": "m", "long_name": "Vorticity-point x-position"}),
            "y_q":    (dims_q, Y_q,    {"units": "m", "long_name": "Vorticity-point y-position"}),
            # ---- Metric terms -------------------------------------------
            "dx_t":   (dims_t, dx_t,   {"units": "m",  "long_name": "Zonal width of T-cell"}),
            "dy_t":   (dims_t, dy_t,   {"units": "m",  "long_name": "Meridional height of T-cell"}),
            "area_t": (dims_t, area_t, {"units": "m2", "long_name": "T-cell area"}),
            "dx_u":   (dims_u, dx_u,   {"units": "m",  "long_name": "Zonal width of U-cell"}),
            "dy_u":   (dims_u, dy_u,   {"units": "m",  "long_name": "Meridional height of U-cell"}),
            "area_u": (dims_u, area_u, {"units": "m2", "long_name": "U-cell area"}),
            "dx_v":   (dims_v, dx_v,   {"units": "m",  "long_name": "Zonal width of V-cell"}),
            "dy_v":   (dims_v, dy_v,   {"units": "m",  "long_name": "Meridional height of V-cell"}),
            "area_v": (dims_v, area_v, {"units": "m2", "long_name": "V-cell area"}),
            # ---- Coriolis -----------------------------------------------
            "f_t":    (dims_t, f_t,    {"units": "s-1", "long_name": "Coriolis parameter at T-points"}),
            "f_q":    (dims_q, f_q,    {"units": "s-1", "long_name": "Coriolis parameter at q-points"}),
            # ---- Ocean mask ---------------------------------------------
            "mask_t": (dims_t, mask_t, {"units": "1",   "long_name": "Ocean mask at T-points (1=ocean, 0=land)"}),
        },
        coords={
            "i_t": ("i_t", np.arange(nx),     {"long_name": "T-point x-index"}),
            "j_t": ("j_t", np.arange(ny),     {"long_name": "T-point y-index"}),
            "i_u": ("i_u", np.arange(nx + 1), {"long_name": "U/q-point x-index"}),
            "j_v": ("j_v", np.arange(ny + 1), {"long_name": "V/q-point y-index"}),
        },
        attrs={
            "staggering": "Arakawa C-grid",
            "description": "pyOCEAN horizontal grid",
            **attrs,
        },
    )
    return ds
