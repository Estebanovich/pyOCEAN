"""
Vertical grid generation for pyOCEAN — Phase 1: z-levels with partial cells.

The vertical coordinate uses z increasing upward, z = 0 at the free surface,
and z < 0 in the ocean interior.  All depths and thicknesses are positive
quantities unless noted otherwise.

Layer numbering
---------------
  k = 0 : topmost (shallowest) layer
  k = nz-1 : bottommost (deepest) layer

Interface (face) positions z_f and layer-centre positions z_t:

    z = 0 ─── z_f[0] = 0
              │  ←── dz[0] ──→  z_t[0] = z_f[0] - dz[0]/2
    z_f[1] = -dz[0]
              │  ←── dz[1] ──→  z_t[1] = z_f[1] - dz[1]/2
    z_f[2] = -(dz[0] + dz[1])
              ⋮
    z_f[nz] = -H_total

Partial cells (hFac)
--------------------
When the bottom topography cuts through a grid cell the fraction of the
cell that is filled with ocean water is:

    hFac_t[k, j, i] = wet_thickness(k, j, i) / dz[k]

where  wet_thickness = min(dz[k], max(0, depth(j,i) + z_f[k])).

Cells with hFac < hfac_min are treated as fully solid (land) to avoid
numerically degenerate very thin cells.

Staggered hFac on U/V faces
----------------------------
The U-face at (j, i-½) sees the minimum depth of the two adjacent T-cells.
This ensures no flow through land for a closed-boundary domain:

    depth_u[j, i] = min(depth_t[j, i-1], depth_t[j, i])    i=1..nx
                  = 0                                         i=0, nx  (walls)

and analogously for V.

References
----------
Marshall et al. (1997) — A finite-volume, incompressible Navier–Stokes model
    for studies of the ocean on parallel computers. J. Geophys. Res.
MITgcm manual (2023) — §2.10 Partial-step topography (hFac).
Bleck (2002) — HYCOM hybrid coordinate formulation (layer thickness).
"""

from __future__ import annotations

import numpy as np
import xarray as xr


# ---------------------------------------------------------------------------
# Public API — 1-D vertical coordinate
# ---------------------------------------------------------------------------

def create_z_grid(
    nz: int,
    dz: np.ndarray | list[float],
) -> xr.Dataset:
    """
    Create a 1-D z-level vertical grid Dataset from explicit layer thicknesses.

    Parameters
    ----------
    nz : int
        Number of vertical layers.
    dz : array-like, shape (nz,)
        Layer thicknesses [m], positive, ordered surface to bottom.
        ``dz[0]`` is the topmost (shallowest) layer.

    Returns
    -------
    xr.Dataset
        Vertical grid with variables ``z_f`` (nz+1 interface depths),
        ``z_t`` (nz layer-centre depths), and ``dz`` (nz thicknesses).
        All depth values are ≤ 0 (z convention).

    Raises
    ------
    ValueError
        If ``len(dz) != nz`` or any thickness is non-positive.
    """
    dz_arr = np.asarray(dz, dtype=float)
    if dz_arr.shape != (nz,):
        raise ValueError(f"dz must have length nz={nz}; got {dz_arr.shape}")
    if np.any(dz_arr <= 0):
        raise ValueError("All layer thicknesses dz must be positive.")

    # Interface depths: z_f[k] = top of layer k  (z_f[0] = 0, surface)
    z_f = np.zeros(nz + 1, dtype=float)
    z_f[1:] = -np.cumsum(dz_arr)              # z_f[k] = -sum(dz[0:k])

    # Layer-centre depths
    z_t = 0.5 * (z_f[:-1] + z_f[1:])         # midpoint of each layer

    return xr.Dataset(
        {
            "z_f": ("k_f", z_f,  {"units": "m", "long_name": "Interface (face) depth, z convention (≤0)"}),
            "z_t": ("k",   z_t,  {"units": "m", "long_name": "Layer-centre depth, z convention (≤0)"}),
            "dz":  ("k",   dz_arr, {"units": "m", "long_name": "Layer thickness"}),
        },
        coords={
            "k":   ("k",   np.arange(nz),     {"long_name": "Layer index (0 = topmost)"}),
            "k_f": ("k_f", np.arange(nz + 1), {"long_name": "Interface index (0 = surface)"}),
        },
        attrs={
            "nz": nz,
            "total_depth": float(-z_f[-1]),
            "coord_type": "z_level",
            "description": "pyOCEAN 1-D z-level vertical grid",
        },
    )


def uniform_z_grid(nz: int, dz_uniform: float) -> xr.Dataset:
    """
    Convenience wrapper: create a z-level grid with uniform layer thickness.

    Parameters
    ----------
    nz : int
        Number of layers.
    dz_uniform : float
        Uniform layer thickness [m].

    Returns
    -------
    xr.Dataset
        Same structure as ``create_z_grid``.

    Examples
    --------
    >>> g = uniform_z_grid(nz=20, dz_uniform=25.0)  # 500 m total depth
    >>> float(g.attrs["total_depth"])
    500.0
    """
    return create_z_grid(nz, np.full(nz, dz_uniform))


def stretched_z_grid(
    nz: int,
    dz_surface: float,
    total_depth: float,
) -> xr.Dataset:
    """
    Create a geometrically stretched z-level grid.

    Layer thicknesses increase geometrically from the surface downward:

        dz[k] = dz_surface · r^k

    The stretch factor r is found numerically so that
    ``sum(dz) = total_depth``.

    Parameters
    ----------
    nz : int
        Number of layers.
    dz_surface : float
        Thickness of the topmost layer [m].
    total_depth : float
        Total water column depth [m].

    Returns
    -------
    xr.Dataset
        Same structure as ``create_z_grid``.

    Raises
    ------
    ValueError
        If a valid stretch factor cannot be found (e.g. uniform spacing
        already overshoots the total depth).

    Examples
    --------
    >>> g = stretched_z_grid(nz=30, dz_surface=5.0, total_depth=3000.0)
    >>> abs(g.dz.values.sum() - 3000.0) < 1e-6
    True
    """
    if nz * dz_surface > total_depth:
        raise ValueError(
            f"Uniform spacing dz_surface={dz_surface} × nz={nz} = "
            f"{nz * dz_surface} m already exceeds total_depth={total_depth} m."
        )
    if nz * dz_surface == total_depth:
        # Already uniform — no stretch needed
        return uniform_z_grid(nz, dz_surface)

    # Solve for r: dz_surface * (r^nz - 1) / (r - 1) = total_depth
    # Rearranged: f(r) = dz_surface * sum_{k=0}^{nz-1} r^k - total_depth = 0
    from scipy.optimize import brentq  # local import to avoid top-level scipy dep

    def _residual(r: float) -> float:
        return dz_surface * (r**nz - 1.0) / (r - 1.0) - total_depth

    # r must be > 1 for stretching; upper bound: a large r that overshoots
    r_max = 2.0
    while _residual(r_max) < 0:
        r_max *= 2.0

    r = brentq(_residual, 1.0 + 1e-12, r_max, xtol=1e-12, rtol=1e-10)
    dz_arr = dz_surface * r ** np.arange(nz)

    # Correct any floating-point drift so sum is exactly total_depth
    dz_arr[-1] += total_depth - dz_arr.sum()

    return create_z_grid(nz, dz_arr)


# ---------------------------------------------------------------------------
# Public API — 3-D topography / partial-cell fractions
# ---------------------------------------------------------------------------

def compute_hfac(
    z_grid: xr.Dataset,
    depth: np.ndarray,
    hfac_min: float = 0.1,
) -> xr.Dataset:
    """
    Compute partial-cell fractions (hFac) from bathymetric depth.

    The wet fraction of T-cell (k, j, i) is:

        hFac_t[k,j,i] = min(dz[k], max(0, depth[j,i] + z_f[k])) / dz[k]
                                                      (MITgcm manual §2.10)

    Cells with 0 < hFac < hfac_min are collapsed to 0 (treated as land)
    to avoid numerically degenerate ultra-thin cells.

    U-face and V-face depths are the **minimum** of the two adjacent T-cell
    depths, ensuring no flow through solid walls:

        depth_u[j, i] = min(depth_t[j, i-1], depth_t[j, i])   i = 1..nx-1
                       = 0                                       i = 0, nx

    Parameters
    ----------
    z_grid : xr.Dataset
        1-D vertical grid from ``create_z_grid`` or equivalent.
    depth : np.ndarray, shape (ny, nx)
        Bathymetric depth at T-points [m], positive downward (≥ 0).
        depth = 0 means land.
    hfac_min : float, optional
        Minimum allowed partial-cell fraction. Cells thinner than this
        fraction are set to 0 (land). Default 0.1 (matches MITgcm default).

    Returns
    -------
    xr.Dataset
        Dataset containing:

        ``hfac_t(k, j_t, i_t)``  — T-cell wet fraction
        ``hfac_u(k, j_t, i_u)``  — U-face wet fraction
        ``hfac_v(k, j_v, i_t)``  — V-face wet fraction
        ``dz_t(k, j_t, i_t)``    — Effective T-layer thickness = dz·hfac_t [m]
        ``depth_t(j_t, i_t)``    — Bathymetry at T-points [m]
        ``depth_u(j_t, i_u)``    — Bathymetry at U-faces [m]
        ``depth_v(j_v, i_t)``    — Bathymetry at V-faces [m]

    Raises
    ------
    ValueError
        If ``depth`` has negative values or wrong number of dimensions.
    """
    depth = np.asarray(depth, dtype=float)
    if depth.ndim != 2:
        raise ValueError(f"depth must be 2-D (ny, nx); got shape {depth.shape}")
    if np.any(depth < 0):
        raise ValueError("depth must be ≥ 0 everywhere (0 = land).")

    ny, nx = depth.shape
    nz = int(z_grid.attrs["nz"])
    dz_1d = z_grid["dz"].values       # (nz,)
    z_f_1d = z_grid["z_f"].values     # (nz+1,)

    # ---- hFac at T-points ------------------------------------------------
    # Broadcast: z_f_1d[k] has shape (nz,1,1) after slicing [:-1]
    z_top = z_f_1d[:-1, np.newaxis, np.newaxis]     # (nz, 1,  1 ) top of layer k
    dz_3d = dz_1d[:, np.newaxis, np.newaxis]         # (nz, 1,  1 )
    dep_3d = depth[np.newaxis, :, :]                 # (1,  ny, nx)

    # wet thickness = min(dz[k], max(0, depth[j,i] + z_f[k]))
    # z_f[k] is negative, depth is positive, so depth + z_f[k] = depth below top
    wet_t = np.minimum(dz_3d, np.maximum(0.0, dep_3d + z_top))
    hfac_t = wet_t / dz_3d                           # (nz, ny, nx)

    # Apply minimum thickness constraint (collapse near-zero cells to land)
    hfac_t[hfac_t < hfac_min] = 0.0

    # ---- Staggered depth at U-faces (shape ny × (nx+1)) -----------------
    depth_u = np.zeros((ny, nx + 1), dtype=float)
    # Interior U-faces: minimum of the two straddling T-cells
    depth_u[:, 1:-1] = np.minimum(depth[:, :-1], depth[:, 1:])
    # Wall faces: closed boundary → depth = 0 (solid wall)
    depth_u[:, 0] = 0.0
    depth_u[:, -1] = 0.0

    # ---- Staggered depth at V-faces (shape (ny+1) × nx) -----------------
    depth_v = np.zeros((ny + 1, nx), dtype=float)
    depth_v[1:-1, :] = np.minimum(depth[:-1, :], depth[1:, :])
    depth_v[0, :] = 0.0
    depth_v[-1, :] = 0.0

    # ---- hFac at U-faces -------------------------------------------------
    z_top_u = z_top                                  # same z_f, shape (nz,1,1)
    dep_u_3d = depth_u[np.newaxis, :, :]             # (1, ny, nx+1)
    wet_u = np.minimum(dz_3d, np.maximum(0.0, dep_u_3d + z_top_u))
    hfac_u = wet_u / dz_3d
    hfac_u[hfac_u < hfac_min] = 0.0

    # ---- hFac at V-faces -------------------------------------------------
    dep_v_3d = depth_v[np.newaxis, :, :]             # (1, ny+1, nx)
    wet_v = np.minimum(dz_3d, np.maximum(0.0, dep_v_3d + z_top_u))
    hfac_v = wet_v / dz_3d
    hfac_v[hfac_v < hfac_min] = 0.0

    # ---- Effective T-layer thickness ------------------------------------
    dz_t = dz_3d * hfac_t                            # (nz, ny, nx)

    # ---- Assemble Dataset ------------------------------------------------
    dims_t = ["k", "j_t", "i_t"]
    dims_u = ["k", "j_t", "i_u"]
    dims_v = ["k", "j_v", "i_t"]

    ds = xr.Dataset(
        {
            "hfac_t":  (dims_t,           hfac_t,   {"units": "1",   "long_name": "T-cell wet fraction"}),
            "hfac_u":  (dims_u,           hfac_u,   {"units": "1",   "long_name": "U-face wet fraction"}),
            "hfac_v":  (dims_v,           hfac_v,   {"units": "1",   "long_name": "V-face wet fraction"}),
            "dz_t":    (dims_t,           dz_t,     {"units": "m",   "long_name": "Effective T-layer thickness"}),
            "depth_t": (["j_t", "i_t"],   depth,    {"units": "m",   "long_name": "Bathymetry at T-points (positive down)"}),
            "depth_u": (["j_t", "i_u"],   depth_u,  {"units": "m",   "long_name": "Bathymetry at U-faces"}),
            "depth_v": (["j_v", "i_t"],   depth_v,  {"units": "m",   "long_name": "Bathymetry at V-faces"}),
        },
        coords={
            "k":   ("k",   np.arange(nz),     {"long_name": "Layer index"}),
            "i_t": ("i_t", np.arange(nx),     {"long_name": "T-point x-index"}),
            "j_t": ("j_t", np.arange(ny),     {"long_name": "T-point y-index"}),
            "i_u": ("i_u", np.arange(nx + 1), {"long_name": "U-face x-index"}),
            "j_v": ("j_v", np.arange(ny + 1), {"long_name": "V-face y-index"}),
        },
        attrs={
            "nz": nz, "ny": ny, "nx": nx,
            "hfac_min": hfac_min,
            "description": "pyOCEAN partial-cell (hFac) topography",
        },
    )
    return ds
