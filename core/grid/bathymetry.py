"""
Bathymetry loading and preprocessing for pyOCEAN — Phase 2.

Pipeline
--------
1. Load raw bathymetry from a GEBCO-format NetCDF file (``load_gebco``).
2. Regrid to the model's T-point positions (``regrid_to_model``).
3. Smooth with a weighted 9-point filter (``smooth_bathymetry``).
4. Remove isolated ocean cells (``fill_isolated_cells``).
5. Enforce a minimum ocean depth (``apply_min_depth``).
6. One-call convenience wrapper (``build_bathymetry``).

Sign convention
---------------
  GEBCO elevation:   negative in ocean, positive on land  [m]
  Model depth:       positive downward, 0 = land           [m]

Smoothing
---------
The default smoothing kernel follows the HYCOM ``topo_smooth_skip`` approach
(HYCOM-tools-master/topo/src/topo_smooth_skip.f):

    weights = [[1, 2, 1],      / 16      (Hann-squared kernel)
               [2, 4, 2],
               [1, 2, 1]]

Land neighbours contribute the centre cell's value to the stencil sum,
preserving depth near coastlines.  The optional ``n_skip`` parameter
prevents smoothing within ``n_skip`` grid cells of any land boundary.

An alternative 5-point Shapiro (1970) kernel is available via
``kernel='shapiro'``, which excludes land neighbours from the average.

Isolated cells
--------------
Ocean cells with fewer than 2 ocean lateral neighbours are set to land.
This follows ``topo_zthin.f`` in HYCOM-tools, which removes single-width
inlets and isolated sea cells that cannot sustain advective flow.

References
----------
GEBCO Compilation Group (2023) — GEBCO 2023 Grid.
    doi:10.5285/f98b053b-0cbc-6c23-d9da-7de7d696a50d
HYCOM-tools-master/topo/src/topo_smooth_skip.f — 9-point weighted smoother
    with nskip land-boundary buffer.
HYCOM-tools-master/topo/src/topo_zthin.f — isolated cell removal.
Shapiro (1970) — Smoothing, filtering, and boundary effects.
    Rev. Geophys. Space Phys., 8(2), 359–387.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr
from scipy.interpolate import RegularGridInterpolator


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def load_gebco(
    path: str | Path,
    lon_bounds: tuple[float, float],
    lat_bounds: tuple[float, float],
) -> xr.DataArray:
    """
    Load GEBCO (or ETOPO-style) bathymetry for a geographic subdomain.

    Reads the ``elevation`` variable (or ``z`` as a fallback) from a
    CF-compliant NetCDF4 file and returns a DataArray clipped to the
    requested lon/lat bounds plus a 0.5° margin, ensuring the model domain
    is fully covered for subsequent interpolation.

    Parameters
    ----------
    path : str or Path
        Path to the GEBCO NetCDF4 file.
    lon_bounds : (lon_min, lon_max)
        Longitude extent of the model domain [°E].
    lat_bounds : (lat_min, lat_max)
        Latitude extent of the model domain [°N].

    Returns
    -------
    xr.DataArray
        Elevation [m], negative in ocean, positive on land.
        Coordinates: ``lon`` [°E], ``lat`` [°N].

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    KeyError
        If neither ``elevation`` nor ``z`` is present in the file.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Bathymetry file not found: {path}")

    ds = xr.open_dataset(path)

    # Detect the elevation variable name
    if "elevation" in ds:
        elev_name = "elevation"
    elif "z" in ds:
        elev_name = "z"
    else:
        raise KeyError(
            f"Cannot find elevation variable in {path}. "
            f"Available variables: {list(ds.data_vars)}"
        )

    raw = ds[elev_name]

    # Normalise coordinate names to 'lon' / 'lat' regardless of source naming
    rename_map: dict[str, str] = {}
    for dim in raw.dims:
        dl = dim.lower()
        if dl in ("lon", "longitude", "x"):
            rename_map[dim] = "lon"
        elif dl in ("lat", "latitude", "y"):
            rename_map[dim] = "lat"
    if rename_map:
        raw = raw.rename(rename_map)

    # Clip with a margin so interpolation near domain edges is reliable
    margin = 0.5
    lon_min, lon_max = lon_bounds
    lat_min, lat_max = lat_bounds

    lon_mask = (raw.lon >= lon_min - margin) & (raw.lon <= lon_max + margin)
    lat_mask = (raw.lat >= lat_min - margin) & (raw.lat <= lat_max + margin)
    raw = raw.isel(lon=lon_mask, lat=lat_mask)

    raw.attrs.update({
        "units": "m",
        "long_name": "Elevation (negative in ocean)",
        "source": str(path),
    })
    return raw


def regrid_to_model(
    raw: xr.DataArray,
    lon_t: np.ndarray,
    lat_t: np.ndarray,
    method: str = "linear",
) -> np.ndarray:
    """
    Regrid raw elevation data onto model T-point positions.

    Bilinear (or nearest-neighbour) interpolation is performed via
    ``scipy.interpolate.RegularGridInterpolator``.  Points that fall outside
    the raw data extent are treated as land (elevation = +1 m).

    The GEBCO sign convention is flipped and clamped to give model depth:

        depth[j,i] = max(0, −elevation[j,i])          [m, positive downward]

    Parameters
    ----------
    raw : xr.DataArray
        Elevation [m] on a regular lon/lat grid (from ``load_gebco``).
        Must have coordinates named ``lon`` and ``lat``.
    lon_t : np.ndarray, shape (ny, nx) or (nx,)
        Longitude of model T-points [°E].
    lat_t : np.ndarray, shape (ny, nx) or (ny,)
        Latitude of model T-points [°N].
    method : {'linear', 'nearest'}, optional
        Interpolation method. Default ``'linear'``.

    Returns
    -------
    np.ndarray, shape (ny, nx)
        Bathymetric depth at T-points [m], positive downward, 0 = land.
    """
    # Extract 1-D coordinate vectors, ensuring lat is in increasing order
    lat_vals = raw.lat.values.copy()
    lon_vals = raw.lon.values.copy()

    # Determine array orientation: assume DataArray dims are (lat, lon)
    if raw.dims[0] == "lon":
        elev_vals = raw.values.T   # → (lat, lon)
    else:
        elev_vals = raw.values.copy()

    if lat_vals[0] > lat_vals[-1]:
        lat_vals = lat_vals[::-1]
        elev_vals = elev_vals[::-1, :]

    interp = RegularGridInterpolator(
        (lat_vals, lon_vals),
        elev_vals,
        method=method,
        bounds_error=False,
        fill_value=1.0,   # outside domain → treat as land (positive elevation)
    )

    lon_t = np.asarray(lon_t, dtype=float)
    lat_t = np.asarray(lat_t, dtype=float)

    # Support both 1-D (regular-grid shorthand) and 2-D input
    if lon_t.ndim == 1 and lat_t.ndim == 1:
        LON, LAT = np.meshgrid(lon_t, lat_t)
    else:
        LON = lon_t
        LAT = lat_t

    ny, nx = LON.shape
    pts = np.column_stack([LAT.ravel(), LON.ravel()])
    elev_interp = interp(pts).reshape(ny, nx)

    # Flip sign: ocean (negative elevation) → positive depth; land/above-SL → 0
    depth = np.maximum(0.0, -elev_interp)
    return depth


def smooth_bathymetry(
    depth: np.ndarray,
    n_passes: int = 1,
    mask: np.ndarray | None = None,
    kernel: str = "hann",
    n_skip: int = 0,
) -> np.ndarray:
    """
    Apply a weighted low-pass filter to smooth bathymetry.

    Two kernels are available:

    ``'hann'`` (default) — 9-point weighted kernel from HYCOM
    ``topo_smooth_skip.f``:

        weights = [[1, 2, 1],      / 16
                   [2, 4, 2],
                   [1, 2, 1]]

        Land neighbours substitute the centre cell's value, so the filter
        reduces to a weighted self-average near coastlines rather than
        artificially biasing depth toward zero.

    ``'shapiro'`` — 5-point Shapiro (1970) filter:

        depth'[j,i] = (1 - α)·depth[j,i]
                    + (α / n_ocn)·Σ_{ocean nbrs} depth[nbr]

        with α = 0.5.  Land neighbours are excluded from the stencil.

    The ``n_skip`` parameter (HYCOM ``nskip``) prevents smoothing within
    ``n_skip`` grid cells of any land boundary, preserving coastal detail.

    Parameters
    ----------
    depth : np.ndarray, shape (ny, nx)
        Bathymetric depth [m], positive downward, 0 = land.
    n_passes : int, optional
        Number of filter passes. Default 1.
    mask : np.ndarray of int, shape (ny, nx), optional
        Ocean mask (1 = ocean, 0 = land).  If None, inferred from
        ``depth > 0``.
    kernel : {'hann', 'shapiro'}, optional
        Smoothing kernel. Default ``'hann'`` (HYCOM convention).
    n_skip : int, optional
        Do not smooth ocean cells within this many grid cells of land.
        Default 0 (smooth all ocean cells).  Mirrors HYCOM ``nskip``.

    Returns
    -------
    np.ndarray, shape (ny, nx)
        Smoothed depth [m].  Land cells remain 0.

    References
    ----------
    HYCOM-tools-master/topo/src/topo_smooth_skip.f
    Shapiro (1970) — Rev. Geophys. Space Phys., 8(2), 359–387.
    """
    if n_passes == 0:
        return depth.copy()

    if kernel not in ("hann", "shapiro"):
        raise ValueError(f"kernel must be 'hann' or 'shapiro'; got '{kernel}'")

    if mask is None:
        mask = (depth > 0).astype(np.int8)

    m = mask.astype(float)

    # Compute the smoothing-eligible mask (ocean cells ≥ n_skip from land)
    eligible = _erode_mask(mask, n_skip)   # shape (ny, nx), dtype int8

    d = depth.copy()

    for _ in range(n_passes):
        if kernel == "hann":
            d = _smooth_hann(d, m, eligible)
        else:
            d = _smooth_shapiro(d, m, eligible)

    return d


def fill_isolated_cells(
    depth: np.ndarray,
    min_ocean_neighbors: int = 2,
) -> np.ndarray:
    """
    Remove ocean cells with too few ocean lateral neighbours.

    Cells with fewer than ``min_ocean_neighbors`` ocean cells among their
    four lateral neighbours (N/S/E/W) are converted to land (depth = 0).
    The process iterates until no more cells are removed, so narrow inlets
    and isolated basins connected only through such cells are fully cleaned.

    This mirrors the logic in HYCOM ``topo_zthin.f`` which removes ocean
    cells with ≥ 3 land neighbours (equivalently, < 2 ocean neighbours).

    Parameters
    ----------
    depth : np.ndarray, shape (ny, nx)
        Bathymetric depth [m], positive downward, 0 = land.
    min_ocean_neighbors : int, optional
        Minimum required ocean lateral neighbours. Default 2
        (removes cells with ≥ 3 land neighbours, as in HYCOM).

    Returns
    -------
    np.ndarray, shape (ny, nx)
        Depth array with isolated cells set to 0.
    """
    d = depth.copy()

    while True:
        m_pad = np.pad((d > 0).astype(np.int8), 1,
                       mode="constant", constant_values=0)
        n_ocn_nbrs = (
            m_pad[:-2, 1:-1] +   # north
            m_pad[2:,  1:-1] +   # south
            m_pad[1:-1, :-2] +   # west
            m_pad[1:-1, 2:]       # east
        )
        remove = (d > 0) & (n_ocn_nbrs < min_ocean_neighbors)
        if not remove.any():
            break
        d[remove] = 0.0

    return d


def apply_min_depth(
    depth: np.ndarray,
    min_depth: float,
) -> np.ndarray:
    """
    Enforce a minimum ocean depth.

    Ocean cells (depth > 0) shallower than ``min_depth`` are raised to
    exactly ``min_depth``.  Land cells (depth = 0) are unaffected.

    Parameters
    ----------
    depth : np.ndarray
        Bathymetric depth [m], positive downward, 0 = land.
    min_depth : float
        Minimum allowed ocean depth [m].

    Returns
    -------
    np.ndarray
        Depth array with all ocean cells ≥ min_depth.
    """
    d = depth.copy()
    ocean = d > 0
    d[ocean] = np.maximum(d[ocean], min_depth)
    return d


def build_bathymetry(
    h_grid: xr.Dataset,
    source_path: str | Path,
    min_depth: float = 10.0,
    n_smooth_passes: int = 2,
    n_skip: int = 1,
    kernel: str = "hann",
    method: str = "linear",
) -> np.ndarray:
    """
    Full pipeline: load GEBCO → regrid → smooth → fill isolated → min depth.

    Convenience function that chains ``load_gebco``, ``regrid_to_model``,
    ``smooth_bathymetry``, ``fill_isolated_cells``, and ``apply_min_depth``
    using the model's spherical horizontal grid to determine target positions
    and domain bounds.

    Parameters
    ----------
    h_grid : xr.Dataset
        Horizontal grid from ``create_spherical_grid``.
        Must contain ``lon_t`` and ``lat_t`` (2-D coordinate arrays).
    source_path : str or Path
        Path to GEBCO NetCDF4 file.
    min_depth : float, optional
        Minimum ocean depth [m].  Default 10 m.
    n_smooth_passes : int, optional
        Number of smoother passes.  Default 2.
    n_skip : int, optional
        Do not smooth within this many cells of land.  Default 1.
    kernel : {'hann', 'shapiro'}, optional
        Smoothing kernel.  Default ``'hann'`` (HYCOM convention).
    method : {'linear', 'nearest'}, optional
        Interpolation method for regridding.  Default ``'linear'``.

    Returns
    -------
    np.ndarray, shape (ny, nx)
        Bathymetric depth at T-points [m], positive downward, 0 = land.
        Ready to pass directly to ``compute_hfac``.

    Raises
    ------
    KeyError
        If ``h_grid`` lacks ``lon_t`` / ``lat_t`` (i.e., a Cartesian grid
        was passed; for Cartesian grids supply the depth array directly).
    """
    if "lon_t" not in h_grid or "lat_t" not in h_grid:
        raise KeyError(
            "'h_grid' must have 'lon_t' and 'lat_t' variables. "
            "Use create_spherical_grid for real-bathymetry runs. "
            "For Cartesian idealised runs, build the depth array directly."
        )

    lon_t = h_grid["lon_t"].values    # (ny, nx)
    lat_t = h_grid["lat_t"].values    # (ny, nx)

    lon_bounds = (float(lon_t.min()), float(lon_t.max()))
    lat_bounds = (float(lat_t.min()), float(lat_t.max()))

    raw   = load_gebco(source_path, lon_bounds, lat_bounds)
    depth = regrid_to_model(raw, lon_t, lat_t, method=method)
    depth = smooth_bathymetry(depth, n_passes=n_smooth_passes,
                              kernel=kernel, n_skip=n_skip)
    depth = fill_isolated_cells(depth)
    depth = apply_min_depth(depth, min_depth)
    return depth


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _erode_mask(mask: np.ndarray, n: int) -> np.ndarray:
    """
    Erode the ocean mask inward by ``n`` cells.

    After n passes, cells within n grid cells of any land boundary are
    set to 0 in the returned mask.  This implements the HYCOM ``nskip``
    land-buffer logic from ``topo_smooth_skip.f``.

    Parameters
    ----------
    mask : np.ndarray, shape (ny, nx), dtype int
        Ocean mask (1 = ocean, 0 = land).
    n : int
        Number of erosion passes.

    Returns
    -------
    np.ndarray, shape (ny, nx), dtype int8
        Eroded mask: 1 only for ocean cells ≥ n cells from any land.
    """
    m = mask.astype(np.int8).copy()
    for _ in range(n):
        m_pad = np.pad(m, 1, mode="constant", constant_values=0)
        min_nbr = np.minimum(
            np.minimum(m_pad[:-2, 1:-1], m_pad[2:, 1:-1]),
            np.minimum(m_pad[1:-1, :-2], m_pad[1:-1, 2:]),
        )
        m = (m & min_nbr).astype(np.int8)
    return m


# HYCOM topo_smooth_skip.f 9-point kernel weights / normalisation
_HANN_KERNEL = np.array([[1., 2., 1.],
                          [2., 4., 2.],
                          [1., 2., 1.]]) / 16.0

# Shapiro (1970) filter strength
_SHAPIRO_ALPHA = 0.5


def _smooth_hann(
    d: np.ndarray,
    m: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """
    One pass of the HYCOM 9-point Hann-squared smoother.

    Land neighbours contribute the centre cell's value (not zero), so the
    filter degrades gracefully near coastlines rather than biasing depths.

    Parameters
    ----------
    d : np.ndarray, shape (ny, nx)  — current depth
    m : np.ndarray, shape (ny, nx)  — full ocean mask (float)
    eligible : np.ndarray           — smoothing-eligible mask (int8)
    """
    # Zero-pad depth and mask for neighbour lookups
    d_pad = np.pad(d, 1, mode="constant", constant_values=0.0)
    m_pad = np.pad(m, 1, mode="constant", constant_values=0.0)

    ny, nx = d.shape
    sh = np.zeros((ny, nx))

    for dj in (-1, 0, 1):
        for di in (-1, 0, 1):
            w = _HANN_KERNEL[dj + 1, di + 1]
            nbr_d = d_pad[1 + dj: 1 + dj + ny, 1 + di: 1 + di + nx]
            nbr_m = m_pad[1 + dj: 1 + dj + ny, 1 + di: 1 + di + nx]
            # Land neighbour → substitute centre depth (HYCOM convention)
            effective_d = np.where(nbr_m > 0, nbr_d, d)
            sh += w * effective_d

    # Only update eligible ocean cells; land stays 0
    d_new = np.where(eligible > 0, sh, d)
    d_new = np.where(m > 0, d_new, 0.0)
    return d_new


def _smooth_shapiro(
    d: np.ndarray,
    m: np.ndarray,
    eligible: np.ndarray,
) -> np.ndarray:
    """
    One pass of the Shapiro (1970) 5-point land-masked filter.

    Land neighbours are excluded from the stencil average.

    Parameters
    ----------
    d : np.ndarray, shape (ny, nx)  — current depth
    m : np.ndarray, shape (ny, nx)  — full ocean mask (float)
    eligible : np.ndarray           — smoothing-eligible mask (int8)
    """
    d_pad = np.pad(d, 1, mode="constant", constant_values=0.0)
    m_pad = np.pad(m, 1, mode="constant", constant_values=0.0)

    ny, nx = d.shape
    north = d_pad[0:ny,     1:nx + 1]
    south = d_pad[2:ny + 2, 1:nx + 1]
    west  = d_pad[1:ny + 1, 0:nx]
    east  = d_pad[1:ny + 1, 2:nx + 2]

    mn = m_pad[0:ny,     1:nx + 1]
    ms = m_pad[2:ny + 2, 1:nx + 1]
    mw = m_pad[1:ny + 1, 0:nx]
    me = m_pad[1:ny + 1, 2:nx + 2]

    n_ocn  = mn + ms + mw + me
    sum_nb = north * mn + south * ms + west * mw + east * me

    d_avg = np.where(
        n_ocn > 0,
        (1 - _SHAPIRO_ALPHA) * d + (_SHAPIRO_ALPHA / np.maximum(n_ocn, 1)) * sum_nb,
        d,
    )
    d_new = np.where(eligible > 0, d_avg, d)
    d_new = np.where(m > 0, d_new, 0.0)
    return d_new
