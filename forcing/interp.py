"""
Interpolation of canonical forcing onto the model grid and model time.

Shared by every forcing adapter: an adapter only translates names and units
(``forcing/canonical.py``); everything that depends on the model grid lives
here.

Operations
----------
``fill_land``      Sea-over-land extrapolation, so that land NaNs in the
                   source do not contaminate coastal ocean points during
                   bilinear interpolation.
``to_model_grid``  Bilinear interpolation from a regular lon/lat source grid
                   to the T, U or V points of the Arakawa C-grid.
``interp_time``    Linear interpolation between forcing records.

The sea-over-land step is the standard preprocessing used for HYCOM
(``HYCOM-tools``: ``landfill``) and ROMS/CROCO (``fillmask`` in
croco_tools); here it is an iterative 4-neighbour mean.

Curvilinear sources (WRF, native HYCOM ``.a/.b``) are not handled yet.

References
----------
Press, W.H. et al. (2007). Numerical Recipes, 3rd ed., §3.6 (bilinear
    interpolation on a regular grid).
"""

from __future__ import annotations

import numpy as np
import xarray as xr

#: C-grid point → (y-dimension, x-dimension) of the pyOCEAN horizontal grid.
_POINT_DIMS: dict[str, tuple[str, str]] = {
    "t": ("j_t", "i_t"),
    "u": ("j_t", "i_u"),
    "v": ("j_v", "i_t"),
}


# ---------------------------------------------------------------------------
# Horizontal interpolation
# ---------------------------------------------------------------------------

def to_model_grid(ds: xr.Dataset, grid: xr.Dataset, point: str = "t") -> xr.Dataset:
    """
    Bilinearly interpolate a regular lon/lat dataset to C-grid points.

    For a target point (λ, φ) inside the source cell with corners
    (λ₀, φ₀)–(λ₁, φ₁), with s = (λ − λ₀)/(λ₁ − λ₀) and r = (φ − φ₀)/(φ₁ − φ₀):

        ψ(λ, φ) = (1−s)(1−r) ψ₀₀ + s(1−r) ψ₁₀ + (1−s) r ψ₀₁ + s r ψ₁₁
                                                    (Press et al. 2007, eq. 3.6.5)

    The source may use 0–360 °E or −180–180 °E longitudes and either
    latitude ordering.  Land NaNs must be removed first with ``fill_land``.

    Parameters
    ----------
    ds : xr.Dataset
        Canonical forcing with 1-D coordinates ``lat`` [°N] and ``lon`` [°E].
        Variables without both ``lat`` and ``lon`` are passed through.
    grid : xr.Dataset
        pyOCEAN horizontal grid (``create_spherical_grid``), providing
        ``lon_{point}`` and ``lat_{point}``.
    point : {'t', 'u', 'v'}
        Target C-grid location.

    Returns
    -------
    xr.Dataset
        Same variables with (lat, lon) replaced by the point's grid
        dimensions, e.g. (time, j_t, i_u) for ``point='u'``.

    Raises
    ------
    ValueError
        If ``point`` is invalid or the source does not cover the model domain.
    """
    if point not in _POINT_DIMS:
        raise ValueError(f"point must be one of {sorted(_POINT_DIMS)}, got {point!r}")
    dims = _POINT_DIMS[point]

    src = ds.sortby("lat").sortby("lon")
    src_lon = src["lon"].values
    src_lat = src["lat"].values

    lon_t = grid[f"lon_{point}"].values
    lat_t = grid[f"lat_{point}"].values
    # Express target longitudes in the source convention (0–360 or ±180).
    lon_t = np.mod(lon_t - src_lon[0], 360.0) + src_lon[0]

    tol = 1e-9
    if (lon_t.min() < src_lon[0] - tol or lon_t.max() > src_lon[-1] + tol
            or lat_t.min() < src_lat[0] - tol or lat_t.max() > src_lat[-1] + tol):
        raise ValueError(
            "source grid does not cover the model domain: source "
            f"lon [{src_lon[0]:g}, {src_lon[-1]:g}], lat [{src_lat[0]:g}, {src_lat[-1]:g}]; "
            f"model {point}-points lon [{lon_t.min():g}, {lon_t.max():g}], "
            f"lat [{lat_t.min():g}, {lat_t.max():g}]"
        )
    # Clip round-off so that edge points are not flagged out of bounds.
    lon_t = np.clip(lon_t, src_lon[0], src_lon[-1])
    lat_t = np.clip(lat_t, src_lat[0], src_lat[-1])

    out = src.interp(
        lon=xr.DataArray(lon_t, dims=dims),
        lat=xr.DataArray(lat_t, dims=dims),
        method="linear",
    ).drop_vars(["lon", "lat"])

    for name in out.data_vars:
        if set(dims) <= set(out[name].dims):
            other = [d for d in out[name].dims if d not in dims]
            out[name] = out[name].transpose(*other, *dims)
        out[name].attrs = dict(ds[name].attrs)
    out.attrs = dict(ds.attrs)
    return out


# ---------------------------------------------------------------------------
# Sea-over-land extrapolation
# ---------------------------------------------------------------------------

def fill_land(da: xr.DataArray, lat_dim: str = "lat", lon_dim: str = "lon") -> xr.DataArray:
    """
    Replace NaNs by iterative 4-neighbour averaging (sea-over-land).

    At each sweep, every NaN cell with at least one valid neighbour
    (N, S, E, W) receives the mean of those neighbours:

        ψ[j, i] ← (1/n) Σ_{valid neighbours} ψ[j', i']

    Sweeps repeat until no further cell can be filled.  Valid (ocean)
    values are never modified.  A horizontal slice that is entirely NaN
    (e.g. a level deeper than the whole domain) is left unchanged.

    Parameters
    ----------
    da : xr.DataArray
        Field with horizontal dimensions ``lat_dim`` and ``lon_dim``; any
        other dimensions (time, depth) are treated slice by slice.
    lat_dim, lon_dim : str
        Names of the horizontal dimensions.

    Returns
    -------
    xr.DataArray
        New array with the same dims, coords and attrs.
    """
    order = da.dims
    work = da.transpose(..., lat_dim, lon_dim)
    a = np.array(work.values, dtype=float, copy=True)

    while True:
        nan = np.isnan(a)
        if not nan.any():
            break
        padded = np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)],
                        constant_values=np.nan)
        neighbours = np.stack([
            padded[..., :-2, 1:-1],   # south
            padded[..., 2:, 1:-1],    # north
            padded[..., 1:-1, :-2],   # west
            padded[..., 1:-1, 2:],    # east
        ])
        count = np.sum(~np.isnan(neighbours), axis=0)
        fillable = nan & (count > 0)
        if not fillable.any():
            break
        total = np.nansum(neighbours, axis=0)
        a[fillable] = total[fillable] / count[fillable]

    return work.copy(data=a).transpose(*order)


# ---------------------------------------------------------------------------
# Time interpolation
# ---------------------------------------------------------------------------

def interp_time(ds: xr.Dataset, t: np.datetime64) -> xr.Dataset:
    """
    Linearly interpolate forcing records to model time ``t``.

        ψ(t) = (1 − w) ψ(tₙ) + w ψ(tₙ₊₁),   w = (t − tₙ)/(tₙ₊₁ − tₙ)

    This is the same two-record linear interpolation used by MITgcm
    (``pkg/exf/exf_set_fld.F``) and HYCOM (``forfun.F90``).

    Parameters
    ----------
    ds : xr.Dataset
        Forcing with a strictly increasing ``time`` coordinate.
    t : np.datetime64
        Model time.

    Returns
    -------
    xr.Dataset
        Fields at time ``t``, without a ``time`` dimension.

    Raises
    ------
    ValueError
        If ``t`` lies outside the forcing records (no extrapolation).
    """
    times = ds["time"].values
    t = np.datetime64(t, "ns")
    if t < times[0] or t > times[-1]:
        raise ValueError(f"time {t} is outside the forcing records [{times[0]}, {times[-1]}]")

    n = int(np.searchsorted(times, t, side="right")) - 1
    if n == times.size - 1:
        return ds.isel(time=n, drop=True)

    w = float((t - times[n]) / (times[n + 1] - times[n]))
    with xr.set_options(keep_attrs=True):
        out = (1.0 - w) * ds.isel(time=n, drop=True) + w * ds.isel(time=n + 1, drop=True)
    return out
