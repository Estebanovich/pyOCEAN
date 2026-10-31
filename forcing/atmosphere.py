"""
Atmosphere provider for the bulk surface fluxes (N2b).

``AtmosphereProvider(t)`` returns the canonical near-surface atmosphere
(``forcing/canonical.py``) on the model T points at model time
``origin + t``.  The bulk-flux component calls it once per step at
t^{n+½}, so it must be

* **a pure function of time** (spec decision 6): the result depends only on
  ``t`` and on the source data, never on the order or history of calls, so
  two instances, a restarted run and a run without restart give the same
  arrays bit for bit;
* **cheap on repeated calls** inside one record interval: each source
  record is taken to the model grid once and the two records that bracket
  ``t`` are cached.

Algorithm
---------
For a record k of the source (on its own lon/lat grid):

1. sea-over-land: ``forcing.interp.fill_land`` replaces land NaNs by the
   iterative 4-neighbour mean (HYCOM-tools ``landfill``, croco_tools
   ``fillmask``).  It acts only where the source is NaN over land: WRF is
   masked with LANDMASK, ERA5 only if a land-sea mask ``lsm`` is given
   (``era5_atmosphere(..., lsm=...)``; land = lsm > ``lsm_threshold``,
   default 0 as CROCO ``make_ERA5.m:78-81``; all eight ERA5 fields u10, v10,
   t2m, d2m, msl, ssrd, strd and tp are masked).  ERA5 carries real values
   over land, so without ``lsm`` model cells within one ERA5 cell of the
   coast are interpolated from land values (a warning is given);
2. bilinear interpolation to the T points, ``forcing.interp.to_model_grid``
   (Press et al. 2007, eq. 3.6.5).

Then, between the bracketing records t_n ≤ t ≤ t_{n+1} (seconds from
``origin``):

    ψ(t) = (1 − w) ψ_n + w ψ_{n+1},    w = (t − t_n)/(t_{n+1} − t_n)

the two-record linear interpolation of MITgcm ``pkg/exf/exf_set_fld.F`` and
HYCOM ``forfun.F90`` (``forcing.interp.interp_time``).  Both steps are
linear, so the result equals ``interp_time`` followed by the spatial steps
up to round-off.  At t = t_last the last record is returned; outside
[t_first, t_last] a ``ValueError`` names the dates (no extrapolation).

Output fields (all (ny, nx) float64 at the T points)
----------------------------------------------------
    u10, v10   10 m wind components [m s⁻¹]
    t2m        2 m air temperature [K]
    q2m        2 m specific humidity [kg kg⁻¹]
    msl        mean sea-level pressure [Pa]
    sw_down    downwelling shortwave at the surface [W m⁻²]
    lw_down    downwelling longwave at the surface [W m⁻²]
    precip     precipitation, liquid-water-equivalent rate [m s⁻¹]

``precip`` is always returned (controller ruling R1 of N2b): a source
without precipitation gives zeros, with a warning at construction.

The conversions of each product (ERA5 accumulations → centred rates,
dew point → q2m; WRF de-accumulation, mixing ratio → q2m) are done by the
adapters in ``forcing/sources/``; this module never touches units.

References
----------
Press, W.H. et al. (2007). Numerical Recipes, 3rd ed., §3.6.
Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol.
    Soc. 146, 1999–2049.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from forcing.interp import fill_land, to_model_grid
from forcing.sources.base import BBox
from forcing.sources.era5 import ERA5Source
from forcing.sources.wrf import WRFSource

#: Canonical fields returned by the provider, in this order.
FIELDS: tuple[str, ...] = ("u10", "v10", "t2m", "q2m", "msl", "sw_down", "lw_down", "precip")
_REQUIRED: tuple[str, ...] = FIELDS[:-1]

__all__ = ["AtmosphereProvider", "BBox", "FIELDS", "era5_atmosphere", "wrf_atmosphere"]


class AtmosphereProvider:
    """
    Canonical atmosphere on the model T points as a pure function of time.

    Parameters
    ----------
    source_ds : xr.Dataset
        Canonical atmosphere (``forcing/canonical.py``) on its original
        regular lon/lat grid: coordinates ``time`` (strictly increasing),
        ``lat`` [°N], ``lon`` [°E]; variables u10, v10, t2m, q2m, msl,
        sw_down, lw_down [, precip].  May be lazy: records are read one at a
        time.
    grid : xr.Dataset
        pyOCEAN horizontal grid with ``lon_t``, ``lat_t`` (ny, nx).
    origin : np.datetime64
        Model time origin; ``t`` is in seconds from it.
    window : (np.datetime64, np.datetime64), optional
        The period the run asked for.  ``coverage()`` and the times accepted are trimmed to it (N2b task 4,
        Task 3 review m2): the ERA5 and WRF adapters load one record of margin on each side, and past t1 the
        hour-ending rates (SW, LW, precipitation) of that margin record are centred 30 min after t1, so it serves
        only the interpolation at t1, never a step beyond it.  None (default): the records themselves.

    Raises
    ------
    ValueError
        If a required variable is missing or the source has no records.
    """

    def __init__(self, source_ds: xr.Dataset, grid: xr.Dataset, origin: np.datetime64,
                 window: tuple[np.datetime64, np.datetime64] | None = None) -> None:
        missing = [v for v in _REQUIRED if v not in source_ds]
        if missing:
            raise ValueError(f"atmosphere source is missing canonical variables {missing}")
        if source_ds.sizes.get("time", 0) < 1:
            raise ValueError("atmosphere source has no time records")
        self._has_precip = "precip" in source_ds
        if not self._has_precip:
            warnings.warn("atmosphere source has no 'precip': the provider returns "
                          "precip = 0 m s-1 (no rain freshwater flux)", UserWarning, stacklevel=2)

        names = [v for v in FIELDS if v in source_ds]
        self._src = source_ds[names]
        self._grid = grid
        self._origin = np.datetime64(origin, "ns")
        self._times = source_ds["time"].values.astype("datetime64[ns]")
        if self._times.size > 1 and not np.all(np.diff(self._times) > np.timedelta64(0, "ns")):
            raise ValueError("atmosphere source time must be strictly increasing")
        # Record times in seconds from origin (exact: integer ns → float s)
        self._t = (self._times - self._origin).astype("timedelta64[ns]").astype(np.int64) / 1e9
        self._shape = grid["lon_t"].shape
        self._cache: dict[int, dict[str, np.ndarray]] = {}
        lo, hi = float(self._t[0]), float(self._t[-1])
        if window is not None:
            w0, w1 = ((np.datetime64(w, "ns") - self._origin).astype(np.int64) / 1e9 for w in window)
            lo, hi = max(lo, w0), min(hi, w1)
            if lo > hi:
                raise ValueError(f"atmosphere records [{self._times[0]}, {self._times[-1]}] do not overlap the "
                                 f"window [{window[0]}, {window[1]}]")
        self._cov = (lo, hi)

    # ------------------------------------------------------------------
    @property
    def origin(self) -> np.datetime64:
        """Model time origin (``t`` = 0) [datetime64[ns]]."""
        return self._origin

    def coverage(self) -> tuple[float, float]:
        """Times the provider serves [s from ``origin``]: the first and last records, trimmed to ``window``."""
        return self._cov

    # ------------------------------------------------------------------
    def __call__(self, t: float) -> dict[str, np.ndarray]:
        """
        Atmosphere at model time ``origin + t``.

        Parameters
        ----------
        t : float
            Model time [s from ``origin``].

        Returns
        -------
        dict[str, np.ndarray]
            ``FIELDS`` → (ny, nx) float64 arrays at the T points (new arrays,
            owned by the caller).

        Raises
        ------
        ValueError
            If ``t`` is outside the source records.
        """
        t = float(t)
        if not (self._cov[0] <= t <= self._cov[1]):
            when = self._origin + np.timedelta64(int(round(t * 1e9)), "ns")
            c0, c1 = (self._origin + np.timedelta64(int(round(c * 1e9)), "ns") for c in self._cov)
            raise ValueError(f"time {when} (t = {t} s) is outside the atmosphere records [{c0}, {c1}]")
        n = int(np.searchsorted(self._t, t, side="right")) - 1
        if n == self._t.size - 1:
            last = self._record(n)
            self._evict({n})
            return {k: v.copy() for k, v in last.items()}

        a, b = self._record(n), self._record(n + 1)
        self._evict({n, n + 1})
        w = (t - self._t[n]) / (self._t[n + 1] - self._t[n])
        # ψ(t) = (1 − w) ψ_n + w ψ_{n+1}   (MITgcm exf_set_fld.F, HYCOM forfun.F90)
        return {k: (1.0 - w) * a[k] + w * b[k] for k in FIELDS}

    # ------------------------------------------------------------------
    def _record(self, n: int) -> dict[str, np.ndarray]:
        """Record n on the model T points: land fill, then bilinear (cached)."""
        if n not in self._cache:
            rec = self._src.isel(time=n, drop=True).load()
            filled = xr.Dataset({k: fill_land(rec[k]) for k in rec.data_vars})
            on_grid = to_model_grid(filled, self._grid, point="t")
            out = {k: np.ascontiguousarray(on_grid[k].values, dtype=np.float64)
                   for k in on_grid.data_vars}
            if not self._has_precip:
                out["precip"] = np.zeros(self._shape)
            self._cache[n] = out
        return self._cache[n]

    def _evict(self, keep: set[int]) -> None:
        for k in [k for k in self._cache if k not in keep]:
            del self._cache[k]


# ---------------------------------------------------------------------------
# Product constructors
# ---------------------------------------------------------------------------

def era5_atmosphere(paths: list[str | Path], grid: xr.Dataset, origin: np.datetime64,
                    t0: datetime, t1: datetime, bbox: BBox | None = None,
                    lsm: str | Path | xr.DataArray | None = None,
                    lsm_threshold: float = 0.0) -> AtmosphereProvider:
    """
    Atmosphere provider from hourly ERA5 single-level files.

    ``forcing.sources.era5.ERA5Source`` reads the files and converts them:
    accumulations ssrd, strd [J m⁻²] and tp [m] over the hour ending at the
    stamp (ECMWF ERA5 data documentation) → rates /3600 s centred on the
    stamps; d2m and msl → q2m (IFS Cy41r2 IV eq. 7.5).

    Parameters
    ----------
    paths : list of str or Path
        ERA5 NetCDF files (instantaneous and accumulated, any split).
    grid : xr.Dataset
        pyOCEAN horizontal grid.
    origin : np.datetime64
        Model time origin.
    t0, t1 : datetime
        Period to load (one record of margin is added by the adapter); the provider's coverage is trimmed
        to it (``window``).
    bbox : BBox, optional
        Region to read; default ``BBox.from_grid(grid)`` (1° margin).
    lsm : str, Path, xr.DataArray or None
        ERA5 land-sea mask (CDS variable ``land_sea_mask``, named ``lsm`` in the file; a path or a DataArray on
        the same grid as the data).  Where lsm > ``lsm_threshold`` all eight ERA5 fields (u10, v10, t2m, d2m,
        msl, ssrd, strd, tp) are set to NaN before the sea-over-land fill, so coastal ocean cells see only
        ocean values.  The mask is applied automatically whenever one is available; an ``lsm`` variable
        inside ``paths`` is used if this is None.  Without any mask the fields are used as delivered (they
        hold land values) and a ``UserWarning`` says so.  Reference practice: CROCO croco_tools masks every
        field with any lsm != 0 (``Aforc_ERA5/make_ERA5.m:78-81``, ``interp_ERA5.m``) and fills it with
        ``Preprocessing_tools/get_missing_val.m`` (objective analysis with decorrelation scale Roa, or a
        nearest-neighbour fallback; ours: iterative 4-neighbour mean); MITgcm ``exf_set_fld.F:236-237,
        293-294`` does no land masking.
    lsm_threshold : float
        Land where lsm > lsm_threshold [0-1]; default 0.0 (CROCO), 0.5 keeps mostly-sea coastal cells.
    """
    bbox = BBox.from_grid(grid) if bbox is None else bbox
    ds = ERA5Source(paths, lsm=lsm, lsm_threshold=lsm_threshold).load(t0, t1, bbox)
    return AtmosphereProvider(ds, grid, origin, window=(np.datetime64(t0, "ns"), np.datetime64(t1, "ns")))


def wrf_atmosphere(paths: list[str | Path], grid: xr.Dataset, origin: np.datetime64,
                   t0: datetime, t1: datetime) -> AtmosphereProvider:
    """
    Atmosphere provider from WRF-ARW ``wrfout`` files (rectilinear Mercator
    or lat-lon grids), read and converted by ``forcing.sources.wrf.WRFSource``
    (land masked so it is filled sea-over-land; precipitation de-accumulated
    per cycle; mixing ratio → specific humidity).
    """
    ds = WRFSource(paths).load(t0, t1, BBox.from_grid(grid))
    return AtmosphereProvider(ds, grid, origin, window=(np.datetime64(t0, "ns"), np.datetime64(t1, "ns")))
