"""
GLORYS12 ocean reanalysis adapter (CMEMS ``GLOBAL_MULTIYEAR_PHY_001_030`` / ``GLOBAL_ANALYSISFORECAST_PHY_001_024``).

Reads the NetCDF subsets written by ``copernicusmarine subset`` (see ``scripts/fetch_glorys.py``) and converts
them to the canonical ocean (``forcing/canonical.py``) for initial conditions, open-boundary exteriors and
nudging, the data CEFI uses for the NWA12 configuration (``write_glorys_initial.py``,
``write_glorys_boundary.py``; ``docs/design/regional_refs/cefi_tools.md`` §1, §3).

Product layout (checked 2026-11-01 on ``glorys12_my_20170429_20170903_box.nc``, product
``cmems_mod_glo_phy_my_0.083deg_P1D-m`` v202311)
------------------------------------------------------------------------------------------------------------
| Variable | Canonical | Notes                                                                           |
|----------|-----------|---------------------------------------------------------------------------------|
| thetao   | temp      | potential temperature θ [°C]                                                    |
| so       | salt      | practical salinity S_P [PSU]                                                    |
| uo, vo   | u, v      | eastward / northward velocity [m s⁻¹]                                           |
| zos      | ssh       | sea surface height above geoid [m]                                              |

50 levels, ``depth`` [m] positive down (0.494 … 5728 m), 1/12° ``latitude``/``longitude`` [°N, °E, −180…180],
daily means.  Land and below-bed values are NaN (about half of the subset).  The data are θ and S_P, i.e.
``ts_kind = "theta_sp"`` (pass it to ``initial_state``/``boundary_data``; a model with ``eos='teos10'``
converts once).

Missing values
--------------
Land and below-bed NaN are filled with the **nearest ocean node of the same level**, as CEFI's
``fill_glorys_nn.sh`` (``cdo setmisstonn``, CEFI-tools ``boundary/fill_glorys_nn.sh:32``): ``fill_nearest`` in
``forcing/interp.py``.  Levels that are NaN everywhere stay NaN and are handled by the vertical fill of
``to_model_levels`` / ``ocean_init.source_layers``.

Time
----
``open`` keeps the records bracketing [t0, t1] (the last at or before t0 and the first at or after t1), so that
time interpolation is defined on the whole window.  A window not inside the files raises ``ValueError`` naming the
coverage.

References
----------
Lellouche, J.-M. et al. (2021). The Copernicus Global 1/12° Oceanic and Sea Ice GLORYS12 Reanalysis. Front. Earth
    Sci. 9, 698876, doi:10.3389/feart.2021.698876 [verificar].
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from forcing.canonical import CANONICAL_VARS
from forcing.interp import fill_nearest
from forcing.sources.base import BBox, ForcingSource, register_source, subset_bbox

_NAMES = {"thetao": "temp", "so": "salt", "uo": "u", "vo": "v", "zos": "ssh"}


@register_source("glorys")
class GLORYSSource(ForcingSource):
    """
    GLORYS12 NetCDF subsets (CMEMS), one or several files (any split in time or variables).

    Parameters
    ----------
    paths : list of str or Path
        Files with ``thetao``, ``so`` (and optionally ``uo``, ``vo``, ``zos``) on
        (time, depth, latitude, longitude).
    """

    kind = "ocean"
    #: Convention of ``temp`` and ``salt``: potential temperature θ [°C] and practical salinity S_P.
    ts_kind = "theta_sp"

    def __init__(self, paths: list[str | Path]) -> None:
        if not paths:
            raise ValueError("GLORYSSource needs at least one file in paths")
        self.paths = [Path(p) for p in paths]

    # ------------------------------------------------------------------
    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """Merge the files, keep the records bracketing [t0, t1] and the cells covering ``bbox``."""
        files = [xr.open_dataset(p) for p in self.paths]
        try:
            ds = xr.combine_by_coords(files, combine_attrs="override", join="outer", compat="override")
            ds = ds.rename({"latitude": "lat", "longitude": "lon"}) if "latitude" in ds.coords else ds
            ds = ds.sortby("time")
            times = ds["time"].values
            a, b = np.datetime64(t0, "ns"), np.datetime64(t1, "ns")
            if a < times[0] or b > times[-1]:
                raise ValueError(f"requested window [{a}, {b}] is outside the GLORYS files, which cover "
                                 f"[{times[0]}, {times[-1]}]")
            n0 = int(np.searchsorted(times, a, side="right")) - 1
            n1 = int(np.searchsorted(times, b, side="left"))
            ds = ds.isel(time=slice(n0, n1 + 1))
            return subset_bbox(ds, bbox).load()
        finally:
            for f in files:
                f.close()

    # ------------------------------------------------------------------
    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename, fill land/below-bed NaN with the nearest ocean node of each level, float64, lat increasing."""
        ds = ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0).sortby(["lat", "lon"])
        out = xr.Dataset(coords={"time": ds["time"].values,
                                 "depth": ds["depth"].values.astype(float),
                                 "lat": ds["lat"].values.astype(float),
                                 "lon": ds["lon"].values.astype(float)})
        for src, name in _NAMES.items():
            if src in ds:
                da = ds[src].astype(float).assign_coords(depth=out["depth"]) if "depth" in ds[src].dims \
                    else ds[src].astype(float)
                out[name] = fill_nearest(da.assign_coords(lat=out["lat"], lon=out["lon"]))
                meta = CANONICAL_VARS["ocean"][name]
                out[name].attrs = {"units": meta["units"], "standard_name": meta["standard_name"]}
        out["temp"].attrs["comment"] = "GLORYS thetao (potential temperature); NaN filled by nearest ocean node"
        out.attrs["source_product"] = ds.attrs.get("title", "GLORYS12 (CMEMS)")
        out.attrs["source_files"] = ", ".join(p.name for p in self.paths)
        return out
