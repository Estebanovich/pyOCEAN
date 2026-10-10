"""
HYCOM z-level NetCDF adapter: ESPC-D-V02 and GOFS 3.1 (glby0.08).

Reads the z-level ("3z") NetCDF products distributed by HYCOM.org/FNMOC
and converts them to the canonical ocean (``forcing/canonical.py``) for
initial conditions, open-boundary values and nudging.  This is the data the
HYCOM-IOA nesting scripts download (``~/HYCOM/01_download_netcdf_5nest.sh``).

Product layout (verified on the ESPC-D-V02 forecast files, 2026-10-10)
-----------------------------------------------------------------------
| File code | Variable    | Canonical | Notes                                   |
|-----------|-------------|-----------|-----------------------------------------|
| t3z       | water_temp  | temp      | standard_name sea_water_temperature     |
| s3z       | salinity    | salt      | practical salinity                      |
| u3z, v3z  | water_u/_v  | u, v      | eastward / northward [m s⁻¹]            |
| ssh       | surf_el     | ssh       | [m]                                     |

One variable per file (any number of files and lead times may be given),
NetCDF-3 classic with packed shorts, 40 depths [m, positive down],
longitudes 0–360 °E, time in hours since 2000-01-01, plus a ``tau``
(forecast lead) variable that is dropped.

Temperature
-----------
``water_temp`` is labelled ``sea_water_temperature``, i.e. *in situ*
temperature in CF terms, whereas pyOCEAN carries potential temperature θ.
By default the adapter converts with TEOS-10 (IOC et al. 2010):

    p  = p(z, φ)                    (gsw.p_from_z)
    S_A = S_A(S_P, p, λ, φ)         (gsw.SA_from_SP)
    θ  = θ(S_A, t, p, p_ref = 0)    (gsw.pt0_from_t)

The difference is ≈ 0.01 °C at 100 m, 0.1 °C at 1000 m and 0.4 °C at
4000 m.  Pass ``temperature="potential"`` if a product is known to store θ.

References
----------
Metzger, E.J. et al. (2014). US Navy operational global ocean and Arctic
    ice prediction systems. Oceanography 27(3), 32–43.
IOC, SCOR & IAPSO (2010). The international thermodynamic equation of
    seawater – 2010 (TEOS-10). Manuals and Guides 56, UNESCO.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import gsw
import numpy as np
import xarray as xr

from forcing.canonical import CANONICAL_VARS
from forcing.sources.base import BBox, ForcingSource, register_source, subset_bbox

_NAMES = {"water_temp": "temp", "salinity": "salt", "water_u": "u",
          "water_v": "v", "surf_el": "ssh"}
_TEMPERATURE_KINDS = ("in_situ", "potential")


@register_source("gofs")
@register_source("espc")
class HYCOMZSource(ForcingSource):
    """
    HYCOM z-level NetCDF (ESPC-D-V02 forecasts/analyses, GOFS 3.1).

    Parameters
    ----------
    paths : list of str or Path
        Per-variable files (t3z, s3z, u3z, v3z, ssh), any lead times.
    temperature : {'in_situ', 'potential'}
        What ``water_temp`` contains.  Default 'in_situ' (CF metadata of the
        product); in-situ values are converted to potential temperature.
    """

    kind = "ocean"

    def __init__(self, paths: list[str | Path], temperature: str = "in_situ") -> None:
        if not paths:
            raise ValueError("HYCOMZSource needs at least one file in paths")
        if temperature not in _TEMPERATURE_KINDS:
            raise ValueError(f"temperature must be one of {_TEMPERATURE_KINDS}, got {temperature!r}")
        self.paths = [Path(p) for p in paths]
        self.temperature = temperature

    # ------------------------------------------------------------------
    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """Merge the per-variable files and subset to [t0, t1] and ``bbox``."""
        files = [xr.open_dataset(p, drop_variables=["tau"]) for p in self.paths]
        try:
            ds = xr.merge(files, compat="override", join="outer")
            ds = ds.sel(time=slice(np.datetime64(t0, "ns"), np.datetime64(t1, "ns")))
            return subset_bbox(ds, bbox).load()
        finally:
            for f in files:
                f.close()

    # ------------------------------------------------------------------
    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename variables, convert longitudes to ±180 °E and temperature to θ."""
        ds = ds.assign_coords(lon=((ds["lon"] + 180.0) % 360.0) - 180.0).sortby("lon")
        out = xr.Dataset(coords={k: ds[k] for k in ds.coords if k in ("time", "depth", "lat", "lon")})
        for src, name in _NAMES.items():
            if src in ds:
                out[name] = ds[src]

        comment = "potential temperature as delivered"
        if "temp" in out and self.temperature == "in_situ":
            if "salt" not in out:
                raise ValueError("converting in-situ temperature to potential temperature "
                                 "needs the salinity (s3z) files")
            out["temp"] = _potential_temperature(out["temp"], out["salt"])
            comment = "converted from in situ water_temp with TEOS-10 gsw.pt0_from_t"

        for name in out.data_vars:
            meta = CANONICAL_VARS["ocean"][name]
            out[name].attrs = {"units": meta["units"], "standard_name": meta["standard_name"]}
        if "temp" in out:
            out["temp"].attrs["comment"] = comment
        out.attrs["source_product"] = "HYCOM z-level NetCDF (ESPC-D-V02 / GOFS 3.1)"
        out.attrs["source_files"] = ", ".join(p.name for p in self.paths)
        return out


def _potential_temperature(t: xr.DataArray, sp: xr.DataArray) -> xr.DataArray:
    """In-situ → potential temperature (p_ref = 0 dbar) with TEOS-10."""
    lat = t["lat"]
    lon = t["lon"]
    p = xr.apply_ufunc(gsw.p_from_z, -t["depth"], lat)
    sa = xr.apply_ufunc(gsw.SA_from_SP, sp, p, lon, lat)
    theta = xr.apply_ufunc(gsw.pt0_from_t, sa, t, p)
    return theta.transpose(*t.dims)
