"""
WRF-ARW ``wrfout`` atmospheric forcing adapter.

Reads raw WRF output files, such as the WRF-PACIFICO cycles that force
HYCOM-IOA (``~/HYCOM/force2ab_PAC_MEX.py``), and converts them to the
canonical atmosphere (``forcing/canonical.py``).

WRF variables used
------------------
| WRF            | Meaning                                  | Canonical | Conversion                              |
|----------------|------------------------------------------|-----------|-----------------------------------------|
| U10, V10       | 10 m wind, grid-relative [m s⁻¹]         | u10, v10  | rotate to earth with COSALPHA/SINALPHA  |
| T2             | 2 m temperature [K]                      | t2m       | none                                    |
| Q2             | 2 m water-vapour mixing ratio [kg kg⁻¹]  | q2m       | q = r / (1 + r)                         |
| PSFC           | surface pressure [Pa]                    | msl       | none (≈ sea-level pressure over ocean)  |
| SWDOWN         | downward shortwave at surface [W m⁻²]    | sw_down   | none (instantaneous)                    |
| GLW            | downward longwave at surface [W m⁻²]     | lw_down   | none (instantaneous)                    |
| RAINC + RAINNC | precipitation accumulated since the start of the cycle [mm] | precip | de-accumulate per file, centre, mm → m |
| LANDMASK       | 1 = land                                 | —         | land points set to NaN (sea-over-land)  |

Wind rotation (WRF-ARW User's Guide, "Wind rotation"; Skamarock et al. 2019):

    u_earth = U cos α − V sin α,    v_earth = V cos α + U sin α

On a Mercator grid α = 0 and the rotation is the identity.

Forecast cycles
---------------
Each file is treated as one forecast cycle: accumulated precipitation
restarts at zero in every cycle, so it is de-accumulated file by file
before the cycles are joined.  Where cycles overlap in time, the newest
cycle (latest start time) is kept, as in operational practice.  The
interval-mean rate (A_k − A_{k−1})/Δt belongs to the middle of the interval
and is interpolated back to the output times, consistent with the ERA5
adapter; at the first and last record of a cycle the nearest interval rate
is used.  The HYCOM-IOA preprocessing instead assigns the interval rate to
the end of the interval (a 30 min shift for hourly output).

Limitations
-----------
Only rectilinear grids (latitude depending only on south_north and
longitude only on west_east, i.e. MAP_PROJ = 3 Mercator or 6 lat-lon) are
supported, since ``forcing.interp.to_model_grid`` interpolates on 1-D axes.
Lambert or polar-stereographic domains raise ``NotImplementedError``.

References
----------
Skamarock, W.C. et al. (2019). A Description of the Advanced Research WRF
    Model Version 4. NCAR/TN-556+STR.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from forcing.canonical import CANONICAL_VARS
from forcing.sources.base import BBox, ForcingSource, register_source, subset_bbox

_RECTILINEAR_TOL_DEG: float = 1e-4


@register_source("wrf")
class WRFSource(ForcingSource):
    """
    WRF-ARW output (``wrfout_d0X_*``) from local NetCDF files.

    Parameters
    ----------
    paths : list of str or Path
        ``wrfout`` files, one per forecast cycle (any order).
    mask_land : bool
        Set land points (LANDMASK = 1, and LAKEMASK = 1 when present) to NaN
        so that ``forcing.interp.fill_land`` extrapolates marine air over
        land before interpolation.  Default True.
    """

    kind = "atmosphere"

    def __init__(self, paths: list[str | Path], mask_land: bool = True) -> None:
        if not paths:
            raise ValueError("WRFSource needs at least one file in paths")
        self.paths = [Path(p) for p in paths]
        self.mask_land = mask_land

    # ------------------------------------------------------------------
    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """
        Read every cycle, de-accumulate precipitation, join the cycles and
        subset in time (± one output interval) and space.
        """
        cycles = sorted((self._read_cycle(p) for p in self.paths),
                        key=lambda c: c["time"].values[0])
        ds = xr.concat(cycles, dim="time", coords="minimal", compat="override")
        # Overlapping times: keep the newest cycle (it comes last after sorting)
        keep = ~pd.Index(ds["time"].values).duplicated(keep="last")
        ds = ds.isel(time=np.nonzero(keep)[0]).sortby("time")

        dt = np.median(np.diff(ds["time"].values)) if ds.sizes["time"] > 1 else np.timedelta64(0)
        ds = ds.sel(time=slice(np.datetime64(t0, "ns") - dt, np.datetime64(t1, "ns") + dt))
        return subset_bbox(ds, bbox)

    # ------------------------------------------------------------------
    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Rename WRF fields and convert units to canonical form."""
        out = xr.Dataset(coords={k: ds[k] for k in ("time", "lat", "lon")})
        out["u10"] = ds["U10"]
        out["v10"] = ds["V10"]
        out["t2m"] = ds["T2"]
        out["q2m"] = ds["Q2"] / (1.0 + ds["Q2"])
        out["msl"] = ds["PSFC"]
        out["sw_down"] = ds["SWDOWN"]
        out["lw_down"] = ds["GLW"]
        out["precip"] = ds["PRECIP_RATE"]
        for name in out.data_vars:
            meta = CANONICAL_VARS["atmosphere"][name]
            out[name].attrs = {"units": meta["units"], "standard_name": meta["standard_name"]}
        out.attrs["source_product"] = "WRF-ARW wrfout (Skamarock et al. 2019)"
        out.attrs["source_files"] = ", ".join(p.name for p in self.paths)
        return out

    # ------------------------------------------------------------------
    def _read_cycle(self, path: Path) -> xr.Dataset:
        """One wrfout file → (time, lat, lon) dataset with earth winds and precip rate."""
        with xr.open_dataset(path, decode_times=False) as f:
            time = _parse_times(f["Times"].values)
            lat2d = f["XLAT"].isel(Time=0).values
            lon2d = f["XLONG"].isel(Time=0).values
            lat, lon = _rectilinear_axes(lat2d, lon2d, path)

            u, v = f["U10"].values, f["V10"].values
            if "COSALPHA" in f and "SINALPHA" in f:
                cosa, sina = f["COSALPHA"].values, f["SINALPHA"].values
                u, v = u * cosa - v * sina, v * cosa + u * sina

            acc = f["RAINC"].values + f["RAINNC"].values          # mm
            bucket = f.attrs.get("BUCKET_MM", 0.0)
            if bucket and bucket > 0 and "I_RAINC" in f and "I_RAINNC" in f:
                acc = acc + bucket * (f["I_RAINC"].values + f["I_RAINNC"].values)

            fields = {
                "U10": u, "V10": v,
                "T2": f["T2"].values, "Q2": f["Q2"].values, "PSFC": f["PSFC"].values,
                "SWDOWN": f["SWDOWN"].values, "GLW": f["GLW"].values,
                "PRECIP_RATE": _centred_precip_rate(acc, time),
            }
            land = None
            if self.mask_land and "LANDMASK" in f:
                land = f["LANDMASK"].isel(Time=0).values > 0.5
                if "LAKEMASK" in f:
                    land |= f["LAKEMASK"].isel(Time=0).values > 0.5

        dims = ("time", "lat", "lon")
        ds = xr.Dataset({k: (dims, np.asarray(val, dtype=float)) for k, val in fields.items()},
                        coords={"time": time, "lat": lat, "lon": lon})
        if land is not None:
            ds = ds.where(~xr.DataArray(land, dims=("lat", "lon")))
        return ds


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _parse_times(times: np.ndarray) -> np.ndarray:
    """WRF ``Times`` ('YYYY-MM-DD_HH:MM:SS', as char matrix or byte strings) → datetime64[ns]."""
    if times.ndim == 2:
        strings = [b"".join(row).decode() for row in times]
    else:
        strings = [t.decode() if isinstance(t, bytes) else str(t) for t in times]
    return pd.to_datetime(strings, format="%Y-%m-%d_%H:%M:%S").values.astype("datetime64[ns]")


def _rectilinear_axes(lat2d: np.ndarray, lon2d: np.ndarray, path: Path) -> tuple[np.ndarray, np.ndarray]:
    """1-D lat/lon axes, or NotImplementedError if the grid is not rectilinear."""
    lat = lat2d[:, 0]
    lon = lon2d[0, :]
    if (np.max(np.abs(lat2d - lat[:, None])) > _RECTILINEAR_TOL_DEG
            or np.max(np.abs(lon2d - lon[None, :])) > _RECTILINEAR_TOL_DEG):
        raise NotImplementedError(
            f"{path.name}: WRF grid is not rectilinear (e.g. Lambert projection); "
            "only Mercator / lat-lon domains are supported for now"
        )
    return lat, lon


def _centred_precip_rate(acc_mm: np.ndarray, time: np.ndarray) -> np.ndarray:
    """
    Accumulated precipitation within one cycle → rate at the output times [m s⁻¹].

        r_{k−½} = max(A_k − A_{k−1}, 0) / (1000 Δt_k)   at t_{k−½} = (t_{k−1} + t_k)/2

    then linear interpolation to t_k; first and last records take the
    nearest interval rate.
    """
    nt = acc_mm.shape[0]
    if nt < 2:
        return np.zeros_like(acc_mm)
    dt = (np.diff(time) / np.timedelta64(1, "s")).astype(float)
    rate_mid = np.maximum(np.diff(acc_mm, axis=0), 0.0) / (1000.0 * dt[:, None, None])
    t_mid = (time[:-1] + (time[1:] - time[:-1]) / 2).astype("datetime64[ns]")
    rate = xr.DataArray(rate_mid, dims=("time", "y", "x"), coords={"time": t_mid})
    out = rate.interp(time=time)
    out = out.bfill("time").ffill("time")
    return out.values
