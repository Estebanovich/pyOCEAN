"""
ERA5 single-level atmospheric forcing adapter.

Reads hourly ERA5 reanalysis NetCDF files from the Copernicus Climate Data
Store (Hersbach et al. 2020) and converts them to the canonical atmosphere
(``forcing/canonical.py``).

ERA5 variables used (CDS short names)
-------------------------------------
| ERA5  | Meaning                               | Canonical | Conversion                         |
|-------|---------------------------------------|-----------|------------------------------------|
| u10   | 10 m eastward wind [m s⁻¹]            | u10       | none                               |
| v10   | 10 m northward wind [m s⁻¹]           | v10       | none                               |
| t2m   | 2 m temperature [K]                   | t2m       | none                               |
| d2m   | 2 m dew-point temperature [K]         | q2m       | dew point + msl → specific humidity|
| msl   | mean sea-level pressure [Pa]          | msl       | none                               |
| ssrd  | surface solar radiation down [J m⁻²]  | sw_down   | accumulation → rate, centred       |
| strd  | surface thermal radiation down [J m⁻²]| lw_down   | accumulation → rate, centred       |
| tp    | total precipitation [m]               | precip    | accumulation → rate, centred       |

Land
----
ERA5 single-level fields are defined over land (t2m, d2m, u10, ... carry the
land-surface values, not NaN), so ``forcing.interp.fill_land`` has nothing to
fill and the bilinear step mixes land values into every model cell within
one ERA5 cell (~28 km) of the coast.  The land-sea mask ``lsm`` (CDS variable
``land_sea_mask``, invariant) removes them.  The mask is applied
automatically whenever an ``lsm`` is available, from the ``lsm`` argument (a
path to an ERA5 lsm NetCDF file, single or no time, or an ``xr.DataArray``)
or from an ``lsm`` variable inside the ERA5 files; with none a
``UserWarning`` is given and the fields are used as delivered.

* Rule: a cell is land where ``lsm > lsm_threshold`` (default 0.0, so any
  land fraction counts, as CROCO).  CROCO: ``mask(mask ~= 0) = 1``
  (croco_tools ``Aforc_ERA5/make_ERA5.m:78-81``, b31b038).  ``lsm_threshold=0.5``
  gives the usual "mostly land" mask and keeps more coastal ERA5 cells.
* Variables touched: every field the adapter reads, that is u10, v10, t2m,
  d2m, msl, ssrd, strd and tp (CROCO likewise applies ``get_missing_val`` to
  every field, ``Aforc_ERA5/interp_ERA5.m``).  Masked cells are NaN and
  ``fill_land`` replaces them from the sea.
* Fill: ``fill_land`` is an iterative 4-neighbour mean; CROCO's
  ``Preprocessing_tools/get_missing_val.m`` is an objective analysis with
  decorrelation scale Roa, or a nearest-neighbour fallback.
* The default request of CROCO includes lsm
  (``Aforc_ERA5/era5_crocotools_param.py:48``), and so does
  ``validation/bulk_points/fetch_era5.py`` (default groups).

MITgcm ``pkg/exf/exf_interp.F`` has no land handling (plain bilinear
interpolation of the input field, 4 surrounding points, no mask test); the
only masking is ``EXF_FILTER_RL`` on the model grid after interpolation
(``exf_set_fld.F:236-237, 293-294``), so land values of the input do reach
coastal ocean points unless the user prepares sea-filled files.

ERA5's own evaporation ``e`` is deliberately not used: evaporation is
computed by the bulk formula so that latent heat and freshwater fluxes are
consistent (as in MITgcm, CROCO and HYCOM).

Accumulated fields
------------------
In hourly ERA5 data each accumulated value covers the hour *ending* at its
time stamp, whatever the sampling interval of the downloaded file.  Its
mean rate therefore belongs to t − 30 min.  The adapter divides by the
accumulation period and interpolates the rates linearly back onto the
instantaneous time stamps, so that all canonical variables share one time
axis.  At the last record (no later value to interpolate with) the
hour-ending rate is used.  Small negative values (numerical noise in the
IFS post-processing) are set to zero.

Supported file layouts
----------------------
* New CDS (2024+): coordinate ``valid_time``; instantaneous and accumulated
  variables in separate files; extra coordinates ``number`` and ``expver``.
* Legacy CDS: coordinate ``time``, all variables in one file.

References
----------
Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol.
    Soc. 146, 1999–2049.
ECMWF (2016). IFS Documentation Cy41r2, Part IV: Physical Processes,
    §7.2.1, eq. 7.5 (saturation vapour pressure over water; constants of
    Buck 1981).
Buck, A.L. (1981). New equations for computing vapor pressure and
    enhancement factor. J. Appl. Meteor. 20, 1527–1532.
"""

from __future__ import annotations

import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from forcing.canonical import CANONICAL_VARS
from forcing.sources.base import BBox, ForcingSource, register_source, subset_bbox

# Thermodynamic constants of the IFS (Cy41r2, Part IV, §7.2.1)
_ES_A1: float = 611.21      # Pa
_ES_A3: float = 17.502      # –
_ES_A4: float = 32.19       # K
_T_TRIPLE: float = 273.16   # K
_EPS: float = 0.621981      # R_d / R_v

_INSTANT = ("u10", "v10", "t2m", "d2m", "msl")
_ACCUM = {"ssrd": "sw_down", "strd": "lw_down", "tp": "precip"}


def specific_humidity_from_dewpoint(d2m: np.ndarray, p: np.ndarray) -> np.ndarray:
    """
    Specific humidity from dew-point temperature and pressure.

        e = a₁ exp[a₃ (T_d − T₀) / (T_d − a₄)]          (IFS Cy41r2 IV, eq. 7.5)
        q = ε e / (p − (1 − ε) e),   ε = R_d/R_v = 0.621981

    with a₁ = 611.21 Pa, a₃ = 17.502, a₄ = 32.19 K, T₀ = 273.16 K
    (saturation over water, Buck 1981).  The vapour pressure of air with
    dew point T_d equals the saturation vapour pressure at T_d.

    Parameters
    ----------
    d2m : np.ndarray
        Dew-point temperature [K].
    p : np.ndarray
        Pressure [Pa]; mean sea-level pressure is used over the ocean.

    Returns
    -------
    np.ndarray
        Specific humidity [kg kg⁻¹].
    """
    e = _ES_A1 * np.exp(_ES_A3 * (d2m - _T_TRIPLE) / (d2m - _ES_A4))
    return _EPS * e / (p - (1.0 - _EPS) * e)


def _open_era5(path: Path) -> xr.Dataset:
    """
    Open one ERA5 file with the h5netcdf engine, or netCDF4 for NetCDF3 files.

    With the netCDF4 engine, re-opening files that the user already holds open
    through ``xr.open_dataset`` (two new-CDS files, one of them read) crashed
    the process with a SIGSEGV inside netCDF-C on the second load; it did not
    depend on the xarray (2023.6, 2024.11), netCDF4 (1.7.1, 1.7.2) or HDF5
    (1.14.3, 1.14.4) version (tests/test_era5.py::
    test_load_while_user_holds_the_files_open).  h5netcdf reads the files
    through h5py, outside netCDF-C, and decodes the same values, dtypes and
    attributes (checked on the 2017-07 instant/accum/mean files).  NetCDF3
    files (some legacy CDS deliveries) are not HDF5, so they use netCDF4.
    """
    try:
        return xr.open_dataset(path, engine="h5netcdf")
    except OSError:
        return xr.open_dataset(path, engine="netcdf4")


@register_source("era5")
class ERA5Source(ForcingSource):
    """
    ERA5 hourly single-level reanalysis from local NetCDF files.

    Parameters
    ----------
    paths : list of str or Path
        ERA5 NetCDF files (any combination of instantaneous/accumulated
        files and time chunks) containing the variables listed in the
        module docstring.
    accumulation_seconds : float
        Accumulation period of ssrd, strd and tp [s]; 3600 for hourly ERA5.
    lsm : str, Path, xr.DataArray or None
        ERA5 land-sea mask (fraction of land, 0-1) as an ERA5 NetCDF file or
        a DataArray with latitude/longitude (or lat/lon) and optionally a
        single time.  If None, an ``lsm`` variable inside ``paths`` is used
        when present.  With a mask, all eight fields are set to NaN where
        lsm > ``lsm_threshold`` before ``fill_land``; with none a
        ``UserWarning`` says that coastal cells may see land values.
    lsm_threshold : float
        A cell is land where lsm > lsm_threshold [fraction, 0-1]; 0.0 (default)
        is CROCO's rule (any non-zero land fraction, ``make_ERA5.m:78-81``),
        0.5 keeps the cells that are mostly sea.
    """

    kind = "atmosphere"

    def __init__(self, paths: list[str | Path], accumulation_seconds: float = 3600.0,
                 lsm: str | Path | xr.DataArray | None = None, lsm_threshold: float = 0.0) -> None:
        if not paths:
            raise ValueError("ERA5Source needs at least one file in paths")
        self.paths = [Path(p) for p in paths]
        self.accumulation_seconds = accumulation_seconds
        self.lsm = lsm
        self.lsm_threshold = float(lsm_threshold)

    # ------------------------------------------------------------------
    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """
        Open, merge and subset the files.

        The time window is widened by one accumulation period on each side,
        so that centring accumulated fields and time interpolation at t0 and
        t1 have data on both sides.
        """
        files = [_open_era5(p) for p in self.paths]
        try:
            parts = [self._normalise(f) for f in files]
            in_file = next((p["lsm"] for p in parts if "lsm" in p.data_vars), None)
            parts = [p.drop_vars("lsm") if "lsm" in p.data_vars else p for p in parts]
            ds = self._combine(parts)

            missing = [v for v in (*_INSTANT, *_ACCUM) if v not in ds]
            if missing:
                raise ValueError(f"ERA5 files are missing variables {missing}")

            pad = np.timedelta64(int(self.accumulation_seconds), "s")
            ds = ds.sel(time=slice(np.datetime64(t0, "ns") - pad, np.datetime64(t1, "ns") + pad))
            # Subset lazily, then read only the needed block into memory
            out = subset_bbox(ds, bbox).load()
            land = self._land_mask(out, in_file)
            if land is None:
                warnings.warn("ERA5Source: no land-sea mask (lsm) given: ERA5 carries real values over land, so "
                              "ocean cells within one ERA5 cell of the coast may see land values of every field. "
                              "Pass lsm=... (CDS variable 'land_sea_mask') to mask them.", UserWarning,
                              stacklevel=3)
            else:
                for v in (*_INSTANT, *_ACCUM):
                    out[v] = out[v].where(~land)
            self._check_finite(out)
            return out
        finally:
            for f in files:
                f.close()

    # ------------------------------------------------------------------
    def _land_mask(self, out: xr.Dataset, in_file: xr.DataArray | None) -> xr.DataArray | None:
        """
        Boolean land mask (lsm > ``lsm_threshold``) on the lat/lon points of ``out``, or None.

        The mask comes from the ``lsm`` argument (path or DataArray) or else from an ``lsm`` variable of the
        ERA5 files.  A time axis is reduced to its first record (the field is invariant).  The mask is matched
        to the data grid by nearest point within 1e-3 degrees; a mismatch is an error, not a silent shift.
        """
        src = self.lsm if self.lsm is not None else in_file
        if src is None:
            return None
        if isinstance(src, (str, Path)):
            with xr.open_dataset(src, engine="h5netcdf") as f:
                name = "lsm" if "lsm" in f else next(iter(f.data_vars))
                src = f[name].load()
        src = src.rename({k: v for k, v in {"valid_time": "time", "latitude": "lat",
                                            "longitude": "lon"}.items() if k in src.dims})
        if "time" in src.dims:
            src = src.isel(time=0)
        src = src.reset_coords(drop=True)
        m = src.reindex(lat=out["lat"], lon=out["lon"], method="nearest", tolerance=1e-3)
        if m.isnull().any():
            raise ValueError("ERA5 lsm does not cover the lon/lat points of the data (it must be the same ERA5 "
                             "grid, requested with an area that contains the forcing region)")
        return m > self.lsm_threshold

    @staticmethod
    def _check_finite(out: xr.Dataset) -> None:
        """
        Refuse missing data, accept NaN over land.

        NaN that is the same at every time (the land mask, or land that the user's file leaves undefined) is
        expected and is filled by ``fill_land``.  A whole record of a variable that is NaN, or a variable with
        no finite value, means a missing file or time chunk (an outer join of monthly files leaves NaN: the
        original bug was a CROCO blow-up on day 31, task 7 review m1) and raises, as does NaN that changes
        with time.
        """
        t = out["time"].values
        for v in (*_INSTANT, *_ACCUM):
            fin = np.isfinite(out[v].values)                       # (time, lat, lon)
            if not fin.any():
                raise ValueError(f"ERA5 variable {v} has no finite value in the requested region and period")
            empty = ~fin.any(axis=(1, 2))
            if empty.any():
                raise ValueError(f"ERA5 variable {v} is entirely non-finite at {t[empty][:3]} (a time chunk of "
                                 f"this variable is missing from the files, or the record is empty)")
            if not (fin == fin[0]).all():
                raise ValueError(f"ERA5 variable {v} has NaN that changes with time (not a land mask): "
                                 f"incomplete record in [{t[0]}, {t[-1]}]")

    # ------------------------------------------------------------------
    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Convert ERA5 variables to canonical names, units and time stamps."""
        out = xr.Dataset(coords={k: ds[k] for k in ("time", "lat", "lon")})
        for name in ("u10", "v10", "t2m", "msl"):
            out[name] = ds[name]
        out["q2m"] = specific_humidity_from_dewpoint(ds["d2m"], ds["msl"])
        for era5_name, name in _ACCUM.items():
            out[name] = self._centred_rate(ds[era5_name])

        for name in out.data_vars:
            meta = CANONICAL_VARS["atmosphere"][name]
            out[name].attrs = {"units": meta["units"], "standard_name": meta["standard_name"]}
        out.attrs["source_product"] = "ERA5 hourly single levels (Hersbach et al. 2020)"
        out.attrs["source_files"] = ", ".join(p.name for p in self.paths)
        return out

    # ------------------------------------------------------------------
    @staticmethod
    def _combine(parts: list[xr.Dataset]) -> xr.Dataset:
        """
        One dataset from files split by variable group and by time chunk.

        Each variable is concatenated along time over the files that hold it
        (sorted; a stamp present in two chunks is kept once), then the
        variables are merged.  A plain ``xr.merge(compat="override")`` of all
        files keeps the first chunk of each variable and leaves NaN in the
        others (found with July + August 2017 files, N2b task 7); grouping
        files by their exact variable set still did so when one month carried
        an extra variable (task 7 review m1), hence per variable.
        """
        names = sorted({v for p in parts for v in p.data_vars})
        joined = []
        for name in names:
            g = sorted((p[[name]] for p in parts if name in p.data_vars), key=lambda d: d["time"].values[0])
            if len(g) > 1:
                c = xr.concat(g, dim="time", data_vars="minimal", coords="minimal", compat="override",
                              join="outer")
                _, first = np.unique(c["time"].values, return_index=True)
                g = [c.isel(time=np.sort(first))]
            joined.append(g[0])
        return xr.merge(joined, compat="override", join="outer")

    @staticmethod
    def _normalise(ds: xr.Dataset) -> xr.Dataset:
        """Uniform coordinate names across CDS layouts; drop ensemble/version coords."""
        ds = ds.rename({k: v for k, v in {"valid_time": "time", "latitude": "lat",
                                           "longitude": "lon"}.items() if k in ds.dims or k in ds.coords})
        return ds.drop_vars([c for c in ("number", "expver") if c in ds.coords])

    def _centred_rate(self, acc: xr.DataArray) -> xr.DataArray:
        """
        Hour-ending accumulation → rate at the instantaneous time stamps.

            r(t_k − Δ/2) = max(A(t_k), 0) / Δ,   then linear interpolation to t_k
        """
        rate = acc.clip(min=0.0) / self.accumulation_seconds
        half = np.timedelta64(int(self.accumulation_seconds * 500), "ms")
        centred = rate.assign_coords(time=rate["time"] - half).interp(time=rate["time"])
        return centred.fillna(rate)
