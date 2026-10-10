"""
Canonical forcing contract for pyOCEAN.

Every external forcing product (ERA5, WRF, GFS, ESPC-HYCOM, CMEMS, …) is
translated by an adapter in ``forcing/sources/`` into this single,
source-independent form.  The model core only ever sees canonical data, so
adding a new product never touches the dynamics.  The design follows the
separation used by MOM6, whose ocean core receives fluxes from a coupler
instead of reading atmospheric products itself (Adcroft et al. 2019).

Conventions
-----------
* CF-1.8 standard names and SI units (unit strings in UDUNITS spelling).
* Coordinates: ``time`` (datetime64, strictly increasing), ``lat`` [°N],
  ``lon`` [°E]; ocean fields also ``depth`` [m, positive downward].
* Fluxes are positive **downward** (into the ocean), as in
  ``forcing/surface.py``.
* Accumulated fields (ERA5 ``ssrd``, ``strd``, ``tp``; GFS averages) are
  converted to instantaneous rates by the adapter, never downstream.

``valid_range`` is deliberately generous: it is not a quality-control
threshold but a guard against unit errors, which typically shift values by
orders of magnitude (K ↔ °C, J m⁻² accumulated ↔ W m⁻², m h⁻¹ ↔ m s⁻¹).

References
----------
Adcroft, A. et al. (2019). The GFDL global ocean and sea ice model OM4.0.
    J. Adv. Model. Earth Syst. 11, 3167–3211.
Eaton, B. et al. (2020). NetCDF Climate and Forecast (CF) Metadata
    Conventions, version 1.8.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

# ---------------------------------------------------------------------------
# Canonical variable tables
# ---------------------------------------------------------------------------

#: kind → variable name → metadata.
#: ``required`` variables must be present; the others are validated only
#: when an adapter provides them.
CANONICAL_VARS: dict[str, dict[str, dict]] = {
    "atmosphere": {
        "u10": {"units": "m s-1", "standard_name": "eastward_wind",
                "valid_range": (-100.0, 100.0), "required": True},
        "v10": {"units": "m s-1", "standard_name": "northward_wind",
                "valid_range": (-100.0, 100.0), "required": True},
        "t2m": {"units": "K", "standard_name": "air_temperature",
                "valid_range": (180.0, 340.0), "required": True},
        "q2m": {"units": "kg kg-1", "standard_name": "specific_humidity",
                "valid_range": (0.0, 0.05), "required": True},
        "msl": {"units": "Pa", "standard_name": "air_pressure_at_mean_sea_level",
                "valid_range": (85000.0, 110000.0), "required": True},
        "sw_down": {"units": "W m-2",
                    "standard_name": "surface_downwelling_shortwave_flux_in_air",
                    "valid_range": (0.0, 1500.0), "required": True},
        "lw_down": {"units": "W m-2",
                    "standard_name": "surface_downwelling_longwave_flux_in_air",
                    "valid_range": (50.0, 600.0), "required": True},
        # Liquid-water-equivalent rate; record rain rates are ~1e-4 m s-1.
        "precip": {"units": "m s-1", "standard_name": "lwe_precipitation_rate",
                   "valid_range": (0.0, 1e-3), "required": False},
    },
    "ocean": {
        "temp": {"units": "degC", "standard_name": "sea_water_potential_temperature",
                 "valid_range": (-3.0, 40.0), "required": True},
        "salt": {"units": "psu", "standard_name": "sea_water_salinity",
                 "valid_range": (0.0, 45.0), "required": True},
        "u": {"units": "m s-1", "standard_name": "eastward_sea_water_velocity",
              "valid_range": (-5.0, 5.0), "required": False},
        "v": {"units": "m s-1", "standard_name": "northward_sea_water_velocity",
              "valid_range": (-5.0, 5.0), "required": False},
        "ssh": {"units": "m", "standard_name": "sea_surface_height_above_geoid",
                "valid_range": (-5.0, 5.0), "required": False},
    },
}

#: kind → coordinates every canonical dataset of that kind must carry.
_REQUIRED_COORDS: dict[str, tuple[str, ...]] = {
    "atmosphere": ("time", "lat", "lon"),
    "ocean": ("time", "depth", "lat", "lon"),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def validate_canonical(ds: xr.Dataset, kind: str) -> xr.Dataset:
    """
    Check that ``ds`` obeys the canonical forcing contract for ``kind``.

    Checks, in order:
      1. ``kind`` is known.
      2. Required coordinates exist; ``time`` is strictly increasing and
         ``depth`` (ocean) is non-negative, i.e. positive downward.
      3. Required variables exist.
      4. Every canonical variable present has the canonical ``units``
         attribute, is not entirely NaN (NaN over land is allowed), and has
         finite values inside ``valid_range``.

    Parameters
    ----------
    ds : xr.Dataset
        Output of a forcing adapter.
    kind : {'atmosphere', 'ocean'}
        Forcing category.

    Returns
    -------
    xr.Dataset
        The same object, unchanged, so the call can be chained.

    Raises
    ------
    ValueError
        On the first violation found, naming the offending variable.
    """
    if kind not in CANONICAL_VARS:
        raise ValueError(
            f"unknown forcing kind {kind!r}; expected one of {sorted(CANONICAL_VARS)}"
        )

    _check_coords(ds, kind)

    for name, meta in CANONICAL_VARS[kind].items():
        if name not in ds:
            if meta["required"]:
                raise ValueError(f"{kind} forcing is missing required variable {name!r}")
            continue
        _check_variable(ds[name], name, meta)

    return ds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check_coords(ds: xr.Dataset, kind: str) -> None:
    for coord in _REQUIRED_COORDS[kind]:
        if coord not in ds.coords:
            raise ValueError(f"{kind} forcing is missing coordinate {coord!r}")

    time = ds["time"].values
    if time.size > 1 and not np.all(np.diff(time) > np.timedelta64(0, "ns")):
        raise ValueError("time must be strictly increasing")

    if "depth" in _REQUIRED_COORDS[kind] and np.any(ds["depth"].values < 0):
        raise ValueError("depth must be positive downward (all values ≥ 0 m)")


def _check_variable(da: xr.DataArray, name: str, meta: dict) -> None:
    units = da.attrs.get("units")
    if units != meta["units"]:
        raise ValueError(
            f"{name}: units {units!r} do not match canonical units {meta['units']!r}"
        )

    vmin = float(da.min(skipna=True))
    vmax = float(da.max(skipna=True))
    if np.isnan(vmin):
        raise ValueError(f"{name}: all values are NaN")

    lo, hi = meta["valid_range"]
    if vmin < lo or vmax > hi:
        raise ValueError(
            f"{name}: values [{vmin:.4g}, {vmax:.4g}] outside valid range "
            f"[{lo:g}, {hi:g}] {meta['units']} — check unit conversion"
        )
