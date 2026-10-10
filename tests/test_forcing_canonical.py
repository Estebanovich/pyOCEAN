"""
Tests for forcing.canonical — the source-independent forcing contract.

Every forcing adapter (ERA5, WRF, GFS, ESPC-HYCOM, CMEMS, …) must deliver an
xarray Dataset that passes ``validate_canonical``.  These tests check that
the validator catches the classic preprocessing mistakes: missing variables,
wrong unit strings, Kelvin/Celsius confusion, accumulated fields that were
never converted to rates, and non-monotonic time axes.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from forcing.canonical import CANONICAL_VARS, validate_canonical


# ---------------------------------------------------------------------------
# Fixtures — minimal valid canonical datasets
# ---------------------------------------------------------------------------

def _coords(nt: int = 3) -> dict:
    return {
        "time": np.array(
            ["2015-04-01T00", "2015-04-01T01", "2015-04-01T02"][:nt],
            dtype="datetime64[ns]",
        ),
        "lat": np.array([18.0, 19.0, 20.0]),
        "lon": np.array([-98.0, -97.0, -96.0, -95.0]),
    }


def _atmosphere_ds() -> xr.Dataset:
    """Physically plausible tropical atmosphere, all canonical units."""
    c = _coords()
    shape = (3, 3, 4)
    values = {
        "u10": 5.0, "v10": -2.0, "t2m": 300.0, "q2m": 0.018,
        "msl": 101300.0, "sw_down": 800.0, "lw_down": 400.0, "precip": 1e-7,
    }
    ds = xr.Dataset(coords=c)
    for name, val in values.items():
        ds[name] = xr.DataArray(
            np.full(shape, val), dims=("time", "lat", "lon"),
            attrs={"units": CANONICAL_VARS["atmosphere"][name]["units"]},
        )
    return ds


def _ocean_ds() -> xr.Dataset:
    c = _coords()
    c["depth"] = np.array([0.5, 10.0, 100.0])
    shape3 = (3, 3, 3, 4)
    ds = xr.Dataset(coords=c)
    for name, val in {"temp": 25.0, "salt": 36.0, "u": 0.1, "v": -0.1}.items():
        ds[name] = xr.DataArray(
            np.full(shape3, val), dims=("time", "depth", "lat", "lon"),
            attrs={"units": CANONICAL_VARS["ocean"][name]["units"]},
        )
    ds["ssh"] = xr.DataArray(
        np.full((3, 3, 4), 0.2), dims=("time", "lat", "lon"),
        attrs={"units": CANONICAL_VARS["ocean"]["ssh"]["units"]},
    )
    return ds


# ---------------------------------------------------------------------------
# Registry content
# ---------------------------------------------------------------------------

class TestCanonicalRegistry:

    def test_atmosphere_variables_defined(self):
        expected = {"u10", "v10", "t2m", "q2m", "msl", "sw_down", "lw_down", "precip"}
        assert expected <= set(CANONICAL_VARS["atmosphere"])

    def test_ocean_variables_defined(self):
        assert {"temp", "salt", "u", "v", "ssh"} <= set(CANONICAL_VARS["ocean"])

    def test_every_variable_has_units_range_and_standard_name(self):
        for kind, table in CANONICAL_VARS.items():
            for name, meta in table.items():
                assert {"units", "standard_name", "valid_range"} <= set(meta), (kind, name)
                lo, hi = meta["valid_range"]
                assert lo < hi, (kind, name)


# ---------------------------------------------------------------------------
# Validator — accepts valid data
# ---------------------------------------------------------------------------

class TestValidateAccepts:

    def test_valid_atmosphere_passes(self):
        ds = _atmosphere_ds()
        assert validate_canonical(ds, "atmosphere") is ds

    def test_valid_ocean_passes(self):
        validate_canonical(_ocean_ds(), "ocean")

    def test_optional_variables_may_be_absent(self):
        # precip is optional: a dry-forcing experiment is legitimate
        ds = _atmosphere_ds().drop_vars("precip")
        validate_canonical(ds, "atmosphere")

    def test_nan_over_land_is_allowed(self):
        ds = _ocean_ds()
        ds["temp"][:, :, 0, 0] = np.nan
        validate_canonical(ds, "ocean")


# ---------------------------------------------------------------------------
# Validator — rejects bad data
# ---------------------------------------------------------------------------

class TestValidateRejects:

    def test_unknown_kind(self):
        with pytest.raises(ValueError, match="kind"):
            validate_canonical(_atmosphere_ds(), "volcano")

    def test_missing_required_variable(self):
        ds = _atmosphere_ds().drop_vars("t2m")
        with pytest.raises(ValueError, match="t2m"):
            validate_canonical(ds, "atmosphere")

    def test_wrong_units_string(self):
        ds = _atmosphere_ds()
        ds["msl"].attrs["units"] = "hPa"
        with pytest.raises(ValueError, match="msl.*units"):
            validate_canonical(ds, "atmosphere")

    def test_celsius_air_temperature_detected(self):
        # Classic bug: t2m converted to °C but labelled K
        ds = _atmosphere_ds()
        ds["t2m"][:] = 27.0
        with pytest.raises(ValueError, match="t2m.*range"):
            validate_canonical(ds, "atmosphere")

    def test_unconverted_accumulated_shortwave_detected(self):
        # ERA5 ssrd hourly accumulation ≈ 800 W/m² × 3600 s = 2.9e6 J/m²
        ds = _atmosphere_ds()
        ds["sw_down"][:] = 2.9e6
        with pytest.raises(ValueError, match="sw_down.*range"):
            validate_canonical(ds, "atmosphere")

    def test_precip_in_metres_per_hour_detected(self):
        # 50 mm/h left as metres per hour = 0.05 → far above any rate in m/s
        ds = _atmosphere_ds()
        ds["precip"][:] = 0.05
        with pytest.raises(ValueError, match="precip.*range"):
            validate_canonical(ds, "atmosphere")

    def test_missing_time_coordinate(self):
        ds = _atmosphere_ds().isel(time=0, drop=True)
        with pytest.raises(ValueError, match="time"):
            validate_canonical(ds, "atmosphere")

    def test_non_monotonic_time(self):
        ds = _atmosphere_ds().isel(time=[0, 2, 1])
        with pytest.raises(ValueError, match="time.*increasing"):
            validate_canonical(ds, "atmosphere")

    def test_missing_lat_lon(self):
        ds = _atmosphere_ds().rename({"lat": "latitude"})
        with pytest.raises(ValueError, match="lat"):
            validate_canonical(ds, "atmosphere")

    def test_ocean_requires_depth_positive_down(self):
        ds = _ocean_ds().assign_coords(depth=-_ocean_ds().depth)
        with pytest.raises(ValueError, match="depth"):
            validate_canonical(ds, "ocean")

    def test_all_nan_variable_rejected(self):
        ds = _atmosphere_ds()
        ds["u10"][:] = np.nan
        with pytest.raises(ValueError, match="u10.*NaN"):
            validate_canonical(ds, "atmosphere")
