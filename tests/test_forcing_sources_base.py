"""
Tests for forcing.sources.base — the adapter interface and source registry.

A forcing adapter translates one external product (ERA5, WRF, ESPC-HYCOM,
TPXO, …) into the canonical form.  These tests use a tiny synthetic adapter
so that no network access or real data files are needed.
"""

from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_spherical_grid
from forcing.canonical import CANONICAL_VARS
from forcing.sources.base import (
    BBox,
    ForcingSource,
    available_sources,
    get_source,
    register_source,
)


# ---------------------------------------------------------------------------
# Synthetic adapter
# ---------------------------------------------------------------------------

class _ToyAtmosphere(ForcingSource):
    """Pretends to be a product that stores air temperature in °C as 'T_air'."""

    kind = "atmosphere"

    def __init__(self, t_air_celsius: float = 27.0) -> None:
        self.t_air_celsius = t_air_celsius

    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        time = np.array([t0, t1], dtype="datetime64[ns]")
        lat = np.arange(bbox.lat_min, bbox.lat_max + 1.0, 1.0)
        lon = np.arange(bbox.lon_min, bbox.lon_max + 1.0, 1.0)
        shape = (time.size, lat.size, lon.size)
        raw = {
            "U": 5.0, "V": 0.0, "T_air": self.t_air_celsius, "Q": 0.018,
            "P": 101300.0, "SW": 500.0, "LW": 400.0,
        }
        return xr.Dataset(
            {k: (("time", "lat", "lon"), np.full(shape, v)) for k, v in raw.items()},
            coords={"time": time, "lat": lat, "lon": lon},
        )

    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        ds = ds.rename({"U": "u10", "V": "v10", "T_air": "t2m", "Q": "q2m",
                        "P": "msl", "SW": "sw_down", "LW": "lw_down"})
        ds["t2m"] = ds["t2m"] + 273.15
        for name in ds.data_vars:
            ds[name].attrs["units"] = CANONICAL_VARS["atmosphere"][name]["units"]
        return ds


class _ForgetsKelvin(_ToyAtmosphere):
    """Buggy adapter: forgets the °C → K conversion."""

    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        ds = super().to_canonical(ds)
        # Keep the (now wrong) 'K' label: the bug is in the values, not the metadata
        ds["t2m"] = (ds["t2m"] - 273.15).assign_attrs(ds["t2m"].attrs)
        return ds


T0 = datetime(2015, 4, 1, 0)
T1 = datetime(2015, 4, 1, 6)


# ---------------------------------------------------------------------------
# BBox
# ---------------------------------------------------------------------------

class TestBBox:

    def test_rejects_inverted_latitudes(self):
        with pytest.raises(ValueError, match="lat"):
            BBox(lon_min=-98.0, lon_max=-80.0, lat_min=30.0, lat_max=18.0)

    def test_rejects_inverted_longitudes(self):
        with pytest.raises(ValueError, match="lon"):
            BBox(lon_min=-80.0, lon_max=-98.0, lat_min=18.0, lat_max=30.0)

    def test_from_grid_covers_all_c_grid_points(self):
        grid = create_spherical_grid(nx=10, ny=8, dlon=1.0, dlat=1.0, lon0=-95.0, lat0=20.0)
        bb = BBox.from_grid(grid, pad=0.0)
        # U and V faces extend to the domain edges
        assert bb.lon_min == pytest.approx(-95.0)
        assert bb.lon_max == pytest.approx(-85.0)
        assert bb.lat_min == pytest.approx(20.0)
        assert bb.lat_max == pytest.approx(28.0)

    def test_from_grid_padding(self):
        grid = create_spherical_grid(nx=10, ny=8, dlon=1.0, dlat=1.0, lon0=-95.0, lat0=20.0)
        bb = BBox.from_grid(grid, pad=1.5)
        assert bb.lon_min == pytest.approx(-96.5)
        assert bb.lat_max == pytest.approx(29.5)


# ---------------------------------------------------------------------------
# ForcingSource.load
# ---------------------------------------------------------------------------

class TestForcingSourceLoad:

    def test_abstract_class_cannot_be_instantiated(self):
        with pytest.raises(TypeError):
            ForcingSource()

    def test_load_returns_canonical_dataset(self):
        bb = BBox(-98.0, -94.0, 18.0, 21.0)
        ds = _ToyAtmosphere().load(T0, T1, bb)
        assert ds["t2m"].values == pytest.approx(300.15)
        assert ds["t2m"].attrs["units"] == "K"

    def test_load_records_provenance(self):
        ds = _ToyAtmosphere().load(T0, T1, BBox(-98.0, -94.0, 18.0, 21.0))
        assert ds.attrs["pyocean_source"] == "_ToyAtmosphere"
        assert ds.attrs["pyocean_kind"] == "atmosphere"

    def test_load_validates_adapter_output(self):
        with pytest.raises(ValueError, match="t2m.*range"):
            _ForgetsKelvin().load(T0, T1, BBox(-98.0, -94.0, 18.0, 21.0))

    def test_load_rejects_inverted_time_window(self):
        with pytest.raises(ValueError, match="t0"):
            _ToyAtmosphere().load(T1, T0, BBox(-98.0, -94.0, 18.0, 21.0))


# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

class TestRegistry:

    def test_register_and_get(self):
        register_source("toy_test_registry")(_ToyAtmosphere)
        src = get_source("toy_test_registry", t_air_celsius=20.0)
        assert isinstance(src, _ToyAtmosphere)
        assert src.t_air_celsius == 20.0
        assert "toy_test_registry" in available_sources()

    def test_unknown_source_lists_available(self):
        with pytest.raises(KeyError, match="available"):
            get_source("does_not_exist")

    def test_duplicate_name_rejected(self):
        register_source("toy_test_duplicate")(_ToyAtmosphere)
        with pytest.raises(ValueError, match="already registered"):
            register_source("toy_test_duplicate")(_ForgetsKelvin)

    def test_registered_class_must_be_forcing_source(self):
        with pytest.raises(TypeError):
            register_source("toy_not_a_source")(dict)
