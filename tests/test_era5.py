"""
Tests for forcing.sources.era5 — ERA5 single-level adapter.

Synthetic NetCDF files mimic the two layouts delivered by the Copernicus
Climate Data Store (CDS):
  * new CDS (2024+): coordinate ``valid_time``, instantaneous and accumulated
    variables split into separate files (``stepType-instant`` / ``-accum``);
  * legacy CDS: a single file with coordinate ``time``.
Both use ``latitude`` (descending) and ``longitude``.
"""

from __future__ import annotations

import subprocess
import sys
import textwrap
from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_spherical_grid
from forcing.bulk import compute_surface_fluxes
from forcing.interp import fill_land, interp_time, to_model_grid
from forcing.sources.base import BBox, get_source
from forcing.sources.era5 import ERA5Source, specific_humidity_from_dewpoint

HOUR = np.timedelta64(1, "h")
T_START = np.datetime64("2015-04-01T00", "ns")
NT = 8


# ---------------------------------------------------------------------------
# Synthetic ERA5 files
# ---------------------------------------------------------------------------

def _fields(lon: np.ndarray, lat: np.ndarray, sw_rate=None, tp_per_hour=0.0036):
    """Return dict of (dims, data) for ERA5 short names, all constant except as given."""
    shape = (NT, lat.size, lon.size)
    dims = ("t", "latitude", "longitude")
    if sw_rate is None:
        sw_rate = np.full(NT, 500.0)
    ssrd = np.broadcast_to(3600.0 * np.asarray(sw_rate)[:, None, None], shape).copy()
    return {
        "instant": {
            "u10": (dims, np.full(shape, 7.0)),
            "v10": (dims, np.full(shape, -2.0)),
            "t2m": (dims, np.full(shape, 298.0)),
            "d2m": (dims, np.full(shape, 293.15)),
            "msl": (dims, np.full(shape, 101325.0)),
        },
        "accum": {
            "ssrd": (dims, ssrd),
            "strd": (dims, np.full(shape, 3600.0 * 400.0)),
            "tp": (dims, np.full(shape, tp_per_hour)),
        },
    }


def _write(
    tmp: Path,
    layout: str = "new",
    lon: np.ndarray | None = None,
    sw_rate=None,
    tp_per_hour: float = 0.0036,
    drop: tuple[str, ...] = (),
) -> list[Path]:
    if lon is None:
        lon = np.arange(-100.0, -89.75, 0.25)
    lat = np.arange(25.0, 14.75, -0.25)                       # descending, as ERA5
    times = T_START + np.arange(NT) * HOUR
    groups = _fields(lon, lat, sw_rate, tp_per_hour)
    tname = "valid_time" if layout == "new" else "time"

    def to_ds(vars_: dict) -> xr.Dataset:
        vars_ = {k: (tuple(tname if d == "t" else d for d in v[0]), v[1])
                 for k, v in vars_.items() if k not in drop}
        return xr.Dataset(vars_, coords={tname: times, "latitude": lat, "longitude": lon})

    if layout == "new":
        paths = [tmp / "data_stream-oper_stepType-instant.nc", tmp / "data_stream-oper_stepType-accum.nc"]
        to_ds(groups["instant"]).assign_coords(number=0, expver=("valid_time", ["0001"] * NT)).to_netcdf(paths[0])
        to_ds(groups["accum"]).to_netcdf(paths[1])
        return paths
    path = tmp / "era5_legacy.nc"
    to_ds({**groups["instant"], **groups["accum"]}).to_netcdf(path)
    return [path]


BB = BBox(lon_min=-97.0, lon_max=-93.0, lat_min=18.0, lat_max=22.0)
T0 = datetime(2015, 4, 1, 1)
T1 = datetime(2015, 4, 1, 6)


# ---------------------------------------------------------------------------
# Humidity conversion
# ---------------------------------------------------------------------------

class TestSpecificHumidity:

    def test_saturation_vapour_pressure_at_20C(self):
        # WMO (2008) table: e_s(20 °C) over water = 23.39 hPa
        p = 101325.0
        q = specific_humidity_from_dewpoint(np.array(293.15), np.array(p))
        e = q * p / (0.621981 + 0.378019 * q)
        assert e == pytest.approx(2339.0, rel=2e-3)

    def test_increases_with_dewpoint(self):
        q = specific_humidity_from_dewpoint(np.array([280.0, 290.0, 300.0]), np.full(3, 1e5))
        assert np.all(np.diff(q) > 0)

    def test_decreases_with_pressure(self):
        q = specific_humidity_from_dewpoint(np.full(2, 295.0), np.array([95000.0, 105000.0]))
        assert q[0] > q[1]


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

class TestERA5Load:

    def test_registered_as_era5(self, tmp_path):
        src = get_source("era5", paths=_write(tmp_path))
        assert isinstance(src, ERA5Source)

    @pytest.mark.parametrize("layout", ["new", "legacy"])
    def test_load_gives_canonical_atmosphere(self, tmp_path, layout):
        ds = ERA5Source(_write(tmp_path, layout)).load(T0, T1, BB)
        assert set(ds.data_vars) == {"u10", "v10", "t2m", "q2m", "msl", "sw_down", "lw_down", "precip"}
        assert set(ds.coords) >= {"time", "lat", "lon"}
        assert ds.attrs["pyocean_source"] == "ERA5Source"

    def test_instantaneous_fields_pass_through(self, tmp_path):
        ds = ERA5Source(_write(tmp_path)).load(T0, T1, BB)
        np.testing.assert_allclose(ds.u10, 7.0)
        np.testing.assert_allclose(ds.t2m, 298.0)
        np.testing.assert_allclose(ds.msl, 101325.0)

    def test_q2m_from_dewpoint_and_pressure(self, tmp_path):
        ds = ERA5Source(_write(tmp_path)).load(T0, T1, BB)
        expected = specific_humidity_from_dewpoint(np.array(293.15), np.array(101325.0))
        np.testing.assert_allclose(ds.q2m, expected, rtol=1e-12)

    def test_constant_accumulations_become_rates(self, tmp_path):
        ds = ERA5Source(_write(tmp_path)).load(T0, T1, BB)
        np.testing.assert_allclose(ds.sw_down, 500.0, rtol=1e-12)
        np.testing.assert_allclose(ds.lw_down, 400.0, rtol=1e-12)
        np.testing.assert_allclose(ds.precip, 0.0036 / 3600.0, rtol=1e-12)

    def test_accumulations_centred_on_instantaneous_times(self, tmp_path):
        # ssrd at hour k accumulates over (k−1, k]; its mean rate belongs to k − 30 min.
        # For a linear rate R(t) = 100 + 50 t[h], the value at hour k must be R(k).
        hours = np.arange(NT)
        sw_mean_over_previous_hour = 100.0 + 50.0 * (hours - 0.5)
        paths = _write(tmp_path, sw_rate=sw_mean_over_previous_hour)
        ds = ERA5Source(paths).load(T0, T1, BB)
        h = (ds.time.values - T_START) / HOUR
        interior = h < NT - 1
        np.testing.assert_allclose(
            ds.sw_down.isel(lat=0, lon=0).values[interior], 100.0 + 50.0 * h[interior], rtol=1e-12
        )

    def test_negative_accumulations_clipped(self, tmp_path):
        sw = np.full(NT, 300.0)
        sw[3] = -1e-4                                   # ERA5 numerical noise at night
        ds = ERA5Source(_write(tmp_path, sw_rate=sw, tp_per_hour=-1e-9)).load(T0, T1, BB)
        assert float(ds.sw_down.min()) >= 0.0
        assert float(ds.precip.min()) >= 0.0

    def test_spatial_subset_covers_bbox(self, tmp_path):
        ds = ERA5Source(_write(tmp_path)).load(T0, T1, BB)
        assert float(ds.lon.min()) <= BB.lon_min and float(ds.lon.max()) >= BB.lon_max
        assert float(ds.lat.min()) <= BB.lat_min and float(ds.lat.max()) >= BB.lat_max
        assert ds.lon.size < 41 and ds.lat.size < 41   # actually subset

    def test_time_window_padded_by_one_record(self, tmp_path):
        ds = ERA5Source(_write(tmp_path)).load(T0, T1, BB)
        assert ds.time.values[0] == np.datetime64(T0, "ns") - HOUR
        assert ds.time.values[-1] == np.datetime64(T1, "ns") + HOUR

    def test_longitudes_0_360(self, tmp_path):
        paths = _write(tmp_path, lon=np.arange(260.0, 270.25, 0.25))
        ds = ERA5Source(paths).load(T0, T1, BB)
        assert ds.lon.size > 0
        np.testing.assert_allclose(ds.u10, 7.0)

    def test_missing_variable_reported(self, tmp_path):
        with pytest.raises(ValueError, match="d2m"):
            ERA5Source(_write(tmp_path, drop=("d2m",))).load(T0, T1, BB)

    def test_no_files_rejected(self):
        with pytest.raises(ValueError, match="paths"):
            ERA5Source([])


    def test_netcdf3_file_still_read(self, tmp_path):
        # Old CDS deliveries can be NetCDF3, which h5netcdf cannot open: the
        # adapter falls back to the netCDF4 engine for them.
        path = _write(tmp_path, layout="legacy")[0]
        nc3 = tmp_path / "era5_legacy_nc3.nc"
        with xr.open_dataset(path) as f:
            f.load().to_netcdf(nc3, format="NETCDF3_64BIT")
        ds = ERA5Source([nc3]).load(T0, T1, BB)
        np.testing.assert_allclose(ds.u10, 7.0)

    def test_time_chunks_are_concatenated(self, tmp_path):
        # Monthly CDS deliveries: instant and accum files for two consecutive time chunks (N2b task 7 found that
        # xr.merge(compat="override") kept the first chunk's u10 … and left NaN in the second).  Every variable
        # must hold the values of its own chunk.
        paths = _write(tmp_path)
        chunks = []
        for i, sl in enumerate((slice(0, 4), slice(4, NT))):
            for p in paths:
                with xr.open_dataset(p) as f:
                    part = f.isel(valid_time=sl).load()
                if "u10" in part:
                    part["u10"] = part["u10"] + 2.0 * i                       # 7 m/s, then 9 m/s
                if "ssrd" in part:
                    part["ssrd"] = part["ssrd"] * (1.0 + i)                  # 500, then 1000 W/m²
                q = tmp_path / f"chunk{i}_{p.name}"
                part.to_netcdf(q)
                chunks.append(q)
        ds = ERA5Source(chunks).load(datetime(2015, 4, 1, 0), datetime(2015, 4, 1, 7), BB)
        assert np.isfinite(ds.to_array()).all()
        u = ds.u10.isel(lat=0, lon=0).values
        np.testing.assert_allclose(u, [7.0] * 4 + [9.0] * 4)
        sw = ds.sw_down.isel(lat=0, lon=0).values
        # hour-ending accumulations centred on the stamps: the hour ending at t4 (chunk 1) is centred 30 min
        # before t4, so the stamp t3 sees half of each chunk
        np.testing.assert_allclose(sw[1:3], 500.0)
        np.testing.assert_allclose(sw[3], 750.0)
        np.testing.assert_allclose(sw[4:], 1000.0)

    def _chunks(self, tmp_path, extra_in_first: bool = False, drop_accum_second: bool = False) -> list[Path]:
        """The synthetic new-CDS files split in two time chunks (u10 = 7, then 9 m/s)."""
        out = []
        for i, sl in enumerate((slice(0, 4), slice(4, NT))):
            for p in _write(tmp_path):
                if drop_accum_second and i == 1 and "accum" in p.name:
                    continue
                with xr.open_dataset(p) as f:
                    part = f.isel(valid_time=sl).load()
                if "u10" in part:
                    part["u10"] = part["u10"] + 2.0 * i
                    if extra_in_first and i == 0:
                        part["sst"] = part["t2m"] - 1.0              # a field only the first month requested
                q = tmp_path / f"c{i}_{p.name}"
                part.to_netcdf(q)
                out.append(q)
        return out

    def test_chunks_with_different_variable_lists(self, tmp_path):
        # Review m1 of N2b task 7: one month with an extra variable formed its own group, and the merge of the
        # groups brought back NaN in the other month.  Every required variable must hold both chunks.
        ds = ERA5Source(self._chunks(tmp_path, extra_in_first=True)).load(datetime(2015, 4, 1, 0),
                                                                          datetime(2015, 4, 1, 7), BB)
        np.testing.assert_allclose(ds.u10.isel(lat=0, lon=0).values, [7.0] * 4 + [9.0] * 4)
        assert np.isfinite(ds.to_array()).all()

    def test_missing_chunk_of_a_variable_is_an_error(self, tmp_path):
        # the accumulated file of the second month is missing: no silent NaN, a clear error naming the variables
        with pytest.raises(ValueError, match="ssrd|non-finite"):
            ERA5Source(self._chunks(tmp_path, drop_accum_second=True)).load(datetime(2015, 4, 1, 0),
                                                                            datetime(2015, 4, 1, 7), BB)


# ---------------------------------------------------------------------------
# Files the user already holds open
# ---------------------------------------------------------------------------

def test_load_while_user_holds_the_files_open(tmp_path):
    """
    A user (e.g. in a notebook) holds both new-CDS files open with
    ``xr.open_dataset`` and has read from one; loading the same files with
    ``ERA5Source`` three times must not crash.  With the netCDF4 engine on both
    sides this was a SIGSEGV inside netCDF-C on the second load (xarray
    2023.6/2024.11, netCDF4 1.7.1/1.7.2, HDF5 1.14.3/1.14.4; pure netCDF4 or a
    single file did not crash), so it runs in a subprocess.
    """
    paths = _write(tmp_path)
    script = textwrap.dedent(f"""
        import sys
        sys.path.insert(0, {str(Path(__file__).resolve().parents[1])!r})
        from datetime import datetime
        import xarray as xr
        from forcing.sources.base import BBox
        from forcing.sources.era5 import ERA5Source
        paths = {[str(p) for p in paths]!r}
        held = [xr.open_dataset(p) for p in paths]
        float(held[1]["ssrd"].isel(valid_time=2).mean())
        for _ in range(3):
            ERA5Source(paths).load(datetime(2015, 4, 1, 1), datetime(2015, 4, 1, 6),
                                   BBox(-97.0, -93.0, 18.0, 22.0))
        print(float(held[1]["ssrd"].isel(valid_time=3).mean()))
    """)
    run = subprocess.run([sys.executable, "-c", script], capture_output=True, text=True, timeout=300)
    assert run.returncode == 0, f"exit {run.returncode}\n{run.stderr[-2000:]}"


# ---------------------------------------------------------------------------
# End to end: ERA5 → model grid → model time → surface fluxes
# ---------------------------------------------------------------------------

def test_era5_to_surface_fluxes(tmp_path):
    grid = create_spherical_grid(nx=12, ny=10, dlon=0.25, dlat=0.25, lon0=-96.0, lat0=19.0)
    bbox = BBox.from_grid(grid, pad=0.5)
    atm = ERA5Source(_write(tmp_path)).load(T0, T1, bbox)
    atm = xr.Dataset({k: fill_land(v) for k, v in atm.data_vars.items()}, attrs=atm.attrs)
    atm_t = interp_time(to_model_grid(atm, grid, point="t"), np.datetime64("2015-04-01T03:20"))
    sst = np.full((10, 12), 27.0)
    fluxes = compute_surface_fluxes(atm_t, sst=sst)
    assert fluxes["tau_x"].shape == (10, 12)
    for k, v in fluxes.items():
        assert np.all(np.isfinite(v)), k
    assert np.all(fluxes["tau_x"] > 0) and np.all(fluxes["tau_y"] < 0)
