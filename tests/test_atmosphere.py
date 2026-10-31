"""
Tests for forcing.atmosphere — the atmosphere provider of the bulk fluxes (N2b task 3).

The provider is a pure function of model time: record → (land fill, bilinear to
the model T points) once, then linear interpolation in time between the two
records that bracket t.  Synthetic datasets test the interpolation, purity,
coverage and land fill; the real ERA5 tests (north-east US shelf, July–August
2017, downloaded by ``validation/bulk_points/fetch_era5.py``) are skipped when
the files are not on disk.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_spherical_grid
from forcing.canonical import CANONICAL_VARS
from forcing.interp import fill_land, interp_time, to_model_grid
from forcing.sources.era5 import ERA5Source, specific_humidity_from_dewpoint

from forcing import atmosphere
from forcing.atmosphere import AtmosphereProvider

ORIGIN = np.datetime64("2017-07-01T00:00", "ns")
HOUR = np.timedelta64(1, "h")
FIELDS = ("u10", "v10", "t2m", "q2m", "msl", "sw_down", "lw_down", "precip")


# ---------------------------------------------------------------------------
# Synthetic canonical atmosphere
# ---------------------------------------------------------------------------

def _grid() -> xr.Dataset:
    return create_spherical_grid(nx=6, ny=5, dlon=0.5, dlat=0.5, lon0=-74.0, lat0=39.0)


def _synthetic(nt: int = 3, precip: bool = True, seed: int = 1,
               land: np.ndarray | None = None) -> xr.Dataset:
    """Canonical (time, lat, lon) dataset with random fields in their valid ranges."""
    rng = np.random.default_rng(seed)
    lat = np.arange(38.0, 43.01, 0.25)[::-1]          # descending, as ERA5
    lon = np.arange(-75.0, -67.99, 0.25)
    time = ORIGIN - HOUR + HOUR * np.arange(nt)        # records at −1 h, 0 h, +1 h
    shape = (nt, lat.size, lon.size)
    base = {"u10": (0.0, 8.0), "v10": (0.0, 8.0), "t2m": (293.0, 3.0),
            "q2m": (0.012, 0.002), "msl": (101300.0, 500.0), "sw_down": (400.0, 150.0),
            "lw_down": (380.0, 20.0), "precip": (2e-6, 1e-6)}
    ds = xr.Dataset(coords={"time": time, "lat": lat, "lon": lon})
    for name, (mean, sd) in base.items():
        if name == "precip" and not precip:
            continue
        a = mean + sd * rng.standard_normal(shape)
        if name in ("sw_down", "precip"):
            a = np.abs(a)
        if land is not None:
            a = np.where(land[None], np.nan, a)
        meta = CANONICAL_VARS["atmosphere"][name]
        ds[name] = (("time", "lat", "lon"),
                    a, {"units": meta["units"], "standard_name": meta["standard_name"]})
    return ds


def _reference(ds: xr.Dataset, grid: xr.Dataset, t: float) -> dict[str, np.ndarray]:
    """interp_time first, then land fill and bilinear to the model T points."""
    rec = interp_time(ds, ORIGIN + np.timedelta64(int(round(t * 1e9)), "ns"))
    filled = xr.Dataset({k: fill_land(rec[k]) for k in rec.data_vars})
    on_grid = to_model_grid(filled, grid, point="t")
    return {k: on_grid[k].values for k in on_grid.data_vars}


# ---------------------------------------------------------------------------
# Synthetic tests
# ---------------------------------------------------------------------------

def test_linear_in_time_exact():
    """provider(t) = interp_time then bilinear, to round-off (1e-14 of the field scale)."""
    ds, grid = _synthetic(), _grid()
    prov = AtmosphereProvider(ds, grid, ORIGIN)
    for t in (-3600.0, 0.0, 1800.0, 900.0, -2700.0, 3600.0):
        got = prov(t)
        ref = _reference(ds, grid, t)
        assert set(got) == set(FIELDS)
        for k in FIELDS:
            assert got[k].shape == (5, 6)
            scale = np.abs(ref[k]).max()
            np.testing.assert_allclose(got[k], ref[k], rtol=0, atol=1e-14 * scale, err_msg=k)


def test_pure_function_of_time():
    """Order of calls and a second instance do not change the result (bit for bit)."""
    ds, grid = _synthetic(nt=5), _grid()
    t1, t2 = 1234.5, 7000.0
    a = AtmosphereProvider(ds, grid, ORIGIN)
    first_t2 = a(t2)
    first_t1 = a(t1)
    again_t2 = a(t2)
    b = AtmosphereProvider(ds, grid, ORIGIN)
    b_t1 = b(t1)
    b_t2 = b(t2)
    for k in FIELDS:
        assert np.array_equal(first_t2[k], again_t2[k]), k
        assert np.array_equal(first_t2[k], b_t2[k]), k
        assert np.array_equal(first_t1[k], b_t1[k]), k
    # The returned arrays are the caller's: mutating them does not touch the cache
    again_t2["t2m"][:] = 0.0
    assert np.array_equal(a(t2)["t2m"], b_t2["t2m"])


def test_out_of_coverage_raises():
    """Outside the records: ValueError naming the dates; coverage() gives the window [s]."""
    ds, grid = _synthetic(), _grid()
    prov = AtmosphereProvider(ds, grid, ORIGIN)
    assert prov.coverage() == (-3600.0, 3600.0)
    prov(-3600.0)
    prov(3600.0)
    for t in (-3600.5, 3601.0):
        with pytest.raises(ValueError, match="2017-06-30T23:00"):
            prov(t)


def test_land_filled_before_interp():
    """NaN over land next to the coast: every model cell finite (sea-over-land first)."""
    lat = np.arange(38.0, 43.01, 0.25)[::-1]
    lon = np.arange(-75.0, -67.99, 0.25)
    land = (lon[None, :] < -72.6) & (lat[:, None] > 40.6)   # cuts through the model domain
    ds, grid = _synthetic(land=land), _grid()
    # Without the fill, coastal model points would be NaN
    raw = to_model_grid(ds.isel(time=1), grid)["t2m"].values
    assert np.isnan(raw).any() and np.isfinite(raw).any()
    out = AtmosphereProvider(ds, grid, ORIGIN)(900.0)
    for k in FIELDS:
        assert np.isfinite(out[k]).all(), k


def test_missing_precip_gives_zeros_and_warns():
    """Controller ruling R1: no precip in the source → zeros, warning at construction."""
    ds, grid = _synthetic(precip=False), _grid()
    with pytest.warns(UserWarning, match="precip"):
        prov = AtmosphereProvider(ds, grid, ORIGIN)
    out = prov(0.0)
    assert np.array_equal(out["precip"], np.zeros((5, 6)))


def test_missing_required_variable_raises():
    ds = _synthetic().drop_vars("lw_down")
    with pytest.raises(ValueError, match="lw_down"):
        AtmosphereProvider(ds, _grid(), ORIGIN)


# ---------------------------------------------------------------------------
# Real ERA5 (north-east US shelf, July–August 2017)
# ---------------------------------------------------------------------------

ERA5_DIR = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/forcing/era5")
ERA5_FILES = sorted(ERA5_DIR.glob("era5_nwa_shelf_2017-0[78]_*.nc"))
_JULY = [ERA5_DIR / f"era5_nwa_shelf_2017-07_{g}.nc" for g in ("instant", "accum", "mean")]
needs_era5 = pytest.mark.skipif(not all(p.exists() for p in _JULY),
                                reason="real ERA5 July 2017 files not on disk (fetch_era5.py)")


def _read_raw(paths: list[Path]) -> xr.Dataset:
    """Raw ERA5 files read into memory and closed (normalised coordinate names)."""
    parts = []
    for p in paths:
        with xr.open_dataset(p) as f:
            parts.append(ERA5Source._normalise(f).load())
    return xr.merge(parts, compat="override", join="outer")


def _model_files() -> list[Path]:
    return [p for p in ERA5_FILES if not p.name.endswith("_mean.nc")]


@needs_era5
def test_era5_conversions_by_hand():
    """
    One point (40 °N, 70 °W, an ERA5 node) and one hour (2017-07-15 12:00) of the
    real files, converted by hand.

    ssrd, strd and tp are accumulations over the hour *ending* at the stamp
    (ECMWF ERA5 data documentation); divided by 3600 s they are the mean rates at
    t − 30 min, and the adapter centres them back on the stamp, so the canonical
    value at t_k is the mean of the hour-ending rates of t_k and t_{k+1}:

        sw_down(t_k) = ½ [ssrd(t_k) + ssrd(t_{k+1})] / 3600     [W m⁻²]
        lw_down(t_k) = ½ [strd(t_k) + strd(t_{k+1})] / 3600     [W m⁻²]
        precip(t_k)  = ½ [tp(t_k)   + tp(t_{k+1})]   / 3600     [m s⁻¹]

    q2m from d2m and msl with the IFS saturation formula (Cy41r2 IV eq. 7.5),
    written out here and also checked against ``specific_humidity_from_dewpoint``.

    Tolerance: the files are float32 and the adapter keeps their dtype, so its
    conversions round to float32 (measured: sw_down 121.155560 vs 121.155556 by
    hand in float64, 3.3e-8 relative); rel = 1e-6 is a few float32 ulps.
    """
    stamp = np.datetime64("2017-07-15T12:00", "ns")
    raw = _read_raw(ERA5_FILES)
    pt = raw.sel(lat=40.0, lon=-70.0)
    k0 = pt.sel(time=stamp)
    k1 = pt.sel(time=stamp + HOUR)
    sw = 0.5 * (float(k0["ssrd"]) + float(k1["ssrd"])) / 3600.0
    lw = 0.5 * (float(k0["strd"]) + float(k1["strd"])) / 3600.0
    pr = 0.5 * (max(float(k0["tp"]), 0.0) + max(float(k1["tp"]), 0.0)) / 3600.0
    td, p = float(k0["d2m"]), float(k0["msl"])
    e = 611.21 * np.exp(17.502 * (td - 273.16) / (td - 32.19))
    q = 0.621981 * e / (p - (1.0 - 0.621981) * e)
    assert q == pytest.approx(float(specific_humidity_from_dewpoint(td, p)), rel=1e-15)

    # A 4×4 model grid whose first T point is that ERA5 node
    grid = create_spherical_grid(nx=4, ny=4, dlon=0.25, dlat=0.25, lon0=-70.125, lat0=39.875)
    assert grid["lon_t"].values[0, 0] == -70.0 and grid["lat_t"].values[0, 0] == 40.0
    origin = np.datetime64("2017-07-15T00:00", "ns")
    prov = atmosphere.era5_atmosphere(_model_files(), grid, origin,
                                      datetime(2017, 7, 15), datetime(2017, 7, 16), bbox=None)
    out = {k: v[0, 0] for k, v in prov(12 * 3600.0).items()}
    assert out["sw_down"] == pytest.approx(sw, rel=1e-6)
    assert out["lw_down"] == pytest.approx(lw, rel=1e-6)
    assert out["precip"] == pytest.approx(pr, rel=1e-6, abs=1e-15)
    assert out["q2m"] == pytest.approx(q, rel=1e-6)
    for name in ("u10", "v10", "t2m", "msl"):
        assert out[name] == pytest.approx(float(k0[name]), rel=1e-6, abs=1e-6), name


#: Measured on the real files (2017-07, whole 38–43 °N × 75–68 °W box): relative
#: difference of the July mean of the centred rates against the ERA5 mean fluxes,
#: (ours − ERA5)/ERA5, with July + August on disk (the last July record centred
#: with 2017-08-01 00:00).  July files alone: sw +3.48e-4, lw −4.39e-5.
MEASURED_JULY_REL_DIFF = {"sw_down": -3.64e-5, "lw_down": -4.53e-5}


@needs_era5
def test_era5_monthly_radiation_matches_mean_fluxes():
    """
    July mean of sw_down and lw_down (from the ssrd/strd accumulations) equals the
    July mean of the ERA5 mean-rate fields (avg_sdswrf/avg_sdlwrf, formerly
    msdwswrf/msdwlwrf) to 1 %.

    Both refer to the hour ending at the stamp, so ssrd/3600 and avg_sdswrf agree
    record by record; the only difference in the monthly mean is the half-hour
    centring of the accumulations (the July mean of the centred rates replaces the
    hour ending 2017-07-01 00:00 by half of it and adds half of the hour ending
    2017-08-01 00:00).
    """
    src = ERA5Source(ERA5_FILES)
    t0, t1 = datetime(2017, 7, 1), datetime(2017, 7, 31, 23)
    can = src.to_canonical(src.open(t0, t1, atmosphere.BBox(-75.0, -68.0, 38.0, 43.0)))
    july = slice(np.datetime64(t0, "ns"), np.datetime64(t1, "ns"))
    raw = _read_raw([p for p in ERA5_FILES if p.name.endswith("_mean.nc")])
    sw_name = "avg_sdswrf" if "avg_sdswrf" in raw else "msdwswrf"
    lw_name = "avg_sdlwrf" if "avg_sdlwrf" in raw else "msdwlwrf"
    rel = {}
    for name, ref_name in (("sw_down", sw_name), ("lw_down", lw_name)):
        ours = float(can[name].sel(time=july).mean())
        ref = float(raw[ref_name].sel(time=july).mean())
        rel[name] = (ours - ref) / ref
    print(f"July relative difference vs ERA5 mean fluxes: {rel}")
    assert abs(rel["sw_down"]) < 0.01
    assert abs(rel["lw_down"]) < 0.01
