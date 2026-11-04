"""
Quality control of hourly moored-buoy series (NDBC standard meteorological files: SST, air temperature, wind,
pressure) before they are used as truth in a benchmark.

Each sample is classified with checks that keep real ocean/atmosphere signal:

==========================  =================================================================
defect                      real signal kept
==========================  =================================================================
out of range                —
spike (≤ 2 samples)         upwelling / frontal passage: coherent over many hours, so the
                            deviation from the running median stays small
stuck sensor                calm, quantised night-time SST: identical values only for hours
                            shorter than ``stuck_hours``
offset from a reference     different representativeness (point vs grid cell): a robust,
                            record-wide tolerance; only departures far beyond it are flagged
==========================  =================================================================

The spike and stuck tests are those of ``diagnostics.sealevel_qc`` (running median + local MAD; runs of
identical values), applied without an expected tide.  Range limits follow the gross-range test of the U.S. IOOS
QARTOD manual for temperature and salinity (U.S. IOOS 2020, "Manual for real-time quality control of in-situ
temperature and salinity data", v2.1, tests 9–12) [verificar the test numbers]; NDBC's own sentinels (99, 999,
9999 by column) mark missing values (NDBC "Measurement descriptions and units",
https://www.ndbc.noaa.gov/faq/measdes.shtml).
"""
from __future__ import annotations

import gzip
from pathlib import Path

import numpy as np
import pandas as pd

from diagnostics.sealevel_qc import _MAD_TO_SIGMA, _runs, flag_flatlines, flag_spikes

# missing-value sentinel of each NDBC stdmet column (a real wind direction of 99° is valid; 999 is missing)
NDBC_MISSING = {"WDIR": 999.0, "WSPD": 99.0, "GST": 99.0, "WVHT": 99.0, "DPD": 99.0, "APD": 99.0, "MWD": 999.0,
                "PRES": 9999.0, "ATMP": 999.0, "WTMP": 999.0, "DEWP": 999.0, "VIS": 99.0, "TIDE": 99.0}


def read_ndbc_stdmet(path: str | Path) -> pd.DataFrame:
    """
    Read an NDBC historical standard-meteorological file (``<id>h<year>.txt[.gz]``).

    The file has a header line (``#YY MM DD hh mm WDIR …``), a units line (``#yr mo dy …``) and whitespace-separated
    rows; times are UTC and mark the end of the averaging period.  Column sentinels ``NDBC_MISSING`` become NaN.

    Returns
    -------
    DataFrame indexed by UTC time (datetime64[ns]) with float columns WDIR [degT], WSPD, GST [m/s], WVHT [m],
    DPD, APD [s], MWD [degT], PRES [hPa], ATMP, WTMP, DEWP [°C], VIS [nmi], TIDE [ft] (those present).
    """
    path = Path(path)
    opener = gzip.open if path.suffix == ".gz" else open
    with opener(path, "rt") as f:
        lines = f.read().splitlines()
    names = lines[0].lstrip("#").split()
    rows = [ln.split() for ln in lines if ln.strip() and not ln.startswith("#")]
    df = pd.DataFrame(rows, columns=names).apply(pd.to_numeric, errors="coerce")
    year = df.pop("YY" if "YY" in df else "YYYY")
    t = pd.to_datetime(dict(year=year, month=df.pop("MM"), day=df.pop("DD"), hour=df.pop("hh"),
                            minute=df.pop("mm")))
    df.index = pd.DatetimeIndex(t.to_numpy("datetime64[ns]"), name="time")
    for c, miss in NDBC_MISSING.items():
        if c in df:
            df[c] = df[c].astype(float).where(~np.isclose(df[c].astype(float), miss))
    return df.astype(float)


def qc_buoy_series(t: np.ndarray, x: np.ndarray, valid_range: tuple[float, float],
                   step: np.timedelta64 = np.timedelta64(1, "h"),
                   t_start: np.datetime64 | None = None, t_end: np.datetime64 | None = None,
                   spike_min: float = 0.5, spike_k: float = 6.0, stuck_hours: float = 12.0,
                   resolution: float = 1e-9, min_gap_hours: float = 3.0) -> dict:
    """
    Gross-range, spike, stuck-sensor and gap checks of one buoy variable.

    The samples are first put on a regular grid ``t_start + k·step`` (nearest grid time; several samples in one
    bin are averaged), covering [t_start, t_end) (default: first to last sample).  Spikes: |x − median₇(x)| >
    max(``spike_k`` σ, ``spike_min``) in runs of ≤ 2 samples, σ = 1.4826 MAD over 121 samples
    (``sealevel_qc.flag_spikes``).  Stuck: identical consecutive values (|Δx| < ``resolution``/2; NDBC SST is
    quantised to 0.1 °C, so the default compares exact repeats) for more than ``stuck_hours``.

    Parameters
    ----------
    t : datetime64 (n,), x : values (n,) in the variable's units, NaN where missing
    valid_range : (min, max) accepted values (e.g. SST −2…35 °C)

    Returns
    -------
    dict with t (grid), x (gridded), range, spike, stuck (bool masks on the grid), ok (finite and none of the
    above), gaps [(start, hours)] of runs of missing samples of at least ``min_gap_hours``, coverage (fraction of
    grid samples that are ok).
    """
    t = np.asarray(t, "datetime64[ns]")
    x = np.asarray(x, float)
    step = np.timedelta64(step, "ns")
    t0 = np.datetime64(t_start, "ns") if t_start is not None else t.min()
    t1 = np.datetime64(t_end, "ns") if t_end is not None else t.max() + step
    n = int(np.ceil((t1 - t0) / step))
    grid = t0 + np.arange(n) * step
    k = np.round((t - t0) / step).astype(np.int64)
    inside = (k >= 0) & (k < n) & np.isfinite(x)
    g = pd.Series(x[inside]).groupby(k[inside]).mean()
    xg = np.full(n, np.nan)
    xg[g.index.to_numpy()] = g.to_numpy()

    lo, hi = valid_range
    with np.errstate(invalid="ignore"):
        rng = np.isfinite(xg) & ((xg < lo) | (xg > hi))
    x1 = np.where(rng, np.nan, xg)
    spike = flag_spikes(x1, half_window=3, mad_window=121, k=spike_k, min_dev=spike_min, max_run=2)
    spike &= np.isfinite(x1)
    max_samples = int(np.timedelta64(int(stuck_hours * 3600), "s") / step)
    stuck = flag_flatlines(x1, resolution=resolution, max_samples=max_samples)
    ok = np.isfinite(x1) & ~spike & ~stuck

    missing = ~np.isfinite(xg)
    min_gap = min_gap_hours * 3600.0 / (step / np.timedelta64(1, "s"))
    gaps = [(grid[a], float((b - a) * step / np.timedelta64(1, "h"))) for a, b in _runs(missing) if b - a >= min_gap]
    return dict(t=grid, x=xg, range=rng, spike=spike, stuck=stuck, ok=ok, gaps=gaps, coverage=float(ok.mean()))


def flag_reference_departure(x: np.ndarray, ref: np.ndarray, k: float = 5.0, min_dev: float = 1.0) -> np.ndarray:
    """
    Flag samples far from an independent reference on the same times (e.g. GLORYS or ERA5 SST at the buoy).

    d = x − ref minus its record median (a constant bias is representativeness, not a defect); flagged where
    |d| > max(k σ, ``min_dev``), σ = 1.4826 MAD(d).  ``min_dev`` [units of x] must exceed the real point-vs-grid
    differences (upwelling fronts, diurnal warm layers): this catches calibration offsets and fouling, not
    model error.
    """
    d = np.asarray(x, float) - np.asarray(ref, float)
    d = d - np.nanmedian(d)
    sigma = _MAD_TO_SIGMA * np.nanmedian(np.abs(d))
    with np.errstate(invalid="ignore"):
        return np.abs(d) > max(k * sigma, min_dev)
