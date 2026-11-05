"""Tests of diagnostics.buoy_qc: NDBC standard-meteorological reader and QC of hourly buoy series."""
from __future__ import annotations

import gzip

import numpy as np
import pandas as pd
import pytest

from diagnostics.buoy_qc import flag_reference_departure, qc_buoy_series, read_ndbc_stdmet

STDMET = """\
#YY  MM DD hh mm WDIR WSPD GST  WVHT   DPD   APD MWD   PRES  ATMP  WTMP  DEWP  VIS  TIDE
#yr  mo dy hr mn degT m/s  m/s     m   sec   sec degT   hPa  degC  degC  degC  nmi    ft
2017 07 01 00 50  99  5.1  6.2  0.81  6.67  4.75 999 1012.3  22.1  21.4  18.0 99.0 99.00
2017 07 01 01 50 999 99.0 99.0 99.00 99.00 99.00 999 9999.0 999.0 999.0 999.0 99.0 99.00
2017 07 01 02 50 210  6.0  7.4  0.90  7.14  4.90 180 1011.8  22.0  21.5  17.9 99.0 99.00
"""


def _hourly(n: int, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Synthetic summer SST [°C]: slow warming + diurnal cycle + 0.02 °C noise, hourly."""
    t = np.datetime64("2017-07-01T00:00", "ns") + np.arange(n) * np.timedelta64(1, "h")
    hours = np.arange(n, dtype=float)
    rng = np.random.default_rng(seed)
    x = 21.0 + 2.0 * hours / n + 0.4 * np.sin(2 * np.pi * hours / 24.0) + 0.02 * rng.standard_normal(n)
    return t, x


def test_read_ndbc_stdmet_sentinels_and_time(tmp_path):
    p = tmp_path / "44025h2017.txt.gz"
    with gzip.open(p, "wt") as f:
        f.write(STDMET)
    df = read_ndbc_stdmet(p)
    assert list(df.index) == list(pd.to_datetime(["2017-07-01 00:50", "2017-07-01 01:50", "2017-07-01 02:50"]))
    assert df["WTMP"].iloc[0] == pytest.approx(21.4)
    assert df["WDIR"].iloc[0] == pytest.approx(99.0)          # 99° is a real direction, 999 is missing
    assert np.isnan(df["WDIR"].iloc[1]) and np.isnan(df["WTMP"].iloc[1]) and np.isnan(df["PRES"].iloc[1])
    assert np.isnan(df["MWD"].iloc[0]) and np.isnan(df["VIS"].iloc[0]) and np.isnan(df["TIDE"].iloc[2])
    assert df["PRES"].iloc[2] == pytest.approx(1011.8)


def test_range_spike_and_real_upwelling():
    t, x = _hourly(24 * 20)
    x[50] = 45.0                                   # outside the valid range
    x[100] += 3.0                                  # one-sample spike
    x[300:312] -= np.linspace(0.0, 5.0, 12)        # upwelling front: 5 °C in 12 h, then stays
    x[312:] -= 5.0
    q = qc_buoy_series(t, x, valid_range=(-2.0, 35.0))
    assert q["range"][50] and q["range"].sum() == 1
    assert q["spike"][100]
    assert not q["spike"][295:330].any()           # a coherent cooling is ocean signal
    assert not q["ok"][50] and not q["ok"][100] and q["ok"][305]


def test_stuck_sensor_long_run_only():
    t, x = _hourly(24 * 10)
    x[40:43] = x[40]                               # 3 h identical: allowed (calm, quantised)
    x[100:112] = x[100]                            # 12 h identical: stuck
    q = qc_buoy_series(t, x, valid_range=(-2.0, 35.0), stuck_hours=6.0)
    assert not q["stuck"][40:43].any()
    assert q["stuck"][100:112].all()


def test_gaps_and_coverage():
    t, x = _hourly(24 * 10)
    keep = np.ones(t.size, bool)
    keep[48:96] = False                            # two days missing
    q = qc_buoy_series(t[keep], x[keep], valid_range=(-2.0, 35.0), t_start=t[0], t_end=t[-1] + np.timedelta64(1, "h"))
    assert len(q["gaps"]) == 1
    start, hours = q["gaps"][0]
    assert start == t[48] and hours == pytest.approx(48.0)
    assert q["coverage"] == pytest.approx(1 - 48 / t.size)
    assert q["t"].size == t.size                   # put on the full hourly grid


def test_reference_departure_block():
    t, x = _hourly(24 * 20)
    ref = x + 0.3 + 0.2 * np.sin(np.arange(x.size) / 40.0)   # biased, smoothly different reference
    x = x.copy()
    x[200:248] += 3.0                              # two days off by 3 °C (calibration, fouling)
    bad = flag_reference_departure(x, ref)
    assert bad[200:248].all()
    assert bad.sum() == 48
