#!/usr/bin/env python
"""
Speed check of NWA12-cut runs (θ_bc diagnostic, regional chain Task 9 §14; the θ_bc = ½ near-inertial mode of the
README): for each run directory and each daily output between ``--d0`` and ``--d1``

  max|u|     largest |u|, |v| over every face and layer [m/s];
  p99 S      99th percentile of the top-layer speed at the T cells south of ``--lat`` (u, v averaged to the cell);

and, from the per-step domain max |u| of ``series.npz`` over the same days (detrended, Hann window), the period of the
largest spectral peak between 10 and 40 h and the fraction of the 10–40 h variance within ±10 % of the inertial
frequency f at ``--lat-f`` (default 38 °N, the Gulf Stream region: inertial period 19.4 h).

Usage: python validation/nwa12_cut/speed_check.py RUN_DIR [RUN_DIR …] [--d0 60.5] [--d1 123] [--lat 39.5]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.nwa12_cut import case as cs  # noqa: E402

OMEGA = 7.2921e-5


def daily(run: Path, d0: float, d1: float, lat_max: float) -> list[tuple[float, float, float]]:
    """[(day, max|u|, p99 top-layer speed south of lat_max)] of the daily output files in [d0, d1]."""
    out = []
    g = cs.CASE.inputs / "grid.nc"
    if g.exists():
        with xr.open_dataset(g) as gd:
            lat = gd["lat_t"].values
    else:
        lat = cs.grid(cs.CASE)[0]["lat_t"].values
    for f in sorted((run / "out").glob("out_*.nc")):
        if f.name.startswith("._"):
            continue
        with xr.open_dataset(f) as d:
            day = float((d["time"].values[0] - cs.CASE.t0) / np.timedelta64(1, "D"))
            if not (d0 <= day <= d1):
                continue
            u, v = d["u"].values[0], d["v"].values[0]
            us = 0.5 * (u[-1][:, 1:] + u[-1][:, :-1])
            vs = 0.5 * (v[-1][1:] + v[-1][:-1])
            sp = np.hypot(us, vs)
        sel = (lat < lat_max) & np.isfinite(sp)
        out.append((day, float(max(np.nanmax(np.abs(u)), np.nanmax(np.abs(v)))), float(np.percentile(sp[sel], 99))))
    return out


def spectrum(run: Path, d0: float, d1: float, lat_f: float) -> tuple[float, float]:
    """(period of the largest 10–40 h peak [h], fraction of the 10–40 h variance within ±10 % of f)."""
    s = np.load(run / "series.npz")
    t, x = s["t"] / 86400.0, s["umax"]
    m = (t >= d0) & (t <= d1)
    t, x = t[m], x[m]
    dt = float(np.median(np.diff(t))) * 86400.0
    x = x - np.polyval(np.polyfit(t, x, 1), t)
    p = np.abs(np.fft.rfft(x * np.hanning(x.size))) ** 2
    fr = np.fft.rfftfreq(x.size, dt)                       # [Hz]
    period_h = np.where(fr > 0, 1.0 / np.where(fr > 0, fr, 1.0) / 3600.0, np.inf)
    band = (period_h >= 10.0) & (period_h <= 40.0)
    f_in = 2.0 * OMEGA * np.sin(np.deg2rad(lat_f)) / (2.0 * np.pi)
    near = band & (np.abs(fr - f_in) <= 0.1 * f_in)
    peak = float(period_h[band][np.argmax(p[band])])
    return peak, float(p[near].sum() / p[band].sum())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="+", type=Path)
    ap.add_argument("--d0", type=float, default=60.5)
    ap.add_argument("--d1", type=float, default=123.0)
    ap.add_argument("--lat", type=float, default=39.5)
    ap.add_argument("--lat-f", type=float, default=38.0)
    a = ap.parse_args()
    rows = {r.name: {d: (mx, p) for d, mx, p in daily(r, a.d0, a.d1, a.lat)} for r in a.runs}
    days = sorted(set().union(*[set(v) for v in rows.values()]))
    print("day  " + "  ".join(f"{n[:28]:>28}" for n in rows))
    print("     " + "  ".join(f"{'max|u|':>13} {'p99 S':>14}" for _ in rows))
    for d in days:
        print(f"{d:5.1f} " + "  ".join(f"{rows[n][d][0]:13.2f} {rows[n][d][1]:14.2f}" if d in rows[n] else " " * 28
                                       for n in rows))
    for r in a.runs:
        vals = list(rows[r.name].values())
        peak, frac = spectrum(r, a.d0, a.d1, a.lat_f)
        print(f"{r.name}: max over days of max|u| {max(v[0] for v in vals):.2f} m/s, of p99 S "
              f"{max(v[1] for v in vals):.2f} m/s; umax spectrum peak {peak:.1f} h, "
              f"fraction of 10–40 h variance near f {frac:.2f}")


if __name__ == "__main__":
    main()
