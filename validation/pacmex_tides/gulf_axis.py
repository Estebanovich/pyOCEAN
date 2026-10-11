#!/usr/bin/env python
"""
Gulf of California co-oscillation diagnostics for a set of tidal runs.

Along the Gulf axis (mouth 23°N → head 31.3°N) the M2 amplitude and Greenwich
phase of each run are compared with TPXO10 (which assimilates altimetry and
gauges, and is accurate here: 2–3 cm RSS against the gauges).  Summary
metrics:
  amplification  A(31.3°N) / A(23°N)        TPXO ≈ 5.0
  A_min          minimum amplitude 26–28.5°N (the amphidromic region), TPXO ≈ 12 cm
  phase rms      rms of G_run − G_TPXO along the axis north of 25°N
  gauge RSS      median 8-constituent RSS at the TICON gauges inside the Gulf

Usage:  python validation/pacmex_tides/gulf_axis.py TAG [TAG ...]
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "validation" / "tide_gauges"))
sys.path.insert(0, str(Path(__file__).parent))

from compare_nest import gauge_constants, rss  # noqa: E402
from compare_ticon_tpxo import ATLAS, TICON, tpxo_at  # noqa: E402
from diagnostics.tide_gauges import load_ticon  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import CONSTITUENTS  # noqa: E402

OUT = Path(__file__).parent / "out"
CONS = list(CONSTITUENTS)
AXIS = [(23.0, -108.3), (24.0, -109.0), (25.0, -109.8), (26.0, -110.5), (26.5, -110.85), (27.0, -111.2),
        (27.5, -111.6), (28.0, -112.0), (28.8, -112.9), (29.6, -113.5), (30.5, -114.1), (31.3, -114.5)]


def m2_at(run, lat: float, lon: float) -> complex:
    k = list(run["ha_cons"]).index("m2")
    la, lo, hc = run["lat_t"], run["lon_t"], run["hc_map"][..., k]
    d = np.hypot((lo - lon) * np.cos(np.deg2rad(lat)), la - lat)
    d = np.where(np.isfinite(hc), d, np.inf)
    j, i = np.unravel_index(np.argmin(d), d.shape)
    return hc[j, i] if d[j, i] < 0.1 else np.nan


def main() -> None:
    tags = sys.argv[1:]
    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], None, ATLAS / "grid_tpxo10atlas_v2.nc",
                   lon_range=(-116, -107), lat_range=(22, 32.5))
    tp = np.array([tpxo_at(tc, la, lo)[0][0] for la, lo in AXIS])
    ticon = load_ticon(TICON, (-115.0, -108.5), (23.0, 32.0), CONS)
    gulf = [(round(r.lat, 3), round(r.lon, 3)) for _, r in ticon.iterrows() if r.lon > -114.8]

    def metrics(h):
        a = np.abs(h)
        mid = [i for i, (la, _) in enumerate(AXIS) if 26.0 <= la <= 28.5]
        north = [i for i, (la, _) in enumerate(AXIS) if la >= 25.0]
        dg = (np.rad2deg(np.angle(tp[north]) - np.angle(h[north])) + 180) % 360 - 180   # G_run − G_TPXO
        return a[-1] / a[0], 100 * a[mid].min(), np.sqrt(np.nanmean(dg ** 2))

    amp, amin, ph = metrics(tp)
    print(f"{'run':16s} {'amplif':>7s} {'A_min':>7s} {'phase rms':>9s} {'gauge RSS':>9s}")
    print(f"{'TPXO10':16s} {amp:7.2f} {amin:6.1f}  {ph:8.1f}°")
    for t in tags:
        run = np.load(OUT / f"run_{t}.npz")
        h = np.array([m2_at(run, la, lo) for la, lo in AXIS])
        amp, amin, ph = metrics(h)
        g = gauge_constants(t, tc)
        r = []
        for key in gulf:
            if key in g:
                obs = ticon[(np.abs(ticon.lat - key[0]) < 1e-3) & (np.abs(ticon.lon - key[1]) < 1e-3)]
                r.append(100 * rss(g[key] - obs.sort_values("end").iloc[-1]["hc"]))
        print(f"{t:16s} {amp:7.2f} {amin:6.1f}  {ph:8.1f}°  {np.median(r) if r else np.nan:8.2f} cm  ({len(r)} gauges)")


if __name__ == "__main__":
    main()
