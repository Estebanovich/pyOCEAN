#!/usr/bin/env python
"""
Effect of the intertidal topography (wetting and drying) on the M2 co-oscillation
of the northern Gulf of California.

Compares, north of 27.5°N, against TPXO10 (axis) and TICON-3 (gauges):
    s_min5          1/25° parent, depth floor 5 m
    n100_min5_par   1/100° nest, floor 5 m, parent boundary
    wd_floor5       1/120° GEBCO nest, floor 5 m, wet-dry solver (no drying happens)
    wd_raw          1/120° GEBCO nest, raw intertidal topography, wetting and drying (HYCOM cb drag)
    wd_raw_z0_*     same with log-law C_d(D, z0) (physics/bottom_friction.py)

Usage:  python validation/pacmex_tides/compare_wd.py [TAG ...]
Output: out/wd_summary.txt, out/wd_m2_maps.png
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
from gulf_axis import ATLAS, CONS, TICON, load_ticon, m2_at, read_tpxo, tpxo_at  # noqa: E402

OUT = Path(__file__).parent / "out"
AXIS = [(28.0, -112.0), (28.8, -112.9), (29.6, -113.5), (30.5, -114.1), (31.0, -114.4), (31.3, -114.5),
        (31.6, -114.7)]
GAUGES = [(28.96, -113.55, "Bahía de los Ángeles"), (27.92, -110.88, "Guaymas")]


def gauge_from_map(r, lat: float, lon: float) -> np.ndarray:
    """Harmonic constants of ``hc_map`` at the cell nearest to (lat, lon) that belongs to the largest
    connected body of permanently wet cells (depth > 0), so that isolated pits are never sampled."""
    wet = np.nan_to_num(r["depth"], nan=-1.0) > 0
    lab, _ = ndimage.label(wet)
    main = lab == np.bincount(lab.ravel())[1:].argmax() + 1
    dist = np.hypot(r["lat_t"] - lat, (r["lon_t"] - lon) * np.cos(np.deg2rad(lat)))
    dist[~main] = np.inf
    return r["hc_map"][np.unravel_index(dist.argmin(), dist.shape)]


def main() -> None:
    tags = sys.argv[1:] or ["s_min5", "n100_min5_par", "wd_floor5", "wd_raw", "wd_raw_z0_1e-4", "wd_raw_z0_1e-3"]
    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], None, ATLAS / "grid_tpxo10atlas_v2.nc",
                   lon_range=(-116, -108), lat_range=(24.5, 32.5))
    ticon = load_ticon(TICON, (-115.5, -109.0), (27.0, 32.0), CONS)
    tp = np.array([tpxo_at(tc, la, lo)[0][0] for la, lo in AXIS])
    lines = ["M2 along the axis, amplitude [cm] / phase error vs TPXO [°] (G_run − G_TPXO):",
             f"{'':15s}" + "".join(f"{la:>12.1f}N" for la, _ in AXIS),
             f"{'TPXO10':15s}" + "".join(f"{100 * abs(h):9.1f} {np.rad2deg(-np.angle(h)) % 360:3.0f}°" for h in tp)]
    rows = {}
    for t in tags:
        f = OUT / f"run_{t}.npz"
        if not f.exists():
            continue
        r = np.load(f)
        h = np.array([m2_at(r, la, lo) for la, lo in AXIS])
        dg = (np.rad2deg(np.angle(tp) - np.angle(h)) + 180) % 360 - 180
        rows[t] = (r, h)
        lines.append(f"{t:15s}" + "".join(f"{100 * abs(a):9.1f} {d:+4.0f}" if np.isfinite(a) else f"{'—':>14s}"
                                          for a, d in zip(h, dg)))
        ok = np.isfinite(h)
        lines.append(f"{'':15s}  amplitude rms error {100 * np.sqrt(np.mean((np.abs(h[ok]) - np.abs(tp[ok])) ** 2)):.1f} cm, "
                     f"phase rms {np.sqrt(np.mean(dg[ok] ** 2)):.1f}°, head/29.6N ratio "
                     f"{abs(h[5]) / abs(h[2]):.2f} (TPXO {abs(tp[5]) / abs(tp[2]):.2f})")
    lines.append("\nTICON gauges, M2 only (amplitude cm, phase error °, vector error |ΔM2| cm; S2/K1 from the 30-day"
                 "\n  map analysis are aliased by K2/P1, so no multi-constituent RSS here);"
                 "\n  model value at the nearest cell of the main permanently-wet water body (GEBCO 2021 has"
                 "\n  spurious 100–270 m pits inside Guaymas bay, isolated from the Gulf by never-flooded land)")
    obs = {}
    for la, lo, name in GAUGES:
        o = ticon[(np.abs(ticon.lat - la) < 0.01) & (np.abs(ticon.lon - lo) < 0.01)]
        obs[name] = o.sort_values("end").iloc[-1]["hc"] if len(o) else None
    for t, (r, _) in [("TPXO10", (None, None))] + list(rows.items()):
        parts = []
        for la, lo, name in GAUGES:
            o = obs[name]
            if o is None:
                parts.append(f"{name}: —")
                continue
            if r is None:
                h = tpxo_at(tc, la, lo)[0]
                cons = CONS
            else:
                h, cons = gauge_from_map(r, la, lo), list(r["ha_cons"])
            k = [CONS.index(c) for c in cons]
            h, o6 = np.asarray(h)[[cons.index(c) for c in cons]], o[k]
            if r is None:
                h = h[k]
            dgm = (np.rad2deg(np.angle(o6[0]) - np.angle(h[0])) + 180) % 360 - 180
            parts.append(f"{name}: {100 * abs(h[0]):.1f} (obs {100 * abs(o6[0]):.1f}), {dgm:+.0f}°, |ΔM2| {100 * abs(h[0] - o6[0]):.1f}")
        lines.append(f"  {t:15s} " + " | ".join(parts))
    if "wd_raw" in rows:
        r = rows["wd_raw"][0]
        wf = r["wet_fraction"]
        dep = r["depth"]
        inter = np.isfinite(dep) & (wf > 0.02) & (wf < 0.98)
        lines.append(f"\nwd_raw intertidal cells (wet 2–98 % of the time): {inter.sum()} "
                     f"({inter.sum() * (0.925 * 0.79):.0f} km² approx.)")
    (OUT / "wd_summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    fig, ax = plt.subplots(1, len(rows), figsize=(5 * len(rows), 5))
    for a, (t, (r, _)) in zip(np.atleast_1d(ax), rows.items()):
        k = list(r["ha_cons"]).index("m2")
        lo, la, hc = r["lon_t"], r["lat_t"], r["hc_map"][..., k]
        m = (la >= 28.0) & (lo >= -115.3) & (lo <= -110.5)
        jj, ii = np.nonzero(m.any(axis=1))[0], np.nonzero(m.any(axis=0))[0]
        s = (slice(jj[0], jj[-1] + 1), slice(ii[0], ii[-1] + 1))
        c = a.pcolormesh(lo[s], la[s], 100 * np.abs(hc[s]), vmin=0, vmax=200, shading="auto")
        a.contour(lo[s], la[s], np.rad2deg(-np.angle(hc[s])) % 360, levels=np.arange(0, 360, 20), colors="w",
                  linewidths=0.4)
        a.set_title(t, fontsize=9)
        a.set_aspect("equal")
    fig.colorbar(c, ax=np.atleast_1d(ax).tolist(), shrink=0.7, label="M2 amplitude [cm]")
    fig.savefig(OUT / "wd_m2_maps.png", dpi=90)
    plt.close(fig)


if __name__ == "__main__":
    main()
