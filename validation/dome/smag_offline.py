#!/usr/bin/env python
"""
Offline reconstruction of the Smagorinsky viscosity on the daily records of a DOME run (spec 2026-10-30, criterion 5).

ν = max(ah, C Δ² |S|) is recomputed with ``core.dynamics.hor_visc.smagorinsky_numpy`` from the recorded u, v, h.
**Approximate:** the series does not hold the model's face thicknesses, so h_u, h_v are the mean of the two cells
(no clipping at the shallower bed) and the wet masks are the land masks of every layer.  The counts of points where
C Δ² |S| exceeds the stability bound therefore differ from the model's ``n_visc_limited``, which also counts points
next to clipped or vanished faces (there the bound × h_min/h falls below ah).

For days 10, 30 and 50 it prints, over the h points of layers ≥ 1 m: how many points the bound limits, the fraction
with C Δ² |S| > ah, and the median, 99th percentile and maximum of C Δ² |S| [m² s⁻¹].

Usage: python validation/dome/smag_offline.py pyocean_hybrid_gls_smag0.15 [more run names]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.hor_visc import smag_geometry, smagorinsky_numpy  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from validation.dome import cases, series  # noqa: E402

DAYS = (10, 30, 50)


def masks(c: cases.Case) -> tuple[np.ndarray, ...]:
    """Wet u/v faces, active corners and (no) open faces of the 10-km grid, from the land mask of H."""
    wet = np.isfinite(c.H)
    wu = np.zeros((c.ny, c.nx + 1), bool)
    wu[:, 1:-1] = wet[:, 1:] & wet[:, :-1]
    wv = np.zeros((c.ny + 1, c.nx), bool)
    wv[1:-1] = wet[1:] & wet[:-1]
    mq = np.zeros((c.ny + 1, c.nx + 1), bool)
    mq[1:-1, 1:-1] = wu[:-1, 1:-1] & wu[1:, 1:-1] & wv[1:-1, :-1] & wv[1:-1, 1:]
    return wet, wu, wv, mq, np.zeros_like(wu), np.zeros_like(wv)


def snapshot(s: dict, n: int, c: cases.Case, geom: dict, C: float, dt: float) -> dict:
    """Statistics of ν at record n over the h points of layers ≥ 1 m (approximate face thicknesses)."""
    wet, wu, wv, mq, ou, ov = masks(c)
    h, u, v = np.nan_to_num(s["h"][n]), np.nan_to_num(s["u"][n]), np.nan_to_num(s["v"][n])
    hu = np.zeros_like(u)
    hu[..., 1:-1] = 0.5 * (h[..., 1:] + h[..., :-1])
    hv = np.zeros_like(v)
    hv[:, 1:-1] = 0.5 * (h[:, 1:] + h[:, :-1])
    nu = smagorinsky_numpy(u, v, h, hu, hv, wu, wv, mq, ou, ov, geom, C, c.nu_h, dt)[2]
    raw = smagorinsky_numpy(u, v, h, hu, hv, wu, wv, mq, ou, ov, geom, C, c.nu_h, 1e-9)[2]   # bound ≫ ν
    thick = wet[None] & (h >= 1.0)
    return dict(limited=int((thick & (raw > nu + 1e-12)).sum()), frac_above_ah=float(np.mean(raw[thick] > c.nu_h + 1e-9)),
                median=float(np.median(raw[thick])), p99=float(np.percentile(raw[thick], 99)), max=float(raw[thick].max()))


def main(names: list[str]) -> None:
    c = cases.get(10)
    geom = smag_geometry(create_cartesian_grid(nx=c.nx, ny=c.ny, dx=c.dx, dy=c.dx, f0=c.f0))
    for name in names:
        s, meta = series.load(cases.RUNS / "10km" / name / "series.npz")
        for d in DAYS:
            n = int(np.argmin(np.abs(s["t"] - d * 86400.0)))
            r = snapshot(s, n, c, geom, float(meta.get("smag", 0.15)), float(meta.get("dt", 600.0)))
            print(f"{name} day {d}: limited (h ≥ 1 m) {r['limited']}, C Δ²|S| > ah in {100 * r['frac_above_ah']:.1f} %, "
                  f"median {r['median']:.0f}, p99 {r['p99']:.0f}, max {r['max']:.0f} m² s⁻¹")


if __name__ == "__main__":
    main(sys.argv[1:])
