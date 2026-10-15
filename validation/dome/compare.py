#!/usr/bin/env python
"""
Tables of the DOME benchmark (spec 2026-10-14-dome-overflow §7): every series.npz under cases.RUNS/<res>km/*/ with the
metrics of validation/dome/metrics.py; x' = x − 1050 km (inlet centre), y' = y − 600 km (slope top) as Legg et al. 2006.
Writes the Markdown to RUNS/compare_<res>km.txt and prints it.

Usage: python validation/dome/compare.py 10 [--t0 30 --t1 50]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.dome import cases, metrics, series  # noqa: E402

XP = (-200.0, -300.0, -500.0, -800.0)             # x' [km] for the path and the transport


def _at(x, f, xp_km):
    return [f[np.argmin(np.abs(x - (xx * 1e3 + 1050e3)))] for xx in xp_km]


def row(name: str, s: dict, meta: dict, t0: float, t1: float) -> dict:
    x, Y = metrics.path_Y(s, t0, t1)
    _, T = metrics.plume_transport(s, t0, t1)
    _, aE = metrics.entrainment(s, t0, t1)
    xa = (x > 750e3) & (x < 1050e3)
    rho = metrics.final_density(s, t0, t1)
    rpe = metrics.rpe_series(s)
    thin, thick = metrics.thin_vs_thick(s)
    r = metrics.ranges(s)
    days = s["t"][-1] / 86400.0
    return dict(name=name, Y=[(y - 600e3) / 1e3 for y in _at(x, Y, XP)], T=[q / 1e6 for q in _at(x, T, XP)],
                aE=float(np.nanmean(aE[xa])), rho=rho, bf=-cases.G * (rho - cases.RHO_TOP) / cases.RHO0 * 1e3,
                cd=[metrics.centroid_depth(s, d * 86400.0) for d in (10, 20, 30, days)],
                drpe=(rpe[-1] - rpe[0]) / rpe[0], T_rng=r["T"], s_rng=r["s"], thin=thin, thick=thick,
                trunc=meta.get("n_trunc", ""), lim=meta.get("n_limited", ""), dt=meta.get("dt", np.nan),
                cost=meta.get("wall_s", np.nan) / days, days=days)


def table(rows: list[dict]) -> str:
    L = ["| run | días | Y'(x' = −200/−300/−500/−800 km) [km] | T(x') [Sv] | α_E (x' 0…−300) | ρ_f / b_f [1e-3 m s⁻²] | "
         "centroide 10/20/30/fin [m] | ΔRPE/RPE₀ | T rango | s rango | |u| delgadas/gruesas | trunc / lim | Δt [s] | "
         "s por día |", "|" + "---|" * 14]
    for r in rows:
        L.append(f"| {r['name']} | {r['days']:.0f} | {' / '.join(f'{v:.0f}' for v in r['Y'])} | "
                 f"{' / '.join(f'{v:.1f}' for v in r['T'])} | {r['aE']:.2e} | {r['rho']:.3f} / {r['bf']:.2f} | "
                 f"{' / '.join(f'{v:.0f}' for v in r['cd'])} | {r['drpe']:.2e} | "
                 f"{r['T_rng'][0]:.3f}–{r['T_rng'][1]:.3f} | {r['s_rng'][0]:.1e}–{r['s_rng'][1]:.6f} | "
                 f"{r['thin']:.2f} / {r['thick']:.2f} | {r['trunc']} / {r['lim']} | {r['dt']:.0f} | {r['cost']:.1f} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("--t0", type=float, default=30.0)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--only", default="", help="substring filter on run names")
    a = ap.parse_args()
    rows = []
    for p in sorted((cases.RUNS / f"{a.res_km:g}km").glob("*/series.npz")):
        if a.only and a.only not in p.parent.name:
            continue
        s, meta = series.load(p)
        if s["t"][-1] < a.t1 * 86400.0:            # short checks (Δt ladder, 2-day runs) are not in the window
            continue
        rows.append(row(p.parent.name, s, meta, a.t0 * 86400.0, a.t1 * 86400.0))
    leg = cases.LEGG_10KM
    txt = (f"# DOME {a.res_km:g} km, métricas promediadas en los días {a.t0:g}–{a.t1:g}\n\n" + table(rows) +
           f"\n\nLegg 2006, 10 km (Figs. 15–17): Y'(x' = {', '.join(f'{v:g}' for v in leg['x_km'])}) z "
           f"{', '.join(f'{v:g}' for v in leg['y_z_km'])}; iso {', '.join(f'{v:g}' for v in leg['y_iso_km'])}; "
           f"iso ν_v 0.5 {', '.join(f'{v:g}' for v in leg['y_iso_visc_km'])}; b_f z {leg['bf_z'] * 1e3:g}, "
           f"iso {leg['bf_iso'] * 1e3:g}, iso ν_v 0.5 {leg['bf_iso_visc'] * 1e3:g} (1e-3 m s⁻²)\n")
    out = cases.RUNS / f"compare_{a.res_km:g}km.txt"
    out.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
