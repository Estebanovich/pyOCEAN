#!/usr/bin/env python
"""
Tables of the four-model benchmark part A (spec §A4–A5): per case and ν_h, every configuration side by side:
(RPE − RPE₀)/RPE₀ at the end, dRPE/dt [W m⁻²] over the spec window, the T range, and the physics of the case;
then the cost table (cost.json). Missing runs show '—'.

Usage: python validation/ilicak_benchmark/compare.py [lock|overflow|iw|seamount]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.ilicak_benchmark import campaign, cases, metrics, series  # noqa: E402

NAMES = {("pyocean", "zstar"): "pyOCEAN z*", ("pyocean", "sigma"): "pyOCEAN σ", ("pyocean", "hybrid"): "pyOCEAN híbrida",
         ("mom6", "zstar"): "MOM6 z*", ("mom6", "rho"): "MOM6 ρ", ("mom6", "sigma"): "MOM6 σ",
         ("croco", "sigma"): "CROCO σ", ("mitgcm", "zstar"): "MITgcm z*"}
WINDOW = {"lock": (15 * 3600.0, 17 * 3600.0), "overflow": (13 * 3600.0, 37 * 3600.0),
          "iw": (10 * 86400.0, 100 * 86400.0), "seamount": (0.0, 10 * 86400.0)}


def summary(case: str, model: str, coord: str, nu: float) -> dict:
    """The metrics of one run (spec §A4)."""
    s, meta = series.load(cases.run_dir(case, model, coord, nu) / "series.npz")
    r = metrics.rpe(s)
    lo, hi = metrics.t_range(s)
    out = dict(rpe_rel_end=float((r[-1] - r[0]) / r[0]), drpe_dt=metrics.drpe_dt(s, r, *WINDOW[case]),
               t_min=lo, t_max=hi, wall_s=meta.get("wall_s"))
    if case == "lock":
        out.update({k: v / metrics.BENJAMIN for k, v in metrics.lock_fronts(s).items()})
    elif case == "overflow":
        xc, dep = metrics.overflow_descent(s)
        xa, da = metrics.overflow_cold_centroid(s)
        out.update(x_c_end=xc[-1] / 1e3, depth_c_end=dep[-1], x_cold_end=xa[-1] / 1e3, depth_cold_end=da[-1])
    elif case == "iw":
        a, t = metrics.iw_amplitude(s), s["t"]
        first, last = t <= 20 * 86400.0, t >= t[-1] - 20 * 86400.0
        out.update(amp_ratio=float(np.sqrt(np.mean(a[last] ** 2)) / np.sqrt(np.mean(a[first] ** 2))))
    else:
        out.update(umax_end=float(metrics.max_speed(s)[-1] * 100.0))
    return out


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))


def table(case: str) -> list[str]:
    extra = {"lock": ["dense_nose", "light_nose", "dense_body", "light_body"], "overflow": ["x_c_end", "depth_c_end", "x_cold_end", "depth_cold_end"],
             "iw": ["amp_ratio"], "seamount": ["umax_end"]}[case]
    cols = ["rpe_rel_end", "drpe_dt", "t_min", "t_max"] + extra
    rows = [f"## {case}", "", "| ν_h | modelo | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for nu in cases.get(case).nu_h:
        for model, coord in campaign.CONFIGS[case]:
            try:
                r = summary(case, model, coord, nu)
            except FileNotFoundError:
                r = {}
            rows.append(f"| {nu:g} | {NAMES[(model, coord)]} | " + " | ".join(_fmt(r.get(k)) for k in cols) + " |")
    cj = cases.RUNS / case / "cost.json"
    if cj.exists():
        rows += ["", "| modelo | Δt máx [s] | s por hora simulada (Δt máx) | s por hora simulada (Δt de resultados) | procesos |",
                 "|---|---|---|---|---|"]
        for key, v in json.loads(cj.read_text()).items():
            m, co = key.split("_", 1)
            v = v or {}
            rows.append(f"| {NAMES.get((m, co), key)} | {_fmt(v.get('dt_max'))} | {_fmt(v.get('s_per_hour'))} | "
                        f"{_fmt(v.get('s_per_hour_results_dt'))} | {_fmt(v.get('threads'))} |")
    return rows


if __name__ == "__main__":
    for c in (sys.argv[1:] or list(campaign.CONFIGS)):
        print("\n".join(table(c)), "\n")
