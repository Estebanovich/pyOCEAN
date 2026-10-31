#!/usr/bin/env python
"""
Tables of the four-model benchmark part A (spec §A4–A5): per case and ν_h, every configuration side by side:
(RPE − RPE₀)/RPE₀ at the end, dRPE/dt [W m⁻²] over the spec window (lock: the 15–17 h mean), the T range and the
new extrema in cells ≥ 1 m (t_excess [°C], v_excess = volume fraction), and the physics of the case (IW: damping of
the oscillation, metrics.oscillation_ratio);
then the cost table (cost.json). Missing runs show '—'.  Extra cases (``EXTRA``: 'overflow_smag', Ilicak et al.
2012 Fig. 19b) use the metrics and window of their base case and add the base-case run at the same ν_h as a
reference row, and, when it exists, the operator control of the base case (tag ``CONTROL_TAG``, run with
``--smag 1e-9``: the background ν_h in the thickness-weighted tensor operator, no Smagorinsky term), which separates
the operator change from the Smagorinsky term (final review M2).  Tagged sensitivity runs of a case (``TAGGED``:
the overflow z* with the Jackson et al. 2008 shear mixing, run_pyocean.py --shear jackson, tag 'jackson'; Jackson
shear-mixing plan, Task 6) are added as rows when their series exists.

Usage: python validation/ilicak_benchmark/compare.py [lock|overflow|iw|seamount|overflow_smag]
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
# extra case → (base case, its configurations); not in campaign.CONFIGS, so neither the results nor the cost
# campaign runs them
EXTRA = {"overflow_smag": ("overflow", [("pyocean", "zstar")])}
TAGGED = {"overflow": [("pyocean", "zstar", "jackson", "pyOCEAN z* + Jackson (Pr = 1)")]}
CONTROL_TAG = "smag1e-09"          # operator control of an EXTRA case: base case run with C = 1e-9 (C ≈ 0)


def base(case: str) -> str:
    """The case whose metrics and window ``case`` uses (itself unless an EXTRA case)."""
    return EXTRA[case][0] if case in EXTRA else case


def summary(case: str, model: str, coord: str, nu: float, tag: str = "") -> dict:
    """The metrics of one run (spec §A4); ``tag`` selects a tagged run directory (cases.run_dir)."""
    s, meta = series.load(cases.run_dir(case, model, coord, nu, tag) / "series.npz")
    r = metrics.rpe(s)
    lo, hi = metrics.t_range(s)
    exc, frac = metrics.t_excess(s)
    kind = base(case)
    out = dict(rpe_rel_end=float((r[-1] - r[0]) / r[0]), drpe_dt=metrics.drpe_dt(s, r, *WINDOW[kind]),
               t_min=lo, t_max=hi, t_excess=exc, v_excess=frac, wall_s=meta.get("wall_s"))
    if meta.get("smag"):
        out.update(n_visc_limited=meta.get("n_visc_limited"), nu_t_max=meta.get("nu_t_max"))
    if kind == "lock":
        out.update({k: v / metrics.BENJAMIN for k, v in metrics.lock_fronts(s).items()})
    elif kind == "overflow":
        xc, dep = metrics.overflow_descent(s)
        xa, da = metrics.overflow_cold_centroid(s)
        out.update(x_c_end=xc[-1] / 1e3, depth_c_end=dep[-1], x_cold_end=xa[-1] / 1e3, depth_cold_end=da[-1])
    elif kind == "iw":
        out.update(amp_ratio=metrics.oscillation_ratio(s["t"], metrics.iw_amplitude(s)))
    else:
        out.update(umax_end=float(metrics.max_speed(s)[-1] * 100.0))
    return out


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))


def table(case: str) -> list[str]:
    extra = {"lock": ["dense_nose", "light_nose", "dense_body", "light_body"], "overflow": ["x_c_end", "depth_c_end", "x_cold_end", "depth_cold_end"],
             "iw": ["amp_ratio"], "seamount": ["umax_end"]}[base(case)]
    if case in EXTRA:
        extra = extra + ["n_visc_limited", "nu_t_max"]
    cols = ["rpe_rel_end", "drpe_dt", "t_min", "t_max", "t_excess", "v_excess"] + extra
    rows = [f"## {case}", "", "| ν_h | modelo | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    configs = EXTRA[case][1] if case in EXTRA else campaign.CONFIGS[case]
    for nu in cases.get(case).nu_h:
        runs = [(case, model, coord, NAMES[(model, coord)], "") for model, coord in configs]
        if case in EXTRA:                                  # the base-case run at the same ν_h, as reference
            runs += [(base(case), model, coord, f"{NAMES[(model, coord)]} ({base(case)})", "") for model, coord in configs]
            runs += [(base(case), model, coord, f"{NAMES[(model, coord)]} ({base(case)}, control de operador, C≈0)",
                      CONTROL_TAG) for model, coord in configs
                     if (cases.run_dir(base(case), model, coord, nu, CONTROL_TAG) / "series.npz").exists()]
        runs += [(case, model, coord, label, tag) for model, coord, tag, label in TAGGED.get(case, [])
                 if (cases.run_dir(case, model, coord, nu, tag) / "series.npz").exists()]
        for cs, model, coord, label, tag in runs:
            try:
                r = summary(cs, model, coord, nu, tag)
            except FileNotFoundError:
                r = {}
            rows.append(f"| {nu:g} | {label} | " + " | ".join(_fmt(r.get(k)) for k in cols) + " |")
    cj = cases.RUNS / case / "cost.json"
    if cj.exists():
        rows += ["", "| modelo | Δt máx [s] | s por hora simulada (Δt máx) | s por hora simulada (Δt de resultados) | procesos |",
                 "|---|---|---|---|---|"]
        for key, v in json.loads(cj.read_text()).items():
            m, co = key.split("_", 1)
            v = v or {}
            dtm = ("≥ " if v.get("at_top") else "") + _fmt(v.get("dt_max"))
            rows.append(f"| {NAMES.get((m, co), key)} | {dtm} | {_fmt(v.get('s_per_hour'))} | "
                        f"{_fmt(v.get('s_per_hour_results_dt'))} | {_fmt(v.get('threads'))} |")
    return rows


if __name__ == "__main__":
    for c in (sys.argv[1:] or list(campaign.CONFIGS) + list(EXTRA)):
        print("\n".join(table(c)), "\n")
