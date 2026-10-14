#!/usr/bin/env python
"""
Run matrix of the four-model benchmark part A (spec §A2): results runs (3 ν_h × 7 configurations per 2-D case, the
seamount in 6) and the cost ladder (Δt × 1, 2, 4, 8, 16 for 1 simulated hour — 1 day for the internal wave and the
seamount — the largest stable Δt gives the cost). One run at a time; pyOCEAN with 5 threads; MOM6, CROCO and MITgcm
on one process (NJ = 1 in MOM6 needs one rank; CROCO is a serial build).

Usage: python validation/ilicak_benchmark/campaign.py results [case] | cost [case]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.ilicak_benchmark import cases, metrics, series  # noqa: E402

HERE = Path(__file__).resolve().parent
_TWO_D = [("pyocean", "zstar"), ("pyocean", "sigma"), ("pyocean", "hybrid"), ("mom6", "zstar"), ("mom6", "rho"),
          ("croco", "sigma"), ("mitgcm", "zstar")]
CONFIGS = {"lock": _TWO_D, "overflow": _TWO_D, "iw": _TWO_D,
           "seamount": [("pyocean", "sigma"), ("pyocean", "zstar"), ("mom6", "sigma"), ("mom6", "zstar"),
                        ("croco", "sigma"), ("mitgcm", "zstar")]}
COST_HOURS = {"lock": 1.0, "overflow": 1.0, "iw": 24.0, "seamount": 24.0}


def _num(v: float | None) -> str:
    return "" if v is None else f"{v:g}"


def command(case: str, model: str, coord: str, nu: float, dt: float | None = None, hours: float | None = None,
            tag: str = "") -> list[str]:
    """The command line of one run."""
    if model == "pyocean":
        cmd = [sys.executable, str(HERE / "run_pyocean.py"), case, coord, f"{nu:g}"]
        cmd += (["--dt", f"{dt:g}"] if dt else []) + (["--hours", f"{hours:g}"] if hours else [])
        return cmd + (["--tag", tag] if tag else [])
    return [str(HERE / f"run_{model}.sh"), "run", case, coord, f"{nu:g}", _num(dt), _num(hours), tag]


def _run(cmd: list[str]) -> None:
    env = {**os.environ, "NUMBA_NUM_THREADS": "5"}
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(" ".join(cmd), "→", (r.stdout.strip().splitlines() or [""])[-1], flush=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)


def results(case: str | None = None, force: bool = False) -> list[Path]:
    """Every results run of ``case`` (all cases if None); existing series.npz are kept unless force."""
    done = []
    for c in ([case] if case else list(CONFIGS)):
        for nu in cases.get(c).nu_h:
            for model, coord in CONFIGS[c]:
                p = cases.run_dir(c, model, coord, nu) / "series.npz"
                if force or not p.exists():
                    _run(command(c, model, coord, nu))
                if p.exists():
                    done.append(p)
    return done


def stable(path: Path, umax: float = 15.0) -> bool:
    """A run is stable when every T is finite where h > 0 and max |u|, |v| < umax [m s⁻¹] over faces ≥ 1 cm thick
    (the overflow reaches ~8 m s⁻¹, the free fall √(2 g′ 1500 m), in every model)."""
    s, _ = series.load(path)
    T = np.where(s["h"] > 0, s["T"], 0.0)
    return bool(np.isfinite(T).all() and np.isfinite(s["u"]).all() and metrics.max_speed(s).max() < umax)


def cost(case: str, model: str, coord: str, factors=(1, 2, 4, 8, 16)) -> dict:
    """Largest stable Δt of the ladder (middle ν_h, COST_HOURS) and its cost [s per simulated hour]."""
    c = cases.get(case)
    nu = c.nu_h[len(c.nu_h) // 2]
    best = None
    for f in factors:
        dt = c.dt * f
        tag = f"cost_dt{dt:g}"
        _run(command(case, model, coord, nu, dt, COST_HOURS[case], tag))
        p = cases.run_dir(case, model, coord, nu, tag) / "series.npz"
        if not (p.exists() and stable(p)):
            break
        _, meta = series.load(p)
        best = dict(dt_max=dt, s_per_hour=meta["wall_s"] / COST_HOURS[case], threads=meta["threads"])
    res = cases.run_dir(case, model, coord, nu) / "series.npz"
    if res.exists():
        _, meta = series.load(res)
        (best := best or {}).update(s_per_hour_results_dt=meta["wall_s"] / (meta["n_steps"] * meta["dt"] / 3600.0))
    out = cases.RUNS / case / "cost.json"
    table = json.loads(out.read_text()) if out.exists() else {}
    table[f"{model}_{coord}"] = best
    out.write_text(json.dumps(table, indent=1))
    return best or {}


if __name__ == "__main__":
    what, which = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    if what == "results":
        results(which)
    elif what == "cost":
        for cs in ([which] if which else list(CONFIGS)):
            for model, coord in CONFIGS[cs]:
                print(cs, model, coord, cost(cs, model, coord), flush=True)
    else:
        raise SystemExit("usage: campaign.py results [case] | cost [case]")
