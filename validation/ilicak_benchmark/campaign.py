#!/usr/bin/env python
"""
Run matrix of the four-model benchmark part A (spec §A2): results runs (3 ν_h × 7 configurations per 2-D case, the
seamount in 6) and the cost ladder (Δt × 1, 2, 4, 8, 16 over COST_HOURS — past the overflow's descent — the largest
stable Δt gives the cost, spec §A2; a rung at the top of the ladder is a lower bound). One run at a time; results runs
with pyOCEAN on 5 threads; the cost ladder on one process for every model (NJ = 1 in MOM6 needs one rank; CROCO and
MITgcm are serial builds).

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
COST_HOURS = {"lock": 2.0, "overflow": 12.0, "iw": 48.0, "seamount": 48.0}    # the overflow descends at 3–8 h


def _num(v: float | None) -> str:
    return "" if v is None else f"{v:g}"


def command(case: str, model: str, coord: str, nu: float, dt: float | None = None, hours: float | None = None,
            tag: str = "", threads: int | None = None) -> list[str]:
    """The command line of one run (``threads`` only for pyOCEAN; the other models run on one process)."""
    if model == "pyocean":
        cmd = [sys.executable, str(HERE / "run_pyocean.py"), case, coord, f"{nu:g}"]
        cmd += (["--dt", f"{dt:g}"] if dt else []) + (["--hours", f"{hours:g}"] if hours else [])
        return cmd + (["--tag", tag] if tag else []) + (["--threads", str(threads)] if threads else [])
    return [str(HERE / f"run_{model}.sh"), "run", case, coord, f"{nu:g}", _num(dt), _num(hours), tag]


def _run(cmd: list[str]) -> None:
    env = {**os.environ, "NUMBA_NUM_THREADS": "1" if cmd[-2:] == ["--threads", "1"] else "5"}
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


def stable(path: Path, umax: float = 15.0, ref_umax: float | None = None) -> bool:
    """A run is stable when every T is finite where h > 0 and max |u|, |v| < umax [m s⁻¹] over faces ≥ 1 cm thick
    (the overflow reaches ~8 m s⁻¹, the free fall √(2 g′ 1500 m), in every model)."""
    s, _ = series.load(path)
    T = np.where(s["h"] > 0, s["T"], 0.0)
    top = metrics.max_speed(s).max()
    finite = bool(np.isfinite(T).all() and np.isfinite(s["u"]).all())
    if ref_umax is not None:                         # relative to the results run (MOM6 z* overflow reaches 17 m/s)
        return finite and top <= 2.0 * ref_umax + 0.1
    return finite and top < umax


def cost(case: str, model: str, coord: str, factors=(1, 2, 4, 8, 16)) -> dict:
    """Largest stable Δt of the ladder (middle ν_h, COST_HOURS) and its cost [s per simulated hour]."""
    c = cases.get(case)
    nu = c.nu_h[len(c.nu_h) // 2]
    best = None
    res = cases.run_dir(case, model, coord, nu) / "series.npz"
    ref = None
    if res.exists():
        sr, _ = series.load(res)
        ref = float(metrics.max_speed(sr)[: int(COST_HOURS[case] * 3600 / c.out_s) + 1].max())
    for f in factors:
        dt = c.dt * f
        tag = f"cost_dt{dt:g}"
        _run(command(case, model, coord, nu, dt, COST_HOURS[case], tag, threads=1))   # one process, as the others
        p = cases.run_dir(case, model, coord, nu, tag) / "series.npz"
        if not (p.exists() and stable(p, ref_umax=ref)):
            break
        _, meta = series.load(p)
        best = dict(dt_max=dt, at_top=f == factors[-1], s_per_hour=meta["wall_s"] / (meta["n_steps"] * dt / 3600.0),
                    threads=meta["threads"])
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
