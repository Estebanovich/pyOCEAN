#!/usr/bin/env python
"""
N1 smoke runs (plan docs/superpowers/plans/2026-10-28-n1-component-core.md, task 14): short runs of the benchmark
cases built with the code under ``--root`` (this branch, or a worktree of ``main``), hashing the final state and
timing the step.  Run once per root, one run at a time on the 5 performance cores, then compare:

    git worktree add ../pyOCEAN_main_n1 main
    python -I scripts/n1_smoke.py --root ../pyOCEAN_main_n1 --out <dir>/main.json
    python -I scripts/n1_smoke.py --root . --out <dir>/branch.json
    python -I scripts/n1_smoke.py --compare <dir>/main.json <dir>/branch.json

Cost: median wall time per step of 20 steps after 5 warm-up steps, 5 threads.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
import warnings
from pathlib import Path

import numpy as np

COST = ("bayiw_cut_zstar", "bayiw_cut_hybrid", "ilicak_overflow_zstar")


def _hash(st) -> str:
    h = hashlib.sha256()
    for name in ("eta", "u", "v", "T", "h", "tr", "Qx", "Qy", "Fz", "mix_num"):
        a = getattr(st, name, None)
        if a is not None:
            h.update(name.encode())
            h.update(np.ascontiguousarray(a).tobytes())
    return h.hexdigest()


def jobs() -> dict:
    """name → (build() → (model, state, step(model, state, n) → state), number of steps)."""
    from validation.bayiw_cut import cut
    from validation.bayiw_seiche.run_pyocean import wind_at
    from validation.dome import cases as dome_cases
    from validation.dome import run_pyocean as dome
    from validation.ilicak_benchmark import cases as ili_cases
    from validation.ilicak_benchmark import run_pyocean as ili
    from validation.internal_wave import case as iw
    from validation.seamount import run_seamount as smt

    def bayiw(coord):
        def build():
            m, st, wind = cut.build_model("2layer", "bay", coord, 120.0)
            tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))

            def step(m, st, n):
                tau[:-1] = wind_at(n * m.dt, wind)
                return m.step(st, tau_y=tau)
            return m, st, step
        return build

    def ilicak(name, coord, nu):
        def build():
            m, st = ili.build(ili_cases.get(name), coord, nu)
            return m, st, lambda m, st, n: m.step(st)
        return build

    def dome_build():
        m, st = dome.build(dome_cases.get(10.0), "zstar")
        return m, st, lambda m, st, n: m.step(st)

    def iw_build():
        m, st = iw.build_model("zstar", 500.0, "arakawa")
        return m, st, lambda m, st, n: m.step(st)

    def seamount_build():
        m, st, _ = smt.setup(8000.0, 20, 600.0, "jacobian", "sigma")
        return m, st, lambda m, st, n: m.step(st)

    return {
        "bayiw_cut_zstar": (bayiw("zstar"), 30),                         # 1 h at 120 s
        "bayiw_cut_hybrid": (bayiw("hybrid"), 30),
        "ilicak_lock_zstar": (ilicak("lock", "zstar", 1.0), None),       # 30 min at the case Δt
        "ilicak_lock_sigma": (ilicak("lock", "sigma", 1.0), None),
        "ilicak_lock_hybrid": (ilicak("lock", "hybrid", 1.0), None),
        "ilicak_overflow_zstar": (ilicak("overflow", "zstar", 1000.0), 40),
        "dome_10km_zstar": (dome_build, 144),                             # 1 day at 600 s
        "internal_wave_zstar": (iw_build, 8),                             # ~1 h at 500 s
        "seamount_sigma": (seamount_build, 144),                          # 1 day at 600 s
    }


def run(root: Path, out: Path) -> None:
    sys.path.insert(0, str(root.resolve()))
    warnings.simplefilter("ignore")
    from core.dynamics import sigma_baroclinic as sb
    sb.set_threads(5)
    res = {"_root": str(root.resolve())}
    for name, (build, nsteps) in jobs().items():
        m, st, step = build()
        if nsteps is None:
            nsteps = int(round(1800.0 / m.dt))
        times = []
        for n in range(nsteps):
            t0 = time.perf_counter()
            st = step(m, st, n)
            times.append(time.perf_counter() - t0)
        cost = None
        if name in COST:                                                 # extra steps for the timing only
            for n in range(nsteps, nsteps + 25):
                t0 = time.perf_counter()
                st2 = step(m, st, n)
                times.append(time.perf_counter() - t0)
                st = st2
            cost = float(np.median(times[-20:]))
        res[name] = {"hash": _hash(st) if cost is None else res.get(name, {}).get("hash"), "steps": nsteps,
                     "cost": cost}
        if cost is not None:                                             # hash taken at nsteps: rebuild and rerun
            m, st, step = build()
            for n in range(nsteps):
                st = step(m, st, n)
            res[name]["hash"] = _hash(st)
        print(f"{name:24s} {nsteps:4d} steps  hash {res[name]['hash'][:12]}"
              + ("" if cost is None else f"  {cost * 1e3:7.1f} ms/step"), flush=True)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1))


def compare(a: Path, b: Path) -> int:
    ra, rb = json.loads(a.read_text()), json.loads(b.read_text())
    names = [k for k in ra if not k.startswith("_")]
    same = [k for k in names if ra[k]["hash"] == rb[k]["hash"]]
    print(f"{len(same)}/{len(names)} identical")
    for k in names:
        line = f"  {k:24s} {'identical' if k in same else 'DIFFERS'}"
        if ra[k]["cost"]:
            line += f"  cost {rb[k]['cost'] / ra[k]['cost']:.3f} × main ({ra[k]['cost'] * 1e3:.1f} → " \
                    f"{rb[k]['cost'] * 1e3:.1f} ms)"
        print(line)
    return 0 if len(same) == len(names) else 1


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path)
    ap.add_argument("--out", type=Path)
    ap.add_argument("--compare", nargs=2, type=Path)
    args = ap.parse_args()
    if args.compare:
        sys.exit(compare(*args.compare))
    run(args.root, args.out)
