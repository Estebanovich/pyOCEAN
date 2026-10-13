#!/usr/bin/env python
"""
Step cost of the BayIW cut (spec 2026-10-13-ale-serial-glue-performance §5): s/step with 1 and 5 threads, the
1→5 speed-up (Amdahl: serial share) and the functions with the most self time, for z* and hybrid radiating and
z* nested.  s/step: 30 steps after 5 warm-up steps (Numba compilation), without the profiler; the breakdown comes
from 30 further steps under cProfile.

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_cut/profile_step.py [zstar_radiate hybrid_radiate zstar_nest]
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from validation.bayiw_cut import cut  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402

CASES = {"zstar_radiate": ("zstar", "radiate"), "hybrid_radiate": ("hybrid", "radiate"), "zstar_nest": ("zstar", "nest")}


def measure(coord: str, mode: str, threads: int, profile: bool = False):
    set_threads(threads)
    state = cut.nest_state("2layer", "bay") if mode == "nest" else None
    m, st, wind = cut.build_model("2layer", "bay", coord, boundary_state=state)
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))
    for n in range(5):
        tau[:-1] = wind_at(n * m.dt, wind)
        st = m.step(st, tau_y=tau)
    t0 = time.perf_counter()                                     # timed without the profiler
    for n in range(5, 35):
        tau[:-1] = wind_at(n * m.dt, wind)
        st = m.step(st, tau_y=tau)
    elapsed = (time.perf_counter() - t0) / 30
    pr = None
    if profile:                                                  # breakdown: 30 more steps under cProfile
        pr = cProfile.Profile()
        pr.enable()
        for n in range(35, 65):
            tau[:-1] = wind_at(n * m.dt, wind)
            st = m.step(st, tau_y=tau)
        pr.disable()
    return elapsed, pr


def self_times(pr) -> dict:
    return {(k[0].split("/")[-1], k[2]): v[2] / 30 for k, v in pstats.Stats(pr).stats.items()}


def main() -> None:
    for name in sys.argv[1:] or list(CASES):
        coord, mode = CASES[name]
        t1, p1 = measure(coord, mode, 1, profile=True)
        t5, p5 = measure(coord, mode, 5, profile=True)
        s1, s5 = self_times(p1), self_times(p5)
        print(f"\n{name}: {t1:.3f} s/step (1 thread), {t5:.3f} s/step (5 threads), speed-up {t1 / t5:.2f}")
        print("  function                         self ms (5 thr)  speed-up 1→5")
        for k, v in sorted(s5.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {k[1][:32]:32s} {1e3 * v:8.1f}        {s1.get(k, 0.0) / max(v, 1e-12):5.2f}")


if __name__ == "__main__":
    main()
