#!/usr/bin/env python
"""
Bit-for-bit restart check on the BayIW cut (Plan D): 60 h straight against 30 h, write_restart, a fresh model,
read_restart, 30 h; η, u, v, T and h must be identical.  Two-layer case with the bay, wind and open boundaries.

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_cut/restart_check.py [hybrid|zstar]
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from core.restart import read_restart, write_restart  # noqa: E402
from validation.bayiw_cut import cut  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402


def run(m, st, wind, n0: int, n1: int):
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))
    for n in range(n0, n1):
        tau[:-1] = wind_at(n * m.dt, wind)                              # v-face j = south of cell j
        st = m.step(st, tau_y=tau)
    return st


def main() -> None:
    coord = sys.argv[1] if len(sys.argv) > 1 else "hybrid"
    set_threads(5)
    n_half = int(round(30 * 3600 / 120.0))
    tic = time.time()
    m, st, wind = cut.build_model("2layer", "bay", coord)
    ref = run(m, st, wind, 0, 2 * n_half)
    m1, st1, wind = cut.build_model("2layer", "bay", coord)
    st1 = run(m1, st1, wind, 0, n_half)
    path = cut.RUNS / "restart_check" / f"rst_{coord}.nc"
    path.parent.mkdir(parents=True, exist_ok=True)
    write_restart(path, m1, st1)
    m2, _, wind = cut.build_model("2layer", "bay", coord)
    st2 = run(m2, read_restart(path, m2), wind, n_half, 2 * n_half)
    same = {k: bool(np.array_equal(getattr(st2, k), getattr(ref, k), equal_nan=True)) for k in ("eta", "u", "v", "T", "h")}
    print(f"{coord}: 60 h straight vs 30 h + restart + 30 h: identical {same}, time {st2.time == ref.time}, "
          f"trunc {st2.n_trunc}/{ref.n_trunc}, {(time.time() - tic) / 60:.1f} min")


if __name__ == "__main__":
    main()
