#!/usr/bin/env python
"""
Per-evaluation cost of the scalar density-anomaly kernels used by the nonlinear finite-volume PGF
(core/vertical/pgf_nonlinear.py): ``physics.eos.roquet_rho.rho_anom_elem`` (TEOS-10, Roquet et al. 2015,
MOM6 ROQUET_RHO) against ``physics.eos.wright_full.rho_anom_elem`` (Wright 1997, MOM6 WRIGHT_FULL), and of the
profile interpolation ``_interp`` of the PGF's T/S reference-profile mode.

Two figures per kernel, single thread, after JIT warm-up, best of 7 repeats over N = 2e6 points:

* ``throughput``: independent points in a loop (out[i] = f(T[i], S[i], p[i])), the regime of the PGF kernels;
* ``latency``: each point's T depends on the previous result (a dependency chain), the cost of one isolated call.

    python -I scripts/eos_microbench.py

Run on an otherwise idle machine (one P-core), with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 (cost brief .superpowers/sdd/2026-11-01-teos10-cost/brief.md, step 1).
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
from numba import njit

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.vertical.pgf_nonlinear import _interp                        # noqa: E402
from physics.eos.roquet_rho import rho_anom_elem as roquet_anom          # noqa: E402
from physics.eos.wright_full import rho_anom_elem as wright_anom         # noqa: E402

N = 2_000_000
REPEATS = 7
RHO_REF = 1035.0


@njit
def _thr_roquet(T, S, p, out):  # pragma: no cover
    for i in range(T.size):
        out[i] = roquet_anom(T[i], S[i], p[i], RHO_REF)


@njit
def _thr_wright(T, S, p, out):  # pragma: no cover
    for i in range(T.size):
        out[i] = wright_anom(T[i], S[i], p[i], RHO_REF)


@njit
def _lat_roquet(T, S, p):  # pragma: no cover
    acc = 0.0
    for i in range(T.size):
        acc = roquet_anom(T[i] + 1e-30 * acc, S[i], p[i], RHO_REF)
    return acc


@njit
def _lat_wright(T, S, p):  # pragma: no cover
    acc = 0.0
    for i in range(T.size):
        acc = wright_anom(T[i] + 1e-30 * acc, S[i], p[i], RHO_REF)
    return acc


@njit
def _thr_interp(z, zr, fr, out):  # pragma: no cover
    for i in range(z.size):
        out[i] = _interp(z[i], zr, fr)


def _best(f, *a) -> float:
    f(*a)
    t = []
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        f(*a)
        t.append(time.perf_counter() - t0)
    return min(t) / N * 1e9


def main() -> None:
    rng = np.random.default_rng(0)
    T = rng.uniform(0.0, 30.0, N)
    S = rng.uniform(30.0, 37.0, N)
    p = rng.uniform(0.0, 6e6, N)
    out = np.empty(N)
    zr = np.linspace(-600.0, 0.0, 49)
    fr = rng.uniform(5.0, 25.0, 49)
    z = rng.uniform(-600.0, 0.0, N)
    print(f"ns per evaluation, 1 thread, N = {N:.0e}, best of {REPEATS}")
    print(f"  roquet_rho.rho_anom_elem   throughput {_best(_thr_roquet, T, S, p, out):6.2f}   "
          f"latency {_best(_lat_roquet, T, S, p):6.2f}")
    print(f"  wright_full.rho_anom_elem  throughput {_best(_thr_wright, T, S, p, out):6.2f}   "
          f"latency {_best(_lat_wright, T, S, p):6.2f}")
    print(f"  pgf_nonlinear._interp (49 levels, random z) throughput {_best(_thr_interp, z, zr, fr, out):6.2f}")


if __name__ == "__main__":
    main()
