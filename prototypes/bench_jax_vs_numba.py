#!/usr/bin/env python
"""
Benchmark of the tracer-advection RHS: NumPy reference, fused Numba kernel and the
JAX prototype, on the BayIW size (2 tracers × 50 × 352 × 560) and a small case for
correctness.  Threads: Numba via set_threads; JAX/XLA via XLA_FLAGS (set before
importing jax, see --jax-threads).  Keep the total ≤ 5 (P-cores).

Usage:  python prototypes/bench_jax_vs_numba.py [--threads 5] [--reps 5]
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from pathlib import Path

ap = argparse.ArgumentParser()
ap.add_argument("--threads", type=int, default=5)
ap.add_argument("--reps", type=int, default=5)
args = ap.parse_args()
# XLA CPU always on one thread: its default pool takes every core (E-cores included) and macOS cannot pin
# threads, so JAX is compared per core with Numba on 1 thread; Numba is also timed on --threads (≤ 5 P-cores)
os.environ["XLA_FLAGS"] = "--xla_cpu_multi_thread_eigen=false intra_op_parallelism_threads=1"

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "prototypes"))

import numba  # noqa: E402
import numpy as np  # noqa: E402

import jax_tracer_advection as jx  # noqa: E402
from core.dynamics import sigma_baroclinic as sb  # noqa: E402
from core.dynamics.sigma_baroclinic import SigmaBaroclinic  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402

numba.set_num_threads(args.threads)


def case(nz, ny, nx, seed=0):
    rng = np.random.default_rng(seed)
    c = np.stack([rng.normal(10, 1, (nz, ny, nx)), rng.normal(100, 5, (nz, ny, nx))])
    ok = rng.random((nz, ny, nx)) > 0.05
    Qx = rng.normal(size=(nz, ny, nx + 1))
    Qy = rng.normal(size=(nz, ny + 1, nx))
    Qy[:, -1] = 0.0
    Fz = rng.normal(size=(nz + 1, ny, nx))
    Fz[0] = Fz[-1] = 0.0
    ext = c + rng.normal(0, 0.1, c.shape)
    return c, Qx, Qy, Fz, ok, ext


# ---- correctness on a small case: JAX and Numba vs the NumPy reference
nz, ny, nx = 6, 9, 13
c, Qx, Qy, Fz, ok, ext = case(nz, ny, nx)
hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=0.0)
m = SigmaBaroclinic(hg, np.full((ny, nx), 30.0), dt=10.0, n_layers=nz, open_boundaries=("west", "east", "south"))
m.T_ext = ext[0]
ext_ref = np.stack([m.T_ext, m.T_ext**2])
ref = m._rhs_numpy(c, Qx, Qy, Fz, ok)
nb = m._rhs(c, Qx, Qy, Fz, ok)
jj = np.asarray(jx.advect_rhs(c, Qx, Qy, Fz, ok, ext_ref, open_w=True, open_e=True, open_s=True))
print(f"max |Numba − NumPy| = {np.abs(nb - ref).max():.2e}   max |JAX − NumPy| = {np.abs(jj - ref).max():.2e}")

# ---- timing on the BayIW size
c, Qx, Qy, Fz, ok, ext = case(50, 352, 560, seed=1)
flags = (True, True, True, False)


def timeit(f, reps):
    f()                                                     # compile / warm up
    t = time.perf_counter()
    for _ in range(reps):
        f()
    return (time.perf_counter() - t) / reps


numba.set_num_threads(1)                                   # the kernel is called directly (no dispatch wrapper)
t_nb1 = timeit(lambda: sb._advect_div(c, Qx, Qy, Fz, ok, ext, *flags, True), args.reps)
numba.set_num_threads(args.threads)
t_nb = timeit(lambda: sb._advect_div(c, Qx, Qy, Fz, ok, ext, *flags, True), args.reps)
cj, Qxj, Qyj, Fzj, okj, extj = (jx.jnp.asarray(a) for a in (c, Qx, Qy, Fz, ok, ext))
t_jx = timeit(lambda: jx.advect_rhs(cj, Qxj, Qyj, Fzj, okj, extj, open_w=True, open_e=True,
                                    open_s=True).block_until_ready(), args.reps)
print(f"BayIW size 2×50×352×560:  Numba 1 thread {t_nb1:.3f} s   Numba {args.threads} threads {t_nb:.3f} s   "
      f"JAX 1 thread {t_jx:.3f} s   (JAX/Numba per core = {t_jx / t_nb1:.2f})")
