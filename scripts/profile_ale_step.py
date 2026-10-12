#!/usr/bin/env python
"""
Timing and profile of the ALE step of SigmaBaroclinic (spec docs/superpowers/specs/2026-10-12-ale-step-performance-design.md).

--case bayiw: BayIW two-layer with the bay, 560 × 352 (validation/bayiw_baroclinic/bayiw.py, needs the external volume);
--case box: 200 × 120 × 30 z* box with open boundaries and a sponge (no external data).
Usage: NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw [--coord zstar] [--steps 5] [--solver cg] [--profile]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402


def box_model(coord: str, solver: str | None, mix: bool = True):
    """200 × 120 × 30 box, 400 m deep with a 100 m shelf, open W/E/S edges, 10-cell sponge, linear stratification."""
    nx, ny, nz = 200, 120, 30
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=2000.0, dy=2000.0, f0=1e-4)
    H = np.where(np.arange(ny)[:, None] > 90, 100.0, 400.0) * np.ones((1, nx))
    kw = dict(regrid_params=dict(dz_top=np.full(nz, 400.0 / nz)))
    if coord == "hybrid":
        rho = 1000.0 - 0.2 * (10.0 + 0.02 * np.linspace(-400.0, 0.0, 200))
        kw = dict(regrid_params=dict(dz_top=np.full(nz, 1.0), rho_target=np.linspace(rho.min() - 0.05, rho.max() + 0.05,
                                                                                     nz + 1)))
    if solver:
        kw["solver"] = solver
    kw["mix_diagnostic"] = mix
    m = SigmaBaroclinic(hg, H, dt=120.0, n_layers=nz, rho0=1000.0, t_coef=0.2, z0=None, closure="constant",
                        nu_const=1e-5, ah=10.0, kh=10.0, kv=1e-5, pgf="fv", rho_ref="auto", vertical_coordinate="ale",
                        regrid=coord, open_boundaries=("west", "east", "south"), sponge_cells=10,
                        sponge_tau=(1200.0, 120.0), **kw)
    z = m.layer_heights(np.zeros((ny, nx)))
    x = (np.arange(nx) + 0.5) * 2000.0
    T = 10.0 + 0.02 * (z + 5.0 * np.sin(-np.pi * z / 400.0) * np.exp(-((x - 200e3) / 20e3) ** 2)[None, None, :])
    return m, m.initial_state(np.zeros((ny, nx)), T), None


def bayiw_model(coord: str, solver: str | None, mix: bool = True):
    from validation.bayiw_baroclinic import bayiw
    m, st, wind = bayiw.build_model("2layer", "bay", coord, 120.0)
    m.mix_diagnostic = mix
    if solver:
        m.core.fs.solver = solver
    return m, st, wind


def run(m, st, wind, n: int, t0: int):
    tau = None
    if wind is not None:
        from validation.bayiw_baroclinic import bayiw
        from validation.bayiw_seiche.run_pyocean import wind_at
        tau = np.zeros((bayiw.NY + 1, bayiw.NX))
    for k in range(n):
        if tau is not None:
            tau[:-1, :] = wind_at((t0 + k) * m.dt, wind)
        st = m.step(st, tau_y=tau)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=("bayiw", "box"), default="box")
    ap.add_argument("--coord", choices=("zstar", "hybrid"), default="zstar")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--solver", choices=("direct", "cg"), default=None)
    ap.add_argument("--profile", action="store_true")
    ap.add_argument("--no-mix", action="store_true", help="switch off the T² spurious-mixing diagnostic")
    a = ap.parse_args()
    set_threads(5)
    m, st, wind = (bayiw_model if a.case == "bayiw" else box_model)(a.coord, a.solver, not a.no_mix)
    st = run(m, st, wind, 3, 0)                                              # warm-up (JIT, LU)
    tic = time.perf_counter()
    st = run(m, st, wind, a.steps, 3)
    print(f"{a.case} {a.coord} solver={m.core.fs.solver} mix={m.mix_diagnostic}: "
          f"{(time.perf_counter() - tic) / a.steps:.3f} s/step")
    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        run(m, st, wind, a.steps, 3 + a.steps)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
