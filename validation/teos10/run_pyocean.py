#!/usr/bin/env python
"""
Production runs of the TEOS-10 validation (Task 9 of docs/superpowers/plans/2026-10-31-teos10-3d.md; spec criteria
4 and 5; case definitions in ``validation/teos10/cases.py``).  One run at a time on the 5 performance cores.

    python -m validation.teos10.run_pyocean rest seamount wright_full zstar [--rho-ref none]
    python -m validation.teos10.run_pyocean cabbeling teos10
    python -m validation.teos10.run_pyocean thermobaric teos10 4000
    python -m validation.teos10.run_pyocean --all            # every run of RUN_LIST, one after the other

Output ``RUNS/<case name>.npz`` (``cases.RUNS`` = /Volumes/ESTEBAN_MIT/pyOCEAN_runs/teos10, never into the repo):
  every case: time [s] of the records (every ``record_every`` steps from t = 0), umax (nt,) max |u|, |v| over the
  domain [m s⁻¹], wall [s], s_per_step [s], n_steps;
  rest: pgf0 (initial max |PGF| at u and v faces [m s⁻²]), umax_final, the argmax (k, j, i) of |u| at the end;
  cabbeling: dense_vol (nt,) [m³] and dense_z (nt,) [m] (``cases.denser_volume``), umax;
  thermobaric: rise (nt,) (``cases.warm_rise``), T_col and S_col (nt, N) of column (2, 2) bed first in the EOS's
  variables, z_col (nt, N) [m].

Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=5
(CLAUDE.md).
"""
from __future__ import annotations

import argparse
import time

import numpy as np

from core.dynamics.sigma_baroclinic import set_threads
from validation.teos10 import cases as C

RUN_LIST: list[tuple] = (
    [("rest", k, e, c, "auto") for k in ("seamount", "bayiw_step") for c in ("zstar", "hybrid") for e in C.EOS_NAMES]
    + [("rest", k, e, c, "none") for k in ("seamount", "bayiw_step") for c in ("zstar", "hybrid") for e in C.EOS_NAMES]
    + [("cabbeling", e) for e in C.EOS_NAMES]
    + [("thermobaric", e, d) for e in C.EOS_NAMES for d in (100.0, 4000.0)]
)


def build(spec: tuple, fast: bool = False) -> C.Case:
    """The case of a ``RUN_LIST`` entry."""
    if spec[0] == "rest":
        _, kind, eos, coord, ref = spec
        return C.rest_case(kind, eos, coord, rho_ref=None if ref == "none" else "auto", fast=fast)
    if spec[0] == "cabbeling":
        return C.cabbeling_case(spec[1], fast=fast)
    if spec[0] == "thermobaric":
        return C.thermobaric_case(spec[1], float(spec[2]), fast=fast)
    raise ValueError(f"unknown case {spec[0]!r}")


def _umax(st) -> float:
    return float(max(np.abs(st.u).max(), np.abs(st.v).max()))


def run(spec: tuple, fast: bool = False, save: bool = True, record_hours: float = 1.0) -> dict:
    """Run one case to its end and return (and save) its records."""
    case = build(spec, fast)
    m, st = case.model, case.state
    every = max(1, int(round(record_hours * 3600.0 / m.dt)))
    rec: dict[str, list] = {"time": [], "umax": []}
    kind = spec[0]
    out: dict = {}
    if kind == "rest":
        gx, gy = m.pressure_gradient(st.eta, st.T, st.h, st.S)
        out["pgf0"] = float(max(np.abs(gx).max(), np.abs(gy).max()))

    def record(s) -> None:
        rec["time"].append(s.time)
        rec["umax"].append(_umax(s))
        if kind == "cabbeling":
            v, z = C.denser_volume(case, s)
            rec.setdefault("dense_vol", []).append(v)
            rec.setdefault("dense_z", []).append(z)
        elif kind == "thermobaric":
            rec.setdefault("rise", []).append(C.warm_rise(case, s))
            rec.setdefault("T_col", []).append(s.T[:, 2, 2].copy())
            rec.setdefault("S_col", []).append(s.S[:, 2, 2].copy())
            rec.setdefault("z_col", []).append(m.layer_heights(s.eta, s.h)[:, 2, 2].copy())

    record(st)
    t0 = time.perf_counter()
    for n in range(1, case.n_steps + 1):
        st = m.step(st)
        if n % every == 0 or n == case.n_steps:
            if not (np.isfinite(st.u).all() and np.isfinite(st.T).all()):
                raise FloatingPointError(f"{case.name}: non-finite state at step {n}")
            record(st)
    wall = time.perf_counter() - t0
    out.update({k: np.array(v) for k, v in rec.items()})
    out.update(wall=wall, n_steps=case.n_steps, s_per_step=wall / case.n_steps, umax_final=_umax(st),
               n_trunc=int(st.n_trunc), n_limited=int(st.n_limited))
    if kind == "rest":
        out["argmax_u"] = np.array(np.unravel_index(np.abs(st.u).argmax(), st.u.shape))
    if save:
        C.RUNS.mkdir(parents=True, exist_ok=True)
        np.savez(C.RUNS / f"{case.name}{'_fast' if fast else ''}.npz", **out)
    return dict(name=case.name, **out)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("case", nargs="*", help="rest KIND EOS COORD | cabbeling EOS | thermobaric EOS DEPTH")
    ap.add_argument("--rho-ref", default="auto", choices=("auto", "none"), help="rest only")
    ap.add_argument("--all", action="store_true", help="every run of RUN_LIST, one at a time")
    ap.add_argument("--fast", action="store_true", help="the coarse 1-day versions of the tests")
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    set_threads(a.threads)
    if a.all:
        specs = RUN_LIST
    elif a.case and a.case[0] == "rest":
        specs = [tuple(a.case) + (a.rho_ref,)]
    elif a.case and a.case[0] == "thermobaric":
        specs = [(a.case[0], a.case[1], float(a.case[2]))]
    else:
        specs = [tuple(a.case)]
    for spec in specs:
        r = run(spec, fast=a.fast)
        extra = ""
        if spec[0] == "cabbeling":
            extra = f", dense volume {r['dense_vol'][-1]:.3e} m³ at z {r['dense_z'][-1]:.1f} m"
        elif spec[0] == "thermobaric":
            extra = f", warm rise {r['rise'][-1]:.3f}"
        print(f"{r['name']}: {r['n_steps']} steps, {r['wall']:.0f} s ({1e3 * r['s_per_step']:.1f} ms/step), "
              f"max|u| {r['umax'].max():.3e} m/s{extra}", flush=True)


if __name__ == "__main__":
    main()
