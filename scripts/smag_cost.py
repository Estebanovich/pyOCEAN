#!/usr/bin/env python
"""
Cost of the Smagorinsky viscosity (plan 2026-10-30-smagorinsky-viscosity, task 3).

Median wall time per step of 50 steps after 10 warm-up steps, 5 threads, for ``smagorinsky`` C = 0 (off) and
C = 0.15 (MOM6 OM4 value, Adcroft et al. 2019), on the same case otherwise:

* ``bayiw_cut_zstar``: BayIW cut, z* on the 50 MITgcm levels (validation/bayiw_cut/cut.py:171), Δt 120 s;
* ``dome_10km_hybrid_gls``: DOME 10 km, hybrid + GLS (validation/dome/run_pyocean.py:61), Δt 600 s.

``--profile`` also times the horizontal-viscosity term alone (``m._momentum``) against the whole step.

    python -I scripts/smag_cost.py [--cases bayiw_cut_zstar dome_10km_hybrid_gls] [--profile]

Run alone, with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

NWARM, NMEAS = 10, 50
C_ON = 0.15


def _bayiw(c: float):
    """BayIW cut z* with Smagorinsky C: the model and a step(model, state, n) function."""
    from validation.bayiw_cut import cut
    from validation.bayiw_seiche.run_pyocean import wind_at
    m, st, wind = cut.build_model("2layer", "bay", "zstar", 120.0)
    m.smagorinsky = c
    _push(m, c)
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))

    def step(m, st, n):
        tau[:-1] = wind_at(n * m.dt, wind)
        return m.step(st, tau_y=tau)
    return m, st, step


def _dome(c: float):
    """DOME 10 km hybrid + GLS with Smagorinsky C."""
    from validation.dome import cases
    from validation.dome import run_pyocean as dome
    m, st = dome.build(cases.get(10.0), "hybrid", closure="gls")
    m.smagorinsky = c
    _push(m, c)
    return m, st, lambda m, st, n: m.step(st)


def _push(m, c: float) -> None:
    """Make the coefficient reach the momentum component (set after construction, as ke_scheme in N1)."""
    if hasattr(m, "_momentum"):
        m._momentum.smagorinsky = c


CASES = {"bayiw_cut_zstar": _bayiw, "dome_10km_hybrid_gls": _dome}


def measure(name: str, c: float) -> float:
    """Median s/step of NMEAS steps after NWARM warm-up steps."""
    m, st, step = CASES[name](c)
    for n in range(NWARM):
        st = step(m, st, n)
    t = []
    for n in range(NWARM, NWARM + NMEAS):
        t0 = time.perf_counter()
        st = step(m, st, n)
        t.append(time.perf_counter() - t0)
    return float(np.median(t))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    from core.dynamics import sigma_baroclinic as sb
    sb.set_threads(5)
    print(f"{'case':24s} {'C=0 [s/step]':>14s} {'C=0.15 [s/step]':>16s} {'ratio':>7s}")
    for name in a.cases:
        t0, t1 = measure(name, 0.0), measure(name, C_ON)
        print(f"{name:24s} {t0:14.4f} {t1:16.4f} {t1 / t0:7.3f}", flush=True)


if __name__ == "__main__":
    main()
