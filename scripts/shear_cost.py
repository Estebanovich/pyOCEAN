#!/usr/bin/env python
"""
Cost of the Jackson et al. (2008) shear mixing (plan 2026-10-30-jackson-shear-mixing, task 5; spec criterion 7).

Median wall time per step of 50 steps after 10 warm-up steps, 5 threads, without (``shear_mixing=None``) and with
``shear_mixing='jackson'`` (Pr = 1), on the same case otherwise; the shear mixing is passed to the constructor.

* ``bayiw_cut_zstar``: BayIW cut, z* on the 50 MITgcm levels (validation/bayiw_cut/cut.py), Δt 120 s;
* ``dome_10km_hybrid_kvlow_smag``: DOME 10 km, hybrid, constant closure with ν_v = 1e-4 (``cases.NU_V_LOW``,
  the ``kv_low`` variant) and Smagorinsky C = 0.15, Δt 600 s — the case that motivated the scheme.

The script checks that the term is active (max κ_shear > 0 after the timed steps) and times the closure ``update``
(wrapped on the instance) to print its share of the step, and the implicit vertical solves (``vertical_implicit``, ν and κ) per step.  Each (case, mode) runs in its own process
(``--one``), and the two modes alternate (off, on, off, on) to expose order effects.

    python -I scripts/shear_cost.py [--cases bayiw_cut_zstar dome_10km_hybrid_kvlow_smag] [--rounds 2]

Run alone, with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
import warnings
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

NWARM, NMEAS = 10, 50
C_SMAG = 0.15


def _bayiw(kw: dict):
    """BayIW cut z* (constant closure): model, state, step(model, state, n)."""
    from validation.bayiw_cut import cut
    from validation.bayiw_seiche.run_pyocean import wind_at
    m, st, wind = cut.build_model("2layer", "bay", "zstar", 120.0, **kw)
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))

    def step(m, st, n):
        tau[:-1] = wind_at(n * m.dt, wind)
        return m.step(st, tau_y=tau)
    return m, st, step


def _dome(kw: dict):
    """DOME 10 km hybrid, constant closure ν_v = NU_V_LOW, Smagorinsky 0.15."""
    from validation.dome import cases
    from validation.dome import run_pyocean as dome
    m, st = dome.build(cases.get(10.0), "hybrid", nu_v=cases.NU_V_LOW, closure="constant", smag=C_SMAG, **kw)
    return m, st, lambda m, st, n: m.step(st)


CASES = {"bayiw_cut_zstar": _bayiw, "dome_10km_hybrid_kvlow_smag": _dome}


def measure(name: str, shear: bool) -> dict:
    """Median s/step, the median time of the closure ``update`` per step and max κ_shear for one configuration."""
    from core.dynamics import sigma_baroclinic as sb
    sb.set_threads(5)
    solves = []                                                    # implicit vertical solves (ν and κ), all callers
    import core.dynamics.sigma_baroclinic as sbm
    import core.dynamics.sigma_wd as swd
    import core.model.driver as drv
    vi = swd.vertical_implicit

    def timed_vi(*a, **k):
        t0 = time.perf_counter()
        out = vi(*a, **k)
        solves.append(time.perf_counter() - t0)
        return out
    swd.vertical_implicit = drv.vertical_implicit = sbm.vertical_implicit = timed_vi
    m, st, step = CASES[name](dict(shear_mixing="jackson", shear_prandtl=1.0) if shear else {})
    clo = m._closure
    acc = []
    orig = clo.update

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = orig(*a, **k)
        acc.append(time.perf_counter() - t0)
        return out
    clo.update = timed
    for n in range(NWARM):
        st = step(m, st, n)
    acc.clear()
    solves.clear()
    t = []
    for n in range(NWARM, NWARM + NMEAS):
        t0 = time.perf_counter()
        st = step(m, st, n)
        t.append(time.perf_counter() - t0)
    kmax = float(np.max(clo.kd_shear)) if shear else 0.0
    if shear:
        assert len(acc) >= NMEAS, f"{name}: the wrapped closure update ran {len(acc)} times in {NMEAS} steps"
        assert kmax > 0.0, f"{name}: shear mixing is on but max κ_shear = 0 (term not active)"
        assert type(clo).__name__ == "ShearAddedClosure", type(clo).__name__
    return dict(step=float(np.median(t)), update=float(np.median(acc)) if acc else 0.0, kmax=kmax, ncall=len(acc),
                solve=float(sum(solves)) / NMEAS)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cases", nargs="+", default=list(CASES), choices=list(CASES))
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--one", nargs=2, metavar=("CASE", "MODE"), help="internal: one configuration, JSON on stdout")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    if a.one:
        print("RESULT " + json.dumps(measure(a.one[0], a.one[1] == "on")))
        return
    for name in a.cases:
        res = {"off": [], "on": []}
        for _ in range(a.rounds):
            for mode in ("off", "on"):
                out = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--one", name, mode],
                                     capture_output=True, text=True, check=True, cwd=ROOT).stdout
                res[mode].append(json.loads(out.split("RESULT ", 1)[1]))
                r = res[mode][-1]
                print(f"  {name} {mode:3s} {r['step']:.4f} s/step  update {r['update'] * 1e3:7.2f} ms"
                      f"  solves {r['solve'] * 1e3:6.2f} ms  kmax {r['kmax']:.3e}", flush=True)
        ratios = [on["step"] / off["step"] for on, off in zip(res["on"], res["off"])]
        t0 = np.median([r["step"] for r in res["off"]])
        t1 = np.median([r["step"] for r in res["on"]])
        u1 = np.median([r["update"] for r in res["on"]])
        v0 = np.median([r["solve"] for r in res["off"]])
        v1 = np.median([r["solve"] for r in res["on"]])
        print(f"{name:30s} off {t0:.4f}  on {t1:.4f} s/step  ratio min/median/max "
              f"{min(ratios):.3f}/{np.median(ratios):.3f}/{max(ratios):.3f}  "
              f"update(on) {u1 * 1e3:.1f} ms = {100 * u1 / t1:.1f} % of the step;  implicit vertical solves "
              f"{v0 * 1e3:.1f} -> {v1 * 1e3:.1f} ms per step", flush=True)


if __name__ == "__main__":
    main()
