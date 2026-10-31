#!/usr/bin/env python
"""
Cost of the drag-driven BBL mixing (plan 2026-10-30-bbl-drag-mixing, task 5; spec criterion 6: ratio <= 1.05).

Median wall time per step of 50 steps after 10 warm-up steps, 5 threads, without (``bbl_mixing=None``) and with
``bbl_mixing='lotw'`` (``physics.mixing.bbl_mixing``, MOM6 ``add_LOTW_BBL_diffusivity``), same case otherwise:

* ``bayiw_cut_zstar``: BayIW cut, z* on the 50 MITgcm levels (validation/bayiw_cut/cut.py), Delta t 120 s, with
  ``bottom_drag='quadratic'`` in both modes (the BBL mixing needs a bed drag; the ratio isolates the mixing);
* ``dome_10km_hybrid_kvlow_smag_jackson``: DOME 10 km, hybrid, constant closure with nu_v = 1e-4 (``cases.NU_V_LOW``),
  Smagorinsky C = 0.15 and ``shear_mixing='jackson'`` (Pr = 1), Delta t 600 s (the drag is the case's own quadratic one).

The script checks that the term is active (max Kd_BBL > 0 in the timed steps, ``BBLAddedClosure``) and prints the
share of the step spent in (i) the whole closure ``update``, (ii) the BBL part of it (outer ``update`` minus the
wrapped closure's), (iii) the drag-work energy of the driver (``Driver._ale_bed_energy``).  Each (case, mode) runs in
its own process (``--one``) and the modes alternate (off, on, off, on, ...) to expose order effects.

    python -I scripts/bbl_cost.py [--cases ...] [--rounds 3]

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
    """BayIW cut z* (constant closure), quadratic bed drag: model, state, step(model, state, n)."""
    from validation.bayiw_cut import cut
    from validation.bayiw_seiche.run_pyocean import wind_at
    m, st, wind = cut.build_model("2layer", "bay", "zstar", 120.0, bottom_drag="quadratic", **kw)
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))

    def step(m, st, n):
        tau[:-1] = wind_at(n * m.dt, wind)
        return m.step(st, tau_y=tau)
    return m, st, step


def _dome(kw: dict):
    """DOME 10 km hybrid, constant closure nu_v = NU_V_LOW, Smagorinsky 0.15, Jackson shear mixing (Pr = 1)."""
    from validation.dome import cases
    from validation.dome import run_pyocean as dome
    m, st = dome.build(cases.get(10.0), "hybrid", nu_v=cases.NU_V_LOW, closure="constant", smag=C_SMAG,
                       shear_mixing="jackson", shear_prandtl=1.0, **kw)
    return m, st, lambda m, st, n: m.step(st)


CASES = {"bayiw_cut_zstar": _bayiw, "dome_10km_hybrid_kvlow_smag_jackson": _dome}


def _wrap(obj, name: str, acc: list) -> None:
    """Time every call of the bound method ``obj.name`` (instance wrapper)."""
    orig = getattr(obj, name)

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = orig(*a, **k)
        acc.append(time.perf_counter() - t0)
        return out
    setattr(obj, name, timed)


def measure(name: str, bbl: bool) -> dict:
    """Median s/step, per-step time in the closure update, its BBL part and the drag energy; max Kd_BBL."""
    from core.dynamics import sigma_baroclinic as sb
    sb.set_threads(5)
    m, st, step = CASES[name](dict(bbl_mixing="lotw") if bbl else {})
    clo = m._closure
    outer, inner, energy = [], [], []
    _wrap(clo, "update", outer)
    if bbl:
        _wrap(clo.base, "update", inner)
    _wrap(m._driver, "_ale_bed_energy", energy)
    for n in range(NWARM):
        st = step(m, st, n)
    for a in (outer, inner, energy):
        a.clear()
    t, kmax = [], 0.0
    for n in range(NWARM, NWARM + NMEAS):
        t0 = time.perf_counter()
        st = step(m, st, n)
        t.append(time.perf_counter() - t0)
        if bbl:
            kmax = max(kmax, float(np.max(clo.kd_bbl)))
    if bbl:
        assert type(clo).__name__ == "BBLAddedClosure", type(clo).__name__
        assert len(outer) >= NMEAS, f"{name}: closure update ran {len(outer)} times in {NMEAS} steps"
        assert kmax > 0.0, f"{name}: BBL mixing is on but max Kd_BBL = 0 in the timed steps (term not active)"
    per_step = lambda a: float(sum(a)) / NMEAS                      # total time per step (several calls possible)
    bbl_part = per_step(outer) - per_step(inner) if bbl else 0.0
    return dict(step=float(np.median(t)), update=per_step(outer), bbl_update=bbl_part,
                energy=per_step(energy) if bbl else 0.0, kmax=kmax, ncall=len(outer))


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
                      f"  bbl part {r['bbl_update'] * 1e3:6.2f} ms  drag energy {r['energy'] * 1e3:6.2f} ms"
                      f"  Kd_BBL max {r['kmax']:.3e}", flush=True)
        ratios = [on["step"] / off["step"] for on, off in zip(res["on"], res["off"])]
        med = lambda mode, key: float(np.median([r[key] for r in res[mode]]))
        t0, t1 = med("off", "step"), med("on", "step")
        print(f"{name:38s} off {t0:.4f}  on {t1:.4f} s/step  ratio min/median/max "
              f"{min(ratios):.3f}/{np.median(ratios):.3f}/{max(ratios):.3f}  "
              f"BBL update {med('on', 'bbl_update') * 1e3:.1f} ms = {100 * med('on', 'bbl_update') / t1:.1f} % of the step; "
              f"drag energy {med('on', 'energy') * 1e3:.1f} ms = {100 * med('on', 'energy') / t1:.1f} %;  "
              f"Kd_BBL max {med('on', 'kmax'):.3e}", flush=True)


if __name__ == "__main__":
    main()
