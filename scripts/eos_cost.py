#!/usr/bin/env python
"""
Cost of the nonlinear equation of state in the 3-D step (plan docs/superpowers/plans/2026-10-31-teos10-3d.md,
task 10; spec criterion 11, Esteban's decision 4: ratio <= 2.5 per step for ``wright_full`` and ``teos10`` against
``linear``).  Timing protocol of ``scripts/bulk_cost.py``: median wall time per step of 50 steps after 10 warm-up
steps, 5 threads, one process per mode (``--one``), modes alternating (linear, wright_full, teos10, linear, ...),
3 rounds.

Case: BayIW cut, z* on the 50 MITgcm levels (``validation/bayiw_cut/cut.py`` ``build_model``, same arguments),
Delta t 120 s, ``salinity=True`` with S = 35 uniform in every mode, no surface forcing:

* ``linear``: the case's linear EOS (rho0 999.8, alpha rho0 = T_COEF, t_ref 11.4) plus s_coef 0.78, s_ref 35;
* ``wright_full``: ``eos='wright_full'`` (MOM6 WRIGHT_FULL, physics/eos/wright_full.py), rho0 1035 (default);
* ``teos10``: ``eos='teos10'`` (Roquet et al. 2015 polynomial, physics/eos/roquet_rho.py), rho0 1035; the
  MITgcm levels are read as potential temperature and practical salinity (``ts_kind='theta_sp'``, converted once).

``rho_ref='auto'`` in every mode (the T/S reference profile with a nonlinear EOS).  The per-phase breakdown times
every call of the ten driver phases (``Driver._ale_<phase>``, ``core.model.driver.PHASES``) with instance wrappers
and reports the mean time per step of each phase, so the extra cost can be placed (PGF in ``explicit_momentum``,
closure/regrid density in ``tracers``/``vertical_coordinate``).

    python -I scripts/eos_cost.py [--rounds 3]

Run alone on an otherwise idle machine, with OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1
MKL_NUM_THREADS=1.
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
DT = 120.0
MODES = ("linear", "wright_full", "teos10")
S0 = 35.0


def _build(eos: str):
    """BayIW cut z* as ``cut.build_model('2layer', 'bay', 'zstar', 120)``, with salinity and the given EOS."""
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid_stretched
    from validation.bayiw_baroclinic import bayiw
    from validation.bayiw_cut import cut
    d = cut.cut_inputs("2layer", "bay", "py")
    hg = create_cartesian_grid_stretched(d["dx"], d["dy"], x0=d["x0"], y0=d["y0"], f0=bayiw.F0)
    if eos == "linear":
        ekw = dict(eos="linear", rho0=bayiw.RHO0, t_coef=bayiw.T_COEF, t_ref=bayiw.CASES["2layer"]["t_ref"],
                   s_coef=0.78, s_ref=S0)
    else:
        ekw = dict(eos=eos)
    m = SigmaBaroclinic(hg, d["H"], dt=DT, n_layers=bayiw.DELR.size, z0=None, closure="constant", nu_const=1e-5,
                        ah=100.0, kh=100.0, kv=1e-5, pgf="fv", rho_ref="auto", vertical_coordinate="ale",
                        regrid="zstar", regrid_params=dict(dz_top=bayiw.DELR.copy()),
                        open_boundaries=("west", "east", "south"), orlanski_cmax=0.45, sponge_cells=30,
                        sponge_tau=(1200.0, 60.0), sponge_tracer=False, mix_diagnostic=False, salinity=True, **ekw)
    h_src, T_src = bayiw.mitgcm_source(d["H"], d["T"])
    st = m.initial_state(np.zeros(d["H"].shape), T_src, h_src=h_src, S=np.full(np.shape(T_src), S0))
    return m, st


def _wrap(obj, name: str, acc: list) -> None:
    """Time every call of the bound method ``obj.name`` (instance wrapper; ``Driver.step`` uses getattr)."""
    orig = getattr(obj, name)

    def timed(*a, **k):
        t0 = time.perf_counter()
        out = orig(*a, **k)
        acc.append(time.perf_counter() - t0)
        return out
    setattr(obj, name, timed)


def measure(eos: str) -> dict:
    """Median and mean s/step and the mean time per step of each driver phase."""
    from core.dynamics import sigma_baroclinic as sb
    from core.model.driver import PHASES
    sb.set_threads(5)
    m, st = _build(eos)
    assert m.eos_name == eos, f"model EOS {m.eos_name!r} != {eos!r}"
    acc = {p: [] for p in PHASES}
    for p in PHASES:
        _wrap(m._driver, "_ale_" + p, acc[p])
    for _ in range(NWARM):
        st = m.step(st)
    for a in acc.values():
        a.clear()
    t = []
    for _ in range(NMEAS):
        t0 = time.perf_counter()
        st = m.step(st)
        t.append(time.perf_counter() - t0)
    assert np.all(np.isfinite(st.eta)) and np.all(np.isfinite(st.T[np.isfinite(st.T)])), "state not finite"
    for p in PHASES:
        assert len(acc[p]) == NMEAS, f"phase {p} ran {len(acc[p])} times in {NMEAS} steps"
    return dict(step=float(np.median(t)), mean=float(np.mean(t)),
                phases={p: float(sum(acc[p])) / NMEAS for p in PHASES})


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=3)
    ap.add_argument("--one", choices=MODES, help="internal: one mode, JSON on stdout")
    a = ap.parse_args()
    warnings.simplefilter("ignore")
    if a.one:
        print("RESULT " + json.dumps(measure(a.one)))
        return
    from core.model.driver import PHASES
    name = "bayiw_cut_zstar_salinity"
    res = {mode: [] for mode in MODES}
    for _ in range(a.rounds):
        for mode in MODES:
            out = subprocess.run([sys.executable, "-I", str(Path(__file__).resolve()), "--one", mode],
                                 capture_output=True, text=True, check=True, cwd=ROOT).stdout
            res[mode].append(json.loads(out.split("RESULT ", 1)[1]))
            r = res[mode][-1]
            print(f"  {name} {mode:11s} median {r['step']:.4f} mean {r['mean']:.4f} s/step", flush=True)
    med = lambda mode, key: float(np.median([r[key] for r in res[mode]]))          # noqa: E731
    medp = lambda mode, p: float(np.median([r["phases"][p] for r in res[mode]]))    # noqa: E731
    out = {}
    for mode in MODES[1:]:
        ratios = [r["step"] / l["step"] for r, l in zip(res[mode], res["linear"])]
        out[mode] = ratios
        print(f"{name}  {mode:11s} {med(mode, 'step'):.4f} vs linear {med('linear', 'step'):.4f} s/step (median)  "
              f"ratio min/median/max {min(ratios):.3f}/{np.median(ratios):.3f}/{max(ratios):.3f}  "
              f"(criterion 11: <= 2.5)", flush=True)
    print("\nper-phase mean time per step [ms] (median over rounds)")
    print(f"  {'phase':28s}" + "".join(f"{m_:>13s}" for m_ in MODES) + "".join(f"{'d ' + m_:>16s}" for m_ in MODES[1:]))
    for p in PHASES:
        v = [medp(m_, p) * 1e3 for m_ in MODES]
        print(f"  {p:28s}" + "".join(f"{x:13.2f}" for x in v) + "".join(f"{x - v[0]:16.2f}" for x in v[1:]))
    tot = [sum(medp(m_, p) for p in PHASES) * 1e3 for m_ in MODES]
    print(f"  {'sum of phases':28s}" + "".join(f"{x:13.2f}" for x in tot) + "".join(f"{x - tot[0]:16.2f}"
                                                                                       for x in tot[1:]))
    print("RATIOS " + json.dumps(dict(ratios=out, res=res)))


if __name__ == "__main__":
    main()
