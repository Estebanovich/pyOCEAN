"""
Run the pyOCEAN single-column cases (N3 task 8) and save hourly snapshots of one column, as CROCO's scm_his.nc
(NWRT = 120 steps of 30 s).

    python -m validation.turbulence_columns.run_pyocean kato_phillips sigma gls [--production energy]
    python -m validation.turbulence_columns.run_pyocean --matrix          # every run of the task-8 matrix

Outputs: ``/Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns/pyocean/<case>_<coord>_<closure>[_energy].npz``
(never into the repo) with, at t = 0, 1, 2, … h: time [s], zw (interface heights, (nt, N+1), z ≤ 0, k = 0 bed),
T, u, v (layers, (nt, N)), tke, eps, akv, akt (interfaces, (nt, N+1); KPP: akv and akt = κ_s, no tke/eps), hbl
(KPP boundary-layer depth [m]), the spread of the column over the 5 × 5 box (it must stay 1-D), and the wall time.

Runs one at a time; set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1.
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from validation.turbulence_columns.cases import CASES, KPP_CASES, NAMES, NWRT, RUNS, STRATIFIED, build

OUT = RUNS / "pyocean"


def run_tag(name: str, coord: str, closure: str, production: str = "croco") -> str:
    return f"{name}_{coord}_{closure}" + ("_energy" if production == "energy" else "")


def matrix() -> list[tuple[str, str, str, str]]:
    """The task-8 runs: every case in σ and ALE z* with the GLS, hybrid where there is stratification, KPP in the
    three surface cases (σ and z*), and Kato–Phillips with gls_production 'energy' in σ, z* and hybrid (with and
    without regrid)."""
    runs = []
    for name in NAMES:
        coords = ("sigma", "zstar", "hybrid") if name in STRATIFIED else ("sigma", "zstar")
        runs += [(name, c, "gls", "croco") for c in coords]
    runs += [(name, c, "kpp", "croco") for name in KPP_CASES for c in ("sigma", "zstar")]
    runs += [("kato_phillips", c, "gls", "energy") for c in ("sigma", "zstar", "hybrid")]
    return runs


def _snapshot(model, st, closure: str) -> dict:
    j = i = 0
    h = st.h[:, j, i] if st.h is not None else model.layer_volumes(st.eta)[:, j, i] / model.area[j, i]
    zw = np.concatenate([[0.0], np.cumsum(h)]) - h.sum()           # bed first, z ≤ 0 (η = 0 in a flat periodic box)
    out = dict(zw=zw, T=st.T[:, j, i].copy(), u=st.u[:, j, i].copy(), v=st.v[:, j, i].copy(),
               spread=max(float(np.ptp(st.T, axis=(1, 2)).max()), float(np.ptp(st.u, axis=(1, 2)).max()),
                          float(np.ptp(st.v, axis=(1, 2)).max())))
    n1 = st.T.shape[0] + 1
    if closure == "gls":
        for k in ("tke", "eps", "akv", "akt"):
            out[k] = np.asarray(st.closure[k]).reshape(n1, -1)[:, 0].copy()
        out["hbl"] = np.nan
    else:
        cl = model._closure
        akv = None if st.closure is None else st.closure.get("akv")
        out["akv"] = np.zeros(n1) if akv is None else np.asarray(akv).reshape(n1, -1)[:, 0].copy()
        kt = cl.tracer_diffusivity()
        out["akt"] = (np.zeros(n1) if kt is None else np.asarray(kt).reshape(n1, -1)[:, 0].copy())
        out["tke"] = out["eps"] = np.full(n1, np.nan)
        hbl = getattr(cl, "obl_depth", None)
        out["hbl"] = np.nan if hbl is None else float(np.ravel(hbl)[0])
    return out


def run(name: str, coord: str, closure: str, production: str = "croco", hours: float | None = None,
        save: bool = True) -> dict:
    """Run one case; returns the snapshot dict (and writes it to OUT when ``save``)."""
    warnings.simplefilter("ignore")
    case = CASES[name]
    model, st, forcing = build(name, coord, closure, gls_production=production)
    n_steps = case.n_steps if hours is None else int(round(hours * 3600.0 / model.dt))
    snaps = [dict(time=0.0, **_snapshot(model, st, closure))]
    t0 = time.perf_counter()
    for n in range(1, n_steps + 1):
        st = forcing.step(model, st)
        if n % NWRT == 0:
            if not (np.isfinite(st.T).all() and np.isfinite(st.u).all()):
                raise FloatingPointError(f"{name} {coord} {closure}: non-finite state at step {n}")
            snaps.append(dict(time=st.time, **_snapshot(model, st, closure)))
    wall = time.perf_counter() - t0
    out = {k: np.array([s[k] for s in snaps]) for k in snaps[0]}
    out.update(wall=wall, n_steps=n_steps, s_per_step=wall / n_steps, dz=case.dz, depth=case.depth)
    if save:
        OUT.mkdir(parents=True, exist_ok=True)
        np.savez(OUT / f"{run_tag(name, coord, closure, production)}.npz", **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("name", nargs="?", choices=NAMES)
    ap.add_argument("coord", nargs="?", choices=("sigma", "zstar", "hybrid"))
    ap.add_argument("closure", nargs="?", choices=("gls", "kpp"))
    ap.add_argument("--production", default="croco", choices=("croco", "energy"))
    ap.add_argument("--matrix", action="store_true", help="run the whole task-8 matrix, one run at a time")
    ap.add_argument("--skip-existing", action="store_true")
    a = ap.parse_args()
    runs = matrix() if a.matrix else [(a.name, a.coord, a.closure, a.production)]
    for name, coord, closure, prod in runs:
        tag = run_tag(name, coord, closure, prod)
        if a.skip_existing and (OUT / f"{tag}.npz").exists():
            continue
        out = run(name, coord, closure, prod)
        print(f"{tag}: {out['n_steps']} steps, {out['wall']:.1f} s wall ({1e3 * out['s_per_step']:.1f} ms/step), "
              f"max 5x5 spread {out['spread'].max():.1e}", flush=True)


if __name__ == "__main__":
    main()
