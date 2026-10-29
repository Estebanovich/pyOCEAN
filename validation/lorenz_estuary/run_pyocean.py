#!/usr/bin/env python
"""
pyOCEAN runs of the idealized estuary of Lorenz et al. (2021) (case.py): ``cycles`` M2 periods, the last ``avg`` of
them averaged by the in-model TEF and the region budget (Lorenz §4b: 500 cycles of spin-up + a 100-cycle mean), as
GETM does during the run.  Outputs (never in the repository) go to ``--out``/<exp>_<config>/:

  summary.json   time-mean TEF bulk values, region budget sums (Qr, Qsurf, dV, dS, dS2, F_surf^{s²}, M_phy, M_num,
                 M_tot, M_sponge), Me/Mc/Mp/Mcp (eqs. 32–35), s_surf (36), M_model_tot, run metadata and cost
  section.npz    final S, u, η and layer heights; thickness-weighted mean S and u over the averaging window
  series.npz     one record per 1/4 M2 cycle (η at the ocean edge, transect and river end; region volume and mean S;
                 max S; max |u|; n_limited)
  restart.nc + restart_extra.pkl   checkpoint every ``--checkpoint`` cycles (model state via core.restart, plus the
                 step counter, TEF/region accumulators, mean-section sums and series, which core.restart does not hold)

Progress lines go to stdout (flushed), so a background run can be followed with ``tail -f``.

Usage:
    python validation/lorenz_estuary/run_pyocean.py {A,B,C,D} {sigma_gls,ale_sigma,ale_sigma_gls} [--cycles 600] [--avg 100]
        [--out DIR] [--dt S] [--threads 5] [--checkpoint 25] [--resume] [--hours H]
``--hours H`` runs H simulated hours only and prints the cost per step (README, cost table); nothing is written.
"""
from __future__ import annotations

import argparse
import json
import pickle
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.model.threads import set_threads  # noqa: E402
from core.restart import read_restart, write_restart  # noqa: E402
from validation.lorenz_estuary import case  # noqa: E402

N_WARM = 3                       # steps excluded from the cost (Numba compilation)


def log(msg: str) -> None:
    print(msg, flush=True)


def record(m, st) -> list[float]:
    """One series record: t [s], η at i = 0, the transect and the river end [m], region volume [m³] and mean S,
    domain max S [g kg⁻¹], max |u| [m s⁻¹], n_limited."""
    reg = m.tef_region_budget.mask
    V = m.layer_volumes(st.eta, st.h)
    Vr = V[:, reg].sum()
    return [st.time, st.eta[0, 0], st.eta[0, m.tef_face], st.eta[0, -1], Vr, (V * st.S)[:, reg].sum() / Vr,
            float(np.nanmax(st.S)), float(np.abs(st.u).max()), float(st.n_limited)]


SERIES_COLS = ["t", "eta_ocean", "eta_transect", "eta_river", "V_region", "S_region", "S_max", "umax", "n_limited"]


def run(exp: str, config: str, cycles: float, avg: float, out: Path | None, dt: float | None, checkpoint: float,
        resume: bool, hours: float | None) -> dict:
    """Integrate, average and write (module docstring); returns the summary."""
    m, st, forcing = case.build(exp, config, dt=dt)
    dt = m.dt
    steps_cycle = case.T_M2 / dt
    n_total = int(round(hours * 3600.0 / dt)) if hours else int(round(cycles * steps_cycle))
    n_avg_start = n_total - int(round(avg * steps_cycle)) if not hours else 0
    every = max(1, int(round(steps_cycle / 4)))
    n_ckpt = max(1, int(round(checkpoint * steps_cycle)))
    rdir = None if hours else out / f"{exp}_{config}"
    run_id = dict(cycles=float(cycles), avg=float(avg), dt=float(dt))
    extra = dict(n=0, series=[], VS=None, Vu=None, V=None, tef=None, region=None, run_id=run_id)
    if resume and rdir is not None and (rdir / "restart.nc").exists():
        with open(rdir / "restart_extra.pkl", "rb") as fh:
            extra = pickle.load(fh)
        if extra.get("run_id") != run_id:
            raise SystemExit(f"--resume: checkpoint written with {extra.get('run_id')}, this run is {run_id}")
        st = read_restart(rdir / "restart.nc", m)
        m.tef, m.tef_region_budget = extra["tef"], extra["region"]
        log(f"resumed at step {extra['n']} (t = {st.time / 3600:.2f} h)")
    if rdir is not None:
        rdir.mkdir(parents=True, exist_ok=True)
    log(f"Lorenz estuary {exp} {config}: dt = {dt:.4f} s ({steps_cycle:.2f} steps per M2), {n_total} steps "
        f"({n_total * dt / case.T_M2:.1f} cycles), averaging from step {n_avg_start}")
    t_wall0, t_warm, n0 = time.perf_counter(), None, extra["n"]
    for n in range(extra["n"], n_total):
        if n == n_avg_start:
            case.reset_tef(m)
            extra["VS"] = extra["Vu"] = extra["V"] = None
            log(f"averaging window starts at t = {st.time / 3600:.2f} h")
        st = m.step(st, **forcing(n))
        if n == n0 + N_WARM - 1:
            t_warm = time.perf_counter()
        if n >= n_avg_start:
            V = m.layer_volumes(st.eta, st.h)
            uc = 0.5 * (st.u[..., 1:] + st.u[..., :-1])
            if extra["V"] is None:
                extra["V"], extra["VS"], extra["Vu"] = V.copy(), V * st.S, V * uc
            else:
                extra["V"] += V
                extra["VS"] += V * st.S
                extra["Vu"] += V * uc
        if (n + 1) % every == 0:
            extra["series"].append(record(m, st))
            if not np.isfinite(st.S).all() or not np.isfinite(st.u).all():
                log(f"BLOW-UP at step {n + 1}, t = {st.time / 3600:.2f} h")
                raise SystemExit(2)
        if (n + 1) % int(round(steps_cycle)) == 0 or n + 1 == n_total:
            el = time.perf_counter() - t_wall0
            per = el / (n + 1 - n0)
            r = extra["series"][-1] if extra["series"] else [np.nan] * len(SERIES_COLS)
            log(f"step {n + 1}/{n_total} t = {st.time / 3600:8.2f} h ({st.time / case.T_M2:6.1f} cycles)  "
                f"wall {el / 3600:6.2f} h  {per:.3f} s/step  ETA {(n_total - n - 1) * per / 3600:6.2f} h  "
                f"S_reg {r[5]:.3f} S_max {r[6]:.3f} |u|max {r[7]:.3f} n_lim {r[8]:.0f}")
        if rdir is not None and (n + 1) % n_ckpt == 0 and n + 1 < n_total:
            if not (np.isfinite(st.S).all() and np.isfinite(st.u).all()):     # never overwrite a good checkpoint
                log(f"BLOW-UP at step {n + 1}, t = {st.time / 3600:.2f} h (checkpoint not written)")
                raise SystemExit(2)
            extra.update(n=n + 1, tef=m.tef, region=m.tef_region_budget)
            write_restart(rdir / "restart.tmp.nc", m, st)               # write, then rename: a run killed
            with open(rdir / "restart_extra.tmp.pkl", "wb") as fh:       # mid-write keeps the last checkpoint
                pickle.dump(extra, fh)
            (rdir / "restart.tmp.nc").replace(rdir / "restart.nc")
            (rdir / "restart_extra.tmp.pkl").replace(rdir / "restart_extra.pkl")
    wall = time.perf_counter() - t_wall0
    n_run = n_total - n0
    per_step = (time.perf_counter() - t_warm) / max(n_run - N_WARM, 1) if t_warm else wall / max(n_run, 1)
    cost = dict(wall_s=wall, s_per_step=per_step, steps=n_run, dt=dt,
                full_run_h=per_step * int(round(case.SPINUP_CYCLES + case.AVG_CYCLES) * steps_cycle) / 3600.0)
    if hours:
        log(f"COST {exp} {config}: {per_step:.4f} s/step after {N_WARM} warm-up steps, dt = {dt:.3f} s, "
            f"{n_run} steps in {wall:.1f} s; 600 cycles → {cost['full_run_h']:.2f} h; "
            f"max|u| {np.abs(st.u).max():.3f} m/s, S [{np.nanmin(st.S):.3f}, {np.nanmax(st.S):.3f}], "
            f"eta_ocean range {np.ptp([r[1] for r in extra['series']]) if extra['series'] else 0:.3f} m")
        return cost
    res = case.summary(m)
    res.update(exp=exp, config=config, cycles=cycles, avg=avg, cost=cost, table_1=case.TABLE_1[exp],
               transect_face=m.tef_face, n_limited=int(st.n_limited))
    with open(rdir / "summary.json", "w") as fh:
        json.dump(res, fh, indent=1, default=float)
    np.savez(rdir / "section.npz", S=st.S, u=st.u, eta=st.eta, z=m.layer_heights(st.eta, st.h), H=m.H,
             S_mean=extra["VS"] / extra["V"], u_mean=extra["Vu"] / extra["V"], V_mean=extra["V"],
             x=(np.arange(case.NX) + 0.5) * case.DX)
    np.savez(rdir / "series.npz", data=np.array(extra["series"]), columns=np.array(SERIES_COLS))
    log(f"DONE {exp} {config}: Me {res['Me']:.6g}  M_model_tot {res['M_model_tot']:.6g}  "
        f"|Me − M|/M {abs(res['Me'] - res['M_model_tot']) / abs(res['M_model_tot']):.2e}  wall {wall / 3600:.2f} h")
    return res


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("exp", choices=sorted(case.TABLE_1))
    p.add_argument("config", choices=case.CONFIGS)
    p.add_argument("--cycles", type=float, default=case.SPINUP_CYCLES + case.AVG_CYCLES)
    p.add_argument("--avg", type=float, default=case.AVG_CYCLES)
    p.add_argument("--out", type=Path, default=case.RUNS)
    p.add_argument("--dt", type=float, default=None, help="time step [s] (default case.DT_DEFAULT[config])")
    p.add_argument("--threads", type=int, default=5)
    p.add_argument("--checkpoint", type=float, default=25.0, help="checkpoint interval [M2 cycles]")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--hours", type=float, default=None, help="cost measurement: run H simulated hours only")
    a = p.parse_args()
    if a.avg > a.cycles:
        p.error("--avg must not exceed --cycles")
    set_threads(a.threads)
    run(a.exp, a.config, a.cycles, a.avg, a.out, a.dt, a.checkpoint, a.resume, a.hours)


if __name__ == "__main__":
    main()
