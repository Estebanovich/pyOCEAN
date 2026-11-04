#!/usr/bin/env python
"""
pyOCEAN runs of the NWA12 cut (plan Task 9; spec 2026-11-01-regional-chain criteria 4 and 6; case.py).

Modes
-----
--mode tides   homogeneous ocean at rest, TPXO10 boundary tide + tidal potential + SAL only (criterion 4), 30 days
--mode full    the parity configuration (GLORYS initial state and exterior, ERA5 bulk, GloFAS rivers, tides), 123
               days once ERA5 covers 2017-05-01 → 2017-09-01

Each run writes to ``RUNS/runs/<mode>_dt<Δt>[_<tag>]/``:
  out/out_<time>.nc  CF NetCDF4 output (η, T, S, u, v, layer thickness), one file per record every ``--out-hours``
                  (default 24 h; one file each because Zarr's metadata consolidation trips over the AppleDouble
                  ``._*`` files macOS writes on the external run volume)
  series.npz      every step: η at the CO-OPS gauges and the buoys (nearest wet T cell), SST and SSS there, and the
                  largest |u|, |v| of the domain (the upper-ocean mode of θ_bc = ½ shows here, README)
  budget.npz      domain volume, heat and salt budgets (diagnostics.domain_budget; full mode), per-run sums
  meta.json       Δt, steps, CFL truncations, limiter events, Smagorinsky-limited points, cost per simulated day
  restart_dayD.nc with ``--restart-at D`` (core.restart, bit-for-bit; D may be fractional, e.g. 60.5 = between two
                  daily GLORYS/GloFAS records), with restart_dayD_budget.npz (the budget accumulators, reloaded
                  by ``--start-from``)

Checks
------
--restart-check   run D days straight writing a restart at D/2, then a fresh model from the restart to D; the two
                  final states must be identical (η, u, v, T, S, h, closure state)
--threads-check H run H hours with 1 thread and with ``--threads``; the final states must be identical

On a blow-up (non-finite or |η| > 20 m, |u| > 10 m/s) ``diagnostics.postmortem.BlowupGuard`` reports the location and
dumps the state (blowup.npz).

The 123-day parity run, once ERA5 is complete (README):

    export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 NUMBA_NUM_THREADS=5
    python validation/nwa12_cut/run_pyocean.py --mode full --days 123 --dt 300 --threads 5 --restart-at 60.5

Restart check of a run with ``--restart-at D`` (continue from the restart, then compare the two final states):

    python validation/nwa12_cut/run_pyocean.py --mode full --days 123 --dt 300 \
        --start-from RUNS/runs/full_dt300/restart_day60.5.nc --tag from_restart
    python validation/nwa12_cut/run_pyocean.py --mode full --compare RUNS/runs/full_dt300/final.nc \
        RUNS/runs/full_dt300_from_restart/final.nc
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from core.restart import read_restart, write_restart  # noqa: E402
from diagnostics.domain_budget import DomainBudget  # noqa: E402
from diagnostics.postmortem import BlowupGuard  # noqa: E402
from validation.nwa12_cut import case as cs  # noqa: E402

N_WARM = 3                       # steps excluded from the cost (Numba compilation)
#: largest stable Δt measured (Task 9, never tuned): 600 s is stable with tides only; the full run at 600 s grows the
#: upper-ocean mode of θ_bc = ½ from day ~25 (README), 300 s does not
DEFAULT_DT = {"tides": 600.0, "full": 300.0}
STATE_FIELDS = ("eta", "u", "v", "T", "S", "h")


def run_dir(case: cs.Case, mode: str, dt: float, tag: str = "") -> Path:
    return case.runs / "runs" / (f"{mode}_dt{dt:g}" + (f"_{tag}" if tag else ""))


def same_state(a, b) -> dict[str, bool]:
    """Bit-for-bit comparison of two SigmaState (prognostic fields, closure state, counters)."""
    out = {k: bool(np.array_equal(getattr(a, k), getattr(b, k), equal_nan=True)) for k in STATE_FIELDS}
    ca, cb = a.closure or {}, b.closure or {}
    out["closure"] = set(ca) == set(cb) and all(np.array_equal(ca[k], cb[k], equal_nan=True) for k in ca)
    out["time"] = a.time == b.time
    out["counters"] = (a.n_trunc, a.n_limited) == (b.n_trunc, b.n_limited)
    return out


class Recorder:
    """Point series every step and CF output every ``out_every`` steps."""

    def __init__(self, b: cs.Built, out: Path | None, out_every: int, start: str) -> None:
        self.b, self.out_every = b, out_every
        self.names = list(b.cells)
        self.jj = np.array([b.cells[n][0] for n in self.names], int)
        self.ii = np.array([b.cells[n][1] for n in self.names], int)
        self.t, self.eta, self.sst, self.sss, self.umax = [], [], [], [], []
        self.out, self.start = (None if out is None else out / "out"), start
        if self.out is not None:
            self.out.mkdir(exist_ok=True)

    def __call__(self, n: int, st) -> None:
        self.t.append(st.time)
        self.umax.append(max(float(np.abs(st.u).max()), float(np.abs(st.v).max())))
        self.eta.append(st.eta[self.jj, self.ii].copy())
        self.sst.append(st.T[-1, self.jj, self.ii].copy())
        self.sss.append(st.S[-1, self.jj, self.ii].copy() if st.S is not None else np.full(len(self.names), np.nan))
        if self.out is not None and n % self.out_every == 0:
            from diagnostics.output import OutputWriter
            stamp = str(np.datetime64(self.start, "s") + np.timedelta64(int(round(st.time)), "s")).replace(":", "")
            w = OutputWriter(self.out / f"out_{stamp}.nc", self.b.hg, self.b.H, n_layers=self.b.model.n,
                             start=self.start, fmt="netcdf", title="pyOCEAN NWA12 cut", model=self.b.model)
            w.write(st.time, eta=st.eta, u=st.u, v=st.v, T=st.T, S=st.S, h_layer=st.h)
            w.close()

    def save(self, path: Path) -> None:
        np.savez(path, t=np.array(self.t), names=np.array(self.names), j=self.jj, i=self.ii,
                 lat=self.b.hg["lat_t"].values[self.jj, self.ii], lon=self.b.hg["lon_t"].values[self.jj, self.ii],
                 eta=np.array(self.eta), sst=np.array(self.sst), sss=np.array(self.sss), umax=np.array(self.umax))


def budget_path(restart_path: Path) -> Path:
    """Domain-budget accumulators saved next to a restart file (``<stem>_budget.npz``)."""
    return Path(restart_path).with_name(Path(restart_path).stem + "_budget.npz")


def integrate(b: cs.Built, st, n0: int, n1: int, *, budget: DomainBudget | None = None, rec: Recorder | None = None,
              guard: BlowupGuard | None = None, restart_at: int | None = None, restart_path: Path | None = None,
              log=print):
    """Steps n0 → n1 (step n ends at n·Δt); returns (state, wall seconds of steps after the warm-up, steps timed)."""
    m = b.model
    tic, timed = None, 0
    t_last = time.perf_counter()
    for n in range(n0 + 1, n1 + 1):
        if n == n0 + N_WARM + 1:
            tic = time.perf_counter()
        st = m.step(st)
        if tic is not None:
            timed += 1
        if budget is not None:
            budget.add(st)
        if rec is not None:
            rec(n, st)
        if guard is not None:
            guard.check(step=n, time=st.time, eta=st.eta, u=st.u, v=st.v, T=st.T)
        if restart_at is not None and n == restart_at:
            write_restart(restart_path, m, st)
            if budget is not None:                                    # the budget accumulators go with it
                np.savez(budget_path(restart_path), **budget.state())
            log(f"restart written at step {n} ({st.time / 86400:g} d): {restart_path}")
        if time.perf_counter() - t_last > 600.0:
            t_last = time.perf_counter()
            log(f"step {n}/{n1} day {st.time / 86400:.2f}: |η|max {np.nanmax(np.abs(st.eta)):.3f} m, "
                f"trunc {st.n_trunc}, limited {st.n_limited}")
    wall = 0.0 if tic is None else time.perf_counter() - tic
    return st, wall, timed


def _guard(b: cs.Built, out: Path | None) -> BlowupGuard:
    dx = float(np.nanmin(b.hg["dx_t"].values))
    return BlowupGuard(limits={"eta": 20.0, "u": 10.0, "v": 10.0}, dump_path=None if out is None else
                       out / "blowup.npz", lat=b.hg["lat_t"].values, lon=b.hg["lon_t"].values, depth=b.H,
                       dt=b.model.dt, dx=dx)


def meta_of(b: cs.Built, st, mode: str, dt: float, n_steps: int, wall: float, timed: int, threads: int) -> dict:
    s_per_step = wall / max(timed, 1)
    return dict(mode=mode, dt=dt, n_steps=n_steps, days=n_steps * dt / 86400.0, threads=threads,
                wall_s_per_step=s_per_step, wall_s_per_sim_day=s_per_step * 86400.0 / dt,
                n_trunc=int(st.n_trunc), n_limited=int(st.n_limited), n_visc_limited=int(st.n_visc_limited or 0),
                eta_absmax=float(np.nanmax(np.abs(st.eta))), shape=list(st.T.shape), edges=list(b.edges),
                window=[int(x) for x in b.hg.attrs.get("window", [])],
                rivers=0 if b.river is None else int(b.river.cells.shape[0]))


def run(case: cs.Case, mode: str, days: float, dt: float, threads: int = 5, restart_at: float | None = None,
        tag: str = "", out_hours: float = 24.0, write: bool = True, start_from: Path | None = None,
        log=print, overrides: dict | None = None, allow_restart_change: tuple[str, ...] = (),
        deep_sponge: float | None = None, deep_sponge_3d: bool = False, perturb_T: float | None = None,
        perturb_seed: int = 1) -> dict:
    """
    One run (module docstring); returns the metadata (and writes them).  ``overrides``: ``SigmaBaroclinic``
    arguments replacing the parity values (``--override KEY=VALUE``; diagnostics only, recorded in ``meta.json``);
    ``allow_restart_change``: restart configuration entries allowed to differ on ``start_from``
    (``core.restart.read_restart(allow_changed=...)``); ``deep_sponge``: τ [days] of the diagnostic interior T/S
    sponge over H > 500 m (``case.deep_sponge``, ``--deep-sponge``; the domain budget is then not computed, the
    sponge is not a boundary flux); ``deep_sponge_3d``: its target per column (GLORYS May–June mean at each column,
    ``case.deep_sponge(per_column=True)``) instead of one profile; ``perturb_T``: amplitude [°C] of seeded uniform
    noise added to the initial θ of every wet cell (``case.perturb_temperature``, ``--perturb-T``; diagnostic twin
    run for intrinsic variability, not with ``start_from``).
    """
    overrides = dict(overrides or {})
    sponge_kw, sponge_meta = {}, None
    if deep_sponge is not None:
        hg, H = cs.grid(case)
        sponge_kw, sponge_meta = (cs.deep_sponge(case, hg, H, tau_days=deep_sponge, per_column=True)
                                  if deep_sponge_3d else cs.deep_sponge(case, hg, H, tau_days=deep_sponge))
    set_threads(threads)
    out = run_dir(case, mode, dt, tag) if write else None
    if out is not None:
        out.mkdir(parents=True, exist_ok=True)
    n_steps = int(round(days * 86400.0 / dt))
    check_coverage(case, mode, days)
    b = cs.build(case, mode, dt, days, **overrides, **sponge_kw)
    st, n0 = b.state, 0
    if perturb_T is not None:
        if start_from is not None:
            raise SystemExit("--perturb-T perturbs the initial state: not with --start-from")
        st.T = cs.perturb_temperature(st.T, b.H, perturb_T, perturb_seed)
    if start_from is not None:
        st = read_restart(start_from, b.model, allow_changed=tuple(allow_restart_change))
        n0 = int(round(st.time / dt))
        log(f"started from {start_from} at step {n0}")
    budget, budget_scope = None, None
    if mode == "full" and sponge_meta is None:
        budget = DomainBudget(b.model)
        bp = None if start_from is None else budget_path(start_from)
        if bp is not None and bp.exists():                           # continue the accumulation of the first leg
            with np.load(bp) as d:
                budget.load(d)
            budget_scope = f"from t0, accumulators reloaded from {bp.name}"
        else:
            budget.start(st)
            budget_scope = "from t0" if start_from is None else "from the restart only (no saved accumulators)"
    rec = Recorder(b, out, max(1, int(round(out_hours * 3600.0 / dt))), str(case.t0)) if write else None
    if rec is not None and n0 == 0:
        rec(0, st)
    ra = None if restart_at is None else int(round(restart_at * 86400.0 / dt))
    rpath = None if out is None or ra is None else out / f"restart_day{restart_at:g}.nc"
    st, wall, timed = integrate(b, st, n0, n_steps, budget=budget, rec=rec, guard=_guard(b, out), restart_at=ra,
                                restart_path=rpath, log=log)
    meta = meta_of(b, st, mode, dt, n_steps - n0, wall, timed, threads)
    if rec is not None and rec.umax:
        meta["u_absmax_run"] = float(np.max(rec.umax))
    if overrides:
        meta["overrides"] = {k: repr(v) for k, v in overrides.items()}
    if allow_restart_change:
        meta["allow_restart_change"] = list(allow_restart_change)
    if sponge_meta is not None:
        meta["deep_sponge"] = sponge_meta
    if perturb_T is not None:
        meta["perturb_T"] = dict(amp=perturb_T, seed=perturb_seed, kind="uniform, every wet cell and layer")
    if budget is not None:
        meta["budget"] = budget.residuals()
        meta["budget_scope"] = budget_scope
    if write:
        rec.save(out / "series.npz")
        if budget is not None:
            np.savez(out / "budget.npz", **budget.state())
        write_restart(out / "final.nc", b.model, st)
        (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def check_coverage(case: cs.Case, mode: str, days: float) -> None:
    """The full mode needs ERA5 to the end of the run (the download may still be in progress): say so at once."""
    if mode != "full":
        return
    end = case.t0 + np.timedelta64(int(round(days * 86400)), "s")
    cov = cs.era5_coverage(case)
    if cov < end:
        raise SystemExit(f"ERA5 in {case.era5} covers {case.t0} … {cov} (months with both instant and accum files); "
                         f"the run needs it to {end}: wait for the download or pass --days "
                         f"{float((cov - case.t0) / np.timedelta64(1, 'D')):.4g} or less")


def restart_check(case: cs.Case, mode: str, days: float, dt: float, threads: int = 5, log=print) -> dict:
    """D days straight (restart written at D/2) vs a fresh model from that restart to D: bit for bit?"""
    set_threads(threads)
    out = run_dir(case, mode, dt, "restart_check")
    out.mkdir(parents=True, exist_ok=True)
    n = int(round(days * 86400.0 / dt))
    check_coverage(case, mode, days)
    half = n // 2
    rpath = out / f"restart_step{half}.nc"
    b = cs.build(case, mode, dt, days)
    bud = DomainBudget(b.model) if mode == "full" else None
    if bud is not None:
        bud.start(b.state)
    ref, wall, timed = integrate(b, b.state, 0, n, budget=bud, guard=_guard(b, None), restart_at=half,
                                 restart_path=rpath, log=log)
    b2 = cs.build(case, mode, dt, days)
    st2 = read_restart(rpath, b2.model)
    st2, _, _ = integrate(b2, st2, half, n, guard=_guard(b2, None), log=log)
    same = same_state(ref, st2)
    meta = dict(check="restart", mode=mode, days=days, dt=dt, restart_step=half, identical=same,
                all_identical=all(same.values()), **{k: v for k, v in meta_of(b, ref, mode, dt, n, wall, timed,
                                                                             threads).items()
                                                     if k in ("wall_s_per_sim_day", "n_trunc", "n_limited")})
    if bud is not None:
        meta["budget"] = bud.residuals()
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def threads_check(case: cs.Case, mode: str, hours: float, dt: float, threads: int = 5, log=print) -> dict:
    """``hours`` with 1 thread and with ``threads``: identical final states?"""
    n = int(round(hours * 3600.0 / dt))
    res = {}
    for th in (1, threads):
        set_threads(th)
        b = cs.build(case, mode, dt, max(hours / 24.0, dt / 86400.0))
        st, wall, timed = integrate(b, b.state, 0, n, guard=_guard(b, None), log=log)
        res[th] = (st, wall / max(timed, 1))
    same = same_state(res[1][0], res[threads][0])
    meta = dict(check="threads", mode=mode, hours=hours, dt=dt, threads=[1, threads], identical=same,
                all_identical=all(same.values()), s_per_step={str(k): v[1] for k, v in res.items()})
    out = run_dir(case, mode, dt, "threads_check")
    out.mkdir(parents=True, exist_ok=True)
    (out / "meta.json").write_text(json.dumps(meta, indent=1, default=float))
    return meta


def compare_restarts(a: Path, b: Path) -> dict:
    """Bit-for-bit comparison of every variable of two restart files (``final.nc`` of a straight run and of a run
    continued from its mid-run restart)."""
    import xarray as xr
    with xr.open_dataset(a) as da, xr.open_dataset(b) as db:
        names = sorted(set(da.data_vars) | set(db.data_vars))
        same = {n: bool(n in da and n in db and np.array_equal(da[n].values, db[n].values, equal_nan=True))
                for n in names}
    return dict(check="restart_files", a=str(a), b=str(b), all_identical=all(same.values()),
                different=[n for n, ok in same.items() if not ok], n_variables=len(names))


def parse_overrides(items: list[str]) -> dict:
    """``["theta_bc=1.0", …]`` → {"theta_bc": 1.0, …}; values parsed with ``ast.literal_eval`` (strings stay strings
    when they are not literals)."""
    import ast
    out = {}
    for item in items:
        if "=" not in item:
            raise SystemExit(f"--override {item!r}: use KEY=VALUE")
        k, v = item.split("=", 1)
        try:
            out[k.strip()] = ast.literal_eval(v.strip())
        except (ValueError, SyntaxError):
            out[k.strip()] = v.strip()
    return out


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=("tides", "full"), required=True)
    ap.add_argument("--days", type=float, default=None, help="default: 30 (tides), 123 (full)")
    ap.add_argument("--dt", type=float, default=None,
                    help="time step [s]; default the largest stable one measured: 600 (tides), 300 (full; README)")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--restart-at", type=float, default=None, help="write a restart at this day")
    ap.add_argument("--start-from", type=Path, default=None, help="continue from this restart file")
    ap.add_argument("--tag", default="")
    ap.add_argument("--out-hours", type=float, default=24.0)
    ap.add_argument("--restart-check", action="store_true")
    ap.add_argument("--threads-check", type=float, default=None, metavar="HOURS")
    ap.add_argument("--tide-velocity-depth", choices=("model", "tpxo"), default=None,
                    help="BoundaryTides velocity_depth (default: the case's, 'tpxo' (R19); 'model' = U/D_model, R3)")
    ap.add_argument("--override", action="append", default=[], metavar="KEY=VALUE",
                    help="replace a parity SigmaBaroclinic argument (value parsed with ast.literal_eval), repeatable; "
                         "diagnostics only, e.g. --override theta_bc=1.0")
    ap.add_argument("--allow-restart-change", action="append", default=[], metavar="NAME",
                    help="restart configuration entry allowed to differ on --start-from (e.g. theta_bc), repeatable")
    ap.add_argument("--deep-sponge", type=float, default=None, metavar="TAU_DAYS",
                    help="diagnostic interior T/S sponge over H > 500 m toward one GLORYS May–June slope-water "
                         "profile at rate 1/TAU (case.deep_sponge; no domain budget)")
    ap.add_argument("--deep-sponge-3d", action="store_true",
                    help="with --deep-sponge: per-column target (GLORYS May–June mean at each column) instead of "
                         "one profile (case.deep_sponge(per_column=True))")
    ap.add_argument("--perturb-T", type=float, default=None, metavar="AMP",
                    help="diagnostic twin: add seeded uniform noise in [-AMP, AMP] °C to the initial θ of every wet "
                         "cell (case.perturb_temperature; intrinsic-variability test)")
    ap.add_argument("--perturb-seed", type=int, default=1)
    ap.add_argument("--compare", nargs=2, type=Path, default=None, metavar=("A", "B"),
                    help="compare two restart files bit for bit and exit")
    return ap


def main() -> None:
    a = parser().parse_args()
    days = a.days if a.days is not None else (30.0 if a.mode == "tides" else 123.0)
    a.dt = a.dt if a.dt is not None else DEFAULT_DT[a.mode]
    case = cs.CASE if a.tide_velocity_depth is None else cs.with_paths(cs.CASE,
                                                                          tide_velocity_depth=a.tide_velocity_depth)

    def log(msg: str) -> None:
        print(msg, flush=True)
    if a.compare is not None:
        res = compare_restarts(*a.compare)
    elif a.threads_check is not None:
        res = threads_check(case, a.mode, a.threads_check, a.dt, a.threads, log=log)
    elif a.restart_check:
        res = restart_check(case, a.mode, days, a.dt, a.threads, log=log)
    else:
        res = run(case, a.mode, days, a.dt, a.threads, a.restart_at, a.tag, a.out_hours, start_from=a.start_from,
                  log=log, overrides=parse_overrides(a.override), allow_restart_change=tuple(a.allow_restart_change),
                  deep_sponge=a.deep_sponge, deep_sponge_3d=a.deep_sponge_3d,
                  perturb_T=a.perturb_T, perturb_seed=a.perturb_seed)
    print(json.dumps(res, indent=1, default=float))


if __name__ == "__main__":
    main()
