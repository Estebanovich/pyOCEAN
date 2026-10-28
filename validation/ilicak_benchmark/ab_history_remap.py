"""AB3 history remap after the ALE regrid (as MOM6 remap_dyn_split_RK2_aux_vars): effect on the Ilicak cases.
Usage: python validation/ilicak_benchmark/ab_history_remap.py <case> <coord> <nu> [hours]   → prints the metrics without and with the remap."""
import json, sys, time, warnings
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
warnings.simplefilter("ignore")
import numpy as np
from core.model.driver import Driver
from core.vertical.ale import face_fractions
from core.dynamics.sigma_baroclinic import set_threads
from validation.ilicak_benchmark import cases, metrics
from validation.ilicak_benchmark.run_pyocean import build, record

ORIG = Driver._ale_finalize


def remapping_finalize(self, ctx):
    ORIG(self, ctx)
    vc = self.m._vertical
    hu_lag, hv_lag = vc.mean_face_thickness(ctx.lf.h_dagger)
    hu1, hv1 = vc.mean_face_thickness(ctx.h_new)
    wu1, wv1 = face_fractions(hu1), face_fractions(hv1)
    ctx.new_state.g_hist = [(vc.remap_faces(gu, hu_lag, hu1, wu1), vc.remap_faces(gv, hv_lag, hv1, wv1))
                            for gu, gv in ctx.new_state.g_hist]


def run(case_name, coord, nu, hours, remap):
    Driver._ale_finalize = remapping_finalize if remap else ORIG
    c = cases.get(case_name)
    m, st = build(c, coord, nu, c.dt)
    n_steps = int(round(hours * 3600.0 / c.dt))
    every = max(1, int(round(min(c.out_s, hours * 3600.0) / c.dt)))
    recs, times = [record(m, st)], [0.0]
    tic = time.time()
    err = None
    try:
        for n in range(1, n_steps + 1):
            st = m.step(st)
            if n % every == 0 or n == n_steps:
                recs.append(record(m, st)); times.append(n * c.dt)
    except Exception as e:                                   # a stopped run is a result too
        err = f"{type(e).__name__} at {n * c.dt / 3600:.1f} h: {str(e)[:90]}"
    s = {k: np.array([r[k] for r in recs]) for k in ("h", "T", "u", "v", "eta")}
    s.update(t=np.array(times), x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H)
    out = dict(remap=remap, wall=round(time.time() - tic, 1), n_trunc=int(st.n_trunc), stopped=err,
               t_end_h=times[-1] / 3600)
    rpe = metrics.rpe(s)
    out["dRPE_rel"] = float((rpe[-1] - rpe[0]) / rpe[0])
    lo, hi = metrics.t_range(s)
    out["T_range"] = (round(lo, 6), round(hi, 6))
    out["max_u"] = float(np.nanmax(metrics.max_speed(s)))
    if case_name == "iw":
        out["amp_ratio"] = metrics.oscillation_ratio(s["t"], metrics.iw_amplitude(s))
    if case_name == "lock" and err is None:
        out["fronts"] = {k: (round(v, 4) if isinstance(v, float) else v) for k, v in metrics.lock_fronts(s).items()}
    return out


if __name__ == "__main__":
    case_name, coord, nu = sys.argv[1], sys.argv[2], float(sys.argv[3])
    hours = float(sys.argv[4]) if len(sys.argv) > 4 else cases.get(case_name).hours
    set_threads(5)
    for remap in (False, True):
        print(json.dumps(dict(case=case_name, coord=coord, nu=nu, **run(case_name, coord, nu, hours, remap))),
              flush=True)
