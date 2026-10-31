"""
Run the pyOCEAN ERA5 column (N2b task 7) and save hourly snapshots of the profiles and hourly means of the bulk
fluxes of one column, as the MITgcm diagnostics and the CROCO his/avg files.

    python -m validation.bulk_column.run_pyocean ly04 [--days 45]
    python -m validation.bulk_column.run_pyocean --all [--days 45]       # ly04, coare, coare_croco, fms, one at a time

Runs (``cases.RUNS_PYOCEAN``): 'ly04' (scheme 'ncar_ly04', vs MITgcm exf), 'coare' (vs CROCO), 'coare_croco'
('coare' with CROCO's coupling: absolute wind, SST of the top layer; spec criterion 5, step 4) and 'fms' (the
pyOCEAN default 'ncar', FMS form, for information); diagnostic runs (README §4): 'flux_mitgcm' and 'flux_croco'
(no bulk: the reference model's own hourly-mean fluxes prescribed, so only the ocean models differ),
'flux_croco_bg' and 'coare_bg' (with CROCO's hard-wired KPP interior background ν = 1e-4, κ = 1e-5 m² s⁻¹).

Output ``RUNS/pyocean_<days>d/<tag>.npz`` (never into the repo):
  time_snap [s] (0, 1 h, …), T, S, u, v (nt, N) bed first [°C, g/kg, m s⁻¹], hbl (nt,) KPP boundary-layer depth [m],
  eta (nt,) [m];
  time_mean [s] (end of each hour) and the hourly means of ``SurfaceFluxes.last``: q_sens, q_lat, q_lw, q_sw
  [W m⁻², + into the ocean], evap, precip, fw [m s⁻¹ in the model convention: mass flux / ρ₀], tau_x, tau_y
  [N m⁻²], sst_bulk [°C];
  spread (max over the run of the 5 × 5 spread of T, S, u, v: the column must stay 1-D), spread_hourly (nt, 4),
  wall time.

Set OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1 (CLAUDE.md).
"""
from __future__ import annotations

import argparse
import time
import warnings

import numpy as np

from validation.bulk_column import cases as C

STEPS_PER_HOUR = int(round(3600.0 / C.DT))
FLUXES = ("q_sens", "q_lat", "q_lw", "q_sw", "evap", "precip", "fw", "tau_x", "tau_y", "sst_bulk")


def out_path(tag: str, days: int):
    return C.RUNS / f"pyocean_{days}d" / f"{tag}.npz"


def _snap(model, st) -> dict:
    j = i = 0
    return dict(T=st.T[:, j, i].copy(), S=st.S[:, j, i].copy(),
                u=0.5 * (st.u[:, j, i] + st.u[:, j, i + 1]), v=0.5 * (st.v[:, j, i] + st.v[:, j + 1, i]),
                hbl=float(np.ravel(getattr(model._closure, "obl_depth", np.array([np.nan])))[0]),
                eta=float(st.eta[j, i]))


def _spread(st) -> np.ndarray:
    """Largest 5 × 5 spread of T, S, u, v over the layers (4,) [°C, g/kg, m s⁻¹]: the column must stay 1-D."""
    return np.array([float(np.ptp(a, axis=(1, 2)).max()) for a in (st.T, st.S, st.u, st.v)])


def _prescribed(ref: str, days: int):
    """Hourly-mean fluxes of a reference model (``compare.read_mitgcm`` / ``read_croco``), held constant over each
    hour, as ``step`` arguments: τ [N m⁻²] uniform on the faces, heat {Q_ns, Q_sw} [W m⁻²], fw = (P − E)/ρ₀ [m s⁻¹]
    (mass fluxes back to the model's volume convention)."""
    from validation.bulk_column import compare
    r = compare.read_mitgcm(days) if ref == "mitgcm" else compare.read_croco(days)
    shape = (C.NXY, C.NXY)

    def at(t: float) -> dict:
        h = min(int(t // 3600.0), r["t_mean"].size - 1)
        one = np.ones(shape)
        return dict(tau_x=np.full((C.NXY, C.NXY + 1), r["tau_x"][h]), tau_y=np.full((C.NXY + 1, C.NXY), r["tau_y"][h]),
                    heat={"Q_ns": (r["q_lw"][h] + r["q_sens"][h] + r["q_lat"][h]) * one, "Q_sw": r["q_sw"][h] * one},
                    fw=(r["P"][h] - r["E"][h]) / C.RHO0 * one,
                    last=dict(q_sens=r["q_sens"][h], q_lat=r["q_lat"][h], q_lw=r["q_lw"][h], q_sw=r["q_sw"][h],
                              evap=r["E"][h] / C.RHO0, precip=r["P"][h] / C.RHO0,
                              fw=(r["P"][h] - r["E"][h]) / C.RHO0, tau_x=r["tau_x"][h], tau_y=r["tau_y"][h],
                              sst_bulk=np.nan))
    return at


def run(tag: str, days: int = C.DAYS, save: bool = True, atm=None) -> dict:
    warnings.simplefilter("ignore")
    cfg = C.RUNS_PYOCEAN[tag]
    model, st = C.build_pyocean(cfg, days, atm=atm)
    forcing = _prescribed(cfg.prescribed, days) if cfg.prescribed else None
    n = C.n_steps(days)
    snaps = [dict(time=0.0, **_snap(model, st))]
    acc = {k: 0.0 for k in FLUXES}
    means: list[dict] = []
    spread = [_spread(st)]
    t0 = time.perf_counter()
    for step in range(1, n + 1):
        if forcing is None:
            st = model.step(st)
            last = {k: float(v[0, 0]) for k, v in model._surface_fluxes.last.items()}
        else:
            f = forcing(st.time)
            st = model.step(st, f["tau_x"], f["tau_y"], fw=f["fw"], heat=f["heat"])
            last = f["last"]
        for k in FLUXES:
            acc[k] += float(last[k])
        if step % STEPS_PER_HOUR == 0:
            if not (np.isfinite(st.T).all() and np.isfinite(st.u).all()):
                raise FloatingPointError(f"{tag}: non-finite state at step {step}")
            snaps.append(dict(time=st.time, **_snap(model, st)))
            means.append(dict(time=st.time, **{k: v / STEPS_PER_HOUR for k, v in acc.items()}))
            acc = {k: 0.0 for k in FLUXES}
            spread.append(_spread(st))
    wall = time.perf_counter() - t0
    out = {k: np.array([s[k] for s in snaps]) for k in snaps[0] if k != "time"}
    out["time_snap"] = np.array([s["time"] for s in snaps])
    out.update({k: np.array([m[k] for m in means]) for k in FLUXES})
    out["time_mean"] = np.array([m["time"] for m in means])
    out.update(spread=float(np.max(spread)), spread_hourly=np.array(spread), wall=wall, n_steps=n, s_per_step=wall / n,
               days=days)
    if save:
        p = out_path(tag, days)
        p.parent.mkdir(parents=True, exist_ok=True)
        np.savez(p, **out)
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("tag", nargs="?", choices=tuple(C.RUNS_PYOCEAN))
    ap.add_argument("--days", type=int, default=C.DAYS)
    ap.add_argument("--all", action="store_true", help="the four bulk runs (cases.BULK_RUNS)")
    a = ap.parse_args()
    tags = list(C.BULK_RUNS) if a.all else [a.tag]
    atm = C.atmosphere(a.days)
    for tag in tags:
        out = run(tag, a.days, atm=atm)
        print(f"{tag}: {out['n_steps']} steps, {out['wall']:.0f} s wall ({1e3 * out['s_per_step']:.1f} ms/step), "
              f"final SST {out['T'][-1, -1]:.3f} °C, 5x5 spread {out['spread']:.1e}", flush=True)


if __name__ == "__main__":
    main()
