#!/usr/bin/env python
"""
BayIW baroclinic run with the ALE mode of pyOCEAN, matching Esteban's MITgcm runs (spec
docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md §3–4).  Every 900 s: η, T at
110/250 m and U, V at 110 m at the 7 stations; T at 110 m in the MITgcm zoom window; the YZ section of
T along i = 280 from Gamma on the MITgcm level centres; health diagnostics.  A plain state every 6 h
(manual restart; the bit-for-bit ALE restart is Plan D).

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py 2layer bay [--coord zstar]
           [--dt 120] [--hours 120] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402

STATION_ORDER = tuple(bayiw.STATIONS)


def face_sample(m, st, z0: float, axis: str) -> np.ndarray:
    """U (axis 'u', west face of cell i) or V ('v', south face of cell j) at height z0 at the stations."""
    zc = m.layer_heights(st.eta, st.h)
    H = np.nan_to_num(np.asarray(m.H, float), nan=-np.inf)
    out = []
    for i, j in bayiw.STATIONS.values():
        (j0, i0), (j1, i1) = ((j, i - 1), (j, i)) if axis == "u" else ((j - 1, i), (j, i))
        vel = st.u[:, j, i] if axis == "u" else st.v[:, j, i]
        zf = 0.5 * (zc[:, j0, i0] + zc[:, j1, i1])
        hf = 0.5 * (st.h[:, j0, i0] + st.h[:, j1, i1])
        bed = -min(H[j0, i0], H[j1, i1])
        out.append(bayiw.sample_at_depth(zf, vel, hf, z0, np.asarray(bed)))
    return np.array(out, float)


def record(m, st, buf: dict) -> None:
    """Append the 900-s outputs of state st to buf."""
    zc = m.layer_heights(st.eta, st.h)
    bed = -np.asarray(m.H, float)                                       # NaN on land → NaN samples
    ij = [(j, i) for i, j in bayiw.STATIONS.values()]
    jj, ii = np.array([p[0] for p in ij]), np.array([p[1] for p in ij])
    buf["eta"].append(st.eta[jj, ii])
    for name, z0 in (("T110", bayiw.Z110), ("T250", bayiw.Z250)):
        buf[name].append(bayiw.sample_at_depth(zc[:, jj, ii], st.T[:, jj, ii], st.h[:, jj, ii], z0, bed[jj, ii]))
    buf["U110"].append(face_sample(m, st, bayiw.Z110, "u"))
    buf["V110"].append(face_sample(m, st, bayiw.Z110, "v"))
    sl = np.s_[:, :, bayiw.ZOOM_I0:bayiw.ZOOM_I0 + bayiw.ZOOM_NI]
    buf["T110_zoom"].append(bayiw.sample_at_depth(zc[sl], st.T[sl], st.h[sl], bayiw.Z110,
                                                  bed[:, bayiw.ZOOM_I0:bayiw.ZOOM_I0 + bayiw.ZOOM_NI]).astype(np.float32))
    js = np.s_[:, bayiw.GAMMA_J0:, bayiw.GAMMA_I]
    buf["gamma"].append(np.array([bayiw.sample_at_depth(zc[js], st.T[js], st.h[js], z0, bed[bayiw.GAMMA_J0:, bayiw.GAMMA_I])
                                  for z0 in bayiw.Z_C], np.float32))
    buf["n_trunc"].append(st.n_trunc)
    buf["umax"].append(float(max(np.nanmax(np.abs(st.u)), np.nanmax(np.abs(st.v)))))
    buf["heat"].append(m.tracer_content(st))


def save(out: Path, buf: dict, meta: dict) -> None:
    np.savez(out / "series.npz", **{k: np.array(v) for k, v in buf.items()}, meta_json=json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=tuple(bayiw.CASES))
    ap.add_argument("bay", choices=("bay", "nobay"))
    ap.add_argument("--coord", default="zstar", choices=("zstar", "hybrid"))
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--hours", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or bayiw.RUNS / f"bayiw_{a.case}" / f"{a.coord}_{a.bay}"
    out.mkdir(parents=True, exist_ok=True)
    set_threads(5)
    m, st, wind = bayiw.build_model(a.case, a.bay, a.coord, a.dt)
    meta = dict(vars(a), out=str(out), stations=bayiw.STATIONS, order=STATION_ORDER, z110=bayiw.Z110,
                z250=bayiw.Z250, zoom_i0=bayiw.ZOOM_I0, gamma=(bayiw.GAMMA_I, bayiw.GAMMA_J0))
    buf = {k: [] for k in ("t_h", "eta", "T110", "T250", "U110", "V110", "T110_zoom", "gamma", "n_trunc", "umax",
                           "heat", "rpe_t_h", "rpe")}
    buf["t_h"].append(0.0)
    record(m, st, buf)
    buf["rpe_t_h"].append(0.0)
    buf["rpe"].append(m.rpe(st))
    snap = int(round(6 * 3600.0 / a.dt))
    nsteps = int(round(a.hours * 3600.0 / a.dt))
    tau = np.zeros((bayiw.NY + 1, bayiw.NX))
    tic = time.time()
    with open(out / "log.txt", "a") as log:
        for n in range(nsteps):
            tau[:-1, :] = wind_at(n * a.dt, wind)                          # v-face j = south of cell j
            st = m.step(st, tau_y=tau)
            if int((n + 1) * a.dt // 900.0) > int(n * a.dt // 900.0):      # first step past each 900-s mark
                buf["t_h"].append((n + 1) * a.dt / 3600.0)                  # (dt need not divide 900 s)
                record(m, st, buf)
            if (n + 1) % snap == 0 or n + 1 == nsteps:
                t_h = (n + 1) * a.dt / 3600.0
                buf["rpe_t_h"].append(t_h)
                buf["rpe"].append(m.rpe(st))
                np.savez(out / f"state_{t_h:05.1f}h.npz", eta=st.eta, h=st.h, T=st.T, u=st.u, v=st.v, time=st.time)
                save(out, buf, meta)
                msg = (f"{a.case} {a.bay} {a.coord}: t = {t_h:6.1f} h  umax {buf['umax'][-1]:.3f} m/s  "
                       f"trunc {st.n_trunc}  ({(time.time() - tic) / 60:.1f} min)")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
    save(out, buf, meta)
    print("saved", out / "series.npz")


if __name__ == "__main__":
    main()
