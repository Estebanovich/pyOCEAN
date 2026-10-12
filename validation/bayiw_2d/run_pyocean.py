#!/usr/bin/env python
"""
BayIW 2-D section in pyOCEAN (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md §4): every
900 s η, T at 110/250 m and V at 110 m along the section, and the T section on the MITgcm level centres.

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py 2layer bay --coord zstar [--dt 30] [--hours 60]
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
from validation.bayiw_2d import section  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402


def record(m, st, buf: dict) -> None:
    """Append η, T110, T250, V110 (v-faces), the T section on Z_C, truncations and heat content."""
    zc = m.layer_heights(st.eta, st.h)[:, :, 0]
    h, T, bed = st.h[:, :, 0], st.T[:, :, 0], -np.asarray(m.H, float)[:, 0]
    buf["eta"].append(st.eta[:, 0].copy())
    buf["T110"].append(bayiw.sample_at_depth(zc, T, h, bayiw.Z110, bed))
    buf["T250"].append(bayiw.sample_at_depth(zc, T, h, bayiw.Z250, bed))
    zf = np.concatenate([zc[:, :1], 0.5 * (zc[:, 1:] + zc[:, :-1]), zc[:, -1:]], axis=1)      # v-faces
    hf = np.concatenate([h[:, :1], 0.5 * (h[:, 1:] + h[:, :-1]), h[:, -1:]], axis=1)
    bedf = np.concatenate([bed[:1], np.maximum(bed[1:], bed[:-1]), bed[-1:]])
    buf["V110"].append(bayiw.sample_at_depth(zf, st.v[:, :, 0], hf, bayiw.Z110, bedf))
    buf["T_sec"].append(np.array([bayiw.sample_at_depth(zc, T, h, z0, bed) for z0 in bayiw.Z_C]))
    buf["n_trunc"].append(st.n_trunc)
    buf["heat"].append(m.tracer_content(st))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=tuple(bayiw.CASES))
    ap.add_argument("bay", choices=("bay", "nobay"))
    ap.add_argument("--coord", default="zstar", choices=("zstar", "hybrid"))
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=60.0)
    a = ap.parse_args()
    out = section.RUNS / "pyocean" / f"{a.case}_{a.bay}_{a.coord}"
    out.mkdir(parents=True, exist_ok=True)
    set_threads(5)
    m, st, wind = section.build_model(a.case, a.bay, a.coord, a.dt)
    buf = {k: [] for k in ("t_h", "eta", "T110", "T250", "V110", "T_sec", "n_trunc", "heat")}
    buf["t_h"].append(0.0)
    record(m, st, buf)
    rpe = [m.rpe(st)]
    tau = np.zeros((st.eta.shape[0] + 1, 1))
    tic = time.time()
    for n in range(int(round(a.hours * 3600 / a.dt))):
        tau[:-1, :] = wind_at(n * a.dt, wind)                                   # v-face j = south of cell j
        st = m.step(st, tau_y=tau)
        if int((n + 1) * a.dt // 900.0) > int(n * a.dt // 900.0):              # first step past each 900-s mark
            buf["t_h"].append((n + 1) * a.dt / 3600.0)
            record(m, st, buf)
    rpe.append(m.rpe(st))
    np.savez(out / "series.npz", **{k: np.array(v) for k, v in buf.items()}, rpe=np.array(rpe),
             meta_json=json.dumps(vars(a)))
    print(f"{a.case} {a.bay} {a.coord}: {a.hours} h in {(time.time() - tic) / 60:.1f} min, trunc {st.n_trunc}")


if __name__ == "__main__":
    main()
