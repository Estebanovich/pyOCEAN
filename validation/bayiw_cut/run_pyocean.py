#!/usr/bin/env python
"""
BayIW cut in pyOCEAN (spec docs/superpowers/specs/2026-10-12-bayiw-cut-design.md §1.2): every 900 s η, T on the
fixed MITgcm levels 20–40 and U/V on levels 30 and 40 (1-based; conservative PPM_H4 remap with η in the top level,
validation/bayiw_2d/section.to_mitgcm_levels), as MITgcm's diagnostics of the cut.

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_cut/run_pyocean.py 2layer bay --coord zstar [--dt 120]
       [--mode radiate|nest] [--hours 60]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from validation.bayiw_2d.section import to_mitgcm_levels  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_cut import cut  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402

T_LEV = slice(19, 40)                     # MITgcm levels 20–40 (1-based), surface first
UV_LEV = [29, 39]                         # levels 30 and 40


def record(m, st, buf: dict) -> None:
    H = np.asarray(m.H, float)
    Hu, Hv = m._edge_face_depths()                                   # face beds; open edges = boundary cell
    hu, hv = m._mean_face_thickness(st.h)
    buf["eta"].append(st.eta.astype(np.float32))
    buf["T"].append(to_mitgcm_levels(st.h, st.T, H)[::-1][T_LEV].astype(np.float32))
    buf["U"].append(to_mitgcm_levels(hu, st.u, Hu)[::-1][UV_LEV].astype(np.float32))
    buf["V"].append(to_mitgcm_levels(hv, st.v, Hv)[::-1][UV_LEV].astype(np.float32))
    buf["n_trunc"].append(st.n_trunc)
    buf["heat"].append(m.tracer_content(st))
    wet = np.isfinite(H)
    buf["sumh"].append(float(np.abs(st.h.sum(0) - (np.nan_to_num(H) + st.eta))[wet].max()))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=tuple(bayiw.CASES))
    ap.add_argument("bay", choices=("bay", "nobay"))
    ap.add_argument("--coord", default="zstar", choices=("zstar", "hybrid"))
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--mode", default="radiate", choices=("radiate", "nest"))
    ap.add_argument("--hours", type=float, default=60.0)
    a = ap.parse_args()
    tag = f"{a.case}_{a.bay}_{a.coord}_{a.mode}" + ("" if a.dt == 120.0 else f"_dt{a.dt:g}")
    out = cut.RUNS / "pyocean" / tag
    out.mkdir(parents=True, exist_ok=True)
    set_threads(5)
    state = cut.nest_state(a.case, a.bay) if a.mode == "nest" else None
    m, st, wind = cut.build_model(a.case, a.bay, a.coord, a.dt, boundary_state=state)
    buf = {k: [] for k in ("t", "eta", "T", "U", "V", "n_trunc", "heat", "sumh")}
    buf["t"].append(0.0)
    record(m, st, buf)
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))
    tic = time.time()
    nsteps = int(round(a.hours * 3600 / a.dt))
    for n in range(nsteps):
        tau[:-1] = wind_at(n * a.dt, wind)                              # v-face j = south of cell j
        st = m.step(st, tau_y=tau)
        if int((n + 1) * a.dt // 900.0) > int(n * a.dt // 900.0):      # first step past each 900-s mark
            buf["t"].append((n + 1) * a.dt)
            record(m, st, buf)
    el = time.time() - tic
    np.savez(out / "series.npz", **{k: np.array(v) for k, v in buf.items()}, s_per_step=el / nsteps)
    print(f"{tag}: {a.hours} h in {el / 60:.1f} min ({el / nsteps:.2f} s/step), trunc {st.n_trunc}")


if __name__ == "__main__":
    main()
