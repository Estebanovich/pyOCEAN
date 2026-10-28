#!/usr/bin/env python
"""
BayIW 2-D section in pyOCEAN (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md §4): every
900 s η, the T and V sections on the MITgcm levels (thickness-weighted means) and T at 110/250 m and V at
110 m sampled from them exactly as the MITgcm output (compare.mit_at_depth).

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py 2layer bay --coord zstar [--dt 30] [--hours 60] [--layers 25]
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
from validation.bayiw_2d import compare, section  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402


def record(m, st, buf: dict) -> None:
    """
    Append η, the T and V sections on the MITgcm levels (conservative PPM_H4 remap, surface first,
    section.to_mitgcm_levels), T110, T250, V110 sampled from them exactly as MITgcm (compare.mit_at_depth),
    truncations and heat content.
    """
    H = np.asarray(m.H, float)[:, 0]
    Hc = np.nan_to_num(H)
    Tl = section.to_mitgcm_levels(st.h, st.T, m.H)[::-1, :, 0]                  # (50, ny), surface first
    hv = m.face_thickness(st.h)[1]                                          # v-face layers (N, ny+1, 1)
    Hf = np.concatenate([Hc[:1], np.minimum(Hc[1:], Hc[:-1]), Hc[-1:]])           # face bed: shallower side
    Vl = section.to_mitgcm_levels(hv, st.v, Hf[:, None])[::-1, :, 0]             # (50, ny+1)
    lev = dict(T=np.nan_to_num(Tl)[None], V=np.nan_to_num(Vl)[None], zc=bayiw.Z_C)
    buf["eta"].append(st.eta[:, 0].copy())
    buf["T110"].append(compare.mit_at_depth(lev, H, "T", bayiw.Z110)[0])
    buf["T250"].append(compare.mit_at_depth(lev, H, "T", bayiw.Z250)[0])
    buf["V110"].append(compare.mit_at_depth(lev, H, "V", bayiw.Z110)[0])
    buf["T_sec"].append(Tl)
    buf["V_sec"].append(Vl)
    buf["n_trunc"].append(st.n_trunc)
    buf["heat"].append(m.tracer_content(st))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=tuple(bayiw.CASES))
    ap.add_argument("bay", choices=("bay", "nobay"))
    ap.add_argument("--coord", default="zstar", choices=("zstar", "hybrid"))
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=60.0)
    ap.add_argument("--layers", type=int, default=25, help="hybrid layers")
    a = ap.parse_args()
    tag = a.coord if a.coord == "zstar" or a.layers == 25 else f"{a.coord}{a.layers}"
    out = section.RUNS / "pyocean" / f"{a.case}_{a.bay}_{tag}"
    out.mkdir(parents=True, exist_ok=True)
    set_threads(5)
    m, st, wind = section.build_model(a.case, a.bay, a.coord, a.dt, a.layers)
    buf = {k: [] for k in ("t_h", "eta", "T110", "T250", "V110", "T_sec", "V_sec", "n_trunc", "heat")}
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
