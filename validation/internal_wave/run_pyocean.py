#!/usr/bin/env python
"""
MITgcm internal_wave in pyOCEAN (spec docs/superpowers/specs/2026-10-12-baroclinic-obc-internal-wave-design.md
§3): u, T and η on MITgcm's fixed levels at the first step past every 1800-s mark (MITgcm's dump times).

Usage: NUMBA_NUM_THREADS=5 python validation/internal_wave/run_pyocean.py zstar|hybrid|sigma [--periods 10] [--dt 500]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from validation.internal_wave import case  # noqa: E402


def record(m, st, buf: dict) -> None:
    """u at the u-faces (20, nx+1) and T (20, nx) on the MITgcm levels, surface first, and η (nx,)."""
    H = np.asarray(m.H, float)[0]
    Hc = np.nan_to_num(H)
    Hf = np.concatenate([Hc[:1], np.minimum(Hc[1:], Hc[:-1]), [0.0]])          # west face = first cell; east wall
    hu = m._mean_face_thickness(st.h)[0]                                         # (N, 1, nx+1)
    buf["U"].append(case.to_levels(hu, st.u, Hf[None])[:, 0, :])
    buf["T"].append(case.to_levels(st.h, st.T, H[None])[:, 0, :])
    buf["eta"].append(st.eta[0].copy())
    buf["n_trunc"].append(st.n_trunc)
    buf["heat"].append(m.tracer_content(st))
    buf["sumh"].append(float(np.nanmax(np.abs(st.h.sum(0)[0] - (Hc + st.eta[0]))[np.isfinite(H)])))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("coord", choices=("zstar", "hybrid", "sigma"))
    ap.add_argument("--periods", type=float, default=10.0)
    ap.add_argument("--dt", type=float, default=500.0)
    a = ap.parse_args()
    out = case.RUNS / "pyocean" / a.coord
    out.mkdir(parents=True, exist_ok=True)
    set_threads(5)
    m, st = case.build_model(a.coord, a.dt)
    buf = {k: [] for k in ("t", "U", "T", "eta", "n_trunc", "heat", "sumh")}
    buf["t"].append(0.0)
    record(m, st, buf)
    tic = time.time()
    for n in range(int(round(a.periods * case.PERIOD / a.dt))):
        st = m.step(st)
        if int((n + 1) * a.dt // 1800.0) > int(n * a.dt // 1800.0):           # first step past each 1800-s mark
            buf["t"].append((n + 1) * a.dt)
            record(m, st, buf)
    np.savez(out / "series.npz", **{k: np.array(v) for k, v in buf.items()})
    print(f"{a.coord}: {a.periods} periods in {(time.time() - tic) / 60:.1f} min, trunc {st.n_trunc}")


if __name__ == "__main__":
    main()
