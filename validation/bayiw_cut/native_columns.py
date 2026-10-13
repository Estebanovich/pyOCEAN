#!/usr/bin/env python
"""
BayIW cut, hybrid diagnosis (README "Diagnóstico de la híbrida"): run the cut with bay saving the native layer
columns (h, T) at the stations every 900 s, then compare the interface rise from the column heat,
    ζ = −(Σ h T − (Σ h T)₀) / (T_upper − T_lower),
with MITgcm's (compare.heat_displacement on its levels 20–40).  No level sampling is involved on the pyOCEAN side.

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_cut/native_columns.py run 2layer hybrid [--ke gudonov]
       python validation/bayiw_cut/native_columns.py compare 2layer hybrid [--ke gudonov]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.bayiw_baroclinic.compare import corr_lag, rms  # noqa: E402
from validation.bayiw_cut import compare, cut  # noqa: E402


def out_path(case: str, coord: str, ke: str) -> Path:
    return cut.RUNS / "pyocean" / f"native_{case}_bay_{coord}{'' if ke == 'arakawa' else '_' + ke}.npz"


def run(case: str, coord: str, ke: str, hours: float = 60.0) -> None:
    """60 h with the wind, as run_pyocean, recording the native columns at the stations."""
    from core.dynamics.sigma_baroclinic import set_threads
    from validation.bayiw_seiche.run_pyocean import wind_at
    set_threads(5)
    m, st, wind = cut.build_model(case, "bay", coord)
    m.ke_scheme = ke
    stn = cut.local_stations("py")
    buf = {"t": [0.0], "h": [], "T": []}

    def rec() -> None:
        buf["h"].append(np.array([st.h[:, j, i] for i, j in stn.values()]))
        buf["T"].append(np.array([st.T[:, j, i] for i, j in stn.values()]))
    rec()
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))
    for n in range(int(round(hours * 3600 / m.dt))):
        tau[:-1] = wind_at(n * m.dt, wind)
        st = m.step(st, tau_y=tau)
        if int((n + 1) * m.dt // 900.0) > int(n * m.dt // 900.0):
            buf["t"].append((n + 1) * m.dt)
            rec()
    np.savez(out_path(case, coord, ke), names=list(stn), **{k: np.array(v) for k, v in buf.items()})


def report(case: str, coord: str, ke: str) -> list[str]:
    """Native ζ against MITgcm's: rms ratio and zero-lag correlation per station."""
    d = np.load(out_path(case, coord, ke))
    names = list(d["names"])
    heat = (d["h"] * d["T"]).sum(-1)
    z = -(heat - heat[:1]) / (compare.T_UP - compare.T_LO)
    mit = compare.load_mit(case, "bay")
    tr = mit["t"]
    zm = compare.heat_displacement(mit["T"], compare.DZ, compare.T_UP, compare.T_LO)
    L = [f"# BayIW cut {case} bay, native {coord} ({ke}) interface rise vs MITgcm: rms ratio, corr"]
    for s, (i, j) in cut.local_stations("py").items():
        r, p = zm[:, j, i], np.interp(tr, d["t"], z[:, names.index(s)])
        if compare.resolvable("zeta", r):
            L.append(f"{s:8s} {rms(p) / rms(r):5.2f} {corr_lag(tr / 3600, r, p, 0.0)[0]:5.2f}   "
                     f"(MITgcm rms {1e3 * rms(r):.1f} mm)")
    print("\n".join(L))
    return L


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("what", choices=("run", "compare"))
    ap.add_argument("case", choices=("2layer", "linear"))
    ap.add_argument("coord", choices=("zstar", "hybrid"))
    ap.add_argument("--ke", default="arakawa", choices=("arakawa", "gudonov"))
    a = ap.parse_args()
    run(a.case, a.coord, a.ke) if a.what == "run" else report(a.case, a.coord, a.ke)
