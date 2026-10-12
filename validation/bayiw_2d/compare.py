#!/usr/bin/env python
"""
Compare the BayIW 2-D section in pyOCEAN with MITgcm on the same section (spec
docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md §5).  MITgcm T and V are sampled at 110/250 m
from its level centres (bed first, piecewise linear between centres, as pyOCEAN's sampling).  The last block
compares MITgcm with walls against MITgcm periodic in x (rotating 2.5-D slice): what pyOCEAN's missing
periodic boundary changes.

Usage: python validation/bayiw_2d/compare.py 2layer zstar [nobal]   (MITgcm reference variant: '' = Esteban's
balanceFacS = −1, 'nobal' = Orlanski + sponge without the zero-net-transport balance)
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.bayiw_2d import section  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_baroclinic.compare import (corr_lag, dominant_period, interface_thickness,  # noqa: E402
                                                 relative_field_error, rms, window)

IW = ("Seiche", "Seiche2", "Omega", "C")                # stations with the internal-wave criteria


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def mit_at_depth(d: dict, H: np.ndarray, var: str, z0: float) -> np.ndarray:
    """
    MITgcm T (or V at the v-faces) at height z0 [m] for every dump: (n, 352) or (n, 353); NaN below the bed.
    MITgcm levels are surface first: flipped to bed first for bayiw.sample_at_depth.
    """
    X = d[var][:, ::-1]
    Hf = np.concatenate([H[:1], np.fmin(H[1:], H[:-1]), H[-1:]]) if var == "V" else H
    Hf = np.nan_to_num(Hf, nan=0.0)
    zc = d["zc"][::-1, None] * np.ones((1, X.shape[2]))
    h = np.where(zc > -Hf[None], 1.0, 0.0)                                   # levels above the bed (z-levels)
    return np.array([bayiw.sample_at_depth(zc, X[n], h, z0, -Hf) for n in range(X.shape[0])])


def _seiche(L: list, tm: np.ndarray, a: dict, b: dict, ta: np.ndarray | None = None, label: str = "py") -> None:
    """Δη = η_bay − η_nobay at Seiche and Seiche2 (0–27 h): MITgcm a vs b (b on times ta)."""
    ta = tm if ta is None else ta
    L.append(f"\n## Seiche Δη (0–27 h): period MIT/{label} [h], range MIT/{label} [mm], corr")
    for s in ("Seiche", "Seiche2"):
        j = section.SECTION_STATIONS[s]
        dm = a["bay"]["eta"][:, j] - a["nobay"]["eta"][:, j]
        dp = np.interp(tm, ta, b["bay"]["eta"][:, j] - b["nobay"]["eta"][:, j])
        tw, xm = window(tm, dm, 0.0, 27.0)
        _, xp = window(tm, dp, 0.0, 27.0)
        Pm, Pp = dominant_period(tw, xm, 1.0, 10.0), dominant_period(tw, xp, 1.0, 10.0)
        Rm, Rp = np.ptp(xm), np.ptp(xp)
        c = corr_lag(tw, xm, xp, 0.0)[0]
        L.append(f"{s:8s} {Pm:5.2f} {Pp:5.2f} | {1e3 * Rm:6.2f} {1e3 * Rp:6.2f} | {c:5.2f} | "
                 f"{_flag(abs(Pp / Pm - 1) <= 0.02 and abs(Rp / Rm - 1) <= 0.10 and c >= 0.95)}")


def report(case: str, coord: str, ref: str = "") -> list[str]:
    """Print and return the comparison tables of one case and pyOCEAN coordinate against MITgcm variant ref."""
    L = [f"# BayIW 2-D section {case}, pyOCEAN {coord} vs MITgcm {ref or 'balanced'} (same section, walls in x)"]
    bays = ("bay", "nobay")
    H = {b: section.section_inputs(case, b)["H"][:, 0] for b in bays}
    mit = {b: section.read_mitgcm(case, b, ref) for b in bays}
    py = {b: dict(np.load(section.RUNS / "pyocean" / f"{case}_{b}_{coord}" / "series.npz")) for b in bays}
    tm, tp = mit["bay"]["t_h"], py["bay"]["t_h"]
    st = section.SECTION_STATIONS
    _seiche(L, tm, mit, py, tp)
    # internal waves: T anomaly at 110 and 250 m, 0–60 h
    for var, z0 in (("T110", bayiw.Z110), ("T250", bayiw.Z250)):
        L.append(f"\n## {var} anomaly (0–60 h): period MIT/py [h], rms py/MIT, corr, lag [h], best corr")
        for b in bays:
            Xm = mit_at_depth(mit[b], H[b], "T", z0)
            for s, j in st.items():
                xm = Xm[:, j] - Xm[0, j]
                xp = np.interp(tm, tp, py[b][var][:, j] - py[b][var][0, j])
                if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                    L.append(f"{b:5s} {s:8s}   (below the bed or no signal)")
                    continue
                Pm, Pp = dominant_period(tm, xm), dominant_period(tm, xp)
                ratio = rms(xp) / rms(xm)
                c0, lag, c = corr_lag(tm, xm, xp)
                ok = abs(Pp / Pm - 1) <= 0.05 and abs(ratio - 1) <= 0.20 and (c0 >= 0.8 if s in IW else True)
                L.append(f"{b:5s} {s:8s} {Pm:6.2f} {Pp:6.2f} | {ratio:5.2f} | {c0:5.2f} | {lag:+5.2f} {c:5.2f} | "
                         f"{_flag(ok)}")
    # bay effect ΔT110
    L.append("\n## Bay effect ΔT110 (bay − nobay), 0–60 h: rms py/MIT, corr")
    Xb = mit_at_depth(mit["bay"], H["bay"], "T", bayiw.Z110)
    Xn = mit_at_depth(mit["nobay"], H["nobay"], "T", bayiw.Z110)
    for s, j in st.items():
        dm = (Xb[:, j] - Xb[0, j]) - (Xn[:, j] - Xn[0, j])
        dp = np.interp(tm, tp, (py["bay"]["T110"][:, j] - py["bay"]["T110"][0, j])
                       - (py["nobay"]["T110"][:, j] - py["nobay"]["T110"][0, j]))
        if not (np.isfinite(dm).all() and np.isfinite(dp).all()) or rms(dm) == 0:
            continue
        ratio, c = rms(dp) / rms(dm), corr_lag(tm, dm, dp, 0.0)[0]
        L.append(f"{s:8s} {ratio:5.2f} | {c:5.2f} | {_flag(1 / 3 <= ratio <= 3 and c > 0)}")
    # V at 110 m
    L.append("\n## V110 (0–60 h): rms py/MIT, corr   [bay | nobay]")
    for s, j in st.items():
        row = []
        for b in bays:
            xm = mit_at_depth(mit[b], H[b], "V", bayiw.Z110)[:, j]
            xp = np.interp(tm, tp, py[b]["V110"][:, j])
            if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                row.append("   —     ")
                continue
            row.append(f"{rms(xp) / rms(xm):4.2f} {corr_lag(tm, xm, xp, 0.0)[0]:+4.2f}")
        L.append(f"{s:8s} " + " | ".join(row))
    # T section error and interface
    L.append("\n## T section anomaly (bay): relative rms error at 6, 24, 60 h"
             + ("; interface thickness MIT/py [m]" if case == "2layer" else ""))
    below = bayiw.Z_C[:, None] < -np.nan_to_num(H["bay"], nan=0.0)[None]
    for hr in (6.0, 24.0, 60.0):
        km, kp = int(np.argmin(np.abs(tm - hr))), int(np.argmin(np.abs(tp - hr)))
        Tm_abs = np.where(below, np.nan, mit["bay"]["T"][km])
        Tm = Tm_abs - np.where(below, np.nan, mit["bay"]["T"][0])
        Tp = np.where(below, np.nan, py["bay"]["T_sec"][kp] - py["bay"]["T_sec"][0])
        line = f"{hr:4.0f} h  {relative_field_error(Tp, Tm):5.2f}"
        if case == "2layer":
            line += (f" | {interface_thickness(Tm_abs, bayiw.Z_C, 18.52, 22.77):6.1f} "
                     f"{interface_thickness(np.where(below, np.nan, py['bay']['T_sec'][kp]), bayiw.Z_C, 18.52, 22.77):6.1f}")
        L.append(line)
    for b in bays:
        p = py[b]
        L.append(f"\n## Health py {b}: trunc {int(p['n_trunc'][-1])}, heat change {p['heat'][-1] / p['heat'][0] - 1:+.2e}, "
                 f"RPE gain {p['rpe'][-1] - p['rpe'][0]:.3e} J")
    # rotation: MITgcm periodic (rotating 2.5-D) vs MITgcm walls
    try:
        per = {b: section.read_mitgcm(case, b, "periodic") for b in bays}
    except FileNotFoundError:
        per = None
    if per is not None:
        L.append("\n# Rotation effect: MITgcm periodic in x (u free, f active) vs MITgcm walls")
        _seiche(L, tm, mit, per, per["bay"]["t_h"], "per")
        L.append("\n## T110 anomaly (bay): period walls/periodic [h], rms per/walls, corr")
        Xw, Xp = mit_at_depth(mit["bay"], H["bay"], "T", bayiw.Z110), mit_at_depth(per["bay"], H["bay"], "T", bayiw.Z110)
        for s, j in st.items():
            xw, xr = Xw[:, j] - Xw[0, j], Xp[:, j] - Xp[0, j]
            if not np.isfinite(xw).all() or rms(xw) == 0:
                continue
            L.append(f"{s:8s} {dominant_period(tm, xw):6.2f} {dominant_period(tm, xr):6.2f} | "
                     f"{rms(xr) / rms(xw):5.2f} | {corr_lag(tm, xw, xr, 0.0)[0]:5.2f}")
    print("\n".join(L))
    return L


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2], sys.argv[3] if len(sys.argv) > 3 else "")
