#!/usr/bin/env python
"""
BayIW cut: pyOCEAN against MITgcm on the same cut (phase 1, ref 'mit') or both nested against Esteban's full
run (phase 2, ref 'full') — spec docs/superpowers/specs/2026-10-12-bayiw-cut-design.md §2.  Everything on the
pyOCEAN grid (MITgcm's interior); MITgcm's diagnostics are 900-s averages ending at their time stamp, so their
times are shifted by −450 s (pyOCEAN stores snapshots).

Usage: python validation/bayiw_cut/compare.py 2layer zstar [radiate|full|phases|dt]
       (full: phase 2 nested vs the full run; phases: pyOCEAN error phase 1 / 2; dt: the Δt = 30 s control)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_baroclinic.compare import corr_lag, dominant_period, relative_field_error, rms, window  # noqa: E402
from validation.bayiw_cut import cut  # noqa: E402

ZC = bayiw.Z_C[19:40]                    # level centres of MITgcm levels 20–40 (the T diagnostics)
K110, K250 = 10, 20                      # levels 30 and 40 inside them
T_UP, T_LO = 22.77, 18.52                 # two-layer upper / lower temperatures [°C]
T_MID = 0.5 * (T_UP + T_LO)               # central isotherm [°C]
DZ = np.diff(-bayiw.Z_F)[19:40]           # thicknesses of levels 20–40 [m]
IW = ("Seiche", "Seiche2", "Omega", "C")


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def to_py_grid(mit: dict) -> dict:
    """MITgcm cut fields without the open-boundary ring: rows 1:, columns/faces 1:-1 (north row = coast)."""
    out = dict(mit)
    for k in ("eta", "T", "U", "V"):
        if k in mit:
            out[k] = mit[k][..., 1:, 1:-1]
    return out


def isotherm_height(T: np.ndarray, zc: np.ndarray, t_mid: float) -> np.ndarray:
    """Height [m] of the t_mid isotherm (first crossing from the top) from T (n, nl, ny, nx) on levels zc (nl,)
    surface first, linear between level centres; NaN where it is not crossed."""
    s = T - t_mid
    cross = (s[:, :-1] * s[:, 1:] <= 0) & np.isfinite(s[:, :-1]) & np.isfinite(s[:, 1:]) & (s[:, :-1] != s[:, 1:])
    k = np.argmax(cross, axis=1)
    valid = cross.any(axis=1)
    take = lambda a, kk: np.take_along_axis(a, kk[:, None], axis=1)[:, 0]          # noqa: E731
    s0, s1 = take(s, k), take(s, k + 1)
    z0, z1 = zc[k], zc[k + 1]
    with np.errstate(invalid="ignore", divide="ignore"):
        z = z0 + (z1 - z0) * s0 / (s0 - s1)
    return np.where(valid, z, np.nan)


def heat_displacement(T: np.ndarray, dz: np.ndarray, t_upper: float, t_lower: float) -> np.ndarray:
    """
    Interface rise ζ [m] (positive up) of a two-layer fluid from the heat anomaly of the sampled levels:
        ζ(t) = −Σ_k (T_k(t) − T_k(0)) Δz_k / (T_upper − T_lower)
    T (n, nl, ny, nx) on levels of thickness dz (nl,), NaN below the bed.  Unlike the interpolated isotherm it
    does not see a symmetric smearing of the interface (diffusion conserves the column heat), only its
    displacement; horizontal heat advection inside a layer is not separated.
    """
    a = np.nan_to_num(T - T[:1])
    return -np.einsum("nkji,k->nji", a, dz) / (t_upper - t_lower)


NOBASE = ("V110", "eta")                 # series compared as they are (others: change from t = 0)
FLOOR = {"T": 1e-6, "V": 1e-6, "U": 1e-6, "eta": 1e-5, "iso": 1e-4, "zeta": 1e-4}   # resolvable rms [°C, m/s, m]
PHASE2_KEYS = ("eta", "V110")            # plus the case's T metrics (report_full)


def resolvable(key: str, x: np.ndarray) -> bool:
    """A reference series is worth a ratio only if its rms exceeds the floor of its kind (FLOOR, by key prefix)."""
    floor = FLOOR.get(key, FLOOR.get(key[0], 0.0))
    return bool(np.isfinite(x).all() and rms(x) > floor)


def field_at(t: np.ndarray, F: np.ndarray, t0: float) -> np.ndarray:
    """F (n, …) at time t0, linear between the two records that bracket it (t increasing)."""
    k = int(np.clip(np.searchsorted(t, t0), 1, t.size - 1))
    w = float(np.clip((t0 - t[k - 1]) / (t[k] - t[k - 1]), 0.0, 1.0))
    return (1.0 - w) * F[k - 1] + w * F[k]


def map_error(ta: np.ndarray, Fa: np.ndarray, T0a: np.ndarray, tb: np.ndarray, Fb: np.ndarray, T0b: np.ndarray,
              t0: float, mask: np.ndarray) -> float:
    """Relative rms error of model a's map anomaly against b's at t0, each on its own initial field, inside mask."""
    a = np.where(mask, field_at(ta, Fa, t0) - T0a, np.nan)
    b = np.where(mask, field_at(tb, Fb, t0) - T0b, np.nan)
    return relative_field_error(a, b)


def mit_t0(case: str, bay: str) -> np.ndarray:
    """MITgcm's true initial T (input theta) on the pyOCEAN grid at the sampled levels 20–40; its first diagnostic
    record is already a 900-s average."""
    T = cut.cut_inputs(case, bay, "py")["T"][19:40]
    return np.where(np.isfinite(cut.cut_inputs(case, bay, "py")["H"])[None] & (T != 0), T, np.nan)


def load_py(case: str, bay: str, coord: str, mode: str = "radiate", dt_tag: str = "") -> dict:
    p = dict(np.load(cut.RUNS / "pyocean" / f"{case}_{bay}_{coord}_{mode}{dt_tag}" / "series.npz"))
    p["T"] = np.where(np.isnan(p["T"]), np.nan, p["T"]).astype(float)
    return p


def load_mit(case: str, bay: str, mode: str = "radiate") -> dict:
    d = to_py_grid(cut.read_cut(case, bay, mode))
    d["t"] = d["t"] - 450.0                                                       # averages → mid-interval
    d["T"] = np.where(d["T"] == 0.0, np.nan, d["T"])                             # land / below the bed
    return d


def mit_seconds(case: str, bay: str, mode: str) -> float:
    txt = (cut.RUNS / "mitgcm" / f"{case}_{bay}_{mode}" / "output.txt").read_text()
    m = re.search(r"^real\s+([\d.]+)", txt, re.M)
    return float(m.group(1)) if m else float("nan")


def station_series(d: dict, case: str) -> dict:
    """η, T110, T250, cell-centred V110 (and U110 for A, B) and, in two layers, the isotherm height per station."""
    out = {}
    iso = isotherm_height(d["T"], ZC, T_MID) if case == "2layer" else None
    zeta = heat_displacement(d["T"], DZ, T_UP, T_LO) if case == "2layer" else None
    for s, (i, j) in cut.local_stations("py").items():
        U, V = np.nan_to_num(d["U"]), np.nan_to_num(d["V"])
        r = dict(eta=d["eta"][:, j, i], T110=d["T"][:, K110, j, i], T250=d["T"][:, K250, j, i],
                 V110=0.5 * (V[:, 0, j, i] + V[:, 0, j + 1, i]), U110=0.5 * (U[:, 0, j, i] + U[:, 0, j, i + 1]))
        if iso is not None:
            r["iso"] = iso[:, j, i]
            r["zeta"] = zeta[:, j, i]
        out[s] = r
    return out


def report(case: str, coord: str, mode: str = "radiate", ref: str = "mit", dt_tag: str = "") -> list[str]:
    """Spec §2 tables of pyOCEAN against the reference ('mit': MITgcm on the same cut)."""
    bays = ("bay", "nobay")
    py = {b: load_py(case, b, coord, mode, dt_tag) for b in bays}
    rf = {b: load_mit(case, b, mode) for b in bays}
    L = [f"# BayIW cut {case}, pyOCEAN {coord} ({mode}{dt_tag}) vs {'MITgcm on the same cut' if ref == 'mit' else ref}"]
    sp = {b: station_series(py[b], case) for b in bays}
    sr = {b: station_series(rf[b], case) for b in bays}
    tr, tp = rf["bay"]["t"], py["bay"]["t"]
    L.append("\n## Seiche Δη (0–27 h): period ref/py [h], range ref/py [mm], corr")
    for s in ("Seiche", "Seiche2"):
        dr = sr["bay"][s]["eta"] - sr["nobay"][s]["eta"]
        dp = np.interp(tr, tp, sp["bay"][s]["eta"] - sp["nobay"][s]["eta"])
        tw, xr = window(tr, dr, 0.0, 27.0 * 3600)
        _, xp = window(tr, dp, 0.0, 27.0 * 3600)
        th = tw / 3600.0
        Pr, Pp = dominant_period(th, xr, 1.0, 10.0), dominant_period(th, xp, 1.0, 10.0)
        Rr, Rp = np.ptp(xr), np.ptp(xp)
        c = corr_lag(th, xr, xp, 0.0)[0]
        L.append(f"{s:8s} {Pr:5.2f} {Pp:5.2f} | {1e3 * Rr:6.2f} {1e3 * Rp:6.2f} | {c:5.2f} | "
                 f"{_flag(abs(Pp / Pr - 1) <= 0.02 and abs(Rp / Rr - 1) <= 0.10 and c >= 0.95)}")
    keys = ("eta", "T110", "T250", "V110") + (("iso", "zeta") if case == "2layer" else ())   # zeta: complementary
    crit = "iso" if case == "2layer" else "T110"
    for key in keys:
        L.append(f"\n## {key} (0–60 h): rms py/ref, corr, lag [h], best corr" + ("   [criterion]" if key == crit else ""))
        for b in bays:
            for s in cut.STATIONS:
                xr = sr[b][s][key] - (sr[b][s][key][0] if key not in NOBASE else 0.0)
                xp = np.interp(tr, tp, sp[b][s][key] - (sp[b][s][key][0] if key not in NOBASE else 0.0))
                if not (resolvable(key, xr) and np.isfinite(xp).all()):
                    continue
                ratio = rms(xp) / rms(xr)
                c0, lag, cb = corr_lag(tr / 3600.0, xr, xp)
                ok = abs(ratio - 1) <= 0.20 and c0 >= 0.8
                L.append(f"{b:5s} {s:8s} {ratio:5.2f} | {c0:5.2f} | {lag:+5.2f} {cb:5.2f}"
                         + (f" | {_flag(ok)}" if key == crit and s in IW else ""))
    L.append("\n## U110 at A and B (0–60 h): rms py/ref, corr   [bay | nobay]")
    for s in ("A", "B"):
        row = []
        for b in bays:
            xr, xp = sr[b][s]["U110"], np.interp(tr, tp, sp[b][s]["U110"])
            row.append(f"{rms(xp) / rms(xr):4.2f} {corr_lag(tr / 3600, xr, xp, 0.0)[0]:+4.2f}" if resolvable("U110", xr) else "—")
        L.append(f"{s:8s} " + " | ".join(row))
    L.append("\n## Bay effect ΔT110 (bay − nobay), 0–60 h: rms py/ref, corr")
    for s in cut.STATIONS:
        def eff(series, t_src):
            a = series["bay"][s]["T110"] - series["bay"][s]["T110"][0]
            b = series["nobay"][s]["T110"] - series["nobay"][s]["T110"][0]
            return np.interp(tr, t_src, a - b) if t_src is not tr else a - b
        dr, dp = eff(sr, tr), eff(sp, tp)
        if not (resolvable("T110", dr) and np.isfinite(dp).all()):
            continue
        ratio, c = rms(dp) / rms(dr), corr_lag(tr / 3600, dr, dp, 0.0)[0]
        L.append(f"{s:8s} {ratio:6.2f} | {c:5.2f} | {_flag(1 / 3 <= ratio <= 3 and c > 0)}")
    L.append("\n## T110 map anomaly (bay, each model on its own initial T, linear in time): relative rms error")
    T0p, T0r = py["bay"]["T"][0, K110], mit_t0(case, "bay")[K110]
    for hr in (6.0, 24.0, 60.0):
        e = map_error(tp, py["bay"]["T"][:, K110], T0p, tr, rf["bay"]["T"][:, K110], T0r, hr * 3600, np.isfinite(T0p))
        L.append(f"{hr:4.0f} h  {e:5.2f}")
    for b in bays:
        p = py[b]
        L.append(f"\n## Health py {b}: trunc {int(p['n_trunc'][-1])}, max|Σh − H − η| {p['sumh'].max():.1e} m, "
                 f"heat change {p['heat'][-1] / p['heat'][0] - 1:+.2e}; cost py {float(p['s_per_step']):.2f} s/step, "
                 f"MITgcm {mit_seconds(case, b, mode) / 60:.1f} min/run (5 MPI)")
    print("\n".join(L))
    return L


# ---------------------------------------------------------------- phase 2: nested runs against the full run
T_END2 = 59.5 * 3600.0                   # MITgcm's periodic OB calendar interpolates toward record 179 (the
                                         # initial state, also used for t < 600 s) after 214 200 s


def interior_mask(shape: tuple[int, int], n: int = 10) -> np.ndarray:
    """Cells at least n away from the open west, east and south edges (the north row is the coast)."""
    m = np.zeros(shape, bool)
    m[n:, n:-n] = True
    return m


def phase2_ok(e_py: float, e_mit: float) -> bool:
    """Spec §2 phase 2: pyOCEAN's nested error is at most 1.5 times MITgcm's (the floor of nesting itself)."""
    return e_py <= 1.5 * e_mit


def load_full(case: str, bay: str) -> dict:
    d = dict(cut.full_reference(case, bay))
    d["T"] = np.where(d["T"] == 0.0, np.nan, d["T"]).astype(float)
    return d


def _rel(x: np.ndarray, ref: np.ndarray) -> float:
    """Relative rms error rms(x − ref) / rms(ref)."""
    return rms(x - ref) / rms(ref)


def report_full(case: str, coord: str = "zstar") -> list[str]:
    """
    Spec §2 phase 2: pyOCEAN nested and MITgcm nested against Esteban's full run (snapshots, 0–59.5 h), on the
    pyOCEAN grid.  Each metric is an error against the full run; PASS when pyOCEAN's ≤ 1.5 × MITgcm's.  MITgcm's
    cut diagnostics are 900-s averages (shifted −450 s), the full run's are snapshots: that residual is in its
    error (class c).
    """
    bays = ("bay", "nobay")
    fr = {b: load_full(case, b) for b in bays}
    py = {b: load_py(case, b, coord, "nest") for b in bays}
    mn = {b: load_mit(case, b, "nest") for b in bays}
    t = fr["bay"]["t"][fr["bay"]["t"] <= T_END2 + 1.0]
    nt = t.size
    sf = {b: station_series(fr[b], case) for b in bays}
    sp = {b: station_series(py[b], case) for b in bays}
    sm = {b: station_series(mn[b], case) for b in bays}
    on = lambda series, tsrc: np.interp(t, tsrc, series)                       # noqa: E731
    L = [f"# BayIW cut {case}, nested: pyOCEAN {coord} and MITgcm against the full run (error py | MITgcm)"]
    L.append("\n## Seiche Δη (0–27 h): |period error|, |range error|, 1 − corr")
    for s in ("Seiche", "Seiche2"):
        d = {k: on(v["bay"][s]["eta"] - v["nobay"][s]["eta"], ts) for k, v, ts in
             (("f", sf, fr["bay"]["t"]), ("p", sp, py["bay"]["t"]), ("m", sm, mn["bay"]["t"]))}
        tw, xf = window(t, d["f"], 0.0, 27.0 * 3600)
        th = tw / 3600.0
        Pf, Rf = dominant_period(th, xf, 1.0, 10.0), np.ptp(xf)
        e = {}
        for k in ("p", "m"):
            x = window(t, d[k], 0.0, 27.0 * 3600)[1]
            e[k] = (abs(dominant_period(th, x, 1.0, 10.0) / Pf - 1), abs(np.ptp(x) / Rf - 1),
                    1 - corr_lag(th, xf, x, 0.0)[0])
        ok = all(phase2_ok(a, max(b, 0.01)) for a, b in zip(e["p"], e["m"]))
        L.append(f"{s:8s} " + " | ".join(f"{a:.3f} {b:.3f}" for a, b in zip(e["p"], e["m"])) + f" | {_flag(ok)}")
    keys = PHASE2_KEYS + (("iso", "zeta") if case == "2layer" else ("T110",))
    for key in keys:
        L.append(f"\n## {key} (0–59.5 h): relative rms error py | MITgcm")
        for b in bays:
            for s in cut.STATIONS:
                base = lambda v: v[b][s][key] - (v[b][s][key][0] if key not in NOBASE else 0.0)   # noqa: E731
                xf = on(base(sf), fr[b]["t"])
                xp, xm = on(base(sp), py[b]["t"]), on(base(sm), mn[b]["t"])
                if not (resolvable(key, xf) and np.isfinite(xp).all() and np.isfinite(xm).all()):
                    continue
                ep, em = _rel(xp, xf), _rel(xm, xf)
                L.append(f"{b:5s} {s:8s} {ep:5.2f} | {em:5.2f} | {_flag(phase2_ok(ep, em))}")
    L.append("\n## Bay effect ΔT110 (bay − nobay): relative rms error py | MITgcm")
    for s in cut.STATIONS:
        def eff(v, tb, tn):
            a = v["bay"][s]["T110"] - v["bay"][s]["T110"][0]
            c = v["nobay"][s]["T110"] - v["nobay"][s]["T110"][0]
            return on(a, tb) - on(c, tn)
        df = eff(sf, fr["bay"]["t"], fr["nobay"]["t"])
        dp, dm = eff(sp, py["bay"]["t"], py["nobay"]["t"]), eff(sm, mn["bay"]["t"], mn["nobay"]["t"])
        if not (resolvable("T110", df) and np.isfinite(dp).all() and np.isfinite(dm).all()):
            continue
        ep, em = _rel(dp, df), _rel(dm, df)
        L.append(f"{s:8s} {ep:5.2f} | {em:5.2f} | {_flag(phase2_ok(ep, em))}")
    L.append("\n## T110 map anomaly (bay, 10 cells off the open edges, own initial T): relative rms error py | MITgcm")
    msk = interior_mask(fr["bay"]["eta"].shape[1:])
    tf, Ff, T0f = fr["bay"]["t"], fr["bay"]["T"][:, K110], fr["bay"]["T"][0, K110]
    for hr in (6.0, 24.0, 59.5):
        ep = map_error(py["bay"]["t"], py["bay"]["T"][:, K110], py["bay"]["T"][0, K110], tf, Ff, T0f, hr * 3600, msk)
        em = map_error(mn["bay"]["t"], mn["bay"]["T"][:, K110], mit_t0(case, "bay")[K110], tf, Ff, T0f, hr * 3600, msk)
        L.append(f"{hr:4.1f} h  {ep:5.2f} | {em:5.2f} | {_flag(phase2_ok(ep, em))}")
    for b in bays:
        p = py[b]
        L.append(f"\n## Health py {b}: trunc {int(p['n_trunc'][-1])}, max|Σh − H − η| {p['sumh'].max():.1e} m, "
                 f"cost {float(p['s_per_step']):.2f} s/step; samples {nt}")
    print("\n".join(L))
    return L


def phase_table(case: str, coord: str = "zstar") -> list[str]:
    """
    pyOCEAN's relative rms error (0–59.5 h) in phase 1 (radiating, against MITgcm radiating) and phase 2 (nested,
    against Esteban's full run) at the stations: equal errors mean that nesting adds none; the difference is what
    the boundaries bring.
    """
    keys = PHASE2_KEYS + (("iso", "zeta") if case == "2layer" else ("T110",))
    L = [f"# BayIW cut {case}, pyOCEAN {coord}: relative rms error phase 1 / phase 2"]
    for b in ("bay", "nobay"):
        m1, p1 = load_mit(case, b, "radiate"), load_py(case, b, coord, "radiate")
        f2, p2 = load_full(case, b), load_py(case, b, coord, "nest")
        s = {k: station_series(d, case) for k, d in (("m1", m1), ("p1", p1), ("f2", f2), ("p2", p2))}
        t = f2["t"][f2["t"] <= T_END2 + 1.0]
        for key in keys:
            row = []
            for st in cut.STATIONS:
                base = lambda d: d[st][key] - (d[st][key][0] if key not in NOBASE else 0.0)   # noqa: E731
                r1, x1 = np.interp(t, m1["t"], base(s["m1"])), np.interp(t, p1["t"], base(s["p1"]))
                r2, x2 = np.interp(t, f2["t"], base(s["f2"])), np.interp(t, p2["t"], base(s["p2"]))
                if resolvable(key, r1) and resolvable(key, r2) and np.isfinite(x1).all() and np.isfinite(x2).all():
                    row.append(f"{st} {_rel(x1, r1):.2f}/{_rel(x2, r2):.2f}")
            L.append(f"{b:5s} {key:5s} " + "  ".join(row))
    print("\n".join(L))
    return L


def dt_table(case: str = "2layer", coord: str = "zstar", dt_tag: str = "_dt30") -> list[str]:
    """The Δt control (bay only): rms ratio and correlation against MITgcm, control | Δt = 120 s."""
    mit = load_mit(case, "bay")
    tr = mit["t"]
    a, c = load_py(case, "bay", coord), load_py(case, "bay", coord, dt_tag=dt_tag)
    sm, sa, sc = station_series(mit, case), station_series(a, case), station_series(c, case)
    L = [f"# BayIW cut {case} bay {coord}: {dt_tag[1:]} | dt120 against MITgcm (rms ratio, corr)"]
    for st in cut.STATIONS:
        for key in ("eta", "V110") + (("iso", "zeta") if case == "2layer" else ("T110",)):
            r = sm[st][key] - (sm[st][key][0] if key not in NOBASE else 0.0)
            if not resolvable(key, r):
                continue
            out = []
            for x, tt in ((sc, c["t"]), (sa, a["t"])):
                q = np.interp(tr, tt, x[st][key] - (x[st][key][0] if key not in NOBASE else 0.0))
                out.append(f"{rms(q) / rms(r):5.2f} {corr_lag(tr / 3600, r, q, 0.0)[0]:5.2f}")
            L.append(f"{st:8s} {key:5s} " + " | ".join(out))
    print("\n".join(L))
    return L


if __name__ == "__main__":
    mode = sys.argv[3] if len(sys.argv) > 3 else "radiate"
    if mode == "phases":
        phase_table(sys.argv[1], sys.argv[2])
    elif mode == "dt":
        dt_table(sys.argv[1], sys.argv[2])
    elif mode == "full":
        report_full(sys.argv[1], sys.argv[2])
    else:
        report(sys.argv[1], sys.argv[2], mode)
