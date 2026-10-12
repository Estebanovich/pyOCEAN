#!/usr/bin/env python
"""
Compare pyOCEAN BayIW baroclinic runs with the MITgcm references (spec
docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md §6).

MITgcm: η and T at 110 m from the zoom fields (0–120 h, columns i = 130 … 429); T at 250 m and U, V at
110 m from the station series (27–120 h); the Gamma YZ section.  pyOCEAN: <RUNS>/bayiw_<case>/<coord>_<bay>/series.npz.

Usage: python validation/bayiw_baroclinic/compare.py 2layer zstar
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_seiche.compare import efold_hours  # noqa: E402

STATION_ORDER = tuple(bayiw.STATIONS)
IW_STATIONS = ("Seiche", "Seiche2", "Omega", "C")                       # spec §6: correlation ≥ 0.7 here
SNAP_H = (6.0, 24.0, 60.0, 90.0)


def window(t_h: np.ndarray, x: np.ndarray, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Samples with t0 ≤ t < t1 (time on the first axis)."""
    m = (t_h >= t0) & (t_h < t1)
    return t_h[m], x[m]


def rms(x: np.ndarray) -> float:
    """rms of the departure from the mean (NaNs ignored)."""
    return float(np.sqrt(np.nanmean((x - np.nanmean(x)) ** 2)))


def dominant_period(t_h: np.ndarray, x: np.ndarray, pmin: float = 1.0, pmax: float = 30.0) -> float:
    """Period [h] of the spectral peak between pmin and pmax: linear detrend, Hann window, 8× zero padding."""
    dt = float(np.median(np.diff(t_h)))
    y = x - np.polyval(np.polyfit(t_h, x, 1), t_h)
    y = y * np.hanning(y.size)
    nfft = 8 * y.size
    p = np.abs(np.fft.rfft(y, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, dt)
    band = (f >= 1.0 / pmax) & (f <= 1.0 / pmin)
    return float(1.0 / f[band][np.argmax(p[band])])


def corr_lag(t_h: np.ndarray, a: np.ndarray, b: np.ndarray, max_lag_h: float = 3.0) -> tuple[float, float, float]:
    """(correlation at zero lag, lag [h] of b behind a with the best correlation, that correlation)."""
    dt = float(np.median(np.diff(t_h)))
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    c0 = float(np.corrcoef(a, b)[0, 1])
    best = (c0, 0.0)
    for s in range(1, int(round(max_lag_h / dt)) + 1):
        for lag, (x, y) in ((s * dt, (a[:-s], b[s:])), (-s * dt, (a[s:], b[:-s]))):
            c = float(np.corrcoef(x, y)[0, 1])
            if c > best[0]:
                best = (c, lag)
    return c0, best[1], best[0]


def relative_field_error(a: np.ndarray, b: np.ndarray) -> float:
    """rms(a − b) / rms(b) over the points where both are finite (land is NaN in both)."""
    ok = np.isfinite(a) & np.isfinite(b)
    den = np.sqrt(np.mean(b[ok] ** 2))
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)) / den) if den > 0 else 0.0


def interface_thickness(T_sec: np.ndarray, z: np.ndarray, t_lo: float, t_hi: float) -> float:
    """
    Mean thickness [m] between the 10 % and 90 % isotherms of ΔT = t_hi − t_lo over the columns of the section
    T_sec (nz, ncol) on heights z (nz,), where both are found.
    """
    out = []
    for col in T_sec.T:
        ok = np.isfinite(col)
        if ok.sum() < 3:
            continue
        zz, tt = z[ok], col[ok]
        o = np.argsort(tt)
        lo, hi = t_lo + 0.1 * (t_hi - t_lo), t_lo + 0.9 * (t_hi - t_lo)
        if tt.min() > lo or tt.max() < hi:
            continue
        out.append(abs(np.interp(hi, tt[o], zz[o]) - np.interp(lo, tt[o], zz[o])))
    return float(np.mean(out)) if out else float("nan")


def mitgcm_stations(case: str, var: str) -> tuple[np.ndarray, dict]:
    """
    MITgcm series at the stations: var 'eta' / 'T110' from the zoom fields (0–120 h; mask_zoom is 1 on land → NaN),
    'T250' / 'U110' / 'V110' from the station series (27–120 h).  Returns (t_h, {station: (bay, nobay)}).
    """
    if var in ("eta", "T110"):
        stem = {"eta": "Eta_zoom_sfc", "T110": "Temp_zoom_z110m"}[var]
        out, t = {}, None
        for src in ("bay", "nobay"):
            d = np.load(bayiw.REF / f"{case}_{stem}_{src}.npz")
            t = d["t_horas"]
            for name, (i, j) in bayiw.STATIONS.items():
                x = d["data"][:, j, i - bayiw.ZOOM_I0].astype(float)
                out.setdefault(name, []).append(np.where(d["mask_zoom"][j, i - bayiw.ZOOM_I0] == 0, x, np.nan))
        return t, {k: tuple(v) for k, v in out.items()}
    f = {"T250": "Temp_T250m_series", "U110": "UV_UV110m_series", "V110": "UV_UV110m_series"}[var]
    key = {"T250": "T250m", "U110": "U110m", "V110": "V110m"}[var]
    d = np.load(bayiw.REF / f"{case}_{f}.npz")
    return d["Seiche_t_horas"], {k: (d[f"{k}_{key}_bay"], d[f"{k}_{key}_nobay"]) for k in bayiw.STATIONS}


def pyocean(case: str, coord: str, bay: str) -> dict:
    """The runner output of one pyOCEAN run as a dict of arrays."""
    d = np.load(bayiw.RUNS / f"bayiw_{case}" / f"{coord}_{bay}" / "series.npz")
    return {k: d[k] for k in d.files}


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report(case: str, coord: str) -> list[str]:
    """The spec §6 tables for one case and coordinate, as text lines (also printed)."""
    L = [f"# BayIW {case}, pyOCEAN {coord} vs MITgcm"]
    py = {b: pyocean(case, coord, b) for b in ("bay", "nobay")}
    tp = py["bay"]["t_h"]
    t_end = float(min(py["bay"]["t_h"][-1], py["nobay"]["t_h"][-1]))     # 60-h runs (Esteban, 2026-10-12)
    col = {s: k for k, s in enumerate(STATION_ORDER)}
    # 1. seiche Δη = η_bay − η_nobay, 0–27 h
    tm, em = mitgcm_stations(case, "eta")
    L.append("\n## Seiche Δη (0–27 h): period [h], range [mm], corr")
    for s in ("Seiche", "Seiche2"):
        dm = em[s][0] - em[s][1]
        dp = py["bay"]["eta"][:, col[s]] - py["nobay"]["eta"][:, col[s]]
        tw, xm = window(tm, dm, 0.0, 27.0)
        _, xp = window(tp, dp, 0.0, 27.0)
        xp_i = np.interp(tw, tp, dp)
        Pm, Pp = dominant_period(tw, xm, 1.0, 10.0), dominant_period(tw, xp_i, 1.0, 10.0)
        Rm, Rp = np.nanmax(xm) - np.nanmin(xm), np.nanmax(xp) - np.nanmin(xp)
        c = corr_lag(tw, xm, xp_i, 0.0)[0]
        L.append(f"{s:8s} MIT {Pm:5.2f} h {1e3 * Rm:6.2f} mm | py {Pp:5.2f} h {1e3 * Rp:6.2f} mm | corr {c:5.2f} | "
                 f"e-fold MIT {efold_hours(tw, xm):5.1f} py {efold_hours(tw, xp_i):5.1f} h | "
                 f"{_flag(abs(Pp / Pm - 1) <= 0.02 and abs(Rp / Rm - 1) <= 0.15 and c >= 0.9)}")
    # 2. internal waves: T anomaly at 110 m, bay and nobay runs, forced and free windows
    tm, Tm = mitgcm_stations(case, "T110")
    for wname, (t0, t1) in (("forced 0–60 h", (0.0, 60.0)), ("free 60–120 h", (60.0, 120.25))):
        if t_end < t1 - 1.0:                                            # the run does not cover this window
            continue
        L.append(f"\n## T anomaly at 110 m, {wname}: period MIT/py [h], rms py/MIT, corr, best lag [h]")
        for b, kb in (("bay", 0), ("nobay", 1)):
            for s in STATION_ORDER:
                xm_full = Tm[s][kb] - Tm[s][kb][0]
                xp_full = py[b]["T110"][:, col[s]] - py[b]["T110"][0, col[s]]
                tw, xm = window(tm, xm_full, t0, t1)
                xp = np.interp(tw, tp, xp_full)
                if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                    continue
                Pm, Pp = dominant_period(tw, xm), dominant_period(tw, xp)
                ratio = rms(xp) / rms(xm)
                c0, lag, c = corr_lag(tw, xm, xp)
                crit = abs(Pp / Pm - 1) <= 0.05 and abs(ratio - 1) <= 0.25
                if t0 == 0.0 and s in IW_STATIONS:
                    crit = crit and c0 >= 0.7
                L.append(f"{b:5s} {s:8s} {Pm:6.2f} {Pp:6.2f} | {ratio:5.2f} | {c0:5.2f} | {lag:+5.2f} ({c:4.2f}) | {_flag(crit)}")
    # bay effect ΔT110 = bay − nobay
    L.append("\n## Bay effect ΔT110 (bay − nobay), 0–60 h: rms py/MIT, corr")
    for s in STATION_ORDER:
        dm = (Tm[s][0] - Tm[s][0][0]) - (Tm[s][1] - Tm[s][1][0])
        dp = (py["bay"]["T110"][:, col[s]] - py["bay"]["T110"][0, col[s]]) - \
             (py["nobay"]["T110"][:, col[s]] - py["nobay"]["T110"][0, col[s]])
        tw, xm = window(tm, dm, 0.0, 60.0)
        xp = np.interp(tw, tp, dp)
        if not np.isfinite(xm).all() or rms(xm) == 0:
            continue
        ratio, c = rms(xp) / rms(xm), corr_lag(tw, xm, xp, 0.0)[0]
        L.append(f"{s:8s} {ratio:5.2f} | {c:5.2f} | {_flag(1 / 3 <= ratio <= 3 and c > 0)}")
    # 3. T250, U110, V110 from the series (27–120 h): rms ratio and corr
    for var in ("T250", "U110", "V110"):
        tm3, X = mitgcm_stations(case, var)
        keep = tm3 <= t_end + 1e-9                                      # never extrapolate pyOCEAN past its end
        tm3, X = tm3[keep], {k: (v[0][keep], v[1][keep]) for k, v in X.items()}
        L.append(f"\n## {var} (27–{t_end:.0f} h): rms py/MIT, corr   [bay | nobay]")
        for s in STATION_ORDER:
            row = []
            for b, kb in (("bay", 0), ("nobay", 1)):
                xm = X[s][kb] - (X[s][kb][0] if var == "T250" else 0.0)
                xp_full = py[b][var][:, col[s]]
                xp = np.interp(tm3, tp, xp_full - (np.interp(tm3[0], tp, xp_full) if var == "T250" else 0.0))
                if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                    row.append("   —   ")
                    continue
                row.append(f"{rms(xp) / rms(xm):4.2f} {corr_lag(tm3, xm, xp, 0.0)[0]:+4.2f}")
            L.append(f"{s:8s} " + " | ".join(row))
    # 4. spatial: T110 zoom anomaly, relative error at the snapshot times (bay run)
    L.append("\n## T110 zoom anomaly (bay): relative rms error at "
             + ", ".join(f"{h:g} h" for h in SNAP_H if h <= t_end))
    zm = np.load(bayiw.REF / f"{case}_Temp_zoom_z110m_bay.npz")
    zmask = zm["mask_zoom"] == 0                                       # mask_zoom: 1 on land, 0 in the ocean
    errs = []
    for h in [x for x in SNAP_H if x <= t_end]:
        km, kp = int(np.argmin(np.abs(zm["t_horas"] - h))), int(np.argmin(np.abs(tp - h)))
        am = np.where(zmask, zm["data"][km] - zm["data"][0], np.nan)
        ap = py["bay"]["T110_zoom"][kp] - py["bay"]["T110_zoom"][0]
        errs.append(relative_field_error(ap, am))
    L.append("   " + "  ".join(f"{e:5.2f}" for e in errs))
    # 5. health and the interface (two-layer)
    for b in ("bay", "nobay"):
        p = py[b]
        L.append(f"\n## Health {b}: trunc {int(p['n_trunc'][-1])}, max|u| {np.nanmax(p['umax']):.3f} m/s, "
                 f"heat change {p['heat'][-1] / p['heat'][0] - 1:+.2e}, RPE gain {p['rpe'][-1] - p['rpe'][0]:.3e} J")
    if case == "2layer":
        g = np.load(bayiw.REF / f"{case}_secThetaAnom_yz_Gamma.npz", allow_pickle=True)
        L.append("\n## Interface thickness (10–90 % of ΔT) along the Gamma section [m]: MIT | py")
        for h in [x for x in SNAP_H if x <= t_end]:
            km, kp = int(np.argmin(np.abs(g["t_horas"] - h))), int(np.argmin(np.abs(tp - h)))
            Tm_sec = np.where(g["mask2d"], g["anom"][km].astype(float), np.nan)
            Tm_sec = np.where(Tm_sec == 0, np.nan, Tm_sec)
            Tp_sec = py["bay"]["gamma"][kp].astype(float)
            L.append(f"{h:5.0f} h  {interface_thickness(Tm_sec, -np.abs(g['RC']), 18.52, 22.77):6.1f} | "
                     f"{interface_thickness(Tp_sec, bayiw.Z_C, 18.52, 22.77):6.1f}")
    print("\n".join(L))
    return L


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2])
