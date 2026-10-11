#!/usr/bin/env python
"""
Compare the pyOCEAN baroclinic BayIW runs with the MITgcm reference (bay − nobay
differences, as in the thesis analysis of Output_IWBay).

Reference series (MITgcm, processed by the thesis scripts, every 900 s):
    npz_out/realistic_series.npz             η_bay, η_nobay at the stations (27–120 h)
    npz_out/realistic_W_W110m_series.npz     W_bay, W_nobay at k = 29 (W at RF = −106.6 m)
    npz_out/realistic_Temp_T110m_series.npz  T_bay, T_nobay at k = 29 (RC = −111.4 m)
Stations on land in nobay (MITgcm writes 0): the bay series alone, as in the thesis.
Also: amplitude of the barotropic seiche per 8 h block, 0–120 h (Eta_zoom_sfc_bay).
pyOCEAN: pyocean_iw_{bay,nobay}_realistic_stations.nc (run_bayiw.py, CF timeSeriesProfile;
legacy .npz runs are read too).

Metrics per station and variable (27–120 h, the thesis window):
    rms of both series, correlation, rms difference; spectral peak frequency and
    band variance in the near-inertial band (0.03–0.06 cph, f = 0.040 cph) and the
    seiche band (0.20–0.30 cph, seiche 0.2495 cph); ratio pyOCEAN/MITgcm.

Usage:  python compare_iw.py [--tag ""]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import iw_output

REF = Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/Output_IWBay/npz_out")
SSD_OUT = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_iw")            # large outputs live on the SSD
OUT = SSD_OUT if SSD_OUT.parent.exists() else Path(__file__).parent / "out"
BANDS = {"inertial": (0.03, 0.06), "seiche": (0.20, 0.30)}
T0_H = 27.0
Z_REF = -111.4      # MITgcm level used for "110 m": k = 29 (nearest RC); T[k] lives at RC[29]
Z_REF_W = -106.6    # … but W[k] lives at the top face RF[29] (MITgcm C-grid), so ΔW110 is at −106.6 m


def psd(x: np.ndarray, dt_h: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided periodogram with a Hann window [unit² cph⁻¹]."""
    x = np.asarray(x, float) - np.nanmean(x)
    x = np.nan_to_num(x)
    w = np.hanning(x.size)
    X = np.fft.rfft(x * w)
    f = np.fft.rfftfreq(x.size, dt_h)
    return f, 2 * dt_h * np.abs(X) ** 2 / np.sum(w**2)


def metrics(t_ref, ref, t_py, py, dt_h=0.25) -> dict:
    sel = t_ref >= T0_H - 1e-9
    t, r = t_ref[sel], ref[sel]
    p = np.interp(t, t_py, py)
    out = dict(rms_mit=float(np.std(r)), rms_py=float(np.std(p)),
               corr=float(np.corrcoef(r, p)[0, 1]) if np.std(p) > 0 and np.std(r) > 0 else float("nan"),
               rmse=float(np.sqrt(np.mean((p - r) ** 2))))
    fr, Pr = psd(r, dt_h)
    fp, Pp = psd(p, dt_h)
    df = fr[1] - fr[0]
    for name, (lo, hi) in BANDS.items():
        b = (fr >= lo) & (fr <= hi)
        out[f"{name}_var_mit"] = float(Pr[b].sum() * df)
        out[f"{name}_var_py"] = float(Pp[b].sum() * df)
        out[f"{name}_ratio"] = out[f"{name}_var_py"] / out[f"{name}_var_mit"] if out[f"{name}_var_mit"] > 0 else np.nan
        out[f"{name}_fpeak_mit"] = float(fr[b][np.argmax(Pr[b])])
        out[f"{name}_fpeak_py"] = float(fp[b][np.argmax(Pp[b])])
    return out


def py_series(d_bay, d_nob, name: str, k_station: int, H_st: float) -> tuple[np.ndarray, np.ndarray]:
    """
    bay − nobay series of pyOCEAN at a station; 'eta', or 'W110'/'T110' interpolated to Z_REF
    on the σ heights of each record, z = −H + (H + η) σ  (interfaces for w, layer centres for T).
    """
    t = d_bay["t"] / 3600.0

    def one(d):
        eta = d["eta_st"][:, k_station]
        if name == "eta":
            return eta
        sw = d["sigma_w"]
        s = sw if name == "W110" else 0.5 * (sw[1:] + sw[:-1])
        zf = -H_st + (H_st + eta)[:, None] * s[None, :]
        f = d["w_st" if name == "W110" else "T_st"][:, :, k_station]
        z_ref = Z_REF_W if name == "W110" else Z_REF
        return np.array([np.interp(z_ref, zf[n], f[n]) for n in range(len(t))])
    if d_nob is None:
        return t, one(d_bay)
    return t, one(d_bay) - one(d_nob)


def seiche_decay(d_bay, block_h: float = 8.0) -> list[str]:
    """
    Amplitude √2·std (linear trend removed) of η_bay at station Seiche in blocks of 8 h, 0–120 h:
    the barotropic seiche and its decay, MITgcm from <scenario>_Eta_zoom_sfc_bay.npz (zoom starts at i = 130).
    """
    b = np.load(REF / "realistic_Eta_zoom_sfc_bay.npz")
    i, j = d_bay["stations"]["Seiche"]
    tm, xm = b["t_horas"], b["data"][:, j, i - 130]
    tp, xp = d_bay["t"] / 3600.0, d_bay["eta_st"][:, list(d_bay["stations"]).index("Seiche")]

    def amp(t, x, t0):
        m = (t >= t0) & (t < t0 + block_h)
        y = x[m] - np.polyval(np.polyfit(t[m], x[m], 1), t[m])
        return np.sqrt(2.0) * y.std()
    out = ["", "η_bay at Seiche, amplitude per 8 h block [mm]:  window  MITgcm  pyOCEAN  ratio"]
    for t0 in np.arange(0.0, 120.0, block_h):
        a, c = amp(tm, xm, t0), amp(tp, xp, t0)
        out.append(f"  {t0:5.0f}–{t0 + block_h:<5.0f} {a * 1e3:8.3f} {c * 1e3:8.3f} {c / a:6.2f}")
    return out


def load(case: str, tag: str):
    """Records of a pyOCEAN run (NetCDF stem, else legacy .npz); None if absent."""
    stem = OUT / f"pyocean_iw_{case}_realistic{tag}"
    if stem.with_name(stem.name + "_stations.nc").exists():
        return iw_output.read_records(stem)
    npz = stem.with_suffix(".npz")
    return iw_output.read_records(npz) if npz.exists() else None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", default="")
    args = ap.parse_args()
    d_bay = load("bay", args.tag)
    d_nob = load("nobay", args.tag)
    if d_nob is not None and not np.array_equal(d_bay["t"], d_nob["t"]):
        raise SystemExit("bay and nobay records differ in time (incomplete run?)")
    gaps = np.flatnonzero(np.abs(np.diff(d_bay["t"]) - 900.0) > 1e-6)
    if gaps.size:
        raise SystemExit(f"gaps in the pyOCEAN series after t = {d_bay['t'][gaps] / 3600} h")
    stations = d_bay["stations"]
    bat = -np.fromfile("/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan/input/bahia_01_expand_bat.bin",
                       ">f8").reshape(352, 560)
    refs = {"eta": (np.load(REF / "realistic_series.npz"), "{s}_eta", "mm", 1e3),
            "W110": (np.load(REF / "realistic_W_W110m_series.npz"), "{s}_W110m", "mm/s", 1e3),
            "T110": (np.load(REF / "realistic_Temp_T110m_series.npz"), "{s}_T110m", "mK", 1e3)}
    lines = [f"pyOCEAN vs MITgcm BayIW realistic, bay − nobay, {T0_H:.0f}–120 h  (tag '{args.tag}')",
             "var    station   rms_MIT   rms_py   corr   rmse   | inertial var ratio, fpeak MIT/py | "
             "seiche var ratio, fpeak MIT/py"]
    results = {}
    for var, (ref, key, unit, sc) in refs.items():
        for k, (s, (i, j)) in enumerate(stations.items()):
            rk = key.format(s=s)
            if rk + "_bay" not in ref.files:
                continue
            H_st = min(bat[j, i], 599.9)
            # a station on land in nobay (MITgcm writes 0 there): the thesis "difference" is the bay series
            land = not np.any(ref[rk + "_nobay"])
            r_ref = ref[rk + "_bay"] if land else ref[rk + "_bay"] - ref[rk + "_nobay"]
            t_py, py = py_series(d_bay, None if land else d_nob, var, k, H_st)
            r = metrics(ref[f"{s}_t_horas"], r_ref, t_py, py)
            results[f"{var}/{s}"] = r
            lab = s + ("*" if land else "")
            lines.append(f"{var:5s}  {lab:8s} {r['rms_mit'] * sc:8.3f} {r['rms_py'] * sc:8.3f} {r['corr']:6.2f} "
                         f"{r['rmse'] * sc:7.3f} {unit:5s}| {r['inertial_ratio']:6.2f}  {r['inertial_fpeak_mit']:.3f}/"
                         f"{r['inertial_fpeak_py']:.3f} | {r['seiche_ratio']:6.2f}  {r['seiche_fpeak_mit']:.3f}/"
                         f"{r['seiche_fpeak_py']:.3f}")
    lines.append("* station on land in nobay: bay series alone (as in the thesis), anomalies for rms/corr")
    lines += seiche_decay(d_bay)
    txt = "\n".join(lines)
    print(txt)
    (OUT / f"compare_iw{args.tag}.txt").write_text(txt + "\n")
    (OUT / f"compare_iw{args.tag}.json").write_text(json.dumps(results, indent=1))


if __name__ == "__main__":
    main()
