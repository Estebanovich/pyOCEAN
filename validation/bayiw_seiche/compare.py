#!/usr/bin/env python
"""
Compare the pyOCEAN barotropic BayIW run with the MITgcm reference series.

MITgcm series: Output_IWBay/npz_out/<scenario>_series.npz (η every 900 s at
the analysis stations, 27–120 h, bay / nobay / diff).  The seiche signal is
Δη = η_bay − η_nobay; its dominant frequency reported in the thesis
documentation is 0.2495 cph (4.0 h), against 0.3034 cph (3.3 h) for the
closed quarter-wave estimate c/(4L), c = √(g·164 m), L = 119 km.

Metrics: spectral peak of Δη (Hann-windowed periodogram, 8× zero padding,
parabolic peak refinement), amplitude envelope, and correlation of the
Δη series after removing their means.

Usage:  python compare.py      (writes out/compare.png and prints a table)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

HERE = Path(__file__).parent
MIT_NPZ = Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/Output_IWBay/npz_out")
SCENARIOS = ("realistic", "linear", "2layer")
STATION = "Seiche"
T_MIN, T_MAX = 27.0, 120.0
F_THEORY = np.sqrt(9.81 * 164.0) / (4 * 119e3) * 3600      # cph
STATION_ZOOM_IJ = (345, 150)    # Seiche (i=280, j=345) in the zoom arrays (zoom starts at i=130)


def peak_frequency(t_h: np.ndarray, x: np.ndarray, band=(0.1, 0.6)) -> tuple[float, np.ndarray, np.ndarray]:
    """Dominant frequency [cph] of x(t) within ``band``; returns (f_peak, f, psd)."""
    dt = np.median(np.diff(t_h))
    x = (x - x.mean()) * np.hanning(x.size)
    nfft = 8 * x.size
    psd = np.abs(np.fft.rfft(x, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, dt)
    m = (f >= band[0]) & (f <= band[1])
    k = np.nonzero(m)[0][np.argmax(psd[m])]
    a, b, c = np.log(psd[k - 1:k + 2])
    shift = 0.5 * (a - c) / (a - 2 * b + c)                # parabolic refinement
    return f[k] + shift * (f[1] - f[0]), f, psd


def efold_hours(t_h: np.ndarray, x: np.ndarray, t_end: float = 24.0, block: float = 4.0) -> float:
    """e-folding time [h] of the seiche envelope: log-linear fit of 4-h ranges over 0–t_end."""
    starts = np.arange(0.0, t_end, block)
    env = np.array([np.ptp(x[(t_h >= s) & (t_h < s + block)]) for s in starts])
    return -1.0 / np.polyfit(starts + block / 2, np.log(env), 1)[0]


def load_pyocean() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    bay = np.load(HERE / "out" / "pyocean_bay.npz")
    nobay = np.load(HERE / "out" / "pyocean_nobay.npz")
    t = bay["t_horas"]
    return t, bay[STATION], bay[STATION] - nobay[STATION]


def main() -> None:
    t_py, eta_py, deta_py = load_pyocean()
    w = (t_py >= T_MIN - 1e-9) & (t_py <= T_MAX + 1e-9)
    f_py, fp, pp = peak_frequency(t_py[w], deta_py[w])

    rows = [("pyOCEAN (barotropic)", f_py, np.ptp(deta_py[w]), np.nan)]
    curves = {"pyOCEAN": (t_py, deta_py, fp, pp)}
    for sc in SCENARIOS:
        path = MIT_NPZ / f"{sc}_series.npz"
        if not path.exists():
            continue
        d = np.load(path)
        t, x = d[f"{STATION}_t_horas"], d[f"{STATION}_eta_diff"]
        m = (t >= T_MIN - 1e-9) & (t <= T_MAX + 1e-9)
        f_m, fm, pm = peak_frequency(t[m], x[m])
        common = np.intersect1d(np.round(t[m], 4), np.round(t_py[w], 4))
        a = x[m][np.isin(np.round(t[m], 4), common)]
        b = deta_py[w][np.isin(np.round(t_py[w], 4), common)]
        corr = np.corrcoef(a - a.mean(), b - b.mean())[0, 1]
        rows.append((f"MITgcm {sc}", f_m, np.ptp(x[m]), corr))
        curves[f"MITgcm {sc}"] = (t, x, fm, pm)

    print(f"Δη at station {STATION}, window {T_MIN:.0f}–{T_MAX:.0f} h; "
          f"closed quarter-wave theory {F_THEORY:.4f} cph ({1 / F_THEORY:.2f} h)")
    print(f"{'run':24s} {'f_peak [cph]':>12s} {'period [h]':>10s} {'range Δη [m]':>13s} {'corr w/ pyOCEAN':>16s}")
    for name, f, rng, c in rows:
        print(f"{name:24s} {f:12.4f} {1 / f:10.2f} {rng:13.4f} {c:16.3f}")

    # ---- Energetic window 0–27 h from the MITgcm Eta fields (every 15 min) ----
    print(f"\nΔη at {STATION}, energetic window 0–{T_MIN:.0f} h (MITgcm from <scenario>_Eta_zoom_sfc_*.npz)")
    print(f"{'run':24s} {'period [h]':>10s} {'range [mm]':>10s} {'e-fold [h]':>10s} {'corr':>6s} {'RMSE [mm]':>9s}")
    early = t_py <= T_MIN + 1e-9
    f0, f0_axis, p0 = peak_frequency(t_py[early], deta_py[early])
    curves["pyOCEAN"] = (t_py, deta_py, f0_axis, p0)          # spectra on the same 0–27 h window
    print(f"{'pyOCEAN (barotropic)':24s} {1 / f0:10.2f} {np.ptp(deta_py[early]) * 1e3:10.2f} "
          f"{efold_hours(t_py, deta_py):10.1f} {'':>6s} {'':>9s}")
    for sc in SCENARIOS:
        pb, pn = MIT_NPZ / f"{sc}_Eta_zoom_sfc_bay.npz", MIT_NPZ / f"{sc}_Eta_zoom_sfc_nobay.npz"
        if not (pb.exists() and pn.exists()):
            continue
        b, n = np.load(pb), np.load(pn)
        j, i = STATION_ZOOM_IJ
        t, x = b["t_horas"], b["data"][:, j, i] - n["data"][:, j, i]
        m = t <= T_MIN + 1e-9
        f, _, _ = peak_frequency(t[m], x[m])
        common = np.intersect1d(np.round(t[m], 3), np.round(t_py[early], 3))
        a = x[m][np.isin(np.round(t[m], 3), common)]
        c = deta_py[early][np.isin(np.round(t_py[early], 3), common)]
        print(f"{'MITgcm ' + sc:24s} {1 / f:10.2f} {np.ptp(x[m]) * 1e3:10.2f} {efold_hours(t, x):10.1f} "
              f"{np.corrcoef(a, c)[0, 1]:6.3f} {np.sqrt(np.mean((a - c) ** 2)) * 1e3:9.2f}")
        curves[f"MITgcm {sc}"] = (t, x, *peak_frequency(t[m], x[m])[1:])

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    for name, (t, x, f, p) in curves.items():
        ax1.plot(t, x, lw=1.2 if name == "pyOCEAN" else 0.8, label=name)
        ax2.semilogy(f, p / p.max(), label=name)
    ax1.set(xlim=(0, 48), xlabel="t [h]", ylabel="Δη = η_bay − η_nobay [m]",
            title=f"BayIW seiche at station {STATION}: pyOCEAN (barotropic) vs MITgcm")
    ax1.axvspan(T_MIN, T_MAX, color="0.9", zorder=0)
    ax2.axvline(F_THEORY, color="k", ls="--", lw=0.8, label="c/(4L) closed theory")
    ax2.set(xlim=(0, 1.0), ylim=(1e-4, 2), xlabel="frequency [cph]", ylabel="normalised PSD (0–27 h)")
    for ax in (ax1, ax2):
        ax.legend(fontsize=8)
        ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(HERE / "out" / "compare.png", dpi=130)
    print("figure:", HERE / "out" / "compare.png")


if __name__ == "__main__":
    main()
