#!/usr/bin/env python
"""
Figures of the DOME benchmark (spec 2026-10-14-dome-overflow §8), written to cases.RUNS/figs/ (never to the repo):
the bottom dye of every 50-day run at days 10, 20, 30 and 50, and the time-mean path Y'(x') and plume transport T(x')
(days 30–50) against Legg et al. (2006) at 10 km; dome_<res>km_bottom_dye.mp4/.gif, the bottom dye of every 50-day
run day by day.

Usage: python validation/dome/figures.py 10 [--no-anim] [--runs name1,name2,... --sub dir]
  --runs keeps only those run directories (e.g. the Smagorinsky set) and --sub writes to RUNS/figs/<dir>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "manual" / "figures"))

from validation.dome import cases, metrics, series  # noqa: E402

DAYS = (10, 20, 30, 50)
LABELS = {"mitgcm_z_legg": "MITgcm z (Legg physics)", "mitgcm_z_common": "MITgcm z (common physics)",
          "mom6_rho_common": "MOM6 ρ", "pyocean_zstar": "pyOCEAN z*", "pyocean_sigma": "pyOCEAN σ",
          "pyocean_hybrid": "pyOCEAN hybrid", "pyocean_hybrid_gls": "pyOCEAN hybrid + GLS",
          "pyocean_hybrid_gls_smag0.15": "pyOCEAN hybrid + GLS + Smag 0.15",
          "pyocean_hybrid_gls_smag0.5": "pyOCEAN hybrid + GLS + Smag 0.5",
          "pyocean_hybrid_gls_smag1e-09": "pyOCEAN hybrid + GLS, operator control C≈0",
          "pyocean_hybrid_kv_low_smag0.15": "pyOCEAN hybrid ν_v 1e-4 + Smag 0.15"}


def bottom_dye(s: dict, n: int) -> np.ndarray:
    """s in the deepest layer thicker than 1 m of each column at record n (NaN on land)."""
    h = np.nan_to_num(s["h"][n])
    k = np.where(h > 1.0, np.arange(h.shape[0])[:, None, None], -1).max(axis=0)
    out = np.take_along_axis(np.nan_to_num(s["s"][n]), np.maximum(k, 0)[None], axis=0)[0]
    return np.where((k >= 0) & np.isfinite(s["H"]), out, np.nan)


def anim_bottom_dye(res_km: float, data: list, out: Path | None = None) -> None:
    """Bottom dye of every 50-day run, one frame per daily record, 2 rows × 3 panels."""
    import pyocean_style as ps

    ps.use()
    ncol = 3
    nrow = int(np.ceil(len(data) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(11.0, 2.1 * nrow + 0.5), sharex=True, sharey=True,
                            squeeze=False, constrained_layout=True)
    meshes = []
    for ax, (name, s) in zip(axs.flat, data):
        ax.set_facecolor(ps.LAND)
        ax.grid(False)
        meshes.append(ax.pcolormesh(s["x"] / 1e3, s["y"] / 1e3, np.ma.masked_invalid(bottom_dye(s, 0)), vmin=0,
                                    vmax=1, cmap=ps.CMAPS["fraction"].reversed(), rasterized=True))
        ax.contour(s["x"] / 1e3, s["y"] / 1e3, np.nan_to_num(s["H"]), [1000, 2000, 3000], colors=ps.MUTED,
                   linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_title(LABELS.get(name, name), fontsize=9)
    for ax in axs.flat[len(data):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("x [km]")
    for ax in axs[:, 0]:
        ax.set_ylabel("y [km]")
    ps.colorbar(fig, meshes[0], axs, "bottom dye s", shrink=0.7)
    t = data[0][1]["t"]
    head = f"DOME {res_km:g} km (Legg et al. 2006, case 1): bottom dye, isobaths 1000/2000/3000 m"
    title = fig.suptitle(f"{head} — day {t[0] / 86400:4.1f}", x=0.02, ha="left", fontsize=11)
    ps.freeze_layout(fig)

    def update(n: int):
        title.set_text(f"{head} — day {t[n] / 86400:4.1f}")
        for m, (_, s) in zip(meshes, data):
            m.set_array(np.ma.masked_invalid(bottom_dye(s, min(n, s["t"].size - 1))))

    ps.save_animation(fig, update, t.size, (out or cases.RUNS / "figs") / f"dome_{res_km:g}km_bottom_dye", fps=6)


def main(res_km: float, anim: bool = True, only: list[str] | None = None, sub: str = "") -> None:
    runs = [p for p in sorted((cases.RUNS / f"{res_km:g}km").glob("*/series.npz"))
            if not only or p.parent.name in only]
    data = []
    for p in runs:
        s, _ = series.load(p)
        if s["t"][-1] >= 50 * 86400.0:
            data.append((p.parent.name, s))
    out = cases.RUNS / "figs" / sub if sub else cases.RUNS / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(data), len(DAYS), figsize=(3.2 * len(DAYS), 1.7 * len(data)), squeeze=False,
                           sharex=True, sharey=True)
    for r, (name, s) in enumerate(data):
        for c, d in enumerate(DAYS):
            n = int(np.argmin(np.abs(s["t"] - d * 86400.0)))
            im = ax[r, c].pcolormesh(s["x"] / 1e3, s["y"] / 1e3, bottom_dye(s, n), vmin=0, vmax=1, cmap="viridis")
            ax[r, c].contour(s["x"] / 1e3, s["y"] / 1e3, np.nan_to_num(s["H"]), [1000, 2000, 3000], colors="w",
                             linewidths=0.4)
            ax[r, c].set_title(f"{LABELS.get(name, name)}, día {d}", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6, label="s de fondo")
    fig.savefig(out / f"dome_{res_km:g}km_bottom_dye.png", dpi=110)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in data:
        x, Y = metrics.path_Y(s, 30 * 86400.0, 50 * 86400.0)
        _, T = metrics.plume_transport(s, 30 * 86400.0, 50 * 86400.0)
        a1.plot((x - 1050e3) / 1e3, (Y - 600e3) / 1e3, label=LABELS.get(name, name))
        a2.plot((x - 1050e3) / 1e3, T / 1e6, label=LABELS.get(name, name))
    if res_km == 10:
        leg = cases.LEGG_10KM
        for key, mk in (("y_z_km", "s"), ("y_iso_km", "o"), ("y_iso_visc_km", "^")):
            a1.plot(leg["x_km"], leg[key], "k" + mk, ls="none", label=f"Legg {key[2:-3]}")
    a1.set(xlabel="x' [km]", ylabel="Y' [km]", xlim=(-1000, 50), title="Trayectoria media, días 30–50")
    a2.set(xlabel="x' [km]", ylabel="T [Sv]", xlim=(-1000, 50), title="Transporte de la pluma (s > 0.01)")
    a1.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(out / f"dome_{res_km:g}km_path_transport.png", dpi=110)
    plt.close(fig)
    print(out)
    if anim:
        order = list(LABELS)
        anim_bottom_dye(res_km, sorted(data, key=lambda d: order.index(d[0]) if d[0] in order else len(order)), out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--runs", default="", help="comma-separated run directories to keep (default: every 50-day run)")
    ap.add_argument("--sub", default="", help="subdirectory of RUNS/figs for the output")
    a = ap.parse_args()
    main(a.res_km, not a.no_anim, [r for r in a.runs.split(",") if r] or None, a.sub)
