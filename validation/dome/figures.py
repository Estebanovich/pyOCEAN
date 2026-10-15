#!/usr/bin/env python
"""
Figures of the DOME benchmark (spec 2026-10-14-dome-overflow §8), written to cases.RUNS/figs/ (never to the repo):
the bottom dye of every 50-day run at days 10, 20, 30 and 50, and the time-mean path Y'(x') and plume transport T(x')
(days 30–50) against Legg et al. (2006) at 10 km.

Usage: python validation/dome/figures.py 10
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.dome import cases, metrics, series  # noqa: E402

DAYS = (10, 20, 30, 50)


def bottom_dye(s: dict, n: int) -> np.ndarray:
    """s in the deepest layer thicker than 1 m of each column at record n (NaN on land)."""
    h = np.nan_to_num(s["h"][n])
    k = np.where(h > 1.0, np.arange(h.shape[0])[:, None, None], -1).max(axis=0)
    out = np.take_along_axis(np.nan_to_num(s["s"][n]), np.maximum(k, 0)[None], axis=0)[0]
    return np.where((k >= 0) & np.isfinite(s["H"]), out, np.nan)


def main(res_km: float) -> None:
    runs = [p for p in sorted((cases.RUNS / f"{res_km:g}km").glob("*/series.npz"))]
    data = []
    for p in runs:
        s, _ = series.load(p)
        if s["t"][-1] >= 50 * 86400.0:
            data.append((p.parent.name, s))
    out = cases.RUNS / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(data), len(DAYS), figsize=(3.2 * len(DAYS), 1.7 * len(data)), squeeze=False,
                           sharex=True, sharey=True)
    for r, (name, s) in enumerate(data):
        for c, d in enumerate(DAYS):
            n = int(np.argmin(np.abs(s["t"] - d * 86400.0)))
            im = ax[r, c].pcolormesh(s["x"] / 1e3, s["y"] / 1e3, bottom_dye(s, n), vmin=0, vmax=1, cmap="viridis")
            ax[r, c].contour(s["x"] / 1e3, s["y"] / 1e3, np.nan_to_num(s["H"]), [1000, 2000, 3000], colors="w",
                             linewidths=0.4)
            ax[r, c].set_title(f"{name}, día {d}", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6, label="s de fondo")
    fig.savefig(out / f"dome_{res_km:g}km_bottom_dye.png", dpi=110)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    for name, s in data:
        x, Y = metrics.path_Y(s, 30 * 86400.0, 50 * 86400.0)
        _, T = metrics.plume_transport(s, 30 * 86400.0, 50 * 86400.0)
        a1.plot((x - 1050e3) / 1e3, (Y - 600e3) / 1e3, label=name)
        a2.plot((x - 1050e3) / 1e3, T / 1e6, label=name)
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


if __name__ == "__main__":
    main(float(sys.argv[1]))
