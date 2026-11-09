"""
Figures of the manual page theory/wetting-drying.md.

1. wd_schematic.png   A sloping beach on the C grid: cell beds H_i (negative = land above the datum), the free surface,
                      and the face rule of WetDryFreeSurface._face_depth evaluated on this example (no time stepping):
                      H_f = min(H_L, H_R) (the higher bed), η upwind (here u = 0, so the higher η of the two cells),
                      D_f = H_f + η_up, wet if D_f > d_min.
2. Copies (not moves) of the tracked validation figures:
      validation/thacker/out/thacker.png                 → thacker.png
      validation/tidal_flat/out/tidal_flat.png           → tidal_flat_2dh.png
      validation/tidal_flat/out/tidal_flat_vs3d_z0.png   → tidal_flat_vs3d_z0.png
      validation/tidal_flat/out/tidal_flat_3d_gls.png    → tidal_flat_3d_gls.png

Run:  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/manual/figures/make_wetting_drying.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

os.environ.setdefault("NUMBA_NUM_THREADS", "1")
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import Rectangle  # noqa: E402

from core.dynamics.wetting_drying import WetDryFreeSurface  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402

OUT = ROOT / "docs/manual/img/wetting-drying"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "grey": "#666666", "black": "#000000", "sand": "#d9c49a"}


def fig_schematic() -> None:
    nx, d_min = 9, 0.05
    H = np.array([2.0, 1.5, 1.0, 0.55, 0.12, -0.25, -0.6, -1.0, -1.4])[None, :]       # bed depth below datum [m]
    surface = 0.4                                                                       # still level of the sea [m]
    eta = np.where(H + surface > 0, surface, -H)                                        # dry cells: η = −H (D = 0)
    eta[0, 5] = -H[0, 5] + 0.02                                                         # a 2 cm film (dry: < d_min)
    hg = create_cartesian_grid(nx=nx, ny=1, dx=1.0, dy=1.0, f0=0.0)
    fs = WetDryFreeSurface(hg, H, dt=1.0, d_min=d_min)
    Du, wet = fs._face_depth(eta, np.zeros((1, nx + 1)), axis=1)
    D = (H + eta)[0]

    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    for i in range(nx):
        bed = -H[0, i]
        ax.add_patch(Rectangle((i, -2.4), 1, bed + 2.4, facecolor=C["sand"], edgecolor=C["grey"], lw=0.6))
        if D[i] > 0:
            ax.add_patch(Rectangle((i, bed), 1, D[i], facecolor=C["sky"], alpha=0.55, edgecolor="none"))
        state = "wet" if D[i] > d_min else ("film" if D[i] > 0 else "dry")
        ax.text(i + 0.5, -2.25, f"H={H[0, i]:+.2f}\nD={D[i]:.2f}\n{state}", ha="center", fontsize=7.5)
        ax.plot(i + 0.5, eta[0, i], "o", color=C["blue"] if D[i] > d_min else C["grey"], ms=4)
    for f in range(1, nx):
        hf = fs.H_u[0, f]
        col = C["green"] if wet[0, f] else C["red"]
        if wet[0, f]:
            ax.plot([f, f], [-hf, -hf + Du[0, f]], color=col, lw=3)
        ax.text(f, 1.62, f"{'wet' if wet[0, f] else 'dry'}\n$D_f$={Du[0, f]:.2f}", ha="center", fontsize=7.5,
                color=col)
        ax.plot(f, -hf, "_", color=C["black"], ms=10, mew=2)
    ax.axhline(surface, color=C["blue"], lw=0.8, ls="--")
    ax.text(nx - 0.05, surface + 0.05, "sea level", ha="right", fontsize=8, color=C["blue"])
    ax.axhline(0.0, color=C["grey"], lw=0.6, ls=":")
    ax.text(0.05, 0.03, "datum z = 0", fontsize=8, color=C["grey"])
    ax.set_xlim(0, nx)
    ax.set_ylim(-2.45, 2.0)
    ax.set_xlabel("cell index i (u-faces at integer positions)")
    ax.set_ylabel("z [m]")
    ax.set_title(f"Face rule of WetDryFreeSurface on a beach (d_min = {d_min} m, u = 0): "
                 r"$D_f = \min(H_L, H_R) + \eta_{up}$, wet if $D_f > d_{min}$" "\n"
                 "black tick = face bed (the higher of the two cells); green bar = wet face total depth $D_f$ (dry faces: $D_f$ set to 0)",
                 fontsize=9.5)
    fig.tight_layout()
    fig.savefig(OUT / "wd_schematic.png", dpi=150)
    plt.close(fig)


def copy_validation() -> None:
    pairs = [("validation/thacker/out/thacker.png", "thacker.png"),
             ("validation/tidal_flat/out/tidal_flat.png", "tidal_flat_2dh.png"),
             ("validation/tidal_flat/out/tidal_flat_vs3d_z0.png", "tidal_flat_vs3d_z0.png"),
             ("validation/tidal_flat/out/tidal_flat_3d_gls.png", "tidal_flat_3d_gls.png")]
    for src, dst in pairs:
        shutil.copy2(ROOT / src, OUT / dst)


if __name__ == "__main__":
    fig_schematic()
    copy_validation()
    print("written to", OUT)
