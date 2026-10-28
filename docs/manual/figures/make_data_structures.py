#!/usr/bin/env python
"""
Figures of the manual page architecture/data-structures.md:

1. cgrid_cell.png    — one Arakawa C-grid cell (j, i) with the indices of T/η, u, v and f_q (q-points);
2. cgrid_domain.png  — a 3 × 2 domain: nx + 1 u-faces per row, ny + 1 v-faces per column;
3. vertical_index.png — the layer index convention: σ / ALE columns (k = 0 at the bed, interfaces 0 … N,
   η on top, −H at the bed) next to the legacy phase-1 z-levels (k = 0 at the surface).

Usage:  /opt/anaconda3/bin/python docs/manual/figures/make_data_structures.py
Output: docs/manual/img/data-structures/
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "img" / "data-structures"
# Okabe–Ito colour-blind-safe palette
C_T, C_U, C_V, C_Q, C_W = "#000000", "#0072B2", "#D55E00", "#009E73", "#CC79A7"
plt.rcParams.update({"font.size": 10})


def _arrow(ax, x0, y0, x1, y1, color, lw=2.0):
    ax.add_patch(FancyArrowPatch((x0, y0), (x1, y1), arrowstyle="-|>", mutation_scale=14, color=color, lw=lw))


def cgrid_cell() -> None:
    """One T-cell (j, i) with its four faces and four corners."""
    fig, ax = plt.subplots(figsize=(6.2, 5.6))
    ax.add_patch(Rectangle((0, 0), 1, 1, fill=False, lw=1.5))
    # T / eta / h at the centre
    ax.plot(0.5, 0.5, "o", color=C_T, ms=10)
    ax.text(0.5, 0.40, "T[k, j, i]\nη[j, i], H[j, i]\nh[k, j, i] (ALE)", ha="center", va="top", fontsize=9)
    # u faces: west (j, i) and east (j, i+1)
    for x, lab in ((0.0, "u[k, j, i]"), (1.0, "u[k, j, i+1]")):
        _arrow(ax, x - 0.09, 0.5, x + 0.09, 0.5, C_U)
        ax.text(x, 0.58, lab, ha="center", color=C_U, fontsize=9)
    # v faces: south (j, i) and north (j+1, i)
    for y, lab in ((0.0, "v[k, j, i]"), (1.0, "v[k, j+1, i]")):
        _arrow(ax, 0.5, y - 0.09, 0.5, y + 0.09, C_V)
        ax.text(0.56, y + 0.03 if y == 0 else y - 0.07, lab, color=C_V, fontsize=9)
    # q corners
    for (x, y), lab in (((0, 0), "f_q[j, i]"), ((1, 0), "f_q[j, i+1]"), ((0, 1), "f_q[j+1, i]"),
                        ((1, 1), "f_q[j+1, i+1]")):
        ax.plot(x, y, "s", color=C_Q, ms=8)
        ax.text(x + (-0.04 if x == 0 else 0.04), y + (-0.07 if y == 0 else 0.04), lab, color=C_Q, fontsize=9,
                ha="right" if x == 0 else "left")
    ax.annotate("", xy=(1.0, -0.22), xytext=(0.0, -0.22), arrowprops=dict(arrowstyle="<->", color="0.4"))
    ax.text(0.5, -0.29, "dx_t[j, i] (cell width); dy_u = face length L_u", ha="center", fontsize=8, color="0.3")
    ax.annotate("", xy=(-0.42, 1.0), xytext=(-0.42, 0.0), arrowprops=dict(arrowstyle="<->", color="0.4"))
    ax.text(-0.47, 0.5, "dy_t[j, i]", rotation=90, va="center", ha="right", fontsize=8, color="0.3")
    ax.set_xlim(-0.75, 1.55)
    ax.set_ylim(-0.4, 1.25)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Arakawa C-grid cell (j, i): x → i, y → j", fontsize=10)
    handles = [plt.Line2D([], [], marker="o", ls="", color=C_T, label="T-point: T, η, H, h, ρ (j_t, i_t)"),
               plt.Line2D([], [], marker=r"$\rightarrow$", ls="", ms=12, color=C_U, label="u-face (j_t, i_u)"),
               plt.Line2D([], [], marker=r"$\uparrow$", ls="", ms=12, color=C_V, label="v-face (j_v, i_t)"),
               plt.Line2D([], [], marker="s", ls="", color=C_Q, label="q-corner: f_q, ζ, PV (j_v, i_u)")]
    ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout()
    fig.savefig(OUT / "cgrid_cell.png", dpi=150)
    plt.close(fig)


def cgrid_domain() -> None:
    """A 3 × 2 domain with every point labelled by its array index."""
    nx, ny = 3, 2
    fig, ax = plt.subplots(figsize=(7.5, 5.0))
    for j in range(ny):
        for i in range(nx):
            ax.add_patch(Rectangle((i, j), 1, 1, fill=False, lw=1.0, color="0.5"))
            ax.plot(i + 0.5, j + 0.5, "o", color=C_T, ms=7)
            ax.text(i + 0.5, j + 0.38, f"T[{j},{i}]", ha="center", va="top", fontsize=8)
    for j in range(ny):
        for i in range(nx + 1):
            _arrow(ax, i - 0.08, j + 0.5, i + 0.08, j + 0.5, C_U, lw=1.6)
            ax.text(i, j + 0.6, f"u[{j},{i}]", ha="center", color=C_U, fontsize=8)
    for j in range(ny + 1):
        for i in range(nx):
            _arrow(ax, i + 0.5, j - 0.08, i + 0.5, j + 0.08, C_V, lw=1.6)
            ax.text(i + 0.56, j + 0.03, f"v[{j},{i}]", color=C_V, fontsize=8)
    for j in range(ny + 1):
        for i in range(nx + 1):
            ax.plot(i, j, "s", color=C_Q, ms=5)
    ax.text(nx / 2, -0.35, f"T, η: (ny, nx) = ({ny}, {nx});  u: (ny, nx+1) = ({ny}, {nx + 1});  "
            f"v: (ny+1, nx) = ({ny + 1}, {nx});  f_q: (ny+1, nx+1)", ha="center", fontsize=8.5)
    ax.text(-0.15, ny / 2, "west edge\nfaces u[:, 0]", ha="right", va="center", fontsize=8, color=C_U)
    ax.text(nx + 0.15, ny / 2, "east edge\nfaces u[:, nx]", ha="left", va="center", fontsize=8, color=C_U)
    ax.set_xlim(-1.2, nx + 1.2)
    ax.set_ylim(-0.6, ny + 0.4)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title("Index layout of a 3 × 2 domain (edge faces are walls unless the edge is open)", fontsize=10)
    fig.tight_layout()
    fig.savefig(OUT / "cgrid_domain.png", dpi=150)
    plt.close(fig)


def vertical_index() -> None:
    """σ / ALE (k = 0 at the bed) versus legacy z-levels (k = 0 at the surface)."""
    fig, axs = plt.subplots(1, 3, figsize=(10, 4.8), sharey=True)
    N, H, eta = 5, 20.0, 1.0

    # (a) sigma: h_k = D dsigma_k
    ax = axs[0]
    zw = -H + (H + eta) * np.linspace(0, 1, N + 1)
    for k in range(N):
        ax.add_patch(Rectangle((0, zw[k]), 1, zw[k + 1] - zw[k], fc="#56B4E9" if k % 2 else "#BFE3F7", ec="k", lw=0.8))
        ax.text(0.5, 0.5 * (zw[k] + zw[k + 1]), f"layer k = {k}", ha="center", va="center", fontsize=8.5)
    for k in range(N + 1):
        ax.text(1.04, zw[k], f"interface {k}", va="center", fontsize=7.5, color=C_W)
    ax.set_title("σ (SigmaBaroclinic default)\nh_k = D Δσ_k, D = H + η", fontsize=9.5)

    # (b) ALE z* over a shallow bed: vanished layers at the bed
    ax = axs[1]
    Hs = 12.0
    dz_top = np.full(N, H / N)                           # surface first
    s = (Hs + eta) / Hs
    zs = eta - s * np.concatenate([[0.0], np.cumsum(dz_top)])     # nominal interfaces, surface first
    zs = np.maximum(zs, -Hs)[::-1]                        # bed first, clipped at the bed
    for k in range(N):
        thick = zs[k + 1] - zs[k]
        if thick > 1e-9:
            ax.add_patch(Rectangle((0, zs[k]), 1, thick, fc="#E69F00" if k % 2 else "#F6D9A6", ec="k", lw=0.8))
            ax.text(0.5, 0.5 * (zs[k] + zs[k + 1]), f"layer k = {k}", ha="center", va="center", fontsize=8.5)
    ax.annotate("layers k = 0, 1 vanished\n(h ≈ 0 at the bed)", xy=(0.5, -Hs), xytext=(0.5, -17.5), ha="center",
                fontsize=8, arrowprops=dict(arrowstyle="->"))
    ax.add_patch(Rectangle((0, -H), 1, H - Hs, fc="0.75", ec="none"))
    ax.set_title("ALE z* (H = 12 m)\nh[k] prognostic, Σ h = H + η", fontsize=9.5)

    # (c) legacy z-levels: k = 0 at the top
    ax = axs[2]
    zf = -np.linspace(0, H, N + 1)
    for k in range(N):
        ax.add_patch(Rectangle((0, zf[k + 1]), 1, zf[k] - zf[k + 1], fc="#CC79A7" if k % 2 else "#EBC8DE", ec="k", lw=0.8))
        ax.text(0.5, 0.5 * (zf[k] + zf[k + 1]), f"level k = {k}", ha="center", va="center", fontsize=8.5)
    ax.set_title("legacy phase-1 z-levels\n(core/grid/vertical.py)", fontsize=9.5)

    for ax in axs:
        ax.axhline(0.0, color="0.4", ls=":", lw=0.8)
        ax.set_xlim(-0.1, 1.6)
        ax.set_xticks([])
    axs[0].axhline(eta, color=C_U, lw=1.5)
    axs[0].text(0.02, eta + 0.3, "η", color=C_U)
    axs[1].axhline(eta, color=C_U, lw=1.5)
    axs[0].set_ylabel("z [m] (0 at rest surface, −H at the bed)")
    axs[0].set_ylim(-21, 3)
    fig.tight_layout()
    fig.savefig(OUT / "vertical_index.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    OUT.mkdir(parents=True, exist_ok=True)
    cgrid_cell()
    cgrid_domain()
    vertical_index()
