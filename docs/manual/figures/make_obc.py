"""
Figures of the manual page theory/open-boundaries.md.

1. obc_schematic.png   One open edge of SigmaBaroclinic: exterior (OBData), the Flather face (depth mean, implicit in
                       the free-surface matrix), the Orlanski/nudged boundary face (baroclinic u′), the interior faces
                       1 and 2 used by the phase-speed estimate, and the sponge band with the MITgcm OBCS profile.
2. sponge_rates.png    The actual relaxation rates of core.dynamics.sigma_baroclinic._sponge_rates for the default
                       sponge_tau = (1200 s, 60 s) and two band widths, and the equivalent e-folding time.

Run:  NUMBA_NUM_THREADS=1 OMP_NUM_THREADS=1 python docs/manual/figures/make_obc.py
"""

from __future__ import annotations

import os
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
from matplotlib.patches import FancyArrowPatch, Rectangle  # noqa: E402

from core.dynamics.sigma_baroclinic import _sponge_rates  # noqa: E402

OUT = ROOT / "docs/manual/img/open-boundaries"
OUT.mkdir(parents=True, exist_ok=True)
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10})
C = {"blue": "#0072B2", "orange": "#E69F00", "green": "#009E73", "red": "#D55E00", "purple": "#CC79A7",
     "sky": "#56B4E9", "grey": "#666666", "black": "#000000"}


def fig_schematic() -> None:
    fig, ax = plt.subplots(figsize=(10.0, 4.6))
    ncell, nsp = 9, 5                               # cells drawn, sponge band width (cells)
    H = 1.0
    # exterior
    ax.add_patch(Rectangle((-3.5, 0), 3.5, H, facecolor="#e8e8e8", edgecolor="none"))
    ax.text(-1.75, 0.72, "exterior\n(OBData: η, ū, T(z), u_normal(z)\non its own z grid zw;\n"
            "or a resting exterior)", ha="center", va="center", fontsize=8.5)
    # sponge band shading: rate profile as alpha
    for d in range(1, nsp + 1):
        rate = (nsp - d) / nsp
        ax.add_patch(Rectangle((d - 1, 0), 1, H, facecolor=C["orange"], alpha=0.08 + 0.45 * rate,
                               edgecolor="none"))
    # cells
    for i in range(ncell):
        ax.add_patch(Rectangle((i, 0), 1, H, fill=False, edgecolor=C["grey"], lw=0.8))
        ax.plot(i + 0.5, 0.5, "o", color=C["black"], ms=3)
        for k in (0.25, 0.5, 0.75):                  # layers
            ax.plot([i, i + 1], [k, k], color=C["grey"], lw=0.4, ls=":")
    # faces
    ax.plot([0, 0], [0, H], color=C["red"], lw=3.5)
    ax.annotate("open face B\n• Flather: ū_B (implicit in the η matrix)\n• Orlanski r, nudging or clamp: u′_B\n"
                "• tracer: upstream (T_ext on inflow)", xy=(0, 0.15), xytext=(-3.45, 0.06), fontsize=8,
                color=C["red"], arrowprops=dict(arrowstyle="->", color=C["red"]))
    for i, lab in ((1, "face 1"), (2, "face 2")):
        ax.plot([i, i], [0, H], color=C["blue"], lw=2.2)
        ax.text(i, H + 0.04, lab, ha="center", color=C["blue"], fontsize=8.5)
    ax.text(0, H + 0.04, "B", ha="center", color=C["red"], fontsize=9, weight="bold")
    ax.annotate("", xy=(0.0, 0.62), xytext=(2.0, 0.62),
                arrowprops=dict(arrowstyle="<-", color=C["blue"], lw=1.2))
    ax.text(1.0, 0.65, r"$r = c\,\Delta t/\Delta x$ from $u'_1, u'_2$", ha="center", fontsize=8, color=C["blue"])
    # sponge brace
    ax.annotate("", xy=(0, -0.12), xytext=(nsp, -0.12), arrowprops=dict(arrowstyle="<->", color=C["orange"]))
    ax.text(nsp / 2, -0.24, "sponge band (sponge_cells = N): u′ → u′_ext and T → T_ext, rate (N − d)/(N τ(d))",
            ha="center", fontsize=8.5, color="#a05a00")
    ax.text(nsp + 2, -0.24, "interior", ha="center", fontsize=9)
    # barotropic arrow
    ax.add_patch(FancyArrowPatch((0.4, 0.9), (-0.6, 0.9), arrowstyle="-|>", mutation_scale=12, color=C["green"]))
    ax.text(-0.65, 0.97, r"$Q = L D\,[\bar u_{ext,out} + \sqrt{g/D}(\eta_c - \eta_{ext})]$", fontsize=8.5,
            color=C["green"], ha="right")
    ax.set_xlim(-3.6, ncell + 0.1)
    ax.set_ylim(-0.32, H + 0.15)
    ax.set_aspect("auto")
    ax.axis("off")
    ax.set_title("West open edge of SigmaBaroclinic (x–z view; dotted lines = layers, dots = T points)")
    fig.tight_layout()
    fig.savefig(OUT / "obc_schematic.png", dpi=150)
    plt.close(fig)


def fig_sponge() -> None:
    fig, axs = plt.subplots(1, 2, figsize=(9.6, 3.8))
    for n_cells, tau, col, lab in ((10, (1200.0, 60.0), C["blue"], "N = 10, sponge_tau = (1200, 60) s (default τ)"),
                                   (30, (1200.0, 60.0), C["orange"], "N = 30, (1200, 60) s (BayIW runs)"),
                                   (10, (1200.0, 120.0), C["green"], "N = 10, (1200, 120) s (channel tests)")):
        rc, ru, _ = _sponge_rates((1, 40), ("west",), n_cells, *tau)
        d = np.arange(1, 41)
        axs[0].plot(d, rc[0], "o-", color=col, ms=3, label=lab)
        with np.errstate(divide="ignore"):
            axs[1].semilogy(d, np.where(rc[0] > 0, 1.0 / rc[0], np.nan) / 3600.0, "o-", color=col, ms=3)
    axs[0].set_xlabel("cell distance from the open edge d (1 = boundary cell)")
    axs[0].set_ylabel("relaxation rate at cells [s⁻¹]")
    axs[0].set_title("_sponge_rates: (N − d) / (N τ(d)), τ(d) = [(N − d)τ_b + dτ_i]/N")
    axs[0].legend(fontsize=8)
    axs[0].grid(True, alpha=0.3)
    axs[1].set_xlabel("cell distance from the open edge d")
    axs[1].set_ylabel("e-folding time 1/rate [h]")
    axs[1].set_title("equivalent relaxation time (0 rate at d ≥ N)")
    axs[1].grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(OUT / "sponge_rates.png", dpi=150)
    plt.close(fig)
    rc, _, _ = _sponge_rates((1, 12), ("west",), 10, 1200.0, 60.0)
    print("default N=10 rates (cells 1..10):", np.array2string(rc[0, :10], precision=5))


if __name__ == "__main__":
    fig_schematic()
    fig_sponge()
    print("written to", OUT)
