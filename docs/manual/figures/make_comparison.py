"""
Figures of the manual page "Comparison with other models" (docs/manual/comparison.md).

Writes PNGs to docs/manual/img/comparison/.  No model is run: every number is copied from a
validation README, cited next to the data below.

    python docs/manual/figures/make_comparison.py
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "img" / "comparison"
OUT.mkdir(parents=True, exist_ok=True)

# Okabe–Ito colour-blind-safe palette
C = {"pyOCEAN z*": "#0072B2", "pyOCEAN hybrid": "#56B4E9", "MOM6 z*": "#E69F00", "MOM6 ρ": "#F0E442",
     "CROCO σ": "#009E73", "MITgcm z*": "#D55E00", "HYCOM": "#CC79A7"}
plt.rcParams.update({"font.size": 10, "axes.titlesize": 11, "legend.fontsize": 9})


def cost_ilicak() -> None:
    """Cost at the largest stable Δt, one process each (validation/ilicak_benchmark/README.md, "Costo",
    table "Comparación principal", re-measured 2026-10-14).  Seamount: MOM6 column is MOM6 σ."""
    cases = ["Lock exchange", "Overflow", "Internal wave", "Seamount"]
    cost = {                       # s of wall clock per simulated hour
        "pyOCEAN z*": [0.049, 0.45, 0.0019, 0.22],
        "MOM6 z*": [0.073, 4.10, 0.0024, 0.18],
        "CROCO σ": [0.015, 0.91, 0.0008, 0.12],
        "MITgcm z*": [0.101, 4.51, 0.0037, 0.025],
        "HYCOM": [np.nan, 5.41, np.nan, np.nan],      # overflow only, ν = 1000, Δt 20 s (marginal)
    }
    dtmax = {                      # largest stable Δt [s] ("≥ 160" = top of the ladder, plotted as 160)
        "pyOCEAN z*": [160, 80, 2400, 600],
        "MOM6 z*": [160, 20, 2400, 1200],
        "CROCO σ": [160, 10, 2400, 600],
        "MITgcm z*": [160, 10, 2400, 1200],
        "HYCOM": [np.nan, 20, np.nan, np.nan],
    }
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    x = np.arange(len(cases))
    w = 0.16
    for ax, data, ylab, title in ((axes[0], cost, "wall-clock s per simulated hour", "(a) cost at the largest stable Δt"),
                                  (axes[1], dtmax, "largest stable Δt [s]", "(b) largest stable Δt")):
        for m, (name, vals) in enumerate(data.items()):
            ax.bar(x + (m - 2) * w, vals, w, color=C[name], edgecolor="k", linewidth=0.4, label=name)
        ax.set_yscale("log")
        ax.set_xticks(x, cases)
        ax.set_ylabel(ylab)
        ax.set_title(title)
        ax.grid(axis="y", which="both", alpha=0.3)
    axes[1].text(0, 200, "≥ 160 for all\n(top of ladder)", ha="center", fontsize=8)
    axes[0].legend(loc="upper right", ncol=1, frameon=False)
    fig.text(0.5, 0.005, "One process per model (pyOCEAN one thread). Seamount: MOM6 in σ. HYCOM: overflow only, ν = 1000. "
             "Source: validation/ilicak_benchmark/README.md", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "cost_ilicak.png", dpi=150)
    plt.close(fig)


def mixing_ilicak() -> None:
    """Lock exchange dRPE/dt (mean 15–17 h) and internal-wave oscillation ratio (last/first 20 days)
    against the grid Reynolds number (validation/ilicak_benchmark/README.md, "Resultados").  CROCO is
    left out of (a): its REGIONAL schemes create new T extrema, so its RPE is not comparable (README item 4)."""
    re_lock = np.array([2480, 25, 2.5])
    drpe = {
        "pyOCEAN z*": [3.86e-3, 3.29e-3, 1.08e-3],
        "pyOCEAN hybrid": [4.2e-6, 4.3e-6, 4.3e-6],
        "MOM6 z*": [3.91e-3, 3.36e-3, 1.07e-3],
        "MOM6 ρ": [9.4e-4, 8.9e-4, 9.2e-4],
        "MITgcm z*": [4.28e-3, 2.88e-3, 7.7e-4],
    }
    re_iw = np.array([1.9e5, 120, 12])
    osc = {
        "pyOCEAN z*": [0.74, 0.58, 0.26],
        "pyOCEAN hybrid": [0.50, 0.63, 0.27],
        "MOM6 z*": [0.67, 0.55, 0.25],
        "MOM6 ρ": [0.65, 0.65, 0.26],
        "CROCO σ": [0.68, 0.60, 0.26],
        "MITgcm z*": [0.72, 0.56, 0.24],
    }
    marks = {"pyOCEAN z*": "o", "pyOCEAN hybrid": "s", "MOM6 z*": "^", "MOM6 ρ": "v", "CROCO σ": "D",
             "MITgcm z*": "P"}
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.2))
    for name, vals in drpe.items():
        axes[0].loglog(re_lock, vals, marker=marks[name], color=C[name], mec="k", mew=0.4, label=name)
    axes[0].set_xlabel(r"grid Reynolds number Re$_\Delta$ = U$\Delta$x/$\nu_h$")
    axes[0].set_ylabel(r"dRPE/dt [W m$^{-2}$]")
    axes[0].set_title("(a) lock exchange: spurious mixing rate")
    axes[0].grid(which="both", alpha=0.3)
    axes[0].legend(frameon=False, loc="center right")
    for name, vals in osc.items():
        axes[1].semilogx(re_iw, vals, marker=marks[name], color=C[name], mec="k", mew=0.4, label=name)
    axes[1].set_xlabel(r"grid Reynolds number Re$_\Delta$")
    axes[1].set_ylabel("oscillation left after 200 days (final / initial)")
    axes[1].set_title("(b) internal wave: damping")
    axes[1].grid(which="both", alpha=0.3)
    axes[1].set_ylim(0, 0.85)
    fig.text(0.5, 0.005, "Source: validation/ilicak_benchmark/README.md (runs of 2026-10-13/14)", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "mixing_ilicak.png", dpi=150)
    plt.close(fig)


def cost_bayiw_cut() -> None:
    """BayIW 3-D cut (198 × 159, two-layer with bay, wind, radiating boundaries), 5 processes or threads
    (validation/mom6_bayiw/README.md, "Costo en el corte BayIW").  MITgcm: 4.4 s/h at its Δt = 30 s."""
    dt = np.array([120, 240, 360, 600])
    mom6_z = [8.97, 4.51, 3.06, 1.94]
    mom6_h = [4.67, 2.37, 1.62, 1.02]
    py_z = {120: 9.13, 600: 1.82}
    py_h = {120: 4.95, 600: 0.99}
    fig, ax = plt.subplots(figsize=(6.5, 4))
    ax.loglog(dt, mom6_z, "^-", color=C["MOM6 z*"], mec="k", mew=0.4, label="MOM6 z* (50 layers)")
    ax.loglog(dt, mom6_h, "v--", color=C["MOM6 z*"], mec="k", mew=0.4, label="MOM6 HYCOM1 (25 layers)")
    ax.loglog(list(py_z), list(py_z.values()), "o-", color=C["pyOCEAN z*"], mec="k", mew=0.4,
              label="pyOCEAN z* (50 layers)")
    ax.loglog(list(py_h), list(py_h.values()), "s--", color=C["pyOCEAN hybrid"], mec="k", mew=0.4,
              label="pyOCEAN hybrid (25 layers)")
    ax.loglog([30], [4.4], "P", color=C["MITgcm z*"], mec="k", mew=0.4, ms=9, label="MITgcm z (50 levels, Δt 30 s)")
    ax.set_xlabel("time step Δt [s]")
    ax.set_ylabel("wall-clock s per simulated hour")
    ax.set_title("BayIW 3-D cut, 5 processes / threads")
    ax.grid(which="both", alpha=0.3)
    ax.legend(frameon=False, fontsize=8, loc="lower left")
    fig.text(0.5, 0.005, "Source: validation/mom6_bayiw/README.md", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.03, 1, 1))
    fig.savefig(OUT / "cost_bayiw_cut.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    cost_ilicak()
    mixing_ilicak()
    cost_bayiw_cut()
    print("wrote", sorted(p.name for p in OUT.glob("*.png")))
