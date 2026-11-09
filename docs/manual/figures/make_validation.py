#!/usr/bin/env python
"""
Summary charts for the Validation chapter of the pyOCEAN manual.

Every number below is copied from a README or summary file in ``validation/`` (path given next to each
constant).  Nothing is run: the script only draws.  Output: ``docs/manual/img/validation/summary_*.png``.

Usage:  python docs/manual/figures/make_validation.py
"""
from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.ticker  # noqa: E402,F401
import numpy as np  # noqa: E402

OUT = Path(__file__).resolve().parents[1] / "img" / "validation"
OUT.mkdir(parents=True, exist_ok=True)

# Categorical colours, fixed order (validated colour-blind-safe reference palette of the dataviz guide):
# one colour per model, the same in every chart.
C = {"pyOCEAN": "#2a78d6", "MOM6": "#eb6834", "CROCO": "#1baf7a", "MITgcm": "#eda100", "HYCOM": "#e87ba4",
     "TPXO10": "#6250d6"}
PY_HYB = "#184f95"          # darker step of the pyOCEAN blue for the hybrid coordinate
GREY = "#52514e"

plt.rcParams.update({"font.size": 10, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.color": "#e4e3df", "grid.linewidth": 0.6, "axes.axisbelow": True})


def save(fig: plt.Figure, name: str) -> None:
    fig.tight_layout()
    fig.savefig(OUT / name, dpi=150)
    plt.close(fig)
    print("wrote", OUT / name)


# ------------------------------------------------------------------------------------------------------------
# 1. Cost at the largest stable time step, one process (validation/ilicak_benchmark/README.md, "Costo", table
#    "Comparación principal"; HYCOM: validation/ilicak_benchmark/hycom/README.md, overflow only, ν = 1000)
COST = {  # case: {model: (dt_max [s], s per simulated hour)}
    "Lock exchange": {"pyOCEAN": (160, 0.049), "MOM6": (160, 0.073), "CROCO": (160, 0.015), "MITgcm": (160, 0.101)},
    "Overflow": {"pyOCEAN": (80, 0.45), "MOM6": (20, 4.10), "CROCO": (10, 0.91), "MITgcm": (10, 4.51),
                 "HYCOM": (20, 5.41)},
    "Internal wave": {"pyOCEAN": (2400, 0.0019), "MOM6": (2400, 0.0024), "CROCO": (2400, 0.0008),
                      "MITgcm": (2400, 0.0037)},
    "Seamount": {"pyOCEAN": (600, 0.22), "MOM6": (1200, 0.18), "CROCO": (600, 0.12), "MITgcm": (1200, 0.025)},
}


def chart_cost() -> None:
    fig, axes = plt.subplots(1, 4, figsize=(10, 3.6))
    for ax, (case, d) in zip(axes, COST.items()):
        names = list(d)
        vals = [d[n][1] for n in names]
        ax.bar(range(len(names)), vals, color=[C[n] for n in names], width=0.7)
        for k, n in enumerate(names):
            ax.text(k, vals[k] * 1.15, f"{vals[k]:g}\nΔt {d[n][0]}", ha="center", va="bottom", fontsize=8,
                    color="#0b0b0b")
        ax.set_yscale("log")
        ax.set_ylim(min(vals) / 3, max(vals) * 8)
        ax.set_xticks(range(len(names)), names, rotation=45, ha="right", fontsize=9)
        ax.set_title(case, fontsize=10)
        ax.grid(axis="x", visible=False)
    axes[0].set_ylabel("wall-clock s per simulated hour")
    fig.suptitle("Ilicak et al. (2012) cases: cost at the largest stable Δt [s], one process", fontsize=10)
    save(fig, "summary_ilicak_cost.png")


# ------------------------------------------------------------------------------------------------------------
# 2. Lock exchange: dRPE/dt [W m⁻²] (mean 15–17 h) against the grid Reynolds number Re_Δ = UΔx/ν_h
#    (validation/ilicak_benchmark/README.md, "Lock exchange (17 h)"; Re_Δ 2480, 25, 2.5 for ν 0.1, 10, 100).
#    CROCO is left out: its REGIONAL schemes create new T extrema, so its RPE is not comparable (README item 4).
RE = np.array([2480.0, 25.0, 2.5])
DRPE = {"pyOCEAN z*": [3.86e-3, 3.29e-3, 1.08e-3], "pyOCEAN hybrid": [4.2e-6, 4.3e-6, 4.3e-6],
        "MOM6 z*": [3.91e-3, 3.36e-3, 1.07e-3], "MOM6 ρ": [9.4e-4, 8.9e-4, 9.2e-4],
        "MITgcm z*": [4.28e-3, 2.88e-3, 7.7e-4]}
STYLE = {"pyOCEAN z*": (C["pyOCEAN"], "o", "-"), "pyOCEAN hybrid": (PY_HYB, "s", "-"),
         "MOM6 z*": (C["MOM6"], "o", "--"), "MOM6 ρ": (C["MOM6"], "s", ":"), "MITgcm z*": (C["MITgcm"], "^", "--")}


def chart_rpe() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, v in DRPE.items():
        col, mk, ls = STYLE[name]
        ax.plot(RE, v, ls, marker=mk, color=col, lw=2, ms=8, label=name)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("grid Reynolds number Re_Δ = UΔx/ν_h")
    ax.set_ylabel("dRPE/dt [W m⁻²], 15–17 h")
    ax.set_title("Lock exchange: spurious mixing against Re_Δ (lower is better)", fontsize=10)
    ax.legend(fontsize=8, ncol=2, loc="center right")
    save(fig, "summary_lock_rpe.png")


# ------------------------------------------------------------------------------------------------------------
# 3. Internal wave: oscillation ratio (std of the 15.1 °C isotherm, last 20 / first 20 of 200 days)
#    (validation/ilicak_benchmark/README.md, "Onda interna (200 días)")
NU_IW = [0.01, 15.0, 150.0]
OSC = {"pyOCEAN z*": [0.74, 0.58, 0.26], "pyOCEAN hybrid": [0.50, 0.63, 0.27], "MOM6 z*": [0.67, 0.55, 0.25],
       "MOM6 ρ": [0.65, 0.65, 0.26], "MITgcm z*": [0.72, 0.56, 0.24], "CROCO σ": [0.68, 0.60, 0.26]}
STYLE_IW = dict(STYLE, **{"CROCO σ": (C["CROCO"], "D", "--")})


def chart_iw() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 4.0))
    for name, v in OSC.items():
        col, mk, ls = STYLE_IW[name]
        ax.plot(NU_IW, v, ls, marker=mk, color=col, lw=2, ms=8, label=name)
    ax.set_xscale("log")
    ax.set_xlabel("horizontal viscosity ν_h [m² s⁻¹]")
    ax.set_ylabel("oscillation, final / initial")
    ax.set_ylim(0, 0.85)
    ax.set_title("Ilicak internal wave: damping after 200 days", fontsize=10)
    ax.legend(fontsize=8, ncol=2)
    save(fig, "summary_iw_damping.png")


# ------------------------------------------------------------------------------------------------------------
# 4. Seamount at rest: max|u| at day 10 [cm/s]
#    pyOCEAN variants: validation/seamount/README.md ("Resultados (10 días)", 384 × 288 km, Δx 6 km, 20 layers);
#    four-model rows: validation/ilicak_benchmark/README.md ("Seamount (10 días, sin viscosidad)").
#    MITgcm z* is exactly 0 (flat isopycnals on z-levels) and cannot be drawn on a log axis.
SEAMOUNT = [
    ("pyOCEAN σ, two-term PGF", 42.2, C["pyOCEAN"]),
    ("pyOCEAN σ, Jacobian", 42.2, C["pyOCEAN"]),
    ("pyOCEAN σ, Jacobian + ρ_ref", 0.41, C["pyOCEAN"]),
    ("pyOCEAN σ, FV + ρ_ref", 1.36, C["pyOCEAN"]),
    ("pyOCEAN z*, FV, no ρ_ref", 1.85, C["pyOCEAN"]),
    ("pyOCEAN z*, FV + ρ_ref", 0.10, C["pyOCEAN"]),
    ("4-model: pyOCEAN σ (FV + ρ_ref)", 0.21, C["pyOCEAN"]),
    ("4-model: pyOCEAN z*", 0.08, C["pyOCEAN"]),
    ("4-model: CROCO σ", 36.0, C["CROCO"]),
    ("4-model: MOM6 σ (growing)", 154.0, C["MOM6"]),
]


def chart_seamount() -> None:
    fig, ax = plt.subplots(figsize=(7.5, 4.6))
    y = np.arange(len(SEAMOUNT))[::-1]
    for yy, (name, v, col) in zip(y, SEAMOUNT):
        ax.barh(yy, v, color=col, height=0.65)
        ax.text(v * 1.15, yy, f"{v:g}", va="center", fontsize=8)
    ax.axvline(1.0, color=GREY, ls="--", lw=1.2)
    ax.text(1.05, y[-1] - 0.9, "Beckmann & Haidvogel (1993): ~1 cm/s", fontsize=8, color=GREY)
    ax.set_yticks(y, [s[0] for s in SEAMOUNT], fontsize=8)
    ax.set_xscale("log")
    ax.set_xlim(0.03, 600)
    ax.set_ylim(y[-1] - 1.3, y[0] + 0.6)
    ax.set_xlabel("spurious max|u| at day 10 [cm/s] (log scale; MITgcm z* = 0 exactly, not drawn)")
    ax.set_title("Seamount at rest (rx1 = 10.7): pressure-gradient error", fontsize=10)
    ax.grid(axis="y", visible=False)
    save(fig, "summary_seamount.png")


# ------------------------------------------------------------------------------------------------------------
# 5. PAC_MEX tides: median 8-constituent RSS against 31 TICON-3 records [cm]
#    (validation/pacmex_tides/out/summary.txt: all, Pacific, Gulf of California; TPXO10 the same in every run)
RSS_RUNS = ["C_d 2.5e-3", "JSL ×0.25", "JSL ×0.5", "JSL ×1.0", "JSL ×0.5 + cb"]
RSS = {"All 31 records": ([1.95, 1.82, 1.70, 1.63, 1.63], 1.87),
       "Pacific": ([1.60, 1.56, 1.50, 1.44, 1.46], 1.79),
       "Gulf of California": ([12.54, 12.39, 12.25, 11.98, 11.13], 5.12)}


def chart_tides() -> None:
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6))
    for ax, (region, (vals, tpxo)) in zip(axes, RSS.items()):
        x = np.arange(len(vals))
        ax.bar(x, vals, color=C["pyOCEAN"], width=0.7, label="pyOCEAN")
        ax.axhline(tpxo, color=C["TPXO10"], lw=2, ls="--", label=f"TPXO10 {tpxo:g}")
        for k, v in enumerate(vals):
            ax.text(k, v * 1.01, f"{v:.2f}", ha="center", va="bottom", fontsize=8)
        ax.set_xticks(x, RSS_RUNS, rotation=40, ha="right", fontsize=8)
        ax.set_ylim(0, max(max(vals), tpxo) * 1.2)
        ax.set_title(region, fontsize=10)
        ax.grid(axis="x", visible=False)
        ax.legend(fontsize=8, loc="lower right")
    axes[0].set_ylabel("median RSS vs gauges [cm]")
    fig.suptitle("PAC_MEX barotropic tides: bottom/internal-tide drag ensemble against TICON-3", fontsize=10)
    save(fig, "summary_tide_rss.png")


# ------------------------------------------------------------------------------------------------------------
# 6. Wetting and drying: TIDAL_FLAT against CROCO 3D (η rms over wet cells, last period [cm])
#    (validation/tidal_flat/README.md, "pyOCEAN 3D: capas sigma + GLS k-ε") and Thacker convergence
#    (validation/thacker/README.md, rms after one period vs Δx).
TF = [("2-D, C_d = 2.5e-3", 27.0), ("2-D, log-law z₀ = 1e-4", 4.5), ("3-D σ, parabolic ν", 5.2),
      ("3-D σ, GLS k-ε", 1.2)]
TH_DX = np.array([4.0, 2.0, 1.0, 0.5])
TH_RMS = np.array([10.2, 3.8, 1.8, 1.05])


def chart_wd() -> None:
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.8))
    y = np.arange(len(TF))[::-1]
    a1.barh(y, [v for _, v in TF], color=C["pyOCEAN"], height=0.6)
    for yy, (_, v) in zip(y, TF):
        a1.text(v + 0.4, yy, f"{v:g}", va="center", fontsize=9)
    a1.set_yticks(y, [n for n, _ in TF], fontsize=9)
    a1.set_xlabel("η rms difference from CROCO 3D [cm]")
    a1.set_title("TIDAL_FLAT, pyOCEAN vs CROCO 3D (GLS)", fontsize=10)
    a1.set_xlim(0, 31)
    a1.grid(axis="y", visible=False)
    a2.loglog(TH_DX, TH_RMS, "-o", color=C["pyOCEAN"], lw=2, ms=8, label="pyOCEAN")
    a2.loglog(TH_DX, TH_RMS[2] * (TH_DX / 1.0) ** 1, ":", color=GREY, label="slope 1")
    a2.loglog(TH_DX, TH_RMS[2] * (TH_DX / 1.0) ** 2, "--", color=GREY, label="slope 2")
    a2.set_xticks(TH_DX, [f"{d:g}" for d in TH_DX])
    a2.xaxis.set_minor_formatter(matplotlib.ticker.NullFormatter())
    a2.set_xlabel("Δx [km]")
    a2.set_ylabel("η rms after one period [cm]")
    a2.set_title("Thacker (1981) basin: convergence", fontsize=10)
    a2.legend(fontsize=8)
    save(fig, "summary_wetting_drying.png")


# ------------------------------------------------------------------------------------------------------------
# 7. BayIW cut (198 × 159 × 50, 5 processes or threads): wall-clock s per simulated hour at Δt = 120 s
#    (validation/mom6_bayiw/README.md, "Costo en el corte"; MITgcm with its own Δt = 30 s)
BAYIW_COST = [("MITgcm z, Δt 30 s", 4.4, C["MITgcm"]), ("MOM6 z*", 8.97, C["MOM6"]),
              ("pyOCEAN z*", 9.13, C["pyOCEAN"]), ("MOM6 HYCOM1 (25)", 4.67, C["MOM6"]),
              ("pyOCEAN hybrid (25)", 4.95, PY_HYB)]


def chart_bayiw_cost() -> None:
    fig, ax = plt.subplots(figsize=(6.5, 3.4))
    x = np.arange(len(BAYIW_COST))
    vals = [v for _, v, _ in BAYIW_COST]
    ax.bar(x, vals, color=[c for *_, c in BAYIW_COST], width=0.65)
    for k, v in enumerate(vals):
        ax.text(k, v + 0.15, f"{v:g}", ha="center", va="bottom", fontsize=9)
    ax.set_xticks(x, [n for n, *_ in BAYIW_COST], rotation=20, ha="right", fontsize=9)
    ax.set_ylabel("s per simulated hour")
    ax.set_ylim(0, 11)
    ax.set_title("BayIW 3-D cut: cost, 5 processes / threads, Δt = 120 s", fontsize=10)
    ax.grid(axis="x", visible=False)
    save(fig, "summary_bayiw_cost.png")


if __name__ == "__main__":
    chart_cost()
    chart_rpe()
    chart_iw()
    chart_seamount()
    chart_tides()
    chart_wd()
    chart_bayiw_cost()
