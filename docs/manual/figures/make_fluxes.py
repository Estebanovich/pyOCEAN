"""
Figures of the manual page theory/air-sea-fluxes.md.

Writes PNGs to docs/manual/img/air-sea-fluxes/:
  transfer_coefficients.png   C_D, C_H (Stanton) and C_E (Dalton) against wind speed for unstable,
                              near-neutral and stable air, from the real forcing.bulk.bulk_ncar
                              (Large & Yeager 2004, 2 iterations) and forcing.bulk.bulk_kara
                              (Kara et al. 2005 = HYCOM flxflg=6)
  shortwave_profile.png       fraction of the penetrating shortwave left at depth z,
                              forcing.surface._sw_profile (Paulson & Simpson 1977, Jerlov I)
  mitgcm_exf_compare.png      copy of validation/mitgcm_exf/compare.png

Cheap function calls only (no time stepping).  Run from anywhere:
    python docs/manual/figures/make_fluxes.py
"""

from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS",
           "NUMBA_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

from forcing.bulk import bulk_kara, bulk_ncar, T0_K  # noqa: E402
from forcing.sources.era5 import specific_humidity_from_dewpoint  # noqa: E402
from forcing.surface import _sw_profile  # noqa: E402

OUT = ROOT / "docs" / "manual" / "img" / "air-sea-fluxes"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 9})

# Okabe–Ito colours (colour-blind safe)
COLORS = {"unstable": "#0072B2", "neutral": "#000000", "stable": "#D55E00"}

SST = 25.0          # °C
P = 101325.0        # Pa
RH = 0.8            # relative humidity of the air at 2 m
CASES = {           # T_air − SST [K]
    "unstable": -3.0,
    "neutral": 0.0,
    "stable": +3.0,
}


def q_from_rh(t_air_k: np.ndarray, rh: float, p: float) -> np.ndarray:
    """Specific humidity [kg/kg] of air at T with relative humidity rh: the dew point that gives e = rh·e_s(T)."""
    # e_s(T) with the IFS/Buck constants used by the ERA5 adapter; invert for the dew point
    a1, a3, a4, t3 = 611.21, 17.502, 32.19, 273.16
    es = a1 * np.exp(a3 * (t_air_k - t3) / (t_air_k - a4))
    x = np.log(rh * es / a1) / a3                      # (Td − T3)/(Td − a4)
    td = (t3 - x * a4) / (1.0 - x)
    return specific_humidity_from_dewpoint(td, p)


def transfer_coefficients() -> None:
    u = np.linspace(1.0, 30.0, 300)
    fig, axes = plt.subplots(1, 3, figsize=(10, 3.6), sharex=True)
    for name, dT in CASES.items():
        t_air = np.full_like(u, SST + dT + T0_K)
        q_air = q_from_rh(t_air, RH, P)
        sst = np.full_like(u, SST)
        p = np.full_like(u, P)
        nc = bulk_ncar(u, 0.0 * u, t_air, q_air, sst, p)
        ka = bulk_kara(u, 0.0 * u, t_air, q_air, sst, p, sss=35.0)
        for ax, a, b in zip(axes, (nc.cd, nc.ch, nc.ce), (ka.cd, ka.ch, ka.ce)):
            ax.plot(u, 1e3 * a, color=COLORS[name], lw=1.6, label=f"ncar, $T_a-$SST = {dT:+.0f} K")
            ax.plot(u, 1e3 * b, color=COLORS[name], lw=1.4, ls="--", label=f"kara, $T_a-$SST = {dT:+.0f} K")
    for ax, title in zip(axes, ("drag $C_D$", "sensible heat $C_H$", "latent heat $C_E$")):
        ax.set_title(title)
        ax.set_xlabel("wind speed $|\\Delta U|$ at 10 m [m s$^{-1}$]")
        ax.set_ylim(0.0, 3.0)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("coefficient $\\times 10^{3}$")
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, frameon=False, bbox_to_anchor=(0.5, -0.02))
    fig.suptitle(f"Transfer coefficients at 10 m: SST = {SST:.0f} °C, RH = {RH:.0%} at 2 m, "
                 f"p = {P / 100:.0f} hPa (bulk_ncar n_iter = 2, solid; bulk_kara, dashed)", fontsize=10)
    fig.tight_layout(rect=(0, 0.12, 1, 0.95))
    fig.savefig(OUT / "transfer_coefficients.png", dpi=150)
    plt.close(fig)


def shortwave_profile() -> None:
    z = -np.linspace(0.0, 60.0, 601)
    F = _sw_profile(z)
    fig, ax = plt.subplots(figsize=(4.2, 4.0))
    ax.plot(F, -z, color="#0072B2", lw=1.8, label="$F(z)=R\\,e^{z/\\mu_1}+(1-R)\\,e^{z/\\mu_2}$")
    ax.plot(0.58 * np.exp(z / 0.35), -z, color="#D55E00", ls="--", lw=1.2, label="red band, $\\mu_1$ = 0.35 m")
    ax.plot(0.42 * np.exp(z / 23.0), -z, color="#009E73", ls=":", lw=1.4, label="blue-green, $\\mu_2$ = 23 m")
    ax.invert_yaxis()
    ax.set_xlabel("fraction of surface shortwave remaining")
    ax.set_ylabel("depth [m]")
    ax.set_xlim(0, 1)
    ax.grid(alpha=0.3)
    ax.legend(loc="lower right", fontsize=8)
    ax.set_title("Jerlov type I (Paulson & Simpson 1977)")
    fig.tight_layout()
    fig.savefig(OUT / "shortwave_profile.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    transfer_coefficients()
    shortwave_profile()
    src = ROOT / "validation" / "mitgcm_exf" / "compare.png"
    if src.exists():
        shutil.copy(src, OUT / "mitgcm_exf_compare.png")
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
