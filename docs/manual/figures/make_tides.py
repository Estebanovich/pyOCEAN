"""
Figures of the manual page theory/tides.md.

Writes PNGs to docs/manual/img/tides/:
  nodal_factors.png       nodal amplitude factor f and phase correction u of M2, S2, K1, O1, 2025–2045
                          (forcing.tides.nodal_corrections, OTPS / Schureman 1958)
  equilibrium_tide.png    equilibrium tide α η_eq at Mazatlán (23.2°N, 106.4°W), August 2026:
                          the 8 constituents and M2 alone (forcing.tides.equilibrium_constants + predict)
  sal_m2_pacmex.png       copy of validation/tides/sal_m2_pacmex.png
  m2_map_pacmex.png       copy of validation/pacmex_tides/out/m2_map_jsl0.5_cb.png

Cheap function calls only (no time stepping).  Run from anywhere:
    /opt/anaconda3/bin/python docs/manual/figures/make_tides.py
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

from forcing.tides import CONSTITUENTS, equilibrium_constants, nodal_corrections, predict, to_mjd  # noqa: E402

OUT = ROOT / "docs" / "manual" / "img" / "tides"
plt.rcParams.update({"font.size": 10, "axes.titlesize": 10, "legend.fontsize": 9})
COLORS = {"m2": "#0072B2", "s2": "#009E73", "k1": "#D55E00", "o1": "#CC79A7"}


def nodal_factors() -> None:
    t = np.arange(np.datetime64("2025-01-01"), np.datetime64("2045-01-02"), np.timedelta64(10, "D"))
    cons = ["m2", "s2", "k1", "o1"]
    f, u = nodal_corrections(to_mjd(t), cons)
    years = 2025 + (t - np.datetime64("2025-01-01")) / np.timedelta64(1, "D") / 365.25
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    for k, c in enumerate(cons):
        a1.plot(years, f[:, k], color=COLORS[c], lw=1.6, label=c.upper())
        a2.plot(years, np.rad2deg(u[:, k]), color=COLORS[c], lw=1.6, label=c.upper())
    a1.set_ylabel("amplitude factor $f$ [–]")
    a2.set_ylabel("phase correction $u$ [°]")
    a2.set_xlabel("year")
    for a in (a1, a2):
        a.grid(alpha=0.3)
    a1.legend(ncol=4, loc="upper center")
    a1.set_title("Nodal corrections (18.61-year lunar node cycle), forcing.tides.nodal_corrections")
    fig.tight_layout()
    fig.savefig(OUT / "nodal_factors.png", dpi=150)
    plt.close(fig)


def equilibrium_tide() -> None:
    lon, lat = np.array([-106.4]), np.array([23.2])
    t = np.arange(np.datetime64("2026-08-01T00:00"), np.datetime64("2026-08-31T00:00"), np.timedelta64(30, "m"))
    cons = list(CONSTITUENTS)
    hc = equilibrium_constants(lon, lat, cons)                  # (1, 8), α included
    all8 = predict(t, hc, cons)[:, 0]
    m2 = predict(t, hc[:, :1], ["m2"])[:, 0]
    diurnal = predict(t, hc[:, [2, 3, 5, 7]], ["k1", "o1", "p1", "q1"])[:, 0]
    days = (t - t[0]) / np.timedelta64(1, "D")
    fig, ax = plt.subplots(figsize=(9, 3.4))
    ax.plot(days, 100 * all8, color="#000000", lw=1.0, label="8 constituents")
    ax.plot(days, 100 * m2, color="#0072B2", lw=0.9, alpha=0.8, label="M2 only")
    ax.plot(days, 100 * diurnal, color="#D55E00", lw=0.9, alpha=0.8, label="K1+O1+P1+Q1")
    ax.set_xlabel("days since 2026-08-01 00:00 UT")
    ax.set_ylabel("$\\alpha\\,\\eta_{eq}$ [cm]")
    ax.set_title("Equilibrium tide with Love-number factor α = 1 + k − h at 23.2°N, 106.4°W")
    ax.grid(alpha=0.3)
    ax.legend(ncol=3, loc="upper right")
    fig.tight_layout()
    fig.savefig(OUT / "equilibrium_tide.png", dpi=150)
    plt.close(fig)


def main() -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    nodal_factors()
    equilibrium_tide()
    for src, dst in ((ROOT / "validation" / "tides" / "sal_m2_pacmex.png", "sal_m2_pacmex.png"),
                     (ROOT / "validation" / "pacmex_tides" / "out" / "m2_map_jsl0.5_cb.png", "m2_map_pacmex.png")):
        if src.exists():
            shutil.copy(src, OUT / dst)
    print("wrote", sorted(p.name for p in OUT.iterdir()))


if __name__ == "__main__":
    main()
