"""
Effect of MOM6's bottom N² overwrite on the LOTW BBL diffusivity (TEOS-10 in the 3-D core, task 6 review,
spec decision 8; ``docs/design/teos10_refs/mom6_eos.md`` §5).  Measurement made before the fix: since 2026-11-01 the
model's BBL path applies the same overwrite (``physics.mixing.bbl_mixing.n2_bottom_bulk``), so this script now
records the size of the change the fix made.

MOM6 ``find_N2`` (MOM_set_diffusivity.F90:1202-1258, called at :481) replaces N² at the two lowest interior
interfaces with a bulk value before ``add_LOTW_BBL_diffusivity`` (:671, used at :1762).  Without tidal mixing
(h_amp = 0) the first pass from the bed exits at once and always adds one more interface (:1219-1223), so

    N²_bot = G (Δρ_1 + Δρ_2) / (½(h_0 + h_1) + ½(h_1 + h_2)),    Δρ_K = max(∂ρ/∂T ΔT + ∂ρ/∂S ΔS, 0)

at model interfaces 1 and 2 (k = 0 at the bed).  The cases of ``columns.bbl_cases`` give N² directly; with
N²_K = G Δρ_K / ½(h_K−1 + h_K) (:1197-1200; Boussinesq, h = dz) the clipped jump is
G Δρ_K = max(N²_K, 0) ½(h_K−1 + h_K).
This script computes the Kd of ``physics.mixing.bbl_mixing.lotw_bbl_column`` with the local N² (pyOCEAN today) and
with N²_bot at the two lowest interfaces (MOM6), and prints per column the max and median relative Kd change over
the interfaces where either Kd is > 0.

Usage:  python validation/mom6_column/bbl_n2bot_effect.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from columns import bbl_cases  # noqa: E402
from physics.mixing.bbl_mixing import BBLParams, lotw_bbl_column  # noqa: E402


def n2_bottom_overwrite(dz: np.ndarray, n2: np.ndarray) -> np.ndarray:
    """N² (N+1,) with interfaces 1 and 2 (bed first) replaced by MOM6's N²_bot (module docstring; h_amp = 0).
    dz (N,) [m], N ≥ 3; n2 (N+1,) [s⁻²] of the clipped jumps' column.  Returns a copy."""
    dz, out = np.asarray(dz, float), np.array(n2, float)
    if dz.size < 3:
        raise ValueError("needs at least three layers (two interior interfaces above the bed layer)")
    dzi = 0.5 * (dz[:2] + dz[1:3])                                     # ½(h_K−1 + h_K) at interfaces 1, 2
    gdrho = np.maximum(out[1:3], 0.0) * dzi                            # G Δρ_K, clipped (:1173-1176)
    out[1] = out[2] = gdrho.sum() / dzi.sum()                          # (:1229-1231, 1251, 1253)
    return out


def measure() -> list[tuple[str, float, float]]:
    rows = []
    for c in bbl_cases():
        p = BBLParams(**c["p"])
        k_loc = lotw_bbl_column(c["dz"], c["n2"], c["ustar"], c["absf"], c["tke"], p)
        k_bot = lotw_bbl_column(c["dz"], n2_bottom_overwrite(c["dz"], c["n2"]), c["ustar"], c["absf"], c["tke"], p)
        act = (k_loc > 0) | (k_bot > 0)
        rel = np.abs(k_bot - k_loc)[act] / np.maximum(np.abs(k_loc), np.abs(k_bot))[act] if act.any() else np.zeros(1)
        rows.append((c["name"], float(rel.max()), float(np.median(rel))))
    return rows


if __name__ == "__main__":
    print(f"{'column':32s} {'max rel ΔKd':>12s} {'median rel ΔKd':>15s}")
    for name, mx, md in measure():
        print(f"{name:32s} {mx:12.3e} {md:15.3e}")
