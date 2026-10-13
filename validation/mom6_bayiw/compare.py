#!/usr/bin/env python
"""
Side-by-side table of the BayIW 2-D section (spec 2026-10-13-mom6-bayiw-benchmark-design §3): MITgcm 'orl' as the
reference; pyOCEAN z*/hybrid and MOM6 z*/HYCOM1 series (bayiw_2d series.npz).  Per station (with bay): V110 and
T110 rms ratio · zero-lag correlation, and in two layers the heat-based interface rise ζ (bayiw_cut.compare
.heat_displacement on MITgcm levels 20–40); per model: the interface thickness and the T-section error at 24 h.

Usage: python validation/mom6_bayiw/compare.py 2layer|real
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.bayiw_2d import compare as c2d, section  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_baroclinic.compare import corr_lag, relative_field_error, rms  # noqa: E402
from validation.bayiw_cut import compare as ccut  # noqa: E402

MODELS = {"pyOCEAN z*": "zstar", "pyOCEAN híbrida": "hybrid", "MOM6 z*": "mom6zstar", "MOM6 HYCOM1": "mom6hycom1"}


def zeta(T_levels: np.ndarray) -> np.ndarray:
    """Heat-based interface rise (n, ny) [m] from T on the 50 MITgcm levels (n, 50, ny), surface first."""
    T = T_levels[:, 19:40, :, None]
    return ccut.heat_displacement(np.where(T == 0, np.nan, T), ccut.DZ, ccut.T_UP, ccut.T_LO)[:, :, 0]


def table(case: str) -> list[str]:
    H = section.section_inputs(case, "bay")["H"][:, 0]
    mit = section.read_mitgcm(case, "bay", "orl")
    tm = mit["t_h"]
    lev = dict(T=mit["T"], V=mit["V"], zc=bayiw.Z_C)
    ref = {"V110": c2d.mit_at_depth(lev, H, "V", bayiw.Z110), "T110": c2d.mit_at_depth(lev, H, "T", bayiw.Z110)}
    if case == "2layer":
        ref["zeta"] = zeta(mit["T"])
    L = [f"# BayIW 2-D section {case} with bay: rms model/MITgcm · zero-lag corr (MITgcm 'orl' = reference)"]
    keys = ("V110", "T110") + (("zeta",) if case == "2layer" else ())
    for key in keys:
        L.append(f"\n## {key}")
        L.append("station  " + "".join(f"{m:>20s}" for m in MODELS))
        for st, j in section.SECTION_STATIONS.items():
            r = ref[key][:, j] - (ref[key][0, j] if key != "V110" else 0.0)
            if not ccut.resolvable(key, r):
                continue
            row = []
            for tag in MODELS.values():
                p = dict(np.load(section.RUNS / "pyocean" / f"{case}_bay_{tag}" / "series.npz"))
                x = zeta(p["T_sec"])[:, j] if key == "zeta" else p[key][:, j]
                x = np.interp(tm, p["t_h"], x - (x[0] if key != "V110" else 0.0))
                row.append(f"{rms(x) / rms(r):9.2f} · {corr_lag(tm, r, x, 0.0)[0]:5.2f}")
            L.append(f"{st:8s} " + "".join(f"{v:>20s}" for v in row))
    L.append("\n## T section at 24 h: relative rms error of the anomaly; interface thickness [m] (MITgcm "
             f"{c2d.interface_thickness(mit['T'][int(np.argmin(abs(tm - 24)))], bayiw.Z_C, ccut.T_LO, ccut.T_UP):.1f})"
             if case == "2layer" else "\n## T section at 24 h: relative rms error of the anomaly")
    k0 = int(np.argmin(abs(tm - 24.0)))
    for name, tag in MODELS.items():
        p = dict(np.load(section.RUNS / "pyocean" / f"{case}_bay_{tag}" / "series.npz"))
        kp = int(np.argmin(abs(p["t_h"] - 24.0)))
        err = relative_field_error(np.nan_to_num(p["T_sec"][kp] - p["T_sec"][0]), mit["T"][k0] - mit["T"][0])
        extra = (f", interface {c2d.interface_thickness(np.nan_to_num(p['T_sec'][kp]), bayiw.Z_C, ccut.T_LO, ccut.T_UP):.1f}"
                 if case == "2layer" else "")
        L.append(f"{name:18s} {err:5.2f}{extra}")
    print("\n".join(L))
    return L


if __name__ == "__main__":
    table(sys.argv[1])
