#!/usr/bin/env python
"""
Table 2 of Lorenz et al. (2021) for pyOCEAN against GETM (structure; the results and the (a)–(d) classification are
filled in after the eight long runs, N2a task 8 step 4).

GETM's column is transcribed from the local text copy (l. 923-1030, rounded as printed).  The text copy loses the
minus signs (printed "2"); signs were restored from the PDF and checked with eq. (32): D's dS2 is −35.16 (eq. 32
with the printed bulk values gives ≈ −37), B's +237 and A's +65.3.

Usage: python validation/lorenz_estuary/compare.py [--runs DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.lorenz_estuary import case  # noqa: E402

# Table 2 rows: (label, key in summary.json, transform of the summary value to the printed quantity)
ROWS = [
    ("s_in [g/kg]", "s_in", None),
    ("s_out [g/kg]", "s_out", None),
    ("[(s²)_in]^½ [g/kg]", "s2_in", np.sqrt),
    ("[(s²)_out]^½ [g/kg]", "s2_out", np.sqrt),
    ("dV [m³/s]", "dV", None),
    ("dS [g/kg m³/s]", "dS", None),
    ("dS2 [(g/kg)² m³/s]", "dS2", None),
    ("F_surf^{s²} [(g/kg)² m³/s]", "Fs2_surf", None),
    ("Me", "Me", None),
    ("Mc", "Mc", None),
    ("Mp", "Mp", None),
    ("Mcp", "Mcp", None),
    ("M_model^tot", "M_model_tot", None),
    ("M_model^phy", "M_model_phy", None),
    ("M_model^num", "M_model_num", None),
]

GETM = {   # Lorenz et al. (2021) Table 2, in the order of ROWS
    "A": [30.00, 32.67, 30.00, 32.73, 0.00, 1.18, 65.3, 5279, 212.3, 403.9, 203.6, 396.7, 212.3, 103.5, 108.8],
    "B": [36.63, 37.34, 36.63, 37.38, 0.36, 5.55, 237, 3982, 3646, 3678, 3982, 3982, 3646, 3438, 208],
    "C": [36.57, 37.12, 36.58, 37.18, 0.34, 6.94, 339.3, 3786, 3464, 3495, 3786, 3786, 3464, 2911, 553],
    "D": [24.04, 13.98, 24.05, 15.24, 0.30, 8.48, -35.16, 946.8, 2445, 2884, 2193, 2628, 2445, 2413, 32],
}

CONFIGS = ("sigma_gls", "ale_sigma")

# Classification of each difference (CLAUDE.md, Validation Strategy): (a) our bug, (b) missing physics,
# (c) numerical / configuration choice, (d) reference-model error.  Filled after the runs (task 8 step 4).
CLASSIFICATION: dict[str, str] = {}


def load(runs: Path, exp: str, config: str) -> dict | None:
    """summary.json of one run, or None when it has not been run."""
    f = runs / f"{exp}_{config}" / "summary.json"
    return json.loads(f.read_text()) if f.exists() else None


def column(summary: dict | None) -> list[float]:
    """The ROWS values of one run (NaN where missing)."""
    if summary is None:
        return [np.nan] * len(ROWS)
    out = []
    for _, key, fn in ROWS:
        v = summary.get(key, np.nan)
        out.append(float(fn(v)) if fn is not None and np.isfinite(v) else float(v))
    return out


def table(runs: Path, exp: str) -> str:
    """Markdown table of experiment ``exp``: GETM, pyOCEAN per configuration, and |Me − M_model^tot|/M."""
    cols = {c: column(load(runs, exp, c)) for c in CONFIGS}
    lines = [f"### Experiment {exp} (Table 1: Q_r, Q_evap, Q_precip, M2 = {case.TABLE_1[exp]})", "",
             "| Quantity | GETM | " + " | ".join(f"pyOCEAN {c}" for c in CONFIGS) + " |",
             "|---|---|" + "---|" * len(CONFIGS)]
    for r, (label, _, _) in enumerate(ROWS):
        lines.append(f"| {label} | {GETM[exp][r]:g} | " + " | ".join(f"{cols[c][r]:.4g}" for c in CONFIGS) + " |")
    ident = []
    for c in CONFIGS:
        s = load(runs, exp, c)
        ident.append("—" if s is None else f"{abs(s['Me'] - s['M_model_tot']) / abs(s['M_model_tot']):.1e}")
    lines.append("| \\|Me − M_tot\\|/M_tot | machine accuracy | " + " | ".join(ident) + " |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=case.RUNS)
    a = p.parse_args()
    for exp in sorted(case.TABLE_1):
        print(table(a.runs, exp), end="\n\n")


if __name__ == "__main__":
    main()
