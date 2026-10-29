#!/usr/bin/env python
"""
Table 2 of Lorenz et al. (2021) for pyOCEAN against GETM, with the (a)–(d) class of each difference (CLAUDE.md,
Validation Strategy; N2a task 8b), and the steadiness of each run over the averaging window (series.npz).

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
# (c) numerical / configuration choice, (d) reference-model error; "?" marks a hypothesis.  Per experiment: (row keys,
# class, reason).  Rows not listed agree with GETM or are exact (M_model^tot = Me).  Evidence: README "Resultados"
# (task 8b; one-cycle replays from the cycle-575 checkpoints locate M_num cell by cell).
SAL = ("s_in", "s_out", "s2_in", "s2_out")
STORE = ("dS", "dS2")
MIX = ("Fs2_surf", "Me", "Mc", "Mp", "Mcp", "M_model_tot")
SPLIT = ("M_model_phy", "M_model_num")
_DV = (("dV",), "d?", "GETM 0.30–0.36 m³/s = +10–12 cm of mean level in the region per 100 cycles; pyOCEAN ≤ 0.015; "
                      "open")
CLASSIFICATION: dict[str, list[tuple[tuple[str, ...], str, str]]] = {
    "A": [
        (("s_in", "s2_in"), "c", "ALE-σ 31.2 vs 30.0: constant κ = 1e-4 without tide spreads the evaporated salt "
                                 "west of the transect (GOTM k-ε gives κ ≪ 1e-4 in the stable outflow) [hypothesis]; "
                                 "σ+GLS = GETM"),
        (("s_out", "s2_out"), "c", "both within 1.5 %; closure"),
        (STORE, "c", "ALE-σ exactly steady; σ+GLS fluctuates (convective episodes: S_region std 0.34 g/kg, S_max "
                     "37–41), the window keeps storage; GETM's is not zero either"),
        (MIX, "c", "ALE-σ 3×: constant ν = κ = 1e-4 far above k-ε levels in a tideless stratified channel; σ+GLS "
                   "+27 %: CROCO Canuto-A k-ε vs GOTM k-ε and the window (Δt 120 s run: Me 295, 9 % apart)"),
        (SPLIT, "c", "WENO5 (+FCT) vs TVD Superbee; ALE-σ M_num ≈ 0 because constant κ leaves no sharp gradient"),
    ],
    "B": [
        (SAL, "c", "salt plug at x ≈ 34 km in both, σ+GLS saltier (max 40.6 vs ≈ 37), ALE-σ hardly any (31.6): "
                   "closure (constant low ν → strong exchange flushes the plug) [hypothesis]; neither run steady "
                   "(S_region −3.0 %/100 cycles σ+GLS, −1.3 % ALE-σ)"),
        _DV,
        (STORE, "c", "opposite sign: GETM salting, pyOCEAN still freshening from S = 30 at rest (initial state not "
                     "given in the article)"),
        (MIX, "c", "Q_r + Q_surf = 0, so Me ≈ F_surf^{s²} ∝ (surface S of the E zone)²: ALE-σ +4 %, σ+GLS +23 %"),
        (SPLIT, "c", "M_num ≈ Q_r s_head²: 589 of ALE-σ's 596 (one cycle) and 117 of σ+GLS's 585 in the river inflow "
                     "cell i = 499, where S = 0 enters water of S ≈ 10 (ALE-σ) / 3–6 (σ+GLS); rest: scheme"),
    ],
    "C": [
        (SAL, "c", "as B: σ+GLS plug saltier (40.6), ALE-σ none (31.6) [hypothesis]; neither steady (−3.0 %, "
                   "−1.2 %/100 cycles)"),
        _DV,
        (STORE, "c", "opposite sign, as B"),
        (MIX, "c", "ALE-σ −36 % (F −39 %): constant κ mixes away the thin fresh P layer that GETM keeps; σ+GLS +14 %"),
        (SPLIT, "c", "ALE-σ M_num 0.2 (no river, no sharp gradient under constant κ); σ+GLS 391 vs 553: scheme"),
    ],
    "D": [
        (SAL, "c", "ALE-σ s_in − s_out 2.7 vs 10.1 and Q_in 44 vs ≈ 7 m³/s (Knudsen with GETM's values): constant low "
                   "ν → strong gravitational circulation [hypothesis]; σ+GLS region nearly fresh (S_region 3–8, "
                   "head S = 0); a salt intrusion at cycles 478–494, just before the window"),
        _DV,
        (STORE, "c", "σ+GLS: the window (cycles 500–600) samples only the drain after a salt intrusion at cycles "
                     "478–494 (S_region 8.0 → 4.4, ≈ −0.036 per cycle): dS = −79 dominates Me; ALE-σ steady "
                     "(−0.06 %/100 cycles)"),
        (MIX, "c", "ALE-σ +92 % (strong exchange); σ+GLS −67 % from the storage of the drain (its Mp 2288 vs "
                   "GETM 2193)"),
        (SPLIT, "c", "ALE-σ M_num 79 = river inflow cell (78 in one cycle, S ≈ 3.5 there); σ+GLS 0.1: fresh head"),
    ],
}

# Steadiness over the averaging window (cycle means of series.npz): drifting when |slope| of the region-mean S is
# ≥ STEADY_REL of its mean per 100 cycles, fluctuating when its standard deviation is ≥ STEADY_REL of the mean.
STEADY_REL = 5e-3


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
    cls = row_classes(exp)
    lines = [f"### Experiment {exp} (Table 1: Q_r, Q_evap, Q_precip, M2 = {case.TABLE_1[exp]})", "",
             "| Quantity | GETM | " + " | ".join(f"pyOCEAN {c}" for c in CONFIGS) + " | Class |",
             "|---|---|" + "---|" * len(CONFIGS) + "---|"]
    for r, (label, key, _) in enumerate(ROWS):
        lines.append(f"| {label} | {GETM[exp][r]:g} | " + " | ".join(f"{cols[c][r]:.4g}" for c in CONFIGS)
                     + f" | {cls.get(key, '—')} |")
    ident = []
    for c in CONFIGS:
        s = load(runs, exp, c)
        ident.append("—" if s is None else f"{abs(s['Me'] - s['M_model_tot']) / abs(s['M_model_tot']):.1e}")
    lines.append("| \\|Me − M_tot\\|/M_tot | machine accuracy | " + " | ".join(ident) + " | — |")
    lines += ["", "Classes:"] + [f"- {', '.join(keys)}: ({c}) {why}" for keys, c, why in CLASSIFICATION.get(exp, [])]
    return "\n".join(lines)


def row_classes(exp: str) -> dict[str, str]:
    """Row key → class letter(s) of experiment ``exp`` (CLASSIFICATION); a key in several entries joins them."""
    out: dict[str, str] = {}
    for keys, c, _ in CLASSIFICATION.get(exp, []):
        for k in keys:
            out[k] = c if k not in out else f"{out[k]}, {c}"
    return out


def steadiness(runs: Path, exp: str, config: str, last: float = 100.0) -> dict | None:
    """
    Steadiness of one run over its last ``last`` M2 cycles from series.npz (records every ¼ cycle): cycle means of the
    region-mean salinity S_region [g kg⁻¹] and of the domain maximum S_max, their linear trend per 100 cycles, the
    standard deviation of the S_region cycle means, and a verdict ('steady', 'fluctuating', 'drifting'; STEADY_REL).
    Cycle means remove the tidal signal that a ¼-cycle sampling would alias.  None when the run has no series.
    """
    f = runs / f"{exp}_{config}" / "series.npz"
    if not f.exists():
        return None
    d = np.load(f)
    a, cols = d["data"], [str(c) for c in d["columns"]]
    t = a[:, cols.index("t")] / case.T_M2
    cyc = np.floor(t - 1e-6).astype(int)
    first = int(round(t[-1])) - int(round(last))
    idx = np.arange(first, int(round(t[-1])))
    sel = [i for i in idx if (cyc == i).any()]
    mean = lambda col: np.array([a[cyc == i, cols.index(col)].mean() for i in sel])   # noqa: E731
    Sr, Sm = mean("S_region"), mean("S_max")
    tc = np.asarray(sel) + 0.5
    slope, slope_max = (100.0 * np.polyfit(tc, x, 1)[0] for x in (Sr, Sm))
    rel, std = slope / Sr.mean(), Sr.std() / Sr.mean()
    verdict = "drifting" if abs(rel) >= STEADY_REL else "fluctuating" if std >= STEADY_REL else "steady"
    return dict(S_region=float(Sr.mean()), slope=float(slope), rel=float(rel), std=float(Sr.std()),
                S_max=float(Sm.mean()), slope_max=float(slope_max), verdict=verdict)


def steadiness_table(runs: Path, last: float = 100.0) -> str:
    """Markdown table of ``steadiness`` for every experiment and configuration."""
    lines = [f"### Steadiness over the last {last:g} cycles (cycle means)", "",
             "| Run | S_region [g/kg] | trend [g/kg per 100 cycles] | trend / mean | std | S_max | S_max trend | "
             "verdict |", "|---|---|---|---|---|---|---|---|"]
    for exp in sorted(case.TABLE_1):
        for c in CONFIGS:
            s = steadiness(runs, exp, c, last)
            if s is None:
                lines.append(f"| {exp} {c} | — | — | — | — | — | — | not run |")
                continue
            lines.append(f"| {exp} {c} | {s['S_region']:.3f} | {s['slope']:+.3f} | {100 * s['rel']:+.2f} % | "
                         f"{s['std']:.3f} | {s['S_max']:.2f} | {s['slope_max']:+.3f} | {s['verdict']} |")
    return "\n".join(lines)


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--runs", type=Path, default=case.RUNS)
    a = p.parse_args()
    for exp in sorted(case.TABLE_1):
        print(table(a.runs, exp), end="\n\n")
    print(steadiness_table(a.runs))


if __name__ == "__main__":
    main()
