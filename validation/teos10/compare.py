#!/usr/bin/env python
"""
Tables of the TEOS-10 validation (Task 9 of docs/superpowers/plans/2026-10-31-teos10-3d.md; spec criteria 4, 5 and
10) from the runs of ``run_pyocean.py`` (``cases.RUNS``) and of ``validation/bulk_column/run_pyocean.py --eos``;
written into ``README.md`` between the markers ``<!-- teos10-tables:start -->`` and ``<!-- teos10-tables:end -->``
(the classification and the "what pyOCEAN should improve" list stay hand-written below them).

    python -m validation.teos10.compare [--print]
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validation.teos10 import cases as C

README = Path(__file__).parent / "README.md"
START, END = "<!-- teos10-tables:start -->", "<!-- teos10-tables:end -->"
BULK = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column")


def _load(name: str) -> dict | None:
    p = C.RUNS / f"{name}.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}


def rest_table() -> list[str]:
    """Criterion 4: initial max |PGF| and max over 10 days of max |u|, |v| per case, ratio to the linear EOS."""
    out = ["| case | coord | ρ_ref | EOS | max\\|PGF\\| t=0 [m s⁻²] | max\\|u\\| 10 d [m s⁻¹] | final [m s⁻¹] | "
           "ratio to linear | ≤ 2× |", "|---|---|---|---|---|---|---|---|---|"]
    for kind in ("seamount", "bayiw_step"):
        for coord in ("zstar", "hybrid"):
            for ref in ("auto", "none"):
                lin = _load(f"rest_{kind}_linear_{coord}_{ref}")
                for eos in C.EOS_NAMES:
                    r = _load(f"rest_{kind}_{eos}_{coord}_{ref}")
                    if r is None:
                        continue
                    umax = float(r["umax"].max())
                    ratio = umax / float(lin["umax"].max()) if lin is not None else np.nan
                    ok = "—" if eos == "linear" or ref == "none" else ("yes" if ratio <= 2.0 else "**no**")
                    out.append(f"| {kind} | {coord} | {ref} | {eos} | {float(r['pgf0']):.2e} | {umax:.2e} | "
                               f"{float(r['umax_final']):.2e} | {ratio:.2f} | {ok} |")
    return out


def cabbeling_table() -> list[str]:
    """Criterion 5a: denser water than both sources (volume, excess-weighted height) and the flow it drives."""
    out = ["| EOS | first denser water [h] | volume 2 h / 6 h / 1 d [10⁹ m³] | domain [10⁹ m³] | "
           "lowest excess-weighted height [m] (at [h]) | max\\|u\\| [m s⁻¹] |", "|---|---|---|---|---|---|"]
    for eos in C.EOS_NAMES:
        r = _load(f"cabbeling_{eos}")
        if r is None:
            continue
        t_h = r["time"] / 3600.0
        v = r["dense_vol"]
        first = f"{t_h[np.argmax(v > 0)]:.0f}" if (v > 0).any() else "never"
        pick = [float(np.interp(h, t_h, v)) / 1e9 for h in (2.0, 6.0, 24.0)]
        z = r["dense_z"]
        if np.isfinite(z).any():
            k = int(np.nanargmin(z))
            low = f"{z[k]:.1f} ({t_h[k]:.0f})"
        else:
            low = "—"
        dom = 64 * 500.0 * 500.0 * 200.0 / 1e9
        out.append(f"| {eos} | {first} | {pick[0]:.2f} / {pick[1]:.2f} / {pick[2]:.2f} | {dom:.1f} | {low} | "
                   f"{float(r['umax'].max()):.2e} |")
    return out


def thermobaric_table() -> list[str]:
    """Criterion 5b: fraction of the bed water's heat anomaly above the initial interface."""
    out = ["| EOS | interface [m] | rise 1 h | rise 6 h | rise 10 d | bed T final [°C] | overturn |",
           "|---|---|---|---|---|---|---|"]
    for eos in C.EOS_NAMES:
        for d in (100, 4000):
            r = _load(f"thermobaric_{eos}_{d}")
            if r is None:
                continue
            t_h = r["time"] / 3600.0
            rise = r["rise"]
            a, b = (float(np.interp(h, t_h, rise)) for h in (1.0, 6.0))
            out.append(f"| {eos} | {d} | {a:.3f} | {b:.3f} | {float(rise[-1]):.3f} | {float(r['T_col'][-1, 0]):.3f} |"
                       f" {'yes' if rise[-1] > 0.5 else 'no'} |")
    return out


def bulk_table() -> list[str]:
    """Criterion 10: the N2b ERA5 column ('fms', 45 days) with the three EOS; SST as θ."""
    out = ["| EOS | c_p [J kg⁻¹ K⁻¹] | final SST θ [°C] | mean SST last 5 d [°C] | Δ vs linear [°C] | "
           "Δ teos10 − wright_full [°C] | final hbl [m] |", "|---|---|---|---|---|---|---|"]
    runs = {"linear": BULK / "pyocean_45d" / "fms.npz", "wright_full": BULK / "pyocean_45d_wright_full" / "fms.npz",
            "teos10": BULK / "pyocean_45d_teos10" / "fms.npz"}
    cp = {"linear": 3985.0, "wright_full": 3991.86795711963, "teos10": 3991.86795711963}
    sst, mean5 = {}, {}
    hbl = {}
    for eos, p in runs.items():
        if not p.exists():
            continue
        with np.load(p) as d:
            th = d["theta"] if "theta" in d.files else d["T"]
            t = d["time_snap"]
            sst[eos] = float(th[-1, -1])
            mean5[eos] = float(th[t >= t[-1] - 5 * 86400.0, -1].mean())
            hbl[eos] = float(d["hbl"][-1])
    for eos in sst:
        dl = sst[eos] - sst["linear"] if "linear" in sst else np.nan
        dw = sst["teos10"] - sst["wright_full"] if eos == "teos10" and "wright_full" in sst else np.nan
        out.append(f"| {eos} | {cp[eos]:.2f} | {sst[eos]:.3f} | {mean5[eos]:.3f} | {dl:+.3f} | "
                   f"{'' if np.isnan(dw) else f'{dw:+.3f}'} | {hbl[eos]:.1f} |")
    return out


def tables() -> str:
    lin = C.linear_equivalent()
    head = [f"Equivalent linear EOS (``cases.linear_equivalent``): α = {lin['t_coef']:.5f} kg m⁻³ °C⁻¹, "
            f"β = {lin['s_coef']:.5f} kg m⁻³ PSU⁻¹, θ₀ = {lin['t_ref']:.4f} °C, S₀ = {lin['s_ref']:.4f}, "
            f"ρ₀ = {lin['rho0']:.0f}; rms misfit to ρ_Wright(θ, S_P, 0) on the profile {lin['rms']:.4f} kg m⁻³.", ""]
    blocks = [("### Criterion 4: rest over steep topography (10 days)", rest_table()),
              ("### Criterion 5a: cabbeling lock exchange (10 days)", cabbeling_table()),
              ("### Criterion 5b: thermobaric column (10 days)", thermobaric_table()),
              ("### Criterion 10: N2b ERA5 column with the three EOS (45 days, run 'fms')", bulk_table())]
    lines = head
    for title, rows in blocks:
        lines += [title, ""] + rows + [""]
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--print", action="store_true", help="print the tables instead of writing the README")
    a = ap.parse_args()
    text = tables()
    if a.print:
        print(text)
        return
    src = README.read_text()
    i, j = src.index(START) + len(START), src.index(END)
    README.write_text(src[:i] + "\n" + text + "\n" + src[j:])
    print(f"tables written to {README}")


if __name__ == "__main__":
    main()
