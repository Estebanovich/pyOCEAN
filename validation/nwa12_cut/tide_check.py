#!/usr/bin/env python
"""
Tidal quick-look of NWA12-cut runs at the CO-OPS gauges (no score: criterion 4 is judged against MOM6 on the same cut,
Task 10/11).

For each run directory: least-squares harmonic analysis (``diagnostics.sealevel_qc.harmonic_analysis``, OTPS basis
with nodal corrections) of the per-step η series at each gauge's nearest wet cell (``series.npz``), after the first
``--skip`` days (start-up); constituents chosen so that every pair is separated by the record (Rayleigh R = 1).  The
observed constants are NOAA CO-OPS ``harcon_<id>.json`` (amplitude, Greenwich phase lag), converted to the OTPS
complex convention h = A e^{−iG} (``diagnostics.tide_gauges.complex_from_amp_phase``).  Prints amplitude, phase and
the M2 vector error |h_mod − h_obs| per gauge and its mean and median.

``--boundary`` also prints the amplitude of the boundary tidal velocity (Σ_c |ū_c| of the 10 constituents and |ū_M2|)
on the open faces for both ``BoundaryTides.velocity_depth`` choices ('model' = U/D_model, ruling R3; 'tpxo' =
U/h_TPXO).

Usage: python validation/nwa12_cut/tide_check.py RUN_DIR [RUN_DIR …] [--skip 1] [--boundary]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from diagnostics.sealevel_qc import harmonic_analysis  # noqa: E402
from diagnostics.tide_gauges import complex_from_amp_phase  # noqa: E402
from forcing.tides import OMEGA  # noqa: E402
from validation.nwa12_cut import case as cs  # noqa: E402

CANDIDATES = ("m2", "s2", "n2", "k1", "o1", "q1", "k2", "p1")


def separable(duration_s: float, rayleigh: float = 1.0) -> list[str]:
    """Candidates (in priority order) kept only if separated from every kept one by the record."""
    out: list[str] = []
    for c in CANDIDATES:
        if all(abs(OMEGA[c] - OMEGA[k]) * duration_s >= 2 * np.pi * rayleigh for k in out):
            out.append(c)
    return out


def gauge_m2(run: Path, skip_days: float = 1.0, case: cs.Case = cs.CASE) -> dict:
    """{station id: (h_obs, h_mod)} of M2 and the constituents used, for one run directory."""
    s = np.load(run / "series.npz")
    keep = s["t"] >= skip_days * 86400.0
    t = case.t0 + (s["t"][keep] * 1e9).astype("timedelta64[ns]")
    cons = separable(float(s["t"][keep][-1] - s["t"][keep][0]))
    out = {}
    for k, name in enumerate(map(str, s["names"])):
        if not name.startswith("coops_"):
            continue
        sid = name[6:]
        hc = harmonic_analysis(t, s["eta"][keep, k], cons)["hc"]
        hj = json.loads((Path(case.tides) / f"harcon_{sid}.json").read_text())["HarmonicConstituents"]
        m2 = [c for c in hj if c["name"] == "M2"][0]
        out[sid] = (complex_from_amp_phase(np.array(m2["amplitude"]), np.array(m2["phase_GMT"])), hc[0])
    return dict(stations=out, constituents=cons)


def phase(h: complex) -> float:
    """Greenwich phase lag [°] of an OTPS complex constant."""
    return float(np.rad2deg(-np.angle(h)) % 360.0)


def report(run: Path, skip_days: float) -> None:
    r = gauge_m2(run, skip_days)
    print(f"\n{run.name}: constituents {r['constituents']}, days ≥ {skip_days:g}")
    print(f"{'station':>8} {'A_obs':>6} {'A_mod':>6} {'G_obs':>6} {'G_mod':>6} {'ΔG':>6} {'|Δ|':>6}")
    err = []
    for sid, (o, m) in r["stations"].items():
        dg = (phase(m) - phase(o) + 180.0) % 360.0 - 180.0
        err.append(abs(m - o))
        print(f"{sid:>8} {abs(o):6.3f} {abs(m):6.3f} {phase(o):6.1f} {phase(m):6.1f} {dg:6.1f} {abs(m - o):6.3f}")
    print(f"M2 vector error: mean {np.mean(err):.3f} m, median {np.median(err):.3f} m, n = {len(err)}")


def boundary_velocity(case: cs.Case = cs.CASE) -> None:
    """Amplitude of the tidal velocity on the open faces for velocity_depth 'model' and 'tpxo'."""
    from core.components.boundary_tides import BoundaryTides
    hg, H = cs.grid(case)
    edges = cs.open_edges(H)
    Hn = np.nan_to_num(H)
    Hu = np.zeros((H.shape[0], H.shape[1] + 1))
    Hv = np.zeros((H.shape[0] + 1, H.shape[1]))
    Hu[:, 0], Hu[:, -1], Hv[0, :], Hv[-1, :] = Hn[:, 0], Hn[:, -1], Hn[0, :], Hn[-1, :]
    c = cs.tide_constants(case, hg)
    for vd in ("model", "tpxo"):
        bt = BoundaryTides(c, list(cs.CONSTITUENTS), case.t0, velocity_depth=vd).bind(hg, edges, Hu, Hv)
        amp_all, amp_m2 = [], []
        for e, (_, _, ok, _, hc_vel) in bt._faces.items():
            amp_all.append(np.abs(hc_vel[ok]).sum(axis=1))
            amp_m2.append(np.abs(hc_vel[ok][:, 0]))
        a, m = np.concatenate(amp_all), np.concatenate(amp_m2)
        print(f"velocity_depth={vd!r}: open faces {a.size}, Σ|ū_c| max {a.max():.3f} m/s (p99 "
              f"{np.percentile(a, 99):.3f}, median {np.median(a):.3f}); |ū_M2| max {m.max():.3f} m/s")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("runs", nargs="*", type=Path)
    ap.add_argument("--skip", type=float, default=1.0, help="days of start-up left out (default 1)")
    ap.add_argument("--boundary", action="store_true", help="print the boundary tidal-velocity amplitudes")
    a = ap.parse_args()
    for run in a.runs:
        report(run, a.skip)
    if a.boundary:
        boundary_velocity()


if __name__ == "__main__":
    main()
