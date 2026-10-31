#!/usr/bin/env python
"""
Tables of the DOME benchmark (spec 2026-10-14-dome-overflow §7): every series.npz under cases.RUNS/<res>km/*/ with the
metrics of validation/dome/metrics.py; x' = x − 1050 km (inlet centre), y' = y − 600 km (slope top) as Legg et al. 2006.
Prints the Markdown and, without ``--only``, writes it to RUNS/compare_<res>km.txt.  'visc' is the Smagorinsky counter
``n_visc_limited`` (h and q points × steps where the stability bound cut ν; empty without Smagorinsky): with a
background ah > 0 every point where the bound × h_min/h falls below ν (thin layers) counts, so it is not the number
of points where C Δ² |S| itself reached the bound.  Runs with Smagorinsky end in '_smag<C>' (run_pyocean.py --smag C); '_smag1e-09' is
the operator control (C ≈ 0: the background ah in the thickness-weighted tensor operator, no Smagorinsky term) and
'_preI1' marks runs kept from before the clipped-face fix 47f2730 (final review I1); both are labelled in the table.

Against the truth (Jackson shear-mixing plan, Task 6; spec 2026-10-30 criterion 4): the corner-mean entrainment E
(Legg eq. 14, metrics.corner_entrainment, U_rms over the plume cells and over all water) and a second table against
Legg's non-hydrostatic Case 1 and his 10-km z and isopycnal runs (legg2006_nonhydro.json: text and digitized values
with their sources): E, plume transport T(x' = −200/−300 km), mean α_E over −180 < x' < −50 km, b_f, and the
buoyancy class holding the most dye at x' = −300 km (Fig. 4, metrics.tracer_by_class).  Legg averages over the
statistically steady state (the latter half for Fig. 4); here the window [t0, t1].  Geometry differences (MOM6's
100-km channel, sponge west, quadratic drag vs Legg's 50-km channel, Orlanski, no-slip) are class (c).

Usage: python validation/dome/compare.py 10 [--t0 30 --t1 50] [--only sub1,sub2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.dome import cases, metrics, series  # noqa: E402

XP = (-200.0, -300.0, -500.0, -800.0)             # x' [km] for the path and the transport
LEGG_JSON = Path(__file__).resolve().parent / "legg2006_nonhydro.json"
XA_TRUTH = (-180e3, -50e3)                        # x' window of α_E against Legg's Fig. 3b (his red fit)


def legg_truth() -> dict:
    """The values of legg2006_nonhydro.json (key → dict with value, source, method, …)."""
    return json.loads(LEGG_JSON.read_text())["values"]


def _xa_mean(x, f, lo, hi) -> float:
    m = (x - 1050e3 > lo) & (x - 1050e3 < hi)
    return float(np.nanmean(np.asarray(f, float)[m]))


def max_class(x, edges, S, xp_km: float) -> tuple[float, float]:
    """Buoyancy bounds of the class with the most tracer at the column nearest x' [m s⁻²]."""
    i = int(np.argmin(np.abs(x - (xp_km * 1e3 + 1050e3))))
    c = int(np.argmax(S[:, i]))
    return float(edges[c]), float(edges[c + 1])


def _at(x, f, xp_km):
    return [f[np.argmin(np.abs(x - (xx * 1e3 + 1050e3)))] for xx in xp_km]


def label(name: str) -> str:
    """Run name for the table, with the operator control and the pre-I1 runs marked."""
    if name.endswith("_preI1"):
        return f"{name} (antes de la corrección I1)"
    if name.endswith("_smag1e-09"):
        return f"{name} (control de operador, C≈0)"
    return name


def row(name: str, s: dict, meta: dict, t0: float, t1: float) -> dict:
    x, Y = metrics.path_Y(s, t0, t1)
    _, T = metrics.plume_transport(s, t0, t1)
    _, aE = metrics.entrainment(s, t0, t1)
    xa = (x > 750e3) & (x < 1050e3)
    rho = metrics.final_density(s, t0, t1)
    rpe = metrics.rpe_series(s)
    thin, thick = metrics.thin_vs_thick(s)
    r = metrics.ranges(s)
    days = s["t"][-1] / 86400.0
    ec = metrics.corner_entrainment(s, t0, t1, parts=True)
    xb, eb, Sb = metrics.tracer_by_class(s, 0.5 * s["t"][-1], s["t"][-1])     # latter half of the run (Legg Fig. 4)
    return dict(E=ec["E"], E_all=metrics.corner_entrainment(s, t0, t1, urms="all"), Ec=ec,
                aE_truth=_xa_mean(x, aE, *XA_TRUTH), cls300=max_class(xb, eb, Sb, -300.0),
                cls100=max_class(xb, eb, Sb, -100.0), name=label(name), Y=[(y - 600e3) / 1e3 for y in _at(x, Y, XP)], T=[q / 1e6 for q in _at(x, T, XP)],
                aE=float(np.nanmean(aE[xa])), rho=rho, bf=-cases.G * (rho - cases.RHO_TOP) / cases.RHO0 * 1e3,
                cd=[metrics.centroid_depth(s, d * 86400.0) for d in (10, 20, 30, days)],
                drpe=(rpe[-1] - rpe[0]) / rpe[0], T_rng=r["T"], s_rng=r["s"], thin=thin, thick=thick,
                trunc=meta.get("n_trunc", ""), lim=meta.get("n_limited", ""), vlim=meta.get("n_visc_limited", ""),
                dt=meta.get("dt", np.nan),
                cost=meta.get("wall_s", np.nan) / days, days=days)


def table(rows: list[dict]) -> str:
    L = ["| run | días | Y'(x' = −200/−300/−500/−800 km) [km] | T(x') [Sv] | α_E (x' 0…−300) | ρ_f / b_f [1e-3 m s⁻²] | "
         "centroide 10/20/30/fin [m] | ΔRPE/RPE₀ | T rango | s rango | |u| delgadas/gruesas | trunc / lim / visc | "
         "Δt [s] | "
         "s por día | E esquina (U_rms pluma / todo) |", "|" + "---|" * 15]
    for r in rows:
        L.append(f"| {r['name']} | {r['days']:.0f} | {' / '.join(f'{v:.0f}' for v in r['Y'])} | "
                 f"{' / '.join(f'{v:.1f}' for v in r['T'])} | {r['aE']:.2e} | {r['rho']:.3f} / {r['bf']:.2f} | "
                 f"{' / '.join(f'{v:.0f}' for v in r['cd'])} | {r['drpe']:.2e} | "
                 f"{r['T_rng'][0]:.3f}–{r['T_rng'][1]:.3f} | {r['s_rng'][0]:.1e}–{r['s_rng'][1]:.6f} | "
                 f"{r['thin']:.2f} / {r['thick']:.2f} | {r['trunc']} / {r['lim']} / {r['vlim']} | {r['dt']:.0f} | "
                 f"{r['cost']:.1f} | {r['E']:.2e} / {r['E_all']:.2e} |")
    return "\n".join(L)


def truth_table(rows: list[dict]) -> str:
    """Each run against Legg's non-hydrostatic Case 1 (text/digitized values of legg2006_nonhydro.json)."""
    v = legg_truth()
    T = v["transport_nonhydro"]
    tx = dict(zip(T["x_km"], T["value"]))
    aE = v["alphaE_nonhydro"]
    a_nh = np.nanmean([q for xx, q in zip(aE["x_km"], aE["value"]) if q is not None and -180 < xx < -50])
    tc = v["tracer_by_class_nonhydro"]
    i300 = tc["x_km"].index(-300.0) if -300.0 in tc["x_km"] else int(np.argmin(np.abs(np.array(tc["x_km"]) + 300)))
    col = [r[i300] or 0.0 for r in tc["value"]]
    cls_nh = tc["b_class_bounds"][int(np.argmax(col))]
    L = ["| run | E esquina (pluma / todo) | T_out, T_in [Sv] | S [1e9 m²] | U_rms [m/s] | T(x' −200/−300) [Sv] | "
         "α_E (−180…−50 km) | b_f [1e-3 m s⁻²] | clase con más trazador x' −300 / −100 km [1e-3 m s⁻²] |",
         "|" + "---|" * 9,
         f"| Legg no hidrostático 0.5 km (verdad) | {v['corner_E_nonhydro']['value']:.2e} (texto) | — | — | — | "
         f"{-tx[-200] / 1e6:.1f} / {-tx[-300] / 1e6:.1f} (dig.) | {a_nh:.2e} (dig.) | "
         f"{v['bf_nonhydro']['value'] * 1e3:.2f} (dig.) | {cls_nh[0] * 1e3:g}…{cls_nh[1] * 1e3:g} (dig.) |",
         f"| Legg 10 km z (MITgcm, ν_v 0.5) | {v['corner_E_10km_z']['value']:.2e} (dig.) | — | — | — | — | — | "
         f"{v['bf_10km_z']['value'] * 1e3:.2f} (dig.) | — |",
         f"| Legg 10 km isopicnal (HIM, ν_v 1e-4 + Ri) | {v['corner_E_10km_iso']['value']:.2e} (dig.) | — | — | — | — | — | "
         f"{v['bf_10km_iso']['value'] * 1e3:.2f} (dig.); ν_v alta {v['bf_10km_iso_visc']['value'] * 1e3:.2f} | — |"]
    for r in rows:
        ec = r["Ec"]
        L.append(f"| {r['name']} | {r['E']:.2e} / {r['E_all']:.2e} | {ec['T_out'] / 1e6:.2f}, {ec['T_in'] / 1e6:.2f} | "
                 f"{ec['S'] / 1e9:.2f} | {ec['U_rms']:.3f} | {r['T'][0]:.1f} / {r['T'][1]:.1f} | {r['aE_truth']:.2e} | "
                 f"{r['bf']:.2f} | {r['cls300'][0] * 1e3:g}…{r['cls300'][1] * 1e3:g} / "
                 f"{r['cls100'][0] * 1e3:g}…{r['cls100'][1] * 1e3:g} |")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("--t0", type=float, default=30.0)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--only", default="", help="substring filter on run names (comma-separated: any of them)")
    a = ap.parse_args()
    rows = []
    for p in sorted((cases.RUNS / f"{a.res_km:g}km").glob("*/series.npz")):
        if a.only and not any(o and o in p.parent.name for o in a.only.split(",")):
            continue
        s, meta = series.load(p)
        if s["t"][-1] < a.t1 * 86400.0:            # short checks (Δt ladder, 2-day runs) are not in the window
            continue
        rows.append(row(p.parent.name, s, meta, a.t0 * 86400.0, a.t1 * 86400.0))
    leg = cases.LEGG_10KM
    txt = (f"# DOME {a.res_km:g} km, métricas promediadas en los días {a.t0:g}–{a.t1:g}\n\n" + table(rows) +
           f"\n\nLegg 2006, 10 km (Figs. 15–17): Y'(x' = {', '.join(f'{v:g}' for v in leg['x_km'])}) z "
           f"{', '.join(f'{v:g}' for v in leg['y_z_km'])}; iso {', '.join(f'{v:g}' for v in leg['y_iso_km'])}; "
           f"iso ν_v 0.5 {', '.join(f'{v:g}' for v in leg['y_iso_visc_km'])}; b_f z {leg['bf_z'] * 1e3:g}, "
           f"iso {leg['bf_iso'] * 1e3:g}, iso ν_v 0.5 {leg['bf_iso_visc'] * 1e3:g} (1e-3 m s⁻²)\n\n"
           f"## Contra la verdad no hidrostática de Legg (2006), Caso 1 (validation/dome/legg2006_nonhydro.json)\n\n"
           + truth_table(rows) + "\n")
    if not a.only:                                   # a filtered table must not replace the full one on disk
        (cases.RUNS / f"compare_{a.res_km:g}km.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
