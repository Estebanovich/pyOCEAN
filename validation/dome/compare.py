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
statistically steady state (the latter half for Fig. 4); here E, T, α_E and b_f use the window [t0, t1] (days 30–50
by default) and the tracer by class the latter half of the run (days 25–50 for 50 days), as Legg's Fig. 4 (this
supersedes the spec's "13-day window" wording).  Geometry differences (MOM6's 100-km channel, sponge west, quadratic
drag vs Legg's 50-km channel, Orlanski, no-slip) are class (c).

E is not resolution-independent: S (the area of the s = 0.01 surface) grows as the resolution resolves the plume
top (Legg's 8.6e-4 with T_out(−50 km) ≈ 18.7 Sv implies U_rms S ≈ 1.2e10 m³ s⁻¹, ~6× MITgcm at 10 km).  The
like-for-like references of a 10-km run are Legg's 10-km z (2.19e-4) and isopycnal (9.4e-4) values; the
non-hydrostatic 8.6e-4 is an order of magnitude only.  At 10 km E also moves by ~×2 with the definition (threshold
0.01/0.05/0.1, U_rms over the plume or all water) and ±20 % with the window (MITgcm legg: 0.55–1.96e-3), so E
differences below ~×2 are not significant.

``--theory`` (BBL drag-mixing plan, Task 6; spec 2026-10-30-bbl-drag-mixing goal A): a third table, the theoretical
diagnostic of validation/dome/theory.py for every run, as window means over x' ∈ (−180, −50) km (Legg's α_E window)
and (−500, −200) km: plume height h, ΔU, Δb, Ri_B, Fr, interfaces with gradient Ri < 0.25 across the plume interface
(Jackson et al. 2008 §6 asks for ≥ 2–4), median Ri_g, Ellison–Turner E(Ri_B) and Xu E(Ri_g) against the measured
α_E (eq. 12), the bottom Ekman excess transport over the plume's down-slope transport and u_plume/(g′s/f) (Nof
1983), with Legg's Table 2 numbers and each run's frictional Ekman number.

α_E and E (2026-10-30, read in the PDF): Legg's 8.6e-4 (p. 79) is the corner-mean E of eq. 14, (T_out − T_in)/(U_rms S);
the α_E column of the truth table is the mean over −180 < x' < −50 km of the blue curve of Fig. 3b, α_E(x) of eq. 12
(dT/dx)/(L Ū) from the time-mean transport, digitized every 10 km (1.03e-3); his red linear-fit segment of the same
equation, drawn over −145 … −50 km, gives 1.4–2.6e-3 (alphaE_fit_nonhydro).  Both are labelled with their equation.

Usage: python validation/dome/compare.py 10 [--t0 30 --t1 50] [--only sub1,sub2] [--theory]
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
    """Buoyancy bounds of the class with the most tracer at the column nearest x' [m s⁻²]; (NaN, NaN) without dye."""
    i = int(np.argmin(np.abs(x - (xp_km * 1e3 + 1050e3))))
    if not np.nansum(S[:, i]) > 0.0:
        return float("nan"), float("nan")
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


def _cls(b: tuple[float, float]) -> str:
    return "—" if not np.isfinite(b[0]) else f"{b[0] * 1e3:g}…{b[1] * 1e3:g}"


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
    L = ["| run | E esquina ec. 14 (pluma / todo) | T_out, T_in [Sv] | S [1e9 m²] | U_rms [m/s] | T(x' −200/−300) [Sv] | "
         "α_E ec. 12 (−180…−50 km) | b_f [1e-3 m s⁻²] | clase con más trazador x' −300 / −100 km [1e-3 m s⁻²] |",
         "|" + "---|" * 9,
         f"| Legg no hidrostático 0.5 km (verdad) | {v['corner_E_nonhydro']['value']:.2e} (texto, ec. 14; solo orden de magnitud) | — | — | — | "
         f"{-tx[-200] / 1e6:.1f} / {-tx[-300] / 1e6:.1f} (dig.) | {a_nh:.2e} (dig., Fig. 3b azul) | "
         f"{v['bf_nonhydro']['value'] * 1e3:.2f} (dig.) | {cls_nh[0] * 1e3:g}…{cls_nh[1] * 1e3:g} (dig.) |",
         f"| Legg 10 km z (MITgcm, ν_v 0.5) | {v['corner_E_10km_z']['value']:.2e} (dig.; referencia equivalente) | — | — | — | — | — | "
         f"{v['bf_10km_z']['value'] * 1e3:.2f} (dig.) | — |",
         f"| Legg 10 km isopicnal (HIM, ν_v 1e-4 + Ri) | {v['corner_E_10km_iso']['value']:.2e} (dig.; referencia equivalente) | — | — | — | — | — | "
         f"{v['bf_10km_iso']['value'] * 1e3:.2f} (dig.); ν_v alta {v['bf_10km_iso_visc']['value'] * 1e3:.2f} | — |"]
    for r in rows:
        ec = r["Ec"]
        L.append(f"| {r['name']} | {r['E']:.2e} / {r['E_all']:.2e} | {ec['T_out'] / 1e6:.2f}, {ec['T_in'] / 1e6:.2f} | "
                 f"{ec['S'] / 1e9:.2f} | {ec['U_rms']:.3f} | {r['T'][0]:.1f} / {r['T'][1]:.1f} | {r['aE_truth']:.2e} | "
                 f"{r['bf']:.2f} | {_cls(r['cls300'])} / {_cls(r['cls100'])} |")
    L += ["", "E: las referencias equivalentes a 10 km son las de Legg a 10 km (z 2.19e-4, isopicnal 9.4e-4); el 8.6e-4 "
          "no hidrostático (0.5 km) es solo un orden de magnitud, porque S crece con la resolución (U_rms S ≈ 1.2e10 "
          "m³/s en Legg, ~6× MITgcm a 10 km). A 10 km E cambia ~×2 con la definición (umbral 0.01–0.1, U_rms en la "
          "pluma o en toda el agua: las dos columnas) y ±20 % con la ventana: diferencias menores que ~×2 no son "
          "significativas. Ventanas: E, T, α_E y b_f en [t0, t1]; clases en la segunda mitad de la corrida (Legg Fig. 4).",
          "", "α_E y E no son el mismo número: el 8.6e-4 del texto de Legg (p. 79) es el E medio de la esquina, "
          "ec. 14; "
          "la columna α_E es la media en −180 < x' < −50 km de la curva azul de la Fig. 3b, α_E(x) de la ec. 12 "
          "(dT/dx)/(L Ū), digitalizada cada 10 km (1.03e-3). El segmento rojo de la Fig. 3b (ajuste lineal de T, "
          "la misma ec. 12, dibujado en −145…−50 km) da 1.4–2.6e-3."]
    return "\n".join(L)


def _f(v: float, fmt: str = ".2f") -> str:
    return "—" if v is None or not np.isfinite(v) else format(v, fmt)


def theory_table(items: list[tuple[str, dict]], t0: float, t1: float) -> str:
    """Theory table (``--theory``) of (name, theory.summary(...)) pairs: window means near / far."""
    from validation.dome import theory
    c = cases.get(10.0)
    L0 = theory.legg_numbers(c.nu_v, c.cd)
    L = [f"Legg (2006) Tabla 2, Caso 1 (de cases.py): Fr {L0['Fr']:.2f}, Δb₀/(N²h₀) {L0['db_ratio']:.1f}, "
         f"estiramiento L_ρ tanα/h₀ {L0['stretching']:.2f}, W/L_ρ {L0['width']:.2f}, N/f {L0['N_over_f']:.1f}, "
         f"Ekman de arrastre C_d Δb₀ tanα/(f²h₀) {L0['drag_ekman']:.3f} (Legg: 1, 12, 0.8, 4.1, 23.0, 0.127). "
         f"Ekman friccional ν_v/(h₀²f) = {L0['frictional_ekman']:.3f} con ν_v 0.5, "
         f"{theory.legg_numbers(cases.NU_V_LOW, c.cd)['frictional_ekman']:.1e} con 1e-4 "
         "(> 0.1: laminar, Cenedese 2004).",
         "",
         "Ventanas x' (−180…−50 km) / (−500…−200 km), días "
         f"{t0 / 86400:g}–{t1 / 86400:g}; columna sobre la trayectoria de cada registro.",
         "",
         "| run | ν_v / Ek_f | h [m] | ΔU [m/s] | Δb [1e-3 m s⁻²] | Ri_B | Fr | celdas Ri<0.25 / banda | "
         "Ri_g mediana | E_ET(Ri_B) | E_Xu(Ri_g) | α_E medido (ec. 12) | Ekman fondo / bajada pluma | "
         "u_pluma/(g′s/f) | fracción Ri<0.25 |",
         "|" + "---|" * 15]
    for name, r in items:
        n, f = r["near"], r["far"]
        nu = r["nu_v"]
        ek = theory.legg_numbers(nu, c.cd)["frictional_ekman"]
        nus = "GLS" if nu is None else f"{nu:g} / {ek:.1e}"

        def two(k, fmt=".2f", scale=1.0):
            return f"{_f(n[k] * scale, fmt)} / {_f(f[k] * scale, fmt)}"
        L.append(f"| {label(name)} | {nus} | {two('h', '.0f')} | {two('dU')} | {two('db', '.2f', 1e3)} | "
                 f"{two('RiB')} | {two('Fr')} | {_f(n['n_low'], '.1f')} / {_f(n['n_band'], '.1f')} ; "
                 f"{_f(f['n_low'], '.1f')} / {_f(f['n_band'], '.1f')} | {two('ri_g')} | {two('E_ET', '.1e')} | "
                 f"{two('E_Xu', '.1e')} | {two('aE', '.1e')} | {two('ek', '.2f')} | {two('nof')} | "
                 f"{_f(r['frac_low'], '.2f')} |")
    L += ["", "h: altura de la pluma adherida al fondo (s > 0.01); ΔU, Δb: media de la pluma contra los 100 m de agua "
          "justo encima; Ri_B = Δb h/ΔU², Fr = ΔU/√(Δb h) de las medias temporales; celdas: interfaces de la banda "
          "0.01 < s < 0.99 con Ri de gradiente < 0.25 (Jackson §6 pide ≥ 2–4); E_ET: Ellison–Turner (Xu ecs. 3–4), "
          "E_Xu: Xu ec. 16 con la mediana de Ri_g; Ekman: exceso talud abajo en los 10 m inferiores sobre el interior "
          "de la pluma, dividido por el transporte talud abajo de la pluma (Ezer: 22 %); Nof: u_t = g′s/f "
          "(Ezer ec. 2). La fracción Ri<0.25 es de todas las interfaces de la banda, juntando todas las x y todos "
          "los registros (no solo las ventanas). "
          "Ri_B depende mucho del umbral de la pluma: de s > 0.01 a s > 0.1 baja ×1.4–4.6 en −180…−50 km (MITgcm "
          "legg 9.1 → 6.5; híbrido kv_low + Smag + Jackson 6.5 → 1.4), porque la envoltura diluida entra en h. Con ΔU "
          "contra los 100 m de encima no es el Ri_B ≈ 0.4–0.5 de Legg, que no da su ΔU operativo [inferencia]: solo "
          "se comparan corridas entre sí. E_ET(Ri_B) = 0 con Ri_B ≥ 0.8 sale de esta definición, no del modelo; "
          "'—' = sin pluma. Las leyes de laboratorio no tienen rotación: Legg §3.3 (p. 85) da E = 8e-3–4e-2 con su "
          "Ri_B y dice que el entrainment rotante es \"considerablemente menor\"; contra el E de la esquina 8.6e-4 "
          "son 9–50× [inferencia]. Definiciones en validation/dome/theory.py."]
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("--t0", type=float, default=30.0)
    ap.add_argument("--t1", type=float, default=50.0)
    ap.add_argument("--only", default="", help="substring filter on run names (comma-separated: any of them)")
    ap.add_argument("--theory", action="store_true", help="add the theoretical plume diagnostic (theory.py)")
    a = ap.parse_args()
    rows, items = [], []
    for p in sorted((cases.RUNS / f"{a.res_km:g}km").glob("*/series.npz")):
        if a.only and not any(o and o in p.parent.name for o in a.only.split(",")):
            continue
        s, meta = series.load(p)
        if s["t"][-1] < a.t1 * 86400.0:            # short checks (Δt ladder, 2-day runs) are not in the window
            continue
        rows.append(row(p.parent.name, s, meta, a.t0 * 86400.0, a.t1 * 86400.0))
        if a.theory:
            from validation.dome import theory                 # summaries only: the series are not kept
            items.append((p.parent.name, theory.summary(s, meta, a.t0 * 86400.0, a.t1 * 86400.0)))
    leg = cases.LEGG_10KM
    txt = (f"# DOME {a.res_km:g} km, métricas promediadas en los días {a.t0:g}–{a.t1:g}\n\n" + table(rows) +
           f"\n\nLegg 2006, 10 km (Figs. 15–17): Y'(x' = {', '.join(f'{v:g}' for v in leg['x_km'])}) z "
           f"{', '.join(f'{v:g}' for v in leg['y_z_km'])}; iso {', '.join(f'{v:g}' for v in leg['y_iso_km'])}; "
           f"iso ν_v 0.5 {', '.join(f'{v:g}' for v in leg['y_iso_visc_km'])}; b_f z {leg['bf_z'] * 1e3:g}, "
           f"iso {leg['bf_iso'] * 1e3:g}, iso ν_v 0.5 {leg['bf_iso_visc'] * 1e3:g} (1e-3 m s⁻²)\n\n"
           f"## Contra la verdad no hidrostática de Legg (2006), Caso 1 (validation/dome/legg2006_nonhydro.json)\n\n"
           + truth_table(rows) + "\n")
    if a.theory:
        txt += ("\n## Diagnóstico teórico de la pluma (validation/dome/theory.py)\n\n"
                + theory_table(items, a.t0 * 86400.0, a.t1 * 86400.0) + "\n")
    if not a.only:                                   # a filtered table must not replace the full one on disk
        (cases.RUNS / f"compare_{a.res_km:g}km.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
