#!/usr/bin/env python
"""
Figures of the DOME benchmark (spec 2026-10-14-dome-overflow §8), written to cases.RUNS/figs/ (never to the repo):
the bottom dye of every 50-day run at days 10, 20, 30 and 50, and the time-mean path Y'(x') and plume transport T(x')
(days 30–50) against Legg et al. (2006) at 10 km; dome_<res>km_bottom_dye.mp4/.gif, the bottom dye of every 50-day
run day by day; dome_<res>km_legg_truth.png and dome_<res>km_tracer_by_class.png (Jackson shear-mixing plan, Task 6),
the plume transport T(x') and α_E(x') against Legg's non-hydrostatic Case 1 (Fig. 3, digitized,
legg2006_nonhydro.json) and the tracer by buoyancy class (Fig. 4, latter half of the run) of each run beside Legg's.

dome_<res>km_legg_scorecard.png (Task 8): b_f, corner E, T(x' = −200 km), α_E and the mean dye buoyancy at
x' = −300 km of each run against Legg's truth; ``--before-after base,new[,ref]``: dome_<res>km_before_after.png (bottom
dye of the baseline and the new run, path, transport and tracer by class against Legg) and the bottom-dye animation
of those runs only.

Usage: python validation/dome/figures.py 10 [--no-anim] [--runs name1,name2,... --sub dir]
       python validation/dome/figures.py 10 --before-after base,new[,ref] [--sub dir] [--no-anim]
  --runs keeps only those run directories, in that order (e.g. the Smagorinsky set), and --sub writes to
  RUNS/figs/<dir>/.
"""
from __future__ import annotations

import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "manual" / "figures"))

from validation.dome import cases, metrics, series  # noqa: E402

DAYS = (10, 20, 30, 50)
LABELS = {"mitgcm_z_legg": "MITgcm z (Legg physics)", "mitgcm_z_common": "MITgcm z (common physics)",
          "mom6_rho_common": "MOM6 ρ", "pyocean_zstar": "pyOCEAN z*", "pyocean_sigma": "pyOCEAN σ",
          "pyocean_hybrid": "pyOCEAN hybrid", "pyocean_hybrid_gls": "pyOCEAN hybrid + GLS",
          "pyocean_hybrid_gls_smag0.15": "pyOCEAN hybrid + GLS + Smag 0.15",
          "pyocean_hybrid_gls_smag0.5": "pyOCEAN hybrid + GLS + Smag 0.5",
          "pyocean_hybrid_gls_smag1e-09": "pyOCEAN hybrid + GLS, operator control C≈0",
          "pyocean_hybrid_kv_low_smag0.15": "pyOCEAN hybrid ν_v 1e-4 + Smag 0.15",
          "pyocean_hybrid_kv_low_smag0.15_jackson": "pyOCEAN hybrid ν_v 1e-4 + Smag 0.15 + Jackson",
          "pyocean_hybrid_kv_low_smag0.15_jacksonpr0": "pyOCEAN hybrid ν_v 1e-4 + Smag 0.15 + Jackson Pr 0",
          "pyocean_hybrid_jackson": "pyOCEAN hybrid + Jackson",
          "pyocean_zstar_kv_low_smag0.15": "pyOCEAN z* ν_v 1e-4 + Smag 0.15",
          "pyocean_zstar_kv_low_smag0.15_jackson": "pyOCEAN z* ν_v 1e-4 + Smag 0.15 + Jackson",
          "pyocean_sigma_kv_low_smag0.15": "pyOCEAN σ ν_v 1e-4 + Smag 0.15",
          "pyocean_sigma_kv_low_smag0.15_jackson": "pyOCEAN σ ν_v 1e-4 + Smag 0.15 + Jackson"}


def bottom_dye(s: dict, n: int) -> np.ndarray:
    """s in the deepest layer thicker than 1 m of each column at record n (NaN on land)."""
    h = np.nan_to_num(s["h"][n])
    k = np.where(h > 1.0, np.arange(h.shape[0])[:, None, None], -1).max(axis=0)
    out = np.take_along_axis(np.nan_to_num(s["s"][n]), np.maximum(k, 0)[None], axis=0)[0]
    return np.where((k >= 0) & np.isfinite(s["H"]), out, np.nan)


def anim_bottom_dye(res_km: float, data: list, out: Path | None = None, stem: str | None = None) -> None:
    """Bottom dye of every 50-day run, one frame per daily record, rows of 3 panels; ``stem`` names the files
    (default dome_<res>km_bottom_dye)."""
    import pyocean_style as ps

    ps.use()
    ncol = 3
    nrow = int(np.ceil(len(data) / ncol))
    fig, axs = plt.subplots(nrow, ncol, figsize=(11.0, 2.1 * nrow + 0.5), sharex=True, sharey=True,
                            squeeze=False, constrained_layout=True)
    meshes = []
    for ax, (name, s) in zip(axs.flat, data):
        ax.set_facecolor(ps.LAND)
        ax.grid(False)
        meshes.append(ax.pcolormesh(s["x"] / 1e3, s["y"] / 1e3, np.ma.masked_invalid(bottom_dye(s, 0)), vmin=0,
                                    vmax=1, cmap=ps.CMAPS["fraction"].reversed(), rasterized=True))
        ax.contour(s["x"] / 1e3, s["y"] / 1e3, np.nan_to_num(s["H"]), [1000, 2000, 3000], colors=ps.MUTED,
                   linewidths=0.5)
        ax.set_aspect("equal")
        ax.set_title(LABELS.get(name, name), fontsize=9)
    for ax in axs.flat[len(data):]:
        ax.set_visible(False)
    for ax in axs[-1]:
        ax.set_xlabel("x [km]")
    for ax in axs[:, 0]:
        ax.set_ylabel("y [km]")
    ps.colorbar(fig, meshes[0], axs, "bottom dye s", shrink=0.7)
    t = data[0][1]["t"]
    head = f"DOME {res_km:g} km (Legg et al. 2006, case 1): bottom dye, isobaths 1000/2000/3000 m"
    title = fig.suptitle(f"{head} — day {t[0] / 86400:4.1f}", x=0.02, ha="left", fontsize=11)
    ps.freeze_layout(fig)

    def update(n: int):
        title.set_text(f"{head} — day {t[n] / 86400:4.1f}")
        for m, (_, s) in zip(meshes, data):
            m.set_array(np.ma.masked_invalid(bottom_dye(s, min(n, s["t"].size - 1))))

    ps.save_animation(fig, update, t.size, (out or cases.RUNS / "figs") / (stem or f"dome_{res_km:g}km_bottom_dye"),
                      fps=6)


def legg_truth_figures(data: list, out: Path, res_km: float, t0: float = 30 * 86400.0, t1: float = 50 * 86400.0):
    """T(x') and α_E(x') of each run against Legg's non-hydrostatic Case 1 (Fig. 3a, 3b digitized), and the tracer by
    buoyancy class (Fig. 4) of each run, latter half of the run, beside Legg's (legg2006_nonhydro.json)."""
    from validation.dome.compare import legg_truth
    v = legg_truth()
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[: max(len(data), 1)] if len(data) > 10 else [None] * len(data)
    for (name, s), col in zip(data, cols):
        x, T = metrics.plume_transport(s, t0, t1)
        _, aE = metrics.entrainment(s, t0, t1)
        ls = "--" if "jackson" in name else "-"
        a1.plot((x - 1050e3) / 1e3, T / 1e6, ls, color=col, label=LABELS.get(name, name))
        a2.plot((x - 1050e3) / 1e3, aE, ls, color=col, label=LABELS.get(name, name))
    tr = v["transport_nonhydro"]
    a1.plot(tr["x_km"], -np.array(tr["value"]) / 1e6, "k", lw=2.5, label="Legg no hidrostático (dig.)")
    ae = v["alphaE_nonhydro"]
    a2.plot(ae["x_km"], [np.nan if q is None else q for q in ae["value"]], "k", lw=2, label="Legg no hidrostático (dig.)")
    a1.set(xlabel="x' [km]", ylabel="T [Sv] (hacia el oeste)", xlim=(-700, 100), title="Transporte de la pluma (s > 0.01)")
    a2.set(xlabel="x' [km]", ylabel="α_E", xlim=(-700, 0), ylim=(-6e-3, 4e-3), title="α_E(x') (Legg ec. 12)")
    if len(data) > 6:                                                # many runs: legend below the panels
        h, lab = a1.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.set_size_inches(11, 5.4)
        fig.tight_layout(rect=(0, 0.2, 1, 1))
    else:
        a1.legend(fontsize=6)
        fig.tight_layout()
    fig.savefig(out / f"dome_{res_km:g}km_legg_truth.png", dpi=110)
    plt.close(fig)

    tc = v["tracer_by_class_nonhydro"]
    n = len(data) + 1
    fig, ax = plt.subplots(n, 1, figsize=(6, 2.0 * n), sharex=True, squeeze=False)
    lb = np.array(tc["b_class_bounds"])
    M = np.array([[np.nan if q is None else q for q in r] for r in tc["value"]])
    xe = np.concatenate([np.array(tc["x_km"]) - 12.5, [tc["x_km"][-1] + 12.5]])
    be = np.concatenate([lb[:, 1], [lb[-1, 0]]])
    im = ax[0, 0].pcolormesh(xe, be * 1e3, M, cmap="jet", vmin=0)
    fig.colorbar(im, ax=ax[0, 0], label="×10⁶ m²")
    ax[0, 0].set_title("Legg no hidrostático, Fig. 4 (dig.)", fontsize=8)
    for k, (name, s) in enumerate(data, start=1):
        x, edges, S = metrics.tracer_by_class(s, 0.5 * s["t"][-1], s["t"][-1])
        im = ax[k, 0].pcolormesh((np.concatenate([x - 0.5 * s["dx"], [x[-1] + 0.5 * s["dx"][-1]]]) - 1050e3) / 1e3,
                                 edges * 1e3, S / 1e6, cmap="jet", vmin=0)
        fig.colorbar(im, ax=ax[k, 0], label="×10⁶ m²")
        ax[k, 0].set_title(LABELS.get(name, name), fontsize=8)
    for a in ax[:, 0]:
        a.set(ylabel="b [1e-3 m s⁻²]", xlim=(-700, 300), ylim=(-20, 0))
    ax[-1, 0].set_xlabel("x' [km]")
    fig.tight_layout()
    fig.savefig(out / f"dome_{res_km:g}km_tracer_by_class.png", dpi=110)
    plt.close(fig)


def class_profile(s: dict, xp_km: float, t0: float | None = None, t1: float | None = None):
    """Share of the dye of the column nearest x' = xp_km in each buoyancy class (Legg Fig. 4 bins), mean over
    [t0, t1] (default: the latter half of the run): (class centres [m s⁻²], fractions summing to 1; NaN without dye)."""
    t1 = s["t"][-1] if t1 is None else t1
    t0 = 0.5 * s["t"][-1] if t0 is None else t0
    x, edges, S = metrics.tracer_by_class(s, t0, t1)
    i = int(np.argmin(np.abs(x - (xp_km * 1e3 + metrics.X_INLET))))
    col = S[:, i]
    tot = col.sum()
    return 0.5 * (edges[1:] + edges[:-1]), (col / tot if tot > 0 else np.full(col.shape, np.nan))


def legg_class_profile(xp_km: float):
    """Legg's digitized non-hydrostatic Fig. 4 at the column nearest x' = xp_km: (class centres, fractions)."""
    from validation.dome.compare import legg_truth
    tc = legg_truth()["tracer_by_class_nonhydro"]
    i = int(np.argmin(np.abs(np.array(tc["x_km"]) - xp_km)))
    col = np.array([r[i] or 0.0 for r in tc["value"]])
    b = np.array(tc["b_class_bounds"]).mean(axis=1)
    return b, col / col.sum()


def before_after(data: list, out: Path, res_km: float, t0: float = 30 * 86400.0, t1: float = 50 * 86400.0,
                 xp_km: float = -300.0) -> Path:
    """
    One run against its baseline (Jackson shear-mixing plan, Task 8): ``data`` = [(name, series), …] with the
    baseline first, the new run second and optional references after.  Top rows: bottom dye of the baseline and of the
    new run at days 10, 20, 30 and 50.  Bottom row: mean path Y'(x') over [t0, t1] with Legg's 10-km points
    (cases.LEGG_10KM), plume transport T(x') with Legg's non-hydrostatic curve (Fig. 3a, digitized), and the share of
    the dye by buoyancy class at x' = xp_km (latter half of the run) with Legg's non-hydrostatic Fig. 4 (digitized).
    Writes dome_<res>km_before_after.png and returns its path.
    """
    from validation.dome.compare import legg_truth
    v = legg_truth()
    fig = plt.figure(figsize=(13, 8.2))
    gs = fig.add_gridspec(3, len(DAYS) + 1, height_ratios=[1, 1, 1.9], width_ratios=[1] * len(DAYS) + [0.04],
                          hspace=0.5, wspace=0.08)
    im = None
    for r, (name, s) in enumerate(data[:2]):
        for c, d in enumerate(DAYS):
            ax = fig.add_subplot(gs[r, c])
            n = int(np.argmin(np.abs(s["t"] - d * 86400.0)))
            im = ax.pcolormesh(s["x"] / 1e3, s["y"] / 1e3, bottom_dye(s, n), vmin=0, vmax=1, cmap="viridis",
                               rasterized=True)
            ax.contour(s["x"] / 1e3, s["y"] / 1e3, np.nan_to_num(s["H"]), [1000, 2000, 3000], colors="w",
                       linewidths=0.4)
            ax.set_title(f"día {d}", fontsize=8, loc="left")
            if c == 0:
                ax.text(0.0, 1.22, f"({'ab'[r]}) {LABELS.get(name, name)}", transform=ax.transAxes, fontsize=9.5,
                        fontweight="bold")
            if c == 0:
                ax.set_ylabel("y [km]", fontsize=8)
            else:
                ax.set_yticklabels([])
            ax.tick_params(labelsize=7)
        fig.colorbar(im, cax=fig.add_subplot(gs[r, -1]), label="s de fondo")
    bot = gs[2, :].subgridspec(1, 3, wspace=0.3)
    a1, a2, a3 = (fig.add_subplot(bot[0, k]) for k in range(3))
    lb, lc = legg_class_profile(xp_km)
    for name, s in data:
        lab = LABELS.get(name, name)
        x, Y = metrics.path_Y(s, t0, t1)
        _, T = metrics.plume_transport(s, t0, t1)
        line, = a1.plot((x - metrics.X_INLET) / 1e3, (Y - metrics.Y_TOP) / 1e3, label=lab)
        a2.plot((x - metrics.X_INLET) / 1e3, T / 1e6, color=line.get_color())
        b, f = class_profile(s, xp_km)
        a3.plot(f, b * 1e3, "-o", ms=3, color=line.get_color())
    leg = cases.LEGG_10KM
    for key, mk, lab in (("y_z_km", "s", "Legg 10 km z"), ("y_iso_km", "o", "Legg 10 km isopicnal")):
        a1.plot(leg["x_km"], leg[key], "k" + mk, ls="none", mfc="none", label=lab)
    tr = v["transport_nonhydro"]
    a2.plot(tr["x_km"], -np.array(tr["value"]) / 1e6, "k", lw=2.2, label="Legg no hidrostático 0.5 km (dig.)")
    a3.plot(lc, lb * 1e3, "k-", lw=2.2)
    a1.set(xlabel="x' [km]", ylabel="Y' [km]", xlim=(-1000, 50), title=f"Trayectoria media, días {t0 / 86400:.0f}–{t1 / 86400:.0f}")
    a2.set(xlabel="x' [km]", ylabel="T [Sv] (hacia el oeste)", xlim=(-700, 50), title="Transporte de la pluma (s > 0.01)")
    a3.set(xlabel="fracción del trazador (segunda mitad)", ylabel="b [1e-3 m s⁻²]", ylim=(-20, 0),
           title=f"Trazador por clase en x' = {xp_km:.0f} km")
    for a in (a1, a2, a3):
        a.tick_params(labelsize=7)
        a.title.set_fontsize(9)
    h1, l1 = a1.get_legend_handles_labels()
    h2, l2 = a2.get_legend_handles_labels()
    fig.legend(h1 + h2, l1 + l2, loc="lower center", ncol=3, fontsize=7.5, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.suptitle(f"DOME {res_km:g} km: {LABELS.get(data[1][0], data[1][0])} frente a su línea base", x=0.02,
                 ha="left", fontsize=11)
    path = out / f"dome_{res_km:g}km_before_after.png"
    fig.savefig(path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    return path


def mean_class_b(s: dict, xp_km: float) -> float:
    """Dye-weighted mean buoyancy of the column nearest x' = xp_km over the latter half of the run [m s⁻²]
    (the first moment of ``class_profile``)."""
    b, f = class_profile(s, xp_km)
    return float(np.sum(b * f))


def legg_scorecard(data: list, out: Path, res_km: float, t0: float = 30 * 86400.0, t1: float = 50 * 86400.0) -> Path:
    """
    Each run against Legg's truth, one horizontal bar per run and metric (Jackson shear-mixing plan, Task 8): b_f
    (Fig. 15), corner E (eq. 14, U_rms over the plume; Fig. 14), T(x' = −200 km) (Fig. 3a), mean α_E over
    −180 < x' < −50 km (Fig. 3b) and the dye-weighted mean buoyancy at x' = −300 km (Fig. 4, latter half).  Vertical
    lines: Legg's non-hydrostatic value (solid; digitized except E, from the text) and his 10-km z (dotted) and
    isopycnal (dashed) runs where he gives them.  Writes dome_<res>km_legg_scorecard.png and returns its path.
    """
    from validation.dome.compare import XA_TRUTH, _xa_mean, legg_truth
    v = legg_truth()
    names = [LABELS.get(n, n) for n, _ in data]
    vals = {k: [] for k in ("bf", "E", "T200", "aE", "b300")}
    for _, s in data:
        rho = metrics.final_density(s, t0, t1)
        vals["bf"].append(-cases.G * (rho - cases.RHO_TOP) / cases.RHO0 * 1e3)
        vals["E"].append(metrics.corner_entrainment(s, t0, t1) * 1e4)
        x, T = metrics.plume_transport(s, t0, t1)
        vals["T200"].append(T[int(np.argmin(np.abs(x - metrics.X_INLET + 200e3)))] / 1e6)
        _, aE = metrics.entrainment(s, t0, t1)
        vals["aE"].append(_xa_mean(x, aE, *XA_TRUTH) * 1e3)
        vals["b300"].append(mean_class_b(s, -300.0) * 1e3)
    tr = v["transport_nonhydro"]
    ae = v["alphaE_nonhydro"]
    lb, lc = legg_class_profile(-300.0)
    truth = {"bf": (v["bf_nonhydro"]["value"] * 1e3, v["bf_10km_z"]["value"] * 1e3, v["bf_10km_iso"]["value"] * 1e3),
             "E": (v["corner_E_nonhydro"]["value"] * 1e4, v["corner_E_10km_z"]["value"] * 1e4,
                   v["corner_E_10km_iso"]["value"] * 1e4),
             "T200": (-dict(zip(tr["x_km"], tr["value"]))[-200.0] / 1e6, None, None),
             "aE": (np.nanmean([q for xx, q in zip(ae["x_km"], ae["value"]) if q is not None and -180 < xx < -50]) * 1e3,
                    None, None),
             "b300": (float(np.sum(lb * lc)) * 1e3, None, None)}
    titles = {"bf": "b_f [1e-3 m s⁻²]\n(Fig. 15)", "E": "E esquina [1e-4]\n(ec. 14, Fig. 14)",
              "T200": "T(x' = −200 km) [Sv]\n(Fig. 3a)", "aE": "α_E medio, −180 < x' < −50 km\n[1e-3] (Fig. 3b)",
              "b300": "b medio del trazador en x' = −300 km\n[1e-3 m s⁻²] (Fig. 4)"}
    fig, axs = plt.subplots(1, 5, figsize=(16, 0.42 * len(data) + 2.2), sharey=True)
    yy = np.arange(len(data))[::-1]
    colors = ["#f58518" if "jackson" in n else "#4c78a8" for n, _ in data]
    for ax, k in zip(axs, vals):
        ax.barh(yy, vals[k], color=colors, height=0.65)
        nh, z10, iso10 = truth[k]
        ax.axvline(nh, color="k", lw=1.8, label="Legg no hidrostático")
        if z10 is not None:
            ax.axvline(z10, color="k", ls=":", lw=1.2, label="Legg 10 km z")
            ax.axvline(iso10, color="k", ls="--", lw=1.2, label="Legg 10 km isopicnal")
        ax.axvline(0.0, color="0.6", lw=0.6)
        ax.set_title(titles[k], fontsize=8)
        ax.tick_params(labelsize=7)
    axs[0].set_yticks(yy)
    axs[0].set_yticklabels(names, fontsize=7)
    from matplotlib.patches import Patch
    h, lab = axs[1].get_legend_handles_labels()
    h += [Patch(color="#4c78a8"), Patch(color="#f58518")]
    lab += ["sin Jackson", "con Jackson"]
    fig.legend(h, lab, loc="lower center", ncol=5, fontsize=8, frameon=False)
    fig.suptitle(f"DOME {res_km:g} km contra Legg et al. (2006), Caso 1 (días {t0 / 86400:.0f}–{t1 / 86400:.0f}; "
                 "verdad no hidrostática digitalizada salvo E, del texto)", x=0.01, ha="left", fontsize=10)
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    path = out / f"dome_{res_km:g}km_legg_scorecard.png"
    fig.savefig(path, dpi=110)
    plt.close(fig)
    return path


def main(res_km: float, anim: bool = True, only: list[str] | None = None, sub: str = "") -> None:
    runs = [p for p in sorted((cases.RUNS / f"{res_km:g}km").glob("*/series.npz"))
            if not only or p.parent.name in only]
    if only:                                                         # the order given on the command line
        runs.sort(key=lambda p: only.index(p.parent.name))
    data = []
    for p in runs:
        s, _ = series.load(p)
        if s["t"][-1] >= 50 * 86400.0:
            data.append((p.parent.name, s))
    out = cases.RUNS / "figs" / sub if sub else cases.RUNS / "figs"
    out.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(len(data), len(DAYS), figsize=(3.2 * len(DAYS), 1.7 * len(data)), squeeze=False,
                           sharex=True, sharey=True)
    for r, (name, s) in enumerate(data):
        for c, d in enumerate(DAYS):
            n = int(np.argmin(np.abs(s["t"] - d * 86400.0)))
            im = ax[r, c].pcolormesh(s["x"] / 1e3, s["y"] / 1e3, bottom_dye(s, n), vmin=0, vmax=1, cmap="viridis")
            ax[r, c].contour(s["x"] / 1e3, s["y"] / 1e3, np.nan_to_num(s["H"]), [1000, 2000, 3000], colors="w",
                             linewidths=0.4)
            if c == 0:                       # run label once per row, left-aligned; the day inside each panel
                ax[r, c].set_title(LABELS.get(name, name), fontsize=7, loc="left")
            ax[r, c].text(0.03, 0.05, f"día {d}", transform=ax[r, c].transAxes, color="w", fontsize=7)
    fig.colorbar(im, ax=ax, shrink=0.6, label="s de fondo")
    fig.savefig(out / f"dome_{res_km:g}km_bottom_dye.png", dpi=110)
    plt.close(fig)

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4))
    cols = plt.get_cmap("tab20")(np.linspace(0, 1, 20))[: len(data)] if len(data) > 10 else [None] * len(data)
    for (name, s), col in zip(data, cols):
        x, Y = metrics.path_Y(s, 30 * 86400.0, 50 * 86400.0)
        _, T = metrics.plume_transport(s, 30 * 86400.0, 50 * 86400.0)
        ls = "--" if "jackson" in name else "-"
        a1.plot((x - 1050e3) / 1e3, (Y - 600e3) / 1e3, ls, color=col, label=LABELS.get(name, name))
        a2.plot((x - 1050e3) / 1e3, T / 1e6, ls, color=col, label=LABELS.get(name, name))
    if res_km == 10:
        leg = cases.LEGG_10KM
        for key, mk in (("y_z_km", "s"), ("y_iso_km", "o"), ("y_iso_visc_km", "^")):
            a1.plot(leg["x_km"], leg[key], "k" + mk, ls="none", label=f"Legg {key[2:-3]}")
    a1.set(xlabel="x' [km]", ylabel="Y' [km]", xlim=(-1000, 50), title="Trayectoria media, días 30–50")
    a2.set(xlabel="x' [km]", ylabel="T [Sv]", xlim=(-1000, 50), title="Transporte de la pluma (s > 0.01)")
    if len(data) > 6:                                                # many runs: legend below the panels
        h, lab = a1.get_legend_handles_labels()
        fig.legend(h, lab, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.set_size_inches(11, 5.6)
        fig.tight_layout(rect=(0, 0.22, 1, 1))
    else:
        a1.legend(fontsize=7)
        fig.tight_layout()
    fig.savefig(out / f"dome_{res_km:g}km_path_transport.png", dpi=110)
    plt.close(fig)
    legg_truth_figures(data, out, res_km)
    legg_scorecard(data, out, res_km)
    print(out)
    if anim:
        order = list(LABELS)
        anim_bottom_dye(res_km, sorted(data, key=lambda d: order.index(d[0]) if d[0] in order else len(order)), out)


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--runs", default="", help="comma-separated run directories to keep (default: every 50-day run)")
    ap.add_argument("--sub", default="", help="subdirectory of RUNS/figs for the output")
    ap.add_argument("--before-after", default="", help="baseline,new[,ref,...]: only the before/after figure and the "
                    "bottom-dye animation of these runs (dome_<res>km_before_after.png, dome_<res>km_bottom_dye_before_after)")
    a = ap.parse_args()
    if a.before_after:
        names = [r for r in a.before_after.split(",") if r]
        data = [(n, series.load(cases.RUNS / f"{a.res_km:g}km" / n / "series.npz")[0]) for n in names]
        out = cases.RUNS / "figs" / a.sub if a.sub else cases.RUNS / "figs"
        out.mkdir(parents=True, exist_ok=True)
        print(before_after(data, out, a.res_km))
        if not a.no_anim:
            anim_bottom_dye(a.res_km, data, out, stem=f"dome_{a.res_km:g}km_bottom_dye_before_after")
    else:
        main(a.res_km, not a.no_anim, [r for r in a.runs.split(",") if r] or None, a.sub)
