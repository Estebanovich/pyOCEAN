#!/usr/bin/env python
"""
Figures and animations of the BayIW 2-D section (validation/bayiw_2d/README.md), written to RUNS/figs/ (never to
the repo): MITgcm (Orlanski-only reference, 'orl') against pyOCEAN z* and hybrid (25 layers, corrected floor) and,
where they exist (two-layer, real), MOM6 z* and HYCOM1 (validation/mom6_bayiw) on the same section, all on MITgcm's
50 levels.

- sections_<case>.png : T anomaly T − T(0) of the bay run at 6, 24 and 60 h, isotherms of T on top (linear, real).
- hovmoller_2layer.png : two-layer: displacement ζ(y, t) of the central isotherm (20.64 °C), which moves only
  ~0.2 m in the section, too little to see in a T section.
- stations_<case>.png : bay effect Δη = η_bay − η_nobay at Seiche, T anomaly at 110 m (Omega, C), V at 110 m (Omega).
- section_<case>.mp4/.gif : the T anomaly section of the bay run, 0–60 h every 15 min, the models stacked,
  with η at Seiche below (linear, real); for the two-layer case ζ(y) and η(y) of the models instead.

Usage: python validation/bayiw_2d/figures.py [2layer linear real] [--no-anim]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "docs" / "manual" / "figures"))

import pyocean_style as ps  # noqa: E402
from validation.bayiw_2d import section  # noqa: E402
from validation.bayiw_2d.compare import mit_at_depth  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_cut.compare import isotherm_height  # noqa: E402

FIGS = section.RUNS / "figs"
BASE = (("mitgcm", "MITgcm z (reference)"), ("pyocean_zstar", "pyOCEAN z*"), ("pyocean_hybrid", "pyOCEAN hybrid 25"))
# MOM6 on the same section (validation/mom6_bayiw: two-layer and real only; first output at 0.25 h)
MOM6 = (("mom6", "MOM6 z*", "mom6zstar"), ("mom6_hycom1", "MOM6 HYCOM1", "mom6hycom1"))
MODELS = BASE                                   # set per case by main(): BASE + the MOM6 runs that exist
TIMES = (6.0, 24.0, 60.0)
ISOTHERMS = {"2layer": [20.64], "linear": None, "real": None}     # None: 8 evenly spaced levels of the initial T
# Per case: colour field ('abs' = T, 'anom' = T − T(0)), z window [m], y window [km from the south edge].
# Two-layer: a close-up of the interface (≈ 115 m), whose displacement is ~10 m.
VIEW = {"2layer": ("abs", (-175.0, -60.0), (1300.0, 1836.2)), "linear": ("anom", (-620.0, 0.0), (1100.0, 1836.2)),
        "real": ("anom", (-620.0, 0.0), (1100.0, 1836.2))}
TITLES = {"2layer": "two-layer", "linear": "linear stratification", "real": "realistic February (density-equivalent T)"}


def load(case: str, bay: str) -> dict:
    """Section fields of the models on MITgcm's levels: t_h (n,), T (n, 50, 352) NaN below the bed, η (n, 352),
    T110, V110 (n, 352|353); plus H (352,) and the along-section coordinate y (352,) [km] from the south edge."""
    H = section.section_inputs(case, bay)["H"][:, 0]
    below = bayiw.Z_C[:, None] < -np.nan_to_num(H, nan=0.0)[None]
    m = section.read_mitgcm(case, bay, "orl")
    out = {"mitgcm": dict(t_h=m["t_h"], T=np.where(below, np.nan, m["T"]), eta=m["eta"],
                          T110=mit_at_depth(m, H, "T", bayiw.Z110), V110=mit_at_depth(m, H, "V", bayiw.Z110))}
    runs = [("pyocean_zstar", "zstar"), ("pyocean_hybrid", "hybrid")] + [(k, c) for k, _, c in MOM6]
    for key, coord in runs:
        path = section.RUNS / "pyocean" / f"{case}_{bay}_{coord}" / "series.npz"
        if not path.exists():
            continue
        p = np.load(path)
        out[key] = dict(t_h=p["t_h"], T=np.where(below, np.nan, p["T_sec"]), eta=p["eta"], T110=p["T110"],
                        V110=p["V110"])
    dy = section.section_inputs(case, bay)["dy"]
    out["H"], out["y"], out["y_edges"] = H, (np.cumsum(dy) - 0.5 * dy) / 1e3, np.concatenate([[0], np.cumsum(dy)]) / 1e3
    return out


def nearest(t: np.ndarray, t0: float) -> int:
    return int(np.argmin(np.abs(t - t0)))


def anomaly_limit(case: str, d: dict) -> float:
    if case == "2layer":
        return 3.0
    a = d["mitgcm"]["T"] - d["mitgcm"]["T"][:1]
    return float(np.nanpercentile(np.abs(a), 99.5))


def isotherm_levels(case: str, d: dict) -> np.ndarray:
    if ISOTHERMS[case] is not None:
        return np.array(ISOTHERMS[case])
    T0 = d["mitgcm"]["T"][0]
    return np.linspace(np.nanpercentile(T0, 3), np.nanpercentile(T0, 97), 8)


def colour_field(case: str, T: np.ndarray, n: int) -> np.ndarray:
    return np.ma.masked_invalid(T[n] if VIEW[case][0] == "abs" else T[n] - T[0])


def colour_style(case: str, d: dict) -> tuple[dict, str]:
    """pcolormesh keywords and colour-bar label of a case."""
    if VIEW[case][0] == "abs":
        T0 = d["mitgcm"]["T"][0]
        return dict(cmap=ps.CMAPS["temperature"], vmin=np.nanmin(T0), vmax=np.nanmax(T0)), "T [°C]"
    v = anomaly_limit(case, d)
    return dict(cmap=ps.CMAPS["anomaly"], vmin=-v, vmax=v), "T − T(0) [°C]"


def draw_section(ax, case: str, d: dict, key: str, n: int, levels: np.ndarray):
    """One section panel: T or T anomaly (colour), isotherms (lines), bed; returns (mesh, contour set)."""
    _, (zmin, ztop), ywin = VIEW[case]
    T = d[key]["T"]
    pc = ax.pcolormesh(d["y_edges"], bayiw.Z_F, colour_field(case, T, n), shading="flat", rasterized=True,
                       **colour_style(case, d)[0])
    cs = ax.contour(d["y"], bayiw.Z_C, T[n], levels=levels, colors=ps.INK, linewidths=0.6)
    ps.fill_bed(ax, np.repeat(d["y_edges"], 2)[1:-1], -np.repeat(np.nan_to_num(d["H"]), 2), zmin=zmin)
    ax.set_ylim(zmin, ztop)
    ax.set_xlim(*ywin)
    ps.section_axes(ax, xlabel="", ylabel="z [m]")
    return pc, cs


T_MID = 20.64                                            # central isotherm of the two-layer case [°C]


def interface(d: dict) -> None:
    """Add ζ = z(T_MID) − z(T_MID, t = 0) (n, 352) [m] to each model of a two-layer case."""
    for key, _ in MODELS:
        z = isotherm_height(d[key]["T"][..., None], bayiw.Z_C, T_MID)[..., 0]
        d[key]["zeta"] = z - z[:1]


def fig_hovmoller(d: dict) -> None:
    ywin = VIEW["2layer"][2]
    v = float(np.nanpercentile(np.abs(d["mitgcm"]["zeta"]), 99.5))
    fig, axs = ps.figure(1, len(MODELS), aspect=0.42, sharey=True)
    for ax, (key, label) in zip(axs, MODELS):
        pc = ax.pcolormesh(d["y"], d[key]["t_h"], np.ma.masked_invalid(d[key]["zeta"]),
                           cmap=ps.CMAPS["anomaly"], vmin=-v, vmax=v, shading="nearest", rasterized=True)
        ax.set_xlim(*ywin)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("y [km]")
        ax.grid(False)
        for s, j in section.SECTION_STATIONS.items():
            if d["y"][j] > ywin[0]:
                ax.axvline(d["y"][j], color=ps.MUTED, lw=0.5, ls=":")
    axs[0].set_ylabel("time [h]")
    ps.colorbar(fig, pc, axs, "ζ of the 20.64 °C isotherm [m]", shrink=0.9)
    fig.suptitle("BayIW 2-D section, two-layer, bay: interface displacement", x=0.02, ha="left", fontsize=11)
    path = FIGS / "hovmoller_2layer.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def anim_interface(d: dict) -> None:
    """Two-layer animation: ζ(y) and η(y) of the models."""
    ywin = VIEW["2layer"][2]
    t = d["pyocean_zstar"]["t_h"]
    sel = d["y"] >= ywin[0]
    zl = 1.1 * max(np.nanmax(np.abs(d[k]["zeta"][:, sel])) for k, _ in MODELS)
    el = 1.1e3 * max(np.nanmax(np.abs(d[k]["eta"][:, sel])) for k, _ in MODELS)
    fig, (ax_z, ax_e) = plt.subplots(2, 1, figsize=(9.6, 6.4), sharex=True, constrained_layout=True)
    lines = {}
    for key, _ in MODELS:
        lz, = ax_z.plot(d["y"], d[key]["zeta"][0], **ps.model(key, label=dict(MODELS)[key], markers=False, lw=1.4))
        le, = ax_e.plot(d["y"], 1e3 * d[key]["eta"][0], **ps.model(key, label=dict(MODELS)[key], markers=False, lw=1.4))
        lines[key] = (lz, le)
    for ax in (ax_z, ax_e):
        ax.axhline(0, color=ps.MUTED, lw=0.6)
        for s, j in section.SECTION_STATIONS.items():
            if d["y"][j] > ywin[0]:
                ax.axvline(d["y"][j], color=ps.MUTED, lw=0.5, ls=":")
                if ax is ax_z:
                    ax.text(d["y"][j], 0.97, f" {s}", transform=ax.get_xaxis_transform(), fontsize=7.5,
                            color=ps.MUTED, ha="left", va="top")
    ax_z.axvspan(d["y_edges"][324], ywin[1], color=ps.WATER, alpha=0.5, lw=0, zorder=0)
    ax_e.axvspan(d["y_edges"][324], ywin[1], color=ps.WATER, alpha=0.5, lw=0, zorder=0)
    ax_z.set_ylim(-zl, zl)
    ax_e.set_ylim(-el, el)
    ax_z.set_ylabel("ζ of 20.64 °C [m]")
    ax_e.set_ylabel("η [mm]")
    ax_e.set_xlabel("y from the south edge [km] (shaded: shelf and bay, H = 163 m)")
    ax_e.set_xlim(*ywin)
    ax_z.legend(ncol=3, loc="lower left", fontsize=8.5)
    title = fig.suptitle(f"BayIW 2-D section, two-layer, bay — t = {t[0]:5.2f} h", x=0.02, ha="left", fontsize=11)
    ps.freeze_layout(fig)

    def update(n: int):
        title.set_text(f"BayIW 2-D section, two-layer, bay — t = {t[n]:5.2f} h")
        for key, _ in MODELS:
            k = nearest(d[key]["t_h"], t[n])
            lines[key][0].set_ydata(d[key]["zeta"][k])
            lines[key][1].set_ydata(1e3 * d[key]["eta"][k])

    ps.save_animation(fig, update, len(t), FIGS / "section_2layer")


def mark_stations(ax, d: dict) -> None:
    for s, j in section.SECTION_STATIONS.items():
        if s == "Gamma":
            continue
        ax.axvline(d["y"][j], color=ps.MUTED, lw=0.5, ls=":", zorder=5)
        ax.text(d["y"][j], 0.03, f" {s}", transform=ax.get_xaxis_transform(), fontsize=7.5, color=ps.MUTED,
                ha="left", va="bottom", zorder=6)


def fig_sections(case: str, d: dict) -> None:
    levels = isotherm_levels(case, d)
    fig, axs = ps.figure(len(MODELS), 3, aspect=0.21 * len(MODELS), sharex=True, sharey=True)
    for r, (key, label) in enumerate(MODELS):
        for c, hr in enumerate(TIMES):
            pc, _ = draw_section(axs[r, c], case, d, key, nearest(d[key]["t_h"], hr), levels)
            if r == 0:
                axs[r, c].set_title(f"{hr:g} h")
            if c == 0:
                axs[r, c].set_ylabel(f"{label}\nz [m]")
            else:
                axs[r, c].set_ylabel("")
    for ax in axs[-1]:
        ax.set_xlabel("y from the south edge [km]")
    ps.colorbar(fig, pc, axs, colour_style(case, d)[1], shrink=0.6)
    fig.suptitle(f"BayIW 2-D section, {TITLES[case]}, bay: {colour_style(case, d)[1]} and isotherms", x=0.02, ha="left",
                 fontsize=11)
    path = FIGS / f"sections_{case}.png"
    FIGS.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_stations(case: str, d: dict, dn: dict) -> None:
    st = section.SECTION_STATIONS
    rows = (("Δη bay − nobay at Seiche [mm]", lambda k: 1e3 * (d[k]["eta"][:, st["Seiche"]]
                                                               - np.interp(d[k]["t_h"], dn[k]["t_h"],
                                                                           dn[k]["eta"][:, st["Seiche"]]))),
            ("T′ at 110 m, Omega [°C]", lambda k: d[k]["T110"][:, st["Omega"]] - d[k]["T110"][0, st["Omega"]]),
            ("T′ at 110 m, C [°C]", lambda k: d[k]["T110"][:, st["C"]] - d[k]["T110"][0, st["C"]]),
            ("V at 110 m, Omega [cm/s]", lambda k: 1e2 * d[k]["V110"][:, st["Omega"]]))
    fig, axs = ps.figure(len(rows), 1, aspect=0.95, sharex=True)
    for ax, (label, f) in zip(axs, rows):
        for key, _ in MODELS:
            ax.plot(d[key]["t_h"], f(key), **ps.model(key, label=dict(MODELS)[key], markers=False, lw=1.3))
        ax.set_ylabel(label, fontsize=9.5)
    axs[0].legend(ncol=3, loc="upper right")
    axs[-1].set_xlabel("time [h]")
    axs[-1].set_xlim(0, 60)
    fig.suptitle(f"BayIW 2-D section, {TITLES[case]}: station series", x=0.02, ha="left", fontsize=11)
    path = FIGS / f"stations_{case}.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def anim_section(case: str, d: dict) -> None:
    levels = isotherm_levels(case, d)
    t = d["pyocean_zstar"]["t_h"]
    n = len(MODELS)
    fig = plt.figure(figsize=(9.6, 2.35 * n + 1.35), constrained_layout=True)
    gs = fig.add_gridspec(n + 1, 1, height_ratios=(1,) * n + (0.55,))
    axs = [fig.add_subplot(gs[r]) for r in range(n)]
    ax_eta = fig.add_subplot(gs[n])
    state = {}
    for ax, (key, label) in zip(axs, MODELS):
        pc, cs = draw_section(ax, case, d, key, 0, levels)
        ax.set_title(label)
        state[key] = (pc, cs)
        mark_stations(ax, d)
    axs[-1].set_xlabel("y from the south edge [km]")
    ps.colorbar(fig, state["mitgcm"][0], axs, colour_style(case, d)[1], shrink=0.5)
    j = section.SECTION_STATIONS["Seiche"]
    for key, label in MODELS:
        ax_eta.plot(d[key]["t_h"], 1e3 * d[key]["eta"][:, j], **ps.model(key, label=label, markers=False, lw=1.1))
    cursor = ax_eta.axvline(0, color=ps.WARN, lw=1.0)
    ax_eta.set_xlim(0, 60)
    ax_eta.set_xlabel("time [h]")
    ax_eta.set_ylabel("η at Seiche [mm]")
    ax_eta.legend(ncol=3, loc="upper right", fontsize=8.5)
    title = fig.suptitle(f"BayIW 2-D section, {TITLES[case]}, bay — t = {t[0]:5.2f} h", x=0.02, ha="left",
                         fontsize=11)
    ps.freeze_layout(fig)

    def update(n: int):
        title.set_text(f"BayIW 2-D section, {TITLES[case]}, bay — t = {t[n]:5.2f} h")
        for ax, (key, _) in zip(axs, MODELS):
            pc, cs = state[key]
            k = nearest(d[key]["t_h"], t[n])
            T = d[key]["T"]
            pc.set_array(colour_field(case, T, k))
            cs.remove()
            state[key] = (pc, ax.contour(d["y"], bayiw.Z_C, T[k], levels=levels, colors=ps.INK, linewidths=0.6))
        cursor.set_xdata([t[n], t[n]])

    ps.save_animation(fig, update, len(t), FIGS / f"section_{case}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["2layer", "linear", "real"])
    ap.add_argument("--no-anim", action="store_true")
    a = ap.parse_args()
    ps.use()
    global MODELS
    for case in a.cases:
        d, dn = load(case, "bay"), load(case, "nobay")
        MODELS = BASE + tuple((k, lab) for k, lab, _ in MOM6 if k in d and k in dn)
        if case == "2layer":
            interface(d)
            fig_hovmoller(d)
        else:
            fig_sections(case, d)
        fig_stations(case, d, dn)
        if not a.no_anim:
            anim_interface(d) if case == "2layer" else anim_section(case, d)


if __name__ == "__main__":
    main()
