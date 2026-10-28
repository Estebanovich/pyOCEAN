#!/usr/bin/env python
"""
Figures and animations of the four-model benchmark part A (validation/ilicak_benchmark/README.md), written to
RUNS/figs/ (never to the repo), from the common series of every run (series.py): pyOCEAN z*/σ/hybrid, MOM6 z*/ρ
(σ in the seamount), CROCO σ and MITgcm z*.

- sections_<case>.png (lock, overflow, iw): T on each model's own layers, one row per model
  and one column per ν_h (iw: T minus the background profile, Petersen 2015 eq. 26).  Empty panel = no run (the
  model stopped).
- rpe_<case>.png: (RPE − RPE₀)/RPE₀ against time per ν_h (Ilicak et al. 2012, Fig. 5–6), metrics.rpe.
- iw_amplitude.png: displacement of the 15.1 °C isotherm at x = 125 km (metrics.iw_amplitude), first and last 20
  days, per ν_h.
- seamount.png: max speed against time (metrics.max_speed) and the column-max speed at day 10 (log scale), per
  model.
- section_<case>_nu<ν>.mp4/.gif: the sections in time, one row per model (iw: daily).

Usage: python validation/ilicak_benchmark/figures.py [lock overflow iw seamount] [--no-anim]
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
from validation.ilicak_benchmark import cases, metrics, series  # noqa: E402

FIGS = cases.RUNS / "figs"
# (model, coord, style key, label)
MODELS = (("pyocean", "zstar", "pyocean_zstar", "pyOCEAN z*"), ("pyocean", "sigma", "pyocean_sigma", "pyOCEAN σ"),
          ("pyocean", "hybrid", "pyocean_hybrid", "pyOCEAN hybrid"), ("mom6", "zstar", "mom6", "MOM6 z*"),
          ("mom6", "rho", "mom6_rho", "MOM6 ρ"), ("croco", "sigma", "croco", "CROCO σ"),
          ("mitgcm", "zstar", "mitgcm", "MITgcm z*"))
SEAMOUNT = (("pyocean", "zstar", "pyocean_zstar", "pyOCEAN z*"), ("pyocean", "sigma", "pyocean_sigma", "pyOCEAN σ"),
            ("mom6", "sigma", "mom6", "MOM6 σ"), ("croco", "sigma", "croco", "CROCO σ"),
            ("mitgcm", "zstar", "mitgcm", "MITgcm z*"))
TITLE = {"lock": "Lock exchange", "overflow": "Overflow", "iw": "Internal wave", "seamount": "Seamount"}
# colour limits of the sections and the anim frame stride
# and the time of the still sections [s] (Ilicak et al. 2012: lock 17 h, overflow 3 h; iw: the end, 200 d)
VIEW = {"lock": dict(vmin=5.0, vmax=30.0, stride=1, unit="h", zmin=-20.0, snap=17 * 3600.0),
        "overflow": dict(vmin=10.0, vmax=20.0, stride=1, unit="h", zmin=-2050.0, snap=3 * 3600.0),
        "iw": dict(vmin=-2.0, vmax=2.0, stride=4, unit="d", zmin=-500.0, snap=200 * 86400.0)}


def load(case: str, model: str, coord: str, nu: float) -> dict | None:
    """The series of one run, or None when the run does not exist (stopped or not run)."""
    path = cases.run_dir(case, model, coord, nu) / "series.npz"
    return series.load(path)[0] if path.exists() else None


def interfaces(s: dict, n: int) -> np.ndarray:
    """Interface heights z_f [m] (nz+1, nx) of record n of a 2-D (y = 1) series: η minus the cumulative layer
    thicknesses, surface first."""
    h = s["h"][n, :, 0, :]
    return s["eta"][n, 0, :][None] - np.concatenate([np.zeros((1, h.shape[1])), np.cumsum(h, 0)])


def field(case: str, s: dict, n: int) -> np.ndarray:
    """What the sections show (nz, nx): T [°C], or for the internal wave T minus the background profile
    T_b(z) = t_bot + (t_top − t_bot)(z_bot − z)/z_bot at the layer centres (Petersen 2015, eq. 26)."""
    T = s["T"][n, :, 0, :]
    if case != "iw":
        return T
    zf = interfaces(s, n)
    zc = 0.5 * (zf[1:] + zf[:-1])
    t_bot, t_top, z_bot = 10.1, 20.1, -487.5
    return T - (t_bot + (t_top - t_bot) * (z_bot - zc) / z_bot)


def step_mesh(s: dict, n: int, C: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(X [km], Z [m], C) for pcolormesh with flat columns: each column keeps its own interface heights (z* partial
    cells and vanished layers drawn as they are, not smoothed between neighbours); the zero-width quads between
    columns are masked."""
    zf = interfaces(s, n)
    nz, nx = C.shape
    xe = np.concatenate([[0.0], np.cumsum(s["dx"])]) / 1e3
    X = np.empty((nz + 1, 2 * nx))
    X[:, 0::2], X[:, 1::2] = xe[:-1], xe[1:]
    Z = np.repeat(zf, 2, axis=1)
    Cm = np.full((nz, 2 * nx - 1), np.nan)
    Cm[:, 0::2] = np.where(s["h"][n, :, 0, :] > 1e-6, C, np.nan)
    return X, Z, np.ma.masked_invalid(Cm)


def draw(ax, case: str, s: dict | None, n: int, label: str = ""):
    """One section panel; returns the mesh (None for a missing run)."""
    ax.grid(False)
    ax.set_facecolor(ps.FAINT)
    ax.set_ylim(VIEW[case]["zmin"], 0.0)
    if s is None:
        ax.text(0.5, 0.5, "no run (stopped)", transform=ax.transAxes, ha="center", va="center", color=ps.MUTED,
                fontsize=8)
        return None
    v = VIEW[case]
    cmap = ps.CMAPS["anomaly" if case == "iw" else "temperature"]
    pc = ax.pcolormesh(*step_mesh(s, n, field(case, s, n)), cmap=cmap, vmin=v["vmin"], vmax=v["vmax"],
                       rasterized=True)
    if case == "overflow":
        xe = np.concatenate([[0.0], np.cumsum(s["dx"])]) / 1e3
        ps.fill_bed(ax, np.repeat(xe, 2)[1:-1], -np.repeat(s["H"][0], 2), zmin=-2050.0)
    if label:
        ax.text(0.01, 0.06, label, transform=ax.transAxes, fontsize=7.5, color=ps.PAPER if case == "lock" else ps.INK,
                va="bottom")
    return pc


def cbar_label(case: str) -> str:
    return "T − T_b(z) [°C]" if case == "iw" else "T [°C]"


def when(case: str, t: float) -> str:
    return f"{t / 86400:.1f} d" if VIEW[case]["unit"] == "d" else f"{t / 3600:.1f} h"


def fig_sections(case: str, runs: dict) -> None:
    c = cases.get(case)
    fig, axs = ps.figure(len(MODELS), len(c.nu_h), aspect=1.05 if case != "overflow" else 1.25,
                         sharex=True, sharey=True)
    pc = None
    for r, (m, co, key, label) in enumerate(MODELS):
        for j, nu in enumerate(c.nu_h):
            s = runs[(m, co, nu)]
            p = draw(axs[r, j], case, s, 0 if s is None else int(np.argmin(np.abs(s["t"] - VIEW[case]["snap"]))))
            pc = pc or p
            if r == 0:
                axs[r, j].set_title(f"$\\nu_h$ = {nu:g} m² s⁻¹")
            if j == 0:
                axs[r, j].set_ylabel(f"{label}\nz [m]", fontsize=8)
            if r == len(MODELS) - 1:
                axs[r, j].set_xlabel("x [km]")
    ps.colorbar(fig, pc, axs, cbar_label(case), shrink=0.5)
    fig.suptitle(f"{TITLE[case]} (Ilicak et al. 2012), T at {when(case, VIEW[case]['snap'])}", x=0.02, ha="left",
                 fontsize=11)
    _save(fig, f"sections_{case}.png")


def fig_rpe(case: str, runs: dict) -> None:
    c = cases.get(case)
    fig, axs = ps.figure(1, len(c.nu_h), aspect=0.36, sharex=True)
    axs = np.atleast_1d(axs)
    scale = 86400.0 if case in ("iw", "seamount") else 3600.0
    for ax, nu in zip(axs, c.nu_h):
        for m, co, key, label in (MODELS if case != "seamount" else SEAMOUNT):
            s = runs.get((m, co, nu))
            if s is None:
                continue
            r = metrics.rpe(s)
            ax.plot(s["t"] / scale, (r - r[0]) / r[0], **ps.model(key, markers=False, lw=1.2, label=label))
        ax.set_title(f"$\\nu_h$ = {nu:g} m² s⁻¹" if case != "seamount" else "$\\nu_h$ = 0")
        ax.set_xlabel("time [d]" if scale == 86400.0 else "time [h]")
        ax.ticklabel_format(axis="y", style="sci", scilimits=(-2, 2))
    axs[0].set_ylabel("(RPE − RPE₀) / RPE₀")
    axs[-1].legend(fontsize=7, loc="upper left")
    fig.suptitle(f"{TITLE[case]}: reference potential energy (spurious mixing)", x=0.02, ha="left", fontsize=11)
    _save(fig, f"rpe_{case}.png")


def fig_iw_amplitude(runs: dict) -> None:
    """The first and the last 20 days side by side (metrics.oscillation_ratio compares their std)."""
    c = cases.get("iw")
    fig, axs = ps.figure(len(c.nu_h), 2, aspect=0.8, sharey="row", gridspec_kw=dict(wspace=0.04))
    end = c.hours / 24.0
    for r, nu in enumerate(c.nu_h):
        for m, co, key, label in MODELS:
            s = runs[(m, co, nu)]
            if s is None:
                continue
            a = metrics.iw_amplitude(s)
            for ax in axs[r]:
                ax.plot(s["t"] / 86400.0, a, **ps.model(key, markers=False, lw=1.0, label=label))
        axs[r, 0].set_xlim(0, 20)
        axs[r, 1].set_xlim(end - 20, end)
        axs[r, 0].set_ylabel(f"$\\nu_h$ = {nu:g}\nζ(15.1 °C) [m]", fontsize=9)
    axs[0, 0].set_title("first 20 days")
    axs[0, 1].set_title("last 20 days")
    for ax in axs[-1]:
        ax.set_xlabel("time [d]")
    axs[0, 1].legend(ncol=2, fontsize=6.5, loc="lower right")
    fig.suptitle("Internal wave: isotherm displacement at x = 125 km", x=0.02, ha="left", fontsize=11)
    _save(fig, "iw_amplitude.png")


def fig_seamount(runs: dict) -> None:
    fig = plt.figure(figsize=(ps.WIDTH, ps.WIDTH * 0.62), constrained_layout=True)
    gs = fig.add_gridspec(2, len(SEAMOUNT), height_ratios=[1.0, 1.1])
    ax = fig.add_subplot(gs[0, :])
    have = [(k, lab, runs[(m, co, 0.0)]) for m, co, k, lab in SEAMOUNT if runs[(m, co, 0.0)] is not None]
    for key, label, s in have:
        ax.semilogy(s["t"] / 86400.0, 1e2 * np.maximum(metrics.max_speed(s), 1e-6),
                    **ps.model(key, markers=True, lw=1.2, ms=3, label=label))
    ax.set_xlabel("time [d]")
    ax.set_ylabel("max |u| [cm/s]")
    ax.legend(ncol=5, fontsize=7, loc="lower right")
    ax.set_title("spurious flow from the pressure-gradient error (Beckmann & Haidvogel 1993)", fontsize=9)
    speeds = {}
    for key, label, s in have:  # column max of the cell-centre speed in layers thicker than 1 cm
        uc = 0.5 * (s["u"][-1, ..., 1:] + s["u"][-1, ..., :-1])
        vc = 0.5 * (s["v"][-1, :, 1:, :] + s["v"][-1, :, :-1, :])
        sp = np.where(s["h"][-1] > 0.01, np.hypot(uc, vc), 0.0).max(0)
        speeds[label] = (s, sp)
    norm = matplotlib.colors.LogNorm(1e-3, 1e2)           # [cm/s]: the models span 13 orders of magnitude
    for j, (label, (s, sp)) in enumerate(speeds.items()):
        a = fig.add_subplot(gs[1, j])
        xe = (np.concatenate([[0.0], np.cumsum(s["dx"])]) - s["dx"].sum() / 2) / 1e3
        ye = (np.concatenate([[0.0], np.cumsum(s["dy"])]) - s["dy"].sum() / 2) / 1e3
        pc = a.pcolormesh(xe, ye, np.maximum(1e2 * sp, 1e-3), cmap=ps.CMAPS["speed"], norm=norm, rasterized=True)
        a.contour(0.5 * (xe[1:] + xe[:-1]), 0.5 * (ye[1:] + ye[:-1]), s["H"], levels=[1000, 2500, 4000],
                  colors=ps.MUTED, linewidths=0.5)
        a.set_aspect("equal")
        a.grid(False)
        a.set_title(f"{label}\nmax {1e2 * metrics.max_speed(s)[-1]:.2g} cm/s", fontsize=8)
        a.set_xlabel("x [km]")
        if j:
            a.set_yticklabels([])
        else:
            a.set_ylabel("y [km]")
    ps.colorbar(fig, pc, fig.axes[1:], "|u| day 10 [cm/s]", shrink=0.9)
    fig.suptitle("Seamount: spurious currents at rest ($\\nu_h$ = 0, 10 days)", x=0.02, ha="left", fontsize=11)
    _save(fig, "seamount.png")


def anim_sections(case: str, runs: dict, nu: float) -> None:
    rows = [(lab, runs[(m, co, nu)]) for m, co, _, lab in MODELS]
    ref = next(s for _, s in rows if s is not None)
    frames = np.arange(0, ref["t"].size, VIEW[case]["stride"])
    height = {"lock": 0.72, "overflow": 1.05, "iw": 0.95}[case]
    fig, axs = plt.subplots(len(rows), 1, figsize=(8.0, 8.0 * height), sharex=True, sharey=True,
                            constrained_layout=True)
    pcs = [draw(ax, case, s, 0, lab) for ax, (lab, s) in zip(axs, rows)]
    for ax in axs:
        ax.set_ylabel("z [m]", fontsize=8)
    axs[-1].set_xlabel("x [km]")
    ps.colorbar(fig, next(p for p in pcs if p is not None), axs, cbar_label(case), shrink=0.5)
    head = f"{TITLE[case]}, $\\nu_h$ = {nu:g} m² s⁻¹"
    title = fig.suptitle(f"{head} — t = {when(case, ref['t'][0])}", x=0.02, ha="left", fontsize=11)
    ps.freeze_layout(fig)

    def update(f: int):
        n = frames[f]
        title.set_text(f"{head} — t = {when(case, ref['t'][n])}")
        for i, (ax, (lab, s)) in enumerate(zip(axs, rows)):
            if s is None:
                continue
            k = min(n, s["t"].size - 1)
            pcs[i].remove()
            v = VIEW[case]
            pcs[i] = ax.pcolormesh(*step_mesh(s, k, field(case, s, k)),
                                   cmap=ps.CMAPS["anomaly" if case == "iw" else "temperature"],
                                   vmin=v["vmin"], vmax=v["vmax"], rasterized=True)

    ps.save_animation(fig, update, frames.size, FIGS / f"section_{case}_nu{nu:g}", fps=8 if case != "iw" else 12,
                      gif_width=960 if case != "iw" else 640)


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["lock", "overflow", "iw", "seamount"])
    ap.add_argument("--no-anim", action="store_true")
    a = ap.parse_args()
    ps.use()
    for case in a.cases:
        c = cases.get(case)
        models = SEAMOUNT if case == "seamount" else MODELS
        runs = {(m, co, nu): load(case, m, co, nu) for m, co, _, _ in models for nu in c.nu_h}
        fig_rpe(case, runs)
        if case == "seamount":
            fig_seamount(runs)
            continue
        fig_sections(case, runs)
        if case == "iw":
            fig_iw_amplitude(runs)
        if not a.no_anim:
            for nu in c.nu_h:
                anim_sections(case, runs, nu)


if __name__ == "__main__":
    main()
