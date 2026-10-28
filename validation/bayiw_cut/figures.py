#!/usr/bin/env python
"""
Figures and animations of the BayIW 3-D cut (validation/bayiw_cut/README.md), written to RUNS/figs/ (never to the
repo): MITgcm on the same cut (radiating to a resting exterior, Orlanski only) against pyOCEAN z* and hybrid
(25 layers, corrected floor), all on pyOCEAN's domain (MITgcm's interior).

- maps_<case>.png : the internal-wave field of the bay run at 6, 24 and 60 h — T − T(0) at 110 m (linear, real) or
  the displacement ζ of the 20.645 °C isotherm (two-layer).
- stations_<case>.png : η at Seiche (the bay seiche; Seiche is land in the no-bay run), the same internal-wave
  field at Omega, C, A and B, V at 110 m at Omega.
- plan_<case>.mp4/.gif : η (top) and the internal-wave field (bottom) of the bay run, 0–60 h every 15 min.
- The bay effect, bay − nobay on the same grid and dump times (the bay itself is land in the no-bay run, shaded):
  maps_diff_<case>.png (internal-wave field), stations_diff_<case>.png (Δη at Omega and C, Δ field at Omega, C, A,
  B, ΔV at 110 m at Omega) and plan_diff_<case>.mp4/.gif (Δη and Δ field).

Usage: python validation/bayiw_cut/figures.py [2layer linear real] [--no-anim] [--no-diff]
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
from validation.bayiw_cut import compare, cut  # noqa: E402

FIGS = cut.RUNS / "figs"
MODELS = (("mitgcm", "MITgcm z (reference)"), ("pyocean_zstar", "pyOCEAN z*"), ("pyocean_hybrid", "pyOCEAN hybrid 25"))
TIMES = (6.0, 24.0, 60.0)
TITLES = {"2layer": "two-layer", "linear": "linear stratification", "real": "realistic February (density-equivalent T)"}
SHOW = ("Seiche", "Seiche2", "Omega", "A", "B", "C", "Gamma")


def iw_label(case: str) -> str:
    return "ζ of the central isotherm [m]" if case == "2layer" else "T − T(0) at 110 m [°C]"


def reduce(d: dict, case: str, H: np.ndarray) -> dict:
    """Keep what the figures need: t [h], η (n, ny, nx) [m] and the internal-wave field iw (n, ny, nx), NaN on land,
    and V at 110 m (n, ny+1, nx) [m s⁻¹]."""
    land = np.isnan(H)
    if case == "2layer":
        z = compare.isotherm_height(d["T"], compare.ZC, compare.T_MID)
        iw = z - z[:1]
    else:
        T = d["T"][:, compare.K110]
        iw = T - T[:1]
    t = d["t"] / 3600.0
    return dict(t_h=t, eta=np.where(land, np.nan, d["eta"]), iw=np.where(land, np.nan, iw).astype(np.float32),
                V110=d["V"][:, 0].astype(np.float32))


def load(case: str, bay: str) -> dict:
    inp = cut.cut_inputs(case, bay, "py")
    H = inp["H"]
    out = {"mitgcm": reduce(compare.load_mit(case, bay), case, H)}
    for key, coord in (("pyocean_zstar", "zstar"), ("pyocean_hybrid", "hybrid")):
        out[key] = reduce(compare.load_py(case, bay, coord), case, H)
    xe = (inp["x0"] + np.concatenate([[0], np.cumsum(inp["dx"])])) / 1e3
    ye = (inp["y0"] + np.concatenate([[0], np.cumsum(inp["dy"])])) / 1e3
    out.update(H=H, xe=xe - xe[0], ye=ye - ye[0])
    out["xc"], out["yc"] = 0.5 * (out["xe"][1:] + out["xe"][:-1]), 0.5 * (out["ye"][1:] + out["ye"][:-1])
    return out


def difference(b: dict, n: dict) -> dict:
    """Bay effect bay − nobay of every model: t_h, η, iw (n, ny, nx) and V110 (n, ny+1, nx); NaN where either run is
    land (the bay).  Both runs of a model share the dump times (checked)."""
    out = {"H": b["H"], "xe": b["xe"], "ye": b["ye"], "xc": b["xc"], "yc": b["yc"],
           "bay_only": np.isfinite(b["H"]) & np.isnan(n["H"])}
    for key, _ in MODELS:
        if not np.allclose(b[key]["t_h"], n[key]["t_h"]):
            raise ValueError(f"{key}: bay and nobay dump times differ")
        out[key] = {"t_h": b[key]["t_h"], **{v: b[key][v] - n[key][v] for v in ("eta", "iw", "V110")}}
    return out


def shade_bay(ax, d: dict) -> None:
    """The bay (water only in the bay run) in a light shade: no difference is defined there."""
    ax.pcolormesh(d["xe"], d["ye"], np.ma.masked_where(~d["bay_only"], np.ones(d["H"].shape)),
                  cmap=matplotlib.colors.ListedColormap([ps.FAINT]), rasterized=True, zorder=2)


def nearest(t: np.ndarray, t0: float) -> int:
    return int(np.argmin(np.abs(t - t0)))


def limit(d: dict, var: str, q: float = 99.5) -> float:
    return float(np.nanpercentile(np.abs(d["mitgcm"][var]), q))


def plan_axes(ax, d: dict) -> None:
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_facecolor(ps.LAND)
    ax.contour(d["xc"], d["yc"], np.nan_to_num(d["H"], nan=0.0), levels=[170.0, 590.0], colors=ps.MUTED,
               linewidths=0.5)
    for s, (i, j) in cut.local_stations("py").items():
        if s in SHOW:
            ax.plot(d["xc"][i], d["yc"][j], marker="o", ms=2.5, color=ps.INK, zorder=5)
            ax.text(d["xc"][i], d["yc"][j], f" {s}", fontsize=6.5, color=ps.INK, va="center", zorder=5)


def fig_maps(case: str, d: dict, diff: bool = False) -> None:
    v = limit(d, "iw")
    fig, axs = ps.figure(3, 3, aspect=0.8, sharex=True, sharey=True)
    for r, (key, label) in enumerate(MODELS):
        for c, hr in enumerate(TIMES):
            ax = axs[r, c]
            pc = ax.pcolormesh(d["xe"], d["ye"], np.ma.masked_invalid(d[key]["iw"][nearest(d[key]["t_h"], hr)]),
                               cmap=ps.CMAPS["anomaly"], vmin=-v, vmax=v, rasterized=True)
            plan_axes(ax, d)
            if diff:
                shade_bay(ax, d)
            if r == 0:
                ax.set_title(f"{hr:g} h")
            if c == 0:
                ax.set_ylabel(f"{label}\ny [km]")
            if r == 2:
                ax.set_xlabel("x [km]")
    ps.colorbar(fig, pc, axs, ("Δ " if diff else "") + iw_label(case), shrink=0.6)
    what = "bay effect (bay − nobay; grey: the bay)" if diff else "bay"
    fig.suptitle(f"BayIW 3-D cut, {TITLES[case]}, {what}, isobaths 170 and 590 m", x=0.02, ha="left", fontsize=11)
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / f"maps{'_diff' if diff else ''}_{case}.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def fig_stations(case: str, d: dict, diff: bool = False) -> None:
    st = cut.local_stations("py")

    def at(key: str, var: str, s: str) -> np.ndarray:
        i, j = st[s]
        return d[key][var][:, j, i]

    unit = "ζ [m]" if case == "2layer" else "T′ 110 m [°C]"
    if diff:                                 # Seiche and Seiche2 are in the bay: land in the no-bay run
        rows = [(f"Δη, {s} [mm]", lambda k, s=s: 1e3 * at(k, "eta", s)) for s in ("Omega", "C")]
        unit = "Δ" + unit
    else:
        rows = [("η, Seiche [mm]", lambda k: 1e3 * at(k, "eta", "Seiche"))]
    for s in ("Omega", "C", "A", "B"):
        rows.append((f"{unit}, {s}", lambda k, s=s: at(k, "iw", s)))
    rows.append((f"{'Δ' if diff else ''}V 110 m, Omega [cm/s]", lambda k: 1e2 * at(k, "V110", "Omega")))
    fig, axs = ps.figure(len(rows), 1, aspect=1.15, sharex=True)
    for ax, (label, f) in zip(axs, rows):
        for key, _ in MODELS:
            ax.plot(d[key]["t_h"], f(key), **ps.model(key, markers=False, lw=1.3))
        ax.set_ylabel(label, fontsize=9)
    axs[0].legend(ncol=3, loc="upper right")
    axs[-1].set_xlabel("time [h]")
    axs[-1].set_xlim(0, 60)
    what = "bay effect (bay − nobay) at the stations" if diff else "station series"
    fig.suptitle(f"BayIW 3-D cut, {TITLES[case]}: {what}", x=0.02, ha="left", fontsize=11)
    path = FIGS / f"stations{'_diff' if diff else ''}_{case}.png"
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def anim_plan(case: str, d: dict, diff: bool = False) -> None:
    t = d["pyocean_zstar"]["t_h"]
    ve, vi = 1e3 * limit(d, "eta", 99.0), limit(d, "iw")
    fig, axs = plt.subplots(2, 3, figsize=(11.0, 7.6), sharex=True, sharey=True, constrained_layout=True)
    mesh = {}
    for c, (key, label) in enumerate(MODELS):
        mesh[key] = (
            axs[0, c].pcolormesh(d["xe"], d["ye"], np.ma.masked_invalid(1e3 * d[key]["eta"][0]),
                                 cmap=ps.CMAPS["ssh"], vmin=-ve, vmax=ve, rasterized=True),
            axs[1, c].pcolormesh(d["xe"], d["ye"], np.ma.masked_invalid(d[key]["iw"][0]),
                                 cmap=ps.CMAPS["anomaly"], vmin=-vi, vmax=vi, rasterized=True))
        axs[0, c].set_title(label)
        axs[1, c].set_xlabel("x [km]")
        for r in range(2):
            plan_axes(axs[r, c], d)
            if diff:
                shade_bay(axs[r, c], d)
    axs[0, 0].set_ylabel("y [km]")
    axs[1, 0].set_ylabel("y [km]")
    pre = "Δ" if diff else ""
    ps.colorbar(fig, mesh["mitgcm"][0], axs[0], f"{pre}η [mm]", shrink=0.85)
    ps.colorbar(fig, mesh["mitgcm"][1], axs[1], f"{pre} {iw_label(case)}".strip(), shrink=0.85)
    head = f"BayIW 3-D cut, {TITLES[case]}, " + ("bay − nobay (grey: the bay)" if diff else "bay")
    title = fig.suptitle(f"{head} — t = {t[0]:5.2f} h", x=0.02, ha="left", fontsize=11)
    ps.freeze_layout(fig)

    def update(n: int):
        title.set_text(f"{head} — t = {t[n]:5.2f} h")
        for key, _ in MODELS:
            k = nearest(d[key]["t_h"], t[n])
            mesh[key][0].set_array(np.ma.masked_invalid(1e3 * d[key]["eta"][k]))
            mesh[key][1].set_array(np.ma.masked_invalid(d[key]["iw"][k]))

    ps.save_animation(fig, update, len(t), FIGS / f"plan{'_diff' if diff else ''}_{case}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["2layer", "linear", "real"])
    ap.add_argument("--no-anim", action="store_true")
    ap.add_argument("--no-diff", action="store_true")
    a = ap.parse_args()
    ps.use()
    for case in a.cases:
        d = load(case, "bay")
        fig_maps(case, d)
        fig_stations(case, d)
        if not a.no_anim:
            anim_plan(case, d)
        if a.no_diff:
            continue
        dd = difference(d, load(case, "nobay"))
        fig_maps(case, dd, diff=True)
        fig_stations(case, dd, diff=True)
        if not a.no_anim:
            anim_plan(case, dd, diff=True)


if __name__ == "__main__":
    main()
