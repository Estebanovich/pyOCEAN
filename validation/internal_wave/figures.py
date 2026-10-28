#!/usr/bin/env python
"""
Figures and animation of internal_wave (validation/internal_wave/README.md), written to RUNS/figs/ (never to the
repo): MITgcm against pyOCEAN z*, σ and hybrid with the Arakawa kinetic energy (the default since 2026-10-12), all
on MITgcm's levels and columns 2 … 60 (compare.load).

- harmonics.png: amplitude of u at the forcing frequency ω over the last 4 periods (case.harmonic), one row per
  model, and the depth-integrated baroclinic energy flux ⟨F⟩(x) (case.energy_flux) of every model.
- section.mp4/.gif: u on the section, 10 forcing periods, one row per model.

Usage: python validation/internal_wave/figures.py [--no-anim]
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

import pyocean_style as ps  # noqa: E402
from validation.internal_wave import case, compare  # noqa: E402

FIGS = case.RUNS / "figs"
MODELS = (("mitgcm", "MITgcm z (reference)", None), ("pyocean_zstar", "pyOCEAN z*", "zstar_arakawa"),
          ("pyocean_sigma", "pyOCEAN σ", "sigma_arakawa"), ("pyocean_hybrid", "pyOCEAN hybrid", "hybrid_arakawa"))


def load() -> tuple[dict, dict]:
    """({key: {t, u (n, 20, 59) at the cell centres [m/s], T, eta}}, geometry {xe [km], ze [m], H [m], below})."""
    inp = case.load_inputs()
    out = {}
    for key, _, run in MODELS:
        mit, py = compare.load(run or "zstar_arakawa")
        d = mit if run is None else py
        out[key] = dict(t=d["t"], u=compare.cell_u(d["U"]), T=d["T"], eta=d["eta"])
    H = np.nan_to_num(case.mitgcm_depth(inp["H"][1:], inp["dz"]))
    zf = -np.concatenate([[0.0], np.cumsum(inp["dz"])])
    below = zf[:-1, None] <= -H[None] + 1e-6                   # cells entirely below the bed
    geo = dict(xe=mit["x_u"] / 1e3, xc=mit["x_c"] / 1e3, ze=zf, H=H, below=below, inp=inp)
    return out, geo


def masked(geo: dict, X: np.ndarray) -> np.ma.MaskedArray:
    return np.ma.masked_where(geo["below"] | ~np.isfinite(X), X)


def section_axes(ax, geo: dict) -> None:
    ax.grid(False)
    ax.set_facecolor(ps.PAPER)
    ps.fill_bed(ax, np.repeat(geo["xe"], 2)[1:-1], -np.repeat(geo["H"], 2), zmin=geo["ze"][-1])
    ax.set_ylim(geo["ze"][-1], 0.0)
    ax.set_xlim(geo["xe"][0], geo["xe"][-1])


def fig_harmonics(data: dict, geo: dict) -> None:
    omega = 2 * np.pi / case.PERIOD
    fig, axs = ps.figure(len(MODELS) + 1, 1, aspect=1.15, sharex=True,
                         gridspec_kw=dict(height_ratios=[1] * len(MODELS) + [1.2]))
    vmax = None
    for ax, (key, label, _) in zip(axs, MODELS):
        d = data[key]
        w = compare._window(d["t"])
        amp, _ = case.harmonic(d["t"][w], np.nan_to_num(d["u"][w]), omega)
        vmax = vmax or float(np.nanpercentile(masked(geo, amp).compressed(), 99.5)) * 1e3
        pc = ax.pcolormesh(geo["xe"], geo["ze"], masked(geo, 1e3 * amp), cmap=ps.CMAPS["amplitude"], vmin=0,
                           vmax=vmax, rasterized=True)
        section_axes(ax, geo)
        ax.set_ylabel(f"{label}\nz [m]", fontsize=8)
    ps.colorbar(fig, pc, axs[:-1], "|u| at ω [mm/s]", shrink=0.8)
    ax = axs[-1]
    Hc = geo["H"]
    for key, label, _ in MODELS:
        d = data[key]
        w = compare._window(d["t"])
        F = case.energy_flux(d["u"][w], d["T"][w], d["eta"][w], Hc, geo["inp"]["dz"], geo["inp"]["tref"]).mean(0)
        ax.plot(geo["xc"], F, **ps.model(key, markers=False, lw=1.4, label=label))
    ax.axhline(0.0, color=ps.MUTED, lw=0.6)
    ax.set_ylabel(r"$\langle F \rangle$ [W/m]")
    ax.set_xlabel("x [km]")
    ax.legend(fontsize=7, ncol=2)
    fig.suptitle("internal_wave: last 4 forcing periods (u amplitude at ω, baroclinic energy flux)", x=0.02,
                 ha="left", fontsize=11)
    _save(fig, "harmonics.png")


def anim_section(data: dict, geo: dict) -> None:
    t = data["mitgcm"]["t"]
    v = 1e3 * float(np.nanpercentile(np.abs(masked(geo, data["mitgcm"]["u"][-30:]).compressed()), 99.5))
    fig, axs = plt.subplots(len(MODELS), 1, figsize=(8.0, 7.0), sharex=True, sharey=True, constrained_layout=True)
    meshes = []
    for ax, (key, label, _) in zip(axs, MODELS):
        meshes.append(ax.pcolormesh(geo["xe"], geo["ze"], masked(geo, 1e3 * data[key]["u"][0]),
                                    cmap=ps.CMAPS["velocity"], vmin=-v, vmax=v, rasterized=True))
        section_axes(ax, geo)
        ax.set_ylabel("z [m]", fontsize=8)
        ax.text(0.01, 0.06, label, transform=ax.transAxes, fontsize=8, color=ps.INK)
    axs[-1].set_xlabel("x [km]")
    ps.colorbar(fig, meshes[0], axs, "u [mm/s]", shrink=0.6)
    head = "internal_wave: mode-1 wave forced at the west edge, onto the slope"
    title = fig.suptitle(f"{head} — t = {t[0] / case.PERIOD:5.2f} periods", x=0.02, ha="left", fontsize=11)
    ps.freeze_layout(fig)

    def update(n: int):
        title.set_text(f"{head} — t = {t[n] / case.PERIOD:5.2f} periods")
        for m, (key, _, _) in zip(meshes, MODELS):
            d = data[key]
            k = int(np.argmin(np.abs(d["t"] - t[n])))
            m.set_array(masked(geo, 1e3 * d["u"][k]))

    ps.save_animation(fig, update, t.size, FIGS / "section")


def _save(fig, name: str) -> None:
    FIGS.mkdir(parents=True, exist_ok=True)
    path = FIGS / name
    fig.savefig(path)
    plt.close(fig)
    print("wrote", path)


def main() -> None:
    ps.use()
    data, geo = load()
    fig_harmonics(data, geo)
    if "--no-anim" not in sys.argv:
        anim_section(data, geo)


if __name__ == "__main__":
    main()
