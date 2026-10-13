#!/usr/bin/env python
"""
Compare pyOCEAN with MITgcm on internal_wave (spec docs/superpowers/specs/2026-10-12-baroclinic-obc-internal-wave-
design.md §3.3-3.4): harmonic fits at ω (and 2ω) over the last 4 forcing periods, each model on its own dump
times; MITgcm's boundary cell (column 1) and the u face west of it are dropped, so both share the faces
2 … 61 and the columns 2 … 60 of MITgcm.

Usage: python validation/internal_wave/compare.py zstar|hybrid|sigma[_arakawa]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.internal_wave import case  # noqa: E402

OMEGA = 2 * np.pi / case.PERIOD
DEPTHS = (-15.0, -95.0, -175.0)                              # level centres 2, 10, 18 (surface first)
X_FLUX = (2000.0, 6000.0, 10000.0)                           # flat part, slope, shallow part [m]


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def _window(t: np.ndarray, periods: float = 4.0) -> np.ndarray:
    return t >= t[-1] - periods * case.PERIOD - 1.0


def load(coord: str) -> tuple[dict, dict]:
    """(MITgcm, pyOCEAN) on the shared grid: t, U (n, 20, 60), T (n, 20, 59), eta (n, 59), x_u (60,), x_c (59,)."""
    d = case.read_mitgcm()
    inp = case.load_inputs()
    xf = np.concatenate([[0.0], np.cumsum(inp["dx"])])        # MITgcm faces
    mit = dict(t=d["t"], U=d["U"][:, :, 1:], T=d["T"][:, :, 1:], eta=d["eta"][:, 1:], x_u=xf[1:] - xf[1],
               x_c=d["x"][1:] - xf[1])
    land = np.isnan(case.mitgcm_depth(inp["H"][1:], inp["dz"]))
    below = case.to_levels(np.ones((1, 1, 59)), np.zeros((1, 1, 59)),
                           np.nan_to_num(case.mitgcm_depth(inp["H"][1:], inp["dz"]))[None])[:, 0, :]
    mit["T"] = np.where(np.isnan(below)[None] | land[None, None], np.nan, mit["T"])
    p = dict(np.load(case.RUNS / "pyocean" / coord / "series.npz"))
    py = dict(t=p["t"], U=p["U"], T=p["T"], eta=p["eta"], x_u=mit["x_u"], x_c=mit["x_c"], n_trunc=p["n_trunc"],
              heat=p["heat"], sumh=p["sumh"])
    return mit, py


def report(coord: str) -> list[str]:
    mit, py = load(coord)
    inp = case.load_inputs()
    zc = -(np.arange(case.NZ) + 0.5) * case.DZ
    L = [f"# internal_wave, pyOCEAN {coord} vs MITgcm (last 4 periods; x from pyOCEAN's west face)"]
    fits = {}
    for name, d in (("MIT", mit), ("py", py)):
        w = _window(d["t"])
        tw = d["t"][w]
        Tp = d["T"][w] - inp["tref"][None, :, None]
        fits[name] = dict(u=case.harmonic(tw, np.nan_to_num(d["U"][w]), OMEGA),
                          T=case.harmonic(tw, np.nan_to_num(Tp), OMEGA),
                          u2=case.harmonic(tw, np.nan_to_num(d["U"][w]), 2 * OMEGA))
    # mode-1 projection in the flat part (H = 200 m): u on cos(πz/H), T′ on sin(πz/H); robust at the nodes
    flat_u = (mit["x_u"] < 2500.0) & (mit["x_u"] > 0.0)
    flat_c = mit["x_c"] < 2500.0
    L.append("\n## Mode 1 in the flat part (x < 2.5 km): amplitude py/MIT, phase py − MIT [deg] (PASS ±10 %, ±10°)")
    for var, shape, flat, xs in (("u", np.cos(np.pi * zc / 200.0), flat_u, mit["x_u"]),
                                 ("T", np.sin(np.pi * zc / 200.0), flat_c, mit["x_c"])):
        res = {}
        for name, d in (("MIT", mit), ("py", py)):
            w = _window(d["t"])
            X = d["U"][w] if var == "u" else d["T"][w] - inp["tref"][None, :, None]
            a = (np.nan_to_num(X[:, :, flat]) * shape[None, :, None]).sum(1) / (shape**2).sum()
            res[name] = case.harmonic(d["t"][w], a, OMEGA)
        r = res["py"][0] / res["MIT"][0]
        dp = (res["py"][1] - res["MIT"][1] + 180.0) % 360.0 - 180.0
        ok = np.all(np.abs(r - 1) <= 0.10) and np.all(np.abs(dp) <= 10.0)
        L.append(f"{var}: x [km] " + " ".join(f"{v / 1e3:5.2f}" for v in xs[flat][::2]))
        L.append(f"   ratio  " + " ".join(f"{v:5.2f}" for v in r[::2]) + f"   (range {r.min():.2f}–{r.max():.2f})")
        L.append(f"   dphase " + " ".join(f"{v:+5.1f}" for v in dp[::2]) + f"   {_flag(ok)}")
    for var, flat, x in (("u", flat_u, mit["x_u"]), ("T", flat_c, mit["x_c"])):
        L.append(f"\n## {var} at ω (information): amplitude py/MIT and phase py − MIT [deg] at x = 0.5, 1.5, 2.4, 4, 6, 8,"
                 " 10 km (unstable near the nodes of the mode)")
        am, pm = fits["MIT"][var]
        ap, pp = fits["py"][var]
        for z0 in DEPTHS:
            k = int(np.argmin(np.abs(zc - z0)))
            cols = [int(np.argmin(np.abs(x - xx))) for xx in (500, 1500, 2400, 4000, 6000, 8000, 10000)]
            r = ap[k] / np.where(am[k] > 0, am[k], np.nan)
            dp = (pp[k] - pm[k] + 180.0) % 360.0 - 180.0
            L.append(f"z={z0:6.0f} m ratio " + " ".join(f"{r[c]:5.2f}" for c in cols)
                     + " | dphase " + " ".join(f"{dp[c]:+6.1f}" for c in cols))
        both = np.isfinite(am) & np.isfinite(ap) & (am > 0)
        err = np.sqrt(np.nansum(np.abs(ap * np.exp(1j * np.radians(pp)) - am * np.exp(1j * np.radians(pm)))[both] ** 2)
                      / np.nansum(am[both] ** 2))
        L.append(f"relative rms error of the ω field over the section: {err:.3f}")
    L.append("\n## Baroclinic energy flux ⟨F⟩ [W/m] at x = 2, 6, 10 km: MIT, py, ratio (PASS ±15 % where |F| > 1 % of "
             "F at 2 km)")
    Fs = {}
    for name, d in (("MIT", mit), ("py", py)):
        w = _window(d["t"])
        Uc = 0.5 * (d["U"][w][:, :, 1:] + d["U"][w][:, :, :-1])                   # u at the cell centres
        Hc = np.nan_to_num(case.mitgcm_depth(inp["H"][1:], inp["dz"]))
        Fs[name] = case.energy_flux(Uc, d["T"][w], d["eta"][w], Hc, inp["dz"], inp["tref"]).mean(0)
    for xx in X_FLUX:
        c = int(np.argmin(np.abs(mit["x_c"] - xx)))
        r = Fs["py"][c] / Fs["MIT"][c]
        c2 = int(np.argmin(np.abs(mit["x_c"] - X_FLUX[0])))
        small = abs(Fs["MIT"][c]) < 0.01 * abs(Fs["MIT"][c2])
        L.append(f"x={xx / 1e3:4.0f} km  {Fs['MIT'][c]:8.4f} {Fs['py'][c]:8.4f}  {r:5.2f} "
                 + ("(negligible flux)" if small else _flag(abs(r - 1) <= 0.15)))
    x50 = {}
    for name in ("MIT", "py"):
        F = Fs[name] / Fs[name][int(np.argmin(np.abs(mit["x_c"] - X_FLUX[0])))]
        i = int(np.flatnonzero((mit["x_c"] > X_FLUX[0]) & (F < 0.5))[0])
        x50[name] = np.interp(0.5, [F[i], F[i - 1]], [mit["x_c"][i], mit["x_c"][i - 1]])
    L.append(f"flux down to 50 % of its 2-km value at x = {x50['MIT'] / 1e3:.2f} km (MIT), {x50['py'] / 1e3:.2f} km "
             f"(py): {_flag(abs(x50['py'] - x50['MIT']) <= 500.0)} (±0.5 km, ¼ of the local wavelength)")
    L.append("\n## 2ω amplitude of u near the bed, shallow part (x > 9 km): MIT, py [mm/s]")
    a2m, _ = fits["MIT"]["u2"]
    a2p, _ = fits["py"]["u2"]
    for xx in (9000.0, 10000.0, 11000.0, 12000.0):
        c = int(np.argmin(np.abs(mit["x_u"] - xx)))
        k = np.flatnonzero(np.isfinite(mit["U"][-1, :, c]) & (np.abs(mit["U"][-1, :, c]) > 0))
        k = k[-1] if k.size else case.NZ - 1
        L.append(f"x={xx / 1e3:4.1f} km level {k}: {1e3 * a2m[k, c]:6.3f} {1e3 * a2p[k, c]:6.3f}")
    L.append(f"\n## Health py: trunc {int(py['n_trunc'][-1])}, max|Σh − H − η| {py['sumh'].max():.1e} m, "
             f"tracer content change {py['heat'][-1] - py['heat'][0]:+.2e} (T ≈ ±0.05 °C around 0; open west edge)")
    print("\n".join(L))
    return L


if __name__ == "__main__":
    report(sys.argv[1])
