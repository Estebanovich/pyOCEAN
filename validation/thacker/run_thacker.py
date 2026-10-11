#!/usr/bin/env python
"""
Thacker (1981) wetting–drying benchmark — CROCO's THACKER (THACKER_2DV) case
reproduced with pyOCEAN and compared with the exact solution and, if
available, the CROCO run.

Setup (CROCO OCEAN/ana_grid.F, ana_initial.F, TEST_CASES/croco.in.Thacker):
    channel 200 km, 200 cells of 1 km, h = 10 m (1 − x²/80 km²), f = 0
    η(x, 0) = 2 η₀ D x / L (η₀ = 0.1), u = 0;  Δt = 60 s, 4320 steps (72 h), no drag
Exact solution (planar surface, uniform u):
    η = A₀ x cos ωt + η₀² D sin² ωt,  u = −η₀ ω L sin ωt,  A₀ = 2 η₀ D / L,  ω² = 2 g D / L²

Usage:  python validation/thacker/run_thacker.py [--croco PATH/thacker_his.nc] [--theta 0.55]
        (CROCO must be built with ZETA_DRY_IO so that ``zeta`` is η on dry cells too;
         run directory ~/pyOCEAN_refs/croco_runs/thacker, see validation/thacker/README.md)
Output: validation/thacker/out/thacker_pyocean.npz, thacker.png, summary.txt
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

from core.dynamics.wetting_drying import WetDryFreeSurface  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402

G, D0, L, ETA0 = 9.81, 10.0, 80e3, 0.1
OMEGA = np.sqrt(2 * G * D0 / L**2)
OUT = Path(__file__).parent / "out"


def exact(x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    H = D0 * (1 - (x / L) ** 2)
    eta = 2 * ETA0 * D0 / L * x * np.cos(OMEGA * t) + ETA0**2 * D0 * np.sin(OMEGA * t) ** 2
    wet = H + eta > 0
    return np.where(wet, eta, -H), wet


def shoreline(x: np.ndarray, depth: np.ndarray, dmin: float = 0.05) -> tuple[float, float]:
    wet = depth > dmin + 1e-5                    # CROCO stores dry cells at depth Dcrit in float32
    return (x[wet].min(), x[wet].max()) if wet.any() else (np.nan, np.nan)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--croco", type=Path)
    ap.add_argument("--theta", type=float, default=0.55)
    ap.add_argument("--dx", type=float, default=1000.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--hours", type=float, default=72.0)
    ap.add_argument("--dmin", type=float, default=0.01, help="dry threshold [m] (CROCO Dcrit = 0.01 for THACKER)")
    ap.add_argument("--every", type=int, default=30, help="output every N steps (CROCO NWRT = 30)")
    args = ap.parse_args()

    nx = int(round(200e3 / args.dx))
    hg = create_cartesian_grid(nx=nx, ny=1, dx=args.dx, dy=args.dx, f0=0.0)
    x = -100e3 + (np.arange(nx) + 0.5) * args.dx
    H = D0 * (1 - (x / L) ** 2)
    fs = WetDryFreeSurface(hg, H[None, :], dt=args.dt, theta=args.theta, g=G, d_min=args.dmin)
    eta = exact(x, 0.0)[0][None, :]
    u, v = np.zeros((1, nx + 1)), np.zeros((2, nx))
    nsteps = int(round(args.hours * 3600 / args.dt))
    ts, etas, us = [0.0], [eta[0].copy()], [u[0].copy()]
    vol0 = np.sum(np.maximum(H + eta[0], 0))
    vol_err = clip = 0.0
    for n in range(nsteps):
        eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
        clip += fs.last_clip_volume
        vol_err = max(vol_err, abs(np.sum(np.maximum(H + eta[0], 0)) - vol0) / vol0)
        if (n + 1) % args.every == 0:
            ts.append((n + 1) * args.dt)
            etas.append(eta[0].copy())
            us.append(u[0].copy())
    ts, etas = np.array(ts), np.array(etas)
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / "thacker_pyocean.npz", t=ts, x=x, eta=etas, u=np.array(us), H=H)

    ex = np.array([exact(x, t)[0] for t in ts])
    wet = np.array([exact(x, t)[1] for t in ts])
    err = np.where(wet, etas - ex, np.nan)
    sh_mod = np.array([shoreline(x, H + e, args.dmin) for e in etas])
    sh_ex = np.array([shoreline(x, H + e, args.dmin) for e in ex])
    lines = [f"pyOCEAN Thacker: dx {args.dx:.0f} m, dt {args.dt:.0f} s, theta {args.theta}, {args.hours:.0f} h "
             f"({args.hours * 3600 * OMEGA / 2 / np.pi:.2f} periods)",
             f"  η error in the wet region: rms {100 * np.sqrt(np.nanmean(err ** 2)):.2f} cm, "
             f"max {100 * np.nanmax(np.abs(err)):.2f} cm (wet η range ±{np.nanmax(np.abs(np.where(wet, ex, np.nan))):.2f} m)",
             f"  last period rms {100 * np.sqrt(np.nanmean(err[ts > ts[-1] - 2 * np.pi / OMEGA] ** 2)):.2f} cm",
             f"  shoreline error (right/left): rms {np.sqrt(np.nanmean((sh_mod[:, 1] - sh_ex[:, 1]) ** 2)) / 1e3:.2f} / "
             f"{np.sqrt(np.nanmean((sh_mod[:, 0] - sh_ex[:, 0]) ** 2)) / 1e3:.2f} km",
             f"  volume error max {vol_err:.1e}; clipped volume {clip:.1e} m³ per m of channel"]

    croco = None
    if args.croco and args.croco.exists():
        import xarray as xr
        ds = xr.open_dataset(args.croco, decode_times=False)
        # interior cells xi 1..Lm at the middle eta row (0 and Lm+1 are ghost points)
        tc = ds["scrum_time"].values
        xc = ds["x_rho"].values[1, 1:-1]
        hc = ds["h"].values[1, 1:-1]
        dcrit = float(ds["Dcrit"].values) if "Dcrit" in ds else 0.01
        zc = ds["zeta"].values[:, 1, 1:-1]
        # requires CROCO built with ZETA_DRY_IO: otherwise wrt_his.F writes ζ + h on some dry cells
        zc = np.where(hc + zc > dcrit + 1e-5, zc, -hc)                 # dry (depth = Dcrit): surface on the bed
        exc = np.array([exact(xc, t)[0] for t in tc])
        wetc = np.array([exact(xc, t)[1] for t in tc])
        errc = np.where(wetc, zc - exc, np.nan)
        shc = np.array([shoreline(xc, hc + z, dcrit) for z in zc])
        shec = np.array([shoreline(xc, hc + z, dcrit) for z in exc])
        lines += [f"CROCO ({args.croco.name}): {len(tc)} records",
                  f"  η error in the wet region: rms {100 * np.sqrt(np.nanmean(errc ** 2)):.2f} cm, "
                  f"max {100 * np.nanmax(np.abs(errc)):.2f} cm",
                  f"  shoreline error (right/left): rms {np.sqrt(np.nanmean((shc[:, 1] - shec[:, 1]) ** 2)) / 1e3:.2f} / "
                  f"{np.sqrt(np.nanmean((shc[:, 0] - shec[:, 0]) ** 2)) / 1e3:.2f} km",
                  f"  last period rms {100 * np.sqrt(np.nanmean(errc[tc > tc[-1] - 2 * np.pi / OMEGA] ** 2)):.2f} cm"]
        croco = (tc, xc, zc, hc)

    fig, ax = plt.subplots(2, 1, figsize=(11, 8))
    for k, frac in enumerate((0.0, 0.25, 0.5, 0.75)):
        t = ts[-1] - (1 - frac) * 2 * np.pi / OMEGA
        i = int(np.argmin(np.abs(ts - t)))
        ax[0].plot(x / 1e3, etas[i], f"C{k}", lw=1.5, label=f"pyOCEAN t={ts[i] / 3600:.1f} h")
        ax[0].plot(x / 1e3, ex[i], f"C{k}--", lw=1)
        if croco is not None:
            j = int(np.argmin(np.abs(croco[0] - ts[i])))
            ax[0].plot(croco[1] / 1e3, croco[2][j], f"C{k}:", lw=1.5)
    ax[0].plot(x / 1e3, -H, "k", lw=1)
    ax[0].set_ylim(-3, 3)
    ax[0].set_xlabel("x [km]")
    ax[0].set_ylabel("η [m]  (solid pyOCEAN, dashed exact, dotted CROCO)")
    ax[0].legend(fontsize=7)
    ax[1].plot(ts / 3600, sh_mod[:, 1] / 1e3, "C0", label="pyOCEAN right shoreline")
    ax[1].plot(ts / 3600, sh_ex[:, 1] / 1e3, "k--", label="exact")
    if croco is not None:
        ax[1].plot(croco[0] / 3600, shc[:, 1] / 1e3, "C3:", label="CROCO")
    ax[1].set_xlabel("t [h]")
    ax[1].set_ylabel("x_shore [km]")
    ax[1].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / "thacker.png", dpi=110)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
