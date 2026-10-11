#!/usr/bin/env python
"""
Tidal flat wetting–drying benchmark — CROCO's TIDAL_FLAT case run as a 2DH
(barotropic) problem with pyOCEAN and with CROCO, compared cell by cell.

There is no exact solution: this is a cross-model test of the moving
shoreline under a strong tide with nonlinear momentum and bottom drag.

Setup (CROCO OCEAN/ana_grid.F, analytical.F, TEST_CASES/croco.in.Tidal_flat):
    channel 100 km, 200 cells of 500 m, f = 0
    bed: h₁ = 16 m, hᵢ = 16 − 20 (i − 1)/201 m for i ≥ 2 (≈ −3.9 m at the east end;
         the last ~19 km are a flat that floods and drains)
    west boundary: η_ext = 2 sin(2πt / 12 h). CROCO OBC_M2CHARACT + OBC_REDUCED_PHYSICS ignores
         ū_ext = 0 and uses the interior velocity (u2dbc.F), i.e. a Flather condition
         u_b = u_int − √(g/D)(η − η_ext) that pins η to η_ext; pyOCEAN: Flather on the total depth
         with ū_ext = u of the first interior face at the previous step; walls elsewhere
    quadratic drag C_d = 2.5e-3 on the total depth, no viscosity,
    momentum advection third-order upwind (CROCO UV_HADV_UP3, pyOCEAN upwind3)
    dry threshold 0.1 m (CROCO D_wetdry), Δt = 15 s, 36 h (3 tidal periods)
The CROCO run drops SOLVE3D/GLS/MUSTANG from the original 3D case; see README.md.

Usage:  python validation/tidal_flat/run_tidal_flat.py [--croco PATH/tidal_flat_his.nc] [--theta 0.55]
Output: validation/tidal_flat/out/tidal_flat_pyocean.npz, tidal_flat.png, summary.txt
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

from core.dynamics.nonlinear import advection_u  # noqa: E402
from core.dynamics.sigma_wd import SigmaWetDry  # noqa: E402
from core.dynamics.wetting_drying import WetDryFreeSurface  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from physics.bottom_friction import log_law_cd  # noqa: E402

G, NX, DX = 9.81, 200, 500.0
H_DEEP, H_SHALLOW = 16.0, -4.0
ETA_B, T_TIDE = 2.0, 12.0 * 3600.0
STATIONS_KM = (10.0, 50.0, 75.0, 80.0, 82.5, 85.0, 87.5, 90.0)
OUT = Path(__file__).parent / "out"


def croco_bed() -> np.ndarray:
    """Bed depth of the interior cells i = 1..200 as in CROCO ana_grid.F [m, < 0 = land]."""
    i = np.arange(1, NX + 1, dtype=float)
    return np.where(i < 2, H_DEEP, H_DEEP - (i - 1) / (NX + 1) * (H_DEEP - H_SHALLOW))


def shoreline(x: np.ndarray, depth: np.ndarray, dmin: float) -> float:
    """Eastmost wet cell centre [m] (depth > dmin, CROCO stores dry cells at Dcrit in float32)."""
    wet = depth > dmin + 1e-5
    return x[wet].max() if wet.any() else np.nan


def run(theta: float, dt: float, hours: float, dmin: float, cd: float, adv: str | None, every: int,
        z0: float | None = None):
    hg = create_cartesian_grid(nx=NX, ny=1, dx=DX, dy=DX, f0=0.0)
    H = croco_bed()[None, :]
    fl_u = np.zeros((1, NX + 1), bool)
    fl_u[0, 0] = True
    fs = WetDryFreeSurface(hg, H, dt=dt, theta=theta, g=G, d_min=dmin, flather_u=fl_u)
    eta = np.where(H > dmin, 0.0, dmin - H)              # dry cells start at the threshold (CROCO ana_initial)
    u, v = np.zeros((1, NX + 1)), np.zeros((2, NX))
    H_u = fs.H_u
    nsteps = int(round(hours * 3600 / dt))
    ts, etas, us = [0.0], [eta[0].copy()], [u[0].copy()]
    clip = 0.0
    for n in range(nsteps):
        t_mid = (n + 0.5) * dt
        # total depth at the faces (mean of the two cells), as the drag sees it
        D_u = np.full(u.shape, np.nan)
        D_u[:, 1:-1] = 0.5 * (H[:, :-1] + eta[:, :-1] + H[:, 1:] + eta[:, 1:])
        D_u[:, 0] = H[:, 0] + eta[:, 0]
        wet_u = np.isfinite(H_u) & np.isfinite(D_u) & (D_u > dmin)
        tend = -advection_u(u, np.zeros((2, NX)), hg, wet_u, scheme=adv) if adv else 0.0
        # Quadratic bottom drag on the total depth, implicit:  ∂u/∂t = −C_d |u| u / D
        Dd = np.maximum(np.nan_to_num(D_u, nan=dmin), dmin)
        # C_d constant, or log-law C_d(D) (Soulsby 1997) as seen by CROCO's 3-D run with z0
        r = (cd if z0 is None else log_law_cd(Dd, z0)) * np.abs(u) / Dd
        u_hat = (u + dt * tend) / (1.0 + dt * r)
        eta_ext = np.full(u.shape, ETA_B * np.sin(2 * np.pi * t_mid / T_TIDE))
        eta, u, v = fs.step(eta, u, v, u_hat, v.copy(), flather_eta=eta_ext,
                              flather_ubar=np.full(u.shape, u[0, 1]))
        clip += fs.last_clip_volume
        if not np.isfinite(eta).all():
            raise RuntimeError(f"blow-up at step {n}")
        if (n + 1) % every == 0:
            ts.append((n + 1) * dt)
            etas.append(eta[0].copy())
            us.append(u[0].copy())
    return np.array(ts), np.array(etas), np.array(us), H[0], clip


def run3d(theta: float, dt: float, hours: float, dmin: float, z0: float, adv: str | None, every: int,
          n_layers: int, closure: str = "parabolic"):
    """
    Same case with the homogeneous σ-layer model (``SigmaWetDry``): N equal layers,
    first-layer log-law bed stress with z0, vertical viscosity ``closure``,
    horizontal advection per layer (no vertical advection yet).
    Returns ts, η, depth-mean ū, H, clipped volume and the layer velocities.
    """
    hg = create_cartesian_grid(nx=NX, ny=1, dx=DX, dy=DX, f0=0.0)
    H = croco_bed()[None, :]
    fl_u = np.zeros((1, NX + 1), bool)
    fl_u[0, 0] = True
    m = SigmaWetDry(hg, H, dt=dt, n_layers=n_layers, z0=z0, closure=closure, theta=theta, g=G, d_min=dmin,
                    flather_u=fl_u)
    eta = np.where(H > dmin, 0.0, dmin - H)
    u, v = np.zeros((n_layers, 1, NX + 1)), np.zeros((n_layers, 2, NX))
    ds = m.dsigma[:, None, None]
    nsteps = int(round(hours * 3600 / dt))
    ts, etas, us, uk = [0.0], [eta[0].copy()], [u.sum(0)[0] * 0], [u[:, 0].copy()]
    clip = 0.0
    for n in range(nsteps):
        t_mid = (n + 0.5) * dt
        Du, _ = m.face_depths(eta)
        wet_u = np.isfinite(m.fs.H_u) & np.isfinite(Du) & (Du > dmin)
        G_u = None
        if adv:
            G_u = np.stack([-advection_u(u[k], np.zeros((2, NX)), hg, wet_u, scheme=adv) for k in range(n_layers)])
        ubar = (ds * u).sum(0)
        eta_ext = np.full(ubar.shape, ETA_B * np.sin(2 * np.pi * t_mid / T_TIDE))
        eta, u, v = m.step(eta, u, v, G_u=G_u, flather_eta=eta_ext, flather_ubar=np.full(ubar.shape, ubar[0, 1]))
        clip += m.fs.last_clip_volume
        if not np.isfinite(eta).all():
            raise RuntimeError(f"blow-up at step {n}")
        if (n + 1) % every == 0:
            ts.append((n + 1) * dt)
            etas.append(eta[0].copy())
            us.append((ds * u).sum(0)[0])
            uk.append(u[:, 0].copy())
    return np.array(ts), np.array(etas), np.array(us), H[0], clip, np.array(uk)


def read_croco(path: Path) -> dict:
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    j = ds.sizes["eta_rho"] // 2                                      # middle row (rows are identical)
    h = ds["h"].values[j, 1:-1]
    dcrit = float(ds["Dcrit"].values) if "Dcrit" in ds else 0.1
    z = ds["zeta"].values[:, j, 1:-1]
    z = np.where(h + z > dcrit + 1e-5, z, dcrit - h)                   # requires ZETA_DRY_IO
    ub = ds["ubar"].values[:, j, :]                                    # faces i = 1..200 (west face at index 0)
    return dict(t=ds["scrum_time"].values, x=ds["x_rho"].values[j, 1:-1], h=h, zeta=z, ubar=ub, dcrit=dcrit)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--croco", type=Path)
    ap.add_argument("--theta", type=float, default=0.55)
    ap.add_argument("--dt", type=float, default=15.0)
    ap.add_argument("--hours", type=float, default=36.0)
    ap.add_argument("--dmin", type=float, default=0.1)
    ap.add_argument("--cd", type=float, default=2.5e-3)
    ap.add_argument("--z0", type=float, help="log-law bed roughness [m] instead of a constant C_d (CROCO 3D: 1e-4)")
    ap.add_argument("--adv", default="upwind3", help="'upwind3', 'centred' or 'none'")
    ap.add_argument("--every", type=int, default=60, help="output every N steps (CROCO NWRT = 60)")
    ap.add_argument("--layers", type=int, default=0, help="σ-layers (0 = 2-D model); needs --z0")
    ap.add_argument("--closure", default="parabolic", help="vertical viscosity: gls, parabolic, constant, none")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    adv = None if args.adv == "none" else args.adv
    uk = None
    if args.layers:
        if args.z0 is None:
            raise SystemExit("--layers needs --z0")
        ts, etas, us, H, clip, uk = run3d(args.theta, args.dt, args.hours, args.dmin, args.z0, adv, args.every,
                                          args.layers, args.closure)
    else:
        ts, etas, us, H, clip = run(args.theta, args.dt, args.hours, args.dmin, args.cd, adv, args.every,
                                    args.z0)
    x = (np.arange(NX) + 0.5) * DX
    OUT.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(OUT / f"tidal_flat_pyocean{args.tag}.npz", t=ts, x=x, eta=etas, u=us, H=H,
                        **({} if uk is None else {"u_layers": uk}))
    sh = np.array([shoreline(x, H + e, args.dmin) for e in etas])
    last = ts > ts[-1] - T_TIDE
    model = f"σ-layers {args.layers} ({args.closure})" if args.layers else "2-D"
    lines = [f"pyOCEAN tidal flat [{model}]: dx {DX:.0f} m, dt {args.dt:.0f} s, theta {args.theta}, dmin {args.dmin} m, "
             f"{f'z0 {args.z0} m (log law)' if args.z0 else f'Cd {args.cd}'}, advection {args.adv}, {args.hours:.0f} h",
             f"  shoreline (last period): {np.nanmin(sh[last]) / 1e3:.2f}–{np.nanmax(sh[last]) / 1e3:.2f} km; "
             f"clipped volume {clip:.1e} m³ per channel width"]

    c = None
    if args.croco and args.croco.exists():
        c = read_croco(args.croco)
        if not np.allclose(c["h"], H, atol=1e-4):
            raise ValueError("CROCO bed differs from croco_bed()")
        k = [int(np.argmin(np.abs(c["t"] - t))) for t in ts]
        if np.max(np.abs(c["t"][k] - ts)) > 1.0:
            raise ValueError("CROCO and pyOCEAN output times do not match")
        zc, uc = c["zeta"][k], c["ubar"][k]
        shc = np.array([shoreline(x, H + z, c["dcrit"]) for z in zc])
        wet_both = (H + etas > args.dmin + 1e-5) & (H + zc > args.dmin + 1e-5)
        d = np.where(wet_both, etas - zc, np.nan)
        lines += [f"CROCO ({args.croco.name}), last tidal period:",
                  f"  CROCO shoreline {np.nanmin(shc[last]) / 1e3:.2f}–{np.nanmax(shc[last]) / 1e3:.2f} km; "
                  f"shoreline difference rms {np.sqrt(np.nanmean((sh - shc)[last] ** 2)) / 1e3:.2f} km, "
                  f"max {np.nanmax(np.abs(sh - shc)[last]) / 1e3:.2f} km",
                  f"  η difference where both wet: rms {100 * np.sqrt(np.nanmean(d[last] ** 2)):.1f} cm, "
                  f"max {100 * np.nanmax(np.abs(d[last])):.1f} cm",
                  "  stations  x[km]  η range py/CROCO [m]   η rms diff [cm]   ū max py/CROCO [m/s]   ū rms diff [m/s]"]
        for xs in STATIONS_KM:
            i = int(np.argmin(np.abs(x - xs * 1e3)))
            ep, ec = etas[last, i], zc[last, i]
            up, ucs = 0.5 * (us[last, i] + us[last, i + 1]), 0.5 * (uc[last, i] + uc[last, i + 1])
            lines.append(f"           {xs:6.1f}  {np.ptp(ep):5.2f} / {np.ptp(ec):5.2f}        "
                         f"{100 * np.sqrt(np.mean((ep - ec) ** 2)):6.1f}           "
                         f"{np.abs(up).max():5.2f} / {np.abs(ucs).max():5.2f}           "
                         f"{np.sqrt(np.mean((up - ucs) ** 2)):5.3f}")

    fig, ax = plt.subplots(3, 1, figsize=(11, 11))
    for k_, frac in enumerate((0.0, 0.25, 0.5, 0.75)):
        i = int(np.argmin(np.abs(ts - (ts[-1] - (1 - frac) * T_TIDE))))
        ax[0].plot(x / 1e3, np.where(H + etas[i] > args.dmin, etas[i], np.nan), f"C{k_}", lw=1.5,
                   label=f"t = {ts[i] / 3600:.1f} h")
        if c is not None:
            ax[0].plot(x / 1e3, np.where(H + zc[i] > args.dmin, zc[i], np.nan), f"C{k_}:", lw=1.5)
    ax[0].fill_between(x / 1e3, -H, -H_DEEP - 1, color="0.8")
    ax[0].set_xlim(60, 100)
    ax[0].set_ylim(-4.5, 2.5)
    ax[0].set_xlabel("x [km]")
    ax[0].set_ylabel("η [m] (solid pyOCEAN, dotted CROCO)")
    ax[0].legend(fontsize=7)
    ax[1].plot(ts / 3600, sh / 1e3, "C0", label="pyOCEAN")
    if c is not None:
        ax[1].plot(ts / 3600, shc / 1e3, "C3:", label="CROCO")
    ax[1].set_ylabel("eastmost wet cell [km]")
    ax[1].legend(fontsize=8)
    i = int(np.argmin(np.abs(x - 85e3)))
    ax[2].plot(ts / 3600, etas[:, i], "C0", label="η pyOCEAN, x = 85 km")
    ax[2].plot(ts / 3600, 0.5 * (us[:, i] + us[:, i + 1]), "C2", label="ū pyOCEAN")
    if c is not None:
        ax[2].plot(ts / 3600, zc[:, i], "C0:", label="η CROCO")
        ax[2].plot(ts / 3600, 0.5 * (uc[:, i] + uc[:, i + 1]), "C2:", label="ū CROCO")
    ax[2].set_xlabel("t [h]")
    ax[2].legend(fontsize=8)
    fig.tight_layout()
    fig.savefig(OUT / f"tidal_flat{args.tag}.png", dpi=110)
    (OUT / f"summary{args.tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
