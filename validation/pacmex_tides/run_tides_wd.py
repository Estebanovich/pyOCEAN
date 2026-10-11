#!/usr/bin/env python
"""
Northern Gulf of California tidal nest with wetting and drying.

Grid       1/120° (~0.9 km), 2×2 block mean of GEBCO 2021 (15″) — raw, with the
           intertidal flats of the Colorado delta (no depth floor); land above
           +8 m is a permanent wall; only the Gulf connected component is kept
Dynamics   linear momentum (Coriolis, body force) + quadratic drag on the total
           depth + JSL drag; continuity with the total depth and wetting/drying
           (core.dynamics.wetting_drying, Casulli 1990, CG solver)
Tides      Flather on the deep (> 20 m) edge faces with the harmonic constants of a
           1/25° parent run (--bc parent:…), body tide − SAL, ramp 2 days
Control    --floor 5 floors the depth at 5 m (as the HYCOM-IOA bathymetry): no
           drying, same solver — isolates the effect of the intertidal topography

Usage:  python run_tides_wd.py --bc parent:out/run_s_min5.npz --tag wd_raw
        python run_tides_wd.py --bc parent:out/run_s_min5.npz --floor 5 --tag wd_floor5
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402
from scipy import ndimage  # noqa: E402

sys.path.insert(0, str(Path(__file__).parent))
import run_tides as RT  # noqa: E402
from core.dynamics.free_surface import open_boundary_hfac  # noqa: E402
from core.dynamics.nonlinear import cell_to_faces  # noqa: E402
from core.dynamics.wetting_drying import WetDryFreeSurface  # noqa: E402
from core.grid.horizontal import create_spherical_grid  # noqa: E402
from core.grid.vertical import compute_hfac, create_z_grid  # noqa: E402
from forcing.sources.hycom_ab import read_field  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import (  # noqa: E402
    body_force,
    boundary_tide_constants,
    boundary_tides_at,
    equilibrium_constants,
    predict,
    read_sal_hycom,
)
from physics.tides.drag import jsl_to_rh, regrid_bilinear, rh_at_faces, tidal_drag_rate  # noqa: E402

GEBCO = RT.REFS / "gebco2021_alto_golfo" / "gebco2021_golfo_norte.nc"
BOX = dict(lon=(-115.3, -110.5), lat=(27.5, 31.95))
SEED = (29.5, -113.0)                  # a point of the Gulf (connected component)
WALL_ELEV = 8.0                        # m: land higher than this never floods
DLL = 1.0 / 120.0


def build_grid(floor: float | None) -> dict:
    e = xr.open_dataset(GEBCO)["elevation"].sel(latitude=slice(*BOX["lat"]), longitude=slice(*BOX["lon"]))
    ny, nx = (e.shape[0] // 2) * 2, (e.shape[1] // 2) * 2
    E = e.values[:ny, :nx].astype(float).reshape(ny // 2, 2, nx // 2, 2).mean(axis=(1, 3))
    lat = e.latitude.values[:ny].reshape(-1, 2).mean(axis=1)
    lon = e.longitude.values[:nx].reshape(-1, 2).mean(axis=1)
    H = -E
    H[E > WALL_ELEV] = np.nan
    lab, _ = ndimage.label(np.isfinite(H))
    js, is_ = np.argmin(np.abs(lat - SEED[0])), np.argmin(np.abs(lon - SEED[1]))
    H[lab != lab[js, is_]] = np.nan                              # Gulf only (drop the Pacific side)
    if floor is not None:
        H = np.where(np.isfinite(H), np.maximum(H, floor), np.nan)
    return dict(H=H, lat0=float(lat[0]) - DLL / 2, lon0=float(lon[0]) - DLL / 2)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--days", type=float, default=33.0)
    ap.add_argument("--spinup", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=60.0)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--cd", type=float, default=RT.CD)
    ap.add_argument("--drgscl", type=float, default=0.5)
    ap.add_argument("--floor", type=float, default=None, help="depth floor [m] (control run, no drying)")
    ap.add_argument("--d-min", type=float, default=0.05)
    ap.add_argument("--bc", required=True, help="parent:<run_*.npz> or tpxo")
    ap.add_argument("--benchmark", type=int, default=0)
    ap.add_argument("--tag", default="wd")
    args = ap.parse_args()

    t_setup = time.time()
    gr = build_grid(args.floor)
    H = gr["H"]
    ny, nx = H.shape
    act = np.isfinite(H)
    hg = create_spherical_grid(nx, ny, DLL, DLL, lon0=gr["lon0"], lat0=gr["lat0"])
    lon_t, lat_t = hg["lon_t"].values, hg["lat_t"].values

    # open (Flather) faces: deep edge faces of the Gulf component
    deep = act & (np.nan_to_num(H) > 20.0)
    fl_u = np.zeros((ny, nx + 1), bool)
    fl_v = np.zeros((ny + 1, nx), bool)
    fl_u[:, 0], fl_u[:, -1] = deep[:, 0], deep[:, -1]
    fl_v[0, :], fl_v[-1, :] = deep[0, :], deep[-1, :]
    fs = WetDryFreeSurface(hg, H, dt=args.dt, theta=args.theta, g=RT.G, d_min=args.d_min,
                           flather_u=fl_u, flather_v=fl_v, solver="cg", cg_rtol=1e-10)

    # boundary constants (transport / model depth at the open faces)
    vg = create_z_grid(1, [float(np.nanmax(H))])
    hf = open_boundary_hfac(compute_hfac(vg, np.where(act, np.maximum(np.nan_to_num(H), 0.0), 0.0), hfac_min=0.0),
                            ["west", "east", "south", "north"])
    tc = read_tpxo([RT.ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in RT.CONS],
                   [RT.ATLAS / f"u_{c}_tpxo10_atlas_30_v2.nc" for c in RT.CONS], RT.ATLAS / "grid_tpxo10atlas_v2.nc",
                   lon_range=(BOX["lon"][0] - 0.5, BOX["lon"][1] + 0.5),
                   lat_range=(BOX["lat"][0] - 0.5, BOX["lat"][1] + 0.5))
    if args.bc.startswith("parent:"):
        tc = RT.parent_constants(Path(args.bc.split(":", 1)[1]), tc)
    bcon = boundary_tide_constants(tc, hg, vg, hf, fl_u, fl_v)

    # body tide − SAL, drag coefficients, JSL — interpolated from the PAC_MEX / global grids
    lon1, lat1 = RT._pacmex_axes()
    sal_full = read_sal_hycom(RT.SAL_B, RT.IDM, RT.JDM)
    sal = np.stack([regrid_bilinear(sal_full[..., k].real, lat1, lon1, lat_t, lon_t)
                    + 1j * regrid_bilinear(sal_full[..., k].imag, lat1, lon1, lat_t, lon_t)
                    for k in range(sal_full.shape[-1])], axis=-1)
    body_flat = (equilibrium_constants(lon_t, lat_t, RT.CONS) - sal).reshape(-1, len(RT.CONS))
    cb = read_field(RT.HYCOM / "relax" / "DRAG" / "cb.b", "CB", idm=RT.IDM, jdm=RT.JDM)
    cd_t = regrid_bilinear(np.where(np.isfinite(cb), cb, args.cd), lat1, lon1, lat_t, lon_t)
    cd_u, cd_v = rh_at_faces(cd_t, act)
    r_jsl_u, r_jsl_v = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    if args.drgscl > 0:
        glon = read_field(RT.GLB / "topo/topo/regional.grid.b", "plon")[0]
        glat = read_field(RT.GLB / "topo/topo/regional.grid.b", "plat")[:, 0]
        jsl = regrid_bilinear(read_field(RT.GLB / "relax_DRAG/relax/DRAG/JSL.b", "JSL", idm=4500, jdm=2251),
                              glat, glon, lat_t, lon_t)
        Hpos = np.where(act & (np.nan_to_num(H) > 0), H, np.nan)
        r_jsl_u, r_jsl_v = rh_at_faces(tidal_drag_rate(jsl_to_rh(jsl, Hpos), np.nan_to_num(Hpos), args.drgscl), act)
    f_u = 2 * RT.OMEGA_E * np.sin(np.deg2rad(hg["lat_u"].values))
    f_v = 2 * RT.OMEGA_E * np.sin(np.deg2rad(hg["lat_v"].values))
    face_u = np.isfinite(fs.H_u) | fl_u
    face_v = np.isfinite(fs.H_v) | fl_v

    gauges = RT.stations(hg, act & (np.nan_to_num(H) > 0), BOX, DLL)
    print(f"grid {nx}×{ny}, active {act.sum()}, floodable {np.sum(act & (np.nan_to_num(H) <= 0))}, "
          f"Flather faces {fl_u.sum() + fl_v.sum()}, {len(gauges)} gauges; setup {time.time() - t_setup:.0f} s",
          flush=True)

    t0 = np.datetime64(args.start, "s")
    dt = args.dt
    nsteps = args.benchmark or int(round(args.days * 86400 / dt))
    n_spin = int(round(args.spinup * 86400 / dt))
    every_g, every_ha = int(round(600 / dt)), int(round(3600 / dt))
    tau_ramp = 2 * 86400.0 / 5.0

    eta = np.where(act, np.where(np.nan_to_num(H) > 0, 0.0, -np.nan_to_num(H)), 0.0)
    u, v = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    g_prev = None
    nb = 1 + 2 * len(RT.HA_CONS)
    ata, atb = np.zeros((nb, nb)), np.zeros((nb, ny * nx))
    wet_hours = np.zeros((ny, nx))
    n_ha = 0
    g_t, g_eta = [], []
    H0 = np.nan_to_num(H)
    tic = time.time()
    for n in range(nsteps):
        t_mid = t0 + np.timedelta64(int((n + 0.5) * dt), "s")
        ramp = 1.0 - np.exp(-(n + 0.5) * dt / tau_ramp)
        eta_b = ramp * predict(np.array([t_mid]), body_flat, RT.CONS)[0].reshape(ny, nx)
        gx, gy = body_force(hg, eta_b, g=RT.G)
        v_at_u = np.zeros_like(u)
        v_at_u[:, 1:-1] = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
        u_at_v = np.zeros_like(v)
        u_at_v[1:-1, :] = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
        gu = np.where(face_u, f_u * v_at_u + gx, 0.0)
        gv = np.where(face_v, -f_v * u_at_v + gy, 0.0)
        gu_ab, gv_ab = (gu, gv) if g_prev is None else (1.5 * gu - 0.5 * g_prev[0], 1.5 * gv - 0.5 * g_prev[1])
        g_prev = (gu, gv)
        # drag on the total depth (faces), floored at d_min
        eta_u, eta_v = cell_to_faces(eta + H0, act)                  # total depth averaged to faces
        Du, Dv = np.maximum(eta_u, args.d_min), np.maximum(eta_v, args.d_min)
        r_u = cd_u * np.hypot(u, v_at_u) / Du + r_jsl_u
        r_v = cd_v * np.hypot(v, u_at_v) / Dv + r_jsl_v
        u_hat = (u + dt * gu_ab) / (1.0 + dt * r_u)
        v_hat = (v + dt * gv_ab) / (1.0 + dt * r_v)
        bt = boundary_tides_at(bcon, t_mid, ramp=ramp)
        eta, u, v = fs.step(eta, u, v, u_hat, v_hat, flather_eta=(bt["eta_u"], bt["eta_v"]),
                            flather_ubar=bt["ubar_u"], flather_vbar=bt["vbar_v"])
        if not np.isfinite(eta).all() or np.abs(np.where(act, eta, 0)).max() > 20:
            raise RuntimeError(f"blow-up at step {n}")
        t_new = t0 + np.timedelta64(int((n + 1) * dt), "s")
        if (n + 1) % every_g == 0:
            g_t.append(t_new)
            g_eta.append([eta[j, i] for j, i in gauges.values()])
        if n + 1 > n_spin and (n + 1) % every_ha == 0:
            a = RT.ha_basis(t_new)
            ata += np.outer(a, a)
            atb += a[:, None] * eta.ravel()[None, :]
            wet_hours += (H0 + eta) > args.d_min
            n_ha += 1
        if args.benchmark and n == 4:
            tic5 = time.time()
        if (n + 1) % (every_ha * 24) == 0:
            el = time.time() - tic
            print(f"day {(n + 1) * dt / 86400:5.1f}  max|η| {np.abs(np.where(act, eta, 0)).max():.2f} m  "
                  f"max|u| {np.abs(u).max():.2f} m/s  wet {np.sum(act & ((H0 + eta) > args.d_min))}  "
                  f"CG its {fs.cg_iterations}  {el / 60:.1f} min (ETA {el / (n + 1) * (nsteps - n - 1) / 60:.0f} min)",
                  flush=True)
    if args.benchmark:
        per = (time.time() - tic5) / (nsteps - 5)
        print(f"benchmark: {per:.3f} s/step → {per * 33 * 86400 / dt / 3600:.2f} h for 33 days at Δt={dt:.0f} s")
        return
    x = np.linalg.solve(ata, atb)
    hc = (x[1::2] + 1j * x[2::2]).T.reshape(ny, nx, len(RT.HA_CONS))
    hc[~act] = np.nan
    RT.OUT.mkdir(exist_ok=True)
    np.savez_compressed(
        RT.OUT / f"run_{args.tag}.npz", gauge_keys=np.array(list(gauges)), gauge_ij=np.array(list(gauges.values())),
        gauge_t=np.array(g_t, "datetime64[s]"), gauge_eta=np.array(g_eta, np.float32),
        hc_map=hc.astype(np.complex64), ha_cons=np.array(RT.HA_CONS), lon_t=lon_t, lat_t=lat_t, depth=H,
        wet_fraction=(wet_hours / max(n_ha, 1)).astype(np.float32),
        meta=json.dumps(vars(args) | dict(box=BOX, dll=DLL, wall_elev=WALL_ELEV)))
    print("saved", RT.OUT / f"run_{args.tag}.npz", f"total {(time.time() - t_setup) / 60:.1f} min")


if __name__ == "__main__":
    main()
