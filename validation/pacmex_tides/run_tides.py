#!/usr/bin/env python
"""
Phase 2, step 8: barotropic tidal run on a PAC_MEX subdomain (Mexican Pacific
and Gulf of California) on the HYCOM-IOA grid and bathymetry.

Configuration
-------------
Grid        PAC_MEX regional.grid (uniform 0.04°), subdomain 118–92°W, 12–32°N
Bathymetry  regional.depth_final (GLBy0.08 + GEBCO 1/25°, E. Cruz Isidro 2026),
            minimum depth 10 m (no wetting/drying), one layer with partial cells
Dynamics    linear shallow water on the sphere (Vallis 2006 §3.1):
              ∂u/∂t = f v − g ∇η + g ∇η_body − C_d |u| u / H
              ∂η/∂t = −∇·(H u)
            Coriolis + body force with AB2, quadratic bottom drag semi-implicit
            (C_d = 2.5e-3), implicit θ-method free surface (θ = 0.6, slight
            damping of grid-scale gravity waves; tides ωΔt ≈ 0.02 are unaffected)
Tides       8 constituents (m2 s2 k1 o1 n2 p1 k2 q1):
            - Flather radiation OBC on all open edges with TPXO10-atlas-v2
              elevation and transport (velocity = transport / model depth)
            - body tide α η_eq minus per-constituent SAL from the HYCOM-IOA file
              tidal.salReIm_tpxo10 (HYCOM tidein=2 convention, η_body = α η_eq − F)
            - ramp 1 − exp(−5t/τ), τ = 2 days (HYCOM tidrmp form; 0.9994 at day 3)
Output      out/run_<tag>.npz: η every 10 min at tide-gauge points; harmonic
            constants of M2 S2 K1 O1 N2 Q1 on the whole grid (online least
            squares after the spin-up; P1 and K2 cannot be separated in 30 days).

Usage
-----
    python run_tides.py --benchmark 40          # timing only
    python run_tides.py --start 2026-08-01 --days 33 --spinup 3
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")          # one P-core per run (SuperLU is single-threaded)

import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import xarray as xr  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dynamics.free_surface import ImplicitFreeSurface, open_boundary_hfac  # noqa: E402
from core.dynamics.nonlinear import (  # noqa: E402
    advection_u,
    advection_v,
    cell_to_faces,
    laplacian_u,
    laplacian_v,
    nonlinear_continuity,
)
from core.grid.horizontal import create_spherical_grid  # noqa: E402
from core.grid.vertical import compute_hfac, create_z_grid  # noqa: E402
from forcing.sources.hycom_ab import read_field  # noqa: E402
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from physics.tides.drag import jsl_to_rh, regrid_bilinear, rh_at_faces, tidal_drag_rate  # noqa: E402
from forcing.tides import (  # noqa: E402
    CONSTITUENTS,
    body_force,
    boundary_tide_constants,
    boundary_tides_at,
    equilibrium_constants,
    predict,
    read_sal_hycom,
)

HYCOM = Path.home() / "HYCOM"
REFS = Path.home() / "pyOCEAN_refs"
ATLAS = REFS / "TPXO10_atlas_v2_nc"
SAL_B = REFS / "tidal.salReIm_tpxo10.b"
IDM, JDM, DLL = 1600, 1125, 0.04
GLB = REFS / "hycom_GLBp0.08"                         # HYCOM-examples GLBp0.08: JSL.[ab] + regional.grid
BOX = dict(lon=(-118.0, -92.0), lat=(12.0, 32.0))
GEBCO100 = Path.home() / "Downloads" / "HYCOM_Oceanografia" / "gebco_100_NUEVA_final.nc"   # 1/100°, depth > 0, NaN land
MIN_DEPTH = 10.0
CD = 2.5e-3
G, OMEGA_E = 9.81, 7.2921e-5
CONS = list(CONSTITUENTS)
HA_CONS = ["m2", "s2", "k1", "o1", "n2", "q1"]          # separable in ~30 days
OUT = Path(__file__).parent / "out"


def _pacmex_axes() -> tuple[np.ndarray, np.ndarray]:
    plon = read_field(HYCOM / "regional.grid.b", "plon")
    plat = read_field(HYCOM / "regional.grid.b", "plat")
    return plon[0], plat[:, 0]


def subdomain(domain: str = "pacmex", box: dict | None = None) -> dict:
    """
    Depth and grid origin of the run domain.

    pacmex   : subdomain of the PAC_MEX grid (0.04°), HYCOM-IOA regional.depth_final;
               SAL and cb are sliced from the PAC_MEX files.
    gebco100 : nest on the 1/100° GEBCO grid (gebco_100_NUEVA_final); SAL and cb are
               bilinearly interpolated from the PAC_MEX files.
    """
    box = box or BOX
    lon1, lat1 = _pacmex_axes()
    if domain == "pacmex":
        i0, i1 = np.searchsorted(lon1, box["lon"][0]), np.searchsorted(lon1, box["lon"][1])
        j0, j1 = np.searchsorted(lat1, box["lat"][0]), np.searchsorted(lat1, box["lat"][1])
        depth = read_field(HYCOM / "regional.depth_final.b", "depth", idm=IDM, jdm=JDM)[j0:j1, i0:i1]
        sal = lambda: read_sal_hycom(SAL_B, IDM, JDM)[j0:j1, i0:i1]                       # noqa: E731
        cb = lambda: read_field(HYCOM / "relax" / "DRAG" / "cb.b", "CB", idm=IDM, jdm=JDM)[j0:j1, i0:i1]  # noqa: E731
        dll, lon0, lat0 = DLL, float(lon1[i0]) - DLL / 2, float(lat1[j0]) - DLL / 2
    elif domain == "gebco100":
        ds = xr.open_dataset(GEBCO100)
        sub = ds["elevation"].sel(lon=slice(*box["lon"]), lat=slice(*box["lat"]))
        depth = sub.values
        dll = 0.01
        lon0, lat0 = float(sub.lon[0]) - dll / 2, float(sub.lat[0]) - dll / 2
        i0 = j0 = -1
        LON, LAT = np.meshgrid(sub.lon.values, sub.lat.values)

        def sal():
            full = read_sal_hycom(SAL_B, IDM, JDM)
            return np.stack([regrid_bilinear(full[..., k].real, lat1, lon1, LAT, LON)
                             + 1j * regrid_bilinear(full[..., k].imag, lat1, lon1, LAT, LON)
                             for k in range(full.shape[-1])], axis=-1)

        def cb():
            full = read_field(HYCOM / "relax" / "DRAG" / "cb.b", "CB", idm=IDM, jdm=JDM)
            return regrid_bilinear(np.where(np.isfinite(full), full, CD), lat1, lon1, LAT, LON)
    else:
        raise ValueError(domain)
    depth = np.where(np.isfinite(depth) & (depth > 0), np.maximum(depth, MIN_DEPTH), 0.0)
    return dict(domain=domain, box=box, i0=i0, j0=j0, depth=depth, dll=dll, lon0=lon0, lat0=lat0, sal=sal, cb=cb)


def parent_constants(path: Path, tpxo: xr.Dataset) -> xr.Dataset:
    """
    Boundary constants from a parent run (one-way offline nesting, harmonic form):
    η and transports U = H u, V = H v of HA_CONS on the parent C grid, with P1 and K2
    inferred from K1 and S2 by the median complex TPXO ratio over the parent domain.
    Same layout as ``read_tpxo`` (h, U, V; NaN on land) for ``boundary_tide_constants``.
    """
    r = np.load(path)
    names = list(r["ha_cons"])
    h, U, V = r["hc_map"], r["hc_U"], r["hc_V"]
    ratio = {}
    for inf, ref in (("p1", "k1"), ("k2", "s2")):
        q = (tpxo.h.sel(constituent=inf) / tpxo.h.sel(constituent=ref)).values
        ratio[inf] = (ref, np.nanmedian(q.real) + 1j * np.nanmedian(q.imag))
    stack = {}
    for key, arr in (("h", h), ("U", U), ("V", V)):
        cols = [arr[..., names.index(c)] if c in names else ratio[c][1] * arr[..., names.index(ratio[c][0])]
                for c in CONS]
        stack[key] = np.moveaxis(np.array(cols), 0, 0)
    return xr.Dataset(
        dict(h=(("constituent", "lat", "lon"), stack["h"]),
             U=(("constituent", "lat", "lon_u"), stack["U"]),
             V=(("constituent", "lat_v", "lon"), stack["V"])),
        coords=dict(constituent=CONS, lat=r["lat_1d"], lon=r["lon_1d"], lon_u=r["lon_u_1d"], lat_v=r["lat_v_1d"]))


def stations(hg, wet: np.ndarray, box: dict, dll: float) -> dict:
    """Tide-gauge points: TICON-3 coastal records in the box → nearest wet T cell (≤ 3 cells)."""
    sys.path.insert(0, str(ROOT / "validation" / "tide_gauges"))
    from compare_ticon_tpxo import TICON  # noqa: E402
    from diagnostics.tide_gauges import load_ticon  # noqa: E402

    st = load_ticon(TICON, box["lon"], box["lat"], CONS)
    lon_t, lat_t = hg["lon_t"].values, hg["lat_t"].values
    jw, iw = np.nonzero(wet)
    out = {}
    for k, r in st.iterrows():
        d = np.hypot((lon_t[jw, iw] - r["lon"]) * np.cos(np.deg2rad(r["lat"])), lat_t[jw, iw] - r["lat"])
        n = int(np.argmin(d))
        if d[n] <= max(3 * dll, 0.12) * 1.01:                    # ≤ 3 cells (≤ 12 km on the nest)
            out[f"{r['lat']:.3f}_{r['lon']:.3f}_{r['start'].year}"] = (int(jw[n]), int(iw[n]))
    return out


def ha_basis(t: np.datetime64) -> np.ndarray:
    """Row of the least-squares design matrix [1, f cos θ, −f sin θ …] for HA_CONS at time t."""
    row = [1.0]
    for c in HA_CONS:
        for unit in (1.0, 1j):
            row.append(float(predict(np.array([t]), np.array([unit]), [c])[0]))
    return np.array(row)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2026-08-01")
    ap.add_argument("--days", type=float, default=33.0)
    ap.add_argument("--spinup", type=float, default=3.0)
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--theta", type=float, default=0.6)
    ap.add_argument("--no-sal", action="store_true", help="body tide without SAL (sensitivity)")
    ap.add_argument("--cd", type=float, default=CD, help="quadratic bottom drag coefficient")
    ap.add_argument("--cb-file", action="store_true",
                    help="spatial quadratic drag from ~/HYCOM/relax/DRAG/cb.[ab] (HYCOM-IOA, cb < 0)")
    ap.add_argument("--drgscl", type=float, default=0.0,
                    help="JSL internal-tide drag scale (HYCOM tiddrg=1; 0 = off)")
    ap.add_argument("--benchmark", type=int, default=0, help="run N steps and report timing")
    ap.add_argument("--tag", default="ref")
    ap.add_argument("--nonlinear", action="store_true",
                    help="total depth H+η in continuity and drag, momentum advection (core.dynamics.nonlinear)")
    ap.add_argument("--adv", default="upwind3", choices=["upwind3", "centred"],
                    help="momentum advection scheme with --nonlinear (centred is unstable in 1-cell channels)")
    ap.add_argument("--ah", type=float, default=0.0, help="Laplacian viscosity A_h [m²/s] (free slip)")
    ap.add_argument("--domain", default="pacmex", choices=["pacmex", "gebco100"])
    ap.add_argument("--box", nargs=4, type=float, metavar=("LON0", "LON1", "LAT0", "LAT1"))
    ap.add_argument("--bc", default="tpxo", help="'tpxo' or 'parent:<run_*.npz>' (harmonic one-way nesting)")
    ap.add_argument("--snapshots", nargs=2, metavar=("START", "END"),
                    help="also save hourly η fields between these times (e.g. to compare with HYCOM archives)")
    args = ap.parse_args()

    t_setup = time.time()
    box = dict(lon=tuple(args.box[:2]), lat=tuple(args.box[2:])) if args.box else BOX
    sd = subdomain(args.domain, box)
    ny, nx = sd["depth"].shape
    hg = create_spherical_grid(nx, ny, sd["dll"], sd["dll"], lon0=sd["lon0"], lat0=sd["lat0"])
    vg = create_z_grid(1, [float(sd["depth"].max())])
    hf = compute_hfac(vg, sd["depth"], hfac_min=0.0)
    hf = open_boundary_hfac(hf, ["west", "east", "south", "north"])
    hu = (vg.dz.values[:, None, None] * hf.hfac_u.values)[0]
    hv = (vg.dz.values[:, None, None] * hf.hfac_v.values)[0]
    wet_u, wet_v, wet_t = hu > 0, hv > 0, sd["depth"] > 0
    fl_u = np.zeros_like(wet_u)
    fl_u[:, [0, -1]] = wet_u[:, [0, -1]]
    fl_v = np.zeros_like(wet_v)
    fl_v[[0, -1], :] = wet_v[[0, -1], :]
    fs = ImplicitFreeSurface(hg, vg, hf, dt=args.dt, theta=args.theta, g=G, flather_u=fl_u, flather_v=fl_v)

    # ---- bottom drag coefficients at faces --------------------------------------
    cd_t = np.full((ny, nx), args.cd)
    if args.cb_file:
        cb = sd["cb"]()
        cd_t = np.where(np.isfinite(cb), cb, args.cd)
    cd_u, cd_v = rh_at_faces(cd_t, wet_t)
    r_jsl_u, r_jsl_v = np.zeros_like(hu), np.zeros_like(hv)
    if args.drgscl > 0:
        glon = read_field(GLB / "topo/topo/regional.grid.b", "plon")[0]
        glat = read_field(GLB / "topo/topo/regional.grid.b", "plat")[:, 0]
        jsl_g = read_field(GLB / "relax_DRAG/relax/DRAG/JSL.b", "JSL", idm=4500, jdm=2251)
        jsl = regrid_bilinear(jsl_g, glat, glon, hg["lat_t"].values, hg["lon_t"].values)
        # rate on the p-grid (HYCOM: drag on p-points, increment averaged to faces), then to faces
        rh = jsl_to_rh(jsl, np.where(wet_t, sd["depth"], np.nan))
        r_jsl_u, r_jsl_v = rh_at_faces(tidal_drag_rate(rh, sd["depth"], args.drgscl), wet_t)
        print(f"JSL drag: drgscl {args.drgscl}, max rate {r_jsl_u.max() * 86400:.2f}/day", flush=True)

    f_u = 2 * OMEGA_E * np.sin(np.deg2rad(hg["lat_u"].values))
    f_v = 2 * OMEGA_E * np.sin(np.deg2rad(hg["lat_v"].values))
    lon_t, lat_t = hg["lon_t"].values, hg["lat_t"].values

    tc = read_tpxo([ATLAS / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONS],
                   [ATLAS / f"u_{c}_tpxo10_atlas_30_v2.nc" for c in CONS], ATLAS / "grid_tpxo10atlas_v2.nc",
                   lon_range=(box["lon"][0] - 0.5, box["lon"][1] + 0.5),
                   lat_range=(box["lat"][0] - 0.5, box["lat"][1] + 0.5))
    body_hc = equilibrium_constants(lon_t, lat_t, CONS)                       # (ny, nx, nc)
    if not args.no_sal:
        body_hc = body_hc - sd["sal"]()
    body_flat = body_hc.reshape(-1, len(CONS))
    if args.bc.startswith("parent:"):
        tc = parent_constants(Path(args.bc.split(":", 1)[1]), tc)
    bcon = boundary_tide_constants(tc, hg, vg, hf, fl_u, fl_v)             # interpolate once

    gauges = stations(hg, wet_t, box, sd["dll"])
    print(f"grid {nx}×{ny}, wet {wet_t.mean():.2f}, Flather faces {fl_u.sum() + fl_v.sum()}, "
          f"{len(gauges)} gauges; setup {time.time() - t_setup:.0f} s", flush=True)

    t0 = np.datetime64(args.start, "s")
    dt = args.dt
    nsteps = args.benchmark or int(round(args.days * 86400 / dt))
    n_spin = int(round(args.spinup * 86400 / dt))
    every_g = int(round(600 / dt))
    every_ha = int(round(3600 / dt))
    tau_ramp = 2 * 86400.0 / 5.0

    eta = np.zeros((ny, nx))
    u = np.zeros((1, ny, nx + 1))
    v = np.zeros((1, ny + 1, nx))
    g_prev = None
    s_prev = None
    prev_state = None
    nb = 1 + 2 * len(HA_CONS)
    ata = np.zeros((nb, nb))
    atb = np.zeros((nb, ny * nx))
    atb_U = np.zeros((nb, ny * (nx + 1)))
    atb_V = np.zeros((nb, (ny + 1) * nx))
    g_t, g_eta = [], []
    snap_t, snap = [], []
    snap_win = (np.datetime64(args.snapshots[0], "s"), np.datetime64(args.snapshots[1], "s")) if args.snapshots else None
    tic = time.time()
    for n in range(nsteps):
        t_mid = t0 + np.timedelta64(int((n + 0.5) * dt), "s")
        sec = (n + 0.5) * dt
        ramp = 1.0 - np.exp(-sec / tau_ramp)

        # body force (equilibrium tide − SAL) at mid-step
        eta_b = ramp * predict(np.array([t_mid]), body_flat, CONS)[0].reshape(ny, nx)
        gx, gy = body_force(hg, eta_b, g=G)

        # Coriolis on the C grid (4-point averages)
        v_at_u = np.zeros_like(u[0])
        v_at_u[:, 1:-1] = 0.25 * (v[0, :-1, :-1] + v[0, 1:, :-1] + v[0, :-1, 1:] + v[0, 1:, 1:])
        u_at_v = np.zeros_like(v[0])
        u_at_v[1:-1, :] = 0.25 * (u[0, :-1, :-1] + u[0, :-1, 1:] + u[0, 1:, :-1] + u[0, 1:, 1:])
        gu = np.where(wet_u, f_u * v_at_u + gx, 0.0)
        gv = np.where(wet_v, -f_v * u_at_v + gy, 0.0)
        if args.nonlinear:
            gu -= advection_u(u[0], v[0], hg, wet_u, scheme=args.adv)
            gv -= advection_v(u[0], v[0], hg, wet_v, scheme=args.adv)
        if args.ah > 0:
            gu += args.ah * laplacian_u(u[0], hg, wet_u)
            gv += args.ah * laplacian_v(v[0], hg, wet_v)
        if g_prev is None:
            gu_ab, gv_ab = gu, gv
        else:
            gu_ab, gv_ab = 1.5 * gu - 0.5 * g_prev[0], 1.5 * gv - 0.5 * g_prev[1]
        g_prev = (gu, gv)

        # bottom drag, semi-implicit: û = (u + Δt G) / (1 + Δt (C_d |u| / H + drgscl rh / H))
        if args.nonlinear:                                   # total depth H + η at the faces
            eta_u, eta_v = cell_to_faces(eta, wet_t)
            Du, Dv = np.maximum(hu + eta_u, 1.0), np.maximum(hv + eta_v, 1.0)
        else:
            Du, Dv = hu, hv
        r_u = np.where(wet_u, cd_u * np.hypot(u[0], v_at_u) / np.where(wet_u, Du, 1.0), 0.0) + r_jsl_u
        r_v = np.where(wet_v, cd_v * np.hypot(v[0], u_at_v) / np.where(wet_v, Dv, 1.0), 0.0) + r_jsl_v
        u_hat = ((u[0] + dt * gu_ab) / (1.0 + dt * r_u))[None]
        v_hat = ((v[0] + dt * gv_ab) / (1.0 + dt * r_v))[None]

        bt = boundary_tides_at(bcon, t_mid, ramp=ramp)
        s_ab = None
        if args.nonlinear:                                   # explicit −∇·(η u), AB2
            s_nl = nonlinear_continuity(eta, u[0], v[0], hg, wet_t)
            s_ab = s_nl if s_prev is None else 1.5 * s_nl - 0.5 * s_prev
            s_prev = s_nl
        eta, u, v = fs.step(eta, u, v, u_hat, v_hat, flather_eta=(bt["eta_u"], bt["eta_v"]),
                            flather_ubar=bt["ubar_u"], flather_vbar=bt["vbar_v"], eta_tendency=s_ab)

        if not np.isfinite(eta).all() or np.abs(eta).max() > 20:
            bad = np.where(np.isfinite(eta), np.abs(eta), np.inf)
            j, i = np.unravel_index(np.argmax(bad), bad.shape)
            np.savez_compressed(OUT / f"blowup_{args.tag}.npz", eta=eta, u=u[0], v=v[0], step=n,
                                **({} if prev_state is None else dict(eta_prev=prev_state[0], u_prev=prev_state[1],
                                                                       v_prev=prev_state[2])))
            raise RuntimeError(f"blow-up at step {n} (day {(n + 1) * dt / 86400:.2f}): max|η| = "
                               f"{np.nanmax(np.abs(eta)):.2f} m at lon {lon_t[j, i]:.2f} lat {lat_t[j, i]:.2f} "
                               f"depth {sd['depth'][j, i]:.0f} m; max|u| {np.nanmax(np.abs(u)):.2f} m/s")
        if (n + 1) % every_ha == 0:
            prev_state = (eta.copy(), u[0].copy(), v[0].copy())

        t_new = t0 + np.timedelta64(int((n + 1) * dt), "s")
        if (n + 1) % every_g == 0:
            g_t.append(t_new)
            g_eta.append([eta[j, i] for j, i in gauges.values()])
        if snap_win and (n + 1) % every_ha == 0 and snap_win[0] <= t_new <= snap_win[1]:
            snap_t.append(t_new)
            snap.append(np.where(wet_t, eta, np.nan).astype(np.float32))
        if n + 1 > n_spin and (n + 1) % every_ha == 0:
            a = ha_basis(t_new)
            ata += np.outer(a, a)
            atb += a[:, None] * eta.ravel()[None, :]
            if args.nonlinear:
                eta_u, eta_v = cell_to_faces(eta, wet_t)
                atb_U += a[:, None] * ((hu + eta_u) * u[0]).ravel()[None, :]
                atb_V += a[:, None] * ((hv + eta_v) * v[0]).ravel()[None, :]
            else:
                atb_U += a[:, None] * (hu * u[0]).ravel()[None, :]
                atb_V += a[:, None] * (hv * v[0]).ravel()[None, :]
        if args.benchmark and n == 4:
            tic5 = time.time()
        if (n + 1) % (every_ha * 24) == 0:
            el = time.time() - tic
            print(f"day {(n + 1) * dt / 86400:5.1f}  max|η| {np.abs(eta).max():.2f} m  "
                  f"max|u| {np.abs(u).max():.2f} m/s  {el / 60:.1f} min "
                  f"(ETA {el / (n + 1) * (nsteps - n - 1) / 60:.0f} min)", flush=True)

    if args.benchmark:
        per = (time.time() - tic5) / (nsteps - 5)
        print(f"benchmark: {per:.3f} s/step → {per * 33 * 86400 / dt / 3600:.2f} h for 33 days at Δt={dt:.0f} s")
        return

    x = np.linalg.solve(ata, atb)                                        # (nb, npts)
    hc_map = (x[1::2] + 1j * x[2::2]).T.reshape(ny, nx, len(HA_CONS))
    hc_map[~wet_t] = np.nan

    def ha_faces(b: np.ndarray, shape: tuple, wet: np.ndarray) -> np.ndarray:
        y = np.linalg.solve(ata, b)
        out = (y[1::2] + 1j * y[2::2]).T.reshape(*shape, len(HA_CONS))
        out[~wet] = np.nan
        return out.astype(np.complex64)
    hc_U = ha_faces(atb_U, (ny, nx + 1), wet_u)
    hc_V = ha_faces(atb_V, (ny + 1, nx), wet_v)
    OUT.mkdir(exist_ok=True)
    np.savez_compressed(
        OUT / f"run_{args.tag}.npz",
        gauge_keys=np.array(list(gauges)), gauge_ij=np.array(list(gauges.values())),
        gauge_t=np.array(g_t, "datetime64[s]"), gauge_eta=np.array(g_eta, np.float32),
        hc_map=hc_map.astype(np.complex64), ha_cons=np.array(HA_CONS),
        lon_t=lon_t, lat_t=lat_t, depth=sd["depth"], hc_U=hc_U, hc_V=hc_V,
        lon_1d=lon_t[0], lat_1d=lat_t[:, 0], lon_u_1d=hg["lon_u"].values[0], lat_v_1d=hg["lat_v"].values[:, 0],
        snap_t=np.array(snap_t, "datetime64[s]"), snap_eta=np.array(snap, np.float32),
        meta=json.dumps(vars(args) | dict(box=box, dll=sd["dll"], min_depth=MIN_DEPTH, i0=int(sd["i0"]),
                                          j0=int(sd["j0"]))))
    print("saved", OUT / f"run_{args.tag}.npz", f"total {(time.time() - t_setup) / 60:.1f} min")


if __name__ == "__main__":
    main()
