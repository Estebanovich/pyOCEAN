#!/usr/bin/env python
"""
Barotropic reproduction of the MITgcm BayIW_Rectan experiment with pyOCEAN.

Reference runs (MITgcm, Cruz Isidro M.Sc. thesis), on /Volumes/ESTEBAN_MIT:
    verification/BayIW_Rectan{,_2layer}/run_expand{,_nobay}
    560 × 352 telescopic Cartesian grid (5–13.3 km), flat ocean (1000 m in the
    bathymetry file but 599.9 m effective: MITgcm clips at Σ delR) with a
    rectangular bay (164 m, 119 km long), f-plane f0 = 6.97e-5 s⁻¹, implicit
    free surface, Δt = 30 s, meridional wind pulse (8 records of 1200 s,
    τy ≤ 0.099 N m⁻²; stage 1 = 0–60 h with the wind file, stage 2 = 60–120 h
    without forcing), open S/W/E boundaries (Orlanski + 30-cell sponge).

pyOCEAN configuration (barotropic, linear):
    single layer as deep as the clipped bathymetry, with hFac partial cells,
    ∂u/∂t = f v − g ∂η/∂x − u/τ_sp,   ∂v/∂t = −f u − g ∂η/∂y + τ_y/(ρ₀ H) − v/τ_sp
    implicit θ = 1 free surface (core.dynamics.free_surface), Δt = 30 s,
    closed walls + 30-cell velocity sponge on S/W/E (τ_sp from 60 s at the
    wall to 1200 s at the inner edge, as MITgcm's OBCS sponge).

Differences from MITgcm (by design, barotropic check of the seiche):
    no stratification, no advection, no viscosity (A_h = 100 m² s⁻¹ gives a
    damping time ≫ the seiche period on 5-km cells), no Orlanski radiation.

Output: η every 900 s at the analysis stations (coords_iwbay.py), saved as
    out/pyocean_{bay,nobay}.npz

Usage:  python run_pyocean.py bay|nobay [--hours 120] [--dt 30]
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dynamics.free_surface import ImplicitFreeSurface  # noqa: E402
from core.grid.horizontal import create_cartesian_grid_stretched  # noqa: E402
from core.grid.vertical import compute_hfac, create_z_grid  # noqa: E402

MIT = Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan/input")
FILES = {"bay": "bahia_01_expand", "nobay": "nobahia_01_expand"}
NX, NY = 560, 352
RHO0, G, F0 = 1025.0, 9.81, 6.97e-5          # MITgcm data: rhoNil≈999.8 for EOS; ρ₀ for τ/ρ₀H
WIND_PERIOD = 1200.0
STATIONS = {"Seiche": (280, 345), "Seiche2": (280, 325), "A": (230, 315), "B": (330, 315),
            "C": (280, 265), "Gamma": (280, 215), "Omega": (280, 315)}
SPONGE_N, TAU_BOUND, TAU_INNER = 30, 60.0, 1200.0


def read_mit(name: str, shape: tuple[int, ...]) -> np.ndarray:
    """MITgcm binary input: big-endian float64 (readBinaryPrec = 64)."""
    return np.fromfile(MIT / name, ">f8").reshape(shape)


def sponge_rate(n: int, m: int) -> np.ndarray:
    """
    1/τ on an (n, m) array: open S, W, E boundaries, closed N (MITgcm data.obcs).
    τ varies linearly from TAU_BOUND at the boundary row to TAU_INNER at row 30.
    """
    j, i = np.mgrid[0:n, 0:m]
    dist = np.minimum.reduce([j, i, m - 1 - i]).astype(float)        # cells from S, W, E
    tau = TAU_BOUND + (TAU_INNER - TAU_BOUND) * dist / (SPONGE_N - 1)
    return np.where(dist < SPONGE_N, 1.0 / tau, 0.0)


def wind_at(t: float, records: np.ndarray) -> np.ndarray:
    """
    Linear interpolation between records centred at (n + ½)·period, cyclic
    within the first cycle (MITgcm periodicExternalForcing, stage 1) and zero
    afterwards (stage 2 of the MITgcm runs has no wind file).
    """
    nrec = records.shape[0]
    if t >= nrec * WIND_PERIOD:
        return np.zeros_like(records[0])
    s = t / WIND_PERIOD - 0.5
    n0 = int(np.floor(s)) % nrec
    w = s - np.floor(s)
    return (1.0 - w) * records[n0] + w * records[(n0 + 1) % nrec]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=FILES)
    ap.add_argument("--hours", type=float, default=120.0)
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--theta", type=float, default=1.0)
    ap.add_argument("--max-depth", type=float, default=599.9,
                    help="clip bathymetry like MITgcm does at Σ delR (599.9 m in BayIW_Rectan)")
    args = ap.parse_args()

    stem = FILES[args.case]
    dx = np.fromfile(MIT / f"{stem}_dx.bin", ">f8")
    dy = np.fromfile(MIT / f"{stem}_dy.bin", ">f8")
    # MITgcm truncates the bathymetry at the bottom of its vertical grid (Σ delR):
    # the "1000 m" ocean of the bathy file is 599.9 m deep in the reference runs
    # (Depth in grid.t*.nc).
    depth = np.minimum(np.maximum(-read_mit(f"{stem}_bat.bin", (NY, NX)), 0.0), args.max_depth)
    wind = read_mit("merid_local_bay_560x352.bin", (-1, NY, NX))    # τ_y at V-points (j = south face)

    hg = create_cartesian_grid_stretched(dx, dy, x0=-1400e3, y0=-1621e3, f0=F0)
    vg = create_z_grid(1, [float(depth.max())])
    hf = compute_hfac(vg, depth)
    fs = ImplicitFreeSurface(hg, vg, hf, dt=args.dt, theta=args.theta, g=G)

    hu = (vg.dz.values[:, None, None] * hf.hfac_u.values)[0]          # (ny, nx+1)
    hv = (vg.dz.values[:, None, None] * hf.hfac_v.values)[0]          # (ny+1, nx)
    wet_u, wet_v = hu > 0, hv > 0
    r_u = sponge_rate(NY, NX + 1)
    r_v = sponge_rate(NY + 1, NX)

    eta = np.zeros((NY, NX))
    u = np.zeros((1, NY, NX + 1))
    v = np.zeros((1, NY + 1, NX))
    nsteps = int(round(args.hours * 3600 / args.dt))
    every = int(round(900.0 / args.dt))
    series = {k: [] for k in STATIONS}
    t_out = []
    g_prev = None
    t0 = time.time()

    for n in range(nsteps):
        t = n * args.dt
        # ---- tendencies except −g∇η (AB2-extrapolated) -------------------------
        v_at_u = np.zeros_like(u[0])
        v_at_u[:, 1:-1] = 0.25 * (v[0, :-1, :-1] + v[0, 1:, :-1] + v[0, :-1, 1:] + v[0, 1:, 1:])
        u_at_v = np.zeros_like(v[0])
        u_at_v[1:-1, :] = 0.25 * (u[0, :-1, :-1] + u[0, :-1, 1:] + u[0, 1:, :-1] + u[0, 1:, 1:])
        tau_v = np.zeros_like(v[0])
        tau_v[:-1, :] = wind_at(t, wind)                               # V-face j = south of cell j
        gu = np.where(wet_u, F0 * v_at_u - r_u * u[0], 0.0)
        gv = np.where(wet_v, -F0 * u_at_v + np.where(wet_v, tau_v / (RHO0 * np.where(wet_v, hv, 1.0)), 0.0)
                      - r_v * v[0], 0.0)
        if g_prev is None:
            gu_ab, gv_ab = gu, gv
        else:
            gu_ab, gv_ab = 1.5 * gu - 0.5 * g_prev[0], 1.5 * gv - 0.5 * g_prev[1]
        g_prev = (gu, gv)
        eta, u, v = fs.step(eta, u, v, u + args.dt * gu_ab[None], v + args.dt * gv_ab[None])

        if (n + 1) % every == 0:
            t_out.append((n + 1) * args.dt / 3600.0)
            for k, (i, j) in STATIONS.items():
                series[k].append(eta[j, i])
            if len(t_out) % 16 == 0:
                print(f"{args.case}: t = {t_out[-1]:6.1f} h  η_Seiche = {series['Seiche'][-1]:+.4f} m"
                      f"  ({(time.time() - t0) / 60:.1f} min)", flush=True)

    out = Path(__file__).parent / "out"
    out.mkdir(exist_ok=True)
    np.savez(out / f"pyocean_{args.case}.npz", t_horas=np.array(t_out),
             **{k: np.array(x) for k, x in series.items()},
             meta_json=json.dumps(vars(args) | {"stations": STATIONS}))
    print("saved", out / f"pyocean_{args.case}.npz")


if __name__ == "__main__":
    main()
