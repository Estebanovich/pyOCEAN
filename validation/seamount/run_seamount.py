#!/usr/bin/env python
"""
Seamount test: pressure-gradient error of the σ-layer model over steep topography.

Setup (Beckmann & Haidvogel 1993, §2c; the standard σ-coordinate PGF test):
    Gaussian seamount h = H₀ − H exp(−(x² + y²)/L²), H₀ = 4500 m, H = 4050 m (δ = 0.9), L = 25 km
    (max slope 13.7 %), f-plane f₀ = 1e-4 s⁻¹, closed 384 × 288 km box
    resting ocean with level isopycnals ρ(z) = ρ₀ + Δρ (1 − exp(z/d)), Δρ = 3 kg m⁻³, d = 1000 m
The exact pressure gradient is zero: every velocity is numerical error.
Metrics: instantaneous PGF error max|PGF| at t = 0 and the spurious current max|u| after ``days``.

Usage:  python validation/seamount/run_seamount.py [--dx 6000] [--layers 20] [--days 10]
            [--pgf twoterm|jacobian|fv] [--ref] [--coord sigma|zstar]
        e.g. --pgf fv --coord zstar: ALE z* layers (vertical_coordinate="ale") with the finite-volume PGF
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, reference_profile, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from diagnostics.domain_check import rx0, rx1  # noqa: E402

H0, HS, LS, F0 = 4500.0, 4050.0, 25e3, 1e-4
RHO0, DRHO, DECAY, TCOEF = 1000.0, 3.0, 1000.0, 0.2
OUT = Path(__file__).parent / "out"


def setup(dx: float, n_layers: int, dt: float, pgf: str, coord: str = "sigma"):
    nx, ny = int(round(384e3 / dx)), int(round(288e3 / dx))
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=F0)
    x = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    y = (np.arange(ny) + 0.5) * dx - ny * dx / 2
    h = H0 - HS * np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / LS**2)
    kw = {}
    if coord == "zstar":
        kw = dict(vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(n_layers, H0 / n_layers)))
    m = SigmaBaroclinic(hg, h, dt=dt, n_layers=n_layers, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, ah=0.0, pgf=pgf, **kw)
    z = m.layer_heights(np.zeros((ny, nx)))                              # cell-centre heights, η = 0
    rho = RHO0 + DRHO * (1 - np.exp(z / DECAY))
    T = (RHO0 - rho) / TCOEF                                              # linear EOS ρ = ρ₀ − α T
    return m, m.initial_state(np.zeros((ny, nx)), T), h


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dx", type=float, default=6000.0)
    ap.add_argument("--layers", type=int, default=20)
    ap.add_argument("--dt", type=float, default=600.0)
    ap.add_argument("--days", type=float, default=10.0)
    ap.add_argument("--pgf", default="twoterm")
    ap.add_argument("--ref", action="store_true", help="subtract the horizontally averaged ρ(z) of the initial state")
    ap.add_argument("--coord", default="sigma", choices=["sigma", "zstar"])
    args = ap.parse_args()
    set_threads(5)                                                        # the 5 performance cores (run with NUMBA_NUM_THREADS=5)
    m, st, h = setup(args.dx, args.layers, args.dt, args.pgf, args.coord)
    if args.ref:
        z = m.layer_heights(st.eta, st.h)
        m.rho_ref = reference_profile(z, m.density(st.T))
    tag = args.pgf + ("_ref" if args.ref else "") + ("_zstar" if args.coord == "zstar" else "")
    gx, gy = m._baroclinic_pgf(st.eta, st.T, st.h)
    r1 = rx1(h, np.linspace(-1, 0, args.layers + 1))[0].max()
    lines = [f"seamount {h.shape[1]}×{h.shape[0]}×{args.layers}, dx {args.dx / 1e3:.0f} km, pgf {tag}: "
             f"rx0 {rx0(h)[0].max():.2f}, rx1 {r1:.1f}",
             f"  t=0 max|PGF| {max(np.abs(gx).max(), np.abs(gy).max()):.2e} m s⁻² "
             f"(≈ {max(np.abs(gx).max(), np.abs(gy).max()) / F0 * 100:.1f} cm/s geostrophic)"]
    tic = time.time()
    umax = []
    for n in range(int(round(args.days * 86400 / args.dt))):
        st = m.step(st)
        if (n + 1) % int(round(86400 / args.dt)) == 0:
            umax.append(max(np.abs(st.u).max(), np.abs(st.v).max()))
    lines.append("  max|u| per day [cm/s]: " + " ".join(f"{100 * u:.2f}" for u in umax)
                 + f"   ({(time.time() - tic) / 60:.1f} min)")
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / f"summary_{tag}.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
