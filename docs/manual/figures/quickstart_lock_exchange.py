#!/usr/bin/env python
"""
Quick-start run of the pyOCEAN manual: a small lock exchange with SigmaBaroclinic.

A 64 km × 20 m channel holds cold (dense) water on the left and warm (light) water on the right,
separated by a vertical "lock" at x = 32 km.  At t = 0 the lock is removed: the dense water slides
under the light water as a gravity current whose front speed, after the initial acceleration, is
Benjamin's (1968) U = ½ √(g′ H).  The full benchmark (128 × 40 cells, 17 h, against CROCO) is
validation/lock_exchange/run_lock_exchange.py; this version is coarser (128 × 20 cells) and short (8 h)
so it runs in well under a minute on one thread.

Usage (from the repository root, base conda env):
    python docs/manual/figures/quickstart_lock_exchange.py              # σ-layers
    python docs/manual/figures/quickstart_lock_exchange.py --coord zstar
    python docs/manual/figures/quickstart_lock_exchange.py --coord hybrid
Output: docs/manual/img/quickstart/lock_<coord>_sections.png and lock_<coord>_fronts.png
"""

from __future__ import annotations

import os

# 1. One thread for everything: Numba kernels, BLAS and OpenMP (must be set before importing numpy/numba).
for _v in ("NUMBA_NUM_THREADS", "OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
           "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
import warnings  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

# 2. pyOCEAN is not an installed package: put the repository root on the import path.
ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402

# 3. Case parameters (SI units).
L, H = 64e3, 20.0                 # channel length and depth [m]
NX, NZ = 128, 20                  # cells along the channel, layers
DT, HOURS = 10.0, 8.0             # time step [s], run length [h]
RHO0, TCOEF = 1000.0, 0.28        # reference density [kg m-3], thermal coefficient α [kg m-3 °C-1]
T_COLD, T_WARM = 5.0 / TCOEF, 10.0 / TCOEF        # Δρ = α ΔT = 5 kg m-3
G_PRIME = 9.81 * 5.0 / RHO0                       # reduced gravity g′ = g Δρ/ρ0 [m s-2]
U_BENJAMIN = 0.5 * np.sqrt(G_PRIME * H)           # Benjamin front speed [m s-1] ≈ 0.495
OUT = ROOT / "docs" / "manual" / "img" / "quickstart"


def build(coord: str) -> tuple[SigmaBaroclinic, object]:
    """The model and its resting initial state for coord ∈ {'sigma', 'zstar', 'hybrid'}."""
    # 4. Horizontal grid: uniform Cartesian C-grid, one row (a 2-D x–z section), no rotation.
    hg = create_cartesian_grid(nx=NX, ny=1, dx=L / NX, dy=L / NX, f0=0.0)
    bathy = np.full((1, NX), H)                                    # rest depth H (j, i) [m], positive down
    common = dict(dt=DT, n_layers=NZ, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant")
    if coord == "sigma":
        # 5a. σ-Eulerian mode (the default vertical coordinate): fixed fractions of the column.
        m = SigmaBaroclinic(hg, bathy, nu_const=1e-6, **common)
    elif coord == "zstar":
        # 5b. ALE z*: nominal level thicknesses dz_top (surface first) that sum to the depth.
        m = SigmaBaroclinic(hg, bathy, nu_const=1e-6, vertical_coordinate="ale", regrid="zstar",
                            regrid_params=dict(dz_top=np.full(NZ, H / NZ)), ke_scheme="gudonov", **common)
    elif coord == "hybrid":
        # 5c. ALE hybrid (HYCOM1 rule): NZ + 1 interface target densities spanning both water masses,
        #     a 1 cm z* floor (so the interior is isopycnal), MOM6 lock-exchange viscosities.
        #     The model warns that this floor (Σ dz_top = 0.2 m) stops above the bed: intended here, the two
        #     homogeneous water masses should each sit in an isopycnal layer, so the warning is silenced.
        warnings.filterwarnings("ignore", message="hybrid z\\* floor")
        rho_l, rho_d = RHO0 - TCOEF * T_WARM, RHO0 - TCOEF * T_COLD
        params = dict(dz_top=np.full(NZ, 0.01), rho_target=np.linspace(rho_l - 1.0, rho_d + 1.0, NZ + 1))
        m = SigmaBaroclinic(hg, bathy, nu_const=1e-4, ah=0.01, vertical_coordinate="ale", regrid="hybrid",
                            regrid_params=params, ke_scheme="gudonov", **common)
    else:
        raise ValueError(coord)
    # 6. Initial temperature (k, j, i): cold left of the lock, warm right, in every layer; η = 0.
    T0 = np.where(np.arange(NX) < NX // 2, T_COLD, T_WARM)[None, None, :].repeat(NZ, axis=0)
    st = m.initial_state(np.zeros((1, NX)), T0)
    return m, st


def layer_thickness(m: SigmaBaroclinic, st) -> np.ndarray:
    """Layer thicknesses (k, j, i) [m] in either mode: volume / cell area."""
    return m.layer_volumes(st.eta, st.h) / m.area[None]


def fronts(m: SigmaBaroclinic, st, x: np.ndarray, thick: float = 1.0) -> tuple[float, float]:
    """Distance from the lock [m] of the dense (rightward) and light (leftward) fronts: last column where
    more than ``thick`` metres of that water mass remain."""
    half = 0.5 * (T_COLD + T_WARM)
    h = layer_thickness(m, st)[:, 0]
    dense = np.where(st.T[:, 0] < half, h, 0.0).sum(0)
    light = np.where(st.T[:, 0] > half, h, 0.0).sum(0)
    return x[dense > thick].max() - L / 2, L / 2 - x[light > thick].min()


def section(ax, m: SigmaBaroclinic, st, title: str):
    """α T on the actual layer geometry (cells drawn between their interfaces)."""
    h = layer_thickness(m, st)[:, 0]
    zw = -H + np.concatenate([np.zeros((1, NX)), np.cumsum(h, axis=0)])        # interfaces, bed first
    xe = np.linspace(0.0, L, NX + 1) / 1e3
    Xe = np.repeat(xe[None], NZ + 1, axis=0)
    Ze = np.concatenate([zw, zw[:, -1:]], axis=1)                                # pad the last column
    pc = ax.pcolormesh(Xe, Ze, TCOEF * st.T[:, 0], cmap="RdBu_r", vmin=TCOEF * T_COLD, vmax=TCOEF * T_WARM,
                       shading="flat")
    for k in range(NZ + 1):                                                      # layer interfaces
        ax.plot((np.arange(NX) + 0.5) * L / NX / 1e3, zw[k], color="0.2", lw=0.4)
    ax.set_title(title, fontsize=10)
    ax.set_ylabel("z [m]")
    return pc


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--coord", default="sigma", choices=("sigma", "zstar", "hybrid"))
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    set_threads(1)                                   # 7. threads of the compiled kernels (≤ NUMBA_NUM_THREADS)

    m, st = build(args.coord)
    x = (np.arange(NX) + 0.5) * L / NX
    st0 = st
    c0 = m.tracer_content(st)                        # Σ V T, conserved in a closed basin
    rpe0 = m.rpe(st)                                 # reference potential energy [J]

    # 8. Time loop: one call to step() per Δt; diagnostics every 30 steps (5 min).
    every = 30
    t, fr = [0.0], [(0.0, 0.0)]
    tic = time.time()
    for n in range(int(round(HOURS * 3600 / DT))):
        st = m.step(st)
        if (n + 1) % every == 0:
            t.append(st.time)
            fr.append(fronts(m, st, x))
    wall = time.time() - tic
    t, fr = np.array(t), np.array(fr)

    # 9. Report: conservation, mixing, front speed between 2 and 6 h.
    i2, i6 = np.argmin(abs(t - 2 * 3600)), np.argmin(abs(t - 6 * 3600))
    speed = (fr[i6] - fr[i2]) / (t[i6] - t[i2])
    print(f"coord = {args.coord}: {len(t) - 1} samples, {int(HOURS * 3600 / DT)} steps in {wall:.1f} s "
          f"({1e3 * wall / (HOURS * 3600 / DT):.1f} ms/step)")
    print(f"  heat content drift   : {m.tracer_content(st) / c0 - 1:+.2e}")
    print(f"  RPE gain             : {m.rpe(st) - rpe0:.3e} J  ({(m.rpe(st) - rpe0) / rpe0:.2e} of RPE0)")
    print(f"  T range              : {TCOEF * st.T.min():.4f} – {TCOEF * st.T.max():.4f} (initial 5 – 10)")
    print(f"  max |u|              : {np.abs(st.u).max():.3f} m/s;  max |eta| {np.abs(st.eta).max():.2e} m")
    print(f"  front speed 2–6 h    : dense {speed[0]:.3f}, light {speed[1]:.3f} m/s "
          f"= {speed[0] / U_BENJAMIN:.2f} / {speed[1] / U_BENJAMIN:.2f} U_Benjamin ({U_BENJAMIN:.3f} m/s)")
    if m.ale:
        print(f"  ALE: CFL truncations {st.n_trunc}, outflow-limited cell-layers {st.n_limited}")

    # 10. Figures.
    fig, ax = plt.subplots(2, 1, figsize=(8, 5.2), sharex=True)
    section(ax[0], m, st0, f"{args.coord}: initial α T [kg m⁻³] (t = 0)")
    pc = section(ax[1], m, st, f"{args.coord}: α T after {HOURS:.0f} h")
    ax[1].set_xlabel("x [km]")
    fig.colorbar(pc, ax=ax, label="α T [kg m⁻³]")
    fig.savefig(OUT / f"lock_{args.coord}_sections.png", dpi=150)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(t / 3600, fr[:, 0] / 1e3, color="#0072B2", lw=2, label="dense front (bottom, →)")
    ax.plot(t / 3600, fr[:, 1] / 1e3, color="#D55E00", lw=2, ls="--", label="light front (top, ←)")
    ax.plot(t / 3600, U_BENJAMIN * t / 1e3, "k:", label=f"Benjamin U t, U = {U_BENJAMIN:.3f} m/s")
    ax.set_xlabel("time [h]")
    ax.set_ylabel("distance from the lock [km]")
    ax.set_title(f"Lock exchange ({args.coord}, {NX} × {NZ}): front position", fontsize=10)
    ax.grid(alpha=0.3)
    ax.legend(fontsize=9)
    fig.tight_layout()
    fig.savefig(OUT / f"lock_{args.coord}_fronts.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
