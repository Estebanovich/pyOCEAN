#!/usr/bin/env python
"""
Lock exchange: front speed and spurious mixing — pyOCEAN vs CROCO GRAV_ADJ vs theory.

Setup (CROCO OCEAN/ana_grid.F, ana_initial.F, TEST_CASES/croco.in.Grav_adj; Ilicak et al. 2012):
    channel 64 km × 20 m, 128 × 40 (Δx = 500 m), f = 0, closed, no bottom drag
    T = 5/α left, 10/α right (Δρ = 5 kg m⁻³, linear EOS ρ = ρ₀ − α T, α = 0.28)
    Δt = 10 s, 17 h; vertical viscosity 1e-6 m² s⁻¹, no explicit tracer diffusion
Truth: Benjamin (1968) front speed U = ½ √(g′ H) = 0.495 m s⁻¹ (after the initial acceleration).
Spurious mixing: reference potential energy (Winters et al. 1995; Ilicak et al. 2012) and the online
discrete variance decay (Burchard & Rennau 2008); Ilicak-style sweep of the lateral viscosity
A_h → grid Reynolds number Re_Δ = U Δx / A_h.

Usage:  python validation/lock_exchange/run_lock_exchange.py [--croco PATH/gravadj_his.nc]
            [--ah 0 1 10 100] [--adv flux] [--tracer weno5] [--hours 17]
Output: validation/lock_exchange/out/lock_<tag>.npz, lock_exchange.png, summary.txt
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import argparse  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from concurrent.futures import ProcessPoolExecutor  # noqa: E402
from pathlib import Path  # noqa: E402

import matplotlib  # noqa: E402

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from diagnostics.mixing import reference_potential_energy  # noqa: E402

L, H, NX, NZ, DT = 64e3, 20.0, 128, 40, 10.0
TCOEF, RHO0 = 0.28, 1000.0
T_COLD, T_WARM = 5.0 / TCOEF, 10.0 / TCOEF
U_BENJAMIN = 0.5 * np.sqrt(9.81 * 5.0 / RHO0 * H)
OUT = Path(__file__).parent / "out"


def fronts(x: np.ndarray, T: np.ndarray) -> tuple[float, float]:
    """Dense (bottom, rightward) and light (top, leftward) front distances from the lock [m]."""
    half = 0.5 * (T_COLD + T_WARM)
    return x[T[0] < half].max() - L / 2, L / 2 - x[T[-1] > half].min()


def run_pyocean(ah: float, adv: str, tracer: str, hours: float, every: int = 90) -> dict:
    dx = L / NX
    hg = create_cartesian_grid(nx=NX, ny=1, dx=dx, dy=dx, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, NX), H), dt=DT, n_layers=NZ, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-6, ah=ah, adv=adv, tracer_scheme=tracer)
    T = np.where(np.arange(NX) < NX // 2, T_COLD, T_WARM)[None, None, :].repeat(NZ, 0)
    st = m.initial_state(np.zeros((1, NX)), T)
    x = (np.arange(NX) + 0.5) * dx
    ts, fr, rpe, mix, trange = [0.0], [(0.0, 0.0)], [m.rpe(st)], [0.0], [(T.min(), T.max())]
    tic = time.time()
    for n in range(int(round(hours * 3600 / DT))):
        st = m.step(st)
        if (n + 1) % every == 0:
            ts.append(st.time)
            fr.append(fronts(x, st.T[:, 0]))
            rpe.append(m.rpe(st))
            mix.append(float(st.mix_num.sum()))
            trange.append((st.T.min(), st.T.max()))
    return dict(t=np.array(ts), fronts=np.array(fr), rpe=np.array(rpe), mix=np.array(mix), T=st.T[:, 0],
                u=st.u[:, 0], trange=np.array(trange), x=x, ah=ah, adv=adv, tracer=tracer,
                seconds=time.time() - tic)


def read_croco(path: Path) -> dict:
    import xarray as xr
    ds = xr.open_dataset(path, decode_times=False)
    j = 1
    t = ds["scrum_time"].values
    T = ds["temp"].values[:, :, j, 1:-1]
    x = ds["x_rho"].values[j, 1:-1]
    h = ds["h"].values[j, 1:-1]
    zeta = ds["zeta"].values[:, j, 1:-1]
    hc, cs_w, s_w = float(ds["hc"]), ds["Cs_w"].values, ds["s_w"].values
    # Vtransform 2: z_w = ζ + (ζ + h) (hc s + h C(s)) / (hc + h)
    z0 = (hc * s_w[:, None] + h[None] * cs_w[:, None]) / (hc + h[None])
    rpe = []
    dx = x[1] - x[0]
    for k in range(len(t)):
        zw = zeta[k][None] + (zeta[k][None] + h[None]) * z0
        vol = np.diff(zw, axis=0) * dx
        rpe.append(reference_potential_energy(RHO0 - TCOEF * T[k], vol, area=dx * len(x)))
    return dict(t=t, x=x, T=T, fronts=np.array([fronts(x, T[k]) for k in range(len(t))]), rpe=np.array(rpe),
                trange=np.array([(T[k].min(), T[k].max()) for k in range(len(t))]))


def speed(t: np.ndarray, f: np.ndarray, t0: float = 2 * 3600, t1: float = 8 * 3600) -> float:
    i0, i1 = int(np.argmin(np.abs(t - t0))), int(np.argmin(np.abs(t - t1)))
    return (f[i1] - f[i0]) / (t[i1] - t[i0])


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--croco", type=Path)
    ap.add_argument("--ah", type=float, nargs="+", default=[0.0, 1.0, 10.0, 100.0])
    ap.add_argument("--adv", default="flux")
    ap.add_argument("--tracer", default="weno5")
    ap.add_argument("--hours", type=float, default=17.0)
    ap.add_argument("--workers", type=int, default=4, help="concurrent runs (≤ 5: P-cores only)")
    args = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)

    with ProcessPoolExecutor(max_workers=min(args.workers, 5)) as ex:
        runs = list(ex.map(run_pyocean, args.ah, [args.adv] * len(args.ah), [args.tracer] * len(args.ah),
                           [args.hours] * len(args.ah)))
    for r in runs:
        np.savez_compressed(OUT / f"lock_{r['adv']}_{r['tracer']}_ah{r['ah']:g}.npz", **{k: v for k, v in r.items()})

    lines = [f"Lock exchange 128×40, Δt = {DT:.0f} s, {args.hours:.0f} h; Benjamin U = {U_BENJAMIN:.3f} m/s",
             f"{'run':34s} {'dense / light front speed 2–8 h [m/s]':>40s} {'ΔRPE/RPE₀ (end)':>17s} {'T range (end)':>18s}"]
    croco = None
    if args.croco and args.croco.exists():
        croco = read_croco(args.croco)
        f = croco["fronts"]
        lines.append(f"{'CROCO GRAV_ADJ (WENO5, no A_h)':34s} {speed(croco['t'], f[:, 0]):19.3f} / {speed(croco['t'], f[:, 1]):.3f}"
                     f"{(croco['rpe'][-1] - croco['rpe'][0]) / croco['rpe'][0]:22.2e}"
                     f"   {croco['trange'][-1, 0] * TCOEF:5.2f}–{croco['trange'][-1, 1] * TCOEF:5.2f}")
    for r in runs:
        f = r["fronts"]
        re_d = U_BENJAMIN * (L / NX) / r["ah"] if r["ah"] > 0 else np.inf
        lines.append(f"{'pyOCEAN ' + r['adv'] + '/' + r['tracer'] + f' A_h={r["ah"]:g} (Re_Δ={re_d:.3g})':34s}"
                     f" {speed(r['t'], f[:, 0]):19.3f} / {speed(r['t'], f[:, 1]):.3f}"
                     f"{(r['rpe'][-1] - r['rpe'][0]) / r['rpe'][0]:22.2e}"
                     f"   {r['trange'][-1, 0] * TCOEF:5.2f}–{r['trange'][-1, 1] * TCOEF:5.2f}"
                     f"   ({r['seconds'] / 60:.1f} min)")

    fig, ax = plt.subplots(3, 1, figsize=(10, 11))
    if croco is not None:
        ax[0].plot(croco["t"] / 3600, croco["fronts"][:, 0] / 1e3, "k", lw=2, label="CROCO dense")
        ax[1].plot(croco["t"] / 3600, (croco["rpe"] - croco["rpe"][0]) / croco["rpe"][0], "k", lw=2, label="CROCO")
    for i, r in enumerate(runs):
        ax[0].plot(r["t"] / 3600, r["fronts"][:, 0] / 1e3, f"C{i}", label=f"pyOCEAN A_h={r['ah']:g}")
        ax[1].plot(r["t"] / 3600, (r["rpe"] - r["rpe"][0]) / r["rpe"][0], f"C{i}", label=f"A_h={r['ah']:g}")
    tt = np.linspace(0, args.hours, 50)
    ax[0].plot(tt, U_BENJAMIN * tt * 3600 / 1e3, "k:", label="Benjamin ½√(g′H) t")
    ax[0].set_ylabel("dense front [km]")
    ax[0].legend(fontsize=8)
    ax[1].set_ylabel("(RPE − RPE₀)/RPE₀")
    ax[1].set_yscale("symlog", linthresh=1e-7)
    ax[1].legend(fontsize=8)
    r = runs[0]
    zc = -H + (np.arange(NZ) + 0.5) / NZ * H
    c = ax[2].pcolormesh(r["x"] / 1e3, zc, r["T"] * TCOEF, cmap="RdBu_r", shading="auto")
    ax[2].set_title(f"pyOCEAN T at {args.hours:.0f} h (A_h = {r['ah']:g})", fontsize=9)
    ax[2].set_xlabel("x [km]")
    fig.colorbar(c, ax=ax[2], label="α T [kg m⁻³]")
    fig.tight_layout()
    fig.savefig(OUT / "lock_exchange.png", dpi=100)
    (OUT / "summary.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
