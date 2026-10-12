"""
Internal seiche in pyOCEAN with the two kinetic-energy schemes of the vector-invariant form.

Closed channel L = 60 km, H = 200 m, linear stratification N = 0.02 s⁻¹ (c₁ = NH/π = 1.27 m/s),
mode-1 standing seiche ζ = 5 m sin(−πz/H) cos(πx/L); period 2L/c₁ = 26.2 h.  The kinetic energy
oscillates at twice the seiche frequency: the spacing of its maxima gives the period and their
ratio the damping per period (energy-conserving numerics: ratio → 1).

Usage: NUMBA_NUM_THREADS=5 python validation/mom6_ke_schemes/seiche_pyocean.py
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from core.vertical.regrid import target_densities  # noqa: E402

RHO0, TCOEF, H, N_BV, L, NX, NZ, DT = 1000.0, 0.2, 200.0, 0.02, 60e3, 60, 20, 120.0


def build(label: str):
    hg = create_cartesian_grid(nx=NX, ny=1, dx=L / NX, dy=L / NX, f0=0.0)
    kw = dict(dt=DT, n_layers=NZ, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant", nu_const=1e-5)
    if label != "sigma_eulerian":
        coord, ke = label.split("/")
        kw.update(vertical_coordinate="ale", momentum="vector_invariant", ke_scheme=ke, pgf="fv", regrid=coord)
        if coord == "zstar":
            kw["regrid_params"] = dict(dz_top=np.full(NZ, H / NZ))
        elif coord == "sigma":
            kw["regrid_params"] = {}
    m0 = SigmaBaroclinic(hg, np.full((1, NX), H), **{k: v for k, v in kw.items() if k not in ("regrid", "regrid_params")}
                         if label == "sigma_eulerian" else {**kw, "regrid": "sigma", "regrid_params": {}})
    x = (np.arange(NX) + 0.5) * L / NX
    z = m0.layer_heights(np.zeros((1, NX)))
    dTdz = N_BV**2 * RHO0 / (9.81 * TCOEF)

    def temp(zz):
        zeta = 5.0 * np.sin(-np.pi * zz / H) * np.cos(np.pi * x / L)[None, None, :]
        return 10.0 + dTdz * (zz + zeta)

    if label.startswith("hybrid"):
        rho = m0.density(temp(z))
        kw["regrid_params"] = dict(dz_top=np.full(NZ, 0.5), rho_target=target_densities(rho.ravel(), NZ))
        m = SigmaBaroclinic(hg, np.full((1, NX), H), **kw)
        # the hybrid initial grid is σ (uniform) layers; init_iterations regrid–remap passes build the isopycnals
        return m, m.initial_state(np.zeros((1, NX)), temp(m.layer_heights(np.zeros((1, NX)))))
    m = m0 if label == "sigma_eulerian" else SigmaBaroclinic(hg, np.full((1, NX), H), **kw)
    zz = m.layer_heights(np.zeros((1, NX)))
    return m, m.initial_state(np.zeros((1, NX)), temp(zz))


def kinetic_energy(m, st) -> float:
    V = m.layer_volumes(st.eta, st.h)[:, 0, :]
    Vf = np.zeros_like(st.u[:, 0, :])
    Vf[:, 1:-1] = 0.5 * (V[:, :-1] + V[:, 1:])
    return float(0.5 * (Vf * st.u[:, 0, :] ** 2).sum())


def run(label: str, hours: float = 110.0) -> dict:
    m, st = build(label)
    t, ke = [], []
    tic = time.perf_counter()
    for n in range(int(hours * 3600 / DT)):
        st = m.step(st)
        t.append((n + 1) * DT / 3600)
        ke.append(kinetic_energy(m, st))
    t, ke = np.array(t), np.array(ke)
    i = [k for k in range(1, len(ke) - 1) if ke[k] > ke[k - 1] and ke[k] >= ke[k + 1] and ke[k] > 0.2 * ke.max()]
    per = 2 * np.mean(np.diff(t[i])) if len(i) > 2 else float("nan")
    dec = (ke[i[-1]] / ke[i[0]]) ** (1.0 / (len(i) - 1)) if len(i) > 2 else float("nan")
    return dict(period=per, decay_per_half_period=dec, n_peaks=len(i), wall=time.perf_counter() - tic,
                trunc=getattr(st, "n_trunc", 0))


def main() -> None:
    set_threads(5)
    print(f"seiche: teoría 2L/c1 = {2 * L / (N_BV * H / np.pi) / 3600:.2f} h")
    for label in ("sigma_eulerian", "sigma/arakawa", "sigma/gudonov", "zstar/arakawa", "zstar/gudonov",
                  "hybrid/arakawa", "hybrid/gudonov"):
        r = run(label)
        print(f"{label:16s} período {r['period']:6.2f} h  KE pico/pico (½ período) {r['decay_per_half_period']:.4f}  "
              f"picos {r['n_peaks']}  trunc {r['trunc']}  ({r['wall'] / 60:.1f} min)")


if __name__ == "__main__":
    main()
