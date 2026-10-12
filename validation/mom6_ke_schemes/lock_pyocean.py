"""
Hybrid and ALE-σ lock exchange in pyOCEAN with the two kinetic-energy schemes and 10 or 21 layers
(MOM6 lock_exchange uses 21), MOM6 viscosity for the hybrid runs (KV 1e-4, KH 0.01), and the two
ALE layer thickness fluxes ('upwind' donor cell, 'ppm' MOM6 continuity_PPM).

Usage: NUMBA_NUM_THREADS=5 python validation/mom6_ke_schemes/lock_pyocean.py [filter, e.g. hybrid]
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import tests.test_sigma_ale as t  # noqa: E402
from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from tests.test_sigma_ale import RHO0, TCOEF  # noqa: E402


def run(rule: str, nz: int, ke: str, flux: str = "upwind", nx: int = 128, depth: float = 20.0,
        dt: float = 20.0) -> dict:
    if rule == "hybrid":
        m, st = t._hybrid_lock(nx, nz, depth, dt, thickness_flux=flux)
    else:
        m = t._ale_box("sigma", nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params={})
        st = m.initial_state(np.zeros((1, nx)), np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :]
                             .repeat(nz, 0) / TCOEF)
    m.ke_scheme = ke
    r0 = m.rpe(st)
    x = (np.arange(nx) + 0.5) * 500.0
    half = 7.5 / TCOEF
    fronts, umax = {}, 0.0
    for n in range(int(6 * 3600 / dt)):
        st = m.step(st)
        umax = max(umax, float(np.abs(st.u).max()))
        if (n + 1) * dt in (7200.0, 21600.0):
            dense = np.where(st.T < half, st.h, 0.0).sum(0)[0]
            light = np.where(st.T > half, st.h, 0.0).sum(0)[0]
            fronts[(n + 1) * dt] = {d: (x[dense > d].max() - 32e3, 32e3 - x[light > d].min()) for d in (1.0, 5.0)}
    U = 0.5 * np.sqrt(9.81 * 5.0 / RHO0 * depth)
    sp = {d: tuple((fronts[21600.0][d][k] - fronts[7200.0][d][k]) / 14400 / U for k in range(2)) for d in (1.0, 5.0)}
    return dict(speed=sp, rpe=m.rpe(st) - r0, umax=umax, trunc=st.n_trunc)


def main(only: str | None = None) -> None:
    """All cases, or only those whose label contains ``only`` (e.g. 'hybrid', 'ppm')."""
    set_threads(5)
    cases = [(r, nz, ke, fl) for r, nz in (("hybrid", 10), ("hybrid", 21)) for fl in ("upwind", "ppm")
             for ke in ("arakawa", "gudonov")]
    cases += [("sigma", nz, ke, "upwind") for nz in (10, 20) for ke in ("arakawa", "gudonov")]
    print(f"{'caso':28s} | frente 1 m (d, l) | cuerpo 5 m (d, l) | RPE 6 h [J] | max|u| | trunc")
    for rule, nz, ke, fl in cases:
        label = f"{rule} {nz} {ke} {fl}"
        if only and only not in label:
            continue
        r = run(rule, nz, ke, fl)
        s1, s5 = r["speed"][1.0], r["speed"][5.0]
        print(f"{label:28s} | {s1[0]:5.2f} {s1[1]:5.2f} U     | {s5[0]:5.2f} {s5[1]:5.2f} U     "
              f"| {r['rpe']:9.2e}   | {r['umax']:.2f}   | {r['trunc']}", flush=True)


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)
