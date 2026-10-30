#!/usr/bin/env python
"""
pyOCEAN runs of the four-model benchmark (spec §A2): ALE z*, σ or hybrid with the model's defaults (vector-invariant
momentum, KE Arakawa, WENO5-Z tracers, PPM_H4 remap, PPM continuity, finite-volume PGF), closed free-slip walls, no
bottom drag, ν_v = 1e-4, no tracer diffusion, linear EOS of cases.py.

- z*: dz_top = the case's z-levels; σ: uniform σ-layers; hybrid: dz_top = 0.01 m floors (isopycnal interior, as the
  hybrid lock exchange of tests/test_sigma_ale.py) and rho_target = cases.targets (the same as MOM6 'RHO').
- initial state: T of cases.T_fn at the layer centres (z*, σ); for the hybrid, T on the case z-levels as h_src
  (MOM6 Z_INIT_ALE_REMAPPING; validation/internal_wave/case.py:189-193).
- seamount: the reference profile rho_ref = 'auto' (validation/seamount: best pyOCEAN setting, recorded in README).
- wall_s: the time of the steps after the first 5 (Numba compilation), scaled to all steps.

Usage: NUMBA_NUM_THREADS=5 python validation/ilicak_benchmark/run_pyocean.py lock zstar 10 [--dt 10] [--hours 17]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from validation.ilicak_benchmark import cases, series  # noqa: E402

N_WARM = 5


def build(case: cases.Case, coord: str, nu: float, dt: float | None = None, smag: float | None = None):
    """The ALE model of ``case`` in ``coord`` ∈ {'zstar', 'sigma', 'hybrid'} with ν_h = nu, and its initial state.
    ``smag`` is the Smagorinsky coefficient C (None: the case's own, ``case.smag``; 0 = off)."""
    smag = case.smag if smag is None else float(smag)
    hg = create_cartesian_grid(nx=case.nx, ny=case.ny, dx=float(case.dx[0]), dy=float(case.dy[0]), f0=case.f0)
    nz = case.nz
    if coord == "zstar":
        params = dict(dz_top=case.dz.copy())
    elif coord == "sigma":
        params = {}
    elif coord == "hybrid":
        params = dict(dz_top=np.full(nz, 0.01), rho_target=cases.targets(case, nz))
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, case.H.copy(), dt=case.dt if dt is None else dt, n_layers=nz, rho0=cases.RHO0,
                        t_coef=cases.ALPHA, t_ref=cases.T_REF, z0=None, closure="constant", nu_const=cases.NU_V,
                        ah=nu, g=cases.G, pgf="fv", rho_ref="auto" if case.name == "seamount" else None,
                        vertical_coordinate="ale", regrid=coord, regrid_params=params, smagorinsky=smag)
    eta = np.zeros((case.ny, case.nx))
    X, Y = case.x[None, None, :], case.y[None, :, None]
    if coord == "hybrid":
        zf = np.concatenate([[0.0], np.cumsum(case.dz)])[:, None, None]
        h_src = (np.minimum(zf[1:], case.H[None]) - np.minimum(zf[:-1], case.H[None]))[::-1]     # bed first
        T_src = case.T_fn(X, Y, case.zc[:, None, None])[::-1]
        return m, m.initial_state(eta, T_src, h_src=h_src)
    zc = m.layer_heights(eta)                                                   # (N, ny, nx) bed first, ≤ 0
    return m, m.initial_state(eta, case.T_fn(X, Y, zc))


def record(m, st) -> dict:
    """The series fields of state ``st``, layers flipped to surface first."""
    return dict(h=st.h[::-1].copy(), T=np.where(m.fs.active[None], st.T[::-1], np.nan),     # land: NaN
                u=st.u[::-1].copy(), v=st.v[::-1].copy(), eta=st.eta.copy())


def run(case_name: str, coord: str, nu: float, dt: float | None = None, hours: float | None = None, tag: str = "",
        threads: int = 5, smag: float | None = None) -> Path:
    """Run and write cases.run_dir(case, 'pyocean', coord, nu, tag)/series.npz; returns its path.  ``smag``
    (Smagorinsky C, None: the case's own) > 0 given explicitly adds the tag 'smag<C>' unless ``tag`` is set; the
    meta then also hold ``smag``, ``n_visc_limited`` and ``nu_t_max`` (max ν at the output times)."""
    set_threads(threads)
    c = cases.get(case_name)
    if smag is not None and smag > 0.0 and not tag and smag != c.smag:
        tag = f"smag{smag:g}"
    smag = c.smag if smag is None else float(smag)
    out = cases.run_dir(case_name, "pyocean", coord, nu, tag) / "series.npz"
    out.unlink(missing_ok=True)                    # a failed re-run must not leave the old series behind
    dt = c.dt if dt is None else dt
    hours = c.hours if hours is None else hours
    m, st = build(c, coord, nu, dt, smag)
    n_steps = int(round(hours * 3600.0 / dt))
    every = max(1, int(round(min(c.out_s, hours * 3600.0) / dt)))
    recs, times = [record(m, st)], [0.0]
    tic, nu_max = None, float("nan")
    for n in range(1, n_steps + 1):
        if n == N_WARM + 1:
            tic = time.perf_counter()
        st = m.step(st)
        if n % every == 0 or n == n_steps:
            recs.append(record(m, st))
            times.append(n * dt)
            if smag > 0.0:
                nu_max = np.fmax(nu_max, np.nanmax(m._momentum.last_nu_t))
    wall = (time.perf_counter() - tic) * n_steps / max(1, n_steps - N_WARM) if tic else float("nan")
    s = {k: np.array([r[k] for r in recs]) for k in ("h", "T", "u", "v", "eta")}
    s.update(t=np.array(times), x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H)
    meta = dict(model="pyOCEAN", coord=coord, case=case_name, nu=nu, dt=dt, n_steps=n_steps, wall_s=wall,
                threads=threads, n_trunc=int(st.n_trunc))
    if smag > 0.0:
        meta.update(smag=smag, n_visc_limited=int(st.n_visc_limited or 0), nu_t_max=float(nu_max))
    return series.save(out, s, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("coord", choices=["zstar", "sigma", "hybrid"])
    ap.add_argument("nu", type=float)
    ap.add_argument("--dt", type=float)
    ap.add_argument("--hours", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--smag", type=float, default=None, help="Smagorinsky C (default: the case's, 0 = off)")
    a = ap.parse_args()
    print(run(a.case, a.coord, a.nu, a.dt, a.hours, a.tag, a.threads, a.smag))


if __name__ == "__main__":
    main()
