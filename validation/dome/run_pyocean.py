#!/usr/bin/env python
"""
pyOCEAN runs of the DOME benchmark (spec 2026-10-14-dome-overflow §5): ALE z*, σ or hybrid with the model defaults
(vector-invariant momentum, KE Arakawa, WENO5-Z + FCT, PPM_H4, PPM continuity, finite-volume PGF), the common physics
of the case (Laplacian ν_h, ν_v, no tracer diffusion, quadratic drag C_d over hbbl), one passive tracer s (the inflow
dye), the interior sponges of MOM6 DOME and the north inflow as a 'clamped' edge with its exact transport.

Usage: NUMBA_NUM_THREADS=5 python validation/dome/run_pyocean.py 10 zstar [--dt 600] [--days 50] [--tag x]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from boundaries.obcs import OBData  # noqa: E402
from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from validation.dome import cases, series  # noqa: E402

N_WARM = 5


def build(case: cases.Case, coord: str, dt: float | None = None, nu_v: float | None = None):
    """The ALE model of ``case`` in ``coord`` ∈ {'zstar', 'sigma', 'hybrid'} and its resting initial state."""
    hg = create_cartesian_grid(nx=case.nx, ny=case.ny, dx=case.dx, dy=case.dx, f0=case.f0)
    nz = case.nz
    dz = np.full(nz, cases.HMAX / nz)                                  # 25 × 144 m (Legg's MITgcm levels)
    if coord == "zstar":
        params = dict(dz_top=dz)
    elif coord == "sigma":
        params = {}
    elif coord == "hybrid":
        params = dict(dz_top=np.full(nz, 0.01), rho_target=cases.targets(nz))   # isopycnal interior (Ilicak)
    else:
        raise ValueError(f"unknown coord {coord!r}")
    prof = cases.inflow_profile(case)

    def state(t):
        return {"north": OBData(side="north", T=prof["T"], u_normal=prof["v"], zw=prof["zw"],
                                tracers=prof["s"][None], transport=prof["transport"])}
    zs = -np.linspace(0.0, cases.HMAX, 361)
    m = SigmaBaroclinic(hg, case.H.copy(), dt=case.dt if dt is None else dt, n_layers=nz, rho0=cases.RHO0,
                        t_coef=cases.ALPHA, t_ref=cases.T_REF, z0=None, closure="constant",
                        nu_const=case.nu_v if nu_v is None else nu_v, ah=case.nu_h, g=cases.G, pgf="fv",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params, n_tracers=1,
                        open_boundaries=("north",), open_boundary_mode={"north": "clamped"},
                        open_boundary_state=state, sponge_rate=cases.sponge_rate(case),
                        sponge_T=cases.T_ambient(0.5 * (zs[1:] + zs[:-1])), sponge_zw=zs,
                        bottom_drag="quadratic", cd=case.cd, hbbl=case.hbbl)
    eta = np.zeros((case.ny, case.nx))
    if coord == "hybrid":                                             # T on 10-m levels as the source (MOM6 Z_INIT)
        zf = np.arange(0.0, cases.HMAX + 10.0, 10.0)[:, None, None]
        Hc = np.nan_to_num(case.H)[None]
        h_src = (np.minimum(zf[1:], Hc) - np.minimum(zf[:-1], Hc))[::-1]
        T_src = cases.T_ambient(-0.5 * (zf[1:] + zf[:-1]))[::-1] * np.ones_like(h_src)
        return m, m.initial_state(eta, T_src, h_src=h_src, tr=np.zeros((1,) + h_src.shape))
    zc = m.layer_heights(eta)
    return m, m.initial_state(eta, cases.T_ambient(zc))


def record(m, st) -> dict:
    """Series fields of ``st``, layers flipped to surface first, land NaN."""
    act = m.fs.active[None]
    return dict(h=st.h[::-1].copy(), T=np.where(act, st.T[::-1], np.nan), s=np.where(act, st.tr[0][::-1], np.nan),
                u=st.u[::-1].copy(), v=st.v[::-1].copy(), eta=st.eta.copy())


def run(res_km: float, coord: str, dt: float | None = None, days: float | None = None, tag: str = "",
        threads: int = 5, nu_v: float | None = None) -> Path:
    """Run and write cases.run_dir(res_km, 'pyocean', coord, tag)/series.npz; returns its path."""
    set_threads(threads)
    c = cases.get(res_km)
    out = cases.run_dir(res_km, "pyocean", coord, tag) / "series.npz"
    out.unlink(missing_ok=True)
    dt = c.dt if dt is None else dt
    days = c.days if days is None else days
    m, st = build(c, coord, dt, nu_v)
    n_steps = int(round(days * 86400.0 / dt))
    every = max(1, int(round(c.out_s / dt)))
    recs, times, tic = [record(m, st)], [0.0], None
    for n in range(1, n_steps + 1):
        if n == N_WARM + 1:
            tic = time.perf_counter()
        st = m.step(st)
        if n % every == 0 or n == n_steps:
            recs.append(record(m, st))
            times.append(n * dt)
    wall = (time.perf_counter() - tic) * n_steps / max(1, n_steps - N_WARM) if tic else float("nan")
    s = {k: np.array([r[k] for r in recs]) for k in ("h", "T", "s", "u", "v", "eta")}
    s.update(t=np.array(times), x=c.x, y=c.y, dx=np.full(c.nx, c.dx), dy=np.full(c.ny, c.dx), H=c.H)
    meta = dict(model="pyOCEAN", coord=coord, res_km=res_km, dt=dt, n_steps=n_steps, wall_s=wall, threads=threads,
                n_trunc=int(st.n_trunc), n_limited=int(st.n_limited),
                nu_v=c.nu_v if nu_v is None else nu_v)
    return series.save(out, s, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("coord", choices=["zstar", "sigma", "hybrid"])
    ap.add_argument("--dt", type=float)
    ap.add_argument("--days", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--nu-v", type=float)
    a = ap.parse_args()
    print(run(a.res_km, a.coord, a.dt, a.days, a.tag, a.threads, a.nu_v))


if __name__ == "__main__":
    main()
