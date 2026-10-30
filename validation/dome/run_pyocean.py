#!/usr/bin/env python
"""
pyOCEAN runs of the DOME benchmark (spec 2026-10-14-dome-overflow §5): ALE z*, σ or hybrid with the model defaults
(vector-invariant momentum, KE Arakawa, WENO5-Z + FCT, PPM_H4, PPM continuity, finite-volume PGF), the common physics
of the case (Laplacian ν_h, ν_v, no tracer diffusion, quadratic drag C_d over hbbl), one passive tracer s (the inflow
dye), the interior sponges of MOM6 DOME and the north inflow as a 'clamped' edge with its exact transport.

``--closure gls`` (N3 task 10, spec 2026-10-29-n3-turbulence-closures decision 10): the stratified k-ε GLS
(``closure='gls'``, ``gls_stratified=True``) gives ν and the tracer diffusivity κ_t instead of the constant ν_v; only
the closure changes from the baseline — the quadratic drag C_d over ``hbbl`` is the same, and its u*² is the bed
condition of the GLS (``core.model.driver._ale_bed_ustar2``); ``kv`` = 0, so κ_t keeps the GLS floor (CROCO nuws
= 1e-6 m² s⁻¹, gls_mixing.F:152), and ν its floor 1e-5.  Its outputs go to the run directory with the tag 'gls'
(``--tag`` replaces it), next to the baseline.

Usage: NUMBA_NUM_THREADS=5 python validation/dome/run_pyocean.py 10 zstar [--dt 600 (default)] [--days 50] [--tag x]
           [--closure {constant,gls}] [--smag C] [--kv-low]
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
DT_MAX = 600.0      # [s] the 10-km campaign stops at day 36 with 1200 s (θ_bc = ½ instability, README point 5)


def default_dt(case: cases.Case) -> float:
    """pyOCEAN's default step: the case step (MOM6's), capped at DT_MAX."""
    return min(case.dt, DT_MAX)


CLOSURES = ("constant", "gls")


def closure_kwargs(case: cases.Case, closure: str, nu_v: float | None = None) -> dict:
    """
    Model arguments of the vertical closure: 'constant' (the baseline, ν = ν_v, no tracer diffusion) or 'gls'
    (module docstring; spec decision 10: kv = 0 → the GLS κ_t floor 1e-6, nothing else changes).
    """
    if closure == "constant":
        return dict(closure="constant", nu_const=case.nu_v if nu_v is None else nu_v)
    if closure == "gls":
        if nu_v is not None:
            raise ValueError("--nu-v applies to closure 'constant' only (the GLS computes ν)")
        return dict(closure="gls", gls_stratified=True, kv=0.0)
    raise ValueError(f"unknown closure {closure!r}; use one of {CLOSURES}")


def build(case: cases.Case, coord: str, dt: float | None = None, nu_v: float | None = None,
          closure: str = "constant", smag: float = 0.0):
    """The ALE model of ``case`` in ``coord`` ∈ {'zstar', 'sigma', 'hybrid'} with the vertical ``closure``
    ∈ {'constant', 'gls'} and its resting initial state.
    ``smag`` is the Smagorinsky coefficient C (0 = off; the background ν_h = ``case.nu_h`` is its floor)."""
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
    m = SigmaBaroclinic(hg, case.H.copy(), dt=default_dt(case) if dt is None else dt, n_layers=nz, rho0=cases.RHO0,
                        t_coef=cases.ALPHA, t_ref=cases.T_REF, z0=None, ah=case.nu_h, g=cases.G, pgf="fv",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params, n_tracers=1,
                        open_boundaries=("north",), open_boundary_mode={"north": "clamped"},
                        open_boundary_state=state, sponge_rate=cases.sponge_rate(case),
                        sponge_T=cases.T_ambient(0.5 * (zs[1:] + zs[:-1])), sponge_zw=zs,
                        bottom_drag="quadratic", cd=case.cd, hbbl=case.hbbl,
                        smagorinsky=float(smag), **closure_kwargs(case, closure, nu_v))
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


def run(res_km: float, coord: str, dt: float | None = None, days: float | None = None, tag: str | None = None,
        threads: int = 5, nu_v: float | None = None, closure: str = "constant", write: bool = True,
        smag: float = 0.0, kv_low: bool = False) -> Path | dict:
    """Run and write cases.run_dir(res_km, 'pyocean', coord, tag)/series.npz; returns its path.  ``tag`` None:
    '' for the constant closure, 'gls' for the GLS.  ``write`` False (cost measurement): nothing is written and the
    run metadata are returned.  ``smag`` > 0 (Smagorinsky C) appends '_smag<C>' to the default tag
    ('smag0.15', 'gls_smag0.15'); ``kv_low`` is the variant ν_v = cases.NU_V_LOW (constant closure only; default tag
    'kv_low', so the baseline is not overwritten).  With smag > 0 the meta also hold ``smag``, ``n_visc_limited``
    and ``nu_t_max`` (max ν at the output times)."""
    set_threads(threads)
    c = cases.get(res_km)
    if kv_low:
        if closure != "constant" or nu_v is not None:
            raise ValueError("kv_low sets ν_v = 1e-4 of the constant closure; do not combine with --closure gls / --nu-v")
        nu_v = cases.NU_V_LOW
    if tag is None:
        tag = ("kv_low" if kv_low else "" if closure == "constant" else closure)
        if smag > 0.0:
            tag = f"{tag}_smag{smag:g}" if tag else f"smag{smag:g}"
    out = cases.run_dir(res_km, "pyocean", coord, tag) / "series.npz"
    if write:
        out.unlink(missing_ok=True)
    dt = default_dt(c) if dt is None else dt
    days = c.days if days is None else days
    m, st = build(c, coord, dt, nu_v, closure, smag)
    n_steps = int(round(days * 86400.0 / dt))
    every = max(1, int(round(c.out_s / dt)))
    recs, times, tic, nu_max = [record(m, st)], [0.0], None, float("nan")
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
    s = {k: np.array([r[k] for r in recs]) for k in ("h", "T", "s", "u", "v", "eta")}
    s.update(t=np.array(times), x=c.x, y=c.y, dx=np.full(c.nx, c.dx), dy=np.full(c.ny, c.dx), H=c.H)
    meta = dict(model="pyOCEAN", coord=coord, res_km=res_km, dt=dt, n_steps=n_steps, wall_s=wall, threads=threads,
                n_trunc=int(st.n_trunc), n_limited=int(st.n_limited),
                nu_v=(c.nu_v if nu_v is None else nu_v) if closure == "constant" else None, closure=closure, T_min=float(np.nanmin(s["T"])),
                T_max=float(np.nanmax(s["T"])))
    if smag > 0.0:
        meta.update(smag=smag, n_visc_limited=int(st.n_visc_limited or 0), nu_t_max=float(nu_max))
    if not write:
        return meta
    return series.save(out, s, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("coord", choices=["zstar", "sigma", "hybrid"])
    ap.add_argument("--dt", type=float)
    ap.add_argument("--days", type=float)
    ap.add_argument("--tag", default=None, help="run-directory tag (default '' constant, 'gls' GLS)")
    ap.add_argument("--threads", type=int, default=5)
    ap.add_argument("--nu-v", type=float)
    ap.add_argument("--closure", choices=CLOSURES, default="constant")
    ap.add_argument("--smag", type=float, default=0.0, help="Smagorinsky C (default 0 = off)")
    ap.add_argument("--kv-low", action="store_true", help="variant kv_low: constant ν_v = 1e-4 (tag 'kv_low')")
    ap.add_argument("--cost", action="store_true", help="cost measurement: write nothing, print the metadata")
    a = ap.parse_args()
    print(run(a.res_km, a.coord, a.dt, a.days, a.tag, a.threads, a.nu_v, a.closure, write=not a.cost, smag=a.smag, kv_low=a.kv_low))


if __name__ == "__main__":
    main()
