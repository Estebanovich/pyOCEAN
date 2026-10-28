"""
Golden-matrix cases for N1 (plan docs/superpowers/plans/2026-10-28-n1-component-core.md, task 0).

Small configurations of ``SigmaBaroclinic`` that cover every branch the 3-D core distinguishes (σ / z* / hybrid /
ALE-σ, tracer time scheme, momentum form, KE scheme, thickness flux, PGF, closure, drag, open boundaries, sponges,
passive tracers, mixing diagnostic, thin-layer coupling).  ``run_case`` returns SHA-256 hashes of every state field
after a few steps; ``tests/data/n1_golden/hashes.json`` holds the values produced by ``main`` before the N1
refactoring, so any change of a single bit in any field fails ``tests/test_n1_golden.py``.

Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import dataclasses
import hashlib
from collections.abc import Callable

import numpy as np

from boundaries.obcs import OBData
from core.dynamics import sigma_baroclinic as sb
from core.dynamics.sigma_baroclinic import SigmaBaroclinic, SigmaState
from core.grid.horizontal import create_cartesian_grid

NX, NY, NZ = 10, 6, 6
DX, DT = 2000.0, 60.0
RHO0, TCOEF = 1000.0, 0.2
H = (20.0 + 2.0 * np.arange(NX))[None, :] * np.ones((NY, 1))          # 20 … 38 m, sloping in x
HMAX = float(H.max())
STEPS = (1, 2, 3, 10)

Case = tuple[SigmaBaroclinic, SigmaState, Callable[[int], tuple[np.ndarray | None, np.ndarray | None]]]


def _grid(f0: float = 1e-4):
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DX, f0=f0)


def _T(nz: int = NZ) -> np.ndarray:
    """Stable stratification (warm on top, k = 0 at the bed) plus a weak front in x."""
    k = np.arange(nz)[:, None, None]
    i = np.arange(NX)[None, None, :]
    return (10.0 + 3.0 * k / (nz - 1) + 0.2 * i / NX) * np.ones((nz, NY, NX))


def _eta() -> np.ndarray:
    x = (np.arange(NX) + 0.5) / NX
    y = (np.arange(NY) + 0.5) / NY
    return 0.05 * np.cos(np.pi * x)[None, :] * np.cos(np.pi * y)[:, None]


def _wind(on: bool) -> Callable[[int], tuple[np.ndarray | None, np.ndarray | None]]:
    tx = np.full((NY, NX + 1), 0.05)
    ty = np.full((NY + 1, NX), -0.02)
    return (lambda n: (tx, ty)) if on else (lambda n: (None, None))


def _sigma(**kw) -> SigmaBaroclinic:
    base = dict(rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6, d_min=0.05)
    base.update(kw)
    return SigmaBaroclinic(_grid(), H, dt=DT, n_layers=NZ, **base)


def _zstar_params(nz: int = NZ) -> dict:
    return dict(dz_top=np.full(nz, HMAX / nz))


def _hybrid_params(nz: int = 8) -> dict:
    rho = RHO0 - TCOEF * np.array([13.3, 9.9])                          # densities of the top and bottom water
    return dict(dz_top=np.full(nz, HMAX / nz), rho_target=np.linspace(rho[0] - 0.13, rho[1] + 0.17, nz + 1))


def _ale(rule: str = "zstar", nz: int = NZ, params: dict | None = None, **kw) -> SigmaBaroclinic:
    if params is None:
        params = {"zstar": _zstar_params(nz), "hybrid": _hybrid_params(nz), "sigma": {}}[rule]
    base = dict(rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6,
                d_min=0.05, vertical_coordinate="ale", regrid=rule, regrid_params=params, pgf="fv")
    base.update(kw)
    return SigmaBaroclinic(_grid(), H, dt=DT, n_layers=nz, **base)


def _west_state(nz_data: int = 8, tracers: int = 0):
    """Time-varying west exterior on its own z grid: tilted T and a mode-1 normal velocity."""
    depth = float(H[0, 0])
    zw = np.linspace(0.0, -depth, nz_data + 1)[:, None].repeat(NY, 1)
    zc = 0.5 * (zw[1:] + zw[:-1])

    def state(t: float) -> dict:
        T = 13.0 + 0.1 * zc + 0.2 * np.sin(2 * np.pi * t / 3600.0)
        u = 0.05 * np.cos(np.pi * zc / depth) * np.cos(2 * np.pi * t / 3600.0)
        tr = None if not tracers else np.ones((tracers, nz_data, NY))
        return {"west": OBData(side="west", T=T, u_normal=u, zw=zw, tracers=tr)}
    return state


def _north_inflow(nz_data: int = 4):
    """'clamped' north edge with an exact prescribed transport (MOM6 SIMPLE)."""
    Hn = H[-1]
    zw = np.linspace(0.0, -1.0, nz_data + 1)[:, None] * Hn[None, :]
    q = np.full(NX, -20.0)

    def state(t: float) -> dict:
        return {"north": OBData(side="north", T=np.full((nz_data, NX), 11.0),
                                u_normal=(q / (Hn * DX))[None, :].repeat(nz_data, 0), zw=zw, transport=q)}
    return state


def _hybrid_src():
    """Fine source z-levels (1 m), bed first, clipped at the bed, and T on them."""
    zf = -np.arange(0.0, HMAX + 1.0, 1.0)[::-1]
    h_src = np.clip(zf[1:, None, None], -H[None], None) - np.clip(zf[:-1, None, None], -H[None], None)
    zc = -H[None] + np.cumsum(h_src, axis=0) - 0.5 * h_src
    h_src[-1] += _eta()                                                  # Σ h_src = H + η
    return h_src, 13.3 + 0.09 * zc


def _start(m: SigmaBaroclinic, wind: bool, **init) -> Case:
    nz = m.core.n_layers
    T = init.pop("T", None)
    st = m.initial_state(_eta(), _T(nz) if T is None else T, **init)
    return m, st, _wind(wind)


def _sponge_kw(ntr: int = 0) -> dict:
    r = np.zeros((NY, NX))
    r[:, -3:] = 2e-4
    kw = dict(sponge_rate=r, sponge_T=np.array([13.0, 12.0, 11.0, 10.5]),
              sponge_zw=np.linspace(0.0, -HMAX, 5))
    if ntr:
        kw["sponge_tr"] = np.full((ntr, 4), 0.5)
    return kw


def _case_builders() -> dict[str, Callable[[], Case]]:
    c: dict[str, Callable[[], Case]] = {}
    # ---------------- σ (Eulerian)
    c["sigma_jacobian"] = lambda: _start(_sigma(pgf="jacobian"), True)
    c["sigma_twoterm"] = lambda: _start(_sigma(pgf="twoterm"), False)
    c["sigma_fv"] = lambda: _start(_sigma(pgf="fv"), True)
    c["sigma_gls"] = lambda: _start(_sigma(closure="gls", z0=1e-3), True)
    c["sigma_parabolic"] = lambda: _start(_sigma(closure="parabolic", z0=1e-3), False)
    c["sigma_loglaw"] = lambda: _start(_sigma(z0=1e-3), True)
    c["sigma_pc"] = lambda: _start(_sigma(tracer_time="pc"), False)
    c["sigma_upwind3_adv"] = lambda: _start(_sigma(adv="upwind3"), True)
    c["sigma_centred_adv"] = lambda: _start(_sigma(adv="centred"), False)
    c["sigma_open_noprovider"] = lambda: _start(_sigma(open_boundaries=("west", "south"), sponge_cells=3), True)
    c["sigma_obc_state"] = lambda: _start(_sigma(open_boundaries=("west",), sponge_cells=3,
                                                 open_boundary_state=_west_state()), False)
    c["sigma_kh_kv"] = lambda: _start(_sigma(kh=5.0, kv=1e-5), True)
    c["sigma_rho_ref_auto"] = lambda: _start(_sigma(rho_ref="auto"), False)
    # ---------------- z*
    c["zstar_vi_arakawa"] = lambda: _start(_ale("zstar"), True)
    c["zstar_vi_godunov"] = lambda: _start(_ale("zstar", ke_scheme="gudonov"), False)
    c["zstar_flux"] = lambda: _start(_ale("zstar", momentum="flux"), True)
    c["zstar_upwind_thick"] = lambda: _start(_ale("zstar", thickness_flux="upwind"), False)
    c["zstar_pc"] = lambda: _start(_ale("zstar", tracer_time="pc"), True)
    c["zstar_quadratic_drag"] = lambda: _start(_ale("zstar", bottom_drag="quadratic", cd=2e-3, hbbl=5.0), True)
    c["zstar_sponge_interior"] = lambda: _start(_ale("zstar", n_tracers=1, **_sponge_kw(1)), False,
                                                tr=np.zeros((1, NZ, NY, NX)))
    c["zstar_tracers2"] = lambda: _start(_ale("zstar", n_tracers=2, kh=5.0, kv=1e-5), True,
                                         tr=np.stack([_T() / 13.0, np.ones((NZ, NY, NX))]))
    c["zstar_nomixdiag"] = lambda: _start(_ale("zstar", mix_diagnostic=False), False)
    c["zstar_obc_radiating"] = lambda: _start(_ale("zstar", open_boundaries=("west",), sponge_cells=3,
                                                   open_boundary_state=_west_state()), True)
    c["zstar_obc_clamped_transport"] = lambda: _start(_ale("zstar", open_boundaries=("north",),
                                                           open_boundary_mode={"north": "clamped"},
                                                           open_boundary_state=_north_inflow()), False)
    # ---------------- hybrid
    c["hybrid_default"] = lambda: _start(_ale("hybrid", nz=8, rho_ref="auto"), True)
    c["hybrid_cutoff"] = lambda: _start(_ale("hybrid", nz=8, thin_layer_cutoff=0.5), False)
    c["hybrid_coupling_none"] = lambda: _start(_ale("hybrid", nz=8, thin_layer_coupling="none"), True)
    c["hybrid_hsrc_init"] = lambda: (lambda hs: _start(_ale("hybrid", nz=8), False, T=hs[1], h_src=hs[0]))(
        _hybrid_src())
    c["hybrid_open_radiating"] = lambda: _start(_ale("hybrid", nz=8, open_boundaries=("west",), sponge_cells=3,
                                                     open_boundary_state=_west_state()), True)
    # ---------------- ALE σ
    c["alesigma_fv"] = lambda: _start(_ale("sigma"), True)
    return c


CASES = _case_builders()
REUSE_CASE = "zstar_reuse"           # same model, two initial_state calls; hashes of the second run


def _h(a) -> str:
    return hashlib.sha256(np.ascontiguousarray(a).tobytes()).hexdigest()


def state_hashes(st: SigmaState, model: SigmaBaroclinic | None = None) -> dict[str, str]:
    """
    SHA-256 of every array of the state (bytes of the C-contiguous array, so dtype and every bit count), the AB
    history flattened as ``g_hist.{i}.{u|v}``, the closure state as ``closure.{name}`` and the scalars by ``repr``.
    Before N1 task 2 the GLS state lives in ``model.core.gls``; it is hashed under the same keys.
    """
    out: dict[str, str] = {}
    for f in dataclasses.fields(st):
        v = getattr(st, f.name)
        if v is None:
            continue
        if f.name == "g_hist":
            for i, (gu, gv) in enumerate(v):
                out[f"g_hist.{i}.u"], out[f"g_hist.{i}.v"] = _h(gu), _h(gv)
        elif isinstance(v, dict):
            for k, a in v.items():
                out[f"{f.name}.{k}"] = _h(a)
        elif isinstance(v, np.ndarray):
            out[f.name] = _h(v)
        else:
            out[f.name] = repr(v)
    gls = getattr(getattr(model, "core", None), "gls", None)
    if gls is not None and "closure.tke" not in out:
        for k in ("tke", "eps", "akv"):
            out[f"closure.{k}"] = _h(getattr(gls, k))
    return out


def _advance(m: SigmaBaroclinic, st: SigmaState, wind, steps=STEPS) -> dict[int, dict[str, str]]:
    out, n = {}, 0
    for target in steps:
        while n < target:
            tx, ty = wind(n)
            st = m.step(st, tau_x=tx, tau_y=ty)
            n += 1
        out[target] = state_hashes(st, m)
    return out


def run_case(name: str, steps=STEPS) -> dict[int, dict[str, str]]:
    sb.set_threads(1)
    if name == REUSE_CASE:
        m, st, wind = CASES["zstar_vi_arakawa"]()
        _advance(m, st, wind, steps=(5,))
        st2 = m.initial_state(_eta(), _T())
        return _advance(m, st2, wind, steps)
    m, st, wind = CASES[name]()
    return _advance(m, st, wind, steps)


def all_case_names() -> list[str]:
    return sorted(CASES) + [REUSE_CASE]


def diff_fields(got: dict[str, str], ref: dict[str, str]) -> list[str]:
    keys = sorted(set(got) | set(ref))
    return [k for k in keys if got.get(k) != ref.get(k)]
