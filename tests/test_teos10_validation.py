"""
TEOS-10 in the 3-D core, Task 9 (plan docs/superpowers/plans/2026-10-31-teos10-3d.md; spec
docs/superpowers/specs/2026-10-31-teos10-3d-design.md, criteria 4, 5, 8 and 9): the validation cases of
``validation/teos10/cases.py`` in their fast form (coarse grids, ≤ 1 day); the production runs and the comparison
table are ``validation/teos10/run_pyocean.py`` and ``compare.py`` (README there).

* criterion 4: the WOA23 profile at rest over the seamount and the BayIW bay-mouth step, z* and hybrid, with
  ``rho_ref="auto"``: the spurious currents of ``wright_full`` and ``teos10`` stay within 2× those of the
  equivalent linear EOS (fitted to the same profile, same initial layers), and the initial pressure-gradient force
  as well (the reference profile removes the compressibility and the thermobaric part of ρ);
* criterion 5a: equal-density waters at p = 0 make denser water when they mix (cabbeling), which sinks; the linear
  EOS makes none;
* criterion 5b: cold, fresh water over warm, salty water overturns with the interface at 4000 m and not at 100 m;
* criterion 8: the variance budget of S and T closes to 1e-12 with the three EOS in σ and z*; volume and salt are
  conserved;
* criterion 9: restart and 1 = 5 threads bit for bit with ``wright_full`` and ``teos10`` in σ, z* and hybrid.

Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import PyOceanConfigWarning
from core.model.threads import _THREADS, set_threads
from core.restart import read_restart, write_restart
from validation.teos10 import cases as C

NONLINEAR = ("wright_full", "teos10")


def _run_rest(kind: str, eos: str, coord: str) -> tuple[float, float]:
    """(initial max |PGF| [m s⁻²], max over 1 day of max |u|, |v| [m s⁻¹]) of the fast rest case."""
    case = C.rest_case(kind, eos, coord, fast=True)
    m, st = case.model, case.state
    gx, gy = m.pressure_gradient(st.eta, st.T, st.h, st.S)
    pgf0 = float(max(np.abs(gx).max(), np.abs(gy).max()))
    umax = 0.0
    for _ in range(case.n_steps):
        st = m.step(st)
        umax = max(umax, float(np.abs(st.u).max()), float(np.abs(st.v).max()))
    assert np.isfinite(st.T).all() and (st.h >= 0).all()
    return pgf0, umax


# ---------------------------------------------------------------- criterion 4
@pytest.mark.parametrize("coord", ["zstar", "hybrid"])
@pytest.mark.parametrize("kind", ["seamount", "bayiw_step"])
def test_rest_profile_mode_quiet(kind, coord):
    """With rho_ref='auto' the nonlinear spurious currents are ≤ 2× those of the equivalent linear EOS (and so is the
    initial PGF: no thermobaric or compressibility residual beyond the linear one).  z*: round-off level (no
    interpolation inside a layer); hybrid: thick layers cut by the topography carry the layer mean of a curved
    profile, which PLM does not reconstruct (the same for every EOS, README §criterion 4)."""
    res = {eos: _run_rest(kind, eos, coord) for eos in C.EOS_NAMES}
    pgf_lin, u_lin = res["linear"]
    for eos in NONLINEAR:
        pgf0, umax = res[eos]
        assert umax <= 2.0 * u_lin, (eos, umax, u_lin)
        assert pgf0 <= 2.0 * pgf_lin, (eos, pgf0, pgf_lin)
    if coord == "zstar":
        assert max(r[1] for r in res.values()) < 1e-9


# ---------------------------------------------------------------- criterion 5a
@pytest.mark.parametrize("eos", C.EOS_NAMES)
def test_cabbeling_sinks(eos):
    """Lock exchange of (5 °C, 34.0) and (15 °C, S_B) of equal ρ at p = 0, 6 h: with a nonlinear EOS water denser
    than both appears, its volume and its excess density Σ V (ρ − ρ_src) grow at 2, 4 and 6 h, and its
    excess-weighted height lies below mid-depth (it sinks); the linear EOS makes none and nothing moves."""
    case = C.cabbeling_case(eos, fast=True, days=0.25)
    m, st = case.model, case.state
    _, (tb, sb_), rho_src = C.cabbeling_waters(eos)
    assert abs(C.surface_density(eos, np.array([tb]), np.array([sb_]))[0] - rho_src) < 1e-10
    vols, zs, excess = [], [], []
    every = case.n_steps // 3
    for n in range(1, case.n_steps + 1):
        st = m.step(st)
        if n % every == 0:
            v, z = C.denser_volume(case, st)
            vols.append(v)
            zs.append(z)
            rho = C.surface_density(eos, st.T, st.S)
            excess.append(float((m.layer_volumes(st.eta, st.h) * np.maximum(rho - rho_src, 0.0)).sum()))
    if eos == "linear":
        assert vols == [0.0] * 3
        assert np.abs(st.u).max() < 1e-8                    # equal density, no cabbeling: nothing moves
        return
    depth = float(m.H.max())
    assert vols[0] > 0 and all(b > a for a, b in zip(vols, vols[1:])), vols
    assert all(b > a for a, b in zip(excess, excess[1:])), excess
    assert max(zs) < -0.5 * depth, zs


# ---------------------------------------------------------------- criterion 5b
@pytest.mark.parametrize("eos", C.EOS_NAMES)
def test_thermobaric_overturn(eos):
    """Cold, fresh over warm, salty water (Task 6 column), GLS, 6 h: with a nonlinear EOS the column overturns with
    the interface at 4000 m (most of the bed water's heat crosses the interface) and stays put at 100 m; the linear
    control (α, β of the pair at p = 0) stays put at both depths."""
    rise = {}
    for interface in (100.0, 4000.0):
        case = C.thermobaric_case(eos, interface, fast=True, days=0.25)
        m, st = case.model, case.state
        for _ in range(case.n_steps):
            st = m.step(st)
        assert np.isfinite(st.T).all()
        rise[interface] = C.warm_rise(case, st)
    # stable interface: only the GLS floor κ_t = 1e-6 m² s⁻¹ (CROCO nuws) leaks heat across it, ~√(κ t) = 0.15 m
    # of the 200 m warm layer in 6 h (measured 5e-6); an overturn moves most of it (> 0.5)
    assert rise[100.0] < 1e-3, rise
    if eos == "linear":
        assert rise[4000.0] < 1e-3, rise
    else:
        assert rise[4000.0] > 0.5, rise


# ---------------------------------------------------------------- shared small box (criteria 8 and 9)
NX, NY, NZ = 8, 6, 8
H = (40.0 + 5.0 * np.arange(NX))[None, :] * np.ones((NY, 1))


def _front() -> tuple[np.ndarray, np.ndarray]:
    """θ (or Θ) and S_P (or S_A) with stable stratification and a front in x (the Task 4 box)."""
    k = np.arange(NZ)[:, None, None] / (NZ - 1)
    i = np.arange(NX)[None, None, :] / NX
    return ((10.0 + 4.0 * k + 1.0 * i) * np.ones((NZ, NY, NX)), (35.2 - 0.3 * k - 0.4 * i) * np.ones((NZ, NY, NX)))


def _box(eos: str, coord: str, **kw) -> SigmaBaroclinic:
    base = dict(z0=1e-3, closure="constant", nu_const=1e-4, ah=5.0, kh=20.0, kv=1e-3, theta=0.6, d_min=0.05,
                salinity=True, eos=eos, rho_ref="auto")
    if eos == "linear":
        base.update(s_coef=0.78, s_ref=35.0, t_ref=10.0)
    if coord == "sigma":
        base["pgf"] = "jacobian"
    else:
        base.update(vertical_coordinate="ale", pgf="fv", regrid=coord)
        params = dict(dz_top=np.full(NZ, H.max() / NZ))
        if coord == "hybrid":
            params["rho_target"] = (np.linspace(1025.8, 1027.6, NZ + 1) if eos != "linear"
                                    else np.linspace(999.0, 1000.6, NZ + 1))
        base["regrid_params"] = params
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        return SigmaBaroclinic(create_cartesian_grid(nx=NX, ny=NY, dx=2000.0, dy=2000.0, f0=1e-4), H, dt=60.0,
                               n_layers=NZ, **base)


def _initial(m: SigmaBaroclinic):
    T, S = _front()
    return m.initial_state(np.zeros((NY, NX)), T, S=S, ts_kind=m._eos.ts_kind)


# ---------------------------------------------------------------- criterion 8
@pytest.mark.parametrize("coord", ["sigma", "zstar"])
@pytest.mark.parametrize("eos", C.EOS_NAMES)
def test_variance_budget_closes(eos, coord):
    """Σ V¹ c¹² − Σ V⁰ c⁰² = Δt Σ (surf − num_adv − phy_v − phy_h − num_remap − sponge) to 1e-12 for S and T, every
    step, with wind; volume and salt content unchanged to 1e-13 (closed basin, no freshwater)."""
    m = _box(eos, coord, variance_budget=True)
    st = _initial(m)
    tx, ty = np.full((NY, NX + 1), 0.1), np.full((NY + 1, NX), -0.05)
    V_start = m.layer_volumes(st.eta, st.h)
    salt0 = float((V_start * st.S).sum())
    for _ in range(10):
        V0, S0, T0 = m.layer_volumes(st.eta, st.h), st.S, st.T
        st = m.step(st, tx, ty)
        V1, b = m.layer_volumes(st.eta, st.h), st.budget
        for c, c0, c1 in (("S", S0, st.S), ("T", T0, st.T)):
            lhs = (V1 * c1 ** 2).sum() - (V0 * c0 ** 2).sum()
            rhs = m.dt * (b[f"{c}_surf"].sum() - b[f"{c}_num_adv"].sum() - b[f"{c}_phy_v"].sum()
                          - b[f"{c}_phy_h"].sum() - b.get(f"{c}_num_remap", np.zeros(1)).sum()
                          - b[f"{c}_sponge"].sum())
            assert abs(lhs - rhs) <= 1e-12 * (V0 * c0 ** 2).sum(), (c, lhs, rhs)
    V_end = m.layer_volumes(st.eta, st.h)
    assert abs(V_end.sum() - V_start.sum()) <= 1e-13 * V_start.sum()
    assert abs(float((V_end * st.S).sum()) - salt0) <= 1e-13 * salt0
    assert np.abs(st.u).max() > 0


# ---------------------------------------------------------------- criterion 9
@pytest.mark.parametrize("coord", ["sigma", "zstar", "hybrid"])
@pytest.mark.parametrize("eos", NONLINEAR)
def test_restart_and_threads_bitwise(tmp_path, eos, coord):
    """4 steps on 5 threads = 2 steps on 1 thread + restart into a fresh model + 2 steps on 1 thread, bit for bit
    (stratified GLS k-ε, the closure that uses the locally referenced N² of the nonlinear EOS; wind)."""
    tx, ty = np.full((NY, NX + 1), 0.1), np.full((NY + 1, NX), -0.05)
    make = lambda: _box(eos, coord, closure="gls")              # noqa: E731
    old = _THREADS[0]
    try:
        set_threads(5)
        assert _THREADS[0] == 5
        m = make()
        ref = _initial(m)
        for _ in range(4):
            ref = m.step(ref, tx, ty)
        set_threads(1)
        m1 = make()
        st = _initial(m1)
        for _ in range(2):
            st = m1.step(st, tx, ty)
        write_restart(tmp_path / "r.nc", m1, st)
        m2 = make()
        st = read_restart(tmp_path / "r.nc", m2)
        for _ in range(2):
            st = m2.step(st, tx, ty)
    finally:
        set_threads(old)
    for name in ("eta", "u", "v", "T", "S", "h"):
        a, b = getattr(st, name), getattr(ref, name)
        if a is None:
            assert b is None
            continue
        assert np.array_equal(a, b), name
    assert st.time == ref.time
    assert np.abs(ref.u).max() > 0
