"""
Periodic horizontal boundaries (N3 task 3, spec docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md
item 5; ``SigmaBaroclinic(periodic=...)``, ``core.model.periodic``).

* a doubly periodic 2×2 column under uniform wind and heat stays horizontally uniform (σ and ALE z*), the setting
  of the CROCO 5×5 and MOM6 2×2 single-column cases;
* a tracer carried once around a periodic channel comes back and is conserved;
* η = a cos(2πx/L) in a periodic channel oscillates with the period of the discrete C-grid gravity wave;
* every horizontal operator sees the wrap: shifting the whole problem (bathymetry, state, forcing) by whole cells
  along a periodic axis shifts the solution (an edge treated as a wall breaks this at 1e-4 or more);
* an open boundary on a periodic axis is a pre-flight error; restarts are bit for bit and refuse a different
  ``periodic``; without ``periodic`` the N1 golden matrix is unchanged.

Index order (k, j, i), k = 0 at the bed; u-faces (ny, nx+1), v-faces (ny+1, nx).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError
from core.restart import read_restart, write_restart
from tests.n1_cases import run_case

G = 9.81


def _model(nx, ny, H, nz, dt, per=("x", "y"), f0=1e-4, dx=1000.0, ale=False, **kw):
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=f0)
    base = dict(rho0=1000.0, t_coef=0.2, z0=None, closure="constant", nu_const=1e-3, theta=0.6, d_min=0.05)
    if ale:
        base.update(vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, float(np.max(H)) / nz)),
                    pgf="fv")
    base.update(kw)
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, periodic=per, **base)


# ---------------------------------------------------------------- uniform column
@pytest.mark.parametrize("ale", [False, True], ids=["sigma", "zstar"])
def test_periodic_column_stays_uniform(ale):
    nz, H = 8, np.full((2, 2), 40.0)
    m = _model(2, 2, H, nz, dt=60.0, ale=ale, kv=1e-4)
    T = (12.0 + 4.0 * np.arange(nz) / (nz - 1))[:, None, None] * np.ones((nz, 2, 2))
    st = m.initial_state(np.zeros((2, 2)), T)
    tx, ty = np.full((2, 3), 0.1), np.full((3, 2), 0.05)              # uniform wind stress [N m⁻²]
    heat = {"Q_ns": np.full((2, 2), -50.0), "Q_sw": np.full((2, 2), 200.0)}
    for _ in range(200):
        st = m.step(st, tx, ty, heat=heat)
    fields = [("eta", st.eta), ("u", st.u), ("v", st.v), ("T", st.T)] + ([("h", st.h)] if ale else [])
    for name, a in fields:
        assert np.max(np.abs(a - a[..., :1, :1])) <= 1e-12, name          # every column (and face) equal
    assert np.max(np.abs(st.u)) > 1e-3 and np.max(np.abs(st.v)) > 1e-3       # the column is driven
    assert np.array_equal(st.u[..., 0], st.u[..., -1]) and np.array_equal(st.v[:, 0], st.v[:, -1])
    assert st.T[-1].mean() != T[-1].mean()                                    # heat entered


# ---------------------------------------------------------------- channel tracer
def test_periodic_x_tracer_returns():
    nx, ny, nz, dx, U = 40, 2, 2, 1000.0, 0.5
    m = _model(nx, ny, np.full((ny, nx), 20.0), nz, dt=200.0, per=("x",), f0=0.0, t_coef=0.0, closure="none")
    x = np.arange(nx)
    c0 = 1.0 + np.exp(-0.5 * ((x - 12.0) / 3.0) ** 2)
    st = m.initial_state(np.zeros((ny, nx)), c0[None, None, :] * np.ones((nz, ny, nx)))
    st.u[:] = U                                                           # uniform flow, also through the wrap face
    content0 = m.tracer_content(st)
    steps = int(round(nx * dx / U / m.dt))                                # one lap: 400 steps
    for _ in range(steps):
        st = m.step(st)
    assert np.max(np.abs(st.u - U)) < 1e-12 and np.max(np.abs(st.eta)) < 1e-12
    assert m.tracer_content(st) == pytest.approx(content0, rel=1e-12, abs=0)
    c = st.T[0, 0]
    assert int(np.argmax(c)) == int(np.argmax(c0))                         # back where it started
    assert np.max(np.abs(c - c0)) < 0.05                                   # WENO5 after 40 cells
    assert np.allclose(st.T, st.T[:1, :1], rtol=0, atol=1e-12)


# ---------------------------------------------------------------- gravity wave
@pytest.mark.parametrize("ale", [False, True], ids=["sigma", "zstar"])
def test_periodic_free_surface_wave_period(ale):
    nx, dx, H0, a, dt = 20, 1000.0, 10.0, 1e-3, 10.0
    m = _model(nx, 1, np.full((1, nx), H0), 2, dt=dt, per=("x",), f0=0.0, theta=0.5, closure="none", ale=ale,
               t_coef=0.0)
    L = nx * dx
    xc = (np.arange(nx) + 0.5) * dx
    st = m.initial_state(a * np.cos(2 * np.pi * xc / L)[None, :], np.full((2, 1, nx), 10.0))
    t, e = [0.0], [st.eta[0, 0]]
    for _ in range(int(3.2 * 2030 / dt)):
        st = m.step(st)
        t.append(st.time)
        e.append(st.eta[0, 0])
    t, e = np.array(t), np.array(e)
    up = np.nonzero((e[:-1] < 0) & (e[1:] >= 0))[0]                      # upward zero crossings
    tc = t[up] - e[up] * (t[up + 1] - t[up]) / (e[up + 1] - e[up])
    period = float(np.mean(np.diff(tc)))
    # C-grid dispersion ω = (2 √(gH) / Δx) sin(k Δx / 2), k = 2π/L (centred in time with θ = ½)
    omega = 2.0 * np.sqrt(G * H0) / dx * np.sin(np.pi * dx / L)
    assert len(tc) >= 2
    assert period == pytest.approx(2 * np.pi / omega, rel=0.01)
    assert np.max(np.abs(st.u[..., 0] - st.u[..., -1])) == 0.0


# ---------------------------------------------------------------- translation invariance
NXr, NYr, NZr = 7, 6, 5


def _roll(a, kind, sx, sy):
    """Shift by (sx, sy) cells; u/v keep their duplicated wrap face equal to face 0."""
    if kind == "c":
        return np.roll(np.roll(a, sx, axis=-1), sy, axis=-2)
    if kind == "u":
        b = _roll(a[..., :-1], "c", sx, sy)
        return np.concatenate([b, b[..., :1]], axis=-1)
    b = _roll(a[..., :-1, :], "c", sx, sy)
    return np.concatenate([b, b[..., :1, :]], axis=-2)


ROLL_CASES = {
    "sigma_flux_gls_kh_ah": dict(closure="gls", z0=1e-3, ah=5.0, kh=2.0),
    "sigma_fv_upwind3_pc": dict(pgf="fv", adv="upwind3", tracer_time="pc"),
    "zstar_vi_fct_tracer_drag": dict(ale=True, ah=5.0, kh=2.0, kv=1e-5, n_tracers=1, bottom_drag="quadratic",
                                     ke_scheme="godunov"),
    "zstar_flux_upwind": dict(ale=True, momentum="flux", thickness_flux="upwind"),
}


@pytest.mark.parametrize("per", [("x", "y"), ("x",), ("y",)], ids=["xy", "x", "y"])
@pytest.mark.parametrize("case", list(ROLL_CASES))
def test_periodic_operators_are_translation_invariant(case, per):
    rng = np.random.default_rng(1)
    sx, sy = (3 if "x" in per else 0), (2 if "y" in per else 0)
    H = 30.0 + 5.0 * rng.random((NYr, NXr))                              # non-uniform bed, shifted with the rest
    T = 10 + 3 * np.arange(NZr)[:, None, None] / (NZr - 1) + 0.5 * rng.random((NZr, NYr, NXr))
    eta = 0.05 * rng.standard_normal((NYr, NXr))
    uu, vv = 0.05 * rng.standard_normal((2, NZr, NYr, NXr))
    u, v = np.concatenate([uu, uu[..., :1]], -1), np.concatenate([vv, vv[:, :1]], 1)
    if "x" not in per:
        u[..., 0] = u[..., -1] = 0.0
    if "y" not in per:
        v[:, 0] = v[:, -1] = 0.0
    tx = 0.1 * rng.standard_normal((NYr, NXr))
    ty = 0.1 * rng.standard_normal((NYr, NXr))
    tx, ty = np.concatenate([tx, tx[:, :1]], 1), np.concatenate([ty, ty[:1]], 0)
    kw = dict(ROLL_CASES[case])
    ale = kw.pop("ale", False)
    tr0 = rng.random((1, NZr, NYr, NXr)) if kw.get("n_tracers") else None
    runs = []
    for shift in ((0, 0), (sx, sy)):
        f = lambda a, kind: _roll(a, kind, *shift)                         # noqa: E731
        m = _model(NXr, NYr, f(H, "c"), NZr, dt=30.0, per=per, ale=ale, **kw)
        st = m.initial_state(f(eta, "c"), f(T, "c"), tr=None if tr0 is None else f(tr0, "c"))
        st.u, st.v = f(u, "u"), f(v, "v")
        for _ in range(6):
            st = m.step(st, f(tx, "u"), f(ty, "v"))
        runs.append(st)
    a, b = runs
    for name, kind in (("eta", "c"), ("u", "u"), ("v", "v"), ("T", "c"), ("h", "c"), ("tr", "c")):
        x, y = getattr(a, name), getattr(b, name)
        if x is not None:
            assert np.max(np.abs(_roll(x, kind, sx, sy) - y)) < 1e-12, name
    if "x" in per:
        assert np.array_equal(a.u[..., 0], a.u[..., -1])
    if "y" in per:
        assert np.array_equal(a.v[:, 0], a.v[:, -1])
    assert np.max(np.abs(a.u)) > 0.01


# ---------------------------------------------------------------- configuration
def test_periodic_with_open_edge_same_axis_is_an_error():
    H = np.full((4, 5), 20.0)
    with pytest.raises(ConfigurationError, match="same axis 'x'"):
        _model(5, 4, H, 3, dt=30.0, per=("x",), open_boundaries=("west",))
    with pytest.raises(ConfigurationError, match="same axis 'y'"):
        _model(5, 4, H, 3, dt=30.0, per=("x", "y"), open_boundaries=("north",))
    with pytest.raises(ConfigurationError, match="unknown periodic axes"):
        _model(5, 4, H, 3, dt=30.0, per=("z",))
    m = _model(5, 4, H, 3, dt=30.0, per=("x",), open_boundaries=("south", "north"))    # the other axis: fine
    st = m.initial_state(np.zeros((4, 5)), np.full((3, 4, 5), 10.0))
    st = m.step(st, np.full((4, 6), 0.1), None)
    assert np.isfinite(st.u).all()


def test_periodic_restart_bit_for_bit(tmp_path):
    def make(per=("x", "y")):
        return _model(4, 3, 30.0 + np.arange(12.0).reshape(3, 4), 4, dt=30.0, per=per, ale=True, closure="constant",
                      kh=1.0)

    def start(m):
        T = 10 + np.arange(4.0)[:, None, None] + 0.1 * np.arange(12.0).reshape(3, 4)
        return m.initial_state(0.01 * np.cos(np.arange(12.0)).reshape(3, 4), T)

    tx = np.full((3, 5), 0.1)
    m = make()
    st = start(m)
    for _ in range(6):
        st = m.step(st, tx, None)
    m1 = make()
    st1 = start(m1)
    for _ in range(3):
        st1 = m1.step(st1, tx, None)
    write_restart(tmp_path / "rst.nc", m1, st1)
    m2 = make()
    st2 = read_restart(tmp_path / "rst.nc", m2)
    for _ in range(3):
        st2 = m2.step(st2, tx, None)
    for name in ("eta", "u", "v", "T", "h"):
        assert np.array_equal(getattr(st2, name), getattr(st, name)), name
    with pytest.raises(ValueError, match="periodic"):
        read_restart(tmp_path / "rst.nc", make(per=("x",)))


GOLDEN = json.loads((Path(__file__).resolve().parents[1] / "tests" / "data" / "n1_golden" / "hashes.json").read_text())


@pytest.mark.parametrize("name", ["sigma_jacobian", "sigma_gls", "zstar_vi_arakawa", "hybrid_default"])
def test_no_periodic_is_bitwise(name):
    got = run_case(name)
    for step, ref in GOLDEN[name].items():
        assert got[int(step)] == ref, f"{name} step {step}"
