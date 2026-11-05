"""
Biharmonic Smagorinsky viscosity in the 3-D model (``SigmaBaroclinic(..., smagorinsky_bi=C, ah_vel_scale=…,
ah_bi=…)``; regional chain, Gulf Stream fix step 4).  The operator is tested in tests/test_hor_visc_biharmonic.py;
here the wiring: off = bit for bit, periodic halos, σ ≡ ALE for fixed layers, counter, restart, pre-flight,
open edges with both corner modes.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.dynamics import sigma_baroclinic as sb
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError
from core.model.registry import REGISTRY
from core.restart import read_restart, write_restart
from tests import n1_cases as nc
from tests.test_sigma_ale import _ale_open_box

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "data" / "n1_golden" / "hashes.json").read_text())
RHO0, TCOEF = 1000.0, 0.2


def _hvisc(m, st):
    terms, _, _ = m._momentum_terms(st)
    return {name: (tu, tv) for name, tu, tv, _ in terms}["hviscosity"]


def _periodic_box(nx=16, ny=12, nz=2, H=30.0, **kw):
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((ny, nx), H), dt=60.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, H / nz)), pgf="fv", periodic=("x", "y"), **kw)
    return m, m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 10.0))


# ------------------------------------------------------------------ off = bit for bit
@pytest.mark.parametrize("name, build", [
    ("zstar_vi_arakawa", lambda: nc._start(nc._ale("zstar", smagorinsky_bi=0.0, ah_vel_scale=0.0, ah_bi=0.0), True)),
    ("sigma_gls", lambda: nc._start(nc._sigma(closure="gls", z0=1e-3, gls_stratified=False, smagorinsky_bi=0.0),
                                    True)),
])
def test_zero_biharmonic_is_bitwise(name, build):
    sb.set_threads(1)
    m, st, wind = build()
    got = nc._advance(m, st, wind)
    for step, ref in GOLDEN[name].items():
        assert got[int(step)] == ref, nc.diff_fields(got[int(step)], ref)


# ------------------------------------------------------------------ doubly periodic
@pytest.mark.parametrize("axis", ["x", "y"])
def test_periodic_sinusoid_decays_at_rate_ah_k4_everywhere(axis):
    # Constant A_h on a doubly periodic grid: every face, the wrap faces included, gets the exact discrete
    # −A_h (4 sin²(kΔ/2)/Δ²)² u (halo of 3 in the pad–crop).
    nx = ny = 16
    A, d = 1e8, 1000.0
    m, st = _periodic_box(nx, ny, ah_bi=A)
    k = 2 * np.pi / (8 * d)
    if axis == "x":
        st.u = np.broadcast_to(0.1 * np.cos(k * np.arange(nx + 1) * d)[None, None, :], st.u.shape).copy()
    else:
        st.u = np.broadcast_to(0.1 * np.cos(k * (np.arange(ny) + 0.5) * d)[None, :, None], st.u.shape).copy()
    st.v = np.zeros_like(st.v)
    fu, fv = _hvisc(m, st)
    exact = -A * (4 * np.sin(k * d / 2) ** 2 / d ** 2) ** 2 * st.u
    np.testing.assert_allclose(fu, exact, rtol=0, atol=1e-10 * np.abs(exact).max())
    np.testing.assert_allclose(fv, 0.0, atol=1e-10 * np.abs(exact).max())
    assert np.all(m._momentum.last_ah_t == A)


def test_periodic_conserves_momentum_and_dissipates():
    m, st = _periodic_box(smagorinsky_bi=0.03, ah_vel_scale=0.01)
    rng = np.random.default_rng(3)
    st.u = 0.1 * rng.standard_normal(st.u.shape)
    st.v = 0.1 * rng.standard_normal(st.v.shape)
    st.u[..., -1] = st.u[..., 0]
    st.v[:, -1] = st.v[:, 0]
    hu, hv = m._mean_face_thickness(st.h)
    Au, Av = m.hg["area_u"].values, m.hg["area_v"].values
    mom0 = (hu * st.u * Au)[..., :-1].sum()
    ke = lambda s: ((hu * s.u ** 2 * Au)[..., :-1].sum() + (hv * s.v ** 2 * Av)[:, :-1].sum())  # noqa: E731
    ke0 = ke(st)
    for _ in range(30):
        fu, fv = _hvisc(m, st)
        assert np.array_equal(fu[..., 0], fu[..., -1]) and np.array_equal(fv[:, 0], fv[:, -1])
        st.u, st.v = st.u + m.dt * fu, st.v + m.dt * fv
    assert abs((hu * st.u * Au)[..., :-1].sum() - mom0) <= 1e-12 * (hu * np.abs(st.u) * Au).sum()
    assert ke(st) < 0.99 * ke0


# ------------------------------------------------------------------ σ ≡ ALE for fixed layers
def test_sigma_and_ale_fixed_layers_agree():
    nx, ny, nz, H = 12, 9, 4, 40.0
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=800.0, dy=1000.0, f0=1e-4)
    common = dict(rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant", nu_const=1e-4, pgf="fv",
                  smagorinsky_bi=0.03, ah_vel_scale=0.01)
    ms = SigmaBaroclinic(hg, np.full((ny, nx), H), dt=60.0, n_layers=nz, **common)
    ma = SigmaBaroclinic(hg, np.full((ny, nx), H), dt=60.0, n_layers=nz, vertical_coordinate="ale",
                         regrid="zstar", regrid_params=dict(dz_top=np.full(nz, H / nz)), **common)
    T = np.full((nz, ny, nx), 10.0)
    ss, sa = ms.initial_state(np.zeros((ny, nx)), T), ma.initial_state(np.zeros((ny, nx)), T)
    rng = np.random.default_rng(7)
    u = 0.2 * rng.standard_normal((nz, ny, nx + 1))
    v = 0.2 * rng.standard_normal((nz, ny + 1, nx))
    u[..., 0] = u[..., -1] = 0.0
    v[:, 0] = v[:, -1] = 0.0
    ss.u, ss.v, sa.u, sa.v = u, v, u.copy(), v.copy()
    fs_, gs_ = _hvisc(ms, ss)
    fa, ga = _hvisc(ma, sa)
    scale = max(np.abs(fs_).max(), np.abs(gs_).max())
    assert scale > 0
    np.testing.assert_allclose(fa, fs_, rtol=0, atol=1e-12 * scale)
    np.testing.assert_allclose(ga, gs_, rtol=0, atol=1e-12 * scale)


# ------------------------------------------------------------------ counter, restart, registry
def test_counter_and_diagnostic():
    sb.set_threads(1)
    m, st, wind = nc._start(nc._ale("hybrid", nz=8, smagorinsky_bi=0.03, ah_bi=1e16, ah=0.0), True)
    counts = []
    for n in range(3):
        st = m.step(st, *wind(n))
        counts.append(st.n_visc_limited)
    assert counts[0] > 0 and counts[2] > counts[1] > counts[0]
    assert m._momentum.last_ah_t.shape == st.T.shape and np.all(m._momentum.last_ah_t >= 0)
    assert "ah_bi" in REGISTRY


def test_restart_bit_for_bit_with_biharmonic(tmp_path):
    sb.set_threads(1)

    def build(c=0.03):
        return nc._start(nc._ale("hybrid", nz=8, smagorinsky_bi=c, ah_vel_scale=0.01, ah=0.0), True)

    m, st, wind = build()
    for n in range(4):
        st = m.step(st, *wind(n))
    m1, s1, w1 = build()
    for n in range(2):
        s1 = m1.step(s1, *w1(n))
    path = tmp_path / "bih.nc"
    write_restart(path, m1, s1)
    m2, _, w2 = build()
    s2 = read_restart(path, m2)
    assert s2.n_visc_limited == s1.n_visc_limited
    for n in range(2, 4):
        s2 = m2.step(s2, *w2(n))
    assert nc.state_hashes(s2, m2) == nc.state_hashes(st, m)
    m3, _, _ = build(0.06)
    with pytest.raises(ValueError, match="smagorinsky_bi"):
        read_restart(path, m3)


# ------------------------------------------------------------------ pre-flight, assignment
@pytest.mark.parametrize("kw, match", [
    (dict(smagorinsky_bi=-0.1, ah=0.0), "smagorinsky_bi"),
    (dict(ah_vel_scale=float("nan"), ah=0.0), "ah_vel_scale"),
    (dict(ah_bi=-1.0, ah=0.0), "ah_bi"),
    (dict(smagorinsky_bi=0.03, smagorinsky=0.15, ah=0.0), "Laplacian"),
    (dict(smagorinsky_bi=0.03, ah=5.0), "Laplacian"),
])
def test_preflight(kw, match):
    with pytest.raises(ConfigurationError, match=match):
        nc._sigma(**kw)


def test_assignable_after_construction():
    m = nc._sigma(ah=0.0)
    m.smagorinsky_bi = 0.05
    m.ah_vel_scale = 0.02
    m.ah_bi = 3.0
    mo = m._momentum
    assert (mo.smagorinsky_bi, mo.ah_vel_scale, mo.ah_bi) == (0.05, 0.02, 3.0)


@pytest.mark.parametrize("built, name, value, match", [
    (dict(smagorinsky=0.15, ah=0.0), "smagorinsky_bi", 0.03, "Laplacian"),
    (dict(ah=5.0), "ah_vel_scale", 0.01, "Laplacian"),
    (dict(ah=0.0, smagorinsky_bi=0.03), "smagorinsky", 0.15, "Laplacian"),
    (dict(ah=0.0, smagorinsky_bi=0.03), "ah", 2.0, "Laplacian"),
    (dict(ah=0.0), "ah_bi", -1.0, "ah_bi"),
    (dict(ah=0.0), "smagorinsky_bi", float("nan"), "smagorinsky_bi"),
    (dict(ah=0.0), "smagorinsky", -0.1, "smagorinsky"),
])
def test_assignment_after_construction_is_checked(built, name, value, match):
    """Final review I1: a later assignment gets the pre-flight checks and leaves the model unchanged on refusal."""
    m = nc._sigma(**built)
    before = getattr(m, name)
    with pytest.raises(ConfigurationError, match=match):
        setattr(m, name, value)
    assert getattr(m, name) == before


# ------------------------------------------------------------------ open edges
@pytest.mark.parametrize("tangential", ["none", "mom6"])
def test_open_box_runs_and_open_faces_get_no_tendency(tangential):
    m, st = _ale_open_box("zstar", ny=6, open_=("west", "east"), smagorinsky_bi=0.03, ah_vel_scale=0.01, f0=1e-4,
                          momentum="vector_invariant", obc_tangential=tangential)
    m._driver.keep_terms = True
    seen = 0.0
    for _ in range(20):
        st = m.step(st)
        fu, fv = m._driver.last_terms["hviscosity"]
        assert np.all(fu[:, :, 0] == 0.0) and np.all(fu[:, :, -1] == 0.0)
        assert np.all(np.isfinite(fu)) and np.all(np.isfinite(fv))
        seen = max(seen, np.abs(fu).max())
    assert seen > 0
    assert all(np.all(np.isfinite(a)) for a in (st.u, st.v, st.eta, st.h, st.T))
