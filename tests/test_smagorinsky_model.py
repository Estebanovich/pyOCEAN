"""
Smagorinsky viscosity integrated in the 3-D model (σ and ALE), Task 2 of
docs/superpowers/plans (spec docs/superpowers/specs/2026-10-30-smagorinsky-viscosity-design.md).

The operator itself (``core.dynamics.hor_visc``) is tested in ``tests/test_hor_visc.py``; here the wiring:
off-by-default bit for bit, periodic halos, σ ≡ ALE for fixed layers, wetting–drying masks, restart,
pre-flight and open edges.  Index order (k, j, i), k = 0 at the bed.
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
from core.restart import read_restart, write_restart
from tests import n1_cases as nc
from tests.test_sigma_ale import _ale_open_box

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "data" / "n1_golden" / "hashes.json").read_text())
RHO0, TCOEF = 1000.0, 0.2


def _hvisc(m: SigmaBaroclinic, st) -> tuple[np.ndarray, np.ndarray]:
    """The 'hviscosity' term of the model's explicit momentum terms for state ``st``."""
    terms, _, _ = m._momentum_terms(st)
    return {name: (tu, tv) for name, tu, tv, _ in terms}["hviscosity"]


# ------------------------------------------------------------------ off = bit for bit
@pytest.mark.parametrize("name, build", [
    ("zstar_vi_arakawa", lambda: nc._start(nc._ale("zstar", smagorinsky=0.0), True)),
    ("sigma_gls", lambda: nc._start(nc._sigma(closure="gls", z0=1e-3, gls_stratified=False, smagorinsky=0.0),
                                    True)),
])
def test_zero_smagorinsky_is_bitwise(name, build):
    sb.set_threads(1)
    m, st, wind = build()
    got = nc._advance(m, st, wind)
    for step, ref in GOLDEN[name].items():
        assert got[int(step)] == ref, nc.diff_fields(got[int(step)], ref)


# ------------------------------------------------------------------ doubly periodic: momentum, energy
def test_doubly_periodic_conserves_momentum():
    # The viscous operator alone (no Coriolis, uniform ρ so no PGF, η = 0 so no surface pressure), stepped
    # forward 50 times through the model's own term path (periodic pad–crop with halo 2): the stress
    # divergence telescopes, so Σ h u A is kept to round-off, and the kinetic energy only decreases.
    nx, ny, nz, H = 16, 12, 3, 30.0
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((ny, nx), H), dt=60.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, H / nz)), pgf="fv", periodic=("x", "y"),
                        smagorinsky=0.15)
    st = m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 10.0))
    rng = np.random.default_rng(3)
    u = 0.1 * rng.standard_normal((nz, ny, nx + 1))
    v = 0.1 * rng.standard_normal((nz, ny + 1, nx))
    u[..., -1] = u[..., 0]                                  # the wrap face is one face stored twice
    v[:, -1] = v[:, 0]
    st.u, st.v = u, v
    hu, hv = m._mean_face_thickness(st.h)
    Au, Av = hg["area_u"].values, hg["area_v"].values

    def moments(st):
        mu = (hu * st.u * Au)[..., :-1].sum()
        mv = (hv * st.v * Av)[:, :-1].sum()
        scale_u = (hu * np.abs(st.u) * Au)[..., :-1].sum()
        scale_v = (hv * np.abs(st.v) * Av)[:, :-1].sum()
        ke = 0.5 * ((hu * st.u ** 2 * Au)[..., :-1].sum() + (hv * st.v ** 2 * Av)[:, :-1].sum())
        return mu, mv, scale_u, scale_v, ke

    mu0, mv0, su, sv, ke0 = moments(st)
    ke = ke0
    for _ in range(50):
        fu, fv = _hvisc(m, st)
        assert np.array_equal(fu[..., 0], fu[..., -1]) and np.array_equal(fv[:, 0], fv[:, -1])
        st.u, st.v = st.u + m.dt * fu, st.v + m.dt * fv
        mu, mv, _, _, ke_new = moments(st)
        assert ke_new <= ke
        ke = ke_new
    assert abs(mu - mu0) <= 1e-12 * su
    assert abs(mv - mv0) <= 1e-12 * sv
    assert ke < 0.99 * ke0                                   # it did dissipate
    assert np.all(m._momentum.last_nu_t > 0)


# ------------------------------------------------------------------ σ ≡ ALE for fixed layers
def test_sigma_and_ale_fixed_layers_agree():
    nx, ny, nz, H = 12, 9, 4, 40.0
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=800.0, dy=1000.0, f0=1e-4)
    common = dict(rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant", nu_const=1e-4, pgf="fv",
                  smagorinsky=0.15, ah=0.5)
    ms = SigmaBaroclinic(hg, np.full((ny, nx), H), dt=60.0, n_layers=nz, **common)
    ma = SigmaBaroclinic(hg, np.full((ny, nx), H), dt=60.0, n_layers=nz, vertical_coordinate="ale",
                         regrid="zstar", regrid_params=dict(dz_top=np.full(nz, H / nz)), **common)
    T = np.full((nz, ny, nx), 10.0)
    ss = ms.initial_state(np.zeros((ny, nx)), T)
    sa = ma.initial_state(np.zeros((ny, nx)), T)
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
    np.testing.assert_allclose(ma._momentum.last_nu_t, ms._momentum.last_nu_t, rtol=1e-12)


# ------------------------------------------------------------------ wetting and drying (Review Focus 2)
def test_drying_face_gets_no_viscous_tendency():
    nx, ny, nz = 10, 6, 3
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=500.0, dy=500.0, f0=0.0)
    Hb = np.tile(np.linspace(8.0, 0.4, nx), (ny, 1))                 # a beach with a flat 0.4 m shelf at the east
    Hb[:, -3:] = 0.4
    m = SigmaBaroclinic(hg, Hb, dt=10.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, smagorinsky=0.15, d_min=0.05)
    T = np.full((nz, ny, nx), 10.0)
    st_wet = m.initial_state(np.zeros((ny, nx)), T)
    rng = np.random.default_rng(11)
    u = 0.1 * rng.standard_normal((nz, ny, nx + 1))
    v = 0.1 * rng.standard_normal((nz, ny + 1, nx))
    u[..., 0] = u[..., -1] = 0.0
    v[:, 0] = v[:, -1] = 0.0
    st_wet.u, st_wet.v = u, v
    fu, fv = _hvisc(m, st_wet)
    _, wu0, _ = m._momentum_terms(st_wet)
    assert wu0[:, -2].all()                                         # step n−1: the face is wet …
    assert np.abs(fu[:, :, -2]).max() > 0                           # … and has a viscous tendency
    # step n: the sea level falls, the three shelf columns dry out (D = 0.4 − 0.37 < d_min)
    st_dry = m.initial_state(np.full((ny, nx), -0.37), T)
    st_dry.u, st_dry.v = u.copy(), v.copy()
    fu, fv = _hvisc(m, st_dry)
    _, wu, wv = m._momentum_terms(st_dry)
    assert not wu[:, -2].any()
    assert np.all(np.isfinite(fu)) and np.all(np.isfinite(fv))
    assert np.all(fu[:, ~wu] == 0.0) and np.all(fv[:, ~wv] == 0.0)
    # the wet face next to the dry one sees only free-slip corners there: same as with u = 0 on the dry faces
    u0 = np.where(wu[None], u, 0.0)
    v0 = np.where(wv[None], v, 0.0)
    st_dry.u, st_dry.v = u0, v0
    fu0, fv0 = _hvisc(m, st_dry)
    assert np.array_equal(fu, fu0) and np.array_equal(fv, fv0)
    # and the model steps through the drying with finite fields
    st = st_dry
    for _ in range(5):
        st = m.step(st)
    assert all(np.all(np.isfinite(a)) for a in (st.u, st.v, st.eta))


# ------------------------------------------------------------------ restart
def test_restart_bit_for_bit_with_smagorinsky(tmp_path):
    sb.set_threads(1)

    def build():
        return nc._start(nc._ale("hybrid", nz=8, smagorinsky=0.15), True)

    m, st, wind = build()
    for n in range(4):
        st = m.step(st, *wind(n))
    assert st.n_visc_limited is not None and st.n_visc_limited >= 0
    m1, s1, w1 = build()
    for n in range(2):
        s1 = m1.step(s1, *w1(n))
    path = tmp_path / "smag.nc"
    write_restart(path, m1, s1)
    m2, _, w2 = build()
    s2 = read_restart(path, m2)
    assert s2.n_visc_limited == s1.n_visc_limited
    for n in range(2, 4):
        s2 = m2.step(s2, *w2(n))
    assert nc.state_hashes(s2, m2) == nc.state_hashes(st, m)
    assert s2.n_visc_limited == st.n_visc_limited
    # the configuration check refuses another coefficient
    m3, _, _ = nc._start(nc._ale("hybrid", nz=8, smagorinsky=0.3), True)
    with pytest.raises(ValueError, match="smagorinsky"):
        read_restart(path, m3)


def test_counter_accumulates_limited_points():
    # hybrid layers with vanished thickness: hrat_min → 0 makes the bound 0 there, so ν is limited
    sb.set_threads(1)
    m, st, wind = nc._start(nc._ale("hybrid", nz=8, smagorinsky=0.15), True)
    counts = []
    for n in range(3):
        st = m.step(st, *wind(n))
        counts.append(st.n_visc_limited)
    assert counts[0] > 0
    assert counts[1] >= counts[0] and counts[2] >= counts[1]
    assert counts[1] - counts[0] == m._momentum.last_n_limited or counts[2] - counts[1] == m._momentum.last_n_limited


def test_counter_off_by_default():
    m, st, wind = nc._start(nc._ale("zstar"), True)
    st = m.step(st, *wind(0))
    assert st.n_visc_limited is None


# ------------------------------------------------------------------ pre-flight
def test_preflight_rejects_negative_smagorinsky():
    with pytest.raises(ConfigurationError, match="smagorinsky"):
        nc._sigma(smagorinsky=-0.1)


def test_smagorinsky_assignable_after_construction():
    m = nc._sigma()
    m.smagorinsky = 0.2
    assert m._momentum.smagorinsky == 0.2


# ------------------------------------------------------------------ open edges
def test_open_box_runs_and_edge_faces_zero():
    m, st = _ale_open_box("zstar", ny=4, open_=("west",), smagorinsky=0.15, f0=1e-4)
    m._driver.keep_terms = True
    seen = 0.0
    for _ in range(20):
        st = m.step(st)
        fu, fv = m._driver.last_terms["hviscosity"]
        assert np.all(fu[:, :, 0] == 0.0)                            # the open west faces
        assert np.all(np.isfinite(fu)) and np.all(np.isfinite(fv))
        seen = max(seen, np.abs(fu).max())
    assert seen > 0
    assert all(np.all(np.isfinite(a)) for a in (st.u, st.v, st.eta, st.h, st.T))
