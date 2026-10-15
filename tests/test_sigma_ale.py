"""
Tests for the ALE mode of core/dynamics/sigma_baroclinic.py (spec
docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md §2, §5, §7): configuration,
prognostic layer thickness, Lagrangian step + regrid + remap, finite-volume pressure gradient.
"""

from __future__ import annotations

import functools
from pathlib import Path

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, reference_profile
from core.grid.horizontal import create_cartesian_grid
from core.vertical import regrid as rg

RHO0, TCOEF = 1000.0, 0.28


def _ale_box(rule="zstar", nx=32, nz=10, depth=20.0, dx=2000.0, dt=30.0, bathy=None, params=None, nu=1e-6, **kw):
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    H = np.full((1, nx), depth) if bathy is None else bathy
    if params is None:
        params = dict(dz_top=np.full(nz, np.nanmax(H) / nz))
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                           nu_const=nu, d_min=0.05, vertical_coordinate="ale", regrid=rule,
                           regrid_params=params, **{"theta": 0.6, **kw})


# ---------------------------------------------------------------- configuration
@pytest.mark.parametrize("kw, match", [
    (dict(vertical_coordinate="zz"), "vertical_coordinate"),
    (dict(regrid="isopycnal"), "regrid"),
    (dict(remap_scheme="pqm"), "remap_scheme"),
    (dict(z0=1e-3), "z0"),
    (dict(closure="gls"), "closure"),
    (dict(params={}), "dz_top"),
    (dict(momentum="weno"), "momentum"),
    (dict(ke_scheme="weno"), "ke_scheme"),
    (dict(thickness_flux="weno"), "thickness_flux"),
])
def test_ale_configuration_errors(kw, match):
    base = dict(rule="zstar")
    if "regrid" in kw:
        base["rule"] = kw.pop("regrid")
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    args = dict(dt=10.0, n_layers=4, z0=None, closure="constant", vertical_coordinate="ale", regrid=base["rule"],
                regrid_params=dict(dz_top=np.full(4, 5.0)))
    if "params" in kw:
        args["regrid_params"] = kw.pop("params")
    args.update(kw)
    with pytest.raises(ValueError, match=match):
        SigmaBaroclinic(hg, np.full((1, 8), 20.0), **args)


def test_zstar_rejects_cells_that_can_dry():
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.linspace(-1.0, 10.0, 8)[None]
    with pytest.raises(ValueError, match="dry"):
        SigmaBaroclinic(hg, H, dt=10.0, n_layers=4, z0=None, closure="constant", vertical_coordinate="ale",
                        regrid="zstar", regrid_params=dict(dz_top=np.full(4, 2.5)))


def test_ale_initial_state_has_the_rule_grid_and_sigma_mode_has_no_h():
    m = _ale_box(bathy=np.array([[20.0, 13.0, 7.0, 20.0]]), nx=4, nz=4, params=dict(dz_top=np.full(4, 5.0)))
    st = m.initial_state(np.zeros((1, 4)), np.full((4, 1, 4), 10.0))
    np.testing.assert_allclose(st.h.sum(0), [[20.0, 13.0, 7.0, 20.0]], rtol=1e-14)
    np.testing.assert_allclose(st.h[:, 0, 0], 5.0, rtol=1e-14)
    # 13 m column on 5 m levels (surface first 5, 5, partial 3); the deepest level collapses at the bed
    np.testing.assert_allclose(st.h[::-1, 0, 1][:3], [5.0, 5.0, 3.0 - 1e-3], atol=1e-12)
    assert abs(st.h[0, 0, 1] - 1e-3) < 1e-12
    z = m.layer_heights(np.zeros((1, 4)), st.h)
    np.testing.assert_allclose(z[-1, 0, 0], -2.5)
    sig = SigmaBaroclinic(create_cartesian_grid(nx=4, ny=1, dx=1e3, dy=1e3, f0=0.0), np.full((1, 4), 20.0),
                          dt=10.0, n_layers=4)
    assert sig.initial_state(np.zeros((1, 4)), np.full((4, 1, 4), 10.0)).h is None and not sig.ale


def test_hybrid_initial_state_resolves_a_continuous_stratification():
    # The hybrid start used the z* floor of dz_top (19 × 0.5 m + one 190 m bed layer), so T sampled on it
    # lost the deep structure and the seiche blew up; MOM6 starts from uniform layers
    # (THICKNESS_CONFIG = "uniform", MOM_state_initialization.F90:271) and regrids (REGRID_ACCELERATE_INIT, :488)
    H, nz = 200.0, 20
    dTdz = 0.02**2 * RHO0 / (9.81 * TCOEF)

    def temp(z):                                                      # linear + mode-1 displacement of 5 m
        return 10.0 + dTdz * (z + 5.0 * np.sin(-np.pi * z / H))

    def heat(z0, z1):                                                 # ∫ T dz between heights z0 < z1
        def F(z):
            return 10.0 * z + dTdz * (0.5 * z**2 + 5.0 * H / np.pi * np.cos(-np.pi * z / H))
        return F(z1) - F(z0)

    rho = RHO0 - TCOEF * temp(np.linspace(-H, 0.0, 401))
    params = dict(dz_top=np.full(nz, 0.5), rho_target=rg.target_densities(rho, nz))
    m = _ale_box("hybrid", nx=4, nz=nz, depth=H, params=params)
    T0 = temp(m.layer_heights(np.zeros((1, 4))))
    st = m.initial_state(np.zeros((1, 4)), T0)
    assert st.h.max() < 2.0 * H / nz, st.h[:, 0, 0]                  # isopycnal layers ≈ 10 m, no thick bed layer
    np.testing.assert_allclose((st.h * st.T).sum(0), (H / nz * T0).sum(0), rtol=1e-13)
    # every layer keeps its mean T up to the midpoint-sampling error dTdz·5(π/H)²h³/24 = 0.051 dTdz [m²]
    zw = -H + np.concatenate([np.zeros((1, 1, 4)), np.cumsum(st.h, 0)])
    np.testing.assert_allclose(st.h * st.T, heat(zw[:-1], zw[1:]), atol=0.1 * dTdz)


def test_fv_pgf_is_exact_for_linear_stratification_over_a_seamount():
    from tests.test_sigma_baroclinic import _seamount_model
    m, z = _seamount_model("fv")
    T = 20.0 + 0.005 * z
    gx, gy = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13


@pytest.mark.xfail(strict=False, reason="pending Esteban's decision (ALE Plan B, Task 5): with the binned "
                   "reference profile FV 7.888e-8 vs jacobian 7.876e-8 m/s² (tie, +0.15 %); without a profile "
                   "FV 8.5e-5 vs 4.7e-5; with the exact profile FV 5e-17, jacobian 0")
def test_fv_pgf_error_over_the_seamount_is_not_larger_than_the_jacobian():
    # spec §7: "error del seamount ≤ al actual" — both with the reference profile of the state
    from tests.test_sigma_baroclinic import _seamount_model
    err = {}
    for scheme in ("jacobian", "fv"):
        m, z = _seamount_model(scheme)
        rho = RHO0 + 3.0 * (1 - np.exp(z / 1000.0))
        T = (RHO0 - rho) / 0.2
        m.rho_ref = reference_profile(z, rho)
        gx, gy = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
        err[scheme] = max(np.abs(gx).max(), np.abs(gy).max())
    assert err["fv"] <= err["jacobian"]


def test_fv_pgf_tabulates_any_callable_reference_profile():
    from tests.test_sigma_baroclinic import _seamount_model
    m, z = _seamount_model("fv")
    rho = RHO0 + 3.0 * (1 - np.exp(z / 1000.0))
    T = (RHO0 - rho) / 0.2
    prof = reference_profile(z, rho)
    m.rho_ref = prof
    g1 = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)[0]
    m.rho_ref = lambda zz: prof(zz)
    g2 = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)[0]
    np.testing.assert_allclose(g2, g1, atol=1e-10)


# ---------------------------------------------------------------- the ALE step
@pytest.mark.parametrize("rule", ["sigma", "zstar"])
def test_uniform_tracer_stays_uniform_and_is_conserved_while_the_surface_sloshes(rule):
    m = _ale_box(rule)
    x = (np.arange(32) + 0.5) * 2000.0
    eta = 0.5 * np.cos(np.pi * x / 64e3)[None, :]
    st = m.initial_state(eta, np.full((10, 1, 32), 7.0))
    c0 = m.tracer_content(st)
    for _ in range(200):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
        assert (st.h >= 0).all()
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-13
    assert np.abs(st.eta).max() > 0.1


@pytest.mark.parametrize("rule", ["sigma", "zstar"])
def test_resting_stratification_over_a_flat_bottom_stays_at_rest(rule):
    m = _ale_box(rule)
    z = m.layer_heights(np.zeros((1, 32)))
    st = m.initial_state(np.zeros((1, 32)), 5.0 + 0.5 * (z + 20.0))
    for _ in range(100):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-12 and np.abs(st.eta).max() < 1e-12


def test_online_mixing_includes_the_remap_and_matches_the_global_variance_loss():
    nx, nz = 40, 10
    m = _ale_box("zstar", nx=nx, nz=nz, dx=1000.0, dt=20.0, ah=1.0)
    T = np.where(np.arange(nx) < nx // 2, 10.0, 0.0)[None, None, :].repeat(nz, 0)
    st = m.initial_state(np.zeros((1, nx)), T)
    for _ in range(50):
        var0 = m.tracer_content(st, power=2)
        st = m.step(st)
        lost = (var0 - m.tracer_content(st, power=2)) / m.dt
        assert np.isclose(st.mix_num.sum(), lost, rtol=1e-9, atol=1e-12 * abs(var0))
    assert st.mix_num.sum() > 0


def test_sigma_ale_on_a_drying_beach_stays_positive_and_uniform():
    # review focus 1: cells that dry and flood under ALE σ
    nx = 30
    H = np.linspace(4.0, -0.5, nx)[None, :]
    m = _ale_box("sigma", nx=nx, nz=4, dx=200.0, dt=5.0, bathy=H, params={})
    x = np.arange(nx)
    eta = np.maximum(0.4 * np.cos(np.pi * x / nx)[None, :], -H + 0.0)
    st = m.initial_state(eta, np.full((4, 1, nx), 3.0))
    c0 = m.tracer_content(st)
    for _ in range(300):
        st = m.step(st)
        assert np.isfinite(st.h).all() and (st.h >= 0).all()
        np.testing.assert_allclose(st.h.sum(0), np.maximum(H + st.eta, 0.0), atol=1e-9)
    wet = (H + st.eta)[0] > 0.05
    np.testing.assert_allclose(st.T[:, 0, wet], 3.0, atol=1e-10)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-10      # drying cells: h† round-off clipped at 0


# ---------------------------------------------------------------- regression ALE-σ vs σ (spec §7, integration 1)
def test_ale_sigma_lock_exchange_matches_benjamin_and_conserves_heat():
    nx, nz, depth, dt = 128, 20, 20.0, 20.0
    m = _ale_box("sigma", nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params={}, ke_scheme="gudonov")
    T = np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :].repeat(nz, 0) / TCOEF
    st = m.initial_state(np.zeros((1, nx)), T)
    c0, r0 = m.tracer_content(st), m.rpe(st)
    x = (np.arange(nx) + 0.5) * 500.0
    half = 7.5 / TCOEF
    fronts = {}
    for n in range(int(6 * 3600 / dt)):
        st = m.step(st)
        t = (n + 1) * dt
        if t in (2 * 3600.0, 6 * 3600.0):
            fronts[t] = (x[st.T[0, 0] < half].max() - 32e3, 32e3 - x[st.T[-1, 0] > half].min())
    U = 0.5 * np.sqrt(9.81 * 5.0 / RHO0 * depth)
    for k in range(2):
        speed = (fronts[6 * 3600.0][k] - fronts[2 * 3600.0][k]) / (4 * 3600)
        assert abs(speed / U - 1) < 0.10
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-12
    assert m.rpe(st) >= r0


def test_ale_sigma_internal_wave_matches_the_eulerian_sigma_model():
    # mode-1 internal wave in a closed 60 km channel (N = 0.02 s⁻¹, 200 m): after 2 h the ALE-σ
    # temperature differs from the Eulerian σ model by much less than the wave signal
    from tests.test_sigma_baroclinic import _channel
    m_s, st_s = _channel(60, 30e3)
    m_a, st_a = _channel(60, 30e3, vertical_coordinate="ale", regrid="sigma")
    T0 = st_s.T.copy()
    for _ in range(60):
        st_s = m_s.step(st_s)
        st_a = m_a.step(st_a)
    signal = np.abs(st_s.T - T0).max()
    assert signal > 0.05
    assert np.abs(st_a.T - st_s.T).max() < 0.1 * signal


# ---------------------------------------------------------------- momentum in vanished layers
@pytest.mark.parametrize("compiled", [True, False])
def test_flux_form_momentum_in_a_vanished_cell_relaxes_to_the_inflow_without_overshoot(compiled):
    # a current flooding a 1 cm layer: with V_u in the denominator the explicit tendency would
    # move u by Q Δt/V_u ≫ 1 times (u_in − u); the limiter caps the rate at 1/(2 Δt) (AB3 stable)
    from core.dynamics.sigma_baroclinic import _flux_form_u, _flux_form_u_numpy
    f = _flux_form_u if compiled else _flux_form_u_numpy
    n, ny, nx = 1, 1, 6
    u = np.zeros((n, ny, nx + 1))
    u[..., :4] = 0.3                                                     # the current, west of face 4
    Qx = np.zeros_like(u)
    Qx[..., :4] = 0.3 * 8.0 * 100.0                                     # thick layer: Q = u h L
    Qy, Fz = np.zeros((n, ny + 1, nx)), np.zeros((n + 1, ny, nx))
    V = np.full((n, ny, nx), 8.0 * 100.0 * 500.0)
    V[..., 3:] = 0.01 * 100.0 * 500.0                                   # vanished east of cell 2
    ok = np.ones_like(u, bool)
    dt = 20.0
    g0 = f(u, Qx, Qy, Fz, V, ok)                                         # no limiter (σ path)
    g = f(u, Qx, Qy, Fz, V, ok, dt=dt)
    assert abs(dt * g0[0, 0, 4]) > 1.0                                   # unlimited: overshoots
    assert 0.0 < dt * g[0, 0, 4] <= 0.5 * 0.3 + 1e-12                    # limited: at most half way
    np.testing.assert_array_equal(g[..., :3], g0[..., :3])               # thick cells unchanged


@pytest.mark.parametrize("compiled", [True, False])
def test_flux_form_momentum_in_a_vanished_cell_with_outflow_only_stays_bounded(compiled):
    # ALE Plan B2c (hybrid lock exchange, 3.4 m s⁻² in a 1 mm layer): a vanished u-cell that only
    # exports (east transport large, west zero) — the WENO face value differs from u, so the
    # outflow term Q_out (u_face − u)/V_u also needs the limiter, not only the inflow
    from core.dynamics.sigma_baroclinic import _flux_form_u, _flux_form_u_numpy
    f = _flux_form_u if compiled else _flux_form_u_numpy
    n, ny, nx = 1, 1, 8
    u = np.array([0.0, 1.0, 1.0, 1.0, 0.0, 3.0, 3.0, 3.0, 0.0])[None, None, :]
    Qx = np.zeros_like(u)
    Qx[0, 0, 3], Qx[0, 0, 4], Qx[0, 0, 5] = -400.0, 400.0, 400.0        # u-cell of face 4: out east only
    Qy, Fz = np.zeros((n, ny + 1, nx)), np.zeros((n + 1, ny, nx))
    V = np.full((n, ny, nx), 0.001 * 100.0 * 500.0)                      # vanished layer everywhere
    ok = np.ones_like(u, bool)
    dt = 20.0
    g0 = f(u, Qx, Qy, Fz, V, ok)
    g = f(u, Qx, Qy, Fz, V, ok, dt=dt)
    spread = np.abs(u[0, 0, 2:7] - u[0, 0, 4]).max()                    # WENO stencil around face 4
    assert abs(dt * g0[0, 0, 4]) > spread                                # unlimited: overshoots
    assert abs(dt * g[0, 0, 4]) <= 0.5 * spread + 1e-12


@pytest.mark.parametrize("compiled", [True, False])
def test_flux_form_momentum_limiter_leaves_thick_cells_below_courant_one_half(compiled):
    # final review of Plan B2: counting Σ|Q| (in + out) engaged the limiter at u Δt/Δx > ¼ in
    # uniform flow; with max(inflow, outflow) it stays off below ½, as in the σ path
    from core.dynamics.sigma_baroclinic import _flux_form_u, _flux_form_u_numpy
    f = _flux_form_u if compiled else _flux_form_u_numpy
    n, ny, nx = 1, 1, 8
    dt, dx, h, L = 20.0, 500.0, 10.0, 500.0
    u = (0.3 * dx / dt) * (1.0 + 0.1 * np.sin(np.arange(nx + 1)))[None, None, :]   # Courant ≈ 0.3
    Qx = u * h * L
    Qy, Fz = np.zeros((n, ny + 1, nx)), np.zeros((n + 1, ny, nx))
    V = np.full((n, ny, nx), h * dx * L)
    ok = np.ones_like(u, bool)
    np.testing.assert_array_equal(f(u, Qx, Qy, Fz, V, ok, dt=dt), f(u, Qx, Qy, Fz, V, ok))


# ---------------------------------------------------------------- hybrid rule in the model
# target margin [kg m⁻³] that puts both water masses at the centre of a density bracket: with 21 layers a
# margin of 1 puts them exactly on targets and each may split between two layers
CENTRED_MARGIN = {10: 1.0, 21: 0.78125}


def _hybrid_lock(nx=128, nz=21, depth=20.0, dt=20.0, margin=None, **kw):
    """Lock exchange (Δρ = 5 kg m⁻³) on isopycnal layers: each water mass sits in one layer, which
    has vanished on the other side of the lock.  Targets span [ρ_l − margin, ρ_d + margin].  21 layers
    as MOM6 lock_exchange: the PPM thickness flux needs them (10 layers: dense body 1.26 U)."""
    margin = CENTRED_MARGIN[nz] if margin is None else margin
    m0 = _ale_box("sigma", nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params={})
    rho_l, rho_d = m0.density(10.0 / TCOEF), m0.density(5.0 / TCOEF)
    params = dict(dz_top=np.full(nz, 0.01), rho_target=np.linspace(rho_l - margin, rho_d + margin, nz + 1))
    m = _ale_box("hybrid", nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params=params, nu=1e-4, ah=0.01,  # MOM6 KV, KH
                 **{"ke_scheme": "gudonov", **kw})                                 # density current: Godunov
    T = np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :].repeat(nz, 0) / TCOEF
    return m, m.initial_state(np.zeros((1, nx)), T)


@functools.lru_cache(maxsize=None)
def _lock_run(rule: str, nx: int = 128, nz: int = 21, depth: float = 20.0, dt: float = 20.0):
    """6 h lock exchange: front speeds / Benjamin for 1 m and 5 m thick fronts, RPE gain, checks."""
    if rule == "hybrid":
        m, st = _hybrid_lock(nx, nz, depth, dt)
    else:
        m = _ale_box(rule, nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params={}, ke_scheme="gudonov")
        st = m.initial_state(np.zeros((1, nx)), np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :]
                             .repeat(nz, 0) / TCOEF)
    thick0 = bool((st.h.max(axis=0) > 0.9 * depth).all())
    c0, r0 = m.tracer_content(st), m.rpe(st)
    x = (np.arange(nx) + 0.5) * 500.0
    half = 7.5 / TCOEF
    fronts, consistent, umax = {}, True, 0.0
    for n in range(int(6 * 3600 / dt)):
        st = m.step(st)
        umax = max(umax, float(np.abs(st.u).max()))
        consistent &= bool(np.allclose(st.h.sum(0), depth + st.eta, rtol=1e-12))
        t = (n + 1) * dt
        if t in (2 * 3600.0, 6 * 3600.0):
            dense = np.where(st.T < half, st.h, 0.0).sum(0)[0]
            light = np.where(st.T > half, st.h, 0.0).sum(0)[0]
            fronts[t] = {d: (x[dense > d].max() - 32e3, 32e3 - x[light > d].min()) for d in (1.0, 5.0)}
    U = 0.5 * np.sqrt(9.81 * 5.0 / RHO0 * depth)
    speed = {d: tuple((fronts[6 * 3600.0][d][k] - fronts[2 * 3600.0][d][k]) / (4 * 3600) / U for k in range(2))
             for d in (1.0, 5.0)}
    return dict(speed=speed, rpe_gain=m.rpe(st) - r0, heat=m.tracer_content(st) / c0 - 1,
                one_thick_layer=thick0, consistent=consistent, umax=umax, n_trunc=st.n_trunc)


def test_hybrid_lock_exchange_fronts_leave_the_lock_at_the_benjamin_speed():
    # spec 2026-10-12 §6: body (5 m of the 10 m current) at Benjamin ±10 % (MOM6: 0.98–1.05 U)
    r = _lock_run("hybrid")
    assert r["one_thick_layer"] and r["consistent"]
    for k in range(2):
        assert abs(r["speed"][5.0][k] - 1) < 0.10, r["speed"]
    assert abs(r["heat"]) < 1e-12


def test_hybrid_lock_exchange_mixes_much_less_than_sigma():
    r, s = _lock_run("hybrid"), _lock_run("sigma")
    assert r["rpe_gain"] < 0.2 * s["rpe_gain"], (r["rpe_gain"], s["rpe_gain"])
    assert r["one_thick_layer"] and r["consistent"]
    assert abs(r["heat"]) < 1e-12


def test_hybrid_lock_exchange_nose_moves_at_the_benjamin_speed_without_runaway_velocities():
    # spec §6: 1 m fronts within ±10 %, max|u| ≤ 1.5 × MOM6 (0.56 m/s), no truncation.  Needs the PPM
    # thickness flux and 21 layers (MOM6's configuration): 1.05 / 0.98 U, 0.74 m/s; upwind or 10 layers fail
    r = _lock_run("hybrid")
    for k in range(2):
        assert abs(r["speed"][1.0][k] - 1) < 0.10, r["speed"]
    assert r["umax"] <= 0.84, r["umax"]
    assert r["n_trunc"] == 0


def test_regrid_runs_on_the_model_threads(monkeypatch):
    # ≤ 5 threads (performance cores): the parallel regrid kernel must not inherit Numba's default
    import numba
    from core.dynamics import sigma_baroclinic as sb
    seen = []
    orig = rg.regrid_columns

    def spy(*a, **k):
        seen.append(numba.get_num_threads())
        return orig(*a, **k)

    monkeypatch.setattr(rg, "regrid_columns", spy)
    old = sb._THREADS[0]
    try:
        sb.set_threads(2)
        numba.set_num_threads(3)
        m = _ale_box("zstar", nx=8)
        st = m.initial_state(np.zeros((1, 8)), np.full((10, 1, 8), 5.0))
        numba.set_num_threads(3)
        m.step(st)
    finally:
        sb.set_threads(old)
    assert seen and set(seen) == {2}


# ---------------------------------------------------------------- bay-mouth step (spec §7, integration 2)
def _step_section(coord, nz=30, **extra):
    """2-D section across a 600 → 164 m step (BayIW bay mouth), at rest with a tanh pycnocline at 80 m."""
    nx, dx = 40, 1000.0
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    H = np.where(np.arange(nx) < nx // 2, 600.0, 164.0)[None, :]
    kw = dict(pgf="jacobian")
    if coord == "zstar":
        kw = dict(pgf="fv", vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, 600.0 / nz)))
        kw.update(extra)
    m = SigmaBaroclinic(hg, H, dt=60.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, rho_ref="auto", **kw)
    z = m.layer_heights(np.zeros((1, nx)))
    return m, m.initial_state(np.zeros((1, nx)), 15.0 + 5.0 * np.tanh((z + 80.0) / 30.0))


def test_bay_mouth_step_spurious_flow_sigma_vs_zstar():
    # σ-layers over rx0 = 0.57 drive a spurious current (the BayIW problem); z* with the
    # finite-volume PGF keeps the resting state
    umax = {}
    for coord in ("sigma", "zstar"):
        m, st = _step_section(coord)
        for _ in range(720):                                            # 12 h
            st = m.step(st)
        assert np.isfinite(st.u).all()
        if coord == "zstar":
            assert (st.h >= 0).all()
        umax[coord] = np.abs(st.u).max()
    assert umax["zstar"] < 0.05 * umax["sigma"]
    assert umax["zstar"] < 1e-3


# ---------------------------------------------------------------- vector-invariant momentum
def test_momentum_defaults_vector_invariant_in_ale_and_flux_in_sigma():
    assert _ale_box("zstar", nx=8).momentum == "vector_invariant"
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=10.0, n_layers=4)
    assert m.momentum == "flux"
    with pytest.raises(ValueError, match="vector_invariant"):
        SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=10.0, n_layers=4, momentum="vector_invariant")


def test_ale_first_step_with_initial_velocity_uses_the_fallback_transports():
    # review focus 4: no transports yet (st.Qx is None) but u ≠ 0 — the PV flux and KE use L u h̄
    m = _ale_box("zstar", nx=16)
    st = m.initial_state(np.zeros((1, 16)), np.full((10, 1, 16), 7.0))
    st.u[:, 0, 1:-1] = 0.05
    gu, gv = m._tendencies(st)
    assert np.isfinite(gu).all() and np.isfinite(gv).all()
    st2 = m.step(st)
    assert st2.Qx is not None and np.isfinite(st2.u).all()


def test_ale_first_step_pv_flux_uses_the_fallback_transports_l_u_h():
    # final review: with ny = 1 and f = 0 the PV flux is identically 0, so test the fallback on a
    # 2-D f-plane with non-uniform u (uniform u would be pinned to −f u by BOUND_CORIOLIS)
    from core.dynamics import vector_invariant as vi
    nx, ny, nz = 8, 6, 3
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=2000.0, dy=2000.0, f0=1e-4)
    m = SigmaBaroclinic(hg, np.full((ny, nx), 20.0), dt=30.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 20.0 / nz)))
    st = m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 7.0))
    rng = np.random.default_rng(2)
    st.u[..., 1:-1] = rng.normal(0.0, 0.1, st.u[..., 1:-1].shape)
    assert st.Qx is None
    cu, cv = m._vector_invariant(st)
    g = m._vi
    hu, hv = m._mean_face_thickness(st.h)
    zeta = vi.vorticity_q(st.u, st.v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"])
    q = vi.pv_q(zeta, m._fq, st.h, g["area_t"], m.fs.active)
    ref_u, ref_v = vi.coriolis_pv(q, m._fq[None] + zeta, m.fs.Lu[None] * st.u * hu, m.fs.Lv[None] * st.v * hv,
                                  st.u, st.v, g["dx_u"], g["dy_v"], bound=True)
    K = vi.kinetic_energy(st.u, st.v, g["area_u"], g["area_v"], g["area_t"], m.ke_scheme)
    kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
    np.testing.assert_allclose(cv, ref_v + ky, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(cu, ref_u + kx, rtol=1e-12, atol=1e-20)
    assert np.abs(ref_v).max() > 0.1 * np.abs(cv).max()                 # the PV flux is a real part of it


def test_ale_truncates_a_runaway_velocity_and_stops_beyond_max_truncations():
    # Arakawa KE is symmetric around the face, so the runaway velocity survives the step to be truncated
    m = _ale_box("zstar", nx=16, dt=30.0, ke_scheme="arakawa")
    st = m.initial_state(np.zeros((1, 16)), np.full((10, 1, 16), 7.0))
    st.u[5, 0, 8] = 0.6 * 2000.0 / 30.0                                  # Courant 0.6 in a 2 m layer (K symmetric
    st2 = m.step(st)                                                      # around this face: it survives the step)
    assert np.abs(st2.u).max() <= 0.5 * 2000.0 / 30.0 + 1e-12
    assert st2.n_trunc >= 1
    m0 = _ale_box("zstar", nx=16, dt=30.0, ke_scheme="arakawa", max_truncations=0)
    st = m0.initial_state(np.zeros((1, 16)), np.full((10, 1, 16), 7.0))
    st.u[5, 0, 8] = 0.6 * 2000.0 / 30.0
    with pytest.raises(RuntimeError, match="truncation"):
        m0.step(st)


def test_vector_invariant_coriolis_in_a_rotating_ale_basin_keeps_rest_and_stays_finite():
    # f-plane 2-D z* basin with a coast (review focus 1): rest stays rest; a sloshing start stays finite
    nx, ny, nz = 10, 8, 4
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=2000.0, dy=2000.0, f0=1e-4)
    H = np.full((ny, nx), 20.0)
    H[3, 4] = np.nan                                                    # an island
    m = SigmaBaroclinic(hg, H, dt=30.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 5.0)))
    st = m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 7.0))
    for _ in range(20):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-12 and np.abs(st.v).max() < 1e-12
    x = (np.arange(nx) + 0.5) * 2000.0
    st = m.initial_state(0.2 * np.cos(np.pi * x / (nx * 2000.0))[None, :].repeat(ny, 0), np.full((nz, ny, nx), 7.0))
    for _ in range(100):
        st = m.step(st)
    assert np.isfinite(st.u).all() and np.isfinite(st.v).all() and np.abs(st.v).max() > 0


@pytest.mark.parametrize("rule", ["sigma", "zstar"])
def test_ppm_thickness_flux_keeps_uniform_tracer_and_column_consistency(rule):
    m = _ale_box(rule, thickness_flux="ppm")
    x = (np.arange(32) + 0.5) * 2000.0
    st = m.initial_state(0.5 * np.cos(np.pi * x / 64e3)[None, :], np.full((10, 1, 32), 7.0))
    c0 = m.tracer_content(st)
    for _ in range(200):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
        assert (st.h >= 0).all()
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-13


def test_ppm_thickness_flux_keeps_the_zstar_step_at_rest():
    m, st = _step_section("zstar", thickness_flux="ppm")
    for _ in range(720):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-10


def test_ale_uses_the_ppm_thickness_flux_by_default():
    # MOM6 continuity_PPM (decision of 2026-10-12: with 21 layers the hybrid lock exchange matches MOM6)
    assert _ale_box("zstar").thickness_flux == "ppm"
    assert _ale_box("zstar", thickness_flux="upwind").thickness_flux == "upwind"


# ---------------------------------------------------------------- open boundaries (Plan C, scope A)
def test_radiation_with_face_weights_keeps_the_depth_mean_and_a_zero_weighted_baroclinic_part():
    # spec 2026-10-12-ale-open-boundaries §3.2: Orlanski on u′, re-centred with the face weights
    from core.vertical import ale
    hg = create_cartesian_grid(nx=8, ny=2, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((2, 8), 20.0), dt=30.0, n_layers=4, z0=None, closure="constant",
                        open_boundaries=("west", "east"))
    st = m.initial_state(np.zeros((2, 8)), np.full((4, 2, 8), 7.0))
    m.step(st)                                                           # sets the Flather face depths
    rng = np.random.default_rng(21)
    w = ale.face_fractions(rng.uniform(0.0, 4.0, (4, 2, 9)))
    u_old, u_new = rng.normal(0.0, 0.2, (4, 2, 9)), rng.normal(0.0, 0.2, (4, 2, 9))
    mean0 = (w * u_new).sum(0)
    m._radiate(u_old, u_new, 2, 0, w)
    np.testing.assert_allclose((w * u_new).sum(0)[:, 0], mean0[:, 0], atol=1e-15)
    _, prime = ale.baroclinic_split(u_new, w)
    np.testing.assert_allclose((w * prime).sum(0)[:, 0], 0.0, atol=1e-15)


def _ale_open_box(rule, thickness_flux="ppm", nx=30, ny=1, open_=("west", "east"), f0=0.0, **kw):
    """20 m box with open edges, a Gaussian η bump at t = 0 (it leaves through Flather), uniform T = 7."""
    nz = 6
    params = dict(dz_top=np.full(nz, 20.0 / nz))
    if rule == "hybrid":
        # asymmetric margins: the water density must not fall exactly on a target (it would split layers)
        params = dict(dz_top=np.full(nz, 0.5), rho_target=np.linspace(RHO0 - TCOEF * 7.0 - 1.0,
                                                                    RHO0 - TCOEF * 7.0 + 1.5, nz + 1))
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=f0)
    m = SigmaBaroclinic(hg, np.full((ny, nx), 20.0), dt=30.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid=rule,
                        regrid_params=params, pgf="fv", open_boundaries=open_, thickness_flux=thickness_flux, **kw)
    x = (np.arange(nx) + 0.5) * 1000.0
    y = (np.arange(ny) + 0.5) * 1000.0
    r2 = (x[None, :] - x.mean()) ** 2 + ((y[:, None] - y.mean()) ** 2 if ny > 1 else 0.0)
    return m, m.initial_state(0.3 * np.exp(-r2 / 3e3**2), np.full((nz, ny, nx), 7.0))


@pytest.mark.parametrize("thickness_flux", ["ppm", "upwind"])                # review focus 5
@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_open_boundaries_keep_a_uniform_tracer_uniform(rule, thickness_flux):
    # spec §5 integration 1: the open-face layer fluxes add up to the Flather flux and inflow is the
    # exterior value, or a uniform tracer would change; review focus 2 (η ≠ 0 at t = 0)
    m, st = _ale_open_box(rule, thickness_flux)
    vol0 = m.layer_volumes(st.eta, st.h).sum()
    for _ in range(120):                                                 # 1 h
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
        assert (st.h >= 0).all()
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert m.layer_volumes(st.eta, st.h).sum() < 0.9 * vol0 + 0.1 * (20.0 * 30e3 * 1e3)   # the bump left
    assert np.abs(st.u[:, 0, [0, -1]]).max() > 0                         # boundary faces carry flow


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_open_boundaries_on_four_edges_keep_a_uniform_tracer_uniform(rule):
    # review focus 3: v-faces (south, north) and a sponge on all four edges in 2-D
    m, st = _ale_open_box(rule, nx=14, ny=12, open_=("west", "east", "south", "north"), sponge_cells=3)
    for _ in range(60):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert np.abs(st.v[:, [0, -1], :]).max() > 0


def test_ale_open_boundaries_with_a_sponge_keep_a_resting_stratification_at_rest():
    # spec §5 integration 2 (z*)
    nz, nx = 10, 40
    hg = create_cartesian_grid(nx=nx, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, nx), 200.0), dt=120.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-5, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 20.0)), pgf="fv", open_boundaries=("west", "east"),
                        sponge_cells=8, sponge_tau=(1200.0, 120.0))
    z = m.layer_heights(np.zeros((1, nx)))
    st = m.initial_state(np.zeros((1, nx)), 10.0 + 0.02 * z)
    for _ in range(200):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-10


def test_ale_open_boundary_next_to_a_drying_cell_is_rejected():
    # spec §4 item 2: drying columns at ALE open boundaries are out of scope
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.linspace(-1.0, 10.0, 8)[None]
    with pytest.raises(ValueError, match="H ≤ 0"):
        SigmaBaroclinic(hg, H, dt=10.0, n_layers=4, z0=None, closure="constant", vertical_coordinate="ale",
                        regrid="sigma", regrid_params={}, open_boundaries=("west",))


N_CH, H_CH = 0.02, 200.0


def _ale_channel(rule, nx, x0, dx=1000.0, nz=10, dt=120.0, amp=5.0, **kw):
    """Linear stratification (N = 0.02 s⁻¹) with a mode-1 isopycnal bump at x0, ALE z* or hybrid."""
    dTdz = N_CH**2 * RHO0 / (9.81 * TCOEF)
    params = dict(dz_top=np.full(nz, H_CH / nz))
    if rule == "hybrid":
        rho = RHO0 - TCOEF * (10.0 + dTdz * np.linspace(-H_CH, 0.0, 401))
        params = dict(dz_top=np.full(nz, 1.0), rho_target=rg.target_densities(rho, nz))
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, nx), H_CH), dt=dt, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-5, vertical_coordinate="ale", regrid=rule,
                        regrid_params=params, pgf="fv", **kw)
    x = (np.arange(nx) + 0.5) * dx
    z = m.layer_heights(np.zeros((1, nx)))
    zeta = amp * np.sin(-np.pi * z / H_CH) * np.exp(-((x - x0) / 5e3) ** 2)[None, None, :]
    return m, m.initial_state(np.zeros((1, nx)), 10.0 + dTdz * (z + zeta))


def _on_levels(m, st, z_levels):
    """T interpolated in z from the layer centres to fixed levels (1, nz_levels, nx): layers move in ALE."""
    zc = m.layer_heights(st.eta, st.h)[:, 0, :]
    T = st.T[:, 0, :]
    return np.stack([np.interp(z_levels, zc[:, i], T[:, i]) for i in range(T.shape[1])], axis=-1)


@functools.lru_cache(maxsize=None)
def _ale_interior_error(rule, open_kw=(), nx_small=60, pad=0):
    """rms interior T difference (on fixed z levels) from a 300 km channel, relative to the initial anomaly,
    after 16 h; also the T range of the small channel over the run (min, max) and its initial range."""
    open_kw = dict(open_kw)
    zl = np.linspace(-195.0, -5.0, 39)
    big, sb = _ale_channel(rule, 300, 150e3)
    small, ss = _ale_channel(rule, nx_small + 2 * pad, (nx_small / 2 + pad) * 1e3, **open_kw)
    T_init = _on_levels(small, ss, zl)[:, pad:pad + nx_small]
    rng0 = (float(ss.T.min()), float(ss.T.max()))
    lo, hi = rng0
    for _ in range(int(round(16 * 3600 / big.dt))):
        sb, ss = big.step(sb), small.step(ss)
        lo, hi = min(lo, float(ss.T.min())), max(hi, float(ss.T.max()))
    sl = slice(150 - nx_small // 2, 150 + nx_small // 2)
    diff = _on_levels(small, ss, zl)[:, pad:pad + nx_small] - _on_levels(big, sb, zl)[:, sl]
    bg = T_init.mean(axis=-1, keepdims=True)
    err = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean((T_init - bg) ** 2))
    return err, (lo, hi), rng0


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_closed_channel_reflects_the_internal_wave(rule):
    assert _ale_interior_error(rule)[0] > 0.3                             # σ: 0.82


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_orlanski_radiation_lets_most_of_the_internal_wave_out(rule):
    # measured (z-level metric): z* 0.162, hybrid 0.198 (σ 0.149). The hybrid margin is small: inflow T comes
    # from the exterior remapped in z (spec §4.3, as MOM6 z-space segment data), so water of another density
    # class enters displaced isopycnal layers and the regrid re-sorts it (inflow by layer index: 0.180)
    err = _ale_interior_error(rule, (("open_boundaries", ("west", "east")),))[0]
    assert err < 0.2, err                                                # σ: 0.157


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_sponge_with_radiation_absorbs_the_internal_wave(rule):
    kw = (("open_boundaries", ("west", "east")), ("sponge_cells", 10), ("sponge_tau", (1200.0, 120.0)))
    err, (lo, hi), (lo0, hi0) = _ale_interior_error(rule, kw, pad=10)
    assert err < 0.05, err                                               # σ: 0.015
    span = hi0 - lo0                                                     # review focus 4: no new extrema
    assert lo >= lo0 - 0.01 * span and hi <= hi0 + 0.01 * span, (lo, hi, lo0, hi0)


def test_ale_open_face_transport_uses_the_old_boundary_velocity(monkeypatch):
    # the old velocity of an open face must not be masked by the wet flag of the free surface (False at
    # the domain edge): masked, the θ-average dropped (1 − θ) uⁿ and the open edge reflected the internal
    # wave (channel error 0.93 vs σ 0.15)
    from core.dynamics import sigma_baroclinic as sb
    m, st = _ale_open_box("zstar")
    st.u[:, 0, 0] = np.linspace(-0.02, 0.02, st.u.shape[0])
    seen = []
    orig = sb.open_face_transport

    def spy(h, un, uo, *a):
        seen.append(uo.copy())
        return orig(h, un, uo, *a)

    monkeypatch.setattr(sb, "open_face_transport", spy)
    m.step(st)
    np.testing.assert_array_equal(seen[0], st.u[:, :, 0])


@pytest.mark.parametrize("bad", ["none", "shape"])
def test_ale_open_boundaries_need_a_consistent_exterior_profile(bad):
    # T_ext and h_ext are set together by initial_state; a state built otherwise (or T_ext replaced alone,
    # as one may do in σ) must give a clear error, not a TypeError or a silent mismatched remap
    m, st = _ale_open_box("zstar")
    if bad == "none":
        m.h_ext = None
    else:
        m.T_ext = m.T_ext[:-1]
    with pytest.raises(ValueError, match="h_ext"):
        m.step(st)


# ---------------------------------------------------------------- open boundaries with rotation (BayIW phase 0)
@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_rotating_open_boundaries_keep_a_uniform_tracer_uniform(rule):
    m, st = _ale_open_box(rule, nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4, sponge_cells=3)
    vol0 = m.layer_volumes(st.eta, st.h).sum()
    for _ in range(120):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert np.isfinite(st.u).all() and np.isfinite(st.v).all()
    assert np.abs(st.v).max() > 0                                        # rotation turned the outflow
    assert m.layer_volumes(st.eta, st.h).sum() < vol0


def test_rotating_open_boundaries_keep_a_resting_stratification_at_rest():
    nz, nx, ny = 10, 16, 12
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=1e-4)
    m = SigmaBaroclinic(hg, np.full((ny, nx), 200.0), dt=120.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-5, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 20.0)), pgf="fv",
                        open_boundaries=("west", "east", "south"), sponge_cells=4, sponge_tau=(1200.0, 120.0))
    z = m.layer_heights(np.zeros((ny, nx)))
    st = m.initial_state(np.zeros((ny, nx)), 10.0 + 0.02 * z)
    for _ in range(200):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-10 and np.abs(st.v).max() < 1e-10


# ---------------------------------------------------------------- performance regression (spec 2026-10-12-ale-step-performance §4)
def _perf_cases():
    # fingerprints of tests/data/ale_perf_reference.json were generated with the Godunov KE (then the default) and
    # with the baroclinic transport weighted by the free-surface θ = 0.6 (theta_bc = θ, the scheme before 2026-10-13):
    # pinning theta_bc = 0.6 keeps them a bit-level guard of the kernels and shows the IW fix is isolated in theta_bc
    # …and before the robustness defaults (spec 2026-10-14): thin-layer coupling, blending and FCT off
    old = dict(theta_bc=0.6, thin_layer_coupling="none", thin_layer_cutoff=0.0, tracer_limiter="none")
    return {
        "lock": lambda: _hybrid_lock(128, 21, 20.0, 20.0, solver="direct", **old),
        "channel": lambda: _ale_channel("hybrid", 60, 30e3, open_boundaries=("west", "east"), sponge_cells=10,
                                        sponge_tau=(1200.0, 120.0), solver="direct", ke_scheme="gudonov", **old),
        "step": lambda: _step_section("zstar", solver="direct", **old),
        "openbox": lambda: _ale_open_box("hybrid", nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4,
                                         sponge_cells=3, solver="direct", ke_scheme="gudonov", **old),
        # the defaults since 2026-10-14 (θ_bc ½, thin-layer coupling, FCT, outflow limiter): generated from commit
        # 7a499aa + the depth-mean fix of the final review, the end-to-end guard of the new default path (spec §4)
        "lock_defaults": lambda: _hybrid_lock(128, 21, 20.0, 20.0, solver="direct"),
        "openbox_defaults": lambda: _ale_open_box("hybrid", nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4,
                                                  sponge_cells=3, solver="direct"),
    }


@pytest.mark.parametrize("name", ["lock", "channel", "step", "openbox", "lock_defaults", "openbox_defaults"])
def test_ale_results_unchanged_by_the_performance_kernels(name):
    # scalar fingerprints (sum, Σx², max|x|, 7 samples) of the 100-step states computed before any kernel
    # (tests/data/ale_perf_reference.json, generated from the NumPy code of commit c230423)
    import json
    ref = json.load(open(Path(__file__).parent / "data" / "ale_perf_reference.json"))[name]
    m, st = _perf_cases()[name]()
    for _ in range(100):
        st = m.step(st)
    for k, r in ref.items():
        a = np.asarray(getattr(st, k), float).ravel()
        scale = max(r["maxabs"], 1e-30)
        tol = 1e-10 * scale
        assert abs(np.abs(a).max() - r["maxabs"]) <= tol, (k, "max")
        np.testing.assert_allclose(a[r["idx"]], r["vals"], rtol=0, atol=tol, err_msg=k)
        assert abs(a.sum() - r["sum"]) <= tol * a.size, (k, "sum")
        assert abs((a * a).sum() - r["sumsq"]) <= 2 * tol * scale * a.size, (k, "sumsq")


def test_ale_uses_the_lagged_lu_cg_free_surface_by_default():
    assert _ale_box("zstar").core.fs.solver == "cg"
    sig = SigmaBaroclinic(create_cartesian_grid(nx=4, ny=1, dx=1e3, dy=1e3, f0=0.0), np.full((1, 4), 20.0),
                          dt=10.0, n_layers=4)
    assert sig.core.fs.solver == "direct"                                # σ stays bit-identical


@pytest.mark.parametrize("build", ["openbox", "step"])
def test_cg_free_surface_matches_the_direct_solve(build):
    # review focus 5: open boundaries (Flather) and a 600 → 164 m step
    def make(solver):
        if build == "openbox":
            return _ale_open_box("hybrid", nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4, sponge_cells=3,
                                 solver=solver)
        return _step_section("zstar", solver=solver)
    (m1, s1), (m2, s2) = make("direct"), make("cg")
    for _ in range(50):
        s1, s2 = m1.step(s1), m2.step(s2)
    assert np.abs(s1.eta - s2.eta).max() <= 1e-9
    assert np.abs(s1.u - s2.u).max() <= 1e-9 * max(np.abs(s1.u).max(), 1e-3)


def test_transport_thickness_cache_is_keyed_on_identity_and_recomputes_new_arrays():
    # review focus 4: the cache returns the stored result only for the same array object
    m, st = _ale_open_box("zstar")
    a = m._transport_thickness(st.h)
    assert m._transport_thickness(st.h) is a
    h2 = st.h.copy()
    h2[0] *= 0.5
    b = m._transport_thickness(h2)
    assert b is not a and not np.allclose(b[0][0], a[0][0])


def test_mixing_diagnostic_off_leaves_T_unchanged():
    # ALE performance task 5.5: the T² stack is only the spurious-mixing diagnostic
    runs = []
    for mix in (True, False):
        m, st = _ale_open_box("hybrid", nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4, sponge_cells=3,
                              mix_diagnostic=mix)
        for _ in range(30):
            st = m.step(st)
        runs.append(st)
    a, b = runs
    for k in ("T", "h", "u", "v", "eta"):
        assert np.array_equal(getattr(a, k), getattr(b, k)), k
    assert a.mix_num is not None and b.mix_num is None


def test_diffusion_kernel_matches_numpy():
    # ALE performance task 5.6: random ALE state with land, a reference profile and vanished layers
    rng = np.random.default_rng(12)
    ny, nx, nz = 8, 11, 6
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.full((ny, nx), 50.0)
    H[2:4, 5] = np.nan
    m = SigmaBaroclinic(hg, H, dt=60.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        kh=25.0, vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, 50.0 / nz)),
                        pgf="fv")
    m.rho_ref = reference_profile(np.linspace(-50.0, 0.0, 20), RHO0 - TCOEF * (10.0 + 0.05 * np.linspace(-50.0, 0.0, 20)))
    h = rng.uniform(0.0, 15.0, (nz, ny, nx))
    h[1] = 0.0
    T = rng.uniform(5.0, 15.0, h.shape)
    V = m.layer_volumes(np.zeros((ny, nx)), h)
    a = m._diffuse_h(T, V, np.zeros((ny, nx)), h)
    b = m._diffuse_h_numpy(T, V, np.zeros((ny, nx)), h)
    assert np.array_equal(a, b)


def test_layer_heights_cache_is_keyed_on_identity():
    m, st = _ale_open_box("zstar")
    a = m.layer_heights(st.eta, st.h)
    assert m.layer_heights(st.eta, st.h) is a
    h2 = st.h.copy()
    h2[-1] *= 0.5
    assert m.layer_heights(st.eta, h2) is not a


# ---------------------------------------------------------------- initial state from a fine source grid
def _hybrid_step_model(nz=25, n_light=10):
    """Channel with a 600 → 164 m step (BayIW bay mouth) on hybrid layers: n_light z* layers of 1.5 m over
    isopycnal layers; linear stratification T = 20 + 0.02 z (ρ linear in z)."""
    H = np.where(np.arange(8) < 4, 600.0, 164.0)[None, :]
    m0 = _ale_box("sigma", nx=8, nz=nz, depth=600.0, dx=5000.0, bathy=H, params={})
    rho_top, rho_bot = m0.density(20.0), m0.density(20.0 - 0.02 * 600.0)
    tgt = np.concatenate([rho_top - 0.01 * np.arange(n_light, 0, -1), np.linspace(rho_top, rho_bot, nz + 1 - n_light)])
    m = _ale_box("hybrid", nx=8, nz=nz, depth=600.0, dx=5000.0, bathy=H, nu=1e-5, ah=100.0, kh=100.0, pgf="fv",
                 rho_ref="auto", params=dict(dz_top=np.full(nz, 1.5), rho_target=tgt))
    dz = 1.5                                                       # source z-levels (= top layers), bed first, clipped
    zf = -np.arange(0.0, 600.0 + dz, dz)[::-1]                     # interfaces −600 … 0
    h_src = np.clip(zf[1:, None, None], -H[None], None) - np.clip(zf[:-1, None, None], -H[None], None)
    zc = -H[None] + np.cumsum(h_src, axis=0) - 0.5 * h_src
    return m, h_src, 20.0 + 0.02 * zc


def test_hybrid_initial_state_from_a_fine_source_is_horizontally_uniform():
    # Hybrid init on σ-layers then remapped lost the surface structure of deep columns (24 m top σ layer
    # → uniform T over the top 20 m), a 0.5 °C step across the bay mouth (BayIW 2-D section).  From a fine
    # source grid (MOM6 Z_INIT_ALE_REMAPPING, MOM_state_initialization.F90:2928-2935) the top z* layers
    # get the exact layer means in deep and shallow columns alike.
    m, h_src, T_src = _hybrid_step_model()
    st = m.initial_state(np.zeros((1, 8)), T_src, h_src=h_src)
    np.testing.assert_allclose(st.h.sum(axis=0), m.H, rtol=1e-12)
    zc = m.layer_heights(st.eta, st.h)
    top = slice(-10, None)
    np.testing.assert_allclose(st.T[top], 20.0 + 0.02 * zc[top], atol=1e-9)
    np.testing.assert_allclose(st.T[top, 0, 0], st.T[top, 0, 7], atol=1e-9)


def test_hybrid_step_stays_at_rest_from_a_fine_source():
    m, h_src, T_src = _hybrid_step_model()
    st = m.initial_state(np.zeros((1, 8)), T_src, h_src=h_src)
    for _ in range(120):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-6


def test_initial_state_source_grid_must_fill_the_column():
    m, h_src, T_src = _hybrid_step_model()
    with pytest.raises(ValueError, match="h_src"):
        m.initial_state(np.zeros((1, 8)), T_src, h_src=0.5 * h_src)


@pytest.mark.parametrize("rule", ["zstar", "sigma", "hybrid"])
def test_initial_state_from_source_all_rules_with_eta_and_land(rule):
    # h_src path for every rule: Σh = H + η with η ≠ 0, layer T = source means, land column T = t_ref
    H = np.array([[600.0, 600.0, 164.0, np.nan]])
    nz = 12
    m0 = _ale_box("sigma", nx=4, nz=nz, depth=600.0, bathy=H, params={})
    rho_top, rho_bot = m0.density(20.0), m0.density(20.0 - 0.02 * 600.0)
    params = {"zstar": dict(dz_top=np.full(nz, 50.0)), "sigma": {},
              "hybrid": dict(dz_top=np.full(nz, 1.5), rho_target=np.linspace(rho_top - 0.05, rho_bot, nz + 1))}[rule]
    m = _ale_box(rule, nx=4, nz=nz, depth=600.0, bathy=H, params=params)
    eta = np.array([[0.03, -0.02, 0.01, 0.0]])
    D = np.nan_to_num(H) + eta
    n_src = 200
    h_src = np.where(np.isfinite(H), D / n_src, 0.0)[None].repeat(n_src, 0)
    zc = -np.nan_to_num(H)[None] + np.cumsum(h_src, 0) - 0.5 * h_src
    T_src = 20.0 + 0.02 * zc
    st = m.initial_state(eta, T_src, h_src=h_src)
    wet = np.isfinite(H)[0]
    np.testing.assert_allclose(st.h.sum(0)[0, wet], D[0, wet], rtol=1e-12)
    z = m.layer_heights(st.eta, st.h)
    ok = (st.h > 1.0) & wet[None, None, :]                             # end cells are PCM (remap.py): skip films
    np.testing.assert_allclose(st.T[ok], (20.0 + 0.02 * z)[ok], atol=1e-5)          # PCM bed cell: ~5e-6
    assert (st.T[:, 0, ~wet] == m.t_ref).all()


def test_initial_state_source_errors():
    m, h_src, T_src = _hybrid_step_model()
    with pytest.raises(ValueError, match="N_src"):
        m.initial_state(np.zeros((1, 8)), T_src[:-1], h_src=h_src)
    hg = create_cartesian_grid(nx=8, ny=1, dx=5000.0, dy=5000.0, f0=0.0)
    sig = SigmaBaroclinic(hg, m.H, dt=30.0, n_layers=10, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                          nu_const=1e-5, theta=0.6, d_min=0.05)
    with pytest.raises(ValueError, match="ale"):
        sig.initial_state(np.zeros((1, 8)), T_src, h_src=h_src)


from boundaries.obcs import OBData


def _west_state(T_of_z, u_of_z, depth=20.0, nz=40, ny=1):
    zw = np.linspace(0.0, -depth, nz + 1)[:, None].repeat(ny, 1)
    zc = 0.5 * (zw[1:] + zw[:-1])
    return OBData(side="west", T=T_of_z(zc), u_normal=u_of_z(zc), zw=zw)


def test_open_boundary_state_errors():
    with pytest.raises(ValueError, match="open_boundary_mode"):
        _ale_box("zstar", open_boundaries=("west",), open_boundary_mode={"east": "clamped"})
    with pytest.raises(ValueError, match="open_boundary_mode"):
        _ale_box("zstar", open_boundaries=("west",), open_boundary_mode={"west": "sticky"})
    m = _ale_box("zstar", open_boundaries=("west",),
                 open_boundary_state=lambda t: {"east": _west_state(lambda z: 10 + 0 * z, lambda z: 0 * z)})
    st = m.initial_state(np.zeros((1, 32)), np.full((10, 1, 32), 10.0))
    with pytest.raises(ValueError, match="not in open_boundaries"):
        m.step(st)


def test_no_state_keeps_the_resting_exterior_bit_for_bit():
    # the existing open-boundary channel with and without the new arguments at their defaults
    kw = dict(open_boundaries=("west", "east"), sponge_cells=4)
    a = _ale_box("zstar", **kw)
    b = _ale_box("zstar", open_boundary_state=None, open_boundary_mode=None, **kw)
    T = 10.0 + np.linspace(0, 1, 32)[None, None, :] * np.ones((10, 1, 1))
    sa, sb = a.initial_state(np.zeros((1, 32)), T), b.initial_state(np.zeros((1, 32)), T)
    for _ in range(20):
        sa, sb = a.step(sa), b.step(sb)
    assert np.array_equal(sa.u, sb.u) and np.array_equal(sa.T, sb.T) and np.array_equal(sa.h, sb.h)


def test_radiating_relaxes_the_boundary_baroclinic_velocity_to_the_exterior():
    # constant exterior u′ = 0.05 cos(πz/H); with τ_in = 10Δt the incoming boundary u′ approaches it
    H = 20.0
    st_fn = lambda t: {"west": _west_state(lambda z: 10 + 0 * z, lambda z: 0.05 * np.cos(np.pi * z / H))}  # noqa: E731
    m = _ale_box("zstar", open_boundaries=("west",), open_boundary_state=st_fn, obc_tau_in=300.0)
    st = m.initial_state(np.zeros((1, 32)), np.full((10, 1, 32), 10.0))
    for _ in range(100):
        st = m.step(st)
    zc = m.layer_heights(st.eta, st.h)[:, 0, 0]
    up = st.u[:, 0, 0] - st.u[:, 0, 0].mean()
    ref = 0.05 * np.cos(np.pi * zc / H)
    assert np.corrcoef(up, ref)[0, 1] > 0.99 and abs(np.std(up) / np.std(ref) - 1) < 0.2


def test_flather_follows_the_exterior_surface():
    # exterior η = 1 cm at a west edge of a channel closed in the east: the basin fills towards 1 cm
    st_fn = lambda t: {"west": OBData(side="west", T=np.full((4, 1), 10.0), u_normal=np.zeros((4, 1)),  # noqa: E731
                                      zw=np.linspace(0, -20, 5)[:, None], eta=np.array([0.01]))}
    m = _ale_box("zstar", nx=8, open_boundaries=("west",), open_boundary_state=st_fn)
    st = m.initial_state(np.zeros((1, 8)), np.full((10, 1, 8), 10.0))
    for _ in range(400):
        st = m.step(st)
    np.testing.assert_allclose(st.eta, 0.01, atol=1e-3)


def _mode1_state(u0, omega, N, H, nz=80):
    """Mode-1 internal wave entering at x = 0: u = u0 cos(πz/H) sin(ωt), T′ from w = −∂ζ/∂t (linear, f = 0)."""
    zw = np.linspace(0.0, -H, nz + 1)[:, None]
    zc = 0.5 * (zw[1:] + zw[:-1])
    alpha_g = 9.81 * 2e-4
    Tz = N**2 / alpha_g                                         # background dT/dz [°C/m]
    k = np.pi * omega / (N * H)

    def at(t):
        u = u0 * np.cos(np.pi * zc / H) * np.sin(omega * t)
        # u_x = −u0 k cos(πz/H) cos(ωt − kx); w = −∫u_x dz = u0 k (H/π) sin(πz/H) cos(ωt − kx) (w = 0 at z = 0, −H);
        # ζ = ∫w dt = (u0 k H/(π ω)) sin(πz/H) sin(ωt) at x = 0; T = T0(z − ζ) = 20 + Tz z − Tz ζ
        zeta = (u0 * k * H / (np.pi * omega)) * np.sin(np.pi * zc / H) * np.sin(omega * t)
        return {"west": OBData(side="west", T=20.0 + Tz * zc - Tz * zeta, u_normal=u, zw=zw)}
    return at


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_clamped_west_radiating_east_carries_the_analytic_mode1_wave(rule):
    # flat 200 m channel, N² = 1e-6, period 44 567 s (MITgcm internal_wave): a progressive mode-1 wave
    # u = u0 cos(πz/H) sin(ωt − kx), k = πω/(NH) (λ ≈ 2.8 km) leaves through a radiating + sponge east edge.
    # Interior amplitude ±5 % of the viscously decaying wave, phase ±10° after 3 periods (spec §3.2).
    # Arakawa KE: the default Godunov KE damps this wave by a further ~9 % over 5 km at dx = 100 m (first order
    # in dx; Task 4 diagnosis, reported as a numerical-choice finding), which is not what this test is about.
    H, N, P, u0, nz, nx, dx = 200.0, 1e-3, 44567.0, 0.024, 20, 100, 100.0
    omega = 2 * np.pi / P
    k = np.pi * omega / (N * H)
    Tz = N**2 / (9.81 * 2e-4)
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    Hb = np.full((1, nx), H)
    T0 = lambda z: 20.0 + Tz * z                                     # noqa: E731
    if rule == "zstar":
        params = dict(dz_top=np.full(nz, H / nz))
    else:
        m0 = SigmaBaroclinic(hg, Hb, dt=250.0, n_layers=nz, rho0=RHO0, t_coef=RHO0 * 2e-4, t_ref=20.0, z0=None,
                             closure="constant", nu_const=1e-3, vertical_coordinate="ale", regrid="sigma",
                             regrid_params={})
        rho = m0.density(T0(np.linspace(0, -H, 50)))
        sp = (rho.max() - rho.min()) / nz                               # half an interval beyond each extreme
        params = dict(dz_top=np.full(nz, 1.0), rho_target=np.linspace(rho.min() - 0.5 * sp, rho.max() + 0.5 * sp, nz + 1))
    m = SigmaBaroclinic(hg, Hb, dt=250.0, n_layers=nz, rho0=RHO0, t_coef=RHO0 * 2e-4, t_ref=20.0, z0=None,
                        closure="constant", nu_const=1e-3, ah=1e-2, kh=1e-2, kv=1e-3, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=rule, regrid_params=params,
                        open_boundaries=("west", "east"), sponge_cells=20, sponge_tau=(3000.0, 250.0),
                        open_boundary_state=_mode1_state(u0, omega, N, H),
                        open_boundary_mode={"west": "clamped", "east": "radiating"}, ke_scheme="arakawa")
    zf = np.linspace(-H, 0.0, 401)
    h_src = np.full((400, 1, nx), H / 400)
    zc_src = (zf[1:] + zf[:-1])[:, None, None] * 0.5 * np.ones((1, 1, nx))
    st = m.initial_state(np.zeros((1, nx)), T0(zc_src), h_src=h_src)
    steps = int(round(3 * P / 250.0))
    x = (np.arange(nx + 1) * dx)[None, :]
    probe = np.arange(5, 60)                                         # interior u-faces, away from the sponge
    rec = []
    for n in range(steps + int(round(P / 250.0))):                   # 3 periods spin-up, then one period recorded
        st = m.step(st)
        if n >= steps:
            hf = m._mean_face_thickness(st.h)[0][:, 0, :]
            w = hf / np.maximum(hf.sum(0), 1e-12)
            u_bot_mode = (w * st.u[:, 0, :] * np.cos(np.pi * (np.cumsum(hf, 0) - 0.5 * hf - H) / H)).sum(0)
            rec.append((st.time, u_bot_mode[probe]))
    t = np.array([r[0] for r in rec])
    a = np.array([r[1] for r in rec])                                # mode-1 projection ≈ (u0/2) sin(ωt − kx)
    c, s = np.cos(omega * t), np.sin(omega * t)
    A = np.linalg.lstsq(np.stack([s, c], 1), a, rcond=None)[0]       # a ≈ A0 sin ωt + A1 cos ωt
    amp = np.hypot(A[0], A[1]) / (0.5 * u0)
    phase = np.degrees(np.arctan2(-A[1], A[0]))                      # sin(ωt − φ) → φ = kx
    expected = np.degrees(k * x[0, probe])
    # linear viscous decay of mode 1 (ω ≪ N): amplitude rate ½[(ν + κ) m² + (A_h + K_h) k²], m = π/H, at the
    # group speed c_g = NH/π
    rate = 0.5 * ((1e-3 + 1e-3) * (np.pi / H) ** 2 + (1e-2 + 1e-2) * k**2)
    decay = np.exp(-rate * x[0, probe] / (N * H / np.pi))
    assert np.all(np.abs(amp / decay - 1.0) < 0.05)
    dphi = (phase - expected + 180.0) % 360.0 - 180.0
    assert np.all(np.abs(dphi - np.median(dphi)) < 10.0) and abs(np.median(dphi)) < 10.0


def test_clamped_edges_have_no_sponge():
    # a Dirichlet edge needs no relaxation band: a sponge toward the edge profile copied along the normal
    # forces the whole band in phase and distorts an incoming wave (Task 4 diagnosis)
    m = _ale_box("zstar", open_boundaries=("west", "east"), sponge_cells=4,
                 open_boundary_mode={"west": "clamped", "east": "radiating"},
                 open_boundary_state=lambda t: {"west": _west_state(lambda z: 10 + 0 * z, lambda z: 0 * z)})
    assert (m._sp_c[0, :4] == 0).all() and m._sp_c[0, -1] > 0
    assert (m._sp_u[0, :4] == 0).all() and m._sp_u[0, -1] > 0


def test_clamped_mode_needs_an_exterior_state():
    # a Dirichlet edge without exterior data would silently radiate with no sponge
    with pytest.raises(ValueError, match="clamped.*open_boundary_state"):
        _ale_box("zstar", open_boundaries=("west",), open_boundary_mode={"west": "clamped"})


def test_default_kinetic_energy_scheme_is_arakawa():
    # Esteban 2026-10-12: Arakawa by default — Godunov damps progressive internal waves ~9 % per 5 km at
    # dx = 100 m (validation/internal_wave/README.md); Godunov stays an option for density currents
    assert _ale_box("zstar").ke_scheme == "arakawa"


def test_hybrid_warns_when_the_z_floor_stops_above_the_bed():
    # BayIW two-layer (2026-10-13): dz_top = the first 25 delR reached 74.8 m of 600 m, so the cold water stayed one
    # 484-m slab layer that ignored the bay-mouth step (interface rise 0.06 of MITgcm's); MOM6 OM4's floor reaches
    # the bed (coord_hycom.F90:253-267 is the only thing that gives absent density targets thickness)
    import warnings
    tgt = np.linspace(1020.0, 1028.0, 10)
    with pytest.warns(UserWarning, match="floor"):
        _ale_box("hybrid", nx=4, nz=10, depth=100.0, params=dict(dz_top=np.full(10, 1.0), rho_target=tgt))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)
        _ale_box("hybrid", nx=4, nz=10, depth=100.0, params=dict(dz_top=np.full(10, 10.0), rho_target=tgt))


def test_reference_profile_kernel_matches_np_interp():
    # perf part 2, task 4: the parallel interpolation must reproduce np.interp exactly, constant beyond the ends
    from core.dynamics.sigma_baroclinic import ReferenceProfile
    assert hasattr(ReferenceProfile, "_interp_kernel")
    rng = np.random.default_rng(5)
    z = np.sort(rng.uniform(-600, 0, 50))
    prof = ReferenceProfile(z, 1020 + 0.01 * -z)
    zz = rng.uniform(-700, 50, (6, 70, 80))                     # beyond both ends too; ≥ 1000 elements: kernel path
    assert np.array_equal(prof(zz), np.interp(zz, prof.z, prof.rho))
    assert prof(zz).shape == zz.shape
    small = rng.uniform(-700, 50, (3, 4))
    assert np.array_equal(prof(small), np.interp(small, prof.z, prof.rho))


def test_advect_kernel_matches_numpy():
    # perf part 2, task 5: the fused SSP-RK3 stages reproduce the NumPy combination, vanished columns keep C
    m = _ale_box("zstar", nx=12, nz=6)
    st = m.initial_state(np.zeros((1, 12)), 10.0 + np.random.default_rng(6).random((6, 1, 12)))
    for _ in range(3):
        st = m.step(st)
    V0 = m.layer_volumes(st.eta, st.h)
    V1 = V0 * 1.01
    V1[:, :, 0] = 0.0
    ok = V1 > 0
    C = np.stack([st.T, st.T**2])
    Fz = np.zeros((m.n + 1,) + m.H.shape)
    a = m._advect(C, V0, V1, st.Qx, st.Qy, Fz, ok)
    b = m._advect_numpy(C, V0, V1, st.Qx, st.Qy, Fz, ok)
    assert np.array_equal(a, b) and np.isfinite(a).all()


def test_ab3_kernel_matches_numpy():
    # perf part 2, task 7 round 2: the AB3 combination of the explicit tendencies, fused
    from core.dynamics.sigma_baroclinic import AB3, ab3_combine
    rng = np.random.default_rng(9)
    g0, g1, g2 = (rng.normal(size=(5, 6, 7)) for _ in range(3))
    assert np.array_equal(ab3_combine(g0, g1, g2), AB3[0] * g0 + AB3[1] * g1 + AB3[2] * g2)
    assert np.array_equal(ab3_combine(g0, g1, None), 1.5 * g0 - 0.5 * g1)


def test_reference_profile_kernel_keeps_nan_heights():
    # final review: NumPy returns NaN for a NaN height; the kernel returned the bottom value when fp[0] == fp[1]
    from core.dynamics.sigma_baroclinic import ReferenceProfile
    prof = ReferenceProfile(np.array([-600.0, -500.0, -100.0, 0.0]), np.array([1027.0, 1027.0, 1025.0, 1024.0]))
    zz = np.full(2000, -300.0)
    zz[::7] = np.nan
    assert np.array_equal(prof(zz), np.interp(zz, prof.z, prof.rho), equal_nan=True)


def test_new_kernel_wrappers_reject_mismatched_shapes():
    # final review: Numba does not check bounds, so a shape mistake would read past the arrays silently
    from core.dynamics.sigma_baroclinic import ab3_combine
    from core.vertical import ale
    a, b = np.zeros((3, 4, 5)), np.zeros((3, 4, 6))
    with pytest.raises(ValueError, match="shape"):
        ab3_combine(a, b, None)
    with pytest.raises(ValueError, match="shape"):
        ale.transport_velocity(a, b, np.ones((4, 5), bool), a, np.ones((4, 5)), np.ones((4, 5)), 0.5)
    with pytest.raises(ValueError, match="shape"):
        ale.transport_velocity(a, a, np.ones((4, 6), bool), a, np.ones((4, 5)), np.ones((4, 5)), 0.5)


def test_baroclinic_theta_defaults_to_centred():
    # θ_bc = ½ in every coordinate since the outflow limiter and the thin-layer coupling (spec 2026-10-14 §5);
    # an explicit value always wins
    hyb = dict(dz_top=np.full(10, 0.01), rho_target=np.linspace(1000.0, 1001.0, 11))
    assert _ale_box("zstar").theta_bc == 0.5
    assert _ale_box("sigma").theta_bc == 0.5
    assert _ale_box("hybrid", params=hyb).theta_bc == 0.5
    assert _ale_box("hybrid", params=hyb, theta_bc=0.6).theta_bc == 0.6


def test_robustness_defaults():
    hyb = dict(dz_top=np.full(10, 0.01), rho_target=np.linspace(1000.0, 1001.0, 11))
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = _ale_box("zstar")
    assert m.thin_layer_coupling == "mom6" and m.tracer_limiter == "fct" and m.thin_layer_cutoff == 0.0
    assert _ale_box("hybrid", params=hyb).thin_layer_cutoff == 0.0          # HYCOM blending off: it locks fronts
    e = SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=30.0, z0=None, closure="constant")      # σ Eulerian
    assert e.thin_layer_coupling == "none" and e.tracer_limiter == "none" and e.thin_layer_cutoff == 0.0
    assert _ale_box("zstar", tracer_time="pc").tracer_limiter == "none"


def test_internal_wave_is_not_damped_by_the_free_surface_theta():
    # IW damping fix (2026-10-13; validation/ilicak_benchmark): the θ = 0.6 off-centring of the free surface was
    # applied to the baroclinic layer transport too, damping internal waves ~(θ−½)ω²Δt (Ilicak internal wave, ν = 0.01:
    # oscillation 0.45 of MOM6/CROCO/MITgcm's 0.65–0.72 after 200 days).  Ilicak IW, z*, Δt 600 s, 30 days.
    from validation.ilicak_benchmark import cases
    c = cases.get("iw")
    hg = create_cartesian_grid(nx=c.nx, ny=1, dx=float(c.dx[0]), dy=float(c.dy[0]), f0=0.0)
    ke = {}
    for name, kw in (("default", {}), ("old", dict(theta_bc=0.6, thin_layer_coupling="none", thin_layer_cutoff=0.0, tracer_limiter="none"))):
        m = SigmaBaroclinic(hg, c.H.copy(), dt=600.0, n_layers=c.nz, rho0=cases.RHO0, t_coef=cases.ALPHA, z0=None,
                            closure="constant", nu_const=1e-4, ah=0.01, pgf="fv", vertical_coordinate="ale",
                            regrid="zstar", regrid_params=dict(dz_top=c.dz.copy()), mix_diagnostic=False, **kw)
        eta = np.zeros((1, c.nx))
        st = m.initial_state(eta, c.T_fn(c.x[None, None, :], c.y[None, :, None], m.layer_heights(eta)))
        k_t = []
        for n in range(1, 30 * 144 + 1):
            st = m.step(st)
            if n % 36 == 0:
                hu = np.zeros_like(st.u[:, 0])
                hu[:, 1:-1] = 0.5 * (st.h[:, 0, 1:] + st.h[:, 0, :-1])
                k_t.append((st.u[:, 0] ** 2 * hu).sum())
        k_t = np.array(k_t)                                        # every 6 h
        ke[name] = k_t[80:].mean() / k_t[:20].mean()              # days 20–30 / days 0–5
    assert ke["default"] > 0.8 and ke["old"] < 0.75, ke


def test_thin_layer_coupling_ties_a_vanished_layer_to_its_neighbours():
    # one column of the implicit viscosity with the coupling thickness of 'mom6': a layer vanished on one side (1 µm)
    # between two 10 m layers takes their velocity in one step — its relaxation time h_thin·h_int/ν ≈ 0.1 s ≪ Δt
    # (a 1 mm layer: ≈ 100 s, tied within a few steps of 10 s); momentum Σ dz u is conserved
    from core.dynamics.sigma_wd import vertical_implicit
    from core.vertical import ale
    hu_mean = np.array([[10.0], [25.0], [10.0]])                    # thin layer looks 25 m thick on the mean
    q_a = np.array([[10.0], [1e-6], [10.0]]); q_b = np.array([[10.0], [50.0], [10.0]])
    dz = np.maximum(ale.viscous_face_thickness(q_a, q_b), ale.DZ_FLOOR)
    u0 = np.array([[1.0], [5.0], [0.0]])
    nu = np.full((4, 1), 1e-4)
    u1 = vertical_implicit(u0, dz, nu, np.zeros(1), 10.0)
    assert abs(u1[1, 0] - 0.5 * (u1[0, 0] + u1[2, 0])) < 0.01 * abs(u0[1, 0] - 0.5)
    np.testing.assert_allclose((dz * u1).sum(), (dz * u0).sum(), rtol=1e-12)
    u_old = vertical_implicit(u0, hu_mean, nu, np.zeros(1), 10.0)   # the mean thickness barely couples it
    assert abs(u_old[1, 0] - u0[1, 0]) < 0.01


def test_thin_layer_coupling_option_is_validated():
    with pytest.raises(ValueError, match="thin_layer_coupling"):
        _ale_box("zstar", thin_layer_coupling="harmonic")


def test_thin_layer_cutoff_option_is_validated():
    assert _ale_box("zstar", thin_layer_cutoff=0.5).thin_layer_cutoff == 0.5
    with pytest.raises(ValueError, match="thin_layer_cutoff"):
        _ale_box("zstar", thin_layer_cutoff=-1.0)


# ---------------------------------------------------------------- FCT (spec 2026-10-14-hybrid-robustness §4)
def _step_box(nx=60, limiter="fct", T=None, u0=0.6, nz=4, **kw):
    """z* box with a uniform eastward layer velocity and a T step: one advection call of the ALE step."""
    m = _ale_box("zstar", nx=nx, nz=nz, dx=1000.0, dt=30.0, tracer_limiter=limiter, mix_diagnostic=False, **kw)
    st = m.initial_state(np.zeros((1, nx)), np.where(np.arange(nx) < nx // 2, 1.0, 0.0)[None, None, :].repeat(nz, 0)
                         if T is None else T)
    return m, st


def test_fct_keeps_a_pulse_inside_its_range_where_weno_overshoots():
    # a two-cell square pulse at Courant 0.5: unlimited WENO5-Z overshoots by 0.10 in one step (a sharp step does not)
    nx, dt, dx = 60, 30.0, 1000.0
    pulse = np.where((np.arange(nx) >= 20) & (np.arange(nx) < 22), 1.0, 0.0)[None, None, :].repeat(4, 0)
    for limiter, bounded in (("fct", True), ("none", False)):
        m, st = _step_box(nx, limiter, T=pulse)
        V = m.layer_volumes(st.eta, st.h)
        Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = 0.5 * dx / dt * st.h[:, :, 1:] * dx    # Courant 0.5 eastwards
        Qx[:, :, -2:] = 0.0
        Qy = np.zeros(st.v.shape)
        div = Qx[..., 1:] - Qx[..., :-1]
        V1 = V - dt * div
        ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
        T = st.T[None]
        out = m._advect_fct(T, V, V1, Qx, Qy, ok) if limiter == "fct" else m._advect(T, V, V1, Qx, Qy,
                                                                                    np.zeros((m.n + 1, 1, nx)), ok)
        inside = (out.min() >= -1e-12) and (out.max() <= 1 + 1e-12)
        assert inside == bounded, (limiter, out.min(), out.max())


def test_fct_with_unit_coefficients_is_the_rk3_result():
    m, st = _step_box(40, "fct")
    V = m.layer_volumes(st.eta, st.h)
    rng = np.random.default_rng(0)
    Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = rng.uniform(-0.3, 0.3, Qx[:, :, 1:-1].shape) * 1000.0 * st.h[:, :, 1:]
    Qy = np.zeros(st.v.shape)
    V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
    ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
    T = (np.sin(np.arange(40) / 5.0)[None, None, None, :] + 2.0) * np.ones(st.T.shape)[None]
    ref = m._advect(T, V, V1, Qx, Qy, np.zeros((m.n + 1, 1, 40)), ok)
    got = m._advect_fct(T, V, V1, Qx, Qy, ok, unlimited=True)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_fct_conserves_and_keeps_high_order_on_a_smooth_profile():
    nx = 80
    errs = {}
    for limiter in ("fct", "none"):
        m, st = _step_box(nx, limiter)
        x = np.arange(nx)
        T0 = (1.0 + 0.5 * np.sin(2 * np.pi * x / nx))[None, None, :].repeat(m.n, 0)
        st = m.initial_state(np.zeros((1, nx)), T0)
        V = m.layer_volumes(st.eta, st.h)
        Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = 0.5 * 1000.0 / m.dt * st.h[:, :, 1:] * 1000.0
        Qx[:, :, -1] = 0.0; Qx[:, :, 1] = 0.0
        V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
        ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
        out = (m._advect_fct(T0[None], V, V1, Qx, np.zeros(st.v.shape), ok) if limiter == "fct" else
               m._advect(T0[None], V, V1, Qx, np.zeros(st.v.shape), np.zeros((m.n + 1, 1, nx)), ok))[0]
        np.testing.assert_allclose((V1 * out).sum(), (V * T0).sum(), rtol=1e-12)
        exact = 1.0 + 0.5 * np.sin(2 * np.pi * (x - 0.5) / nx)                            # shifted by 0.5 cell
        errs[limiter] = np.abs(out[0, 0, 5:-5] - exact[5:-5]).max()
    assert errs["fct"] <= 1.5 * errs["none"], errs


def test_fct_bounds_a_thin_layer_receiving_other_water():
    m, st = _step_box(20, "fct")
    h = st.h.copy(); h[1, 0, 10] = 1e-3                              # 1 mm cell in layer 1
    T = np.ones_like(st.T); T[1, 0, 10] = 0.0; T[1, 0, :10] = 5.0
    V = m.layer_volumes(st.eta, h)
    Qx = np.zeros(st.u.shape); Qx[1, 0, 10] = 2.0; Qx[1, 0, 11] = 1.0                 # inflow 2, outflow 1 m³/s
    V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
    ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
    out = m._advect_fct(T[None], V, V1, Qx, np.zeros(st.v.shape), ok)[0]
    assert 0.0 - 1e-12 <= out[1, 0, 10] <= 5.0 + 1e-12


def test_fct_limits_each_tracer_of_the_stack():
    m, st = _step_box(30, "fct")
    V = m.layer_volumes(st.eta, st.h)
    Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = 0.9 * 1000.0 / m.dt * st.h[:, :, 1:] * 1000.0; Qx[:, :, -2:] = 0.0
    V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
    ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
    C = np.stack([st.T, st.T ** 2 * 3.0])
    out = m._advect_fct(C, V, V1, Qx, np.zeros(st.v.shape), ok)
    for t, (lo, hi) in enumerate(((0.0, 1.0), (0.0, 3.0))):
        assert out[t].min() >= lo - 1e-12 and out[t].max() <= hi + 1e-12


def test_fct_open_edge_faces_are_upstream():
    m, st = _ale_open_box("zstar", tracer_limiter="fct")
    C = st.T[None]
    Qx = np.zeros(st.u.shape); Qx[:, :, 0] = 3.0; Qx[:, :, -1] = -2.0
    FH = m._faces_h(C, Qx, np.zeros(st.v.shape), np.ones(C.shape[1:], bool))[0]
    FL = m._upwind_faces_h(C, Qx, np.zeros(st.v.shape))[0]
    np.testing.assert_array_equal(FH[..., [0, -1]], FL[..., [0, -1]])


def test_fct_numba_coefficients_match_numpy():
    from core.dynamics.sigma_baroclinic import _fct_coeff_kernel, fct_coefficients_numpy
    rng = np.random.default_rng(3)
    nt, n, ny, nx = 2, 3, 4, 5
    c0 = rng.uniform(0, 1, (nt, n, ny, nx)); cL = rng.uniform(0, 1, (nt, n, ny, nx))
    V1 = rng.uniform(0.5, 2, (n, ny, nx)) * 1e6
    Ax = rng.normal(0, 1e4, (nt, n, ny, nx + 1)); Ax[..., [0, -1]] = 0
    Ay = rng.normal(0, 1e4, (nt, n, ny + 1, nx)); Ay[..., [0, -1], :] = 0
    ok = rng.uniform(0, 1, (n, ny, nx)) > 0.1
    ref = fct_coefficients_numpy(c0, cL, V1, Ax, Ay, ok, 30.0)
    got = _fct_coeff_kernel(c0, cL, V1, Ax, Ay, ok, 30.0)
    np.testing.assert_allclose(got[0], ref[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(got[1], ref[1], rtol=0, atol=1e-14)


def test_tracer_limiter_option_is_validated():
    with pytest.raises(ValueError, match="tracer_limiter"):
        _ale_box("zstar", tracer_limiter="clip")
    with pytest.raises(ValueError, match="tracer_limiter"):
        _ale_box("zstar", tracer_limiter="fct", tracer_time="pc")


def test_fct_bounds_include_the_layers_above_and_below():
    # an along-layer maximum (σ layers bent over a summit) spanned by the layer above is not clipped to donor cell
    from core.dynamics.sigma_baroclinic import _fct_coeff_kernel, fct_coefficients_numpy
    c0 = np.zeros((1, 3, 1, 3)); c0[0, 1, 0] = [0.5, 1.0, 0.5]; c0[0, 2, 0] = 2.0     # layers 0 (=0), 1 (bent), 2
    cL = c0.copy()
    V1 = np.full((3, 1, 3), 1e6)
    Ax = np.zeros((1, 3, 1, 4)); Ax[0, 1, 0, 1] = 1e3                # antidiffusive flux into layer 1's peak
    Ay = np.zeros((1, 3, 2, 3))
    ok = np.ones((3, 1, 3), bool)
    for f in (fct_coefficients_numpy, _fct_coeff_kernel):
        Cx, _ = f(c0, cL, V1, Ax, Ay, ok, 30.0)
        assert Cx[0, 1, 0, 1] > 0.5, f.__name__


@pytest.mark.parametrize("open_", [("west", "east", "south", "north"), ()])
def test_fct_fused_kernels_match_the_numpy_oracle(open_):
    # the Numba path of the FCT (stages, donor/antidiffusive fluxes, final update) = the NumPy reference, with
    # 2-D flow, two tracers, a thin and a dry cell, and inflow/outflow on all four open edges (or closed)
    m, st = _ale_open_box("zstar", nx=14, ny=12, open_=open_, tracer_limiter="fct")
    rng = np.random.default_rng(7)
    h = st.h.copy(); h[2, 5, 6] = 1e-3
    V = m.layer_volumes(st.eta, h)
    Qx = rng.uniform(-0.4, 0.4, st.u.shape) * 1e3 * np.concatenate([h[..., :1], h], axis=-1)
    Qy = rng.uniform(-0.4, 0.4, st.v.shape) * 1e3 * np.concatenate([h[:, :1], h], axis=1)
    if not open_:
        Qx[..., [0, -1]] = 0.0; Qy[:, [0, -1]] = 0.0
    V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:] - Qy[:, :-1])
    ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
    T = 7.0 + rng.normal(0, 1, st.T.shape)
    C = np.stack([T, T ** 2])
    ref = m._advect_fct_numpy(C, V, V1, Qx, Qy, ok)
    got = m._advect_fct(C, V, V1, Qx, Qy, ok)
    np.testing.assert_allclose(got, ref, rtol=1e-13, atol=0)


def test_thin_layer_coupling_keeps_the_wind_driven_depth_mean():
    # a top layer of 1 mm on the west side of a face, wind 0.1 N/m²: with the coupling thickness the wind entered ū
    # as Δt τ / h_harmonic (η 6.5e-3 m vs 1.4e-4 after one step; final review 2026-10-14); now both couplings agree
    import dataclasses
    nx, out = 32, {}
    for cpl in ("none", "mom6"):
        m = _ale_box("zstar", nx=nx, nz=10, depth=20.0, dt=30.0, nu=1e-4, thin_layer_coupling=cpl,
                     mix_diagnostic=False)
        st = m.initial_state(np.zeros((1, nx)), np.full((10, 1, nx), 7.0))
        h = st.h.copy(); h[-2, :, :16] += h[-1, :, :16] - 1e-3; h[-1, :, :16] = 1e-3
        u = st.u.copy(); u[-1, :, 16] = 0.01                                      # drains the thin side
        st = dataclasses.replace(st, h=h, u=u)
        tau = np.zeros((1, nx + 1)); tau[:, 1:-1] = 0.1
        out[cpl] = m.step(st, tau_x=tau)
    eta_none, eta_mom6 = np.abs(out["none"].eta).max(), np.abs(out["mom6"].eta).max()
    assert eta_mom6 < 1.2 * eta_none, (eta_none, eta_mom6)


def test_thin_layer_coupling_forces_the_depth_mean_on_the_clipped_face_column():
    # a 20 → 12 m step with no thin layer: the coupling thickness equals the mean one, so both couplings must give
    # the same step; the depth-mean correction first divided τ by the mean cell depth (16 m) instead of the face
    # column clipped at the shallower bed (12 m) that the free surface uses (BayIW seiche range −2.5 %, 2026-10-14)
    nx, out = 32, {}
    H = np.where(np.arange(nx) < 16, 20.0, 12.0)[None, :]
    for cpl in ("none", "mom6"):
        m = _ale_box("zstar", nx=nx, nz=10, bathy=H, dt=30.0, nu=1e-4, thin_layer_coupling=cpl,
                     mix_diagnostic=False)
        st = m.initial_state(np.zeros((1, nx)), np.full((10, 1, nx), 7.0))
        tau = np.zeros((1, nx + 1)); tau[:, 1:-1] = 0.1
        out[cpl] = m.step(st, tau_x=tau)
    np.testing.assert_allclose(out["mom6"].eta, out["none"].eta, rtol=0, atol=1e-12 * np.abs(out["none"].eta).max())


# ---------------------------------------------------------------- passive tracers (DOME spec §1)
def _tracer_box(rule, ntr=1, open_=False, **kw):
    """Sloping 2-D channel with stratification and a sloshing surface; tracer 0 = T at t = 0."""
    nx, nz = 24, 8
    H = np.linspace(30.0, 12.0, nx)[None]
    params = dict(dz_top=np.full(nz, 30.0 / nz))
    if rule == "hybrid":
        params["rho_target"] = np.linspace(996.0, 999.5, nz + 1)
    extra = {}
    if open_:
        from boundaries.obcs import OBData
        zw = np.linspace(0.0, -30.0, 4)[:, None]

        def state(t):
            return {"west": OBData(side="west", T=np.full((3, 1), 14.0), u_normal=np.full((3, 1), 0.02), zw=zw,
                                   ubar=np.full(1, 0.02))}
        extra = dict(open_boundaries=("west",), open_boundary_mode={"west": "clamped"}, open_boundary_state=state)
    m = _ale_box(rule, nx=nx, nz=nz, bathy=H, params=params, dt=20.0, n_tracers=ntr, **extra, **kw)
    zc = m.layer_heights(np.zeros((1, nx)))
    T = 10.0 + 0.2 * (zc + 30.0)
    eta = 0.05 * np.cos(np.pi * np.arange(nx) / (nx - 1))[None]
    return m, m.initial_state(eta, T, tr=np.stack([T] * ntr) if ntr else None)


@pytest.mark.parametrize("rule", ["zstar", "sigma", "hybrid"])
def test_a_passive_tracer_started_as_T_stays_identical_to_T(rule):
    # oracle: same stack, same kernels, same remap — any special-casing of tracers breaks bit-identity
    m, st = _tracer_box(rule)
    for _ in range(40):
        st = m.step(st)
    assert st.tr.shape == (1,) + st.T.shape
    assert np.array_equal(st.tr[0], st.T)


def test_passive_tracer_is_conserved_and_stays_in_0_1_in_a_closed_box():
    m, st = _tracer_box("hybrid")
    x = np.arange(24)[None, None, :]
    st.tr = np.where((x > 8) & (x < 14), 1.0, 0.0) * np.ones_like(st.T)[None]
    c0 = m.tracer_content(st, tracer=0)
    for _ in range(60):
        st = m.step(st)
    assert abs(m.tracer_content(st, tracer=0) - c0) <= 1e-12 * abs(c0)
    assert st.tr.min() >= -1e-12 and st.tr.max() <= 1.0 + 1e-12          # FCT + PPM remap: round-off only


def test_inflow_tracer_is_zero_when_the_exterior_gives_no_tracers():
    m, st = _tracer_box("zstar", open_=True)
    st.tr = np.zeros_like(st.tr)
    for _ in range(30):
        st = m.step(st)
    assert np.abs(st.tr).max() == 0.0


def test_tracers_need_ale_and_matching_shapes():
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    with pytest.raises(ValueError, match="n_tracers"):
        SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=10.0, n_layers=4, z0=None, n_tracers=1)
    m, _ = _tracer_box("zstar", ntr=2)
    with pytest.raises(ValueError, match="tr"):
        m.initial_state(np.zeros((1, 24)), np.full((8, 1, 24), 10.0), tr=np.zeros((1, 8, 1, 24)))


def test_no_tracers_leaves_the_state_without_tr():
    m, st = _tracer_box("zstar", ntr=0)
    assert m.step(st).tr is None


# ---------------------------------------------------------------- interior sponge (DOME spec §2)
def _sponge_box(rate=1e-4, rule="zstar", **kw):
    nx, nz = 16, 6
    zw = np.linspace(0.0, -20.0, 5)
    target = np.full(4, 10.0)                                          # uniform: the remap of T* is exact
    r = np.zeros((1, nx))
    r[:, :4] = rate
    m = _ale_box(rule, nx=nx, nz=nz, depth=20.0, dt=60.0, sponge_rate=r, sponge_T=target, sponge_zw=zw, **kw)
    return m, m.initial_state(np.zeros((1, nx)), np.full((nz, 1, nx), 15.0))


def test_interior_sponge_relaxes_T_at_the_implicit_rate_and_leaves_the_rest_alone():
    m, st = _sponge_box()
    for _ in range(10):
        st = m.step(st)
    expected = 10.0 + 5.0 / (1.0 + 60.0 * 1e-4) ** 10                   # (1 + Δt r)⁻ⁿ toward T* = 10
    np.testing.assert_allclose(st.T[:, 0, 0], expected, rtol=1e-10)      # inside the band (neighbours alike)
    np.testing.assert_allclose(st.T[:, 0, 8:], 15.0, atol=1e-10)          # outside: WENO stencils see only 15


def test_interior_sponge_with_zero_rate_is_bit_identical_to_none():
    m0, s0 = _sponge_box(rate=0.0)
    m1 = _ale_box("zstar", nx=16, nz=6, depth=20.0, dt=60.0)
    s1 = m1.initial_state(np.zeros((1, 16)), np.full((6, 1, 16), 15.0))
    for _ in range(5):
        s0, s1 = m0.step(s0), m1.step(s1)
    assert np.array_equal(s0.T, s1.T) and np.array_equal(s0.h, s1.h)


def test_interior_sponge_ignores_land_and_checks_shapes():
    nx = 16
    H = np.full((1, nx), 20.0)
    H[0, :2] = np.nan
    r = np.full((1, nx), 1e-4)
    m = _ale_box("zstar", nx=nx, nz=6, bathy=H, dt=60.0, sponge_rate=r, sponge_T=np.full(4, 10.0),
                 sponge_zw=np.linspace(0.0, -20.0, 5))
    st = m.step(m.initial_state(np.zeros((1, nx)), np.full((6, 1, nx), 15.0)))
    assert np.isfinite(st.T[:, 0, 2:]).all()
    with pytest.raises(ValueError, match=r"sponge_rate shape"):
        _ale_box("zstar", nx=nx, nz=6, dt=60.0, sponge_rate=np.zeros((2, nx)), sponge_T=np.full(4, 10.0),
                 sponge_zw=np.linspace(0.0, -20.0, 5))


def test_interior_sponge_relaxes_passive_tracers_toward_their_target():
    m, st = _sponge_box(n_tracers=1)
    st.tr = np.ones_like(st.tr)
    for _ in range(10):
        st = m.step(st)
    np.testing.assert_allclose(st.tr[0, :, 0, 0], 1.0 / (1.0 + 60.0 * 1e-4) ** 10, rtol=1e-10)
    np.testing.assert_allclose(st.tr[0, :, 0, 8:], 1.0, atol=1e-10)


# ---------------------------------------------------------------- quadratic bed drag in ALE (DOME spec §3)
def _drag_channel(rule="zstar", U0=0.5, cd=2e-3, nz=6, depth=50.0, **kw):
    """Homogeneous closed channel, uniform flow U0 at t = 0 (no pressure gradient: uniform η, uniform T)."""
    nx = 12
    m = _ale_box(rule, nx=nx, nz=nz, depth=depth, dt=60.0, bottom_drag="quadratic", cd=cd, nu=1e-2, **kw)
    st = m.initial_state(np.zeros((1, nx)), np.full((nz, 1, nx), 10.0))
    u = np.zeros_like(st.u)
    u[..., 1:-1] = U0
    st.u = u
    return m, st


def _ubar(st, i=6):
    h = 0.5 * (st.h[:, 0, i - 1] + st.h[:, 0, i])
    return float((st.u[:, 0, i] * h).sum() / h.sum())


def test_quadratic_drag_removes_the_bed_stress_from_the_depth_mean():
    # closed channel: walls make the flow slosh, so the drag is isolated by differencing against cd = 0;
    # one step: Δū = −Δt C_d |u_b| u_b / D (implicit factor 1/(1 + Δt r/D) ≈ 1 − 1e-6 here)
    a = _drag_channel(cd=0.0)
    b = _drag_channel(cd=2e-3)
    ua, ub = _ubar(a[0].step(a[1])), _ubar(b[0].step(b[1]))
    np.testing.assert_allclose(ub - ua, -60.0 * 2e-3 * 0.5 * 0.5 / 50.0, rtol=0.02)


def test_quadratic_drag_reaches_the_flow_above_vanished_bed_layers():
    # z* with 10 levels over 20 m in a 7-m column: the bed layers vanish; the drag must still act on the water
    nx, nz = 12, 10
    H = np.full((1, nx), 7.0)
    out = []
    for cd in (0.0, 2e-3):
        m = _ale_box("zstar", nx=nx, nz=nz, bathy=H, params=dict(dz_top=np.full(nz, 2.0)), dt=60.0,
                     bottom_drag="quadratic", cd=cd, nu=1e-2)
        st = m.initial_state(np.zeros((1, nx)), np.full((nz, 1, nx), 10.0))
        st.u = np.where(np.arange(nx + 1)[None, None] % nx == 0, 0.0, 0.3) * np.ones_like(st.u)
        for _ in range(20):
            st = m.step(st)
        assert np.isfinite(st.u).all()
        out.append(_ubar(st))
    expected = 20 * 60.0 * 2e-3 * 0.3**2 / 7.0                           # ≈ 0.031 m/s removed by the drag
    assert out[0] - out[1] > 0.5 * expected


def test_bottom_drag_off_is_bit_identical_and_background_velocity_alone_does_not_accelerate():
    m0 = _ale_box("zstar", nx=12, nz=6, depth=50.0, dt=60.0, nu=1e-2)
    m1, _ = _drag_channel(U0=0.0, drag_bg_vel=0.05)
    s0 = m0.initial_state(np.zeros((1, 12)), np.full((6, 1, 12), 10.0))
    s1 = m1.initial_state(np.zeros((1, 12)), np.full((6, 1, 12), 10.0))
    for _ in range(10):
        s0, s1 = m0.step(s0), m1.step(s1)
    assert np.array_equal(s1.u, s0.u)
    with pytest.raises(ValueError, match="bottom_drag"):
        _ale_box("zstar", nx=12, nz=6, depth=50.0, dt=60.0, bottom_drag="loglaw")


def test_bed_drag_rates_are_finite_on_empty_faces():
    # dry or fully vanished face columns: no bbl water → only the background velocity sets r
    m, st = _drag_channel(drag_bg_vel=0.05)
    ru, rv = m._bed_drag_rates(st.u, st.v, np.zeros_like(st.u), np.zeros_like(st.v))
    np.testing.assert_allclose(ru, 2e-3 * 0.05, rtol=1e-14)
    np.testing.assert_allclose(rv, 2e-3 * 0.05, rtol=1e-14)


# ---------------------------------------------------------------- prescribed inflow transport (DOME spec §4)
def _inflow_box(Q=-50.0, H=None):
    """Closed 2-D box (x, y = 3 rows) fed through its north edge: 'clamped' with an exact transport."""
    from boundaries.obcs import OBData
    nx, ny, nz = 6, 3, 4
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.full((ny, nx), 20.0) if H is None else H
    zw = np.linspace(0.0, -20.0, nz + 1)[:, None] * np.ones((1, nx))

    def state(t):
        q = np.full(nx, Q)
        return {"north": OBData(side="north", T=np.full((nz, nx), 9.0), u_normal=np.full((nz, nx), Q / 20000.0),
                                zw=zw, transport=q)}
    m = SigmaBaroclinic(hg, H, dt=30.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 5.0)), open_boundaries=("north",),
                        open_boundary_mode={"north": "clamped"}, open_boundary_state=state)
    return m, m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 10.0))


def test_prescribed_transport_enters_exactly_and_the_volume_grows_by_it():
    m, st = _inflow_box()
    V0 = float(np.nansum((m.H + st.eta) * m.area))
    for _ in range(20):
        st = m.step(st)
        np.testing.assert_allclose(st.Qy[:, -1, :].sum(axis=0), -50.0, rtol=1e-12)
    V1 = float(np.nansum((m.H + st.eta) * m.area))
    np.testing.assert_allclose(V1 - V0, 20 * 30.0 * 6 * 50.0, rtol=1e-10)


def test_prescribed_transport_skips_faces_next_to_land():
    H = np.full((3, 6), 20.0)
    H[-1, :2] = np.nan
    m, st = _inflow_box(H=H)
    for _ in range(5):
        st = m.step(st)
    assert np.all(st.Qy[:, -1, :2] == 0.0)
    np.testing.assert_allclose(st.Qy[:, -1, 2:].sum(axis=0), -50.0, rtol=1e-12)
