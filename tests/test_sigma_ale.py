"""
Tests for the ALE mode of core/dynamics/sigma_baroclinic.py (spec
docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md §2, §5, §7): configuration,
prognostic layer thickness, Lagrangian step + regrid + remap, finite-volume pressure gradient.
"""

from __future__ import annotations

import functools

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
                           nu_const=nu, theta=0.6, d_min=0.05, vertical_coordinate="ale", regrid=rule,
                           regrid_params=params, **kw)


# ---------------------------------------------------------------- configuration
@pytest.mark.parametrize("kw, match", [
    (dict(vertical_coordinate="zz"), "vertical_coordinate"),
    (dict(regrid="isopycnal"), "regrid"),
    (dict(remap_scheme="pqm"), "remap_scheme"),
    (dict(open_boundaries=("west",)), "open boundaries"),
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
    m = _ale_box("sigma", nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params={})
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
                 **kw)
    T = np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :].repeat(nz, 0) / TCOEF
    return m, m.initial_state(np.zeros((1, nx)), T)


@functools.lru_cache(maxsize=None)
def _lock_run(rule: str, nx: int = 128, nz: int = 21, depth: float = 20.0, dt: float = 20.0):
    """6 h lock exchange: front speeds / Benjamin for 1 m and 5 m thick fronts, RPE gain, checks."""
    if rule == "hybrid":
        m, st = _hybrid_lock(nx, nz, depth, dt)
    else:
        m = _ale_box(rule, nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, params={})
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
