"""
Tests for core/dynamics/sigma_baroclinic.py — stratified σ-layer model:
flux-form tracers consistent with the free surface, diasigma velocity ω,
baroclinic pressure gradient, AB3 momentum tendencies and online mixing
diagnostics.
"""

from __future__ import annotations

import numpy as np

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from diagnostics.mixing import reference_potential_energy

RHO0, TCOEF = 1000.0, 0.28


def _box(nx=32, nz=10, depth=20.0, dx=2000.0, dt=30.0, ah=0.0, **kw):
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    H = np.full((1, nx), depth)
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                           nu_const=1e-6, ah=ah, theta=0.6, d_min=0.05, **kw)


def test_uniform_tracer_stays_uniform_and_is_conserved_while_the_surface_sloshes():
    m = _box()
    x = (np.arange(32) + 0.5) * 2000.0
    eta = 0.5 * np.cos(np.pi * x / 64e3)[None, :]
    T = np.full((10, 1, 32), 7.0)
    st = m.initial_state(eta, T)
    c0 = m.tracer_content(st)
    for _ in range(200):
        st = m.step(st)
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-13
    assert np.abs(st.eta).max() > 0.1                                   # it really sloshed


def test_resting_stratification_over_a_flat_bottom_stays_at_rest():
    m = _box()
    z = np.linspace(0, 1, 10)[:, None, None]
    T = np.broadcast_to(5.0 + 10.0 * z, (10, 1, 32)).copy()             # warm on top, horizontally uniform
    st = m.initial_state(np.zeros((1, 32)), T)
    for _ in range(100):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-12 and np.abs(st.eta).max() < 1e-12


def test_lock_exchange_front_moves_at_the_benjamin_speed_and_conserves_heat():
    # Benjamin (1968): energy-conserving gravity current front U = ½ √(g′ H); measured after the
    # initial acceleration (2–6 h).  CROCO GRAV_ADJ gives 0.46–0.49 m/s, the advective-form
    # momentum advection only 0.42 (front speed is set by the momentum flux balance).
    nx, nz, depth, dt = 128, 20, 20.0, 20.0
    m = _box(nx=nx, nz=nz, depth=depth, dx=500.0, dt=dt, ah=0.0)
    T = np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :].repeat(nz, 0) / TCOEF   # cold (dense) left
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
    assert m.rpe(st) >= r0                                                # mixing only increases RPE


def test_online_numerical_mixing_matches_the_global_variance_loss():
    nx, nz = 40, 10
    m = _box(nx=nx, nz=nz, dx=1000.0, dt=20.0, ah=1.0)
    T = np.where(np.arange(nx) < nx // 2, 10.0, 0.0)[None, None, :].repeat(nz, 0)
    st = m.initial_state(np.zeros((1, nx)), T)
    for _ in range(50):
        var0 = m.tracer_content(st, power=2)
        st = m.step(st)
        lost = (var0 - m.tracer_content(st, power=2)) / m.dt
        assert np.isclose(st.mix_num.sum(), lost, rtol=1e-9, atol=1e-12 * abs(var0))
    assert st.mix_num.sum() > 0


def test_rpe_helper_uses_the_layer_volumes():
    m = _box(nx=8, nz=4)
    T = np.zeros((4, 1, 8))
    T[:2] = 10.0                                                          # warm (light) bottom → unstable, RPE sorts it
    st = m.initial_state(np.zeros((1, 8)), T)
    rho = RHO0 - TCOEF * T
    vol = np.full(rho.shape, 2000.0 * 2000.0 * 5.0)
    assert np.isclose(m.rpe(st), reference_potential_energy(rho, vol, area=8 * 2000.0 * 2000.0))


# ---------------------------------------------------------------- face reconstructions
def test_weno5_faces_are_fifth_order_on_smooth_data():
    from core.dynamics.sigma_baroclinic import _weno5_faces
    errs = []
    for n in (40, 80):
        x = (np.arange(n) + 0.5) / n
        h = 1.0 / n
        c = (np.cos(2 * np.pi * (x - h / 2)) - np.cos(2 * np.pi * (x + h / 2))) / (2 * np.pi * h)   # cell means of sin
        faces = _weno5_faces(c, np.ones(n + 1), np.ones(n, bool), axis=0)
        exact = np.sin(2 * np.pi * np.arange(n + 1) / n)
        errs.append(np.abs(faces[4:-3] - exact[4:-3]).max())
    assert np.log2(errs[0] / errs[1]) > 4.0


def test_weno5_faces_do_not_overshoot_a_step():
    from core.dynamics.sigma_baroclinic import _weno5_faces
    c = np.where(np.arange(30) < 15, 0.0, 1.0)
    for sign in (1.0, -1.0):
        f = _weno5_faces(c, np.full(31, sign), np.ones(30, bool), axis=0)
        assert f.min() > -1e-3 and f.max() < 1 + 1e-3


def test_lock_exchange_with_weno5_stays_within_the_initial_range():
    nx, nz = 64, 20
    m = _box(nx=nx, nz=nz, dx=1000.0, dt=20.0, ah=0.0, tracer_scheme="weno5")
    T = np.where(np.arange(nx) < nx // 2, 5.0, 10.0)[None, None, :].repeat(nz, 0)
    st = m.initial_state(np.zeros((1, nx)), T)
    for _ in range(int(3 * 3600 / 20.0)):
        st = m.step(st)
    assert st.T.min() > 5.0 - 0.05 and st.T.max() < 10.0 + 0.05           # UP3 overshoots by ~50 %


# ---------------------------------------------------------------- predictor–corrector tracer stepping
def test_centred6_faces_are_sixth_order():
    from core.dynamics.sigma_baroclinic import _centred_faces
    errs = []
    for n in (40, 80):
        x = (np.arange(n) + 0.5) / n
        h = 1.0 / n
        c = (np.cos(2 * np.pi * (x - h / 2)) - np.cos(2 * np.pi * (x + h / 2))) / (2 * np.pi * h)
        faces = _centred_faces(c, np.ones(n, bool), axis=0)
        exact = np.sin(2 * np.pi * np.arange(n + 1) / n)
        errs.append(np.abs(faces[4:-3] - exact[4:-3]).max())
    assert np.log2(errs[0] / errs[1]) > 5.0


def test_predictor_corrector_tracers_keep_constancy_conservation_and_the_mixing_identity():
    m = _box(nx=40, nz=10, dx=1000.0, dt=20.0, tracer_time="pc")
    x = (np.arange(40) + 0.5) * 1000.0
    st = m.initial_state(0.4 * np.cos(np.pi * x / 40e3)[None, :], np.full((10, 1, 40), 3.0))
    for _ in range(100):
        st = m.step(st)
    np.testing.assert_allclose(st.T, 3.0, atol=1e-11)
    T = np.where(np.arange(40) < 20, 10.0, 0.0)[None, None, :].repeat(10, 0)
    st = m.initial_state(np.zeros((1, 40)), T)
    c0 = m.tracer_content(st)
    for _ in range(50):
        var0 = m.tracer_content(st, power=2)
        st = m.step(st)
        lost = (var0 - m.tracer_content(st, power=2)) / m.dt
        assert np.isclose(st.mix_num.sum(), lost, rtol=1e-9, atol=1e-12 * abs(var0))
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-12


# ---------------------------------------------------------------- pressure-gradient schemes
def _seamount_model(pgf, rho_ref=None, nx=24, ny=20, nz=12):
    dx = 8000.0
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=1e-4)
    x = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    y = (np.arange(ny) + 0.5) * dx - ny * dx / 2
    h = 4500.0 - 4050.0 * np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / 25e3**2)
    m = SigmaBaroclinic(hg, h, dt=600.0, n_layers=nz, rho0=RHO0, t_coef=0.2, z0=None, closure="constant",
                        nu_const=1e-4, pgf=pgf, rho_ref=rho_ref)
    z = -h[None] + h[None] * ((np.arange(nz) + 0.5) / nz)[:, None, None]
    return m, z


def test_jacobian_pgf_is_exact_for_a_linear_density_over_a_flat_bed():
    # ρ′ = a x + b z: exact PGF_x = −(g/ρ₀) a (η − z) = (g a/ρ₀) z at η = 0; the two-term form with a
    # constant top half-layer agrees to second order
    hg = create_cartesian_grid(nx=10, ny=6, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.full((6, 10), 30.0)
    nz = 8
    zc = -30.0 + 30.0 * (np.arange(nz) + 0.5) / nz
    xc = (np.arange(10) + 0.5) * 1000.0
    a, b = 1e-4, -0.02
    rho = RHO0 + a * xc[None, None, :] + b * zc[:, None, None] + 0 * np.ones((nz, 6, 10))
    T = (RHO0 - rho) / 0.2
    eta = np.zeros((6, 10))
    gx, _ = SigmaBaroclinic(hg, H, dt=10.0, n_layers=nz, t_coef=0.2, pgf="jacobian")._baroclinic_pgf(eta, T)
    exact = 9.81 * a / RHO0 * zc[:, None, None]
    np.testing.assert_allclose(gx[:, :, 1:-1], np.broadcast_to(exact, gx[:, :, 1:-1].shape), rtol=1e-10, atol=1e-16)
    g2, _ = SigmaBaroclinic(hg, H, dt=10.0, n_layers=nz, t_coef=0.2, pgf="twoterm")._baroclinic_pgf(eta, T)
    np.testing.assert_allclose(g2[:, :, 1:-1], gx[:, :, 1:-1], atol=1e-9)


def test_jacobian_pgf_is_exact_for_linear_stratification_over_a_seamount():
    m, z = _seamount_model("jacobian")
    T = 20.0 + 0.005 * z                                                  # ρ linear in z
    gx, gy = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13
    m2, _ = _seamount_model("twoterm")
    gx2, _ = m2._baroclinic_pgf(np.zeros(z.shape[1:]), T)
    assert np.abs(gx2).max() > 1e-6                                       # the two-term form is not


def test_jacobian_pgf_reduces_the_seamount_error_somewhat_for_exponential_stratification():
    errs = {}
    for pgf in ("twoterm", "jacobian"):
        m, z = _seamount_model(pgf)
        T = -15.0 * (1 - np.exp(z / 1000.0))                             # ρ = ρ₀ + 3 (1 − e^{z/1000})
        gx, gy = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
        errs[pgf] = max(np.abs(gx).max(), np.abs(gy).max())
    # measured: the 2nd-order Jacobian (Song 1998) removes only ~18 % of the error for exponential
    # stratification over this steep seamount (rx1 ≈ 10) — Marsaleix et al. 2009 warn that low-order
    # Jacobians are not automatically better; the reference profile does the heavy lifting
    assert errs["jacobian"] < 0.9 * errs["twoterm"]


def test_reference_profile_removes_the_error_of_a_resting_stratification():
    prof = lambda zz: RHO0 + 3.0 * (1 - np.exp(zz / 1000.0))             # noqa: E731
    m, z = _seamount_model("jacobian", rho_ref=prof)
    T = (RHO0 - prof(z)) / 0.2
    gx, gy = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13


def test_reference_profile_from_the_model_state_cuts_the_seamount_error():
    # ρ_ref(z) estimated from the horizontally averaged model density (no analytic profile)
    from core.dynamics.sigma_baroclinic import reference_profile
    m, z = _seamount_model("jacobian")
    T = -15.0 * (1 - np.exp(z / 1000.0))
    rho = m.density(T)
    gx0, _ = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
    m.rho_ref = reference_profile(z, rho)
    gx1, _ = m._baroclinic_pgf(np.zeros(z.shape[1:]), T)
    assert np.abs(gx1).max() < 0.01 * np.abs(gx0).max()


def test_auto_reference_profile_is_built_from_the_initial_state():
    m, z = _seamount_model("jacobian", rho_ref="auto")
    T = -15.0 * (1 - np.exp(z / 1000.0))
    st = m.initial_state(np.zeros(z.shape[1:]), T)
    assert callable(m.rho_ref)
    gx, gy = m._baroclinic_pgf(st.eta, st.T)
    m2, _ = _seamount_model("jacobian")
    gx2, _ = m2._baroclinic_pgf(st.eta, st.T)
    assert np.abs(gx).max() < 0.01 * np.abs(gx2).max()


# ---------------------------------------------------------------- Coriolis
def test_inertial_oscillation_has_period_two_pi_over_f():
    # uniform current in a large closed f-plane box, no stratification: far from the walls the
    # current turns at the inertial frequency, u = U cos(ft), v = −U sin(ft)
    # (2000 km box, 10 m deep: wall-generated gravity waves need ~28 h to reach the centre)
    f0, nx, dx, dt = 1e-4, 80, 25000.0, 300.0
    hg = create_cartesian_grid(nx=nx, ny=nx, dx=dx, dy=dx, f0=f0)
    m = SigmaBaroclinic(hg, np.full((nx, nx), 10.0), dt=dt, n_layers=2, z0=None, closure="none")
    st = m.initial_state(np.zeros((nx, nx)), np.full((2, nx, nx), 10.0))
    st.u[:, :, 1:-1] = 0.1
    T = 2 * np.pi / f0
    n_quarter = int(round(0.25 * T / dt))
    for _ in range(n_quarter):
        st = m.step(st)
    c = nx // 2
    assert abs(st.u[0, c, c]) < 0.01 and abs(st.v[0, c, c] + 0.1) < 0.01     # quarter period: u → −v
    for _ in range(n_quarter):
        st = m.step(st)
    assert abs(st.u[0, c, c] + 0.1) < 0.01                                     # half period: reversed


def test_coriolis_conserves_kinetic_energy_of_the_term():
    # the discrete Coriolis term does no work: Σ u·(f v̄) + Σ v·(−f ū) = 0 on a uniform grid
    rng = np.random.default_rng(5)
    hg = create_cartesian_grid(nx=12, ny=10, dx=1000.0, dy=1000.0, f0=1e-4)
    m = SigmaBaroclinic(hg, np.full((10, 12), 50.0), dt=60.0, n_layers=3)
    u = rng.normal(size=(3, 10, 13))
    v = rng.normal(size=(3, 11, 12))
    u[..., [0, -1]] = 0.0
    v[:, [0, -1], :] = 0.0
    cu, cv = m._coriolis(u, v)
    assert abs(np.sum(u * cu) + np.sum(v * cv)) < 1e-12 * np.sum(u**2)


# ---------------------------------------------------------------- wind stress
def test_uniform_wind_gives_the_inertial_ekman_transport():
    # far from the walls of a large f-plane box, a uniform eastward stress switched on at t = 0
    # gives the depth-integrated transport  Mx = (τ/ρ₀f) sin ft,  My = (τ/ρ₀f)(cos ft − 1):
    # on average τ/(ρ₀ f) to the right of the wind (Ekman 1905; Gill 1982 §9.2)
    f0, nx, dx, dt, depth, tau = 1e-4, 80, 25000.0, 300.0, 10.0, 0.1
    hg = create_cartesian_grid(nx=nx, ny=nx, dx=dx, dy=dx, f0=f0)
    m = SigmaBaroclinic(hg, np.full((nx, nx), depth), dt=dt, n_layers=5, rho0=RHO0, z0=None,
                        closure="constant", nu_const=1e-3)
    st = m.initial_state(np.zeros((nx, nx)), np.full((5, nx, nx), 10.0))
    tau_x = np.full((nx, nx + 1), tau)
    scale = tau / (RHO0 * f0)                                           # 1 m² s⁻¹
    c = nx // 2
    err = 0.0
    for _ in range(int(round(2 * np.pi / f0 / dt))):
        st = m.step(st, tau_x=tau_x)
        D = depth + st.eta[c, c]
        Mx = D / 5 * st.u[:, c, c].sum()
        My = D / 5 * st.v[:, c, c].sum()
        ft = f0 * st.time
        err = max(err, abs(Mx - scale * np.sin(ft)), abs(My - scale * (np.cos(ft) - 1)))
    assert err < 0.005 * scale                                         # measured 0.045 %
    assert st.u[-1, c, c] != st.u[0, c, c]                              # stress enters at the top


# ---------------------------------------------------------------- open boundaries
N_BV, H_CH = 0.02, 200.0


def _channel(nx, x0, dx=1000.0, nz=10, dt=120.0, **kw):
    """Linear stratification (N = 0.02 s⁻¹, c₁ = NH/π = 1.27 m/s) with a mode-1 isopycnal bump at x0."""
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, nx), H_CH), dt=dt, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-5, **kw)
    x = (np.arange(nx) + 0.5) * dx
    z = -H_CH + H_CH * ((np.arange(nz) + 0.5) / nz)[:, None, None]
    dTdz = N_BV**2 * RHO0 / (9.81 * TCOEF)
    zeta = 5.0 * np.sin(-np.pi * z / H_CH) * np.exp(-((x - x0) / 5e3) ** 2)[None, None, :]
    T = 10.0 + dTdz * (z + zeta)
    return m, m.initial_state(np.zeros((1, nx)), T)


def _run(m, st, hours):
    for _ in range(int(round(hours * 3600 / m.dt))):
        st = m.step(st)
    return st


def test_open_boundaries_keep_a_uniform_tracer_uniform():
    # Flather faces carry volume; the layer transports there must add up to exactly the free-surface
    # flux and the inflowing tracer must be the boundary value, or a uniform tracer would change
    hg = create_cartesian_grid(nx=30, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, 30), 20.0), dt=30.0, n_layers=6, z0=None, closure="constant",
                        nu_const=1e-4, open_boundaries=("west", "east"))
    x = (np.arange(30) + 0.5) * 1000.0
    st = m.initial_state(0.3 * np.exp(-((x - 12e3) / 3e3) ** 2)[None, :], np.full((6, 1, 30), 7.0))
    vol0 = m.layer_volumes(st.eta).sum()
    st = _run(m, st, 1.0)
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert m.layer_volumes(st.eta).sum() < 0.9 * vol0 + 0.1 * (20.0 * 30e3 * 1e3)   # the bump left
    assert np.abs(st.u[:, 0, [0, -1]]).max() > 0                         # boundary faces carry flow


def _interior_error(open_kw, nx_small=60, pad=0):
    """rms interior T difference from a 300 km 'truth' channel, relative to the initial anomaly."""
    big, sb = _channel(300, 150e3)
    small, ss = _channel(nx_small + 2 * pad, (nx_small / 2 + pad) * 1e3, **open_kw)
    T_init = ss.T[..., pad:pad + nx_small].copy()
    sb, ss = _run(big, sb, 16.0), _run(small, ss, 16.0)
    sl = slice(150 - nx_small // 2, 150 + nx_small // 2)
    diff = ss.T[..., pad:pad + nx_small] - sb.T[..., sl]
    bg = T_init.mean(axis=-1, keepdims=True)
    return np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean((T_init - bg) ** 2))


def test_closed_channel_reflects_the_internal_wave():
    assert _interior_error({}) > 0.3                                     # the reference the OBCs must beat


def test_orlanski_radiation_lets_most_of_the_internal_wave_out():
    # Flather for the depth mean + Orlanski (1976) radiation of the baroclinic normal velocity.
    # Measured 0.157 (closed 0.82): a single phase speed per layer cannot fit the mode-1 + mode-2
    # mixture of the pulse — the known weakness of radiation-only OBCs for stratified flow
    # (Jensen 1998; Palma & Matano 1998); hence the sponge below
    assert _interior_error(dict(open_boundaries=("west", "east"))) < 0.2


def test_sponge_with_radiation_absorbs_the_internal_wave():
    # flow-relaxation sponge (Martinsen & Engedahl 1987; MITgcm OBCS profile) on the baroclinic
    # velocity and the tracer anomaly, outside the 60 km interior
    # measured 0.015 (Orlanski alone in the same padded domain 0.133): sponge + radiation is the most
    # transparent combination (Chapman 1985), as in the MITgcm BayIW runs
    kw = dict(open_boundaries=("west", "east"), sponge_cells=10, sponge_tau=(1200.0, 120.0))
    assert _interior_error(kw, pad=10) < 0.05
