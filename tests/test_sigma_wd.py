"""
Tests for the homogeneous 3-D (σ-layer) wetting–drying model:
core/dynamics/sigma_wd.py, physics/mixing/parabolic.py and the first-layer
log-law drag in physics/bottom_friction.py.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics.sigma_wd import SigmaWetDry, vertical_implicit
from core.dynamics.wetting_drying import WetDryFreeSurface
from core.grid.horizontal import create_cartesian_grid
from physics.bottom_friction import KAPPA, first_layer_cd, log_law_cd
from physics.mixing.parabolic import parabolic_viscosity

G = 9.81


# ---------------------------------------------------------------- building blocks
def test_first_layer_cd_is_the_law_of_the_wall_at_mid_layer():
    dz, z0 = 1.0, 1e-4
    assert np.isclose(first_layer_cd(dz, z0), (KAPPA / np.log(0.5 * dz / z0)) ** 2)
    assert first_layer_cd(1e-4, 1e-4, cd_max=0.1) == 0.1


def test_parabolic_viscosity_profile():
    # ν(z) = κ u* z (1 − z/D) at the layer interfaces (0 at bed and surface → floor ν_min)
    D, ustar = 10.0, 0.05
    sig_w = np.linspace(0, 1, 11)[:, None]
    nu = parabolic_viscosity(np.array([ustar]), np.array([D]), sig_w, nu_min=1e-6)
    z = sig_w[:, 0] * D
    np.testing.assert_allclose(nu[1:-1, 0], KAPPA * ustar * z[1:-1] * (1 - z[1:-1] / D))
    assert nu[0, 0] == nu[-1, 0] == 1e-6


def _dense_solve(u, dz, nu, r, dt):
    n = u.size
    A = np.zeros((n, n))
    for k in range(n):
        A[k, k] = dz[k]
        if k > 0:
            c = dt * nu[k] / (0.5 * (dz[k] + dz[k - 1]))
            A[k, k] += c
            A[k, k - 1] -= c
        if k < n - 1:
            c = dt * nu[k + 1] / (0.5 * (dz[k] + dz[k + 1]))
            A[k, k] += c
            A[k, k + 1] -= c
    A[0, 0] += dt * r
    return np.linalg.solve(A, dz * u)


@pytest.mark.parametrize("closure,z0", [("constant", None), ("none", None), ("parabolic", 1e-3)])
def test_surface_stress_puts_its_momentum_into_the_column(closure, z0):
    # kinematic surface stress τ/ρ₀ enters as the top boundary flux of the implicit viscous solve:
    # without bed stress Σ Δz_k u_k gains exactly Δt τ/ρ₀; with bed stress somewhat less
    hg = create_cartesian_grid(nx=4, ny=3, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaWetDry(hg, np.full((3, 4), 20.0), dt=60.0, n_layers=6, z0=z0, closure=closure, nu_const=1e-3)
    u = np.zeros((6, 3, 5))
    tau = np.full((3, 5), 1e-4)                                         # 0.1 N m⁻² / 1000 kg m⁻³
    out = m.column_step(u, None, np.full((3, 5), 20.0), tau=tau)
    gained = (out * 20.0 / 6).sum(0)
    if z0 is None:
        np.testing.assert_allclose(gained, 60.0 * 1e-4, rtol=1e-12)
        assert out[-1, 0, 0] > out[0, 0, 0] if closure == "constant" else out[-1, 0, 0] > 0
    else:
        assert np.all(gained <= 60.0 * 1e-4 * (1 + 1e-12)) and np.all(gained > 0)


def test_vertical_implicit_matches_dense_solve():
    rng = np.random.default_rng(0)
    n, m = 8, 5
    u = rng.normal(size=(n, m))
    dz = rng.uniform(0.2, 1.0, size=(n, m))
    nu = rng.uniform(1e-3, 1e-1, size=(n + 1, m))
    r = rng.uniform(0, 1e-2, size=m)
    out = vertical_implicit(u, dz, nu, r, dt=30.0)
    for j in range(m):
        np.testing.assert_allclose(out[:, j], _dense_solve(u[:, j], dz[:, j], nu[:, j], r[j], 30.0), rtol=1e-10)


def test_vertical_implicit_conserves_momentum_without_bed_stress():
    rng = np.random.default_rng(1)
    u = rng.normal(size=(10, 3))
    dz = np.full((10, 3), 0.5)
    out = vertical_implicit(u, dz, np.full((11, 3), 0.05), np.zeros(3), dt=100.0)
    np.testing.assert_allclose((dz * out).sum(0), (dz * u).sum(0), rtol=1e-12)
    assert out.std(0).max() < u.std(0).max()                     # and it mixes


# ---------------------------------------------------------------- full model
def _channel(nx=40, depth=5.0, n_layers=20, z0=1e-3, dt=60.0, closure="parabolic"):
    hg = create_cartesian_grid(nx=nx, ny=1, dx=500.0, dy=500.0, f0=0.0)
    H = np.full((1, nx), depth)
    return hg, SigmaWetDry(hg, H, dt=dt, n_layers=n_layers, z0=z0, theta=0.6, d_min=0.05, closure=closure)


def test_uniform_flow_reaches_the_2d_log_law_chezy_velocity():
    """
    Open-channel flow driven by a uniform body force g S over a flat bed, closed
    channel replaced by a periodic-like interior column: the steady depth-mean
    velocity of the 3-D model must match the 2-D log-law C_d, ū = √(g S D / C_d(D)).
    """
    depth, S, z0 = 5.0, 1e-5, 1e-3
    hg, m = _channel(nx=4, depth=depth, z0=z0)
    u = np.zeros((m.n_layers, 1, 5))
    for _ in range(4000):                                        # ~3 days, steady
        u = m.column_step(u, np.full(u.shape, G * S), depth=np.full((1, 5), depth))
    ubar = (m.dsigma[:, None, None] * u).sum(0)[0, 2]
    expected = np.sqrt(G * S * depth / log_law_cd(depth, z0))
    assert abs(ubar / expected - 1) < 0.05                      # 0.96–0.99 for 5–40 layers, D = 1–15 m
    prof = u[:, 0, 2]
    assert np.all(np.diff(prof) > 0)                              # monotonic, fastest at the surface


def test_frictionless_layers_reproduce_the_2d_model():
    # Thacker-like sloshing in a flat channel, no friction, no viscosity: σ-layers stay uniform
    nx = 40
    hg = create_cartesian_grid(nx=nx, ny=1, dx=500.0, dy=500.0, f0=0.0)
    H = np.full((1, nx), 5.0)
    x = (np.arange(nx) + 0.5) * 500.0
    eta0 = 0.2 * np.cos(np.pi * x / (nx * 500.0))[None, :]
    m3 = SigmaWetDry(hg, H, dt=30.0, n_layers=6, z0=None, theta=0.6, d_min=0.05, closure="none")
    m2 = WetDryFreeSurface(hg, H, dt=30.0, theta=0.6, d_min=0.05)
    e3, u3, v3 = eta0.copy(), np.zeros((6, 1, nx + 1)), np.zeros((6, 2, nx))
    e2, u2, v2 = eta0.copy(), np.zeros((1, nx + 1)), np.zeros((2, nx))
    for _ in range(200):
        e3, u3, v3 = m3.step(e3, u3, v3)
        e2, u2, v2 = m2.step(e2, u2, v2, u2.copy(), v2.copy())
    np.testing.assert_allclose(e3, e2, atol=1e-12)
    np.testing.assert_allclose(u3, np.broadcast_to(u2, u3.shape), atol=1e-12)


def test_rejects_unknown_closure():
    with pytest.raises(ValueError):
        _channel(closure="kpp")


def test_gls_closure_runs_through_the_full_step_and_stays_positive():
    # tidal-like sloshing in a flat channel with the k-ε closure: finite, positive turbulence
    nx = 30
    hg = create_cartesian_grid(nx=nx, ny=1, dx=500.0, dy=500.0, f0=0.0)
    H = np.full((1, nx), 4.0)
    x = (np.arange(nx) + 0.5) * 500.0
    m = SigmaWetDry(hg, H, dt=30.0, n_layers=8, z0=1e-4, closure="gls", theta=0.6, d_min=0.05)
    eta = 0.3 * np.cos(np.pi * x / (nx * 500.0))[None, :]
    u, v = np.zeros((8, 1, nx + 1)), np.zeros((8, 2, nx))
    vol0 = eta.sum()
    for _ in range(300):
        eta, u, v = m.step(eta, u, v)
    assert np.isfinite(eta).all() and np.isfinite(u).all()
    assert abs(eta.sum() - vol0) < 1e-10
    assert np.all(m.gls.tke > 0) and m.gls.akv.max() > 10 * m.gls.nu_min     # turbulence was generated


def test_compiled_vertical_solver_matches_the_numpy_reference():
    from core.dynamics.sigma_wd import _vertical_implicit_numpy
    rng = np.random.default_rng(11)
    n, m = 12, 57
    u = rng.normal(size=(n, m))
    dz = rng.uniform(0.2, 3.0, size=(n, m))
    nu = rng.uniform(1e-5, 1e-1, size=(n + 1, m))
    r = rng.uniform(0, 1e-2, size=m)
    tau = rng.normal(0, 1e-4, size=m)
    np.testing.assert_allclose(vertical_implicit(u, dz, nu, r, 60.0, tau_top=tau),
                               _vertical_implicit_numpy(u, dz, nu, r, 60.0, tau_top=tau), rtol=1e-13, atol=1e-15)


def _wd_basin():
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=12, ny=10, dx=1000.0, dy=1000.0, f0=1e-4)
    m = SigmaWetDry(hg, np.full((10, 12), 20.0), dt=30.0, n_layers=4, z0=None, closure="constant", nu_const=1e-3)
    x = (np.arange(12) + 0.5) * 1000.0
    y = (np.arange(10) + 0.5) * 1000.0
    eta = 0.3 * np.exp(-((x[None] - 6e3) ** 2 + (y[:, None] - 5e3) ** 2) / 3e3**2)
    rng = np.random.default_rng(1)
    u = 0.05 * rng.standard_normal((4, 10, 13))
    v = 0.05 * rng.standard_normal((4, 11, 12))
    u[..., [0, -1]] = 0.0
    v[:, [0, -1]] = 0.0
    return m, eta, u, v


def test_explicit_sigma_geometry_reproduces_the_default_step_bit_for_bit():
    m, eta, u, v = _wd_basin()
    e0, u0, v0 = m.step(eta, u, v)
    Du, Dv = m.face_depths(eta)
    ds = m.dsigma[:, None, None]
    flo = lambda D: np.maximum(np.nan_to_num(D, nan=m.d_min), m.d_min)                 # noqa: E731
    e1, u1, v1 = m.step(eta, u, v, wu=np.broadcast_to(ds, u.shape), wv=np.broadcast_to(ds, v.shape),
                        hu=flo(Du)[None] * ds, hv=flo(Dv)[None] * ds)
    assert np.array_equal(e0, e1) and np.array_equal(u0, u1) and np.array_equal(v0, v1)


def test_layer_transports_with_face_fractions_add_up_to_the_free_surface_flux():
    m, eta, u, v = _wd_basin()
    w = np.array([0.05, 0.15, 0.3, 0.5])[:, None, None]                 # not the σ fractions
    wu, wv = np.broadcast_to(w, u.shape), np.broadcast_to(w, v.shape)
    Du, Dv = m.face_depths(eta)
    e1, u1, v1 = m.step(eta, u, v, wu=wu, wv=wv, hu=np.nan_to_num(Du)[None] * w, hv=np.nan_to_num(Dv)[None] * w)
    th, fs = m.fs.theta, m.fs
    un, vn = np.where(fs.last_wet_u, u, 0.0), np.where(fs.last_wet_v, v, 0.0)
    Qx = (fs.Lu * fs.last_Du)[None] * wu * (th * u1 + (1 - th) * un)
    Qy = (fs.Lv * fs.last_Dv)[None] * wv * (th * v1 + (1 - th) * vn)
    div = (Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:] - Qy[:, :-1]).sum(0)
    np.testing.assert_allclose(fs.area * (e1 - eta), -m.dt * div, atol=1e-9 * fs.area.max() * 0.3)


def test_viscous_coupling_keeps_the_bed_drag_thickness():
    # the log-law bed drag C_d(dz_0) must use dz_drag when given, not the (thin) viscous thickness
    from core.dynamics.sigma_wd import SigmaWetDry
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=3, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaWetDry(hg, np.full((1, 3), 20.0), dt=10.0, n_layers=3, z0=1e-3, closure="constant", nu_const=1e-4)
    u = np.ones((3, 1, 4))
    dz_visc = np.full((3, 1, 4), 6.0); dz_visc[0] = 1e-3          # thin viscous bed thickness
    dz_mean = np.full((3, 1, 4), 6.0)
    a = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_visc, dz_drag=dz_mean[0])
    b = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_visc)
    assert not np.allclose(a, b)                                   # dz_drag is used
    c = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_mean, dz_drag=dz_mean[0])
    d = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_mean)
    np.testing.assert_array_equal(c, d)                            # same as before when dz_drag = dz[0]


def _coupled_columns(z0):
    from core.dynamics.sigma_wd import SigmaWetDry
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=3, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaWetDry(hg, np.full((1, 3), 50.0), dt=60.0, n_layers=3, z0=z0, closure="constant", nu_const=1e-3)
    dz_mean = np.full((3, 1, 4), 50.0 / 3); dz_mean[-1] = 25.0; dz_mean[0] = 50.0 / 3 - 25.0 + 50.0 / 3
    dz_visc = dz_mean.copy(); dz_visc[-1] = 2e-3                   # top layer: 1 mm on one side, 50 m on the other
    w = dz_mean / dz_mean.sum(0)
    D = np.full((1, 4), dz_mean.sum(0)[0, 0])
    return m, dz_mean, dz_visc, w, D


@pytest.mark.parametrize("z0, tau", [(None, 1e-4), (None, 0.0), (1e-3, 1e-4)])
def test_coupled_viscous_step_keeps_the_depth_mean_physics(z0, tau):
    # with the thin-layer coupling thickness the implicit solve conserves Σ dz_visc u, not the depth mean Σ w u of
    # the free surface: internal friction must not change ū, and the wind must change it by Δt τ/D (minus the bed
    # stress), not by Δt τ/dz_visc_top (final review 2026-10-14: 0.25 m/s instead of 2e-5 per step)
    from physics.bottom_friction import first_layer_cd
    m, dz_mean, dz_visc, w, D = _coupled_columns(z0)
    u = np.zeros((3, 1, 4)); u[-1] = 0.5; u[0] = 0.1                # sheared column
    taus = np.full((1, 4), tau)
    out = m.column_step(u, None, D, tau=taus, dz=dz_visc, dz_drag=dz_mean[0], dz_mean=dz_mean)
    r = 0.0 if z0 is None else first_layer_cd(dz_mean[0, 0, 0], z0, cd_max=m.cd_max) * abs(u[0, 0, 0])
    expect = (w * u).sum(0) + m.dt * (tau - r * out[0]) / D
    np.testing.assert_allclose((w * out).sum(0), expect, rtol=1e-12, atol=1e-15)
    if tau == 0.0:                                                  # the thin layer is still tied to its neighbour
        assert abs(out[-1, 0, 0] - out[1, 0, 0]) < abs(u[-1, 0, 0] - u[1, 0, 0])


def test_column_step_with_a_given_bed_rate_matches_the_implicit_formula():
    from core.dynamics.sigma_wd import SigmaWetDry
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=3, ny=1, dx=100.0, dy=100.0)
    m = SigmaWetDry(hg, np.full((1, 3), 10.0), dt=10.0, n_layers=1, z0=None, closure="none")
    u = np.full((1, 4), 1.0)
    out = m.column_step(u, None, np.full(4, 10.0), r_bed=np.full(4, 0.01))
    np.testing.assert_allclose(out, 1.0 / (1.0 + 10.0 * 0.01 / 10.0), rtol=1e-14)   # (1 + Δt r / h)⁻¹
