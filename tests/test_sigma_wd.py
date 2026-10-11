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
