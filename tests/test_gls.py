"""
Tests for physics/mixing/gls.py — generic length scale k-ε closure with the
Canuto et al. (2001) A stability functions, as in CROCO gls_mixing.F.
"""

from __future__ import annotations

import numpy as np

from physics.mixing.gls import GLSKEpsilon, KAPPA


def test_neutral_constants_match_canuto_a():
    g = GLSKEpsilon(n_layers=10, n_columns=1)
    # Umlauf & Burchard (2005), Table 3: Canuto A, c_μ⁰ = 0.5270
    assert abs(g.cm0 - 0.527) < 2e-3


def test_turbulence_decays_without_shear_and_stays_positive():
    g = GLSKEpsilon(n_layers=10, n_columns=3)
    g.tke[:] = 1e-3
    g.eps[:] = 1e-6
    dz = np.full((10, 3), 1.0)
    u = np.zeros((10, 3))
    k0 = g.tke[1:-1].copy()
    for _ in range(50):
        g.step(u, np.zeros_like(u), dz, ustar2_bot=np.zeros(3), dt=10.0)
    assert np.all(g.tke[1:-1] < k0)
    assert np.all(g.tke >= g.tke_min) and np.all(g.eps > 0) and np.all(g.akv >= g.nu_min)


def test_log_layer_equilibrium_in_steady_channel_flow():
    """
    Steady open-channel flow, body force g S, log-law bed stress: near the bed the
    closure must give the equilibrium log layer, ν ≈ κ u* z and k ≈ u*²/c_μ⁰².
    """
    from physics.bottom_friction import first_layer_cd
    from core.dynamics.sigma_wd import vertical_implicit

    n, D, S, z0, dt = 40, 10.0, 1e-5, 1e-3, 30.0
    g = GLSKEpsilon(n_layers=n, n_columns=1)
    dz = np.full((n, 1), D / n)
    u = np.zeros((n, 1))
    for _ in range(20000):                                       # ~7 days
        cd = first_layer_cd(dz[0], z0)
        u = vertical_implicit(u + dt * 9.81 * S, dz, g.akv, cd * np.abs(u[0]), dt)
        g.step(u, np.zeros_like(u), dz, ustar2_bot=cd * u[0] ** 2, dt=dt)
    ustar = np.sqrt(9.81 * S * D)                                # momentum balance τ_b = ρ g S D
    assert np.isclose(np.sqrt(first_layer_cd(dz[0], z0)) * abs(u[0, 0]), ustar, rtol=0.02)
    z = np.arange(n + 1) * D / n
    near = slice(2, 8)                                           # 0.5–1.75 m above the bed
    np.testing.assert_allclose(g.akv[near, 0], KAPPA * ustar * z[near] * (1 - z[near] / D), rtol=0.25)
    np.testing.assert_allclose(g.tke[near, 0], ustar**2 / g.cm0**2, rtol=0.25)
