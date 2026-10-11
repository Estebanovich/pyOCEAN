"""
Tests for physics/bottom_friction.py — depth-dependent drag coefficient of a
depth-averaged (2-D) model from a logarithmic bottom boundary layer.
"""

from __future__ import annotations

import numpy as np

from physics.bottom_friction import KAPPA, log_law_cd


def test_matches_soulsby_formula():
    # Soulsby (1997) eq. 37 with the exact mean over the full depth
    D, z0 = 10.0, 1e-3
    expected = (KAPPA / (np.log(D / z0) - 1 + z0 / D)) ** 2
    assert np.isclose(log_law_cd(D, z0), expected, rtol=1e-12)


def test_reproduces_stress_of_the_log_profile():
    # u(z) = (u*/κ) ln(z/z0) on z0 < z < D, u = 0 below: τ/ρ = u*² must equal C_d ū²
    D, z0, ustar = 3.0, 1e-4, 0.05
    z = np.linspace(z0, D, 2_000_001)
    u = ustar / KAPPA * np.log(z / z0)
    ubar = np.trapz(u, z) / D
    assert np.isclose(log_law_cd(D, z0) * ubar**2, ustar**2, rtol=1e-6)


def test_decreases_with_depth():
    D = np.array([0.2, 0.5, 1.0, 2.0, 5.0, 20.0])
    cd = log_law_cd(D, 1e-4)
    assert np.all(np.diff(cd) < 0)
    # CROCO TIDAL_FLAT 3D (z0 = 1e-4) gives C_d,eff ≈ 2.8e-3 at 0.2–0.5 m, 1.3e-3 at 12–20 m
    assert 2.5e-3 < log_law_cd(0.35, 1e-4) < 3.5e-3
    assert 1.2e-3 < log_law_cd(15.0, 1e-4) < 1.6e-3


def test_clipped_in_very_shallow_and_dry_water():
    D = np.array([0.0, 1e-4, 2.7e-4, np.nan, -1.0, 100.0])
    cd = log_law_cd(D, 1e-4, cd_min=1.1e-3, cd_max=0.1)
    assert np.all(np.isfinite(cd))
    np.testing.assert_allclose(cd[:5], 0.1)
    assert cd[5] == 1.1e-3                        # deep water (1.02e-3) hits the floor
