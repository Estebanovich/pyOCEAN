"""
Tests for diagnostics/mixing.py — spurious-mixing diagnostics:
reference potential energy (Winters et al. 1995; Ilicak et al. 2012) and the
discrete variance decay of an advection step (Burchard & Rennau 2008).
"""

from __future__ import annotations

import numpy as np
import pytest

from diagnostics.mixing import numerical_variance_decay, reference_potential_energy

G = 9.81


def test_rpe_of_a_stable_two_layer_state_is_its_potential_energy():
    # already sorted: RPE = PE = g Σ ρ z V (z = height above the bottom)
    area, h = 100.0, 10.0
    rho = np.array([1001.0, 1000.0])                       # bottom, top layer, 5 m each
    vol = np.array([5.0, 5.0]) * area
    expected = G * area * (1001.0 * 2.5 * 5.0 + 1000.0 * 7.5 * 5.0)
    assert np.isclose(reference_potential_energy(rho, vol, area), expected, rtol=1e-12)


def test_rpe_is_independent_of_where_the_water_is():
    # a lock exchange at t = 0 (dense left / light right) has the RPE of the sorted two-layer state
    nx, nz = 8, 4
    rho = np.where(np.arange(nx) < nx // 2, 1001.0, 1000.0)[None, :].repeat(nz, 0)
    vol = np.ones((nz, nx))
    sorted_rho = np.sort(rho.ravel())[::-1]                 # heaviest first (bottom)
    assert np.isclose(reference_potential_energy(rho, vol, area=nx),
                      reference_potential_energy(sorted_rho, np.ones(nx * nz), area=nx))


def test_rpe_increases_under_mixing_and_not_under_stirring():
    rng = np.random.default_rng(0)
    rho = rng.uniform(1000, 1002, size=200)
    vol = rng.uniform(0.5, 1.5, size=200)
    r0 = reference_potential_energy(rho, vol, area=3.0)
    perm = rng.permutation(200)
    assert np.isclose(reference_potential_energy(rho[perm], vol[perm], area=3.0), r0, rtol=1e-13)   # stirring
    mixed = np.full(200, np.sum(rho * vol) / vol.sum())                                              # full mixing
    assert reference_potential_energy(mixed, vol, area=3.0) > r0


def test_rpe_with_hypsometry_matches_flat_case():
    # area constant with height → same as the flat formula
    rho = np.array([1002.0, 1001.0, 1000.0])
    vol = np.array([2.0, 3.0, 5.0])
    flat = reference_potential_energy(rho, vol, area=2.0)
    z_levels = np.linspace(0.0, 10.0, 101)
    hyp = reference_potential_energy(rho, vol, hypsometry=(z_levels, np.full(101, 2.0)))
    assert np.isclose(hyp, flat, rtol=1e-6)


def _upwind_1d(c: np.ndarray, courant: float) -> np.ndarray:
    """First-order upwind, periodic, u > 0."""
    return c - courant * (c - np.roll(c, 1))


def test_variance_decay_sums_to_the_global_loss_of_variance():
    # Burchard & Rennau (2008): Σ M_num Δt V = Σ c² V − Σ (A c)² V for a conservative scheme
    x = np.linspace(0, 1, 100, endpoint=False)
    c = np.exp(-((x - 0.5) / 0.05) ** 2)
    dt, C = 0.1, 0.4
    m = numerical_variance_decay(lambda f: _upwind_1d(f, C), c, dt)
    lost = (np.sum(c**2) - np.sum(_upwind_1d(c, C) ** 2)) / dt
    assert np.isclose(m.sum(), lost, rtol=1e-12)
    assert lost > 0


def test_variance_decay_vanishes_for_an_exact_shift():
    c = np.random.default_rng(2).normal(size=64)
    m = numerical_variance_decay(lambda f: _upwind_1d(f, 1.0), c, 1.0)    # Courant 1: exact translation
    np.testing.assert_allclose(m, 0.0, atol=1e-12)


def test_variance_decay_matches_upwind_numerical_diffusivity():
    # first-order upwind: κ_num = u Δx (1 − C)/2  →  M_num ≈ 2 κ_num (∂c/∂x)² (Burchard 2012)
    n, u = 2000, 1.0
    dx = 1.0 / n
    C = 0.25
    dt = C * dx / u
    x = (np.arange(n) + 0.5) * dx
    c = np.sin(2 * np.pi * x)
    m = numerical_variance_decay(lambda f: _upwind_1d(f, C), c, dt)
    kappa = u * dx * (1 - C) / 2
    expected = 2 * kappa * (2 * np.pi) ** 2 * np.mean(np.cos(2 * np.pi * x) ** 2)
    assert np.isclose(m.mean(), expected, rtol=1e-2)


def test_rejects_ambiguous_geometry():
    with pytest.raises(ValueError):
        reference_potential_energy(np.ones(3), np.ones(3))
