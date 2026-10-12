"""
Tests for core/vertical/pgf.py — finite-volume pressure-gradient force (Adcroft, Hallberg &
Harrison 2008; MOM6 MOM_PressureForce_FV / MOM_density_integrals) with a linear EOS.
"""

from __future__ import annotations

import numpy as np

from core.vertical import pgf

RHO0, ALPHA, T0, G = 1000.0, 0.2, 0.0, 9.81


def _layers(H, eta, n, rng=None):
    """Interfaces bed first: uniform, or random monotone (tilted, some 1 mm layers) with rng."""
    ny, nx = H.shape
    if rng is None:
        frac = np.broadcast_to(np.linspace(0, 1, n + 1)[:, None, None], (n + 1, ny, nx))
    else:
        f = rng.uniform(0.0, 1.0, (n, ny, nx))
        f[rng.uniform(size=f.shape) < 0.2] = 1e-5                       # some vanished layers
        frac = np.concatenate([np.zeros((1, ny, nx)), np.cumsum(f, 0) / f.sum(0)], 0)
    D = H + eta
    zw = -H[None] + D[None] * frac
    return np.diff(zw, axis=0), 0.5 * (zw[1:] + zw[:-1])


def _call(h, H, eta, T, dx=1000.0, ref=None, mass_weight=True):
    ny, nx = H.shape
    du, dv = np.full((ny, nx + 1), dx), np.full((ny + 1, nx), dx)
    return pgf.fv_pgf(h, H, eta, T, du, dv, RHO0, ALPHA, T0, ref, G, mass_weight=mass_weight)


def test_plm_edges_are_exact_for_a_linear_profile_on_an_irregular_column():
    h = np.array([3.0, 1.0, 1e-3, 7.0, 2.0])
    zc = np.cumsum(h) - 0.5 * h
    T = 4.0 + 0.3 * zc
    tb, tt = np.empty(5), np.empty(5)
    pgf.pressure_edges_plm(h, T, tb, tt)
    np.testing.assert_allclose(tb, 4.0 + 0.3 * (zc - 0.5 * h), atol=1e-12)
    np.testing.assert_allclose(tt, 4.0 + 0.3 * (zc + 0.5 * h), atol=1e-12)


def test_no_force_for_horizontally_uniform_linear_stratification_on_tilted_layers():
    rng = np.random.default_rng(3)
    ny, nx, n = 4, 7, 12
    H = rng.uniform(50.0, 400.0, (ny, nx))
    eta = np.zeros((ny, nx))
    h, zc = _layers(H, eta, n, rng)
    T = 20.0 + 0.02 * zc                                                 # ρ linear in z
    gx, gy = _call(h, H, eta, T, mass_weight=False)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13
    # mass weighting acts only next to vanished layers: exact on steep σ-layers without them
    h, zc = _layers(H, eta, n)
    gx, gy = _call(h, H, eta, 20.0 + 0.02 * zc)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13


def test_exact_for_density_linear_in_x_and_z_over_a_flat_bed():
    # ρ′ = a x + b z: PGF_x = −(g/ρ₀) ∂/∂x ∫_z^0 ρ′ dz′ = (g a/ρ₀) z, averaged over each layer
    ny, nx, n = 3, 8, 6
    H = np.full((ny, nx), 30.0)
    eta = np.zeros((ny, nx))
    h, zc = _layers(H, eta, n)
    xc = (np.arange(nx) + 0.5) * 1000.0
    a, b = 1e-4, -0.02
    rho = RHO0 + a * xc[None, None, :] + b * zc
    gx, gy = _call(h, H, eta, (RHO0 - rho) / ALPHA)
    np.testing.assert_allclose(gx[:, :, 1:-1], np.broadcast_to(G * a / RHO0 * zc[:, :, :1], gx[:, :, 1:-1].shape),
                               rtol=1e-10, atol=1e-16)
    assert np.abs(gy).max() < 1e-15


def test_reference_profile_equal_to_the_water_gives_no_force():
    ny, nx, n = 2, 5, 8
    H = np.array([[100.0, 160.0, 220.0, 300.0, 380.0]] * ny)
    eta = np.zeros((ny, nx))
    h, zc = _layers(H, eta, n)
    rho_of = lambda z: RHO0 + 3.0 * (1 - np.exp(z / 100.0))           # noqa: E731
    T = (RHO0 - rho_of(zc)) / ALPHA
    gx0, _ = _call(h, H, eta, T)                                         # without the profile: σ-type error
    gx1, _ = _call(h, H, eta, T, ref=rho_of)
    assert np.abs(gx1).max() < 1e-3 * np.abs(gx0).max()


def test_mass_weighting_hides_the_value_of_a_layer_vanished_below_the_neighbour_bed():
    # z*-like step: left column 100 m with its two deepest layers vanished at the bed; right
    # column 300 m.  Right layer 1 (−200 … −110 m) lies wholly below the left bed, so the force on
    # it must not depend on the (meaningless) T of the vanished left layer
    n = 4
    H = np.array([[100.0, 300.0]])
    eta = np.zeros((1, 2))
    h = np.array([[1e-3, 1e-3, 50.0 - 2e-3, 50.0], [100.0, 90.0, 60.0, 50.0]]).T[:, None, :]
    T = np.full((n, 1, 2), 10.0)

    def force_change(mw):
        T2 = T.copy()
        T2[1, 0, 0] = 25.0                                               # garbage in the vanished layer
        return abs(_call(h, H, eta, T2, mass_weight=mw)[0][1, 0, 1] - _call(h, H, eta, T, mass_weight=mw)[0][1, 0, 1])

    assert force_change(False) > 0
    assert force_change(True) < 1e-3 * force_change(False)
