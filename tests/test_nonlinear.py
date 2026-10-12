"""
Tests for core.dynamics.nonlinear — nonlinear shallow-water terms on the C grid.

    ∂η/∂t = −∇·((H + η) u)  →  linear part −∇·(H u) (implicit, ImplicitFreeSurface)
                               + nonlinear part −∇·(η u) (explicit, here)
    ∂u/∂t = … − (u·∇)u + A_h ∇²u ;   bottom drag uses the total depth H + η
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics.free_surface import ImplicitFreeSurface
from core.dynamics.nonlinear import (
    advection_u,
    advection_v,
    cell_to_faces,
    laplacian_u,
    laplacian_v,
    nonlinear_continuity,
)
from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import compute_hfac, create_z_grid

NX, NY, DX, DY = 12, 9, 1000.0, 2000.0


def _grid():
    hg = create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DY, f0=0.0)
    x_u = np.arange(NX + 1)[None, :] * DX * np.ones((NY, 1))            # u-face positions
    y_u = (np.arange(NY)[:, None] + 0.5) * DY * np.ones((1, NX + 1))
    x_v = (np.arange(NX)[None, :] + 0.5) * DX * np.ones((NY + 1, 1))
    y_v = np.arange(NY + 1)[:, None] * DY * np.ones((1, NX))
    return hg, x_u, y_u, x_v, y_v


def _wet():
    wet_t = np.ones((NY, NX), bool)
    wet_u = np.ones((NY, NX + 1), bool)
    wet_u[:, [0, -1]] = False                                             # closed walls
    wet_v = np.ones((NY + 1, NX), bool)
    wet_v[[0, -1], :] = False
    return wet_t, wet_u, wet_v


class TestFaces:

    def test_mean_of_adjacent_wet_cells(self):
        f = np.array([[1.0, 3.0, 0.0], [2.0, 4.0, 6.0]])
        wet = np.array([[True, True, False], [True, True, True]])
        fu, fv = cell_to_faces(f, wet)
        assert fu[0, 1] == pytest.approx(2.0) and fu[0, 2] == pytest.approx(3.0) and fu[0, 3] == 0.0
        assert fv[1, 0] == pytest.approx(1.5) and fv[1, 2] == pytest.approx(6.0)


class TestContinuity:

    def test_volume_conserved_in_closed_basin(self):
        hg, *_ = _grid()
        wet_t, wet_u, wet_v = _wet()
        rng = np.random.default_rng(0)
        eta = rng.normal(0, 0.5, (NY, NX))
        u = np.where(wet_u, rng.normal(0, 1, wet_u.shape), 0.0)
        v = np.where(wet_v, rng.normal(0, 1, wet_v.shape), 0.0)
        tend = nonlinear_continuity(eta, u, v, hg, wet_t)
        assert abs(np.sum(tend * hg["area_t"].values)) < 1e-9 * np.sum(np.abs(tend) * hg["area_t"].values)

    def test_gradient_advected_by_uniform_flow(self):
        # η = a x, u = U → −∂(η U)/∂x = −U a in the interior
        hg, x_u, *_ = _grid()
        wet_t, wet_u, wet_v = _wet()
        a, U = 1e-4, 0.5
        x_t = (np.arange(NX)[None, :] + 0.5) * DX * np.ones((NY, 1))
        tend = nonlinear_continuity(a * x_t, np.where(wet_u, U, 0.0), np.zeros(wet_v.shape), hg, wet_t)
        np.testing.assert_allclose(tend[:, 1:-1], -U * a, rtol=1e-10)


class TestAdvection:

    def test_uniform_flow_has_no_advection(self):
        hg, *_ = _grid()
        _, wet_u, wet_v = _wet()
        u, v = np.where(wet_u, 0.7, 0.0), np.where(wet_v, -0.3, 0.0)
        assert np.allclose(advection_u(u, v, hg, wet_u)[:, 2:-2], 0.0)
        assert np.allclose(advection_v(u, v, hg, wet_v)[2:-2, :], 0.0)

    def test_u_du_dx_exact_for_linear_field(self):
        hg, x_u, *_ = _grid()
        _, wet_u, wet_v = _wet()
        a = 1e-4
        u = np.where(wet_u, a * x_u, 0.0)
        adv = advection_u(u, np.zeros(wet_v.shape), hg, wet_u)
        np.testing.assert_allclose(adv[1:-1, 2:-2], (a * x_u * a)[1:-1, 2:-2], rtol=1e-10)

    def test_v_du_dy_exact_for_linear_field(self):
        hg, x_u, y_u, *_ = _grid()
        _, wet_u, wet_v = _wet()
        b, V = 2e-5, 0.4
        u = np.where(wet_u, b * y_u, 0.0)
        v = np.where(wet_v, V, 0.0)
        adv = advection_u(u, v, hg, wet_u)
        np.testing.assert_allclose(adv[2:-2, 2:-2], V * b, rtol=1e-10)

    def test_v_dv_dy_exact_for_linear_field(self):
        hg, _, _, x_v, y_v = _grid()
        _, wet_u, wet_v = _wet()
        a = 3e-5
        v = np.where(wet_v, a * y_v, 0.0)
        adv = advection_v(np.zeros(wet_u.shape), v, hg, wet_v)
        np.testing.assert_allclose(adv[2:-2, 1:-1], (a * y_v * a)[2:-2, 1:-1], rtol=1e-10)

    def test_dry_faces_untouched(self):
        hg, x_u, *_ = _grid()
        _, wet_u, wet_v = _wet()
        adv = advection_u(np.where(wet_u, 1e-4 * x_u, 0.0), np.zeros(wet_v.shape), hg, wet_u)
        assert np.all(adv[~wet_u] == 0.0)


class TestViscosity:

    def test_laplacian_of_quadratic(self):
        hg, x_u, y_u, x_v, y_v = _grid()
        _, wet_u, wet_v = _wet()
        u = np.where(wet_u, (x_u / DX) ** 2 + (y_u / DY) ** 2, 0.0)
        lap = laplacian_u(u, hg, wet_u)
        np.testing.assert_allclose(lap[1:-1, 2:-2], 2.0 / DX**2 + 2.0 / DY**2, rtol=1e-10)
        v = np.where(wet_v, (x_v / DX) ** 2, 0.0)
        np.testing.assert_allclose(laplacian_v(v, hg, wet_v)[2:-2, 1:-1], 2.0 / DX**2, rtol=1e-10)

    def test_free_slip_at_walls_conserves_momentum_of_uniform_flow(self):
        hg, *_ = _grid()
        _, wet_u, _ = _wet()
        assert np.allclose(laplacian_u(np.where(wet_u, 1.0, 0.0), hg, wet_u), 0.0)


class TestFreeSurfaceSource:

    def test_explicit_eta_tendency_added(self):
        # A uniform source raises η uniformly in a closed basin, with no flow
        hg = create_cartesian_grid(nx=6, ny=5, dx=1000.0, dy=1000.0, f0=0.0)
        vg = create_z_grid(1, [50.0])
        hf = compute_hfac(vg, np.full((5, 6), 50.0))
        fs = ImplicitFreeSurface(hg, vg, hf, dt=60.0, theta=0.6)
        u, v = np.zeros(hf.hfac_u.shape), np.zeros(hf.hfac_v.shape)
        eta, u1, v1 = fs.step(np.zeros((5, 6)), u, v, u, v, eta_tendency=np.full((5, 6), 1e-5))
        np.testing.assert_allclose(eta, 60.0 * 1e-5, rtol=1e-10)
        assert np.allclose(u1, 0.0) and np.allclose(v1, 0.0)


def test_nonlinear_basin_conserves_volume_over_many_steps():
    # Implicit linear operator + explicit −∇·(η u) (AB2): total volume Σ A η stays constant
    nx, ny = 20, 8
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=1e-4)
    vg = create_z_grid(1, [5.0])
    hf = compute_hfac(vg, np.full((ny, nx), 5.0))
    fs = ImplicitFreeSurface(hg, vg, hf, dt=20.0, theta=0.6)
    wet_t = np.ones((ny, nx), bool)
    x = (np.arange(nx) + 0.5) / nx
    eta = 1.0 * np.cos(np.pi * x)[None, :] * np.ones((ny, 1))           # η/H = 0.2: strongly nonlinear
    u, v = np.zeros(hf.hfac_u.shape), np.zeros(hf.hfac_v.shape)
    area = hg["area_t"].values
    v0 = np.sum(area * eta)
    prev = None
    for _ in range(300):
        s = nonlinear_continuity(eta, u[0], v[0], hg, wet_t)
        s_ab = s if prev is None else 1.5 * s - 0.5 * prev
        prev = s
        eta, u, v = fs.step(eta, u, v, u.copy(), v.copy(), eta_tendency=s_ab)
    assert np.isfinite(eta).all() and np.abs(eta).max() < 1.5
    assert abs(np.sum(area * eta) - v0) < 1e-9 * np.sum(area * np.abs(eta))


def test_advection_exact_for_linear_field_on_stretched_grid():
    from core.grid.horizontal import create_cartesian_grid_stretched
    dx = np.array([1000.0, 1500.0, 800.0, 2000.0, 1200.0, 900.0, 1700.0])
    dy = np.full(5, 1000.0)
    hg = create_cartesian_grid_stretched(dx, dy, x0=0.0, y0=0.0, f0=0.0)
    x_u = np.concatenate([[0.0], np.cumsum(dx)])[None, :] * np.ones((5, 1))
    wet_u = np.ones((5, 8), bool)
    a = 1e-4
    adv = advection_u(a * x_u, np.zeros((6, 7)), hg, wet_u)
    np.testing.assert_allclose(adv[:, 1:-1], (a * x_u * a)[:, 1:-1], rtol=1e-10)


class TestUpwind3:
    """Third-order upwind-biased advection (Leonard 1979; ROMS/MITgcm 'third-order upwind')."""

    def test_exact_for_quadratic_field(self):
        hg, x_u, *_ = _grid()
        _, wet_u, wet_v = _wet()
        a, b, c = 0.3, 1e-4, 2e-9
        u = np.where(wet_u, a + b * x_u + c * x_u**2, 0.0)
        adv = advection_u(u, np.zeros(wet_v.shape), hg, wet_u, scheme="upwind3")
        np.testing.assert_allclose(adv[1:-1, 3:-3], (u * (b + 2 * c * x_u))[1:-1, 3:-3], rtol=1e-9)

    def test_checkerboard_is_damped_not_ignored(self):
        # u = U0 + ε(−1)^i: centred advection is blind to the 2Δx mode, upwind3 damps it
        hg, x_u, *_ = _grid()
        _, wet_u, wet_v = _wet()
        pert = 0.05 * (-1.0) ** np.arange(NX + 1)[None, :] * np.ones((NY, 1))
        u = np.where(wet_u, 0.5 + pert, 0.0)
        cen = advection_u(u, np.zeros(wet_v.shape), hg, wet_u)[2:-2, 3:-3]
        up3 = advection_u(u, np.zeros(wet_v.shape), hg, wet_u, scheme="upwind3")[2:-2, 3:-3]
        assert np.allclose(cen, 0.0)
        assert np.all(np.sign(-up3) == -np.sign(pert[2:-2, 3:-3]))         # tendency −adv opposes ε

    def test_v_component_and_uniform_flow(self):
        hg, _, _, x_v, y_v = _grid()
        _, wet_u, wet_v = _wet()
        a = 3e-5
        v = np.where(wet_v, a * y_v, 0.0)
        adv = advection_v(np.zeros(wet_u.shape), v, hg, wet_v, scheme="upwind3")
        np.testing.assert_allclose(adv[3:-3, 1:-1], (a * y_v * a)[3:-3, 1:-1], rtol=1e-10)
        u0 = np.where(wet_u, 0.7, 0.0)
        assert np.allclose(advection_u(u0, np.where(wet_v, -0.3, 0.0), hg, wet_u, scheme="upwind3")[2:-2, 3:-3], 0.0)

    def test_unknown_scheme(self):
        hg, *_ = _grid()
        _, wet_u, wet_v = _wet()
        with pytest.raises(ValueError, match="scheme"):
            advection_u(np.zeros(wet_u.shape), np.zeros(wet_v.shape), hg, wet_u, scheme="quick")


def test_upwind3_damps_checkerboard_next_to_walls():
    # Where the 4-point stencil reaches land (1-cell channels, coasts) the scheme
    # must stay dissipative (first-order upwind), not fall back to centred
    hg, *_ = _grid()
    _, wet_u, wet_v = _wet()
    pert = 0.05 * (-1.0) ** np.arange(NX + 1)[None, :] * np.ones((NY, 1))
    u = np.where(wet_u, 0.5 + pert, 0.0)
    up3 = advection_u(u, np.zeros(wet_v.shape), hg, wet_u, scheme="upwind3")
    cols = [2, NX - 2]                     # ±1 neighbours wet, the ±2 neighbour is the wall face
    near = up3[2:-2, cols]
    assert np.all(np.sign(-near) == -np.sign(pert[2:-2, cols]))


def test_laplacian_3d_matches_the_per_layer_laplacian():
    # ALE performance task 5.2: one 3-D kernel call, the same per-element arithmetic (bit-identical)
    from core.dynamics.nonlinear import _spacings_u, _spacings_v, laplacian_3d, laplacian_v
    from core.grid.horizontal import create_cartesian_grid_stretched
    rng = np.random.default_rng(8)
    ny, nx, n = 9, 13, 4
    hg = create_cartesian_grid_stretched(rng.uniform(500.0, 1500.0, nx), rng.uniform(500.0, 1500.0, ny), f0=0.0)
    wet_u = rng.uniform(size=(ny, nx + 1)) > 0.2
    wet_v = rng.uniform(size=(ny + 1, nx)) > 0.2
    u, v = rng.normal(size=(n, ny, nx + 1)), rng.normal(size=(n, ny + 1, nx))
    ref_u = np.stack([laplacian_u(u[k], hg, wet_u) for k in range(n)])
    ref_v = np.stack([laplacian_v(v[k], hg, wet_v) for k in range(n)])
    assert np.array_equal(laplacian_3d(u, wet_u, *_spacings_u(hg)), ref_u)
    assert np.array_equal(laplacian_3d(v, wet_v, *_spacings_v(hg)), ref_v)
