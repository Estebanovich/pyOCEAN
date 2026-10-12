"""
Tests for core/dynamics/vector_invariant.py — vector-invariant momentum terms of the ALE mode
(spec docs/superpowers/specs/2026-10-12-ale-vector-invariant-momentum-design.md §3, §6).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics import vector_invariant as vi
from core.grid.horizontal import create_cartesian_grid


def _grid(nx=6, ny=5, dx=1000.0, dy=800.0, f0=1e-4):
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dy, f0=f0)
    names = ("dx_u", "dy_v", "dx_v", "dy_u", "area_t", "area_u", "area_v", "f_q", "x_u", "y_u", "x_v", "y_v")
    g = {k: np.asarray(hg[k].values, float) for k in names}
    return hg, g


def test_f_plane_uniform_flow():
    # spec §6.1: uniform u, uniform h, ζ = 0 → CAu = 0, CAv = −f u, ∇K = 0
    _, g = _grid()
    n, ny, nx = 2, 5, 6
    act = np.ones((ny, nx), bool)
    u = np.full((n, ny, nx + 1), 0.3)
    v = np.zeros((n, ny + 1, nx))
    h = np.full((n, ny, nx), 10.0)
    zeta = vi.vorticity_q(u, v, g["dx_u"], g["dy_v"], vi.corner_area(g["dx_u"], g["dy_v"]), vi.corner_mask(act))
    np.testing.assert_array_equal(zeta, 0.0)
    q = vi.pv_q(zeta, g["f_q"], h, g["area_t"], act)
    Qx, Qy = g["dy_u"][None] * u * 10.0, g["dx_v"][None] * v * 10.0
    cu, cv = vi.coriolis_pv(q, g["f_q"][None] + zeta, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound=False)
    np.testing.assert_allclose(cu, 0.0, atol=1e-18)
    np.testing.assert_allclose(cv[:, 1:-1, :], -1e-4 * 0.3, rtol=1e-12)
    K = vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"])
    kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
    np.testing.assert_allclose(K, 0.5 * 0.3**2, rtol=1e-12)
    np.testing.assert_allclose(kx, 0.0, atol=1e-18)
    np.testing.assert_allclose(ky, 0.0, atol=1e-18)


def test_pv_flux_equals_the_energy_conserving_coriolis_of_the_sigma_model():
    # spec §6.2: h uniform, ζ = 0, transports L u h → identical to SigmaBaroclinic._coriolis
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    hg, g = _grid(dx=1000.0, dy=1000.0)
    n, ny, nx = 3, 5, 6
    m = SigmaBaroclinic(hg, np.full((ny, nx), 20.0), dt=10.0, n_layers=n)
    rng = np.random.default_rng(0)
    u, v = rng.normal(size=(n, ny, nx + 1)), rng.normal(size=(n, ny + 1, nx))
    u[..., [0, -1]] = 0.0
    v[:, [0, -1], :] = 0.0
    h = np.full((n, ny, nx), 7.0)
    act = np.ones((ny, nx), bool)
    q = vi.pv_q(np.zeros((n, ny + 1, nx + 1)), g["f_q"], h, g["area_t"], act)
    cu, cv = vi.coriolis_pv(q, np.broadcast_to(g["f_q"], q.shape), g["dy_u"][None] * u * 7.0,
                            g["dx_v"][None] * v * 7.0, u, v, g["dx_u"], g["dy_v"], bound=False)
    cu0, cv0 = m._coriolis(u, v)
    np.testing.assert_allclose(cu, cu0, rtol=1e-12, atol=1e-20)
    np.testing.assert_allclose(cv, cv0, rtol=1e-12, atol=1e-20)


@pytest.mark.parametrize("stretched", [False, True])
def test_sadourny_energy_pv_flux_does_no_work(stretched):
    # spec §6.3 + review focus 2: Σ (Qx CAu Δx_u + Qy CAv Δy_v) = 0 for any q, closed domain
    n, ny, nx = 2, 5, 7
    _, g = _grid(nx=nx, ny=ny)
    rng = np.random.default_rng(1)
    if stretched:
        g["dx_u"] = g["dx_u"] * rng.uniform(0.5, 1.5, g["dx_u"].shape)
        g["dy_v"] = g["dy_v"] * rng.uniform(0.5, 1.5, g["dy_v"].shape)
    q = rng.normal(size=(n, ny + 1, nx + 1))
    Qx, Qy = rng.normal(size=(n, ny, nx + 1)), rng.normal(size=(n, ny + 1, nx))
    Qx[..., [0, -1]] = 0.0
    Qy[:, [0, -1], :] = 0.0
    cu, cv = vi.coriolis_pv(q, q, Qx, Qy, Qx, Qy, g["dx_u"], g["dy_v"], bound=False)
    work = (Qx * cu * g["dx_u"][None]).sum() + (Qy * cv * g["dy_v"][None]).sum()
    assert abs(work) < 1e-12 * np.abs(Qx * cu * g["dx_u"][None]).sum()


def test_solid_body_rotation_vorticity_and_free_slip_coast():
    # spec §6.4 + review focus 1
    _, g = _grid(nx=8, ny=7, f0=0.0)
    ny, nx = 7, 8
    om = 1e-5
    u = (-om * g["y_u"])[None]
    v = (om * g["x_v"])[None]
    act = np.ones((ny, nx), bool)
    act[3, 4] = False                                                   # an island
    mq = vi.corner_mask(act)
    zeta = vi.vorticity_q(u, v, g["dx_u"], g["dy_v"], vi.corner_area(g["dx_u"], g["dy_v"]), mq)
    np.testing.assert_allclose(zeta[0][mq], 2 * om, rtol=1e-10)
    np.testing.assert_array_equal(zeta[0][~mq], 0.0)
    assert not mq[3:5, 4:6].any() and not mq[0].any() and not mq[:, -1].any()


def test_kinetic_energy_schemes():
    # spec §6.5: K = u²/2 for uniform u; Godunov uses only inflow velocities
    _, g = _grid()
    n, ny, nx = 1, 5, 6
    u = np.full((n, ny, nx + 1), -0.4)
    v = np.zeros((n, ny + 1, nx))
    for s in ("arakawa", "gudonov"):
        np.testing.assert_allclose(vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"], s), 0.08, rtol=1e-12)
    u = np.zeros((n, ny, nx + 1))
    u[0, 2, 3] = 1.0                                                    # leaves cell (2, 2) east, enters (2, 3) west
    K = vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"], "gudonov")
    assert K[0, 2, 2] == 0.0 and K[0, 2, 3] > 0.0


def test_bound_coriolis_caps_the_pv_flux_of_a_vanished_layer():
    # spec §6.6: q·vh ≈ (f+ζ) v h_face/h_q explodes when h_q ≪ h_face; the clip caps it at max (f+ζ) v
    _, g = _grid(nx=4, ny=4)
    n, ny, nx = 1, 4, 4
    act = np.ones((ny, nx), bool)
    h = np.full((n, ny, nx), 1e-3)                                      # vanished layer
    u = np.zeros((n, ny, nx + 1))
    v = np.full((n, ny + 1, nx), 0.1)
    v[:, [0, -1], :] = 0.0
    zeta = np.zeros((n, ny + 1, nx + 1))
    q = vi.pv_q(zeta, g["f_q"], h, g["area_t"], act)
    Qy = g["dx_v"][None] * v * 10.0                                     # transport of the thick upwind layer
    Qx = np.zeros_like(u)
    cu0, _ = vi.coriolis_pv(q, g["f_q"][None] + zeta, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound=False)
    cu, _ = vi.coriolis_pv(q, g["f_q"][None] + zeta, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound=True)
    assert np.abs(cu0).max() > 100 * 1e-4 * 0.1
    assert np.abs(cu).max() <= 1e-4 * 0.1 * (1 + 1e-12)
    h0 = np.zeros_like(h)                                               # empty cells: no NaN
    assert np.isfinite(vi.pv_q(zeta, g["f_q"], h0, g["area_t"], act)).all()


def test_cfl_truncation_caps_the_velocity_and_counts_only_thick_faces():
    # spec §6.7 (MOM6 vertvisc_limit_vel, MOM_vert_friction.F90:2555-2715)
    dt, dx = 10.0, 100.0
    u = np.zeros((2, 1, 4))
    u[0, 0, 1] = 7.0                                                     # Courant 0.7 → 0.45 Δx/Δt = 4.5 m/s
    u[1, 0, 2] = -7.0
    h_face = np.ones_like(u)
    h_face[1] = 1e-3                                                     # thin layer: truncated, not counted
    dx_t = np.full((1, 3), dx)
    out, n, first = vi.truncate_cfl(u, dt, dx_t, h_face, 0.01)
    assert out[0, 0, 1] == pytest.approx(4.5) and out[1, 0, 2] == pytest.approx(-4.5)
    assert n == 1 and first == (0, 0, 1)
    same, n0, f0 = vi.truncate_cfl(np.full_like(u, 4.0), dt, dx_t, h_face, 0.01)
    np.testing.assert_array_equal(same, 4.0)
    assert n0 == 0 and f0 is None


def test_cfl_truncation_uses_the_donor_cell_width():
    # review focus 5: u > 0 takes the west cell, u < 0 the east cell
    dt = 10.0
    dx_t = np.array([[100.0, 1000.0]])
    u = np.array([[[0.0, 7.0, 0.0]]])                                   # west cell 100 m: Courant 0.7
    out, _, _ = vi.truncate_cfl(u, dt, dx_t, np.ones_like(u), 0.01)
    assert out[0, 0, 1] == pytest.approx(4.5)
    u = np.array([[[0.0, -7.0, 0.0]]])                                  # east cell 1000 m: Courant 0.07
    out, _, _ = vi.truncate_cfl(u, dt, dx_t, np.ones_like(u), 0.01)
    assert out[0, 0, 1] == -7.0


def test_implicit_viscosity_slaves_a_vanishing_layer_to_its_neighbours():
    # spec §6.8 (guard of existing behaviour: passes before any change).  MOM6 form
    # (MOM_vert_friction.F90:711-783): dz_k u_k + Δt ν/(½(dz_k+dz_k±1)) coupling; with ν = 1e-4,
    # Δt = 20 s and 1 m neighbours a layer thinner than ~4Δtν/H ≈ 8 mm loses its own tendency.
    # Ruling (plan): the spec's "1 mm" is at the threshold (the layer only moves half way); 0.1 mm
    # is well below it.
    from core.dynamics.sigma_wd import vertical_implicit
    u = np.array([[0.5], [5.0], [0.5]])                                 # thin middle layer with a large tendency
    dz = np.array([[1.0], [1e-4], [1.0]])
    out = vertical_implicit(u, dz, np.full((4, 1), 1e-4), np.zeros(1), 20.0)
    assert abs(out[1, 0] - 0.5) < 0.1


# ---------------------------------------------------------------- performance kernels (ALE performance task 5.3)
def _vi_random(seed=3, n=4, ny=7, nx=9):
    from core.grid.horizontal import create_cartesian_grid_stretched
    rng = np.random.default_rng(seed)
    hg = create_cartesian_grid_stretched(rng.uniform(500.0, 1500.0, nx), rng.uniform(500.0, 1500.0, ny), f0=1e-4)
    g = {k: np.asarray(hg[k].values, float) for k in ("dx_u", "dy_v", "area_t", "area_u", "area_v")}
    active = rng.uniform(size=(ny, nx)) > 0.15
    g["area_q"], g["mask_q"] = vi.corner_area(g["dx_u"], g["dy_v"]), vi.corner_mask(active)
    u, v = rng.normal(0, 0.3, (n, ny, nx + 1)), rng.normal(0, 0.3, (n, ny + 1, nx))
    h = rng.uniform(0.0, 10.0, (n, ny, nx))
    h[0, 2, 3] = 0.0
    Qx, Qy = rng.normal(0, 50.0, u.shape), rng.normal(0, 50.0, v.shape)
    fq = np.full((ny + 1, nx + 1), 1e-4)
    return g, active, u, v, h, Qx, Qy, fq


def test_vorticity_and_pv_kernels_match_numpy():
    g, active, u, v, h, Qx, Qy, fq = _vi_random()
    z = vi.vorticity_q(u, v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"])
    assert np.array_equal(z, vi.vorticity_q_numpy(u, v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"]))
    assert np.array_equal(vi.pv_q(z, fq, h, g["area_t"], active), vi.pv_q_numpy(z, fq, h, g["area_t"], active))


@pytest.mark.parametrize("bound", [True, False])
def test_coriolis_kernel_matches_numpy(bound):
    g, active, u, v, h, Qx, Qy, fq = _vi_random(4)
    z = vi.vorticity_q_numpy(u, v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"])
    q = vi.pv_q_numpy(z, fq, h, g["area_t"], active)
    a = vi.coriolis_pv(q, fq[None] + z, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound)
    b = vi.coriolis_pv_numpy(q, fq[None] + z, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


@pytest.mark.parametrize("scheme", ["arakawa", "gudonov"])
def test_kinetic_energy_kernel_matches_numpy(scheme):
    g, active, u, v, h, Qx, Qy, fq = _vi_random(5)
    a = vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"], scheme)
    b = vi.kinetic_energy_numpy(u, v, g["area_u"], g["area_v"], g["area_t"], scheme)
    assert np.array_equal(a, b)


def test_fused_vorticity_and_pv_match_numpy():
    g, active, u, v, h, Qx, Qy, fq = _vi_random(6)
    z, q = vi.vorticity_and_pv(u, v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"], fq, h, g["area_t"], active)
    zr = vi.vorticity_q_numpy(u, v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"])
    assert np.array_equal(z, zr) and np.array_equal(q, vi.pv_q_numpy(zr, fq, h, g["area_t"], active))
