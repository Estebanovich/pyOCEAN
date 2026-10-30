"""
Tests of the Laplacian Smagorinsky operator in stress-tensor form (core/dynamics/hor_visc.py).

Plan docs/superpowers/plans/2026-10-30-smagorinsky-viscosity.md, Task 1; Review Focus 1, 3, 4, 5.
Reference: MOM6 src/parameterizations/lateral/MOM_hor_visc.F90 (docs/design/smag_refs/mom6_hor_visc.md).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics import hor_visc as hv
from core.grid.horizontal import create_cartesian_grid, create_cartesian_grid_stretched
from core.model.threads import _THREADS, set_threads
from core.vertical.remap import H_NEGLECT

BOUND = 0.8 * (6.0 / 11.0) / 4.0       # AB3-rescaled MOM6 BOUND_KH coefficient (spec, "Definiciones")


# ---------------------------------------------------------------- helpers
def _closed_masks(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Box with walls: edge faces closed, every interior face wet; corners with four wet faces."""
    wet_u = np.ones((ny, nx + 1), bool)
    wet_u[:, 0] = wet_u[:, -1] = False
    wet_v = np.ones((ny + 1, nx), bool)
    wet_v[0] = wet_v[-1] = False
    mask_q = np.zeros((ny + 1, nx + 1), bool)
    mask_q[1:-1, 1:-1] = (wet_u[:-1, 1:-1] & wet_u[1:, 1:-1] & wet_v[1:-1, :-1] & wet_v[1:-1, 1:])
    return wet_u, wet_v, mask_q


def _no_open(ny: int, nx: int) -> tuple[np.ndarray, np.ndarray]:
    return np.zeros((ny, nx + 1), bool), np.zeros((ny + 1, nx), bool)


def _stretched_grid():
    dx = np.linspace(1000.0, 1600.0, 7)
    dy = np.linspace(800.0, 1200.0, 6)
    return create_cartesian_grid_stretched(dx, dy)


def _random_state(rng, n, ny, nx, wet_u, wet_v, h_zero_frac: float = 0.0, hmax: float = 50.0):
    u = rng.normal(0.0, 0.3, (n, ny, nx + 1)) * wet_u
    v = rng.normal(0.0, 0.3, (n, ny + 1, nx)) * wet_v
    def _h(shape):
        h = rng.uniform(0.0, hmax, shape)
        h[rng.uniform(size=shape) < h_zero_frac] = 0.0
        return h
    return u, v, _h((n, ny, nx)), _h((n, ny, nx + 1)), _h((n, ny + 1, nx))


def _reference_nu(u, v, g, C):
    """C Δ² |S| at h and q with explicit NumPy (Cartesian stretched grid, plan "Global Constraints")."""
    dx_t, dy_t = g["dx_t"].values, g["dy_t"].values
    dx_u, dy_v = g["dx_u"].values, g["dy_v"].values
    dx_v, dy_u = g["dx_v"].values, g["dy_u"].values
    ny, nx = dx_t.shape
    DT = (u[..., 1:] - u[..., :-1]) / dx_t - (v[:, 1:] - v[:, :-1]) / dy_t
    DS = np.zeros((u.shape[0], ny + 1, nx + 1))
    # ∂v/∂x at corner (J, I) uses the distance between the T-points, dx_u; ∂u/∂y uses dy_v.
    dxq = dx_u[1:, 1:-1]                                  # dx_u is uniform in j on this grid
    dyq = dy_v[1:-1, 1:]                                  # dy_v is uniform in i on this grid
    DS[:, 1:-1, 1:-1] = ((v[:, 1:-1, 1:] - v[:, 1:-1, :-1]) / dxq
                         + (u[:, 1:, 1:-1] - u[:, :-1, 1:-1]) / dyq)
    S_t = np.sqrt(DT**2 + 0.25 * (DS[:, :-1, :-1]**2 + DS[:, 1:, 1:]**2 + DS[:, 1:, :-1]**2 + DS[:, :-1, 1:]**2))
    S_q = np.sqrt(DS[:, 1:-1, 1:-1]**2 + 0.25 * (DT[:, :-1, :-1]**2 + DT[:, 1:, 1:]**2
                                                 + DT[:, 1:, :-1]**2 + DT[:, :-1, 1:]**2))
    # Δ² = 2 dx² dy² / (dx² + dy²); at q, dx_q = mean of the dx_v around it in i, dy_q of the dy_u in j
    d2_t = 2 * dx_t**2 * dy_t**2 / (dx_t**2 + dy_t**2)
    dx_qc = 0.5 * (dx_v[1:-1, :-1] + dx_v[1:-1, 1:])
    dy_qc = 0.5 * (dy_u[:-1, 1:-1] + dy_u[1:, 1:-1])
    d2_q = 2 * dx_qc**2 * dy_qc**2 / (dx_qc**2 + dy_qc**2)
    return C * d2_t * S_t, C * d2_q * S_q, DT, DS


def _call(fn, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom, C, ah, dt):
    return fn(u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom, C, ah, dt, H_NEGLECT)


# ---------------------------------------------------------------- geometry
def test_geometry_q_spacings_and_delta():
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    dx_v, dy_u = g["dx_v"].values, g["dy_u"].values
    assert geom["dx_q"].shape == (7, 8) and geom["dy_q"].shape == (7, 8)
    np.testing.assert_allclose(geom["dx_q"][:, 1:-1], 0.5 * (dx_v[:, :-1] + dx_v[:, 1:]), rtol=1e-15)
    np.testing.assert_array_equal(geom["dx_q"][:, 0], dx_v[:, 0])
    np.testing.assert_array_equal(geom["dx_q"][:, -1], dx_v[:, -1])
    np.testing.assert_allclose(geom["dy_q"][1:-1], 0.5 * (dy_u[:-1] + dy_u[1:]), rtol=1e-15)
    np.testing.assert_array_equal(geom["dy_q"][0], dy_u[0])
    np.testing.assert_array_equal(geom["dy_q"][-1], dy_u[-1])
    dx, dy = g["dx_t"].values, g["dy_t"].values
    np.testing.assert_allclose(geom["delta2_t"], 2 * dx**2 * dy**2 / (dx**2 + dy**2), rtol=1e-15)
    np.testing.assert_allclose(geom["inv_lap_t"], 1.0 / (1 / dx**2 + 1 / dy**2), rtol=1e-15)
    for key in ("dx_t", "dy_t", "dx_u", "dy_u", "dx_v", "dy_v", "area_t", "area_u", "area_v",
                "area_q", "delta2_q", "inv_lap_q"):
        assert key in geom


# ---------------------------------------------------------------- Review Focus 3: stretched grid
def test_nu_matches_mom6_formula_on_stretched_grid():
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 6, 7
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    rng = np.random.default_rng(11)
    u = rng.normal(0, 0.5, (3, ny, nx + 1)) * wet_u
    v = rng.normal(0, 0.5, (3, ny + 1, nx)) * wet_v
    h_t, h_u, h_v = np.full((3, ny, nx), 20.0), np.full(u.shape, 20.0), np.full(v.shape, 20.0)
    C = 0.15
    _, _, nu_t, nu_q, n_lim = _call(hv.smagorinsky_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q,
                                    open_u, open_v, geom, C, 0.0, 1e-3)
    ref_t, ref_q, _, _ = _reference_nu(u, v, g, C)
    assert n_lim == 0
    np.testing.assert_allclose(nu_t, ref_t, rtol=1e-13, atol=0)
    np.testing.assert_allclose(nu_q[:, 1:-1, 1:-1], ref_q, rtol=1e-13, atol=0)
    assert np.all(nu_q[:, 0] == 0) and np.all(nu_q[:, -1] == 0)   # edge corners: no shear stress
    assert np.all(nu_q[:, :, 0] == 0) and np.all(nu_q[:, :, -1] == 0)


# ---------------------------------------------------------------- Review Focus 4: constant ν
def test_constant_nu_shear_wave_decays_analytically():
    nx, ny, d = 40, 16, 1000.0
    g = create_cartesian_grid(nx, ny, d, d, 1e-4)
    geom = hv.smag_geometry(g)
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    Ly, U, ah = ny * d, 0.2, 100.0
    y_t = (np.arange(ny) + 0.5) * d
    u = (U * np.cos(np.pi * y_t / Ly))[None, :, None] * np.ones((1, ny, nx + 1)) * wet_u
    v = np.zeros((1, ny + 1, nx))
    h_t, h_u, h_v = np.full((1, ny, nx), 10.0), np.full(u.shape, 10.0), np.full(v.shape, 10.0)
    fu, fv, nu_t, nu_q, n_lim = _call(hv.smagorinsky_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q,
                                      open_u, open_v, geom, 0.0, ah, 1.0)
    assert n_lim == 0
    interior = slice(3, nx - 2)                       # faces ≥ 3 cells from the walls in x
    exact = -ah * (np.pi / Ly) ** 2 * u[:, :, interior]
    rel = np.abs(fu[:, :, interior] - exact) / np.abs(exact).max()
    assert rel.max() <= 2 * (np.pi * d / Ly) ** 2
    assert rel.max() > 0                               # O(Δy²) discretisation error, not zero
    np.testing.assert_allclose(fv, 0.0, atol=1e-20)


def test_quiet_flow_gives_background_ah():
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 6, 7
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    u, v = np.zeros((2, ny, nx + 1)), np.zeros((2, ny + 1, nx))
    h_t, h_u, h_v = np.full((2, ny, nx), 5.0), np.full(u.shape, 5.0), np.full(v.shape, 5.0)
    fu, fv, nu_t, nu_q, n_lim = _call(hv.smagorinsky_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q,
                                      open_u, open_v, geom, 0.15, 7.0, 60.0)
    assert np.all(nu_t == 7.0)                         # every cell has a wet face here
    assert np.all(nu_q[:, mask_q] == 7.0)              # active corners
    assert np.all(nu_q[:, ~mask_q] == 0.0)             # no shear stress at land/edge corners
    assert np.all(fu == 0.0) and np.all(fv == 0.0) and n_lim == 0


# ---------------------------------------------------------------- energy
def test_energy_dissipation_nonnegative():
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 6, 7
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    rng = np.random.default_rng(2026)
    for _ in range(20):
        u, v, h_t, h_u, h_v = _random_state(rng, 4, ny, nx, wet_u, wet_v, h_zero_frac=0.3)
        fu, fv, *_ = _call(hv.smagorinsky_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q,
                           open_u, open_v, geom, 0.15, 1.0, 600.0)
        assert np.all(np.isfinite(fu)) and np.all(np.isfinite(fv))
        a_u, a_v = h_u * u * fu * geom["area_u"], h_v * v * fv * geom["area_v"]
        work = a_u.sum() + a_v.sum()
        scale = np.abs(a_u).sum() + np.abs(a_v).sum()
        assert work <= 1e-12 * scale
        assert work < 0                                 # the operator does dissipate here


# ---------------------------------------------------------------- Review Focus 1: vanished layers
def test_vanished_layers_finite_and_zero_nu():
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 6, 7
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    rng = np.random.default_rng(5)
    u, v, h_t, h_u, h_v = _random_state(rng, 2, ny, nx, wet_u, wet_v)
    h_t += 1.0; h_u += 1.0; h_v += 1.0
    h_u[1, 2, 3] = 0.0                                 # one vanished face: corners (2, 3) and (3, 3)
    h_v[0, 4, 5] = 0.0                                 # and another: corners (4, 5) and (4, 6)
    h_t[0, 1, 1] = 0.0                                 # a vanished cell with thick faces
    out = _call(hv.smagorinsky_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q,
                open_u, open_v, geom, 0.15, 2.0, 600.0)
    fu, fv, nu_t, nu_q, n_lim = out
    for a in out[:4]:
        assert np.all(np.isfinite(a))
    assert nu_q[1, 2, 3] == 0.0 and nu_q[1, 3, 3] == 0.0
    assert nu_q[0, 4, 5] == 0.0 and nu_q[0, 4, 6] == 0.0
    assert nu_t[1, 2, 2] == 0.0 and nu_t[1, 2, 3] == 0.0     # cells on both sides of the face
    assert nu_t[0, 3, 5] == 0.0 and nu_t[0, 4, 5] == 0.0
    assert fu[1, 2, 3] == 0.0 and fv[0, 4, 5] == 0.0          # no stress reaches the vanished faces
    assert n_lim >= 4                                         # ah = 2 was cut to 0 at those points


# ---------------------------------------------------------------- stability bound
def test_bound_rescaled_for_ab3_and_counted():
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 6, 7
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    rng = np.random.default_rng(7)
    u, v, h_t, h_u, h_v = _random_state(rng, 2, ny, nx, wet_u, wet_v)
    h_t += 0.5; h_u += 0.5; h_v += 0.5
    u[1, :, :4] = 0.0; v[1, :, :4] = 0.0              # a quiet block: |S| = 0 there, not limited
    u[1, :, 4] = 0.0
    C, ah, dt = 1e6, 1e-3, 600.0
    fu, fv, nu_t, nu_q, n_lim = _call(hv.smagorinsky_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q,
                                      open_u, open_v, geom, C, ah, dt)
    # thin-neighbour ratio (MOM6 hrat_min, MOM_hor_visc.F90:1033-1037, 1368-1371)
    # at h points only the wet faces count (a wall face gets no tendency, see the module docstring)
    hu_w, hv_w = np.where(wet_u, h_u, np.inf), np.where(wet_v, h_v, np.inf)
    hmin_t = np.minimum(np.minimum(hu_w[..., :-1], hu_w[..., 1:]), np.minimum(hv_w[:, :-1], hv_w[:, 1:]))
    hrat_t = np.minimum(1.0, hmin_t / (h_t + H_NEGLECT))
    a, b = h_u[:, :-1, 1:-1], h_u[:, 1:, 1:-1]
    c, d = h_v[:, 1:-1, :-1], h_v[:, 1:-1, 1:]
    h2u, h2v = 4 * a * b, 4 * c * d
    hq = 2 * h2u * h2v / (H_NEGLECT**3 + (h2u + h2v) * (a + b + c + d))
    hrat_q = np.minimum(1.0, np.minimum(np.minimum(a, b), np.minimum(c, d)) / (hq + H_NEGLECT))
    bound_t = BOUND / dt * geom["inv_lap_t"] * hrat_t
    bound_q = BOUND / dt * geom["inv_lap_q"][1:-1, 1:-1] * hrat_q
    assert np.all(nu_t <= bound_t * (1 + 1e-12))
    assert np.all(nu_q[:, 1:-1, 1:-1] <= bound_q * (1 + 1e-12))
    smag_t, smag_q, _, _ = _reference_nu(u, v, g, C)
    raw_t, raw_q = np.maximum(ah, smag_t), np.maximum(ah, smag_q)
    expected = int((raw_t > bound_t).sum() + (raw_q > bound_q).sum())
    assert 0 < expected < nu_t.size + raw_q.size        # some points limited, some not
    assert n_lim == expected


# ---------------------------------------------------------------- open edges
def test_open_edge_faces_get_no_tendency():
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 6, 7
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    wet_u[:, 0] = True                                 # the west edge is open (wet)
    open_u, open_v = _no_open(ny, nx)
    open_u[:, 0] = True
    rng = np.random.default_rng(3)
    u, v, h_t, h_u, h_v = _random_state(rng, 2, ny, nx, wet_u, wet_v)
    h_t += 1.0; h_u += 1.0; h_v += 1.0
    args = (h_t, h_u, h_v, wet_u, wet_v)
    fu, fv, nu_t, nu_q, _ = _call(hv.smagorinsky_numpy, u, v, *args, mask_q, open_u, open_v, geom, 0.15, 1.0, 60.0)
    assert np.all(fu[:, :, 0] == 0.0)
    assert np.all(nu_q[:, :, 0] == 0.0)
    # A caller that marks the edge corners wet (e.g. a padded halo) still gets D_S = 0 there.
    mq = mask_q.copy()
    mq[:, 0] = True
    out2 = _call(hv.smagorinsky_numpy, u, v, *args, mq, open_u, open_v, geom, 0.15, 1.0, 60.0)
    for a, b in zip((fu, fv, nu_t, nu_q), out2[:4]):
        assert np.array_equal(a, b)
    # An open column of faces inside the domain: its corners lose their shear (D_S = 0) exactly
    # as if they were land corners, and the faces get no tendency.
    ou = np.zeros_like(open_u)
    ou[:, 3] = True
    fa, fva, nta, nqa, _ = _call(hv.smagorinsky_numpy, u, v, *args, mq, ou, open_v, geom, 0.15, 1.0, 60.0)
    mq2 = mq.copy()
    mq2[:, 3] = False
    fb, fvb, ntb, nqb, _ = _call(hv.smagorinsky_numpy, u, v, *args, mq2, np.zeros_like(ou), open_v, geom,
                                 0.15, 1.0, 60.0)
    assert np.all(fa[:, :, 3] == 0.0)
    assert np.array_equal(np.delete(fa, 3, axis=2), np.delete(fb, 3, axis=2))
    assert np.array_equal(fva, fvb) and np.array_equal(nta, ntb) and np.array_equal(nqa, nqb)


# ---------------------------------------------------------------- Review Focus 5: Numba = NumPy
@pytest.mark.parametrize("threads", [1, 5])
def test_numba_equals_numpy_any_threads(threads):
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 6, 7
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    wet_u[:, 0] = True
    open_u, open_v = _no_open(ny, nx)
    open_u[:, 0] = True
    wet_u[2, 4] = False                                # a dry interior face
    mask_q[2:4, 4] = False
    rng = np.random.default_rng(99)
    u, v, h_t, h_u, h_v = _random_state(rng, 5, ny, nx, wet_u, wet_v, h_zero_frac=0.2)
    old = _THREADS[0]
    try:
        set_threads(threads)
        for C, ah, dt in ((0.15, 1.0, 600.0), (1e6, 0.5, 600.0), (0.0, 100.0, 1.0)):
            a = _call(hv.smagorinsky_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v,
                      geom, C, ah, dt)
            b = _call(hv.smagorinsky, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v,
                      geom, C, ah, dt)
            for x, y in zip(a[:4], b[:4]):
                assert np.array_equal(x, y)
            assert a[4] == b[4]
    finally:
        set_threads(old)
