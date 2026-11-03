"""
Tests of the biharmonic Smagorinsky operator in stress-tensor form (core/dynamics/hor_visc.py, ``biharmonic``).

Regional chain, Gulf Stream fix step 4 (NWA12: BIHARMONIC, SMAG_BI_CONST 0.03, AH_VEL_SCALE 0.01, BOUND_AH,
OBC_ZERO_BIHARMONIC).  Reference: MOM6 src/parameterizations/lateral/MOM_hor_visc.F90, git 0e9a561
(docs/design/regional_refs/mom6_biharmonic.md).  Index order (k, j, i); corners (N, ny+1, nx+1).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics import hor_visc as hv
from core.dynamics import open_boundary as obc
from core.grid.horizontal import create_cartesian_grid, create_cartesian_grid_stretched
from core.model.threads import _THREADS, set_threads
from core.vertical.remap import H_NEGLECT


# ---------------------------------------------------------------- helpers
def _closed_masks(ny: int, nx: int):
    wet_u = np.ones((ny, nx + 1), bool)
    wet_u[:, 0] = wet_u[:, -1] = False
    wet_v = np.ones((ny + 1, nx), bool)
    wet_v[0] = wet_v[-1] = False
    return wet_u, wet_v, _mask_q(wet_u, wet_v)


def _mask_q(wet_u, wet_v):
    ny, nx = wet_u.shape[0], wet_v.shape[1]
    mask_q = np.zeros((ny + 1, nx + 1), bool)
    mask_q[1:-1, 1:-1] = wet_u[:-1, 1:-1] & wet_u[1:, 1:-1] & wet_v[1:-1, :-1] & wet_v[1:-1, 1:]
    return mask_q


def _no_open(ny, nx):
    return np.zeros((ny, nx + 1), bool), np.zeros((ny + 1, nx), bool)


def _stretched_grid():
    return create_cartesian_grid_stretched(np.linspace(1000.0, 1600.0, 8), np.linspace(800.0, 1200.0, 7))


def _ones(n, ny, nx, h=10.0):
    return np.full((n, ny, nx), h), np.full((n, ny, nx + 1), h), np.full((n, ny + 1, nx), h)


def _call(fn, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom, C, vel, ah, dt, **kw):
    return fn(u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom, C, vel, ah, dt, H_NEGLECT, **kw)


# ---------------------------------------------------------------- analytic decay A k⁴
@pytest.mark.parametrize("axis", ["x", "y"])
def test_sinusoid_decays_at_rate_ah_k4(axis):
    # u = U cos(k s) along one axis, constant A_h, uniform h: away from the walls the operator is the discrete
    # −A_h ∂⁴u/∂s⁴ = −A_h (4 sin²(kΔ/2)/Δ²)² u exactly, and A_h k⁴ to O((kΔ)²).
    nx, ny, d, A = (48, 12, 1000.0, 1e9) if axis == "x" else (12, 48, 1000.0, 1e9)
    g = create_cartesian_grid(nx, ny, d, d, 1e-4)
    geom = hv.smag_geometry(g)
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    k, U = 2 * np.pi / (8 * d), 0.2
    if axis == "x":
        x_u = np.arange(nx + 1) * d
        u = (U * np.cos(k * x_u))[None, None, :] * np.ones((1, ny, nx + 1)) * wet_u
        sl = np.s_[:, 2:-2, 6:nx - 5]                      # the walls act on rows 0 and ny−1 (MOM6 Del2 at land faces)
    else:
        y_t = (np.arange(ny) + 0.5) * d
        u = (U * np.cos(k * y_t))[None, :, None] * np.ones((1, ny, nx + 1)) * wet_u
        sl = np.s_[:, 6:ny - 6, 3:-3]
    v = np.zeros((1, ny + 1, nx))
    fu, fv, ah_t, ah_q, n_lim = _call(hv.biharmonic_numpy, u, v, *_ones(1, ny, nx), wet_u, wet_v, mask_q,
                                      open_u, open_v, geom, 0.0, 0.0, A, 1.0)
    assert n_lim == 0 and np.all(ah_t == A)
    discrete = -A * (4 * np.sin(k * d / 2) ** 2 / d ** 2) ** 2 * u[sl]
    np.testing.assert_allclose(fu[sl], discrete, rtol=0, atol=1e-10 * np.abs(discrete).max())
    rate = -fu[sl] / u[sl]
    ok = np.abs(u[sl]) > 0.5 * U
    np.testing.assert_allclose(rate[ok], A * k ** 4, rtol=(k * d) ** 2 / 3)
    np.testing.assert_allclose(fv[sl[0], sl[1], 3:-3], 0.0, atol=1e-12 * np.abs(discrete).max())  # walls only


# ---------------------------------------------------------------- uniform flow and linear shear
@pytest.mark.parametrize("profile", ["uniform", "shear_y", "shear_x"])
def test_uniform_flow_and_linear_shear_have_no_tendency(profile):
    nx, ny, d = 20, 18, 1500.0
    g = create_cartesian_grid(nx, ny, d, d, 1e-4)
    geom = hv.smag_geometry(g)
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    y_t = (np.arange(ny) + 0.5) * d
    x_t = (np.arange(nx) + 0.5) * d
    u = np.full((2, ny, nx + 1), 0.3)
    v = np.full((2, ny + 1, nx), -0.1)
    if profile == "shear_y":
        u = u + 1e-5 * y_t[None, :, None]
    if profile == "shear_x":
        v = v + 2e-5 * x_t[None, None, :]
    u, v = u * wet_u, v * wet_v
    fu, fv, *_ = _call(hv.biharmonic_numpy, u, v, *_ones(2, ny, nx), wet_u, wet_v, mask_q, open_u, open_v, geom,
                       0.03, 0.01, 0.0, 600.0)
    inner = np.s_[:, 4:-4, 4:-4]
    scale = np.abs(fu).max() + np.abs(fv).max()
    assert scale > 0                                    # the walls do act
    np.testing.assert_allclose(fu[inner], 0.0, atol=1e-12 * scale)
    np.testing.assert_allclose(fv[inner], 0.0, atol=1e-12 * scale)


# ---------------------------------------------------------------- energy
def test_constant_ah_dissipates_energy():
    # A_h h uniform: the operator is −A_h L(L u) with L the stress-tensor Laplacian, so Σ h u F A = −A_h h ‖L u‖²
    # (summation by parts on the stretched grid, closed walls).
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 7, 8
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    rng = np.random.default_rng(2026)
    for _ in range(20):
        u = rng.normal(0, 0.3, (3, ny, nx + 1)) * wet_u
        v = rng.normal(0, 0.3, (3, ny + 1, nx)) * wet_v
        h_t, h_u, h_v = _ones(3, ny, nx, 25.0)
        fu, fv, *_ = _call(hv.biharmonic_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v,
                           geom, 0.0, 0.0, 1e8, 600.0)
        a_u, a_v = h_u * u * fu * geom["area_u"], h_v * v * fv * geom["area_v"]
        work = a_u.sum() + a_v.sum()
        assert work < 0
        assert work <= -1e-3 * (np.abs(a_u).sum() + np.abs(a_v).sum())


def test_smagorinsky_ah_dissipates_energy_on_smooth_flows():
    # Variable A_h (Smagorinsky): MOM6's form ∇·(A_h ∇(∇²u)) is not sign-definite in general, but for random
    # smooth flows on the stretched grid the work is negative (the case where it matters is the grid scale).
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 7, 8
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    rng = np.random.default_rng(7)
    for _ in range(20):
        u = rng.normal(0, 0.3, (2, ny, nx + 1)) * wet_u
        v = rng.normal(0, 0.3, (2, ny + 1, nx)) * wet_v
        h_t, h_u, h_v = _ones(2, ny, nx, 25.0)
        fu, fv, *_ = _call(hv.biharmonic_numpy, u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v,
                           geom, 0.03, 0.01, 0.0, 600.0)
        assert (h_u * u * fu * geom["area_u"]).sum() + (h_v * v * fv * geom["area_v"]).sum() < 0


# ---------------------------------------------------------------- bound
def _ab3_energy(fn_tend, u, v, dt, nsteps, area_u, area_v):
    """AB3 (the model's explicit momentum step, Durran 2010) with the operator alone; kinetic energy per step."""
    hist, ke = [], []
    for n in range(nsteps):
        fu, fv = fn_tend(u, v)
        hist.insert(0, (fu, fv))
        hist = hist[:3]
        if len(hist) == 1:
            cu, cv = hist[0]
        elif len(hist) == 2:
            cu = 1.5 * hist[0][0] - 0.5 * hist[1][0]
            cv = 1.5 * hist[0][1] - 0.5 * hist[1][1]
        else:
            cu = (23 * hist[0][0] - 16 * hist[1][0] + 5 * hist[2][0]) / 12
            cv = (23 * hist[0][1] - 16 * hist[1][1] + 5 * hist[2][1]) / 12
        u, v = u + dt * cu, v + dt * cv
        ke.append(0.5 * ((u ** 2 * area_u).sum() + (v ** 2 * area_v).sum()))
    return np.array(ke)


def test_bound_keeps_ab3_stable_at_the_limit():
    # A background A_h 1e6× above the bound: every point is limited to Ah_max hrat, and AB3 with the operator alone
    # damps a random field (grid-scale checkerboard included) instead of blowing up.  Without the limit (dt 10×
    # smaller in the bound but the same step) it would grow.
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 7, 8
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    rng = np.random.default_rng(3)
    u = rng.normal(0, 0.3, (2, ny, nx + 1)) * wet_u
    v = rng.normal(0, 0.3, (2, ny + 1, nx)) * wet_v
    u[:, ::2, ::2] *= -1.0                                         # strong grid-scale content
    h = _ones(2, ny, nx, 20.0)
    dt = 600.0

    def tend(dt_bound):
        def f(u, v):
            fu, fv, *_ = _call(hv.biharmonic_numpy, u, v, *h, wet_u, wet_v, mask_q, open_u, open_v, geom,
                               0.0, 0.0, 1e20, dt_bound)
            return fu, fv
        return f

    _, _, ah_t, ah_q, n_lim = _call(hv.biharmonic_numpy, u, v, *h, wet_u, wet_v, mask_q, open_u, open_v, geom,
                                    0.0, 0.0, 1e20, dt)
    act_t = wet_u[:, :-1] | wet_u[:, 1:] | wet_v[:-1] | wet_v[1:]
    assert n_lim == 2 * (act_t.sum() + mask_q.sum())
    ke = _ab3_energy(tend(dt), u, v, dt, 300, geom["area_u"], geom["area_v"])
    assert np.all(np.isfinite(ke)) and ke[-1] < ke[0] and np.all(np.diff(ke[5:]) <= 1e-12 * ke[0])
    with np.errstate(over="ignore", invalid="ignore"):
        ke_bad = _ab3_energy(tend(dt / 10.0), u, v, dt, 300, geom["area_u"], geom["area_v"])
    assert not (np.isfinite(ke_bad[-1]) and ke_bad[-1] < ke[0])


def test_background_and_smagorinsky_coefficients():
    # Ah = max(AH, AH_VEL_SCALE Δ³, SMAG_BI_CONST Δ⁴ |S|) with Δ² = 2Δx²Δy²/(Δx²+Δy²) (MOM_hor_visc.F90:3446-3468)
    g = _stretched_grid()
    geom = hv.smag_geometry(g)
    ny, nx = 7, 8
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    u, v = np.zeros((1, ny, nx + 1)), np.zeros((1, ny + 1, nx))
    _, _, ah_t, ah_q, n = _call(hv.biharmonic_numpy, u, v, *_ones(1, ny, nx), wet_u, wet_v, mask_q, open_u,
                                open_v, geom, 0.03, 0.01, 0.0, 1.0)
    d2 = geom["delta2_t"]
    np.testing.assert_allclose(ah_t[0], 0.01 * d2 * np.sqrt(d2), rtol=1e-15)
    d2q = geom["delta2_q"]
    np.testing.assert_allclose(ah_q[0][mask_q], (0.01 * d2q * np.sqrt(d2q))[mask_q], rtol=1e-15)
    assert np.all(ah_q[0][~mask_q] == 0.0)
    rng = np.random.default_rng(1)
    u = rng.normal(0, 0.3, (1, ny, nx + 1)) * wet_u
    v = rng.normal(0, 0.3, (1, ny + 1, nx)) * wet_v
    _, _, ah_t, _, _ = _call(hv.biharmonic_numpy, u, v, *_ones(1, ny, nx), wet_u, wet_v, mask_q, open_u, open_v,
                             geom, 0.03, 0.0, 0.0, 1.0)
    _, _, nu_t, _, _ = hv.smagorinsky_numpy(u, v, *_ones(1, ny, nx), wet_u, wet_v, mask_q, open_u, open_v, geom,
                                            1.0, 0.0, 1.0)
    np.testing.assert_allclose(ah_t, 0.03 * d2 * nu_t, rtol=1e-14)   # C Δ⁴|S| = 0.03 Δ² (1 · Δ²|S|)


def test_bound_uniform_grid_value():
    # MOM6 BOUND_AH on a uniform grid: Ah_Max = 0.8 · 0.5 / (Δt · 32/Δ⁴) = 0.8 Δ⁴/(64 Δt); pyOCEAN × 6/11 (AB3)
    nx, ny, d, dt = 10, 9, 2000.0, 300.0
    g = create_cartesian_grid(nx, ny, d, d, 0.0)
    geom = hv.smag_geometry(g)
    wet_u, wet_v, mask_q = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    u, v = np.zeros((1, ny, nx + 1)), np.zeros((1, ny + 1, nx))
    _, _, ah_t, ah_q, _ = _call(hv.biharmonic_numpy, u, v, *_ones(1, ny, nx), wet_u, wet_v, mask_q, open_u,
                                open_v, geom, 0.0, 0.0, 1e30, dt)
    expect = 0.8 * (6.0 / 11.0) * d ** 4 / (64.0 * dt)
    np.testing.assert_allclose(ah_t[0, 2:-2, 2:-2], expect, rtol=1e-13)
    np.testing.assert_allclose(ah_q[0, 2:-2, 2:-2], expect, rtol=1e-13)


# ---------------------------------------------------------------- open faces
def _channel(nx=16, ny=14, d=1000.0):
    g = create_cartesian_grid(nx, ny, d, d, 1e-4)
    geom = hv.smag_geometry(g)
    wet_u, wet_v, _ = _closed_masks(ny, nx)
    open_u, open_v = _no_open(ny, nx)
    open_u[:, 0] = open_u[:, -1] = True                    # open west and east edges
    wet_u = wet_u | open_u                                 # the caller marks open faces wet (Momentum)
    return g, geom, wet_u, wet_v, _mask_q(wet_u, wet_v), open_u, open_v


@pytest.mark.parametrize("corners", ["none", "mom6"])
def test_linear_shear_through_open_edges_has_no_interior_tendency(corners):
    # u = a y through the open west and east faces: the open-face normal velocity enters the tension, Del2u = 0 on
    # the open faces (OBC_ZERO_BIHARMONIC) and in the interior, so rows away from the walls feel no u tendency,
    # right up to the open edges; the open faces themselves get none.  The shear strain at the open-edge corners
    # decides v: 'mom6' (computed, = a for v = vt = 0) keeps Del2v = 0 there and v feels nothing; 'none'
    # (MOM6 OBC_STRAIN_ZERO, D_S = 0 at those corners) gives Del2v ≠ 0 in the edge column, so the v next to the
    # edges is accelerated.
    g, geom, wet_u, wet_v, mask_q, open_u, open_v = _channel()
    ny, nx = wet_u.shape[0], wet_v.shape[1]
    y_t = (np.arange(ny) + 0.5) * 1000.0
    u = (0.2 + 1e-5 * y_t)[None, :, None] * np.ones((2, ny, nx + 1)) * wet_u
    v = np.zeros((2, ny + 1, nx))
    kw = {}
    if corners == "mom6":
        flags = obc.tangential_corners(open_u, open_v, ("west", "east"))
        vt = np.zeros((2, ny + 1, nx + 1))
        kw = dict(obc_q=flags, obc_strain=hv.open_corner_strain(u, v, vt, flags, *_ones(2, ny, nx), geom))
    fu, fv, *_ = _call(hv.biharmonic_numpy, u, v, *_ones(2, ny, nx), wet_u, wet_v, mask_q, open_u, open_v, geom,
                       0.03, 0.01, 0.0, 600.0, **kw)
    assert np.all(fu[:, :, 0] == 0.0) and np.all(fu[:, :, -1] == 0.0)
    scale = np.abs(fu).max()
    assert scale > 0
    np.testing.assert_allclose(fu[:, 4:-4], 0.0, atol=1e-12 * scale)
    if corners == "mom6":
        np.testing.assert_allclose(fv[:, 4:-4], 0.0, atol=1e-12 * scale)
    else:
        np.testing.assert_allclose(fv[:, 4:-4, 3:-3], 0.0, atol=1e-12 * scale)
        assert np.abs(fv[:, 4:-4, :2]).max() > 1e-3 * scale


def test_open_face_is_not_a_wall():
    # A uniform through-flow: with the open faces marked wet it has no tendency anywhere; with them treated as
    # walls (the error fixed for the Laplacian on 2026-11-03) the boundary cells would see (u₁ − 0)/Δx.
    g, geom, wet_u, wet_v, mask_q, open_u, open_v = _channel()
    ny, nx = wet_u.shape[0], wet_v.shape[1]
    u = np.full((1, ny, nx + 1), 0.4) * wet_u
    v = np.zeros((1, ny + 1, nx))
    fu, fv, *_ = _call(hv.biharmonic_numpy, u, v, *_ones(1, ny, nx), wet_u, wet_v, mask_q, open_u, open_v, geom,
                       0.03, 0.01, 0.0, 600.0)
    assert np.all(fu == 0.0) and np.all(fv == 0.0)
    wall_u = wet_u & ~open_u
    fu_w, *_ = _call(hv.biharmonic_numpy, u * wall_u, v, *_ones(1, ny, nx), wall_u, wet_v, _mask_q(wall_u, wet_v),
                     open_u, open_v, geom, 0.03, 0.01, 0.0, 600.0)
    assert np.abs(fu_w[:, :, 1:3]).max() > 0


# ---------------------------------------------------------------- Numba = NumPy, threads
def _random_case(seed=4, with_obc=False):
    g, geom, wet_u, wet_v, mask_q, open_u, open_v = _channel(nx=11, ny=9)
    ny, nx = wet_u.shape[0], wet_v.shape[1]
    wet_u[3, 5] = False                                        # an island face
    wet_v[4, 6] = False
    mask_q = _mask_q(wet_u, wet_v)
    rng = np.random.default_rng(seed)
    N = 3
    u = rng.normal(0, 0.4, (N, ny, nx + 1)) * wet_u
    v = rng.normal(0, 0.4, (N, ny + 1, nx)) * wet_v
    h_t = rng.uniform(0.0, 30.0, (N, ny, nx))
    h_u = rng.uniform(0.0, 30.0, (N, ny, nx + 1))
    h_v = rng.uniform(0.0, 30.0, (N, ny + 1, nx))
    h_u[0, 2, 2] = 0.0
    kw = {}
    if with_obc:
        flags = obc.tangential_corners(open_u, open_v, ("west", "east"))
        vt = rng.normal(0, 0.3, (N, ny + 1, nx + 1))
        kw = dict(obc_q=flags, obc_strain=hv.open_corner_strain(u, v, vt, flags, h_t, h_u, h_v, geom))
    return (u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom), kw


@pytest.mark.parametrize("with_obc", [False, True])
@pytest.mark.parametrize("threads", [1, 5])
def test_numba_equals_numpy_bit_for_bit(with_obc, threads):
    args, kw = _random_case(with_obc=with_obc)
    old = _THREADS[0]
    try:
        set_threads(threads)
        a = hv.biharmonic(*args, 0.03, 0.01, 1e3, 600.0, H_NEGLECT, **kw)
    finally:
        set_threads(old)
    b = hv.biharmonic_numpy(*args, 0.03, 0.01, 1e3, 600.0, H_NEGLECT, **kw)
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(x, y)
    assert a[4] == b[4]
    assert all(np.all(np.isfinite(x)) for x in a[:4])


def test_numba_counts_limited_points_as_numpy():
    args, kw = _random_case(seed=9)
    a = hv.biharmonic(*args, 0.03, 0.01, 1e16, 600.0, H_NEGLECT)
    b = hv.biharmonic_numpy(*args, 0.03, 0.01, 1e16, 600.0, H_NEGLECT)
    assert a[4] == b[4] > 0


def test_mom6_corners_change_only_the_edge_columns():
    args, kw = _random_case(with_obc=True)
    on = hv.biharmonic_numpy(*args, 0.03, 0.01, 0.0, 600.0, H_NEGLECT, **kw)
    off = hv.biharmonic_numpy(*args, 0.03, 0.01, 0.0, 600.0, H_NEGLECT)
    assert not np.array_equal(on[1][:, 1:-1, 0], off[1][:, 1:-1, 0])     # v next to the west edge feels them
    np.testing.assert_array_equal(on[1][:, :, 4:-4], off[1][:, :, 4:-4])  # four cells in: unchanged
    np.testing.assert_array_equal(on[0][:, :, 5:-5], off[0][:, :, 5:-5])
