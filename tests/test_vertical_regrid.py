"""
Tests for core/vertical/regrid.py — grid-generation rules per column (spec §3):
z* (Adcroft & Campin 2004; MOM6 coord_zlike.F90), σ (coord_sigma.F90) and the
hybrid z*–isopycnal HYCOM1 rule (coord_hycom.F90).  Interfaces zw are heights,
bed first (zw[0] = −H, zw[N] = η).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.vertical import regrid as rg

DELR = np.array([1.0, 2.0, 3.0, 5.0, 8.0, 12.0, 20.0, 30.0, 50.0, 70.0])   # surface first, Σ = 201


def _zstar(H, eta, dz=DELR, h_min=rg.H_MIN):
    zw = np.empty(dz.size + 1)
    rg.zstar_column(H, eta, dz, h_min, zw)
    return zw


def test_zstar_full_depth_equals_scaled_levels():
    H, eta = 201.0, 0.5
    zw = _zstar(H, eta)
    s = (H + eta) / H
    expected = eta - s * np.r_[0.0, np.cumsum(DELR)]           # surface first
    np.testing.assert_allclose(zw, expected[::-1], rtol=1e-14, atol=1e-12)


def test_zstar_shallow_column_collapses_bottom_layers():
    H, eta = 40.0, 0.0
    zw = _zstar(H, eta)
    h = np.diff(zw)
    assert zw[0] == -H and zw[-1] == eta
    np.testing.assert_allclose(h.sum(), H + eta, rtol=1e-14)
    assert (h >= rg.H_MIN - 1e-12).all()                               # round-off of z ~ 40 m
    np.testing.assert_allclose(h[-4:][::-1], DELR[:4], rtol=1e-12)      # top 4 levels unchanged (Σ = 11 m)
    # nominal levels 30, 50, 70 m lie below the bed: collapsed to h_min there
    assert np.count_nonzero(np.isclose(h, rg.H_MIN, rtol=0, atol=1e-12)) == 3
    np.testing.assert_allclose(h[3], 9.0 - 3 * rg.H_MIN, rtol=1e-12)    # truncated 20 m level (partial cell)


def test_zstar_min_thickness_capped_by_depth():
    zw = _zstar(0.005, 0.0)                                             # 5 mm of water, 10 layers
    h = np.diff(zw)
    np.testing.assert_allclose(h.sum(), 0.005, rtol=1e-12)
    assert (h > 0).all()


def test_sigma_reproduces_fractions():
    ds = np.array([0.1, 0.2, 0.3, 0.4])                                 # bed first
    zw = np.empty(5)
    rg.sigma_column(100.0, 2.0, ds, rg.H_MIN, zw)
    np.testing.assert_allclose(np.diff(zw), 102.0 * ds, rtol=1e-14)
    assert zw[0] == -100.0 and zw[-1] == 2.0


def _hybrid(H, eta, h, rho, targets, dz=DELR, max_h=None, h_min=rg.H_MIN):
    n = h.size
    zw = np.empty(n + 1)
    mh = np.full(n, np.inf) if max_h is None else max_h
    rg.hycom1_column(H, eta, h, rho, targets, dz, mh, h_min, zw)
    return zw


def _linear_column(H=201.0, n=10, rho_top=1024.0, drho_dz=0.02):
    zw = np.linspace(-H, 0.0, n + 1)
    zc = 0.5 * (zw[1:] + zw[:-1])
    return np.diff(zw), rho_top - drho_dz * zc                          # bed first, ρ grows downward


def test_hybrid_interfaces_on_target_densities():
    h, rho = _linear_column()
    # targets (surface first) whose z positions lie below the z* floor
    z_t = -np.array([0.0, 30, 60, 80, 100, 120, 140, 160, 170, 178, 201.0])   # interior cells only
    targets = 1024.0 - 0.02 * z_t
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    np.testing.assert_allclose(zw[::-1][1:-1], z_t[1:-1], atol=1e-9)    # interior interfaces


def test_hybrid_respects_zstar_floor():
    h, rho = _linear_column()
    targets = np.full(11, 1024.0)                                       # all targets lighter than the column
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    zstar = np.empty(11)
    rg.zstar_column(201.0, 0.0, DELR, rg.H_MIN, zstar)
    np.testing.assert_allclose(zw, zstar, atol=1e-9)                    # pushed down to the z* floor


def test_hybrid_unstratified_column_does_not_fail():
    h = np.full(10, 20.1)
    rho = np.full(10, 1025.0)                                           # well mixed
    targets = np.linspace(1024.0, 1027.0, 11)
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    assert np.isfinite(zw).all() and (np.diff(zw) > 0).all()


def test_hybrid_targets_out_of_range_keep_layers_separate():
    h, rho = _linear_column()
    targets = np.linspace(1000.0, 1100.0, 11)                           # far outside the column range
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    assert (np.diff(zw) >= rg.H_MIN * (1 - 1e-12)).all()
    np.testing.assert_allclose(np.diff(zw).sum(), 201.0, rtol=1e-14)


def test_hybrid_nonmonotonic_profile_is_monotonized():
    h, rho = _linear_column()
    rho[3] += 0.5                                                       # a dense blob above lighter water
    targets = np.linspace(1024.0, 1028.0, 11)
    zw = _hybrid(201.0, 0.0, h, rho, targets)
    assert (np.diff(zw) > 0).all()
    rho_m = rho.copy()                                                  # monotonized by hand, bed upward
    for k in range(1, rho_m.size):
        rho_m[k] = min(rho_m[k], rho_m[k - 1])
    np.testing.assert_allclose(zw, _hybrid(201.0, 0.0, h, rho_m, targets), rtol=0, atol=1e-12)


def test_hybrid_maximum_layer_thickness():
    h, rho = _linear_column()
    targets = np.r_[1024.0, np.full(9, 1028.0), 1028.1]                 # would put everything at the bed
    zw = _hybrid(201.0, 0.0, h, rho, targets, max_h=np.full(10, 25.0))
    assert (np.diff(zw) <= 25.0 + 1e-9).all()
    np.testing.assert_allclose(np.diff(zw).sum(), 201.0, rtol=1e-14)


def test_blend_shallow_sigma_endpoints_and_middle():
    a = np.array([-10.0, -6.0, 0.0])
    b = np.array([-10.0, -4.0, 0.0])
    zw = np.empty(3)
    rg.blend_shallow_sigma(a, b, 5.0, 10.0, 30.0, zw)
    np.testing.assert_allclose(zw, b)                                   # D ≤ d_sigma: pure σ
    rg.blend_shallow_sigma(a, b, 40.0, 10.0, 30.0, zw)
    np.testing.assert_allclose(zw, a)                                   # D ≥ d_hybrid: pure hybrid
    rg.blend_shallow_sigma(a, b, 20.0, 10.0, 30.0, zw)
    np.testing.assert_allclose(zw, 0.5 * (a + b))


def test_regrid_columns_hybrid_matches_single_column():
    h, rho = _linear_column()
    ny, nx = 2, 3
    H = np.full((ny, nx), 201.0)
    eta = np.zeros((ny, nx))
    h3 = np.repeat(h[:, None, None], ny * nx, 1).reshape(10, ny, nx)
    r3 = np.repeat(rho[:, None, None], ny * nx, 1).reshape(10, ny, nx)
    targets = np.linspace(1024.0, 1028.0, 11)
    params = dict(dz_top=DELR, rho_target=targets, max_h_top=np.full(10, np.inf),
                  dsigma=np.full(10, 0.1), d_sigma=0.0, d_hybrid=0.0, h_min=rg.H_MIN)
    zw3 = rg.regrid_columns("hybrid", H, eta, h3, r3, params, np.ones((ny, nx), bool))
    np.testing.assert_allclose(zw3[:, 1, 2], _hybrid(201.0, 0.0, h, rho, targets), rtol=0, atol=1e-12)


@pytest.mark.parametrize("D", [0.002, 0.01, 0.05, 0.1, 0.5, 1.0])
def test_sigma_stretched_shallow_columns_keep_positive_layers(D):
    # Song–Haidvogel σ in very shallow water: the bed-up inflation to h_min must not leave a
    # negative surface layer (wetting–drying zone)
    from core.grid.sigma_levels import stretched_sigma
    ds = stretched_sigma(50, 5.0)
    H = 10.0
    zw = np.empty(51)
    rg.sigma_column(H, D - H, ds, rg.H_MIN, zw)
    h = np.diff(zw)
    assert (h >= min(rg.H_MIN, D / 50) * (1 - 1e-9)).all()
    np.testing.assert_allclose(h.sum(), D, rtol=1e-12)


def test_kernels_calling_remap_are_not_disk_cached():
    # Numba's on-disk cache only checks the caller's own file: a cached regrid kernel that
    # calls remap._bound_edges would keep a stale copy after remap.py changes
    for f in (rg._p1m_h2_positions, rg.hycom1_column, rg._regrid_columns):
        assert type(f._cache).__name__ == "NullCache", f.__name__


def _grid_inputs(ny=2, nx=2, n=10):
    h, rho = _linear_column(n=n)
    H = np.full((ny, nx), 201.0)
    h3 = np.repeat(h[:, None, None], ny * nx, 1).reshape(n, ny, nx)
    r3 = np.repeat(rho[:, None, None], ny * nx, 1).reshape(n, ny, nx)
    return H, np.zeros((ny, nx)), h3, r3, np.ones((ny, nx), bool)


@pytest.mark.parametrize("rule, drop", [("hybrid", "rho"), ("hybrid", "rho_target"), ("hybrid", "dz_top"),
                                        ("zstar", "dz_top"), ("sigma", "dsigma")])
def test_regrid_columns_requires_its_parameters(rule, drop):
    H, eta, h3, r3, wet = _grid_inputs()
    params = dict(dz_top=DELR, dsigma=np.full(10, 0.1), rho_target=np.linspace(1024, 1028, 11))
    rho = r3
    if drop == "rho":
        rho = None
    else:
        params.pop(drop)
    with pytest.raises(ValueError, match=drop):
        rg.regrid_columns(rule, H, eta, h3, rho, params, wet)


@pytest.mark.parametrize("bad", ["H_zero", "H_nan", "D_negative", "rho_nan"])
def test_regrid_columns_rejects_invalid_wet_columns(bad):
    H, eta, h3, r3, wet = _grid_inputs()
    params = dict(dz_top=DELR, rho_target=np.linspace(1024, 1028, 11))
    if bad == "H_zero":
        H[1, 1] = 0.0
    elif bad == "H_nan":
        H[1, 1] = np.nan
    elif bad == "D_negative":
        eta[1, 1] = -202.0
    else:
        r3[4, 1, 1] = np.nan
    with pytest.raises(ValueError):
        rg.regrid_columns("hybrid", H, eta, h3, r3, params, wet)
    wet[1, 1] = False                                                   # the same column dry: fine
    assert np.isfinite(rg.regrid_columns("hybrid", H, eta, h3, r3, params, wet)[:, 0, 0]).all()


def test_p1m_positions_defined_when_no_cell_matches():
    # a NaN density matches no cell: positions must stay defined (the input grid), not garbage
    h = np.full(5, 2.0)
    rho = np.array([1024.0, 1025.0, np.nan, 1027.0, 1028.0])
    x = np.full(6, -999.0)
    rg._p1m_h2_positions(h, rho, np.linspace(1024.0, 1028.0, 6), x)
    assert np.isfinite(x).all() and (x >= 0).all() and (x <= 10.0).all()
