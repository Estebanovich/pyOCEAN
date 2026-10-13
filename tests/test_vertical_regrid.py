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


def _hybrid(H, eta, h, rho, targets, dz=DELR, max_h=None, h_min=rg.H_MIN, keep=False):
    n = h.size
    zw = np.empty(n + 1)
    mh = np.full(n, np.inf) if max_h is None else max_h
    rg.hycom1_column(H, eta, h, rho, targets, dz, mh, h_min, keep, zw)
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


def test_hybrid_ignores_the_stale_density_of_a_massless_bed_layer():
    # gravity-current head: 0.7 m of dense water (layer 1) arrived over a massless bed layer that
    # still holds the light water of the start.  Monotonising with that light value would make the
    # whole column light and dissolve the current; massless layers take the density of the layer
    # with mass above them (HYCOM hybgen.F90:904-925)
    n = 10
    rho_l, rho_d = 1020.0, 1025.0
    targets = np.linspace(rho_l - 1.0, rho_d + 1.0, n + 1)
    h = np.full(n, rg.H_MIN)
    h[1], h[8], h[9] = 0.7, 19.3 - 7 * rg.H_MIN - 0.01, 0.01
    rho = np.full(n, rho_l)
    rho[1:8] = rho_d                                                    # dense current and vanished layers above it
    zw = _hybrid(20.0, 0.0, h, rho, targets, dz=np.full(n, 0.01))
    dense_top = -20.0 + rg.H_MIN + 0.7
    assert dense_top - zw[n - 2] < 0.1 * 0.7       # < 10 % of the dense water goes into the light layer (was 100 %)


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


def _window_targets(H, n=10):
    """Isopycnals crowded towards the surface (depths H s², s = 0 … 1), so they differ from uniform σ."""
    return 1024.0 + 0.02 * H * np.linspace(0.0, 1.0, n + 1) ** 2


def _window_case(H, d_sigma=50.0, d_hybrid=100.0, n=10):
    """Hybrid, σ and z* interfaces of a linearly stratified column of depth H (η = 0)."""
    zw_t = np.linspace(-H, 0.0, n + 1)
    zc = 0.5 * (zw_t[1:] + zw_t[:-1])
    h, rho = np.diff(zw_t), 1024.0 - 0.02 * zc
    dz = np.full(n, 2.0)                                                  # z* floor: top 20 m
    targets = _window_targets(H, n)
    hyb, sig, zs = np.empty(n + 1), np.empty(n + 1), np.empty(n + 1)
    rg.hycom1_column(H, 0.0, h, rho, targets, dz, np.full(n, np.inf), rg.H_MIN, False, hyb)
    rg.sigma_column(H, 0.0, np.full(n, 1.0 / n), rg.H_MIN, sig)
    rg.zstar_column(H, 0.0, dz, rg.H_MIN, zs)
    out = np.empty(n + 1)
    rg.window_shallow_sigma(hyb, sig, zs, H, H, d_sigma, d_hybrid, rg.H_MIN, out)
    return hyb, sig, zs, out


def test_window_is_sigma_in_shallow_water_and_hybrid_in_deep_water():
    hyb, sig, _, out = _window_case(40.0)                               # D ≤ d_sigma
    np.testing.assert_array_equal(out, sig)
    hyb, sig, _, out = _window_case(120.0)                              # D ≥ d_hybrid
    np.testing.assert_array_equal(out, hyb)


def test_window_keeps_the_isopycnals_that_fit_in_the_window():
    H = 75.0                                                            # q = 0.5
    hyb, sig, zs, out = _window_case(H)
    q = 0.5
    lo, hi = q * (-H) + (1 - q) * sig, q * zs + (1 - q) * sig
    assert ((out >= lo - 1e-12) & (out <= hi + 1e-12)).all()
    inside = (hyb > lo) & (hyb < hi)
    inside[[0, -1]] = False
    # depths: hybrid 2, 4, 6.75, 12, … (z* floor, then H s²), window tops 4.75, 9.5, …: 3 deep
    # interfaces stay on their isopycnal, 6 shallow ones are clipped (a position blend would move all)
    assert inside.sum() == 3 and (~inside[1:-1]).sum() == 6
    np.testing.assert_allclose(out[inside], hyb[inside], atol=1e-12)    # those stay on their isopycnal
    clipped = ~inside
    clipped[[0, -1]] = False
    np.testing.assert_allclose(out[clipped], np.clip(hyb, lo, hi)[clipped], atol=1e-12)
    assert (np.diff(out) > 0).all()
    np.testing.assert_allclose(out[-1] - out[0], H, rtol=1e-14)


@pytest.mark.parametrize("H, other", [(50.0 + 1e-6, "sig"), (100.0 - 1e-6, "hyb")])
def test_window_is_continuous_at_both_ends_of_the_transition(H, other):
    # no jump in the grid when the total depth crosses d_sigma or d_hybrid (tide, wetting–drying)
    hyb, sig, _, out = _window_case(H)
    ref = sig if other == "sig" else hyb
    assert np.abs(out - ref).max() < 1e-4


def test_regrid_columns_hybrid_with_transition_matches_the_window():
    hyb, sig, zs, out = _window_case(75.0)
    n = 10
    zw_t = np.linspace(-75.0, 0.0, n + 1)
    zc = 0.5 * (zw_t[1:] + zw_t[:-1])
    h3 = np.diff(zw_t)[:, None, None].repeat(2, 1)
    r3 = (1024.0 - 0.02 * zc)[:, None, None].repeat(2, 1)
    params = dict(dz_top=np.full(n, 2.0), rho_target=_window_targets(75.0, n),
                  dsigma=np.full(n, 1.0 / n), d_sigma=50.0, d_hybrid=100.0, keep_isopycnal=False)
    zw3 = rg.regrid_columns("hybrid", np.full((2, 1), 75.0), np.zeros((2, 1)), h3, r3, params, np.ones((2, 1), bool))
    np.testing.assert_allclose(zw3[:, 0, 0], out, atol=1e-12)


def test_light_top_targets_keep_the_mixed_layer_in_z():
    # 40 m mixed layer over a linear pycnocline: with targets that start at the surface density the
    # first interface sinks to the base of the mixed layer (HYCOM1 lumps it); targets lighter than
    # any water (OM4 style) keep the top interfaces on their z* floor
    H, n = 201.0, 10
    zw_t = np.linspace(-H, 0.0, n + 1)
    zc = 0.5 * (zw_t[1:] + zw_t[:-1])
    rho = np.where(zc > -40.0, 1024.0, 1024.0 + 0.02 * (-40.0 - zc))
    h = np.diff(zw_t)
    t0 = rg.target_densities(rho, n)
    t3 = rg.target_densities(rho, n, n_light=3)
    assert t3.shape == (n + 1,) and (np.diff(t3) > 0).all() and (t3[:3] < rho.min()).all()
    np.testing.assert_allclose(t3[3:], np.linspace(rho.min(), rho.max(), n - 2))
    np.testing.assert_array_equal(t0, rg.target_densities(rho, n, n_light=0))
    z0 = _hybrid(H, 0.0, h, rho, t0)[::-1]                              # surface first
    z3 = _hybrid(H, 0.0, h, rho, t3)[::-1]
    assert z0[1] <= -35.0                                               # lumped mixed layer
    np.testing.assert_allclose(z3[1:4], -np.cumsum(DELR)[:3], atol=1e-12)   # on the z* floor


def test_target_densities_rejects_bad_n_light():
    with pytest.raises(ValueError):
        rg.target_densities(np.array([1024.0, 1025.0]), 4, n_light=5)


def test_sigma_rule_accepts_floodable_columns_and_hybrid_only_in_its_sigma_zone():
    H = np.array([[-0.5, 10.0]])
    eta = np.array([[0.7, 0.0]])                                        # flooded land: D = 0.2 m
    h3 = np.ones((10, 1, 2))
    wet = np.ones((1, 2), bool)
    zw = rg.regrid_columns("sigma", H, eta, h3, None, dict(dsigma=np.full(10, 0.1)), wet)
    np.testing.assert_allclose(zw[-1] - zw[0], H + eta, rtol=1e-12)
    r3 = np.full((10, 1, 2), 1025.0)
    base = dict(dz_top=np.ones(10), rho_target=np.linspace(1024, 1026, 11), dsigma=np.full(10, 0.1))
    with pytest.raises(ValueError):
        rg.regrid_columns("hybrid", H, eta, h3, r3, base, wet)                       # no σ zone
    zw = rg.regrid_columns("hybrid", H, eta, h3, r3, dict(base, d_sigma=5.0, d_hybrid=20.0), wet)
    np.testing.assert_allclose(np.diff(zw[:, 0, 0]), 0.02, rtol=1e-12)               # pure σ there
    with pytest.raises(ValueError):
        rg.regrid_columns("zstar", H, eta, h3, None, base, wet)


def test_regrid_columns_hybrid_matches_single_column():
    h, rho = _linear_column()
    ny, nx = 2, 3
    H = np.full((ny, nx), 201.0)
    eta = np.zeros((ny, nx))
    h3 = np.repeat(h[:, None, None], ny * nx, 1).reshape(10, ny, nx)
    r3 = np.repeat(rho[:, None, None], ny * nx, 1).reshape(10, ny, nx)
    targets = np.linspace(1024.0, 1028.0, 11)
    params = dict(dz_top=DELR, rho_target=targets, max_h_top=np.full(10, np.inf),
                  dsigma=np.full(10, 0.1), d_sigma=0.0, d_hybrid=0.0, h_min=rg.H_MIN, keep_isopycnal=False)
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


def test_hybrid_flooded_land_deeper_than_d_sigma_stays_sigma():
    # final review of Plan B: land (H ≤ 0) flooded deeper than d_sigma (storm surge) must stay pure σ,
    # not raise mid-run or need a hybrid grid with no z* floor
    H = np.array([[-0.5, 10.0]])
    eta = np.array([[3.0, 0.0]])                                        # flooded land: D = 2.5 m > d_sigma
    h3 = np.ones((10, 1, 2))
    r3 = np.full((10, 1, 2), 1025.0)
    params = dict(dz_top=np.ones(10), rho_target=np.linspace(1024, 1026, 11), dsigma=np.full(10, 0.1),
                  d_sigma=1.0, d_hybrid=2.0)
    zw = rg.regrid_columns("hybrid", H, eta, h3, r3, params, np.ones((1, 2), bool))
    np.testing.assert_allclose(np.diff(zw[:, 0, 0]), 0.25, rtol=1e-12)


# ---------------------------------------------------------------- B2c: isopycnal layer preservation
def _consistent_column(n=10, H=200.0, seed=0):
    """Bed-first layers whose densities lie inside their own target brackets (surface-first targets)."""
    rng = np.random.default_rng(seed)
    targets = np.linspace(1020.0, 1030.0, n + 1)                        # surface first
    mid_top = 0.5 * (targets[:-1] + targets[1:])                        # layer j (surface first)
    h = rng.uniform(0.5, 1.5, n)
    h *= H / h.sum()
    return h, mid_top[::-1].copy(), targets                              # h, ρ bed first


def test_keep_isopycnal_leaves_consistent_layers_exactly_in_place():
    h, rho, targets = _consistent_column()
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=True)
    np.testing.assert_allclose(zw, -200.0 + np.concatenate([[0.0], np.cumsum(h)]), rtol=0, atol=1e-11)
    moved = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=False)
    assert np.abs(moved - zw).max() > 1.0                                # P1M would move them


def test_keep_isopycnal_regrids_only_the_interfaces_of_an_inconsistent_layer():
    h, rho, targets = _consistent_column()
    rho[4] = 1021.2                                                      # bed-first layer 4 out of its bracket
    zin = -200.0 + np.concatenate([[0.0], np.cumsum(h)])
    p1m = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=False)
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=True)
    other = [k for k in range(11) if k not in (4, 5)]
    np.testing.assert_allclose(zw[other], zin[other], rtol=0, atol=1e-11)
    for k in (4, 5):                                                     # P1M position, between the kept 3 and 6
        np.testing.assert_allclose(zw[k], np.clip(p1m[k], zin[3], zin[6]), rtol=0, atol=1e-11)
    assert (np.diff(zw) > 0).all()


def test_keep_isopycnal_counts_massless_layers_as_consistent():
    # a vanished layer with a stale density between two consistent thick ones does not unpin them
    h, rho, targets = _consistent_column()
    h[3] = rg.H_MIN
    rho[3] = 1029.9                                                      # far outside its bracket
    h *= 200.0 / h.sum()
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=True)
    np.testing.assert_allclose(zw, -200.0 + np.concatenate([[0.0], np.cumsum(h)]), rtol=0, atol=1e-11)


def test_keep_isopycnal_still_applies_the_zstar_floor_and_keeps_order():
    # consistent layers squeezed against the surface: the z* floor pushes the kept interfaces
    # down; the column stays ordered and adds up to D
    h, rho, targets = _consistent_column()
    h[-3:] = [0.2, 0.2, 0.2]                                             # three thin top layers
    h *= 200.0 / h.sum()
    dz = np.full(10, 5.0)                                                # floor: 5 m per layer from the top
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=dz, keep=True)
    assert (np.diff(zw) > 0).all() and zw[0] == -200.0 and zw[-1] == 0.0
    np.testing.assert_allclose(zw[::-1][1:4], -np.cumsum(dz)[:3], atol=1e-11)


def test_regrid_columns_defaults_to_keep_isopycnal():
    h, rho, targets = _consistent_column()
    h3, r3 = h[:, None, None].repeat(2, 2), rho[:, None, None].repeat(2, 2)
    base = dict(dz_top=np.full(10, 1e-6), rho_target=targets)
    zw = rg.regrid_columns("hybrid", np.full((1, 2), 200.0), np.zeros((1, 2)), h3, r3, base, np.ones((1, 2), bool))
    np.testing.assert_allclose(zw[:, 0, 0], _hybrid(200.0, 0.0, h, rho, targets, dz=base["dz_top"], keep=True),
                               rtol=0, atol=1e-12)
    zw0 = rg.regrid_columns("hybrid", np.full((1, 2), 200.0), np.zeros((1, 2)), h3, r3,
                            dict(base, keep_isopycnal=False), np.ones((1, 2), bool))
    np.testing.assert_allclose(zw0[:, 0, 1], _hybrid(200.0, 0.0, h, rho, targets, dz=base["dz_top"], keep=False),
                               rtol=0, atol=1e-12)


def test_keep_isopycnal_interfaces_stay_inside_the_shallow_window():
    # review focus 2 (guard: passes before Task 1 and must still pass after): with q = 0.5 every
    # interface, kept or not, lies between L_k = q(−H) + (1−q)σ_k and U_k = q z*_k + (1−q)σ_k
    H = 7.5
    h, rho, targets = _consistent_column(H=H)
    params = dict(dz_top=np.full(10, 0.2), rho_target=targets, dsigma=np.full(10, 0.1), d_sigma=5.0, d_hybrid=10.0)
    zw = rg.regrid_columns("hybrid", np.full((1, 1), H), np.zeros((1, 1)), h[:, None, None], rho[:, None, None],
                           params, np.ones((1, 1), bool))[:, 0, 0]
    sig, zs = np.empty(11), np.empty(11)
    rg.sigma_column(H, 0.0, params["dsigma"], rg.H_MIN, sig)
    rg.zstar_column(H, 0.0, params["dz_top"], rg.H_MIN, zs)
    q = 0.5
    lo, hi = q * (-H) + (1 - q) * sig, q * zs + (1 - q) * sig
    assert ((zw >= lo - 1e-9) & (zw <= hi + 1e-9)).all()
    assert (np.diff(zw) > 0).all()


# ---------------------------------------------------------------- regrid on a source grid (initial state, MOM6)
def test_hybrid_regrids_a_finer_source_column_onto_fewer_layers():
    # MOM6 Z_INIT_ALE_REMAPPING regrids on the data's own levels (MOM_state_initialization.F90:2928):
    # 40 source layers of a linear profile, 10 target interfaces → the same isopycnal depths as a 10-layer source
    hs, rhos = _linear_column(n=40)
    z_t = -np.array([0.0, 30, 60, 80, 100, 120, 140, 160, 170, 178, 201.0])
    targets = 1024.0 - 0.02 * z_t
    zw = np.empty(11)
    rg.hycom1_column(201.0, 0.0, hs, rhos, targets, DELR, np.full(10, np.inf), rg.H_MIN, False, zw)
    np.testing.assert_allclose(zw[::-1][1:-1], z_t[1:-1], atol=1e-9)
    zw3 = rg.regrid_columns("hybrid", np.full((1, 1), 201.0), np.zeros((1, 1)), hs[:, None, None],
                            rhos[:, None, None], dict(dz_top=DELR, rho_target=targets, keep_isopycnal=False),
                            np.ones((1, 1), bool), n_out=10)
    assert zw3.shape == (11, 1, 1)
    np.testing.assert_allclose(zw3[:, 0, 0], zw, atol=1e-12)


def test_snap_to_jumps_splits_the_layer_that_straddles_a_density_jump():
    # a two-layer source (jump at −116 m) and a model grid whose layer 2 spans −120 … −100 m: the nearer
    # interface (−120) moves onto the jump, so no model layer holds both water masses; small jumps are left
    zw_src = np.array([-200.0, -150.0, -116.0, -80.0, 0.0])
    rho_src = np.array([1028.0, 1028.0, 1026.0, 1026.0])                    # bed first
    zw = np.array([-200.0, -120.0, -100.0, -50.0, 0.0])
    out = rg.snap_to_jumps(zw[:, None, None].copy(), zw_src[:, None, None], rho_src[:, None, None], 0.5)
    np.testing.assert_allclose(out[:, 0, 0], [-200.0, -116.0, -100.0, -50.0, 0.0])
    keep = rg.snap_to_jumps(zw[:, None, None].copy(), zw_src[:, None, None], rho_src[:, None, None], 5.0)
    np.testing.assert_allclose(keep[:, 0, 0], zw)
