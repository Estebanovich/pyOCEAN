"""
Tests for core/vertical/remap.py — conservative 1-D vertical remapping
(White & Adcroft 2008; MOM6 MOM_remapping.F90 remapping_core_h).
Columns are ordered with k = 0 at the bed; any consistent order works.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.vertical import remap as rm


def _remap(h0, u0, h1, scheme):
    u1 = np.empty(len(h1))
    rm.remap_column(np.asarray(h0, float), np.asarray(u0, float), np.asarray(h1, float), scheme, u1)
    return u1


@pytest.mark.parametrize("scheme", [rm.PCM, rm.PLM, rm.PPM_H4])
def test_identity_when_grid_unchanged(scheme):
    rng = np.random.default_rng(0)
    h = rng.uniform(0.5, 3.0, 12)
    u = rng.normal(10, 2, 12)
    np.testing.assert_allclose(_remap(h, u, h, scheme), u, rtol=1e-14)


@pytest.mark.parametrize("scheme", [rm.PCM, rm.PLM, rm.PPM_H4])
def test_conservation_to_roundoff(scheme):
    rng = np.random.default_rng(1)
    h0 = rng.uniform(0.0, 4.0, 20)
    h0[[3, 4, 11]] = 0.0                                   # vanished source layers
    u0 = rng.normal(15, 3, 20)
    h1 = rng.uniform(0.1, 2.0, 15)
    h1 *= h0.sum() / h1.sum()
    u1 = _remap(h0, u0, h1, scheme)
    assert abs((h1 * u1).sum() - (h0 * u0).sum()) <= 1e-12 * abs((h0 * u0).sum())


def test_pcm_piecewise_constant_values():
    # source: two cells (1 m of 1.0, 1 m of 3.0); target: one 2 m cell and the reverse split
    np.testing.assert_allclose(_remap([1, 1], [1.0, 3.0], [2.0], rm.PCM), [2.0])
    np.testing.assert_allclose(_remap([1, 1], [1.0, 3.0], [0.5, 1.5], rm.PCM), [1.0, (0.5 + 3.0) / 1.5])


def test_vanished_target_takes_local_value():
    u1 = _remap([1.0, 1.0], [1.0, 3.0], [1.0, 0.0, 1.0], rm.PCM)
    assert np.isfinite(u1).all()
    np.testing.assert_allclose(u1[[0, 2]], [1.0, 3.0])
    assert 1.0 <= u1[1] <= 3.0


def test_roundoff_mismatch_in_total_thickness():
    h0 = np.array([1.0, 2.0, 3.0])
    u0 = np.array([4.0, 5.0, 6.0])
    h1 = np.array([3.0, 3.0 + 1e-13])                       # target 1e-13 m thicker
    u1 = _remap(h0, u0, h1, rm.PCM)
    assert np.all((u1 >= 4.0 - 1e-12) & (u1 <= 6.0 + 1e-12))
    assert abs((h1 * u1).sum() - (h0 * u0).sum()) < 1e-11


def test_source_thicker_than_target_last_target_absorbs():
    # source 1e-9 m thicker than the target: the excess must land in the last target cell
    h0 = np.array([1.0, 2.0, 3.0 + 1e-9])
    u0 = np.array([4.0, 5.0, 6.0])
    h1 = np.array([3.0, 3.0])
    u1 = _remap(h0, u0, h1, rm.PCM)
    assert abs((h1 * u1).sum() - (h0 * u0).sum()) < 1e-12


def test_plm_exact_for_linear_profile_interior():
    # u = 2 + 3 z sampled as cell means on a nonuniform grid: interior PLM reproduces the line
    h = np.array([1.0, 0.5, 2.0, 1.0, 1.5])
    z = np.concatenate([[0.0], np.cumsum(h)])
    u = 2 + 3 * 0.5 * (z[1:] + z[:-1])
    coef = np.empty((5, 3))
    rm.reconstruct(h, u, rm.PLM, rm.H_NEGLECT, rm.H_NEGLECT_EDGE, coef)
    for k in (1, 2, 3):
        np.testing.assert_allclose(coef[k, 0], 2 + 3 * z[k], rtol=1e-12)           # left edge
        np.testing.assert_allclose(coef[k, 0] + coef[k, 1], 2 + 3 * z[k + 1], rtol=1e-12)
    np.testing.assert_allclose(coef[[0, 4], 1], 0.0)                               # PCM at the ends


@pytest.mark.parametrize("scheme", [rm.PCM, rm.PLM])
def test_no_new_extrema(scheme):
    rng = np.random.default_rng(3)
    h0 = rng.uniform(0.2, 2.0, 30)
    u0 = np.where(np.arange(30) < 15, 10.0, 20.0) + rng.normal(0, 0.5, 30)
    h1 = rng.uniform(0.1, 1.0, 50)
    h1 *= h0.sum() / h1.sum()
    u1 = _remap(h0, u0, h1, scheme)
    assert u1.min() >= u0.min() - 1e-12 and u1.max() <= u0.max() + 1e-12


def test_plm_less_diffusive_than_pcm_on_smooth_profile():
    h0 = np.full(40, 1.0)
    z = np.arange(40) + 0.5
    u0 = np.sin(2 * np.pi * z / 40)
    h1 = np.full(40, 1.0)
    shift = 0.3                                   # remap to a grid shifted by 0.3 cells, then back
    hs = np.concatenate([[shift], np.full(39, 1.0), [1.0 - shift]])
    err = {}
    for scheme in (rm.PCM, rm.PLM):
        u = u0.copy()
        for _ in range(10):
            us = _remap(h0, u, hs, scheme)
            u = _remap(hs, us, h1, scheme)
        err[scheme] = np.abs(u - u0).max()
    assert err[rm.PLM] < 0.5 * err[rm.PCM]


def test_ppm_exact_for_quadratic_interior():
    # cell means of u = 1 + z + z²: the 4th-order edges are exact, and so are interior polynomials
    # where the profile is monotonic (no limiter action)
    h = np.array([1.0, 0.7, 1.3, 0.9, 1.1, 0.8, 1.2])
    z = np.concatenate([[0.0], np.cumsum(h)]) + 1.0          # z > 0: monotonic
    F = lambda x: x + x**2 / 2 + x**3 / 3                     # antiderivative of 1 + z + z²
    u = (F(z[1:]) - F(z[:-1])) / h
    coef = np.empty((7, 3))
    rm.reconstruct(h, u, rm.PPM_H4, rm.H_NEGLECT, rm.H_NEGLECT_EDGE, coef)
    for k in range(2, 5):
        np.testing.assert_allclose(coef[k, 0], 1 + z[k] + z[k] ** 2, rtol=1e-10)
        np.testing.assert_allclose(coef[k].sum(), 1 + z[k + 1] + z[k + 1] ** 2, rtol=1e-10)


def test_ppm_no_new_extrema_and_conservation():
    rng = np.random.default_rng(4)
    h0 = rng.uniform(0.2, 2.0, 40)
    u0 = np.where(np.arange(40) < 20, 10.0, 20.0) + rng.normal(0, 0.3, 40)
    h1 = rng.uniform(0.1, 1.0, 60)
    h1 *= h0.sum() / h1.sum()
    u1 = _remap(h0, u0, h1, rm.PPM_H4)
    assert u1.min() >= u0.min() - 1e-12 and u1.max() <= u0.max() + 1e-12
    assert abs((h1 * u1).sum() - (h0 * u0).sum()) <= 1e-12 * abs((h0 * u0).sum())


def test_ppm_less_diffusive_than_plm():
    h0 = np.full(40, 1.0)
    u0 = np.sin(2 * np.pi * (np.arange(40) + 0.5) / 40)
    hs = np.concatenate([[0.3], np.full(39, 1.0), [0.7]])
    err = {}
    for scheme in (rm.PLM, rm.PPM_H4):
        u = u0.copy()
        for _ in range(10):
            u = _remap(hs, _remap(h0, u, hs, scheme), h0, scheme)
        err[scheme] = np.abs(u - u0).max()
    assert err[rm.PPM_H4] < err[rm.PLM]


def test_short_columns_reduce_order():
    for n in (1, 2, 3):
        h0 = np.ones(n)
        u0 = np.arange(n, dtype=float)
        u1 = _remap(h0, u0, h0, rm.PPM_H4)
        np.testing.assert_allclose(u1, u0, rtol=1e-14)


def test_zstar_like_column_with_many_vanished_bottom_layers():
    # 50 layers, only the top 6 thick (shallow column in z*): no NaN, conservation
    h0 = np.r_[np.full(44, 0.0), np.linspace(5, 20, 6)]
    u0 = np.r_[np.full(44, 12.0), np.linspace(12, 18, 6)]
    h1 = np.r_[np.full(44, 1e-3), np.linspace(5, 20, 6)]
    h1[-1] -= 44e-3
    for scheme in (rm.PCM, rm.PLM, rm.PPM_H4):
        u1 = _remap(h0, u0, h1, scheme)
        assert np.isfinite(u1).all()
        assert abs((h1 * u1).sum() - (h0 * u0).sum()) <= 1e-12 * (h0 * u0).sum()


def test_remap_columns_matches_single_column():
    rng = np.random.default_rng(5)
    N, ny, nx = 8, 3, 4
    h0 = rng.uniform(0.5, 2.0, (N, ny, nx))
    f = rng.normal(10, 1, (N, ny, nx))
    h1 = rng.uniform(0.5, 2.0, (N, ny, nx))
    h1 *= h0.sum(0) / h1.sum(0)
    wet = np.ones((ny, nx), bool)
    wet[0, 0] = False
    out = rm.remap_columns(h0, f, h1, rm.PPM_H4, wet)
    ref = np.empty(N)
    rm.remap_column(h0[:, 1, 2], f[:, 1, 2], h1[:, 1, 2], rm.PPM_H4, ref)
    np.testing.assert_allclose(out[:, 1, 2], ref, rtol=0, atol=0)
    np.testing.assert_array_equal(out[:, 0, 0], f[:, 0, 0])


def test_ppm_end_values_with_vanished_layers_in_end_stencil():
    # vanished layers inside the 4-cell end stencil made a direct 4×4 solve singular
    # (ZeroDivisionError); MOM6 end_value_h4 (closed form) gives these values
    u1 = _remap([0, 1, 0, 0, 2, 3, 1], [5, 10, 11, 12, 13, 15, 16], np.ones(7), rm.PPM_H4)
    np.testing.assert_allclose(u1, [10, 12.642, 13.358, 14.338, 15.052, 15.61, 16], atol=1e-3)


def test_ppm_interleaved_vanished_layers_many_columns():
    # Review Focus 2: few active layers among many vanished ones, at both ends
    rng = np.random.default_rng(5)
    for _ in range(200):
        n = 50
        h0 = np.where(rng.random(n) < 0.85, rng.choice([0.0, 1e-10, 1e-3], n), rng.uniform(0.5, 50.0, n))
        u0 = rng.normal(15, 3, n)
        h1 = rng.uniform(0.1, 2.0, n)
        h1 *= h0.sum() / h1.sum()
        u1 = _remap(h0, u0, h1, rm.PPM_H4)
        assert np.isfinite(u1).all()
        assert abs((h1 * u1).sum() - (h0 * u0).sum()) <= 1e-11 * abs((h0 * u0).sum())


def test_fully_vanished_column_keeps_its_values():
    # a dry column (D = 0) has no content to remap: keep the source values instead of 0
    # (a rewetting intertidal cell would otherwise come back with T = S = 0)
    for scheme in (rm.PCM, rm.PLM, rm.PPM_H4):
        u0 = np.array([20.0, 21.0, 22.0, 23.0])
        np.testing.assert_array_equal(_remap(np.zeros(4), u0, np.zeros(4), scheme), u0)
        np.testing.assert_allclose(_remap(np.zeros(4), u0, np.zeros(2), scheme), 21.5)
