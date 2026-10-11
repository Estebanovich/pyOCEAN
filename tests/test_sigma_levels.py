"""
Tests for core/grid/sigma_levels.py — layer fractions Δσ_k (k = 0 at the bed)
for stretched σ-coordinates: Song & Haidvogel (1994) stretching (the
S-coordinate of SCRUM/ROMS with h_c = 0) and fractions taken from a list of
z-level thicknesses (e.g. MITgcm ``delR``), so a σ model reproduces the
z-levels of a reference run over its maximum depth.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.grid.sigma_levels import sigma_from_thicknesses, stretched_sigma


def _C(s, ts, tb):
    # Song & Haidvogel (1994) as written in Shchepetkin & McWilliams (2003), eq. 4.3
    return (1 - tb) * np.sinh(ts * s) / np.sinh(ts) + tb * (np.tanh(ts * (s + 0.5)) / (2 * np.tanh(ts / 2)) - 0.5)


def test_no_stretching_is_uniform():
    np.testing.assert_allclose(stretched_sigma(10, 0.0, 0.0), 0.1, rtol=1e-14)


@pytest.mark.parametrize("ts,tb", [(3.0, 0.0), (5.0, 0.4), (6.0, 1.0)])
def test_fractions_are_the_song_haidvogel_increments(ts, tb):
    ds = stretched_sigma(16, ts, tb)
    assert np.all(ds > 0) and ds.sum() == pytest.approx(1.0, abs=1e-14)
    s_w = -1.0 + np.arange(17) / 16                                     # interfaces, bed → surface
    np.testing.assert_allclose(np.concatenate([[0.0], np.cumsum(ds)]), 1.0 + _C(s_w, ts, tb), atol=1e-14)


def test_surface_stretching_refines_the_top_and_bottom_parameter_the_bed():
    ds = stretched_sigma(20, 5.0, 0.0)
    assert ds[-1] < 0.2 * ds[0]                                          # k = N−1 is the surface layer
    db = stretched_sigma(20, 5.0, 1.0)
    assert db[0] < 0.5 * db[10] and db[-1] < 0.5 * db[10]


def test_fractions_from_z_level_thicknesses_listed_top_first():
    ds = sigma_from_thicknesses([1.0, 2.0, 3.0, 4.0])                    # MITgcm delR order: surface first
    np.testing.assert_allclose(ds, [0.4, 0.3, 0.2, 0.1])
    with pytest.raises(ValueError):
        sigma_from_thicknesses([1.0, 0.0, 2.0])
