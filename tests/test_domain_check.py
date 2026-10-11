"""
Tests for diagnostics/domain_check.py — pre-run checks of a model domain
(the most frequent pain point in community model forums: setting up a regional
domain, and finding why it blows up).
"""

from __future__ import annotations

import numpy as np

from core.grid.horizontal import create_cartesian_grid
from diagnostics.domain_check import check_domain, find_pits, isolated_water_bodies, rx0, rx1


def test_rx0_is_the_beckmann_haidvogel_slope_factor():
    H = np.array([[10.0, 30.0, 30.0]])
    r, (j, i, axis) = rx0(H)
    assert np.isclose(r.max(), 20.0 / 40.0) and (j, i, axis) == (0, 0, "x")   # face between cells 0 and 1


def test_rx0_ignores_land():
    H = np.array([[10.0, np.nan, 300.0]])
    r, _ = rx0(H)
    assert r.max() == 0.0


def test_rx1_is_the_haney_number_and_zero_over_a_flat_bed():
    sigma_w = np.linspace(-1.0, 0.0, 11)                     # interfaces, bed → surface
    assert rx1(np.full((1, 5), 50.0), sigma_w)[0].max() == 0.0
    H = np.array([[10.0, 20.0]])
    z = sigma_w[:, None, None] * H[None]                     # σ-levels: z_k = σ_k H
    num = np.abs(z[1:, 0, 1] - z[1:, 0, 0] + z[:-1, 0, 1] - z[:-1, 0, 0])
    den = np.abs(z[1:, 0, 1] + z[1:, 0, 0] - z[:-1, 0, 1] - z[:-1, 0, 0])
    assert np.isclose(rx1(H, sigma_w)[0].max(), (num / den).max())


def test_isolated_water_bodies_finds_the_guaymas_pit():
    # main water body + a deep pocket connected only through floodable land (H ≤ 0)
    H = np.full((6, 10), 30.0)
    H[:, 6:] = -2.0                                          # floodable flat
    H[2:4, 8:10] = 150.0                                     # GEBCO-like pit behind the flat
    bodies = isolated_water_bodies(H, min_depth=0.0)
    assert len(bodies) == 1
    b = bodies[0]
    assert b.n_cells == 4 and b.max_depth == 150.0 and b.j == 2 and b.i == 8


def test_find_pits_flags_cells_much_deeper_than_their_neighbours():
    H = np.full((7, 7), 10.0)
    H[3, 3] = 120.0
    pits = find_pits(H, ratio=3.0, min_excess=20.0)
    assert [(p.j, p.i) for p in pits] == [(3, 3)]
    assert find_pits(np.full((7, 7), 10.0)) == []


def test_check_domain_report_with_cfl_advice():
    hg = create_cartesian_grid(nx=20, ny=10, dx=500.0, dy=500.0, f0=1e-4)
    H = np.full((10, 20), 20.0)
    H[:, 10:] = 200.0                                        # cliff: rx0 = 180/220
    H[5, 5] = np.nan
    rep = check_domain(hg, H, dt=600.0, u_max=1.0, sigma_layers=10, implicit_free_surface=True,
                       gprime_h=0.5)
    text = rep.summary()
    assert any("rx0" in w for w in rep.warnings)
    assert any("advective CFL" in e for e in rep.errors)                 # 1 m/s · 600 s / 500 m = 1.2
    assert rep.dt_max_advective == 500.0 / 1.0 * rep.cfl_target
    assert "isolated" in text.lower() or "Isolated" in text
    rep2 = check_domain(hg, np.full((10, 20), 20.0), dt=60.0, u_max=1.0, implicit_free_surface=True)
    assert rep2.errors == [] and rep2.ok


def test_explicit_free_surface_requires_the_external_wave_cfl():
    hg = create_cartesian_grid(nx=10, ny=10, dx=1000.0, dy=1000.0, f0=0.0)
    rep = check_domain(hg, np.full((10, 10), 4000.0), dt=10.0, u_max=0.1, implicit_free_surface=False)
    assert any("external gravity wave CFL" in e for e in rep.errors)      # √(g·4000) ≈ 198 m/s → 1.98
