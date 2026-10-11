"""
Tests for core.dynamics.wetting_drying — semi-implicit free surface with
wetting and drying (Casulli 1990; Casulli & Cheng 1992).

The face transport uses the total depth D_f = max(0, H_f + η_upwind), H_f the
shallower of the two adjacent cells; a face with D_f ≤ D_min is dry (u = 0).

Analytic reference: Thacker (1981) planar oscillation in a parabolic channel
h(x) = D (1 − x²/L²) — the setup of CROCO's THACKER (THACKER_2DV) test case:
    η(x, t) = A₀ x cos ωt + η₀² D sin² ωt,   u = −η₀ ω L sin ωt,
    A₀ = 2 η₀ D / L,   ω² = 2 g D / L²
(u is uniform, so momentum advection vanishes exactly).
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics.free_surface import ImplicitFreeSurface
from core.dynamics.wetting_drying import WetDryFreeSurface
from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import compute_hfac, create_z_grid

G = 9.81
D0, L, ETA0 = 10.0, 80e3, 0.1          # CROCO THACKER: depth 10 m, Lt = 80 km, ETA0 = 0.1
OMEGA = np.sqrt(2 * G * D0 / L**2)
PERIOD = 2 * np.pi / OMEGA


def thacker_exact(x: np.ndarray, t: float) -> tuple[np.ndarray, np.ndarray]:
    """Exact η (−H where dry) and wet mask."""
    H = D0 * (1 - (x / L) ** 2)
    eta = 2 * ETA0 * D0 / L * x * np.cos(OMEGA * t) + ETA0**2 * D0 * np.sin(OMEGA * t) ** 2
    wet = H + eta > 0
    return np.where(wet, eta, -H), wet


def thacker_run(dx: float, t_end: float, dt: float = 60.0, theta: float = 0.55):
    nx = int(round(200e3 / dx))
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    x = -100e3 + (np.arange(nx) + 0.5) * dx
    H = D0 * (1 - (x / L) ** 2)[None, :]
    fs = WetDryFreeSurface(hg, H, dt=dt, theta=theta, g=G)
    eta, _ = thacker_exact(x, 0.0)
    eta = eta[None, :]
    u, v = np.zeros((1, nx + 1)), np.zeros((2, nx))
    area = hg["area_t"].values
    vol0 = np.sum(area * np.maximum(H + eta, 0.0))
    min_depth, vol_err, clipped = np.inf, 0.0, 0.0
    for _ in range(int(round(t_end / dt))):
        eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
        clipped += fs.last_clip_volume
        min_depth = min(min_depth, float(np.min(H + eta)))
        vol_err = max(vol_err, abs(np.sum(area * np.maximum(H + eta, 0.0)) - vol0) / vol0)
    thacker_run.clipped = clipped
    return x, eta[0], u[0], min_depth, vol_err


class TestLakeAtRest:

    def test_emerged_land_stays_at_rest(self):
        # Well-balanced: water at η = 0, dry land above datum with η = −H (surface on the bed)
        nx = 40
        hg = create_cartesian_grid(nx=nx, ny=3, dx=500.0, dy=500.0, f0=1e-4)
        x = np.arange(nx) - nx / 2
        H = np.tile(5.0 - 0.02 * x**2, (3, 1))                        # wet centre, emerged flanks
        eta = np.where(H > 0, 0.0, -H)
        fs = WetDryFreeSurface(hg, H, dt=30.0, theta=0.6, g=G)
        u, v = np.zeros((3, nx + 1)), np.zeros((4, nx))
        e1 = eta.copy()
        for _ in range(200):
            e1, u, v = fs.step(e1, u, v, u.copy(), v.copy())
        assert np.max(np.abs(u)) == 0.0 and np.max(np.abs(v)) == 0.0
        np.testing.assert_allclose(e1, eta, atol=1e-12)


class TestThacker:

    def test_one_period_against_exact(self):
        x, eta, u, min_depth, vol_err = thacker_run(1000.0, PERIOD)
        ex, wet = thacker_exact(x, PERIOD)
        err = np.abs(eta - ex)[wet]
        assert np.max(err) < 0.10 and np.sqrt(np.mean(err**2)) < 0.02     # η range ±2 m

    def test_shoreline_position(self):
        t = 0.75 * PERIOD                                                 # mid-excursion
        x, eta, *_ = thacker_run(1000.0, t)
        H = D0 * (1 - (x / L) ** 2)
        wet_mod = H + eta > 0.05
        _, wet_ex = thacker_exact(x, t)
        assert abs(x[wet_mod].max() - x[wet_ex].max()) <= 2000.0
        assert abs(x[wet_mod].min() - x[wet_ex].min()) <= 2000.0

    def test_convergence_with_resolution(self):
        errs = []
        for dx in (4000.0, 2000.0, 1000.0):
            x, eta, *_ = thacker_run(dx, PERIOD)
            ex, wet = thacker_exact(x, PERIOD)
            errs.append(np.sqrt(np.mean((eta - ex)[wet] ** 2)))
        assert errs[1] < 0.75 * errs[0] and errs[2] < 0.75 * errs[1]

    def test_mass_conserved_and_depth_non_negative(self):
        *_, min_depth, vol_err = thacker_run(1000.0, 2 * PERIOD)
        assert min_depth >= -1e-10
        assert vol_err < 1e-9
        # positive by construction: the guard only removes round-off (~1e-12 m depths)
        assert thacker_run.clipped < 1e-12 * 1e9            # total volume ≈ 1e9 m³


class TestDeepLimit:

    def test_matches_linear_free_surface_without_drying(self):
        # Small amplitude, deep flat basin: same as ImplicitFreeSurface (D ≈ H)
        nx, H0 = 30, 1000.0
        hg = create_cartesian_grid(nx=nx, ny=4, dx=2000.0, dy=2000.0, f0=0.0)
        vg = create_z_grid(1, [H0])
        hf = compute_hfac(vg, np.full((4, nx), H0))
        lin = ImplicitFreeSurface(hg, vg, hf, dt=100.0, theta=0.6, g=G)
        wd = WetDryFreeSurface(hg, np.full((4, nx), H0), dt=100.0, theta=0.6, g=G)
        x = (np.arange(nx) + 0.5) / nx
        e_lin = e_wd = 1e-3 * np.cos(np.pi * x)[None, :] * np.ones((4, 1))
        ul, vl = np.zeros((1, 4, nx + 1)), np.zeros((1, 5, nx))
        uw, vw = np.zeros((4, nx + 1)), np.zeros((5, nx))
        for _ in range(100):
            e_lin, ul, vl = lin.step(e_lin, ul, vl, ul.copy(), vl.copy())
            e_wd, uw, vw = wd.step(e_wd, uw, vw, uw.copy(), vw.copy())
        np.testing.assert_allclose(e_wd, e_lin, atol=1e-3 * 1e-5 * 50)


class TestValidation:

    def test_wall_cells(self):
        hg = create_cartesian_grid(nx=5, ny=3, dx=100.0, dy=100.0, f0=0.0)
        H = np.full((3, 5), 2.0)
        H[1, 2] = np.nan                                                    # permanent wall
        fs = WetDryFreeSurface(hg, H, dt=10.0)
        eta = np.zeros((3, 5))
        eta[0, 0] = 0.5
        u, v = np.zeros((3, 6)), np.zeros((4, 5))
        for _ in range(50):
            eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
        assert u[1, 2] == 0.0 and u[1, 3] == 0.0 and v[1, 2] == 0.0 and v[2, 2] == 0.0
        assert eta[1, 2] == 0.0

    def test_bad_theta(self):
        hg = create_cartesian_grid(nx=3, ny=3, dx=100.0, dy=100.0, f0=0.0)
        with pytest.raises(ValueError, match="theta"):
            WetDryFreeSurface(hg, np.ones((3, 3)), dt=1.0, theta=0.4)
