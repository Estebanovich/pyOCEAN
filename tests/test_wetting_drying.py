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

import sys
from pathlib import Path

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

    @staticmethod
    def _amplitude_after(periods: int) -> float:
        t_end = periods * PERIOD
        x, eta, *_ = thacker_run(1000.0, t_end, theta=0.5)
        m = np.abs(x) < 60e3
        return np.polyfit(x[m], eta[m], 1)[0] / (2 * ETA0 * D0 / L * np.cos(OMEGA * t_end))

    def test_front_dissipation_regression(self):
        # Known limitation: faces that become wet start from rest, a momentum sink at the
        # moving shoreline: −1 %/period of the planar amplitude (0.970 after 3 periods)
        assert self._amplitude_after(3) > 0.965

    @pytest.mark.xfail(strict=True, reason="front momentum loss ~1 %/period; CROCO THACKER loses 0.2 %/period")
    def test_front_dissipation_as_low_as_croco(self):
        assert self._amplitude_after(3) > 0.99

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


class TestFlather:
    """Flather (1976) radiation on domain-edge faces, with the total depth D = H + η_c."""

    @staticmethod
    def _channel(nx=200, H0=10.0, dx=500.0):
        hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
        fu = np.zeros((1, nx + 1), bool)
        fu[0, -1] = True                                                    # open east edge
        return hg, np.full((1, nx), H0), fu

    def test_outgoing_wave_leaves_the_domain(self):
        hg, H, fu = self._channel()
        c = np.sqrt(G * 10.0)
        dt = 0.5 * 500.0 / c
        fs = WetDryFreeSurface(hg, H, dt=dt, theta=0.5, g=G, flather_u=fu)
        x = (np.arange(200) + 0.5) * 500.0
        eta = 0.01 * np.exp(-((x - 70e3) / 4e3) ** 2)[None, :]           # small: η/H = 1e-3
        u, v = np.zeros((1, 201)), np.zeros((2, 200))
        area = hg.area_t.values
        e0 = np.sum(area * eta**2)
        zero = np.zeros((1, 201))
        for _ in range(int(2.5 * 100e3 / c / dt)):
            eta, u, v = fs.step(eta, u, v, u.copy(), v.copy(), flather_eta=zero, flather_ubar=zero)
        assert np.sum(area * eta**2) < 1e-3 * e0

    def test_volume_budget_through_flather_face(self):
        hg, H, fu = self._channel(nx=20)
        fs = WetDryFreeSurface(hg, H, dt=30.0, theta=0.7, g=G, flather_u=fu)
        rng = np.random.default_rng(1)
        eta = rng.normal(0, 0.05, (1, 20))
        u = rng.normal(0, 0.1, (1, 21))
        u[0, [0, -1]] = 0.0
        v = np.zeros((2, 20))
        ext_eta, ext_u = np.full((1, 21), 0.03), np.full((1, 21), -0.02)
        area = hg.area_t.values
        vol0 = np.sum(area * eta)
        eta1, u1, _ = fs.step(eta, u, v, u.copy(), v.copy(), flather_eta=ext_eta, flather_ubar=ext_u)
        D = 10.0 + eta[0, -1]                                               # total depth of the boundary cell at n
        dy = hg.dy_u.values[0, -1]
        q = lambda e: dy * D * (-0.02 + np.sqrt(G / D) * (e[0, -1] - 0.03))   # noqa: E731
        assert np.sum(area * eta1) == pytest.approx(vol0 - 30.0 * (0.3 * q(eta) + 0.7 * q(eta1)), rel=1e-12)
        assert u1[0, -1] == pytest.approx(q(eta1) / (dy * D), rel=1e-12)

    def test_deep_limit_matches_linear_free_surface_with_flather(self):
        nx, H0 = 40, 1000.0
        hg = create_cartesian_grid(nx=nx, ny=2, dx=5000.0, dy=5000.0, f0=0.0)
        vg = create_z_grid(1, [H0])
        from core.dynamics.free_surface import open_boundary_hfac
        hf = open_boundary_hfac(compute_hfac(vg, np.full((2, nx), H0)), ["east"])
        fu = np.zeros((2, nx + 1), bool)
        fu[:, -1] = True
        lin = ImplicitFreeSurface(hg, vg, hf, dt=120.0, theta=0.6, g=G, flather_u=fu)
        wd = WetDryFreeSurface(hg, np.full((2, nx), H0), dt=120.0, theta=0.6, g=G, flather_u=fu)
        el = ew = np.zeros((2, nx))
        ul, vl = np.zeros((1, 2, nx + 1)), np.zeros((1, 3, nx))
        uw, vw = np.zeros((2, nx + 1)), np.zeros((3, nx))
        peak, worst = 0.0, 0.0
        for n in range(200):
            ext = np.full((2, nx + 1), 1e-3 * np.sin(2 * np.pi * (n + 0.5) * 120.0 / 44714.0))
            el, ul, vl = lin.step(el, ul, vl, ul.copy(), vl.copy(), flather_eta=ext, flather_ubar=np.zeros_like(ext))
            ew, uw, vw = wd.step(ew, uw, vw, uw.copy(), vw.copy(), flather_eta=ext, flather_ubar=np.zeros_like(ext))
            peak, worst = max(peak, np.abs(el).max()), max(worst, np.abs(ew - el).max())
        assert peak > 5e-4                                                  # the tide got in
        assert worst < 1e-3 * 1e-5 * 10                                     # D = H + η vs H: η/H = 1e-6

    def test_dry_boundary_cell_closes_the_face(self):
        hg, H, fu = self._channel(nx=10)
        H[0, -1] = -1.0                                                     # emerged land at the open edge
        fs = WetDryFreeSurface(hg, H, dt=10.0, theta=0.6, g=G, flather_u=fu)
        eta = np.where(H > 0, 0.0, -H)
        u, v = np.zeros((1, 11)), np.zeros((2, 10))
        e1, u1, _ = fs.step(eta, u, v, u.copy(), v.copy(), flather_eta=np.full((1, 11), 0.5),
                            flather_ubar=np.zeros((1, 11)))
        assert u1[0, -1] == 0.0 and e1[0, -1] == pytest.approx(1.0)


def test_cg_solver_matches_direct():
    # Warm-started Jacobi-preconditioned CG (large grids) = direct solve
    x, e_direct, *_ = thacker_run(2000.0, 0.5 * PERIOD)
    nx = x.size
    hg = create_cartesian_grid(nx=nx, ny=1, dx=2000.0, dy=2000.0, f0=0.0)
    H = D0 * (1 - (x / L) ** 2)[None, :]
    fs = WetDryFreeSurface(hg, H, dt=60.0, theta=0.55, g=G, solver="cg", cg_rtol=1e-12)
    eta = thacker_exact(x, 0.0)[0][None, :]
    u, v = np.zeros((1, nx + 1)), np.zeros((2, nx))
    for _ in range(int(round(0.5 * PERIOD / 60.0))):
        eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
    np.testing.assert_allclose(eta[0], e_direct, atol=1e-8)


# ---------------------------------------------------------------------------
# Cross-model: CROCO TIDAL_FLAT (2DH) — moving shoreline under a 4 m tide with
# advection and quadratic drag.  Reference written by
# validation/tidal_flat/make_reference.py from the CROCO run (README there).
# ---------------------------------------------------------------------------
_TF_REF = Path(__file__).parent / "data" / "tidal_flat_croco_reference.npz"


@pytest.fixture(scope="module")
def tidal_flat():
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "tidal_flat"))
    from run_tidal_flat import DX, croco_bed, run, shoreline
    ref = np.load(_TF_REF)
    ts, etas, us, H, clip = run(theta=0.55, dt=15.0, hours=36.0, dmin=0.1, cd=2.5e-3, adv="upwind3", every=60)
    x = (np.arange(H.size) + 0.5) * DX
    k = [int(np.argmin(np.abs(ts - t))) for t in ref["t"]]
    sh = np.array([shoreline(x, H + e, 0.1) for e in etas[k]])
    return ref, etas[k], us[k], sh, clip, croco_bed()


class TestTidalFlatVsCroco:
    def test_bed_matches_croco(self, tidal_flat):
        ref, *_, bed = tidal_flat
        np.testing.assert_allclose(bed, ref["h"], atol=1e-5)

    def test_surface_matches_croco_at_stations(self, tidal_flat):
        ref, etas, *_ = tidal_flat
        d = etas[:, ref["i_sta"]] - ref["zeta_sta"]
        assert np.sqrt(np.mean(d ** 2)) < 0.02                 # CROCO tide range 4 m
        assert np.abs(d).max() < 0.06

    def test_velocity_matches_croco_at_stations(self, tidal_flat):
        ref, _, us, *_ = tidal_flat
        i = ref["i_sta"]
        up = 0.5 * (us[:, i] + us[:, i + 1])
        assert np.sqrt(np.mean((up - ref["ubar_sta"]) ** 2)) < 0.04

    def test_shoreline_within_one_cell(self, tidal_flat):
        ref, _, _, sh, *_ = tidal_flat
        assert np.abs(sh - ref["shoreline"]).max() <= 500.0 + 1e-6

    def test_no_volume_clipped(self, tidal_flat):
        assert tidal_flat[4] == 0.0


class TestTidalFlatVsCroco3D:
    """
    CROCO's original 3-D TIDAL_FLAT (GLS k-ε, log-law bed with z₀ = 1e-4 m;
    sediment off).  Its effective drag falls with depth, so a constant C_d
    over-damps the tide (η ~30 cm rms off at the stations); the log-law C_d(D) of
    physics.bottom_friction recovers it to a few cm.
    """

    @staticmethod
    def _eta_rms(z0, cd=2.5e-3):
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "tidal_flat"))
        from run_tidal_flat import run
        ref = np.load(Path(__file__).parent / "data" / "tidal_flat_croco_3d_reference.npz")
        ts, etas, *_ = run(theta=0.55, dt=15.0, hours=36.0, dmin=0.1, cd=cd, adv="upwind3", every=60, z0=z0)
        k = [int(np.argmin(np.abs(ts - t))) for t in ref["t"]]
        return np.sqrt(np.mean((etas[k][:, ref["i_sta"]] - ref["zeta_sta"]) ** 2))

    def test_constant_cd_overdamps(self):
        assert self._eta_rms(None) > 0.20

    def test_sigma_gls_model_matches_croco_3d(self):
        # 10 σ-layers, k-ε GLS (Canuto A), first-layer log law: the 3-D physics of the CROCO run
        sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "tidal_flat"))
        from run_tidal_flat import run3d
        ref = np.load(Path(__file__).parent / "data" / "tidal_flat_croco_3d_reference.npz")
        ts, etas, us, *_ = run3d(theta=0.55, dt=15.0, hours=36.0, dmin=0.1, z0=1e-4, adv="upwind3", every=60,
                                 n_layers=10, closure="gls")
        k = [int(np.argmin(np.abs(ts - t))) for t in ref["t"]]
        i = ref["i_sta"]
        assert np.sqrt(np.mean((etas[k][:, i] - ref["zeta_sta"]) ** 2)) < 0.025          # 1.2 cm (ū 3.1 cm/s)
        up = 0.5 * (us[k][:, i] + us[k][:, i + 1])
        assert np.sqrt(np.mean((up - ref["ubar_sta"]) ** 2)) < 0.05

    def test_log_law_drag_matches_croco_3d(self):
        assert self._eta_rms(1e-4) < 0.08                  # 6.6 cm at the stations (constant C_d: ~30)
