"""
Tests for core.dynamics.free_surface — implicit (θ-method) linear free surface,
and the AB2 + implicit free-surface step in core.timestepping.ab2.

Validation against exact discrete solutions
-------------------------------------------
For a closed flat basin of length L, depth H, f = 0, the C-grid mode
η_i = A cos(k x_i), k = π/L, x_i = (i + ½)Δx, is an exact eigenvector of the
discrete operator with spatial frequency

    ω = (2c/Δx) sin(kΔx/2),   c = √(gH)                       (Durran 2010, §4.3)

and the θ-method advances it by the complex factor

    λ = (1 + (1 − θ) iωΔt) / (1 − θ iωΔt)                     (Durran 2010, §2.3)

so η_i^n = A |λ|ⁿ cos(n arg λ) cos(k x_i) exactly.  θ = ½ (Crank–Nicolson) is
neutral (|λ| = 1); θ = 1 (backward Euler, MITgcm default) damps.  As Δx → 0 and Δt → 0
the period tends to Merian's 2L/c.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from core.dynamics.free_surface import ImplicitFreeSurface
from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import compute_hfac, create_z_grid
from core.state import ModelState
from core.timestepping.ab2 import ab2_step_implicit_fs

G = 9.81


def _basin(nx=40, ny=3, dx=1000.0, depth=20.0, nz=2, bathy=None):
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=0.0)
    vg = create_z_grid(nz, [depth / nz] * nz)
    d = np.full((ny, nx), depth) if bathy is None else bathy
    return hg, vg, compute_hfac(vg, d)


def _seiche_eta(hg, amp=0.1):
    nx = hg.attrs["nx"]
    dx = float(hg.attrs["dx"])
    x = (np.arange(nx) + 0.5) * dx
    return amp * np.cos(np.pi * x / (nx * dx))[None, :] * np.ones((hg.attrs["ny"], 1))


def _run_free(fs, hg, vg, eta0, nsteps):
    """Pure free-surface dynamics: no other tendencies."""
    st = ModelState.from_grids(hg, vg)
    st.eta.values[:] = eta0
    eta, u, v = st.eta.values.copy(), st.u.values.copy(), st.v.values.copy()
    series = [eta.copy()]
    for _ in range(nsteps):
        eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
        series.append(eta.copy())
    return np.array(series)


def _lambda(theta, omega, dt):
    return (1 + (1 - theta) * 1j * omega * dt) / (1 - theta * 1j * omega * dt)


# ===========================================================================
# Operator properties
# ===========================================================================

class TestOperator:

    def test_matrix_symmetric_positive_definite(self):
        hg, vg, hf = _basin(nx=8, ny=5)
        fs = ImplicitFreeSurface(hg, vg, hf, dt=100.0)
        A = fs.matrix.toarray()
        np.testing.assert_allclose(A, A.T, rtol=1e-14, atol=0)
        assert np.linalg.eigvalsh(A).min() > 0.0

    def test_rest_state_stays_at_rest(self):
        hg, vg, hf = _basin()
        fs = ImplicitFreeSurface(hg, vg, hf, dt=200.0)
        series = _run_free(fs, hg, vg, np.full((3, 40), 0.25), 20)
        np.testing.assert_allclose(series[-1], 0.25, atol=1e-12)

    def test_volume_conserved_closed_basin_with_island_and_slope(self):
        rng = np.random.default_rng(0)
        bathy = np.linspace(5.0, 60.0, 12)[None, :] * np.ones((10, 1))
        bathy[4:6, 5:7] = 0.0                                   # island
        hg, vg, hf = _basin(nx=12, ny=10, depth=60.0, nz=6, bathy=bathy)
        fs = ImplicitFreeSurface(hg, vg, hf, dt=300.0, theta=0.6)
        eta = rng.normal(0, 0.1, (10, 12)) * (bathy > 0)
        u = rng.normal(0, 0.2, hf.hfac_u.shape) * (hf.hfac_u.values > 0)
        v = rng.normal(0, 0.2, hf.hfac_v.shape) * (hf.hfac_v.values > 0)
        area = hg.area_t.values
        vol0 = np.sum(area * eta)
        for _ in range(10):
            eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
        assert np.sum(area * eta) == pytest.approx(vol0, abs=1e-9 * np.sum(area))

    def test_land_cells_and_walls_stay_dry(self):
        bathy = np.full((6, 8), 30.0)
        bathy[2:4, 3:5] = 0.0
        hg, vg, hf = _basin(nx=8, ny=6, depth=30.0, bathy=bathy)
        fs = ImplicitFreeSurface(hg, vg, hf, dt=100.0)
        eta0 = np.where(bathy > 0, 0.05 * np.arange(8)[None, :], 0.0)
        series = _run_free(fs, hg, vg, eta0, 5)
        assert np.all(series[-1][bathy == 0] == 0.0)


# ===========================================================================
# Seiche: exact discrete solution
# ===========================================================================

class TestSeiche:

    @pytest.mark.parametrize("theta", [0.5, 0.75, 1.0])
    @pytest.mark.parametrize("courant", [0.5, 4.0])
    def test_mode_evolution_exact(self, theta, courant):
        hg, vg, hf = _basin(nx=40, dx=1000.0, depth=20.0)
        c = np.sqrt(G * 20.0)
        dt = courant * 1000.0 / c
        fs = ImplicitFreeSurface(hg, vg, hf, dt=dt, theta=theta, g=G)
        eta0 = _seiche_eta(hg)
        n = 60
        series = _run_free(fs, hg, vg, eta0, n)
        k = np.pi / 40000.0
        omega = 2 * c / 1000.0 * np.sin(k * 1000.0 / 2)
        lam = _lambda(theta, omega, dt)
        expected = (np.abs(lam) ** np.arange(n + 1) * np.cos(np.arange(n + 1) * np.angle(lam)))
        np.testing.assert_allclose(series[:, 1, 0] / eta0[1, 0], expected, atol=1e-9)

    def test_backward_euler_damps(self):
        hg, vg, hf = _basin()
        dt = 4.0 * 1000.0 / np.sqrt(G * 20.0)
        series = _run_free(ImplicitFreeSurface(hg, vg, hf, dt=dt, theta=1.0), hg, vg, _seiche_eta(hg), 400)
        assert np.abs(series[-50:]).max() < 0.2 * np.abs(series[0]).max()

    def test_period_converges_to_merian(self):
        # Fine grid, small Courant number, CN: period → 2L/√(gH)
        L, H = 40000.0, 20.0
        hg, vg, hf = _basin(nx=200, dx=L / 200, depth=H)
        c = np.sqrt(G * H)
        dt = 2.0 * (L / 200) / c                                   # ~200 steps per period
        series = _run_free(ImplicitFreeSurface(hg, vg, hf, dt=dt, theta=0.5), hg, vg, _seiche_eta(hg), 1200)
        sig = series[:, 1, 0]
        up = np.nonzero((sig[:-1] < 0) & (sig[1:] >= 0))[0]
        t_cross = (up + sig[up] / (sig[up] - sig[up + 1])) * dt   # linear interpolation
        period = np.mean(np.diff(t_cross))
        assert period == pytest.approx(2 * L / c, rel=2e-3)

    def test_stable_far_beyond_explicit_cfl(self):
        hg, vg, hf = _basin()
        dt = 20.0 * 1000.0 / np.sqrt(G * 20.0)                  # Courant number 20
        series = _run_free(ImplicitFreeSurface(hg, vg, hf, dt=dt, theta=1.0), hg, vg, _seiche_eta(hg), 200)
        assert np.all(np.isfinite(series)) and np.abs(series).max() <= 0.1 + 1e-12


# ===========================================================================
# Open boundary with prescribed normal transport
# ===========================================================================

class TestOpenBoundary:

    def test_prescribed_inflow_fills_basin(self):
        hg, vg, hf = _basin(nx=10, ny=4, depth=20.0)
        open_u = np.zeros(hf.hfac_u.shape[1:], dtype=bool)
        open_u[:, 0] = True                                       # west face open
        hf_open = hf.copy()
        hf_open["hfac_u"].values[:, :, 0] = 1.0                   # wet open face
        fs = ImplicitFreeSurface(hg, vg, hf_open, dt=60.0, open_u=open_u)
        st = ModelState.from_grids(hg, vg)
        u = st.u.values.copy()
        u[:, :, 0] = 0.1                                          # inflow 0.1 m/s
        eta, v = st.eta.values.copy(), st.v.values.copy()
        q = 0.1 * 20.0 * 1000.0 * 4                               # u·H·dy·ny  [m³/s]
        vol = []
        for _ in range(10):
            eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
            vol.append(np.sum(hg.area_t.values * eta))
        np.testing.assert_allclose(np.diff(vol), q * 60.0, rtol=1e-10)
        np.testing.assert_allclose(u[:, :, 0], 0.1)                # prescribed face untouched


# ===========================================================================
# AB2 + implicit free surface step
# ===========================================================================

class TestAB2ImplicitStep:

    def _setup(self):
        hg, vg, hf = _basin(nx=20, ny=20, dx=2000.0, depth=50.0, nz=3)
        hg = hg.assign(f_q=xr.full_like(hg.f_q, 1e-4))
        dt = 3.0 * 2000.0 / np.sqrt(G * 50.0)
        fs = ImplicitFreeSurface(hg, vg, hf, dt=dt, theta=1.0)
        return hg, vg, hf, fs, dt

    def test_no_tendency_matches_pure_free_surface(self):
        hg, vg, hf, fs, dt = self._setup()
        st = ModelState.from_grids(hg, vg)
        st.eta.values[:] = 0.1 * np.exp(-((np.arange(20) - 10) ** 2) / 8.0)[None, :]
        zero = lambda s: s * 0.0                                    # noqa: E731
        st1, _ = ab2_step_implicit_fs(st, zero, dt, fs)
        eta_ref, _, _ = fs.step(st.eta.values, st.u.values, st.v.values, st.u.values, st.v.values)
        np.testing.assert_allclose(st1.eta.values, eta_ref, atol=1e-14)
        assert st1.time == pytest.approx(dt)

    def test_tracers_advanced_by_ab2(self):
        hg, vg, hf, fs, dt = self._setup()
        st = ModelState.from_grids(hg, vg, T_init=20.0)

        def heating(s):
            t = s * 0.0
            t.T.values[:] = 1e-5
            return t

        st1, g0 = ab2_step_implicit_fs(st, heating, dt, fs)
        st2, _ = ab2_step_implicit_fs(st1, heating, dt, fs, tendency_prev=g0)
        np.testing.assert_allclose(st2.T.values, 20.0 + 2 * dt * 1e-5, rtol=1e-12)

    def test_velocity_zero_on_walls(self):
        hg, vg, hf, fs, dt = self._setup()
        st = ModelState.from_grids(hg, vg)
        st.eta.values[:] = 0.05

        def push(s):
            t = s * 0.0
            t.u.values[:] = 1e-4
            return t

        st1, _ = ab2_step_implicit_fs(st, push, dt, fs)
        assert np.all(st1.u.values[:, :, 0] == 0.0) and np.all(st1.u.values[:, :, -1] == 0.0)

    def test_interior_prescribed_face_is_not_pressure_coupled(self):
        # A gate at an interior face with prescribed zero transport splits the basin:
        # each side conserves its own volume.
        hg, vg, hf = _basin(nx=10, ny=4, depth=20.0)
        open_u = np.zeros(hf.hfac_u.shape[1:], dtype=bool)
        open_u[:, 5] = True
        fs = ImplicitFreeSurface(hg, vg, hf, dt=60.0, open_u=open_u)
        st = ModelState.from_grids(hg, vg)
        eta = np.where(np.arange(10)[None, :] < 5, 0.2, 0.0) * np.ones((4, 1))
        u, v = st.u.values.copy(), st.v.values.copy()
        area = hg.area_t.values
        west0 = np.sum((area * eta)[:, :5])
        for _ in range(20):
            eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
        assert np.sum((area * eta)[:, :5]) == pytest.approx(west0, rel=1e-12)
