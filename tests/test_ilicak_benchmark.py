"""Four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design): definitions, writers, readers,
metrics. No simulation longer than a few steps."""
from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")


def test_lock_exchange_definition():
    from validation.ilicak_benchmark import cases
    c = cases.get("lock")
    assert (c.nx, c.ny, c.nz) == (128, 1, 20) and np.allclose(c.dx, 500.0) and np.allclose(c.H, 20.0)
    assert np.allclose(c.dz, 1.0) and (c.dt, c.hours) == (10.0, 17.0) and c.nu_h == (0.1, 10.0, 100.0)
    T = c.T_fn(c.x[None, None, :], c.y[None, :, None], c.zc[:, None, None])
    assert T.shape == (20, 1, 128) and (T[:, :, :64] == 5.0).all() and (T[:, :, 64:] == 30.0).all()
    assert np.isclose(cases.density(np.array(5.0)) - cases.density(np.array(30.0)), 5.0)


def test_overflow_definition():
    from validation.ilicak_benchmark import cases
    c = cases.get("overflow")
    assert (c.nx, c.ny, c.nz) == (200, 1, 100) and np.allclose(c.dz, 20.0)
    d = lambda x: 500 + 0.5 * 1500 * (1 + np.tanh((x - 40e3) / 7e3))          # noqa: E731  Petersen 2015 eq. 24
    np.testing.assert_allclose(c.H[0], d(c.x))
    assert c.H[0, 0] < 501.0 and c.H[0, -1] > 1999.0
    T = c.T_fn(c.x[None, None, :], c.y[None, :, None], c.zc[:, None, None])
    assert (T[:, 0, c.x < 20e3] == 10.0).all() and (T[:, 0, c.x >= 20e3] == 20.0).all()
    assert np.isclose(cases.density(np.array(10.0)) - cases.density(np.array(20.0)), 2.0)
    assert (c.dt, c.hours) == (10.0, 40.0) and c.nu_h == (1.0, 100.0, 1000.0)


def test_internal_wave_definition():
    from validation.ilicak_benchmark import cases
    c = cases.get("iw")
    assert (c.nx, c.ny, c.nz) == (50, 1, 20) and np.allclose(c.dz, 25.0) and np.allclose(c.H, 500.0)
    zc = c.zc
    assert np.isclose(zc[0], -12.5) and np.isclose(zc[-1], -487.5)
    far = c.T_fn(np.array(2.5e3), np.array(0.0), zc)                             # outside the bump: background
    np.testing.assert_allclose(far, 10.1 + 10.0 * (-487.5 - zc) / -487.5)        # Petersen eq. 26
    mid = c.T_fn(np.array(125e3), np.array(0.0), zc)
    # bump: −A sin(π(z + 12.5)/(−475)) ≤ 0 → colder water lifted at x0; zero at the top and bottom centres
    assert np.isclose(mid[0], far[0]) and np.isclose(mid[-1], far[-1]) and (mid[1:-1] < far[1:-1]).all()
    assert np.isclose((far - mid).max(), 2.0, atol=0.05)
    assert (c.dt, c.hours) == (300.0, 200 * 24.0) and c.nu_h == (0.01, 15.0, 150.0)


def test_seamount_definition_matches_validation_seamount():
    from validation.ilicak_benchmark import cases
    from validation.seamount import run_seamount as rs
    c = cases.get("seamount")
    assert (c.nx, c.ny, c.nz) == (64, 48, 20) and c.f0 == 1e-4 and c.nu_h == (0.0,)
    assert np.isclose(c.H.min(), rs.H0 - rs.HS * np.exp(-2 * 3e3 ** 2 / rs.LS ** 2))   # summit cells at (±3, ±3) km
    assert np.isclose(c.H.max(), rs.H0, rtol=1e-3)
    z = np.array([0.0, -1000.0, -4000.0])
    rho = cases.density(c.T_fn(np.array(0.0), np.array(0.0), z))
    np.testing.assert_allclose(rho, rs.RHO0 + rs.DRHO * (1 - np.exp(z / rs.DECAY)))


def test_targets_put_the_extreme_waters_at_layer_centres():
    from validation.ilicak_benchmark import cases
    c = cases.get("lock")
    t = cases.targets(c, 20)
    assert t.size == 21 and (np.diff(t) > 0).all()
    lo, hi = cases.density(np.array(30.0)), cases.density(np.array(5.0))
    assert np.isclose(0.5 * (t[0] + t[1]), lo) and np.isclose(0.5 * (t[-2] + t[-1]), hi)


def test_series_round_trip(tmp_path):
    from validation.ilicak_benchmark import series
    nt, nz, ny, nx = 2, 3, 1, 4
    s = dict(t=np.array([0.0, 900.0]), x=np.arange(nx) + 0.5, y=np.array([0.5]), dx=np.ones(nx), dy=np.ones(ny),
             H=np.full((ny, nx), 3.0), h=np.ones((nt, nz, ny, nx)), T=np.zeros((nt, nz, ny, nx)),
             u=np.zeros((nt, nz, ny, nx + 1)), v=np.zeros((nt, nz, ny + 1, nx)), eta=np.zeros((nt, ny, nx)))
    p = series.save(tmp_path / "series.npz", s, dict(model="x", wall_s=1.5))
    s2, meta = series.load(p)
    assert meta["wall_s"] == 1.5 and set(s2) == set(series.KEYS)
    with pytest.raises(ValueError, match="shape"):
        series.save(tmp_path / "bad.npz", {**s, "u": np.zeros((nt, nz, ny, nx))}, {})
