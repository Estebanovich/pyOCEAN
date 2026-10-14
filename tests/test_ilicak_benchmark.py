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


def _flat_series(h_layers, T_layers, dx=1.0, nx=1):
    """One-record series with layers (surface first) h_layers [m] and T_layers [°C] in every column."""
    h = np.asarray(h_layers, float)[None, :, None, None] * np.ones((1, 1, 1, nx))
    T = np.asarray(T_layers, float)[None, :, None, None] * np.ones((1, 1, 1, nx))
    nz = h.shape[1]
    return dict(t=np.array([0.0]), x=np.arange(nx) * dx + dx / 2, y=np.array([0.5]), dx=np.full(nx, dx),
                dy=np.array([1.0]), H=h[0].sum(0), h=h, T=T, u=np.zeros((1, nz, 1, nx + 1)),
                v=np.zeros((1, nz, 2, nx)), eta=np.zeros((1, 1, nx)))


def test_rpe_does_not_depend_on_how_the_column_is_split():
    from validation.ilicak_benchmark import metrics
    a = metrics.rpe(_flat_series(np.full(20, 1.0), [30.0] * 10 + [5.0] * 10))          # 20 z-levels
    b = metrics.rpe(_flat_series([10.0, 10.0], [30.0, 5.0]))                            # two layers
    c = metrics.rpe(_flat_series([10.0, 0.0, 4.0, 6.0], [30.0, 17.0, 5.0, 5.0]))       # σ-like, one vanished layer
    np.testing.assert_allclose([b[0], c[0]], a[0], rtol=1e-9)
    # analytic: g Σ ρ ∫ z dz, light 10 m on dense 10 m, unit area
    exact = 9.81 * (999.0 * 0.5 * 10.0 ** 2 + 994.0 * 0.5 * (20.0 ** 2 - 10.0 ** 2))
    np.testing.assert_allclose(a[0], exact, rtol=1e-9)


def test_rpe_hypsometry_exact_for_two_column_steps():
    from validation.ilicak_benchmark import metrics
    # columns of 10 m and 30 m, unit width: uniform density fills the deep column's bottom 20 m first
    s = _flat_series([10.0], [10.0], nx=2)
    s["h"] = np.array([10.0, 30.0])[None, None, None, :]
    s["T"] = np.full((1, 1, 1, 2), 10.0)
    s["H"] = np.array([[10.0, 30.0]])
    z, A = metrics.hypsometry(s["H"], s["dx"], s["dy"])
    assert z[0] == 0.0 and A[0] == 1.0 and A[-1] == 2.0 and np.diff(z).max() <= 40.0 / 50000 * 1.0001
    rho = 998.0
    exact = 9.81 * rho * (0.5 * 20.0 ** 2 + 2 * 0.5 * (30.0 ** 2 - 20.0 ** 2))
    np.testing.assert_allclose(metrics.rpe(s)[0], exact, rtol=1e-9)


def test_drpe_dt_is_per_unit_surface_area():
    from validation.ilicak_benchmark import metrics
    s = _flat_series([1.0], [5.0], dx=500.0, nx=4)
    s["t"] = np.array([0.0, 3600.0, 7200.0])
    rpe = np.array([0.0, 2000.0, 4000.0])
    assert np.isclose(metrics.drpe_dt(s, rpe, 0.0, 7200.0), (4000.0 / 7200.0) / (4 * 500.0 * 1.0))


def test_lock_fronts_track_a_steady_gravity_current():
    from validation.ilicak_benchmark import metrics
    nx, dx, U = 128, 500.0, 0.5
    t = np.arange(0, 9) * 3600.0
    nz = 20
    h = np.ones((t.size, nz, 1, nx))
    T = np.full((t.size, nz, 1, nx), 30.0)
    x = np.arange(nx) * dx + dx / 2
    for n, tt in enumerate(t):
        T[n, 10:, 0, x < 32e3 + U * tt] = 5.0           # dense bottom half advancing east
        T[n, 10:, 0, x < 32e3] = 5.0
        T[n, :10, 0, (x < 32e3) & (x >= 32e3 - U * tt)] = 30.0
        T[n, :10, 0, x < 32e3 - U * tt] = 5.0           # light top half advancing west
    s = dict(t=t, x=x, y=np.array([250.0]), dx=np.full(nx, dx), dy=np.array([500.0]), H=np.full((1, nx), 20.0),
             h=h, T=T, u=np.zeros((t.size, nz, 1, nx + 1)), v=np.zeros((t.size, nz, 2, nx)),
             eta=np.zeros((t.size, 1, nx)))
    f = metrics.lock_fronts(s)
    for k in ("dense_nose", "light_nose", "dense_body", "light_body"):
        assert abs(f[k] - U) < 0.02, (k, f[k])


def test_overflow_descent_and_iw_amplitude_and_t_range():
    from validation.ilicak_benchmark import metrics
    s = _flat_series([10.0, 10.0], [20.0, 10.0], nx=3)
    s["T"][0, :, 0, 0] = [20.0, 20.0]
    xc, dep = metrics.overflow_descent(s)
    assert np.allclose(xc, s["x"][1:].mean()) and np.allclose(dep, 15.0)
    # isotherm 11.25 °C between centres at 5 m and 15 m: record 0 (20, 10 °C) → 13.75 m deep;
    # record 1 (12.5, 10 °C: colder top at x0) → 10 m deep, i.e. lifted 3.75 m
    s2 = _flat_series([10.0, 10.0], [20.0, 10.0], dx=125e3, nx=2)
    s2 = {k: (np.concatenate([v, v]) if k in ("h", "T", "u", "v", "eta") else v) for k, v in s2.items()}
    s2["t"], s2["x"] = np.array([0.0, 3600.0]), np.array([0.0, 125e3])
    s2["T"][1, 0, 0, 1] = 12.5
    np.testing.assert_allclose(metrics.iw_amplitude(s2, T_iso=11.25), [0.0, 3.75], atol=1e-12)
    lo, hi = metrics.t_range(s)
    assert (lo, hi) == (10.0, 20.0)


@pytest.mark.parametrize("name,coord", [("lock", "zstar"), ("lock", "hybrid"), ("overflow", "sigma"),
                                        ("iw", "hybrid"), ("seamount", "zstar")])
def test_pyocean_builds_each_case_and_records_surface_first(name, coord):
    from validation.ilicak_benchmark import cases, run_pyocean
    c = cases.get(name)
    m, st = run_pyocean.build(c, coord, c.nu_h[-1])
    r = run_pyocean.record(m, st)
    nz = st.h.shape[0]
    assert r["h"].shape == (nz, c.ny, c.nx) and r["u"].shape == (nz, c.ny, c.nx + 1)
    np.testing.assert_allclose(r["h"].sum(0), c.H, rtol=1e-9)                     # rest, η = 0
    # surface first: the top layer of the lock's east half is the warm water, the top of the IW column the warmest
    if name == "lock":
        assert np.allclose(r["T"][0, 0, -1], 30.0) and np.allclose(r["T"][-1, 0, 0], 5.0)
    if name == "iw":
        assert r["T"][0, 0, 0] > r["T"][-1, 0, 0]
    T0 = c.T_fn(c.x[None, None, :], c.y[None, :, None], c.zc[:, None, None])
    assert np.nanmin(r["T"]) >= T0.min() - 1e-6 and np.nanmax(r["T"]) <= T0.max() + 1e-6


def test_pyocean_short_run_writes_a_series(tmp_path, monkeypatch):
    from validation.ilicak_benchmark import cases, run_pyocean, series
    monkeypatch.setattr(cases, "RUNS", tmp_path)
    p = run_pyocean.run("lock", "zstar", 10.0, hours=60.0 / 3600.0, threads=1)     # 6 steps
    s, meta = series.load(p)
    assert s["t"][0] == 0.0 and s["t"][-1] == 60.0 and meta["model"] == "pyOCEAN" and meta["n_steps"] == 6
    heat = (s["h"] * s["T"]).sum(axis=(1, 2, 3))
    np.testing.assert_allclose(heat, heat[0], rtol=1e-10)
