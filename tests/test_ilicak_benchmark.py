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


def test_mom6_source_centres_reproduce_the_case_interfaces():
    from validation.ilicak_benchmark import cases, mom6
    for name in ("lock", "overflow", "iw", "seamount"):
        c = cases.get(name)
        cen = mom6.source_centres(c.dz)
        np.testing.assert_allclose(0.5 * (cen[1:] + cen[:-1]), np.cumsum(c.dz)[:-1], atol=1e-9)


def test_mom6_input_free_slip_linear_eos_and_coordinates():
    from validation.ilicak_benchmark import cases, mom6
    c = cases.get("lock")
    txt = mom6.mom_input(c, "rho", 10.0, 10.0, 17.0)
    p = dict(l.split(" = ", 1) for l in txt.splitlines() if " = " in l)
    assert p["NIGLOBAL"] == "128" and p["NJGLOBAL"] == "1" and p["NK"] == "20" and p["REENTRANT_Y"] == "False"
    assert p["BOTTOMDRAGLAW"] == "True" and p["CDRAG"] == "0.0" and p["KV_BBL_MIN"] == "0.0" and p["NOSLIP"] == "False"
    assert p["DRHO_DT"] == "-0.2" and p["DRHO_DS"] == "0.0" and p["RHO_REF_LINEAR_EOS"] == "1000.0"
    assert p["G_EARTH"] == "9.81" and p["KH"] == "10.0" and p["KV"] == "0.0001" and p["KHTR"] == "0.0" and p["KD"] == "0.0"
    assert p["REGRIDDING_COORDINATE_MODE"] == '"RHO"' and p["COORD_CONFIG"] == '"layer_ref"'
    t = cases.targets(c, 20)
    assert np.isclose(float(p["LIGHTEST_DENSITY"]), 0.5 * (t[0] + t[1]))
    assert np.isclose(float(p["DENSITY_RANGE"]), 0.5 * (t[-2] + t[-1]) - 0.5 * (t[0] + t[1]))
    assert p["TRACER_ADVECTION_SCHEME"] == '"PPM:H3"' and p["KE_SCHEME"] == '"KE_ARAKAWA"'
    assert p["DT"] == "10.0" and p["DAYMAX"] == "61200.0" and p["TIMEUNIT"] == "1.0" and p["SAVE_INITIAL_CONDS"] == "True"
    z = dict(l.split(" = ", 1) for l in mom6.mom_input(c, "zstar", 10.0, 10.0, 17.0).splitlines() if " = " in l)
    assert z["REGRIDDING_COORDINATE_MODE"] == '"ZSTAR"' and "COORD_CONFIG" not in z
    s = dict(l.split(" = ", 1) for l in mom6.mom_input(cases.get("seamount"), "sigma", 0.0, 600.0, 240.0)
             .splitlines() if " = " in l)
    assert s["REGRIDDING_COORDINATE_MODE"] == '"SIGMA"' and float(s["F_0"]) == 1e-4 and s["NJGLOBAL"] == "48"


def _fake_mom6_output(d, c, nt=2):
    """prog.nc + MOM_IC.nc in MOM6's layout (surface first, u on NI+1 faces) holding a known field."""
    nz = c.nz
    T = np.arange(nz, dtype=float)[:, None, None] * np.ones((nz, c.ny, c.nx))       # T = k: k = 0 at the surface
    h = np.broadcast_to(c.dz[:, None, None], (nz, c.ny, c.nx))
    for fname, times, tvar in (("prog.nc", np.arange(1, nt + 1) * c.out_s / 86400.0, "temp"),
                               ("MOM_IC.nc", np.array([0.0]), "Temp")):
        with netCDF4.Dataset(d / fname, "w") as ds:
            for n, sz in (("Time", None), ("zl", nz), ("yh", c.ny), ("xh", c.nx), ("xq", c.nx + 1), ("yq", c.ny + 1)):
                ds.createDimension(n, sz)
            ds.createVariable("Time", "f8", ("Time",))[:] = times
            ds["Time"].units = "days since 0001-01-01 00:00:00"
            ds.createVariable(tvar, "f8", ("Time", "zl", "yh", "xh"))[:] = np.broadcast_to(T, (times.size,) + T.shape)
            ds.createVariable("h", "f8", ("Time", "zl", "yh", "xh"))[:] = np.broadcast_to(h, (times.size,) + h.shape)
            ds.createVariable("u", "f8", ("Time", "zl", "yh", "xq"))[:] = 0.0
            ds.createVariable("v", "f8", ("Time", "zl", "yq", "xh"))[:] = 0.0
            if fname == "prog.nc":
                ds.createVariable("SSH", "f8", ("Time", "yh", "xh"))[:] = 0.0


def test_mom6_reader_puts_the_initial_state_first_and_keeps_surface_first(tmp_path):
    from validation.ilicak_benchmark import cases, mom6
    c = cases.get("lock")
    _fake_mom6_output(tmp_path, c)
    s = mom6.read(tmp_path, "lock")
    assert s["t"].tolist() == [0.0, 900.0, 1800.0]
    assert s["T"].shape == (3, 20, 1, 128) and s["T"][0, 0, 0, 0] == 0.0 and s["T"][0, -1, 0, 0] == 19.0
    np.testing.assert_allclose(s["h"][1].sum(0), 20.0)


CROCO_SRC = __import__("pathlib").Path.home() / "pyOCEAN_refs/croco/OCEAN"


@pytest.mark.skipif(not CROCO_SRC.exists(), reason="CROCO source not on disk")
def test_croco_in_and_cppdefs_have_uniform_sigma_and_no_diffusion():
    from validation.ilicak_benchmark import cases, croco
    cpp = croco.cppdefs("overflow", (CROCO_SRC / "cppdefs.h").read_text())
    assert "#define PYO_OVERFLOW" in cpp and "#undef REGIONAL" in cpp and "#define REGIONAL" not in cpp
    block = cpp[cpp.index("defined PYO_LOCK"):cpp.index("END OF CONFIGURATION CHOICE")]
    for key in ("NEW_S_COORD", "UV_VIS2", "UV_HADV_UP3", "UV_VADV_SPLINES", "TS_HADV_RSUP3", "TS_VADV_SPLINES",
                "OUT_DOUBLE", "ANA_SMFLUX", "ANA_STFLUX", "ANA_BTFLUX", "NO_FRCFILE"):
        assert f"# define {key}" in block, key
    for key in ("ANA_GRID", "ANA_INITIAL", "SALINITY", "NONLIN_EOS", "TS_DIF2", "UV_VIS_SMAGO", "LMD_MIXING",
                "GLS_MIXING", "MASKING", "UV_COR"):
        assert f"# undef  {key}" in block, key
    par = croco.param("overflow", (CROCO_SRC / "param.h").read_text())
    assert "parameter (LLm0=200, MMm0=1, N=100)" in par
    txt = croco.croco_in(cases.get("overflow"), 100.0, 10.0, 40.0)
    assert "0.d0 0.d0 1.e16" in txt and "1.e-4 0." in txt and "100.0 0." in txt
    assert "0. 0. 0. 0. 0." in txt and "0. 0.0 0. 0.2 0." in txt
    lines = txt.splitlines()
    ts = lines[lines.index("time_stepping: NTIMES dt[sec] NDTFAST NINFO") + 1].split()
    assert ts[:2] == ["14400", "10.0"]
    sm = croco.cppdefs("seamount", (CROCO_SRC / "cppdefs.h").read_text())
    assert "# define UV_COR" in sm[sm.index("defined PYO_SEAMOUNT"):]


def test_croco_grid_and_initial_files_hold_the_case(tmp_path):
    from validation.ilicak_benchmark import cases, croco
    c = cases.get("overflow")
    croco.write_run("overflow", 100.0, tmp_path)
    with netCDF4.Dataset(tmp_path / "croco_grd.nc") as ds:
        h = ds["h"][:]
        assert h.shape == (3, 202) and ds["spherical"][:].tobytes().strip() in (b"F", b"f")
        np.testing.assert_allclose(h[1, 1:-1], c.H[0])
        np.testing.assert_allclose(ds["pm"][:], 1 / 1e3)
    with netCDF4.Dataset(tmp_path / "croco_ini.nc") as ds:
        T = ds["temp"][0]
        assert T.shape == (100, 3, 202) and ds["scrum_time"].units.startswith("second")
        assert (T[:, 1, 1] == 10.0).all() and (T[:, 1, -2] == 20.0).all()   # s_rho bottom first; x < 20 km cold


def _fake_croco_his(d, c, nt=2):
    """A history file in CROCO's layout: s_rho bottom first, ghost rho columns/rows, u on LLm+1 faces."""
    N, L, M = c.nz, c.nx, c.ny
    with netCDF4.Dataset(d / "croco_his.nc", "w") as ds:
        for n, sz in (("time", None), ("s_rho", N), ("s_w", N + 1), ("eta_rho", M + 2), ("xi_rho", L + 2),
                      ("xi_u", L + 1), ("eta_v", M + 1)):
            ds.createDimension(n, sz)
        ds.createVariable("scrum_time", "f8", ("time",))[:] = np.arange(nt) * c.out_s
        ds.createVariable("h", "f8", ("eta_rho", "xi_rho"))[:] = np.pad(c.H, 1, mode="edge")
        ds.createVariable("s_w", "f8", ("s_w",))[:] = np.linspace(-1, 0, N + 1)
        ds.createVariable("Cs_w", "f8", ("s_w",))[:] = -np.linspace(-1, 0, N + 1) ** 2
        ds.createVariable("hc", "f8", ())[:] = 1e16
        ds.createVariable("zeta", "f8", ("time", "eta_rho", "xi_rho"))[:] = 0.0
        k = np.arange(N, dtype=float)[:, None, None] * np.ones((N, M + 2, L + 2))   # temp = k, k = 0 at the bed
        ds.createVariable("temp", "f8", ("time", "s_rho", "eta_rho", "xi_rho"))[:] = np.broadcast_to(k, (nt,) + k.shape)
        ds.createVariable("u", "f8", ("time", "s_rho", "eta_rho", "xi_u"))[:] = 0.0
        ds.createVariable("v", "f8", ("time", "s_rho", "eta_v", "xi_rho"))[:] = 0.0


def test_croco_reader_flips_to_surface_first_and_drops_ghosts(tmp_path):
    from validation.ilicak_benchmark import cases, croco
    c = cases.get("lock")
    _fake_croco_his(tmp_path, c)
    s = croco.read(tmp_path, "lock")
    assert s["t"].tolist() == [0.0, 900.0] and s["T"].shape == (2, 20, 1, 128) and s["u"].shape == (2, 20, 1, 129)
    assert s["T"][0, 0, 0, 0] == 19.0 and s["T"][0, -1, 0, 0] == 0.0                # surface first
    np.testing.assert_allclose(s["h"][0], 1.0, rtol=1e-12)                          # uniform σ: 20 m / 20


def test_mitgcm_data_and_size_for_zstar_os7mp_free_slip():
    from validation.ilicak_benchmark import cases, mitgcm
    c = cases.get("lock")
    size = mitgcm.size_h(c)
    for s in ("sNx = 129", "sNy =   1", "OLx =   4", "OLy =   4", "Nr  =  20"):
        assert s in size, s
    txt = mitgcm.data(c, 10.0, 10.0, 17.0)
    for s in ("eosType='LINEAR'", "tAlpha=2.E-4", "sBeta=0.", "rhoNil=1000.", "rhoConst=1000.", "gravity=9.81",
              "viscAh=10.0", "viscAr=1.E-4", "no_slip_sides=.FALSE.", "no_slip_bottom=.FALSE.", "diffKhT=0.",
              "diffKrT=0.", "tempAdvScheme=7", "staggerTimeStep=.TRUE.", "select_rStar=2", "nonlinFreeSurf=4",
              "exactConserv=.TRUE.", "implicitFreeSurface=.TRUE.", "hFacMin=0.1", "deltaT=10.0", "endTime=61210.0",
              "delX=129*500.0", "delR=20*1.0", "saltStepping=.FALSE.", "readBinaryPrec=64"):
        assert s in txt, s
    assert "tRef=20*0.0" in txt and "bottomDragLinear=0." in txt and "bottomDragQuadratic=0." in txt


def test_mitgcm_binary_inputs(tmp_path):
    from validation.ilicak_benchmark import cases, mitgcm
    c = cases.get("overflow")
    mitgcm.write_run("overflow", 100.0, tmp_path)
    b = np.fromfile(tmp_path / "bathy.bin", ">f8").reshape(1, 201)
    np.testing.assert_allclose(-b[0, :200], c.H[0])
    assert b[0, 200] == 0.0                                                      # the land column (east wall)
    T = np.fromfile(tmp_path / "theta.bin", ">f8").reshape(100, 1, 201)
    assert (T[:, 0, 0] == 10.0).all() and (T[:, 0, 199] == 20.0).all()


def _fake_mitgcm_mnc(d, c, nt=2, n_land=1):
    """MNC diagnostics in MITgcm's layout: surface first, one land column east, t = 0 absent (snapshots)."""
    Nx = c.nx + n_land
    with netCDF4.Dataset(d / "diag3d.0000000000.t001.nc", "w") as ds:
        for n, sz in (("T", None), ("Z", c.nz), ("Y", c.ny), ("X", Nx), ("Xp1", Nx + 1), ("Yp1", c.ny + 1)):
            ds.createDimension(n, sz)
        ds.createVariable("T", "f8", ("T",))[:] = np.arange(1, nt + 1) * c.out_s
        k = np.arange(c.nz, dtype=float)[:, None, None] * np.ones((c.nz, c.ny, Nx))
        ds.createVariable("THETA", "f8", ("T", "Z", "Y", "X"))[:] = np.broadcast_to(k, (nt,) + k.shape)
        ds.createVariable("UVEL", "f8", ("T", "Z", "Y", "Xp1"))[:] = 0.0
        ds.createVariable("VVEL", "f8", ("T", "Z", "Yp1", "X"))[:] = 0.0
        hf = np.ones((c.nz, c.ny, Nx))
        hf[:, :, -1] = 0.0
        ds.createVariable("hFactorC", "f8", ("T", "Z", "Y", "X"))[:] = np.broadcast_to(hf, (nt,) + hf.shape)
    with netCDF4.Dataset(d / "diag2d.0000000000.t001.nc", "w") as ds:
        for n, sz in (("T", None), ("Y", c.ny), ("X", Nx)):
            ds.createDimension(n, sz)
        ds.createVariable("T", "f8", ("T",))[:] = np.arange(1, nt + 1) * c.out_s
        ds.createVariable("ETAN", "f8", ("T", "Y", "X"))[:] = 0.0
    with netCDF4.Dataset(d / "grid.t001.nc", "w") as ds:
        for n, sz in (("Z", c.nz), ("Y", c.ny), ("X", Nx)):
            ds.createDimension(n, sz)
        ds.createVariable("hFacC", "f8", ("Z", "Y", "X"))[:] = hf
        ds.createVariable("drF", "f8", ("Z",))[:] = c.dz


def test_mitgcm_reader_adds_t0_drops_land_and_keeps_surface_first(tmp_path):
    from validation.ilicak_benchmark import cases, mitgcm
    c = cases.get("lock")
    _fake_mitgcm_mnc(tmp_path, c)
    mitgcm.write_run("lock", 10.0, tmp_path)                                                # theta.bin for t = 0
    s = mitgcm.read(tmp_path, "lock")
    assert s["t"].tolist() == [0.0, 900.0, 1800.0] and s["T"].shape == (3, 20, 1, 128)
    assert s["T"][1, 0, 0, 0] == 0.0 and s["T"][1, -1, 0, 0] == 19.0                        # k = 0 at the surface
    assert s["T"][0, 0, 0, 0] == 5.0 and s["T"][0, 0, 0, -1] == 30.0                        # t = 0 from theta.bin
    np.testing.assert_allclose(s["h"].sum(1), 20.0)


SMOKE = [("pyocean", "zstar"), ("pyocean", "sigma"), ("pyocean", "hybrid"), ("mom6", "zstar"), ("mom6", "rho"),
         ("croco", "sigma"), ("mitgcm", "zstar")]


@pytest.mark.parametrize("model,coord", SMOKE)
def test_smoke_lock_exchange_one_hour(model, coord):
    from validation.ilicak_benchmark import cases, metrics, series
    p = cases.run_dir("lock", model, coord, 10.0, "smoke") / "series.npz"
    if not p.exists():
        pytest.skip("smoke run not on disk")
    s, meta = series.load(p)
    assert s["t"][0] == 0.0 and np.isclose(s["t"][-1], 3600.0) and meta["wall_s"] > 0
    np.testing.assert_allclose(s["h"].sum(1), 20.0 + s["eta"], atol=1e-6)            # column = depth + η
    T0 = np.where(s["h"][0] > 0, s["T"][0], np.nan)
    assert np.nanmin(T0) == 5.0 and np.nanmax(T0) == 30.0                              # the reader's initial state
    assert np.isfinite(np.where(s["h"] > 0, s["T"], 0.0)).all()                        # extrema are results, not checked
    assert np.isfinite(metrics.rpe(s)).all()          # a falling RPE (new extrema, Ilicak Fig. 7) is a result
    assert metrics.max_speed(s)[-1] > 0.1                                              # the fronts are moving


def test_campaign_matrix_and_commands():
    from validation.ilicak_benchmark import campaign
    two_d = [("pyocean", "zstar"), ("pyocean", "sigma"), ("pyocean", "hybrid"), ("mom6", "zstar"), ("mom6", "rho"),
             ("croco", "sigma"), ("mitgcm", "zstar")]
    for c in ("lock", "overflow", "iw"):
        assert campaign.CONFIGS[c] == two_d
    assert campaign.CONFIGS["seamount"] == [("pyocean", "sigma"), ("pyocean", "zstar"), ("mom6", "sigma"),
                                            ("mom6", "zstar"), ("croco", "sigma"), ("mitgcm", "zstar")]
    cmd = campaign.command("lock", "mom6", "rho", 10.0, dt=20.0, hours=1.0, tag="cost")
    assert cmd[-7:] == ["run", "lock", "rho", "10", "20", "1", "cost"] and cmd[0].endswith("run_mom6.sh")
    py = campaign.command("iw", "pyocean", "hybrid", 15.0)
    assert py[1].endswith("run_pyocean.py") and py[2:5] == ["iw", "hybrid", "15"]


def test_stable_rejects_nan_and_fast_flow(tmp_path):
    from validation.ilicak_benchmark import campaign, series
    s = _flat_series([1.0], [5.0], nx=2)
    series.save(tmp_path / "ok.npz", s, {})
    assert campaign.stable(tmp_path / "ok.npz")
    s["u"] = s["u"] + 20.0
    series.save(tmp_path / "fast.npz", s, {})
    assert not campaign.stable(tmp_path / "fast.npz")
    s["u"] = s["u"] * 0
    s["T"] = s["T"] * np.nan
    series.save(tmp_path / "nan.npz", s, {})
    assert not campaign.stable(tmp_path / "nan.npz")


def test_max_speed_ignores_vanished_layers():
    from validation.ilicak_benchmark import metrics
    s = _flat_series([10.0, 0.001], [5.0, 5.0], nx=2)
    s["u"][0, 1, 0, 1] = 9.0                                   # fast flow in a 1-mm layer: carries no mass
    s["u"][0, 0, 0, 1] = 0.5
    assert np.isclose(metrics.max_speed(s)[0], 0.5)
