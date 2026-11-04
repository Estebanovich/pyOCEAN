"""
Tests of the NWA12 cut case (validation/nwa12_cut; regional chain plan Task 9) and of the library pieces it added
(``forcing.ocean_init.exterior_provider``/``source_level_grid``, ``diagnostics.domain_budget``, the curvilinear
pre-flight warning).

The small regional case (Review Focus 5) is the real pipeline of ``case.build`` on synthetic files written in the
layouts of the real products: a rotated 12 × 10 supergrid with land, an 8-level vgrid, a GLORYS12 subset (daily,
NaN under land), ERA5 hourly instant/accum files with a land-sea mask, a GloFAS v4 discharge file with a river and
TPXO-like constants for the 10 NWA12 constituents; tides ramped over half a day.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.model.contract import PyOceanConfigWarning
from tests.test_supergrid import _rotated_supergrid
from validation.nwa12_cut import case as cs

T0 = np.datetime64("2017-05-01T00:00", "ns")
NX, NY, NZ = 12, 10, 8
DZ = np.array([2.0, 3.0, 5.0, 10.0, 20.0, 40.0, 60.0, 80.0])     # surface first, Σ = 220 m
LON0, LAT0, D = -70.0, 38.0, 10_000.0


# ---------------------------------------------------------------------------------------------------- synthetic case
def _depth() -> np.ndarray:
    j, i = np.mgrid[0:NY, 0:NX]
    d = 40.0 + 14.0 * i + 3.0 * j
    d[:, :2] = 0.0                                   # coast in the west: the west edge is closed
    d[(j >= 7) & (i < 5)] = 0.0                      # a cape reaching the north edge (partly open north edge)
    d[2, 2] = 2.5                                    # MINIMUM_DEPTH rule: 1 < d < 4 → 4
    return d


def _write_case(tmp: Path) -> cs.Case:
    hg = _rotated_supergrid(NX, NY, D, 15.0, lon0=LON0, lat0=LAT0)
    hg.to_netcdf(tmp / "hgrid.nc")
    xr.Dataset({"depth": (("ny", "nx"), _depth().astype("f4"))}).to_netcdf(tmp / "topog.nc")
    xr.Dataset({"dz": ("nz", DZ)}).to_netcdf(tmp / "vgrid.nc")
    lon_n, lat_n = hg["x"].values, hg["y"].values
    lo = np.arange(np.floor(lon_n.min() * 12) / 12 - 0.25, lon_n.max() + 0.3, 1 / 12)
    la = np.arange(np.floor(lat_n.min() * 12) / 12 - 0.25, lat_n.max() + 0.3, 1 / 12)
    # GLORYS12 (time, depth, latitude, longitude), float32, NaN west of the coast
    days = np.arange(-1, 4)
    dep = np.array([0.494, 5.0, 15.0, 40.0, 80.0, 150.0, 250.0], np.float32)
    t, z, y, x = np.meshgrid(days, dep.astype(float), la, lo, indexing="ij")
    land = (x < LON0 + 0.2)
    th = 18.0 - 0.04 * z + 0.5 * (y - LAT0) + 0.2 * t
    so = 35.0 - 0.002 * z + 0.1 * (x - LON0)
    uo = 0.10 + 0.05 * np.sin(y) + 0.01 * t
    vo = -0.05 + 0.03 * np.cos(x)
    for f in (th, so, uo, vo):
        f[land] = np.nan
    zos = 0.05 * (x[:, 0] - LON0) + 0.01 * t[:, 0]
    zos[land[:, 0]] = np.nan
    gl = xr.Dataset({"thetao": (("time", "depth", "latitude", "longitude"), th.astype("f4")),
                     "so": (("time", "depth", "latitude", "longitude"), so.astype("f4")),
                     "uo": (("time", "depth", "latitude", "longitude"), uo.astype("f4")),
                     "vo": (("time", "depth", "latitude", "longitude"), vo.astype("f4")),
                     "zos": (("time", "latitude", "longitude"), zos.astype("f4"))},
                    coords={"time": T0 + days * np.timedelta64(1, "D"), "depth": dep,
                            "latitude": la.astype("f4"), "longitude": lo.astype("f4")})
    gl.to_netcdf(tmp / "glorys.nc")
    # ERA5 hourly, CDS 'new' layout (valid_time, latitude descending), instant + accum files and a land-sea mask
    (tmp / "era5").mkdir()
    lat_e = np.arange(np.ceil(lat_n.max() * 4) / 4 + 0.5, lat_n.min() - 0.75, -0.25)
    lon_e = np.arange(np.floor(lon_n.min() * 4) / 4 - 0.5, lon_n.max() + 0.75, 0.25)
    vt = T0 - np.timedelta64(1, "D") + np.arange(5 * 24) * np.timedelta64(1, "h")
    hr = np.arange(vt.size)[:, None, None]
    Y, X = np.meshgrid(lat_e, lon_e, indexing="ij")
    shp = (vt.size,) + Y.shape
    rng = np.random.default_rng(3)
    inst = {"u10": 6.0 + 2.0 * np.sin(hr / 7.0) + 0.0 * Y, "v10": -3.0 + np.cos(hr / 5.0) + 0.1 * X - 0.1 * X,
            "t2m": 290.0 + 2.0 * np.sin(2 * np.pi * hr / 24.0) + 0.3 * (Y - LAT0),
            "d2m": 285.0 + 0.5 * rng.standard_normal(shp), "msl": 101300.0 + 50.0 * (X - LON0) + 0.0 * hr}
    acc = {"ssrd": 3600.0 * np.maximum(0.0, 600.0 * np.sin(2 * np.pi * (hr - 6) / 24.0)) + 0.0 * Y,
           "strd": 3600.0 * (360.0 + 0.0 * Y + 0.0 * hr), "tp": 1e-4 * (1.0 + np.sin(hr / 3.0)) + 0.0 * Y}

    def write(group: dict, kind: str) -> None:
        ds = xr.Dataset({k: (("valid_time", "latitude", "longitude"), np.broadcast_to(v, shp).astype("f4"))
                         for k, v in group.items()},
                        coords={"valid_time": vt, "latitude": lat_e, "longitude": lon_e})
        ds.to_netcdf(tmp / "era5" / f"era5_syn_2017-05_{kind}.nc")
    write(inst, "instant")
    write(acc, "accum")
    lsm = np.where(X < LON0 + 0.15, 1.0, 0.0)[None]
    xr.Dataset({"lsm": (("valid_time", "latitude", "longitude"), lsm.astype("f4"))},
               coords={"valid_time": vt[:1], "latitude": lat_e, "longitude": lon_e}).to_netcdf(tmp / "era5" / "lsm.nc")
    # GloFAS v4 (valid_time = end of the daily mean), lon 0–360, sea NaN, one river reaching the west coast
    (tmp / "glofas").mkdir()
    lat_g = np.round(np.arange(lat_n.max() + 0.3, lat_n.min() - 0.3, -0.05), 3)
    lon_g = np.round(np.arange(lon_n.min() - 0.5, lon_n.max() + 0.3, 0.05), 3)
    LG, AG = np.meshgrid(lon_g, lat_g)
    sea = LG > LON0 + 0.2
    nd = 7
    q = np.where(sea, np.nan, 0.2)[None].repeat(nd, axis=0)
    jr = int(np.argmin(np.abs(lat_g - (LAT0 + 0.3))))
    ir = int(np.flatnonzero(~sea[jr])[-1])
    q[:, jr, ir - 6:ir + 1] = 30.0 + 5.0 * np.arange(7)[None, :] + 2.0 * np.arange(nd)[:, None]
    vtg = T0 - np.timedelta64(1, "D") + np.arange(nd) * np.timedelta64(1, "D")
    xr.Dataset({"avg_dis": (("valid_time", "latitude", "longitude"), q.astype("f4"))},
               coords={"valid_time": vtg, "latitude": lat_g, "longitude": lon_g + 360.0}).to_netcdf(
        tmp / "glofas" / "glofas_v4_consolidated_2017-05.nc")
    return cs.Case(name="syn", hgrid=tmp / "hgrid.nc", topog=tmp / "topog.nc", vgrid=tmp / "vgrid.nc",
                   glorys=(tmp / "glorys.nc",), era5=tmp / "era5", era5_lsm=tmp / "era5" / "lsm.nc",
                   glofas=tmp / "glofas", ldd=None, tpxo=_tpxo(lo, la), tides=None, runs=tmp / "runs",
                   window=(0, NX, 0, NY), points={"p": (float(lat_n.mean()), float(lon_n.mean()))},
                   tide_ramp_days=0.5, t1=T0 + np.timedelta64(4, "D"))


def _tpxo(lo: np.ndarray, la: np.ndarray) -> xr.Dataset:
    """TPXO-like constants of the 10 NWA12 constituents, smooth in space [m, m² s⁻¹]."""
    nc = len(cs.CONSTITUENTS)
    amp = np.array([0.5, 0.1, 0.1, 0.03, 0.08, 0.06, 0.03, 0.01, 0.01, 0.01])
    X, Y = np.meshgrid(lo, la)
    ph = np.deg2rad(30.0 + 10.0 * np.arange(nc))[:, None, None] + 0.3 * (X - LON0)[None]
    h = amp[:, None, None] * np.exp(-1j * ph)
    U = 20.0 * h * np.exp(-0.5j)
    V = 10.0 * h
    return xr.Dataset({"h": (("constituent", "lat", "lon"), h), "U": (("constituent", "lat", "lon_u"), U),
                       "V": (("constituent", "lat_v", "lon"), V),
                       "hu": (("lat", "lon_u"), 60.0 + 0.0 * X), "hv": (("lat_v", "lon"), 55.0 + 0.0 * X)},
                      coords={"constituent": list(cs.CONSTITUENTS), "lat": la, "lon": lo, "lon_u": lo - 1 / 24,
                              "lat_v": la - 1 / 24})


@pytest.fixture(scope="module")
def syn(tmp_path_factory) -> cs.Case:
    return _write_case(tmp_path_factory.mktemp("nwa12_syn"))


def _build(case: cs.Case, days: float = 2.0, dt: float = 1800.0, **kw) -> cs.Built:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)                 # GloFAS substitute without ldd (R11)
        return cs.build(case, "full", dt, days, **kw)


# ---------------------------------------------------------------------------------------------------- window rules
def test_choose_window_rules():
    nj, ni = 60, 70
    lon, lat = np.meshgrid(-80.0 + 0.1 * np.arange(ni), 30.0 + 0.1 * np.arange(nj))
    depth = np.full((nj, ni), 100.0)
    depth[:, :12] = 0.0                                              # coast west of −78.85 °E
    depth[40:, :] = 0.0                                              # coast north of 34 °N
    box = (-79.5, -75.0, 31.0, 35.0)
    # 1. index bounding box of the T points inside the box; open edges = edges with a wet cell
    w = cs.choose_window(lon, lat, depth, box, points={})
    assert w == (5, 51, 10, 51)
    H = cs.mom6_depth(depth[w[2]:w[3], w[0]:w[1]])
    assert cs.open_edges(H) == ("east", "south")
    # 2. a point 4 cells from the open east edge pushes it out to 10 cells; a point near a coast edge does not
    w2 = cs.choose_window(lon, lat, depth, box, points={"near_east": (32.0, -75.4), "near_north": (34.95, -77.0)})
    assert w2[1] - 1 - int(np.argmin(np.abs(lon[0] + 75.4))) == cs.MIN_EDGE_CELLS
    assert w2[3] == w[3] and w2[0] == w[0]
    # 3. no open-edge cell with H ≤ 0: MOM6's masking leaves land (NaN) or ≥ MINIMUM_DEPTH
    d = cs.mom6_depth(np.array([-3.0, 0.0, 0.5, 1.0, 1.5, 3.9, 5.0, 7000.0]))
    np.testing.assert_array_equal(d, [np.nan, np.nan, np.nan, np.nan, 4.0, 4.0, 5.0, 6500.0])
    with pytest.raises(ValueError, match="outside the window"):
        cs.choose_window(lon, lat, depth, box, points={"far": (20.0, -60.0)})
    with pytest.raises(ValueError, match="edge of the parent grid"):
        cs.choose_window(lon, lat, depth, (-79.5, -73.1, 31.0, 35.0), points={"p": (32.0, -73.5)})


def test_edge_runs_shorter_than_four_cells_become_land():
    # Ruling R14: a 2-cell inlet on the west edge (an estuary interior) is land; a 4-cell run and the open sea stay.
    H = np.full((12, 8), 50.0)
    H[:, 0] = np.nan
    H[3:5, 0] = 20.0                                                 # 2-cell estuary run → land
    H[7:11, 0] = 30.0                                                # 4-cell run → open
    H[0, 2:4] = np.nan                                               # south edge: runs of 1 and 4 cells
    H2, cells = cs.edit_edges(H)
    assert cells == [(0, 1), (3, 0), (4, 0)]
    assert np.isnan(H2[3:5, 0]).all() and np.isfinite(H2[7:11, 0]).all() and np.isfinite(H2[0, 4:]).all()
    assert np.array_equal(np.isfinite(H2[1:-1, 1:]), np.isfinite(H[1:-1, 1:]))      # the interior is untouched
    lone = np.full((6, 6), np.nan)
    lone[:, 2:] = 10.0
    lone[2:4, :2] = 10.0                                             # the only west-edge water: a 2-cell inlet
    assert cs.open_edges(lone) == ("west", "east", "south", "north")
    assert cs.open_edges(cs.edit_edges(lone)[0]) == ("east", "south", "north")      # west closes: nothing remains


@pytest.mark.skipif(not cs.CASE.hgrid.exists(), reason="NWA12 grid not on disk")
def test_case_topography_edit_closes_upper_delaware_bay():
    hg, H = cs.grid(cs.CASE)
    edits = cs.topography_edits(hg)
    assert [(e["j"], e["i"]) for e in edits] == [(43, 0), (44, 0)]
    assert all(39.2 < e["lat"] < 39.31 and abs(e["lon"] + 75.44) < 1e-3 for e in edits)
    assert np.isfinite(H[:, 0]).sum() == 23 and cs.open_edges(H) == ("west", "east", "south", "north")


@pytest.mark.skipif(not cs.CASE.hgrid.exists(), reason="NWA12 grid not on disk")
def test_case_window_is_choose_window():
    hg = cs.load_hgrid(cs.CASE)
    w = cs.choose_window(hg["x"].values[1::2, 1::2], hg["y"].values[1::2, 1::2], cs.load_depth(cs.CASE))
    assert w == cs.CASE.window


# ---------------------------------------------------------------------------------------------------- configuration
def test_parity_config_preflight_clean(syn):
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        b = _build(syn, days=1.0)
    assert not [w for w in rec if issubclass(w.category, PyOceanConfigWarning)], [str(w.message) for w in rec]
    m = b.model
    assert m.ale and m.regrid_rule == "zstar" and m.momentum == "vector_invariant"
    assert (m.smagorinsky, m.ah, m.smagorinsky_bi, m.ah_vel_scale, m.ah_bi) == (0.0, 0.0, 0.03, 0.01, 0.0)  # NWA12
    assert m.eos_name == "wright_full" and m.cp == 3992.0 and m.rho0 == 1035.0
    assert (m.obc_tau_in, m.obc_tau_out) == (3 * 86400.0, 360 * 86400.0)
    assert b.edges == ("east", "south", "north") and set(m.obc_mode.values()) == {"radiating"}
    assert m.boundary_tides.velocity_depth == "tpxo"                # R19
    # R21: NWA12's G_EARTH = 9.8 reaches every consumer of g; full PGF mass weighting
    cl = m._closure
    gs = [m.g, m.core.g, m.fs.g, m._pgf.g, m._tidal_potential.g] + [c.g for c in (cl, getattr(cl, "base", None))
                                                                     if c is not None and hasattr(c, "g")]
    assert all(x == 9.8 for x in gs), gs
    assert m.pgf_mass_weight == "full" and m._pgf.mass_weight == "full"
    assert m.sal_beta == 0.05 and m.sal_flather and m.tidal_potential is not None and not m.atm_pressure
    assert m.river_input is not None and m.river_input.Q.sum() > 0
    assert np.nanmin(b.H) == 4.0                                     # MINIMUM_DEPTH
    for bad in (dict(vertical_coordinate="sigma_eulerian"), dict(momentum="flux"), dict(smagorinsky_bi=0.0)):
        with pytest.raises(ValueError, match="curvilinear metric terms"):
            cs.parity_config(syn, b.hg, b.H, 600.0, edges=b.edges, **bad)
    lap = cs.parity_config(syn, b.hg, b.H, 600.0, edges=b.edges, smagorinsky=0.15, smagorinsky_bi=0.0,
                           ah_vel_scale=0.0)                                 # the Laplacian is still accepted
    assert lap["smagorinsky"] == 0.15


def test_preflight_warns_flux_form_on_supergrid(syn):
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    hg, H = cs.grid(syn)
    with pytest.warns(PyOceanConfigWarning, match="curvilinear"):
        SigmaBaroclinic(hg, H, dt=60.0, n_layers=3, ah=10.0)          # σ flux form + constant Laplacian


# ---------------------------------------------------------------------------------------------------- exterior
def test_exterior_provider_zw_and_depth_mean():
    from forcing.ocean_init import exterior_provider, source_level_grid
    d = np.array([1.0, 10.0, 30.0, 60.0])
    vg = source_level_grid(d)
    np.testing.assert_allclose(vg["z_f"].values, [0.0, -5.5, -20.0, -45.0, -75.0])
    np.testing.assert_allclose(vg["z_t"].values, -d)
    times = T0 + np.arange(2) * np.timedelta64(1, "D")
    n = 3
    u = np.array([[1.0, 2.0, 0.5], [1.0, 2.0, 0.5], [3.0, 0.0, 0.5], [5.0, 0.0, 0.5]])     # (k, i) surface first

    def ds(scale):
        return {"t": xr.Dataset({"temp": (("time", "k", "j_t", "i_t"), np.full((2, 4, 1, n), 10.0)),
                                 "salt": (("time", "k", "j_t", "i_t"), np.full((2, 4, 1, n), 35.0)),
                                 "ssh": (("time", "j_t", "i_t"), np.full((2, 1, n), 0.1))}, coords={"time": times}),
                "v": xr.Dataset({"v": (("time", "k", "j_v", "i_t"), scale * np.stack([u[:, None], u[:, None]])
                                       .repeat(2, axis=2))}, coords={"time": times})}
    depth = np.array([40.0, 100.0, 0.0])
    prov = exterior_provider({"south": ds(1.0)}, vg, {"south": depth}, T0)
    ob = prov(43200.0)["south"]
    assert ob.zw.shape == (5, n) and ob.zw[-1, 0] <= -101.0          # moved below the deepest edge cell
    # D = 40: layers 5.5, 14.5, 20 (clipped); D = 100: 5.5, 14.5, 25, 55 (last extended)
    np.testing.assert_allclose(ob.ubar[0], (1 * 5.5 + 1 * 14.5 + 3 * 20.0) / 40.0)
    np.testing.assert_allclose(ob.ubar[1], (2 * 5.5 + 2 * 14.5 + 0 * 25.0 + 0 * 55.0) / 100.0)
    assert ob.ubar[2] == 0.0
    np.testing.assert_allclose(ob.eta, 0.1)


# ---------------------------------------------------------------------------------------------------- small regional runs
def _run(b: cs.Built, st, n0: int, n1: int, budget=None):
    for _ in range(n0, n1):
        st = b.model.step(st)
        if budget is not None:
            budget.add(st)
    return st


def _assert_identical(a, b) -> None:
    from validation.nwa12_cut.run_pyocean import same_state
    same = same_state(a, b)
    assert all(same.values()), same


def test_restart_mid_run_bit_for_bit_small(syn, tmp_path):
    # Review Focus 5: GLORYS exterior, ramped tides, ERA5 bulk and runoff all vary in time across the restart.
    from core.dynamics.sigma_baroclinic import set_threads
    from core.restart import read_restart, write_restart
    set_threads(2)
    dt, n, k = 1800.0, 96, 15                                           # 2 days; restart at 7.5 h:
    assert k * dt < syn.tide_ramp_days * 86400.0                        # during the tidal ramp,
    assert (k * dt) % 86400.0 != 0.0 and (k * dt) % 3600.0 != 0.0       # between daily GLORYS/GloFAS and hourly ERA5
    b = _build(syn, days=2.0, dt=dt)
    ref = _run(b, b.state, 0, n)
    assert ref.n_trunc == 0 and np.isfinite(ref.T).all()
    b1 = _build(syn, days=2.0, dt=dt)
    half = _run(b1, b1.state, 0, k)
    write_restart(tmp_path / "r.nc", b1.model, half)
    b2 = _build(syn, days=2.0, dt=dt)
    st = _run(b2, read_restart(tmp_path / "r.nc", b2.model), k, n)
    _assert_identical(ref, st)
    assert float(np.abs(ref.eta).max()) > 1e-3                          # the tide is in


def test_threads_1_equals_5_small(syn):
    from core.dynamics.sigma_baroclinic import set_threads
    out = []
    for th in (1, 5):
        set_threads(th)
        b = _build(syn, days=0.5, dt=1800.0)
        out.append(_run(b, b.state, 0, 24))
    set_threads(5)
    _assert_identical(*out)


def test_domain_budget_closes_small(syn):
    # Criterion 6 on the small case: volume, heat and salt with open faces, bulk P − E and heat, and rivers.
    from diagnostics.domain_budget import DomainBudget
    from core.dynamics.sigma_baroclinic import set_threads
    set_threads(2)
    b = _build(syn, days=1.0, dt=1800.0)
    bud = DomainBudget(b.model)
    bud.start(b.state)
    _run(b, b.state, 0, 48, bud)
    r = bud.residuals()
    for k in ("V", "T", "S"):
        assert r[k]["rel_terms"] <= 1e-10 and r[k]["rel_gross"] <= r[k]["rel_terms"], (k, r[k])
    assert r["V"]["surf"] != 0.0 and r["V"]["open"] != 0.0 and r["T"]["surf"] != 0.0 and r["T"]["fw"] != 0.0
    assert r["S"]["open"] != 0.0


def test_river_set_is_independent_of_run_length(syn):
    # Ruling R17: the pour points are chosen over the case period, so runs of any length (and their restarts)
    # have the same rivers.
    a, b = _build(syn, days=0.5).river, _build(syn, days=2.0).river
    assert np.array_equal(a.cells, b.cells) and np.array_equal(a.Q, b.Q) and a.cells.shape[0] > 0


def test_domain_budget_accumulators_survive_a_restart(syn, tmp_path):
    # run_pyocean saves DomainBudget.state() next to a restart and reloads it on --start-from: the split
    # accumulation equals the straight one bit for bit.
    from core.restart import read_restart, write_restart
    from diagnostics.domain_budget import DomainBudget
    b = _build(syn, days=1.0, dt=1800.0)
    ref = DomainBudget(b.model)
    ref.start(b.state)
    _run(b, b.state, 0, 20, ref)
    b1 = _build(syn, days=1.0, dt=1800.0)
    first = DomainBudget(b1.model)
    first.start(b1.state)
    st = _run(b1, b1.state, 0, 9, first)
    write_restart(tmp_path / "r.nc", b1.model, st)
    np.savez(tmp_path / "r_budget.npz", **first.state())
    b2 = _build(syn, days=1.0, dt=1800.0)
    second = DomainBudget(b2.model)
    with np.load(tmp_path / "r_budget.npz") as d:
        second.load(d)
    _run(b2, read_restart(tmp_path / "r.nc", b2.model), 9, 20, second)
    assert second.residuals() == ref.residuals() and second.n_steps == 20


def test_parse_overrides():
    from validation.nwa12_cut.run_pyocean import parse_overrides
    assert parse_overrides(["theta_bc=1.0", "regrid=zstar", "cd = 0.002"]) == {"theta_bc": 1.0, "regrid": "zstar",
                                                                             "cd": 0.002}
    with pytest.raises(SystemExit):
        parse_overrides(["theta_bc"])


def test_restart_with_allowed_theta_bc_change(syn, tmp_path):
    # read_restart is strict by default; allow_changed=("theta_bc",) lets a diagnostic run continue a restart with a
    # different θ_bc (warning), and the runner's --override/--allow-restart-change path does it end to end.
    import dataclasses
    from core.restart import read_restart, write_restart
    from validation.nwa12_cut import run_pyocean as rp
    b = _build(syn, days=1.0, dt=1800.0)
    st = _run(b, b.state, 0, 4)
    write_restart(tmp_path / "r.nc", b.model, st)
    b1 = _build(syn, days=1.0, dt=1800.0, theta_bc=1.0)
    assert b1.model.theta_bc == 1.0
    with pytest.raises(ValueError, match="theta_bc"):
        read_restart(tmp_path / "r.nc", b1.model)
    with pytest.raises(ValueError, match="allow_changed"):
        read_restart(tmp_path / "r.nc", b1.model, allow_changed=("no_such_entry",))
    with pytest.warns(UserWarning, match="theta_bc changed"):
        st1 = read_restart(tmp_path / "r.nc", b1.model, allow_changed=("theta_bc",))
    st1 = _run(b1, st1, 4, 6)
    assert np.isfinite(st1.u).all() and st1.n_trunc == 0
    case = dataclasses.replace(syn, runs=tmp_path / "runs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        rp.run(case, "full", 0.25, 1800.0, threads=2, restart_at=0.125, tag="base", out_hours=6.0, log=lambda m: None)
        meta = rp.run(case, "full", 0.25, 1800.0, threads=2, tag="tbc1", out_hours=6.0, log=lambda m: None,
                      start_from=rp.run_dir(case, "full", 1800.0, "base") / "restart_day0.125.nc",
                      overrides={"theta_bc": 1.0}, allow_restart_change=("theta_bc",))
    assert meta["overrides"] == {"theta_bc": "1.0"} and meta["allow_restart_change"] == ["theta_bc"]
    assert meta["n_steps"] == 6 and meta["n_trunc"] == 0 and meta["budget_scope"].startswith("from t0")


def test_deep_sponge(syn, monkeypatch, tmp_path):
    # Diagnostic interior sponge (intrusion diagnosis): rate 1/τ over H > h_min with a taper in cells from the
    # shallow side, 0 elsewhere; one profile = area-weighted mean of the GLORYS time mean over the selected cells;
    # the runner's --deep-sponge path builds it, relaxes T, S and skips the domain budget.
    hg, H = cs.grid(syn)
    kw, meta = cs.deep_sponge(syn, hg, H, tau_days=5.0, h_min=120.0, taper_cells=2,
                              period=("2017-04-30", "2017-05-03"), t_ref_depth=None)
    r, deep = kw["sponge_rate"], np.isfinite(H) & (np.nan_to_num(H) > 120.0)
    assert (r[~deep] == 0).all() and (r[deep] > 0).all() and r.max() == pytest.approx(1 / (5 * 86400.0))
    assert np.isclose(r[deep], 0.5 / (5 * 86400.0)).any()               # first deep cell next to the shelf: w = ½
    zw = kw["sponge_zw"]
    assert zw[0] == 0 and (np.diff(zw) < 0).all() and kw["sponge_T"].shape == kw["sponge_S"].shape == (zw.size - 1,)
    # synthetic S = 35 − 0.002 z + 0.1 (x − LON0): the profile decreases with depth by 0.002 per metre
    with xr.open_dataset(syn.glorys[0]) as ds:
        dep = ds["depth"].values.astype(float)
    assert np.allclose(np.diff(kw["sponge_S"]), -0.002 * np.diff(dep), atol=1e-4)
    assert meta["sponge_cells"] == deep.sum() and meta["profile_cells"] == deep.sum()
    b = _build(syn, days=0.25, dt=1800.0, **kw)
    assert b.model.sponge_rate is not None
    st = _run(b, b.state, 0, 2)
    assert np.isfinite(st.T).all() and np.isfinite(st.S).all()
    from validation.nwa12_cut import run_pyocean as rp
    orig = cs.deep_sponge
    monkeypatch.setattr(cs, "deep_sponge", lambda case, hg, H, tau_days: orig(
        case, hg, H, tau_days=tau_days, h_min=120.0, taper_cells=2, period=("2017-04-30", "2017-05-03"),
        t_ref_depth=None))
    import dataclasses
    case = dataclasses.replace(syn, runs=tmp_path / "runs")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        meta = rp.run(case, "full", 0.125, 1800.0, threads=2, tag="sp", out_hours=3.0, log=lambda m: None,
                      deep_sponge=5.0)
    assert meta["deep_sponge"]["tau_days"] == 5.0 and "budget" not in meta and meta["n_trunc"] == 0


def test_output_writes_layer_thickness(syn, tmp_path):
    # The new output field h_layer (ALE layer thickness; 'h' is the static bathymetry) round-trips through
    # OutputWriter with its CF standard name.
    from diagnostics.output import OutputWriter
    b = _build(syn, days=0.5, dt=1800.0)
    st = b.state
    w = OutputWriter(tmp_path / "o.nc", b.hg, b.H, n_layers=b.model.n, fmt="netcdf", model=b.model)
    w.write(0.0, eta=st.eta, T=st.T, h_layer=st.h)
    w.close()
    with xr.open_dataset(tmp_path / "o.nc") as d:
        np.testing.assert_array_equal(d["h_layer"].values[0], st.h)
        assert d["h_layer"].attrs["standard_name"] == "cell_thickness" and "h" in d     # static bathymetry kept


def test_domain_budget_refuses_uncovered_terms(syn):
    from diagnostics.domain_budget import DomainBudget
    b = _build(syn, days=0.5, dt=1800.0, kh=10.0)
    with pytest.raises(ValueError, match="kh > 0"):
        DomainBudget(b.model)


# ---------------------------------------------------------------------------------------------------- MOM6 (Task 10)
from validation.nwa12_cut import check_mom6 as cm  # noqa: E402
from validation.nwa12_cut import mom6_inputs as mi  # noqa: E402

_PGF0 = cm.RUNS / "mom6_pgf0"


def _has(run: Path, stem: str) -> bool:
    return run.exists() and any(not p.name.startswith("._") for p in run.glob(f"*{stem}*.nc"))


def test_cefi_functions_compile_from_cefi_source():
    """CEFI's pure-NumPy helpers are taken from tools/boundary/boundary.py itself (no xesmf needed)."""
    if not (mi.CEFI / "tools" / "boundary" / "boundary.py").exists():
        pytest.skip("CEFI-regional-MOM6 not on disk")
    f = mi._cefi()
    u = xr.DataArray(np.array([[1.0, 0.0]]), dims=("time", "locations"))
    v = xr.DataArray(np.array([[0.0, 1.0]]), dims=("time", "locations"))
    ur, vr = f["rotate_uv"](u, v, xr.DataArray(np.array([np.pi / 2, 0.0]), dims=("locations",)))
    np.testing.assert_allclose(ur.transpose("time", "locations").values, [[0.0, 0.0]], atol=1e-15)
    np.testing.assert_allclose(vr.transpose("time", "locations").values, [[-1.0, 1.0]], atol=1e-15)
    # ellipse round trip without rotation returns the amplitudes and phases (Xu 2000)
    uc = np.array([[0.3 * np.exp(-0.4j), 0.1 * np.exp(1.1j)]])
    vc = np.array([[0.2 * np.exp(0.9j), 0.05 * np.exp(-2.0j)]])
    ua, va, up, vp = f["ep2ap"](*f["ap2ep"](uc, vc))
    np.testing.assert_allclose(ua * np.exp(-1j * up), uc, atol=1e-14)
    np.testing.assert_allclose(va * np.exp(-1j * vp), vc, atol=1e-14)


def test_nearest_fill_and_flood():
    """nearest_s2d / setmisstonn / flood stand-ins for xesmf, cdo and HCtFlood."""
    lon, lat = np.meshgrid(np.arange(5.0), np.arange(4.0))
    k = mi.nearest_s2d(lon, lat, np.array([0.1, 3.9]), np.array([0.2, 2.8]))
    assert list(k) == [0, 3 * 5 + 4]
    f = np.arange(20.0).reshape(4, 5)
    f[1:3, 1:3] = np.nan
    g = mi.setmisstonn(f[None], lon, lat)[0]
    assert np.isfinite(g).all() and g[1, 1] in (f[0, 1], f[1, 0])
    fl = mi.flood(f)
    assert np.isfinite(fl).all()
    np.testing.assert_array_equal(fl[np.isfinite(f)], f[np.isfinite(f)])


@pytest.mark.skipif(not _has(_PGF0, "ocean_static"), reason="MOM6 ocean_static.nc not on disk (run_mom6.sh pgf0)")
def test_static_metrics_match_mom6():
    """Criterion 1: MOM6's static grid on the cut = grid_from_supergrid to ≤ 1e-12 relative (f_t excluded, R9)."""
    r = cm.static_check(_PGF0)
    bad = {k: e for k, (e, n) in r.items() if e > cm.TOL_GRID or n == 0}
    assert not bad, bad


@pytest.mark.skipif(not (_PGF0 / "MOM_IC.nc").exists(), reason="MOM6 MOM_IC.nc not on disk (run_mom6.sh pgf0)")
def test_mom6_reads_the_shared_initial_state():
    """Review M3: MOM6's initial state (MOM_IC.nc) is ic_full.nc bit for bit, velocities included."""
    r = cm.ic_check(_PGF0)
    assert max(r.values()) == 0.0, r


@pytest.mark.skipif(not _has(_PGF0, "pgf0"), reason="MOM6 pgf0 output not on disk (run_mom6.sh pgf0)")
def test_initial_pgf_matches_mom6():
    """Criterion 7: MOM6's initial PFu, PFv = pyOCEAN's on the same file (ic_full.nc) to ≤ 1e-11 m s⁻², with the
    NWA12 PGF settings (G_EARTH 9.8, full mass weighting)."""
    r = cm.pgf_check(_PGF0)
    assert max(r["max_du"], r["max_dv"]) <= cm.TOL_PGF, r


# ---------------------------------------------------------------------------------------------------- Task 11 compare
def test_tidal_constants_recovers_m2_k1():
    """Plan Task 11 step 1: the harmonic helper returns amplitude to 1e-6 m and Greenwich phase to 1e-4° on a
    synthetic M2 + K1 series built with the OTPS basis of forcing.tides (nodal corrections at each time)."""
    from diagnostics.tide_gauges import complex_from_amp_phase
    from forcing.tides import predict
    from validation.nwa12_cut import compare as cmp

    t = T0 + np.arange(0, 30 * 24 * 6) * np.timedelta64(10, "m")
    amp, pha = np.array([0.734, 0.112]), np.array([351.7, 187.25])
    h = 0.4 + predict(t, complex_from_amp_phase(amp, pha), ["m2", "k1"])
    r = cmp.tidal_constants(t, h, ["m2", "k1"])
    np.testing.assert_allclose(r["amp"], amp, atol=1e-6)
    dphi = (r["phase"] - pha + 180.0) % 360.0 - 180.0
    assert np.abs(dphi).max() < 1e-4
    assert abs(r["mean"] - 0.4) < 1e-6
    # inferred constituent: P1 = ratio · K1, solved jointly
    ratio = 0.33 * np.exp(-1j * np.deg2rad(4.0))
    hc = complex_from_amp_phase(np.array([0.734, 0.112, 0.112 * 0.33]), np.array([351.7, 187.25, 191.25]))
    h2 = predict(t, hc, ["m2", "k1", "p1"])
    r2 = cmp.tidal_constants(t, h2, ["m2", "k1", "p1"], inference={"p1": ("k1", ratio)})
    np.testing.assert_allclose(r2["amp"], np.abs(hc), atol=1e-6)


def test_tidal_errors_vector_and_phase():
    from validation.nwa12_cut import compare as cmp

    obs = np.array([1.0 * np.exp(-1j * np.deg2rad(10.0))])
    mod = np.array([0.9 * np.exp(-1j * np.deg2rad(20.0))])
    e = cmp.tidal_errors(mod, obs)
    assert abs(e["amp_ratio"][0] - 0.9) < 1e-12
    assert abs(e["dphase"][0] - 10.0) < 1e-9                        # model lags the observation by 10°
    assert abs(e["vector"][0] - abs(mod[0] - obs[0])) < 1e-15


def test_accepted_mask_excludes_qc_intervals():
    from validation.nwa12_cut import compare as cmp

    t = np.datetime64("2017-05-01T00:00", "ns") + np.arange(48) * np.timedelta64(1, "h")
    excl = {"gaps": [["2017-05-01T02:12", "2017-05-01T04:54"]], "inferred": [["2017-05-01T10:00", "2017-05-01T10:00"]],
            "spikes": [["2017-05-01T20:06", "2017-05-01T20:06"]],
            "step_candidates_rejected": [["2017-05-02T06:00", "2017-05-02T06:00"]]}     # not a fault: ignored
    ok = cmp.accepted(t, excl)
    bad = np.flatnonzero(~ok)
    assert list(bad) == [3, 4, 10]          # 03:00 and 04:00 inside the gap, 10:00 inferred; 20:06 is off the hour


def test_godin_filter_removes_tides_keeps_slow_signal():
    from validation.nwa12_cut import compare as cmp

    th = np.arange(24 * 40, dtype=float)
    slow = 0.2 * np.sin(2 * np.pi * th / (24 * 10.0))
    x = slow + 0.8 * np.cos(2 * np.pi * th / 12.4206) + 0.1 * np.cos(2 * np.pi * th / 23.9345)
    y = cmp.godin(x)
    ok = np.isfinite(y)
    assert ok.sum() == x.size - 2 * 35                                 # 71-h support
    def gain(period_h):                                                 # Godin's amplitude response
        a = lambda n: np.sin(np.pi * n / period_h) / (n * np.sin(np.pi / period_h))
        return a(24) ** 2 * a(25)
    assert abs(gain(12.4206)) < 4e-3 and abs(gain(23.9345)) < 2e-3
    assert np.abs(y[ok] - gain(240.0) * slow[ok]).max() < 0.8 * 4e-3 + 0.1 * 2e-3
    x[100] = np.nan
    y2 = cmp.godin(x)
    assert np.isnan(y2[100 - 35:100 + 36]).all() and np.isfinite(y2[200])


def test_daily_mean_and_skill():
    from validation.nwa12_cut import compare as cmp

    t = np.datetime64("2017-05-01T00:00", "ns") + np.arange(72) * np.timedelta64(1, "h")
    x = np.repeat([1.0, 2.0, 3.0], 24)
    x[30:50] = np.nan                                                   # day 2 keeps 4 of 24 h
    d, m = cmp.daily_mean(t, x, min_count=12)
    assert list(d.astype("datetime64[D]").astype(str)) == ["2017-05-01", "2017-05-02", "2017-05-03"]
    assert m[0] == 1.0 and np.isnan(m[1]) and m[2] == 3.0
    s = cmp.skill(np.array([1.0, 2.0, 3.0, np.nan]), np.array([1.5, 2.5, 3.5, 9.0]))
    assert s["n"] == 3 and abs(s["bias"] - 0.5) < 1e-12 and abs(s["rmse"] - 0.5) < 1e-12
    assert abs(s["crmse"]) < 1e-12 and abs(s["corr"] - 1.0) < 1e-12


def test_inverse_barometer():
    from validation.nwa12_cut import compare as cmp

    p = np.array([101325.0, 101425.0, 101225.0])
    eta = cmp.inverse_barometer(p, rho0=1025.0, g=9.81)
    np.testing.assert_allclose(eta, -(p - p.mean()) / (1025.0 * 9.81))


def test_undo_step_average_recovers_constants():
    """MOM6's hourly SSH is the mean of the 6 step-end values of the coupling hour (MOM.F90:1056, 1100); undoing the
    box average (phase shift ω·(n−1)Δ/2, gain sin(nωΔ/2)/(n sin(ωΔ/2))) recovers the instantaneous constants."""
    from diagnostics.tide_gauges import complex_from_amp_phase
    from forcing.tides import predict
    from validation.nwa12_cut import compare as cmp

    cons = ["m2", "s2", "n2", "k1", "o1"]
    hc = complex_from_amp_phase(np.array([0.6, 0.12, 0.14, 0.09, 0.06]), np.array([10.0, 40.0, 350.0, 180.0, 200.0]))
    t = T0 + np.arange(1, 29 * 24 + 1) * np.timedelta64(1, "h")
    dt = np.timedelta64(600, "s")
    y = np.mean([predict(t - k * dt, hc, cons) for k in range(6)], axis=0)
    raw = cmp.tidal_constants(t, y, cons)["hc"]
    assert np.abs(raw - hc).max() > 0.05                                      # 25-min lag: 12° in M2
    fixed = cmp.undo_step_average(raw, cons, n=6, dt=600.0)
    assert np.abs(fixed - hc).max() < 2e-4                                    # nodal u, f drift within the hour


def test_box_mean_matches_step_average():
    from validation.nwa12_cut import compare as cmp

    t6 = T0 + np.arange(0, 48 * 10) * np.timedelta64(6, "m")
    x = np.sin(2 * np.pi * ((t6 - T0) / np.timedelta64(1, "s")) / 44714.0)
    t = T0 + np.arange(2, 46) * np.timedelta64(1, "h")
    y = cmp.box_mean(t6, x, t, n=6, dt=600.0)
    sec = (t - T0) / np.timedelta64(1, "s")
    ref = np.mean([np.sin(2 * np.pi * (sec - k * 600.0) / 44714.0) for k in range(6)], axis=0)
    assert np.abs(y - ref).max() < 2e-4                                       # linear interpolation of 6-min data
    x[100] = np.nan
    assert np.isnan(cmp.box_mean(t6, x, t, n=6, dt=600.0)).sum() >= 1


# ---------------------------------------------------------------------------------------------------- Task 11 OISST
def test_daily_mean_of_fields():
    """``daily_mean`` also averages (nt, ny, nx) fields day by day, with the count rule per cell."""
    from validation.nwa12_cut import compare as cmp

    t = T0 + np.arange(48) * np.timedelta64(1, "h")
    f = np.zeros((48, 2, 3))
    f[:24] = 1.0
    f[24:] = np.arange(24)[:, None, None] * np.ones((1, 2, 3))
    f[24:40, 1, 2] = np.nan                                            # 8 of 24 h left in one cell of day 2
    d, m = cmp.daily_mean(t, f, min_count=12)
    assert m.shape == (2, 2, 3)
    assert np.all(m[0] == 1.0)
    assert abs(m[1, 0, 0] - 11.5) < 1e-12 and np.isnan(m[1, 1, 2])


def test_oisst_files_skip_appledouble(tmp_path):
    from validation.nwa12_cut import compare as cmp

    for n in ("oisst-avhrr-v02r01.20170502.nc", "oisst-avhrr-v02r01.20170501.nc",
              "._oisst-avhrr-v02r01.20170501.nc", "notes.txt"):
        (tmp_path / n).write_bytes(b"")
    assert [p.name for p in cmp.oisst_files(tmp_path)] == ["oisst-avhrr-v02r01.20170501.nc",
                                                           "oisst-avhrr-v02r01.20170502.nc"]


def test_coastal_pixels_eight_neighbours():
    from validation.nwa12_cut import compare as cmp

    land = np.zeros((5, 5), bool)
    land[0, 0] = True
    c = cmp.coastal_pixels(land)
    assert c[1, 1] and c[0, 1] and c[1, 0]                             # diagonal neighbour counts
    assert not c[0, 0]                                                  # land itself is not a coastal water pixel
    assert c.sum() == 3


def test_oisst_qc_flags():
    """Day coverage, time stamps, land-mask stability, range, ice, err statistics and the coastal-representativeness
    flag: a coastal pixel more than ``dep_max`` from the reference on average is flagged, an open-ocean pixel with the
    same departure is not (mesoscale position is not an analysis fault), a coastal pixel without reference is
    'unverified'."""
    from validation.nwa12_cut import compare as cmp

    days = np.datetime64("2017-05-01T12:00", "ns") + np.array([0, 1, 3, 4]) * np.timedelta64(1, "D")   # 05-03 missing
    ny, nx = 4, 5
    sst = np.full((4, ny, nx), 15.0)
    sst[:, 0, 0] = np.nan                                               # land
    sst[:, 3, 4] = np.nan                                               # land
    err = np.full_like(sst, 0.2)
    err[:, 1, 1] = 0.5
    ice = np.full_like(sst, np.nan)
    anom = np.zeros_like(sst)
    ref = np.full_like(sst, 15.0)
    ref[:, 0, 0] = np.nan
    ref[:, 3, 4] = np.nan
    ref[:, 0, 1] = 11.0                                                 # coastal, OISST 4 °C warmer
    ref[:, 2, 2] = 11.0                                                 # open ocean, same departure
    ref[:, 2, 3] = np.nan                                               # coastal (next to (3, 4)), no reference
    q = cmp.oisst_qc(days, sst, err, ice, anom, ref=ref, t_start=np.datetime64("2017-05-01"),
                     t_end=np.datetime64("2017-05-05"), dep_max=3.0)
    assert q["days_missing"] == ["2017-05-03"]
    assert q["n_days"] == 4 and q["n_off_noon"] == 0 and q["land_constant"]
    assert q["n_out_of_range"] == 0 and q["n_ice"] == 0
    assert abs(q["err_max"] - 0.5) < 1e-6
    assert q["pixel"][0, 1] == "flagged" and q["pixel"][2, 2] == "ok" and q["pixel"][2, 3] == "unverified"
    assert q["pixel"][0, 0] == "land"
    assert q["n_flagged"] == 1 and q["n_unverified"] == 1
    # range and ice faults are counted
    sst[2, 1, 2] = 40.0
    ice[1, 1, 3] = 10.0
    q = cmp.oisst_qc(days, sst, err, ice, anom, ref=ref, t_start=np.datetime64("2017-05-01"),
                     t_end=np.datetime64("2017-05-05"))
    assert q["n_out_of_range"] == 1 and q["n_ice"] == 1


def test_region_skill_area_weighted():
    """Area-weighted daily bias and RMSE of f − ref over region masks, per period (RMSE of a period = root of the
    mean of the daily mean squares)."""
    from validation.nwa12_cut import compare as cmp

    t = T0 + np.arange(3) * np.timedelta64(1, "D")
    ref = np.zeros((3, 2, 2))
    f = np.zeros((3, 2, 2))
    f[:, 0, 0] = 1.0
    f[:, 0, 1] = -1.0
    f[2] = 2.0
    area = np.array([[3.0, 1.0], [1.0, 1.0]])
    reg = {"top": np.array([[True, True], [False, False]]), "all": np.ones((2, 2), bool)}
    per = {"first2": np.array([True, True, False]), "all": np.ones(3, bool)}
    rows, bias_t = cmp.region_skill(t, f, ref, reg, area, per, source="X")
    r = {(x["region"], x["period"]): x for x in rows}
    assert abs(r["top", "first2"]["bias"] - 0.5) < 1e-12               # (3·1 − 1·1)/4
    assert abs(r["top", "first2"]["rmse"] - 1.0) < 1e-12
    assert abs(r["all", "all"]["bias"] - (2 * (2 / 6) + 2) / 3) < 1e-12
    assert r["top", "all"]["cells"] == 2 and r["top", "all"]["n_days"] == 3 and r["top", "all"]["source"] == "X"
    np.testing.assert_allclose(bias_t["top"][1], [0.5, 0.5, 2.0])


def test_tide_check_undoes_mom6_step_average(tmp_path):
    """tide_check reads a MOM6 run's hourly SSH as the coupling-hour mean of its 6 step-end values (MOM6 0e9a561
    MOM.F90:1050-1100) and undoes it, as compare.model_eta does; a pyOCEAN run (no ocean_hourly file) is untouched."""
    import json
    from types import SimpleNamespace

    from diagnostics.tide_gauges import complex_from_amp_phase
    from forcing.tides import predict
    from validation.nwa12_cut import compare as cmp
    from validation.nwa12_cut import tide_check as tc

    cons = ["m2", "s2", "n2", "k1", "o1", "q1"]
    hc = complex_from_amp_phase(np.array([0.6, 0.12, 0.14, 0.09, 0.06, 0.01]),
                                np.array([10.0, 40.0, 350.0, 180.0, 200.0, 190.0]))
    sec = np.arange(1, 30 * 24 + 1) * 3600.0
    t = T0 + (sec * 1e9).astype("timedelta64[ns]")
    y = np.mean([predict(t - k * np.timedelta64(600, "s"), hc, cons) for k in range(6)], axis=0)
    (tmp_path / "harcon_9999999.json").write_text(json.dumps({"HarmonicConstituents": [
        {"name": "M2", "amplitude": 0.6, "phase_GMT": 10.0}]}))
    case = SimpleNamespace(t0=T0, tides=tmp_path)
    for kind in ("mom6", "pyocean"):
        run = tmp_path / kind
        run.mkdir()
        np.savez(run / "series.npz", t=sec, names=np.array(["coops_9999999"]), eta=y[:, None])
        if kind == "mom6":
            (run / "20170501.ocean_hourly.nc").write_bytes(b"")
            (run / "input.nml").write_text("&coupler_nml\n    dt_cpld = 3600\n/\n")
            (run / "MOM_parameter_doc.short").write_text("DT = 600.0  ! [s]\n")
        assert cmp.step_average_of(run) == ((6, 600.0) if kind == "mom6" else (1, 3600.0))
        o, m = tc.gauge_m2(run, skip_days=1.0, case=case)["stations"]["9999999"]
        if kind == "mom6":
            assert abs(m - o) < 2e-4                                  # corrected: instantaneous M2 recovered
        else:
            assert abs(m - o) > 0.1                                   # a box-averaged series read as snapshots


# ---------------------------------------------------------------------------------------------------- Task 11 in situ
def test_column_to_bins_layers_and_bed():
    """Layer values (surface first) on depth bins: linear between centres, top value above the first centre, the
    bottom value down to the bed, NaN below it; vanished bed layers are skipped."""
    from validation.nwa12_cut import compare as cmp

    h = np.array([2.0, 2.0, 4.0, 1e-3, 1e-3])                 # centres 1, 3, 6; bed at 8 m; two vanished layers
    x = np.array([10.0, 8.0, 2.0, 99.0, 99.0])
    np.testing.assert_allclose(cmp.column_centres(h[:3]), [1.0, 3.0, 6.0])
    zb = np.array([0.5, 2.0, 4.5, 7.5, 8.5])
    np.testing.assert_allclose(cmp.column_to_bins(h, x, zb), [10.0, 9.0, 5.0, 2.0, np.nan])


def test_bracket_linear_time_weights():
    from validation.nwa12_cut import compare as cmp

    times = T0 + np.array([0, 24, 48]) * np.timedelta64(1, "h")
    t = T0 + np.array([6, 24, 47, 60, -1]) * np.timedelta64(1, "h")
    k0, k1, w = cmp.bracket(times, t)
    np.testing.assert_array_equal(k0, [0, 0, 1, -1, -1])
    np.testing.assert_array_equal(k1, [1, 1, 2, -1, -1])
    np.testing.assert_allclose(w[:3], [0.25, 1.0, 23 / 24])


class _FakeReader:
    """Two daily levels on a 2 × 3 grid with 3 layers; T = 10 + day + layer, u, v uniform per level."""

    def __init__(self):
        self.times = T0 + np.array([0, 24]) * np.timedelta64(1, "h")

    def _f(self, k, uv=False):
        h = np.ones((3, 2, 3))
        T = 10.0 + k + np.arange(3)[:, None, None] * np.ones((3, 2, 3))
        f = dict(h=h, T=T, S=35.0 + 0 * T)
        if uv:
            f["u"] = np.full((3, 2, 4), 1.0 + k)
            f["v"] = np.full((3, 3, 3), 0.0)
        return f

    def get(self, k):
        return self._f(k)

    def load(self, k, uv=False):
        return self._f(k, uv)


def test_sample_columns_time_interpolation_and_rotation():
    from validation.nwa12_cut import compare as cmp

    rd = _FakeReader()
    t = T0 + np.array([12, 24]) * np.timedelta64(1, "h")
    zb = np.array([0.5, 1.5, 2.5])
    th, S = cmp.sample_columns(rd, t, np.array([0, 1]), np.array([2, 0]), zb)
    np.testing.assert_allclose(th, [[10.5, 11.5, 12.5], [11.0, 12.0, 13.0]])
    np.testing.assert_allclose(S, 35.0)
    # grid x rotated 90° counter-clockwise from east: u_x = 1 is a northward current
    rot = (np.zeros((2, 3)), np.ones((2, 3)))
    ue, vn = cmp.sample_columns(rd, t[:1], np.array([0]), np.array([1]), zb, kind="uv", rot=rot)
    np.testing.assert_allclose(ue, 0.0, atol=1e-12)
    np.testing.assert_allclose(vn, 1.5)


def test_at_depth_clamps_to_model_bottom():
    from validation.nwa12_cut import compare as cmp

    zb = np.array([0.5, 1.5, 2.5, 3.5])
    x = np.array([[1.0, 2.0, 3.0, np.nan], [1.0, 2.0, 3.0, 4.0]])
    np.testing.assert_allclose(cmp.at_depth(x, zb, np.array([10.0, 2.0])), [3.0, 2.5])
    assert np.isnan(cmp.at_depth(x, zb, np.array([10.0, 10.0]), clamp_bottom=False)).all()


def test_band_stats_common_samples():
    """Every source is scored on the same samples: a bin missing in one source is dropped for all."""
    from validation.nwa12_cut import compare as cmp

    zb = np.array([5.0, 15.0, 45.0])
    obs = np.array([[1.0, 2.0, 3.0], [1.0, 2.0, 3.0]])
    a = obs + 1.0
    b = obs - 0.5
    b[1, 1] = np.nan
    rows = cmp.band_stats(obs, {"a": a, "b": b}, zb, {"g": np.array([True, True])}, "theta",
                          bands=((0.0, 10.0), (10.0, 30.0)))
    r = {(x["band"], x["source"]): x for x in rows}
    assert r[("0-10", "a")]["n"] == 2 and r[("0-10", "a")]["bias"] == pytest.approx(1.0)
    assert r[("10-30", "b")]["n"] == 1 and r[("10-30", "a")]["n"] == 1
    assert r[("10-30", "b")]["rmse"] == pytest.approx(0.5)


def test_layer_mean_and_obs_at():
    from validation.nwa12_cut import compare as cmp

    zb = np.arange(0.5, 10.0, 1.0)
    x = np.tile(np.arange(10.0), (2, 1))
    x[1, :4] = np.nan                                   # 1 of 5 bins in 0–5 m: < half filled
    m = cmp.layer_mean(x, zb, 0.0, 5.0)
    assert m[0] == pytest.approx(2.0) and np.isnan(m[1])
    t = T0 + np.arange(0, 48 * 4) * np.timedelta64(15, "m")
    y = np.arange(t.size, dtype=float)
    y[:48] = np.nan                                     # half of day 1 missing
    q = T0 + np.array([12, 36]) * np.timedelta64(1, "h")
    d = cmp._obs_at(t, y, q, daily=True, min_frac=0.8)
    assert np.isnan(d[0]) and d[1] == pytest.approx(np.mean(y[96:192]))
    s = cmp._obs_at(t, y, q, daily=False, min_frac=0.0)
    assert s[1] == pytest.approx(np.mean(y[142:147]))   # ±30 min around 36 h: samples 142…146


def test_nearest_wet_skips_land():
    from validation.nwa12_cut import compare as cmp

    lat2, lon2 = np.meshgrid(np.array([40.0, 40.1]), np.array([-70.0, -69.9, -69.8]), indexing="ij")
    wet = np.ones((2, 3), bool)
    wet[0, 0] = False
    j, i, d = cmp.nearest_wet(lat2, lon2, wet, np.array([40.0]), np.array([-70.0]))
    assert (j[0], i[0]) in ((1, 0), (0, 1))
    assert d[0] == pytest.approx(8.5, abs=3.0)


def test_mld_test_summary():
    from validation.nwa12_cut import compare as cmp

    n = 20
    g = np.ones(n, bool)
    met = {"obs": {k: np.full(n, 10.0) for k in ("mld", "tcl", "t05", "t030", "t3060", "s030")}}
    met["A"] = {k: v.copy() for k, v in met["obs"].items()}
    met["B"] = {k: v.copy() for k, v in met["obs"].items()}
    met["A"]["mld"] = np.linspace(4, 8, n)              # shallower and warmer where shallower
    met["A"]["t05"] = 10.0 + (8 - met["A"]["mld"])
    met["A"]["t030"] = 10.0 + np.linspace(0, 2, n)      # column heat excess carried by salt: advection
    met["A"]["s030"] = 10.0 + np.linspace(0, 1, n)
    r = cmp.mld_test(dict(met=met, groups={"g": g}), a="A", b="B", group="g")
    assert r["corr_dT030_dS030"] == pytest.approx(1.0)
    assert r["n"] == n and r["frac_a_shallower"] == 1.0
    assert r["corr_dT05_dMLD"] == pytest.approx(-1.0)
    assert r["A_mld_median"] == pytest.approx(6.0)


def test_bottom_values_skip_vanished_layers():
    from validation.nwa12_cut import compare as cmp

    h = np.zeros((4, 1, 3))
    h[:, 0, 0] = [2.0, 2.0, 1e-3, 1e-3]             # z* column with two vanished bed layers
    h[:, 0, 1] = [2.0, 2.0, 2.0, 2.0]
    x = np.arange(12.0).reshape(4, 1, 3)
    b = cmp.bottom_values(h, x)
    assert b[0, 0] == x[1, 0, 0] and b[0, 1] == x[3, 0, 1] and np.isnan(b[0, 2])
