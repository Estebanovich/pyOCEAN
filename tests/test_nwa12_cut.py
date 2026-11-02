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
    assert m.ale and m.regrid_rule == "zstar" and m.momentum == "vector_invariant" and m.smagorinsky == 0.15
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
    for bad in (dict(vertical_coordinate="sigma_eulerian"), dict(momentum="flux"), dict(smagorinsky=0.0)):
        with pytest.raises(ValueError, match="curvilinear metric terms"):
            cs.parity_config(syn, b.hg, b.H, 600.0, edges=b.edges, **bad)


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
