"""
River runoff (regional chain Task 8; brief .superpowers/sdd/2026-11-01-regional-chain/task-8-brief.md):
GloFAS discharge → coastal model cells (``forcing.sources.glofas.glofas_mouths``, CEFI tools/rivers rule) and
Q/A_t as a real freshwater volume source (``core.components.runoff``, N2a path).

* mapping: a mouth on model land or on the grid edge goes to the nearest wet, interior, H > 0 coast cell and is
  reported; total discharge conserved exactly; CEFI's ldd rule; daily-mean midpoints; coverage raises;
* model: volume gain = ∫ Q dt and salt content unchanged (σ and z*); with bulk P − E both act; off is bit for bit;
  restart between records bit for bit and the river set checked; pre-flight.

Synthetic GloFAS: 0.05° cells, sea (NaN) east of 73.6 °W, a river flowing east along 40.425 °N to a pit next to
the sea, a second coastal mouth in the southern edge row of the model, an inland pit and small random land
discharge (< 1 m³ s⁻¹, below the threshold).  Model: 12 × 8 cells of 0.1°, land west of 73.6 °W.
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import glob

import numpy as np
import pytest
import xarray as xr

from core.components.surface_fluxes import BulkFluxes
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_spherical_grid
from core.model.contract import ConfigurationError
from core.restart import read_restart, write_restart
from forcing.sources.glofas import RiverInput, cefi_pour_points, coast_mask, glofas_mouths

NX, NY, NZ = 12, 8, 2
LON0, LAT0, DL = -74.0, 40.0, 0.1
DAY0 = np.datetime64("2017-06-01T00:00", "ns")
RHO0 = 1000.0
SAL = dict(salinity=True, s_coef=0.78, s_ref=35.0)


def _grid():
    return create_spherical_grid(nx=NX, ny=NY, dlon=DL, dlat=DL, lon0=LON0, lat0=LAT0)


def _bathy() -> np.ndarray:
    H = np.full((NY, NX), 20.0)
    H[:, :4] = np.nan                                                  # land west of 73.6 °W
    return H


def _glofas(n_days: int = 6, seed: int = 0) -> xr.Dataset:
    """Synthetic GloFAS v4 file: ``avg_dis`` (valid_time, latitude, longitude), latitude north → south, lon 0–360."""
    rng = np.random.default_rng(seed)
    lat = np.round(np.arange(40.975, 39.80, -0.05), 3)
    lon = np.round(np.arange(-74.475, -72.70, 0.05), 3)
    LON, LAT = np.meshgrid(lon, lat)
    base = rng.uniform(0.0, 0.5, LON.shape)                           # land drainage below 1 m³/s
    base[LON > -73.6] = np.nan                                         # sea
    q = np.repeat(base[None], n_days, axis=0)
    days = np.arange(n_days)
    jr = int(np.argmin(np.abs(lat - 40.425)))
    land_cols = np.flatnonzero(lon < -73.6)
    for c, ic in enumerate(land_cols[2:]):                            # the river, increasing downstream
        q[:, jr, ic] = 10.0 + 10.0 * c + 5.0 * days
    js = int(np.argmin(np.abs(lat - 40.025)))                          # mouth in the model's southern edge row
    q[:, js, land_cols[-1]] = 20.0 + 2.0 * days
    ji = int(np.argmin(np.abs(lat - 40.725)))                          # inland pit, 18 cells from the sea
    q[:, ji, 2] = 50.0
    vt = DAY0 + np.timedelta64(1, "D") + days.astype("timedelta64[D]")  # valid_time = end of the daily mean
    return xr.Dataset({"avg_dis": (("valid_time", "latitude", "longitude"), q.astype(np.float32))},
                      coords={"valid_time": vt.astype("datetime64[ns]"), "latitude": lat, "longitude": lon + 360.0})


def _river(**kw) -> RiverInput:
    H = _bathy()
    return glofas_mouths(_glofas(), _grid(), DAY0 + np.timedelta64(12, "h"), DAY0 + np.timedelta64(5 * 24 + 12, "h"),
                         wet=np.isfinite(H) & (np.nan_to_num(H) > 0), **kw)


# ---------------------------------------------------------------------------------------------------- mapping

def test_mouth_on_land_moved_to_nearest_wet_interior():
    """Review focus 3: mouths on land or on the edge row go to the nearest wet, interior, H > 0 coast cell."""
    r = _river()
    H = _bathy()
    wet = np.isfinite(H) & (np.nan_to_num(H) > 0)
    coast = coast_mask(wet)
    for j, i in r.cells:
        assert 0 < j < NY - 1 and 0 < i < NX - 1
        assert wet[j, i] and H[j, i] > 0 and coast[j, i]
    # the two mouths kept: the river (in model land cell (4, 3)) and the edge-row mouth (in (0, 3), land)
    assert r.mouth_lat.size == 2
    k_river = int(np.argmin(np.abs(r.mouth_lat - 40.425)))
    k_edge = 1 - k_river
    river_cells = {tuple(r.cells[c]) for c in r.mouth_cells[k_river]}
    edge_cells = {tuple(r.cells[c]) for c in r.mouth_cells[k_edge]}
    assert river_cells == {(4, 4)}                                     # same row, first wet column
    assert edge_cells == {(1, 4)}                                      # moved off the edge row
    text = "\n".join(r.report)
    assert "[land]" in text and "inland pit not used" in text and "40.725N" in text
    assert "dropped" in text and "< 1 m3/s" in text


def test_total_discharge_conserved():
    """Σ over model cells = Σ over the GloFAS mouths, exactly (to round-off), at every record and in between."""
    r = _river()
    ds = _glofas()
    q = ds.avg_dis.values.astype(float)
    lat, lon = ds.latitude.values, ds.longitude.values - 360.0
    expect = np.zeros(q.shape[0])
    for la, lo in zip(r.mouth_lat, r.mouth_lon):
        expect += q[:, int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))]
    np.testing.assert_allclose(r.Q.sum(axis=1), expect, rtol=1e-14, atol=0)
    np.testing.assert_allclose(r.mouth_Q.sum(axis=1), expect, rtol=1e-14, atol=0)
    # a pour point whose cell straddles two model cells is split and still conserved
    g = _grid()
    shifted = _glofas().assign_coords(latitude=_glofas().latitude - 0.025)   # centres on model cell faces
    r2 = glofas_mouths(shifted, g, DAY0 + np.timedelta64(12, "h"), DAY0 + np.timedelta64(36, "h"),
                       wet=np.isfinite(_bathy()))
    assert any(len(t) > 1 for t in r2.mouth_cells)
    np.testing.assert_allclose(r2.Q.sum(axis=1), r2.mouth_Q @ r2.mouth_inside, rtol=1e-14, atol=0)
    assert (r.mouth_inside == 1.0).all()
    t = 0.37 * 86400.0
    np.testing.assert_allclose(r.discharge(t).sum(), np.interp(t, r.seconds, expect), rtol=1e-14)


@pytest.mark.parametrize("n_sub", [4, 5])
def test_pour_point_straddling_the_edge_injects_its_inside_fraction(n_sub):
    """
    Review F1: a pour-point cell centred on the southern grid edge (40.0 °N) injects only its sub-cells inside the
    grid, as CEFI's conservative regrid on the full grid then cut.  The edge is parallel to the sub-cell rows, so
    the inside fraction moves by whole rows: exactly 1/2 with n_sub = 4 (2 of 4 rows), 2/5 with n_sub = 5 (the
    middle row is just outside: centre at 39.9999 °N), i.e. Q/2 to within 1/n_sub (1/n_sub² per sub-cell).
    """
    ds = _glofas().assign_coords(latitude=_glofas().latitude - 0.0251)
    r = glofas_mouths(ds, _grid(), DAY0 + np.timedelta64(12, "h"), DAY0 + np.timedelta64(36, "h"),
                      wet=np.isfinite(_bathy()), n_sub=n_sub)
    k = int(np.argmin(np.abs(r.mouth_lat - 39.9999)))
    assert abs(r.mouth_lat[k] - 39.9999) < 1e-9
    expect = 0.5 if n_sub == 4 else 0.4
    assert r.mouth_inside[k] == pytest.approx(expect, abs=1e-15)
    assert abs(r.mouth_inside[k] - 0.5) <= 1.0 / n_sub
    assert sum(r.mouth_cells[k].values()) == pytest.approx(expect, abs=1e-15)
    W = np.zeros(r.cells.shape[0])
    for c, f in r.mouth_cells[k].items():
        W[c] = f
    np.testing.assert_allclose(W.sum() * r.mouth_Q[:, k], expect * r.mouth_Q[:, k], rtol=1e-15)
    np.testing.assert_allclose(r.Q.sum(axis=1), r.mouth_Q @ r.mouth_inside, rtol=1e-14)
    assert any("straddle the grid edge" in line for line in r.report)


def test_step_beyond_the_runoff_records_raises():
    """Review F2: a model step whose tⁿ + Δt/2 is past the last GloFAS record raises, naming GloFAS and the dates."""
    import dataclasses
    r = _river()
    m = _model(runoff=r)
    st = dataclasses.replace(_start(m), time=float(r.seconds[-1]))
    with pytest.raises(ValueError, match=r"GloFAS runoff covers 2017-06-01T12:00.*2017-06-06T12:00.*outside"):
        m.step(st)


def test_times_are_daily_midpoints_and_coverage_raises():
    """valid_time is the end of the 24-h mean: the value sits at its day's midpoint; outside the records raises."""
    r = _river()
    assert r.times[0] == DAY0 + np.timedelta64(12, "h")
    assert r.seconds[1] - r.seconds[0] == 86400.0
    q0, q1 = r.Q[0], r.Q[1]
    np.testing.assert_allclose(r.discharge(0.25 * 86400.0), 0.75 * q0 + 0.25 * q1, rtol=1e-15)
    with pytest.raises(ValueError, match="GloFAS runoff covers"):
        r.discharge(-1.0)
    with pytest.raises(ValueError, match="GloFAS covers"):
        glofas_mouths(_glofas(), _grid(), DAY0, DAY0 + np.timedelta64(2, "D"), wet=np.isfinite(_bathy()))


def test_cefi_ldd_rule():
    """CEFI's pour points: ldd = 5 next to the sea, grown through adjacent outlets; not on the outermost rows."""
    ldd = np.full((7, 8), 2.0)
    ldd[:, 6:] = np.nan                                                # sea in the east
    ldd[3, 5] = 5.0                                                    # outlet at the coast
    ldd[3, 4] = 5.0                                                    # outlet chain inland (an estuary)
    ldd[3, 3] = 5.0
    ldd[1, 1] = 5.0                                                    # inland pit, not connected
    ldd[0, 5] = 5.0                                                    # coastal, but on the outermost row
    p = cefi_pour_points(ldd)
    assert set(zip(*np.nonzero(p))) == {(3, 5), (3, 4), (3, 3)}


# ---------------------------------------------------------------------------------------------------- model

class _Atm:
    """Atmosphere stub on the model T points: calm air near the SST, light rain."""

    def __init__(self, shape: tuple[int, int]) -> None:
        self.shape, self.origin = shape, DAY0

    def coverage(self) -> tuple[float, float]:
        return -86400.0, 30 * 86400.0

    def __call__(self, t: float) -> dict[str, np.ndarray]:
        one = np.ones(self.shape)
        return dict(u10=2.0 * one, v10=0.0 * one, t2m=293.15 * one, q2m=0.0140 * one, msl=101300.0 * one,
                    sw_down=0.0 * one, lw_down=430.0 * one, precip=1e-4 * one)


def _model(kind: str = "sigma", runoff=None, bulk: bool = False, **kw) -> SigmaBaroclinic:
    H = _bathy()
    base = dict(rho0=RHO0, t_coef=0.2, t_ref=20.0, closure="constant", nu_const=1e-3, ah=0.0, theta=1.0,
                d_min=0.05, **SAL)
    if kind == "zstar":
        base.update(vertical_coordinate="ale", regrid="zstar", pgf="fv",
                    regrid_params=dict(dz_top=np.full(NZ, 20.0 / NZ)))
    else:
        base.update(z0=1e-3)
    if runoff is not None:
        base["runoff"] = runoff
    if bulk:
        base["surface_fluxes"] = BulkFluxes(_Atm(H.shape))
    base.update(kw)
    return SigmaBaroclinic(_grid(), H, dt=600.0, n_layers=NZ, **base)


def _start(m: SigmaBaroclinic):
    return m.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), 20.0), S=np.full((NZ, NY, NX), 35.0))


def _volume(m, st) -> float:
    act = m.fs.active
    return float(np.sum((m.area * np.maximum(m.H + st.eta, 0.0))[act]))


def _salt(m, st) -> float:
    V = m.layer_volumes(st.eta, st.h)
    return float(np.nansum(V * st.S))


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
def test_volume_and_salt_budget_with_runoff(kind):
    """20 steps of 600 s: ΔV = Σ Δt Σ Q(tⁿ + Δt/2) to 1e-13 of V; Σ V S unchanged to 1e-13; fresher at the mouth."""
    r = _river()
    m = _model(kind, runoff=r)
    st = _start(m)
    V0, S0, q_vol = _volume(m, st), _salt(m, st), 0.0
    for _ in range(20):
        q_vol += m.dt * float(r.discharge(st.time + 0.5 * m.dt).sum())
        st = m.step(st)
    dV = _volume(m, st) - V0
    assert abs(dV - q_vol) <= 1e-13 * V0, (dV, q_vol)
    assert abs(dV / q_vol - 1.0) < 1e-6
    assert abs(_salt(m, st) - S0) <= 1e-13 * S0
    j, i = r.cells[np.argmax(r.Q.mean(axis=0))]
    assert st.S[-1, j, i] < 35.0 - 1e-6                               # fresh water in the top layer at the mouth
    np.testing.assert_allclose(m.runoff[j, i] * m.area[j, i],
                               r.discharge(st.time - 0.5 * m.dt)[np.argmax(r.Q.mean(axis=0))], rtol=1e-14)


def test_runoff_with_bulk():
    """With bulk fluxes the runoff is added to P − E (no refusal); the fw of the step is the sum."""
    r = _river()
    a, b = _model(runoff=r, bulk=True), _model(bulk=True)
    sa, sb = _start(a), _start(b)
    sa, sb = a.step(sa), b.step(sb)
    diff = a._fluxes.fw - b._fluxes.fw
    np.testing.assert_allclose(diff, a.runoff, rtol=1e-12, atol=1e-18)
    assert np.abs(b._fluxes.fw).max() > 0 and np.abs(a.runoff).max() > 0
    with pytest.raises(ConfigurationError, match="fw"):               # prescribed fw with bulk stays refused
        a.step(sa, fw=np.zeros((NY, NX)))


def test_runoff_adds_to_a_prescribed_fw():
    """Without bulk, step(fw=...) and the runoff add."""
    r = _river()
    m = _model(runoff=r)
    st = m.step(_start(m), fw=np.full((NY, NX), 1e-7))
    expect = np.full((NY, NX), 1e-7) + m.runoff
    np.testing.assert_allclose(m._fluxes.fw[np.isfinite(m.H)], expect[np.isfinite(m.H)], rtol=1e-15)


def test_off_bit_for_bit():
    """runoff=None (explicit) is the model without the argument, phase by phase."""
    a, b = _model(), _model(runoff=None)
    assert a._runoff is None and b._runoff is None
    a._driver.debug = b._driver.debug = True
    sa, sb = _start(a), _start(b)
    for _ in range(4):
        sa, sb = a.step(sa), b.step(sb)
    assert a._driver.phase_hashes == b._driver.phase_hashes
    assert a.runoff is None


def test_restart_between_records_bit_for_bit(tmp_path):
    """6 steps = 3 + restart + 3 with the restart 30 min after a record (between two); the river set is checked."""
    r = _river()
    m = _model("zstar", runoff=r)
    st = _start(m)
    for _ in range(3):
        st = m.step(st)
    assert 0.0 < st.time < r.seconds[1]
    path = tmp_path / "r.nc"
    write_restart(path, m, st)
    for _ in range(3):
        st = m.step(st)
    m2 = _model("zstar", runoff=r)
    st2 = read_restart(path, m2)
    for _ in range(3):
        st2 = m2.step(st2)
    for f in ("eta", "u", "v", "T", "S", "h"):
        np.testing.assert_array_equal(getattr(st2, f), getattr(st, f))
    with pytest.raises(ValueError, match="runoff"):
        read_restart(path, _model("zstar"))
    other = RiverInput(cells=r.cells[:-1], times=r.times, Q=r.Q[:, :-1], origin=r.origin)
    with pytest.raises(ValueError, match="runoff"):
        read_restart(path, _model("zstar", runoff=other))
    later = RiverInput(cells=r.cells, times=r.times, Q=r.Q, origin=r.origin + np.timedelta64(1, "h"))
    with pytest.raises(ValueError, match="runoff_origin"):
        read_restart(path, _model("zstar", runoff=later))


def test_preflight():
    """Salinity required, cells on land or outside refused, all at once."""
    r = _river()
    bad = RiverInput(cells=np.array([[2, 1], [2, 40]]), times=r.times, Q=np.ones((r.times.size, 2)), origin=r.origin)
    with pytest.raises(ConfigurationError) as e:
        _model(runoff=bad, salinity=False)
    msg = str(e.value)
    assert "runoff needs salinity=True" in msg and "outside the grid" in msg
    bad2 = RiverInput(cells=np.array([[2, 1]]), times=r.times, Q=np.ones((r.times.size, 1)), origin=r.origin)
    with pytest.raises(ConfigurationError, match="on land"):
        _model(runoff=bad2)
    with pytest.raises(ConfigurationError, match="RiverInput"):
        _model(runoff=object())


# ---------------------------------------------------------------------------------------------------- real data

_GLOFAS = sorted(glob.glob("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/data/glofas/glofas_v4_consolidated_2017-*.nc"))
_HGRID = "/Users/estebancruz/pyOCEAN_refs/nwa12_grid/ocean_hgrid.nc"


@pytest.mark.skipif(len(_GLOFAS) < 6 or not glob.glob(_HGRID), reason="GloFAS 2017 files or the NWA12 grid absent")
def test_real_glofas_on_the_nwa12_cut():
    """The cut box of Task 9 (75.5–68.5 °W, 36.5–41.8 °N): Hudson, Connecticut and Delaware are among the mouths."""
    from core.grid.supergrid import grid_from_supergrid
    hg = xr.open_dataset(_HGRID)
    depth = xr.open_dataset(_HGRID.replace("ocean_hgrid", "ocean_topog")).depth.values
    x, y = hg.x.values[1::2, 1::2], hg.y.values[1::2, 1::2]
    jj, ii = np.nonzero((x > -75.5) & (x < -68.5) & (y > 36.5) & (y < 41.8))
    g = grid_from_supergrid(hg, depth, window=(ii.min(), ii.max() + 1, jj.min(), jj.max() + 1))
    r = glofas_mouths(_GLOFAS, g, "2017-05-01", "2017-09-01")
    np.testing.assert_allclose(r.Q.sum(axis=1), r.mouth_Q @ r.mouth_inside, rtol=1e-13)
    qm = r.mouth_Q.mean(axis=0)
    big = {(round(float(la), 3), round(float(lo), 3)) for la, lo, q in zip(r.mouth_lat, r.mouth_lon, qm) if q > 300}
    assert {(40.875, -73.925), (41.275, -72.325), (39.875, -75.225)} <= big   # Hudson, Connecticut, Delaware
    wet = np.asarray(g.mask_t.values, bool)
    assert all(wet[j, i] for j, i in r.cells)


_LDD = "/Users/estebancruz/pyOCEAN_refs/glofas_aux/ldd_glofas_v4_0.nc"


@pytest.mark.skipif(len(_GLOFAS) < 6 or not glob.glob(_HGRID) or not glob.glob(_LDD),
                    reason="GloFAS 2017 files, the NWA12 grid or the GloFAS v4 ldd absent")
def test_real_glofas_with_ldd_on_the_nwa12_cut():
    """Ruling R11: CEFI's exact ldd rule on the cut; Hudson, Delaware, Connecticut present; total conserved."""
    import warnings
    from core.grid.supergrid import grid_from_supergrid
    from forcing.sources.glofas import open_ldd
    hg = xr.open_dataset(_HGRID)
    depth = xr.open_dataset(_HGRID.replace("ocean_hgrid", "ocean_topog")).depth.values
    x, y = hg.x.values[1::2, 1::2], hg.y.values[1::2, 1::2]
    jj, ii = np.nonzero((x > -75.5) & (x < -68.5) & (y > 36.5) & (y < 41.8))
    g = grid_from_supergrid(hg, depth, window=(ii.min(), ii.max() + 1, jj.min(), jj.max() + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", UserWarning)                   # the ldd path does not warn
        r = glofas_mouths(_GLOFAS, g, "2017-05-01", "2017-09-01", ldd=open_ldd(_LDD))
    assert r.method == "ldd" and r.mouth_lat.size > 0
    ds = xr.open_mfdataset(_GLOFAS, combine="by_coords")
    lat, lon = ds.latitude.values, ds.longitude.values - 360.0
    t = np.isin(ds.valid_time.values - np.timedelta64(12, "h"), r.times)
    q = ds.avg_dis.values[t].astype(float)
    expect = sum(f * q[:, int(np.argmin(np.abs(lat - la))), int(np.argmin(np.abs(lon - lo)))]
                 for la, lo, f in zip(r.mouth_lat, r.mouth_lon, r.mouth_inside))
    np.testing.assert_allclose(r.Q.sum(axis=1), expect, rtol=1e-14, atol=0)
    qm = r.mouth_Q.mean(axis=0)
    big = {(round(float(la), 3), round(float(lo), 3)) for la, lo, qq in zip(r.mouth_lat, r.mouth_lon, qm) if qq > 300}
    assert {(40.875, -73.925), (41.275, -72.325), (39.875, -75.225)} <= big   # Hudson, Connecticut, Delaware


def test_substitute_warns():
    """Ruling R11: without ldd the substitute is used with a UserWarning naming the ldd file."""
    with pytest.warns(UserWarning, match="ldd"):
        _river()
