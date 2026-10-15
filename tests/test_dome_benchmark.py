"""DOME benchmark pipeline (spec docs/superpowers/specs/2026-10-14-dome-overflow-design.md §5–§7)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.dome import cases


def test_bathymetry_matches_mom6_dome_initialize_topography():
    c = cases.get(10.0)
    assert (c.nx, c.ny) == (150, 70)
    j_shelf = np.searchsorted(c.y, 450e3)                            # mid-slope: 600 + 0.01·(600 − 450) km
    np.testing.assert_allclose(c.H[j_shelf, 0], 600.0 + 0.01 * (600e3 - c.y[j_shelf]), rtol=1e-12)
    assert np.all(c.H[c.y < 300e3] == 3600.0)
    north = c.y >= 600e3
    chan = (c.x > 1000e3) & (c.x < 1100e3)
    assert np.all(c.H[np.ix_(north, chan)] == 600.0)
    assert np.all(np.isnan(c.H[np.ix_(north, ~chan)]))


def test_inflow_transport_is_legg_and_close_to_mom6_by_density_class():
    c = cases.get(10.0)
    p = cases.inflow_profile(c)
    db0 = cases.G * cases.DRHO / cases.RHO0
    legg = db0 * cases.H0**2 / (2 * c.f0)                            # Legg 2006 §2: T_in = Δb₀ h₀² / (2f) ≈ 8.55 Sv
    assert abs(-p["transport"].sum() - legg) / legg < 0.05
    rho = cases.RHO0 - cases.ALPHA * (p["T"] - cases.T_REF)
    dz = -np.diff(p["zw"], axis=0)
    flux = -p["v"] * dz * c.dx                                         # southward flux of every profile cell
    dense = flux[rho > 1031.9].sum()
    mom6 = cases.mom6_layer_transports()
    assert abs(dense - (-mom6[-1])) / (-mom6[-1]) < 0.25              # densest class: 7.45 Sv in MOM6 (diff. (c))


def test_inflow_profile_is_geostrophic_and_ri_bounded():
    c = cases.get(10.0)
    p = cases.inflow_profile(c)
    i = np.argmax(-p["transport"])                                    # the right-hand (west) wall of the channel
    assert p["s"][:, i].max() == 1.0 and p["s"][:, i].min() == 0.0
    assert np.all(np.diff(p["T"][:, i]) <= 1e-12)                     # stable: T decreases downward (surface first)
    assert np.all(p["v"] <= 0.0)                                       # southward into the domain


def test_sponge_profile_matches_mom6():
    c = cases.get(10.0)
    r = cases.sponge_rate(c)
    peak = 10.0 / 86400.0
    assert r[0, 0] == pytest.approx(peak) and r[0, 75] == 0.0
    i150 = np.argmin(np.abs(c.x - 150e3))                              # on the ramp (cell centre 145 km)
    assert r[0, i150] == pytest.approx(peak * (200e3 - c.x[i150]) / 100e3, rel=1e-12)


@pytest.mark.parametrize("coord", ["zstar", "sigma", "hybrid"])
def test_pyocean_dome_builds_and_takes_steps_with_inflow_tracer_and_sponge(coord):
    from validation.dome import run_pyocean
    c = cases.get(50.0)                                               # 30 × 14 coarse smoke grid
    m, st = run_pyocean.build(c, coord)
    for _ in range(6):
        st = m.step(st)
    assert np.isfinite(st.u).all() and st.tr.max() > 0.0               # dye entered through the channel
    q = st.Qy[:, -1, :].sum()
    np.testing.assert_allclose(q, cases.inflow_profile(c)["transport"].sum(), rtol=1e-10)


def _synthetic_series(nt=3):
    """A uniform-layer box with a dye slab at y ∈ [200, 300] km, 1 m/s westward flow inside it."""
    c = cases.get(50.0)
    nz = 4
    h = np.full((nt, nz, c.ny, c.nx), 900.0)
    s_ = np.zeros_like(h)
    in_slab = (c.y >= 200e3) & (c.y <= 300e3)
    s_[:, -1][:, in_slab, :] = 1.0                                     # bottom layer (surface-first index −1)
    u = np.zeros((nt, nz, c.ny, c.nx + 1))
    u[:, -1][:, in_slab, :] = -1.0
    T = np.full_like(h, cases.T_of_rho(1031.0))
    return dict(t=np.arange(nt) * 86400.0, x=c.x, y=c.y, dx=np.full(c.nx, c.dx), dy=np.full(c.ny, c.dx),
                H=np.full((c.ny, c.nx), 3600.0), h=h, T=T, s=s_, u=u, v=np.zeros((nt, nz, c.ny + 1, c.nx)),
                eta=np.zeros((nt, c.ny, c.nx)))


def test_metrics_on_a_synthetic_dye_slab():
    from validation.dome import metrics
    s = _synthetic_series()
    x, Y = metrics.path_Y(s, 0.0, 2 * 86400.0)
    np.testing.assert_allclose(Y, 250e3, rtol=0.05)                    # slab centre (cell-centred quadrature)
    x, T = metrics.plume_transport(s, 0.0, 2 * 86400.0)
    n_slab = ((s["y"] >= 200e3) & (s["y"] <= 300e3)).sum()
    np.testing.assert_allclose(T[1:-1], 1.0 * 900.0 * n_slab * s["dy"][0], rtol=1e-12)
    assert metrics.final_density(s, 0.0, 2 * 86400.0) == pytest.approx(1031.0)
    assert metrics.centroid_depth(s, 0.0) == pytest.approx(3600.0 - 450.0)
    assert metrics.ranges(s)["s"] == (0.0, 1.0)


def test_mom6_input_has_the_common_physics_and_the_dome_configs():
    from validation.dome import mom6
    txt = mom6.mom_input(cases.get(10.0), "common")
    for line in ('TOPO_CONFIG = "DOME"', 'OBC_USER_CONFIG = "DOME"', 'SPONGE_CONFIG = "DOME"', "USE_DOME_TRACER = True",
                 "LAPLACIAN = True", "KH = 50", "BIHARMONIC = False", "KV = 0.5", "KD = 0.0",
                 "USE_JACKSON_PARAM = False", "CDRAG = 0.002", "DRAG_BG_VEL = 0.0", "HBBL = 10.0", "DAYMAX = 50.0",
                 "NIGLOBAL = 150", "NJGLOBAL = 70", "LAYOUT = 1, 1"):
        assert line in txt, line
    assert "KV = 0.0001" in mom6.mom_input(cases.get(10.0), "kv_low")
    lines = mom6.override_lines(cases.get(10.0), "common")
    assert "#override KV = 0.5" in lines and "KH = 50" in lines and "LAPLACIAN = True" in lines
    assert '#override OBC_SEGMENT_001 = "J=N,I=220:200,SIMPLE"' in mom6.override_lines(cases.get(5.0), "common")


def test_mitgcm_namelists_have_the_variant_physics():
    from validation.dome import mitgcm
    d = mitgcm.data_text(cases.get(10.0), "legg")
    for s in ("viscAh=50.", "viscAr=0.5", "bottomDragQuadratic=0.002", "no_slip_bottom=.TRUE.", "diffKhT=0.",
              "diffKrT=0.", "tempAdvScheme=77", "nonlinFreeSurf=4", "select_rStar=2", "f0=0.0001", "hFacMin=0.1"):
        assert s in d, s
    assert "no_slip_bottom=.FALSE." in mitgcm.data_text(cases.get(10.0), "common")
    o = mitgcm.obcs_text(cases.get(10.0))
    assert "useOBCSprescribe=.TRUE." in o and "OBNvFile='OBNv.bin'" in o and "OBNptrFile(1)='OBNs.bin'" in o


def test_mitgcm_inflow_files_carry_the_case_transport():
    from validation.dome import mitgcm
    c = cases.get(10.0)
    v, T, s_ = mitgcm.inflow_levels(c)
    dz = mitgcm.wet_levels(c)                                        # (Nr,) wet thickness of the 600-m channel column
    q = (v * dz[:, None]).sum(axis=0) * c.dx
    np.testing.assert_allclose(q[:c.nx], cases.inflow_profile(c)["transport"], rtol=1e-12, atol=1e-6)


def test_mitgcm_reader_joins_mnc_files_split_at_the_size_limit(tmp_path):
    """mnc starts a new diag file when one reaches mnc_max_fsize (the 5-km diag3d): the reader joins them in time."""
    import netCDF4
    from validation.dome import mitgcm
    for it, times in ((0, [0.0, 86400.0]), (14113, [172800.0])):
        with netCDF4.Dataset(tmp_path / f"diag3d.{it:010d}.t001.nc", "w") as ds:
            ds.createDimension("T", None)
            ds.createDimension("Z", 2)
            ds.createVariable("T", "f8", ("T",))[:] = times
            ds.createVariable("THETA", "f8", ("T", "Z"))[:] = np.array(times)[:, None] * np.ones(2)
    f = str(tmp_path / "diag3d.*.t001.nc")
    np.testing.assert_array_equal(mitgcm._join(f, "T"), [0.0, 86400.0, 172800.0])
    assert mitgcm._join(f, "THETA").shape == (3, 2)


def test_mitgcm_sponge_relaxes_the_dye(tmp_path):
    """rbcs relaxes ptracer 1 only with its own mask, relaxMaskFile(3) (1 = T, 2 = S; rbcs_init_fixed.F): without it
    MITgcm warns 'No relaxation for pTr = 1' and the dye piles up in the west sponge."""
    from validation.dome import mitgcm
    txt = (mitgcm.write_run(10.0, tmp_path, "common", days=1.0) / "data.rbcs").read_text()
    assert "relaxMaskFile(3)='rbcs_mask.bin'" in txt


def test_pyocean_default_time_step_is_the_stable_one():
    """At 10 km / 1200 s (MOM6's step) θ_bc = ½ grows a basin mode until the run stops at day 36 (README point 5);
    the pyOCEAN runner defaults to 600 s, MOM6 keeps the case step."""
    from validation.dome import run_pyocean
    assert run_pyocean.default_dt(cases.get(10.0)) == 600.0
    assert run_pyocean.default_dt(cases.get(5.0)) == 600.0
    assert cases.get(10.0).dt == 1200.0
