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


def test_dome_gls_short_run():
    """
    ``--closure gls`` (N3 task 10, spec decision 10): the hybrid DOME with the stratified GLS instead of the constant
    ν_v — only the closure changes: the quadratic drag C_d over hbbl is the baseline's, kv = 0 so κ_t keeps the GLS
    floor 1e-6.  Coarse 50 km grid, 6 steps: dye enters with the exact transport, the GLS mixes, T stays in its
    initial range (FCT), no CFL truncation.
    """
    import warnings
    from validation.dome import run_pyocean
    warnings.simplefilter("ignore")
    c = cases.get(50.0)
    m, st = run_pyocean.build(c, "hybrid", closure="gls")
    m0, _ = run_pyocean.build(c, "hybrid")
    assert (m.core.closure, m.gls_stratified, m.kv) == ("gls", True, 0.0)
    assert m._closure.gls.kt_min == 1e-6                                # the GLS (CROCO nuws) floor of κ_t
    assert m0.core.closure == "constant"
    for name in ("cd", "hbbl"):                                          # the drag of the baseline
        assert getattr(m._bed_drag, name) == getattr(m0._bed_drag, name) == getattr(c, name)
    T0 = (np.nanmin(st.T[:, m.fs.active]), np.nanmax(st.T[:, m.fs.active]))
    for _ in range(6):
        st = m.step(st)
    assert np.isfinite(st.u).all() and st.tr.max() > 0.0
    np.testing.assert_allclose(st.Qy[:, -1, :].sum(), cases.inflow_profile(c)["transport"].sum(), rtol=1e-10)
    akt = st.closure["akt"]
    assert np.isfinite(akt).all() and akt.min() >= 1e-6 and akt.max() > 1e-4      # the GLS mixes
    T = st.T[:, m.fs.active]
    inflow_T = cases.inflow_profile(c)["T"]
    lo, hi = min(T0[0], np.min(inflow_T)), max(T0[1], np.max(inflow_T))
    assert lo - 1e-9 <= T.min() and T.max() <= hi + 1e-9
    assert st.n_trunc == 0
    with pytest.raises(ValueError, match="nu-v"):
        run_pyocean.closure_kwargs(c, "gls", nu_v=0.5)


def test_dome_smag_option_reaches_model():
    """--smag C reaches SigmaBaroclinic(smagorinsky=C); the tag gets '_smag<C>'; one step records n_visc_limited and ν."""
    from validation.dome import run_pyocean
    c = cases.get(50.0)
    m, st = run_pyocean.build(c, "hybrid", closure="gls", smag=0.15)
    assert m.smagorinsky == 0.15
    st = m.step(st)
    assert st.n_visc_limited is not None and np.nanmax(m._momentum.last_nu_t) >= c.nu_h
    m0, st0 = run_pyocean.build(c, "zstar")
    assert m0.smagorinsky == 0.0 and m0.step(st0).n_visc_limited is None
    meta = run_pyocean.run(50.0, "hybrid", dt=600.0, days=600.0 * 2 / 86400.0, threads=1, closure="gls", write=False,
                           smag=0.15)
    assert meta["smag"] == 0.15 and meta["n_visc_limited"] >= 0 and meta["nu_t_max"] >= c.nu_h
    assert "smag" not in run_pyocean.run(50.0, "zstar", dt=600.0, days=600.0 * 2 / 86400.0, threads=1, write=False)
    kv = run_pyocean.run(50.0, "hybrid", dt=600.0, days=600.0 * 2 / 86400.0, threads=1, write=False, kv_low=True,
                         smag=0.15)
    assert kv["nu_v"] == 1e-4


# ---- Jackson shear-mixing plan, Task 6: Legg (2006) metrics against the non-hydrostatic truth ---------------------

def _corner_series(nt=3, nz=4, U0=0.3, V0=0.2, h=900.0):
    """Uniform-layer box on the 10-km grid: dye s = 1 in the bottom layer everywhere and also in the layer above it
    in the corner row y' ∈ (−50, −40) km; westward U0 at every u-face of the dyed cells, southward V0 at the
    y' = 0 faces of the corner columns (bottom layer)."""
    c = cases.get(10.0)
    shp = (nt, nz, c.ny, c.nx)
    hh = np.full(shp, h)
    s_ = np.zeros(shp)
    s_[:, -1] = 1.0
    j55 = int(np.argmin(np.abs(c.y - 555e3)))                          # the row −50 < y' < −40 km
    s_[:, -2, j55, :] = 1.0
    u = np.zeros((nt, nz, c.ny, c.nx + 1))
    u[:, -1] = -U0
    u[:, -2, j55, :] = -U0
    v = np.zeros((nt, nz, c.ny + 1, c.nx))
    i0, i1 = int(np.argmin(np.abs(c.x - 1005e3))), int(np.argmin(np.abs(c.x - 1095e3)))
    jf = int(np.argmin(np.abs((np.arange(c.ny + 1) * c.dx) - 600e3)))    # the v-face at y' = 0
    v[:, -1, jf, i0:i1 + 1] = -V0
    T = np.full(shp, cases.T_of_rho(1031.0))
    return c, dict(t=np.arange(nt) * 86400.0, x=c.x, y=c.y, dx=np.full(c.nx, c.dx), dy=np.full(c.ny, c.dx),
                   H=np.full((c.ny, c.nx), nz * h), h=hh, T=T, s=s_, u=u, v=v, eta=np.zeros((nt, c.ny, c.nx)))


def test_corner_entrainment_synthetic():
    """Legg eq. 14 in −50 < x' < 50, −50 < y' < 0 km: (T_out − T_in)/(U_rms S) with T_out through x' = −50 km,
    T_in through y' = 0, S the area of the s = 0.01 surface inside the corner (tops of the dyed columns plus the
    side of the thicker row), U_rms over the dyed cells of the corner."""
    from validation.dome import metrics
    U0, V0, h = 0.3, 0.2, 900.0
    c, s = _corner_series(U0=U0, V0=V0, h=h)
    dx = c.dx
    T_out = U0 * h * dx * (5 + 1)                    # 5 rows of the bottom layer + the second layer of row 55
    T_in = V0 * h * dx * 10                          # 10 corner columns
    S = 10 * 5 * dx * dx + 10 * h * dx               # tops + the side between rows 55 and 56
    # U_rms over the dyed corner cells: (u, v) at cell centres; row 59 holds ½V0 in its bottom layer
    n_cells = 10 * 6
    U2 = (n_cells * U0 ** 2 + 10 * (0.5 * V0) ** 2) / n_cells
    E = (T_out - T_in) / (np.sqrt(U2) * S)
    got = metrics.corner_entrainment(s, 0.0, 2 * 86400.0)
    assert got == pytest.approx(E, rel=1e-6)
    parts = metrics.corner_entrainment(s, 0.0, 2 * 86400.0, parts=True)
    assert parts["T_out"] == pytest.approx(T_out, rel=1e-12) and parts["T_in"] == pytest.approx(T_in, rel=1e-12)
    assert parts["S"] == pytest.approx(S, rel=1e-12)


def test_tracer_by_class_conserves():
    """Σ_c Σ_x S(c, x) Δx = the time mean of Σ s V in the window (Legg Fig. 4: ∫ s dy dz per x and class)."""
    from validation.dome import metrics
    _, s = _corner_series()
    rng = np.random.default_rng(1)
    s["s"] = rng.uniform(0.0, 1.0, s["s"].shape)
    s["T"] = cases.T_of_rho(rng.uniform(1030.0, 1032.0, s["T"].shape))
    s["h"] = s["h"] * rng.uniform(0.0, 2.0, s["h"].shape)
    x, edges, S = metrics.tracer_by_class(s, 86400.0, 2 * 86400.0)
    assert S.shape == (19, x.size) and edges.size == 20
    np.testing.assert_allclose(edges, np.linspace(-0.019, 0.0, 20))
    want = np.mean([(s["s"][n] * metrics._vol(s)[n]).sum() for n in (1, 2)])
    assert (S * s["dx"][None, :]).sum() == pytest.approx(want, rel=1e-12)


def test_tracer_by_class_puts_the_inflow_where_legg_does():
    """The undiluted inflow (ρ = 1032, b = −0.019011) falls in Legg's densest class (−0.019, −0.018) (Fig. 4, JSON),
    and the lightest water (b = 0) in (−0.001, 0)."""
    from validation.dome import metrics
    _, s = _corner_series()
    s["T"][:] = cases.T_of_rho(1032.0)
    s["T"][:, 0] = cases.T_of_rho(1030.0)
    s["s"][:] = 1.0
    x, edges, S = metrics.tracer_by_class(s, 0.0, 0.0)
    tot = S.sum(axis=1)
    assert tot[0] > 0 and tot[-1] > 0 and tot[1:-1].sum() == 0.0
    assert (edges[0], edges[1]) == pytest.approx((-0.019, -0.018)) and edges[-1] == 0.0


def test_corner_entrainment_with_ale_layering():
    """A hybrid-like corner column: uneven layers and a zero-thickness undyed layer between ambient and plume.  The
    vanished layer adds no horizontal piece; the raised plume top (1800 m instead of 2700 m) adds a riser of 900 m
    against each of its four region neighbours; U_rms weights the thicker dyed cell by its volume."""
    from validation.dome import metrics
    U0, V0, h = 0.3, 0.2, 900.0
    c, s = _corner_series(U0=U0, V0=V0, h=h)
    j = int(np.argmin(np.abs(c.y - 575e3)))
    i = int(np.argmin(np.abs(c.x - 1035e3)))
    s["h"][:, :, j, i] = [1200.0, 600.0, 0.0, 1800.0]
    s["s"][:, :, j, i] = [0.0, 0.0, 0.0, 1.0]
    dx = c.dx
    T_out, T_in = U0 * h * dx * 6, V0 * h * dx * 10
    S = 10 * 5 * dx * dx + 10 * h * dx + 4 * 900.0 * dx
    U2 = U0 ** 2 + 10 * (0.5 * V0) ** 2 / 61.0                        # 60 dyed cells of h, one of 2h
    p = metrics.corner_entrainment(s, 0.0, 2 * 86400.0, parts=True)
    assert p["S"] == pytest.approx(S, rel=1e-12)
    assert p["U_rms"] == pytest.approx(np.sqrt(U2), rel=1e-12)
    assert p["E"] == pytest.approx((T_out - T_in) / (np.sqrt(U2) * S), rel=1e-12)
    with pytest.raises(ValueError, match="urms"):
        metrics.corner_entrainment(s, 0.0, 0.0, urms="rms")


def test_max_class_without_dye_is_nan():
    from validation.dome import compare
    x = np.array([1050e3 - 300e3, 1050e3])
    b = compare.max_class(x, np.linspace(-0.019, 0.0, 20), np.zeros((19, 2)), -300.0)
    assert np.isnan(b[0]) and compare._cls(b) == "—"


def test_jackson_tag():
    """--shear jackson appends '_jackson' to the run tag, '_jacksonpr0' with --shear-pr 0; the flags reach the model."""
    from validation.dome import run_pyocean
    p = run_pyocean.parser()
    a = p.parse_args(["10", "hybrid", "--kv-low", "--smag", "0.15", "--shear", "jackson", "--shear-pr", "0"])
    assert run_pyocean.default_tag(a.closure, a.kv_low, a.smag, a.shear, a.shear_pr) == "kv_low_smag0.15_jacksonpr0"
    a = p.parse_args(["10", "hybrid", "--kv-low", "--smag", "0.15", "--shear", "jackson"])
    assert run_pyocean.default_tag(a.closure, a.kv_low, a.smag, a.shear, a.shear_pr) == "kv_low_smag0.15_jackson"
    assert run_pyocean.default_tag("constant", False, 0.0, "jackson", 1.0) == "jackson"
    assert run_pyocean.default_tag("constant", False, 0.0, None, 1.0) == ""
    assert run_pyocean.default_tag("gls", False, 0.15, None, 1.0) == "gls_smag0.15"
    m, _ = run_pyocean.build(cases.get(50.0), "zstar", nu_v=cases.NU_V_LOW, smag=0.15,
                             **run_pyocean.shear_kwargs("jackson", 0.0))
    assert m.shear_mixing == "jackson" and m.shear_prandtl == 0.0
    assert run_pyocean.shear_kwargs(None, 1.0) == {}


def test_legg_json_has_sources():
    """Every value of the Legg (2006) truth file carries its source (figure, PDF page) and method; digitized values
    also their reading uncertainty."""
    import json
    from pathlib import Path
    d = json.loads((Path(__file__).resolve().parents[1] / "validation/dome/legg2006_nonhydro.json").read_text())
    vals = d["values"]
    for key in ("corner_E_nonhydro", "corner_E_10km_z", "corner_E_10km_iso", "transport_nonhydro",
                "alphaE_nonhydro", "tracer_by_class_nonhydro", "bf_nonhydro", "bf_10km_z", "bf_10km_iso"):
        assert key in vals, key
    for key, v in vals.items():
        assert v.get("source") and v.get("method") in ("text", "digitized"), key
        if v["method"] == "digitized":
            assert "uncertainty" in v, key
    assert vals["corner_E_nonhydro"]["value"] == 8.6e-4 and vals["corner_E_nonhydro"]["method"] == "text"
