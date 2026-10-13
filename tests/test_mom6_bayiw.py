"""MOM6 BayIW benchmark inputs, namelists and readers (spec 2026-10-13-mom6-bayiw-benchmark-design); no simulation."""
from __future__ import annotations

import numpy as np
import pytest
netCDF4 = pytest.importorskip("netCDF4")


def test_ts_z_centres_reproduce_the_mitgcm_interfaces():
    from validation.bayiw_baroclinic import bayiw
    from validation.mom6_bayiw import inputs
    c = inputs.source_centres()
    edges = 0.5 * (c[1:] + c[:-1])                     # MOM6: midpoints of the centres are the source edges
    np.testing.assert_allclose(edges, -bayiw.Z_F[1:-1], atol=1e-9)
    assert (np.diff(c) > 0).all()


def test_mom6_inputs_match_bayiw_at_corners_and_stations(tmp_path):
    from validation.bayiw_2d import section
    from validation.mom6_bayiw import inputs
    d = inputs.domain("section", "2layer", "bay")
    inputs.write_inputs("section", "2layer", "bay", tmp_path)
    with netCDF4.Dataset(tmp_path / "INPUT" / "topog.nc") as ds:
        depth = ds["depth"][:]
    s = section.section_inputs("2layer", "bay")
    assert depth.shape == (352, inputs.NX_SECTION) and (depth[:, 0] == 0).all() and (depth[:, 2:] == 0).all()
    np.testing.assert_allclose(depth[:, 1], np.nan_to_num(s["H"][:, 0]))
    with netCDF4.Dataset(tmp_path / "INPUT" / "ocean_hgrid.nc") as ds:
        dxs = ds["dx"][:]
        assert ds.dimensions["nxp"].size == 2 * inputs.NX_SECTION + 1 and ds.dimensions["nyp"].size == 705
    np.testing.assert_allclose(dxs[0, 2] + dxs[0, 3], s["dx"][0])  # two half-cell lengths = the cell width
    with netCDF4.Dataset(tmp_path / "INPUT" / "ts_z.nc") as ds:
        T = ds["ptemp"][0]
        assert "_FillValue" in ds["ptemp"].ncattrs()
    j = section.SECTION_STATIONS["Omega"]
    np.testing.assert_allclose(T[:, j, 1], s["T"][:, j, 0])


def test_mom6_wind_file_is_on_north_faces_and_records(tmp_path):
    from validation.mom6_bayiw import inputs
    d = inputs.domain("section", "2layer", "bay")
    inputs.write_inputs("section", "2layer", "bay", tmp_path)
    with netCDF4.Dataset(tmp_path / "INPUT" / "wind.nc") as ds:
        ty = ds["STRESS_Y"][:]
        assert ds["STRESS_X"][:].max() == 0.0
    # bayiw wind row j = south face of cell j (MITgcm); MOM6 row j = north face of cell j = bayiw row j + 1; the
    # northern coast face is 0 and the open southern edge cannot be given (MOM6 fills it with 0)
    assert ty.shape == (d["wind"].shape[0], 352, inputs.NX_SECTION)
    np.testing.assert_allclose(ty[7, :-1, 1], d["wind"][7, 1:, 1])
    assert (ty[:, -1] == 0).all()


def test_coordinate_files_hold_the_bayiw_targets(tmp_path):
    from validation.bayiw_baroclinic import bayiw
    from validation.mom6_bayiw import inputs
    inputs.write_inputs("section", "2layer", "bay", tmp_path)
    with netCDF4.Dataset(tmp_path / "INPUT" / "hyb.nc") as ds:
        sig, dz = ds["sigma2"][:], ds["dz"][:]
    np.testing.assert_allclose(dz, bayiw.hybrid_dz_top(25))
    assert sig.size == 26 and 996.0 < sig.min() < sig.max() < 1000.0 and (np.diff(sig) > 0).all()
    with netCDF4.Dataset(tmp_path / "INPUT" / "layers.nc") as ds:
        lay = ds["Layer"][:]
    np.testing.assert_allclose(lay, 0.5 * (sig[1:] + sig[:-1]))


def test_mom_input_section_has_walls_and_south_obc(tmp_path):
    from validation.mom6_bayiw import config
    d = config.write_case("section", "2layer", "bay", "hycom1", tmp_path, hours=60.0)
    txt = (d / "MOM_input").read_text()
    for s in ('REENTRANT_X = False', 'GRID_CONFIG = "mosaic"', 'ROTATION = "betaplane"', 'BETA = 0.0',
              'EQN_OF_STATE = "LINEAR"', 'DRHO_DS = 0.0', 'KE_SCHEME = "KE_ARAKAWA"', 'KH = 100.0',
              'KHTR = 100.0', 'KV = 1.0E-05', 'KD = 1.0E-05', 'BOTTOMDRAGLAW = False', 'NOSLIP = False',
              'REGRIDDING_COORDINATE_MODE = "HYCOM1"', 'ALE_COORDINATE_CONFIG = "HYBRID:hyb.nc,sigma2,dz"',
              'REMAPPING_SCHEME = "PPM_H4"', 'Z_INIT_ALE_REMAPPING = True', 'WIND_FILE_DAYS_PER_RECORD = -72',
              'OBC_NUMBER_OF_SEGMENTS = 1', 'OBC_SEGMENT_001 = "J=0,I=0:N,FLATHER,ORLANSKI"', 'DT = 120.0',
              'LAYOUT = 1, 5', 'NIGLOBAL = 10', 'NJGLOBAL = 352', 'NK = 25'):
        assert s in txt, s
    assert "clock_grain = 'ROUTINE'" in (d / "input.nml").read_text()


def test_mom_input_cut_has_three_open_edges_and_zstar_levels(tmp_path):
    from validation.bayiw_baroclinic import bayiw
    from validation.mom6_bayiw import config
    d = config.write_case("cut", "2layer", "bay", "zstar", tmp_path, hours=3.0)
    txt = (d / "MOM_input").read_text()
    assert 'OBC_NUMBER_OF_SEGMENTS = 3' in txt and 'NK = 50' in txt and 'NIGLOBAL = 198' in txt
    assert 'REGRIDDING_COORDINATE_MODE = "ZSTAR"' in txt
    assert "ALE_RESOLUTION = " + ", ".join(f"{x:.6g}" for x in bayiw.DELR) in txt


def test_series_from_mom6_matches_the_pyocean_record_layout(tmp_path):
    from validation.bayiw_2d import section
    from validation.mom6_bayiw import inputs, read_mom6
    H = np.nan_to_num(section.section_inputs("2layer", "bay")["H"][:, 0])
    nt, nk, ny, nx = 3, 4, 352, inputs.NX_SECTION
    h = np.repeat((H / nk)[None, None, :, None], nk, 1).repeat(nt, 0).repeat(nx, 3)
    temp = 15.0 + np.arange(nk)[None, :, None, None] * np.ones((nt, nk, ny, nx))     # surface first, as MOM6
    with netCDF4.Dataset(tmp_path / "prog.nc", "w") as ds:
        for n, s in (("Time", nt), ("zl", nk), ("yh", ny), ("xh", nx), ("yq", ny + 1)):
            ds.createDimension(n, s)
        ds.createVariable("Time", "f8", ("Time",))[:] = np.array([900.0, 1800.0, 2700.0]) / 86400.0
        ds["Time"].units = "days since 0001-01-01 00:00:00"
        ds.createVariable("h", "f8", ("Time", "zl", "yh", "xh"))[:] = h
        ds.createVariable("temp", "f8", ("Time", "zl", "yh", "xh"))[:] = temp
        ds.createVariable("v", "f8", ("Time", "zl", "yq", "xh"))[:] = 0.0
        ds.createVariable("SSH", "f8", ("Time", "yh", "xh"))[:] = 0.01
    s = read_mom6.section_series(tmp_path, "2layer", "bay")
    for k in ("t_h", "eta", "T110", "T250", "V110", "T_sec", "V_sec", "n_trunc", "heat", "rpe"):
        assert k in s, k
    np.testing.assert_allclose(s["t_h"], [0.25, 0.5, 0.75])
    np.testing.assert_allclose(s["eta"][0], 0.01)
    exp = section.to_mitgcm_levels(h[0, ::-1, :, 1:2], temp[0, ::-1, :, 1:2], H[:, None])[::-1, :, 0]
    np.testing.assert_allclose(s["T_sec"][0], exp)                # NaN below the bed, as run_pyocean.record
    assert s["T_sec"].shape == (3, 50, ny) and s["V_sec"].shape == (3, 50, ny + 1)


def test_hybgen_layers_come_from_the_linear_coordinate(tmp_path):
    # FMS refuses the 1-D read of COORD_FILE ("FMS I/O requires a domain input"); the two-layer targets are evenly
    # spaced, so COORD_CONFIG = "linear" gives MOM6 exactly their midpoints (MOM_coord_initialization.F90:526)
    from validation.mom6_bayiw import config, inputs
    d = config.write_case("section", "2layer", "bay", "hybgen", tmp_path, hours=1.0)
    txt = (d / "MOM_input").read_text()
    dm = inputs.domain("section", "2layer", "bay")
    t = inputs.targets("2layer", dm["T"], dm["H"])
    assert 'COORD_CONFIG = "linear"' in txt and 'COORD_FILE' not in txt
    assert f"LIGHTEST_DENSITY = {t[0]:.10g}" in txt and f"DENSITY_RANGE = {t[-1] - t[0]:.10g}" in txt
    with pytest.raises(ValueError, match="evenly"):
        config.write_case("section", "real", "bay", "hybgen", tmp_path / "r", hours=1.0)
