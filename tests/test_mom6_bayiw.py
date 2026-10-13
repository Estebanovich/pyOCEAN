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
    assert depth.shape == (352, 3) and (depth[:, 0] == 0).all() and (depth[:, 2] == 0).all()
    np.testing.assert_allclose(depth[:, 1], np.nan_to_num(s["H"][:, 0]))
    with netCDF4.Dataset(tmp_path / "INPUT" / "ocean_hgrid.nc") as ds:
        dxs = ds["dx"][:]
        assert ds.dimensions["nxp"].size == 7 and ds.dimensions["nyp"].size == 705
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
    assert ty.shape == (d["wind"].shape[0], 352, 3)
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
