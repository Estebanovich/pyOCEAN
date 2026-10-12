"""Tests of the BayIW 2-D section tools (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.bayiw_baroclinic import bayiw

HAVE = (bayiw.CASES["2layer"]["dir"] / "input").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="BayIW MITgcm inputs not mounted")


def test_section_column_is_i280_with_the_bay():
    from validation.bayiw_2d import section
    d = section.section_inputs("2layer", "bay")
    assert d["dx"].shape == (1,) and d["dy"].shape == (352,) and d["H"].shape == (352, 1)
    assert 120.0 < d["H"][345, 0] < 170.0 and abs(d["H"][200, 0] - 599.9) < 1e-9
    assert d["T"].shape == (50, 352, 1) and d["wind"].shape == (180, 352, 1)
    assert section.SECTION_STATIONS == {"Gamma": 215, "C": 265, "Omega": 315, "Seiche2": 325, "Seiche": 345}
    full = bayiw.load_inputs("2layer", "bay")
    np.testing.assert_array_equal(d["T"][:, :, 0], full[3][:, :, 280])
    assert d["x0"] == bayiw.X0 + float(full[0][:280].sum())


def test_mitgcm_inputs_round_trip_to_the_column(tmp_path):
    # review focus 1: big-endian float64, (Nr, Ny, Nx) and (nrec, Ny, Nx); Nx = 3 with land walls at i = 0, 2
    from validation.bayiw_2d import section
    section.write_mitgcm_inputs("2layer", "bay", tmp_path)
    d = section.section_inputs("2layer", "bay")
    T = np.fromfile(tmp_path / "input" / "theta.bin", ">f8").reshape(50, 352, 3)
    W = np.fromfile(tmp_path / "input" / "wind.bin", ">f8").reshape(180, 352, 3)
    B = np.fromfile(tmp_path / "input" / "bathy.bin", ">f8").reshape(352, 3)
    for i in range(3):
        np.testing.assert_array_equal(T[:, :, i], d["T"][:, :, 0])
        np.testing.assert_array_equal(W[:, :, i], d["wind"][:, :, 0])
    np.testing.assert_array_equal(B[:, 1], np.where(np.isfinite(d["H"][:, 0]), -d["H"][:, 0], 0.0))
    assert (B[:, 0] == 0).all() and (B[:, 2] == 0).all()
    data = (tmp_path / "data").read_text()
    assert "endTime=216000." in data and "'input/theta.bin'" in data and "'input/wind.bin'" in data
    assert "../input" not in data and "hydrogSaltFile" not in data and "deltaT=30." in data
    obcs = (tmp_path / "data.obcs").read_text()
    assert "OB_Jsouth = 0, 1, 0," in obcs and "OB_Iwest" not in obcs and "OBCS_balanceFacS=-1." in obcs


def test_mitgcm_periodic_variant_has_three_wet_columns(tmp_path):
    from validation.bayiw_2d import section
    section.write_mitgcm_inputs("linear", "nobay", tmp_path, periodic=True)
    B = np.fromfile(tmp_path / "input" / "bathy.bin", ">f8").reshape(352, 3)
    np.testing.assert_array_equal(B[:, 0], B[:, 1])
    np.testing.assert_array_equal(B[:, 2], B[:, 1])
    assert "OB_Jsouth = 3*1," in (tmp_path / "data.obcs").read_text()
    assert "hydrogSaltFile" not in (tmp_path / "data").read_text()


def test_read_mitgcm_section_shapes():
    # review focus 3: Nx = 3 (section = column 1, Xp1 = 4), time variable T, 241 dumps over 60 h
    from validation.bayiw_2d import section
    if not (section.RUNS / "mitgcm" / "2layer_bay").exists():
        pytest.skip("MITgcm section run not done yet")
    d = section.read_mitgcm("2layer", "bay")
    n = d["t_h"].size
    assert abs(d["t_h"][-1] - 60.0) < 1e-6 and n == 241
    assert d["eta"].shape == (n, 352) and d["T"].shape == (n, 50, 352) and d["V"].shape == (n, 50, 353)
    assert d["U"].shape == (n, 50, 352) and np.abs(d["U"]).max() == 0.0          # walls: no cross-section flow
    np.testing.assert_allclose(d["T"][0, 0, 200], 22.767, atol=1e-3)
    assert np.isfinite(d["eta"]).all() and np.abs(d["eta"]).max() > 0


def test_section_model_steps():
    # review focus 4: nx = 1 with closed west/east edges and open south
    from validation.bayiw_2d import section
    for coord in ("zstar", "hybrid"):
        m, st, wind = section.build_model("2layer", "bay", coord)
        assert st.h.shape[1:] == (352, 1)
        tau = np.zeros((353, 1))
        tau[:-1, :] = wind[3]
        for _ in range(3):
            st = m.step(st, tau_y=tau)
        assert np.isfinite(st.u).all() and np.isfinite(st.v).all() and st.n_trunc == 0
        assert np.abs(st.u).max() == 0.0                                       # walls: no cross-section flow
        np.testing.assert_allclose(st.h.sum(0), np.nan_to_num(m.H) + st.eta, rtol=1e-12, atol=1e-12)


def test_mit_at_depth_samples_surface_first_levels():
    # MITgcm levels are surface first; a field quadratic in z must come back as the linear interpolation between
    # the two neighbouring level centres at 110 m (not between the bed and the surface), NaN below the bed
    from validation.bayiw_2d import compare
    zc = bayiw.Z_C
    T = np.broadcast_to((zc ** 2)[None, :, None], (2, 50, 4)).copy()
    H = np.array([600.0, 300.0, 100.0, np.nan])
    out = compare.mit_at_depth(dict(T=T, zc=zc), H, "T", bayiw.Z110)
    np.testing.assert_allclose(out[:, :2], np.interp(bayiw.Z110, zc[::-1], (zc ** 2)[::-1]), rtol=1e-12)
    assert np.isnan(out[:, 2:]).all()


def test_mitgcm_nobal_variant_turns_the_balance_off(tmp_path):
    # MITgcm balanceFacS = −1 zeroes the net south transport (obcs_balance_flow.F:76): closed for the
    # barotropic mode in a section; 'nobal' keeps Orlanski + sponge only, the closest to pyOCEAN's Flather
    from validation.bayiw_2d import section
    section.write_mitgcm_inputs("2layer", "bay", tmp_path, balance=False)
    obcs = (tmp_path / "data.obcs").read_text()
    assert "useOBCSbalance=.FALSE." in obcs and "balanceFacS" not in obcs and "useOrlanskiSouth=.TRUE." in obcs


def test_mitgcm_orl_variant_has_no_sponge(tmp_path):
    # 'orl': Orlanski only (no balance, no velocity sponge), to test whether the full-velocity sponge reflects
    from validation.bayiw_2d import section
    section.write_mitgcm_inputs("2layer", "bay", tmp_path, balance=False, sponge=False)
    obcs = (tmp_path / "data.obcs").read_text()
    assert "useOBCSsponge=.FALSE." in obcs and "useOBCSbalance=.FALSE." in obcs and "OBCS_PARM03" not in obcs


def test_section_hybrid_layer_count_is_a_parameter():
    from validation.bayiw_2d import section
    m, st, _ = section.build_model("linear", "bay", "hybrid", n_layers=50)
    assert st.h.shape[0] == 50
