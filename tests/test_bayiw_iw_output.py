"""
Tests of the CF NetCDF output of the baroclinic BayIW runner
(validation/bayiw_iw/iw_output.py): stations as a CF discrete-sampling-geometry
timeSeriesProfile file, sections and z-slices in a second file; the records read
back in the runner's in-memory layout, and legacy .npz runs convert to the same files.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "bayiw_iw"))
import iw_output as io  # noqa: E402

NT, NK, NS, NY, NB = 5, 4, 3, 6, (3, 2)


@pytest.fixture
def run(tmp_path):
    """Synthetic records with the runner's shapes and the static metadata."""
    rng = np.random.default_rng(1)
    sigma_w = np.array([0.0, 0.1, 0.3, 0.6, 1.0])                          # model σ: bed = 0, surface = 1
    rec = {"t": 900.0 * np.arange(1, NT + 1),
           "eta_st": rng.normal(0, 1e-3, (NT, NS)),
           "T_st": rng.normal(15, 1, (NT, NK, NS)), "u_st": rng.normal(0, 0.1, (NT, NK, NS)),
           "v_st": rng.normal(0, 0.1, (NT, NK, NS)), "w_st": rng.normal(0, 1e-5, (NT, NK + 1, NS)),
           "T_axis": rng.normal(15, 1, (NT, NK, NY)), "v_axis": rng.normal(0, 0.1, (NT, NK, NY)),
           "w_axis": rng.normal(0, 1e-5, (NT, NK + 1, NY)), "eta_axis": rng.normal(0, 1e-3, (NT, NY)),
           "w_z": rng.normal(0, 1e-5, (NT, 2) + NB).astype(np.float32),
           "T_z": rng.normal(15, 1, (NT, 2) + NB).astype(np.float32)}
    rec["T_z"][:, 1, 0, 0] = np.nan                                          # below the bed
    static = {"stations": {"A": (1, 2), "B": (3, 4), "C": (5, 0)},
              "x_st": np.array([1e3, 3e3, 5e3]), "y_st": np.array([2e3, 4e3, 0.0]),
              "h_st": np.array([100.0, 160.0, 599.9]), "sigma_w": sigma_w,
              "x_axis": 7e3, "i_axis": 7, "y_axis": 1e3 * np.arange(NY), "h_axis": np.linspace(50, 600, NY),
              "depths": (110.0, 250.0), "box": ((1, 4), (2, 4)),
              "x_box": np.array([2e3, 3e3]), "y_box": np.array([1e3, 2e3, 3e3]),
              "attrs": {"case": "bay", "dt": 300.0}}
    return rec, static, tmp_path / "pyocean_iw_bay_test"


def test_round_trip(run):
    """Station series are exact (float64); sections and slices to float32 precision, NaN kept."""
    rec, static, stem = run
    io.write(stem, rec, static)
    back = io.read_records(stem)
    for k in ("t", "eta_st", "T_st", "u_st", "v_st", "w_st"):
        np.testing.assert_array_equal(back[k], rec[k], err_msg=k)
    for k in ("T_axis", "v_axis", "w_axis", "eta_axis", "w_z", "T_z"):
        np.testing.assert_allclose(back[k], rec[k], rtol=1e-6, atol=1e-12, err_msg=k)
    assert np.isnan(back["T_z"][:, 1, 0, 0]).all()
    np.testing.assert_allclose(back["sigma_w"], static["sigma_w"], atol=1e-15)   # σ_CF = σ − 1 round-off
    np.testing.assert_array_equal(back["h_st"], static["h_st"])
    assert back["stations"] == {k: tuple(v) for k, v in static["stations"].items()}


def test_cf_metadata(run):
    """DSG timeSeriesProfile for stations; σ formula_terms reproduce the model layer heights."""
    rec, static, stem = run
    io.write(stem, rec, static)
    st = xr.open_dataset(f"{stem}_stations.nc", decode_times=False)
    assert st.attrs["featureType"] == "timeSeriesProfile"
    assert st.attrs["Conventions"].startswith("CF-1.8")
    assert st["station_name"].attrs["cf_role"] == "timeseries_id"
    assert st["temp"].dims == ("station", "time", "s_rho")
    assert st["time"].attrs["units"].startswith("seconds since")
    assert json.loads(st.attrs["run_config"])["case"] == "bay"
    # CF ocean_sigma_coordinate: z = η + σ (h + η) must equal the model's −H + (H + η) σ_c
    terms = dict(zip(*[iter(st["s_rho"].attrs["formula_terms"].replace(":", "").split())] * 2))
    eta, s, h = st[terms["eta"]], st[terms["sigma"]], st[terms["depth"]]
    z_cf = (eta + s * (h + eta)).transpose("station", "time", "s_rho").values
    sc = 0.5 * (static["sigma_w"][1:] + static["sigma_w"][:-1])
    H, e = static["h_st"][:, None, None], rec["eta_st"].T[:, :, None]
    np.testing.assert_allclose(z_cf, -H + (H + e) * sc[None, None, :], atol=1e-12)
    f = xr.open_dataset(f"{stem}_fields.nc", decode_times=False)
    assert f["temp_section"].dims == ("time", "s_rho", "y")
    assert f["temp_z"].dims == ("time", "depth", "yb", "xb")
    assert f["depth"].attrs["positive"] == "down"
    assert f["temp_section"].encoding["zlib"] and f["temp_section"].dtype == np.float32
    st.close()
    f.close()


def test_legacy_npz_conversion(run, tmp_path):
    """A run saved by the old runner (.npz) converts to the same NetCDF records."""
    rec, static, stem = run
    old = tmp_path / "old.npz"
    np.savez(old, **rec, z_st=np.zeros((NK, NS)), stations=json.dumps(static["stations"]),
             box=json.dumps([list(b) for b in static["box"]]), depths=np.array(static["depths"]),
             sigma_w=static["sigma_w"], H_axis=static["h_axis"], meta_json=json.dumps(static["attrs"]))
    legacy = io.read_records(old)
    assert set(rec) <= set(legacy)
    io.write(stem, legacy, static)
    back = io.read_records(stem)
    np.testing.assert_array_equal(back["T_st"], rec["T_st"])
    np.testing.assert_allclose(back["w_axis"], rec["w_axis"], rtol=1e-6)


def test_truncate_for_restart(run):
    """Continuing from a restart keeps only the records up to the restart time."""
    rec, static, stem = run
    io.write(stem, rec, static)
    back = io.read_records(stem, t_max=2700.0)
    assert back["t"].tolist() == [900.0, 1800.0, 2700.0]
    assert back["w_z"].shape[0] == 3
