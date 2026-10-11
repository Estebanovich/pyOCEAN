"""
Tests for diagnostics/output.py — CF-1.8 / SGRID model output in NetCDF4 or
Zarr (appendable, chunked; readable by xarray, Pangeo tools and
anemoi-datasets).
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from diagnostics.output import OutputWriter

NY, NX, NZ = 4, 6, 3


def _states(n=3):
    rng = np.random.default_rng(0)
    for k in range(n):
        yield 600.0 * (k + 1), dict(eta=rng.normal(size=(NY, NX)), u=rng.normal(size=(NZ, NY, NX + 1)),
                                    v=rng.normal(size=(NZ, NY + 1, NX)), T=rng.normal(size=(NZ, NY, NX)))


def _writer(path, fmt):
    hg = create_cartesian_grid(nx=NX, ny=NY, dx=1000.0, dy=1000.0, f0=1e-4)
    H = np.full((NY, NX), 20.0)
    H[0, 0] = np.nan
    return OutputWriter(path, hg, H, n_layers=NZ, start="2026-08-01T00:00:00", fmt=fmt,
                        title="test run")


@pytest.mark.parametrize("fmt,name", [("zarr", "out.zarr"), ("netcdf", "out.nc")])
def test_roundtrip_is_exact_and_time_decodes(tmp_path, fmt, name):
    w = _writer(tmp_path / name, fmt)
    written = []
    for t, f in _states():
        w.write(t, **f)
        written.append(f)
    w.close()
    ds = xr.open_zarr(tmp_path / name) if fmt == "zarr" else xr.open_dataset(tmp_path / name)
    assert ds.sizes["time"] == 3
    for k, f in enumerate(written):
        for n in ("eta", "u", "v", "T"):
            assert np.array_equal(ds[n].isel(time=k).values, f[n]), n
    assert str(ds.time.values[0])[:19] == "2026-08-01T00:10:00"


def test_cf_and_sgrid_metadata(tmp_path):
    w = _writer(tmp_path / "out.zarr", "zarr")
    t, f = next(_states(1))
    w.write(t, **f)
    w.close()
    ds = xr.open_zarr(tmp_path / "out.zarr")
    assert ds.attrs["Conventions"].startswith("CF-1.8")
    assert ds["eta"].attrs["standard_name"] == "sea_surface_height_above_geoid"
    assert ds["u"].attrs["standard_name"] == "sea_water_x_velocity" and ds["u"].dims == ("time", "s_rho", "yh", "xq")
    assert ds["v"].dims == ("time", "s_rho", "yq", "xh")
    assert ds["s_rho"].attrs["standard_name"] == "ocean_sigma_coordinate"
    assert ds["s_rho"].attrs["formula_terms"] == "sigma: s_rho eta: eta depth: h"
    topo = ds["grid"].attrs
    assert topo["cf_role"] == "grid_topology" and topo["topology_dimension"] == 2
    assert "xh: xq (padding: both)" in topo["face_dimensions"] or "xq" in topo["edge1_dimensions"]
    assert np.isnan(ds["h"].values[0, 0]) and ds["mask"].values[0, 0] == 0


def test_two_dimensional_fields_and_unknown_names(tmp_path):
    w = _writer(tmp_path / "out.nc", "netcdf")
    w.write(60.0, eta=np.zeros((NY, NX)), ubar=np.ones((NY, NX + 1)), vbar=np.ones((NY + 1, NX)))
    with pytest.raises(ValueError):
        w.write(120.0, eta=np.zeros((NY, NX)), mystery=np.zeros((7, 7)))
    w.close()
    ds = xr.open_dataset(tmp_path / "out.nc")
    assert ds["ubar"].dims == ("time", "yh", "xq") and ds["ubar"].attrs["units"] == "m s-1"


def test_stretched_sigma_coordinates_are_written(tmp_path):
    hg = create_cartesian_grid(nx=NX, ny=NY, dx=1000.0, dy=1000.0, f0=1e-4)
    w = OutputWriter(tmp_path / "out.nc", hg, np.full((NY, NX), 20.0), n_layers=3, fmt="netcdf",
                     dsigma=[0.5, 0.3, 0.2])                                # bed → surface
    w.write(60.0, eta=np.zeros((NY, NX)))
    w.close()
    ds = xr.open_dataset(tmp_path / "out.nc")
    np.testing.assert_allclose(ds["s_w"].values, [-1.0, -0.5, -0.2, 0.0])
    np.testing.assert_allclose(ds["s_rho"].values, [-0.75, -0.35, -0.1])
