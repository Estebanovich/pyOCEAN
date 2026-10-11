"""
Tests for core/restart.py — bit-for-bit restarts of the σ-layer models.

A run interrupted, written to disk, read into a *new* model instance and
continued must give arrays identical (np.array_equal, not allclose) to the
uninterrupted run.  Restart reproducibility repeatedly breaks in community
models (MOM6 #975/#208/#508, SCHISM #140) because of time-stepping history,
hidden model state (turbulence) or precision loss in the file.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.restart import read_restart, write_restart

NX, NZ = 24, 8


def _model(closure="gls", dt=20.0, **kw):
    hg = create_cartesian_grid(nx=NX, ny=1, dx=1000.0, dy=1000.0, f0=1e-4)
    H = np.linspace(20.0, 8.0, NX)[None, :]
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=NZ, rho0=1000.0, t_coef=0.2, z0=1e-3, closure=closure,
                           nu_const=1e-4, ah=1.0, **kw)


def _initial(m):
    x = np.arange(NX)
    T = np.where(x < NX // 2, 10.0, 15.0)[None, None, :].repeat(NZ, 0) + np.linspace(0, 2, NZ)[:, None, None]
    return m.initial_state(0.1 * np.cos(np.pi * x / NX)[None, :], T)


def _assert_identical(a, b):
    for name in ("eta", "u", "v", "T"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    assert a.time == b.time


@pytest.mark.parametrize("closure", ["gls", "constant"])
def test_restart_is_bit_for_bit(tmp_path, closure):
    m = _model(closure)
    st = _initial(m)
    for _ in range(40):
        st = m.step(st)
    ref, ref_gls = st, (m.core.gls.tke.copy() if m.core.gls is not None else None)

    m1 = _model(closure)
    st1 = _initial(m1)
    for _ in range(15):
        st1 = m1.step(st1)
    write_restart(tmp_path / "rst.nc", m1, st1)

    m2 = _model(closure)                                     # fresh instance: no hidden state carried over
    st2 = read_restart(tmp_path / "rst.nc", m2)
    for _ in range(25):
        st2 = m2.step(st2)
    _assert_identical(st2, ref)
    if ref_gls is not None:
        assert np.array_equal(m2.core.gls.tke, ref_gls)


def test_restart_covers_the_ab3_start_up(tmp_path):
    # restarting after 1 step (AB history incomplete) must also reproduce the continuous run
    m = _model()
    st = _initial(m)
    for _ in range(6):
        st = m.step(st)
    m1 = _model()
    st1 = m1.step(_initial(m1))
    write_restart(tmp_path / "rst.nc", m1, st1)
    m2 = _model()
    st2 = read_restart(tmp_path / "rst.nc", m2)
    for _ in range(5):
        st2 = m2.step(st2)
    _assert_identical(st2, st)


def test_restart_refuses_an_incompatible_model(tmp_path):
    m = _model()
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    with pytest.raises(ValueError, match="dt"):
        read_restart(tmp_path / "rst.nc", _model(dt=10.0))
    with pytest.raises(ValueError, match="closure"):
        read_restart(tmp_path / "rst.nc", _model(closure="constant"))


def test_restart_file_is_cf_netcdf_with_float64(tmp_path):
    import xarray as xr
    m = _model()
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    ds = xr.open_dataset(tmp_path / "rst.nc")
    assert ds["u"].dtype == np.float64 and ds["T"].dims == ("layer", "y", "x")
    assert "units" in ds["eta"].attrs and ds.attrs["model"] == "SigmaBaroclinic"


def _open_kw():
    from core.grid.sigma_levels import stretched_sigma
    return dict(closure="constant", open_boundaries=("west", "east"), sponge_cells=4, sponge_tau=(600.0, 60.0),
                dsigma=stretched_sigma(NZ, 3.0, 0.5), rho_ref="auto")


def test_restart_with_open_boundaries_stretched_layers_and_auto_reference_profile(tmp_path):
    # hidden state beyond the AB history: the external (inflow / sponge) tracer and the reference
    # density profile built by initial_state must be restored too
    m = _model(**_open_kw())
    st = _initial(m)
    for _ in range(40):
        st = m.step(st)
    m1 = _model(**_open_kw())
    st1 = _initial(m1)
    for _ in range(15):
        st1 = m1.step(st1)
    write_restart(tmp_path / "rst.nc", m1, st1)
    m2 = _model(**_open_kw())
    st2 = read_restart(tmp_path / "rst.nc", m2)
    for _ in range(25):
        st2 = m2.step(st2)
    _assert_identical(st2, st)


def test_restart_refuses_other_layers_or_boundaries(tmp_path):
    m = _model(**_open_kw())
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    kw = _open_kw()
    kw.pop("dsigma")
    with pytest.raises(ValueError, match="sigma"):
        read_restart(tmp_path / "rst.nc", _model(**kw))
    kw = _open_kw()
    kw["open_boundaries"] = ("west",)
    with pytest.raises(ValueError, match="open_boundaries"):
        read_restart(tmp_path / "rst.nc", _model(**kw))
