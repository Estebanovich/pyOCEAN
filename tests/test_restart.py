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


# ---------------------------------------------------------------- ALE (Plan D)
def _ale_model(rule="zstar", open_=False, refactor_iterations=None):
    from boundaries.obcs import OBData
    hg = create_cartesian_grid(nx=NX, ny=1, dx=1000.0, dy=1000.0, f0=1e-4)
    H = np.linspace(20.0, 8.0, NX)[None, :]
    params = dict(dz_top=np.full(NZ, 20.0 / NZ))
    if rule == "hybrid":
        params["rho_target"] = np.linspace(996.0, 999.5, NZ + 1)
    kw = dict(closure="constant", z0=None, nu_const=1e-4, ah=1.0, vertical_coordinate="ale", regrid=rule,
              regrid_params=params, rho_ref="auto")
    if open_:
        zw = np.linspace(0.0, -20.0, 5)[:, None]

        def state(t):                                                # exterior that varies in time
            a = np.sin(2 * np.pi * t / 3600.0)
            return {"west": OBData(side="west", T=np.full((4, 1), 12.0 + a), u_normal=np.full((4, 1), 0.01 * a),
                                   zw=zw, eta=np.full(1, 0.02 * a), ubar=np.full(1, 0.01 * a))}
        kw.update(open_boundaries=("west", "east"), sponge_cells=4, sponge_tau=(600.0, 60.0),
                  open_boundary_state=state)
    m = SigmaBaroclinic(hg, H, dt=20.0, n_layers=NZ, rho0=1000.0, t_coef=0.2, **kw)
    if refactor_iterations is not None:
        m.fs.refactor_iterations = refactor_iterations
    return m


def _ale_run(make, n_before, n_after, path):
    """(continuous n_before + n_after steps, restarted at n_before into a fresh model) final states."""
    m = make()
    st = _initial(m)
    for _ in range(n_before + n_after):
        st = m.step(st)
    m1 = make()
    st1 = _initial(m1)
    for _ in range(n_before):
        st1 = m1.step(st1)
    write_restart(path, m1, st1)
    m2 = make()
    st2 = read_restart(path, m2)
    for _ in range(n_after):
        st2 = m2.step(st2)
    return st, st2


@pytest.mark.parametrize("rule, open_", [("zstar", False), ("sigma", False), ("hybrid", True)])
def test_ale_restart_is_bit_for_bit(tmp_path, rule, open_):
    ref, st = _ale_run(lambda: _ale_model(rule, open_), 15, 25, tmp_path / "rst.nc")
    _assert_identical(st, ref)
    assert np.array_equal(st.h, ref.h)


def test_ale_restart_keeps_the_lagged_lu_preconditioner(tmp_path):
    # never refactorised after the first step: the restarted run must precondition CG with the same old
    # matrix, or CG stops at a different iterate within its tolerance
    ref, st = _ale_run(lambda: _ale_model("zstar", refactor_iterations=10**6), 15, 25, tmp_path / "rst.nc")
    _assert_identical(st, ref)
    assert np.array_equal(st.h, ref.h)


def test_ale_restart_refuses_other_regrid_parameters(tmp_path):
    m = _ale_model("hybrid")
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    m2 = _ale_model("hybrid")
    m2.regrid_params["rho_target"] = np.linspace(996.0, 999.6, NZ + 1)
    with pytest.raises(ValueError, match="rho_target"):
        read_restart(tmp_path / "rst.nc", m2)
    with pytest.raises(ValueError, match="regrid"):
        read_restart(tmp_path / "rst.nc", _ale_model("zstar"))


def test_restart_refuses_a_different_baroclinic_theta(tmp_path):
    m = _model()
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    with pytest.raises(ValueError, match="theta_bc"):
        read_restart(tmp_path / "rst.nc", _model(theta_bc=0.6))


def test_ale_restart_keeps_the_outflow_limiter_count(tmp_path):
    m = _ale_model("zstar")
    st = m.step(_initial(m))
    st.n_limited = 7
    write_restart(tmp_path / "rst.nc", m, st)
    assert read_restart(tmp_path / "rst.nc", _ale_model("zstar")).n_limited == 7


def test_ale_restart_refuses_another_thin_layer_coupling(tmp_path):
    m = _ale_model("zstar")
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    other = _ale_model("zstar")
    other.thin_layer_coupling = "mom6" if m.thin_layer_coupling == "none" else "none"
    with pytest.raises(ValueError, match="thin_layer_coupling"):
        read_restart(tmp_path / "rst.nc", other)


def test_ale_restart_refuses_another_thin_layer_cutoff(tmp_path):
    m = _ale_model("zstar")
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    other = _ale_model("zstar")
    other.thin_layer_cutoff = m.thin_layer_cutoff + 0.5
    with pytest.raises(ValueError, match="thin_layer_cutoff"):
        read_restart(tmp_path / "rst.nc", other)


def test_ale_restart_refuses_another_tracer_limiter(tmp_path):
    m = _ale_model("zstar")
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    other = _ale_model("zstar")
    other.tracer_limiter = "fct" if m.tracer_limiter == "none" else "none"
    with pytest.raises(ValueError, match="tracer_limiter"):
        read_restart(tmp_path / "rst.nc", other)
