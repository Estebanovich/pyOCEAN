"""
TEOS-10 in the 3-D core, Task 8 (plan docs/superpowers/plans/2026-10-31-teos10-3d.md; spec goal G, criterion 10):
inputs (``ts_kind`` of ``initial_state``, the open-boundary exterior, the interior sponge, ``forcing.ocean_init``),
outputs (CF names, ``theta`` and ``SP``) and the T-S diagram.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from boundaries.obcs import OBData
from core.components.eos import make_eos
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError, PyOceanConfigWarning
from core.restart import write_restart
from diagnostics.output import OutputWriter
from diagnostics.ts_diagram import plot_ts, ts_histogram
from physics.eos.convert import SR_OVER_SP, from_model_ts, sa_to_sp, sp_to_sa, theta_to_ct, to_model_ts

NX, NY, NZ = 8, 6, 10
DX, DT = 2000.0, 60.0
H = (40.0 + 5.0 * np.arange(NX))[None, :] * np.ones((NY, 1))
HMAX = float(H.max())


def _grid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DX, f0=1e-4)


def _ts():
    k = np.arange(NZ)[:, None, None] / (NZ - 1)
    i = np.arange(NX)[None, None, :] / NX
    return ((10.0 + 4.0 * k + 1.0 * i) * np.ones((NZ, NY, NX)), (35.2 - 0.3 * k - 0.4 * i) * np.ones((NZ, NY, NX)))


def _model(eos="teos10", **kw):
    base = dict(z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6, d_min=0.05, salinity=True, eos=eos,
                rho_ref="auto", vertical_coordinate="ale", pgf="fv", regrid="zstar",
                regrid_params=dict(dz_top=np.full(NZ, HMAX / NZ)))
    if eos == "linear":
        base.update(s_coef=0.78, s_ref=35.0, t_ref=10.0)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        return SigmaBaroclinic(_grid(), H, dt=DT, n_layers=NZ, **base)


# ---------------------------------------------------------------- initial_state
def test_initial_state_converts_once():
    """θ, S_P given to a teos10 model arrive as Θ = CT_from_pt(S_A, θ), S_A = (35.16504/35) S_P; ct_sa is untouched."""
    T, S = _ts()
    m = _model("teos10")
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)                       # default ts_kind="theta_sp"
    assert np.array_equal(m.T_ext, st.T) and np.array_equal(m.S_ext, st.S)
    # the layer-mean constants: exact on a uniform column, conversion applied once (not twice)
    Tu, Su = np.full((NZ, NY, NX), 12.0), np.full((NZ, NY, NX), 35.0)
    st = once = m.initial_state(np.zeros((NY, NX)), Tu, S=Su)
    assert np.allclose(st.S, SR_OVER_SP * 35.0, rtol=0, atol=1e-12)
    assert np.allclose(st.T, theta_to_ct(SR_OVER_SP * 35.0, 12.0), rtol=0, atol=1e-12)
    # ts_kind="ct_sa": taken as is (a uniform column survives the start-up exactly)
    st = m.initial_state(np.zeros((NY, NX)), Tu, S=Su, ts_kind="ct_sa")
    assert np.allclose(st.S, 35.0, rtol=0, atol=1e-12) and np.allclose(st.T, 12.0, rtol=0, atol=1e-12)
    # data already in Θ/S_A with the default ts_kind are converted twice (the documented pitfall, pinned here)
    twice = m.initial_state(np.zeros((NY, NX)), once.T, S=once.S)
    assert np.allclose(twice.S, SR_OVER_SP * 35.0 * SR_OVER_SP, rtol=0, atol=1e-10)
    assert "exactly once" in SigmaBaroclinic.initial_state.__doc__ and "already" in SigmaBaroclinic.initial_state.__doc__


def test_initial_state_needs_s_to_convert_and_linear_untouched():
    T, S = _ts()
    m = _model("linear")
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    assert np.array_equal(st.S, S)                                         # theta_sp on a θ/S_P EOS: no conversion
    with pytest.raises(ValueError, match="needs S"):
        _model("linear", salinity=False).initial_state(np.zeros((NY, NX)), T, ts_kind="ct_sa")
    with pytest.raises(ConfigurationError, match="ts_kind"):
        _model("teos10", ts_kind="kelvin")


# ---------------------------------------------------------------- open boundary and sponge
def _ob(side="west", n=NY, nz=3, T0=12.0, S0=35.0):
    zw = -np.linspace(0.0, HMAX, nz + 1)[:, None] * np.ones((1, n))
    return OBData(side=side, T=np.full((nz, n), T0), S=np.full((nz, n), S0), u_normal=np.zeros((nz, n)), zw=zw,
                  eta=np.zeros(n))


def test_obc_and_sponge_convert():
    exterior = lambda t: {"west": _ob(T0=12.0 + t / 3600.0)}                # noqa: E731
    m = _model("teos10", open_boundaries=("west",), open_boundary_state=exterior, ts_kind="theta_sp")
    for t in (0.0, 7200.0):                                                 # converted at every call
        got = m.obc_state(t)["west"]
        SA = sp_to_sa(35.0)
        assert np.allclose(got.S, SA, rtol=0, atol=1e-12)
        assert np.allclose(got.T, theta_to_ct(SA, 12.0 + t / 3600.0), rtol=0, atol=1e-12)
    # ct_sa and a θ/S_P EOS: the provider is not even wrapped
    m = _model("teos10", open_boundaries=("west",), open_boundary_state=exterior, ts_kind="ct_sa")
    assert m.obc_state is exterior
    # reverse: Θ, S_A data into a wright_full (θ, S_P) model
    m = _model("wright_full", open_boundaries=("west",), open_boundary_state=exterior, ts_kind="ct_sa")
    ref = to_model_ts(np.full((3, NY), 12.0), np.full((3, NY), 35.0), "theta_sp", "ct_sa")
    got = m.obc_state(0.0)["west"]
    assert np.array_equal(got.T, ref[0]) and np.array_equal(got.S, ref[1])
    # T without S cannot be converted
    bad = lambda t: {"west": OBData(side="west", T=np.full((3, NY), 12.0), u_normal=np.zeros((3, NY)),  # noqa: E731
                                    zw=_ob().zw)}
    m = _model("teos10", open_boundaries=("west",), open_boundary_state=bad)
    with pytest.raises(ValueError, match="needs both"):
        m.obc_state(0.0)

    # interior sponge profiles
    zw = -np.array([0.0, 20.0, 40.0, HMAX])
    Ts, Ss = np.array([14.0, 12.0, 10.0]), np.array([35.0, 35.1, 35.2])
    rate = np.full((NY, NX), 1e-5)
    m = _model("teos10", sponge_rate=rate, sponge_T=Ts, sponge_zw=zw, sponge_S=Ss)
    SA = sp_to_sa(Ss)
    assert np.allclose(m._sp_S, SA, rtol=0, atol=1e-12) and np.allclose(m._sp_T, theta_to_ct(SA, Ts), rtol=0, atol=1e-12)
    m = _model("teos10", sponge_rate=rate, sponge_T=Ss * 0 + 3.0, sponge_zw=zw, sponge_S=Ss, ts_kind="ct_sa")
    assert np.array_equal(m._sp_S, Ss) and np.allclose(m._sp_T, 3.0)
    with pytest.raises(ConfigurationError, match="sponge_S"):
        _model("teos10", sponge_rate=rate, sponge_T=Ts, sponge_zw=zw)


# ---------------------------------------------------------------- forcing.ocean_init
def test_ocean_init_converts_once():
    from forcing.ocean_init import boundary_data
    time = np.array(["2020-01-01"], dtype="datetime64[ns]")
    zl = np.array([5.0, 20.0])
    dims = ("time", "k", "j_t", "i_t")
    T = np.full((1, 2, 3, 4), 12.0)
    S = np.full((1, 2, 3, 4), 35.0)
    prepared = {"t": xr.Dataset({"temp": (dims, T), "salt": (dims, S)}, coords={"time": time})}
    kw = dict(prepared=prepared, side="west", t=time[0], remove_mean_ssh=False)
    plain = boundary_data(**kw)
    assert np.array_equal(plain.T, T[0, :, :, 0]) and np.array_equal(plain.S, S[0, :, :, 0])
    conv = boundary_data(**kw, eos_ts_kind="ct_sa")
    assert np.allclose(conv.S, sp_to_sa(35.0), rtol=0, atol=1e-12)
    assert np.allclose(conv.T, theta_to_ct(sp_to_sa(35.0), 12.0), rtol=0, atol=1e-12)
    same = boundary_data(**kw, eos_ts_kind="ct_sa", ts_kind="ct_sa")
    assert np.array_equal(same.T, T[0, :, :, 0])


# ---------------------------------------------------------------- output
def _writer(path, eos):
    hg = _grid()
    return OutputWriter(path, hg, np.full((NY, NX), 50.0), n_layers=NZ, start="2026-08-01T00:00:00", fmt="netcdf",
                        eos=eos)


def test_output_theta_sp(tmp_path):
    T, S = _ts()
    Th, SA = theta_to_ct(sp_to_sa(S), T), sp_to_sa(S)                        # Θ, S_A as the teos10 model carries them
    w = _writer(tmp_path / "t.nc", "teos10")
    w.write(60.0, T=Th, S=SA)
    w.close()
    ds = xr.open_dataset(tmp_path / "t.nc")
    assert ds["T"].attrs["standard_name"] == "sea_water_conservative_temperature"
    assert ds["S"].attrs["standard_name"] == "sea_water_absolute_salinity" and ds["S"].attrs["units"] == "g kg-1"
    assert ds["theta"].attrs["standard_name"] == "sea_water_potential_temperature"
    assert ds["SP"].attrs["standard_name"] == "sea_water_practical_salinity"
    # criterion 10: θ/S_P → Θ/S_A → θ/S_P is the identity to round-off
    assert np.allclose(ds["theta"].values[0], T, rtol=0, atol=1e-9)
    assert np.allclose(ds["SP"].values[0], S, rtol=0, atol=1e-12)
    assert np.array_equal(ds["T"].values[0], Th)                             # the prognostic T is written untouched


def test_output_other_eos_unchanged(tmp_path):
    T, S = _ts()
    for eos in ("linear", "wright_full"):
        w = _writer(tmp_path / f"{eos}.nc", eos)
        w.write(60.0, T=T, S=S)
        w.close()
        ds = xr.open_dataset(tmp_path / f"{eos}.nc")
        assert "theta" not in ds and "SP" not in ds
        assert ds["T"].attrs["standard_name"] == "sea_water_potential_temperature"
    assert xr.open_dataset(tmp_path / "linear.nc")["S"].attrs["standard_name"] == "sea_water_salinity"
    assert xr.open_dataset(tmp_path / "wright_full.nc")["S"].attrs["standard_name"] == "sea_water_practical_salinity"
    with pytest.raises(ValueError, match="unknown eos"):
        _writer(tmp_path / "x.nc", "foo")


def test_output_eos_from_model(tmp_path):
    """Final review I2: a writer given the model takes its EOS, so a teos10 run cannot be written with θ labels."""
    T, S = _ts()
    m = _model("teos10")
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    w = OutputWriter(tmp_path / "m.nc", _grid(), np.full((NY, NX), 50.0), n_layers=NZ, fmt="netcdf", model=m)
    w.write(60.0, T=st.T, S=st.S)
    w.close()
    ds = xr.open_dataset(tmp_path / "m.nc")
    assert ds["T"].attrs["standard_name"] == "sea_water_conservative_temperature"
    assert ds["S"].attrs["standard_name"] == "sea_water_absolute_salinity"
    assert "theta" in ds and "SP" in ds
    with pytest.raises(ValueError, match="eos"):                             # contradicting the model is refused
        OutputWriter(tmp_path / "x.nc", _grid(), np.full((NY, NX), 50.0), n_layers=NZ, fmt="netcdf", model=m,
                     eos="linear")
    lin = _model("linear")                                                   # linear: byte-identical to eos default
    st = lin.initial_state(np.zeros((NY, NX)), T, S=S)
    for name, kw in (("a.nc", dict(model=lin)), ("b.nc", {})):
        w = OutputWriter(tmp_path / name, _grid(), np.full((NY, NX), 50.0), n_layers=NZ, fmt="netcdf", **kw)
        w.write(60.0, T=st.T, S=st.S)
        w.close()
    assert (tmp_path / "a.nc").read_bytes() == (tmp_path / "b.nc").read_bytes()


def test_restart_attributes_follow_eos(tmp_path):
    T, S = _ts()
    for eos, name in (("teos10", "sea_water_conservative_temperature"), ("linear", None)):
        m = _model(eos)
        st = m.initial_state(np.zeros((NY, NX)), T, S=S)
        write_restart(tmp_path / f"{eos}.nc", m, st)
        ds = xr.open_dataset(tmp_path / f"{eos}.nc")
        assert ds["T"].attrs.get("standard_name") == name


# ---------------------------------------------------------------- T-S diagram
def test_ts_histogram_volume():
    rng = np.random.default_rng(1)
    T, S = rng.uniform(2, 20, (4, 5, 6)), rng.uniform(34, 36, (4, 5, 6))
    V = rng.uniform(1.0, 2.0, T.shape)
    V[0, 0, 0] = np.nan                                                      # a dry/land cell is dropped
    hist = ts_histogram(T, S, V, np.linspace(0, 25, 26), np.linspace(33, 37, 17))
    assert hist.shape == (25, 16) and np.isclose(hist.sum(), np.nansum(V), rtol=1e-14)
    T_out = T.copy()
    T_out[1, 1, 1] = 100.0                                                   # outside the bins: not counted
    assert ts_histogram(T_out, S, V, np.linspace(0, 25, 26), np.linspace(33, 37, 17)).sum() < np.nansum(V)


def test_plot_ts_runs():
    import matplotlib
    matplotlib.use("Agg")
    T, S = _ts()
    eos = make_eos("teos10")
    Th, SA = theta_to_ct(sp_to_sa(S), T), sp_to_sa(S)
    ax = plot_ts(Th, SA, np.ones_like(T), eos, p_ref=2e7)
    assert ax.get_xlabel().startswith("S_A") and len(ax.collections) >= 2
    import matplotlib.pyplot as plt
    plt.close("all")
