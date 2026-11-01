"""
TEOS-10 in the 3-D core, Task 4 (plan docs/superpowers/plans/2026-10-31-teos10-3d.md; spec
docs/superpowers/specs/2026-10-31-teos10-3d-design.md goals B, E, G, H): the equation of state in the model —
constructor, pressure gradient, ``rho_ref="auto"`` T/S profiles, K_h on T − T_ref and S − S_ref, restart,
pre-flight and the registry.  The linear EOS stays bit for bit (``tests/test_n1_golden.py`` and the smoke hashes of
``scripts/n1_smoke.py``).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, TSReferenceProfile
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError, PyOceanConfigWarning
from core.model.registry import REGISTRY, variable_specs
from core.restart import read_restart, write_restart
from physics.eos.convert import pressure_from_z

NX, NY, NZ = 8, 6, 10
DX, DT = 2000.0, 60.0
H = (40.0 + 5.0 * np.arange(NX))[None, :] * np.ones((NY, 1))         # 40 … 75 m, sloping in x
HMAX = float(H.max())


def _grid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DX, f0=1e-4)


def _front(nz: int = NZ) -> tuple[np.ndarray, np.ndarray]:
    """T and S with a stable stratification (warm and fresh on top) and a front in x."""
    k = np.arange(nz)[:, None, None] / (nz - 1)
    i = np.arange(NX)[None, None, :] / NX
    T = (10.0 + 4.0 * k + 1.0 * i) * np.ones((nz, NY, NX))
    S = (35.2 - 0.3 * k - 0.4 * i) * np.ones((nz, NY, NX))
    return T, S


def _params(coord: str, eos: str) -> dict:
    if coord == "sigma":
        return dict(pgf="jacobian")
    base = dict(vertical_coordinate="ale", pgf="fv", regrid=coord)
    if coord == "zstar":
        base["regrid_params"] = dict(dz_top=np.full(NZ, HMAX / NZ))
    else:                       # hybrid: targets inside the σ₂ range the pre-flight accepts with a nonlinear EOS
        base["regrid_params"] = dict(dz_top=np.full(NZ, HMAX / NZ), rho_target=np.linspace(1025.8, 1027.6, NZ + 1))
    return base


def _model(coord: str = "zstar", eos: str = "teos10", **kw) -> SigmaBaroclinic:
    base = dict(z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6, d_min=0.05, salinity=True,
                eos=eos, rho_ref="auto")
    if eos == "linear":
        base.update(s_coef=0.78, s_ref=35.0, t_ref=10.0)
    base.update(_params(coord, eos))
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        return SigmaBaroclinic(_grid(), H, dt=DT, n_layers=NZ, **base)


# ---------------------------------------------------------------- construction and steps
@pytest.mark.parametrize("eos", ["wright_full", "teos10"])
@pytest.mark.parametrize("coord", ["sigma", "zstar", "hybrid"])
def test_eos_choice_builds(eos, coord):
    m = _model(coord, eos)
    assert m.eos_name == eos and m._eos.name == eos and not m._eos.linear
    assert m.rho0 == 1035.0 and m.cp == 3991.86795711963 and m.p_ref == 2e7 and m.rho_pgf_ref == 1035.0
    T, S = _front()
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    assert isinstance(m.rho_ref, TSReferenceProfile) and m._ts_ref is not None
    for _ in range(10):
        st = m.step(st)
    for name in ("eta", "u", "v", "T", "S"):
        assert np.isfinite(getattr(st, name)).all(), name
    assert np.abs(st.u).max() > 0                                     # the front drives a flow


def test_linear_defaults_unchanged():
    m = _model("zstar", "linear", rho_ref=None)
    assert m.eos_name == "linear" and m._eos.linear and m.rho0 == 1000.0 and m.cp == 3985.0
    assert m._ts_ref is None


def test_layer_density_pressure():
    """Driver._layer_density: linear = eos.density(T, S); nonlinear: ρ at p = g ρ₀ (Σ h above + ½ h_k)."""
    T, S = _front()
    h = np.full((NZ, NY, NX), 5.0)
    ml = _model("zstar", "linear")
    assert np.array_equal(ml._driver._layer_density(T, S, h), ml._eos.density(T, S))
    m = _model("zstar", "teos10")
    depth = (NZ - np.arange(NZ) - 0.5)[:, None, None] * 5.0          # centre depths, k = 0 at the bed
    expected = m._eos.density(T, S, m.g * m.rho0 * depth * np.ones((1, NY, NX)))
    np.testing.assert_allclose(m._driver._layer_density(T, S, lambda: h), expected, rtol=1e-15)


def test_reference_profile_from_mean_ts():
    """rho_ref='auto': ρ_ref(z) = ρ(T_ref, S_ref, −g ρ₀ z) of the horizontal-mean initial profiles."""
    m = _model("sigma", "wright_full")
    T, S = _front()
    m.initial_state(np.zeros((NY, NX)), T, S=S)
    z, Tr, Sr = m._ts_ref
    assert np.all(np.diff(z) > 0) and z[0] >= -HMAX and z[-1] <= 0.0
    zz = np.array([-30.0, -10.0])
    Ti, Si = np.interp(zz, z, Tr), np.interp(zz, z, Sr)
    np.testing.assert_allclose(m.rho_ref(zz), m._eos.density(Ti, Si, pressure_from_z(zz, m.rho0, m.g)), rtol=1e-15)


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
def test_profile_diffusion_keeps_reference(coord):
    """
    A resting state on the T/S reference profile, sloping layers and K_h > 0: the force and the K_h of
    T − T_ref(z), S − S_ref(z) vanish, so nothing changes in 20 steps (T and S linear in z, so the binned profile
    is exact to round-off).
    """
    m = _model(coord, "teos10", kh=50.0)
    z = m.layer_heights(np.zeros((NY, NX)))
    T, S = 12.0 + 0.05 * z, 35.0 - 0.004 * z
    st0 = m.initial_state(np.zeros((NY, NX)), T, S=S)
    st = st0
    for _ in range(20):
        st = m.step(st)
    assert np.abs(st.T - st0.T).max() <= 1e-13
    assert np.abs(st.S - st0.S).max() <= 1e-13


def test_profile_diffusion_acts_on_anomaly():
    """With a front, K_h diffuses T − T_ref(z) (≠ the field): the profile mode differs from the field mode."""
    m = _model("sigma", "teos10", kh=50.0)
    T, S = _front()
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    V = m.layer_volumes(st.eta)
    dif = m._tracer_diffusion
    a = dif.diffuse_h(st.T, V, st.eta, anomaly="T")
    b = dif.diffuse_h(st.T, V, st.eta, anomaly=False)
    Tref, _ = m.rho_ref.ts(m.layer_heights(st.eta))
    np.testing.assert_allclose(a, Tref + dif.diffuse_h(st.T - Tref, V, st.eta, anomaly=False), rtol=0, atol=1e-12)
    assert np.abs(a - b).max() > 1e-9


def test_mass_weight_options():
    T, S = _front()
    out = {}
    for mw in ("vanished", "full", "none"):
        m = _model("zstar", "teos10", pgf_mass_weight=mw)
        st = m.initial_state(np.zeros((NY, NX)), T, S=S)
        out[mw] = m.pressure_gradient(st.eta, st.T, st.h, st.S)[0]
        assert np.isfinite(out[mw]).all()
    ml = _model("zstar", "linear", pgf_mass_weight="none")
    st = ml.initial_state(np.zeros((NY, NX)), T, S=S)
    assert np.isfinite(ml.pressure_gradient(st.eta, st.T, st.h, st.S)[0]).all()


# ---------------------------------------------------------------- pre-flight
def test_preflight_errors_together():
    with pytest.raises(ConfigurationError) as e:
        _model("zstar", "foo")
    assert "unknown eos 'foo'" in str(e.value)
    with pytest.raises(ConfigurationError) as e:
        _model("zstar", "wright_full", salinity=False, t_coef=0.3, rho_ref=lambda z: 1025.0 + 0 * z,
               pgf_mass_weight="bogus")
    msg = str(e.value)
    for part in ("needs salinity=True", "t_coef=0.3 does not apply", "rho_ref as a function",
                 "unknown pgf_mass_weight"):
        assert part in msg, part
    with pytest.raises(ConfigurationError, match="needs a nonlinear eos"):
        _model("zstar", "linear", pgf_mass_weight="full")
    with pytest.raises(ConfigurationError, match="rho_target"):
        _model("hybrid", "teos10", regrid_params=dict(dz_top=np.full(NZ, HMAX / NZ),
                                                      rho_target=np.linspace(25.8, 27.6, NZ + 1)))


def test_preflight_warnings():
    kw = dict(z0=None, closure="constant", d_min=0.05, salinity=True, eos="teos10")
    with pytest.warns(PyOceanConfigWarning, match="compressibility"):
        SigmaBaroclinic(_grid(), H, dt=DT, n_layers=NZ, **kw)
    with pytest.warns(PyOceanConfigWarning, match="outside 1000–1050"):
        SigmaBaroclinic(_grid(), H, dt=DT, n_layers=NZ, rho0=1070.0, rho_ref="auto", **kw)


# ---------------------------------------------------------------- restart
def test_restart_eos_mismatch(tmp_path):
    """Review Focus 5: a teos10 restart loaded into a wright_full (or linear) model raises, naming the EOS."""
    m = _model("zstar", "teos10")
    T, S = _front()
    st = m.step(m.initial_state(np.zeros((NY, NX)), T, S=S))
    path = tmp_path / "teos10.nc"
    write_restart(path, m, st)
    with pytest.raises(ValueError, match="eos"):
        read_restart(path, _model("zstar", "wright_full"))
    with pytest.raises(ValueError, match="eos"):
        read_restart(path, _model("zstar", "linear"))


@pytest.mark.parametrize("coord", ["sigma", "hybrid"])
def test_restart_nonlinear_bit_for_bit(tmp_path, coord):
    """The T/S profile and the EOS configuration go through the restart: 2 + 2 steps = 4 steps, bit for bit."""
    T, S = _front()
    m = _model(coord, "teos10", kh=20.0)
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    for _ in range(2):
        st = m.step(st)
    write_restart(tmp_path / "r.nc", m, st)
    ref = st
    for _ in range(2):
        ref = m.step(ref)
    m2 = _model(coord, "teos10", kh=20.0)
    st2 = read_restart(tmp_path / "r.nc", m2)
    assert np.array_equal(m2._ts_ref[1], m._ts_ref[1]) and np.array_equal(m2._ts_ref[2], m._ts_ref[2])
    for _ in range(2):
        st2 = m2.step(st2)
    for name in ("eta", "u", "v", "T", "S"):
        assert np.array_equal(getattr(st2, name), getattr(ref, name)), name


def test_linear_restart_without_eos_attributes_loads(tmp_path):
    """Files written before this change (no config_eos, config_cp, config_p_ref) still load into a linear model."""
    import xarray as xr
    m = _model("zstar", "linear")
    T, S = _front()
    st = m.step(m.initial_state(np.zeros((NY, NX)), T, S=S))
    path = tmp_path / "old.nc"
    write_restart(path, m, st)
    with xr.open_dataset(path) as ds:
        ds = ds.load()
    for k in ("config_eos", "config_cp", "config_p_ref", "config_pgf_mass_weight"):
        del ds.attrs[k]
    old = tmp_path / "old_stripped.nc"
    ds.to_netcdf(old, engine="netcdf4", format="NETCDF4")
    st2 = read_restart(old, _model("zstar", "linear"))
    assert np.array_equal(st2.T, st.T)


# ---------------------------------------------------------------- registry
def test_cf_names_teos10():
    reg = variable_specs("teos10")
    assert reg["T"].standard_name == "sea_water_conservative_temperature"
    assert reg["S"].standard_name == "sea_water_absolute_salinity" and reg["S"].units == "g kg-1"
    assert variable_specs("wright_full")["S"].standard_name == "sea_water_practical_salinity"
    assert variable_specs("linear") == REGISTRY
    assert REGISTRY["T"].standard_name == "sea_water_potential_temperature"      # the static registry is unchanged
    with pytest.raises(ValueError, match="unknown eos"):
        variable_specs("foo")
