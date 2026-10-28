"""
N2a (plan docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md): salinity in the 3-D core.

Task 1: S in the state, the linear EOS in (T, S), restart and pre-flight.
EOS: ρ = ρ₀ − α (T − T₀) + β (S − S₀)  (Vallis 2006, eq. 1.156), S in g kg⁻¹, β = ``s_coef``.
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from core.components.eos import LinearEOS
from core.model.contract import ConfigurationError, PyOceanConfigWarning
from core.model.registry import REGISTRY
from core.restart import read_restart, write_restart
from tests import n1_cases as nc

GOLDEN = json.loads((Path(__file__).parent / "data" / "n1_golden" / "hashes.json").read_text())


def _salty_zstar():
    """``zstar_vi_arakawa`` of the golden matrix with salinity on: β = 0.78, S₀ = 30, S ≡ 30."""
    m = nc._ale("zstar", salinity=True, s_coef=0.78, s_ref=30.0)
    return nc._start(m, True, S=np.full((nc.NZ, nc.NY, nc.NX), 30.0))


# ---------------------------------------------------------------- equation of state
def test_linear_eos_without_S_is_bitwise_today():
    T = nc._T()
    eos = LinearEOS(rho0=1000.0, t_coef=0.2, t_ref=10.0)
    assert np.array_equal(eos.density(T), 1000.0 - 0.2 * (T - 10.0))


def test_linear_eos_with_S():
    eos = LinearEOS(rho0=1000.0, t_coef=0.2, t_ref=10.0, s_coef=0.78, s_ref=30.0)
    T = np.array([10.0, 12.0, 5.0])
    S = np.array([30.0, 31.0, 0.0])
    expected = np.array([1000.0, 1000.0 - 0.4 + 0.78, 1000.0 + 1.0 - 0.78 * 30.0])
    np.testing.assert_allclose(eos.density(T, S), expected, rtol=1e-15, atol=1e-12)


def test_model_density_with_and_without_S():
    m = nc._ale("zstar", salinity=True, s_coef=0.78, s_ref=30.0)
    T, S = nc._T(), np.full((nc.NZ, nc.NY, nc.NX), 31.0)
    assert np.array_equal(m.density(T), nc.RHO0 - nc.TCOEF * (T - 0.0))
    assert np.array_equal(m.density(T, S), nc.RHO0 - nc.TCOEF * (T - 0.0) + 0.78 * (S - 30.0))


def test_s_coef_s_ref_fixed_after_construction():
    m = nc._ale("zstar", salinity=True, s_coef=0.78, s_ref=30.0)
    assert (m.s_coef, m.s_ref, m.salinity) == (0.78, 30.0, True)
    for name in ("s_coef", "s_ref"):
        with pytest.raises(AttributeError, match="fixed after construction"):
            setattr(m, name, 1.0)


# ---------------------------------------------------------------- state and registry
def test_salinity_needs_initial_S():
    m = nc._ale("zstar", salinity=True, s_coef=0.78, s_ref=30.0)
    with pytest.raises(ConfigurationError, match="salinity=True needs S in initial_state"):
        m.initial_state(nc._eta(), nc._T())


def test_S_without_salinity_is_an_error():
    m = nc._ale("zstar")
    with pytest.raises(ConfigurationError, match="salinity"):
        m.initial_state(nc._eta(), nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 30.0))


def test_initial_state_carries_S():
    m, st, _ = _salty_zstar()
    assert st.S.shape == (nc.NZ, nc.NY, nc.NX) and np.all(st.S == 30.0)
    ms = nc._sigma(salinity=True, s_coef=0.78, s_ref=30.0)
    S = 30.0 + np.arange(nc.NZ)[:, None, None] * np.ones((nc.NZ, nc.NY, nc.NX))
    sts = ms.initial_state(nc._eta(), nc._T(), S=S)
    assert np.array_equal(sts.S, S) and sts.S is not S


def test_initial_state_remaps_S_from_source_like_T():
    # hybrid start from 1 m source levels: S given as an affine function of T is remapped exactly like T
    h_src, T_src = nc._hybrid_src()
    S_src = 2.0 * T_src + 5.0
    m = nc._ale("hybrid", nz=8, salinity=True)
    st = m.initial_state(nc._eta(), T_src, h_src=h_src, S=S_src)
    wet = m.fs.active[None] & (st.h > 0)
    np.testing.assert_allclose(st.S[wet], (2.0 * st.T + 5.0)[wet], rtol=1e-12, atol=1e-11)


def test_registry_has_S():
    s = REGISTRY["S"]
    assert (s.dims, s.location, s.units, s.standard_name, s.kind) == (
        ("k", "y", "x"), "t", "1e-3", "sea_water_salinity", "prognostic")


# ---------------------------------------------------------------- pre-flight
def test_s_coef_without_salinity_warns():
    with pytest.warns(PyOceanConfigWarning, match="s_coef"):
        nc._ale("zstar", s_coef=0.78)


@pytest.mark.parametrize("arg", ["fw", "variance_budget", "tef_face"])
def test_salinity_only_arguments_need_salinity(arg):
    from core.model.preflight import preflight
    import inspect
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    a = {n: p.default for n, p in inspect.signature(SigmaBaroclinic.__init__).parameters.items() if n != "self"}
    a[arg] = np.zeros((nc.NY, nc.NX)) if arg == "fw" else (True if arg == "variance_budget" else 3)
    issues = [i for i in preflight(a, frozenset({arg})) if i.level == "error"]
    assert any(arg in i.message and "salinity=True" in i.message for i in issues)
    a["salinity"] = True
    assert not [i for i in preflight(a, frozenset({arg, "salinity"})) if i.level == "error"]


# ---------------------------------------------------------------- dynamics and restart
# Task 2 removes this mark.  Not strict: in task 1 S is carried unchanged through the step and does not enter
# the density, so the dynamics are already bitwise and the test XPASSes (strict=True would fail the suite).
@pytest.mark.xfail(strict=False, reason="task 2")
def test_uniform_S_at_s_ref_leaves_dynamics_bitwise():
    warnings.simplefilter("ignore")
    m, st, wind = _salty_zstar()
    for n in range(10):
        st = m.step(st, *wind(n))
    assert st.S is not None
    got, ref = nc.state_hashes(st, m), GOLDEN["zstar_vi_arakawa"]["10"]
    for key in ("eta", "u", "v", "T", "h"):
        assert got[key] == ref[key], key


def test_restart_with_salinity_bit_for_bit(tmp_path):
    warnings.simplefilter("ignore")
    m, st, wind = _salty_zstar()
    st.S = st.S + 0.01 * np.arange(nc.NX)[None, None, :]              # not uniform: every bit of S is checked
    S0 = st.S.copy()
    a = st
    for n in range(3):
        a = m.step(a, *wind(n))
    m2, st2, _ = _salty_zstar()
    st2.S = S0.copy()
    for n in range(2):
        st2 = m2.step(st2, *wind(n))
    assert st2.S is not None
    write_restart(tmp_path / "r.nc", m2, st2)
    import netCDF4
    with netCDF4.Dataset(tmp_path / "r.nc") as ds:
        assert ds["S"].getncattr("component") == "state"
        assert int(ds.getncattr("config_salinity")) == 1
    m3, _, _ = _salty_zstar()
    b = read_restart(tmp_path / "r.nc", m3)
    assert np.array_equal(b.S, st2.S)
    b = m3.step(b, *wind(2))
    ha, hb = nc.state_hashes(a, m), nc.state_hashes(b, m3)
    assert "S" in ha and ha == hb


def test_restart_rejects_salinity_mismatch(tmp_path):
    warnings.simplefilter("ignore")
    m, st, _ = _salty_zstar()
    write_restart(tmp_path / "r.nc", m, st)
    m_plain, _, _ = nc.CASES["zstar_vi_arakawa"]()
    with pytest.raises(ValueError, match="salinity"):
        read_restart(tmp_path / "r.nc", m_plain)
