"""
N2a (plan docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md): salinity in the 3-D core.

Task 1: S in the state, the linear EOS in (T, S), restart and pre-flight.
Task 2: S stepped along the path of T (advection, K_h without the T_ref anomaly, K_v, boundary sponge, open-edge
inflow, remap, interior sponge) in σ and ALE; stack layout [T, (T²), S, (S²), tr…].
EOS: ρ = ρ₀ − α (T − T₀) + β (S − S₀)  (Vallis 2006, eq. 1.156), S in g kg⁻¹, β = ``s_coef``.
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import dataclasses
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
def test_uniform_S_at_s_ref_leaves_dynamics_bitwise():
    # Ruling R1 (task 3): with S the FV PGF takes the density path fv_pgf_rho, distinct arithmetic from the T path,
    # so the dynamics equal the T-only golden run to round-off (relative 1e-12 of each field), no longer bit for bit
    warnings.simplefilter("ignore")
    m, st, wind = _salty_zstar()
    m0, st0, _ = nc._start(nc._ale("zstar"), True)
    for n in range(10):
        st, st0 = m.step(st, *wind(n)), m0.step(st0, *wind(n))
    assert st.S is not None
    assert nc.state_hashes(st0, m0)["u"] == GOLDEN["zstar_vi_arakawa"]["10"]["u"]       # the reference is the golden
    for key in ("eta", "u", "v", "T", "h"):
        a, b = getattr(st, key), getattr(st0, key)
        np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-12 * np.abs(b).max(), err_msg=key)


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


# ---------------------------------------------------------------- task 2: S in the time step
def _with_S(state_fn, S_of=None):
    """Wrap an exterior provider: every OBData gets S = S_of(ob) (default: S = T)."""
    def state(t: float) -> dict:
        return {e: dataclasses.replace(ob, S=np.array(ob.T, float) if S_of is None else S_of(ob))
                for e, ob in state_fn(t).items()}
    return state


def _run(m, st, wind, n=10):
    for i in range(n):
        st = m.step(st, *wind(i))
    return st


def test_stack_layout():
    from core.model.driver import _stack_layout
    assert _stack_layout(nc._sigma()) == {"T": 0, "T2": 1, "n": 2}
    assert _stack_layout(nc._sigma(salinity=True)) == {"T": 0, "T2": 1, "S": 2, "n": 3}
    assert _stack_layout(nc._ale("zstar", n_tracers=2)) == {"T": 0, "T2": 1, "tr": 2, "n": 4}
    assert _stack_layout(nc._ale("zstar", salinity=True, mix_diagnostic=False, n_tracers=1)) == \
        {"T": 0, "S": 1, "tr": 2, "n": 3}
    assert _stack_layout(nc._ale("zstar", salinity=True, n_tracers=1)) == {"T": 0, "T2": 1, "S": 2, "tr": 3, "n": 4}


def test_S_equals_T_when_both_passive_sigma():
    warnings.simplefilter("ignore")
    m = nc._sigma(kh=5.0, kv=1e-5, t_coef=0.0, salinity=True, s_coef=0.0)
    T0 = nc._T()
    st = _run(m, m.initial_state(nc._eta(), T0, S=T0.copy()), nc._wind(True))
    assert not np.array_equal(st.T, T0)                                # T has moved
    assert np.array_equal(st.S, st.T)


def test_S_equals_T_sigma_open_edges():
    # σ with a time-varying west exterior whose S equals its T, and a sponge band: S sees the inflow and the sponge
    warnings.simplefilter("ignore")
    m = nc._sigma(open_boundaries=("west",), sponge_cells=3, open_boundary_state=_with_S(nc._west_state()),
                  kh=5.0, kv=1e-5, t_coef=0.0, salinity=True)
    T0 = nc._T()
    st = _run(m, m.initial_state(nc._eta(), T0, S=T0.copy()), nc._wind(True))
    assert np.array_equal(st.S, st.T)


def test_S_equals_passive_tracer_ale():
    warnings.simplefilter("ignore")
    m = nc._ale("zstar", salinity=True, s_coef=0.0, n_tracers=1, kh=5.0, kv=1e-5)
    S0 = 30.0 + 0.1 * nc._T() + 0.05 * np.arange(nc.NY)[None, :, None]
    st = _run(m, m.initial_state(nc._eta(), nc._T(), tr=S0[None].copy(), S=S0.copy()), nc._wind(True))
    assert not np.array_equal(st.S, S0)
    assert np.array_equal(st.S, st.tr[0])


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_S_equals_T_ale_open_edges_and_sponges(rule):
    # S passive (s_coef 0) with S₀ = T₀ and every S source equal to T's: S = T through the inflow, the boundary
    # sponge band, K_h (both without the T_ref anomaly with salinity), K_v, the remap and the interior sponge
    # (sponge_S = sponge_T).  α = β = 0: T and S both passive (the FV PGF takes the density path, no 1/α; task 3)
    warnings.simplefilter("ignore")
    nz = 8 if rule == "hybrid" else nc.NZ
    sp = nc._sponge_kw()
    m = nc._ale(rule, nz=nz, open_boundaries=("west",), sponge_cells=3,
                open_boundary_state=_with_S(nc._west_state()), kh=5.0, kv=1e-5, salinity=True, t_coef=0.0,
                sponge_S=sp["sponge_T"].copy(), **sp)
    T0 = nc._T(nz)
    st = _run(m, m.initial_state(nc._eta(), T0, S=T0.copy()), nc._wind(True))
    assert np.isfinite(st.u).all() and not np.array_equal(st.T, T0)
    assert np.array_equal(st.S, st.T)


def test_uniform_S_stays_uniform_with_open_edges():
    warnings.simplefilter("ignore")
    S30 = lambda ob: np.full(np.shape(ob.T), 30.0)                      # noqa: E731
    m = nc._ale("zstar", open_boundaries=("west",), sponge_cells=3, salinity=True, s_coef=0.78, s_ref=30.0,
                open_boundary_state=_with_S(nc._west_state(), S30))
    st = _run(m, m.initial_state(nc._eta(), nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 30.0)), nc._wind(True))
    assert np.abs(st.S - 30.0).max() < 1e-12


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
def test_total_salt_exact_closed(coord):
    warnings.simplefilter("ignore")
    kw = dict(salinity=True, s_coef=0.78, s_ref=30.0, kh=5.0, kv=1e-5)
    m = nc._sigma(**kw) if coord == "sigma" else nc._ale("zstar", **kw)
    S0 = 30.0 + 0.5 * np.arange(nc.NX)[None, None, :] + 0.2 * np.arange(nc.NZ)[:, None, None] * np.ones(
        (nc.NZ, nc.NY, nc.NX))
    st0 = m.initial_state(nc._eta(), nc._T(), S=S0)
    s0 = m.tracer_content(st0, tracer="S")
    st = _run(m, st0, nc._wind(True))
    assert not np.array_equal(st.S, S0)
    np.testing.assert_allclose(m.tracer_content(st, tracer="S"), s0, rtol=1e-13)


def test_open_edge_without_S_is_an_error():
    warnings.simplefilter("ignore")
    m = nc._ale("zstar", open_boundaries=("west",), sponge_cells=3, salinity=True, s_coef=0.78, s_ref=30.0,
                open_boundary_state=nc._west_state())
    st = m.initial_state(nc._eta(), nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 30.0))
    with pytest.raises(ValueError, match="west.*S"):
        m.step(st)


def test_interior_sponge_relaxes_S():
    # a strong interior sponge drives S toward sponge_S in the sponge columns and leaves the others alone
    warnings.simplefilter("ignore")
    sp = nc._sponge_kw()
    sp["sponge_rate"] = np.where(sp["sponge_rate"] > 0, 1.0, 0.0)       # Δt r = 60: nearly the target
    m = nc._ale("zstar", salinity=True, s_coef=0.0, sponge_S=np.full(4, 25.0), **sp)
    st = _run(m, m.initial_state(nc._eta(), nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 30.0)), nc._wind(False), 2)
    assert np.abs(st.S[..., -3:] - 25.0).max() < 0.1
    assert np.abs(st.S[..., :4] - 30.0).max() < 1e-12


def test_sponge_S_needs_salinity_and_interior_sponge():
    sp = nc._sponge_kw()
    with pytest.raises(ConfigurationError, match="sponge_S needs salinity=True"):
        nc._ale("zstar", sponge_S=np.full(4, 25.0), **sp)
    with pytest.warns(PyOceanConfigWarning, match="sponge_S"):                # as sponge_T without sponge_rate
        nc._ale("zstar", salinity=True, sponge_S=np.full(4, 25.0))
    with pytest.raises(ValueError, match="sponge_S shape"):
        nc._ale("zstar", salinity=True, sponge_S=np.full(3, 25.0), **sp)


def test_restart_with_salinity_and_open_edges_bit_for_bit(tmp_path):
    warnings.simplefilter("ignore")

    def build():
        m = nc._ale("zstar", open_boundaries=("west",), sponge_cells=3, salinity=True, s_coef=0.78, s_ref=30.0,
                    open_boundary_state=_with_S(nc._west_state(), lambda ob: 30.0 + 0.1 * np.asarray(ob.T)))
        S0 = 30.0 + 0.01 * np.arange(nc.NX)[None, None, :] * np.ones((nc.NZ, nc.NY, nc.NX))
        return m, m.initial_state(nc._eta(), nc._T(), S=S0)
    wind = nc._wind(True)
    m, a = build()
    a = _run(m, a, wind, 3)
    m2, b = build()
    b = _run(m2, b, wind, 2)
    write_restart(tmp_path / "r.nc", m2, b)
    m3, _ = build()
    m3.S_ext = None                                                    # must come back from the file
    c = read_restart(tmp_path / "r.nc", m3)
    assert np.array_equal(m3.S_ext, m2.S_ext)
    c = m3.step(c, *wind(2))
    assert nc.state_hashes(a, m) == nc.state_hashes(c, m3)
