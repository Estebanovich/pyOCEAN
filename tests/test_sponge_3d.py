"""
Per-column (3-D) targets of the ALE interior sponge (``core.components.sponge.InteriorSponge``; ruling 2026-11-04).

``sponge_T``, ``sponge_S`` and ``sponge_tr`` may be a profile (nz_s,) shared by every column, as before, or a
per-column target (nz_s, ny, nx); ``sponge_zw`` may be (nz_s + 1,) or (nz_s + 1, ny, nx).  Each column is remapped
onto the current layers exactly as the 1-D path does (MOM6 MOM_ALE_sponge.F90 apply_ALE_sponge: a per-column
target on its own z grid, remapped with remapping_core_h).  Index order (k, j, i), k = 0 at the bed for the model
state; sponge targets and interfaces are surface first.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.model.contract import ConfigurationError
from core.restart import read_restart, write_restart
from tests import n1_cases as nc

NZS = 4
ZW1 = np.linspace(0.0, -nc.HMAX, NZS + 1)
T1 = np.array([13.0, 12.0, 11.0, 10.5])
S1 = np.array([31.0, 32.0, 33.0, 34.0])
RULE_NZ = {"zstar": nc.NZ, "sigma": nc.NZ, "hybrid": 8}


def _rate() -> np.ndarray:
    r = np.zeros((nc.NY, nc.NX))
    r[:, -4:] = 2e-3
    return r


def _b(a: np.ndarray) -> np.ndarray:
    """A surface-first profile (n,) broadcast to every column (n, ny, nx)."""
    return np.ascontiguousarray(a[:, None, None] * np.ones((1, nc.NY, nc.NX)))


def _model(rule: str, **kw):
    warnings.simplefilter("ignore")
    return nc._ale(rule, nz=RULE_NZ[rule], salinity=True, s_coef=0.78, s_ref=33.0, sponge_rate=_rate(), **kw)


def _run(m, n: int = 6):
    nz = RULE_NZ[m.regrid_rule] if hasattr(m, "regrid_rule") else nc.NZ
    st = m.initial_state(nc._eta(), nc._T(nz), S=np.full((nz, nc.NY, nc.NX), 33.0))
    for _ in range(n):
        st = m.step(st, *nc._wind(True)(0))
    return st


def _same(a, b) -> None:
    for f in ("T", "S", "h", "eta", "u", "v"):
        assert np.array_equal(getattr(a, f), getattr(b, f)), f


# ---------------------------------------------------------------- 3-D target = broadcast 1-D: bit for bit
@pytest.mark.parametrize("rule", ["zstar", "sigma", "hybrid"])
@pytest.mark.parametrize("which", ["T_S", "zw", "all"])
def test_broadcast_3d_target_is_bit_identical_to_the_profile(rule, which):
    one = _model(rule, sponge_T=T1, sponge_S=S1, sponge_zw=ZW1)
    kw = dict(sponge_T=T1, sponge_S=S1, sponge_zw=ZW1)
    if which in ("T_S", "all"):
        kw.update(sponge_T=_b(T1), sponge_S=_b(S1))
    if which in ("zw", "all"):
        kw.update(sponge_zw=_b(ZW1))
    three = _model(rule, **kw)
    _same(_run(one), _run(three))


def test_broadcast_3d_tracer_target_is_bit_identical():
    one = _model("zstar", n_tracers=2, sponge_T=T1, sponge_S=S1, sponge_zw=ZW1,
                 sponge_tr=np.stack([np.full(NZS, 0.5), np.linspace(0, 1, NZS)]))
    tr3 = np.stack([_b(np.full(NZS, 0.5)), _b(np.linspace(0, 1, NZS))])
    three = _model("zstar", n_tracers=2, sponge_T=_b(T1), sponge_S=_b(S1), sponge_zw=ZW1, sponge_tr=tr3)

    def run(m):
        st = m.initial_state(nc._eta(), nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 33.0),
                             tr=np.zeros((2, nc.NZ, nc.NY, nc.NX)))
        for _ in range(4):
            st = m.step(st)
        return st
    a, b = run(one), run(three)
    _same(a, b)
    assert np.array_equal(a.tr, b.tr)


# ---------------------------------------------------------------- column-varying target
@pytest.mark.parametrize("rule", ["zstar", "sigma", "hybrid"])
def test_each_column_relaxes_toward_its_own_target(rule):
    # uniform targets that differ by column (in T and S): with Δt r = 60 every sponge column reaches its own value,
    # the unsponged columns keep theirs (neighbours far away), each column's mean is its own target
    Tt = 8.0 + 0.5 * np.arange(nc.NY)[:, None] + 0.25 * np.arange(nc.NX)[None, :]   # (ny, nx)
    St = 30.0 + 0.1 * np.arange(nc.NY)[:, None] - 0.2 * np.arange(nc.NX)[None, :]
    r = np.zeros((nc.NY, nc.NX))
    r[:, -3:] = 1.0
    warnings.simplefilter("ignore")
    m = nc._ale(rule, nz=RULE_NZ[rule], salinity=True, s_coef=0.0, t_coef=0.0, sponge_rate=r,
                sponge_T=np.repeat(Tt[None], NZS, 0), sponge_S=np.repeat(St[None], NZS, 0), sponge_zw=ZW1)
    nz = RULE_NZ[rule]
    st = m.initial_state(np.zeros((nc.NY, nc.NX)), np.full((nz, nc.NY, nc.NX), 12.0),
                         S=np.full((nz, nc.NY, nc.NX), 33.0))
    st = m.step(st)
    w = st.h[..., -3:] > 0
    dT = np.abs(st.T[..., -3:] - Tt[None, :, -3:])[w]
    dS = np.abs(st.S[..., -3:] - St[None, :, -3:])[w]
    assert dT.max() < 0.1 and dS.max() < 0.1
    assert np.abs(st.T[..., :4] - 12.0).max() < 1e-12 and np.abs(st.S[..., :4] - 33.0).max() < 1e-12


def test_column_varying_profile_on_its_own_z_grid():
    # per-column interfaces: a two-level target whose interface depth varies by column; the column content of the
    # relaxed tracer at Δt r → ∞ equals the content of its own target clipped and stretched to the column
    r = np.zeros((nc.NY, nc.NX))
    r[:, -4:] = 1e6
    zmid = -(5.0 + np.arange(nc.NX))[None, :] * np.ones((nc.NY, 1))
    zw = np.stack([np.zeros_like(zmid), zmid, np.full_like(zmid, -nc.HMAX)])
    Tt = np.stack([np.full_like(zmid, 20.0), np.full_like(zmid, 10.0)])
    warnings.simplefilter("ignore")
    m = nc._ale("zstar", t_coef=0.0, sponge_rate=r, sponge_T=Tt, sponge_zw=zw)
    st = m.step(m.initial_state(np.zeros((nc.NY, nc.NX)), np.full((nc.NZ, nc.NY, nc.NX), 15.0)))
    D = st.h.sum(axis=0)
    content = (st.T * st.h).sum(axis=0)
    Hc = nc.H
    expect = (20.0 * -zmid + 10.0 * (Hc + zmid)) / Hc * D             # η = 0 at t = 0, scale = D/H
    np.testing.assert_allclose(content[:, -4:], expect[:, -4:], rtol=1e-6)


# ---------------------------------------------------------------- pre-flight
def test_preflight_reports_every_3d_shape_error_together():
    bad = dict(sponge_rate=np.zeros((2, 3)), sponge_T=np.zeros((NZS, 2, 2)), sponge_S=np.zeros((NZS, nc.NY)),
               sponge_zw=np.zeros((NZS, nc.NY, nc.NX)), sponge_tr=np.zeros((1, NZS + 1)))
    with pytest.raises(ConfigurationError) as e:
        nc._ale("zstar", salinity=True, n_tracers=1, **bad)
    msg = str(e.value)
    for key in ("sponge_rate shape", "sponge_T", "sponge_S shape", "sponge_zw", "sponge_tr"):
        assert key in msg, key


def test_preflight_refuses_non_finite_3d_target_in_sponge_columns_and_increasing_zw():
    T3 = _b(T1)
    T3[2, 0, -1] = np.nan
    zw = _b(ZW1)
    zw[1, 1, -1] = 1.0
    with pytest.raises(ConfigurationError) as e:
        nc._ale("zstar", sponge_rate=_rate(), sponge_T=T3, sponge_zw=zw)
    assert "finite" in str(e.value) and "decreasing" in str(e.value)
    T3 = _b(T1)
    T3[:, 0, 0] = np.nan                                                # outside the sponge columns: allowed
    zw = _b(ZW1)
    zw[:, 0, 0] = np.nan
    warnings.simplefilter("ignore")
    m = nc._ale("zstar", sponge_rate=_rate(), sponge_T=T3, sponge_zw=zw)
    st = m.step(m.initial_state(nc._eta(), nc._T()))
    assert np.isfinite(st.T).all()


# ---------------------------------------------------------------- restart
def test_restart_with_3d_target_is_bit_for_bit_and_checks_the_target(tmp_path):
    rng = np.random.default_rng(3)
    T3 = _b(T1) + 0.1 * rng.standard_normal((NZS, nc.NY, nc.NX))
    S3 = _b(S1) + 0.1 * rng.standard_normal((NZS, nc.NY, nc.NX))

    def build(T=T3):
        return _model("zstar", sponge_T=T, sponge_S=S3, sponge_zw=ZW1)
    m = build()
    st = m.initial_state(nc._eta(), nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 33.0))
    for _ in range(3):
        st = m.step(st)
    write_restart(tmp_path / "r.nc", m, st)
    ref = st
    for _ in range(3):
        ref = m.step(ref)
    m2 = build()
    st2 = read_restart(tmp_path / "r.nc", m2)
    for _ in range(3):
        st2 = m2.step(st2)
    _same(ref, st2)
    T4 = T3.copy()
    T4[0, 0, -1] += 1e-9
    with pytest.raises(ValueError, match="sponge_T"):
        read_restart(tmp_path / "r.nc", build(T4))
