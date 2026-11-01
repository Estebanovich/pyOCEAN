"""
TEOS-10 in the 3-D core, Task 7 (spec docs/superpowers/specs/2026-10-31-teos10-3d-design.md goal F): surface
thermodynamics — the bulk SST/SSS conversion after the ``h_sfc`` average (MOM6 coupler order,
``docs/design/teos10_refs/mom6_eos.md`` §6), the heat phase with the EOS c_p, the exact heat content, and the
below-freezing count.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import gsw
import numpy as np
import pytest

from core.components.surface_fluxes import BulkFluxes, surface_average
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.model.registry import REGISTRY
from physics.eos.convert import sa_to_sp
from tests import n1_cases as nc
from tests.test_eos_model import NX, NY, NZ, H, _front, _grid, _model
from tests.test_surface_fluxes_model import _atm


def _bulk_model(eos: str, scheme: str = "ncar", h_sfc: float = 12.0):
    bulk = BulkFluxes(_atm(_grid()), scheme=scheme, h_sfc=h_sfc)
    m = _model("zstar", eos, surface_fluxes=bulk)
    T, S = _front()
    return m, m.initial_state(np.zeros((NY, NX)), T, S=S)


def test_bulk_sst_converted_after_average():
    """The bulk SST is θ(⟨S_A⟩, ⟨Θ⟩) of the h_sfc-thick mean, not the mean of the per-layer θ (MOM6 order)."""
    m, st = _bulk_model("teos10")
    m._surface_fluxes.forcing(st, 0.0)
    sst = m._surface_fluxes.last["sst_bulk"]
    SA, CT = surface_average(st.h, st.S, 12.0), surface_average(st.h, st.T, 12.0)
    assert np.array_equal(sst, gsw.pt_from_CT(SA, CT))
    mean_of_theta = surface_average(st.h, gsw.pt_from_CT(st.S, st.T), 12.0)
    assert np.max(np.abs(sst - mean_of_theta)) > 1e-8                    # the two orders really differ


def test_bulk_kara_receives_practical_salinity():
    """'kara' takes the SSS: with TEOS-10 it must be S_P = sa_to_sp(⟨S_A⟩)."""
    m, st = _bulk_model("teos10", "kara")
    seen = {}
    import core.components.surface_fluxes as sf
    orig = sf.bulk_kara

    def spy(*a, **k):
        seen["sss"] = k["sss"]
        return orig(*a, **k)
    sf.bulk_kara = spy
    try:
        m._surface_fluxes.forcing(st, 0.0)
    finally:
        sf.bulk_kara = orig
    SA = surface_average(st.h, st.S, 12.0)
    wet = np.isfinite(SA)
    assert np.array_equal(seen["sss"][wet], sa_to_sp(SA)[wet])


def test_bulk_sst_unchanged_for_theta_sp_eos():
    """linear EOS: the SST is the plain average (the path of before Task 7)."""
    m, st = _bulk_model("linear")
    m._surface_fluxes.forcing(st, 0.0)
    assert np.array_equal(m._surface_fluxes.last["sst_bulk"], surface_average(st.h, st.T, 12.0))


@pytest.mark.parametrize("coord", ["sigma", "zstar", "hybrid"])
def test_heat_content_exact_teos10(coord):
    """Criterion 8: c_p⁰ ρ₀ Σ V Θ changes by exactly the prescribed heat (≤ 1e-13 of the change)."""
    m = _model(coord, "teos10")
    T, S = _front()
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    heat = {"Q_ns": np.full((NY, NX), -1e5), "Q_sw": np.full((NY, NX), 1e6)}
    Tref = float(np.mean(T))

    def content(s):
        V = m.layer_volumes(s.eta, s.h)
        act = np.broadcast_to(m.fs.active[None], V.shape)
        return float(m.rho0 * m.cp * np.sum(((s.T - Tref) * V)[act]))
    c0, n = content(st), 6
    for _ in range(n):
        st = m.step(st, heat=heat)
    expected = 9e5 * float(np.sum(m.area[m.fs.active])) * n * m.dt
    assert abs((content(st) - c0) - expected) <= 1e-13 * abs(expected)


def test_cp_defaults():
    assert nc._sigma().cp == 3985.0
    assert _model("sigma", "wright_full").cp == 3991.86795711963
    assert _model("sigma", "teos10").cp == 3991.86795711963
    assert _model("sigma", "teos10", cp=3992.0).cp == 3992.0


@pytest.mark.parametrize("eos", ["linear", "teos10"])
def test_freezing_count(eos):
    m = _model("zstar", eos)
    T, S = _front()
    T = T.copy()
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    drv = m._driver
    assert drv._n_below_freezing(st.eta, st.T, st.S, st.h) == 0
    Tc = st.T.copy()
    Tc[NZ - 1, 2, 3] = -5.0                                              # surface cell, S ≈ 35: Tf ≈ −1.9
    ref = Tc.copy()
    n = 0 if eos == "linear" else 1                                      # the linear EOS has no freezing law (I1/M2)
    assert drv._n_below_freezing(st.eta, Tc, st.S, st.h) == n
    assert np.array_equal(Tc, ref)                                       # counted, never corrected
    st.T[:] = Tc
    m.step(st)
    assert (m.n_below_freezing >= 1) if n else (m.n_below_freezing == 0)  # model attribute, updated each step
    assert "n_below_freezing" in REGISTRY
