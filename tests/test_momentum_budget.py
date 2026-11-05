"""
Tests of ``diagnostics.momentum_budget`` (gs_drift d11 Phase 4): the read-only budget leaves the run bit for bit
unchanged, its terms close the layer-velocity change of every step, and its records are the means of the steps.
Small regional case of tests/test_nwa12_cut.py (synthetic GLORYS, ERA5, GloFAS, tides; the NWA12 parity physics).
"""
from __future__ import annotations

import numpy as np
import pytest

from diagnostics.momentum_budget import AB_TERMS, MomentumBudget, partial_weights
from tests.test_nwa12_cut import _assert_identical, _build, _run, _write_case


@pytest.fixture(scope="module")
def syn(tmp_path_factory):
    return _write_case(tmp_path_factory.mktemp("mb_syn"))


def test_partial_weights():
    h = np.array([50.0, 30.0, 20.0])[:, None]                 # bed first: surface layer 20 m, then 30, then 50
    np.testing.assert_allclose(partial_weights(h, 35.0)[:, 0], [0.0, 15.0, 20.0])
    np.testing.assert_allclose(partial_weights(h, 1000.0)[:, 0], [50.0, 30.0, 20.0])


def test_budget_does_not_change_the_run(syn):
    from core.dynamics.sigma_baroclinic import set_threads
    set_threads(2)
    a = _build(syn, days=0.5, dt=1800.0)
    ref = _run(a, a.state, 0, 12)
    b = _build(syn, days=0.5, dt=1800.0)
    MomentumBudget(b.model, window=(2, 6, 3, 8), z_part=50.0, steps_per_record=4)
    _assert_identical(ref, _run(b, b.state, 0, 12))


def test_budget_closes_and_records_are_step_means(syn):
    from core.dynamics.sigma_baroclinic import set_threads
    set_threads(2)
    b = _build(syn, days=0.5, dt=1800.0)
    mb = MomentumBudget(b.model, window=(2, 6, 3, 8), z_part=50.0, steps_per_record=3)
    steps = []
    orig = mb._accumulate

    def spy(ctx, T):                                          # keep the per-step terms to check the record means
        steps.append({x: (T[x][0].copy(), ctx.hu.copy()) for x in ("DUDT", "PF", "HV", "IMP")})
        orig(ctx, T)
    mb._accumulate = spy
    _run(b, b.state, 0, 6)
    assert len(mb.records) == 2
    assert mb.max_resid < 1e-10                               # (1) closes to round-off on every face
    r = mb.records[1]
    for x in ("DUDT", "PF", "HV", "IMP"):
        mean = np.mean([(h * X).sum(axis=0) for X, h in (s[x] for s in steps[3:6])], axis=0)
        np.testing.assert_allclose(r["I_" + x + "_u"], mean, rtol=1e-12, atol=1e-18)
    tot = sum(r["I_" + x + "_u"] for x in ("PF", "CA", "gKE", "HV", "EXT", "IMP", "BDY", "REM"))
    scale = max(np.abs(r["I_" + x + "_u"]).max() for x in ("PF", "CA", "IMP"))
    assert np.abs(r["I_DUDT_u"] - tot).max() <= 1e-10 * scale
    # the free-surface/viscous split: Σ h VV = τ_s/ρ₀ − τ_b/ρ₀ and BT + VV = IMP
    np.testing.assert_allclose(r["I_VV_u"], r["I_WIND_u"] + r["I_BED_u"], atol=1e-12 * scale)
    np.testing.assert_allclose(r["I_BT_u"] + r["I_VV_u"], r["I_IMP_u"], atol=1e-12 * scale)
    assert np.abs(r["I_BED_u"]).max() > 0 and np.abs(r["I_WIND_u"]).max() > 0
    assert np.abs(r["I_HV_u"]).max() > 0 and np.abs(r["I_CA_u"] - r["I_CAf_u"]).max() > 0
    assert r["W_PF_u"].shape == (b.model.n, 5, 7) and r["W_PF_v"].shape == (b.model.n, 6, 6)
    assert set(AB_TERMS) <= {k[2:-2] for k in r if k.startswith("I_")}


def test_budget_refuses_sigma(syn):
    b = _build(syn, days=0.5, dt=1800.0)
    b.model.ale = False
    with pytest.raises(ValueError):
        MomentumBudget(b.model, window=(0, 1, 0, 1))
