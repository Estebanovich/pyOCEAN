"""Tests of the MITgcm internal_wave benchmark tools (spec 2026-10-12-baroclinic-obc-internal-wave §3)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.internal_wave import case

HAVE = (case.SRC / "input" / "delXvar").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="MITgcm internal_wave inputs not found")


def test_inputs_match_gendata():
    d = case.load_inputs()
    assert d["dx"].shape == (60,) and np.allclose(d["dx"], 13.3e3 / 61)
    assert d["H"][0] == 200.0 and abs(d["H"][-2] - 40.0) < 1e-9 and np.isnan(d["H"][-1])
    assert d["T"].shape == (20, 60) and np.allclose(d["T"][:, 0], d["tref"])
    np.testing.assert_allclose(np.diff(d["tref"]), -0.00509684, rtol=1e-4)


def test_obcs_west_reproduces_obcs_calc():
    # obcs_calc.F: OBWu = U0 * (cos(π rC/rF(Nr+1)) − mean) * sin(2πt/P) * cos(kx (3−2−0.5) Δx);
    # OBWt = tRef + U0 sin(π (k−½)/Nr) sin(2πt/P) sqrt(strat/(α g)) sqrt(ω² − f²)/ω
    d = case.load_inputs()
    t = 0.3 * case.PERIOD
    u, T = case.obcs_west(t, d["dz"])
    zc = -(np.arange(20) + 0.5) * 10.0
    vs = np.cos(np.pi * zc / -200.0)                                       # cos(mz π rC/rF(Nr+1))
    vs = vs - (vs * 10.0).sum() / -200.0                                    # − tmpsum (= Σ vs drF / rF(Nr+1))
    omega = 2 * np.pi / case.PERIOD
    kx = 2 * np.pi / 400.0 * np.sqrt(omega**2 / (1e-6 - omega**2))
    np.testing.assert_allclose(u, 0.024 * vs * np.sin(omega * t) * np.cos(kx * 0.5 * d["dx"][0]), rtol=1e-12)
    strat = 1e-6 / (9.81 * 2e-4)
    dT = 0.024 * np.sin(np.pi * (np.arange(1, 21) - 0.5) / 20) * np.sin(omega * t) \
        * np.sqrt(strat / (2e-4 * 9.81)) * omega / omega
    np.testing.assert_allclose(T, d["tref"] + dT, rtol=1e-12)
    assert abs((u * d["dz"]).sum()) < 1e-15                                 # zero depth mean


def test_mitgcm_depth_partial_cells():
    dz = np.full(20, 10.0)
    np.testing.assert_allclose(case.mitgcm_depth(np.array([200.0, 123.0, 101.5, 100.5, 40.0]), dz),
                               [200.0, 123.0, 102.0, 100.0, 40.0])          # hFac < 0.1 dropped, else ≥ 0.2


def test_read_mitgcm_shapes():
    if not (case.RUNS / "mitgcm" / "run").exists():
        pytest.skip("MITgcm internal_wave run not done yet")
    d = case.read_mitgcm()
    n = d["t"].size
    assert d["U"].shape == (n, 20, 61) and d["T"].shape == (n, 20, 60) and n > 200
    assert abs(d["t"][-1] - 10 * case.PERIOD) < 3600.0
    np.testing.assert_allclose(d["x"], np.cumsum(case.load_inputs()["dx"]) - 0.5 * case.load_inputs()["dx"])
    inp = case.load_inputs()
    np.testing.assert_allclose(np.nan_to_num(case.mitgcm_depth(inp["H"], inp["dz"])), d["depth"], atol=1e-9)
