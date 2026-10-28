"""N1 task 11: fixed-order driver of the 3-D core (core/model/driver.py)."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from core.model.driver import PHASES, Driver
from tests.n1_cases import CASES, _eta, _T, run_case, state_hashes

GOLDEN = json.loads((Path(__file__).parent / "data" / "n1_golden" / "hashes.json").read_text())


@pytest.mark.parametrize("case", ["sigma_obc_state", "hybrid_open_radiating"])
def test_driver_phase_order(case):
    warnings.simplefilter("ignore")
    m, st, wind = CASES[case]()
    assert isinstance(m._driver, Driver)
    m._driver.debug = True
    for n in range(2):
        st = m.step(st, *wind(n))
    assert len(m._driver.phase_hashes) == 2
    assert tuple(m._driver.phase_hashes[-1]) == PHASES


def test_keep_terms_does_not_change_bits():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["zstar_vi_arakawa"]()
    m._driver.keep_terms = True
    for n in range(3):
        st = m.step(st, *wind(n))
    assert state_hashes(st, m) == GOLDEN["zstar_vi_arakawa"]["3"]
    assert list(m._driver.last_terms) == ["pgf", "vector_invariant", "hviscosity"]


def test_driver_reuse_model_two_runs():
    # same model, two initial_state calls: the second run is what main gave (golden case zstar_reuse)
    assert run_case("zstar_reuse") == {int(k): v for k, v in GOLDEN["zstar_reuse"].items()}
    # σ with the direct solver: the second run equals a fresh model
    warnings.simplefilter("ignore")
    m, st, wind = CASES["sigma_jacobian"]()
    for n in range(5):
        st = m.step(st, *wind(n))
    st2 = m.initial_state(_eta(), _T())
    m3, st3, _ = CASES["sigma_jacobian"]()
    for n in range(5):
        st2, st3 = m.step(st2, *wind(n)), m3.step(st3, *wind(n))
    assert state_hashes(st2, m) == state_hashes(st3, m3)
