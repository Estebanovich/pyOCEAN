"""N1 task 2: everything that crosses steps travels in SigmaState (spec §3) — here, the GLS closure state."""
from __future__ import annotations

import warnings

import numpy as np

from tests.n1_cases import CASES, state_hashes


def test_gls_state_travels_in_sigma_state():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["sigma_gls"]()
    assert set(st.closure) == {"tke", "eps", "akv"}
    for n in range(3):
        st = m.step(st, *wind(n))
    assert set(st.closure) == {"tke", "eps", "akv"}
    # a second model continued from st alone (its own GLS object never stepped) gives the same next state
    m2, _, _ = CASES["sigma_gls"]()
    a = state_hashes(m.step(st, *wind(3)))
    b = state_hashes(m2.step(st, *wind(3)))
    assert a == b


def test_constant_closure_has_no_closure_state():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["sigma_jacobian"]()
    st = m.step(st, *wind(0))
    assert st.closure is None
