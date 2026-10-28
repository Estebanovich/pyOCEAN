"""N1 task 4: momentum component — named explicit terms summed in the fixed order of SigmaBaroclinic."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.components.momentum import Momentum, sum_terms
from tests.n1_cases import CASES


@pytest.mark.parametrize("case, names", [
    ("zstar_vi_arakawa", ["pgf", "vector_invariant", "hviscosity"]),
    ("zstar_flux", ["pgf", "coriolis", "advection", "hviscosity"]),
    ("sigma_upwind3_adv", ["pgf", "coriolis", "advection", "hviscosity", "vadvection"]),
    ("sigma_centred_adv", ["pgf", "coriolis", "advection", "hviscosity", "vadvection"]),
])
def test_named_terms_sum_to_the_model_tendency(case, names):
    warnings.simplefilter("ignore")
    m, st, wind = CASES[case]()
    for n in range(2):                       # second step: last-step transports and ω exist
        st = m.step(st, *wind(n))
    assert isinstance(m._momentum, Momentum)
    terms, wet_u, wet_v = m._momentum_terms(st)
    assert [t[0] for t in terms] == names
    gu, gv = sum_terms(terms, wet_u, wet_v)
    ru, rv = m._tendencies(st)
    assert np.array_equal(gu, ru) and np.array_equal(gv, rv)


def test_sum_terms_signs_and_order():
    a, b = np.full((1, 1, 3), 1.0), np.full((1, 1, 3), 0.25)
    wet = np.array([[True, False, True]])
    gu, _ = sum_terms([("x", a.copy(), a.copy(), "+"), ("y", b, b, "-")], wet, wet)
    assert np.array_equal(gu, np.array([[[0.75, 0.0, 0.75]]]))


def test_parameters_set_after_construction_reach_the_components():
    # validation/bayiw_cut/native_columns.py and mom6_ke_schemes/lock_pyocean.py set m.ke_scheme after building;
    # tests set m.kh — the components must see it
    warnings.simplefilter("ignore")
    m, _, _ = CASES["zstar_vi_arakawa"]()
    m.ke_scheme, m.kh, m.ah = "gudonov", 7.0, 3.0
    assert m._momentum.ke_scheme == "gudonov" and m._tracer_diffusion.kh == 7.0 and m._momentum.ah == 3.0
    assert m.ke_scheme == "gudonov" and m.kh == 7.0 and m.ah == 3.0
