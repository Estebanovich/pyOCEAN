"""N1 task 10: ALE vertical coordinate and interior sponge components."""
from __future__ import annotations

import warnings

import numpy as np

from core.components.sponge import InteriorSponge
from core.components.vertical_coordinate import ALECoordinate
from tests.n1_cases import CASES


def test_transport_thickness_cache_identity_preserved():
    warnings.simplefilter("ignore")
    m, st, _ = CASES["hybrid_default"]()
    vc = m._vertical
    assert isinstance(vc, ALECoordinate)
    a = vc.transport_thickness(st.h)
    assert vc.transport_thickness(st.h) is a                       # same h object → cached result
    assert m._transport_thickness(st.h) is a
    assert vc.transport_thickness(st.h.copy()) is not a           # a copy is another array


def test_sigma_model_has_no_ale_coordinate_and_sponge_component():
    warnings.simplefilter("ignore")
    m, _, _ = CASES["sigma_jacobian"]()
    assert m._vertical is None and m._interior_sponge_c is None
    m2, _, _ = CASES["zstar_sponge_interior"]()
    assert isinstance(m2._interior_sponge_c, InteriorSponge)
    assert np.array_equal(m2._interior_sponge_c.rate, m2.sponge_rate)
