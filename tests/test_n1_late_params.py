"""N1 final review (Important 1): model attributes changed after construction reach the step (as on main), or
raise if they cannot (EOS constants), and the restart fingerprint reads what runs."""
from __future__ import annotations

import warnings

import pytest

from tests.n1_cases import CASES, state_hashes


def _run(m, st, wind, n=3):
    for k in range(n):
        st = m.step(st, *wind(k))
    return state_hashes(st)


@pytest.mark.parametrize("case, name, value", [
    ("zstar_quadratic_drag", "cd", 0.1),
    ("zstar_quadratic_drag", "hbbl", 1.0),
    ("zstar_quadratic_drag", "drag_bg_vel", 0.3),
    ("hybrid_open_radiating", "thickness_flux", "upwind"),
    ("hybrid_open_radiating", "tracer_scheme", "upwind1"),
    ("sigma_jacobian", "tracer_time", "pc"),
    ("sigma_jacobian", "pgf", "twoterm"),
    ("zstar_vi_arakawa", "momentum", "flux"),
    ("hybrid_default", "remap_code", 1),
])
def test_late_change_takes_effect_as_on_a_model_built_with_it(case, name, value):
    warnings.simplefilter("ignore")
    m, st, wind = CASES[case]()
    base = _run(m, st, wind)
    m, st, wind = CASES[case]()
    setattr(m, name, value)
    assert getattr(m, name) == value
    assert _run(m, st, wind) != base


@pytest.mark.parametrize("name", ["max_truncations", "h_thin_bed", "regrid_rule"])
def test_late_change_reaches_the_ale_coordinate(name):
    warnings.simplefilter("ignore")
    m, _, _ = CASES["hybrid_default"]()
    value = {"max_truncations": 7, "h_thin_bed": 0.05, "regrid_rule": "zstar"}[name]
    setattr(m, name, value)
    assert getattr(m._vertical, name) == value


@pytest.mark.parametrize("name", ["rho0", "t_coef", "t_ref"])
def test_eos_constants_are_fixed_after_construction(name):
    warnings.simplefilter("ignore")
    m, _, _ = CASES["sigma_jacobian"]()
    with pytest.raises(AttributeError, match="fixed after construction"):
        setattr(m, name, 1.0)


def test_godunov_set_late_is_stored_with_the_kernel_spelling():
    warnings.simplefilter("ignore")
    m, _, _ = CASES["zstar_vi_arakawa"]()
    m.ke_scheme = "godunov"
    assert m.ke_scheme == "gudonov" and m._momentum.ke_scheme == "gudonov"
