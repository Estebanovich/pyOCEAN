"""N1 task 12: pre-flight — all configuration errors at once, warnings for ignored arguments."""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import pytest

from core.model.contract import ConfigurationError, PyOceanConfigWarning
from core.model.driver import Driver
from tests import n1_cases as nc

GOLDEN = json.loads((Path(__file__).parent / "data" / "n1_golden" / "hashes.json").read_text())


def test_preflight_collects_all_errors():
    with pytest.raises(ConfigurationError) as e:
        nc._sigma(pgf="bad", adv="bad")
    msg = str(e.value)
    assert "2 configuration errors" in msg and "pgf" in msg and "adv='bad'" in msg


def test_error_message_names_fix():
    with pytest.raises(ValueError, match="valid: 'flux', 'upwind3', 'centred'"):
        nc._sigma(adv="bad")


def test_ignored_args_warn_and_keep_bits():
    with pytest.warns(PyOceanConfigWarning, match="regrid is ignored"):
        m = nc._sigma(pgf="jacobian", regrid="zstar")
    warnings.simplefilter("ignore")
    m, st, wind = nc._start(m, True)
    for n in range(10):
        st = m.step(st, *wind(n))
    assert nc.state_hashes(st, m) == GOLDEN["sigma_jacobian"]["10"]


def test_no_warning_for_a_plain_configuration():
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyOceanConfigWarning)
        nc._ale("zstar")
        nc._sigma(pgf="fv", kh=5.0)


def test_gudonov_alias_warns_and_godunov_is_the_same_scheme():
    with pytest.warns(PyOceanConfigWarning, match="godunov"):
        a = nc._ale("zstar", ke_scheme="gudonov")
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyOceanConfigWarning)
        b = nc._ale("zstar", ke_scheme="godunov")
    warnings.simplefilter("ignore")
    (a, sa, w), (b, sb_, _) = nc._start(a, False), nc._start(b, False)
    for n in range(3):
        sa, sb_ = a.step(sa, *w(n)), b.step(sb_, *w(n))
    assert nc.state_hashes(sa) == nc.state_hashes(sb_)


def test_driver_accepts_a_single_rate_only():
    warnings.simplefilter("ignore")
    m, _, _ = nc.CASES["zstar_vi_arakawa"]()
    with pytest.raises(ConfigurationError, match="single rate"):
        Driver(m, dt_tracer=2 * m.dt)


def test_salinity_without_s_coef_warns():
    """salinity=True with the default s_coef = 0 leaves S out of ρ: say so before the run (N2a final review I1)."""
    with pytest.warns(PyOceanConfigWarning, match="s_coef = 0"):
        nc._sigma(salinity=True)
    with warnings.catch_warnings():
        warnings.simplefilter("error", PyOceanConfigWarning)
        nc._sigma(salinity=True, s_coef=0.78, s_ref=30.0)
