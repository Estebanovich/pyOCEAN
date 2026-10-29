"""
N3 task 1: the turbulence closure as its own component (``core/components/closure.py``), no physics change.

* ``test_closure_extraction_is_bitwise``: every N1 golden configuration (with and without GLS) reproduces the
  hashes of ``tests/data/n1_golden/hashes.json`` (same check as ``tests/test_n1_golden.py``).
* ``test_gls_stratified_bitwise_after_extraction``: σ + stratified GLS (N2a task 8c) columns run 50 steps and give
  the hashes fixed with the code before the extraction (``tests/data/n3_gls_sigma_hash.json``, written by
  ``scripts/n3_make_gls_hash.py`` at commit 7ee57ad).
* ``test_closure_state_in_restart``: the σ + GLS restart round trip stays bit for bit (``tests/test_restart.py``
  covers it in full; here the closure fields of the component are checked by name).
* interface checks of ``Closure``, ``ConstantClosure`` and ``GLSClosure``.

Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import json
import warnings
from pathlib import Path

import numpy as np
import pytest

from tests import n1_cases as nc
from tests.n1_cases import all_case_names, diff_fields, run_case, state_hashes

ROOT = Path(__file__).resolve().parents[1]
GOLDEN = json.loads((ROOT / "tests" / "data" / "n1_golden" / "hashes.json").read_text())
GLS_HASH_FILE = ROOT / "tests" / "data" / "n3_gls_sigma_hash.json"
GLS_STEPS = (1, 10, 50)


def _gls_cases() -> dict:
    """Stratified σ + GLS configurations on the N1 10×6×6 basin (turbulence seeded, 0.5 m seiche)."""
    k = np.arange(nc.NZ)[:, None, None] / (nc.NZ - 1) * np.ones((1, nc.NY, nc.NX))

    def sal_wind():
        m = nc._sigma(closure="gls", z0=1e-3, gls_stratified=True, salinity=True, t_coef=0.05, s_coef=0.78,
                      s_ref=30.0)
        m.core.gls.tke[:], m.core.gls.eps[:] = 1e-4, 1e-7              # seeded turbulence (test_gls_stratified)
        st = m.initial_state(10.0 * nc._eta(), 12.0 + 0.5 * k, S=30.0 - 0.03 * k)
        return m, st, nc._wind(True)

    def temp_kv():
        m = nc._sigma(closure="gls", z0=1e-3, gls_stratified=True, t_coef=0.2, kv=1e-5, kh=5.0)
        m.core.gls.tke[:], m.core.gls.eps[:] = 1e-4, 1e-7
        st = m.initial_state(10.0 * nc._eta(), 12.0 + 0.1 * k)
        return m, st, nc._wind(False)

    return {"gls_strat_sal_wind": sal_wind, "gls_strat_temp_kv": temp_kv}


def gls_sigma_hashes() -> dict[str, dict[str, dict[str, str]]]:
    """State hashes (closure included) of the stratified σ + GLS cases after ``GLS_STEPS`` steps."""
    from core.dynamics import sigma_baroclinic as sb
    sb.set_threads(1)
    out = {}
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for name, build in _gls_cases().items():
            m, st, wind = build()
            res, n = {}, 0
            for target in GLS_STEPS:
                while n < target:
                    st = m.step(st, *wind(n))
                    n += 1
                res[str(target)] = state_hashes(st, m)
            out[name] = res
    return out


@pytest.mark.parametrize("name", all_case_names())
def test_closure_extraction_is_bitwise(name):
    got = run_case(name)
    for step, ref in GOLDEN[name].items():
        assert got[int(step)] == ref, f"{name} step {step}: fields differ {diff_fields(got[int(step)], ref)}"


def test_gls_stratified_bitwise_after_extraction():
    ref = json.loads(GLS_HASH_FILE.read_text())
    got = gls_sigma_hashes()
    for name, steps in ref.items():
        for step, h in steps.items():
            assert got[name][step] == h, f"{name} step {step}: fields differ {diff_fields(got[name][step], h)}"


def test_closure_state_in_restart(tmp_path):
    from core.restart import read_restart, write_restart
    warnings.simplefilter("ignore")
    m, st, wind = _gls_cases()["gls_strat_sal_wind"]()
    for n in range(3):
        st = m.step(st, *wind(n))
    write_restart(tmp_path / "r.nc", m, st)
    m2, _, _ = _gls_cases()["gls_strat_sal_wind"]()
    st2 = read_restart(tmp_path / "r.nc", m2)
    assert set(m._closure.state_fields()) == {"closure_tke", "closure_eps", "closure_akv", "closure_akt"}
    for k, a in m._closure.state_fields().items():
        assert np.array_equal(a, m2._closure.state_fields()[k]), k
    a, b = m.step(st, *wind(3)), m2.step(st2, *wind(3))
    assert state_hashes(a, m) == state_hashes(b, m2)


def test_constant_closure_interface():
    from core.components.closure import ConstantClosure, make_closure
    m = nc._sigma()
    c = m._closure
    assert isinstance(c, ConstantClosure) and isinstance(make_closure(m), ConstantClosure)
    assert c.viscosity(None) is None and c.tracer_diffusivity() is None and c.nonlocal_flux() is None
    assert c.state_fields() == {} and c.state() is None
    c.remap(None, None)
    c.update(None, None, None, None, None, None)                        # no state, nothing to do


def test_gls_closure_interface():
    from core.components.closure import GLSClosure
    m = nc._sigma(closure="gls", z0=1e-3, gls_stratified=True)
    c = m._closure
    assert isinstance(c, GLSClosure) and c.gls is m.core.gls
    nu_u, nu_v = c.viscosity(None)
    assert nu_u.shape == (nc.NZ + 1, nc.NY, nc.NX + 1) and nu_v.shape == (nc.NZ + 1, nc.NY + 1, nc.NX)
    assert c.tracer_diffusivity().shape == (nc.NZ + 1, nc.NY, nc.NX)
    assert c.nonlocal_flux() is None
    with pytest.raises(RuntimeError, match="ALE"):
        c.remap(np.ones(1), np.ones(1))                                 # σ model: no remap (ALE: N3 task 5)
    m2 = nc._sigma(closure="gls", z0=1e-3, gls_stratified=False)
    assert m2._closure.tracer_diffusivity() is None                    # homogeneous GLS: kv as before
