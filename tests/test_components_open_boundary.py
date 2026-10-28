"""N1 task 6: open-boundary component (core/components/open_boundary.py) — no hidden per-step state."""
from __future__ import annotations

import copy
import warnings

import numpy as np

from core.components.open_boundary import OpenBoundary
from tests.n1_cases import CASES


def _snapshot(obj) -> dict:
    out = {}
    for k, v in vars(obj).items():
        # arrays and containers by value; references to other objects (free surface, provider) by identity
        out[k] = v.copy() if isinstance(v, np.ndarray) else copy.copy(v) if isinstance(v, (dict, list, tuple)) else v
    return out


def _same(a, b) -> bool:
    if isinstance(a, np.ndarray) or isinstance(b, np.ndarray):
        return isinstance(a, np.ndarray) and isinstance(b, np.ndarray) and np.array_equal(a, b, equal_nan=True)
    return a is b or a == b


def test_open_boundary_component_has_no_hidden_step_state():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["zstar_obc_radiating"]()
    ob = m._ob
    assert isinstance(ob, OpenBoundary)
    st = m.step(st, *wind(0))
    before = _snapshot(ob)
    for n in range(1, 3):
        st = m.step(st, *wind(n))
    after = _snapshot(ob)
    changed = [k for k in before if not _same(before[k], after[k])]
    assert changed == [], changed


def test_exterior_is_returned_not_stored():
    warnings.simplefilter("ignore")
    m, st, _ = CASES["sigma_obc_state"]()
    ds = m.ds[:, None, None]
    Hu, Hv = m._ob.edge_face_depths()
    ext = m._ob.evaluate(0.0, Hu[None] * ds, Hv[None] * ds, None, None)
    assert set(ext) >= {"flather", "up_u", "up_v", "own", "state"}
    assert m._ob.evaluate.__self__ is m._ob and not hasattr(m._ob, "_ext")
