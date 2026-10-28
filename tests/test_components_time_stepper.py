"""N1 task 5: AB3 time stepper of the momentum (core/components/time_stepper.py)."""
from __future__ import annotations

import numpy as np

from core.components.time_stepper import AB3, AB3Momentum, ab3_combine
from core.dynamics import sigma_baroclinic as sb


def test_ab3_start_up_and_history():
    rng = np.random.default_rng(1)
    g = [(rng.standard_normal((2, 3, 5)), rng.standard_normal((2, 4, 4))) for _ in range(4)]
    ts = AB3Momentum()
    hist: list = []
    u, v, hist = ts.combine(hist, *g[0])
    assert len(hist) == 1 and u is g[0][0] and v is g[0][1]                   # AB1: the tendency itself
    u, v, hist = ts.combine(hist, *g[1])
    assert len(hist) == 2 and np.array_equal(u, 1.5 * g[1][0] - 0.5 * g[0][0])
    u, v, hist = ts.combine(hist, *g[2])
    assert len(hist) == 3
    assert np.array_equal(v, AB3[0] * g[2][1] + AB3[1] * g[1][1] + AB3[2] * g[0][1])
    _, _, hist = ts.combine(hist, *g[3])
    assert len(hist) == 3 and hist[0][0] is g[3][0] and hist[2][0] is g[1][0]   # newest first, three kept


def test_model_module_reexports_the_same_objects():
    assert sb.ab3_combine is ab3_combine and sb.AB3 is AB3
