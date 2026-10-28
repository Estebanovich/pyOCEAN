"""
Time stepping of the explicit momentum tendencies in the 3-D core (N1, role TIME_STEPPER).

Third-order Adams–Bashforth (Durran 2010, §2.4.2), started with AB1 and AB2 (as MITgcm adams_bashforth3.F:85-103
ramps its weights):

    G* = 23/12 Gⁿ − 16/12 Gⁿ⁻¹ + 5/12 Gⁿ⁻²     (AB2 start-up: 3/2 Gⁿ − 1/2 Gⁿ⁻¹)

Only the momentum uses AB3; the tracers have their own scheme (``tracer_time`` 'ssprk3' or 'pc', SigmaBaroclinic),
spec 2026-10-28 §8.6.  The wind stress is not extrapolated.  The history ``g_hist`` (newest first, at most three
pairs) travels in ``SigmaState``.  Moved unchanged from ``core.dynamics.sigma_baroclinic``.
"""
from __future__ import annotations

import numba
import numpy as np

from core.model import threads
from core.model.contract import BuildContext, Issue, Role

AB3 = (23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0)


@numba.njit(parallel=True, cache=True)
def _ab3_kernel(g0, g1, g2, three, out):  # pragma: no cover
    """AB3[0] g0 + AB3[1] g1 + AB3[2] g2 (three) or 1.5 g0 − 0.5 g1 (AB2 start-up), NumPy's evaluation order.
    1-D arrays (flattened by the caller: ravel inside a parallel kernel may copy, losing the writes)."""
    a0, a1, a2 = 23.0 / 12.0, -16.0 / 12.0, 5.0 / 12.0
    for p in numba.prange(out.size):
        if three:
            out[p] = a0 * g0[p] + a1 * g1[p] + a2 * g2[p]
        else:
            out[p] = 1.5 * g0[p] - 0.5 * g1[p]


def ab3_combine(g0: np.ndarray, g1: np.ndarray, g2: np.ndarray | None) -> np.ndarray:
    """Adams–Bashforth extrapolation of the explicit tendencies (newest g0): AB3 with g2, AB2 without
    (step's start-up).  Fused kernel with the same arithmetic as the NumPy expressions."""
    if np.shape(g1) != np.shape(g0) or (g2 is not None and np.shape(g2) != np.shape(g0)):
        raise ValueError(f"ab3_combine: tendency shapes differ {np.shape(g0)}, {np.shape(g1)}, "
                         f"{None if g2 is None else np.shape(g2)}")       # the kernel does not check bounds
    out = np.empty(np.shape(g0))
    g0, g1 = np.ascontiguousarray(g0, float), np.ascontiguousarray(g1, float)
    numba.set_num_threads(threads._THREADS[0])
    g2 = g0 if g2 is None else np.ascontiguousarray(g2, float)
    _ab3_kernel(g0.ravel(), g1.ravel(), g2.ravel(), g2 is not g0, out.ravel())
    return out


class AB3Momentum:
    """AB3 (AB1, AB2 at start-up) on the explicit momentum tendencies; owns no state (history is in the state)."""
    name = "ab3_momentum"
    role = Role.TIME_STEPPER
    reads = ("g_hist",)
    writes = ("g_hist",)

    def combine(self, g_hist: list, gu: np.ndarray, gv: np.ndarray) -> tuple[np.ndarray, np.ndarray, list]:
        """(G*_u, G*_v, new history): the history is [(gu, gv)] + the two newest of ``g_hist``."""
        hist = [(gu, gv)] + g_hist[:2]
        if len(hist) == 1:
            gu_ab, gv_ab = gu, gv
        elif len(hist) == 2:
            gu_ab, gv_ab = ab3_combine(gu, hist[1][0], None), ab3_combine(gv, hist[1][1], None)
        else:
            gu_ab = ab3_combine(gu, hist[1][0], hist[2][0])
            gv_ab = ab3_combine(gv, hist[1][1], hist[2][1])
        return gu_ab, gv_ab, hist

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}
