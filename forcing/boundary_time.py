"""
Time interpolation of open-boundary exterior states (spec 2026-10-12-baroclinic-obc-internal-wave §1.3): a list
of {edge: OBData} snapshots on the same grids → a function t → {edge: OBData}, linear in time and constant
beyond the ends.  The basis of the HYCOM/ESPC boundary adapters (xarray stays at the I/O boundary).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from boundaries.obcs import OBData

_FIELDS = ("T", "S", "u_normal", "eta", "ubar", "u_tangential")


def interpolated_boundary(times: Sequence[float], states: Sequence[dict[str, OBData]]
                          ) -> Callable[[float], dict[str, OBData]]:
    """Linear-in-time exterior state between snapshots ``states`` at strictly increasing ``times`` [s]."""
    t = np.asarray(times, float)
    if t.ndim != 1 or t.size != len(states) or t.size == 0 or (t.size > 1 and not (np.diff(t) > 0).all()):
        raise ValueError("times must be strictly increasing, one per state")
    edges = set(states[0])
    if any(set(s) != edges for s in states):
        raise ValueError("every state must have the same edges")

    def at(time: float) -> dict[str, OBData]:
        k = int(np.clip(np.searchsorted(t, time, side="right") - 1, 0, t.size - 1))
        k1 = min(k + 1, t.size - 1)
        w = 0.0 if k1 == k else float(np.clip((time - t[k]) / (t[k1] - t[k]), 0.0, 1.0))
        out = {}
        for e in edges:
            a, b = states[k][e], states[k1][e]
            vals = {f: None if getattr(a, f) is None else (1 - w) * getattr(a, f) + w * getattr(b, f) for f in _FIELDS}
            out[e] = OBData(side=a.side, zw=a.zw, **vals)
        return out
    return at
