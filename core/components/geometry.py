"""
Fixed geometry of the 3-D core passed to the components (N1): bed depth, σ fractions, the C-grid face depths and
spacings of the free surface, and the layer-height function of the vertical coordinate (with its cache).

Index order (k, j, i), k = 0 at the bed; u-faces (ny, nx+1), v-faces (ny+1, nx).
"""
from __future__ import annotations

from collections.abc import Callable
from typing import NamedTuple

import numpy as np


class LayerGeometry(NamedTuple):
    H: np.ndarray                    # bed depth at cell centres [m], NaN = wall
    ds: np.ndarray                   # σ-layer fractions Δσ_k (N,), bed first
    sigma_c: np.ndarray              # σ at layer centres (N,), bed = 0
    d_min: float                     # minimum total depth [m]
    H_u: np.ndarray                  # bed depth at u-faces [m], NaN = closed
    H_v: np.ndarray                  # bed depth at v-faces [m], NaN = closed
    du: np.ndarray                   # centre-to-centre distance across u-faces [m]
    dv: np.ndarray                   # centre-to-centre distance across v-faces [m]
    layer_heights: Callable          # (eta, h) → layer-centre heights (N, ny, nx) [m], cached by identity of h
