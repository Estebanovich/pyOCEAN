"""
Layer fractions Δσ_k of a stretched σ-coordinate (k = 0 at the bed, Σ Δσ_k = 1).

In a σ-layer model the layer thicknesses are h_k = D Δσ_k with D = H + η the
total depth, so the fractions alone set where the vertical resolution goes.

Stretching of Song & Haidvogel (1994) — the S-coordinate of SCRUM/ROMS with
h_c = 0 (pure σ), as written by Shchepetkin & McWilliams (2003, eq. 4.3):

    C(s) = (1 − θ_b) sinh(θ_s s)/sinh(θ_s)
           + θ_b [ tanh(θ_s (s + ½)) / (2 tanh(θ_s/2)) − ½ ],     −1 ≤ s ≤ 0

with s uniform at the interfaces and σ_w = 1 + C(s).  θ_s (0 … ~8) refines the
surface, θ_b (0 … 1) also refines the bed; θ_s = 0 is uniform.

Fractions can also be taken from a list of z-level thicknesses (MITgcm
``delR``, surface first): the σ-layers then coincide with those z-levels
wherever the depth equals their sum, and are compressed proportionally in
shallower water.

References
----------
Song, Y. & Haidvogel, D. (1994). A semi-implicit ocean circulation model using a
    generalized topography-following coordinate system. J. Comput. Phys. 115, 228–244.
Shchepetkin, A.F. & McWilliams, J.C. (2003). J. Geophys. Res. 108(C3), 3090.
"""

from __future__ import annotations

import numpy as np


def stretched_sigma(n: int, theta_s: float, theta_b: float = 0.0) -> np.ndarray:
    """Δσ_k (k = 0 at the bed) of ``n`` layers with Song & Haidvogel (1994) stretching."""
    if not 0.0 <= theta_b <= 1.0:
        raise ValueError(f"theta_b must be in [0, 1], got {theta_b}")
    s = -1.0 + np.arange(n + 1) / n                       # interfaces, bed → surface
    if theta_s == 0.0:
        C = s
    else:
        C = ((1.0 - theta_b) * np.sinh(theta_s * s) / np.sinh(theta_s)
             + theta_b * (np.tanh(theta_s * (s + 0.5)) / (2.0 * np.tanh(0.5 * theta_s)) - 0.5))
    ds = np.diff(1.0 + C)
    return ds / ds.sum()


def sigma_from_thicknesses(dz_top_first) -> np.ndarray:
    """Δσ_k (k = 0 at the bed) proportional to z-level thicknesses listed from the surface down [m]."""
    dz = np.asarray(dz_top_first, float)
    if dz.ndim != 1 or np.any(~np.isfinite(dz)) or np.any(dz <= 0):
        raise ValueError("layer thicknesses must be a 1-D list of positive numbers")
    return dz[::-1] / dz.sum()
