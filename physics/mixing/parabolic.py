"""
Parabolic eddy viscosity of a tidal (bottom-driven) boundary layer.

For a homogeneous, unstratified flow driven by the bed stress the mixing-length
closure ℓ = κ z (1 − z/D)^{1/2} with a linear stress profile gives

    ν(z) = κ u* z (1 − z/D)                (Rodi 1993; Bowden 1978; Davies & Jones 1988)

which together with τ_b = ρ₀ u*² yields a logarithmic velocity profile near the
bed.  It is the zero-equation limit of k-ε/GLS (Umlauf & Burchard 2003) in
equilibrium and a standard first closure for barotropic tidal flows.

References
----------
Rodi, W. (1993). Turbulence Models and Their Application in Hydraulics. IAHR.
Umlauf, L. & Burchard, H. (2003). A generic length-scale equation for
    geophysical turbulence models. J. Mar. Res. 61, 235–265.
"""

from __future__ import annotations

import numpy as np

KAPPA = 0.41


def parabolic_viscosity(ustar: np.ndarray, depth: np.ndarray, sigma_w: np.ndarray, nu_min: float = 1e-5,
                        kappa: float = KAPPA) -> np.ndarray:
    """
    ν at the layer interfaces [m² s⁻¹].

    ustar : friction velocity u* = √(τ_b/ρ₀) per column [m s⁻¹], shape (M,)
    depth : total depth D per column [m], shape (M,)
    sigma_w : interface positions σ = z/D from the bed (0) to the surface (1), shape (N+1, 1)
    nu_min : background viscosity (molecular + residual), also at bed and surface
    """
    z = sigma_w * depth
    nu = kappa * ustar * z * (1.0 - sigma_w)
    return np.maximum(nu, nu_min)
