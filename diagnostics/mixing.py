"""
Spurious (numerical) mixing diagnostics.

1. Reference potential energy (RPE) — Winters et al. (1995); used for model
   intercomparison by Ilicak et al. (2012).  The RPE is the potential energy of
   the state obtained by adiabatically re-sorting all water parcels, heaviest
   at the bottom:

       RPE = g ∫ ρ*(z) z dV,

   with ρ* the sorted density.  Advection (stirring) leaves it unchanged; any
   diapycnal mixing — physical or numerical — increases it.  With all physical
   mixing switched off and a linear EOS, dRPE/dt measures the spurious mixing
   of the model.  Here z is the height above the deepest point, so RPE > 0.

2. Discrete variance decay of one advection step — Burchard & Rennau (2008),
   Burchard (2012):

       M_num = ( A(c²) − (A c)² ) / Δt,

   where A is the discrete advection operator.  For a conservative scheme its
   volume sum equals the loss of ∫c² dV over the step, and it is zero for an
   exact translation.  It is local in space and is computed online at every step
   (offline estimates from model output overestimate it by up to ~60 % at hourly
   output; Schlichting et al. 2023, JAMES, doi:10.1029/2022MS003380).

References
----------
Winters, K.B., Lombard, P.N., Riley, J.J. & D'Asaro, E.A. (1995). Available
    potential energy and mixing in density-stratified fluids. J. Fluid Mech.
    289, 115–128.
Ilicak, M., Adcroft, A.J., Griffies, S.M. & Hallberg, R.W. (2012). Spurious
    dianeutral mixing and the role of momentum closure. Ocean Modell. 45–46,
    37–58, doi:10.1016/j.ocemod.2011.10.003.
Burchard, H. & Rennau, H. (2008). Comparative quantification of physically and
    numerically induced mixing in ocean models. Ocean Modell. 20, 293–311,
    doi:10.1016/j.ocemod.2007.10.003.
Klingbeil, K. et al. (2014). Quantification of spurious dissipation and mixing –
    discrete variance decay in a finite-volume framework. Ocean Modell. 81,
    49–64, doi:10.1016/j.ocemod.2014.06.001.
"""

from __future__ import annotations

from typing import Callable

import numpy as np

G = 9.81


def reference_potential_energy(rho: np.ndarray, volume: np.ndarray, area: float | None = None,
                               hypsometry: tuple[np.ndarray, np.ndarray] | None = None,
                               g: float = G) -> float:
    """
    RPE [J] of the density field ``rho`` [kg m⁻³] in cells of volume ``volume`` [m³].

    Geometry of the re-sorted state (give exactly one):
      area : horizontal area of a flat-bottomed, vertical-walled basin [m²]
      hypsometry : (z, A(z)) — heights above the deepest point [m, increasing]
                   and the horizontal area at those heights [m²]
    """
    if (area is None) == (hypsometry is None):
        raise ValueError("give exactly one of area= or hypsometry=")
    r = np.asarray(rho, float).ravel()
    v = np.asarray(volume, float).ravel()
    ok = np.isfinite(r) & (v > 0)
    r, v = r[ok], v[ok]
    order = np.argsort(-r, kind="stable")                  # heaviest first → bottom
    r, v = r[order], v[order]
    vcum = np.concatenate(([0.0], np.cumsum(v)))           # volume below each parcel's interfaces
    if area is not None:
        # flat basin: z = V/A, ∫ z A dz between z_lo and z_hi = A (z_hi² − z_lo²)/2
        z = vcum / area
        moment = 0.5 * area * (z[1:] ** 2 - z[:-1] ** 2)
    else:
        zl, al = (np.asarray(a, float) for a in hypsometry)
        vol_z = np.concatenate(([0.0], np.cumsum(0.5 * (al[1:] + al[:-1]) * np.diff(zl))))
        mom_z = np.concatenate(([0.0], np.cumsum(0.5 * (zl[1:] * al[1:] + zl[:-1] * al[:-1]) * np.diff(zl))))
        if vcum[-1] > vol_z[-1] * (1 + 1e-12):
            raise ValueError("hypsometry holds less volume than the water")
        f = np.interp(vcum, vol_z, mom_z)                  # ∫ z A dz up to the interface
        moment = np.diff(f)
    return float(g * np.sum(r * moment))


def numerical_variance_decay(advect: Callable[[np.ndarray], np.ndarray], c: np.ndarray, dt: float) -> np.ndarray:
    """
    Local numerical variance decay M_num = (A(c²) − (A c)²)/Δt [c² s⁻¹] of one
    advection step ``advect`` (same operator, velocities and fluxes for c and c²;
    Burchard & Rennau 2008).  Multiply by the cell volume and sum for the global rate.
    """
    c = np.asarray(c, float)
    return (advect(c * c) - advect(c) ** 2) / dt
