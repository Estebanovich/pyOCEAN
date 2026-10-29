"""
Bottom drag of the 3-D core (N1, role BOTTOM_DRAG).

``QuadraticBedDrag`` (ALE): linear bed rates r = C_d √(ū_bbl² + v̄_bbl² + u_bg²) [m s⁻¹] at u- and v-faces from the
mean velocity of the bottom ``hbbl`` metres, and the fractions f_k that spread the stress r ū_bbl = Σ_k r f_k u_k
over those metres (spec 2026-10-14-dome §3; MOM6 MOM_vert_friction.F90:2266 couples the BBL over
min(dh, bbl_thick)).  The rates enter the implicit vertical viscosity as per-layer Rayleigh terms.

The σ law-of-the-wall drag (C_d(z₀) on the bed layer, Soulsby 1997) is applied inside the free-surface wrapper
(``SigmaWetDry.column_step``) in N1.  Moved unchanged from ``SigmaBaroclinic._bed_drag_rates``.
"""
from __future__ import annotations

import numpy as np

from core.model.contract import BuildContext, Issue, Role
from core.model.periodic import NONE


class QuadraticBedDrag:
    name = "quadratic_bed_drag"
    role = Role.BOTTOM_DRAG
    reads = ("u", "v", "h")
    writes = ()
    per = NONE                       # periodic axes (core.model.periodic), set by the model

    def __init__(self, cd: float, drag_bg_vel: float, hbbl: float) -> None:
        self.cd, self.drag_bg_vel, self.hbbl = float(cd), float(drag_bg_vel), float(hbbl)

    def rates(self, u: np.ndarray, v: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """(r_u, r_v, f_u, f_v): bed rates at u- and v-faces [m s⁻¹] and bbl fractions (N, ...), k = 0 at the bed;
        ū_bbl: mean of the face column over its bottom hbbl metres, weights min(h_k, hbbl − Σ_{j<k} h_j)⁺; the other
        component is the 4-point C-grid average of its own bbl means."""
        def bbl_weights(h):
            below = np.concatenate([np.zeros((1,) + h.shape[1:]), np.cumsum(h, axis=0)[:-1]])
            w = np.clip(np.minimum(h, self.hbbl - below), 0.0, None)
            tot = w.sum(axis=0)
            return np.where(tot > 0, w / np.where(tot > 0, tot, 1.0), 0.0), tot > 0
        (fu, wet_u), (fv, wet_v) = bbl_weights(hu), bbl_weights(hv)
        ub, vb = (fu * u).sum(axis=0), (fv * v).sum(axis=0)        # 0 where the face has no bbl water
        fu[0], fv[0] = np.where(wet_u, fu[0], 1.0), np.where(wet_v, fv[0], 1.0)   # there the stress acts on k = 0
        vc = 0.5 * (vb[:-1] + vb[1:])                       # v̄ at cell centres (ny, nx)
        uc = 0.5 * (ub[:, :-1] + ub[:, 1:])                 # ū at cell centres (ny, nx)
        v_at_u = np.zeros_like(ub)
        v_at_u[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:])
        u_at_v = np.zeros_like(vb)
        u_at_v[1:-1, :] = 0.5 * (uc[:-1, :] + uc[1:, :])
        if self.per.x:                                      # wrap faces: cells nx−1 and 0
            v_at_u[:, 0] = v_at_u[:, -1] = 0.5 * (vc[:, -1] + vc[:, 0])
        if self.per.y:
            u_at_v[0, :] = u_at_v[-1, :] = 0.5 * (uc[-1, :] + uc[0, :])
        bg2 = self.drag_bg_vel**2
        return self.cd * np.sqrt(ub**2 + v_at_u**2 + bg2), self.cd * np.sqrt(vb**2 + u_at_v**2 + bg2), fu, fv

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}
