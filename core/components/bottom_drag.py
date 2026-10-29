"""
Bottom drag of the 3-D core (N1, role BOTTOM_DRAG).

``QuadraticBedDrag`` (ALE): linear bed rates r = C_d √(ū_bbl² + v̄_bbl² + u_bg²) [m s⁻¹] at u- and v-faces from the
mean velocity of the bottom ``hbbl`` metres, and the fractions f_k that spread the stress r ū_bbl = Σ_k r f_k u_k
over those metres (spec 2026-10-14-dome §3; MOM6 MOM_vert_friction.F90:2266 couples the BBL over
min(dh, bbl_thick)).  The rates enter the implicit vertical viscosity as per-layer Rayleigh terms.

The σ law-of-the-wall drag (C_d(z₀) on the bed layer, Soulsby 1997) is applied inside the free-surface wrapper
(``SigmaWetDry.column_step``) in N1.  Moved unchanged from ``SigmaBaroclinic._bed_drag_rates``.

``LogLawBedDrag`` (ALE with ``z0``, N3 task 4): the first-layer law of the wall of σ (``first_layer_cd``,
CROCO / Soulsby 1997 §3) on a bed layer that cannot vanish.  Its thickness is

    d_b = max( h_b , min(hbbl, D_f) ),

h_b the thickness of the lowest face layer with mass (h > ``h_neglect``), D_f the face column; the drag acts on the
thickness-weighted mean velocity of the bottom d_b metres, weights min(h_k, d_b − Σ_{j<k} h_j)⁺ / d_b with the
vanished layers skipped, as MOM6 averages the velocity of the bottom HBBL for its quadratic drag
(MOM_set_viscosity.F90:651-685, MOM6 0e9a561: ``hweight = MIN(Hbbl − htot_vel, h_at_vel)``, layers thinner than
1.5 Angstrom skipped), and C_d = [κ / ln(d_b / 2 z₀)]² (``first_layer_cd(d_b)``).  So

* a bed layer with mass at least ``hbbl`` thick carries the whole stress with C_d(h_b): exactly σ's first-layer
  law (``test_ale_loglaw_matches_sigma_column``);
* a thin or vanished bed layer (z* under a shallow column, a thin hybrid bed layer) does not set C_d alone:
  C_d ≤ C_d(hbbl) and the stress is spread over the layers of the bottom hbbl metres (as ``QuadraticBedDrag``);
* d_b is continuous in h_b (at h_b = hbbl both branches give the bed layer alone).

The stress is linearised as in σ, r = C_d |U_b*| with U_b* the bbl mean of u + Δt G (the velocity the implicit
solve starts from; σ: ``SigmaWetDry.column_step``), and spread as per-layer Rayleigh rates r_k = r w_k.  The speed is
the full bed speed (N3 task 9b, class-(a) fix): |U_b| = √(u_b² + v̄_b²) at u-faces, v̄_b the 4-point average of the
v-face bbl means (``physics.bottom_friction.cross_velocity_at_faces``), as CROCO get_vbc.F:517-528 and MOM6
MOM_set_viscosity.F90:684-688 (``set_v_at_u``, :1887-1929); before 9b each face used its own component, C_d |u| u.
"""
from __future__ import annotations

import numpy as np

from core.model.contract import BuildContext, Issue, Role
from core.model.periodic import NONE
from physics.bottom_friction import cross_velocity_at_faces, first_layer_cd


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
        v_at_u, u_at_v = cross_velocity_at_faces(ub, vb, self.per.x, self.per.y)
        bg2 = self.drag_bg_vel**2
        return self.cd * np.sqrt(ub**2 + v_at_u**2 + bg2), self.cd * np.sqrt(vb**2 + u_at_v**2 + bg2), fu, fv

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}


def bed_ustar2(tau_u: np.ndarray, tau_v: np.ndarray) -> np.ndarray:
    """
    u*²_b = |τ_b|/ρ₀ (ny, nx) [m² s⁻²] at the cell centres from the kinematic bed stress at u-faces (ny, nx+1) and
    v-faces (ny+1, nx), each averaged from its two faces, as ``SigmaWetDry.closure_forcing`` (CROCO gls_mixing.F).
    With periodic axes the wrap faces are stored twice and equal, so cells 0 and n−1 see the shared face.
    """
    tx = 0.5 * (tau_u[:, :-1] + tau_u[:, 1:])
    ty = 0.5 * (tau_v[:-1, :] + tau_v[1:, :])
    return np.hypot(tx, ty)


class LogLawBedDrag:
    """
    Law-of-the-wall bed drag of the ALE mode (N3 task 4; rule in the module docstring).

    z0 : bed roughness length [m]
    hbbl : bottom boundary-layer thickness of the minimum bed layer [m] (the ``hbbl`` of the quadratic drag)
    cd_max : upper bound of C_d (``SigmaWetDry.cd_max``, CROCO Cdb_max)
    h_neglect : thinner face layers are vanished: no weight, not the bed layer [m] (``core.vertical.ale.DZ_FLOOR``)
    """
    name = "loglaw_bed_drag"
    role = Role.BOTTOM_DRAG
    reads = ("u", "v", "h")
    writes = ()
    per = NONE                       # periodic axes of the 4-point cross average (core.model.periodic), set by the model

    def __init__(self, z0: float, hbbl: float, cd_max: float = 0.1, h_neglect: float = 1e-6) -> None:
        self.z0, self.hbbl, self.cd_max, self.h_neglect = float(z0), float(hbbl), float(cd_max), float(h_neglect)

    def bed_layer(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        (d_b, w, C_d) of face columns h (N, ...) [m], k = 0 at the bed: the bed-layer thickness d_b (...) [m], the
        weights w (N, ...) of the bottom d_b metres (Σ_k w_k = 1; a face without water: w_0 = 1) and C_d(d_b) [–].
        """
        hm = np.where(h > self.h_neglect, h, 0.0)                     # vanished layers skipped (MOM6 Angstrom)
        has = hm > 0
        kb = np.argmax(has, axis=0)                                   # lowest layer with mass
        hb = np.take_along_axis(hm, kb[None], axis=0)[0]              # 0 on a face without water
        db = np.maximum(hb, np.minimum(self.hbbl, hm.sum(axis=0)))
        below = np.concatenate([np.zeros((1,) + hm.shape[1:]), np.cumsum(hm, axis=0)[:-1]])
        w = np.clip(np.minimum(hm, db - below), 0.0, None)
        wet = db > 0
        w = np.where(wet, w / np.where(wet, db, 1.0), 0.0)
        w[0] = np.where(wet, w[0], 1.0)
        cd = first_layer_cd(np.where(wet, db, 0.0), self.z0, cd_max=self.cd_max)   # dry: cd_max (no water)
        return db, w, cd

    def rates(self, uu: np.ndarray, vv: np.ndarray, hu: np.ndarray, hv: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Per-layer linear bed rates r_k = C_d |U_b| w_k at u- and v-faces (N, ny, nx+1), (N, ny+1, nx) [m s⁻¹] for
        the implicit vertical viscosity (``SigmaWetDry.column_step`` ``r_bed``), with u_b = Σ_k w_k uu_k the bbl
        mean of the face's own component and the full bed speed |U_b| = √(u_b² + v̄_b²) (u-faces; v-faces
        √(v_b² + ū_b²)), the cross component the 4-point average of the other faces' bbl means, wrapped on periodic
        axes (CROCO get_vbc.F:517-528).  With v ≡ 0, |U_b| = |u_b| exactly (``np.hypot``): the old rates bit for bit.
        uu, vv : velocities the implicit solve starts from (u + Δt G) [m s⁻¹]; hu, hv : face layer thicknesses (the
        mean, clipped face column of ``face_fractions``) [m].
        """
        (_, wu, cdu), (_, wv, cdv) = self.bed_layer(hu), self.bed_layer(hv)
        ub = np.nan_to_num((wu * uu).sum(axis=0))
        vb = np.nan_to_num((wv * vv).sum(axis=0))
        v_at_u, u_at_v = cross_velocity_at_faces(ub, vb, self.per.x, self.per.y)
        return (cdu * np.hypot(ub, v_at_u))[None] * wu, (cdv * np.hypot(vb, u_at_v))[None] * wv

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}
