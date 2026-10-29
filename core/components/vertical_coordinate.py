"""
ALE vertical coordinate of the 3-D core (N1, role VERTICAL_COORDINATE), moved unchanged from ``SigmaBaroclinic``.

The rule ('sigma', 'zstar', 'hybrid'; ``core.vertical.regrid``) builds the target layer interfaces; fields are
remapped conservatively onto them (``core.vertical.remap``, PCM/PLM/PPM_H4 as MOM6 remapping_core_h); face
geometry (upwind transport thicknesses clipped at the shallower bed for z*/hybrid as HYCOM cnuity; the
direction-independent mean), the remap of face velocities in fraction space (Σ w u conserved) and the CFL
truncation backstop (MOM6 vertvisc_limit_vel).  The face-thickness cache keyed by the identity of h is a rebuildable
cache (thickness arrays are never written in place), the only state the component keeps.

The σ (Eulerian) coordinate needs none of this: ``SigmaBaroclinic`` keeps its D Δσ geometry and builds no
component (spec 2026-10-28 §4: phases 8–9 do nothing in σ).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numba
import numpy as np

from core.dynamics import vector_invariant as vi
from core.model import threads
from core.model.contract import BuildContext, Issue, Role
from core.model.periodic import NONE
from core.vertical import regrid as rg
from core.vertical import remap as rm
from core.vertical.ale import face_fractions, face_transport_thickness


class ALECoordinate:
    name = "ale_coordinate"
    role = Role.VERTICAL_COORDINATE
    reads = ("h", "T", "S", "eta")
    writes = ("h",)
    per = NONE                       # periodic axes (core.model.periodic), set by the model

    def __init__(self, *, H: np.ndarray, active: np.ndarray, n_layers: int, rule: str, params: dict, remap_code,
                 h_thin_bed: float, max_truncations: int, vi_geom: dict, eos) -> None:
        self.H, self.active, self.n_layers = H, active, n_layers
        self.regrid_rule, self.regrid_params, self.remap_code = rule, params, remap_code
        self.h_thin_bed, self.max_truncations, self.vi_geom, self.eos = h_thin_bed, max_truncations, vi_geom, eos
        self._tt_cache: tuple = ()

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}

    def initial_h(self, eta: np.ndarray) -> np.ndarray:
        """
        ALE layer thicknesses of the initial grid at η [m]; 0 on dry/land.  'sigma' and 'zstar': the rule's
        grid (σ where H ≤ 0).  'hybrid': σ-layers, on which T is sampled before ``init_iterations``
        regrid–remap passes build the isopycnal grid, as MOM6 (THICKNESS_CONFIG = "uniform",
        MOM_state_initialization.F90:271; REGRID_ACCELERATE_INIT, :488-507).  The z* floor of dz_top
        (e.g. 19 × 0.5 m + one 190 m bed layer) would lose a continuous stratification.
        """
        H = np.nan_to_num(np.asarray(self.H, float))
        wet = self.active & (H + eta > 0)
        h0 = np.zeros((self.n_layers,) + H.shape)
        rule = "sigma" if self.regrid_rule == "hybrid" else self.regrid_rule
        sig = wet & (H <= 0) if rule == "zstar" else np.zeros_like(wet)
        numba.set_num_threads(threads._THREADS[0])                                 # parallel regrid kernel
        for r, mask in ((rule, wet & ~sig), ("sigma", sig)):
            if mask.any():
                zw = rg.regrid_columns(r, H, eta, h0, None, self.regrid_params, mask)
                h0 = np.where(mask[None], np.diff(zw, axis=0), h0)
        return h0

    def regrid_remap(self, eta: np.ndarray, h_src: np.ndarray, fields: list, T: np.ndarray,
                     S: np.ndarray | None = None):
        """
        Regrid with the rule (spec §3) at surface η and remap ``fields`` (spec §4): (h_new, remapped).
        T [°C] and S [g kg⁻¹] (None without salinity) give the layer densities of the hybrid rule.
        """
        H = np.nan_to_num(np.asarray(self.H, float))
        wet = self.active & (H + eta > 0) & (h_src.sum(axis=0) > 0)
        h_new = self.regrid(eta, h_src, T, wet, S)
        return h_new, [rm.remap_columns(h_src, f, h_new, self.remap_code, wet) for f in fields]

    def regrid(self, eta: np.ndarray, h: np.ndarray, T: np.ndarray, wet: np.ndarray | None = None,
               S: np.ndarray | None = None) -> np.ndarray:
        """
        New layer thicknesses of the rule (spec §3) from layers h with temperatures T [°C] and salinities S
        [g kg⁻¹] (None without salinity); h kept where dry.  The hybrid rule follows the isopycnals of
        ρ(T, S) (N2a; linear EOS, Vallis 2006 eq. 1.156).
        """
        H = np.nan_to_num(np.asarray(self.H, float))
        if wet is None:
            wet = self.active & (H + eta > 0) & (h.sum(axis=0) > 0)
        rho = self.eos.density(T, S) if self.regrid_rule == "hybrid" else None
        numba.set_num_threads(threads._THREADS[0])                                 # parallel regrid kernel
        zw = rg.regrid_columns(self.regrid_rule, H, eta, h, rho, self.regrid_params, wet)
        return np.where(wet[None], np.diff(zw, axis=0), h)

    def transport_thickness(self, h: np.ndarray):
        """
        Upwind face thicknesses ((q_E, q_W) at u-faces, (q_N, q_S) at v-faces swapped to (N, nx, ny+1)):
        clipped at the shallower bed for z* and hybrid, not for σ (core.vertical.ale).  Cached by array
        identity for the last two thickness arrays (hⁿ is used by the tendencies and the step; h_new of one
        step is hⁿ of the next); thickness arrays are never written in place after creation.
        """
        for ref, val in self._tt_cache:
            if ref is h:
                return val
        clip = self.regrid_rule != "sigma"
        H, act = np.nan_to_num(np.asarray(self.H, float)), self.active
        if self.per:                                                   # periodic: pad–crop, halo 1 (two cells)
            p, sw = self.per, (lambda a: a.swapaxes(1, 2))             # noqa: E731
            hp, Hp, ap = p.pad(h, "c", 1), p.pad(H, "c", 1), p.pad(act, "c", 1)
            x = tuple(p.crop(q, "u", 1) for q in face_transport_thickness(hp, Hp, ap, clip))
            y = tuple(sw(p.crop(sw(q), "v", 1)) for q in face_transport_thickness(sw(hp), Hp.T, ap.T, clip))
        else:
            x = face_transport_thickness(h, H, act, clip)
            y = face_transport_thickness(h.swapaxes(1, 2), H.T, act.T, clip)
        val = (x, y)
        self._tt_cache = ((h, val),) + tuple(self._tt_cache)[:1]
        return val

    def mean_face_thickness(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Direction-independent face thickness (q_E + q_W)/2 at u- and v-faces [m]."""
        (qe, qw), (qn, qs) = self.transport_thickness(h)
        return 0.5 * (qe + qw), (0.5 * (qn + qs)).swapaxes(1, 2)

    def remap_faces(self, vel: np.ndarray, hf_src: np.ndarray, hf_tgt: np.ndarray,
                     w_tgt: np.ndarray | None = None) -> np.ndarray:
        """
        Remap face velocities between two face-layer geometries in fraction space: thicknesses
        w_src S → w_tgt S with the same column total S, so Σ w u (the depth mean of the free
        surface) is conserved exactly (spec §2 step 4).  ``w_tgt``: face_fractions(hf_tgt) if already known.
        """
        tot = hf_src.sum(axis=0)
        wet = tot > 0
        S = np.where(wet, tot, 1.0)[None]
        w_tgt = face_fractions(hf_tgt) if w_tgt is None else w_tgt
        return rm.remap_columns(face_fractions(hf_src) * S, vel, w_tgt * S, self.remap_code, wet)

    def truncate(self, u_new: np.ndarray, v_new: np.ndarray, hu1: np.ndarray, hv1: np.ndarray, dt: float,
                 n_trunc_old: int) -> tuple[np.ndarray, np.ndarray, int]:
        """
        CFL velocity truncation backstop (MOM6 vertvisc_limit_vel): faces whose velocity would empty the cell in one
        step are truncated and counted; more than ``max_truncations`` in the run raises with the first face.
        """
        g = self.vi_geom                                                    # CFL truncation backstop (MOM6 vertvisc_limit_vel)
        u_before, v_before = u_new, v_new
        if self.per:                                                   # donor across the wrap face (N3 task 3)
            u_new, nu_, fu = vi.truncate_cfl(u_new, dt, g["dx_t"], hu1, self.h_thin_bed, periodic=self.per.x)
            v_sw, nv_, fv = vi.truncate_cfl(v_new.swapaxes(1, 2), dt, g["dy_t"].T, hv1.swapaxes(1, 2),
                                            self.h_thin_bed, periodic=self.per.y)
        else:
            u_new, nu_, fu = vi.truncate_cfl(u_new, dt, g["dx_t"], hu1, self.h_thin_bed)
            v_sw, nv_, fv = vi.truncate_cfl(v_new.swapaxes(1, 2), dt, g["dy_t"].T, hv1.swapaxes(1, 2),
                                            self.h_thin_bed)
        v_new = v_sw.swapaxes(1, 2)
        n_trunc = n_trunc_old + nu_ + nv_
        if n_trunc > self.max_truncations:
            if fu is not None:
                k, j, i = fu
                where = (f"layer {k} at u-face (j={j}, i={i}): u = {u_before[k, j, i]:.3g} m/s, "
                         f"face thickness {hu1[k, j, i]:.3g} m")
            else:
                k, i, j = fv
                where = (f"layer {k} at v-face (j={j}, i={i}): v = {v_before[k, j, i]:.3g} m/s, "
                         f"face thickness {hv1[k, j, i]:.3g} m")
            raise RuntimeError(f"{n_trunc} CFL velocity truncations exceed max_truncations={self.max_truncations}; "
                               f"first this step: {where}")
        return u_new, v_new, n_trunc
