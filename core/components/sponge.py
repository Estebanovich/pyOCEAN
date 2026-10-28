"""
Interior sponge of the 3-D core (N1, role INTERIOR_SPONGE), moved unchanged from ``SigmaBaroclinic``: implicit
relaxation of T and the passive tracers toward a profile in z, remapped onto the current layers, after the remap
(spec 2026-10-14-dome §2; MOM6 MOM_sponge.F90 apply_sponge, implicit form).  It stays out of the mixing
diagnostic.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numba
import numpy as np

from core.model import threads
from core.model.contract import BuildContext, Issue, Role
from core.vertical import remap as rm


class InteriorSponge:
    """r (ny, nx) [s⁻¹]; T* (nz_s,) on interfaces zw (nz_s+1,) [m] decreasing from the surface; tr* (ntr, nz_s)."""
    name = "interior_sponge"
    role = Role.INTERIOR_SPONGE
    reads = ("T", "tr", "h", "eta")
    writes = ("T", "tr")

    def __init__(self, *, rate: np.ndarray, T: np.ndarray, zw: np.ndarray, tr: np.ndarray, cols: np.ndarray,
                 H: np.ndarray, dt: float, remap_code) -> None:
        self.rate, self.T, self.zw, self.tr, self.cols = rate, T, zw, tr, cols
        self.H, self.dt, self.remap_code = H, dt, remap_code

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}

    def apply(self, T: np.ndarray, tr: np.ndarray | None, eta: np.ndarray, h: np.ndarray):
        """
        Implicit relaxation toward the sponge profile on the layers h (spec 2026-10-14-dome §2):
            T ← (T + Δt r T*)/(1 + Δt r)   (MOM6 MOM_sponge.F90 apply_sponge, implicit form)
        T* is sponge_T on its z interfaces clipped at each bed, stretched to the column total H + η (as the exterior
        profile, core.vertical.ale.exterior_on_layers) and remapped conservatively onto h.
        """
        H = np.nan_to_num(np.asarray(self.H, float))
        ok = self.cols & (H + eta > 0) & (h.sum(axis=0) > 0)
        z = np.maximum(self.zw[:, None, None], -H[None])               # surface first, clipped to each bed
        z[-1] = -H
        hs = (z[:-1] - z[1:])[::-1]                                         # bed first
        tot_s, tot = hs.sum(axis=0), h.sum(axis=0)
        scale = np.where(ok & (tot_s > 0), tot / np.where(tot_s > 0, tot_s, 1.0), 1.0)
        hs = np.ascontiguousarray(hs * scale[None])
        r = self.dt * self.rate[None]
        numba.set_num_threads(threads._THREADS[0])

        def relax(c, prof):
            tgt = rm.remap_columns(hs, np.ascontiguousarray(prof[::-1, None, None] * np.ones((1,) + H.shape)),
                                   np.ascontiguousarray(h), self.remap_code, ok)
            return np.where(ok[None], (c + r * tgt) / (1.0 + r), c)
        T = relax(T, self.T)
        if tr is not None:
            tr = np.stack([relax(c, p) for c, p in zip(tr, self.tr)])
        return T, tr
