"""
Interior sponge of the 3-D core (N1, role INTERIOR_SPONGE), moved unchanged from ``SigmaBaroclinic``: implicit
relaxation of T, the salinity S (``salinity=True``, N2a) and the passive tracers toward a profile in z, remapped onto the current layers, after the remap
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
    """
    r (ny, nx) [s⁻¹]; T* (nz_s,) or per column (nz_s, ny, nx) on interfaces zw (nz_s+1,) or (nz_s+1, ny, nx) [m],
    surface first and decreasing; tr* (ntr, nz_s) or (ntr, nz_s, ny, nx); S* shaped as T* [g kg⁻¹] or None.

    Per-column targets (ruling 2026-11-04) follow MOM6's ALE sponge: each column keeps its own target on its own
    z grid and is remapped onto the model layers every call (MOM6 0e9a561 MOM_ALE_sponge.F90:
    set_up_ALE_sponge_field_fixed 699-755 stores p(nz_data, col) per column; apply_ALE_sponge 1035-1066 calls
    remapping_core_h with the column's source thicknesses and applies the backward-Euler
    (c + Δt r c*)/(1 + Δt r)).  Unlike MOM6 the source column is stretched to the model column total H + η
    (as the 1-D path and core.vertical.ale.exterior_on_layers) instead of extending its last cell to the bed
    (MOM_ALE_sponge.F90:1021); a profile (nz_s,) is the per-column target broadcast, bit for bit.
    """
    name = "interior_sponge"
    role = Role.INTERIOR_SPONGE
    reads = ("T", "S", "tr", "h", "eta")
    writes = ("T", "S", "tr")

    def __init__(self, *, rate: np.ndarray, T: np.ndarray, zw: np.ndarray, tr: np.ndarray, cols: np.ndarray,
                 H: np.ndarray, dt: float, remap_code, S: np.ndarray | None = None) -> None:
        self.rate, self.T, self.zw, self.tr, self.cols = rate, T, zw, tr, cols
        self.H, self.dt, self.remap_code = H, dt, remap_code
        self.S = S

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
        relax = self._relaxer(eta, h)
        T = relax(T, self.T)
        if tr is not None:
            tr = np.stack([relax(c, p) for c, p in zip(tr, self.tr)])
        return T, tr

    def apply_S(self, S: np.ndarray, eta: np.ndarray, h: np.ndarray) -> np.ndarray:
        """The same implicit relaxation of the salinity toward S* (``sponge_S``; N2a), S unchanged without S*."""
        if self.S is None:
            return S
        return self._relaxer(eta, h)(S, self.S)

    def _relaxer(self, eta: np.ndarray, h: np.ndarray):
        """relax(c, profile): c relaxed toward ``profile`` (nz_s,) or (nz_s, ny, nx) remapped onto the layers h
        (see ``apply``).  Columns outside the sponge (r = 0, land, dry) are not remapped (``ok``), so NaN there is
        harmless."""
        H = np.nan_to_num(np.asarray(self.H, float))
        ok = self.cols & (H + eta > 0) & (h.sum(axis=0) > 0)
        zw = self.zw if self.zw.ndim == 3 else self.zw[:, None, None]  # per column or one profile
        z = np.maximum(zw, -H[None])                                    # surface first, clipped to each bed
        z[-1] = -H
        hs = (z[:-1] - z[1:])[::-1]                                         # bed first
        tot_s, tot = hs.sum(axis=0), h.sum(axis=0)
        scale = np.where(ok & (tot_s > 0), tot / np.where(tot_s > 0, tot_s, 1.0), 1.0)
        hs = np.ascontiguousarray(hs * scale[None])
        r = self.dt * self.rate[None]
        numba.set_num_threads(threads._THREADS[0])

        def relax(c, prof):
            src = prof[::-1] if prof.ndim == 3 else prof[::-1, None, None] * np.ones((1,) + H.shape)
            tgt = rm.remap_columns(hs, np.ascontiguousarray(src), np.ascontiguousarray(h), self.remap_code, ok)
            return np.where(ok[None], (c + r * tgt) / (1.0 + r), c)
        return relax
