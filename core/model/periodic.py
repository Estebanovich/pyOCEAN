"""
Periodic horizontal boundaries of the 3-D core (N3 task 3, spec 2026-10-29-n3-turbulence-closures §5).

Convention (C grid, index order (k, j, i), u-faces (ny, nx+1), v-faces (ny+1, nx), corners (ny+1, nx+1)):
with ``"x"`` periodic the u-faces of columns 0 and nx are **one physical face**, stored twice and always equal
(the face between cell nx−1 and cell 0); likewise the v-faces of rows 0 and ny with ``"y"``, and the corner
rows/columns.  Every operator computes both copies from the same (wrapped) neighbours, so they stay equal bit for
bit and the flux divergence of cell 0 and of cell nx−1 use the same flux.  Array shapes do not change.

Operators reach the wrap in one of two ways (as MOM6 and MITgcm do with halos, MOM_domains pass_var /
EXCH_XY_RL — here the halo exists only during the call):

* **pad–crop** (``Periodic.pad`` / ``Periodic.crop``): the inputs of a stencil kernel are extended by ``halo``
  wrapped cells/faces on each periodic side, the kernel runs unchanged on the extended arrays (its own edge
  treatment acts on the halo only), and the result is cropped back.  Exact when ``halo`` ≥ the stencil radius of
  the call (e.g. WENO5 + SSP-RK3 + FCT: 3 stages × 3 + 3 → ``TRACER_HALO``);
* **native**: the few NumPy pair operators (free-surface face depth, gradient and matrix, face depths, Jacobian
  PGF, diffusion fluxes, bed-drag and closure averages, CFL donor width) compute the edge face from cells nx−1
  and 0 directly.

Without periodicity nothing in this module is called (bit for bit).  The pressure, CFL and continuity are those
of an infinite periodic domain; physical consistency (periodic bathymetry, f independent of the periodic
direction) is the user's.
"""
from __future__ import annotations

import numpy as np
import xarray as xr

AXES = ("x", "y")
TRACER_HALO = 13        # SSP-RK3 WENO5 (3 × 3 cells) + FCT donor/bounds/coefficients (3) + 1 spare
MOMENTUM_HALO = 4       # flux-form WENO5 on u (3) + transport averaging (1)


def _face_x(kind: str) -> bool:
    return kind in ("u", "q")


def _face_y(kind: str) -> bool:
    return kind in ("v", "q")


class Periodic:
    """
    Which horizontal axes wrap.  ``axes``: a tuple with ``"x"``, ``"y"`` or both (empty: closed or open edges, the
    default).  Falsy when no axis wraps.
    """

    def __init__(self, axes=()) -> None:
        if isinstance(axes, str):
            axes = (axes,)
        axes = tuple(axes or ())
        bad = [a for a in axes if a not in AXES]
        if bad:
            raise ValueError(f"unknown periodic axes {bad}; use 'x', 'y' or both")
        self.x, self.y = "x" in axes, "y" in axes
        self.axes = tuple(a for a in AXES if a in axes)

    def __bool__(self) -> bool:
        return self.x or self.y

    def __repr__(self) -> str:
        return f"Periodic({self.axes})"

    # ------------------------------------------------------------------ halo
    @staticmethod
    def _index(n_cells: int, face: bool, halo: int) -> np.ndarray:
        """Wrapped source index of each position of the extended axis (cells: n + 2·halo; faces: n + 1 + 2·halo)."""
        return np.arange(-halo, n_cells + halo + (1 if face else 0)) % n_cells

    def pad(self, a, kind: str, halo: int) -> np.ndarray:
        """
        ``a`` with ``halo`` wrapped positions on each side of every periodic axis (last axis x, second-to-last y).
        ``kind``: staggering of the two last axes, 'c' (cells), 'u', 'v' or 'q' (corners).  Faces wrap on the
        unique faces 0 … n−1 (face n is face 0).
        """
        a = np.asarray(a)
        if self.x:
            face = _face_x(kind)
            a = np.take(a, self._index(a.shape[-1] - face, face, halo), axis=-1)
        if self.y:
            face = _face_y(kind)
            a = np.take(a, self._index(a.shape[-2] - face, face, halo), axis=-2)
        return a

    def crop(self, a: np.ndarray, kind: str, halo: int) -> np.ndarray:
        """Inverse of ``pad``: the interior positions, contiguous (``kind`` kept for symmetry with ``pad``)."""
        if self.x:
            a = a[..., halo:a.shape[-1] - halo]
        if self.y:
            a = a[..., halo:a.shape[-2] - halo, :]
        return np.ascontiguousarray(a)

    def apply(self, fn, args: tuple, kinds: tuple, out_kinds, halo: int, **kw):
        """
        fn(*args, **kw) on the padded positional ``args`` (``kinds[i]`` None: passed unchanged), outputs cropped
        with ``out_kinds`` (a kind, or a tuple of kinds for a tuple of outputs).
        """
        pa = tuple(a if k is None or a is None else self.pad(a, k, halo) for a, k in zip(args, kinds))
        out = fn(*pa, **kw)
        if isinstance(out_kinds, str):
            return self.crop(out, out_kinds, halo)
        return tuple(self.crop(o, k, halo) for o, k in zip(out, out_kinds))

    def pad_dataset(self, hg: xr.Dataset, halo: int) -> xr.Dataset:
        """The horizontal grid with ``halo`` wrapped cells/faces (dims j_t/j_v, i_t/i_u), for operators that read
        their spacings from the grid (advective-form momentum, Laplacian viscosity)."""
        kinds = {("j_t", "i_t"): "c", ("j_t", "i_u"): "u", ("j_v", "i_t"): "v", ("j_v", "i_u"): "q"}
        data = {}
        for name, var in hg.data_vars.items():
            k = kinds.get(tuple(var.dims[-2:]))
            if k is not None:
                data[name] = (var.dims, self.pad(var.values, k, halo))
        return xr.Dataset(data, attrs=hg.attrs)


NONE = Periodic()


def periodic_grid(hg: xr.Dataset, per: Periodic) -> xr.Dataset:
    """
    A copy of the horizontal grid whose duplicated edge faces describe the wrap face: with ``"x"`` the u-faces 0
    and nx get Δx_u = ½(Δx_t[nx−1] + Δx_t[0]) (the distance between the two cell centres across the wrap; the
    builders give the walls Δx_t[0] and Δx_t[nx−1]), Δy_u and f_q of face/corner column nx equal to those of
    column 0, area_u = Δx_u Δy_u; likewise in y.  Unchanged grid when nothing wraps.
    """
    if not per:
        return hg
    g = hg.copy(deep=True)

    def put(name, value):
        g[name] = (g[name].dims, value)

    if per.x:
        dxu, dyu, fq = (g[n].values.copy() for n in ("dx_u", "dy_u", "f_q"))
        dxt = g["dx_t"].values
        dxu[:, 0] = dxu[:, -1] = 0.5 * (dxt[:, -1] + dxt[:, 0])
        dyu[:, -1] = dyu[:, 0]
        fq[:, -1] = fq[:, 0]
        put("dx_u", dxu), put("dy_u", dyu), put("area_u", dxu * dyu), put("f_q", fq)
    if per.y:
        dxv, dyv, fq = (g[n].values.copy() for n in ("dx_v", "dy_v", "f_q"))
        dyt = g["dy_t"].values
        dyv[0] = dyv[-1] = 0.5 * (dyt[-1] + dyt[0])
        dxv[-1] = dxv[0]
        fq[-1] = fq[0]
        put("dx_v", dxv), put("dy_v", dyv), put("area_v", dxv * dyv), put("f_q", fq)
    return g
