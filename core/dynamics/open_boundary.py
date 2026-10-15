"""
Time-varying exterior state at open boundaries → model arrays (spec 2026-10-12-baroclinic-obc-internal-wave §1.4):
each edge's OBData profiles live on their own z grid (``zw``, surface first) and are remapped conservatively onto
the model layers of the boundary and sponge cells/faces (the external data on its own z axis, as MOM6
MOM_open_boundary.F90:1188-1204).  Sponge cells take the profile of the nearest open edge at the same along-edge
position, clipped or extended to their own bed.  Pure functions on NumPy arrays; (N, ny, nx), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np

from boundaries.obcs import OBData
from core.vertical import remap as rm
from core.vertical.ale import baroclinic_split

EDGES = ("west", "east", "south", "north")


def validate_state(state: dict[str, OBData], open_edges: tuple[str, ...], shape: tuple[int, int]) -> None:
    """Edges open, shapes (nz, n) / (nz+1, n) with n the edge length, zw decreasing; ValueError otherwise."""
    ny, nx = shape
    for edge, ob in state.items():
        if edge not in open_edges:
            raise ValueError(f"open_boundary_state edge {edge!r} is not in open_boundaries {open_edges}")
        if ob.zw is None or ob.T is None or ob.u_normal is None:
            raise ValueError(f"{edge}: OBData needs zw, T and u_normal")
        n = ny if edge in ("west", "east") else nx
        nz = np.shape(ob.T)[0]
        if np.shape(ob.T) != (nz, n) or np.shape(ob.u_normal) != (nz, n) or np.shape(ob.zw) != (nz + 1, n):
            raise ValueError(f"{edge}: shapes T {np.shape(ob.T)}, u_normal {np.shape(ob.u_normal)}, zw "
                             f"{np.shape(ob.zw)}; expected ({nz}, {n}) and ({nz + 1}, {n})")
        if not (np.diff(ob.zw, axis=0) < 0).all():
            raise ValueError(f"{edge}: zw must decrease from the surface down")
        for name in ("eta", "ubar"):
            a = getattr(ob, name)
            if a is not None and np.shape(a) != (n,):
                raise ValueError(f"{edge}: {name} shape {np.shape(a)}, expected ({n},)")
        if ob.tracers is not None and (np.ndim(ob.tracers) != 3 or np.shape(ob.tracers)[1:] != (nz, n)):
            raise ValueError(f"{edge}: tracers shape {np.shape(ob.tracers)}, expected (ntr, {nz}, {n})")


def _values(ob: OBData, field) -> np.ndarray:
    """The (nz, n) profile of T (``field`` None) or of passive tracer ``field`` (zeros when OBData has none)."""
    if field is None:
        return np.asarray(ob.T, float)
    return np.zeros(np.shape(ob.T)) if ob.tracers is None else np.asarray(ob.tracers, float)[field]


def _along(edge: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(along-edge index, distance from the edge in cells or faces) for every point of a (ny, nx) array."""
    ny, nx = shape
    j, i = np.mgrid[0:ny, 0:nx]
    return {"west": (j, i), "east": (j, nx - 1 - i), "south": (i, j), "north": (i, ny - 1 - j)}[edge]


def _nearest(edges, shape: tuple[int, int]) -> np.ndarray:
    """Index into EDGES of the nearest of ``edges`` at every point (ties: first in EDGES order)."""
    present = [e for e in EDGES if e in edges]
    d = np.stack([_along(e, shape)[1] for e in present])
    return np.array([EDGES.index(e) for e in present])[np.argmin(d, axis=0)]


def owned_by_state(state: dict[str, OBData], open_edges, shape: tuple[int, int]) -> np.ndarray:
    """
    Points whose nearest open edge (among ``open_edges``, all of them, not only those with data) has exterior
    data: the others keep the resting exterior.  Without this a sponge band at an edge without data would take
    the profile of another edge.
    """
    near = _nearest(open_edges, shape)
    return np.isin(near, [EDGES.index(e) for e in state])


def _source_columns(ob: OBData, edge: str, depth: np.ndarray, check: np.ndarray,
                    field=None) -> tuple[np.ndarray, ...]:
    """
    Bed-first source thicknesses and fields (nz, ny, nx) at every point, from the edge profile at the same
    along-edge index, clipped to each point's own bed ``depth`` [m] (the deepest layer extended when the point
    is deeper than the data).  ``check``: points (the edge itself) where the data must reach the bed.
    """
    a, _ = _along(edge, depth.shape)
    zw = np.asarray(ob.zw, float)[:, a]                                 # (nz+1, ny, nx), surface first
    if (check & (zw[-1] > -depth + 1e-6)).any():
        raise ValueError(f"{edge}: exterior data do not reach the bed of the boundary column (zw[-1] > −H)")
    z = np.maximum(zw, -depth[None])
    z[-1] = -depth
    h = (z[:-1] - z[1:])[::-1]
    return h, _values(ob, field)[:, a][::-1], np.asarray(ob.u_normal, float)[:, a][::-1]


def _remap_stretched(h_src: np.ndarray, f_src: np.ndarray, h: np.ndarray, wet: np.ndarray, scheme: int) -> np.ndarray:
    """Source columns stretched to the model column total (z*-like, as exterior_on_layers) and remapped."""
    tot_s, tot = h_src.sum(axis=0), h.sum(axis=0)
    ok = wet & (tot_s > 0) & (tot > 0)
    scale = np.where(ok, tot / np.where(ok, tot_s, 1.0), 1.0)
    out = rm.remap_columns(np.ascontiguousarray(h_src * scale[None]), np.ascontiguousarray(f_src),
                           np.ascontiguousarray(h), scheme, ok)
    return np.where(ok[None], out, 0.0)


def _exterior_tracer_full(state: dict[str, OBData], H: np.ndarray, h: np.ndarray, mask: np.ndarray,
                          scheme: int, open_edges=None, field=None) -> np.ndarray:
    """Reference of exterior_tracer (test oracle): source columns built on the whole domain."""
    H = np.nan_to_num(np.asarray(H, float))
    near = _nearest(state if open_edges is None else open_edges, H.shape)
    out = np.zeros(h.shape)
    for e, ob in state.items():
        sel = mask & (near == EDGES.index(e))
        if not sel.any():
            continue
        h_src, T_src, _ = _source_columns(ob, e, H, sel & (_along(e, H.shape)[1] == 0), field)
        out = np.where(sel[None], _remap_stretched(h_src, T_src, h, sel, scheme), out)
    return out


def _source_columns_at(ob: OBData, edge: str, depth: np.ndarray, a: np.ndarray,
                       check: np.ndarray, field=None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """_source_columns at a list of points (npts,) only: bed-first thicknesses, T (or passive tracer ``field``) and u
    (nz, npts), same arithmetic."""
    zw = np.asarray(ob.zw, float)[:, a]                                 # (nz+1, npts), surface first
    if (check & (zw[-1] > -depth + 1e-6)).any():
        raise ValueError(f"{edge}: exterior data do not reach the bed of the boundary column (zw[-1] > −H)")
    z = np.maximum(zw, -depth[None])
    z[-1] = -depth
    return (z[:-1] - z[1:])[::-1], _values(ob, field)[:, a][::-1], np.asarray(ob.u_normal, float)[:, a][::-1]


def exterior_tracer(state: dict[str, OBData], H: np.ndarray, h: np.ndarray, mask: np.ndarray,
                    scheme: int, open_edges=None, field=None) -> np.ndarray:
    """
    Tₑₓₜ on the layers h (N, ny, nx) in the ``mask`` columns whose nearest open edge (``open_edges``, default
    the edges of ``state``) has data; 0 elsewhere.  ``field``: None = T, i = passive tracer i (OBData.tracers).  Source columns are built and remapped only at the mask
    points (gather–scatter; _exterior_tracer_full is the whole-domain reference with the same arithmetic).
    """
    H = np.nan_to_num(np.asarray(H, float))
    near = _nearest(state if open_edges is None else open_edges, H.shape)
    out = np.zeros(h.shape)
    for e, ob in state.items():
        jj, ii = np.nonzero(mask & (near == EDGES.index(e)))
        if jj.size == 0:
            continue
        a, d = (x[jj, ii] for x in _along(e, H.shape))
        h_src, T_src, _ = _source_columns_at(ob, e, H[jj, ii], a, d == 0, field)
        res = _remap_stretched(h_src[:, :, None], T_src[:, :, None], np.ascontiguousarray(h[:, jj, ii])[:, :, None],
                               np.ones((jj.size, 1), bool), scheme)
        out[:, jj, ii] = res[:, :, 0]
    return out


def _exterior_velocity_full(state: dict[str, OBData], kind: str, Hf: np.ndarray, hf: np.ndarray, mask: np.ndarray,
                      w: np.ndarray, scheme: int, open_edges=None) -> np.ndarray:
    """
    Reference of exterior_velocity (test oracle): source columns built on the whole face array.
    u′ₑₓₜ on the face layers hf in the ``mask`` faces of the edges normal to ``kind`` ('u': west/east,
    'v': south/north): the data's normal velocity remapped to the face layers, minus its mean with the model
    face weights w (Σ w u′ₑₓₜ = 0, so the free-surface flux stays Flather's).  Faces nearer an open edge
    (``open_edges``) without data, and all others, get 0 (the resting exterior).
    """
    Hf = np.nan_to_num(np.asarray(Hf, float))
    normal = ("west", "east") if kind == "u" else ("south", "north")
    sub = {e: ob for e, ob in state.items() if e in normal}
    out = np.zeros(hf.shape)
    if not sub:
        return out
    near = _nearest(sub if open_edges is None else open_edges, Hf.shape)   # all open edges: a face nearest a
                                                                             # tangential edge keeps the rest
    for e, ob in sub.items():
        sel = mask & (near == EDGES.index(e)) & (Hf > 0)
        if not sel.any():
            continue
        h_src, _, u_src = _source_columns(ob, e, Hf, sel & (_along(e, Hf.shape)[1] == 0))
        u = _remap_stretched(h_src, u_src, hf, sel, scheme)
        out = np.where(sel[None], baroclinic_split(u, w)[1], out)
    return out



def exterior_velocity(state: dict[str, OBData], kind: str, Hf: np.ndarray, hf: np.ndarray, mask: np.ndarray,
                      w: np.ndarray, scheme: int, open_edges=None) -> np.ndarray:
    """
    u′ₑₓₜ on the face layers hf in the ``mask`` faces of the edges normal to ``kind`` ('u': west/east,
    'v': south/north): the data's normal velocity remapped to the face layers, minus its mean with the model
    face weights w (Σ w u′ₑₓₜ = 0, so the free-surface flux stays Flather's).  Faces nearer an open edge
    (``open_edges``) without data, and all others, get 0 (the resting exterior).  Built only at the mask
    faces (gather–scatter); _exterior_velocity_full is the reference.
    """
    Hf = np.nan_to_num(np.asarray(Hf, float))
    normal = ("west", "east") if kind == "u" else ("south", "north")
    sub = {e: ob for e, ob in state.items() if e in normal}
    out = np.zeros(hf.shape)
    if not sub:
        return out
    near = _nearest(sub if open_edges is None else open_edges, Hf.shape)   # all open edges: a face nearest a
                                                                             # tangential edge keeps the rest
    for e, ob in sub.items():
        jj, ii = np.nonzero(mask & (near == EDGES.index(e)) & (Hf > 0))
        if jj.size == 0:
            continue
        a, d = (x[jj, ii] for x in _along(e, Hf.shape))
        h_src, _, u_src = _source_columns_at(ob, e, Hf[jj, ii], a, d == 0)
        u = _remap_stretched(h_src[:, :, None], u_src[:, :, None], np.ascontiguousarray(hf[:, jj, ii])[:, :, None],
                             np.ones((jj.size, 1), bool), scheme)
        out[:, jj, ii] = baroclinic_split(u, np.ascontiguousarray(w[:, jj, ii])[:, :, None])[1][:, :, 0]
    return out


def flather_targets(state: dict[str, OBData], shape_u: tuple[int, int], shape_v: tuple[int, int]) -> dict:
    """Flather exterior (η, normal ū) at the open faces, zero elsewhere, in the free-surface argument layout."""
    zu, zv, ubar, vbar = np.zeros(shape_u), np.zeros(shape_v), np.zeros(shape_u), np.zeros(shape_v)
    for e, ob in state.items():
        eta = np.zeros(np.shape(ob.T)[1]) if ob.eta is None else np.asarray(ob.eta, float)
        ub = np.zeros_like(eta) if ob.ubar is None else np.asarray(ob.ubar, float)
        if e == "west":
            zu[:, 0], ubar[:, 0] = eta, ub
        elif e == "east":
            zu[:, -1], ubar[:, -1] = eta, ub
        elif e == "south":
            zv[0, :], vbar[0, :] = eta, ub
        else:
            zv[-1, :], vbar[-1, :] = eta, ub
    return dict(flather_eta=(zu, zv), flather_ubar=ubar, flather_vbar=vbar)
