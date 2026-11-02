"""
Initial and open-boundary conditions from canonical ocean data.

Turns the output of any ocean adapter (ESPC/GOFS z levels, HYCOM native
archives, …; see ``forcing/sources``) into what the model consumes:

* ``initial_state`` → ``core.state.ModelState`` (T, S, u, v, η; w = 0)
* ``boundary_data`` → ``boundaries.obcs.OBData`` for one open side

Pipeline (``prepare_ocean``), done once per forcing file set:

1. sea-over-land fill on every horizontal slice   (``interp.fill_land``)
2. bilinear interpolation to the C-grid points    (``interp.to_model_grid``):
   T, S, η at T-points; u at U-points; v at V-points
3. linear interpolation to the model levels       (``interp.to_model_levels``)

Time interpolation to the model time is done on demand.  This mirrors the
HYCOM nesting chain (``z2h``/``remap`` → ``nest``) and MITgcm OBCS
preprocessing, with the same staggering as pyOCEAN's OBCS:

    west  : T, S, η at i_t = 0      ; u_normal = u at i_u = 0
    east  : T, S, η at i_t = nx − 1 ; u_normal = u at i_u = nx
    south : T, S, η at j_t = 0      ; u_normal = v at j_v = 0
    north : T, S, η at j_t = ny − 1 ; u_normal = v at j_v = ny

Sea surface height
------------------
Each product refers SSH to its own datum (geoid, model mean, steric
reference).  By default the domain-mean η of the initial state, and the
same constant for the boundaries, is removed so that the nest starts at
rest on average; pass ``remove_mean_ssh=False`` to keep the product datum.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from boundaries.obcs import OBData
from core.state import ModelState
from physics.eos.convert import to_model_ts
from forcing.interp import fill_land, interp_time, to_model_grid, to_model_levels
from forcing.rotation import to_grid

_SIDES = {
    "west": ({"i_t": 0}, "u", {"i_u": 0}),
    "east": ({"i_t": -1}, "u", {"i_u": -1}),
    "south": ({"j_t": 0}, "v", {"j_v": 0}),
    "north": ({"j_t": -1}, "v", {"j_v": -1}),
}


def prepare_ocean(ocean: xr.Dataset, h_grid: xr.Dataset, v_grid: xr.Dataset) -> dict[str, xr.Dataset]:
    """
    Canonical ocean → model C-grid points and z levels (all times).

    Parameters
    ----------
    ocean : xr.Dataset
        Canonical ocean (``temp``, ``salt`` and optionally ``u``, ``v``, ``ssh``)
        on (time, depth, lat, lon), e.g. from ``ForcingSource.load``.
    h_grid, v_grid : xr.Dataset
        pyOCEAN horizontal (spherical) and vertical grids.

    Returns
    -------
    dict
        ``"t"``: temp, salt, ssh on (time, k, j_t, i_t) / (time, j_t, i_t);
        ``"u"``: u on (time, k, j_t, i_u); ``"v"``: v on (time, k, j_v, i_t).
    """
    filled = xr.Dataset({name: fill_land(da) for name, da in ocean.data_vars.items()},
                        attrs=dict(ocean.attrs))
    rotate = "cos_t" in h_grid and "sin_t" in h_grid
    if rotate and (("u" in filled) != ("v" in filled)):
        raise ValueError("a grid with a rotation angle (cos_t, sin_t) needs both velocity components "
                         "u (east) and v (north) to rotate them")
    groups = {"t": [n for n in ("temp", "salt", "ssh") if n in filled],
              "u": [n for n in ("u",) if n in filled],
              "v": [n for n in ("v",) if n in filled]}
    out = {}
    for point, names in groups.items():
        if names:
            if rotate and point in ("u", "v"):
                on_grid = _rotated_velocity(filled[["u", "v"]], h_grid, point, names[0])
            else:
                on_grid = to_model_grid(filled[names], h_grid, point=point)
            out[point] = to_model_levels(on_grid, v_grid)
    return out


def _face_angle(h_grid: xr.Dataset, point: str) -> tuple[np.ndarray, np.ndarray]:
    """
    cos α, sin α at the u (``point='u'``, shape (ny, nx+1)) or v (``'v'``, (ny+1, nx)) faces.

    The angle lives at the T points (``cos_t``, ``sin_t``: counter-clockwise angle α of grid-x from east,
    ``core.grid.supergrid``).  A face takes the mean of the two T points it separates (one-sided at the
    outer faces) renormalised to unit length: (cos α, sin α) <- (c, s)/sqrt(c² + s²).  The data are rotated
    **at the face** (east and north interpolated to the face lon/lat, then rotated with this angle), so the
    normal velocity of the open boundary is the grid-normal component at the face where the model holds it.
    """
    c, s = h_grid["cos_t"].values, h_grid["sin_t"].values
    axis = 1 if point == "u" else 0

    def face(a: np.ndarray) -> np.ndarray:
        both = np.concatenate([np.take(a, [0], axis=axis), a, np.take(a, [-1], axis=axis)], axis=axis)
        n = a.shape[axis]
        return 0.5 * (np.take(both, range(0, n + 1), axis=axis) + np.take(both, range(1, n + 2), axis=axis))

    cf, sf = face(c), face(s)
    norm = np.hypot(cf, sf)
    return cf / norm, sf / norm


def _rotated_velocity(vel: xr.Dataset, h_grid: xr.Dataset, point: str, name: str) -> xr.Dataset:
    """East/north components at C-grid ``point`` rotated to the grid component ``name`` ('u' = x, 'v' = y)."""
    on_grid = to_model_grid(vel, h_grid, point=point)
    cos_a, sin_a = _face_angle(h_grid, point)
    u_x, v_y = to_grid(on_grid["u"].values, on_grid["v"].values, cos_a, sin_a)
    out = on_grid[[name]].copy()
    out[name] = on_grid[name].copy(data=u_x if name == "u" else v_y)
    out[name].attrs = dict(vel[name].attrs)
    return out


def prepare_ocean_source(ocean: xr.Dataset, h_grid: xr.Dataset) -> xr.Dataset:
    """
    Temperature and salinity on the model T points but on the **data's own z levels** (no vertical step).

    For starts that remap from the source grid (``SigmaBaroclinic.initial_state(..., h_src=...)``, MOM6
    Z_INIT_ALE_REMAPPING): ``source_layers`` turns this into layer means and thicknesses.
    """
    filled = xr.Dataset({n: fill_land(ocean[n]) for n in ("temp", "salt")}, attrs=dict(ocean.attrs))
    return to_model_grid(filled, h_grid, point="t")


def source_layers(src: xr.Dataset, H: np.ndarray, eta: np.ndarray, t: np.datetime64,
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Source layers of the data levels in each model column: thicknesses and means, bed first.

    The interfaces are half-way between the data depths d_n (CEFI ``z_to_dz``, ``boundary.py:244-276``), the first at
    the surface and the last extended to the bed; each is clipped at the bed depth H, so sum h_src = H + eta
    in a wet column (eta added to the top layer):

        h_n = min((d_n + d_{n+1})/2, H) - min((d_{n-1} + d_n)/2, H)   (d_{-1/2} = 0, d_{N+1/2} = inf),

    and T_n, S_n are the data at d_n (levels below the bed of the source filled downward with the deepest valid
    value).  Remapping these layers with a conservative scheme conserves the column integrals of T and S.

    Parameters
    ----------
    src : xr.Dataset
        ``prepare_ocean_source`` output (temp, salt on (time, depth, j_t, i_t)).
    H : np.ndarray
        Resting depth (ny, nx) [m], 0 on land.
    eta : np.ndarray
        Sea surface height (ny, nx) [m].
    t : np.datetime64
        Time of the start.

    Returns
    -------
    h_src, T_src, S_src : np.ndarray
        Each (N_src, ny, nx), bed first; zeros on land.
    """
    at_t = interp_time(src, t).sortby("depth")
    d = at_t["depth"].values.astype(float)
    mid = np.concatenate([[0.0], 0.5 * (d[:-1] + d[1:]), [np.inf]])
    H = np.asarray(H, float)
    wet = H > 0
    h = np.minimum(mid[1:, None, None], H[None]) - np.minimum(mid[:-1, None, None], H[None])
    h[0] += np.where(wet, np.asarray(eta, float), 0.0)
    out = []
    for name in ("temp", "salt"):
        f = at_t[name].ffill("depth").transpose("depth", "j_t", "i_t").values
        if np.isnan(f[h > 0]).any():
            raise ValueError(f"{name} has no valid value in some wet column of the model")
        out.append(np.where(h > 0, np.nan_to_num(f), 0.0)[::-1])
    return h[::-1].copy(), out[0], out[1]


def _mean_ssh(prepared: dict[str, xr.Dataset], t: np.datetime64) -> float:
    return float(interp_time(prepared["t"][["ssh"]], t)["ssh"].mean())


def initial_state(
    prepared: dict[str, xr.Dataset],
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
    t: np.datetime64,
    remove_mean_ssh: bool = True,
    ts_kind: str = "theta_sp",
    eos_ts_kind: str = "theta_sp",
) -> ModelState:
    """
    Model state at time ``t`` from prepared ocean data.

    Velocity components missing from the data start at rest; w = 0.

    Parameters
    ----------
    prepared : dict
        Output of ``prepare_ocean``.
    h_grid, v_grid : xr.Dataset
        Model grids.
    t : np.datetime64
        Initial time.
    remove_mean_ssh : bool
        Subtract the domain-mean η (default True).
    ts_kind, eos_ts_kind : {'theta_sp', 'ct_sa'}
        Convention of the data (``temp``, ``salt``; ESPC/GOFS and HYCOM give θ, S_P, the default) and of the
        receiving model's EOS (``model._eos.ts_kind``).  T and S are converted once here
        (``physics.eos.convert.to_model_ts``) when they differ; data already in the EOS convention must be passed
        with ``ts_kind == eos_ts_kind`` or they are converted twice.
    """
    state = ModelState.from_grids(h_grid, v_grid)
    at_t = {p: interp_time(ds, t) for p, ds in prepared.items()}
    state.T.values[:], state.S.values[:] = to_model_ts(at_t["t"]["temp"].values, at_t["t"]["salt"].values,
                                                       eos_ts_kind, ts_kind)
    if "u" in at_t:
        state.u.values[:] = at_t["u"]["u"].values
    if "v" in at_t:
        state.v.values[:] = at_t["v"]["v"].values
    if "ssh" in at_t["t"]:
        eta = at_t["t"]["ssh"].values
        state.eta.values[:] = eta - (eta.mean() if remove_mean_ssh else 0.0)
    return state


def boundary_data(
    prepared: dict[str, xr.Dataset],
    side: str,
    t: np.datetime64,
    remove_mean_ssh: bool = True,
    ts_kind: str = "theta_sp",
    eos_ts_kind: str = "theta_sp",
) -> OBData:
    """
    Prescribed exterior values for one open boundary at time ``t``.

    Parameters
    ----------
    prepared : dict
        Output of ``prepare_ocean``.
    side : {'west', 'east', 'south', 'north'}
        Open boundary.
    t : np.datetime64
        Model time.
    remove_mean_ssh : bool
        Subtract the domain-mean η at ``t`` (default True), consistent with
        ``initial_state``.
    ts_kind, eos_ts_kind : {'theta_sp', 'ct_sa'}
        As in ``initial_state``: T and S of the exterior are converted once here when the data convention
        differs from the model's.

    Returns
    -------
    OBData
        T, S, u_normal of shape (nz, n_bdy) and η of shape (n_bdy,).
    """
    if side not in _SIDES:
        raise ValueError(f"side must be one of {sorted(_SIDES)}, got {side!r}")
    t_sel, vel_name, vel_sel = _SIDES[side]

    tp = interp_time(prepared["t"], t).isel(t_sel)
    u_normal = None
    vel_point = "u" if vel_name == "u" else "v"
    if vel_point in prepared:
        u_normal = interp_time(prepared[vel_point], t)[vel_name].isel(vel_sel).values
    eta = None
    if "ssh" in tp:
        eta = tp["ssh"].values - (_mean_ssh(prepared, t) if remove_mean_ssh else 0.0)
    T_b, S_b = to_model_ts(tp["temp"].values, tp["salt"].values, eos_ts_kind, ts_kind)
    return OBData(side=side, T=T_b, S=S_b, u_normal=u_normal, eta=eta)


def source_level_grid(depth: np.ndarray) -> xr.Dataset:
    """
    A vertical grid whose layer centres are the data levels themselves (``z_t = −depth``), so that
    ``prepare_ocean(..., v_grid=source_level_grid(d))`` keeps the data on its own levels (``to_model_levels`` is then
    the identity, apart from the downward fill below the source bed).  The interfaces ``z_f`` are half-way between
    the levels, the first at the surface and the last one level-spacing below the deepest level (CEFI ``z_to_dz``,
    ``boundary.py:244-276``; MOM6 segment data on GLORYS levels).

    depth : (n,) increasing data depths [m, positive down].  Returns ``z_t`` (k), ``z_f`` (k_f), ``dz`` (k) [m].
    """
    d = np.asarray(depth, float)
    if d.ndim != 1 or d.size < 2 or not (np.diff(d) > 0).all() or d[0] <= 0:
        raise ValueError("depth must be ≥ 2 increasing positive data depths [m]")
    mid = 0.5 * (d[1:] + d[:-1])
    zf = -np.concatenate([[0.0], mid, [d[-1] + (d[-1] - mid[-1])]])
    return xr.Dataset({"z_t": ("k", -d), "z_f": ("k_f", zf), "dz": ("k", zf[:-1] - zf[1:])},
                      coords={"k": np.arange(d.size), "k_f": np.arange(d.size + 1)})


def exterior_provider(prepared: dict[str, dict[str, xr.Dataset]], v_grid: xr.Dataset,
                      edge_depth: dict[str, np.ndarray], origin: np.datetime64, t0=None, t1=None,
                      remove_mean_ssh: bool = False, ts_kind: str = "theta_sp", eos_ts_kind: str = "theta_sp"):
    """
    Time-varying open-boundary exterior for ``SigmaBaroclinic(open_boundary_state=...)`` from prepared ocean data:
    a function ``t → {side: OBData}`` of the model time t [s from ``origin``], linear in time between the data
    records (``forcing.boundary_time.interpolated_boundary``; constant beyond the first and last record).

    Each record is ``boundary_data`` (T, S, the grid-normal velocity at the open face, η) with the two fields the 3-D
    'radiating' edge needs: ``zw``, the data's own interfaces (``v_grid.z_f``, surface first, the last moved down to
    the deepest edge cell if needed, so that the data reach every bed), and ``ubar``, the depth mean of the normal
    velocity over the edge cell's water column D (the face depth of an open edge,
    ``OpenBoundary.edge_face_depths``), with the data layers clipped at D and the deepest extended to it, the
    arithmetic of ``core.dynamics.open_boundary._source_columns``:

        ū = Σ_k u_k (min(−z_{k+1}, D) − min(−z_k, D)) / D,   0 where D = 0.

    Parameters
    ----------
    prepared : {side: prepare_ocean output}
        One prepared set per side.  It may be built on the whole model grid (the same dict for every side) or on
        an edge strip of it (the edge row or column of T points with the two face rows/columns around it), which
        keeps the memory of long runs small; ``remove_mean_ssh`` then needs the whole grid and must stay False.
    v_grid : xr.Dataset
        The vertical grid ``prepare_ocean`` used, with ``z_f`` (e.g. ``source_level_grid`` of the data levels).
    edge_depth : {side: (n,) array}
        Rest depth of the boundary cells along each side [m], 0 or NaN on land.
    origin : np.datetime64
        Model time origin.
    t0, t1 : datetimes, optional
        Keep only the records bracketing [t0, t1].
    remove_mean_ssh, ts_kind, eos_ts_kind
        As ``boundary_data``.  Default ``remove_mean_ssh=False``: the product datum is kept (MOM6 reads zos as is).
    """
    from forcing.boundary_time import interpolated_boundary

    origin = np.datetime64(origin, "ns")
    zf = np.asarray(v_grid["z_f"].values, float)
    sides = tuple(prepared)
    times = np.asarray(prepared[sides[0]]["t"]["time"].values, dtype="datetime64[ns]")
    if t0 is not None:
        a = np.datetime64(t0, "ns")
        times = times[int(max(np.searchsorted(times, a, side="right") - 1, 0)):]
    if t1 is not None:
        b = np.datetime64(t1, "ns")
        times = times[:int(min(np.searchsorted(times, b, side="left"), times.size - 1)) + 1]
    deepest = max(float(np.nanmax(np.nan_to_num(np.asarray(d, float)))) for d in edge_depth.values())
    zw1 = zf.copy()
    zw1[-1] = min(zw1[-1], -deepest - 1.0)
    states = []
    for t in times:
        st = {}
        for side in sides:
            ob = boundary_data(prepared[side], side, t, remove_mean_ssh=remove_mean_ssh, ts_kind=ts_kind,
                               eos_ts_kind=eos_ts_kind)
            n = np.shape(ob.T)[1]
            D = np.nan_to_num(np.asarray(edge_depth[side], float))
            if D.shape != (n,):
                raise ValueError(f"{side}: edge_depth shape {D.shape}, expected ({n},)")
            z = np.minimum(-zw1[:, None], D[None])                      # depth of the interfaces, clipped at D
            z[-1] = D
            dz = z[1:] - z[:-1]
            u = np.nan_to_num(np.asarray(ob.u_normal, float))
            ubar = np.where(D > 0, (u * dz).sum(axis=0) / np.where(D > 0, D, 1.0), 0.0)
            st[side] = OBData(side=side, T=ob.T, S=ob.S, u_normal=u, eta=None if ob.eta is None else
                              np.nan_to_num(ob.eta), zw=np.repeat(zw1[:, None], n, axis=1), ubar=ubar)
        states.append(st)
    secs = (times - origin).astype("timedelta64[ns]").astype(np.int64) / 1e9
    return interpolated_boundary(secs, states)
