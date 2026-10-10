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
from forcing.interp import fill_land, interp_time, to_model_grid, to_model_levels

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
    groups = {"t": [n for n in ("temp", "salt", "ssh") if n in filled],
              "u": [n for n in ("u",) if n in filled],
              "v": [n for n in ("v",) if n in filled]}
    out = {}
    for point, names in groups.items():
        if names:
            on_grid = to_model_grid(filled[names], h_grid, point=point)
            out[point] = to_model_levels(on_grid, v_grid)
    return out


def _mean_ssh(prepared: dict[str, xr.Dataset], t: np.datetime64) -> float:
    return float(interp_time(prepared["t"][["ssh"]], t)["ssh"].mean())


def initial_state(
    prepared: dict[str, xr.Dataset],
    h_grid: xr.Dataset,
    v_grid: xr.Dataset,
    t: np.datetime64,
    remove_mean_ssh: bool = True,
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
    """
    state = ModelState.from_grids(h_grid, v_grid)
    at_t = {p: interp_time(ds, t) for p, ds in prepared.items()}
    state.T.values[:] = at_t["t"]["temp"].values
    state.S.values[:] = at_t["t"]["salt"].values
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
    return OBData(side=side, T=tp["temp"].values, S=tp["salt"].values, u_normal=u_normal, eta=eta)
