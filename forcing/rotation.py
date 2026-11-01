"""
Vector rotation between earth axes (east, north) and grid axes (x, y).

On a curvilinear (rotated) grid the model momentum and the surface stress live
on grid axes, whereas the atmosphere (ERA5, WRF earth-relative) and the tides
and open-boundary data are given as east/north components.  With the angle
convention of ``core.grid.supergrid.grid_from_supergrid`` (MOM6 ``angle_dx``,
``cos_t = cos α``, ``sin_t = sin α`` with α the counter-clockwise angle of
grid-x from east):

    u_east  = cos α u_x − sin α v_y,        (to_earth)
    v_north = sin α u_x + cos α v_y,

    u_x =  cos α u_east + sin α v_north,    (to_grid, the transpose)
    v_y = −sin α u_east + cos α v_north.

The rotation is orthogonal, so ``to_grid`` and ``to_earth`` are exact
inverses up to round-off and the vector magnitude is preserved.

MOM6 rotates the wind/stress to the ocean grid in the coupler/forcing code:
see ``docs/design/regional_refs/mom6_psurf_runoff.md`` (SIS2 / FMS coupler
``rotate_vector`` and ``MOM_surface_forcing_gfdl.F90`` ``rotate_vector`` calls
on the stress, with the cos/sin of ``G%cos_rot``, ``G%sin_rot``).
Arrays may have any shape provided ``cos_a`` and ``sin_a`` broadcast to it.
"""

from __future__ import annotations

import numpy as np

__all__ = ["to_grid", "to_earth"]


def to_grid(u_east: np.ndarray, v_north: np.ndarray, cos_a: np.ndarray,
            sin_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Earth components (east, north) → grid components (x, y) [same units].

    u_x = cos α u_east + sin α v_north,  v_y = −sin α u_east + cos α v_north.
    """
    return cos_a * u_east + sin_a * v_north, -sin_a * u_east + cos_a * v_north


def to_earth(u_x: np.ndarray, v_y: np.ndarray, cos_a: np.ndarray,
             sin_a: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Grid components (x, y) → earth components (east, north) [same units].

    u_east = cos α u_x − sin α v_y,  v_north = sin α u_x + cos α v_y.
    """
    return cos_a * u_x - sin_a * v_y, sin_a * u_x + cos_a * v_y
