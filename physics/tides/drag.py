"""
Internal-tide (topographic wave) drag on the barotropic tide.

Over rough topography the barotropic tide loses energy to internal waves,
which a barotropic or coarse model does not resolve.  Jayne & St. Laurent
(2001) parameterise this as a linear drag whose rate scales as
½ κ h² N_b / H; HYCOM (Arbic et al. 2010; Buijsman et al. 2015) uses a
precomputed global field JSL [1/s] (HYCOM-examples GLBp0.08 relax/DRAG) and
reads ``tidal.rh`` = JSL · depth [m/s] with ``tiddrg = 1``.

Preparation (HYCOM-examples wiki §8B, ``tidal.JSLH.07.lim24.csh``):

    JSL → model grid: plain bilinear (``isuba_field`` without masks)
        → clip at 1/day (``hycom_clip 0 1.1574e-5``)
        → zero below 1/(10 days) (``hycom_bandmask 0 1.1574e-6`` + ``hycom_void``)
    rh = JSL · depth, 0 on land (``hycom_expr``)

Use (``mod_barotp.F90``, ``tiddrg = 1``, ``thkdrg = 0``): implicit drag on
the depth-mean velocity with the scalar tensor t = drgscl · rh,

    ū ← ū / (1 + Δt · drgscl · rh / H)

so the damping rate is drgscl · JSL (rh / H) — at most drgscl per day.

References
----------
Jayne, S.R. & St. Laurent, L.C. (2001). Parameterizing tidal dissipation over
    rough topography. Geophys. Res. Lett. 28, 811–814.
Arbic, B.K., Wallcraft, A.J. & Metzger, E.J. (2010). Concurrent simulation of
    the eddying general circulation and tides in a global ocean model. Ocean
    Modell. 32, 175–187.
"""

from __future__ import annotations

import numpy as np

MAX_RATE = 1.1574e-5      # 1/s: hycom_clip ceiling, 1/day
MIN_RATE = 1.1574e-6      # 1/s: hycom_bandmask floor, 1/(10 days)


def jsl_to_rh(jsl: np.ndarray, depth: np.ndarray, max_rate: float = MAX_RATE,
              min_rate: float = MIN_RATE) -> np.ndarray:
    """
    rh = clip(JSL) · depth [m s⁻¹] (HYCOM ``tidal.rh``).

    Parameters
    ----------
    jsl : JSL drag rate on the model grid [s⁻¹]
    depth : model depth [m]; NaN or ≤ 0 is land
    max_rate : ceiling of JSL (``hycom_clip``) [s⁻¹]
    min_rate : JSL ≤ this is set to 0 (``hycom_bandmask`` + ``hycom_void``) [s⁻¹]
    """
    j = np.clip(np.nan_to_num(np.asarray(jsl, float), nan=0.0), 0.0, max_rate)
    j = np.where(j <= min_rate, 0.0, j)
    d = np.nan_to_num(np.asarray(depth, float), nan=0.0)
    return np.where(d > 0, j * d, 0.0)


def regrid_bilinear(field: np.ndarray, lat_axis: np.ndarray, lon_axis: np.ndarray,
                    lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """
    Plain bilinear interpolation from a rectilinear (lat, lon) grid, as
    HYCOM ``isuba_field`` without land masks (land values take part like
    any other node).  NaN in ``field`` is treated as 0 (HYCOM data void →
    0 after ``hycom_void``).
    """
    from scipy.interpolate import RegularGridInterpolator

    f = np.nan_to_num(np.asarray(field, float), nan=0.0)
    interp = RegularGridInterpolator((np.asarray(lat_axis, float), np.asarray(lon_axis, float)), f,
                                     bounds_error=True)
    lat, lon = np.asarray(lat, float), np.asarray(lon, float)
    return interp(np.column_stack([lat.ravel(), lon.ravel()])).reshape(lat.shape)


def rh_at_faces(rh: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    rh at C-grid u- and v-faces: mean of the adjacent wet T-cells (one cell
    at coasts and domain edges, 0 if none) — HYCOM applies the drag on the
    p-grid and averages its increment to the faces.

    Returns rh_u (ny, nx+1) and rh_v (ny+1, nx).
    """
    rh = np.where(wet, np.asarray(rh, float), 0.0)
    w = wet.astype(float)
    ny, nx = rh.shape
    s_u, n_u = np.zeros((ny, nx + 1)), np.zeros((ny, nx + 1))
    s_u[:, :-1] += rh
    s_u[:, 1:] += rh
    n_u[:, :-1] += w
    n_u[:, 1:] += w
    s_v, n_v = np.zeros((ny + 1, nx)), np.zeros((ny + 1, nx))
    s_v[:-1] += rh
    s_v[1:] += rh
    n_v[:-1] += w
    n_v[1:] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_u > 0, s_u / n_u, 0.0), np.where(n_v > 0, s_v / n_v, 0.0)


def tidal_drag_rate(rh_face: np.ndarray, depth_face: np.ndarray, drgscl: float) -> np.ndarray:
    """Linear drag rate r = drgscl · rh / H [s⁻¹] (0 where H = 0)."""
    h = np.asarray(depth_face, float)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(h > 0, drgscl * np.asarray(rh_face, float) / h, 0.0)


def implicit_linear_drag(u: np.ndarray, rate: np.ndarray, dt: float) -> np.ndarray:
    """ū ← ū / (1 + Δt r): backward-Euler linear drag, unconditionally stable (HYCOM mod_barotp)."""
    return np.asarray(u, float) / (1.0 + dt * np.asarray(rate, float))
