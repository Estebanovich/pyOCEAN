"""
Arakawa C horizontal grid from a MOM6 / FMS supergrid (``ocean_hgrid.nc``).

A supergrid has twice the model resolution: with ``ni × nj`` model cells it
has ``(2nj+1) × (2ni+1)`` nodes.  Node parity (0-based Python indices of
``x``, ``y``): T at (odd j, odd i), u at (odd j, even i), v at (even j, odd i),
q at (even j, even i).  ``dx[r, k]`` is the length of the edge from node
``(r, k)`` to ``(r, k+1)``, ``dy[r, k]`` from ``(r, k)`` to ``(r+1, k)`` and
``area[r, k]`` the area of supergrid cell ``(r, k)``.

Every metric follows MOM6 ``set_grid_metrics_from_mosaic``
(MOM6 git 0e9a561, ``src/initialization/MOM_grid_initialize.F90:170-305``),
read and translated to Python indices in
``docs/design/regional_refs/mom6_grid.md`` §2-§3:

* lengths are sums of two supergrid half-segments (``:270-286``);
* T and q areas are sums of four supergrid sub-cells, in MOM6's order of
  summation (pairs of diagonal sub-cells first, ``:296-301``);
* u and v areas are MOM6's products ``areaCu = dxCu · dyCu``,
  ``areaCv = dyCv · dxCv`` (``:1334, 1342``), **unmasked** here (MOM6 masks
  them through the open face length ``dy_Cu``, ``dx_Cv``, ``:1332, 1340``;
  masking is left to plan Task 3);
* half-segments and sub-cell areas beyond the grid edge are linearly
  extrapolated, ``v_out = 2 v(edge) − v(edge∓1)``, in MOM6's order (south,
  north, then west, east; ``extrapolate_metric``, ``:1139-1169``).  This sets
  the outermost-face metrics (``dx_u`` at I = 0 and I = ni, ``dy_v`` at
  J = 0 and J = nj, ``dx_q``, ``dy_q``, ``area_q`` on the edges).

The rotation angle is MOM6's corner (q-point) formula
(``MOM_shared_initialization.F90:571-626``, ``GRID_ROTATION_ANGLE_BUGS =
False``), **not** the supergrid's ``angle_dx`` (MOM6 never reads it,
mom6_grid.md §6).  The Coriolis parameter is ``f = 2Ω sin φ`` at q points
(``MOM_shared_initialization.F90:504-506``); ``f_t`` at T points has no MOM6
counterpart (pyOCEAN choice, class c).

Array index order: (j, i); T arrays (nj, ni), u (nj, ni+1), v (nj+1, ni),
q (nj+1, ni+1), as in :mod:`core.grid.horizontal`.
"""

from __future__ import annotations

import numpy as np
import xarray as xr

from core.grid.horizontal import R_EARTH, _assemble_dataset

#: Largest accepted departure of the grid-x/grid-y angle from 90° [rad].
MAX_SKEW: float = 1e-3


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def supergrid_from_arrays(
    x: np.ndarray,
    y: np.ndarray,
    dx: np.ndarray,
    dy: np.ndarray,
    area: np.ndarray,
    angle_dx: np.ndarray,
    units: str = "degrees",
) -> xr.Dataset:
    """
    Build an ``ocean_hgrid.nc``-like supergrid Dataset from arrays.

    Parameters
    ----------
    x, y : np.ndarray, shape (nyp, nxp) = (2nj+1, 2ni+1)
        Node positions: longitude, latitude [degrees] or, with
        ``units="m"``, planar coordinates [m] (synthetic grids only; MOM6
        assumes degrees, ``MOM_grid_initialize.F90:256-259``).
    dx : np.ndarray, shape (nyp, nxp-1)
        Edge length from node (r, k) to (r, k+1) [m].
    dy : np.ndarray, shape (nyp-1, nxp)
        Edge length from node (r, k) to (r+1, k) [m].
    area : np.ndarray, shape (nyp-1, nxp-1)
        Supergrid cell area [m²].
    angle_dx : np.ndarray, shape (nyp, nxp)
        Counter-clockwise angle of grid-x from east [degrees] (FRE-NCtools
        convention; stored for completeness, not used by the metrics).
    units : str
        ``"degrees"`` or ``"m"``.

    Returns
    -------
    xr.Dataset
        Variables ``x``, ``y``, ``dx``, ``dy``, ``area``, ``angle_dx`` with
        the FMS dimension names ``nyp``, ``nxp``, ``ny``, ``nx``.
    """
    x = np.asarray(x, dtype=float)
    nyp, nxp = x.shape
    shapes = {"y": (nyp, nxp), "dx": (nyp, nxp - 1), "dy": (nyp - 1, nxp),
              "area": (nyp - 1, nxp - 1), "angle_dx": (nyp, nxp)}
    arrs = {"y": y, "dx": dx, "dy": dy, "area": area, "angle_dx": angle_dx}
    for name, shp in shapes.items():
        if np.shape(arrs[name]) != shp:
            raise ValueError(f"supergrid '{name}' has shape {np.shape(arrs[name])}, expected {shp}")
    _check_supergrid_shape(nyp, nxp)
    geo = _is_geographic(units)
    xu = "degrees_east" if geo else "m"
    yu = "degrees_north" if geo else "m"
    return xr.Dataset(
        {
            "x": (("nyp", "nxp"), x, {"units": xu}),
            "y": (("nyp", "nxp"), np.asarray(y, dtype=float), {"units": yu}),
            "dx": (("nyp", "nx"), np.asarray(dx, dtype=float), {"units": "m"}),
            "dy": (("ny", "nxp"), np.asarray(dy, dtype=float), {"units": "m"}),
            "area": (("ny", "nx"), np.asarray(area, dtype=float), {"units": "m2"}),
            "angle_dx": (("nyp", "nxp"), np.asarray(angle_dx, dtype=float),
                         {"units": "degrees"}),
        }
    )


def grid_from_supergrid(
    hgrid: xr.Dataset,
    depth: np.ndarray | None = None,
    window: tuple[int, int, int, int] | None = None,
    omega: float = 7.2921e-5,
    max_skew: float = MAX_SKEW,
) -> xr.Dataset:
    """
    pyOCEAN C-grid Dataset from a MOM6 supergrid, as MOM6 builds it.

    Metric semantics are those of :mod:`core.grid.horizontal`:
    ``dx_u`` = distance between the two T points across a u face
    (MOM6 ``dxCu``), ``dy_u`` = u-face length (``dyCu``), ``dx_v`` = v-face
    length (``dxCv``), ``dy_v`` = T-to-T distance across the v face
    (``dyCv``); mapping table in mom6_grid.md §3 and §8.

    Rotation.  With MOM6's clockwise angle ``θ`` of grid-north from true
    north (``MOM_shared_initialization.F90:604-622``) the outputs are
    ``cos_t = cos θ = cos_rot`` and ``sin_t = −sin θ = −sin_rot``, i.e.
    ``sin_t``, ``cos_t`` describe the counter-clockwise angle of grid-x from
    east (≈ the supergrid's ``angle_dx``; ``θ ≈ −angle_dx`` to O(Δ²),
    mom6_grid.md §6), so that

        u_east  = cos_t u_x − sin_t v_y,
        v_north = sin_t u_x + cos_t v_y.

    Window.  ``window = (i0, i1, j0, j1)`` (T-cell indices of the full grid,
    end exclusive) cuts the supergrid first (even offsets, mom6_grid.md §7)
    and then builds the cut as a grid of its own, i.e. what MOM6 computes
    when run on the cut ``ocean_hgrid.nc``.  T metrics, positions, angles
    and every interior face/corner equal slicing the full result; the
    outermost-face metrics (``dx_u`` at I = 0, ni; ``dy_v`` at J = 0, nj;
    ``dx_q``, ``dy_q``, ``area_q`` on the window edge) use MOM6's linear
    extrapolation and differ from the parent grid's values unless the
    half-segments vary linearly (class c, mom6_grid.md §3).  ``x_*``,
    ``y_*`` are distances from the window's own SW corner.

    Parameters
    ----------
    hgrid : xr.Dataset
        Supergrid with ``x``, ``y``, ``dx``, ``dy``, ``area`` (FMS layout).
        ``x.attrs['units']`` containing "degree" (default when absent) means
        lon/lat; ``"m"`` means a planar synthetic grid, for which lon/lat
        and f are NaN (no latitude) and the angle uses the planar form.
    depth : np.ndarray or None
        Bed depth, positive down [m], shape of the full grid (nj, ni) or of
        the window; ``mask_t = depth > 0``.  None → all ocean.
    window : (i0, i1, j0, j1) or None
        Sub-domain in T-cell indices of the full grid.
    omega : float
        Earth's rotation rate Ω [s⁻¹] (MOM6 ``OMEGA``, NWA12 7.2921e-5).
    max_skew : float
        Largest accepted departure of the grid-x/grid-y angle from 90° [rad],
        checked in wet cells when ``depth`` is given, in every cell otherwise.

    Returns
    -------
    xr.Dataset
        The variables of :func:`core.grid.horizontal._assemble_dataset` plus
        ``dx_q``, ``dy_q``, ``area_q``, ``lon_*``, ``lat_*`` (t, u, v, q),
        ``cos_t``, ``sin_t``; attrs ``grid_type="mom6_supergrid"``.

    Raises
    ------
    ValueError
        Bad supergrid shapes, window, depth shape, or a grid whose
        grid-x/grid-y angle departs from 90° by more than ``max_skew``
        (wet cells with ``depth``, all cells without; names the skew and the
        worst T cell (j, i), window indices).
    """
    nyp_f, nxp_f = hgrid["x"].shape
    _check_supergrid_shape(nyp_f, nxp_f)
    nj_f, ni_f = (nyp_f - 1) // 2, (nxp_f - 1) // 2
    geo = _is_geographic(hgrid["x"].attrs.get("units", "degrees"))

    if window is None:
        i0, i1, j0, j1 = 0, ni_f, 0, nj_f
    else:
        i0, i1, j0, j1 = (int(v) for v in window)
        if not (0 <= i0 < i1 <= ni_f and 0 <= j0 < j1 <= nj_f):
            raise ValueError(f"window {window} outside the grid (ni={ni_f}, nj={nj_f})")

    # ---- cut the supergrid at even offsets (mom6_grid.md §7) -------------
    sn = slice(2 * j0, 2 * j1 + 1)   # node rows
    sc = slice(2 * j0, 2 * j1)       # cell rows
    en = slice(2 * i0, 2 * i1 + 1)   # node columns
    ec = slice(2 * i0, 2 * i1)       # cell columns
    X = np.asarray(hgrid["x"].values[sn, en], dtype=float)
    Y = np.asarray(hgrid["y"].values[sn, en], dtype=float)
    dx = np.asarray(hgrid["dx"].values[sn, ec], dtype=float)
    dy = np.asarray(hgrid["dy"].values[sc, en], dtype=float)
    a = np.asarray(hgrid["area"].values[sc, ec], dtype=float)
    nj, ni = j1 - j0, i1 - i0

    mask_t = _mask_from_depth(depth, (nj_f, ni_f), (nj, ni), (j0, j1, i0, i1))

    # ---- non-orthogonality check from the supergrid geometry -------------
    _check_orthogonal(X, Y, geo, mask_t, max_skew)

    # ---- edge extrapolation (MOM_grid_initialize.F90:1139-1169) ----------
    # dx only needs columns outside (index -1 and 2ni), dy rows outside,
    # area both: rows first (south, north), then columns (west, east).
    dxp = _extrap_cols(dx)
    dyp = _extrap_rows(dy)
    ap = _extrap_cols(_extrap_rows(a))

    # ---- lengths: sums of two half-segments (mom6_grid.md §3 table) ------
    # dxT = dx[2j+1, 2i] + dx[2j+1, 2i+1]             (:270)
    dx_t = dx[1::2, 0::2] + dx[1::2, 1::2]
    # dyT = dy[2j, 2i+1] + dy[2j+1, 2i+1]             (:271)
    dy_t = dy[0::2, 1::2] + dy[1::2, 1::2]
    # dxCu = dx[2j+1, 2I-1] + dx[2j+1, 2I]            (:275); dxp[:, k+1] = dx[:, k]
    dx_u = dxp[1::2, 0::2] + dxp[1::2, 1::2]
    # dyCu = dy[2j, 2I] + dy[2j+1, 2I]                (:276)
    dy_u = dy[0::2, 0::2] + dy[1::2, 0::2]
    # dxCv = dx[2J, 2i] + dx[2J, 2i+1]                (:280)
    dx_v = dx[0::2, 0::2] + dx[0::2, 1::2]
    # dyCv = dy[2J-1, 2i+1] + dy[2J, 2i+1]            (:281); dyp[r+1] = dy[r]
    dy_v = dyp[0::2, 1::2] + dyp[1::2, 1::2]
    # dxBu = dx[2J, 2I-1] + dx[2J, 2I]                (:285)
    dx_q = dxp[0::2, 0::2] + dxp[0::2, 1::2]
    # dyBu = dy[2J-1, 2I] + dy[2J, 2I]                (:286)
    dy_q = dyp[0::2, 0::2] + dyp[1::2, 0::2]

    # ---- areas -------------------------------------------------------------
    # areaT = (a[2j,2i] + a[2j+1,2i+1]) + (a[2j+1,2i] + a[2j,2i+1])   (:296-297)
    area_t = (a[0::2, 0::2] + a[1::2, 1::2]) + (a[1::2, 0::2] + a[0::2, 1::2])
    # areaBu = (a[2J-1,2I-1] + a[2J,2I]) + (a[2J,2I-1] + a[2J-1,2I])  (:300-301)
    area_q = (ap[0::2, 0::2] + ap[1::2, 1::2]) + (ap[1::2, 0::2] + ap[0::2, 1::2])
    # areaCu = dxCu · dyCu, areaCv = dyCv · dxCv (:1334, :1342; unmasked here, masking in Task 3)
    area_u = dx_u * dy_u
    area_v = dy_v * dx_v

    # ---- positions (MOM_grid_initialize.F90:222-254) ------------------------
    pos = {"t": (slice(1, None, 2), slice(1, None, 2)),
           "u": (slice(1, None, 2), slice(0, None, 2)),
           "v": (slice(0, None, 2), slice(1, None, 2)),
           "q": (slice(0, None, 2), slice(0, None, 2))}
    if geo:
        lon = {k: X[s] for k, s in pos.items()}
        lat = {k: Y[s] for k, s in pos.items()}
        lon0, lat0 = X[0, 0], Y[0, 0]
        # Arc-length positions from the SW corner, as create_spherical_grid;
        # longitude differences wrapped to (−180°, 180°] (dateline-safe).
        xy = {k: (R_EARTH * np.deg2rad(_wrap_lon(lon[k] - lon0)) * np.cos(np.deg2rad(lat0)),
                  R_EARTH * np.deg2rad(lat[k] - lat0)) for k in pos}
        # f = (2Ω) sin(π φ / 180)   (MOM_shared_initialization.F90:504-506)
        f_q = (2.0 * omega) * np.sin((np.pi * lat["q"]) / 180.0)
        f_t = (2.0 * omega) * np.sin((np.pi * lat["t"]) / 180.0)
        theta = _mom6_rotation_angle(lon["q"], lat["q"], lon["t"])
    else:
        nan = {k: np.full(X[s].shape, np.nan) for k, s in pos.items()}
        lon, lat = nan, {k: v.copy() for k, v in nan.items()}
        xy = {k: (X[s], Y[s]) for k, s in pos.items()}
        f_q = np.full(lat["q"].shape, np.nan)
        f_t = np.full(lat["t"].shape, np.nan)
        theta = _planar_rotation_angle(X[0::2, 0::2], Y[0::2, 0::2])
    cos_t = np.cos(theta)
    sin_t = -np.sin(theta)

    ds = _assemble_dataset(
        X_t=xy["t"][0], Y_t=xy["t"][1], X_u=xy["u"][0], Y_u=xy["u"][1],
        X_v=xy["v"][0], Y_v=xy["v"][1], X_q=xy["q"][0], Y_q=xy["q"][1],
        dx_t=dx_t, dy_t=dy_t, area_t=area_t,
        dx_u=dx_u, dy_u=dy_u, area_u=area_u,
        dx_v=dx_v, dy_v=dy_v, area_v=area_v,
        f_t=f_t, f_q=f_q, mask_t=mask_t,
        attrs={
            "grid_type": "mom6_supergrid",
            "nx": ni, "ny": nj,
            "window": [i0, i1, j0, j1],
            "omega": omega,
            "coordinates": "lonlat" if geo else "planar",
            "edge_metrics": "MOM6 linear extrapolation of half-segments "
                            "(MOM_grid_initialize.F90:1139-1169)",
            "rotation": "cos_t = cos_rot, sin_t = -sin_rot (MOM6); "
                        "u_east = cos_t*u_x - sin_t*v_y",
        },
    )
    dims = {"t": ["j_t", "i_t"], "u": ["j_t", "i_u"], "v": ["j_v", "i_t"], "q": ["j_v", "i_u"]}
    names = {"t": "T", "u": "U", "v": "V", "q": "q"}
    ds["dx_q"] = xr.DataArray(dx_q, dims=dims["q"], attrs={
        "units": "m", "long_name": "Grid-x distance between the v points around q (MOM6 dxBu)"})
    ds["dy_q"] = xr.DataArray(dy_q, dims=dims["q"], attrs={
        "units": "m", "long_name": "Grid-y distance between the u points around q (MOM6 dyBu)"})
    ds["area_q"] = xr.DataArray(area_q, dims=dims["q"], attrs={
        "units": "m2", "long_name": "q-cell area (MOM6 areaBu)"})
    for k in ("t", "u", "v", "q"):
        ds[f"lon_{k}"] = xr.DataArray(lon[k], dims=dims[k], attrs={
            "units": "degrees_east", "long_name": f"{names[k]}-point longitude"})
        ds[f"lat_{k}"] = xr.DataArray(lat[k], dims=dims[k], attrs={
            "units": "degrees_north", "long_name": f"{names[k]}-point latitude"})
    ds["cos_t"] = xr.DataArray(cos_t, dims=dims["t"], attrs={
        "units": "1", "long_name": "cos of the angle of grid-x counter-clockwise from east"})
    ds["sin_t"] = xr.DataArray(sin_t, dims=dims["t"], attrs={
        "units": "1", "long_name": "sin of the angle of grid-x counter-clockwise from east"})
    return ds


# ---------------------------------------------------------------------------
# Private helpers
# ---------------------------------------------------------------------------

def _check_supergrid_shape(nyp: int, nxp: int) -> None:
    """A supergrid has 2n+1 nodes per direction with n ≥ 1 model cells."""
    if nyp < 3 or nxp < 3 or nyp % 2 == 0 or nxp % 2 == 0:
        raise ValueError(f"supergrid node array shape ({nyp}, {nxp}) must be odd and ≥ 3")


def _is_geographic(units: str) -> bool:
    """True for lon/lat in degrees, False for planar metres."""
    u = str(units).lower()
    if "degree" in u:
        return True
    if u in ("m", "meter", "meters", "metre", "metres"):
        return False
    raise ValueError(f"supergrid x units '{units}' not understood (degrees or m)")


def _mask_from_depth(
    depth: np.ndarray | None,
    full_shape: tuple[int, int],
    win_shape: tuple[int, int],
    win: tuple[int, int, int, int],
) -> np.ndarray:
    """mask_t = depth > 0 (depth positive down [m]); all ocean without depth."""
    if depth is None:
        return np.ones(win_shape, dtype=np.int8)
    d = np.asarray(depth)
    j0, j1, i0, i1 = win
    if d.shape == full_shape:
        d = d[j0:j1, i0:i1]
    elif d.shape != win_shape:
        raise ValueError(f"depth shape {d.shape} matches neither the full grid "
                         f"{full_shape} nor the window {win_shape}")
    return (d > 0).astype(np.int8)


def _extrap_rows(v: np.ndarray) -> np.ndarray:
    """Add one row south and north: v_s = 2 v[0] − v[1], v_n = 2 v[-1] − v[-2]
    (MOM6 extrapolate_metric, MOM_grid_initialize.F90:1150-1158)."""
    out = np.empty((v.shape[0] + 2, v.shape[1]))
    out[1:-1] = v
    out[0] = 2.0 * v[0] - v[1]
    out[-1] = 2.0 * v[-1] - v[-2]
    return out


def _extrap_cols(v: np.ndarray) -> np.ndarray:
    """Add one column west and east, as :func:`_extrap_rows` (:1160-1168)."""
    out = np.empty((v.shape[0], v.shape[1] + 2))
    out[:, 1:-1] = v
    out[:, 0] = 2.0 * v[:, 0] - v[:, 1]
    out[:, -1] = 2.0 * v[:, -1] - v[:, -2]
    return out


def _wrap_lon(d: np.ndarray) -> np.ndarray:
    """Longitude difference wrapped to (−180°, 180°] [degrees]; values
    already inside are returned unchanged (bit for bit)."""
    d = np.asarray(d, dtype=float)
    out = d.copy()
    bad = (d <= -180.0) | (d > 180.0)
    out[bad] = 180.0 - np.mod(180.0 - d[bad], 360.0)
    return out


def _modulo_around_point(x: np.ndarray, xc: np.ndarray, period: float) -> np.ndarray:
    """x shifted by a multiple of ``period`` into [xc − L/2, xc + L/2)
    (MOM_shared_initialization.F90:629-642)."""
    return np.mod(x - (xc - 0.5 * period), period) + (xc - 0.5 * period)


def _mom6_rotation_angle(lon_q: np.ndarray, lat_q: np.ndarray, lon_t: np.ndarray) -> np.ndarray:
    """
    MOM6 clockwise angle θ of grid-north from true north at T points [rad].

    MOM_shared_initialization.F90:604-622 (GRID_ROTATION_ANGLE_BUGS=False):

        lon_scale = cos( (π/720) [(φ_SW + φ_NE) + (φ_SE + φ_NW)] )
        θ = atan2( lon_scale [(λ_NW − λ_SE) + (λ_NE − λ_SW)],
                   (φ_NW − φ_SE) + (φ_NE − φ_SW) )

    with corner longitudes taken modulo 360 around the T longitude.
    Fortran corner (I-1, J-1) of T cell (i, j) is Python q (j, i).
    """
    pi_720deg = np.arctan(1.0) / 180.0
    l_sw = _modulo_around_point(lon_q[:-1, :-1], lon_t, 360.0)   # lonB(1,1)
    l_se = _modulo_around_point(lon_q[:-1, 1:], lon_t, 360.0)    # lonB(2,1)
    l_nw = _modulo_around_point(lon_q[1:, :-1], lon_t, 360.0)    # lonB(1,2)
    l_ne = _modulo_around_point(lon_q[1:, 1:], lon_t, 360.0)     # lonB(2,2)
    p_sw, p_se, p_nw, p_ne = lat_q[:-1, :-1], lat_q[:-1, 1:], lat_q[1:, :-1], lat_q[1:, 1:]
    lon_scale = np.cos(pi_720deg * ((p_sw + p_ne) + (p_se + p_nw)))
    return np.arctan2(lon_scale * ((l_nw - l_se) + (l_ne - l_sw)),
                      (p_nw - p_se) + (p_ne - p_sw))


def _planar_rotation_angle(x_q: np.ndarray, y_q: np.ndarray) -> np.ndarray:
    """The MOM6 corner formula on a planar grid (x east, y north, metres;
    lon_scale = 1, no modulo) — synthetic grids only."""
    sw, se, nw, ne = (slice(None, -1), slice(None, -1)), (slice(None, -1), slice(1, None)), \
        (slice(1, None), slice(None, -1)), (slice(1, None), slice(1, None))
    return np.arctan2((x_q[nw] - x_q[se]) + (x_q[ne] - x_q[sw]),
                      (y_q[nw] - y_q[se]) + (y_q[ne] - y_q[sw]))


def _check_orthogonal(X: np.ndarray, Y: np.ndarray, geo: bool, mask_t: np.ndarray,
                      max_skew: float = MAX_SKEW) -> None:
    """
    Raise if a T cell's grid-x and grid-y directions are not orthogonal.

    pyOCEAN's C-grid operators assume an orthogonal curvilinear grid (as
    MOM6, ROMS).  Grid-x at T(j, i) is the vector between the u nodes
    (2j+1, 2i) → (2j+1, 2i+2), grid-y between the v nodes
    (2j, 2i+1) → (2j+2, 2i+1); on the sphere longitude differences are
    wrapped to ±180° and scaled by cos φ_T (local tangent plane).
    The skew |π/2 − ∠(x, y)| must not exceed ``max_skew`` [rad] in wet
    cells (``mask_t == 1``, all ones without a depth; land cells carry no
    flow).  NWA12 has land cells
    with aspect ratio 7 and skew up to 2.6e-3 rad (float32-exact node
    positions, inherited from the HYCOM grid; the skew halves when the
    stencil span doubles, so it is mostly rounding); its wet cells stay
    below 6.5e-4 rad.
    """
    ex = X[1::2, 2::2] - X[1::2, 0:-2:2]
    nx_ = Y[1::2, 2::2] - Y[1::2, 0:-2:2]
    ey = X[2::2, 1::2] - X[0:-2:2, 1::2]
    ny_ = Y[2::2, 1::2] - Y[0:-2:2, 1::2]
    if geo:
        c = np.cos(np.deg2rad(Y[1::2, 1::2]))
        ex = (np.mod(ex + 180.0, 360.0) - 180.0) * c
        ey = (np.mod(ey + 180.0, 360.0) - 180.0) * c
    cross = ex * ny_ - nx_ * ey
    dot = ex * ey + nx_ * ny_
    skew = np.abs(0.5 * np.pi - np.arctan2(np.abs(cross), dot))
    skew = np.where(mask_t == 1, skew, 0.0)
    if not np.all(np.isfinite(skew)):
        j, i = np.unravel_index(np.argmax(~np.isfinite(skew)), skew.shape)
        raise ValueError(f"degenerate supergrid cell at (j, i) = ({j}, {i})")
    k = int(np.argmax(skew))
    if skew.flat[k] > max_skew:
        j, i = np.unravel_index(k, skew.shape)
        raise ValueError(
            f"non-orthogonal supergrid: skew {skew.flat[k]:.3e} rad > max_skew {max_skew:g} "
            f"at T cell (j, i) = ({j}, {i})")
