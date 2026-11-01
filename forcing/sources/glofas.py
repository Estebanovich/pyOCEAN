"""
GloFAS river discharge → coastal volume sources of the model (regional chain Task 8).

Reads GloFAS daily mean river discharge (CEMS ``cems-glofas-historical``; v4 consolidated ``avg_dis`` or the v3
``dis24`` CEFI uses) [m³ s⁻¹], picks the river mouths (pour points) and maps each to a wet coastal T cell of the
model grid, following CEFI's ``tools/rivers/write_runoff_glofas.py`` (CEFI-regional-MOM6 git 36b8f49; reference
read ``docs/design/regional_refs/cefi_tools.md`` §5).  The result, :class:`RiverInput`, is a list of model cells
with a discharge time series; ``SigmaBaroclinic(..., runoff=RiverInput)`` adds Q/area_t to the real freshwater
flux of those cells (``core.components.runoff``).

CEFI's rule and how it is followed here
---------------------------------------
1. **Pour points** (``write_runoff_glofas.py:249-266``): GloFAS cells with local drain direction ``ldd == 5``
   (outlet) that touch the ocean (NaN in ``ldd``) in a 3 × 3 window, grown iteratively through neighbouring
   outlets (≤ 20 iterations); the window expansion sets the outermost GloFAS rows/columns to False
   (``expand_mask_true``, ``:61-79``).  With ``ldd=`` this is done exactly (:func:`cefi_pour_points`).
2. **Model coast cells** (``get_coast_mask``, ``:25-41``): wet cells with land in one of the four neighbours,
   excluding the outermost rows/columns of the grid (domain edges are not coasts) — :func:`coast_mask`.
3. **Discharge → model cells** (``:92-113``): CEFI converts the discharge of the pour points to a flux with a
   spherical GloFAS cell area and regrids it conservatively (xesmf) onto the model T cells.  Here each pour-point
   cell is split into ``n_sub`` × ``n_sub`` equal sub-cells in latitude/longitude, each carrying Q/n_sub², and each
   sub-cell goes to the model T cell that contains its centre (great-circle quadrilateral of the four corner
   points ``lon_q``, ``lat_q``).  This is the area-weighted (conservative) split of a GloFAS cell among the model
   cells it overlaps, to O(1/n_sub) in the overlap fractions, but it conserves the discharge **exactly**: CEFI's
   flux conversion uses an approximate GloFAS cell area while ESMF uses its own, so CEFI's Σ runoff·area need not
   equal Σ discharge (cefi_tools.md §5 remark b; class c).  Sub-cells outside the grid are dropped and the
   discharge is shared among the sub-cells inside (a mouth wholly outside the grid is dropped and reported).
4. NWA12-specific edits of CEFI in full-grid indices (``:115-144, 165-171``: Pacific drainage, southern boundary
   row, Hudson Bay, Mississippi rescaling, Susquehanna → lower Chesapeake) are **not** applied: they belong to the
   full NWA12 grid (ruling R8: MOM6's runoff is built on the full grid then cut; Task 11 compares per-mouth totals).
5. **Nearest coastal cell** (``:146-185``): every model cell's share moves to its nearest coast cell (xesmf
   ``nearest_s2d`` from the coast-cell centres; here the nearest in chord distance on the unit sphere, the same
   ordering as great-circle distance), which conserves the total exactly.

Pour points without the drain-direction map (substitute, ruling-style note)
--------------------------------------------------------------------------
CEFI needs the GloFAS local drain direction map (``ldd_file``, ``runoff_glofas.yaml``), which is not among the
local data (2026-11-01).  Without ``ldd=``, the outlets are found from the discharge itself (:func:`pit_pour_points`):

* an **outlet (pit)** is a land cell whose time-mean discharge is ≥ that of every land neighbour in its 3 × 3
  window.  In a D8 network (LISFLOOD's ldd) the discharge of a cell includes everything upstream, so every cell
  that drains into a neighbour has a neighbour with at least its discharge; a cell with none drains nowhere in
  the window, i.e. it is an ldd = 5 cell (exact for the time mean up to routing lags, which a mean over the
  window removes; ties are kept).  Cells on the outermost GloFAS rows/columns are excluded as in CEFI (there a
  "pit" is a river leaving the data box);
* a pit is **coastal** if it lies within ``max_inland_cells`` GloFAS cells (Euclidean index distance) of an ocean
  cell, ocean = NaN connected to the edge of the data box (inland NaN lakes are not ocean).  CEFI instead keeps
  pits connected to the ocean through a chain of adjacent ``ldd == 5`` cells; GloFAS v4 represents the Hudson and
  Delaware estuaries as land, with the main-stem outlet 4 (Hudson, 40.875 °N 73.925 °W) and 12 (Delaware,
  Philadelphia 39.875 °N 75.225 °W) cells from the NaN sea, which the chain reaches only if the estuary cells are
  outlets in ``ldd``.  With the 2017 data every pit with ≥ 0.5 m³ s⁻¹ inside the data box lies within 12 cells
  of the sea, so the default 15 is not binding there; pits beyond it are listed in the report.

This substitute differs from CEFI where coastal cells drain into each other along the shore (the smaller of two
adjacent coastal outlets is not a local maximum and is lost: its discharge is not counted, a few m³ s⁻¹ at most),
and in which estuary cells count as mouths.  Pass ``ldd=`` (GloFAS v4 ``ldd`` on the same 0.05° grid) for CEFI's
exact rule.  [Implementer's choice pending a controller ruling; class c for the parity comparison of Task 11.]

Time
----
GloFAS v4 ``valid_time`` is the **end** of the 24-h mean (the April file holds 2017-04-02 … 2017-05-01), so each
value is placed at the midpoint of its day, ``valid_time − 12 h`` (``stamp='end'``; ``'centre'`` and ``'start'``
for other conventions).  :class:`RiverInput` is linear in time between those midpoints and raises
``ValueError`` outside them (spec review focus 4).  ``open`` keeps the records bracketing [t0, t1].

Index order (j, i) for model T cells; GloFAS arrays (lat, lon) as stored.

References
----------
Harrigan, S. et al. (2020). GloFAS-ERA5 operational global river discharge reanalysis 1979–present. Earth Syst.
    Sci. Data 12, 2043–2060, doi:10.5194/essd-12-2043-2020 [verificar].
Ross, A. C. et al. (2023). A high-resolution, data-assimilative regional ocean model of the Northwest Atlantic
    (MOM6-NWA12). Geosci. Model Dev. 16, 6943–6985, doi:10.5194/gmd-16-6943-2023 [verificar] — CEFI's rivers tool.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import xarray as xr
from numpy.lib.stride_tricks import sliding_window_view
from scipy import ndimage
from scipy.spatial import cKDTree

_VARS = ("avg_dis", "dis24", "dis")
_STAMP_SHIFT_H = {"end": -12, "centre": 0, "start": 12}


# ---------------------------------------------------------------------------------------------------- container

@dataclass
class RiverInput:
    """
    River discharge at model T cells.

    Attributes
    ----------
    cells : (n, 2) int — (j, i) of the model T cells that receive runoff (unique, sorted).
    times : (nt,) datetime64[ns] — record times (midpoints of the daily means).
    Q : (nt, n) float — discharge into each cell [m³ s⁻¹].
    origin : datetime64[ns] — model time origin: model time t [s] is the date ``origin + t``.
    report : list[str] — how mouths were selected and moved (mouths dropped, mouths moved to a coast cell).
    mouth_lat, mouth_lon : (m,) — GloFAS pour-point centres [°N, °E in −180…180].
    mouth_Q : (nt, m) — discharge of each pour point [m³ s⁻¹] (Σ_m mouth_Q = Σ_n Q to round-off).
    mouth_cells : list of dict — per mouth, {cell index in ``cells``: fraction of its discharge}.
    method : 'ldd' (CEFI's rule) or 'pits' (substitute without the drain-direction map).
    """

    cells: np.ndarray
    times: np.ndarray
    Q: np.ndarray
    origin: np.datetime64
    report: list[str] = field(default_factory=list)
    mouth_lat: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mouth_lon: np.ndarray = field(default_factory=lambda: np.zeros(0))
    mouth_Q: np.ndarray = field(default_factory=lambda: np.zeros((0, 0)))
    mouth_cells: list = field(default_factory=list)
    method: str = "user"

    def __post_init__(self) -> None:
        self.cells = np.asarray(self.cells, dtype=np.int64).reshape(-1, 2)
        self.times = np.asarray(self.times, dtype="datetime64[ns]").ravel()
        self.Q = np.asarray(self.Q, dtype=float).reshape(self.times.size, self.cells.shape[0])
        self.origin = np.datetime64(self.origin, "ns")
        self._t = (self.times - self.origin).astype("timedelta64[ns]").astype(np.int64) / 1e9

    @property
    def seconds(self) -> np.ndarray:
        """Record times in seconds from ``origin``."""
        return self._t

    def coverage(self) -> tuple[float, float]:
        """First and last record [s from ``origin``]."""
        return float(self._t[0]), float(self._t[-1])

    def _date(self, t: float) -> np.datetime64:
        return self.origin + np.timedelta64(int(round(t * 1e9)), "ns")

    def discharge(self, t: float) -> np.ndarray:
        """
        Discharge of every cell at model time t [s from ``origin``], linear in time between records [m³ s⁻¹].
        Raises ``ValueError`` naming the coverage when t is outside it (spec review focus 4).
        """
        t0, t1 = self.coverage()
        if not (t0 <= t <= t1):
            raise ValueError(f"GloFAS runoff covers {self.times[0]} … {self.times[-1]} (daily-mean midpoints); "
                             f"model time {self._date(t)} (t = {t} s from {self.origin}) is outside")
        n = int(np.searchsorted(self._t, t, side="right")) - 1
        if n >= self._t.size - 1:
            return self.Q[-1].copy()
        w = (t - self._t[n]) / (self._t[n + 1] - self._t[n])
        return (1.0 - w) * self.Q[n] + w * self.Q[n + 1]

    def config(self) -> dict[str, str | int]:
        """Restart check: the river set (count and a hash of the cells) and the time origin."""
        return dict(runoff="on", runoff_n=int(self.cells.shape[0]),
                    runoff_cells=hashlib.sha1(np.ascontiguousarray(self.cells, np.int64).tobytes()).hexdigest(),
                    runoff_origin=str(self.origin))


# ---------------------------------------------------------------------------------------------------- pour points

def _expand_mask_true(mask: np.ndarray, window: int = 3) -> np.ndarray:
    """
    True where any cell of the ``window`` × ``window`` box is True; edges, where the box does not fit, False.
    Port of CEFI ``expand_mask_true`` (``write_runoff_glofas.py:61-79``).
    """
    wind = sliding_window_view(mask, (window, window)).any(axis=(2, 3))
    out = np.zeros(mask.shape, dtype=bool)
    i = (window - 1) // 2
    out[i:-i, i:-i] = wind
    return out


def cefi_pour_points(ldd: np.ndarray, max_iter: int = 20) -> np.ndarray:
    """
    CEFI's pour points (``write_runoff_glofas.py:249-266``): ``ldd == 5`` cells with an ocean (NaN) cell in their
    3 × 3 window, grown through adjacent ``ldd == 5`` cells until the count stops changing.
    """
    ldd = np.asarray(ldd, float)
    outlet = ldd == 5.0
    adjacent = outlet & _expand_mask_true(np.isnan(ldd))
    for _ in range(max_iter):
        n = int(adjacent.sum())
        adjacent = outlet & _expand_mask_true(adjacent)
        if int(adjacent.sum()) == n:
            return adjacent
    raise RuntimeError("CEFI pour-point growth did not converge in 20 iterations")


def pit_pour_points(q_mean: np.ndarray, max_inland_cells: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """
    Substitute pour points without the drain-direction map (module docstring): land cells (finite ``q_mean``) that
    are local maxima of the time-mean discharge in their 3 × 3 window of land cells, not on the outermost rows or
    columns, within ``max_inland_cells`` of the ocean (NaN connected to the edge of the array).  Returns the mask
    and the inland pits beyond the distance (for the report).
    """
    q = np.asarray(q_mean, float)
    land = np.isfinite(q)
    lab, _ = ndimage.label(~land)
    edge = np.unique(np.concatenate([lab[0], lab[-1], lab[:, 0], lab[:, -1]]))
    ocean = np.isin(lab, edge[edge > 0])
    qp = np.pad(np.where(land, q, -np.inf), 1, constant_values=-np.inf)
    nb = np.delete(sliding_window_view(qp, (3, 3)).reshape(*q.shape, 9), 4, axis=2).max(axis=2)
    pit = land & (q >= nb)
    pit[0, :] = pit[-1, :] = pit[:, 0] = pit[:, -1] = False
    dist = ndimage.distance_transform_edt(~ocean) if ocean.any() else np.full(q.shape, np.inf)
    near = dist <= max_inland_cells
    return pit & near, pit & ~near


def coast_mask(wet: np.ndarray) -> np.ndarray:
    """
    Model coast cells, CEFI ``get_coast_mask`` (``write_runoff_glofas.py:25-41``): wet cells with a land cell among
    the four neighbours, the outermost rows and columns excluded (open or closed edges are not coasts).
    """
    w = np.asarray(wet, bool)
    land = np.pad(~w, 1, constant_values=False)
    nb = land[1:-1, :-2] | land[1:-1, 2:] | land[:-2, 1:-1] | land[2:, 1:-1]
    c = w & nb
    c[0, :] = c[-1, :] = c[:, 0] = c[:, -1] = False
    return c


# ---------------------------------------------------------------------------------------------------- geometry

def _xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Unit vectors (…, 3) of lon/lat [°]."""
    lo, la = np.deg2rad(lon), np.deg2rad(lat)
    return np.stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)], axis=-1)


def _containing_cell(grid: xr.Dataset, lon: np.ndarray, lat: np.ndarray, k: int = 8) -> np.ndarray:
    """
    Flat index of the model T cell whose great-circle quadrilateral (corners ``lon_q``, ``lat_q``) contains each
    point, −1 outside the grid.  Candidates: the ``k`` nearest T centres; inside = the point on the inner side of the
    four edge great circles (the side of the cell centre).
    """
    lon_t, lat_t = (np.asarray(grid[v].values, float) for v in ("lon_t", "lat_t"))
    cq = _xyz(np.asarray(grid["lon_q"].values, float), np.asarray(grid["lat_q"].values, float))
    ny, nx = lon_t.shape
    ct = _xyz(lon_t, lat_t).reshape(-1, 3)
    p = _xyz(lon, lat)
    _, cand = cKDTree(ct).query(p, k=min(k, ct.shape[0]))
    cand = np.atleast_2d(cand.reshape(p.shape[0], -1))
    out = np.full(p.shape[0], -1, dtype=np.int64)
    for c in range(cand.shape[1]):
        idx = cand[:, c]
        todo = out < 0
        if not todo.any():
            break
        j, i = np.divmod(idx, nx)
        corners = [cq[j, i], cq[j, i + 1], cq[j + 1, i + 1], cq[j + 1, i]]
        inside = np.ones(p.shape[0], bool)
        for a in range(4):
            nrm = np.cross(corners[a], corners[(a + 1) % 4])
            s_c = np.einsum("ij,ij->i", nrm, ct[idx])
            s_p = np.einsum("ij,ij->i", nrm, p)
            inside &= s_p * np.sign(s_c) >= 0.0
        out[todo & inside] = idx[todo & inside]
    return out


# ---------------------------------------------------------------------------------------------------- reader

def _open(path_or_ds) -> xr.Dataset:
    if isinstance(path_or_ds, xr.Dataset):
        return path_or_ds
    if isinstance(path_or_ds, (str, Path)):
        paths = [path_or_ds]
    else:
        paths = list(path_or_ds)
    dss = [xr.open_dataset(p) for p in sorted(str(p) for p in paths)]
    tname = "valid_time" if "valid_time" in dss[0].dims else "time"
    return xr.concat(dss, dim=tname) if len(dss) > 1 else dss[0]


def _names(ds: xr.Dataset) -> tuple[str, str, str, str]:
    var = next((v for v in _VARS if v in ds.data_vars), None)
    if var is None:
        raise ValueError(f"GloFAS file without a discharge variable (looked for {', '.join(_VARS)})")
    tname = "valid_time" if "valid_time" in ds[var].dims else "time"
    la = "latitude" if "latitude" in ds[var].dims else "lat"
    lo = "longitude" if "longitude" in ds[var].dims else "lon"
    return var, tname, la, lo


def glofas_mouths(path_or_ds, grid: xr.Dataset, t0, t1, min_mean_discharge: float = 1.0, *,
                  origin=None, ldd: np.ndarray | xr.DataArray | None = None, wet: np.ndarray | None = None,
                  stamp: str = "end", max_inland_cells: float = 15.0, n_sub: int = 5) -> RiverInput:
    """
    River mouths of a GloFAS discharge file mapped to the model's coastal T cells (CEFI rule, module docstring).

    Parameters
    ----------
    path_or_ds : path, list of paths or Dataset with ``avg_dis`` (v4) / ``dis24`` (v3) [m³ s⁻¹] on
        (time, latitude, longitude); longitudes 0–360 or −180–180.
    grid : model grid Dataset with ``lon_t``, ``lat_t``, ``lon_q``, ``lat_q`` (``grid_from_supergrid``) and
        ``mask_t`` (or pass ``wet``).
    t0, t1 : model run window (dates); the records bracketing it are kept, and it must lie inside the data.
    min_mean_discharge : pour points whose mean discharge over the kept records is below it [m³ s⁻¹] are dropped
        (reported with their total).
    origin : model time origin (t = 0); default ``t0``.
    ldd : GloFAS local drain direction on the same grid as the discharge: CEFI's exact pour points.  None: the
        substitute (pits of the discharge, :func:`pit_pour_points`).
    wet : (ny, nx) bool wet T cells (H > 0); default ``grid.mask_t``.
    stamp : what the file's time stamps mark: 'end' (GloFAS v4 ``valid_time``), 'centre' or 'start' of the day.
    max_inland_cells : substitute only, farthest pit from the sea [GloFAS cells].
    n_sub : sub-cells per side of a GloFAS cell for the conservative split.

    Returns
    -------
    RiverInput
    """
    if stamp not in _STAMP_SHIFT_H:
        raise ValueError(f"stamp={stamp!r}: use 'end', 'centre' or 'start'")
    ds = _open(path_or_ds)
    var, tname, la, lo = _names(ds)
    da = ds[var].transpose(tname, la, lo)
    mid = da[tname].values.astype("datetime64[ns]") + np.timedelta64(_STAMP_SHIFT_H[stamp], "h")
    order = np.argsort(mid, kind="stable")
    mid = mid[order]
    t0, t1 = np.datetime64(t0, "ns"), np.datetime64(t1, "ns")
    if t0 < mid[0] or t1 > mid[-1] or t1 < t0:
        raise ValueError(f"GloFAS covers {mid[0]} … {mid[-1]} (daily-mean midpoints, stamp={stamp!r}); the window "
                         f"{t0} … {t1} is not inside")
    a = int(np.searchsorted(mid, t0, side="right")) - 1
    b = int(np.searchsorted(mid, t1, side="left"))
    sel = order[a:b + 1]
    times = mid[a:b + 1]
    q = np.asarray(da.values, float)[sel]                                      # (nt, nlat, nlon)
    glat = np.asarray(da[la].values, float)
    glon = np.asarray(da[lo].values, float)
    glon = np.where(glon >= 180.0, glon - 360.0, glon)
    q_mean = q.mean(axis=0)                                                   # NaN where any record is NaN (sea)
    report: list[str] = []
    if ldd is not None:
        ldd_a = np.asarray(ldd.values if isinstance(ldd, xr.DataArray) else ldd, float)
        if ldd_a.shape != q_mean.shape:
            raise ValueError(f"ldd shape {ldd_a.shape} differs from the discharge grid {q_mean.shape}")
        pour, method = cefi_pour_points(ldd_a), "ldd"
        report.append(f"pour points: CEFI rule (ldd == 5 next to the sea, grown through outlets): {int(pour.sum())}")
    else:
        pour, far = pit_pour_points(q_mean, max_inland_cells)
        method = "pits"
        report.append(f"pour points: substitute without ldd (pits of the mean discharge within {max_inland_cells:g} "
                      f"cells of the sea): {int(pour.sum())}")
        for jg, ig in zip(*np.nonzero(far & (q_mean >= min_mean_discharge))):
            report.append(f"inland pit not used: {glat[jg]:.3f}N {glon[ig]:.3f}E, mean Q {q_mean[jg, ig]:.1f} m3/s")
    pour &= np.isfinite(q_mean)
    small = pour & (q_mean < min_mean_discharge)
    if small.any():
        report.append(f"dropped {int(small.sum())} pour points with mean Q < {min_mean_discharge:g} m3/s "
                      f"(total {float(q_mean[small].sum()):.2f} m3/s)")
    pour &= ~small
    jg, ig = np.nonzero(pour)

    # ---- sub-cells → containing model cell → nearest coast cell
    wet_m = np.asarray(grid["mask_t"].values if wet is None else wet).astype(bool)
    ny, nx = wet_m.shape
    coast = coast_mask(wet_m)
    if not coast.any():
        raise ValueError("the grid has no coast cell (wet cell next to land, away from the edges)")
    lon_t, lat_t = (np.asarray(grid[v].values, float) for v in ("lon_t", "lat_t"))
    ct = _xyz(lon_t, lat_t).reshape(-1, 3)
    coast_idx = np.flatnonzero(coast.ravel())
    _, near = cKDTree(ct[coast_idx]).query(ct)
    nearest_coast = coast_idx[near]                                           # flat model index → flat coast index
    dlat = float(np.median(np.abs(np.diff(glat)))) if glat.size > 1 else 0.05
    dlon = float(np.median(np.abs(np.diff(glon)))) if glon.size > 1 else 0.05
    off = (np.arange(n_sub) + 0.5) / n_sub - 0.5
    oy, ox = (g.ravel() for g in np.meshgrid(off, off, indexing="ij"))
    m = jg.size
    if m:
        plat = (glat[jg][:, None] + oy[None] * dlat).ravel()
        plon = (glon[ig][:, None] + ox[None] * dlon).ravel()
        home = _containing_cell(grid, plon, plat).reshape(m, n_sub * n_sub)
    else:
        home = np.zeros((0, n_sub * n_sub), dtype=np.int64)
    mouth_targets: list[dict[int, float]] = []
    keep = np.zeros(m, bool)
    for k in range(m):
        h = home[k][home[k] >= 0]
        if h.size == 0:
            mouth_targets.append({})
            continue
        keep[k] = True
        tgt, cnt = np.unique(nearest_coast[h], return_counts=True)
        mouth_targets.append({int(c): float(n) / h.size for c, n in zip(tgt, cnt)})
    n_out = int((~keep).sum())
    if n_out:
        report.append(f"dropped {n_out} pour points outside the model grid "
                      f"(total {float(q_mean[jg[~keep], ig[~keep]].sum()):.2f} m3/s)")
    jg, ig = jg[keep], ig[keep]
    mouth_targets = [t for t, k_ in zip(mouth_targets, keep) if k_]
    home = home[keep]
    cells_flat = np.array(sorted({c for t in mouth_targets for c in t}), dtype=np.int64)
    pos = {int(c): n for n, c in enumerate(cells_flat)}
    mouth_Q = q[:, jg, ig]                                                    # (nt, m)
    W = np.zeros((len(mouth_targets), cells_flat.size))
    for k, tgt in enumerate(mouth_targets):
        for c, f in tgt.items():
            W[k, pos[c]] = f
    Q = mouth_Q @ W if cells_flat.size else np.zeros((times.size, 0))
    cells = np.stack(np.divmod(cells_flat, nx), axis=1) if cells_flat.size else np.zeros((0, 2), np.int64)
    # ---- report the moves (every pour point is on GloFAS land, so all are moved to a coast cell)
    order_m = np.argsort(-q_mean[jg, ig])
    for k in order_m:
        h = home[k][home[k] >= 0]
        hj, hi = np.divmod(int(np.bincount(h).argmax()), nx)
        why = ("land" if not wet_m[hj, hi] else
               "edge row/column" if hj in (0, ny - 1) or hi in (0, nx - 1) else
               "wet, not coastal" if not coast[hj, hi] else "coast")
        dest = ", ".join(f"({c // nx}, {c % nx}) {f:.2f}" for c, f in mouth_targets[k].items())
        report.append(f"mouth {glat[jg[k]]:.3f}N {glon[ig[k]]:.3f}E, mean Q {q_mean[jg[k], ig[k]]:.2f} m3/s: "
                      f"in cell ({hj}, {hi}) [{why}] -> {dest}")
    return RiverInput(cells=cells, times=times, Q=Q, origin=t0 if origin is None else origin, report=report,
                      mouth_lat=glat[jg], mouth_lon=glon[ig], mouth_Q=mouth_Q,
                      mouth_cells=[{pos[c]: f for c, f in t.items()} for t in mouth_targets], method=method)
