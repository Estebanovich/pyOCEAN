"""
Native HYCOM archive adapter: archv / archm files on hybrid layers → canonical
ocean on fixed z levels.

Used for initial and boundary conditions from a parent HYCOM run — GOFS /
ESPC native archives or the HYCOM-IOA nests themselves — and for comparing
pyOCEAN against HYCOM output.

Archive conventions (HYCOM-src ``mod_archiv.F90``; HYCOM-tools ``archv2data3z.f``)
------------------------------------------------------------------------------
* ``srfhgt`` is geopotential [m² s⁻²]:   η = srfhgt / (thref · 9806) = srfhgt / 9.806
* ``thknss`` is layer thickness in pressure units [Pa]:   h = dp / 9806
  (``onem`` = g / svref = 9806 Pa per metre)
* ``temp`` is potential temperature (HYCOM's prognostic variable), ``salin``
  practical salinity.
* In instantaneous archives (``archv``, artype 1) ``u-vel.``/``v-vel.`` are
  baroclinic and the total velocity is u-vel. + ``u_btrop``
  (archv2data3z.f, lines 554–559); mean archives (``archm``, artype 2)
  already contain the total velocity.
* C-grid: u(i, j) on the western face and v(i, j) on the southern face of
  p-cell (i, j).  Velocities are averaged to p-points.
* Time: model day; for yrflag = 3 it counts days since 1900-12-31 00:00.

Layer → z remapping
-------------------
Each target depth d takes the value of the layer that contains it
(piecewise-constant sampling, the default ``itype`` of archv2data3z):

    z_{k+½} = Σ_{m≤k} h_m,    value(d) = ψ_k  for  z_{k−½} ≤ d < z_{k+½}

Vanished layers (h = 0) are skipped automatically; depths below the local
sea floor are NaN (fill them with ``forcing.interp.to_model_levels``).

Only rectilinear (uniform lat–lon or Mercator) HYCOM grids are supported.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import numpy as np
import xarray as xr

from forcing.canonical import CANONICAL_VARS
from forcing.sources.base import BBox, ForcingSource, register_source, subset_bbox
from forcing.sources.hycom_ab import read_b, read_field

ONEM: float = 9806.0           # Pa per metre of layer thickness (HYCOM onem)
G_HYCOM: float = 9.806         # m s⁻²  (thref·9806 with thref = 1e-3)
_RECTILINEAR_TOL_DEG: float = 1e-4

#: The 40 standard depths of the GOFS 3.1 / ESPC-D-V02 z-level products [m].
GOFS_DEPTHS = np.array([
    0, 2, 4, 6, 8, 10, 12, 15, 20, 25, 30, 35, 40, 45, 50, 60, 70, 80, 90, 100,
    125, 150, 200, 250, 300, 350, 400, 500, 600, 700, 800, 900, 1000, 1250, 1500,
    2000, 2500, 3000, 4000, 5000], dtype=float)


def remap_layers_to_z(values: np.ndarray, h: np.ndarray, depths: np.ndarray) -> np.ndarray:
    """
    Sample layered fields at fixed depths (piecewise-constant within layers).

    Parameters
    ----------
    values : np.ndarray, shape (nk, ...)
        Layer values.
    h : np.ndarray, shape (nk, ...)
        Layer thicknesses [m]; NaN on land.
    depths : np.ndarray, shape (nz,)
        Target depths [m, positive down].

    Returns
    -------
    np.ndarray, shape (nz, ...)
        Values at the target depths; NaN below the sea floor and on land.
    """
    nk = h.shape[0]
    bottoms = np.cumsum(h, axis=0)                               # z_{k+½}, (nk, ...)
    out = np.full((depths.size,) + h.shape[1:], np.nan)
    sea = np.isfinite(bottoms[-1])                               # land columns: NaN thickness
    for n, d in enumerate(depths):
        k = np.sum(bottoms <= d, axis=0)                         # layers entirely above d
        wet = (k < nk) & sea
        layer = np.where(wet, k, 0)
        sampled = np.take_along_axis(values, layer[None, ...], axis=0)[0]
        out[n] = np.where(wet, sampled, np.nan)
    return out


@register_source("hycom_archive")
class HYCOMArchiveSource(ForcingSource):
    """
    HYCOM native archives (``archv.*`` / ``archm.*`` .a/.b pairs).

    Parameters
    ----------
    paths : list of str or Path
        Archive ``.b`` files (their ``.a`` files must sit alongside).
    grid : str or Path
        ``regional.grid.b`` of the archive domain.
    depths : np.ndarray
        Target z levels [m]; default the 40 GOFS/ESPC depths.
    velocity : {'auto', 'baroclinic', 'total'}
        What ``u-vel.``/``v-vel.`` contain.  'auto' = total for files named
        ``archm*``, baroclinic otherwise.
    """

    kind = "ocean"

    def __init__(self, paths: list[str | Path], grid: str | Path,
                 depths: np.ndarray = GOFS_DEPTHS, velocity: str = "auto") -> None:
        if not paths:
            raise ValueError("HYCOMArchiveSource needs at least one archive in paths")
        if velocity not in ("auto", "baroclinic", "total"):
            raise ValueError(f"velocity must be 'auto', 'baroclinic' or 'total', got {velocity!r}")
        self.paths = [Path(p) for p in paths]
        self.grid = Path(grid)
        self.depths = np.asarray(depths, dtype=float)
        self.velocity = velocity

    # ------------------------------------------------------------------
    @staticmethod
    def archive_time(path_b: str | Path) -> np.datetime64:
        """Valid time of an archive from its model day (yrflag = 3 only)."""
        b = read_b(path_b)
        yrflag = b.header.get("yrflag", 3)
        if yrflag != 3:
            raise NotImplementedError(f"{Path(path_b).name}: yrflag={yrflag}; only yrflag=3 is supported")
        day = b.records[0].day
        return np.datetime64("1900-12-31T00:00", "ns") + np.timedelta64(int(round(day * 86400e3)), "ms")

    def open_all(self) -> xr.Dataset:
        """Canonical dataset of every archive over the whole domain."""
        ds = xr.concat([self._read_archive(p) for p in self.paths], dim="time").sortby("time")
        return self.to_canonical(ds)

    # ------------------------------------------------------------------
    def open(self, t0: datetime, t1: datetime, bbox: BBox) -> xr.Dataset:
        """Read archives with valid time in [t0, t1] and subset to ``bbox``."""
        t0, t1 = np.datetime64(t0, "ns"), np.datetime64(t1, "ns")
        chosen = [p for p in self.paths if t0 <= self.archive_time(p) <= t1]
        if not chosen:
            raise ValueError(f"no HYCOM archive with valid time in [{t0}, {t1}]")
        ds = xr.concat([self._read_archive(p) for p in chosen], dim="time").sortby("time")
        return subset_bbox(ds, bbox)

    def to_canonical(self, ds: xr.Dataset) -> xr.Dataset:
        """Attach canonical units and provenance (fields are already converted)."""
        for name in ds.data_vars:
            meta = CANONICAL_VARS["ocean"][name]
            ds[name].attrs = {"units": meta["units"], "standard_name": meta["standard_name"]}
        ds["temp"].attrs["comment"] = "HYCOM potential temperature, sampled from layers"
        ds.attrs["source_product"] = "HYCOM native archive (layers remapped to z)"
        ds.attrs["source_files"] = ", ".join(p.name for p in self.paths)
        return ds

    # ------------------------------------------------------------------
    def _axes(self) -> tuple[np.ndarray, np.ndarray]:
        plon = read_field(self.grid, "plon")
        plat = read_field(self.grid, "plat")
        lon, lat = plon[0, :], plat[:, 0]
        if (np.nanmax(np.abs(plon - lon[None, :])) > _RECTILINEAR_TOL_DEG
                or np.nanmax(np.abs(plat - lat[:, None])) > _RECTILINEAR_TOL_DEG):
            raise NotImplementedError(f"{self.grid.name}: HYCOM grid is not rectilinear")
        return lat, lon

    def _is_total_velocity(self, path_b: Path) -> bool:
        if self.velocity == "auto":
            return path_b.name.startswith("archm")
        return self.velocity == "total"

    def _read_archive(self, path_b: Path) -> xr.Dataset:
        lat, lon = self._axes()
        b = read_b(path_b)
        layers = sorted({r.k for r in b.records if r.name == "thknss"})

        def stack(name: str) -> np.ndarray:
            return np.stack([read_field(path_b, name, k=k) for k in layers])

        h = stack("thknss") / ONEM
        u = stack("u-vel.")
        v = stack("v-vel.")
        if not self._is_total_velocity(path_b):
            u = u + read_field(path_b, "u_btrop", k=0)[None]
            v = v + read_field(path_b, "v_btrop", k=0)[None]
        u_p = _faces_to_centres(u, axis=-1)
        v_p = _faces_to_centres(v, axis=-2)

        fields = {
            "temp": remap_layers_to_z(stack("temp"), h, self.depths),
            "salt": remap_layers_to_z(stack("salin"), h, self.depths),
            "u": remap_layers_to_z(u_p, h, self.depths),
            "v": remap_layers_to_z(v_p, h, self.depths),
        }
        dims = ("depth", "lat", "lon")
        ds = xr.Dataset({k: (dims, x) for k, x in fields.items()},
                        coords={"depth": self.depths, "lat": lat, "lon": lon})
        ds["ssh"] = (("lat", "lon"), read_field(path_b, "srfhgt", k=0) / G_HYCOM)
        return ds.expand_dims(time=[self.archive_time(path_b)])


def _faces_to_centres(x: np.ndarray, axis: int) -> np.ndarray:
    """
    Average C-grid face values to p-points: ψ_p(i) = ½ [ψ(i) + ψ(i+1)].

    Face i is the western (or southern) face of cell i, so cell i lies between
    faces i and i+1.  At the last cell, or where one face is land (NaN), the
    available face is used.
    """
    nxt = np.roll(x, -1, axis=axis)
    idx = [slice(None)] * x.ndim
    idx[axis] = -1
    nxt[tuple(idx)] = np.nan                      # no face beyond the last cell
    with np.errstate(invalid="ignore"):
        return np.where(np.isnan(x), nxt, np.where(np.isnan(nxt), x, 0.5 * (x + nxt)))
