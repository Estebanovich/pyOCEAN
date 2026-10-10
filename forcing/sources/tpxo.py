"""
Reader for TPXO tidal atlases (NetCDF, OTIS/atlas layout).

Loads complex harmonic constants of elevation and barotropic transport from
the TPXO atlas files (Egbert & Erofeeva 2002), e.g. TPXO10-atlas-v2 at 1/30°
as used for the HYCOM-IOA tidal ports (~/HYCOM/GUIA_mareas_PAC_MEX.md):

    h_<c>_tpxo10_atlas_30_v2.nc   hRe, hIm         elevation          (usually mm)
    u_<c>_tpxo10_atlas_30_v2.nc   URe, UIm, VRe, VIm   transport       (usually cm² s⁻¹)
    grid_tpxo10atlas_v2.nc        lon_z, lat_z, lon_u, lat_u, lon_v, lat_v, hz

Arrays are stored (nx, ny) = (lon, lat), longitudes 0–360 °E, on an Arakawa C
grid: U at (lon_u, lat_z), V at (lon_z, lat_v).  Units are read from each
variable's ``units`` attribute and converted to SI.  The complex constant is
h = Re + i·Im, the convention of OTPS and of ``forcing.tides.predict``
(phase G = atan2(−Im, Re)).

The TPXO licence is academic: the files are read in place and never copied.

References
----------
Egbert, G.D. & Erofeeva, S.Y. (2002). Efficient inverse modeling of
    barotropic ocean tides. J. Atmos. Oceanic Technol. 19, 183–204.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import xarray as xr

_LENGTH = {"mm": 1e-3, "millimeter": 1e-3, "millimeters": 1e-3, "millimetre": 1e-3,
           "cm": 1e-2, "centimeter": 1e-2, "centimeters": 1e-2,
           "m": 1.0, "meter": 1.0, "meters": 1.0, "metre": 1.0}


def _si_factor(units: str, kind: str) -> float:
    """SI factor for 'length' (elevation) or 'transport' (length²/time) unit strings."""
    u = units.strip().lower()
    if kind == "length":
        if u in _LENGTH:
            return _LENGTH[u]
    else:
        m = re.match(r"^(\w+?)\s*(?:\^2|\*\*2|2)\s*(?:/|\s)\s*(?:s|sec|second)(?:-1)?$", u.replace(" s-1", "/s"))
        if m and m.group(1) in _LENGTH:
            return _LENGTH[m.group(1)] ** 2
    raise ValueError(f"unrecognised TPXO units {units!r} for {kind}")


def _constituent(path: Path) -> str:
    m = re.search(r"(?:^|_)([a-z0-9]+)_tpxo", path.name.lower()) or re.search(r"_([a-z0-9]+)\.nc$", path.name.lower())
    if not m:
        raise ValueError(f"cannot infer the constituent from file name {path.name}")
    return m.group(1)


def _axis(ds: xr.Dataset, grid: xr.Dataset | None, name: str) -> np.ndarray:
    src = ds if name in ds else grid
    if src is None or name not in src:
        raise ValueError(f"coordinate {name} not found (pass the TPXO grid file)")
    return np.asarray(src[name].values, dtype=float)


def _to_lat_lon(x: np.ndarray, nlon: int, nlat: int) -> np.ndarray:
    """TPXO stores (nx, ny); return (ny, nx)."""
    if x.shape == (nlon, nlat):
        return x.T
    if x.shape == (nlat, nlon):
        return x
    raise ValueError(f"array shape {x.shape} does not match (lon, lat) = ({nlon}, {nlat})")


def _subset(lon: np.ndarray, lat: np.ndarray, lon_range, lat_range) -> tuple[np.ndarray, np.ndarray]:
    """Indices (sorted by ±180 longitude) covering the ranges, plus one cell margin."""
    lon180 = ((lon + 180.0) % 360.0) - 180.0
    order = np.argsort(lon180, kind="stable")
    ls = lon180[order]
    ii = np.nonzero((ls >= lon_range[0]) & (ls <= lon_range[1]))[0]
    jj = np.nonzero((lat >= lat_range[0]) & (lat <= lat_range[1]))[0]
    if ii.size == 0 or jj.size == 0:
        raise ValueError("TPXO files do not cover the requested region")
    ii = order[max(ii[0] - 1, 0):min(ii[-1] + 2, ls.size)]
    jj = np.arange(max(jj[0] - 1, 0), min(jj[-1] + 2, lat.size))
    return ii, jj


def read_tpxo(
    h_files: list[str | Path],
    u_files: list[str | Path] | None,
    grid_file: str | Path | None,
    lon_range: tuple[float, float],
    lat_range: tuple[float, float],
) -> xr.Dataset:
    """
    Read TPXO elevation (and optionally transport) constants over a region.

    Parameters
    ----------
    h_files : list of paths
        ``h_<c>_*.nc`` files, one per constituent.
    u_files : list of paths or None
        ``u_<c>_*.nc`` transport files (same constituents), or None.
    grid_file : path or None
        TPXO grid file with the coordinates (needed if the h/u files lack them).
    lon_range, lat_range : (min, max)
        Region [°E in −180–180, °N].

    Returns
    -------
    xr.Dataset
        ``h`` (constituent, lat, lon) complex [m]; with ``u_files`` also
        ``U`` (constituent, lat, lon_u) and ``V`` (constituent, lat_v, lon)
        complex transports [m² s⁻¹].  Longitudes in −180–180 °E.
    """
    grid = xr.open_dataset(grid_file) if grid_file is not None else None
    try:
        h_list, U_list, V_list, names = [], [], [], []
        for k, hp in enumerate(map(Path, h_files)):
            with xr.open_dataset(hp, mask_and_scale=True) as ds:
                lon, lat = _axis(ds, grid, "lon_z"), _axis(ds, grid, "lat_z")
                ii, jj = _subset(lon, lat, lon_range, lat_range)
                f_re = _si_factor(ds["hRe"].attrs.get("units", ""), "length")
                f_im = _si_factor(ds["hIm"].attrs.get("units", ""), "length")
                h = (_to_lat_lon(np.nan_to_num(ds["hRe"].values.astype(float)), lon.size, lat.size) * f_re
                     + 1j * _to_lat_lon(np.nan_to_num(ds["hIm"].values.astype(float)), lon.size, lat.size) * f_im)
                h_list.append(h[np.ix_(jj, ii)])
                lon_z, lat_z = lon[ii], lat[jj]
            names.append(_constituent(hp))
            if u_files is not None:
                with xr.open_dataset(Path(u_files[k])) as du:
                    lu, lv = _axis(du, grid, "lon_u"), _axis(du, grid, "lat_v")
                    fu = _si_factor(du["URe"].attrs.get("units", ""), "transport")
                    fv = _si_factor(du["VRe"].attrs.get("units", ""), "transport")
                    U = _to_lat_lon(du["URe"].values + 1j * du["UIm"].values, lu.size, lat.size) * fu
                    V = _to_lat_lon(du["VRe"].values + 1j * du["VIm"].values, lon.size, lv.size) * fv
                    U_list.append(np.nan_to_num(U)[np.ix_(jj, ii)])
                    V_list.append(np.nan_to_num(V)[np.ix_(jj, ii)])
                    lon_u, lat_v = lu[ii], lv[jj]
    finally:
        if grid is not None:
            grid.close()

    to180 = lambda x: ((x + 180.0) % 360.0) - 180.0                         # noqa: E731
    out = xr.Dataset(
        {"h": (("constituent", "lat", "lon"), np.stack(h_list), {"units": "m"})},
        coords={"constituent": names, "lat": lat_z, "lon": to180(lon_z)},
    )
    if u_files is not None:
        out["U"] = xr.DataArray(np.stack(U_list), dims=("constituent", "lat", "lon_u"),
                                coords={"lon_u": to180(lon_u)}, attrs={"units": "m2 s-1"})
        out["V"] = xr.DataArray(np.stack(V_list), dims=("constituent", "lat_v", "lon"),
                                coords={"lat_v": lat_v}, attrs={"units": "m2 s-1"})
    out.attrs["source_files"] = ", ".join(Path(p).name for p in h_files)
    return out
