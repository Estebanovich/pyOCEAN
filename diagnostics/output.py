"""
CF-1.8 / SGRID model output for the C-grid models, in NetCDF4 or Zarr.

Why: output friction is a recurrent complaint in community models (one file per
processor, missing vertical coordinate metadata, broken diagnostics; MOM6,
MITgcm, SCHISM, Veros trackers — docs/design/vision_roadmap.md §3b item 7), and
ML pipelines (anemoi-datasets, Pangeo) need self-describing, chunked stores.

Conventions
-----------
* CF-1.8 standard names and units; time as "seconds since <start>".
* Vertical: σ-layers as CF ``ocean_sigma_coordinate`` (z = η + σ (h + η),
  formula_terms "sigma: s_rho eta: eta depth: h").
* Horizontal staggering: SGRID 2-D grid topology (Arakawa C) with dimensions
  xh/yh (cell centres), xq (u-faces, nx+1) and yq (v-faces, ny+1) — the names
  used by MOM6 output, understood by xgcm.

Formats
-------
* ``fmt="zarr"`` appends each record to the store (``append_dim="time"``):
  long runs never hold the history in memory and a crash keeps what was written.
* ``fmt="netcdf"`` buffers the records and writes one NetCDF4 file on ``close``.

Arrays are written as float64, so reading back is exact.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

# name → (staggering, units, standard_name or None, long_name)
FIELDS = {
    "eta": ("t2", "m", "sea_surface_height_above_geoid", "sea surface height"),
    "u": ("u3", "m s-1", "sea_water_x_velocity", "layer velocity at u-faces"),
    "v": ("v3", "m s-1", "sea_water_y_velocity", "layer velocity at v-faces"),
    "T": ("t3", "degC", "sea_water_potential_temperature", "temperature"),
    "S": ("t3", "1e-3", "sea_water_salinity", "salinity"),
    "ubar": ("u2", "m s-1", "barotropic_sea_water_x_velocity", "depth-mean velocity at u-faces"),
    "vbar": ("v2", "m s-1", "barotropic_sea_water_y_velocity", "depth-mean velocity at v-faces"),
    "mix_num": ("t3", "degC2 m3 s-1", None, "online numerical mixing (Burchard & Rennau 2008)"),
}
DIMS = {"t2": ("yh", "xh"), "u2": ("yh", "xq"), "v2": ("yq", "xh"),
        "t3": ("s_rho", "yh", "xh"), "u3": ("s_rho", "yh", "xq"), "v3": ("s_rho", "yq", "xh")}


class OutputWriter:
    """
    path : output file (.nc) or store (.zarr);  h_grid : horizontal grid;  H : bathymetry [m] (NaN = wall)
    n_layers : number of σ-layers (0 for 2-D runs);  start : ISO date of t = 0
    dsigma : layer fractions Δσ_k, k = 0 at the bed (default uniform; the model's ``core.dsigma``)
    """

    def __init__(self, path: str | Path, h_grid: xr.Dataset, H: np.ndarray, n_layers: int = 0,
                 start: str = "2000-01-01T00:00:00", fmt: str = "zarr", title: str = "pyOCEAN output",
                 attrs: dict | None = None, dsigma=None) -> None:
        if fmt not in ("zarr", "netcdf"):
            raise ValueError("fmt must be 'zarr' or 'netcdf'")
        self.path, self.fmt, self.start = Path(path), fmt, np.datetime64(start, "s")
        self.ny, self.nx = H.shape
        self.n = n_layers
        self.dsigma = np.full(n_layers, 1.0 / max(n_layers, 1)) if dsigma is None else np.asarray(dsigma, float)
        self._records: list[xr.Dataset] = []
        self._first = True
        self.static = self._static(h_grid, np.asarray(H, float), title, attrs or {})

    # ------------------------------------------------------------------ metadata
    def _static(self, hg: xr.Dataset, H: np.ndarray, title: str, attrs: dict) -> xr.Dataset:
        def coord(key, units, std, axis, dim):
            v = hg[key].values
            v = v if v.ndim == 1 else (v[0, :] if dim.startswith("x") else v[:, 0])
            return xr.Variable((dim,), np.asarray(v, float), {"units": units, "standard_name": std, "axis": axis})
        lon = "lon_t" in hg                                      # spherical grid: CF longitude/latitude
        cx = ("degrees_east", "longitude") if lon else ("m", "projection_x_coordinate")
        cy = ("degrees_north", "latitude") if lon else ("m", "projection_y_coordinate")
        kx = "lon_t" if lon else "x_t"
        ky = "lat_t" if lon else "y_t"
        kxu = "lon_u" if lon else "x_u"
        kyv = "lat_v" if lon else "y_v"
        coords = {"xh": coord(kx, *cx, "X", "xh"), "yh": coord(ky, *cy, "Y", "yh"),
                  "xq": coord(kxu, *cx, "X", "xq"), "yq": coord(kyv, *cy, "Y", "yq")}
        data = {
            "h": xr.Variable(("yh", "xh"), H, {"units": "m", "standard_name": "sea_floor_depth_below_geoid",
                                               "long_name": "bathymetry (< 0: floodable land, NaN: wall)"}),
            "mask": xr.Variable(("yh", "xh"), np.isfinite(H).astype(np.int8),
                                {"long_name": "active cell mask", "flag_values": "0 1", "flag_meanings": "land water"}),
            "grid": xr.Variable((), np.int32(0), {
                "cf_role": "grid_topology", "topology_dimension": 2, "node_dimensions": "xq yq",
                "face_dimensions": "xh: xq (padding: both) yh: yq (padding: both)",
                "edge1_dimensions": "xq: xh yh",                    # u-faces
                "edge2_dimensions": "xh yq: yh",                    # v-faces
                "face_coordinates": "xh yh", "edge1_coordinates": "xq yh", "edge2_coordinates": "xh yq",
                "long_name": "SGRID topology of the Arakawa C-grid"}),
        }
        if self.n:
            s_w = np.concatenate([[-1.0], -1.0 + np.cumsum(self.dsigma)])
            s_w[-1] = 0.0
            coords["s_rho"] = xr.Variable(("s_rho",), 0.5 * (s_w[1:] + s_w[:-1]), {
                "standard_name": "ocean_sigma_coordinate", "long_name": "σ at layer centres",
                "positive": "up", "formula_terms": "sigma: s_rho eta: eta depth: h", "axis": "Z"})
            coords["s_w"] = xr.Variable(("s_w",), s_w, {"long_name": "σ at layer interfaces", "positive": "up"})
        gattrs = {"Conventions": "CF-1.8 SGRID-0.3", "title": title, "source": "pyOCEAN",
                  "history": f"created {np.datetime64('now', 's')}"}
        gattrs.update(attrs)
        return xr.Dataset(data, coords=coords, attrs=gattrs)

    # ------------------------------------------------------------------ records
    def _record(self, time: float, fields: dict) -> xr.Dataset:
        data = {}
        for name, arr in fields.items():
            if name not in FIELDS:
                raise ValueError(f"unknown output field {name!r}; known: {sorted(FIELDS)}")
            stag, units, std, long = FIELDS[name]
            dims = DIMS[stag]
            arr = np.asarray(arr, np.float64)
            shape = {"yh": self.ny, "xh": self.nx, "yq": self.ny + 1, "xq": self.nx + 1, "s_rho": self.n}
            if arr.shape != tuple(shape[d] for d in dims):
                raise ValueError(f"{name}: shape {arr.shape} does not match {dims}")
            a = {"units": units, "long_name": long, "grid": "grid",
                 "location": {"t": "face", "u": "edge1", "v": "edge2"}[stag[0]]}
            if std:
                a["standard_name"] = std
            data[name] = xr.Variable(("time",) + dims, arr[None], a)
        t = (self.start + np.timedelta64(int(round(time)), "s")).astype("datetime64[ns]")
        return xr.Dataset(data, coords={"time": ("time", [t])})

    def write(self, time: float, **fields: np.ndarray) -> None:
        """Add one record at model time ``time`` [s since ``start``]."""
        rec = self._record(time, fields)
        if self.fmt == "netcdf":
            self._records.append(rec)
            return
        if self._first:
            ds = xr.merge([self.static, rec])
            enc = {"time": {"units": f"seconds since {self.start}", "dtype": "float64"}}
            ds.to_zarr(self.path, mode="w", encoding=enc)
            self._first = False
        else:
            rec.to_zarr(self.path, append_dim="time")

    def close(self) -> None:
        """Write the buffered NetCDF4 file (no-op for Zarr)."""
        if self.fmt == "netcdf" and self._records:
            ds = xr.merge([self.static, xr.concat(self._records, dim="time")])
            enc = {"time": {"units": f"seconds since {self.start}", "dtype": "float64"}}
            ds.to_netcdf(self.path, engine="netcdf4", format="NETCDF4", encoding=enc)
            self._records = []
