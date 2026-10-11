"""
CF NetCDF output of the baroclinic BayIW runs (run_bayiw.py), two files per run:

``<stem>_stations.nc`` — CF-1.8 discrete sampling geometry ``timeSeriesProfile``
    (orthogonal multidimensional, CF §H.5): ζ, T, u, v at the layer centres and w at
    the layer interfaces, dims (station, time, s_rho|s_w), float64 (exact).  The
    vertical coordinate is CF ``ocean_sigma_coordinate``, z = ζ + σ (h + ζ), with
    σ ∈ [−1, 0] (the model's σ − 1).
``<stem>_fields.nc`` — the Y–Z section along the bay axis (dims time, s_rho|s_w, y)
    with its own σ formula terms, and w, T interpolated to fixed depths in a box
    around the bay (time, depth, yb, xb); float32, zlib, one chunk per record.
    Sections are not a DSG feature type, hence the second file.

Time is "seconds since 2000-01-01" (t = 0 is the model start; idealised run, no
calendar date).  Files are written to a temporary name and renamed, so a crash
while writing never destroys the previous output.

``read_records`` returns the runner's in-memory layout (t, eta_st (time, station),
T_st (time, k, station), …, T_z (time, depth, yb, xb)) from these files or from a
legacy .npz of the first BayIW runs.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
import xarray as xr

START = "2000-01-01T00:00:00"
STATION_VARS = {   # record key → (name, vertical dim, units, standard_name, long_name)
    "eta_st": ("zeta", None, "m", "sea_surface_height_above_geoid", "sea surface height"),
    "T_st": ("temp", "s_rho", "degC", "sea_water_potential_temperature", "temperature"),
    "u_st": ("u", "s_rho", "m s-1", "sea_water_x_velocity", "x velocity at the cell centre"),
    "v_st": ("v", "s_rho", "m s-1", "sea_water_y_velocity", "y velocity at the cell centre"),
    "w_st": ("w", "s_w", "m s-1", "upward_sea_water_velocity", "vertical velocity at layer interfaces"),
}
FIELD_VARS = {     # record key → (name, dims, units, standard_name, long_name)
    "eta_axis": ("zeta_section", ("time", "y"), "m", "sea_surface_height_above_geoid", "sea surface height"),
    "T_axis": ("temp_section", ("time", "s_rho", "y"), "degC", "sea_water_potential_temperature", "temperature"),
    "v_axis": ("v_section", ("time", "s_rho", "y"), "m s-1", "sea_water_y_velocity", "y velocity at the cell centre"),
    "w_axis": ("w_section", ("time", "s_w", "y"), "m s-1", "upward_sea_water_velocity",
               "vertical velocity at layer interfaces"),
    "T_z": ("temp_z", ("time", "depth", "yb", "xb"), "degC", "sea_water_potential_temperature",
            "temperature at fixed depth"),
    "w_z": ("w_z", ("time", "depth", "yb", "xb"), "m s-1", "upward_sea_water_velocity",
            "vertical velocity at fixed depth"),
}


def _sigma_coords(sigma_w: np.ndarray, eta: str, depth: str) -> dict:
    """CF σ coordinates from the model interfaces σ_w (bed = 0, surface = 1)."""
    s_w = np.asarray(sigma_w, float) - 1.0
    return {"s_rho": xr.Variable(("s_rho",), 0.5 * (s_w[1:] + s_w[:-1]), {
                "standard_name": "ocean_sigma_coordinate", "long_name": "σ at layer centres", "positive": "up",
                "formula_terms": f"sigma: s_rho eta: {eta} depth: {depth}", "axis": "Z"}),
            "s_w": xr.Variable(("s_w",), s_w, {
                "standard_name": "ocean_sigma_coordinate", "long_name": "σ at layer interfaces", "positive": "up",
                "formula_terms": f"sigma: s_w eta: {eta} depth: {depth}"})}


def _time(t: np.ndarray) -> xr.Variable:
    return xr.Variable(("time",), np.asarray(t, float), {
        "standard_name": "time", "units": f"seconds since {START}", "axis": "T",
        "comment": "t = 0 is the model start (idealised run, no calendar date)"})


def _attrs(static: dict, what: str, feature: str | None) -> dict:
    a = {"Conventions": "CF-1.8", "title": f"pyOCEAN BayIW internal waves: {what}", "source": "pyOCEAN",
         "history": f"created {np.datetime64('now', 's')}", "run_config": json.dumps(static["attrs"])}
    if feature:
        a["featureType"] = feature
    return a


def stations_dataset(rec: dict, static: dict) -> xr.Dataset:
    """Stations as a CF timeSeriesProfile dataset (station, time, s_rho|s_w)."""
    names = list(static["stations"])
    ij = np.array([static["stations"][n] for n in names])
    coords = {"time": _time(rec["t"]), **_sigma_coords(static["sigma_w"], "zeta", "h")}
    data = {
        "station_name": xr.Variable(("station",), np.array(names, dtype=object),
                                    {"cf_role": "timeseries_id", "long_name": "station name"}),
        "x": xr.Variable(("station",), np.asarray(static["x_st"], float),
                         {"standard_name": "projection_x_coordinate", "units": "m", "axis": "X"}),
        "y": xr.Variable(("station",), np.asarray(static["y_st"], float),
                         {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}),
        "i": xr.Variable(("station",), ij[:, 0].astype(np.int32), {"long_name": "model column index (0-based)"}),
        "j": xr.Variable(("station",), ij[:, 1].astype(np.int32), {"long_name": "model row index (0-based)"}),
        "h": xr.Variable(("station",), np.asarray(static["h_st"], float),
                         {"standard_name": "sea_floor_depth_below_geoid", "units": "m", "positive": "down"}),
    }
    for key, (name, zdim, units, std, long) in STATION_VARS.items():
        a = np.asarray(rec[key], float)
        if zdim is None:
            dims, a = ("station", "time"), a.T                                 # (time, st) → (st, time)
        else:
            dims, a = ("station", "time", zdim), a.transpose(2, 0, 1)          # (time, k, st) → (st, time, k)
        data[name] = xr.Variable(dims, a, {"units": units, "standard_name": std, "long_name": long,
                                           "coordinates": "x y"})
    return xr.Dataset(data, coords=coords, attrs=_attrs(static, "stations", "timeSeriesProfile"))


def fields_dataset(rec: dict, static: dict) -> xr.Dataset:
    """Bay-axis Y–Z section and fixed-depth slices in the box around the bay."""
    (j0, j1), (i0, i1) = static["box"]
    coords = {
        "time": _time(rec["t"]), **_sigma_coords(static["sigma_w"], "zeta_section", "h_section"),
        "y": xr.Variable(("y",), np.asarray(static["y_axis"], float),
                         {"standard_name": "projection_y_coordinate", "units": "m", "axis": "Y"}),
        "depth": xr.Variable(("depth",), np.asarray(static["depths"], float),
                             {"standard_name": "depth", "units": "m", "positive": "down", "axis": "Z"}),
        "yb": xr.Variable(("yb",), np.asarray(static["y_box"], float),
                          {"standard_name": "projection_y_coordinate", "units": "m",
                           "long_name": f"y of the box rows j = {j0}…{j1 - 1}"}),
        "xb": xr.Variable(("xb",), np.asarray(static["x_box"], float),
                          {"standard_name": "projection_x_coordinate", "units": "m",
                           "long_name": f"x of the box columns i = {i0}…{i1 - 1}"}),
    }
    data = {"h_section": xr.Variable(("y",), np.asarray(static["h_axis"], float),
                                     {"standard_name": "sea_floor_depth_below_geoid", "units": "m",
                                      "positive": "down"}),
            "x_section": xr.Variable((), float(static["x_axis"]),
                                     {"standard_name": "projection_x_coordinate", "units": "m",
                                      "long_name": f"x of the section (column i = {static['i_axis']})"})}
    for key, (name, dims, units, std, long) in FIELD_VARS.items():
        data[name] = xr.Variable(dims, np.asarray(rec[key], np.float32),
                                 {"units": units, "standard_name": std, "long_name": long})
    return xr.Dataset(data, coords=coords, attrs=_attrs(static, "bay-axis section and fixed-depth slices", None))


def _write(ds: xr.Dataset, path: Path, enc: dict) -> None:
    tmp = path.with_name(path.name + ".tmp")
    ds.to_netcdf(tmp, engine="netcdf4", format="NETCDF4", encoding=enc)
    os.replace(tmp, path)


def write(stem: str | Path, rec: dict, static: dict) -> None:
    """Write ``<stem>_stations.nc`` (float64) and ``<stem>_fields.nc`` (float32, zlib) atomically."""
    stem = Path(stem)
    st = stations_dataset(rec, static)
    enc = {v: {"zlib": True, "complevel": 4} for v in st.data_vars if st[v].dtype.kind == "f"}
    _write(st, stem.with_name(stem.name + "_stations.nc"), enc)
    f = fields_dataset(rec, static)
    enc = {}
    for v in f.data_vars:
        if "time" in f[v].dims:
            enc[v] = {"zlib": True, "complevel": 4, "chunksizes": (1,) + f[v].shape[1:]}
    _write(f, stem.with_name(stem.name + "_fields.nc"), enc)


def read_records(path: str | Path, t_max: float | None = None) -> dict:
    """
    Records of a run in the runner's layout, from ``<stem>_{stations,fields}.nc`` (``path`` = stem)
    or a legacy .npz; ``t_max`` keeps the records with t ≤ t_max (continuation from a restart).
    Static entries: stations {name: (i, j)}, sigma_w (model σ, bed = 0), h_st (or None for .npz).
    """
    path = Path(path)
    out: dict = {}
    if path.suffix == ".npz":
        d = np.load(path)
        for k in ("t", *STATION_VARS, *FIELD_VARS):
            out[k] = d[k]
        out["stations"] = {k: tuple(v) for k, v in json.loads(str(d["stations"])).items()}
        out["sigma_w"] = d["sigma_w"]
        out["h_st"] = None
        out["H_axis"] = d["H_axis"]
        out["meta"] = json.loads(str(d["meta_json"]))
    else:
        with xr.open_dataset(path.with_name(path.name + "_stations.nc"), decode_times=False) as st:
            out["t"] = st["time"].values
            for key, (name, zdim, *_) in STATION_VARS.items():
                a = st[name].values
                out[key] = a.T if zdim is None else a.transpose(1, 2, 0)
            out["stations"] = {str(n): (int(i), int(j)) for n, i, j in
                               zip(st["station_name"].values, st["i"].values, st["j"].values)}
            out["sigma_w"] = st["s_w"].values + 1.0
            out["h_st"] = st["h"].values
            out["meta"] = json.loads(st.attrs["run_config"])
        with xr.open_dataset(path.with_name(path.name + "_fields.nc"), decode_times=False) as f:
            for key, (name, *_) in FIELD_VARS.items():
                out[key] = f[name].values
            out["H_axis"] = f["h_section"].values
    if t_max is not None:
        keep = out["t"] <= t_max + 1e-6
        for k in ("t", *STATION_VARS, *FIELD_VARS):
            out[k] = out[k][keep]
    return out
