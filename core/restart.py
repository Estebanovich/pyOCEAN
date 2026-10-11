"""
Bit-for-bit restart files for the σ-layer models (``SigmaBaroclinic``).

A restart must hold *everything* the next step reads, not only the prognostic
fields — the classic failures in community models are missing time-stepping
history, hidden model state and precision loss (MOM6 issues #975, #208, #508;
SCHISM #140).  For ``SigmaBaroclinic`` that is:

* η, u, v, T and the model time;
* the AB3 history of the explicit momentum tendencies (0–3 levels, so a restart
  during the AB start-up also reproduces the continuous run);
* the layer transports Qx, Qy, Fz and ω of the last step (flux-form momentum
  advection uses them), and the last numerical-mixing field;
* the GLS turbulence state k, ε, ν held by the model (``core.gls``).

Arrays are written as float64 NetCDF4 (CF attributes, named dimensions), so
reading back is exact.  The model configuration (Δt, layers, closure, schemes,
EOS, grid shape) is stored as attributes and checked on reading: restarting
into a different configuration raises instead of silently diverging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, SigmaState

_FIELDS = {          # name: (dims, units, long name)
    "eta": (("y", "x"), "m", "sea surface height"),
    "u": (("layer", "y", "xu"), "m s-1", "layer velocity at u-faces"),
    "v": (("layer", "yv", "x"), "m s-1", "layer velocity at v-faces"),
    "T": (("layer", "y", "x"), "degC", "temperature"),
}
_OPTIONAL = {
    "omega": (("level", "y", "x"), "m s-1", "diasigma velocity at layer interfaces"),
    "Qx": (("layer", "y", "xu"), "m3 s-1", "layer volume transport through u-faces"),
    "Qy": (("layer", "yv", "x"), "m3 s-1", "layer volume transport through v-faces"),
    "Fz": (("level", "y", "x"), "m3 s-1", "volume flux through layer interfaces"),
    "mix_num": (("layer", "y", "x"), "degC2 m3 s-1", "online numerical mixing (Burchard & Rennau 2008)"),
}


def _config(model: SigmaBaroclinic) -> dict:
    """Configuration that must match between the writer and the reader."""
    core = model.core
    return dict(dt=model.dt, n_layers=model.n, theta=model.fs.theta, d_min=model.fs.d_min,
                closure=core.closure, z0=-1.0 if core.z0 is None else core.z0, nu_const=core.nu_const,
                adv=model.adv, tracer_scheme=model.tracer_scheme, tracer_time=model.tracer_time,
                rho0=model.rho0, t_coef=model.t_coef, t_ref=model.t_ref, ah=model.ah,
                ny=int(model.H.shape[0]), nx=int(model.H.shape[1]))


def write_restart(path: str | Path, model: SigmaBaroclinic, state: SigmaState) -> None:
    """Write ``state`` and the model's hidden state to a float64 NetCDF4 restart file."""
    data = {}
    for name, (dims, units, long) in _FIELDS.items():
        data[name] = xr.Variable(dims, np.asarray(getattr(state, name), np.float64),
                                 {"units": units, "long_name": long})
    for name, (dims, units, long) in _OPTIONAL.items():
        val = getattr(state, name)
        if val is not None:
            data[name] = xr.Variable(dims, np.asarray(val, np.float64), {"units": units, "long_name": long})
    if state.g_hist:
        data["ab_hist_u"] = xr.Variable(("hist", "layer", "y", "xu"), np.stack([g[0] for g in state.g_hist]),
                                        {"units": "m s-2", "long_name": "AB history of explicit u tendencies, newest first"})
        data["ab_hist_v"] = xr.Variable(("hist", "layer", "yv", "x"), np.stack([g[1] for g in state.g_hist]),
                                        {"units": "m s-2", "long_name": "AB history of explicit v tendencies, newest first"})
    gls = model.core.gls
    if gls is not None:
        ny, nx = model.H.shape
        for name, units in (("tke", "m2 s-2"), ("eps", "m2 s-3"), ("akv", "m2 s-1")):
            data[f"gls_{name}"] = xr.Variable(("level", "y", "x"), getattr(gls, name).reshape(-1, ny, nx),
                                              {"units": units, "long_name": f"GLS {name} at layer interfaces"})
    attrs = {"model": type(model).__name__, "time": float(state.time), "time_units": "s",
             "Conventions": "CF-1.8", "title": "pyOCEAN bit-for-bit restart"}
    attrs.update({f"config_{k}": v for k, v in _config(model).items()})
    xr.Dataset(data, attrs=attrs).to_netcdf(path, engine="netcdf4", format="NETCDF4")


def read_restart(path: str | Path, model: SigmaBaroclinic) -> SigmaState:
    """Read a restart into a ``SigmaState`` and restore the hidden state of ``model``."""
    with xr.open_dataset(path, engine="netcdf4") as ds:
        ds = ds.load()
    if ds.attrs.get("model") != type(model).__name__:
        raise ValueError(f"restart written by {ds.attrs.get('model')}, not {type(model).__name__}")
    bad = []
    for k, v in _config(model).items():
        stored = ds.attrs.get(f"config_{k}")
        if isinstance(v, str) or isinstance(stored, str):
            same = str(stored) == str(v)
        else:
            same = stored is not None and np.isclose(float(stored), float(v), rtol=0, atol=0)
        if not same:
            bad.append(f"{k}: restart {stored!r} vs model {v!r}")
    if bad:
        raise ValueError("restart incompatible with the model configuration — " + "; ".join(bad))
    kw = {name: ds[name].values.copy() for name in _FIELDS}
    for name in _OPTIONAL:
        kw[name] = ds[name].values.copy() if name in ds else None
    hist = []
    if "ab_hist_u" in ds:
        hist = [(gu.copy(), gv.copy()) for gu, gv in zip(ds["ab_hist_u"].values, ds["ab_hist_v"].values)]
    if model.core.gls is not None:
        if "gls_tke" not in ds:
            raise ValueError("restart has no GLS state but the model uses closure='gls'")
        gls = model.core.gls
        gls.tke, gls.eps, gls.akv = (ds[f"gls_{n}"].values.reshape(gls.tke.shape).copy() for n in ("tke", "eps", "akv"))
    return SigmaState(g_hist=hist, time=float(ds.attrs["time"]), **kw)
