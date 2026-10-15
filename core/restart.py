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
* the GLS turbulence state k, ε, ν held by the model (``core.gls``);
* the lagged-LU CG preconditioner of the free surface (the matrix it factorised and the refactor
  flag): CG stops at an iterate that depends on it;
* ALE (``vertical_coordinate='ale'``, Plan D): the layer thicknesses h, the exterior profile kept in z
  (T_ext on h_ext) and the CFL truncation count.  ``open_boundary_state`` is a function and is passed
  to the model again; its edges and modes are checked.

Arrays are written as float64 NetCDF4 (CF attributes, named dimensions), so
reading back is exact.  The model configuration (Δt, layers, closure, schemes,
EOS, grid shape) is stored as attributes and checked on reading: restarting
into a different configuration raises instead of silently diverging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from core.dynamics.sigma_baroclinic import ReferenceProfile, SigmaBaroclinic, SigmaState

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
    "h": (("layer", "y", "x"), "m", "layer thickness (ALE), bed first"),
}


def _text(v) -> str:
    """A configuration value as exact text (arrays: every element at 17 significant digits)."""
    a = np.asarray(v)
    if a.dtype.kind in "fiub":
        return " ".join(f"{x:.17g}" for x in np.atleast_1d(a).astype(float).ravel())
    return str(v)


def _config(model: SigmaBaroclinic) -> dict:
    """Configuration that must match between the writer and the reader."""
    core = model.core
    rr = model.rho_ref
    cfg = dict(dt=model.dt, n_layers=model.n, theta=model.fs.theta, d_min=model.fs.d_min,
                sigma_w=" ".join(f"{x:.17g}" for x in core.sigma_w[:, 0]), pgf=model.pgf,
                rho_ref="none" if rr is None else ("auto" if isinstance(rr, (str, ReferenceProfile)) else "callable"),
                open_boundaries=",".join(model.open) or "none",
                sponge_cells=model.sponge_cells, sponge_tau=" ".join(f"{t:.17g}" for t in model.sponge_tau),
                orlanski_cmax=model.orlanski_cmax, sponge_tracer=int(model.sponge_tracer), kh=model.kh, kv=model.kv,
                closure=core.closure, z0=-1.0 if core.z0 is None else core.z0, nu_const=core.nu_const,
                adv=model.adv, tracer_scheme=model.tracer_scheme, tracer_time=model.tracer_time,
                rho0=model.rho0, t_coef=model.t_coef, t_ref=model.t_ref, ah=model.ah,
                theta_bc=model.theta_bc,
                ny=int(model.H.shape[0]), nx=int(model.H.shape[1]))
    if getattr(model, "ale", False):                           # ALE (Plan D): the rule and every scheme choice
        cfg.update(vertical_coordinate="ale", regrid=model.regrid_rule, remap=int(model.remap_code),
                   thickness_flux=model.thickness_flux, momentum=model.momentum, ke_scheme=model.ke_scheme,
                   h_thin_bed=model.h_thin_bed, max_truncations=model.max_truncations,
                   thin_layer_coupling=model.thin_layer_coupling,
                   thin_layer_cutoff=model.thin_layer_cutoff,
                   tracer_limiter=model.tracer_limiter,
                   mix_diagnostic=int(model.mix_diagnostic), obc_tau_in=model.obc_tau_in, obc_tau_out=model.obc_tau_out,
                   obc_mode=",".join(f"{e}:{m}" for e, m in sorted(model.obc_mode.items())) or "none",
                   obc_state="none" if model.obc_state is None else "function",
                   n_tracers=int(getattr(model, "n_tracers", 0)))
        cfg.update({f"regrid_{k}": _text(v) for k, v in sorted(model.regrid_params.items())})
    return cfg


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
    if (model.open or getattr(model, "ale", False)) and model.T_ext is not None:
        data["T_ext"] = xr.Variable(("layer", "y", "x"), np.asarray(model.T_ext, np.float64),
                                    {"units": "degC", "long_name": "external temperature (open-boundary inflow, sponge)"})
    if getattr(model, "ale", False) and getattr(model, "h_ext", None) is not None:
        data["h_ext"] = xr.Variable(("layer", "y", "x"), np.asarray(model.h_ext, np.float64),
                                    {"units": "m", "long_name": "layers of the external temperature (kept in z)"})
    if getattr(state, "tr", None) is not None:
        data["tr"] = xr.Variable(("tracer", "layer", "y", "x"), np.asarray(state.tr, np.float64),
                                 {"units": "1", "long_name": "passive tracers (ALE)"})
    if getattr(model, "tr_ext", None) is not None:
        data["tr_ext"] = xr.Variable(("tracer", "layer", "y", "x"), np.asarray(model.tr_ext, np.float64),
                                     {"units": "1", "long_name": "external passive tracers (kept in z with h_ext)"})
    pre = model.fs.preconditioner_state() if hasattr(model.fs, "preconditioner_state") else None
    if pre is not None:
        data["fs_lu_data"] = xr.Variable(("lu_nnz",), pre["data"], {"long_name": "lagged CG preconditioner matrix (CSC data)"})
        data["fs_lu_indices"] = xr.Variable(("lu_nnz",), pre["indices"].astype(np.int64))
        data["fs_lu_indptr"] = xr.Variable(("lu_ptr",), pre["indptr"].astype(np.int64))
    if isinstance(model.rho_ref, ReferenceProfile):
        data["rho_ref_z"] = xr.Variable(("ref",), model.rho_ref.z, {"units": "m", "long_name": "reference profile height"})
        data["rho_ref_rho"] = xr.Variable(("ref",), model.rho_ref.rho,
                                          {"units": "kg m-3", "long_name": "reference density profile ρ_ref(z)"})
    attrs = {"model": type(model).__name__, "time": float(state.time), "time_units": "s", "n_trunc": int(state.n_trunc),
             "n_limited": int(state.n_limited),
             "Conventions": "CF-1.8", "title": "pyOCEAN bit-for-bit restart"}
    attrs.update({f"config_{k}": v for k, v in _config(model).items()})
    if pre is not None:
        attrs.update(fs_lu_rows=int(pre["shape"][0]), fs_lu_cols=int(pre["shape"][1]), fs_lu_refactor=int(pre["refactor"]))
    xr.Dataset(data, attrs=attrs).to_netcdf(path, engine="netcdf4", format="NETCDF4")


def read_restart(path: str | Path, model: SigmaBaroclinic) -> SigmaState:
    """Read a restart into a ``SigmaState`` and restore the hidden state of ``model``."""
    with xr.open_dataset(path, engine="netcdf4") as ds:
        ds = ds.load()
    if ds.attrs.get("model") != type(model).__name__:
        raise ValueError(f"restart written by {ds.attrs.get('model')}, not {type(model).__name__}")
    if not getattr(model, "ale", False) and "config_regrid" in ds.attrs:
        raise ValueError("restart written by an ALE model (vertical_coordinate='ale'), not a σ model")
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
    if model.open:
        if "T_ext" not in ds:
            raise ValueError("restart has no external tracer but the model has open boundaries")
        model.T_ext = ds["T_ext"].values.copy()
    if getattr(model, "ale", False):
        if "T_ext" in ds:
            model.T_ext = ds["T_ext"].values.copy()
        if "h_ext" in ds:
            model.h_ext = ds["h_ext"].values.copy()
        if "tr_ext" in ds:
            model.tr_ext = ds["tr_ext"].values.copy()
    if hasattr(model.fs, "set_preconditioner_state"):
        model.fs.set_preconditioner_state(
            dict(data=ds["fs_lu_data"].values.copy(), indices=ds["fs_lu_indices"].values.astype(np.int32),
                 indptr=ds["fs_lu_indptr"].values.astype(np.int32),
                 shape=(int(ds.attrs["fs_lu_rows"]), int(ds.attrs["fs_lu_cols"])),
                 refactor=bool(ds.attrs["fs_lu_refactor"])) if "fs_lu_data" in ds else None)
    if isinstance(model.rho_ref, (str, ReferenceProfile)):
        if "rho_ref_z" not in ds:
            raise ValueError("restart has no reference density profile but the model uses rho_ref='auto'")
        model.rho_ref = ReferenceProfile(ds["rho_ref_z"].values.copy(), ds["rho_ref_rho"].values.copy())
    return SigmaState(g_hist=hist, time=float(ds.attrs["time"]), n_trunc=int(ds.attrs.get("n_trunc", 0)),
                      n_limited=int(ds.attrs.get("n_limited", 0)),
                      tr=ds["tr"].values.copy() if "tr" in ds else None, **kw)
