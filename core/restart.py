"""
Bit-for-bit restart files for the σ-layer models (``SigmaBaroclinic``).

A restart must hold *everything* the next step reads, not only the prognostic
fields — the classic failures in community models are missing time-stepping
history, hidden model state and precision loss (MOM6 issues #975, #208, #508;
SCHISM #140).  For ``SigmaBaroclinic`` that is:

* η, u, v, T, the salinity S (``salinity=True``, N2a) and the model time;
* the AB3 history of the explicit momentum tendencies (0–3 levels, so a restart
  during the AB start-up also reproduces the continuous run);
* the layer transports Qx, Qy, Fz and ω of the last step (flux-form momentum
  advection uses them), and the last numerical-mixing field;
* the GLS turbulence state k, ε, ν (and κ_t) of the closure component (``core.components.closure``; file
  variables ``gls_*``), or the KPP viscosity ν of the last step (``kpp_akv``, N3 task 7); a restart written with
  another closure is rejected; with ``shear_mixing`` the lagged Jackson shear viscosity after the base's fields
  (``shear_kv``, whatever the base), and a restart written with another shear_mixing is rejected; the drag-driven
  BBL mixing (``bbl_mixing``) has no state of its own, but its scheme and parameters are checked; the bulk
  surface fluxes (``surface_fluxes``, N2b) have none either (the atmosphere is a pure function of time): their
  scheme, averaging depths, wind choice, radiation constants, heights and the atmosphere's time origin are checked;
  the tides of the open boundary and the tidal potential (regional chain Task 5) have no state either: their
  constituents, reference and nodal dates, ramp and edges, and the scalar SAL ``sal_beta``/``sal_flather``, are
  checked (the harmonic constants themselves are not); the atmospheric pressure (``atm_pressure``, Task 6) has
  no state either: the flag, the reference ``atm_pressure_ref`` and the edges of its inverse barometer are checked;
  river runoff (``runoff``, Task 8) has none either: the river set (count and a hash of the cells) and the time
  origin are checked;
* the lagged-LU CG preconditioner of the free surface (the matrix it factorised and the refactor
  flag): CG stops at an iterate that depends on it;
* ALE (``vertical_coordinate='ale'``, Plan D): the layer thicknesses h, the exterior profile kept in z
  (T_ext on h_ext) and the CFL truncation count.  ``open_boundary_state`` is a function and is passed
  to the model again; its edges and modes are checked;
* the equation of state (TEOS-10 in 3-D, spec 2026-10-31-teos10-3d goal H): its name, ρ₀, c_p and the hybrid
  ``p_ref`` (and with a nonlinear EOS ``rho_pgf_ref``) are checked; a restart of another EOS is refused (T and S
  are θ/S_P or Θ/S_A depending on it, and nothing is converted).  With ``rho_ref="auto"`` the reference profile
  is stored: ρ_ref(z) for the linear EOS, T_ref(z) and S_ref(z) for a nonlinear one.

Arrays are written as float64 NetCDF4 (CF attributes, named dimensions), so
reading back is exact.  The model configuration (Δt, layers, closure, schemes,
EOS, grid shape) is stored as attributes and checked on reading: restarting
into a different configuration raises instead of silently diverging.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import xarray as xr

from core.components.closure import STATE_PREFIX as CLOSURE_PREFIX
from core.components.closure import BBLAddedClosure, KPPClosure, ShearAddedClosure, find_closure
from core.model.registry import variable_specs
from core.dynamics.sigma_baroclinic import ReferenceProfile, SigmaBaroclinic, SigmaState, TSReferenceProfile

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
    "S": (("layer", "y", "x"), "1e-3", "salinity"),
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
    eos = getattr(model, "eos_name", "linear")
    cfg = dict(dt=model.dt, n_layers=model.n, theta=model.fs.theta, d_min=model.fs.d_min,
                sigma_w=" ".join(f"{x:.17g}" for x in core.sigma_w[:, 0]), pgf=model.pgf,
                rho_ref="none" if rr is None else ("auto" if isinstance(rr, (str, ReferenceProfile, TSReferenceProfile))
                                                   else "callable"),
                open_boundaries=",".join(model.open) or "none",
                sponge_cells=model.sponge_cells, sponge_tau=" ".join(f"{t:.17g}" for t in model.sponge_tau),
                orlanski_cmax=model.orlanski_cmax, sponge_tracer=int(model.sponge_tracer), kh=model.kh, kv=model.kv,
                closure=core.closure, z0=-1.0 if core.z0 is None else core.z0, nu_const=core.nu_const,
                adv=model.adv, tracer_scheme=model.tracer_scheme, tracer_time=model.tracer_time,
                rho0=model.rho0, t_coef=model.t_coef, t_ref=model.t_ref, ah=model.ah, smagorinsky=model.smagorinsky,
                salinity=int(model.salinity), s_coef=model.s_coef, s_ref=model.s_ref,
                theta_bc=model.theta_bc,
                ny=int(model.H.shape[0]), nx=int(model.H.shape[1]),
                periodic=",".join(model.periodic.axes) or "none",              # N3 task 3: periodic axes
                eos=eos, cp=model.cp, p_ref=getattr(model, "p_ref", 2e7),       # TEOS-10 in 3-D, Task 4
                pgf_mass_weight=getattr(model, "pgf_mass_weight", "vanished"))
    if eos != "linear":
        cfg.update(rho_pgf_ref=model.rho_pgf_ref)
    if core.closure == "gls":                                   # N2a task 8c: stratified or homogeneous k-ε
        cfg.update(gls_stratified=int(getattr(model, "gls_stratified", False)),
                   gls_production=str(getattr(model, "gls_production", "croco")))   # N3 task 5
    if core.closure == "kpp":                                   # N3 task 7: every KPP parameter
        cl = find_closure(model._closure, KPPClosure)          # the KPP base under shear / BBL mixing
        cfg.update({f"kpp_{k}": _text("none" if v is None else v) for k, v in sorted(vars(cl.params).items())})
        cfg.update(kpp_cp=cl.cp)
    sm = getattr(model, "shear_mixing", None)                  # shear-mixing Task 4: scheme, Pr and every parameter
    cfg.update(shear_mixing="none" if sm is None else str(sm))
    if sm is not None:
        sc = find_closure(model._closure, ShearAddedClosure)   # under a BBL wrapper too
        cfg.update(shear_prandtl=sc.prandtl)
        cfg.update({f"jackson_{k}": _text(v) for k, v in sorted(vars(sc.params).items())})
    bm = getattr(model, "bbl_mixing", None)                    # BBL plan Task 4: scheme and every parameter
    cfg.update(bbl_mixing="none" if bm is None else str(bm))
    if bm is not None:
        bc = find_closure(model._closure, BBLAddedClosure)
        cfg.update({f"bbl_{k}": _text(v) for k, v in sorted(vars(bc.bbl_params).items())})
    sf = getattr(model, "surface_fluxes", None)                # bulk fluxes (N2b task 4): no state, the config
    cfg.update(surface_fluxes="none" if sf is None else "bulk")
    if sf is not None:
        org = getattr(sf.atmosphere, "origin", None)
        cfg.update(bulk_scheme=sf.scheme, bulk_h_sfc=sf.h_sfc, bulk_h_uv_sfc="none" if sf.h_uv_sfc is None
                   else _text(sf.h_uv_sfc), bulk_relative_wind=int(sf.relative_wind), bulk_albedo=sf.albedo,
                   bulk_emissivity=sf.emissivity, bulk_z_u=sf.z_u, bulk_z_t=sf.z_t, bulk_z_q=sf.z_q,
                   bulk_origin="none" if org is None else str(np.datetime64(org, "ns")))
    # tides (regional chain Task 5): no state (pure functions of t), so only the configuration is checked
    bt = getattr(model, "_boundary_tides", None)
    cfg.update(boundary_tides="none" if bt is None else "on", sal_beta=float(getattr(model, "sal_beta", 0.0)),
               sal_flather=int(getattr(model, "sal_flather", False)))
    if bt is not None:
        cfg.update({f"tides_{k}": v for k, v in bt.config().items()})
        cfg.update(tides_edges=",".join(bt.edges))
    ap = getattr(model, "_atm_pressure", None)                # atmospheric pressure (Task 6): no state, the config
    cfg.update(ap.config() if ap is not None else dict(atm_pressure="none"))
    ro = getattr(model, "_runoff", None)                      # river runoff (Task 8): no state, the river set
    cfg.update(ro.config() if ro is not None else dict(runoff="none"))
    tp = getattr(model, "_tidal_potential", None)
    cfg.update(tidal_potential="none" if tp is None else "on")
    if tp is not None:
        cfg.update({f"tidal_potential_{k}": v for k, v in tp.config().items()})
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
        sp = getattr(model, "sponge_rate", None)
        cfg.update(sponge_rate="none" if sp is None else _text(sp),
                   sponge_T="none" if sp is None else _text(model._interior_sponge_c.T),
                   sponge_zw="none" if sp is None else _text(model._interior_sponge_c.zw),
                   sponge_S="none" if sp is None or model._interior_sponge_c.S is None
                   else _text(model._interior_sponge_c.S))
        cfg.update(bottom_drag=str(getattr(model, "bottom_drag", None)), cd=model.cd, drag_bg_vel=model.drag_bg_vel,
                   hbbl=model.hbbl)
        cfg.update({f"regrid_{k}": _text(v) for k, v in sorted(model.regrid_params.items())})
    return cfg


# configuration entries added after files were written: a file without them was written with these values
_CONFIG_DEFAULTS = {"salinity": 0, "s_coef": 0.0, "s_ref": 35.0, "sponge_S": "none",         # N2a: files before salinity
                    "gls_stratified": 0,                                                    # N2a 8c: homogeneous GLS
                    "gls_production": "croco",                                              # N3 task 5: CROCO P
                    "periodic": "none",                                                     # N3 task 3: no periodic axis
                    "smagorinsky": 0.0,                                                     # Smagorinsky off
                    "shear_mixing": "none",                                                 # no shear mixing
                    "bbl_mixing": "none",                                                   # no BBL mixing
                    "surface_fluxes": "none",                                               # no bulk fluxes
                    "eos": "linear", "p_ref": 2e7, "pgf_mass_weight": "vanished",           # before TEOS-10 in 3-D
                    "boundary_tides": "none", "sal_beta": 0.0, "sal_flather": 0,            # before tides in 3-D
                    "tidal_potential": "none",
                    "atm_pressure": "none",                                                 # before p_a forcing
                    "runoff": "none"}                                                       # before river runoff

# configuration entries not checked when a file lacks them: files written before TEOS-10 in 3-D did not store c_p,
# which could be any value the run passed
_UNCHECKED_IF_MISSING = ("cp",)

_SECTIONS = {"ab_hist": "time_stepper", "gls_": "closure", "kpp_": "closure", "shear_": "closure",
             "T_ext": "open_boundary", "h_ext": "open_boundary",
             "tr_ext": "open_boundary", "S_ext": "open_boundary", "fs_lu_": "free_surface", "rho_ref_": "pgf",
             "tef_": "diagnostics"}


def _file_prefix(cl, name: str) -> str:
    """File prefix of a closure field: the closure's ``restart_prefix``, except the Jackson shear viscosity, which
    is always ``shear_kv`` whatever the base closure (shear-mixing Task 4, fix round 1)."""
    return "" if name == "shear_kv" else cl.restart_prefix


def _component_of(name: str) -> str:
    """Section of a restart variable: the component that owns it (``core.model.contract.Role`` values), or
    'state' for the variables of ``SigmaState``.  Reading does not depend on it: files written before N1, without
    sections, read the same."""
    for prefix, comp in _SECTIONS.items():
        if name.startswith(prefix):
            return comp
    return "state"


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
    cl = model._closure
    closure = state.closure if state.closure is not None else cl.state()
    if closure is not None and cl.stateful:
        # the closure's own fields (``Closure.state_fields`` names without the prefix: GLS tke, eps, akv[, akt];
        # KPP akv), in σ and ALE (N3 tasks 5 and 7), under the closure's file prefix (``Closure.restart_prefix``:
        # gls_ as the files written before N3, kpp_) that ``read_restart`` reads back
        ny, nx = model.H.shape
        units_of = {"tke": "m2 s-2", "eps": "m2 s-3", "akv": "m2 s-1", "akt": "m2 s-1", "shear_kv": "m2 s-1"}
        for name in (k[len(CLOSURE_PREFIX):] for k in cl.state_fields()):
            units = units_of.get(name, "1")
            pref = _file_prefix(cl, name)
            data[f"{pref}{name}"] = xr.Variable(("level", "y", "x"), closure[name].reshape(-1, ny, nx),
                                                {"units": units,
                                                 "long_name": f"{(pref[:-1] or 'closure').upper()} {name} at layer "
                                                                              "interfaces"})
    if (model.open or getattr(model, "ale", False)) and model.T_ext is not None:
        data["T_ext"] = xr.Variable(("layer", "y", "x"), np.asarray(model.T_ext, np.float64),
                                    {"units": "degC", "long_name": "external temperature (open-boundary inflow, sponge)"})
    if getattr(model, "ale", False) and getattr(model, "h_ext", None) is not None:
        data["h_ext"] = xr.Variable(("layer", "y", "x"), np.asarray(model.h_ext, np.float64),
                                    {"units": "m", "long_name": "layers of the external temperature (kept in z)"})
    if (model.open or getattr(model, "ale", False)) and getattr(model, "S_ext", None) is not None:
        data["S_ext"] = xr.Variable(("layer", "y", "x"), np.asarray(model.S_ext, np.float64),
                                    {"units": "1e-3", "long_name": "external salinity (open-boundary inflow, sponge)"})
    if getattr(state, "tr", None) is not None:
        data["tr"] = xr.Variable(("tracer", "layer", "y", "x"), np.asarray(state.tr, np.float64),
                                 {"units": "1", "long_name": "passive tracers (ALE)"})
    if getattr(model, "tr_ext", None) is not None:
        data["tr_ext"] = xr.Variable(("tracer", "layer", "y", "x"), np.asarray(model.tr_ext, np.float64),
                                     {"units": "1", "long_name": "external passive tracers (kept in z with h_ext)"})
    if getattr(model, "tef", None) is not None:                      # N2b task 5: TEF accumulators
        for k, v in model.tef.state().items():
            data[f"tef_{k}"] = xr.Variable(("tef_bin",) if np.ndim(v) else (), v, {"long_name": f"TEF transect {k}"})
    if getattr(model, "tef_region_budget", None) is not None:
        for k, v in model.tef_region_budget.state().items():
            dims = ("tef_three",) if np.ndim(v) else ()
            data[f"tef_region_{k}"] = xr.Variable(dims, v, {"long_name": f"TEF region {k}"})
    pre = model.fs.preconditioner_state() if hasattr(model.fs, "preconditioner_state") else None
    if pre is not None:
        data["fs_lu_data"] = xr.Variable(("lu_nnz",), pre["data"], {"long_name": "lagged CG preconditioner matrix (CSC data)"})
        data["fs_lu_indices"] = xr.Variable(("lu_nnz",), pre["indices"].astype(np.int64))
        data["fs_lu_indptr"] = xr.Variable(("lu_ptr",), pre["indptr"].astype(np.int64))
    if isinstance(model.rho_ref, TSReferenceProfile):                # nonlinear EOS: the T and S profiles
        data["rho_ref_z"] = xr.Variable(("ref",), model.rho_ref.z, {"units": "m", "long_name": "reference profile height"})
        data["rho_ref_T"] = xr.Variable(("ref",), model.rho_ref.T, {"units": "degC",
                                                                   "long_name": "reference temperature profile"})
        data["rho_ref_S"] = xr.Variable(("ref",), model.rho_ref.S, {"units": "1e-3",
                                                                   "long_name": "reference salinity profile"})
    if isinstance(model.rho_ref, ReferenceProfile):
        data["rho_ref_z"] = xr.Variable(("ref",), model.rho_ref.z, {"units": "m", "long_name": "reference profile height"})
        data["rho_ref_rho"] = xr.Variable(("ref",), model.rho_ref.rho,
                                          {"units": "kg m-3", "long_name": "reference density profile ρ_ref(z)"})
    attrs = {"model": type(model).__name__, "time": float(state.time), "time_units": "s", "n_trunc": int(state.n_trunc),
             "n_limited": int(state.n_limited),
             **({} if state.n_visc_limited is None else {"n_visc_limited": int(state.n_visc_limited)}),
             "Conventions": "CF-1.8", "title": "pyOCEAN bit-for-bit restart"}
    attrs.update({f"config_{k}": v for k, v in _config(model).items()})
    if pre is not None:
        attrs.update(fs_lu_rows=int(pre["shape"][0]), fs_lu_cols=int(pre["shape"][1]), fs_lu_refactor=int(pre["refactor"]))
    if model.eos_name != "linear":                                   # TEOS-10 in 3-D task 8: T, S in the EOS convention
        for name, sp in ((k, v) for k, v in variable_specs(model.eos_name).items() if k in ("T", "S")):
            if name in data:
                data[name].attrs.update(units=sp.units, standard_name=sp.standard_name, long_name=sp.long_name)
    for name, var in data.items():                                   # N1: one section per component
        var.attrs["component"] = _component_of(name)
    xr.Dataset(data, attrs=attrs).to_netcdf(path, engine="netcdf4", format="NETCDF4")


def _n_visc_limited(ds: xr.Dataset, model: SigmaBaroclinic) -> int | None:
    """
    Smagorinsky-limited point count of the file; a file without it (written before the option, or with it off)
    gives 0 when the model runs Smagorinsky and None (never counted, as a fresh state) otherwise.
    """
    n = ds.attrs.get("n_visc_limited")
    if n is not None:
        return int(n)
    return 0 if getattr(model, "smagorinsky", 0.0) > 0 else None


def read_restart(path: str | Path, model: SigmaBaroclinic, allow_changed: tuple[str, ...] = ()) -> SigmaState:
    """
    Read a restart into a ``SigmaState`` and restore the hidden state of ``model``.

    ``allow_changed``: names of configuration entries (``_config`` keys, e.g. ``("theta_bc",)``) that may differ
    between the restart and the model — a deliberate change at the restart, for diagnostics (regional chain Task 9,
    θ_bc test of the NWA12 cut).  The model's value is used and a ``UserWarning`` names each change.  Default empty:
    every entry must match (strict, as before).  Unknown names raise ValueError.
    """
    allow_changed = tuple(allow_changed)
    unknown = [k for k in allow_changed if k not in _config(model)]
    if unknown:
        raise ValueError(f"allow_changed {unknown}: not restart configuration entries (have {sorted(_config(model))})")
    with xr.open_dataset(path, engine="netcdf4") as ds:
        ds = ds.load()
    if ds.attrs.get("model") != type(model).__name__:
        raise ValueError(f"restart written by {ds.attrs.get('model')}, not {type(model).__name__}")
    if not getattr(model, "ale", False) and "config_regrid" in ds.attrs:
        raise ValueError("restart written by an ALE model (vertical_coordinate='ale'), not a σ model")
    written = ds.attrs.get("config_closure")
    if written is not None and str(written) != model.core.closure:  # Review Focus 3 (N3 task 7)
        raise ValueError(f"restart written with closure={written!r} but the model uses closure={model.core.closure!r}: "
                         "the turbulence state of one closure cannot start another; restart with the same closure")
    written_eos = str(ds.attrs.get("config_eos", _CONFIG_DEFAULTS["eos"]))
    if written_eos != getattr(model, "eos_name", "linear"):          # Review Focus 5 (TEOS-10 in 3-D)
        raise ValueError(f"restart written with eos={written_eos!r} but the model uses eos={model.eos_name!r}: T and "
                         "S follow the convention of each equation of state (θ/S_P or Θ/S_A) and a restart is not "
                         "converted; restart with the same eos")
    bad = []
    for k, v in _config(model).items():
        if k in _UNCHECKED_IF_MISSING and f"config_{k}" not in ds.attrs:
            continue
        stored = ds.attrs.get(f"config_{k}", _CONFIG_DEFAULTS.get(k))
        if isinstance(v, str) or isinstance(stored, str):
            same = str(stored) == str(v)
        else:
            same = stored is not None and np.isclose(float(stored), float(v), rtol=0, atol=0)
        if not same:
            if k in allow_changed:
                import warnings
                warnings.warn(f"read_restart: {k} changed at the restart ({stored!r} → {v!r}), allowed by allow_changed",
                              UserWarning, stacklevel=2)
                continue
            bad.append(f"{k}: restart {stored!r} vs model {v!r}")
    if bad:
        raise ValueError("restart incompatible with the model configuration — " + "; ".join(bad))
    if model.salinity and "S" not in ds:
        raise ValueError("restart has no salinity S but the model has salinity=True")
    kw = {name: ds[name].values.copy() for name in _FIELDS}
    for name in _OPTIONAL:
        kw[name] = ds[name].values.copy() if name in ds else None
    hist = []
    if "ab_hist_u" in ds:
        hist = [(gu.copy(), gv.copy()) for gu, gv in zip(ds["ab_hist_u"].values, ds["ab_hist_v"].values)]
    cl = model._closure                                              # closure component (N3 task 1)
    if cl.stateful:
        pref = cl.restart_prefix
        names = {k: _file_prefix(cl, k[len(CLOSURE_PREFIX):]) + k[len(CLOSURE_PREFIX):] for k in cl.state_fields()}
        missing = [v for v in names.values() if v not in ds]
        if missing:
            raise ValueError(f"restart has no {(pref[:-1] or 'closure').upper()} state ({', '.join(missing)}) but "
                             f"the model uses "
                             f"closure={model.core.closure!r}")
        # file names keep the gls_ prefix of the files written before N3; the component's fields are closure_*
        cl.load_state({k: ds[v].values for k, v in names.items()})
        closure = cl.state()
    else:
        closure = None
    if model.open:
        if "T_ext" not in ds:
            raise ValueError("restart has no external tracer but the model has open boundaries")
        model.T_ext = ds["T_ext"].values.copy()
        if model.salinity:
            if "S_ext" not in ds:
                raise ValueError("restart has no external salinity S_ext but the model has salinity=True and open "
                                 "boundaries")
            model.S_ext = ds["S_ext"].values.copy()
    if getattr(model, "ale", False):
        if "S_ext" in ds:
            model.S_ext = ds["S_ext"].values.copy()
        if "T_ext" in ds:
            model.T_ext = ds["T_ext"].values.copy()
        if "h_ext" in ds:
            model.h_ext = ds["h_ext"].values.copy()
        if "tr_ext" in ds:
            model.tr_ext = ds["tr_ext"].values.copy()
    if getattr(model, "tef", None) is not None:
        keys = ("vol", "salt", "salt2", "time")
        miss = [f"tef_{k}" for k in keys if f"tef_{k}" not in ds]
        if miss:
            raise ValueError(f"restart has no TEF state ({', '.join(miss)}) but the model has tef_face")
        model.tef.load({k: ds[f"tef_{k}"].values for k in keys})
    if getattr(model, "tef_region_budget", None) is not None:
        reg = [k for k in ds.variables if str(k).startswith("tef_region_")]
        if "tef_region_time" not in ds:
            raise ValueError("restart has no TEF region state (tef_region_*) but the model has tef_region")
        model.tef_region_budget.load({k[len("tef_region_"):]: ds[k].values for k in reg})
    if hasattr(model.fs, "set_preconditioner_state"):
        model.fs.set_preconditioner_state(
            dict(data=ds["fs_lu_data"].values.copy(), indices=ds["fs_lu_indices"].values.astype(np.int32),
                 indptr=ds["fs_lu_indptr"].values.astype(np.int32),
                 shape=(int(ds.attrs["fs_lu_rows"]), int(ds.attrs["fs_lu_cols"])),
                 refactor=bool(ds.attrs["fs_lu_refactor"])) if "fs_lu_data" in ds else None)
    if isinstance(model.rho_ref, (str, ReferenceProfile, TSReferenceProfile)):
        if not model._eos.linear:
            if "rho_ref_T" not in ds:
                raise ValueError("restart has no reference T/S profile but the model uses rho_ref='auto' with "
                                 f"eos={model.eos_name!r}")
            model.rho_ref = TSReferenceProfile(ds["rho_ref_z"].values.copy(), ds["rho_ref_T"].values.copy(),
                                               ds["rho_ref_S"].values.copy(), model._eos, model.g)
        elif "rho_ref_z" not in ds:
            raise ValueError("restart has no reference density profile but the model uses rho_ref='auto'")
        else:
            model.rho_ref = ReferenceProfile(ds["rho_ref_z"].values.copy(), ds["rho_ref_rho"].values.copy())
    return SigmaState(g_hist=hist, time=float(ds.attrs["time"]), n_trunc=int(ds.attrs.get("n_trunc", 0)),
                      n_limited=int(ds.attrs.get("n_limited", 0)),
                      n_visc_limited=_n_visc_limited(ds, model),
                      tr=ds["tr"].values.copy() if "tr" in ds else None, closure=closure, **kw)
