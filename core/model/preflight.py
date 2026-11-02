"""
Pre-flight check of a 3-D model configuration (N1, spec 2026-10-28-n1-component-core-design.md §5).

Every argument-level problem is found before anything is built and reported together, in one
``ConfigurationError`` (MITgcm CONFIG_CHECK and CROCO check_kwds collect their errors the same way,
docs/design/n1_refs/mitgcm.md §2, croco.md §2); arguments that would be silently ignored give a
``PyOceanConfigWarning``.  Messages keep the wording of the checks they replace (``SigmaBaroclinic.__init__``),
followed where useful by the valid options (Oceananigans style).  Checks that need the grid (array shapes, cells
that can dry next to open edges, the hybrid floor) stay where the grid is built.
"""
from __future__ import annotations

import numpy as np

from core.model.contract import Issue
from physics.mixing.bbl_mixing import BBLParams
from physics.mixing.kappa_shear import JacksonParams
from physics.mixing.kpp_cvmix import KPPParams
from physics.surface.shortwave import JERLOV

EDGES = ("west", "east", "south", "north")
PGF = ("twoterm", "jacobian", "fv")
TRACER_TIME = ("ssprk3", "pc")
TRACER_SCHEMES = ("weno5", "weno5js", "upwind3", "upwind1")
ADV = ("flux", "upwind3", "centred")
KE_SCHEMES = ("arakawa", "gudonov", "godunov")
REMAP = ("pcm", "plm", "ppm_h4")
BBL_SCHEMES = ("lotw",)
EOS_NAMES = ("linear", "wright_full", "teos10")
PGF_MASS_WEIGHTS = ("vanished", "full", "none")
LINEAR_EOS_ONLY = ("t_coef", "t_ref", "s_coef", "s_ref")   # coefficients of ρ₀ − α (T − T₀) + β (S − S₀)
SIGMA2_RANGE = (1020.0, 1045.0)    # [kg m⁻³] ocean potential densities at p_ref = 2e7 Pa (σ₂ + 1000 ≈ 1030–1038)

# arguments that only the ALE coordinate uses (ignored, with a warning, in σ)
ALE_ONLY = ("regrid", "regrid_params", "remap_scheme", "h_thin_bed", "init_iterations", "max_truncations",
            "thin_layer_cutoff", "mix_diagnostic", "thickness_flux")

# arguments that act on salinity and need salinity=True (N2a)
SALINITY_ONLY = ("fw", "variance_budget", "tef_face", "tef_region", "sponge_S")


def _opts(values) -> str:
    return ", ".join(repr(v) for v in values)


def jackson_params_errors(jp: JacksonParams) -> list[str]:
    """
    One message per invalid field of ``JacksonParams`` (Jackson et al. 2008; MOM_kappa_shear.F90), so that all
    errors are reported together.  Ranges follow the use of each field in ``physics/mixing/kappa_shear.py``:
    ri_crit > 0 (Ri_c S² − N² threshold); shearmix_rate ≥ 0 (γ, K_src = 2γ|S|…); fri_curvature > −1 (the source
    denominator Ri_c S² + F_c N² must stay positive where N² < Ri_c S²); kappa_buoy_scale > 0 (λ, 1/λ²);
    tke_n_decay, tke_shear_decay, kd_trunc_frac, tke_background, lz_rescale ≥ 0; kd_seed, tol_err > 0;
    max_rino_it, max_ks_it integers ≥ 1; max_kap_src_chg > 0.5 (tol_dksrc_low = (x − 0.5)/x must be > 0,
    MOM_kappa_shear.F90:1014-1019).  Every value must be finite.
    """
    rules = (("ri_crit", lambda v: v > 0, "> 0 (Jackson et al. 2008: 0.25)"),
             ("shearmix_rate", lambda v: v >= 0, "≥ 0 (γ, the rate of shear mixing)"),
             ("fri_curvature", lambda v: v > -1, "> −1 (keeps Ri_c S² + F_c N² positive)"),
             ("kappa_buoy_scale", lambda v: v > 0, "> 0 (λ; the buoyancy length uses 1/λ²)"),
             ("tke_n_decay", lambda v: v >= 0, "≥ 0"), ("tke_shear_decay", lambda v: v >= 0, "≥ 0"),
             ("kd_seed", lambda v: v > 0, "> 0 (first guess of κ)"),
             ("kd_trunc_frac", lambda v: v >= 0, "≥ 0 (κ_trunc / κ0)"),
             ("tol_err", lambda v: v > 0, "> 0 (relative tolerance of the κ iteration)"),
             ("max_kap_src_chg", lambda v: v > 0.5, "> 0.5 (otherwise tol_dksrc_low ≤ 0)"),
             ("tke_background", lambda v: v >= 0, "≥ 0 (q0)"),
             ("lz_rescale", lambda v: v >= 0, "≥ 0 (0 behaves as 1)"))
    out = []
    for name, ok, rule in rules:
        v = getattr(jp, name)
        try:
            good = bool(np.isfinite(v) and ok(v))
        except TypeError:
            good = False
        if not good:
            out.append(f"jackson_params.{name}={v!r}: must be a finite number {rule}")
    for name in ("max_rino_it", "max_ks_it"):
        v = getattr(jp, name)
        if isinstance(v, bool) or not isinstance(v, (int, np.integer)) or v < 1:
            out.append(f"jackson_params.{name}={v!r}: must be an integer ≥ 1 (iteration count)")
    return out


def bbl_params_errors(bp: BBLParams) -> list[str]:
    """
    One message per invalid field of ``BBLParams`` (drag-driven BBL mixing, ``physics.mixing.bbl_mixing``), so that
    all errors are reported together.  bbl_effic ∈ [0, 1] (a fraction of the drag work); max_decay > 0 (the decay
    scale cap, 1/max_decay); von_karman > 0; kd_max > 0 (the LOTW Kd where the mixing work is zero, SD:1769-1772;
    MOM6's default −1 would give a negative Kd).  Every value must be finite; use_omega a bool.
    """
    rules = (("bbl_effic", lambda v: 0.0 <= v <= 1.0, "in [0, 1] (fraction of the drag work; MOM6 BBL_EFFIC 0.2)"),
             ("max_decay", lambda v: v > 0, "> 0 [m] (BBL_MIXING_MAX_DECAY 200)"),
             ("von_karman", lambda v: v > 0, "> 0 (VON_KARMAN_BBL 0.41)"),
             ("kd_max", lambda v: v > 0, "> 0 [m² s⁻¹] (KD_MAX, OM4 0.1; used where the mixing work is zero)"))
    out = []
    for name, ok, rule in rules:
        v = getattr(bp, name)
        try:
            good = bool(np.isfinite(v) and ok(v))
        except TypeError:
            good = False
        if not good:
            out.append(f"bbl_params.{name}={v!r}: must be a finite number {rule}")
    if not isinstance(bp.use_omega, (bool, np.bool_)):
        out.append(f"bbl_params.use_omega={bp.use_omega!r}: must be True or False")
    return out


def preflight(a: dict, given: frozenset[str]) -> list[Issue]:
    """
    Issues of the constructor arguments ``a`` of ``SigmaBaroclinic`` (``given``: the names passed explicitly).
    Errors keep the messages of the former inline checks, so callers matching them keep working.
    """
    out: list[Issue] = []

    def err(comp: str, msg: str) -> None:
        out.append(Issue("error", comp, msg))

    def warn(comp: str, msg: str) -> None:
        out.append(Issue("warning", comp, msg))

    vc = a["vertical_coordinate"]
    ale = vc == "ale"
    if vc not in ("sigma_eulerian", "ale"):
        err("vertical_coordinate", f"unknown vertical_coordinate {vc!r}; use 'sigma_eulerian' or 'ale'")
    if a["pgf"] not in PGF:
        err("pgf", f"unknown pressure-gradient scheme {a['pgf']!r} (valid: {_opts(PGF)})")
    if a["tracer_time"] not in TRACER_TIME:
        err("tracer_advection", f"unknown tracer time stepping {a['tracer_time']!r} (valid: {_opts(TRACER_TIME)})")
    if a["tracer_scheme"] not in TRACER_SCHEMES:
        err("tracer_advection", f"unknown tracer scheme {a['tracer_scheme']!r} (valid: {_opts(TRACER_SCHEMES)})")
    if a["adv"] not in ADV:
        err("momentum", f"unknown momentum advection adv={a['adv']!r} (valid: {_opts(ADV)})")
    smag = a.get("smagorinsky", 0.0)
    if not (np.isfinite(smag) and smag >= 0.0):
        err("momentum", f"smagorinsky={smag!r}: the Smagorinsky coefficient must be a finite number ≥ 0 "
                        "(0 = off; OM4 uses 0.15)")
    _curvilinear_issues(a, ale, smag, warn)
    # ---- open boundaries
    open_ = tuple(a["open_boundaries"])
    bad = set(open_) - set(EDGES)
    if bad:
        err("open_boundary", f"unknown open boundaries {sorted(bad)}; use {EDGES}")
    if a["sponge_cells"] and not open_:
        err("open_boundary", "a sponge needs open boundaries")
    if a["open_boundary_state"] is not None and not open_:
        err("open_boundary", "open_boundary_state needs open_boundaries")
    modes = {e: "radiating" for e in open_}
    for e, mode in (a["open_boundary_mode"] or {}).items():
        if e not in open_ or mode not in ("radiating", "clamped", "eta"):
            err("open_boundary", f"open_boundary_mode {{{e!r}: {mode!r}}}: edges must be open and modes "
                                 "'radiating', 'clamped' or 'eta'")
        else:
            modes[e] = mode
    if a["open_boundary_state"] is None and "clamped" in modes.values():
        err("open_boundary", "a 'clamped' open_boundary_mode needs open_boundary_state (the exterior to impose)")
    _tide_issues(a, given, modes, err, warn)
    # ---- periodic axes (N3 task 3): an axis is periodic or has open/closed edges, not both
    per = a.get("periodic", ()) or ()
    per = (per,) if isinstance(per, str) else tuple(per)
    bad_axes = [ax for ax in per if ax not in ("x", "y")]
    if bad_axes:
        err("periodic", f"unknown periodic axes {bad_axes}; use 'x', 'y' or both")
    for ax, edges in (("x", ("west", "east")), ("y", ("south", "north"))):
        both = [e for e in edges if e in open_]
        if ax in per and both:
            err("periodic", f"periodic={per!r} and open boundary {both} on the same axis {ax!r}: a periodic axis has "
                            "no edges; drop one of the two")
    # ---- thin layers, tracers, sponge, drag
    tlc = a["thin_layer_coupling"]
    if tlc is not None and tlc not in ("none", "mom6"):
        err("free_surface", f"unknown thin_layer_coupling {tlc!r}; use 'none' or 'mom6'")
    cut = a["thin_layer_cutoff"]
    if cut is not None and not cut >= 0.0:
        err("free_surface", f"thin_layer_cutoff must be ≥ 0 m, got {cut!r}")
    lim = a["tracer_limiter"]
    if lim is not None and lim not in ("none", "fct"):
        err("tracer_advection", f"unknown tracer_limiter {lim!r}; use 'none' or 'fct'")
    if lim == "fct" and not (ale and a["tracer_time"] == "ssprk3" and a["tracer_scheme"] in ("weno5", "weno5js")):
        err("tracer_advection", "tracer_limiter='fct' needs vertical_coordinate='ale', tracer_time='ssprk3' and a "
                                "WENO5 tracer_scheme")
    cp = a.get("cp")
    if cp is not None and not (cp > 0):
        err("heat", f"cp={cp!r}: the specific heat must be > 0 [J kg⁻¹ K⁻¹]")
    if a.get("jerlov", "I") not in JERLOV:
        err("heat", f"unknown jerlov type {a['jerlov']!r} (valid: {_opts(JERLOV)})")
    ntr = a["n_tracers"]
    if int(ntr) < 0 or (int(ntr) and not ale):
        err("tracer_advection", f"n_tracers={ntr!r}: passive tracers need vertical_coordinate='ale' and n ≥ 0")
    if a["sponge_rate"] is not None and not ale:
        err("interior_sponge", "sponge_rate (interior sponge) needs vertical_coordinate='ale'")
    bd = a["bottom_drag"]
    if bd not in (None, "quadratic"):
        err("bottom_drag", f"unknown bottom_drag {bd!r}; use None or 'quadratic'")
    elif bd == "quadratic" and not ale:
        err("bottom_drag", "bottom_drag='quadratic' needs vertical_coordinate='ale' (σ uses z0, the log-law)")
    # ---- momentum form and KE scheme
    momentum = a["momentum"] or ("vector_invariant" if ale else "flux")
    if momentum not in ("flux", "vector_invariant"):
        err("momentum", f"unknown momentum {momentum!r}; use 'vector_invariant' or 'flux'")
    elif momentum == "vector_invariant" and not ale:
        err("momentum", "momentum='vector_invariant' needs vertical_coordinate='ale' (the σ model keeps the "
                        "flux form, which matches CROCO)")
    ke = a["ke_scheme"]
    if ke not in KE_SCHEMES:
        err("momentum", f"unknown ke_scheme {ke!r}; use 'arakawa' or 'godunov'")
    elif ke == "gudonov" and "ke_scheme" in given:
        warn("momentum", "ke_scheme='gudonov' is a misspelling kept for compatibility; use 'godunov'")
    if "ke_scheme" in given and momentum == "flux" and ke in KE_SCHEMES:
        warn("momentum", f"ke_scheme={ke!r} is ignored with momentum='flux' (it applies to the vector-invariant "
                         "form only)")
    if a["thickness_flux"] not in ("upwind", "ppm"):
        err("layer_transport", f"unknown thickness_flux {a['thickness_flux']!r}; use 'upwind' or 'ppm'")
    # ---- ALE coordinate
    if ale:
        rule = a["regrid"]
        if rule not in ("sigma", "zstar", "hybrid"):
            err("vertical_coordinate", f"unknown regrid rule {rule!r}; use 'sigma', 'zstar' or 'hybrid'")
        if a["remap_scheme"] not in REMAP:
            err("vertical_coordinate", f"unknown remap_scheme {a['remap_scheme']!r}; use one of {sorted(REMAP)}")
        if a["z0"] is not None and bd == "quadratic":
            err("bottom_drag", "z0 (law-of-the-wall drag) and bottom_drag='quadratic' are two bed drags; give one "
                               "(z0=None for the quadratic drag)")
        elif a["z0"] is not None and not (a["z0"] > 0 and a.get("hbbl", 10.0) > 0):
            err("bottom_drag", f"z0={a['z0']!r}, hbbl={a.get('hbbl', 10.0)!r}: the ALE law-of-the-wall drag needs "
                               "z0 > 0 and hbbl > 0 [m]")
        if a["closure"] not in ("constant", "none", "gls", "kpp"):
            err("closure", "vertical_coordinate='ale' needs closure 'constant', 'none', 'gls' or 'kpp' (the "
                           "parabolic ν assumes σ-layers)")
        params = a["regrid_params"] or {}
        for key in {"sigma": (), "zstar": ("dz_top",), "hybrid": ("dz_top", "rho_target")}.get(rule, ()):
            if key not in params:
                err("vertical_coordinate", f"regrid={rule!r} needs regrid_params[{key!r}]")
    elif vc == "sigma_eulerian":
        for name in ALE_ONLY:
            if name in given:
                warn("vertical_coordinate", f"{name} is ignored with vertical_coordinate='sigma_eulerian' (it applies "
                                            "to the ALE coordinate only)")
        if tlc == "mom6":
            warn("free_surface", "thin_layer_coupling='mom6' has no effect with vertical_coordinate='sigma_eulerian'")
    # ---- salinity (N2a, spec 2026-10-28-n2a-salinity-freshwater §7).  ``fw`` is a step() argument and
    # ``variance_budget`` (task 6) and ``tef_face`` (task 7) are constructor arguments
    sal = bool(a.get("salinity", False))
    for name in SALINITY_ONLY:
        v = a.get(name)
        if v is not None and v is not False and not sal:
            err("salinity", f"{name} needs salinity=True (it acts on the salinity S)")
    # TEF and Lorenz's relation (N2a task 7, diagnostics.tef)
    tf = a.get("tef_face")
    bathy = a.get("bathy")
    if tf is not None:
        shp = () if bathy is None else np.shape(bathy)
        nx = shp[1] if len(shp) == 2 else None
        if int(tf) != tf or tf < 0 or (nx is not None and tf > nx):
            err("tef", f"tef_face={tf!r} must be a u-face column 0 … nx" + ("" if nx is None else f" (= {nx})"))
        if a.get("kh", 0.0) > 0:
            warn("tef", "tef_face with kh > 0: the along-layer diffusive salt flux through the transect is not an "
                        "exchange-flow transport and is not in the TEF bins, so Lorenz's relation (32) misses it")
    if a.get("tef_sign", 1) not in (1, -1):
        err("tef", f"tef_sign must be +1 or −1, got {a.get('tef_sign')!r}")
    if a.get("tef_region") is not None and not a.get("variance_budget", False):
        err("tef", "tef_region integrates the variance budget over the region: it needs variance_budget=True")
    eos = a.get("eos", "linear")
    nonlinear = eos in EOS_NAMES and eos != "linear"
    _eos_issues(a, given, eos, nonlinear, sal, ale, err, warn)
    if not sal and a.get("s_coef", 0.0) != 0.0 and not nonlinear:
        warn("salinity", f"s_coef={a['s_coef']!r} is ignored with salinity=False (there is no S in the density); "
                         "pass salinity=True")
    if sal and a.get("s_coef", 0.0) == 0.0 and eos == "linear":
        warn("salinity", "salinity=True with s_coef = 0: S is carried but does not enter the density "
                         "(ρ = ρ₀ − α (T − T₀) + β (S − S₀) with β = s_coef); pass s_coef (e.g. 0.78) for an active "
                         "salinity")
    if sal and a.get("rho_ref") is not None and a.get("kh", 0.0) > 0 and eos == "linear":
        warn("tracer_diffusion", "salinity=True with rho_ref and kh > 0: K_h diffuses T and S themselves, not the "
                                 "anomaly T − T_ref(z) of the reference profile (a density profile does not split "
                                 "into T_ref and S_ref), so along sloping σ-layers it also mixes the mean "
                                 "stratification")
    # stratified GLS (N2a task 8c)
    if "gls_stratified" in given and a["closure"] != "gls":
        warn("closure", f"gls_stratified={a['gls_stratified']!r} is ignored with closure={a['closure']!r} (it applies "
                        "to closure='gls' only)")
    if a["closure"] == "gls" and a.get("gls_stratified", True) and a.get("kv", 0.0) > 0:
        warn("closure", f"closure='gls' with gls_stratified=True: kv={a['kv']!r} is the floor of the GLS tracer "
                        "diffusivity, κ_t = max(c_μ′ k²/ε, kv), not a constant (kv = 0 keeps CROCO's floor 1e-6)")
    gp = a.get("gls_production", "croco")                              # N3 task 5
    if gp not in ("croco", "energy"):
        err("closure", f"unknown gls_production {gp!r}; use 'croco' (P = ν S², CROCO) or 'energy' (Burchard 2002)")
    elif "gls_production" in given and a["closure"] != "gls":
        warn("closure", f"gls_production={gp!r} is ignored with closure={a['closure']!r} (it applies to "
                        "closure='gls' only)")
    # KPP (N3 task 7)
    kp = a.get("kpp_params")
    if kp is not None and not isinstance(kp, KPPParams):
        err("closure", f"kpp_params must be a physics.mixing.kpp_cvmix.KPPParams, got {type(kp).__name__}")
    elif "kpp_params" in given and kp is not None and a["closure"] != "kpp":
        warn("closure", f"kpp_params is ignored with closure={a['closure']!r} (it applies to closure='kpp' only)")
    if a["closure"] == "kpp":
        for name, bg in (("kv", "kappa_bg"), ("nu_const", "nu_bg")):
            if name in given:
                warn("closure", f"{name}={a[name]!r} is ignored with closure='kpp' (ν and κ come from KPP; a "
                                f"background is KPPParams.{bg})")
    # Jackson et al. (2008) shear mixing added to the closure (shear-mixing plan, Task 4)
    sm = a.get("shear_mixing")
    if sm is not None and sm not in ("jackson",):
        err("closure", f"unknown shear_mixing={sm!r} (valid: None, 'jackson')")
    if sm is not None and a["closure"] in ("gls", "parabolic"):
        err("closure", f"shear_mixing={sm!r} cannot be added to closure={a['closure']!r}: the GLS k-ε already "
                       "produces shear turbulence (double counting) and the parabolic ν is a fixed bed profile; use "
                       "closure 'constant', 'none' or 'kpp'")
    pr = a.get("shear_prandtl", 1.0)
    try:
        pr_ok = bool(np.isfinite(pr) and pr >= 0.0)
    except TypeError:
        pr_ok = False
    if not pr_ok:
        err("closure", f"shear_prandtl={pr!r}: the turbulent Prandtl number ν_shear/κ_shear must be a finite number "
                       "≥ 0 (Jackson et al. 2008: 1; 0 = diffusivity only)")
    jp = a.get("jackson_params")
    if jp is not None and not isinstance(jp, JacksonParams):
        err("closure", f"jackson_params must be a physics.mixing.kappa_shear.JacksonParams, got {type(jp).__name__}")
    elif jp is not None:
        for msg in jackson_params_errors(jp):
            err("closure", msg)
    if sm is None:
        for name in ("shear_prandtl", "jackson_params"):
            if name in given:
                warn("closure", f"{name} is ignored without shear_mixing (it applies to shear_mixing='jackson')")
    # drag-driven BBL mixing added to the closure (BBL plan, Task 4; spec 2026-10-30-bbl-drag-mixing)
    bm = a.get("bbl_mixing")
    if bm is not None and bm not in BBL_SCHEMES:
        err("closure", f"unknown bbl_mixing={bm!r} (valid: None, 'lotw')")
    if bm is not None:
        if a["closure"] in ("gls", "parabolic"):
            err("closure", f"bbl_mixing={bm!r} cannot be added to closure={a['closure']!r}: the GLS k-ε already has "
                           "a bed boundary condition for k (double counting) and the parabolic ν is a fixed bed "
                           "profile; use closure 'constant', 'none' or 'kpp'")
        if not ale:
            err("closure", f"bbl_mixing={bm!r} needs vertical_coordinate='ale' (σ, z* or hybrid ALE): in the σ "
                           "model the bed drag lives inside the column solver, without the rates its energy needs")
        if bd != "quadratic" and a["z0"] is None:
            err("closure", f"bbl_mixing={bm!r} needs a bed drag, bottom_drag='quadratic' or z0 (law of the wall): "
                           "its energy is the work of the drag")
    bp = a.get("bbl_params")
    if bp is not None and not isinstance(bp, BBLParams):
        err("closure", f"bbl_params must be a physics.mixing.bbl_mixing.BBLParams, got {type(bp).__name__}")
    elif bp is not None:
        for msg in bbl_params_errors(bp):
            err("closure", msg)
    if bm is None and "bbl_params" in given:
        warn("closure", "bbl_params is ignored without bbl_mixing (it applies to bbl_mixing='lotw')")
    # bulk surface fluxes (N2b task 4, spec 2026-10-31-n2b-bulk-surface-fluxes criterion 6)
    sf = a.get("surface_fluxes")
    if sf is not None:
        from core.components.surface_fluxes import bulk_errors
        for msg in bulk_errors(sf, sal):
            err("surface_fluxes", msg)
    _atm_pressure_issues(a, given, sf, err, warn)
    ro = a.get("runoff")                                       # river runoff (regional chain Task 8)
    if ro is not None:
        from core.components.runoff import runoff_errors
        for msg in runoff_errors(ro, a.get("bathy")):
            err("runoff", msg)
        if not sal:
            err("salinity", "runoff needs salinity=True (river water enters with S = 0, the real freshwater path)")
    if a["sponge_rate"] is None and ({"sponge_T", "sponge_zw", "sponge_tr", "sponge_S"} & given):
        warn("interior_sponge", "sponge_T / sponge_zw / sponge_tr / sponge_S are ignored without sponge_rate")
    return out


def _curvilinear_issues(a: dict, ale: bool, smag: float, warn) -> None:
    """
    Operators without curvilinear metric terms (regional chain Task 3, ``docs/design/regional_refs/metric_audit.md``
    §2.3, class b): on a grid built from a MOM6 supergrid (``grid_type == 'mom6_supergrid'``) the σ flux-form and
    advective momentum and the constant-A_h Laplacian (``smagorinsky`` = 0 with ``ah`` > 0) treat the grid components
    of the velocity as Cartesian ones and miss the curvature terms.  The invariant forms are metric-correct: use
    ``vertical_coordinate='ale'`` with the vector-invariant momentum (the default there) and ``smagorinsky`` > 0 (the
    stress-tensor operator, ``ah`` its floor) — the NWA12 cut configuration.  Warnings only: a run is consistent to
    the size of the missing terms.
    """
    hg = a.get("h_grid")
    attrs = getattr(hg, "attrs", None) or {}
    if attrs.get("grid_type") != "mom6_supergrid":
        return
    momentum = a["momentum"] or ("vector_invariant" if ale else "flux")
    if momentum == "flux":
        warn("momentum", "curvilinear (supergrid) grid with the flux-form momentum: it has no curvilinear metric "
                         "terms (metric_audit.md §2.3, class b); use vertical_coordinate='ale' with the "
                         "vector-invariant momentum")
    if smag == 0.0 and a.get("ah", 0.0) > 0.0:
        warn("momentum", "curvilinear (supergrid) grid with the constant-A_h Laplacian (smagorinsky = 0, ah > 0): it "
                         "is not the stress-tensor divergence and misses the metric terms (metric_audit.md §2.3, class "
                         "b); use smagorinsky > 0 (ah becomes its floor)")


def _atm_pressure_issues(a: dict, given: frozenset[str], sf, err, warn) -> None:
    """
    Atmospheric pressure (regional chain Task 6, ``core.components.atm_pressure``): ``atm_pressure`` is a bool and
    needs ``surface_fluxes`` (p_a is the ``msl`` of the bulk component's atmosphere); ``atm_pressure_ref`` is None
    or a finite pressure in Pa (a value in hPa is caught), and is ignored without ``atm_pressure``.
    """
    ap = a.get("atm_pressure", False)
    if not isinstance(ap, (bool, np.bool_)):
        err("atm_pressure", f"atm_pressure must be True or False, got {ap!r}")
    elif ap and sf is None:
        err("atm_pressure", "atm_pressure=True needs surface_fluxes: the sea-level pressure p_a is the 'msl' of the "
                            "bulk fluxes' atmosphere (BulkFluxes(atmosphere, ...)) at tⁿ + Δt/2")
    ref = a.get("atm_pressure_ref")
    if ref is not None:
        try:
            ok = bool(np.isfinite(ref) and 5e4 <= float(ref) <= 1.2e5)
        except (TypeError, ValueError):
            ok = False
        if not ok:
            err("atm_pressure", f"atm_pressure_ref={ref!r}: the reference sea-level pressure p̄_a must be a number in "
                                "Pa between 5e4 and 1.2e5 (e.g. 1e5, as SIS2), or None for the domain mean")
        if not ap:
            warn("atm_pressure", "atm_pressure_ref is ignored without atm_pressure=True")


def _tide_issues(a: dict, given: frozenset[str], modes: dict[str, str], err, warn) -> None:
    """
    Tides at the open boundary, body tide and scalar SAL (regional chain Task 5,
    ``core.components.boundary_tides``): ``boundary_tides`` needs a 'radiating' edge (the tide is added to the
    Flather exterior of those edges only), valid constituents/dates/ramp; ``tidal_potential=True`` takes its
    constituents from ``boundary_tides``; ``sal_beta`` ∈ [0, 1).
    """
    from core.components.boundary_tides import BoundaryTides, TidalPotential
    bt = a.get("boundary_tides")
    if bt is not None:
        if not isinstance(bt, BoundaryTides):
            err("open_boundary", f"boundary_tides must be a core.components.boundary_tides.BoundaryTides, got "
                                 f"{type(bt).__name__}")
        else:
            for msg in bt.issues():
                err("open_boundary", msg)
        if "radiating" not in modes.values():
            err("open_boundary", "boundary_tides needs at least one 'radiating' open edge: the tide is added to the "
                                 "Flather exterior (η, ū) of the 'radiating' edges "
                                 f"(open edges and modes: {modes or 'none'})")
    tp = a.get("tidal_potential", False)
    if tp is True:
        if bt is None:
            err("tidal_potential", "tidal_potential=True takes its constituents and dates from boundary_tides; "
                                   "without boundary_tides pass a TidalPotential(constituents, ref_date, ...)")
        elif isinstance(bt, BoundaryTides):
            for msg in TidalPotential.from_boundary_tides(bt).issues():
                err("tidal_potential", msg)
    elif isinstance(tp, TidalPotential):
        for msg in tp.issues():
            err("tidal_potential", msg)
    elif tp is not False and tp is not None:
        err("tidal_potential", f"tidal_potential must be False, True or a TidalPotential, got {type(tp).__name__}")
    beta = a.get("sal_beta", 0.0)
    try:
        ok = bool(np.isfinite(beta) and 0.0 <= beta < 1.0)
    except TypeError:
        ok = False
    if not ok:
        err("free_surface", f"sal_beta={beta!r}: the scalar self-attraction and loading factor must be in [0, 1) "
                            "(MOM6 SAL_SCALAR_VALUE; NWA12 0.05)")
    elif beta == 0.0 and "sal_flather" in given and a.get("sal_flather"):
        warn("free_surface", "sal_flather=True is ignored with sal_beta=0 (the Flather speed is √(g D) either way)")


def _eos_issues(a: dict, given: frozenset[str], eos, nonlinear: bool, sal: bool, ale: bool, err, warn) -> None:
    """
    Equation-of-state rules (spec 2026-10-31-teos10-3d goal H): unknown EOS; a nonlinear EOS without salinity, with
    the linear coefficients, with a ``rho_ref`` function (a density profile does not give T_ref, S_ref), with
    hybrid ``rho_target`` outside the σ₂ range, or with ρ₀ outside 1000–1050 (warning); σ-layers with a nonlinear
    EOS and no ``rho_ref`` (warning: the compressibility enters the σ anomaly); the 'fv' bed mass weighting.
    """
    if eos not in EOS_NAMES:
        err("eos", f"unknown eos {eos!r} (valid: {_opts(EOS_NAMES)})")
    mw = a.get("pgf_mass_weight", "vanished")
    if mw not in PGF_MASS_WEIGHTS:
        err("pgf", f"unknown pgf_mass_weight {mw!r} (valid: {_opts(PGF_MASS_WEIGHTS)})")
    elif mw == "full" and eos == "linear":
        err("pgf", "pgf_mass_weight='full' needs a nonlinear eos ('wright_full' or 'teos10'); the linear finite-volume "
                   "pressure gradient takes 'vanished' or 'none'")
    elif "pgf_mass_weight" in given and a["pgf"] != "fv":
        warn("pgf", f"pgf_mass_weight={mw!r} is ignored with pgf={a['pgf']!r} (it applies to pgf='fv' only)")
    tk = a.get("ts_kind", "theta_sp")
    if tk not in ("theta_sp", "ct_sa"):
        err("eos", f"unknown ts_kind {tk!r} (valid: 'theta_sp', 'ct_sa')")
    elif tk != ("ct_sa" if eos == "teos10" else "theta_sp") and eos in EOS_NAMES:
        if a.get("sponge_T") is not None and a.get("sponge_S") is None:
            err("eos", f"sponge_T in ts_kind={tk!r} needs sponge_S to be converted to the convention of eos={eos!r}")
        if not sal:
            err("eos", f"ts_kind={tk!r} differs from the convention of eos={eos!r}: the conversion needs "
                       "salinity=True")
    p_ref = a.get("p_ref", 2e7)
    if not (np.isfinite(p_ref) and p_ref >= 0):
        err("eos", f"p_ref={p_ref!r}: the reference pressure of the hybrid potential density must be a finite "
                   "number ≥ 0 [Pa] (2e7 = σ₂)")
    rpr = a.get("rho_pgf_ref")
    if rpr is not None and not (np.isfinite(rpr) and rpr > 0):
        err("pgf", f"rho_pgf_ref={rpr!r}: must be a finite density > 0 [kg m⁻³]")
    elif rpr is not None and eos == "linear":
        warn("pgf", "rho_pgf_ref is ignored with eos='linear' (it applies to a nonlinear EOS)")
    if not nonlinear:
        return
    if not sal:
        err("eos", f"eos={eos!r} needs salinity=True: its density depends on S")
    for name in LINEAR_EOS_ONLY:
        if name in given:
            err("eos", f"{name}={a[name]!r} does not apply with eos={eos!r} (it is a coefficient of the linear EOS)")
    rr = a.get("rho_ref")
    if callable(rr):
        err("pgf", f"rho_ref as a function cannot be used with eos={eos!r}: a density profile does not give the "
                   "T_ref(z), S_ref(z) the nonlinear EOS needs; use rho_ref='auto' (or None)")
    rho0 = a.get("rho0")
    if rho0 is not None and not 1000.0 <= rho0 <= 1050.0:
        warn("eos", f"rho0={rho0!r} with eos={eos!r} is outside 1000–1050 kg m⁻³: the Boussinesq density sets the "
                    "EOS pressure −g ρ₀ z and 1/ρ₀ of the pressure gradient (MOM6 RHO_0 1035)")
    tgt = (a.get("regrid_params") or {}).get("rho_target")
    if ale and a.get("regrid") == "hybrid" and tgt is not None:
        t = np.asarray(tgt, float)
        lo, hi = SIGMA2_RANGE
        if t.size and not (np.isfinite(t).all() and (t >= lo).all() and (t <= hi).all()):
            err("vertical_coordinate", f"regrid_params['rho_target'] ({t.min()!r} … {t.max()!r}) is outside the "
                                       f"plausible range {lo:g}–{hi:g} kg m⁻³ of the potential density at p_ref "
                                       f"(σ₂ + 1000 at 2e7 Pa) the hybrid regrid uses with eos={eos!r}")
    sigma = a["vertical_coordinate"] == "sigma_eulerian" or (ale and a.get("regrid") == "sigma")
    if sigma and rr is None:
        warn("pgf", f"σ-layers with eos={eos!r} and rho_ref=None: the compressibility (~4–5 kg m⁻³ per km) enters "
                    "the density anomaly along sloping layers whole; pass rho_ref='auto' (the T/S reference profile)")
