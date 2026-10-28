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

from core.model.contract import Issue

EDGES = ("west", "east", "south", "north")
PGF = ("twoterm", "jacobian", "fv")
TRACER_TIME = ("ssprk3", "pc")
TRACER_SCHEMES = ("weno5", "weno5js", "upwind3", "upwind1")
ADV = ("flux", "upwind3", "centred")
KE_SCHEMES = ("arakawa", "gudonov", "godunov")
REMAP = ("pcm", "plm", "ppm_h4")

# arguments that only the ALE coordinate uses (ignored, with a warning, in σ)
ALE_ONLY = ("regrid", "regrid_params", "remap_scheme", "h_thin_bed", "init_iterations", "max_truncations",
            "thin_layer_cutoff", "mix_diagnostic", "thickness_flux")

# arguments that act on salinity and need salinity=True (N2a)
SALINITY_ONLY = ("fw", "variance_budget", "tef_face", "sponge_S")


def _opts(values) -> str:
    return ", ".join(repr(v) for v in values)


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
        if e not in open_ or mode not in ("radiating", "clamped"):
            err("open_boundary", f"open_boundary_mode {{{e!r}: {mode!r}}}: edges must be open and modes "
                                 "'radiating' or 'clamped'")
        else:
            modes[e] = mode
    if a["open_boundary_state"] is None and "clamped" in modes.values():
        err("open_boundary", "a 'clamped' open_boundary_mode needs open_boundary_state (the exterior to impose)")
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
        if a["z0"] is not None:
            err("bottom_drag", "vertical_coordinate='ale' needs z0=None for now: the law-of-the-wall drag uses "
                               "the bed layer, which can vanish under z*")
        if a["closure"] not in ("constant", "none"):
            err("closure", "vertical_coordinate='ale' needs closure 'constant' or 'none' for now "
                           "(GLS and the parabolic ν assume σ-layers)")
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
    # ``variance_budget`` / ``tef_face`` arrive with later N2a tasks: absent names are simply not checked
    sal = bool(a.get("salinity", False))
    for name in SALINITY_ONLY:
        v = a.get(name)
        if v is not None and v is not False and not sal:
            err("salinity", f"{name} needs salinity=True (it acts on the salinity S)")
    if not sal and a.get("s_coef", 0.0) != 0.0:
        warn("salinity", f"s_coef={a['s_coef']!r} is ignored with salinity=False (there is no S in the density); "
                         "pass salinity=True")
    if sal and a.get("rho_ref") is not None and a.get("kh", 0.0) > 0:
        warn("tracer_diffusion", "salinity=True with rho_ref and kh > 0: K_h diffuses T and S themselves, not the "
                                 "anomaly T − T_ref(z) of the reference profile (a density profile does not split "
                                 "into T_ref and S_ref), so along sloping σ-layers it also mixes the mean "
                                 "stratification")
    if a["sponge_rate"] is None and ({"sponge_T", "sponge_zw", "sponge_tr", "sponge_S"} & given):
        warn("interior_sponge", "sponge_T / sponge_zw / sponge_tr / sponge_S are ignored without sponge_rate")
    return out
