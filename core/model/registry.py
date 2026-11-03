"""
Single variable registry of the 3-D core (N1, spec §3; CLAUDE.md "single variable registry").

Every variable carried by ``SigmaState`` is described once: dimensions, C-grid location, units, CF standard name
and kind (prognostic, history needed by the next step, state of a component, diagnostic).  Restart and pre-flight
use it in N1; the CF output adopts it in N4.  Index order (k, j, i), k = 0 at the bed; u-faces have nx+1 columns,
v-faces ny+1 rows, interfaces N+1 levels.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Literal


@dataclass(frozen=True)
class VarSpec:
    name: str
    dims: tuple[str, ...]
    location: Literal["t", "u", "v", "w", "q", "scalar"]
    units: str
    standard_name: str | None
    kind: Literal["prognostic", "history", "component_state", "diagnostic"]
    long_name: str = ""


_SPECS = [
    VarSpec("eta", ("y", "x"), "t", "m", "sea_surface_height_above_geoid", "prognostic", "sea surface height"),
    VarSpec("u", ("k", "y", "xq"), "u", "m s-1", "sea_water_x_velocity", "prognostic", "layer velocity at u-faces"),
    VarSpec("v", ("k", "yq", "x"), "v", "m s-1", "sea_water_y_velocity", "prognostic", "layer velocity at v-faces"),
    VarSpec("T", ("k", "y", "x"), "t", "degC", "sea_water_potential_temperature", "prognostic", "temperature"),
    VarSpec("S", ("k", "y", "x"), "t", "1e-3", "sea_water_salinity", "prognostic", "salinity"),
    VarSpec("h", ("k", "y", "x"), "t", "m", "cell_thickness", "prognostic", "ALE layer thickness"),
    VarSpec("tr", ("n", "k", "y", "x"), "t", "1", None, "prognostic", "passive tracers (ALE)"),
    VarSpec("g_hist", ("hist", "k", "y", "xq|yq"), "u", "m s-2", None, "history",
            "explicit momentum tendencies of the last steps (AB3), newest first"),
    VarSpec("Qx", ("k", "y", "xq"), "u", "m3 s-1", None, "history", "layer volume transport of the last step, x"),
    VarSpec("Qy", ("k", "yq", "x"), "v", "m3 s-1", None, "history", "layer volume transport of the last step, y"),
    VarSpec("Fz", ("kw", "y", "x"), "w", "m3 s-1", None, "history", "diapycnal (σ) volume flux of the last step"),
    VarSpec("omega", ("kw", "y", "x"), "w", "m s-1", None, "history", "σ-coordinate vertical velocity ω"),
    VarSpec("mix_num", ("k", "y", "x"), "t", "degC2 m3 s-1", None, "diagnostic",
            "online numerical mixing (Burchard & Rennau 2008)"),
    VarSpec("budget", ("term", "k", "y", "x"), "t", "(1e-3)2 m3 s-1 | degC2 m3 s-1", None, "diagnostic",
            "discrete variance budget of S and T of the last step, per operator (variance_budget, N2a; "
            "Klingbeil et al. 2014): {S,T}_{num_adv,surf,phy_h,phy_v,sponge[,num_remap]}"),
    VarSpec("n_trunc", (), "scalar", "1", None, "diagnostic", "CFL velocity truncations so far (ALE)"),
    VarSpec("n_limited", (), "scalar", "1", None, "diagnostic", "cell-layers scaled by the outflow limiter (ALE)"),
    VarSpec("n_below_freezing", (), "scalar", "1", None, "diagnostic",
            "wet cell-layers whose T is below the freezing point at the layer-centre pressure, last step "
            "(counted only; nonlinear EOS with salinity, 0 otherwise)"),
    VarSpec("n_visc_limited", (), "scalar", "1", None, "diagnostic",
            "h and q points whose Smagorinsky viscosity the AB3 stability bound reduced, summed over steps"),
    VarSpec("nu_smag", ("k", "y", "x"), "t", "m2 s-1", None, "diagnostic",
            "Smagorinsky horizontal viscosity at h points of the last step (output on request)"),
    VarSpec("obc_rtan", ("k", "yq", "xq"), "q", "1", None, "history",
            "Orlanski ratio (grid points per step) at the open-edge corners of the last step, the phase speed of the "
            "radiated tangential velocity (obc_tangential='mom6'; MOM_open_boundary.F90:3031-3036)"),
    VarSpec("closure", (), "scalar", "", None, "component_state", "state of the turbulence closure"),
    VarSpec("closure.tke", ("kw", "y", "x"), "w", "m2 s-2", None, "component_state",
            "turbulent kinetic energy (GLS k-ε)"),
    VarSpec("closure.eps", ("kw", "y", "x"), "w", "m2 s-3", None, "component_state", "dissipation rate (GLS k-ε)"),
    VarSpec("closure.akv", ("kw", "y", "x"), "w", "m2 s-1", None, "component_state", "eddy viscosity (GLS k-ε, or KPP of the last step: N3 task 7)"),
    VarSpec("closure.shear_kv", ("kw", "y", "x"), "w", "m2 s-1", None, "component_state",
            "Jackson et al. (2008) shear viscosity ν_shear = Pr κ_shear of the last step, used lagged "
            "(shear_mixing)"),
    VarSpec("kd_shear", ("kw", "y", "x"), "w", "m2 s-1", None, "diagnostic",
            "Jackson et al. (2008) shear-driven tracer diffusivity κ_shear at the layer interfaces of the last step, "
            "added to the closure's κ (shear_mixing='jackson'; ShearAddedClosure.kd_shear)"),
    VarSpec("kv_shear", ("kw", "y", "x"), "w", "m2 s-1", None, "diagnostic",
            "Jackson et al. (2008) shear-driven viscosity ν_shear = Pr κ_shear at the layer interfaces (cell "
            "centres), added to the next step's vertical viscosity (shear_mixing='jackson'; "
            "ShearAddedClosure.kv_shear)"),
    VarSpec("kd_bbl", ("kw", "y", "x"), "w", "m2 s-1", None, "diagnostic",
            "drag-driven bottom-boundary-layer diffusivity Kd_BBL at the layer interfaces of the last step (MOM6 "
            "add_LOTW_BBL_diffusivity, energy = bbl_effic × the bed-drag work), added to the closure's κ "
            "(bbl_mixing='lotw'; BBLAddedClosure.kd_bbl)"),
    # bulk surface fluxes of the last step (N2b task 4, core.components.surface_fluxes; SurfaceFluxes.last), at the
    # T points, positive into the ocean, 0 on land and dry cells
    VarSpec("q_sens", ("y", "x"), "t", "W m-2", "surface_downward_sensible_heat_flux", "diagnostic",
            "bulk sensible heat flux"),
    VarSpec("q_lat", ("y", "x"), "t", "W m-2", "surface_downward_latent_heat_flux", "diagnostic",
            "bulk latent heat flux"),
    VarSpec("q_lw", ("y", "x"), "t", "W m-2", "surface_net_downward_longwave_flux", "diagnostic",
            "net longwave radiation"),
    VarSpec("q_sw", ("y", "x"), "t", "W m-2", "surface_net_downward_shortwave_flux", "diagnostic",
            "net shortwave radiation (1 − albedo) SW↓"),
    VarSpec("evap", ("y", "x"), "t", "m s-1", None, "diagnostic",
            "evaporation as a volume flux at the model ρ₀ (mass flux / ρ₀), positive out of the ocean"),
    VarSpec("precip", ("y", "x"), "t", "m s-1", None, "diagnostic",
            "precipitation as a volume flux at the model ρ₀ (1000 P / ρ₀)"),
    VarSpec("tau_x_bulk", ("y", "x"), "t", "N m-2", "surface_downward_x_stress", "diagnostic",
            "bulk wind stress at the T points, x (averaged to the u faces for the model)"),
    VarSpec("tau_y_bulk", ("y", "x"), "t", "N m-2", "surface_downward_y_stress", "diagnostic",
            "bulk wind stress at the T points, y (averaged to the v faces for the model)"),
    VarSpec("sst_bulk", ("y", "x"), "t", "degC", "sea_surface_temperature", "diagnostic",
            "SST seen by the bulk formula: thickness-weighted mean over the top h_sfc m"),
    VarSpec("p_atm", ("y", "x"), "t", "Pa", "air_pressure_at_mean_sea_level", "diagnostic",
            "sea-level pressure p_a of the last step (msl of the bulk atmosphere at tⁿ + Δt/2), applied as "
            "−∇p_a/ρ₀ and as the inverse barometer at 'radiating' edges (atm_pressure, regional chain Task 6; an "
            "attribute of the model, not a SigmaState field)"),
    VarSpec("runoff", ("y", "x"), "t", "m s-1", None, "diagnostic",
            "river runoff Q/A_t of the last step at tⁿ + Δt/2 [volume flux, positive into the ocean], added to the "
            "real freshwater flux fw (runoff, regional chain Task 8; an attribute of the model, not a SigmaState "
            "field; MOM6 lrunoff/ρ₀ in kg m-2 s-1 / 1000)"),
    VarSpec("closure.akt", ("kw", "y", "x"), "w", "m2 s-1", None, "component_state",
            "eddy tracer diffusivity (stratified GLS k-ε, gls_stratified; N2a task 8c)"),
]

REGISTRY: dict[str, VarSpec] = {s.name: s for s in _SPECS}

# prognostic T and S of each equation of state (TEOS-10 in 3-D, spec 2026-10-31-teos10-3d goal G): the CF standard
# name and units of the variables the EOS takes (θ, S_P for Wright 1997; Θ, S_A for TEOS-10 / Roquet et al. 2015).
# The linear EOS keeps the generic entries above.
_TS_BY_EOS = {
    "wright_full": (VarSpec("T", ("k", "y", "x"), "t", "degC", "sea_water_potential_temperature", "prognostic",
                            "potential temperature θ"),
                    VarSpec("S", ("k", "y", "x"), "t", "1", "sea_water_practical_salinity", "prognostic",
                            "practical salinity S_P")),
    "teos10": (VarSpec("T", ("k", "y", "x"), "t", "degC", "sea_water_conservative_temperature", "prognostic",
                       "conservative temperature Θ"),
               VarSpec("S", ("k", "y", "x"), "t", "g kg-1", "sea_water_absolute_salinity", "prognostic",
                       "absolute salinity S_A = (35.16504/35) S_P (no δS_A, as MOM6)")),
}


def variable_specs(eos: str = "linear") -> dict[str, VarSpec]:
    """
    The registry for a model with equation of state ``eos`` ("linear", "wright_full" or "teos10"): ``REGISTRY``
    with T and S described in the convention of that EOS.  ``REGISTRY`` itself is the linear one.

    Consumers: the CF output (``diagnostics.output.OutputWriter(model=...)`` or ``(eos=...)``) and the restart
    attributes use T and S of this table.  Not every entry is a ``SigmaState`` field: ``location="scalar"``
    diagnostics such as ``n_below_freezing`` are attributes of the model, so code that iterates the registry to read
    the state must skip them (or read them from the model).
    """
    if eos == "linear":
        return dict(REGISTRY)
    if eos not in _TS_BY_EOS:
        raise ValueError(f"unknown eos {eos!r}: expected 'linear', 'wright_full' or 'teos10'")
    return {**REGISTRY, **{s.name: s for s in _TS_BY_EOS[eos]}}
