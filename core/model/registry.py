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
    VarSpec("closure", (), "scalar", "", None, "component_state", "state of the turbulence closure"),
    VarSpec("closure.tke", ("kw", "y", "x"), "w", "m2 s-2", None, "component_state",
            "turbulent kinetic energy (GLS k-ε)"),
    VarSpec("closure.eps", ("kw", "y", "x"), "w", "m2 s-3", None, "component_state", "dissipation rate (GLS k-ε)"),
    VarSpec("closure.akv", ("kw", "y", "x"), "w", "m2 s-1", None, "component_state", "eddy viscosity (GLS k-ε)"),
]

REGISTRY: dict[str, VarSpec] = {s.name: s for s in _SPECS}
