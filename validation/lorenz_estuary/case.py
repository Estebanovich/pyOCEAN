"""
Case definition of the idealized 2-D estuary of Lorenz, Klingbeil & Burchard (2021, "Impact of evaporation and
precipitation on estuarine mixing", JPO 51, 1319-1333, doi:10.1175/jpo-d-20-0158.1; local copy
``Lorenz_2021 - Impact of Evaporation and Precipitation on Estuarine Mixing.md``, §4b l. 838-857, Table 1 l. 881-906,
Fig. 2 caption l. 827-837, Table 2 l. 923-1030; transect positions read from Fig. 2 of the PDF, p. 1324).

Everything the paper gives is used as given; what it does not give is marked **[no dado en el artículo]** with the
simplest choice or GETM's default (marked [verificar] when that default is quoted from memory), never tuned to
Table 2 (CLAUDE.md, Validation Strategy).

Geometry (§4b): channel 100 km long, dx = dy = 200 m (500 × 1 cells), bed linear from 15 m at the ocean boundary
(west, x = 0) to 5 m at the river end (east, x = 100 km), 60 uniform σ-layers.  Ocean boundary: S = 30 g kg⁻¹,
M2 tide of 0.5 m amplitude when tides are on.  Every freshwater transport (river, area-integrated E, P) is
5.0 m³ s⁻¹.  T constant.  500 M2 cycles to the quasi-periodic state, then a 100-cycle average.

Table 1 (m³ s⁻¹, m):  Exp  Q_r  Q_evap  Q_precip  M2
                       A   0.0   −5.0     0.0     0.0     E over the inner (east) half
                       B   5.0   −5.0     0.0     0.5     E over the middle third
                       C   0.0   −5.0     5.0     0.5     E middle third, P east third
                       D   5.0   −5.0     5.0     0.5     E middle third, P east third

Two pyOCEAN configurations (spec §6, Esteban):
  * ``sigma_gls``: Eulerian σ, stratified k-ε GLS (Canuto A with buoyancy, physics/mixing/gls.py, N2a task 8c:
    N² from ρ(S), B = −κ_t N², α_N in the stability functions) for ν and for the vertical diffusivity κ_t of S,
    as GETM's k-ε (GOTM); log-law bed drag z0 = 1e-3 m; ``kv`` = 0, so κ_t keeps CROCO's floor 1e-6 m² s⁻¹
    (task 8c supersedes N2a Review Focus 5, where S had the constant ``kv`` and GLS had no buoyancy);
  * ``ale_sigma``: ALE with the σ grid rule, constant ν and κ, quadratic bed drag over the whole column.

Lorenz's exact relation (eq. 32) needs, in pyOCEAN (diagnostics/tef.py, task 7 report): kh = 0 (the diffusive
transect flux has no volume, TEF cannot bin it), no sponge cells in the region, and every open face of the region a
fresh inflow ('clamped' river, S = 0).  ``check_identity_assumptions`` asserts them.
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

from pathlib import Path
from typing import Callable

import numpy as np

from boundaries.obcs import OBData
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from diagnostics.tef import TEFRegion, TEFTransect, lorenz_mixing

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/lorenz_estuary")

# --- geometry (Lorenz 2021 §4b, l. 838-843) ------------------------------------------------------------------------
NX, NY, NZ = 500, 1, 60
DX = 200.0                                  # [m] dx = dy
L = NX * DX                                 # [m] 100 km
H_OCEAN, H_RIVER = 15.0, 5.0                # [m] bed depth at the ocean boundary (x = 0) and the river end (x = L)
I_THIRD = int(round(NX / 3))                # 167: first cell of the middle third (x = 33.4 km)
I_TWO_THIRDS = int(round(2 * NX / 3))       # 333: first cell of the east third (x = 66.6 km)

# --- forcing (Table 1, l. 881-906; §4b l. 843-846) -----------------------------------------------------------------
TABLE_1: dict[str, tuple[float, float, float, float]] = {   # Q_r, Q_evap, Q_precip [m³ s⁻¹], M2 amplitude [m]
    "A": (0.0, -5.0, 0.0, 0.0),
    "B": (5.0, -5.0, 0.0, 0.5),
    "C": (0.0, -5.0, 5.0, 0.5),
    "D": (5.0, -5.0, 5.0, 0.5),
}
S_OCEAN = 30.0                              # [g kg⁻¹] open-boundary salinity (§4b l. 843)
S_RIVER = 0.0                               # [g kg⁻¹] river water (fresh)
T_M2 = 12.4206012 * 3600.0                  # [s] M2 period (44 714.16 s)
OMEGA_M2 = 2.0 * np.pi / T_M2               # [s⁻¹]
SPINUP_CYCLES, AVG_CYCLES = 500, 100        # §4b l. 851-857: 500 cycles to the quasi-periodic state, 100-cycle mean

# Transects (dashed lines of Fig. 2): the west edge of the evaporation zone — x = 50 km in A, x = L/3 in B–D.
# The region (estuary) is every cell east of the u-face column ``TRANSECT[exp]``.
TRANSECT = {"A": NX // 2, "B": I_THIRD, "C": I_THIRD, "D": I_THIRD}

# --- physics not given in the article ------------------------------------------------------------------------------
G = 9.81                                    # [m s⁻²]
RHO0 = 1025.0                               # [kg m⁻³] [no dado en el artículo] GETM rho_0 = 1025 [verificar]
S_COEF = 0.78                               # [kg m⁻³ (g kg⁻¹)⁻¹] β [no dado en el artículo]: GETM linear EOS dsr0
                                            #   = 0.78 (getm.inp template) [verificar]; GETM's default may be UNESCO,
                                            #   whose β at S ≈ 30, T = 10 °C is ≈ 0.77
S_REF = S_OCEAN                             # [g kg⁻¹] S₀ of the linear EOS (only shifts ρ; no dynamical effect)
T_CONST = 10.0                              # [°C] constant temperature [no dado en el artículo]; α = 0
KV = 1e-4                                   # [m² s⁻¹] ale_sigma: vertical diffusivity of S and viscosity ν
                                            #   [no dado en el artículo] — GETM: GOTM k-ε + background avmback =
                                            #   avhback = 0 [verificar] plus molecular; 1e-4 is pyOCEAN's default
                                            #   nu_const, the simplest constant (not tuned)
KV_GLS = 0.0                                # [m² s⁻¹] sigma_gls: floor of the GLS κ_t; 0 = CROCO's nuws = 1e-6
                                            #   (gls_mixing.F:152), closest to GETM's zero background [verificar]
NU_ALE = KV                                 # [m² s⁻¹] ALE constant viscosity (same choice)
AH = 0.0                                    # [m² s⁻¹] horizontal viscosity [no dado en el artículo]; 0 = GETM Am
                                            #   default [verificar]
KH = 0.0                                    # [m² s⁻¹] horizontal diffusivity: 0 (needed by the exact TEF relation)
Z0 = 1e-3                                   # [m] bed roughness, sigma_gls log law [no dado en el artículo] (brief)
H_BBL = 20.0                                # [m] ALE: drag on the depth-mean velocity (hbbl ≥ the deepest column)
SPONGE_CELLS = 10                           # west sponge (2 km) [no dado en el artículo]; GETM prescribes S at the
                                            #   boundary without a sponge; pyOCEAN's tested Orlanski +
                                            #   sponge (spec §6); its inner edge (x = 2 km) is
                                            #   31.4 km west of the B–D transect (x = 33.4 km)
INIT_S = S_OCEAN                            # initial S uniform at rest [no dado en el artículo]
# [s] default steps, a whole number per M2 cycle (Δt search with the prescribed-η tide, README):
#   sigma_gls: T_M2/746 ≈ 59.9 s — A blows up at 240 s (explicit vertical advection of S in the ≈ 0.13 m top layer
#     under the evaporation jet, ~22 h); with the stratified GLS (task 8c, ν down to 1e-5 under the halocline) the
#     ebb surface jet reaches 1–1.5 m/s near x = 68 km and at 120 s (|u| Δt/Δx ≈ 0.75–0.9) B, C and D blow up at
#     cycles 99–134 (grid-scale 3–4 Δx mode of the AB3 WENO5 momentum, A_h = 0, growing ×1.3 per step); 60 s keeps
#     |u| Δt/Δx ≤ 0.45 (A ran 600 cycles at 120 s; all four use 60 s);
#   ale_sigma: T_M2/498 ≈ 89.8 s — B stops at 150 s (CFL velocity truncation, |u| dt/dx > 0.5, of the 0.67 m/s ebb at
#     the mouth face) and is stable at 120 s, where the 0.78 m/s peak already gives |u| dt/dx = 0.47
DT_DEFAULT = {"sigma_gls": T_M2 / 746, "ale_sigma": T_M2 / 498}


def cd_loglaw(D: float = 10.0, z0: float = Z0, kappa: float = 0.4) -> float:
    """
    Drag coefficient of the depth-mean velocity for a logarithmic profile u(z) = (u*/κ) ln(z/z0) over depth D:
    ū = (u*/κ)(ln(D/z0) − 1), so C_d = [κ / (ln(D/z0) − 1)]² (Soulsby 1997, eq. 37 with the depth-mean form).  At the
    mid-channel depth D = 10 m and z0 = 1e-3 m: C_d = 2.37e-3.  Used by ``ale_sigma`` so both configurations have
    the same bed roughness (derived, not tuned).
    """
    return (kappa / (np.log(D / z0) - 1.0)) ** 2


def bathymetry() -> np.ndarray:
    """H (ny, nx) [m] at cell centres x_i = (i + ½) dx of the linear bed 15 m (x = 0) → 5 m (x = L)."""
    x = (np.arange(NX) + 0.5) * DX
    return (H_OCEAN + (H_RIVER - H_OCEAN) * x / L)[None, :]


def fw_field(exp: str, area: np.ndarray) -> np.ndarray:
    """
    Surface freshwater flux fw = P − E (ny, nx) [m s⁻¹, positive into the ocean], uniform over each zone so that
    Σ fw·A = Q_evap over the E zone and Q_precip over the P zone (Table 1; caption of Fig. 2: A, E over the inner
    half; B–D, E over the middle third and P over the east third).
    """
    _, Qe, Qp, _ = TABLE_1[exp]
    i = np.arange(NX)[None, :] * np.ones((NY, 1), int)
    if exp == "A":
        e_zone, p_zone = i >= NX // 2, np.zeros((NY, NX), bool)
    else:
        e_zone, p_zone = (i >= I_THIRD) & (i < I_TWO_THIRDS), (i >= I_TWO_THIRDS) & (Qp != 0)
    fw = np.zeros((NY, NX))
    fw[e_zone] = Qe / area[e_zone].sum()
    if p_zone.any():
        fw[p_zone] = Qp / area[p_zone].sum()
    return fw


def open_boundary_state(exp: str, H: np.ndarray) -> Callable[[float], dict]:
    """
    Exterior state t [s] → {edge: OBData}: west, the ocean at S = 30 and η = a sin(ω t) (M2, a from Table 1; starts
    at 0, no ramp [no dado en el artículo]) imposed as the sea level of the west cells (open_boundary_mode 'eta',
    Dirichlet as GETM's elevation boundary, ruling R4; u′ and S as 'radiating', S inflow 30 + sponge); east
    (Q_r > 0), the river, fresh, 'clamped' with its exact transport −Q_r (into the domain, model sign).  T constant.
    """
    Qr, _, _, amp = TABLE_1[exp]
    nz_e = 4

    def edge(side: str, depth: float, S: float, q: float | None, eta: float) -> OBData:
        zw = np.linspace(0.0, -depth, nz_e + 1)[:, None]
        u = np.zeros((nz_e, NY)) if q is None else np.full((nz_e, NY), q / (depth * DX))
        return OBData(side=side, T=np.full((nz_e, NY), T_CONST), S=np.full((nz_e, NY), S), u_normal=u, zw=zw,
                      eta=np.full(NY, eta), transport=None if q is None else np.full(NY, q))

    def state(t: float) -> dict:
        out = {"west": edge("west", H[0, 0], S_OCEAN, None, amp * np.sin(OMEGA_M2 * t))}
        if Qr > 0:
            out["east"] = edge("east", H[0, -1], S_RIVER, -Qr, 0.0)
        return out
    return state


def config_kwargs(config: str) -> dict:
    """Model arguments of the two configurations (module docstring)."""
    if config == "sigma_gls":
        return dict(vertical_coordinate="sigma_eulerian", closure="gls", z0=Z0, kv=KV_GLS, gls_stratified=True)
    if config == "ale_sigma":
        return dict(vertical_coordinate="ale", regrid="sigma", pgf="fv", closure="constant", nu_const=NU_ALE, z0=None,
                    kv=KV, bottom_drag="quadratic", cd=cd_loglaw(), hbbl=H_BBL)
    raise ValueError(f"unknown config {config!r}; use 'sigma_gls' or 'ale_sigma'")


def build(exp: str, config: str, dt: float | None = None, S_init: np.ndarray | None = None, **overrides):
    """
    The model of experiment ``exp`` ∈ {A, B, C, D} in ``config`` ∈ {sigma_gls, ale_sigma}, its initial state (rest,
    η = 0, S = 30, T constant) and ``forcing(n) → dict(tau_x=None, tau_y=None, fw=…)`` for step n (fw constant).
    The model carries salinity, the exact variance budget and the in-model TEF on the transect of ``exp``.
    ``S_init`` (nz, ny, nx) replaces the uniform initial S (tests); other keywords override model arguments.
    """
    if exp not in TABLE_1:
        raise ValueError(f"unknown experiment {exp!r}; use one of {sorted(TABLE_1)}")
    Qr = TABLE_1[exp][0]
    H = bathymetry()
    hg = create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DX, f0=0.0)        # 2-D slice: f = 0 [no dado en el artículo]
    region = np.zeros((NY, NX), bool)
    region[:, TRANSECT[exp]:] = True
    edges = ("west", "east") if Qr > 0 else ("west",)
    modes = {"west": "eta"}                                           # prescribed M2 elevation (Lorenz l. 843-845)
    modes.update(overrides.pop("open_boundary_mode", None) or {})
    if Qr > 0:
        modes["east"] = "clamped"                                     # the river: exact transport, never dropped
    kw = dict(rho0=RHO0, t_coef=0.0, t_ref=T_CONST, g=G, salinity=True, s_coef=S_COEF, s_ref=S_REF, ah=AH, kh=KH,
              open_boundaries=edges, open_boundary_mode=modes,
              open_boundary_state=open_boundary_state(exp, H), sponge_cells=SPONGE_CELLS,
              variance_budget=True, tef_face=TRANSECT[exp], tef_sign=1, tef_region=region)
    kw.update(config_kwargs(config))
    kw.update(overrides)
    m = SigmaBaroclinic(hg, H, dt=DT_DEFAULT[config] if dt is None else dt, n_layers=NZ, **kw)
    S0 = np.full((NZ, NY, NX), INIT_S) if S_init is None else S_init
    st = m.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), T_CONST), S=S0)
    fw = fw_field(exp, np.asarray(m.area))

    def forcing(n: int) -> dict:
        return dict(tau_x=None, tau_y=None, fw=fw)
    check_identity_assumptions(m)
    return m, st, forcing


def check_identity_assumptions(m: SigmaBaroclinic) -> None:
    """
    Assert what Lorenz's exact relation needs and the model does not enforce (task 7 report, concerns 1–2):
    kh = 0; no sponge cell in the region; every open face of the region a 'clamped' fresh inflow (S = 0, transport
    into the domain); salinity, the variance budget and the TEF transect on.
    """
    assert m.salinity and m.variance_budget and m.tef is not None and m.tef_region_budget is not None, \
        "TEF/budget not on"
    reg = m.tef_region_budget.mask
    assert m.kh == 0.0, "kh > 0: the diffusive transect flux is not binned by TEF"
    assert m._sp_c is None or not (m._sp_c[reg] > 0).any(), "sponge cells inside the region"
    assert not reg[:, : m.tef_face].any() and reg[:, m.tef_face:].all(), "region ≠ cells east of the transect"
    ext = m.obc_state(0.0) if m.obc_state is not None else {}
    for edge, inside in (("west", reg[:, 0].any()), ("east", reg[:, -1].any()), ("south", reg[0].any()),
                         ("north", reg[-1].any())):
        if edge in m.open and inside:
            ob = ext[edge]
            assert m.obc_mode[edge] == "clamped" and ob.transport is not None, f"{edge}: not a clamped river"
            assert (np.asarray(ob.S) == 0.0).all(), f"{edge}: river not fresh"
            into = -1.0 if edge in ("east", "north") else 1.0
            assert (into * np.asarray(ob.transport) >= 0).all() and (into * np.asarray(ob.u_normal) >= 0).all(), \
                f"{edge}: river outflow"


def reset_tef(m: SigmaBaroclinic) -> None:
    """Start new TEF and region sums (the averaging window begins at the next step)."""
    m.tef = TEFTransect(m.tef_face, sign=m.tef.sign, n_bins=m.tef.n_bins,
                        s_range=(float(m.tef.edges[0]), float(m.tef.edges[-1])))
    m.tef_region_budget = TEFRegion(m.tef_region_budget.mask)


def summary(m: SigmaBaroclinic) -> dict:
    """
    Time-mean results of the window since the last ``reset_tef``: TEF bulk values (Lorenz eqs. 20–22), region inputs
    (Qr, Qsurf, dV, dS, dS2, F_surf^{s²}, M_phy, M_num, M_tot, M_sponge), Me/Mc/Mp/Mcp (eqs. 32–35), s_surf (36) and
    M_model_tot (Table 2 rows).
    """
    bulk = m.tef.bulk()
    r = m.tef_region_budget.inputs()
    lm = lorenz_mixing(bulk, r["Qr"], r["Qsurf"], r["dV"], r["dS"], r["dS2"], r["Fs2_surf"])
    out = {k: float(v) for k, v in bulk.items()}
    out.update({k: float(v) for k, v in r.items()})
    out.update({k: float(v) for k, v in lm.items()})
    out.update(M_model_tot=float(r["M_tot"]), M_model_phy=float(r["M_phy"]), M_model_num=float(r["M_num"]),
               window_s=float(m.tef.time))
    return out
