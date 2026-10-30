"""
DOME overflow, Case 1 (Ref) of Legg, Hallberg & Girton (2006), Ocean Modell. 11, 69–97 (§2, eq. 3–8, tables 1–3), in
the geometry of MOM6's implementation (src/user/DOME_initialization.F90, Hallberg) — one definition for pyOCEAN,
MOM6 and MITgcm (spec docs/superpowers/specs/2026-10-14-dome-overflow-design.md §5).  x east, y north [m]; the
inflow enters through the north edge of a 100-km channel at 1000 < x < 1100 km and turns west along the slope.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome")
RHO0, G = 1031.0, 9.8                       # MOM6 DOME MOM_input (RHO_0, G_EARTH)
ALPHA, T_REF = 0.2, 0.0                     # linear EOS ρ = ρ₀ − α (T − T_ref) [kg m⁻³ °C⁻¹, °C]
DRHO, H0, HE, HMAX, RI = 2.0, 300.0, 600.0, 3600.0, 1.0 / 3.0   # Legg tables 1–2, MOM6 DOME defaults
RHO_TOP = 1030.0                             # surface density (MOM6 LIGHTEST_DENSITY); bottom 1032

# Legg et al. 2006, Case 1 (Ref) at Δx = 10 km, read from the PDF figures (±10 km in y, ±0.5e-3 m s⁻² in b_f): mean
# path Y'(x') of the dye (Figs. 16a, 17; x' from the inlet centre, y' from the slope top, negative down-slope) and
# final buoyancy b_f = −g(ρ − 1030)/ρ₀ (Fig. 15).  z: MITgcm, ν_v = 0.5; iso: HIM, ν_v = 1e-4 (and 0.5: Fig. 17 dotted).
LEGG_10KM = dict(
    x_km=np.array([-200.0, -300.0, -500.0, -700.0, -900.0]),
    y_z_km=np.array([-80.0, -105.0, -150.0, -170.0, -250.0]),
    y_iso_km=np.array([-90.0, -95.0, -100.0, -120.0, np.nan]),
    y_iso_visc_km=np.array([-95.0, -110.0, -130.0, -145.0, np.nan]),
    bf_z=-11.7e-3, bf_iso=-11.0e-3, bf_iso_visc=-12.0e-3)


@dataclass
class Case:
    name: str
    dx: float          # grid spacing [m] (dx = dy)
    nx: int
    ny: int
    nz: int
    x: np.ndarray      # cell centres [m]
    y: np.ndarray
    H: np.ndarray      # bed depth [m], NaN = land
    f0: float          # Coriolis [s⁻¹] (f-plane)
    dt: float          # default time step [s]
    days: float        # run length [days]
    out_s: float       # output interval [s]
    nu_h: float        # Laplacian horizontal viscosity [m² s⁻¹]
    nu_v: float        # vertical viscosity [m² s⁻¹]
    cd: float          # quadratic drag coefficient
    hbbl: float        # bottom boundary layer thickness for the drag velocity [m]


def T_of_rho(rho):
    """T [°C] of density ρ [kg m⁻³] with the linear EOS."""
    return T_REF + (RHO0 - np.asarray(rho, float)) / ALPHA


def T_ambient(z):
    """Ambient T at height z ≤ 0 [m]: linear density from 1030 at the surface to 1032 at 3600 m (N² const)."""
    return T_of_rho(RHO_TOP + DRHO * np.clip(-np.asarray(z, float), 0.0, HMAX) / HMAX)


NU_V_LOW = 1e-4      # [m² s⁻¹] variant 'kv_low' (spec §5): the vertical viscosity of MOM6/MITgcm kv_low, here for the hybrid + Smagorinsky


def get(res_km: float) -> Case:
    """The case at dx = dy = res_km (10 or 5): MOM6 DOME_initialize_topography (DOME_initialization.F90:97-110)."""
    dx = float(res_km) * 1e3
    nx, ny = int(round(1500e3 / dx)), int(round(700e3 / dx))
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    H = np.minimum(HE + 0.01 * (600e3 - Y), HMAX)
    north = Y >= 600e3
    chan = (X > 1000e3) & (X < 1100e3)
    H = np.where(north, np.where(chan, HE, np.nan), H)
    nu_h = 50.0 * dx / 10e3                                        # Legg table 3: 50 m² s⁻¹ at 10 km, ∝ Δx
    return Case(name=f"dome{res_km:g}km", dx=dx, nx=nx, ny=ny, nz=25, x=x, y=y, H=H, f0=1e-4,
                dt=1200.0 * dx / 10e3, days=50.0, out_s=86400.0, nu_h=nu_h, nu_v=0.5, cd=2e-3, hbbl=10.0)


def targets(nz: int = 25) -> np.ndarray:
    """Hybrid density targets at the interfaces of MOM6's classes R_k = 1030 + (k − 1)·2/24 (layer centres)."""
    R = RHO_TOP + np.arange(nz) * DRHO / (nz - 1)
    return np.concatenate([[R[0] - 0.5 * (R[1] - R[0])], 0.5 * (R[1:] + R[:-1]), [R[-1] + 0.5 * (R[1] - R[0])]])


def _F(zeta):
    """Legg eq. 8 transition: 1 above, 0 below, ζ/(Ri(ζ+1)) + ½ between (ζ = 0 at the interface, −1 at the bed)."""
    lo, hi = -RI / (2.0 + RI), RI / (2.0 - RI)
    mid = zeta / (RI * (zeta + 1.0)) + 0.5
    return np.where(zeta >= hi, 1.0, np.where(zeta <= lo, 0.0, mid))


def inflow_profile(case: Case, nz_bdy: int = 60) -> dict:
    """
    Inflow on the north edge (Legg eq. 3–7) on a uniform z grid of the 600-m channel, surface first: z depth
    positive down in the formulas (Legg's convention, spec §5), returned as heights zw ≤ 0.
        h = h₀ e^{−x_w/L_ρ},  L_ρ = √(Δb₀h₀)/f,  ζ = (h_e − z − h)/h
        b = min(b₀ − Δb₀(1 − F), b₀ − N² z),   v = −U₀ e^{−x_w/L_ρ}(1 − F),  U₀ = √(Δb₀h₀)
    x_w is the distance from the channel's west wall (the right-hand wall looking downstream, DOME_initialization.F90
    :450-457).  Buoyancy b = −g(ρ − ρ_top)/ρ₀ → T with the linear EOS; s = 1 where the water is denser than the
    ambient at the same depth.  Columns outside the channel: ambient T, v = 0, s = 0, transport 0.
    Returns zw (nz+1, nx) [m], T (nz, nx) [°C], v (nz, nx) [m s⁻¹], s (nz, nx), transport (nx,) [m³ s⁻¹].
    """
    db0 = G * DRHO / RHO0
    U0 = np.sqrt(db0 * H0)
    Lr = U0 / case.f0
    N2 = db0 / HMAX
    zw = -np.linspace(0.0, HE, nz_bdy + 1)[:, None] * np.ones((1, case.nx))
    zc = 0.5 * (zw[1:] + zw[:-1])
    depth = -zc
    xw = case.x - 1000e3
    chan = (case.x > 1000e3) & (case.x < 1100e3)
    h = H0 * np.exp(-np.clip(xw, 0.0, None) / Lr)
    zeta = (HE - depth - h[None]) / h[None]
    F = _F(zeta)
    b = np.minimum(-db0 * (1.0 - F), -N2 * depth)                     # b₀ = 0 at the surface
    rho = RHO_TOP - RHO0 * b / G
    T = np.where(chan[None], T_of_rho(rho), T_ambient(zc))
    v = np.where(chan[None], -U0 * np.exp(-np.clip(xw, 0.0, None) / Lr)[None] * (1.0 - F), 0.0)
    s = np.where(chan[None] & (b < -N2 * depth - 1e-12), 1.0, 0.0)
    dz = -np.diff(zw, axis=0)
    transport = (v * dz).sum(axis=0) * case.dx
    return dict(zw=zw, T=T, v=v, s=s, transport=transport)


def sponge_rate(case: Case) -> np.ndarray:
    """MOM6 DOME_initialize_sponges (DOME_initialization.F90:218-240): 10 day⁻¹ within 100 km of the W/E walls,
    linear to 0 at 200 km; Idamp = max(W, E); land 0.  [s⁻¹], (ny, nx)."""
    peak = 10.0 / 86400.0

    def ramp(d):
        return np.where(d < 100e3, peak, np.where(d < 200e3, peak * (200e3 - d) / 100e3, 0.0))
    r = np.maximum(ramp(case.x), ramp(1500e3 - case.x))[None] * np.ones((case.ny, 1))
    return np.where(np.isfinite(case.H), r, 0.0)


def mom6_layer_transports(nz: int = 25) -> np.ndarray:
    """MOM6 DOME_set_OBC_data per-layer transports tr_k [m³ s⁻¹] (DOME_initialization.F90:402-446), Boussinesq."""
    gp = G * DRHO / RHO0
    c = np.sqrt(H0 * gp)
    tr0 = -H0 * c * 0.5 * c / 1e-4

    def y(r):
        return (2 * RI * r + RI + 2) / (2 - RI)
    out = np.empty(nz)
    for k in range(1, nz + 1):
        rst = -1.0 if k == 1 else -1.0 + (k - 1.5) / (nz - 1)
        rsb = 0.0 if k == nz else -1.0 + (k - 0.5) / (nz - 1)
        yt, yb = y(rst), y(rsb)
        t = tr0 * 2 / (RI * (2 - RI)) * ((np.log(yt) + 1) / yt - (np.log(yb) + 1) / yb)
        if k == nz:
            t += tr0 * 2 / (RI * (2 + RI)) * np.log((2 + RI) / (2 - RI))
        out[k - 1] = t
    return out


def run_dir(res_km: float, model: str, coord: str, tag: str = "") -> Path:
    """Output directory of one run."""
    return RUNS / f"{res_km:g}km" / f"{model}_{coord}{('_' + tag) if tag else ''}"
