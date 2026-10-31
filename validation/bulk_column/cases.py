"""
ERA5-forced single column on the US northeast shelf: the case definition shared by pyOCEAN, MITgcm and CROCO
(N2b task 7, spec docs/superpowers/specs/2026-10-31-n2b-bulk-surface-fluxes-design.md, criterion 5).

Every value the three models must share is defined here once; the reference inputs (``mitgcm/input/data*``,
``croco/croco.in``, the CROCO source patches of ``croco/build_croco.sh``) repeat them and
``tests/test_bulk_column.py`` checks that they agree.  Nothing is tuned to the result (CLAUDE.md, Validation Strategy).

Point (README §1)
-----------------
P = 40.24 °N, 73.16 °W, 1.3 km south-southwest of NDBC buoy 44025 (40.251 °N, 73.164 °W, water depth 36.3 m
[verificar]).  The bilinear ERA5 stencil of P is the four 0.25° nodes 40.00/40.25 °N × 73.25/73.00 °W, all sea:
their July 2017 mean daily range of t2m is 1.55–1.71 K, the open-ocean value (1.7–1.8 K at 70 °W), while the
nodes touching Long Island (40.50 °N, 73.25 °W) and New Jersey (74.00 °W) give 1.9–7.3 K.  The buoy position
itself (40.251 °N) would put a 0.4 % weight on the 40.50 °N row, which is why P sits just south of it.  ERA5
values over land are not masked (Task 3 review m4), so this is the check that the column sees no land air.

Column
------
H = 36 m (the 44025 depth), 36 uniform 1 m layers in the three models (CROCO σ with θ_s = θ_b = 0 and
h_c = 1e16, MITgcm delR = 36 × 1 m, pyOCEAN σ), flat bottom, f = 2Ω sin φ at P.  pyOCEAN and CROCO: 5 × 5 doubly
periodic, Δx = 2 km (the ``validation/turbulence_columns`` box); MITgcm 1 × 1 (a periodic singleton,
``verification/1D_ocean_ice_column``).  Δt = 120 s.  Period 2017-07-01 00:00 → 2017-08-15 00:00 UTC (45 days).

Initial state: at rest; T and S of the World Ocean Atlas 2023, July, all decades (decav), 0.25°, statistical mean
``t_an``/``s_an``, the cell 40.0–40.25 °N × 73.25–73.0 °W that contains P (centre 40.125 °N, 73.125 °W; data to
40 m), linearly interpolated in depth to the layer centres.  Source (OPeNDAP, read 2026-10-31):
https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/temperature/netcdf/decav/0.25/woa23_decav_t07_04.nc and
the same path with salinity/s07; the subset is kept in ``RUNS/woa/``.

Physics shared by the three models
----------------------------------
* linear EOS ρ = ρ₀ [1 − α(T − T_ref) + β(S − S_ref)], ρ₀ = 1025 kg m⁻³, α = 2.0e-4 K⁻¹, β = 7.5e-4 (g/kg)⁻¹
  (TEOS-10 ``gsw.alpha``/``gsw.beta`` at 15 °C, 31.5 g/kg, p = 0: 2.06e-4, 7.44e-4); T_ref = 15 °C, S_ref = 31.5.
  The 3-D pyOCEAN model has only the linear EOS (spec decision 11), and CROCO SINGLE_COLUMN uses its linear EOS;
* c_p = 3985 J kg⁻¹ K⁻¹ (CROCO scalars.h:640; MITgcm HeatCapacity_Cp and pyOCEAN ``cp`` set to it), g = 9.81;
* KPP for the surface boundary layer only: pyOCEAN ``closure='kpp'`` (CVMix port, ``KPPParams()``), MITgcm
  ``pkg/kpp`` with the interior shear and convective mixing off (difm0 = difs0 = dift0 = 0, difmcon = difscon =
  diftcon = 0), CROCO ``LMD_SKPP`` + ``LMD_NONLOCAL`` (the 'kppsfc' set of ``validation/turbulence_columns``).
  No background viscosity or diffusivity in any model (pyOCEAN's KPP default, spec anti-goal);
* bed stress: law of the wall on the bottom layer, z₀ = 1 mm (pyOCEAN ``z0``, CROCO Zob, MITgcm zRoughBot);
* shortwave: (1 − 0.066) SW↓ in the three models (LY04 albedo, pyOCEAN ``BulkFluxes.albedo``; MITgcm exf_albedo;
  CROCO receives the net SW in ``radsw``), absorbed with the Paulson & Simpson (1977) two-band profile of Jerlov
  type III in the three (pyOCEAN ``jerlov='III'``, MITgcm swfrac.F jwtype = 5 and CROCO lmd_swfrac.F
  lmd_Jwt = 5 patched in the build copies; their source fixes type IA and I).  Type III is the turbid end of the
  Jerlov open-ocean types; the summer Middle Atlantic Bight shelf is coastal water [verificar], and with type I
  8.8 % of the SW would reach the 36 m bed, where the three models treat it differently (README §2);
* longwave: ε (LW↓ − σ SST⁴), ε = 0.97 for LY04 (MITgcm ``EXF_LWDOWN_WITH_EMISSIVITY``, ocean_emissivity 0.97;
  pyOCEAN ``radiative_fluxes``); CROCO's own LW↓ − 0.985 σ SST⁴ for COARE (pyOCEAN ``CoareFluxes.q_lw``);
* SST and surface current of the bulk formulae = the top 1 m (pyOCEAN ``h_sfc`` = 1 m = the top layer here).

Index order (k, j, i), k = 0 at the bed (pyOCEAN); profiles saved bed first.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bulk_column")
ERA5_DIR = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/forcing/era5")

# ------------------------------------------------------------------------------------------------ point and column
LAT, LON = 40.24, -73.16          # [°N], [°E]
BUOY = ("NDBC 44025", 40.251, -73.164, 36.3)      # name, lat, lon, depth [m] [verificar]
DEPTH = 36.0                      # [m]
N = 36                            # layers (1 m)
NXY, DX = 5, 2000.0               # pyOCEAN / CROCO box: 5 × 5, Δx = 2 km (CROCO SINGLE_COLUMN Length 10 km)
DT = 120.0                        # [s]
OMEGA = 7.2921e-5                 # [rad s⁻¹]
F0 = 2.0 * OMEGA * np.sin(np.deg2rad(LAT))        # [s⁻¹] ≈ 9.42e-5

START = datetime(2017, 7, 1)
DAYS = 45                          # 2017-07-01 00:00 → 2017-08-15 00:00
DAYS_JULY = 30                     # fallback if the August accumulations are missing: 1–31 July 00:00 (ruling)

# ------------------------------------------------------------------------------------------------ shared physics
RHO0 = 1025.0                     # [kg m⁻³]
ALPHA, BETA = 2.0e-4, 7.5e-4      # [K⁻¹], [(g/kg)⁻¹]
T_REF, S_REF = 15.0, 31.5         # [°C], [g/kg]
CP = 3985.0                       # [J kg⁻¹ K⁻¹]
G = 9.81                          # [m s⁻²]
Z0 = 1e-3                         # [m] bed roughness
ALBEDO = 0.066                    # [–]
EMISSIVITY = 0.97                 # [–] LY04 / MITgcm
JERLOV, JERLOV_INDEX = "III", 5   # Paulson & Simpson type (MITgcm jwtype / CROCO lmd_Jwt index 1..5 = I..III)
RHO_FW = 1000.0                   # [kg m⁻³] ERA5 tp is metres of liquid water

# ------------------------------------------------------------------------------------------------ WOA23 profile
WOA_URL = ("https://www.ncei.noaa.gov/thredds-ocean/dodsC/woa23/DATA/{var}/netcdf/decav/0.25/"
           "woa23_decav_{v}07_04.nc")
WOA_CELL = (40.125, -73.125)      # centre of the 0.25° cell holding P
WOA_DEPTH = np.array([0.0, 5.0, 10.0, 15.0, 20.0, 25.0, 30.0, 35.0, 40.0])                  # [m]
WOA_T = np.array([22.09, 21.17, 18.93, 15.42, 12.25, 10.01, 8.27, 7.56, 6.81])               # t_an [°C]
WOA_S = np.array([30.84, 30.99, 31.35, 31.84, 32.15, 32.32, 32.40, 32.42, 32.45])            # s_an [g/kg]


def layer_depths() -> np.ndarray:
    """Depths of the layer centres below the surface [m], bed first (N,)."""
    return (DEPTH - (np.arange(N) + 0.5) * (DEPTH / N))


def initial_TS() -> tuple[np.ndarray, np.ndarray]:
    """WOA23 July T [°C] and S [g/kg] at the layer centres, bed first (N,), linear in depth."""
    d = layer_depths()
    return np.interp(d, WOA_DEPTH, WOA_T), np.interp(d, WOA_DEPTH, WOA_S)


def rho_linear(T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """ρ = ρ₀ [1 − α(T − T_ref) + β(S − S_ref)] [kg m⁻³] (the shared linear EOS)."""
    return RHO0 * (1.0 - ALPHA * (np.asarray(T) - T_REF) + BETA * (np.asarray(S) - S_REF))


# ------------------------------------------------------------------------------------------------ period, ERA5
def period(days: int = DAYS) -> tuple[datetime, datetime]:
    return START, START + timedelta(days=days)


def n_steps(days: int = DAYS) -> int:
    return int(round(days * 86400.0 / DT))


def era5_paths(days: int = DAYS) -> list[str]:
    """The instantaneous and accumulated ERA5 files covering ``period(days)`` (the 'mean' files are not used)."""
    end = START + timedelta(days=days, hours=1)                 # + the adapter's one-record margin
    months = sorted({(START + timedelta(hours=h)).strftime("%Y-%m")
                     for h in range(int((end - START).total_seconds() // 3600) + 1)})
    paths = []
    for m in months:
        for grp in ("instant", "accum"):
            p = ERA5_DIR / f"era5_nwa_shelf_{m}_{grp}.nc"
            if not p.exists():
                raise FileNotFoundError(f"missing ERA5 file {p}")
            paths.append(str(p))
    return paths


def era5_available(days: int = DAYS) -> bool:
    try:
        era5_paths(days)
        return True
    except FileNotFoundError:
        return False


def grid():
    """The 5 × 5 Cartesian grid with every T point at P (lon_t, lat_t) for the ERA5 provider."""
    import xarray as xr

    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=NXY, ny=NXY, dx=DX, dy=DX, f0=F0)
    for pt, (x, y) in {"t": ("x_t", "y_t"), "u": ("x_u", "y_u"), "v": ("x_v", "y_v")}.items():
        if x in hg and y in hg:
            hg[f"lon_{pt}"] = xr.zeros_like(hg[x] + hg[y]) + LON
            hg[f"lat_{pt}"] = xr.zeros_like(hg[x] + hg[y]) + LAT
    return hg


def atmosphere(days: int = DAYS):
    """``forcing.atmosphere.era5_atmosphere`` at P for ``period(days)`` (model time 0 = START)."""
    from forcing.atmosphere import era5_atmosphere
    t0, t1 = period(days)
    return era5_atmosphere(era5_paths(days), grid(), np.datetime64(START, "ns"), t0, t1)


def point_series(days: int = DAYS, atm=None) -> dict[str, np.ndarray]:
    """
    The canonical atmosphere at P on the hourly ERA5 stamps 0, 1 h, …, ``days`` d (the provider's own records,
    so a linear interpolation of these samples is the provider's): time [s], u10, v10 [m s⁻¹], t2m [K],
    q2m [kg kg⁻¹], msl [Pa], sw_down, lw_down [W m⁻²], precip [m s⁻¹ of liquid water].  The same series feeds the
    MITgcm and CROCO inputs.
    """
    atm = atmosphere(days) if atm is None else atm
    t = np.arange(days * 24 + 1) * 3600.0
    out: dict[str, list] = {}
    for tk in t:
        rec = atm(tk)
        for k, v in rec.items():
            if np.ptp(v) > 1e-9 * max(1.0, float(np.abs(v).max())):
                raise RuntimeError(f"{k} is not uniform over the column box")
            out.setdefault(k, []).append(float(v[0, 0]))
    res = {k: np.array(v) for k, v in out.items()}
    res["time"] = t
    return res


# ------------------------------------------------------------------------------------------------ pyOCEAN
@dataclass(frozen=True)
class Run:
    """One pyOCEAN configuration of the benchmark."""
    tag: str
    scheme: str                   # BulkFluxes.scheme
    relative_wind: bool = True
    h_sfc: float = 1.0
    prescribed: str | None = None    # 'mitgcm' | 'croco': no bulk, the reference's own hourly-mean fluxes
    croco_background: bool = False   # KPP interior ν = 1e-4, κ = 1e-5 m² s⁻¹ (CROCO lmd_vmix.F:116-117, hard-wired)
    h_mix: float = 0.0               # horizontal viscosity = diffusivity [m² s⁻¹] (ah, kh): zero on a uniform column,
                                     # damps the KPP-amplified round-off differences between the 25 columns (README §5)


RUNS_PYOCEAN: dict[str, Run] = {
    "ly04": Run("ly04", "ncar_ly04"),                           # vs MITgcm exf (NCAR reference)
    "coare": Run("coare", "coare"),                             # vs CROCO bulk_flux.F (COARE reference)
    # CROCO's coupling (step 4).  Without h_mix the 25 columns drifted apart from round-off to 3.1 °C: KPP amplifies
    # column-to-column differences in discrete OBL-deepening events, and absolute wind lacks the relative-wind
    # damping between the events (README §5)
    "coare_croco": Run("coare_croco", "coare", relative_wind=False, h_sfc=0.0, h_mix=100.0),
    "fms": Run("fms", "ncar"),                                  # pyOCEAN default (FMS form), for information
    # diagnostic runs that separate the ocean model from the bulk formula (README §4)
    # (prescribed stress has no relative-wind feedback either: the same h_mix as coare_croco, inert on a 1-D column)
    "flux_mitgcm": Run("flux_mitgcm", "none", prescribed="mitgcm", h_mix=100.0),
    "flux_croco": Run("flux_croco", "none", prescribed="croco", h_mix=100.0),
    "flux_croco_bg": Run("flux_croco_bg", "none", prescribed="croco", croco_background=True, h_mix=100.0),
    "coare_bg": Run("coare_bg", "coare", croco_background=True, h_mix=100.0),
}
BULK_RUNS = ("ly04", "coare", "coare_croco", "fms")
#: CROCO's interior mixing with LMD_MIXING (lmd_vmix.F:116-117, lmd_nuwm, lmd_nuws; always added below the OBL)
CROCO_NU_BG, CROCO_KAPPA_BG = 1.0e-4, 1.0e-5


def build_pyocean(run: Run, days: int = DAYS, atm=None):
    """(model, state) of a pyOCEAN run: σ, KPP, salinity, bulk fluxes from ERA5 at P (or no bulk component for
    the prescribed-flux runs, which ``run_pyocean`` forces with the reference's hourly fluxes)."""
    from core.components.surface_fluxes import BulkFluxes
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from physics.mixing.kpp_cvmix import KPPParams
    hg = grid()
    bulk = None
    if run.prescribed is None:
        atm = atmosphere(days) if atm is None else atm
        bulk = BulkFluxes(atm, scheme=run.scheme, relative_wind=run.relative_wind, h_sfc=run.h_sfc,
                          albedo=ALBEDO, emissivity=EMISSIVITY)
    kpp = (KPPParams(kappa_bg=CROCO_KAPPA_BG, nu_bg=CROCO_NU_BG) if run.croco_background else KPPParams())
    kw = dict(rho0=RHO0, t_coef=RHO0 * ALPHA, t_ref=T_REF, salinity=True, s_coef=RHO0 * BETA, s_ref=S_REF,
              closure="kpp", kpp_params=kpp, z0=Z0, g=G, cp=CP, jerlov=JERLOV, theta=0.6, d_min=0.05,
              periodic=("x", "y"), surface_fluxes=bulk, ah=run.h_mix, kh=run.h_mix)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        model = SigmaBaroclinic(hg, np.full((NXY, NXY), DEPTH), dt=DT, n_layers=N, **kw)
    T, S = initial_TS()
    one = np.ones((N, NXY, NXY))
    st = model.initial_state(np.zeros((NXY, NXY)), T[:, None, None] * one, S=S[:, None, None] * one)
    return model, st
