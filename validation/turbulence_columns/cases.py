"""
The seven CROCO single-column cases (``#define SINGLE_COLUMN``) as pyOCEAN columns (N3 task 8; spec
docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md, criteria 1 and 3, item 6.1).

Every value below is CROCO's (``docs/design/n2c_refs/croco_column_cases.md``, which cites each ``file:line`` of
``~/pyOCEAN_refs/croco/OCEAN``); nothing is tuned (CLAUDE.md, Validation Strategy).

Common to all cases (croco_column_cases.md §0):

* 5 × 5 doubly periodic box, dx = dy = 2 km (param.h:78-93, ana_grid.F:297-300), flat bottom, uniform σ
  (THETA_S = THETA_B = 0, Hc = 1e16), Δt = 30 s;
* linear EOS ρ = ρ₀ − Tcoef (T − T₀) with ρ₀ = 1024 (EKBBL: 1.22 kg m⁻³), Tcoef = 0.2 kg m⁻³ K⁻¹, T₀ = 16 °C,
  no salinity (rho_eos.F:360-362, croco.in.* l. 43-47); c_p = 3985 J kg⁻¹ K⁻¹ (scalars.h:640);
* GLS k-ε, Canuto A, stratified (pyOCEAN ``gls_stratified=True``, the CROCO port physics/mixing/gls.py), floors
  ν ≥ 1e-5, κ_t ≥ 1e-6 m² s⁻¹ hard-wired in gls_mixing.F:150-153 (CROCO ignores Akv_bak/Akt_bak with GLS_MIXING:
  pyOCEAN ``kv = 0``);
* KPP (``closure='kpp'``, the CVMix port physics/mixing/kpp_cvmix.py with its defaults, no background) for
  kato_phillips, willis_deardorff and diurnal_cycle (spec decision 8).

Case table (croco_column_cases.md §1-4; ``CASES`` below):

  case                 N    H [m]  f [s⁻¹]  T(z) [°C], z ≤ 0   surface                     ∂v/∂t forcing      Zob   h
  kato_phillips        100  50     0      16 + N₀²/(gα) z      u*² = 1e-4 (x)               —                  0*    30
  willis_deardorff     50   50     0      22 + 0.1 z           Q = −100 W m⁻²               —                  0*    72
  diurnal_cycle        150  150    0      22 + 0.01 z          Q_sw(t)†, Q_ns = −75 W m⁻²   —                  0*    192
  forced_ekbbl         40   1500   1e-4   20                   —                            + f U_G, 10 m/s    0.1   24
  forced_dbleek        25   25     1e-4   20                   u*² = −6.25e-6 (x)           + f U_G, 1.5 mm/s  0.05  24
  forced_nonrotbbl     50   5      0      20                   —                            −g∂ζ/∂y            0.01  30
  forced_oscnonrotbbl  50   5      0      20                   —                            −g∂ζ/∂y cos ωt     0.01  60

N₀ = 0.01 s⁻¹; † Q_sw = 264.91 max(cos 2π(t/1 d − ½), 0) W m⁻²; g∂ζ/∂y = 1.962e-5 m s⁻², ω = 2π/12.4 h.
(*) Zob = 0 and Cdb_max = 0: no bed stress; the GLS bed condition uses Zobmin = 1e-4 m (pyOCEAN ``z0=None``: free
slip and the GLS default z0_bed = 1e-4).  α = Tcoef/ρ₀.  With Zob > 0 CROCO's bed stress is the first-layer law of
the wall C_d = [κ/ln((z_r − z_w)/Zob)]² (Cdb_max = 1e7), pyOCEAN ``z0`` (same formula; in ALE ``hbbl`` = ½ Δz so the
bed layer alone carries it, as in σ).

Two CROCO forcings have no pyOCEAN step argument and are added by the harness, without changing the model:

* the body force of the FORCED_* cases (prsgrd.F:63-76, 455-462: rv gets on_v·f·U_G, or −gdzetady[·cos ωt], i.e.
  ∂v/∂t += F): appended to the explicit momentum terms as a named term "body_force" (``Momentum.terms`` order: last),
  so it goes through the AB3 extrapolation like CROCO's pressure-gradient term; the wrapper is per model instance;
* CROCO's small bottom heat flux of the three stratified cases (analytical.F:133-136; ``btflx`` < 0 cools the bed
  cell, step3d_t.F:567, 617-618): ΔT_b = Δt btflx / h_b after each step, into the lowest layer thicker than the GLS
  thin-layer floor ``H_MASSLESS`` = 1 mm (the bed layer itself in σ and z*).

``build`` checks that the body-force hook took (``_check_body_force``) and raises otherwise.

``build(name, coord, closure)`` returns (model, state, forcing); ``forcing.step(model, state)`` advances one step.
``coord`` is 'sigma' (Eulerian σ), 'zstar' (ALE z*, layers fixed in a periodic flat column) or 'hybrid' (ALE
HYCOM1 with the initial isopycnals as targets and the uniform σ grid as z* floor: in a mixed layer the layers keep
their z* thickness, at its base the interfaces whose densities were mixed away collapse — a moving, partly
massless grid).

Index order (k, j, i), k = 0 at the bed; interface arrays (N+1, …), 0 = bed.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.components.closure import H_MASSLESS
from core.components.momentum import sum_terms
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from physics.mixing.kpp_cvmix import KPPParams

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/turbulence_columns")
CROCO_RUNS = RUNS / "croco"

G = 9.81                     # [m s⁻²] CROCO g (scalars.h)
CP = 3985.0                  # [J kg⁻¹ K⁻¹] CROCO Cp (scalars.h:640)
TCOEF, T0 = 0.2, 16.0        # [kg m⁻³ K⁻¹], [°C] lin_EOS_cff (croco.in.* l. 46-47)
NXY, DX = 5, 2000.0          # 5 × 5 cells, Length_XI = Length_ETA = 10 km (param.h:78-93, ana_grid.F:297-300)
DT = 30.0                    # [s] (croco.in.* l. 4)
NWRT = 120                   # output every 120 steps = 1 h (croco.in.* l. 25)
GDZETADY = 2e-6 * G          # [m s⁻²] prsgrd.F:74
OMEGA_M2 = 2.0 * np.pi / (12.4 * 3600.0)   # [s⁻¹] prsgrd.F:75
# initial T of the FORCED_* cases: the default branch of ana_initial.F:1222 (not T0 = 16 as croco_column_cases.md
# §4 says; irrelevant for the dynamics, T is uniform)
T_UNIFORM = 20.0             # [°C]

NAMES = ("kato_phillips", "willis_deardorff", "diurnal_cycle", "forced_ekbbl", "forced_dbleek",
         "forced_nonrotbbl", "forced_oscnonrotbbl")
COORDS = ("sigma", "zstar", "hybrid")
STRATIFIED = ("kato_phillips", "willis_deardorff", "diurnal_cycle")
KPP_CASES = STRATIFIED


@dataclass(frozen=True)
class Case:
    """One CROCO SINGLE_COLUMN case (croco_column_cases.md; units in the field comments)."""
    name: str
    croco_key: str           # CROCO cpp key
    n: int                   # layers (param.h:78-93)
    depth: float             # [m] (ana_grid.F:119-137)
    f0: float                # [s⁻¹]
    rho0: float              # [kg m⁻³] (croco.in.* l. 44)
    dTdz: float              # [K m⁻¹] initial gradient (0: T = T0 uniform)
    T_surf: float            # [°C] initial T at z = 0
    ustar2_x: float          # [m² s⁻²] kinematic surface stress sustr (analytical.F:378-389)
    q_ns: float              # [W m⁻²] non-solar surface heat flux, + into the ocean (analytical.F:876-879)
    q_sw_peak: float         # [W m⁻²] peak of the half-rectified cosine shortwave (analytical.F:553-557)
    btflx: float             # [°C m s⁻¹] bottom kinematic heat flux, + into the water (analytical.F:133-136)
    body_v: str              # 'none' | 'geostrophic' (+f U_G) | 'pgrad' (−g∂ζ/∂y) | 'pgrad_osc' (· cos ωt)
    ug: float                # [m s⁻¹] U_G of prsgrd.F:63-72
    zob: float               # [m] bed roughness (croco.in.* l. 63); 0 = no bed stress
    hours: float             # run length (NTIMES × dt, croco.in.* l. 4)

    @property
    def dz(self) -> float:
        return self.depth / self.n

    @property
    def n_steps(self) -> int:
        return int(round(self.hours * 3600.0 / DT))

    @property
    def alpha(self) -> float:
        """Thermal expansion α = Tcoef/ρ₀ [K⁻¹]."""
        return TCOEF / self.rho0

    @property
    def n2_initial(self) -> float:
        """N₀² = g α ∂T/∂z [s⁻²]."""
        return G * self.alpha * self.dTdz


def _kato_dTdz() -> float:
    # T = 16 + (0.01)²/(g Tcoef/ρ₀) z  (ana_initial.F:590-599): N₀ = 0.01 s⁻¹
    return 0.01**2 / (G * TCOEF / 1024.0)


CASES: dict[str, Case] = {
    "kato_phillips": Case("kato_phillips", "KATO_PHILIPS", 100, 50.0, 0.0, 1024.0, _kato_dTdz(), 16.0,
                          0.01**2, 0.0, 0.0, -1e-5 * 0.01**2 / (G * TCOEF / 1024.0), "none", 0.0, 0.0, 30.0),
    "willis_deardorff": Case("willis_deardorff", "WILLIS_DEARDORFF", 50, 50.0, 0.0, 1024.0, 0.1, 22.0,
                             0.0, -100.0, 0.0, -1e-5 * 0.01, "none", 0.0, 0.0, 72.0),
    "diurnal_cycle": Case("diurnal_cycle", "DIURNAL_CYCLE", 150, 150.0, 0.0, 1024.0, 0.01, 22.0,
                          0.0, -75.0, 264.91, -1e-5 * 0.01, "none", 0.0, 0.0, 192.0),
    "forced_ekbbl": Case("forced_ekbbl", "FORCED_EKBBL", 40, 1500.0, 1e-4, 1.22, 0.0, T_UNIFORM,
                         0.0, 0.0, 0.0, 0.0, "geostrophic", 10.0, 0.1, 24.0),
    "forced_dbleek": Case("forced_dbleek", "FORCED_DBLEEK", 25, 25.0, 1e-4, 1024.0, 0.0, T_UNIFORM,
                          -(0.0025**2), 0.0, 0.0, 0.0, "geostrophic", 0.0015, 0.05, 24.0),
    "forced_nonrotbbl": Case("forced_nonrotbbl", "FORCED_NONROTBBL", 50, 5.0, 0.0, 1024.0, 0.0, T_UNIFORM,
                             0.0, 0.0, 0.0, 0.0, "pgrad", 0.0, 0.01, 30.0),
    "forced_oscnonrotbbl": Case("forced_oscnonrotbbl", "FORCED_OSCNONROTBBL", 50, 5.0, 0.0, 1024.0, 0.0, T_UNIFORM,
                                0.0, 0.0, 0.0, 0.0, "pgrad_osc", 0.0, 0.01, 60.0),
}


def shortwave(case: Case, t: float) -> float:
    """CROCO srflx [W m⁻²] at time t [s]: 264.91 max(cos(2π(t/86400 − ½)), 0) (analytical.F:553-557)."""
    return case.q_sw_peak * max(np.cos(2.0 * np.pi * (t / 86400.0 - 0.5)), 0.0)


def body_force(case: Case, t: float) -> float:
    """∂v/∂t forcing [m s⁻²] of prsgrd.F:455-462 at time t [s]."""
    if case.body_v == "geostrophic":
        return case.f0 * case.ug
    if case.body_v == "pgrad":
        return -GDZETADY
    if case.body_v == "pgrad_osc":
        return -GDZETADY * np.cos(OMEGA_M2 * t)
    return 0.0


def initial_T(case: Case) -> np.ndarray:
    """Initial T at the uniform layer centres (n,) [°C], k = 0 at the bed (ana_initial.F:568-599)."""
    z = -case.depth + (np.arange(case.n) + 0.5) * case.dz          # centre height, z ≤ 0
    return case.T_surf + case.dTdz * z


def _hybrid_targets(case: Case) -> np.ndarray:
    """HYCOM1 interface target densities (n+1,), surface first: the initial ρ at the uniform interfaces."""
    zw = -np.arange(case.n + 1) * case.dz                          # surface first, z ≤ 0
    T = case.T_surf + case.dTdz * zw
    return case.rho0 - TCOEF * (T - T0)


@dataclass
class ColumnForcing:
    """Time-dependent surface forcing of a case; ``step`` advances the model one Δt (body force via the wrapper)."""
    case: Case
    shape: tuple[int, int]
    tau_x: np.ndarray = field(init=False)
    tau_y: np.ndarray = field(init=False)

    def __post_init__(self) -> None:
        ny, nx = self.shape
        self.tau_x = np.full((ny, nx + 1), self.case.rho0 * self.case.ustar2_x)   # τ = ρ₀ u*² [N m⁻²]
        self.tau_y = np.zeros((ny + 1, nx))

    def heat(self, t: float) -> dict | None:
        c = self.case
        if c.q_ns == 0.0 and c.q_sw_peak == 0.0:
            return None
        out = {"Q_ns": np.full(self.shape, c.q_ns)}
        if c.q_sw_peak:
            out["Q_sw"] = np.full(self.shape, shortwave(c, t))
        return out

    def step(self, model: SigmaBaroclinic, st):
        new = model.step(st, self.tau_x, self.tau_y, heat=self.heat(st.time))
        if self.case.btflx:
            # into the lowest layer thicker than the GLS thin-layer floor H_MASSLESS (a vanished hybrid bed layer
            # would otherwise take Δt btflx / h₀ → ∞); conservative: the whole flux enters one layer per column
            h = new.h if new.h is not None else model.layer_volumes(new.eta) / model.area[None]
            kb = np.argmax(h > H_MASSLESS, axis=0)                     # (ny, nx), 0 when the bed layer is thick
            hb = np.take_along_axis(h, kb[None], axis=0)[0]
            T = new.T.copy()
            np.put_along_axis(T, kb[None], np.take_along_axis(T, kb[None], axis=0) + model.dt * self.case.btflx
                              / hb[None], axis=0)
            new.T = T
        return new


def _install_body_force(model: SigmaBaroclinic, case: Case) -> None:
    """Append ∂v/∂t = F(t) as the last named explicit momentum term (``core.components.momentum``), this instance
    only.  The momentum is horizontally uniform, so the wet-face mask of ``sum_terms`` is all True."""
    mom = model._momentum
    inner = mom.terms

    def terms(st, *a, **kw):
        out = inner(st, *a, **kw)
        tv = np.full(st.v.shape, body_force(case, st.time))
        out.append(("body_force", np.zeros(st.u.shape), tv, "+"))
        return out
    mom.terms = terms


def _check_body_force(model: SigmaBaroclinic, st, case: Case) -> None:
    """Fail loudly if the hook of ``_install_body_force`` does not reach the explicit momentum the driver sums
    (it relies on the private ``model._momentum.terms``, looked up per call by ``model._momentum_terms``): the last
    term must be "body_force" and adding it must change the v tendency by F(t) on the wet v-faces."""
    terms, wet_u, wet_v = model._momentum_terms(st)
    if not terms or terms[-1][0] != "body_force":
        raise RuntimeError("the FORCED_* body force did not reach Momentum.terms (private hook changed?)")
    _, gv = sum_terms(terms, wet_u, wet_v)
    _, gv0 = sum_terms(terms[:-1], wet_u, wet_v)
    F = body_force(case, st.time)
    if F == 0.0 or not np.allclose((gv - gv0)[:, wet_v], F, rtol=1e-12, atol=0.0):
        raise RuntimeError(f"the FORCED_* body force does not change the v tendency by F = {F!r} m s⁻²")


def build(name: str, coord: str = "sigma", closure: str = "gls", *, gls_production: str = "croco",
          kpp_params: KPPParams | None = None, nxy: int = NXY, **overrides):
    """
    (model, state, forcing) of case ``name`` in coordinate ``coord`` ('sigma' | 'zstar' | 'hybrid') with
    ``closure`` 'gls' or 'kpp'.  ``overrides`` go to ``SigmaBaroclinic``.
    """
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; use {NAMES}")
    if coord not in COORDS:
        raise ValueError(f"unknown coordinate {coord!r}; use {COORDS}")
    if closure not in ("gls", "kpp"):
        raise ValueError("closure must be 'gls' or 'kpp'")
    c = CASES[name]
    hg = create_cartesian_grid(nx=nxy, ny=nxy, dx=DX, dy=DX, f0=c.f0)
    kw = dict(rho0=c.rho0, t_coef=TCOEF, t_ref=T0, closure=closure, theta=0.6, d_min=0.05, g=G, cp=CP,
              periodic=("x", "y"), z0=(c.zob or None))
    if closure == "gls":
        kw.update(gls_stratified=True, gls_production=gls_production)
    else:
        kw.update(kpp_params=kpp_params or KPPParams())
    if coord != "sigma":
        params = dict(dz_top=np.full(c.n, c.dz))
        if coord == "hybrid":
            params["rho_target"] = _hybrid_targets(c)
        kw.update(vertical_coordinate="ale", regrid=coord, pgf="fv", regrid_params=params, hbbl=0.5 * c.dz)
    kw.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # pre-flight notes (e.g. kv with kpp) are expected here
        model = SigmaBaroclinic(hg, np.full((nxy, nxy), c.depth), dt=DT, n_layers=c.n, **kw)
    if c.body_v != "none":
        _install_body_force(model, c)
    T = initial_T(c)[:, None, None] * np.ones((c.n, nxy, nxy))
    st = model.initial_state(np.zeros((nxy, nxy)), T)
    if c.body_v != "none":
        _check_body_force(model, st, c)
    return model, st, ColumnForcing(c, (nxy, nxy))
