"""
Case definitions of the TEOS-10 validation (Task 9 of the plan docs/superpowers/plans/2026-10-31-teos10-3d.md; spec
docs/superpowers/specs/2026-10-31-teos10-3d-design.md, criteria 4, 5, 8, 9 and 10).  One definition per case; the
runner (``run_pyocean.py``), the comparison (``compare.py``) and ``tests/test_teos10_validation.py`` read it.

Cases
-----
* ``rest_case(kind, eos, coord)``: the WOA23 profile at ~40 °N 70 °W (``woa_gulfstream_profile.py``), at rest and
  horizontally uniform, over steep topography, 10 days (criterion 4).  ``kind`` 'seamount' is Beckmann & Haidvogel
  (1993, §2c) as ``validation/seamount`` and the Ilicak benchmark (64 × 48 × 20, Δx 6 km, Δt 600 s, f 1e-4 s⁻¹);
  'bayiw_step' is a 2-D section across the BayIW bay-mouth step 600 → 164 m (``tests/test_sigma_ale.py``
  ``_step_section``; 80 × 1 × 50, Δx 1 km, Δt 120 s, f 0).  ``coord`` 'zstar' (uniform z* levels of H_max/N) or
  'hybrid' (HYCOM1 rule, z* floors ``hybrid_floors`` from 2 m at the top to the bed, isopycnal interior with
  ``rho_target`` from the profile; with a nonlinear EOS the targets are potential densities at p_ref = 2e7 Pa,
  spec goal D).  The exact
  pressure gradient is zero: every velocity is numerical error.
* ``cabbeling_case(eos)``: a lock exchange of two waters of equal density at p = 0 (criterion 5a): A = (5 °C, 34.0)
  and B = (15 °C, S_B) with S_B found by Newton so that ρ(B, 0) = ρ(A, 0) with the case's EOS.  2-D, 64 × 1 × 20,
  Δx 500 m, H 200 m, z*, K_h 10 m² s⁻¹ along the layers (the only thing that mixes the two waters at rest).
* ``thermobaric_case(eos, interface)``: the two-layer column of Task 6 (``validation/mom6_column/eos_columns.py``
  ``THERMOBARIC_COLD`` = (0 °C, 34.5) over ``THERMOBARIC_WARM`` = (4 °C, 34.95), 200 m of warm water on the bed)
  with the interface at 100 m or 4000 m, 5 × 5 doubly periodic, z*, stratified GLS k-ε (criterion 5b).

Equivalent linear EOS (criterion 4): ρ_lin = ρ₀ − α (θ − θ₀) + β (S_P − S₀), with α, β and the intercept fitted by
least squares to ρ_Wright(θ, S_P, p = 0) on the profile levels (``linear_equivalent``; the code is in the README),
ρ₀ = 1035 kg m⁻³ as the nonlinear EOS, θ₀ the profile mean and S₀ chosen so that the intercept is ρ₀ exactly.

Conventions: the data are θ and S_P (``ts_kind="theta_sp"``, what WOA gives after the in-situ → θ conversion); a
``teos10`` model converts them once in ``initial_state`` (S_A = (35.16504/35) S_P, Θ = CT_from_pt; spec goal G).  The
cabbeling and thermobaric waters are defined in the EOS's own variables (θ/S_P for linear and wright_full, Θ/S_A for
teos10) and passed with ``ts_kind`` equal to the EOS convention, so the same numbers reach both EOS (as Task 6).
Index order (k, j, i), k = 0 at the bed; z [m] ≤ 0 up from the rest surface.
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.components.eos import make_eos
from core.dynamics.sigma_baroclinic import SigmaBaroclinic, SigmaState
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import PyOceanConfigWarning
from physics.eos import wright_full

ROOT = Path(__file__).resolve().parents[2]
PROFILE_FILE = ROOT / "tests" / "data" / "woa23_gulfstream_profile.npz"
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/teos10")

RHO0, G, P_REF = 1035.0, 9.81, 2e7          # [kg m⁻³], [m s⁻²], [Pa] (NWA12 RHO_0, spec goal D)
EOS_NAMES = ("linear", "wright_full", "teos10")
DAY = 86400.0

# Beckmann & Haidvogel (1993) seamount (validation/seamount/run_seamount.py)
SM_H0, SM_HS, SM_LS, SM_F0 = 4500.0, 4050.0, 25e3, 1e-4
# cabbeling waters (θ or Θ [°C], S_P or S_A)
CAB_A = (5.0, 34.0)
CAB_TB = 15.0
# thermobaric waters of Task 6 (validation/mom6_column/eos_columns.py:279-280)
THERMO_COLD, THERMO_WARM = (0.0, 34.5), (4.0, 34.95)
THERMO_WARM_THICKNESS = 200.0               # [m] of warm water on the bed


@dataclass
class Case:
    """A built case: the model, its initial state, the number of steps and what the metrics need."""
    name: str
    model: SigmaBaroclinic
    state: SigmaState
    n_steps: int
    meta: dict = field(default_factory=dict)


# ---------------------------------------------------------------- profile
def load_profile() -> dict:
    """The WOA23 profile (depth [m] positive down, θ [°C], S_P [PSU]), surface first."""
    with np.load(PROFILE_FILE) as d:
        return {k: d[k].astype(float) for k in d.files}


def profile_at(z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """θ [°C] and S_P [PSU] of the profile at heights z [m] ≤ 0 (linear in depth; below the deepest level the
    deepest values, above the surface level the surface values)."""
    p = load_profile()
    d = -np.asarray(z, float)
    return np.interp(d, p["depth"], p["theta"]), np.interp(d, p["depth"], p["SP"])


def linear_equivalent() -> dict:
    """
    Linear EOS fitted to the profile: least squares of ρ_Wright(θ, S_P, 0) [kg m⁻³] on the profile levels against
    c − α (θ − θ₀) + β (S_P − S₀) with θ₀, S₀ the profile means; then S₀ is moved so that the intercept is
    ρ₀ = 1035 (α, β unchanged).  Returns the ``SigmaBaroclinic`` keywords rho0, t_coef = α, t_ref = θ₀,
    s_coef = β, s_ref, and the rms misfit [kg m⁻³].
    """
    p = load_profile()
    th, sp = p["theta"], p["SP"]
    rho = wright_full.density(th, sp, np.zeros_like(th))
    t0, s0 = th.mean(), sp.mean()
    A = np.column_stack([np.ones_like(th), -(th - t0), sp - s0])
    (c, alpha, beta), *_ = np.linalg.lstsq(A, rho, rcond=None)
    rms = float(np.sqrt(np.mean((A @ np.array([c, alpha, beta]) - rho) ** 2)))
    s_ref = s0 - (c - RHO0) / beta
    return dict(rho0=RHO0, t_coef=float(alpha), t_ref=float(t0), s_coef=float(beta), s_ref=float(s_ref), rms=rms)


def _eos_kwargs(eos: str) -> dict:
    if eos == "linear":
        lin = linear_equivalent()
        return dict(eos="linear", **{k: lin[k] for k in ("rho0", "t_coef", "t_ref", "s_coef", "s_ref")})
    if eos in ("wright_full", "teos10"):
        return dict(eos=eos, rho0=RHO0)
    raise ValueError(f"unknown eos {eos!r}; expected one of {EOS_NAMES}")


def _regrid_density(eos: str, theta: np.ndarray, sp: np.ndarray) -> np.ndarray:
    """The density the hybrid regrid uses for data θ, S_P: linear ρ, or potential density at p_ref (σ₂ + 1000)
    of the EOS in its own convention (spec goal D)."""
    kw = _eos_kwargs(eos)
    if eos == "linear":
        e = make_eos("linear", kw["rho0"], t_coef=kw["t_coef"], t_ref=kw["t_ref"], s_coef=kw["s_coef"],
                     s_ref=kw["s_ref"])
        return e.density(theta, sp)
    from physics.eos.convert import to_model_ts
    e = make_eos(eos, RHO0)
    T, S = to_model_ts(theta, sp, e.ts_kind, "theta_sp")
    return e.potential_density(T, S, P_REF)


def hybrid_targets(eos: str, n: int, zmax: float) -> np.ndarray:
    """
    n+1 interface target densities [kg m⁻³], surface first, increasing.  Nonlinear EOS: σ₂ + 1000 evenly spaced,
    the lightest and the densest profile water (0 … zmax) at the centres of the first and last layers (as
    ``validation/ilicak_benchmark/cases.targets``, MOM6 COORD_CONFIG 'layer_ref').  Linear EOS: the linear density
    of the profile at the depths where the ``wright_full`` targets sit, so that the equivalent linear run starts
    on the same hybrid layers as the control and the comparison of criterion 4 isolates the EOS (outside the
    profile range: linear extrapolation in density).
    """
    z = -np.linspace(0.0, zmax, 400)
    th, sp = profile_at(z)
    rho = _regrid_density("wright_full" if eos == "linear" else eos, th, sp)
    lo, hi = float(rho.min()), float(rho.max())
    d = (hi - lo) / (n - 1)
    tgt = lo - 0.5 * d + d * np.arange(n + 1)
    if eos != "linear":
        return tgt
    rho_lin = _regrid_density("linear", th, sp)
    keep = np.concatenate([[True], np.diff(rho) > 0])        # strictly increasing abscissa (neutral abyss)
    r, rl = rho[keep], rho_lin[keep]
    out = np.interp(tgt, r, rl)
    lo_s, hi_s = (rl[1] - rl[0]) / (r[1] - r[0]), (rl[-1] - rl[-2]) / (r[-1] - r[-2])
    out = np.where(tgt < r[0], rl[0] + lo_s * (tgt - r[0]), out)
    return np.where(tgt > r[-1], rl[-1] + hi_s * (tgt - r[-1]), out)


def hybrid_floors(n: int, hmax: float, first: float = 2.0) -> np.ndarray:
    """z* floor thicknesses dz_top (n,) [m] of the hybrid rule: geometric from ``first`` at the top, Σ = hmax (the
    floor reaches the bed, as OM4's hycom1_75_800m and ``validation/bayiw_baroclinic/bayiw.hybrid_dz_top``: a floor
    that stops short leaves the homogeneous deep water in one massless-target slab).  The ratio r solves
    first (rⁿ − 1)/(r − 1) = hmax by bisection."""
    lo, hi = 1.0 + 1e-12, 4.0
    for _ in range(200):
        r = 0.5 * (lo + hi)
        if first * (r ** n - 1.0) / (r - 1.0) > hmax:
            hi = r
        else:
            lo = r
    dz = first * r ** np.arange(n)
    return dz * (hmax / dz.sum())


# ---------------------------------------------------------------- rest over steep topography (criterion 4)
def _rest_geometry(kind: str, fast: bool) -> dict:
    if kind == "seamount":
        dx, nz = (12e3, 10) if fast else (6e3, 20)
        nx, ny = int(round(384e3 / dx)), int(round(288e3 / dx))
        x = (np.arange(nx) + 0.5) * dx - nx * dx / 2
        y = (np.arange(ny) + 0.5) * dx - ny * dx / 2
        H = SM_H0 - SM_HS * np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / SM_LS ** 2)
        return dict(nx=nx, ny=ny, dx=dx, nz=nz, H=H, f0=SM_F0, dt=600.0)
    if kind == "bayiw_step":
        nx, nz = (40, 20) if fast else (80, 50)
        H = np.where(np.arange(nx) < nx // 2, 600.0, 164.0)[None, :]
        return dict(nx=nx, ny=1, dx=1000.0, nz=nz, H=H, f0=0.0, dt=120.0)
    raise ValueError(f"unknown rest case {kind!r}; expected 'seamount' or 'bayiw_step'")


def rest_case(kind: str, eos: str, coord: str, *, rho_ref: str | None = "auto", fast: bool = False,
              days: float | None = None) -> Case:
    """
    The WOA profile at rest over ``kind`` ('seamount' | 'bayiw_step') with ``eos`` and ``coord`` ('zstar' |
    'hybrid'), 10 days (``fast``: half the resolution and 1 day).  ``rho_ref`` 'auto' (pyOCEAN's reference
    profile, the default of real runs) or None (MOM6's method: constant RHO_PGF_REF = ρ₀ subtracted).
    Constant ν_v = 1e-4 m² s⁻¹, no horizontal viscosity or diffusion (as the seamount of the Ilicak benchmark),
    no bed drag.
    """
    geo = _rest_geometry(kind, fast)
    nx, ny, nz, H = geo["nx"], geo["ny"], geo["nz"], geo["H"]
    hmax = float(H.max())
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=geo["dx"], dy=geo["dx"], f0=geo["f0"])
    if coord == "zstar":
        params = dict(dz_top=np.full(nz, hmax / nz))
    elif coord == "hybrid":
        params = dict(dz_top=hybrid_floors(nz, hmax), rho_target=hybrid_targets(eos, nz, hmax))
    else:
        raise ValueError(f"unknown coord {coord!r}; expected 'zstar' or 'hybrid'")
    kw = dict(z0=None, closure="constant", nu_const=1e-4, ah=0.0, g=G, salinity=True, rho_ref=rho_ref,
              vertical_coordinate="ale", pgf="fv", regrid=coord, regrid_params=params, **_eos_kwargs(eos))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        m = SigmaBaroclinic(hg, H, dt=geo["dt"], n_layers=nz, **kw)
    eta = np.zeros((ny, nx))
    if coord == "hybrid":                      # T, S on z-levels as the source grid (MOM6 Z_INIT_ALE_REMAPPING)
        nsrc = 4 * nz
        zf = np.linspace(0.0, hmax, nsrc + 1)[:, None, None]
        h_src = (np.minimum(zf[1:], H[None]) - np.minimum(zf[:-1], H[None]))[::-1]          # bed first
        zc = -0.5 * (zf[1:] + zf[:-1])[::-1] * np.ones((1, ny, nx))
        th, sp = profile_at(zc)
        st = m.initial_state(eta, th, S=sp, h_src=h_src, ts_kind="theta_sp")
    else:
        th, sp = profile_at(m.layer_heights(eta))
        st = m.initial_state(eta, th, S=sp, ts_kind="theta_sp")
    days = (1.0 if fast else 10.0) if days is None else days
    return Case(f"rest_{kind}_{eos}_{coord}_{'auto' if rho_ref else 'none'}", m, st,
                int(round(days * DAY / geo["dt"])), dict(kind=kind, eos=eos, coord=coord, rho_ref=rho_ref))


# ---------------------------------------------------------------- cabbeling (criterion 5a)
def cabbeling_waters(eos: str) -> tuple[tuple[float, float], tuple[float, float], float]:
    """
    (A, B, ρ_src): A = ``CAB_A``, B = (``CAB_TB``, S_B) in the EOS's own variables with ρ(B, p = 0) = ρ(A, 0)
    [kg m⁻³].  Nonlinear EOS: S_B by Newton on f(S) = ρ(T_B, S, 0) − ρ(A) with ∂ρ/∂S of the EOS.  Linear: the
    linear EOS of ``cabbeling_linear`` has S_B fixed by its β (ρ(B) = ρ(A) by construction).
    """
    (ta, sa), tb = CAB_A, CAB_TB
    if eos == "linear":
        lin = cabbeling_linear()
        sb = sa + lin["t_coef"] * (tb - ta) / lin["s_coef"]
        return (ta, sa), (tb, sb), lin["rho0"]
    e = make_eos(eos, RHO0)
    rho_a = float(e.density(np.array([ta]), np.array([sa]), np.zeros(1))[0])
    s = sa + 0.2 * (tb - ta)                                # first guess: ~0.2 PSU per °C
    for _ in range(50):
        r = float(e.density(np.array([tb]), np.array([s]), np.zeros(1))[0]) - rho_a
        _, drds = e.density_derivs(np.array([tb]), np.array([s]), np.zeros(1))
        ds = -r / float(drds[0])
        s += ds
        if abs(ds) < 1e-14:
            break
    return (ta, sa), (tb, s), rho_a


def cabbeling_linear() -> dict:
    """The linear control of the cabbeling case: α = −∂ρ/∂T of Wright at the mean of A and the Wright B (p = 0),
    β = α ΔT/ΔS so that the two waters have the same density exactly, intercept ρ₀ = 1035 at A."""
    (ta, sa), (tb, sb), _ = cabbeling_waters("wright_full")
    drdt, _ = wright_full.density_derivs(np.array([0.5 * (ta + tb)]), np.array([0.5 * (sa + sb)]), np.zeros(1))
    alpha = -float(drdt[0])
    beta = alpha * (tb - ta) / (sb - sa)
    return dict(rho0=RHO0, t_coef=alpha, t_ref=ta, s_coef=beta, s_ref=sa)


def surface_density(eos: str, T: np.ndarray, S: np.ndarray) -> np.ndarray:
    """ρ(T, S, p = 0) [kg m⁻³] with the case's EOS (T, S in its own variables)."""
    if eos == "linear":
        lin = cabbeling_linear()
        return make_eos("linear", lin["rho0"], t_coef=lin["t_coef"], t_ref=lin["t_ref"], s_coef=lin["s_coef"],
                        s_ref=lin["s_ref"]).density(T, S)
    return make_eos(eos, RHO0).density(T, S, np.zeros_like(T))


def cabbeling_case(eos: str, *, fast: bool = False, days: float | None = None) -> Case:
    """Lock exchange of A (west half) and B (east half) at rest, equal ρ at p = 0; 10 days (``fast``: 1 day)."""
    nx, nz, dx, depth, dt = 64, 20, 500.0, 200.0, 60.0
    if fast:
        nx, nz = 32, 10
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    H = np.full((1, nx), depth)
    if eos == "linear":
        ekw = dict(eos="linear", **cabbeling_linear())
    else:
        ekw = dict(eos=eos, rho0=RHO0)
    kw = dict(z0=None, closure="constant", nu_const=1e-4, kv=1e-5, ah=1.0, kh=10.0, g=G, salinity=True,
              rho_ref=None, vertical_coordinate="ale", pgf="fv", regrid="zstar",
              regrid_params=dict(dz_top=np.full(nz, depth / nz)), **ekw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        m = SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, **kw)
    (ta, sa), (tb, sb), rho_src = cabbeling_waters(eos)
    west = (np.arange(nx) < nx // 2)[None, None, :]
    T = np.where(west, ta, tb) * np.ones((nz, 1, nx))
    S = np.where(west, sa, sb) * np.ones((nz, 1, nx))
    st = m.initial_state(np.zeros((1, nx)), T, S=S, ts_kind=m._eos.ts_kind)
    days = (1.0 if fast else 10.0) if days is None else days
    return Case(f"cabbeling_{eos}", m, st, int(round(days * DAY / dt)),
                dict(eos=eos, A=(ta, sa), B=(tb, sb), rho_src=rho_src))


def denser_volume(case: Case, st: SigmaState, tol: float = 1e-6) -> tuple[float, float]:
    """(volume [m³] of water denser than both sources at p = 0 by more than ``tol`` [kg m⁻³], the mean height [m]
    of the excess density Σ V (ρ − ρ_src) z / Σ V (ρ − ρ_src) over that water (NaN without it)).  The sources
    have the same ρ(·, 0) (to round-off); the excess-weighted height follows the densest water as it sinks."""
    m, eos = case.model, case.meta["eos"]
    (ta, sa), (tb, sb) = case.meta["A"], case.meta["B"]
    rho_max = max(surface_density(eos, np.array([ta]), np.array([sa]))[0],
                  surface_density(eos, np.array([tb]), np.array([sb]))[0])
    rho = surface_density(eos, st.T, st.S)
    V = m.layer_volumes(st.eta, st.h)
    dense = (rho > rho_max + tol) & (V > 0)
    vol = float(V[dense].sum())
    if vol == 0.0:
        return 0.0, float("nan")
    z = m.layer_heights(st.eta, st.h)
    w = (V * (rho - rho_max))[dense]
    return vol, float((w * z[dense]).sum() / w.sum())


# ---------------------------------------------------------------- thermobaricity (criterion 5b)
def thermobaric_case(eos: str, interface: float, *, fast: bool = False, days: float | None = None) -> Case:
    """
    Cold, fresh water over 200 m of warm, salty water with the interface at ``interface`` [m] (100 or 4000), at
    rest in a 5 × 5 doubly periodic box (Δx 2 km), z* with 10 m layers (100 m: 30 layers) or 50 m layers (4000 m:
    84 layers), stratified GLS k-ε (the closure that mixes a statically unstable column), no forcing; 10 days
    (``fast``: 1 day, layers twice as thick).  The waters are the same numbers in both EOS (their own variables);
    the linear control is ``thermobaric_linear``.
    """
    depth = interface + THERMO_WARM_THICKNESS
    dz = (10.0 if interface <= 500.0 else 50.0) * (2.0 if fast else 1.0)
    nz = int(round(depth / dz))
    nxy, dt = 5, 60.0
    hg = create_cartesian_grid(nx=nxy, ny=nxy, dx=2000.0, dy=2000.0, f0=1e-4)
    ekw = dict(eos="linear", **thermobaric_linear()) if eos == "linear" else dict(eos=eos, rho0=RHO0)
    kw = dict(z0=None, closure="gls", gls_stratified=True, ah=0.0, g=G, salinity=True, periodic=("x", "y"),
              vertical_coordinate="ale", pgf="fv", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, dz)),
              **ekw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        m = SigmaBaroclinic(hg, np.full((nxy, nxy), depth), dt=dt, n_layers=nz, **kw)
    zc = m.layer_heights(np.zeros((nxy, nxy)))
    warm = zc < -interface
    T = np.where(warm, THERMO_WARM[0], THERMO_COLD[0])
    S = np.where(warm, THERMO_WARM[1], THERMO_COLD[1])
    st = m.initial_state(np.zeros((nxy, nxy)), T, S=S, ts_kind=m._eos.ts_kind)
    days = (1.0 if fast else 10.0) if days is None else days
    return Case(f"thermobaric_{eos}_{int(interface)}", m, st, int(round(days * DAY / dt)),
                dict(eos=eos, interface=float(interface)))


def thermobaric_linear() -> dict:
    """The linear control of the thermobaric case: α = −∂ρ/∂θ and β = ∂ρ/∂S_P of Wright at the mean of the two
    waters and p = 0 (the pair is stable there, so a linear EOS keeps it stable at every depth); ρ₀ = 1035 at the
    cold water."""
    t, s = 0.5 * (THERMO_COLD[0] + THERMO_WARM[0]), 0.5 * (THERMO_COLD[1] + THERMO_WARM[1])
    drdt, drds = wright_full.density_derivs(np.array([t]), np.array([s]), np.zeros(1))
    return dict(rho0=RHO0, t_coef=-float(drdt[0]), t_ref=THERMO_COLD[0], s_coef=float(drds[0]),
                s_ref=THERMO_COLD[1])


def warm_rise(case: Case, st: SigmaState) -> float:
    """Fraction of the bed water's heat anomaly found above the initial interface [-]: Σ V (T − T_cold) over
    z > −interface divided by the total Σ V (T − T_cold) (heat content is conserved, closed periodic box).
    0 at the start; it grows only if warm water rises across the interface (overturn)."""
    m = case.model
    V = m.layer_volumes(st.eta, st.h)
    z = m.layer_heights(st.eta, st.h)
    a = V * (st.T - THERMO_COLD[0])
    return float(a[z > -case.meta["interface"]].sum() / a.sum())
