"""
MOM6's CVMix single-column KPP tests (MOM6-examples/ocean_only/CVmix_SCM_tests) as pyOCEAN columns, the run of
both models and the comparison (N3 task 9; spec docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md,
criterion 2 and decision 9).

    validation/turbulence_columns/mom6/run_mom6.sh wind_only             # MOM6 reference (≈ 8 s per case)
    python -m validation.turbulence_columns.mom6_scm run wind_only zstar # pyOCEAN (or --all)
    python -m validation.turbulence_columns.mom6_scm compare             # tables of every case on disk

Every value below is MOM6's (``common_KPP/MOM_input`` and the case's ``KPP/MOM_override``, read in
``docs/design/n2c_refs/kpp_mom6_cvmix.md`` §5 and checked against the ``MOM_parameter_doc.all`` that the local
MOM6 binary writes); nothing is tuned (CLAUDE.md, Validation Strategy).

Common configuration:

* flat 400 m, 400 uniform layers of 1 m (NK = 400, ALE z*, THICKNESS_CONFIG coord), 2 × 2 cells reentrant
  in x and y (pyOCEAN ``periodic=("x", "y")``), Δt = DT = DT_THERM = 1200 s, 5 days (360 steps);
* f = F_0 = 1e-4 s⁻¹ (betaplane, β = 0), g = G_EARTH = 9.80616 m s⁻², ρ₀ = RHO_0 = 1000 kg m⁻³;
* linear EOS, dρ/dT = −0.255 kg m⁻³ K⁻¹ (pyOCEAN ``t_coef`` = 0.255), dρ/dS = 0.764 with S = 35 uniform and no
  salt flux: S never changes, so pyOCEAN runs without salinity (the density differences that KPP sees are the
  same);
* c_p = C_P = 3992.10322329649 J kg⁻¹ K⁻¹: MOM6's value in this test (common_KPP/MOM_input), used as pyOCEAN
  ``cp`` so that Q → F_T = Q/(ρ₀ c_p) and B_f = (g/ρ₀) 0.255 F_T are the same numbers (B_f = ∓6.2638e-8 m² s⁻³
  at ∓100 W m⁻², = MOM6's KPP_buoyFlux);
* initial T(z) = max(4, 15 + 0.01 z) (SCM_CVMix_tests.F90 with SCM_L2_TEMP = 15, SCM_L2_DTDZ = 0.01 K m⁻¹,
  SCM_TEMP_MLD = 0) at the layer centres — 11.005 °C at the bottom layer, N² = 2.50e-5 s⁻²; skin_warming_wind
  overrides SCM_L2_DTDZ = 0 (isothermal 15 °C; confirmed in its MOM_parameter_doc.all);
* KPP: CVMix defaults (pyOCEAN ``KPPParams``) with INTERP_TYPE = "cubic" (common_KPP/MOM_input);
  no interior mixing (KD = KV = 0: pyOCEAN ``kv = 0``), no bed drag (CDRAG = 0: ``z0 = None``), no shortwave.

Cases (MOM_override; τ_x = CONST_WIND_TAUX, gust = GUST_CONST, Q = SENSIBLE_HEAT_FLUX, + into the ocean):

  case               τ_x [Pa]  gust [Pa]  Q [W m⁻²]  T(z)           u* = ((|τ| + gust)/ρ₀)^½
  wind_only          0.1       0          0          15 + 0.01 z    0.01
  cooling_only       0         0          −100       15 + 0.01 z    0
  mech_only          0         0.05       0          15 + 0.01 z    7.07e-3
  skin_warming_wind  0.1       0          +100       15             0.01

u* in MOM6 is sqrt((|τ| + GUST_CONST)/ρ₀) (solo_driver/MOM_surface_forcing.F90:451-462); the gust enters only u*,
not the momentum.  pyOCEAN's step has no gust argument, so the harness adds gust/ρ₀ to the surface u*² that the
driver passes to ``KPPClosure.update`` (``_install_gust``, this instance only; checked by ``check_gust`` after the
first step), as task 8 added the body force — the model is not changed.

Diagnostics, defined the same way for both models (README, section "Columnas MOM6/CVMix"):

* OBL depth h: KPP's boundary-layer depth of each step (MOM6 ``KPP_OBLdepth``, pyOCEAN ``KPPClosure.obl_depth``),
  averaged over the 2-hour output windows as MOM6's diag_table does (``time_avg = .true.``; 6 steps).
* Mixed-layer depth MLD₀₀₃: MOM6 ``diagnoseMLDbyDensityDifference`` (MOM_diagnose_MLD.F90:34-247, HREF_FOR_MLD = 0):
  the depth, interpolated linearly between layer centres, where ρ − ρ(top layer) first reaches 0.03 kg m⁻³; the
  deepest centre if it never does.  Computed by ``mld_003`` from the 2-hourly T snapshots of both models
  (MOM6 prog.nc; ``mld_003`` reproduces MOM6's own window-averaged MLD_003 in the comparison table).

Index order (k, j, i), k = 0 at the bed; interface arrays (N+1, …).
"""
from __future__ import annotations

import argparse
import time
import warnings
from dataclasses import dataclass

import numpy as np

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from physics.mixing.kpp_cvmix import KPPParams
from validation.turbulence_columns.cases import RUNS

MOM6_RUNS = RUNS / "mom6"
PY_OUT = RUNS / "pyocean_mom6"

NK, DEPTH = 400, 400.0        # NK, MAXIMUM_DEPTH [m]
NXY = 2                       # NIGLOBAL = NJGLOBAL = 2, reentrant
DX = 2.0 * 111.0e3 / NXY      # [m] LENLON = LENLAT = 2° Cartesian (irrelevant in a horizontally uniform column)
DT = 1200.0                   # [s] DT = DT_THERM
DAYS = 5.0                    # DAYMAX
NOUT = 6                      # output every 2 h = 6 steps
F0 = 1.0e-4                   # [s⁻¹] F_0
G = 9.80616                   # [m s⁻²] G_EARTH
RHO0 = 1000.0                 # [kg m⁻³] RHO_0
DRHO_DT = -0.255              # [kg m⁻³ K⁻¹] DRHO_DT
CP = 3992.10322329649         # [J kg⁻¹ K⁻¹] C_P (common_KPP/MOM_input)
T_SURF, DTDZ, T_MIN = 15.0, 0.01, 4.0   # SCM_L2_TEMP [°C], SCM_L2_DTDZ [K m⁻¹], the max(4, ·) of SCM_CVMix_tests.F90
MLD_DRHO = 0.03               # [kg m⁻³] MLD_003

NAMES = ("wind_only", "cooling_only", "mech_only", "skin_warming_wind")
CRITERION = ("wind_only", "cooling_only", "mech_only")      # spec criterion 2: ±5 %; skin_warming_wind reported
COORDS = ("zstar", "sigma")


@dataclass(frozen=True)
class SCMCase:
    """One CVmix_SCM_tests case (MOM_override values; units in the comments)."""
    name: str
    taux: float              # [Pa] CONST_WIND_TAUX (WIND_CONFIG const; 0 for 'zero')
    gust: float              # [Pa] GUST_CONST
    q: float                 # [W m⁻²] SENSIBLE_HEAT_FLUX (BUOY_CONFIG const; 0 for 'zero'), + into the ocean
    dtdz: float              # [K m⁻¹] SCM_L2_DTDZ

    @property
    def ustar(self) -> float:
        """u* = ((|τ| + gust)/ρ₀)^½ [m s⁻¹] (MOM_surface_forcing.F90:459)."""
        return float(np.sqrt((abs(self.taux) + self.gust) / RHO0))

    @property
    def bflux(self) -> float:
        """Surface buoyancy flux B_f = −(g/ρ₀) dρ/dT Q/(ρ₀ c_p) [m² s⁻³], + = stabilising (MOM6/CVMix)."""
        return -G / RHO0 * DRHO_DT * self.q / (RHO0 * CP)

    @property
    def n_steps(self) -> int:
        return int(round(DAYS * 86400.0 / DT))


CASES: dict[str, SCMCase] = {
    "wind_only": SCMCase("wind_only", 0.1, 0.0, 0.0, DTDZ),
    "cooling_only": SCMCase("cooling_only", 0.0, 0.0, -100.0, DTDZ),
    "mech_only": SCMCase("mech_only", 0.0, 0.05, 0.0, DTDZ),
    "skin_warming_wind": SCMCase("skin_warming_wind", 0.1, 0.0, 100.0, 0.0),
}


def initial_T(case: SCMCase) -> np.ndarray:
    """T(z_c) = max(4, 15 + dT/dz min(0, z_c)) at the 1 m layer centres (NK,) [°C], k = 0 at the bed
    (SCM_CVMix_tests.F90, SCM_TEMP_MLD = 0)."""
    zc = -DEPTH + (np.arange(NK) + 0.5) * (DEPTH / NK)
    return np.maximum(T_MIN, T_SURF + case.dtdz * np.minimum(0.0, zc))


def _install_gust(model: SigmaBaroclinic, case: SCMCase) -> None:
    """u*² += GUST_CONST/ρ₀ in the KPP update of this instance (MOM6 u* = ((|τ| + gust)/ρ₀)^½); the gust does not
    enter the momentum, as in MOM6 (only ``forces%ustar``/``tau_mag`` carry it).  The u*² passed on is kept in
    ``model._closure.gust_us2`` for ``check_gust``."""
    cl = model._closure
    inner = cl.update
    add = case.gust / RHO0

    def update(u, v, h, rho, ustar2_sfc, ustar2_bot, bflux=None):
        us2 = (np.zeros(model.H.shape) if ustar2_sfc is None else np.asarray(ustar2_sfc, float)) + add
        cl.gust_us2 = us2
        return inner(u, v, h, rho, us2, ustar2_bot, bflux=bflux)
    cl.gust_us2 = None
    cl.update = update


def check_gust(model: SigmaBaroclinic, case: SCMCase) -> None:
    """Fail loudly if, after a step, the KPP closure did not receive u*² = (|τ| + gust)/ρ₀ through the hook of
    ``_install_gust`` (it relies on the driver calling ``model._closure.update``)."""
    us2 = getattr(model._closure, "gust_us2", None)
    if us2 is None or not np.allclose(us2, case.ustar ** 2, rtol=1e-12, atol=0.0):
        raise RuntimeError(f"GUST_CONST did not reach the KPP u*² (got {us2!r}, want {case.ustar ** 2!r})")


def build(name: str, coord: str = "zstar", kpp_params: KPPParams | None = None, **overrides):
    """(model, state, tau_x, tau_y, heat) of case ``name`` in 'zstar' (ALE z*, MOM6's coordinate) or 'sigma'."""
    if name not in CASES:
        raise ValueError(f"unknown case {name!r}; use {NAMES}")
    if coord not in COORDS:
        raise ValueError(f"unknown coordinate {coord!r}; use {COORDS}")
    c = CASES[name]
    hg = create_cartesian_grid(nx=NXY, ny=NXY, dx=DX, dy=DX, f0=F0)
    kw = dict(rho0=RHO0, t_coef=-DRHO_DT, t_ref=T_SURF, closure="kpp", g=G, cp=CP, periodic=("x", "y"), z0=None,
              theta=0.6, d_min=0.05, kpp_params=kpp_params or KPPParams(interp_type="cubic"))
    if coord == "zstar":
        kw.update(vertical_coordinate="ale", regrid="zstar", pgf="fv",
                  regrid_params=dict(dz_top=np.full(NK, DEPTH / NK)), hbbl=0.5)
    kw.update(overrides)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")              # pre-flight notes (e.g. kv with kpp) are expected here
        model = SigmaBaroclinic(hg, np.full((NXY, NXY), DEPTH), dt=DT, n_layers=NK, **kw)
    if c.gust:
        _install_gust(model, c)
    T = initial_T(c)[:, None, None] * np.ones((NK, NXY, NXY))
    st = model.initial_state(np.zeros((NXY, NXY)), T)
    tau_x = np.full((NXY, NXY + 1), c.taux)
    tau_y = np.zeros((NXY + 1, NXY))
    heat = {"Q_ns": np.full((NXY, NXY), c.q)} if c.q else None
    return model, st, tau_x, tau_y, heat


# --------------------------------------------------------------------------------------------- diagnostics
def mld_003(zw: np.ndarray, T: np.ndarray, drho: float = MLD_DRHO) -> float:
    """
    MLD [m] of one profile as MOM6 ``diagnoseMLDbyDensityDifference`` with HREF_FOR_MLD = 0
    (MOM_diagnose_MLD.F90:131-224): ρ_s = ρ(top layer); going down, the first centre k where Δρ_k = ρ_k − ρ_s ≥ drho
    with Δρ_{k−1} < drho and Δρ_k > Δρ_{k−1}; the depth is interpolated linearly between the two centres.  If no
    centre reaches drho, the deepest centre.  Linear EOS: Δρ = dρ/dT (T − T_s) = 0.255 (T_s − T).
    zw (N+1,) interface heights, bed first, z ≤ 0;  T (N,) [°C], bed first.
    """
    zw, T = np.asarray(zw, float)[::-1], np.asarray(T, float)[::-1]      # surface first
    dz = -np.diff(zw)                                                      # (N,) > 0
    depth_c = np.cumsum(dz) - 0.5 * dz
    drk = DRHO_DT * (T - T[0])
    for k in range(1, len(T)):
        d0, d1 = drk[k - 1], drk[k]
        if d1 - d0 > 0.0 and d0 < drho <= d1:
            a = (drho - d0) / (d1 - d0)
            return float(depth_c[k] * a + depth_c[k - 1] * (1.0 - a))
    return float(depth_c[-1])


def load_mom6(name: str) -> dict | None:
    """MOM6 outputs of one case: 2-hourly T snapshots (prog.nc, prepended with MOM_IC.nc at t = 0), and the
    2-hour window means of KPP_OBLdepth and MLD_003 (visc.nc) [s, m, °C, m s⁻¹]."""
    d = MOM6_RUNS / name
    if not (d / "visc.nc").exists():
        return None
    import netCDF4
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p, v = netCDF4.Dataset(d / "prog.nc"), netCDF4.Dataset(d / "visc.nc")
        ic = netCDF4.Dataset(d / "MOM_IC.nc")
        h = np.asarray(p["h"][:, :, 0, 0], float)
        T = np.asarray(p["temp"][:, :, 0, 0], float)
        T0 = np.asarray(ic["Temp"][0, :, 0, 0], float) if "Temp" in ic.variables else np.asarray(
            ic["temp"][0, :, 0, 0], float)
        h0 = np.asarray(ic["h"][0, :, 0, 0], float)
        out = dict(t_snap=np.concatenate([[0.0], 86400.0 * np.asarray(p["Time"][:], float)]),
                   t_win=86400.0 * np.asarray(v["Time"][:], float),
                   obl=np.asarray(v["KPP_OBLdepth"][:, 0, 0], float),
                   mld_diag=np.asarray(v["MLD_003"][:, 0, 0], float),
                   ustar=float(v["KPP_uStar"][0, 0, 0]), bflux=float(np.ravel(v["KPP_buoyFlux"][0, 0, 0, 0])[0]),
                   u_sfc=np.concatenate([[0.0], np.asarray(p["u"][:, 0, 0, 0], float)]))
        for f in (p, v, ic):
            f.close()
    # MOM6 is surface first: reverse to bed first, interfaces from the thicknesses
    hh = np.vstack([h0[None], h])[:, ::-1]
    out["T"] = np.vstack([T0[None], T])[:, ::-1]
    out["zw"] = np.concatenate([np.zeros((hh.shape[0], 1)), np.cumsum(hh, axis=1)], axis=1) - hh.sum(1)[:, None]
    out["mld"] = np.array([mld_003(z, t) for z, t in zip(out["zw"], out["T"])])
    out["sst"] = out["T"][:, -1]
    return out


# ----------------------------------------------------------------------------------------------- pyOCEAN run
def run(name: str, coord: str = "zstar", save: bool = True) -> dict:
    """Run one case for 5 days; 2-hourly snapshots (T, zw, u, v of column (0, 0)) and per-step OBL depth, its
    2-hour window means as MOM6's KPP_OBLdepth, and the wall time."""
    warnings.simplefilter("ignore")
    c = CASES[name]
    model, st, tx, ty, heat = build(name, coord)

    def snap(st):
        h = st.h[:, 0, 0] if st.h is not None else model.layer_volumes(st.eta)[:, 0, 0] / model.area[0, 0]
        zw = np.concatenate([[0.0], np.cumsum(h)]) - h.sum()
        return dict(t=st.time, zw=zw, T=st.T[:, 0, 0].copy(), u=st.u[:, 0, 0].copy(), v=st.v[:, 0, 0].copy())

    snaps, obl, spread = [snap(st)], [], 0.0
    t0 = time.perf_counter()
    for n in range(1, c.n_steps + 1):
        st = model.step(st, tx, ty, heat=heat)
        obl.append(float(np.ravel(model._closure.obl_depth)[0]))
        if n == 1 and c.gust:
            check_gust(model, c)
        if n % NOUT == 0:
            if not (np.isfinite(st.T).all() and np.isfinite(st.u).all()):
                raise FloatingPointError(f"{name} {coord}: non-finite state at step {n}")
            snaps.append(snap(st))
            spread = max(spread, float(np.ptp(st.T, axis=(1, 2)).max()), float(np.ptp(st.u, axis=(1, 2)).max()))
    wall = time.perf_counter() - t0
    obl = np.array(obl)
    out = {k: np.array([s[k] for s in snaps]) for k in snaps[0]}
    out.update(obl_step=obl, obl=obl.reshape(-1, NOUT).mean(axis=1),
               t_win=DT * (np.arange(len(obl) // NOUT) * NOUT + 0.5 * NOUT),
               mld=np.array([mld_003(z, t) for z, t in zip(out["zw"], out["T"])]), sst=out["T"][:, -1],
               wall=wall, n_steps=c.n_steps, s_per_step=wall / c.n_steps, spread=spread)
    if save:
        PY_OUT.mkdir(parents=True, exist_ok=True)
        np.savez(PY_OUT / f"{name}_{coord}_kpp.npz", **out)
    return out


def load_py(name: str, coord: str) -> dict | None:
    path = PY_OUT / f"{name}_{coord}_kpp.npz"
    if not path.exists():
        return None
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


# ------------------------------------------------------------------------------------------------- comparison
def rel(a: float, b: float) -> float:
    """(a − b)/b [%]."""
    return 100.0 * (a - b) / b


def compare_case(name: str, coord: str) -> dict | None:
    """Final (day 5) OBL and MLD of pyOCEAN vs MOM6, and the largest |relative difference| over days 1-5 [m, %]."""
    mo, py = load_mom6(name), load_py(name, coord)
    if mo is None or py is None:
        return None
    day1_w = mo["t_win"] >= 86400.0
    day1_s = mo["t_snap"] >= 86400.0
    r_obl = 100.0 * (py["obl"] - mo["obl"]) / mo["obl"]
    r_mld = 100.0 * (py["mld"] - mo["mld"]) / mo["mld"]
    return dict(obl_m=mo["obl"][-1], obl_p=py["obl"][-1], obl_rel=r_obl[-1],
                obl_max=float(np.abs(r_obl[day1_w]).max()),
                mld_m=mo["mld"][-1], mld_p=py["mld"][-1], mld_rel=r_mld[-1],
                mld_max=float(np.abs(r_mld[day1_s]).max()),
                mld_diag_m=mo["mld_diag"][-1], sst_m=mo["sst"][-1], sst_p=py["sst"][-1],
                us_m=mo["u_sfc"][-1], us_p=py["u"][-1, -1], wall=float(py["wall"]),
                ms_step=1e3 * float(py["s_per_step"]), spread=float(py["spread"]))


def table() -> str:
    lines = ["| caso | coord | h_OBL MOM6 | h_OBL pyOCEAN | Δ final | máx \\|Δ\\| d 1-5 | MLD MOM6 (diag) | "
             "MLD pyOCEAN | Δ final | máx \\|Δ\\| d 1-5 | SST MOM6 / py | u_sup MOM6 / py | ±5 % |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for name in NAMES:
        for coord in COORDS:
            r = compare_case(name, coord)
            if r is None:
                continue
            ok = "—" if name not in CRITERION else (
                "pasa" if abs(r["obl_rel"]) <= 5 and abs(r["mld_rel"]) <= 5 else "NO")
            lines.append(
                f"| {name} | {coord} | {r['obl_m']:.2f} | {r['obl_p']:.2f} | {r['obl_rel']:+.1f} % | "
                f"{r['obl_max']:.1f} % | {r['mld_m']:.2f} ({r['mld_diag_m']:.2f}) | {r['mld_p']:.2f} | "
                f"{r['mld_rel']:+.1f} % | {r['mld_max']:.1f} % | {r['sst_m']:.3f} / {r['sst_p']:.3f} | "
                f"{r['us_m']:+.4f} / {r['us_p']:+.4f} | {ok} |")
    return "\n".join(lines)


def series(name: str, coord: str, every_h: int = 12) -> str:
    """OBL and MLD every ``every_h`` hours, MOM6 vs pyOCEAN."""
    mo, py = load_mom6(name), load_py(name, coord)
    if mo is None or py is None:
        return ""
    k = every_h // 2
    rows = [f"{name} {coord}: t [h] | OBL MOM6 | OBL py | MLD MOM6 | MLD py"]
    for i in range(k - 1, len(mo["obl"]), k):
        rows.append(f"  {mo['t_win'][i] / 3600:6.1f} | {mo['obl'][i]:7.2f} | {py['obl'][i]:7.2f} | "
                    f"{mo['mld'][i + 1]:7.2f} | {py['mld'][i + 1]:7.2f}")
    return "\n".join(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    r = sub.add_parser("run")
    r.add_argument("name", nargs="?", choices=NAMES)
    r.add_argument("coord", nargs="?", choices=COORDS, default="zstar")
    r.add_argument("--all", action="store_true", help="the four cases in z* and σ, one run at a time")
    c = sub.add_parser("compare")
    c.add_argument("--series", action="store_true")
    a = ap.parse_args()
    if a.cmd == "run":
        runs = [(n, co) for n in NAMES for co in COORDS] if a.all else [(a.name, a.coord)]
        for n, co in runs:
            out = run(n, co)
            print(f"{n} {co}: {out['n_steps']} steps, {out['wall']:.1f} s wall "
                  f"({1e3 * out['s_per_step']:.1f} ms/step), 2x2 spread {out['spread']:.1e}", flush=True)
    else:
        print(table())
        if a.series:
            for n in NAMES:
                for co in COORDS:
                    print(series(n, co))


if __name__ == "__main__":
    main()
