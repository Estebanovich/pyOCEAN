"""
MOM6 runs of the TEOS-10 rest case (criterion 4, open item 2 of the TEOS-10 row in CLAUDE.md): the WOA23 profile at
rest over the Beckmann & Haidvogel seamount (``cases.rest_case("seamount", eos, coord, rho_ref=None)``), 10 days, in
compiled MOM6 (git 0e9a561, ``~/pyOCEAN_refs/MOM6/.testing/build/repro/MOM6``), against pyOCEAN's MOM6 mode
(constant RHO_PGF_REF = ρ₀ subtracted, no reference profile).

Same initial state.  MOM6 starts from **pyOCEAN's own initial state** (h, T, S of ``rest_case`` after its
initialisation, layers flipped to surface first): THICKNESS_CONFIG = "thickness_file" and TS_CONFIG = "file"
(MOM_state_initialization.F90:304-308, 402, 1780-1820) with REMAP_AFTER_INITIALIZATION = False (MOM_ALE.F90:293-297;
the default True regrids the initial condition once more, MOM.F90:3469-3497).  The z* initial grid is the same rule
(pyOCEAN zstar_column = MOM6 coord_zlike to 3e-14 m); the hybrid one is pyOCEAN's (regrid on the 4N data levels +
snap_to_jumps + keep_isopycnal) and MOM6 regrids it with its own HYCOM1 rule at the end of the first step.  So the
initial |PGF| of the two models is computed on identical (h, T, S).  The teos10 state is already Θ, S_A
(``physics/eos/convert.py``: S_A = 35.16504/35 S_P, Θ = CT_from_pt), what ROQUET_RHO takes.

Settings (every one cited to MOM6 source, git 0e9a561, paths relative to ``~/pyOCEAN_refs/MOM6/src``):
  EOS          LINEAR (RHO_REF_LINEAR_EOS = ρ₀, T/S_REF_LINEAR_EOS, DRHO_DT = −α, DRHO_DS = β of
               ``cases.linear_equivalent``; equation_of_state/MOM_EOS.F90:1598-1636), WRIGHT_FULL, ROQUET_RHO
               (:1563-1586); RHO_0 = RHO_PGF_REF = 1035, RHO_PGF_REF_BUG = False (core/MOM_PressureForce_FV.F90:
               2061-2075; inert when RHO_PGF_REF = RHO_0).
  PGF          ANALYTIC_FV_PGF (finite volume), RECONSTRUCT_FOR_PRESSURE (default = USE_REGRIDDING) with
               PRESSURE_RECONSTRUCTION_SCHEME = 1 (PLM) and BOUNDARY_EXTRAPOLATION_PRESSURE = True
               (MOM_PressureForce_FV.F90:2173-2190) → int_density_dz_generic_plm for all three EOS;
               MASS_WEIGHT_IN_PRESSURE_GRADIENT = True, _TOP = False (its default follows the former, :2124-2133),
               MASS_WEIGHT_IN_PGF_VANISHED_ONLY = True with RESET_INTXPA_H_NONVANISHED = 0.01 m (:2138-2141,
               2166-2169) = pyOCEAN pgf_mass_weight "vanished" with h_vanished 0.01 m (core/vertical/pgf.py:51);
               CORRECTION_INTXPA = RESET_INTXPA_INTEGRAL = False, SSH_IN_EOS_PRESSURE_FOR_PGF = False.
  coordinate   ZSTAR with the 20 uniform levels H_max/N (ALE_COORDINATE_CONFIG "FILE:vgrid.nc,dz"), or HYCOM1 with
               pyOCEAN's floors and targets ("HYBRID:vgrid.nc,sigma2,dz", ALE/MOM_regridding.F90:530-563; the
               "sigma2" variable holds full densities ρ(T, S, P_REF) in kg m⁻³, as the targets are compared with
               calculate_density at P_REF, coord_hycom.F90:214-219), P_REF = 2e7 (MOM_regridding.F90:971-993),
               INTERPOLATION_SCHEME P1M_H2 and BOUNDARY_EXTRAPOLATION False (defaults, :195-197),
               MIN_THICKNESS 1e-3 m (default, = pyOCEAN H_MIN), HYCOM1_ONLY_IMPROVES False (default).
               REMAPPING_SCHEME = VELOCITY_REMAPPING_SCHEME = PPM_H4 (= pyOCEAN remap_scheme default).
  momentum     vector invariant, SADOURNY75_ENERGY, BOUND_CORIOLIS = True (core/MOM_CoriolisAdv.F90:2163-2166;
               pyOCEAN core/dynamics/vector_invariant.py), KE_ARAKAWA; no horizontal viscosity (LAPLACIAN and
               BIHARMONIC False); KV = 1e-4; free-slip bed: BOTTOMDRAGLAW with CDRAG = 0, KV_BBL_MIN = 0
               (MOM_vert_friction.F90:2263-2271); continuity PPM (default) = pyOCEAN thickness_flux 'ppm'.
  tracers      PPM:H3, KHTR = 0, KD = 0 (pyOCEAN: WENO5 + FCT, kh = kv = 0).
  time         DT = DT_THERM = 600 s, split RK2 with DTBT = −0.98 (pyOCEAN: AB3 with the implicit free surface,
               θ = 0.6, θ_bc = ½); MAXTRUNC large so a truncating run is not stopped (truncations counted in
               ocean.stats and reported).

Outputs: 'prog' (u, v, h hourly snapshots, float32) of the 10-day run; a separate one-step run writes PFu, PFv at
t = Δt: with BEGW = 0 (default) the corrector's PressureForce uses h at time n (core/MOM_dynamics_split_RK2.F90:
840-848, posted at :1120), i.e. the initial-state PGF, the quantity pyOCEAN's ``pgf0`` is.

Index order of the pyOCEAN state (k, j, i), k = 0 at the bed; MOM6 files are surface first.
"""
from __future__ import annotations

import json
from pathlib import Path

import netCDF4
import numpy as np

from validation.teos10 import cases as C

RUNS = C.RUNS / "mom6_rest"
H_THICK = 1.0          # [m] faces whose two cells are both at least this thick count as "thick" (metric umax_thick)


def run_dir(eos: str, coord: str, mode: str = "run", n_layers: int | None = None, drag: bool = False) -> Path:
    """Output directory of one MOM6 run ('run' = 10 days, 'pgf0' = one step for the initial PGF)."""
    tag = ("" if n_layers is None else f"_n{int(n_layers)}") + ("_drag" if drag else "")
    return RUNS / f"{eos}_{coord}{tag}" / mode


def _case(eos: str, coord: str, n_layers: int | None = None, drag: bool = False) -> C.Case:
    return C.rest_case("seamount", eos, coord, rho_ref=None, n_layers=n_layers, drag=drag)


def mom_input(case: C.Case, eos: str, coord: str, days: float, drag: bool = False) -> str:
    """The MOM_input text of one run."""
    m = case.model
    ny, nx = m.H.shape
    dx = C._rest_geometry("seamount", False)["dx"]
    hmax = float(np.nanmax(m.H))
    L = [f"NIGLOBAL = {nx}", f"NJGLOBAL = {ny}", f"NK = {m.n}",
         "REENTRANT_X = False", "REENTRANT_Y = False", "LAYOUT = 1, 1", "IO_LAYOUT = 1, 1",
         'INPUTDIR = "INPUT"', 'GRID_CONFIG = "cartesian"', 'AXIS_UNITS = "k"', "WESTLON = 0.0", "SOUTHLAT = 0.0",
         f"LENLON = {nx * dx / 1e3:.10g}", f"LENLAT = {ny * dx / 1e3:.10g}",
         'TOPO_CONFIG = "file"', 'TOPO_FILE = "topog.nc"', 'TOPO_VARNAME = "depth"',
         "MINIMUM_DEPTH = 0.5", "MASKING_DEPTH = 0.0", f"MAXIMUM_DEPTH = {hmax!r}",
         'ROTATION = "betaplane"', f"F_0 = {C.SM_F0}", "BETA = 0.0", f"G_EARTH = {C.G}",
         f"RHO_0 = {C.RHO0}", f"RHO_PGF_REF = {C.RHO0}", "RHO_PGF_REF_BUG = False"]
    if eos == "linear":
        k = C._eos_kwargs("linear")
        L += ['EQN_OF_STATE = "LINEAR"', f"RHO_REF_LINEAR_EOS = {k['rho0']!r}", f"T_REF_LINEAR_EOS = {k['t_ref']!r}",
              f"S_REF_LINEAR_EOS = {k['s_ref']!r}", f"DRHO_DT = {-k['t_coef']!r}", f"DRHO_DS = {k['s_coef']!r}"]
    else:
        L += [f'EQN_OF_STATE = "{ {"wright_full": "WRIGHT_FULL", "teos10": "ROQUET_RHO"}[eos] }"']
    L += ["USE_REGRIDDING = True", 'REMAPPING_SCHEME = "PPM_H4"', 'VELOCITY_REMAPPING_SCHEME = "PPM_H4"',
          "REMAP_AFTER_INITIALIZATION = False"]
    if coord == "zstar":
        L += ['REGRIDDING_COORDINATE_MODE = "ZSTAR"', 'ALE_COORDINATE_CONFIG = "FILE:vgrid.nc,dz"']
    elif coord == "hybrid":
        L += ['REGRIDDING_COORDINATE_MODE = "HYCOM1"', 'ALE_COORDINATE_CONFIG = "HYBRID:vgrid.nc,sigma2,dz"',
              f"P_REF = {C.P_REF}", 'INTERPOLATION_SCHEME = "P1M_H2"', "BOUNDARY_EXTRAPOLATION = False"]
    else:
        raise ValueError(f"unknown coord {coord!r}")
    L += ['THICKNESS_CONFIG = "thickness_file"', 'THICKNESS_FILE = "ic.nc"', 'THICKNESS_IC_VAR = "h"',
          'TS_CONFIG = "file"', 'TS_FILE = "ic.nc"', 'TEMP_IC_VAR = "ptemp"', 'SALT_IC_VAR = "salt"',
          # finite-volume PGF as NWA12 except the mass weighting (pyOCEAN 'vanished', see the docstring)
          "ANALYTIC_FV_PGF = True", "RECONSTRUCT_FOR_PRESSURE = True", "PRESSURE_RECONSTRUCTION_SCHEME = 1",
          "BOUNDARY_EXTRAPOLATION_PRESSURE = True", "MASS_WEIGHT_IN_PRESSURE_GRADIENT = True",
          "MASS_WEIGHT_IN_PRESSURE_GRADIENT_TOP = False", "MASS_WEIGHT_IN_PGF_VANISHED_ONLY = True",
          "RESET_INTXPA_H_NONVANISHED = 0.01", "CORRECTION_INTXPA = False", "RESET_INTXPA_INTEGRAL = False",
          "SSH_IN_EOS_PRESSURE_FOR_PGF = False", "USE_INACCURATE_PGF_RHO_ANOM = False",
          'TRACER_ADVECTION_SCHEME = "PPM:H3"', 'CORIOLIS_SCHEME = "SADOURNY75_ENERGY"', "BOUND_CORIOLIS = True",
          'KE_SCHEME = "KE_ARAKAWA"', "LAPLACIAN = False", "BIHARMONIC = False", "KV = 1.0E-04",
          "KHTR = 0.0", "KD = 0.0", "BOTTOMDRAGLAW = True"]
    # free-slip bed (CDRAG 0, KV_BBL_MIN 0), or the drag sensitivity: MOM6 defaults CDRAG 3e-3, KV_BBL_MIN = KV,
    # HBBL 10 m (MOM_set_viscosity.F90:3129-3140, 3226-3228), no DRAG_BG_VEL; pyOCEAN cases.rest_case(drag=True)
    L += (["CDRAG = 0.003", "HBBL = 10.0", "DRAG_BG_VEL = 0.0"] if drag
          else ["CDRAG = 0.0", "KV_BBL_MIN = 0.0", "HBBL = 0.001"])
    L += [
          "HMIX_FIXED = 0.5", "KV_ML_INVZ2 = 0.0", 'BUOY_CONFIG = "zero"', 'WIND_CONFIG = "zero"',
          f"DT = {case.model.dt}", f"DT_THERM = {case.model.dt}", "DTBT = -0.98", "MAXTRUNC = 100000000",
          'U_TRUNC_FILE = "U_velocity_truncations"', 'V_TRUNC_FILE = "V_velocity_truncations"',
          "TIMEUNIT = 1.0", f"DAYMAX = {days * C.DAY}", "ENERGYSAVEDAYS = 3600.0", "RESTART_CONTROL = -1",
          "SAVE_INITIAL_CONDS = True", 'IC_OUTPUT_FILE = "MOM_IC"']
    return "\n".join(L) + "\n"


DIAG_RUN = '''"TEOS-10 rest {name}"
1 1 1 0 0 0
"prog", 1, "hours", 1, "days", "Time",
"ocean_model", "u", "u", "prog", "all", .false., "none", 2
"ocean_model", "v", "v", "prog", "all", .false., "none", 2
"ocean_model", "h", "h", "prog", "all", .false., "none", 2
'''

DIAG_PGF0 = '''"TEOS-10 rest PGF0 {name}"
1 1 1 0 0 0
"pf", {dt:g}, "seconds", 1, "seconds", "Time",
"ocean_model", "PFu", "PFu", "pf", "all", .false., "none", 1
"ocean_model", "PFv", "PFv", "pf", "all", .false., "none", 1
"ocean_model", "h", "h", "pf", "all", .false., "none", 1
'''

NML = ''' &MOM_input_nml
         output_directory = './',
         input_filename = 'n'
         restart_input_dir = './',
         restart_output_dir = 'RESTART/',
         parameter_filename = 'MOM_input',
                              'MOM_override' /
 &diag_manager_nml
 /
 &fms_nml
         clock_grain = 'ROUTINE',
         domains_stack_size = 4000000,
         stack_size = 0 /
'''


def write_case(eos: str, coord: str, mode: str = "run", n_layers: int | None = None,
               out_dir: Path | None = None, drag: bool = False) -> Path:
    """A ready-to-run MOM6 directory (MOM_input, MOM_override, diag_table, input.nml, INPUT/{topog,ic,vgrid}.nc,
    RESTART/) for the 10-day run ('run') or the one-step initial-PGF run ('pgf0')."""
    case = _case(eos, coord, n_layers, drag)
    m, st = case.model, case.state
    out = Path(out_dir) if out_dir is not None else run_dir(eos, coord, mode, n_layers, drag)
    (out / "RESTART").mkdir(parents=True, exist_ok=True)
    (out / "INPUT").mkdir(exist_ok=True)
    ny, nx = m.H.shape
    with netCDF4.Dataset(out / "INPUT" / "topog.nc", "w") as ds:
        ds.createDimension("ny", ny)
        ds.createDimension("nx", nx)
        ds.createVariable("depth", "f8", ("ny", "nx"))[:] = np.nan_to_num(m.H)
    with netCDF4.Dataset(out / "INPUT" / "ic.nc", "w") as ds:      # surface first (MOM6 k = 1 at the top)
        for n, s in (("zl", m.n), ("lath", ny), ("lonh", nx)):
            ds.createDimension(n, s)
        for name, val in (("h", st.h), ("ptemp", st.T), ("salt", st.S)):
            ds.createVariable(name, "f8", ("zl", "lath", "lonh"))[:] = np.asarray(val, float)[::-1]
    p = m.regrid_params
    with netCDF4.Dataset(out / "INPUT" / "vgrid.nc", "w") as ds:   # dz surface first; targets surface first
        ds.createDimension("nz", m.n)
        ds.createDimension("nzp1", m.n + 1)
        ds.createVariable("dz", "f8", ("nz",))[:] = np.asarray(p["dz_top"], float)
        if coord == "hybrid":
            ds.createVariable("sigma2", "f8", ("nzp1",))[:] = np.asarray(p["rho_target"], float)
    days = 10.0 if mode == "run" else m.dt / C.DAY
    (out / "MOM_input").write_text(mom_input(case, eos, coord, days, drag))
    (out / "MOM_override").write_text("")
    name = f"{eos} {coord}"
    (out / "diag_table").write_text(DIAG_RUN.format(name=name) if mode == "run"
                                    else DIAG_PGF0.format(name=name, dt=m.dt))
    (out / "input.nml").write_text(NML)
    (out / "case.json").write_text(json.dumps(dict(eos=eos, coord=coord, mode=mode, n_layers=n_layers, drag=drag,
                                                   name=case.name)))
    return out


# ---------------------------------------------------------------- shared metrics (pyOCEAN and MOM6)
def face_speed_max(u: np.ndarray, v: np.ndarray, h: np.ndarray, h_thick: float = H_THICK) -> tuple[float, float]:
    """(max |u|, |v| over every face and layer [m s⁻¹], the same over faces whose two cells are both ≥ ``h_thick``
    thick).  u (N, ny, nx+1), v (N, ny+1, nx), h (N, ny, nx), any layer order but the same for the three; the
    boundary faces (walls) are included in the first value and excluded from the second."""
    u, v, h = (np.nan_to_num(np.asarray(a, float)) for a in (u, v, h))
    allmax = float(max(np.abs(u).max(), np.abs(v).max()))
    thick = h >= h_thick
    tu = thick[:, :, 1:] & thick[:, :, :-1]
    tv = thick[:, 1:, :] & thick[:, :-1, :]
    um = np.abs(u[:, :, 1:-1])[tu]
    vm = np.abs(v[:, 1:-1, :])[tv]
    thickmax = float(max(um.max() if um.size else 0.0, vm.max() if vm.size else 0.0))
    return allmax, thickmax


def read_run(d: Path) -> dict:
    """Hourly max |u|, |v| series (all faces and thick faces) of a 10-day MOM6 run, with record 0 = t = 0 (rest),
    the argmax (k bed first, j, i) of |u| at the end and the truncation count from ocean.stats."""
    d = Path(d)
    t, umax, uthick = [0.0], [0.0], [0.0]
    with netCDF4.Dataset(d / "prog.nc") as ds:
        unit = ds["Time"].units.split()[0].lower()
        tt = np.asarray(ds["Time"][:], float) * {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[unit]
        for n in range(tt.size):
            u = np.ma.filled(ds["u"][n], 0.0)
            v = np.ma.filled(ds["v"][n], 0.0)
            h = np.ma.filled(ds["h"][n], 0.0)
            a, b = face_speed_max(u, v, h)
            t.append(tt[n])
            umax.append(a)
            uthick.append(b)
        uf = np.abs(np.ma.filled(ds["u"][-1], 0.0))[::-1]
    out = dict(time=np.array(t), umax=np.array(umax), umax_thick=np.array(uthick),
               argmax_u=np.array(np.unravel_index(uf.argmax(), uf.shape)))
    out.update(stats(d))
    return out


def stats(d: Path) -> dict:
    """Truncations (cumulative, last line) and the maximum CFL over the run from ocean.stats (ASCII)."""
    trunc, cfl = 0, 0.0
    for line in (Path(d) / "ocean.stats").read_text().splitlines():
        parts = [p.strip() for p in line.split(",")]
        if len(parts) > 4 and parts[0].isdigit():
            trunc = int(parts[2])
            cfl = max(cfl, float(parts[4].split()[-1]))
    return dict(n_trunc=trunc, cfl_max=cfl)


def read_pgf0(d: Path) -> dict:
    """max |PFu|, |PFv| [m s⁻²] of the one-step run (= the PGF of the initial state, module docstring), over all
    faces and over faces whose two cells are ≥ ``H_THICK`` thick."""
    with netCDF4.Dataset(Path(d) / "pf.nc") as ds:
        pu = np.ma.filled(ds["PFu"][0], 0.0)
        pv = np.ma.filled(ds["PFv"][0], 0.0)
        h = np.ma.filled(ds["h"][0], 0.0)
    a, b = face_speed_max(pu, pv, h)
    return dict(pgf0=a, pgf0_thick=b)


def check_ic(d: Path, eos: str, coord: str, n_layers: int | None = None, drag: bool = False) -> dict:
    """Largest difference between MOM6's saved initial state (MOM_IC.nc) and pyOCEAN's: h [m], T, S."""
    case = _case(eos, coord, n_layers, drag)
    st = case.state
    with netCDF4.Dataset(Path(d) / "MOM_IC.nc") as ds:
        h = np.asarray(ds["h"][0], float)[::-1]
        T = np.asarray(ds["Temp"][0], float)[::-1]
        S = np.asarray(ds["Salt"][0], float)[::-1]
    return dict(dh=float(np.abs(h - st.h).max()), dT=float(np.abs(T - st.T).max()),
                dS=float(np.abs(S - st.S).max()))


def save(eos: str, coord: str, n_layers: int | None = None, drag: bool = False) -> Path:
    """series.npz of a finished pair of runs (10-day + one-step) in the run's parent directory."""
    base = run_dir(eos, coord, "run", n_layers, drag).parent
    r = read_run(base / "run")
    r.update(read_pgf0(base / "pgf0"))
    r.update({f"ic_{k}": v for k, v in check_ic(base / "run", eos, coord, n_layers, drag).items()})
    np.savez(base / "series.npz", **r)
    return base / "series.npz"


def load(eos: str, coord: str, n_layers: int | None = None, drag: bool = False) -> dict | None:
    p = run_dir(eos, coord, "run", n_layers, drag).parent / "series.npz"
    if not p.exists():
        return None
    with np.load(p) as d:
        return {k: d[k] for k in d.files}
