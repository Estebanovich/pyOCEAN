"""
MOM6 runs of the four-model benchmark (spec §A2): cartesian grid (GRID_CONFIG 'cartesian', AXIS_UNITS 'k'), depth
file, T on the case z-levels as a z-file with ALE remapping (MOM_state_initialization.F90:2697-2722; the data are on
the model grid, so only the vertical remap), linear EOS in T, PPM_H4 remap, PPM:H3 tracers, KE_ARAKAWA, Laplacian
KH, KV = 1e-4, no tracer diffusion, free-slip bed (BOTTOMDRAGLAW with CDRAG = 0 and KV_BBL_MIN = 0; with
BOTTOMDRAGLAW = False the bed is viscous no-slip, MOM_vert_friction.F90:2263-2271), g = 9.81.
Coordinates: 'zstar', 'rho' (COORD_CONFIG 'layer_ref' with cases.targets), 'sigma'. 2-D cases: NJGLOBAL = 1 on one
rank (NJ = 1 fails on several ranks).
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from validation.ilicak_benchmark import cases

FILL = -1.0e20


def source_centres(dz: np.ndarray) -> np.ndarray:
    """Centres c_k [m, positive down] whose midpoints are the case interfaces (MOM6 builds the source edges as
    midpoints of the centres, MOM_horizontal_regridding.F90:454-480)."""
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    c = np.empty(dz.size)
    c[0] = 0.5 * zf[1]
    for k in range(dz.size - 1):
        c[k + 1] = 2.0 * zf[k + 1] - c[k]
    return c


def mom_input(case: cases.Case, coord: str, nu: float, dt: float, hours: float) -> str:
    """The MOM_input text of one run."""
    modes = {"zstar": "ZSTAR", "rho": "RHO", "sigma": "SIGMA"}
    if coord not in modes:
        raise ValueError(f"unknown coord {coord!r}")
    L = [f"NIGLOBAL = {case.nx}", f"NJGLOBAL = {case.ny}", f"NK = {case.nz}",
         "REENTRANT_X = False", "REENTRANT_Y = False", "LAYOUT = 1, 1", "IO_LAYOUT = 1, 1",
         'INPUTDIR = "INPUT"', 'GRID_CONFIG = "cartesian"', 'AXIS_UNITS = "k"', "WESTLON = 0.0", "SOUTHLAT = 0.0",
         f"LENLON = {case.dx.sum() / 1e3:.10g}", f"LENLAT = {case.dy.sum() / 1e3:.10g}",
         'TOPO_CONFIG = "file"', 'TOPO_FILE = "topog.nc"', 'TOPO_VARNAME = "depth"',
         "MINIMUM_DEPTH = 0.5", "MASKING_DEPTH = 0.0", f"MAXIMUM_DEPTH = {float(np.nanmax(case.H)):.10g}",
         'ROTATION = "betaplane"', f"F_0 = {case.f0}", "BETA = 0.0", "G_EARTH = 9.81", "RHO_0 = 1000.0",
         'EQN_OF_STATE = "LINEAR"', "RHO_REF_LINEAR_EOS = 1000.0", f"T_REF_LINEAR_EOS = {cases.T_REF}",
         f"DRHO_DT = {-cases.ALPHA}", "DRHO_DS = 0.0",
         "USE_REGRIDDING = True", f'REGRIDDING_COORDINATE_MODE = "{modes[coord]}"', 'REMAPPING_SCHEME = "PPM_H4"']
    if coord == "rho":
        t = cases.targets(case, case.nz)
        lay = 0.5 * (t[1:] + t[:-1])
        L += ['COORD_CONFIG = "layer_ref"', f"LIGHTEST_DENSITY = {lay[0]:.12g}",
              f"DENSITY_RANGE = {lay[-1] - lay[0]:.12g}", 'INTERPOLATION_SCHEME = "P1M_H4"',
              "BOUNDARY_EXTRAPOLATION = True"]
    L += ["INIT_LAYERS_FROM_Z_FILE = True", 'TEMP_SALT_Z_INIT_FILE = "ts_z.nc"', 'Z_INIT_FILE_PTEMP_VAR = "ptemp"',
          'Z_INIT_FILE_SALT_VAR = "salt"', "Z_INIT_ALE_REMAPPING = True", 'Z_INIT_REMAPPING_SCHEME = "PPM_H4"',
          "TEMP_SALT_INIT_VERTICAL_REMAP_ONLY = True"]
    if coord != "zstar":
        L += ["Z_INIT_REMAP_GENERAL = True", "Z_INIT_REMAP_FULL_COLUMN = True"]
    L += ['TRACER_ADVECTION_SCHEME = "PPM:H3"', 'KE_SCHEME = "KE_ARAKAWA"', "LAPLACIAN = True", f"KH = {float(nu)}",
          "BIHARMONIC = False", "NOSLIP = False", f"KV = {cases.NU_V}", "KHTR = 0.0", "KD = 0.0",
          "BOTTOMDRAGLAW = True", "CDRAG = 0.0", "KV_BBL_MIN = 0.0", "HBBL = 0.001",
          "HMIX_FIXED = 0.5", "KV_ML_INVZ2 = 0.0", 'BUOY_CONFIG = "zero"', 'WIND_CONFIG = "zero"',
          f"DT = {float(dt)}", f"DT_THERM = {float(dt)}", "DTBT = -0.98",
          "TIMEUNIT = 1.0", f"DAYMAX = {hours * 3600.0}", f"ENERGYSAVEDAYS = {case.out_s}", "RESTART_CONTROL = -1",
          "SAVE_INITIAL_CONDS = True", 'IC_OUTPUT_FILE = "MOM_IC"']
    return "\n".join(L) + "\n"


DIAG = '''"Ilicak {name}"
1 1 1 0 0 0
"prog", {out:g}, "seconds", 1, "seconds", "Time",
"ocean_model", "SSH", "SSH", "prog", "all", .false., "none", 2
"ocean_model", "temp", "temp", "prog", "all", .false., "none", 2
"ocean_model", "u", "u", "prog", "all", .false., "none", 2
"ocean_model", "v", "v", "prog", "all", .false., "none", 2
"ocean_model", "h", "h", "prog", "all", .false., "none", 2
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


def write_case(case_name: str, coord: str, nu: float, out_dir: Path, dt: float | None = None,
               hours: float | None = None) -> Path:
    """A ready-to-run MOM6 directory: MOM_input, MOM_override, diag_table, input.nml, INPUT/{topog,ts_z}.nc, RESTART/."""
    c = cases.get(case_name)
    out = Path(out_dir)
    (out / "RESTART").mkdir(parents=True, exist_ok=True)
    (out / "INPUT").mkdir(exist_ok=True)
    with netCDF4.Dataset(out / "INPUT" / "topog.nc", "w") as ds:
        ds.createDimension("ny", c.ny)
        ds.createDimension("nx", c.nx)
        ds.createVariable("depth", "f8", ("ny", "nx"))[:] = np.nan_to_num(c.H)
    cen = source_centres(c.dz)
    T = c.T_fn(c.x[None, None, :], c.y[None, :, None], -cen[:, None, None])
    T = np.where(cen[:, None, None] < c.H[None] + 0.5 * c.dz[:, None, None], T, FILL)     # below the bed: fill
    with netCDF4.Dataset(out / "INPUT" / "ts_z.nc", "w") as ds:
        for n, s in (("Time", None), ("depth", c.nz), ("lat", c.ny), ("lon", c.nx)):
            ds.createDimension(n, s)
        ds.createVariable("Time", "f8", ("Time",))[:] = [0.0]
        ds.createVariable("depth", "f8", ("depth",))[:] = cen
        ds["depth"].units, ds["depth"].positive = "m", "down"
        ds.createVariable("lat", "f8", ("lat",))[:] = c.y / 1e3
        ds.createVariable("lon", "f8", ("lon",))[:] = c.x / 1e3
        for name, val in (("ptemp", T), ("salt", np.where(T == FILL, FILL, 35.0))):
            v = ds.createVariable(name, "f8", ("Time", "depth", "lat", "lon"), fill_value=FILL)
            v[0] = val
    c_dt, c_h = (c.dt if dt is None else dt), (c.hours if hours is None else hours)
    (out / "MOM_input").write_text(mom_input(c, coord, nu, c_dt, c_h))
    (out / "MOM_override").write_text("")
    (out / "diag_table").write_text(DIAG.format(name=case_name, out=min(c.out_s, c_h * 3600.0)))
    (out / "input.nml").write_text(NML)
    return out


def read(run_dir: Path, case_name: str) -> dict:
    """Series fields from prog.nc, with record 0 = MOM_IC.nc (the model's own initial state); layers surface first."""
    c = cases.get(case_name)
    run_dir = Path(run_dir)
    fields = {k: [] for k in ("t", "h", "T", "u", "v", "eta")}
    for fname, tvar in (("MOM_IC.nc", "Temp"), ("prog.nc", "temp")):
        with netCDF4.Dataset(run_dir / fname) as ds:
            unit = ds["Time"].units.split()[0].lower()            # 'seconds since …' with TIMEUNIT = 1, else 'days'
            t = np.asarray(ds["Time"][:], float) * {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[unit]
            for n in range(t.size):
                fields["t"].append(t[n] if fname == "prog.nc" else 0.0)
                fields["h"].append(np.asarray(ds["h"][n], float))
                fields["T"].append(np.asarray(ds[tvar][n].filled(np.nan) if np.ma.isMA(ds[tvar][n])
                                              else ds[tvar][n], float))
                fields["u"].append(np.asarray(np.ma.filled(ds["u"][n], 0.0), float))
                fields["v"].append(np.asarray(np.ma.filled(ds["v"][n], 0.0), float))
                fields["eta"].append(np.asarray(np.ma.filled(ds["SSH"][n], 0.0), float) if "SSH" in ds.variables
                                     else np.zeros((c.ny, c.nx)))
    s = {k: np.array(v) for k, v in fields.items()}
    # η = Σh − H: MOM6's SSH diagnostic differs from the layer sum by up to ~2 mm (it is not the layer column), and
    # the metrics need the geometry the layers actually have
    s["eta"] = s["h"].sum(1) - np.nan_to_num(c.H)[None]
    s.update(x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H)
    return s


def wall_s(run_dir: Path) -> float:
    """Main-loop time [s] of the FMS clock table in stdout.txt."""
    from validation.mom6_bayiw.read_mom6 import clock_table
    return clock_table(Path(run_dir) / "stdout.txt")["Main loop"]
