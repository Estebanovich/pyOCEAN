"""
MOM6 namelists for the BayIW benchmark (spec 2026-10-13-mom6-bayiw-benchmark-design §2): the physics of Esteban's
MITgcm runs (linear EOS, Laplacian 100 m² s⁻¹, vertical 1e-5, free slip, no bottom drag, f-plane, Arakawa KE),
ALE z* / HYCOM1 / HYBGEN with PPM_H4 remapping, z-data initial state with ALE remapping, file wind stress every
1200 s, radiating open edges (Flather + Orlanski) toward a resting exterior.  Parameter names and file:line in
src/ of ~/pyOCEAN_refs/MOM6 are in the plan's Global Constraints.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from validation.bayiw_baroclinic import bayiw
from validation.mom6_bayiw import inputs

COORDS = {"zstar": "ZSTAR", "hycom1": "HYCOM1", "hybgen": "HYBGEN"}
OBC_DATA = {"south": '"V=value:0.0,SSH=value:0.0"', "west": '"U=value:0.0,SSH=value:0.0"',
            "east": '"U=value:0.0,SSH=value:0.0"'}               # normal velocity and SSH: a resting exterior
SEGMENTS = {"south": '"J=0,I=0:N,FLATHER,ORLANSKI"', "west": '"I=0,J=N:0,FLATHER,ORLANSKI"',
            "east": '"I=N,J=0:N,FLATHER,ORLANSKI"'}


def _list(a) -> str:
    return ", ".join(f"{x:.6g}" for x in np.asarray(a, float))


def mom_input(kind: str, case: str, coord: str, nx: int, ny: int, hours: float, dt: float,
              layout: tuple[int, int], targets: np.ndarray | None = None) -> str:
    nk = 50 if coord == "zstar" else 25
    edges = ("south",) if kind == "section" else ("south", "west", "east")
    L = [f"NIGLOBAL = {nx}", f"NJGLOBAL = {ny}", f"NK = {nk}", "REENTRANT_X = False", "REENTRANT_Y = False",
         f"LAYOUT = {layout[0]}, {layout[1]}", "IO_LAYOUT = 1, 1",
         'INPUTDIR = "INPUT"', 'GRID_CONFIG = "mosaic"', 'GRID_FILE = "ocean_hgrid.nc"',
         'TOPO_CONFIG = "file"', 'TOPO_FILE = "topog.nc"', 'TOPO_VARNAME = "depth"',
         "MINIMUM_DEPTH = 1.0", "MASKING_DEPTH = 0.0", f"MAXIMUM_DEPTH = {float(np.sum(bayiw.DELR)):.6g}",
         'ROTATION = "betaplane"', f"F_0 = {bayiw.F0:.6E}", "BETA = 0.0",
         "G_EARTH = 9.81", "ENABLE_BUGS_BY_DEFAULT = False",                  # MITgcm/pyOCEAN g; MOM6 default 9.8
         # OBC T/S filled from the interior at start-up (no TEMP/SALT segment data: the exterior is the initial
         # profile at rest); these three "bug" behaviours are kept on purpose (MOM_open_boundary.F90:764-770)
         "OBC_TEMP_SALT_NEEDED_BUG = True", "OBC_RESERVOIR_INIT_BUG = True", "OBC_TS_RESERVOIR_INIT_BUG = True",
         'EQN_OF_STATE = "LINEAR"', f"RHO_0 = {bayiw.RHO0}", f"RHO_REF_LINEAR_EOS = {bayiw.RHO0}",
         f"T_REF_LINEAR_EOS = {bayiw.CASES[case]['t_ref']}", f"DRHO_DT = {-bayiw.T_COEF:.6g}", "DRHO_DS = 0.0",
         "USE_REGRIDDING = True", f'REGRIDDING_COORDINATE_MODE = "{COORDS[coord]}"', 'REMAPPING_SCHEME = "PPM_H4"']
    if coord == "zstar":
        L += ['ALE_COORDINATE_CONFIG = "PARAM"', "ALE_RESOLUTION = " + _list(bayiw.DELR)]
    elif coord == "hycom1":
        L += ['ALE_COORDINATE_CONFIG = "HYBRID:hyb.nc,sigma2,dz"']
    else:
        # layer densities: COORD_CONFIG "linear" (MOM_coord_initialization.F90:526, Rlay at the midpoints of
        # evenly spaced interfaces); "file" fails in this FMS ("FMS I/O requires a domain input")
        t = np.asarray(targets, float)
        if not np.allclose(np.diff(t), np.diff(t)[0], rtol=1e-9, atol=0.0):
            raise ValueError("HYBGEN here needs evenly spaced targets (the two-layer centred targets)")
        dz = _list(bayiw.hybrid_dz_top(25))
        L += ['COORD_CONFIG = "linear"', f"LIGHTEST_DENSITY = {t[0]:.10g}", f"DENSITY_RANGE = {t[-1] - t[0]:.10g}",
              f"HYBGEN_DEEP_DZ_PR0FILE = {dz}", f"HYBGEN_SHALLOW_DZ_PR0FILE = {dz}"]
    L += ["INIT_LAYERS_FROM_Z_FILE = True", 'TEMP_SALT_Z_INIT_FILE = "ts_z.nc"', 'Z_INIT_FILE_PTEMP_VAR = "ptemp"',
          'Z_INIT_FILE_SALT_VAR = "salt"', "Z_INIT_ALE_REMAPPING = True", 'Z_INIT_REMAPPING_SCHEME = "PPM_H4"',
          "TEMP_SALT_INIT_VERTICAL_REMAP_ONLY = True"]
    if coord != "zstar":
        L += ["Z_INIT_REMAP_GENERAL = True", "Z_INIT_REMAP_FULL_COLUMN = True"]
    L += ['KE_SCHEME = "KE_ARAKAWA"', "LAPLACIAN = True", "KH = 100.0", "BIHARMONIC = False", "NOSLIP = False",
          "KV = 1.0E-05", "KD = 1.0E-05", "KHTR = 100.0", "BOTTOMDRAGLAW = False", "HBBL = 1.0",
          "HMIX_FIXED = 1.0", "KV_ML_INVZ2 = 0.0", "DIRECT_STRESS = False", 'BUOY_CONFIG = "zero"',
          'WIND_CONFIG = "file"', 'WIND_FILE = "wind.nc"', 'WIND_STAGGER = "C"', "WIND_FILE_DAYS_PER_RECORD = -72",
          f"DT = {dt}", f"DT_THERM = {dt}", "DTBT = -0.98", "TIMEUNIT = 3600.0", f"DAYMAX = {hours}",
          "ENERGYSAVEDAYS = 0.25", f"OBC_NUMBER_OF_SEGMENTS = {len(edges)}"]
    for n, e in enumerate(edges, 1):
        L += [f"OBC_SEGMENT_{n:03d} = {SEGMENTS[e]}", f"OBC_SEGMENT_{n:03d}_DATA = {OBC_DATA[e]}"]
    return "\n".join(L) + "\n"


DIAG = '''"BayIW {kind}"
1 1 1 0 0 0
"prog", 900, "seconds", 1, "days", "Time",
"ocean_model", "SSH", "SSH", "prog", "all", .false., "none", 2
"ocean_model", "temp", "temp", "prog", "all", .false., "none", 2
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


def write_case(kind: str, case: str, bay: str, coord: str, out_dir: Path, hours: float, dt: float = 120.0,
               layout: tuple[int, int] = (1, 5)) -> Path:
    """A ready-to-run MOM6 directory: MOM_input, MOM_override (empty), diag_table, input.nml, INPUT/, RESTART/."""
    out = Path(out_dir)
    (out / "RESTART").mkdir(parents=True, exist_ok=True)
    inputs.write_inputs(kind, case, bay, out)
    d = inputs.domain(kind, case, bay)
    ny, nx = d["H"].shape
    tgt = inputs.targets(case, d["T"], d["H"])
    (out / "MOM_input").write_text(mom_input(kind, case, coord, nx, ny, hours, dt, layout, tgt))
    (out / "MOM_override").write_text("")
    (out / "diag_table").write_text(DIAG.format(kind=kind))
    (out / "input.nml").write_text(NML)
    return out
