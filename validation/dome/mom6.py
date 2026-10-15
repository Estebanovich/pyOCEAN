"""
MOM6 ρ runs of the DOME benchmark (spec 2026-10-14-dome-overflow §6.1): the ocean_only/DOME example of MOM6-examples
(layer mode, 25 isopycnal layers R_k = 1030 + (k − 1)·2/24, no thermodynamics; DOME topography, inflow
DOME_set_OBC_data, sponges DOME_initialize_sponges and dye tr_D1 of src/user/DOME_initialization.F90) with the common
physics of the case written in MOM_override: Laplacian KH (Legg's z-model, ∝ Δx), no biharmonic/Smagorinsky,
KV = 0.5 (variant 'kv_low': 1e-4), KD = 0, no Jackson shear mixing, CDRAG 0.002 without background velocity,
HBBL 10 m, 50 days.  The example's MOM_input is copied unchanged (read only).
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from validation.dome import cases

EXAMPLE = Path.home() / "pyOCEAN_refs" / "MOM6-examples" / "ocean_only" / "DOME"
# parameters the example's MOM_input already sets (→ '#override'); the others are plain lines (MOM6 FATALs on an
# #override of a parameter that was never set)
SET_IN_EXAMPLE = {"NIGLOBAL", "NJGLOBAL", "DT", "DT_THERM", "DAYMAX", "SMAGORINSKY_AH", "AH_VEL_SCALE", "KV", "KD",
                  "USE_JACKSON_PARAM", "CDRAG", "DRAG_BG_VEL", "HBBL", "OBC_SEGMENT_001"}

DIAG = '''"DOME pyOCEAN benchmark"
1 1 1 0 0 0
"prog", 1, "days", 1, "days", "Time",
"ocean_model", "h", "h", "prog", "all", .false., "none", 2
"ocean_model", "u", "u", "prog", "all", .false., "none", 2
"ocean_model", "v", "v", "prog", "all", .false., "none", 2
"ocean_model", "tr_D1", "tr_D1", "prog", "all", .false., "none", 2
"ocean_model", "SSH", "SSH", "prog", "all", .false., "none", 2
'''

NML = ''' &MOM_input_nml
         output_directory = './',
         input_filename = 'n'
         restart_input_dir = 'INPUT/',
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


def overrides(c: cases.Case, variant: str) -> dict:
    """Parameter → value of the run (plain, without #override)."""
    kv = {"common": c.nu_v, "kv_low": 1e-4}[variant]
    i0, i1 = int(round(1000e3 / c.dx)), int(round(1100e3 / c.dx))       # channel q-points (DOME_INFLOW_LON/WIDTH)
    return {"NIGLOBAL": c.nx, "NJGLOBAL": c.ny, "DT": c.dt, "DT_THERM": c.dt, "DAYMAX": 50.0, "LAPLACIAN": "True",
            "KH": f"{c.nu_h:g}", "BIHARMONIC": "False", "SMAGORINSKY_AH": "False", "AH_VEL_SCALE": 0.0, "KV": kv,
            "KD": 0.0, "USE_JACKSON_PARAM": "False", "CDRAG": 0.002, "DRAG_BG_VEL": 0.0, "HBBL": 10.0,
            "OBC_SEGMENT_001": f'"J=N,I={i1}:{i0},SIMPLE"', "LAYOUT": "1, 1", "IO_LAYOUT": "1, 1",
            "SAVE_INITIAL_CONDS": "True", 'IC_OUTPUT_FILE': '"MOM_IC"'}


def override_lines(c: cases.Case, variant: str) -> list[str]:
    """MOM_override lines: '#override ' only for parameters the example's MOM_input already sets."""
    return [("#override " if k in SET_IN_EXAMPLE else "") + f"{k} = {v}" for k, v in overrides(c, variant).items()]


def mom_input(c: cases.Case, variant: str = "common") -> str:
    """The example's MOM_input followed by this run's parameters as plain lines (what the run uses; for checks)."""
    return (EXAMPLE / "MOM_input").read_text() + "\n" + "\n".join(
        l.removeprefix("#override ") for l in override_lines(c, variant)) + "\n"


def write_case(res_km: float, out_dir: Path, variant: str = "common", days: float | None = None) -> Path:
    """MOM_input (example, unchanged), MOM_override, diag_table, input.nml, INPUT/ and RESTART/ in ``out_dir``."""
    c = cases.get(res_km)
    d = Path(out_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "INPUT").mkdir(exist_ok=True)
    (d / "RESTART").mkdir(exist_ok=True)
    (d / "MOM_input").write_text((EXAMPLE / "MOM_input").read_text())
    lines = override_lines(c, variant)
    if days is not None:
        lines = [l if not l.startswith("#override DAYMAX") else f"#override DAYMAX = {float(days)}" for l in lines]
    (d / "MOM_override").write_text("\n".join(lines) + "\n")
    (d / "diag_table").write_text(DIAG)
    (d / "input.nml").write_text(NML)
    return d


def read(run_dir: Path, res_km: float) -> dict:
    """
    Series fields from prog.nc (daily) with record 0 = MOM_IC.nc; layers surface first (MOM6 k = 1 is the top).
    T is the temperature of the layer's target density R_k with the case's linear EOS (layer mode: no T field);
    layers thinner than 1 mm get NaN T.  s = tr_D1 (0 at t = 0 if MOM_IC holds no tracer).  η = Σh − H.
    """
    c = cases.get(res_km)
    run_dir = Path(run_dir)
    R = cases.RHO_TOP + np.arange(c.nz) * cases.DRHO / (c.nz - 1)
    Tk = cases.T_of_rho(R)[:, None, None]
    out = {k: [] for k in ("t", "h", "s", "u", "v")}
    for fname in ("MOM_IC.nc", "prog.nc"):
        with netCDF4.Dataset(run_dir / fname) as ds:
            unit = ds["Time"].units.split()[0].lower()
            t = np.asarray(ds["Time"][:], float) * {"seconds": 1.0, "hours": 3600.0, "days": 86400.0}[unit]
            for n in range(t.size):
                out["t"].append(t[n] if fname == "prog.nc" else 0.0)
                out["h"].append(np.asarray(np.ma.filled(ds["h"][n], 0.0), float))
                tr = ds.variables.get("tr_D1")
                out["s"].append(np.zeros_like(out["h"][-1]) if tr is None else np.asarray(np.ma.filled(tr[n], 0.0), float))
                out["u"].append(np.asarray(np.ma.filled(ds["u"][n], 0.0), float))
                out["v"].append(np.asarray(np.ma.filled(ds["v"][n], 0.0), float))
    s = {k: np.array(v) for k, v in out.items()}
    land = ~np.isfinite(c.H)[None, None]
    s["h"] = np.where(land, 0.0, s["h"])
    s["T"] = np.where((s["h"] > 1e-3) & ~land, Tk[None] * np.ones_like(s["h"]), np.nan)
    if np.nanmax(s["s"]) < 1e-10:
        # the example's dye never enters: DOME_set_OBC_data fills segment%field(1)%buffer_src but register_segment_tracer
        # leaves the segment value at 0 (MOM_open_boundary.F90:5390-5400; class (d), 2026-10-14). With KD = 0 and no
        # Jackson mixing the layers keep their density, so the inflow water of the two densest classes (R ≥ 1031.92,
        # 86 % of the inflow; absent over the slope at t = 0) is the thickness gained since t = 0: s = (h − h⁰)/h where
        # the gain exceeds 1 m (smaller gains are wave displacements of the deep layers) — proxy, see the README
        dense = (np.arange(c.nz) >= c.nz - 2)[None, :, None, None]
        gain = s["h"] - s["h"][:1]
        s["s"] = np.where(dense & (gain > 1.0), gain / np.where(s["h"] > 1e-3, s["h"], 1.0), 0.0)
    s["s"] = np.where(land, np.nan, s["s"])
    s["eta"] = s["h"].sum(1) - np.nan_to_num(c.H)[None]
    s.update(x=c.x, y=c.y, dx=np.full(c.nx, c.dx), dy=np.full(c.ny, c.dx), H=c.H)
    return s


def wall_s(run_dir: Path) -> float:
    """Main-loop time [s] of the FMS clock table in stdout.txt."""
    from validation.mom6_bayiw.read_mom6 import clock_table
    return clock_table(Path(run_dir) / "stdout.txt")["Main loop"]
