"""
CROCO runs of the four-model benchmark (spec §A2): σ coordinate, grid and initial state read from NetCDF (no
analytic routine is edited), own case keys PYO_* in a run-dir cppdefs.h/param.h (they override the source copies,
jobcomp), the REGIONAL default schemes (UV_HADV_UP3, UV_VADV_SPLINES, TS_HADV_RSUP3, TS_VADV_SPLINES; RSUP3 carries
an explicit biharmonic diffusion |u|Δx³/12, cppdefs_dev.h:423-436 — class c), constant Laplacian ν_h along σ
(UV_VIS2 + UV_MIX_S, no Smagorinsky), Akv = 1e-4 and Akt = 0, no drag, free slip (gamma2 = 1), linear EOS
ρ = 1000 + R0 − Tcoef (T − T0) with R0 = 0, T0 = T_ref, Tcoef = 0.2, g = 9.81 (scalars.h:633).
Uniform σ needs NEW_S_COORD with θs = θb = 0 and hc = 1e16 (set_scoord.F:169-184, set_depth.F:158-189).
RESET_RHO0 is forced for a linear EOS (cppdefs_dev.h:264-270): ρ₀ = 1000 + volume mean of σ_t (~1e-3 of g/ρ₀).
"""
from __future__ import annotations

import re
from pathlib import Path

import netCDF4
import numpy as np

from validation.ilicak_benchmark import cases

KEYS = {"lock": "PYO_LOCK", "overflow": "PYO_OVERFLOW", "iw": "PYO_IW", "seamount": "PYO_SEAMOUNT"}
SRC = Path.home() / "pyOCEAN_refs/croco/OCEAN"
JOBCOMP = Path.home() / "pyOCEAN_refs/croco_runs/grav_adj/jobcomp"

_BLOCK = """#elif defined PYO_LOCK || defined PYO_OVERFLOW || defined PYO_IW || defined PYO_SEAMOUNT
/* pyOCEAN four-model benchmark (validation/ilicak_benchmark/croco.py) */
# undef  OPENMP
# undef  MPI
# define SOLVE3D
# define NEW_S_COORD
# define UV_ADV
{cor}
# define UV_VIS2
# undef  UV_VIS_SMAGO
# define UV_MIX_S
# define UV_HADV_UP3
# define UV_VADV_SPLINES
# define TS_HADV_RSUP3
# define TS_VADV_SPLINES
# undef  TS_DIF2
# undef  TS_DIF4
# undef  SALINITY
# undef  NONLIN_EOS
# undef  LMD_MIXING
# undef  GLS_MIXING
# undef  MASKING
# undef  ANA_GRID
# undef  ANA_INITIAL
# define ANA_SMFLUX
# define ANA_STFLUX
# define ANA_BTFLUX
# define NO_FRCFILE
# define OUT_DOUBLE
"""


def cppdefs(case_name: str, src: str) -> str:
    """The run-dir cppdefs.h: REGIONAL off, the case key on, the PYO_* block before the end of the choices."""
    cor = "# define UV_COR" if case_name == "seamount" else "# undef  UV_COR"
    out = re.sub(r"^#define REGIONAL\b.*$", f"#undef REGIONAL\n#define {KEYS[case_name]}", src, count=1, flags=re.M)
    anchor = "#endif /* END OF CONFIGURATION CHOICE */"
    if anchor not in out:
        raise ValueError("cppdefs.h: end-of-choices anchor not found")
    return out.replace(anchor, _BLOCK.format(cor=cor) + anchor, 1)


def param(case_name: str, src: str) -> str:
    """The run-dir param.h with the PYO_* grid sizes before the REGIONAL entry."""
    c = cases.get(case_name)
    add = f"#elif defined {KEYS[case_name]}\n      parameter (LLm0={c.nx}, MMm0={c.ny}, N={c.nz})\n"
    anchor = "#elif defined REGIONAL"
    if anchor not in src:
        raise ValueError("param.h: REGIONAL anchor not found")
    return src.replace(anchor, add + anchor, 1)


def _ndtfast(case: cases.Case, dt: float) -> int:
    """Barotropic substeps: explicit 2-D step with √(gH) Δt_fast / Δx ≤ 0.1."""
    c = np.sqrt(cases.G * float(np.nanmax(case.H)))
    return max(10, int(np.ceil(dt * c / (0.1 * float(case.dx.min())))))


def croco_in(case: cases.Case, nu: float, dt: float, hours: float) -> str:
    """The croco.in text (keywords of read_inp.F; every keyword the CPP keys enable must be present)."""
    n = int(round(hours * 3600.0 / dt))
    nwrt = max(1, int(round(min(case.out_s, hours * 3600.0) / dt)))
    return "\n".join([
        "title:", f"        pyOCEAN benchmark {case.name}",
        "time_stepping: NTIMES dt[sec] NDTFAST NINFO", f"        {n} {float(dt)} {_ndtfast(case, dt)} {nwrt}",
        "S-coord: THETA_S, THETA_B, Hc (m)", "        0.d0 0.d0 1.e16",
        "grid: filename", "        croco_grd.nc",
        "initial: NRREC filename", "        1", "        croco_ini.nc",
        "restart: NRST, NRPFRST / filename", f"        {n} -1", "        croco_rst.nc",
        "history: LDEFHIS, NWRT, NRPFHIS / filename", f"        T {nwrt} 0", "        croco_his.nc",
        "primary_history_fields: zeta UBAR VBAR U V wrtT(1:NT)", "        T T T T T T",
        "auxiliary_history_fields: rho Omega W Akv Akt Aks Visc3d Diff3d HBL HBBL Bostr Bustr Bvstr Wstr Ustr Vstr "
        "Shfl rsw rlw lat sen Hm HEL BIOL", "        F F F F F F F F F F F F F 20*F",     # as croco.in.Grav_adj (≥ 33 items)
        "rho0:", "        1000.",
        "bottom_drag: RDRG(m/s), RDRG2, Zob [m], Cdb_min, Cdb_max", "        0. 0. 0. 0. 0.",
        "gamma2:", "        1.",
        "lin_EOS_cff: R0 [kg/m3], T0 [Celsius], S0 [PSU], TCOEF [1/Celsius], SCOEF [1/PSU]",
        f"        0. {cases.T_REF} 0. {cases.ALPHA} 0.",
        "vertical_mixing: Akv_bak, Akt_bak [m^2/sec]", "        1.e-4 0.",
        "lateral_visc: VISC2, VISC4 [m^2/sec]", f"        {float(nu)} 0.",
        "tracer_diff4: TNU4(1:NT) [m^4/sec for all]", "        0.",          # RSUP3 implies TS_DIF4: only |u|Δx³/12
        ""])


def write_build(case_name: str, build_dir: Path) -> Path:
    """jobcomp + run-dir cppdefs.h and param.h for one case (a build per grid: CROCO sizes are compile-time)."""
    d = Path(build_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "jobcomp").write_text(JOBCOMP.read_text())
    (d / "jobcomp").chmod(0o755)
    (d / "cppdefs.h").write_text(cppdefs(case_name, (SRC / "cppdefs.h").read_text()))
    (d / "param.h").write_text(param(case_name, (SRC / "param.h").read_text()))
    return d


def _pad(a: np.ndarray) -> np.ndarray:
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")


def write_run(case_name: str, nu: float, run_dir: Path, dt: float | None = None, hours: float | None = None) -> Path:
    """croco_grd.nc, croco_ini.nc (T at the uniform σ-layer centres, bottom first) and croco.in."""
    c = cases.get(case_name)
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    M, L, N = c.ny + 2, c.nx + 2, c.nz
    xr = np.concatenate([[c.x[0] - c.dx[0]], c.x, [c.x[-1] + c.dx[-1]]])
    yr = np.concatenate([[c.y[0] - c.dy[0]], c.y, [c.y[-1] + c.dy[-1]]])
    H = _pad(np.nan_to_num(c.H))
    with netCDF4.Dataset(d / "croco_grd.nc", "w") as ds:
        for n, sz in (("xi_rho", L), ("eta_rho", M), ("xi_u", L - 1), ("eta_v", M - 1), ("one", 1)):
            ds.createDimension(n, sz)
        ds.createVariable("spherical", "S1", ("one",))[:] = np.array([b"F"])
        ds.createVariable("xl", "f8", ())[:] = float(c.dx.sum())
        ds.createVariable("el", "f8", ())[:] = float(c.dy.sum())
        for name, val in (("h", H), ("f", np.full((M, L), c.f0)), ("pm", np.full((M, L), 1.0 / float(c.dx[0]))),
                          ("pn", np.full((M, L), 1.0 / float(c.dy[0]))),
                          ("x_rho", np.broadcast_to(xr[None], (M, L))), ("y_rho", np.broadcast_to(yr[:, None], (M, L)))):
            ds.createVariable(name, "f8", ("eta_rho", "xi_rho"))[:] = val
    sc = (np.arange(N) + 0.5) / N - 1.0                                 # σ-layer centres, bottom first, uniform
    z = sc[:, None, None] * H[None]                                      # η = 0
    T = c.T_fn(xr[None, None, :], yr[None, :, None], z)
    with netCDF4.Dataset(d / "croco_ini.nc", "w") as ds:
        for n, sz in (("time", None), ("s_rho", N), ("eta_rho", M), ("xi_rho", L), ("xi_u", L - 1), ("eta_v", M - 1)):
            ds.createDimension(n, sz)
        t = ds.createVariable("scrum_time", "f8", ("time",))
        t[:] = [0.0]
        t.units = "second"
        ds.createVariable("zeta", "f8", ("time", "eta_rho", "xi_rho"))[:] = 0.0
        ds.createVariable("ubar", "f8", ("time", "eta_rho", "xi_u"))[:] = 0.0
        ds.createVariable("vbar", "f8", ("time", "eta_v", "xi_rho"))[:] = 0.0
        ds.createVariable("u", "f8", ("time", "s_rho", "eta_rho", "xi_u"))[:] = 0.0
        ds.createVariable("v", "f8", ("time", "s_rho", "eta_v", "xi_rho"))[:] = 0.0
        ds.createVariable("temp", "f8", ("time", "s_rho", "eta_rho", "xi_rho"))[0] = T
    (d / "croco.in").write_text(croco_in(c, nu, c.dt if dt is None else dt, c.hours if hours is None else hours))
    return d


def read(run_dir: Path, case_name: str) -> dict:
    """Series fields of croco_his.nc: ghosts dropped, layers flipped to surface first, h from z_w (Vtransform 2:
    z_w = ζ + (ζ + h)(hc s_w + h Cs_w)/(hc + h), set_depth.F:158-189)."""
    c = cases.get(case_name)
    with netCDF4.Dataset(Path(run_dir) / "croco_his.nc") as ds:
        t = np.asarray(ds["scrum_time"][:], float)
        hb = np.asarray(ds["h"][:], float)[1:-1, 1:-1]
        sw, csw, hc = (np.asarray(ds[k][:], float) for k in ("s_w", "Cs_w", "hc"))
        zeta = np.asarray(ds["zeta"][:], float)[:, 1:-1, 1:-1]
        T = np.asarray(ds["temp"][:], float)[:, ::-1, 1:-1, 1:-1]
        u = np.asarray(ds["u"][:], float)[:, ::-1, 1:-1, :]
        v = np.asarray(ds["v"][:], float)[:, ::-1, :, 1:-1]
    zw = zeta[:, None] + (zeta[:, None] + hb[None, None]) * (
        (hc * sw[None, :, None, None] + hb[None, None] * csw[None, :, None, None]) / (hc + hb[None, None]))
    h = np.diff(zw, axis=1)[:, ::-1]
    return dict(t=t, x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H, h=h, T=T, u=u, v=v, eta=zeta)


def wall_s(run_dir: Path) -> float:
    """'real' seconds of /usr/bin/time -p (time.txt): CROCO prints no wall clock without MPI (timers_roms.F:40-77)."""
    for line in (Path(run_dir) / "time.txt").read_text().splitlines():
        if line.startswith("real"):
            return float(line.split()[1])
    raise ValueError("no 'real' line in time.txt")
