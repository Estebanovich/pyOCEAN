"""
MITgcm runs of the four-model benchmark (spec §A2): z* (select_rStar = 2, nonlinFreeSurf = 4, NONLIN_FRSURF) with
partial cells (hFacMin = 0.1), implicit free surface, linear EOS in T (rhoNil = rhoConst = 1000, uniform tRef,
find_rho.F:98-108), OS7MP tracers (tempAdvScheme = 7, staggerTimeStep; OLx = OLy = 4, gad_check.F:115-128), its default
flux-form momentum, Laplacian viscAh, viscAr = 1e-4, no tracer diffusion, free slip (defaults are no-slip,
set_defaults.F:132-133), no drag, g = 9.81. Walls from land: one land column east (and one land row north in 3-D),
because the exchanges are periodic. Output: MNC diagnostics snapshots (THETA, UVEL, VVEL, hFactorC, ETAN); the cell
thickness under z* is hFactorC·drF (diagnostics_fill_state.F:451-453). t = 0 comes from the inputs (hFacC of the
grid file, theta.bin, η = 0). Copies of verification/advect_xz/code/{CPP_OPTIONS.h, DIAGNOSTICS_SIZE.h, GAD_OPTIONS.h}
(NONLIN_FRSURF defined).
"""
from __future__ import annotations

import glob
from pathlib import Path

import netCDF4
import numpy as np

from validation.ilicak_benchmark import cases

MIT = Path("/Volumes/ESTEBAN_MIT/MITgcm")


def _shape(case: cases.Case) -> tuple[int, int]:
    """(Nx, Ny) with the land column (and row in 3-D)."""
    return case.nx + 1, (case.ny + 1 if case.ny > 1 else 1)


def size_h(case: cases.Case) -> str:
    nx, ny = _shape(case)
    return f"""C SIZE.h of the pyOCEAN four-model benchmark ({case.name})
      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
      PARAMETER (
     &           sNx = {nx:3d},
     &           sNy = {ny:3d},
     &           OLx =   4,
     &           OLy =   4,
     &           nSx =   1,
     &           nSy =   1,
     &           nPx =   1,
     &           nPy =   1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = {case.nz:3d})
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx, MAX_OLY = OLy )
"""


def write_build(case_name: str, code_dir: Path) -> Path:
    c = cases.get(case_name)
    d = Path(code_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SIZE.h").write_text(size_h(c))
    (d / "packages.conf").write_text("gfd\ndiagnostics\nmnc\n")
    src = MIT / "verification" / "advect_xz" / "code"
    for f in ("CPP_OPTIONS.h", "DIAGNOSTICS_SIZE.h", "GAD_OPTIONS.h"):
        (d / f).write_text((src / f).read_text())
    return d


def data(case: cases.Case, nu: float, dt: float, hours: float) -> str:
    nx, ny = _shape(case)
    dzs = ", ".join(f"{z:.10g}" for z in case.dz) if np.ptp(case.dz) else f"{case.nz}*{float(case.dz[0])}"
    return f""" &PARM01
 tRef={case.nz}*{cases.T_REF}, sRef={case.nz}*35., eosType='LINEAR', tAlpha=2.E-4, sBeta=0.,
 rhoNil=1000., rhoConst=1000., gravity=9.81,
 viscAh={float(nu)}, viscAr=1.E-4, implicitViscosity=.TRUE.,
 no_slip_sides=.FALSE., no_slip_bottom=.FALSE., bottomDragLinear=0., bottomDragQuadratic=0.,
 diffKhT=0., diffKrT=0., saltStepping=.FALSE.,
 tempAdvScheme=7, staggerTimeStep=.TRUE., f0={case.f0}, beta=0.,
 implicitFreeSurface=.TRUE., exactConserv=.TRUE., select_rStar=2, nonlinFreeSurf=4,
 hFacInf=0.2, hFacSup=2.0, hFacMin=0.1,
 readBinaryPrec=64, writeBinaryPrec=64,
 &
 &PARM02
 cg2dMaxIters=1000, cg2dTargetResidual=1.E-13,
 &
 &PARM03
 nIter0=0, deltaT={float(dt)}, endTime={hours * 3600.0}, abEps=0.1,
 dumpFreq=0., chkptFreq=0., pChkptFreq=0., monitorFreq={min(case.out_s, hours * 3600.0)},
 &
 &PARM04
 usingCartesianGrid=.TRUE., delX={nx}*{float(case.dx[0])}, delY={ny}*{float(case.dy[0])}, delR={dzs},
 &
 &PARM05
 bathyFile='bathy.bin', hydrogThetaFile='theta.bin',
 &
"""


def write_run(case_name: str, nu: float, run_dir: Path, dt: float | None = None, hours: float | None = None) -> Path:
    c = cases.get(case_name)
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    nx, ny = _shape(c)
    dt = c.dt if dt is None else dt
    hours = c.hours if hours is None else hours
    (d / "data").write_text(data(c, nu, dt, hours))
    (d / "data.pkg").write_text(" &PACKAGES\n useDiagnostics=.TRUE., useMNC=.TRUE.,\n &\n")
    (d / "eedata").write_text(" &EEPARMS\n nTx=1, nTy=1,\n &\n")
    out = min(c.out_s, hours * 3600.0)
    (d / "data.diagnostics").write_text(f""" &DIAGNOSTICS_LIST
  fields(1:4,1) = 'THETA   ','UVEL    ','VVEL    ','hFactorC',
  fileName(1) = 'diag3d', frequency(1) = -{out}, timePhase(1) = 0.,
  fields(1,2) = 'ETAN    ', fileName(2) = 'diag2d', frequency(2) = -{out}, timePhase(2) = 0.,
 &
 &DIAG_STATIS_PARMS
 &
""")
    (d / "data.mnc").write_text(" &MNC_01\n mnc_use_outdir=.FALSE., mnc_outdir_str='mnc_',\n &\n")
    bathy = np.zeros((ny, nx))
    bathy[:c.ny, :c.nx] = -np.nan_to_num(c.H)                          # negative depth, 0 on land
    np.ascontiguousarray(bathy, ">f8").tofile(d / "bathy.bin")
    xs = np.concatenate([c.x, [c.x[-1] + c.dx[-1]]])
    ys = np.concatenate([c.y, [c.y[-1] + c.dy[-1]]])[:ny]
    T = c.T_fn(xs[None, None, :], ys[None, :, None], c.zc[:, None, None])
    np.ascontiguousarray(T, ">f8").tofile(d / "theta.bin")
    return d


def _join(pattern: str, name: str) -> np.ndarray:
    """One MNC variable (single tile in this setup)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    with netCDF4.Dataset(files[0]) as ds:
        return np.asarray(ds[name][:], float)


def read(run_dir: Path, case_name: str) -> dict:
    """Series fields: diagnostics snapshots after a t = 0 record built from the inputs; land dropped; surface first."""
    c = cases.get(case_name)
    d = Path(run_dir)
    nx, ny = c.nx, c.ny
    t = _join(str(d / "diag3d.*.t001.nc"), "T")
    T = _join(str(d / "diag3d.*.t001.nc"), "THETA")[:, :, :ny, :nx]
    U = _join(str(d / "diag3d.*.t001.nc"), "UVEL")[:, :, :ny, :nx + 1]
    V = _join(str(d / "diag3d.*.t001.nc"), "VVEL")[:, :, :ny + 1, :nx]
    hf = _join(str(d / "diag3d.*.t001.nc"), "hFactorC")[:, :, :ny, :nx]
    eta = _join(str(d / "diag2d.*.t001.nc"), "ETAN")[:, :ny, :nx]
    drf = _join(str(d / "grid.t001.nc"), "drF")
    hf0 = _join(str(d / "grid.t001.nc"), "HFacC")[:, :ny, :nx]
    Nx, Ny = _shape(c)
    T0 = np.fromfile(d / "theta.bin", ">f8").reshape(c.nz, Ny, Nx)[:, :ny, :nx]
    if ny > 1:
        V[:, :, -1] = 0.0                                                  # the north wall face
    U[:, :, :, -1] = 0.0                                                   # the east wall face
    h = np.concatenate([(hf0 * drf[:, None, None])[None], hf * drf[None, :, None, None]])
    Tall = np.concatenate([T0[None], T])
    s = dict(t=np.concatenate([[0.0], t]), h=h, T=np.where(h > 0, Tall, np.nan),
             u=np.concatenate([np.zeros((1,) + U.shape[1:]), U]), v=np.concatenate([np.zeros((1,) + V.shape[1:]), V]),
             eta=np.concatenate([np.zeros((1, ny, nx)), eta]))
    s.update(x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=(hf0 * drf[:, None, None]).sum(0))
    return s


def wall_s(run_dir: Path) -> float:
    """Wall clock of section "MAIN_DO_LOOP [THE_MAIN_LOOP]" in output.txt."""
    lines = (Path(run_dir) / "output.txt").read_text().splitlines()
    for i, line in enumerate(lines):
        if 'Seconds in section "MAIN_DO_LOOP' in line:
            for nxt in lines[i:i + 6]:
                if "Wall clock time" in nxt:
                    return float(nxt.split(":")[-1])
    raise ValueError("no MAIN_DO_LOOP timing in output.txt")
