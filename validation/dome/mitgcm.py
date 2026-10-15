"""
MITgcm z runs of the DOME benchmark (spec 2026-10-14-dome-overflow §6.2): the case of cases.py on 25 levels of 144 m
with partial cells (hFacMin = 0.1), z* (select_rStar = 2, nonlinFreeSurf = 4: the sea level rises ~35 m in 50 days),
linear EOS ρ = 1031 − 0.2 T (tAlpha = 0.2/1031), flux-limited tracers (tempAdvScheme = 77, as Legg et al. 2006,
GAD.h:41 ENUM_FLUX_LIMIT), vector-invariant momentum, Laplacian viscAh, viscAr = ν_v, quadratic bottom drag 2e-3,
no tracer diffusion.  Variant 'legg': no-slip bed and sides (Legg eq. 10, the MITgcm defaults of their runs);
'common': free slip + drag, as pyOCEAN.  Inflow: OBCS prescribed on the north row (OBNv/OBNt/OBNptr from
cases.inflow_profile, averaged onto the levels so that Σ v dz dx is the case transport); the dye is ptracer 1;
sponges: rbcs relaxation of T and the dye toward the ambient profile / 0 with MOM6 DOME's rates.
Grid: one land row south and one land column east (periodic exchanges) and the OBCS row north (MITgcm sets the
exterior values on its own row, so the open face is the south face of that row = pyOCEAN's north edge).
"""
from __future__ import annotations

import glob
from pathlib import Path

import netCDF4
import numpy as np

from validation.dome import cases

MIT = Path("/Volumes/ESTEBAN_MIT/MITgcm")
NR, DR = 25, 144.0


def _shape(c: cases.Case) -> tuple[int, int]:
    """(Nx, Ny): case + east land column; case + south land row + north OBCS row."""
    return c.nx + 1, c.ny + 2


def size_h(c: cases.Case) -> str:
    nx, ny = _shape(c)
    return f"""C SIZE.h of the pyOCEAN DOME benchmark ({c.name})
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
     &           Nr  = {NR:3d})
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx, MAX_OLY = OLy )
"""


def write_build(res_km: float, code_dir: Path) -> Path:
    """SIZE.h, packages.conf and the CPP options (advect_xz copies: NONLIN_FRSURF defined) in ``code_dir``."""
    c = cases.get(res_km)
    d = Path(code_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SIZE.h").write_text(size_h(c))
    (d / "packages.conf").write_text("gfd\nobcs\nptracers\nrbcs\ndiagnostics\nmnc\n")
    src = MIT / "verification" / "advect_xz" / "code"
    for f in ("CPP_OPTIONS.h", "DIAGNOSTICS_SIZE.h", "GAD_OPTIONS.h"):
        (d / f).write_text((src / f).read_text())
    return d


def wet_levels(c: cases.Case) -> np.ndarray:
    """Wet thickness of each level in the 600-m channel column [m] (Nr,)."""
    top = np.arange(NR) * DR
    return np.clip(np.minimum(cases.HE, top + DR) - top, 0.0, None)


def inflow_levels(c: cases.Case):
    """(v, T, s) (Nr, Nx) of the OBCS row: the 60-level case profile averaged onto the wet part of each level
    (conservative, so Σ v dz dx = transport); dry levels and land columns: v = s = 0, T ambient."""
    p = cases.inflow_profile(c)
    nx, _ = _shape(c)
    zt, zb = -p["zw"][:-1, 0], -p["zw"][1:, 0]                         # source depths (positive down)
    top = np.arange(NR) * DR
    w = np.clip(np.minimum(zb[None], np.minimum(top + DR, cases.HE)[:, None]) - np.maximum(zt[None], top[:, None]),
                0.0, None)                                              # overlap (Nr, nsrc)
    dz = w.sum(axis=1)
    avg = lambda f: np.where(dz[:, None] > 0, (w @ f) / np.where(dz > 0, dz, 1.0)[:, None], 0.0)   # noqa: E731
    zc = -(top + 0.5 * DR)
    out = []
    for name, fill in (("v", 0.0), ("T", None), ("s", 0.0)):
        f = np.zeros((NR, nx))
        f[:, :c.nx] = avg(p[name])
        dry = dz[:, None] <= 0
        if fill is None:
            amb = cases.T_ambient(zc)[:, None] * np.ones((1, nx))
            f = np.where(dry, amb, f)
            f[:, c.nx:] = amb[:, c.nx:]
        out.append(f)
    return tuple(out)


def data_text(c: cases.Case, variant: str, days: float | None = None) -> str:
    nx, ny = _shape(c)
    days = c.days if days is None else days
    dt = 600.0 * c.dx / 10e3
    noslip = {"legg": ".TRUE.", "common": ".FALSE."}[variant]
    return f""" &PARM01
 tRef={NR}*{cases.T_REF}, sRef={NR}*35., eosType='LINEAR', tAlpha={cases.ALPHA / cases.RHO0:.10g}, sBeta=0.,
 rhoNil={cases.RHO0}, rhoConst={cases.RHO0}, gravity={cases.G},
 viscAh={c.nu_h:g}., viscAr={c.nu_v:g}, implicitViscosity=.TRUE., vectorInvariantMomentum=.TRUE.,
 no_slip_sides={noslip}, no_slip_bottom={noslip}, bottomDragLinear=0., bottomDragQuadratic={c.cd:g},
 diffKhT=0., diffKrT=0., saltStepping=.FALSE.,
 tempAdvScheme=77, staggerTimeStep=.TRUE., f0={c.f0:g}, beta=0.,
 implicitFreeSurface=.TRUE., exactConserv=.TRUE., select_rStar=2, nonlinFreeSurf=4,
 hFacInf=0.2, hFacSup=2.0, hFacMin=0.1,
 readBinaryPrec=64, writeBinaryPrec=64,
 &
 &PARM02
 cg2dMaxIters=1000, cg2dTargetResidual=1.E-13,
 &
 &PARM03
 nIter0=0, deltaT={dt}, endTime={days * 86400.0 + dt}, abEps=0.1,
 dumpFreq=0., chkptFreq=0., pChkptFreq=0., monitorFreq=86400.,
 &
 &PARM04
 usingCartesianGrid=.TRUE., delX={nx}*{c.dx}, delY={ny}*{c.dx}, delR={NR}*{DR},
 &
 &PARM05
 bathyFile='bathy.bin', hydrogThetaFile='theta.bin',
 &
"""


def obcs_text(c: cases.Case) -> str:
    nx, _ = _shape(c)
    return f""" &OBCS_PARM01
 OB_Jnorth={nx}*-1,
 useOBCSprescribe=.TRUE., useOBCSbalance=.FALSE., useOrlanskiNorth=.FALSE.,
 OBNvFile='OBNv.bin', OBNtFile='OBNt.bin', OBNptrFile(1)='OBNs.bin',
 &
 &OBCS_PARM02
 &
"""


def write_run(res_km: float, run_dir: Path, variant: str = "common", days: float | None = None) -> Path:
    """data, data.pkg, data.obcs, data.rbcs, data.ptracers, data.diagnostics, eedata and the binary inputs."""
    c = cases.get(res_km)
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    nx, ny = _shape(c)
    be = lambda a, name: np.ascontiguousarray(a, ">f8").tofile(d / name)     # noqa: E731
    (d / "data").write_text(data_text(c, variant, days))
    (d / "data.pkg").write_text(" &PACKAGES\n useOBCS=.TRUE., usePTRACERS=.TRUE., useRBCS=.TRUE.,\n"
                                " useDiagnostics=.TRUE., useMNC=.TRUE.,\n &\n")
    (d / "eedata").write_text(" &EEPARMS\n nTx=1, nTy=1,\n &\n")
    (d / "data.obcs").write_text(obcs_text(c))
    peak = 10.0 / 86400.0
    (d / "data.rbcs").write_text(f""" &RBCS_PARM01
 tauRelaxT={1.0 / peak:g}, relaxMaskFile(1)='rbcs_mask.bin', relaxTFile='rbcs_T.bin',
 rbcsForcingPeriod=0., rbcsForcingCycle=0.,
 &
 &RBCS_PARM02
 useRBCpTrNum(1)=.TRUE., tauRelaxPTR(1)={1.0 / peak:g}, relaxPtracerFile(1)='rbcs_s.bin',
 &
""")
    (d / "data.ptracers").write_text(""" &PTRACERS_PARM01
 PTRACERS_numInUse=1, PTRACERS_names(1)='dye', PTRACERS_advScheme(1)=77,
 PTRACERS_diffKh(1)=0., PTRACERS_diffKr(1)=0., PTRACERS_initialFile(1)=' ',
 &
""")
    (d / "data.diagnostics").write_text(""" &DIAGNOSTICS_LIST
  fields(1:5,1) = 'THETA   ','UVEL    ','VVEL    ','hFactorC','TRAC01  ',
  fileName(1) = 'diag3d', frequency(1) = -86400., timePhase(1) = 0.,
  fields(1,2) = 'ETAN    ', fileName(2) = 'diag2d', frequency(2) = -86400., timePhase(2) = 0.,
 &
 &DIAG_STATIS_PARMS
 &
""")
    (d / "data.mnc").write_text(" &MNC_01\n mnc_use_outdir=.FALSE., mnc_outdir_str='mnc_',\n &\n")
    bathy = np.zeros((ny, nx))
    bathy[1:c.ny + 1, :c.nx] = -np.nan_to_num(c.H)
    bathy[c.ny + 1, :c.nx] = -np.nan_to_num(c.H[-1])                    # OBCS row: the channel, land elsewhere
    be(bathy, "bathy.bin")
    zc = -(np.arange(NR) + 0.5) * DR
    T3 = cases.T_ambient(zc)[:, None, None] * np.ones((1, ny, nx))
    be(T3, "theta.bin")
    be(T3, "rbcs_T.bin")
    rate = np.zeros((ny, nx))
    rate[1:c.ny + 1, :c.nx] = cases.sponge_rate(c)
    be(np.broadcast_to(rate / peak, (NR, ny, nx)), "rbcs_mask.bin")
    be(np.zeros((NR, ny, nx)), "rbcs_s.bin")
    v, T, s = inflow_levels(c)
    be(v, "OBNv.bin")
    be(T, "OBNt.bin")
    be(s, "OBNs.bin")
    return d


def _join(pattern: str, name: str) -> np.ndarray:
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    with netCDF4.Dataset(files[0]) as ds:
        return np.asarray(ds[name][:], float)


def read(run_dir: Path, res_km: float) -> dict:
    """Series fields (surface first): the interior rows 1..ny and columns 0..nx−1; v on the ny+1 faces (the last
    one is the open face, the south face of the OBCS row); u on the nx+1 faces (the last one the east wall)."""
    c = cases.get(res_km)
    d = Path(run_dir)
    nx, ny = c.nx, c.ny
    f3 = str(d / "diag3d.*.t001.nc")
    t = _join(f3, "T")
    T = _join(f3, "THETA")[:, :, 1:ny + 1, :nx]
    S = _join(f3, "TRAC01")[:, :, 1:ny + 1, :nx]
    U = _join(f3, "UVEL")[:, :, 1:ny + 1, :nx + 1]
    V = _join(f3, "VVEL")[:, :, 1:ny + 2, :nx]
    hf = _join(f3, "hFactorC")[:, :, 1:ny + 1, :nx]
    eta = _join(str(d / "diag2d.*.t001.nc"), "ETAN")
    eta = eta.reshape(eta.shape[0], *eta.shape[-2:])[:, 1:ny + 1, :nx]
    U[..., -1] = 0.0
    h = hf * DR
    if t[0] > 0:                                   # no snapshot at iteration 0: the inputs are the initial state
        hf0 = _join(str(d / "grid.t001.nc"), "hFacC")[:, 1:ny + 1, :nx]
        Nx, Ny = _shape(c)
        T0 = np.fromfile(d / "theta.bin", ">f8").reshape(NR, Ny, Nx)[:, 1:ny + 1, :nx]
        h = np.concatenate([(hf0 * DR)[None], h])
        T = np.concatenate([T0[None], T])
        S = np.concatenate([np.zeros((1,) + S.shape[1:]), S])
        U = np.concatenate([np.zeros((1,) + U.shape[1:]), U])
        V = np.concatenate([np.zeros((1,) + V.shape[1:]), V])
        eta = np.concatenate([np.zeros((1, ny, nx)), eta])
        t = np.concatenate([[0.0], t])
    land = h <= 0
    s = dict(t=t, h=h, T=np.where(land, np.nan, T), s=np.where(land, np.nan, S), u=U, v=V, eta=eta)
    s.update(x=c.x, y=c.y, dx=np.full(nx, c.dx), dy=np.full(ny, c.dx), H=c.H)
    return s


def wall_s(run_dir: Path) -> float:
    """Wall clock of the main loop (output.txt), as validation/ilicak_benchmark/mitgcm.py."""
    from validation.ilicak_benchmark.mitgcm import wall_s as ws
    return ws(run_dir)
