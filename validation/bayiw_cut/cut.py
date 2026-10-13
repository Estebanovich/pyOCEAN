"""
BayIW 3-D cut (spec docs/superpowers/specs/2026-10-12-bayiw-cut-design.md): MITgcm on global i = 180–379,
j = 192–351 with its open-boundary cells on the ring (OB_Iwest = 1, OB_Ieast = −1, OB_Jsouth = 1: T in the OB
cell, u/v at the face next to the interior, MITgcm obcs_apply_ts.F:205, obcs_apply_uv.F:205); pyOCEAN on
MITgcm's interior, global i = 181–378, j = 193–351, with its open faces at MITgcm's prescribed u/v points.
Phase 1 radiates to a resting exterior (MITgcm Orlanski only); phase 2 nests both in Esteban's full runs.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from validation.bayiw_baroclinic import bayiw

I0, I1, J0, J1 = 180, 380, 192, 352            # MITgcm cut (global, half-open)
PI0, PI1, PJ0, PJ1 = 181, 379, 193, 352        # pyOCEAN domain = MITgcm interior
RUNS = bayiw.RUNS / "bayiw_cut"
STATIONS = bayiw.STATIONS


def _box(which: str) -> tuple[int, int, int, int]:
    return (I0, I1, J0, J1) if which == "mit" else (PI0, PI1, PJ0, PJ1)


def cut_inputs(case: str, bay: str, which: str = "py") -> dict:
    """dx, dy [m], H (ny, nx) [m] (MITgcm effective depth, NaN land), T (50, ny, nx), wind (nrec, ny, nx), x0, y0."""
    dx, dy, H, T, wind = bayiw.load_inputs(case, bay)
    i0, i1, j0, j1 = _box(which)
    return dict(dx=dx[i0:i1].copy(), dy=dy[j0:j1].copy(), H=H[j0:j1, i0:i1].copy(), T=T[:, j0:j1, i0:i1].copy(),
                wind=wind[:, j0:j1, i0:i1].copy(), x0=bayiw.X0 + float(dx[:i0].sum()),
                y0=bayiw.Y0 + float(dy[:j0].sum()))


def local_stations(which: str = "py") -> dict[str, tuple[int, int]]:
    i0, _, j0, _ = _box(which)
    return {n: (i - i0, j - j0) for n, (i, j) in STATIONS.items()}


def _write(path: Path, a: np.ndarray) -> None:
    np.ascontiguousarray(a, ">f8").tofile(path)


DIAGNOSTICS = """ &DIAGNOSTICS_LIST
  fields(1,1)  = 'ETAN',
  fileName(1)  = 'diag_eta',
  frequency(1) = 900.,
  fields(1,2)  = 'THETA',
  levels(1,2)  = {levels},
  fileName(2)  = 'diag_theta',
  frequency(2) = 900.,
  fields(1,3)  = 'UVEL',
  fields(2,3)  = 'VVEL',
  levels(1,3)  = 30., 40.,
  fileName(3)  = 'diag_uv',
  frequency(3) = 900.,
 &
 &DIAG_STATIS_PARMS
 &
"""


def write_mitgcm_inputs(case: str, bay: str, run_dir: Path, mode: str = "radiate") -> None:
    """
    MITgcm inputs and namelists of the cut in run_dir: input/{dx,dy,bathy,theta,wind}.bin (big-endian float64),
    data (the case's data.stage1: section files, endTime 216 000 s, state every 6 h, no pickup, no salt file),
    data.obcs (three open edges; 'radiate': Orlanski only, no balance, no sponge; 'nest': prescribe from the
    OB files of write_obcs_files, no balance, no sponge), data.pkg (OBCS, MNC, diagnostics), data.mnc,
    data.diagnostics (η, THETA on levels 20–40, U/V on levels 30 and 40 every 900 s), eedata.
    """
    run_dir = Path(run_dir)
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    d = cut_inputs(case, bay, "mit")
    _write(run_dir / "input" / "dx.bin", d["dx"])
    _write(run_dir / "input" / "dy.bin", d["dy"])
    _write(run_dir / "input" / "bathy.bin", np.where(np.isfinite(d["H"]), -d["H"], 0.0))
    _write(run_dir / "input" / "theta.bin", d["T"])
    _write(run_dir / "input" / "wind.bin", d["wind"])
    src = (bayiw.CASES[case]["dir"] / "run_expand" / "data.stage1").read_text()
    for key, val in (("delYFile", "'input/dy.bin'"), ("delXFile", "'input/dx.bin'"), ("bathyFile", "'input/bathy.bin'"),
                     ("meridWindFile", "'input/wind.bin'"), ("hydrogThetaFile", "'input/theta.bin'")):
        src = re.sub(rf"{key} *=.*", f"{key}={val},", src)
    src = re.sub(r" *hydrogSaltFile *=.*\n", "", src)
    src = re.sub(r"endTime *=.*", "endTime=216000.,", src)
    src = re.sub(r"pChkptFreq *=.*", "pChkptFreq=0.,", src)
    src = re.sub(r"dumpFreq *=.*", "dumpFreq=21600.,", src)
    src = re.sub(r"xgOrigin *=.*", f"xgOrigin={d['x0']!r},", src)
    src = re.sub(r"ygOrigin *=.*", f"ygOrigin={d['y0']!r},", src)
    (run_dir / "data").write_text(src)
    common = (" &OBCS_PARM01\n OB_Jsouth = 200*1,\n OB_Iwest = 160*1,\n OB_Ieast = 160*-1,\n"
              " useOBCSbalance=.FALSE.,\n useOBCSsponge=.FALSE.,\n")
    if mode == "radiate":
        body = (" useOrlanskiSouth=.TRUE.,\n useOrlanskiWest=.TRUE.,\n useOrlanskiEast=.TRUE.,\n"
                " useOBCSprescribe=.FALSE.,\n &\n &OBCS_PARM02\n cvelTimeScale=250.0,\n CMAX=0.45,\n &\n")
    elif mode == "nest":
        files = "".join(f" OB{e}{v}File='input/ob_{e}{v}.bin',\n" for e in "SWE" for v in ("u", "v", "t", "eta"))
        body = " useOBCSprescribe=.TRUE.,\n" + files + " &\n &OBCS_PARM02\n &\n"
    else:
        raise ValueError(f"unknown mode {mode!r}")
    (run_dir / "data.obcs").write_text(common + body)
    (run_dir / "data.pkg").write_text(" &PACKAGES\n useMNC=.TRUE.,\n useOBCS=.TRUE.,\n useDiagnostics=.TRUE.,\n &\n")
    (run_dir / "data.mnc").write_text(" &MNC_01\n mnc_use_outdir=.TRUE.,\n mnc_outdir_str='mnc_',\n monitor_mnc=.FALSE.,\n"
                                      " pickup_read_mnc=.FALSE.,\n pickup_write_mnc=.FALSE.,\n &\n")
    (run_dir / "data.diagnostics").write_text(DIAGNOSTICS.format(levels=", ".join(f"{k}." for k in range(20, 41))))
    (run_dir / "eedata").write_text(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n")
