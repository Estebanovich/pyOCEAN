"""
BayIW 2-D section along the bay axis (column i = 280 of the MITgcm grid) for pyOCEAN and MITgcm (spec
docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md): inputs from Esteban's MITgcm binaries
(validation/bayiw_baroclinic/bayiw.py), MITgcm namelists of stage 1 (0–60 h, periodic wind) with only the
south boundary open, the MITgcm MNC reader and the pyOCEAN section model.

MITgcm is periodic in x, so a one-column MITgcm section is a rotating x-invariant (2.5-D) slice, while
pyOCEAN has no periodic x and closes a one-column channel with walls (u = 0, so f does nothing).  The
comparison therefore runs MITgcm with Nx = 3 and land in columns 0 and 2 (walls, same as pyOCEAN); the
'periodic' variant (three identical wet columns) measures what the missing rotation changes.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from validation.bayiw_baroclinic import bayiw

I_SEC = 280
SECTION_STATIONS = {"Gamma": 215, "C": 265, "Omega": 315, "Seiche2": 325, "Seiche": 345}
RUNS = bayiw.RUNS / "bayiw_2d"
NX_MIT = 3                                              # MITgcm columns: wall | section | wall (or 3 wet, periodic)


def section_inputs(case: str, bay: str) -> dict:
    """
    The column i = 280: dx (1,), dy (352,) [m], H (352, 1) [m] NaN on land, T (50, 352, 1) [°C],
    wind τ_y (nrec, 352, 1) [N m⁻²], and x0, the x of its west edge [m] (xgOrigin + Σ dx[:280]).
    """
    dx, dy, H, T, wind = bayiw.load_inputs(case, bay)
    s = slice(I_SEC, I_SEC + 1)
    return dict(dx=dx[s].copy(), dy=dy.copy(), H=H[:, s].copy(), T=T[:, :, s].copy(), wind=wind[:, :, s].copy(),
                x0=bayiw.X0 + float(dx[:I_SEC].sum()))


def _write(path: Path, a: np.ndarray) -> None:
    """MITgcm input file: big-endian float64 (readBinaryPrec = 64), C order = Fortran (x fastest)."""
    np.ascontiguousarray(a, ">f8").tofile(path)


def write_mitgcm_inputs(case: str, bay: str, run_dir: Path, periodic: bool = False) -> None:
    """
    MITgcm inputs and namelists of the section in run_dir: input/{dx,dy,bathy,theta,wind}.bin (big-endian
    float64, Nx = 3: the section column repeated, with land in columns 0 and 2 unless periodic) and data (the
    case's run_expand/data.stage1 with the section files, endTime 216 000 s and no pickup; hydrogSaltFile
    dropped: salt is not stepped and sBeta = 0), data.obcs (south only: Orlanski CMAX 0.45, cvelTimeScale
    250 s, 30-cell velocity sponge 60 → 1200 s, balanceFacS = −1, as Esteban's runs), data.pkg (OBCS, MNC),
    data.mnc, eedata.
    """
    run_dir = Path(run_dir)
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    d = section_inputs(case, bay)
    rep = lambda a: np.repeat(a, NX_MIT, axis=-1)                                  # noqa: E731
    bathy = rep(np.where(np.isfinite(d["H"]), -d["H"], 0.0))
    if not periodic:
        bathy[:, 0] = bathy[:, 2] = 0.0
    _write(run_dir / "input" / "dx.bin", rep(d["dx"]))
    _write(run_dir / "input" / "dy.bin", d["dy"])
    _write(run_dir / "input" / "bathy.bin", bathy)
    _write(run_dir / "input" / "theta.bin", rep(d["T"]))
    _write(run_dir / "input" / "wind.bin", rep(d["wind"]))
    src = (bayiw.CASES[case]["dir"] / "run_expand" / "data.stage1").read_text()
    src = re.sub(r"delYFile *=.*", "delYFile = 'input/dy.bin',", src)
    src = re.sub(r"delXFile *=.*", "delXFile = 'input/dx.bin',", src)
    src = re.sub(r"bathyFile *=.*", "bathyFile='input/bathy.bin',", src)
    src = re.sub(r"meridWindFile *=.*", "meridWindFile='input/wind.bin',", src)
    src = re.sub(r"hydrogThetaFile *=.*", "hydrogThetaFile='input/theta.bin',", src)
    src = re.sub(r" *hydrogSaltFile *=.*\n", "", src)
    src = re.sub(r"endTime *=.*", "endTime=216000.,", src)
    src = re.sub(r"pChkptFreq *=.*", "pChkptFreq=0.,", src)
    (run_dir / "data").write_text(src)
    ob = "3*1" if periodic else "0, 1, 0"
    (run_dir / "data.obcs").write_text(
        f" &OBCS_PARM01\n OB_Jsouth = {ob},\n useOrlanskiSouth=.TRUE.,\n useOBCSsponge=.TRUE.,\n"
        " useLinearSponge=.FALSE.,\n OBCSsponge_Salt=.FALSE.,\n OBCSsponge_Theta=.FALSE.,\n useOBCSbalance=.TRUE.,\n"
        " useOBCSprescribe=.FALSE.,\n OBCS_balanceFacS=-1.,\n &\n &OBCS_PARM02\n cvelTimeScale=250.0,\n CMAX=0.45,\n &\n"
        " &OBCS_PARM03\n spongeThickness=30,\n Urelaxobcsinner=1200.E0,\n Vrelaxobcsinner=1200.E0,\n"
        " Urelaxobcsbound=60.E0,\n Vrelaxobcsbound=60.E0,\n &\n")
    (run_dir / "data.pkg").write_text(" &PACKAGES\n useMNC=.TRUE.,\n useOBCS=.TRUE.,\n &\n")
    (run_dir / "data.mnc").write_text(" &MNC_01\n mnc_use_outdir=.TRUE.,\n mnc_outdir_str='mnc_',\n monitor_mnc=.FALSE.,\n"
                                      " pickup_read_mnc=.FALSE.,\n pickup_write_mnc=.FALSE.,\n &\n")
    (run_dir / "eedata").write_text(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n")
