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


def write_mitgcm_inputs(case: str, bay: str, run_dir: Path, periodic: bool = False, balance: bool = True,
                        sponge: bool = True) -> None:
    """
    MITgcm inputs and namelists of the section in run_dir: input/{dx,dy,bathy,theta,wind}.bin (big-endian
    float64, Nx = 3: the section column repeated, with land in columns 0 and 2 unless periodic) and data (the
    case's run_expand/data.stage1 with the section files, endTime 216 000 s and no pickup; hydrogSaltFile
    dropped: salt is not stepped and sBeta = 0), data.obcs (south only: Orlanski CMAX 0.45, cvelTimeScale
    250 s, 30-cell velocity sponge 60 → 1200 s, balanceFacS = −1 as Esteban's runs, or no balance when
    balance is False: −1 zeroes the net transport through the south edge, obcs_balance_flow.F:76, which closes a
    section for the barotropic mode; sponge False drops the velocity sponge), data.pkg (OBCS, MNC), data.mnc, eedata.
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
    bal = " useOBCSbalance=.TRUE.,\n OBCS_balanceFacS=-1.,\n" if balance else " useOBCSbalance=.FALSE.,\n"
    (run_dir / "data.obcs").write_text(
        f" &OBCS_PARM01\n OB_Jsouth = {ob},\n useOrlanskiSouth=.TRUE.,\n useOBCSsponge=.{str(sponge).upper()}.,\n"
        " useLinearSponge=.FALSE.,\n OBCSsponge_Salt=.FALSE.,\n OBCSsponge_Theta=.FALSE.,\n" + bal +
        " useOBCSprescribe=.FALSE.,\n &\n &OBCS_PARM02\n cvelTimeScale=250.0,\n CMAX=0.45,\n &\n"
        + (" &OBCS_PARM03\n spongeThickness=30,\n Urelaxobcsinner=1200.E0,\n Vrelaxobcsinner=1200.E0,\n"
           " Urelaxobcsbound=60.E0,\n Vrelaxobcsbound=60.E0,\n &\n" if sponge else ""))
    (run_dir / "data.pkg").write_text(" &PACKAGES\n useMNC=.TRUE.,\n useOBCS=.TRUE.,\n &\n")
    (run_dir / "data.mnc").write_text(" &MNC_01\n mnc_use_outdir=.TRUE.,\n mnc_outdir_str='mnc_',\n monitor_mnc=.FALSE.,\n"
                                      " pickup_read_mnc=.FALSE.,\n pickup_write_mnc=.FALSE.,\n &\n")
    (run_dir / "eedata").write_text(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n")


def read_mitgcm(case: str, bay: str, variant: str = "") -> dict:
    """
    MITgcm MNC state dumps of a section run (one tile, t001), on the section column x-index 1: time [h],
    η (n, 352) [m], T (n, 50, 352) [°C], U at its west face (n, 50, 352) and V (n, 50, 353) [m s⁻¹], level
    centres zc (50,) [m].  variant '' (walls) or 'periodic'.
    """
    import glob

    import netCDF4
    run = RUNS / "mitgcm" / (f"{case}_{bay}" + (f"_{variant}" if variant else ""))
    files = sorted(glob.glob(str(run / "mnc_*" / "state.*.t001.nc")))
    if not files:
        raise FileNotFoundError(f"no MNC state files in {run}")
    t, eta, T, U, V = [], [], [], [], []
    for f in files:
        with netCDF4.Dataset(f) as ds:
            t.append(np.asarray(ds["T"][:], float))
            eta.append(np.asarray(ds["Eta"][:, :, 1], float))
            T.append(np.asarray(ds["Temp"][:, :, :, 1], float))
            U.append(np.asarray(ds["U"][:, :, :, 1], float))
            V.append(np.asarray(ds["V"][:, :, :, 1], float))
    order = np.argsort(np.concatenate(t))
    cat = lambda xs: np.concatenate(xs)[order]                                     # noqa: E731
    return dict(t_h=cat(t) / 3600.0, eta=cat(eta), T=cat(T), U=cat(U), V=cat(V), zc=bayiw.Z_C.copy())


def build_model(case: str, bay: str, coord: str = "zstar", dt: float = 30.0, n_layers: int = 25):
    """
    pyOCEAN ALE model of the section with the MITgcm parameters (spec 3-D §3): z* on the 50 delR levels or
    hybrid with n_layers layers (25 as the 3-D benchmark; N_LIGHT light targets keep the top 14.7 m in z*); west/east walls, south open
    (Flather + Orlanski 0.45 + 30-cell velocity sponge 60 → 1200 s), no mixing diagnostic.  Δt = 30 s as
    MITgcm.  Returns (model, initial state, wind records (nrec, 352, 1)).
    """
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid_stretched
    d = section_inputs(case, bay)
    hg = create_cartesian_grid_stretched(d["dx"], d["dy"], x0=d["x0"], y0=bayiw.Y0, f0=bayiw.F0)
    t_ref = bayiw.CASES[case]["t_ref"]
    if coord == "zstar":
        nz, params = bayiw.DELR.size, dict(dz_top=bayiw.DELR.copy())
    elif coord == "hybrid":
        nz = n_layers
        rho = bayiw.RHO0 - bayiw.T_COEF * (d["T"][:, np.isfinite(d["H"])] - t_ref)
        tgt = (bayiw.centred_targets(float(rho.min()), float(rho.max()), nz, bayiw.N_LIGHT) if case == "2layer"
               else bayiw.profile_targets(rho, nz, bayiw.N_LIGHT))
        params = dict(dz_top=bayiw.DELR[:nz].copy(), rho_target=tgt)
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, d["H"], dt=dt, n_layers=nz, rho0=bayiw.RHO0, t_coef=bayiw.T_COEF, t_ref=t_ref, z0=None,
                        closure="constant", nu_const=1e-5, ah=100.0, kh=100.0, kv=1e-5, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params, open_boundaries=("south",),
                        orlanski_cmax=0.45, sponge_cells=30, sponge_tau=(1200.0, 60.0), sponge_tracer=False,
                        mix_diagnostic=False)
    z = m.layer_heights(np.zeros(d["H"].shape))
    st = m.initial_state(np.zeros(d["H"].shape), bayiw.mitgcm_profile_at(z, d["T"]))
    return m, st, d["wind"]
