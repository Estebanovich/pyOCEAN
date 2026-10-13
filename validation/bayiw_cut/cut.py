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


def read_mnc(run_dir: Path, prefix: str, var: str, k=None, t_slice=slice(None)) -> tuple[np.ndarray, np.ndarray]:
    """
    One MNC variable of a run, tiles joined: time (n,) [s] (absent for grid files: zeros) and the field (n, …,
    Y|Yp1, X|Xp1).  Tiles are placed by their first X/Xp1 and Y/Yp1 coordinate; on face dimensions (Xp1, Yp1)
    a tile's first face equals its neighbour's last and is dropped.  ``k``: level indices to keep (axis −3).
    """
    import glob

    import netCDF4
    files = sorted(glob.glob(str(Path(run_dir) / "mnc_*" / f"{prefix}.*t*.nc")))
    blocks: dict[str, list] = {}
    for f in files:
        blocks.setdefault(Path(f).name.rsplit(".t", 1)[0], []).append(f)      # one time block = all its tiles
    times, fields = [], []
    for b, fs in sorted(blocks.items()):
        tiles = []
        for f in fs:
            with netCDF4.Dataset(f) as ds:
                v = ds[var]
                dims = v.dimensions
                xd, yd = dims[-1], dims[-2]
                a = v[t_slice] if "T" in dims else v[:]
                a = np.asarray(a, float)
                if k is not None:
                    a = np.take(a, k, axis=-3)
                tiles.append((float(ds[xd][0]), float(ds[yd][0]), xd.endswith("p1"), yd.endswith("p1"), a,
                              np.asarray(ds["T"][t_slice], float) if "T" in ds.variables else np.zeros(1)))
        tiles = list({(t[0], t[1]): t for t in reversed(tiles)}.values())     # same tile in several mnc dirs
        xs, ys = sorted({t[0] for t in tiles}), sorted({t[1] for t in tiles})   # (two-stage runs): keep one
        rows = []
        for y in ys:
            row = [t for t in tiles if t[1] == y]
            row.sort(key=lambda t: t[0])
            rows.append(np.concatenate([r[4][..., 1:] if (r[2] and n > 0) else r[4] for n, r in enumerate(row)], -1))
        yface = tiles[0][3]
        field = np.concatenate([r[..., 1:, :] if (yface and n > 0) else r for n, r in enumerate(rows)], -2)
        times.append(tiles[0][5])
        fields.append(field)
    t = np.concatenate(times)
    f = np.concatenate(fields) if fields[0].ndim and "grid" not in prefix else fields[0][None]
    o = np.argsort(t) if prefix != "grid" else slice(None)
    return t[o], f[o]


def read_cut(case: str, bay: str, mode: str) -> dict:
    """The cut run's diagnostics: t, η (n, 160, 200), T (n, 21, 160, 200) on levels 20–40, U (n, 2, 160, 201)
    and V (n, 2, 161, 200) on levels 30 and 40 (1-based), and the grid depth."""
    run = RUNS / "mitgcm" / f"{case}_{bay}_{mode}"
    t, eta = read_mnc(run, "diag_eta", "ETAN")
    _, T = read_mnc(run, "diag_theta", "THETA")
    _, U = read_mnc(run, "diag_uv", "UVEL")
    _, V = read_mnc(run, "diag_uv", "VVEL")
    _, depth = read_mnc(run, "grid", "Depth")
    return dict(t=t, eta=eta[:, 0] if eta.ndim == 4 else eta, T=T, U=U, V=V, depth=depth[0])


def full_run_dir(case: str, bay: str) -> Path:
    """Esteban's full run: run_expand (bay) or run_expand_nobay."""
    return bayiw.CASES[case]["dir"] / ("run_expand" if bay == "bay" else "run_expand_nobay")


def build_model(case: str, bay: str, coord: str = "zstar", dt: float = 120.0, boundary_state=None):
    """
    pyOCEAN on MITgcm's interior of the cut (spec §1.2): z* on the 50 delR levels or hybrid with 25 layers
    (N_LIGHT light targets); west/east/south open with Flather + Orlanski 0.45 + 30-cell u′ sponge (phase 1),
    or with ``boundary_state`` (phase 2, 'radiating' toward Esteban's full run); initial state from the MITgcm
    levels.  Returns (model, initial state, wind (nrec, 159, 198)).
    """
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid_stretched
    d = cut_inputs(case, bay, "py")
    hg = create_cartesian_grid_stretched(d["dx"], d["dy"], x0=d["x0"], y0=d["y0"], f0=bayiw.F0)
    t_ref = bayiw.CASES[case]["t_ref"]
    if coord == "zstar":
        nz, params = bayiw.DELR.size, dict(dz_top=bayiw.DELR.copy())
    elif coord == "hybrid":
        nz = 25
        rho = bayiw.RHO0 - bayiw.T_COEF * (d["T"][:, np.isfinite(d["H"])] - t_ref)
        tgt = (bayiw.centred_targets(float(rho.min()), float(rho.max()), nz, bayiw.N_LIGHT) if case == "2layer"
               else bayiw.profile_targets(rho, nz, bayiw.N_LIGHT))
        params = dict(dz_top=bayiw.DELR[:nz].copy(), rho_target=tgt)
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, d["H"], dt=dt, n_layers=nz, rho0=bayiw.RHO0, t_coef=bayiw.T_COEF, t_ref=t_ref, z0=None,
                        closure="constant", nu_const=1e-5, ah=100.0, kh=100.0, kv=1e-5, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params,
                        open_boundaries=("west", "east", "south"), orlanski_cmax=0.45, sponge_cells=30,
                        sponge_tau=(1200.0, 60.0), sponge_tracer=False, mix_diagnostic=False,
                        open_boundary_state=boundary_state)
    h_src, T_src = bayiw.mitgcm_source(d["H"], d["T"])
    st = m.initial_state(np.zeros(d["H"].shape), T_src, h_src=h_src)
    return m, st, d["wind"]
