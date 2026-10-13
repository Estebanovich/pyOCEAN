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
        files = "".join(f" OB{e}{v}File='input/ob_{e}{v}.bin',\n" for e in "SWE" for v in ("u", "v", "t"))
        # no OB*etaFile: only allowed with nonlinFreeSurf (obcs_check.F:563-566); η follows from the transport
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
        params = dict(dz_top=bayiw.hybrid_dz_top(nz), rho_target=tgt)
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


# ---------------------------------------------------------------- phase 2: boundaries from the full runs
NREC, PERIOD_FORCING = 180, 1200.0                       # MITgcm periodic forcing calendar (data.stage1)


def read_full_box(run_dir: Path, var: str, j0: int, j1: int, i0: int, i1: int, case: str, bay: str,
                  t_max: float = 216000.0, levels: list[int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    A global index box [j0, j1) × [i0, i1) of a full-run state variable (cell or face indices, as the variable),
    read tile by tile without joining the whole domain: time (n,) [s] and (n, nz, j1 − j0, i1 − i0).  Tiles are
    placed from their first X/Xp1 and Y/Yp1 coordinate on the case's global grid; duplicated tiles (one per
    stage directory) are read once; a time written by two stages is kept once; only times ≤ t_max; ``levels``
    (0-based, surface first) reads only those levels of a 3-D variable.
    """
    import glob

    import netCDF4
    dx, dy = bayiw.load_inputs(case, bay)[:2]
    gx = {"X": bayiw.X0 + np.cumsum(dx) - 0.5 * dx, "Xp1": bayiw.X0 + np.concatenate([[0.0], np.cumsum(dx)])}
    gy = {"Y": bayiw.Y0 + np.cumsum(dy) - 0.5 * dy, "Yp1": bayiw.Y0 + np.concatenate([[0.0], np.cumsum(dy)])}
    blocks: dict[str, dict] = {}
    for f in sorted(glob.glob(str(Path(run_dir) / "mnc_*" / "state.*t*.nc"))):
        blocks.setdefault(Path(f).name.rsplit(".t", 1)[0], {})[Path(f).name] = f     # dedupe by tile name
    ts, out = [], []
    for name, tiles in sorted(blocks.items()):
        arr, tt = None, None
        for f in tiles.values():
            with netCDF4.Dataset(f) as ds:
                v = ds[var]
                xd, yd = v.dimensions[-1], v.dimensions[-2]
                ti0 = int(np.argmin(np.abs(gx[xd] - float(ds[xd][0]))))
                tj0 = int(np.argmin(np.abs(gy[yd] - float(ds[yd][0]))))
                nxt, nyt = ds.dimensions[xd].size, ds.dimensions[yd].size
                a0, a1 = max(i0, ti0), min(i1, ti0 + nxt)
                b0, b1 = max(j0, tj0), min(j1, tj0 + nyt)
                if tt is None:
                    tt = np.asarray(ds["T"][:], float)
                if a0 >= a1 or b0 >= b1:
                    continue
                if v.ndim == 4 and levels is not None:
                    sel = np.asarray(v[:, list(levels), b0 - tj0:b1 - tj0, a0 - ti0:a1 - ti0], float)
                else:
                    sel = np.asarray(v[:, ..., b0 - tj0:b1 - tj0, a0 - ti0:a1 - ti0], float)
                if sel.ndim == 3:
                    sel = sel[:, None]
                if arr is None:
                    arr = np.full((tt.size, sel.shape[1], j1 - j0, i1 - i0), np.nan)
                arr[:, :, b0 - j0:b1 - j0, a0 - i0:a1 - i0] = sel
        keep = tt <= t_max + 1.0
        if arr is not None and keep.any():
            ts.append(tt[keep])
            out.append(arr[keep])
    t = np.concatenate(ts)
    t, o = np.unique(t, return_index=True)                  # a stage's last dump = the next stage's first
    return t, np.concatenate(out)[o]


def boundary_series(case: str, bay: str) -> dict:
    """
    Full-run values at MITgcm's open-boundary placement of the cut, every 900 s (cached in RUNS): for west/east
    T and η in the OB cell (global column 180 / 379, rows 192–351), u at the face next to the interior
    (181 / 379), v at the south faces of the OB cells; for south T and η in row 192 (columns 180–379), v at face
    193, u at the west faces of the OB cells.  Arrays (n, 50, m) and η (n, m).
    """
    cache = RUNS / f"boundary_{case}_{bay}.npz"
    if cache.exists():
        z = np.load(cache)
        return {"t": z["t"], **{e: {k: z[f"{e}_{k}"] for k in ("T", "u", "v", "eta")} for e in ("west", "east", "south")}}
    run = full_run_dir(case, bay)
    rd = lambda var, j0, j1, i0, i1: read_full_box(run, var, j0, j1, i0, i1, case, bay)    # noqa: E731
    t, wT = rd("Temp", J0, J1, I0, I0 + 1)
    out = {"t": t}
    for e, ic, iu in (("west", I0, I0 + 1), ("east", I1 - 1, I1 - 1)):
        out[e] = dict(T=rd("Temp", J0, J1, ic, ic + 1)[1][..., 0], u=rd("U", J0, J1, iu, iu + 1)[1][..., 0],
                      v=rd("V", J0, J1, ic, ic + 1)[1][..., 0], eta=rd("Eta", J0, J1, ic, ic + 1)[1][:, 0, :, 0])
    out["south"] = dict(T=rd("Temp", J0, J0 + 1, I0, I1)[1][:, :, 0], v=rd("V", J0 + 1, J0 + 2, I0, I1)[1][:, :, 0],
                        u=rd("U", J0, J0 + 1, I0, I1)[1][:, :, 0], eta=rd("Eta", J0, J0 + 1, I0, I1)[1][:, 0, 0])
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.savez(cache, t=t, **{f"{e}_{k}": out[e][k] for e in ("west", "east", "south") for k in ("T", "u", "v", "eta")})
    return out


def _fill_below_bed(T: np.ndarray) -> np.ndarray:
    """MITgcm writes 0 (land) below the bed: give those levels the deepest wet value (axis −2 = levels, surface first)."""
    T = np.where(T == 0.0, np.nan, T)
    for k in range(1, T.shape[-2]):
        T[..., k, :] = np.where(np.isnan(T[..., k, :]), T[..., k - 1, :], T[..., k, :])
    return np.nan_to_num(T)


def full_reference(case: str, bay: str) -> dict:
    """
    Phase-2 reference (spec §1.4): Esteban's full run on the pyOCEAN grid, as the series of run_pyocean — η
    (n, 159, 198), T on MITgcm levels 20–40 (n, 21, 159, 198), U on levels 30/40 at the faces 181–379
    (n, 2, 159, 199), V at the faces 193–352 (n, 2, 160, 198); snapshots every 900 s, float32, cached in RUNS.
    MITgcm's zeros below the bed are left as they are (compare.load_full masks them).
    """
    cache = RUNS / f"full_{case}_{bay}.npz"
    if cache.exists():
        with np.load(cache) as z:
            return {k: z[k] for k in z.files}
    run = full_run_dir(case, bay)
    rd = lambda var, j0, j1, i0, i1, lev=None: read_full_box(run, var, j0, j1, i0, i1, case, bay,    # noqa: E731
                                                            levels=lev)
    t, eta = rd("Eta", PJ0, PJ1, PI0, PI1)
    out = dict(t=t, eta=eta[:, 0].astype(np.float32),
               T=rd("Temp", PJ0, PJ1, PI0, PI1, list(range(19, 40)))[1].astype(np.float32),
               U=rd("U", PJ0, PJ1, PI0, PI1 + 1, [29, 39])[1].astype(np.float32),
               V=rd("V", PJ0, PJ1 + 1, PI0, PI1, [29, 39])[1].astype(np.float32))
    np.savez(cache, **out)
    return out


def nest_state(case: str, bay: str):
    """
    t → {edge: OBData} for the pyOCEAN domain (rows 193–351 for west/east, columns 181–378 for south) from the
    full run (spec §1.4): zw = the MITgcm interfaces, T and η of MITgcm's OB cell, u_normal = the velocity at
    pyOCEAN's open face (MITgcm's prescribed u/v point), ubar its depth mean over the face's water column;
    linear in time between the 900-s dumps (forcing.boundary_time.interpolated_boundary).
    """
    from boundaries.obcs import OBData
    from forcing.boundary_time import interpolated_boundary
    s = boundary_series(case, bay)
    zw1 = -np.concatenate([[0.0], np.cumsum(bayiw.DELR)])
    dz = bayiw.DELR[:, None]
    H = cut_inputs(case, bay, "mit")["H"]
    face_depth = dict(west=np.fmin(H[1:, 0], H[1:, 1]), east=np.fmin(H[1:, -1], H[1:, -2]),
                      south=np.fmin(H[0, 1:-1], H[1, 1:-1]))
    sl = dict(west=slice(1, None), east=slice(1, None), south=slice(1, -1))
    norm = dict(west="u", east="u", south="v")
    states = []
    for n in range(s["t"].size):
        st = {}
        for e in ("west", "east", "south"):
            T = _fill_below_bed(s[e]["T"][n][:, sl[e]].copy())
            u = np.nan_to_num(s[e][norm[e]][n][:, sl[e]])
            D = np.nan_to_num(face_depth[e])
            ub = np.where(D > 0, (u * dz).sum(0) / np.where(D > 0, D, 1.0), 0.0)
            st[e] = OBData(side=e, T=T, u_normal=u, zw=np.repeat(zw1[:, None], T.shape[1], 1),
                           eta=np.nan_to_num(s[e]["eta"][n][sl[e]]), ubar=ub)
        states.append(st)
    return interpolated_boundary(s["t"], states)


def write_obcs_files(case: str, bay: str, run_dir: Path) -> None:
    """
    MITgcm OBCS prescribe files (big-endian float64, (nrec, Nr, n) and η (nrec, n)) on the forcing calendar
    (get_periodic_interval.F: record r at (r − ½)·1200 s, cycle 216 000 s): the full-run values interpolated
    in time; the last record holds the initial state (the cycle wraps it onto t < 600 s).
    """
    s = boundary_series(case, bay)
    tc = (np.arange(NREC) + 0.5) * PERIOD_FORCING
    (Path(run_dir) / "input").mkdir(parents=True, exist_ok=True)

    def interp(a: np.ndarray) -> np.ndarray:
        flat = np.nan_to_num(a).reshape(a.shape[0], -1)
        out = np.stack([np.interp(tc, s["t"], flat[:, c]) for c in range(flat.shape[1])], axis=1)
        out = out.reshape((NREC,) + a.shape[1:])
        out[-1] = np.nan_to_num(a[0])
        return out
    for e, tag in (("south", "S"), ("west", "W"), ("east", "E")):
        for k, v in (("u", "u"), ("v", "v"), ("T", "t"), ("eta", "eta")):
            _write(Path(run_dir) / "input" / f"ob_{tag}{v}.bin", interp(s[e][k]))
