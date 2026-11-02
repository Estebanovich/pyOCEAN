#!/usr/bin/env python
"""
Checks of MOM6 + SIS2 on the NWA12 cut (regional chain, plan Task 10; spec criteria 1, 4 and 7).

Criterion 1 (grid): MOM6's ``ocean_static.nc`` (diag_table of ``run_mom6.sh``, written in double precision) against
``core.grid.supergrid.grid_from_supergrid`` on the same cut (``case.grid``), ≤ 1e-12 relative, pointwise
(|a − b| / |b|, or |a − b| where b = 0):

| MOM6 static | pyOCEAN | where |
|---|---|---|
| geolon, geolat / _u / _v / _c | lon_t, lat_t / lon_u, lat_u / lon_v, lat_v / lon_q, lat_q | all points |
| dxt, dyt, area_t | dx_t, dy_t, area_t | all T points |
| dxCu, dyCu | dx_u, dy_u | all u faces |
| dxCv, dyCv | dx_v, dy_v | all v faces |
| dyCuo, dxCvo | dy_u, dx_v (open face lengths = masked lengths, mom6_grid.md §3) | wet faces |
| area_u, area_v | area_u, area_v (MOM6 products dxCu·dy_Cu, dyCv·dx_Cv, ruling R2) | wet faces |
| area_q | area_q | all q points |
| Coriolis | f_q | all q points |
| cos_rot, sin_rot | cos_t, −sin_t (rotation convention of grid_from_supergrid) | all T points |

``f_t`` has no MOM6 counterpart and is excluded (ruling R9; mom6_grid.md §5).  ``dx_q``, ``dy_q`` (MOM6 dxBu, dyBu)
are not static diagnostics of MOM6 and are not compared.

Criterion 7 (initial pressure gradient): MOM6's PFu, PFv of the one-step 'pgf0' run (TIDES and CALCULATE_SAL off,
BEGW = 0: the force of the state at the start of the step, MOM_dynamics_split_RK2.F90:511-515, posted at :1120)
against pyOCEAN's ``SigmaBaroclinic.pressure_gradient`` on the **same file** (``ic_full.nc``, which is pyOCEAN's own
initial state, ``mom6_inputs.py``), η = Σh − D as MOM6 sees it.  pyOCEAN's pressure gradient leaves out the
surface term −g ∇η (the barotropic solver's, ``core/vertical/pgf_nonlinear.py:415-417``), so the comparison is
PFu = pgf_pyOCEAN − g (η_{i+1} − η_i)/dx_u.  The pyOCEAN model is the parity configuration with MOM6's NWA12
settings where they differ (``PGF_SETTINGS``: G_EARTH = 9.8 and MASS_WEIGHT_IN_PRESSURE_GRADIENT with
MASS_WEIGHT_IN_PGF_VANISHED_ONLY = False ↔ ``pgf_mass_weight='full'``); the parity configuration itself (g = 9.81,
'vanished') is reported too.  Faces: both columns wet, not on the domain edge.

Criterion 4 (tides, helper): ``mom6_series`` turns MOM6's hourly SSH snapshots at the gauges' cells
(``inputs/pyocean/stations.nc``) into a ``series.npz`` that ``tide_check.py`` analyses like a pyOCEAN run.

Usage: python validation/nwa12_cut/check_mom6.py [--static RUN] [--pgf RUN] [--series RUN] [--cost RUN …]
Index order: MOM6 files (time, z, y, x) surface first; pyOCEAN (k, j, i) with k = 0 at the bed.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import netCDF4
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.nwa12_cut import case as cs  # noqa: E402

RUNS = cs.RUNS / "runs"
INPUTS = cs.RUNS / "inputs" / "mom6"
TOL_GRID = 1e-12                  # criterion 1, relative
TOL_PGF = 1e-11                   # criterion 7 [m s⁻²]
#: MOM6 NWA12 settings that enter the PGF and differ from pyOCEAN's parity_config (NWA12 MOM_input: G_EARTH = 9.8,
#: MASS_WEIGHT_IN_PRESSURE_GRADIENT = True, MASS_WEIGHT_IN_PGF_VANISHED_ONLY = False)
PGF_SETTINGS = dict(g=9.8, pgf_mass_weight="full")

STATIC = (  # (MOM6 name, pyOCEAN name, sign, where)
    ("geolon", "lon_t", 1, "all"), ("geolat", "lat_t", 1, "all"),
    ("geolon_u", "lon_u", 1, "all"), ("geolat_u", "lat_u", 1, "all"),
    ("geolon_v", "lon_v", 1, "all"), ("geolat_v", "lat_v", 1, "all"),
    ("geolon_c", "lon_q", 1, "all"), ("geolat_c", "lat_q", 1, "all"),
    ("dxt", "dx_t", 1, "all"), ("dyt", "dy_t", 1, "all"), ("area_t", "area_t", 1, "all"),
    ("dxCu", "dx_u", 1, "all"), ("dyCu", "dy_u", 1, "all"), ("dxCv", "dx_v", 1, "all"), ("dyCv", "dy_v", 1, "all"),
    ("dyCuo", "dy_u", 1, "wet_u"), ("dxCvo", "dx_v", 1, "wet_v"),
    ("area_u", "area_u", 1, "wet_u"), ("area_v", "area_v", 1, "wet_v"), ("area_q", "area_q", 1, "all"),
    ("Coriolis", "f_q", 1, "all"), ("cos_rot", "cos_t", 1, "all"), ("sin_rot", "sin_t", -1, "all"),
)


def _find(run: Path, stem: str) -> Path:
    fs = sorted(p for p in Path(run).glob(f"*{stem}*.nc") if not p.name.startswith("._"))
    if not fs:
        raise FileNotFoundError(f"no {stem} file in {run}")
    return fs[0]


def _read(ds: netCDF4.Dataset, name: str) -> np.ndarray:
    v = ds[name]
    v.set_auto_mask(False)
    a = np.asarray(v[:], float)
    miss = [v.getncattr(k) for k in ("missing_value", "_FillValue") if k in v.ncattrs()]
    for m in miss:
        a = np.where(a == m, np.nan, a)
    return a.squeeze()


# ------------------------------------------------------------------------------------------------ criterion 1
def static_check(run: Path = RUNS / "mom6_pgf0", case: cs.Case = cs.CASE) -> dict:
    """{MOM6 name: (max relative error, n compared)} of ocean_static.nc against grid_from_supergrid."""
    hg, H = cs.grid(case)
    out = {}
    with netCDF4.Dataset(_find(run, "ocean_static")) as ds:
        masks = {w: _read(ds, w) > 0.5 for w in ("wet_u", "wet_v")}
        for mom, pyo, sign, where in STATIC:
            a = _read(ds, mom)
            b = sign * hg[pyo].values
            if a.shape != b.shape:
                raise ValueError(f"{mom} {a.shape} vs {pyo} {b.shape}")
            sel = np.isfinite(a) & (masks[where] if where != "all" else True)
            d = np.abs(a - b)[sel]
            den = np.abs(b)[sel]
            rel = np.where(den > 0, d / np.where(den > 0, den, 1.0), d)
            out[mom] = (float(rel.max()) if rel.size else 0.0, int(sel.sum()))
    return out


# ------------------------------------------------------------------------------------------------ criterion 7
def read_ic(path: Path = INPUTS / "ic_full.nc") -> dict:
    """The MOM6 initial-condition file in pyOCEAN order (k = 0 at the bed)."""
    with netCDF4.Dataset(path) as ds:
        return {k: np.asarray(ds[k][:], float)[::-1].copy() for k in ("h", "temp", "salt", "u", "v")}


def pyocean_pgf(ic: dict, case: cs.Case = cs.CASE, **settings) -> tuple[np.ndarray, np.ndarray, dict]:
    """pyOCEAN's total pressure-gradient force (PFu, PFv convention: with −g ∇η) of the state ``ic`` (k = 0 at the
    bed), for the parity configuration with ``settings`` overriding it.  Returns (fu, fv, info)."""
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    hg, H = cs.grid(case)
    edges = cs.open_edges(H)
    cfg = cs.parity_config(case, hg, H, 600.0, edges=edges, **settings)
    for k in ("open_boundary_state", "boundary_tides", "surface_fluxes", "runoff"):
        cfg[k] = None
    cfg.update(tidal_potential=False, sal_flather=False, open_boundaries=(), open_boundary_mode=None)
    m = SigmaBaroclinic(hg, H, **cfg)
    wet = np.isfinite(H)
    h = np.where(wet[None], ic["h"], 0.0)
    eta = np.where(wet, h.sum(0) - np.nan_to_num(H), 0.0)
    fu, fv = m.pressure_gradient(eta, ic["temp"], h, ic["salt"])
    g = float(m.g) if hasattr(m, "g") else float(settings.get("g", 9.81))
    dxu, dyv = hg["dx_u"].values, hg["dy_v"].values
    su = np.zeros(fu.shape[1:])
    sv = np.zeros(fv.shape[1:])
    su[:, 1:-1] = -g * (eta[:, 1:] - eta[:, :-1]) / dxu[:, 1:-1]
    sv[1:-1, :] = -g * (eta[1:, :] - eta[:-1, :]) / dyv[1:-1, :]
    return fu + su[None], fv + sv[None], dict(g=g, pgf_mass_weight=cfg.get("pgf_mass_weight", "vanished"),
                                              rho_pgf_ref=m.rho_pgf_ref)


def _interior_faces(H: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    wet = np.isfinite(H)
    mu = np.zeros((H.shape[0], H.shape[1] + 1), bool)
    mv = np.zeros((H.shape[0] + 1, H.shape[1]), bool)
    mu[:, 1:-1] = wet[:, 1:] & wet[:, :-1]
    mv[1:-1, :] = wet[1:, :] & wet[:-1, :]
    return mu, mv


def pgf_check(run: Path = RUNS / "mom6_pgf0", ic_path: Path = INPUTS / "ic_full.nc", case: cs.Case = cs.CASE,
              settings: dict | None = None) -> dict:
    """Max |PFu − pyOCEAN|, |PFv − pyOCEAN| [m s⁻²] over the interior wet faces (all layers, and layers whose two
    neighbouring cells are both thicker than 1 mm), with the max |PF| for scale."""
    settings = PGF_SETTINGS if settings is None else settings
    with netCDF4.Dataset(_find(run, "pgf0")) as ds:
        pu = np.nan_to_num(_read(ds, "PFu"))[::-1]
        pv = np.nan_to_num(_read(ds, "PFv"))[::-1]
        hm = np.nan_to_num(_read(ds, "h"))[::-1]
    ic = read_ic(ic_path)
    fu, fv, info = pyocean_pgf(ic, case, **settings)
    hg, H = cs.grid(case)
    mu, mv = _interior_faces(H)
    h = ic["h"]
    thick_u = (h[:, :, 1:] > 1e-3) & (h[:, :, :-1] > 1e-3)
    thick_v = (h[:, 1:, :] > 1e-3) & (h[:, :-1, :] > 1e-3)
    tu = np.zeros_like(pu, bool)
    tv = np.zeros_like(pv, bool)
    tu[:, :, 1:-1] = thick_u
    tv[:, 1:-1, :] = thick_v
    du = np.abs(pu - fu) * mu[None]
    dv = np.abs(pv - fv) * mv[None]
    return dict(settings=info, max_du=float(du.max()), max_dv=float(dv.max()),
                max_du_thick=float((du * tu).max()), max_dv_thick=float((dv * tv).max()),
                max_pfu=float(np.abs(pu * mu[None]).max()), max_pfv=float(np.abs(pv * mv[None]).max()),
                argmax_du=[int(x) for x in np.unravel_index(np.argmax(du), du.shape)],
                # informative only: MOM6's h in the pgf0 file is the mean over the step, after the end-of-step
                # z* regrid/remap, not the initial h
                h_mom6_step_mean_minus_ic=float(np.abs(hm - h).max()))


# ------------------------------------------------------------------------------------------------ criterion 4
def mom6_series(run: Path = RUNS / "mom6_tides", case: cs.Case = cs.CASE) -> Path:
    """series.npz (t [s from case.t0], names, eta (nt, n)) of MOM6's hourly SSH snapshots at the gauges' cells, the
    layout of run_pyocean.py's series that tide_check.py reads."""
    import xarray as xr
    with xr.open_dataset(case.inputs / "stations.nc") as ds:
        names = [str(x) for x in ds["station"].values]
        jj, ii = ds["j"].values.astype(int), ds["i"].values.astype(int)
    keep = [k for k in range(len(names)) if jj[k] >= 0]
    with netCDF4.Dataset(_find(run, "ocean_hourly")) as ds:
        tv = ds["time"]
        t = netCDF4.num2date(tv[:], tv.units, getattr(tv, "calendar", "gregorian"))
        ssh = _read(ds, "SSH")
    ts = np.array([np.datetime64(x.isoformat(), "ns") for x in t])
    sec = (ts - case.t0) / np.timedelta64(1, "s")
    eta = np.stack([ssh[:, jj[k], ii[k]] for k in keep], axis=1)
    p = Path(run) / "series.npz"
    np.savez(p, t=sec, names=np.array([names[k] for k in keep]), eta=eta)
    return p


# ------------------------------------------------------------------------------------------------ cost
def cost(run: Path) -> dict:
    """Wall time and seconds per simulated day of a finished run (``timing.json`` written by run_mom6.sh)."""
    t = json.loads((Path(run) / "timing.json").read_text())
    days = t["days"] if t["days"] > 0 else t["seconds"] / 86400.0
    return dict(wall_s=t["wall_s"], days=days, s_per_day=t["wall_s"] / days, nproc=t["nproc"])


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--static", type=Path, nargs="?", const=RUNS / "mom6_pgf0")
    ap.add_argument("--pgf", type=Path, nargs="?", const=RUNS / "mom6_pgf0")
    ap.add_argument("--series", type=Path, nargs="?", const=RUNS / "mom6_tides")
    ap.add_argument("--cost", type=Path, nargs="*")
    a = ap.parse_args()
    if a.static:
        r = static_check(a.static)
        for k, (e, n) in r.items():
            print(f"{k:>10}: max rel {e:.2e} ({n} points) {'ok' if e <= TOL_GRID else 'FAIL'}")
    if a.pgf:
        for s in (PGF_SETTINGS, {}):
            r = pgf_check(a.pgf, settings=s)
            print(json.dumps(r, indent=1))
    if a.series:
        print(mom6_series(a.series))
    for run in a.cost or []:
        print(run.name, cost(run))


if __name__ == "__main__":
    main()
