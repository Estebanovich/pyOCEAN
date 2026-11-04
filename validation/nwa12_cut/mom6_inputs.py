#!/usr/bin/env python
"""
MOM6 + SIS2 inputs of the NWA12 cut (regional chain, plan Task 10; spec docs/superpowers/specs/
2026-11-01-regional-chain-design.md §F, criteria 1, 4 and 7), written from the same case definition as pyOCEAN
(``case.py``) to ``RUNS/inputs/mom6/`` (never to the repository).

What is written and how (CEFI tools: docs/design/regional_refs/cefi_tools.md; MOM6 grid: mom6_grid.md §7):

| file | how |
|---|---|
| ``ocean_hgrid.nc`` | the NWA12 supergrid cut at the case window with even offsets (mom6_grid.md §7), every variable kept |
| ``ocean_topog.nc`` | THE case topography (``case.grid``: MOM6 depth rules + ruling R14 edge edit), float64, ``ntiles`` dim (as ``make_mosaic.sh``'s ``ncap2``) |
| ``ocean_mask.nc``, ``land_mask.nc`` | 1/0 from the case topography (ocean / land fraction at T cells, ``make_quick_mosaic`` layout) |
| ``ocean_mosaic.nc``, ``grid_spec.nc`` | NWA12's files (they only name the tile files, same names on the cut) |
| ``*_mosaic_tile1X*_mosaic_tile1.nc`` | NWA12's exchange grids re-indexed to the window; each cell is in the atmosphere×ocean (and land×ocean) grid when the case topography is ocean, in the atmosphere×land grid otherwise, with its own NWA12 exchange area (no FRE-NCtools: atmosphere = land = ocean grid, so ``make_quick_mosaic`` gives one exchange cell per model cell, cefi_tools.md §6) |
| ``vgrid_75_2m.nc``, ``diag_dz.nc`` | NWA12's files |
| ``ic_full.nc`` | **pyOCEAN's own initial state** of the parity run (``case.build(CASE, 'full')``: θ, S_P remapped from GLORYS, η = zos, u, v from GLORYS), layers flipped to surface first: h (THICKNESS_CONFIG = "thickness_file"), temp, salt (TS_CONFIG = "file"), u, v (VELOCITY_CONFIG = "file"). Both models start from the same (h, θ, S_P, u, v), so criterion 7 compares the PGF of one state; CEFI's ``write_glorys_initial.py`` (z* centres + MOM6's own ALE remap) is not used (class c, it would start MOM6 from a different state) |
| ``ic_tides.nc`` | pyOCEAN's tides-only initial state (θ = 10 °C, S_P = 35, at rest, η = 0) |
| ``{uv,thetao,so,zos}_00n.nc`` | GLORYS segments, CEFI ``write_glorys_boundary.py`` algorithm (below) |
| ``{tu,tz}_00n.nc`` | TPXO10 tidal segments, CEFI ``write_tpxo_boundary.py`` algorithm on TPXO10 (ruling R3/R19) |
| ``ERA5_{var}_2017_padded.nc`` | CEFI ``tools/atmos`` layout for FMS ``data_override`` (below) |
| ``seawifs-clim-1997-2010.nwa12_cut.nc`` | SeaWiFS chlorophyll for MANIZZA_05 in the reference run: the OM4 climatology flooded and regridded conservatively to the cut as CEFI ``tools/opacity/regrid_opacity.py`` makes NWA12's file (rulings R23, R25; ``write_chl``) |
| ``glofas_runoff_2017.nc`` | the case's GloFAS v4 river input (``inputs/pyocean/rivers.nc``, CEFI ldd rule, R11/R12/R17) as runoff [kg m⁻² s⁻¹] on the cut (ruling R8, below) |

**CEFI tools without xesmf / HCtFlood.**  ``xesmf`` and ``HCtFlood`` are not installed and no conda environment has
them (2026-11-02; the brief forbids installing into ``base``).  CEFI's own pure-NumPy functions are executed from
the CEFI source itself (``_cefi``: ``rotate_uv``, ``fill_missing``, ``z_to_dz``, ``ap2ep``, ``ep2ap`` of
``tools/boundary/boundary.py``, git 36b8f49, parsed and compiled from the file, not copied); the two xesmf/HCtFlood
steps are reimplemented here (class c, documented):

* ``nearest_s2d`` (xesmf/ESMF nearest source to destination): the nearest source point to each destination point in
  3-D Cartesian distance on the unit sphere (scipy ``cKDTree``); ties broken by the tree, which may differ from
  ESMF's at exactly equidistant points;
* ``cdo setmisstonn`` (GLORYS pre-fill, ``boundary/fill_glorys_nn.sh:32``): every missing point of a level takes the
  value of the nearest valid point of the same level and time (same Cartesian distance);
* ``HCtFlood.kara.flood_kara`` (TPXO flood over land, ``boundary.py:67-101``): iterative flood, each missing point
  with at least one valid point among its 8 neighbours takes their mean, until none is missing [verificar against
  HCtFlood's source, not in ``~/pyOCEAN_refs``].

GLORYS segments (``boundary.py:420-706``): GLORYS pre-filled → nearest point of each supergrid node of the edge
(``Segment.coords``, BRUSHCUTTER_MODE) → velocities rotated with ``angle_dx`` (0 on this cut) → ``fill_missing``
(bfill along the segment, ffill in z, NaN → 0) → layer thicknesses ``z_to_dz`` (interfaces half-way between GLORYS
levels, last at 6500 m), files ``NETCDF3_64BIT``, ``_FillValue`` 1e20, time unlimited.  The GLORYS daily records are
stamped at 00:00 already (CEFI floors the first one to midnight).

TPXO segments (``write_tpxo_boundary.py``): TPXO10 atlas v2 (local) instead of TPXO9 v1; the velocities are TPXO's
own, u = U/h_TPXO at the U nodes and v = V/h_TPXO at the V nodes (ruling R19, the same as pyOCEAN's
``velocity_depth='tpxo'``; CEFI reads TPXO9's ``ua``, ``va`` velocity files); complex h = hRe + i hIm (= ha e^{−i hp},
``write_tpxo_boundary.py:22``); flood over land, nearest supergrid node, ``fill_missing``, zamp = |h|,
zphase = −arg h; u, v → ellipse (``ap2ep``), inclination − angle_dx, back (``ep2ap``); one dummy time
1980-12-01, constituents in the order of OBC_TIDE_CONSTITUENTS (M2 S2 N2 K2 K1 O1 P1 Q1 MM MF).

ERA5 (``tools/atmos/pad_era5.py``, ``era5_lp.sh``, ``era5_sphum.py``): the case's monthly CDS files concatenated in
time, latitude south → north, unpacked float32, ``time`` the record dimension in hours since 1900-01-01
(gregorian); longitudes in the supergrid's −180…180 convention.  Differences from CEFI (class c): 2-m specific
humidity from ``d2m`` and **``msl``** (CEFI uses the surface pressure ``sp``, which the case did not download;
pyOCEAN's ERA5 adapter uses ``msl`` too); liquid precipitation lp = tp and no snowfall file (CEFI lp = max(tp − sf, 0)
and fprec = sf; ``sf`` was not downloaded, May–August south of 42 °N); the files cover 2017-04-29 → 2017-09-03 (CEFI
pads a year with the next 1 January).  Scale factors stay in ``data_table`` as NWA12's (J m⁻² h⁻¹ → W m⁻², m h⁻¹ →
kg m⁻² s⁻¹).

Rivers (ruling R8): CEFI's ``write_runoff_glofas.py`` needs xesmf and the full-NWA12 GloFAS field (the case
downloaded only the cut's box), so the pyOCEAN-equivalent file is used: the case's 72 river cells (CEFI ldd rule,
nearest coast cell) with runoff = Q ρ_fw / A_t, ρ_fw = 1000 kg m⁻³ (CEFI's conversion) and A_t the cut's T-cell area
(``grid_from_supergrid`` = MOM6 areaT), at the daily times of ``rivers.nc``.  MOM6 (Boussinesq) turns the mass flux
into volume with RHO_0 = 1035, pyOCEAN adds the volume Q: MOM6's river volume is 1000/1035 of pyOCEAN's (class c).

Usage: python validation/nwa12_cut/mom6_inputs.py [--only grid,ic,obc,tides,era5,runoff,chl]
       python validation/nwa12_cut/mom6_inputs.py --only ic_pert      (ic_full_pert.nc of the diagnostic twin)
       python validation/nwa12_cut/mom6_inputs.py --run-dir {pgf0,tides,full,full_jerlov,full_pert,full_aspy[_<sw>-<sw>…][_pert]}
       (run_mom6.sh calls it; full_aspy switches off the MOM6 candidates of the Gulf Stream drift, ASPY_SWITCHES)
Index order: pyOCEAN (k, j, i) with k = 0 at the bed; MOM6 files are surface first, (z, y, x).
"""
from __future__ import annotations

import argparse
import ast
import json
import shutil
import sys
from pathlib import Path

import netCDF4
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.nwa12_cut import case as cs  # noqa: E402

CEFI = cs.REFS / "CEFI-regional-MOM6"
NWA12_GRID = cs.REFS / "nwa12_grid"
OUT = cs.RUNS / "inputs" / "mom6"
#: segment numbers (NWA12: 1 south, 2 north, 3 east; the cut adds 4 west) and MOM6 segment strings
#: (``OBC_SEGMENT_00n``; NWA12 MOM_input, the west one with the same orientation rule)
SEGMENTS = {"south": 1, "north": 2, "east": 3, "west": 4}
SEGMENT_STRING = {"south": "J=0,I=0:N", "north": "J=N,I=N:0", "east": "I=N,J=0:N", "west": "I=0,J=N:0"}
TIME_UNITS = "days since 1950-01-01 00:00:00"
FILL = 1.0e20
CHL_FILE = "seawifs-clim-1997-2010.nwa12_cut.nc"
#: NWA12's chlorophyll (Zenodo 7893727 settings, physics-only MOM_input:640-642: CHL_FROM_FILE = True, CHL_FILE =
#: "seawifs-clim-1997-2010.nwa12.nc", CHL_VARNAME = "chlor_a") is not public; it is made by CEFI
#: tools/opacity/regrid_opacity.py from this OM4 tripolar climatology (ruling R25; CESM inputdata
#: ocn/mom/tx0.25v1/), which ``write_chl`` regrids the same way onto the cut
CHL_SOURCE = cs.REFS / "nwa12_grid" / "seawifs-clim-1997-2010.1440x1080.v20180328.nc"


# ------------------------------------------------------------------------------------------------ CEFI functions
def _cefi(names: tuple[str, ...] = ("rotate_uv", "fill_missing", "z_to_dz", "ap2ep", "ep2ap")) -> dict:
    """CEFI's pure-NumPy/xarray functions, compiled from ``tools/boundary/boundary.py`` itself (the module imports
    xesmf at the top, so it cannot be imported here)."""
    src_path = CEFI / "tools" / "boundary" / "boundary.py"
    tree = ast.parse(src_path.read_text())
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    missing = set(names) - {n.name for n in keep}
    if missing:
        raise RuntimeError(f"{src_path} lacks {missing}")
    ns: dict = {"np": np, "xarray": xr}
    exec(compile(ast.Module(body=keep, type_ignores=[]), str(src_path), "exec"), ns)
    return {n: ns[n] for n in names}


# ------------------------------------------------------------------------------------------------ geometry helpers
def _xyz(lon: np.ndarray, lat: np.ndarray) -> np.ndarray:
    """Unit vectors on the sphere of points (lon, lat) [degrees] → (n, 3)."""
    lo, la = np.deg2rad(np.ravel(lon)), np.deg2rad(np.ravel(lat))
    return np.column_stack([np.cos(la) * np.cos(lo), np.cos(la) * np.sin(lo), np.sin(la)])


def nearest_s2d(src_lon: np.ndarray, src_lat: np.ndarray, dst_lon: np.ndarray, dst_lat: np.ndarray,
                valid: np.ndarray | None = None) -> np.ndarray:
    """Flat index of the nearest source point (among ``valid``) of every destination point (xesmf ``nearest_s2d``,
    Cartesian distance on the unit sphere).  src_* (any shape, same), dst_* (m,)."""
    from scipy.spatial import cKDTree
    src = _xyz(src_lon, src_lat)
    idx = np.arange(src.shape[0])
    if valid is not None:
        idx = idx[np.ravel(valid)]
        src = src[idx]
    _, k = cKDTree(src).query(_xyz(dst_lon, dst_lat))
    return idx[k]


def setmisstonn(field: np.ndarray, lon2: np.ndarray, lat2: np.ndarray) -> np.ndarray:
    """``cdo setmisstonn`` per 2-D slice: NaN → value of the nearest valid point of the same slice.  ``field``
    (..., ny, nx); the valid mask is taken per slice."""
    f = np.array(field, dtype=float, copy=True)
    flat = f.reshape(-1, f.shape[-2] * f.shape[-1])
    cache: dict = {}
    for s in range(flat.shape[0]):
        bad = np.isnan(flat[s])
        if not bad.any() or bad.all():
            continue
        key = bad.tobytes()
        if key not in cache:
            src = nearest_s2d(lon2, lat2, lon2.ravel()[bad], lat2.ravel()[bad], valid=~bad)
            cache[key] = src
        flat[s, bad] = flat[s, cache[key]]
    return f


def flood(field: np.ndarray, max_iter: int = 10000) -> np.ndarray:
    """Iterative 8-neighbour mean flood of NaN points over (..., ny, nx) (stand-in for HCtFlood ``flood_kara``)."""
    f = np.array(field, dtype=float, copy=True)
    for _ in range(max_iter):
        bad = np.isnan(f)
        if not bad.any():
            return f
        p = np.pad(f, [(0, 0)] * (f.ndim - 2) + [(1, 1), (1, 1)], constant_values=np.nan)
        ny, nx = f.shape[-2:]
        stack = np.stack([p[..., 1 + dj:1 + dj + ny, 1 + di:1 + di + nx]
                          for dj in (-1, 0, 1) for di in (-1, 0, 1) if (dj, di) != (0, 0)])
        n = np.sum(np.isfinite(stack), axis=0)
        s = np.nansum(stack, axis=0)
        fill = bad & (n > 0)
        if not fill.any():
            return f                                      # slices that are all NaN stay NaN
        f[fill] = s[fill] / n[fill]
    raise RuntimeError("flood did not converge")


# ------------------------------------------------------------------------------------------------ netCDF helpers
def _copy_vars(src: netCDF4.Dataset, dst: netCDF4.Dataset, slices: dict | None = None) -> None:
    """Copy dimensions, variables and attributes of ``src`` into ``dst``, slicing dimensions per ``slices``."""
    slices = slices or {}
    for name, dim in src.dimensions.items():
        n = len(range(*slices[name].indices(len(dim)))) if name in slices else len(dim)
        dst.createDimension(name, None if dim.isunlimited() else n)
    dst.setncatts({a: src.getncattr(a) for a in src.ncattrs()})
    for name, var in src.variables.items():
        fill = var.getncattr("_FillValue") if "_FillValue" in var.ncattrs() else None
        out = dst.createVariable(name, var.datatype, var.dimensions, fill_value=fill)
        out.setncatts({a: var.getncattr(a) for a in var.ncattrs() if a != "_FillValue"})
        sl = tuple(slices.get(d, slice(None)) for d in var.dimensions)
        var.set_auto_maskandscale(False)
        out.set_auto_maskandscale(False)
        out[...] = var[sl] if var.dimensions else var[...]


def _time_var(ds: netCDF4.Dataset, times: np.ndarray, units: str = TIME_UNITS) -> None:
    t = ds.createVariable("time", "f8", ("time",))
    t.units, t.calendar, t.cartesian_axis = units, "gregorian", "T"
    t[:] = netCDF4.date2num([_pydt(x) for x in np.asarray(times)], units, "gregorian")


def _pydt(t: np.datetime64):
    from datetime import datetime
    return datetime.fromisoformat(str(np.datetime64(t, "s")))


# ------------------------------------------------------------------------------------------------ grid
def write_grid(case: cs.Case, out: Path) -> dict:
    """Supergrid, topography, masks, mosaics and exchange grids of the cut (module table)."""
    i0, i1, j0, j1 = cs.window_of(case)
    sl = {"nyp": slice(2 * j0, 2 * j1 + 1), "nxp": slice(2 * i0, 2 * i1 + 1),
          "ny": slice(2 * j0, 2 * j1), "nx": slice(2 * i0, 2 * i1)}
    with netCDF4.Dataset(case.hgrid) as src, netCDF4.Dataset(out / "ocean_hgrid.nc", "w") as dst:
        _copy_vars(src, dst, sl)
        dst.history = f"cut of NWA12 ocean_hgrid.nc, window (i0, i1, j0, j1) = {(i0, i1, j0, j1)} (pyOCEAN Task 10)"
    hg, H = cs.grid(case)
    ny, nx = H.shape
    ocean = np.isfinite(H)
    with netCDF4.Dataset(out / "ocean_topog.nc", "w") as ds:
        ds.createDimension("ntiles", 1)
        ds.createDimension("ny", ny)
        ds.createDimension("nx", nx)
        ds.createVariable("ntiles", "i4", ("ntiles",))[:] = 1
        d = ds.createVariable("depth", "f8", ("ny", "nx"))
        d.units, d.standard_name = "meters", "topographic depth at Arakawa C h-points"
        d[:] = np.nan_to_num(H)
        ds.comment = ("NWA12 ocean_topog.nc cut to the case window, MOM6 depth rules (MASKING_DEPTH 1, MINIMUM_DEPTH "
                      "4, MAXIMUM_DEPTH 6500) and the edge edit of ruling R14 (case.grid): the case topography")
    for fname, val, std in (("ocean_mask.nc", ocean, "ocean fraction at T-cell centers"),
                            ("land_mask.nc", ~ocean, "land fraction at T-cell centers")):
        with netCDF4.Dataset(out / fname, "w") as ds:
            ds.createDimension("ny", ny)
            ds.createDimension("nx", nx)
            m = ds.createVariable("mask", "f8", ("ny", "nx"))
            m.standard_name, m.units = std, "none"
            m[:] = val.astype(float)
    for fname in ("ocean_mosaic.nc", "grid_spec.nc", "vgrid_75_2m.nc", "diag_dz.nc"):
        shutil.copyfile(NWA12_GRID / fname, out / fname)
    # exchange grids: per-cell area from NWA12's atmXocn (ocean cells) and atmXlnd (land cells)
    area = np.full((ny, nx), np.nan)
    for fname in ("atmos_mosaic_tile1Xocean_mosaic_tile1.nc", "atmos_mosaic_tile1Xland_mosaic_tile1.nc"):
        with netCDF4.Dataset(NWA12_GRID / fname) as ds:
            c = np.asarray(ds["tile2_cell"][:]) - 1                          # 1-based (i, j) → 0-based
            a = np.asarray(ds["xgrid_area"][:], float)
            inside = (c[:, 0] >= i0) & (c[:, 0] < i1) & (c[:, 1] >= j0) & (c[:, 1] < j1)
            area[c[inside, 1] - j0, c[inside, 0] - i0] = a[inside]
    if not np.isfinite(area).all():
        raise RuntimeError("NWA12 exchange grids do not cover every cell of the window")
    jj, ii = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
    for fname, sel, contact in (("atmos_mosaic_tile1Xocean_mosaic_tile1.nc", ocean,
                                 "atmos_mosaic:tile1::ocean_mosaic:tile1"),
                                ("land_mosaic_tile1Xocean_mosaic_tile1.nc", ocean,
                                 "land_mosaic:tile1::ocean_mosaic:tile1"),
                                ("atmos_mosaic_tile1Xland_mosaic_tile1.nc", ~ocean,
                                 "atmos_mosaic:tile1::land_mosaic:tile1")):
        # NWA12's files run i fastest within j (tile1_cell (567,1), (568,1), …)
        cells = np.column_stack([ii[sel] + 1, jj[sel] + 1]).astype("i4")
        with netCDF4.Dataset(NWA12_GRID / fname) as src, netCDF4.Dataset(out / fname, "w") as ds:
            ds.createDimension("string", 255)
            ds.createDimension("ncells", cells.shape[0])
            ds.createDimension("two", 2)
            cv = ds.createVariable("contact", "S1", ("string",))
            cv.setncatts({a: src["contact"].getncattr(a) for a in src["contact"].ncattrs()})
            cv[:] = netCDF4.stringtoarr(contact, 255)
            for name in ("tile1_cell", "tile2_cell"):
                v = ds.createVariable(name, "i4", ("ncells", "two"))
                v.setncatts({a: src[name].getncattr(a) for a in src[name].ncattrs()})
                v[:] = cells
            v = ds.createVariable("xgrid_area", "f8", ("ncells",))
            v.setncatts({a: src["xgrid_area"].getncattr(a) for a in src["xgrid_area"].ncattrs()})
            v[:] = area[sel]
            for name in ("tile1_distance", "tile2_distance"):
                v = ds.createVariable(name, "f8", ("ncells", "two"))
                v.setncatts({a: src[name].getncattr(a) for a in src[name].ncattrs()})
                v[:] = 0.0
            ds.history = (f"NWA12 {fname} re-indexed to the window {(i0, i1, j0, j1)}, cells by the case topography "
                          "(pyOCEAN Task 10)")
    with netCDF4.Dataset(NWA12_GRID / "ocean_mask.nc") as ds:
        nwa_mask = np.asarray(ds["mask"][j0:j1, i0:i1]) > 0.5
    return dict(window=[i0, i1, j0, j1], ny=ny, nx=nx, n_ocean=int(ocean.sum()),
                mask_changes_vs_nwa12=[[int(j), int(i)] for j, i in zip(*np.nonzero(nwa_mask != ocean))])


# ------------------------------------------------------------------------------------------------ initial state
def write_ic(name: str, h: np.ndarray, T: np.ndarray, S: np.ndarray, u: np.ndarray, v: np.ndarray,
             out: Path, comment: str) -> Path:
    """MOM6 initial-condition file (surface first): h [m], temp θ [°C], salt S_P [PSU] on (zl, yh, xh), u on
    (zl, yh, xq) and v on (zl, yq, xh) with the symmetric face counts (nx + 1, ny + 1).  Inputs (k, j, i) with k = 0
    at the bed (pyOCEAN order)."""
    n, ny, nx = h.shape
    p = out / f"ic_{name}.nc"
    with netCDF4.Dataset(p, "w") as ds:
        for d, s in (("zl", n), ("yh", ny), ("xh", nx), ("yq", ny + 1), ("xq", nx + 1)):
            ds.createDimension(d, s)
        for var, val, dims, units in (("h", h, ("zl", "yh", "xh"), "m"), ("temp", T, ("zl", "yh", "xh"), "degC"),
                                      ("salt", S, ("zl", "yh", "xh"), "psu"), ("u", u, ("zl", "yh", "xq"), "m s-1"),
                                      ("v", v, ("zl", "yq", "xh"), "m s-1")):
            x = ds.createVariable(var, "f8", dims)
            x.units = units
            x[:] = np.asarray(val, float)[::-1]
        ds.comment = comment
    return p


def write_ics(case: cs.Case, out: Path, modes: tuple[str, ...] = ("full", "tides")) -> dict:
    """pyOCEAN's initial states of the 'full' and 'tides' runs (``case.build``) as MOM6 IC files."""
    info = {}
    hg, H = cs.grid(case)
    for mode in modes:
        b = cs.build(case, "full" if mode == "full_pert" else mode, dt=600.0, days=1.0, hg=hg, H=H)
        st = b.state
        T, note = st.T, ""
        if mode == "full_pert":
            T = cs.perturb_temperature(st.T, H, PERTURB_AMP, PERTURB_SEED)
            note = f" + case.perturb_temperature(amp {PERTURB_AMP} °C, seed {PERTURB_SEED}) on θ"
        p = write_ic(mode, st.h, T, st.S, st.u, st.v, out,
                     f"pyOCEAN case.build(CASE, '{mode}') initial state (validation/nwa12_cut/case.py){note}, "
                     "surface first")
        eta_h = np.where(np.isfinite(H), st.h.sum(0) - np.nan_to_num(H), 0.0)
        info[mode] = dict(file=p.name, max_abs_eta_minus_sum_h=float(np.abs(np.where(np.isfinite(H), st.eta, 0.0)
                                                                              - eta_h).max()))
    return info


# ------------------------------------------------------------------------------------------------ segments
def segment_coords(hgrid: xr.Dataset, border: str) -> xr.Dataset:
    """Supergrid nodes of an edge (CEFI ``Segment.coords``, boundary.py:334-360), angle_dx in radians."""
    sel = {"south": dict(nyp=0), "north": dict(nyp=-1), "west": dict(nxp=0), "east": dict(nxp=-1)}[border]
    ang = hgrid["angle_dx"]
    if ang.attrs.get("units") == "degrees":
        ang = np.radians(ang)
    return xr.Dataset({"lon": hgrid["x"].isel(**sel), "lat": hgrid["y"].isel(**sel), "angle": ang.isel(**sel)})


def _seg_dims(border: str, num: int) -> tuple[str, str, str]:
    s = f"segment_{num:03d}"
    loc = f"nx_{s}" if border in ("south", "north") else f"ny_{s}"
    return s, loc, (f"ny_{s}" if border in ("south", "north") else f"nx_{s}")


def _write_segment(path: Path, border: str, num: int, coords: xr.Dataset, times: np.ndarray,
                   fields: dict[str, np.ndarray], extra_dim: tuple[str, int] | None = None,
                   time_units: str = TIME_UNITS) -> None:
    """One segment file in CEFI's layout (``Segment.to_netcdf``, ``expand_dims``, ``rename_dims``): every field
    (time, [z|constituent,] locations) gets the unit dimension (ny for south/north, nx for east/west) and the
    location dimension, NETCDF3_64BIT, _FillValue 1e20."""
    s, loc, unit = _seg_dims(border, num)
    n = coords["lon"].size
    with netCDF4.Dataset(path, "w", format="NETCDF3_64BIT_OFFSET") as ds:
        ds.createDimension("time", None)
        if extra_dim is not None:
            ds.createDimension(*extra_dim)
        ds.createDimension(loc, n)
        ds.createDimension(unit, 1)
        _time_var(ds, times, time_units)
        for c in ("lon", "lat"):
            v = ds.createVariable(f"{c}_{s}", "f8", (loc,))
            v[:] = coords[c].values
        for name, val in fields.items():
            val = np.asarray(val, float)
            mid = () if val.ndim == 2 else (extra_dim[0],)
            dims = ("time",) + mid + ((unit, loc) if border in ("south", "north") else (loc, unit))
            v = ds.createVariable(name, "f8", dims, fill_value=FILL)
            v[:] = np.where(np.isfinite(val), val, FILL).reshape(val.shape[:-1] + (
                (1, n) if border in ("south", "north") else (n, 1)))


def write_glorys_segments(case: cs.Case, out: Path, t0: np.datetime64 | None = None,
                          t1: np.datetime64 | None = None) -> dict:
    """{uv,thetao,so,zos}_00n.nc from GLORYS with CEFI's algorithm (module docstring)."""
    f = _cefi()
    hgrid = xr.open_dataset(out / "ocean_hgrid.nc")
    edges = cs.open_edges(cs.grid(case)[1])
    with xr.open_dataset(case.glorys[0]) as g:
        t0 = np.datetime64("2017-04-29") if t0 is None else t0
        t1 = np.datetime64("2017-09-03") if t1 is None else t1
        g = g.sel(time=slice(t0, t1)).load()
    lon2, lat2 = np.meshgrid(g["longitude"].values.astype(float), g["latitude"].values.astype(float))
    times = g["time"].values
    z = g["depth"].values.astype(float)
    filled = {v: setmisstonn(g[v].values.astype(float), lon2, lat2) for v in ("thetao", "so", "uo", "vo", "zos")}
    info = {"times": [str(times[0]), str(times[-1]), int(times.size)], "levels": int(z.size), "segments": {}}
    for border in edges:
        num = SEGMENTS[border]
        s = f"segment_{num:03d}"
        co = segment_coords(hgrid, border)
        k = nearest_s2d(lon2, lat2, co["lon"].values, co["lat"].values)
        sel = {v: a.reshape(a.shape[:-2] + (-1,))[..., k] for v, a in filled.items()}

        def da(a, zdim=True):
            dims = ("time", "z", "locations") if zdim else ("time", "locations")
            coords = {"time": times, "locations": np.arange(k.size)}
            if zdim:
                coords["z"] = z
            return xr.DataArray(a, dims=dims, coords=coords)

        ang = xr.DataArray(co["angle"].values, dims=("locations",))
        ur, vr = f["rotate_uv"](da(sel["uo"]), da(sel["vo"]), ang)
        uv = f["fill_missing"](xr.Dataset({"u": ur, "v": vr}), fill="b").transpose("time", "z", "locations")
        dz = f["z_to_dz"](uv).values
        _write_segment(out / f"uv_{num:03d}.nc", border, num, co, times,
                       {f"u_{s}": uv["u"].values, f"v_{s}": uv["v"].values, f"dz_u_{s}": dz, f"dz_v_{s}": dz},
                       extra_dim=(f"nz_{s}", z.size))
        for var in ("thetao", "so"):
            tr = f["fill_missing"](xr.Dataset({var: da(sel[var])}), fill="b").transpose("time", "z", "locations")
            _write_segment(out / f"{var}_{num:03d}.nc", border, num, co, times,
                           {f"{var}_{s}": tr[var].values, f"dz_{var}_{s}": f["z_to_dz"](tr).values},
                           extra_dim=(f"nz_{s}", z.size))
        zz = f["fill_missing"](xr.Dataset({"zos": da(sel["zos"], False)}), zdim=None, fill="b")
        _write_segment(out / f"zos_{num:03d}.nc", border, num, co, times, {f"zos_{s}": zz["zos"].values})
        info["segments"][border] = dict(num=num, nodes=int(k.size))
    return info


def write_tpxo_segments(case: cs.Case, out: Path) -> dict:
    """{tu,tz}_00n.nc from TPXO10 with CEFI's algorithm (module docstring)."""
    f = _cefi()
    hgrid = xr.open_dataset(out / "ocean_hgrid.nc")
    hg, H = cs.grid(case)
    edges = cs.open_edges(H)
    tc = cs.tide_constants(case, hg)
    if list(tc["constituent"].values) != list(cs.CONSTITUENTS):
        raise ValueError(f"TPXO constituents {list(tc['constituent'].values)} != {cs.CONSTITUENTS}")
    lon_z, lat_z = np.meshgrid(tc["lon"].values, tc["lat"].values)
    lon_u, lat_u = np.meshgrid(tc["lon_u"].values, tc["lat"].values)
    lon_v, lat_v = np.meshgrid(tc["lon"].values, tc["lat_v"].values)
    h = tc["h"].values
    u = tc["U"].values / tc["hu"].values[None]            # TPXO's own velocity [m s⁻¹] (R19)
    v = tc["V"].values / tc["hv"].values[None]
    fl = {name: (flood(a.real), flood(a.imag)) for name, a in (("h", h), ("u", u), ("v", v))}
    times = np.array([np.datetime64("1980-12-01T00:00", "ns")])
    info = {}
    for border in edges:
        num = SEGMENTS[border]
        s = f"segment_{num:03d}"
        co = segment_coords(hgrid, border)
        out_c = {}
        for name, (lo, la) in (("h", (lon_z, lat_z)), ("u", (lon_u, lat_u)), ("v", (lon_v, lat_v))):
            k = nearest_s2d(lo, la, co["lon"].values, co["lat"].values)
            re, im = (x.reshape(x.shape[0], -1)[:, k] for x in fl[name])
            dims = ("constituent", "locations")
            re = f["fill_missing"](xr.DataArray(re, dims=dims), xdim="locations", zdim=None)
            im = f["fill_missing"](xr.DataArray(im, dims=dims), xdim="locations", zdim=None)
            out_c[name] = re.values + 1j * im.values
        zamp, zphase = np.abs(out_c["h"]), -np.angle(out_c["h"])
        uc = xr.DataArray(out_c["u"], dims=("constituent", "locations"))
        vc = xr.DataArray(out_c["v"], dims=("constituent", "locations"))
        SEMA, ECC, INC, PHA = f["ap2ep"](uc, vc)
        INC = INC - co["angle"].values[np.newaxis, :]
        ua, va, up, vp = f["ep2ap"](SEMA, ECC, INC, PHA)
        tu = {}
        for nm, a in ((f"uamp_{s}", ua), (f"vamp_{s}", va), (f"uphase_{s}", up), (f"vphase_{s}", vp)):
            a = f["fill_missing"](xr.DataArray(np.asarray(a, float), dims=("constituent", "locations")),
                                  xdim="locations", zdim=None)
            tu[nm] = a.values[None]
        nc = len(cs.CONSTITUENTS)
        _write_segment(out / f"tz_{num:03d}.nc", border, num, co, times,
                       {f"zamp_{s}": zamp[None], f"zphase_{s}": zphase[None]}, extra_dim=("constituent", nc))
        _write_segment(out / f"tu_{num:03d}.nc", border, num, co, times, tu, extra_dim=("constituent", nc))
        info[border] = dict(num=num, m2_zamp_range=[float(zamp[0].min()), float(zamp[0].max())])
    return info


# ------------------------------------------------------------------------------------------------ ERA5
ERA5_VARS = {"msl": "instant", "t2m": "instant", "u10": "instant", "v10": "instant", "d2m": "instant",
             "ssrd": "accum", "strd": "accum", "tp": "accum"}


def _era5_series(case: cs.Case, var: str) -> xr.DataArray:
    kind = ERA5_VARS[var]
    files = sorted(Path(case.era5).glob(case.era5_pattern.format(kind=kind)))
    parts = []
    for p in files:
        with xr.open_dataset(p, engine="h5netcdf") as ds:
            tn = "valid_time" if "valid_time" in ds.coords else "time"
            parts.append(ds[var].rename({tn: "time"}).load())
    da = xr.concat(parts, dim="time")
    _, keep = np.unique(da["time"].values, return_index=True)
    da = da.isel(time=np.sort(keep)).sortby("latitude")
    return da.drop_vars([c for c in da.coords if c not in ("time", "latitude", "longitude")])


def write_era5(case: cs.Case, out: Path) -> dict:
    """ERA5_{msl,t2m,u10,v10,sphum,ssrd,strd,lp}_2017_padded.nc for FMS data_override (module docstring)."""
    fields = {v: _era5_series(case, v) for v in ERA5_VARS}
    # 2-m specific humidity from dew point and pressure (CEFI tools/atmos/era5_sphum.py:12-36, msl for sp)
    td, p = fields.pop("d2m").astype(float), fields["msl"].astype(float)
    e = 611.2 * np.exp(17.67 * (td - 273.15) / (td - 29.65))
    r = 0.622 * e / (p - e)
    fields["sphum"] = (r / (1.0 + r)).astype("float32")
    fields["lp"] = fields.pop("tp").clip(min=0.0)
    info = {}
    for var, da in fields.items():
        times = da["time"].values
        lat = da["latitude"].values.astype(float)
        lon = ((da["longitude"].values.astype(float) + 180.0) % 360.0) - 180.0
        o = np.argsort(lon)
        path = out / f"ERA5_{var}_2017_padded.nc"
        with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
            ds.createDimension("time", None)
            ds.createDimension("lat", lat.size)
            ds.createDimension("lon", lon.size)
            _time_var(ds, times, "hours since 1900-01-01 00:00:00")
            la = ds.createVariable("lat", "f8", ("lat",))
            la.units, la.cartesian_axis, la.long_name = "degrees_north", "Y", "latitude"
            la[:] = lat
            lo = ds.createVariable("lon", "f8", ("lon",))
            lo.units, lo.cartesian_axis, lo.long_name = "degrees_east", "X", "longitude"
            lo[:] = lon[o]
            x = ds.createVariable(var, "f4", ("time", "lat", "lon"))
            x.units = {"sphum": "kg kg-1", "lp": "m"}.get(var, da.attrs.get("units", ""))
            val = np.asarray(da.values, "f4")[..., o]
            if not np.isfinite(val).all():
                raise ValueError(f"ERA5 {var} has non-finite values")
            x[:] = val
        info[var] = dict(file=path.name, times=[str(times[0]), str(times[-1]), int(times.size)])
    return info


# ------------------------------------------------------------------------------------------------ rivers
def write_runoff(case: cs.Case, out: Path) -> dict:
    """glofas_runoff_2017.nc: the case's river cells as runoff [kg m⁻² s⁻¹] on the cut (ruling R8)."""
    hg, H = cs.grid(case)
    with xr.open_dataset(case.inputs / "rivers.nc") as r:
        r = r.load()
    ny, nx = H.shape
    area = hg["area_t"].values
    j, i = r["j"].values.astype(int), r["i"].values.astype(int)
    Q = r["Q"].values                                        # (time, cell) [m³ s⁻¹]
    run = np.zeros((Q.shape[0], ny, nx))
    np.add.at(run, (slice(None), j, i), Q * 1000.0 / area[j, i])
    path = out / "glofas_runoff_2017.nc"
    with netCDF4.Dataset(path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("time", None)
        ds.createDimension("y", ny)
        ds.createDimension("x", nx)
        _time_var(ds, r["time"].values)
        yv = ds.createVariable("y", "f8", ("y",))
        yv.cartesian_axis = "Y"
        yv[:] = np.arange(1, ny + 1)
        xv = ds.createVariable("x", "f8", ("x",))
        xv.cartesian_axis = "X"
        xv[:] = np.arange(1, nx + 1)
        v = ds.createVariable("runoff", "f8", ("time", "y", "x"))
        v.units = "kg m-2 s-1"
        v.long_name = "GloFAS v4 river runoff, case river cells (CEFI ldd rule), Q * 1000 / areaT"
        v[:] = run
        a = ds.createVariable("area", "f8", ("y", "x"))
        a.units = "m2"
        a[:] = area
    total = float((run * area).sum(axis=(1, 2)).mean() / 1000.0)
    return dict(file=path.name, cells=int(j.size), mean_total_m3s=total,
                times=[str(r["time"].values[0]), str(r["time"].values[-1]), int(r["time"].size)])


# ------------------------------------------------------------------------------------------------ chlorophyll
def _overlap(src_b: np.ndarray, dst_b: np.ndarray) -> np.ndarray:
    """1-D overlap lengths (n_dst, n_src) of intervals with edges src_b (n_src + 1), dst_b (n_dst + 1), increasing."""
    lo = np.maximum(dst_b[:-1, None], src_b[None, :-1])
    hi = np.minimum(dst_b[1:, None], src_b[None, 1:])
    return np.clip(hi - lo, 0.0, None)


def write_chl(case: cs.Case, out: Path, src: Path = CHL_SOURCE, margin: float = 2.0) -> dict:
    """
    NWA12's chlorophyll for MANIZZA_05 on the cut, built as CEFI ``tools/opacity/regrid_opacity.py`` builds
    ``seawifs-clim-1997-2010.nwa12.nc`` (rulings R23, R25): the OM4 SeaWiFS climatology flooded over land
    (``flood_kara``, regrid_opacity.py:26; here the 8-neighbour flood stand-in, ``flood``), then regridded
    **conservatively** from its cell corners (``lon_crnr``, ``lat_crnr``) to the model cells bounded by the supergrid
    corners ``x[::2, ::2]``, ``y[::2, ::2]`` (regrid_opacity.py:10-15, 28-36; xesmf 'conservative'), written as
    ``chlor_a(time, yh, xh)`` with ``area``, ``lon_crnr``, ``lat_crnr`` and integer ``xh``, ``yh`` axes
    (:37-55) and the source's monthly modulo time axis.

    Without xesmf (class c): over the cut (and ``margin`` degrees around it) both grids are exactly rectilinear in
    longitude and latitude (checked here: the OM4 tripolar grid is regular Mercator south of 65 °N, the cut has angle
    0), so each overlap is a spherical rectangle with area R² Δλ Δ(sin φ) and the conservative weights are the exact
    separable product w = Δλ_overlap · Δ(sin φ)_overlap / (Δλ · Δ(sin φ))_target.  ESMF's cells have great-circle
    edges instead of parallels, a difference of second order in the cell size (c).  Every target cell is fully
    covered, so normalised and unnormalised conservative remapping agree.
    """
    if not Path(src).exists():
        raise FileNotFoundError(f"{src} not on disk (ruling R25: OM4 SeaWiFS climatology, CESM inputdata "
                                "ocn/mom/tx0.25v1/seawifs-clim-1997-2010.1440x1080.v20180328.nc)")
    with netCDF4.Dataset(out / "ocean_hgrid.nc") as h:
        xb, yb, area_sg = (np.asarray(h[v][:], float) for v in ("x", "y", "area"))
    xq, yq = xb[::2, ::2], yb[::2, ::2]
    if np.abs(xq - xq[:1]).max() > 0 or np.abs(yq - yq[:, :1]).max() > 0:
        raise ValueError("the cut is not rectilinear in lon/lat: the separable conservative remap does not apply")
    lon_t_b, lat_t_b = xq[0], yq[:, 0]
    with netCDF4.Dataset(src) as ds:
        lc = np.asarray(ds["lon_crnr"][:], float)
        la = np.asarray(ds["lat_crnr"][:], float)
        lc = ((lc - lon_t_b.min() + 180.0) % 360.0) + lon_t_b.min() - 180.0     # same longitude branch as the cut
        box = ((lc >= lon_t_b.min() - margin) & (lc <= lon_t_b.max() + margin)
               & (la >= lat_t_b.min() - margin) & (la <= lat_t_b.max() + margin))
        J, I = np.nonzero(box)
        j0, j1, i0, i1 = J.min(), J.max() + 1, I.min(), I.max() + 1      # corner indices
        lcs, las = lc[j0:j1, i0:i1], la[j0:j1, i0:i1]
        if np.abs(lcs - lcs[:1]).max() > 0 or np.abs(las - las[:, :1]).max() > 0:
            raise ValueError("source cells are not rectilinear around the cut")
        chl = np.asarray(ds["chlor_a"][:, j0:j1 - 1, i0:i1 - 1].filled(np.nan), float)
        tvar = ds["time"]
        t_attrs = {k: tvar.getncattr(k) for k in tvar.ncattrs()}
        times = np.asarray(tvar[:], float)
        c_attrs = {k: ds["chlor_a"].getncattr(k) for k in ds["chlor_a"].ncattrs()
                   if k not in ("_FillValue", "missing_value", "coordinates", "mesh_coordinates")}
    src_lon_b, src_lat_b = lcs[0], las[:, 0]
    if src_lon_b[0] > lon_t_b[0] or src_lon_b[-1] < lon_t_b[-1] or src_lat_b[0] > lat_t_b[0] \
            or src_lat_b[-1] < lat_t_b[-1]:
        raise ValueError("source subset does not cover the cut")
    flooded = flood(chl)                                         # regrid_opacity.py:26 (flood_kara)
    wx = _overlap(src_lon_b, lon_t_b) / np.diff(lon_t_b)[:, None]
    s_src, s_dst = np.sin(np.deg2rad(src_lat_b)), np.sin(np.deg2rad(lat_t_b))
    wy = _overlap(s_src, s_dst) / np.diff(s_dst)[:, None]
    if not (np.allclose(wx.sum(1), 1.0, atol=1e-12) and np.allclose(wy.sum(1), 1.0, atol=1e-12)):
        raise ValueError("conservative weights do not sum to 1 (target not covered)")
    reg = np.einsum("ja,tab,ib->tji", wy, flooded, wx)             # (time, yh, xh)
    area = (area_sg[::2, ::2] + area_sg[1::2, 1::2]) + (area_sg[1::2, ::2] + area_sg[::2, 1::2])  # :18
    ny, nx = reg.shape[1:]
    with netCDF4.Dataset(out / CHL_FILE, "w", format="NETCDF3_64BIT_OFFSET") as d:
        d.createDimension("time", None)
        for n, m in (("yh", ny), ("xh", nx), ("yq", ny + 1), ("xq", nx + 1)):
            d.createDimension(n, m)
        tv = d.createVariable("time", "f8", ("time",))
        tv.setncatts(t_attrs)
        tv[:] = times
        for n, m in (("xh", nx), ("yh", ny)):
            v = d.createVariable(n, "i4", (n,))
            v.cartesian_axis = n[0].upper()
            v[:] = np.arange(m, dtype="i4")
        c = d.createVariable("chlor_a", "f8", ("time", "yh", "xh"))
        c.setncatts(c_attrs)
        c[:] = reg
        d.createVariable("area", "f8", ("yh", "xh"))[:] = area
        d.createVariable("lon_crnr", "f8", ("yq", "xq"))[:] = xq
        d.createVariable("lat_crnr", "f8", ("yq", "xq"))[:] = yq
        d.history = ("OM4 SeaWiFS climatology flooded and regridded conservatively to the NWA12 cut as CEFI "
                     "tools/opacity/regrid_opacity.py (pyOCEAN Task 10, R25; exact rectilinear overlaps, no xesmf)")
    return dict(file=CHL_FILE, source=str(src), chl_range=[float(reg.min()), float(reg.max())],
                chl_mean_may_aug=float(reg[4:8].mean()))


# ------------------------------------------------------------------------------------------------ run directories
HERE = Path(__file__).resolve().parent
MODES = {
    # one coupling step of 600 s: PFu, PFv of the initial state (criterion 7); tides and SAL off so that the
    # force is the pressure gradient of (h, θ, S_P) only (MOM_PressureForce_FV.F90:1185-1196 adds them otherwise);
    # SAVE_INITIAL_CONDS writes MOM_IC.nc, the state MOM6 read (u, v check of check_mom6.ic_check).  The shortwave
    # does not enter the first step's PGF, so pgf0 uses the Jerlov control (no chlorophyll file needed)
    "pgf0": dict(days=0, seconds=600, dt_cpld=600, ic="full", data="full", sw="jerlov"),
    "tides": dict(days=30, seconds=0, dt_cpld=3600, ic="tides", data="tides", sw="jerlov"),   # criterion 4
    # the MOM6 reference (ruling R23): NWA12's MANIZZA_05 shortwave with the cut SeaWiFS chlorophyll climatology
    "full": dict(days=123, seconds=0, dt_cpld=3600, ic="full", data="full", sw="manizza"),
    # parity control (ruling R23): the same run with pyOCEAN's Jerlov-I shortwave
    "full_jerlov": dict(days=123, seconds=0, dt_cpld=3600, ic="full", data="full", sw="jerlov"),
    # diagnostic twin of "full" (Gulf Stream drift, figs/gs_drift d11): the same run from ic_full_pert.nc, the
    # shared initial θ plus case.perturb_temperature(amp 1e-3 °C, seed 1), the field of run_pyocean --perturb-T
    "full_pert": dict(days=123, seconds=0, dt_cpld=3600, ic="full_pert", data="full", sw="manizza"),
}
PERTURB_AMP, PERTURB_SEED = 1.0e-3, 1             # [°C], seed of ic_full_pert.nc
# "MOM6 as pyOCEAN" (Gulf Stream north-wall drift, figs/gs_drift d11, approved 2026-11-04): NWA12 options that
# pyOCEAN's parity configuration does not have, each switched to what pyOCEAN does.  Sources: line numbers of the
# NWA12 run's MOM_parameter_doc.all (runs/mom6_full, MOM6 0e9a561), of this directory's MOM_input and the get_param
# call.  Mode "full_aspy" switches off all of them; "full_aspy_<a>-<b>" only those named (bisection), with "_pert"
# appended it starts from ic_full_pert.nc (twin).  Diagnostic configuration study, not a parity item.
ASPY_SWITCHES = {
    # Fox-Kemper et al. (2008, 2011) mixed-layer eddy restratification: pyOCEAN has none (class b)
    # doc.all:1830 MIXEDLAYER_RESTRAT = True (FOX_KEMPER_ML_RESTRAT_COEF 1.0 :1841, MLE_FRONT_LENGTH 1500 :1851);
    # MOM_input:413; MOM_mixed_layer_restrat.F90:1691
    "mle": ["#override MIXEDLAYER_RESTRAT = False"],
    # Coriolis: NWA12 picks the dissipating one of two thickness-flux estimates and has BOUND_CORIOLIS off (it is
    # forced off with EN_DIS + SADOURNY75_ENERGY anyway, MOM_CoriolisAdv.F90:2165-2166); pyOCEAN is plain Sadourny
    # energy with the bound always on (core/dynamics/vector_invariant.py, :1008-1023, 1260-1275)
    # doc.all:1325 CORIOLIS_EN_DIS = True, :1341 BOUND_CORIOLIS = False; MOM_input:297, 299;
    # MOM_CoriolisAdv.F90:2074, 2163
    "coriolis": ["#override CORIOLIS_EN_DIS = False", "#override BOUND_CORIOLIS = True"],
    # tracer / thermodynamic step: pyOCEAN advances tracers and diabatic terms every Δt = 600 s.  600 is a multiple
    # of DT = 600 (doc.all:71) and divides dt_cpld = 3600 (input.nml), DT_TRACER_ADVECT ≤ DT_THERM, DIABATIC_FIRST
    # False (doc.all:18; MOM.F90:2666-2668 needs equal steps only with it), THERMO/TRADV_SPANS_COUPLING False
    # doc.all:75 DT_THERM = 1800, :85 DT_TRACER_ADVECT = 1800; MOM_input:21, 23; MOM.F90:2641, 2654
    "dt_therm": ["#override DT_THERM = 600.0", "#override DT_TRACER_ADVECT = 600.0"],
    # OBC tracer reservoir: 0 = the reservoir takes the exterior value on inflow, the interior on outflow at once,
    # pyOCEAN's upwind open-boundary tracer (no reservoir)
    # doc.all:374 OBC_TRACER_RESERVOIR_LENGTH_SCALE_OUT = 9000, :381 _IN = 9000; MOM_input:100, 101;
    # MOM_open_boundary.F90:866, 874
    "reservoir": ["#override OBC_TRACER_RESERVOIR_LENGTH_SCALE_OUT = 0.0",
                  "#override OBC_TRACER_RESERVOIR_LENGTH_SCALE_IN = 0.0"],
    # double diffusion, Large et al. (1994) in MOM_set_diffusivity (USE_KPP False, USE_CVMIX_DDIFF False doc.all:2173,
    # so this is the only double-diffusive path): pyOCEAN has none (class b)
    # doc.all:2066 DOUBLE_DIFFUSION = True; MOM_input:476; MOM_set_diffusivity.F90:2624
    "ddiff": ["#override DOUBLE_DIFFUSION = False"],
}


def mode_config(mode: str) -> dict:
    """Configuration of ``mode``: an entry of MODES, or a "MOM6 as pyOCEAN" mode ``full_aspy`` (all of
    ASPY_SWITCHES), ``full_aspy_<a>-<b>…`` (those switches only) with an optional ``_pert`` suffix (twin start)."""
    if mode in MODES:
        return MODES[mode]
    if mode == "full_aspy" or mode.startswith("full_aspy_"):
        rest, pert = mode[len("full_aspy"):], mode.endswith("_pert")
        if pert:
            rest = rest[:-len("_pert")]
        names = tuple(ASPY_SWITCHES) if rest == "" else tuple(rest.lstrip("_").split("-"))
        bad = [n for n in names if n not in ASPY_SWITCHES]
        if bad or not names or len(set(names)) != len(names) or (rest and not rest.startswith("_")):
            raise ValueError(f"mode {mode!r}: switches must be distinct names of {tuple(ASPY_SWITCHES)}")
        return dict(MODES["full_pert" if pert else "full"], aspy=tuple(n for n in ASPY_SWITCHES if n in names))
    raise ValueError(f"unknown mode {mode!r}; use {', '.join(MODES)} or full_aspy[_<sw>-<sw>…][_pert]")


SHORTWAVE = {
    # NWA12 minus COBALT: MANIZZA_05 (in MOM_input) with the chlorophyll from file (MOM_diabatic_driver / opacity)
    "manizza": ["#override CHL_FROM_FILE = True", f'CHL_FILE = "{CHL_FILE}"', 'CHL_VARNAME = "chlor_a"'],
    # Jerlov type I, Paulson & Simpson (1977) double exponential, the same as pyOCEAN's parity configuration
    # (physics/surface/shortwave.py: R = 0.58, ζ1 = 0.35 m, ζ2 = 23 m); MOM_opacity.F90:155-177 (band 1 =
    # SW_1ST_EXP_RATIO with PEN_SW_SCALE, band 2 the rest with PEN_SW_SCALE_2ND)
    "jerlov": ["#override VAR_PEN_SW = False", 'EXP_OPACITY_SCHEME = "DOUBLE_EXP"', "#override PEN_SW_NBANDS = 2",
               "PEN_SW_SCALE = 0.35", "SW_1ST_EXP_RATIO = 0.58", "PEN_SW_SCALE_2ND = 23.0"],
}
LAYOUT = (1, 5)                                     # (i, j) PEs: 5 MPI ranks, 87 × 18 columns each


def _segment_data(num: int, data: str) -> str:
    s = f"{num:03d}"
    tide = (f"Uamp=file:tu_{s}.nc(uamp),Uphase=file:tu_{s}.nc(uphase),Vamp=file:tu_{s}.nc(vamp),"
            f"Vphase=file:tu_{s}.nc(vphase),SSHamp=file:tz_{s}.nc(zamp),SSHphase=file:tz_{s}.nc(zphase)")
    if data == "tides":                              # homogeneous ocean at rest outside, tide only
        return f"U=value:0.0,V=value:0.0,SSH=value:0.0,TEMP=value:10.0,SALT=value:35.0,{tide}"
    return (f"U=file:uv_{s}.nc(u),V=file:uv_{s}.nc(v),SSH=file:zos_{s}.nc(zos),TEMP=file:thetao_{s}.nc(thetao),"
            f"SALT=file:so_{s}.nc(so),{tide}")


def mode_override(mode: str, edges: tuple[str, ...]) -> str:
    """The block of ``mode`` appended to the cut's MOM_override."""
    c = mode_config(mode)
    ic = f"ic_{c['ic']}.nc"
    L = [f"\n! --- run mode '{mode}' (mom6_inputs.mode_override)",
         f'THICKNESS_FILE = "{ic}"', f'#override TS_FILE = "{ic}"', f'#override SALT_FILE = "{ic}"',
         f'#override VELOCITY_FILE = "{ic}"']
    for e in edges:
        num = SEGMENTS[e]
        pre = "#override " if num <= 3 else ""
        L.append(f'{pre}OBC_SEGMENT_{num:03d}_DATA = "{_segment_data(num, c["data"])}"')
    L += [f"! shortwave: {c['sw']}"] + SHORTWAVE[c["sw"]]
    if mode == "pgf0":
        L += ["#override TIDES = False", "#override CALCULATE_SAL = False", "#override DT_THERM = 600.0",
              "#override DT_TRACER_ADVECT = 600.0", "#override SAVE_INITIAL_CONDS = True"]
    if mode == "tides":
        L += ["! no wind stress: a homogeneous ocean forced by the tide only (pyOCEAN 'tides' mode)",
              "#override WIND_STRESS_MULTIPLIER = 0.0"]
    for n in c.get("aspy", ()):
        L += [f"! as pyOCEAN: {n} (mom6_inputs.ASPY_SWITCHES)"] + ASPY_SWITCHES[n]
    return "\n".join(L) + "\n"


DIAG_EXTRA = {
    "pgf0": ('"pgf0", 600, "seconds", 1, "seconds", "time"',
             "\n".join(f'"ocean_model", "{v}", "{v}", "pgf0", "all", .true., "none", 1'
                       for v in ("PFu", "PFv", "h", "u", "v", "temp", "salt", "SSH"))),
    "tides": ("", ""),
    "full_jerlov": ("", "\n".join(f'"ocean_model", "{v}", "{v}", "ocean_daily", "all", .true., "none", 2'
                                  for v in ("temp", "salt", "u", "v", "h"))),
    "full": ("", "\n".join(f'"ocean_model", "{v}", "{v}", "ocean_daily", "all", .true., "none", 2'
                           for v in ("temp", "salt", "u", "v", "h"))),
    "full_pert": ("", "\n".join(f'"ocean_model", "{v}", "{v}", "ocean_daily", "all", .true., "none", 2'
                                for v in ("temp", "salt", "u", "v", "h"))),
}


def run_dir(mode: str, root: Path | None = None, inputs: Path = OUT, case: cs.Case = cs.CASE) -> Path:
    """A ready-to-run MOM6+SIS2 directory ``RUNS/runs/mom6_<mode>/`` (INPUT/ links to the inputs, templates of
    this directory filled for the mode)."""
    c = mode_config(mode)
    if c["sw"] == "manizza" and not (Path(inputs) / CHL_FILE).exists():
        raise FileNotFoundError(f"{Path(inputs) / CHL_FILE} missing: run mom6_inputs.py --only chl (source "
                                f"{CHL_SOURCE}, rulings R23, R25)")
    d = (cs.RUNS / "runs" if root is None else Path(root)) / f"mom6_{mode}"
    if d.exists():
        shutil.rmtree(d, ignore_errors=True)          # macOS ._* files vanish with their originals
    (d / "INPUT").mkdir(parents=True)
    (d / "RESTART").mkdir()
    for p in sorted(q for q in Path(inputs).glob("*.nc") if not q.name.startswith("._")):
        (d / "INPUT" / p.name).symlink_to(p)
    edges = cs.open_edges(cs.grid(case)[1])
    (d / "INPUT" / "MOM_input").write_text((HERE / "MOM_input").read_text())
    (d / "INPUT" / "MOM_override").write_text((HERE / "MOM_override").read_text() + mode_override(mode, edges))
    lay = f"IO_LAYOUT = 1,1\nLAYOUT = {LAYOUT[0]},{LAYOUT[1]}\n"
    (d / "INPUT" / "MOM_layout").write_text(lay)
    (d / "INPUT" / "SIS_layout").write_text(lay)
    (d / "INPUT" / "SIS_input").write_text((HERE / "SIS_input").read_text())
    (d / "INPUT" / "SIS_override").write_text((HERE / "SIS_override").read_text())
    (d / "input.nml").write_text((HERE / "input.nml").read_text().format(days=c["days"], seconds=c["seconds"],
                                                                         dt_cpld=c["dt_cpld"]))
    files, fields = DIAG_EXTRA["full" if "aspy" in c else mode]
    (d / "diag_table").write_text((HERE / "diag_table").read_text().format(mode=mode, extra_files=files,
                                                                           extra_fields=fields))
    shutil.copyfile(HERE / ("data_table_tides" if c["data"] == "tides" else "data_table"), d / "data_table")
    shutil.copyfile(HERE / "field_table", d / "field_table")
    (d / "run.json").write_text(json.dumps(dict(mode=mode, **c, layout=LAYOUT, inputs=str(inputs)), indent=1))
    return d


# ------------------------------------------------------------------------------------------------ main
def write(case: cs.Case = cs.CASE, only: tuple[str, ...] = ("grid", "ic", "obc", "tides", "era5", "runoff", "chl"),
          out: Path = OUT) -> Path:
    out.mkdir(parents=True, exist_ok=True)
    man_path = out / "manifest.json"
    man = json.loads(man_path.read_text()) if man_path.exists() else {}
    if "grid" in only:
        man["grid"] = write_grid(case, out)
    if "ic" in only:
        man["ic"] = write_ics(case, out)
    if "ic_pert" in only:                                  # diagnostic twin's initial file only (d11)
        man.setdefault("ic", {}).update(write_ics(case, out, modes=("full_pert",)))
    if "obc" in only:
        man["obc"] = write_glorys_segments(case, out)
    if "tides" in only:
        man["tides"] = write_tpxo_segments(case, out)
    if "era5" in only:
        man["era5"] = write_era5(case, out)
    if "runoff" in only:
        man["runoff"] = write_runoff(case, out)
    if "chl" in only:
        man["chl"] = write_chl(case, out)
    man["segments"] = {e: dict(num=SEGMENTS[e], string=SEGMENT_STRING[e]) for e in cs.open_edges(cs.grid(case)[1])}
    man_path.write_text(json.dumps(man, indent=1, default=str))
    return out


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--only", default="grid,ic,obc,tides,era5,runoff,chl")
    ap.add_argument("--run-dir", help=f"write the run directory of a mode instead ({', '.join(MODES)}, "
                                      "full_aspy[_<sw>-<sw>…][_pert])")
    a = ap.parse_args()
    if a.run_dir:
        print(run_dir(a.run_dir))
        return
    print(write(cs.CASE, tuple(x.strip() for x in a.only.split(","))))


if __name__ == "__main__":
    main()
