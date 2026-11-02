#!/usr/bin/env python
"""
pyOCEAN inputs of the NWA12 cut, written once from the case definition (case.py) to
``RUNS/inputs/pyocean/`` (never to the repository):

  grid.nc      the C grid of the cut (core.grid.supergrid.grid_from_supergrid, MOM6 metrics) with H [m] (NaN land,
               MOM6 depth rules) and the open edges in the attributes
  ocean_topog_cut.nc, ocean_mask_cut.nc
               THE case topography (MOM6 depth rules + the edge edit of ruling R14, ``case.edit_edges``) in the MOM6
               layout (depth positive down, 0 on land; mask 1/0), for MOM6 on the same cut (Task 10)
  vgrid.nc     the 75 z* thicknesses (surface first) of vgrid_75_2m.nc
  rivers.nc    the GloFAS v4 river input for t0 → t1 (cells, daily-mean times, Q per cell, mouths; CEFI ldd rule)
  stations.nc  the series points (CO-OPS gauges, NDBC 44025, OOI Pioneer) and their nearest wet T cell

``run_pyocean.py`` builds the same objects from the raw files at start-up (the build is a few seconds); these files
are the record of what the runs used, for MOM6 (Task 10) and later models.

Usage: python validation/nwa12_cut/build_inputs.py [--t1 2017-09-01]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.nwa12_cut import case as cs  # noqa: E402


def write(case: cs.Case, t1: np.datetime64 | None = None) -> Path:
    out = case.inputs
    out.mkdir(parents=True, exist_ok=True)
    hg, H = cs.grid(case)
    edges = cs.open_edges(H)
    g = hg.copy()
    g["H"] = (("j_t", "i_t"), H, {"units": "m", "long_name": "rest depth, MOM6 depth rules (NaN = land)"})
    g.attrs.update(open_edges=",".join(edges), case=case.name, window=list(cs.window_of(case)))
    g.to_netcdf(out / "grid.nc")
    edits = cs.topography_edits(hg)
    note = "; ".join(f"(j={e['j']}, i={e['i']}) {e['lat']:.3f}N {e['lon']:.3f}E" for e in edits)
    xr.Dataset({"depth": (("ny", "nx"), np.nan_to_num(H).astype("f4"),
                          {"units": "meters", "standard_name": "topographic depth at Arakawa C h-points"})},
               attrs={"comment": "NWA12 ocean_topog.nc cut to the case window, MOM6 depth rules (MASKING_DEPTH 1, "
                                 "MINIMUM_DEPTH 4, MAXIMUM_DEPTH 6500) applied, edge runs < 4 wet cells set to land "
                                 f"(ruling R14): {note or 'none'}"}).to_netcdf(out / "ocean_topog_cut.nc")
    xr.Dataset({"mask": (("ny", "nx"), np.isfinite(H).astype("f8"))}).to_netcdf(out / "ocean_mask_cut.nc")
    xr.Dataset({"dz": ("nz", cs.vgrid_dz(case), {"units": "m", "comment": "z* nominal thickness, surface first"})}
               ).to_netcdf(out / "vgrid.nc")
    t1 = case.t1 if t1 is None else np.datetime64(t1, "ns")
    r = cs.rivers(case, hg, H, case.t0, t1)
    xr.Dataset({"j": ("cell", r.cells[:, 0]), "i": ("cell", r.cells[:, 1]),
                "Q": (("time", "cell"), r.Q, {"units": "m3 s-1", "long_name": "river discharge into the T cell"}),
                "mouth_lat": ("mouth", r.mouth_lat), "mouth_lon": ("mouth", r.mouth_lon),
                "mouth_Q": (("time", "mouth"), r.mouth_Q, {"units": "m3 s-1"}),
                "mouth_inside": ("mouth", r.mouth_inside)},
               coords={"time": r.times},
               attrs={"origin": str(r.origin), "method": r.method, "report": "\n".join(r.report)}
               ).to_netcdf(out / "rivers.nc")
    pts = cs.stations(case)
    cells = cs.station_cells(hg, H, pts)
    names = list(pts)
    xr.Dataset({"lat": ("station", [pts[n][0] for n in names]), "lon": ("station", [pts[n][1] for n in names]),
                "j": ("station", [cells.get(n, (-1, -1))[0] for n in names]),
                "i": ("station", [cells.get(n, (-1, -1))[1] for n in names])},
               coords={"station": names},
               attrs={"comment": "j = i = -1: no wet T cell within 3 cells (not recorded)"}).to_netcdf(out / "stations.nc")
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--t1", default=None, help="end of the river input (default: the case end, 2017-09-01)")
    a = ap.parse_args()
    print(write(cs.CASE, None if a.t1 is None else np.datetime64(a.t1)))


if __name__ == "__main__":
    main()
