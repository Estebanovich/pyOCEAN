#!/usr/bin/env python
"""
Generate the JSL internal-tide drag field tidal.rh for PAC_MEX with pyOCEAN
(physics.tides.drag) and compare it with the HYCOM-tools product
(~/HYCOM/drag_jsl/run_drag_jsl.sh → ~/pyOCEAN_refs/tidal.rh.[ab]).

Inputs
------
~/pyOCEAN_refs/hycom_GLBp0.08/relax_DRAG/relax/DRAG/JSL.[ab]   global JSL [1/s], 4500×2251
~/pyOCEAN_refs/hycom_GLBp0.08/topo/topo/regional.grid.[ab]     its uniform 0.08° grid
~/HYCOM/regional.grid.[ab], regional.depth_final.[ab]          PAC_MEX grid and depth

Output
------
~/pyOCEAN_refs/tidal.rh_pyocean.npy   rh [m/s] (jdm, idm), 0 on land

Usage:  python validation/tides/make_tidal_rh.py
"""

from __future__ import annotations

import os

for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(_v, "1")

import sys  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from forcing.sources.hycom_ab import read_field  # noqa: E402
from physics.tides.drag import jsl_to_rh, regrid_bilinear  # noqa: E402

REFS = Path.home() / "pyOCEAN_refs"
GLB = REFS / "hycom_GLBp0.08"
HYCOM = Path.home() / "HYCOM"
IDM, JDM = 1600, 1125


def main() -> None:
    jsl_g = read_field(GLB / "relax_DRAG/relax/DRAG/JSL.b", "JSL", idm=4500, jdm=2251)
    glon = read_field(GLB / "topo/topo/regional.grid.b", "plon")[0]
    glat = read_field(GLB / "topo/topo/regional.grid.b", "plat")[:, 0]
    print(f"global JSL: {np.isnan(jsl_g).mean():.2%} void, max {np.nanmax(jsl_g):.3e} 1/s; "
          f"lon step {np.ptp(np.diff(glon)):.1e} lat step {np.ptp(np.diff(glat)):.1e} (uniform if ~0)")

    plon = read_field(HYCOM / "regional.grid.b", "plon")
    plat = read_field(HYCOM / "regional.grid.b", "plat")
    depth = read_field(HYCOM / "regional.depth_final.b", "depth", idm=IDM, jdm=JDM)

    jsl = regrid_bilinear(jsl_g, glat, glon, plat, plon)
    rh = jsl_to_rh(jsl, depth)
    np.save(REFS / "tidal.rh_pyocean.npy", rh.astype(np.float32))

    ref = np.nan_to_num(read_field(REFS / "tidal.rh.b", "tidal.rh", idm=IDM, jdm=JDM), nan=0.0)
    wet = np.isfinite(depth)
    d = rh - ref
    print(f"pyOCEAN rh: min {rh.min():.4g} max {rh.max():.6g} m/s   HYCOM-tools rh: max {ref.max():.6g} m/s")
    print(f"ocean points {wet.sum()}: max |Δ| {np.abs(d[wet]).max():.3e} m/s, "
          f"rms {np.sqrt(np.mean(d[wet] ** 2)):.3e}, max |Δ|/max rh {np.abs(d[wet]).max() / ref.max():.2e}")
    nz = wet & (ref > 0)
    rel = np.abs(d[nz]) / ref[nz]
    print(f"relative error where rh > 0: median {np.median(rel):.2e}, 99th pct {np.percentile(rel, 99):.2e}, "
          f"max {rel.max():.2e}")
    print(f"zero/non-zero agreement: {np.mean((rh[wet] > 0) == (ref[wet] > 0)):.6f}; "
          f"land rh all zero: {np.all(ref[~wet] == 0)} (HYCOM), {np.all(rh[~wet] == 0)} (pyOCEAN)")
    j, i = np.unravel_index(np.argmax(np.abs(np.where(wet, d, 0))), d.shape)
    print(f"largest difference at lon {plon[j, i]:.2f} lat {plat[j, i]:.2f} depth {depth[j, i]:.0f} m: "
          f"pyOCEAN {rh[j, i]:.5g}  HYCOM {ref[j, i]:.5g}")


if __name__ == "__main__":
    main()
