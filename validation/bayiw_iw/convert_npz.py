#!/usr/bin/env python
"""
Convert the .npz series of the first BayIW runs (run_bayiw.py before the CF
NetCDF output) to pyocean_iw_<tag>_{stations,fields}.nc (iw_output.py).  The
station, section and box coordinates are rebuilt from the model setup; the run
configuration comes from the npz.  The .npz is kept (delete it by hand once the
NetCDF files are checked).

Usage:  python convert_npz.py [bay nobay] [--out DIR]
"""

from __future__ import annotations

import argparse
from pathlib import Path
from types import SimpleNamespace

import numpy as np

import iw_output
from run_bayiw import OUT, setup, static_metadata


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("cases", nargs="*", default=["bay", "nobay"])
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--tag", default="")
    a = ap.parse_args()
    for case in a.cases:
        npz = a.out / f"pyocean_iw_{case}_realistic{a.tag}.npz"
        rec = iw_output.read_records(npz)
        args = SimpleNamespace(**rec["meta"])
        m, _, _ = setup(case, args)
        static = static_metadata(m, args)
        static["attrs"] = rec["meta"]
        stem = npz.with_suffix("")
        iw_output.write(stem, rec, static)
        back = iw_output.read_records(stem)
        for k in ("t", "eta_st", "T_st", "u_st", "v_st", "w_st"):
            assert np.array_equal(back[k], rec[k], equal_nan=True), k
        err = max(float(np.nanmax(np.abs(back[k] - rec[k]) / (np.nanmax(np.abs(rec[k])) + 1e-30)))
                  for k in ("T_axis", "v_axis", "w_axis", "eta_axis", "w_z", "T_z"))
        sizes = {p.name: p.stat().st_size / 1e6 for p in a.out.glob(f"{stem.name}*")}
        print(f"{case}: {len(rec['t'])} records; stations exact, fields max rel. error {err:.1e}; "
              + ", ".join(f"{n} {s:.0f} MB" for n, s in sizes.items()))


if __name__ == "__main__":
    main()
