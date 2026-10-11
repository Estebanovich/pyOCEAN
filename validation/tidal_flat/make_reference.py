#!/usr/bin/env python
"""
Write tests/data/tidal_flat_croco_reference.npz from the CROCO TIDAL_FLAT (2DH) run:
η and ū at the stations of run_tidal_flat.py and the shoreline, over the last tidal period.

Usage:  python validation/tidal_flat/make_reference.py ~/pyOCEAN_refs/croco_runs/tidal_flat/tidal_flat_his.nc
        python validation/tidal_flat/make_reference.py ~/pyOCEAN_refs/croco_runs/tidal_flat3d/tidal_flat_his.nc 3d
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from run_tidal_flat import STATIONS_KM, T_TIDE, read_croco, shoreline  # noqa: E402

DATA = Path(__file__).resolve().parents[2] / "tests" / "data"


def main() -> None:
    c = read_croco(Path(sys.argv[1]).expanduser())
    suffix = f"_{sys.argv[2]}" if len(sys.argv) > 2 else ""
    OUT = DATA / f"tidal_flat_croco{suffix}_reference.npz"
    last = c["t"] > c["t"][-1] - T_TIDE
    i_sta = np.array([int(np.argmin(np.abs(c["x"] - xs * 1e3))) for xs in STATIONS_KM])
    ub = 0.5 * (c["ubar"][:, i_sta] + c["ubar"][:, i_sta + 1])
    sh = np.array([shoreline(c["x"], c["h"] + z, c["dcrit"]) for z in c["zeta"]])
    np.savez_compressed(OUT, t=c["t"][last], h=c["h"], i_sta=i_sta, zeta_sta=c["zeta"][last][:, i_sta],
                        ubar_sta=ub[last], shoreline=sh[last], dcrit=c["dcrit"])
    print("wrote", OUT, f"({last.sum()} records, {OUT.stat().st_size / 1e3:.0f} kB)")


if __name__ == "__main__":
    main()
