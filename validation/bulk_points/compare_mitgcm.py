"""
Compare forcing.bulk.bulk_ncar with MITgcm pkg/exf EXF_BULKFORMULAE on POINTS (step 7 of N2b task 1).

Reads the output of mitgcm_exf_driver (build_reference.sh) and prints the largest difference
max|bulk_ncar − MITgcm| / max|MITgcm| per field, with bulk_ncar at all heights 10 m:
  (1) bulk_ncar as it is;
  (2) bulk_ncar with MITgcm's constants (c_p 1005, Γ 0.010 K/m, ψ-humidity factor kept),
      to separate constants from algorithm.
Usage: python -m validation.bulk_points.compare_mitgcm [~/pyOCEAN_refs/build/mitgcm_exf/out.txt]
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

from forcing import bulk
from validation.bulk_points.points import POINTS, T0_K

DEFAULT_OUT = Path.home() / "pyOCEAN_refs" / "build" / "mitgcm_exf" / "out.txt"


def _ncar(**kw) -> dict[str, np.ndarray]:
    P = POINTS
    f = bulk.bulk_ncar(P["u10"], P["v10"], P["t_air"], P["q_air"], P["sst"] - T0_K, P["p"],
                       u_ocean=P["u_ocean"], v_ocean=P["v_ocean"], z_u=10.0, z_t=10.0, **kw)
    return {"tau": np.hypot(f.tau_x, f.tau_y), "q_sens": f.q_sens, "q_lat": f.q_lat,
            "evap": f.evap}


def main(path: Path) -> dict[str, dict[str, float]]:
    m = np.loadtxt(path)
    mit = {"tau": np.hypot(m[:, 0], m[:, 1]), "q_sens": m[:, 2], "q_lat": m[:, 3],
           "evap": m[:, 4]}
    w = np.hypot(POINTS["u10"] - POINTS["u_ocean"], POINTS["v10"] - POINTS["v_ocean"])
    out = {}
    cp0 = bulk.CP_AIR
    for label, kw, cp in (("as is", {}, cp0), ("MITgcm constants", {"gamma": 0.010}, 1005.0)):
        bulk.CP_AIR = cp
        try:
            ly = _ncar(**kw)
        finally:
            bulk.CP_AIR = cp0
        res = {}
        for k in mit:
            d = np.abs(ly[k] - mit[k])
            res[k] = float(d.max() / np.abs(mit[k]).max())
            hi = w >= 0.5
            res[k + " (|ΔU| ≥ 0.5)"] = float(d[hi].max() / np.abs(mit[k]).max())
        out[label] = res
        print(label)
        for k, v in res.items():
            print(f"  {k:22s} {v:.3e}")
    return out


if __name__ == "__main__":
    main(Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT_OUT)
