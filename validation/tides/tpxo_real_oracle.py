#!/usr/bin/env python
"""
pyTMD extraction of TPXO10-atlas-v2 constants at Mexican Pacific points (oracle).

Run with a Python that has pyTMD (isolated venv); writes tpxo_real_oracle.npz
next to this script.  Compare with:  python validation/tides/check_tpxo_real.py
"""
from pathlib import Path

import numpy as np
import pyTMD.io.ATLAS as ATLAS

ATLAS_DIR = Path("/Users/estebancruz/pyOCEAN_refs/TPXO10_atlas_v2_nc")
CONSTITUENTS = ["m2", "s2", "k1", "o1", "n2", "p1", "k2", "q1"]
POINTS = {"15N110W": (15.0, -110.0), "Mazatlan": (23.20, -106.43), "Manzanillo": (19.05, -104.33),
          "Acapulco": (16.84, -99.91), "Salina Cruz": (16.16, -95.20)}

lat = np.array([p[0] for p in POINTS.values()])
lon = np.array([p[1] for p in POINTS.values()])
amp, ph, _, cons = ATLAS.extract_constants(
    lon % 360.0, lat, ATLAS_DIR / "grid_tpxo10atlas_v2.nc",
    [ATLAS_DIR / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in CONSTITUENTS],
    type="z", method="bilinear", crop=True, scale=1.0 / 1000.0, compressed=False)
np.savez(Path(__file__).with_name("tpxo_real_oracle.npz"), names=list(POINTS), lat=lat, lon=lon,
         constituents=list(cons), amp=np.ma.filled(amp, np.nan), phase=np.ma.filled(ph, np.nan))
print("saved", Path(__file__).with_name("tpxo_real_oracle.npz"))
