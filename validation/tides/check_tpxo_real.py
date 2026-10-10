#!/usr/bin/env python
"""
Compare pyOCEAN's TPXO10-atlas-v2 reading/interpolation with the pyTMD oracle.

Result on 2026-10-10 (8 constituents × 5 points incl. coastal Manzanillo and
Acapulco): max |Δamp| = 9e-8 m, max |Δphase| = 7e-5°.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from forcing.sources.tpxo import read_tpxo  # noqa: E402
from forcing.tides import _interp_complex  # noqa: E402

ATLAS_DIR = Path("/Users/estebancruz/pyOCEAN_refs/TPXO10_atlas_v2_nc")
ref = np.load(Path(__file__).with_name("tpxo_real_oracle.npz"))
cons = [str(c) for c in ref["constituents"]]
tc = read_tpxo([ATLAS_DIR / f"h_{c}_tpxo10_atlas_30_v2.nc" for c in cons], None,
               ATLAS_DIR / "grid_tpxo10atlas_v2.nc", lon_range=(-112.0, -94.0), lat_range=(14.0, 24.0))
h = _interp_complex(tc.h, "lat", "lon", ref["lat"], ref["lon"])
amp = np.abs(h)
ph = np.degrees(np.angle(np.conj(h))) % 360.0
dph = (ph - ref["phase"] + 180.0) % 360.0 - 180.0
print(f"max |Δamp| = {np.nanmax(np.abs(amp - ref['amp'])):.2e} m,  max |Δphase| = {np.nanmax(np.abs(dph)):.2e}°")
for n, name in enumerate(ref["names"]):
    print(f"  {str(name):12s} M2 {amp[n, 0]:.4f} m / {ph[n, 0]:6.2f}°   K1 {amp[n, 2]:.4f} m / {ph[n, 2]:6.2f}°")
