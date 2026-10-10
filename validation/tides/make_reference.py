#!/usr/bin/env python
"""
Reference tidal arguments and predictions from pyTMD (OTPS/TPXO conventions).

pyTMD (Sutterley et al., MIT licence) reproduces the OSU Tidal Prediction
Software (OTPS, Egbert & Erofeeva 2002) used with TPXO.  It is used here only
as an independent test oracle for forcing/tides.py — pyOCEAN does not depend
on it.  Run with a Python that has pyTMD installed (e.g. an isolated venv):

    python validation/tides/make_reference.py      →  tests/data/tide_otps_reference.npz
"""

from pathlib import Path

import numpy as np
import pyTMD
import pyTMD.arguments
import pyTMD.predict

CONSTITUENTS = ["m2", "s2", "k1", "o1", "n2", "p1", "k2", "q1"]     # HYCOM order
MJD_1992 = 48622.0

rng = np.random.default_rng(20261010)
# 200 times between 1990 and 2040, plus a few fixed dates
mjd = np.sort(np.concatenate([rng.uniform(47892.0, 66154.0, 200),
                              [48622.0, 51544.5, 61308.5, 61314.25]]))   # 1992-01-01, 2000-01-01T12, 2026-10-10, 2026-10-16T06
pu, pf, _ = pyTMD.arguments.arguments(mjd, CONSTITUENTS, corrections="OTIS")

# predictions for 5 random sets of complex harmonic constants, hourly over 30 days
t_days = 12700.0 + np.arange(0, 30 * 24) / 24.0        # days since 1992-01-01 (≈ 2026-10)
hc = rng.normal(0, 0.3, (5, len(CONSTITUENTS))) + 1j * rng.normal(0, 0.3, (5, len(CONSTITUENTS)))
eta = np.array([pyTMD.predict.time_series(t_days, np.ma.array(h[None, :], mask=np.zeros((1, len(CONSTITUENTS)), bool)), CONSTITUENTS, corrections="OTIS").data
                for h in hc])

out = Path(__file__).resolve().parents[2] / "tests" / "data" / "tide_otps_reference.npz"
np.savez(out, constituents=np.array(CONSTITUENTS), mjd=mjd, u=pu, f=pf,
         t_days_1992=t_days, hc=hc, eta=eta, pytmd_version=pyTMD.__version__)
print("wrote", out, "pyTMD", pyTMD.__version__)
