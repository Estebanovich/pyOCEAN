#!/usr/bin/env python
"""
Reference values of the long-period constituents MF and MM (regional chain Task 5).

pyTMD (the oracle of ``make_reference.py`` for the eight HYCOM constituents) is not installed in this
environment, so the MF and MM reference is built here from the published tables, independently of
``forcing/tides.py``:

* angular frequencies from the rates of the mean longitudes of Cartwright (1982), as OTPS uses them
  (s: 13.17639648 °/day, p: 0.11140353 °/day): ω_MF = 2 ṡ, ω_MM = ṡ − ṗ;
* astronomical arguments at 1992-01-01 00:00 UT, the OTPS ``constit.h`` epoch: V₀(MF) = 2s, V₀(MM) = s − p
  (Doodson numbers 0 2 0 0 0 0 and 0 1 0 −1 0 0; Kowalik & Luick 2019 Table I.4; MOM6 eq_phase,
  MOM_tidal_forcing.F90:120-150), with the mean longitudes of Cartwright;
* the OTPS constit.h tabulated values (ω and phase_mkB), which must agree with the two items above;
* nodal factors of the OTPS ``nodal`` routine, long-period branch (Schureman 1958):
  MF f = 1.043 + 0.414 cos N, u = −23.7° sin N + 2.7° sin 2N − 0.4° sin 3N; MM f = 1 − 0.130 cos N, u = 0
  [verificar against pyTMD ``arguments(..., corrections='OTIS')`` when it is available];
* MOM6's first-order nodal factors (nodal_fu, MOM_tidal_forcing.F90:187-231), stored for a looser cross-check.

Run:  python validation/tides/make_mm_mf_reference.py   →  tests/data/tide_mm_mf_reference.npz
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

MJD_1992 = 48622.0
DEG_DAY = np.pi / 180.0 / 86400.0                     # °/day → rad/s


def _longitudes(mjd: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean longitudes s, p, N [deg] (Cartwright 1982, as OTPS): T = MJD − 51544.4993."""
    T = np.asarray(mjd, float) - 51544.4993
    return (np.mod(218.3164 + 13.17639648 * T, 360.0), np.mod(83.3535 + 0.11140353 * T, 360.0),
            np.mod(125.0445 - 0.05295377 * T, 360.0))


def main() -> None:
    rng = np.random.default_rng(20261101)
    mjd = np.sort(np.concatenate([rng.uniform(47892.0, 66154.0, 200), [48622.0, 51544.5, 61345.0]]))
    s0, p0, _ = _longitudes(np.array([MJD_1992]))
    phase = {"mf": float(np.mod(np.deg2rad(2 * s0[0]), 2 * np.pi)),
             "mm": float(np.mod(np.deg2rad(s0[0] - p0[0]), 2 * np.pi))}
    omega = {"mf": 2 * 13.17639648 * DEG_DAY, "mm": (13.17639648 - 0.11140353) * DEG_DAY}
    _, _, n = _longitudes(mjd)
    N = np.deg2rad(n)
    f = np.column_stack([1.043 + 0.414 * np.cos(N), 1.0 - 0.130 * np.cos(N)])
    u = np.column_stack([np.deg2rad(-23.7 * np.sin(N) + 2.7 * np.sin(2 * N) - 0.4 * np.sin(3 * N)), 0.0 * N])
    f_mom6 = np.column_stack([1.043 + 0.414 * np.cos(N), 1.0 - 0.130 * np.cos(N)])
    u_mom6 = np.column_stack([np.deg2rad(-23.7 * np.sin(N)), 0.0 * N])
    out = Path(__file__).resolve().parents[2] / "tests" / "data" / "tide_mm_mf_reference.npz"
    np.savez(out, constituents=np.array(["mf", "mm"]), mjd=mjd, f=f, u=u, f_mom6=f_mom6, u_mom6=u_mom6,
             omega_doodson=np.array([omega["mf"], omega["mm"]]), phase_doodson=np.array([phase["mf"], phase["mm"]]),
             omega_otps=np.array([0.053234e-04, 0.026392e-04]), phase_otps=np.array([1.756042456, 1.964021610]),
             source="OTPS constit.h + nodal (Schureman 1958); pyTMD not available")
    print("wrote", out)


if __name__ == "__main__":
    main()
