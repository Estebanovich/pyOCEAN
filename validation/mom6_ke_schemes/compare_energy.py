"""
What the kinetic-energy scheme does in MOM6 itself: energy series (ocean.stats.nc) of the
MOM6-examples cases run with KE_ARAKAWA and KE_GUDONOV (run_case.sh).

For each case: total energy at the end relative to the start (En_end/En_0), the largest and the
final kinetic energy, the Godunov/Arakawa ratio of the final KE, and the velocity truncations.
For sloshing: the period and per-period decay of the KE oscillation.

Usage: python validation/mom6_ke_schemes/compare_energy.py
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np

warnings.filterwarnings("ignore")
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_ke")
CASES = ["lock_exchange", "sloshing_rho", "adjustment2d_rho", "flow_downslope_layer", "flow_downslope_rho",
         "flow_downslope_z", "flow_downslope_sigma", "Phillips_2layer"]


def series(case: str, ke: str) -> dict:
    import netCDF4 as nc
    d = nc.Dataset(RUNS / f"{case}_{ke}" / "ocean.stats.nc")
    t = np.asarray(d["Time"][:], float)
    En = np.asarray(d["En"][:], float)
    KE = np.asarray(d["KE"][:], float)
    KE = KE.sum(axis=-1) if KE.ndim > 1 else KE
    nt = int(np.asarray(d["Ntrunc"][:]).max()) if "Ntrunc" in d.variables else 0
    return dict(t=t, En=En, KE=KE, ntrunc=nt)


def sloshing_period_decay(t: np.ndarray, KE: np.ndarray) -> tuple[float, float]:
    """KE oscillates at twice the sloshing frequency: period from the mean spacing of KE maxima, decay per period."""
    i = [k for k in range(1, len(KE) - 1) if KE[k] > KE[k - 1] and KE[k] >= KE[k + 1]]
    if len(i) < 3:
        return float("nan"), float("nan")
    per = 2 * np.mean(np.diff(t[i]))
    decay = (KE[i[-1]] / KE[i[0]]) ** (1.0 / max(len(i) - 1, 1))
    return per, decay


def main() -> None:
    print(f"{'caso':22s} | {'En_fin/En_0 (A, G)':22s} | {'KE máx (A, G)':22s} | {'KE fin G/A':10s} | trunc (A, G)")
    for c in CASES:
        a, g = series(c, "KE_ARAKAWA"), series(c, "KE_GUDONOV")
        print(f"{c:22s} | {a['En'][-1] / a['En'][0]:.6f}, {g['En'][-1] / g['En'][0]:.6f} | "
              f"{a['KE'].max():.3e}, {g['KE'].max():.3e} | {g['KE'][-1] / a['KE'][-1]:10.3f} | {a['ntrunc']}, {g['ntrunc']}")
        if c == "sloshing_rho":
            for name, s in (("Arakawa", a), ("Godunov", g)):
                per, dec = sloshing_period_decay(s["t"], s["KE"])
                print(f"{'':22s}   sloshing {name}: período {per:.3f} (TIMEUNIT), KE pico/pico por período {dec:.4f}")


if __name__ == "__main__":
    main()
