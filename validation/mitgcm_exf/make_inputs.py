"""
Generate the binary inputs of the MITgcm exf bulk-formula validation experiment.

Every horizontal grid point (40 x 25 = 1000) is an independent, time-constant
air–sea case.  The wind speed sweeps 0.2–35 m/s along x (log-spaced, so calm,
moderate and hurricane-force winds are all present in every row); the air–sea
temperature difference ΔT = T_air(2 m) − SST (−8 … +6 K), relative humidity
(40–100 %), SST (5–30 °C) and wind direction are drawn from a seeded random
generator.  A few rows are forced to special values (ΔT = 0, RH = 100 %,
RH = 40 %) so that near-neutral and saturated cases are well represented.

Files are written as big-endian float64 (MITgcm readBinaryPrec = 64,
exf_iprec = 64), Fortran order (x fastest), into ``input/``.  A NetCDF-free
copy of all case parameters is saved to ``input/cases.npz`` for compare.py.

Run:  /opt/anaconda3/bin/python make_inputs.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

NX: int = 40
NY: int = 25
NR: int = 2
T0_K: float = 273.15         # K, MITgcm cen2kel
RHO_AIR_MIT: float = 1.2     # kg m⁻³, MITgcm atmrho default

HERE = Path(__file__).resolve().parent
INPUT = HERE / "input"


def qsat_ly04(t_k: np.ndarray, rho_air: float) -> np.ndarray:
    """
    Saturation specific humidity over fresh water (LY04 eq 5) [kg kg⁻¹].

        q_sat = 640380 / ρ_a · exp(−5107.4 / T)

    Only used here to turn RH into a specific humidity input; any consistent
    choice is fine because both models receive the same q.
    """
    return 640380.0 / rho_air * np.exp(-5107.4 / t_k)


def make_cases(seed: int = 20261010) -> dict[str, np.ndarray]:
    """Build the (NY, NX) arrays of atmospheric inputs and SST."""
    rng = np.random.default_rng(seed)
    shape = (NY, NX)

    speed = np.broadcast_to(np.geomspace(0.2, 35.0, NX), shape).copy()
    direction = rng.uniform(0.0, 2.0 * np.pi, shape)
    d_t = rng.uniform(-8.0, 6.0, shape)
    rh = rng.uniform(0.40, 1.00, shape)
    sst = rng.uniform(5.0, 30.0, shape)

    # special rows: neutral-ish, saturated, dry
    d_t[0, :] = 0.0
    rh[1, :] = 1.0
    rh[2, :] = 0.40
    d_t[3, :] = -8.0          # strongly unstable
    d_t[4, :] = 6.0           # strongly stable

    t_air_k = sst + d_t + T0_K
    q_air = rh * qsat_ly04(t_air_k, RHO_AIR_MIT)

    return {
        "uwind": speed * np.cos(direction),
        "vwind": speed * np.sin(direction),
        "speed": speed,
        "atemp": t_air_k,
        "aqh": q_air,
        "rh": rh,
        "dT": d_t,
        "sst": sst,
        "swdown": rng.uniform(0.0, 1000.0, shape),
        "lwdown": rng.uniform(300.0, 420.0, shape),
        "precip": rng.uniform(0.0, 1.0e-7, shape),
    }


def write_be(path: Path, arr: np.ndarray) -> None:
    """Write an array as big-endian float64 in C order (= Fortran x-fastest for (.., ny, nx))."""
    np.ascontiguousarray(arr, dtype=">f8").tofile(path)


def main() -> None:
    INPUT.mkdir(exist_ok=True)
    c = make_cases()
    for name in ("uwind", "vwind", "atemp", "aqh", "swdown", "lwdown", "precip"):
        write_be(INPUT / f"{name}.bin", c[name])
    # initial potential temperature: both levels equal to SST so gcmSST = θ(k=1) = SST
    write_be(INPUT / "theta_init.bin", np.broadcast_to(c["sst"], (NR, NY, NX)))
    np.savez(INPUT / "cases.npz", **c)
    print(f"wrote {len(c)} case fields ({NY}x{NX}) to {INPUT}")


if __name__ == "__main__":
    main()
