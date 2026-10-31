"""
Air–sea test points for the bulk-formula comparison of N2b criterion 1.

``POINTS`` is a deterministic set of 250 near-surface states that covers the
regimes where the NCAR bulk algorithm (Large & Yeager 2004) branches:

* wind speed 0.1–30 m s⁻¹ (the 0.5 m s⁻¹ floor of ``ncar_ocean_fluxes`` is
  active below it, surface_flux.F90:836);
* SST −1.8 °C (below the 273.16 K where the FMS saturation table blends over
  ice, sat_vapor_pres_k.inc:249-283) to 32 °C;
* air–sea temperature difference −10 to +5 K (strongly unstable to stable,
  including the |ζ| = 10 clip, surface_flux.F90:865);
* relative humidity 50–95 %, wind directions all round, and a surface
  current of 0.3–1.5 m s⁻¹ on every second point (relative wind,
  surface_flux.F90:349-350).

All quantities are SI with temperatures in K, as ``ncar_ocean_fluxes``:

    u10, v10    wind at z_u [m s⁻¹]
    t_air       air temperature at z_t [K]
    q_air       specific humidity at z_q [kg kg⁻¹]
    sst         surface temperature [K]
    p           sea-level pressure [Pa] (= p_atm = p_surf, OM4 data_table)
    u_ocean, v_ocean  surface current [m s⁻¹]

``write_points`` writes them as text for ``fms_ncar_driver.F90``.
``LY04_VS_FMS_MAX_REL`` holds the measured largest relative difference between
``forcing.bulk.bulk_ncar`` (LY04 / MITgcm form) and ``bulk_ncar_fms`` on these
points (``tests/test_bulk_fms.py::test_ly04_differences_documented``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

T0_K: float = 273.15                 # K, 0 °C

_WIND = np.array([0.1, 0.5, 1.0, 2.0, 4.0, 7.0, 10.0, 15.0, 20.0, 30.0])   # m s⁻¹
_SST_C = np.array([-1.8, 5.0, 15.0, 25.0, 32.0])                           # °C
_DT = np.array([-10.0, -3.0, -0.5, 1.0, 5.0])                              # K, T_air − SST


def _e_sat_bolton(t_c: np.ndarray) -> np.ndarray:
    """Saturation vapour pressure over water [Pa], Bolton (1980) eq. 10 (inputs only)."""
    return 611.2 * np.exp(17.67 * t_c / (t_c + 243.5))


def _make_points() -> dict[str, np.ndarray]:
    wind, sst_c, dt = (a.ravel() for a in np.meshgrid(_WIND, _SST_C, _DT, indexing="ij"))
    n = wind.size
    idx = np.arange(n)
    direction = np.deg2rad((37.0 * idx) % 360.0)
    rh = 0.50 + 0.45 * ((idx * 7) % 10) / 9.0                 # 0.50 … 0.95
    t_air = sst_c + dt + T0_K
    p = 101325.0 + 1500.0 * np.sin(0.3 * idx)                  # 99 825 … 102 825 Pa
    e = rh * _e_sat_bolton(t_air - T0_K)
    q_air = 0.622 * e / (p - 0.378 * e)
    current = np.where(idx % 2 == 1, 0.3 + 1.2 * ((idx * 3) % 5) / 4.0, 0.0)
    cdir = np.deg2rad((113.0 * idx) % 360.0)
    return {
        "u10": wind * np.cos(direction),
        "v10": wind * np.sin(direction),
        "t_air": t_air,
        "q_air": q_air,
        "sst": sst_c + T0_K,
        "p": p,
        "u_ocean": current * np.cos(cdir),
        "v_ocean": current * np.sin(cdir),
    }


POINTS: dict[str, np.ndarray] = _make_points()

#: Column order of the driver input file.
INPUT_FIELDS: tuple[str, ...] = ("u10", "v10", "t_air", "q_air", "sst", "p", "u_ocean", "v_ocean")

#: Column order of the driver output file: FMS sign conventions (fluxes positive
#: upward / on the atmosphere), surface_flux.F90:441-476.
OUTPUT_FIELDS: tuple[str, ...] = (
    "flux_u", "flux_v", "flux_t", "flux_q", "cd_m", "cd_t", "cd_q", "u_star", "w_atm",
)

#: Heights (z_u, z_t, z_q) [m] of the two reference tables: all at 10 m
#: (OM4 data_table, ``ncar_ocean_fluxes``) and T, q at 2 m as ERA5
#: (``ncar_ocean_fluxes_multilevel``, surface_flux.F90:905-1052).
HEIGHTS: dict[str, tuple[float, float, float]] = {"z10": (10.0, 10.0, 10.0), "z2": (10.0, 2.0, 2.0)}

#: Largest difference max|bulk_ncar − bulk_ncar_fms| / max|bulk_ncar_fms| per field on
#: POINTS (all heights 10 m, ρ_a and θ left to each scheme; τ = |τ|), measured 2026-10-31
#: (tau 7.67e-4, q_sens 3.68e-2, q_lat and evap 5.88e-2).  The nine algorithm differences,
#: each alone (README "LY04 vs FMS"; together they reproduce bulk_ncar to 6e-16):
#:   1. flux speed: bulk_ncar max(|ΔU|, 0.5), FMS |ΔU| (floor only in the coefficients,
#:      surface_flux.F90:846 vs :423-425)            q_sens 3.8e-2, q_lat 4.8e-2, τ 2.5e-4
#:   2. q_sat: LY04 eq 5 vs FMS Goff–Gratch table (:308, :320), +0.04 … +2.2 % in q_s
#:                                                    q_lat 5.8e-2, τ 5.4e-4, q_sens 4.0e-4
#:   3. θ_a: bulk_ncar T + 0.0098 z_t, FMS T (p_atm = p_surf, :341, :344)  q_sens 1.0e-2
#:   4. c_p: 1000.5 (LY04) vs 1004.64 (FMS RDGAS/KAPPA)                    q_sens 4.1e-3
#:   5. U₁₀N floored at 0.5 in bulk_ncar, not in FMS (:1020)               q_lat 3.6e-4
#:   6. C_dN slope 1/13.09 (LY04) vs 0.0764 (FMS, CESM)                    τ 6.8e-5
#:   7. g 9.81 vs 9.80                                                      ≤ 4.3e-5
#:   8. ρ_a from θ and 0.608 vs from T and d608 = 0.6078 (:343, :428)        ≤ 4.5e-6
#:   9. stable branch at ζ = 0 (bulk_ncar ζ > 0, FMS ζ ≥ 0)                  0 here
#: All are class (c), choices between published forms; none is a bug of either code.
LY04_VS_FMS_MAX_REL: dict[str, float] = {
    "tau": 7.7e-4,      # 2 (q_sat through ζ) and 1
    "q_sens": 3.7e-2,   # 1 at |ΔU| = 0.1 m/s (factor 5), then 3 and 4
    "q_lat": 5.9e-2,    # 2, amplified where q_a − q_s is small (warm humid air)
    "evap": 5.9e-2,     # as q_lat (same L_v = 2.5e6)
}


def write_points(path: str | Path) -> None:
    """Write POINTS as text, one point per line, columns INPUT_FIELDS, 17 significant digits."""
    data = np.column_stack([POINTS[k] for k in INPUT_FIELDS])
    with open(path, "w") as f:
        f.write(f"{data.shape[0]}\n")
        np.savetxt(f, data, fmt="%.17e")


if __name__ == "__main__":
    import sys

    write_points(sys.argv[1])
