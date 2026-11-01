#!/usr/bin/env python
"""
The World Ocean Atlas 2023 profile of the TEOS-10 validation cases (Task 9 of the plan
docs/superpowers/plans/2026-10-31-teos10-3d.md; spec docs/superpowers/specs/2026-10-31-teos10-3d-design.md,
criterion 4: "perfil realista de T y S (WOA, Corriente del Golfo, ~40°N 70°W)").

Source: WOA23 annual climatology, all decades (``decav``), 1°, statistical mean ``t_an`` (in-situ temperature,
``standard_name = sea_water_temperature`` [°C]) and ``s_an`` (practical salinity, ``sea_water_practical_salinity``),
files ``~/pyOCEAN_refs/WOA23/woa23_decav_t00_01.nc`` and ``woa23_decav_s00_01.nc`` (downloaded 2026-10-31 with
Esteban's authorisation).

Cell: 40 °N 70 °W is the common corner of four 1° cells, all wet: 40.5 °N × 70.5/69.5 °W (shelf, data to 100 m and
40 m) and 39.5 °N × 70.5/69.5 °W (slope, data to 2800 m and 2700 m).  They are equidistant from the corner; the tie
is broken by the deepest data, so the profile is the cell **39–40 °N × 71–70 °W (centre 39.5 °N, 70.5 °W), 75
levels, 0–2800 m**.  The cases go to 4500 m (seamount): below 2800 m θ and S_P keep their 2800 m values
(a neutral abyss in potential density; documented in the README, it is a choice of the case, not data).

Conversion (spec goal G, MOM6 rule): S_A = (35.16504/35) S_P (no δS_A, ``physics.eos.convert.sp_to_sa``),
p = ``gsw.p_from_z(−depth, lat)`` [dbar], θ = ``gsw.pt_from_t(S_A, t, p, 0)`` (IOC et al. 2010).

Output: ``tests/data/woa23_gulfstream_profile.npz`` (depth [m] positive down, theta [°C], SP [PSU], t_insitu
[°C], lat, lon), read by ``validation/teos10/cases.py``, so the cases and the tests do not need the WOA files.

Usage:  python validation/teos10/woa_gulfstream_profile.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import gsw
import numpy as np
import xarray as xr

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from physics.eos.convert import sp_to_sa  # noqa: E402

WOA = Path.home() / "pyOCEAN_refs" / "WOA23"
OUT = ROOT / "tests" / "data" / "woa23_gulfstream_profile.npz"
CORNER = (40.0, -70.0)                    # (lat, lon) [°]


def pick_cell(t_an: xr.DataArray) -> tuple[float, float]:
    """The wet 1° cell nearest to ``CORNER``; ties broken by the deepest valid level."""
    best = None
    for lat in (CORNER[0] - 0.5, CORNER[0] + 0.5):
        for lon in (CORNER[1] - 0.5, CORNER[1] + 0.5):
            t = t_an.sel(lat=lat, lon=lon).values
            ok = np.isfinite(t)
            if not ok.any():
                continue
            dist = np.hypot(lat - CORNER[0], (lon - CORNER[1]) * np.cos(np.radians(CORNER[0])))
            key = (round(dist, 9), -float(t_an.depth.values[ok].max()))
            if best is None or key < best[0]:
                best = (key, lat, lon)
    return best[1], best[2]


def build() -> dict:
    """The profile (depth, θ, S_P, in-situ t) of the chosen cell, surface first."""
    with xr.open_dataset(WOA / "woa23_decav_t00_01.nc", decode_times=False) as dt, \
            xr.open_dataset(WOA / "woa23_decav_s00_01.nc", decode_times=False) as ds:
        t_an = dt.t_an.isel(time=0)
        lat, lon = pick_cell(t_an)
        t = t_an.sel(lat=lat, lon=lon).values.astype(float)
        sp = ds.s_an.isel(time=0).sel(lat=lat, lon=lon).values.astype(float)
        depth = dt.depth.values.astype(float)
    ok = np.isfinite(t) & np.isfinite(sp)
    depth, t, sp = depth[ok], t[ok], sp[ok]
    sa = sp_to_sa(sp)
    p = gsw.p_from_z(-depth, lat)
    theta = gsw.pt_from_t(sa, t, p, 0.0)
    return dict(depth=depth, theta=theta, SP=sp, t_insitu=t, lat=np.float64(lat), lon=np.float64(lon))


def main() -> None:
    prof = build()
    np.savez(OUT, **prof)
    print(f"cell {prof['lat']:.1f} N {prof['lon']:.1f} E, {prof['depth'].size} levels to {prof['depth'][-1]:.0f} m;"
          f" θ {prof['theta'][0]:.2f} → {prof['theta'][-1]:.2f} °C, S_P {prof['SP'][0]:.2f} → {prof['SP'][-1]:.2f}"
          f" -> {OUT}")


if __name__ == "__main__":
    main()
