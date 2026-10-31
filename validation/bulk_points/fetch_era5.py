"""
Download hourly ERA5 single-level fields for the bulk-flux forcing (N2b task 3).

Fetches, from the Copernicus Climate Data Store (``cdsapi``, new CDS 2024+),
the fields that ``forcing.sources.era5.ERA5Source`` converts to the
canonical atmosphere (``forcing/canonical.py``), plus the ERA5 mean-rate
radiation fields used to check the conversion of the accumulations:

| Group   | CDS request name                               | NetCDF name     | Units   |
|---------|------------------------------------------------|-----------------|---------|
| instant | 10m_u_component_of_wind / 10m_v_component_...  | u10, v10        | m s⁻¹   |
| instant | 2m_temperature, 2m_dewpoint_temperature        | t2m, d2m        | K       |
| instant | mean_sea_level_pressure                        | msl             | Pa      |
| accum   | surface_solar_radiation_downwards              | ssrd            | J m⁻²   |
| accum   | surface_thermal_radiation_downwards            | strd            | J m⁻²   |
| accum   | total_precipitation                            | tp              | m       |
| mean    | mean_surface_downward_short_wave_radiation_flux| avg_sdswrf (*)  | W m⁻²   |
| mean    | mean_surface_downward_long_wave_radiation_flux | avg_sdlwrf (*)  | W m⁻²   |

(*) the new CDS writes the mean rates with their GRIB2 short names
(``avg_sdswrf``/``avg_sdlwrf``; ECMWF parameters 235035/235036, formerly
``msdwswrf``/``msdwlwrf``).  In hourly ERA5 both the accumulations and the
mean rates refer to the hour *ending* at the time stamp (ECMWF, "ERA5: data
documentation", §"Accumulations" and "Mean rates/fluxes"); the
accumulation divided by 3600 s therefore equals the mean rate of the same
record.

One request per month and per group keeps every request small and gives
one unarchived NetCDF file per request:

    <outdir>/era5_<tag>_<YYYY-MM>_<group>.nc

Existing files are skipped, so the script can be re-run after an
interruption.

Usage
-----
    python validation/bulk_points/fetch_era5.py \
        --start 2017-07 --end 2017-08 --north 43 --south 38 --west -75 --east -68

The default box and period are those of N2b criterion 5 (north-east US
shelf, July–August 2017).

References
----------
Hersbach, H. et al. (2020). The ERA5 global reanalysis. Q. J. R. Meteorol.
    Soc. 146, 1999–2049.
ECMWF. ERA5: data documentation (Confluence), "Mean rates/fluxes and
    accumulations".
"""

from __future__ import annotations

import argparse
import calendar
import sys
from pathlib import Path

DEFAULT_OUTDIR = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/forcing/era5")

GROUPS: dict[str, list[str]] = {
    "instant": [
        "10m_u_component_of_wind",
        "10m_v_component_of_wind",
        "2m_temperature",
        "2m_dewpoint_temperature",
        "mean_sea_level_pressure",
    ],
    "accum": [
        "surface_solar_radiation_downwards",
        "surface_thermal_radiation_downwards",
        "total_precipitation",
    ],
    # invariant land-sea mask (N2b R7/R7b), part of the default groups as in CROCO's request
    # (croco_tools Aforc_ERA5/era5_crocotools_param.py:48); one file per box; pass it as
    # ERA5Source(paths, lsm=path) / era5_atmosphere(..., lsm=path)
    "lsm": ["land_sea_mask"],
    "mean": [
        "mean_surface_downward_short_wave_radiation_flux",
        "mean_surface_downward_long_wave_radiation_flux",
    ],
}


def months(start: str, end: str) -> list[tuple[int, int]]:
    """Inclusive list of (year, month) from 'YYYY-MM' to 'YYYY-MM'."""
    y0, m0 = (int(s) for s in start.split("-"))
    y1, m1 = (int(s) for s in end.split("-"))
    out = []
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        out.append((y, m))
        y, m = (y + 1, 1) if m == 12 else (y, m + 1)
    return out


def request(year: int, month: int, variables: list[str],
            area: tuple[float, float, float, float]) -> dict:
    """CDS request for one month of hourly ERA5 single-level data.

    ``area`` is (north, west, south, east) in degrees, the CDS order.
    """
    ndays = calendar.monthrange(year, month)[1]
    return {
        "product_type": ["reanalysis"],
        "variable": variables,
        "year": [f"{year:04d}"],
        "month": [f"{month:02d}"],
        "day": [f"{d:02d}" for d in range(1, ndays + 1)],
        "time": [f"{h:02d}:00" for h in range(24)],
        "area": list(area),
        "data_format": "netcdf",
        "download_format": "unarchived",
    }


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    p.add_argument("--start", default="2017-07", help="first month YYYY-MM")
    p.add_argument("--end", default="2017-08", help="last month YYYY-MM")
    p.add_argument("--north", type=float, default=43.0)
    p.add_argument("--south", type=float, default=38.0)
    p.add_argument("--west", type=float, default=-75.0)
    p.add_argument("--east", type=float, default=-68.0)
    p.add_argument("--tag", default="nwa_shelf", help="file-name tag of the box")
    p.add_argument("--outdir", type=Path, default=DEFAULT_OUTDIR)
    p.add_argument("--groups", nargs="+", default=["instant", "accum", "lsm"], choices=list(GROUPS))
    a = p.parse_args(argv)

    import cdsapi  # imported here so --help works without credentials

    a.outdir.mkdir(parents=True, exist_ok=True)
    area = (a.north, a.west, a.south, a.east)
    client = cdsapi.Client()
    for year, month in months(a.start, a.end):
        for group in a.groups:
            if group == "lsm":                       # invariant: one request, first month, one hour
                target = a.outdir / f"era5_{a.tag}_lsm.nc"
                if (year, month) == months(a.start, a.end)[0] and not target.exists():
                    req = request(year, month, GROUPS[group], area)
                    req.update(day=["01"], time=["00:00"])
                    tmp = target.with_suffix(".nc.part")
                    client.retrieve("reanalysis-era5-single-levels", req, str(tmp))
                    tmp.rename(target)
                    print(f"done {target.name}", flush=True)
                continue
            target = a.outdir / f"era5_{a.tag}_{year:04d}-{month:02d}_{group}.nc"
            if target.exists() and target.stat().st_size > 0:
                print(f"skip {target.name} (exists)", flush=True)
                continue
            print(f"request {target.name}", flush=True)
            tmp = target.with_suffix(".nc.part")
            client.retrieve("reanalysis-era5-single-levels",
                            request(year, month, GROUPS[group], area), str(tmp))
            tmp.rename(target)
            print(f"done {target.name} ({target.stat().st_size / 1e6:.1f} MB)", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
