"""
Reference-model inputs of the ERA5 column (N2b task 7): the hourly ERA5 series at the point
(``cases.point_series``, the pyOCEAN provider sampled at its own records) written in the formats of MITgcm
``pkg/exf`` and CROCO ``BULK_FLUX`` (``croco_blk.nc``), plus the initial profiles.

    python -m validation.bulk_column.forcing [--days 45]      # writes RUNS/{series.npz, mitgcm/, croco/} inputs

Unit conventions of each reference (all checked by tests/test_bulk_column.py):

MITgcm exf (``exf_readparms.F``, ``EXF_FIELDS.h``)      CROCO bulk (``get_bulk.F``, docs/design/n2b_refs/croco_bulk.md §1)
  uwind, vwind  m s⁻¹ (10 m)                            uwnd, vwnd at u/v points, wspd = |U₁₀| m s⁻¹
  atemp         K (2 m)                                 tair  °C (2 m)
  aqh           kg kg⁻¹ (2 m)                           rhum  g kg⁻¹ (≥ 2 is read as q, ``spec_hum``,
                                                              bulk_flux.F:769-774; Task 2 ruling m2)
  apressure     Pa (EXF_CALC_ATMRHO)                    —     (CROCO's constant 1e5 Pa, no READ_PATM)
  swdown        W m⁻² downward (exf applies the albedo) radsw net W m⁻² = (1 − 0.066) SW↓
  lwdown        W m⁻² downward                          radlw_in W m⁻² downward (BULK_LW)
  precip        m s⁻¹                                   prate cm day⁻¹ (stf_scale 0.01/86400, get_bulk.F:386)

MITgcm binaries: big-endian float64 (readBinaryPrec = exf_iprec = 64), one record per hour, 1 × 1; initial
profiles top first (MITgcm k = 1 at the surface).  CROCO: the 5 × 5 periodic box has xi_rho = eta_rho = 7
(LLm + 2), xi_u = eta_v = 6; bulk_time in days from the run start (CROCO SINGLE_COLUMN starts at t = 0).
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from validation.bulk_column import cases as C

MITGCM_FIELDS = {"uwind": "u10", "vwind": "v10", "atemp": "t2m", "aqh": "q2m", "apressure": "msl",
                 "swdown": "sw_down", "lwdown": "lw_down", "precip": "precip"}


def write_mitgcm(run_dir: Path, s: dict[str, np.ndarray]) -> None:
    """exf forcing files, bathymetry and initial T, S (``mitgcm/input/data*`` name them)."""
    run_dir.mkdir(parents=True, exist_ok=True)
    for fname, key in MITGCM_FIELDS.items():
        np.asarray(s[key], ">f8").reshape(-1, 1, 1).tofile(run_dir / f"{fname}.bin")
    np.full((1, 1), -C.DEPTH, ">f8").tofile(run_dir / "bathy.bin")
    T, S = C.initial_TS()                                   # bed first
    np.asarray(T[::-1], ">f8").reshape(C.N, 1, 1).tofile(run_dir / "theta_init.bin")
    np.asarray(S[::-1], ">f8").reshape(C.N, 1, 1).tofile(run_dir / "salt_init.bin")


def croco_fields(s: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """The CROCO bulk variables as time series (nt,) in CROCO's units (module docstring)."""
    return {"tair": s["t2m"] - 273.15, "rhum": 1e3 * s["q2m"], "prate": s["precip"] * 100.0 * 86400.0,
            "wspd": np.hypot(s["u10"], s["v10"]), "radlw_in": s["lw_down"],
            "radsw": (1.0 - C.ALBEDO) * s["sw_down"], "uwnd": s["u10"], "vwnd": s["v10"]}


def write_croco(path: Path, s: dict[str, np.ndarray]) -> None:
    """croco_blk.nc for the 5 × 5 periodic SINGLE_COLUMN box (uniform fields)."""
    import netCDF4
    path.parent.mkdir(parents=True, exist_ok=True)
    # CROCO reads the record after the last step's (get_bulk.F stops with "requested time record exceeds the last
    # available" at t = t_end): one copy of the last record at t_end + 1 h, which gets no weight before t_end
    f = {k: np.append(v, v[-1]) for k, v in croco_fields(s).items()}
    time = np.append(s["time"], s["time"][-1] + 3600.0)
    nt = time.size
    nr = C.NXY + 2
    dims = {"rho": ("bulk_time", "eta_rho", "xi_rho"), "u": ("bulk_time", "eta_u", "xi_u"),
            "v": ("bulk_time", "eta_v", "xi_v")}
    shape = {"rho": (nr, nr), "u": (nr, nr - 1), "v": (nr - 1, nr)}
    where = {k: "rho" for k in f}
    where.update(uwnd="u", vwnd="v")
    units = dict(tair="Celsius", rhum="g/kg (specific humidity)", prate="cm/day", wspd="m/s", radlw_in="W/m2",
                 radsw="W/m2", uwnd="m/s", vwnd="m/s")
    with netCDF4.Dataset(path, "w") as nc:
        nc.createDimension("bulk_time", nt)
        nc.createDimension("xi_rho", nr)
        nc.createDimension("eta_rho", nr)
        nc.createDimension("xi_u", nr - 1)
        nc.createDimension("eta_u", nr)
        nc.createDimension("xi_v", nr)
        nc.createDimension("eta_v", nr - 1)
        tv = nc.createVariable("bulk_time", "f8", ("bulk_time",))
        tv[:] = time / 86400.0
        tv.units = "days"
        for k, v in f.items():
            var = nc.createVariable(k, "f8", dims[where[k]])
            var[:] = np.asarray(v)[:, None, None] * np.ones((nt,) + shape[where[k]])
            var.units = units[k]
        nc.title = "pyOCEAN N2b task 7: ERA5 at 40.24N 73.16W (validation/bulk_column)"


def croco_init_header(path: Path) -> None:
    """
    Fortran include for the patched ana_initial.F (``croco/build_croco.sh``): executable assignments of the WOA
    profile at the layer centres, k = 1 at the bed as CROCO, both time levels (cpp expands JR_RANGE/IR_RANGE).
    """
    T, S = C.initial_TS()
    lines = ["! WOA23 July profile, k = 1 at the bed (validation/bulk_column/cases.py; generated by forcing.py)",
             "        do j=JR_RANGE", "          do i=IR_RANGE"]
    for k in range(C.N):
        for name, a in (("itemp", T), ("isalt", S)):
            for n in (1, 2):
                lines.append(f"            t(i,j,{k + 1},{n},{name})={a[k]:.10f}d0")
    lines += ["          enddo", "        enddo", ""]
    path.write_text("\n".join(lines))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--days", type=int, default=C.DAYS)
    a = ap.parse_args()
    s = C.point_series(a.days)
    C.RUNS.mkdir(parents=True, exist_ok=True)
    np.savez(C.RUNS / f"series_{a.days}d.npz", **s)
    write_mitgcm(C.RUNS / f"mitgcm_{a.days}d", s)
    write_croco(C.RUNS / f"croco_{a.days}d" / "croco_blk.nc", s)
    print(f"inputs for {a.days} days written under {C.RUNS}")


if __name__ == "__main__":
    main()
