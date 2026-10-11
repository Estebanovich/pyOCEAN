#!/usr/bin/env python3
"""
Extract the sea-surface height from HYCOM archive files (archs/archv) over a
lon/lat box and save it in one compressed .npz — to be run on the cluster
(ometeotl) next to the archives, so only ~100 MB have to be copied.

Standalone: needs only NumPy (no pyOCEAN).  srfhgt in HYCOM archives is
geopotential g·η [m² s⁻²]; η = srfhgt / 9.806 (archv2data convention).

Usage (on ometeotl):
    python3 extract_archs_ssh.py OUT.npz GRID.b ARCH1.b|DIR [ARCH2.b|DIR ...]
    (a directory is searched recursively for archive .b files: *arch[msv]*.b)
    e.g.
    python3 extract_archs_ssh.py pacmex_ssh_0260271012.npz \\
        /LUSTRE/.../HYCOM-PACIFICO/topo/regional.grid.b \\
        expt_00.0/data/tars_0260271012/000_archs.2026_*.b
Options (environment): LON0 LON1 LAT0 LAT1 (default −118 −92 12 32).
"""

from __future__ import annotations

import glob
import os
import re
import sys

import numpy as np

_ARCH = re.compile(r"^\s*(\S+?)\s*=\s*(\d+)\s+([-+0-9.eE]+)\s+(\d+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s+([-+0-9.eE]+)\s*$")
_HEAD = re.compile(r"^\s*([-+0-9.eE]+)\s+'(\w+)\s*'\s*=")
_GRID = re.compile(r"^\s*(\w+)\s*:\s*min,max\s*=\s*([-+0-9.eE]+)\s+([-+0-9.eE]+)")
G_HYCOM = 9.806
VOID = 2.0 ** 100


def _record(path_a: str, irec: int, idm: int, jdm: int) -> np.ndarray:
    n = idm * jdm
    npad = (4096 - n % 4096) % 4096
    with open(path_a, "rb") as f:
        f.seek(irec * (n + npad) * 4)
        x = np.fromfile(f, ">f4", count=n).astype(np.float64).reshape(jdm, idm)
    return np.where(x > 0.5 * VOID, np.nan, x)


def read_grid(path_b: str) -> tuple[np.ndarray, np.ndarray]:
    idm = jdm = None
    names = []
    for line in open(path_b):
        if m := _HEAD.match(line):
            if m[2] in ("idm", "jdm"):
                idm, jdm = (int(float(m[1])), jdm) if m[2] == "idm" else (idm, int(float(m[1])))
        elif m := _GRID.match(line):
            names.append(m[1])
    a = path_b[:-2] + ".a"
    return _record(a, names.index("plon"), idm, jdm), _record(a, names.index("plat"), idm, jdm)


def read_archive_ssh(path_b: str) -> tuple[float, np.ndarray]:
    """(model day, η [m]) from an archive .b/.a pair."""
    idm = jdm = None
    fields = []
    for line in open(path_b):
        if m := _HEAD.match(line):
            if m[2] == "idm":
                idm = int(float(m[1]))
            elif m[2] == "jdm":
                jdm = int(float(m[1]))
        elif m := _ARCH.match(line):
            fields.append((m[1], float(m[3]), float(m[6]), float(m[7])))
    k = [f[0] for f in fields].index("srfhgt")
    name, day, vmin, vmax = fields[k]
    x = _record(path_b[:-2] + ".a", k, idm, jdm)
    if abs(np.nanmin(x) - vmin) > 1e-4 * max(1.0, abs(vmin)) or abs(np.nanmax(x) - vmax) > 1e-4 * max(1.0, abs(vmax)):
        raise ValueError(f"{path_b}: srfhgt min/max differ from the .b line")
    return day, x / G_HYCOM


def find_archives(args: list[str]) -> list[str]:
    """Archive .b files from file names, glob patterns or directories (searched recursively)."""
    found = []
    for a in args:
        if os.path.isdir(a):
            found += glob.glob(os.path.join(a, "**", "*arch[msv]*.b"), recursive=True)
        elif any(ch in a for ch in "*?["):
            found += glob.glob(a)
        else:
            found.append(a)
    found = sorted({f for f in found if "dummy" not in os.path.basename(f)}, key=os.path.basename)
    if not found:
        listing = {a: sorted(os.listdir(a))[:10] for a in args if os.path.isdir(a)}
        raise SystemExit(f"no HYCOM archive .b files found in {args}; directory contents: {listing}")
    return found


def main(argv: list[str]) -> None:
    out, grid_b, archives = argv[0], argv[1], find_archives(argv[2:])
    print(f"{len(archives)} archives: {os.path.basename(archives[0])} … {os.path.basename(archives[-1])}", flush=True)
    lon0, lon1 = float(os.environ.get("LON0", -118)), float(os.environ.get("LON1", -92))
    lat0, lat1 = float(os.environ.get("LAT0", 12)), float(os.environ.get("LAT1", 32))
    plon, plat = read_grid(grid_b)
    lon1d, lat1d = plon[0], plat[:, 0]
    i0, i1 = np.searchsorted(lon1d, lon0), np.searchsorted(lon1d, lon1)
    j0, j1 = np.searchsorted(lat1d, lat0), np.searchsorted(lat1d, lat1)
    days, ssh = [], []
    for b in archives:
        if "dummy" in b:
            continue
        day, eta = read_archive_ssh(b)
        days.append(day)
        ssh.append(eta[j0:j1, i0:i1].astype(np.float32))
        print(f"{os.path.basename(b)}  day {day:.4f}  η {np.nanmin(ssh[-1]):+.2f} … {np.nanmax(ssh[-1]):+.2f} m",
              flush=True)
    np.savez_compressed(out, model_day=np.array(days), ssh=np.array(ssh), lon=lon1d[i0:i1], lat=lat1d[j0:j1],
                        i0=i0, j0=j0, files=np.array([os.path.basename(b) for b in archives if "dummy" not in b]))
    print("saved", out)


if __name__ == "__main__":
    if len(sys.argv) < 4:
        sys.exit(__doc__)
    main(sys.argv[1:])
