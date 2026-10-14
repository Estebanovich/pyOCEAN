"""
HYCOM 2.3 inputs of the Ilicak overflow cost case (spec 2026-10-13-ilicak-four-model-benchmark-design §A3), from
validation/ilicak_benchmark/cases.py: .a/.b grid and depth (mod_za_mp.h:99, 660, 1050; geopar.F90:37-253), the
initial state as 12-month relax files read with iniflg = 2 (forfun.F90:2037-2085, 2741; inicon.F90:70-88; relax = 0
so nothing relaxes), 7-term σ₀ of stmt_fns.h:51-59, 317 (built with -DEOS_SIG0 -DEOS_7T).
Closed domain: last column and row are land (bigrid.F90:26-95), first column and row too.
"""
from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

import numpy as np

from validation.ilicak_benchmark import cases

SPVAL = 2.0 ** 100                                       # HYCOM land / data void (mod_za_mp.h:99)
ONEM = 9806.0                                            # Pa per metre (mod_cb_arrays.F90:844)
S0 = 35.0                                                # constant salinity [psu]
C = (-1.36471e-1, 4.68181e-2, 8.07004e-1, -7.45353e-3, -2.94418e-3, 3.43570e-5, 3.48658e-5)


def sig0(T, S) -> np.ndarray:
    """HYCOM 7-term σ₀ [kg m⁻³ − 1000] (stmt_fns.h:317):
    σ₀ = c1 + c3 S + T (c2 + c5 S + T (c4 + c7 S + c6 T)), T [°C], S [psu]."""
    c1, c2, c3, c4, c5, c6, c7 = C
    T, S = np.asarray(T, float), np.asarray(S, float)
    return c1 + c3 * S + T * (c2 + c5 * S + T * (c4 + c7 * S + c6 * T))


def geometry() -> dict:
    """The overflow on HYCOM's p-grid (j, i): 200 × 3 ocean cells framed by land; depth [m] and T [°C] (NaN on
    land), every ocean row the bed of cases.py; dz [m] the 100 z-levels of the case."""
    c = cases.get("overflow")
    idm, jdm = c.nx + 2, 5
    depth = np.full((jdm, idm), np.nan)
    depth[1:-1, 1:-1] = c.H[0][None, :]
    T = np.full((jdm, idm), np.nan)
    T[1:-1, 1:-1] = np.where(c.x < 20e3, 10.0, 20.0)[None, :]
    return dict(idm=idm, jdm=jdm, depth=depth, T=T, S0=S0, dz=c.dz)


def _a(path: Path, arrays: list[np.ndarray]) -> list[tuple[float, float]]:
    """Write .a records (big-endian real*4, padded to 4096 words, NaN → 2^100); returns (min, max) of each."""
    out = []
    with open(path, "wb") as f:
        for a in arrays:
            a32 = np.where(np.isnan(a), SPVAL, a).ravel()
            pad = (-a32.size) % 4096
            np.concatenate([a32, np.full(pad, SPVAL)]).astype(">f4").tofile(f)     # concatenate returns native order
            v = a.astype(np.float32)[np.isfinite(a)]
            out.append((float(v.min()), float(v.max())))
    return out


def write_ab(base: Path, records: list[tuple[str, np.ndarray]], header: list[str],
             fmt: Callable[[str, int, float, float], str]) -> None:
    """base.a + base.b: header lines, then fmt(name, k, lo, hi) per record."""
    mm = _a(Path(f"{base}.a"), [r for _, r in records])
    lines = list(header) + [fmt(name, k, lo, hi) for k, ((name, _), (lo, hi)) in enumerate(zip(records, mm))]
    Path(f"{base}.b").write_text("\n".join(lines) + "\n")


def write_grid(out: Path) -> None:
    """regional.grid.[ab]: 18 records in geopar.F90:37-173 order; Δx = Δy = 1000 m, pang = 0, f = 0."""
    g = geometry()
    jdm, idm = g["jdm"], g["idm"]
    x = np.broadcast_to(np.arange(idm, dtype=float)[None, :] * 0.01, (jdm, idm))     # labels only (not metrics)
    y = np.broadcast_to(np.arange(jdm, dtype=float)[:, None] * 0.01, (jdm, idm))
    one = np.full((jdm, idm), 1000.0)
    recs = [("plon", x), ("plat", y), ("qlon", x), ("qlat", y), ("ulon", x), ("ulat", y), ("vlon", x), ("vlat", y),
            ("pang", np.zeros((jdm, idm)))] + [(n, one) for n in ("pscx", "pscy", "qscx", "qscy", "uscx", "uscy",
                                                                  "vscx", "vscy")] + [("cori", np.zeros((jdm, idm)))]
    write_ab(Path(out) / "regional.grid", recs,
             [f"{idm:5d}    'idm   ' = longitudinal array size", f"{jdm:5d}    'jdm   ' = latitudinal  array size",
              "    2    'mapflg' = map flag (uniform)"],
             lambda n, k, lo, hi: f"{n}:  min,max = {lo:15.5f} {hi:15.5f}")


def write_depth(out: Path) -> None:
    """regional.depth.[ab]: 5 title lines and one 'min,max depth' record (geopar.F90:208-253)."""
    g = geometry()
    write_ab(Path(out) / "regional.depth", [("depth", g["depth"])],
             ["Ilicak overflow (pyOCEAN four-model benchmark)", "tanh shelf 500 -> 2000 m, Petersen 2015 eq. 24",
              "closed: land in the first/last column and row", "", ""],
             lambda n, k, lo, hi: f"min,max depth = {lo:10.3f} {hi:10.3f}")


def write_relax(out: Path, kdm: int = 100) -> None:
    """relax.{temp,saln,intf}.[ab]: 12 identical months of T, S and interface pressure [Pa] (k = 1…kdm; interface
    k on top of layer k, the z-levels of the case cut at the bed), plus an empty relax.weird."""
    g = geometry()
    zf = np.concatenate([[0.0], np.cumsum(g["dz"])])[:kdm]
    d = g["depth"]
    rho = sig0(np.nanmin(g["T"]), S0)
    tem, sal, itf = [], [], []
    for _ in range(12):
        for k in range(kdm):
            tem.append(("tem", g["T"]))
            sal.append(("sal", np.where(np.isfinite(d), S0, np.nan)))
            itf.append(("int", np.where(np.isfinite(d), np.minimum(zf[k], np.nan_to_num(d)) * ONEM, np.nan)))
    fmt = lambda n, i, lo, hi: (f"{n}: month,layer,dens,range = {i // kdm + 1:02d} {i % kdm + 1:02d} "   # noqa: E731
                                f"{rho:.3f} {lo:.6e} {hi:.6e}")
    head = ["Ilicak overflow initial state (iniflg = 2; relax = 0)", "", "", "", ""]
    for name, recs in (("relax.temp", tem), ("relax.saln", sal), ("relax.intf", itf)):
        write_ab(Path(out) / name, recs, head, fmt)
    (Path(out) / "relax.weird").write_text("")
