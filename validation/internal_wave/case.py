"""
MITgcm verification/internal_wave (spec 2026-10-12-baroclinic-obc-internal-wave §3): inputs (delXvar,
topog.slope, T.init, data's Tref), the west open-boundary profiles of its code/obcs_calc.F, MITgcm's partial-cell
depth (hFacMin = 0.2) and the MNC reader.  MITgcm imposes T in its boundary cell (column 1, obcs_apply_ts.F:205)
and u at the face east of it (obcs_apply_uv.F:205): the pyOCEAN domain is MITgcm's columns 2 … 60 with its west
face at that u point.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

SRC = Path.home() / "pyOCEAN_refs/MITgcm/verification/internal_wave"
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave")
PERIOD = 44567.0                         # obcs_calc.F obTimeScale [s]
U0, N2, ALPHA, G, F0 = 0.024, 1e-6, 2e-4, 9.81, 0.0
NX, NZ, DZ = 60, 20, 10.0


def load_inputs() -> dict:
    """dx (60,), bed depth H (60,) [m] (NaN on land), T (20, 60) and Tref (20,) [°C] surface first, dz (20,)."""
    inp = SRC / "input"
    dx = np.fromfile(inp / "delXvar", ">f8")
    d = np.fromfile(inp / "topog.slope", ">f8")
    H = np.where(d < 0, -d, np.nan)
    T = np.fromfile(inp / "T.init", ">f8").reshape(NZ, NX)
    txt = (inp / "data").read_text()
    tref = np.array([float(v) for v in re.findall(r"[-+]?\d*\.\d+(?:[eE][-+]?\d+)?",
                                                  re.search(r"Tref *=([^a-zA-Z]*)", txt).group(1))][:NZ])
    return dict(dx=dx, H=H, T=T, tref=tref, dz=np.full(NZ, DZ))


def mitgcm_depth(H: np.ndarray, dz: np.ndarray, hfac_min: float = 0.2) -> np.ndarray:
    """
    MITgcm's effective depth with partial cells (ini_masks_etc.F:108-120, hFacMinDr = 0): the bottom cell's
    fraction hFac is dropped when < hFacMin/2, else raised to at least hFacMin.
    """
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    Hc = np.asarray(H, float)
    out = np.full(Hc.shape, np.nan)
    ok = np.isfinite(Hc) & (Hc > 0)
    l = np.clip(np.searchsorted(zf, np.where(ok, Hc, 0.0), side="left") - 1, 0, dz.size - 1)
    frac = (Hc - zf[l]) / dz[l]
    frac = np.where(frac < 0.5 * hfac_min, 0.0, np.maximum(frac, hfac_min))
    D = zf[l] + frac * dz[l]
    return np.where(ok & (D > 0), D, np.nan)


def obcs_west(t: float, dz: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """OBWu (20,) [m/s] and OBWt (20,) [°C], surface first, of internal_wave/code/obcs_calc.F at time t [s]."""
    rF = -np.concatenate([[0.0], np.cumsum(dz)])
    rC = 0.5 * (rF[1:] + rF[:-1])
    vs = np.cos(1.0 * np.pi * (rC / rF[-1]))
    tmpsum = (vs * dz).sum() / rF[-1]                  # obcs_calc.F literally (rF(Nr+1) < 0; Σ vs drF = 0 here)
    vs = vs - tmpsum
    omega = 2.0 * np.pi / PERIOD
    kx = 1.0 * 2.0 * np.pi / 400.0 * np.sqrt((omega**2 - F0**2) / (1e-6 - omega**2))
    dx1 = load_inputs()["dx"][0]
    u = U0 * vs * np.sin(omega * t) * np.cos(kx * (3.0 - 2.0 - 0.5) * dx1)
    strat = 1e-6 / (G * ALPHA)
    k = np.arange(1, dz.size + 1)
    T = load_inputs()["tref"] + U0 * np.sin(np.pi * (k - 0.5) / dz.size) * np.sin(omega * t) \
        * np.sqrt(strat / (ALPHA * G)) * np.sqrt(omega**2 - F0**2) / omega
    return u, T


def read_mitgcm(run: Path | None = None) -> dict:
    """
    MNC state of the MITgcm run: t [s], x centres [m], U (n, 20, 61), T (n, 20, 60), η (n, 60), Depth (60,).
    The case has two tiles along x (sNx = 30, nSx = 2): they are joined by X, the shared u face kept once.
    """
    import glob

    import netCDF4
    run = RUNS / "mitgcm" / "run" if run is None else run
    blocks = sorted({f.rsplit(".t", 1)[0] for f in glob.glob(str(run / "mnc_*" / "state.*.t*.nc"))})
    if not blocks:
        raise FileNotFoundError(f"no MNC state files in {run}")
    t, U, T, eta = [], [], [], []
    for b in blocks:
        tiles = []
        for f in sorted(glob.glob(b + ".t*.nc")):
            with netCDF4.Dataset(f) as ds:
                tiles.append((float(ds["X"][0]), np.asarray(ds["T"][:], float), np.asarray(ds["U"][:, :, 0, :], float),
                              np.asarray(ds["Temp"][:, :, 0, :], float), np.asarray(ds["Eta"][:, 0, :], float),
                              np.asarray(ds["X"][:], float)))
        tiles.sort(key=lambda a: a[0])
        t.append(tiles[0][1])
        U.append(np.concatenate([tiles[0][2]] + [a[2][..., 1:] for a in tiles[1:]], axis=-1))
        T.append(np.concatenate([a[3] for a in tiles], axis=-1))
        eta.append(np.concatenate([a[4] for a in tiles], axis=-1))
        x = np.concatenate([a[5] for a in tiles])
    o = np.argsort(np.concatenate(t))
    cat = lambda a: np.concatenate(a)[o]                                  # noqa: E731
    depth = []
    for f in sorted(glob.glob(str(run / "mnc_*" / "grid.t*.nc"))):
        with netCDF4.Dataset(f) as ds:
            depth.append((float(ds["X"][0]), np.asarray(ds["Depth"][0, :], float)))
    depth = np.concatenate([d for _, d in sorted(depth, key=lambda a: a[0])])
    return dict(t=cat(t), x=x, U=cat(U), T=cat(T), eta=cat(eta), depth=depth)
