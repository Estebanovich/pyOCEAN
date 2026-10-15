"""The common output series of the DOME benchmark (spec 2026-10-14-dome-overflow §7; as validation/ilicak_benchmark/series.py
plus the inflow dye ``s``): every reader writes it, metrics read it.  Layers are surface first (k = 0 at the surface); u on
the nx+1 x-faces, v on the ny+1 y-faces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KEYS = ("t", "x", "y", "dx", "dy", "H", "h", "T", "s", "u", "v", "eta")


def _check(s: dict) -> None:
    nt, nz, ny, nx = np.shape(s["h"])
    want = dict(t=(nt,), x=(nx,), y=(ny,), dx=(nx,), dy=(ny,), H=(ny, nx), T=(nt, nz, ny, nx),
                s=(nt, nz, ny, nx), u=(nt, nz, ny, nx + 1), v=(nt, nz, ny + 1, nx), eta=(nt, ny, nx))
    for k, shp in want.items():
        if np.shape(s[k]) != shp:
            raise ValueError(f"series {k}: shape {np.shape(s[k])}, expected {shp}")


def save(path: Path, s: dict, meta: dict) -> Path:
    """Write the series ``s`` (KEYS) and ``meta`` (JSON) to ``path`` (.npz)."""
    _check(s)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, meta_json=json.dumps(meta), **{k: np.asarray(s[k], float) for k in KEYS})
    return path


def load(path: Path) -> tuple[dict, dict]:
    """(series, meta) from ``path``."""
    with np.load(path) as f:
        return {k: f[k] for k in KEYS}, json.loads(str(f["meta_json"]))
