"""
Blow-up post-mortem: detect NaN/Inf or out-of-range fields during a run,
report *where* and *what* (field, indices, value, lat/lon, depth, local CFL,
recent history of the maxima) and dump the full state for inspection.

Users of community models repeatedly ask for exactly this (Oceananigans
discussion #4333 "NaN checker that reports where"; HYCOM "neg dp" crashes;
MOM6/COSIMA time-step threads — docs/design/vision_roadmap.md §3b, item 9).

Usage in a time loop::

    guard = BlowupGuard(limits={"eta": 20.0, "u": 5.0}, dump_path=out / "blowup.npz",
                        lat=lat_t, lon=lon_t, depth=H, dt=dt, dx=dx_min)
    for n in range(nsteps):
        eta, u, v = model.step(...)
        guard.check(step=n, time=t, eta=eta, u=u, v=v)

Index convention: the last two axes are (j, i) of the field's own grid (cells,
u- or v-faces); a leading axis is the layer k.  lat/lon/depth are looked up at
the nearest cell (i clipped for face arrays).
"""

from __future__ import annotations

from collections import deque
from pathlib import Path

import numpy as np


class BlowupError(RuntimeError):
    """Raised by ``BlowupGuard`` with a located diagnosis in the message."""


class BlowupGuard:
    """
    limits : {field: max |value|} — exceeding it (or any NaN/Inf in any checked
             field) triggers the post-mortem
    dump_path : .npz written with every field passed to ``check`` plus step/time
    lat, lon, depth : (ny, nx) cell arrays for the report;  dt, dx : for the local CFL of velocities
    history : number of past steps whose maxima are listed in the report
    """

    def __init__(self, limits: dict[str, float] | None = None, dump_path: str | Path | None = None,
                 lat: np.ndarray | None = None, lon: np.ndarray | None = None, depth: np.ndarray | None = None,
                 dt: float | None = None, dx: float | None = None, history: int = 5,
                 velocity_fields: tuple[str, ...] = ("u", "v")) -> None:
        self.limits = dict(limits or {})
        self.dump_path = Path(dump_path) if dump_path is not None else None
        self.lat, self.lon, self.depth, self.dt, self.dx = lat, lon, depth, dt, dx
        self.velocity_fields = velocity_fields
        self.recent: deque = deque(maxlen=history)

    def _where(self, a: np.ndarray, flat: int) -> tuple[str, int, int]:
        idx = np.unravel_index(flat, a.shape)
        j, i = int(idx[-2]), int(idx[-1])
        lab = ", ".join(f"{n}={v}" for n, v in zip(("k", "j", "i")[-len(idx):] if len(idx) <= 3 else
                                                   [f"a{q}" for q in range(len(idx))], idx))
        return lab, j, i

    def _context(self, name: str, a: np.ndarray, flat: int, value: float) -> str:
        lab, j, i = self._where(a, flat)
        parts = [f"{name}[{lab}] = {value:.6g}"]
        ref = self.lat if self.lat is not None else self.depth
        if ref is not None:
            jj, ii = min(j, ref.shape[0] - 1), min(i, ref.shape[1] - 1)
            if self.lat is not None and self.lon is not None:
                parts.append(f"at {self.lat[jj, ii]:.4f}°N {self.lon[jj, ii]:.4f}°E")
            if self.depth is not None:
                parts.append(f"depth {self.depth[jj, ii]:.1f} m")
        if name in self.velocity_fields and self.dt and self.dx and np.isfinite(value):
            parts.append(f"local CFL {abs(value) * self.dt / self.dx:.2f}")
        # neighbourhood of the offending point (same layer)
        sl = a[tuple(np.unravel_index(flat, a.shape)[:-2])] if a.ndim > 2 else a
        nb = sl[max(j - 1, 0):j + 2, max(i - 1, 0):i + 2]
        parts.append("neighbourhood\n" + np.array2string(nb, precision=3, suppress_small=True))
        return ", ".join(parts[:-1]) + "\n    " + parts[-1].replace("\n", "\n    ")

    def check(self, step: int, time: float, **fields: np.ndarray) -> None:
        """Check the fields of one step; raise ``BlowupError`` (after dumping) if something is wrong."""
        problems = []
        for name, a in fields.items():
            a = np.asarray(a)
            bad = ~np.isfinite(a)
            if bad.any():
                flat = int(np.flatnonzero(bad)[0])
                problems.append(f"{int(bad.sum())} NaN/Inf in {name}; first: " + self._context(name, a, flat, float(a.flat[flat])))
                continue
            lim = self.limits.get(name)
            if lim is not None and a.size:
                flat = int(np.argmax(np.abs(a)))
                v = float(a.flat[flat])
                if abs(v) > lim:
                    problems.append(f"|{name}| > {lim}: " + self._context(name, a, flat, v))
        if not problems:
            self.recent.append((step, time, {n: float(np.nanmax(np.abs(np.asarray(a)))) for n, a in fields.items()
                                             if np.asarray(a).size}))
            return
        lines = [f"Blow-up at step {step} (t = {time:.1f} s = {time / 86400:.3f} d):"] + [f"  {p}" for p in problems]
        for name in fields:
            hist = [(s, m[name]) for s, _, m in self.recent if name in m]
            if hist:
                lines.append(f"  recent max|{name}|: " + ", ".join(f"step {s}: {v:.3g}" for s, v in hist))
        if self.dump_path is not None:
            self.dump_path.parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(self.dump_path, step=step, time=time, **{k: np.asarray(v) for k, v in fields.items()})
            lines.append(f"  state dumped to {self.dump_path}")
        raise BlowupError("\n".join(lines))
