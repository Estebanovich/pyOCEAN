# Baroclinic Open-Boundary Forcing + MITgcm `internal_wave` Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let pyOCEAN's open boundaries take a time-varying exterior state, in two modes:
- `'radiating'`: Flather toward (η, ū)ₑₓₜ, Orlanski with CROCO-style nudging, and a sponge toward the exterior;
- `'clamped'`: Dirichlet.

Then compare with MITgcm's `internal_wave` case.

**Architecture:**
- **`OBData`** gains its own vertical grid (`zw`) and `ubar`.
- **A new pure module `core/dynamics/open_boundary.py`** turns `{edge: OBData}` into model arrays: T on the layers of the boundary and sponge cells, u′ on the face layers of the boundary and sponge faces, and the Flather targets.
- **`SigmaBaroclinic`** evaluates the provider once per step at tⁿ⁺¹ and uses those arrays in `_flather_args`, `_open_velocity`, `_radiate`, `_inflow_T` and the tracer sponge.
- **Without a provider**, everything is unchanged: σ bit for bit, ALE to its fingerprints.

**Tech Stack:** Python 3.12, NumPy, Numba (existing kernels), netCDF4, pytest; MITgcm (gfortran, Esteban's optfile, no MPI).

**Spec:** `docs/superpowers/specs/2026-10-12-baroclinic-obc-internal-wave-design.md`

## Global Constraints

- **Cores:** ≤ 5 threads (`NUMBA_NUM_THREADS=5`), never the E-cores, one heavy run at a time.
- **Outputs:** large outputs go to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave/`. Never commit `*.npz`, `log_*.txt` or `pyOCEAN-literature-review-guide.docx`.
- **MITgcm source:** unchanged. Use the case's own `code/` (with its `obcs_calc.F`) and `input/`, optfile `darwin_arm64_gfortran_esteban` and `NETCDF_ROOT=/opt/homebrew`. Without MPI, its output goes to stdout (`output.txt`).
- **Default nudging:** `obc_tau_in = 86400 s`, `obc_tau_out = 864000 s` (CROCO `read_inp.F:2576-2577`, given in days).
- **Without `open_boundary_state`:** σ is bit-identical (`np.array_equal`); ALE matches `tests/data/ale_perf_reference.json`.
- **Commits:** one per task. Messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn
  ```
- **Benchmark rules:** classify every difference (a)–(d); never tune; on a class (a) failure, stop and report to Esteban.

## Review Focus

1. **Exterior data shallower than the boundary column** (`zw[-1] > −H` at an edge). Expected: a clear `ValueError` naming the edge. *Pinned in Task 2, `test_source_must_reach_the_edge_bed`.*
2. **Sponge columns deeper or shallower than the edge column.** Expected: the edge profile extends or clips to each column's own bed, with no NaN. *Pinned in Task 2, `test_exterior_tracer_fills_the_sponge_band`.*
3. **Corner cells inside two sponge bands.** Expected: they take the nearest edge's profile, deterministically. *Pinned in Task 2, `test_nearest_edge_wins_in_corners`.*
4. **A provider returning an edge that is not open, or a mode for a closed edge.** Expected: a `ValueError` at construction or at the first step. *Pinned in Task 3, `test_open_boundary_state_errors`.*
5. **The u′ₑₓₜ depth mean.** Expected: Σ w u′ₑₓₜ = 0 on every open face, so the free-surface flux stays Flather's. *Pinned in Task 2, `test_exterior_velocity_has_zero_weighted_mean`.*

---

### Task 1: `OBData` with its own vertical grid; `interpolated_boundary`

**Files:**
- Modify: `boundaries/obcs/__init__.py` (dataclass `OBData`, near line 124)
- Create: `forcing/boundary_time.py`
- Test: `tests/test_open_boundary_state.py`

**Interfaces:**
- **Produces:**
  - `OBData(side, T=None, S=None, u_normal=None, eta=None, zw=None, ubar=None)`. `zw` is (nz+1, n), surface first, decreasing [m]; `ubar` is (n,).
  - `forcing.boundary_time.interpolated_boundary(times: Sequence[float], states: Sequence[dict[str, OBData]]) -> Callable[[float], dict[str, OBData]]`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests of the time-varying open-boundary exterior state (spec 2026-10-12-baroclinic-obc-internal-wave §1)."""
from __future__ import annotations

import numpy as np
import pytest

from boundaries.obcs import OBData


def _ob(val: float, n: int = 3, nz: int = 4) -> OBData:
    zw = np.linspace(0.0, -100.0, nz + 1)[:, None].repeat(n, 1)
    return OBData(side="west", T=np.full((nz, n), val), u_normal=np.full((nz, n), 2 * val), zw=zw,
                  eta=np.full(n, 0.1 * val), ubar=np.full(n, 0.5 * val))


def test_obdata_has_its_own_vertical_grid_and_ubar():
    ob = _ob(1.0)
    assert ob.zw.shape == (5, 3) and ob.ubar.shape == (3,)
    assert OBData(side="east").zw is None and OBData(side="east").ubar is None   # Phase-1 use unchanged


def test_interpolated_boundary_is_linear_in_time_and_constant_outside():
    from forcing.boundary_time import interpolated_boundary
    f = interpolated_boundary([0.0, 100.0], [{"west": _ob(1.0)}, {"west": _ob(3.0)}])
    mid = f(25.0)["west"]
    np.testing.assert_allclose(mid.T, 1.5)
    np.testing.assert_allclose(mid.u_normal, 3.0)
    np.testing.assert_allclose(mid.eta, 0.15)
    np.testing.assert_allclose(mid.ubar, 0.75)
    np.testing.assert_allclose(mid.zw, _ob(1.0).zw)
    np.testing.assert_allclose(f(-5.0)["west"].T, 1.0)
    np.testing.assert_allclose(f(500.0)["west"].T, 3.0)


def test_interpolated_boundary_rejects_inconsistent_states():
    from forcing.boundary_time import interpolated_boundary
    with pytest.raises(ValueError, match="increasing"):
        interpolated_boundary([10.0, 0.0], [{"west": _ob(1.0)}, {"west": _ob(2.0)}])
    with pytest.raises(ValueError, match="edges"):
        interpolated_boundary([0.0, 1.0], [{"west": _ob(1.0)}, {"east": _ob(2.0)}])
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_open_boundary_state.py -q`
Expected: FAIL. `OBData` has no `zw`, and the module `forcing.boundary_time` does not exist.

- [ ] **Step 3: Implement**

In `boundaries/obcs/__init__.py`, extend `OBData`:
- add these lines to its docstring:
  ```
  zw : np.ndarray (nz+1, n_bdy) or None
      Interface heights of the data's own vertical grid [m], surface first, decreasing (needed by the ALE/σ
      model with a time-varying exterior; spec 2026-10-12-baroclinic-obc-internal-wave §1.1).
  ubar : np.ndarray (n_bdy,) or None
      Exterior depth-mean normal velocity for Flather [m/s]; None = 0.
  ```
- add these two fields:
  ```python
  zw: np.ndarray | None = field(default=None)
  ubar: np.ndarray | None = field(default=None)
  ```

Create `forcing/boundary_time.py`:

```python
"""
Time interpolation of open-boundary exterior states (spec 2026-10-12-baroclinic-obc-internal-wave §1.3): a list
of {edge: OBData} snapshots on the same grids → a function t → {edge: OBData}, linear in time and constant
beyond the ends.  The basis of the HYCOM/ESPC boundary adapters (xarray stays at the I/O boundary).
"""
from __future__ import annotations

from typing import Callable, Sequence

import numpy as np

from boundaries.obcs import OBData

_FIELDS = ("T", "S", "u_normal", "eta", "ubar")


def interpolated_boundary(times: Sequence[float], states: Sequence[dict[str, OBData]]
                          ) -> Callable[[float], dict[str, OBData]]:
    """Linear-in-time exterior state between snapshots ``states`` at strictly increasing ``times`` [s]."""
    t = np.asarray(times, float)
    if t.ndim != 1 or t.size != len(states) or t.size == 0 or (t.size > 1 and not (np.diff(t) > 0).all()):
        raise ValueError("times must be strictly increasing, one per state")
    edges = set(states[0])
    if any(set(s) != edges for s in states):
        raise ValueError("every state must have the same edges")

    def at(time: float) -> dict[str, OBData]:
        k = int(np.clip(np.searchsorted(t, time, side="right") - 1, 0, t.size - 1))
        k1 = min(k + 1, t.size - 1)
        w = 0.0 if k1 == k else float(np.clip((time - t[k]) / (t[k1] - t[k]), 0.0, 1.0))
        out = {}
        for e in edges:
            a, b = states[k][e], states[k1][e]
            vals = {f: None if getattr(a, f) is None else (1 - w) * getattr(a, f) + w * getattr(b, f) for f in _FIELDS}
            out[e] = OBData(side=a.side, zw=a.zw, **vals)
        return out
    return at
```

- [ ] **Step 4: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_open_boundary_state.py tests/test_surface_forcing.py -q`
Expected: all pass (the Phase-1 `OBData` users in `test_surface_forcing.py` are unchanged).

- [ ] **Step 5: Commit**

```bash
git add boundaries/obcs/__init__.py forcing/boundary_time.py tests/test_open_boundary_state.py
git commit -m "Open boundaries: OBData with its own vertical grid and ubar; interpolated_boundary (task 1)"
```

---

### Task 2: `core/dynamics/open_boundary.py`, from exterior state to model arrays

**Files:**
- Create: `core/dynamics/open_boundary.py`
- Test: `tests/test_open_boundary_state.py` (append)

**Interfaces:**
- **Consumes:** `OBData` (Task 1); `core.vertical.remap.remap_columns(h_src, f_src, h_tgt, scheme, wet)` (N0 ≠ N1 allowed); `core.vertical.ale.baroclinic_split(vel, w) -> (mean, prime)`.
- **Produces:**
  - `validate_state(state: dict, open_edges: tuple[str, ...], shape: tuple[int, int]) -> None` (raises `ValueError`);
  - `exterior_tracer(state, H, h, mask, scheme) -> np.ndarray`: (N, ny, nx), Tₑₓₜ on the layers h in the `mask` columns, 0 elsewhere;
  - `exterior_velocity(state, kind, Hf, hf, mask, w, scheme) -> np.ndarray`: u′ₑₓₜ on the face layers hf, with `kind` "u" (N, ny, nx+1) or "v" (N, ny+1, nx), 0 outside `mask`;
  - `flather_targets(state, shape_u, shape_v) -> dict`: `flather_eta=(zu, zv)`, `flather_ubar=zu2` and `flather_vbar=zv2` (the arrays `_flather_args` returns today).

- [ ] **Step 1: Write the failing tests** (append)

```python
from core.dynamics import open_boundary as obc
from core.vertical import remap as rm


def _west(ny: int = 2, nz: int = 40, depth: float = 100.0, T=None, u=None) -> OBData:
    zw = np.linspace(0.0, -depth, nz + 1)[:, None].repeat(ny, 1)
    zc = 0.5 * (zw[1:] + zw[:-1])
    T = 10.0 + 0.1 * zc if T is None else T
    u = np.cos(np.pi * zc / depth) if u is None else u
    return OBData(side="west", T=T, u_normal=u, zw=zw)


def test_validate_state_errors():
    with pytest.raises(ValueError, match="not in open_boundaries"):
        obc.validate_state({"west": _west()}, ("east",), (2, 5))
    with pytest.raises(ValueError, match="shape"):
        obc.validate_state({"west": _west(ny=3)}, ("west",), (2, 5))           # edge length 2, data 3
    bad = _west()
    bad.zw = bad.zw[::-1]
    with pytest.raises(ValueError, match="decrease"):
        obc.validate_state({"west": bad}, ("west",), (2, 5))


def test_source_must_reach_the_edge_bed():
    H = np.full((2, 5), 150.0)                                          # deeper than the 100 m data
    h = np.full((5, 2, 5), 30.0)
    mask = np.zeros((2, 5), bool)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="west.*bed"):
        obc.exterior_tracer({"west": _west()}, H, h, mask, rm.PPM_H4)


def test_exterior_tracer_on_layers_and_sponge_band():
    # linear T(z) on the data grid → exact layer means on any model layers; the sponge band takes the edge
    # profile, extended or clipped to each column's own bed (review focus 2)
    H = np.array([[100.0, 120.0, 75.0, 100.0, 100.0]] * 2)              # 75 m: a data interface
    nz = 7
    h = (H / nz)[None].repeat(nz, 0)
    mask = np.zeros((2, 5), bool)
    mask[:, :3] = True
    T = obc.exterior_tracer({"west": _west()}, H, h, mask, rm.PPM_H4)
    zc = -H[None] + np.cumsum(h, 0) - 0.5 * h
    inner = slice(1, -1)                                                # end cells are PCM (remap.py)
    np.testing.assert_allclose(T[inner, :, 0], (10.0 + 0.1 * zc)[inner, :, 0], atol=1e-10)
    np.testing.assert_allclose(T[inner, :, 2], (10.0 + 0.1 * zc)[inner, :, 2], atol=1e-10)   # clipped at 75 m
    assert np.isfinite(T[:, :, 1]).all() and (T[:, :, 3:] == 0).all()


def test_nearest_edge_wins_in_corners():
    H = np.full((4, 4), 100.0)
    h = np.full((2, 4, 4), 50.0)
    mask = np.ones((4, 4), bool)
    south = OBData(side="south", T=np.full((4, 4), 5.0), u_normal=np.zeros((4, 4)),
                   zw=np.linspace(0, -100, 5)[:, None].repeat(4, 1))
    west = _west(ny=4, nz=4, T=np.full((4, 4), 1.0), u=np.zeros((4, 4)))
    T = obc.exterior_tracer({"west": west, "south": south}, H, h, mask, rm.PPM_H4)
    assert (T[:, 3, 0] == 1.0).all() and (T[:, 0, 3] == 5.0).all()          # nearest edge
    assert (T[:, 0, 0] == 1.0).all()                                          # tie: first in EDGES order


def test_exterior_velocity_has_zero_weighted_mean():
    ny, nx, N = 2, 4, 6
    Hf = np.full((ny, nx + 1), 100.0)
    hf = (Hf / N)[None].repeat(N, 0)
    w = hf / hf.sum(0)
    mask = np.zeros((ny, nx + 1), bool)
    mask[:, :2] = True
    up = obc.exterior_velocity({"west": _west(u=_west().u_normal + 0.3)}, "u", Hf, hf, mask, w, rm.PPM_H4)
    np.testing.assert_allclose((w * up).sum(0)[:, :2], 0.0, atol=1e-14)        # the 0.3 depth mean is removed
    assert np.abs(up[:, :, 0]).max() > 0.5 and (up[:, :, 2:] == 0).all()


def test_flather_targets_at_the_open_faces():
    ob = _west()
    ob.eta, ob.ubar = np.array([0.1, 0.2]), np.array([0.01, 0.02])
    t = obc.flather_targets({"west": ob}, (2, 6), (3, 5))
    np.testing.assert_allclose(t["flather_eta"][0][:, 0], [0.1, 0.2])
    np.testing.assert_allclose(t["flather_ubar"][:, 0], [0.01, 0.02])
    assert (t["flather_eta"][0][:, 1:] == 0).all() and (t["flather_vbar"] == 0).all()
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_open_boundary_state.py -q -k "validate or source or exterior or nearest or flather"`
Expected: FAIL with `ModuleNotFoundError: core.dynamics.open_boundary`.

- [ ] **Step 3: Implement `core/dynamics/open_boundary.py`**

```python
"""
Time-varying exterior state at open boundaries → model arrays (spec 2026-10-12-baroclinic-obc-internal-wave §1.4):
each edge's OBData profiles live on their own z grid (``zw``, surface first) and are remapped conservatively onto
the model layers of the boundary and sponge cells/faces (the external data on its own z axis, as MOM6
MOM_open_boundary.F90:1188-1204).  Sponge cells take the profile of the nearest open edge at the same along-edge
position, clipped or extended to their own bed.  Pure functions on NumPy arrays; (N, ny, nx), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np

from boundaries.obcs import OBData
from core.vertical import remap as rm
from core.vertical.ale import baroclinic_split

EDGES = ("west", "east", "south", "north")


def validate_state(state: dict[str, OBData], open_edges: tuple[str, ...], shape: tuple[int, int]) -> None:
    """Edges open, shapes (nz, n) / (nz+1, n) with n the edge length, zw decreasing; ValueError otherwise."""
    ny, nx = shape
    for edge, ob in state.items():
        if edge not in open_edges:
            raise ValueError(f"open_boundary_state edge {edge!r} is not in open_boundaries {open_edges}")
        if ob.zw is None or ob.T is None or ob.u_normal is None:
            raise ValueError(f"{edge}: OBData needs zw, T and u_normal")
        n = ny if edge in ("west", "east") else nx
        nz = np.shape(ob.T)[0]
        if np.shape(ob.T) != (nz, n) or np.shape(ob.u_normal) != (nz, n) or np.shape(ob.zw) != (nz + 1, n):
            raise ValueError(f"{edge}: shapes T {np.shape(ob.T)}, u_normal {np.shape(ob.u_normal)}, zw "
                             f"{np.shape(ob.zw)}; expected ({nz}, {n}) and ({nz + 1}, {n})")
        if not (np.diff(ob.zw, axis=0) < 0).all():
            raise ValueError(f"{edge}: zw must decrease from the surface down")
        for name in ("eta", "ubar"):
            a = getattr(ob, name)
            if a is not None and np.shape(a) != (n,):
                raise ValueError(f"{edge}: {name} shape {np.shape(a)}, expected ({n},)")


def _along(edge: str, shape: tuple[int, int]) -> tuple[np.ndarray, np.ndarray]:
    """(along-edge index, distance from the edge in cells or faces) for every point of a (ny, nx) array."""
    ny, nx = shape
    j, i = np.mgrid[0:ny, 0:nx]
    return {"west": (j, i), "east": (j, nx - 1 - i), "south": (i, j), "north": (i, ny - 1 - j)}[edge]


def _nearest(state: dict[str, OBData], shape: tuple[int, int]) -> np.ndarray:
    """Index into EDGES of the nearest edge of ``state`` at every point (ties: first in EDGES order)."""
    present = [e for e in EDGES if e in state]
    d = np.stack([_along(e, shape)[1] for e in present])
    return np.array([EDGES.index(e) for e in present])[np.argmin(d, axis=0)]


def _source_columns(ob: OBData, edge: str, depth: np.ndarray, check: np.ndarray) -> tuple[np.ndarray, ...]:
    """
    Bed-first source thicknesses and fields (nz, ny, nx) at every point, from the edge profile at the same
    along-edge index, clipped to each point's own bed ``depth`` [m] (the deepest layer extended when the point
    is deeper than the data).  ``check``: points (the edge itself) where the data must reach the bed.
    """
    a, _ = _along(edge, depth.shape)
    zw = np.asarray(ob.zw, float)[:, a]                                 # (nz+1, ny, nx), surface first
    if (check & (zw[-1] > -depth + 1e-6)).any():
        raise ValueError(f"{edge}: exterior data do not reach the bed of the boundary column (zw[-1] > −H)")
    z = np.maximum(zw, -depth[None])
    z[-1] = -depth
    h = (z[:-1] - z[1:])[::-1]
    return h, np.asarray(ob.T, float)[:, a][::-1], np.asarray(ob.u_normal, float)[:, a][::-1]


def _remap_stretched(h_src: np.ndarray, f_src: np.ndarray, h: np.ndarray, wet: np.ndarray, scheme: int) -> np.ndarray:
    """Source columns stretched to the model column total (z*-like, as exterior_on_layers) and remapped."""
    tot_s, tot = h_src.sum(axis=0), h.sum(axis=0)
    ok = wet & (tot_s > 0) & (tot > 0)
    scale = np.where(ok, tot / np.where(ok, tot_s, 1.0), 1.0)
    out = rm.remap_columns(np.ascontiguousarray(h_src * scale[None]), np.ascontiguousarray(f_src),
                           np.ascontiguousarray(h), scheme, ok)
    return np.where(ok[None], out, 0.0)


def exterior_tracer(state: dict[str, OBData], H: np.ndarray, h: np.ndarray, mask: np.ndarray,
                    scheme: int) -> np.ndarray:
    """Tₑₓₜ on the layers h (N, ny, nx) in the ``mask`` columns (boundary + sponge), 0 elsewhere."""
    H = np.nan_to_num(np.asarray(H, float))
    near = _nearest(state, H.shape)
    out = np.zeros(h.shape)
    for e, ob in state.items():
        sel = mask & (near == EDGES.index(e))
        if not sel.any():
            continue
        h_src, T_src, _ = _source_columns(ob, e, H, sel & (_along(e, H.shape)[1] == 0))
        out = np.where(sel[None], _remap_stretched(h_src, T_src, h, sel, scheme), out)
    return out


def exterior_velocity(state: dict[str, OBData], kind: str, Hf: np.ndarray, hf: np.ndarray, mask: np.ndarray,
                      w: np.ndarray, scheme: int) -> np.ndarray:
    """
    u′ₑₓₜ on the face layers hf in the ``mask`` faces of the edges normal to ``kind`` ('u': west/east,
    'v': south/north): the data's normal velocity remapped to the face layers, minus its mean with the model
    face weights w (Σ w u′ₑₓₜ = 0, so the free-surface flux stays Flather's).  0 elsewhere.
    """
    Hf = np.nan_to_num(np.asarray(Hf, float))
    sub = {e: ob for e, ob in state.items() if (e in ("west", "east")) == (kind == "u")}
    out = np.zeros(hf.shape)
    if not sub:
        return out
    near = _nearest(sub, Hf.shape)
    for e, ob in sub.items():
        sel = mask & (near == EDGES.index(e)) & (Hf > 0)
        if not sel.any():
            continue
        h_src, _, u_src = _source_columns(ob, e, Hf, sel & (_along(e, Hf.shape)[1] == 0))
        u = _remap_stretched(h_src, u_src, hf, sel, scheme)
        out = np.where(sel[None], baroclinic_split(u, w)[1], out)
    return out


def flather_targets(state: dict[str, OBData], shape_u: tuple[int, int], shape_v: tuple[int, int]) -> dict:
    """Flather exterior (η, normal ū) at the open faces, zero elsewhere, in the free-surface argument layout."""
    zu, zv, ubar, vbar = np.zeros(shape_u), np.zeros(shape_v), np.zeros(shape_u), np.zeros(shape_v)
    for e, ob in state.items():
        eta = np.zeros(np.shape(ob.T)[1]) if ob.eta is None else np.asarray(ob.eta, float)
        ub = np.zeros_like(eta) if ob.ubar is None else np.asarray(ob.ubar, float)
        if e == "west":
            zu[:, 0], ubar[:, 0] = eta, ub
        elif e == "east":
            zu[:, -1], ubar[:, -1] = eta, ub
        elif e == "south":
            zv[0, :], vbar[0, :] = eta, ub
        else:
            zv[-1, :], vbar[-1, :] = eta, ub
    return dict(flather_eta=(zu, zv), flather_ubar=ubar, flather_vbar=vbar)
```

If `baroclinic_split` refuses arrays with all-zero columns (land faces), mask them with `np.where(sel[None], …)` before the call, and record a ruling.

- [ ] **Step 4: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_open_boundary_state.py -q`
Expected: all pass.

- [ ] **Step 5: Commit**

```bash
git add core/dynamics/open_boundary.py tests/test_open_boundary_state.py
git commit -m "Open boundaries: exterior state to model arrays (task 2)"
```

---

### Task 3: The model takes the exterior state (`'radiating'`)

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py`:
  - `__init__` (around lines 700-760): new arguments and validation;
  - `_flather_args`, `_open_velocity` and `_radiate` (around 1268-1320);
  - `_inflow_T` and `_exterior` (around 1321-1336);
  - `_step_ale` (around 1412-1470): evaluate the state, tracer sponge;
  - `step` (around 1500-1560): σ path, same.
- Test: `tests/test_sigma_ale.py` (append) and `tests/test_sigma_baroclinic.py` (append).

**Interfaces:**
- **Consumes:** Task 2 functions; `SigmaState.time`.
- **Produces:**
  - `SigmaBaroclinic(..., open_boundary_state=None, open_boundary_mode=None, obc_tau_in=86400.0, obc_tau_out=864000.0)`;
  - `self._ext`, a dict for the current step with keys `t`, `state`, `flather`, `up_u`, `up_v`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sigma_ale.py`:

```python
from boundaries.obcs import OBData


def _west_state(T_of_z, u_of_z, depth=20.0, nz=40, ny=1):
    zw = np.linspace(0.0, -depth, nz + 1)[:, None].repeat(ny, 1)
    zc = 0.5 * (zw[1:] + zw[:-1])
    return OBData(side="west", T=T_of_z(zc), u_normal=u_of_z(zc), zw=zw)


def test_open_boundary_state_errors():
    with pytest.raises(ValueError, match="open_boundary_mode"):
        _ale_box("zstar", open_boundaries=("west",), open_boundary_mode={"east": "clamped"})
    with pytest.raises(ValueError, match="open_boundary_mode"):
        _ale_box("zstar", open_boundaries=("west",), open_boundary_mode={"west": "sticky"})
    m = _ale_box("zstar", open_boundaries=("west",),
                 open_boundary_state=lambda t: {"east": _west_state(lambda z: 10 + 0 * z, lambda z: 0 * z)})
    st = m.initial_state(np.zeros((1, 32)), np.full((10, 1, 32), 10.0))
    with pytest.raises(ValueError, match="not in open_boundaries"):
        m.step(st)


def test_no_state_keeps_the_resting_exterior_bit_for_bit():
    # the existing open-boundary channel with and without the new arguments at their defaults
    kw = dict(open_boundaries=("west", "east"), sponge_cells=4)
    a = _ale_box("zstar", **kw)
    b = _ale_box("zstar", open_boundary_state=None, open_boundary_mode=None, **kw)
    T = 10.0 + np.linspace(0, 1, 32)[None, None, :] * np.ones((10, 1, 1))
    sa, sb = a.initial_state(np.zeros((1, 32)), T), b.initial_state(np.zeros((1, 32)), T)
    for _ in range(20):
        sa, sb = a.step(sa), b.step(sb)
    assert np.array_equal(sa.u, sb.u) and np.array_equal(sa.T, sb.T) and np.array_equal(sa.h, sb.h)


def test_radiating_relaxes_the_boundary_baroclinic_velocity_to_the_exterior():
    # constant exterior u′ = 0.05 cos(πz/H); with τ_in = 10Δt the incoming boundary u′ approaches it
    H = 20.0
    st_fn = lambda t: {"west": _west_state(lambda z: 10 + 0 * z, lambda z: 0.05 * np.cos(np.pi * z / H))}  # noqa: E731
    m = _ale_box("zstar", open_boundaries=("west",), open_boundary_state=st_fn, obc_tau_in=300.0)
    st = m.initial_state(np.zeros((1, 32)), np.full((10, 1, 32), 10.0))
    for _ in range(100):
        st = m.step(st)
    zc = m.layer_heights(st.eta, st.h)[:, 0, 0]
    up = st.u[:, 0, 0] - st.u[:, 0, 0].mean()
    ref = 0.05 * np.cos(np.pi * zc / H)
    assert np.corrcoef(up, ref)[0, 1] > 0.99 and abs(np.std(up) / np.std(ref) - 1) < 0.2


def test_flather_follows_the_exterior_surface():
    # exterior η = 1 cm at a west edge of a channel closed in the east: the basin fills towards 1 cm
    st_fn = lambda t: {"west": OBData(side="west", T=np.full((4, 1), 10.0), u_normal=np.zeros((4, 1)),  # noqa: E731
                                      zw=np.linspace(0, -20, 5)[:, None], eta=np.array([0.01]))}
    m = _ale_box("zstar", nx=8, open_boundaries=("west",), open_boundary_state=st_fn)
    st = m.initial_state(np.zeros((1, 8)), np.full((10, 1, 8), 10.0))
    for _ in range(400):
        st = m.step(st)
    np.testing.assert_allclose(st.eta, 0.01, atol=1e-3)
```

Append to `tests/test_sigma_baroclinic.py` (σ uses the same arrays):

```python
def test_sigma_open_boundary_state_inflow_temperature():
    # σ path: inflow T comes from the time-varying exterior (exterior 12 °C into a 10 °C channel)
    from boundaries.obcs import OBData
    st_fn = lambda t: {"west": OBData(side="west", T=np.full((4, 1), 12.0), u_normal=np.full((4, 1), 0.05),  # noqa: E731
                                      zw=np.linspace(0, -20, 5)[:, None], ubar=np.array([0.05]))}
    m = _sigma_channel(open_boundaries=("west", "east"), open_boundary_state=st_fn)
    st = m.initial_state(np.zeros((1, m.H.shape[1])), np.full((m.n, 1, m.H.shape[1]), 10.0))
    for _ in range(200):
        st = m.step(st)
    assert st.T[:, 0, 0].mean() > 11.0
```

Before writing this test, check whether `tests/test_sigma_baroclinic.py` has a channel helper:

```bash
grep -n "^def _" tests/test_sigma_baroclinic.py
```

- If it has no `_sigma_channel`, add this one:
  ```python
  def _sigma_channel(nx=32, depth=20.0, **kw):
      hg = create_cartesian_grid(nx=nx, ny=1, dx=2000.0, dy=2000.0, f0=0.0)
      return SigmaBaroclinic(hg, np.full((1, nx), depth), dt=30.0, n_layers=10, rho0=1025.0,
                             t_coef=0.2, z0=None, closure="constant", nu_const=1e-5, **kw)
  ```
  with the imports the file already uses.
- If such a helper exists, use it and record a ruling.

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_sigma_baroclinic.py -q -k "open_boundary_state or no_state or radiating or flather_follows or inflow_temperature"`
Expected: FAIL with `TypeError: unexpected keyword argument 'open_boundary_state'`. The bit-for-bit test also fails, on `open_boundary_mode`.

- [ ] **Step 3: Implement**

**(a) `__init__`:**
- Add these arguments after `mix_diagnostic`:
  ```python
  open_boundary_state=None, open_boundary_mode: dict | None = None,
  obc_tau_in: float = 86400.0, obc_tau_out: float = 864000.0
  ```
- Document them in the class docstring (spec §1.2, §2; CROCO u3dbc.F:111-155).
- Validate:
  ```python
  self.obc_state = open_boundary_state
  self.obc_mode = {e: "radiating" for e in self.open}
  for e, mode in (open_boundary_mode or {}).items():
      if e not in self.open or mode not in ("radiating", "clamped"):
          raise ValueError(f"open_boundary_mode {{{e!r}: {mode!r}}}: edges must be open and modes "
                           "'radiating' or 'clamped'")
      self.obc_mode[e] = mode
  if open_boundary_state is not None and not self.open:
      raise ValueError("open_boundary_state needs open_boundaries")
  self.obc_tau_in, self.obc_tau_out = float(obc_tau_in), float(obc_tau_out)
  self._ext = None
  ```
- Build the face masks of the open edge and the sponge band:
  ```python
  self._bc_u = np.zeros(self.fs.H_u.shape, bool)
  self._bc_v = np.zeros(self.fs.H_v.shape, bool)
  ```
  Set column 0 (west) or −1 (east) of `_bc_u`, and row 0 or −1 of `_bc_v`, for each open edge; then OR with `self._sp_u > 0` and `self._sp_v > 0` when there is a sponge.

**(b) Per-step evaluation.** New method:

```python
def _evaluate_exterior(self, t: float, hu: np.ndarray, hv: np.ndarray, wu, wv) -> None:
    """Exterior state at t (spec §1.4): Flather targets and u′ₑₓₜ on the face layers; None without a provider."""
    if self.obc_state is None:
        self._ext = None
        return
    state = self.obc_state(t)
    obc.validate_state(state, self.open, self.H.shape)
    Hu, Hv = np.nan_to_num(self.fs.H_u), np.nan_to_num(self.fs.H_v)
    wu_ = self.ds[:, None, None] * np.ones(hu.shape) if wu is None else wu
    wv_ = self.ds[:, None, None] * np.ones(hv.shape) if wv is None else wv
    self._ext = dict(t=t, state=state, flather=obc.flather_targets(state, Hu.shape, Hv.shape),
                     up_u=obc.exterior_velocity(state, "u", Hu, hu, self._bc_u, wu_, self.remap_code),
                     up_v=obc.exterior_velocity(state, "v", Hv, hv, self._bc_v, wv_, self.remap_code))
```

Call it:
- **ALE:** in `_step_ale`, after `wu, wv` are computed and before `self.core.step`:
  ```python
  self._evaluate_exterior(st.time + dt, hu, hv, wu, wv)
  ```
- **σ:** in `step`, before `self.core.step`, with the rest face layers:
  ```python
  self._evaluate_exterior(st.time + dt, np.nan_to_num(self.fs.H_u)[None] * self.ds[:, None, None],
                          np.nan_to_num(self.fs.H_v)[None] * self.ds[:, None, None], None, None)
  ```
- `import core.dynamics.open_boundary as obc` at the top. `self.remap_code` exists only in ALE: for σ, use `rm.PPM_H4` when `self.ale` is False.

**(c) `_flather_args`:**
```python
if not self.open:
    return {}
if self._ext is not None:
    return self._ext["flather"]
# … unchanged zeros …
```

**(d) `_open_velocity`:**
- The sponge relaxes u′ toward the exterior:
  ```python
  ext = (None, None) if self._ext is None else (self._ext["up_u"], self._ext["up_v"])
  for vel, rate, w, pe in ((u_new, self._sp_u, wu, ext[0]), (v_new, self._sp_v, wv, ext[1])):
      bar, prime = self._baroclinic_part(vel, w)
      r = self.dt * rate[None]
      vel[...] = bar[None] + (prime if pe is None else prime + r * pe) / (1.0 + r)
  ```
  Without a provider, `pe` is None and the expression is the old one exactly: `prime / (1 + r)`, same floating-point operations.
- Radiation: pass `ext=pe` and `mode=self.obc_mode[name]` to `_radiate`.

**(e) `_radiate(..., ext=None, mode='radiating')`:** after `pb` is computed and before the mean is removed, add:

```python
if ext is not None:
    pe = take(ext, edge)
    if mode == "clamped":
        pb = pe.copy()
    else:                                                       # CROCO u3dbc.F:119-127, 144-152
        inflow = cx <= 0.0
        alpha = np.where(inflow, self.dt / self.obc_tau_in, self.dt / self.obc_tau_out)
        pb = (1.0 - alpha) * pb + alpha * pe
```

With `ext=None` nothing changes.

**(f) Exterior T:**
- A new helper gives Tₑₓₜ on any layers when there is a provider:
  ```python
  def _exterior_T(self, h: np.ndarray) -> np.ndarray:
      """Tₑₓₜ on layers h in the boundary/sponge columns: from the provider when set, else the z profile."""
      if self._ext is not None:
          T = obc.exterior_tracer(self._ext["state"], self.H, h, self._bc_mask,
                                  self.remap_code if self.ale else rm.PPM_H4)
          return np.where(self._bc_mask[None], T, 0.0)
      return self._exterior(h) if self.ale else self.T_ext
  ```
- **ALE:** replace `self._exterior(st.h)` (inflow, about line 1457) and `self._exterior(h_lag)` (sponge, about line 1470) with `self._exterior_T(...)`.
- **σ:** set `self._T_in = self._exterior_T(D·Δσ layers)` before the advection when `self._ext` is not None, and use it in the sponge in place of `self.T_ext`. Without a provider, `self._T_in` stays None and `self.T_ext` is used as before.

- [ ] **Step 4: Run them, then the open-boundary and regression tests**

Run:
```bash
NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_sigma_baroclinic.py tests/test_open_boundary_state.py -q
```
Expected: all pass, including the existing channel and ALE fingerprint tests (`test_ale_performance_regression*`, open-boundary channel errors 0.79/0.16/0.016).

- [ ] **Step 5: Commit**

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py tests/test_sigma_baroclinic.py
git commit -m "Open boundaries: time-varying exterior state, Flather/Orlanski+nudging/sponge toward it (task 3)"
```

---

### Task 4: `'clamped'` mode and the analytic progressive wave

**Files:**
- Test: `tests/test_sigma_ale.py` (append). The `'clamped'` branch is in `_radiate` since Task 3; this task proves it.

**Interfaces:**
- **Consumes:** Task 3 arguments.
- **Produces:** `progressive_wave_state(u0, omega, N, H, nz)`, a test helper giving `t → {"west": OBData}` with the mode-1 wave at x = 0.

- [ ] **Step 1: Write the test**

```python
def _mode1_state(u0, omega, N, H, nz=80):
    """Mode-1 internal wave entering at x = 0: u = u0 cos(πz/H) sin(ωt), T′ from w = −∂ζ/∂t (linear, f = 0)."""
    zw = np.linspace(0.0, -H, nz + 1)[:, None]
    zc = 0.5 * (zw[1:] + zw[:-1])
    alpha_g = 9.81 * 2e-4
    Tz = N**2 / alpha_g                                         # background dT/dz [°C/m]
    k = np.pi * omega / (N * H)

    def at(t):
        u = u0 * np.cos(np.pi * zc / H) * np.sin(omega * t)
        # u_x = −u0 k cos(πz/H) cos(ωt − kx); w = −∫u_x dz = u0 k (H/π) sin(πz/H) cos(ωt − kx) (w = 0 at z = 0, −H);
        # ζ = ∫w dt = (u0 k H/(π ω)) sin(πz/H) sin(ωt) at x = 0; T = T0(z − ζ) = 20 + Tz z − Tz ζ
        zeta = (u0 * k * H / (np.pi * omega)) * np.sin(np.pi * zc / H) * np.sin(omega * t)
        return {"west": OBData(side="west", T=20.0 + Tz * zc - Tz * zeta, u_normal=u, zw=zw)}
    return at


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_clamped_west_radiating_east_carries_the_analytic_mode1_wave(rule):
    # flat 200 m channel, N² = 1e-6, period 44 567 s (MITgcm internal_wave): a progressive mode-1 wave
    # u = u0 cos(πz/H) sin(ωt − kx), k = πω/(NH) (λ ≈ 2.8 km) leaves through a radiating + sponge east edge.
    # Interior amplitude ±5 %, phase ±10° after 3 periods (spec §3.2)
    H, N, P, u0, nz, nx, dx = 200.0, 1e-3, 44567.0, 0.024, 20, 100, 100.0
    omega = 2 * np.pi / P
    k = np.pi * omega / (N * H)
    Tz = N**2 / (9.81 * 2e-4)
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    Hb = np.full((1, nx), H)
    T0 = lambda z: 20.0 + Tz * z                                     # noqa: E731
    if rule == "zstar":
        params = dict(dz_top=np.full(nz, H / nz))
    else:
        m0 = SigmaBaroclinic(hg, Hb, dt=250.0, n_layers=nz, rho0=RHO0, t_coef=RHO0 * 2e-4, t_ref=20.0, z0=None,
                             closure="constant", nu_const=1e-3, vertical_coordinate="ale", regrid="sigma",
                             regrid_params={})
        rho = m0.density(T0(np.linspace(0, -H, 50)))
        sp = (rho.max() - rho.min()) / nz                               # half an interval beyond each extreme
        params = dict(dz_top=np.full(nz, 1.0), rho_target=np.linspace(rho.min() - 0.5 * sp, rho.max() + 0.5 * sp, nz + 1))
    m = SigmaBaroclinic(hg, Hb, dt=250.0, n_layers=nz, rho0=RHO0, t_coef=RHO0 * 2e-4, t_ref=20.0, z0=None,
                        closure="constant", nu_const=1e-3, ah=1e-2, kh=1e-2, kv=1e-3, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=rule, regrid_params=params,
                        open_boundaries=("west", "east"), sponge_cells=20, sponge_tau=(3000.0, 250.0),
                        open_boundary_state=_mode1_state(u0, omega, N, H),
                        open_boundary_mode={"west": "clamped", "east": "radiating"})
    zf = np.linspace(-H, 0.0, 401)
    h_src = np.full((400, 1, nx), H / 400)
    zc_src = (zf[1:] + zf[:-1])[:, None, None] * 0.5 * np.ones((1, 1, nx))
    st = m.initial_state(np.zeros((1, nx)), T0(zc_src), h_src=h_src)
    steps = int(round(3 * P / 250.0))
    x = (np.arange(nx + 1) * dx)[None, :]
    probe = np.arange(5, 60)                                         # interior u-faces, away from the sponge
    rec = []
    for n in range(steps + int(round(P / 250.0))):                   # 3 periods spin-up, then one period recorded
        st = m.step(st)
        if n >= steps:
            hf = m._mean_face_thickness(st.h)[0][:, 0, :]
            w = hf / np.maximum(hf.sum(0), 1e-12)
            u_bot_mode = (w * st.u[:, 0, :] * np.cos(np.pi * (np.cumsum(hf, 0) - 0.5 * hf - H) / H)).sum(0)
            rec.append((st.time, u_bot_mode[probe]))
    t = np.array([r[0] for r in rec])
    a = np.array([r[1] for r in rec])                                # mode-1 projection ≈ (u0/2) sin(ωt − kx)
    c, s = np.cos(omega * t), np.sin(omega * t)
    A = np.linalg.lstsq(np.stack([s, c], 1), a, rcond=None)[0]       # a ≈ A0 sin ωt + A1 cos ωt
    amp = np.hypot(A[0], A[1]) / (0.5 * u0)
    phase = np.degrees(np.arctan2(-A[1], A[0]))                      # sin(ωt − φ) → φ = kx
    expected = np.degrees(k * x[0, probe])
    assert np.all(np.abs(amp - 1.0) < 0.05)
    dphi = (phase - expected + 180.0) % 360.0 - 180.0
    assert np.all(np.abs(dphi - np.median(dphi)) < 10.0) and abs(np.median(dphi)) < 10.0
```

- **Projection:** u on cos(πz/H), weighted by thickness, has amplitude u0/2 (half of the vertical average of cos²). Hence the `0.5 * u0`.
- **The two sign conventions** (`x` growing eastward; phase lag = kx) are part of the test. If the measured phase grows with −kx, the clamped forcing has the wrong sign: fix the code, not the test.

- [ ] **Step 2: Run it**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k analytic_mode1`
Expected: PASS for z\* and hybrid, in under about 1 min each. On a FAIL, diagnose with superpowers:systematic-debugging. Things to check:
- the remapped exterior on the clamped face (print u′ at face 0 against u0 cos(πz/H) sin(ωt));
- the reflection from the east (amplitude modulated along x with period λ/2).

Do not loosen the criteria without a ruling.

- [ ] **Step 3: Commit**

```bash
git add tests/test_sigma_ale.py core/dynamics/sigma_baroclinic.py
git commit -m "Open boundaries: clamped mode carries the analytic mode-1 wave (task 4)"
```

---

### Task 5: The `internal_wave` case: inputs, boundary function, MITgcm build/run and reader

**Files:**
- Create: `validation/internal_wave/case.py`, `validation/internal_wave/run_mitgcm.sh`
- Test: `tests/test_internal_wave.py`

**Interfaces:**
- **Produces:**
  - `case.SRC = Path.home() / "pyOCEAN_refs/MITgcm/verification/internal_wave"`;
  - `case.RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave")`;
  - `case.load_inputs() -> dict(dx (60,), H (60,), T (20, 60), tref (20,), dz (20,))` (H > 0 m, NaN on land; T surface first);
  - `case.mitgcm_depth(H, dz, hfac_min=0.2)`: MITgcm effective depth with partial cells;
  - `case.obcs_west(t, dz) -> (u (20,), T (20,))`: the `obcs_calc.F` OBWu/OBWt profiles at time t, surface first;
  - `case.PERIOD = 44567.0`;
  - `case.read_mitgcm() -> dict(t (n,), x (60,), U (n, 20, 61), T (n, 20, 60), eta (n, 60))`;
  - `run_mitgcm.sh build|run`.

- [ ] **Step 1: Write the failing tests**

```python
"""Tests of the MITgcm internal_wave benchmark tools (spec 2026-10-12-baroclinic-obc-internal-wave §3)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.internal_wave import case

HAVE = (case.SRC / "input" / "delXvar").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="MITgcm internal_wave inputs not found")


def test_inputs_match_gendata():
    d = case.load_inputs()
    assert d["dx"].shape == (60,) and np.allclose(d["dx"], 13.3e3 / 61)
    assert d["H"][0] == 200.0 and abs(d["H"][-2] - 40.0) < 1e-9 and np.isnan(d["H"][-1])
    assert d["T"].shape == (20, 60) and np.allclose(d["T"][:, 0], d["tref"])
    np.testing.assert_allclose(np.diff(d["tref"]), -0.00509684, rtol=1e-4)


def test_obcs_west_reproduces_obcs_calc():
    # obcs_calc.F: OBWu = U0 * (cos(π rC/rF(Nr+1)) − mean) * sin(2πt/P) * cos(kx (3−2−0.5) Δx);
    # OBWt = tRef + U0 sin(π (k−½)/Nr) sin(2πt/P) sqrt(strat/(α g)) sqrt(ω² − f²)/ω
    d = case.load_inputs()
    t = 0.3 * case.PERIOD
    u, T = case.obcs_west(t, d["dz"])
    zc = -(np.arange(20) + 0.5) * 10.0
    vs = np.cos(np.pi * zc / -200.0)                                       # cos(mz π rC/rF(Nr+1))
    vs = vs - (vs * 10.0).sum() / -200.0                                    # − tmpsum (= Σ vs drF / rF(Nr+1))
    omega = 2 * np.pi / case.PERIOD
    kx = 2 * np.pi / 400.0 * np.sqrt(omega**2 / (1e-6 - omega**2))
    np.testing.assert_allclose(u, 0.024 * vs * np.sin(omega * t) * np.cos(kx * 0.5 * d["dx"][0]), rtol=1e-12)
    strat = 1e-6 / (9.81 * 2e-4)
    dT = 0.024 * np.sin(np.pi * (np.arange(1, 21) - 0.5) / 20) * np.sin(omega * t) \
        * np.sqrt(strat / (2e-4 * 9.81)) * omega / omega
    np.testing.assert_allclose(T, d["tref"] + dT, rtol=1e-12)
    assert abs((u * d["dz"]).sum()) < 1e-15                                 # zero depth mean


def test_mitgcm_depth_partial_cells():
    dz = np.full(20, 10.0)
    np.testing.assert_allclose(case.mitgcm_depth(np.array([200.0, 123.0, 101.5, 100.5, 40.0]), dz),
                               [200.0, 123.0, 102.0, 100.0, 40.0])          # hFac < 0.1 dropped, else ≥ 0.2


def test_read_mitgcm_shapes():
    if not (case.RUNS / "mitgcm" / "run").exists():
        pytest.skip("MITgcm internal_wave run not done yet")
    d = case.read_mitgcm()
    n = d["t"].size
    assert d["U"].shape == (n, 20, 61) and d["T"].shape == (n, 20, 60) and n > 200
    assert abs(d["t"][-1] - 10 * case.PERIOD) < 3600.0
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_internal_wave.py -q`
Expected: FAIL with `ModuleNotFoundError: validation.internal_wave`.

- [ ] **Step 3: Implement `validation/internal_wave/case.py`**

```python
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
    """MNC state of the MITgcm run: t [s], x centres [m], U (n, 20, 61), T (n, 20, 60), η (n, 60)."""
    import glob

    import netCDF4
    run = RUNS / "mitgcm" / "run" if run is None else run
    files = sorted(glob.glob(str(run / "mnc_*" / "state.*.t*.nc")))
    if not files:
        raise FileNotFoundError(f"no MNC state files in {run}")
    t, U, T, eta = [], [], [], []
    for f in files:
        with netCDF4.Dataset(f) as ds:
            t.append(np.asarray(ds["T"][:], float))
            U.append(np.asarray(ds["U"][:, :, 0, :], float))
            T.append(np.asarray(ds["Temp"][:, :, 0, :], float))
            eta.append(np.asarray(ds["Eta"][:, 0, :], float))
            x = np.asarray(ds["X"][:], float)
    o = np.argsort(np.concatenate(t))
    cat = lambda a: np.concatenate(a)[o]                                  # noqa: E731
    return dict(t=cat(t), x=x, U=cat(U), T=cat(T), eta=cat(eta))
```

- If MITgcm writes `state.*.t001.nc` with two tiles (sNx = 30 × nSx = 2), `read_mitgcm` must concatenate tiles along X: read every `t00?` file of one time block, sort by `X`, then stack. Print `ncdump -h` of one file first and adapt.

- [ ] **Step 4: Write `validation/internal_wave/run_mitgcm.sh`**

```bash
#!/usr/bin/env bash
# MITgcm verification/internal_wave, unchanged code and inputs, run for 10 forcing periods (spec §3.1).
#   run_mitgcm.sh build | run
set -euo pipefail
[[ $# -ge 1 ]] || { echo "usage: $0 build|run" >&2; exit 2; }
MIT=/Volumes/ESTEBAN_MIT/MITgcm
SRC=$HOME/pyOCEAN_refs/MITgcm/verification/internal_wave
BASE=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave/mitgcm
case "$1" in
  build)
    rm -rf "$BASE/build"; mkdir -p "$BASE/build"; cd "$BASE/build"
    export NETCDF_ROOT=/opt/homebrew
    "$MIT/tools/genmake2" -rootdir "$MIT" -mods "$SRC/code" \
        -optfile "$MIT/tools/build_options/darwin_arm64_gfortran_esteban" > genmake.log 2>&1
    make depend > make_depend.log 2>&1
    make -j 5 > make.log 2>&1
    ls -la mitgcmuv ;;
  run)
    RUN="$BASE/run"; rm -rf "$RUN"; mkdir -p "$RUN"; cd "$RUN"
    cp "$SRC"/input/* .
    # 10 periods of 44 567 s at Δt = 500 s = 892 steps; state every 1800 s
    sed -i '' -e 's/nTimeSteps=100,/nTimeSteps=892,/' -e 's/dumpFreq=50000.,/dumpFreq=1800.,/' data
    grep -E "nTimeSteps|dumpFreq" data
    cp "$BASE/build/mitgcmuv" .
    ./mitgcmuv > output.txt 2>&1 || true
    if grep -q "PROGRAM MAIN: Execution ended Normally" output.txt; then echo "OK $RUN"; \
    else echo "FAILED $RUN"; tail -20 output.txt; exit 1; fi ;;
  *) echo "usage: $0 build|run" >&2; exit 2 ;;
esac
```

- **Source tree:** `$MIT` is Esteban's MITgcm tree (`/Volumes/ESTEBAN_MIT/MITgcm`), the one his BayIW builds use. The case comes from `~/pyOCEAN_refs/MITgcm`. If the two trees differ in a way that breaks the build, build against `~/pyOCEAN_refs/MITgcm` instead and record a ruling.
- **Copies:** the data are copied, not linked, because `sed` edits `data`.

- [ ] **Step 5: Build, run, test**

Run:
```bash
chmod +x validation/internal_wave/run_mitgcm.sh
bash validation/internal_wave/run_mitgcm.sh build
bash validation/internal_wave/run_mitgcm.sh run
NUMBA_NUM_THREADS=5 python -m pytest tests/test_internal_wave.py -q
```
Expected:
- `mitgcmuv` listed;
- `OK …/run` in seconds;
- 4 tests passed.

Compare the MITgcm grid `Depth` with `case.mitgcm_depth(load_inputs()["H"], dz)` on columns 1 … 59 (to 1e-9 m); add that check to `test_read_mitgcm_shapes`.

- [ ] **Step 6: Commit**

```bash
git add validation/internal_wave/case.py validation/internal_wave/run_mitgcm.sh tests/test_internal_wave.py
git commit -m "internal_wave: inputs, obcs_calc boundary profiles, MITgcm build/run and reader (task 5)"
```

---

### Task 6: pyOCEAN runner and comparison

**Files:**
- Create: `validation/internal_wave/run_pyocean.py`, `validation/internal_wave/compare.py`
- Modify: `validation/internal_wave/case.py` (add `build_model`, `harmonic`, `energy_flux`)
- Test: `tests/test_internal_wave.py` (append)

**Interfaces:**
- **Consumes:** Task 5 `case.*`; the Task 3/4 model arguments; `bayiw.profile_targets`.
- **Produces:**
  - `case.build_model(coord: str, dt=500.0) -> (m, st)`, with the domain on MITgcm columns 2 … 60;
  - `case.to_levels(h, f, H, dz) -> (20, …)`, surface first, the PPM_H4 remap with the fixed levels and η in the top level (as `validation/bayiw_2d/section.to_mitgcm_levels`);
  - `case.harmonic(t, x, omega) -> (amp, phase_deg)` over the time axis 0;
  - `case.energy_flux(u, T, eta, H, dz, tref) -> (n, ncol)` [W m⁻¹], the vertically integrated baroclinic ⟨p′u′⟩ before time averaging;
  - `run_pyocean.py <coord> [--periods 10]` writes `RUNS/pyocean/<coord>/series.npz` with keys `t`, `U` (n, 20, 60 faces), `T` (n, 20, 59), `eta`;
  - `compare.py <coord>` prints and returns the tables.

- [ ] **Step 1: Write the failing tests** (append)

```python
def test_harmonic_recovers_amplitude_and_phase():
    omega = 2 * np.pi / case.PERIOD
    t = np.arange(0, 4 * case.PERIOD, 1800.0)
    x = 0.3 * np.sin(omega * t - np.radians(40.0))[:, None] * np.ones((1, 3)) + 0.01
    amp, ph = case.harmonic(t, x, omega)
    np.testing.assert_allclose(amp, 0.3, rtol=1e-6)
    np.testing.assert_allclose(ph, 40.0, atol=1e-4)


def test_energy_flux_of_a_mode1_wave():
    # mode-1 wave: u = U cos(πz/H) cos θ, p′ = ρ0 (ω/k) U cos(πz/H) cos θ (f = 0, hydrostatic) →
    # mean flux ∫⟨p′u′⟩dz = ρ0 (ω/k) U² H/4.  p′ is computed from T and η as for the models.
    H, dz, N = 200.0, np.full(20, 10.0), 1e-3
    omega = 2 * np.pi / case.PERIOD
    k = np.pi * omega / (N * H)
    zc = -(np.arange(20) + 0.5) * 10.0
    t = np.arange(0, case.PERIOD, case.PERIOD / 200)
    th = omega * t[:, None]
    U = 0.02
    u = U * np.cos(np.pi * zc / H)[None] * np.cos(th)
    Tz = N**2 / (9.81 * 2e-4)
    zeta = (U * k * H / (np.pi * omega)) * np.sin(np.pi * zc / H)[None] * np.cos(th)    # ζ_t = w, w_z = −u_x
    T = Tz * zc[None] - Tz * zeta
    F = case.energy_flux(u[..., None], T[..., None], np.zeros((t.size, 1)), np.array([H]), dz, Tz * zc)
    np.testing.assert_allclose(F.mean(0), case.RHO0 * omega / k * U**2 * H / 4, rtol=0.03)


def test_build_model_domain_starts_at_mitgcm_column_2():
    m, st = case.build_model("zstar")
    assert st.h.shape == (20, 1, 59) and np.isnan(m.H[0, -1]) and m.H[0, 0] == 200.0
```

- **Signs in the flux test:** the displacement follows w = ∂ζ/∂t and continuity u_x + w_z = 0, so that ⟨p′u′⟩ > 0 for a wave travelling east.
- **`energy_flux` itself:**
  - ρ′ = −ρ0 α (T − tref) and p′(z) = ρ0 g η + ∫_z^0 g ρ′ dz′, taken at the level centres by summing from the top;
  - remove the depth mean of p′ and of u;
  - F = Σ_k (u − ū)(p′ − p̄′) dz_k.

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_internal_wave.py -q -k "harmonic or energy or build_model"`
Expected: FAIL with `AttributeError`.

- [ ] **Step 3: Implement in `case.py`**

```python
RHO0 = 999.8                            # MITgcm default rhoNil (not set in data)


def harmonic(t: np.ndarray, x: np.ndarray, omega: float) -> tuple[np.ndarray, np.ndarray]:
    """Least-squares fit x ≈ a0 + A sin(ωt − φ) along axis 0: amplitude A and phase φ [deg] per point."""
    G_ = np.stack([np.ones_like(t), np.sin(omega * t), np.cos(omega * t)], 1)
    c = np.linalg.lstsq(G_, x.reshape(t.size, -1), rcond=None)[0]
    amp = np.hypot(c[1], c[2]).reshape(x.shape[1:])
    ph = np.degrees(np.arctan2(-c[2], c[1])).reshape(x.shape[1:])
    return amp, ph


def energy_flux(u, T, eta, H, dz, tref) -> np.ndarray:
    """
    Vertically integrated baroclinic energy flux Σ (u − ū)(p′ − p̄′) Δz [W m⁻¹] at each time and column, from
    level values u, T (n, nz, ncol) surface first, η (n, ncol) and the depth H (ncol,) (levels below the bed
    have NaN): ρ′ = −ρ0 α (T − Tref), p′ = ρ0 g η + g Σ_above ρ′ Δz + g ρ′ Δz/2 (linear EOS, hydrostatic).
    """
    dzz = np.where(np.isfinite(T), dz[None, :, None], 0.0)
    rho = np.nan_to_num(-RHO0 * ALPHA * (T - tref[None, :, None]))
    above = np.cumsum(rho * dzz, axis=1) - rho * dzz
    p = RHO0 * G * eta[:, None, :] + G * (above + 0.5 * rho * dzz)
    D = dzz.sum(axis=1)
    ub = np.nansum(np.nan_to_num(u) * dzz, axis=1) / np.where(D > 0, D, 1.0)
    pb = (p * dzz).sum(axis=1) / np.where(D > 0, D, 1.0)
    return ((np.nan_to_num(u) - ub[:, None]) * (p - pb[:, None]) * dzz).sum(axis=1)


def to_levels(h: np.ndarray, f: np.ndarray, H: np.ndarray, dz: np.ndarray = np.full(NZ, DZ)) -> np.ndarray:
    """pyOCEAN layers (N, …) bed first → the fixed MITgcm levels (20, …) surface first, η in the top level
    (PPM_H4, as validation/bayiw_2d/section.to_mitgcm_levels); NaN below the (partial-cell) bed."""
    from core.vertical import remap as rm
    Hc = np.nan_to_num(np.asarray(H, float))
    zf = np.concatenate([[0.0], np.cumsum(dz)]).reshape((-1,) + (1,) * Hc.ndim)
    top, bot = np.minimum(zf[:-1], Hc), np.minimum(zf[1:], Hc)
    h_lev = (bot - top)[::-1]
    tot, lev = h.sum(axis=0), h_lev.sum(axis=0)
    wet = (tot > 0) & (lev > 0)
    h_tgt = np.where(wet[None], h_lev, 0.0)
    h_tgt[-1] += np.where(wet, tot - lev, 0.0)
    out = rm.remap_columns(np.ascontiguousarray(h, float), np.ascontiguousarray(f, float),
                           np.ascontiguousarray(h_tgt), rm.PPM_H4, wet)
    return np.where(wet[None] & (h_lev > 0), out, np.nan)[::-1]


def boundary_state(dz: np.ndarray):
    """t → {'west': OBData}: the obcs_calc.F profiles on the 20 MITgcm levels at MITgcm's boundary face."""
    from boundaries.obcs import OBData
    zw = -np.concatenate([[0.0], np.cumsum(dz)])[:, None]

    def at(t: float):
        u, T = obcs_west(t, dz)
        return {"west": OBData(side="west", T=T[:, None], u_normal=u[:, None], zw=zw)}
    return at


def build_model(coord: str, dt: float = 500.0):
    """
    pyOCEAN on MITgcm's columns 2 … 60 (west face = MITgcm's prescribed u point), MITgcm's partial-cell
    depth, its viscosities and diffusivities, free slip, f = 0; west 'clamped' with the obcs_calc.F profiles,
    east a wall (MITgcm's eastern open boundary sits on land).  coord: zstar | hybrid | sigma (20 layers).
    """
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid_stretched
    from validation.bayiw_baroclinic import bayiw
    d = load_inputs()
    dx = d["dx"][1:]
    H = mitgcm_depth(d["H"][1:], d["dz"])[None, :]
    hg = create_cartesian_grid_stretched(dx, np.array([5e3]), x0=float(d["dx"][0]), y0=0.0, f0=0.0)
    tc = RHO0 * ALPHA
    if coord == "zstar":
        params = dict(dz_top=d["dz"].copy())
    elif coord == "hybrid":
        rho = RHO0 - tc * d["tref"]
        params = dict(dz_top=np.full(NZ, 1.0), rho_target=bayiw.profile_targets(rho, NZ, 0))
    elif coord == "sigma":
        params = {}
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, H, dt=dt, n_layers=NZ, rho0=RHO0, t_coef=tc, t_ref=0.0, z0=None,
                        closure="constant", nu_const=1e-3, ah=1e-2, kh=1e-2, kv=1e-3, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params,
                        open_boundaries=("west",), open_boundary_state=boundary_state(d["dz"]),
                        open_boundary_mode={"west": "clamped"}, mix_diagnostic=False)
    Hc = np.nan_to_num(H)
    zf = np.concatenate([[0.0], np.cumsum(d["dz"])])[:, None, None]
    h_src = (np.minimum(zf[1:], Hc[None]) - np.minimum(zf[:-1], Hc[None]))[::-1]
    T_src = np.broadcast_to(d["T"][:, None, 1:], h_src.shape)[::-1].copy()
    st = m.initial_state(np.zeros(H.shape), T_src, h_src=h_src)
    return m, st
```

- `create_cartesian_grid_stretched(dx, dy, x0, y0, f0)` is the signature the BayIW benchmark uses. If `dy` must have length 1 for one row, pass `np.array([5e3])` (MITgcm `delY` = 5 km).
- MITgcm's `Tref` is horizontally uniform, so `t_ref = 0` with `rho_ref = "auto"` gives the same PGF.

- [ ] **Step 4: Write `run_pyocean.py` and `compare.py`**

`run_pyocean.py`:
- for `coord` in zstar, hybrid and sigma, build the model and step `periods × PERIOD / dt` steps;
- every 1800 s, record:
  - `t`;
  - U on the levels, with `case.to_levels(hu_faces, st.u, Hf)`; the face layers come from `m._mean_face_thickness(st.h)[0]`, and the face bed is the shallower side's depth (as `validation/bayiw_2d/run_pyocean.py:record`);
  - T on the levels, with `case.to_levels(st.h, st.T, m.H)`;
  - `eta`;
- save to `case.RUNS / "pyocean" / coord / "series.npz"`;
- print the elapsed time and `st.n_trunc`.

`compare.py <coord>`:
- **Load:** `case.read_mitgcm()` and drop MITgcm's column 1 (its boundary cell) and its first u face, so that MITgcm and pyOCEAN share the faces 2 … 61 and the columns 2 … 60. Also load the pyOCEAN series. Both are recorded at the same 1800-s marks: align with `np.interp` on t where needed.
- **Window:** the last 4 periods.
- **Report:**
  1. u at ω at three depths (−15, −95, −175 m) along x: amplitude MIT/py, phase difference, and PASS/FAIL on the flat part (x < 2.5 km: ±10 % and ±10°);
  2. T′ at ω, the same;
  3. relative rms error of the u and T′ harmonic fields over the whole section;
  4. mean energy flux `case.energy_flux(...).mean(0)` at x ≈ 2, 6 and 10 km: MIT/py ratio and PASS/FAIL ±15 %;
  5. the 2ω amplitude of u near the bed in the shallow part (x > 9 km): MIT/py;
  6. health: `n_trunc`, max|Σh − H − η| and the heat change.

- [ ] **Step 5: Run the tests and a short smoke run**

Run:
```bash
NUMBA_NUM_THREADS=5 python -m pytest tests/test_internal_wave.py -q
NUMBA_NUM_THREADS=5 python validation/internal_wave/run_pyocean.py zstar --periods 1
```
Expected: the tests pass; one period runs in seconds with trunc 0.

- [ ] **Step 6: Commit**

```bash
git add validation/internal_wave/ tests/test_internal_wave.py
git commit -m "internal_wave: pyOCEAN model, runner and comparison (task 6)"
```

---

### Task 7: Runs, classification and README

**Files:**
- Create: `validation/internal_wave/README.md`

- [ ] **Step 1: All pyOCEAN runs** (sequentially)

```bash
for c in zstar hybrid sigma; do NUMBA_NUM_THREADS=5 python validation/internal_wave/run_pyocean.py $c || exit 1; done
for c in zstar hybrid sigma; do python validation/internal_wave/compare.py $c | tee /Volumes/ESTEBAN_MIT/pyOCEAN_runs/internal_wave/compare_$c.txt; done
```
Expected: three runs with trunc 0 and three tables.

- [ ] **Step 2: Classify and write the README** (Spanish)

The README covers:
- the purpose;
- the configuration, including the column 1 ↔ west face mapping, the partial cells and Flather vs MITgcm's clamped total transport (class (c));
- the analytic test result (Task 4);
- the tables;
- the (a)–(d) classification of every FAIL;
- a "Qué debe mejorar pyOCEAN" section.

On a class (a) FAIL: stop, diagnose with superpowers:systematic-debugging and report to Esteban.

- [ ] **Step 3: Commit**

```bash
git add validation/internal_wave/README.md
git commit -m "internal_wave: results vs MITgcm (task 7)"
```

---

### Task 8: Scorecard and documentation

**Files:**
- Modify: `docs/design/vision_roadmap.md` §7, `CLAUDE.md` (ALE row: time-varying open boundaries; test counts), and the memory `project_status.md`.

- [ ] **Step 1: Write the scorecard rows**

Two rows:
- "Onda interna modo 1 analítica (canal plano, oeste impuesto, este radiante)": the amplitude and phase errors from Task 4;
- "internal_wave de MITgcm": the flat-part amplitude and phase and the energy fluxes, for z\*, hybrid and σ.

- [ ] **Step 2: Full suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider`
Expected: all pass, plus the existing xfails.

- [ ] **Step 3: Commit**

```bash
git add docs/design/vision_roadmap.md CLAUDE.md docs/superpowers/plans/2026-10-12-baroclinic-obc-internal-wave.md
git commit -m "Baroclinic open boundaries + internal_wave: scorecard and docs (task 8)"
```
