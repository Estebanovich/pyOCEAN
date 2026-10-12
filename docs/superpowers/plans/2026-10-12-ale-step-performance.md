# ALE Step Performance Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ALE step of `SigmaBaroclinic` at least 5× faster on BayIW (560 × 352 × 50 z\*, ≤ 1.5 s/step from 8.6 s/step) without changing results beyond round-off.

**Architecture:**
- Profile-driven `numba.njit(parallel=True, cache=True)` kernels replace the hot NumPy code.
- The current NumPy functions stay as `*_numpy` oracles that the tests compare against.
- The free surface switches to CG with a lagged-LU preconditioner in the ALE mode, which already exists in `core/dynamics/wetting_drying.py`.
- After the first wave, the step is profiled again and the second wave is planned from the measurement.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba (installed), SciPy sparse, pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-ale-step-performance-design.md`

## Global Constraints

- **Threads:** at most 5 (`NUMBA_NUM_THREADS=5`). The model calls `numba.set_num_threads(_THREADS[0])` before parallel kernels. Never the efficiency cores. One heavy job at a time.
- **Precision:** float64, no `fastmath`, so kernels can be compared with their NumPy oracle.
- **σ mode:** must stay bit-identical; it does not use the ALE kernels, and its `solver` default stays `"direct"`.
- **ALE results:** must stay equal to round-off. Model-level regression ≤ 1e-10 relative after 100 steps with `solver="direct"`. With CG the difference is bounded by `cg_rtol` = 1e-10 and is documented.
- **Dependencies:** none new; numpy stays 1.26.4.
- **Commits:** at the end of each task. Esteban asked for commits per phase in this work; each task here is one phase. Messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn
  ```
  Never commit `*.npz`, `*.npy`, `log_*.txt` or `pyOCEAN-literature-review-guide.docx`.
- **Docstrings:** every kernel cites the NumPy function it mirrors and the source (MOM6 or HYCOM `file:line`) that function already cites.

## Review Focus

1. **Faces at the domain edges** (faces 0 and nx use the inner cell twice) **and faces next to land**: the kernels must reproduce the padding and masking of the NumPy code exactly. *Pinned by Task 3/4 oracle tests with land columns and edge faces.*
2. **Vanished layers and the bed clip** (q < h_up caps the swept thickness): equal to the oracle. *Pinned by Task 4 `test_ppm_kernel_matches_numpy_with_clip_and_vanished_layers`.*
3. **Newton convergence identical to the NumPy global loop**: NumPy stops when *all* faces have converged, so a per-column early stop would change results at 1e-13. The kernel keeps the global iteration count. *Pinned by Task 4 oracle tests at rtol 1e-12.*
4. **The identity cache of the transport thickness returns a stale value if an array is mutated in place**: the cache must only be used for arrays that are never written after creation. *Pinned by Task 3 `test_transport_thickness_cache_is_keyed_on_identity_and_recomputes_new_arrays`.*
5. **The CG free surface on a dry or very shallow column, or with open boundaries**: must converge, or raise the existing clear error. *Pinned by Task 2 tests on the open-boundary box and the bay-mouth step.*

---

## File Structure

- Create: `scripts/profile_ale_step.py` (timing and cProfile of the ALE step: BayIW or a small box).
- Modify:
  - `core/vertical/ale.py`: Numba kernels for `face_transport_thickness` and `ppm_transport_x`; NumPy versions kept as `face_transport_thickness_numpy`, `ppm_edges_x` (unchanged, used by tests), `ppm_transport_x_numpy`;
  - `core/dynamics/sigma_baroclinic.py`: `solver` default per mode, identity cache in `_transport_thickness`, thread setting at the top of `_step_ale`;
  - `tests/test_vertical_ale.py`, `tests/test_sigma_ale.py`: oracle and regression tests;
  - `CLAUDE.md`, `validation/bayiw_baroclinic/README.md` (created if missing), memory: performance table.

---

### Task 1: Baseline (profile script and model-level references)

**Files:**
- Create: `scripts/profile_ale_step.py`
- Workspace (git-ignored): `.superpowers/sdd/2026-10-12-ale-step-performance/ref_*.npz`

**Interfaces:**
- Produces:
  - `python scripts/profile_ale_step.py --case bayiw|box [--coord zstar|hybrid] [--steps 5] [--solver direct|cg] [--profile]`, which prints `s/step` and, with `--profile`, the cProfile tables;
  - reference states for the regression of Tasks 2–5.

- [ ] **Step 1: Write the profile script**

```python
#!/usr/bin/env python
"""
Timing and profile of the ALE step of SigmaBaroclinic (spec docs/superpowers/specs/2026-10-12-ale-step-performance-design.md).

--case bayiw: BayIW two-layer with the bay, 560 × 352 (validation/bayiw_baroclinic/bayiw.py, needs the external volume);
--case box: 200 × 120 × 30 z* box with open boundaries and a sponge (no external data).
Usage: NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw [--coord zstar] [--steps 5] [--solver cg] [--profile]
"""
from __future__ import annotations

import argparse
import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402


def box_model(coord: str, solver: str | None):
    """200 × 120 × 30 box, 400 m deep with a 100 m shelf, open W/E/S edges, 10-cell sponge, linear stratification."""
    nx, ny, nz = 200, 120, 30
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=2000.0, dy=2000.0, f0=1e-4)
    H = np.where(np.arange(ny)[:, None] > 90, 100.0, 400.0) * np.ones((1, nx))
    kw = dict(regrid_params=dict(dz_top=np.full(nz, 400.0 / nz)))
    if coord == "hybrid":
        rho = 1000.0 - 0.2 * (10.0 + 0.02 * np.linspace(-400.0, 0.0, 200))
        kw = dict(regrid_params=dict(dz_top=np.full(nz, 1.0), rho_target=np.linspace(rho.min() - 0.05, rho.max() + 0.05,
                                                                                     nz + 1)))
    if solver:
        kw["solver"] = solver
    m = SigmaBaroclinic(hg, H, dt=120.0, n_layers=nz, rho0=1000.0, t_coef=0.2, z0=None, closure="constant",
                        nu_const=1e-5, ah=10.0, kh=10.0, kv=1e-5, pgf="fv", rho_ref="auto", vertical_coordinate="ale",
                        regrid=coord, open_boundaries=("west", "east", "south"), sponge_cells=10,
                        sponge_tau=(1200.0, 120.0), **kw)
    z = m.layer_heights(np.zeros((ny, nx)))
    x = (np.arange(nx) + 0.5) * 2000.0
    T = 10.0 + 0.02 * (z + 5.0 * np.sin(-np.pi * z / 400.0) * np.exp(-((x - 200e3) / 20e3) ** 2)[None, None, :])
    return m, m.initial_state(np.zeros((ny, nx)), T), None


def bayiw_model(coord: str, solver: str | None):
    from validation.bayiw_baroclinic import bayiw
    m, st, wind = bayiw.build_model("2layer", "bay", coord, 120.0)
    if solver:
        m.core.fs.solver = solver
    return m, st, wind


def run(m, st, wind, n: int, t0: int):
    tau = None
    if wind is not None:
        from validation.bayiw_baroclinic import bayiw
        from validation.bayiw_seiche.run_pyocean import wind_at
        tau = np.zeros((bayiw.NY + 1, bayiw.NX))
    for k in range(n):
        if tau is not None:
            tau[:-1, :] = wind_at((t0 + k) * m.dt, wind)
        st = m.step(st, tau_y=tau)
    return st


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--case", choices=("bayiw", "box"), default="box")
    ap.add_argument("--coord", choices=("zstar", "hybrid"), default="zstar")
    ap.add_argument("--steps", type=int, default=5)
    ap.add_argument("--solver", choices=("direct", "cg"), default=None)
    ap.add_argument("--profile", action="store_true")
    a = ap.parse_args()
    set_threads(5)
    m, st, wind = (bayiw_model if a.case == "bayiw" else box_model)(a.coord, a.solver)
    st = run(m, st, wind, 3, 0)                                              # warm-up (JIT, LU)
    tic = time.perf_counter()
    st = run(m, st, wind, a.steps, 3)
    print(f"{a.case} {a.coord} solver={m.core.fs.solver}: {(time.perf_counter() - tic) / a.steps:.3f} s/step")
    if a.profile:
        pr = cProfile.Profile()
        pr.enable()
        run(m, st, wind, a.steps, 3 + a.steps)
        pr.disable()
        pstats.Stats(pr).sort_stats("tottime").print_stats(25)


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Measure the baseline**

Run:
```bash
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case box
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw --coord hybrid
```
Expected: three s/step lines. BayIW z\* should be ≈8.6 s/step. Write all three numbers in the ledger as `Baseline: …`.

- [ ] **Step 3: Save the model-level references** (before touching any model code)

```bash
NUMBA_NUM_THREADS=5 python - <<'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np
from core.dynamics.sigma_baroclinic import set_threads
import tests.test_sigma_ale as t
set_threads(5)
W = ".superpowers/sdd/2026-10-12-ale-step-performance/"
cases = {
    "lock": lambda: t._hybrid_lock(128, 21, 20.0, 20.0, solver="direct"),
    "channel": lambda: t._ale_channel("hybrid", 60, 30e3, open_boundaries=("west", "east"), sponge_cells=10,
                                      sponge_tau=(1200.0, 120.0), solver="direct"),
    "step": lambda: t._step_section("zstar", solver="direct"),
    "openbox": lambda: t._ale_open_box("hybrid", nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4,
                                       sponge_cells=3, solver="direct"),
}
for name, build in cases.items():
    m, st = build()
    for _ in range(100):
        st = m.step(st)
    np.savez(W + f"ref_{name}.npz", u=st.u, v=st.v, T=st.T, h=st.h, eta=st.eta)
    print(name, "saved", float(np.abs(st.u).max()))
EOF
```
Expected: four `saved` lines.

If one helper does not accept `solver=` (for example `_hybrid_lock` forwards `**kw` to `_ale_box`, and `_step_section` forwards `**extra` only for z\*), adapt the call so `solver="direct"` reaches `SigmaBaroclinic`, and note it as a ruling.

- [ ] **Step 4: Write the regression helper** (append to `tests/test_sigma_ale.py`; skipped when the references are missing, so CI without the workspace is unaffected)

```python
# ---------------------------------------------------------------- performance regression (spec 2026-10-12-ale-step-performance §4)
_PERF_REF = Path(__file__).resolve().parents[1] / ".superpowers/sdd/2026-10-12-ale-step-performance"


def _perf_cases():
    return {
        "lock": lambda: _hybrid_lock(128, 21, 20.0, 20.0, solver="direct"),
        "channel": lambda: _ale_channel("hybrid", 60, 30e3, open_boundaries=("west", "east"), sponge_cells=10,
                                        sponge_tau=(1200.0, 120.0), solver="direct"),
        "step": lambda: _step_section("zstar", solver="direct"),
        "openbox": lambda: _ale_open_box("hybrid", nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4,
                                         sponge_cells=3, solver="direct"),
    }


@pytest.mark.parametrize("name", ["lock", "channel", "step", "openbox"])
def test_ale_results_unchanged_by_the_performance_kernels(name):
    ref = _PERF_REF / f"ref_{name}.npz"
    if not ref.exists():
        pytest.skip("performance references not generated (plan 2026-10-12-ale-step-performance, Task 1)")
    r = np.load(ref)
    m, st = _perf_cases()[name]()
    for _ in range(100):
        st = m.step(st)
    for k in ("u", "v", "T", "h", "eta"):
        a, b = getattr(st, k), r[k]
        scale = max(float(np.abs(b).max()), 1e-30)
        assert float(np.abs(a - b).max()) <= 1e-10 * scale, (k, float(np.abs(a - b).max()), scale)
```

Add `from pathlib import Path` to the imports of `tests/test_sigma_ale.py` if it is not there.

- [ ] **Step 5: Run it** (must pass against the unchanged code)

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k performance_kernels`
Expected: 4 passed (identical code, so the differences are exactly 0).

- [ ] **Step 6: Commit**

```bash
git add scripts/profile_ale_step.py tests/test_sigma_ale.py
git commit -m "ALE performance: profile script and model-level regression references (task 1)"
```

---

### Task 2: Free surface — CG with the lagged-LU preconditioner by default in ALE

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (`__init__` signature and `SigmaWetDry(...)` call; class docstring)
- Test: `tests/test_sigma_ale.py` (append)

**Interfaces:**
- Produces: `SigmaBaroclinic(..., solver: str | None = None, ...)`, where None means `"cg"` if `vertical_coordinate == "ale"` and `"direct"` otherwise.

- [ ] **Step 1: Write the failing tests**

```python
def test_ale_uses_the_lagged_lu_cg_free_surface_by_default():
    assert _ale_box("zstar").core.fs.solver == "cg"
    sig = SigmaBaroclinic(create_cartesian_grid(nx=4, ny=1, dx=1e3, dy=1e3, f0=0.0), np.full((1, 4), 20.0),
                          dt=10.0, n_layers=4)
    assert sig.core.fs.solver == "direct"                                # σ stays bit-identical


@pytest.mark.parametrize("build", ["openbox", "step"])
def test_cg_free_surface_matches_the_direct_solve(build):
    # review focus 5: open boundaries (Flather) and a 600 → 164 m step
    def make(solver):
        if build == "openbox":
            return _ale_open_box("hybrid", nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4, sponge_cells=3,
                                 solver=solver)
        return _step_section("zstar", solver=solver)
    (m1, s1), (m2, s2) = make("direct"), make("cg")
    for _ in range(50):
        s1, s2 = m1.step(s1), m2.step(s2)
    assert np.abs(s1.eta - s2.eta).max() <= 1e-9
    assert np.abs(s1.u - s2.u).max() <= 1e-9 * max(np.abs(s1.u).max(), 1e-3)
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k "lagged_lu or cg_free_surface"`
Expected: the default test FAILS (`'direct' == 'cg'`); the two equality tests pass, because the CG path already exists.

- [ ] **Step 3: Implement**

In `SigmaBaroclinic.__init__`:
- change `solver: str = "direct"` to `solver: str | None = None`;
- before building `self.core`, add:

```python
        if solver is None:                     # ALE: CG with the lagged-LU preconditioner (1–3 iterations; the
            solver = "cg" if vertical_coordinate == "ale" else "direct"   # σ mode keeps the direct solve, bit-identical)
```

In the class docstring, document `solver`:
```
    solver : free-surface solver: 'direct' (σ default) or 'cg' (ALE default: CG preconditioned with the LU of a
          lagged matrix, core.dynamics.wetting_drying; ≈1 s → ≈0.1 s per step on BayIW 560 × 352)
```

- [ ] **Step 4: Run the tests and the regression**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -p no:cacheprovider`
Expected:
- all pass, including the 4 regression tests (they force `solver="direct"`) and the existing tests (now on CG);
- the 1 xfail stays.

If an existing ALE test fails on CG, compare it with `solver="direct"` to see whether the difference is within the CG tolerance. Fix it, or record a ruling.

- [ ] **Step 5: Measure**

Run: `NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw`
Expected: s/step drops by ≈0.8–0.9 s. Record the number in the ledger.

- [ ] **Step 6: Commit**

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py
git commit -m "ALE: CG free surface with the lagged-LU preconditioner by default (task 2)"
```

---

### Task 3: Transport thickness kernel and identity cache

**Files:**
- Modify:
  - `core/vertical/ale.py`: add `import numba` at the top; rename the current function to `face_transport_thickness_numpy`; new kernel `_ftt_kernel` and wrapper `face_transport_thickness`;
  - `core/dynamics/sigma_baroclinic.py`: `_transport_thickness` cache; thread setting at the top of `_step_ale`.
- Test: `tests/test_vertical_ale.py`, `tests/test_sigma_ale.py` (append)

**Interfaces:**
- Produces: `face_transport_thickness(h, H, active, clip) -> (q_e, q_w)`, with the same signature, shapes and values as before (≤ 1e-14); `face_transport_thickness_numpy` (the oracle).

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vertical_ale.py`:

```python
@pytest.mark.parametrize("clip", [True, False])
def test_transport_thickness_kernel_matches_numpy(clip):
    # review focus 1: domain-edge faces, land columns, vanished layers, a step in the bed
    rng = np.random.default_rng(31)
    n, ny, nx = 7, 5, 11
    H = rng.uniform(20.0, 600.0, (ny, nx))
    H[:, 4] = 150.0
    active = rng.uniform(size=(ny, nx)) > 0.15
    frac = rng.uniform(0.0, 1.0, (n, ny, nx))
    frac[:2] *= rng.uniform(size=(2, ny, nx)) > 0.5                        # vanished bed layers
    h = frac / frac.sum(0) * H[None]
    a = ale.face_transport_thickness(h, H, active, clip)
    b = ale.face_transport_thickness_numpy(h, H, active, clip)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-14, atol=1e-14)
```

Append to `tests/test_sigma_ale.py`:

```python
def test_transport_thickness_cache_is_keyed_on_identity_and_recomputes_new_arrays():
    # review focus 4: the cache returns the stored result only for the same array object
    m, st = _ale_open_box("zstar")
    a = m._transport_thickness(st.h)
    assert m._transport_thickness(st.h) is a
    h2 = st.h.copy()
    h2[0] *= 0.5
    b = m._transport_thickness(h2)
    assert b is not a and not np.allclose(b[0][0], a[0][0])
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_ale.py tests/test_sigma_ale.py -q -k "transport_thickness_kernel or transport_thickness_cache"`
Expected: FAIL. The kernel test fails with `AttributeError: … face_transport_thickness_numpy`; the cache test fails because the second call returns a new tuple (`is` is False).

- [ ] **Step 3: Implement the kernel** (`core/vertical/ale.py`)

Rename the existing `def face_transport_thickness(` to `def face_transport_thickness_numpy(` (body unchanged, docstring first line: "NumPy reference of face_transport_thickness (test oracle)."). Then add:

```python
@numba.njit(parallel=True, cache=True)
def _ftt_kernel(h, H, active, clip, q_e, q_w):  # pragma: no cover
    """Kernel of face_transport_thickness: one face column per (j, f), same operations as the NumPy version."""
    n, ny, nx = h.shape
    for j in numba.prange(ny):
        for f in range(nx + 1):
            iw = f - 1 if f > 0 else 0
            ie = f if f < nx else nx - 1
            if not (active[j, iw] and active[j, ie]):
                for k in range(n):
                    q_e[k, j, f] = 0.0
                    q_w[k, j, f] = 0.0
                continue
            if clip:
                z_bed = max(-H[j, iw], -H[j, ie])
                cw = 0.0
                ce = 0.0
                for k in range(n):
                    cw += h[k, j, iw]
                    ce += h[k, j, ie]
                    q_e[k, j, f] = min(h[k, j, iw], max(0.0, (-H[j, iw] + cw) - z_bed))
                    q_w[k, j, f] = min(h[k, j, ie], max(0.0, (-H[j, ie] + ce) - z_bed))
            else:
                for k in range(n):
                    q_e[k, j, f] = h[k, j, iw]
                    q_w[k, j, f] = h[k, j, ie]


def face_transport_thickness(h: np.ndarray, H: np.ndarray, active: np.ndarray,
                             clip: bool) -> tuple[np.ndarray, np.ndarray]:
    """
    Layer thickness available for transport through the u-faces [m]: q_E (eastward, from the
    west cell) and q_W (westward, from the east cell), each (N, ny, nx+1).  h (N, ny, nx) bed
    first, H (ny, nx) rest depth [m], active (ny, nx).  With ``clip`` (z*, hybrid) only the part of
    the upwind layer above the shallower bed passes (HYCOM cnuity.F90:252-256):

        q = min(h_k,up, max(0, z_top,k,up − max(−H_W, −H_E)))

    Without it (σ: the layers follow the terrain) q = h_k,up.  Domain-edge faces use the inner
    cell; faces next to land are 0.  For the v-faces call it on swapped axes.  Numba kernel
    (_ftt_kernel); face_transport_thickness_numpy is the reference.
    """
    hc = np.ascontiguousarray(np.where(active[None], np.nan_to_num(h), 0.0))
    n, ny, nx = hc.shape
    q_e, q_w = np.empty((n, ny, nx + 1)), np.empty((n, ny, nx + 1))
    _ftt_kernel(hc, np.ascontiguousarray(np.nan_to_num(np.asarray(H, float))), np.ascontiguousarray(active),
                bool(clip), q_e, q_w)
    return q_e, q_w
```

NumPy computes `cumsum` first and then adds `−H`. The kernel does the same (`-H + cw`), so the rounding order matches.

- [ ] **Step 4: Implement the cache and the thread setting** (`core/dynamics/sigma_baroclinic.py`)

Replace `_transport_thickness` with:

```python
    def _transport_thickness(self, h: np.ndarray):
        """
        Upwind face thicknesses ((q_E, q_W) at u-faces, (q_N, q_S) at v-faces swapped to (N, nx, ny+1)):
        clipped at the shallower bed for z* and hybrid, not for σ (core.vertical.ale).  Cached by array
        identity for the last two thickness arrays (hⁿ is used by the tendencies and the step; h_new of one
        step is hⁿ of the next); thickness arrays are never written in place after creation.
        """
        for ref, val in getattr(self, "_tt_cache", ()):
            if ref is h:
                return val
        clip = self.regrid_rule != "sigma"
        H, act = np.nan_to_num(np.asarray(self.H, float)), self.fs.active
        x = face_transport_thickness(h, H, act, clip)
        y = face_transport_thickness(h.swapaxes(1, 2), H.T, act.T, clip)
        self._tt_cache = ((h, (x, y)),) + tuple(getattr(self, "_tt_cache", ()))[:1]
        return x, y
```

At the top of `_step_ale`, right after `dt, th, n = …`, add:

```python
        numba.set_num_threads(_THREADS[0])                                 # parallel ALE kernels (≤ 5 P-cores)
```

Before relying on the cache, check that no code writes into `st.h`, `h_lag` or `h_new` in place: `grep -n "h\[.*\] *=\|h_lag\[\|h_new\[\|\.h\[" core/dynamics/sigma_baroclinic.py`. Any in-place write to an array that may be cached must be removed (create a new array instead), or the cache is wrong. Record what you found in the ledger.

- [ ] **Step 5: Run the tests, the regression and the timing**

Run:
```bash
NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_ale.py tests/test_sigma_ale.py -q -p no:cacheprovider
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw
```
Expected: all pass (the regression shows differences ≤ 1e-10 relative); BayIW drops by a further ≈0.7 s/step. Record it.

- [ ] **Step 6: Commit**

```bash
git add core/vertical/ale.py core/dynamics/sigma_baroclinic.py tests/test_vertical_ale.py tests/test_sigma_ale.py
git commit -m "ALE: Numba transport-thickness kernel with an identity cache (task 3)"
```

---

### Task 4: PPM thickness-flux kernel

**Files:**
- Modify: `core/vertical/ale.py`. Rename `ppm_transport_x` to `ppm_transport_x_numpy` (oracle, unchanged); add `_ppm_edges_kernel`, `_face_flux`, `_ppm_transport_kernel` and the new wrapper `ppm_transport_x` with the same signature.
- Test: `tests/test_vertical_ale.py` (append)

**Interfaces:**
- Produces: `ppm_transport_x(h, ut, L, Q_bt, q_e, q_w, active, dt, dx_t, max_iter=20) -> np.ndarray (N, ny, nx+1)`, same signature and the same values as `ppm_transport_x_numpy` to rtol 1e-12.

- [ ] **Step 1: Write the failing tests**

```python
def _ppm_random(seed, n=6, ny=4, nx=9, clip=False, land=False, vanish=False):
    rng = np.random.default_rng(seed)
    H = rng.uniform(50.0, 300.0, (ny, nx)) if clip else np.full((ny, nx), 100.0)
    active = (rng.uniform(size=(ny, nx)) > 0.2) if land else np.ones((ny, nx), bool)
    frac = rng.uniform(0.05, 1.0, (n, ny, nx))
    if vanish:
        frac[rng.uniform(size=frac.shape) < 0.3] = 0.0
        frac[:, 0, 0] = 0.0
        frac[2, 0, 0] = 1.0
    h = frac / frac.sum(0) * H[None]
    ut = rng.normal(0.0, 0.3, (n, ny, nx + 1))
    qe, qw = ale.face_transport_thickness_numpy(h, H, active, clip)
    dx_t = rng.uniform(800.0, 1200.0, (ny, nx))
    L = rng.uniform(800.0, 1200.0, (ny, nx + 1))
    F0, _ = ale.ppm_fluxes_x(h, ut, L, qe, qw, active, 60.0, dx_t)
    Qbt = F0.sum(0) + rng.normal(0.0, 50.0, (ny, nx + 1))                 # needs a Newton correction
    return h, ut, L, Qbt, qe, qw, active, dx_t


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ppm_kernel_matches_numpy(seed):
    args = _ppm_random(seed)
    a = ale.ppm_transport_x(*args[:7], 60.0, args[7])
    b = ale.ppm_transport_x_numpy(*args[:7], 60.0, args[7])
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("seed", [4, 5])
def test_ppm_kernel_matches_numpy_with_clip_and_vanished_layers(seed):
    # review focus 1–3: bed clip, vanished layers, land, edge faces; Newton with the global stop of the NumPy loop
    args = _ppm_random(seed, clip=True, land=True, vanish=True)
    a = ale.ppm_transport_x(*args[:7], 60.0, args[7])
    b = ale.ppm_transport_x_numpy(*args[:7], 60.0, args[7])
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)
    assert np.isfinite(a).all()
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_ale.py -q -k ppm_kernel`
Expected: FAIL with `AttributeError: module 'core.vertical.ale' has no attribute 'ppm_transport_x_numpy'`.

- [ ] **Step 3: Implement** (`core/vertical/ale.py`)

Rename `def ppm_transport_x(` to `def ppm_transport_x_numpy(` (body unchanged; docstring first line: "NumPy reference of ppm_transport_x (test oracle)."). Then add:

```python
@numba.njit(parallel=True, cache=True)
def _ppm_edges_kernel(h, active, h_min, h_w, h_e):  # pragma: no cover
    """Kernel of ppm_edges_x (same operations, MOM_continuity_PPM.F90:2827-2942 and 3184-3226)."""
    n, ny, nx = h.shape
    for j in numba.prange(ny):
        slp = np.empty(nx)
        for k in range(n):
            for i in range(nx):
                m = 1.0 if active[j, i] else 0.0
                mw = (1.0 if active[j, i - 1] else 0.0) if i > 0 else 0.0
                me = (1.0 if active[j, i + 1] else 0.0) if i < nx - 1 else 0.0
                hc = h[k, j, i]
                hm1 = mw * h[k, j, i - 1] + (1.0 - mw) * hc if i > 0 else hc
                hp1 = me * h[k, j, i + 1] + (1.0 - me) * hc if i < nx - 1 else hc
                s = 0.5 * (hp1 - hm1)
                dmx = max(max(hp1, hm1), hc) - hc
                dmn = hc - min(min(hp1, hm1), hc)
                a = min(abs(s), 2.0 * min(dmx, dmn))
                s = a if s > 0.0 else (-a if s < 0.0 else 0.0)
                slp[i] = s if mw * me * m > 0.0 else 0.0
            for i in range(nx):
                mw = (1.0 if active[j, i - 1] else 0.0) if i > 0 else 0.0
                me = (1.0 if active[j, i + 1] else 0.0) if i < nx - 1 else 0.0
                hc = h[k, j, i]
                hm1 = mw * h[k, j, i - 1] + (1.0 - mw) * hc if i > 0 else hc
                hp1 = me * h[k, j, i + 1] + (1.0 - me) * hc if i < nx - 1 else hc
                sw = slp[i - 1] if i > 0 else 0.0
                se = slp[i + 1] if i < nx - 1 else 0.0
                hw = 0.5 * (hm1 + hc) + (sw - slp[i]) / 6.0
                he = 0.5 * (hp1 + hc) + (slp[i] - se) / 6.0
                curv = 3.0 * ((hw + he) - 2.0 * hc)
                dh = he - hw
                inside = curv > 0.0 and abs(dh) < curv
                if inside and hc <= h_min:
                    h_w[k, j, i] = hc
                    h_e[k, j, i] = hc
                    continue
                sc = 1.0
                if inside:
                    num = 12.0 * curv * (hc - h_min)
                    den = curv * curv + 3.0 * dh * dh
                    if num < den:
                        sc = num / den
                h_w[k, j, i] = hc + sc * (hw - hc)
                h_e[k, j, i] = hc + sc * (he - hc)


@numba.njit(cache=True)
def _face_flux(k, j, f, u, h, h_w, h_e, dx_t, q_e, q_w, L_f, dt, nx):  # pragma: no cover
    """(F, dF/du) of layer k at u-face (j, f): _swept_thickness + _ppm_fluxes for one open face (MOM6 flux_elem)."""
    iw = f - 1 if f > 0 else 0
    ie = f if f < nx else nx - 1
    if u >= 0.0:
        c = min(abs(u) * dt / dx_t[j, iw], 1.0)
        hu, ww, ee = h[k, j, iw], h_w[k, j, iw], h_e[k, j, iw]
        curv3, dh, edge, q = (ww + ee) - 2.0 * hu, ww - ee, ee, q_e[k, j, f]
    else:
        c = min(abs(u) * dt / dx_t[j, ie], 1.0)
        hu, ww, ee = h[k, j, ie], h_w[k, j, ie], h_e[k, j, ie]
        curv3, dh, edge, q = (ww + ee) - 2.0 * hu, ee - ww, ww, q_w[k, j, f]
    mean = edge + c * (0.5 * dh + curv3 * (c - 1.5))
    marg = edge + c * (dh + 3.0 * curv3 * (c - 1.0))
    if q < hu:                                                         # bed clip (z*, hybrid)
        mean = min(mean, q)
        marg = min(marg, q)
    return L_f * u * max(mean, 0.0), L_f * max(marg, 0.0)


@numba.njit(parallel=True, cache=True)
def _ppm_transport_kernel(h, h_w, h_e, ut, L, Q_bt, q_e, q_w, active, dt, dx_t, max_iter, out):  # pragma: no cover
    """
    Kernel of ppm_transport_x_numpy: Newton with bracketing bisection on the uniform correction du of each face
    column (zonal_flux_adjust, MOM_continuity_PPM.F90:1368-1546), stopped when *all* faces have converged, as
    the NumPy loop (identical iteration count), then the HYCOM residual share (cnuity.F90:585-620).
    """
    n, ny, nx = h.shape
    nf = nx + 1
    du = np.zeros((ny, nf))
    lo = np.empty((ny, nf))
    hi = np.empty((ny, nf))
    err = np.empty((ny, nf))
    dsum = np.empty((ny, nf))
    for j in numba.prange(ny):
        for f in range(nf):
            iw = f - 1 if f > 0 else 0
            ie = f if f < nx else nx - 1
            limw, lime = 0.5 * dx_t[j, iw] / dt, 0.5 * dx_t[j, ie] / dt
            mnw, mxe = np.inf, -np.inf
            for k in range(n):
                mnw = min(mnw, limw - ut[k, j, f])
                mxe = max(mxe, -(lime + ut[k, j, f]))
            hi[j, f] = max(min(limw, mnw), 0.0)
            lo[j, f] = min(max(-lime, mxe), 0.0)
            s, sd = 0.0, 0.0
            if active[j, iw] and active[j, ie]:
                for k in range(n):
                    F, dF = _face_flux(k, j, f, ut[k, j, f], h, h_w, h_e, dx_t, q_e, q_w, L[j, f], dt, nx)
                    s += F
                    sd += dF
            err[j, f] = s - Q_bt[j, f]
            dsum[j, f] = sd
    for _ in range(max_iter):
        for j in numba.prange(ny):
            for f in range(nf):
                e, d = err[j, f], du[j, f]
                if e > 0.0:
                    hi[j, f] = d
                if e < 0.0:
                    lo[j, f] = d
                step = -e / dsum[j, f] if dsum[j, f] > 0.0 else 0.0
                new = d + step
                if step > 0.0 and new >= hi[j, f]:
                    new = 0.5 * (d + hi[j, f])
                if step < 0.0 and new <= lo[j, f]:
                    new = 0.5 * (d + lo[j, f])
                du[j, f] = new
                iw = f - 1 if f > 0 else 0
                ie = f if f < nx else nx - 1
                s, sd = 0.0, 0.0
                if active[j, iw] and active[j, ie]:
                    for k in range(n):
                        F, dF = _face_flux(k, j, f, ut[k, j, f] + new, h, h_w, h_e, dx_t, q_e, q_w, L[j, f], dt, nx)
                        s += F
                        sd += dF
                err[j, f] = s - Q_bt[j, f]
                dsum[j, f] = sd
        done = True
        for j in range(ny):
            for f in range(nf):
                if abs(err[j, f]) > 1e-12 * max(abs(Q_bt[j, f]), 1.0):
                    done = False
        if done:
            break
    for j in numba.prange(ny):
        Fk = np.empty(n)
        for f in range(nf):
            iw = f - 1 if f > 0 else 0
            ie = f if f < nx else nx - 1
            if not (active[j, iw] and active[j, ie]):
                for k in range(n):
                    out[k, j, f] = 0.0
                continue
            s = 0.0
            for k in range(n):
                Fk[k] = _face_flux(k, j, f, ut[k, j, f] + du[j, f], h, h_w, h_e, dx_t, q_e, q_w, L[j, f], dt, nx)[0]
                s += Fk[k]
            R = Q_bt[j, f] - s
            te, tw = 0.0, 0.0
            for k in range(n):
                te += q_e[k, j, f]
                tw += q_w[k, j, f]
            use_e = R >= 0.0
            tot = te if use_e else tw
            if tot <= 0.0:
                use_e = not use_e
                tot = te if use_e else tw
            den = tot if tot > 0.0 else 1.0
            for k in range(n):
                out[k, j, f] = Fk[k] + R * (q_e[k, j, f] if use_e else q_w[k, j, f]) / den


def ppm_transport_x(h: np.ndarray, ut: np.ndarray, L: np.ndarray, Q_bt: np.ndarray, q_e: np.ndarray,
                    q_w: np.ndarray, active: np.ndarray, dt: float, dx_t: np.ndarray,
                    max_iter: int = 20) -> np.ndarray:
    """
    Layer volume fluxes through the u-faces [m³ s⁻¹] with PPM thickness (MOM6 continuity_PPM,
    OM4 defaults) and a uniform velocity correction du per face column so that
    Σ_k L (u_k + du) h̄_k(u_k + du) = Q_bt (zonal_flux_adjust, MOM_continuity_PPM.F90:1368-1546:
    Newton with bracketing bisection; du keeps every layer under CFL ½, :773-874).  The
    remaining residual is shared like HYCOM (cnuity.F90:585-620) so Σ_k Q_k = Q_bt to
    round-off.  Only the fluxes change, not u (as MOM6).  For the v-faces call it on swapped axes.
    Numba kernels (_ppm_edges_kernel, _ppm_transport_kernel); ppm_transport_x_numpy is the reference.
    """
    hc = np.ascontiguousarray(np.where(active[None], np.nan_to_num(h), 0.0))
    act = np.ascontiguousarray(active)
    h_w, h_e = np.empty_like(hc), np.empty_like(hc)
    _ppm_edges_kernel(hc, act, H_PPM_MIN, h_w, h_e)
    out = np.empty(np.shape(ut))
    c = np.ascontiguousarray
    _ppm_transport_kernel(hc, h_w, h_e, c(ut, dtype=float), c(L, dtype=float), c(Q_bt, dtype=float),
                          c(q_e, dtype=float), c(q_w, dtype=float), act, float(dt), c(dx_t, dtype=float),
                          int(max_iter), out)
    return out
```

The swapped-axis calls for the v-faces (`h.swapaxes(1, 2)` and the rest) pass non-contiguous views. `np.ascontiguousarray` copies them, which is correct.

- [ ] **Step 4: Edge-value oracle check**

The kernel's edges must equal `ppm_edges_x`. Add this test and run it:

```python
def test_ppm_edges_kernel_matches_numpy():
    args = _ppm_random(7, clip=True, land=True, vanish=True)
    h, active = args[0], args[6]
    hc = np.where(active[None], h, 0.0)
    hw, he = np.empty_like(hc), np.empty_like(hc)
    ale._ppm_edges_kernel(np.ascontiguousarray(hc), np.ascontiguousarray(active), ale.H_PPM_MIN, hw, he)
    rw, re = ale.ppm_edges_x(hc, active)
    np.testing.assert_allclose(hw, rw, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(he, re, rtol=1e-14, atol=1e-14)
```

- [ ] **Step 5: Run the tests, the regression and the timing**

Run:
```bash
NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_ale.py tests/test_sigma_ale.py -q -p no:cacheprovider
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw --coord hybrid
```
Expected: all pass; BayIW z\* drops by a further ≈2 s/step. Record both numbers.

If an oracle test fails at 1e-12 only on faces where Newton bisects, inspect the iteration count: the NumPy stop is global, and the kernel's must be too. Do not loosen the tolerance without a ruling.

- [ ] **Step 6: Commit**

```bash
git add core/vertical/ale.py tests/test_vertical_ale.py
git commit -m "ALE: Numba PPM thickness-flux kernel (edges, swept mean, global Newton, HYCOM residual) (task 4)"
```

---

### Task 5: Second wave, from the new profile

**Files:** decided by the profile and written into this plan before any code (see Step 2).

- [ ] **Step 1: Profile**

Run: `NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw --profile > .superpowers/sdd/2026-10-12-ale-step-performance/profile_wave2.txt`. Read the `tottime` table. Group it by model function (`coriolis_pv`, `kinetic_energy`, `_laplacian`, `fv_pgf`, `zero_thin_bed_velocity`, `baroclinic_split`, `face_fractions`, `_diffuse_h`, `truncate_cfl`, `remap_columns`, `_rhs`, `lagrangian_thickness`, …). Compute each one's share of the step.

- [ ] **Step 2: Write the wave-2 tasks into this plan**

Append "### Task 5.k" sections for every function above 10 % of the step, most expensive first, in the format of Tasks 3–4:
- rename the NumPy function to `*_numpy` (oracle);
- write a parallel kernel with the same operations;
- add an oracle test (random cases with land, edge faces and vanished layers) at rtol ≤ 1e-12, which fails first;
- run the regression;
- measure;
- commit.

Write the actual kernel code into each section before implementing it. Stop adding sections when no function is above 10 % of the step, or when the next gain would be under 5 % of the total (spec §2).

- [ ] **Step 3: Implement the wave-2 tasks** exactly as written in Step 2, one commit each.

---

### Task 6: Results and documentation

**Files:**
- Modify: `CLAUDE.md` (ALE row: s/step on BayIW), memory `project_status.md`; `validation/bayiw_baroclinic/README.md` (create it with a "Rendimiento" section if Task 6 of the BayIW plan has not created it yet).

- [ ] **Step 1: Final measurements**

Run:
```bash
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw --coord hybrid
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case box
NUMBA_NUM_THREADS=5 python scripts/profile_ale_step.py --case bayiw --profile | head -45
```

- [ ] **Step 2: Write the table**

In Spanish, for the README and CLAUDE.md:
- a table of s/step before and after each task (z\*, hybrid, box) and the speed-up;
- the final profile;
- a comparison with MITgcm (0.29 s/step with 8 MPI processes at Δt = 30 s → s per simulated hour);
- if the 1.5 s/step goal was not reached, what dominates and what would come next (JAX/GPU, stage D; or MPI).

- [ ] **Step 3: Full suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider`
Expected: everything passes, plus the existing xfails.

- [ ] **Step 4: Commit**

```bash
git add CLAUDE.md validation/bayiw_baroclinic/README.md docs/superpowers/plans/2026-10-12-ale-step-performance.md
git commit -m "ALE performance: results (BayIW s/step before/after) and docs (task 6)"
```

---

## Wave 2 (written after the Task 5 profile, 2026-10-12)

Profile after Task 4 (BayIW z\*, cProfile 4.84 s/step):

| Piece | s/step | % |
|---|---|---|
| tracer advection (`_advect` + `_rhs`, WENO kernel) | 1.00 | 21 |
| Coriolis PV + KE (`coriolis_pv`, `kinetic_energy`, vorticity) | ≈0.45 | 9 |
| `remap_columns` | 0.35 | 7 |
| `zero_thin_bed_velocity` | 0.34 | 7 |
| `fv_pgf` | 0.29 | 6 |
| `_diffuse_h` | 0.29 | 6 |
| `face_fractions` + `baroclinic_split` | 0.28 | 6 |
| Laplacian (100 per-layer calls rebuilding the spacings from xarray) | 0.22 | 5 |

Only the advection is above 10 %, but each of the others would still save more than 5 % of the step (spec §2), so wave 2 takes them in this order.

### Task 5.1: Advection — skip faces with zero flux

In ALE every vertical flux is zero (Lagrangian step), and land and closed faces have Q = 0. The fused kernel `_advect_div` (`core/dynamics/sigma_baroclinic.py`) still computes WENO there. Change it as follows; the result is the same bit for bit, since q·v = 0 for finite v and the flux arrays start at 0.
- In the x loop, after `q = Qx[k, j, i]`, add `if q == 0.0: continue`.
- In the y loop, after `q = Qy[k, j, i]`, add `if q == 0.0: continue`.
- In the z loop, after `q = Fz[k, j, i]`, add `if q == 0.0: continue`.

**Test:** the existing kernel-vs-NumPy tests (`tests/test_sigma_baroclinic.py`), the σ bit-identity check (save σ channel u, T before the change and `np.array_equal` after, as in the Plan C task 2 script) and the ALE regression. It is a performance-only change, so there is no new behaviour to make fail. Record that as a ruling.

**Measure** BayIW. **Commit:** "ALE performance: skip zero-flux faces in the fused advection kernel (task 5.1)".

### Task 5.2: Laplacian — spacings cached, one 3-D call

`laplacian_u` and `laplacian_v` rebuild the spacings from xarray on each of the 2N calls per step.
1. Add `laplacian_3d(f, wet, hw, he, hs, hn)` to `core/dynamics/nonlinear.py`: a `numba.njit(parallel=True, cache=True)` kernel over (k, j) with the same per-element arithmetic, in the same order, as `_laplacian`:
   - x pair first, then y;
   - neighbour replaced by f where not wet or beyond the edge;
   - NaN spacings replaced by 1.0;
   - `out += 2.0 * ((hi − f)/h_hi − (f − lo)/h_lo)/(h_lo + h_hi)`;
   - 0 where not wet.
2. In `SigmaBaroclinic`, cache `self._lap_u = _spacings_u(self.hg)` and `self._lap_v = _spacings_v(self.hg)` once (lazily).
3. Replace the per-layer `laplacian_u/v` calls with `gu += self.ah * laplacian_3d(st.u, wet_u, *self._lap_u)` (same for v), outside the `for k` loop. The loop keeps the `advection_u/v` branch.

**Test (fails first):** `test_laplacian_3d_matches_the_per_layer_laplacian` in `tests/test_nonlinear.py`, with random u on a stretched grid with land, comparing `np.array_equal` against `np.stack([laplacian_u(u[k], hg, wet) for k])`. Then the σ bit-identity check (σ uses `ah` too) and the ALE regression.

**Measure. Commit.**

### Task 5.3: Vector-invariant terms in kernels

`coriolis_pv` (with `bound`), `kinetic_energy` and `pv_q`/`vorticity_q` become `numba.njit(parallel=True, cache=True)` kernels over (k, j) with the same arithmetic. The NumPy versions are renamed `*_numpy` and kept as oracles. Wrappers keep the public names and signatures.

**Tests (fail first):** in `tests/test_vector_invariant.py`, for random fields with a masked corner and both KE schemes, kernel == `*_numpy` at rtol 1e-14.

**Measure. Commit.**

### Task 5.4: Face fractions, baroclinic split, thin-bed rule in kernels

`face_fractions`, `baroclinic_split` and `zero_thin_bed_velocity` (`core/vertical/ale.py`) become one-pass column kernels; the NumPy versions are kept as `*_numpy` oracles.
- **Thin-bed rule:** keep the `cumprod(hf < h_thin)` run from the bed and the same sums.

**Tests (fail first):** kernel == `*_numpy` at rtol 1e-14 on random faces with all-thin, no-thin and dry columns. Then the ALE regression.

**Measure. Commit.**

After 5.4: profile again. If a piece still saves more than 5 % of the step (candidates: `fv_pgf`, `_diffuse_h`, `remap_columns` call count, NumPy arithmetic inside `_step_ale`), add Task 5.5+ in the same format; otherwise go to Task 6.

## Wave 3 (written after the Task 5.4 profile, 2026-10-12)

Profile after 5.4 (BayIW z\*, cProfile 3.66 s/step):

| Piece | s/step | % |
|---|---|---|
| advection (kernel + RK3 NumPy + copies) | 0.97 | 27 |
| `fv_pgf` (NumPy around the kernels) | 0.36 | 10 |
| `_diffuse_h` | 0.35 | 10 |
| `remap_columns` | 0.31 | 8 |
| `ascontiguousarray` copies | 0.23 | 6 |
| `_step_ale` NumPy | 0.21 | 6 |
| `cumsum` (layer heights) | 0.20 | 5 |

### Task 5.5: Advection without a vertical pass in ALE; optional mixing diagnostic

- **No vertical pass:** `_advect_div` gets a `vert` flag. With `vert=False` (all Fz = 0, the ALE Lagrangian step), `fz` is a (nt, 1, 1, 1) dummy and the z loop and the z terms are skipped. The x + 0.0 terms of the old code do not change a value, so σ (`vert=True`) is bit-identical. `_rhs` passes `vert = bool(np.any(Fz != 0.0))`.
- **Optional mixing diagnostic:** `SigmaBaroclinic(..., mix_diagnostic: bool = True)`. With False, the ALE step advects and remaps only T (no T² stack) and `mix_num` is None.
- **Tests (fail first):** `test_mixing_diagnostic_off_leaves_T_unchanged`, which compares T, h and u with and without the diagnostic under `np.array_equal` (an unknown keyword fails first). Then the σ bit-identity check and the ALE regression.
- **Runner:** the BayIW runner and the profile script get `--no-mix` (the profile reports both settings).

### Task 5.6: Layer-height cache and horizontal diffusion kernel

- **Layer heights:** `layer_heights(eta, h)` caches its result by the identity of `h` (the last two arrays, as `_transport_thickness`).
- **Diffusion:** `_diffuse_h` in ALE becomes one parallel kernel (x and y fluxes with the min face thickness and the same masks, then the divergence and the update). The NumPy version is kept as `_diffuse_h_numpy`.
- **Tests (fail first):** kernel == NumPy (`np.array_equal`) on a random ALE state with land and vanished layers; regression.

After 5.6: profile again, and stop if no piece would save more than 5 % of the step.
