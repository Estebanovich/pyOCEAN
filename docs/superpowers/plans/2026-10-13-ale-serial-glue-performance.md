# ALE step performance, part 2 — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cut the serial share of the ALE step (≈75 % of wall time with 5 threads) and the nested-exterior overhead, so the BayIW cut runs at ≤ 0.25 s/step (z\*), ≤ 0.13 (hybrid) and nested ≤ radiating + 5 %.

**Architecture:** Profile-guided, as plan `2026-10-12-ale-step-performance`: each serial piece becomes a `numba.njit(parallel=True, cache=True)` kernel (or is computed once), and the current NumPy code stays as the test oracle (`*_numpy`). The nested exterior is computed only in the columns that use it. Physics and schemes do not change.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba (installed), SciPy; pytest.

**Spec:** `docs/superpowers/specs/2026-10-13-ale-serial-glue-performance-design.md`

## Global Constraints

- ≤ 5 threads (`NUMBA_NUM_THREADS=5`); kernels call `numba.set_num_threads(_THREADS[0])` like the rest of the model. Never the efficiency cores; one heavy run at a time.
- float64, no `fastmath`.
- σ mode bit-identical (`np.array_equal`).
- Part 1 (exterior) bit-identical in the full model.
- Part 2: every kernel matches its NumPy oracle (identical when the arithmetic is the same, else ≤ 1e-12 relative); fingerprints `tests/data/ale_perf_reference.json` and `tests/data/sigma_obc_reference.json` kept — a change from summation order (≤ 1e-10 relative) is documented in the ledger before regenerating.
- The bit-for-bit restart tests (`tests/test_restart.py`) keep passing.
- No new dependencies. Large outputs to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_cut/`. Never `git add` a directory; never commit `log_*.txt` or `*.npz`.
- Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn`.

## Review Focus

1. **Exterior restricted to edge columns while `sponge_tracer=True`:** the tracer sponge needs T_ext in the whole band; the restriction must switch off. *Pinned in Task 2 (`test_exterior_tracer_band_kept_with_tracer_sponge`).*
2. **`truncate_cfl` first-face report:** the parallel kernel must return the same first counted (k, j, i) as `np.nonzero` order, which the error message prints. *Pinned in Task 3.*
3. **Vanished/dry cells in the RK3 combination:** `V = 0` cells keep C (the `safe` rule); NaN-free output on land. *Pinned in Task 5.*
4. **Open edges with `last_wet` False in the transport-velocity kernel:** the old velocity is zeroed there (`un = 0`), as now. *Pinned in Task 6.*
5. **Out-of-range heights in the reference-profile kernel:** constant beyond the table ends, as `np.interp`. *Pinned in Task 4.*

---

### Task 1: Measurement script and baseline

**Files:**
- Create: `validation/bayiw_cut/profile_step.py`
- Modify: `validation/bayiw_cut/README.md` (append a "Rendimiento" section)

**Interfaces:**
- Produces: `python validation/bayiw_cut/profile_step.py [case ...]` printing, for each of `zstar_radiate`, `hybrid_radiate`, `zstar_nest`: s/step with 1 and 5 threads, the 1→5 speed-up, and the 12 functions with the largest self time at 5 threads with their own speed-up.

- [ ] **Step 1: Write the script**

```python
#!/usr/bin/env python
"""
Step cost of the BayIW cut (spec 2026-10-13-ale-serial-glue-performance §5): s/step with 1 and 5 threads, the
1→5 speed-up (Amdahl: serial share) and the functions with the most self time, for z* and hybrid radiating and
z* nested.  30 steps after 5 warm-up steps (Numba compilation).

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_cut/profile_step.py [zstar_radiate hybrid_radiate zstar_nest]
"""
from __future__ import annotations

import cProfile
import pstats
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from validation.bayiw_cut import cut  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402

CASES = {"zstar_radiate": ("zstar", "radiate"), "hybrid_radiate": ("hybrid", "radiate"), "zstar_nest": ("zstar", "nest")}


def measure(coord: str, mode: str, threads: int, profile: bool = False):
    set_threads(threads)
    state = cut.nest_state("2layer", "bay") if mode == "nest" else None
    m, st, wind = cut.build_model("2layer", "bay", coord, boundary_state=state)
    tau = np.zeros((st.eta.shape[0] + 1, st.eta.shape[1]))
    for n in range(5):
        tau[:-1] = wind_at(n * m.dt, wind)
        st = m.step(st, tau_y=tau)
    pr = cProfile.Profile() if profile else None
    t0 = time.perf_counter()
    if pr:
        pr.enable()
    for n in range(5, 35):
        tau[:-1] = wind_at(n * m.dt, wind)
        st = m.step(st, tau_y=tau)
    if pr:
        pr.disable()
    return (time.perf_counter() - t0) / 30, pr


def self_times(pr) -> dict:
    return {(k[0].split("/")[-1], k[2]): v[2] / 30 for k, v in pstats.Stats(pr).stats.items()}


def main() -> None:
    for name in sys.argv[1:] or list(CASES):
        coord, mode = CASES[name]
        t1, p1 = measure(coord, mode, 1, profile=True)
        t5, p5 = measure(coord, mode, 5, profile=True)
        s1, s5 = self_times(p1), self_times(p5)
        print(f"\n{name}: {t1:.3f} s/step (1 thread), {t5:.3f} s/step (5 threads), speed-up {t1 / t5:.2f}")
        print("  function                         self ms (5 thr)  speed-up 1→5")
        for k, v in sorted(s5.items(), key=lambda kv: -kv[1])[:12]:
            print(f"  {k[1][:32]:32s} {1e3 * v:8.1f}        {s1.get(k, 0.0) / max(v, 1e-12):5.2f}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Run the baseline**

Run: `NUMBA_NUM_THREADS=5 python validation/bayiw_cut/profile_step.py > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_cut/profile_baseline.txt 2>&1; cat /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_cut/profile_baseline.txt | grep -v Warn`
Expected: z\* radiate ≈ 0.39 s/step (5 threads), speed-up ≈ 2; hybrid ≈ 0.20; z\* nest ≈ 0.52.

- [ ] **Step 3: README "Rendimiento" section with the baseline table** (Spanish; case, s/step 1 and 5 threads, speed-up, the five largest serial pieces).

- [ ] **Step 4: Commit**

```bash
git add validation/bayiw_cut/profile_step.py validation/bayiw_cut/README.md
git commit -m "BayIW cut: step profile script and baseline (perf part 2, task 1)"
```

---

### Task 2: Nested exterior only where it is used

**Files:**
- Modify: `core/dynamics/open_boundary.py` (`exterior_tracer`, new `_source_columns_at`)
- Modify: `core/dynamics/sigma_baroclinic.py` (`_exterior_T`, `_exterior`, setup of a column mask)
- Test: `tests/test_open_boundary_state.py`

**Interfaces:**
- Produces: `SigmaBaroclinic._ext_cols` (ny, nx) bool — the columns whose exterior T is used: the boundary cells of the open edges, plus `_bc_mask` (band) when `sponge_tracer` is True. `exterior_tracer(state, H, h, mask, scheme, open_edges)` unchanged signature; it now builds source columns only at `mask` points.

- [ ] **Step 1: Failing tests**

```python
def test_exterior_tracer_equals_the_full_domain_version_at_the_masked_columns():
    from core.dynamics import open_boundary as obc
    rng = np.random.default_rng(3)
    ny, nx, n = 6, 9, 5
    H = np.full((ny, nx), 100.0)
    h = np.full((n, ny, nx), 20.0) * rng.uniform(0.9, 1.1, (1, ny, nx))
    h *= H[None] / h.sum(0)
    zw = np.linspace(0.0, -100.0, 5)[:, None].repeat(ny, 1)
    st = {"west": OBData(side="west", T=rng.uniform(10, 20, (4, ny)), u_normal=np.zeros((4, ny)), zw=zw)}
    band = np.zeros((ny, nx), bool)
    band[:, :3] = True
    full = obc._exterior_tracer_full(st, H, h, band, 2, ("west",))
    edge = np.zeros((ny, nx), bool)
    edge[:, 0] = True
    np.testing.assert_array_equal(obc.exterior_tracer(st, H, h, band, 2, ("west",)), full)
    np.testing.assert_array_equal(obc.exterior_tracer(st, H, h, edge, 2, ("west",))[:, :, 0], full[:, :, 0])


def test_exterior_tracer_band_kept_with_tracer_sponge():
    m = _nested_model(sponge_tracer=True)          # helper below
    assert m._ext_cols.sum() == m._bc_mask.sum()
    m2 = _nested_model(sponge_tracer=False)
    assert m2._ext_cols.sum() < m2._bc_mask.sum() and m2._ext_cols[:, 0].all()


def test_nested_model_is_bit_identical_with_the_restricted_exterior():
    a, b = _nested_model(), _nested_model()
    b._ext_cols = b._bc_mask.copy()                 # the old behaviour: the whole band
    sa, sb = a.initial_state(*_rest(a)), b.initial_state(*_rest(b))
    for _ in range(10):
        sa, sb = a.step(sa), b.step(sb)
    for k in ("eta", "u", "v", "T", "h"):
        assert np.array_equal(getattr(sa, k), getattr(sb, k)), k
```

with helpers (same file):

```python
def _nested_model(sponge_tracer=False):
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid
    ny, nx, n = 4, 20, 6
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=1e-4)
    zw = np.linspace(0.0, -50.0, 6)[:, None].repeat(ny, 1)

    def state(t):
        a = np.sin(2 * np.pi * t / 3600.0)
        return {"west": OBData(side="west", T=np.full((5, ny), 12.0 + a), u_normal=np.full((5, ny), 0.01 * a),
                               zw=zw, eta=np.full(ny, 0.02 * a), ubar=np.full(ny, 0.01 * a))}
    return SigmaBaroclinic(hg, np.full((ny, nx), 50.0), dt=30.0, n_layers=n, rho0=1000.0, t_coef=0.2, z0=None,
                           closure="constant", vertical_coordinate="ale", regrid="zstar",
                           regrid_params=dict(dz_top=np.full(n, 50.0 / n)), open_boundaries=("west", "east"),
                           sponge_cells=5, sponge_tau=(600.0, 60.0), sponge_tracer=sponge_tracer,
                           open_boundary_state=state)


def _rest(m):
    T = 10.0 + 0.1 * np.arange(m.n)[:, None, None] * np.ones((1,) + m.H.shape)
    return np.zeros(m.H.shape), T
```

- [ ] **Step 2: Run, expect FAIL** (`_exterior_tracer_full` and `_ext_cols` missing)

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_open_boundary_state.py -q -k "masked_columns or tracer_sponge or restricted_exterior"`

- [ ] **Step 3: Implement**
  - In `open_boundary.py` rename the current `exterior_tracer` body to `_exterior_tracer_full` (oracle, unchanged) and write `exterior_tracer` that, per edge, gathers the selected points `jj, ii = np.nonzero(sel)`, builds the source columns only there (`_source_columns_at(ob, edge, depth_pts, a_pts, check_pts)` returning (nz, npts) thicknesses and T, with the same clipping arithmetic as `_source_columns`), remaps them with `rm.remap_columns` on arrays shaped (nz, npts, 1) (stretched as `_remap_stretched`, same arithmetic) and scatters into `out[:, jj, ii]`.
  - In `SigmaBaroclinic` setup (where `_bc_mask` is built): `self._ext_cols = self._bc_mask.copy() if self.sponge_tracer else edge_cells` with `edge_cells` = the boundary cells of the open edges (column 0 for west, −1 for east, row 0 for south, −1 for north).
  - `_exterior_T` and `_exterior` use `self._ext_cols` instead of `self._bc_mask`.

- [ ] **Step 4: Run the three tests, then the open-boundary, restart and ALE suites**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_open_boundary_state.py tests/test_restart.py tests/test_sigma_ale.py -q`
Expected: all pass (fingerprints unchanged).

- [ ] **Step 5: Measure** `python validation/bayiw_cut/profile_step.py zstar_nest` — expected ≤ z\* radiate + 5 %. Record in the ledger.

- [ ] **Step 6: Commit**

```bash
git add core/dynamics/open_boundary.py core/dynamics/sigma_baroclinic.py tests/test_open_boundary_state.py
git commit -m "Nested exterior computed only in the columns that use it (perf part 2, task 2)"
```

---

### Task 3: Parallel `truncate_cfl` and `ke_gradient`

**Files:**
- Modify: `core/dynamics/vector_invariant.py`
- Test: `tests/test_vector_invariant.py`

**Interfaces:**
- Produces: `truncate_cfl` and `ke_gradient` keep their signatures and results; `truncate_cfl_numpy`, `ke_gradient_numpy` are the oracles.

- [ ] **Step 1: Failing tests**

```python
@pytest.mark.parametrize("seed", [0, 1, 2])
def test_truncate_cfl_kernel_matches_numpy(seed):
    rng = np.random.default_rng(seed)
    n, ny, nx = 7, 9, 11
    u = rng.normal(0.0, 3.0, (n, ny, nx + 1))
    dx_t = rng.uniform(500.0, 2000.0, (ny, nx))
    h_face = np.where(rng.random((n, ny, nx + 1)) < 0.3, 0.0, rng.uniform(0.0, 5.0, (n, ny, nx + 1)))
    a = vi.truncate_cfl(u, 300.0, dx_t, h_face, 0.01)
    b = vi.truncate_cfl_numpy(u, 300.0, dx_t, h_face, 0.01)
    assert np.array_equal(a[0], b[0]) and a[1] == b[1] and a[2] == b[2]


def test_ke_gradient_kernel_matches_numpy():
    rng = np.random.default_rng(4)
    K = rng.random((5, 8, 10))
    dx_u, dy_v = rng.uniform(500, 900, (8, 11)), rng.uniform(500, 900, (9, 10))
    for a, b in zip(vi.ke_gradient(K, dx_u, dy_v), vi.ke_gradient_numpy(K, dx_u, dy_v)):
        assert np.array_equal(a, b)
```

- [ ] **Step 2: Run, expect FAIL** (`truncate_cfl_numpy` missing)

Run: `python -m pytest tests/test_vector_invariant.py -q -k "kernel_matches"`

- [ ] **Step 3: Implement** — rename the current functions to `*_numpy`; add kernels with the same arithmetic:

```python
@numba.njit(parallel=True, cache=True)
def _truncate_kernel(u, dt, dx_t, h_face, h_count, cfl, out, counted):  # pragma: no cover
    n, ny, nx1 = u.shape
    nx = nx1 - 1
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        for i in range(nx1):
            x = u[k, j, i]
            dxd = (dx_t[j, max(i - 1, 0)] if x >= 0 else dx_t[j, min(i, nx - 1)])
            over = abs(x) * dt / dxd > cfl
            out[k, j, i] = (np.sign(x) * 0.9 * cfl * dxd / dt) if over else x
            counted[k, j, i] = over and h_face[k, j, i] > h_count


def truncate_cfl(u, dt, dx_t, h_face, h_count, cfl=0.5):
    """(docstring as truncate_cfl_numpy).  Numba kernel; truncate_cfl_numpy is the reference."""
    out = np.empty(u.shape)
    counted = np.empty(u.shape, np.bool_)
    _truncate_kernel(np.asarray(u, float), float(dt), np.asarray(dx_t, float), np.asarray(h_face, float),
                     float(h_count), float(cfl), out, counted)
    c = int(counted.sum())
    first = tuple(int(a[0]) for a in np.nonzero(counted)) if c else None
    return out, c, first


@numba.njit(parallel=True, cache=True)
def _ke_grad_kernel(K, dx_u, dy_v, gx, gy):  # pragma: no cover
    n, ny, nx = K.shape
    for p in numba.prange(n * ny):
        k, j = p // ny, p % ny
        gx[k, j, 0] = 0.0
        gx[k, j, nx] = 0.0
        for i in range(1, nx):
            gx[k, j, i] = -(K[k, j, i] - K[k, j, i - 1]) / dx_u[j, i]
    for p in numba.prange(n * (ny + 1)):
        k, J = p // (ny + 1), p % (ny + 1)
        for i in range(nx):
            gy[k, J, i] = 0.0 if (J == 0 or J == ny) else -(K[k, J, i] - K[k, J - 1, i]) / dy_v[J, i]


def ke_gradient(K, dx_u, dy_v):
    """−∂K/∂x at the u-faces and −∂K/∂y at the v-faces [m s⁻²]; domain-edge faces 0.  Numba kernel;
    ke_gradient_numpy is the reference."""
    n, ny, nx = K.shape
    gx, gy = np.empty((n, ny, nx + 1)), np.empty((n, ny + 1, nx))
    _ke_grad_kernel(np.asarray(K, float), np.asarray(dx_u, float), np.asarray(dy_v, float), gx, gy)
    return gx, gy
```

Before each kernel call: `numba.set_num_threads(_threads())` with the module's existing thread helper (the one the other kernels of `vector_invariant.py` use).

- [ ] **Step 4: Run** `python -m pytest tests/test_vector_invariant.py tests/test_sigma_ale.py -q` — all pass, fingerprints unchanged.

- [ ] **Step 5: Commit**

```bash
git add core/dynamics/vector_invariant.py tests/test_vector_invariant.py
git commit -m "Parallel truncate_cfl and ke_gradient kernels (perf part 2, task 3)"
```

---

### Task 4: Reference-profile interpolation kernel and no repeated `nan_to_num`

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (`ReferenceProfile.__call__`; cache `self._H0` = `np.ascontiguousarray(np.nan_to_num(H))`, `self._H0T`, `self._actT` at setup; pass them to `face_transport_thickness`)
- Modify: `core/vertical/ale.py` (`face_transport_thickness`: skip `nan_to_num`/copy when given contiguous float arrays)
- Test: `tests/test_sigma_ale.py`

- [ ] **Step 1: Failing test**

```python
def test_reference_profile_kernel_matches_np_interp():
    from core.dynamics.sigma_baroclinic import ReferenceProfile
    rng = np.random.default_rng(5)
    z = np.sort(rng.uniform(-600, 0, 50))
    prof = ReferenceProfile(z, 1020 + 0.01 * -z)
    zz = rng.uniform(-700, 50, (6, 7, 8))                       # includes heights beyond both ends
    assert np.array_equal(prof(zz), np.interp(zz, prof.z, prof.rho))
    assert prof(zz).shape == zz.shape
```

Write it with the `__call__` replaced by the kernel path but expecting the kernel attribute: also assert `hasattr(ReferenceProfile, "_interp_kernel")` so the test fails before the change.

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** a parallel kernel reproducing `np.interp` (binary search per element; left/right constant; `slope = (fp[j+1]−fp[j])/(xp[j+1]−xp[j])`, `res = slope*(x − xp[j]) + fp[j]`, exactly NumPy's formula in `numpy/core/src/multiarray/compiled_base.c`); `__call__` uses it for arrays with ≥ 1000 elements, `np.interp` otherwise. If `array_equal` fails on an ULP difference, keep `np.interp` and record the ruling — do not loosen the test.

- [ ] **Step 4: Run** `python -m pytest tests/test_sigma_ale.py tests/test_vertical_ale.py tests/test_restart.py -q` — all pass.

- [ ] **Step 5: Commit** (`git add core/dynamics/sigma_baroclinic.py core/vertical/ale.py tests/test_sigma_ale.py`).

---

### Task 5: Fused SSP-RK3 combination in `_advect`

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (`_advect`; keep the current body as `_advect_numpy`)
- Test: `tests/test_sigma_ale.py`

- [ ] **Step 1: Failing test**

```python
def test_advect_kernel_matches_numpy():
    m = _ale_box("zstar", nx=12, nz=6)
    st = m.initial_state(np.zeros((1, 12)), 10.0 + np.random.default_rng(6).random((6, 1, 12)))
    for _ in range(3):
        st = m.step(st)
    V0 = m.layer_volumes(st.eta, st.h)
    V1 = V0 * 1.01
    V1[:, :, 0] = 0.0                                              # a vanished column keeps C
    ok = V1 > 0
    C = np.stack([st.T, st.T**2])
    Fz = np.zeros((m.n + 1,) + m.H.shape)
    a = m._advect(C, V0, V1, st.Qx, st.Qy, Fz, ok)
    b = m._advect_numpy(C, V0, V1, st.Qx, st.Qy, Fz, ok)
    assert np.array_equal(a, b) and np.isfinite(a).all()
```

- [ ] **Step 2: Run, expect FAIL** (`_advect_numpy` missing).

- [ ] **Step 3: Implement** three stage kernels with NumPy's exact expressions (no common coefficient form, so the rounding is identical):

```python
@numba.njit(parallel=True, cache=True)
def _rk_stage(q_a, q_b, r, dt, V0, dV, fV, C, stage, q_out, c_out):  # pragma: no cover
    """SSP-RK3 stage of _advect_numpy: stage 1 q = q0 + dt r; 2: 0.75 q0 + 0.25 (q1 + dt r);
    3: q0/3 + 2/3 (q2 + dt r); then c = q/V where V = V0 + fV dV > 0, else C (the `safe` rule)."""
    m = q_out.size
    qa, qb, rr, cc = q_a.ravel(), q_b.ravel(), r.ravel(), C.ravel()
    qo, co = q_out.ravel(), c_out.ravel()
    nv = V0.size
    v0, dv = V0.ravel(), dV.ravel()
    for p in numba.prange(m):
        if stage == 1:
            q = qa[p] + dt * rr[p]
        elif stage == 2:
            q = 0.75 * qa[p] + 0.25 * (qb[p] + dt * rr[p])
        else:
            q = qa[p] / 3.0 + 2.0 / 3.0 * (qb[p] + dt * rr[p])
        qo[p] = q
        v = v0[p % nv] + fV * dv[p % nv] if fV != 1.0 else v0[p % nv] + dv[p % nv]
        co[p] = q / v if v > 0 else cc[p]
```

`_advect` calls it with `fV = 1.0` (Va = V0 + dV), `0.5` (Vb), and for the last stage passes `V1` directly (`V0=V1, dV=0, fV=1.0`) — check each against `_advect_numpy`'s exact arithmetic (`Va = V0 + dV` vs `Vb = V0 + 0.5 * dV`; `V1` itself for the last `safe`). The `pc` branch (`tracer_time == "pc"`) keeps the NumPy path.

- [ ] **Step 4: Run** `python -m pytest tests/test_sigma_ale.py tests/test_sigma_baroclinic.py tests/test_restart.py -q` — all pass, fingerprints unchanged.

- [ ] **Step 5: Commit** (`git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py`).

---

### Task 6: `_step_ale` glue — transport velocities and the face fractions computed once

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (`_step_ale`)
- Modify: `core/vertical/ale.py` (new `transport_velocity`, oracle `transport_velocity_numpy`)
- Test: `tests/test_vertical_ale.py`

**Interfaces:**
- Produces: `transport_velocity(u_new, u_old, last_wet, w, L, D, theta) -> (ut, Qb)` with `ut = θ u_new + (1−θ) where(last_wet, u_old, 0)` (N, a, b) and `Qb = L D Σ_k w_k ut_k` (a, b).

- [ ] **Step 1: Failing test**

```python
def test_transport_velocity_kernel_matches_numpy():
    rng = np.random.default_rng(7)
    n, a, b = 6, 5, 8
    u1, u0 = rng.normal(size=(n, a, b)), rng.normal(size=(n, a, b))
    wet = rng.random((a, b)) > 0.3                                  # faces dry in the last step: old velocity 0
    w = rng.random((n, a, b))
    w /= w.sum(0)
    L, D = rng.uniform(500, 900, (a, b)), rng.uniform(0, 50, (a, b))
    ut, Qb = ale.transport_velocity(u1, u0, wet, w, L, D, 0.6)
    ut0, Qb0 = ale.transport_velocity_numpy(u1, u0, wet, w, L, D, 0.6)
    assert np.array_equal(ut, ut0) and np.allclose(Qb, Qb0, rtol=1e-14, atol=0)
```

(`Qb` sums over k: the kernel sums in the same k order, so it should be identical; the tolerance only guards NumPy's pairwise summation in `.sum(axis=0)` — if `array_equal` holds, tighten the test to it.)

- [ ] **Step 2: Run, expect FAIL.**

- [ ] **Step 3: Implement** `transport_velocity_numpy` = the current `_step_ale` lines (`un = where(last_wet, u, 0)`, `utx = th u_new + (1−th) un`, `Qbx = Lu last_Du (wu utx).sum(0)`), the kernel per face column, and use it in `_step_ale` for x and (on the v arrays, unswapped) y. In the same task, compute `face_fractions(hu1)`, `face_fractions(hv1)` once and pass them to both `_remap_faces` (new optional `w_tgt` argument) and `zero_thin_bed_velocity`.

- [ ] **Step 4: Run** `python -m pytest tests/test_vertical_ale.py tests/test_sigma_ale.py tests/test_restart.py -q` — all pass, fingerprints unchanged.

- [ ] **Step 5: Measure** all three cases with `profile_step.py`; ledger the numbers.

- [ ] **Step 6: Commit** (`git add core/vertical/ale.py core/dynamics/sigma_baroclinic.py tests/test_vertical_ale.py`).

---

### Task 7: Next profile round, then close

**Files:** as the profile dictates; `validation/bayiw_cut/README.md`, `CLAUDE.md`, `docs/design/vision_roadmap.md` (scorecard: no row change, add the cost figures to the BayIW cut rows).

- [ ] **Step 1: Profile** (`profile_step.py`). If the 1→5 speed-up is < 3.5 and some serial piece is > 5 % of the step, take the largest one and repeat the Task 3–6 pattern for it: rename to `*_numpy`, write the kernel-vs-oracle test first (random input with vanished layers, land and edges), watch it fail, implement, run `tests/test_sigma_ale.py tests/test_restart.py`, commit. Ledger each round with the before/after self time. Stop at the spec's criterion (§2) or when the next round would gain < 5 % of the step; ledger why it stopped.
- [ ] **Step 2: Final measurement** saved to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_cut/profile_final.txt`; README "Rendimiento" table baseline vs final vs the §3 targets; any target missed is stated with the remaining breakdown.
- [ ] **Step 3: Full suite** `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider` — all pass (+ the existing xfails); update the test count in `CLAUDE.md` and the performance sentence of the ALE row.
- [ ] **Step 4: Commit** the docs (explicit paths).
