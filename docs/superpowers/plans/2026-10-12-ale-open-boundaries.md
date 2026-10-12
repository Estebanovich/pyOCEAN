# ALE Open Boundaries (Plan C, scope A) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Let the ALE mode of `SigmaBaroclinic` (`vertical_coordinate="ale"`) run with open boundaries: Flather, per-layer Orlanski radiation and velocity sponge on the thickness-weighted baroclinic part, open-face layer transports on the interior thickness, and an exterior T profile kept in z and remapped to the current layers for inflow and the tracer sponge.

**Architecture:** One code path for σ and ALE. The depth-mean/baroclinic split takes weights: Δσ in σ (the current code, bit-identical) or the ALE face fractions. Three pure functions go into `core/vertical/ale.py`: `baroclinic_split`, `open_face_transport` and `exterior_on_layers`. `SigmaBaroclinic` wires them into `_step_ale`.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba (remap kernels), pytest.

**Spec:** `docs/superpowers/specs/2026-10-12-ale-open-boundaries-design.md` (parent: `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` §6.1).

## Global Constraints

- Never use the 6 efficiency cores: run everything with `NUMBA_NUM_THREADS=5`; the model calls `numba.set_num_threads(_THREADS[0])` before every parallel kernel.
- Install nothing; numpy stays 1.26.4.
- Commit only if Esteban authorized commits for this plan at the handoff. Never commit `pyOCEAN-literature-review-guide.docx`, `log_*.txt` or `*.npz`. Commit messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn
  ```
- σ results must stay bit-identical (spec §2): any σ code touched keeps the `np.tensordot(self.ds, …)` arithmetic when no weights are given.
- Scope A only: no tracer reservoir with L > 0, no τ_in/τ_out nudging, no drying columns at ALE open boundaries, ALE restarts still refused (spec §1).
- Docstrings explain the physics and cite sources as `file:line` (MOM6, CROCO); type hints on every new function (CLAUDE.md).
- Test command for the whole suite: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider`.

## Review Focus

1. **A layer has vanished in the boundary cell** (z\* layer clipped at a shallower bed, massless hybrid layer, or a fully dry column): the open-face transport must give Q_k = 0 for that layer, with no NaN and no division by zero. *Pinned by the Task 1 test `test_open_face_transport_vanished_layers_carry_nothing`.*
2. **Initial η ≠ 0** (Σh_ext = H + η₀ ≠ H): the exterior must be scaled by Σh/Σh_ext, not by (H + η)/H, or inflow T is biased. *Pinned by Task 1 `test_exterior_on_layers_conserves_the_column_integral_after_scaling` and Task 3 `test_ale_open_boundaries_keep_a_uniform_tracer_uniform` (η bump at t = 0).*
3. **South/north open edges (v-faces) in a 2-D domain**: the swapped-axis code must be as right as the u-faces. *Pinned by Task 3 `test_ale_open_boundaries_on_four_edges_keep_a_uniform_tracer_uniform`.*
4. **Inflow of exterior water creates no new extrema** in the hybrid or z\* channel. *Pinned by Task 4 `test_ale_sponge_with_radiation_absorbs_the_internal_wave` (T stays inside its initial range ± 1 %).*
5. **The non-default `thickness_flux='upwind'`** with open boundaries. *Pinned by the Task 3 parametrization over `thickness_flux`.*

---

## File Structure

- `core/vertical/ale.py` (modify): new pure functions `baroclinic_split`, `open_face_transport`, `exterior_on_layers` (the ALE face geometry module already holds the face-thickness helpers these belong with).
- `core/dynamics/sigma_baroclinic.py` (modify):
  - weighted `_baroclinic_part` / `_radiate`;
  - shared `_flather_args` / `_open_velocity`;
  - `_inflow_T`, `_exterior`, `_open_face_transports`;
  - boundary mask;
  - `initial_state` stores `h_ext`;
  - `_step_ale` wiring;
  - remove the Plan C `ValueError`, add the H ≤ 0 check.
- `tests/test_vertical_ale.py` (modify): unit tests of the three functions.
- `tests/test_sigma_ale.py` (modify): weighted radiation test; ALE open-boundary integration tests; ALE channel tests.
- `CLAUDE.md`, `docs/design/vision_roadmap.md` §7, parent spec §9 (modify): status and numbers.

---

### Task 1: Pure open-boundary functions in `core/vertical/ale.py`

**Files:**
- Modify: `core/vertical/ale.py` (append after `lagrangian_thickness`, imports at the top)
- Test: `tests/test_vertical_ale.py` (append)

**Interfaces:**
- Consumes: `core.vertical.remap.remap_columns(h_src, f_src, h_tgt, scheme: int, wet) -> np.ndarray` (existing).
- Produces:
  - `baroclinic_split(vel: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]`: (mean (…), prime (N, …)).
  - `open_face_transport(h_int: np.ndarray, u_new: np.ndarray, u_old: np.ndarray, Q_bt: np.ndarray, L: np.ndarray, theta: float) -> np.ndarray`: arrays (N, m) for one open edge; Q_bt, L (m,); returns Q (N, m) [m³ s⁻¹].
  - `exterior_on_layers(h_ext: np.ndarray, T_ext: np.ndarray, h: np.ndarray, mask: np.ndarray, scheme: int) -> np.ndarray`: (N, ny, nx) arrays, mask (ny, nx); returns T on layers h inside the mask, T_ext elsewhere.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_vertical_ale.py`)

```python
# ---------------------------------------------------------------- Plan C: open boundaries (scope A)
def test_baroclinic_split_with_sigma_weights_matches_the_sigma_mean():
    rng = np.random.default_rng(11)
    ds = np.array([0.4, 0.3, 0.2, 0.1])
    u = rng.normal(0.0, 0.3, (4, 2, 5))
    mean, prime = ale.baroclinic_split(u, np.broadcast_to(ds[:, None, None], u.shape))
    np.testing.assert_allclose(mean, np.tensordot(ds, u, axes=(0, 0)), rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(prime, u - mean[None], rtol=0, atol=0)


def test_baroclinic_split_prime_has_zero_weighted_mean_with_vanished_layers():
    rng = np.random.default_rng(12)
    hf = rng.uniform(0.0, 5.0, (6, 3, 7))
    hf[2] = 0.0                                                          # a vanished layer
    hf[:, 0, 0] = 0.0                                                    # a face with no water
    w = ale.face_fractions(hf)
    u = rng.normal(0.0, 0.5, hf.shape)
    mean, prime = ale.baroclinic_split(u, w)
    np.testing.assert_allclose((w * prime).sum(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(mean[None] + prime, u, rtol=0, atol=1e-15)


def test_open_face_transport_adds_up_to_the_flather_flux():
    rng = np.random.default_rng(13)
    h = rng.uniform(0.1, 8.0, (5, 6))
    un, uo = rng.normal(0.0, 0.4, (5, 6)), rng.normal(0.0, 0.4, (5, 6))
    Qbt, L = rng.normal(0.0, 50.0, 6), np.full(6, 1000.0)
    Q = ale.open_face_transport(h, un, uo, Qbt, L, 0.6)
    np.testing.assert_allclose(Q.sum(axis=0), Qbt, rtol=1e-12, atol=1e-9)


def test_open_face_transport_vanished_layers_carry_nothing():
    # review focus 1: a vanished layer and a fully dry face column
    h = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    un = np.full((4, 2), 0.3)
    uo = np.full((4, 2), -0.1)
    Q = ale.open_face_transport(h, un, uo, np.array([10.0, 0.0]), np.array([100.0, 100.0]), 0.6)
    assert np.isfinite(Q).all()
    assert (Q[[0, 2], 0] == 0).all() and (Q[:, 1] == 0).all()
    np.testing.assert_allclose(Q[:, 0].sum(), 10.0, rtol=1e-14)


def test_open_face_transport_reduces_to_the_sigma_formula():
    # h_k = D Δσ_k: Q_k = Δσ_k Q_bt + L D Δσ_k (θ u′ⁿ⁺¹ + (1 − θ) u′ⁿ), u′ with Δσ weights
    rng = np.random.default_rng(14)
    ds, D, L, th = np.array([0.1, 0.2, 0.3, 0.4]), np.array([20.0, 35.0, 50.0]), np.full(3, 500.0), 0.6
    un, uo = rng.normal(0.0, 0.3, (4, 3)), rng.normal(0.0, 0.3, (4, 3))
    Qbt = rng.normal(0.0, 100.0, 3)
    pn, po = un - np.tensordot(ds, un, axes=(0, 0)), uo - np.tensordot(ds, uo, axes=(0, 0))
    ref = ds[:, None] * Qbt[None] + (L * D)[None] * ds[:, None] * (th * pn + (1 - th) * po)
    Q = ale.open_face_transport(D[None] * ds[:, None], un, uo, Qbt, L, th)
    np.testing.assert_allclose(Q, ref, rtol=1e-12, atol=1e-10)


def test_exterior_on_layers_is_the_identity_on_the_same_grid():
    rng = np.random.default_rng(15)
    h = rng.uniform(0.5, 5.0, (6, 2, 3))
    T = rng.uniform(5.0, 20.0, h.shape)
    out = ale.exterior_on_layers(h, T, h.copy(), np.ones((2, 3), bool), 3)
    np.testing.assert_allclose(out, T, rtol=1e-12)


def test_exterior_on_layers_conserves_the_column_integral_after_scaling():
    # review focus 2: the target column holds H + η ≠ Σ h_ext; scaled by Σh / Σh_ext
    rng = np.random.default_rng(16)
    h_ext = rng.uniform(0.5, 5.0, (6, 1, 4))
    T = rng.uniform(5.0, 20.0, h_ext.shape)
    h = rng.uniform(0.5, 5.0, (6, 1, 4))
    mask = np.array([[True, True, False, True]])
    out = ale.exterior_on_layers(h_ext, T, h, mask, 3)
    scale = h.sum(0) / h_ext.sum(0)
    np.testing.assert_allclose((h * out).sum(0)[mask], (scale * (h_ext * T).sum(0))[mask], rtol=1e-12)
    np.testing.assert_array_equal(out[:, ~mask], T[:, ~mask])          # outside the mask: unchanged


def test_exterior_on_layers_gives_the_layer_means_of_a_linear_profile():
    # fine exterior layers with T linear in z; coarse targets get the exact layer means
    z_w = np.linspace(-100.0, 0.0, 21)
    h_ext = np.diff(z_w)[:, None, None] * np.ones((1, 1, 2))
    T = (10.0 + 0.05 * 0.5 * (z_w[1:] + z_w[:-1]))[:, None, None] * np.ones((1, 1, 2))
    zt = np.array([-100.0, -70.0, -50.0, -45.0, -10.0, 0.0])
    h = np.zeros((20, 1, 2))
    h[:5] = np.diff(zt)[:, None, None]                                   # 5 thick layers + 15 vanished at the top
    out = ale.exterior_on_layers(h_ext, T, h, np.ones((1, 2), bool), 3)
    np.testing.assert_allclose(out[:5, 0, 0], 10.0 + 0.05 * 0.5 * (zt[1:] + zt[:-1]), rtol=1e-10)
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_ale.py -q -k "baroclinic_split or open_face or exterior_on"`
Expected: 8 FAIL with `AttributeError: module 'core.vertical.ale' has no attribute 'baroclinic_split'` (and the same for the other two).

- [ ] **Step 3: Implement** (in `core/vertical/ale.py`; add `from core.vertical import remap as rm` below `import numpy as np`; append the functions at the end of the file)

```python
# ---------------------------------------------------------------- open boundaries (Plan C, scope A)
def baroclinic_split(vel: np.ndarray, w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Depth mean and baroclinic part of a face velocity with layer weights w (Σ_k w_k = 1):

        ū = Σ_k w_k u_k,   u′_k = u_k − ū,   so Σ_k w_k u′_k = 0

    w = Δσ in σ, the face fractions ``face_fractions`` in ALE (spec 2026-10-12-ale-open-boundaries §3.1).
    vel, w (N, …); returns (ū (…), u′ (N, …)) [m s⁻¹].
    """
    mean = (w * vel).sum(axis=0)
    return mean, vel - mean[None]


def open_face_transport(h_int: np.ndarray, u_new: np.ndarray, u_old: np.ndarray, Q_bt: np.ndarray,
                        L: np.ndarray, theta: float) -> np.ndarray:
    """
    Layer volume fluxes through the faces of one open edge [m³ s⁻¹] (spec §3.4):

        Q_k = (h_k / Σ_j h_j) Q_bt + L h_k (θ u′_kⁿ⁺¹ + (1 − θ) u′_kⁿ)

    with h_k the thickness of the interior cell (MOM6 MOM_continuity_PPM.F90:1237-1248) and u′ split
    with the interior fractions h_k/Σh, so Σ_k Q_k = Q_bt (the Flather flux of the free surface) to
    round-off and a vanished layer carries nothing.  h_int, u_new, u_old (N, m); Q_bt, L (m,).
    """
    tot = h_int.sum(axis=0)
    wet = tot > 0
    w = np.where(wet[None], h_int / np.where(wet, tot, 1.0)[None], 0.0)
    pn = u_new - (w * u_new).sum(axis=0)[None]
    po = u_old - (w * u_old).sum(axis=0)[None]
    Q = w * Q_bt[None] + L[None] * h_int * (theta * pn + (1.0 - theta) * po)
    return np.where(wet[None], Q, 0.0)


def exterior_on_layers(h_ext: np.ndarray, T_ext: np.ndarray, h: np.ndarray, mask: np.ndarray,
                       scheme: int) -> np.ndarray:
    """
    Exterior tracer profile on the current layers (spec §3.5): the profile (h_ext, T_ext), kept in z
    since the initial state, is stretched to the current column, h_ext Σh/Σh_ext (= (H + η)/H for a
    resting initial state, z*-like), and remapped conservatively (core.vertical.remap, ``scheme``)
    onto the thicknesses h — the external data on its own z axis of MOM6 (MOM_open_boundary.F90:1188-1204),
    on the *current* thicknesses (avoiding OBC_TRACER_DZ_BUG, :778-781).  Columns outside ``mask``
    (ny, nx) or without water keep T_ext.  (N, ny, nx) arrays.
    """
    tot_e, tot = h_ext.sum(axis=0), h.sum(axis=0)
    wet = np.asarray(mask, bool) & (tot_e > 0) & (tot > 0)
    scale = np.where(wet, tot / np.where(wet, tot_e, 1.0), 1.0)
    out = rm.remap_columns(h_ext * scale[None], T_ext, h, scheme, wet)
    return np.where(wet[None], out, T_ext)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_ale.py -q`
Expected: all pass (19 earlier + 8 new = 27).

- [ ] **Step 5: Commit** (only if authorized; see Global Constraints)

```bash
git add core/vertical/ale.py tests/test_vertical_ale.py
git commit -m "ALE open boundaries: baroclinic split, open-face transport, exterior profile in z (Plan C task 1)"
```

---

### Task 2: Weighted split, radiation and sponge, shared by σ and ALE

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py`:
  - `_baroclinic_part` (~line 1097);
  - `_radiate` (~1102-1125);
  - σ `step` (~1276-1292): Flather argument dict and velocity sponge/radiation moved into helpers.
- Test: `tests/test_sigma_ale.py` (append)

**Interfaces:**
- Consumes: `baroclinic_split` (Task 1).
- Produces (methods of `SigmaBaroclinic`):
  - `_baroclinic_part(vel, w=None) -> (mean, prime)`: w None → the current Δσ tensordot.
  - `_radiate(u_old, u_new, axis, edge, w=None) -> None`.
  - `_flather_args() -> dict` (empty without open boundaries).
  - `_open_velocity(st, u_new, v_new, wu=None, wv=None) -> None`: sponge on u′, then Orlanski, in place.

- [ ] **Step 1: Save a σ reference before touching the code** (for the bit-identical check of spec §5 integration 4)

```bash
mkdir -p .superpowers/sdd/2026-10-12-ale-open-boundaries
NUMBA_NUM_THREADS=5 python - <<'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np
from tests.test_sigma_baroclinic import _channel, _run
m, st = _channel(80, 40e3, open_boundaries=("west", "east"), sponge_cells=10, sponge_tau=(1200.0, 120.0))
st = _run(m, st, 2.0)
np.save(".superpowers/sdd/2026-10-12-ale-open-boundaries/sigma_ref_u.npy", st.u)
np.save(".superpowers/sdd/2026-10-12-ale-open-boundaries/sigma_ref_T.npy", st.T)
print("saved", np.abs(st.u).max())
EOF
```
Expected: `saved <a positive number>`. These `.npy` files live in the git-ignored workspace; never commit them.

- [ ] **Step 2: Write the failing test** (append to `tests/test_sigma_ale.py`)

```python
# ---------------------------------------------------------------- open boundaries (Plan C, scope A)
def test_radiation_with_face_weights_keeps_the_depth_mean_and_a_zero_weighted_baroclinic_part():
    # spec 2026-10-12-ale-open-boundaries §3.2: Orlanski on u′, re-centred with the face weights
    from core.vertical import ale
    hg = create_cartesian_grid(nx=8, ny=2, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((2, 8), 20.0), dt=30.0, n_layers=4, z0=None, closure="constant",
                        open_boundaries=("west", "east"))
    st = m.initial_state(np.zeros((2, 8)), np.full((4, 2, 8), 7.0))
    m.step(st)                                                           # sets the Flather face depths
    rng = np.random.default_rng(21)
    w = ale.face_fractions(rng.uniform(0.0, 4.0, (4, 2, 9)))
    u_old, u_new = rng.normal(0.0, 0.2, (4, 2, 9)), rng.normal(0.0, 0.2, (4, 2, 9))
    mean0 = (w * u_new).sum(0)
    m._radiate(u_old, u_new, 2, 0, w)
    np.testing.assert_allclose((w * u_new).sum(0)[:, 0], mean0[:, 0], atol=1e-15)
    _, prime = ale.baroclinic_split(u_new, w)
    np.testing.assert_allclose((w * prime).sum(0)[:, 0], 0.0, atol=1e-15)
```

- [ ] **Step 3: Run it to verify it fails**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k radiation_with_face_weights`
Expected: FAIL with `TypeError: SigmaBaroclinic._radiate() takes 5 positional arguments but 6 were given`.

- [ ] **Step 4: Implement**

Add `baroclinic_split` to the existing import from `core.vertical.ale` (keep the list sorted):

```python
from core.vertical.ale import (DZ_FLOOR, baroclinic_split, face_fractions, face_thickness, face_transport_thickness,
                               lagrangian_thickness, ppm_transport_x, upwind_transport_x, zero_thin_bed_velocity)
```

Replace `_baroclinic_part` and the two σ-only lines at the end of `_radiate`:

```python
    def _baroclinic_part(self, vel: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Depth mean and baroclinic remainder u_k − ū: Σ Δσ_k u_k (σ, ``w`` None — kept as tensordot so
        σ runs stay bit-identical) or Σ w_k u_k with the ALE face fractions (core.vertical.ale.baroclinic_split).
        """
        if w is None:
            mean = np.tensordot(self.ds, vel, axes=(0, 0))
            return mean, vel - mean[None]
        return baroclinic_split(vel, w)
```

In `_radiate`, change the signature to `def _radiate(self, u_old: np.ndarray, u_new: np.ndarray, axis: int, edge: int, w: np.ndarray | None = None) -> None:`, add to its docstring "``w``: depth-mean weights (Δσ when None, ALE face fractions of hⁿ for both levels)", and replace

```python
        _, po = self._baroclinic_part(u_old)
        bar, pn = self._baroclinic_part(u_new)
```
with
```python
        _, po = self._baroclinic_part(u_old, w)
        bar, pn = self._baroclinic_part(u_new, w)
```
and
```python
        pb -= np.tensordot(self.ds, pb, axes=(0, 0))[None]                            # keep Σ Δσ u′ = 0
```
with
```python
        if w is None:
            pb -= np.tensordot(self.ds, pb, axes=(0, 0))[None]                        # keep Σ Δσ u′ = 0
        else:
            pb -= (take(w, edge) * pb).sum(axis=0)[None]                              # keep Σ w u′ = 0
```

Add the two helpers after `_radiate`:

```python
    def _flather_args(self) -> dict:
        """Flather radiation to a resting exterior (η = 0, ū = 0) for the free-surface step; {} if closed."""
        if not self.open:
            return {}
        zu, zv = np.zeros(self.fs.H_u.shape), np.zeros(self.fs.H_v.shape)
        return dict(flather_eta=(zu, zv), flather_ubar=zu, flather_vbar=zv)

    def _open_velocity(self, st: SigmaState, u_new: np.ndarray, v_new: np.ndarray,
                       wu: np.ndarray | None = None, wv: np.ndarray | None = None) -> None:
        """
        In place: flow-relaxation sponge on the baroclinic velocity (implicit; the depth mean, and so the
        free-surface fluxes, are untouched — barotropic waves leave through Flather), then Orlanski
        radiation at the open edges.  Weights wu, wv: Δσ (None) or the ALE face fractions (spec §3.2-3.3).
        """
        if self._sp_u is not None:
            for vel, rate, w in ((u_new, self._sp_u, wu), (v_new, self._sp_v, wv)):
                bar, prime = self._baroclinic_part(vel, w)
                vel[...] = bar[None] + prime / (1.0 + self.dt * rate[None])
        for name, vel, old, axis, edge, w in (("west", u_new, st.u, 2, 0, wu), ("east", u_new, st.u, 2, -1, wu),
                                              ("south", v_new, st.v, 1, 0, wv), ("north", v_new, st.v, 1, -1, wv)):
            if name in self.open:
                self._radiate(old, vel, axis, edge, w)
```

In the σ `step`, replace the block from `fl = {}` through the radiation loop:

```python
        fl = {}
        if self.open:                                                   # radiate to a resting exterior (η = 0, ū = 0)
            zu, zv = np.zeros(self.fs.H_u.shape), np.zeros(self.fs.H_v.shape)
            fl = dict(flather_eta=(zu, zv), flather_ubar=zu, flather_vbar=zv)
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv, **fl)
        if self._sp_u is not None:
            # flow-relaxation sponge on the baroclinic velocity (implicit; the depth mean, and so the
            # free-surface volume fluxes, are untouched — barotropic waves leave through Flather)
            for vel, rate in ((u_new, self._sp_u), (v_new, self._sp_v)):
                bar, prime = self._baroclinic_part(vel)
                vel[...] = bar[None] + prime / (1.0 + dt * rate[None])
        if self.open:
            for name, vel, old, axis, edge in (("west", u_new, st.u, 2, 0), ("east", u_new, st.u, 2, -1),
                                                ("south", v_new, st.v, 1, 0), ("north", v_new, st.v, 1, -1)):
                if name in self.open:
                    self._radiate(old, vel, axis, edge)
```
with
```python
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv,
                                               **self._flather_args())
        self._open_velocity(st, u_new, v_new)
```

- [ ] **Step 5: Run the new test and the σ bit-identical check**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k radiation_with_face_weights`
Expected: PASS.

Run:
```bash
NUMBA_NUM_THREADS=5 python - <<'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np
from tests.test_sigma_baroclinic import _channel, _run
m, st = _channel(80, 40e3, open_boundaries=("west", "east"), sponge_cells=10, sponge_tau=(1200.0, 120.0))
st = _run(m, st, 2.0)
d = ".superpowers/sdd/2026-10-12-ale-open-boundaries/"
print(np.array_equal(st.u, np.load(d + "sigma_ref_u.npy")), np.array_equal(st.T, np.load(d + "sigma_ref_T.npy")))
EOF
```
Expected: `True True`. If `False`, the σ arithmetic changed: fix it before going on (spec §2).

- [ ] **Step 6: Run the σ suites**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_baroclinic.py tests/test_sigma_ale.py tests/test_restart.py -q`
Expected: all pass (the existing xfail stays xfail).

- [ ] **Step 7: Commit** (only if authorized)

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py
git commit -m "Open boundaries: weighted baroclinic split, radiation and sponge shared by sigma and ALE (Plan C task 2)"
```

---

### Task 3: Open boundaries in the ALE step

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py`:
  - `__init__` (boundary mask after `_sponge_rates`);
  - `_setup_ale` (remove the Plan C `ValueError`, add the H ≤ 0 check);
  - `initial_state` (store `h_ext`, `T_ext` after the hybrid init);
  - `_open_faces` and `_rhs` (inflow from `_inflow_T()`);
  - new `_inflow_T`, `_exterior`, `_open_face_transports`;
  - `_step_ale` wiring;
  - class docstring.
- Test: `tests/test_sigma_ale.py`: remove the open-boundaries case from `test_ale_configuration_errors`; append the integration tests.

**Interfaces:**
- Consumes:
  - from Task 1: `open_face_transport`, `exterior_on_layers`;
  - from Task 2: `_flather_args`, `_open_velocity(st, u_new, v_new, wu, wv)`.
- Produces:
  - attributes `self.h_ext` (N, ny, nx) or None (σ), `self._bc_mask` (ny, nx) bool, `self._T_in` (None in σ);
  - `_inflow_T() -> np.ndarray` (N, ny, nx);
  - `_exterior(h) -> np.ndarray`;
  - `_open_face_transports(Qx, Qy, h, un, vn, u_new, v_new) -> None` (in place).

- [ ] **Step 1: Write the failing tests**

In `test_ale_configuration_errors`, delete the line:
```python
    (dict(open_boundaries=("west",)), "open boundaries"),
```

Append to `tests/test_sigma_ale.py`:

```python
def _ale_open_box(rule, thickness_flux="ppm", nx=30, ny=1, open_=("west", "east"), **kw):
    """20 m box with open edges, a Gaussian η bump at t = 0 (it leaves through Flather), uniform T = 7."""
    nz = 6
    params = dict(dz_top=np.full(nz, 20.0 / nz))
    if rule == "hybrid":
        # asymmetric margins: the water density must not fall exactly on a target (it would split layers)
        params = dict(dz_top=np.full(nz, 0.5), rho_target=np.linspace(RHO0 - TCOEF * 7.0 - 1.0,
                                                                    RHO0 - TCOEF * 7.0 + 1.5, nz + 1))
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((ny, nx), 20.0), dt=30.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid=rule,
                        regrid_params=params, pgf="fv", open_boundaries=open_, thickness_flux=thickness_flux, **kw)
    x = (np.arange(nx) + 0.5) * 1000.0
    y = (np.arange(ny) + 0.5) * 1000.0
    r2 = (x[None, :] - x.mean()) ** 2 + ((y[:, None] - y.mean()) ** 2 if ny > 1 else 0.0)
    return m, m.initial_state(0.3 * np.exp(-r2 / 3e3**2), np.full((nz, ny, nx), 7.0))


@pytest.mark.parametrize("thickness_flux", ["ppm", "upwind"])                # review focus 5
@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_open_boundaries_keep_a_uniform_tracer_uniform(rule, thickness_flux):
    # spec §5 integration 1: the open-face layer fluxes add up to the Flather flux and inflow is the
    # exterior value, or a uniform tracer would change; review focus 2 (η ≠ 0 at t = 0)
    m, st = _ale_open_box(rule, thickness_flux)
    vol0 = m.layer_volumes(st.eta, st.h).sum()
    for _ in range(120):                                                 # 1 h
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
        assert (st.h >= 0).all()
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert m.layer_volumes(st.eta, st.h).sum() < 0.9 * vol0 + 0.1 * (20.0 * 30e3 * 1e3)   # the bump left
    assert np.abs(st.u[:, 0, [0, -1]]).max() > 0                         # boundary faces carry flow


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_open_boundaries_on_four_edges_keep_a_uniform_tracer_uniform(rule):
    # review focus 3: v-faces (south, north) and a sponge on all four edges in 2-D
    m, st = _ale_open_box(rule, nx=14, ny=12, open_=("west", "east", "south", "north"), sponge_cells=3)
    for _ in range(60):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert np.abs(st.v[:, [0, -1], :]).max() > 0


def test_ale_open_boundaries_with_a_sponge_keep_a_resting_stratification_at_rest():
    # spec §5 integration 2 (z*)
    nz, nx = 10, 40
    hg = create_cartesian_grid(nx=nx, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, nx), 200.0), dt=120.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-5, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 20.0)), pgf="fv", open_boundaries=("west", "east"),
                        sponge_cells=8, sponge_tau=(1200.0, 120.0))
    z = m.layer_heights(np.zeros((1, nx)))
    st = m.initial_state(np.zeros((1, nx)), 10.0 + 0.02 * z)
    for _ in range(200):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-10


def test_ale_open_boundary_next_to_a_drying_cell_is_rejected():
    # spec §4 item 2: drying columns at ALE open boundaries are out of scope
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.linspace(-1.0, 10.0, 8)[None]
    with pytest.raises(ValueError, match="H ≤ 0"):
        SigmaBaroclinic(hg, H, dt=10.0, n_layers=4, z0=None, closure="constant", vertical_coordinate="ale",
                        regrid="sigma", regrid_params={}, open_boundaries=("west",))
```

- [ ] **Step 2: Run them to verify they fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k "ale_open or drying_cell or configuration"`
Expected:
- the 4 + 2 + 1 integration tests FAIL with `ValueError: open boundaries with vertical_coordinate='ale' are not implemented yet (ALE Plan C)`;
- the drying test FAILS: the Plan C message does not contain "H ≤ 0";
- the configuration tests pass.

- [ ] **Step 3: Implement**

(a) Import the two Task 1 functions (sorted):

```python
from core.vertical.ale import (DZ_FLOOR, baroclinic_split, exterior_on_layers, face_fractions, face_thickness,
                               face_transport_thickness, lagrangian_thickness, open_face_transport, ppm_transport_x,
                               upwind_transport_x, zero_thin_bed_velocity)
```

(b) In `__init__`, right after `self._sp_c, self._sp_u, self._sp_v = _sponge_rates(...)` and `self.T_ext = None`, add:

```python
        bc = np.zeros(act.shape, bool)                                   # columns that need the exterior profile
        for edge, sl in (("west", np.s_[:, 0]), ("east", np.s_[:, -1]), ("south", np.s_[0, :]), ("north", np.s_[-1, :])):
            if edge in self.open:
                bc[sl] = True
        if self._sp_c is not None:
            bc |= self._sp_c > 0
        self._bc_mask = bc & act
        self.h_ext, self._T_in = None, None
```

(c) In `_setup_ale`, replace

```python
        if self.open:
            raise ValueError("open boundaries with vertical_coordinate='ale' are not implemented yet (ALE Plan C)")
```
with (H must be read the same way as a few lines below it; place this after the `floodable` computation and before `self.regrid_rule, self.regrid_params = ...`):

```python
        edge_cells = np.zeros_like(act)
        for edge, sl in (("west", np.s_[:, 0]), ("east", np.s_[:, -1]), ("south", np.s_[0, :]), ("north", np.s_[-1, :])):
            if edge in self.open:
                edge_cells[sl] = True
        if (edge_cells & floodable).any():
            raise ValueError("open boundaries next to cells with H ≤ 0 (which can dry) are not supported with "
                             "vertical_coordinate='ale' (spec 2026-10-12-ale-open-boundaries §4)")
```

(d) In `initial_state`, after the hybrid `init_iterations` loop and before `return st`:

```python
        if self.ale:                                                    # exterior profile in z (spec §3.5)
            self.T_ext, self.h_ext = st.T.copy(), st.h.copy()
```

(e) Add after `_open_velocity`:

```python
    def _inflow_T(self) -> np.ndarray:
        """Exterior T on the layers of this step: σ, ``T_ext``; ALE, the z profile remapped to hⁿ (spec §3.5)."""
        return self.T_ext if self._T_in is None else self._T_in

    def _exterior(self, h: np.ndarray) -> np.ndarray:
        """Exterior T profile remapped onto thicknesses h in the boundary/sponge columns (core.vertical.ale)."""
        numba.set_num_threads(_THREADS[0])                                 # parallel remap kernel
        return exterior_on_layers(self.h_ext, self.T_ext, h, self._bc_mask, self.remap_code)

    def _open_face_transports(self, Qx: np.ndarray, Qy: np.ndarray, h: np.ndarray, un: np.ndarray, vn: np.ndarray,
                              u_new: np.ndarray, v_new: np.ndarray) -> None:
        """
        In place: layer fluxes through the open faces with the interior thickness, adding up to the
        Flather flux of the free surface (spec §3.4; core.vertical.ale.open_face_transport).
        """
        th = self.fs.theta
        for edge, Q, D, Qo, L, hc, vo, vn1 in (
                ("west", Qx[..., 0], self.fs.last_open_Du[:, 0], self.fs.last_open_Qu[:, 0], self.fs.Lu[:, 0],
                 h[..., 0], un[..., 0], u_new[..., 0]),
                ("east", Qx[..., -1], self.fs.last_open_Du[:, -1], self.fs.last_open_Qu[:, -1], self.fs.Lu[:, -1],
                 h[..., -1], un[..., -1], u_new[..., -1]),
                ("south", Qy[:, 0, :], self.fs.last_open_Dv[0, :], self.fs.last_open_Qv[0, :], self.fs.Lv[0, :],
                 h[:, 0, :], vn[:, 0, :], v_new[:, 0, :]),
                ("north", Qy[:, -1, :], self.fs.last_open_Dv[-1, :], self.fs.last_open_Qv[-1, :], self.fs.Lv[-1, :],
                 h[:, -1, :], vn[:, -1, :], v_new[:, -1, :])):
            if edge in self.open:
                Q[...] = np.where((D > 0)[None], open_face_transport(hc, vn1, vo, Qo, L, th), Q)
```

(`Qx[..., 0]` and `Qy[:, 0, :]` are views, so `Q[...] =` writes into Qx and Qy.)

(f) In `_open_faces`, replace
```python
        ext = np.stack([self.T_ext, self.T_ext**2])[: c.shape[0]]
```
with
```python
        T_in = self._inflow_T()
        ext = np.stack([T_in, T_in**2])[: c.shape[0]]
```
and in `_rhs`, replace
```python
        ext = c if self.T_ext is None or not self.open else np.ascontiguousarray(
            np.stack([self.T_ext, self.T_ext**2])[: c.shape[0]])
```
with
```python
        T_in = self._inflow_T()
        ext = c if T_in is None or not self.open else np.ascontiguousarray(np.stack([T_in, T_in**2])[: c.shape[0]])
```

(g) In `_step_ale`:

1. Replace
   ```python
           eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv,
                                                  wu=wu, wv=wv, hu=np.maximum(hu, DZ_FLOOR), hv=np.maximum(hv, DZ_FLOOR))
   ```
   with
   ```python
           eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv,
                                                  wu=wu, wv=wv, hu=np.maximum(hu, DZ_FLOOR), hv=np.maximum(hv, DZ_FLOOR),
                                                  **self._flather_args())
           self._open_velocity(st, u_new, v_new, wu, wv)                   # sponge on u′, Orlanski (spec §3.2-3.3)
   ```
2. Right after the `if self.thickness_flux == "ppm": … else: …` block that builds Qx, Qy, add:
   ```python
           if self.open:
               self._open_face_transports(Qx, Qy, st.h, un, vn, u_new, v_new)
   ```
3. Before `C = self._advect(...)`:
   ```python
           self._T_in = self._exterior(st.h) if self.open else None          # inflow T on hⁿ (spec §3.5)
   ```
4. After the `if self.kv > 0:` block and before `h_new, (T_new, T2_new) = self._regrid_remap(...)`:
   ```python
           if self._sp_c is not None and self.sponge_tracer:                 # tracer sponge toward the exterior on h†
               r = dt * self._sp_c[None]
               T_lag = (T_lag + r * self._exterior(h_lag)) / (1.0 + r)
   ```
5. Add to the `_step_ale` docstring: "Open boundaries (spec 2026-10-12-ale-open-boundaries): Flather in the free surface, sponge on u′ and Orlanski with the face fractions, open-face fluxes on the interior thickness, inflow T and tracer sponge from the exterior z profile remapped to hⁿ and h†."

(h) In the class docstring, after `pgf : …`, add:
```
    open_boundaries, sponge_cells, sponge_tau, orlanski_cmax, sponge_tracer : also with vertical_coordinate='ale'
          (Plan C scope A: exterior at rest, T_ext/h_ext kept in z from the initial state; no H ≤ 0 cells at an edge)
```

- [ ] **Step 4: Run the tests to verify they pass**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k "ale_open or drying_cell or configuration"`
Expected: all pass.

If the uniform-tracer test fails only for `hybrid`: check that `initial_state` stores `T_ext`/`h_ext` *after* the `init_iterations` loop. If T drifts by more than 1e-11 at the boundary column, compare `Qx[:, :, 0].sum(0)` with `self.fs.last_open_Qu[:, 0]`; they must agree to round-off. Use superpowers:systematic-debugging; do not loosen the tolerances.

- [ ] **Step 5: Run the whole suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider`
Expected: everything passes (the 2 existing xfails stay xfail). Also re-run the σ bit-identical script of Task 2 Step 5. Expected: `True True`.

- [ ] **Step 6: Commit** (only if authorized)

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py
git commit -m "ALE: open boundaries (Flather, Orlanski and sponge on u', open-face fluxes, exterior profile in z) (Plan C task 3)"
```

---

### Task 4: Mode-1 internal-wave channel with z\* and hybrid; documentation

**Files:**
- Test: `tests/test_sigma_ale.py` (append)
- Modify:
  - `CLAUDE.md` (ALE row of the Phase 2 table, test totals);
  - `docs/design/vision_roadmap.md` §7 (row "Fronteras abiertas");
  - `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` §9 item 5.

**Interfaces:**
- Consumes: the ALE open boundaries of Task 3 (`open_boundaries`, `sponge_cells`, `sponge_tau` with `vertical_coordinate="ale"`).
- Produces: measured interior errors (closed / Orlanski / sponge) for z\* and hybrid, for the docs.

- [ ] **Step 1: Write the tests** (append to `tests/test_sigma_ale.py`)

```python
N_CH, H_CH = 0.02, 200.0


def _ale_channel(rule, nx, x0, dx=1000.0, nz=10, dt=120.0, amp=5.0, **kw):
    """Linear stratification (N = 0.02 s⁻¹) with a mode-1 isopycnal bump at x0, ALE z* or hybrid."""
    dTdz = N_CH**2 * RHO0 / (9.81 * TCOEF)
    params = dict(dz_top=np.full(nz, H_CH / nz))
    if rule == "hybrid":
        rho = RHO0 - TCOEF * (10.0 + dTdz * np.linspace(-H_CH, 0.0, 401))
        params = dict(dz_top=np.full(nz, 1.0), rho_target=rg.target_densities(rho, nz))
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    m = SigmaBaroclinic(hg, np.full((1, nx), H_CH), dt=dt, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-5, vertical_coordinate="ale", regrid=rule,
                        regrid_params=params, pgf="fv", **kw)
    x = (np.arange(nx) + 0.5) * dx
    z = m.layer_heights(np.zeros((1, nx)))
    zeta = amp * np.sin(-np.pi * z / H_CH) * np.exp(-((x - x0) / 5e3) ** 2)[None, None, :]
    return m, m.initial_state(np.zeros((1, nx)), 10.0 + dTdz * (z + zeta))


def _on_levels(m, st, z_levels):
    """T interpolated in z from the layer centres to fixed levels (1, nz_levels, nx): layers move in ALE."""
    zc = m.layer_heights(st.eta, st.h)[:, 0, :]
    T = st.T[:, 0, :]
    return np.stack([np.interp(z_levels, zc[:, i], T[:, i]) for i in range(T.shape[1])], axis=-1)


@functools.lru_cache(maxsize=None)
def _ale_interior_error(rule, open_kw=(), nx_small=60, pad=0):
    """rms interior T difference (on fixed z levels) from a 300 km channel, relative to the initial anomaly,
    after 16 h; also the T range of the small channel over the run (min, max) and its initial range."""
    open_kw = dict(open_kw)
    zl = np.linspace(-195.0, -5.0, 39)
    big, sb = _ale_channel(rule, 300, 150e3)
    small, ss = _ale_channel(rule, nx_small + 2 * pad, (nx_small / 2 + pad) * 1e3, **open_kw)
    T_init = _on_levels(small, ss, zl)[:, pad:pad + nx_small]
    rng0 = (float(ss.T.min()), float(ss.T.max()))
    lo, hi = rng0
    for _ in range(int(round(16 * 3600 / big.dt))):
        sb, ss = big.step(sb), small.step(ss)
        lo, hi = min(lo, float(ss.T.min())), max(hi, float(ss.T.max()))
    sl = slice(150 - nx_small // 2, 150 + nx_small // 2)
    diff = _on_levels(small, ss, zl)[:, pad:pad + nx_small] - _on_levels(big, sb, zl)[:, sl]
    bg = T_init.mean(axis=-1, keepdims=True)
    err = np.sqrt(np.mean(diff**2)) / np.sqrt(np.mean((T_init - bg) ** 2))
    return err, (lo, hi), rng0


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_closed_channel_reflects_the_internal_wave(rule):
    assert _ale_interior_error(rule)[0] > 0.3                             # σ: 0.82


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_orlanski_radiation_lets_most_of_the_internal_wave_out(rule):
    err = _ale_interior_error(rule, (("open_boundaries", ("west", "east")),))[0]
    assert err < 0.2, err                                                # σ: 0.157


@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_ale_sponge_with_radiation_absorbs_the_internal_wave(rule):
    kw = (("open_boundaries", ("west", "east")), ("sponge_cells", 10), ("sponge_tau", (1200.0, 120.0)))
    err, (lo, hi), (lo0, hi0) = _ale_interior_error(rule, kw, pad=10)
    assert err < 0.05, err                                               # σ: 0.015
    span = hi0 - lo0                                                     # review focus 4: no new extrema
    assert lo >= lo0 - 0.01 * span and hi <= hi0 + 0.01 * span, (lo, hi, lo0, hi0)
```

- [ ] **Step 2: Run them and record the measured errors**

Run:
```bash
NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k "ale_closed_channel or ale_orlanski or ale_sponge_with"
NUMBA_NUM_THREADS=5 python - <<'EOF'
import sys; sys.path.insert(0, '.')
import tests.test_sigma_ale as t
from core.dynamics.sigma_baroclinic import set_threads
set_threads(5)
for rule in ("zstar", "hybrid"):
    c = t._ale_interior_error(rule)[0]
    o = t._ale_interior_error(rule, (("open_boundaries", ("west", "east")),))[0]
    s = t._ale_interior_error(rule, (("open_boundaries", ("west", "east")), ("sponge_cells", 10),
                                     ("sponge_tau", (1200.0, 120.0))), pad=10)[0]
    print(f"{rule}: closed {c:.3f}  orlanski {o:.3f}  sponge {s:.3f}")
EOF
```
Expected:
- 6 passed;
- printed errors near σ's 0.82 / 0.157 / 0.015.

If a threshold fails: do not loosen it. Read the measured numbers. If z\* passes and hybrid fails, it is a finding about the hybrid layers at the boundary: record a `Ruling:` with the numbers, mark that one case `xfail(strict=True, reason=<measured value>)`, and report it to Esteban at the end.

- [ ] **Step 3: Update the documentation with the measured numbers**

- `docs/design/vision_roadmap.md` §7, row "Fronteras abiertas: onda interna modo 1…": append to the pyOCEAN column "; ALE (Plan C, 2026-10-12): z\* cerrado `<c>` / Orlanski `<o>` / esponja `<s>`, híbrida `<c>` / `<o>` / `<s>`", with the printed values.
- `CLAUDE.md`, ALE row of the Phase 2 table:
  - replace "closed domains, z0=None, constant ν" with "open boundaries (Flather + Orlanski and sponge on the thickness-weighted u′ + exterior T profile in z; mode-1 channel z\* `<c>`/`<o>`/`<s>`, hybrid …; no H ≤ 0 cells at an edge), z0=None, constant ν";
  - update the test counts (`tests/test_sigma_ale.py` and `tests/test_vertical_ale.py`) and the Phase 2 total from the Task 3 suite run plus Task 4's six tests.
- Parent spec `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` §9, item 5: append " — hecho con alcance A el 2026-10-12 (spec 2026-10-12-ale-open-boundaries); reservorio y nudging pendientes".

- [ ] **Step 4: Run the whole suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider`
Expected: everything passes, plus the existing xfails.

- [ ] **Step 5: Commit** (only if authorized)

```bash
git add tests/test_sigma_ale.py CLAUDE.md docs/design/vision_roadmap.md docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md
git commit -m "ALE open boundaries: mode-1 internal-wave channel with z* and hybrid; docs (Plan C task 4)"
```
