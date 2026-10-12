# BayIW Baroclinic Benchmark Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reproduce Esteban's MITgcm BayIW runs (two-layer and linear stratification, with and without the bay) with the ALE mode of pyOCEAN, in z\* and then hybrid, and compare against the precomputed MITgcm station series, fields and section.

**Architecture:**
- `validation/bayiw_baroclinic/bayiw.py` holds the shared setup: MITgcm inputs, model construction, depth sampling and density targets.
- `run_pyocean.py` runs one case and writes 900-s outputs to the external disk.
- `compare.py` computes the spec §6 metrics against `Output_IWBay/npz_out`.
- Runs are long (≈8 h). Each one runs in the background, one at a time, on the 5 P-cores.

**Tech Stack:** Python 3.12, NumPy 1.26.4, Numba, pytest; `core.dynamics.sigma_baroclinic.SigmaBaroclinic` (ALE, open boundaries).

**Spec:** `docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md`

## Global Constraints

- Never use the 6 efficiency cores: `NUMBA_NUM_THREADS=5` and `set_threads(5)`; one model run at a time (no tests or other runs while a long run is going).
- Large outputs go to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_<case>/<coord>_<bay>/`; nothing large in the repo. Never commit `*.npz`, `log_*.txt` or `pyOCEAN-literature-review-guide.docx`.
- Install nothing; numpy stays 1.26.4.
- Commit only when Esteban authorizes it. Messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn
  ```
- Benchmark rules (CLAUDE.md):
  - classify each difference (a) our bug, (b) missing physics, (c) numerical choice, (d) reference error;
  - never tune to this case;
  - if a criterion fails for reason (a), stop and report to Esteban before spending more runs (spec §5).
- Communicate with Esteban in Spanish; code and docstrings in English.
- MITgcm reference roots:
  - two-layer: `/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan_2layer`;
  - linear: `/Volumes/Esteban_SSD/Maestría/MITgcm/verification/BayIW_Rectan_linear`;
  - series: `/Volumes/ESTEBAN_MIT/MITgcm/verification/Output_IWBay/npz_out`.

## Review Focus

1. **Stations or sampling depths below the bed** (the bay is 129–164 m deep, so 250 m lies under it): sampling must give NaN, not the bottom value or garbage. *Pinned by Task 2 `test_sample_at_depth_is_nan_below_the_bed`.*
2. **Vanished z\* layers in the shallow bay** (zero thickness at the bottom of the column): depth interpolation must skip them, with no division by zero. *Pinned by Task 2 `test_sample_at_depth_is_exact_for_a_linear_profile_with_vanished_layers`.*
3. **Two-layer water densities exactly on a hybrid target** (the 21-layer lock tie): both masses must sit at bracket centres. *Pinned by Task 2 `test_centred_targets_put_both_water_masses_at_bracket_centres`.*
4. **Land columns in the zoom window and section**: NaN in pyOCEAN, masked (0) in MITgcm, never compared as numbers. *Pinned by Task 5 `test_relative_field_error_ignores_land`.*
5. **Reference series start at 27 h** (`tmin_horas` 27) while the zoom fields start at 0 h: the windows must use the right source. *Pinned by Task 5 `test_window_selects_the_requested_hours`, and by loading η and T110 from the zoom files (0–120 h).*

---

## File Structure

- Create:
  - `validation/bayiw_baroclinic/bayiw.py` (constants, MITgcm inputs, `build_model`, `sample_at_depth`, `mitgcm_profile_at`, `centred_targets`, `profile_targets`);
  - `validation/bayiw_baroclinic/run_pyocean.py` (runner);
  - `validation/bayiw_baroclinic/compare.py` (metrics, loaders, report);
  - `validation/bayiw_baroclinic/README.md` (results).
- Create tests: `tests/test_bayiw_baroclinic.py` (pure functions; the disk-dependent ones skip if the volumes are missing).
- Modify:
  - `tests/test_sigma_ale.py` (rotating open-boundary tests);
  - `docs/design/vision_roadmap.md` §7, `CLAUDE.md`, memory `project_status.md` (results).

---

### Task 1: Open boundaries with rotation (Phase 0)

**Files:**
- Modify: `tests/test_sigma_ale.py` (`_ale_open_box` gets an `f0` keyword; append the tests)

**Interfaces:**
- Consumes: `_ale_open_box(rule, thickness_flux="ppm", nx=30, ny=1, open_=("west","east"), **kw)` in `tests/test_sigma_ale.py`. Coriolis enters the model only through `h_grid["f_q"]` (`SigmaBaroclinic._fq`, sigma_baroclinic.py:709), so the grid's `f0` is enough.
- Produces: tests only.

- [ ] **Step 1: Write the tests**

In `_ale_open_box`:
- change the signature to `def _ale_open_box(rule, thickness_flux="ppm", nx=30, ny=1, open_=("west", "east"), f0=0.0, **kw):`;
- change its grid line to `hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=f0)`.

Then append:

```python
# ---------------------------------------------------------------- open boundaries with rotation (BayIW phase 0)
@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_rotating_open_boundaries_keep_a_uniform_tracer_uniform(rule):
    m, st = _ale_open_box(rule, nx=14, ny=12, open_=("west", "east", "south"), f0=1e-4, sponge_cells=3)
    vol0 = m.layer_volumes(st.eta, st.h).sum()
    for _ in range(120):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert np.isfinite(st.u).all() and np.isfinite(st.v).all()
    assert np.abs(st.v).max() > 0                                        # rotation turned the outflow
    assert m.layer_volumes(st.eta, st.h).sum() < vol0


def test_rotating_open_boundaries_keep_a_resting_stratification_at_rest():
    nz, nx, ny = 10, 16, 12
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=1e-4)
    m = SigmaBaroclinic(hg, np.full((ny, nx), 200.0), dt=120.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-5, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 20.0)), pgf="fv",
                        open_boundaries=("west", "east", "south"), sponge_cells=4, sponge_tau=(1200.0, 120.0))
    z = m.layer_heights(np.zeros((ny, nx)))
    st = m.initial_state(np.zeros((ny, nx)), 10.0 + 0.02 * z)
    for _ in range(200):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-10 and np.abs(st.v).max() < 1e-10
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k rotating`
Expected: 3 passed. These tests pin existing behaviour; Plan C should already handle f ≠ 0. If one fails, it is a real defect of the open boundaries with rotation. Use superpowers:systematic-debugging, write the fix with its own failing test, and note it in the ledger.

- [ ] **Step 3: Run the ALE file**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -p no:cacheprovider`
Expected: all pass, plus 1 xfail.

---

### Task 2: Shared setup `bayiw.py` (Phase 0)

**Files:**
- Create: `validation/bayiw_baroclinic/bayiw.py`
- Test: `tests/test_bayiw_baroclinic.py`

**Interfaces:**
- Produces:
  - constants `NX=560`, `NY=352`, `F0`, `RHO0`, `T_COEF`, `DELR` (50,), `Z_F` (51,) surface-first interface heights, `Z_C` (50,) level centres, `Z110 = Z_C[29]`, `Z250 = Z_C[39]`, `STATIONS` {name: (i, j)}, `ZOOM_I0 = 130`, `ZOOM_NI = 300`, `GAMMA_I = 280`, `GAMMA_J0 = 215`, `REF`, `RUNS`, `CASES`;
  - `sample_at_depth(zc, f, h, z0, bed) -> np.ndarray`;
  - `mitgcm_profile_at(z, T_mit) -> np.ndarray`;
  - `centred_targets(rho_a, rho_b, n) -> np.ndarray` (n+1,);
  - `profile_targets(rho, n) -> np.ndarray` (n+1,);
  - `load_inputs(case, bay) -> (dx, dy, H, T_mit, wind)`;
  - `build_model(case, bay, coord="zstar", dt=120.0) -> (m, st, wind)`;
  - `cell_centres(d, origin) -> np.ndarray`.

- [ ] **Step 1: Write the failing tests** (`tests/test_bayiw_baroclinic.py`)

```python
"""
Tests of the BayIW baroclinic benchmark tools (spec docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md).
Pure functions always; the ones reading the MITgcm inputs skip when the external volumes are not mounted.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from validation.bayiw_baroclinic import bayiw

HAVE_2LAYER = (bayiw.CASES["2layer"]["dir"] / "input").exists()
HAVE_REF = bayiw.REF.exists()


def _layers(rng, n=12, H=150.0, vanished=(0, 1)):
    h = rng.uniform(1.0, 20.0, n)
    h[list(vanished)] = 0.0                                              # vanished bed layers (z* in the bay)
    h *= H / h.sum()
    zw = -H + np.concatenate([[0.0], np.cumsum(h)])
    return h, 0.5 * (zw[1:] + zw[:-1])


def test_sample_at_depth_is_exact_for_a_linear_profile_with_vanished_layers():
    # review focus 2
    rng = np.random.default_rng(1)
    h, zc = _layers(rng)
    f = 3.0 + 0.1 * zc
    ok = h > 0
    for z0 in np.linspace(zc[ok].min() + 1e-6, zc.max() - 1e-6, 17):
        out = bayiw.sample_at_depth(zc[:, None], f[:, None], h[:, None], z0, np.array([-150.0]))
        np.testing.assert_allclose(out, 3.0 + 0.1 * z0, rtol=1e-12)


def test_sample_at_depth_is_nan_below_the_bed():
    # review focus 1: 250 m under a 150 m bay
    rng = np.random.default_rng(2)
    h, zc = _layers(rng)
    out = bayiw.sample_at_depth(zc[:, None], zc[:, None], h[:, None], -250.0, np.array([-150.0]))
    assert np.isnan(out).all()


def test_sample_at_depth_is_constant_between_the_bed_and_the_deepest_centre():
    rng = np.random.default_rng(3)
    h, zc = _layers(rng)
    f = 3.0 + 0.1 * zc
    k0 = np.flatnonzero(h > 0)[0]
    out = bayiw.sample_at_depth(zc[:, None], f[:, None], h[:, None], -149.9, np.array([-150.0]))
    np.testing.assert_allclose(out, f[k0], rtol=1e-14)


def test_mitgcm_profile_at_takes_the_level_value():
    T = np.arange(50.0)[:, None, None] * np.ones((1, 2, 2))
    z = np.array([-0.5, -bayiw.Z_C[29], -599.0])[:, None, None] * np.ones((1, 2, 2))
    z = -np.abs(z)
    out = bayiw.mitgcm_profile_at(z, T)
    np.testing.assert_array_equal(out[:, 0, 0], [0.0, 29.0, 49.0])


def test_centred_targets_put_both_water_masses_at_bracket_centres():
    # review focus 3: the tie of the 21-layer lock exchange
    rho_a, rho_b, n = 997.526, 998.376, 25
    t = bayiw.centred_targets(rho_a, rho_b, n)
    assert t.size == n + 1 and np.all(np.diff(t) > 0)
    for rho in (rho_a, rho_b):
        k = np.searchsorted(t, rho) - 1
        np.testing.assert_allclose((rho - t[k]) / (t[k + 1] - t[k]), 0.5, atol=1e-9)


def test_profile_targets_leave_a_margin_beyond_the_extremes():
    rho = np.linspace(1000.0, 1006.0, 50)
    t = bayiw.profile_targets(rho, 25)
    assert t.size == 26 and t[0] < rho.min() and t[-1] > rho.max() and np.all(np.diff(t) > 0)


def test_reference_depths_match_the_mitgcm_series():
    np.testing.assert_allclose([bayiw.Z110, bayiw.Z250], [-111.4, -257.75], atol=1e-9)
    assert abs(bayiw.Z_F[-1] + 599.9) < 1e-9


@pytest.mark.skipif(not HAVE_2LAYER, reason="BayIW MITgcm inputs not mounted")
def test_two_layer_inputs_and_model_shapes():
    dx, dy, H, T, wind = bayiw.load_inputs("2layer", "bay")
    assert dx.shape == (bayiw.NX,) and dy.shape == (bayiw.NY,) and T.shape == (50, bayiw.NY, bayiw.NX)
    assert np.nanmax(H) <= 599.9 + 1e-9 and np.isnan(H).any()
    assert 120.0 < H[345, 280] < 170.0                                   # station Seiche is in the bay
    np.testing.assert_allclose(T[0, 200, 280], 22.77, atol=0.01)
    np.testing.assert_allclose(T[-1, 200, 280], 18.52, atol=0.01)
    assert wind.ndim == 3 and wind.shape[1:] == (bayiw.NY, bayiw.NX)
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_baroclinic.py -q`
Expected: collection fails with `ModuleNotFoundError: No module named 'validation.bayiw_baroclinic'`.

- [ ] **Step 3: Implement `validation/bayiw_baroclinic/bayiw.py`**

```python
"""
Shared setup of the BayIW baroclinic benchmark against MITgcm (spec
docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md): the MITgcm inputs of
Esteban's M.Sc. runs (BayIW_Rectan_2layer, BayIW_Rectan_linear; run_expand/data, data.obcs), the
ALE model built with the same parameters, and depth sampling on moving layers.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid_stretched

NX, NY = 560, 352
F0, RHO0, ALPHA = 6.97e-5, 999.8, 2e-4                  # MITgcm data: f0, rhoNil, tAlpha
T_COEF = ALPHA * RHO0                                   # ρ = ρ₀ − ρ₀α (T − tRef) [kg m⁻³ °C⁻¹]
DELR = np.array([1.0, 1.1, 1.2, 1.3, 1.4, 1.5, 1.6, 1.7, 1.9, 2.0, 2.2, 2.4, 2.6, 2.8, 3.0, 3.2, 3.5, 3.8, 4.1,
                 4.4, 4.8, 5.2, 5.6, 6.0, 6.5, 7.1, 7.6, 8.2, 8.9, 9.6, 10.4, 11.3, 12.2, 13.2, 14.3, 15.4, 16.7,
                 18.0, 19.5, 21.1, 22.8, 24.6, 26.6, 28.8, 31.1, 33.7, 36.4, 39.3, 42.5, 45.8])   # PARM04, surface first [m]
Z_F = -np.concatenate([[0.0], np.cumsum(DELR)])         # MITgcm interface heights, surface first [m]
Z_C = 0.5 * (Z_F[:-1] + Z_F[1:])                        # level centres RC [m]
Z110, Z250 = Z_C[29], Z_C[39]                           # z_ref of the 110 m and 250 m reference series
X0, Y0 = -1400e3, -1621e3                               # xgOrigin, ygOrigin [m]
STATIONS = {"Seiche": (280, 345), "Seiche2": (280, 325), "A": (230, 315), "B": (330, 315),
            "C": (280, 265), "Gamma": (280, 215), "Omega": (280, 315)}           # (i, j), Output_IWBay meta
ZOOM_I0, ZOOM_NI = 130, 300                             # zoom arrays: columns i = 130 … 429, all rows
GAMMA_I, GAMMA_J0 = 280, 215                            # YZ section along i = 280 from Gamma (j = 215 … 351)
REF = Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/Output_IWBay/npz_out")
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs")
CASES = {
    "2layer": dict(dir=Path("/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan_2layer"),
                   temp="T_2layer_50z_560x352.bin", t_ref=11.4),
    "linear": dict(dir=Path("/Volumes/Esteban_SSD/Maestría/MITgcm/verification/BayIW_Rectan_linear"),
                   temp="linear_temp_50zlev_560x352.bin", t_ref=33.6),
}
STEMS = {"bay": "bahia_01_expand", "nobay": "nobahia_01_expand"}


def read_bin(path: Path, shape: tuple[int, ...]) -> np.ndarray:
    """MITgcm binary input: big-endian float64 (readBinaryPrec = 64)."""
    return np.fromfile(path, ">f8").reshape(shape)


def cell_centres(d: np.ndarray, origin: float) -> np.ndarray:
    """Cell-centre coordinates [m] of a stretched axis from its spacings (MITgcm XC/YC)."""
    return origin + np.cumsum(d) - 0.5 * d


def load_inputs(case: str, bay: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    MITgcm inputs: dx (NX,), dy (NY,) [m]; rest depth H (NY, NX) [m], NaN on land, clipped at Σ delR = 599.9 m
    as MITgcm does (intentional, confirmed by Esteban); initial T (50, NY, NX) on the delR levels [°C];
    meridional wind stress records (nrec, NY, NX) at the v-faces [N m⁻²].
    """
    inp = CASES[case]["dir"] / "input"
    stem = STEMS[bay]
    dx = np.fromfile(inp / f"{stem}_dx.bin", ">f8")
    dy = np.fromfile(inp / f"{stem}_dy.bin", ">f8")
    B = read_bin(inp / f"{stem}_bat.bin", (NY, NX))
    H = np.where(B < 0, np.minimum(-B, -Z_F[-1]), np.nan)
    T = read_bin(inp / CASES[case]["temp"], (50, NY, NX))
    wind = read_bin(inp / "merid_local_bay_560x352.bin", (-1, NY, NX))
    return dx, dy, H, T, wind


def mitgcm_profile_at(z: np.ndarray, T_mit: np.ndarray) -> np.ndarray:
    """T of the MITgcm level containing each height z (N, NY, NX) [m]: piecewise constant, so a two-layer step stays sharp."""
    k = np.clip(np.searchsorted(-Z_F, -z, side="right") - 1, 0, DELR.size - 1)
    return np.take_along_axis(T_mit, k, axis=0)


def sample_at_depth(zc: np.ndarray, f: np.ndarray, h: np.ndarray, z0: float, bed: np.ndarray) -> np.ndarray:
    """
    f at height z0 [m] by linear interpolation in z between layer centres zc (N, …), bed first, skipping
    layers thinner than 1e-6 m (vanished z* layers): constant between the bed and the deepest centre and
    above the top centre; NaN below the bed ``bed`` (…) = −H and in columns without water.
    """
    n = zc.shape[0]
    ok = h > 1e-6
    k = np.arange(n).reshape((n,) + (1,) * (zc.ndim - 1))
    kb = np.where(ok & (zc <= z0), k, -1).max(axis=0)
    ka = np.where(ok & (zc > z0), k, n).min(axis=0)
    has_b, has_a = kb >= 0, ka < n
    take = lambda a, i: np.take_along_axis(a, np.clip(i, 0, n - 1)[None], axis=0)[0]   # noqa: E731
    zb, za, fb, fa = take(zc, kb), take(zc, ka), take(f, kb), take(f, ka)
    both = has_b & has_a
    w = np.where(both, (z0 - zb) / np.where(both & (za > zb), za - zb, 1.0), 0.0)
    out = np.where(both, fb + w * (fa - fb), np.where(has_b, fb, fa))
    return np.where(ok.any(axis=0) & (z0 >= bed), out, np.nan)


def centred_targets(rho_a: float, rho_b: float, n: int) -> np.ndarray:
    """
    n + 1 interface target densities with two water masses ρ_a < ρ_b at the centres of brackets 1 and n − 2:
    a density exactly on a target can split a water mass between two layers (21-layer lock exchange).
    """
    s = (rho_b - rho_a) / (n - 3)
    return rho_a - 1.5 * s + s * np.arange(n + 1)


def profile_targets(rho: np.ndarray, n: int) -> np.ndarray:
    """n + 1 targets evenly spaced over the density range, half an interval beyond each extreme."""
    lo, hi = float(np.nanmin(rho)), float(np.nanmax(rho))
    s = (hi - lo) / n
    return np.linspace(lo - 0.5 * s, hi + 0.5 * s, n + 1)


def build_model(case: str, bay: str, coord: str = "zstar", dt: float = 120.0):
    """
    The ALE model with the MITgcm parameters (spec §3): z* on the 50 delR levels, or hybrid with 25 layers
    (z* floor = the first 25 delR, Σ = 74.8 m; isopycnal below).  Returns (model, initial state, wind records).
    """
    dx, dy, H, T_mit, wind = load_inputs(case, bay)
    hg = create_cartesian_grid_stretched(dx, dy, x0=X0, y0=Y0, f0=F0)
    t_ref = CASES[case]["t_ref"]
    if coord == "zstar":
        nz, params = DELR.size, dict(dz_top=DELR.copy())
    elif coord == "hybrid":
        nz = 25
        rho = RHO0 - T_COEF * (T_mit[:, np.isfinite(H)] - t_ref)
        tgt = (centred_targets(float(rho.min()), float(rho.max()), nz) if case == "2layer"
               else profile_targets(rho, nz))
        params = dict(dz_top=DELR[:nz].copy(), rho_target=tgt)
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, rho0=RHO0, t_coef=T_COEF, t_ref=t_ref, z0=None,
                        closure="constant", nu_const=1e-5, ah=100.0, kh=100.0, kv=1e-5, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params,
                        open_boundaries=("south", "west", "east"), orlanski_cmax=0.45, sponge_cells=30,
                        sponge_tau=(1200.0, 60.0), sponge_tracer=False)
    z = m.layer_heights(np.zeros(H.shape))
    st = m.initial_state(np.zeros(H.shape), mitgcm_profile_at(z, T_mit))
    return m, st, wind
```

- [ ] **Step 4: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_baroclinic.py -q`
Expected: 8 passed (the inputs test runs when `/Volumes/ESTEBAN_MIT` is mounted).

- [ ] **Step 5: Smoke-build both coordinates**

Run:
```bash
NUMBA_NUM_THREADS=5 python - <<'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np
from validation.bayiw_baroclinic import bayiw
from core.dynamics.sigma_baroclinic import set_threads
set_threads(5)
for coord in ("zstar", "hybrid"):
    m, st, w = bayiw.build_model("2layer", "bay", coord)
    col = st.h[:, 200, 280]
    print(coord, st.h.shape, "Σh", round(col.sum(), 6), "T top/bottom", st.T[-1, 200, 280], st.T[0, 200, 280],
          "thick layers", int((col > 1).sum()))
EOF
```
Expected:
- z\*: shape (50, 352, 560), Σh = 599.9, T top 22.77 / bottom 18.52.
- hybrid: shape (25, …), Σh = 599.9, a few thick layers (z\* floor + the two isopycnal water masses), T top ≈ 22.77, bottom ≈ 18.52.

---

### Task 3: Runner `run_pyocean.py` and a 1-hour smoke run (Phase 0)

**Files:**
- Create: `validation/bayiw_baroclinic/run_pyocean.py`

**Interfaces:**
- Consumes: `bayiw.build_model`, `sample_at_depth`, `STATIONS`, `Z110`, `Z250`, `Z_C`, `ZOOM_I0`, `ZOOM_NI`, `GAMMA_I`, `GAMMA_J0`, `RUNS`; `validation.bayiw_seiche.run_pyocean.wind_at(t, records)`.
- Produces: `<out>/series.npz` with keys:
  - `t_h` (n,);
  - `eta`, `T110`, `T250`, `U110`, `V110` (n, 7), stations in `STATION_ORDER`;
  - `T110_zoom` (n, 352, 300) float32;
  - `gamma` (n, 50, 137) float32 on `Z_C`;
  - `n_trunc`, `umax`, `heat` (n,);
  - `rpe_t_h`, `rpe` (every 6 h);
  - `meta_json`.

  Also `<out>/state_<hhh.h>h.npz` every 6 h, and `<out>/log.txt`.

- [ ] **Step 1: Implement**

```python
#!/usr/bin/env python
"""
BayIW baroclinic run with the ALE mode of pyOCEAN, matching Esteban's MITgcm runs (spec
docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md §3–4).  Every 900 s: η, T at
110/250 m and U, V at 110 m at the 7 stations; T at 110 m in the MITgcm zoom window; the YZ section of
T along i = 280 from Gamma on the MITgcm level centres; health diagnostics.  A plain state every 6 h
(manual restart; the bit-for-bit ALE restart is Plan D).

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py 2layer bay [--coord zstar]
           [--dt 120] [--hours 120] [--out DIR]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402

STATION_ORDER = tuple(bayiw.STATIONS)


def face_sample(m, st, z0: float, axis: str) -> np.ndarray:
    """U (axis 'u', west face of cell i) or V ('v', south face of cell j) at height z0 at the stations."""
    zc = m.layer_heights(st.eta, st.h)
    H = np.nan_to_num(np.asarray(m.H, float), nan=-np.inf)
    out = []
    for i, j in bayiw.STATIONS.values():
        (j0, i0), (j1, i1) = ((j, i - 1), (j, i)) if axis == "u" else ((j - 1, i), (j, i))
        vel = st.u[:, j, i] if axis == "u" else st.v[:, j, i]
        zf = 0.5 * (zc[:, j0, i0] + zc[:, j1, i1])
        hf = 0.5 * (st.h[:, j0, i0] + st.h[:, j1, i1])
        bed = -min(H[j0, i0], H[j1, i1])
        out.append(bayiw.sample_at_depth(zf, vel, hf, z0, np.asarray(bed)))
    return np.array(out, float)


def record(m, st, buf: dict) -> None:
    """Append the 900-s outputs of state st to buf."""
    zc = m.layer_heights(st.eta, st.h)
    bed = -np.asarray(m.H, float)                                       # NaN on land → NaN samples
    ij = [(j, i) for i, j in bayiw.STATIONS.values()]
    jj, ii = np.array([p[0] for p in ij]), np.array([p[1] for p in ij])
    buf["eta"].append(st.eta[jj, ii])
    for name, z0 in (("T110", bayiw.Z110), ("T250", bayiw.Z250)):
        buf[name].append(bayiw.sample_at_depth(zc[:, jj, ii], st.T[:, jj, ii], st.h[:, jj, ii], z0, bed[jj, ii]))
    buf["U110"].append(face_sample(m, st, bayiw.Z110, "u"))
    buf["V110"].append(face_sample(m, st, bayiw.Z110, "v"))
    sl = np.s_[:, :, bayiw.ZOOM_I0:bayiw.ZOOM_I0 + bayiw.ZOOM_NI]
    buf["T110_zoom"].append(bayiw.sample_at_depth(zc[sl], st.T[sl], st.h[sl], bayiw.Z110,
                                                  bed[:, bayiw.ZOOM_I0:bayiw.ZOOM_I0 + bayiw.ZOOM_NI]).astype(np.float32))
    js = np.s_[:, bayiw.GAMMA_J0:, bayiw.GAMMA_I]
    buf["gamma"].append(np.array([bayiw.sample_at_depth(zc[js], st.T[js], st.h[js], z0, bed[bayiw.GAMMA_J0:, bayiw.GAMMA_I])
                                  for z0 in bayiw.Z_C], np.float32))
    buf["n_trunc"].append(st.n_trunc)
    buf["umax"].append(float(max(np.nanmax(np.abs(st.u)), np.nanmax(np.abs(st.v)))))
    buf["heat"].append(m.tracer_content(st))


def save(out: Path, buf: dict, meta: dict) -> None:
    np.savez(out / "series.npz", **{k: np.array(v) for k, v in buf.items()}, meta_json=json.dumps(meta))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=tuple(bayiw.CASES))
    ap.add_argument("bay", choices=("bay", "nobay"))
    ap.add_argument("--coord", default="zstar", choices=("zstar", "hybrid"))
    ap.add_argument("--dt", type=float, default=120.0)
    ap.add_argument("--hours", type=float, default=120.0)
    ap.add_argument("--out", type=Path, default=None)
    a = ap.parse_args()
    out = a.out or bayiw.RUNS / f"bayiw_{a.case}" / f"{a.coord}_{a.bay}"
    out.mkdir(parents=True, exist_ok=True)
    set_threads(5)
    m, st, wind = bayiw.build_model(a.case, a.bay, a.coord, a.dt)
    meta = dict(vars(a), out=str(out), stations=bayiw.STATIONS, order=STATION_ORDER, z110=bayiw.Z110,
                z250=bayiw.Z250, zoom_i0=bayiw.ZOOM_I0, gamma=(bayiw.GAMMA_I, bayiw.GAMMA_J0))
    buf = {k: [] for k in ("t_h", "eta", "T110", "T250", "U110", "V110", "T110_zoom", "gamma", "n_trunc", "umax",
                           "heat", "rpe_t_h", "rpe")}
    buf["t_h"].append(0.0)
    record(m, st, buf)
    buf["rpe_t_h"].append(0.0)
    buf["rpe"].append(m.rpe(st))
    every, snap = int(round(900.0 / a.dt)), int(round(6 * 3600.0 / a.dt))
    nsteps = int(round(a.hours * 3600.0 / a.dt))
    tau = np.zeros((bayiw.NY + 1, bayiw.NX))
    tic = time.time()
    with open(out / "log.txt", "a") as log:
        for n in range(nsteps):
            tau[:-1, :] = wind_at(n * a.dt, wind)                          # v-face j = south of cell j
            st = m.step(st, tau_y=tau)
            if (n + 1) % every == 0:
                buf["t_h"].append((n + 1) * a.dt / 3600.0)
                record(m, st, buf)
            if (n + 1) % snap == 0 or n + 1 == nsteps:
                t_h = (n + 1) * a.dt / 3600.0
                buf["rpe_t_h"].append(t_h)
                buf["rpe"].append(m.rpe(st))
                np.savez(out / f"state_{t_h:05.1f}h.npz", eta=st.eta, h=st.h, T=st.T, u=st.u, v=st.v, time=st.time)
                save(out, buf, meta)
                msg = (f"{a.case} {a.bay} {a.coord}: t = {t_h:6.1f} h  umax {buf['umax'][-1]:.3f} m/s  "
                       f"trunc {st.n_trunc}  ({(time.time() - tic) / 60:.1f} min)")
                print(msg, flush=True)
                log.write(msg + "\n")
                log.flush()
    save(out, buf, meta)
    print("saved", out / "series.npz")


if __name__ == "__main__":
    main()
```

- [ ] **Step 2: Smoke run (1 hour, dt 120)**

Run:
```bash
NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py 2layer bay --hours 1 --out /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/smoke
python - <<'EOF'
import numpy as np
d = np.load("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/smoke/series.npz")
for k in d.files:
    if k != "meta_json":
        print(k, d[k].shape, np.nanmin(d[k]) if d[k].size else None, np.nanmax(d[k]) if d[k].size else None)
EOF
```
Expected:
- `t_h` (5,) from 0 to 1;
- `eta`/`T110`/`U110`/`V110` (5, 7) finite (T110 near 22.77);
- `T250` NaN at the bay stations (Seiche, Seiche2, Omega: the bay is shallower than 257.75 m) and finite elsewhere;
- `T110_zoom` (5, 352, 300) with NaN on land;
- `gamma` (5, 50, 137);
- `n_trunc` 0.

Each step costs ≈8–9 s, so the run takes ≈5 min.

- [ ] **Step 3: Delete the smoke output**

Run: `rm -r /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/smoke` (scratch output written by this task, no other content).

---

### Task 4: Time-step check (Phase 0)

**Files:** none (decision recorded in the ledger and later in the README).

**Interfaces:**
- Consumes: the runner (Task 3).
- Produces: the decision `DT = 120` or `DT = 60` used by Tasks 6–8.

- [ ] **Step 1: Run 12 h at dt 60 and at dt 120** (sequentially, in the background; ≈3.5 h in total)

```bash
NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py 2layer bay --hours 12 --dt 120 --out /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/dtcheck_120 \
 && NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py 2layer bay --hours 12 --dt 60 --out /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/dtcheck_60
```

- [ ] **Step 2: Compare**

```bash
python - <<'EOF'
import numpy as np
R = "/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/"
a, b = np.load(R + "dtcheck_120/series.npz"), np.load(R + "dtcheck_60/series.npz")
for k in ("eta", "T110"):
    x, y = a[k], b[k]
    sig = (y - y[:1]) if k == "T110" else y
    rel = np.sqrt(np.nanmean((x - y) ** 2)) / np.sqrt(np.nanmean((sig - np.nanmean(sig, 0)) ** 2))
    print(k, "rms diff / rms signal = %.3f" % rel)
EOF
```
Expected: two numbers. If both are < 0.05: `Ruling: DT = 120 s`. Otherwise `DT = 60 s`. Write the ruling, with the two numbers, in the ledger.

---

### Task 5: Comparison tools `compare.py`

**Files:**
- Create: `validation/bayiw_baroclinic/compare.py`
- Test: `tests/test_bayiw_baroclinic.py` (append)

**Interfaces:**
- Consumes: `bayiw` constants; the runner output keys (Task 3).
- Produces:
  - `dominant_period(t_h, x, pmin=1.0, pmax=30.0) -> float`;
  - `rms(x) -> float`;
  - `corr_lag(t_h, a, b, max_lag_h=3.0) -> (float, float, float)`;
  - `window(t_h, x, t0, t1) -> (t, x)`;
  - `relative_field_error(a, b) -> float`;
  - `interface_thickness(T_sec, z, t_lo, t_hi) -> float`;
  - `mitgcm_stations(case, var) -> (t_h, {station: (bay, nobay)})`;
  - `pyocean(case, coord, bay) -> dict`;
  - `report(case, coord) -> list[str]`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_bayiw_baroclinic.py`)

```python
from validation.bayiw_baroclinic import compare  # noqa: E402


def test_dominant_period_finds_a_sine():
    t = np.arange(0.0, 60.0, 0.25)
    x = 0.3 * np.sin(2 * np.pi * t / 7.3) + 0.05 * t
    assert abs(compare.dominant_period(t, x) / 7.3 - 1) < 0.01


def test_corr_lag_finds_a_one_hour_delay():
    t = np.arange(0.0, 48.0, 0.25)
    a = np.sin(2 * np.pi * t / 12.0)
    b = np.sin(2 * np.pi * (t - 1.0) / 12.0)
    c0, lag, c = compare.corr_lag(t, a, b)
    assert abs(lag - 1.0) < 1e-9 and c > 0.999 and c0 < c


def test_window_selects_the_requested_hours():
    # review focus 5
    t = np.arange(27.0, 120.25, 0.25)
    tt, xx = compare.window(t, t * 2, 0.0, 60.0)
    assert tt.min() == 27.0 and tt.max() < 60.0 and np.array_equal(xx, tt * 2)


def test_relative_field_error_ignores_land():
    # review focus 4: NaN land in pyOCEAN, masked (NaN after loading) in MITgcm
    a = np.array([[1.0, np.nan], [2.0, 3.0]])
    b = np.array([[1.0, 5.0], [np.nan, 3.0]])
    assert compare.relative_field_error(a, b) == 0.0


def test_interface_thickness_of_a_tanh_profile():
    z = -np.linspace(1.0, 600.0, 600)
    T = 20.65 + 2.125 * np.tanh((z + 116.0) / 10.0)
    # 10 % → 90 % of ΔT of a tanh: 2·atanh(0.8)·L = 21.97 m for L = 10 m
    assert abs(compare.interface_thickness(T[:, None], z, 18.525, 22.775) - 21.97) < 0.5
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_baroclinic.py -q`
Expected: collection error `ImportError: cannot import name 'compare'`.

- [ ] **Step 3: Implement `validation/bayiw_baroclinic/compare.py`**

```python
#!/usr/bin/env python
"""
Compare pyOCEAN BayIW baroclinic runs with the MITgcm references (spec
docs/superpowers/specs/2026-10-12-bayiw-baroclinic-benchmark-design.md §6).

MITgcm: η and T at 110 m from the zoom fields (0–120 h, columns i = 130 … 429); T at 250 m and U, V at
110 m from the station series (27–120 h); the Gamma YZ section.  pyOCEAN: <RUNS>/bayiw_<case>/<coord>_<bay>/series.npz.

Usage: python validation/bayiw_baroclinic/compare.py 2layer zstar
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_seiche.compare import efold_hours  # noqa: E402

STATION_ORDER = tuple(bayiw.STATIONS)
IW_STATIONS = ("Seiche", "Seiche2", "Omega", "C")                       # spec §6: correlation ≥ 0.7 here
SNAP_H = (6.0, 24.0, 60.0, 90.0)


def window(t_h: np.ndarray, x: np.ndarray, t0: float, t1: float) -> tuple[np.ndarray, np.ndarray]:
    """Samples with t0 ≤ t < t1 (time on the first axis)."""
    m = (t_h >= t0) & (t_h < t1)
    return t_h[m], x[m]


def rms(x: np.ndarray) -> float:
    """rms of the departure from the mean (NaNs ignored)."""
    return float(np.sqrt(np.nanmean((x - np.nanmean(x)) ** 2)))


def dominant_period(t_h: np.ndarray, x: np.ndarray, pmin: float = 1.0, pmax: float = 30.0) -> float:
    """Period [h] of the spectral peak between pmin and pmax: linear detrend, Hann window, 8× zero padding."""
    dt = float(np.median(np.diff(t_h)))
    y = x - np.polyval(np.polyfit(t_h, x, 1), t_h)
    y = y * np.hanning(y.size)
    nfft = 8 * y.size
    p = np.abs(np.fft.rfft(y, nfft)) ** 2
    f = np.fft.rfftfreq(nfft, dt)
    band = (f >= 1.0 / pmax) & (f <= 1.0 / pmin)
    return float(1.0 / f[band][np.argmax(p[band])])


def corr_lag(t_h: np.ndarray, a: np.ndarray, b: np.ndarray, max_lag_h: float = 3.0) -> tuple[float, float, float]:
    """(correlation at zero lag, lag [h] of b behind a with the best correlation, that correlation)."""
    dt = float(np.median(np.diff(t_h)))
    ok = np.isfinite(a) & np.isfinite(b)
    a, b = a[ok] - a[ok].mean(), b[ok] - b[ok].mean()
    c0 = float(np.corrcoef(a, b)[0, 1])
    best = (c0, 0.0)
    for s in range(1, int(round(max_lag_h / dt)) + 1):
        for lag, (x, y) in ((s * dt, (a[:-s], b[s:])), (-s * dt, (a[s:], b[:-s]))):
            c = float(np.corrcoef(x, y)[0, 1])
            if c > best[0]:
                best = (c, lag)
    return c0, best[1], best[0]


def relative_field_error(a: np.ndarray, b: np.ndarray) -> float:
    """rms(a − b) / rms(b) over the points where both are finite (land is NaN in both)."""
    ok = np.isfinite(a) & np.isfinite(b)
    den = np.sqrt(np.mean(b[ok] ** 2))
    return float(np.sqrt(np.mean((a[ok] - b[ok]) ** 2)) / den) if den > 0 else 0.0


def interface_thickness(T_sec: np.ndarray, z: np.ndarray, t_lo: float, t_hi: float) -> float:
    """
    Mean thickness [m] between the 10 % and 90 % isotherms of ΔT = t_hi − t_lo over the columns of the section
    T_sec (nz, ncol) on heights z (nz,), where both are found.
    """
    out = []
    for col in T_sec.T:
        ok = np.isfinite(col)
        if ok.sum() < 3:
            continue
        zz, tt = z[ok], col[ok]
        o = np.argsort(tt)
        lo, hi = t_lo + 0.1 * (t_hi - t_lo), t_lo + 0.9 * (t_hi - t_lo)
        if tt.min() > lo or tt.max() < hi:
            continue
        out.append(abs(np.interp(hi, tt[o], zz[o]) - np.interp(lo, tt[o], zz[o])))
    return float(np.mean(out)) if out else float("nan")


def mitgcm_stations(case: str, var: str) -> tuple[np.ndarray, dict]:
    """
    MITgcm series at the stations: var 'eta' / 'T110' from the zoom fields (0–120 h, land 0 → NaN),
    'T250' / 'U110' / 'V110' from the station series (27–120 h).  Returns (t_h, {station: (bay, nobay)}).
    """
    if var in ("eta", "T110"):
        stem = {"eta": "Eta_zoom_sfc", "T110": "Temp_zoom_z110m"}[var]
        out, t = {}, None
        for src in ("bay", "nobay"):
            d = np.load(bayiw.REF / f"{case}_{stem}_{src}.npz")
            t = d["t_horas"]
            for name, (i, j) in bayiw.STATIONS.items():
                x = d["data"][:, j, i - bayiw.ZOOM_I0].astype(float)
                out.setdefault(name, []).append(np.where(d["mask_zoom"][j, i - bayiw.ZOOM_I0] > 0, x, np.nan))
        return t, {k: tuple(v) for k, v in out.items()}
    f = {"T250": "Temp_T250m_series", "U110": "UV_UV110m_series", "V110": "UV_UV110m_series"}[var]
    key = {"T250": "T250m", "U110": "U110m", "V110": "V110m"}[var]
    d = np.load(bayiw.REF / f"{case}_{f}.npz")
    return d["Seiche_t_horas"], {k: (d[f"{k}_{key}_bay"], d[f"{k}_{key}_nobay"]) for k in bayiw.STATIONS}


def pyocean(case: str, coord: str, bay: str) -> dict:
    """The runner output of one pyOCEAN run as a dict of arrays."""
    d = np.load(bayiw.RUNS / f"bayiw_{case}" / f"{coord}_{bay}" / "series.npz")
    return {k: d[k] for k in d.files}


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def report(case: str, coord: str) -> list[str]:
    """The spec §6 tables for one case and coordinate, as text lines (also printed)."""
    L = [f"# BayIW {case}, pyOCEAN {coord} vs MITgcm"]
    py = {b: pyocean(case, coord, b) for b in ("bay", "nobay")}
    tp = py["bay"]["t_h"]
    col = {s: k for k, s in enumerate(STATION_ORDER)}
    # 1. seiche Δη = η_bay − η_nobay, 0–27 h
    tm, em = mitgcm_stations(case, "eta")
    L.append("\n## Seiche Δη (0–27 h): period [h], range [mm], corr")
    for s in ("Seiche", "Seiche2"):
        dm = em[s][0] - em[s][1]
        dp = py["bay"]["eta"][:, col[s]] - py["nobay"]["eta"][:, col[s]]
        tw, xm = window(tm, dm, 0.0, 27.0)
        _, xp = window(tp, dp, 0.0, 27.0)
        xp_i = np.interp(tw, tp, dp)
        Pm, Pp = dominant_period(tw, xm, 1.0, 10.0), dominant_period(tw, xp_i, 1.0, 10.0)
        Rm, Rp = np.nanmax(xm) - np.nanmin(xm), np.nanmax(xp) - np.nanmin(xp)
        c = corr_lag(tw, xm, xp_i, 0.0)[0]
        L.append(f"{s:8s} MIT {Pm:5.2f} h {1e3 * Rm:6.2f} mm | py {Pp:5.2f} h {1e3 * Rp:6.2f} mm | corr {c:5.2f} | "
                 f"e-fold MIT {efold_hours(tw, xm):5.1f} py {efold_hours(tw, xp_i):5.1f} h | "
                 f"{_flag(abs(Pp / Pm - 1) <= 0.02 and abs(Rp / Rm - 1) <= 0.15 and c >= 0.9)}")
    # 2. internal waves: T anomaly at 110 m, bay and nobay runs, forced and free windows
    tm, Tm = mitgcm_stations(case, "T110")
    for wname, (t0, t1) in (("forced 0–60 h", (0.0, 60.0)), ("free 60–120 h", (60.0, 120.25))):
        L.append(f"\n## T anomaly at 110 m, {wname}: period MIT/py [h], rms py/MIT, corr, best lag [h]")
        for b, kb in (("bay", 0), ("nobay", 1)):
            for s in STATION_ORDER:
                xm_full = Tm[s][kb] - Tm[s][kb][0]
                xp_full = py[b]["T110"][:, col[s]] - py[b]["T110"][0, col[s]]
                tw, xm = window(tm, xm_full, t0, t1)
                xp = np.interp(tw, tp, xp_full)
                if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                    continue
                Pm, Pp = dominant_period(tw, xm), dominant_period(tw, xp)
                ratio = rms(xp) / rms(xm)
                c0, lag, c = corr_lag(tw, xm, xp)
                crit = abs(Pp / Pm - 1) <= 0.05 and abs(ratio - 1) <= 0.25
                if t0 == 0.0 and s in IW_STATIONS:
                    crit = crit and c0 >= 0.7
                L.append(f"{b:5s} {s:8s} {Pm:6.2f} {Pp:6.2f} | {ratio:5.2f} | {c0:5.2f} | {lag:+5.2f} ({c:4.2f}) | {_flag(crit)}")
    # bay effect ΔT110 = bay − nobay
    L.append("\n## Bay effect ΔT110 (bay − nobay), 0–60 h: rms py/MIT, corr")
    for s in STATION_ORDER:
        dm = (Tm[s][0] - Tm[s][0][0]) - (Tm[s][1] - Tm[s][1][0])
        dp = (py["bay"]["T110"][:, col[s]] - py["bay"]["T110"][0, col[s]]) - \
             (py["nobay"]["T110"][:, col[s]] - py["nobay"]["T110"][0, col[s]])
        tw, xm = window(tm, dm, 0.0, 60.0)
        xp = np.interp(tw, tp, dp)
        if not np.isfinite(xm).all() or rms(xm) == 0:
            continue
        ratio, c = rms(xp) / rms(xm), corr_lag(tw, xm, xp, 0.0)[0]
        L.append(f"{s:8s} {ratio:5.2f} | {c:5.2f} | {_flag(1 / 3 <= ratio <= 3 and c > 0)}")
    # 3. T250, U110, V110 from the series (27–120 h): rms ratio and corr
    for var in ("T250", "U110", "V110"):
        tm3, X = mitgcm_stations(case, var)
        L.append(f"\n## {var} (27–120 h): rms py/MIT, corr   [bay | nobay]")
        for s in STATION_ORDER:
            row = []
            for b, kb in (("bay", 0), ("nobay", 1)):
                xm = X[s][kb] - (X[s][kb][0] if var == "T250" else 0.0)
                xp_full = py[b][var][:, col[s]]
                xp = np.interp(tm3, tp, xp_full - (np.interp(tm3[0], tp, xp_full) if var == "T250" else 0.0))
                if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                    row.append("   —   ")
                    continue
                row.append(f"{rms(xp) / rms(xm):4.2f} {corr_lag(tm3, xm, xp, 0.0)[0]:+4.2f}")
            L.append(f"{s:8s} " + " | ".join(row))
    # 4. spatial: T110 zoom anomaly, relative error at the snapshot times (bay run)
    L.append("\n## T110 zoom anomaly (bay): relative rms error at " + ", ".join(f"{h:g} h" for h in SNAP_H))
    zm = np.load(bayiw.REF / f"{case}_Temp_zoom_z110m_bay.npz")
    zmask = zm["mask_zoom"] > 0
    errs = []
    for h in SNAP_H:
        km, kp = int(np.argmin(np.abs(zm["t_horas"] - h))), int(np.argmin(np.abs(tp - h)))
        am = np.where(zmask, zm["data"][km] - zm["data"][0], np.nan)
        ap = py["bay"]["T110_zoom"][kp] - py["bay"]["T110_zoom"][0]
        errs.append(relative_field_error(ap, am))
    L.append("   " + "  ".join(f"{e:5.2f}" for e in errs))
    # 5. health and the interface (two-layer)
    for b in ("bay", "nobay"):
        p = py[b]
        L.append(f"\n## Health {b}: trunc {int(p['n_trunc'][-1])}, max|u| {np.nanmax(p['umax']):.3f} m/s, "
                 f"heat change {p['heat'][-1] / p['heat'][0] - 1:+.2e}, RPE gain {p['rpe'][-1] - p['rpe'][0]:.3e} J")
    if case == "2layer":
        g = np.load(bayiw.REF / f"{case}_secThetaAnom_yz_Gamma.npz", allow_pickle=True)
        L.append("\n## Interface thickness (10–90 % of ΔT) along the Gamma section [m]: MIT | py")
        for h in SNAP_H:
            km, kp = int(np.argmin(np.abs(g["t_horas"] - h))), int(np.argmin(np.abs(tp - h)))
            Tm_sec = np.where(g["mask2d"], g["anom"][km].astype(float), np.nan)
            Tm_sec = np.where(Tm_sec == 0, np.nan, Tm_sec)
            Tp_sec = py["bay"]["gamma"][kp].astype(float)
            L.append(f"{h:5.0f} h  {interface_thickness(Tm_sec, -np.abs(g['RC']), 18.52, 22.77):6.1f} | "
                     f"{interface_thickness(Tp_sec, bayiw.Z_C, 18.52, 22.77):6.1f}")
    print("\n".join(L))
    return L


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2])
```

- [ ] **Step 4: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_baroclinic.py -q`
Expected: 13 passed.

- [ ] **Step 5: Check the MITgcm loaders on the real files**

Run:
```bash
python - <<'EOF'
import sys; sys.path.insert(0, '.')
import numpy as np
from validation.bayiw_baroclinic import compare
for var in ("eta", "T110", "T250", "U110"):
    t, d = compare.mitgcm_stations("2layer", var)
    print(var, t[0], t[-1], t.size, {k: (round(float(np.nanstd(v[0])), 5), round(float(np.nanstd(v[1])), 5)) for k, v in list(d.items())[:3]})
EOF
```
Expected:
- eta and T110 run from 0.0 to 120.0 h (481 samples); T250 and U110 run from 27.0 to 120.0 h (373 samples);
- standard deviations are finite and non-zero. T250 is NaN or zero in the shallow bay; if that is the reference's convention, record a ruling and skip those stations.

The `mask2d` / zero handling of the Gamma section is checked against real data in Task 6.

---

### Task 6: Phase 1 — two-layer z\*, bay and nobay

**Files:**
- Create: `validation/bayiw_baroclinic/README.md`

**Interfaces:**
- Consumes: runner (Task 3), DT (Task 4 ruling), `compare.report` (Task 5).
- Produces: `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/zstar_{bay,nobay}/series.npz`; README section "Dos capas, z\*".

- [ ] **Step 1: Run both cases** (background, one after the other; ≈8.4 h each at DT = 120)

```bash
NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py 2layer bay --dt $DT \
 && NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py 2layer nobay --dt $DT
```
Expected: each log ends at t = 120.0 h with trunc 0. If a run raises (CFL truncations, emptied layer, NaN), stop: debug with superpowers:systematic-debugging, classify the cause, and report to Esteban if it is (a).

- [ ] **Step 2: Compare**

Run: `python validation/bayiw_baroclinic/compare.py 2layer zstar | tee /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/compare_zstar.txt`
Expected: the tables of spec §6, with PASS or FAIL per criterion.

Fix any loader issue first (shapes, masks, sign conventions of the Gamma section). It must be visible in the numbers, for example an MITgcm rms of 0 or a correlation of NaN. Record each such fix as a ruling.

- [ ] **Step 3: Classify and write the README section**

Create `validation/bayiw_baroclinic/README.md` (Spanish, CLAUDE.md style):
- title;
- purpose and the link to the spec;
- configuration table (spec §3) and DT ruling;
- section "Dos capas, z\*" with:
  - the seiche, internal-wave (forced and free), bay-effect, T250/UV, spatial-error, health and interface tables;
  - for every FAIL, its class (a)–(d) and reasoning;
- "Qué debe mejorar pyOCEAN" (running list).

- [ ] **Step 4: Gate (spec §5)**

If any FAIL is class (a), stop here. Report to Esteban in Spanish with the numbers and the diagnosis, and wait. Otherwise continue to Task 7.

---

### Task 7: Phase 2 — linear z\*, bay and nobay

**Files:**
- Modify: `validation/bayiw_baroclinic/README.md` (section "Lineal, z\*")

- [ ] **Step 1: Check the linear volume is mounted**

Run: `ls "/Volumes/Esteban_SSD/Maestría/MITgcm/verification/BayIW_Rectan_linear/input/linear_temp_50zlev_560x352.bin"`
Expected: the file is listed. If it is not, stop and ask Esteban to mount `Esteban_SSD`.

- [ ] **Step 2: Run both cases** (background, sequential)

```bash
NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py linear bay --dt $DT \
 && NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py linear nobay --dt $DT
```
Expected: logs end at 120 h, trunc 0.

- [ ] **Step 3: Compare and write the README section**

Run: `python validation/bayiw_baroclinic/compare.py linear zstar | tee /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_linear/compare_zstar.txt`.

Write "Lineal, z\*" with the same tables and classification as Task 6. Same gate as Task 6 Step 4.

---

### Task 8: Phase 3 — hybrid, both cases

**Files:**
- Modify: `validation/bayiw_baroclinic/README.md` (sections "Híbrida" and "z\* contra híbrida contra MITgcm")

- [ ] **Step 1: Run the four hybrid cases** (background, sequential; ≈4.7 h each)

```bash
for c in 2layer linear; do for b in bay nobay; do
  NUMBA_NUM_THREADS=5 python validation/bayiw_baroclinic/run_pyocean.py $c $b --coord hybrid --dt $DT || break 2
done; done
```
Expected: four logs end at 120 h, trunc 0.

- [ ] **Step 2: Compare**

Run:
```bash
python validation/bayiw_baroclinic/compare.py 2layer hybrid | tee /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2layer/compare_hybrid.txt
python validation/bayiw_baroclinic/compare.py linear hybrid | tee /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_linear/compare_hybrid.txt
```

- [ ] **Step 3: Write the comparison section**

In the README, add one table per case with z\*, hybrid and MITgcm side by side:
- seiche period;
- T110 period, rms ratio and correlation at the 4 inner stations;
- bay-effect rms ratio;
- spatial error at 24 h;
- RPE gain;
- interface thickness (two-layer).

Then the classification of the differences and the updated "Qué debe mejorar pyOCEAN".

---

### Task 9: Documentation and scorecard

**Files:**
- Modify:
  - `docs/design/vision_roadmap.md` §7 (new row after "Escalón de la boca de BayIW");
  - `CLAUDE.md` (ALE row: BayIW baroclinic numbers);
  - memory `/Users/estebancruz/.claude/projects/-Users-estebancruz-pyOCEAN/memory/project_status.md`.

- [ ] **Step 1: Add the scorecard row**

`| BayIW baroclínico (dos capas y lineal, con y sin bahía, 120 h) | MITgcm (tesis) | z*: <seiche period / T110 corr / bay effect>; híbrida: <…> | MITgcm 30 s, 50 niveles | <✅/🔶/❌ per criteria> |`, with the measured numbers from the README.

- [ ] **Step 2: Update `CLAUDE.md` and memory**

- `CLAUDE.md`: add the BayIW numbers to the ALE row.
- Memory: replace "Next: BayIW baroclinic benchmark…" with the result and the next steps from "Qué debe mejorar pyOCEAN".

- [ ] **Step 3: Full suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider` (only when no run is going).
Expected: everything passes, plus the existing xfails.

- [ ] **Step 4: Commit** (only if Esteban authorizes)

```bash
git add validation/bayiw_baroclinic/ tests/test_bayiw_baroclinic.py tests/test_sigma_ale.py CLAUDE.md docs/design/vision_roadmap.md docs/superpowers/plans/2026-10-12-bayiw-baroclinic-benchmark.md
git commit -m "BayIW baroclinic benchmark vs MITgcm: two-layer and linear, z* and hybrid"
```
