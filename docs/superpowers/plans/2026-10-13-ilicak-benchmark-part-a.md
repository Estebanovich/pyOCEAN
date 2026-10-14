# Four-model benchmark, part A (Ilicak 2012 cases + seamount) — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the Ilicak et al. (2012) lock exchange, overflow and internal wave (3 ν_h each) and the Beckmann–Haidvogel seamount in pyOCEAN (z\*, σ, hybrid), MOM6 (z\*, ρ; σ for the seamount), CROCO (σ) and MITgcm (z\*), from one common case definition, and compare RPE, case physics and cost with one set of metrics; HYCOM cost behind a 1-h gate.

**Architecture:** `validation/ilicak_benchmark/cases.py` defines each case once (grid, bed, T(x, y, z) function, EOS, ν, Δt, duration). One writer per model turns a case into that model's run directory; one reader per model turns its output into a common `series.npz` (t, h, T, u, v, η, surface-first layers). `metrics.py` computes RPE (`diagnostics/mixing.reference_potential_energy` with the exact hypsometry), the T range, and the physics of each case identically for every model. `campaign.py` runs the matrix and the cost ladder; `compare.py` writes the tables.

**Tech Stack:** Python 3.12 (NumPy 1.26.4, netCDF4, Numba), MOM6 + FMS (repro build), CROCO 2.1.3 (serial, jobcomp), MITgcm (genmake2, MNC + diagnostics), bash.

**Spec:** `docs/superpowers/specs/2026-10-13-ilicak-four-model-benchmark-design.md` (part A only; part B gets its own plan).

## Global Constraints

- Simulations yes, but considerably cheap (Esteban). Any run > 3× its estimate (spec table) is reported before going on.
- ≤ 5 threads / processes; never the efficiency cores; one heavy run at a time. pyOCEAN runs with `NUMBA_NUM_THREADS=5` and `set_threads(5)`.
- No reference-model source is modified (MOM6, FMS, CROCO, MITgcm, HYCOM). Case files (`MOM_input`, `cppdefs.h`, `param.h`, `SIZE.h`, `data*`) are generated in run/build directories outside the repo.
- Outputs under `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/`, never in the repo. Never commit `log_*.txt`, `*.npz` or `pyOCEAN-literature-review-guide.docx`.
- The benchmark only measures: pyOCEAN is not changed in this plan. A clear class-(a) error is reported to Esteban before going on.
- Common physics (spec A1): linear EOS ρ = ρ₀ − α(T − T_ref), ρ₀ = 1000 kg m⁻³, α = 0.2 kg m⁻³ °C⁻¹, T_ref = 0 °C; S constant; g = 9.81; no explicit tracer diffusion, no convective adjustment; ν_v = 1e-4 m² s⁻¹; no bottom drag; free slip; closed walls.
- Each model keeps its own schemes (spec A2): pyOCEAN defaults (vector-invariant momentum with KE Arakawa, WENO5-Z tracers, PPM_H4 remap, PPM continuity, `pgf="fv"`); MOM6 PPM_H4 remap, `TRACER_ADVECTION_SCHEME="PPM:H3"`, `KE_ARAKAWA`; CROCO the REGIONAL defaults of `cppdefs.h` (UV_HADV_UP3, UV_VADV_SPLINES, TS_HADV_RSUP3, TS_VADV_SPLINES — RSUP3 carries an explicit biharmonic diffusion |u|Δx³/12, `cppdefs_dev.h:423-436`, declared as class c); MITgcm `tempAdvScheme=7` (OS7MP), flux-form momentum (its default).
- Tests run no simulation of more than a few steps; no package installs (pip only with `--no-deps`, numpy stays 1.26.4).
- Never `git add` a directory. Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn`.
- Facts used below (subagent reports, this session):
  - **pyOCEAN:** `SigmaBaroclinic(h_grid, bathy, dt, n_layers, rho0, t_coef, t_ref, z0=None, closure="constant", nu_const, ah, pgf, vertical_coordinate="ale", regrid ∈ {"zstar","sigma","hybrid"}, regrid_params, ...)` (`core/dynamics/sigma_baroclinic.py:801-814`); `regrid_params`: z\* `dz_top` (surface first), σ `{}`, hybrid `dz_top` + `rho_target` (n+1 interfaces, surface first, increasing); state `SigmaState` fields `eta (ny,nx)`, `u (N,ny,nx+1)`, `v (N,ny+1,nx)`, `T`, `h` (N,ny,nx), **bed first**; `m.layer_heights(eta, h)` centre heights (z = 0 at rest); `initial_state(eta, T, h_src=None)`; grid `create_cartesian_grid(nx, ny, dx, dy, f0=...)` (`core/grid/horizontal.py:48`); `set_threads` in `sigma_baroclinic`.
  - **MOM6:** executable `~/pyOCEAN_refs/MOM6/.testing/build/repro/MOM6` (dynamic_symmetric, `u` on NI+1 faces); 2-D runs `NJGLOBAL = 1`, `REENTRANT_Y = False`, **1 rank** (NJ = 1 fails on 4 ranks); `GRID_CONFIG="cartesian"`, `AXIS_UNITS="k"`; free-slip bed needs `BOTTOMDRAGLAW=True, CDRAG=0, KV_BBL_MIN=0` (with `BOTTOMDRAGLAW=False` the bed is viscous no-slip, `MOM_vert_friction.F90:2263-2271`); RHO targets from `COORD_CONFIG="layer_ref"` with `LIGHTEST_DENSITY`, `DENSITY_RANGE` (interfaces ±½Δ around the layer densities, `MOM_coord_initialization.F90:164-200`, `MOM_regridding.F90:2462-2477`); `SAVE_INITIAL_CONDS=True` for the t = 0 state; FMS clock "Main loop" via `validation/mom6_bayiw/read_mom6.clock_table`; launcher pattern of `validation/mom6_bayiw/run_mom6.sh run` (polls "Total runtime", kills only its own ranks).
  - **CROCO:** build with a copy of `~/pyOCEAN_refs/croco_runs/grav_adj/jobcomp` (arm64 edits); run-dir `cppdefs.h`/`param.h` override the source; own case keys `PYO_LOCK`, `PYO_OVERFLOW`, `PYO_IW`, `PYO_SEAMOUNT` inserted before `#endif /* END OF CONFIGURATION CHOICE */` (`cppdefs.h:2064`) and before `#elif defined REGIONAL` (`param.h:188`), `#define REGIONAL` (`cppdefs.h:56`) turned to `#undef`; MMm0 = 1 works for x–z; grid file needs `spherical`, `xl`, `el`, `h`, `f`, `pm`, `pn`, `x_rho`, `y_rho` on (eta_rho = MMm+2, xi_rho = LLm+2) with ghosts filled; initial file `scrum_time` (units "second"), `zeta`, `ubar`, `vbar`, `u`, `v`, `temp`; uniform σ needs `NEW_S_COORD` and `S-coord: 0 0 1.e16`; constant Akv/Akt via `vertical_mixing: 1.e-4 0.`; linear EOS `lin_EOS_cff: 0. T0 0. TCOEF 0.` (ρ = 1000 + R0 − Tcoef(T − T0)); `RESET_RHO0` is forced (rho0 = volume mean, ~1e-3 of g/ρ₀, class c); g = 9.81 hard-coded; no wall clock in the log → `/usr/bin/time -p`; history `OUT_DOUBLE`; `z_w = ζ + (ζ + h)·s_w` for hc = 1e16.
  - **MITgcm:** tree `/Volumes/ESTEBAN_MIT/MITgcm`, optfile `tools/build_options/darwin_arm64_gfortran_esteban`; copy `verification/advect_xz/code/{CPP_OPTIONS.h,DIAGNOSTICS_SIZE.h}` (NONLIN_FRSURF defined); OS7MP needs `OLx = OLy = 4`; walls from one land column (and row in 3-D) because EXCH is periodic; defaults to override: `no_slip_sides/bottom` True, `hFacMin` 1, `implicitFreeSurface` False, `rhoNil` 999.8; z\* = `select_rStar=2, nonlinFreeSurf=4, exactConserv=.TRUE.`; binary inputs big-endian `>f8`, bathy negative; cell thickness = `hFactorC`·drF (diagnostic needs nonlinFreeSurf > 0); timing line `Seconds in section "MAIN_DO_LOOP [THE_MAIN_LOOP]"` → `Wall clock time`.

## Review Focus

1. **Layer order and land in every reader:** pyOCEAN and CROCO store layers bed first, MOM6 and MITgcm surface first; MITgcm has an extra land column (and row); CROCO has ghost columns. A wrong flip or offset leaves RPE nearly unchanged but corrupts the fronts. *Pinned in Tasks 4–6 (each reader test checks a known asymmetric field, bed vs surface).*
2. **RPE over a sloping bed:** `SigmaBaroclinic.rpe` assumes a flat basin; the overflow and seamount need the exact hypsometry with step changes in area. *Pinned in Task 2 (`test_rpe_hypsometry_exact_for_two_column_steps`).*
3. **Initial state at t = 0 in the series:** MOM6 prog files start after the first interval and MITgcm snapshot diagnostics may not write iteration 0; RPE₀ must be the model's own initial state, not the case definition. *Pinned in Tasks 4 and 6 (reader tests require `t[0] == 0`).*
4. **Uniform σ in CROCO:** θs = 0 without `NEW_S_COORD` divides by sinh(0); hc must be 1e16 so σ is uniform at 2000–4500 m. *Pinned in Task 5 (`test_croco_in_and_cppdefs_have_uniform_sigma_and_no_diffusion`).*
5. **Free-slip bed in MOM6:** `BOTTOMDRAGLAW=False` silently gives a viscous no-slip bed. *Pinned in Task 4 (`test_mom6_input_free_slip_linear_eos_and_coordinates`).*

---

## File structure

`validation/ilicak_benchmark/`:
- `__init__.py` — empty (package, as `validation/mom6_bayiw/`).
- `cases.py` — `Case` dataclass, the four case builders, `density`, `targets`, run paths.
- `series.py` — save/load the common series and its meta.
- `metrics.py` — hypsometry, RPE series, dRPE/dt, T range, lock fronts, overflow descent, internal-wave amplitude, max speed.
- `run_pyocean.py` — pyOCEAN runner (CLI) writing `series.npz`.
- `mom6.py`, `croco.py`, `mitgcm.py` — writer + reader of each model.
- `run_mom6.sh`, `run_croco.sh`, `run_mitgcm.sh` — `build <case>` / `run <case> <coord> <nu> [dt] [hours] [tag]`.
- `campaign.py` — results matrix and cost ladder.
- `compare.py` — tables (stdout + `README` sections).
- `README.md`.

`tests/test_ilicak_benchmark.py` — all tests of this plan.

---

### Task 1: Case definitions and the common series

**Files:**
- Create: `validation/ilicak_benchmark/__init__.py`, `validation/ilicak_benchmark/cases.py`, `validation/ilicak_benchmark/series.py`
- Test: `tests/test_ilicak_benchmark.py`

**Interfaces:**
- Produces:
  - `cases.RHO0 = 1000.0`, `cases.ALPHA = 0.2`, `cases.T_REF = 0.0`, `cases.G = 9.81`, `cases.NU_V = 1e-4`, `cases.RUNS: Path`
  - `cases.density(T: np.ndarray) -> np.ndarray`
  - `cases.Case` (frozen dataclass): `name: str`, `dx: np.ndarray (nx,)`, `dy: np.ndarray (ny,)`, `H: np.ndarray (ny, nx)` positive depth, `dz: np.ndarray (nz,)` nominal z-level thicknesses surface first, `T_fn: Callable[[x, y, z], T]` (x, y [m] from the domain's west/south edge, z [m] ≤ 0 up from the rest surface; broadcasting), `nu_h: tuple[float, ...]`, `dt: float`, `hours: float`, `out_s: float`, `f0: float`; properties `nx`, `ny`, `nz`, `x` (nx,) centres, `y` (ny,) centres, `zc` (nz,) nominal centres ≤ 0
  - `cases.get(name: str) -> Case` for `"lock"`, `"overflow"`, `"iw"`, `"seamount"`
  - `cases.targets(case: Case, n: int) -> np.ndarray` (n+1,) interface densities, surface first, increasing; the lightest and densest initial water at the centres of the first and last layers (MOM6 `layer_ref`)
  - `cases.run_dir(case: str, model: str, coord: str, nu: float, tag: str = "") -> Path`
  - `series.KEYS = ("t", "x", "y", "dx", "dy", "H", "h", "T", "u", "v", "eta")`
  - `series.save(path: Path, s: dict, meta: dict) -> Path`; `series.load(path: Path) -> tuple[dict, dict]`
  - Series layout: `t (nt,)` s; `x (nx,)`, `y (ny,)`; `dx (nx,)`, `dy (ny,)`; `H (ny, nx)` model rest depth (NaN land); `h (nt, nz, ny, nx)` cell thickness m, **k = 0 at the surface**, 0 on land; `T (nt, nz, ny, nx)` °C, NaN on land; `u (nt, nz, ny, nx+1)`; `v (nt, nz, ny+1, nx)`; `eta (nt, ny, nx)`. Meta: `model, coord, case, nu, dt, n_steps, wall_s` (main-loop wall time, s), `threads`.

- [ ] **Step 1: Write the failing tests**

```python
"""Four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design): definitions, writers, readers,
metrics. No simulation longer than a few steps."""
from __future__ import annotations

import numpy as np
import pytest

netCDF4 = pytest.importorskip("netCDF4")


def test_lock_exchange_definition():
    from validation.ilicak_benchmark import cases
    c = cases.get("lock")
    assert (c.nx, c.ny, c.nz) == (128, 1, 20) and np.allclose(c.dx, 500.0) and np.allclose(c.H, 20.0)
    assert np.allclose(c.dz, 1.0) and (c.dt, c.hours) == (10.0, 17.0) and c.nu_h == (0.1, 10.0, 100.0)
    T = c.T_fn(c.x[None, None, :], c.y[None, :, None], c.zc[:, None, None])
    assert T.shape == (20, 1, 128) and (T[:, :, :64] == 5.0).all() and (T[:, :, 64:] == 30.0).all()
    assert np.isclose(cases.density(np.array(5.0)) - cases.density(np.array(30.0)), 5.0)


def test_overflow_definition():
    from validation.ilicak_benchmark import cases
    c = cases.get("overflow")
    assert (c.nx, c.ny, c.nz) == (200, 1, 100) and np.allclose(c.dz, 20.0)
    d = lambda x: 500 + 0.5 * 1500 * (1 + np.tanh((x - 40e3) / 7e3))          # noqa: E731  Petersen 2015 eq. 24
    np.testing.assert_allclose(c.H[0], d(c.x))
    assert c.H[0, 0] < 501.0 and c.H[0, -1] > 1999.0
    T = c.T_fn(c.x[None, None, :], c.y[None, :, None], c.zc[:, None, None])
    assert (T[:, 0, c.x < 20e3] == 10.0).all() and (T[:, 0, c.x >= 20e3] == 20.0).all()
    assert np.isclose(cases.density(np.array(10.0)) - cases.density(np.array(20.0)), 2.0)
    assert (c.dt, c.hours) == (10.0, 40.0) and c.nu_h == (1.0, 100.0, 1000.0)


def test_internal_wave_definition():
    from validation.ilicak_benchmark import cases
    c = cases.get("iw")
    assert (c.nx, c.ny, c.nz) == (50, 1, 20) and np.allclose(c.dz, 25.0) and np.allclose(c.H, 500.0)
    zc = c.zc
    assert np.isclose(zc[0], -12.5) and np.isclose(zc[-1], -487.5)
    far = c.T_fn(np.array(2.5e3), np.array(0.0), zc)                             # outside the bump: background
    np.testing.assert_allclose(far, 10.1 + 10.0 * (-487.5 - zc) / -487.5)        # Petersen eq. 26
    mid = c.T_fn(np.array(125e3), np.array(0.0), zc)
    # bump: −A sin(π(z + 12.5)/(−475)) ≤ 0 → colder water lifted at x0; zero at the top and bottom centres
    assert np.isclose(mid[0], far[0]) and np.isclose(mid[-1], far[-1]) and (mid[1:-1] < far[1:-1]).all()
    assert np.isclose((far - mid).max(), 2.0, atol=0.05)
    assert (c.dt, c.hours) == (300.0, 200 * 24.0) and c.nu_h == (0.01, 15.0, 150.0)


def test_seamount_definition_matches_validation_seamount():
    from validation.ilicak_benchmark import cases
    from validation.seamount import run_seamount as rs
    c = cases.get("seamount")
    assert (c.nx, c.ny, c.nz) == (64, 48, 20) and c.f0 == 1e-4 and c.nu_h == (0.0,)
    assert np.isclose(c.H.min(), rs.H0 - rs.HS * np.exp(-2 * 3e3 ** 2 / rs.LS ** 2))   # summit cells at (±3, ±3) km
    assert np.isclose(c.H.max(), rs.H0, rtol=1e-3)
    z = np.array([0.0, -1000.0, -4000.0])
    rho = cases.density(c.T_fn(np.array(0.0), np.array(0.0), z))
    np.testing.assert_allclose(rho, rs.RHO0 + rs.DRHO * (1 - np.exp(z / rs.DECAY)))


def test_targets_put_the_extreme_waters_at_layer_centres():
    from validation.ilicak_benchmark import cases
    c = cases.get("lock")
    t = cases.targets(c, 20)
    assert t.size == 21 and (np.diff(t) > 0).all()
    lo, hi = cases.density(np.array(30.0)), cases.density(np.array(5.0))
    assert np.isclose(0.5 * (t[0] + t[1]), lo) and np.isclose(0.5 * (t[-2] + t[-1]), hi)


def test_series_round_trip(tmp_path):
    from validation.ilicak_benchmark import series
    nt, nz, ny, nx = 2, 3, 1, 4
    s = dict(t=np.array([0.0, 900.0]), x=np.arange(nx) + 0.5, y=np.array([0.5]), dx=np.ones(nx), dy=np.ones(ny),
             H=np.full((ny, nx), 3.0), h=np.ones((nt, nz, ny, nx)), T=np.zeros((nt, nz, ny, nx)),
             u=np.zeros((nt, nz, ny, nx + 1)), v=np.zeros((nt, nz, ny + 1, nx)), eta=np.zeros((nt, ny, nx)))
    p = series.save(tmp_path / "series.npz", s, dict(model="x", wall_s=1.5))
    s2, meta = series.load(p)
    assert meta["wall_s"] == 1.5 and set(s2) == set(series.KEYS)
    with pytest.raises(ValueError, match="shape"):
        series.save(tmp_path / "bad.npz", {**s, "u": np.zeros((nt, nz, ny, nx))}, {})
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q`
Expected: FAIL — `ModuleNotFoundError: No module named 'validation.ilicak_benchmark'`.

- [ ] **Step 3: Implement `cases.py` and `series.py`**

`validation/ilicak_benchmark/__init__.py`: empty file.

`validation/ilicak_benchmark/cases.py`:

```python
"""
Case definitions of the four-model benchmark, part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A1).
One definition per case; every model's writer and pyOCEAN's runner read it.

- lock exchange: Ilicak et al. (2012, §3.1, eq. 3–4) with T = 5/30 °C as Petersen et al. (2015, eq. 21–22) and the
  MOM6 lock_exchange example (Ilicak's 35 °C is inconsistent with Δρ = 5 at α = 0.2);
- overflow: Ilicak (2012, §4) with the shelf of Petersen (2015, eq. 24–25, Table 5);
- internal wave: Petersen (2015, eq. 26–27), high amplitude A = 2 °C (Ilicak Fig. 11b); N = 6.34e-3 s⁻¹ here,
  Ilicak quotes 6.26e-3;
- seamount: Beckmann & Haidvogel (1993, §2c), the definition of validation/seamount.

Linear EOS ρ = ρ₀ − α (T − T_ref) [kg m⁻³], ρ₀ = 1000, α = 0.2 kg m⁻³ °C⁻¹, T_ref = 0 °C.
Coordinates: x, y [m] from the west/south edge of the domain; z [m] ≤ 0, up from the rest surface.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np

from validation.seamount import run_seamount as _sm

RHO0, ALPHA, T_REF, G, NU_V = 1000.0, 0.2, 0.0, 9.81, 1e-4
RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark")


def density(T: np.ndarray) -> np.ndarray:
    """Linear EOS ρ = ρ₀ − α (T − T_ref) [kg m⁻³] (Vallis 2006, eq. 1.156 without S)."""
    return RHO0 - ALPHA * (np.asarray(T, float) - T_REF)


@dataclass(frozen=True)
class Case:
    name: str
    dx: np.ndarray                 # (nx,) [m]
    dy: np.ndarray                 # (ny,) [m]
    H: np.ndarray                  # (ny, nx) rest depth [m], positive
    dz: np.ndarray                 # (nz,) nominal z-level thicknesses [m], surface first
    T_fn: Callable                 # T(x, y, z) [°C], broadcasting
    nu_h: tuple                    # horizontal viscosities of the sweep [m² s⁻¹]
    dt: float                      # results time step [s]
    hours: float                   # run length [h]
    out_s: float                   # output interval [s]
    f0: float = 0.0                # Coriolis [s⁻¹]

    @property
    def nx(self) -> int:
        return self.dx.size

    @property
    def ny(self) -> int:
        return self.dy.size

    @property
    def nz(self) -> int:
        return self.dz.size

    @property
    def x(self) -> np.ndarray:
        return np.cumsum(self.dx) - 0.5 * self.dx

    @property
    def y(self) -> np.ndarray:
        return np.cumsum(self.dy) - 0.5 * self.dy

    @property
    def zc(self) -> np.ndarray:
        zf = -np.concatenate([[0.0], np.cumsum(self.dz)])
        return 0.5 * (zf[1:] + zf[:-1])


def _lock() -> Case:
    nx = 128
    T = lambda x, y, z: np.broadcast_to(np.where(np.asarray(x) < 32e3, 5.0, 30.0),            # noqa: E731
                                        np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))).copy()
    return Case("lock", np.full(nx, 500.0), np.array([500.0]), np.full((1, nx), 20.0), np.full(20, 1.0), T,
                (0.1, 10.0, 100.0), 10.0, 17.0, 900.0)


def _overflow() -> Case:
    nx, d1, d2, x0, r = 200, 500.0, 2000.0, 40e3, 7e3
    x = np.arange(nx) * 1e3 + 500.0
    H = d1 + 0.5 * (d2 - d1) * (1.0 + np.tanh((x - x0) / r))                                 # Petersen eq. 24
    T = lambda x, y, z: np.broadcast_to(np.where(np.asarray(x) < 20e3, 10.0, 20.0),           # noqa: E731
                                        np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))).copy()
    return Case("overflow", np.full(nx, 1e3), np.array([1e3]), H[None, :], np.full(100, 20.0), T,
                (1.0, 100.0, 1000.0), 10.0, 40.0, 3600.0)


def _iw() -> Case:
    t_bot, t_top, z_bot, A, L, x0, dz = 10.1, 20.1, -487.5, 2.0, 50e3, 125e3, 25.0

    def T(x, y, z):
        x, z = np.asarray(x, float), np.asarray(z, float)
        base = t_bot + (t_top - t_bot) * (z_bot - z) / z_bot                                     # Petersen eq. 26
        bump = np.where(np.abs(x - x0) < L, -A * np.cos(np.pi * (x - x0) / (2 * L))
                        * np.sin(np.pi * (z + dz / 2) / (z_bot + dz / 2)), 0.0)                  # eq. 27
        return np.broadcast_to(base + bump, np.broadcast_shapes(x.shape, np.shape(y), z.shape)).copy()

    return Case("iw", np.full(50, 5e3), np.array([5e3]), np.full((1, 50), 500.0), np.full(20, dz), T,
                (0.01, 15.0, 150.0), 300.0, 200 * 24.0, 6 * 3600.0)


def _seamount() -> Case:
    dx, nx, ny = 6e3, 64, 48
    xc = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    yc = (np.arange(ny) + 0.5) * dx - ny * dx / 2
    H = _sm.H0 - _sm.HS * np.exp(-(xc[None, :] ** 2 + yc[:, None] ** 2) / _sm.LS ** 2)
    T = lambda x, y, z: np.broadcast_to(                                                      # noqa: E731
        (RHO0 - (_sm.RHO0 + _sm.DRHO * (1 - np.exp(np.asarray(z, float) / _sm.DECAY)))) / ALPHA + T_REF,
        np.broadcast_shapes(np.shape(x), np.shape(y), np.shape(z))).copy()
    return Case("seamount", np.full(nx, dx), np.full(ny, dx), H, np.full(20, _sm.H0 / 20), T,
                (0.0,), 600.0, 240.0, 86400.0, f0=_sm.F0)


_BUILDERS = {"lock": _lock, "overflow": _overflow, "iw": _iw, "seamount": _seamount}


def get(name: str) -> Case:
    """The case ``name`` ∈ {'lock', 'overflow', 'iw', 'seamount'}."""
    if name not in _BUILDERS:
        raise ValueError(f"unknown case {name!r}; expected one of {sorted(_BUILDERS)}")
    return _BUILDERS[name]()


def targets(case: Case, n: int) -> np.ndarray:
    """n+1 interface target densities [kg m⁻³], surface first, increasing, evenly spaced, with the lightest and the
    densest initial water at the centres of the first and last layers (MOM6 COORD_CONFIG 'layer_ref':
    LIGHTEST_DENSITY = ρ_min, DENSITY_RANGE = ρ_max − ρ_min, MOM_coord_initialization.F90:164-200)."""
    T = case.T_fn(case.x[None, None, :], case.y[None, :, None], case.zc[:, None, None])
    rho = density(T)
    lo, hi = float(rho.min()), float(rho.max())
    d = (hi - lo) / (n - 1)
    return np.linspace(lo - 0.5 * d, hi + 0.5 * d, n + 1)


def run_dir(case: str, model: str, coord: str, nu: float, tag: str = "") -> Path:
    """RUNS/<case>/<model>_<coord>_nu<nu>[_<tag>]/ — one run, holding series.npz."""
    return RUNS / case / (f"{model}_{coord}_nu{nu:g}" + (f"_{tag}" if tag else ""))
```

`validation/ilicak_benchmark/series.py`:

```python
"""The common output series of the four-model benchmark (spec §Piezas): every reader writes it, metrics read it.
Layers are surface first (k = 0 at the surface); u on the nx+1 x-faces, v on the ny+1 y-faces."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np

KEYS = ("t", "x", "y", "dx", "dy", "H", "h", "T", "u", "v", "eta")


def _check(s: dict) -> None:
    nt, nz, ny, nx = np.shape(s["h"])
    want = dict(t=(nt,), x=(nx,), y=(ny,), dx=(nx,), dy=(ny,), H=(ny, nx), T=(nt, nz, ny, nx),
                u=(nt, nz, ny, nx + 1), v=(nt, nz, ny + 1, nx), eta=(nt, ny, nx))
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
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q`
Expected: 6 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/ilicak_benchmark/__init__.py validation/ilicak_benchmark/cases.py validation/ilicak_benchmark/series.py tests/test_ilicak_benchmark.py
git commit -m "Ilicak benchmark: case definitions and common series

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 2: Metrics

**Files:**
- Create: `validation/ilicak_benchmark/metrics.py`
- Test: `tests/test_ilicak_benchmark.py` (append)

**Interfaces:**
- Consumes: `cases.density`, `cases.Case`, the series layout of Task 1.
- Produces:
  - `metrics.hypsometry(H: np.ndarray, dx: np.ndarray, dy: np.ndarray, top: float = 10.0) -> tuple[np.ndarray, np.ndarray]`
  - `metrics.rpe(s: dict) -> np.ndarray` (nt,) [J]
  - `metrics.drpe_dt(s: dict, rpe: np.ndarray, t0: float, t1: float) -> float` [W m⁻²]
  - `metrics.t_range(s: dict, h_min: float = 1e-3) -> tuple[float, float]`
  - `metrics.lock_fronts(s: dict, t_fit: tuple = (2 * 3600.0, 8 * 3600.0)) -> dict` keys `dense_nose, light_nose, dense_body, light_body` [m s⁻¹]
  - `metrics.overflow_descent(s: dict, T_mid: float = 15.0) -> tuple[np.ndarray, np.ndarray]` (x_c, depth_c) per record [m]
  - `metrics.iw_amplitude(s: dict, x0: float = 125e3, T_iso: float | None = None) -> np.ndarray` isotherm displacement [m] per record
  - `metrics.max_speed(s: dict) -> np.ndarray` (nt,) [m s⁻¹]
  - `metrics.BENJAMIN = 0.5 * sqrt(G * 5/1000 * 20)` [m s⁻¹]

- [ ] **Step 1: Write the failing tests**

```python
def _flat_series(h_layers, T_layers, dx=1.0, nx=1):
    """One-record series with layers (surface first) h_layers [m] and T_layers [°C] in every column."""
    h = np.asarray(h_layers, float)[None, :, None, None] * np.ones((1, 1, 1, nx))
    T = np.asarray(T_layers, float)[None, :, None, None] * np.ones((1, 1, 1, nx))
    nz = h.shape[1]
    return dict(t=np.array([0.0]), x=np.arange(nx) * dx + dx / 2, y=np.array([0.5]), dx=np.full(nx, dx),
                dy=np.array([1.0]), H=h[0].sum(0), h=h, T=T, u=np.zeros((1, nz, 1, nx + 1)),
                v=np.zeros((1, nz, 2, nx)), eta=np.zeros((1, 1, nx)))


def test_rpe_does_not_depend_on_how_the_column_is_split():
    from validation.ilicak_benchmark import metrics
    a = metrics.rpe(_flat_series(np.full(20, 1.0), [30.0] * 10 + [5.0] * 10))          # 20 z-levels
    b = metrics.rpe(_flat_series([10.0, 10.0], [30.0, 5.0]))                            # two layers
    c = metrics.rpe(_flat_series([10.0, 0.0, 4.0, 6.0], [30.0, 17.0, 5.0, 5.0]))       # σ-like, one vanished layer
    np.testing.assert_allclose([b[0], c[0]], a[0], rtol=1e-12)
    # analytic: g Σ ρ ∫ z dz, light 10 m on dense 10 m, unit area
    exact = 9.81 * (999.0 * 0.5 * 10.0 ** 2 + 994.0 * 0.5 * (20.0 ** 2 - 10.0 ** 2))
    np.testing.assert_allclose(a[0], exact, rtol=1e-12)


def test_rpe_hypsometry_exact_for_two_column_steps():
    from validation.ilicak_benchmark import metrics
    # columns of 10 m and 30 m, unit width: uniform density fills the deep column's bottom 20 m first
    s = _flat_series([10.0], [10.0], nx=2)
    s["h"] = np.array([10.0, 30.0])[None, None, None, :]
    s["T"] = np.full((1, 1, 1, 2), 10.0)
    s["H"] = np.array([[10.0, 30.0]])
    z, A = metrics.hypsometry(s["H"], s["dx"], s["dy"])
    assert z[0] == 0.0 and A[0] == 1.0 and A[-1] == 2.0
    rho = 998.0
    exact = 9.81 * rho * (0.5 * 20.0 ** 2 + 2 * 0.5 * (30.0 ** 2 - 20.0 ** 2))
    np.testing.assert_allclose(metrics.rpe(s)[0], exact, rtol=1e-9)


def test_drpe_dt_is_per_unit_surface_area():
    from validation.ilicak_benchmark import metrics
    s = _flat_series([1.0], [5.0], dx=500.0, nx=4)
    s["t"] = np.array([0.0, 3600.0, 7200.0])
    rpe = np.array([0.0, 2000.0, 4000.0])
    assert np.isclose(metrics.drpe_dt(s, rpe, 0.0, 7200.0), (4000.0 / 7200.0) / (4 * 500.0 * 1.0))


def test_lock_fronts_track_a_steady_gravity_current():
    from validation.ilicak_benchmark import metrics
    nx, dx, U = 128, 500.0, 0.5
    t = np.arange(0, 9) * 3600.0
    nz = 20
    h = np.ones((t.size, nz, 1, nx))
    T = np.full((t.size, nz, 1, nx), 30.0)
    x = np.arange(nx) * dx + dx / 2
    for n, tt in enumerate(t):
        T[n, 10:, 0, x < 32e3 + U * tt] = 5.0           # dense bottom half advancing east
        T[n, 10:, 0, x < 32e3] = 5.0
        T[n, :10, 0, (x < 32e3) & (x >= 32e3 - U * tt)] = 30.0
        T[n, :10, 0, x < 32e3 - U * tt] = 5.0           # light top half advancing west
    s = dict(t=t, x=x, y=np.array([250.0]), dx=np.full(nx, dx), dy=np.array([500.0]), H=np.full((1, nx), 20.0),
             h=h, T=T, u=np.zeros((t.size, nz, 1, nx + 1)), v=np.zeros((t.size, nz, 2, nx)),
             eta=np.zeros((t.size, 1, nx)))
    f = metrics.lock_fronts(s)
    for k in ("dense_nose", "light_nose", "dense_body", "light_body"):
        assert abs(f[k] - U) < 0.02, (k, f[k])


def test_overflow_descent_and_iw_amplitude_and_t_range():
    from validation.ilicak_benchmark import metrics
    s = _flat_series([10.0, 10.0], [20.0, 10.0], nx=3)
    s["T"][0, :, 0, 0] = [20.0, 20.0]
    xc, dep = metrics.overflow_descent(s)
    assert np.allclose(xc, s["x"][1:].mean()) and np.allclose(dep, 15.0)
    # isotherm 11.25 °C between centres at 5 m and 15 m: record 0 (20, 10 °C) → 13.75 m deep;
    # record 1 (12.5, 10 °C: colder top at x0) → 10 m deep, i.e. lifted 3.75 m
    s2 = _flat_series([10.0, 10.0], [20.0, 10.0], dx=125e3, nx=2)
    s2 = {k: (np.concatenate([v, v]) if k in ("h", "T", "u", "v", "eta") else v) for k, v in s2.items()}
    s2["t"], s2["x"] = np.array([0.0, 3600.0]), np.array([0.0, 125e3])
    s2["T"][1, 0, 0, 1] = 12.5
    np.testing.assert_allclose(metrics.iw_amplitude(s2, T_iso=11.25), [0.0, 3.75], atol=1e-12)
    lo, hi = metrics.t_range(s)
    assert (lo, hi) == (10.0, 20.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k "rpe or fronts or descent or drpe"`
Expected: FAIL — `ImportError: cannot import name 'metrics'`.

- [ ] **Step 3: Implement `metrics.py`**

```python
"""
Metrics of the four-model benchmark (spec §A4), identical for every model, on the common series (series.py).

RPE (Winters et al. 1995; Ilicak et al. 2012, eq. 1–2): RPE = g ∫ ρ* z dV with ρ* the adiabatically sorted density
and z the height above the deepest point of each model's own bed; computed by
diagnostics.mixing.reference_potential_energy on the true cell volumes h·Δx·Δy and the exact hypsometry of the bed
(step changes of area at each column's bed height), so z-levels with partial cells, σ and ALE layers are treated alike.
dRPE/dt is per unit surface area [W m⁻²] as in Ilicak (Fig. 6) and Petersen (2015).
"""
from __future__ import annotations

import numpy as np

from diagnostics.mixing import reference_potential_energy
from validation.ilicak_benchmark import cases

BENJAMIN = 0.5 * np.sqrt(cases.G * 5.0 / cases.RHO0 * 20.0)     # Benjamin (1968) u_f = ½ √(g′H) [m s⁻¹]


def hypsometry(H: np.ndarray, dx: np.ndarray, dy: np.ndarray, top: float = 10.0) -> tuple[np.ndarray, np.ndarray]:
    """(z, A(z)): heights above the deepest bed [m, increasing] and the horizontal area [m²] at those heights, with
    each column's area added at its bed height (a 1e-9 m ramp makes the step exact for the trapezoid integral);
    the top is ``top`` m above the rest surface."""
    Hc = np.nan_to_num(np.asarray(H, float))
    area = np.asarray(dy, float)[:, None] * np.asarray(dx, float)[None, :]
    wet = Hc > 0
    hmax = Hc[wet].max()
    bed = hmax - Hc[wet]
    a = area[wet]
    levels = np.unique(bed)
    z, A = [0.0], [a[bed <= levels[0]].sum()]
    for b in levels[1:]:
        z += [b - 1e-9, b]
        A += [A[-1], a[bed <= b].sum()]
    z.append(hmax + top)
    A.append(A[-1])
    return np.asarray(z), np.asarray(A)


def rpe(s: dict) -> np.ndarray:
    """RPE [J] of every record."""
    zA = hypsometry(s["H"], s["dx"], s["dy"])
    vol = s["h"] * (s["dy"][:, None] * s["dx"][None, :])[None, None]
    rho = cases.density(s["T"])
    return np.array([reference_potential_energy(np.where(vol[n] > 0, rho[n], np.nan), vol[n], hypsometry=zA)
                     for n in range(s["t"].size)])


def drpe_dt(s: dict, rpe_t: np.ndarray, t0: float, t1: float) -> float:
    """Mean dRPE/dt between the records nearest t0 and t1, per unit surface area [W m⁻²]."""
    i0, i1 = (int(np.argmin(np.abs(s["t"] - t))) for t in (t0, t1))
    area = float(np.sum(np.isfinite(np.asarray(s["H"], float)) * (s["dy"][:, None] * s["dx"][None, :])))
    return float((rpe_t[i1] - rpe_t[i0]) / (s["t"][i1] - s["t"][i0]) / area)


def t_range(s: dict, h_min: float = 1e-3) -> tuple[float, float]:
    """(min, max) of T over all records in cells thicker than h_min [m] (new extrema = non-monotone advection)."""
    T = np.where(s["h"] > h_min, s["T"], np.nan)
    return float(np.nanmin(T)), float(np.nanmax(T))


def _front(x: np.ndarray, thick: np.ndarray, start: float, sign: int, thr: float) -> float:
    """Distance from ``start`` of the farthest column (east if sign > 0) whose ``thick`` exceeds ``thr``."""
    sel = (sign * (x - start) > 0) & (thick > thr)
    return float(np.max(sign * (x[sel] - start))) if sel.any() else 0.0


def lock_fronts(s: dict, t_fit: tuple = (2 * 3600.0, 8 * 3600.0)) -> dict:
    """Front speeds [m s⁻¹] of the lock exchange (spec §A4; as validation/mom6_lock_exchange): dense water (T < 17.5)
    thicker than 1 m (nose) / 5 m (body) east of the gate, light water (T > 17.5) likewise west of it; speed =
    least-squares slope of the position over t_fit."""
    x, gate = s["x"], 32e3
    sel = (s["t"] >= t_fit[0]) & (s["t"] <= t_fit[1])
    out = {}
    for name, cold, sign, thr in (("dense_nose", True, 1, 1.0), ("light_nose", False, -1, 1.0),
                                  ("dense_body", True, 1, 5.0), ("light_body", False, -1, 5.0)):
        pos = []
        for n in np.flatnonzero(sel):
            T = s["T"][n, :, 0, :]
            mask = (T < 17.5) if cold else (T > 17.5)
            thick = np.where(mask, s["h"][n, :, 0, :], 0.0).sum(0)
            pos.append(_front(x, thick, gate, sign, thr))
        out[name] = float(np.polyfit(s["t"][sel], pos, 1)[0])
    return out


def _centres(s: dict, n: int) -> np.ndarray:
    """Cell-centre depths [m, positive down] of record n (surface-first layers below η)."""
    h = s["h"][n]
    top = np.concatenate([np.zeros((1,) + h.shape[1:]), np.cumsum(h, 0)[:-1]]) - s["eta"][n][None]
    return top + 0.5 * h


def overflow_descent(s: dict, T_mid: float = 15.0) -> tuple[np.ndarray, np.ndarray]:
    """Centre of mass (x_c [m], depth_c [m]) of the dense water (T < T_mid) in every record."""
    xs, ds = [], []
    for n in range(s["t"].size):
        V = s["h"][n] * (s["dy"][:, None] * s["dx"][None, :])[None]
        w = np.where(np.nan_to_num(s["T"][n], nan=np.inf) < T_mid, V, 0.0)
        xs.append(float((w * s["x"][None, None, :]).sum() / w.sum()))
        ds.append(float((w * _centres(s, n)).sum() / w.sum()))
    return np.array(xs), np.array(ds)


def iw_amplitude(s: dict, x0: float = 125e3, T_iso: float | None = None) -> np.ndarray:
    """Displacement [m, positive up] of the isotherm T_iso (default: the mid-depth background value 15.1 °C) in the
    column nearest x0, relative to record 0; linear interpolation between cell centres."""
    T_iso = 15.1 if T_iso is None else T_iso
    i = int(np.argmin(np.abs(s["x"] - x0)))
    depth = []
    for n in range(s["t"].size):
        zc = _centres(s, n)[:, 0, i]
        T = s["T"][n, :, 0, i]
        ok = s["h"][n, :, 0, i] > 1e-3
        zc, T = zc[ok], T[ok]
        depth.append(float(np.interp(T_iso, T[::-1], zc[::-1])))     # T decreases with depth
    depth = np.array(depth)
    return depth[0] - depth


def max_speed(s: dict) -> np.ndarray:
    """max |u|, |v| over faces of every record [m s⁻¹]."""
    return np.maximum(np.abs(s["u"]).reshape(s["t"].size, -1).max(1), np.abs(s["v"]).reshape(s["t"].size, -1).max(1))
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q`
Expected: 11 passed.

- [ ] **Step 5: Commit**

```bash
git add validation/ilicak_benchmark/metrics.py tests/test_ilicak_benchmark.py
git commit -m "Ilicak benchmark: metrics (RPE with exact hypsometry, fronts, descent, IW amplitude)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 3: pyOCEAN runner

**Files:**
- Create: `validation/ilicak_benchmark/run_pyocean.py`
- Test: `tests/test_ilicak_benchmark.py` (append)

**Interfaces:**
- Consumes: `cases.get`, `cases.targets`, `cases.run_dir`, `series.save`, `core.dynamics.sigma_baroclinic.SigmaBaroclinic`, `set_threads`, `core.grid.horizontal.create_cartesian_grid`.
- Produces:
  - `run_pyocean.build(case: Case, coord: str, nu: float, dt: float | None = None) -> tuple[SigmaBaroclinic, SigmaState]` (`coord` ∈ `zstar`, `sigma`, `hybrid`)
  - `run_pyocean.record(m, st) -> dict` (one record of the series fields h, T, u, v, eta, surface first)
  - `run_pyocean.run(case_name: str, coord: str, nu: float, dt: float | None = None, hours: float | None = None, tag: str = "", threads: int = 5) -> Path` (series.npz path)
  - CLI: `python validation/ilicak_benchmark/run_pyocean.py <case> <coord> <nu> [--dt DT] [--hours H] [--tag TAG] [--threads N]`

- [ ] **Step 1: Write the failing tests**

```python
@pytest.mark.parametrize("name,coord", [("lock", "zstar"), ("lock", "hybrid"), ("overflow", "sigma"),
                                        ("iw", "hybrid"), ("seamount", "zstar")])
def test_pyocean_builds_each_case_and_records_surface_first(name, coord):
    from validation.ilicak_benchmark import cases, run_pyocean
    c = cases.get(name)
    m, st = run_pyocean.build(c, coord, c.nu_h[-1])
    r = run_pyocean.record(m, st)
    nz = st.h.shape[0]
    assert r["h"].shape == (nz, c.ny, c.nx) and r["u"].shape == (nz, c.ny, c.nx + 1)
    np.testing.assert_allclose(r["h"].sum(0), c.H, rtol=1e-9)                     # rest, η = 0
    # surface first: the top layer of the lock's east half is the warm water, the top of the IW column the warmest
    if name == "lock":
        assert np.allclose(r["T"][0, 0, -1], 30.0) and np.allclose(r["T"][-1, 0, 0], 5.0)
    if name == "iw":
        assert r["T"][0, 0, 0] > r["T"][-1, 0, 0]
    T0 = c.T_fn(c.x[None, None, :], c.y[None, :, None], c.zc[:, None, None])
    assert np.nanmin(r["T"]) >= T0.min() - 1e-6 and np.nanmax(r["T"]) <= T0.max() + 1e-6


def test_pyocean_short_run_writes_a_series(tmp_path, monkeypatch):
    from validation.ilicak_benchmark import cases, run_pyocean, series
    monkeypatch.setattr(cases, "RUNS", tmp_path)
    p = run_pyocean.run("lock", "zstar", 10.0, hours=60.0 / 3600.0, threads=1)     # 6 steps
    s, meta = series.load(p)
    assert s["t"][0] == 0.0 and s["t"][-1] == 60.0 and meta["model"] == "pyOCEAN" and meta["n_steps"] == 6
    heat = (s["h"] * s["T"]).sum(axis=(1, 2, 3))
    np.testing.assert_allclose(heat, heat[0], rtol=1e-10)
```

- [ ] **Step 2: Run them to see them fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_ilicak_benchmark.py -q -k pyocean`
Expected: FAIL — `ImportError: cannot import name 'run_pyocean'`.

- [ ] **Step 3: Implement `run_pyocean.py`**

```python
#!/usr/bin/env python
"""
pyOCEAN runs of the four-model benchmark (spec §A2): ALE z*, σ or hybrid with the model's defaults (vector-invariant
momentum, KE Arakawa, WENO5-Z tracers, PPM_H4 remap, PPM continuity, finite-volume PGF), closed free-slip walls, no
bottom drag, ν_v = 1e-4, no tracer diffusion, linear EOS of cases.py.

- z*: dz_top = the case's z-levels; σ: uniform σ-layers; hybrid: dz_top = 0.01 m floors (isopycnal interior, as the
  hybrid lock exchange of tests/test_sigma_ale.py) and rho_target = cases.targets (the same as MOM6 'RHO').
- initial state: T of cases.T_fn at the layer centres (z*, σ); for the hybrid, T on the case z-levels as h_src
  (MOM6 Z_INIT_ALE_REMAPPING; validation/internal_wave/case.py:189-193).
- seamount: the reference profile rho_ref = 'auto' (validation/seamount: best pyOCEAN setting, recorded in README).
- wall_s: the time of the steps after the first 5 (Numba compilation), scaled to all steps.

Usage: NUMBA_NUM_THREADS=5 python validation/ilicak_benchmark/run_pyocean.py lock zstar 10 [--dt 10] [--hours 17]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from validation.ilicak_benchmark import cases, series  # noqa: E402

N_WARM = 5


def build(case: cases.Case, coord: str, nu: float, dt: float | None = None):
    """The ALE model of ``case`` in ``coord`` ∈ {'zstar', 'sigma', 'hybrid'} with ν_h = nu, and its initial state."""
    hg = create_cartesian_grid(nx=case.nx, ny=case.ny, dx=float(case.dx[0]), dy=float(case.dy[0]), f0=case.f0)
    nz = case.nz
    if coord == "zstar":
        params = dict(dz_top=case.dz.copy())
    elif coord == "sigma":
        params = {}
    elif coord == "hybrid":
        params = dict(dz_top=np.full(nz, 0.01), rho_target=cases.targets(case, nz))
    else:
        raise ValueError(f"unknown coord {coord!r}")
    m = SigmaBaroclinic(hg, case.H.copy(), dt=case.dt if dt is None else dt, n_layers=nz, rho0=cases.RHO0,
                        t_coef=cases.ALPHA, t_ref=cases.T_REF, z0=None, closure="constant", nu_const=cases.NU_V,
                        ah=nu, g=cases.G, pgf="fv", rho_ref="auto" if case.name == "seamount" else None,
                        vertical_coordinate="ale", regrid=coord, regrid_params=params)
    eta = np.zeros((case.ny, case.nx))
    X, Y = case.x[None, None, :], case.y[None, :, None]
    if coord == "hybrid":
        zf = np.concatenate([[0.0], np.cumsum(case.dz)])[:, None, None]
        h_src = (np.minimum(zf[1:], case.H[None]) - np.minimum(zf[:-1], case.H[None]))[::-1]     # bed first
        T_src = case.T_fn(X, Y, case.zc[:, None, None])[::-1]
        return m, m.initial_state(eta, T_src, h_src=h_src)
    zc = m.layer_heights(eta)                                                   # (N, ny, nx) bed first, ≤ 0
    return m, m.initial_state(eta, case.T_fn(X, Y, zc))


def record(m, st) -> dict:
    """The series fields of state ``st``, layers flipped to surface first."""
    return dict(h=st.h[::-1].copy(), T=np.where(m.fs.active[None], st.T[::-1], np.nan),     # land: NaN
                u=st.u[::-1].copy(), v=st.v[::-1].copy(), eta=st.eta.copy())


def run(case_name: str, coord: str, nu: float, dt: float | None = None, hours: float | None = None, tag: str = "",
        threads: int = 5) -> Path:
    """Run and write cases.run_dir(case, 'pyocean', coord, nu, tag)/series.npz; returns its path."""
    set_threads(threads)
    c = cases.get(case_name)
    dt = c.dt if dt is None else dt
    hours = c.hours if hours is None else hours
    m, st = build(c, coord, nu, dt)
    n_steps = int(round(hours * 3600.0 / dt))
    every = max(1, int(round(min(c.out_s, hours * 3600.0) / dt)))
    recs, times = [record(m, st)], [0.0]
    tic = None
    for n in range(1, n_steps + 1):
        if n == N_WARM + 1:
            tic = time.perf_counter()
        st = m.step(st)
        if n % every == 0 or n == n_steps:
            recs.append(record(m, st))
            times.append(n * dt)
    wall = (time.perf_counter() - tic) * n_steps / max(1, n_steps - N_WARM) if tic else float("nan")
    s = {k: np.array([r[k] for r in recs]) for k in ("h", "T", "u", "v", "eta")}
    s.update(t=np.array(times), x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H)
    meta = dict(model="pyOCEAN", coord=coord, case=case_name, nu=nu, dt=dt, n_steps=n_steps, wall_s=wall,
                threads=threads, n_trunc=int(st.n_trunc))
    return series.save(cases.run_dir(case_name, "pyocean", coord, nu, tag) / "series.npz", s, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case")
    ap.add_argument("coord", choices=["zstar", "sigma", "hybrid"])
    ap.add_argument("nu", type=float)
    ap.add_argument("--dt", type=float)
    ap.add_argument("--hours", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    print(run(a.case, a.coord, a.nu, a.dt, a.hours, a.tag, a.threads))


if __name__ == "__main__":
    main()
```

The wet mask is `m.fs.active` (`sigma_baroclinic.py:1074-1075`); none of the four cases has land columns.

- [ ] **Step 4: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_ilicak_benchmark.py -q -k pyocean`
Expected: 6 passed (the hybrid builds may warn that `sum(dz_top) < max(H)`: expected, the floors are 0.01 m by design).

- [ ] **Step 5: Commit**

```bash
git add validation/ilicak_benchmark/run_pyocean.py tests/test_ilicak_benchmark.py
git commit -m "Ilicak benchmark: pyOCEAN runner (z*, sigma, hybrid)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 4: MOM6 writer, reader and runner

**Files:**
- Create: `validation/ilicak_benchmark/mom6.py`, `validation/ilicak_benchmark/run_mom6.sh`
- Test: `tests/test_ilicak_benchmark.py` (append)

**Interfaces:**
- Consumes: `cases`, `series`, `validation.mom6_bayiw.read_mom6.clock_table`.
- Produces:
  - `mom6.source_centres(dz: np.ndarray) -> np.ndarray` — centres whose midpoints are the case interfaces
  - `mom6.mom_input(case: Case, coord: str, nu: float, dt: float, hours: float) -> str` (`coord` ∈ `zstar`, `rho`, `sigma`)
  - `mom6.write_case(case_name: str, coord: str, nu: float, out_dir: Path, dt: float | None = None, hours: float | None = None) -> Path`
  - `mom6.read(run_dir: Path, case_name: str) -> dict` (series fields; t = 0 from `MOM_IC.nc`)
  - `mom6.wall_s(run_dir: Path) -> float` (FMS "Main loop")
  - `run_mom6.sh run <case> <coord> <nu> [dt] [hours] [tag]` → `series.npz` in `cases.run_dir(case, 'mom6', coord, nu, tag)`

- [ ] **Step 1: Write the failing tests**

```python
def test_mom6_source_centres_reproduce_the_case_interfaces():
    from validation.ilicak_benchmark import cases, mom6
    for name in ("lock", "overflow", "iw", "seamount"):
        c = cases.get(name)
        cen = mom6.source_centres(c.dz)
        np.testing.assert_allclose(0.5 * (cen[1:] + cen[:-1]), np.cumsum(c.dz)[:-1], atol=1e-9)


def test_mom6_input_free_slip_linear_eos_and_coordinates():
    from validation.ilicak_benchmark import cases, mom6
    c = cases.get("lock")
    txt = mom6.mom_input(c, "rho", 10.0, 10.0, 17.0)
    p = dict(l.split(" = ", 1) for l in txt.splitlines() if " = " in l)
    assert p["NIGLOBAL"] == "128" and p["NJGLOBAL"] == "1" and p["NK"] == "20" and p["REENTRANT_Y"] == "False"
    assert p["BOTTOMDRAGLAW"] == "True" and p["CDRAG"] == "0.0" and p["KV_BBL_MIN"] == "0.0" and p["NOSLIP"] == "False"
    assert p["DRHO_DT"] == "-0.2" and p["DRHO_DS"] == "0.0" and p["RHO_REF_LINEAR_EOS"] == "1000.0"
    assert p["G_EARTH"] == "9.81" and p["KH"] == "10.0" and p["KV"] == "0.0001" and p["KHTR"] == "0.0" and p["KD"] == "0.0"
    assert p["REGRIDDING_COORDINATE_MODE"] == '"RHO"' and p["COORD_CONFIG"] == '"layer_ref"'
    t = cases.targets(c, 20)
    assert np.isclose(float(p["LIGHTEST_DENSITY"]), 0.5 * (t[0] + t[1]))
    assert np.isclose(float(p["DENSITY_RANGE"]), 0.5 * (t[-2] + t[-1]) - 0.5 * (t[0] + t[1]))
    assert p["TRACER_ADVECTION_SCHEME"] == '"PPM:H3"' and p["KE_SCHEME"] == '"KE_ARAKAWA"'
    assert p["DT"] == "10.0" and p["DAYMAX"] == "61200.0" and p["TIMEUNIT"] == "1.0" and p["SAVE_INITIAL_CONDS"] == "True"
    z = dict(l.split(" = ", 1) for l in mom6.mom_input(c, "zstar", 10.0, 10.0, 17.0).splitlines() if " = " in l)
    assert z["REGRIDDING_COORDINATE_MODE"] == '"ZSTAR"' and "COORD_CONFIG" not in z
    s = dict(l.split(" = ", 1) for l in mom6.mom_input(cases.get("seamount"), "sigma", 0.0, 600.0, 240.0)
             .splitlines() if " = " in l)
    assert s["REGRIDDING_COORDINATE_MODE"] == '"SIGMA"' and float(s["F_0"]) == 1e-4 and s["NJGLOBAL"] == "48"


def _fake_mom6_output(d, c, nt=2):
    """prog.nc + MOM_IC.nc in MOM6's layout (surface first, u on NI+1 faces) holding a known field."""
    nz = c.nz
    T = np.arange(nz, dtype=float)[:, None, None] * np.ones((nz, c.ny, c.nx))       # T = k: k = 0 at the surface
    h = np.broadcast_to(c.dz[:, None, None], (nz, c.ny, c.nx))
    for fname, times, tvar in (("prog.nc", np.arange(1, nt + 1) * c.out_s / 86400.0, "temp"),
                               ("MOM_IC.nc", np.array([0.0]), "Temp")):
        with netCDF4.Dataset(d / fname, "w") as ds:
            for n, sz in (("Time", None), ("zl", nz), ("yh", c.ny), ("xh", c.nx), ("xq", c.nx + 1), ("yq", c.ny + 1)):
                ds.createDimension(n, sz)
            ds.createVariable("Time", "f8", ("Time",))[:] = times
            ds["Time"].units = "days since 0001-01-01 00:00:00"
            ds.createVariable(tvar, "f8", ("Time", "zl", "yh", "xh"))[:] = np.broadcast_to(T, (times.size,) + T.shape)
            ds.createVariable("h", "f8", ("Time", "zl", "yh", "xh"))[:] = np.broadcast_to(h, (times.size,) + h.shape)
            ds.createVariable("u", "f8", ("Time", "zl", "yh", "xq"))[:] = 0.0
            ds.createVariable("v", "f8", ("Time", "zl", "yq", "xh"))[:] = 0.0
            if fname == "prog.nc":
                ds.createVariable("SSH", "f8", ("Time", "yh", "xh"))[:] = 0.0


def test_mom6_reader_puts_the_initial_state_first_and_keeps_surface_first(tmp_path):
    from validation.ilicak_benchmark import cases, mom6
    c = cases.get("lock")
    _fake_mom6_output(tmp_path, c)
    s = mom6.read(tmp_path, "lock")
    assert s["t"].tolist() == [0.0, 900.0, 1800.0]
    assert s["T"].shape == (3, 20, 1, 128) and s["T"][0, 0, 0, 0] == 0.0 and s["T"][0, -1, 0, 0] == 19.0
    np.testing.assert_allclose(s["h"][1].sum(0), 20.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k mom6`
Expected: FAIL — `ImportError: cannot import name 'mom6'`.

- [ ] **Step 3: Implement `mom6.py` and `run_mom6.sh`**

`validation/ilicak_benchmark/mom6.py`:

```python
"""
MOM6 runs of the four-model benchmark (spec §A2): cartesian grid (GRID_CONFIG 'cartesian', AXIS_UNITS 'k'), depth
file, T on the case z-levels as a z-file with ALE remapping (MOM_state_initialization.F90:2697-2722; the data are on
the model grid, so only the vertical remap), linear EOS in T, PPM_H4 remap, PPM:H3 tracers, KE_ARAKAWA, Laplacian
KH, KV = 1e-4, no tracer diffusion, free-slip bed (BOTTOMDRAGLAW with CDRAG = 0 and KV_BBL_MIN = 0; with
BOTTOMDRAGLAW = False the bed is viscous no-slip, MOM_vert_friction.F90:2263-2271), g = 9.81.
Coordinates: 'zstar', 'rho' (COORD_CONFIG 'layer_ref' with cases.targets), 'sigma'. 2-D cases: NJGLOBAL = 1 on one
rank (NJ = 1 fails on several ranks).
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from validation.ilicak_benchmark import cases

FILL = -1.0e20


def source_centres(dz: np.ndarray) -> np.ndarray:
    """Centres c_k [m, positive down] whose midpoints are the case interfaces (MOM6 builds the source edges as
    midpoints of the centres, MOM_horizontal_regridding.F90:454-480)."""
    zf = np.concatenate([[0.0], np.cumsum(dz)])
    c = np.empty(dz.size)
    c[0] = 0.5 * zf[1]
    for k in range(dz.size - 1):
        c[k + 1] = 2.0 * zf[k + 1] - c[k]
    return c


def mom_input(case: cases.Case, coord: str, nu: float, dt: float, hours: float) -> str:
    """The MOM_input text of one run."""
    modes = {"zstar": "ZSTAR", "rho": "RHO", "sigma": "SIGMA"}
    if coord not in modes:
        raise ValueError(f"unknown coord {coord!r}")
    L = [f"NIGLOBAL = {case.nx}", f"NJGLOBAL = {case.ny}", f"NK = {case.nz}",
         "REENTRANT_X = False", "REENTRANT_Y = False", "LAYOUT = 1, 1", "IO_LAYOUT = 1, 1",
         'INPUTDIR = "INPUT"', 'GRID_CONFIG = "cartesian"', 'AXIS_UNITS = "k"', "WESTLON = 0.0", "SOUTHLAT = 0.0",
         f"LENLON = {case.dx.sum() / 1e3:.10g}", f"LENLAT = {case.dy.sum() / 1e3:.10g}",
         'TOPO_CONFIG = "file"', 'TOPO_FILE = "topog.nc"', 'TOPO_VARNAME = "depth"',
         "MINIMUM_DEPTH = 0.5", "MASKING_DEPTH = 0.0", f"MAXIMUM_DEPTH = {float(np.nanmax(case.H)):.10g}",
         'ROTATION = "betaplane"', f"F_0 = {case.f0}", "BETA = 0.0", "G_EARTH = 9.81", "RHO_0 = 1000.0",
         'EQN_OF_STATE = "LINEAR"', "RHO_REF_LINEAR_EOS = 1000.0", f"T_REF_LINEAR_EOS = {cases.T_REF}",
         f"DRHO_DT = {-cases.ALPHA}", "DRHO_DS = 0.0",
         "USE_REGRIDDING = True", f'REGRIDDING_COORDINATE_MODE = "{modes[coord]}"', 'REMAPPING_SCHEME = "PPM_H4"']
    if coord == "rho":
        t = cases.targets(case, case.nz)
        lay = 0.5 * (t[1:] + t[:-1])
        L += ['COORD_CONFIG = "layer_ref"', f"LIGHTEST_DENSITY = {lay[0]:.12g}",
              f"DENSITY_RANGE = {lay[-1] - lay[0]:.12g}", 'INTERPOLATION_SCHEME = "P1M_H4"',
              "BOUNDARY_EXTRAPOLATION = True"]
    L += ["INIT_LAYERS_FROM_Z_FILE = True", 'TEMP_SALT_Z_INIT_FILE = "ts_z.nc"', 'Z_INIT_FILE_PTEMP_VAR = "ptemp"',
          'Z_INIT_FILE_SALT_VAR = "salt"', "Z_INIT_ALE_REMAPPING = True", 'Z_INIT_REMAPPING_SCHEME = "PPM_H4"',
          "TEMP_SALT_INIT_VERTICAL_REMAP_ONLY = True"]
    if coord != "zstar":
        L += ["Z_INIT_REMAP_GENERAL = True", "Z_INIT_REMAP_FULL_COLUMN = True"]
    L += ['TRACER_ADVECTION_SCHEME = "PPM:H3"', 'KE_SCHEME = "KE_ARAKAWA"', "LAPLACIAN = True", f"KH = {float(nu)}",
          "BIHARMONIC = False", "NOSLIP = False", f"KV = {cases.NU_V}", "KHTR = 0.0", "KD = 0.0",
          "BOTTOMDRAGLAW = True", "CDRAG = 0.0", "KV_BBL_MIN = 0.0", "HBBL = 0.001",
          "HMIX_FIXED = 0.5", "KV_ML_INVZ2 = 0.0", 'BUOY_CONFIG = "zero"', 'WIND_CONFIG = "zero"',
          f"DT = {float(dt)}", f"DT_THERM = {float(dt)}", "DTBT = -0.98",
          "TIMEUNIT = 1.0", f"DAYMAX = {hours * 3600.0}", f"ENERGYSAVEDAYS = {case.out_s}", "RESTART_CONTROL = -1",
          "SAVE_INITIAL_CONDS = True", 'IC_OUTPUT_FILE = "MOM_IC"']
    return "\n".join(L) + "\n"


DIAG = '''"Ilicak {name}"
1 1 1 0 0 0
"prog", {out:g}, "seconds", 1, "seconds", "Time",
"ocean_model", "SSH", "SSH", "prog", "all", .false., "none", 2
"ocean_model", "temp", "temp", "prog", "all", .false., "none", 2
"ocean_model", "u", "u", "prog", "all", .false., "none", 2
"ocean_model", "v", "v", "prog", "all", .false., "none", 2
"ocean_model", "h", "h", "prog", "all", .false., "none", 2
'''

NML = ''' &MOM_input_nml
         output_directory = './',
         input_filename = 'n'
         restart_input_dir = './',
         restart_output_dir = 'RESTART/',
         parameter_filename = 'MOM_input',
                              'MOM_override' /
 &diag_manager_nml
 /
 &fms_nml
         clock_grain = 'ROUTINE',
         domains_stack_size = 4000000,
         stack_size = 0 /
'''


def write_case(case_name: str, coord: str, nu: float, out_dir: Path, dt: float | None = None,
               hours: float | None = None) -> Path:
    """A ready-to-run MOM6 directory: MOM_input, MOM_override, diag_table, input.nml, INPUT/{topog,ts_z}.nc, RESTART/."""
    c = cases.get(case_name)
    out = Path(out_dir)
    (out / "RESTART").mkdir(parents=True, exist_ok=True)
    (out / "INPUT").mkdir(exist_ok=True)
    with netCDF4.Dataset(out / "INPUT" / "topog.nc", "w") as ds:
        ds.createDimension("ny", c.ny)
        ds.createDimension("nx", c.nx)
        ds.createVariable("depth", "f8", ("ny", "nx"))[:] = np.nan_to_num(c.H)
    cen = source_centres(c.dz)
    T = c.T_fn(c.x[None, None, :], c.y[None, :, None], -cen[:, None, None])
    T = np.where(cen[:, None, None] < c.H[None] + 0.5 * c.dz[:, None, None], T, FILL)     # below the bed: fill
    with netCDF4.Dataset(out / "INPUT" / "ts_z.nc", "w") as ds:
        for n, s in (("Time", None), ("depth", c.nz), ("lat", c.ny), ("lon", c.nx)):
            ds.createDimension(n, s)
        ds.createVariable("Time", "f8", ("Time",))[:] = [0.0]
        ds.createVariable("depth", "f8", ("depth",))[:] = cen
        ds["depth"].units, ds["depth"].positive = "m", "down"
        ds.createVariable("lat", "f8", ("lat",))[:] = c.y / 1e3
        ds.createVariable("lon", "f8", ("lon",))[:] = c.x / 1e3
        for name, val in (("ptemp", T), ("salt", np.where(T == FILL, FILL, 35.0))):
            v = ds.createVariable(name, "f8", ("Time", "depth", "lat", "lon"), fill_value=FILL)
            v[0] = val
    c_dt, c_h = (c.dt if dt is None else dt), (c.hours if hours is None else hours)
    (out / "MOM_input").write_text(mom_input(c, coord, nu, c_dt, c_h))
    (out / "MOM_override").write_text("")
    (out / "diag_table").write_text(DIAG.format(name=case_name, out=min(c.out_s, c_h * 3600.0)))
    (out / "input.nml").write_text(NML)
    return out


def read(run_dir: Path, case_name: str) -> dict:
    """Series fields from prog.nc, with record 0 = MOM_IC.nc (the model's own initial state); layers surface first."""
    c = cases.get(case_name)
    run_dir = Path(run_dir)
    fields = {k: [] for k in ("t", "h", "T", "u", "v", "eta")}
    for fname, tvar in (("MOM_IC.nc", "Temp"), ("prog.nc", "temp")):
        with netCDF4.Dataset(run_dir / fname) as ds:
            t = np.asarray(ds["Time"][:], float) * 86400.0
            for n in range(t.size):
                fields["t"].append(t[n] if fname == "prog.nc" else 0.0)
                fields["h"].append(np.asarray(ds["h"][n], float))
                fields["T"].append(np.asarray(ds[tvar][n].filled(np.nan) if np.ma.isMA(ds[tvar][n])
                                              else ds[tvar][n], float))
                fields["u"].append(np.asarray(np.ma.filled(ds["u"][n], 0.0), float))
                fields["v"].append(np.asarray(np.ma.filled(ds["v"][n], 0.0), float))
                fields["eta"].append(np.asarray(np.ma.filled(ds["SSH"][n], 0.0), float) if "SSH" in ds.variables
                                     else np.zeros((c.ny, c.nx)))
    s = {k: np.array(v) for k, v in fields.items()}
    s.update(x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H)
    return s


def wall_s(run_dir: Path) -> float:
    """Main-loop time [s] of the FMS clock table in stdout.txt."""
    from validation.mom6_bayiw.read_mom6 import clock_table
    return clock_table(Path(run_dir) / "stdout.txt")["Main loop"]
```

Note for the implementer: `MOM_IC.nc` may name the thickness `h` and temperature `Temp` (restart names, `MOM_state_initialization`/`MOM_restart`); the reader handles `Temp`. If the real file differs (check `ncdump -h MOM_IC.nc` in Task 7's smoke run), adjust the names in `read` and in `_fake_mom6_output` together, and ledger a ruling.

`validation/ilicak_benchmark/run_mom6.sh`:

```bash
#!/usr/bin/env bash
# MOM6 runs of the four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A2).
#   run_mom6.sh run <case> <coord> <nu> [dt] [hours] [tag]   : write, run on NP ranks (default 1), read → series.npz
# The repro build (-O2) of validation/mom6_bayiw/run_mom6.sh build is reused; MPICH may hang in MPI_Finalize, so the
# launcher waits for "Total runtime" and stops only its own ranks.
set -euo pipefail
EXE=~/pyOCEAN_refs/MOM6/.testing/build/repro/MOM6
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
[[ "${1:-}" == "run" && $# -ge 4 ]] || { echo "usage: $0 run <case> <coord> <nu> [dt] [hours] [tag]" >&2; exit 2; }
CASE=$2; COORD=$3; NU=$4; DT=${5:-}; HRS=${6:-}; TAG=${7:-}
D=$(cd "$ROOT" && python -c "from validation.ilicak_benchmark import cases; \
print(cases.run_dir('$CASE', 'mom6', '$COORD', float('$NU'), '$TAG'))")
rm -rf "$D"; mkdir -p "$D"
(cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import mom6; \
mom6.write_case('$CASE', '$COORD', float('$NU'), Path('$D'), dt=${DT:-None}, hours=${HRS:-None})")
cd "$D"
mpirun -n "${NP:-1}" "$EXE" > stdout.txt 2>&1 &
pid=$!
for _ in $(seq 1 7200); do
  grep -q "Total runtime" stdout.txt 2>/dev/null && break
  kill -0 $pid 2>/dev/null || break
  sleep 1
done
sleep 2; pkill -P $pid 2>/dev/null || true; kill $pid 2>/dev/null || true
grep -q "Total runtime" stdout.txt || { grep -m3 FATAL stdout.txt || tail -5 stdout.txt; echo "FAILED $D" >&2; exit 1; }
(cd "$ROOT" && python -c "
import json
from pathlib import Path
from validation.ilicak_benchmark import cases, mom6, series
c = cases.get('$CASE'); dt = float('${DT:-0}') or c.dt; hrs = float('${HRS:-0}') or c.hours
s = mom6.read(Path('$D'), '$CASE')
meta = dict(model='MOM6', coord='$COORD', case='$CASE', nu=float('$NU'), dt=dt, n_steps=int(round(hrs * 3600 / dt)),
            wall_s=mom6.wall_s(Path('$D')), threads=int('${NP:-1}'))
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k mom6`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
chmod +x validation/ilicak_benchmark/run_mom6.sh
git add validation/ilicak_benchmark/mom6.py validation/ilicak_benchmark/run_mom6.sh tests/test_ilicak_benchmark.py
git commit -m "Ilicak benchmark: MOM6 writer, reader and runner

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 5: CROCO writer, reader and runner

**Files:**
- Create: `validation/ilicak_benchmark/croco.py`, `validation/ilicak_benchmark/run_croco.sh`
- Test: `tests/test_ilicak_benchmark.py` (append)

**Interfaces:**
- Consumes: `cases`, `series`; CROCO source `~/pyOCEAN_refs/croco/OCEAN/{cppdefs.h,param.h}`; `~/pyOCEAN_refs/croco_runs/grav_adj/jobcomp`.
- Produces:
  - `croco.KEYS = {"lock": "PYO_LOCK", "overflow": "PYO_OVERFLOW", "iw": "PYO_IW", "seamount": "PYO_SEAMOUNT"}`
  - `croco.cppdefs(case_name: str, src: str) -> str`; `croco.param(case_name: str, src: str) -> str` (source text in, run-dir text out)
  - `croco.croco_in(case: Case, nu: float, dt: float, hours: float) -> str`
  - `croco.write_build(case_name: str, build_dir: Path) -> Path` (jobcomp, cppdefs.h, param.h)
  - `croco.write_run(case_name: str, nu: float, run_dir: Path, dt=None, hours=None) -> Path` (croco_grd.nc, croco_ini.nc, croco.in)
  - `croco.read(run_dir: Path, case_name: str) -> dict`
  - `croco.wall_s(run_dir: Path) -> float` (`real` of `/usr/bin/time -p` in `time.txt`)
  - `run_croco.sh build <case>`; `run_croco.sh run <case> sigma <nu> [dt] [hours] [tag]`

- [ ] **Step 1: Write the failing tests**

```python
CROCO_SRC = __import__("pathlib").Path.home() / "pyOCEAN_refs/croco/OCEAN"


@pytest.mark.skipif(not CROCO_SRC.exists(), reason="CROCO source not on disk")
def test_croco_in_and_cppdefs_have_uniform_sigma_and_no_diffusion():
    from validation.ilicak_benchmark import cases, croco
    cpp = croco.cppdefs("overflow", (CROCO_SRC / "cppdefs.h").read_text())
    assert "#define PYO_OVERFLOW" in cpp and "#undef REGIONAL" in cpp and "#define REGIONAL" not in cpp
    block = cpp[cpp.index("defined PYO_LOCK"):cpp.index("END OF CONFIGURATION CHOICE")]
    for key in ("NEW_S_COORD", "UV_VIS2", "UV_HADV_UP3", "UV_VADV_SPLINES", "TS_HADV_RSUP3", "TS_VADV_SPLINES",
                "OUT_DOUBLE", "ANA_SMFLUX", "ANA_STFLUX", "ANA_BTFLUX", "NO_FRCFILE"):
        assert f"# define {key}" in block, key
    for key in ("ANA_GRID", "ANA_INITIAL", "SALINITY", "NONLIN_EOS", "TS_DIF2", "UV_VIS_SMAGO", "LMD_MIXING",
                "GLS_MIXING", "MASKING", "UV_COR"):
        assert f"# undef  {key}" in block, key
    par = croco.param("overflow", (CROCO_SRC / "param.h").read_text())
    assert "parameter (LLm0=200, MMm0=1, N=100)" in par
    txt = croco.croco_in(cases.get("overflow"), 100.0, 10.0, 40.0)
    assert "0.d0 0.d0 1.e16" in txt and "1.e-4 0." in txt and "100.0 0." in txt
    assert "0. 0. 0. 0. 0." in txt and "0. 0.0 0. 0.2 0." in txt
    lines = txt.splitlines()
    ts = lines[lines.index("time_stepping: NTIMES dt[sec] NDTFAST NINFO") + 1].split()
    assert ts[:2] == ["14400", "10.0"]
    sm = croco.cppdefs("seamount", (CROCO_SRC / "cppdefs.h").read_text())
    assert "# define UV_COR" in sm[sm.index("defined PYO_SEAMOUNT"):]


def test_croco_grid_and_initial_files_hold_the_case(tmp_path):
    from validation.ilicak_benchmark import cases, croco
    c = cases.get("overflow")
    croco.write_run("overflow", 100.0, tmp_path)
    with netCDF4.Dataset(tmp_path / "croco_grd.nc") as ds:
        h = ds["h"][:]
        assert h.shape == (3, 202) and ds["spherical"][:].tobytes().strip() in (b"F", b"f")
        np.testing.assert_allclose(h[1, 1:-1], c.H[0])
        np.testing.assert_allclose(ds["pm"][:], 1 / 1e3)
    with netCDF4.Dataset(tmp_path / "croco_ini.nc") as ds:
        T = ds["temp"][0]
        assert T.shape == (100, 3, 202) and ds["scrum_time"].units.startswith("second")
        assert (T[:, 1, 1] == 10.0).all() and (T[:, 1, -2] == 20.0).all()   # s_rho bottom first; x < 20 km cold


def _fake_croco_his(d, c, nt=2):
    """A history file in CROCO's layout: s_rho bottom first, ghost rho columns/rows, u on LLm+1 faces."""
    N, L, M = c.nz, c.nx, c.ny
    with netCDF4.Dataset(d / "croco_his.nc", "w") as ds:
        for n, sz in (("time", None), ("s_rho", N), ("s_w", N + 1), ("eta_rho", M + 2), ("xi_rho", L + 2),
                      ("xi_u", L + 1), ("eta_v", M + 1)):
            ds.createDimension(n, sz)
        ds.createVariable("scrum_time", "f8", ("time",))[:] = np.arange(nt) * c.out_s
        ds.createVariable("h", "f8", ("eta_rho", "xi_rho"))[:] = np.pad(c.H, 1, mode="edge")
        ds.createVariable("s_w", "f8", ("s_w",))[:] = np.linspace(-1, 0, N + 1)
        ds.createVariable("Cs_w", "f8", ("s_w",))[:] = -np.linspace(-1, 0, N + 1) ** 2
        ds.createVariable("hc", "f8", ())[:] = 1e16
        ds.createVariable("zeta", "f8", ("time", "eta_rho", "xi_rho"))[:] = 0.0
        k = np.arange(N, dtype=float)[:, None, None] * np.ones((N, M + 2, L + 2))   # temp = k, k = 0 at the bed
        ds.createVariable("temp", "f8", ("time", "s_rho", "eta_rho", "xi_rho"))[:] = np.broadcast_to(k, (nt,) + k.shape)
        ds.createVariable("u", "f8", ("time", "s_rho", "eta_rho", "xi_u"))[:] = 0.0
        ds.createVariable("v", "f8", ("time", "s_rho", "eta_v", "xi_rho"))[:] = 0.0


def test_croco_reader_flips_to_surface_first_and_drops_ghosts(tmp_path):
    from validation.ilicak_benchmark import cases, croco
    c = cases.get("lock")
    _fake_croco_his(tmp_path, c)
    s = croco.read(tmp_path, "lock")
    assert s["t"].tolist() == [0.0, 900.0] and s["T"].shape == (2, 20, 1, 128) and s["u"].shape == (2, 20, 1, 129)
    assert s["T"][0, 0, 0, 0] == 19.0 and s["T"][0, -1, 0, 0] == 0.0                # surface first
    np.testing.assert_allclose(s["h"][0], 1.0, rtol=1e-12)                          # uniform σ: 20 m / 20
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k croco`
Expected: FAIL — `ImportError: cannot import name 'croco'`.

- [ ] **Step 3: Implement `croco.py` and `run_croco.sh`**

`validation/ilicak_benchmark/croco.py`:

```python
"""
CROCO runs of the four-model benchmark (spec §A2): σ coordinate, grid and initial state read from NetCDF (no
analytic routine is edited), own case keys PYO_* in a run-dir cppdefs.h/param.h (they override the source copies,
jobcomp), the REGIONAL default schemes (UV_HADV_UP3, UV_VADV_SPLINES, TS_HADV_RSUP3, TS_VADV_SPLINES; RSUP3 carries
an explicit biharmonic diffusion |u|Δx³/12, cppdefs_dev.h:423-436 — class c), constant Laplacian ν_h along σ
(UV_VIS2 + UV_MIX_S, no Smagorinsky), Akv = 1e-4 and Akt = 0, no drag, free slip (gamma2 = 1), linear EOS
ρ = 1000 + R0 − Tcoef (T − T0) with R0 = 0, T0 = T_ref, Tcoef = 0.2, g = 9.81 (scalars.h:633).
Uniform σ needs NEW_S_COORD with θs = θb = 0 and hc = 1e16 (set_scoord.F:169-184, set_depth.F:158-189).
RESET_RHO0 is forced for a linear EOS (cppdefs_dev.h:264-270): ρ₀ = 1000 + volume mean of σ_t (~1e-3 of g/ρ₀).
"""
from __future__ import annotations

import re
from pathlib import Path

import netCDF4
import numpy as np

from validation.ilicak_benchmark import cases

KEYS = {"lock": "PYO_LOCK", "overflow": "PYO_OVERFLOW", "iw": "PYO_IW", "seamount": "PYO_SEAMOUNT"}
SRC = Path.home() / "pyOCEAN_refs/croco/OCEAN"
JOBCOMP = Path.home() / "pyOCEAN_refs/croco_runs/grav_adj/jobcomp"

_BLOCK = """#elif defined PYO_LOCK || defined PYO_OVERFLOW || defined PYO_IW || defined PYO_SEAMOUNT
/* pyOCEAN four-model benchmark (validation/ilicak_benchmark/croco.py) */
# undef  OPENMP
# undef  MPI
# define SOLVE3D
# define NEW_S_COORD
# define UV_ADV
{cor}
# define UV_VIS2
# undef  UV_VIS_SMAGO
# define UV_MIX_S
# define UV_HADV_UP3
# define UV_VADV_SPLINES
# define TS_HADV_RSUP3
# define TS_VADV_SPLINES
# undef  TS_DIF2
# undef  TS_DIF4
# undef  SALINITY
# undef  NONLIN_EOS
# undef  LMD_MIXING
# undef  GLS_MIXING
# undef  MASKING
# undef  ANA_GRID
# undef  ANA_INITIAL
# define ANA_SMFLUX
# define ANA_STFLUX
# define ANA_BTFLUX
# define NO_FRCFILE
# define OUT_DOUBLE
"""


def cppdefs(case_name: str, src: str) -> str:
    """The run-dir cppdefs.h: REGIONAL off, the case key on, the PYO_* block before the end of the choices."""
    cor = "# define UV_COR" if case_name == "seamount" else "# undef  UV_COR"
    out = re.sub(r"^#define REGIONAL\b.*$", f"#undef REGIONAL\n#define {KEYS[case_name]}", src, count=1, flags=re.M)
    anchor = "#endif /* END OF CONFIGURATION CHOICE */"
    if anchor not in out:
        raise ValueError("cppdefs.h: end-of-choices anchor not found")
    return out.replace(anchor, _BLOCK.format(cor=cor) + anchor, 1)


def param(case_name: str, src: str) -> str:
    """The run-dir param.h with the PYO_* grid sizes before the REGIONAL entry."""
    c = cases.get(case_name)
    add = f"#elif defined {KEYS[case_name]}\n      parameter (LLm0={c.nx}, MMm0={c.ny}, N={c.nz})\n"
    anchor = "#elif defined REGIONAL"
    if anchor not in src:
        raise ValueError("param.h: REGIONAL anchor not found")
    return src.replace(anchor, add + anchor, 1)


def _ndtfast(case: cases.Case, dt: float) -> int:
    """Barotropic substeps: explicit 2-D step with √(gH) Δt_fast / Δx ≤ 0.1."""
    c = np.sqrt(cases.G * float(np.nanmax(case.H)))
    return max(10, int(np.ceil(dt * c / (0.1 * float(case.dx.min())))))


def croco_in(case: cases.Case, nu: float, dt: float, hours: float) -> str:
    """The croco.in text (keywords of read_inp.F; every keyword the CPP keys enable must be present)."""
    n = int(round(hours * 3600.0 / dt))
    nwrt = max(1, int(round(min(case.out_s, hours * 3600.0) / dt)))
    return "\n".join([
        "title:", f"        pyOCEAN benchmark {case.name}",
        "time_stepping: NTIMES dt[sec] NDTFAST NINFO", f"        {n} {float(dt)} {_ndtfast(case, dt)} {nwrt}",
        "S-coord: THETA_S, THETA_B, Hc (m)", "        0.d0 0.d0 1.e16",
        "grid: filename", "        croco_grd.nc",
        "initial: NRREC filename", "        1", "        croco_ini.nc",
        "restart: NRST, NRPFRST / filename", f"        {n} -1", "        croco_rst.nc",
        "history: LDEFHIS, NWRT, NRPFHIS / filename", f"        T {nwrt} 0", "        croco_his.nc",
        "primary_history_fields: zeta UBAR VBAR U V wrtT(1:NT)", "        T T T T T T",
        "auxiliary_history_fields: rho Omega W Akv Akt Aks Visc3d Diff3d HBL HBBL Bostr Bustr Bvstr Wstr Ustr Vstr "
        "Shfl rsw rlw lat sen Hm HEL BIOL", "        F F F F F F F F F F F F F F F F F F F F F F F F",
        "rho0:", "        1000.",
        "bottom_drag: RDRG(m/s), RDRG2, Zob [m], Cdb_min, Cdb_max", "        0. 0. 0. 0. 0.",
        "gamma2:", "        1.",
        "lin_EOS_cff: R0 [kg/m3], T0 [Celsius], S0 [PSU], TCOEF [1/Celsius], SCOEF [1/PSU]",
        f"        0. {cases.T_REF} 0. {cases.ALPHA} 0.",
        "vertical_mixing: Akv_bak, Akt_bak [m^2/sec]", "        1.e-4 0.",
        "lateral_visc: VISC2, VISC4 [m^2/sec]", f"        {float(nu)} 0.",
        ""])


def write_build(case_name: str, build_dir: Path) -> Path:
    """jobcomp + run-dir cppdefs.h and param.h for one case (a build per grid: CROCO sizes are compile-time)."""
    d = Path(build_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "jobcomp").write_text(JOBCOMP.read_text())
    (d / "jobcomp").chmod(0o755)
    (d / "cppdefs.h").write_text(cppdefs(case_name, (SRC / "cppdefs.h").read_text()))
    (d / "param.h").write_text(param(case_name, (SRC / "param.h").read_text()))
    return d


def _pad(a: np.ndarray) -> np.ndarray:
    return np.pad(a, [(0, 0)] * (a.ndim - 2) + [(1, 1), (1, 1)], mode="edge")


def write_run(case_name: str, nu: float, run_dir: Path, dt: float | None = None, hours: float | None = None) -> Path:
    """croco_grd.nc, croco_ini.nc (T at the uniform σ-layer centres, bottom first) and croco.in."""
    c = cases.get(case_name)
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    M, L, N = c.ny + 2, c.nx + 2, c.nz
    xr = np.concatenate([[c.x[0] - c.dx[0]], c.x, [c.x[-1] + c.dx[-1]]])
    yr = np.concatenate([[c.y[0] - c.dy[0]], c.y, [c.y[-1] + c.dy[-1]]])
    H = _pad(np.nan_to_num(c.H))
    with netCDF4.Dataset(d / "croco_grd.nc", "w") as ds:
        for n, sz in (("xi_rho", L), ("eta_rho", M), ("xi_u", L - 1), ("eta_v", M - 1), ("one", 1)):
            ds.createDimension(n, sz)
        ds.createVariable("spherical", "S1", ("one",))[:] = np.array([b"F"])
        ds.createVariable("xl", "f8", ())[:] = float(c.dx.sum())
        ds.createVariable("el", "f8", ())[:] = float(c.dy.sum())
        for name, val in (("h", H), ("f", np.full((M, L), c.f0)), ("pm", np.full((M, L), 1.0 / float(c.dx[0]))),
                          ("pn", np.full((M, L), 1.0 / float(c.dy[0]))),
                          ("x_rho", np.broadcast_to(xr[None], (M, L))), ("y_rho", np.broadcast_to(yr[:, None], (M, L)))):
            ds.createVariable(name, "f8", ("eta_rho", "xi_rho"))[:] = val
    sc = (np.arange(N) + 0.5) / N - 1.0                                 # σ-layer centres, bottom first, uniform
    z = sc[:, None, None] * H[None]                                      # η = 0
    T = c.T_fn(xr[None, None, :], yr[None, :, None], z)
    with netCDF4.Dataset(d / "croco_ini.nc", "w") as ds:
        for n, sz in (("time", None), ("s_rho", N), ("eta_rho", M), ("xi_rho", L), ("xi_u", L - 1), ("eta_v", M - 1)):
            ds.createDimension(n, sz)
        t = ds.createVariable("scrum_time", "f8", ("time",))
        t[:] = [0.0]
        t.units = "second"
        ds.createVariable("zeta", "f8", ("time", "eta_rho", "xi_rho"))[:] = 0.0
        ds.createVariable("ubar", "f8", ("time", "eta_rho", "xi_u"))[:] = 0.0
        ds.createVariable("vbar", "f8", ("time", "eta_v", "xi_rho"))[:] = 0.0
        ds.createVariable("u", "f8", ("time", "s_rho", "eta_rho", "xi_u"))[:] = 0.0
        ds.createVariable("v", "f8", ("time", "s_rho", "eta_v", "xi_rho"))[:] = 0.0
        ds.createVariable("temp", "f8", ("time", "s_rho", "eta_rho", "xi_rho"))[0] = T
    (d / "croco.in").write_text(croco_in(c, nu, c.dt if dt is None else dt, c.hours if hours is None else hours))
    return d


def read(run_dir: Path, case_name: str) -> dict:
    """Series fields of croco_his.nc: ghosts dropped, layers flipped to surface first, h from z_w (Vtransform 2:
    z_w = ζ + (ζ + h)(hc s_w + h Cs_w)/(hc + h), set_depth.F:158-189)."""
    c = cases.get(case_name)
    with netCDF4.Dataset(Path(run_dir) / "croco_his.nc") as ds:
        t = np.asarray(ds["scrum_time"][:], float)
        hb = np.asarray(ds["h"][:], float)[1:-1, 1:-1]
        sw, csw, hc = (np.asarray(ds[k][:], float) for k in ("s_w", "Cs_w", "hc"))
        zeta = np.asarray(ds["zeta"][:], float)[:, 1:-1, 1:-1]
        T = np.asarray(ds["temp"][:], float)[:, ::-1, 1:-1, 1:-1]
        u = np.asarray(ds["u"][:], float)[:, ::-1, 1:-1, :]
        v = np.asarray(ds["v"][:], float)[:, ::-1, :, 1:-1]
    zw = zeta[:, None] + (zeta[:, None] + hb[None, None]) * (
        (hc * sw[None, :, None, None] + hb[None, None] * csw[None, :, None, None]) / (hc + hb[None, None]))
    h = np.diff(zw, axis=1)[:, ::-1]
    return dict(t=t, x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H, h=h, T=T, u=u, v=v, eta=zeta)


def wall_s(run_dir: Path) -> float:
    """'real' seconds of /usr/bin/time -p (time.txt): CROCO prints no wall clock without MPI (timers_roms.F:40-77)."""
    for line in (Path(run_dir) / "time.txt").read_text().splitlines():
        if line.startswith("real"):
            return float(line.split()[1])
    raise ValueError("no 'real' line in time.txt")
```

Note for the implementer: CROCO's `u` in the history file has `xi_u = LLm + 1` faces, which are the nx + 1 x-faces of the interior (the two wall faces included); `v` has `eta_v = MMm + 1`. If the smoke run (Task 7) shows a different count, fix the slices in `read` and the fake file together and ledger it. The `wall_s` of CROCO includes initialisation (~0.1 s); declare it in the README.

`validation/ilicak_benchmark/run_croco.sh`:

```bash
#!/usr/bin/env bash
# CROCO runs of the four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A2).
#   run_croco.sh build <case>                              : jobcomp in RUNS/builds/croco_<case> (serial, -O2)
#   run_croco.sh run <case> sigma <nu> [dt] [hours] [tag]  : write, run, read → series.npz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark
case "${1:-}" in
  build)
    B="$RUNS/builds/croco_$2"; rm -rf "$B"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import croco; \
croco.write_build('$2', Path('$B'))")
    cd "$B" && ./jobcomp > build.log 2>&1 || true
    [[ -x "$B/croco" ]] || { tail -20 "$B/build.log"; echo "FAILED build $2" >&2; exit 1; }
    echo "OK $B/croco" ;;
  run)
    CASE=$2; NU=$4; DT=${5:-}; HRS=${6:-}; TAG=${7:-}
    B="$RUNS/builds/croco_$CASE"
    [[ -x "$B/croco" ]] || "$0" build "$CASE"
    D=$(cd "$ROOT" && python -c "from validation.ilicak_benchmark import cases; \
print(cases.run_dir('$CASE', 'croco', 'sigma', float('$NU'), '$TAG'))")
    rm -rf "$D"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import croco; \
croco.write_run('$CASE', float('$NU'), Path('$D'), dt=${DT:-None}, hours=${HRS:-None})")
    cd "$D"
    /usr/bin/time -p "$B/croco" croco.in > croco.log 2> time.txt || true
    grep -q "MAIN: DONE" croco.log || { tail -15 croco.log; echo "FAILED $D" >&2; exit 1; }
    (cd "$ROOT" && python -c "
from pathlib import Path
from validation.ilicak_benchmark import cases, croco, series
c = cases.get('$CASE'); dt = float('${DT:-0}') or c.dt; hrs = float('${HRS:-0}') or c.hours
s = croco.read(Path('$D'), '$CASE')
meta = dict(model='CROCO', coord='sigma', case='$CASE', nu=float('$NU'), dt=dt, n_steps=int(round(hrs * 3600 / dt)),
            wall_s=croco.wall_s(Path('$D')), threads=1)
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn) ;;
  *) echo "usage: $0 build <case> | run <case> sigma <nu> [dt] [hours] [tag]" >&2; exit 2 ;;
esac
```

Note for the implementer: the end-of-run marker of CROCO's log is assumed to be `MAIN: DONE`; check the end of `~/pyOCEAN_refs/croco_runs/grav_adj/croco.log` and use its actual last line if it differs (ledger it).

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k croco`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
chmod +x validation/ilicak_benchmark/run_croco.sh
git add validation/ilicak_benchmark/croco.py validation/ilicak_benchmark/run_croco.sh tests/test_ilicak_benchmark.py
git commit -m "Ilicak benchmark: CROCO writer, reader and runner

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 6: MITgcm writer, reader and runner

**Files:**
- Create: `validation/ilicak_benchmark/mitgcm.py`, `validation/ilicak_benchmark/run_mitgcm.sh`
- Test: `tests/test_ilicak_benchmark.py` (append)

**Interfaces:**
- Consumes: `cases`, `series`; MITgcm tree `/Volumes/ESTEBAN_MIT/MITgcm`.
- Produces:
  - `mitgcm.MIT = Path("/Volumes/ESTEBAN_MIT/MITgcm")`
  - `mitgcm.size_h(case: Case) -> str`; `mitgcm.write_build(case_name: str, code_dir: Path) -> Path` (SIZE.h, packages.conf, CPP_OPTIONS.h, DIAGNOSTICS_SIZE.h, GAD_OPTIONS.h)
  - `mitgcm.data(case: Case, nu: float, dt: float, hours: float) -> str`
  - `mitgcm.write_run(case_name: str, nu: float, run_dir: Path, dt=None, hours=None) -> Path` (data, data.pkg, data.diagnostics, eedata, bathy.bin, theta.bin)
  - `mitgcm.read(run_dir: Path, case_name: str) -> dict`
  - `mitgcm.wall_s(run_dir: Path) -> float`
  - `run_mitgcm.sh build <case>`; `run_mitgcm.sh run <case> zstar <nu> [dt] [hours] [tag]`
- Grid: one land column east (2-D: Nx = nx + 1, Ny = 1); seamount: one land column east and one land row north (Nx = 65, Ny = 49). The reader drops them.

- [ ] **Step 1: Write the failing tests**

```python
def test_mitgcm_data_and_size_for_zstar_os7mp_free_slip():
    from validation.ilicak_benchmark import cases, mitgcm
    c = cases.get("lock")
    size = mitgcm.size_h(c)
    for s in ("sNx = 129", "sNy =   1", "OLx =   4", "OLy =   4", "Nr  =  20"):
        assert s in size, s
    txt = mitgcm.data(c, 10.0, 10.0, 17.0)
    for s in ("eosType='LINEAR'", "tAlpha=2.E-4", "sBeta=0.", "rhoNil=1000.", "rhoConst=1000.", "gravity=9.81",
              "viscAh=10.0", "viscAr=1.E-4", "no_slip_sides=.FALSE.", "no_slip_bottom=.FALSE.", "diffKhT=0.",
              "diffKrT=0.", "tempAdvScheme=7", "staggerTimeStep=.TRUE.", "select_rStar=2", "nonlinFreeSurf=4",
              "exactConserv=.TRUE.", "implicitFreeSurface=.TRUE.", "hFacMin=0.1", "deltaT=10.0", "endTime=61200.0",
              "delX=129*500.0", "delR=20*1.0", "saltStepping=.FALSE.", "readBinaryPrec=64"):
        assert s in txt, s
    assert "tRef=20*0.0" in txt and "bottomDragLinear=0." in txt and "bottomDragQuadratic=0." in txt


def test_mitgcm_binary_inputs(tmp_path):
    from validation.ilicak_benchmark import cases, mitgcm
    c = cases.get("overflow")
    mitgcm.write_run("overflow", 100.0, tmp_path)
    b = np.fromfile(tmp_path / "bathy.bin", ">f8").reshape(1, 201)
    np.testing.assert_allclose(-b[0, :200], c.H[0])
    assert b[0, 200] == 0.0                                                      # the land column (east wall)
    T = np.fromfile(tmp_path / "theta.bin", ">f8").reshape(100, 1, 201)
    assert (T[:, 0, 0] == 10.0).all() and (T[:, 0, 199] == 20.0).all()


def _fake_mitgcm_mnc(d, c, nt=2, n_land=1):
    """MNC diagnostics in MITgcm's layout: surface first, one land column east, t = 0 absent (snapshots)."""
    Nx = c.nx + n_land
    with netCDF4.Dataset(d / "diag3d.0000000000.t001.nc", "w") as ds:
        for n, sz in (("T", None), ("Z", c.nz), ("Y", c.ny), ("X", Nx), ("Xp1", Nx + 1), ("Yp1", c.ny + 1)):
            ds.createDimension(n, sz)
        ds.createVariable("T", "f8", ("T",))[:] = np.arange(1, nt + 1) * c.out_s
        k = np.arange(c.nz, dtype=float)[:, None, None] * np.ones((c.nz, c.ny, Nx))
        ds.createVariable("THETA", "f8", ("T", "Z", "Y", "X"))[:] = np.broadcast_to(k, (nt,) + k.shape)
        ds.createVariable("UVEL", "f8", ("T", "Z", "Y", "Xp1"))[:] = 0.0
        ds.createVariable("VVEL", "f8", ("T", "Z", "Yp1", "X"))[:] = 0.0
        hf = np.ones((c.nz, c.ny, Nx))
        hf[:, :, -1] = 0.0
        ds.createVariable("hFactorC", "f8", ("T", "Z", "Y", "X"))[:] = np.broadcast_to(hf, (nt,) + hf.shape)
    with netCDF4.Dataset(d / "diag2d.0000000000.t001.nc", "w") as ds:
        for n, sz in (("T", None), ("Y", c.ny), ("X", Nx)):
            ds.createDimension(n, sz)
        ds.createVariable("T", "f8", ("T",))[:] = np.arange(1, nt + 1) * c.out_s
        ds.createVariable("ETAN", "f8", ("T", "Y", "X"))[:] = 0.0
    with netCDF4.Dataset(d / "grid.t001.nc", "w") as ds:
        for n, sz in (("Z", c.nz), ("Y", c.ny), ("X", Nx)):
            ds.createDimension(n, sz)
        ds.createVariable("HFacC", "f8", ("Z", "Y", "X"))[:] = hf
        ds.createVariable("drF", "f8", ("Z",))[:] = c.dz


def test_mitgcm_reader_adds_t0_drops_land_and_keeps_surface_first(tmp_path):
    from validation.ilicak_benchmark import cases, mitgcm
    c = cases.get("lock")
    _fake_mitgcm_mnc(tmp_path, c)
    mitgcm.write_run("lock", 10.0, tmp_path)                                                # theta.bin for t = 0
    s = mitgcm.read(tmp_path, "lock")
    assert s["t"].tolist() == [0.0, 900.0, 1800.0] and s["T"].shape == (3, 20, 1, 128)
    assert s["T"][1, 0, 0, 0] == 0.0 and s["T"][1, -1, 0, 0] == 19.0                        # k = 0 at the surface
    assert s["T"][0, 0, 0, 0] == 5.0 and s["T"][0, 0, 0, -1] == 30.0                        # t = 0 from theta.bin
    np.testing.assert_allclose(s["h"].sum(1), 20.0)
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k mitgcm`
Expected: FAIL — `ImportError: cannot import name 'mitgcm'`.

- [ ] **Step 3: Implement `mitgcm.py` and `run_mitgcm.sh`**

`validation/ilicak_benchmark/mitgcm.py`:

```python
"""
MITgcm runs of the four-model benchmark (spec §A2): z* (select_rStar = 2, nonlinFreeSurf = 4, NONLIN_FRSURF) with
partial cells (hFacMin = 0.1), implicit free surface, linear EOS in T (rhoNil = rhoConst = 1000, uniform tRef,
find_rho.F:98-108), OS7MP tracers (tempAdvScheme = 7, staggerTimeStep; OLx = OLy = 4, gad_check.F:115-128), its default
flux-form momentum, Laplacian viscAh, viscAr = 1e-4, no tracer diffusion, free slip (defaults are no-slip,
set_defaults.F:132-133), no drag, g = 9.81. Walls from land: one land column east (and one land row north in 3-D),
because the exchanges are periodic. Output: MNC diagnostics snapshots (THETA, UVEL, VVEL, hFactorC, ETAN); the cell
thickness under z* is hFactorC·drF (diagnostics_fill_state.F:451-453). t = 0 comes from the inputs (hFacC of the
grid file, theta.bin, η = 0). Copies of verification/advect_xz/code/{CPP_OPTIONS.h, DIAGNOSTICS_SIZE.h, GAD_OPTIONS.h}
(NONLIN_FRSURF defined).
"""
from __future__ import annotations

import glob
from pathlib import Path

import netCDF4
import numpy as np

from validation.ilicak_benchmark import cases

MIT = Path("/Volumes/ESTEBAN_MIT/MITgcm")


def _shape(case: cases.Case) -> tuple[int, int]:
    """(Nx, Ny) with the land column (and row in 3-D)."""
    return case.nx + 1, (case.ny + 1 if case.ny > 1 else 1)


def size_h(case: cases.Case) -> str:
    nx, ny = _shape(case)
    return f"""C SIZE.h of the pyOCEAN four-model benchmark ({case.name})
      INTEGER sNx, sNy, OLx, OLy, nSx, nSy, nPx, nPy, Nx, Ny, Nr
      PARAMETER (
     &           sNx = {nx:3d},
     &           sNy = {ny:3d},
     &           OLx =   4,
     &           OLy =   4,
     &           nSx =   1,
     &           nSy =   1,
     &           nPx =   1,
     &           nPy =   1,
     &           Nx  = sNx*nSx*nPx,
     &           Ny  = sNy*nSy*nPy,
     &           Nr  = {case.nz:3d})
      INTEGER MAX_OLX, MAX_OLY
      PARAMETER ( MAX_OLX = OLx, MAX_OLY = OLy )
"""


def write_build(case_name: str, code_dir: Path) -> Path:
    c = cases.get(case_name)
    d = Path(code_dir)
    d.mkdir(parents=True, exist_ok=True)
    (d / "SIZE.h").write_text(size_h(c))
    (d / "packages.conf").write_text("gfd\ndiagnostics\nmnc\n")
    src = MIT / "verification" / "advect_xz" / "code"
    for f in ("CPP_OPTIONS.h", "DIAGNOSTICS_SIZE.h", "GAD_OPTIONS.h"):
        (d / f).write_text((src / f).read_text())
    return d


def data(case: cases.Case, nu: float, dt: float, hours: float) -> str:
    nx, ny = _shape(case)
    dzs = ", ".join(f"{z:.10g}" for z in case.dz) if np.ptp(case.dz) else f"{case.nz}*{float(case.dz[0])}"
    return f""" &PARM01
 tRef={case.nz}*{cases.T_REF}, sRef={case.nz}*35., eosType='LINEAR', tAlpha=2.E-4, sBeta=0.,
 rhoNil=1000., rhoConst=1000., gravity=9.81,
 viscAh={float(nu)}, viscAr=1.E-4, implicitViscosity=.TRUE.,
 no_slip_sides=.FALSE., no_slip_bottom=.FALSE., bottomDragLinear=0., bottomDragQuadratic=0.,
 diffKhT=0., diffKrT=0., saltStepping=.FALSE.,
 tempAdvScheme=7, staggerTimeStep=.TRUE., f0={case.f0}, beta=0.,
 implicitFreeSurface=.TRUE., exactConserv=.TRUE., select_rStar=2, nonlinFreeSurf=4,
 hFacInf=0.2, hFacSup=2.0, hFacMin=0.1,
 readBinaryPrec=64, writeBinaryPrec=64,
 &
 &PARM02
 cg2dMaxIters=1000, cg2dTargetResidual=1.E-13,
 &
 &PARM03
 nIter0=0, deltaT={float(dt)}, endTime={hours * 3600.0}, abEps=0.1,
 dumpFreq=0., chkptFreq=0., pChkptFreq=0., monitorFreq={min(case.out_s, hours * 3600.0)},
 &
 &PARM04
 usingCartesianGrid=.TRUE., delX={nx}*{float(case.dx[0])}, delY={ny}*{float(case.dy[0])}, delR={dzs},
 &
 &PARM05
 bathyFile='bathy.bin', hydrogThetaFile='theta.bin',
 &
"""


def write_run(case_name: str, nu: float, run_dir: Path, dt: float | None = None, hours: float | None = None) -> Path:
    c = cases.get(case_name)
    d = Path(run_dir)
    d.mkdir(parents=True, exist_ok=True)
    nx, ny = _shape(c)
    dt = c.dt if dt is None else dt
    hours = c.hours if hours is None else hours
    (d / "data").write_text(data(c, nu, dt, hours))
    (d / "data.pkg").write_text(" &PACKAGES\n useDiagnostics=.TRUE., useMNC=.TRUE.,\n &\n")
    (d / "eedata").write_text(" &EEPARMS\n nTx=1, nTy=1,\n &\n")
    out = min(c.out_s, hours * 3600.0)
    (d / "data.diagnostics").write_text(f""" &DIAGNOSTICS_LIST
  fields(1:4,1) = 'THETA   ','UVEL    ','VVEL    ','hFactorC',
  fileName(1) = 'diag3d', frequency(1) = -{out}, timePhase(1) = 0.,
  fields(1,2) = 'ETAN    ', fileName(2) = 'diag2d', frequency(2) = -{out}, timePhase(2) = 0.,
 &
 &DIAG_STATIS_PARMS
 &
""")
    (d / "data.mnc").write_text(" &MNC_01\n mnc_use_outdir=.FALSE., mnc_outdir_str='mnc_',\n &\n")
    bathy = np.zeros((ny, nx))
    bathy[:c.ny, :c.nx] = -np.nan_to_num(c.H)                          # negative depth, 0 on land
    np.ascontiguousarray(bathy, ">f8").tofile(d / "bathy.bin")
    xs = np.concatenate([c.x, [c.x[-1] + c.dx[-1]]])
    ys = np.concatenate([c.y, [c.y[-1] + c.dy[-1]]])[:ny]
    T = c.T_fn(xs[None, None, :], ys[None, :, None], c.zc[:, None, None])
    np.ascontiguousarray(T, ">f8").tofile(d / "theta.bin")
    return d


def _join(pattern: str, name: str) -> np.ndarray:
    """One MNC variable (single tile in this setup)."""
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(pattern)
    with netCDF4.Dataset(files[0]) as ds:
        return np.asarray(ds[name][:], float)


def read(run_dir: Path, case_name: str) -> dict:
    """Series fields: diagnostics snapshots after a t = 0 record built from the inputs; land dropped; surface first."""
    c = cases.get(case_name)
    d = Path(run_dir)
    nx, ny = c.nx, c.ny
    t = _join(str(d / "diag3d.*.t001.nc"), "T")
    T = _join(str(d / "diag3d.*.t001.nc"), "THETA")[:, :, :ny, :nx]
    U = _join(str(d / "diag3d.*.t001.nc"), "UVEL")[:, :, :ny, :nx + 1]
    V = _join(str(d / "diag3d.*.t001.nc"), "VVEL")[:, :, :ny + 1, :nx]
    hf = _join(str(d / "diag3d.*.t001.nc"), "hFactorC")[:, :, :ny, :nx]
    eta = _join(str(d / "diag2d.*.t001.nc"), "ETAN")[:, :ny, :nx]
    drf = _join(str(d / "grid.t001.nc"), "drF")
    hf0 = _join(str(d / "grid.t001.nc"), "HFacC")[:, :ny, :nx]
    Nx, Ny = _shape(c)
    T0 = np.fromfile(d / "theta.bin", ">f8").reshape(c.nz, Ny, Nx)[:, :ny, :nx]
    if ny > 1:
        V[:, :, -1] = 0.0                                                  # the north wall face
    U[:, :, :, -1] = 0.0                                                   # the east wall face
    h = np.concatenate([(hf0 * drf[:, None, None])[None], hf * drf[None, :, None, None]])
    Tall = np.concatenate([T0[None], T])
    s = dict(t=np.concatenate([[0.0], t]), h=h, T=np.where(h > 0, Tall, np.nan),
             u=np.concatenate([np.zeros((1,) + U.shape[1:]), U]), v=np.concatenate([np.zeros((1,) + V.shape[1:]), V]),
             eta=np.concatenate([np.zeros((1, ny, nx)), eta]))
    s.update(x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=(hf0 * drf[:, None, None]).sum(0))
    return s


def wall_s(run_dir: Path) -> float:
    """Wall clock of section "MAIN_DO_LOOP [THE_MAIN_LOOP]" in output.txt."""
    lines = (Path(run_dir) / "output.txt").read_text().splitlines()
    for i, line in enumerate(lines):
        if 'Seconds in section "MAIN_DO_LOOP' in line:
            for nxt in lines[i:i + 6]:
                if "Wall clock time" in nxt:
                    return float(nxt.split(":")[-1])
    raise ValueError("no MAIN_DO_LOOP timing in output.txt")
```

`validation/ilicak_benchmark/run_mitgcm.sh`:

```bash
#!/usr/bin/env bash
# MITgcm runs of the four-model benchmark part A (spec 2026-10-13-ilicak-four-model-benchmark-design §A2).
#   run_mitgcm.sh build <case>                              : genmake2 + make in RUNS/builds/mitgcm_<case> (serial)
#   run_mitgcm.sh run <case> zstar <nu> [dt] [hours] [tag]  : write, run, read → series.npz
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
RUNS=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark
MIT=/Volumes/ESTEBAN_MIT/MITgcm
case "${1:-}" in
  build)
    B="$RUNS/builds/mitgcm_$2"; rm -rf "$B"; mkdir -p "$B/build"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import mitgcm; \
mitgcm.write_build('$2', Path('$B/code'))")
    cd "$B/build"; export NETCDF_ROOT=/opt/homebrew
    "$MIT/tools/genmake2" -rootdir "$MIT" -mods ../code -optfile "$MIT/tools/build_options/darwin_arm64_gfortran_esteban" \
        > genmake.log 2>&1
    make depend > make_depend.log 2>&1 && make -j 5 > make.log 2>&1 || true
    [[ -x mitgcmuv ]] || { tail -20 make.log; echo "FAILED build $2" >&2; exit 1; }
    echo "OK $B/build/mitgcmuv" ;;
  run)
    CASE=$2; NU=$4; DT=${5:-}; HRS=${6:-}; TAG=${7:-}
    B="$RUNS/builds/mitgcm_$CASE/build"
    [[ -x "$B/mitgcmuv" ]] || "$0" build "$CASE"
    D=$(cd "$ROOT" && python -c "from validation.ilicak_benchmark import cases; \
print(cases.run_dir('$CASE', 'mitgcm', 'zstar', float('$NU'), '$TAG'))")
    rm -rf "$D"
    (cd "$ROOT" && python -c "from pathlib import Path; from validation.ilicak_benchmark import mitgcm; \
mitgcm.write_run('$CASE', float('$NU'), Path('$D'), dt=${DT:-None}, hours=${HRS:-None})")
    cd "$D"; cp "$B/mitgcmuv" .
    ./mitgcmuv > output.txt 2>&1 || true
    grep -q "PROGRAM MAIN: Execution ended Normally" output.txt || { tail -15 output.txt; echo "FAILED $D" >&2; exit 1; }
    (cd "$ROOT" && python -c "
from pathlib import Path
from validation.ilicak_benchmark import cases, mitgcm, series
c = cases.get('$CASE'); dt = float('${DT:-0}') or c.dt; hrs = float('${HRS:-0}') or c.hours
s = mitgcm.read(Path('$D'), '$CASE')
meta = dict(model='MITgcm', coord='zstar', case='$CASE', nu=float('$NU'), dt=dt, n_steps=int(round(hrs * 3600 / dt)),
            wall_s=mitgcm.wall_s(Path('$D')), threads=1)
print(series.save(Path('$D') / 'series.npz', s, meta))" 2>&1 | grep -v Warn) ;;
  *) echo "usage: $0 build <case> | run <case> zstar <nu> [dt] [hours] [tag]" >&2; exit 2 ;;
esac
```

Note for the implementer: the reader test builds `theta.bin` with `write_run` and the MNC files by hand. If MITgcm's `grid.t001.nc` names differ (`HFacC`, `drF` are the MNC grid names in `internal_wave/case.py`; check), fix reader and fake file together and ledger it.

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k mitgcm`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
chmod +x validation/ilicak_benchmark/run_mitgcm.sh
git add validation/ilicak_benchmark/mitgcm.py validation/ilicak_benchmark/run_mitgcm.sh tests/test_ilicak_benchmark.py
git commit -m "Ilicak benchmark: MITgcm writer, reader and runner

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 7: Smoke runs — every model on a short lock exchange

**Files:**
- Modify (only if a smoke run disagrees with an assumption above; each change is a ledger ruling): `mom6.py`, `croco.py`, `mitgcm.py`, their scripts and fake-output helpers in the test file.
- Test: `tests/test_ilicak_benchmark.py` (append one test that reads the real smoke outputs, skipped when absent)

**Interfaces:**
- Consumes: all runners; `metrics.rpe`, `metrics.t_range`.
- Produces: confirmed runners; builds in `RUNS/builds/{croco,mitgcm}_<case>` for all four cases.

- [ ] **Step 1: Write the test on real outputs (skipped when absent)**

```python
SMOKE = [("pyocean", "zstar"), ("pyocean", "sigma"), ("pyocean", "hybrid"), ("mom6", "zstar"), ("mom6", "rho"),
         ("croco", "sigma"), ("mitgcm", "zstar")]


@pytest.mark.parametrize("model,coord", SMOKE)
def test_smoke_lock_exchange_one_hour(model, coord):
    from validation.ilicak_benchmark import cases, metrics, series
    p = cases.run_dir("lock", model, coord, 10.0, "smoke") / "series.npz"
    if not p.exists():
        pytest.skip("smoke run not on disk")
    s, meta = series.load(p)
    assert s["t"][0] == 0.0 and np.isclose(s["t"][-1], 3600.0) and meta["wall_s"] > 0
    np.testing.assert_allclose(s["h"].sum(1), 20.0 + s["eta"], atol=1e-6)            # column = depth + η
    lo, hi = metrics.t_range(s)
    assert 5.0 - 0.05 <= lo and hi <= 30.0 + 0.05
    r = metrics.rpe(s)
    assert r[-1] >= r[0] * (1 - 1e-9)                                                  # mixing never un-mixes
    assert metrics.max_speed(s)[-1] > 0.1                                              # the fronts are moving
```

- [ ] **Step 2: Build CROCO and MITgcm for every case, and run the smoke lock exchange in each configuration**

```bash
for c in lock overflow iw seamount; do validation/ilicak_benchmark/run_croco.sh build $c; done
for c in lock overflow iw seamount; do validation/ilicak_benchmark/run_mitgcm.sh build $c; done
for co in zstar sigma hybrid; do NUMBA_NUM_THREADS=5 python validation/ilicak_benchmark/run_pyocean.py lock $co 10 --hours 1 --tag smoke; done
validation/ilicak_benchmark/run_mom6.sh run lock zstar 10 "" 1 smoke
validation/ilicak_benchmark/run_mom6.sh run lock rho 10 "" 1 smoke
validation/ilicak_benchmark/run_croco.sh run lock sigma 10 "" 1 smoke
validation/ilicak_benchmark/run_mitgcm.sh run lock zstar 10 "" 1 smoke
```

Expected: 8 builds print `OK …`; 7 runs print the path of a `series.npz`. A build or run that fails: read its log, fix the generator (not the model source), ledger the ruling, re-run.

Note: an empty `""` for `dt` makes the scripts use the case Δt (`${DT:-None}` → `None` in Python); check this works in the first run and switch to explicit values if the shell passes the empty string differently.

- [ ] **Step 3: Run the smoke test**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_ilicak_benchmark.py -q -k smoke`
Expected: 7 passed. If MOM6's `MOM_IC.nc`, CROCO's face counts or MITgcm's grid names differ from the fake files, fix reader + fake together (ledger) until it passes.

- [ ] **Step 4: Run the whole test file**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_ilicak_benchmark.py -q`
Expected: all passed.

- [ ] **Step 5: Commit**

```bash
git add tests/test_ilicak_benchmark.py validation/ilicak_benchmark/mom6.py validation/ilicak_benchmark/croco.py validation/ilicak_benchmark/mitgcm.py validation/ilicak_benchmark/run_mom6.sh validation/ilicak_benchmark/run_croco.sh validation/ilicak_benchmark/run_mitgcm.sh
git commit -m "Ilicak benchmark: smoke runs of all models on the lock exchange

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

(Stage only the files that changed.)

---

### Task 8: The campaign driver and the results runs

**Files:**
- Create: `validation/ilicak_benchmark/campaign.py`
- Test: `tests/test_ilicak_benchmark.py` (append)

**Interfaces:**
- Consumes: the four runners; `cases`; `series`; `metrics.max_speed`.
- Produces:
  - `campaign.CONFIGS = {"lock": [...], "overflow": [...], "iw": [...], "seamount": [...]}` — (model, coord) per case
  - `campaign.command(case: str, model: str, coord: str, nu: float, dt: float | None = None, hours: float | None = None, tag: str = "") -> list[str]`
  - `campaign.results(case: str | None = None, force: bool = False) -> list[Path]` (skips existing `series.npz`)
  - `campaign.stable(path: Path, umax: float = 5.0) -> bool`
  - `campaign.cost(case: str, model: str, coord: str, factors=(1, 2, 4, 8, 16)) -> dict` (`dt_max`, `s_per_hour`, `s_per_hour_results_dt`)
  - CLI: `python validation/ilicak_benchmark/campaign.py results [case]` / `cost [case]`

- [ ] **Step 1: Write the failing tests**

```python
def test_campaign_matrix_and_commands():
    from validation.ilicak_benchmark import campaign
    two_d = [("pyocean", "zstar"), ("pyocean", "sigma"), ("pyocean", "hybrid"), ("mom6", "zstar"), ("mom6", "rho"),
             ("croco", "sigma"), ("mitgcm", "zstar")]
    for c in ("lock", "overflow", "iw"):
        assert campaign.CONFIGS[c] == two_d
    assert campaign.CONFIGS["seamount"] == [("pyocean", "sigma"), ("pyocean", "zstar"), ("mom6", "sigma"),
                                            ("mom6", "zstar"), ("croco", "sigma"), ("mitgcm", "zstar")]
    cmd = campaign.command("lock", "mom6", "rho", 10.0, dt=20.0, hours=1.0, tag="cost")
    assert cmd[-7:] == ["run", "lock", "rho", "10", "20", "1", "cost"] and cmd[0].endswith("run_mom6.sh")
    py = campaign.command("iw", "pyocean", "hybrid", 15.0)
    assert py[1].endswith("run_pyocean.py") and py[2:5] == ["iw", "hybrid", "15"]


def test_stable_rejects_nan_and_fast_flow(tmp_path):
    from validation.ilicak_benchmark import campaign, series
    s = _flat_series([1.0], [5.0], nx=2)
    series.save(tmp_path / "ok.npz", s, {})
    assert campaign.stable(tmp_path / "ok.npz")
    s["u"] = s["u"] + 9.0
    series.save(tmp_path / "fast.npz", s, {})
    assert not campaign.stable(tmp_path / "fast.npz")
    s["u"] = s["u"] * 0
    s["T"] = s["T"] * np.nan
    series.save(tmp_path / "nan.npz", s, {})
    assert not campaign.stable(tmp_path / "nan.npz")
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k "campaign or stable"`
Expected: FAIL — `ImportError: cannot import name 'campaign'`.

- [ ] **Step 3: Implement `campaign.py`**

```python
#!/usr/bin/env python
"""
Run matrix of the four-model benchmark part A (spec §A2): results runs (3 ν_h × 7 configurations per 2-D case, the
seamount in 6) and the cost ladder (Δt × 1, 2, 4, 8, 16 for 1 simulated hour — 1 day for the internal wave and the
seamount — the largest stable Δt gives the cost). One run at a time; pyOCEAN with 5 threads; MOM6, CROCO and MITgcm
on one process (NJ = 1 in MOM6 needs one rank; CROCO is a serial build).

Usage: python validation/ilicak_benchmark/campaign.py results [case] | cost [case]
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.ilicak_benchmark import cases, metrics, series  # noqa: E402

HERE = Path(__file__).resolve().parent
_TWO_D = [("pyocean", "zstar"), ("pyocean", "sigma"), ("pyocean", "hybrid"), ("mom6", "zstar"), ("mom6", "rho"),
          ("croco", "sigma"), ("mitgcm", "zstar")]
CONFIGS = {"lock": _TWO_D, "overflow": _TWO_D, "iw": _TWO_D,
           "seamount": [("pyocean", "sigma"), ("pyocean", "zstar"), ("mom6", "sigma"), ("mom6", "zstar"),
                        ("croco", "sigma"), ("mitgcm", "zstar")]}
COST_HOURS = {"lock": 1.0, "overflow": 1.0, "iw": 24.0, "seamount": 24.0}


def _num(v: float | None) -> str:
    return "" if v is None else f"{v:g}"


def command(case: str, model: str, coord: str, nu: float, dt: float | None = None, hours: float | None = None,
            tag: str = "") -> list[str]:
    """The command line of one run."""
    if model == "pyocean":
        cmd = [sys.executable, str(HERE / "run_pyocean.py"), case, coord, f"{nu:g}"]
        cmd += (["--dt", f"{dt:g}"] if dt else []) + (["--hours", f"{hours:g}"] if hours else [])
        return cmd + (["--tag", tag] if tag else [])
    return [str(HERE / f"run_{model}.sh"), "run", case, coord, f"{nu:g}", _num(dt), _num(hours), tag]


def _run(cmd: list[str]) -> None:
    env = {**os.environ, "NUMBA_NUM_THREADS": "5"}
    r = subprocess.run(cmd, cwd=ROOT, env=env, capture_output=True, text=True)
    print(" ".join(cmd), "→", (r.stdout.strip().splitlines() or [""])[-1], flush=True)
    if r.returncode:
        print(r.stdout[-2000:], r.stderr[-2000:], flush=True)


def results(case: str | None = None, force: bool = False) -> list[Path]:
    """Every results run of ``case`` (all cases if None); existing series.npz are kept unless force."""
    done = []
    for c in ([case] if case else list(CONFIGS)):
        for nu in cases.get(c).nu_h:
            for model, coord in CONFIGS[c]:
                p = cases.run_dir(c, model, coord, nu) / "series.npz"
                if force or not p.exists():
                    _run(command(c, model, coord, nu))
                if p.exists():
                    done.append(p)
    return done


def stable(path: Path, umax: float = 5.0) -> bool:
    """A run is stable when every T is finite where h > 0 and max |u|, |v| < umax [m s⁻¹]."""
    s, _ = series.load(path)
    T = np.where(s["h"] > 0, s["T"], 0.0)
    return bool(np.isfinite(T).all() and np.isfinite(s["u"]).all() and metrics.max_speed(s).max() < umax)


def cost(case: str, model: str, coord: str, factors=(1, 2, 4, 8, 16)) -> dict:
    """Largest stable Δt of the ladder (middle ν_h, COST_HOURS) and its cost [s per simulated hour]."""
    c = cases.get(case)
    nu = c.nu_h[len(c.nu_h) // 2]
    best = None
    for f in factors:
        dt = c.dt * f
        tag = f"cost_dt{dt:g}"
        _run(command(case, model, coord, nu, dt, COST_HOURS[case], tag))
        p = cases.run_dir(case, model, coord, nu, tag) / "series.npz"
        if not (p.exists() and stable(p)):
            break
        _, meta = series.load(p)
        best = dict(dt_max=dt, s_per_hour=meta["wall_s"] / COST_HOURS[case], threads=meta["threads"])
    res = cases.run_dir(case, model, coord, nu) / "series.npz"
    if res.exists():
        _, meta = series.load(res)
        (best := best or {}).update(s_per_hour_results_dt=meta["wall_s"] / (meta["n_steps"] * meta["dt"] / 3600.0))
    out = cases.RUNS / case / "cost.json"
    table = json.loads(out.read_text()) if out.exists() else {}
    table[f"{model}_{coord}"] = best
    out.write_text(json.dumps(table, indent=1))
    return best or {}


if __name__ == "__main__":
    what, which = sys.argv[1], (sys.argv[2] if len(sys.argv) > 2 else None)
    if what == "results":
        results(which)
    elif what == "cost":
        for cs in ([which] if which else list(CONFIGS)):
            for model, coord in CONFIGS[cs]:
                print(cs, model, coord, cost(cs, model, coord), flush=True)
    else:
        raise SystemExit("usage: campaign.py results [case] | cost [case]")
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k "campaign or stable"`
Expected: 2 passed.

- [ ] **Step 5: Commit**

```bash
chmod +x validation/ilicak_benchmark/campaign.py
git add validation/ilicak_benchmark/campaign.py tests/test_ilicak_benchmark.py
git commit -m "Ilicak benchmark: campaign driver (results matrix, cost ladder)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

- [ ] **Step 6: Results runs, one case at a time (background, one run at a time)**

```bash
python validation/ilicak_benchmark/campaign.py results lock      > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/log_results_lock.txt 2>&1
python validation/ilicak_benchmark/campaign.py results overflow  > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/log_results_overflow.txt 2>&1
python validation/ilicak_benchmark/campaign.py results iw        > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/log_results_iw.txt 2>&1
python validation/ilicak_benchmark/campaign.py results seamount  > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/log_results_seamount.txt 2>&1
```

Expected: 21 + 21 + 21 + 6 `series.npz`. Watch the first run of each case against the spec estimate (lock < 1 min, overflow 2–5 min, IW 1–5 min, seamount 2–5 min); report any run > 3× its estimate before going on. A failed run (`FAILED` in the log): a numerical blow-up is a result (record it, do not tune); a setup error is fixed in the generator with a ledger ruling.

---

### Task 9: Cost runs and HYCOM gate

**Files:**
- Create: `validation/ilicak_benchmark/hycom/gate.sh`, `validation/ilicak_benchmark/hycom/README.md`
- No new tests (the gate is a feasibility check; cost runs use the tested campaign driver).

**Interfaces:**
- Consumes: `campaign.cost`.
- Produces: `RUNS/<case>/cost.json`; `hycom/README.md` with the gate outcome.

- [ ] **Step 1: Cost ladder for all cases**

```bash
python validation/ilicak_benchmark/campaign.py cost > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/log_cost.txt 2>&1
```

Expected: four `cost.json` files with `dt_max`, `s_per_hour` and `s_per_hour_results_dt` for every configuration. Also measure pyOCEAN with one thread on the middle ν_h of each 2-D case, as the other models run on one process:

```bash
for c in lock overflow iw; do NUMBA_NUM_THREADS=1 python validation/ilicak_benchmark/run_pyocean.py $c zstar \
  $(python -c "from validation.ilicak_benchmark import cases; print(cases.get('$c').nu_h[1])") --threads 1 --tag t1 \
  --hours $(python -c "print({'lock': 1, 'overflow': 1, 'iw': 24}['$c'])"); done
```

- [ ] **Step 2: HYCOM gate script**

`validation/ilicak_benchmark/hycom/gate.sh`:

```bash
#!/usr/bin/env bash
# HYCOM 2.3 feasibility gate (spec §A3): build from a copied config without -mavx (arm64), run a resting basin for
# 10 steps on 5 MPI ranks. Time box: ~1 h of work; on failure stop and report (no further HYCOM work).
# HYCOM sources are not modified; the build lives in RUNS/hycom/build.
set -euo pipefail
SRC=~/pyOCEAN/HYCOM-src
B=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom/build
rm -rf "$B"; mkdir -p "$B"; cp -R "$SRC"/. "$B"/
cd "$B"
CFG=$(ls config/*generic*gnu*relo* 2>/dev/null | head -1 || true)
[[ -n "$CFG" ]] || { echo "GATE FAILED: no generic-gnu-relo config in HYCOM-src/config" >&2; exit 1; }
sed -i '' -e 's/-mavx//g' "$CFG"
( make hycom ARCH=generic-gnu-relo TYPE=mpi > make.log 2>&1 ) || { tail -30 make.log; echo "GATE FAILED: build" >&2; exit 1; }
ls -la hycom* | head
echo "GATE: build OK — next: the resting-basin case (10 steps, mpiexec -n 5)"
```

- [ ] **Step 3: Run the gate (time-boxed ~1 h of work)**

Run: `bash validation/ilicak_benchmark/hycom/gate.sh`
Expected: either `GATE: build OK` (then set up the resting basin with HYCOM's own sample inputs in `HYCOM-src` if present, 10 steps on 5 ranks) or `GATE FAILED: …`.
- **Gate fails, or the resting basin does not run within the time box:** write `hycom/README.md` with what was tried, the error, and "HYCOM cost not measured (gate, spec §A3)"; tell Esteban in the final message.
- **Gate passes:** stop and tell Esteban; the overflow cost case for HYCOM (`regional.grid/depth .a/.b`, `blkdat.input` with 100 hybrid layers, 7-term EOS with constant S) is written as a short addendum plan approved by him — it needs the `.a/.b` writer of the previous spec and is not specified here.

- [ ] **Step 4: Commit**

```bash
chmod +x validation/ilicak_benchmark/hycom/gate.sh
git add validation/ilicak_benchmark/hycom/gate.sh validation/ilicak_benchmark/hycom/README.md
git commit -m "Ilicak benchmark: cost ladder and HYCOM gate

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 10: Comparison tables, README, scorecard

**Files:**
- Create: `validation/ilicak_benchmark/compare.py`, `validation/ilicak_benchmark/README.md`
- Modify: `docs/design/vision_roadmap.md` (§7 scorecard), `CLAUDE.md` (Phase 2 table, test count), `docs/literature/reading_list.md` (Petersen 2015, Chen 2024 if missing)
- Test: `tests/test_ilicak_benchmark.py` (append)

**Interfaces:**
- Consumes: `series.load`, `metrics.*`, `cases`, `campaign.CONFIGS`, `cost.json`.
- Produces:
  - `compare.summary(case: str, model: str, coord: str, nu: float) -> dict` keys `rpe_rel_end` ((RPE−RPE₀)/RPE₀ at the end), `drpe_dt` [W m⁻²], `t_min`, `t_max`, plus per case: lock `dense_nose, light_nose, dense_body, light_body` (as /U_Benjamin), overflow `x_c_end` [km], `depth_c_end` [m], iw `amp_ratio`, seamount `umax_end` [cm s⁻¹]
  - `compare.table(case: str) -> list[str]` (markdown rows), CLI `python validation/ilicak_benchmark/compare.py [case]`
  - Windows (spec §A4): lock dRPE/dt from 15 to 17 h; overflow 13–37 h; IW days 10–100; IW `amp_ratio` = rms of `iw_amplitude` over the last 20 days / over the first 20 days.

- [ ] **Step 1: Write the failing test**

```python
def test_compare_summary_on_a_synthetic_lock_run(tmp_path, monkeypatch):
    from validation.ilicak_benchmark import cases, compare, series
    monkeypatch.setattr(cases, "RUNS", tmp_path)
    c = cases.get("lock")
    t = np.arange(0, 18) * 3600.0
    nz, nx = 20, 128
    T = np.where(c.x[None, None, None, :] < 32e3, 5.0, 30.0) * np.ones((t.size, nz, 1, nx))
    s = dict(t=t, x=c.x, y=c.y, dx=c.dx, dy=c.dy, H=c.H, h=np.ones((t.size, nz, 1, nx)), T=T,
             u=np.zeros((t.size, nz, 1, nx + 1)), v=np.zeros((t.size, nz, 2, nx)), eta=np.zeros((t.size, 1, nx)))
    series.save(cases.run_dir("lock", "mom6", "zstar", 10.0) / "series.npz", s,
                dict(model="MOM6", wall_s=1.0, n_steps=6120, dt=10.0))
    r = compare.summary("lock", "mom6", "zstar", 10.0)
    assert r["rpe_rel_end"] == 0.0 and r["drpe_dt"] == 0.0 and (r["t_min"], r["t_max"]) == (5.0, 30.0)
    assert r["dense_nose"] == 0.0                                       # nothing moved
    rows = compare.table("lock")
    assert any("MOM6 z*" in row and "10" in row for row in rows)
```

- [ ] **Step 2: Run it to see it fail**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k compare`
Expected: FAIL — `ImportError: cannot import name 'compare'`.

- [ ] **Step 3: Implement `compare.py`**

```python
#!/usr/bin/env python
"""
Tables of the four-model benchmark part A (spec §A4–A5): per case and ν_h, every configuration side by side:
(RPE − RPE₀)/RPE₀ at the end, dRPE/dt [W m⁻²] over the spec window, the T range, and the physics of the case;
then the cost table (cost.json). Missing runs show '—'.

Usage: python validation/ilicak_benchmark/compare.py [lock|overflow|iw|seamount]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from validation.ilicak_benchmark import campaign, cases, metrics, series  # noqa: E402

NAMES = {("pyocean", "zstar"): "pyOCEAN z*", ("pyocean", "sigma"): "pyOCEAN σ", ("pyocean", "hybrid"): "pyOCEAN híbrida",
         ("mom6", "zstar"): "MOM6 z*", ("mom6", "rho"): "MOM6 ρ", ("mom6", "sigma"): "MOM6 σ",
         ("croco", "sigma"): "CROCO σ", ("mitgcm", "zstar"): "MITgcm z*"}
WINDOW = {"lock": (15 * 3600.0, 17 * 3600.0), "overflow": (13 * 3600.0, 37 * 3600.0),
          "iw": (10 * 86400.0, 100 * 86400.0), "seamount": (0.0, 10 * 86400.0)}


def summary(case: str, model: str, coord: str, nu: float) -> dict:
    """The metrics of one run (spec §A4)."""
    s, meta = series.load(cases.run_dir(case, model, coord, nu) / "series.npz")
    r = metrics.rpe(s)
    lo, hi = metrics.t_range(s)
    out = dict(rpe_rel_end=float((r[-1] - r[0]) / r[0]), drpe_dt=metrics.drpe_dt(s, r, *WINDOW[case]),
               t_min=lo, t_max=hi, wall_s=meta.get("wall_s"))
    if case == "lock":
        out.update({k: v / metrics.BENJAMIN for k, v in metrics.lock_fronts(s).items()})
    elif case == "overflow":
        xc, dep = metrics.overflow_descent(s)
        out.update(x_c_end=xc[-1] / 1e3, depth_c_end=dep[-1])
    elif case == "iw":
        a, t = metrics.iw_amplitude(s), s["t"]
        first, last = t <= 20 * 86400.0, t >= t[-1] - 20 * 86400.0
        out.update(amp_ratio=float(np.sqrt(np.mean(a[last] ** 2)) / np.sqrt(np.mean(a[first] ** 2))))
    else:
        out.update(umax_end=float(metrics.max_speed(s)[-1] * 100.0))
    return out


def _fmt(v) -> str:
    return "—" if v is None else (f"{v:.3g}" if isinstance(v, float) else str(v))


def table(case: str) -> list[str]:
    extra = {"lock": ["dense_nose", "light_nose", "dense_body", "light_body"], "overflow": ["x_c_end", "depth_c_end"],
             "iw": ["amp_ratio"], "seamount": ["umax_end"]}[case]
    cols = ["rpe_rel_end", "drpe_dt", "t_min", "t_max"] + extra
    rows = [f"## {case}", "", "| ν_h | modelo | " + " | ".join(cols) + " |", "|" + "---|" * (len(cols) + 2)]
    for nu in cases.get(case).nu_h:
        for model, coord in campaign.CONFIGS[case]:
            try:
                r = summary(case, model, coord, nu)
            except FileNotFoundError:
                r = {}
            rows.append(f"| {nu:g} | {NAMES[(model, coord)]} | " + " | ".join(_fmt(r.get(k)) for k in cols) + " |")
    cj = cases.RUNS / case / "cost.json"
    if cj.exists():
        rows += ["", "| modelo | Δt máx [s] | s por hora simulada (Δt máx) | s por hora simulada (Δt de resultados) | procesos |",
                 "|---|---|---|---|---|"]
        for key, v in json.loads(cj.read_text()).items():
            m, co = key.split("_", 1)
            v = v or {}
            rows.append(f"| {NAMES.get((m, co), key)} | {_fmt(v.get('dt_max'))} | {_fmt(v.get('s_per_hour'))} | "
                        f"{_fmt(v.get('s_per_hour_results_dt'))} | {_fmt(v.get('threads'))} |")
    return rows


if __name__ == "__main__":
    for c in (sys.argv[1:] or list(campaign.CONFIGS)):
        print("\n".join(table(c)), "\n")
```

- [ ] **Step 4: Run the test, then the full suite**

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k compare` → Expected: 1 passed.
Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider > /tmp/claude-suite.txt 2>&1; tail -3 /tmp/claude-suite.txt` (use the scratchpad path instead of /tmp in practice) → Expected: all passed (1254 + the new tests), 2 xfailed.

- [ ] **Step 5: Tables, reading and README**

Run: `python validation/ilicak_benchmark/compare.py > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/tables.md`

Write `validation/ilicak_benchmark/README.md` (Spanish, as the other benchmark READMEs) with:
1. What it is, the spec link, how to reproduce (`campaign.py results|cost`, `compare.py`), where outputs live.
2. The cases and the per-model configuration (A1–A2), with every class-(c) setup difference: CROCO RSUP3 hyperdiffusion and RESET_RHO0; MOM6 PPM:H3; MITgcm flux-form momentum and partial cells; pyOCEAN seamount with `rho_ref='auto'`; one process for MOM6/CROCO/MITgcm vs 5 threads for pyOCEAN (plus the 1-thread number).
3. The tables of `compare.py` per case.
4. Reading against the criteria of spec §A5 (Benjamin ±10 %; dRPE/dt vs Re_Δ per model — Re_Δ = U Δx/ν_h with U = u_f for lock/overflow (0.496, 1.57 m s⁻¹) and the rms velocity for the IW; T range; IW amplitude; seamount vs ~1 cm s⁻¹; cost > 2× flags), the check of the two prior hypotheses, and the (a)–(d) classification of every difference.
5. "Qué debe mejorar pyOCEAN" (only (a) and (b) items, each with the evidence).
6. References (spec list, with Petersen 2015 and Chen 2024 [verificar] in `docs/literature/reading_list.md`).

Add rows to the scorecard (§7 of `docs/design/vision_roadmap.md`): spurious mixing vs Re_Δ (lock, overflow, IW), seamount four models, cost per model. Update `CLAUDE.md` (a Phase 2 table row for `validation/ilicak_benchmark/` with the headline numbers, the total test count).

- [ ] **Step 6: Commit**

```bash
git add validation/ilicak_benchmark/compare.py validation/ilicak_benchmark/README.md tests/test_ilicak_benchmark.py docs/design/vision_roadmap.md CLAUDE.md docs/literature/reading_list.md
git commit -m "Ilicak benchmark: comparison tables, README and scorecard

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

## Self-review notes (writer)

- Spec coverage: A1 cases → Task 1; A2 configurations → Tasks 3–6, 8 (`CONFIGS`); A3 HYCOM → Task 9; A4 metrics → Task 2 + Task 10 windows; A5 reading/classification → Task 10 README; Piezas → file structure; Pruebas → Tasks 1–6 tests; Entregables → Task 10; budget → Task 8 step 6 watch rule. Part B is excluded by design (own plan).
- Assumptions that only a real run can confirm (MOM6 `MOM_IC.nc` names, CROCO face counts and end-of-log marker, MITgcm grid-file names, the empty-Δt shell argument) are checked by the Task 7 smoke runs, with the rule to fix reader and fake file together and ledger it.
