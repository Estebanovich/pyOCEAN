# CLAUDE.md — pyOCEAN Project

This file provides context and guidelines for Claude Code when working on this project.

---

## Project Overview

pyOCEAN is a regional ocean model built from scratch in Python by Esteban Cruz Isidro, a physical oceanographer and M.Sc. student at UNAM. The primary goal is **educational and scientific**: to understand every component of a HYCOM-like model by implementing it from first principles, and to improve specific physics (especially internal wave dynamics).

This is NOT a wrapper around HYCOM or MITgcm. Every module is written from scratch.

---

## About the Developer

- **Background**: Physical oceanographer with deep hands-on experience in MITgcm and HYCOM
- **M.Sc. research**: Internal wave and seiche dynamics (MITgcm idealised experiments)
- **B.Sc. thesis**: Submarine canyon dynamics during upwelling (MITgcm rotating-tank)
- **Operational experience**: HYCOM-IOA forecast system for Gulf of Mexico (Ometeotl supercomputer)
- **Programming**: Python (strong — NumPy, xarray, Matplotlib, netCDF4, Jupyter), Fortran, Bash
- **No Julia experience** — keep all code in Python

---

## Coding Guidelines

### Language & Style
- **Python only** — no Julia, no Fortran wrappers unless absolutely necessary for performance
- Follow **PEP 8** style conventions
- Use **type hints** on all function signatures
- Write **docstrings** for every function explaining the physics, not just the code
- Keep functions short and single-purpose

### Numerics
- Prefer **NumPy** for array operations
- Use **Numba (@njit)** for performance-critical loops (timestepping, mixing)
- Avoid premature optimization — correctness first, performance second
- Every numerical scheme must reference the original paper in a comment

### Physics Documentation
- Every equation implemented must include:
  1. The equation in LaTeX-style comment
  2. Variable names matching the referenced paper
  3. Units in variable docstrings
- Example:
  ```python
  # Hydrostatic balance: ∂p/∂z = -ρg  (Vallis 2006, eq. 2.75)
  dp_dz = -rho * g
  ```

### Data Structures
- Model state variables are **xarray DataArrays** with proper coordinates and attributes
- All output is written as **NetCDF4** via xarray
- Grid information is stored in an **xarray Dataset**
- Never use raw indices without named dimensions (use xarray `.sel()` / `.isel()`)

### Testing
- Every physics module must have a corresponding test in `tests/`
- Validation tests compare against known analytical solutions or MITgcm output
- Run tests with `pytest tests/`

---

## Physics Reference

The model solves the **hydrostatic primitive equations** under the Boussinesq approximation.

### Key variables
| Variable | Symbol | Units | Description |
|----------|--------|-------|-------------|
| u, v | u, v | m/s | Horizontal velocity components |
| w | w | m/s | Vertical velocity |
| T | θ | °C | Potential temperature |
| S | S | PSU | Salinity |
| p | p | Pa | Pressure |
| ρ | ρ | kg/m³ | In-situ density |
| η | η | m | Sea surface height |
| f | f | 1/s | Coriolis parameter |

### Constants
```python
g   = 9.81       # m/s²  gravitational acceleration
ρ₀  = 1025.0     # kg/m³ reference density (Boussinesq)
Ω   = 7.2921e-5  # rad/s Earth's rotation rate
R   = 6.371e6    # m     Earth's radius
```

---

## Vertical Coordinate Strategy

Phase 1: **z-levels** (simplest, validate physics)
Phase 3: **Hybrid coordinate** — transition zones:
- z-levels near surface (mixed layer)
- Isopycnal in interior (HYCOM-style)
- Sigma near bottom (topography-following)

Coordinate transform code lives in `core/grid/vertical.py`.

---

## Validation Strategy

All physics modules are validated in this order:
1. **Analytical solution** (if available) — e.g., linear wave dispersion
2. **MITgcm comparison** — Esteban's existing simulation output (BayIW_Rectan configs)
3. **HYCOM comparison** — operational Gulf of Mexico output

Validation notebooks live in `notebooks/validation/`.

---

## Development Phases

When working on this project, follow the phase order strictly. Do not add Phase 3 features while Phase 1 is incomplete.

| Phase | Key modules | Validation target |
|-------|-------------|-------------------|
| 1 | grid, dynamics, timestepping, basic mixing | MITgcm rectangular bay seiches |
| 2 | OBCS (Orlanski+sponge), real bathymetry, ERA5 forcing | Gulf of Mexico subdomain |
| 3 | Hybrid vertical coordinate | HYCOM comparison |
| 4 | Internal wave parameterization | M.Sc. thesis results |
| 5 | Operational pipeline, restart files | HYCOM-IOA forecast |

---

## Build Status — Phase 1

Last updated: 2026-03-24.  All tests run with `pytest tests/`.

| Module | File | Tests | Status |
|--------|------|-------|--------|
| Horizontal grid (C-Arakawa) | `core/grid/horizontal.py` | `tests/test_grid_horizontal.py` | ✅ 28/28 |
| Vertical grid (z-levels + hFac) | `core/grid/vertical.py` | `tests/test_grid_vertical.py` | ✅ 37/37 |
| Model state container | `core/state.py` | `tests/test_state.py` | ✅ 52/52 |
| Equation of state | `physics/eos/__init__.py` | `tests/test_eos.py` | ✅ 36/36 |
| Hydrostatic pressure + PGF | `core/dynamics/pressure.py` | `tests/test_pressure.py` | ✅ 34/34 |
| Momentum equations | `core/dynamics/momentum.py` | `tests/test_momentum.py` | ✅ 44/44 |
| Adams–Bashforth 2 timestepper | `core/timestepping/ab2.py` | `tests/test_timestepping.py` | ✅ 19/19 |
| Advection (upwind → PPM) | `numerics/advection/__init__.py` | `tests/test_advection.py` | ✅ 37/37 |
| KPP mixing | `physics/mixing/kpp.py` | `tests/test_kpp.py` | ✅ 32/32 |
| Open BCs + surface forcing | `boundaries/obcs/__init__.py` + `forcing/surface.py` | `tests/test_surface_forcing.py` | ✅ 39/39 |
| NetCDF4 I/O | `diagnostics/io.py` | `tests/test_io.py` | ✅ 22/22 |

**Phase 1 COMPLETE — all 380 tests passing.**

---

## Build Status — Phase 2

Last updated: 2026-10-10.  All tests run with `pytest tests/`.

| Module | File | Tests | Status |
|--------|------|-------|--------|
| Bathymetry load/regrid/smooth | `core/grid/bathymetry.py` | `tests/test_bathymetry.py` | ✅ 38/38 |
| Canonical forcing contract | `forcing/canonical.py` | `tests/test_forcing_canonical.py` | ✅ 18/18 |
| Forcing interpolation (grid/time/land-fill) | `forcing/interp.py` | `tests/test_forcing_interp.py` | ✅ 24/24 |
| Forcing adapter base + registry | `forcing/sources/base.py` | `tests/test_forcing_sources_base.py` | ✅ 13/13 |
| Bulk fluxes NCAR/CORE (LY04/LY09) + radiation | `forcing/bulk.py` | `tests/test_bulk.py` | ✅ 48/48 |
| ERA5 adapter (CDS new + legacy layouts) | `forcing/sources/era5.py` | `tests/test_era5.py` | ✅ 17/17 |
| Bulk fluxes Kara 2005 = HYCOM flxflg=6 (matches compiled HYCOM to 1e-10) | `forcing/bulk.py` | `tests/test_bulk_kara.py` | ✅ 17/17 |
| WRF wrfout adapter (Mercator/lat-lon) | `forcing/sources/wrf.py` | `tests/test_wrf.py` | ✅ 13/13 |
| HYCOM z-level NetCDF adapter (ESPC-D-V02, GOFS 3.1) + `to_model_levels` | `forcing/sources/hycom_z.py`, `forcing/interp.py` | `tests/test_hycom_z.py` | ✅ 16/16 |
| Native HYCOM .a/.b reader (archive, grid, depth, forcing, relax dialects) | `forcing/sources/hycom_ab.py` | `tests/test_hycom_ab.py` | ✅ 20/20 |
| HYCOM native archive (archv/archm, hybrid layers → z) | `forcing/sources/hycom_archive.py` | `tests/test_hycom_archive.py` | ✅ 16/16 |
| Initial state + OBData from canonical ocean | `forcing/ocean_init.py` | `tests/test_ocean_init.py` | ✅ 13/13 |
| Implicit free surface (θ-method, SPD solver) + `ab2_step_implicit_fs` | `core/dynamics/free_surface.py`, `core/timestepping/ab2.py` | `tests/test_free_surface.py` | ✅ 18/18 |
| Stretched Cartesian grid (MITgcm delX/delYFile) | `core/grid/horizontal.py` | `tests/test_grid_horizontal.py` | ✅ +8 |
| Tide prediction, OTPS/TPXO conventions (matches pyTMD to round-off) | `forcing/tides.py` | `tests/test_tides.py` | ✅ 12/12 |
| Flather radiation OBC (implicit, SPD) + `open_boundary_hfac` | `core/dynamics/free_surface.py` | `tests/test_free_surface.py` | ✅ +8 |
| TPXO atlas reader + `boundary_tides` (transport / model depth) | `forcing/sources/tpxo.py`, `forcing/tides.py` | `tests/test_tpxo.py` | ✅ 11/11 |
| Tidal potential (body tide) + SAL (per-constituent and scalar) | `forcing/tides.py`, `core/dynamics/free_surface.py` | `tests/test_tidal_potential.py` | ✅ 13/13 |
| Tide-gauge comparison + defect diagnosis (TICON-3) | `diagnostics/tide_gauges.py` | `tests/test_tide_gauges.py` | ✅ 19/19 |

**Phase 2 in progress — 722 total tests passing.**

### Phase 2 build order

Design docs: `docs/design/forcing_architecture.md` (multi-source forcing), `docs/design/era5_bulk_comparison.md` (MITgcm/CROCO/HYCOM/MOM6 bulk formulae + HYCOM-IOA flags), `docs/design/validation_suite.md` (cross-model benchmark matrix). Cross-model validation configs live in `validation/`.

1. ~~`core/grid/bathymetry.py`~~ ✅ — load_gebco, regrid_to_model, smooth_bathymetry (Hann 9-pt + Shapiro 5-pt, n_skip), fill_isolated_cells, apply_min_depth, build_bathymetry
2. ~~Forcing infrastructure~~ ✅ — `forcing/canonical.py`, `forcing/interp.py`, `forcing/sources/base.py`
3. ~~`forcing/bulk.py` (`bulk_ncar`) + `forcing/sources/era5.py`~~ ✅ — pending: point comparison against MITgcm `exf` output
4. ~~`bulk_kara` (HYCOM flxflg=6) + WRF adapter~~ ✅ — reference table regenerated by `validation/hycom_kara/make_reference.sh`; Lambert/curvilinear WRF still unsupported
5. ~~ESPC/GOFS z-level adapter + `.a/.b` reader~~ ✅ — ESPC `water_temp` is in-situ (CF metadata) → converted to θ with TEOS-10 by default. 5b ✅ native archive → z (piecewise-constant layer sampling, archv baroclinic + u_btrop) and `initial_state`/`boundary_data`. Archive reader validated only on synthetic files (no archv on disk locally).
6. ~~Implicit free surface~~ ✅ — new `ImplicitFreeSurface` (MITgcm θ-method; θ=1 default, θ=½ neutral); matches the exact discrete seiche mode to 1e-9 and Merian's period; 0.14 s/step on 560×352. The old `surface_pressure_cg` (Dirichlet η=0 on walls) is deprecated.
6b. ~~Barotropic BayIW seiche vs MITgcm~~ ✅ — `validation/bayiw_seiche/` (`run_pyocean.py bay|nobay`, `compare.py`). Δη at station Seiche, 0–27 h: period 3.83 h (MITgcm 3.81–3.83), range 16.0 mm (15.7–16.2), corr 0.990–0.998, RMSE 0.18–0.48 mm. pyOCEAN decays slower (e-fold 17.5 h vs 11.4–14.5 h): expected, no baroclinic conversion / viscosity / Orlanski. Uses MITgcm's effective 599.9 m ocean (Σ delR, intentional).
7. ~~Tides + Flather OBC~~ ✅ — `forcing/tides.py` (8 HYCOM constituents, OTPS f/u/phase; reference `tests/data/tide_otps_reference.npz` from `validation/tides/make_reference.py`), Flather in `ImplicitFreeSurface` (outgoing wave < 0.1 % energy left; tidal bay = analytic standing wave within 2 %), TPXO10-atlas reader validated on the REAL atlas (`~/pyOCEAN_refs/TPXO10_atlas_v2_nc`): matches pyTMD to 9e-8 m / 7e-5° incl. coastal points (`validation/tides/check_tpxo_real.py`); real files use lowercase uRe…vIm and land = 0 (masked, land-aware bilinear). 7b ✅ tidal potential + SAL as in HYCOM (mod_tides/mod_barotp): `equilibrium_constants`, `body_tide`, `body_force` (α = 1+k−h, Foreman et al. 1993), per-constituent SAL from `tidal.salReIm` (`read_sal_hycom`, synthetic test only — real file on LUSTRE), scalar SAL `ImplicitFreeSurface(sal_beta=β)` → g(1−β)∇η. A small equatorial basin follows α η_eq − ⟨α η_eq⟩. Tide gauges: `validation/tide_gauges/` — TPXO10 vs TICON-3 with instrumental-defect diagnosis (Mexican Pacific ok records median RSS 1.5 cm; Gulf 1.3 cm; confirmed defects at Progreso 1980–84 (6 h) and Puerto Morelos 2007–14 (×0.1); TPXO10 −30 % M2 at Acapulco; spurious metre-scale TPXO amplitudes at ~2 m nodes). Pending: QC of recent raw data (IOC/SMN) + own harmonic analysis.
8. Validation notebook — Gulf of Mexico / PAC_MEX subdomain run  ← **next**

### Key design decisions made

- **EOS**: two options available — `eos="teos10"` (nonlinear, via `gsw`) and `eos="linear"` (Vallis 2006 eq. 1.156, configurable α/β/T₀/S₀). Linear EOS is default for Phase 1 idealised runs.
- **Pressure**: barotropic term `ρ₀gη` + baroclinic cumulative integral from surface (Marshall et al. 1997, Appendix A). Explicit free surface for Phase 1; `surface_pressure_cg` (Helmholtz 2D CG solver) ready for Phase 2 implicit free surface.
- **Grid spacing**: `dx_u(j_t, i_u)` and `dy_v(j_v, i_t)` from the horizontal grid are used for pressure gradient finite differences (not scalar `dx`/`dy`), so the code is already spherical-grid compatible.
- **ModelState arithmetic**: `__add__`, `__mul__`, `__rmul__` implemented so timestepper writes `state_new = state_old + dt * tendency` directly.

### Known issues / gotchas

- `scipy.sparse.linalg.cg` ≥ 1.12 uses `rtol=` instead of `tol=` — already handled in `pressure.py`.
- `bulk_ncar` keeps `n_iter=2` on purpose (LY04/CESM/MITgcm protocol); a convergence tolerance was evaluated and rejected — see `docs/design/era5_bulk_comparison.md` §7. Weak-wind/stable regime → use COARE 3.0 (to implement).
- netCDF4 in the base conda env emits "numpy.ndarray size changed" RuntimeWarning (binary mismatch); harmless, filter it when running with `-W error`.
- Performance (M3 Pro, 2026-10-10): extra BLAS/OMP threads give no speed-up (SuperLU + NumPy are single-threaded); throughput scales only with independent concurrent runs (≤5, one per P-core). Never use the 6 E-cores.
- `physics/eos/` is a package directory (scaffold); EOS code lives in `physics/eos/__init__.py`, not a separate `physics/eos.py`.

---

## Performance Targets

- Phase 1-2 (pure NumPy): Runs on MacBook Pro, regional domain in minutes
- Phase 3+ (Numba JIT): 10x speedup over pure NumPy
- Phase 5 (MLX/JAX GPU): 50x speedup, comparable to small MPI cluster

---

## File Naming Conventions

- Modules: `snake_case.py`
- Classes: `PascalCase`
- Functions: `snake_case`
- Constants: `UPPER_CASE`
- Test files: `test_<module_name>.py`
- Notebooks: `<phase>_<description>.ipynb` (e.g., `01_shallow_water_validation.ipynb`)

---

## Key References to Keep Nearby

- Bleck (2002) — HYCOM hybrid coordinate formulation
- Marshall et al. (1997) — MITgcm pressure gradient and OBCS
- Large et al. (1994) — KPP mixing scheme
- Durran (2010) — *Numerical Methods for Fluid Dynamics* (time schemes, advection)
- Vallis (2006) — *Atmospheric and Oceanic Fluid Dynamics* (governing equations)
- TEOS-10 (2010) — equation of state
