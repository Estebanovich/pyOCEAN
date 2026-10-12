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
- Inside the time step, model state is plain **NumPy arrays** (e.g. `SigmaState`), as in MITgcm, ROMS, MOM6, Veros, Oceananigans and FESOM2-JAX. This is required for speed and for Numba/JAX (decision of 2026-10-11).
- **xarray is used only at the I/O boundary:** grids (`xarray.Dataset`), forcing and initial/boundary data, output (CF-1.8 NetCDF4/Zarr) and restarts.
- Array index order is documented in each docstring: (k, j, i), with k = 0 at the bed in the σ model; u-faces have nx+1 columns and v-faces ny+1 rows.
- Every model variable is described once in the **single variable registry** (name, dims, C-grid location, units, CF standard name). Output, restart and blow-up diagnostics take their metadata from it (planned in roadmap step 3).

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

## Project Goal

pyOCEAN aims to be the best community regional/coastal ocean model, not a tool tailored to one user. "Best" is measured on accuracy against truth, cost, robustness and usability/transparency. Strategy, gaps and roadmap: `docs/design/vision_roadmap.md` (approved 2026-10-11).

---

## Validation Strategy

All physics modules are validated in this order:
1. **Analytical solution** (if available) — e.g., linear wave dispersion
2. **Laboratory / observations** — tide gauges, ADCP, CTD, lab experiments (the real "truth")
3. **Cross-model comparison** — MITgcm (BayIW_Rectan), CROCO/ROMS, MOM6, HYCOM-IOA

Rules for every benchmark:
- Classify each difference as (a) our bug, (b) missing physics, (c) numerical choice, (d) reference-model error. Only (a) and (b) change the model; never tune to a single case.
- End each benchmark with a "what pyOCEAN should improve" list and update the scorecard (§7 of `vision_roadmap.md`).
- Run the reference case in its full configuration too (e.g. CROCO TIDAL_FLAT in 3D), not only a reduced one.

### Systematic literature checks (local library first, then Consensus)

Before implementing a scheme or parameterisation, and when a benchmark shows an unexplained difference, check the literature to (1) validate the approach against the state of the art and (2) find existing solutions. Consensus searches are a limited budget, so:

1. **Local library first:** `python scripts/litsearch.py "term 1" "term 2"` searches the `.md` text copies in `/Users/estebancruz/Documents/TESIS_InternalWaves-2/Literatura` (recursive; `--convert` makes `.md` for newly downloaded PDFs with pdftotext).
2. **Consensus only if the library has nothing useful** — 1–2 focused searches per decision, not exploratory batches.
3. **Every paper used from Consensus goes into `docs/literature/reading_list.md`** (DOI link, suggested file name `Author_Year - Title.pdf`, ★ if used in code, status ⬜/✅) so Esteban can download it for future local searches.
4. Cite key papers (link/DOI) in the module docstring or benchmark README; references from memory are marked **[verificar]** until checked.

### Reference model source code (check alongside the literature)

Papers rarely give every detail (limiters, boundary treatment, thin-layer guards, step order, defaults). Before implementing a scheme that a production model already has, and when a benchmark differs from that model, read its source too and cite it as `file:line` (model + version) in the docstring or README:

| Model | Local source |
|-------|--------------|
| MOM6 (ALE, remapping, PGF, barotropic) | `~/pyOCEAN_refs/MOM6/src/`; build deps `~/pyOCEAN_refs/FMS` (2025.02.01); idealised inputs `~/pyOCEAN_refs/MOM6-examples/ocean_only/` |
| CROCO / ROMS (σ, GLS, OBC, W/D) | `~/pyOCEAN_refs/croco/` (compiled test cases in `~/pyOCEAN_refs/croco_runs/`) |
| MITgcm (z / partial cells, OBCS, EXF) | `~/pyOCEAN_refs/MITgcm/` (also `/Volumes/ESTEBAN_MIT/MITgcm`, with the BayIW runs) |
| HYCOM (hybgen, barotropic, forcing) | `~/pyOCEAN/HYCOM-src/` (git-ignored) |

Large reads go to a subagent that returns the algorithm with `file:line` references. Report the source as a reference implementation, not as truth: when it disagrees with the paper, say so and decide explicitly.

---

## Development Phases (roadmap stages)

Follow the stage order of `docs/design/vision_roadmap.md` §8; a stage closes when its scorecard rows are ✅ or the difference is explained. Phase 1 (z-level core) is complete; Phase 2 (forcing, tides, OBC, W/D) is largely done.

| Stage | Content | Validation target |
|-------|---------|-------------------|
| A — Quality foundations ✅ (2026-10-11; open: tide runners restart/output wiring, lock-exchange RPE 1.3–1.5× CROCO) | spurious-mixing diagnostics (RPE, variance decay), bit-for-bit restarts, AB3/LF-AM3, LU reuse / PyAMG, CF/Zarr output, domain checker + blow-up post-mortem, log-law drag in regional runs | lock exchange, internal seiche (Ilicak 2012); restart CI |
| B — 3-D physics | vertical advection + stratification in σ-3D, finite-volume PGF, vector-invariant momentum, Casulli subgrid W/D + Stelling–Duinmeijer fronts, low-frequency-only sponge | seamount, Thacker (xfail → pass), BayIW baroclinic seiche vs MITgcm |
| C — ALE hybrid (old Phase 3) | general ALE regrid–remap (PPM/PQM) with z*, σ, z̃, HYCOM-hybrid and adaptive targets; split-explicit barotropic mode | DOME overflow, seamount, HYCOM comparison |
| D — Differentiable | JAX backend (functional kernels), custom VJPs for the free-surface solve and Thomas, smoothed W/D gradients, checkpointing | gradients vs finite differences; inversion vs TICON gauges |
| E — Internal waves (old Phase 4) | baroclinic tides in PAC_MEX, KPP without background diffusivity, optional nonhydrostatic | M.Sc. thesis results, IW spectra |
| F — Operational + ML (old Phase 5) | forecast cycle, regional emulator / learned boundaries (Anemoi, ACE), NVIDIA GPU via JAX | HYCOM-IOA forecast vs independent observations |

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
| JSL internal-tide drag (tidal.rh generation + HYCOM tiddrg=1 drag) | `physics/tides/drag.py` | `tests/test_tidal_drag.py` | ✅ 8/8 |
| Raw sea-level QC + harmonic analysis (spikes, stuck, datum steps, clock, twin sensors) | `diagnostics/sealevel_qc.py` | `tests/test_sealevel_qc.py` | ✅ 37/37 |
| Nonlinear shallow-water terms (−∇·(ηu), advection centred/upwind3, Laplacian viscosity) | `core/dynamics/nonlinear.py` | `tests/test_nonlinear.py` | ✅ 18/18 |
| Wetting and drying (Casulli semi-implicit, total depth, Flather; CG with lagged-LU preconditioner) — Thacker exact + CROCO THACKER/TIDAL_FLAT | `core/dynamics/wetting_drying.py` | `tests/test_wetting_drying.py` | ✅ 24/24 (+1 strict xfail) |
| Log-law depth-dependent bottom drag C_d(D, z₀) + first-layer law of the wall (Soulsby 1997; CROCO) | `physics/bottom_friction.py` | `tests/test_bottom_friction.py` | ✅ 4/4 |
| GLS k-ε turbulence closure, Canuto A (column port of CROCO gls_mixing.F) + parabolic ν | `physics/mixing/gls.py`, `physics/mixing/parabolic.py` | `tests/test_gls.py` | ✅ 3/3 |
| Homogeneous 3-D σ-layer wetting–drying model (Casulli & Cheng 1992), surface stress as top flux — CROCO 3D TIDAL_FLAT to 1.2 cm | `core/dynamics/sigma_wd.py` | `tests/test_sigma_wd.py` | ✅ 11/11 |
| Spurious-mixing diagnostics: RPE (Winters 1995/Ilicak 2012), online variance decay (Burchard & Rennau 2008) | `diagnostics/mixing.py` | `tests/test_mixing_diagnostics.py` | ✅ 8/8 |
| Stratified σ-layer model: flux-form tracers + momentum (WENO5-Z/JS), ω, energy-conserving Coriolis, density-Jacobian PGF + reference profile (seamount 0.2–0.4 cm/s), AB3, SSP-RK3 or predictor–corrector, wind stress (Ekman 0.05 %), open boundaries (Flather + Orlanski + MITgcm-profile sponge; mode-1 IW interior error 0.015 vs 0.82 closed), stretched σ-layers, explicit tracer diffusion (K_h along σ on T − T_ref, implicit K_v), Numba WENO5 (4.6× faster on BayIW 560×352×50: 6.5 s/step) — lock exchange = CROCO | `core/dynamics/sigma_baroclinic.py` | `tests/test_sigma_baroclinic.py` | ✅ 37/37 |
| Bit-for-bit restart (float64 CF NetCDF; AB history, transports, GLS state, open-boundary T_ext, reference ρ profile; config check incl. σ-levels and boundaries) | `core/restart.py` | `tests/test_restart.py` | ✅ 7/7 |
| Domain checker (rx0/rx1, isolated water bodies, pits, isolated cells, CFL advice) | `diagnostics/domain_check.py` | `tests/test_domain_check.py` | ✅ 7/7 |
| Blow-up post-mortem guard (NaN/limit location, lat/lon, depth, CFL, history, state dump) | `diagnostics/postmortem.py` | `tests/test_postmortem.py` | ✅ 4/4 |
| CF-1.8/SGRID output, Zarr (appendable) or NetCDF4, σ formula_terms (stretched s_w) | `diagnostics/output.py` | `tests/test_output.py` | ✅ 5/5 |
| Stretched σ-levels: Song & Haidvogel (1994) stretching, fractions from z-level thicknesses (MITgcm delR) | `core/grid/sigma_levels.py` | `tests/test_sigma_levels.py` | ✅ 6/6 |
| Local literature search (first step before Consensus) | `scripts/litsearch.py` | `tests/test_litsearch.py` | ✅ 3/3 |
| ALE column remapping PCM/PLM/PPM_H4 + threaded 3-D driver (= MOM6 remapping_core_h to ≤ 5e-12 on 120 columns incl. vanished/1e-10/1 mm layers in the end stencils, MOM6 closed-form H4 end values; Hallberg trick; order reduction counts active cells, unlike MOM6; dry columns keep their values) | `core/vertical/remap.py` | `tests/test_vertical_remap.py`, `tests/test_vertical_vs_mom6.py` | ✅ 23/23 + MOM6 |
| ALE grid rules z*, σ, hybrid HYCOM1 (z* floor, max thickness, shallow-σ blend) + 3-D driver with input checks (= MOM6 coord_zlike/coord_sigma/regrid_interp P1M_H2 to ≤ 3e-14 m; σ also clamped from the top, unlike MOM6; full build_hycom1_column not yet compared; vs HYCOM hybgen in `validation/mom6_column/`) | `core/vertical/regrid.py` | `tests/test_vertical_regrid.py`, `tests/test_vertical_vs_mom6.py` | ✅ 29/29 + MOM6 |

**Phase 2 in progress — 980 total tests passing (+1 strict xfail).**

### Phase 2 build order

Design docs: `docs/design/forcing_architecture.md` (multi-source forcing), `docs/design/era5_bulk_comparison.md` (MITgcm/CROCO/HYCOM/MOM6 bulk formulae + HYCOM-IOA flags), `docs/design/validation_suite.md` (cross-model benchmark matrix). Cross-model validation configs live in `validation/`.

1. ~~`core/grid/bathymetry.py`~~ ✅ — load_gebco, regrid_to_model, smooth_bathymetry (Hann 9-pt + Shapiro 5-pt, n_skip), fill_isolated_cells, apply_min_depth, build_bathymetry
2. ~~Forcing infrastructure~~ ✅ — `forcing/canonical.py`, `forcing/interp.py`, `forcing/sources/base.py`
3. ~~`forcing/bulk.py` (`bulk_ncar`) + `forcing/sources/era5.py`~~ ✅ — pending: point comparison against MITgcm `exf` output
4. ~~`bulk_kara` (HYCOM flxflg=6) + WRF adapter~~ ✅ — reference table regenerated by `validation/hycom_kara/make_reference.sh`; Lambert/curvilinear WRF still unsupported
5. ~~ESPC/GOFS z-level adapter + `.a/.b` reader~~ ✅ — ESPC `water_temp` is in-situ (CF metadata) → converted to θ with TEOS-10 by default. 5b ✅ native archive → z (piecewise-constant layer sampling, archv baroclinic + u_btrop) and `initial_state`/`boundary_data`. Archive reader validated only on synthetic files (no archv on disk locally).
6. ~~Implicit free surface~~ ✅ — new `ImplicitFreeSurface` (MITgcm θ-method; θ=1 default, θ=½ neutral); matches the exact discrete seiche mode to 1e-9 and Merian's period; 0.14 s/step on 560×352. The old `surface_pressure_cg` (Dirichlet η=0 on walls) is deprecated.
6b. ~~Barotropic BayIW seiche vs MITgcm~~ ✅ — `validation/bayiw_seiche/` (`run_pyocean.py bay|nobay`, `compare.py`). Δη at station Seiche, 0–27 h: period 3.83 h (MITgcm 3.81–3.83), range 16.0 mm (15.7–16.2), corr 0.990–0.998, RMSE 0.18–0.48 mm. pyOCEAN decays slower (e-fold 17.5 h vs 11.4–14.5 h): expected, no baroclinic conversion / viscosity / Orlanski. Uses MITgcm's effective 599.9 m ocean (Σ delR, intentional).
7. ~~Tides + Flather OBC~~ ✅ — `forcing/tides.py` (8 HYCOM constituents, OTPS f/u/phase; reference `tests/data/tide_otps_reference.npz` from `validation/tides/make_reference.py`), Flather in `ImplicitFreeSurface` (outgoing wave < 0.1 % energy left; tidal bay = analytic standing wave within 2 %), TPXO10-atlas reader validated on the REAL atlas (`~/pyOCEAN_refs/TPXO10_atlas_v2_nc`): matches pyTMD to 9e-8 m / 7e-5° incl. coastal points (`validation/tides/check_tpxo_real.py`); real files use lowercase uRe…vIm and land = 0 (masked, land-aware bilinear). 7b ✅ tidal potential + SAL as in HYCOM (mod_tides/mod_barotp): `equilibrium_constants`, `body_tide`, `body_force` (α = 1+k−h, Foreman et al. 1993), per-constituent SAL from `tidal.salReIm` (`read_sal_hycom`, synthetic test only — real file on LUSTRE), scalar SAL `ImplicitFreeSurface(sal_beta=β)` → g(1−β)∇η. A small equatorial basin follows α η_eq − ⟨α η_eq⟩. Tide gauges: `validation/tide_gauges/` — TPXO10 vs TICON-3 with instrumental-defect diagnosis (Mexican Pacific ok records median RSS 1.5 cm; Gulf 1.3 cm; confirmed defects at Progreso 1980–84 (6 h) and Puerto Morelos 2007–14 (×0.1); TPXO10 −30 % M2 at Acapulco; spurious metre-scale TPXO amplitudes at ~2 m nodes). Real SAL file (`~/pyOCEAN_refs/tidal.salReIm_tpxo10.[ab]`, PAC_MEX 1600×1125) read and checked (`validation/tides/check_sal_real.py`): F ≈ −0.10 η semidiurnal / −0.14 η diurnal vs TPXO10 (corr 0.95–0.99) → file holds −η_SAL, consistent with HYCOM's subtraction (blkdat scalar `tidsal`=0.06 underestimates). Raw-data QC of the IOC/SMN network (`fetch_ioc.py`, `qc_ioc_mexico.py`, README second half): dead/mis-scaled sensors (Veracruz2 float, Isla Clarión ×−0.1), La Paz2 float–radar drift since 2026-08-09, damped Celestún / Sánchez Magallanes, duplicated feeds with 1-min offsets and wrong sensor labels. Single-sensor datum-step detection only yields candidates (all 8 testable ones rejected by twins).
8. PAC_MEX barotropic tidal run (in progress) — `validation/pacmex_tides/` (`run_tides.py`, `compare_gauges.py`, `bench_parallel.py`): 650×500 subdomain (118–92°W, 12–32°N) of the HYCOM-IOA grid/bathymetry, 8 constituents, Flather + TPXO10, body tide − real SAL, C_d = 2.5e-3, θ = 0.6, Δt = 120 s; 33 days in 11 min on one P-core (boundary constants interpolated once: `boundary_tide_constants`/`boundary_tides_at`, 8× faster). vs 31 TICON records: median RSS 1.95 cm (no SAL 3.35, TPXO 1.87); Acapulco 0.6 cm vs TPXO 5–6 cm; Gulf of California amphidrome misplaced (Guaymas 22 vs 13.5 cm, −37°) → drag ensembles: uniform C_d 0.0015–0.006 and JSL drag (drgscl 0.25–1.0, + HYCOM-IOA spatial cb) improve the Pacific (1.44 cm vs TPXO 1.79) but leave the Gulf at 11–12.5 cm (TPXO 5.1): the Gulf error is not a missing sink. `validation/tides/make_tidal_rh.py` regenerates tidal.rh from the global JSL (matches HYCOM-tools to 4e-8 median relative). Parallel: SuperLU = 65 % of the step; Numba prange on the rest gives ≤1.35× overall → use cores for concurrent runs.

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
