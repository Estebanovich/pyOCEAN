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
| N — Modular core + UX (added 2026-10-15, `vision_roadmap.md` §9; before D and before C's split-explicit) | component contract + fixed-order driver extracted from `SigmaBaroclinic` (no physics change), S + TEOS-10 + full surface fluxes in 3-D, KPP as second closure, `Simulation` (typed config, presets, pre-flight, post-mortem, CF/Zarr, term-by-term budgets by default), integrated regional domain builder, installable package + stable API. N1 ✅ 2026-10-28, N2a ✅ 2026-10-29, **N3 ✅ 2026-10-30** (closure as a component, GLS k-ε and KPP in σ and ALE, prescribed heat flux, periodic boundaries, log-law drag in ALE, column benchmarks vs CROCO/MOM6; closed with two explained differences, see the N3 rows) | 1388 tests + benchmarks bit for bit after N1; regional heat/salt case vs MITgcm/HYCOM; quickstart and PAC_MEX from a clean install |
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
| Wetting and drying (Casulli semi-implicit, total depth, Flather; CG with lagged-LU preconditioner) — Thacker exact + CROCO THACKER/TIDAL_FLAT | `core/dynamics/wetting_drying.py` | `tests/test_wetting_drying.py` | ✅ 26/26 (+1 strict xfail; + prescribed transport `flather_q`) |
| Log-law depth-dependent bottom drag C_d(D, z₀) + first-layer law of the wall (Soulsby 1997; CROCO) | `physics/bottom_friction.py` | `tests/test_bottom_friction.py` | ✅ 4/4 |
| GLS k-ε turbulence closure, Canuto A (column port of CROCO gls_mixing.F) + parabolic ν | `physics/mixing/gls.py`, `physics/mixing/parabolic.py` | `tests/test_gls.py` | ✅ 3/3 |
| Homogeneous 3-D σ-layer wetting–drying model (Casulli & Cheng 1992), surface stress as top flux, optional face-layer fractions/thicknesses (ALE geometry) and a separate bed-layer thickness for the log-law drag (`hu_drag`/`hv_drag`) — CROCO 3D TIDAL_FLAT to 1.2 cm | `core/dynamics/sigma_wd.py` | `tests/test_sigma_wd.py` | ✅ 19/19 (+ `r_bed` quadratic drag rate) |
| Spurious-mixing diagnostics: RPE (Winters 1995/Ilicak 2012), online variance decay (Burchard & Rennau 2008) | `diagnostics/mixing.py` | `tests/test_mixing_diagnostics.py` | ✅ 8/8 |
| Stratified σ-layer model: flux-form tracers + momentum (WENO5-Z/JS), ω, energy-conserving Coriolis, density-Jacobian PGF + reference profile (seamount 0.2–0.4 cm/s), AB3, SSP-RK3 or predictor–corrector, wind stress (Ekman 0.05 %), open boundaries (Flather + Orlanski + MITgcm-profile sponge; mode-1 IW interior error 0.015 vs 0.82 closed), stretched σ-layers, explicit tracer diffusion (K_h along σ on T − T_ref, implicit K_v), Numba WENO5 (4.6× faster on BayIW 560×352×50: 6.5 s/step) — lock exchange = CROCO | `core/dynamics/sigma_baroclinic.py` | `tests/test_sigma_baroclinic.py` | ✅ 37/37 |
| Bit-for-bit restart (float64 CF NetCDF; AB history, transports, GLS state, open-boundary T_ext, reference ρ profile, lagged-LU CG preconditioner matrix; config check incl. σ-levels and boundaries; ALE (Plan D, 2026-10-13): h, exterior profile T_ext/h_ext, truncation count, regrid rule/parameters and scheme choices checked — BayIW cut hybrid 60 h = 30 h + restart + 30 h bit for bit, validation/bayiw_cut/restart_check.py; limiter count `n_limited` and the thin-layer coupling/cutoff/tracer limiter checked, 2026-10-14) | `core/restart.py` | `tests/test_restart.py` | ✅ 21/21 (+ tracers, sponge, drag, 2026-10-14) |
| Domain checker (rx0/rx1, isolated water bodies, pits, isolated cells, CFL advice) | `diagnostics/domain_check.py` | `tests/test_domain_check.py` | ✅ 7/7 |
| Blow-up post-mortem guard (NaN/limit location, lat/lon, depth, CFL, history, state dump) | `diagnostics/postmortem.py` | `tests/test_postmortem.py` | ✅ 4/4 |
| CF-1.8/SGRID output, Zarr (appendable) or NetCDF4, σ formula_terms (stretched s_w) | `diagnostics/output.py` | `tests/test_output.py` | ✅ 6/6 (+ tracer fields `trN`) |
| Stretched σ-levels: Song & Haidvogel (1994) stretching, fractions from z-level thicknesses (MITgcm delR) | `core/grid/sigma_levels.py` | `tests/test_sigma_levels.py` | ✅ 6/6 |
| Local literature search (first step before Consensus) | `scripts/litsearch.py` | `tests/test_litsearch.py` | ✅ 3/3 |
| ALE column remapping PCM/PLM/PPM_H4 + threaded 3-D driver (= MOM6 remapping_core_h to ≤ 5e-12 on 120 columns incl. vanished/1e-10/1 mm layers in the end stencils, MOM6 closed-form H4 end values; Hallberg trick; order reduction counts active cells, unlike MOM6; dry columns keep their values) | `core/vertical/remap.py` | `tests/test_vertical_remap.py`, `tests/test_vertical_vs_mom6.py` | ✅ 23/23 + MOM6 |
| ALE grid rules z*, σ, hybrid HYCOM1 (z* floor, max thickness, shallow-σ window transition = keeps the isopycnals that fit between σ and z*, continuous in D; OM4-style light top targets keep the mixed layer in z; floodable H ≤ 0 columns with σ or hybrid in its σ zone; massless layers (h < 2 h_min) take the density of the layer with mass above, as HYCOM hybgen.F90:904-925, so a stale light bed layer does not dissolve a gravity current; isopycnal layer preservation `keep_isopycnal` (default): interfaces between layers inside their target brackets stay put — White et al. 2009, hybgen principle; hybrid lock-exchange RPE gain 35× below ALE-σ) + 3-D driver with input checks (= MOM6 coord_zlike/coord_sigma/regrid_interp P1M_H2 to ≤ 3e-14 m; σ also clamped from the top, unlike MOM6; full build_hycom1_column not yet compared; vs HYCOM hybgen in `validation/mom6_column/`; hybgen not ported, decision of 2026-10-11) | `core/vertical/regrid.py` | `tests/test_vertical_regrid.py`, `tests/test_vertical_vs_mom6.py` | ✅ 44/44 + MOM6 |
| Vector-invariant momentum terms for ALE (MOM6 Sadourny 1975 energy PV flux q·(vh, −uh) with BOUND_CORIOLIS, KE Arakawa (default since 2026-10-12: Godunov damps progressive internal waves ~9 % per 5 km at dx = 100 m) / Godunov (density currents) and its gradient, free-slip vorticity, CFL velocity truncation backstop = MOM6 vertvisc_limit_vel) | `core/dynamics/vector_invariant.py` | `tests/test_vector_invariant.py` | ✅ 20/20 |
| ALE face geometry (donor-cell layer transport clipped at the shallower bed for z*/hybrid = HYCOM cnuity, not for σ; layer fluxes corrected to the free-surface flux = HYCOM upstream barotropic correction; mean of the two clipped thicknesses for ū, viscosity and the thin-bed rule (partial z* layer = MITgcm hFacW); min thickness for along-layer diffusion; thin bed layers = HYCOM; Lagrangian continuity; PPM layer thickness flux = MOM6 continuity_PPM with positive-definite edges, swept mean and Newton barotropic adjustment, residual shared like HYCOM; outflow limiter `limit_outflow` (hybrid robustness plan, 2026-10-14): the outgoing fluxes of a layer are cut to its content and the cut flux goes to the other layers of the same face, so h ≥ 0 after the Lagrangian step and Σ_k Q_k stays the free-surface flux (Numba fast check, cost at noise level); `viscous_face_thickness`: MOM6's upwind-biased thickness for the vertical viscosity (harmonic where water leaves the thin side, arithmetic where thick water advances; MOM_vert_friction.F90:1519-1533, Hallberg & Rhines 1996), fused Numba kernel; `blend_thin_layers`: HYCOM's thin-layer blending (mod_momtum.F90:3055-3065), option only — it locks advancing fronts) | `core/vertical/ale.py` | `tests/test_vertical_ale.py` | ✅ 54/54 |
| Finite-volume PGF (Adcroft et al. 2008, port of MOM6 MOM_PressureForce_FV + density integrals, Boole; PLM of the anomaly T − T_ref(z) so a reference profile is exact, unlike MOM6; bottom mass weighting next to vanished layers only; exact for linear ρ on any layers) | `core/vertical/pgf.py` | `tests/test_vertical_pgf.py` | ✅ 5/5 |
| ALE mode of the stratified model (`vertical_coordinate="ale"`: prognostic h, Lagrangian step + regrid + remap each step, σ/z*/hybrid, `pgf="fv"`, face velocities remapped in fraction space, momentum: vector-invariant by default (`momentum`, `ke_scheme='arakawa'` (default since 2026-10-12; 'gudonov' for density currents, used by the lock-exchange tests and the BayIW runs before that date), `max_truncations`; flux form kept as option, with the vanished-cell limiter V_u ≥ 2Δt max(Q_in, Q_out)); layer thickness flux `thickness_flux` 'ppm' (default, MOM6 continuity_PPM; needs ~20 layers in density currents) or 'upwind' (donor cell, 28 % cheaper); hybrid initial grid = σ-layers + `init_iterations` regrid–remap passes (MOM6 THICKNESS_CONFIG uniform + REGRID_ACCELERATE_INIT); `initial_state(eta, T, h_src=...)` remaps T from a source grid (data z-levels) at every pass, as MOM6 Z_INIT_ALE_REMAPPING — use it for hybrid runs from z-level data (the σ start lost the top metres of deep columns: 1.6 cm/s from rest at the BayIW shelf edge, now 3.3e-4); BayIW 2-D section vs MITgcm (validation/bayiw_2d/: same section, walls in x, Orlanski-only reference): seiche ±3 % corr 1.00, linear z*/hybrid-50 internal waves 0.97–1.05 corr ≥ 0.99, two-layer hybrid open (vanished-layer velocity noise ~1 mm/s, shelf-edge V110); BayIW 3-D cut vs MITgcm on the same cut (validation/bayiw_cut/, 200×160×50, 5 MPI, Orlanski only): z* seiche −1.8 %/±1 %, linear 0.98–1.07, two-layer isotherm 1.01–1.21 and heat displacement 1.03–1.10 (T110 1.4–3.5× = interface diffusion); hybrid-25 two-layer failed (interface nearly still near the bay-mouth step) — fixed 2026-10-13: the z* floor dz_top reached only 74.8 m (now bayiw.hybrid_dz_top to the bed; SigmaBaroclinic warns when a hybrid floor stops above the deepest bed), initial_state(h_src) now regrids once on the source levels as MOM6 (regrid_columns n_out) + snap_to_jumps, and keep_isopycnal also keeps an interface whose target lies between its two layers' densities; 2-D two-layer interface rise 0.94–1.29 of MITgcm's, realistic February case (bayiw 'real', density-equivalent T) z* and hybrid = MITgcm in 2-D and 3-D; open: 3-D two-layer phase lag 1.5–3 h at the shelf edge, linear hybrid in the shallow bay; nested in Esteban's full runs ('radiating' toward the full run, MITgcm OBCS prescribe u/v/T — no η with the linear free surface) η at the stations is 0.03–0.11 from the full run (MITgcm nested 0.05–0.17, phase 1 0.27–1.00) and the baroclinic error equals the phase-1 model difference except inside the sponge band (needs a 3-D exterior); hybrid lock exchange vs MOM6 (KV 1e-4, 21 layers as MOM6, water masses at density-bracket centres): 1 m nose and 5 m body 1.05/0.98 U (MOM6 1.05), max|u| 0.74 vs 0.56 m/s, RPE 8.8e5 J (ALE-σ 1.9e8), no truncation (validation/mom6_ke_schemes/: 10 layers + ppm dense body 1.26 U, 10 layers + upwind nose 1.12 U); z* seamount 0.08 cm/s at day 10 (upwind 0.07); vector-invariant momentum costs the same as the flux form; performance (plan 2026-10-12-ale-step-performance: Numba kernels with NumPy oracles, CG + lagged-LU free surface with exact conservation by default in ALE, optional `mix_diagnostic`): BayIW 560×352 z*/50 layers 8.48 → 3.32 s/step (2.72 without the mixing diagnostic), hybrid/25 layers 4.77 → 1.58 (1.44), validation/bayiw_baroclinic/README.md; part 2 (plan 2026-10-13-ale-serial-glue-performance: nested exterior only where used, parallel truncate_cfl/ke_gradient/ρ_ref interpolation/RK3 stages/AB3/transport velocities, strided views instead of copies): BayIW cut 198×159 (timed without the profiler) z* 0.385 → 0.304 s/step, hybrid 0.197 → 0.160, z* nested 0.539 → 0.328–0.338 (9.1 s per simulated hour vs MITgcm 4.4 with 5 MPI); the remaining time is in memory-bound parallel kernels (remap, advection, FV PGF, PPM flux ≈ 3.5× on 5 threads), validation/bayiw_cut/README.md; time-varying open boundaries (`open_boundary_state(t) → {edge: OBData with zw}`, modes 'radiating' = Flather toward (η, ū)ₑₓₜ + Orlanski with CROCO nudging (τ_in 1 d, τ_out 10 d) + sponge toward the exterior, 'clamped' = Dirichlet, no sponge; core/dynamics/open_boundary.py; analytic mode-1 wave ±5 %, MITgcm internal_wave flat part 0.91–1.01 with Arakawa KE — the default Godunov KE damps progressive internal waves ~9 % per 5 km at dx = 100 m, validation/internal_wave/); open boundaries (Flather + Orlanski and sponge on the thickness-weighted u′, open-face fluxes on the interior thickness, exterior T profile kept in z and remapped to the current layers; mode-1 channel closed / Orlanski / sponge: z* 0.79 / 0.16 / 0.016, hybrid 0.77 / 0.198 / 0.024, σ 0.80 / 0.15 / 0.015; no H ≤ 0 cells at an edge), z0=None, constant ν; ALE-σ lock exchange = Benjamin ±10 %, RPE gain in 6 h σ 2.91e8 J vs ALE-σ 2.69e8 J; bay-mouth step 600→164 m, 12 h: max\|u\| σ+Jacobian 7.3 cm/s vs z*+FV 5e-12 m/s; hybrid robustness plan (2026-10-14, spec 2026-10-14-hybrid-robustness): `theta_bc` = ½ in every coordinate (time-centred baroclinic layer transport; possible in the hybrid with the outflow limiter), `thin_layer_coupling` 'mom6' (default in ALE: upwind-biased face thickness in the implicit vertical viscosity, the bed drag keeps the mean thickness) or 'none', `thin_layer_cutoff` (HYCOM blending, default 0 = off), `tracer_limiter` 'fct' (default with SSP-RK3 WENO5: Zalesak 1979 FCT around the three RK3 stages, bounds over x, y and k ± 1 — along-layer bounds clipped the seamount summit 0.22 → 0.71 cm/s; fused Numba kernels, NumPy oracle `_advect_fct_numpy`), `SigmaState.n_limited`; review fix (2026-10-14): the coupled viscous step restores the depth-mean balance with a uniform correction δ on the clipped free-surface face column (wind and bed stress enter ū as without coupling); Ilicak: hybrid overflow completes at ν = 100 and 1000 (stopped at ν ≤ 100), at ν = 1 only in 1 of 5 trajectories (vanished-layer slab at the east wall, pre-existing: 3 of 5 without δ; open), T in the initial range everywhere (excursion ≤ 1.5e-8 °C; was 33 °C hybrid, 4.7 °C z*), hybrid IW 0.50 / 0.63 / 0.27 (ν = 0.01 still damped, open), hybrid lock fronts 0.94–1.06 U; BayIW 2-D/cut/internal_wave unchanged to ±0.02 (cut linear bay effect closer to MITgcm, 0.83–0.99 → 0.97–1.04, Γ 0.92); cost −1 … +5.5 % per step on BayIW); DOME plan (2026-10-14, spec 2026-10-14-dome-overflow): `n_tracers` passive tracers (`SigmaState.tr`, advected with T by WENO5 + FCT, remapped, edge and interior sponges, restart bit for bit, output `trN`; oracle tr = T bit for bit in z*/σ/hybrid, s ∈ [0, 1] to ±1e-12); interior implicit sponge (`sponge_rate`, `sponge_T`/`sponge_zw` profile in z remapped to the current layers, `sponge_tr`; MOM6 DOME profile); quadratic bed drag over the bottom `hbbl` (`bottom_drag='quadratic'`, `cd`, `drag_bg_vel`; the stress r ū_bbl is spread over the hbbl layers with the ū_bbl weights as per-layer implicit Rayleigh rates, δ with Σ r_k û_k — final review: on layer 0 alone a vanished bed layer passed ~7 % of it at ν = 1e-4); the interior sponge stays out of `mix_num`; exact prescribed transport at 'clamped' Flather faces (`OBData.transport`, MOM6 SIMPLE; DOME inflow exact in the volume budget); open: at 10 km / Δt 1200 s (fΔt = 0.12) the time-centred θ_bc = ½ lets a basin-wide upper-ocean mode grow from day ~28 to CFL truncations at day 36 in z*, σ and hybrid (stable at 600 s and with θ_bc = 1; MOM6 runs 1200 s; separate plan) | `core/dynamics/sigma_baroclinic.py` | `tests/test_sigma_ale.py` | ✅ 136/136 (+1 xfail: FV vs Jacobian over the seamount, pending) |
| Four-model benchmark part A (Ilicak 2012 lock exchange / overflow / internal wave at 3 ν_h + Beckmann–Haidvogel seamount; pyOCEAN z\*/σ/hybrid, MOM6 z\*/ρ, CROCO σ, MITgcm z\*; one case definition, common series, RPE with exact hypsometry, new-extrema metric, cost at the largest stable Δt on one process): z\* = MOM6 z\* in the lock exchange (RPE ±1.6 %), ≈ MITgcm in the overflow but WENO5-Z makes new T extrema there (≤ 4.7 °C in 0.25 % of the volume); internal wave: pyOCEAN damped the oscillation at low ν (0.39–0.45 vs 0.55–0.72) because the free-surface θ = 0.6 also weighted the baroclinic layer transport — fixed 2026-10-14 with `theta_bc` = ½ (time-centred u′, depth mean keeps θ): z\*/σ 0.74 / 0.58 / 0.26 vs 0.65–0.72 / 0.55–0.65 / 0.24–0.26, mixing = MITgcm (the earlier "1.7–2.6× less" was mostly the damped wave); hybrid robustness plan (2026-10-14: outflow limiter, upwind-biased thin-layer coupling, FCT with 3-D bounds, θ_bc = ½ also in the hybrid): hybrid 250–1200× less mixing than z\*, T exactly in range in every case (was 3.2–33 °C; z\* WENO5-Z extrema 4.7 °C → 1.5e-8), overflow completes at ν = 100 and 1000 (MOM6 ρ stops at ν ≤ 100; at ν = 1 the hybrid stops in 4 of 5 trajectories, vanished-layer slab, open), hybrid IW 0.50 / 0.63 / 0.27 (ν = 0.01 open, class a), thin overflow faces still up to 1.9× the thick ones at ν = 1000 and 2.0× at ν = 100 (1.55/1.77 before the depth-mean correction δ, uniform in the column; candidate: δ ∝ visc_rem as MOM6); seamount σ+ρ_ref 0.22 cm/s (CROCO 36, MOM6 σ 154; MOM6 z\* truncates, unexplained); cost cheaper than MOM6/MITgcm in 2-D (overflow Δt 80 s vs 10–20), CROCO 1.7–3× cheaper except the overflow; HYCOM 2.3 (cost only, 100 hybgen layers, `hycom/inputs.py`/`blkdat.py`/`run_hycom.sh`; build copy guards two unallocated tide arrays, mod_hycom.F90:1656/1674, class d): overflow Δt_max 20 s (marginal: one neg. dp event; 10 s without any), 5.41 s/h on 1 rank (1.43 on 5; costs re-measured 2026-10-14 after 3 orphaned MOM6 MPI ranks had contaminated the first ladder), at ν = 1000 because ν = 100 stops with neg. dp on the slope (undiagnosed; dp0k covers the full depth, so layers are mostly z) | `validation/ilicak_benchmark/` | `tests/test_ilicak_benchmark.py` | ✅ 50/50 |
| DOME overflow benchmark (Legg et al. 2006 Case 1 in MOM6's geometry, 10 and 5 km, 50 days; pyOCEAN z\*/σ/hybrid vs MITgcm z (Legg and common physics) and MOM6 ρ (layer mode); one case definition, common series, Legg metrics: path Y(x), plume transport, α_E, final buoyancy, centroid, inflow by class, thin faces, cost): Legg anchor MITgcm b_f −10.85 vs −11.7e-3 (7 %), far path 5 %, near-inlet path 25–35 % less downslope (configuration, c); z\* vs MITgcm z: b_f 1.6–3 %, plume transport ≤ 9 % (10 km) / ≤ 18 % (5 km), path 1–3.5 cells further downslope, α_E +29–47 % (c, scheme/drag, open); hybrid vs MOM6 ρ not comparable (MOM6 dye never enters, MOM_open_boundary.F90:5390-5400, d; thickness proxy blind on the slope), hybrid vs Legg iso: b_f −16.2 vs −12 (missing Ri mixing, b); resolution trend as Legg in every coordinate; inflow volume exact; T and s exactly in range, no truncations at Δt 600 s, 4b/4d absent; found: θ_bc = ½ instability at 10 km / 1200 s (a, separate plan), hybrid inlet averages the exterior over a thick interior layer at 10 km (densest class −29 %, a), MITgcm harness fixes (rbcs dye mask relaxMaskFile(3), mnc files split at 2 GB); kv_low (ν_v 1e-4) dropped: MOM6 and MITgcm stop by day 3–4 without Smagorinsky/Jackson; cost 10 km 14 s/day (5 threads, Δt 600, the runner default) vs MITgcm 4.6 (Δt 600) and MOM6 12.0 (Δt 1200), 5 km 61–64 vs 33.7 and 92.8; hybrid + GLS in ALE (N3): b_f −12.99e-3, see the N3 Lorenz/DOME row | `validation/dome/` | `tests/test_dome_benchmark.py` | ✅ 14/14 |
| N1 component core (stage N, spec `docs/superpowers/specs/2026-10-28-n1-component-core-design.md`, reference reads `docs/design/n1_refs/`): `SigmaBaroclinic` is a facade (same constructor) over one component per role (`core/components/`: LinearEOS, PressureGradient, Momentum with named terms, AB3Momentum, FreeSurfaceVerticalViscosity → BarotropicFluxes, QuadraticBedDrag, OpenBoundary (exterior returned, not stored), σ/ALE LayerTransport, TracerAdvection (inflow stack as argument), TracerDiffusion, ALECoordinate, InteriorSponge) run by a fixed ten-phase driver (`core/model/driver.py`; `debug` phase hashes, `keep_terms`); contract + variable registry + shared thread list (`core/model/`); pre-flight reports every argument error at once and warns for arguments a mode ignores; 'godunov' spelling ('gudonov' kept with a warning); GLS state in `SigmaState.closure`; restart variables tagged by component; public `face_thickness`, `edge_face_depths`, `pressure_gradient`. No physics change: 31 golden configurations bit for bit vs main (`tests/test_n1_golden.py`, `scripts/n1_make_golden.py`), benchmark smoke runs vs main (`scripts/n1_smoke.py`). Found and fixed during N1: `ke_scheme`/`kh` set after construction must reach the components. Open (not N1): AB3 history not remapped after the ALE regrid (MOM6 remaps it) | `core/model/`, `core/components/` | `tests/test_n1_*.py`, `tests/test_components_*.py`, `tests/test_model_*.py`, `tests/test_preflight.py` | ✅ |
| N2a salinity and freshwater (stage N, spec `docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md`): active salinity `SigmaState.S` stepped along the path of T in σ and ALE (WENO5/FCT, kh, kv, sponge, remap, restart, pre-flight); linear EOS in (T, S) and TEOS-10 in every consumer (ρ(T, S) in the PGF, `fv_pgf_rho` without 1/α); real freshwater flux fw = P − E as a volume source of the free surface (real freshwater flux as MITgcm `useRealFreshWaterFlux`, MOM6 applyBoundaryFluxesInOut; not a virtual salt flux), freshwater tracer substep at fixed salt (S = 0 into the top layer, evaporation deficit passed to the layers below); rivers = `'clamped'` open-boundary inflow with S = 0; exact discrete variance budget of S and T per operator (`variance_budget=True`: num_adv, surf, phy_h, phy_v, num_remap, sponge; Klingbeil et al. 2014, Burchard & Rennau 2008); in-model TEF on a transect with 250 salinity bins and the Lorenz et al. (2021) exact mixing relation (eq. 32) in `diagnostics/tef.py`; prescribed-elevation `'eta'` open boundary (Dirichlet η, as GETM and CROCO `OBC_COM_ZSPECIFIED`). Budget exact to round-off (\|Me − M_tot\|/M_tot 8e-11 and 4e-10 in Lorenz A). Open: pre-existing hybrid regrid end-cell interface error 0.1–1.2 m (also in the T-only path); AB3 + WENO5 flux-form momentum has no horizontal viscosity safeguard (A_h = 0; σ + GLS Lorenz needed Δt T_M2/746); ALE mode had no turbulence closure (only constant/none) and the constant closure has no convective adjustment (GLS and KPP in ALE since N3, rows below); TEF accumulators are not in the restart | `core/components/freshwater.py`, `variance_budget.py`, `eos.py`, `tracers.py`, `free_surface.py`, `pgf.py`, `open_boundary.py`, `core/model/driver.py`, `core/dynamics/sigma_baroclinic.py`, `sigma_wd.py`, `wetting_drying.py`, `core/restart.py`, `diagnostics/tef.py` | `tests/test_n2a_salinity.py`, `test_n2a_density.py`, `test_n2a_freshwater.py`, `test_n2a_budget.py`, `test_n2a_tef.py`, `test_clamped_eta_obc.py`, `test_restart.py` | ✅ 29 + 15 + 14 + 11 + 12 + 5 (+1 restart) |
| Stratified GLS k-ε (N2a task 8c, switch `gls_stratified`, default True): buoyancy production −κ_t N² in the k equation, N² from ρ(T, S) of the model, α_N and Galperin et al. (1988) length limit with N, GLS tracer diffusivity κ_t (floor 1e-6 m²/s as CROCO `nuws`) used for S and T instead of the constant `kv`; `gls_stratified=False` keeps the neutral (unstratified) closure | `physics/mixing/gls.py`, `core/dynamics/sigma_wd.py` | `tests/test_gls_stratified.py`, `tests/test_gls.py` | ✅ 14/14 (+3 neutral) |
| Lorenz et al. (2021) idealized estuary benchmark (500 × 1 × 60 channel, E, P, river, M2 0.5 m by prescribed elevation; experiments A–D, 600 M2 cycles, mean of the last 100; pyOCEAN σ + stratified GLS (Δt T_M2/746) and ALE-σ with constant closure (T_M2/498) vs GETM Table 2; one case definition, TEF at the transect, `compare.py`): the exact mixing relation holds to 1e-10 in both configurations (GETM: machine accuracy); Table 2 differences are mostly class (c) closure/advection (ALE-σ constant closure: mixing 3× GETM in A, no plug in B and C, exchange 6× in D; σ + GLS plug 3–4 g/kg saltier than GETM); not ✅ for the mixing magnitudes. Open: B and C (both configurations) and D σ + GLS are not steady at 600 cycles; GETM dV unexplained (+10–12 cm per 100 cycles in B–D, class d?); GLS vs GOTM column check pending; river inflow mixing not separated in M_num (≈ 98 % of ALE-σ M_num in B and D); ALE-σ + GLS (N3) in the N3 Lorenz row | `validation/lorenz_estuary/` | `tests/test_lorenz_estuary.py` | ✅ 12/12 |
| N3 closures as components (stage N, spec `docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md`, reference reads `docs/design/n2c_refs/`): the closure is its own component (`core/components/closure.py`: `ConstantClosure`, `GLSClosure`, `KPPClosure` behind one interface — `viscosity`, `update`, `tracer_diffusivity`, `nonlocal_flux`, `remap`, `state`/`load`; restart variable names stay `gls_*`/`kpp_akv`), run by the same ten-phase driver; no physics change at extraction (task 1, 31 golden configurations bit for bit); closure `'constant'`/`'none'` and σ GLS at default values are bit for bit as before N3 | `core/components/closure.py`, `core/model/driver.py`, `core/model/preflight.py`, `core/restart.py` | `tests/test_closure_component.py`, `tests/test_n1_golden.py` | ✅ |
| N3 prescribed surface heat flux `step(..., heat={"Q_ns", "Q_sw"})` in σ and ALE: non-solar part in the uppermost layer with mass, shortwave by the Paulson & Simpson (1977) Jerlov profile (types I–III; CROCO `lmd_swfrac.F:47-51`), the part reaching the bed stays in the bottom layer with mass (MOM6 `MOM_opacity.F90:824-842`), vanished layers get nothing, land/dry ignored; heat content exact to < 1e-13 of the change, T variance budget keeps closing (≤ 5e-10); `ctx.bflux` feeds KPP; no `heat` = bit for bit. Bulk formulae stay in N2b | `physics/surface/shortwave.py`, `core/model/driver.py` | `tests/test_surface_heat.py` | ✅ 13/13 |
| N3 periodic boundaries `periodic=("x",)|("y",)|("x","y")` in σ and ALE (pad–crop halos for stencil kernels, native wrap pairs for the free surface, PGF and drag; wrap faces stored twice and kept equal bit for bit; refused together with an open edge on the same axis; restart checks it): doubly periodic column stays uniform to 1e-12, tracer returns after one lap, free-surface wave period 9e-5 from the discrete value, translation invariance < 1e-12 in four configurations; cost ≈ +26 % tracer advection at nx = 100 (halo 13) | `core/model/periodic.py`, `core/components/*.py`, `core/dynamics/wetting_drying.py` | `tests/test_periodic.py` | ✅ 23/23 |
| N3 law-of-the-wall bed drag in ALE (`z0` with `LogLawBedDrag`: C_d from the bed layer d_b = max(h_b, min(hbbl, D_f)), stress as per-layer implicit rates over the bottom `hbbl`, MOM6 `MOM_set_viscosity.F90:651-685`; σ ≡ ALE to 1e-12 for a fixed-layer column) and, since task 9b (2026-10-29, class (a) fix), **the full bed speed** √(u² + v̄²) with the cross component from CROCO's 4-point mean (`get_vbc.F:517-526`; `cross_velocity_at_faces`): FORCED_EKBBL bed-layer veering 62.8° → 75.99° vs CROCO 75.77°, DBLEEK 63.23° vs 63.06°; open-edge faces take a zero cross component (as walls), velocities at u + ΔtG (CROCO uⁿ), class c; runs with v ≡ 0 unchanged bit for bit, goldens of z0 + v ≠ 0 regenerated on purpose (`sigma_gls`, `sigma_loglaw`, `sigma_parabolic`, two GLS hashes) | `physics/bottom_friction.py`, `core/components/bottom_drag.py`, `core/dynamics/sigma_wd.py` | `tests/test_ale_loglaw.py`, `tests/test_loglaw_full_speed.py` | ✅ 11 + 5 |
| N3 stratified GLS k-ε in ALE (σ, z\*, hybrid; `gls_production='croco'|'energy'`, Burchard 2002 eq. 16): k, ε, ν, κ_t at the interfaces; remap on the dual grid w_i = ½(h_{i−1} + h_i) with the model's conservative PPM_H4 (Σ w k exact to 1e-12, floors k ≥ 1e-10, ε ≥ 1e-12, Galperin limit with the N² of the last update); thin-layer merge as MOM6 `eliminate_massless` (`MOM_kappa_shear.F90:254-307`, top cluster `:256-265`) with a per-column threshold h_thin = min(10 cm, max(1 mm, 1 % D/N)) (task 9c; HYCOM `onemm`/`tencm` as floor and cap, the relative form is ours; MOM6's Δt form 0.77–2.4 mm was too small; ν, k, κ_t of a column with 1 mm–2 cm layers 1.00 of the column without them, was 6–128×), Numba merge kernel with the NumPy path as oracle (bit for bit; GLS update 2.3× faster); z\* with fixed layers = σ to ≤ 4e-13, hybrid with massless layers finite and floors held, restart bit for bit; open: layers just above the threshold (5 cm vs 4.2 cm: ν 4×), gather of the prior k, ε at a merged interface is not content-conserving | `core/components/closure.py`, `physics/mixing/gls.py`, `physics/mixing/gls_merged.py` | `tests/test_gls_ale.py` | ✅ 14 + thin-layer probes |
| N3 KPP closure (`closure='kpp'`, `KPPClosure`) in σ and ALE: column port of CVMix/MOM6 (`kpp_column`, OBL bulk-Richardson search, CVMix shape functions, enhanced diffusion, non-local term applied explicitly with exact content, constants cited to `cvmix_kpp.F90`/`MOM_CVMix_KPP.F90`, no tuning), B_f(d) = B_ns + B_sw(1 − F_sw(z_bot(d))) with `MXL_SW` (MOM6 `buoyFlux(1)` includes the penetrating shortwave, `MOM_forcing_type.F90:1113,1143`), MOM6 diabatic order (K from the state before the heat), **no background diffusivity by default** (anti-goal; `KPPParams.kappa_bg = nu_bg = 0`), no interior Ri mixing; the sign of the surface buoyancy flux of the legacy `kpp.py` fixed (surface heating is stabilising, task 6); restart holds `akv` only; ν not remapped in ALE (as MOM6) | `physics/mixing/kpp_cvmix.py`, `physics/mixing/kpp.py`, `core/components/closure.py` | `tests/test_kpp_cvmix.py`, `tests/test_kpp_model.py`, `tests/test_kpp.py` | ✅ 11 + 15 (+ 32 legacy) |
| N3 single-column benchmarks `validation/turbulence_columns/` (5×5 periodic, Δt 30 s; CROCO built locally: Kato–Phillips, Willis–Deardorff, diurnal cycle, FORCED_EKBBL/DBLEEK/NONROTBBL/OSCNONROTBBL with GLS σ/z\*/hybrid; MOM6/CVMix `CVmix_SCM_tests` with KPP z\*/σ): Kato–Phillips D_m = CROCO (σ/z\* −0.0 %, hybrid +0.1 %) and +4.4 % over Price (1979) as CROCO, Willis–Deardorff < 1e-3 of the range, diurnal ≤ 0.4 %, DBLEEK ≤ 0.8 %, NONROT ≤ 0.9 %, EKBBL 75.99° vs 75.77° (after 9b), z\* fixed layers = σ ≤ 3e-9; KPP OBL depth and MLD₀.₀₃ vs MOM6 ≤ 0.3 % in wind/cooling/mech cases (T identical to float32, wind-case velocity 0.6–1.3 % class c), vs CROCO KPP 13–27 % shallower (class c; CROCO linear-EOS α lacks 1/ρ₀, `alfabeta.F:133,138`, class d); `gls_production` 'energy' = 'croco' to < 1 cm in D_m; cost 10–25× CROCO in the periodic columns (halo), 38–79 ms/step vs MOM6 16; open: GLS vs GOTM column, k/ε remap numerical-mixing diagnostic | `validation/turbulence_columns/` | `tests/test_turbulence_columns.py` | ✅ 18/18 |
| N3 Lorenz et al. (2021) ALE-σ + GLS (`ale_sigma_gls`: ALE-σ, GLS, z0 = 1e-3, `hbbl` 0.05 m, Δt T_M2/746 because T_M2/498 stops at cycle 131 in D) and DOME hybrid + GLS (`--closure gls`, 10 km, 50 days): Lorenz ALE-σ + GLS within 0–4 % of σ + GLS in S_max, Me, Q_in, s_in; exact mixing relation 1.6e-10 / 1.9e-10 / 1.1e-10 in B / C / D but **1.5e-9 in A** (limit 5e-10): one-signed round-off gain of the ALE remap (+1.4 ε Σ V S² per step, `remap.py:369-372` = MOM6 `MOM_remapping.F90:938-939`), class (c), not fixed (changes the bits of all ALE, waits for Esteban), the per-step bias explains the absolute error only in A (50–60 % in B, D); DOME hybrid b_f −12.99e-3 (−16.18e-3 constant ν; Legg iso-ν0.5 −12.0e-3; 76 % of the way), T in the exact source range, **one** CFL truncation at day 31.8 (1.3 cm face layer, v = 8.96 m/s; criterion "no new truncations" not met), 1072 limiter events, max\|u\| 5.7 vs 3.2 m/s, cost 20.9 vs 14.9 s/day; ν_v = 1e-5 constant aborts by CFL (a finding); dV of GETM still unexplained (pyOCEAN ≤ 0.026). Also found and fixed: 0/0 in the depth-mean correction δ on land faces (`sigma_wd.py`, guarded, wet faces bit for bit) | `validation/lorenz_estuary/`, `validation/dome/` | `tests/test_lorenz_estuary.py`, `tests/test_dome_benchmark.py`, `tests/test_sigma_ale.py` | 🔶 magnitudes class (c); balance A and one truncation explained |

**Phase 2 in progress — 1756 total tests passing (+2 xfail: 1 strict, Thacker; 1 pending decision: FV vs Jacobian PGF over the seamount), N3 closed 2026-10-30.**

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
- Every long run must export `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`: BLAS threads oversubscribe the 5 Numba threads (found in the Lorenz estuary runs, N2a).
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
