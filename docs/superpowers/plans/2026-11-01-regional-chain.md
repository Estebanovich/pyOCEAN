# Regional chain (NWA12 cut) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run pyOCEAN and MOM6 on the same NWA12 cut (Mid-Atlantic Bight, 2017-05-01 → 09-01) with a curvilinear grid, GLORYS open boundaries, 3-D tides, optional atmospheric pressure, GloFAS rivers and ERA5 bulk forcing, and check the chain against the criteria of the spec.

**Architecture:** All general capability goes into the library:
- `core/grid/supergrid.py`: MOM6 supergrid → pyOCEAN grid.
- `forcing/rotation.py`: vector rotation.
- `forcing/sources/glorys.py` and `forcing/sources/glofas.py`: the two new data adapters.
- `forcing/tides.py`: MM and MF.
- `core/components/boundary_tides.py`, `atm_pressure.py`, `runoff.py`: new components run by the existing ten-phase driver.

The case lives only in `validation/nwa12_cut/`, with one case definition that writes the inputs of pyOCEAN and of MOM6. MOM6 runs coupled to SIS2 through FMScoupler, with inputs made by CEFI's own tools.

**Tech Stack:** Python only (NumPy, Numba, xarray, `copernicusmarine`, `cdsapi`). MOM6 + SIS2 + FMS in Fortran only inside `validation/nwa12_cut/` drivers and build scripts.

**Spec:** `docs/superpowers/specs/2026-11-01-regional-chain-design.md` (commit ea27574). Reference reads go to `docs/design/regional_refs/` (Task 1).

## Global Constraints

**Code and environment:**
- Python only; Fortran only in validation drivers and build scripts.
- NumPy arrays inside the step; xarray only at I/O. Index order (k, j, i): u-faces nx+1 columns, v-faces ny+1 rows.
- Type hints and physics docstrings, with paper or `file:line` references.
- Never `pip install` into base; numpy is pinned at 1.26.4.

**Runs:**
- ≤ 5 threads, P-cores only; export `OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 VECLIB_MAXIMUM_THREADS=1 MKL_NUM_THREADS=1`.
- Long runs one at a time; consult Esteban if a run exceeds ~24 h.
- Outputs and data go under `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/`, never the repo. Grid files are in `~/pyOCEAN_refs/nwa12_grid/`.

**Git:**
- Commit with explicit paths; never `git add validation/`, `-A` or `.`.
- Never commit logs, npz, figures, NetCDF or binaries.
- Commit trailer:
  `Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>`
  `Claude-Session: https://claude.ai/code/session_01DapNDveddxQLd53ugW9ton`

**Results:**
- Linear-EOS, Cartesian and spherical runs stay bit for bit: N1 goldens (`tests/test_n1_golden.py`) and 9/9 smoke hashes (`scripts/n1_smoke.py`). The only exception is a documented, deliberate regeneration from the Task 3 metric audit.
- Every new option defaults to off. Off means bit for bit as before.
- Never tune a parameter to the cut. Classify every difference (a)–(d).
- NWA12 parity values are passed as parameters and never become pyOCEAN defaults:
  - velocity nudging 3 d in / 360 d out;
  - 10 constituents M2, S2, N2, K2, K1, O1, P1, Q1, MM, MF;
  - z\* `vgrid_75_2m.nc`;
  - `eos="wright_full"`;
  - bulk `'ncar'`;
  - `atm_pressure=False`.

## Review Focus

1. **Land and NaN at an open edge.** GLORYS cells that are NaN near the coast at a boundary face must take the nearest-ocean value, never NaN, and never values from beneath the bed (test in Task 7).
2. **Supergrid not exactly orthogonal, or angle near ±π.** The rotation must use cos/sin from the supergrid, not `arctan` of differences that wrap. Non-orthogonality above 1e-3 rad raises with the location (test in Task 2).
3. **River mouths.** A mouth that maps to a land cell, an open-boundary cell or a cell with H ≤ 0 goes to the nearest wet interior cell and is reported. Total discharge is conserved exactly (test in Task 8).
4. **Model time outside the coverage.** Time outside GLORYS, TPXO ramp, GloFAS or ERA5 coverage raises a `ValueError` naming the source and the dates, at the first call (tests in Tasks 6, 7 and 8).
5. **Restart between daily records.** A restart at a time between two GLORYS or GloFAS records, during the tidal ramp, is bit for bit with the straight run (test in Task 9).

---

### Task 1: Reference reads (MOM6/SIS2/CEFI)

**Files:**
- Create: `docs/design/regional_refs/mom6_grid.md`
- Create: `docs/design/regional_refs/mom6_obc_tides.md`
- Create: `docs/design/regional_refs/mom6_psurf_runoff.md`
- Create: `docs/design/regional_refs/cefi_tools.md`

**Interfaces:**
- Produces: the algorithms with `file:line` (MOM6 git as in `docs/design/teos10_refs/mom6_eos.md`, CEFI git 36b8f49) that Tasks 2–10 cite.

- [ ] **Step 1: `mom6_grid.md`.** Cover:
  - `set_grid_metrics_from_mosaic`: the supergrid → dxT, dyT, dxCu, dyCu, dxCv, dyCv, dxBu, dyBu, areaT/Cu/Cv/Bu, the angle, lon/lat;
  - which supergrid half-segments are summed;
  - how `angle_dx` is formed;
  - f at the q points;
  - how a regional cut indexes the supergrid (even/odd offsets).
- [ ] **Step 2: `mom6_obc_tides.md`.** Cover:
  - OBC segment data with `OBC_TIDE`: amplitude/phase → SSH and barotropic u, the nodal and equilibrium phase with `TIDE_REF_DATE`, `TIDE_NODAL_REF_DATE`, `TIDE_USE_EQ_PHASE`, and the ramp;
  - whether tides are added to the baroclinic velocity;
  - FLATHER, ORLANSKI and NUDGED with the two time scales;
  - the tracer reservoir;
  - `TIDES=True` body tide and SAL in NWA12 (scalar or read; `TIDE_SAL_SCALAR_VALUE`);
  - MM and MF frequencies and nodal factors.
- [ ] **Step 3: `mom6_psurf_runoff.md`.** Cover:
  - how p_surf enters the PGF and the barotropic solver;
  - what `MAX_P_SURF=0` does;
  - how `lrunoff`/`frunoff` enter (volume source, temperature, salinity, depth spread);
  - where SIS2/coupler rotate wind stress and currents to the grid angle.
- [ ] **Step 4: `cefi_tools.md`.** Cover:
  - inputs and outputs and the exact command lines of `tools/boundary/write_glorys_boundary.py`, `write_tpxo_boundary.py`, `tools/initial/write_glorys_initial.py`, `tools/atmos/*`, `tools/rivers/*` and `tools/grid/make_mosaic.sh`/`make_mask.sh`;
  - which ones need FRE-NCtools;
  - how NWA12 builds `ocean_mosaic`, `grid_spec` and the exchange grids.
- [ ] **Step 5: Commit** the four files.

### Task 2: Grid from a MOM6 supergrid

**Files:**
- Create: `core/grid/supergrid.py`
- Test: `tests/test_supergrid.py`

**Interfaces:**
- Produces: `grid_from_supergrid(hgrid: xr.Dataset, depth: np.ndarray | None = None, window: tuple[int, int, int, int] | None = None, omega: float = 7.2921e-5) -> xr.Dataset`.
  - `window` = (i0, i1, j0, j1) in T-cell indices of the full grid.
  - Returns the same variables as `core/grid/horizontal._assemble_dataset`, plus `dx_q`, `dy_q`, `area_q`, `lon_t`, `lat_t`, `lon_u`, `lat_u`, `lon_v`, `lat_v`, `lon_q`, `lat_q`, `cos_t`, `sin_t` (rotation from grid-x to east), and attrs `grid_type="mom6_supergrid"`.
  - Metric semantics are those of the existing dataset (`dx_u` = distance between T points across a u-face, `dy_u` = u-face length; likewise for v).
- Produces: `supergrid_from_arrays(x: np.ndarray, y: np.ndarray, dx: np.ndarray, dy: np.ndarray, area: np.ndarray, angle_dx: np.ndarray) -> xr.Dataset` (test helper and synthetic builder).

- [ ] **Step 1: Failing tests.**
  - `test_cartesian_supergrid_equals_create_cartesian_grid`: a synthetic Cartesian supergrid (nx=12, ny=9, dx=5 km) → every shared variable equal to `create_cartesian_grid` with `np.array_equal`.
  - `test_spherical_supergrid_equals_create_spherical_grid`: the same against `create_spherical_grid` to ≤ 1e-12 relative.
  - `test_window_matches_full_then_slice`: a window equals slicing the full result.
  - `test_rotated_grid_angle`: a grid rotated by 30° gives cos_t = cos 30°, sin_t = sin 30° to 1e-12.
  - `test_non_orthogonal_raises`: skew > 1e-3 rad raises `ValueError` naming (j, i).
- [ ] **Step 2:** Run `pytest tests/test_supergrid.py -v`. Expect FAIL (module missing).
- [ ] **Step 3: Implement** following `docs/design/regional_refs/mom6_grid.md`:
  - metrics are sums of supergrid half-segments, areas sums of four sub-cells;
  - f_q = 2Ω sin(lat_q);
  - `mask_t` = depth > 0 when `depth` is given, else ones.
- [ ] **Step 4:** Run the tests. Expect PASS. Also `pytest tests/test_grid_horizontal.py`.
- [ ] **Step 5:** Commit `core/grid/supergrid.py` and `tests/test_supergrid.py`.

### Task 3: Metric audit for orthogonal curvilinear grids

**Files:**
- Modify: whichever files the audit finds. Candidates (from `grep -rln "dx_t\|dy_u\|dx_v"`): `core/dynamics/{sigma_baroclinic,sigma_wd,free_surface,vector_invariant,hor_visc,wetting_drying,nonlinear}.py`, `core/components/{layer_transport,vertical_coordinate}.py`, `core/vertical/{pgf,pgf_nonlinear}.py`, the tracer advection kernels, `core/model/periodic.py`.
- Create: `docs/design/regional_refs/metric_audit.md` (table: file:line, metric used, metric MOM6 uses, verdict).
- Test: `tests/test_curvilinear_invariance.py`

**Interfaces:**
- Consumes: `grid_from_supergrid`, `supergrid_from_arrays` (Task 2).
- Produces: no new API. `SigmaBaroclinic` is correct on any orthogonal grid that `grid_from_supergrid` builds.

- [ ] **Step 1: Failing tests** (criterion 2, tolerance 1e-10 relative). Each runs `SigmaBaroclinic` on a Cartesian grid and on the same physical grid expressed as a curvilinear supergrid whose metrics vary per index in the way a rotated, stretched grid's do. The grid has non-uniform dx(j, i) and dy(j, i), with `dy_u ≠ dy_t` and `dx_v ≠ dx_t`. Build it with `supergrid_from_arrays` so that the Cartesian equivalent is a stretched Cartesian grid with the same per-face metrics.
  - `test_free_surface_wave_invariant`;
  - `test_geostrophic_adjustment_invariant` (f-plane);
  - `test_tracer_advection_invariant` (WENO5 + FCT);
  - `test_stratified_rest_invariant` (z\* and hybrid, `eos="wright_full"`).
  - Plus `test_face_metrics_used`: perturb `dy_u` only and assert the u-face transport changes in proportion. This catches code that uses `dy_t` or `dy_v` for a u-face length.
- [ ] **Step 2:** Run. Expect a failure wherever a metric is misused. Record which.
- [ ] **Step 3: Audit and fix.** For every metric use, check it against MOM6's operator (continuity: face lengths dyCu/dxCv; PGF: dxCu/dyCv; Coriolis/vorticity: dxBu, dyBu, areaBu; Smagorinsky: MOM6 `MOM_hor_visc` metrics; tracer fluxes: face lengths / areaT). Fix and write each row of `metric_audit.md`.
- [ ] **Step 4:** Run the new tests, then `pytest tests/` (whole suite).
  - Run `python scripts/n1_smoke.py` against `main` and `pytest tests/test_n1_golden.py`. Both must be bit for bit.
  - If a fix changes a Cartesian/spherical result, stop and record why in `metric_audit.md`. Regenerate the goldens (`scripts/n1_make_golden.py`) only when the old result used the wrong metric on a grid where the two differ (spherical), and list the regenerated cases in the commit message.
- [ ] **Step 5:** Commit the touched files explicitly.

### Task 4: Vector rotation for forcing and boundaries

**Files:**
- Create: `forcing/rotation.py`
- Modify: `forcing/atmosphere.py`, `core/components/surface_fluxes.py`
- Test: `tests/test_rotation.py`

**Interfaces:**
- Produces:
  - `to_grid(u_east, v_north, cos_a, sin_a) -> (u_x, v_y)`
  - `to_earth(u_x, v_y, cos_a, sin_a) -> (u_east, v_north)`
  - These are exact inverses. The angle convention is MOM6's `angle_dx` (`cos_t`, `sin_t` of Task 2).
- `AtmosphereProvider` returns u10, v10 on grid axes when the grid carries `cos_t`/`sin_t`. Without them, behaviour is unchanged and bit for bit.
- The bulk works on the grid-axis wind and current.

- [ ] **Step 1: Failing tests.**
  - `test_round_trip` (1e-15);
  - `test_rotation_30deg` against an analytic field;
  - `test_bulk_stress_rotated_grid_equals_unrotated`: the same atmosphere and ocean on a rotated grid give stress whose `to_earth` equals the unrotated stress to 1e-12;
  - `test_no_angle_bit_for_bit`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement. Rotate in the provider after the bilinear step (MOM6/SIS2 rotate on the ocean grid; cite Task 1 `mom6_psurf_runoff.md`).
- [ ] **Step 4:** Run the tests plus `tests/test_atmosphere.py` and `tests/test_surface_fluxes_model.py`.
- [ ] **Step 5:** Commit.

### Task 5: MM/MF and tides at the 3-D open boundary, body tide in 3-D

**Files:**
- Modify: `forcing/tides.py` (MM, MF in `CONSTITUENTS` tables, frequencies, equilibrium arguments, nodal factors), `forcing/sources/tpxo.py` (read `mm`, `mf` if present in the atlas)
- Create: `core/components/boundary_tides.py`
- Modify: `core/model/driver.py`, `core/dynamics/sigma_baroclinic.py`, `core/model/preflight.py`, `core/restart.py`, `core/components/open_boundary.py`
- Test: `tests/test_tides.py` (extend), `tests/test_boundary_tides_3d.py`

**Interfaces:**
- Produces: `BoundaryTides(constants: xr.Dataset, constituents: list[str], ref_date: np.datetime64, nodal_ref_date: np.datetime64 | None = None, ramp_days: float = 0.0)`.
  - `constants` = TPXO-style η and transport amplitude/phase.
  - Method `at(t: float) -> dict[str, tuple[np.ndarray, np.ndarray]]`: per open edge, (η_tide, ū_tide normal), with u = U/D using the model face depth. Vectors are rotated with `forcing.rotation.to_grid`.
- `SigmaBaroclinic(..., boundary_tides: BoundaryTides | None = None, tidal_potential: bool = False, sal_beta: float = 0.0)`:
  - in `'radiating'` mode, the Flather exterior becomes η_ext + η_tide and ū_ext + ū_tide;
  - u′ is untouched;
  - `tidal_potential` adds the body-force gradient (existing `forcing.tides.body_tide`/`body_force`) as a barotropic term, with SAL as g(1 − β)∇η, as `ImplicitFreeSurface(sal_beta=…)`.
- Constituents follow the TPXO/OTPS naming (`mm`, `mf`). Nodal and equilibrium handling follows Task 1 `mom6_obc_tides.md`.

- [ ] **Step 1: Failing tests.**
  - `test_mm_mf_frequencies_and_nodal` against OTPS/pyTMD values: generate references with pyTMD if installed, else from the OTPS tables in the reference read, and store them in `tests/data/`.
  - `test_existing_eight_bit_for_bit`.
  - `test_3d_tidal_channel`: a straight 3-D channel, z\*, homogeneous, one open edge forced with M2 η amplitude 0.5 m. η at the closed end matches the 2-D `wetting_drying` model's tidal channel solution to ≤ 2 % in amplitude and ≤ 2° in phase after the ramp.
  - `test_tides_off_bit_for_bit`.
  - `test_tidal_potential_3d_equals_2d`: a closed homogeneous basin with only the body tide; the depth-mean η of the 3-D run equals the 2-D `ImplicitFreeSurface` run to 1e-6 relative.
  - `test_restart_during_ramp_bit_for_bit`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
  - The restart stores nothing new (pure function of t) but checks constituents, ref dates and ramp.
  - Pre-flight refuses `boundary_tides` without a `'radiating'` edge.
- [ ] **Step 4:** Run the tests, the suite, goldens and smoke.
- [ ] **Step 5:** Commit.

### Task 6: Atmospheric pressure

**Files:**
- Create: `core/components/atm_pressure.py`
- Modify: `core/model/driver.py`, `core/dynamics/sigma_baroclinic.py`, `core/components/open_boundary.py`, `core/model/preflight.py`, `core/restart.py`, `core/model/registry.py`
- Test: `tests/test_atm_pressure.py`

**Interfaces:**
- `SigmaBaroclinic(..., atm_pressure: bool = False)` needs `surface_fluxes` (it uses `msl` from the same `AtmosphereProvider` at t^{n+½}).
  - Adds −∇p_a/ρ₀ to the depth-mean momentum, the same way η's gradient enters, following Task 1 `mom6_psurf_runoff.md`.
  - At `'radiating'` edges, η_ext += −(p_a − p̄_a)/(ρ₀ g), where p̄_a is the mean over wet T cells at that time.
- Registry diagnostic `p_atm`.

- [ ] **Step 1: Failing tests.**
  - `test_static_inverse_barometer`: closed basin, constant p_a gradient 1 hPa per 100 km, damped run of 10 days. η = −p_a′/(ρ₀ g) to ≤ 1 % and volume exact to 1e-13.
  - `test_ib_open_boundary_no_reflection`: a uniform-in-space p_a step in a channel open at one end. With IB at the edge, η → −Δp/(ρ₀g) and the basin-mean energy left after 2 days is < 1 % of the run without IB.
  - `test_off_bit_for_bit`.
  - `test_preflight_refuses_without_bulk`.
  - `test_restart_bit_for_bit`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests, the suite, goldens and smoke.
- [ ] **Step 5:** Commit.

### Task 7: GLORYS adapter, initial state and boundary exterior

**Files:**
- Create: `forcing/sources/glorys.py`, `scripts/fetch_glorys.py`
- Modify: `forcing/sources/__init__.py` (registry), `forcing/ocean_init.py` (rotate u, v with `to_grid` when the grid has an angle)
- Test: `tests/test_glorys.py`

**Interfaces:**
- `GLORYSSource` with the adapter contract of `forcing/sources/base.py` (pattern: `hycom_z.py`). It reads `thetao` → θ [°C], `so` → S_P, `uo`/`vo` → east/north velocities, and `zos` → η.
  - Land and below-bed NaN are filled by nearest ocean neighbour on each level, as CEFI `fill_glorys_nn.sh` (Task 1 `cefi_tools.md`).
  - `ts_kind="theta_sp"`.
- The existing `prepare_ocean`, `initial_state(..., h_src=...)` and `boundary_data(prepared, side, t)` work with it unchanged, except for the rotation.
- A time outside the files raises `ValueError` naming the coverage.
- `scripts/fetch_glorys.py --start --end --west --east --south --north --outdir` wraps `copernicusmarine subset`.

- [ ] **Step 1: Failing tests** on a small synthetic GLORYS-layout file in `tests/data/` (generated by the test):
  - `test_read_canonical_units_and_dims`;
  - `test_nan_filled_from_nearest_ocean` (Review Focus 1);
  - `test_boundary_faces_on_cgrid_points` (u at u-faces of west/east, v at v-faces of south/north);
  - `test_rotated_grid_velocity`;
  - `test_initial_state_column_content_conserved` (heat and salt ≤ 1e-12 relative);
  - `test_time_outside_coverage_raises`;
  - `@pytest.mark.skipif(no real data)` `test_real_file_matches_copernicusmarine` (criterion 3) on `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/data/glorys/`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and `tests/test_hycom_z.py tests/test_ocean_init.py`.
- [ ] **Step 5:** Commit.

### Task 8: Rivers (GloFAS → coastal volume sources)

**Files:**
- Create: `forcing/sources/glofas.py`, `core/components/runoff.py`
- Modify: `core/model/driver.py`, `core/dynamics/sigma_baroclinic.py`, `core/model/preflight.py`, `core/restart.py`
- Test: `tests/test_runoff.py`

**Interfaces:**
- `glofas_mouths(path_or_ds, grid: xr.Dataset, t0, t1, min_mean_discharge: float = 1.0) -> RiverInput`.
  - Maps GloFAS cells at the coast to the nearest wet, interior, H > 0 model cell, following the CEFI `tools/rivers` rule (Task 1).
  - `RiverInput` holds `cells: np.ndarray (n, 2)` (j, i), `times`, `Q: np.ndarray (nt, n)` [m³ s⁻¹], and `report: list[str]` of moved mouths.
- `SigmaBaroclinic(..., runoff: RiverInput | None = None)`:
  - adds Q/area_t to the real freshwater flux `fw` (N2a path: S = 0, T = top-layer T) together with bulk P − E;
  - linear in time between daily records;
  - registry diagnostic `runoff`.
- A time outside the coverage raises (Review Focus 4).

- [ ] **Step 1: Failing tests.**
  - `test_mouth_on_land_moved_to_nearest_wet_interior` (Review Focus 3);
  - `test_total_discharge_conserved` (Σ over cells = Σ GloFAS mouths, exact);
  - `test_volume_and_salt_budget_with_runoff` (volume gain = ∫Q dt to 1e-13 relative; salt content unchanged to 1e-13);
  - `test_runoff_with_bulk` (both act; no refusal);
  - `test_off_bit_for_bit`;
  - `test_restart_between_records_bit_for_bit`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement.
- [ ] **Step 4:** Run the tests and the suite.
- [ ] **Step 5:** Commit.

### Task 9: Case definition, pyOCEAN inputs and runs

**Files:**
- Create: `validation/nwa12_cut/{case.py, build_inputs.py, run_pyocean.py, README.md}`
- Test: `tests/test_nwa12_cut.py`

**Interfaces:**
- `case.py`:
  - `CASE` with `window` (chosen once by `choose_window()` from `ocean_hgrid.nc`: box 75.5–68.5 °W, 36.5–41.8 °N; every open edge ≥ 10 cells from NDBC 44025 (40.251 °N, 73.164 °W) and OOI Pioneer (40.1 °N, 70.8 °W); no H ≤ 0 cell on an open edge; edges with any wet cell are open);
  - `t0 = 2017-05-01`, `t1 = 2017-09-01`;
  - data paths;
  - `parity_config()` returning the keyword arguments of `SigmaBaroclinic`: z\* with `vgrid_75_2m.nc` (from `~/pyOCEAN_refs/CEFI-regional-MOM6/tools/grid/`), `eos="wright_full"`, `ts_kind="theta_sp"`, bulk `'ncar'`, `'radiating'` with nudging 3 d / 360 d, 10 constituents, `tidal_potential=True` with SAL as NWA12 (Task 1), `atm_pressure=False`, GloFAS, `closure='kpp'`, `shear_mixing='jackson'`, `smagorinsky=0.15`, log-law or quadratic drag as NWA12 (`CDRAG`, Task 1).
- `build_inputs.py`: writes the pyOCEAN grid NetCDF and the river input to `/Volumes/.../nwa12_cut/inputs/pyocean/`.
- `run_pyocean.py --mode {tides,full} --days N --threads 5 --restart-at DAY`:
  - `tides` = homogeneous, boundary tides + potential only, 30 days;
  - `full` = parity, 123 days;
  - writes CF output, series at the gauge and buoy points, the domain budget (`TEFRegion` over all wet cells plus the open-face, bulk and runoff fluxes), and the cost per simulated day.

- [ ] **Step 1: Failing tests** (fast; synthetic or skip without data):
  - `test_choose_window_rules` on a synthetic hgrid;
  - `test_parity_config_preflight_clean` (pre-flight returns no issues on a tiny synthetic version of the case);
  - `test_restart_mid_run_bit_for_bit_small` (Review Focus 5: a 12 × 10 × 8 synthetic regional case with GLORYS-layout boundary files, tides with ramp, ERA5-like atmosphere and runoff; 2 days straight vs 1 day + restart + 1 day);
  - `test_threads_1_equals_5_small`.
- [ ] **Step 2:** Run. Expect FAIL.
- [ ] **Step 3:** Implement. Run on the real data:
  - tides-only (30 days);
  - full (123 days) with a restart at day 60 and a straight control (bit-for-bit check). Criterion 6: budgets ≤ 1e-10 relative, 0 CFL truncations, cost per simulated day.
  - Record the largest stable Δt, measured without tuning: try 600 s, then 300 s.
- [ ] **Step 4:** Run the tests and the suite.
- [ ] **Step 5:** Commit the code and README (no outputs).

### Task 10: MOM6 + SIS2 on the same cut

**Files:**
- Create: `validation/nwa12_cut/{mom6_inputs.py, build_mom6.sh, run_mom6.sh, MOM_input, MOM_override, SIS_input, diag_table, data_table, input.nml}` (templates derived from `exps/NWA12.COBALT` without COBALT, with the cut's grid and segments)
- Create: `validation/nwa12_cut/check_mom6.py`
- Test: `tests/test_nwa12_cut.py` (extend; skip without the MOM6 outputs)

**Interfaces:**
- Consumes: `CASE` (Task 9), the CEFI tools (Task 1 `cefi_tools.md`), SIS2 and the null components in `~/pyOCEAN_refs/` (download agent).
- `mom6_inputs.py`: cut supergrid and topography (`ocean_hgrid.nc`, `ocean_topog.nc` at the window), mosaic/exchange grids per `cefi_tools.md`, and the GLORYS and TPXO segments, initial state, ERA5 and GloFAS through CEFI's scripts. Inputs go to `/Volumes/.../nwa12_cut/inputs/mom6/`.
- `build_mom6.sh`: MOM6 + SIS2 + FMScoupler + null atmos/land against FMS 2025.02.01, in `~/pyOCEAN_refs/build/nwa12_cut/`.
- `run_mom6.sh {tides|full}`: at most 5 MPI ranks.
- `check_mom6.py`:
  - criterion 1: `ocean_static.nc` metrics, areas, f and angle vs `grid_from_supergrid` ≤ 1e-12 relative;
  - criterion 7: initial PGF vs pyOCEAN ≤ 1e-11 m s⁻², using `diag_table` output of the first step's PGF and pyOCEAN's `keep_terms`.

- [ ] **Step 1: Failing tests** (skip without outputs): `test_static_metrics_match_mom6`, `test_initial_pgf_matches_mom6`.
- [ ] **Step 2:** Build. Run tides-only (30 days) and full (123 days). Record the cost per simulated day on 5 ranks.
- [ ] **Step 3:** Run the tests. On failure, classify (a)–(d) in the README.
- [ ] **Step 4:** Commit the scripts and templates (no outputs, no binaries).

### Task 11: Comparison, documentation and scorecard

**Files:**
- Create: `validation/nwa12_cut/compare.py`
- Modify: `validation/nwa12_cut/README.md`, `CLAUDE.md` (new Phase 2 row, test count), `docs/design/vision_roadmap.md` (§7 scorecard, §8 status), `docs/literature/reading_list.md` (papers used: GLORYS, GloFAS, CEFI/NWA12, TPXO)
- Test: `tests/test_nwa12_cut.py` (extend: harmonic analysis helper on a synthetic series)

**Interfaces:**
- `compare.py`:
  - **Tides-only:** M2, S2, N2, K1, O1 amplitude, phase and vector error at the CO-OPS gauges for pyOCEAN, MOM6 and TPXO10 against the CO-OPS `harcon.json` constants. Use `diagnostics/tide_gauges.py` harmonic analysis.
  - **Criterion 4:** the median M2 vector error of pyOCEAN is not worse than MOM6's on the same cut, or the difference is classified.
  - **Full run, first look without scoring:**
    - η at the gauges against the verified hourly levels;
    - SST against GLORYS and, if Esteban authorises the small extra download, NOAA OISST v2.1 (otherwise state it is pending);
    - T and S at 44025 against the buoy.
  - Ends with the "what pyOCEAN should improve" list. Missing NWA12 physics (ePBL, double diffusion, biharmonic Smagorinsky, tracer reservoir) is class (b)/(c), to be decided in project 4.
- [ ] **Step 0: Observation QC first (Esteban, 2026-11-02).** Before any skill number, classify every observation record as usable or corrupted, with the reason. Gauges (CO-OPS 6-min and hourly, 13 stations): `diagnostics/sealevel_qc.py`, which checks spikes, stuck values, datum steps, clock offsets and twin sensors, plus checks against CO-OPS `harcon` and gaps (Montauk). NDBC 44025: range, spike and stuck checks, and gaps. Write the QC table (record, verdict, reason) in the README. Only usable records enter the metrics.
- [ ] **Step 1: Failing test** for the harmonic helper on a synthetic M2 + K1 series: amplitude to 1e-6, phase to 1e-4°.
- [ ] **Step 2:** Implement and run `compare.py`.
- [ ] **Step 3:** Update the README with every criterion (✅/🔶, numbers and classification), then the `CLAUDE.md` row and the roadmap scorecard.
- [ ] **Step 4:** Run the full suite and record the count.
- [ ] **Step 5:** Commit.
