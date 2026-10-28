# Troubleshooting and FAQ

Each item is **symptom → cause → fix**, with the code location of the message or mechanism. Start with the
[domain checker](output-restart-diagnostics.md#domain-checker) before a run and the
[blow-up guard](output-restart-diagnostics.md#blow-up-post-mortem-blowupguard) during it: most failures below are
reported with a location by one of them.

## Run stops with "CFL velocity truncations exceed max_truncations"

**Symptom.** `RuntimeError: N CFL velocity truncations exceed max_truncations=1000; first this step: layer k at u-face
(j=…, i=…): u = … m/s, face thickness … m` (``SigmaBaroclinic._step_ale`` (`core/dynamics/sigma_baroclinic.py:2069-2079`)).

**Cause.** After the remap, every ALE step applies a backstop copied from MOM6 `vertvisc_limit_vel`: where
|u| Δt/Δx_donor > 0.5 the velocity is reset to 0.45 Δx/Δt (``truncate_cfl``
(`core/dynamics/vector_invariant.py:136-170`)). It "should not act"; it is counted only at faces thicker than
`h_thin_bed` (0.01 m), so vanished layers do not count. `st.n_trunc` accumulates over the whole run (and across
restarts). All validation runs end with zero truncations (BayIW cut, `validation/bayiw_cut/README.md`; MITgcm
`internal_wave`, `validation/internal_wave/README.md`).

**Fix.**

1. Read the location in the message. A thick face with a large u means Δt is too long for the flow: estimate
    u_max honestly (an overflow falling down a slope reaches 3–10 m s⁻¹; `validation/ilicak_benchmark/README.md`)
    and use `check_domain(…, u_max=…).dt_max_advective` (`diagnostics/domain_check.py:198-199`).
2. A thin face (a few cm) in a hybrid run points to a vanished-layer problem (next sections).
3. More horizontal viscosity `ah` (lower grid Reynolds number) calms the flow; the Ilicak hybrid overflow completes at
    ν = 100 and 1000 m² s⁻¹ but not reliably at ν = 1 ([known issues](#known-open-issues)).
4. Raising `max_truncations` only lets the run continue with clipped velocities; do it to reach the post-mortem, not
    to produce results.

## NaN or blow-up

**Symptom.** NaN in the state, `wet-dry free-surface CG did not converge` (`core/dynamics/wetting_drying.py:292`),
or a `BlowupError` from the guard.

**Cause → fix**, in order of frequency:

| Cause | How to recognise it | Fix |
|---|---|---|
| Δt too long (advection or internal waves) | the guard's `local CFL` > 1, smooth growth in `recent max` | reduce Δt; check the CFL table of `check_domain` |
| bad cells in the bathymetry | single-cell spike in the `neighbourhood`; `check_domain` reports pits / isolated cells | `fill_isolated_cells`, `smooth_bathymetry` (`core/grid/bathymetry.py:304`, `:221`), or set the cell to NaN |
| land given as 0 instead of NaN | ALE: `cells with H ≤ 0 can dry …` at construction; σ: a shallow "flooded" band along the coast | `bathy = np.where(depth > 0, depth, np.nan)` ([Building a model](building-a-model.md#step-2-the-bathymetry-array)) |
| exterior data inconsistent with the interior | blow-up starts at an open edge | check `OBData` shapes, units, sign of `u_normal` (model sign: east/north positive); add a sponge |
| wind stress in very shallow cells | spike in the top layer of a shallow, nearly dry cell | τ enters as Δt τ/(ρ₀ h_top); reduce Δt or cap τ where D is small |
| vanished layers in hybrid | thin faces in the message, growing `st.n_limited` | see [vanished layers](#negative-or-vanished-layers-in-hybrid-runs) |

Read the report as explained in [Output, restarts and diagnostics](output-restart-diagnostics.md#blow-up-post-mortem-blowupguard);
reload the `.npz` dump to plot the neighbourhood.

## How do I choose Δt?

The free surface is θ-implicit (`WetDryFreeSurface`, Casulli 1990), so the external gravity wave does **not** limit
Δt. The limits that remain:

| Limit | Rule | Where it bites |
|---|---|---|
| advection | u Δt/Δx ≤ 0.5 (the ALE truncation threshold) | fast currents, overflows |
| internal waves | c₁ Δt/Δx ≲ 0.5–1, c₁ the mode-1 speed | stratified shelves |
| Lagrangian outflow (ALE) | no layer may lose more than it holds: `layer k … emptied in the Lagrangian step: h† = … (reduce dt)` (`core/vertical/ale.py:292`) | thin layers, strong convergence |
| inertial | f Δt ≲ 1 (warning in `check_domain`) | large Δt at high latitude |

`check_domain(h_grid, H, dt, u_max, gprime_h=c1**2)` prints the advective Δt for CFL 0.5 and classifies each limit.
Reference values from the validation:

| Case | Δx | Δt used | Largest stable Δt | Source |
|---|---|---|---|---|
| lock exchange (σ) | 500 m | 10 s | ≥ 160 s (z*, 2 h ladder) | `validation/lock_exchange/README.md`, `validation/ilicak_benchmark/README.md` |
| Ilicak overflow (z* / hybrid) | 1 km | 10 s | 80 s / 40 s | `validation/ilicak_benchmark/README.md` |
| Ilicak internal wave | 5 km | 300 s | 2400 s | same |
| seamount | 6 km | 600 s | 600 s (half of MITgcm and MOM6 σ, open) | same |
| BayIW cut (z*, hybrid) | stretched (MITgcm grid) | 120 s (MITgcm: 30 s) | — | `validation/bayiw_cut/README.md`, `validation/bayiw_baroclinic/README.md` |

In the lock exchange, Δt (5 or 10 s) and θ (0.5, 0.6, 1) do not change the RPE (`validation/lock_exchange/README.md`).
In the Ilicak internal wave, Δt matters through the time weight of the baroclinic transport, fixed by `theta_bc = ½`
(Ilicak README, item 9).

## Negative or vanished layers in hybrid runs

**Symptoms.** `limit_outflow: … would reach h = … — barotropic CFL` (`core/vertical/ale.py:390`), a `RuntimeWarning`
`limit_outflow: … still sends out more than it holds after 50 passes` (`:385`), growing `st.n_limited`, truncations at
thin faces, large velocities in layers of a few millimetres.

**Cause.** Hybrid layers whose target density is absent have no mass (`h` down to `h_min` = 1 mm). Since
2026-10-14 the outflow limiter (``limit_outflow`` (`core/vertical/ale.py:331`)) cuts the outgoing fluxes of a layer
to its content and gives the rest to the other layers of the face (Σ_k Q_k unchanged), and the vertical viscosity
uses MOM6's upwind-biased face thickness (`thin_layer_coupling="mom6"`), which ties a draining layer to its neighbours.
`st.n_limited` counts the cell-layers cut so far: a few events are normal; a steady growth means the layer structure is
fighting the flow.

**Fixes.**

1. **Give `dz_top` the full depth.** Σ `dz_top` ≥ max H (the constructor warns otherwise,
    `core/dynamics/sigma_baroclinic.py:1159-1168`). A floor that stops short leaves a homogeneous deep water mass in one
    slab layer (BayIW: interface rise 0.06 of MITgcm's; `validation/bayiw_baroclinic/bayiw.py:166-172`).
2. **Design `rho_target`** with the water masses at bracket centres and light targets for the mixed layer
    (``target_densities`` (`core/vertical/regrid.py:292`); `bayiw.centred_targets`).
3. **Start from the data levels** with `initial_state(eta, T, h_src=...)`.
4. **Keep the defaults** `thin_layer_coupling="mom6"`, `tracer_limiter="fct"`, `thin_layer_cutoff=0`. HYCOM-style
    blending (`thin_layer_cutoff > 0`) slows advancing fronts: hybrid lock exchange 1.05 → 0.70 U for any cutoff ≥ 0.01 m
    (docstring, `:992-998`).
5. More `ah` (the hybrid overflow completes at ν = 100 and 1000, Ilicak README item 7).

Remaining known problems are listed [below](#known-open-issues).

## Spurious currents over steep bathymetry (pressure-gradient error)

**Symptom.** A resting stratified ocean starts to move over slopes; currents of cm s⁻¹ along isobaths.

**Cause.** The σ-coordinate PGF is the difference of two large terms that do not cancel exactly (Haney 1991).
`check_domain` measures the risk with rx0 (Beckmann & Haidvogel) and rx1 (Haney); usual limits rx0 ≲ 0.2,
rx1 ≲ 3–7 (`diagnostics/domain_check.py:19-20`).

**Fixes**, by effect on the Beckmann & Haidvogel seamount (rx0 = 0.27, rx1 = 10.7; `validation/seamount/README.md`,
max|u| at day 10):

| Setting | max\|u\| day 10 |
|---|---|
| σ, two-term or Jacobian, no reference profile | 42.2 cm s⁻¹ |
| σ, Jacobian + `rho_ref="auto"` | 0.41 cm s⁻¹ |
| σ, `pgf="fv"` + `rho_ref="auto"` | 1.36 cm s⁻¹ |
| ALE z*, `pgf="fv"` + `rho_ref="auto"` | 0.10 cm s⁻¹ |
| ALE z*, `pgf="fv"`, no profile | 1.85 cm s⁻¹ |

1. Always set `rho_ref="auto"` for stratified runs over slopes (it removes the resting stratification from the PGF
    and from the K_h diffusion).
2. Prefer ALE z* (or hybrid) with `pgf="fv"` over steep slopes.
3. Smooth the bathymetry where rx1 is large (``smooth_bathymetry(depth, n_passes, mask, kernel="hann", n_skip)``
    (`core/grid/bathymetry.py:221`)) — at the cost of the real topography.

Open: in every configuration the seamount error grows exponentially, doubling every ~2 days after day ~4
(not a PGF effect; unclassified, `validation/seamount/README.md`), and the FV-vs-Jacobian comparison at t = 0 is an
`xfail` awaiting a decision (`tests/test_sigma_ale.py:116`; "FV 7.888e-8 vs jacobian 7.876e-8 m/s²").

## Reflections at open boundaries, or a boundary that drives the interior

**Symptoms.** Waves bounce back from an open edge; a band near the edge oscillates in phase; T drifts near an edge.

| Cause | Fix |
|---|---|
| no sponge: with Orlanski alone the mode-1 channel test has an interior error of 0.16 against 0.015 with a 10-cell sponge (σ); z* 0.16 / 0.016, hybrid 0.198 / 0.024 | add `sponge_cells` (10–30 cells in the validation) with `sponge_tau` (BayIW: `(1200, 60)` s) |
| sponge on a prescribed edge: a band relaxed toward the boundary profile copied along the normal forced 2 km in phase (160° phase error) | fixed: `"clamped"` edges get no sponge (`core/dynamics/sigma_baroclinic.py:1081`) |
| edge without data: its sponge used to relax toward another edge's profile | fixed: each point takes the exterior of its nearest open edge; edges without data keep the resting exterior (``owned_by_state`` (`core/dynamics/open_boundary.py:54`)) |
| time-varying exterior in a `"radiating"` edge with a sponge: the band is relaxed toward the edge profile copied along the normal, as MITgcm OBCS | open: a 3-D exterior over the whole band is needed (`validation/internal_wave/README.md`, "what pyOCEAN should improve" 2; BayIW nested runs, CLAUDE.md) |
| T drifting at the edge | `sponge_tracer=True` relaxes T toward `T_ext`; inflow T is `T_ext` (initial profile unless a provider gives T) |
| exterior data too shallow | `…: exterior data do not reach the bed of the boundary column (zw[-1] > −H)` (`core/dynamics/open_boundary.py:73`): extend the deepest data layer to the bed |
| barotropic mode reflected | Flather radiates toward η = 0, ū = 0 unless `OBData.eta`/`ubar` are given; for tides give them ([Tides](tidal-model.md)) |

Interior error numbers: module docstring (`core/dynamics/sigma_baroclinic.py:46-47`) and CLAUDE.md (ALE row);
phase-forcing findings: `validation/internal_wave/README.md`. ALE does not allow floodable (H ≤ 0) cells at an open
edge (`:1177-1179`). Theory: [Open boundaries](../theory/open-boundaries.md).

## Wetting-and-drying fronts

| Symptom | Cause | Fix / status |
|---|---|---|
| the planar Thacker oscillation loses ~1 % per period (CROCO 0.2 %) | a face that becomes wet starts from u = 0, a momentum sink at the shoreline | known; strict `xfail` `test_front_dissipation_as_low_as_croco` (`tests/test_wetting_drying.py:118-120`); extrapolating velocities to new wet faces made it worse (`validation/thacker/README.md`). Planned: Stelling–Duinmeijer fronts (roadmap stage B) |
| the shoreline lags one cell on the ebb | the front retains a thin film | `validation/tidal_flat/README.md`; within one cell of CROCO |
| choice of `d_min` | dry-face threshold | Thacker: 0.05 m gives η RMS 4.6 cm vs 9.7 cm with 0.01 m; tidal flat: 0.05 m moves the measured shoreline by 2 km only because "wet" is then defined differently (`validation/thacker/README.md`, `validation/tidal_flat/README.md`) |
| `cells with H ≤ 0 can dry: use regrid='sigma', or 'hybrid' with a shallow σ zone` | z* (and hybrid without a window) cannot represent a dry column | σ-Eulerian, ALE `"sigma"`, or hybrid with `d_sigma`, `d_hybrid > 0` (`core/dynamics/sigma_baroclinic.py:1169-1172`) |
| mass lost when drying | positivity guard clipped η | read `m.core.fs.last_clip_volume` (`core/dynamics/wetting_drying.py:302`); it "should not trigger" — a non-zero value means Δt is too long for the drying front |

Theory: [Wetting and drying](../theory/wetting-drying.md); validation: [Wetting and drying](../validation/wetting-drying.md).

## Slow runs

| Cause | Fix | Evidence |
|---|---|---|
| **one thread**: the kernels default to 1 thread | `NUMBA_NUM_THREADS=5` in the environment *before* importing, then `set_threads(5)` (``set_threads`` (`core/dynamics/sigma_baroclinic.py:166`)) | BayIW cut z*: 0.740 s/step on 1 thread, 0.304 on 5 (`validation/bayiw_cut/README.md`) |
| BLAS / OpenMP threads | set them to 1: they give no speed-up (SuperLU and NumPy are single-threaded) | CLAUDE.md, "Known issues" |
| first steps slow | Numba compiles the kernels on first use and caches them | the Ilicak runner times only after 5 warm-up steps (`validation/ilicak_benchmark/run_pyocean.py:22-23`) |
| direct free-surface solve in ALE | keep the ALE default `solver="cg"` (lagged-LU preconditioner) | ≈1 s → ≈0.1 s per step on BayIW 560 × 352 (docstring `:967-968`) |
| the mixing diagnostic | `mix_diagnostic=False` (ALE) when you do not need `mix_num` | BayIW z* 3.32 → 2.72 s/step, hybrid 1.58 → 1.44 (`validation/bayiw_baroclinic/README.md`) |
| PPM thickness flux | `thickness_flux="upwind"` (with few layers or for tests) | 28 % cheaper (docstring `:961-963`) |
| many layers | a hybrid grid needs fewer layers than z* | BayIW cut: hybrid 25 layers 0.160 s/step vs z* 50 layers 0.304 (`validation/bayiw_cut/README.md`) |
| several runs at once | on the development laptop the total number of compute threads over all runs must stay ≤ 5 (performance cores); throughput scales with independent runs, not with BLAS threads | CLAUDE.md, "Known issues" |
| memory bandwidth | the remap, advection, FV PGF and PPM kernels scale only 3.1–3.9× on 5 threads | `validation/bayiw_cut/README.md`; see [Performance](../architecture/performance.md) |

For comparison: the BayIW cut costs 9.1 s per simulated hour in z* (4.8 s hybrid) against 4.4 s for MITgcm on 5 MPI
processes (`validation/bayiw_cut/README.md`).

## Broken NumPy / Numba environment

**Symptom.** `import numba` (or the first pyOCEAN import) fails with an error about the NumPy version, or kernels
fail to compile after installing a package.

**Cause.** Numba pins NumPy. The working environment (checked 2026-10-14) is NumPy 1.26.4 with Numba 0.59.1, SciPy
1.13.1, xarray 2023.6.0, zarr 2.18.7, netCDF4 1.7.1. `pip install` of a package that requires NumPy 2 silently
upgrades it and breaks Numba. `environment.yml` does not pin NumPy (inconsistency reported to the maintainer).

**Fix.** Install new packages with `pip install --no-deps <pkg>==<version>` after checking their requirements, then
run `pip check`; never let pip upgrade NumPy. If it happened, reinstall the NumPy build that matches Numba. See
[Installation](../getting-started/installation.md).

## "numpy.ndarray size changed" warning

**Symptom.** `RuntimeWarning: numpy.ndarray size changed, may indicate binary incompatibility` when netCDF4 is imported.

**Cause.** A binary mismatch of the netCDF4 build in the base conda environment; harmless (CLAUDE.md, "Known issues").

**Fix.** Ignore it, or filter it, e.g. `warnings.filterwarnings("ignore", message="numpy.ndarray size changed")`; when
running tests with `-W error`, add `-W "ignore:numpy.ndarray size changed"`.

## Runtime warnings from sigma_wd.py:243

**Symptom.** In ALE runs with land (NaN) cells: `RuntimeWarning: invalid value encountered in divide` and
`divide by zero encountered in divide` at `core/dynamics/sigma_wd.py:243`, and `invalid value encountered in add` at
`:296`.

**Cause.** With `thin_layer_coupling="mom6"` (the ALE default), ``SigmaWetDry.column_step`` (`:237-243`) restores the
depth-mean balance with a correction divided by Σ w_k; on faces next to land the face column is empty (Σ w = 0), so
0/0 appears. Those faces are dry and `SigmaWetDry.step` masks them (`np.where(wet_u, …, 0)`, `:295-296`). Reproduced for
this manual: the warnings appear with a NaN row in the bathymetry and `"mom6"`, disappear with
`thin_layer_coupling="none"` or without land, and the state stays finite.

**Fix.** Harmless; filter it if it clutters the log
(`warnings.filterwarnings("ignore", category=RuntimeWarning, module="core.dynamics.sigma_wd")`). Do not switch the
coupling off to silence it.

## Which kinetic-energy scheme: `"gudonov"` or `"arakawa"`?

`"gudonov"` is MOM6's own spelling of its upwind (Godunov) kinetic-energy scheme (`KE_GUDONOV`,
`MOM6/src/core/MOM_CoriolisAdv.F90:126`), kept for compatibility; `"godunov"` raises `unknown ke_scheme` (`core/dynamics/sigma_baroclinic.py:1136-1137`).

| | `"arakawa"` (default since 2026-10-12) | `"gudonov"` |
|---|---|---|
| what | energy-conserving K at the cell centre | only the velocities entering a cell |
| good for | internal waves, tides | gravity-current fronts and bores |
| evidence | MITgcm `internal_wave`: mode-1 u 0.91–1.01 of MITgcm (Godunov 0.83–0.99) | ALE-σ lock exchange 0.98 U (Arakawa 0.91/0.84 U) |
| cost | damps nothing extra | damps progressive internal waves ~9 % per 5 km at dx = 100 m, first order in dx |

Sources: `validation/internal_wave/README.md`, docstring `core/dynamics/sigma_baroclinic.py:955-959`. Note that the
validation runners `validation/internal_wave/run_pyocean.py` and `case.build_model` still default to `"gudonov"`
(they predate the change of default); pass `--ke arakawa` to get the model default.

## Other frequent messages

| Message | Fix |
|---|---|
| `vertical_coordinate='ale' needs z0=None for now …` | ALE has no log-law drag yet; use `z0=None` (and `closure="constant"`) or the σ mode |
| `regrid='hybrid' needs regrid_params['rho_target']` | give `dz_top` and `rho_target` ([configuration reference](configuration-reference.md#regrid_params)) |
| UserWarning `hybrid z* floor (Σ dz_top = … m) stops above the deepest bed` | extend `dz_top` to the deepest bed |
| `h_src must be ≥ 0 and fill every wet column: Σ h_src = H + η` | clip the data levels at the bed (`np.minimum(zf, H)`), flip to bed first, and include η in the top level if η ≠ 0 |
| `restart incompatible with the model configuration — …` | build the model with the arguments of the run that wrote the file; the message lists every differing key |
| `restart has no reference density profile but the model uses rho_ref='auto'` | the restart came from a run without `rho_ref="auto"` |
| `unknown output field 'h'` | ALE thickness cannot go through `OutputWriter` yet; see [ALE output](output-restart-diagnostics.md#model-output-outputwriter) |
| `a 'clamped' open_boundary_mode needs open_boundary_state` | a Dirichlet edge needs the exterior to impose |
| `momentum='vector_invariant' needs vertical_coordinate='ale'` | σ keeps the flux form |
| `tracer_limiter='fct' needs vertical_coordinate='ale', tracer_time='ssprk3' and a WENO5 tracer_scheme` | FCT is ALE-only |
| `unknown advection scheme … (use 'centred' or 'upwind3')` at the first step | `adv` is checked only when used: use `"flux"`, `"upwind3"` or `"centred"` |
| `check_domain` reports isolated cells in a one-row channel | expected for `ny = 1`; ignore |

The complete list of messages is in the [configuration reference](configuration-reference.md#errors-raised-for-invalid-configurations).

## Things the model does not do (yet)

- **Salinity, a nonlinear EOS, surface heat/freshwater fluxes** in `SigmaBaroclinic`: one tracer T, linear EOS, wind
    stress only (`core/dynamics/sigma_baroclinic.py:55-58`, `:2084`). Use a density-equivalent T for realistic density.
- **Stratification-aware turbulence:** GLS has no buoyancy term (`physics/mixing/gls.py:14`); ALE only has constant
    viscosity and diffusivity. KPP exists only in the phase-1 z-level code.
- **Log-law drag in ALE** (`z0=None` required).
- **ALE-aware output metadata** (see above).
- **A non-resting initial velocity** through `initial_state` (set `st.u`, `st.v` by hand).

## Known open issues

From the status tables of CLAUDE.md and the validation READMEs (2026-10-14). Class (a) = pyOCEAN bug, (b) missing
physics, (c) numerical choice, (d) reference-model error, as defined in CLAUDE.md.

| Issue | Status | Source |
|---|---|---|
| Hybrid overflow at ν = 1 m² s⁻¹ stops at 20–23 h in 4 of 5 trajectories: ~98 layers of ~1 mm move as a slab next to the east wall and reach 50 m s⁻¹ | (a), open; needs its own design (PGF in vanished layers and their coupling) | `validation/ilicak_benchmark/README.md`, items 7 and 4d |
| Thin overflow faces carry up to 1.9× (ν = 1000) and 2.0× (ν = 100) the velocity of thick faces | (a), open; candidate: distribute the depth-mean correction δ ∝ visc_rem as MOM6 | Ilicak README, 4b |
| Hybrid internal wave at ν = 0.01: oscillation 0.50 vs 0.65–0.72 in the other models; a thin layer drains after day 120 | (a), open | Ilicak README, 4a |
| Hybrid lock-exchange fronts at ν = 0.1 slowed 2–5 % by the thin-layer coupling | open; try the coupling near the bed only (MOM6 default) | Ilicak README, 4c |
| FV vs Jacobian PGF over the seamount | `xfail` pending a decision (tie with the binned reference profile) | `tests/test_sigma_ale.py:116-119` |
| Seamount spurious current doubles every ~2 days in all configurations | unclassified, probably (a) or (c) | `validation/seamount/README.md` |
| Thacker front loses ~1 %/period vs CROCO 0.2 % | strict `xfail` | `tests/test_wetting_drying.py:118` |
| σ lock exchange mixes 1.3–1.5× more than CROCO (w noisier at grid scale) | open; candidates LF-AM3, barotropic coupling | `validation/lock_exchange/README.md` |
| RPE diagnostic inexact with a coarse hypsometry | (a) in the diagnostic; refine the hypsometry meanwhile | Ilicak README, item 12 |
| BayIW 3-D two-layer: phase lag 1.5–3 h at the shelf edge; linear hybrid in the shallow bay | open | CLAUDE.md, ALE row |
| Sponge band with a time-varying exterior is forced in phase (exterior copied along the normal) | open: needs a 3-D exterior | `validation/internal_wave/README.md` |
| Seamount largest stable Δt is half that of MITgcm and MOM6 σ | open (performance) | Ilicak README, 6 |
| Tide runners: restart/output wiring | open (stage A) | CLAUDE.md, stages table |

## References

- Beckmann, A. & Haidvogel, D.B. (1993). *J. Phys. Oceanogr.* 23, 1736–1753. **[verify]**
- Casulli, V. (1990). *J. Comput. Phys.* 86, 56–74.
- Haney, R.L. (1991). On the pressure gradient force over steep topography in sigma coordinate ocean models.
    *J. Phys. Oceanogr.* 21, 610–619. **[verify]**
- Hallberg, R. & Rhines, P. (1996). Buoyancy-driven circulation in an ocean basin with isopycnals intersecting the
    sloping boundary. *J. Phys. Oceanogr.* 26, 913–940. **[verify]**
- Ilicak, M., Adcroft, A.J., Griffies, S.M. & Hallberg, R.W. (2012). *Ocean Modell.* 45–46, 37–58.
    doi:10.1016/j.ocemod.2011.10.003
- Stelling, G.S. & Duinmeijer, S.P.A. (2003). A staggered conservative scheme for every Froude number in rapidly
    varied shallow water flows. *Int. J. Numer. Meth. Fluids* 43, 1329–1354. **[verify]**
- Thacker, W.C. (1981). Some exact solutions to the nonlinear shallow-water wave equations. *J. Fluid Mech.* 107,
    499–508.
