# Oceananigans.jl composition: reference for pyOCEAN N1

Source: `~/pyOCEAN_refs/Oceananigans/src/` at commit 8ab4604 (v0.117.0). All paths below are relative to `src/`. Abbreviation: `HFSM/` = `Models/HydrostaticFreeSurfaceModels/`.

Some names in the brief do not exist in this version. `compute_diffusivities!` is now `compute_closure_fields!`. `build_diffusivity_fields` is now `build_closure_fields`. The tendency launchers are `compute_hydrostatic_free_surface_G{u,v,c}!`, not `calculate_*`. Section 1 also shows the momentum tendencies are no longer computed inside the step. They are computed at the end of `update_state!`.

---

## 1. `HydrostaticFreeSurfaceModel`: construction pipeline

### Struct and arguments
- The struct has 20 fields: `architecture, grid, clock, advection, buoyancy, coriolis, free_surface, forcing, closure, particles, biogeochemistry, velocities, transport_velocities, tracers, pressure, closure_fields, timestepper, auxiliary_fields, vertical_coordinate, boundary_transport` (`HFSM/hydrostatic_free_surface_model.jl:51-74`). The struct is `mutable`, and every field is a type parameter, so a model's whole configuration is part of its type.
- Constructor keywords and defaults (`hydrostatic_free_surface_model.jl:145-163`):
  - `momentum_advection = VectorInvariant()`
  - `tracer_advection = Centered()`
  - `buoyancy = nothing`, `coriolis = nothing`, `closure = nothing`
  - `free_surface = default_free_surface(grid)`
  - `tracers = nothing`, `forcing = NamedTuple()`, `boundary_conditions = NamedTuple()`
  - `timestepper = :QuasiAdamsBashforth2`
  - `biogeochemistry = nothing`, `auxiliary_fields = NamedTuple()`
  - `vertical_coordinate = default_vertical_coordinate(grid)`
- **Default free surface** (`:78-82`): `ImplicitFreeSurface()` on a horizontally regular, static-z `RectilinearGrid` (`XYRegularStaticRG`, defined at `HFSM/HydrostaticFreeSurfaceModels.jl:28`). On any other grid it is `SplitExplicitFreeSurface(grid; cfl=0.7)`.
- **Default vertical coordinate** (`:43-49`): `ZStarCoordinate()` when `grid.z isa MutableVerticalDiscretization`, otherwise `ZCoordinate()`. Both are empty marker structs (`HydrostaticFreeSurfaceModels.jl:32-33`). The z* state itself (`σᶜᶜⁿ`, `σᶜᶜ⁻`, `ηⁿ`, `∂t_σ`) lives on the **grid**, not on the coordinate object (`z_star_coordinate.jl:35-36, 71`).

### Three-stage pipeline: validate → materialize → build
1. **Validation in the public constructor** (`:165-225`):
   - `validate_model_halo(grid, momentum_advection, tracer_advection, closure)` (`:168`; `Models/Models.jl:59-67`) checks that the grid halo covers the largest stencil required by any component (`inflate_halo_size`).
   - z* on a non-mutable grid raises an `ArgumentError` (`:170-174`).
   - Unknown timestepper symbol (`:176-183`). Supported: `(:QuasiAdamsBashforth2, :SplitRungeKutta2..5)` (`:76`).
   - Biogeochemistry may add tracers: `validate_biogeochemistry` (`:191-193`).
   - **Closure-required tracers are appended automatically**: `closure_required_tracers(closure)` (`:196-212`). For example, CATKE returns `(:e,)` (`TurbulenceClosures/.../catke_vertical_diffusivity.jl:156`) and k-ε returns `(:e, :ϵ)` (`tke_dissipation_vertical_diffusivity.jl:175`). If a user names one of them, the constructor raises an `ArgumentError`.
   - `validate_buoyancy(buoyancy, tracernames(tracers))` (`:214`) checks that `required_tracers(buoyancy)` ⊂ tracers (`BuoyancyFormulations/BuoyancyFormulations.jl:29-36`).
   - The tracer names and timestepper are then made type parameters (`Val(tracernames)`). The constructor calls `materialize_…` through `invokelatest`, which acts as a compilation barrier (`:218-225`).
2. **`materialize_hydrostatic_free_surface_model`** (`:228-294`):
   - Advection:
     - `validate_momentum_advection` rejects non-`VectorInvariant` schemes on `OrthogonalSphericalShellGrid` (`:424-429`).
     - `validate_tracer_advection` returns `(default, per-tracer NamedTuple)` (`Models/Models.jl:88-92`).
     - `with_tracers` builds one scheme per tracer (`:242`).
     - `adapt_advection_order` lowers the order where there are too few cells (`:245`).
     - `materialize_advection(scheme, grid)` fills settings that were deferred until the grid is known (`:249`).
   - `materialize_buoyancy(buoyancy, grid)` wraps a formulation in a `BuoyancyForce` (`:251`; `buoyancy_force.jl:104-105`).
   - **Boundary conditions** (`:253-282`), merged in three layers with precedence *user > embedded-in-user-fields > default*:
     - default `FieldBoundaryConditions()` for every field name (`:266-267`)
     - plus free-surface defaults (`:268-269`)
     - plus user/embedded BCs (`:273`)
   - `regularize_field_boundary_conditions(bcs, grid, field_names)` turns placeholder BCs into concrete ones from the field location and grid topology (`:274`; `BoundaryConditions/field_boundary_conditions.jl:294-316`).
   - `add_closure_specific_boundary_conditions` adds, for example, CATKE's surface TKE flux (`:278-282`).
   - Closure: `with_tracers(tracer_names, closure)` gives the closure per-tracer parameters (`:285`). `validate_closure` sorts tuples so CATKE comes first (`:288`; `catke_vertical_diffusivity.jl:164`).
3. **`build_hydrostatic_free_surface_model`** (`:296-355`). This stage allocates fields:
   - Velocities, tracers, `pressure = (; pHY′ = CenterField(grid))` (`:24, :305-307`).
   - `closure_fields = build_closure_fields(...)` (`:308`).
   - `validate_velocity_boundary_conditions` (`:310`): `w.top` must be `nothing` (`:413-417`).
   - `validate_free_surface(arch, fs)` (`:312`): an FFT implicit solver on a distributed architecture is an error (`:419-422`).
   - **`materialize_free_surface(fs, velocities, grid, bcs)`** (`:313`). The user passes an *unmaterialized* object, e.g. `ImplicitFreeSurface{Nothing}` with `displacement = nothing`. Materialization allocates η and builds the solver (`implicit_free_surface.jl:99-112`).
     - `:Default` picks FFT on regular grids and PCG otherwise (`:114-118`).
     - `SplitExplicitFreeSurface` has its own method (`SplitExplicitFreeSurfaces/split_explicit_free_surface.jl:263`).
   - `validate_immersed_boundary`: a free surface plus `GridFittedBoundary` is an error (`:32-41`).
   - **Implicit vertical solver**: `implicit_diffusion_solver(time_discretization(closure), grid)` (`:323`). The solver is also created when the advection scheme or a BC `needs_implicit_solver` (`:324-328`).
   - Tendency storage `Gⁿ`, `G⁻` (AB2 only; `hydrostatic_free_surface_field_tuples.jl:8-20`) and the RK cache `Ψ⁻` (`:357-377`) go into `TimeStepper(...)` (`:330-333`).
   - **Forcing**: `model_forcing(forcing, model_fields, prognostic_fields)` (`:340-341`; `Forcings/model_forcing.jl:42-63`).
     - An unknown key raises an `ArgumentError`.
     - A `Function` is wrapped as `ContinuousForcing{LX,LY,LZ}` at the field location.
     - Arrays and `FieldTimeSeries` become a `Forcing`.
     - A missing entry becomes `Returns(0)`.
   - `transport_velocities` are separate copies of u, v, w (`:342, :382-389`).
   - Finally `update_state!(model)` (`:352`). A model is fully diagnosed and its tendencies are ready **as soon as it is constructed**.

**Pattern.** Each component follows a three-function protocol, written as free functions that dispatch on the component's type:

- `validate_X(x, context)` → error or `x`
- `materialize_X(x, grid, …)` → a new object with allocated storage
- default-constructor helpers

Users pass lightweight "specs". The model returns concrete objects.

---

## 2. Time step: exact order

### Simulation → model
`time_step!(sim)` (`Simulations/run.jl:216-269`) runs these steps in order:
1. `initialize!(sim)` on the first call (`:221`).
2. Aligned Δt (`:223-227`).
3. `time_step!(model, Δt, callbacks=model_callbacks)` (`:241-242`).
4. Diagnostics (`:246-248`).
5. `TimeStepCallsite` callbacks (`:250-252`).
6. Output writers (`:254-256`).

### QuasiAdamsBashforth2 (default)
`time_step!` (`TimeSteppers/quasi_adams_bashforth_2.jl:90-128`):
1. Take an Euler step if Δt changed or this is the first step: `euler = euler | (Δt != clock.last_Δt)` (`:103`), and set `χ = -0.5` (`:110`).
2. `maybe_prepare_first_time_step!` at iteration 0 calls `reconcile_state!` and then `update_state!` (`TimeSteppers/TimeSteppers.jl:46-53`).
3. **`ab2_step!(model, Δt, callbacks)`** (`:114`). It dispatches on the free surface type (`HFSM/hydrostatic_free_surface_ab2_step.jl:18-19`):
   - **Explicit or split-explicit FS** (`:37-73`):
     1. `update_transport_velocities!` copies the pre-step u, v (and w for the explicit FS) into the transport velocities (`:42`; `hydrostatic_free_surface_model.jl:395-405`).
     2. `compute_momentum_flux_bcs!` adds boundary flux divergences to `Gⁿ.u`, `Gⁿ.v` (`:44`; `compute_hydrostatic_flux_bcs.jl:15-25`).
     3. `ab2_step_velocities!` (`:45`; body `:163-174`): explicit AB2 `u += Δt((3/2+χ)Gⁿ − (1/2+χ)G⁻)`. Then the **implicit vertical diffusion** `implicit_step!`, wrapped in `add_barotropic_acceleration!(+Δt)` / `(−Δt)` so that implicit bottom-drag BCs act on a velocity that has already felt −g∇η (`deferred_barotropic_acceleration.jl:4-23`).
     4. `mask_immersed_horizontal_velocities!` (`:46`).
     5. `compute_free_surface_tendency!`, then `step_free_surface!` (`:50-51`):
        - Split-explicit: the depth-integrated slow tendency `GU, GV` (`SplitExplicitFreeSurfaces/compute_slow_tendencies.jl:24`), then barotropic substepping (`step_split_explicit_free_surface.jl:236-295`).
        - Explicit: AB2 on η (`explicit_free_surface.jl:70-74`).
     6. `compute_transport_velocities!` (`:53`).
     7. `fill_halo_regions!((u, v))` (`:57`).
     8. `compute_tracer_tendencies!`, then the `TracerTendencyCallsite` callbacks (`:60-61`).
     9. `ab2_step_grid!`, the z* update (`:65`).
     10. `correct_barotropic_mode!` (`:68`). For split-explicit, the depth mean of u is reset to the barotropic U (`barotropic_pressure_correction.jl:55-61`).
     11. `ab2_step_tracers!` (`:69`): σ-weighted AB2 `θ = (σ⁻θ + Δt G)/σⁿ` (`:268-282`), then `implicit_step!` per tracer (`:250-259`). CATKE's `e` and k-ε's `e, ϵ` are skipped here because their closures step them (`:233-236`).
   - **Implicit FS (predictor–corrector)** (`:89-134`):
     1. `update_transport_velocities!`
     2. `compute_momentum_flux_bcs!`
     3. `ab2_step_velocities!` gives the predictor u*
     4. `enforce_targeted_open_boundary_transport!` (`:106`)
     5. `step_free_surface!`: mask, fill halos, build the RHS from ∇·∫u* dz, `solve!`, `fill_halo_regions!(η)` (`implicit_free_surface.jl:129-155`)
     6. `correct_barotropic_mode!`: `u -= gΔt ∂xη` (`barotropic_pressure_correction.jl:35-46, 63-70`)
     7. Fill the u, v halos
     8. `compute_transport_velocities!`, which also recomputes w̃ through `update_vertical_velocities!` (`implicit_free_surface.jl:168-174`)
     9. `compute_tracer_tendencies!`
     10. `ab2_step_grid!`
     11. `ab2_step_tracers!`
4. `step_lagrangian_particles!` and `cache_previous_tendencies!` (G⁻ ← Gⁿ) (`quasi_adams_bashforth_2.jl:115-116`).
5. `tick!(clock, Δt)` (`:118`).
6. **`step_closure_prognostics!`** steps the closure's own prognostic fields, e.g. CATKE e (`:120`; `update_hydrostatic_free_surface_model_state.jl:106-110`).
7. **`update_state!(model, callbacks)`** (`:121`).

### SplitRungeKutta3 (`timestepper = :SplitRungeKutta3`)
- Coefficients `β = (3, 2, 1)` (`TimeSteppers/split_runge_kutta.jl:112-117`). Stage m advances from the cached state Ψ⁻ with `Δτ = Δt/βᵐ`.
- `time_step!` (`:167-218`):
  1. `maybe_prepare_first_time_step!` (`:251-259`)
  2. `cache_current_fields!` (`:173`)
  3. Precompute the stage times (`:178-181`)
  4. Loop over stages (`:187-210`): `rk_substep!` → `step_closure_prognostics!` → set `clock.time` to the stage time → `update_state!`
  5. Particles, then `iteration += 1`
- `rk_substep!` (`HFSM/hydrostatic_free_surface_rk_step.jl:33-63` for explicit or split-explicit, `:78-122` for implicit) has **the same order as AB2**. Only the update kernel differs: `u = Ψ⁻ + Δτ G` (`:152-176`). The implicit FS resets η to `Ψ⁻.η` before each solve (`implicit_free_surface.jl:157-161`). The RK stepper is self-starting and checkpoints nothing (`split_runge_kutta.jl:265-267`).

### `update_state!` (`HFSM/update_hydrostatic_free_surface_model_state.jl:44-93`)
1. `mask_immersed_field!(tracers)`, `update_model_field_time_series!` (forcing/BC time series), `compute_forcing!`, `update_boundary_conditions!` (`:48-53`).
2. `fill_halo_regions!(tracers; async=true)` (`:60`).
3. `compute_buoyancy_gradients!` (`:67`).
4. **`update_vertical_velocities!`** (`:68`):
   - z*: `update_grid_vertical_velocity!` (∂t_σ = −∇·U/H) (`z_star_coordinate.jl:123`).
   - `compute_w_from_continuity!`: a bottom-up integral `wᵏ⁺¹ = wᵏ − (δ_h·(A u) / Az + Δr ∂t_σ)` (`compute_w_from_continuity.jl:18-22, 52-53, 74-93`).
5. **`update_hydrostatic_pressure!(pHY′, …, buoyancy, tracers)`**: a top-down integral of `−z·g·b · Δz` (`Models/NonhydrostaticModels/update_hydrostatic_pressure.jl:11-21, 35-36`).
6. **`compute_closure_fields!(closure_fields, closure, model)`** (`:70`). The fallback is a no-op (`TurbulenceClosures/TurbulenceClosures.jl:91`).
7. Fill local halos of closure fields and pHY′ (`:75-76`).
8. `UpdateStateCallsite` callbacks (`:78`), then `update_biogeochemical_state!` (`:80`).
9. `update_adaptive_timesteps!`, then **`compute_momentum_tendencies!`** (`:84`; `compute_hydrostatic_free_surface_tendencies.jl:23-38`), then the `TendencyCallsite{MomentumTendency}` callbacks. Last, `update_bounds_preserving_limiters!`.

**Consequence.** The *momentum* tendency for step n+1 is computed at the end of step n, in `update_state!`. *Tracer* tendencies are computed mid-step, after the free surface step, using the updated transport velocities. The z* grid update sits between tracer tendencies and the tracer update. The barotropic correction follows the grid update in RK because it integrates over the column (`rk_step.jl:55-58`).

### Halo/BC application points
- Tracers: start of `update_state!`.
- u, v: after the free surface step (`ab2_step.jl:57, 117`).
- η: inside `step_free_surface!`.
- Closure fields and pHY′: local halos only, after they are computed (`update_state.jl:75-76`).
- Flux BCs are added to tendencies separately:
  - `compute_momentum_flux_bcs!` at the start of the step
  - `compute_tracer_flux_bcs!` inside `compute_tracer_tendencies!` (`compute_hydrostatic_free_surface_tendencies.jl:79`)
- Implicit-flux BCs (e.g. implicit drag) go into the tridiagonal solve (`vertically_implicit_diffusion_solver.jl:289-312`).
- `reconcile_state!` fills all halos and reconciles the FS and z* after `set!` or a restart (`hydrostatic_free_surface_model.jl:431-437`).

---

## 3. Tendency assembly via dispatch

### Kernel function
```julia
@inline function hydrostatic_free_surface_u_velocity_tendency(i, j, k, grid,
    advection, coriolis, closure, u_immersed_bc, velocities, free_surface, tracers,
    buoyancy, closure_fields, hydrostatic_pressure_anomaly, auxiliary_fields, ztype, clock, forcing)
```
(`HFSM/hydrostatic_free_surface_tendency_kernel_functions.jl:26-40`). It returns a single sum (`:46-53`):
```julia
- U_dot_∇u(i,j,k,grid,advection,velocities)
- U_dot_∇u_hydrostatic_metric(...)
- explicit_barotropic_pressure_x_gradient(i,j,k,grid,free_surface)
- x_f_cross_U(i,j,k,grid,coriolis,velocities)
- ∂xᶠᶜᶜ(i,j,k,grid,hydrostatic_pressure_anomaly)
- ∂ⱼ_τ₁ⱼ(i,j,k,grid,closure,closure_fields,clock,model_fields,buoyancy)
- immersed_∂ⱼ_τ₁ⱼ(...)
+ forcing(i,j,k,grid,clock,model_fields)
```
The tracer tendency (`:124-156`) is:

`− div_Uc(advection, total_velocities, c) − ∇_dot_qᶜ(closure, …) − immersed_∇_dot_qᶜ + bgc + forcing`

`total_velocities` = transport velocities + BGC sinking + closure auxiliary (GM bolus) velocity + advective forcing (`:105-110`).

These kernel functions are launched pointwise by `@kernel compute_hydrostatic_free_surface_Gu!` and `Gc!` (`compute_hydrostatic_free_surface_tendencies.jl:215-258`). The launchers pass every component explicitly (`:153-199`, `:119-146`).

### How each component switches its term on or off
Every term is a function of `(i, j, k, grid, component, …)`. A `nothing` or "absent" component dispatches to zero:
- **Coriolis**: `x_f_cross_U(i,j,k,grid,::Nothing,U) = zero(FT)` (`Coriolis/no_rotation.jl:7`). Each scheme (energy- or enstrophy-conserving, etc.) has its own method (`Coriolis/coriolis_schemes.jl:67-137`).
- **Advection**: generic flux form `div_Uc = V⁻¹(δx(F_x) + δy(F_y) + δz(F_z))` (`Advection/tracer_advection_operators.jl:31-35`). It is zero for `::Nothing` advection, `ZeroU` or `ZeroField` (`:38-45`). The scheme only supplies the reconstruction inside `_advective_tracer_flux_*`.
- **Closure**: `∂ⱼ_τ₁ⱼ` is a generic flux divergence over `viscous_flux_ux/uy/uz(…, disc, closure, K, …)` (`TurbulenceClosures/closure_kernel_operators.jl:20-46`). It is zero for `::Nothing` (`nothing_closure.jl:2`). **Implicit vertical flux**: with `VerticallyImplicitTimeDiscretization` the explicit `diffusive_flux_z`/`viscous_flux_wz` return zero (`abstract_scalar_diffusivity_closure.jl:282-283`), and the same coefficients are used by `implicit_step!`. The time discretization is a type parameter of the closure (`AbstractTurbulenceClosure{TimeDiscretization, RequiredHalo}`, `TurbulenceClosures.jl:85`).
- **Free surface**: `explicit_barotropic_pressure_x_gradient` is `g ∂xη` for `ExplicitFreeSurface` (`explicit_free_surface.jl:56-60`) and `0` for implicit or split-explicit (`implicit_free_surface.jl:120`; `SplitExplicitFreeSurfaces.jl:47`). For those, the barotropic pressure enters through `correct_barotropic_mode!`.
- **Buoyancy**: buoyancy appears only through the precomputed `pHY′` and through closures. `update_hydrostatic_pressure!` dispatches to a no-op for `pHY′ === nothing` (`update_hydrostatic_pressure.jl:38`).
- **Forcing**: each field gets a callable (`ContinuousForcing`, `DiscreteForcing`, `Relaxation`, or `Returns(0)`).

### Closure field declaration
Each closure implements up to six functions:

| Function | Purpose | Fallback / reference |
|---|---|---|
| `build_closure_fields(grid, clock, tracer_names, bcs, closure) → NamedTuple` | allocate the closure's fields | `nothing` (`closure_fields.jl:20`); tuples of closures map over members (`:26`). Example: Ri-based diffusivity allocates `(; κc, κu, Ri, previous_compute_time)` at (C,C,F) (`ri_based_vertical_diffusivity.jl:189-195`) |
| `compute_closure_fields!(K, closure, model; parameters)` | compute the fields | (`ri_based_vertical_diffusivity.jl:203`) |
| `viscosity(closure, K)` / `diffusivity(closure, K, id)` | accessors used by the flux operators | (`:183-184`) |
| `step_closure_prognostics!` | step prognostic closure fields | `TurbulenceClosures.jl:94` |
| `initialize_closure_fields!` | initialize them | `TurbulenceClosures.jl:104-114` |
| `closure_required_tracers` | name the tracers the closure needs | `TurbulenceClosures.jl:118-131` |

The required halo size is the type parameter `B` (`:138-140`).

### Tracer and buoyancy declaration
Tracers are a tuple of symbols, or a NamedTuple of preallocated fields (docstring `:126-127`), normalized by `tupleit`. Buoyancy declares `required_tracers`:
- `SeawaterBuoyancy` → `(:T, :S)`, or `(:T,)` or `(:S,)` when the other is a constant (`BuoyancyFormulations/seawater_buoyancy.jl:18-20`)
- `BuoyancyTracer` → `(:b,)` (`buoyancy_tracer.jl:10`)
- `nothing` → `()` (`no_buoyancy.jl:3`)

`validate_buoyancy` raises `"$(req_tracers) must be among the list of tracers to use …"` (`BuoyancyFormulations.jl:29-36`). Note that buoyancy requirements are *validated*, while closure requirements are *appended automatically*.

---

## 4. Simulation layer

- **`Simulation`** (`Simulations/simulation.jl:62-78`) is a mutable struct with these fields:
  - `model`, `Δt`, `stop_iteration`, `stop_time`, `wall_time_limit`
  - `diagnostics`, `output_writers`, `callbacks` (each an `OrderedDict{Symbol,…}`, `:144-146`)
  - `run_wall_time`, `align_time_step`, `running`, `initialized`, `verbose`, `minimum_relative_step`, `automatic_differentiation`

  Built-in callbacks are `stop_time_exceeded`, `stop_iteration_exceeded`, `wall_time_limit_exceeded` (`:148-150`) and a default `nan_checker` every 100 iterations (`:152-155`). It warns when the run would never stop (`:137-140`).
- **`run!(sim; pickup, checkpoint_at_end)`** (`run.jl:151-187`): optional pickup via `set!(sim; checkpoint=…)`, then `while sim.running; time_step!(sim); end`, then an optional final checkpoint (skipped after a NaN), then `finalize!` on each callback. `initialize!(sim)` (`:332-398`) initializes the model only when `iteration == 0`, calls `update_state!`, adds output dependencies (time averages are themselves callbacks), initializes the schedules, and writes the initial output.
- **Callbacks**: `Callback(func, schedule=IterationInterval(1); parameters, callsite=TimeStepCallsite())` (`callback.jl:7-75`); `add_callback!(sim, func, schedule; name)` (`:171-177`). Callsites (`Oceananigans.jl:205-219`):
  - `TimeStepCallsite` runs after the step.
  - `UpdateStateCallsite` runs inside `update_state!`.
  - `TendencyCallsite{MomentumTendency}` and `TracerTendencyCallsite` run after the tendencies are computed, so a user can modify G.

  Only model-callsite callbacks are passed into `time_step!(model)` (`run.jl:206, 241-242`).
- **`TimeStepWizard`** (`time_step_wizard.jl:5-91`) has fields `cfl=0.2, diffusive_cfl=Inf, max_change=1.1, min_change=0.5, max_Δt, min_Δt` and pluggable `cell_advection_timescale(model)` and `cell_diffusion_timescale(model)`. `new_time_step` takes the min of the advective and diffusive limits, clamps it by the change ratios and by max/min Δt, then does an `all_reduce(min)` (`:99-113`). It is a callable struct that sets `simulation.Δt` (`:115-116`) and is registered as a `Callback` (`conjure_time_step_wizard!`, `:124-128`).
- **`NaNChecker`** (`Diagnostics/nan_checker.jl:4-70`): `fields`, `erroring::Bool`, `nan_detected`. When it finds a NaN it sets `simulation.running = false` and either logs or throws. By default it checks only the first prognostic field (`Models/Models.jl:223-234`).
- **Output writers**:
  - `JLD2Writer(model, outputs; filename, schedule, dir, indices, with_halos, array_type=Array{Float32}, file_splitting, overwrite_files, init, including, verbose, part)` (`OutputWriters/jld2_writer.jl:30-40, 179`)
  - `NetCDFWriter(model, outputs; kw...)` (`netcdf_writer.jl:326`)
  - `zarr_writer.jl`

  `outputs` is a NamedTuple of Fields, AbstractOperations, reductions or functions. `schedule` can be `TimeInterval`, `IterationInterval` or `AveragedTimeInterval(…, window)`. The writer is called when `writer.schedule(model)` is true (`run.jl:254-256`).
- **`Checkpointer(model; schedule, dir, prefix, overwrite_files, verbose, cleanup)`** (`checkpointer.jl:9-66`) writes `prognostic_state(simulation)` (`:191-197`). That state is a recursive NamedTuple of each component's `prognostic_state`:
  - model level: `hydrostatic_free_surface_model.jl:455-466`, with clock, velocities, tracers, closure_fields, timestepper (G⁻), free_surface, coriolis, auxiliary_fields, and vertical_coordinate (`z_star_coordinate.jl:222-235`)
  - restore: `restore_prognostic_state!` (`:468-480`)

  Each component owns its own restart contract, and the fallback is `nothing` (`checkpointer.jl:164-167, 246-250`).
- **Diagnostics and budgets**:
  - `AbstractOperation <: AbstractField` (`AbstractOperations/AbstractOperations.jl:28`). Unary, binary and multiary ops and derivatives build lazy expression trees on the grid.
  - `KernelFunctionOperation{LX,LY,LZ}(kernel_function, grid, args...)` (`kernel_function_operation.jl:78-88`) wraps **any model kernel function**, including the tendency functions themselves. For example, `ζ₃ᶠᶠᶜ` is wrapped this way (`:30-60`). This is how term-by-term budgets are built: pass the same pointwise term function (e.g. `div_Uc` with model components) to a KFO.
  - `Average(field; dims)` and `Integral(field; dims)` are `Scan` reductions (`metric_field_reductions.jl:30, 62, 141`). `Field(op)` makes a computed field updated with `compute!` (`computed_field.jl:80`). Auxiliary fields are `compute!`d in the model (`update_state.jl:16`).

---

## 5. Validation / error messages (examples)

- Halo too small: `"The grid halo $user_halo must be at least equal to $required_halo. Note that an ImmersedBoundaryGrid requires an extra halo point…"` (`Models/Models.jl:64-66`).
- z* without a mutable grid: `"The grid … does not support ZStarCoordinate. z must be a MutableVerticalDiscretization…"` (`hydrostatic_free_surface_model.jl:170-174`).
- Bad timestepper: `"timestepper = :X is not supported. Supported timesteppers are: … You can also construct your own TimeStepper"` (`:176-183`).
- Tracer name clash: `"The tracer names … overlap with the closure auxiliary tracer names … cannot be specified explicitly"` (`:200-205`).
- Free surface plus GridFittedBoundary: `"…does not support GridFittedBoundary. Use … GridFittedBottom or PartialCellBottom…"` (`:35-41`).
- Missing buoyancy tracers: `"(:T, :S) must be among the list of tracers to use SeawaterBuoyancy"` (`BuoyancyFormulations.jl:32-33`, `error`, not `ArgumentError`).
- Forcing for a non-prognostic name: `"Invalid forcing: forcing contains an entry for $name, but $name is not a prognostic field! The prognostic fields are …"` (`Forcings/model_forcing.jl:46-51`).
- Closure discretization: `"VerticallyImplicitTimeDiscretization is only supported for VerticalFormulation or ThreeDimensionalFormulation"` (`scalar_diffusivity.jl:124-126`).
- Split-explicit: `"The grid is a required positional argument to SplitExplicitFreeSurface when cfl is specified…"` (`split_explicit_free_surface.jl:189`). Also, variable substepping is not supported on some grids (`:268`).
- `w.top` must be `nothing` (`hydrostatic_free_surface_model.jl:414`). Unsupported momentum advection on a spherical shell (`:429`). FFT solver on a distributed architecture (`:421`). FFT on an irregular grid (`fft_based_implicit_free_surface_solver.jl:16`).
- `set!` with an unknown name (`set_hydrostatic_free_surface_model.jl:68`). Bad pickup (`run.jl:321`). Δt = 0 gives a warning (`quasi_adams_bashforth_2.jl:93`).

Style: the message names the offending value, lists the valid alternatives, and suggests a fix. Most validation runs at construction, and some runs at materialization once the grid is known.

---

## 6. Lessons for pyOCEAN N1

### Translates well
1. **The model is a container of components plus one fixed-order driver.** Physics ordering lives in exactly one function per time scheme (`hydrostatic_ab2_step!`, `rk_substep!`), and the components never call each other. For N1: a `step(state, components, dt)` with the order written out as in §2 and each step a named call, so a bit-for-bit comparison against `SigmaBaroclinic` can be checked step by step.
2. **Spec → validate → materialize.** Users pass light, immutable config objects such as `ImplicitFreeSurface(solver_method=…)` with no arrays. The model then calls `validate(spec, ctx)` and `materialize(spec, grid, bcs) → component` with arrays. In Python this maps to frozen `@dataclass` specs plus a `build(grid, ctx)` method that returns a runtime object. It also gives a natural place for pre-flight checks.
3. **Components declare their requirements.**
   - `required_tracers(buoyancy)` → validated.
   - `closure_required_tracers(closure)` → appended automatically.
   - `build_closure_fields(...)` → the closure owns its arrays.
   - `required_halo_size` → the model checks the halo once.
   - `prognostic_state` / `restore_prognostic_state!` → each component owns its restart piece.

   As Python protocols (`typing.Protocol`) these become methods such as `required_tracers() -> tuple[str, ...]`, `allocate(grid) -> dict[str, np.ndarray]`, `update(state, model)`, `state_dict()` and `load_state_dict()`. This fits pyOCEAN's existing bit-for-bit restart and the single variable registry: each component registers its variables (name, dims, location, units, CF name).
4. **"Absent component = `None` = zero term."** Every term is `term(component, …)` with a `nothing` method returning 0. In Python, use null-object components (`NoCoriolis.tendency(...) -> 0`), or have the driver skip `None`. Avoid scattered `if self.coriolis:` branches.
5. **Separating diagnose from step.** `update_state!` (halos → w from continuity → pressure → closure fields → momentum tendencies) is distinct from the stepping. Callsite hooks (`UpdateState`, `Tendency`, `TracerTendency`) give users and budget diagnostics well-defined access points. pyOCEAN can expose the same named hook points.
6. **Term functions reused for diagnostics.** Because each tendency term is a standalone function, budgets reuse the exact model code (via `KernelFunctionOperation`). For pyOCEAN, write each term as a pure function `term(state, grid, params) -> array` (Numba kernel inside). The driver sums them, and the budget diagnostic calls the same functions. This is also the JAX-ready form: pure functions of (state, params) with no hidden mutation.
7. **Simulation as an event loop.**
   - Ordered dicts of named callbacks, diagnostics and writers, each with a `schedule(model) -> bool`.
   - Stop conditions, NaN checking and the Δt wizard are themselves callbacks.
   - `run!` is just `while running: time_step!`.

   This is straightforward to copy in Python. pyOCEAN's blow-up post-mortem fits as a NaN/limit callback.
8. **Clear error messages** that name the value, the valid options and a fix.

### Avoid or adapt
1. **Multiple dispatch on (component × grid × timestepper) types.** Examples: `validate_momentum_advection` on 6 grid types (`:424-429`), and `materialize_free_surface` × grid × velocity type (`single_column_model_mode.jl:30-35`, `prescribed_hydrostatic_velocity_fields.jl:143-146`). In Python this becomes `isinstance` ladders or `functools.singledispatch`, which dispatches only on the first argument. Prefer methods on the component (`fs.step(model, dt)`) and keep cross-type rules in one explicit validation table.
2. **Pointwise `@inline` term functions summed inside one GPU kernel.** In Julia these fuse at compile time. In NumPy/Numba, per-point Python calls are prohibitively slow. Keep **array-level** term functions, with Numba kernels inside a term, and fuse only hot paths by hand (as pyOCEAN already does for WENO5/FCT).
3. **Type-parameterized everything plus `invokelatest` compile barriers** (`hydrostatic_free_surface_model.jl:218-225, 292`). These exist only to manage Julia compile latency and type inference. There is no Python equivalent and none is needed. The lesson is the cost: model construction compiles a specialized step. Numba has a smaller version of this (JIT on first call); cache it (`cache=True`) and warm it up in pre-flight.
4. **Mutable state hidden in the grid** (z* σ, ηⁿ, ∂t_σ live in `grid.z`). This is convenient for Oceananigans' operators but hurts transparency and JAX purity. pyOCEAN already has prognostic `h`. Keep the layer geometry in the state (`SigmaState`), not in the grid.
5. **Magic auto-appended tracers** are convenient but surprising. Mirror them, but log them in the pre-flight report.
6. **Driver variants per free surface.** There are two near-duplicate step functions (explicit vs implicit FS, `ab2_step.jl:37-73` vs `:89-134`, and the same for RK). In Python, one driver with explicit `fs.predictor/solve/correct` hooks avoids the duplication. Choose the order once and document it, so that ALE regrid/remap, the outflow limiter and FCT keep their fixed positions.
7. **Momentum tendency computed at the end of the previous step (in `update_state!`).** This is efficient but makes "tendency at time n" depend on the restart path (Oceananigans skips the recomputation after a restore, `run.jl:348-351`). For bit-for-bit restarts, either store Gⁿ or recompute deterministically. Document the choice.
