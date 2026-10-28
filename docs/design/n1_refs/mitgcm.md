# MITgcm time-step composition: reference for pyOCEAN N1

Source: `~/pyOCEAN_refs/MITgcm`, git 27256b2 (2026-10-07). Paths are relative to that root. `src/` = `model/src/`, `inc/` = `model/inc/`. All line numbers are from that checkout. This is a reference implementation, not ground truth.

## 1. One time step

### 1.0 Outer structure
- `the_model_main.F:620` calls `INITIALISE_FIXED`, then `:711/:747` calls `THE_MAIN_LOOP`. `the_main_loop.F:369` calls `INITIALISE_VARIA`, and `:681` calls `MAIN_DO_LOOP`, which calls `FORWARD_STEP(iloop, …)` (`main_do_loop.F:201`).
- Everything in one step happens in `src/forward_step.F`. Time is advanced in the middle of the step, right after `DYNAMICS` (`forward_step.F:806-807`: `myIter = nIter0+iLoop`, `myTime = startTime+deltaTClock*iLoop`). Everything after that point (the solver, corrections, staggered thermodynamics, I/O, pickup) already sees time n+1.

### 1.1 `FORWARD_STEP` call order (`src/forward_step.F`)
1. Nonlinear free surface / r* reset of geometry: `RESET_NLFS_VARS`, `UPDATE_R_STAR(.FALSE.)`, `UPDATE_SURF_DR` (`:467-484`, under `NONLIN_FRSURF`).
2. `PTRACERS_SWITCH_ONOFF` (`:501`). `DIAGNOSTICS_SWITCH_ONOFF` and then `DO_STATEVARS_DIAGS(seqFlag=0)` (`:508-511`). This is the first half of the TOTxTEND diagnostics (§4).
3. **`LOAD_FIELDS_DRIVER`** (`:540`), which reads and interpolates forcing: `EXF_GETFORCING` (`load_fields_driver.F:214`, bulk formulae → fu, fv, Qnet, EmPmR…), `EXTERNAL_FIELDS_LOAD` (`:234`), `RBCS_FIELDS_LOAD` (`:249`), `GCHEM_FIELDS_LOAD` (`:243`). Then `BULKF_FORCING`, `CHEAPAML`, `CTRL_MAP_FORCING`, coupler exchange, `EBM` (`forward_step.F:551-608`).
4. **`DO_ATMOSPHERIC_PHYS`** (`:627`), then **`DO_OCEANIC_PHYS`** (`:657`). Details in §1.2.
5. `GCHEM_CALC_TENDENCY` (`:690`) and LONGSTEP options (`:700-723`).
6. **If `.NOT.staggerTimeStep`: `THERMODYNAMICS`** (`:725-736`). This is the synchronous ordering: tracers are advanced with u^n.
7. **`DYNAMICS`** if `momStepping` (`:786-795`). It produces the provisional velocity u* in gU/gV (§1.4).
8. Time counter advance (`:806-807`).
9. `UPDATE_R_STAR(.TRUE.)` / `UPDATE_SIGMA` / `UPDATE_SURF_DR(.TRUE.)` (`:832-854`), then `UPDATE_CG2D` for the nonlinear free surface (`:866-871`). Optional Shapiro or zonal filters on gU/gV (`:880-891`).
10. **`SOLVE_FOR_PRESSURE`** (`:904`). It builds the RHS with `CALC_DIV_GHAT` (`solve_for_pressure.F:183`), adds the freshwater/exactConserv source terms, masks OBCS points (`:~249`, `cg2d_b*maskInC`), and calls `CG2D`, `CG2D_SR` or `CG2D_NSA` (`:282-308`). With `nonHydrostatic`, it also does `CG3D` (`:399-455`).
11. **`MOMENTUM_CORRECTION_STEP`** (`:914`). In `momentum_correction_step.F`: `CALC_GRAD_PHI_SURF` (`:80`), `CORRECTION_STEP` (`:87`, which computes `uVel = gU − Δt·∇φ_s^{n+1}` at `correction_step.F:215`), `OBCS_APPLY_UV(uVel,vVel)` (`:95`), filters (`:110,119`).
12. **`INTEGR_CONTINUITY`** if `calc_wVelocity` (`:928`). In `integr_continuity.F`: barotropic divergence → `dEtaHdt`, then `etaN` update (`:208`), `INTEGRATE_FOR_W` (`:270`), `OBCS_APPLY_W` (`:299`), `UPDATE_ETAH` (`:331`).
13. `CALC_R_STAR` / `CALC_SURF_DR` with the new η (`:949/959`).
14. **If `staggerTimeStep`**: `DO_STAGGER_FIELDS_EXCHANGES`, `DO_STATEVARS_DIAGS(seqFlag=1)`, then **`THERMODYNAMICS`** (`:976-1009`). Here tracers are advanced with u^{n+1} and w^{n+1}.
15. **`TRACERS_CORRECTION_STEP`** (`:1025`). In `tracers_correction_step.F`: filters (`:73-85`), `SALT_FILL` (`:94`), convective adjustment `CONVECTIVE_ADJUSTMENT` / `OPPS_INTERFACE` (`:109-116`).
16. LONGSTEP and `GCHEM_FORCING_SEP` (`:1029-1083`). `DO_FIELDS_BLOCKING_EXCHANGES` (`:1093`, halo exchange of the final state). `DO_STATEVARS_DIAGS(seqFlag=2)` (`:1110`, second half of TOTxTEND).
17. Floats (`FLT_MAIN :1133`), nesting, `MONITOR` (`:1154`), cost, **`DO_THE_MODEL_IO`** (`:1182`), `PTRACERS_RESET` (`:1189`), **`DO_WRITE_PICKUP`** (`:1196`).

### 1.2 `DO_OCEANIC_PHYS` (`src/do_oceanic_phys.F`): forcing and closures, no stepping
- Runtime switches: `calcKPP = useKPP`, `calcGMRedi = useGMRedi`, `calcConvect = ivdc_kappa≠0` (`:268-274`).
- **`OBCS_CALC(myTime+Δt, myIter+1, u,v,w,θ,S)`** (`:322`). It computes the future boundary values (prescribed, Orlanski `ORLANSKI_EAST… obcs_calc.F:99-233`, Stevens `:448`). It is called early on purpose, because sea ice needs the OB fields (comment at `:292`).
- Sea ice, frazil, shelf ice, ice front, salt plume (`:368-558`).
- **`EXTERNAL_FORCING_SURF`** (`:579`). It turns Qnet/EmPmR/fu/fv into `surfaceForcingT/S/U/V` (`external_forcing_surf.F:144-329`), with optional `REMOVE_MEAN_RS` balancing (`:102-112`), `FORCING_SURF_RELAX` (`:174`), `PTRACERS_FORCING_SURF` (`:191`), and `SHELFICE_FORCING_SURF` (`:396`).
- **`OBCS_ADJUST(myTime+Δt)`** (`:595`). It applies balancing and tidal OB forcing once all surface fluxes are known.
- Per-tile loop: **EOS `FIND_RHO_2D` for every k into `rhoInSitu`** (`:758-766`). The `DWNSLP_CALC_RHO` / `BBL_CALC_RHO` variants are at `:736/750`. `FIND_RHO_2D` is called again with the k-1 reference pressure for `GRAD_SIGMA`/convection (`:818-825`). Then `CALC_IVDC` (`:871`), `CALC_OCE_MXLAYER` (`:897`), salt plume (`:904-920`).
- Closures compute **coefficients only**: `KPP_CALC` (`:956`), `PP81_CALC`, `KL10_CALC`, `MY82_CALC` (`:973-995`), `GGL90_CALC` (`:1010`), `GMREDI_CALC_TENSOR` (`:1034`), `DWNSLP_CALC_FLOW` (`:1052`). After the loop come `BBL_CALC_RHS` (`:1083`) and the template hook `MYPACKAGE_CALC_RHS` (`:1090`), then halo exchanges `GMREDI_DO_EXCH`, `KPP_DO_EXCH`, `GGL90_EXCHANGES` (`:1097-1109`) and surface-flux diagnostics (`:1114-1127`).
- The coefficients are only *consumed* later. Diffusivities are gathered by `CALC_3D_DIFFUSIVITY` (`calc_3d_diffusivity.F:176-240`: `KPP_CALC_DIFF_T/S/Ptr`, `GMREDI_CALC_DIFF`, `PP81/KL10/MY82/GGL90_CALC_DIFF`), which `TEMP_INTEGRATE` calls (`temp_integrate.F:233`). Viscosities are gathered by `CALC_VISCOSITY` (`calc_viscosity.F:77-114`), which `DYNAMICS` calls (`dynamics.F:386`).

**Where the EOS is used.** The hydrostatic pressure uses the `rhoInSitu` computed in `DO_OCEANIC_PHYS`. `CALC_PHI_HYD` copies it (`calc_phi_hyd.F:150-155`) and calls `FIND_RHO_2D` itself only when `implicitIntGravWave` is on or `myIter<0` (`:138-149`). So in both orderings the PGF uses θ^n, S^n, which were evaluated before `THERMODYNAMICS` runs.

### 1.3 `THERMODYNAMICS` → `TEMP_INTEGRATE` / `SALT_INTEGRATE` / `PTRACERS_INTEGRATE`
- `thermodynamics.F`: surface correction for the linear free surface, `CALC_WSURF_TR` (`:157`), and `PTRACERS_CALC_WSURF_TR` (`:177`). Per tile: copy u, v, w (`:194-260`), `GMREDI_RESIDUAL_FLOW` (`:272`), then `TEMP_INTEGRATE` (`:321`), `SALT_INTEGRATE` (`:332`), `PTRACERS_INTEGRATE` (`:348`), and **`OBCS_APPLY_TS`** on the new θ and S (`:359`).
- `temp_integrate.F` (the reference pattern for every tracer):
  1. If `AdamsBashforth_T` (AB applied to the state rather than the tendency), `ADAMS_BASHFORTH2/3(kArg=0)` extrapolates θ to n+½ (`:179-195`).
  2. `CALC_3D_DIFFUSIVITY` (`:233`) and multi-dimensional advection `GAD_ADVECTION` / `GAD_SOM_ADVECT` (`:263-277`).
  3. **Loop `DO k=Nr,1,-1`** (`:290`), bottom up, because the vertical flux at the lower interface is reused:
     - `CALC_ADV_FLOW` (`:308`) computes the transports.
     - **`APPLY_FORCING_T(gtForc)`** (`:322`) collects every source term (§2.3). It is diagnosed as `gT_Forc` (`:328`).
     - **`GAD_CALC_RHS`** (`:336/351`) adds the flux divergence to `gT_loc`.
     - Forcing is added inside AB (`tracForcingOutAB≠1`, `:368`), then `ADAMS_BASHFORTH2/3(kArg=k)` (`:376-389`), diagnosed as `AB_gT` (`:392`). If `tracForcingOutAB=1`, forcing is added after AB instead (`:398`).
     - `FREESURF_RESCALE_G` (`:410-433`).
  4. `DWNSLP_APPLY` (`:451`). `TIMESTEP_TRACER` gives `gT_loc = θ + Δt·gT` (`:469`).
  5. Implicit vertical advection and diffusion: `GAD_IMPLICIT_R` or `IMPLDIFF` (`:489/499`).
  6. `CYCLE_AB_TRACER` / `CYCLE_TRACER` write the new θ in place (`:509-523`).
- `pkg/generic_advdiff/gad_calc_rhs.F` (one level k):
  1. The fluxes `fZon`, `fMer`, `fVerT` are zeroed (`:201-208`).
  2. X: advective flux by scheme (`GAD_C2/U3/C4/DST3/DST3FL/OS7MP_ADV_X`, `:257-292`) plus the OBCS upwind override `OBCS_U1_ADV_TRACER` (`:302`). Diagnosed as `ADVx`+suffix (`:316`). Diffusive part `GAD_DIFF_X`, `GAD_BIHARM_X`, `GMREDI_XTRANSPORT` (`:329-346`), diagnosed as `DFxE` (`:366`).
  3. Y is the same (`:376-498`).
  4. R: advective flux (`:505-595`), diagnosed as `ADVr`. Explicit diffusion `GAD_DIFF_R`, `GAD_BIHARM_R`, `GMREDI_RTRANSPORT` (`:612-626`), diagnosed as `DFrE`. KPP non-local flux `KPP_TRANSPORT_T/S/PTR` (`:665-676`), diagnosed as `KPPg`.
  5. The **single flux-divergence update** to `gTracer` comes at `:770-782`, including the `−T·div(U)` term for the advective form.

  Every term is therefore produced first as a *face flux* and diagnosed as a flux. Fluxes are never diagnosed as cell tendencies.

### 1.4 `DYNAMICS` (`src/dynamics.F`)
Per tile:
1. `CALC_GRAD_PHI_SURF` explicit part, only if `implicSurfPress≠1` (`:354`).
2. `CALC_VISCOSITY` (`:386`, closures hook here), `MOM_CALC_3D_STRAIN` (`:395`), `OBCS_COPY_UV_N` (`:410`), `CALC_EDDY_STRESS` (`:418`).
3. **Loop `DO k=1,Nr`** (`:422`), top down:
   - **`CALC_PHI_HYD`** (`:482`) gives `dPhiHydX/Y`.
   - `MOM_FLUXFORM` (`:517`) or `MOM_VECINV` (`:527`), selected by `vectorInvariantMomentum`. Either one writes `gU` (advection, Coriolis, metric terms) plus `guDissip` (viscous and drag terms) as separate arrays.
   - Optional Smagorinsky 3-D (`:539-544`).
   - **`TIMESTEP`** (`:558`).
4. Implicit vertical viscosity and advection: `MOM_U/V_IMPLICIT_R` or `IMPLDIFF` (`:576-595`).
5. `OBCS_SAVE_UV_N` and **`OBCS_APPLY_UV(gU,gV)`** on the provisional velocity (`:607-610`).
6. Non-hydrostatic `CALC_GW`, `TIMESTEP_WVEL` (`:645-652`).

After the tile loop: `OBCS_EXCHANGES` (`:665`) and the `PHIHYD`, `PHIBOT`, `botTauX` diagnostics (`:682-703`).

`src/timestep.F` assembles the momentum tendency:
1. `APPLY_FORCING_U/V` builds `guExt` (`:106-110`).
2. **Synchronous** stepping (`.NOT.staggerTimeStep .AND. .NOT.implicitIntGravWave`) adds `−∇φ_hyd` to `gU` **before AB** (`:116-125`).
3. Dissipation goes in AB if `momDissip_In_AB` (`:128`). Forcing goes in AB if `momForcingOutAB≠1` (`:138`).
4. `ADAMS_BASHFORTH2/3` (`:163-180`), diagnosed as `AB_gU` (`:189`).
5. Forcing and dissipation that stay outside AB are added (`:208-230`).
6. **Staggered** stepping (or IGW) adds `−∇φ_hyd − psFac∇φ_s` *outside* AB (`:320-327`).
7. `gU = uVel + Δt·(gUtmp + gUdPx)·maskW` (`:372-386`).
8. Diagnostics: `Um_dPhiX` (`:409`), `Um_Diss` (`:413`), `Um_Ext` (`:417`), `Um_Cori` (`:422`, fluxform only; vecinv fills it at `mom_vecinv.F:724`).

### 1.5 Adams–Bashforth
- **AB2**, `src/adams_bashforth2.F`: `abFac = 0` on the first step when `startAB=0` and `myIter=nIter0`; otherwise `abFac = ½+abEps` (`:61-65`). In the tendency mode (`kArg>0`, `:84-86`): `AB = abFac·(g^n − g^{n−1})`, then `gNm1 ← g^n` and `g ← g^n + AB`. In the state mode (`kArg=0`, `:74-75`) the same is done on the field itself.
- **AB3**, `src/adams_bashforth3.F` (compile-time option `ALLOW_ADAMSBASHFORTH_3`). The two previous tendencies are stored in a 2-slot array `gtNm(...,2)`, indexed `m1 = 1+MOD(iter+1,2)` and `m2 = 1+MOD(iter,2)` (`:85-86`). The weights ramp up: 0 on the first step, `(α, −α, 0)` on the second, then `(α+β, −α−2β, β)` (`:89-103`), with `alph_AB` and `beta_AB` read at runtime.
- `startAB` per variable (`tempStartAB`, `mom_StartAB`) records how many past tendencies are available. It is set from the pickup content in `check_pickup.F:61-64,181`, so a pickup without tendencies restarts AB cleanly.
- **Where the tendencies accumulate.** Momentum uses the global `gU/gV` (`inc/DYNVARS.h:38-55`): the MOM_* routines write them, then TIMESTEP extrapolates them and finally overwrites them with u*. Tracers use a local `gT_loc` per tile, and only the AB history `gtNm1`/`gtNm` is global (`DYNVARS.h:36-63`). `gT/gS` exist as globals only under `USE_OLD_EXTERNAL_FORCING` (`:69-72`).

### 1.6 Synchronous vs staggered
| | synchronous (default, `set_defaults.F:183`) | `staggerTimeStep=.TRUE.` |
|---|---|---|
| position of `THERMODYNAMICS` | before `DYNAMICS` (`forward_step.F:725`) | after `INTEGR_CONTINUITY` (`:976-1005`) |
| velocity used for tracers | u^n, w^n | u^{n+1}, w^{n+1} |
| `∇φ_hyd` | inside AB (`timestep.F:116`) | outside AB, forward (`:320`) |
| density for PGF | ρ(θ^n) in both cases (`rhoInSitu`) | |

`config_check.F:178` forbids quasiHydrostatic with staggerTimeStep, and `:1069` forbids implicitIntGravWave with staggerTimeStep.

## 2. Package mechanism

### 2.1 Compile time
- A `packages.conf` in the build or mods directory lists the packages. Without it, `pkg/pkg_groups` gives the groups (`default_pkg_list : gfd`, `oceanic : gfd gmredi kpp`).
- `pkg/pkg_depend` gives the dependency rules: `+` means the package is required and added automatically, `-` means a conflict and stops the build, `=` means recommended (header at `:1-12`). Examples: `ptracers +generic_advdiff`, `matrix -gchem`, `bling -dic`, `exf =cal`.
- `tools/genmake2:3738` writes **`PACKAGES_CONFIG.h`** with `#define ALLOW_<PKG>` for enabled packages and `#undef` for disabled ones. Every call site is wrapped in `#ifdef ALLOW_KPP`.
- Package sizes are compile-time too, for example `PTRACERS_num` (`pkg/ptracers/PTRACERS_SIZE.h:16`). So are the `<PKG>_OPTIONS.h` CPP flags (`ALLOW_ORLANSKI`, `ALLOW_ADAMSBASHFORTH_3`, and others).

### 2.2 Runtime and lifecycle
- `PACKAGES_BOOT` reads namelist `/PACKAGES/` from `data.pkg` (`packages_boot.F:46,107,178`). It sets `useOBCS`, `useKPP`, `useRBCS`… (defaults `.FALSE.`, `:113-141`) and prints them (`:283-370`).
- `INITIALISE_FIXED` order (`initialise_fixed.F`):
  1. `INI_PARMS` (`:111`), `PACKAGES_BOOT` (`:117`), **`PACKAGES_READPARMS`** (`:133`; each `XXX_READPARMS` reads `data.xxx`, for example `KPP_READPARMS` at `packages_readparms.F:206`).
  2. `SET_PARMS`, grid, `INI_EOS`, ref state, depths, masks (`:140-190`).
  3. **`PACKAGES_INIT_FIXED`** (`:199`; for example `KPP_INIT_FIXED` at `packages_init_fixed.F:321`). Each package's `XXX_DIAGNOSTICS_INIT` registration happens here too, between `DIAGNOSTICS_INIT_EARLY` and `DIAGNOSTICS_MAIN_INIT` (`:172-176`) and the final `DIAGNOSTICS_INIT_FIXED` (`:673`).
  4. `INI_CORI`, `INI_CG2D`, `INI_CG3D` (`:217-230`), `CONFIG_SUMMARY` (`:237`).
  5. **`PACKAGES_CHECK`** (`:243`), then **`CONFIG_CHECK`** (`:249`).
- `INITIALISE_VARIA` (`initialise_varia.F`): `INI_DYNVARS` (`:205`), `INI_FIELDS` (`:225`, which calls `READ_PICKUP` when `nIter0≠0`, `ini_fields.F:38`), `INI_MIXING`, `INI_FORCING`, **`PACKAGES_INIT_VARIABLES`** (`:263`; for example `KPP_INIT_VARIA` at `packages_init_variables.F:255`). Then a diagnostic `INTEGR_CONTINUITY` gives a consistent w and η at t0 (`:334`).
- **Incompatibility detection** happens at three levels:
  - **Build**: `pkg_depend` rules.
  - **`PACKAGES_CHECK`**: for each package, `IF (useXXX) CALL XXX_CHECK` when it was compiled. If `useXXX` is set but the code is absent, `PACKAGES_ERROR_MSG` is called (`packages_check.F:245-249`). Example checks in `pkg/kpp/kpp_check.F`: KPP requires `implicitDiffusion` (`:136`), `implicitViscosity` (`:141`), overlap ≥ 4 (`:162`), and no convective adjustment (`:128`). Each failure stops the run.
  - **`CONFIG_CHECK`**: about 95 core checks. Each one prints the reason, increments `errCount` (95 occurrences) and lets checking continue; the run stops once at the end (`config_check.F:1152`). Collecting all errors before stopping is the pattern worth copying.

### 2.3 How packages inject tendencies: fixed hook points in the core
- **Coefficient hooks**: `DO_OCEANIC_PHYS` (`XXX_CALC`), `CALC_3D_DIFFUSIVITY` (`XXX_CALC_DIFF`), `CALC_VISCOSITY` (`XXX_CALC_VISC`), and `GAD_CALC_RHS` for KPP's non-local flux and GM's skew fluxes (`*_TRANSPORT`).
- **Source/sink hooks**: `APPLY_FORCING_U/V/T/S` in `src/apply_forcing.F`, one per variable. The order in the T case is:
  1. `EXTERNAL_FORCING_T` (`:455`), which adds `surfaceForcingT/drF/hFac` at the surface level (`external_forcing.F:496-512`) and the penetrating shortwave.
  2. AIM, atm_phys, FIZHI (`:484-498`), addMass and friction heating (`:504-550`).
  3. `FRAZIL_TENDENCY_APPLY_T` (`:697`), `SHELFICE_FORCING_T` (`:705`), `ICEFRONT_TENDENCY_APPLY_T` (`:713`), `SALT_PLUME_TENDENCY_APPLY_T` (`:720`).
  4. **`RBCS_ADD_TENDENCY`** (`:728`) and **`OBCS_SPONGE_T`** (`:737`).
  5. `BBL_TENDENCY_APPLY_T` (`:745`) and the template `MYPACKAGE_TENDENCY_APPLY_T` (`:753`).

  U, V and S have the same structure (`RBCS_ADD_TENDENCY` at `:170/360/960`, `OBCS_SPONGE_U/V/S` at `:180/369/969`, `TAUEDDY_TENDENCY_APPLY_U` at `:162`). Passive tracers use `PTRACERS_APPLY_FORCING`, which calls `GCHEM_ADD_TENDENCY` and `RBCS_ADD_TENDENCY` (`ptracers_apply_forcing.F:73,116`).
- **State-override hooks (OBCS)**: `OBCS_CALC` (future values, `do_oceanic_phys.F:322`), `OBCS_ADJUST` (`:595`), `OBCS_U1_ADV_TRACER` (flux override, `gad_calc_rhs.F:302`), `OBCS_APPLY_UV` on u* (`dynamics.F:610`) and on u^{n+1} (`momentum_correction_step.F:95`), the OB mask in the cg2d RHS (`solve_for_pressure.F`), `OBCS_APPLY_W` (`integr_continuity.F:299`), `OBCS_APPLY_TS` (`thermodynamics.F:359`), `OBCS_APPLY_PTRACER` (`ptracers_integrate.F:514`).
- `pkg/mypackage` is the template package. Its hooks `MYPACKAGE_CALC_RHS` (`do_oceanic_phys.F:1090`), `MYPACKAGE_TENDENCY_APPLY_*`, `MYPACKAGE_DIAGNOSTICS_STATE` (`do_statevars_diags.F:120`) list all the slots a new package can use.

## 3. State organization
- **Prognostic state**: `inc/DYNVARS.h:36-55`, common `/DYNVARS_R/`, holds `etaN, uVel, vVel, wVel, theta, salt, gU, gV` plus the AB history `guNm1, gvNm1, gtNm1, gsNm1` (AB2) or `guNm, gvNm, gtNm, gsNm(...,2)` (AB3). `etaH` (`/DYNVARS_R_2/`, `:75`) is η advanced consistently with the transports (exactConserv). `wVel` sits in the same block, but it is diagnostic: `INTEGRATE_FOR_W` recomputes it.
- **Diagnostic state**: `/DYNVARS_DIAG/` (`:107-117`) holds `rhoInSitu, totPhiHyd, phiHydLow, hMixLayer, IVDConvCount`. Also `sigmaRfield` (`:132`) and `cSound` (`:152`).
- **Forcing**: `inc/FFIELDS.h`. Raw fluxes have one common block each: `fu, fv, Qnet, Qsw, EmPmR, saltFlux, SST, SSS, lambda*ClimRelax, phiTide2d, pLoad, sIceLoad` (`:86-98`), plus `addMass`, `frictionHeating`, `geothermalFlux`, `SWFrac3D`. The processed fluxes are `/SURFACE_FORCING/ surfaceForcingU/V/T/S` (`:257-266`), and `botDragU/V` are kept for diagnostics (`:273`).
- **Free-surface geometry**: `inc/SURFACE.h` holds `Bo_surf, topoZ, phi0surf` (`:23`), `etaHnm1, dEtaHdt, PmEpR` (`/ETA_UPDATES/ :42`), the nonlinear free-surface `hFac_surf*` (`/SURF_CHANGE/ :54`), and the r* factors (`/RSTAR_CHANGE/ :82`, `/RSTAR_FIXED/ :105`).
- **Grid**: `inc/GRID.h` has `/GRID_RL/` (`:314`), `/GRID_RS/` (`:416`), time-varying `/GRID_VAR_RS/` (`:500`, with `hFacC/W/S`, mutable under r*), and `/GRID_I/` (`:528`).
- **Package state** lives in each package's own headers (KPP.h, OBCS_FIELDS.h, …), included only by that package's routines and by the core hook sites.
- **ptracers**:
  - `pTracer` and `gpTrNm1` are arrays with a trailing tracer dimension `PTRACERS_num`, which is fixed at compile time (`PTRACERS_FIELDS.h:17-24`, `PTRACERS_SIZE.h:16`).
  - At runtime, `PTRACERS_numInUse`, `PTRACERS_names`, the per-tracer `advScheme`, `diffKh`, `useKPP`, `useGMRedi`, `AdamsBash_Tr` and `stayPositive` come from `data.ptracers` (`ptracers_readparms.F:53-116`).
  - `PTRACERS_INTEGRATE` loops `iTracer` (`ptracers_integrate.F:154`) and calls the *same* `GAD_CALC_RHS` with identity `GAD_TR = GAD_TR1+iTracer−1` (`:156,315`).
  - Diagnostic names come from `GAD_DIAG_SUFX`: `_TH `, `_SLT`, or `Tr`+ioLabel (`gad_diagnostics_init.F:324-370`). That is how `ADVxTr01` and similar names arise.

## 4. Diagnostics package
- **Registration.** `DIAGNOSTICS_ADDTOLIST(diagNum, name, code, units, title, mate)` (`pkg/diagnostics/diagnostics_addtolist.F`) appends to the available list `cdiag/gdiag/udiag/tdiag` (`DIAGNOSTICS.h:46-65`). It must be called after `DIAGNOSTICS_INIT_EARLY` and before `DIAGNOSTICS_INIT_FIXED` (`addtolist.F` header).
  - Core terms are registered in `diagnostics_main_init.F`, for example `TOTTTEND` at `:820-826`. Package terms are registered in `XXX_DIAGNOSTICS_INIT`, for example `ADVr/ADVx/ADVy/DFrE/DFxE…` per tracer in `gad_diagnostics_init.F:67-100` and `Um_Diss/Um_ImplD/Um_Advec/Um_Cori/Um_dPhiX/Um_Ext/Um_AdvZ3/Um_AdvRe` in `mom_common/mom_diagnostics_init.F:303-425`.
  - The 16-character **parse code** encodes the C-grid location and vertical placement (`doc/outp_pkgs/outp_pkgs.rst:205-260`): character 1 is S/U/V (scalar or vector component), character 2 is U/V/M/Z (C-grid point), character 9 is U/M/L (level ± ½), and character 10 is 0/1/R/L (number of levels). Example: `'UU      MR      '` is a U-point vector at mid-level with Nr levels. `mate` pairs the U and V components for rotation and output.
- **Selection and cost.** `data.diagnostics` chooses fields, frequency and averaging. Expensive terms are guarded with `DIAGNOSTICS_IS_ON('name')`, for example `temp_integrate.F:168-170` and `do_oceanic_phys.F:257-263`. `DIAGNOSTICS_SWITCH_ONOFF` (`forward_step.F:508`) activates what is due this step.
- **Fill.** `DIAGNOSTICS_FILL(field, name, kLev, nLevs, bibjFlg, bi, bj, myThid)` *accumulates* into `qdiag` and increments the counter `ndiag` for time means (`diagnostics_fill.F:23-52`). `bibjFlg` says whether the call is inside the tile loop and whether the array is local or has halos. `kLev>0` fills one level, which is how per-k routines fill as they go. Variants: `DIAGNOSTICS_SCALE_FILL`, `_FILL_RS`, `_FRACT_FILL`.
- **Term-by-term budget.** Each process is filled *where it is computed*:
  - Tracer fluxes (not tendencies) come from `gad_calc_rhs.F`: `ADVx/ADVy/ADVr` at `:316,445,589`, `DFxE/DFyE/DFrE` at `:366,495,647`, `KPPg` at `:741`. The implicit vertical diffusive flux `DFrI` is filled in the implicit solver.
  - Tracer forcing: `gT_Forc` (`temp_integrate.F:328`). AB increment: `AB_gT` (`:392`).
  - Momentum tendencies: `Um_Advec`, filled from gU after the advective terms (`mom_vecinv.F:996`). `Um_Cori` (`mom_vecinv.F:724`), `Um_AdvZ3`/`Um_AdvRe` (`:817,840`), `Um_dPhiX`/`Um_Diss`/`Um_Ext` (`timestep.F:409-418`), `AB_gU` (`:189`), the surface-pressure part (`correction_step.F:104-122`), and `Um_ImplD` (implicit solver).
  - **TOTxTEND** is a difference of two fills. `DO_STATEVARS_DIAGS(seqFlag=0)` at the start of the step fills `−θ·86400/Δt` with the counter incremented (`diagnostics_fill_state.F:383-410`, `do_statevars_diags.F:60-67`). `seqFlag=2` at the end of the step adds `+θ·86400/Δt` without incrementing the counter (`diagnostics_fill_state.F:709-724`). This makes it exact and independent of the scheme: the residual `TOTTTEND − Σ terms` closes the budget.
- Output: `DIAGNOSTICS_WRITE` (via `DO_THE_MODEL_IO`). Regional statistics come from `diagstats_*`.

## 5. Restart (pickup)
- **When.** `DO_WRITE_PICKUP` (`do_write_pickup.F:58-111`) writes a *permanent* pickup every `pChkPtFreq` with suffix `%010d` = iteration (`:81-82`, or `RW_GET_SUFFIX` for a time suffix). It writes a *rolling* pickup every `chkPtFreq` with suffix `checkPtSuff(nCheckLev)` = `ckptA`/`ckptB`, alternating (`:85,108-110`). It also writes one at the end if `writePickupAtEnd`.
- **Order.** `PACKAGES_WRITE_PICKUP` comes first, because each package has its own file: GAD, GGL90, GMRedi, GCHEM, SEAICE… (`packages_write_pickup.F:96-206`). The core `WRITE_PICKUP` comes after.
- **What.** `write_pickup.F:98-380` writes `pickup.<suffix>.data/.meta` with named records: `Uvel, Vvel, Theta, Salt, GuNm1, GuNm2 (only if beta_AB≠0), …, EtaN, dEtaHdt, EtaH…`. The `.meta` file lists the field names (`MDS_WR_METAFILES`, `:380`).
- **Reading.** `READ_PICKUP` uses `pickupSuff` or the `nIter0` suffix (`read_pickup.F:82-89`). It matches fields by name. If AB tendencies are missing it fails unless `pickupStrictlyMatch=.FALSE.` (`:125-132`), and `CHECK_PICKUP` then lowers `startAB` (`check_pickup.F:61-64,181`) so the AB start-up is consistent.

## 6. Lessons for pyOCEAN (N1)

**Copy**
1. **A fixed, explicit step order with named phases**: load forcing → physics/closures (coefficients only) → [tracers if synchronous] → momentum tendency + AB → time advance → barotropic solve → momentum correction → continuity (w, η) → [tracers if staggered] → tracer correction (convective adjustment) → halo/OBC → diagnostics → output → restart. pyOCEAN's `SigmaBaroclinic` step should become a fixed list of such phases in a driver. Choices such as stagger, RK3 or AB3 should be strategy objects, not reorderings.
2. **Separate "compute coefficients" from "apply"**. Closures (KPP, GLS) only produce κ and ν, plus non-local fluxes. One gathering point per quantity (`CALC_3D_DIFFUSIVITY`, `CALC_VISCOSITY`) sums the closures' contributions. This is the right contract for GLS now and KPP in N.
3. **One source hook per prognostic variable** (`apply_forcing_{u,v,T,S,tracer}`) that every component may add to: surface flux, sponge, rbcs-style relaxation, drag. It returns its contribution separately, so it can be diagnosed and placed inside or outside AB (`tracForcingOutAB`, `momForcingOutAB`). pyOCEAN's interior sponge, edge sponge, wind stress and bed drag all fit here.
4. **OBC as a component with several named hooks** (calc future, adjust after fluxes, flux override in advection, apply to provisional u*, apply to final u, mask in the solver RHS, apply to T/S). These map one to one onto what pyOCEAN already does for Flather, Orlanski, the sponge and 'clamped' boundaries. Making the hooks explicit is the main structural gain.
5. **A lifecycle of `read_params → init_fixed → check → init_variables → (step hooks) → write_restart`**. Package checks run after grid setup and before the run. The core config check **accumulates all errors and fails once**. This maps directly onto the N `Simulation` pre-flight.
6. **Diagnostics registered once** (name, location code, units, title, vector mate) and **filled where they are computed**, guarded by an `is_on` check so they cost nothing when off. Tracer terms are diagnosed as **face fluxes** (ADVx/DFxE/…), not tendencies, so budgets over any control volume close. Adopt the naming convention `{ADV,DF}{x,y,r}{E,I}_<tracer>`, `Um_{Advec,Cori,dPhiX,Diss,Ext,ImplD}`, `AB_g*`. **The TOTxTEND difference trick** (subtract the state at the start of the step, add it at the end) gives an exact residual check.
7. **Tracers are generic**: T, S and passive tracers go through one `gad_calc_rhs`. Only an identity and a suffix differ, and per-tracer options are runtime. pyOCEAN's `n_tracers` already fits; make T, S and tr the same component type.
8. **A named-field restart with metadata** and a per-component restart section. AB start-up is lowered automatically when the history is missing. pyOCEAN's restart already checks the configuration; add per-component sections and `startAB` semantics.

**Avoid**
- **Compile-time configuration** (`PACKAGES_CONFIG.h`, `#ifdef ALLOW_*`, `PTRACERS_num`, the AB3 CPP switch). It creates the `useXXX` versus compiled double state and the `PACKAGES_ERROR_MSG` case. In Python, a component is either present in the config or not. Numba/JAX specialization should come from static config at JIT time, not from source variants.
- **Common blocks and global mutable state**. A routine's inputs and outputs are invisible; `gU` is reused as tendency, then AB-extrapolated tendency, then u*. This blocks JAX (purity) and testing. Use an explicit state object (NumPy arrays, as decided) plus explicit per-component scratch, and pass both in and out.
- **Hook calls hard-coded in core routines** (`IF (useRBCS) CALL RBCS_ADD_TENDENCY` inside `apply_forcing.F`). Adding a package means editing the core. Use a registry: components declare which hooks they implement, and the driver iterates in a fixed, documented order. The order must be deterministic, to keep bit-for-bit results.
- **Hidden timing subtleties**: the PGF uses ρ(θ^n) even in staggered mode, `OBCS_CALC` is called with `myTime+Δt`, and time advances in the middle of the step. Document the time level of every hook argument in the contract.
- **Per-k loops with rolling kUp/kDown buffers**. They are an artifact of memory economy; pyOCEAN's vectorized 3-D kernels should stay.
