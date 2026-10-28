# How MOM6 composes its time step (MOM6 commit 0e9a561), for pyOCEAN N1

All paths are relative to `~/pyOCEAN_refs/MOM6/src/`. Line numbers are for commit 0e9a561.

## 1. Call order of one coupling step

### 1a. `step_MOM` (core/MOM.F90:540): the outer cycle

- **Arguments.** `step_MOM(forces_in, fluxes_in, sfc_state, Time_start, time_int_in, CS, Waves, do_dynamics, do_thermodynamics, start_cycle, ...)`. The caller can ask for dynamics only or thermodynamics only (MOM.F90:650-653). Asking for neither is FATAL.
- **Forcing preparation.** Forcing may be rotated or homogenised (674-695). Then the halo exchanges for `taux`/`tauy`/`ustar`/`p_surf` are started (740-760).
- **Choosing dt_therm.** `dt_therm` is set as a whole number of dynamics steps: `ntstep = MAX(1, MIN(n_max, floor(CS%dt_therm/dt)))` and `dt_therm = dt*ntstep` (715-724). With `THERMO_SPANS_COUPLING`, it is a whole multiple of the coupling step instead (709-716).
- **Derived thermodynamic fields (EOS).** `calc_derived_thermo(CS%tv, h, ...)` fills the specific volume `SpV_avg` (770, 796). The resolution functions for the lateral parameterisations are `calc_resoln_function`/`calc_depth_function` (819-820).
- **The DT loop: `do n=1,n_max`** (873). `CS%Time` is set to the middle of the step and `Time_local` to its end (876-878). Each pass does, in order:
  1. **Optional diabatic-first.** If `CS%diabatic_first`, `t_dyn_rel_adv==0` and `do_thermo`, it calls `step_MOM_thermo`, then `ALE_regridding_and_remapping` (if ALE), then `post_diabatic_halo_updates` (889-930).
  2. **Dynamics** (when `do_dyn`). `step_MOM_dynamics(forces, p_surf_begin, p_surf_end, dt, dt_tradv_here, bbl_time_int, CS, Time_local, Waves)` (983).
  3. **Tracer advection,** every DT_TRACER_ADVECT. `do_advection = (MOD(n,ntastep)==0) .or. (n==n_max)`, or a time-based test when advection spans coupling (989-993). When true it calls `step_MOM_tracer_dyn(CS, G, GV, US, h, Time_local)` (997).
  4. **Diabatic step,** every DT_THERM. `do_diabatic = do_thermo .and. ((MOD(n,ntstep)==0) .or. (n==n_max))` (1006-1009). If not diabatic-first, it calls `step_MOM_thermo(..., dtdia, ...)` (1030), then `ALE_regridding_and_remapping` (1033), then `post_diabatic_halo_updates` (1034).
  5. **SSH and sync diagnostics.** `find_eta` gives SSH (1054). `calculate_diagnostic_fields(u, v, h, uh, vh, tv, ADp, CDp, ...)` runs at the thermodynamic sync (1074), followed by `post_tracer_diagnostics_at_sync` and `post_tracer_integral_diagnostics` (1077-1078).
- **After the loop.** Inverse-barometer SSH adjustment (1105). Data assimilation `oda` (1119-1123). `extract_surface_state` (1128). Surface diagnostics (1143-1147). `accumulate_net_input` (1160) and `write_energy` (1164).

**Three clocks are kept in the control structure** (MOM.F90:241-260 inside `MOM_control_struct`):
- `t_dyn_rel_adv`: the time elapsed since tracers were last advected;
- `t_dyn_rel_thermo`: the time offset between the dynamics and the diabatic/ALE step;
- `t_dyn_rel_diag`: the same for diagnostics.

`step_MOM_dynamics` advances all three (1486, 1502-1504). `step_MOM_tracer_dyn` resets `t_dyn_rel_adv` (1649). Consistency is checked with FATAL errors: MOM.F90:897, 998, 1020, and at init "If using DIABATIC_FIRST, DT_TRACER_ADVECT must equal DT_THERM" (2669).

**Rates.**

| Rate | What runs |
|---|---|
| Every **DT** | `step_MOM_dynamics` (PGF, Coriolis/advection, horizontal viscosity, vertical viscosity, barotropic solver, continuity, momentum OBCs). Also accumulates `uhtr`/`vhtr` += `uh*dt` (MOM_dynamics_split_RK2.F90:1083-1090). |
| Every **DT_TRACER_ADVECT** (default DT_THERM, 2659) | `advect_tracer` on the accumulated `uhtr` with h_end = h, `tracer_hordiff`, tracer OBC reservoirs, MEKE when it is not run in the dynamics. |
| Every **DT_THERM** (default DT, 2641) | `diabatic` (KPP/ePBL/set_diffusivity, surface buoyancy fluxes, vertical tracer diffusion, column tracer packages, ALE sponge), then ALE regrid/remap, then the EOS-derived fields. |

### 1b. `step_MOM_dynamics` (MOM.F90:1186)

1. `MOM_stoch_eos_run` and `MOM_calc_varT` (1237-1241).
2. If `thickness_diffuse_first`: `thickness_diffuse` and `interface_filter` act on h and `uhtr` (1246-1274).
3. Porous barriers (1283).
4. `set_viscous_BBL(u, v, h, tv, visc, G, GV, US, set_visc_CSp, pbv)`, the bottom boundary layer viscosity, called once per `bbl_time_int` (1292-1298).
5. Dispatch to the stepper: `step_MOM_dyn_split_RK2` (1333), RK2b (1328) or unsplit (1349/1354).
6. If not `thickness_diffuse_first`: `thickness_diffuse` (GM) (1420), `interface_filter` (1434), `mixedlayer_restrat` (1456).
7. `step_forward_MEKE` when `MEKE_in_dynamics` (1479).
8. Advance the clocks and post u, v, h (1486-1513).

### 1c. `step_MOM_dyn_split_RK2` (core/MOM_dynamics_split_RK2.F90:301)

Signature: `(u_inst, v_inst, h, tv, visc, Time_local, dt, forces, p_surf_begin, p_surf_end, uh, vh, uhtr, vhtr, eta_av, G, GV, US, CS, calc_dtbt, VarMix, MEKE, thickness_diffuse_CSp, pbv, STOCH, Waves)`.

**Predictor stage**
1. `update_OBC_ramp` (465). Save `u_old_rad_OBC` (468).
2. `PressureForce(h, tv, CS%PFu, CS%PFv, G, GV, US, CS%PressureForce_CSp, CS%ALE_CSp, CS%ADp, p_surf, CS%pbce, CS%eta_PF)` (513). The finite-volume PGF calls the EOS density integrals, `calc_SAL` and `calc_tidal_forcing` inside it (MOM_PressureForce_FV.F90:434-457). Stokes PGF follows (529).
3. `update_OBC_data(OBC, G, GV, US, tv, h, update_OBC_CSp, Time_local)` (553). `open_boundary_zero_normal_flow` on PFu/PFv (556).
4. `CorAdCalc(u_av, v_av, h_av, uh, vh, CAu_pred, CAv_pred, OBC, AD_pred, ...)` (566). This is skipped when `CAu_pred_stored`, i.e. stored at the end of the previous step (1101-1106).
5. `u_bc_accel = (CAu_pred + PFu) + diffu` (575-581). `diffu` is the horizontal viscosity carried over from the **previous** corrector.
6. `set_viscous_ML` (612).
7. `vertvisc_coef(up, vp, h, dz, forces, visc, tv, dt, ...)` and `vertvisc_remnant(visc, visc_rem_u, visc_rem_v, dt, ...)` (619-620).
8. `btcalc` and `bt_mass_source` (638-639). `continuity` → `BT_cont` (656). `set_dtbt` (673).
9. `btstep(u_inst, v_inst, eta, dt, u_bc_accel, v_bc_accel, forces, pbce, eta_PF, u_av, v_av, u_accel_bt, v_accel_bt, eta_pred, uhbt, vhbt, G, GV, US, barotropic_CSp, visc_rem_u, visc_rem_v, SpV_avg, ADp, OBC, BT_cont, eta_PF_start, taux_bot, tauy_bot, ...)` (685).
10. `up = u + dt_pred*(u_bc_accel + u_accel_bt)` with `dt_pred = be*dt` (720-735).
11. `vertvisc_coef` then `vertvisc(up, vp, h, forces, visc, dt_pred, OBC, AD_pred, CDp, ...)` (750-766). Wind and bottom stress enter here as boundary fluxes.
12. `continuity(up, vp, h, hp, uh, vh, dt, ..., u_cor=u_av, v_cor=v_av, BT_cont=...)` (793). `radiation_open_bdry_conds` (806).

**Corrector stage**
1. `bt_mass_source` (831). `PressureForce(hp, ...)` (847). `btcalc` (876).
2. `horizontal_viscosity(u_av, v_av, h_av, uh, vh, diffu, diffv, MEKE, VarMix, G, GV, US, hor_visc, tv, dt, OBC, BT, TD, ADp, hu_cont, hv_cont, STOCH)` (894).
3. `CorAdCalc(u_av, v_av, h_av, uh, vh, CAu, CAv, OBC, ADp, ...)` (903).
4. `u_bc_accel = (CAu + PFu) + diffu` (910-916).
5. `btstep(...)` (947), then `eta = eta_pred`.
6. `u_inst += dt*(u_bc_accel + u_accel_bt)` (~985-995).
7. `vertvisc_coef` and `vertvisc(u_inst, ...)` (1012-1022), then `vertvisc_remnant` (1031).
8. `continuity(u_inst, v_inst, h, h, uh, vh, dt, ..., uhbt, vhbt, visc_rem_u, visc_rem_v, u_cor=u_av, v_cor=v_av)` (1052).
9. `radiation_open_bdry_conds` (1071).
10. `h_av = ½(h_av + h)`. `uhtr += uh*dt` (1083-1090). `update_segment_thickness_reservoirs` (1093).
11. Optionally store `CAu_pred` for the next step (1101-1106).
12. Post diagnostics, e.g. `id_PFu`, `id_CAu` (1120-1194).

**Key point:** in MOM6 each "component" is a function that writes an **acceleration array** (PFu, CAu, diffu, u_accel_bt). The stepper adds these arrays itself, explicitly, and applies vertical viscosity as an implicit operator on the updated velocity.

### 1d. `step_MOM_tracer_dyn` (MOM.F90:1523)

1. OBC tracer data: `read_OBC_tracer_data` and `update_OBC_tracer_data` (1578-1582).
2. Choose the sweep direction from `n_dyn_steps_in_adv` (1590).
3. `advect_tracer(h, uhtr, vhtr, OBC, t_dyn_rel_adv, G, GV, US, tracer_adv_CSp, tracer_Reg, x_first)` (1595). Signature at tracer/MOM_tracer_advect.F90:55. It acts on **every** tracer in the registry.
4. `tracer_hordiff` (1618) and `update_segment_tracer_reservoirs` (1624).
5. `post_transport_diagnostics` (1630). Zero `uhtr`/`vhtr` (1642-1649).
6. MEKE when not in the dynamics (1653). T/S halo updates and `calc_derived_thermo` (1663-1677).

### 1e. `step_MOM_thermo` (MOM.F90:1687) and `diabatic` (parameterizations/vertical/MOM_diabatic_driver.F90:286)

`step_MOM_thermo` does, in order:
1. `apply_oda_tracer_increments` (1725).
2. `set_viscous_BBL` (1753).
3. `diabatic(u, v, h, tv, CS%Hml, fluxes, CS%visc, CS%ADp, CS%CDp, dtdia, Time_end_thermo, G, GV, US, diabatic_CSp, stoch_CS, OBC, Waves)` (1774), or `adiabatic` (1807).
4. Halo updates and `calc_derived_thermo` (1813-1823).

`diabatic` itself:
1. `set_BBL_TKE` (374) and `make_frazil` (390). Internal tides: `set_int_tide_input` and `propagate_int_tide` (406-409).
2. Dispatch to `diabatic_ALE_legacy` (416), `diabatic_ALE` (419) or `layered_diabatic` (422).
3. Frazil and MLD diagnostics (450-538).

`diabatic_ALE` (1288) runs:
1. `geothermal_in_place` (1434).
2. `set_pen_shortwave` (1448).
3. `set_diffusivity(u, v, h, u_h, v_h, tv, fluxes, optics, visc, dt, Kd_heat, ...)` for background, shear, tidal and double-diffusive mixing (1467).
4. KPP: `KPP_compute_BLD`, `KPP_calculate` and the non-local transport (1531-1573).
5. `calculate_CVMix_conv` (1589).
6. Surface fluxes: `applyBoundaryFluxesInOut` applies the surface heat, fresh-water and penetrating shortwave fluxes and **changes h** (1610/1657). With ePBL, `energetic_PBL` runs between them (1626).
7. `diagnose_boundary_forcing_tendency` (1667).
8. The implicit vertical diffusion of T and S, `tracer_vertdiff_Eulerian(h, ent_t, dt, tv%T, ...)` (1728-1729), then `diagnose_diabatic_diff_tendency` (1733).
9. `call_tracer_column_fns(h_orig, h, ea, eb, fluxes, BLD, dt, ...)` for every passive tracer package (1894).
10. `apply_ALE_sponge(h, tv, dt, ..., Time)` (1907), which can also act on u and v (`SPONGE_UV`, MOM_ALE_sponge.F90:217).
11. `apply_oda_incupd` (1919).

### 1f. `ALE_regridding_and_remapping` (MOM.F90:1845)

1. `preAle_tracer_diagnostics` (1901) and `pre_ALE_diagnostics` (1916).
2. `ALE_update_regrid_weights` (1917) and `pre_ALE_adjustments` (1919).
3. `ALE_regrid(G, GV, US, h, h_new, dzRegrid, tv, ALE_CSp, PCM_cell=...)` (1926).
4. `ALE_remap_tracers(ALE_CSp, G, GV, h, h_new, tracer_Reg, ...)` (1931). It loops over the registry.
5. `ALE_remap_set_h_vel` (1934-1938) and `ALE_remap_velocities` (1942).
6. Remap the auxiliary state: `remap_dyn_split_RK2_aux_vars` covers `u_av`, `CAu_pred` and `diffu` (1952; MOM_dynamics_split_RK2.F90:1330-1357). `remap_OBC_fields` (1961) and `remap_vertvisc_aux_vars` (1963).
7. `calc_derived_thermo` (1979), `diag_update_remap_grids` (1986) and `postALE_tracer_diagnostics` (1988).

**ALE therefore runs at the DT_THERM rate, after the diabatic step, not every dynamic step.** This differs from pyOCEAN's current regrid–remap every step.

## 2. How state is organised

- **`MOM_control_struct`** (MOM.F90:210-495) is declared `; private`. It holds:
  - the prognostic arrays `h, T, S, u, v, uh, vh, uhtr, vhtr` (211-222);
  - `tv` (thermo_var_ptrs), the clocks, and pointers `G`/`GV`/`US`;
  - **one sub-control-structure per component**: `dyn_split_RK2_CSp` (425), `set_visc_CSp`, `diabatic_CSp` (440), `MEKE_CSp`, `VarMix`, `tracer_Reg` (446), `tracer_adv_CSp`, `tracer_diff_CSp`, `tracer_flow_CSp`, `update_OBC_CSp`, `OBC`, `sponge_CSp`, `ALE_sponge_CSp`, `ALE_CSp` (468), `restart_CS` (493), and others (420-493).
- **`thermo_var_ptrs`** (core/MOM_variables.F90:79-131) is a bundle of *pointers*, not owned storage:
  - `T`, `S`, `p_surf`, `eqn_of_state` (the EOS_type), `P_Ref`, `C_p`, the `T_is_conT`/`S_is_absS` flags;
  - `SpV_avg` plus the `valid_SpV_halo` validity flag;
  - `frazil`, `salt_deficit`, `varT`;
  - `tr_T`/`tr_S`, which point back into the tracer registry.
  
  So T and S are owned by the control structure and *also* registered as tracers.
- **Other state types** in MOM_variables.F90:
  - `vertvisc_type` (258-327): Kv, bottom boundary layer fields, `Kv_shear`, `MLD`. Diabatic **writes** it and dynamics **reads** it; it is the only coupling between the two.
  - `accel_diag_ptrs` (167-238) and `cont_diag_ptrs` (241-255).
  - `BT_cont_type` (331-366) and `ocean_internal_state` (138-164, pointers used by diagnostics).
- **Grids are read-only.** `ocean_grid_type` (core/MOM_grid.F90:28) holds metrics and masks. `verticalGrid_type` (core/MOM_verticalGrid.F90:26) holds nk, H_to_m, g_Earth, Rho0. `unit_scale_type` (framework/MOM_unit_scaling.F90:24) holds the dimensional rescaling factors used by every `get_param(... scale=)`.
- **Forcing types.** `forcing` (thermodynamic fluxes, core/MOM_forcing_type.F90:82) is kept separate from `mech_forcing` (stresses and p_surf, :273). Dynamics sees only `forces`; diabatic sees `fluxes`.
- **Tracer registry** (tracer/MOM_tracer_registry.F90). A package calls `register_tracer(tr_ptr, Reg, param_file, HI, GV, name, longname, units, ..., ad_x, ad_y, df_x, df_y, advection_xy, registry_diags, ..., restart_CS, mandatory, underflow_conc, Tr_out, advect_scheme)` (58-64).
  - It appends to `Reg%Tr(ntr)`; exceeding `MAX_FIELDS_` is FATAL (148-151).
  - It also registers the tracer for restart (`register_restart_field(tr_ptr, Tr%name, mand, restart_CS, ...)`, 268).
  - `lock_tracer_registry` (276) closes registration.
  - `tracer_type` (tracer/MOM_tracer_types.F90:13) carries optional flux and tendency arrays (`df_x`, `advection_xy`, `t_prev`, `Trxh_prev`) and diagnostic ids (`id_adx`, `id_tendency`, 118-127).
  - Advection, horizontal diffusion and ALE remapping loop over `Reg` generically.
- **Tracer packages** plug in through `MOM_tracer_flow_control.F90`, a hand-written dispatcher:
  - `call_tracer_register` (165) reads `USE_IDEAL_AGE_TRACER` etc. (203) and calls each package's `register_*_tracer` function, which returns a logical (256-257);
  - `tracer_flow_control_init` (296), `call_tracer_column_fns` (465), `call_tracer_stocks` (695), `call_tracer_surface_state` (909) and `tracer_flow_control_end` (947) each contain one `if (CS%use_<pkg>)` branch per package (e.g. 536, 632).
  - A package such as `ideal_age_example.F90` provides `register_*` (79), `initialize_*` (216), `*_column_physics` (301), `*_stock` (499), `*_surface_state` (540) and `*_end` (571). That is a de-facto interface, but not an abstract type.
- **Diagnostic registry** (framework/MOM_diag_mediator.F90). Each module registers at init:
  `id = register_diag_field(module_name, field_name, axes, init_time, long_name, units, ..., standard_name, ..., cmor_*, cell_methods, ..., conversion, v_extensive)` (2250-2254).
  - The returned id is −1 when the diag_table does not request the field.
  - At run time it posts with `if (id>0) call post_data(id, field, diag)` (post_data_3d, 1614), inside `enable_averages(dt, Time, diag)` (2197) / `disable_averaging` windows.
  - Remapped vertical diagnostics are refreshed with `diag_update_remap_grids` after every change of h (e.g. MOM.F90:885, 1476, 1986).
- **Budget (term-by-term) diagnostics.**
  - **Momentum.** The tendencies are always-allocated stepper arrays (`CS%PFu`, `CS%CAu`, `CS%diffu`, `CS%u_accel_bt`) posted directly (MOM_dynamics_split_RK2.F90:1120-1194, also h-weighted and depth-integrated through `post_product_u` / `post_product_sum_u`). Implicit terms use the "before/after" trick on **lazily allocated** `accel_diag_ptrs` fields:
    - `vertvisc_init` registers `du_dt_visc` and calls `safe_alloc_ptr(ADp%du_dt_visc, ...)` only if requested (MOM_vert_friction.F90:3091-3096);
    - `vertvisc` stores `u` before the solve (671-675) and replaces it with `(u_new − u_old)/dt` afterwards (836-841).
    
    The KE budget is assembled in `calculate_energy_diagnostics` (diagnostics/MOM_diagnostics.F90, ids at 90-98).
  - **Tracers.** The registry diagnostics are:
    - `<tr>_advection_xy` and `<tr>_tendency` (MOM_tracer_registry.F90:497-509) and `_dfxy_cont_tendency` (523);
    - the pre- and post-ALE remap tendency (`preALE_` / `postALE_tracer_diagnostics`, 682/700);
    - the boundary-forcing and diabatic-diffusion tendencies (MOM_diabatic_driver.F90:1667, 1733).

## 3. How optional parameterisations plug in

- **Module pattern.** Each module exposes:
  - an opaque `type :: <mod>_CS` (often `; private`);
  - `<mod>_init(Time, G, GV, US, param_file, diag, CS, ...)`, which reads parameters, allocates and registers diagnostics;
  - the work routines, which take `CS` as an argument;
  - `<mod>_end(CS)`, which deallocates.
  
  Examples:
  - `vertvisc_init(MIS, Time, G, GV, US, param_file, diag, ADp, dirs, ...)` (MOM_vert_friction.F90:2719) and `vertvisc_end` (3253);
  - `energetic_PBL_init` (MOM_energetic_PBL.F90:3732) and `_end` (4437);
  - `KPP_init` is a *logical function* returning "am I on": `CS%useKPP = KPP_init(param_file, G, GV, US, diag, Time, CS%KPP_CSp, passive=...)` (MOM_diabatic_driver.F90:3674). It reads `USE_KPP` and `return`s early if off (MOM_CVMix_KPP.F90:270-277). `KPP_end` (2039).
  
  The parent driver holds the child CS and a `use_X` logical, and branches on it in the step (e.g. `if (CS%use_energetic_PBL)` at MOM_diabatic_driver.F90:1607).
- **Init order is hard-coded** in `initialize_MOM`:
  1. grid and `MOM_initialize_fixed` (2972-2975);
  2. `tracer_registry_init` (3038) and `EOS_init` (3161);
  3. **restart registration**, then `restart_registry_lock` (3173-3245);
  4. `MOM_initialize_coord` and `ALE_init` (3268-3272);
  5. `MOM_initialize_state` (3338-3397) and OBCs (3417);
  6. `diag_mediator_init` (3570);
  7. `VarMix_init`, `set_visc_init` and `thickness_diffuse_init` (3624-3626);
  8. `initialize_dyn_split_RK2` (3650), which calls `continuity_init`, `PressureForce_init`, `vertvisc_init` and `barotropic_init` (MOM_dynamics_split_RK2.F90:1594-1640);
  9. `MOM_diagnostics_init` (3747) and `diabatic_driver_init` (3756);
  10. `tracer_advect_init` and `tracer_hor_diff_init` (3774-3775);
  11. `register_diags` (3783) and `tracer_flow_control_init` (3821).
- **`get_param`** (framework/MOM_file_parser.F90:117 interface; real version at 1880) has the signature `get_param(CS, modulename, varname, value, desc, units, default, fail_if_missing, do_not_read, do_not_log, debuggingParam, scale, unscaled, old_name)`.
  - The default is applied first, then the file value. `old_name` gives deprecated aliases (1910-1925).
  - `log_param_real` writes name, value, description, units and default to the doc files.
  - The doc files are `MOM_parameter_doc.all`, `.short`, `.layout` and `.debugging` (framework/MOM_document.F90:31-34, 929-938). `doc_param_real` (278) marks whether each value equals the default.
  - `scale=US%s_to_T` converts to internal units at read time (MOM.F90:2635-2640).
  - Unused parameters are reported (`report_unused`, MOM_file_parser.F90:31, 76-79).
  - A common idiom reads the switch silently first, then calls `log_version(..., all_default=.not.on)`, then reads again with the doc string (MOM_CVMix_KPP.F90:270-276).
- **Incompatible options cause FATAL errors at init.** The checks are ad-hoc `if (...) call MOM_error(FATAL, ...)` blocks:
  - MOM.F90:2855-2866: ALE+BULKMIXEDLAYER, ALE without T/S, USE_EOS without thermodynamics, ADIABATIC+BULKMIXEDLAYER;
  - MOM.F90:2669: DIABATIC_FIRST with DT_TRACER_ADVECT ≠ DT_THERM;
  - MOM.F90:2912-2917 and 3365: index rotation;
  - MOM_diabatic_driver.F90:3480-3482: ePBL without ALE; :3876: DOUBLE_DIFFUSION;
  - :3275: adiabatic driver.
  
  There is no central compatibility table; each module checks what it knows.

## 4. Interfaces between pieces

| Routine | Reads | Writes |
|---|---|---|
| `PressureForce(h, tv, PFu, PFv, G, GV, US, CS, ALE_CSp, ADp, p_atm, pbce, eta)` (core/MOM_PressureForce.F90:43) | h, tv (T, S, EOS), p_atm | `PFu`/`PFv` [L T-2]; `pbce` (barotropic–baroclinic coupling); `eta` |
| `CorAdCalc(u, v, h, uh, vh, CAu, CAv, OBC, AD, G, GV, US, CS, pbv, Waves)` (core/MOM_CoriolisAdv.F90:143) | time-averaged u_av, h_av, and uh | `CAu`/`CAv`; optional `AD%gradKEu`, `AD%rv_x_v` |
| `horizontal_viscosity(u, v, h, uh, vh, diffu, diffv, MEKE, VarMix, G, GV, US, CS, tv, dt, OBC, BT, TD, ADp, hu_cont, hv_cont, STOCH)` (parameterizations/lateral/MOM_hor_visc.F90:277) | u, v, h, MEKE, VarMix, BT state | `diffu`/`diffv` |
| `vertvisc_coef(u, v, h, dz, forces, visc, tv, dt, G, GV, US, CS, OBC, VarMix)` (MOM_vert_friction.F90:1271) | visc (Kv, BBL), forces (ustar) | coupling coefficients stored in vertvisc CS |
| `vertvisc(u, v, h, forces, visc, dt, OBC, ADp, CDp, G, GV, US, CS, taux_bot, tauy_bot, fpmix, Waves)` (:542) | wind stress from `forces` (top boundary) | **u, v updated in place** (implicit tridiagonal); `taux_bot`; `ADp%du_dt_visc`, `du_dt_str` |
| `vertvisc_remnant(visc, visc_rem_u, visc_rem_v, dt, ...)` (:1164) | the coefficients above | the fraction of momentum surviving viscosity; `btstep` uses it to project the barotropic acceleration |
| `btstep(U_in, V_in, eta_in, dt, bc_accel_u, bc_accel_v, forces, pbce, eta_PF_in, U_Cor, V_Cor, accel_layer_u, accel_layer_v, eta_out, uhbtav, vhbtav, G, GV, US, CS, visc_rem_u, visc_rem_v, SpV_avg, ADp, OBC, BT_cont, eta_PF_start, taux_bot, tauy_bot, uh0, vh0, u_uh0, v_vh0, etaav)` (core/MOM_barotropic.F90:477) | the summed baroclinic accelerations | `accel_layer_u/v` (the 3-D barotropic correction, intent(out)), `eta_out`, `uhbtav/vhbtav` |
| `continuity` = `continuity_PPM(u, v, hin, h, uh, vh, dt, G, GV, US, CS, OBC, pbv, uhbt, vhbt, visc_rem_u, visc_rem_v, u_cor, v_cor, BT_cont, du_cor, dv_cor)` (core/MOM_continuity.F90:8 renames it; core/MOM_continuity_PPM.F90:94) | u, v, hin; optional uhbt/vhbt target transports | new h, uh, vh; `u_cor` (velocities adjusted so Σ uh matches uhbt) |
| `diabatic(u, v, h, tv, BLD, fluxes, visc, ADp, CDp, dt, Time_end, G, GV, US, CS, stoch_CS, OBC, Waves)` (MOM_diabatic_driver.F90:286) | fluxes | **everything inout**: h (surface mass fluxes), T, S, all tracers, visc (Kv_shear, MLD), BLD |
| `advect_tracer(h_end, uhtr, vhtr, OBC, dt, G, GV, US, CS, Reg, x_first_in, ...)` (tracer/MOM_tracer_advect.F90:55) | accumulated mass fluxes, h_end | all tracers in `Reg` |

**How the accelerations are collected.**
- The split stepper owns `CS%PFu`, `CS%CAu`, `CS%CAu_pred`, `CS%diffu` and `CS%u_accel_bt`, and sums them explicitly: `u_bc_accel = (CAu + PFu) + diffu` (MOM_dynamics_split_RK2.F90:575-581, 910-916). The bracketing fixes the order of summation, which matters for bit-for-bit reproducibility.
- `accel_diag_ptrs` (MOM_variables.F90:167-238) is a *diagnostic* bag: `ADp%CAu/PFu/diffu` point to the stepper's arrays, set in `initialize_dyn_split_RK2` (`Accel_diag%PFu => CS%PFu`, MOM_dynamics_split_RK2.F90:1568-1576; also `MIS%PFu => CS%PFu`, 1557-1567), while `du_dt_visc`, `du_dt_str`, `du_dt_dia`, `gradKEu`, `rv_x_v` and `diag_hfrac_u` are allocated only on request.
- `cont_diag_ptrs` (241-255) holds `uh`, `uhGM` and friends.
- `MOM_accel_chksum` (core/MOM_checksum_packages.F90:201) takes all of them for debugging.

## 5. Restart registration and debug hooks

- **Registration.**
  - `restart_init(param_file, CS%restart_CS)` (MOM.F90:3173), then `set_restart_fields` registers T, S, h and the (u,v) pair (MOM.F90:4045-4057). It uses `register_restart_field(CS%h, "h", .true., restart_CSp, "Layer Thickness", units, conversion=GV%H_to_MKS)` and `register_restart_pair(CS%u, CS%v, u_desc, v_desc, .true., ...)`, where `u_desc = var_desc("u","m s-1",..., hor_grid='Cu')`.
  - Each component then registers its own fields:
    - the stepper: `register_restarts_dyn_split_RK2` (3181), which covers eta, u_av/v_av, CAu_pred, h_av, uh/vh, diffu and taux_bot (MOM_dynamics_split_RK2.F90:1278-1322);
    - tracers through `call_tracer_register` (3193);
    - `MEKE_alloc_register_restart` (3196) and `set_visc_register_restarts` (3197);
    - `mixedlayer_restrat_register_restarts` (3198) and `open_boundary_register_restarts` (3222);
    - waves (3233), stochastic EOS (3237) and `register_diabatic_restarts` (3241).
  - `restart_registry_lock(restart_CSp)` (3245) forbids later additions (`locked`, framework/MOM_restart.F90:121).
  - The generic interface is `register_restart_field` (MOM_restart.F90:143), with 2-D/3-D/4-D variants (`register_restart_field_3d(f_ptr, name, mandatory, CS, longname, units, conversion, hor_grid, z_grid, t_grid, extra_axes)`).
- **On restart.**
  - Each init asks `query_initialized(field, name, restart_CS)` and computes a default if the field was not in the file. Examples: `CS%eta` at MOM_dynamics_split_RK2.F90:1623, `u_av` at 1653-1657 (then `set_initialized`), `h` at MOM.F90:3470.
  - `is_new_run(restart_CS)` gates ramps (MOM_dynamics_split_RK2.F90:1613-1618).
  - `mandatory=.true.` fields must be present.
  - `save_restart` (MOM_restart.F90:1572) writes a checksum attribute per field (1708-1751). `restore_state` (1791) recomputes and compares the checksums, FATAL if `checksum_required` (104, ~1925-1960).
- **Debug hooks.**
  - `CS%debug` gates `hchksum`/`uvchksum` (framework/MOM_checksums.F90:40, 60). These are bit-count checksums plus min/max/mean (`bc_modulus`, 112), computed over interior and halos.
  - Packaged versions are `MOM_state_chksum`, `MOM_thermo_chksum` and `MOM_accel_chksum` (core/MOM_checksum_packages.F90:44, 122, 201). They are called with stage labels such as "Pre-diabatic", "Post-ALE" and "pre-btstep accel" around every major call (MOM.F90:1761-1799, 1907-2010; MOM_dynamics_split_RK2.F90:589, 710, 927, 979).
  - `check_redundant` compares duplicated halo and edge values. `callTree_enter/waypoint/leave` (framework/MOM_error_handler.F90:249) prints the call tree when `showCallTree`. Every call is wrapped in `cpu_clock_begin/end`.
  - `MOM_state_stats` is used under `debugConservation` in the diabatic driver.
  - Index rotation (`ROTATE_INDEX`, MOM.F90:2912) re-runs the model on a rotated grid to catch x/y asymmetry bugs. Answers are kept bit-reproducible across versions with `*_ANSWER_DATE` parameters.

## 6. Lessons for pyOCEAN (Python/Numba now, JAX later)

**Copy**

1. **Components return tendency arrays; the driver sums them** in a fixed, bracketed order (`u_bc_accel = (CA + PF) + diff`). Make each N1 component a pure function `(state, grid, params, forcing) → tendency`. The ordered sum belongs to the driver. This gives budgets for free and matches JAX style.
2. **Keep implicit operators separate from explicit tendencies.** Vertical viscosity and vertical diffusion are in-place implicit solves. Their budget term is diagnosed as (after − before)/dt into a buffer that is allocated only when requested (MOM_vert_friction.F90:671, 836, 3093). pyOCEAN's θ_bc/implicit-Kv steps fit this exactly.
3. **Use explicit sub-cycle clocks** (`t_dyn_rel_adv`, `t_dyn_rel_thermo`) and the rates DT ≤ DT_TRACER_ADVECT ≤ DT_THERM, with tracer advection on accumulated `uhtr`. Even if pyOCEAN keeps one rate now, design the driver with the three slots so that ALE/diabatic at DT_THERM is a configuration choice, not a rewrite. Note that MOM6 regrids at DT_THERM, not every DT.
4. **Use a tracer registry with generic advect/remap/sponge/restart/diagnostic loops,** and treat T and S as registered tracers. pyOCEAN's `n_tracers` already does this implicitly. Make it explicit: name, units, CF name, advection scheme, restart flag and tendency diagnostics per tracer. This merges naturally with the planned single variable registry.
5. **Register restart fields per component, then lock.** Every component that has history (AB levels, `CAu_pred`, `u_av`, GLS state, T_ext) registers it. Add "query_initialized → compute default" and per-field checksums on read. pyOCEAN's restart config check is the equivalent; turn it into per-component registration.
6. **Make parameters self-documenting:** name, value, default, units (with conversion at read time) and description. Write a `.all`/`.short` doc of every parameter used, mark non-defaults, and flag unused keys. In Python this maps to typed dataclasses with field metadata, plus a dump of "differs from default".
7. **Allocate diagnostics lazily:** a registry id is −1 unless requested, and a buffer exists only if posted.
8. **Make staged state checksums a debug flag,** labelled "Pre-X"/"Post-X" around each component. This makes bit-for-bit regressions after N1 trivial to localise. Also consider a rotation/transpose test.
9. **Remap the auxiliary dynamic state** (`u_av`, `CAu_pred`, `diffu`) after ALE (MOM_dynamics_split_RK2.F90:1330-1357). Any multistep history in pyOCEAN (AB2/AB3 tendencies) must be remapped or invalidated when h changes.

**Avoid**

1. **A god struct.** `MOM_control_struct` holds about 40 sub-CS pointers plus state plus clocks. Separate (a) an immutable grid, (b) frozen per-component params, (c) the state pytree, and (d) a diagnostics sink. Mutable `CS` objects passed everywhere block JAX.
2. **Hidden state inside component CSs.** Examples: `vertvisc_coef` stores coefficients in its CS for `vertvisc` to use; `CAu_pred_stored`; the barotropic CS keeps the barotropic state. In pyOCEAN, every piece of carried state should live in the state object so restart, remap and JAX `scan` see it.
3. **Hand-coded dispatchers** (`MOM_tracer_flow_control` with one `if` per package and init order hard-coded in `initialize_MOM`). Use a list of components that implement a protocol (`register`, `init`, `step`/`tendency`, `restart_fields`, `diagnostics`, `end`).
4. **Scattered compatibility checks.** Use one declarative validation pass (requires/conflicts per component) at pre-flight, reporting all conflicts at once instead of the first FATAL.
5. **`inout` everything.** `diabatic` mutates u, v, h, T, S, visc and BLD. Keep in-place updates inside Numba kernels, but make component boundaries return new arrays or explicitly named outputs.
6. **Do not adopt dimensional rescaling (`US`) or halo, pass and clock bookkeeping** in N1. They are MPI/Fortran concerns. Keep units in metadata instead.
