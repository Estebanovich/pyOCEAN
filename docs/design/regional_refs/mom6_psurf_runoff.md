# MOM6 surface pressure, river runoff and vector rotation in the coupler: reference read for pyOCEAN

Sources (read 2026-11-01, regional chain plan Tasks 4, 7 and 8):
- MOM6 `~/pyOCEAN_refs/MOM6`, git `0e9a561` (2026-10-07). `src/...` paths relative to
  `~/pyOCEAN_refs/MOM6/src/`, `FMS_cap/...` relative to `~/pyOCEAN_refs/MOM6/config_src/drivers/FMS_cap/`.
- SIS2 `~/pyOCEAN_refs/SIS2`, git `c287b24` (2026-05-29), paths relative to `SIS2/src/`.
- FMScoupler `~/pyOCEAN_refs/FMScoupler`, git `5b700a4` (2025.02). CEFI builds coupler branch 2025.03
  (`CEFI-regional-MOM6/.gitmodules`); `~/pyOCEAN_refs/FMScoupler_cefi` is `175a9da` (2025.04).
- NWA12: `~/pyOCEAN_refs/CEFI-regional-MOM6`, git `36b8f49`, `exps/NWA12.COBALT/`
  (`MOM_parameter_doc.all` cited `.all:line`, `SIS_parameter_doc.all` cited `SIS.all:line`,
  `data_table`, `input.nml_24hr`).

Reference implementations, not truth; disagreements flagged.

---

## 1. NWA12 answers (brief)

| question | NWA12 | source |
|---|---|---|
| Atmospheric pressure on the ocean? | **No, twice over.** SIS2 `APPLY_SLP_TO_OCEAN = False`, so the pressure it passes to MOM6 is only the ice/snow weight; and `MAX_P_SURF = 0.0` clips that to zero | `SIS.all:59`, `SIS2/src/ice_model.F90:779-784`; `.all:2474` |
| Is ERA5 `msl` used at all? | Yes, by the coupler bulk formula only: `ATM p_surf ← msl` (`data_table:1`) feeds air density and q_sat on the exchange grid | `FMScoupler/full/atm_land_ice_flux_exchange.F90:896, 1036, 1828, 1897` |
| Runoff carries heat? | **No own heat.** `USE_RIVER_HEAT_CONTENT = False`; the land is `land_null` and `runoff_hflx` is not overridden in `data_table`, so runoff enters at the local SST | `.all:2103`; `src/core/MOM_forcing_type.F90:915-919` |
| Runoff salinity | 0 (pure fresh water) | `src/parameterizations/vertical/MOM_diabatic_aux.F90:1035, 1112-1114` |
| Depth spreading | **None in the volume/tracer budget.** The water goes into the top layer(s) through the normal freshwater path. `DO_RIVERMIX = True`, `RIVERMIX_DEPTH = 5.0 m` only adds turbulent kinetic energy to ePBL to mix the river water over ~5 m | `.all:2098, 2101`; `MOM_diabatic_aux.F90:1051-1072` |
| Freshwater aggregation | `AGGREGATE_FW_FORCING = True`, `MINIMUM_FORCING_DEPTH = 0.001 m`, `EVAP_CFL_LIMIT = 0.8` | `.all:1822, 1843, 1848` |
| p_surf interpolation in time | `INTERPOLATE_P_SURF = False` | `.all:114` |
| Wind stress staggering | ocean `WIND_STAGGER = "C"`; SIS2 `ICE_OCEAN_STRESS_STAGGER = "C"`, `CGRID_ICE_DYNAMICS = True` | `.all:2509`; `SIS.all:6, 19` |

## 2. Surface pressure

**From the coupler to MOM6** (`FMS_cap/MOM_surface_forcing_gfdl.F90`). `IOB%p` (the ice-ocean
boundary pressure) is copied to `p_surf_full` and limited (`:613-631` for the thermodynamic
`fluxes`, `:834-852` for the mechanical `forces`):

```
p_surf_full = mask2dT * IOB%p
p_surf      = min(p_surf_full, MAX_P_SURF)     if MAX_P_SURF >= 0
p_surf      = p_surf_full                      if MAX_P_SURF < 0  (default -1)
```

`MAX_P_SURF` is documented as a guard because "the FMS coupling structure does not limit the water
that can be frozen out of the ocean" (`:1438-1444`). **`MAX_P_SURF = 0` makes `p_surf ≡ 0`** wherever
`IOB%p ≥ 0`. `p_surf_full` keeps the unclipped value and is still used for the rigid-ice viscosity
(`USE_RIGID_SEA_ICE = True` in NWA12; `:934-955`), and, because `USE_LIMITED_PATM_SSH = True`
(`.all:2501`), the SSH returned to the coupler is corrected with the *limited* `p_surf`
(`p_surf_SSH => p_surf`, `:309-315, 786-792`).

**What `IOB%p` is in NWA12.** SIS2 builds it in `set_ocean_top_dyn_fluxes`
(`ice_model.F90:779-784`):

```
p_surf = slp2ocean ? (p_atm_surf − 1e5) : 0      ! SLP − 1 standard atmosphere
p_surf = p_surf + g * m_ice                      ! weight of ice + snow + ponds
```

with `APPLY_SLP_TO_OCEAN` default False (`ice_model.F90:1971-1973`; NWA12 `SIS.all:59`). So in NWA12
the atmospheric pressure never reaches MOM6, and the ice weight that does reach it is removed by
`MAX_P_SURF = 0`. The spec's "MAX_P_SURF = 0 recorta la presión a cero" is right but incomplete:
even with `MAX_P_SURF < 0`, NWA12 would still have no atmospheric pressure unless SIS2's
`APPLY_SLP_TO_OCEAN` were also set. If pyOCEAN turns `atm_pressure` on in a later project, the
MOM6 counterpart needs both switches, and SIS2 subtracts 1e5 Pa (not the domain mean) before
passing it.

**How `p_surf` enters the dynamics when it is non-zero.**
- In the split RK2 step, `p_surf => forces%p_surf` (or the time-interpolated `p_surf_end` when
  `INTERPOLATE_P_SURF`; `src/core/MOM_dynamics_split_RK2.F90:452-459`, `src/core/MOM.F90:832-842,
  975-981`) is passed to `PressureForce` (`MOM_dynamics_split_RK2.F90:513-514, 847-848`).
- Boussinesq FV PGF: the pressure at the top interface is
  `pa(1) = ρ_ref g (e(1) − Z_ref) + p_atm` (`src/core/MOM_PressureForce_FV.F90:1250-1260`), so
  every layer feels `−∇p_atm/ρ₀` through the integrals; with `SSH_IN_EOS_PRESSURE_FOR_PGF` the
  reference height for the EOS pressure also includes `p_atm/(ρg)` (`:1262-1275`; NWA12 False).
  Non-Boussinesq: `p(1) = p_atm` (`:298-302`).
- Barotropic solver: `btstep` receives the layer accelerations that already contain `∇p_atm` and
  the `eta_PF` consistent with them; it only adds the correction
  `(eta_BT − eta_PF)·gtot` to the depth-mean pressure force (`src/core/MOM_barotropic.F90:3304-3313`).
  With `INTERPOLATE_P_SURF` the change of p_surf over the step is passed as
  `eta_PF_start = eta_PF − (p_surf_begin − p_surf_end)/(g ρ₀)` (`MOM_dynamics_split_RK2.F90:515-521`)
  and interpolated linearly in barotropic time (`MOM_barotropic.F90:1061-1073`).
- So MOM6 treats p_atm exactly like a surface-height anomaly `p_atm/(ρ₀ g)` in both modes.
  pyOCEAN's plan (Task 7: `−∇p_a/ρ₀` added to the depth-mean momentum like ∇η) is consistent with
  this for a Boussinesq hydrostatic model, because a horizontally varying p_atm is depth-independent.

## 3. River runoff

**Path of the data in NWA12.**
1. `data_table:21` overrides `"ICE", "runoff"` from `INPUT/glofas_runoff_1993.nc` (variable
   `runoff`, kg m⁻² s⁻¹, written on the ocean T grid by CEFI's `write_runoff_glofas.py`, see
   `cefi_tools.md` §5), interpolation `"none"`, scale 1.0. `"calving"` is 0 (`data_table:22`).
   The override is applied to the land→ice boundary (`FMScoupler/full/land_ice_flux_exchange.F90:136-139`).
2. SIS2 copies it to its fast-ice average on ocean cells only (`ice_model.F90:401-419`) and passes it
   on unchanged (`ice_model.F90:660-663`).
3. The coupler moves it to the ice-ocean boundary (`FMScoupler/full/ice_ocean_flux_exchange.F90:292-293`).
4. MOM6: `fluxes%lrunoff = kg_m2_s_conversion · IOB%runoff · mask2dT`
   (`FMS_cap/MOM_surface_forcing_gfdl.F90:502-506`); `frunoff` likewise from calving (`:508-512`).

**Volume source.** `extractFluxes1d` sums
`netMassInOut = dt · (lprec + fprec + evap + lrunoff + lrunoff_glc + vprec + seaice_melt + frunoff + frunoff_glc)`
(`src/core/MOM_forcing_type.F90:638-648`). With `AGGREGATE_FW_FORCING = True` all of it is treated
as one net flux: `netMassOut = netMassInOut`, `netMassIn = 0` (`MOM_diabatic_aux.F90:990-996`).
Then, layer by layer from the top (`:1094-1162`):

```
IforcingDepthScale = 1 / max(H_subroundoff, minimum_forcing_depth − netMassOut)
fraction = min(1, h_k · IforcingDepthScale)            ! limited by EVAP_CFL_LIMIT for losses
dh_k     = max(fraction · netMassOut, −h_k)
dT_k     = fraction · netHeat + dh_k · T_k · EnthalpyConst
dS_k     = fraction · netSalt                          ! no salt with the water
T_k, S_k = (h_old·T_k + dT_k)/h_new, (h_old·S_k + dS_k)/h_new
```

For a net gain (rivers dominate) `minimum_forcing_depth − netMassOut < 0`, so the scale is
`1/H_subroundoff`, `fraction = 1` and **all the water goes into the top layer**. The water carries
the temperature of that layer (`EnthalpyConst = 1` because the FMS cap does not allocate
`heat_content_evap`: `MOM_forcing_type.F90:3605-3606, 3645`, `FMS_cap/MOM_surface_forcing_gfdl.F90:300-302`,
`MOM_diabatic_aux.F90:835-836, 1122`) and no salt. `heat_content_lrunoff = C_p · lrunoff · SST`
is then only a diagnostic (`MOM_forcing_type.F90:915-919`; it is added to `net_heat` only when
`do_enthalpy` is false, `:788-793`). Net effect: runoff is a fresh-water volume source at the local
SST, which is what pyOCEAN's N2a freshwater path already does (fw > 0 into the top layer with S = 0
and T of the layer). The spec's assumption is confirmed.

Note: when a net *loss* occurs, `EVAP_CFL_LIMIT` spreads it downward (`:1104-1107`); for rivers this
never triggers.

**`DO_RIVERMIX`** (`MOM_diabatic_aux.F90:1051-1072`; parameters `:1531-1538`): only with ePBL
(`calculate_energetics` requires `cTKE`, `:825`; ePBL path `MOM_diabatic_driver.F90:1607-1612`).
It adds to the top-layer TKE

```
cTKE += max(0, −½ RIVERMIX_DEPTH · dt · g · ρ₀ · dSV/dS · (lrunoff + frunoff + glc terms) · S₁)
```

i.e. the potential energy needed to mix fresh river water of salinity 0 with ambient salinity S₁
over `RIVERMIX_DEPTH`. It does not move water or salt itself; ePBL decides the mixing. pyOCEAN has
no ePBL; with KPP there is no counterpart (class b/c, to note in the parity list). A crude pyOCEAN
equivalent would be to deliver the river volume over the top 5 m, which is a different (volume)
mechanism and would not be parity.

## 4. Where wind stress and surface currents are rotated

1. **Atmosphere → exchange grid (no rotation).** With `atmos_null`, `u_bot`, `v_bot` come from ERA5
   `u10`, `v10` by `data_override` (`data_table:5-6`, bicubic) and are east/north components. The
   coupler computes stresses on the exchange grid in lat/lon using the relative wind
   `u_atm − u_surf` (`FMScoupler/full/atm_land_ice_flux_exchange.F90:1062, 1297, 1988, 2030`);
   there is no rotation anywhere in FMScoupler (`grep -i rotat` in `full/`, `shared/` finds none).
2. **Lat/lon → ocean grid: SIS2.** `avg_top_quantities` rotates the averaged stresses
   (`SIS_fast_thermo.F90:241-252`):
   `x = u·cos_rot − v·sin_rot`, `y = v·cos_rot + u·sin_rot` ("rotate stress from lat/lon to ocean
   coordinates"), also flipping the sign if the stress comes from the atmosphere's convention
   (`:243`). SIS2's `cos_rot`, `sin_rot` are MOM6's corner-based angle
   (`SIS2/src/SIS_fixed_initialization.F90:127` → `MOM_shared_initialization.F90:571-626`; see
   `mom6_grid.md` §6 for the sign: `sin_rot ≈ −sin(angle_dx)`).
3. **A-grid → C-grid: SIS2.** `set_wind_stresses_C` (`SIS_dyn_trans.F90:2040-2164`) averages the two
   T-point stresses adjacent to each face, weighted by `areaT · ice_free`:
   `τx(I,j) = mask2dCu · (A_i f_i τ_i + A_{i+1} f_{i+1} τ_{i+1}) / (A_i f_i + A_{i+1} f_{i+1})`
   (`:2117-2135`; v analogous `:2138-2150`), and `set_ocean_top_stress_Cgrid` adds the ice-ocean
   stress weighted by ice fraction (`:1613-1783`). The result goes to MOM6 as C-grid stress
   (`FMS_cap/MOM_surface_forcing_gfdl.F90:1100-1117`, `taux = mask2dCu · u_flux`).
   pyOCEAN N2b takes τ to faces as the mean of the wet neighbours (MOM6 A-grid rule,
   `MOM_surface_forcing_gfdl.F90:1086-1097`), not area-weighted; equal on a uniform grid, different
   by O(Δ area) on a curvilinear one (class c).
4. **Ocean grid → lat/lon for the surface current (relative wind): SIS2.**
   `ice_model.F90:1214-1219`: `u = x·cos_rot + y·sin_rot`, `v = y·cos_rot − x·sin_rot`, then the
   coupler puts `Ice%u_surf` on the exchange grid (`atm_land_ice_flux_exchange.F90:1062`).

**Consequence for pyOCEAN Task 4.** The plan rotates "in the provider after the bilinear step";
MOM6/SIS2 rotate the **stress** (computed in lat/lon with the current rotated back to lat/lon), not
the wind. For a bulk formula that is rotation-invariant (|ΔU| and the direction of ΔU), rotating the
wind and the current to the grid first and computing the stress in grid coordinates gives the same
stress to round-off; the order only matters for round-off and for the A→C averaging weights.
The angle used must be MOM6's corner angle for parity (not `angle_dx`; `mom6_grid.md` §6).

## 5. Other bulk-relevant NWA12 coupler settings (for completeness)

`surface_flux_nml`: `ncar_ocean_flux_multilevel = .true.`, `bulk_zu = 10`, `bulk_zt = 2`,
`bulk_zq = 2`, `raoult_sat_vap = .true.` (`input.nml_24hr:141-147`); the multilevel call uses these heights,
not `z_bot` (`FMScoupler/shared/surface_flux.F90:407-412`). `dt_cpld = dt_atmos = 3600 s`;
`ocean_albedo_option = 2`; `rough_scheme = 'beljaars'`; `xgrid_nml interp_method = 'second_order'`;
`gust` constant 1e-4 (`data_table:8`). ERA5 shortwave is split 0.285/0.285/0.215/0.215 into
vis-dir/vis-dif/nir-dir/nir-dif (`data_table:15-18`, scale factors /3600 s).
