# KPP in MOM6 / CVMix: reference read for pyOCEAN N2c

Scope: how MOM6 (git 0e9a561, 2026-10-07) calls CVMix KPP (CVMix-src fce4221, 2026-05-04), the CVMix single-column
test cases (SCM) as configured in MOM6-examples, and what `physics/mixing/kpp.py` lacks to run on σ/ALE layers.
This is a reference implementation, not truth: where it differs from Large et al. (1994) (LMD94) it is flagged.

Paths (abbreviated):
- `KPP.F90` = `~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_CVMix_KPP.F90`
- `DIAB.F90` = `~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_diabatic_driver.F90`
- `cvmix_kpp.F90` = `~/pyOCEAN_refs/MOM6/pkg/CVMix-src/src/shared/cvmix_kpp.F90` (CVMix source IS present)
- `SCM` = `~/pyOCEAN_refs/MOM6-examples/ocean_only/CVmix_SCM_tests/`
- `FT.F90` = `~/pyOCEAN_refs/MOM6/src/core/MOM_forcing_type.F90`

## 1. Where KPP sits in the MOM6 time step

Which driver: the SCM tests set `USE_LEGACY_DIABATIC_DRIVER = True` (`SCM/common_KPP/MOM_input`), and `diabatic()` then calls
`diabatic_ALE_legacy` (`DIAB.F90:415-416`). Order inside it (`DIAB.F90`):

1. `set_diffusivity` -> `Kd_int` (interior, = 0 in the SCM, `KD = KV = KD_MIN = 0`) (729).
2. `Kd_heat = Kd_salt = Kd_int` (+ double diffusion if on) (747-762).
3. `calculateBuoyancyFlux2d` -> `KPP_buoy_flux(k=1..nz+1)`, `KPP_temp_flux` (= net heat minus SW), `KPP_salt_flux` (784; `FT.F90:1156`, `1028-1151`).
4. `find_ustar` -> `U_star` (788; `FT.F90:1187`).
5. `KPP_compute_BLD` (OBL depth, `KPP.F90:1062-1698`) then `KPP_calculate` (K and NLT, `KPP.F90:731-1055`) (792-803).
6. `KPP_get_BLD` -> `visc%MLD`, `visc%h_ML` (805-813); `Kd_int = min(Kd_salt, Kd_heat)` (818).
7. `KPP_NonLocalTransport_temp/_saln` applied **directly to tv%T, tv%S** with the surface flux, before the surface fluxes are applied (845-848; `KPP.F90:1860-1966`).
8. `applyBoundaryFluxesInOut` (surface heat/salt/penetrating SW into h, T, S) (991).
9. Implicit vertical diffusion: `ent_s(K) = dt * Kd_int(K) / (0.5 (dz(k-1)+dz(k)))` (`DIAB.F90:899-900`), then `triDiagTS_Eulerian` / `tracer_vertdiff_Eulerian` (1053-1059).
10. KPP viscosity: `Kv_shear += Kv_KPP` (passed as `visc%Kv_shear`, additive, `KPP.F90:1015-1021`), used by the implicit vertical viscosity in the dynamics step (not in the diabatic driver).

Consequences for pyOCEAN: the nonlocal term is an operator-split redistribution of the surface flux (step 7), not a
countergradient flux inside the tridiagonal; K is evaluated once per `DT_THERM` (1200 s in the SCM) from the state at the
start of the thermodynamic step.

## 2. Inputs to KPP and conventions

Signature `KPP_compute_BLD(CS,G,GV,US,h,Temp,Salt,u,v,tv,uStar,buoyFlux,...)` (`KPP.F90:1062`),
`KPP_calculate(CS,G,GV,US,h,tv,uStar,buoyFlux,Kt,Ks,Kv,nonLocalTransHeat,nonLocalTransScalar,...)` (731).

| Input | Definition | Ref |
|---|---|---|
| `h` | current ALE layer thickness; converted to `dz` (m) with `thickness_to_dz`; interfaces rebuilt from the surface down each call (`iFaceHeight(1)=0`) with optional `CVMix_ZERO_H_WORK_AROUND` floor (`min_thickness`, default 0) and `hcorr` carry | 818-826, 1240-1248 |
| `uStar` | `sqrt(tau_mag/rho0)`; for `WIND_CONFIG="const"` `u* = sqrt((|tau|+GUST_CONST)/RHO_0)` | `solo_driver/MOM_surface_forcing.F90:451-462`, `FT.F90:1187-1249` |
| `buoyFlux(k)` | k=1 surface buoyancy flux **including** the penetrating shortwave absorbed in the whole column (`netHeat = netHeatMinusSW + netPen(:,1)`, `MOM_forcing_type.F90:1113`; `buoyancyFlux(i,1) = -GoRho*(dRhodS*netSalt + dRhodT*netHeat)`, :1143; corrected 2026-10-30, this row said "excluding"); k=2..nz+1 = buoyancy flux carried by SW still unabsorbed at interface k (`-GoRho*dRhodT*netPen(i,k)`, :1147). Boussinesq: `-g/rho0 * (dRho/dT * netHeat + dRho/dS * netSalt)` | `FT.F90` (calculateBuoyancyFlux1d, tail) |
| sign | **positive buoyancy flux = stabilising (buoyancy gain)**. Warming gives `B>0` (stable); cooling `B<0` (convective). `lstable = surf_buoy > 0` | `cvmix_kpp.F90` (coeffs_kpp_low, `lstable`) |
| Effective `B_f` for the OBL search | per candidate depth k: `B_f(k) = buoyFlux(1) - buoyFlux(k+1)` (surface flux minus SW that has penetrated below the candidate depth) | `KPP.F90:1383` |
| `B_f` for K and NLT | `KPP_SHORTWAVE_METHOD="MXL_SW"` (default): `buoyFlux(1) - buoyFlux(int(kOBL)+1)`; `ALL_SW`: `buoyFlux(1)`; `LV1_SW`: `buoyFlux(1)-buoyFlux(2)` | 835-845 |
| `Temp, Salt` | layer values, potential T / practical S; EOS evaluated at **surface pressure** `p_surf` (not at depth) for the bulk-Ri density difference | 1396-1420 |
| `u, v` | C-grid; averaged to T points `0.5(u(I)+u(I-1))` | 1205-1206, 1344-1345, 1363-1364 |
| `Coriolis` | 4-point mean of `f` at corners, only used if `COMPUTE_EKMAN` | 1213 |
| `Kt, Ks, Kv` (in) | existing interior diffusivities; **ignored** unless `MATCH_TECHNIQUE=MatchBoth` (zeroed before calling CVMix, so no interior matching: "equivalent to matching to a zero diffusivity") | 846-855 |

Interface vs layer: K, NLT, N², buoyFlux are on interfaces (nz+1, index 1 = surface); T, S, u, v, h, Ri_b, Vt², w_s used in the
OBL search are on layer centres. `cellHeight(k) = iFace(k) - dh/2`.

## 3. OBL depth search (KPP_compute_BLD)

Per column, for **every layer k as a candidate depth** (assuming OBL depth = `-cellHeight(k)`):

1. Surface layer: `SLdepth = surf_layer_ext (eps=0.1) * max(max(|zc_k|, |iFace(2)|), MINIMUM_OBL_DEPTH)`; the reference T, S, u, v are
   thickness-weighted means over [0, SLdepth] (partial last layer, `delH = min(max(0, SL - hTot), dz)`) (1250-1262, 1331-1352, 1375-1378).
   So the reference is **not** layer 1 only; it follows the candidate depth.
2. `deltaU2(k) = (u_k - U_sl)^2 + (v_k - V_sl)^2` (1363-1388, 1389).
3. `deltaBuoy(k) = g/rho0 * (rho(Tk,Sk,p_surf) - rho(T_sl,S_sl,p_surf))` (1420-1428). Positive when denser than the surface layer.
4. `N2(k) = g/rho0 * (rho_k - rho_{k-1}) / (0.5 (h_{k-1}+h_k))` at the interface above layer k, with both densities at the same
   (surface) pressure; `N = sqrt(max(N2,0))`, `N(nz+1)=0` (1435-1447). CVMix uses `N_cntr(kt) = N_iface(kt+1)` (the interface **below** the layer, `cvmix_kpp.F90` unresolved_shear).
5. `Ws(k)`: scalar velocity scale at `sigma = min(eps, 1) = 0.1` of the candidate depth, `OBL_depth(k) = -zc_k`, using `B_f(k)`
   (`Vt_layer = 1.0` -> `sigma_loc = min(surf_layer_ext, 1)`) (1491-1498; `cvmix_kpp.F90` turbulent_scales_1d_OBL, `sigma_loc = min(surf_layer_ext, sigma_coord)`,
   `l_LMD_ws` default False, so the stable case also uses sigma = eps, unlike LMD94 where stable sigma = actual).
6. Unresolved shear (default branch, no Langmuir): `Vt2(k) = Cv * Vtc * |zc_k| * N_k * Ws(k) / Ri_crit`,
   `Vtc = sqrt(0.2/(c_s*eps)) / kappa^2`, `Cv = 2.1 - 200 N` if `N < 0.002` else `1.7`, floored at `MINIMUM_VT2 = 1e-10`
   (`cvmix_kpp.F90`, unresolved_shear `case DEFAULT`). `KPP_CVt2=1.6` is used **only** with `STOKES_MOST`.
7. Bulk Richardson (LMD94 eq. 21): `Ri_b(k) = (1 - eps/2) * |zc_k| * deltaBuoy(k) / (deltaU2(k) + Vt2(k))`
   (`cvmix_kpp_compute_bulk_Richardson`; the factor `1 - 0.5*surf_layer_ext = 0.95` is CVMix's). If the denominator is 0, `num*1e10`.
8. First layer where `Ri_b > Ri_crit (0.3)`; interpolate `Ri_b` (cubic by `INTERP_TYPE="cubic"`, quadratic is the default) between the
   two centres with the one above as third point; root of `Ri_b(z) = Ri_crit` via cubic root find. No crossing in the column: `OBL = |zc(nz)|`
   (`cvmix_kpp_compute_OBL_depth_low`). Clamps: `k=1` -> `max(OBL, |zc(1)|)`; `min(OBL, OBL_limit)`; Ekman and Monin-Obukhov limits are off in
   the SCM (`COMPUTE_EKMAN = COMPUTE_MONIN_OBUKHOV = False`); `max(OBL, MINIMUM_OBL_DEPTH=0)`.
9. MOM6 post-clamps (1550-1557): `OBL >= |iFace(2)|` (not shallower than the top layer), `OBL <= -zBottomMinusOffset`
   (`iFace(nz+1) + min(max(DEEP_OBL_OFFSET,0), 0.1 D)`, and not above `iFace(2)`), then `kOBL = CVMix_kpp_compute_kOBL_depth` =
   `k + 0.25` if OBL is in the upper half of layer k, `k + 0.75` if in the lower half (`cvmix_kpp.F90`, kOBL function).
10. Smoothing: `N_SMOOTH = 0` default, none in the SCM. If `>0`: 5-point (centre weight 1-4*0.125, neighbours 0.125) on `OBLdepth`,
    optional `DEEPEN_ONLY_VIA_SMOOTHING`, cap at total depth, then `kOBL` recomputed (`KPP.F90:1704-1815`, 1698). Horizontal only.
11. `FIXED_OBLDEPTH` option bypasses the search (1462-1466). `ER_depth`/`StokesMOST`, Langmuir (`USE_KPP_LT_K`, `USE_KPP_LT_VT2`) are
    off in the SCM and can be ignored for N2c.

Nothing carries over between steps (OBLdepth is restart-registered only for `FPMIX`, `KPP.F90:241-243`); `OBLdepthprev` is used only for the Langmuir MLD guess.

## 4. Diffusivity and nonlocal transport (KPP_calculate -> CVMix_coeffs_kpp)

Parameters of the effective profile (`MOM_parameter_doc.all` of the SCM; CVMix defaults in `cvmix_kpp.F90` init):
`kappa = 0.4`, `eps = 0.1`, `Ri_crit = 0.3`, `zeta_m = -0.2`, `zeta_s = -1`, `a_m = (1-16 zeta_m)^(-1/4) (1-4 zeta_m)`, `c_m = 12 (1-16 zeta_m)^(-1/4)`,
`a_s = sqrt(1-16 zeta_s)(1+8 zeta_s)`, `c_s = 24 sqrt(1-16 zeta_s) = 98.96`, `Cstar = 10`, `CS2 = nonlocal_coeff = Cstar kappa (kappa eps c_s)^(1/3) = 6.327`
(`cvmix_kpp.F90` init, 360-377, 576-584).

Velocity scales (`compute_phi_inv`, non-Stokes): `zeta = sigma_loc * OBL * B_f * kappa / u*^3`; stable `phi^-1 = 1/(1+5 zeta)`; unstable
`phi_m^-1 = (1-16 zeta)^(1/4)` for `zeta >= zeta_m`, else `(a_m - c_m zeta)^(1/3)`; `phi_s^-1 = (1-16 zeta)^(1/2)` for `zeta >= zeta_s`, else `(a_s - c_s zeta)^(1/3)`;
`w = phi^-1 kappa u*`. When `u* = 0`: stable -> 0; unstable -> `w_x = kappa (-c_x min(eps,sigma) OBL kappa B_f)^(1/3)` with `c_m`, `c_s`.
**LMD94's strongly convective cubic branch is implemented; pyOCEAN has only the 1/4 and 1/2 laws.**

Profile (`MATCH_TECHNIQUE="SimpleShapes"`, default and SCM): `K_x(sigma) = OBL * w_x(sigma) * G(sigma)`, `G(sigma) = sigma (1-sigma)^2` (cubic `[0,1,-2,1]`), same G for
momentum, heat and salt; **no matching to the interior at the OBL base** (G(1)=G'(1)=0) and `Mdiff_OBL`-terms vanish because the incoming K is zeroed.
`sigma = -zw(k)/OBL` at interfaces k = 2..kwup (`kwup = floor(kOBL)`); surface interface stays 0; interfaces below `ktup+1` keep the incoming interior K
(`ktup = nint(kOBL)-1`) (`cvmix_coeffs_kpp_low`, non-Stokes branch). Because `G(0)=0` and `G ~ sigma` while `w_x` is evaluated at the local sigma
(only capped at eps), K rises linearly from the surface, as in LMD94 but without the interior-matching terms (`a2, a3` with `Gat1, dGat1`).

Enhanced diffusion (`ENHANCE_DIFFUSION = True`): at the interface straddling the OBL base, with `delta = (OBL + zt(ktup)) / (zt(ktup) - zt(ktup+1))`,
`K_enh = (1-delta)^2 K_ktup + delta^2 K_OBL`, `K(ktup+1) = (1-delta) K_old + delta K_enh`, and the nonlocal coefficient is rescaled by `K_new/K_old` when `ktup = kwup-1`
(`cvmix_kpp_compute_enhanced_diff`, LMD94 eq. 3.10-style). Needed to reproduce MOM6's K profile near the OBL base.

Nonlocal transport (non-Stokes): only for `.not.lstable` (`B_f <= 0`): `NLT_x(k) = CS2 * G_NL(sigma(k))`, `G_NL = sigma(1-sigma)^2`; zero at k=1 (surface) and below the OBL.
`NLT_shape = "CVMix"` (default) keeps the CVMix result; the other `NLT_SHAPE` options (LINEAR, PARABOLIC, CUBIC, CUBIC_LMD) overwrite it only when `surfBuoyFlux < 0` (`KPP.F90:961-994`).
`MATCH_TECHNIQUE="ParabolicNonLocal"` would force `CS2 = 1` and `G_NL = (1-sigma)^2`. Heat and salt use the same shape.

Application (`KPP.F90:1899-1900`): `dT/dt|_NLT(k) = (NLT(k) - NLT(k+1)) / h(k) * F_T`, with `F_T` = `KPP_temp_flux` = **non-solar** surface heat flux in K m s-1
(`Q_nonsolar / (rho0 cp)`, `netHeatMinusSW`), `F_S` = `netSalt` including the FW dilution term `-S1 * netH`. `T += dt * dT/dt` once per call. The integral over the column is zero (NLT(1)=0 and NLT = 0 at the OBL base), so the term only redistributes.

Output K added to the interior (`KPPisAdditive=True`) or replacing it where non-zero (`KPP_IS_ADDITIVE=False`) (1015-1030). `KPP_ZERO_DIFFUSIVITY` zeroes K for testing (980). A negative K
aborts (936-954).

## 5. CVMix single-column tests (SCM)

Layout: `SCM/{wind_only,cooling_only,mech_only,skin_warming_wind}/{KPP,EPBL,BML}/`; every `KPP/MOM_input` is identical to `SCM/common_KPP/MOM_input`
(verified with `diff`); the case is defined by `MOM_override` only. **No reference output is stored**: only `zgrid.nc`/`zgrid.cdl` (diagnostic z grid:
35 levels 0.5-375 m, 36 interfaces) and `*.ipynb` notebooks that read `KPP/prog_z.nc`, `KPP/visc.nc` produced on an Intel/GFDL workstation
(`wind_only.ipynb` first cell). The notebooks compare BML vs KPP vs EPBL, they are not regression baselines. Reference runs must be generated locally.

Common configuration (`SCM/common_KPP/MOM_input`):
- Domain: 2x2 cells (`NIGLOBAL=NJGLOBAL=2`, `REENTRANT_Y`), Cartesian 2 deg, flat bottom `MAXIMUM_DEPTH = 400 m`, `NK = 400`, `ALE_COORDINATE_CONFIG = UNIFORM`, `REGRIDDING_COORDINATE_MODE = "Z*"`:
  **400 uniform layers of 1 m**, `MINIMUM_DEPTH = 0.5`, `MIN_THICKNESS = 0.001`.
- `SPLIT = False`, `DT = DT_THERM = 1200 s`, `DAYMAX = 5 d`, output every 2 h (`diag_table`: `prog`, `prog_z`, `surffluxes`, `visc`).
- Coriolis: `ROTATION = "betaplane"`, `F_0 = 1e-4 s-1`, `BETA = 0`. **Momentum is active** (inertial oscillations, shear); `CDRAG = 0`, `KV = 0`, `HBBL = 10`,
  `HARMONIC_VISC` on, `KHTR = 600` (irrelevant in a uniform 2x2 box).
- EOS: `LINEAR`, `RHO_0 = 1000`, `DRHO_DT = -0.255`, `DRHO_DS = 0.764`, `G_EARTH = 9.80616`, `C_P = 3992.10322329649`. Salinity 35 uniform (`SCM_L1_SALT = SCM_L2_SALT = 35`, `SCM_L2_DSDZ = 0`).
- Initial T (`SCM_CVMix_tests_TS_init`, `src/user/SCM_CVMix_tests.F90`): `T(zc) = max(4, SCM_L2_TEMP + SCM_L2_DTDZ * min(0, zc + SCM_TEMP_MLD))`, `SCM_TEMP_MLD = 0`, `SCM_L2_TEMP = 15`, `SCM_L2_DTDZ = 0.01` (so `T = 15 + 0.01 z`, i.e. 11 C at 400 m, colder downward; `N^2 = g * 0.255e-3 * 0.01 = 2.50e-5 s-2` uniform).
- Shortwave: none (`SW = 0`, `PEN_SW_NBANDS=1`, `PEN_SW_SCALE=0`, `VAR_PEN_SW` False): `buoyFlux(k>=2) = 0`, so MXL_SW/ALL_SW coincide and the penetrating-SW logic is **not exercised**.
- KPP block: `INTERP_TYPE = "cubic"`, `ANSWER_DATE = 20251231`, everything else default (`N_SMOOTH 0`, `RI_CRIT 0.3`, `VON_KARMAN 0.4`, `ENHANCE_DIFFUSION True`, `MATCH_TECHNIQUE SimpleShapes`, `INTERP_TYPE2 LMD94`, `NLT_SHAPE CVMix`, `APPLY_NONLOCAL_TRANSPORT True`, `KPP_IS_ADDITIVE True`, `KPP_SHORTWAVE_METHOD MXL_SW`, `MINIMUM_OBL_DEPTH 0`, `MINIMUM_VT2 1e-10`, `SURF_LAYER_EXTENT 0.1`, no Ekman/MO limit, no Langmuir).
- Interior mixing is zero: `KD = KD_MIN = KV = 0`, no shear mixing, no convection scheme. **All mixing below the OBL is numerical (ALE remap PLM) only**; diffusion is purely KPP.
- Diagnostics worth reproducing (`diag_table`): `KPP_OBLdepth`, `KPP_BulkRi`, `KPP_Ws`, `KPP_N2`, `KPP_Vt2`, `KPP_BulkUz2`, `KPP_BulkDrho`, `KPP_Kheat/Ksalt/Kv`, `KPP_NLT*`, `KPP_Tsurf/Usurf`, `MLD_003`, `Kd_interface`, `temp`, `u`, `v`.

Per case (differences from the common file are only in `MOM_override`):

| Case | Wind stress | u* (m/s) | Gust | Surface heat (`SENSIBLE_HEAT_FLUX`) | B_f (m2/s3, + = stable) | Initial T | Tests |
|---|---|---|---|---|---|---|---|
| `wind_only` | `CONST_WIND_TAUX = 0.1 Pa` (`WIND_CONFIG const`) | 0.01 (`sqrt(0.1/1000)`) | 0 | none (`BUOY_CONFIG zero`) | 0 | `15 + 0.01 z` (N2 = 2.5e-5) | shear-driven deepening against stratification; inertial rotation (f = 1e-4); Ri_b from resolved shear + Vt2; **no nonlocal term** |
| `cooling_only` | none (`WIND_CONFIG zero`) | **0** (needs the u* = 0 convective-limit branch) | 0 | -100 W m-2 (`BUOY_CONFIG const`; `sens` is into the ocean, `FT.F90:115` "typically negative") | about -6.26e-8 (`-g/rho0 * 0.255 * 100/(rho0 cp)`) | `15 + 0.01 z` | pure convection: w_s from `(-B_f sigma OBL)^(1/3)`, nonlocal term, Vt2 with `w_s` only; no shear |
| `mech_only` | none | `sqrt(0.05/1000) = 7.07e-3` (`GUST_CONST = 0.05 Pa`, `ustar = sqrt(gust/Rho0)`) | 0.05 Pa | none | 0 | `15 + 0.01 z` | mixing by u* only, **no momentum source** (no wind stress, so no shear; only the u* in w_s), tests the Vt2 and `w_s` neutral-stable branch |
| `skin_warming_wind` | `CONST_WIND_TAUX = 0.1 Pa` | 0.01 | 0 | **+100 W m-2** | about +6.26e-8 (stable) | isothermal 15 C (`#override SCM_L2_DTDZ = 0.0`) | stable surface forcing: `1/(1+5 zeta)`, shallow OBL, nonlocal = 0, skin warming |

Expectations to check against: `wind_only` and `skin_warming_wind` need the full momentum equation (rotation + implicit vertical viscosity with KPP Kv), so the
test is a coupled column (T, S, u, v) in a 1-D box, not KPP alone.

How to run (nothing needs building):
`~/pyOCEAN_refs/MOM6/.testing/build/symmetric/MOM6` (30 MB, built 2026-10-11, linked with `-lcvmix`, contains `MOM_CVMix_KPP.o`, `SCM_CVMix_tests.o`) and an `repro/` build.
Copy `SCM/common_KPP/{MOM_input,input.nml,diag_table,zgrid.nc}` plus the case `MOM_override` into a run directory, create `INPUT/` and `RESTART/`,
`mpirun -n 1 .../MOM6 > stdout.txt` exactly as `validation/mom6_lock_exchange/run_mom6.sh` (MPICH hangs in `MPI_Finalize`; wait for the final timing table, then stop). Output `prog.nc`, `prog_z.nc`, `visc.nc`, `surffluxes.nc`.
Cost: 400 layers x 4 columns x 360 steps, seconds. (`~/pyOCEAN_refs/build/column/` holds only the remap/regrid/hybgen drivers.) The binary's source tree is the one read here
(`git` 0e9a561 for MOM6; its answers can differ at 1e-12 from older published SCM runs, `ANSWER_DATE=20251231` pins the arithmetic vintage).

Additional unit-level reference: `~/pyOCEAN_refs/MOM6/pkg/CVMix-src/reg_tests/kpp/` (`input.nl`, `kpp-test.sh`, NCL plots; baselines under `CVMix-src/baselines/kpp` if populated) tests `cvmix_coeffs_kpp` on fixed profiles, a cheaper target for component-level checks (bulk Ri, K shape, enhanced diffusion).

Open point: the SCM `skin_warming_wind` override is written `#override SCM_L2_DTDZ = 0.0` (MOM_override syntax), and `MOM_parameter_doc.short` indeed omits the parameter, consistent with an isothermal start; confirm in `stdout`/`MOM_parameter_doc.all` of the first run.

## 6. ALE-specific notes for N2c

- KPP runs on the **current** layers after the Lagrangian/regrid step has produced `h`; it never sees fixed z levels. All geometry is rebuilt from `h` each call, bottom-up in index but accumulated **from the surface** (index 1 = top). pyOCEAN's σ model has k = 0 at the **bed**: reverse the column (or accumulate depth from the top) before calling.
- No special treatment of vanished layers other than `CVMix_ZERO_H_WORK_AROUND` (`min_thickness`, default 0): `dh = max(dz + hcorr, min_thickness)` with a carried error `hcorr` so interface depth is preserved (1240-1248). In the SCM thickness is 1 m so this never acts; in hybrid/σ layers with `h -> 0` set a non-zero value or guard N2 and Ri_b (`N2` divides by `0.5(h_{k-1}+h_k) + H_subroundoff`).
- `OBL >= |iFace(2)|` (top-layer thickness) and `OBL <= D - offset` keep the search inside the grid; with σ-layers of 0.5-2 m near the surface the cubic interpolation of Ri_b between layer centres is the only sub-grid refinement.
- `kOBL`'s 0.25/0.75 quantisation and the enhanced-diffusion blend are what make K continuous when OBL moves through a layer; any reimplementation that rounds OBL to an interface gives a staircase in K and in the nonlocal term.
- No horizontal smoothing in the SCM; for regional runs `N_SMOOTH` (default 0) and `DEEPEN_ONLY_VIA_SMOOTHING` exist (1704-1815, uses 5-point stencil with land mask).
- OBL depth feeds `visc%MLD`/`h_ML` (mixed-layer restratification, vertical viscosity); no equivalent consumers in pyOCEAN yet.
- Depth-dependent EOS is **not** used in the OBL search (all densities at `p_surf`); keep the same choice (also removes the compressibility term from `N2` in a σ model).

## 7. pyOCEAN `physics/mixing/kpp.py` (769 lines) and `tests/test_kpp.py`: what exists

Implemented (Phase 1, z-levels, array layout `(nz, ny, nx)` with k = 0 at the **surface**, `z_t < 0`, 1-D `dz(nz)`, 1-D `z_f(nz+1)`; velocities `u (nz,ny,nx+1)`, `v (nz,ny+1,nx)`):

| Function | What | Ref |
|---|---|---|
| `_velocity_scales` | LMD94 App. B: stable `1/(1+5 zeta)`, unstable 1/4 and 1/2 laws, clipped to `[1e-10, 0.5]` | kpp.py:145-204 |
| `obl_depth` | `Ri_b` per layer, linear interpolation (not cubic), reference = **layer 0**, no `(1 - eps/2)`, `Vt2 = 1.8 d N w_s / Ri_cr` | 214-333 |
| `interior_diffusivity` | LMD94 shear (`nu0 f(Ri)`, `Ri0 = 0.7`), `K_bg = 1e-5`, `K_conv = 1e-2` | 336-425 |
| `_blmix` | `G = 3 g sigma^2 - 2 g sigma^3` (matched to interior, zero slope), `w` at `eps h_bl`, `gamma = C_G / (w_s h_bl)` times surface flux, only if `b_flux > 0` | 428-552 |
| `kpp_tendency` | explicit tendency of T, S, u, v on a `ModelState`; `u* = sqrt(|tau|/1025)`; `alpha_T = 2e-4`, `beta_S = 7.4e-4` hard-coded | 555-769 |

Tests (`tests/test_kpp.py`, 32): shapes/positivity/uniform-state invariance, stable vs unstable scales, an analytic single-layer `Ri_b`, conservation with no flux, smoothing,
heating gives nonlocal warming. None compares with CVMix/MOM6, none has `u* = 0` or a deep convective case.

## 8. Gap list: pyOCEAN kpp.py vs MOM6/CVMix, and what is needed on σ/ALE layers

Numbering = suggested order of work.

**A. Sign and conventions (fix first; likely bugs)**
1. Buoyancy-flux sign. kpp.py: `b_flux = g(alpha wT - beta wS)` with `wT = Q_net/(rho0 cp)` and Q positive into the ocean, but the docstring and `zeta = -kappa d b_f/u*^3`, `unstable = b_flux > 0`, `gamma` all treat `b_flux > 0` as **destabilising**. Heating then produces `b_flux > 0` -> treated as convective. MOM6/CVMix: positive = stable. Define one convention (recommend MOM6's: `B_f = -g/rho0 (drho/dT F_T + drho/dS F_S)`, with F_T = Q/(rho0 cp)) and make the nonlocal sign follow `lstable`. `test_surface_heating_produces_nonlocal_warming` currently enshrines the wrong physics.
2. `RHO_0 = 1025`, `G = 9.81`, `CP`, `alpha_T/beta_S` are module constants; the SCM needs `rho0 = 1000`, `g = 9.80616`, `C_p = 3992.103`, and the EOS used by the model (linear: `drho/dT = -0.255`). Take them from the model/EOS component and compute `drho/dT, drho/dS` from it (MOM6 uses the local EOS derivatives at the surface T, S).
3. Surface-flux inputs: KPP needs `u*` (not only tau), the **non-solar** heat flux, the FW-diluted salt flux and the SW profile (`buoyFlux(k)`, k >= 2) separately; kpp.py takes a single `Q_net`.

**B. OBL depth**
4. Surface-layer reference: average T, S, u, v over `eps * max(|zc_k|, |zface_2|)` per candidate depth (kpp.py uses layer 0).
5. Bulk-Ri factor `(1 - eps/2)`, `B_f(k) = B(1) - B_SW(k+1)` per candidate (penetrating SW), `Ws` evaluated at `sigma = min(eps, 1)` of the candidate depth with `B_f(k)` (kpp.py: `w_s` at depth d, no eps).
6. Unresolved shear: `Cv = 1.7` (2.1 - 200 N for N < 0.002), factor `Vtc = sqrt(0.2/(c_s eps))/kappa^2 = 0.8885` (kpp.py: `C_V = 1.8`, no Vtc, i.e. about +19 % bigger Vt2 plus the missing 0.95), `N` taken at the interface **below** the layer, `Vt2 >= 1e-10`. (Check LMD94 eq. 23 `C_v` against the paper before choosing; CVMix is a reference, not truth.)
7. Cubic (or at least documented) interpolation of `Ri_b` for the crossing; clamps `OBL >= |zface_2|`, `OBL <= D - 0.1 D offset`, optional Ekman/MO limits; `kOBL` as 0.25/0.75 fractional index.
8. N² from densities at a common (surface) pressure with `0.5 (h_{k-1}+h_k)` spacing and guards for vanished layers (kpp.py uses `dz_c` of 1-D centres and extrapolates the bottom value).

**C. Velocity scales**
9. Add the strongly convective cubic branch (`zeta < zeta_m = -0.2` for momentum, `zeta < zeta_s = -1` for scalars, `a_m, c_m, a_s, c_s`), the `u* = 0` limits (zero if stable, `kappa (-c B sigma OBL kappa)^(1/3)` if unstable; **required by `cooling_only`**), and remove the 0.5 m/s clip and the `u*^3 + EPS` regularisation.

**D. Profile and nonlocal term**
10. Profile shape: CVMix SimpleShapes `G = sigma (1-sigma)^2` with `w(sigma)` evaluated at each interface (`sigma_loc = min(eps, sigma)`) and **no interior matching** (kpp.py: `G = 3 g sigma^2 - 2 g sigma^3`, `w` at `eps h_bl`, matching `K_int` at the base). Keep matching as an option; MOM6 never matches. Option set to replicate: `MATCH_TECHNIQUE` (SimpleShapes first), `NLT_SHAPE`.
11. Enhanced diffusion at the interface straddling the OBL base (`delta` blend) and `kOBL`-based `kwup/ktup` interface selection.
12. Nonlocal term: replace `K gamma wphi` inside the flux by the CVMix form `NLT(k) = CS2 sigma (1-sigma)^2` (`CS2 = 6.327`, zero when `B_f >= 0`), applied as a **separate redistribution** `dT = dt (NLT(k)-NLT(k+1))/h_k * F_T` with the non-solar surface flux, once per step, before the surface flux is applied; conserved by construction. Apply to S with `F_S` (diluted) and, as an option, `MXL_SW` buoyancy flux for the stable/unstable test.
13. Output K on interfaces for **momentum, heat and salt separately** (kpp.py returns `K_m, K_s`; CVMix gives `Kv`, `Kheat`, `Ksalt`); additive to the interior K (`KPP_IS_ADDITIVE`), interior shear/convective terms optional (the SCM sets them to zero, the existing `interior_diffusivity` would add `K_bg = 1e-5` and spoil the comparison, so make them separately switchable).

**E. Integration with the stepping**
14. Time stepping: kpp.py gives an explicit tendency (stability `dt <= h^2/2K`, violated for K ~ 1e-2 m2/s with 1 m layers at 1200 s: `1/(2*1e-2)*1 = 50 s`). MOM6 solves implicit tridiagonals; pyOCEAN already has implicit tracer vertical diffusion (`K_v` implicit) and `FreeSurfaceVerticalViscosity`; N2c should only **provide interface `Kv`, `Kt`, `Ks` and the NLT coefficients**, and let the existing implicit solvers apply them (as GLS does through `SigmaState.closure`, `core/components/free_surface.py:59-74`, registry `closure.akv` `core/model/registry.py:51`).
15. Data layout: kpp.py works on `xr` `ModelState` with 1-D `dz`/`z_t`; the σ/ALE model has 3-D `h (k,j,i)` (k = 0 at bed), plain NumPy, Numba fast path (`SigmaState`). Needs: `h`-based interface/centre geometry per column (reverse k; carry `hcorr` min-thickness rule), C-grid `u, v -> T` with `h`-weighting (the σ model's face fractions) and wet/dry masks, a 3-D `u*` field (from the wind stress component and, in regional runs, the log-law stress), and a Numba column kernel (loop over (j,i) is independent; use the shared thread list in `core/model/threads.py`).
16. Component contract (N1): a `KPPClosure` component with `compute(state, forcing) -> {Kv, Kt, Ks, nlt_t, nlt_s, obl_depth, kobl}` before `AB3Momentum`/`TracerDiffusion`, the NLT redistribution as an explicit tracer tendency phase, restart variables tagged by component (`obl_depth` is not needed for restart in MOM6), diagnostics via the registry (`KPP_OBLdepth`, `KPP_BulkRi`, `KPP_Ws`, ...).
17. Salinity and EOS: the σ-3D core currently has a linear T-only EOS; `B_f` requires S and a surface-pressure EOS (roadmap stage N, "S + TEOS-10 + full surface fluxes in 3-D").

**F. Validation plan (suggested)**
- Unit level against `CVMix-src/reg_tests/kpp` (bulk Ri, OBL interpolation, G shape, enhanced diffusion) with the same inputs from `input.nl`.
- Column level: the four SCM cases with the local MOM6 binary (`.testing/build/symmetric/MOM6`), `NK=400` uniform 1 m layers, `DT=1200`, 5 d. Metrics from `visc.nc`/`prog_z.nc`: OBL depth(t), SST(t), `MLD_003`, T(z,t), u(z,t), Kd. Classify differences (a)-(d) as in CLAUDE.md. `mech_only` and `wind_only` check the stable/neutral branch and shear; `cooling_only` the convective limit with u* = 0 (needs gap 9); `skin_warming_wind` the stable branch.
- Then the same cases on a σ grid and on hybrid/ALE layers (non-uniform h) to test the layer-geometry handling (gaps 15, kOBL, min thickness).
- Literature check before coding (CLAUDE.md rule): local library via `scripts/litsearch.py "KPP" "nonlocal" "bulk Richardson"`; Large et al. 1994, Griffies et al. 2015 (CVMix/KPP in MOM6), van Roekel et al. 2018 (KPP in CESM/MOM6 revisions: `Cv`, `Ri_b` surface-layer averaging, "TKE-based" `eps`) **[verificar]**.

## 9. Where MOM6/CVMix deviates from LMD94 (decide explicitly)

- No matching of K to the interior at the OBL base with SimpleShapes (`G(1) = 0`), i.e. OBL base K is always zero before enhancement.
- `w_s` for stable forcing evaluated at `sigma = eps` (`l_LMD_ws = False`), where LMD94 uses the true sigma.
- `Ri_b` carries `1 - eps/2`; density difference at surface pressure; surface-layer average moves with the trial depth.
- Nonlocal coefficient `Cstar kappa (kappa eps c_s)^(1/3) = 6.327` with shape `sigma(1-sigma)^2` (LMD94 eq. 20 `G(sigma)` cubic with matching), and the `MOM6` option `NLT_SHAPE` can replace it.
- No Ekman/Monin-Obukhov limits by default in either the SCM or `MOM_input` of the SCM (LMD94 recommends both for stable forcing).
- Penetrating SW enters `B_f` through `buoyFlux(1) - buoyFlux(k+1)` (CVMix design, not in LMD94, which has a single surface flux).
