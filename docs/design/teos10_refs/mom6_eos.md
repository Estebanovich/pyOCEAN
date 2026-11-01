# MOM6 equations of state: reference read for pyOCEAN nonlinear EOS

MOM6 source: `~/pyOCEAN_refs/MOM6`, git `0e9a561` (`git -C ~/pyOCEAN_refs/MOM6 rev-parse --short HEAD`).
All `file:line` references are relative to `~/pyOCEAN_refs/MOM6/src/` unless stated otherwise
(`config_src/...` is relative to `~/pyOCEAN_refs/MOM6/`). NWA12 configuration:
`~/pyOCEAN_refs/CEFI-regional-MOM6/exps/NWA12.COBALT/MOM_parameter_doc.all` (CEFI repo git `36b8f49`).
Read 2026-10-31. This is a reference implementation, not truth; disagreements with papers or
with MOM6's own documentation are flagged.

---

## 1. WRIGHT_FULL (`equation_of_state/MOM_EOS_Wright_full.F90`)

**Reference.** Wright, D. G., 1997: An equation of state for use in ocean models: Eckart's formula
revisited. J. Atmos. Oceanic Technol., 14(3), 735-740 (`MOM_EOS_Wright_full.F90:986-990`).
"Full range" fit to UNESCO 1981: −2 < θ < 40 °C, 0 < S < 40 PSU, 0 < p < 1e8 Pa
(`:17-18`, `EoS_fit_range_Wright_full` `:374-390`). Originally coded by R. Hallberg (2000),
anomaly form 2018 (`:983-984`).

**Inputs (MKS inside the module).** T = potential temperature referenced to the surface [°C]
(`:77`), S = practical salinity [PSU] (`:78`), p = pressure [Pa] (`:79`). No CT/SA conversion is
done for Wright; if the model carried CT/SA it would be fed in unconverted (see §3).

**Coefficients** (`:23-37`, exact strings as in the code):

| a | value | b | value | c | value |
|---|---|---|---|---|---|
| a0 | 7.133718e-4 m³ kg⁻¹ | b0 | 5.613770e8 Pa | c0 | 1.609893e5 m² s⁻² |
| a1 | 2.724670e-7 m³ kg⁻¹ °C⁻¹ | b1 | 3.600337e6 Pa °C⁻¹ | c1 | 8.427815e2 |
| a2 | -1.646582e-7 m³ kg⁻¹ PSU⁻¹ | b2 | -3.727194e4 Pa °C⁻² | c2 | -6.931554 |
| | | b3 | 1.660557e2 Pa °C⁻³ | c3 | 3.869318e-2 |
| | | b4 | 6.844158e5 Pa PSU⁻¹ | c4 | -1.664201e2 |
| | | b5 | -8.389457e3 Pa °C⁻¹ PSU⁻¹ | c5 | -2.765195 |

These are Wright (1997) Table 1, second column (extended formula fitted over the full range,
rms ρ error 0.015 kg m⁻³), in the paper's own mks units (Pa, m³ kg⁻¹, m² s⁻²); all 15 checked
against the paper on 2026-11-01.

**Functional form** (Tumlirz/Eckart form, specific volume α = A + λ/(P + p)), written with
MOM6's exact association (`density_elem_Wright_full`, `:86-89`):

```
al0    = a0 + (a1*T + a2*S)                                          [m3 kg-1]
p0     = b0 + ( b4*S + T*(b1 + (T*(b2 + b3*T) + b5*S)) )             [Pa]
lambda = c0 + ( c4*S + T*(c1 + (T*(c2 + c3*T) + c5*S)) )             [m2 s-2]
rho    = (p + p0) / (lambda + al0*(p + p0))                          [kg m-3]
```

Specific volume `al0 + lambda/(p + p0)` (`:142-145`).

**Anomaly forms, used to keep precision** (the subtraction is folded in algebraically, not done
after the fact):
- density anomaly ρ − ρ_ref (`density_anomaly_elem_Wright_full`, `:110-120`):
  ```
  pa_000 = b0*(1 - a0*rho_ref) - rho_ref*c0
  al_TS  = a1*T + a2*S ;  al0 = a0 + al_TS
  p_TSp  = p + (b4*S + T*(b1 + (T*(b2 + b3*T) + b5*S)))
  lam_TS = c4*S + T*(c1 + (T*(c2 + c3*T) + c5*S))
  rho - rho_ref = (pa_000 + (p_TSp - rho_ref*(p_TSp*al0 + (b0*al_TS + lam_TS)))) /
                  ((c0 + lam_TS) + al0*(b0 + p_TSp))
  ```
- specific-volume anomaly (`:168-173`): `al_TS + (lambda' + (a0 - spv_ref)*p_TSp)/(b0 + p_TSp)`
  with `lambda' = c0 + (a0 - spv_ref)*b0 + lam_TS`.
- The array driver uses the anomaly form whenever `rho_ref` is present (`:920-943`).

**First derivatives** (`calculate_density_derivs_elem_Wright_full`, `:194-202`):
```
I_denom2 = 1 / (lambda + al0*(p + p0))**2
drho_dT  = I_denom2 * ( lambda*(b1 + (T*(2*b2 + 3*b3*T) + b5*S))
                        - (p+p0)*((p+p0)*a1 + (c1 + (T*(2*c2 + 3*c3*T) + c5*S))) )
drho_dS  = I_denom2 * ( lambda*(b4 + b5*T) - (p+p0)*((p+p0)*a2 + (c4 + c5*T)) )
```
**Compressibility** (`calculate_compress_elem_Wright_full`, `:324-330`):
`rho = (p+p0)*I_denom`, `drho_dp = lambda*I_denom**2` [s² m⁻²] (= 1/c_s²).
**Second derivatives** (∂²ρ/∂S², ∂²ρ/∂S∂T, ∂²ρ/∂T², ∂²ρ/∂S∂p, ∂²ρ/∂T∂p) `:241-271`.
**Specific-volume derivatives** `:292-301`.
**Layer-mean specific volume in pressure** (`avg_spec_vol_Wright_full`, `:358-367`): series
truncated at ε⁸, documented valid for |ε| < 0.34 (`:334-336`).

Other Wright variants: `WRIGHT` (`MOM_EOS_Wright.F90`, reduced-range coefficients, keeps an
optional second-derivative bug `USE_WRIGHT_2ND_DERIV_BUG`, `MOM_EOS.F90:1637-1643`) and
`WRIGHT_REDUCED` (`MOM_EOS_Wright_red.F90`, same reduced fit "refactored for consistent answers
between compiler settings", `equation_of_state/_Equation_of_State.dox:29-30`).

---

## 2. ROQUET_RHO (`equation_of_state/MOM_EOS_Roquet_rho.F90`), ROQUET_SPV, TEOS10

**Reference.** Roquet, Madec, McDougall & Barker 2015, Ocean Modelling 90:29-43
(`MOM_EOS_Roquet_rho.F90:687-689`). Code supplied by F. Roquet to MOM6 (`:676-683`).
`EQN_OF_STATE="NEMO"` is an alias of `ROQUET_RHO` (`MOM_EOS.F90:1583-1586`).

**Inputs.** T = Conservative Temperature Θ [°C] (`:196`), S = Absolute Salinity S_A [g kg⁻¹]
(`:197`), p = pressure [Pa] (`:198`). No conversion inside (the conversion lines are commented
out, `:222-225`).

**Scaled variables** (`:14-18`, `:218-220`):
```
zs = sqrt( |S + rdeltaS| * r1_S0 )   rdeltaS = 32 g/kg,  r1_S0 = 0.875/35.16504 kg/g
                                      (= 1/S_u with S_u = 40*35.16504/35)
zt = T                                (the 1/T_u = I_Ts = 0.025 °C⁻¹ is folded into the coefficients)
zp = p [Pa]                           (Pa2kb = 1e-8 = 1/(1e8 Pa) folded into the coefficients)
```
Note the salinity offset here is 32 g/kg; in ROQUET_SPV and in GSW `gsw_specvol` it is 24
(`MOM_EOS_Roquet_SpV.F90:16`; `equation_of_state/TEOS10/gsw_mod_teos10_constants.f90`:
`offset = deltaS*gsw_sfac`, deltaS = 24, lines 45-49).

**Form** (`density_elem_Roquet_rho`, `:227-246`): ρ = ρ00p(p) + ρ_TS(zs, zt, p) with
```
rho00p = zp*(R00 + zp*(R01 + zp*(R02 + zp*(R03 + zp*(R04 + zp*R05)))))     (6 coefficients, :22-27)
rhoTS  = (rhoTS0 + rho0S0) + zp*(rhoTS1 + zp*(rhoTS2 + zp*rhoTS3))
```
`rho0S0` is the 7-term polynomial in zs at zt = 0 (`:241`), `rhoTS0..3` the zt-dependent and
p¹..p³ parts (`:227-239`), all in nested Horner form; the exact nesting is part of the
bit-for-bit answer.

**Coefficients.** `EOS000`…`EOS013`: **52** coefficients at `MOM_EOS_Roquet_rho.F90:33-84`
(28 with p⁰, 15 with p¹, 6 with p², 3 with p³), plus 6 reference-profile coefficients
`R00`…`R05` at `:22-27`. Each is the Roquet number times its scale factor
(`*I_Ts**b`, `*Pa2kb**c`), evaluated by the compiler as a Fortran `parameter`.
Derivative coefficients `ALP…` (∂ρ/∂Θ, `:86-120`) and `BET…` (∂ρ/∂zs × r1_S0/2 factors, `:122-156`)
are derived from the `EOS…` set.

**Inconsistent term counts in MOM6's own text (flag):** the module footer says "52 terms"
(`:683`), the `.dox` says "a 75-member polynomial for density" (`_Equation_of_State.dox:62-64`),
and `MOM_EOS_Roquet_SpV.F90:768` says "55 terms". The code has 52 + 6. The 75-term polynomial is
GSW's `gsw_specvol` (75 `v…` coefficients, `TEOS10/gsw_mod_specvol_coefficients.f90`), which is
what `EQN_OF_STATE="TEOS10"` evaluates (below). Roquet et al. (2015) name the forms: polyTEOS10-bsq (density,
52-term anomaly + 6-term reference profile = `ROQUET_RHO`), polyTEOS10-55t (specific volume,
52 + 3 terms = the "55 terms" of `MOM_EOS_Roquet_SpV`) and polyTEOS10-75t (GSW's specific volume).

**Anomaly form** (`:305-308`): `rho_ref` is subtracted from `rho0S0` **before** the
pressure-dependent terms are added (precision device, different from TEOS10's after-the-fact
subtraction).

**Derivatives available.** ∂ρ/∂Θ and ∂ρ/∂S_A (`:342-413`; ∂ρ/∂S_A = (Σ BET·…)/zs because
dzs/dS_A = r1_S0/(2 zs), comment `:410-411`), second derivatives (`:416-505`), specific-volume
derivatives (`:509-528`), compressibility = ∂ρ_TS/∂p + ∂ρ00p/∂p (`:534-597`). Specific volume is
`1/ρ` (`:321`, `:335-336`). Fit range Θ −6…40 °C, S_A 0…42 g/kg, p 0…1e8 Pa (`:611-616`).

**ROQUET_SPV** (`MOM_EOS_Roquet_SpV.F90`): the same structure for specific volume,
v = v00p(p) + v_TS (`:247-248`), 52 `SPV…` + 6 `V0…` coefficients, offset 24 g/kg, density
= 1/v (`:330`); intended for non-Boussinesq models (`:760-768`).

**TEOS10** (`MOM_EOS_TEOS10.F90`): calls GSW-Fortran shipped in `equation_of_state/TEOS10/`:
`gsw_rho(S, T, p*1e-4)` (`:55`, Pa → dbar `Pa2db = 1e-4`, `:19`). `gsw_rho = 1/gsw_specvol`
(`TEOS10/gsw_rho.f90:31`), and `gsw_specvol` is the **75-term polynomial** in
xs = sqrt(gsw_sfac·S_A + offset), ys = 0.025·Θ, z = 1e-4·p (`TEOS10/gsw_specvol.f90:31-47`), not
the full Gibbs function. Anomaly = `gsw_rho - rho_ref` (`:67-68`, no precision device). The GSW p is
*sea* pressure; MOM6 passes its ocean pressure (0 at the surface plus `p_surf` when
`USE_PSURF_IN_EOS`), consistent with sea pressure. So in MOM6 "ROQUET_RHO" and "TEOS10" are two
different polynomial fits to TEOS-10 (52+6-term density vs 75-term specific volume) and are not
identical to round-off.

---

## 3. Dispatch in `equation_of_state/MOM_EOS.F90`

**Selection.** `EQN_OF_STATE` read in `EOS_init` (`MOM_EOS.F90:1563-1594`); default
`WRIGHT_FULL` (`:186`). Strings: LINEAR, UNESCO, JACKETT_MCD (→ UNESCO), WRIGHT, WRIGHT_REDUCED,
WRIGHT_FULL, TEOS10, NEMO (→ ROQUET_RHO), ROQUET_RHO, ROQUET_SPV, JACKETT_06 (`:161-185`).
Linear EOS parameters `RHO_T0_S0`, `DRHO_DT` (−0.2), `DRHO_DS` (0.8), `DRHO_DP` (0) (`:1598-1636`).

**Interface every EOS provides** (deferred in `MOM_EOS_base_type.F90:17-36`): `density_elem`,
`density_anomaly_elem` (with `rho_ref`), `spec_vol_elem`, `spec_vol_anomaly_elem` (with `spv_ref`),
`calculate_density_derivs_elem` (∂ρ/∂T, ∂ρ/∂S), `calculate_density_second_derivs_elem`,
`calculate_specvol_derivs_elem`, `calculate_compress_elem` (ρ, ∂ρ/∂p), `EOS_fit_range`.
Array/scalar wrappers in the base type (`:40-62`); Wright_full and Roquet override the array
density/spec-vol for speed (`MOM_EOS_Wright_full.F90:63-66`). Public front ends in `MOM_EOS.F90`:
`calculate_density` (`:70`, `rho_ref` optional → anomaly form, `:224-228`, `:336-350`),
`calculate_density_derivs` (`:751-874`), `calculate_specific_vol_derivs`, `calculate_compress`,
`calculate_TFreeze`, `average_specific_vol`, `analytic_int_density_dz` (`:1385-1500`).
All front ends convert model units to MKS (`RL2_T2_to_Pa`, `C_to_degC`, `S_to_ppt`) and rescale
the output; with unit scaling 1 they call the EOS directly (`:336-338`).

**Analytic integrals exist only for LINEAR and the three Wright EOS**
(`analytic_int_density_dz`, `:1442-1498`; any other EOS → FATAL `:1497`). `EOS_QUADRATURE`
default is `.not.(LINEAR or WRIGHT or WRIGHT_REDUCED or WRIGHT_FULL)` (`:1651-1657`), i.e. True for
ROQUET/TEOS10/UNESCO/JACKETT06.

**Freezing point, `TFREEZE_FORM`** (`:1659-1712`; default LINEAR, but TEOS10 for TEOS10/ROQUET EOS,
`:1659-1662`; warning if TEOS-family EOS is used with LINEAR/MILLERO, `:1714-1719`):
- LINEAR: `T_Fr = (TFREEZE_S0_P0 + DTFREEZE_DS*S) + DTFREEZE_DP*p` (`MOM_TFreeze.F90:62`,`:84`);
  defaults 0.0 °C, −0.054 °C ppt⁻¹, 0.0 °C Pa⁻¹ (`MOM_EOS.F90:1683-1696`).
- MILLERO_78 (`MOM_TFreeze.F90:95-135`), TEOS_POLY (23-term polynomial, `:140-205`),
  TEOS10 = `gsw_ct_freezing_exact(S_A, p*1e-4, saturation_fraction=0)` (`:231-250`).
- With `use_conT_absS`, S is converted SA→SP for the freezing formula if `TFREEZE_S_IS_PRACS`
  and the result θ→Θ via `gsw_ct_from_pt` if `TFREEZE_T_IS_POTT` (`MOM_EOS.F90:555-585`);
  defaults of both are True for LINEAR/MILLERO, False for TEOS forms (`:1699-1712`).

**θ/SP ↔ Θ/SA conversions.**
- SA = (35.16504/35)·SP and SP = (35/35.16504)·SA (`prac_saln_to_abs_saln` `:1988-2022`,
  `abs_saln_to_prac_saln` `:1948-1982`). This is the reference salinity S_R; **no spatially varying
  absolute-salinity anomaly δS_A is used anywhere** (no `gsw_sa_from_sp`/`gsw_deltasa` call in
  `src/`). `gsw_sr_from_sp`/`gsw_sp_from_sr` are re-exported (`MOM_EOS.F90:58-60`).
- θ→Θ: `poTemp_to_consTemp` (`MOM_temperature_convert.F90:61-78`), a polynomial in θ and
  x = sqrt(max(S_A·0.025·35/35.16504, 0)) with coefficients H00…H70 copied from GSW and divided by
  c_p0 = 3991.86795711963 (`:13-53`). Θ→θ: `consTemp_to_poTemp` (`:105-148`), rational first guess
  + 1.5 modified Newton (McDougall) iterations, "converge to 64-bit machine precision" (`:132-134`).
  Wrapped by `pot_temp_to_cons_temp` / `cons_temp_to_pot_temp` (`MOM_EOS.F90:1854-1942`).
- The coupler cap uses GSW directly: `gsw_pt_from_ct(SSS, SST)` and `gsw_sp_from_sr(SSS)`
  (`config_src/drivers/FMS_cap/ocean_model_MOM.F90:879-899`; same in the NUOPC cap,
  `config_src/drivers/nuopc_cap/mom_ocean_model_nuopc.F90:951-970`).
- Initial conditions from z-level files: `convert_temp_salt_for_TEOS10`
  (`MOM_EOS.F90:1824-1848`, called at `initialization/MOM_state_initialization.F90:2854`)
  converts S ← 35.16504/35·S then T ← poTemp_to_consTemp(T, S_new). **It is triggered by the EOS
  choice (TEOS10, ROQUET_RHO, ROQUET_SPV), not by `USE_CONTEMP_ABSSAL`** (`:1839-1840`).

**`USE_CONTEMP_ABSSAL`** (`core/MOM.F90:2543-2549`, default False) sets `tv%T_is_conT` and
`tv%S_is_absS` and is passed to `EOS_init` (`core/MOM.F90:3161`) where it only affects the
freezing-point conversions above. It also renames the prognostic variables
(`contemp`/`abssalt`, `core/MOM.F90:3053-3070`), drives diagnostic conversions
(`diagnostics/MOM_diagnostics.F90:427, 503, …`) and the coupler conversion. MOM6 does **not**
check that `USE_CONTEMP_ABSSAL` matches the EOS: Wright with CT/SA, or TEOS10 with
`USE_CONTEMP_ABSSAL=False` (initial data converted to Θ/S_A by the EOS trigger while the model
labels and couples them as θ/SP), are both accepted. This is a MOM6 configuration trap, not a
documented feature [unclear whether intended].

**Heat capacity.** `C_P` (`core/MOM.F90:2755-2759`): default 3991.86795711963 J kg⁻¹ K⁻¹ (the
TEOS-10 c_p0 that defines Θ), independent of `USE_CONTEMP_ABSSAL` and of the EOS. It converts heat
fluxes to temperature fluxes (`conv2watt`, `core/MOM.F90:3074`). Ice shelf reads the same parameter
(`ice_shelf/MOM_ice_shelf.F90:1827-1830`).

---

## 4. Density integrals for the finite-volume PGF

Files: `core/MOM_density_integrals.F90` (generic), `equation_of_state/MOM_EOS_Wright_full.F90`
(analytic), `core/MOM_PressureForce_FV.F90` (assembly, Boussinesq routine `PressureForce_FV_Bouss`
from ~`:1000`).

### 4.0 Which path is used (important)

- `ANALYTIC_FV_PGF` (`core/MOM_PressureForce.F90:103-116`) only chooses **finite-volume (Adcroft
  et al. 2008) vs Montgomery** PGF. It does *not* choose analytic vs quadrature integration
  (the parameter name is misleading).
- In `PressureForce_FV_Bouss`, `use_ALE = CS%reconstruct .and. use_EOS` (`:1120`).
  With `RECONSTRUCT_FOR_PRESSURE` (default = `USE_REGRIDDING`, `:2173-2178`) and
  `PRESSURE_RECONSTRUCTION_SCHEME` 1 or 3 → `int_density_dz_generic_plm` (`:1285-1293`);
  2 → `int_density_dz_generic_ppm` (`:1294-1300`). **These are always quadrature, for every EOS,
  including WRIGHT_FULL.**
- Only without reconstruction (scheme 0 or `RECONSTRUCT_FOR_PRESSURE=False`) is
  `int_density_dz` called (`:1303-1307`), which dispatches on `EOS_QUADRATURE`:
  generic PCM quadrature if True, analytic (Linear/Wright only) if False
  (`MOM_density_integrals.F90:95-103`).
- Consequence for NWA12 (WRIGHT_FULL, ALE, `RECONSTRUCT_FOR_PRESSURE=True`, scheme 1): the PGF
  uses `int_density_dz_generic_plm`, i.e. Boole quadrature of the Wright density at reconstructed
  T, S. The analytic Wright integral is never reached in NWA12.

### 4.a Analytic Wright integral (`int_density_dz_wright_full`, `MOM_EOS_Wright_full.F90:397-662`)

Boussinesq form: the EOS pressure is p = −g ρ0 (z − Z_0p) (`GxRho = G_e*rho_0`, `:506-512`;
`p_ave`, `:559`). With PCM T, S in the layer and `al0, p0, lambda` computed once per cell
(`:552-554`), and writing L = (p0 + p_ave) + λ/al0, ε = ½ g ρ0 Δz / L (`:561-563`):

ρ(p) = 1/al0 − (λ/al0²)/(L + (p − p_ave)), so the exact layer integral is
∫ g ρ dz = (1/ρ0)∫ρ dp = g Δz ρ(p_ave) − (λ/(ρ0 al0²))·[ln((1+ε)/(1−ε)) − 2ε].
MOM6 truncates the log series at ε⁹ (`:567-571`):
```
rho_anom = (p0 + p_ave)*(I_Lzz*I_al0) - rho_ref            ! = rho(p_mid) - rho_ref
rem      = (I_Rho*(lambda*I_al0**2)) * eps2*(1/3 + eps2*(0.2 + eps2*(1/7 + eps2/9)))
dpa      = g*rho_anom*dz - 2*eps*rem                        ! Δ of pressure anomaly [Pa]
intz_dpa = 0.5*g*rho_anom*dz**2 - dz*(1 + eps)*rem          ! ∫ over the layer of (pa - pa_top)
```
(`rho_ref` here is subtracted after evaluation, not via the anomaly form.)
Horizontal integrals `intx_dpa`/`inty_dpa` (`:574-660`): 5 sub-columns at x-fractions
0, ¼, ½, ¾, 1; the end values are the column `dpa`; at the 3 interior ones **the EOS
coefficients al0, p0, λ (not T, S) are interpolated linearly** (with mass weights) and the same
closed form is evaluated (`:599-612`); then Boole in x:
`intx_dpa = (7(f1+f5) + 32(f2+f4) + 12 f3)/90` (`:615`). The analytic path does not receive
`MassWghtInterpVanOnly` (`MOM_density_integrals.F90:101-102`). The non-Boussinesq counterpart is
`int_spec_vol_dp_wright_full` (`:669-917`).

### 4.b Generic quadrature (`MOM_density_integrals.F90`)

**PCM** (`int_density_dz_generic_pcm`, `:110-417`) and **PLM** (`int_density_dz_generic_plm`,
`:420-~900`; PPM analogous). PLM details:

1. *Vertical, per column* (`:590-640`): 5 equally spaced points in the layer, n = 1..5 top to
   bottom, weights `wt_t = (5-n)/4`, `wt_b = 1 - wt_t` (`:579-582`):
   ```
   p5(n) = -GxRho*((e_top - Z_0p) - 0.25*(n-1)*dz)         ! GxRho = g*rho_0 (:539)
   S5(n) = wt_t*S_t + wt_b*S_b ;  T5(n) = wt_t*T_t + wt_b*T_b
   r5    = calculate_density(T5, S5, p5, rho_ref=rho_ref)  ! anomaly form (:607)
   rho_anom = (7(r1+r5) + 32(r2+r4) + 12 r3)/90             ! Boole (:617)
   dpa      = g*dz*rho_anom                                 ! (:618)
   intz_dpa = 0.5*g*dz**2*(rho_anom - (16(r4-r2) + 7(r5-r1))/90)   ! (:622-623)
   ```
   i.e. **5 EOS calls per cell per layer**.
2. *Horizontal, per u-face* (`:643-751`): edge values T_t, T_b, S_t, S_b of the two columns are
   mixed by the mass weights (below) into Ttl, Ttr, Tbl, Tbr, … (`:669-685`); at the 3 interior
   sub-columns m = 2, 3, 4 (fractions ¼, ½, ¾) the top and bottom T, S are linear in x, the layer
   thickness `dz_x` and the top pressure are linear in x, the 5 vertical points are linear between
   top and bottom (`:687-717`): **15 EOS calls per face per layer**. Each sub-column gives
   `intz(m) = g*dz_x*Boole(r15)` (`:735-739`); with `intz(1) = dpa_L`, `intz(5) = dpa_R`,
   `intx_dpa = (7(intz1+intz5) + 32(intz2+intz4) + 12 intz3)/90` (`:748-749`). Same for v-faces
   (`:754-…`). So the face integral is a 5 × 5 Boole tensor rule in (x, z) over the trapezoid,
   with T, S bilinear in the sub-column fractions.
3. *T, S reconstruction* (`RECONSTRUCT_FOR_PRESSURE`, `PRESSURE_RECONSTRUCTION_SCHEME=1`):
   `TS_PLM_edge_values` → `ALE_PLM_edge_values` for **S and T separately** (`ALE/MOM_ALE.F90:1512-1596`):
   `PLM_slope_wa` + `PLM_monotonized_slope` in the interior (`:1566-1578`), and with
   `BOUNDARY_EXTRAPOLATION_PRESSURE=True` (default, `core/MOM_PressureForce_FV.F90:2186-2190`)
   `PLM_extrapolate_slope` in the top and bottom layers (`:1579-1586`), PCM there otherwise
   (`:1587-1591`). Functions from `ALE/PLM_functions.F90` (`MOM_ALE.F90:55-56`). MOM6 does **not**
   reconstruct density; density is evaluated pointwise from reconstructed T and S.
4. *Mass weighting* (`MASS_WEIGHT_IN_PRESSURE_GRADIENT` → bit 0, `_TOP` → bit 1,
   `core/MOM_PressureForce_FV.F90:2124-2149`; `MOM_density_integrals.F90:547-555, 652-685`):
   ```
   hWght = massWeightToggle * max(0, -D_L - e_R(K), -D_R - e_L(K))       ! one column's top below the other's bed
   hWght = max(hWght, TopToggle * max(0, e_R(K+1) - e_L(1), e_L(K+1) - e_R(1)))
   if both layers > h_nonvanished: hWght = massWeightNVonlyToggle*hWght   ! VanOnly option
   hWght = hWght*((hL - hR)/(hL + hR))**2 ; iDenom = 1/(hWght*(hL+hR) + hL*hR)
   T_left_edge = ((hWght*hR)*T_R + (hWght*hL + hR*hL)*T_L)*iDenom   (etc.)
   ```
   Only T and S are biased; dz and pressure stay linear in x.
5. *`rho_ref` subtraction* (`RHO_PGF_REF`, default `RHO_0`, `core/MOM_PressureForce_FV.F90:2063`):
   passed into `calculate_density(..., rho_ref=rho_ref)` so the EOS returns ρ − ρ_ref in its
   anomaly form (Wright: rearranged numerator; Roquet: subtracted from `rho0S0` before the p terms;
   TEOS10: plain subtraction). `USE_INACCURATE_PGF_RHO_ANOM` switches to computing ρ and
   subtracting afterwards (`MOM_density_integrals.F90:560-561, 608-611, 626-638`).
6. *EOS pressure in the PGF*: p = −g ρ0 (z − Z_0p) with Z_0p = `G%meanSL` by default
   (`core/MOM_PressureForce_FV.F90:1262-1274`); η and p_atm enter only with
   `SSH_IN_EOS_PRESSURE_FOR_PGF=True`. `rho_0` here is `GV%Rho0` unless `RHO_PGF_REF_BUG`
   (`:1130-1139`).
7. *Which path for which EOS when `ANALYTIC_FV_PGF=True`*: see §4.0. With reconstruction,
   quadrature for all EOS. Without reconstruction, analytic for LINEAR and Wright family
   (unless `EOS_QUADRATURE=True`), PCM quadrature for ROQUET/TEOS10 (forced: analytic is FATAL).

### 4.c Boussinesq assembly (`PressureForce_FV_Bouss`, `core/MOM_PressureForce_FV.F90`)

- Interface heights `e` from the bottom up (`:1196-1198`).
- Surface pressure anomaly `pa(1) = g*rho_ref*(e(1) - Z_ref) [+ p_atm]` (`:1250-1260`;
  `GxRho_ref = g*rho_ref`, `:1138`).
- `pa(K+1) = pa(K) + dpa(k)` (`:1338-1343`); SAL/tides modify `e` and `pa` (`:1345-1381`).
- `intx_pa(1) = ½(pa_L(1) + pa_R(1))` (+ optional `CORRECTION_INTXPA`, `:1396-1447, 1539`),
  `intx_pa(K+1) = intx_pa(K) + intx_dpa(k)` (`:1550-1556`); optional `RESET_INTXPA_INTEGRAL`
  rebuilds them (`:1567-1791`).
- Force (`:1796-1801`, v analogous `:1807-1812`):
  ```
  PFu = [ (pa_L(K)*h_L + intz_dpa_L) - (pa_R(K)*h_R + intz_dpa_R)
          + (h_R - h_L)*intx_pa(K) - (e_R(K+1) - e_L(K+1))*intx_dpa ]
        * 2 / (rho_0 * dx * (h_L + h_R + h_neglect))
  ```
  ρ0 enters (i) the EOS pressure p = −g ρ0 (z − Z_0p) and (ii) the 1/ρ0 factor;
  `rho_ref` enters only as the subtracted density and in the top `pa`; g enters `dpa`, `pa(1)`.
- `GFS_scale ≠ 1` (reduced surface gravity) adds a barotropic correction (`:1845-1872`).

---

## 5. N² and buoyancy for mixing

| Scheme | Quantity | T, S used | Pressure | Form |
|---|---|---|---|---|
| kappa_shear (`parameterizations/vertical/MOM_kappa_shear.F90`) | N² at interfaces | interface means T_int = ½(T_{k−1}+T_k), S_int (`:1131-1132`) of the (massless-merged) column | p(1) = p_surf; p(K) = p(K−1) + g ρ0 h_{k−1} (`:1128-1130`; `gR0 = H_to_RZ*g`, `:1008`) | **locally referenced**: `dbuoy_dT/S = −(g/ρ0)·∂ρ/∂T,S` at (T_int, S_int, p_K) (`:1135-1136`, `g_R0 = g/ρ0` `:1009`); N² = max(0, [dbuoy_dT (T_{k−1}−T_k) + dbuoy_dS (S_{k−1}−S_k)]/dz_int) (`:1496-1500`) |
| set_diffusivity `find_N2` (bkgnd, tidal, BBL LOTW, drag) (`MOM_set_diffusivity.F90`) | ΔρInt, N²_int, N²_lay, N²_bot | filled T_f, S_f (massless layers filled by `vert_fill_TS`, `:463`), interface means (`:1170-1171`) | p(1) = `fluxes%p_surf` or 0; p(K) = p(K−1) + g ρ0 h_{k−1} (`:1160-1169`) | **locally referenced**: Δρ_K = max(∂ρ/∂T (T_k−T_{k−1}) + ∂ρ/∂S (S_k−S_{k−1}), 0) with derivatives at (T_int, S_int, p_K) (`:1173-1180`); N²_int = (g/ρ0) Δρ_K / (½(h_{k−1}+h_k)) (`:1197-1200`); N²_lay from the two bounding Δρ (`:1192-1195`); near the bed N²_int is **overwritten by N²_bot**, the bulk Σ Δρ / Σ h over at least the bottom interface plus one more (or `h_amp` with tidal mixing) (`:1202-1258`); unfiltered Δρ restored after (`:1260-1262`) |
| double diffusion (`MOM_set_diffusivity.F90`) | α ΔT, β ΔS | T_f, S_f interface means | same running hydrostatic pressure (`:1323-1334`) | locally referenced derivatives |
| KPP / CVMix (`MOM_CVMix_KPP.F90`, `KPP_compute_BLD`) | bulk Ri ΔB(k), local N² | layer k vs surface-layer mean (T_sr, S_sr over `SLdepth`, `:1335-1367`); layer k vs k−1 | `pRef` = pressure at the **top interface of layer k**, p_surf + Σ_{k'<k} g ρ0 h_{k'} (`:1237, 1396-1408`) | **density difference at a common pressure** (full EOS calls, no derivatives): ΔB(k) = (g/ρ0)[ρ(T_k,S_k,p_k^top) − ρ(T_sr,S_sr,p_k^top)] (`:1420, 1428-1430`); N²(k) = (g/ρ0)[ρ(T_k,S_k,p_k^top) − ρ(T_{k−1},S_{k−1},p_k^top)] / (½(h_{k−1}+h_k)) (`:1435-1436`) |

Notes: all three use a hydrostatic pressure built from h with ρ0 (not in-situ ρ), starting from
`p_surf`. KPP's comment at `:1406` says "pressure at interface between k and km1", which is the
top interface of layer k (the value is used before being incremented). Non-Boussinesq variants use
specific-volume derivatives (`MOM_kappa_shear.F90:1137-1146`; KPP `:1431-1433`).
NWA12 runs ePBL + Jackson kappa_shear + double diffusion, no KPP, no LOTW BBL
(NWA12 doc `:1806, 1981, 1968, 1869, 1913`).

kappa_shear computes dbuoy_dT, dbuoy_dS **once per column**, from T and S after the κ0 step
(`MOM_kappa_shear.F90:1052-1088`, `:1126-1136`), and keeps them fixed while it carries T and S
separately through every `calculate_projected_state` of the sub-steps (`:1379-1506`, N² `:1493-1503`).
pyOCEAN ports this two-tracer path since 2026-11-01 (before: one locally referenced buoyancy, MOM6's N²
only at the start; second-order difference); equal to compiled MOM6 bit for bit
(`tests/test_vertical_vs_mom6.py::test_kappa_shear_nonlinear_eos_matches_mom6`).

**Spec decision 8 — bottom N² of the BBL (checked in task 6, 2026-10-31; reclassified after the task-6
review).** MOM6 builds `N2_int` in `find_N2` (called at `MOM_set_diffusivity.F90:481`) and, before
`add_LOTW_BBL_diffusivity` receives it (`:671`, used through max(N2_int, N2_min) at `:1762`), overwrites the
near-bed interfaces with a bulk value (`:1202-1258`): going up from the bed it sums the clipped jumps
Δρ_K = max(∂ρ/∂T ΔT + ∂ρ/∂S ΔS, 0) and the thicknesses ½(h_{k−1} + h_k) until the layer-centre height passes
`h_amp` (0 without tidal mixing), always adding one more interface (`:1219-1223`).  So, **unconditionally** and
also with h_amp = 0, N²_bot = (g/ρ₀)(Δρ_nz + Δρ_nz−1) / (½(h_nz + h_nz−1) + ½(h_nz−1 + h_nz−2)) replaces N² at the
**two lowest interior interfaces** (`:1236-1258`; the layer N² of the bottom layers too).  pyOCEAN's port
(`physics/mixing/bbl_mixing.py`) starts at `add_LOTW_BBL_diffusivity` and takes the local interface N² of the
(merged) column; it does **not** apply the overwrite.  The omission is unintentional: the compiled-MOM6
comparison of the BBL task (`validation/mom6_column/bbl_driver.F90:85-91`) passes `N2_int` straight to
`add_LOTW_BBL_diffusivity`, bypassing `find_N2`, so the 15-column agreement never covered this step.
**Class (a), our bug.  Fixed on 2026-11-01** with Esteban's approval, in a separate follow-up (it changes the BBL
results of linear-EOS runs, which this project's anti-goal kept bit for bit, and the DOME BBL numbers):
`physics.mixing.bbl_mixing.n2_bottom_bulk`, checked against compiled MOM6 `find_N2` (extracted verbatim,
`bbl_driver n2`) on 8 columns and chained into `add_LOTW_BBL_diffusivity`
(`tests/test_vertical_vs_mom6.py::test_bbl_bottom_n2_matches_mom6_find_n2`, `::test_bbl_kd_with_bottom_n2_matches_mom6`).  The nonlinear EOS does not change the issue (both codes use the locally referenced Δρ).

Measured effect (`validation/mom6_column/bbl_n2bot_effect.py`, measured before the fix: Kd of `lotw_bbl_column` with the
local N² vs with N²_bot at interfaces 1 and 2, on the 15 columns of `columns.bbl_cases` that feed `bbl_driver`;
relative change |ΔKd|/max(Kd) over the interfaces where either Kd > 0):

| column | max rel ΔKd | median rel ΔKd |
|---|---|---|
| stratified | 1.4e-2 | 0 |
| N² ≤ 0 at two interfaces | 0 | 0 |
| N² ≤ 0, no Ω floor | 0 | 0 |
| f = 0 | 1.6e-2 | 0 |
| 1 mm layers | 1.1e-2 | 0 |
| large u* (200 m cap) | 2.0e-3 | 0 |
| ample energy (uncut) | 0 | 0 |
| random 0 | 1.00 | 0.91 |
| random 1 | 0.18 | 0.18 |
| random 2 | 0.22 | 0.11 |
| random 3 | 0.37 | 0.37 |
| random 4 | 0.16 | 0.16 |
| random 5 | 2.0e-2 | 2.0e-2 |
| random 6 | 0.12 | 0.12 |
| random 7 | 7.8e-2 | 7.8e-2 |

Smooth stratification (cases 1, 4–7): ≤ 1.6 % at the two bottom interfaces only (median 0).  Random N² profiles
(cases 8–15): the bulk N² changes the energy spent at the bottom and with it Kd up the whole column, 2–37 %
typical; in random 0 the energy runs out at a different interface (Kd → 0 there, 100 %).  N² ≤ 0 at both bottom
interfaces gives the same Kd (N²_bot = 0, then the Ω² floor or KD_MAX in both codes).

---

## 6. Surface buoyancy flux and SST

**Buoyancy flux** (`core/MOM_forcing_type.F90`, `calculateBuoyancyFlux1d` `:1028-1151`):
- Derivatives at **top-layer** T(:,j,1), S(:,j,1) (not the HMIX_SFC_PROP mean) and pressure
  `tv%p_surf` (0 if not associated) (`:1116-1117, 1138-1139`).
- Boussinesq: B(1) = −(g/ρ0)(∂ρ/∂S · netSalt + ∂ρ/∂T · netHeat) with
  netHeat = non-solar + shortwave absorbed above interface 1, netSalt including the dilution
  −S₁·netH (`:1105-1113, 1135-1144`); penetrating SW at interface k:
  B(k) = −(g/ρ0) ∂ρ/∂T · netPen(k), **with the same surface ∂ρ/∂T** (`:1146-1148`).
- Non-Boussinesq: specific-volume derivatives × g (`:1119-1133`).

**SST/SSS exchanged** (`core/MOM.F90`, `extract_surface_state` `:4136-4536`): SST, SSS are the
thickness-weighted means of `tv%T`, `tv%S` over `HMIX_SFC_PROP` (`:4235-4300`), i.e. **Θ and S_A when
`USE_CONTEMP_ABSSAL`**; `sfc_state%T_is_conT` is set (`:4207-4208`). The conversion happens in the
coupler cap, after averaging: `t_surf = gsw_pt_from_ct(SSS, SST) + 273.15`,
`s_surf = gsw_sp_from_sr(SSS)` (`config_src/drivers/FMS_cap/ocean_model_MOM.F90:879-899`;
NUOPC `:951-970`). The bulk formulae (in the coupler/atmosphere) therefore see θ and SP. Returned heat
fluxes are converted to temperature tendencies with `C_P` (§3). The solo driver has no such
conversion (not needed for idealized forcing).

---

## 7. Regridding coordinate density

- `P_REF` (`ALE/MOM_regridding.F90:971-993`): read only for state-dependent coordinates (RHO,
  HYCOM1, …), default 2e7 Pa; also `REGRID_COMPRESSIBILITY_FRACTION` (default 0, `:987-992`).
  The model-wide `tv%P_Ref` is the same parameter read in `core/MOM.F90:2732, 2764`
  (NWA12 doc `:146`).
- HYCOM1 (`build_grid_HyCOM1`, `ALE/MOM_regridding.F90:2045-2069`): per layer
  `p_col(k) = tv%P_Ref + compressibility_fraction*(z_mid*g*ρ0 − tv%P_Ref)`, so with the default 0
  it is P_REF; density `calculate_density(T, S, p_col, rho_col)` in
  `ALE/coord_hycom.F90:214` (or `:312` in the only-improves branch), then forced monotonic
  `rho_col(k) = min(rho_col(k), rho_col(k+1))` (`:217-219`). New-layer pressures interpolated
  linearly (`:236-237`).
- RHO coordinate: `pres(:) = CS%ref_pressure`, `calculate_density` (`ALE/coord_rho.F90:135-136, 234-258`).
- HYBGEN: `p_col(:) = CS%ref_pressure` (`P_REF`), `calculate_density` (`ALE/MOM_hybgen_regrid.F90:117, 439, 453`).
- ADAPTIVE: density derivatives at in-situ pressure `zInt*ρ0 g` (`ALE/coord_adapt.F90:227`).
So the hybrid target density is in-situ ρ at 2000 dbar from the model's (θ or Θ) and (SP or S_A):
σ₂ exactly for TEOS10/Roquet with Θ, S_A; an approximation of σ₂ for Wright with θ (since ρ(θ,S,p_r)
is potential density referenced to p_r only if θ is referenced to p_r; MOM6's θ is referenced to 0).
[the last statement is our analysis, not a MOM6 comment]

---

## 8. Cost

No timing or flop notes for the EOS were found in the MOM6 source (`grep -i
"expensive|flops|efficien|faster"` over `equation_of_state/`, `core/MOM_density_integrals.F90`,
`core/MOM_PressureForce_FV.F90` returns nothing relevant). Our counts from the code:
- WRIGHT_FULL density: ~10 multiplies + ~10 adds for al0, p0, λ and 1 division (`:86-89`).
- ROQUET_RHO density: 52 + 6 coefficients in Horner form, ~60 fused multiply-adds + 1 sqrt (`:227-246`).
- TEOS10: 75-term `gsw_specvol` + 1 sqrt + 1 division per ρ (`TEOS10/gsw_specvol.f90:31-47`, `gsw_rho.f90:31`).
- PGF quadrature (§4.b): 5 EOS calls per cell per layer + 15 per u-face + 15 per v-face ≈ 35 EOS
  evaluations per cell per layer per PGF call; the Wright analytic path needs 1 coefficient set
  per cell per layer and 3 closed-form evaluations per face (no EOS calls at sub-columns).
- Precomputation: only the analytic Wright path stores per-cell al0, p0, λ for the layer
  (`MOM_EOS_Wright_full.F90:551-556`) and reuses them for the faces. The generic paths precompute
  nothing per column; they pack the 5 or 15 points into 1-D arrays and make one vectorized EOS call
  per j-row (`MOM_density_integrals.F90:585-587, 604-611, 720-728`). `EOS_domain` limits the index range.

---

## NWA12 (CEFI `exps/NWA12.COBALT/MOM_parameter_doc.all`)

| Parameter | Value | line |
|---|---|---|
| EQN_OF_STATE | "WRIGHT_FULL" | 514 |
| EOS_QUADRATURE | False | 519 |
| USE_CONTEMP_ABSSAL | False | 21 |
| TFREEZE_FORM | "LINEAR" | 522 |
| TFREEZE_S0_P0 / DTFREEZE_DS / DTFREEZE_DP | 0.0 / −0.054 / −7.75e-8 °C Pa⁻¹ | 526 / 529 / 532 |
| TFREEZE_S_IS_PRACS / TFREEZE_T_IS_POTT | True / True | 535 / 538 |
| C_P | 3992.0 (default 3991.86795711963) | 139 |
| RHO_0 | 1035.0 | 498 |
| BOUSSINESQ | True | 502 |
| G_EARTH | 9.8 | 496 |
| USE_PSURF_IN_EOS | True | 143 |
| P_REF | 2.0e7 Pa | 146 |
| USE_REGRIDDING | True | 39 |
| REGRIDDING_COORDINATE_MODE | "Z*" | 663 |
| ALE_COORDINATE_CONFIG | "FILE:vgrid_75_2m.nc,dz" (NK = 75) | 675 |
| REMAPPING_SCHEME | "PPM_H4" | 717 |
| ANALYTIC_FV_PGF | True (selects FV, not analytic integration) | 1387 |
| RHO_PGF_REF | 1035.0 | 1394 |
| SSH_IN_EOS_PRESSURE_FOR_PGF | False | 1412 |
| MASS_WEIGHT_IN_PRESSURE_GRADIENT | True | 1417 |
| MASS_WEIGHT_IN_PRESSURE_GRADIENT_TOP | False | 1420 |
| MASS_WEIGHT_IN_PGF_VANISHED_ONLY | False | 1424 |
| CORRECTION_INTXPA / RESET_INTXPA_INTEGRAL | False / False | 1427 / 1429 |
| USE_INACCURATE_PGF_RHO_ANOM | False | 1435 |
| RECONSTRUCT_FOR_PRESSURE | True | 1438 |
| PRESSURE_RECONSTRUCTION_SCHEME | 1 (PLM) | 1442 |
| BOUNDARY_EXTRAPOLATION_PRESSURE | True | 1449 |
| HMIX_SFC_PROP / HMIX_UV_SFC_PROP | 1.0 m / 0.0 m | 101 / 105 |
| FRAZIL / BOUND_SALINITY | True / True | 123 / 129 |
| CALC_RHO_FOR_SEA_LEVEL | False | 10 |
| STOCH_EOS / USE_STANLEY_PGF | False / False | 1128 / 1453 |
| BBL_USE_EOS | True | 1037 |
| ENERGETICS_SFC_PBL / USE_JACKSON_PARAM / DOUBLE_DIFFUSION / USE_KPP | True / True / True / False | 1806 / 1981 / 1968 / 1869 |

So NWA12 is θ / SP, Wright full-range, Z*, PLM-reconstructed T, S with Boole quadrature in the PGF,
bottom mass weighting on, top mass weighting off. `REGRID_COMPRESSIBILITY_FRACTION` is not listed
(Z* is not state dependent).

---

## Implications for a Python/Numba port

**To be comparable to compiled MOM6 to round-off:**
1. The EOS kernels with MOM6's exact parenthesization: Wright_full `density_elem` and the
   **anomaly form** `density_anomaly_elem` (`MOM_EOS_Wright_full.F90:110-120`), derivatives
   (`:194-202`), compressibility (`:324-330`); Roquet_rho density/anomaly (`:227-308`, `rho_ref`
   subtracted from `rho0S0` before the p terms) and its ALP/BET derivative coefficient sets.
   Coefficients must be built exactly as Fortran builds them (`EOS010 = 2.6010145068e+01*I_Ts`,
   `R01 = -5.2099962525*Pa2kb**2`, …); in Python compute the products with the same factors
   in the same order (Fortran evaluates `parameter` expressions at compile time in double; check one
   coefficient against a compiled dump).
2. The PGF path NWA12 actually uses: `int_density_dz_generic_plm` (5-point Boole in z, 5 sub-columns
   in x with interior T, S linear in x and z, 15 EOS calls per face), PLM edge values of **T and S
   separately** (`ALE/MOM_ALE.F90:1537-1596`), mass weighting of T, S edge values only, `rho_ref`
   passed into the EOS. pyOCEAN's current `fv_pgf_rho` reconstructs the **density anomaly**
   ρ − ρ_ref(z) with PLM and does not call the EOS at quadrature points; with a nonlinear EOS that is a
   different discretisation (cabbeling/thermobaricity inside the layer are lost, and PLM-limited ρ
   ≠ ρ(PLM T, PLM S)). For a MOM6 comparison the T, S reconstruction + pointwise EOS is required.
   pyOCEAN's T − T_ref(z) device (exact for a reference profile) has no MOM6 counterpart; MOM6
   relies on `rho_ref` subtraction inside the EOS only.
3. EOS pressure in the PGF: **p = −g ρ0 (z − meanSL)** in Pa (g = 9.8, ρ0 = 1035 in NWA12), not
   in-situ hydrostatic pressure, not including η or p_atm (unless `SSH_IN_EOS_PRESSURE_FOR_PGF`).
   For mixing (kappa_shear, find_N2, KPP): p_surf + Σ g ρ0 h from the top.
4. N² conventions differ per scheme (§5): kappa_shear and set_diffusivity use derivatives at the
   interface pressure and interface-mean T, S (locally referenced); KPP uses full-EOS density
   differences at the top-interface pressure of layer k. Port each with its own convention;
   find_N2's bottom overwrite with N²_bot matters for the LOTW BBL port.
5. Surface buoyancy flux: derivatives at the top-layer T, S and p_surf (0 when absent), the same
   ∂ρ/∂T for the penetrating SW at depth.

**Pitfalls.**
- Units: MOM6 EOS functions take **Pa**; GSW takes **dbar** (×1e-4, `MOM_EOS_TEOS10.F90:19,55`); Roquet
  folds 1e-8 (kbar) into coefficients. Python `gsw` takes dbar and sea pressure.
- `gsw.rho` (Python GSW) is also the 75-term polynomial, so MOM6 `TEOS10` should match Python `gsw`
  to round-off (not verified here); MOM6 `ROQUET_RHO` will not match `gsw` to round-off (different fit).
- Salinity offsets: Roquet_rho 32 g/kg, Roquet_SpV and GSW 24 g/kg.
- S_A = 35.16504/35 · SP without δS_A; θ ↔ Θ via MOM6's own polynomial + 1.5 Newton steps
  (`MOM_temperature_convert.F90`), while the coupler uses `gsw_pt_from_ct`; the two are not
  bit-identical.
- MOM6 converts initial z-data to Θ/S_A when the EOS is TEOS10/Roquet regardless of
  `USE_CONTEMP_ABSSAL`; replicate the trigger exactly or document the difference.
- `C_P` is a free parameter (NWA12 3992.0, default 3991.86795711963); with Θ prognostic the TEOS-10
  value is the consistent one.
- SST to the bulk formula: average Θ over 1 m first, then convert to θ (MOM6 order).
- Wright analytic integral (if ever ported) interpolates EOS coefficients, not T, S, and only applies
  to PCM layers; the series is valid for |ε| < 0.34.
- `rho_ref` (RHO_PGF_REF) and `Rho0` are distinct: `rho_ref` is subtracted and sets `pa(1)`;
  `Rho0` sets the EOS pressure and the 1/ρ0 factor.
