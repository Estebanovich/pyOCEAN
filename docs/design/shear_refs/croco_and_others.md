# CROCO, MITgcm and the overflow reference configurations — shear mixing reference read

Sources: CROCO ab63b58 (`~/pyOCEAN_refs/croco/`), MITgcm 27256b2 (`~/pyOCEAN_refs/MITgcm/`), MOM6-examples
(`~/pyOCEAN_refs/MOM6-examples/ocean_only/`). Citations `file:line`.

## 1. CROCO: LMD interior Ri mixing (`OCEAN/lmd_vmix.F`, `LMD_RIMIX`)

Active only with `LMD_MIXING` (KPP family); the REGIONAL default sets LMD_SKPP, LMD_BKPP, **LMD_RIMIX**,
LMD_CONVEC, LMD_NONLOCAL (`OCEAN/cppdefs.h:250-262`). GLS_MIXING is the alternative (`cppdefs.h:253`).
CROCO's OVERFLOW test case defines no vertical mixing scheme at all (`cppdefs.h:1015-1033`).

Constants (Fortran `parameter`, not run-time, `lmd_vmix.F:108-138`): Ri₀ = 0.7, ν⁰_m = ν⁰_s = 50e-4 m² s⁻¹,
internal-wave background ν_w = 1e-4, κ_w = 1e-5 m² s⁻¹ (`lmd_nuwm`, `lmd_nuws`), convective ν_c = 0.1 m² s⁻¹.

Algorithm (`lmd_vmix_tile`, `:89-532`):
1. Shear at ρ points and w levels from the two adjacent u (v) faces (`:217-229`):
   ∂u/∂z = [(u_{i,k+1} − u_{i,k}) + (u_{i+1,k+1} − u_{i+1,k})] / [2 (z_r,k+1 − z_r,k)]; Ri = N²/max(S², 1e-10)
   with N² = `bvf` from `rho_eos.F:284-288` (or `:411` for the linear EOS).
2. `RI_HSMOOTH` (always on with LMD_RIMIX, `:144-148`): Ri averaged to ψ points (4-point, masked) and back
   (`:276-307`) — a horizontal 1-2-1-like filter.
3. `RI_VSMOOTH` (`:312-325`): in-place 0.25/0.5/0.25 filter for k = N−2 … 2 **bottom-up from the top index**,
   so each k uses the already-smoothed k+1 (recursive, like MOM6's).
4. Coefficients (`:346-353`, `:387-389`):

        Ri⁺ = max(0, Ri);  ν_sx = [1 − min(1, Ri⁺/Ri₀)²]³
        K_v = ν_w + ν⁰_m ν_sx + ν_c ν_sxc,   K_t = κ_w + ν⁰_s ν_sx + ν_c ν_sxc,   K_s = K_t

   ν_sxc = 1 if N² < 0 only when LMD_CONVEC and neither SKPP nor BKPP (`:372-380`): with KPP the convection
   comes from KPP. The internal-wave background ν_w, κ_w is always added (unless LMD_NUW_GARGETT, `:360-367`).
5. Surface/bed interfaces padded from the nearest interior value for the KPP blending (`:508-532`); the surface
   and bottom boundary layers (lmd_skpp/lmd_bkpp) then replace or blend these interior values.

Same functional form as CVMix LMD94 (`cvmix_shear.md` §2) with Ri₀ = 0.7, but always with horizontal + vertical
Ri smoothing and a non-zero background (1e-4 / 1e-5). The background conflicts with pyOCEAN's anti-goal (no
background diffusivity, M.Sc. thesis finding on IW spectra).

CROCO GLS (`gls_mixing.F`, already ported to `physics/mixing/gls.py`) is the CROCO route to shear mixing: P = ν S²
in the k equation, stratification through B = −κ_t N² and the Canuto-A stability functions of Ri-like α_N, α_M.

## 2. MITgcm

### 2.1 pkg/kpp interior Ri mixing (`pkg/kpp/kpp_routines.F`, `Ri_iwmix`, `:1032-1250`)

- S² at tracer points as the mean of the four adjacent face differences (`kpp_calc.F:453-470`; `KPP_SMOOTH_SHSQ`
  adds the 8 next faces with weight 1/8, `:471-490`). Note shsq is |ΔU|², not divided by Δz²; Ri =
  Δb·Δz / |ΔU|² (`kpp_routines.F:1125-1132`: `dblocSm (zgrid(ki) − zgrid(ki+1)) / max(Shsq, phepsi)`).
- Below the bottom (`ki ≥ kmtj`) Ri and N² copy the value above (`:1122-1124`).
- Optional vertical 1-2-1 smoothing `num_v_smooth_Ri` times (`ALLOW_KPP_VERTICALLY_SMOOTH`, `:1142-1160`,
  default 0, `kpp_readparms.F:128`).
- Shear function (`:1175-1180`, `Rig = MAX(diffus(i,ki,1), 0)` at `:1177`): f(Ri) = [1 − min(1, max(Ri, 0)/Ri∞)²]³ — same as L94 / CROCO.
  `KPP_SCALE_SHEARMIXING` multiplies by S⁴/(S⁴ + 1e-16) (Polzin 1996, from MOM5, `:1181-1186`).
- Convection (`:1168-1173`): f_con = [1 − min(1, (N²_c − max(N², N²_c))/N²_c)²]³ with N²_c = BVSQcon.
- Coefficients (`:1206-1208`): ν = viscArNr + f_con·difmcon + f(Ri)·difm0, κ_S,T = diffusKz + f_con·difcon +
  f(Ri)·dif0 (background additive).
- Defaults (`kpp_readparms.F:128-138`): Riinfty 0.7, BVSQcon −2e-5 s⁻², difm0 = difs0 = dift0 = 5e-3 m² s⁻¹,
  difmcon = difscon = diftcon = 0.1 m² s⁻¹. `EXCLUDE_KPP_SHEAR_MIX` turns the shear term off (`:1192-1197`).

### 2.2 pkg/ggl90 (TKE closure; brief)

Gaspar et al. (1990) one-equation TKE: K_m = c_k l √e (`ggl90_calc.F:447`, `:454`), c_k = 0.1, c_ε = 0.7,
e_min 1e-11, K_max 100 m² s⁻¹ (`ggl90_readparms.F:108-122`). Shear enters through the TKE production; the
Ri dependence is in the Prandtl number (`ggl90_calc.F:579-586`):
Pr = 1 for Ri < 0.2, Pr = min(10, 5 Ri) otherwise, K_h = K_m / Pr (`:598`). Mixing length options `mxlMaxFlag`
(`ggl90_mixinglength.F:166-200`). Equivalent role to pyOCEAN's GLS.

### 2.3 Legg's DOME MITgcm setup

Not on disk. pyOCEAN's MITgcm DOME harness (`validation/dome/mitgcm.py`) reproduces "Legg's physics at 10 km"
with constant viscAr = ν_v (0.5 m² s⁻¹), diffKrT = 0 and no mixing package (`mitgcm.py:106-108`; data.pkg only
OBCS, PTRACERS, RBCS, Diagnostics, MNC, `:150-151`); README states "sin mezcla por Ri" for the common physics
(`validation/dome/README.md:24-28`). Legg's isopycnal reference (HIM) used Hallberg (2000) Ri mixing
(README `:133`) [verificar in Legg et al. 2006, not in the local library].

## 3. Reference overflow configurations

### 3.1 MOM6 DOME (`MOM6-examples/ocean_only/DOME/`)

Layer mode (`USE_REGRIDDING = False`, `MOM_parameter_doc.all:39`; `COORD_CONFIG = "layer_ref"`, `:516`), NK = 25,
DT = 1200 s, **DT_THERM = 3600 s** (`MOM_input:27`, `:31`) — the kappa-shear Δt.
Shear mixing: `USE_JACKSON_PARAM = True`, `MAX_RINO_IT = 25` (`MOM_input:377-384`), everything else default
(`MOM_parameter_doc.all:1504-…`: RINO_CRIT 0.25, SHEARMIX_RATE 0.089, KD_KAPPA_SHEAR_0 = 1e-4 (= KD),
KD_TRUNC 1e-6, FRI_CURVATURE −0.97, TKE_N/S 0.24/0.14, λ 0.82, TOL 0.1, MAX_KAPPA_SHEAR_IT 13, ITER_BUG False).
**`PRANDTL_TURB = 0.0`** (`MOM_input:228`): no shear viscosity. Momentum: KV = 1e-4 (`:247`), quadratic drag
CDRAG = 0.002 over HBBL = 10 m with DRAG_BG_VEL 0.05 (`:230-241`), biharmonic Smagorinsky SMAG_BI_CONST = 0.06
with AH_VEL_SCALE 0.003 (`:283-290`). Diffusivity: KD = 1e-4 (`:373`) + Jackson + BBL mixing (BBL_EFFIC 0.2,
`MOM_parameter_doc.all:1427`). In layer mode the diffusivity acts through `layered_diabatic`
(`MOM_diabatic_driver.F90:1942`, `:2178-2182`) as diapycnal entrainment.

### 3.2 MOM6 flow_downslope (Ilicak et al. 2012 overflow; `flow_downslope/{layer,rho,sigma,z}`)

No shear mixing in any variant: USE_JACKSON_PARAM, USE_LMD94, USE_PP81 all False
(`z/MOM_parameter_doc.all:1637`, `:1657`, `:1660`; same in layer/rho/sigma). KD = 0, KV = 1e-4, CDRAG = 0,
KH = 1e4 with KH_VEL_SCALE 0.003 (`common/MOM_input:225`, `:237`, `:286-288`), DT 300 / DT_THERM 900. `rho` is
marked broken (`rho/__NOTE__`). Ilicak et al. (2012) also run the overflow with ν_v = 1e-4 and zero explicit
tracer diffusion (local library, Ilicak_2012 text lines ~383, ~973): the Ilicak overflow benchmark is a
**spurious-mixing** test and needs no shear parameterisation; adding one changes the case.

### 3.3 Hallberg & Adcroft (2009)

Uses DOME with Jackson et al. (2008) shear mixing as the source of plume entrainment (local library,
Hallberg_Adcroft_2009 text ~1391, ~1416, ~1436).

## 4. Summary of the Ri functions

| scheme | κ(Ri) | Ri_c | max | background | time integration |
|--------|-------|------|-----|------------|------------------|
| JHL 2008 (MOM6) | from κ/L_d² − ∂²κ/∂z² = 2γ|S|(Ri_c−Ri)/(Ri_c+F_c Ri), with TKE eq. | 0.25 | none (energy-limited) | κ0 only smooths inputs | sub-stepped, time-averaged |
| L94 (CVMix, CROCO, MITgcm) | ν⁰[1 − (Ri/Ri₀)²]³ | 0.7 (MOM6 0.8) | 5e-3 | CROCO 1e-4/1e-5; MITgcm viscAr/diffKr; CVMix 0 | none |
| PP81 (CVMix) | ν₀/(1+αRi)ⁿ, κ = ν/(1+αRi) | none | 0.01 | 0 | none |
| GGL90 (MITgcm) | TKE with Pr(Ri) = max(1, min(10, 5Ri)) for Ri ≥ 0.2 | — | 100 | e_min | prognostic TKE |
