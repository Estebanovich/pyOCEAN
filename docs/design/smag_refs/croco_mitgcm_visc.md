# Smagorinsky / horizontal viscosity in CROCO and MITgcm (reference read, 2026-10-30)

Paths: CROCO = `~/pyOCEAN_refs/croco/OCEAN/` (version in the local checkout, no tag read), MITgcm = `~/pyOCEAN_refs/MITgcm/`.
Reference implementations, not truth. Nothing was modified.

## 1. MITgcm (pkg/mom_common, pkg/mom_fluxform, pkg/mom_vecinv)

### 1.1 Smagorinsky formula (Laplacian, "C2")
- Header formula: `A_h = viscAh + 0.25 L^2 viscAhGrid/dt + (viscC2smag/pi)^2 L^2 sqrt(Tension^2 + Strain^2)`; biharmonic `... + 0.125 L^4 (viscC4smag/pi)^2 sqrt(T^2+S^2)` — `pkg/mom_common/mom_calc_visc.F:19-31`.
- Coded: `smag2fac=(viscC2smag/pi)**2`, `smag4fac=0.125*(viscC4smag/pi)**2` — `mom_calc_visc.F:196-197`. Smag is active iff `viscC2smag` or `viscC4smag` /= 0 (`:191-193`). Default `viscC2smag = viscC4smag = 0` — `model/src/set_defaults.F:129,148`. Suggested range 2.2-4 (Griffies & Hallberg 2000), 0.2-0.9 (Smagorinsky 1993) — `mom_calc_visc.F:61-63`. NOTE the (C/pi)^2 convention: C=2.2 gives (C/pi)^2 = 0.49, i.e. Cs_classic ~ 0.7 on L^2 with |D| = sqrt(T^2+S^2) (no factor sqrt2).
- Strain definition (no factor 2, no sqrt 2): tension `T = (dyG u|_E - dyG u|_W - dxG v|_N + dxG v|_S)/rA` (= du/dx - dv/dy at cell centre) — `mom_calc_tension.F:51-56`; shear strain `S = (dyC v|_i - dyC v|_{i-1} + dxC u|_j - dxC u|_{j-1})/rAz` (= dv/dx + du/dy at the corner/vorticity point) — `mom_calc_strain.F:54-59`. Cell-centre value: `sqarg = T^2 + 0.25(S_{i+1,j}^2 + S_{i,j+1}^2 + S_{i,j}^2 + S_{i+1,j+1}^2)`; corner value: `S^2 + 0.25(T_{i,j}^2 + T_{i,j-1}^2 + T_{i-1,j}^2 + T_{i-1,j-1}^2)` — `mom_calc_visc.F:483-495, 633-645`. So D_MIT = sqrt((du/dx-dv/dy)^2 + (dv/dx+du/dy)^2), and D_CROCO^2 = 2 dudx^2 + 2 dvdy^2 + S^2 = T^2 + S^2 + (du/dx+dv/dy)^2 (see 2.1): they differ only by the divergence term, which MITgcm omits.
- Length scale: `L^2 = 2/(1/dxF^2 + 1/dyF^2)` at tracer points (harmonic-mean-like, = dx^2 for dx=dy), `L2_Z` same at corners; `useAreaViscLength=.TRUE.` switches to the cell area `L^2 = rA` — `pkg/mom_common/mom_init_fixed.F:94-107, 120-130`; default false — `model/src/set_defaults.F:207`. `L4rdt = L^4/(32 dt)` — `mom_init_fixed.F:113-114`.
- Result: `viscAh_D = smag2fac * L^2 * sqarg` (`mom_calc_visc.F:493-495`), added to `viscAh + viscAhGrid*0.25 L^2/dt (+Leith)` into `Alin` (`:503-504`), then clipped (1.2).
- A separate `smag3D` package (`mom_calc_smag_3d.F`, `useSmag3D`, `smag3D_coeff` default 1e-2 — `set_defaults.F:119,205`) exists for 3-D Smagorinsky; not the one above, not read.

### 1.2 Limits / caps (all applied after summing)
- Floor: `A_min = max(viscAhGridMin*L^2/(4 dt), Uscl)`; `Uscl = sqrt(2)/viscAhReMax * sqrt(KE L^2)` (grid Reynolds number floor) — `mom_calc_visc.F:171-174, 393-395, 514-515`. `viscAhGridMin=0`, `viscAhReMax=0` by default — `set_defaults.F:122,125`.
- Cap (CFL): `A_max = min(viscAhGridMax * 0.25 L^2/dt, viscAhMax)`; `viscAh = min(A_max, max(A_min, Alin))` — `mom_calc_visc.F:516-518` (corner: `:661-665`). Biharmonic: `A4_max = min(viscA4GridMax * L^4/(32 dt), viscA4Max)` — `:529-531`. Defaults `viscAhGridMax = viscAhMax = viscA4GridMax = viscA4Max = 1e21` (i.e. no cap unless set) — `set_defaults.F:123-124,142,144`. Recommended 1 for GridMax (`mom_calc_visc.F:66-67`). `dt` is `deltaTMom` (`:166-167`).
- In the constant-viscosity (non-variable) branch the old formula `min(viscAh + viscAhGrid*rAw/dt, viscAhMax)` is used for side drag — `mom_u_sidedrag.F:71-76`.

### 1.3 Stress form
- Flux-form momentum (`vectorInvariantMomentum=.FALSE.`, default): NOT a stress tensor; Laplacian of each component, `F_x = -A_D (u_{i+1}-u_i) dyF drF hFacC / dxF`, `F_y = -A_Z (u_j - u_{j-1}) dxV drF hFacZ / dyU` — `pkg/mom_fluxform/mom_u_xviscflux.F:56-67`, `mom_u_yviscflux.F:57-72`; with biharmonic via `del2u` (`mom_u_del2u.F`). `A_D` (centre) and `A_Z` (corner) are the variable Smagorinsky fields passed from `MOM_CALC_VISC` (`mom_fluxform.F:454-456, 613-618`).
- Vector-invariant: default divergence/vorticity form `MOM_VI_HDISSIP` (grad(div) - curl(vort)); `useStrainTensionVisc=.TRUE.` (default false, `set_defaults.F:208`) selects the true tension/strain stress-tensor form `MOM_HDISSIP`, harmonic only (biharmonic STOPs) — `pkg/mom_vecinv/mom_vecinv.F:418-434`, `pkg/mom_common/mom_hdissip.F:45-85`. In that form `uDissip = [dyF^2 A_t T]_{i-1}^{i} /(dyG^2 dxC) + [dxV^2 A_s S]_{j}^{j+1}/(dxC^2 dyG)` (`:45-61`), A_t at centres, A_s at corners.
- hFac: viscous fluxes carry `hFacC`/`hFacZ` and `drF` (partial-cell thickness-weighted; flux form above). Tension/strain themselves "ignore lopping factors" (`mom_calc_tension.F:49`, `mom_calc_strain.F:52`); `hFacZ` (corner open fraction) comes from `MOM_CALC_HFACZ` (`mom_fluxform.F:280`).
- Smag in the strain-tension path uses "masked strain as if free-slip since side-drag is computed separately" — `mom_vecinv.F:419-420`.

### 1.4 Boundary condition
- `no_slip_sides` default `.TRUE.` (`set_defaults.F:132`), `sideDragFactor=2` (`:135`). No-slip is a body force in cells next to walls: `-(hFacZClosed dxV/dyU /(rAw hFacW)) * 2 * A_Z * u` (variable A uses `viscAh_Z`) — `mom_common/mom_u_sidedrag.F:100-135`, applied `mom_fluxform.F:656-668`. Free slip: `no_slip_sides=.FALSE.` (no wall term; closed corner faces have hFacZ=0 so no flux). Vorticity at walls multiplied by `sideMaskFac = sideDragFactor` only if no-slip — `mom_fluxform.F:264-269, 348-350`.

### 1.5 Time level
- Explicit. Dissipation computed from `uFld = uVel` (time n, `mom_fluxform.F:315`) and enters the tendency `gU` (`guDiss`), then the standard Adams-Bashforth extrapolation with `abEps` applies like every explicit term (AB2 file `model/src/adams_bashforth2.F`; not traced further). Viscosity coefficients are recomputed every step from the same velocity (no lag). Stability cap uses `deltaTMom`.

## 2. CROCO (OCEAN/)

### 2.1 Smagorinsky formula (`UV_VIS2` + `UV_VIS_SMAGO`)
- CPP: `UV_VIS_SMAGO` => `VIS_COEF_3D` (`cppdefs_dev.h:351-353`); `UV_VIS_SMAGO_3D` additionally defines `UV_VIS2`, `TS_DIF2`, `UV_VIS_SMAGO`, `TS_DIF_SMAGO` (`:345-350`). Default orientation `UV_MIX_S` (along sigma) unless `UV_MIX_GEO` (`:337-341`).
- Comment block: `A = Cs L^2 D`, `D = sqrt(2 D_ij D_ij)`, horizontal `L = sqrt(dx dy)` — `hmix_coef.F:106-124`.
- Coded: `defrate = (du/dx)^2 + (dv/dy)^2 + 0.5 (dv/dx + du/dy)^2` at rho points, with `du/dx = (u_{i+1}-u_i) pm`, and the cross terms from a 0.125-weighted four-point average of u, v differences — `hmix_coef.F:131-141`. Then `visc3d_r = horcon * om_r*on_r * sqrt(2*defrate)`, i.e. `A = horcon * dx*dy * D`, `D^2 = 2 dudx^2 + 2 dvdy^2 + (dvdx+dudy)^2` — `hmix_coef.F:190-203`. `horcon = 0.1` for plain `UV_VIS_SMAGO` (`:49`), `0.01` with `UV_VIS_SMAGO_3D` (`:46-47`). So Cs L^2 D with Cs=0.1 (equivalent classical (Cs)^2=0.1 -> Cs~0.32), L^2 = cell area, no pi. Includes the divergence (2 dudx^2 + 2 dvdy^2), unlike MITgcm (T^2 = (dudx - dvdy)^2).
- 3-D variant adds vertical shear and a Lilly buoyancy factor `sqrt(max(0, 1 - (N^2/D^2)/Ric))`, Ric=0.25, Pr=0.3 (`hmix_coef.F:144-183, 195-203, 46-47`); not needed for 2-D horizontal.
- Corner viscosity `visc3d_p` = 0.25 sum of four neighbouring rho values — `hmix_coef.F:312-318` (psi points). Edge values copied from interior (`visc3d_r(Istr-1,..)=visc3d_r(Istr,..)`, `:236-262`).

### 2.2 Combination with constant / sponge viscosity and caps
- Not additive: coefficient used is `max(0, visc2_r, visc2_sponge_r, visc3d_r)` — `uv3dmix_S.F:179-186` (rho points), `:206-213` (psi points, with `0.25*` and the four-Hz sum, `:214`), vertical-face/NBQ copies `:334-341`. `visc2` is the namelist `lateral_visc` first value (`read_inp.F:2396-2401`).
- NO stability cap and no Re floor in the CROCO code read (no dt/dx^2 limiter in `hmix_coef.F` or `uv3dmix_S.F`). Sponge viscosity: `visc2_sponge_r = (0.01 /(pm pn dt)) * wrk` (`set_nudgcof.F:224`; fine/child grid 0.005, `set_nudgcof_fine.F:151`), i.e. the explicit-limit scale dx^2/(100 dt) times a [0,1] sponge shape.

### 2.3 Stress form and weighting
- `UV_MIX_S` (`uv3dmix_S.F`): divergence of the symmetric stress tensor `s_xx = -s_yy = du/dx - dv/dy`, `s_xy = s_yx = du/dy + dv/dx`, along sigma surfaces, Wajsowicz (1993) / Griffies & Hallberg (2000) — header `uv3dmix_S.F:78-111`. Normal stress at rho points: `cff = Hz * A_r * (pmon (pn_u u)|_i^{i+1} - pnom (pm_v v)|_j^{j+1})`, `UFx = on^2 cff`, `VFe = om^2 cff`; shear stress at psi points `cff = 0.25 A_p (sum of 4 Hz) (pmon (pn_v v)|_{i-1}^i + pnom (pm_u u)|_{j-1}^j)`, `UFe = om_p^2 cff`, `VFx = on_p^2 cff` — `uv3dmix_S.F:178-229`. Thickness-weighted (Hz). `UV_MIX_GEO` is `uv3dmix_GP.F` (geopotential rotation; same coefficient choice at `uv3dmix_GP.F:323-328, 352-353, 430-435`), not read in detail.
- Update: `u_new(Hz*u) += dt * pm pn * (pn(UFx_i - UFx_{i-1}) + pm(UFe_{j+1} - UFe_j))`; also added to barotropic forcing `rufrc` — `uv3dmix_S.F:244-252, 283-290`.

### 2.4 Boundary condition
- Via `pmask` at psi (corner) points: `pmask = rmask product of the 4 cells` (0 at coasts => free slip, shear stress zero); `if (gamma2 < 0) pmask = 2 - pmask` => no-slip (2 at coast) — `setup_grid1.F:166-170`, applied `uv3dmix_S.F:230-232`. `gamma2` is a namelist/scalars.h parameter (default not read here).

### 2.5 Time level
- Explicit forward-Euler increment with `dt` on the velocity at `nstp` (`u(..,nstp)` in the stress, `uv3dmix_S.F:~165-225`), added to the Hz*u at target index `indx=3-nstp` (`:121`). Smag coefficient uses `u(..,nrhs)` (`hmix_coef.F:131-141`); `hvisc_coef` is called each 3-D step just before `prsgrd`/`rhs3d` (`step.F:484`) and, for the barotropic-only case, at `step.F:775`; nrhs=nstp in the predictor (`step.F:188-189`). `uv3dmix` is called after `pre_step3d` (`step.F:508-510`). So: coefficient and stress from the same (old) time level, no lag beyond the time-stepper's own.

## 3. Viscosity actually used in the reference runs of our benchmarks

| Run (data file) | Momentum form | Horizontal viscosity | Smag / Grid / caps | no_slip | dt | Vertical |
|---|---|---|---|---|---|---|
| DOME 10 km `mitgcm_z_common` (`/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/10km/mitgcm_z_common/data:4-5,16`) | vector-invariant | viscAh = 50 m^2/s | none set (viscC2smag=0, viscAhGrid=0, caps 1e21) | sides/bottom FALSE (free slip) | 600 s | viscAr 0.5, implicit, quad. bottom drag 2e-3 |
| DOME 10 km `mitgcm_z_legg` (`.../mitgcm_z_legg/data:4-5`) | vector-invariant | viscAh = 50 | none | sides/bottom TRUE | 600 s | viscAr 0.5 |
| DOME 10 km `mitgcm_z_kv_low` (`.../mitgcm_z_kv_low/data:4`) | vector-invariant | viscAh = 50 | none | FALSE | 600 s | viscAr 1e-4 |
| DOME 5 km `mitgcm_z_common` (`.../5km/mitgcm_z_common/data:4-5,16`) | vector-invariant | viscAh = 25 | none | FALSE | 300 s | viscAr 0.5 |
| Ilicak 2012 benchmark MITgcm (`~/pyOCEAN/validation/ilicak_benchmark/mitgcm.py:5,70-71,82`) | flux-form | viscAh = nu (the case's 3 values, 1/10/100/1000 style) | none | FALSE | per case | viscAr 1e-4 |
| BayIW `verification/BayIW_Rectan/run_expand` (+nobay, 2layer; `/Volumes/ESTEBAN_MIT/MITgcm/verification/BayIW_Rectan*/run_expand*/data:4-7,58`) | flux-form (vectorInvariantMomentum not set) | viscAh = 100 m^2/s | none | FALSE | 30 s | viscAr 1e-5; diffKhT = diffKhS = 100, diffKrT 1e-5 |
| BayIW 2-D / cut runs (`/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2d/mitgcm/*/data`, `bayiw_cut/mitgcm/*/data`, lines 4-7,56) | flux-form | viscAh = 100 | none | FALSE | 30 s | viscAr 1e-5 |

No MITgcm benchmark run of ours used Smagorinsky or viscAhGrid; all use constant viscAh (the MOM6 Ilicak/DOME counterparts were not read here).

## 4. Points to decide for pyOCEAN
1. Strain convention differs: MITgcm D = sqrt(T^2+S^2) with (C/pi)^2; CROCO D = sqrt(2 Dij Dij) incl. divergence with Cs L^2 (0.1). State the convention and which Cs maps to which.
2. MITgcm caps (CFL, `viscAhGridMax*0.25 L^2/dt`) and optional Re floor exist; CROCO has none (relies on explicit dt choice) — a cap is advisable for AB3 + explicit.
3. Combination: MITgcm adds constant + Smag then clips; CROCO takes the max of constant, sponge, Smag.
4. Stress form: CROCO uses the thickness-weighted symmetric tensor along sigma (matches a layer model); MITgcm flux form is a plain Laplacian with hFac weights (strain-tension form only in vector-invariant + `useStrainTensionVisc`).
5. Wall BC: both give free slip by zeroing corner shear at coasts; no-slip is a doubled shear (CROCO pmask=2) or a 2x side-drag body force (MITgcm).

## Summary (5 lines)
1. MITgcm: A = (C/pi)^2 L^2 sqrt(T^2+S^2) (L^2 = 2/(1/dx^2+1/dy^2) or area), added to viscAh + viscAhGrid*L^2/(4dt), clipped to [GridMin, min(viscAhGridMax*0.25L^2/dt, viscAhMax)] and Re-floor; defaults off (C=0, caps 1e21); explicit, time n, hFac-weighted flux-form Laplacian (stress-tensor only with useStrainTensionVisc).
2. CROCO: A = 0.1 dx dy sqrt(2 D_ij D_ij) (0.01 in the 3-D variant), used as max(visc2, sponge, Smag), no stability cap, thickness-weighted symmetric stress tensor along sigma (uv3dmix_S.F), explicit forward Euler on u(nstp), free/no-slip through pmask (gamma2).
3. Both are Laplacian-primary (MITgcm also has a biharmonic C4smag; CROCO UV_VIS4 not used with Smag); both evaluate coefficient and stress from the same time-level velocity.
4. Our reference runs (DOME MITgcm 10/5 km: viscAh 50/25, free slip, vector-invariant; BayIW and Ilicak MITgcm: viscAh 100/nu, free slip, flux form, dt 30 s) all use constant viscAh, no Smagorinsky.
5. Report: /Users/estebancruz/pyOCEAN/docs/design/smag_refs/croco_mitgcm_visc.md (psi-point averaging lines of hmix_coef.F:312-318 and the AB handling of guDiss were not traced line-exact).
