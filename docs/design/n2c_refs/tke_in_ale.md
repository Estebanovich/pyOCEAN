# Prognostic turbulence quantities (TKE, ε / q²l / ψ) through ALE regrid–remap: reference-source read

Purpose: input for the stratified GLS k-ε closure in pyOCEAN's ALE mode (stage N2c).
Sources are reference implementations, not truth; where they disagree with the literature this is said.
Versions: HYCOM = local `~/pyOCEAN/HYCOM-src/` (no version string; files cited by name:line);
MOM6 = `~/pyOCEAN_refs/MOM6` commit 0e9a561 (2026-10-07); CROCO = `~/pyOCEAN_refs/croco` commit ab63b58 (2026-04-07);
Oceananigans = `~/pyOCEAN_refs/Oceananigans` commit 8ab4604 (2026-10-28).
All HYCOM paths below are relative to `HYCOM-src/`; MOM6 paths relative to `MOM6/src/parameterizations/vertical/`.

Short answer: **only HYCOM (Mellor–Yamada 2.5, `mlflag=5`) carries prognostic turbulence variables through a hybrid regrid, and it does it by treating q² and q²l as ordinary layer-mean tracers in hybgen** (same PCM/PLM/PPM remap as T, S). MOM6 carries **no** prognostic TKE across steps (kappa-shear re-solves from a seed every call; TKE is output only). CROCO (σ only) steps tke/gls on w-levels with horizontal flux advection and optional first-order vertical advection. Oceananigans (z\*) carries TKE as a σ-weighted tracer (e = column content / σⁿ).

---

## 1. HYCOM (Mellor–Yamada 2.5, mlflag = 5)

Note: the local source holds the q2/q2l bookkeeping and the column solver `mxmyaij` (`mxkprf.F90:1798`) but the second half (diffusivities part B) was only skimmed.

### 1.1 Variables and location
- `q2` (TKE) and `q2l` (TKE × length scale), two time levels (m, n), array dims `(i,j,0:kkmy25+1,2)` (`mod_cb_arrays.F90:511-514, 1644-1645`); `kkmy25 = kk` is enforced (`blkdat.F90:1554-1566`). Old-time-level copies `oq2, oq2l` for the Robert–Asselin filter (`mod_cb_arrays.F90:51-52`).
- **Location**: header of `mxmyaij` (`mxkprf.F90:1806-1810`): q², q²l "are calculated on interfaces in a special vertical grid where interfaces are centred at the surface, bottom, and at mid-depths of each HYCOM layer (kdm+2 interfaces)". So index k = 1..kk is the **layer-centre** value of HYCOM layer k; k = 0 surface, k = kk+1 bottom. The "interfaces" of the M-Y grid are HYCOM layer centres; the M-Y layers are the staggered cells between centres (`z`, `dz`, `zz`, `dzz`, `mxkprf.F90:1914-1935`).
- Initial value `smll = 1e-8` everywhere incl. k = 0, kk+1 (`inimy.F90:26, 42-45`; `inicon.F90:163-175`; note the odd `q2(...,1)`/`q2l(...,2)` index pair there, likely a harmless HYCOM quirk).
- Mixing coefficients `vctymy, difqmy, diftmy` live on the same grid, 0..kk (`inimy.F90`, `mxkprf.F90:2126-2139`).

### 1.2 Algorithm per time step (numbered)
1. **Advection (tsadvc)**: q2 and q2l are advected layer by layer with the *same* mass fluxes (`uflx, vflx`) and the same `advem` routine as T/S/tracers, with the positive-definite flag `pdq2 = 1.0` (`mod_tsadvc.F90:1760-1762, 2036-2047`). Old values saved in `q2old/q2lold` (`:1924-1925`). Horizontal diffusion with `tsdff_2x` (`:2181`). Halo exchange `:1838-1839, 2144-2145`.
2. **Robert–Asselin filter** in thickness-weighted form, with `oq2/oq2l` (`mod_asselin.F90:68-72, 228-239`): `dpsold = dpold*oq2; dpsmid = dpmid*q2(m); dpsnew = dpnew*q2(n); q = 0.5*ra2fac*(dpsold+dpsnew-2*dpsmid)`, so layer content (dp·q²) is what is filtered, same as T/S.
3. **Vertical mixing (`mxmyaij`, called from `mxkprf.F90:425`)**:
   a. `dpmm = max(1 mm, dp)`; deepest "substantial" layer `klist` = last layer thicker than 10 cm (1 mm with `MASSLESS_1MM`), at least 2 (`:1859-1879`). Only layers 1..klist enter the M-Y grid; below that nothing is mixed (thin-layer guard by truncating the column).
   b. Build the staggered M-Y grid (z, dz, zz, dzz) from layer thicknesses (`:1914-1935`).
   c. Shear and buoyancy are evaluated **between adjacent layer centres** (= at the M-Y "layers"/staggered level k): `delu = u1d(k)-u1d(k-1)`, `dbloc(k) = g·Δθ/(dzz(k-1)·dh)` using layer-centre ρ (σ-θ or locally referenced density via `dsiglocdt/ds`) (`:1938-1995, 2032-2057`). So S² and N² are finite differences between layer means divided by the distance between layer centres, not by layer thickness. Stable-only N² beneath the mixed layer (`min(0, ...)` when `p > dpbl`, `:2040`).
   d. Floor: `q2, q2l ← max(smll, ·)` at both time levels for k = 0..kk+1 (`:1997-2004`).
   e. Production at M-Y level k: `prod = Kh·N² + Km·sef·S²` using the *previous-step* `diftmy/vctymy` (`:2059-2067`); length scale `turlen = q2l/q2`, stability `gh = min(0.028, l² N²/q²)`, `stf` dissipation modifier (`:2043-2092`).
   f. Implicit tridiagonal solves (forward/back sweep) for q2 (`:2018-2107`), boundary values from wind stress (`gg(1)`) and bottom drag (`q2(kmy1) = const1·|τ_b|`); then for q2l with `ee(2)=0`, `gg(2) = -κ z(2) h q2(1)`, `q2l(kmy1)=0` (`:2110-2160`); wall-proximity function `e2my` in dissipation. New diffusivities `difqmy, vctymy` blended 50/50 with the previous values and floored (`:2126-2139`).
4. **hybgen (regrid)**: `nums1d = ntracr + 4` columns in the 1-D remap array when `mxlmy` (`hybgen.F90:373`); q2 and q2l are packed as columns `ntracr+3, ntracr+4` (`:961-962`), remapped with the **same** `hybgen_pcm_remap / plm_remap / ppm_remap` as T, S, tracers (`:1463-1479`; PLM/PPM limiter in `hybgen_plm_coefs :2325`, `hybgen_ppm_coefs :2519`), and unpacked (`:1511-1512`). They are therefore **layer-mean quantities in the remap** (the staggered-centre semantics are ignored — the value at a layer centre is treated as the layer mean; this is a modelling approximation of HYCOM, not a derivation).
5. **Pre-remap layer moves in hybgen**: entrainment of a vanishing layer into the one above does `q2(k-1) ← q2(k-1) − q (q2(k-1) − q2(k))` (mass-fraction mix, `hybgen.F90:604-607, 679-686`); the "dense/light" swap moves (`:724-731`); when a layer interface is moved `p_hat ≠ 0`, `q2(k-1) += qtr (q2(k) − q2(k-1))` with `qtr = p_hat/max(p_hat, dp)` (`:862-879`); **massless layers copy the layer above** (`q2(k) = q2(k-1)`, `:937-939`).
6. **Conservation**: `lconserve = .false.` by default (`:320`); when on, a multiplicative `offset` rescales each remapped column so column integral Σ q·dz is conserved, q2 and q2l included (`:1447-1449, 1515-1521, 1545-1629`, sums `asum(ntracr+3/4)` at `:1599-1600`). Same hybgen routine `hybgencj` is used for the barotropic-time-level fix (`:1948-2118`).

### 1.3 Positivity / thin layers
- Positivity: floor `smll = 1e-8` at start of every mixing call (1.2 step 3d), `pdq2` positive-definite flag in advection, PLM/PPM limiter in remap (monotone ⇒ stays within neighbouring values ⇒ non-negative).
- Thin layers: `dpmm ≥ 1 mm` in the grid geometry, `klist` truncation at the last layer > 10 cm, `dpthin = 0.001·onemm` passed to remap (`hybgen.F90:366, 1463`), massless layers copy the layer above.
- Not carried: no N², S² at hybgen time; the closure recomputes them every step from the post-remap state.

---

## 2. MOM6 (ALE; kappa-shear, CVMix shear, ePBL)

### 2.1 Result: no prognostic TKE crosses the regrid/remap step
- `Calculate_kappa_shear` (`MOM_kappa_shear.F90:135`) has `kappa_io` intent(inout) described as "the value from the previous timestep, which may accelerate the iteration" (`:151-153`) but in the code the column initial guess is **re-seeded each call**: `kappa(K) = CS%kappa_seed` (`:325`). `tke_io` is `intent(out)` (`:156`) "TKE at each interface (not layer!)", stored in `visc%TKE_turb` (`core/MOM_variables.F90:324`; filled in `MOM_set_diffusivity.F90:423-432`). `TKE_turb` is only set/diagnosed, consumed nowhere else (grep: set_viscosity allocation 2833-2839 and diabatic_driver comments only). It is a **diagnostic**, not state; no entry in `MOM_ALE.F90`, no restart. The TKE inside `find_kappa_tke` is an *instantaneous quasi-steady equation* solved with TKE = κ/K_Q (`:1560-1700`, e.g. `TKE(K) = kappa(K)/K_Q(K)`, `TKE_min` floor) iterated `max_KS_it` times within the step.
- `MOM_energetic_PBL.F90` is diagnostic/energetic: no TKE field; only a stored boundary-layer depth used as first guess (`MLD_in = MLD_io`, `MOM_energetic_PBL.F90:452, 674-680`).
- `MOM_CVMix_shear.F90` (`:64-218`): stateless Richardson-number scheme.
So MOM6 sidesteps the problem by making the turbulence closure a diagnostic of the instantaneous resolved state (N², S²) on the current grid.

### 2.2 How kappa-shear builds N² and S² on interfaces of a remapped grid (numbered)
1. Per column, optionally **eliminate massless layers** (`eliminate_massless`, `:254-307`): a layer thinner than `dz_massless = 0.1 sqrt(dt·κ0)` (`:221`) is merged with its neighbour (`kc(k)` map; `h_lay, dz_lay, u0xdz, v0xdz, T0xdz, S0xdz` accumulate thickness-weighted sums; the top `nkml` layers are never merged). Only clusters of massless layers merge. `kf(k)` = fractional position used to interpolate results back.
2. A step of background diffusion (`k0dt = dt·κ0`) is applied to u, v, T, S on the reduced column with an implicit tridiagonal (`:1046-1080`): this regularises gradients across thin layers (no-slip bottom for u, v; insulating T, S).
3. Interface geometry: interface "thickness" `h_Int, dz_Int` is the **harmonic-mean-weighted** half-sum of the two adjacent layer thicknesses, with weights `wt_a, wt_b` constructed so that thin layers get thin interface cells and the totals add up (`:1081-1100`); `Idz`/`I_dz_int` are its inverses.
4. N² and S² at interface K (`calculate_projected_state`, called at `:1157`): `N²(K) = (dbuoy_dT·ΔT + dbuoy_dS·ΔS)·I_dz_int(K)`, `S²(K) = (Δu² + Δv²)·I_dz_int²`, with T/S/ρ at the interface from the mean of the two layer values and the pressure integrated from the top (`:1103-1155`). Stored as `N2_init, S2_init`; during iterations they are *projected* forward: u, v, ρ updated implicitly with the trial κ (`:1200-1356`; predictor–corrector with `tke_pred`, `kappa_pred`).
5. TKE and κ solved on the interfaces with a hexadiagonal system (`find_kappa_tke`, `:1509+`); `I_L2_bdry` wall-distance factor computed from the *harmonic* distances to the top and bottom (`:1040-1050`), K_Q = κ/TKE, TKE_decay `sqrt(C_N² N² + C_S² S²)` (`:1670`).
6. Results interpolated back to the original (unmerged) interfaces with `kf` (`:340-365`), time-averaged TKE `tke_avg` over the sub-iterations (`:1313, 1352`).
CVMix shear (`MOM_CVMix_shear.F90:130-157`): `dz_int = ½(dz(k-1)+dz(k)) + dZ_subroundoff`; `N² = Δρ_local(k, k-1)·g/ρ₀ / dz_int` with both densities evaluated at the interface pressure (pRef) (`:118-153`), `S² = (ΔU² + ΔV²)/dz_int²`, `Ri = max(0,N²)/max(S², 1e-10 s⁻²)`. The `subroundoff` regularises vanished layers; no merging.

### 2.3 Variables and location
Interface-based κ, Kv, TKE (nz+1); T-point (h-point) or vertex (`kappa_shear_at_vertex`) columns; layers carry u, v, T, S only.

### 2.4 Positivity / thin layers
`TKE_min = max(TKE_bg, 1e-20 …)`, κ=0 where Ri > Ri_crit (`:1640-1660`), massless-layer merging (2.2 step 1), background-diffusion step before computing gradients, `dz_subroundoff`. Reference comment: MOM6 follows Jackson et al. (2008) for kappa-shear; **[verificar]** details against the paper.

---

## 3. CROCO (σ only; GLS k-ε / k-ω etc.; `OCEAN/gls_mixing.F`)

Not ALE but relevant to the time-stepping of tke/gls on moving σ levels.
1. **Variables**: `trb(i,j,k,time,itke/igls)` on **w-levels** k = 0..N (N+1 levels), two time levels (nstp, nnew); `Akv, Akt` on w-levels too. Floors: `tke_min = 1e-10` (`:145`), `gls_min` derived consistently from `tke_min` and `eps_min` (`:320-325`), applied by `MAX(…, trb_min)` after each solve (`:749-764`).
2. **Advection of tke and gls** (inside `gls_mixing`, before the vertical implicit solve, `:340-470`): optional first-order upwind horizontal advection (`GLS_HADV_UP1`) with w-level fluxes `HUon_w = ½(Huon(k)+Huon(k+1))` (`:352-366`) and tendency divided by the w-cell thickness `2/(Hz(k)+Hz(k+1))` (`:369`); optional first-order upwind vertical advection (`GLS_VADV_UP1`) with `We` (`:431-460`). Without these CPPs tke_new = trb (pure local vertical mixing; `:413-414`).
3. **Shear**: `dudz = (u(k+1) − u(k)) · 1/(½(Hz(k)+Hz(k+1)))`-type difference between the two layers adjacent to w-level k, `shear2 = du²+dv²` (`:478-540`; 3-D deformation option `GLS_SHEAR_3D`). Buoyancy frequency `bvf` from the equation of state (Brunt–Väisälä at w-levels) elsewhere in the routine.
4. **Source terms**: `Sprod = cff1·Akv_old·shear2`, buoyancy production `Bprod`, dissipation `cff2·diss`; semi-implicit: if `(Bprod+Sprod) > 0` buoyancy production is explicit, else implicit via `-invG·Bprod` in the diagonal (`:600-665`). Tridiagonal solve in k with boundary values from surface/bottom friction velocity and law-of-the-wall lengths (`:674-730`).
5. **Limiters**: Galperin length limit `L_lim = galp sqrt(2 tke)/sqrt(max(eps, N²))` applied to gls after the solve (`:829-834`), `eps_min`, `tke_min`.
6. **No explicit layer-thickness-change correction**: the transported variable is not rescaled by the ratio of Hz(new)/Hz(old) in this routine (observation: the tendency uses the *current* Hz; CROCO relies on the σ-layer thickness change being small per step. I did not trace the free-surface-change path further: **[verificar]** if needed). `gls_smooth.h` is an optional horizontal smoothing (`:790, 805`).

## 4. Oceananigans (z\* vertical coordinate; CATKE, `TKEBasedVerticalDiffusivities/`)
1. TKE `e` is a prognostic **cell-centre tracer** like T, S. Under z\* the grid has a time-dependent stretching factor σ (`σⁿ, σ⁻, σ⁻¹ⁿ`).
2. Time stepping stores the product: `e^{n+1} = (σ⁻ e⁻ + Δt G) / σⁿ` (RK, `time_step_catke_equation.jl:280-290`) and `e += Δτ (α G_total − β G⁻)/σⁿ` for the AB2 case (`:245-262`), `G_total = slow_G + fast_G·σⁿ` — i.e. column content is conserved through the vertical stretch exactly as any tracer. The fast part (local production/dissipation) is integrated with the thickness-weighted tendency, the slow part (advection, diffusion) as for tracers.
3. κe and the mixing length `Le` are closure fields at cell faces/centres; positivity through the linearly implicit (`L`) dissipation `implicit_linear_coefficient` (`:322+`).
4. No regrid to a different grid (only vertical scaling of a fixed-topology z\*) so no remap question arises.

---

## 5. Cross-model comparison

| | TKE prognostic across regrid? | Location | Remap treatment | S², N² | Thin / vanished layers |
|---|---|---|---|---|---|
| HYCOM MY2.5 | yes (q², q²l, 2 time levels) | layer centres (staggered M-Y grid) | layer-mean tracer: PCM/PLM/PPM, optional multiplicative conservation (off), massless layer copies layer above | finite difference between layer centres / centre distance | dpmm ≥ 1 mm, `klist` truncation at > 10 cm, floor `smll` |
| MOM6 | no (diagnostic output, re-solved) | interfaces | n/a | interface; harmonic-weighted interface thickness, merged massless layers | `eliminate_massless`, background diffusion step, `dz_subroundoff` |
| CROCO (σ) | yes (tke, gls) | w-levels | n/a (σ moves smoothly), no thickness rescale | shear between adjacent layer means / ½(Hz+Hz), N² = bvf | tke_min / gls_min floors, Galperin limit |
| Oceananigans z\* | yes (e) | cell centres | σ-weighted tracer (e·σ conserved) | centre-to-centre differences | linearly implicit dissipation |

## 6. Options for pyOCEAN

Facts to design around: pyOCEAN GLS (`physics/mixing/gls.py`, CROCO port) carries `tke, gls` on w-levels in σ mode; ALE stores layer thickness `h` per column and remaps layer-mean tracers every step (PPM, `core/vertical/remap.py`), with possibly vanished layers and a `limit_outflow` mechanism.

**Option A – Carry (k, ψ) as layer-mean tracers (HYCOM-like).** Convert the w-level tke/gls to layer means (e.g. thickness-weighted average of the two interface values, or define them at layer centres as HYCOM does), advect and remap them with the existing tracer machinery (WENO5/FCT + PPM remap, `n_tracers` infrastructure), and rebuild interface values for the solver.
- Pros: reuses the passive-tracer stack (advection, FCT, sponge, restart, output already tested bit-for-bit); conserves column content Σ h·k with the same remap; closest to a production precedent (HYCOM MY2.5); simplest to implement.
- Cons: interface↔layer conversion adds smoothing and a representation error where the grid changes by more than a layer; ψ=k^m l^n is nonlinear in k and l, so remap k and ψ independently but then ε = cμ⁰³ k^1.5/l is not conserved (HYCOM does the same for q²l, so precedent exists); positivity must be enforced (PPM limiter + floor); HYCOM's own layer/centre inconsistency carried over unless the conversion is done carefully.

**Option B – Interface variables, remap by interpolation to new interfaces (CROCO/MOM6-in-spirit).** Keep k and ψ on interfaces; after the regrid, interpolate them in height (linear, monotone) from old interface positions to the new ones, with floors `tke_min, gls_min`, and optionally rescale to keep Σ k·Δz.
- Pros: no layer-mean conversion; natural for a closure whose shear/N² are interface quantities; variable-by-variable limiters can be tuned; consistent with how CROCO stores it.
- Cons: new remap code (the existing tracer remap works on layer means), horizontal advection must be done on interface variables with interface-averaged fluxes (like CROCO `HUon_w`), conservative treatment of vanished layers must be written; no production ALE precedent.

**Option C – Stateless / quasi-equilibrium closure at the current grid (MOM6-like).** Do not carry k, ψ; each step re-solve a quasi-steady (or a short-time-stepped) k-ε balance from the current N², S² starting from a seed, with TKE a diagnostic.
- Pros: removes the remap problem entirely and is trivially robust to vanished layers; precedent in MOM6; restart is trivial.
- Cons: loses memory/advection of turbulence, which is the point of a 2-equation model (decay after wind events, advected TKE in gravity currents); would differ from the σ-mode GLS and break the "same closure in all coordinates" goal; a quasi-steady k-ε is not what Canuto-A calibration assumes.

Practical points that apply to any option (from the sources): (i) floor k and ψ at every step and keep a Galperin limit after the solve (CROCO `:829-834`); (ii) compute shear and N² with an interface thickness that is the half-sum of the adjacent layers plus a tiny regulariser, or HYCOM's centre distance, never the single-layer thickness; (iii) in vanished layers (h < threshold) either merge for the closure solve (MOM6 `eliminate_massless`) or truncate the solved column (HYCOM `klist`) and copy the neighbour's value into the vanished layer (HYCOM `hybgen.F90:937-939`); (iv) store a reference of where the closure acts on the post-remap state so the first solve after the regrid uses consistent h, u, v, T; (v) advect with the same thickness fluxes as the layer tracers so that column content Σ h·k is the quantity conserved (HYCOM, Oceananigans); (vi) for restart bit-for-bit, k and ψ become part of the tagged restart variables (the closure state already lives in `SigmaState.closure`).
