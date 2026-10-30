# MOM6 kappa-shear (Jackson, Hallberg & Legg 2008) — reference read

Source: `~/pyOCEAN_refs/MOM6` at commit 0e9a561 (2026-10-07). All paths below are relative to
`~/pyOCEAN_refs/MOM6/src/`. Citations are `file:line`. This is a reference implementation, not ground truth.
Jackson et al. (2008, JPO 38, 1033–1053) is **not** in the local library (litsearch finds it only as a citation in
Hallberg & Adcroft 2009, Ilicak 2012, Chang 2026). Paper-side statements are marked **[verificar]**.

## 1. What is solved (per column, per call)

Interface quantities (K = 1 … nz+1, K = 1 surface, MOM6 orders top → bottom): κ (diffusivity), Q = TKE per unit
mass, N², S². Layers carry u, v, T, S, h.

### 1.1 Source of κ (local Ri)

`parameterizations/vertical/MOM_kappa_shear.F90:1646-1652` (also `:1243`, `:1276` in the time-step search):

    if N² < Ri_c S²:  K_src = 2 γ |S| · (Ri_c S² − N²) / (Ri_c S² + F_c N²)      [s⁻¹]
    else:             K_src = 0

i.e. K_src = 2 γ |S| F(Ri), F(Ri) = (Ri_c − Ri)/(Ri_c + F_c Ri), with γ = `SHEARMIX_RATE` (0.089),
Ri_c = `RINO_CRIT` (0.25), F_c = `FRI_CURVATURE` (−0.97). Written without dividing by S² (no Ri singularity).
If no interface has a source, κ = 0 everywhere and Q = TKE_min (`:1657-1665`).

### 1.2 κ equation (elliptic, steady within an iteration)

`:1783-1803` (tridiagonal, Dirichlet κ = 0 at top and bottom `:1678`):

    κ_K (h_Int,K / L_d,K² + 1/dz_{k−1} + 1/dz_k) − κ_{K−1}/dz_{k−1} − κ_{K+1}/dz_k = h_Int,K · K_src,K

i.e. continuous form  κ/L_d² − ∂²κ/∂z² = K_src  [verificar against Jackson eq. for κ].
Decay length (`:1785`, `:1790`):

    1/L_d² = (N²/λ² + f²)/Q + 1/L_bdry²,      λ = KAPPA_BUOY_SCALE_COEF (0.82)

Boundary length (`:1039-1049`):

    1/L_bdry² = [(d_top + d_bot)/(d_top d_bot)] · [(h_top + h_bot)/(h_top h_bot)] / LZ_RESCALE²

(Boussinesq: (1/d_top + 1/d_bot)², d = distance to surface / bed). This is what makes κ → 0 at both boundaries
**even inside a bottom overflow**: the bed is treated like a wall at distance d_bot.
f² = mean of the 4 corner f² (`:317-318`).
`KAPPA_N_OVER_S_SCALE_COEF2` (`lambda2_N_S`) is read (`:2208-2213`) but **never used** in the solver (only
declared `:56`, read). Inert parameter.

Truncation (`:1797-1817`): κ < KD_TRUNC → 0 (and the active range ends there); between 1× and 2× KD_TRUNC,
κ = 2(κ − trunc) (continuous ramp).

### 1.3 TKE equation (steady, implicit, per iteration)

`:1709-1773` (tridiagonal, Q(1) = q0 = TKE_BACKGROUND at the surface `:1728`, Q(nz+1) = TKE_min at the bed
`:1739`; the "no-flux" top/bottom BCs exist but are hard-wired off, `:1608-1609`):

    0 = ∂_z[(κ + κ0) ∂_z Q] + (κ + κ0) S² − κ N² − (Q − q0) · D,     D = √(c_N² N² + c_S² S²)

Discrete: coupling aQ_k = (½(κ_K + κ_{K+1}) + κ0)/dz_k (`:1718`); source (κ+κ0)S² + q0 D (`:1732`); the
buoyancy sink κ N² is linearised as Q · (κ/Q) N² with K_Q = κ/Q from the previous iterate (`:1733`).
D replaces JHL's c_N N + c_S S (commented `:1669`). c_N = `TKE_N_DECAY_CONST` 0.24, c_S = `TKE_SHEAR_DECAY_CONST`
0.14. TKE_min = max(TKE_bg, 1e-20 m² s⁻²) (`:1632`).

### 1.4 Coupled iteration (inner loop, `find_kappa_tke`)

`:1701-2041`. Alternates the Q solve and the κ solve (Picard), up to `MAX_RINO_IT` (default 50; DOME 25). When the
relative change of κ and Q drops below `Newton_err` (starts 0.2, `:1638`; halved after an aborted Newton step
`:2032-2038`) it switches to a full Newton solve of the coupled system (`:1825-2001`). Convergence: |Δκ| ≤
`KAPPA_SHEAR_TOL_ERR` (0.1) × (κ0 + κ) everywhere (`:2003-2040`). Returns κ, Q, K_Q, the source
K_src and a "local source" (K_src + κ0 terms + positive diffusive part) used by the outer time-step control
(`:2050-2066`).

`KAPPA_SHEAR_ITER_BUG` (default False, `:2267-2270`): the Newton derivative dκ/dQ used `US%m_to_Z` instead of
`dz_Int(K)` (`:1860-1866`): dimensionally inconsistent, "undercorrections when dz > 1 m". Implement the corrected
branch (`dz_Int`).

## 2. Time integration over the model step (outer loop, `kappa_shear_column`)

`:866-1374`. The returned κ is the **time average over Δt** of a κ that evolves as mixing erodes the shear
(the Hallberg 2000 idea: mixing reduces S², so a κ computed from the initial state overshoots) [verificar
Hallberg 2000, MWR 128].

1. **Background smoothing first** (`:1052-1088`): one implicit step of κ0 = `KD_KAPPA_SHEAR_0` diffusion of u, v,
   T, S over Δt. u, v: no-slip at the bed (`:1071-1074`), T, S insulating (`:1075-1078`). Then N², S² from that
   state (`:1167-1172`, `calculate_projected_state` with dt = 0). κ0 is *not* returned as part of κ.
2. Interface cell sizes h_Int: harmonic-mean-weighted split of each layer between its two interfaces
   (`:1091-1109`), so interfaces next to thin layers get thin cells and Σ h_Int = column thickness; top and
   bottom interfaces have zero size.
3. Gradients (`calculate_projected_state`, `:1379-1506`): S² = |Δu|²/Δz²,
   N² = max(0, (∂b/∂T ΔT + ∂b/∂S ΔS)/Δz) with Δz = ½(dz_{k−1} + dz_k) (`:1030-1037`), ∂b/∂T, ∂b/∂S from the EOS
   at interface T, S and pressure (`:1127-1147`). **N² is clipped at 0**: kappa-shear does no convection.
4. Loop `itt = 1 … MAX_KAPPA_SHEAR_IT` (13, `:1191`):
   - κ_out = `find_kappa_tke`(current state) (`:1198-1199`).
   - Find dt_now ≤ dt_rem by halving (up to (max_KS_it+1−itt)/2 times, `:1218-1266`) then 5 bisection
     refinements (`:1267-1294`): the state is projected by diffusing u, v, T, S with κ_out over ½ dt_test, and
     the new K_src must stay within [kappa_src − 0.95·local_src, kappa_src + 10·local_src]
     (`KAPPA_SHEAR_MAX_KAP_SRC_CHG` = 10, `:1012-1018`, `:1226-1230`) or within tol × accumulated source.
     `USE_RESTRICTIVE_TOLERANCE_CHECK` (False) picks min/max of the two tests (`:1245-1255`).
     dt_now = min(dt_test (1 + tol) + dt_inc, dt_rem) (`:1296`). The last iteration takes all dt_rem.
   - Predictor–corrector (`:1316-1357`): project with κ_out over dt_now → κ_pred; κ_mid = ½(κ_out + κ_pred);
     re-project with κ_mid → κ_pred again; κ_mid = ½(κ_out + κ_pred); accumulate κ_avg += κ_mid dt_now/Δt,
     Q_avg likewise; seed the next iteration with κ_pred.
   - Advance the column state (u, v, T, S) with κ_mid over dt_now (`:1361-1368`) — **only inside the solver**;
     the model's u, v, T, S are not changed here.
   - If κ_out ≡ 0, the remainder of Δt is credited with κ = 0 (`:1305-1314`).
5. Initial guess each call: κ = `KD_SEED_KAPPA_SHEAR` = 1 m² s⁻¹ (`:325`, `:700`, param `:2179-2183`). The
   previous step's κ is **not** used despite the `kappa_io` intent(inout) docstring (`:151-153`). Q is not state
   (TKE_turb is diagnostic only; see `docs/design/n2c_refs/tke_in_ale.md:48`).

Cost: per column, ≤ 13 outer iterations × (≤ 7 halvings + 5 refinements, each a 4-field tridiagonal projection)
+ 2–3 `find_kappa_tke` calls per outer iteration, each ≤ 50 Picard/Newton iterations of 2 tridiagonals. The code
comment says an acceptable step "is found long before the minimum is reached" (`:1229-1236`). No cost numbers in
the source; reputation in the literature of being expensive in global runs [verificar]. Columns are independent
(OpenMP over j, `:228-230`) — trivially Numba-`prange`-able.

## 3. Thin / vanished layers

`KAPPA_SHEAR_ELIM_MASSLESS` (default True, `:2230-2235`), `:254-303`:
- dz_massless = 0.1 √(κ0 Δt) (`:220-221`). With the defaults κ0 = 1e-7, Δt = 3600 s: 1.9 mm; DOME
  (κ0 = KD = 1e-4, DT_THERM = 3600): 6 cm.
- Going down the column, a layer starts a new merged layer only if it is thicker than dz_massless and the current
  merged layer has mass (`:263-264`); thin layers join the merged layer above (top layers join the first massive
  one). Sums of h, dz, u h, v h, T h, S h (`:268-279`). `KAPPA_SHEAR_MERGE_ML` merges the bulk mixed layers
  (layer mode only, `:2285-2291`).
- Back to model interfaces by linear interpolation in the fractional thickness kf within the merged layer
  (`:286-298`, `:355-363`).
pyOCEAN already ports this pattern for GLS/KPP (`core/components/closure.py:52-118`, `active_layers`,
`thin_layer_threshold`) with its own per-column threshold.

## 4. Where κ goes

- `Calculate_kappa_shear` returns Kd_shear = κ_avg and **Kv_shear = PRANDTL_TURB × κ_avg** (`:390-394`),
  TKE_turb (diagnostic). `PRANDTL_TURB` default 1.0 (`:2240-2242`).
- Diffusivity: added to the background in `set_diffusivity`:
  Kd_int = Kd_shear + ½(Kd_lay,k−1 + Kd_lay,k) (`MOM_set_diffusivity.F90:571-581`) — **additive**.
- Viscosity: Kv_shear is kept in `visc` and added in the **next** dynamics step's implicit vertical viscosity,
  averaged from the two h points to the face (`MOM_vert_friction.F90:2163-2205`; vertex version `:2208-2216`).
- KPP, if on, adds to Kd/Kv by default (`KPP_IS_ADDITIVE` = True, `MOM_CVMix_KPP.F90:419-422`, `:1015-1028`).
- ALE: Kd_shear and Kv_shear are remapped by `interpolate_column` only if `REMAP_AUXILIARY_VARS` = True
  (default False, `core/MOM.F90:2588-2592`, `:1947-1965`; `MOM_set_viscosity.F90:2937-2961`;
  `ALE/MOM_ALE.F90:1319-1355`). Default: kept by interface index.
- Restart: Kd_shear, Kv_shear (`MOM_set_viscosity.F90:2819-2830`) — but κ is re-seeded each call, so Kd_shear
  is effectively diagnostic; Kv_shear matters (lagged one step).

## 5. Call order (`MOM_diabatic_driver.F90`)

ALE path (`diabatic_ALE`, `:1288`; legacy `:553`; layer mode `layered_diabatic`, `:1942`):
1. `find_uv_at_h` (`:714-720`, `:1452-1458`): u, v to h points (area-weighted, `MOM_diabatic_aux.F90:503-560`).
2. `set_diffusivity` (`:729-733`, `:1467-1471`): kappa-shear **first** (`MOM_set_diffusivity.F90:413-441`),
   or CVMix shear (`:445-451`), then background, BBL, tidal, double diffusion; Kd_shear added (`:571-581`).
3. KPP (`MOM_diabatic_driver.F90:747-` …), additive to the interior Kd/Kv.
4. CVMix convection, then implicit tracer diffusion.
The Δt seen by kappa-shear is the thermodynamic step (DT_THERM).

Vertex option `VERTEX_SHEAR` (default False, `:2124-2127`; `Calc_kappa_shear_vertex` `:417-862`): u, v
h-weighted to corners (`:574-578`), T, S, h, dz 4-point masked means (`:580-597`); Kv at vertices used directly
(`Kv_shear_Bu`), Kd averaged back to tracer points by 4-point arithmetic mean (`:839-845`) or geometric /
thickness-weighted options (`:792-838`). Bugs default to `ENABLE_BUGS_BY_DEFAULT` (= True): `VERTEX_SHEAR_VISCOSITY_BUG`
(`:2130-2133`), `VERTEX_SHEAR_OBC_BUG` (`:2136-2143`). Tracer-point mode avoids all of these.

Other effects of USE_JACKSON_PARAM: BBL thickness capped at ½ HBBL (`MOM_set_viscosity.F90:929-931`, `:3081`,
`:3259`); only one shear scheme allowed (`MOM_CVMix_shear.F90:265-273`).

## 6. Parameters (MOM_input name, default, source line)

| name | default | line |
|------|---------|------|
| USE_JACKSON_PARAM | False | 2120 |
| VERTEX_SHEAR | False | 2124 |
| RINO_CRIT | 0.25 | 2159 |
| SHEARMIX_RATE | 0.089 | 2162 |
| MAX_RINO_IT | 50 | 2166 |
| KD_KAPPA_SHEAR_0 | max(KD, 1e-7) m² s⁻¹ | 2170-2178 |
| KD_SEED_KAPPA_SHEAR | 1.0 m² s⁻¹ | 2179 |
| KD_TRUNC_KAPPA_SHEAR | 0.01 × KD_KAPPA_SHEAR_0 | 2184 |
| FRI_CURVATURE | −0.97 | 2189 |
| TKE_N_DECAY_CONST | 0.24 | 2193 |
| TKE_SHEAR_DECAY_CONST | 0.14 | 2200 |
| KAPPA_BUOY_SCALE_COEF | 0.82 | 2204 |
| KAPPA_N_OVER_S_SCALE_COEF2 | 0.0 (unused) | 2208 |
| LZ_RESCALE | 1.0 | 2214 |
| KAPPA_SHEAR_TOL_ERR | 0.1 | 2219 |
| TKE_BACKGROUND | 0.0 m² s⁻² | 2226 |
| KAPPA_SHEAR_ELIM_MASSLESS | True | 2230 |
| MAX_KAPPA_SHEAR_IT | 13 | 2236 |
| PRANDTL_TURB | 1.0 | 2240 |
| VEL_UNDERFLOW | 0.0 | 2243 |
| KAPPA_SHEAR_MAX_KAP_SRC_CHG | 10.0 | 2248 |
| KAPPA_SHEAR_ITER_BUG | False | 2267 |
| KAPPA_SHEAR_ALL_LAYER_TKE_BUG | False | 2271 |
| USE_RESTRICTIVE_TOLERANCE_CHECK | False | 2276 |

JHL's three best-fit sets (`:2106-2112`): (Ri_c, rate, F_c, λ, c_N, c_S) = (0.25, 0.089, −0.97, 0.82, 0.24, 0.14),
(0.30, 0.085, −0.94, 0.86, 0.26, 0.13), (0.35, 0.088, −0.89, 0.81, 0.28, 0.12).

## 7. Gotchas for pyOCEAN

1. **MOM6 DOME uses PRANDTL_TURB = 0** (`MOM6-examples/ocean_only/DOME/MOM_input:228`): Jackson gives only
   diffusivity there, **no shear viscosity**. The momentum is held by KV = 1e-4, the BBL and biharmonic
   Smagorinsky (SMAG_BI_CONST 0.06). The statement in `validation/dome/README.md:309` ("una mezcla por Ri … es lo
   que usa el DOME de MOM6 con su ν_v bajo") is half right: it supplies tracer mixing, not the vertical momentum
   coupling that `kv_low` lacks. See `croco_and_others.md` §DOME.
2. Kappa-shear has no convective mixing (N² clipped ≥ 0, `:1499-1505`); pyOCEAN's constant closure also has none
   (CLAUDE.md N2a open item). A separate convective adjustment is needed alongside.
3. κ vanishes at the bed by construction (L_bdry, `:1036-1049`, Dirichlet κ = 0). The plume-interface mixing
   survives because the plume interface is ≥ 1 layer above the bed; a one-layer plume on a coarse hybrid grid may
   get little entrainment (relevant to the "plume is one or two thick layers" finding, README DOME+GLS point 3).
4. Δt matters: the solver integrates over the thermodynamic Δt with sub-steps; port the time averaging, not just
   the steady κ(Ri) — otherwise large Δt over-mixes (Hallberg 2000) [verificar].
5. Merging threshold scales with √(κ0 Δt); pyOCEAN's closures already use a per-column relative threshold —
   reuse it and document the difference (as for GLS).
6. No state between steps except the lagged Kv: restart needs only Kv (as `kpp_akv`).
