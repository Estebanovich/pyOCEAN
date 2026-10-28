# pyOCEAN N1 map: `SigmaBaroclinic` as it is today

All references are to `/Users/estebancruz/pyOCEAN` at commit ac5fd09. `SB` means `core/dynamics/sigma_baroclinic.py` (2314 lines; the class runs from :934 to :2314). `SWD` means `core/dynamics/sigma_wd.py`. `WD` means `core/dynamics/wetting_drying.py`. The constructor has 56 arguments (`SB:1019-1037`).

## 1. Constructor arguments by concern

The composition is nested: `SigmaBaroclinic` builds a `SigmaWetDry` (`self.core`, `SB:1095-1097`), and that builds a `WetDryFreeSurface` (`self.fs = self.core.fs`, `SB:1121`; `SWD:177`).

| Concern | Arguments | Where checked or stored |
|---|---|---|
| Grid / geometry | `h_grid`, `bathy` (NaN = wall, H ≤ 0 = floodable), `n_layers`, `dsigma`, `g` | `dsigma` is checked in `SWD:181-185`. `f_q` comes from `h_grid` (`SB:1122`). The ALE vector-invariant geometry `self._vi` (dx_u, dy_v, area_*, dx_t, dy_t, area_q, mask_q) is built **only in ALE** (`SB:1224-1228`). |
| Vertical coordinate / ALE | `vertical_coordinate` ('sigma_eulerian' or 'ale'), `regrid` (sigma/zstar/hybrid), `regrid_params`, `remap_scheme`, `h_thin_bed`, `init_iterations`, `thickness_flux` (ppm/upwind), `max_truncations` | `_setup_ale` at `SB:1164-1228`. Required keys per rule: `SB:1197-1199`. Hybrid floor warning: `SB:1200-1209`. Floodable cells need σ or a hybrid σ-zone: `SB:1210-1213`. No open edges next to floodable cells: `SB:1214-1220`. Optional keys (`d_sigma`, `d_hybrid`, `h_min`, `keep_isopycnal`, `max_h_top`) are read with defaults in `core/vertical/regrid.py:387-393`. |
| EOS / density / reference | `rho0`, `t_coef`, `t_ref`, `rho_ref` (None, 'auto' or a callable) | The linear T-only EOS is the method `density` (`SB:1351-1353`). `'auto'` is **replaced in place by `initial_state`** (`SB:1251-1254`, `:1302-1305`). |
| Momentum | `momentum` ('flux' or 'vector_invariant'), `ke_scheme` ('arakawa' or 'gudonov'), `adv` ('flux', 'upwind3', 'centred'), `ah` | `momentum`/`ke_scheme` are checked in `SB:1171-1179`. vector_invariant requires ALE (`SB:1174`). **`adv` is never validated** (stored at `SB:1120`). It is ignored when vector_invariant is on (`SB:1589`, `:1597`). |
| Tracer advection | `tracer_scheme` (weno5, weno5js, upwind3, upwind1), `tracer_time` ('ssprk3' or 'pc'), `tracer_limiter` ('fct' or 'none') | Checked in `SB:1041-1046`. The FCT default and requirements (ALE + ssprk3 + WENO5) are at `SB:1129-1138`. |
| PGF | `pgf` ('jacobian', 'twoterm', 'fv') | `SB:1038-1040`. Allowed in both σ and ALE. |
| Free surface | `theta`, `theta_bc`, `solver` ('direct' or 'cg'), `d_min` | The solver default depends on the coordinate: cg in ALE, direct in σ (`SB:1093-1094`). `theta_bc=None` becomes ½ (`SB:1064-1065`). WetDryFreeSurface also has `precond` and `refactor_iterations` (`WD:90-107`), which **are not exposed** by SB. |
| Vertical mixing | `closure` ('constant', 'gls', 'parabolic', 'none'), `nu_const`, `kv` (tracer K_v, constant), `kh` (tracer K_h along layers) | `closure` is validated in `SWD:175` (`CLOSURES`, `SWD:49`). ALE forces constant/none (`SB:1192-1194`). The closure sets **momentum ν only**; tracers always use the constant `kv` (`SB:2201-2205`, `:2304-2308`). |
| Bottom drag | `z0` (σ log-law, passed to SWD), `bottom_drag` (None or 'quadratic'), `cd`, `drag_bg_vel`, `hbbl` | ALE requires `z0=None` (`SB:1189-1191`). 'quadratic' requires ALE (`SB:1158-1162`). These are mutually exclusive, and the exclusion is checked. |
| Open boundaries | `open_boundaries`, `sponge_cells`, `sponge_tau`, `orlanski_cmax`, `sponge_tracer`, `open_boundary_state` (t → {edge: OBData}), `open_boundary_mode`, `obc_tau_in`, `obc_tau_out` | `SB:1047-1063`. A sponge needs open edges. A state provider needs open edges. 'clamped' needs a provider. Clamped edges get no sponge (`SB:1098`). Masks `_bc_mask`, `_ext_cols`, `_bc_u`, `_bc_v`: `SB:1101-1117`. Flather faces go to SWD/WD (`SB:1083-1097`). |
| Interior sponge | `sponge_rate`, `sponge_T`, `sponge_zw`, `sponge_tr` | `SB:1143-1157`. ALE only. Shape checks are present. **`sponge_T`/`sponge_zw` given without `sponge_rate` are silently ignored.** |
| Passive tracers | `n_tracers` | `SB:1139-1141`. ALE only. |
| Thin layers | `thin_layer_coupling` ('none' or 'mom6'), `thin_layer_cutoff` | `SB:1066-1076`. Defaults: 'mom6' in ALE and 'none' in σ; cutoff 0. |
| Wetting/drying | implicit through `bathy` H ≤ 0 and `d_min` | Handled by WD. ALE restrictions are in `SB:1210-1220`. |
| Diagnostics | `mix_diagnostic` | `SB:1081`. In ALE only: **σ always advects T²** (`SB:2299`). |
| Time stepping | `dt` | Momentum is always AB3, with AB1 then AB2 start-up (`SB:2253-2261`). "ssprk3"/"pc" refer to tracers only. |

**Arguments that are silently ignored in σ mode:** `regrid`, `regrid_params`, `remap_scheme`, `h_thin_bed`, `init_iterations`, `max_truncations`, `thin_layer_cutoff`, `mix_diagnostic`, and `thickness_flux` (validated, then unused). `thin_layer_coupling` can be set to 'mom6' in σ without error but has no effect. `ke_scheme` is ignored with flux-form momentum. A future pre-flight validator should turn these into explicit errors or warnings. That changes no physics.

## 2. State containers

**`SigmaState`** (`SB:106-124`) is a dataclass. Index order is (k, j, i), with k = 0 at the bed.

| Field | Shape | Role |
|---|---|---|
| `eta` | (ny, nx) | prognostic |
| `u` | (N, ny, nx+1) | prognostic |
| `v` | (N, ny+1, nx) | prognostic |
| `T` | (N, ny, nx) | prognostic |
| `h` | (N, ny, nx), ALE only | prognostic |
| `tr` | (ntr, N, ny, nx), ALE only | prognostic |
| `time` | scalar | prognostic |
| `g_hist` | list of up to 3 (gu, gv), newest first | **history** (AB). Built at `SB:2254`. |
| `Qx`, `Qy`, `Fz` | layer transports of the last step | **history**. The next step's flux-form momentum reads them (`SB:1590-1596`), as does the vector-invariant PV flux (`SB:1562-1565`). |
| `omega` | (N+1, ny, nx), σ only | **history**. Read by `_tendencies` only when `adv != 'flux'` (`SB:1607-1613`). |
| `mix_num` | | diagnostic (not read by `step`) |
| `n_trunc`, `n_limited` | ALE | counters, read and accumulated (`SB:2229-2242`) |

**Hidden state outside `SigmaState`.** This is what makes N1 non-trivial.

- **GLS:** `self.core.gls` holds tke, eps and akv. It is advanced inside `SigmaWetDry.step` (`SWD:318-319`, `:335-356`).
- **Lagged-LU CG preconditioner:** `self.fs._lu`, `_lu_matrix` and `_refactor` (`WD:105-107`, `:282-302`), plus the counters `cg_iterations` and `lu_factorizations`.
- **"Last step" geometry:** WD writes `fs.last_wet_u/v`, `fs.last_Du/Dv`, `fs.last_open_Qu/Qv/Du/Dv` and `fs.last_clip_volume` (`WD:309-342`). SB reads them **after** `core.step` for the layer transports (`SB:2167-2168`, `:2277-2292`), for Orlanski (`SB:1750`) and for the open-face transports (`SB:1912-1922`).
- **Exterior profile:** `self.T_ext`, `self.h_ext` and `self.tr_ext` are set by `initial_state` (`SB:1255`, `:1265-1267`, `:1329-1330`).
- **Reference profile:** `self.rho_ref` (mutated from 'auto' to a `ReferenceProfile`).
- **Per-step scratch passed between methods through `self`:**
  - `self._ext` (exterior at tⁿ⁺¹, `SB:1787-1802`)
  - `self._T_in` and `self._tr_in` (inflow values, set at `SB:2186-2188` or `SB:2271`, consumed by `_ext_stack` `SB:1951-1963`)
- **Caches:**
  - `self._zc_cache` (layer centres keyed by the identity of h, `SB:1343-1348`)
  - `self._tt_cache` (face transport thickness keyed by identity, `SB:1419-1427`; `tests/test_sigma_ale.py:832-836` asserts the identity hit)
  - `self._lap` (Laplacian spacings, `SB:1602-1603`)
- **Module-global thread count:** `_THREADS` (`SB:164-173`).

**What `core/restart.py` saves** (`write_restart`, `:96-144`):

- Always: eta, u, v and T (`:36-41`).
- If present: omega, Qx, Qy, Fz, mix_num and h (`:42-49`).
- The AB history (`:106-110`).
- The GLS tke/eps/akv (`:111-116`).
- T_ext when the model is open or ALE (`:117-119`); h_ext in ALE (`:120-122`).
- tr and tr_ext (`:123-128`).
- The lagged-LU matrix (CSC) and the refactor flag (`:129-133`, `:142-143`).
- The ReferenceProfile z and rho (`:134-137`).
- time, n_trunc and n_limited as attributes (`:138-139`).
- A configuration fingerprint (`_config`, `:60-93`) that is compared exactly on read (`:155-165`). It reads the private `model._sp_T` and `model._sp_zw` (`:88-89`).

Not saved, and recomputed on the next step: `_ext`, `_T_in`, the caches, and `fs.last_*`. These are all rewritten before they are read within a step, so they are safe.

## 3. `step()`: ordered sub-steps

### Common prologue (`SB:2246-2263`)

1. `gu, gv = _tendencies(st)` (`SB:1579-1614`). The order of evaluation:
   1. Face depths and wet masks from η (`core.face_depths`, `SWD:263-272`).
   2. PGF: `_baroclinic_pgf` (`SB:1449-1521`), or `_fv_pgf` (`SB:1523-1533`, which calls `core/vertical/pgf.fv_pgf`). This reads `rho_ref` and `layer_heights` (cached).
   3. Coriolis. Vector-invariant (ALE default): `_vector_invariant` (`SB:1551-1569`) reads `st.h` and the **previous `st.Qx/Qy`**, or falls back to L u h̄. Otherwise `_coriolis` (`SB:1535-1549`).
   4. Flux-form momentum advection (`_flux_form_u`, `SB:707-713`; kernel `SB:627`). It reads the previous `st.Qx/Qy/Fz` and V; the vanished-cell limiter is used only in ALE (`SB:1592`).
   5. The advective form (`nonlinear.advection_u/v`, a per-layer loop, `SB:1597-1600`).
   6. Laplacian `ah` (`SB:1601-1606`).
   7. σ vertical advection from the previous `st.omega` (only when adv ≠ 'flux', `SB:1607-1613`).
   8. Masking with the wet faces (`SB:1614`).

   The summation order matters for bit-for-bit results: PGF, then += Coriolis/VI, then += advection, then += Laplacian, then −= vertical advection.
2. AB history and combination (`ab3_combine`, Numba `SB:859-884`). AB1 on the first step, AB2 on the second, AB3 after that.
3. `tu, tv = tau/rho0` (`SB:2262-2263`). Wind is **not** AB-extrapolated.
4. ALE branches to `_step_ale`. σ continues inline.

### σ path (`SB:2266-2314`)

1. If there is an `open_boundary_state`: `_evaluate_exterior(t+dt, …)` writes `self._ext`, and `_T_in = _exterior_T(D0 Δσ)` (`SB:2266-2271`).
2. **Free surface plus vertical viscosity:** `core.step(eta, u, v, G=AB, tau, **_flather_args())` (`SWD:275-320`). Inside it:
   - `column_step` for u and for v: implicit ν, plus log-law drag `first_layer_cd(dz0, z0)` with r = C_d|û₀| (predicted velocity, `SWD:234-237`), plus the wind as a top flux.
   - `fs.step`: the θ-implicit η solve, Flather, then **writes `fs.last_*`**. The direct `spsolve` is used in σ by default (`WD:279-280`).
   - Barotropic correction û + (ū_new − û) (`SWD:316-317`).
   - GLS update (`SWD:318-319`).
3. `_open_velocity` (in place): the sponge on u′, then Orlanski plus nudging (`SB:1764-1785`, `_radiate` `SB:1715-1753`). `_baroclinic_part` uses `np.tensordot` "so σ runs stay bit-identical" (`SB:1710-1712`).
4. **Layer transports:** θ on the depth mean and θ_bc on u′ (`SB:2277-2284`). These read `fs.last_wet_*` and `fs.last_D*`. Open faces are overwritten (`SB:2285-2292`).
5. V0 and V1, then **ω/Fz diagnosed from continuity** by a cumsum from the bed (`SB:2293-2297`).
6. **Tracer advection:** `_advect(stack[T, T²], V0, V1, Qx, Qy, Fz, ok)` (`SB:2299`). SSP-RK3 goes through the `_rk_stage` kernels and `_rhs` → `_advect_div` (WENO5) or `_rhs_numpy` (UP3, upwind1, or 'pc'). Open-edge inflow values come from `_ext_stack` (`SB:1951-1963`).
7. `mix_num` (`SB:2301`).
8. `kh` diffusion along σ (`_diffuse_h_numpy`, `SB:1668-1703`).
9. `kv` implicit (`vertical_implicit`, `SB:2304-2308`).
10. Tracer sponge toward the exterior (`SB:2309-2311`).
11. `omega = Fz/area`, then a new `SigmaState` is returned (`SB:2312-2314`). Velocity is not remapped and there is no regrid.

### ALE path (`_step_ale`, `SB:2120-2243`)

1. Face geometry of hⁿ: `_transport_thickness` (cached), giving the means hu and hv and the face fractions wu and wv (`SB:2136-2138`).
2. `_evaluate_exterior(t+dt, hu, hv, wu, wv)` (`SB:2139`).
3. Viscous thickness: either 'mom6' `viscous_face_thickness` with `hu_drag` and `hu_mean`, or the mean (`SB:2140-2147`).
4. Quadratic drag rates at uⁿ over hbbl (`_bed_drag_rates`, `SB:1852-1876`; used at `SB:2148-2152`).
5. **Free surface plus vertical viscosity:** `core.step(…, wu, wv, **visc, **flather)` (`SB:2153-2154`). This is the same SWD path, plus the δ depth-mean correction (`SWD:253-260`). The solver is CG with the lagged LU (`WD:281-303`) and exact-conservation recompute of η (`WD:325-343`).
6. `_open_velocity` with the face fractions (`SB:2155`).
7. `zero_thin_bed_velocity` (`SB:2156-2157`).
8. Optional `blend_thin_layers` (`SB:2158-2164`).
9. `transport_velocity`: θ/θ_bc split, which reads `fs.last_*` (`SB:2166-2168`).
10. Layer thickness fluxes: `ppm_transport_x` (default) or `upwind_transport_x`. y is handled by swapaxes (`SB:2169-2176`).
11. Open-face transports, in place (`SB:2177-2178`).
12. `limit_outflow`, which also returns `n_lim` (`SB:2180`).
13. Lagrangian thickness h† (`SB:2181-2182`). Then V0, V1, and Fz = 0 (`SB:2183-2185`).
14. `_T_in` and `_tr_in` on hⁿ (`SB:2186-2188`).
15. **Tracer advection:** stack [T, (T²), tr…]. `_advect_fct` (SSP-RK3 WENO5 inside Zalesak; kernels `_fct_stage`, `_fct_fluxes`, `_fct_coeff_kernel`, `_fct_final`, `SB:1993-2026`) or `_advect` (`SB:2189-2194`).
16. mix_num, advection part (`SB:2195`).
17. kh on h† (Numba `_diffuse_ale_kernel`, `SB:1645-1666`). Passive tracers are diffused without the anomaly (`SB:2198-2200`).
18. kv implicit (`SB:2201-2205`).
19. Edge tracer sponge on h† (`SB:2206-2210`).
20. **Regrid plus remap** of [T, (T²), tr…] (`_regrid_remap` → `rg.regrid_columns` and `rm.remap_columns`, `SB:1395-1410`, used at `SB:2211-2214`).
21. mix_num, remap part (`SB:2215-2216`).
22. **Interior sponge** after the remap (`SB:2217-2218`, `SB:1878-1903`).
23. Velocity remap in face-fraction space from h† to hⁿ⁺¹, then thin-bed zeroing (`SB:2219-2223`, `SB:1435-1446`).
24. `truncate_cfl` backstop, counter and RuntimeError (`SB:2224-2240`).
25. New `SigmaState` (`SB:2241-2243`).

### Mode summary

| Scheme | Where | Path |
|---|---|---|
| Momentum: AB3 | always | `SB:2253-2261` |
| Tracers: SSP-RK3 | σ | `_advect` (`SB:2094-2118`) |
| Tracers: SSP-RK3 | ALE | `_advect_fct` or `_advect` |
| Tracers: predictor–corrector ('pc') | σ or ALE | `_advect_numpy` (`SB:2079-2084`), NumPy only, with no FCT |

The free-surface solve sits in exactly one place: `core.step` → `fs.step`. Vertical viscosity and bed drag are applied in the same call, before the barotropic solve.

## 4. External use of private members

**`validation/` and `docs/`:**

- `validation/bayiw_cut/run_pyocean.py:33-34`: `m._edge_face_depths()` and `m._mean_face_thickness(st.h)`. The roadmap cites :31-32; the actual lines are 33-34.
- `validation/bayiw_2d/run_pyocean.py:36`: `m._mean_face_thickness`.
- `validation/internal_wave/run_pyocean.py:28`: `m._mean_face_thickness`.
- `validation/seamount/run_seamount.py:71`: `m._baroclinic_pgf`. Line 69 also sets `m.rho_ref` directly.
- `docs/manual/figures/make_pgf.py:127-129`: `_baroclinic_pgf` and `_mean_face_thickness`.
- `prototypes/bench_jax_vs_numba.py:62-83`: `m._rhs_numpy`, `m._rhs`, `sb._advect_div`.

**`core/`:** `core/restart.py:88-89` reads `model._sp_T` and `model._sp_zw`. It also uses `model.core.gls`, `model.fs.preconditioner_state` and `model.T_ext/h_ext/tr_ext`.

**`tests/`** (the main constraint, because the 1388 tests must pass unchanged):

- `test_sigma_baroclinic.py`:
  - `_baroclinic_pgf` (188-246)
  - `_coriolis` (280)
  - `_diffuse_h` (484)
  - module `sb._weno5_faces` (516-519)
  - `_rhs_numpy` and `_rhs` (542-543)
  - `m._T_in` (572)
- `test_sigma_ale.py`:
  - `_baroclinic_pgf` (112-142)
  - `_tendencies` (452)
  - `_vector_invariant`, `_vi`, `_fq`, `_mean_face_thickness` (471-476)
  - `_radiate` (560)
  - `_transport_thickness` with an identity assertion (832-836)
  - `_diffuse_h` and `_diffuse_h_numpy` (870-871)
  - `_sp_c` and `_sp_u` (1106-1107)
  - `_advect` and `_advect_numpy` (1161-1162)
  - `_advect_fct`, `_advect_fct_numpy`, `_faces_h`, `_upwind_faces_h` (1296-1423)
  - `_bed_drag_rates` (1661)
  - assignments to `m.h_ext`, `m.T_ext` and `m.rho_ref` (731-733, 127-141, 865)
- `test_vector_invariant.py:59`: `m._coriolis`.
- `test_open_boundary_state.py:196-203`: reads and **assigns** `_ext_cols` and `_bc_mask`.
- **Monkeypatches of module-level names:**
  - `test_sigma_ale.py:391` patches `rg.regrid_columns`. This works only while SB calls `rg.regrid_columns` through the module attribute.
  - `test_sigma_ale.py:720` patches `sb.open_face_transport`. This works only while the name is looked up in the `sigma_baroclinic` namespace.

  Moving these calls into a new module silently disables both spies.
- **Module-level imports from SB:** `_flux_form_u(_numpy)`, `_weno5_faces(_numpy)`, `_up3_faces`, `_centred_faces`, `fct_coefficients_numpy`, `_fct_coeff_kernel`, `ab3_combine`, `AB3`, `ReferenceProfile`, `reference_profile`, `set_threads`, `_THREADS` (via `sb._THREADS`, `test_sigma_ale.py:392`), and `SigmaState`.

**Recommendation:** N1 keeps every one of these names in `sigma_baroclinic.py`, as re-exports or thin delegating methods, and adds public replacements (`face_thickness(h)`, `edge_face_depths()`, `pressure_gradient(...)`) for the runners.

## 5. Pure functions vs logic inside the class

**Already pure, array-in/array-out:**

- `core/vertical/regrid.py`: `regrid_columns` (:345), `snap_to_jumps` (:425), `target_densities`.
- `core/vertical/remap.py`: `remap_columns` (:418).
- `core/vertical/ale.py`: face_thickness (:66), viscous_face_thickness (:82), blend_thin_layers (:122), face_transport_thickness (:212), upwind_transport_x (:233), lagrangian_thickness (:278), limit_outflow (:331), ppm_transport_x (:736), transport_velocity (:804), open_face_transport (:834), exterior_on_layers (:857), face_fractions (:888), baroclinic_split (:912), zero_thin_bed_velocity (:950).
- `core/vertical/pgf.py`: `fv_pgf` (:165).
- `core/dynamics/vector_invariant.py`: vorticity_and_pv, coriolis_pv, kinetic_energy, ke_gradient, truncate_cfl, corner_*.
- `core/dynamics/open_boundary.py`: validate_state, flather_targets, exterior_velocity, exterior_tracer, owned_by_state.
- `physics/bottom_friction.py`: `first_layer_cd` (:48), `log_law_cd` (:31).
- `physics/mixing/parabolic.py:27`.
- `SWD`: `vertical_implicit` (:87).
- `nonlinear.advection_u/v`, `laplacian_3d`.
- `diagnostics/mixing.reference_potential_energy`.

**Stateful but encapsulated:** `GLSKEpsilon` (`physics/mixing/gls.py:90`; `step` at :137) and `WetDryFreeSurface` (lagged LU, `last_*`).

**Still inside the class, needing extraction:**

- The Jacobian and two-term PGF (`SB:1449-1521`).
- Energy-conserving Coriolis (`SB:1535-1549`).
- The composition of the momentum tendency (`SB:1579-1614`).
- The WENO, UP3 and FCT kernels are module-level functions, but their drivers (`_rhs`, `_advect`, `_advect_fct`, `_faces_h`, `_upwind_faces_h`, `_open_faces`, `_ext_stack`) depend on `self.open`, `self.tracer_scheme`, `self._T_in` and `self._tr_in`.
- Horizontal diffusion (`SB:1645-1703`).
- Orlanski, nudging and sponge (`SB:1715-1785`). Sponge rates are already a module function, `_sponge_rates` (`SB:907`).
- Exterior evaluation (`SB:1787-1850`).
- Quadratic bbl drag (`SB:1852-1876`).
- Interior sponge (`SB:1878-1903`).
- Open-face transports (`SB:1905-1922`).
- The σ transport and ω block (inline `SB:2276-2297`).
- The whole ALE step choreography (`SB:2120-2243`).
- The linear EOS (`SB:1351`) and layer geometry (`SB:1333-1360`).
- In SWD: `column_step` (`SWD:196-261`) mixes closure, drag, wind and the δ correction.

## 6. Physics in the repo that is not connected to the 3-D core

- **TEOS-10 and the S-aware linear EOS** (`physics/eos/__init__.py`: `density_teos10` :63, `density_linear(T, S, …)` :123, `compute_density` :203). These take a Phase-1 `ModelState` with z-levels (`core/state.py:60`).

  To plug in:
  - Add `S` to `SigmaState`, the tracer stack, remap, the restart (`_FIELDS`) and the exterior/OBData path.
  - Replace every `self.density(T)` call: `SB:1254`, `:1305`, `:1317`, `:1372`, `:1407` (regrid), `:1475` (Jacobian PGF).
  - Make the hybrid regrid density potential (σ₂-like) rather than in situ.

  The hard part is `fv_pgf`. It is **hard-wired to the linear EOS**: it receives `rho0, t_coef, t_ref` and integrates the density analytically with Boole's rule (`core/vertical/pgf.py:95`, `:165`). It needs an EOS callable plus the derivative-free integral, as MOM6 does with `int_density_dz`. `ReferenceProfile` and the "anomaly" diffusion (`SB:1654-1655`) also invert the linear EOS to get T_ref. With S, the anomaly diffusion is ill-defined and must become ρ-based or be switched off.
- **Bulk fluxes:** `forcing/bulk.py` (`compute_surface_fluxes` :734, `bulk_ncar`, `bulk_kara`) and `forcing/surface.py:188` (`surface_tendency`, `ModelState`/z-level, k = 0 at the surface).

  Today `step` takes only τ (`SB:2246`). Heat and freshwater need:
  - a surface flux applied to the top layer, ideally as the top boundary flux of the existing implicit `vertical_implicit` call for T (`SB:2201-2205`), which currently passes `zero` as the bed term and no top flux;
  - penetrating shortwave over h (`forcing/surface.py:129` `_sw_profile`);
  - E − P as a volume flux in the free surface. `WD.step` has no source term; this is a real physics change and does not belong in N1.
- **KPP:** `physics/mixing/kpp.py:555` (`kpp_tendency`). It is written for the z-level `ModelState` and is an explicit tendency for AB2.

  For the 3-D core it must be re-expressed column-wise on (h, T, S, u, v), with k = 0 at the bed (reverse order), returning ν/κ at interfaces and the non-local term. Those feed `column_step(nu=…)` (already an argument, `SWD:196-197`) and the tracer `vertical_implicit` (which today takes the constant `kv` only). It needs N² from the EOS and a surface buoyancy flux from bulk.
- **GLS is homogeneous** (`physics/mixing/gls.py:12`, B = 0). It is coupled only to momentum (`SWD:302-303`, `:335-356`) and gives tracers no κ. Making it stratified requires N² and passing akt to the tracer solve. That is a physics change, also outside N1.
- **Salinity:** absent from SB (module docstring `SB:56-57`). Passive tracers (`n_tracers`) already exercise the whole multi-tracer advection, remap, sponge and OBC path. S could initially ride as an "active" tracer slot with the same machinery.
- **Output** (`diagnostics/output.py:36-45`): CF σ only. It has no `h` field and no ALE vertical coordinate, and runners use their own `.npz` files.

## 7. Proposed seams and bit-for-bit risks

**Seams that follow the existing call graph.** The driver keeps the current order.

1. **EOS:** `density(T[, S])` plus `T_from_rho` for the reference and the anomaly. For N1 it is only a wrapper around `SB:1351`. `fv_pgf` keeps its linear arguments.
2. **VerticalGeometry / Coordinate:**
   - σ: D·Δσ.
   - ALE: `initial_h`, `regrid` and `regrid_remap`, `remap_faces`, `transport_thickness` (keep the identity cache), `mean_face_thickness`, `layer_heights` (keep its cache), `edge_face_depths`.
   - Subclasses or strategies: `SigmaEulerian` and `ALE(rule)`. The ALE-only arguments move here, which removes the silent ignores.
3. **PGF:** jacobian, twoterm or fv, plus the reference profile. Contract: `(eta, T, h|None) → (gx, gy)`, as `_baroclinic_pgf` today.
4. **Momentum:**
   - `FluxForm(adv)`: Coriolis plus flux, advective, or vertical advection. It reads the previous Qx/Qy/Fz/ω.
   - `VectorInvariant(ke_scheme)`: reads the previous Qx/Qy and the geometry `_vi`.
   - A `HorizontalViscosity(ah)` component.
   - The `_tendencies` sum order must be kept exactly.
5. **TimeStepper (AB3):** owns `g_hist` and `ab3_combine`.
6. **FreeSurface plus VerticalViscosity:** this is currently one call (`SWD.step`). In N1, wrap it as is. Splitting column_step (closure, drag, wind, δ) from `fs.step` is possible, but `fs.last_*` must then become explicit return values (a "BarotropicFluxes" record), not attributes.
7. **Closure:** constant, parabolic or GLS. It returns ν at interfaces. GLS state goes into the component state for restart.
8. **BottomDrag:** log-law (σ, inside column_step) or quadratic hbbl (ALE, returns `r_u`, `r_v`).
9. **LayerTransport:** the σ θ/θ_bc expression and Fz; in ALE, transport_velocity, ppm/upwind, open faces, `limit_outflow`, and `lagrangian_thickness`.
10. **TracerAdvection:** WENO5/UP3 × SSP-RK3/pc × FCT. It takes the inflow stack `ext` **as an argument** instead of reading `self._T_in`.
11. **TracerDiffusion:** kh, plus kv. It is the natural hook for κ from KPP or GLS and for surface heat flux.
12. **OpenBoundary:**
    - `evaluate(t)` produces the `_ext`-like record.
    - `flather_args`.
    - `relax_velocity` (sponge plus Orlanski).
    - `inflow_tracers(h)`.
    - `tracer_sponge(h)`.
    - The masks `_bc_mask`, `_ext_cols`, `_bc_u`, `_bc_v`.
13. **InteriorSponge:** after the remap.
14. **SurfaceFluxes:** τ only in N1, passed as `tu`/`tv`.
15. **Diagnostics:** mix_num (advection plus remap), n_trunc and truncate_cfl, n_limited, `fs.last_clip_volume`, `cg_iterations`.

**Risks for bit-for-bit equivalence:**

- **Order of floating-point operations.**
  - `_tendencies` accumulates in place in a fixed order (PGF, then +C, then +adv, then +lap, then −vadv).
  - The σ depth mean uses `np.tensordot` (BLAS) in `_baroclinic_part` (`SB:1711`), but `(ds*u).sum(0)` in the transports (`SB:2281-2282`) and in `_radiate`'s re-centring (`SB:1747`). Unifying them would change bits.
  - ALE uses the Numba `baroclinic_split`.
  - The θ/θ_bc expression in `SB:2281` must stay character for character. So must the `_rk_stage` and `_ab3_kernel` expressions, which are documented as matching NumPy's evaluation order (`SB:837-884`).
  - `face_fractions` is computed once and reused (`SB:2221`).
- **In-place mutation.** These are hidden data flow that a functional component contract would need to copy, and copies are fine for bits but cost memory and time.
  - `_open_velocity`/`_radiate` write into `u_new` (`SB:1776-1779`, `:1753`).
  - `_open_face_transports` writes into `Qx` (`SB:1922`).
  - `snap_to_jumps` writes into `zw` (`SB:1321`).
  - `limit_outflow` returns new arrays.
- **Identity caches.** `_tt_cache` and `_zc_cache` key on `is`. If the driver copies `h` (for example `dataclasses.replace` or `np.asarray(..., copy)`), the results stay identical but performance drops. If anything mutates `h` in place, the results become **wrong**: the docstring (`SB:1416-1418`) relies on "never written in place".
- **Dual-path dispatch.** Several functions have NumPy and Numba twins that are *claimed* to agree bit for bit, but the dispatch differs by mode, and tests compare the twins:
  - `_diffuse_h`: Numba only when `h` is given (ALE); σ uses NumPy (`SB:1651-1652`).
  - `ReferenceProfile`: `np.interp` below 1000 points, a Numba kernel with `fastmath={"contract"}` above (`SB:801`, `:893-899`).
  - `_rhs`: Numba only for WENO5 and not centred.

  Keep the dispatch conditions exactly.
- **Numba flags and threads.**
  - `cache=True` kernels can move modules safely, but `fastmath` and `parallel` flags must not change.
  - `set_threads` claims thread-count independence (`SB:169-171`); the kernels use row-parallel `prange` without float reductions. Any new reduction (for example a parallel sum in a diagnostic) would break this.
  - `numba.set_num_threads(_THREADS[0])` is called before each kernel. A component in another module must read the same `_THREADS` list, not a copy.
- **Hidden state across steps.** These must be owned by components and checkpointed exactly as `core/restart.py` does today, with `_config` attribute names kept or migrated:
  - GLS
  - lagged LU (`_refactor` decides refactorisation from the previous step's CG iteration count, `WD:301-302`)
  - `st.Qx/Qy/Fz/omega` as history
  - `T_ext`/`h_ext`/`tr_ext`
  - `rho_ref` set by `initial_state`

  Old restart files must still read.
- **Ordering constraints that are implicit today.**
  - `core.step` must precede anything reading `fs.last_*`.
  - `_evaluate_exterior` must precede `core.step`, because Flather targets come from `_ext`.
  - `_T_in`/`_tr_in` must be set before advection and are consumed through `self`.
  - The ALE tracer sponge acts on h† before the regrid, but the interior sponge acts after the remap. This choice keeps it out of `mix_num`.
  - The σ path evaluates the exterior only when `obc_state` is set; ALE always calls `_evaluate_exterior`.
- **Test coupling.** About 40 private-member uses in tests plus two module-namespace monkeypatches (section 4). N1 should keep delegating shims in `sigma_baroclinic.py` and route the patched calls through the same module attributes, or update the tests in the same commit, accepting that "tests unchanged" is no longer literally true.
