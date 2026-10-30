# pyOCEAN: current horizontal momentum viscosity (map for the Smagorinsky design)

Read-only survey, 2026-10-30. Paths relative to the repo root. No code was modified.

## 1. The existing constant Laplacian viscosity `ah`

### Where it is declared and routed
- Constructor argument `ah: float = 0.0`, "horizontal Laplacian viscosity [m² s⁻¹] (free slip)":
  `core/dynamics/sigma_baroclinic.py:952` (docstring), `:1070` (signature), stored `:1198`, passed to the `Momentum`
  component `:1220` (`adv=adv, ah=ah, ke_scheme=..., ale=..., dt=...`).
- Post-construction assignment `m.ah = x` is forwarded to the component by the `_PARAMS` table:
  `"ah": ((("_momentum", "ah"),), None, None)` at `:1853`; tested in `tests/test_components_momentum.py:44-46`.
- Restart: `ah` is part of the config that must match between writer and reader, `core/restart.py:76`.
  There is no Smagorinsky-like state; `ah` is a scalar, nothing is stored per cell.
- Pre-flight (`core/model/preflight.py`) has no check of `ah` (grep for `ah`/`viscos` finds nothing). No CFL/diffusive
  limit warning exists for it.

### Operator: scalar ∇² on each velocity component, NOT a stress divergence
- Term name `"hviscosity"`, 4th in the fixed term order (pgf, coriolis|vector_invariant, advection, hviscosity,
  vadvection): `core/components/momentum.py:4-5` (docstring "Laplacian viscosity A_h ∇²u along layers"),
  appended at `:204-215` as `("hviscosity", ah*∇²u, ah*∇²v, "+")`.
- `∇²` = `laplacian_3d` (`core/dynamics/nonlinear.py:224`) with Numba kernel `_laplacian_3d_kernel`
  (`nonlinear.py:~191-221`, `@njit(parallel=True)`, prange over k*j); NumPy oracle `_laplacian`/`laplacian_u`/`laplacian_v`
  (`nonlinear.py:~177-190, 245+`). Test `tests/test_nonlinear.py:238-249` checks kernel == per-layer NumPy, bit for bit;
  `test_nonlinear.py:122-134` check ∇² of a quadratic (`2/DX²`) and of a constant (0).
- Stencil: non-uniform 3-point second difference per axis, `2((f₊−f)/h₊ − (f−f₋)/h₋)/(h₋+h₊)`, spacings from
  `_spacings_u/_spacings_v` (`nonlinear.py:99-125`, built once and cached in `Momentum._lap`, `momentum.py:204-207`).
  It acts on u at u-faces and v at v-faces independently: no deformation tensor, no cross terms ∂v/∂x in the u-equation,
  no metric terms on a spherical grid, no cell-centre/corner (D_T, D_S) tension and shear. This is "viscosity as
  A_h ∇²u", which is not rotation-invariant on curvilinear grids and does not conserve angular momentum.
  A Smagorinsky ν(x,y,k) would need the tension/shear rates at cell centres and corners, i.e. new stencils not present.
- **No thickness weighting**: acts on the layer velocity `st.u`, `st.v` only (`momentum.py:213-215`), not on h·u. In ALE
  with vanished/very thin layers the operator is applied to the velocity of those layers as is (noise there is untouched
  by thickness; MOM6 weights by h, `MOM_hor_visc`). Horizontal exchange is along layers (k fixed), as the docstring says.
- Free slip: neighbours that are dry are replaced by the centre value (zero gradient), kernel
  `nonlinear.py` `lo = f[k,j,i-1] if i>0 and wet[j,i-1] else fc` (same for hi, j±1). Domain edges (i=0, i=nx) also fall
  back to the centre value. The result is zero on dry faces (`out[...] = 0` where `~wet`) and `sum_terms` zeroes dry faces
  again (`momentum.py:39-52`). No no-slip option, no viscous boundary layer.
- Wet mask: `wet_u = isfinite(H_u) & isfinite(Du) & (Du > d_min)`, `core/dynamics/sigma_baroclinic.py:1618-1620`
  (`Du` the total face depth from the free surface), so W/D toggles the mask each step with η.

### Time level and time stepping
- Evaluated in the driver's phase 2 (`explicit_momentum`, `core/model/driver.py:~228-234`) on `ctx.st`, the state at tⁿ
  (`_momentum_terms(ctx.st)` → `sum_terms`). It sits inside the explicit tendency G and is therefore extrapolated by AB3
  with all other explicit terms (`core/components/time_stepper.py:50-66`, phase 3 `_time_combine`, driver `:239`).
  There is no RK3 stage for momentum (RK3/FCT is for tracers only). So explicit viscosity runs under AB3: its
  diffusive stability limit is that of AB3, which is milder than AB2 but not implicit; nothing in the code checks
  `ah*dt/dx² `.
- The implicit part of the step is the vertical viscosity and bed drag (`FreeSurfaceVerticalViscosity`), not `ah`.
- ALE: same term, same phase, same code for `momentum='flux'` and `'vector_invariant'`; the only ALE-specific difference is
  that `Momentum.terms` is called with `form == 'vector_invariant'`, in which case the "vector_invariant" term replaces
  "coriolis" and "advection" is skipped (`momentum.py:166-176`). The hviscosity block (`:204`) is shared.
  `core/dynamics/vector_invariant.py:230-300` is NOT viscosity: lines 241-247/289-295 are the thickness-weighted PV
  averaging (`ah` is a local variable there for "area·h", a name clash to avoid when searching).
- Open boundaries: the Laplacian knows nothing about them; edge faces use the zero-gradient fallback. Open-edge velocity
  is later set by the boundary phases (Flather, Orlanski, sponge: driver phase 5). A Smagorinsky strain rate at the first
  interior cell next to an open edge will use these faces, so exterior/edge treatment must be decided.
- Periodic boundaries: pad-crop with halo 1 of u, v, wet, and of the horizontal grid (`momentum.py:206-212`,
  `core/model/periodic.py`; `MOMENTUM_HALO = 4`, `periodic.py:32`, is for flux-form advection). A strain-rate stencil of
  radius 1 on corners needs halo ≥ 1 (probably 2 if ν is itself smoothed or if a biharmonic is built as ∇²(ν∇²u)).
- σ vs ALE: identical code. In σ-Eulerian mode layers follow the terrain, so "along layers" is along σ surfaces (known
  to be non-isopycnal and not horizontal over steep slopes); in ALE z*/hybrid it is along the layer interfaces.

### Tests that exercise `ah`
- `tests/test_sigma_baroclinic.py:79` (`ah=1.0`, mixing diagnostic); `tests/test_n1_golden.py` via `tests/n1_cases.py:59,76`
  (`ah=5.0`, 31 golden configurations, bit for bit: any change to the existing `ah` path, including operand order,
  changes hashes); `tests/test_periodic.py:123,125` (`ah=5.0` with periodic axes, translation invariance);
  `tests/test_restart.py:27,179` (`ah=1.0`); `tests/test_sigma_ale.py:176,319,918,1090,1254` (ALE boxes, hybrid, lock
  exchange KH = 0.01 style); `tests/test_n2a_density.py:167`, `test_n2a_tef.py:51`, `test_n2a_freshwater.py:30`,
  `test_surface_heat.py:123`; `tests/test_components_momentum.py:44`.
- `ah` is a scalar only; there is no spatially variable viscosity anywhere (no 2-D array form, no per-layer form).

## 2. Architectural plug-in points for a Smagorinsky viscosity

- **Component contract** (`core/model/contract.py:83-93`): `name`, `role` (`Role` enum `:27-42`), `reads`, `writes`
  (registry names), `check(ctx) -> list[Issue]`, `state_fields()`. `param(...)` at `:55` documents units/cite/benchmark.
  There is no `Role` for horizontal viscosity. Options: (i) extend `Momentum` with `viscosity` strategy objects (the
  `hviscosity` term already is a named term, so a `SmagorinskyViscosity` could return the term pair); (ii) new
  `Role.HORIZONTAL_VISCOSITY` between momentum terms. Since the term list order fixes floating-point results (golden
  tests), any new option must leave the `ah` constant path bit for bit.
- **Driver** (`core/model/driver.py:1-40`): ten fixed phases; viscosity belongs in phase 2 `explicit_momentum`
  (`_explicit_momentum`, `driver.py:~228`, uses `keep_terms` to expose named terms, `last_terms`). It is explicit (AB3).
  An implicit/semi-implicit horizontal solve would need a new sub-phase between 3 and 4. `debug=True` hashes per phase
  (golden machinery) which will cover a new term for free.
- **Pre-flight** (`core/model/preflight.py`: `preflight(a, given)`, helpers `err`/`warn` at `:46-50`; "ignored argument"
  warnings e.g. `:134`, `:161`): add `smag_coef`/`visc_scheme` validation, warn when both `ah>0` and Smagorinsky are
  given (policy: max, as MOM6/OM4 Adcroft 2019 §2.2.5), CFL-viscous check `ν_max dt/Δ² ≤ bound` (none exists today).
  Arguments are plumbed through `SigmaBaroclinic.__init__` (`sigma_baroclinic.py:1070`) and the `given` set.
- **Registry** (`core/model/registry.py:31-56`, `VarSpec(name, dims, location, units, standard_name, kind)`): diagnostic
  fields (e.g. `ah_smag` at t or q points, "diagnostic" kind) are declared here; restart/output read it. Nothing is
  needed for state (ν is a pure function of u, v, grid), so restart bit-for-bit needs only the config entry
  (`core/restart.py:76` pattern) and nothing in `SigmaState`.
- **Numba kernel pattern**: Numba `@njit(parallel=True, cache=True)` kernel with `prange` over `n*ny` rows (see
  `_laplacian_3d_kernel`, `_vort_pv_kernel` at `vector_invariant.py:~225`), NumPy reference kept as oracle
  (`vorticity_q_numpy`, `pv_q_numpy`, `_laplacian`), tests assert `np.array_equal` kernel vs oracle
  (`tests/test_vector_invariant.py:191-199`, `tests/test_nonlinear.py:238-249`). Wrapper does
  `numba.set_num_threads(threads._THREADS[0])` before the call (`momentum.py:208`, `tracers.py:139`).
- **Threads** (`core/model/threads.py`): one shared list `_THREADS = [1]`; `set_threads(n)` clamps to
  `numba.config.NUMBA_NUM_THREADS`; the model re-exports it (`sigma_baroclinic.py:120`). Rows are independent so results
  are bit-identical for any thread count: a new kernel must keep per-element arithmetic independent of the prange split.
  Runs must export OPENBLAS/OMP threads = 1 (CLAUDE.md).
- **Grid data**: `_spacings_u/_v` give only 1-D neighbour spacings; a strain-rate stencil needs `dx_t, dy_t, dx_u, dy_u,
  dx_v, dy_v, area_t, area_q` (all used in `vector_invariant.py` via `g["dx_u"]`, `g["dy_v"]`, `g["area_q"]`,
  `g["mask_q"]`, `Momentum._padded_vi_geometry`, `momentum.py:112`), so corner vorticity/strain stencils already exist for
  PV and can be reused (`vorticity_q`, free-slip corners = 0, `vector_invariant.py:~260`).
- **Kinetic energy/Archimedes**: because the existing operator is not a stress-divergence, a Smagorinsky ν inserted in
  `ν ∇²u` form would not be dissipative-consistent; a proper form (MOM6 `horizontal_viscosity`, CROCO/ROMS `visc3d`) takes
  the tension at T points and shear at q points. Design decision pending.

## 3. Benchmark evidence

### DOME overflow (`validation/dome/README.md`)
- Reference viscosity: Legg 2006 case: ν_h Laplacian 50 m²/s at 10 km, 25 at 5 km (∝ Δx), ν_v = 0.5, no tracer diffusion, no
  Ri mixing (`dome/README.md:25-27`). MOM6 reference: `LAPLACIAN`, `KH`, `KV = 0.5`, "sin Jackson ni Smagorinsky" (`:35`);
  the MOM6 DOME example itself uses biharmonic Smagorinsky (`SMAG_BI_CONST = 0.06`, `:182`) and Jackson (`:181`).
- pyOCEAN used constant ν_h through `ah` (hybrid ν_v = 0.5 baseline, row `:210`); with hybrid + GLS (N3) ν_v falls to a
  floor outside the plume: max|u| 5.7 vs 3.2 m/s, 1072 limiter events, **one CFL truncation at day 31.8** (1.3 cm face
  layer, v = 8.96 m/s), `:221-225`; the ν_v = 1e-5 constant control aborted after 38 s wall with 1034 CFL truncations
  (> `max_truncations` = 1000), `:226`.
- `kv_low` (ν_v 1e-4) dropped for all models: MOM6 stops at day 3 (truncations), MITgcm at day ~4 (cg2d); "without
  biharmonic Smagorinsky viscosity or Jackson mixing the case is ill-posed (c/b)", `:148`. Open item 4: "(b) Smagorinsky
  biharmónica, para que ν_v bajo sea un caso bien planteado", `:182`; item 2 of the N3 findings, `:233`.
- Roadmap entries pointing here: `docs/design/vision_roadmap.md:166, 176, 216` (open: "viscosidad horizontal de
  Smagorinsky para la híbrida").

### Lorenz et al. (2021) estuary (`validation/lorenz_estuary/README.md`)
- Reference GETM: no horizontal viscosity given in the paper ("no dado en el artículo"); pyOCEAN uses `AH = 0.0`
  (`validation/lorenz_estuary/case.py:92, 214`).
- σ + stratified GLS leaves ν ≈ 1e-5 below the halocline; at ebb a 1-1.5 m/s surface jet near x ≈ 68 km gives
  |u|Δt/Δx = 0.75-0.9, beyond the AB3 + WENO5 flux-form limit at A_h = 0: a 3-4 Δx mode grows ×1.3 per step, then the free
  surface matrix is singular (`README:90`). Remedy was Δt = T_M2/746 (CFL ≤ 0.36-0.46) (`:99-101`), a numerical choice (c).
  Final list item 3 (`:367`): needs "(i) a small biharmonic or Smagorinsky viscosity option (MOM6, ROMS), (ii) a
  `truncate_cfl` backstop or at least an advective-CFL warning in σ, (iii) post-mortem catching the grid mode before the
  solver failure". ALE-σ constant closure T_M2/498 (`:47`). Also `:349`: with A_h = 0 the velocity at the front has grid-scale
  oscillations (42.5 → 36 → 26 → 22 cm/s).

### Ilicak et al. (2012) four-model benchmark (`validation/ilicak_benchmark/`)
- Case ν_h list (README `:10-15`): lock exchange 0.1, 10, 100; overflow 1, 100, 1000; internal wave 0.01, 15, 150;
  seamount 0 ("sin viscosidad", `:143`). Common physics: ν_v = 1e-4, free slip, no explicit tracer diffusion (`:17-21`).
  Re_Δ = UΔx/ν_h at `:62-66` (lock 2480/25/2.5; overflow 1570/16/1.6; IW 1.9e5/120/12).
- pyOCEAN passes it **via `ah=nu`** of `SigmaBaroclinic` (`validation/ilicak_benchmark/run_pyocean.py:49`), constant,
  Laplacian, same code path as §1. MOM6/CROCO/MITgcm use their constant Laplacian too. The reference paper's Smagorinsky
  runs (Ilicak 2012, Figs. 19) were not reproduced; no adaptive viscosity case exists in the suite.
- Seamount at ν = 0: pyOCEAN σ 0.17-0.21 cm/s, z* 0.02-0.08; MOM6 z* truncates from day 1 (`:143-152`).
- Related ALE findings that interact with viscosity: hybrid IW damped at ν = 0.01 (open), overflow hybrid fails at ν = 1 in
  4 of 5 trajectories (vanished-layer slab, open), thin overflow faces up to 1.9-2.0× the thick ones.
- `docs/design/vision_roadmap.md` mentions benchmark smoke runs; BayIW cut/2-D/internal_wave READMEs use small constant
  viscosity (e.g. `tests/test_sigma_ale.py:319`: "MOM6 KV, KH").

## 4. Local literature (library: /Users/estebancruz/Documents/TESIS_InternalWaves-2/Literatura)

`python scripts/litsearch.py "Smagorinsky" "Leith" "biharmonic"` and `"grid Reynolds" "lateral viscosity"` run from the
repo root (9 scanned PDFs cannot be searched; none of them concern viscosity). Hits:

- **Ilicak_2012 - Spurious Dianeutral Mixing and the Role of Momentum Closure.md** (the key one). Spurious dianeutral
  transport ∝ lateral grid Reynolds number Re_Δ = UΔ/ν (abstract, line 45). Smagorinsky viscosity defined as
  ν_SM = (C_s Δ)²|S| (eq. 11, line ~1819) with scaling |S| ~ U/Δ → ν ~ C_s² Δ U, so Re_Δ ≈ 1/C_s² (eqs. 12-14). Recommended
  C_s = 1/√2 (γ ≈ 2.22), Re_Δ = 2 (eq. 15, "aligned with Griffies and Hallberg 2000"), vs LES 0.17 giving Re_Δ ≈ 35, "excessive
  spurious mixing" (line ~1878). Tests: dam break, overflow, baroclinic eddies with MITgcm + background ν_h = 1e2
  plus Smagorinsky C_s = 1/√2 reduces noise and spurious mixing (lines 2032-2053). Critical Re_Δ < 2 for centred schemes
  (also quoted in Gibson_2017 l.311, Petersen_2015 l.1061).
- **Ilicak_2016 - Quantifying Spatial Distribution of Spurious Mixing.md**: Smagorinsky has less numerical mixing than Leith
  with the same non-dimensional constant (γ = 2.2; Re_Δ ≈ 2 for Smagorinsky and ≈ 3 for Leith); also proposes modified Leith
  (Exp5) (lines 35, 489-514, 570).
- **Adcroft_2019 - OM4.0.md** (lines 674-676): OM4p25 uses biharmonic viscosity = max(dynamic biharmonic Smagorinsky with
  coefficient 0.06 (Griffies & Hallberg 2000), static u₄Δ³ with u₄ = 0.01 m/s); OM4p5 adds Laplacian = max(Smagorinsky 0.15,
  static u₂Δ with u₂ = 0.01 m/s), scaled to zero where the first baroclinic deformation radius is resolved (Hallberg 2013).
- **Holmes_2020 - The Geography of Numerical Mixing.md** (lines 70-72, 735-790): keeping viscosity fixed while refining the
  grid reduces numerical mixing by ~35 %; ACCESS-OM2 uses biharmonic Smagorinsky B = C²Δ⁴|D|/8, C = 2 (eqs. 12-13 of
  Griffies & Hallberg 2000, their eq. 18); recommends not reducing viscosity with resolution.
- **Megann_2022 - Assessment of the z-tilde Coordinate...md** (lines 262-268): NEMO default bilaplacian −1.5e11 m⁴/s at the
  equator, ∝ (max grid dimension)³; Megann & New 2001 showed this is lower than optimal (cell Re above critical); raising
  viscosity reduced drift up to 50 % (l.481).
- **Petersen_2015 - ALE in MPAS-Ocean.md**: ReΔ < 2 stability condition for unphysical extrema (Griffies 2004); biharmonic
  operator with ν₄ for real-world runs (lines 866-867, 1061); ReΔ = 250 vs 2.5 comparisons.
- **Gibson_2017** (Re_Δ < 2 for centred advection), **Klingbeil_2018** (grid Reynolds constraint, coastal models),
  **Banerjee_2024** (biharmonic operators sign-indefinite, tracer), **Garinet_2024** (biharmonic filtered), **Chang_2026**
  (regional MOM6: biharmonic Smagorinsky coefficient 0.015 with a 0.01 m/s velocity scale),
  **Hafner_2021** (Veros: biharmonic 1e11 m⁴/s² instead of default Laplacian), **Herzfeld_2011** (regional model with
  Smagorinsky horizontal viscosity), **Bleck_2002** (HYCOM biharmonic smoothing, deformation-dependent), **Meunier_2025**
  (lateral viscosity A_h as a calibration parameter), **Eden_Iske_2019**, **Cushman-Roisin & Beckers (2011)** eq. 4.10
  (Smagorinsky ν = (c Δx Δy) sqrt(deformation)), **Mason_2010** (viscosity decreased away from boundaries in nests).
- What the library says about the choice: Laplacian ν = (C_sΔ)²|S| with C_s ≈ 0.7 gives Re_Δ ≈ 2 (Ilicak 2012, 2016);
  biharmonic: Griffies & Hallberg 2000 recommend coefficient ~0.06 (OM4) up to C = 2 (ACCESS), scaled Δ³-Δ⁴ so scale
  selectivity is higher, with a static floor u₄Δ³ (Adcroft 2019). Biharmonic is the production choice in global models; a
  Laplacian Smagorinsky with large C_s is more dissipative but more robust in near-grid-scale regional cases such as DOME's.
  For small domains / eddy-permitting the choice is not settled in the library; DOME/Lorenz evidence in §3 favours at least a
  small background plus a Smagorinsky safeguard.

### Missing locally (no file found in the library; check `docs/literature/reading_list.md`)
- Smagorinsky (1963), Smagorinsky (1993): not in library (only cited in Ilicak/Cushman-Roisin).
- **Griffies & Hallberg (2000), Biharmonic friction with a Smagorinsky-like viscosity** (MWR 128, 2935): not in library, not
  in `reading_list.md` (grep for Smagorinsky/Griffies finds only Griffies et al. 2000 *spurious mixing*, line 39, and
  Griffies 2020, line 61). Add it.
- Leith (1996), Fox-Kemper & Menemenlis (2008, Can mesoscale eddies be parameterized...?), Jackson et al. (2008, listed in
  dome README only as [verificar]), Hallberg (2013, resolution function), Megann & New (2001), Griffies (2004) book,
  Bryan/Redi-style viscosity reviews: absent. Held in `reading_list.md`: Ilicak 2012 (★ line 23), Ilicak 2016 (31), Ilicak 2014
  (32), Holmes 2020 (40), Megann 2022 (55), Megann 2024 (56). Adcroft 2019 is in the library (not matched by the grep of the list).
- Reference model sources not yet read for this topic: MOM6 `~/pyOCEAN_refs/MOM6/src/parameterizations/lateral/MOM_hor_visc.F90`
  (Smagorinsky Laplacian/biharmonic, `BOUND_KH`, `BETTER_BOUND_KH`, `SMAG_BI_CONST`), MITgcm `model/src/calc_visc.F`/
  `mom_calc_visc.F` (viscC2Smag, viscA4Grid), CROCO `visc3d.F`/`uv3dmix.F` (to be read by a subagent).

## Summary (5 lines)
1. `ah` is a constant scalar, plain Laplacian `A_h ∇²u` on each velocity component along layers (free slip, no thickness weighting, no tensor form), explicit in the AB3 tendency, same code for σ and ALE.
2. No variable-ν path, no preflight check, no viscous CFL limit; `ah` is hashed in 31 golden configs, so the Smagorinsky option must leave `ah` bit for bit.
3. Natural slot: a new named term / strategy in `Momentum.terms` (driver phase 2), Numba prange kernel + NumPy oracle, strain stencils reusing the PV corner geometry; needs halos for periodic and an open-edge decision.
4. Evidence: DOME hybrid+GLS truncation and the aborted ν_v = 1e-5 control, kv_low dropped (MOM6/MITgcm also need Smagorinsky/Jackson), Lorenz σ+GLS forced Δt T_M2/746 with A_h = 0; Ilicak runs use constant `ah=nu`.
5. Library supports ν=(C_sΔ)²|S| with C_s≈1/√2 (Re_Δ≈2) or biharmonic Smagorinsky 0.06 (OM4); Griffies & Hallberg 2000, Smagorinsky 1963, Leith 1996, Fox-Kemper & Menemenlis 2008 are missing locally and absent from the reading list.
