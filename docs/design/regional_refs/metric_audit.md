# Metric audit of `SigmaBaroclinic` for orthogonal curvilinear grids

Regional chain, plan `docs/superpowers/plans/2026-11-01-regional-chain.md` Task 3 (2026-11-01, branch
`regional-chain`). Question: does every horizontal operator of the 3-D core use the metric that MOM6's operator
uses, so that the model is correct on any orthogonal grid built by `core.grid.supergrid.grid_from_supergrid`
(the NWA12 cut)? MOM6 source `~/pyOCEAN_refs/MOM6/src`, git `0e9a561`; `file:line` below is relative to it.
Metric names and the MOM6 mapping are in `mom6_grid.md` §3 and §8:

| pyOCEAN | MOM6 | meaning |
|---|---|---|
| `dx_u`, `dy_u` | `dxCu`, `dyCu` | T→T distance across the u-face; u-face length |
| `dx_v`, `dy_v` | `dxCv`, `dyCv` | v-face length; T→T distance across the v-face |
| `dx_t`, `dy_t`, `area_t` | `dxT`, `dyT`, `areaT` | T-cell widths and area |
| `dx_q`, `dy_q`, `area_q` | `dxBu`, `dyBu`, `areaBu` | corner spacings and area (supergrid grids only) |
| `area_u`, `area_v` | `areaCu = dxCu·dy_Cu`, `areaCv = dyCv·dx_Cv` | products; MOM6 masks them by the open face |

Inside the model the free-surface core (`core/dynamics/wetting_drying.py:134-136`) renames them once:
`fs.Lu = dy_u`, `fs.du = dx_u`, `fs.Lv = dx_v`, `fs.dv = dy_v`, `fs.area = area_t`; most components read these.

Verdicts: ✅ = same metric as MOM6; **fixed** = changed by this task; (c) = a deliberate numerical choice that
differs from MOM6 but is consistent on an orthogonal grid; (b) = missing physics on a curved grid, not a metric
swap (not changed here; recommendation given).

## 1. Audit table

| pyOCEAN `file:line` | operator | metric used | MOM6 operator and metric | verdict |
|---|---|---|---|---|
| `core/dynamics/wetting_drying.py:289, 343, 459` | barotropic continuity (θ-transport, η update) | `Lu·D·u`, `Lv·D·v` (face lengths), divided by `area` | `MOM_continuity_PPM.F90:446, 484` (`IareaT`), `:732` (`dy_Cu`) | ✅ |
| `core/dynamics/wetting_drying.py:218-219` | free-surface gradient g∇η | `(η_E − η_W)/du`, `/dv` | `MOM_barotropic.F90:1690-1691` (`IdxCu`); MOM6 PGF `IdxCu`/`IdyCv` | ✅ |
| `core/dynamics/wetting_drying.py:379-380` | implicit free-surface matrix | `g θ² Δt² Lu D/du` | MITgcm `model/src/ini_cg2d.F:89-93` (`dyG·hFacW·recip_dxC`) | ✅ |
| `core/dynamics/wetting_drying.py:180, 185` | Flather faces | face length `Lu`, `Lv` | MOM6 OBC uses `dy_Cu`/`dx_Cv` | ✅ |
| `core/components/layer_transport.py:66-67` | σ layer transports | `Lu·Du·Δσ·u` | `dy_Cu` | ✅ |
| `core/components/layer_transport.py:117-118, 127-135` | ALE layer transport (upwind/PPM) | face length `Lu`/`Lv`; CFL of the swept PPM thickness over `dx_t`/`dy_t` | `MOM_continuity_PPM.F90:732-733, 782, 1297-1298` (`dy_Cu`, `IdxT`; `CONT_PPM_VOLUME_BASED_CFL = False`) | ✅ |
| `core/components/layer_transport.py:140-144`; `core/vertical/ale.py:288, 360` | outflow limiter, Lagrangian thickness | `area_t` | `MOM_continuity_PPM.F90:446` (`IareaT`) | ✅ |
| `core/dynamics/sigma_baroclinic.py:1713-1716`; `core/model/driver.py:612` | layer volumes, ω | `area_t` | `areaT` | ✅ |
| `core/components/tracers.py` (`_advect_div`, `_advect_faces_h`, `_fct_fluxes` in `core/dynamics/sigma_baroclinic.py:257-463`) | WENO5 / FCT tracer advection | layer volume transports `Qx`, `Qy` (from the continuity) and cell volumes `h·area_t`; reconstruction in index space | `MOM_tracer_advect.F90:204-212` (`uhr`, `areaT·h`; PPM in index space) | ✅ |
| `core/components/tracers.py:412, 446-454` | horizontal tracer diffusion | `kh·Lu/du·h` | `MOM_tracer_hor_diff.F90:278, 282` (`dy_Cu·IdxCu`, `dx_Cv·IdyCv`) | ✅ |
| `core/components/pgf.py:160-161, 214, 218, 245`; `core/vertical/pgf.py:159`; `core/vertical/pgf_nonlinear.py:279, 348` | baroclinic PGF (σ Jacobian / two-term, FV linear and nonlinear EOS) | difference over `du` (= `dx_u`), `dv` (= `dy_v`) | `MOM_PressureForce_FV.F90:835, 843, 869, 872` (`IdxCu`, `IdyCv`) | ✅ |
| `core/dynamics/vector_invariant.py:52-66, 234-247` | relative vorticity ζ | circulation `v·dy_v`, `u·dx_u` | `MOM_CoriolisAdv.F90:424-425` (`dyCv`, `dxCu`) | ✅ |
| `core/dynamics/sigma_baroclinic.py:1511-1513`; `core/components/momentum.py:173-185` (was `vi.corner_area(dx_u, dy_v)`) | ζ = circulation / **A_q** | was ½(Δx_u + Δx_u′)·½(Δy_v + Δy_v′); now the grid's `area_q` (`vi.grid_corner_area`, `vector_invariant.py:40-49`) | `MOM_CoriolisAdv.F90:609` (`IareaBu`) | **fixed** (supergrid grids only; grids without `area_q` keep the old form, bit for bit) |
| `core/dynamics/vector_invariant.py` `pv_q` | PV q = (f+ζ) ΣA/Σ(A h) | `area_t` of the four cells | `MOM_CoriolisAdv.F90:336-337, 365-366, 625-627` (`areaT`) | ✅ |
| `core/dynamics/vector_invariant.py` `coriolis_pv` | Sadourny energy PV flux | transports `Qx`, `Qy` (face lengths), over `dx_u`, `dy_v` | `MOM_CoriolisAdv.F90:818-823, 1072-1076` (`vh`, `IdxCu`; `uh`, `IdyCv`) | ✅ |
| `core/dynamics/vector_invariant.py` `kinetic_energy` | KE (Arakawa / Godunov) | `area_u`, `area_v`, `area_t` (unmasked) | `MOM_CoriolisAdv.F90:1412-1415` (`areaCu`, `areaCv`, `IareaT`; masked) | ✅ (masking: §3) |
| `core/dynamics/vector_invariant.py` `ke_gradient` | −∇K | over `dx_u`, `dy_v` | `MOM_CoriolisAdv.F90:1528, 1533` (`IdxCu`, `IdyCv`) | ✅ |
| `core/components/momentum.py:68-118, 156-171` (`_coriolis`) | **σ (flux-form) Coriolis** | was ¼ Σ f_q v, unweighted | Sadourny energy with uniform h: `¼ IdxCu Σ f (v dxCv)`, `MOM_CoriolisAdv.F90:818-823, 1072-1076` | **fixed** (class a): §2.1 |
| `core/dynamics/hor_visc.py:74-110` (`smag_geometry`) | Smagorinsky tension/shear, stress divergence, Δ², bound | `dx_t`, `dy_t`, `dx_u`, `dy_u`, `dx_v`, `dy_v`, `area_u`, `area_v`; corner `dx_q`, `dy_q` were face means | `MOM_hor_visc.F90:707-722, 1757-1781, 3319-3341, 3403` (`dxT`, `dyT`, `IdyCu`, `IdxCv`, `IdyCv`, `IdxCu`, `dxBu`, `dyBu`, `IareaCu`, `IareaCv`) | **fixed** for supergrid grids (`dx_q`, `dy_q`, `area_q` from the grid); ✅ otherwise |
| `core/dynamics/nonlinear.py:99-123` (`_spacings_u`, `_spacings_v`) | constant-A_h Laplacian, advective-form momentum ('upwind3'/'centred') | u→u along x `dx_t`, along y was ½(Δy_u + Δy_u′); v→v along x was ½(Δx_v + Δx_v′), along y `dy_t` | distance between u points along y = `dyBu`, between v points along x = `dxBu` | **fixed** for supergrid grids (`dy_q`, `dx_q`); form is (b): §2.3 |
| `core/dynamics/sigma_baroclinic.py:726-800` (`_flux_form_u`) | σ flux-form momentum advection | u-cell volume ½(V_L + V_R), centre/corner transports | (ROMS/CROCO flux form; MOM6 has only the vector-invariant form) | (b): no curvilinear metric terms, §2.3 |
| `core/components/vertical_coordinate.py:164-169`; `core/dynamics/vector_invariant.py:156-210` (`truncate_cfl`) | CFL truncation backstop | `|u| Δt / dx_t(donor)` | `MOM_vert_friction.F90:2598-2643` (`u Δt dy_Cu / areaT`, volume CFL) | (c): equal on a rectangle, O(curvature) apart on a curved grid; a backstop that should not act; kept for bit for bit |
| `core/components/open_boundary.py:116-135` | Orlanski radiation | phase speed in index units | ROMS `u3dbc` (index units) | (c), as ROMS |
| `core/components/surface_fluxes.py` | τ from T points to faces | mean of the two wet neighbours (no metric) | `MOM_surface_forcing_gfdl.F90:1086-1097` (A-grid mean) | ✅ (vector rotation is Task 4) |
| `physics/bottom_friction.py` `cross_velocity_at_faces` | cross-velocity for the bed speed | 4-point mean (no metric) | CROCO `get_vbc.F:517-526` | (c), as CROCO |
| `core/model/periodic.py:122-170` (`periodic_grid`) | wrap-face metrics | `dx_u`, `dy_u`, `area_u = dx_u·dy_u`, `f_q`; now also `dx_q`, `dy_q`, `area_q` of the wrap corner column/row | MOM6 reentrant halos | **fixed**: corner metrics wrapped when present (A_q = Δx_q Δy_q at the wrap, MOM6's fallback `MOM_dyn_horgrid.F90:509`, class c); unchanged without them |
| `core/model/periodic.py` `pad_dataset` | halos for grid-reading operators | pads every (t, u, v, q) variable | — | ✅ (includes `dx_q`, `dy_q`, `area_q`) |
| `core/dynamics/free_surface.py:173-281` (`ImplicitFreeSurface`, phase-2 2-D model) | continuity, gradient, matrix | `dy_u`, `dx_u`, `dx_v`, `dy_v`, `area_t` | as above | ✅ (not used by `SigmaBaroclinic`) |
| `core/dynamics/pressure.py`, `core/dynamics/momentum.py` (phase-1 z-level line) | — | — | — | not audited: not used by `SigmaBaroclinic` |
| `f_t` | — | used only by `diagnostics/domain_check.py:210` (CFL advice) | no MOM6 counterpart | ✅ (not in any operator) |

## 2. Fixes and decisions

### 2.1 σ flux-form Coriolis (class a)

`Momentum.coriolis` in the flux form (σ, and ALE with `momentum='flux'`) took the unweighted mean of the four
neighbouring velocities, ¼ Σ f_q v. Its docstring claims an antisymmetric operator that does no work, which holds
only when every face length equals the centre distance of the neighbouring faces. On a stretched, spherical or
curvilinear grid Σ A_u u C_u + Σ A_v v C_v ≠ 0 and no discrete geostrophic state is steady. Now
(`core/components/momentum.py:68-118`):

    C_u = ¼/Δx_u Σ₄ f_q (v Δx_v),        C_v = −¼/Δy_v Σ₄ f_q (u Δy_u)

the Sadourny (1975) energy form of MOM6 (`MOM_CoriolisAdv.F90:818-823, 1072-1076`) with a uniform thickness.
Σ A_u u C_u + Σ A_v v C_v = 0 exactly with A_u = Δx_u Δy_u, A_v = Δx_v Δy_v, and the state
η = f/(gH) · ¼ Σ₄ ψ with transports Q_x = −δ_y ψ, Q_y = δ_x ψ is steady on any orthogonal grid. When all the
ratios Δx_v/Δx_u and Δy_u/Δy_v are exactly 1 (`coriolis_metrics` returns None: uniform Cartesian grids) the old
unweighted expression runs, bit for bit. **Changed results:** σ (and ALE `momentum='flux'`) runs with f ≠ 0 on
stretched Cartesian grids (BayIW σ configurations) and on spherical grids. None of the N1 goldens (uniform
Cartesian) nor of the 9 smoke cases (ALE vector-invariant on the stretched BayIW cut; σ cases uniform) change, so
no golden was regenerated. Measured on the conformal test grid: the geostrophic state drifted by
5.8e-5 U per step before, 5e-12 U after.

### 2.2 Corner metrics from the supergrid (vorticity, Smagorinsky, Laplacian spacings)

pyOCEAN's own grids (`core.grid.horizontal`) carry no corner metrics, so the operators rebuilt them from face
metrics. A supergrid grid carries MOM6's `dxBu`, `dyBu`, `areaBu` (`dx_q`, `dy_q`, `area_q`), which differ from the
rebuilt values by O(curvature) (1e-4 relative in `area_q` on the conformal test grid). The vorticity area
(`vi.grid_corner_area`), the Smagorinsky corner spacings (`smag_geometry`) and the Laplacian/advective spacings
(`_spacings_u`, `_spacings_v`) now use them when the grid has them; without them nothing changes (bit for bit).
In the interior of a stretched Cartesian grid both are equal (invariance tests below).

### 2.3 Not changed: operators without curvilinear metric terms (class b)

- σ flux-form momentum advection (`_flux_form_u`) and the advective forms 'upwind3'/'centred' advect the grid
  components of the velocity as Cartesian components; on a curved grid ROMS adds the curvilinear terms
  (`CURVGRID`: u v ∂(1/n)/∂ξ − v² ∂(1/m)/∂η, …).
- The constant-A_h Laplacian (`laplacian_3d`, used when `smagorinsky = 0` and `ah > 0`) is ∇² of each component,
  not the stress-tensor divergence; on a curved grid it lacks the metric terms (and does not conserve angular
  momentum).

Both need restructuring, not a metric swap, so they are out of this task. **Recommendation for curvilinear grids
(NWA12):** `vertical_coordinate='ale'` with the default vector-invariant momentum (invariant form: no metric
terms needed, MOM6) and `smagorinsky > 0` (tensor form, MOM6) instead of the constant Laplacian — what the NWA12
cut will use. A σ run on a curvilinear grid is consistent to the size of the missing curvature terms.

### 2.4 `area_u`, `area_v`: unmasked (decision)

MOM6 masks `areaCu = dxCu · dy_Cu` by the open face (`MOM_grid_initialize.F90:1332-1335`). pyOCEAN keeps
`area_u = dx_u · dy_u` unmasked: it enters only (i) the Arakawa/Godunov KE, multiplied by u² of a face whose
velocity is zero when it is land or dry, and (ii) the Smagorinsky stress divergence `iarea_u`, whose tendency is set
to zero on non-wet faces (`hor_visc.py`, `np.where(wet_u & ~open_u, fu, 0)`). Both give MOM6's result, and the
dynamic wet/dry mask of pyOCEAN is stricter than MOM6's static land mask. No operator needs the masked area.

### 2.5 CFL truncation (class c, kept)

MOM6 truncates on the volume CFL u Δt Δy_Cu / A_T; pyOCEAN on u Δt / Δx_T of the donor cell. They are equal on a
rectangle and differ by O(curvature) on a curved grid; the truncation is a backstop that should never act (counted
in `SigmaState`), and changing it would alter the bit-for-bit results of runs where it does act. Recorded as (c).

## 3. Tests (`tests/test_curvilinear_invariance.py`) and mutation evidence

Invariance (criterion 2, 1e-10 relative): stretched Cartesian grid ≡ the same grid as a planar MOM6 supergrid
(`test_free_surface_wave_invariant` σ/z\* with Smagorinsky, `test_geostrophic_adjustment_invariant` σ/z\*,
`test_tracer_advection_invariant` WENO5 + FCT, `test_stratified_rest_invariant` z\*/hybrid with `wright_full`).

Exactness on a non-separable orthogonal grid (conformal map z = w + c w², every metric varies in i and j,
neighbouring metrics differ by ≥ 0.2 %, Δx_t and Δy_t by 70 % over the domain):
`test_face_metrics_used` (dy_u or dx_v perturbed at one face → that face's transport in the same proportion),
`test_divergence_free_flow_steady`, `test_geostrophic_balance_steady` (σ and z\*), `test_tracer_uniform_and_conserved`,
`test_coriolis_does_no_work_on_curvilinear_grid`, `test_pgf_metric_used` (Δx_u, Δy_v of the baroclinic PGF: σ Jacobian, FV linear, FV `wright_full`),
`test_supergrid_corner_metrics_used`.

Each metric swap below was applied temporarily to the code and the test file run (script in the task report):

| mutation (temporary) | file | tests that fail |
|---|---|---|
| M1 continuity u-face length `dy_u` → `dy_t` of the east cell | `wetting_drying.py:135` | `face_metrics_used[x-σ, x-z*]`, `divergence_free_flow_steady[σ, z*]`, `geostrophic_balance_steady[σ, z*]` |
| M2 continuity v-face length `dx_v` → `dx_t` of the north cell | `wetting_drying.py:136` | `face_metrics_used[y-σ, y-z*]`, `divergence_free_flow_steady[σ, z*]`, `geostrophic_balance_steady[σ, z*]` |
| M3 free-surface / PGF distance `dx_u` → `dx_t` of the east cell | `wetting_drying.py:135` | `geostrophic_balance_steady[σ, z*]`, `pgf_metric_used[x-σ, x-z*, x-z*-wright_full]` |
| M4 σ Coriolis unweighted (the code before this task) | `momentum.py` `coriolis_metrics` | `geostrophic_balance_steady[σ]`, `coriolis_does_no_work` |
| M5 σ Coriolis `v Δx_v` → `v Δy_v` | `momentum.py` `_coriolis` | `geostrophic_balance_steady[σ]`, `coriolis_does_no_work` |
| M6 vector-invariant first-step transport `Δy_u` → `Δx_u` | `momentum.py` `vector_invariant` | `geostrophic_balance_steady[z*]` |
| M7 vorticity area `area_q` → `corner_area(Δx_u, Δy_v)` (before this task) | `vector_invariant.py` `grid_corner_area` | `supergrid_corner_metrics_used` |
| M8 Smagorinsky corner spacings from face means (before this task) | `hor_visc.py` `smag_geometry` | `supergrid_corner_metrics_used` |
| M9 ALE Lagrangian thickness over `Δx_u Δy_v` instead of `area_t` | `layer_transport.py:144` | `tracer_uniform_and_conserved` |
| M10 ALE layer volume `h Δx_u Δy_v` instead of `h area_t` | `sigma_baroclinic.py:1713` | `tracer_uniform_and_conserved` |
| M11 FV PGF, nonlinear EOS: `Δx_u` of the west neighbour face | `pgf.py:245` | `pgf_metric_used[x-z*-wright_full]` |
| M12 σ Jacobian PGF: `Δx_u` of the west neighbour face | `pgf.py:160` | `pgf_metric_used[x-σ]` |
| M13 FV PGF, linear EOS: `Δx_u` of the west neighbour face | `pgf.py:214` | `pgf_metric_used[x-z*]` |

Every mutation is caught; the invariance tests pass under every mutation (the same code runs on both grids, by
design: they check the supergrid path, not the operators). The vorticity area enters ζ, which is O(Ro) in the
geostrophic test, so M7 is caught only by the dependency test.

Before the fixes, the new tests failed in `test_geostrophic_balance_steady[sigma]`,
`test_coriolis_does_no_work_on_curvilinear_grid` (σ Coriolis, §2.1) and `test_supergrid_corner_metrics_used` (§2.2);
the continuity, PGF and tracer operators already passed (✅ rows).

Not covered by a swap-sensitive test: the CFL truncation (backstop, §2.5), the Orlanski phase speed (index units),
the advective-form momentum and constant Laplacian spacings (forms of class b, §2.3; checked by the dependency of
`_spacings_u/_v` on `dy_q`/`dx_q` through the invariance runs only).

## 4. Bit for bit on existing grids

- `pytest tests/test_n1_golden.py`: 33/33 bit for bit (uniform Cartesian goldens; `coriolis_metrics` returns None,
  no `area_q`/`dx_q` in the grid). No golden regenerated.
- `scripts/n1_smoke.py` against `main` (worktree of `main`, 2026-11-01): 9/9 hashes identical, cost
  0.97–1.00 × main (`bayiw_cut_zstar` on the stretched BayIW cut uses the vector-invariant form, which this task did
  not change for grids without corner metrics).
- Results that do change (deliberate, §2.1): σ and ALE `momentum='flux'` runs with f ≠ 0 on stretched Cartesian
  or spherical grids (the Coriolis weights Δx_v/Δx_u, Δy_u/Δy_v are no longer dropped). No stored reference in the
  test suite covers that combination.
