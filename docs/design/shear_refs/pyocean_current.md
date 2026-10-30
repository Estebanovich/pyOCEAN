# pyOCEAN today: what exists for shear mixing and where a new scheme plugs in

Repository state: `main` at b5079a6 (2026-10-30). Citations `file:line` relative to the repo root.

## 1. KPP (`physics/mixing/kpp_cvmix.py`, `KPPClosure`)

- Boundary-layer KPP only: bulk Richardson number for the OBL depth (`kpp_cvmix.py:368-…`, L94 eq. 21),
  `ri_crit` = 0.3 is the **bulk** Ri_c (`:106-108`, `:160`), not an interior gradient-Ri scheme.
- Interior: "MOM6 passes zero interior K" (`:44-47`); the module docstring lists "the interior shear mixing of L94
  (§4)" as **not implemented** (`:69-75`); `KPPClosure` docstring: "Interior Richardson-number mixing (L94 §4) is
  not part of 'kpp' … decided here: off" (`core/components/closure.py:652-653`). So CLAUDE.md is right: no
  interior Ri mixing anywhere in pyOCEAN.
- `kappa_bg`/`nu_bg` default 0 (`kpp_cvmix.py:72-75`, `:157`): no background by design (anti-goal).

## 2. Stratified GLS k-ε (`physics/mixing/gls.py`, `GLSClosure`)

GLS **is** a shear-driven closure: P = ν S² (`gls.py:14`, `:257-260`; or the energy-conserving Burchard 2002 form,
`gls_production='energy'`), buoyancy B = −κ_t N² and Canuto-A stability functions (Ri-dependent through α_N, α_M),
Galperin limit (`gls.py:21-30`). It mixes where Ri is low and switches off where N² dominates, with a
prognostic k, ε (memory, advection-free). Floors ν ≥ 1e-5, κ_t ≥ 1e-6 (CROCO `nuws`, `gls.py:97`, `:190`).
ALE support: thin-layer merge (MOM6 `eliminate_massless` pattern, per-column threshold,
`core/components/closure.py:52-118`), dual-grid remap of k, ε (`:503-531`).

Behaviour in DOME hybrid + GLS (10 km, 50 days; `validation/dome/README.md:190-235`):
- b_f −16.18 → −12.99 × 10⁻³ m s⁻² (Legg iso-ν0.5 −12.0, Legg z −11.7): 76 % of the way (`:211`, `:221`).
- Plume transport flat at 18–22 Sv vs 35–37 Sv in MITgcm / z\* (`:223`); α_E 2.29e-3 vs MITgcm 1.94e-3:
  GLS dilutes near the inlet, then little entrainment downstream. Hypothesis (c): the hybrid plume is 1–2 thick
  layers, so entrainment hangs on κ_t at a single interface (`:223`).
- ν falls to its floor (1e-5) outside the plume (hypothesis, not measured): max|u| 5.66 m/s, 1072 outflow-limiter
  events, 1 CFL truncation (`:224`). With ν_v = 1e-5 constant the run aborts (1034 truncations, `:226`).
- With Smagorinsky C = 0.15: truncations 1 → 0, b_f −13.31…−13.68 (slightly *less* mixing: less lateral shear →
  less GLS production, hypothesis, `:279-283`).
- `kv_low` (ν_v = 1e-4 constant, no closure) + Smagorinsky completes 50 days but with 61 truncations, 239 337
  limiter events, b_f −17.91, 9–12 Sv: "falta mezcla por cizalla (Jackson et al. 2008, o el GLS) con ν_v bajo",
  class (b) (`:297-301`, `:309`).
- README list items to do: "Mezcla por Ri (Jackson et al. 2008) como opción frente al GLS en la híbrida"
  (`:234`), "(b) Mezcla por cizalla con ν_v bajo" (`:309`).

Caveat from the MOM6 read (`mom6_kappa_shear.md` §7.1): MOM6's own DOME runs Jackson with PRANDTL_TURB = 0
(diffusivity only). Jackson as MOM6 configures it would *not* couple the layers in momentum for `kv_low`; it is
the GLS-like viscosity (Pr ≈ 1) that does. Decide Pr explicitly (Jackson default 1.0).

## 3. Closure interface (`core/components/closure.py`)

Base class `Closure` (`:126-191`): `viscosity(st)` → (ν_u, ν_v) at face interfaces (N+1, ny, nx+1) /
(N+1, ny+1, nx) or None; `update(u, v, h, rho, ustar2_sfc, ustar2_bot, bflux)` with cell-centre u, v (N, ny, nx);
`tracer_diffusivity()` → κ (N+1, ny, nx) or None; `nonlocal_flux()`; `remap(h_src, h_dst)`; `bind_ale`;
`load`/`state` (SigmaState.closure, by reference); `state_fields`/`load_state` (restart); `check(ctx)`.
Class attributes: `stateful`, `update_phase` ('implicit' = phase 4 after the implicit viscosity, as GLS;
'tracers' = phase 7, as KPP), `restart_prefix`. Index (k, j, i), k = 0 at the bed; interfaces 0 = bed, N = surface.

Driver use (`core/model/driver.py`):
- σ: `nu = cl.viscosity(st)` → free-surface/implicit viscosity (`:423-426`); update after it if
  `update_phase == "implicit"` (`:430-…`).
- ALE: same at `:566-577`; inputs from `_ale_closure_forcing` (`:580-597`): u, v averaged to centres
  (½ of the two faces), h = hⁿ, surface and bed u*².
- Tracers: `kt = cl.tracer_diffusivity()`; if not None it **replaces** the constant `kv` (σ `:483-492`,
  ALE `:712-718`); for GLS, `kv` becomes κ_t's floor (`:422`, `:565`).
- Viscosity: if the closure returns ν it **replaces** the solver's own ν (`core/dynamics/sigma_wd.py:279-286`:
  given ν wins; else parabolic / `nu_const` / 0).
- `make_closure` (`closure.py:874-893`) maps `core.closure` ∈ {'constant', 'none', 'parabolic' (σ only),
  'gls', 'kpp'} to one component; ALE accepts 'constant', 'none', 'gls', 'kpp'
  (`core/dynamics/sigma_baroclinic.py:1340-1342`, pre-flight `core/model/preflight.py:155-156`).

## 4. Can closures be combined today? No.

`closure` is a single string; one component per role (`Role.CLOSURE`). Each closure's ν/κ replaces the constant
values (`sigma_wd.py:279-286`, `driver.py:483-492`); KPP ignores `kv`/`nu_const` with a pre-flight warning
(`preflight.py:222-226`); GLS treats `kv` as a floor, not an addition. There is no additive path. MOM6, by
contrast, sums: background Kd + Kd_shear (`MOM_set_diffusivity.F90:571-581`), KPP additive by default
(`MOM_CVMix_KPP.F90:419-422`, `:1015-1028`), Kv_shear added to the viscous Kv (`MOM_vert_friction.F90:2163-2205`),
and forbids two *shear* schemes at once (`MOM_CVMix_shear.F90:265-273`).

Sensible combinations: constant (background ν, κ) + shear; KPP + shear (MOM6/CROCO/MITgcm standard: interior
shear below the OBL). **Not** GLS + shear: GLS already produces shear turbulence (double counting; MOM6 has no
k-ε to combine with, CROCO uses either LMD_RIMIX or GLS, `croco/OCEAN/cppdefs.h:250-262`).

## 5. Where a shear scheme would live (options, not decided)

- Column kernel: `physics/mixing/kappa_shear.py` (JHL 2008, Numba `@njit` per column with a NumPy oracle, as
  `gls_merged.py`) and/or `physics/mixing/ri_mixing.py` (L94/PP81: algebraic, NumPy). Index (k, j, i), k = 0 at
  the bed — the MOM6 code is top-down, so flip as `kpp_column` does (`kpp_cvmix.py` conventions block).
- Component, two shapes:
  (a) a new `ShearClosure(Closure)` in `core/components/closure.py` that also owns an optional constant
      background (ν_bg, κ_bg) = 'constant + shear' in one component; `closure='kappa_shear'` (or 'ri');
  (b) an additive layer: `Closure` gets a composite (`CompositeClosure([base, shear])` summing `viscosity` and
      `tracer_diffusivity`, `update` called on each, `remap`/state per part, restart prefixes kept) so that
      'constant', 'kpp' and later others can take `shear_mixing='jackson'|'lmd94'|'pp81'|None`. Pre-flight refuses
      'gls' + shear.
  (b) mirrors MOM6's additive design and keeps KPP + shear possible; (a) is smaller.
- Phase: MOM6 computes Kd_shear in the diabatic step before KPP and tracer diffusion, and Kv_shear is used by the
  *next* dynamics step (lagged). pyOCEAN 'tracers' phase (7, like KPP: κ for this step's tracers, ν for the next
  step) is the MOM6 order; 'implicit' (4, like GLS) uses the post-viscosity velocities. Both are consistent with
  the existing interface.
- Inputs already available in `update`: cell-centre u, v (MOM6 tracer-point mode does the same averaging,
  `find_uv_at_h`), h, ρ of the model EOS (N² as `gls.buoyancy_frequency2`), f at centres (`make_closure`,
  `closure.py:886`), Δt.
- Thin layers: reuse `thin_layer_threshold` / `active_layers` and the KPP `_merged` pattern (`closure.py:755-…`)
  — the same MOM6 merge, already reviewed.
- State: JHL has none between steps except the lagged ν (restart like `kpp_akv`); L94/PP81 none. ALE: keep ν by
  interface index (MOM6 default `REMAP_AUXILIARY_VARS = False`).
- Missing companion: no convective adjustment in 'constant' (CLAUDE.md N2a open item); JHL clips N² ≥ 0 and does
  no convection; L94 gives Ri < 0 → ν⁰ (5e-3), weak as convection. A dense overflow is statically stable, so this
  matters less for DOME than for the general closure.
- Tests: column against MOM6's `kappa_shear_column` on fixed profiles (as `tests/test_vertical_vs_mom6.py` for
  remap), Ri = ∞ → 0, Ri ≥ Ri_c → 0 (JHL: only where N² ≥ Ri_c S² at every iterate), κ → 0 at bed and surface,
  merged column ≡ column without vanished layers, restart bit for bit, no change with the scheme off (goldens).
