# Configuration reference

Every constructor argument of the user-facing classes and functions, with the default read from the code
signature and the automatic (`None` →) rules applied in `__init__` / `_setup_ale`. Line numbers refer to the code
on 2026-10-14; the function or class name is given too because lines drift. For the "how" and "why", see
[Building a model](building-a-model.md); for the numerics, the [Theory](../theory/equations.md) pages.

Conventions used in all tables: arrays are `(k, j, i)` with **k = 0 at the bed** unless stated; u-faces have nx + 1
columns, v-faces ny + 1 rows; "σ" means `vertical_coordinate="sigma_eulerian"`, "ALE" means `"ale"`.

## `SigmaBaroclinic` — the stratified 3-D model

``SigmaBaroclinic.__init__`` (`core/dynamics/sigma_baroclinic.py:1005`). Arguments are listed in signature order.

### Grid, time step, equation of state

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `h_grid` | `xr.Dataset` | required | — | horizontal C-grid | output of `create_*_grid` | uses `dx_u, dy_u, dx_v, dy_v, area_t, f_q`; ALE also `dx_t, dy_t, area_u, area_v`; `mask_t` is **ignored** |
| `bathy` | array `(ny, nx)` | required | m | rest depth, positive down | > 0 water; ≤ 0 floodable land; NaN wall | active cells `np.isfinite(bathy)` (`:1065`) |
| `dt` | float | required | s | time step | > 0 | baroclinic and barotropic share one Δt (θ-implicit free surface) |
| `n_layers` | int | 10 | — | number of layers N | ≥ 1 | ALE: also the target grid size |
| `rho0` | float | 1000.0 | kg m⁻³ | Boussinesq reference density ρ₀ | > 0 | divides τ and the PGF |
| `t_coef` | float | 0.2 | kg m⁻³ °C⁻¹ | α of the linear EOS ρ = ρ₀ − α(T − T₀) | any (0: passive T) | ``density`` (`:1285`) |
| `t_ref` | float | 0.0 | °C | T₀ of the EOS | any | also T of dry columns in `initial_state(h_src=...)` |
| `g` | float | 9.81 | m s⁻² | gravity | > 0 | |

### Vertical mixing and bed

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `z0` | float or None | None | m | bed roughness of the first-layer law of the wall, C_d = [κ/ln(Δz₁/2z₀)]² | None (no bed stress) or > 0 | ALE requires None (`:1148-1150`); C_d clipped to [1e-4, 0.1] (``first_layer_cd`` (`physics/bottom_friction.py:48`)) |
| `closure` | str | `"constant"` | — | vertical viscosity | `"gls"`, `"parabolic"`, `"constant"`, `"none"` (`core/dynamics/sigma_wd.py:49`) | `"kpp"` (CVMix/MOM6, since N3) is also accepted. ALE: `"constant"`, `"none"`, `"gls"` and `"kpp"` since N3 (not `"parabolic"`). GLS = CROCO k-ε Canuto A, with buoyancy by default (`gls_stratified=True`; `False` = neutral, `physics/mixing/gls.py:14`); parabolic ν = κu*z(1 − z/D) needs `z0` |
| `nu_const` | float | 1e-4 | m² s⁻¹ | viscosity of `"constant"` | ≥ 0 | |
| `kv` | float | 0.0 | m² s⁻¹ | vertical tracer diffusivity, implicit, constant | ≥ 0 | independent of the closure |
| `kh` | float | 0.0 | m² s⁻¹ | horizontal tracer diffusivity along layers, explicit flux form | ≥ 0 | diffuses T − T_ref(z) when a reference profile is set (`_diffuse_h_numpy`, `:1600`); ALE uses min(h_L, h_R) at faces |
| `ah` | float | 0.0 | m² s⁻¹ | horizontal Laplacian viscosity, free slip | ≥ 0 | Re_Δ = UΔx/A_h controls spurious mixing (Ilicak et al. 2012) |
| `smagorinsky` | float | 0.0 (off, bit for bit as before) | — | Smagorinsky coefficient C (MOM6 `SMAG_LAP_CONST`): thickness-weighted Laplacian stress-tensor viscosity ν = max(`ah`, C Δ² \|S\|), Δ² = 2Δx²Δy²/(Δx²+Δy²), per layer, free slip | finite, ≥ 0 (pre-flight error otherwise); assignable after construction like `ah` | recommended 0.15 (OM4p5, Adcroft et al. 2019); 0.5 reproduces Ilicak et al. 2012 Fig. 19b (C_s = 1/√2). `ah` is then the background (the maximum, not a sum). ν is bounded every step for AB3 (≈ 0.055 Δx²/Δt, times the thin-face ratio); points where the bound acted are counted in `SigmaState.n_visc_limited` (summed over steps; `None` when off; kept in the restart) and also include thin or clipped faces where the bound is below `ah`. The last step's ν at h points is the registry variable `nu_smag` (`self._momentum.last_nu_t`; output on request). Free slip at land and dry faces, no tendency on open-boundary faces, periodic axes supported. Not shown to extend the stable Δt (Lorenz D). Theory: [Momentum](../theory/momentum.md) |

### Free surface

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `theta` | float | 0.6 | — | implicit weight of the free surface | 0.5 ≤ θ ≤ 1 (`core/dynamics/wetting_drying.py:94-95`) | ½ neutral, 1 damps gravity waves |
| `d_min` | float | 0.05 | m | face total depth below which a face is dry (CROCO `Dcrit` analogue) | > 0 | Thacker: 0.05 better than 0.01 (`validation/thacker/README.md`) |
| `solver` | str or None | None → `"cg"` in ALE, `"direct"` in σ (`:1076-1077`) | — | free-surface solver | `"direct"`, `"cg"` | CG uses the LU of a lagged matrix as preconditioner and recomputes η from the final fluxes (exact conservation); ≈1 s → ≈0.1 s per step on BayIW 560 × 352 (docstring `:967-968`) |
| `theta_bc` | float or None | None → 0.5 (`:1047-1048`) | — | time weight of the *baroclinic* part u′ of the layer transport | 0 … 1 | the depth mean keeps `theta`; ½ removed an internal-wave damping ~(θ − ½)ω²Δt (Ilicak IW 0.45 → 0.74, `validation/ilicak_benchmark/README.md`) |

### Momentum

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `momentum` | str or None | None → `"vector_invariant"` in ALE, `"flux"` in σ (`:1130`) | — | form of the momentum equation | `"vector_invariant"` (ALE only, `:1133-1135`), `"flux"` | vector invariant = MOM6 Sadourny PV flux + ∇K |
| `adv` | str | `"flux"` | — | momentum advection when `momentum="flux"` | `"flux"` (WENO5-Z flux form), `"upwind3"`, `"centred"` (advective form, `core/dynamics/nonlinear.py:165`) | **not checked in `__init__`**: a wrong value raises at the first step from `advection_u` (`core/dynamics/nonlinear.py:162`). Ignored with `"vector_invariant"` |
| `ke_scheme` | str | `"arakawa"` | — | kinetic energy of the vector-invariant form | `"arakawa"`, `"gudonov"` (MOM6 spelling) | `"gudonov"` for gravity-current fronts, `"arakawa"` for waves (docstring `:955-959`); ALE only |
| `max_truncations` | int | 1000 | — | ALE: stop after this many CFL velocity truncations at faces thicker than `h_thin_bed` | ≥ 0 | raises `RuntimeError` (`:2069-2079`); truncation at \|u\|Δt/Δx_donor > 0.5 to 0.45 Δx/Δt (``truncate_cfl`` (`core/dynamics/vector_invariant.py:172`)) |

### Tracers

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `tracer_scheme` | str | `"weno5"` | — | face values of the tracer advection | `"weno5"` (WENO5-Z), `"weno5js"`, `"upwind3"`, `"upwind1"` (`:1027`) | UP3 overshoots ~50 % at a lock-exchange nose (`:18-19`) |
| `tracer_time` | str | `"ssprk3"` | — | tracer time stepping | `"ssprk3"`, `"pc"` (CROCO predictor–corrector) (`:1024`) | `"pc"` runs the NumPy path and cannot use FCT |
| `tracer_limiter` | str or None | None → `"fct"` in ALE with `"ssprk3"` and WENO5, else `"none"` (`:1112-1114`) | — | Zalesak (1979) FCT around the SSP-RK3 WENO5 step | `"fct"`, `"none"` | `"fct"` outside ALE + SSP-RK3 + WENO5 raises (`:1117-1120`); bounds use x, y and k ± 1 neighbours |
| `mix_diagnostic` | bool | True | — | ALE: advect and remap T² for `st.mix_num` | True, False | False halves the tracer work and sets `mix_num=None`; σ always computes it. BayIW z*: 3.32 → 2.72 s/step (`validation/bayiw_baroclinic/README.md`) |

### Pressure gradient

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `pgf` | str | `"jacobian"` | — | baroclinic pressure gradient | `"jacobian"` (Song 1998 density Jacobian), `"twoterm"`, `"fv"` (Adcroft et al. 2008, `core/vertical/pgf.py`) (`:1021`) | ALE runs in the validation all use `"fv"` |
| `rho_ref` | None, `"auto"`, or callable | None | kg m⁻³ | reference profile ρ_ref(z) subtracted before the PGF and the K_h diffusion | None (ρ′ = ρ − ρ₀); `"auto"` (built by `initial_state`, 400 bins, ``reference_profile`` (`:780`)); any callable z → ρ | seamount 42 → 0.41 cm s⁻¹ (`validation/seamount/README.md`) |

### Vertical coordinate (ALE)

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `vertical_coordinate` | str | `"sigma_eulerian"` | — | σ with ω, or ALE | `"sigma_eulerian"`, `"ale"` (`:1127`) | |
| `dsigma` | array `(N,)` or None | None (uniform) | — | σ fractions Δσ_k, bed first | positive, Σ = 1 to 1e-12 (`core/dynamics/sigma_wd.py:175`) | σ: the layers; ALE: default of `regrid_params["dsigma"]`. Builders: ``stretched_sigma``, ``sigma_from_thicknesses`` (`core/grid/sigma_levels.py`) |
| `regrid` | str | `"sigma"` | — | ALE grid rule | `"sigma"`, `"zstar"`, `"hybrid"` (`:1144`) | **ALE's default is ALE-σ**, not z* |
| `regrid_params` | dict or None | None → `{}` + `dsigma` | — | parameters of the rule | see [below](#regrid_params) | required keys checked (`:1156-1158`); unknown keys are ignored but written to restart files |
| `remap_scheme` | str | `"ppm_h4"` | — | vertical remap reconstruction | `"pcm"`, `"plm"`, `"ppm_h4"` (`:103`) | = MOM6 `remapping_core_h` to ≤ 5e-12 (CLAUDE.md) |
| `thickness_flux` | str | `"ppm"` | — | layer thickness at the faces in the Lagrangian step | `"ppm"` (MOM6 `continuity_PPM`), `"upwind"` (donor cell) (`:1139`) | upwind 28 % cheaper; PPM needs ~20 layers in density currents (`:961-963`) |
| `h_thin_bed` | float | 0.01 | m | bed face layers thinner than this carry no velocity (HYCOM); also the threshold for counting truncations | ≥ 0 | |
| `init_iterations` | int | 10 | — | regrid–remap passes of the hybrid initial state without `h_src` | ≥ 0 | |
| `thin_layer_coupling` | str or None | None → `"mom6"` in ALE, `"none"` in σ (`:1049-1050`) | — | face thickness of the implicit vertical viscosity | `"mom6"` (upwind-biased, MOM6 `MOM_vert_friction.F90:1519-1533`), `"none"` | ignored in σ; with land cells `"mom6"` emits harmless divide warnings ([Troubleshooting](troubleshooting.md#runtime-warnings-from-sigma_wdpy243)) |
| `thin_layer_cutoff` | float or None | None → 0.0 (off) (`:1051-1052`) | m | HYCOM blending of thin layers with the layer above | ≥ 0 (`:1057-1058`) | any cutoff ≥ 0.01 m slowed the hybrid lock-exchange front 1.05 → 0.70 U (docstring `:992-998`) |

### Open boundaries

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `open_boundaries` | tuple of str | `()` | — | open edges | subset of `("west", "east", "south", "north")` (`:1030-1032`) | Flather + Orlanski + upstream tracers; ALE: no H ≤ 0 cell at an open edge (`:1177-1179`) |
| `sponge_cells` | int | 0 | cells | width of the relaxation band | ≥ 0; > 0 needs open boundaries (`:1033-1034`) | MITgcm OBCS profile, rate (N − d)/(N τ(d)) (`_sponge_rates`, `:906`); not on `"clamped"` edges |
| `sponge_tau` | tuple | `(1200.0, 60.0)` | s | `(τ_inner, τ_boundary)` | > 0 | τ(d) = [(N − d) τ_boundary + d τ_inner]/N |
| `orlanski_cmax` | float | 1.0 | — | maximum Courant number r = cΔt/Δx of the Orlanski radiation of u′ | ≥ 0 | BayIW and MITgcm use 0.45 |
| `sponge_tracer` | bool | True | — | also relax T toward the exterior in the band | | BayIW runs: False |
| `open_boundary_state` | callable or None | None | — | t [s] → `{edge: OBData}` exterior, evaluated at tⁿ⁺¹ | edges must be open (`:1036-1037`; ``validate_state`` (`core/dynamics/open_boundary.py:19`)) | None = resting exterior with the initial T profile |
| `open_boundary_mode` | dict or None | None → `"radiating"` on every open edge | — | per-edge treatment of u′ | dict edge → `"radiating"` or `"clamped"` (`:1039-1043`) | `"clamped"` needs `open_boundary_state` (`:1044-1045`) |
| `obc_tau_in` | float | 86400.0 | s | nudging of the radiated u′ toward u′ₑₓₜ on inflow (`"radiating"` with a state) | > 0 | CROCO `u3dbc.F` |
| `obc_tau_out` | float | 864000.0 | s | same, on outflow | > 0 | |

`OBData` (`boundaries/obcs/__init__.py:124`) fields used by the model: `side`; `T (nz, n)` [°C]; `u_normal (nz, n)`
[m s⁻¹] (u for west/east, v for south/north, model sign); `zw (nz+1, n)` [m], surface first, strictly decreasing,
reaching the bed of the boundary column; optional `eta (n,)` [m] and `ubar (n,)` [m s⁻¹] for Flather (None = 0).
`n` = ny on west/east, nx on south/north. `S` is ignored (no salinity).

### Methods and state

| Method | Signature | Returns |
|---|---|---|
| ``initial_state`` (`:1190`) | `(eta, T, h_src=None)` | resting `SigmaState`; sets `rho_ref` if `"auto"`, `T_ext`, `h_ext` |
| ``step`` (`:2084`) | `(st, tau_x=None, tau_y=None)`, τ [N m⁻²] at u-/v-faces | new `SigmaState` |
| ``layer_heights`` (`:1267`) | `(eta, h=None)` | layer-centre heights (N, ny, nx) [m], z = 0 at rest |
| ``layer_volumes`` (`:1289`) | `(eta, h=None)` | layer volumes [m³] |
| ``density`` (`:1285`) | `(T)` | ρ [kg m⁻³] |
| ``tracer_content`` (`:1296`) | `(st, power=1)` | Σ V Tᵖ |
| ``rpe`` (`:1300`) | `(st)` | RPE [J] with the flat-basin geometry |

`SigmaState` (`:107`): `eta, u, v, T` (prognostic), `h` (ALE thickness, else None), `g_hist` (AB history, newest
first, ≤ 3), `omega` (σ only), `mix_num`, `Qx, Qy, Fz` (last transports), `time` [s], `n_trunc`, `n_limited`.

Module functions: ``set_threads(n)`` (`:166`, default 1 thread, clipped to `NUMBA_NUM_THREADS`);
``reference_profile(z, rho, n_bins=400)`` (`:780`) → `ReferenceProfile`.

### `regrid_params`

Read by ``regrid_columns`` (`core/vertical/regrid.py:345`). `dsigma` is filled in from the model when absent
(`core/dynamics/sigma_baroclinic.py:1155`).

| Key | Rules | Required | Default | Units | Meaning |
|---|---|---|---|---|---|
| `dz_top` | zstar, hybrid | yes | — | m | nominal layer thicknesses, **surface first** (MITgcm `delR`). z*: z = η + (D/H) z*; levels below the bed collapse to `h_min`, the deepest active one is truncated (partial cell). Hybrid: the **z* floor** — an interface is never above its z* position; Σ should reach the deepest bed (warning otherwise, `:1159-1168`) |
| `rho_target` | hybrid | yes | — | kg m⁻³ | N + 1 interface target densities, surface first, increasing, in the model EOS. Helpers: ``target_densities(rho_top, n_layers, n_light=0, d_rho_light=0.01)`` (`core/vertical/regrid.py:292`) |
| `max_h_top` | hybrid | no | ∞ | m | maximum thickness of each layer, surface first (MOM6 `coord_hycom.F90:270-279`) |
| `d_sigma` | hybrid | no | 0 | m | total depth below which the column is pure σ (shallow window) |
| `d_hybrid` | hybrid | no | 0 (no window) | m | total depth above which the column is pure hybrid; > 0 needed for H ≤ 0 cells (`:1169-1172`) |
| `keep_isopycnal` | hybrid | no | True | — | interfaces between layers already in their density brackets stay put (White et al. 2009); False = MOM6 HYCOM1 (P1M everywhere) |
| `dsigma` | sigma; hybrid with `d_hybrid > 0` | auto | the model's `dsigma` | — | σ fractions, bed first |
| `h_min` | all | no | 1e-3 (`H_MIN`, `core/vertical/regrid.py:42`) | m | minimum layer thickness, applied as min(h_min, D/N) |

## `SigmaWetDry` — the homogeneous σ core

``SigmaWetDry.__init__`` (`core/dynamics/sigma_wd.py:161`). `SigmaBaroclinic` builds one as `m.core`, passing
`n_layers, z0, closure, nu_const, theta, g, d_min, dsigma, solver` and Flather masks on the open edges
(`core/dynamics/sigma_baroclinic.py:1078-1080`); `nu_min` and `cd_max` stay at their defaults. It can also be run
alone as the homogeneous 3-D wetting–drying model (CROCO 3-D TIDAL_FLAT to 1.2 cm, CLAUDE.md).

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `h_grid` | `xr.Dataset` | required | — | horizontal grid | | |
| `bathy` | array | required | m | as `SigmaBaroclinic` | | |
| `dt` | float | required | s | time step | | |
| `n_layers` | int | 10 | — | σ-layers | | |
| `z0` | float or None | **1e-4** | m | bed roughness (None: no bed stress) | | different default from `SigmaBaroclinic` |
| `closure` | str | **`"parabolic"`** | — | `"gls"`, `"parabolic"`, `"constant"`, `"none"` | (`:166-167`) | |
| `nu_const` | float | **1e-2** | m² s⁻¹ | viscosity of `"constant"` | | |
| `nu_min` | float | 1e-5 | m² s⁻¹ | background of the parabolic profile | | |
| `theta` | float | 0.6 | — | free-surface weight | [0.5, 1] | |
| `g` | float | 9.81 | m s⁻² | | | |
| `d_min` | float | 0.05 | m | dry-face threshold | | |
| `flather_u`, `flather_v` | bool arrays or None | None | — | domain-edge faces with Flather | edge faces only | |
| `solver` | str | `"direct"` | — | `"direct"`, `"cg"` | | |
| `cd_max` | float | 0.1 | — | upper bound of the log-law C_d | | |
| `dsigma` | array or None | None | — | σ fractions, bed first | positive, Σ = 1 (`:175-176`) | |

``SigmaWetDry.step`` (`:258`) takes `(eta, u, v, G_u, G_v, flather_eta, flather_ubar, flather_vbar, tau_u, tau_v,
wu, wv, hu, hv, hu_drag, hv_drag, hu_mean, hv_mean)`; τ here is **kinematic** (τ/ρ₀, m² s⁻²).

## `WetDryFreeSurface` — the 2-D wet–dry free surface

``WetDryFreeSurface.__init__`` (`core/dynamics/wetting_drying.py:90`): Casulli (1990) θ-method with the upwind total
depth at the faces; the matrix changes every step. It is the only class in `core/dynamics/wetting_drying.py`; the
barotropic tidal runs built on it are described in [Tides and the 2-D tidal model](tidal-model.md).

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `h_grid` | `xr.Dataset` | required | — | grid (`dx_u, dy_u, dx_v, dy_v, area_t`) | | |
| `bathy` | array | required | m | H > 0 water, < 0 floodable, NaN wall | | |
| `dt` | float | required | s | | | |
| `theta` | float | 0.6 | — | implicit weight | [0.5, 1] (`:94-95`) | |
| `g` | float | 9.81 | m s⁻² | | | |
| `d_min` | float | 0.05 | m | dry-face threshold | | |
| `flather_u`, `flather_v` | bool arrays | None | — | Flather faces | domain edge only (`:139`) | closed while the adjacent cell is dry |
| `solver` | str | `"direct"` | — | `"direct"` (spsolve each step) or `"cg"` | (`:96-97`) | `"cg"` recomputes η from the final transports (MITgcm exactConserv) |
| `cg_rtol` | float | 1e-10 | — | relative CG tolerance | | read at every step: `m.core.fs.cg_rtol = …` works after construction |
| `precond` | str | `"lu"` | — | `"lu"` (LU of a lagged matrix) or `"jacobi"` | (`:98-99`) | 1–3 vs 97–117 iterations on the northern Gulf of California nest (docstring) |
| `refactor_iterations` | int | 5 | — | refactorise the lagged LU when CG needs more iterations | | |

`step(eta, u_n, v_n, u_hat, v_hat, flather_eta=None, flather_ubar=None, flather_vbar=None)` (`:227`); diagnostics
after a step: `last_clip_volume`, `cg_iterations`, `lu_factorizations`, `last_wet_u/v`, `last_Du/Dv`.

## `ImplicitFreeSurface` — the linear free surface of the z-level and tidal models

``ImplicitFreeSurface.__init__`` (`core/dynamics/free_surface.py:143`): MITgcm θ-method on fixed depth, matrix
factorised once. Used by the 2-D tidal runs (PAC_MEX) and the phase-1 z-level model, not by `SigmaBaroclinic`.

| Name | Type | Default | Units | Meaning | Valid values | Notes |
|---|---|---|---|---|---|---|
| `h_grid` | `xr.Dataset` | required | — | horizontal grid | | |
| `v_grid` | `xr.Dataset` | required | — | z-level grid (`dz`) | ``create_z_grid`` etc. (`core/grid/vertical.py:63`) | |
| `hfac` | `xr.Dataset` | required | — | partial-cell fractions | ``compute_hfac(z_grid, depth, hfac_min=0.1)`` (`core/grid/vertical.py:222`) | open edges with ``open_boundary_hfac(hfac, sides)`` (`:74`) |
| `dt` | float | required | s | | | |
| `theta` | float | **1.0** | — | implicit weight (MITgcm default) | [0.5, 1] (`:159-160`) | |
| `g` | float | 9.81 | m s⁻² | | | |
| `open_u`, `open_v` | bool arrays | None | — | faces with prescribed normal velocity (taken from û) | | |
| `flather_u`, `flather_v` | bool arrays | None | — | Flather faces | edge, wet (`:205-209`) | |
| `sal_beta` | float | 0.0 | — | scalar self-attraction and loading: g(1 − β)∇η | [0, 1) (`:163-164`) | HYCOM `tidsal` |
| `solver` | str | `"direct"` | — | `"direct"` (LU once) or `"cg"` (Jacobi) | | |
| `cg_rtol` | float | 1e-12 | — | CG tolerance | | |

`step(eta, u_n, v_n, u_hat, v_hat, flather_eta=None, flather_ubar=None, flather_vbar=None, eta_tendency=None)` (`:284`).

## Grid builders

| Function | Arguments (defaults) | Notes |
|---|---|---|
| ``create_cartesian_grid`` (`core/grid/horizontal.py:48`) | `nx, ny, dx, dy, x0=0.0, y0=0.0, f0=1e-4, beta=0.0` | f = f₀ + βy; attrs `dx`, `dy` |
| ``create_cartesian_grid_stretched`` (`:163`) | `dx (nx,), dy (ny,), x0=0.0, y0=0.0, f0=1e-4, beta=0.0` | dx_u = distance between centres (MITgcm dxC); no scalar `dx` attribute; raises on non-positive spacing (`:208-209`) |
| ``create_spherical_grid`` (`:246`) | `nx, ny, dlon, dlat, lon0=0.0, lat0=0.0` | f = 2Ω sin φ; adds `lon_*`, `lat_*` (CF lon/lat in the output) |
| ``set_land_mask`` (`:378`) | `grid, mask` | sets `mask_t`; **not read by the σ/ALE model** (use NaN in `bathy`) |
| ``stretched_sigma`` (`core/grid/sigma_levels.py:33`) | `n, theta_s, theta_b=0.0` | Song & Haidvogel (1994) fractions, bed first; θ_b ∈ [0, 1] |
| ``sigma_from_thicknesses`` (`:47`) | `dz_top_first` | fractions proportional to z-level thicknesses listed surface first |
| ``target_densities`` (`core/vertical/regrid.py:292`) | `rho_top, n_layers, n_light=0, d_rho_light=0.01` | hybrid targets; 0 ≤ n_light < n_layers |
| ``build_bathymetry`` (`core/grid/bathymetry.py:379`) | `h_grid, source_path, min_depth=10.0, n_smooth_passes=2, n_skip=1, kernel="hann", method="linear"` | **returns 0 for land**: convert to NaN for the σ/ALE models ([Regional setup](regional-setup.md)) |

## Output, restart and diagnostics

| Object | Arguments (defaults) | Notes |
|---|---|---|
| ``OutputWriter`` (`diagnostics/output.py:49`) | `path, h_grid, H, n_layers=0, start="2000-01-01T00:00:00", fmt="zarr", title="pyOCEAN output", attrs=None, dsigma=None` | `fmt` `"zarr"` (append per record) or `"netcdf"` (buffered until `close`); `write(time, **fields)` with fields `eta, u, v, T, S, ubar, vbar, mix_num` (`:35-44`) |
| ``write_restart`` (`core/restart.py:89`) | `path, model, state` | float64 NetCDF4 |
| ``read_restart`` (`core/restart.py:134`) | `path, model` → `SigmaState` | checks the configuration (`_config`, `:60`) |
| ``BlowupGuard`` (`diagnostics/postmortem.py:35`) | `limits=None, dump_path=None, lat=None, lon=None, depth=None, dt=None, dx=None, history=5, velocity_fields=("u", "v")` | `check(step, time, **fields)` raises `BlowupError` |
| ``check_domain`` (`diagnostics/domain_check.py:178`) | `h_grid, H, dt, u_max=1.0, sigma_layers=None, implicit_free_surface=True, gprime_h=None, cfl_target=0.5, rx0_max=0.2, rx1_max=6.0, g=9.81` | returns `DomainReport` (`.ok`, `.summary()`, `.dt_max_advective`) |
| ``reference_potential_energy`` (`diagnostics/mixing.py:52`) | `rho, volume, area=None, hypsometry=None, g=9.81` | give exactly one of `area`, `hypsometry` |
| ``numerical_variance_decay`` (`diagnostics/mixing.py:87`) | `advect, c, dt` | offline M_num of one advection operator |
| legacy ``write_snapshot`` / ``read_snapshot`` / ``DiagnosticsWriter`` (`diagnostics/io.py:79,119,165`) | phase-1 `ModelState` only | k = 0 at the **surface** there |

Restart configuration keys compared on reading (`core/restart.py:60-86`): `dt, n_layers, theta, d_min, sigma_w, pgf,
rho_ref` (none / auto / callable), `open_boundaries, sponge_cells, sponge_tau, orlanski_cmax, sponge_tracer, kh, kv,
closure, z0, nu_const, adv, tracer_scheme, tracer_time, rho0, t_coef, t_ref, ah, smagorinsky, theta_bc, ny, nx`; in ALE also
`regrid, remap, thickness_flux, momentum, ke_scheme, h_thin_bed, max_truncations, thin_layer_coupling,
thin_layer_cutoff, tracer_limiter, mix_diagnostic, obc_tau_in, obc_tau_out, obc_mode, obc_state` (function or none)
and every `regrid_params` entry. **Not compared:** `solver`, `g`, `init_iterations`, the content of
`open_boundary_state` or of a user callable `rho_ref`.

## Errors raised for invalid configurations

Grouped by where they are raised. The message text is the code's.

### At construction of `SigmaBaroclinic`

| Message (abridged) | Cause | Line |
|---|---|---|
| `unknown pressure-gradient scheme …` | `pgf` not in jacobian/twoterm/fv | `core/dynamics/sigma_baroclinic.py:1022` |
| `unknown tracer time stepping …` | `tracer_time` | `:1025` |
| `unknown tracer scheme …` | `tracer_scheme` | `:1028` |
| `unknown open boundaries […]; use ('west', 'east', 'south', 'north')` | misspelt edge | `:1032` |
| `a sponge needs open boundaries` | `sponge_cells > 0` with no open edge | `:1034` |
| `open_boundary_state needs open_boundaries` | | `:1037` |
| `open_boundary_mode {…}: edges must be open and modes 'radiating' or 'clamped'` | | `:1041` |
| `a 'clamped' open_boundary_mode needs open_boundary_state (the exterior to impose)` | | `:1045` |
| `unknown thin_layer_coupling …` / `thin_layer_cutoff must be ≥ 0 m` | | `:1055`, `:1058` |
| `unknown closure …` / `dsigma must be N positive fractions summing to 1` | from `SigmaWetDry` | `core/dynamics/sigma_wd.py:167`, `:176` |
| `theta must be in [0.5, 1]` / `solver must be 'direct' or 'cg'` | from `WetDryFreeSurface` | `core/dynamics/wetting_drying.py:95`, `:97` |
| `unknown vertical_coordinate …` | | `core/dynamics/sigma_baroclinic.py:1128` |
| `unknown momentum …` | | `:1132` |
| `momentum='vector_invariant' needs vertical_coordinate='ale' (the σ model keeps the flux form, which matches CROCO)` | | `:1134` |
| `unknown ke_scheme …` / `unknown thickness_flux …` | | `:1137`, `:1140` |
| `smagorinsky=…: the Smagorinsky coefficient must be a finite number ≥ 0` | negative or non-finite `smagorinsky` | `core/model/preflight.py:64-67` |
| `unknown regrid rule …` / `unknown remap_scheme …` | ALE only | `:1145`, `:1147` |
| *(removed in N3)* `vertical_coordinate='ale' needs z0=None ...` — ALE accepts `z0` now; `z0` together with `bottom_drag='quadratic'` is an error (two bed drags) | | `core/model/preflight.py` |
| `vertical_coordinate='ale' needs closure 'constant' or 'none' for now (GLS and the parabolic ν assume σ-layers)` | | `:1152` |
| `regrid='zstar' needs regrid_params['dz_top']` (and `'rho_target'` for hybrid) | | `:1158` |
| `cells with H ≤ 0 can dry: use regrid='sigma', or 'hybrid' with a shallow σ zone (regrid_params d_sigma, d_hybrid > 0) covering them` | floodable cells with z*, or hybrid without a window — also the symptom of 0-for-land bathymetry | `:1171` |
| `open boundaries next to cells with H ≤ 0 (which can dry) are not supported with vertical_coordinate='ale'` | | `:1178` |
| `unknown tracer_limiter …` / `tracer_limiter='fct' needs vertical_coordinate='ale', tracer_time='ssprk3' and a WENO5 tracer_scheme` | | `:1116`, `:1119` |
| **UserWarning** `hybrid z* floor (Σ dz_top = … m) stops above the deepest bed (… m)` | Σ `dz_top` < max H | `:1165` |

### In `initial_state` and the first steps

| Message | Cause | Line |
|---|---|---|
| `initial_state(h_src=...) needs vertical_coordinate='ale'` | | `:1231` |
| `h_src and T must both be (N_src, ny, nx)` | shapes | `:1234` |
| `h_src must be ≥ 0 and fill every wet column: Σ h_src = H + η` | data levels not clipped at the bed, or η ≠ 0 not included (rtol 1e-10, atol 1e-9 m) | `:1239` |
| `regrid_columns(…) needs params[…]` / `wet columns need finite H and D = H + eta > 0` / `z* needs H > 0 in every wet column` / `hybrid needs H > 0 unless it has a shallow σ zone` / `non-finite density in a wet column` | regrid inputs (NaN T in a wet cell is the usual cause) | `core/vertical/regrid.py:364-381` |
| `ALE open boundaries need the exterior profile T_ext and h_ext, set together and with the layer shape …` | `T_ext`/`h_ext` replaced by hand inconsistently, or the model stepped without `initial_state`/`read_restart` | `core/dynamics/sigma_baroclinic.py:1765` |
| `open_boundary_state edge … is not in open_boundaries` / `OBData needs zw, T and u_normal` / `shapes T …, u_normal …, zw …` / `zw must decrease from the surface down` / `eta shape …` | `open_boundary_state` output | `core/dynamics/open_boundary.py:24-37` |
| `…: exterior data do not reach the bed of the boundary column (zw[-1] > −H)` | exterior profile too shallow | `core/dynamics/open_boundary.py:73`, `:110` |
| `unknown advection scheme … (use 'centred' or 'upwind3')` | bad `adv` (only checked here) | `core/dynamics/nonlinear.py:162` |

### During the run (`RuntimeError`)

| Message | Meaning | Line |
|---|---|---|
| `N CFL velocity truncations exceed max_truncations=…; first this step: layer k at u-face (j, i): u = …, face thickness … m` | ALE velocity backstop acted too often | `core/dynamics/sigma_baroclinic.py:2078` |
| `layer k of cell (j, i) emptied in the Lagrangian step: h† = … m (reduce dt)` | outflow CFL | `core/vertical/ale.py:292` |
| `limit_outflow: layer k of cell (j, i) would reach h = … m … — barotropic CFL` / `face … must carry … out of a donor column that holds no water` | the limiter cannot keep h ≥ 0 | `core/vertical/ale.py:390`, `:432` |
| **RuntimeWarning** `limit_outflow: … still sends out more than it holds after 50 passes, but its inflow keeps h ≥ 0` | FCT bound weakened locally; run continues | `core/vertical/ale.py:385` |
| `wet-dry free-surface CG did not converge (info=…)` | CG failure, usually after a blow-up | `core/dynamics/wetting_drying.py:292` |

### Output and restart

| Message | Line |
|---|---|
| `fmt must be 'zarr' or 'netcdf'` / `unknown output field …; known: [...]` / `…: shape … does not match …` | `diagnostics/output.py:60`, `:114`, `:120` |
| `restart written by …, not SigmaBaroclinic` / `restart written by an ALE model …, not a σ model` | `core/restart.py:139`, `:141` |
| `restart incompatible with the model configuration — key: restart … vs model …; …` | `core/restart.py:152` |
| `restart has no GLS state …` / `… no external tracer …` / `… no reference density profile …` | `core/restart.py:161`, `:166`, `:181` |
| `give exactly one of area= or hypsometry=` / `hypsometry holds less volume than the water` | `diagnostics/mixing.py:64`, `:81` |

## References

- Adcroft, A., Hallberg, R. & Harrison, M. (2008). A finite volume discretization of the pressure gradient force
    using analytic integration. *Ocean Modell.* 22, 106–113. doi:10.1016/j.ocemod.2008.02.001 **[verify]**
- Casulli, V. (1990). Semi-implicit finite difference methods for the two-dimensional shallow water equations.
    *J. Comput. Phys.* 86, 56–74.
- Ilicak, M., Adcroft, A.J., Griffies, S.M. & Hallberg, R.W. (2012). *Ocean Modell.* 45–46, 37–58.
    doi:10.1016/j.ocemod.2011.10.003
- Song, Y.T. (1998). A general pressure gradient formulation for ocean models. Part I. *Mon. Weather Rev.* 126,
    3213–3230. **[verify]**
- Song, Y. & Haidvogel, D. (1994). *J. Comput. Phys.* 115, 228–244.
- White, L., Adcroft, A. & Hallberg, R. (2009). High-order regridding–remapping schemes for continuous isopycnal and
    generalized coordinates in ocean models. *J. Comput. Phys.* 228, 8665–8692.
- Zalesak, S.T. (1979). Fully multidimensional flux-corrected transport algorithms for fluids. *J. Comput. Phys.* 31,
    335–362. doi:10.1016/0021-9991(79)90051-2
