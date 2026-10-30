# CVMix shear mixing (PP81, LMD94 interior) as MOM6 calls it — reference read

Sources: MOM6 0e9a561 (`~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_CVMix_shear.F90`, 374 lines) and
CVMix fce4221 (`~/pyOCEAN_refs/MOM6/pkg/CVMix-src/src/shared/cvmix_shear.F90`). Citations `file:line`.

## 1. Ri computation in MOM6 (`MOM_CVMix_shear.F90:64-218`)

Per column, at the h point (u, v already averaged to h points by `find_uv_at_h`):
- Density of layers k and k−1 **both at the pressure of interface k** (top of layer k): arrays of 2·nz values
  (`:117-137`); pRef accumulates ρ₀ g h (`:132`).
- dRho = (g/ρ₀)(ρ_k − ρ_{k−1}) (Boussinesq, `:145-146`); dz_int = ½(dz_{k−1} + dz_k) + dZ_subroundoff (`:150`).
- N² = dRho/dz_int; S² = (Δu² + Δv²)/dz_int² (`:151-152`).
- Ri_g(k) = max(0, N²) / max(S², 1e-10 s⁻²) (`:153`). Negative N² → Ri = 0 (full shear mixing, no separate
  convection here). Ri at the bottom interface copies the one above (`:161`).
- **Surface interface k = 1**: km1 = max(1, k−1) = 1, so ΔU = 0 and dRho = 0 → Ri = 0/1e-10 = 0 → the
  maximum coefficient at the surface interface (`:121`, `:140-153`). Harmless if the surface value is unused
  (`MOM_set_diffusivity.F90:576` says Kd(1) "isn't actually used") [verificar for Kv in vert_friction].
- Optional smoothing `N_SMOOTH_RI` (default 0, `:303-306`): vanished layers (h ≤ 1e-10 m) first take the Ri of
  the interface above (`:168-170`), then N passes of a 1-2-1 filter **in place, top-down** (`:174-184`); the
  loop starts at k = 3, so interface 2 is never smoothed, and the filter uses the already-smoothed value above
  (recursive, not a pure 1-2-1).
- The scheme **overwrites** kd, kv (intent out, `:73-76`); MOM6 then adds them to the background in
  `set_diffusivity` (`MOM_set_diffusivity.F90:571-581`), as for kappa-shear. Only one shear scheme may be on
  (`:269-273`).

## 2. Formulas (`cvmix_shear.F90`)

Documented `:126-156`, coded in `cvmix_coeffs_shear_low` `:380-434`.

**PP81** (`:384-402`):

    ν = ν₀ / (1 + α Ri)^n + ν_b,      κ = ν / (1 + α Ri) + κ_b       (Ri ≤ 0 treated as Ri = 0)

CVMix defaults (`:195-225`): ν₀ = PP_nu_zero = 0.01 m² s⁻¹ (PP81 used 5e-3), α = 5, n = 2, ν_b = κ_b = 0
(PP81 used 1e-4 / 1e-5). **MOM6 passes none of the PP parameters** (`MOM_CVMix_shear.F90:307-311`), so
`USE_PP81` always runs with the CVMix defaults 0.01, 5, 2, 0, 0.

**LMD94 interior** ('KPP', `:404-423`):

    κ = ν⁰                              Ri < 0
    κ = ν⁰ [1 − (Ri/Ri₀)²]^p₁           0 ≤ Ri < Ri₀
    κ = 0                               Ri ≥ Ri₀
    ν = Pr · κ

(the doc block writes "Ri/Ri₀ ^2" ambiguously; the code is (Ri/Ri₀)², `:416-417`). CVMix defaults (`:227-254`):
ν⁰ = 5e-3 m² s⁻¹, Ri₀ = 0.7, p₁ = 3, Pr = 1. MOM6 parameters (`MOM_CVMix_shear.F90:288-306`):

| MOM_input | default | note |
|-----------|---------|------|
| USE_LMD94 | False | `:255-257` |
| USE_PP81 | False | `:260-262` |
| NU_ZERO | 5e-3 m² s⁻¹ | `:288-290` |
| RI_ZERO | **0.8** | `:291-295` (CVMix 0.7, L94 0.7, CROCO 0.7, MITgcm 0.7) |
| PRANDTL_CVMIX_SHEAR | 1.0 | `:296-298` |
| KPP_EXP | 3.0 | `:299-302` |
| N_SMOOTH_RI | 0 | `:303-306` |

## 3. Comparison with kappa-shear (for the pyOCEAN decision)

- Local, algebraic, one pass per column: cost ~ a few array ops per interface; trivially vectorised in NumPy.
- No time integration: the coefficient is from the state at the start of the step. With large Δt the mixed
  column can overshoot (κ stays at ν⁰ for the whole step even after the shear is eroded) — the problem kappa-shear's
  time averaging addresses (`mom6_kappa_shear.md` §2).
- No boundary length scale: an interface next to the bed gets the full ν⁰ if Ri < Ri₀.
- Thin layers: MOM6's only guard is the smoothing branch (vanished layers copy Ri from above); without smoothing,
  a 1 mm layer gives Δz ≈ ½(h + 1 mm) and its own S², N². pyOCEAN should merge thin layers first, as for GLS/KPP.
- Grid-scale noise: L94 recommend vertical smoothing of Ri; CROCO and MITgcm both offer it (`croco_and_others.md`).
- Known in the literature: Ri-based schemes with ν⁰ = 5e-3 give too little entrainment for overflows compared
  with LES/Jackson [verificar; Legg et al. 2006/2008 overflow papers not in the local library].
