# MOM6 biharmonic Smagorinsky viscosity (BIHARMONIC, SMAGORINSKY_AH, BOUND_AH, OBC_ZERO_BIHARMONIC)

Reading of MOM6 git 0e9a561 (`~/pyOCEAN_refs/MOM6/src`) for the regional chain, step 4 of the Gulf Stream fix
sequence (approved by Esteban). File paths are relative to `src/`; `F:` is
`parameterizations/lateral/MOM_hor_visc.F90`. Indices are MOM6's: cells `(i, j)`, u faces `(I, j)` east of cell
`i`, v faces `(i, J)` north of cell `j`, corners `(I, J)`. pyOCEAN's u face `I` lies **west** of cell `I`
(between cells `I−1` and `I`) and its corner `[J, I]` is shared by cells `(J−1, J) × (I−1, I)`, so MOM6 `i+1 ↔`
pyOCEAN cell `I` for a face, and MOM6 `j+1 ↔` pyOCEAN row `J` for a v face. The earlier Laplacian reading is
`docs/design/smag_refs/mom6_hor_visc.md`; the open-boundary corner treatment is
`docs/design/regional_refs/mom6_obc_tangential.md`.

## 0. NWA12 settings

From the MOM6 run itself (`/Volumes/ESTEBAN_MIT/pyOCEAN_runs/nwa12_cut/runs/mom6_full/MOM_parameter_doc.all`, lines
in brackets) and `validation/nwa12_cut/MOM_input:347-358`, `:86`:

| parameter | NWA12 | default | meaning |
|---|---|---|---|
| `LAPLACIAN` | False [1566] | False | no Laplacian viscosity |
| `BIHARMONIC` | True [1568] | True | biharmonic viscosity |
| `AH` | 0.0 [1571] | 0 | constant background [m⁴ s⁻¹] |
| `AH_VEL_SCALE` | 0.01 [1573] | 0 | background `AH_VEL_SCALE Δ³` [m s⁻¹] |
| `AH_TIME_SCALE` | 0.0 [1577] | 0 | unused |
| `SMAGORINSKY_AH` | True [1581] | False | biharmonic Smagorinsky |
| `LEITH_AH`, `USE_LEITHY` | False [1583, 1585] | False | — |
| `BOUND_AH` | True [1588] | True | stability bound |
| `RE_AH` | 0 [1590] | 0 | — |
| `SMAG_BI_CONST` | 0.03 [1593] | 0 | Smagorinsky constant |
| `BOUND_CORIOLIS_BIHARM` | False [1595] | `BOUND_CORIOLIS` (False [1341]) | no quadratic Coriolis term |
| `USE_LAND_MASK_FOR_HVISC` | True [1601] | True | land-masked face thickness |
| `HORVISC_BOUND_COEF` | 0.8 [1604] | 0.8 | bound coefficient |
| `USE_CONT_THICKNESS` | False [1563] | False | face thickness from h, not the continuity solver |
| `NOSLIP` | False [1320] | False | free slip (no slip + biharmonic is FATAL, F:3205-3207) |
| `OBC_ZERO_BIHARMONIC` | True [311] | False | Del2 = 0 on open faces |
| `OBC_COMPUTED_STRAIN` | True (`MOM_input:84`) | False | computed D_S at open-edge corners |
| `DT` | 600 [71] | — | the Δt of the bound (F:3194-3198) |

No MEKE, no backscatter (`EY24_EBT_BS`), no GME, no anisotropic viscosity, `HORVISC_NKBLOCK` irrelevant.

## 1. Parameters and static fields (`hor_visc_init`)

- `BIHARMONIC` F:3002-3005, `AH` F:3006-3009, `AH_VEL_SCALE` F:3010-3015, `AH_TIME_SCALE` F:3016-3021,
  `SMAGORINSKY_AH` F:3022-3025, `LEITH_AH` F:3026-3029, `BOUND_AH` F:3038-3041, `SMAG_BI_CONST` F:3056-3059,
  `BOUND_CORIOLIS_BIHARM` F:3082-3091, `USE_LAND_MASK_FOR_HVISC` F:3105-3108, `HORVISC_BOUND_COEF` F:3109-3112,
  `DT` (`Idt = 1/dt`) F:3194-3198.
- Metric products (F:3428-3436): `Idx2dyCu = IdxCu² IdyCu`, `Idxdy2u = IdxCu IdyCu²`, `Idx2dyCv = IdxCv² IdyCv`,
  `Idxdy2v = IdxCv IdyCv²`; `dx2h = dxT²`, `dy2h = dyT²` (F:3340), `dx2q = dxBu²`, `dy2q = dyBu²` (F:3319).
- h points (F:3444-3473): `grid_sp_h2 = 2 dx2h dy2h/(dx2h + dy2h)`, `grid_sp_h3 = grid_sp_h2^{3/2}`,
  `Biharm_const_xx = SMAG_BI_CONST grid_sp_h2²` (F:3450), `Ah_bg_xx = max(AH, AH_VEL_SCALE grid_sp_h3)` (F:3468; the
  `AH_TIME_SCALE` branch F:3470-3471 is off). q points the same with the corner spacings (F:3488-3505).
- `Ah_Limit = 0.3/(64 dt)` (F:3440) is computed and not used by the bound below (legacy).
- **Bound** (F:3534-3590, "should avoid overshoots when bound_coef < 0.5, but empirically work for ≲ 1.0"): the
  auxiliary sums at faces
  - `u0u(I,j) = Idxdy2u (dy2h(i+1) DY_dxT(i+1) (IdyCu(I+1)+IdyCu(I)) + dy2h(i) DY_dxT(i) (IdyCu(I)+IdyCu(I−1)))`
    `+ Idx2dyCu (dx2q(J) DX_dyBu(J) (IdxCu(j+1)+IdxCu(j)) + dx2q(J−1) DX_dyBu(J−1) (IdxCu(j)+IdxCu(j−1)))` (F:3538-3541),
  - `u0v` (F:3542-3545), `v0u` (F:3548-3551), `v0v` (F:3552-3555) analogously,
  - `denom_h = max( dy2h (DY_dxT (IdyCu u0u)_{e+w} + DX_dyT (IdxCv v0u)_{n+s}) max(IdyCu IareaCu)_{e,w},`
    `dx2h (DY_dxT (IdyCu u0v)_{e+w} + DX_dyT (IdxCv v0v)_{n+s}) max(IdxCv IareaCv)_{n,s} )` (F:3557-3566),
    `Ah_Max_xx = bound_coef · 0.5 · Idt / denom` (F:3567-3569); q points F:3576-3588.
  - On a uniform grid every sum is 8/Δ², `denom = 32/Δ⁴`, `Ah_Max = 0.8 Δ⁴/(64 Δt)` [derived]: the 2Δ checkerboard
    has λ = 64 A_h/Δ⁴, so the bound is λΔt ≤ bound_coef — the same design target as the Laplacian bound
    (λΔt ≤ 0.8 with `Kh_Max = 0.8·0.25/(Δt denom)`, F:3510-3532).
  - The sums reach one metric point beyond the cell (`IdyCu(I+1)` of the east face, `dx2q(J−1)` …): MOM6 reads its
    halo metrics.

## 2. Per-step operator (`horizontal_viscosity`)

Per layer `k` (block of layers `kk`), along the layer.

1. **Strains** (as the Laplacian): tension `sh_xx = dudx − dvdy` at h, F:707-715, no mask (u, v are 0 on land faces);
   `dvdx`, `dudy` at q, F:718-722.
2. **Face thicknesses** (`USE_LAND_MASK_FOR_HVISC`): `h_u = ½(mask2dT(i) h(i) + mask2dT(i+1) h(i+1))`, F:769-779
   (land faces carry half the wet cell's h). Open segments: `h_u = h` of the interior cell, F:857-890; cross-corner
   projection F:891-916.
3. **Open-edge strain** (`OBC_STRAIN_COMPUTED`, F:784-856): see `mom6_obc_tangential.md` §4.2.
4. **Shear strain**, free slip: `sh_xy = mask2dBu (dvdx + dudy)`, F:924-929.
5. **Del2** (F:946-971), at every face including land faces and one halo row/column:
   `Del2u(I,j) = Idx2dyCu (dx2q(J) sh_xy(J) − dx2q(J−1) sh_xy(J−1)) + Idxdy2u (dy2h(i+1) sh_xx(i+1) − dy2h(i) sh_xx(i))`
   (F:948-950), `Del2v(i,J) = Idxdy2v (dy2q(I) sh_xy(I) − dy2q(I−1) sh_xy(I−1)) − Idx2dyCv (dx2h(j+1) sh_xx(j+1) − dx2h(j) sh_xx(j))`
   (F:952-954). **Not masked at land faces**: a coastal face gets `±Idxdy2u dy2h sh_xx(wet cell)` (the land cell's
   tension is 0). This is ∇²u in the "stress-tensor" form (div of the strain tensor without h).
   `OBC_ZERO_BIHARMONIC` (F:956-970): `Del2u = 0` on the faces of E/W segments, `Del2v = 0` on N/S segments
   (only the normal component on the segment face).
6. **|S|** at h (F:1017-1023) and q (F:1353-1359), as the Laplacian; `hq` harmonic (F:1362-1367); `hrat_min` at h
   from the four `h_u`, `h_v` (land faces included, at half thickness) over `h + h_neglect` (F:1026-1037) and at q
   from the four faces over `hq` (F:1369-1374).
7. **A_h at h** (F:1187-1256): `Ah = Ah_bg_xx`; `AhSm = Biharm_const_xx Shear_mag`, `Ah = max(Ah, AhSm)` (F:1203-1207;
   the `bound_Coriolis` branch F:1196-1201 is off); `BOUND_AH` without `BOUND_KH`: `Ah = min(Ah, hrat_min Ah_Max_xx)`
   (F:1253-1256; with both Laplacian and biharmonic bounded, the biharmonic gets `visc_bound_rem` of the budget,
   F:1121-1133, 1249-1252 — not used in NWA12).
8. **str_xx** (F:1299-1310): `d_del2u = IdyCu(I) Del2u(I) − IdyCu(I−1) Del2u(I−1)`,
   `d_del2v = IdxCv(J) Del2v(J) − IdxCv(J−1) Del2v(J−1)`, `str_xx += Ah (DY_dxT d_del2u − DX_dyT d_del2v)`.
   With `LAPLACIAN` False `str_xx` starts at 0.
9. **Gradient of Del2 at q** (F:1314-1351): `dDel2vdx = DY_dxBu (Del2v(i+1) IdyCv(i+1) − Del2v(i) IdyCv(i))`,
   `dDel2udy = DX_dyBu (Del2u(j+1) IdxCu(j+1) − Del2u(j) IdxCu(j))` (F:1317-1318). Open segments: with
   `OBC_STRAIN_ZERO` both are 0, with `OBC_STRAIN_FREESLIP` the cross-boundary one is 0 (F:1328-1349); with
   `OBC_STRAIN_COMPUTED` (NWA12) nothing is changed, so the cross-boundary derivative reads the halo `Del2v`
   beyond the segment (built from halo velocities and metrics).
10. **A_h at q** (F:1581-1640) as at h; `str_xy += Ah (dDel2vdx + dDel2udy)` (F:1669-1672).
11. **Thickness weighting** (F:1736-1752): `str_xx *= h`, `str_xy *= hq · mask2dBu` (free slip). `mask2dBu = 1` at
    the open-edge corners between two open wet faces (`mom6_obc_tangential.md` §4.1).
12. **Divergence** (F:1754-1796):
    `diffu = (IdxCu (dx2q(J−1) str_xy(J−1) − dx2q(J) str_xy(J)) + IdyCu (dy2h(i) str_xx(i) − dy2h(i+1) str_xx(i+1))) IareaCu / (h_u + h_neglect)`
    and the v analogue; `diffu = 0` on E/W segment faces, `diffv = 0` on N/S segment faces (F:1762-1773, 1784-1795).
    With constant `Ah` and `h` the result is `−Ah ∇⁴u` [derived]: the sign convention is `str = −(stress)`.
13. **Time level** (`core/MOM_dynamics_split_RK2.F90:892-897`): `diffu` is computed once per baroclinic step from the
    predictor-average `u_av, v_av, h_av`, used in the corrector (`:915`) and in the next step's predictor (`:577`).

## 3. What pyOCEAN implements (`core/dynamics/hor_visc.py`, `biharmonic` / `biharmonic_numpy`)

`SigmaBaroclinic(..., smagorinsky_bi=SMAG_BI_CONST, ah_vel_scale=AH_VEL_SCALE, ah_bi=AH)`, σ and ALE, through
`Momentum.smagorinsky_terms(..., bih=True)`; term "hviscosity"; `Momentum.last_ah_t`, registry `ah_bi`,
`SigmaState.n_visc_limited`; restart config `smagorinsky_bi`, `ah_vel_scale`, `ah_bi`.

Line by line as §2 items 1, 4-12, with these choices:

| item | MOM6 | pyOCEAN | class |
|---|---|---|---|
| bound | `0.8 · 0.5/(Δt denom)` (λΔt ≤ 0.8, forward-step design) | `× 6/11` for the AB3 step (λΔt ≤ 0.8 · 6/11, AB3's real-axis interval), as the Laplacian (`_bound_factor`) | c |
| `denom` beyond the domain edge | halo metrics | edge metrics repeated outward (`biharmonic_geometry`) | c (edge cells only) |
| `hrat_min` at h | four faces, land faces at ½h (→ ≤ ½ at the coast) | wet faces only, as pyOCEAN's Laplacian (a land face gets no tendency) | c |
| face thickness | `USE_LAND_MASK_FOR_HVISC` ½(h_L + h_R), open faces from the interior | the same (`Momentum._smag_thickness`, `_open_face_thickness`) | — |
| Del2 at land faces | not masked | not masked (D_T = 0 outside the domain) | — |
| Del2 on open faces | `OBC_ZERO_BIHARMONIC` (NWA12) | always (no switch) | — for NWA12 |
| str_xy at open-edge corners | `ZERO`: 0; `COMPUTED`: cross-boundary δ(Del2) from the halo | 0 in both modes: with `obc_tangential='mom6'` the exterior Del2 is mirrored (zero gradient) so the cross term is 0, and the along-edge term is a difference of two open-face Del2 = 0; this is MOM6's `OBC_STRAIN_FREESLIP` result for the biharmonic part. The computed D_S of those corners still enters |S| and Del2 of the column/row next to the edge | c |
| combined Laplacian + biharmonic | `visc_bound_rem` | refused by the pre-flight | not ported |
| `BOUND_CORIOLIS_BIHARM`, Leith, MEKE, `RE_AH`, `AH_TIME_SCALE` | available | not ported (off in NWA12) | — |
| time level | lagged `u_av` of the predictor | uⁿ inside AB3, as every explicit term | c |

Consequence of `OBC_ZERO_BIHARMONIC` (both models): an x-uniform jet through an open edge feels a biharmonic
tendency at the first interior face (`d_del2u` of the boundary cell has one Del2u = 0), unlike the Laplacian, whose
edge tendency equals the interior one with the open-face velocity in the tension.

Energy: for constant `A_h h` the operator is `−A_h h ‖∇²u‖²`-dissipative (summation by parts,
`tests/test_hor_visc_biharmonic.py::test_constant_ah_dissipates_energy`). With variable `A_h` (Smagorinsky) MOM6's
form `∇·(A_h ∇(∇²u))` is not sign-definite in general (Griffies & Hallberg 2000 build the dissipative form with
√A_h on both Laplacians); random smooth fields dissipate (test), and pyOCEAN keeps MOM6's form.

## 4. Compiled-MOM6 comparison: not done (why)

The column drivers (`validation/mom6_column/*_driver.F90`) call pure routines (`int_density_dz_generic_plm`,
`kappa_shear` column kernels, `find_N2`). `horizontal_viscosity` needs an initialised `hor_visc_CS` from
`hor_visc_init(Time, G, GV, US, param_file, diag, CS)` — a `param_file_type` read from files, a `diag_ctrl` with
registered diagnostics, a full `ocean_grid_type` with an MPP domain and halos (`MOM_domains`), plus `MEKE_type`,
`VarMix_CS`, `ocean_OBC_type` with segments and the `thermo_var_ptrs`/`barotropic_CS` arguments. Building that
outside the model is a small MOM6 driver of its own (several hours with the FMS initialisation), so the port is
line by line with the formulas above, the analytic tests (discrete `−A_h (4 sin²(kΔ/2)/Δ²)²` decay to 1e-10,
`Ah_Max` on a uniform grid = `0.8·(6/11) Δ⁴/(64Δt)` to 1e-13, `Ah` = `max(AH, AH_VEL_SCALE Δ³, C Δ⁴|S|)` to
1e-15), and the NWA12 runs against compiled MOM6 (the full model).
