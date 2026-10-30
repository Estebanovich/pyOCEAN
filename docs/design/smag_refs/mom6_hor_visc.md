# MOM6 horizontal viscosity (Laplacian / biharmonic Smagorinsky) - reference read

Source: MOM6 at `~/pyOCEAN_refs/MOM6/src/` (local checkout, no version tag read).
`F` = `src/parameterizations/lateral/MOM_hor_visc.F90` (4374 lines). All `file:line` refer to this checkout.
Examples: `~/pyOCEAN_refs/MOM6-examples/ocean_only/`. Items marked **[derived]** are my own algebra on the code, not stated in MOM6.
This is a reference implementation, not truth.

## 0. Where it is called and the data it sees
- Split-RK2 dynamics calls `horizontal_viscosity(u_av, v_av, h_av, uh, vh, diffu, diffv, ..., dt, hu_cont=BT_cont%h_u, hv_cont=BT_cont%h_v)` with the time-averaged velocity and thickness of the predictor (`src/core/MOM_dynamics_split_RK2.F90:894-898`; corrector call at `:1646`). `diffu/diffv` are accelerations [m s-2] added to the momentum RHS: an explicit term, not part of the implicit vertical solve.
- Main routine `horizontal_viscosity_block` (loop over k-blocks of size `nkk = HORVISC_NKBLOCK`, 0 means all layers): F:324-331, F:2863. Every layer k is treated independently (pure along-layer operator, in z*, sigma, hybrid or layered mode alike; nothing in the stencil sees the vertical coordinate). Per-layer 2-D work arrays only.
- Master switches: `LAPLACIAN` (default False, F:2906), `BIHARMONIC` (default **True**, F:3002; with `AH=0`, no Smagorinsky/Leith, it adds nothing). If neither is on, a WARNING "very bad idea" and return (F:3207-3211).

## 1. Laplacian Smagorinsky (SMAGORINSKY_KH)

### Parameters and defaults (hor_visc_init)
| param | default | line | note |
|---|---|---|---|
| `LAPLACIAN` | False | F:2906 | needed for everything below |
| `KH` | 0 m2/s | F:2910 | static background |
| `KH_BG_MIN` | 0 | F:2914 | floor, applied to Kh at h (F:1084) and q (F:1468) after resolution scaling |
| `KH_VEL_SCALE` | 0 m/s | F:2918 | Kh_bg = max(KH, KH_VEL_SCALE * sqrt(grid_sp2)) |
| `KH_SIN_LAT`, `KH_PWR_OF_SINE` | 0, 4 | F:2925-2932 | latitude-dependent background, taken as max |
| `SMAGORINSKY_KH` | False | F:2935 | forced False if not LAPLACIAN (F:2938) |
| `SMAG_LAP_CONST` | 0 (mandatory if Smag) | F:2939 | doc says "often 0.15" |
| `LEITH_KH`, `LEITH_LAP_CONST` | False, 0 | F:2943, F:2947 | doc "often 1.0" |
| `BOUND_KH` | **True** | F:2957 | forced False if not LAPLACIAN (F:2964) |
| `ADD_LES_VISCOSITY` | False | F:2971 | sum instead of max |
| `HORVISC_BOUND_COEF` | 0.8 | F:3109 | scales both bounds |
| `NOSLIP` | False | F:3118 | |
| `USE_LAND_MASK_FOR_HVISC` | True | F:3105 | |

### Formula
- Rate-of-strain pieces, C-grid, at h-points and q-points (F:707-717, 719-722; metric terms folded into `DY_dxT`, `DX_dyBu`, etc.):
  - `dudx = DY_dxT * (IdyCu(I) u(I) - IdyCu(I-1) u(I-1))`, `dvdy = DX_dyT * (IdxCv(J) v(J) - IdxCv(J-1) v(J-1))` (F:708-712 region, k-loop at F:707).
  - **Tension** (h-point): `sh_xx = dudx - dvdy` (F:714).
  - `dvdx = DY_dxBu*(v(i+1) IdyCv(i+1) - v(i) IdyCv(i))`, `dudy = DX_dyBu*(u(j+1) IdxCu(j+1) - u(j) IdxCu(j))` (F:719-722).
  - **Shearing strain** (q-point): `sh_xy = mask2dBu * (dvdx + dudy)` free-slip (F:927); `(2 - mask2dBu) * (dvdx+dudy)` no-slip (F:921). Full sum, no factor 1/2.
  - These are the forms of Smagorinsky (1993) / Griffies & Hallberg (2000), as stated in the code comment at F:702-705 region (comment text "advocated by Smagorinsky (1993) and discussed in Griffies and Hallberg (2000)").
- Shear magnitude:
  - h-point: `Shear_mag = sqrt( sh_xx^2 + 0.25*(sh_xy(I-1,J-1)^2 + sh_xy(I,J)^2 + sh_xy(I-1,J)^2 + sh_xy(I,J-1)^2) )` (F:1019-1022). Squares of the four surrounding q-values are averaged (not the average of values).
  - q-point: `Shear_mag = sqrt( sh_xy^2 + 0.25*(sh_xx(i,j)^2 + sh_xx(i+1,j+1)^2 + sh_xx(i,j+1)^2 + sh_xx(i+1,j)^2) )` (F:1356-1359).
  - **[derived]** In 2-D, 2 S_ij S_ij = D_T^2 + D_S^2 + D_div^2; MOM6's Shear_mag omits the divergence, so it is sqrt(2 S_ij S_ij - div^2), i.e. |S| of the standard Smagorinsky without the dilatation term.
- Grid-scale length: **harmonic mean of dx^2 and dy^2**, at h-points `grid_sp_h2 = 2 dx^2 dy^2 /(dx^2 + dy^2)` (F:3386) and at q-points `grid_sp_q2` likewise (F:3411). It equals dx^2 when dx = dy. It is the *squared* length (no extra square).
- Static factor: `Laplac2_const_xx = SMAG_LAP_CONST * grid_sp_h2` (F:3389), same at q (F:3414).
- **Smagorinsky viscosity:** `Kh_smag = SMAG_LAP_CONST * grid_sp_h2 * Shear_mag` [m2/s]. The constant multiplies the area directly (no square on the constant), so `SMAG_LAP_CONST = 0.15` plays the role of the usual C_s^2 **[derived]** (usual form `(C_s Delta)^2 |S|`; the doc "often 0.15" matches an area coefficient). Note Delta here is the harmonic grid-spacing mean, not 2 dx.
- Static background at h-points: `Kh_bg_xx = max(KH, KH_VEL_SCALE * sqrt(grid_sp_h2))` (F:3394); optionally `max` with a 2-D file (`USE_KH_BG_2D`, F:3396) and with `KH_SIN_LAT*|sin lat|^pwr` (F:3398-3401). q-points analogous (F:3416-3436).
- **Combination rule (default `ADD_LES_VISCOSITY=False`):** h-points `Kh = Kh_bg_xx` (F:1060 region), then `Kh = max(Kh, Smag)` and `Kh = max(Kh, Leith)` (F:1067-1070); q-points F:1427-1450. If `ADD_LES_VISCOSITY=True` they are *added* to the background (F:1062-1065, F:1437-1440). MEKE (`Ku`), anisotropic and `Res_fn` (resolution function, only if `RESOLN_SCALED_KH`) pieces come after and are *additive* / multiplicative respectively (F:1090-1135, F:1470-1500). Resolution scaling multiplies Kh by `Res_fn_h`/`Res_fn_q` before the floor (F:1076-1078; q: F:1457-1466 region).
- Kh is evaluated separately at h-points (for str_xx) and at q-points (for str_xy), each with its own Shear_mag (not interpolated).
- Floor `KH_BG_MIN` (default 0) is applied after the max/add of all LES pieces and resolution scaling, at h (F:1084) and q (F:1468).

## 2. Biharmonic Smagorinsky (SMAGORINSKY_AH)
- Params (hor_visc_init): `BIHARMONIC` default True (F:3002); `AH` 0 m4/s (F:3006); `AH_VEL_SCALE` 0 m/s (F:3010); `AH_TIME_SCALE` 0 s (F:3016); `SMAGORINSKY_AH` False (F:3022); `SMAG_BI_CONST` 0 (mandatory if used; doc "typically 0.015 - 0.06", F:3056); `BOUND_AH` **True** (F:3038); `BACKSCATTER_UNDERBOUND` False (F:3047); `BOUND_CORIOLIS_BIHARM` defaults to `BOUND_CORIOLIS` (False unless set; F:3083), `BOUND_CORIOLIS_VEL` default `MAXVEL` (3e8 m/s) (F:3094-3098). `NOSLIP` + `BIHARMONIC` is FATAL (F:3205-3207).
- Static factors: `Biharm_const_xx = SMAG_BI_CONST * grid_sp_h2^2` (F:3447, q: F:3480), `Ah_bg = max(AH, AH_VEL_SCALE * grid_sp_h2 * sqrt(grid_sp_h2))` = `AH_VEL_SCALE*dx^3` (F:3466, q: F:3492), and `max` with `grid_sp_h2^2/AH_TIME_SCALE` if set (F:3469-3470).
- Viscosity: `Ah = max(Ah_bg, SMAG_BI_CONST * grid_sp_h2^2 * Shear_mag)` (F:1204-1207, q: F:1598-1601), i.e. `C_bi dx^4 |S|`. Same Shear_mag as the Laplacian case.
- `BOUND_CORIOLIS_BIHARM`: `AhSm = Shear_mag*(Biharm_const + Biharm_const2*Shear_mag)`, `Biharm_const2 = grid_sp^6 * fmax/(5 V_c^2)`, `V_c = BOUND_CORIOLIS_VEL` (F:1199-1202, F:3440-3456, F:3079-3105): a term quadratic in the shear that keeps the viscous drag comparable to Coriolis.
- Biharmonic operator: Del2u/Del2v from `sh_xx`, `sh_xy` (F:949-954), gradient of Del2 (`dDel2vdx`, `dDel2udy`, F:1313-1318), then `str_xx += Ah*(DY_dxT*d(IdyCu Del2u) - DX_dyT*d(IdxCv Del2v))` (F:1301-1306), `str_xy += Ah*(dDel2vdx + dDel2udy)` (F:1604-1610 region). Two sweeps of the first-derivative stencils.
- **What OM4/global configs use:** OM4 files themselves are not in the local tree. The nearest local reference is `ocean_only/global_ALE/hycom/MOM_input:305-325`: `LAPLACIAN=True`, `KH_VEL_SCALE=0.01`, `AH_VEL_SCALE=0.05`, `SMAGORINSKY_AH=True`, `SMAG_BI_CONST=0.06`, `BACKSCATTER_UNDERBOUND=True`, `MAXVEL=6`. Low-resolution idealised examples use the same biharmonic Smagorinsky 0.06.

## 3. Stability bounds (BOUND_KH / BOUND_AH)
- There is **no `BETTER_BOUND_KH/AH` parameter in this MOM6 version**: grep finds only diagnostic descriptions (F:3694-3696). The "newer method" is the combined branch when *both* `BOUND_KH` and `BOUND_AH` are true (F:1121-1129, F:1500-1510). Do not search for the old flag names.
- Static limits, computed in init (F:3377-3380 for Laplacian bookkeeping, bounds at F:3503-3532 Laplacian, F:3534-3590 biharmonic):
  - `Kh_Max_xx = HORVISC_BOUND_COEF * 0.25 * Idt / denom_h`, `denom_h = max( dy2 * DY_dxT * (IdyCu(I)+IdyCu(I-1)) * max(IdyCu*IareaCu), dx2 * DX_dyT * (IdxCv(J)+IdxCv(J-1)) * max(IdxCv*IareaCv) )` (F:3503-3512). `Kh_Max_xy` at q: F:3514-3529. `Idt = 1/DT` of the *baroclinic* dynamics step (F:3190 region).
  - **[derived]** uniform grid, dx=dy=D: denom = 2/D^2, so `Kh_max = 0.8*0.25*D^2/(2 dt) = 0.1 D^2/dt` (classical explicit limit would be 0.25 D^2/dt; the 0.8 factor and the 2 give a margin of 2.5).
  - `Ah_Max_xx = HORVISC_BOUND_COEF * 0.5 * Idt / denom` with `denom` built from the squared stencil (`u0u, u0v, v0u, v0v`, F:3540-3560, F:3562-3578). **[derived]** uniform grid: `Ah_max = 0.0125 D^4/dt` (denom = 32/D^4 times area factor, coef 0.8*0.5/32).
  - The old constants `Kh_Limit = 0.3/(4 dt)` and `Ah_Limit = 0.3/(64 dt)` are assigned (F:3376, F:3440) but not used afterwards (grep shows no other use): dead code.
- Application (per step, every layer): the limit is multiplied by `hrat_min = min(1, h_min/(h_stress_point + h_neglect))`, `h_min` = min of the four neighbouring face thicknesses (h-point: h_u(I), h_u(I-1), h_v(J), h_v(J-1), F:1033-1037; q-point: F:1368-1371). So a thin neighbour layer shrinks the allowed viscosity in proportion (this is the thin-layer guard).
  - Only BOUND_KH: `Kh = min(Kh, hrat_min*Kh_Max)` (F:1131-1133, F:1510-1512).
  - Both: `visc_bound_rem = 1 - Kh/Kh_max_here` (0 if Kh reached the cap), and then `Ah = min(Ah, visc_bound_rem*hrat_min*Ah_Max)` (F:1120-1129, F:1252-1256): the biharmonic gets only the part of the stability budget the Laplacian left (a unified Kh+Ah bound, so the sum is stable).
  - Only BOUND_AH: `Ah = min(Ah, hrat_min*Ah_Max)` (F:1256-1259).
- Both `bound_Kh`/`bound_Ah` are forced False if the operator is off (F:2964, F:3041). `BACKSCATTER_UNDERBOUND` only matters for negative Kh.

## 4. Stress-tensor form and thickness weighting
- Per layer, stress in **flux form with layer thickness** (Griffies/Hallberg): strains -> viscosities -> `str_xx = -Kh*sh_xx` (F:1170) at h-points, `str_xy = -Kh*sh_xy` (F:1558) at q-points (plus biharmonic increments), then multiplied by thickness:
  - `str_xx *= h(i,j,k) * reduction_xx` (F:1727-1730)
  - `str_xy *= hq * mask2dBu * reduction_xy` (F:1734-1737; with no-slip without the mask, F:1733)
  - `reduction_xx/xy` = minimum of open-fraction ratios `dy_Cu/dyCu`, `dx_Cv/dxCv` (partial-cell/blocked-face widths; F:3325-3366). 1 where faces are full.
- Thickness at q-points `hq`: **harmonic-type mean** of the four face thicknesses (F:1362-1367):
  `h2uq = 4 h_u(I,j) h_u(I,j+1)`, `h2vq = 4 h_v(i,J) h_v(i+1,J)`, `hq = 2 h2uq h2vq / ( h_neglect^3 + (h2uq+h2vq)*(h_u(I,j)+h_u(I,j+1)+h_v(i,J)+h_v(i+1,J)) )`. It goes to 0 if any of the four is 0 (a vanished neighbour layer shuts the shear stress). `h_neglect = GV%H_subroundoff` (F:550).
- Thickness at faces: `h_u = 0.5*(mask2dT h(i) + mask2dT h(i+1))` when `USE_LAND_MASK_FOR_HVISC` (F:772-779), plain mean otherwise (F:781-790); or the continuity solver's `hu_cont/hv_cont` if `USE_CONT_THICKNESS` (default False, F:2898; F:758-766 region). DOME sets `USE_LAND_MASK_FOR_HVISC=False` (DOME/MOM_input:289-292).
- Tendency (divergence of the thickness-weighted stress, divided by face thickness) (F:1745-1750, F:1766-1771):
  `diffu(I,j) = [ IdxCu*(dx2q(J-1) str_xy(J-1) - dx2q(J) str_xy(J)) + IdyCu*(dy2h(i) str_xx(i) - dy2h(i+1) str_xx(i+1)) ] * IareaCu / (h_u + h_neglect)`
  `diffv(i,J) = [ IdyCv*(dy2q(I-1) str_xy(I-1) - dy2q(I) str_xy(I)) - IdxCv*(dx2h(i,j) str_xx(i,j) - dx2h(i,j+1) str_xx(i,j+1)) ] * IareaCv / (h_v + h_neglect)`
  (`dx2q = dxBu^2` etc.; the dx^2/dy^2 factors come from folding the metric terms of the tension/shear into the stress, F:3254-3265 region). The same tensor is symmetric, so the divergence form conserves momentum (flux form) and `1/h` on the outside gives acceleration. Dividing by `h_u + h_neglect` is the only vanished-layer guard on the output side; on the input side the bounds shrink Kh by `hrat_min`.
- Open-boundary faces: `diffu/diffv` are zeroed on OBC faces (F:1754-1763, F:1778-1787, comment "not the right boundary condition").
- Energy budget diagnostic `FrictWork` (F:1790+) = stress x strain, only if requested.

## 5. Boundary conditions
- **Free slip (default, NOSLIP=False):** `sh_xy = mask2dBu*(dvdx+dudy)` (F:927): the shear strain is zero at land q-points; `str_xy` is multiplied by `mask2dBu` (F:1736). No stress on closed walls except normal-component tension (u=0 on the wall face, so dudx is computed with u_wall=0).
- **No slip (NOSLIP=True, Laplacian only):** `sh_xy = (2 - mask2dBu)*(dvdx+dudy)` (F:921): at a land q-point the strain is doubled (ghost velocity -u mirrored), and at coastal vorticity points `hq` and `hrat_min` are rebuilt from the two wet face thicknesses only: sum if only one side is wet, harmonic `2 hu hv/(hu+hv+h_neglect)` if both (F:1376-1398); `str_xy` is not masked (F:1731-1733). The doc says free-slip is "strongly encouraged"; with biharmonic it is FATAL (F:3205-3207).
- Land h-points: str_xx uses `h(i,j)` which is 0/ignored for land via `reduction_xx` or h; `USE_LAND_MASK_FOR_HVISC` only affects the face thicknesses.
- **Open boundaries** (`OBC_STRAIN_*`, F:824-884 region): the q-point strain on an OBC segment is set to zero (`OBC_STRAIN_ZERO`: both dvdx, dudy = 0), free-slip (tangential-direction term zero), computed from the OBC tangential velocity (`2*(v_tan - v_interior)/dx`) or specified (`tangential_grad`). `OBC_SPECIFIED_STRAIN_BUG` default True (F:3140). Thickness on the segment face is the interior h (F:830-870), corner thickness copied inward (F:873-905). `OBC%zero_biharmonic` zeroes Del2 on the segment (F:962-975).

## 6. Parameters in the examples (what the reference benchmarks used)
Source: each directory's `MOM_parameter_doc.all` (resolved values) and `MOM_input`.
| case | grid, DT | LAPLACIAN | KH | KH_VEL_SCALE | SMAG_KH | BOUND_KH | BIHARMONIC | AH_VEL_SCALE | SMAG_AH / SMAG_BI | BOUND_AH | NOSLIP |
|---|---|---|---|---|---|---|---|---|---|---|---|
| **DOME** | 150x70, dx = 1500 km/150 = **10 km**, DT 1200 s, 25 layers, KV 1e-4 | **False** (MOM_parameter_doc.all:1057) | - | - | - | - | **True** (:1059) | **0.003** (MOM_input:283) | **True / 0.06** (MOM_input:287-289) | True (default, doc:1079) | False (doc:1001) |
| lock_exchange | 128x4, 0.5 km, DT 250 s, 21 layers | True (:1042) | **0.01** (:1044) | 0 | False | **False** (:1059) | False | - | - | - | False (:939) |
| ISOMIP (rho) | 240 pts, DT 300 s, 25 layers | True (:1182) | 0.0 | 0 | False | True | True (:1209) | 0.001 (:1214) | True / 0.06 (:1222, :1234) | True | False |
| seamount (z) | 40 pts, DT 900 s, 20 layers | True (:1208) | 1000 (:1210) | 0.003 (:1214) | False | True | False | - | - | - | False |
| flow_downslope (z) | 80 pts, DT 300 s, 40 layers | True (:1195) | 1e4 (:1197) | 0.003 (:1201) | False | True | False | - | - | - | False |
| sloshing / adjustment2d | | True | - | - | **True / 0.15** (`sloshing/common/MOM_input:289-291`) | | | | | | |
| global_ALE (hycom/layer/z) | | True | - | 0.01 | False | | True | 0.05 | True / 0.06 | | |
- DOME in detail (what our benchmark reference used): biharmonic only, `Ah_bg = 0.003 dx^3`, `Ah_smag = 0.06 dx^4 |S|`, Coriolis-bound on (`BOUND_CORIOLIS_BIHARM=True`, `BOUND_CORIOLIS_VEL=10`, doc:1086-1092), `BOUND_AH=True`, `HORVISC_BOUND_COEF=0.8`, `USE_LAND_MASK_FOR_HVISC=False`, free slip, no Leith, no Laplacian (also `BOUND_CORIOLIS=True`, MOM_input:267). **[derived]** at dx=10 km: Ah_bg = 3.0e9 m4/s; Smag = 6e14*|S| m4/s (|S| = 1e-5 s-1 gives 6e9); stability cap `Ah_max = 0.0125 dx^4/DT = 1.0e11 m4/s` at DT 1200. The Laplacian Smagorinsky is **not** used in DOME. The DOME case has `DOME` 5 km variants only if the user reconfigures.
- Other `SMAGORINSKY_KH=True` cases: sloshing (z, rho, layer) and adjustment2d with 0.15; default `ADD_LES_VISCOSITY=False`.

## 7. Cost structure and Leith
- Passes per k-block (all per layer, 2-D): (1) dudx, dvdy, sh_xx at h; dvdx, dudy, sh_xy at q (F:707-722, F:919-929); (2) h_u/h_v (F:772-790); (3) optional vorticity/Leith gradients; (4) Shear_mag at h (F:1018-1023), hrat_min (F:1033-1037), Kh at h (F:1060-1071), bounds (F:1121-1133), str_xx; (5) Shear_mag at q (F:1355-1360), hq (F:1362-1367), hrat_min (F:1368-1371), Kh at q (F:1427-1512), str_xy; (6) thickness-weight str (F:1727-1737); (7) divergence for diffu, diffv (F:1745-1771). Biharmonic adds the Del2 sweep (F:949-954) and the gradient-of-Laplacian sweep (F:1313-1318) plus Ah at h and q. Work arrays: dudx, dvdy, dvdx, dudy, sh_xx, sh_xy, Shear_mag (reused h then q), h_u, h_v, hq, hrat_min, Kh/Ah, str_xx, str_xy (+ Del2u, Del2v, dDel2vdx, dDel2udy for biharmonic). No tridiagonal solve; purely stencil/point-wise. The code comments note the Kh loops can be decomposed/vectorised (F:1063 comment). `HORVISC_NKBLOCK` chops layers into blocks for stack/cache (F:2863).
- Halo: Smag at an h-point needs the 4 surrounding q strains; Leith needs halo >= 3 (F:3290-3294 region).
- Leith (`LEITH_KH`): `Kh_Leith = LEITH_LAP_CONST * grid_sp^3 * |grad(vorticity) + grad(divergence)| / pi^3` (const `grid_sp_h3 = grid_sp_h2^{1.5}`, F:3387, F:3390, applied `*inv_PI3`, F:1068-1070); `USE_BETA_IN_LEITH` default = LEITH_KH (F:3059), `MODIFIED_LEITH` adds the divergence gradient, `USE_QG_LEITH_VISC` takes min with a QG form (needs slopes). Biharmonic Leith: `Leith_bi_const * grid_sp^6 |Del2 vort|/pi^6` (F:1210-1215 region, F:3449-3451). Needs vorticity and its gradient: ~1.5-2x the Smagorinsky cost. Leith+E (`USE_LEITHY`) = harmonic backscatter + biharmonic Leith, smooths velocity twice, many blocking halo updates (F:3139-3151).

## Notes useful for the design
1. MOM6 Laplacian Smag uses `max(background, Smag)` (not sum) by default; both evaluated on h-points (for tension) and q-points (for shear), each from the same rate-of-strain magnitude.
2. Bound = `0.8 * 0.25 / (dt * denom) * hrat_min` per point: depends on grid metrics and the **thinnest neighbour layer**; with the Laplacian and biharmonic together, Ah only gets the remaining budget.
3. Thickness weighting of the stress: `h` at tension points, harmonic-like `hq` at shear points; output divided by arithmetic face thickness.
4. Free slip default (zero shear stress at land q-points); no-slip only Laplacian.
5. No "better bound" switch exists here; DOME reference used biharmonic Smagorinsky 0.06 + `AH_VEL_SCALE` 0.003 + Coriolis-bound, bounded, free slip.

## Summary (5 lines)
1. Laplacian Smagorinsky: `Kh = max(KH, KH_VEL_SCALE*sqrt(Dg2), SMAG_LAP_CONST*Dg2*sqrt(sh_xx^2 + mean4(sh_xy^2)))`, `Dg2` = harmonic mean of dx^2, dy^2, h- and q-point values separately, strains `sh_xx = du/dx - dv/dy`, `sh_xy = dv/dx + du/dy`.
2. Biharmonic: `Ah = max(AH, AH_VEL_SCALE*Dg^3, SMAG_BI_CONST*Dg^4*|S|)` (+ optional Coriolis-bound quadratic term); OM-like and DOME use 0.06 (DOME also AH_VEL_SCALE 0.003, no Laplacian, dx = 10 km, DT 1200 s).
3. Bounds: `Kh <= 0.8*0.25/(dt*denom)*hrat_min` (= 0.1 dx^2/dt uniform), `Ah <= 0.8*0.5/(dt*denom)*hrat_min` (= 0.0125 dx^4/dt uniform), the biharmonic gets only what the Laplacian left; no BETTER_BOUND flag in this version.
4. Stress form: per layer, `str = -K*strain` times `h` (h-points) or harmonic-type `hq` (q-points), divergence divided by `h_u + h_neglect`; free-slip default (shear zero at land q-points); OBC strain zero/free-slip/computed.
5. Report file: `/Users/estebancruz/pyOCEAN/docs/design/smag_refs/mom6_hor_visc.md` (some line numbers are approximate to a few lines where marked "region").
