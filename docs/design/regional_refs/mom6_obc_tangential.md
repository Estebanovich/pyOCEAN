# MOM6 tangential open-boundary velocity (ORLANSKI_TAN, NUDGED_TAN) and the "computed" corner vorticity/strain

Reading of MOM6 git 0e9a561 (`~/pyOCEAN_refs/MOM6/src`) for the regional chain, step 3 of the Gulf Stream fix
sequence (approved by Esteban 2026-11-02). NWA12 uses, on every segment,
`"FLATHER,ORLANSKI,NUDGED,ORLANSKI_TAN,NUDGED_TAN"`, `VELOCITY_NUDGING_TIMESCALES = 3.0, 360.0` (days),
`OBC_COMPUTED_VORTICITY = True`, `OBC_COMPUTED_STRAIN = True`, `OBC_ZERO_BIHARMONIC = True`,
`OBC_RADIATION_MAX = 1.0`, `OBC_RAD_VEL_WT = 0.3` (`validation/nwa12_cut/MOM_input:77-99`, MOM_override:12-13),
DT = 600 s. File paths below are relative to `src/`. Indices are MOM6's (I, J corners/faces; i, j cells; Fortran
is case-insensitive, so `v_new(i,J)` at an east segment with face index `I` is the column `i = I`, the **last
interior** column).

## 1. Parsing (what the keywords switch on)

`setup_u_point_obc` / `setup_v_point_obc` (`core/MOM_open_boundary.F90:1770-1898`, `:1900-2030`):

| keyword | flags | line |
|---|---|---|
| `ORLANSKI_TAN` | `radiation = .true.`, `radiation_tan = .true.` | `:1817-1820` |
| `NUDGED_TAN` | `nudged_tan = .true.` | `:1838-1840` |
| (`NUDGED` or `NUDGED_TAN`) | `Velocity_nudging_timescale_in/out` from `..._VELOCITY_NUDGING_TIMESCALES` (days) | `:1858-1869` |

`OBC_COMPUTED_VORTICITY` → `OBC%vorticity_config = OBC_VORTICITY_COMPUTED` (`:589-640`);
`OBC_COMPUTED_STRAIN` → `OBC%strain_config = OBC_STRAIN_COMPUTED` (`:653-703`).
`OBC_RADIATION_MAX` (`rx_max`, grid points per step, default 1) and `OBC_RAD_VEL_WT` (`gamma_uv`, default 0.3)
(`:847-857`).

## 2. What is radiated: the tangential velocity at the corner (q) points of the segment

`segment%tangential_vel(I,J,k)` lives on the q-points **on the boundary line** (for an E/W segment at u-face `I`,
the corners `(I, J)`, `J = JsdB … JedB`; for N/S at v-face `J`, corners `(I, J)`). It is the layer velocity
parallel to the segment at the boundary itself — half a cell outside the last interior tangential velocity. It is
**not** a prognostic unknown of the momentum equations; it only sets the vorticity and shear strain at those
corners (§4).

Each call of `radiation_open_bdry_conds` (`:2891-3957`), e.g. east segment (`:3026-3062`):

```
if (gamma_u < 1)   ! NWA12: gamma_uv = 0.3
   rx_tang_rad(I,J,k) = ½ (rx_norm_rad(I,j,k) + rx_norm_rad(I,j+1,k))       interior corners   (:3034-3036)
   rx_tang_rad at the two end corners = rx_norm_rad of their single face                        (:3031-3032)
else
   dhdt = v_old(i,J) − v_new(i,J);  dhdx = v_new(i,J) − v_new(i−1,J)                             (:3039-3040)
   rx_tang_rad = min(dhdt/dhdx, rx_max) if dhdt·dhdx > 0 else 0                                  (:3041)
tangential_vel(I,J,k) = (v_new(i,J,k) + rx·v_new(i−1,J,k)) / (1 + rx)       ORLANSKI_TAN          (:3045-3049)
tau   = tau_in if rx_tang_rad ≤ 0 else tau_out                              NUDGED_TAN            (:3051-3062)
gamma_2 = dt/(tau + dt)
tangential_vel = (1 − gamma_2) tangential_vel + gamma_2 nudged_tangential_vel
```

* **Phase speed.** With `gamma_uv < 1` (NWA12) the tangential phase speed is **not** estimated from the
  tangential velocity: it is the corner average of the normal-velocity phase speed `rx_norm_rad` of the two
  adjacent boundary faces. `rx_norm_rad` is the running mean `(1 − γ_u) rx_old + γ_u rx_new` of the 1-D Orlanski
  estimate `rx_new = min(dhdt/dhdx, rx_max)` (0 unless `dhdt·dhdx > 0`) of the normal velocity at the first
  interior face (`:2953-2970`); it is in units of grid points per step, limited to `[0, rx_max = 1]`, and is kept
  in the restart (`OBC%rx_normal`, `:2971-2974`). It is not remapped by ALE (only `MOM_open_boundary.F90` touches
  `rx_normal`; grep of `src/`).
* **Radiated value.** An upstream extrapolation from the interior: `rx = 0` gives the last interior tangential
  velocity `v(i,J)` (a free-slip, zero-gradient boundary value), `rx = 1` the mean of the last two.
* **Inflow/outflow for the nudging.** As for the normal component (`mom6_obc_tides.md` §4, R7 note), the short
  time scale `tau_in` is used whenever the phase speed is not outward (`rx ≤ 0`), not by the sign of the flow.
  NWA12: τ_in = 3 d, τ_out = 360 d, so `gamma_2 = 600/(259200+600) = 2.31e-3` per call on "inflow" and
  `1.93e-5` on "outflow".
* **The nudging has no memory.** `tangential_vel` is overwritten by the radiated value at every call, so the
  nudging is a one-call blend `tangential_vel = rad + gamma_2 (ext − rad)`, not a relaxation that accumulates over
  steps. With `rx = 0`, `tangential_vel − v(i,J) = gamma_2 (v_ext − v(i,J))`: the corner sees the exterior only
  through a 0.23 % pull per call. ORLANSKI_TAN + NUDGED_TAN is therefore close to a free-slip (zero normal
  gradient) tangential condition with a weak pull to the exterior; it does not impose the exterior tangential
  velocity.
* West (`:3271-3307`): `(v_new(i+1,J) + rx v_new(i+2,J))/(1+rx)`; north (`:3515-3550` analogue, `u_new(I,j)`,
  `u_new(I,j−1)`); south (`:3760-3794`): `(u_new(I,j+1) + ry u_new(I,j+2))/(1+ry)`.
* `nudged_tangential_vel` is the exterior: `update_OBC_dynamics_data` sets
  `tangential_vel = V_data + tidal_vt` and copies it to `nudged_tangential_vel` (`:4930-4951`); the segment
  `V` (E/W) / `U` (N/S) data are on the segment's q-points, rotated to the grid (CEFI `write_glorys_boundary`;
  NWA12 `OBC_SEGMENT_00n_DATA` U/V `file:uv_00n.nc`, `MOM_input:195-197`). NWA12's tide adds `tidal_vt`.

## 3. Order in the step (split RK2)

`core/MOM_dynamics_split_RK2.F90`:

1. `update_OBC_data` (`:552-554`): `tangential_vel ← exterior` (and `nudged_tangential_vel`).
2. Predictor Coriolis `CorAdCalc` (`:561-569`) only when `CAu_pred` was not stored at the end of the last step
   (`STORE_CORIOLIS_ACCEL`, default True: normally skipped; on the first step it sees the **exterior** value).
3. Predictor barotropic step, then `radiation_open_bdry_conds(u_av, …, dt_pred)` (`:801-806`): tangential_vel
   radiated + nudged from the predicted velocities.
4. Corrector `horizontal_viscosity` (`:894-898`) and `CorAdCalc` (`:903-904`) use that tangential_vel.
5. Corrector barotropic step, `radiation_open_bdry_conds(u_inst, …, dt)` (`:1068-1071`).
6. `CorAdCalc` for the next step's stored predictor (`:1093-1107`) with the step-5 tangential_vel.

So every momentum operator uses a tangential_vel radiated from the most recent velocities (the end of the last
step or of the predictor), nudged once.

## 4. Where it enters the momentum equations

### 4.1 Vorticity in the Coriolis/advection term (`core/MOM_CoriolisAdv.F90:453-604`)

Circulations `dvdx = v(i+1,J) dyCv(i+1,J) − v(i,J) dyCv(i,J)`, `dudy = u(I,j+1) dxCu(I,j+1) − u(I,j) dxCu(I,j)`
(`:421-426`). At the corners of an open segment, `OBC_VORTICITY_COMPUTED` replaces only the cross-boundary
derivative of the tangential velocity by a one-sided half-cell difference to the boundary value (`:470-478`,
`:516-524`):

```
E:  dvdx(I,J) = 2 (tangential_vel(I,J) − v(i,J))   dyCv(i,J)
W:  dvdx(I,J) = 2 (v(i+1,J) − tangential_vel(I,J)) dyCv(i+1,J)
N:  dudy(I,J) = 2 (tangential_vel(I,J) − u(I,j))   dxCu(I,j)
S:  dudy(I,J) = 2 (u(I,j+1) − tangential_vel(I,J)) dxCu(I,j+1)
```

The along-boundary derivative of the **normal** velocity (`dudy` on E/W, `dvdx` on N/S) is kept as computed
from the boundary-face normal velocities (Flather + Orlanski + nudging). ζ = (dvdx − dudy)·IareaBu with
`mask2dBu = 1` at these corners: `initialize_masks` opens a vertex between two wet OBC faces of the same
orientation (`initialization/MOM_grid_initialize.F90:1296-1311`); the outer corners of the domain (two segments
meeting) stay closed unless `OBC_OPEN_CORNERS` (`:1312-1323`). The thickness at the corner is projected with a
no-gradient condition: `hArea_u/hArea_v` across the boundary take the interior cell (`:480-488`, `:526-533`,
`:555-588`). `q = (f + ζ)/h_q` then enters the PV flux of the faces around the corner: on an E/W segment the v of
the last interior column, on an N/S segment the u of the first interior row — which multiplies the boundary
normal transport (`vh` through a south inflow edge).

The other configurations: `ZERO` sets both `dvdx` and `dudy` to 0 (ζ = 0), `FREESLIP` only the cross-boundary
one (`:462-469`, `:508-515`).

### 4.2 Shear strain in the horizontal viscosity (`parameterizations/lateral/MOM_hor_visc.F90:784-856, 917-928`)

`OBC_STRAIN_COMPUTED`, with `DX_dyBu = dxBu/dyBu`, `DY_dxBu = dyBu/dxBu` (`:795-807`, `:824-836`):

```
E:  dvdx(I,J) = 2 DY_dxBu(I,J) (tangential_vel(I,J) − v(i,J))   IdyCv(i,J)
W:  dvdx(I,J) = 2 DY_dxBu(I,J) (v(i+1,J) − tangential_vel(I,J)) IdyCv(i+1,J)
N:  dudy(I,J) = 2 DX_dyBu(I,J) (tangential_vel(I,J) − u(I,j))   IdxCu(I,j)
S:  dudy(I,J) = 2 DX_dyBu(I,J) (u(I,j+1) − tangential_vel(I,J)) IdxCu(I,j+1)
```

and `sh_xy = mask2dBu (dvdx + dudy)` (`:927`) with the along-boundary derivative of the normal velocity kept.
The face thicknesses next to the segment are projected from the interior cell (`h_u(I,j) = h(i,j)` E,
`h(i+1,j)` W; `h_v(i,J) = h(i,j)` N, `h(i,j+1)` S; `:857-880`), so `hq` at the boundary corner (`:1362-1367`) is
built from interior thicknesses. The stress `str_xy = ν_q hq sh_xy` at the corner enters the divergence for the
interior tangential velocity next to the segment; the tendency of the boundary-normal faces is zeroed
(`:1761-1793`). `sh_xy` at the corner also enters |S| at the h points around it (`:1018-1022`).
`OBC_ZERO_BIHARMONIC` (`:956-…`) zeroes the Laplacian of the velocity on open faces for the biharmonic operator;
pyOCEAN has no biharmonic operator (class b, case.py parity table), so it does not apply.

## 5. pyOCEAN before this change (for the record)

* Vector-invariant ζ: `vi.corner_mask` treats the outside of the domain as land, so ζ = 0 at every corner on an
  open edge — MOM6's `OBC_VORTICITY_ZERO`, not free slip: the along-edge shear of the inflow normal velocity
  (−∂u/∂y of a jet entering through a west edge) is missing from the PV at the edge corners.
* Smagorinsky: D_S = 0 at every domain-edge corner (`hor_visc.smagorinsky`, active corners are interior only) —
  `OBC_STRAIN_ZERO`; no shear stress at the open-edge corners. The open faces are not "wet" in the operator
  (`H_u` is NaN outside the domain), so they carry no tension either.
* The case.py parity table called this "free-slip"; it is the ZERO configuration (corrected with this change).

## 6. What pyOCEAN implements (`obc_tangential='mom6'`)

* Corner set: on every `'radiating'` edge, the corners on the edge line between two open wet boundary faces
  (MOM6 `mask2dBu` rule); the outer corners of the domain stay as before.
* Phase speed: the corner average of the normal-velocity Orlanski ratio `r` of pyOCEAN's own normal radiation
  (`OpenBoundary.radiate`, ROMS implicit form, clipped to `[0, orlanski_cmax]`), ends one-sided, as MOM6's
  `gamma_uv < 1` branch. pyOCEAN's normal `r` has no running mean (class c, as before this change). `r` of the
  last step is a state field (`SigmaState.obc_rtan`, restart), not remapped (as MOM6).
* Tangential velocity at the start of the step from the current state: `(v_int + r v_int2)/(1 + r)`, nudged once
  with `γ = Δt/(τ + Δt)`, τ = `obc_tau_in` when `r ≤ 0`, else `obc_tau_out`, toward the exterior
  `OBData.u_tangential` remapped to the corner layers (0 = resting exterior when the provider gives none).
  MOM6 radiates from the end-of-step velocities of the last step: the same values (the ALE remap of velocities
  between the two is the only difference).
* Vorticity: §4.1 formulas at those corners, q recomputed there. Strain: §4.2 with the boundary-face thicknesses
  projected from the interior cells (h_u of the open face = the boundary cell's h; the outer v/u face mirrors the
  inner one) and |S| at the corner with the two interior tensions mirrored outside (MOM6 uses halo values; class c).
* Not ported: OBLIQUE, GRADIENT/`_GRAD`, SPECIFIED vorticity/strain, open domain corners, the `rx` running mean.
