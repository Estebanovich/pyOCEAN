# MOM6 open boundaries with tides, body tide and SAL: reference read for pyOCEAN regional runs

MOM6 source: `~/pyOCEAN_refs/MOM6`, git `0e9a561` (2026-10-07). `file:line` references are relative
to `~/pyOCEAN_refs/MOM6/src/` unless stated otherwise. NWA12: `~/pyOCEAN_refs/CEFI-regional-MOM6`
git `36b8f49`, `exps/NWA12.COBALT/MOM_parameter_doc.all` (cited `.all:line`), `INPUT/MOM_override`
and `tools/boundary/` (git `36b8f49`). CEFI builds MOM6 from its `dev/cefi` fork
(`.gitmodules`); line numbers are GFDL's. Read 2026-11-01 for the regional chain (plan Tasks 5, 6
and 10). Reference implementation, not truth; disagreements flagged.

---

## 1. NWA12 settings (answers to the brief)

| quantity | NWA12 value | source |
|---|---|---|
| segments | 3: south `J=0,I=0:N`, north `J=N,I=N:0`, east `I=N,J=0:N`; each `FLATHER,ORLANSKI,NUDGED,ORLANSKI_TAN,NUDGED_TAN` | `.all:344, 350, 356` |
| velocity nudging | `3.0, 360.0` days (inflow, outflow) on all three | `.all:346, 352, 358` |
| Orlanski | `OBC_RADIATION_MAX = 1.0`, `OBC_RAD_VEL_WT = 0.3` | `.all:362, 366` |
| tracer reservoir | `OBC_TRACER_RESERVOIR_LENGTH_SCALE_OUT/IN = 9000 m`; thickness reservoir 0 (off) | `.all:371, 377, 381, 384` |
| data on supergrid | `BRUSHCUTTER_MODE = True` | `.all:399` |
| OBC remapping | `PPM_H4` | `.all:387` |
| boundary tides | `OBC_TIDE_N_CONSTITUENTS = 10`, `"M2,S2,N2,K2,K1,O1,P1,Q1,MM,MF"` | `.all:324, 410` |
| reference dates | `TIDE_REF_DATE = 1993,1,1`, `TIDE_USE_EQ_PHASE = True`, `TIDE_ADD_NODAL = True`, `TIDE_NODAL_REF_DATE = 1993,7,2` (also in `INPUT/MOM_override`) | `.all:1315, 1318, 1380, 1382` |
| **ramp** | **`RAMP_OBCS = False`** (`OBC_RAMP_TIMESCALE = 1.0` day logged but unused) | `.all:319, 322` |
| body tide | `TIDES = True`, all ten `TIDE_xx = True` | `.all:1135, 1279-1306` |
| **SAL** | **scalar**: `CALCULATE_SAL = True`, `SAL_SCALAR_APPROX = True`, `SAL_SCALAR_VALUE = 0.05`, `TIDAL_SAL_FROM_FILE = False`, `USE_PREVIOUS_TIDES = False`, `SAL_USE_BPA = False`, `SAL_HARMONICS = False`. The parameter is `SAL_SCALAR_VALUE`; `TIDE_SAL_SCALAR_VALUE` is its old name (`parameterizations/lateral/MOM_tidal_forcing.F90:368`, `old_name='TIDE_SAL_SCALAR_VALUE'`); the SAL itself is computed in `parameterizations/lateral/MOM_self_attr_load.F90` (`calc_SAL`, `scalar_SAL_sensitivity` :130-135; the parameter read again at :291-296; Task 1 review note, corrected in Task 11) | `.all:1137, 1263-1274, 1309-1312` |
| SAL in Flather | `TIDAL_SAL_FLATHER = True`, `BAROTROPIC_TIDAL_SAL_BUG = False` | `.all:1645, 1649` |
| tide answers | `TIDES_ANSWER_DATE = 20230630` (legacy path, §6) | `.all:1398` |
| bottom drag | `BOTTOMDRAGLAW = True` (quadratic), `LINEAR_DRAG = False`, `CDRAG = 0.003`, `HBBL = 10.0 m`, `DRAG_BG_VEL = 0.0`, `BBL_USE_TIDAL_BG = False`, `CHANNEL_DRAG = True`, `BBL_THICK_MIN = 0.1 m`, `DRAG_AS_BODY_FORCE = False` | `.all:992-1041` |
| `CD_TIDES` | `0.0018`, but only used for `BBL_tidal_dis = ρ cd_tides utide³` with `UTIDE = 0.0` and `INT_TIDE_DISSIPATION = False`, so it has **no effect** in NWA12 (`config_src/drivers/FMS_cap/MOM_surface_forcing_gfdl.F90:1644, 1679`) | `.all` surface-forcing block |
| OBC bug flags | `OBC_HOR_INDEXING_BUG = True`, `OBC_TRACER_DZ_BUG = True`, `OBC_PROJECTION_BUG = True`, `EXTERIOR_OBC_BUG = True`, `OBC_TEMP_SALT_NEEDED_BUG = True`, `OBC_TS_RESERVOIR_INIT_BUG = True`, `OBC_RESERVOIR_INIT_BUG = True`, `OBC_VALUE_UPDATE_BUG = True` | `.all:325-340, 452, 620, 813, 816` |

So for pyOCEAN's `parity_config()` (plan Task 10): quadratic drag `cd = 0.003`, `hbbl = 10`,
`drag_bg_vel = 0`; `sal_beta = 0.05`; no ramp. The plan's tidal ramp (Task 5) is a pyOCEAN choice,
not NWA12's. `CHANNEL_DRAG` (drag on each layer in proportion to the bottom fraction it overlies) has
no pyOCEAN equivalent; class (b)/(c).

## 2. Segment data: files, supergrid and the indexing bug

`OBC_SEGMENT_00n_DATA` (`.all:633-637`): `U=file:uv_00n.nc(u)`, `V=...(v)`, `SSH=file:zos_00n.nc(zos)`,
`TEMP=file:thetao_00n.nc(thetao)`, `SALT=file:so_00n.nc(so)`, `Uamp/Uphase/Vamp/Vphase` from
`tu_00n.nc`, `SSHamp/SSHphase` from `tz_00n.nc` (CEFI writers in `cefi_tools.md`). Field names and
which are "tidal" (third axis = constituent, no vertical remap): `core/MOM_open_boundary.F90:117-118,
1429-1433`; units: `Uamp` m/s, `SSHamp` m (`:1522, 1527`).

**Brushcutter mode** reads the segment on supergrid nodes: `2·n_seg − 1` values per segment
(`:4464-4475`), corner (q) nodes at odd 1-based positions and face points at even positions
(`:4522-4525`).

**`OBC_HOR_INDEXING_BUG = True` (NWA12) shifts face data by half a cell.** For fields on the faces
(normal velocity, SSH, T, S and the tidal amplitude/phase of SSH and normal velocity,
`field_is_on_face` `:1412-1425`), the face values are taken from
`tmp_buffer(2*(JsdB+off+1) + bug_offset : … : 2)` with `bug_offset = −1` (`:4527-4544`; same for the
layer thicknesses `:4589-4606`). The comment says it outright: "The bug that is being corrected here
is the use of the odd indexed points for both the corners and the faces" (`:4523-4526`). With the bug,
face `j` takes the value at the supergrid node half a cell before it (the south corner on an E/W
segment, the west corner on an N/S segment). Tangential fields (on corners) are unaffected. This is a
reference-model error, class (d), kept by NWA12 for answers. pyOCEAN should sample at the face; for
a bit-level MOM6 comparison of the cut, the MOM6 run must either keep the flag (and pyOCEAN accept a
half-cell shift in the comparison) or set it False. Decision for Esteban.

Each segment's data are linearly interpolated in time by `time_interp_external` (`:4491`) and
vertically remapped with `OBC_REMAPPING_SCHEME` onto the model layers (tidal fields excepted,
`:4612-4617`).

## 3. Tides on the open boundary (`OBC_TIDE`)

**Initialisation** (`initialize_obc_tides`, `core/MOM_open_boundary.F90:1544-1644`):
- `OBC%time_ref = TIDE_REF_DATE` at 00:00 (`:1583-1592`); with `TIDE_USE_EQ_PHASE` the astronomical
  longitudes are evaluated once at that date (`:1595`).
- The nodal longitude comes from `TIDE_NODAL_REF_DATE` if given (`:1599-1603`), else from the
  reference date (`:1604-1611`).
- Per constituent: frequency `TIDE_xx_FREQ` (default `tidal_frequency`, `:1622-1627`), equilibrium
  phase `eq_phase(name, longitudes at time_ref)` (`:1630-1634`), nodal `f`, `u` from
  `nodal_fu(name, N at nodal date)` (`:1637-1642`).
- **f and u are constants for the whole run**, fixed at `TIDE_NODAL_REF_DATE` = 1993-07-02 in NWA12
  (mid-year of the first year). They are not updated in time. For a multi-year run this is an
  approximation (N moves 19.3°/yr); for the 1-year cut it is the NWA12 choice. pyOCEAN's OTPS-style
  nodal correction evaluated at each time would differ (class c).

**Each update** (`update_OBC_dynamics_data`, `:4835-4985`), with
`time_delta = Time − time_ref` in seconds (`:4851-4852`):

```
tidal_amp   = fn(c) * Amp(c)                                                     (:4888)
tidal_phase = (time_delta * omega(c) - Phase(c)) + (eq_phase(c) + un(c))          (:4889-4890)
tidal_x    += tidal_amp * cos(tidal_phase)                                        (:4891)
```

for the normal velocity (`Uamp/Uphase` on E/W, `Vamp/Vphase` on N/S, `:4884-4893`), the tangential
velocity (`:4932-4941`) and the elevation (`SSHamp/SSHphase`, `:4968-4978`). Phases are Greenwich
phase lags in radians (CEFI writes `zphase = −angle(hRe + i hIm)` with `h = ha·exp(−i hp)`, i.e.
`zphase = hp`; `tools/boundary/boundary.py:771`, `write_tpxo_boundary.py:22`).

**Where the tide goes:**
- SSH: `segment%SSH = ramp · (zos + tidal_elev)` (`:4980-4983`); with `RAMP_OBCS = False`,
  `ramp_value = 1` (`:4980`). The ramp, when on, multiplies the **total** SSH (GLORYS + tide), not
  only the tide, and is linear in time over `OBC_RAMP_TIMESCALE` from its activation
  (`update_OBC_ramp`, `:5100-5137`); it is applied to SSH only ("Ramping SSH only so far", `:714-717`).
- Normal velocity: the tidal velocity is **added to every layer**:
  `normal_vel(k) = u_GLORYS(k) + tidal_vn` (`:4900, 4912`), then
  `normal_trans(k) = normal_vel(k) · h(k) · dyCu` and
  `normal_vel_bt = Σ normal_trans / (max(Htot, 1e-12 m) · dyCu)` (`:4896-4917`). So the tide enters
  the baroclinic (layer) boundary velocities as a depth-uniform increment, and therefore also the
  barotropic `normal_vel_bt` used by Flather. The nudging target `nudged_normal_vel = normal_vel`
  (`:4923-4927`) also contains the tide.
- Tangential velocity: `tangential_vel(k) = v_GLORYS(k) + tidal_vt` (`:4943-4945`), nudging target
  likewise (`:4947-4951`).
- The tidal velocity amplitudes are **velocities** (m/s), not transports. CEFI produces them from
  TPXO9 `ua`, `va` (cm/s × 0.01, `write_tpxo_boundary.py:32, 43`). pyOCEAN's plan (Task 5) converts
  TPXO transport with the model face depth, `u = U/D`; that is a different (arguably better, class c)
  choice from NWA12, which uses TPXO's own depth implicitly. Also, NWA12 uses **TPXO9 v1**
  (`tools/boundary/tpxo_obc.yaml:24`, `/work/acr/tpxo9/`), not TPXO10.

**Equilibrium arguments** (`eq_phase`, `parameterizations/lateral/MOM_tidal_forcing.F90:120-150`),
after Kowalik & Luick (2019) Table I.4 [verificar against the book]:

| | V₀ | | V₀ |
|---|---|---|---|
| M2 | 2(h − s) | K1 | h + π/2 |
| S2 | 0 | O1 | −2s + h − π/2 |
| N2 | −3s + 2h + p | P1 | −h − π/2 |
| K2 | 2h | Q1 | −3s + h + p − π/2 |
| MF | 2s | MM | s − p |

with mean longitudes at `time_ref` (`astro_longitudes_init`, `:91-114`; K&L eq. I.71 after
Schureman 1958), T in Julian centuries since 1900-01-01 00:00 with 36525 days (the code notes K&L
use 36526, `:103-105`):

```
s = 277.0248 + 481267.8906 T + 0.0011 T²     (moon)
h = 280.1895 +  36000.7689 T + 3.0310e-4 T²  (sun)
p = 334.3853 +   4069.0340 T − 0.0103 T²     (lunar perigee)
N = 259.1568 −   1934.142  T + 0.0021 T²     (ascending node)
```

These V₀ assume `time_ref` at Greenwich midnight (`:88-90`), so the hour-angle term of the diurnal
and semidiurnal arguments is 180° or 360° and drops out (checked by hand for M2, K1, O1, P1 against
Schureman's T + h − 90° etc.).

**Nodal factors** (`nodal_fu`, `:187-231`, K&L Table I.6, first-order in N):

| | f | u [deg] |
|---|---|---|
| M2, N2 | 1 − 0.037 cos N | −2.1 sin N |
| S2, P1 | 1 | 0 |
| K2 | 1.024 + 0.286 cos N | −17.7 sin N |
| K1 | 1.006 + 0.115 cos N | −8.9 sin N |
| O1, Q1 | 1.009 + 0.187 cos N | 10.8 sin N |
| **MF** | **1.043 + 0.414 cos N** | **−23.7 sin N** |
| **MM** | **1 − 0.130 cos N** | **0** |

These are simpler than OTPS/pyTMD's (which include more terms; pyOCEAN's `forcing/tides.py` follows
OTPS). The plan's Task 5 test against pyTMD will therefore not reproduce MOM6's MM/MF factors; for
parity, the MOM6 values above are the reference (class c).

**Frequencies** (`tidal_frequency`, `:154-182`; NWA12 logs the same, `.all:412-448`), rad/s:
M2 1.4051890e-4, S2 1.4544410e-4, N2 1.3787970e-4, K2 1.4584234e-4, K1 0.7292117e-4,
O1 0.6759774e-4, P1 0.7252295e-4, Q1 0.6495854e-4, **MF 0.053234e-4**, **MM 0.026392e-4**
(periods 13.661 d and 27.555 d). They are rounded to 5–8 significant figures; over one year the
phase drift from rounding is ≤ ~1e-3 rad. The phase is advanced as ω·(t − t_ref) from the fixed V₀,
not recomputed from the astronomical arguments at t.

## 4. Radiation and nudging of the baroclinic velocity (ORLANSKI, NUDGED)

`radiation_open_bdry_conds` (`core/MOM_open_boundary.F90:2891-3957`), called after the predictor
and corrector of the split RK2 step (`core/MOM_dynamics_split_RK2.F90:806, 1071`). For an eastern
segment and each layer (`:2950-3026`):

```
dhdt = u_old(I-1) - u_new(I-1)                     ! old − new, interior point
dhdx = u_new(I-1) - u_new(I-2)
rx_new = (dhdt*dhdx > 0) ? min(dhdt/dhdx, rx_max) : 0                (:2956-2958)
rx_avg = (1-gamma_u) * rx_old + gamma_u * rx_new   (gamma_u = OBC_RAD_VEL_WT = 0.3)  (:2959-2964)
normal_vel(I) = (u_new(I) + rx_avg * u_new(I-1)) / (1 + rx_avg)       (:2969)
```

(`rx` is in grid points per step, capped at `OBC_RADIATION_MAX = 1`; the running mean of `rx` is
kept in the restart, `:2972-2974`.) Then, with NUDGED (`:3013-3025`):

```
tau = (dhdt*dhdx <= 0) ? tau_in : tau_out          ! 3 d : 360 d in NWA12
gamma_2 = dt / (tau + dt)
normal_vel = (1 - gamma_2) * normal_vel + gamma_2 * nudged_normal_vel
```

**"Inflow" here is not the sign of the velocity.** The short timescale is used whenever the Orlanski
phase speed is not outward (`dhdt·dhdx ≤ 0`), i.e. when no outgoing wave is detected, regardless of
the flow direction. A reader of the parameter description ("for inflow, then outflow", `:1859-1867`)
would expect the velocity sign. pyOCEAN's `'radiating'` mode uses CROCO's nudging (τ_in 1 d,
τ_out 10 d) chosen by flow direction (CLAUDE.md, ALE row); for NWA12 parity the switch must follow
MOM6's rule (class c, decide in Task 6). The tangential velocity gets the same treatment with
`ORLANSKI_TAN`, `NUDGED_TAN` (`:3027-3095`).

## 5. Flather (barotropic) and the SAL correction of its wave speed

In the barotropic solver, Flather faces (`set_up_BT_OBC`, `core/MOM_barotropic.F90:4430-4459`):
`ubt_outer = segment%normal_vel_bt` (GLORYS + tide, §3), `SSH_outer = segment%SSH + Z_ref`
(`:4455-4456`), `dZ = bathyT + eta` of the interior cell and
`Cg = sqrt(dgeo_de · g′₁ · dZ)` (`:4434-4446`). Each barotropic substep, eastern face
(`:3904-3925`):

```
cfl     = dtbt * Cg * IdxCu
u_inlet = cfl * ubt_old(I-1) + (1 - cfl) * ubt_old(I)
ssh_in  = eta(i) + (0.5 - cfl) * (eta(i) - eta(i-1))            ! Boussinesq
ubt(I)  = 0.5 * ((u_inlet + ubt_outer) + (Cg/dZ) * (ssh_in - SSH_outer))
```

This is the characteristic form (half from inside, half from outside), not the textbook
`u = u_ext + (c/H)(η − η_ext)`; the two agree when `u_inlet = u_ext`. pyOCEAN's Flather is implicit
in the free-surface solve; class (c).

`dgeo_de = (1 − β) + G_extra` with `β = SAL_SCALAR_VALUE = 0.05` (`:1179-1186`), and the Flather
speed uses it only if `TIDAL_SAL_FLATHER = True` (NWA12; `:1216`, parameter `:5766`). So the NWA12
Flather wave speed is `sqrt(0.95 g D)`.

## 6. Body tide and scalar SAL (`TIDES = True`)

**Equilibrium tide** (`MOM_tidal_forcing.F90`): spatial structures
(`tidal_forcing_init`, `:288-296`): semidiurnal `cos²φ cos 2λ / −cos²φ sin 2λ`, diurnal
`sin 2φ cos λ / −sin 2φ sin λ`, long period `0.5 − 1.5 sin²φ` (MF, MM: `struct = 3`, `:452, 457`).
Love-number factors `love_no` (= 1 + k − h): 0.693 (M2, S2, N2, K2, MF, MM), 0.736 (K1), 0.695 (O1,
Q1), 0.706 (P1); equilibrium amplitudes 0.242334, 0.112743, 0.046397, 0.030684, 0.141565, 0.100661,
0.046848, 0.019273, 0.042041 (MF), 0.022191 m (MM) (`:411-459`). Phases `TIDE_xx_PHASE_T0` default
to `eq_phase` at `TIDE_REF_DATE` (`:464-471`, read at `:488`; NWA12 logs them, e.g. M2 9.3476 rad, `.all:1323`).
Nodal f, u at `TIDE_NODAL_REF_DATE` (`:530-561`), fixed in time as on the boundary.

```
e_tide_eq = Σ_c amp·love·f · [cos(ω t + φ0 + u) cos_struct + sin(ω t + φ0 + u) sin_struct]
```

(`calc_tidal_forcing`, `:638-646`; legacy version `:673-750`, same sum at `:716-726`).

**Scalar SAL** (`parameterizations/lateral/MOM_self_attr_load.F90`): `eta_sal = linear_scaling ·
SSH` with `linear_scaling = eta_prop = SAL_SCALAR_VALUE` (no BPA, not doubled because
`USE_PREVIOUS_TIDES = False`; `:96-99, 312-324`). SSH is the **total** sea-surface height
`Σh − D` (clipped at `min(−D − meanSL, 0)` start, `core/MOM_PressureForce_FV.F90:1157-1166`), not a
tidal part: GLORYS-driven SSH anomalies also get the 5 % SAL reduction.

**How they enter the momentum (NWA12, `TIDES_ANSWER_DATE = 20230630`).** In
`PressureForce_FV_Bouss` the legacy path (`:1185-1191`) calls `calc_tidal_forcing_legacy`, which
returns `e_sal_and_tide = e_sal + e_tide_eq (+ file/previous SAL terms)`, and lowers the bottom
interface: `e(nz+1) = e(nz+1) − e_sal_and_tide`, after which all interfaces are rebuilt upward from
the bottom (`:1193-1196`). Every interface, including the surface, is shifted by
`−(β η + η_eq)`, so the surface-pressure gradient becomes `g ∇[(1 − β) η − η_eq]` (equilibrium
tide as `love·amp…`). The code comment warns that this old algorithm moves interfaces *before*
the density integrals and "is therefore incorrect without SSH_IN_EOS_PRESSURE_FOR_PGF=True"
(`:1150-1153`); NWA12 has `SSH_IN_EOS_PRESSURE_FOR_PGF = False` (`.all:1412`). The error is a
small baroclinic contamination of the tidal forcing through the pressure in the EOS (class d,
NWA12's choice for answers). The same `eta` with the tide terms is passed to the barotropic solver
as `eta_PF` (`:1880-1898`), and the barotropic gravity-wave coupling uses `dgeo_de = 1 − β`
(§5). pyOCEAN's planned `sal_beta` (g(1 − β)∇η) plus `body_force` matches this in the barotropic
mode; it does not reproduce the legacy interface shift in the baroclinic PGF.

## 7. Tracer reservoir

Length scales: `Tr_InvLscale_out/in = 1/9000 m⁻¹` (`> 0` → relax; `< 0` → never update; `= 0` →
instant, sentinel −1; `core/MOM_open_boundary.F90:890-902`). After each tracer advection step,
`update_segment_tracer_reservoirs` (`:6150-6271`, called from `core/MOM.F90:1624` with the
accumulated transports `uhtr`) updates the reservoir value `tres` per layer and face (`:6214-6234`):

```
flux_to_res = dir * uhr                    ! accumulated volume through the face, positive outward
face_area   = (h_interior + H_subroundoff) * dyCu
L_out = max(0, flux_to_res * InvL_out / face_area)     ! = distance moved out / 9 km
L_in  = min(0, flux_to_res * InvL_in  / face_area)     ! ≤ 0 on inflow
tres  = ( tres + L_out * T_interior − L_in * T_external ) / (1 + L_out − L_in)
```

(`a_in`, `a_out` vanish for nonzero length scales.) So the reservoir relaxes toward the interior
value on outflow and toward the external (GLORYS) value on inflow, each in proportion to the
distance the water moved relative to 9 km — an implicit update. `tres` is the tracer value carried by
inflowing fluxes (`flux = uh · tres`, `tracer/MOM_tracer_advect.F90:680-689, 704-713`) and the
exterior value in the reconstruction (`:544-546`). pyOCEAN has no reservoir; its open-boundary
tracers use the exterior profile directly (class b for parity, as the plan already says).

NWA12's `OBC_RESERVOIR_INIT_BUG = True` initialises the reservoir from the interior concentrations
at the start of a new run, ignoring the specified boundary values (`:764-767, 5622-5640`,
`core/MOM.F90:3851-3854`): the first inflow carries interior water until the relaxation acts.

## 8. What pyOCEAN must decide (for Tasks 5, 6, 10)

1. Tidal velocity: velocity from TPXO (NWA12) or transport / model depth (plan). Plan choice is
   physically better; parity differs (c).
2. Tidal velocity added to every layer (MOM6) vs to the depth mean only. Equivalent in the depth
   mean; differs in the baroclinic nudging target.
3. Nodal factors: fixed at `TIDE_NODAL_REF_DATE` with K&L first-order formulas (MOM6) vs OTPS at t.
4. Nudging switch by Orlanski phase speed (MOM6) vs flow direction (pyOCEAN 'radiating').
5. `OBC_HOR_INDEXING_BUG` half-cell shift in NWA12's segment data (d).
6. No ramp in NWA12.
7. SAL: β = 0.05 on total SSH and in the Flather speed; legacy interface shift in the PGF (d).
