# MOM6 horizontal grid from a supergrid: reference read for pyOCEAN curvilinear grids

MOM6 source: `~/pyOCEAN_refs/MOM6`, git `0e9a561` (2026-10-07; `git -C ~/pyOCEAN_refs/MOM6 rev-parse --short HEAD`).
All `file:line` references are relative to `~/pyOCEAN_refs/MOM6/src/` unless stated otherwise.
SIS2: `~/pyOCEAN_refs/SIS2`, git `c287b24`. NWA12 configuration:
`~/pyOCEAN_refs/CEFI-regional-MOM6/exps/NWA12.COBALT/` (CEFI repo git `36b8f49`). CEFI builds MOM6 from
the `dev/cefi` fork (`CEFI-regional-MOM6/.gitmodules`), not from GFDL `0e9a561`; line numbers below are
GFDL's. Read 2026-11-01 for the regional chain (spec
`docs/superpowers/specs/2026-11-01-regional-chain-design.md`, plan Task 2). This is a reference
implementation, not truth; disagreements with the obvious reading are flagged.

---

## 1. Entry point and NWA12 settings

`set_grid_metrics` (`initialization/MOM_grid_initialize.F90:60-120`) dispatches on `GRID_CONFIG`;
`"mosaic"` calls `set_grid_metrics_from_mosaic` (`:93`), then `set_derived_dyn_horgrid`
(`:108-110`, body `framework/MOM_dyn_horgrid.F90:467-513`) for reciprocals. It is called from
`MOM_initialize_fixed` (`initialization/MOM_fixed_initialization.F90:78`), which later sets f
(`:186`) and the rotation angle (`:201`), in that order.

NWA12 (`MOM_parameter_doc.short`): `GRID_CONFIG = "mosaic"` (`:50`), `GRID_FILE = "ocean_hgrid.nc"`
(`:58`), `NIGLOBAL = 775` (`:42`), `NJGLOBAL = 845` (`:45`), `REENTRANT_X = False` (`:40`);
`MOM_parameter_doc.all`: `ROTATION = "2omegasinlat"` (`:474`), `OMEGA = 7.2921E-05` (`:480`),
`GRID_ROTATION_ANGLE_BUGS = False` (`:482`), `RAD_EARTH = 6.378E+06` (`:238`, not used for the
metrics of a mosaic grid).

## 2. Supergrid layout (file `ocean_hgrid.nc`)

The supergrid has twice the model resolution. With `ni`, `nj` model cells it has `2ni+1 × 2nj+1`
nodes. Variables read by MOM6 and their position (`MOM_grid_initialize.F90:214, 239, 263-264, 291`):

| variable | dims (Python order) | MOM6 position | meaning |
|---|---|---|---|
| `x`, `y` | `(nyp, nxp) = (2nj+1, 2ni+1)` | `CORNER` | lon, lat of every supergrid node [degrees] |
| `dx` | `(nyp, nx) = (2nj+1, 2ni)` | `NORTH_FACE` | length of the supergrid edge from node `(j, i)` to `(j, i+1)` [m] |
| `dy` | `(ny, nxp) = (2nj, 2ni+1)` | `EAST_FACE` | length of the edge from node `(j, i)` to `(j+1, i)` [m] |
| `area` | `(ny, nx) = (2nj, 2ni)` | `CENTER` | area of supergrid cell `(j, i)` [m²] |
| `angle_dx` | `(nyp, nxp)` | — | **not read by MOM6** (§6) |

Node parity (0-based Python indices of `x`, `y`): T points at (odd j, odd i), u points at (odd j,
even i), v points at (even j, odd i), q (corner, vorticity) points at (even j, even i). This follows
from `MOM_grid_initialize.F90:222-254` with the Fortran global supergrid index starting at 0
(`tmpZ` lower bound `2*isd-2`, `:180`), so Fortran `tmpZ(k)` = Python `x[k]`. CEFI's tools use the
same parity: T = `x[1::2, 1::2]`, q = `x[::2, ::2]` (`tools/rivers/write_runoff_glofas.py:85-88`),
u = `[1::2, 0::2]`, v = `[0::2, 1::2]` (`tools/initial/write_glorys_initial.py:97-99`).

## 3. Metrics: which half-segments are summed

Notation: Python 0-based model indices. T cell `(j, i)`, `j ∈ [0, nj)`, `i ∈ [0, ni)`; u face
`(j, I)`, `I ∈ [0, ni]` (I = west face of cell I, so `I = ni` is the east edge); v face `(J, i)`,
`J ∈ [0, nj]`; q point `(J, I)`. MOM6's Fortran `i` (1-based T) = Python `i+1`; Fortran `I` (u) =
Python `I` (both start at 0 in symmetric memory). The translation of each Fortran line
(`tmpV = dx`, `tmpU = dy`; `tmpV(i,j)` = `dx[j, i-1]`, `tmpU(i,j)` = `dy[j-1, i]`) gives:

| MOM6 (`MOM_grid_initialize.F90`) | Python on the supergrid | pyOCEAN name | meaning |
|---|---|---|---|
| `dxT(i,j)` `:270` | `dx[2j+1, 2i] + dx[2j+1, 2i+1]` | `dx_t` | x-width of the T cell through its centre |
| `dyT(i,j)` `:271` | `dy[2j, 2i+1] + dy[2j+1, 2i+1]` | `dy_t` | y-height of the T cell through its centre |
| `dxCu(I,j)` `:275` | `dx[2j+1, 2I-1] + dx[2j+1, 2I]` | `dx_u` | distance T(i−1) → T(i) across the u face |
| `dyCu(I,j)` `:276` | `dy[2j, 2I] + dy[2j+1, 2I]` | `dy_u` | length of the u face (q → u → q) |
| `dxCv(i,J)` `:280` | `dx[2J, 2i] + dx[2J, 2i+1]` | `dx_v` | length of the v face |
| `dyCv(i,J)` `:281` | `dy[2J-1, 2i+1] + dy[2J, 2i+1]` | `dy_v` | distance T(j−1) → T(j) across the v face |
| `dxBu(I,J)` `:285` | `dx[2J, 2I-1] + dx[2J, 2I]` | `dx_q` | x-distance between the v points around q |
| `dyBu(I,J)` `:286` | `dy[2J-1, 2I] + dy[2J, 2I]` | `dy_q` | y-distance between the u points around q |
| `areaT(i,j)` `:296-297` | `(a[2j,2i] + a[2j+1,2i+1]) + (a[2j+1,2i] + a[2j,2i+1])` | `area_t` | four supergrid sub-cells (`a` = `area`) |
| `areaBu(I,J)` `:300-301` | `(a[2J-1,2I-1] + a[2J,2I]) + (a[2J,2I-1] + a[2J-1,2I])` | `area_q` | four sub-cells around q |
| `areaCu(I,j)` | **not from the supergrid**: `dxCu · dy_Cu` (`MOM_grid_initialize.F90:1334`) | `area_u` | see below |
| `areaCv(i,J)` | **not from the supergrid**: `dyCv · dx_Cv` (`MOM_grid_initialize.F90:1342`; 1341 in the first version of these notes, corrected in Task 11) | `area_v` | see below |

The parentheses in the area sums are MOM6's order of summation (pairs of diagonal sub-cells first);
reproduce them to be bit for bit. CEFI computes the T area with the same grouping
(`write_runoff_glofas.py:90`).

**Velocity-cell areas.** `areaCu` and `areaCv` are products, not sums of sub-cells, and they use the
*open* face length: `dy_Cu = mask2dCu · dyCu` (`:1332`) and `dx_Cv = mask2dCv · dxCv` (`:1339`), set
in `initialize_masks`. They are therefore zero on land faces and are recomputed if a channel width
is narrowed (`reset_face_lengths_*`, `initialization/MOM_shared_initialization.F90:747-751, 762-766`;
NWA12 `CHANNEL_CONFIG = "none"`, `MOM_parameter_doc.all:461`;
`initialization/MOM_fixed_initialization.F90:141-160`). `IareaCu = mask2dCu / areaCu` with the
Adcroft reciprocal (`MOM_grid_initialize.F90:1335`). A pyOCEAN `area_u` from four supergrid
sub-cells straddling the u point would differ from MOM6's `area_u` by O(curvature) — use the product
`dx_u · dy_u` for parity. `areaT` and `areaBu` are not masked.

**Derived quantities and clipping.** `set_derived_dyn_horgrid` clips negative lengths to 0 and forms
`IdxT = 1/dxT` etc. with `Adcroft_reciprocal` (1/x, or 0 when x = 0;
`framework/MOM_dyn_horgrid.F90:478-511`, `MOM_grid_initialize.F90:1173-1179`). If `areaBu ≤ 0` it is
reset to `dxBu · dyBu` (`MOM_dyn_horgrid.F90:508`); it is positive for any real supergrid.

**Halo / boundary values.** Before the sums, every supergrid array is extended into the halo by
`extrapolate_metric` (`MOM_grid_initialize.F90:221, 242, 266-267, 293`): missing values
(999 for x, y; 0 for dx, dy, area) are replaced by linear extrapolation `2 v(k±1) − v(k±2)` from the
interior (`:1139-1169`). Consequence for a regional grid: the metrics of the outermost faces
(`dxCu` at `I = 0` and `I = ni`, `dyCv` at `J = 0` and `J = nj`, the corner `areaBu`, `dxBu`, `dyBu`
on the edges) use half-segments **extrapolated** outside the file, not the parent grid's values.
When pyOCEAN cuts a window from the full NWA12 supergrid it has the true outside half-segments;
for MOM6 parity on the same cut (Task 10, criterion 1) pyOCEAN must extrapolate the same way, i.e.
cut first, then extrapolate. Class (c) choice; flag it in `grid_from_supergrid`.

## 4. Geographic positions

`geoLonT(i,j) = x[2j+1, 2i+1]`, `geoLonBu(I,J) = x[2J, 2I]`, `geoLonCu(I,j) = x[2j+1, 2I]`,
`geoLonCv(i,J) = x[2J, 2i+1]` (`MOM_grid_initialize.F90:222-233`), and the same for latitude
from `y` (`:243-254`). pyOCEAN: `lon_t/lat_t`, `lon_q/lat_q`, `lon_u/lat_u`, `lon_v/lat_v`.
The 1-D axis labels `gridLonT`, `gridLatT` (`:316-349`) are for diagnostics only.

## 5. Coriolis parameter (q points only)

`ROTATION = "2omegasinlat"` → `set_rotation_planetary`
(`initialization/MOM_shared_initialization.F90:485-510`):

    f(I,J) = (2.0 * omega) * sin((PI * geoLatBu(I,J)) / 180.)          (:504-506)

with `PI = 4 atan(1)` (`:502`) and `OMEGA` default 7.2921e-5 s⁻¹ (`:500-501`). f lives only at q
points (`G%CoriolisBu`, `MOM_fixed_initialization.F90:186`); MOM6 has no f at T points.
pyOCEAN `f_q` = this formula with the same association; pyOCEAN's existing `f_t` has no MOM6
counterpart (MOM6 Coriolis schemes average the four `CoriolisBu` around a point when they need it),
so `f_t` is a pyOCEAN choice (class c) and must not enter a parity comparison.

## 6. Rotation angle: MOM6 does not read `angle_dx`

`angle_dx` is never read by MOM6 or SIS2 (`grep -rn angle_dx ~/pyOCEAN_refs/MOM6/src` finds only
`.testing/tc4/gen_grid.F90`, a file writer). The angle is **computed from the corner (q-point)
coordinates** in `initialize_grid_rotation_angle`
(`initialization/MOM_shared_initialization.F90:571-626`), with `GRID_ROTATION_ANGLE_BUGS = False`
(NWA12) taking the second branch (`:604-622`):

```
pi_720deg = atan(1.0) / 180.0                         ! = (pi/180)/4
len_lon   = 360.0 (or G%len_lon if > 0)
for each T cell (i,j), with corners B(I-1,J-1), B(I,J-1), B(I-1,J), B(I,J):
  lonB(m,n) = modulo_around_point(geoLonBu(I+m-2,J+n-2), geoLonT(i,j), len_lon)   ! m,n = 1,2
  lon_scale = cos(pi_720deg*((latB(I-1,J-1) + latB(I,J)) + (latB(I,J-1) + latB(I-1,J))))
  angle = atan2( lon_scale*((lonB(1,2) - lonB(2,1)) + (lonB(2,2) - lonB(1,1))),
                 (latB(I-1,J) - latB(I,J-1)) + (latB(I,J) - latB(I-1,J-1)) )
  sin_rot(i,j) = sin(angle) ; cos_rot(i,j) = cos(angle)
```

`modulo_around_point(x, xc, L) = modulo(x − (xc − L/2), L) + (xc − L/2)` (`:629-642`) removes the
±360° wrap, so atan2 of differences is safe here (contrary to plan risk 2, which assumed it is not).
`lon_scale` is cos of the mean corner latitude (the four latitudes summed, times π/720).

**Sign convention.** The code comment calls `angle` "the clockwise angle from lat/lon to ocean grid
(e.g. angle of ocean "north" from true north)" (`:619-620`). The numerator is the eastward
displacement and the denominator the northward displacement along the grid's j direction (the
diagonals' sum), so `angle > 0` when grid-north points east of true north, i.e. grid-x points
south of east. FRE-NCtools' `angle_dx` (as used by CEFI, `tools/boundary/boundary.py:18-42`) is the
**counter-clockwise** angle of grid-x from true east. Hence

    angle_MOM6 ≈ −angle_dx,   cos_rot ≈ cos(angle_dx),   sin_rot ≈ −sin(angle_dx).

Checked numerically (scratch script, 2026-11-01): a 5 km grid rotated by angle_dx = +20° at 35°N
gives `angle_MOM6 = −19.987°`. The two are **not equal to round-off**: MOM6's value is a finite-difference
estimate on a sphere with a mean-latitude `cos`, `angle_dx` comes from the grid generator. The
difference (here 0.013°) is O(Δ²) in the grid spacing and curvature.

**Rotation formulas actually applied** (all with MOM6's `cos_rot`, `sin_rot` at T points):
- lat/lon → grid (wind stress, SIS2 `SIS_fast_thermo.F90:251-252`):
  `x = u cos_rot − v sin_rot`, `y = v cos_rot + u sin_rot`;
- grid → lat/lon (surface currents to the coupler, SIS2 `ice_model.F90:1214-1219`):
  `u = x cos_rot + y sin_rot`, `v = y cos_rot − x sin_rot`.

With `cos_rot = cos α`, `sin_rot = −sin α` (α = `angle_dx`) the first is CEFI's `rotate_uv`
(`urot = cos α u + sin α v`, `vrot = −sin α u + cos α v`, `boundary.py:40-41`). So CEFI's
preprocessed inputs (OBC velocities, tidal ellipses, initial u, v) are rotated with `angle_dx`, while
the online coupling (wind stress, currents) is rotated with MOM6's corner-based angle. NWA12 itself
mixes the two conventions; the mismatch is O(0.01°), class (d) in the reference model.

**Consequence for pyOCEAN Task 2 / Task 10 criterion 1.** `ocean_static.nc` `cos_rot`/`sin_rot`
(if written) come from §6's formula. A `grid_from_supergrid` that builds `cos_t`, `sin_t` from
`angle_dx` will not match them to 1e-12. Either compute the angle with MOM6's corner formula (for
parity and for the online stress rotation) and keep `angle_dx` only to rotate CEFI-style
preprocessed inputs, or relax the criterion for the angle to the measured O(Δ²) difference. Decision
for Esteban; recommended: corner formula for `cos_t`, `sin_t` (it is what both MOM6 and SIS2 use
online), and a separate check `|angle_MOM6 + angle_dx| < tol` as a grid sanity test.

Also note the sign of pyOCEAN's planned `sin_t` ("rotation from grid-x to east", plan Task 2): it
must be stated as either `sin(angle_dx)` (= −`sin_rot`) or `sin_rot`; the two differ in sign.

## 7. How a regional window indexes the supergrid

MOM6 always reads the whole `GRID_FILE` and requires `NIGLOBAL = (nxp − 1)/2`,
`NJGLOBAL = (nyp − 1)/2` (the supergrid domain is cloned with `refine=2`,
`MOM_grid_initialize.F90:209-210`). A cut for MOM6 is therefore a new, smaller `ocean_hgrid.nc`.
For model cells `i ∈ [i0, i1)`, `j ∈ [j0, j1)` (Python, end exclusive), keep the window's corners:

| variable | slice |
|---|---|
| `x`, `y`, `angle_dx` | `[2*j0 : 2*j1 + 1, 2*i0 : 2*i1 + 1]` |
| `dx` | `[2*j0 : 2*j1 + 1, 2*i0 : 2*i1]` |
| `dy` | `[2*j0 : 2*j1, 2*i0 : 2*i1 + 1]` |
| `area` | `[2*j0 : 2*j1, 2*i0 : 2*i1]` |
| `ocean_topog.nc` `depth` | `[j0 : j1, i0 : i1]` |

The supergrid offsets must be **even** (`2*i0`, `2*j0`): an odd offset makes the first node a T or
face point and swaps the T/q roles of the whole grid. The new file then has `NIGLOBAL = i1 − i0`,
`NJGLOBAL = j1 − j0`. Open-boundary segment files must be rewritten for the cut (they are on the
supergrid edge nodes, `mom6_obc_tides.md` §2), and the edge metrics follow §3 "Halo / boundary
values".

## 8. Mapping summary for `grid_from_supergrid`

| pyOCEAN | MOM6 | location / dims (pyOCEAN) |
|---|---|---|
| `dx_t`, `dy_t`, `area_t` | `dxT`, `dyT`, `areaT` | T, `(nj, ni)` |
| `dx_u`, `dy_u` | `dxCu`, `dyCu` | u, `(nj, ni+1)` |
| `area_u` | `areaCu = dxCu · dy_Cu` (masked) | u |
| `dx_v`, `dy_v` | `dxCv`, `dyCv` | v, `(nj+1, ni)` |
| `area_v` | `areaCv = dyCv · dx_Cv` (masked) | v |
| `dx_q`, `dy_q`, `area_q` | `dxBu`, `dyBu`, `areaBu` | q, `(nj+1, ni+1)` |
| `f_q` | `CoriolisBu` | q |
| `lon_*`, `lat_*` | `geoLon*`, `geoLat*` | all four |
| `cos_t`, `sin_t` | `cos_rot`, `sin_rot` (sign: §6) | T |

Open: whether `area_u`, `area_v` in pyOCEAN are used masked or unmasked (MOM6 masks them through
`dy_Cu`, `dx_Cv`); to be decided in Task 3's audit.
