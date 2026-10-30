# Momentum equations

This page describes how pyOCEAN advances the horizontal velocity of each layer: the advection of momentum, the
Coriolis force, the horizontal viscosity, and the implicit vertical viscosity with its two boundary fluxes (bed
drag and wind stress). The stratified model `SigmaBaroclinic` has two forms of the momentum equation:

- the **flux form** (momentum-conserving, WENO5-Z, as CROCO), the default with Eulerian σ-layers
    (`vertical_coordinate="sigma_eulerian"`);
- the **vector-invariant form** (potential-vorticity flux plus kinetic-energy gradient, as MOM6), the default
    in ALE mode (`vertical_coordinate="ale"`).

The pressure-gradient force is on its own page ([Pressure gradient](pressure-gradient.md)). The free surface,
the Adams–Bashforth extrapolation and the coupling between the depth mean and the layers are on
[Free surface and time stepping](free-surface-time-stepping.md). You need this page to choose `momentum`,
`ke_scheme`, `adv`, `ah`, `nu_const`, `thin_layer_coupling` and `max_truncations`, and to know what each one does.

## Physics

For each layer $k$ (thickness $h_k$, $k = 0$ at the bed) the hydrostatic Boussinesq momentum equation is

$$
\frac{\partial \mathbf{u}_k}{\partial t}
= \underbrace{-(\mathbf{u}\cdot\nabla)\mathbf{u}_k - \frac{\omega}{h}\frac{\partial \mathbf{u}}{\partial k}}_{\text{advection}}
\;-\; f\,\hat{\mathbf{z}}\times\mathbf{u}_k
\;-\; g\nabla\eta \;+\; \mathbf{PGF}_{bc,k}
\;+\; A_h\nabla^2\mathbf{u}_k
\;+\; \frac{\partial}{\partial z}\left(\nu\frac{\partial \mathbf{u}}{\partial z}\right),
$$

with the surface and bed boundary conditions

$$
\nu\frac{\partial \mathbf{u}}{\partial z}\Big|_{\eta} = \frac{\boldsymbol{\tau}_s}{\rho_0},
\qquad
\nu\frac{\partial \mathbf{u}}{\partial z}\Big|_{-H} = C_d\,|\mathbf{u}_b|\,\mathbf{u}_b .
$$

In the vector-invariant form the advection and Coriolis terms are rewritten with the identity
$(\mathbf{u}\cdot\nabla)\mathbf{u} = (\zeta\,\hat{\mathbf z})\times\mathbf{u} + \nabla K$, $K = \tfrac12|\mathbf u|^2$:

$$
\frac{\partial u}{\partial t} = q\,(vh) - \frac{\partial K}{\partial x} + \dots,\qquad
\frac{\partial v}{\partial t} = -q\,(uh) - \frac{\partial K}{\partial y} + \dots,\qquad
q = \frac{f+\zeta}{h}.
$$

In ALE mode there is no vertical advection term: the layers move with the flow during the Lagrangian step and
the velocity is then remapped onto the new grid ([Vertical coordinates and ALE](vertical-ale.md)).

### Which terms go where in the time step

| Term | Treatment | Where |
|---|---|---|
| Advection, Coriolis, baroclinic PGF, $A_h\nabla^2\mathbf u$ | explicit, collected in $G_k$, extrapolated with AB3 | `SigmaBaroclinic._tendencies` (`core/dynamics/sigma_baroclinic.py:1511`) |
| $-g\nabla\eta$ | θ-implicit, on the depth mean | `WetDryFreeSurface.step` (`core/dynamics/wetting_drying.py:227`) |
| Vertical viscosity, bed drag, wind stress | implicit (backward Euler) per column | `SigmaWetDry.column_step` (`core/dynamics/sigma_wd.py:187`) |

The wind stress and the bed drag are boundary fluxes of the implicit vertical viscosity. They do not pass through
the AB3 extrapolation (`SigmaBaroclinic.step`, `core/dynamics/sigma_baroclinic.py:2084-2101`; as ROMS/CROCO
`sustr`/`svstr`).

## Grid and notation

The grid is an Arakawa C grid ([Grids and bathymetry](grids.md)). Arrays are (k, j, i) with $k = 0$ at the bed:

| Symbol | Code | Shape | Location |
|---|---|---|---|
| $u$ | `st.u` | (N, ny, nx+1) | u-faces |
| $v$ | `st.v` | (N, ny+1, nx) | v-faces |
| $h$ | `st.h` (ALE) or $D\,\Delta\sigma_k$ (σ) | (N, ny, nx) | cells |
| $Q^x, Q^y$ | `st.Qx`, `st.Qy` | as $u$, $v$ | layer volume transports [m³ s⁻¹] of the last step |
| $F^z$ | `st.Fz` | (N+1, ny, nx) | volume flux through interface $k$ [m³ s⁻¹] (0 at bed and surface; ≡ 0 in ALE) |
| $f_q$ | `h_grid["f_q"]` | (ny+1, nx+1) | cell corners |
| $\Delta x_u, \Delta y_v$ | `dx_u`, `dy_v` | faces | centre-to-centre distances across the face |
| $L_u, L_v$ | `dy_u`, `dx_v` | faces | face lengths |
| $A_t, A_u, A_v$ | `area_t`, `area_u`, `area_v` | | cell and face areas |

Corner $[J, I]$ is shared by cells $(J-1, J)\times(I-1, I)$ (`core/dynamics/vector_invariant.py:12-13`).

## (a) Flux form (σ default)

### Momentum-conserving advection with WENO5-Z

`_flux_form_u` (`core/dynamics/sigma_baroclinic.py:706`, kernel `_flux_form_u_kernel` at `:627`, NumPy oracle
`_flux_form_u_numpy` at `:714`) computes the advection of $u$ by the layer transports, as ROMS/CROCO
(Shchepetkin & McWilliams 2005):

$$
\left.\frac{\partial u}{\partial t}\right|_{adv}
= -\frac{1}{V_u}\Big[\delta_x(\mathcal U\,\tilde u) + \delta_y(\mathcal V\,\tilde u) + \delta_k(\mathcal W\,\tilde u)
- u\,\big(\delta_x\mathcal U + \delta_y\mathcal V + \delta_k\mathcal W\big)\Big].
$$

The control volume is the u-cell, from the centre of cell $i-1$ to the centre of cell $i$. Its transports are
averages of the layer transports of the tracer cells:

| Transport | Through | Stencil (u-face $(k,j,i)$) |
|---|---|---|
| $\mathcal U_c = \tfrac12(Q^x_{k,j,c} + Q^x_{k,j,c+1})$ | cell centre $c$ (between u-faces $c$ and $c+1$) | $c = i-1, i$ |
| $\mathcal V_{J} = \tfrac12(Q^y_{k,J,i-1} + Q^y_{k,J,i})$ | corner $(J, i)$ | $J = j, j+1$ |
| $\mathcal W_{kk} = \tfrac12(F^z_{kk,j,i-1} + F^z_{kk,j,i})$ | interface $kk$ at the u-face | $kk = k, k+1$ |
| $V_u = \tfrac12(V_{k,j,i-1} + V_{k,j,i})$ | volume of the u-cell | $V = h A_t$ |

$\tilde u$ is the WENO5-Z value of $u$ at each control-volume face, taken upwind of the transport through it
(same reconstruction as for tracers, see [Tracers](tracers.md#horizontal-schemes); momentum always uses the Z
variant, `True` in `_wval`). The stencil is five u-points along the flux direction; it drops to WENO3 when only
three are wet and to first-order upwind next to walls (`_wval`, `:176`).

Subtracting $u$ times the mass divergence of the u-cell turns the conservative flux divergence into a velocity
tendency, while the momentum flux itself stays conservative. That flux balance sets the speed of fronts and
bores (Borden & Meiburg 2013; Ungarish et al. 2018, cited in the docstring at `:726`): in the lock exchange the
advective form gave a front 16 % slow, the flux form gives the CROCO front positions within one cell
(`validation/lock_exchange/README.md`).

The $v$ tendency is the same function on swapped axes (`_tendencies`, `:1526-1528`). In the σ mode the term is
skipped on the very first step, because the transports of a previous step do not exist yet (`st.Qx is None`,
`:1522`).

!!! note "Vanished-cell limiter (flux form in ALE only)"
    With `momentum="flux"` in ALE mode a u-cell can be nearly empty, and $Q\,\Delta t/V_u \gg 1$. The code then
    replaces the volume by

    $$
    V_u \leftarrow \max\big(V_u,\; 2\Delta t\,\max(Q_{in}, Q_{out})\big),
    $$

    with $Q_{in}$, $Q_{out}$ the transports into and out of the u-cell (`:696-701`). A vanished cell flooded by a
    current then relaxes to the current's velocity at a rate of at most $1/(2\Delta t)$, inside the AB3 stability
    limit; cells with $u\Delta t/\Delta x \le \tfrac12$ are unchanged. It has the purpose of MOM6's CFL truncation
    (`MOM_vert_friction.F90:2555`) but is continuous (docstring at `:727-735`).

### Advective forms `adv="upwind3"` and `adv="centred"`

`adv` selects the advection of the flux-form branch. With `adv="upwind3"` or `"centred"` the model uses the
advective operators of the 2-D nonlinear module, layer by layer (`_tendencies`, `:1529-1532`):

$$
(\mathbf u\cdot\nabla)u = u\,\partial_x u + \bar v\,\partial_y u,
\qquad \bar v = \tfrac14\,(v_{j,i-1} + v_{j+1,i-1} + v_{j,i} + v_{j+1,i}),
$$

(`advection_u`, `core/dynamics/nonlinear.py:165`; `advection_v`, `:178`). The derivative is

- `"centred"`: $(u_{+1} - u_{-1})/(h_- + h_+)$ (`_centred`, `:125`), zero where a neighbour is dry;
- `"upwind3"`: third-order upwind-biased (Leonard 1979, `_upwind3`, `:132`),

    $$
    c>0:\ \frac{2u_{+1} + 3u_0 - 6u_{-1} + u_{-2}}{6h},\qquad
    c<0:\ \frac{-u_{+2} + 6u_{+1} - 3u_0 - 2u_{-1}}{6h},
    $$

    that is the centred derivative plus a hyperdiffusion $|c|h^3/12\,\partial^4 u$. Where the 4-point stencil
    reaches land it falls back to first-order upwind, which stays dissipative.

Vertical advection is then added separately, centred in layer index (`_vertical_advection`, `:1503`):

$$
\frac{\omega}{h}\frac{\partial u}{\partial k}\Big|_k
= \frac{\omega_{k+\frac12}(u_{k+1} - u_k) + \omega_{k-\frac12}(u_k - u_{k-1})}{2\,h_k},
$$

with $\omega$ of the last step averaged to the face and $h_k = D_u\,\Delta\sigma_k$. Metric terms
$uv\tan\varphi/R$ are neglected (`core/dynamics/nonlinear.py:16-18`).

### Energy-conserving Coriolis

`_coriolis` (`core/dynamics/sigma_baroclinic.py:1467`) computes $+f\bar v$ at u-faces and $-f\bar u$ at
v-faces with $f$ at the corners. Each adjacent (u, v) pair shares one corner $q$ and is weighted by
$\tfrac14 f_q$:

$$
\mathrm{Cor}^u_{k,j,I} = \tfrac14\Big[f_{q;\,j,I}\,(v_{k,j,I-1} + v_{k,j,I}) + f_{q;\,j+1,I}\,(v_{k,j+1,I-1} + v_{k,j+1,I})\Big],
$$

$$
\mathrm{Cor}^v_{k,J,i} = -\tfrac14\Big[f_{q;\,J,i}\,(u_{k,J-1,i} + u_{k,J,i}) + f_{q;\,J,i+1}\,(u_{k,J-1,i+1} + u_{k,J,i+1})\Big].
$$

The operator is antisymmetric, so it does no work (Sadourny 1975; Arakawa & Lamb 1977, as cited in the
docstring). Strictly, the antisymmetry holds for the unweighted sum $\sum u\,\mathrm{Cor}^u + \sum v\,\mathrm{Cor}^v$;
the kinetic energy $\sum h u^2$ is conserved exactly only when the face areas and thicknesses are uniform.

## (b) Vector-invariant form (ALE default)

`momentum="vector_invariant"` is the default when `vertical_coordinate="ale"` and is not allowed with Eulerian σ
(`_setup_ale`, `:1130-1135`). `_vector_invariant` (`:1483`) calls the kernels of
`core/dynamics/vector_invariant.py`, a port of MOM6 `MOM_CoriolisAdv.F90` (commit 0e9a561). Why: the flux form
divides by the volume of the momentum cell and accelerates the nose of a gravity current in a layer that thins
(hybrid lock exchange, `validation/mom6_lock_exchange`); Bleck & Smith (1990, app. B) and Higdon (2005) describe
the problem, and MOM6 and HYCOM use this form (`core/dynamics/vector_invariant.py:7-10`).

### Relative vorticity and potential vorticity at the corners

`vorticity_and_pv` (`core/dynamics/vector_invariant.py:257`, oracles `vorticity_q_numpy` `:40` and `pv_q_numpy`
`:63`):

$$
\zeta_{J,I} = \frac{\big(v_{J,I}\,\Delta y_{v;J,I} - v_{J,I-1}\,\Delta y_{v;J,I-1}\big) - \big(u_{J,I}\,\Delta x_{u;J,I} - u_{J-1,I}\,\Delta x_{u;J-1,I}\big)}{A_{q;J,I}}
$$

in circulation form. **Free slip:** $\zeta = 0$ at every corner that is not surrounded by four wet cells
(`corner_mask`, `:27`; MOM6 `mask2dBu`, `MOM_CoriolisAdv.F90:595-610`). The corner area is
$A_q = \overline{\Delta x_u}^{\,y}\;\overline{\Delta y_v}^{\,x}$ (`corner_area`, `:33`).

$$
q_{J,I} = \frac{(f_{q;J,I} + \zeta_{J,I})\;\sum A_t}{\sum A_t\,h + V_{neglect}},
\qquad V_{neglect} = 10^{-28}\ \mathrm{m^3},
$$

with the sums over the wet cells around the corner (`MOM_CoriolisAdv.F90:337, 364-367, 429-437, 625-629`;
`V_NEGLECT` at `core/dynamics/vector_invariant.py:24`).

### Sadourny (1975) energy-conserving PV flux

`coriolis_pv` (`:338`, oracle `coriolis_pv_numpy` `:74`; `MOM_CoriolisAdv.F90:818-823, 1071-1076`):

$$
CA^u_{j,I} = \frac{1}{4\,\Delta x_{u;j,I}}\Big[q_{j+1,I}\big(Q^y_{j+1,I-1} + Q^y_{j+1,I}\big) + q_{j,I}\big(Q^y_{j,I-1} + Q^y_{j,I}\big)\Big],
$$

$$
CA^v_{J,i} = -\frac{1}{4\,\Delta y_{v;J,i}}\Big[q_{J,i}\big(Q^x_{J-1,i} + Q^x_{J,i}\big) + q_{J,i+1}\big(Q^x_{J-1,i+1} + Q^x_{J,i+1}\big)\Big],
$$

in each layer, with $Q^x, Q^y$ the layer volume transports [m³ s⁻¹]. These are the transports of the last
Lagrangian step (after the outflow limiter), as MOM6 uses the `uh` of its last continuity call; on the first step,
or after a restart without them, $Q^x = L_u\,u\,\bar h_u$ with the mean face thickness
(`_vector_invariant`, `core/dynamics/sigma_baroclinic.py:1494-1497`). Domain-edge faces get 0.

**BOUND_CORIOLIS** (always on, `bound=True` at `:1498`; MOM6 `:1008-1023, 1260-1275`): $CA^u$ is clipped to the
range of the four products

$$
(f+\zeta)_{j+1,I}\,v_{j+1,I},\quad (f+\zeta)_{j+1,I}\,v_{j+1,I-1},\quad (f+\zeta)_{j,I}\,v_{j,I},\quad (f+\zeta)_{j,I}\,v_{j,I-1},
$$

and $CA^v$ to the range of $-(f+\zeta)\,u$ at its four neighbours. Reason: $q\,vh \approx (f+\zeta)\,v\,h_{face}/h_q$
grows without bound when the corner thickness $h_q$ is much smaller than the face thickness (a vanished layer).

### Kinetic energy and its gradient

`kinetic_energy` (`:371`, oracle `kinetic_energy_numpy` `:106`; MOM6 `gradKE`, `MOM_CoriolisAdv.F90:1377-1535`):

=== "ke_scheme = 'arakawa' (default)"

    $$
    K_{j,i} = \frac{A_{u,E}\,u_E^2 + A_{u,W}\,u_W^2 + A_{v,N}\,v_N^2 + A_{v,S}\,v_S^2}{4\,A_{t;j,i}}
    $$

    Energy conserving; the MOM6 production choice (OM4/OM5 and every MOM6 example except three,
    `validation/mom6_ke_schemes/README.md`).

=== "ke_scheme = 'gudonov'"

    $$
    K_{j,i} = \frac{\max\big(u_m^2 A_{u,E},\; u_p^2 A_{u,W}\big) + \max\big(v_m^2 A_{v,N},\; v_p^2 A_{v,S}\big)}{2\,A_{t;j,i}},
    \quad u_p = \max(u_W, 0),\ u_m = \min(u_E, 0)
    $$

    (and the same for $v$): only the velocities that **enter** the cell count (upwind). The spelling
    `'gudonov'` follows MOM6's `KE_GUDONOV` and is the one the code accepts.

The gradient is $-\partial K/\partial x = -(K_{j,i} - K_{j,i-1})/\Delta x_{u;j,i}$ at u-faces and the same in $y$
(`ke_gradient`, `:204`); domain-edge faces get 0.

!!! tip "Which kinetic-energy scheme"
    | Use | Scheme | Evidence |
    |---|---|---|
    | Internal waves, tides, general use | `'arakawa'` (default since 2026-10-12) | `'gudonov'` damps progressive internal waves by ~9 % of amplitude per 5 km at $\Delta x$ = 100 m (8 % at 50 m: first order in $\Delta x$). With Arakawa, mode-1 $u$ is 0.91–1.01 of MITgcm's (`validation/internal_wave/README.md`). |
    | Density currents, bores, lock exchange | `'gudonov'` | ALE-σ lock exchange (20 layers) fronts 0.98 / 0.98 $U$ with Godunov against 0.91 / 0.84 $U$ with Arakawa (`validation/mom6_lock_exchange/README.md`); MOM6 itself uses Godunov only in `lock_exchange`, `adjustment2d` and `external_gwave`. |

    In the Ilicak internal wave, with the old transport weighting, switching to Godunov dropped the kinetic energy
    left at days 40–60 from 0.58 to 0.35 (`validation/ilicak_benchmark/README.md`, item 9).

### CFL velocity truncation (backstop)

After the remap of the face velocities, `truncate_cfl` (`core/dynamics/vector_invariant.py:172`, oracle
`truncate_cfl_numpy` `:136`; MOM6 `vertvisc_limit_vel`, `MOM_vert_friction.F90:2555-2715`) clips any layer
velocity whose donor-cell Courant number exceeds ½:

$$
\frac{|u|\,\Delta t}{\Delta x_{donor}} > \tfrac12
\;\Rightarrow\;
u \leftarrow \operatorname{sign}(u)\;0.9\cdot\tfrac12\,\frac{\Delta x_{donor}}{\Delta t},
$$

with $\Delta x_{donor}$ the width of the cell the water comes from (west cell for $u \ge 0$). Truncations at faces
thicker than `h_thin_bed` (default 0.01 m) are counted in `SigmaState.n_trunc` (MOM6 counts faces thicker than
3 Å). When the running count exceeds `max_truncations` (default 1000) the step raises `RuntimeError` naming the
first layer and face (`_step_ale`, `core/dynamics/sigma_baroclinic.py:2063-2079`). The truncation should never
act in a healthy run; a growing `n_trunc` signals a time step too long or a layer that has lost its dynamics.

### Vertical advection in ALE

None. In the Lagrangian step the interfaces move with the flow ($F^z \equiv 0$, `:2036`). The vertical
redistribution of momentum is done by the conservative remap of $u$, $v$ in face-fraction space, which keeps
$\sum_k w_k u_k$ exact (`_remap_faces`, `:1367`). See [Vertical coordinates and ALE](vertical-ale.md).

## (c) Viscosity, bed drag and surface stress

### Horizontal Laplacian viscosity `ah`

`ah` [m² s⁻¹] adds $A_h\nabla^2 u$ to $G$ in both momentum forms (`_tendencies`, `:1533-1538`; kernel
`laplacian_3d`, `core/dynamics/nonlinear.py:224`):

$$
\nabla^2 u\big|_{j,i} = \sum_{\text{axis}}\frac{2}{h_- + h_+}\left(\frac{u_+ - u_0}{h_+} - \frac{u_0 - u_-}{h_-}\right),
$$

a non-uniform 3-point stencil per direction, with the u-to-u spacings of `_spacings_u` (`:99`: $\Delta x_t$ in $x$,
the mean of $\Delta y_u$ in $y$). **Free slip:** a dry or missing neighbour is replaced by $u_0$, so there is no
gradient towards land. The operator acts on $u$ itself, not thickness-weighted ($\nabla\cdot(h\nabla u)/h$).
Default `ah = 0`.

Guide: the grid Reynolds number $Re_\Delta = U\Delta x/A_h$ controls both the noise and the spurious mixing
(Ilicak et al. 2012). In the lock exchange the RPE gain fell from 3.22e-5 ($A_h = 0$) to 1.26e-5
($A_h = 100$, $Re_\Delta = 2.5$) (`validation/lock_exchange/README.md`).

### Smagorinsky horizontal viscosity `smagorinsky`

`smagorinsky = C` (dimensionless, MOM6 `SMAG_LAP_CONST`; 0.15 is the OM4p5 value) replaces the constant Laplacian above
by the Laplacian Smagorinsky viscosity in thickness-weighted stress-tensor form, one layer at a time and along the
layer (σ, z\* and hybrid alike), as MOM6 (`MOM_hor_visc.F90`). It is off by default and then bit for bit as before
(`tests/test_smagorinsky_model.py::test_zero_smagorinsky_is_bitwise`). Kernels and equations: `core/dynamics/hor_visc.py`
(module docstring `:1-67`, NumPy oracle `smagorinsky_numpy` `:116`, Numba kernel `:219`, driver `smagorinsky` `:341`);
the model calls it from `Momentum.smagorinsky_terms` (`core/components/momentum.py:204`), which replaces the `hviscosity`
term (`:272-274`). Smagorinsky (1963, 1993); Griffies & Hallberg (2000).

**Strain rates** on the C grid (tension at cell centres, shearing strain at the corners; `MOM_hor_visc.F90:707-722`):

$$
D_T = \frac{\partial u}{\partial x} - \frac{\partial v}{\partial y}, \qquad
D_S = \frac{\partial v}{\partial x} + \frac{\partial u}{\partial y},
$$

with the metric forms of MOM6, e.g. $\partial u/\partial x|_h = (\Delta y_T/\Delta x_T)\,(u_{i+1/2}/\Delta y_{u,i+1/2} - u_{i-1/2}/\Delta y_{u,i-1/2})$.

**Shear magnitude** without the divergence (MOM6 and MITgcm; `MOM_hor_visc.F90:1018-1022, 1355-1359`), with the squares of
the four surrounding values of the other strain averaged:

$$
|S|_h = \sqrt{D_T^2 + \tfrac14\textstyle\sum_4 D_S^2}, \qquad |S|_q = \sqrt{D_S^2 + \tfrac14\textstyle\sum_4 D_T^2}.
$$

**Viscosity**, evaluated separately at cell centres ($\nu_h$) and at corners ($\nu_q$), with the harmonic-mean grid scale
(`:3382-3389, 3403-3414, 1060-1070`):

$$
\Delta^2 = \frac{2\,\Delta x^2\Delta y^2}{\Delta x^2 + \Delta y^2}, \qquad \nu = \max\!\big(A_h,\; C\,\Delta^2\,|S|\big).
$$

`ah` is the background (the maximum, not a sum: MOM6 `ADD_LES_VISCOSITY = False`). $C$ plays the role of the usual
$C_s^2$, since the constant multiplies the area directly (Ilicak et al. 2012 write $\nu = (C_s\Delta)^2|S|$, eq. 11: their
$C_s = 1/\sqrt2$ is $C = 0.5$). The grid has no corner spacings: $\Delta x_q$ is the mean of the two $\Delta x_v$ next to
the corner, $\Delta y_q$ the mean of the two $\Delta y_u$ (`smag_geometry`, `:74`).

**Stability bound, rescaled for AB3.** The viscous term is explicit in the AB3 momentum step. MOM6's `BOUND_KH`
(`:3507-3529, 1121-1133, 1500-1512`) divides $0.8\cdot 0.25/\Delta t$ by the **maximum** of the two per-direction metric
terms ($\approx 2/\Delta x^2$ and $2/\Delta y^2$, `:3509-3513`), not by their sum; on a uniform square grid that is
$0.1\,\Delta x^2/\Delta t$, 0.8 of the forward-Euler limit. pyOCEAN's sum form below agrees with it on a square grid and is
more conservative on anisotropic cells. AB3 has a smaller stability interval on the negative real axis,
$\lambda\Delta t \le 6/11$ (Durran 2010), so with MOM6's bound the limited $\nu$ would be unstable by about 1.5 times.
pyOCEAN applies the bound to the final $\nu$ at $h$ and $q$ points,

$$
\nu \le \frac{0.8\cdot(6/11)}{4\,\Delta t\,(1/\Delta x^2 + 1/\Delta y^2)}\;h_{rat,\min}, \qquad
h_{rat,\min} = \min\!\Big(1,\ \frac{h_{\min}}{h_{point} + h_{neglect}}\Big),
$$

about $0.055\,\Delta x^2/\Delta t$ on a uniform grid (`BOUND_COEF`, `AB3_STABILITY` `:70-71`, `_bound_factor` `:110`).
$h_{\min}$ is the thinnest of the four faces around the point (`:1033-1037, 1368-1371`), the **thin-layer factor**: a
layer that is thin next to a thick neighbour gets a proportionally smaller $\nu$. At $h$ points only wet faces enter
$h_{\min}$ (MOM6 uses the land-masked half thickness; a land face receives no tendency, so it does not constrain the
step). In ALE (z\*, σ, hybrid) the face thickness of the operator is MOM6's unclipped land-masked mean
$h_u = \tfrac12(h_L + h_R)$ at wet faces, 0 at land and dry faces (`MOM_hor_visc.F90:769-782`;
`Momentum._unclipped_face_thickness`), used for $h_{\min}$, the $h_q$ harmonic form and the division by $h_u$. It is
not the bed-clipped `mean_face_thickness` of the transport: in z\*/hybrid that face is ~0 for a layer below the
shallower bed of a step, which drove $\nu$ (the background `ah` too) to ~0 in the deep cell next to every bathymetric
step (final review I1; `test_background_viscosity_kept_at_zstar_step`). A layer below the shallower bed is a wall for
that layer, as a land face. With the unclipped mean, $h_{rat,\min} \ge \tfrac12$ next to a vanished neighbour; it goes
to 0 only where the layer vanishes on both sides of a face. The bound is always on when Smagorinsky is on. Points where it strictly reduced $\nu$ are counted in
`SigmaState.n_visc_limited` (see the caveat below).

**Thickness-weighted stresses and their divergence** (`:1362-1367, 1727-1737, 1757-1781`):

$$
\sigma_{xx} = \nu_h\,h\,D_T, \qquad \sigma_{xy} = \nu_q\,h_q\,D_S,
$$

$$
h_q = \frac{2\,h_{2uq}\,h_{2vq}}{h_{neglect}^3 + (h_{2uq} + h_{2vq})(h_{u,s} + h_{u,n} + h_{v,w} + h_{v,e})},\quad
h_{2uq} = 4h_{u,s}h_{u,n},\ h_{2vq} = 4h_{v,w}h_{v,e},
$$

so $h_q \to 0$ when any of the four faces vanishes (a vanished layer shuts the shear stress). The tendency at a face is the
divergence of the stress divided by the **face** thickness plus a negligible thickness,

$$
F_u = \frac{1}{A_u\,(h_u + h_{neglect})}\left[\frac{\Delta y_{T,e}^2\sigma_{xx,e} - \Delta y_{T,w}^2\sigma_{xx,w}}{\Delta y_u}
+ \frac{\Delta x_{q,n}^2\sigma_{xy,n} - \Delta x_{q,s}^2\sigma_{xy,s}}{\Delta x_u}\right],
$$

and the same for $F_v$ with the signs of MOM6 (`:1757-1793`). `h_neglect` is `core.vertical.remap.H_NEGLECT`. The face
thickness is the same one used elsewhere in the step (`_smag_thickness`, `momentum.py:177`). By summation by parts
$\sum h_u u F_u A_u + \sum h_v v F_v A_v = -\sum \nu h D_T^2 A_T - \sum \nu_q h_q D_S^2 A_q \le 0$: the operator only removes
kinetic energy (`test_energy_dissipation_nonnegative`).

**Boundaries.**

- *Free slip.* $D_S = 0$ at corners that are not surrounded by four wet faces (land, and dry faces of wetting and drying);
  a dry or land face has $u = 0$ in the strains and receives no tendency (`test_drying_face_gets_no_viscous_tendency`).
- *Open edges.* The faces of an open boundary get no viscous tendency, and $D_S = 0$ at their corners (MOM6
  `OBC_STRAIN_ZERO`; `:1761-1771, 1783-1793`). The domain-edge faces are zero too
  (`test_open_edge_faces_get_no_tendency`).
- *Vanished layers.* $\nu$ is 0 where the cell has no wet face and at corners that are not active; layers with $h = 0$
  stay finite (`test_vanished_layers_finite_and_zero_nu`).
- *Periodic axes.* Pad–crop with halos of 2 cells (the stencil reaches the tension of the neighbour cell through the corner
  shear), geometry padded once (`smagorinsky_terms`). A doubly periodic box conserves momentum
  (`test_doubly_periodic_conserves_momentum`).

**Checks.** $\nu$ equals the MOM6 formula on a stretched grid (`test_nu_matches_mom6_formula_on_stretched_grid`); a shear
wave decays at the analytical rate with constant $\nu$; a quiet flow returns the background `ah`; Numba equals the NumPy
oracle for any thread count; σ and ALE with fixed layers agree; restart is bit for bit (`tests/test_hor_visc.py`,
`tests/test_smagorinsky_model.py`).

**Results** (details and classes of difference in the three README files):

| Case | Without | With Smagorinsky | Source |
|---|---|---|---|
| DOME hybrid + GLS, 10 km, 50 days, $C$ = 0.15 | 1 CFL truncation, 1072 limiter events, max\|u\| thick/thin faces 5.52/5.66 m s⁻¹ (MITgcm 2.04), $b_f$ −12.99e-3, $\alpha_E$ 2.29e-3 | 0 truncations, 1 limiter event, max\|u\| 3.54/3.10 m s⁻¹, $b_f$ −13.31e-3, $\alpha_E$ 1.83e-3, path 3–9 km further downslope; cost +2.4 % | `validation/dome/README.md` |
| Ilicak overflow, z\*, $A_h$ = 100 + $C$ = 0.5 (their Fig. 19b) | RPE gain 1.44e-5 | 7.46e-6 (−48 %), dRPE/dt −52 %, no new extrema: the paper's trend | `validation/ilicak_benchmark/README.md` |
| Lorenz D, σ + GLS, $\Delta t = T_{M2}/498$, $C$ = 0.15 | also completes 150 cycles (control), max\|u\| 1.46 m s⁻¹ | completes, max\|u\| 2.06 m s⁻¹; exact mixing relation 8.5e-12 | `validation/lorenz_estuary/README.md` |

The DOME and Ilicak rows (z\*/hybrid ALE) are the reruns after the I1 fix of the face thickness (commit 47f2730); the runs
before it lost the background viscosity next to every bathymetric step (class a). With $C > 0$ the background `ah` also
moves from the plain Laplacian to the thickness-weighted tensor operator, so each case has an **operator control** with
$C$ = 1e-9 ($C \approx 0$) that separates the two changes:

- *DOME:* the operator change alone lowers max\|u\| on thick faces from 5.52 to 3.56 m s⁻¹ and the limiter events from
  1072 to 641, but keeps the truncation and leaves $b_f$ at −13.05e-3. The Smagorinsky term removes the truncation and
  the remaining limiter events, lowers $\alpha_E$ by 26 % and moves $b_f$ to −13.31e-3, slightly further from Legg's −12.0e-3.
- *Ilicak:* the operator change alone moves the RPE gain by −1.4 %; the reduction is the Smagorinsky term (−47 % against
  the control).

Smagorinsky is therefore **not shown to extend the stable time step** in the Lorenz case (the control without it is also
stable), and production stays at $T_{M2}/746$. DOME with a low vertical viscosity (`kv_low`, $\nu_v$ = 1e-4, $A_h$ = 50,
$C$ = 0.15) completes 50 days, where MOM6 and MITgcm stop on day 3–4 without Smagorinsky. It is not robust: 61
truncations and 239 337 limiter events, and from day 37 the interior layers of the deep basin speed up to 8.29 m s⁻¹ at
day 50, which is 99.5 % of the truncation threshold, with a depth mean ≤ 0.93 m s⁻¹ (class b, missing shear-driven
vertical mixing). Cost per step with the term active (`scripts/smag_cost.py`, after the fix): BayIW cut z\* 1.041,
DOME 0.992 (noise) (`validation/bayiw_cut/README.md`).

**Caveat on `n_visc_limited`.** It counts every point where the bound strictly reduced $\nu$, including points where the
bound falls below the background `ah` (thin-layer factor, or a large `ah`), even if $C\Delta^2|S|$ is small. Read it as
"the bound acted", not as "Smagorinsky reached the bound". Before the I1 fix the bed-clipped faces at every z\* step
dominated it (74 fixed points per step in the z\* overflow, ≈ 3900 per step in DOME). With the unclipped face mean it is
0 in the z\* overflow and ≈ 14 points per step in DOME hybrid + GLS with $C$ = 0.15 (thin layers).

### Vertical viscosity: implicit, one tridiagonal per column

`SigmaWetDry.column_step` (`core/dynamics/sigma_wd.py:187`) solves, for each face column, the backward-Euler
problem (`vertical_implicit`, `:88`; Numba Thomas solver `_thomas_columns`, `:53`; NumPy reference
`_vertical_implicit_numpy`, `:99`):

$$
\Delta z_k\,\frac{\hat u_k - u^*_k}{\Delta t}
= \nu_{k+\frac12}\frac{\hat u_{k+1} - \hat u_k}{h_{k+\frac12}}
- \nu_{k-\frac12}\frac{\hat u_k - \hat u_{k-1}}{h_{k-\frac12}}
- \delta_{k,0}\; r\,\hat u_0
+ \delta_{k,N-1}\,\frac{\tau_s}{\rho_0},
$$

$$
u^*_k = u^n_k + \Delta t\,G_k^{AB3},\qquad h_{k+\frac12} = \tfrac12(\Delta z_k + \Delta z_{k+1}).
$$

As a matrix: lower and upper diagonals $-\Delta t\,\nu_{k\pm\frac12}/h_{k\pm\frac12}$, main diagonal
$\Delta z_k$ + the two couplings + $\Delta t\,r$ in the bed layer; right-hand side $\Delta z_k u^*_k$ +
$\Delta t\,\tau_s/\rho_0$ in the top layer. Columns are independent and solved in parallel. The solve conserves
$\sum_k \Delta z_k\,u_k$ apart from the two boundary fluxes.

- $\Delta z_k$: $D_f\,\Delta\sigma_k$ in σ ($D_f$ the face total depth, mean of the two cells,
    `face_depths`, `:246`); in ALE the face thickness `hu` passed by `_step_ale` (below).
- $\nu$: `closure="constant"` (`nu_const`, default 1e-4 m² s⁻¹ in `SigmaBaroclinic`), `"none"`, or in σ mode
    `"gls"` / `"parabolic"` ([Turbulence closures and bottom friction](turbulence-friction.md)). ALE allows only
    `"constant"` and `"none"` (`_setup_ale`, `:1151-1153`).
- With `closure="none"` and no bed drag the solve is skipped and the wind enters the top layer as
    $\Delta t\,\tau_s/(\rho_0\Delta z_{N-1})$ (`:231-234`).

### Bed drag and surface stress as boundary fluxes

- **Bed drag** (σ mode only): $r = C_d\,|u^*_0|$, explicit speed, implicit velocity, with the first-layer
    law of the wall $C_d = [\kappa/\ln(\Delta z_0/2z_0)]^2$ capped at `cd_max` (`first_layer_cd`,
    `physics/bottom_friction.py`; `:214-221`). It needs `z0` (default `None` in `SigmaBaroclinic` = no drag).
    **ALE refuses `z0`** because the bed layer can vanish under z* (`_setup_ale`, `:1148-1150`). Details on
    [Turbulence closures and bottom friction](turbulence-friction.md).
- **Surface stress:** `step(st, tau_x, tau_y)` takes $\tau$ [N m⁻²] at u- and v-faces; $\tau/\rho_0$ is the flux
    into the top layer (`:2100-2101`). An Ekman-transport test gives $\tau/\rho_0 f$ to 0.05 % (module docstring,
    `:36-37`). Bulk formulae: [Air–sea fluxes](air-sea-fluxes.md).

### Thin-layer coupling in the ALE viscous step

In ALE a layer can be nearly massless on one side of a face. With the mean face thickness
$\bar h = \tfrac12(q_W + q_E)$, a layer that has vanished on one side still looks thick in the viscous solve and is
barely coupled to its neighbours, so it can accelerate alone. `thin_layer_coupling` (default `"mom6"` in ALE,
`"none"` in σ; `:1049-1050`) passes instead MOM6's upwind-biased thickness to the viscous solve
(`viscous_face_thickness`, `core/vertical/ale.py:82`; `MOM_vert_friction.F90:1519-1533`; Hallberg & Rhines 1996):

$$
h^{visc}_k =
\begin{cases}
\dfrac{2\,q_a q_b}{q_a + q_b + h_{neglect}} & \text{if } u_k\,(q_b - q_a) > 0 \quad\text{(water leaves the thinner side)}\\[2ex]
\tfrac12\,(q_a + q_b) & \text{otherwise (thick water advances over thin)}
\end{cases}
$$

with $q_a$, $q_b$ the transport thicknesses of the west (south) and east (north) cells and
$h_{neglect} = 10^{-10}$ m. The harmonic mean ties a draining, nearly vanished layer to its neighbours (tie time
$\approx h_{thin}\,h_{interface}/\nu$, Higdon 2005 §3.1); the arithmetic mean where thick water advances does not
hold back a gravity-current nose (harmonic everywhere slowed the overflow plume from 10 to 2 m s⁻¹ at 4 h,
docstring `:96-98`). The docstring names the case it was built for as "overflow ν = 100: 2 → 24 m/s on faces
with a side < 1 cm" (`:93-95`); together with the outflow limiter it let the hybrid overflow finish at $\nu$ = 100
and 1000 (max $|u|$ 17 m s⁻¹, before 47; `validation/ilicak_benchmark/README.md`, item 4). The solver receives $\max(h^{visc}, 10^{-6}\ \mathrm{m})$ (`DZ_FLOOR`, `:63`). The bed drag would
keep the mean bed-layer thickness (`hu_drag`), and the ALE log-law drag (N3) uses it.

`_step_ale` (`core/dynamics/sigma_baroclinic.py:1997-2006`) passes `hu=max(h^visc, DZ_FLOOR)`,
`hu_drag=max(h̄_0, DZ_FLOOR)` and `hu_mean = h̄` to `SigmaWetDry.step`.

### Depth-mean correction δ of the coupled viscous step

The implicit solve conserves $\sum_k h^{visc}_k u_k$, but the free surface uses the depth mean
$\bar u = \sum_k w_k u_k$ with the fractions $w_k = \bar h_k/D_f$ of the mean thickness clipped at the shallower
bed (the "face column" of the free surface). With $h^{visc} \ne \bar h$, internal friction would change $\bar u$,
and the wind would enter $\bar u$ as $\Delta t\,\tau/h^{visc}_{top}$ instead of $\Delta t\,\tau/D_f$. The review
fix of 2026-10-14 (`column_step`, `core/dynamics/sigma_wd.py:237-243`) adds a uniform correction $\delta$ to the
solved column so that the depth mean obeys its own balance:

$$
\sum_k w_k(\hat u_k + \delta) = \sum_k w_k u^*_k + \frac{\Delta t}{D_f}\Big(\frac{\tau_s}{\rho_0} - r\,(\hat u_0 + \delta)\Big),
$$

$$
\delta = \frac{\sum_k w_k u^*_k + \Delta t\,\tau_s/(\rho_0 D_f) - \sum_k w_k \hat u_k - \Delta t\,r\,\hat u_0/D_f}{\sum_k w_k + \Delta t\,r/D_f},
\qquad D_f = \max\Big(\sum_k \bar h_k,\ d_{min}\Big).
$$

The shear $\hat u_k - \hat u_{k'}$ is kept; the bed stress is implicit at $\hat u_0 + \delta$. It is applied only
when `hu_mean` is passed (ALE with `thin_layer_coupling="mom6"`). Tests: a 1 mm top layer with a 0.1 N m⁻² wind,
where the coupling first gave η = 6.5e-3 m against 1.4e-4 m after one step
(`test_thin_layer_coupling_keeps_the_wind_driven_depth_mean`), and a 20 → 12 m step where the first version
divided τ by the mean cell depth (16 m) instead of the clipped face column (12 m), which shifted the BayIW seiche
range by −2.5 % (`test_thin_layer_coupling_forces_the_depth_mean_on_the_clipped_face_column`,
`tests/test_sigma_ale.py:1427-1460`). The thin overflow faces are still up to 1.9× faster than thick ones at
$\nu$ = 1000; a candidate is $\delta \propto$ `visc_rem` as in MOM6 (`CLAUDE.md`, Ilicak benchmark row).

### Other velocity rules of the ALE step

- **Thin bed layers carry no velocity** (`zero_thin_bed_velocity`, `core/vertical/ale.py:950`; HYCOM
    `hybgen.F90:1757-1764`): the run of face layers thinner than `h_thin_bed` starting at the bed gets $u = 0$ and
    the removed $\sum w u$ is spread over the other layers in proportion to $w$, so $\bar u$ is unchanged.
    Applied after the free surface and again after the remap (`:2008-2009, 2061-2062`).
- **HYCOM thin-layer blending** (`thin_layer_cutoff`, default 0 = off; `blend_thin_layers`, `core/vertical/ale.py:122`;
    HYCOM `mod_momtum.F90:3055-3065`): from the top down, $u_k \leftarrow [u_k q + u_{above}(c - q)]/c$ with
    $q = \min(h_k, c)$ on the donor thickness, then a uniform correction restores $\sum w u$. Off by default
    because blending the vanished layer ahead of a gravity-current nose gives it the counter-flowing velocity of
    the layer above (hybrid lock exchange front 1.05 → 0.70 $U$ for any cutoff ≥ 0.01 m, docstring `:992-998`).

## (d) Legacy z-level model

The phase-1 z-level model (`core/dynamics/momentum.py`, `ModelState` with xarray fields) solves

$$
\partial_t u = +f\bar v - \frac{1}{\rho_0}\partial_x p + \nu_h\nabla^2 u,\qquad
\partial_t v = -f\bar u - \frac{1}{\rho_0}\partial_y p + \nu_h\nabla^2 v
$$

(`momentum_tendency`, `:430`; Vallis 2006 eq. 14.12). There is no momentum advection and no vertical mixing.
The Coriolis term averages $f$ to the face (two corners) and $v$ to the u-face (four points,
`_v_to_u`, `coriolis_u` `:198`, `coriolis_v` `:250`); the Laplacian is a uniform-grid 5-point stencil with
free-slip padding along the walls (`_laplacian_u`, `:146`). The free-surface tendency is
$\partial_t\eta = -\sum_k \Delta z_k\,\nabla\cdot\mathbf u_k$ (`eta_tendency`, `:302`) and $w$ is diagnosed
downward from $w = 0$ at the surface (`diagnose_w`, `:361`). It is advanced with AB2
([Free surface and time stepping](free-surface-time-stepping.md#legacy-ab2)). Use it only for the phase-1 tests;
new work uses `SigmaBaroclinic`.

## Options and defaults

| Option | Values | Default | Meaning | Where |
|---|---|---|---|---|
| `momentum` | `"flux"`, `"vector_invariant"` | `"flux"` in σ, `"vector_invariant"` in ALE | momentum form; vector-invariant needs ALE | `_setup_ale`, `:1130` |
| `adv` | `"flux"`, `"upwind3"`, `"centred"` | `"flux"` | advection of the flux-form branch | `_tendencies`, `:1521-1532` |
| `ke_scheme` | `"arakawa"`, `"gudonov"` | `"arakawa"` | kinetic energy of the vector-invariant form | `vi.kinetic_energy` |
| `max_truncations` | int | 1000 | ALE: stop after this many counted CFL truncations | `_step_ale`, `:2069` |
| `ah` | ≥ 0 [m² s⁻¹] | 0 | horizontal Laplacian viscosity, free slip; background (maximum) when `smagorinsky` > 0 | `:1533` |
| `smagorinsky` | ≥ 0 [-] | 0 (off) | Smagorinsky coefficient $C$; ν = max(`ah`, C Δ² \|S\|), AB3-bounded | `core/dynamics/hor_visc.py` |
| `closure` | `"constant"`, `"none"`, `"gls"`, `"parabolic"` | `"constant"` | vertical viscosity (ALE: constant/none) | `SigmaWetDry` |
| `nu_const` | [m² s⁻¹] | 1e-4 | constant vertical viscosity | `SigmaWetDry` |
| `z0` | [m] or `None` | `None` | bed roughness of the log-law drag (σ only) | `column_step` |
| `thin_layer_coupling` | `"none"`, `"mom6"` | `"mom6"` in ALE, else `"none"` | thickness of the viscous coupling | `:1049` |
| `thin_layer_cutoff` | ≥ 0 [m] | 0 (off) | HYCOM blending of thin layers | `:1051` |
| `h_thin_bed` | [m] | 0.01 | bed face layers thinner than this carry no velocity; truncation count threshold | `_setup_ale` |

## Validation

| Test | Result | Source |
|---|---|---|
| Lock exchange, flux form + WENO5-Z | fronts 0.463 / 0.486 m s⁻¹ = CROCO; Benjamin 0.495 | `validation/lock_exchange/README.md` |
| Same, advective `upwind3` | front 0.42 m s⁻¹ (16 % slow) | same |
| Hybrid lock exchange, vector-invariant, 21 layers, Godunov, PPM | 1.05 / 0.98 $U$ = MOM6 | `validation/mom6_ke_schemes/README.md` |
| Mode-1 internal wave vs MITgcm | Arakawa 0.91–1.01; Godunov 0.83–0.99 | `validation/internal_wave/README.md` |
| Ekman transport under wind | 0.05 % | module docstring |
| Vanished layer tied by the coupling | `test_thin_layer_coupling_ties_a_vanished_layer_to_its_neighbours` | `tests/test_sigma_ale.py` |

More on the [Idealised benchmarks](../validation/idealised.md) page.

## Known limitations

- No metric (curvature) terms; the WENO/UP3 interpolations of the flux form assume uniform spacing
    (module docstring, `:55-58`).
- Flux-form momentum is skipped on the first σ step (no transports yet).
- ALE has the law-of-the-wall drag since N3 (`LogLawBedDrag`, over the bottom `hbbl` metres, full bed speed), or the quadratic `hbbl` drag; `z0` with `bottom_drag='quadratic'` is an error.
- Thin overflow faces still move up to 1.9× faster than thick ones at $\nu$ = 1000 (Ilicak overflow, open).
- The constant horizontal viscosity `ah` is a plain Laplacian of $u$, not thickness-weighted. The optional Smagorinsky viscosity is a thickness-weighted Laplacian; there is no biharmonic (out of scope), Leith or backscatter. Its `n_visc_limited` counter does not separate Smagorinsky hits from thin-face limits.

## References

- Smagorinsky, J. (1963). General circulation experiments with the primitive equations. I. The basic experiment. *Mon. Wea. Rev.* 91, 99–164. [verify]
- Griffies, S.M. & Hallberg, R.W. (2000). Biharmonic friction with a Smagorinsky-like viscosity for use in large-scale eddy-permitting ocean models. *Mon. Wea. Rev.* 128, 2935–2946. [verify]
- Durran, D.R. (2010). *Numerical Methods for Fluid Dynamics*, 2nd ed., Springer (AB3 stability interval).
- Arakawa, A. & Lamb, V.R. (1981). A potential enstrophy and energy conserving scheme for the shallow water
    equations. *Mon. Wea. Rev.* 109, 18–36. [verify] (the σ-mode docstring cites "Arakawa & Lamb 1977").
- Bleck, R. & Smith, L.T. (1990). A wind-driven isopycnic coordinate model of the north and equatorial Atlantic
    Ocean. *J. Geophys. Res.* 95, 3273–3285. [verify]
- Borden, Z. & Meiburg, E. (2013). Circulation based models for Boussinesq gravity currents. *Phys. Fluids* 25. [verify]
- Hallberg, R. & Rhines, P. (1996). Buoyancy-driven circulation in an ocean basin with isopycnals intersecting
    the sloping boundary. *J. Phys. Oceanogr.* 26, 913–940. [verify]
- Higdon, R.L. (2005). A two-level time-stepping method for layered ocean circulation models: further
    development and testing. *J. Comput. Phys.* 206. doi:[10.1016/j.jcp.2004.12.011](https://doi.org/10.1016/j.jcp.2004.12.011)
- Ilicak, M., Adcroft, A.J., Griffies, S.M. & Hallberg, R.W. (2012). Spurious dianeutral mixing and the role of
    momentum closure. *Ocean Modell.* 45–46, 37–58. doi:[10.1016/j.ocemod.2011.10.003](https://doi.org/10.1016/j.ocemod.2011.10.003)
- Leonard, B.P. (1979). A stable and accurate convective modelling procedure based on quadratic upstream
    interpolation. *Comput. Methods Appl. Mech. Eng.* 19, 59–98.
- Sadourny, R. (1975). The dynamics of finite-difference models of the shallow-water equations.
    *J. Atmos. Sci.* 32, 680–689.
- Shchepetkin, A.F. & McWilliams, J.C. (2005). The Regional Oceanic Modeling System (ROMS): a split-explicit,
    free-surface, topography-following-coordinate oceanic model. *Ocean Modell.* 9, 347–404.
    doi:[10.1016/j.ocemod.2004.08.002](https://doi.org/10.1016/j.ocemod.2004.08.002)
- Ungarish, M. et al. (2018). [verify] (cited in `_flux_form_u_numpy` for front speeds).
- Vallis, G.K. (2006). *Atmospheric and Oceanic Fluid Dynamics*. Cambridge University Press.
- MOM6 source, commit 0e9a561: `src/core/MOM_CoriolisAdv.F90`, `src/parameterizations/vertical/MOM_vert_friction.F90`
    (local copy `~/pyOCEAN_refs/MOM6/src/`).
- HYCOM source: `hybgen.F90`, `mod_momtum.F90` (local copy `~/pyOCEAN/HYCOM-src/`).
