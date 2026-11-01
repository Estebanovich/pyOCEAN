# Governing equations

pyOCEAN solves the **hydrostatic, Boussinesq primitive equations** on an Arakawa C grid. This page states
the continuous equations that each model line solves, the equation of state, the constants, the surface
and bed boundary conditions, and the conservation properties that the discretisation keeps. The discrete
forms are on the pages for each term (linked below); the grids are on [Grids and bathymetry](grids.md)
and the vertical coordinate on [Vertical coordinates and ALE](vertical-ale.md).

pyOCEAN has three model lines. They share the grid, the forcing and the diagnostics, but they do not
solve exactly the same equations:

| Model line | Class / entry point | Vertical coordinate | Prognostic variables | Equation of state |
|---|---|---|---|---|
| Stratified 3-D core (the main model) | `SigmaBaroclinic` (`core/dynamics/sigma_baroclinic.py:933`) | σ-layers (`vertical_coordinate="sigma_eulerian"`, default) or ALE z\*/σ/hybrid (`"ale"`) | η, u_k, v_k, T_k (and h_k in ALE) | linear in T only |
| 2-D tidal model (barotropic, with or without wetting–drying) | `ImplicitFreeSurface` (`core/dynamics/free_surface.py:109`) or `WetDryFreeSurface` (`core/dynamics/wetting_drying.py:51`), driven by `validation/pacmex_tides/run_tides.py` | none (depth-averaged) | η, ū, v̄ | none (ρ₀ only) |
| Legacy phase-1 z-level model | `ModelState` (`core/state.py:60`), `core/dynamics/momentum.py`, `core/timestepping/ab2.py` | z-levels with partial cells | η, u, v, T, S | TEOS-10 or linear (T, S) |

!!! note "Salinity"
    The stratified core advects one active tracer, the temperature T, and its density depends on T only
    (module docstring, "salinity not included", `core/dynamics/sigma_baroclinic.py:55-58`) in the pre-N2a text; since N2a salinity is a second active tracer (`salinity=True`, linear or TEOS-10 density).

## Coordinates and notation

- Horizontal coordinates x (east), y (north); on the sphere λ (longitude) and φ (latitude), with
    dx = R cos φ dλ and dy = R dφ (`create_spherical_grid`, `core/grid/horizontal.py:246`).
- Vertical coordinate z, positive upward, **z = 0 at the rest surface**. The bed is at z = −H(x, y) and the
    free surface at z = η(x, y, t). The total depth is D = H + η.
- In the σ and ALE models the layer index k runs **from the bed**: k = 0 is the bottom layer, k = N − 1
    the top layer (`SigmaState`, `core/dynamics/sigma_baroclinic.py:107`). The legacy z-level model counts
    from the surface (k = 0 at the top; `core/grid/vertical.py:8-11`).
- A subscript h denotes the horizontal part of a vector or operator, ∇ = (∂/∂x, ∂/∂y).

### Symbols

| Symbol | Meaning | Units | Code name |
|---|---|---|---|
| $u, v$ | horizontal velocity components (per layer in 3-D) | m s⁻¹ | `u` (N, ny, nx+1), `v` (N, ny+1, nx) |
| $\mathbf{u}$ | horizontal velocity vector (u, v) | m s⁻¹ | — |
| $w$ | vertical velocity (diagnosed) | m s⁻¹ | `diagnose_w` (legacy) |
| $\omega$ | volume flux per unit area through a σ-layer interface, positive upward | m s⁻¹ | `SigmaState.omega` (N+1, ny, nx) |
| $\tilde w$ | dia-surface velocity through a generalised-coordinate interface | m s⁻¹ | 0 in the ALE Lagrangian step |
| $\eta$ | free-surface height | m | `eta` (ny, nx) |
| $H$ | depth of the bed below the datum (> 0 water, ≤ 0 floodable land, NaN wall) | m | `bathy`, `fs.H` |
| $D = H + \eta$ | total water depth | m | `D` |
| $h_k$ | layer thickness | m | `SigmaState.h` (ALE), $D\,\Delta\sigma_k$ (σ) |
| $\Delta\sigma_k$ | σ-layer fraction, $\sum_k \Delta\sigma_k = 1$ | 1 | `dsigma`, `core.dsigma` |
| $T$ | potential temperature (the only active tracer of the core) | °C | `T` (N, ny, nx) |
| $S$ | salinity (legacy model only) | PSU | `ModelState.S` |
| $\rho$ | density | kg m⁻³ | `density(T)` |
| $\rho_0$ | Boussinesq reference density | kg m⁻³ | `rho0` (default 1000 in `SigmaBaroclinic`) |
| $\alpha$ | linear EOS coefficient of the core, $-\partial\rho/\partial T$ | kg m⁻³ °C⁻¹ | `t_coef` (default 0.2) |
| $T_0$ | reference temperature of the linear EOS | °C | `t_ref` (default 0) |
| $\rho_{\rm ref}(z)$ | optional reference density profile removed in the pressure gradient | kg m⁻³ | `rho_ref` (None, a callable, or `"auto"`) |
| $p$ | hydrostatic pressure | Pa | `hydrostatic_pressure` (legacy) |
| $g$ | gravitational acceleration | m s⁻² | `g` (9.81) |
| $f$ | Coriolis parameter | s⁻¹ | `f_t` (cell centres), `f_q` (corners) |
| $\zeta$ | relative vorticity at cell corners | s⁻¹ | `vorticity_and_pv` |
| $q$ | potential vorticity $(f+\zeta)/h$ at corners | m⁻¹ s⁻¹ | `vorticity_and_pv` |
| $K$ | kinetic energy per unit mass $\tfrac12\lvert\mathbf u\rvert^2$ | m² s⁻² | `kinetic_energy` |
| $\nu$ | vertical viscosity | m² s⁻¹ | `nu_const` (default 1e-4), GLS / parabolic in σ |
| $A_h$ | horizontal Laplacian viscosity | m² s⁻¹ | `ah` (default 0) |
| $K_h, K_v$ | horizontal (along-layer) and vertical tracer diffusivities | m² s⁻¹ | `kh`, `kv` (default 0) |
| $\boldsymbol\tau_s$ | surface (wind) stress | N m⁻² | `step(st, tau_x, tau_y)` |
| $C_d$ | quadratic bottom drag coefficient | 1 | `first_layer_cd`, `cd` |
| $z_0$ | bed roughness length | m | `z0` |
| $\eta_{\rm body}$ | effective tidal forcing height ($\alpha_c\eta_{\rm eq}$ − SAL) | m | `body_tide` |
| $\beta_{\rm SAL}$ | scalar self-attraction and loading factor | 1 | `sal_beta` |
| $\theta$ | implicit weight of the free surface | 1 | `theta` (default 0.6) |

## Hydrostatic Boussinesq primitive equations

### Cartesian form

For a Boussinesq fluid in hydrostatic balance (Vallis 2006, §2.4; Marshall et al. 1997):

$$
\frac{\partial \mathbf u}{\partial t} + (\mathbf u\cdot\nabla)\mathbf u + w\frac{\partial\mathbf u}{\partial z}
+ f\,\hat{\mathbf z}\times\mathbf u
= -\frac{1}{\rho_0}\nabla_z p + \frac{\partial}{\partial z}\Big(\nu\frac{\partial\mathbf u}{\partial z}\Big)
+ A_h\nabla^2\mathbf u ,
$$

$$
\frac{\partial p}{\partial z} = -\rho g , \qquad
\nabla\cdot\mathbf u + \frac{\partial w}{\partial z} = 0 ,
$$

$$
\frac{\partial T}{\partial t} + \nabla\cdot(\mathbf u T) + \frac{\partial (wT)}{\partial z}
= \nabla\cdot(K_h\nabla T) + \frac{\partial}{\partial z}\Big(K_v\frac{\partial T}{\partial z}\Big),
\qquad \rho = \rho(T[,S,p]).
$$

Integrating the hydrostatic relation from the surface, with the atmospheric pressure set to zero,

$$
p(z) = \rho_0 g(\eta - z) + g\int_z^{\eta}\rho'\,dz', \qquad \rho' = \rho - \rho_0 ,
$$

so the horizontal pressure-gradient force splits into a **barotropic** part, handled by the free-surface
solver, and a **baroclinic** part, handled by the pressure-gradient routines:

$$
-\frac{1}{\rho_0}\nabla_z p = -g\nabla\eta \;-\; \frac{g}{\rho_0}\int_z^{\eta}\nabla_z\rho'\,dz' .
$$

In the code the first term is the θ-weighted $-g\nabla\eta$ of the free surface
([Free surface and time stepping](free-surface-time-stepping.md)), and the second is
`SigmaBaroclinic._baroclinic_pgf` (`core/dynamics/sigma_baroclinic.py:1381`) or the finite-volume
`fv_pgf` (`core/vertical/pgf.py:165`) ([Pressure gradient](pressure-gradient.md)). With a reference
profile $\rho_{\rm ref}(z)$ (option `rho_ref`), $\rho'$ is replaced by $\rho - \rho_{\rm ref}(z)$; the
pressure of a horizontally uniform $\rho_{\rm ref}(z)$ has no horizontal gradient at fixed z, so the
result is unchanged in exact arithmetic but the large cancelling terms of a resting stratification are
removed (Haney 1991; `_baroclinic_pgf` docstring).

Integrating continuity over the column with the kinematic conditions below gives the free-surface
equation

$$
\frac{\partial\eta}{\partial t} + \nabla\cdot\int_{-H}^{\eta}\mathbf u\,dz = 0 .
$$

### Spherical form

On a spherical-polar grid ($x = R\cos\varphi\,\lambda$, $y = R\varphi$) the operators become

$$
\nabla\phi = \Big(\frac{1}{R\cos\varphi}\frac{\partial\phi}{\partial\lambda},
\frac{1}{R}\frac{\partial\phi}{\partial\varphi}\Big), \qquad
\nabla\cdot\mathbf u = \frac{1}{R\cos\varphi}\Big[\frac{\partial u}{\partial\lambda}
+ \frac{\partial (v\cos\varphi)}{\partial\varphi}\Big],
\qquad f = 2\Omega\sin\varphi .
$$

pyOCEAN never writes these derivatives explicitly. All operators are written in **finite-volume form with
the grid metrics** (`dx_*`, `dy_*`, `area_*`; see [Grids](grids.md)): a divergence is the sum of face
fluxes $L_f\,u_f$ divided by the cell area, a gradient is a difference divided by the distance between
the two points. The same code therefore runs on Cartesian, stretched and spherical grids. The
consequences are:

- the **curvature (metric) terms** $u v\tan\varphi/R$ and $u^2\tan\varphi/R$ of the advective
    momentum equations are neglected in the advective forms of `core/dynamics/nonlinear.py`
    ("$|u|/R \ll f$ at regional scales", module docstring);
- the **vector-invariant form** of the ALE core computes the vorticity in circulation form,
    $\zeta = [\delta_x(v\,\Delta y) - \delta_y(u\,\Delta x)]/A_q$ (`vorticity_q`,
    `core/dynamics/vector_invariant.py`), which contains the metric term of a curvilinear grid implicitly;
- the Coriolis parameter is the full $2\Omega\sin\varphi$ at cell centres (`f_t`) and corners (`f_q`) on
    spherical grids, and $f_0 + \beta y$ on Cartesian ones.

!!! warning "Spherical areas"
    `create_spherical_grid` uses $A = \Delta x\,\Delta y = R^2\cos\varphi_T\,\Delta\lambda\,\Delta\varphi$
    (`core/grid/horizontal.py:312-314`), the midpoint approximation of the exact area
    $R^2\Delta\lambda(\sin\varphi_n - \sin\varphi_s)$. The relative difference is
    $O(\Delta\varphi^2)$ (≈ 1e-7 for 0.04° cells) and is not corrected.

## Generalised vertical coordinate (ALE form)

The ALE mode (`vertical_coordinate="ale"`) solves the layer-integrated equations of a generalised vertical
coordinate s (Adcroft & Hallberg 2006; Griffies et al. 2020). For layer k, between the interfaces
$z_{k-\frac12}$ (below) and $z_{k+\frac12}$ (above), with thickness $h_k = z_{k+\frac12} - z_{k-\frac12}$:

**Thickness (volume) equation**

$$
\frac{\partial h_k}{\partial t} + \nabla\cdot(h_k\mathbf u_k)
+ \big(\tilde w_{k+\frac12} - \tilde w_{k-\frac12}\big) = 0 ,
\qquad \tilde w_{-\frac12} = \tilde w_{N-\frac12} = 0 ,
$$

where $\tilde w$ is the volume flux per unit area through the moving interface (dia-surface velocity).
Summing over the layers, $\sum_k h_k = H + \eta$ and

$$
\frac{\partial\eta}{\partial t} + \nabla\cdot\sum_k h_k\mathbf u_k = 0 .
$$

**Tracer equation (thickness weighted)**

$$
\frac{\partial (h_k T_k)}{\partial t} + \nabla\cdot(h_k\mathbf u_k T_k)
+ \big[\tilde w T\big]_{k-\frac12}^{k+\frac12}
= \nabla\cdot(h_k K_h\nabla_s T_k) + \Big[K_v\frac{\partial T}{\partial z}\Big]_{k-\frac12}^{k+\frac12} .
$$

**Momentum equation (vector-invariant, default in ALE)**

$$
\frac{\partial\mathbf u_k}{\partial t} + (f+\zeta_k)\,\hat{\mathbf z}\times\mathbf u_k + \nabla_s K_k
+ \frac{\tilde w}{h}\,\delta_k\mathbf u
= -\frac{1}{\rho_0}\nabla_z p\Big|_k
+ \frac{1}{h_k}\Big[\nu\frac{\partial\mathbf u}{\partial z}\Big]_{k-\frac12}^{k+\frac12}
+ A_h\nabla^2\mathbf u_k ,
$$

with $K_k = \tfrac12\lvert\mathbf u_k\rvert^2$. The term $(f+\zeta)\hat{\mathbf z}\times\mathbf u$ is
written as a potential-vorticity flux $q\,(\overline{vh}, -\overline{uh})$, $q = (f+\zeta)/h$, as in
MOM6 (Sadourny 1975; `core/dynamics/vector_invariant.py`). The flux form

$$
\frac{\partial (h_k\mathbf u_k)}{\partial t} + \nabla\cdot(h_k\mathbf u_k\mathbf u_k)
+ \big[\tilde w\,\mathbf u\big]_{k-\frac12}^{k+\frac12} + f\hat{\mathbf z}\times h_k\mathbf u_k = \dots
$$

is kept as the option `momentum="flux"`. See [Momentum equations](momentum.md).

**The ALE splitting.** pyOCEAN does not compute $\tilde w$ from a coordinate equation. It integrates the
equations above with $\tilde w = 0$ (the *Lagrangian step*, interfaces move with the flow), and then
moves the interfaces to the target grid of the chosen rule (*regrid*) and maps all fields conservatively
onto the new layers (*remap*). The remap is the discrete equivalent of the $\tilde w$ terms (Griffies et
al. 2020). This is the subject of [Vertical coordinates and ALE](vertical-ale.md).

## σ-coordinate form (σ-Eulerian mode)

The default mode `vertical_coordinate="sigma_eulerian"` uses fixed fractions $\Delta\sigma_k$:

$$
h_k = D\,\Delta\sigma_k, \qquad z_{k+\frac12} = -H + D\,\sigma_{k+\frac12},
\qquad \sigma_{k+\frac12} = \sum_{m\le k}\Delta\sigma_m \in [0, 1],
$$

with σ measured from the bed (`SigmaWetDry.sigma_w`, `core/dynamics/sigma_wd.py:178`). $\Delta\sigma_k$
is uniform by default, or stretched (`core.grid.sigma_levels`; [Grids](grids.md#stretched-layers)). The
interfaces move with D but keep their fractions, so the dia-surface flux is not zero: it is the σ-velocity
ω, diagnosed from the layer volume budget. For the layer volume $V_k = h_k A$ and the layer transports
$Q_k$ through the faces,

$$
\frac{V_k^{n+1} - V_k^{n}}{\Delta t} + \nabla_h\!\cdot Q_k + \big(F_{k+\frac12} - F_{k-\frac12}\big) = 0 ,
\qquad F_{-\frac12} = 0 ,
$$

so (`SigmaBaroclinic.step`, `core/dynamics/sigma_baroclinic.py:2084`, the `Fz` block)

$$
F_{k+\frac12} = F_{k-\frac12} - \Big[\frac{V_k^{n+1} - V_k^n}{\Delta t} + \nabla_h\!\cdot Q_k\Big],
\qquad \omega_{k+\frac12} = F_{k+\frac12}/A .
$$

Because the layer transports are built so that $\sum_k Q_k$ is exactly the volume flux of the
free-surface solve,

$$
Q_k = L\,D_f\,\Delta\sigma_k\big[\theta_{bc}u_k^{n+1} + (1-\theta_{bc})u_k^n
+ (\theta - \theta_{bc})(\bar u^{n+1} - \bar u^{n})\big],\qquad \bar u = \sum_k\Delta\sigma_k u_k ,
$$

$F_{N-\frac12}$ vanishes to round-off at the surface (it is set to 0) and a uniform tracer stays uniform
(module docstring, item 3). The continuous σ equations are the layer equations above with $\tilde w = \omega$
and $h_k = D\Delta\sigma_k$; momentum uses the flux form with WENO5 face values and the energy-conserving
Coriolis term ([Momentum](momentum.md)).

## Equation of state

=== "Stratified core (σ and ALE)"

    Three equations of state since the TEOS-10 in 3-D project (2026-11-01): `eos="linear"` (default, below),
    `"wright_full"` (Wright 1997 on θ, S_P, MOM6 `WRIGHT_FULL`) and `"teos10"` (Roquet et al. 2015 on Θ, S_A,
    MOM6 `ROQUET_RHO`). The nonlinear ones and every consumer (pressure gradient, closures, hybrid regrid,
    surface heat, inputs and outputs) are described in [Equation of state](equation-of-state.md). The default is
    linear (`SigmaBaroclinic.density`, `core/dynamics/sigma_baroclinic.py:1285`; "as CROCO `lin_EOS_cff`", module
    docstring); with `salinity=True` it adds $+\beta(S - S_0)$ (`s_coef`, `s_ref`):

    $$
    \rho = \rho_0 - \alpha\,(T - T_0),
    $$

    with **α in kg m⁻³ °C⁻¹** (`t_coef`, default 0.2), $\rho_0$ = `rho0` (default 1000 kg m⁻³) and $T_0$ =
    `t_ref` (default 0 °C). The same formula is used in the finite-volume pressure gradient
    (`fv_pgf(..., rho0, t_coef, t_ref, ...)`, `core/vertical/pgf.py:165`) and in the hybrid regrid rule,
    which needs the layer densities (`SigmaBaroclinic._regrid`, `core/dynamics/sigma_baroclinic.py:1334`).
    Realistic cases use a *density-equivalent temperature*: the observed density converted to T with this
    linear law (CLAUDE.md, BayIW "real" case).

=== "Legacy z-level model"

    `physics/eos/__init__.py` provides two equations of state:

    - **TEOS-10** (`density_teos10`, line 63): `gsw.rho(SA, CT, p)` with the approximations SA ≈ S and
        CT ≈ θ, and the pressure $p \approx \rho_0 g\lvert z\rvert$ (in dbar, $\rho_0$ = 1025 kg m⁻³).
    - **Linear** (`density_linear`, line 123; Vallis 2006, eq. 1.156):

        $$
        \rho = \rho_0\,[1 - \alpha(T - T_0) + \beta(S - S_0)],
        $$

        here with **α and β relative** (1/°C, 1/PSU): defaults α = 2.0e-4, β = 7.4e-4, $T_0$ = 10 °C,
        $S_0$ = 35, $\rho_0$ = 1025 kg m⁻³.

    `compute_density(state, z_grid, eos="teos10")` (line 203) selects one of them; its default is TEOS-10.

!!! warning "Two meanings of α"
    The core's `t_coef` is an absolute coefficient (kg m⁻³ °C⁻¹): `t_coef = 0.2` with `rho0 = 1000` is a
    relative expansion of 2e-4 °C⁻¹. The legacy `density_linear(alpha=...)` is relative (°C⁻¹).

## Constants

| Constant | Value | Where |
|---|---|---|
| $g$ | 9.81 m s⁻² | `G` in `core/dynamics/sigma_baroclinic.py:101`, `sigma_wd.py:48`, `wetting_drying.py`, `physics/eos/__init__.py:55`; argument `g` of `SigmaBaroclinic` |
| $\rho_0$ (stratified core) | 1000 kg m⁻³ (default of `rho0` with `eos="linear"`; 1035 with a nonlinear EOS) | `SigmaBaroclinic.__init__`, `core/dynamics/sigma_baroclinic.py:1005` |
| $\rho_0$ (legacy z model) | 1025 kg m⁻³ | `RHO_0` in `physics/eos/__init__.py:56` and `core/dynamics/momentum.py` |
| $\Omega$ | 7.2921e-5 rad s⁻¹ | `OMEGA`, `core/grid/horizontal.py:41` |
| $R$ | 6.371e6 m | `R_EARTH`, `core/grid/horizontal.py:40` |
| $\kappa$ (von Kármán) | 0.41 | `KAPPA`, `physics/bottom_friction.py:28` |
| Love factor $\alpha_c = 1 + k_c - h_c$ | 0.693 (semidiurnal), 0.736 (K1), 0.695 (O1, Q1), 0.706 (P1) | `LOVE_ALPHA`, `forcing/tides.py:345` |

!!! note "ρ₀ differs from CLAUDE.md"
    CLAUDE.md lists $\rho_0$ = 1025 kg m⁻³ as the project constant. That value is used by the legacy
    z-level model and the EOS module; the stratified core defaults to `rho0 = 1000`. Set `rho0` explicitly
    when a case needs 1025.

## Boundary conditions

### Surface (z = η)

- **Kinematic:** no volume flux through the free surface (no evaporation or precipitation is applied to
    the volume): $w = \partial_t\eta + \mathbf u\cdot\nabla\eta$. Discretely, the surface ω is zero
    (`Fz[-1] = 0` in σ, $\tilde w_{N-\frac12} = 0$ in ALE) and η changes only through the divergence of
    $\sum_k Q_k$.
- **Stress:** $\nu\,\partial\mathbf u/\partial z = \boldsymbol\tau_s/\rho_0$. The kinematic stress
    $\boldsymbol\tau_s/\rho_0$ enters the implicit vertical viscosity as the flux into the top layer, not
    through the Adams–Bashforth extrapolation (`SigmaBaroclinic.step` docstring; `vertical_implicit`,
    `core/dynamics/sigma_wd.py:88`). Wind stress is given at u- and v-faces as `tau_x` (ny, nx+1) and
    `tau_y` (ny+1, nx) [N m⁻²]. See [Air–sea fluxes](air-sea-fluxes.md).
- **Heat flux:** none in the stratified core at present (no surface tracer flux term in
    `SigmaBaroclinic.step`); the bulk formulae of `forcing/bulk.py` are described in
    [Air–sea fluxes](air-sea-fluxes.md).

### Bed (z = −H)

- **Kinematic:** no flux through the bed, $w = -\mathbf u\cdot\nabla H$; discretely $F_{-\frac12} = 0$.
- **Stress (σ mode with `z0` set):** quadratic drag on the bottom-layer velocity with the first-layer law
    of the wall (`first_layer_cd`, `physics/bottom_friction.py:48`; Soulsby 1997):

    $$
    \nu\frac{\partial\mathbf u}{\partial z}\Big|_{-H} = C_d\,\lvert u_0\rvert\,u_0, \qquad
    C_d = \Big[\frac{\kappa}{\ln(\Delta z_0/2z_0)}\Big]^2,\quad 10^{-4}\le C_d\le 0.1 ,
    $$

    applied implicitly per velocity component (`column_step`, `core/dynamics/sigma_wd.py:187`: $r = C_d\lvert u_0\rvert$ of the component itself).
- **ALE mode (since N3):** `z0` is accepted. Because the bed layer can vanish under z\*, the law acts on
    $d_b=\max(h_b,\min(h_{bbl},D_f))$ with the full bed speed (`LogLawBedDrag`, `core/components/bottom_drag.py:88`),
    or the quadratic `hbbl` drag (`QuadraticBedDrag`, `:44`). See [Turbulence closures and bottom friction](turbulence-friction.md).

### Lateral

- **Walls and land** (NaN depth): no normal flow; the face is closed in the free-surface matrix and its
    velocity is zero.
- **Laplacian viscosity and vorticity: free slip** (`laplacian_3d`; ζ = 0 at corners that touch land or the
    domain edge, `vorticity_q`).
- **Open boundaries**: Flather radiation of the depth mean, Orlanski radiation and nudging of the baroclinic
    velocity, upstream tracers, sponge; see [Open boundaries and nesting](open-boundaries.md).
- **Wetting and drying**: faces with total depth below `d_min` (default 0.05 m) are dry; see
    [Wetting and drying](wetting-drying.md).

## 2-D shallow-water equations of the tidal model

The barotropic tidal runs (`validation/pacmex_tides/run_tides.py`, `run_tides_wd.py`) solve the
depth-averaged equations on the sphere (module docstring of `run_tides.py`; Vallis 2006 §3.1):

$$
\frac{\partial\bar{\mathbf u}}{\partial t} + \epsilon\,(\bar{\mathbf u}\cdot\nabla)\bar{\mathbf u}
+ f\hat{\mathbf z}\times\bar{\mathbf u}
= -g(1-\beta_{\rm SAL})\nabla\eta + g\nabla\eta_{\rm body}
- \frac{C_d\lvert\bar{\mathbf u}\rvert\bar{\mathbf u}}{D_\epsilon} - r_{\rm IT}\bar{\mathbf u}
+ A_h\nabla^2\bar{\mathbf u},
$$

$$
\frac{\partial\eta}{\partial t} + \nabla\cdot(H\bar{\mathbf u}) + \epsilon\,\nabla\cdot(\eta\bar{\mathbf u}) = 0 ,
\qquad D_\epsilon = H + \epsilon\,\eta ,
$$

with $\epsilon = 1$ for the nonlinear option (`--nonlinear`) and 0 otherwise:

- the linear part ($-g\nabla\eta$, $\nabla\cdot(H\bar{\mathbf u})$) is the θ-implicit free surface
    `ImplicitFreeSurface` (Dukowicz & Smith 1994; Campin et al. 2004), or `WetDryFreeSurface` with the
    total depth at the faces when cells can dry (Casulli 1990, 2009);
- the nonlinear continuity term $-\nabla\cdot(\eta\bar{\mathbf u})$ is the explicit source
    `nonlinear_continuity` (`core/dynamics/nonlinear.py:60`), and the momentum advection is
    `advection_u/v` (centred or third-order upwind; `core/dynamics/nonlinear.py:165`);
- the tidal body force is $g\nabla\eta_{\rm body}$ with
    $\eta_{\rm body} = \sum_c[\alpha_c\eta_{{\rm eq},c} - \eta_{{\rm SAL},c}]$ (`body_tide`, `body_force`,
    `forcing/tides.py:393,421`; Foreman et al. 1993); alternatively a scalar SAL reduces the gravity of the
    surface pressure gradient to $g(1-\beta_{\rm SAL})$ (`sal_beta`, `core/dynamics/free_surface.py:166`);
- $r_{\rm IT}$ is the optional linear internal-tide (JSL) drag of `physics/tides/drag.py`.

Details: [Tides](tides.md) and [Tides and the 2-D tidal model](../user-guide/tidal-model.md).

## Conservation and energy properties

| Property | How it is kept | Where |
|---|---|---|
| Total volume | Flux-form free surface; with the CG solver η is recomputed from the divergence of the final transports (MITgcm exactConserv) | `WetDryFreeSurface` (`solver="cg"`), `core/dynamics/wetting_drying.py:90` |
| Column consistency $\sum_k h_k = H+\eta$ (ALE) | Layer fluxes corrected to the free-surface flux (uniform correction + HYCOM residual share), outflow limiter keeps $\sum_k Q_k$ | `upwind_transport_x`, `ppm_transport_x`, `limit_outflow` (`core/vertical/ale.py`) |
| Layer thickness ≥ 0 (ALE) | Outflow limiter: no layer sends out more than it holds in Δt | `limit_outflow`, `core/vertical/ale.py:331` |
| Tracer content $\sum hT$ | Flux-form advection; conservative remap with the "Hallberg trick" (residual to the thickest sub-cell) | `_advect`, `remap_column` (`core/vertical/remap.py:307`) |
| Tracer constancy (uniform T stays uniform) | Layer transports consistent with the volume change (ω from the budget in σ; Lagrangian $h^\dagger$ in ALE) | `SigmaBaroclinic.step`, `_step_ale` |
| Tracer bounds | FCT limiter (Zalesak 1979) around SSP-RK3 WENO5 in ALE (default) | `tracer_limiter="fct"` |
| Depth-mean velocity $\sum_k w_k u_k$ across a remap | Face velocities remapped in face-fraction space | `_remap_faces`, `core/dynamics/sigma_baroclinic.py:1367` |
| Coriolis does no work | Energy-conserving C-grid average (σ, flux form; Sadourny 1975, Arakawa & Lamb 1977) or Sadourny energy PV flux (ALE vector invariant) | `_coriolis` (`:1467`), `vi.coriolis_pv` |
| Kinetic-energy gradient | `ke_scheme="arakawa"` (energy conserving, default) or `"gudonov"` (upwind, dissipative at fronts) | `vi.kinetic_energy` |
| Pressure gradient of a resting stratification | Exact for ρ linear in z (Jacobian in σ; finite volume on any layers; reference profile) | `_baroclinic_pgf`, `fv_pgf` |
| Spurious mixing (diagnostic) | Online variance decay (Burchard & Rennau 2008): T and T² advected and remapped together, `mix_num` | `SigmaState.mix_num` |
| Reference potential energy (diagnostic) | Winters et al. (1995), Ilicak et al. (2012) | `SigmaBaroclinic.rpe`, `diagnostics/mixing.py` |

What is **not** conserved exactly: total energy (implicit free surface with θ > ½ damps gravity waves;
the vertical remap of velocity conserves the depth-mean momentum but not the kinetic energy), and
momentum near vanished layers, where the CFL truncation backstop `truncate_cfl` (MOM6
`vertvisc_limit_vel`) may act (counted in `SigmaState.n_trunc`).

## Validation

Each term is validated on its own page; the scorecard is in [Validation](../validation/index.md). For the
equations as a whole: lock exchange vs Benjamin's front speed and CROCO/MOM6, seamount (pressure
gradient at rest), Ilicak et al. (2012) internal wave and overflow, BayIW seiche and internal waves vs
MITgcm (CLAUDE.md, Build Status Phase 2).

## Known limitations

- ~~One active tracer (T), linear EOS in the stratified core~~: salinity since N2a; nonlinear EOS (`wright_full`,
    `teos10`) since 2026-11-01 ([Equation of state](equation-of-state.md)).
- No surface heat or freshwater flux in the stratified core; volume is closed except through open
    boundaries.
- Bed drag in ALE only over the bottom `hbbl` metres (N3); open-edge faces take a zero cross component.
- Curvature terms neglected in the advective momentum forms on spherical grids.
- Hydrostatic only (a non-hydrostatic option is roadmap stage E).

## References

- Adcroft, A. & Hallberg, R. (2006). On methods for solving the oceanic equations of motion in generalized
    vertical coordinates. *Ocean Modelling* 11, 224–233. doi:10.1016/j.ocemod.2004.12.007
- Arakawa, A. & Lamb, V. R. (1977). Computational design of the basic dynamical processes of the UCLA
    general circulation model. *Methods Comput. Phys.* 17, 173–265. **[verify]**
- Burchard, H. & Rennau, H. (2008). Comparative quantification of physically and numerically induced
    mixing in ocean models. *Ocean Modelling* 20, 293–311. **[verify]** (DOI not in the reading list)
- Campin, J.-M., Adcroft, A., Hill, C. & Marshall, J. (2004). Conservation of properties in a free-surface
    model. *Ocean Modelling* 6, 221–244. **[verify]**
- Casulli, V. (1990). Semi-implicit finite difference methods for the two-dimensional shallow water
    equations. *J. Comput. Phys.* 86, 56–74. **[verify]**
- Casulli, V. (2009). A high-resolution wetting and drying algorithm for free-surface hydrodynamics.
    *Int. J. Numer. Meth. Fluids* 60, 391–408. doi:10.1002/fld.1896
- Dukowicz, J. K. & Smith, R. D. (1994). Implicit free-surface method for the Bryan–Cox–Semtner ocean
    model. *J. Geophys. Res.* 99(C4), 7991–8014. **[verify]**
- Foreman, M. G. G., Henry, R. F., Walters, R. A. & Ballantyne, V. A. (1993). A finite element model for
    tides and resonance along the north coast of British Columbia. *J. Geophys. Res.* 98(C2), 2509–2531. **[verify]**
- Griffies, S. M. et al. (2020). A primer on the vertical Lagrangian-remap method in ocean models based on
    finite volume generalized vertical coordinates. *JAMES* 12, e2019MS001954. doi:10.1029/2019MS001954
- Haney, R. L. (1991). On the pressure gradient force over steep topography in sigma coordinate ocean
    models. *J. Phys. Oceanogr.* 21, 610–619. doi:10.1175/1520-0485(1991)021<0610:OTPGFO>2.0.CO;2 **[verify]**
- Ilicak, M., Adcroft, A., Griffies, S. M. & Hallberg, R. W. (2012). Spurious dianeutral mixing and the role
    of momentum closure. *Ocean Modelling* 45–46, 37–58. **[verify]**
- Marshall, J., Adcroft, A., Hill, C., Perelman, L. & Heisey, C. (1997). A finite-volume, incompressible
    Navier–Stokes model for studies of the ocean on parallel computers. *J. Geophys. Res.* 102(C3), 5753–5766. **[verify]**
- Sadourny, R. (1975). The dynamics of finite-difference models of the shallow-water equations.
    *J. Atmos. Sci.* 32, 680–689. **[verify]**
- Soulsby, R. (1997). *Dynamics of Marine Sands.* Thomas Telford, London. **[verify]**
- Vallis, G. K. (2006). *Atmospheric and Oceanic Fluid Dynamics.* Cambridge University Press.
- Winters, K. B., Lombard, P. N., Riley, J. J. & D'Asaro, E. A. (1995). Available potential energy and mixing
    in density-stratified fluids. *J. Fluid Mech.* 289, 115–128. **[verify]**
- Zalesak, S. T. (1979). Fully multidimensional flux-corrected transport algorithms for fluids.
    *J. Comput. Phys.* 31, 335–362. doi:10.1016/0021-9991(79)90051-2
