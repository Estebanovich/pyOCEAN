# Equation of state

The stratified 3-D model (`SigmaBaroclinic`, σ and ALE) has three equations of state behind one component
interface (`core/components/eos.py`): the **linear** EOS (default, unchanged bit for bit), **`wright_full`**
(Wright 1997, the EOS of MOM6 NWA12) and **`teos10`** (the Roquet et al. 2015 polynomial fit to TEOS-10). The two
nonlinear ones depend on pressure, so they bring cabbeling and thermobaricity into the model. They enter the
pressure gradient, the $N^2$ of every closure, the hybrid regrid, the horizontal diffusion with a reference
profile, the surface heat flux and the input/output conversions. This page describes what each consumer does.
The arguments are listed in the [Configuration reference](../user-guide/configuration-reference.md#equation-of-state).

Added by the "TEOS-10 in the 3-D core" project (spec `docs/superpowers/specs/2026-10-31-teos10-3d-design.md`,
MOM6 reference read `docs/design/teos10_refs/mom6_eos.md`, MOM6 git `0e9a561`). Validation:
`validation/teos10/README.md` and `validation/mom6_column/README.md`.

!!! note "Phase-1 z-level model"
    The legacy z-level model (`ModelState`) keeps its own `physics/eos/__init__.py` (`eos="teos10"` through `gsw`,
    or a relative linear law). Nothing on this page applies to it, and it does not use the 3-D EOS component.

## The three equations

| `eos=` | Formula | Prognostic T, S | Pressure | ρ₀ default | c_p default [J kg⁻¹ K⁻¹] | Code | Reference |
|---|---|---|---|---|---|---|---|
| `"linear"` | $\rho = \rho_0 - \alpha (T - T_0) + \beta (S - S_0)$ | T [°C], S | none | 1000 | 3985 | `LinearEOS` | Vallis (2006) eq. 1.156 |
| `"wright_full"` | Wright (1997) full-range rational fit | θ [°C], S_P | yes | 1035 | 3991.86795711963 | `WrightFullEOS`, `physics/eos/wright_full.py` | MOM6 `MOM_EOS_Wright_full.F90` |
| `"teos10"` | Roquet et al. (2015) polynomial (52 + 6 coefficients) | Θ [°C], S_A [g kg⁻¹] | yes | 1035 | 3991.86795711963 ($c_p^0$) | `TEOS10EOS`, `physics/eos/roquet_rho.py` | MOM6 `MOM_EOS_Roquet_rho.F90` (`ROQUET_RHO`) |

The linear EOS uses absolute coefficients: α = `t_coef` [kg m⁻³ °C⁻¹], β = `s_coef` [kg m⁻³ PSU⁻¹].
`t_coef`, `t_ref`, `s_coef` and `s_ref` are rejected (pre-flight error) with a nonlinear EOS.

Both nonlinear kernels are written in Numba with a NumPy oracle. They provide

- the in-situ density $\rho(T, S, p)$;
- the anomaly $\rho - \rho_{ref}$ in MOM6's form (the subtraction is folded into the numerator, so it does not
    lose digits);
- the derivatives $\partial\rho/\partial T$ and $\partial\rho/\partial S$ at constant $p$;
- the potential density $\rho(T, S, p_{ref})$.

All four are **bit for bit** equal to compiled MOM6 (`validation/mom6_column/eos_driver.F90`, 2048 points,
T −2…32 °C, S 0…42, p 0…6e7 Pa, freshwater included; max |Δ| = 0; `tests/test_eos_nonlinear.py`).

!!! warning "MOM6's \"TEOS10\" is a different fit"
    pyOCEAN's `"teos10"` is MOM6's `ROQUET_RHO`, not MOM6's `TEOS10` option (the 75-term `gsw_specvol` polynomial).
    The two fits differ beyond round-off. Against `gsw` (the 75-term polynomial) the Roquet density differs by
    5.4e-2 kg m⁻³ at most for S ≥ 2. That maximum is in brackish water at 5500–6000 dbar, outside the oceanic range
    of the fits. The RMS is 7.6e-3 and, for 30 < S_A < 40, the maximum is 3.0e-2. The test bound `ROQUET_FIT_MAX` is
    1.1 × the measured maximum, because the published fit error has not been checked yet: Roquet (2015) is not in
    the local library **[verificar Roquet 2015]**.

### Pressure and salinity conventions

- **Pressure** in the EOS is Boussinesq, $p = -g\rho_0 z$, with $z$ measured from the rest surface. It includes
    neither η nor the atmospheric pressure, as in MOM6 (`MOM_density_integrals.F90:579-751`). The error is of order
    $\rho_0 g \eta$, negligible for the EOS. The closures use $p = g\rho_0\sum h$ from the surface down to the
    interface (`physics/eos/stratification.py`, `interface_pressure`).
- **Absolute salinity** uses MOM6's rule, $S_A = (35.16504/35)\,S_P$ (`physics/eos/convert.py`, `SR_OVER_SP`).
    It has no spatial anomaly δS_A. This is class (c) against strict TEOS-10 (spec, Claude decision 3).
- The model never converts between θ/S_P and Θ/S_A **inside** the step. Conversions happen only at the input and
    output boundaries ([Inputs and outputs](#inputs-and-outputs)). The only exception is the 2-D surface field
    handed to the bulk formulae.

## Pressure gradient

### Finite volume (`pgf="fv"`, ALE)

With a nonlinear EOS, `core/vertical/pgf_nonlinear.py` (`fv_pgf_nonlinear`) ports MOM6
`int_density_dz_generic_plm` and the assembly of `PressureForce_FV_Bouss` (Adcroft et al. 2008;
`MOM_density_integrals.F90`, `MOM_PressureForce_FV.F90:1285-1300`):

- T and S are reconstructed separately with PLM, in MOM6's operation order.
- The density is evaluated at the 5 Boole points in $z$ of each layer and at 5 sub-columns along each face
    (0, ¼, ½, ¾, 1). That is about 35 EOS evaluations per cell and layer, the dominant cost (see [Cost](#cost)).
- A reference density is subtracted inside the evaluation (anomaly form).

The kernel has **two modes** (spec, Claude decision 9):

=== "MOM6 mode, `rho_ref=None`"

    T and S use plain PLM. The constant subtracted is `rho_pgf_ref` (MOM6 `RHO_PGF_REF`, default ρ₀). This mode is
    the MOM6 method, and it is **bit for bit** equal to compiled MOM6 on 3330 faces × layers: 4 cases × 2 EOS ×
    3 mass-weighting modes, with thin, vanished and stepped layers (`validation/mom6_column/pgf_driver.F90`,
    `tests/test_pgf_nonlinear.py`).

=== "Profile mode, `rho_ref=\"auto\"` (recommended)"

    `initial_state` stores the horizontal-mean initial profiles $T_{ref}(z)$ and $S_{ref}(z)$
    (`TSReferenceProfile`, 400 depth bins). The PLM reconstructs the anomalies $T' = T - T_{ref}$ and
    $S' = S - S_{ref}$. At each quadrature point the kernel evaluates

    $$
    \rho\big(T_{ref}(z_q) + T'(z_q),\, S_{ref}(z_q) + S'(z_q),\, p_q\big) - \rho\big(T_{ref}(z_q), S_{ref}(z_q), p_q\big).
    $$

    A column that lies on the profile has a zero anomaly at every height, on any layers. This keeps the advantage
    pyOCEAN already had with the linear `rho_ref` profile. A `rho_ref` given as a function is rejected with a
    nonlinear EOS, because a density profile does not give T_ref and S_ref.

Bed mass weighting, `pgf_mass_weight` (spec, Claude decision 10):

- `"vanished"` (default): pyOCEAN's existing rule, only next to vanished layers.
- `"full"`: NWA12's setting, `MASS_WEIGHT_IN_PGF_VANISHED_ONLY = False`. Nonlinear EOS only.
- `"none"`.

The linear path (`fv_pgf` / `fv_pgf_rho`, PLM of the anomaly) is unchanged.

### σ schemes (`pgf="jacobian"`, `"twoterm"`)

The σ schemes take the in-situ density at the cell pressure $p(z)$ minus $\rho_{ref}(z)$. With σ-layers, a
nonlinear EOS and `rho_ref=None`, pre-flight warns. The compressibility (~4–5 kg m⁻³ per km) then enters the
density anomaly along sloping layers in full.

## Stratification of the closures

The closures receive a `DensityContext` (`physics/eos/stratification.py`) from the driver
(`Driver._closure_density`). With a nonlinear EOS they use **locally referenced** stratification, as MOM6. With the
linear EOS every closure keeps its old expression, bit for bit.

- **GLS, BBL (`bbl_mixing='lotw'`) and Jackson shear mixing:**

    $$
    N^2 = -\frac{g}{\rho_0}\,\frac{\partial_T\rho\,\Delta T + \partial_S\rho\,\Delta S}{\Delta z},
    $$

    The derivatives are taken at the mean T and S of the interface and at the interface pressure
    (`MOM_kappa_shear.F90:1128-1136, 1496-1500`; `MOM_set_diffusivity.F90:1160-1200`). Thin layers are merged
    first, as before.
- **KPP:** the bulk-Richardson buoyancy differences and the local $N^2$ come from the full density of the two parts
    at the common pressure of the top interface of layer k (`MOM_CVMix_KPP.F90:1396-1436`). The surface buoyancy flux
    uses ∂ρ/∂T and ∂ρ/∂S of the top layer at p = 0 (`surface_alpha_beta`; `MOM_forcing_type.F90:1116-1148`)
    instead of `t_coef`/`s_coef`. The penetrating shortwave uses the same ∂ρ/∂T.

`n2_local`, the interface pressure and `kpp_delta_rho` are **bit for bit** equal to compiled MOM6 on 30 columns and
both EOS (`tests/test_eos_closures.py`). The thermobaric column has $N^2 > 0$ at 100 m (2.3e-6 s⁻² Wright,
1.9e-6 Roquet) and $N^2 < 0$ at 4000 m (−1.9e-5 s⁻²).

!!! warning "Open items in the closures"
    - **BBL near-bed N², class (a), fix deferred.** MOM6 `find_N2` replaces N² at the two lowest interfaces with a
        bulk N²_bot (`MOM_set_diffusivity.F90:1202-1258`). pyOCEAN's BBL port omits this replacement. Fixing it would
        change linear-EOS BBL runs and the validated DOME numbers, so it waits for Esteban's approval. Measured
        effect on Kd (`validation/mom6_column/bbl_n2bot_effect.py`, `mom6_eos.md` §5): ≤ 1.6 % in smooth columns,
        2–37 % in random columns, and 100 % in one column where the energy runs out at a different interface.
    - **Jackson, class (c).** With a nonlinear EOS, Jackson matches MOM6 only at the starting N². MOM6 projects T and
        S separately inside the sub-steps; pyOCEAN does not port that projection, so the sub-step N² differs at
        second order.

## Hybrid regrid

With a nonlinear EOS the HYCOM1 rule follows the **potential density at `p_ref`** (default 2e7 Pa, so
$\rho = \sigma_2 + 1000$; `ALECoordinate.regrid_density`, `core/components/vertical_coordinate.py:80`,
`MOM_regridding.F90:2058-2062`). The column rule (`core.vertical.regrid.hycom1_column`) treats it as it treats the
linear density: massless layers first take the density of the nearest massive layer above (HYCOM
`hybgen.F90:904-925`), and only then is the column forced monotonic as in MOM6, with a running minimum from the bed
upward (`coord_hycom.F90:214-219`). Monotonising before the fill would let the stale light water of a massless bed
layer lighten the whole column (final review C1). The targets `regrid_params['rho_target']` are given in that density. Pre-flight refuses
targets outside 1020–1045 kg m⁻³ (`SIGMA2_RANGE`).

## Horizontal diffusion with a reference profile

With a nonlinear EOS and `rho_ref="auto"`, `kh` acts on $T - T_{ref}(z)$ and $S - S_{ref}(z)$ of the same
`TSReferenceProfile`. With the linear EOS, $T_{ref}$ still comes from inverting $\rho_{ref}$ with `t_coef`.

## Surface thermodynamics

- **Heat capacity.** The default `cp` is $c_p^0 = 3991.86795711963$ J kg⁻¹ K⁻¹ for both nonlinear EOS. For
    `teos10` this value is exact, because the heat content is $c_p^0\,\Theta$. For `wright_full` it is MOM6's
    default. The NWA12 case passes 3992.0. The linear EOS keeps 3985. With `teos10` the heat content
    $c_p^0 \sum V\Theta$ closes to ≤ 1e-13 of the change, with bulk fluxes and with prescribed heat
    (`tests/test_eos_surface.py`).
- **Bulk fluxes** (N2b, [Air–sea fluxes](air-sea-fluxes.md)) receive SST and SSS in θ and S_P. With `teos10` the
    model first averages Θ and S_A over the top `h_sfc`, then converts: $S_P = (35/35.16504) S_A$ and
    θ = `gsw.pt_from_CT`. This is the order of MOM6's coupler (`FMS_cap/ocean_model_MOM.F90:879-899`). The heat
    flux that comes back enters Θ with $c_p^0$.
- **Freezing point.** It is a function of the EOS only: MOM6's `LINEAR` form for `wright_full`, as NWA12
    (T_f = 0 − 0.054 S − 7.75e-8 p), and `gsw.CT_freezing` for `teos10`. The model has no frazil or ice. The model
    attribute `n_below_freezing` (registry diagnostic) counts the cells below the freezing point in the last step
    (nonlinear EOS with salinity; the linear EOS has no freezing law and the count stays 0). `gsw` is evaluated
    only on the cold cells (T < 0.5 °C).

## Inputs and outputs

- **`ts_kind`.** T and S given to the model are declared as `"theta_sp"` (default; θ and S_P, what
    `forcing/ocean_init.py` delivers) or `"ct_sa"` (Θ and S_A). A `teos10` model converts `"theta_sp"` data once,
    when it receives them. This applies to `initial_state(..., ts_kind=)`, the constructor's open-boundary exterior
    and sponge profiles (constructor `ts_kind`), and `forcing.ocean_init` (`eos_ts_kind`). Data that are already
    Θ/S_A must be passed with `ts_kind="ct_sa"`, or they are converted twice.
- **Output.** `OutputWriter(..., model=m)` takes the EOS from the model (or `eos=` by name; a value that
    contradicts the model is refused) and writes the prognostic variables with their CF standard names:
    `sea_water_conservative_temperature` and `sea_water_absolute_salinity` with `teos10`. With `teos10` it also
    writes θ and S_P as diagnostics.
- **T-S diagram.** `diagnostics/ts_diagram.py` provides `ts_histogram` (volume per T-S class) and `plot_ts`, with
    σ₀ or σ₂ contours of the same EOS.
- **Restart.** The restart stores the EOS name, ρ₀, c_p, `p_ref` and `pgf_mass_weight`, and checks them on load.
    Restarts are bit for bit, and 1 thread = 5 threads bit for bit, with `wright_full` and `teos10` in σ, z\* and
    hybrid (`tests/test_teos10_validation.py`).

## What the validation shows

From `validation/teos10/README.md` (WOA23 Gulf Stream profile, 39.5 °N 70.5 °W):

- **Rest over steep topography** (seamount and the BayIW mouth step, z\* and hybrid, 10 days, `rho_ref="auto"`):
    - With `wright_full`/`teos10` the spurious currents are 0.59–1.25× those of the equivalent linear EOS (limit 2×).
    - The force is **not zero at round-off** next to z\* partial bed cells: 1.4e-7 m s⁻². The reason is the
        interpolation of the 400-bin reference profile, class (c).
    - In the hybrid the force is 2e-5 m s⁻² (seamount) and 3e-6 m s⁻² (step). This is truncation of thick layers,
        class (c).
    - Both residuals are the same with the linear EOS. The hybrid seamount residual falls from 0.64 to 0.0021 m s⁻¹
        between 10 and 50 layers.
- **Cabbeling.** In a lock exchange of two waters of equal density, both nonlinear EOS make denser water in the
    first hour, and it sinks. The linear EOS makes nothing (max|u| 3.7e-9 m s⁻¹).
- **Thermobaricity.** A cold, fresh layer over a warm, salty one is stable with the interface at 100 m and
    overturns with the interface at 4000 m (bed T 4.00 → 0.20 °C). The linear control is stable at any depth.
- **Budgets.** The variance budget of T and S closes to ≤ 1e-12 with the three EOS; volume and salt to ≤ 1e-13.
- **ERA5 bulk column** (N2b). `teos10` − `wright_full` = −0.009 °C in final SST. Most of it is c_p: `wright_full`
    puts the flux into θ with $c_p^0$ while the real c_p near the surface is ≈ 4018. The two nonlinear EOS end
    +0.09/+0.10 °C above the linear one, because the nonlinear α is 46 % larger at 25 °C.

## Cost

Measured with `scripts/eos_cost.py`: BayIW cut z\*, 50 layers, Δt 120 s, 5 threads, the three EOS alternated in
separate processes, 3 rounds. The ratio per step against the linear EOS is in the table below. Most of the extra
time is in the explicit-momentum phase, which holds the nonlinear finite-volume pressure gradient: with the T/S
reference profile about 70 EOS evaluations per cell and layer, MOM6's method as is (Esteban's decision 4).

| EOS | s/step (median of 3 rounds) | ratio to linear (min/median/max) | criterion 11 (≤ 2.5×) |
|---|---|---|---|
| `linear` | 0.390 | 1 | — |
| `wright_full` | 0.534 | 1.364 / 1.369 / 1.375 | met |
| `teos10` | 0.729 | 1.866 / 1.872 / 1.873 | met |

| Phase (mean ms per step) | linear | wright_full | teos10 |
|---|---|---|---|
| explicit momentum (PGF) | 64.3 | 190.5 (+126.2) | 377.6 (+313.3) |
| tracers | 123.0 | 143.6 (+20.7) | 148.6 (+25.6) |
| vertical coordinate | 26.6 | 25.8 | 29.5 (+2.8) |
| all other phases | 170.5 | 167.6 | 168.8 |
| sum | 384.3 | 527.5 | 724.4 |

These figures follow three exact optimisations of 2026-11-01. Every result is unchanged bit for bit. Before them
`teos10` cost 3.16× and `wright_full` 1.90×.

1. The scalar EOS kernels are inlined (`inline="always"`). Before, the Roquet polynomial was called out of line
   at every point, and one evaluation fell from 13.7 to 5.3 ns (`scripts/eos_microbench.py`).
2. In the profile mode, one walked index into the reference profile serves T_ref and S_ref at each point. Before,
   each point ran two bisections.
3. The column pass makes no allocation per column and no EOS calls in inactive columns.

The remaining extra time is the EOS arithmetic itself.

## References

- Adcroft, A., Hallberg, R. & Harrison, M. (2008). A finite volume discretization of the pressure gradient force
    using analytic integration. *Ocean Modell.* 22, 106–113. doi:10.1016/j.ocemod.2008.02.001
- IOC, SCOR & IAPSO (2010). *The international thermodynamic equation of seawater – 2010.* Manuals and Guides 56,
    UNESCO.
- McDougall, T. J. (1987). Thermobaricity, cabbeling, and water-mass conversion. *J. Geophys. Res.* 92,
    5448–5464. **[verificar]**
- Roquet, F., Madec, G., McDougall, T. J. & Barker, P. M. (2015). Accurate polynomial expressions for the density
    and specific volume of seawater using the TEOS-10 standard. *Ocean Modell.* 90, 29–43.
    doi:10.1016/j.ocemod.2015.04.002 **[verificar]**
- Vallis, G. K. (2006). *Atmospheric and Oceanic Fluid Dynamics.* Cambridge University Press.
- Wright, D. G. (1997). An equation of state for use in ocean models: Eckart's formula revisited.
    *J. Atmos. Oceanic Technol.* 14, 735–740. **[verificar]**
