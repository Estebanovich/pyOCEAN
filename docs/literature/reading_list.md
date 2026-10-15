# Lista de lectura de pyOCEAN: artículos por descargar a la biblioteca local

**Biblioteca local:** `/Users/estebancruz/Documents/TESIS_InternalWaves-2/Literatura`

**Formato:** cada artículo va como `Autor_Año - Título.pdf`. `python scripts/litsearch.py --convert` crea la copia `.md` que usa la búsqueda.

**Procedimiento** (también en `CLAUDE.md`, "Systematic literature checks"):
1. Buscar primero en la biblioteca local: `python scripts/litsearch.py "término 1" "término 2"`.
2. Solo si ahí no está, buscar en Consensus, con 1–2 búsquedas enfocadas por decisión.
3. Agregar aquí cada artículo nuevo que se use, con su DOI. Cuando esté descargado, cambiar el estado a **local** y correr `--convert`.

**PDF escaneados** (sin capa de texto: la búsqueda local no los ve; se leen como imágenes): Beckmann_Haidvogel_1993, Haney_1991.

**Leyenda:**
- ★ = se usa directamente en el código o en un benchmark; descargar primero.
- Estado: ⬜ por descargar · ✅ local.
- [verificar] = DOI escrito de memoria, no salió de Consensus; comprobarlo al descargar.

## 1. Mezcla espuria y diagnósticos (etapa A, `diagnostics/mixing.py`, `validation/lock_exchange`)

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| ★ | Ilicak et al. 2012, Spurious dianeutral mixing and the role of momentum closure, Ocean Modell. | [10.1016/j.ocemod.2011.10.003](https://doi.org/10.1016/j.ocemod.2011.10.003) | `Ilicak_2012 - Spurious Dianeutral Mixing and the Role of Momentum Closure.pdf` | ✅ |
| ★ | Burchard & Rennau 2008, Comparative quantification of physically and numerically induced mixing, Ocean Modell. | [10.1016/j.ocemod.2007.10.003](https://doi.org/10.1016/j.ocemod.2007.10.003) | `Burchard_Rennau_2008 - Comparative Quantification of Physically and Numerically Induced Mixing.pdf` | ✅ |
| ★ | Legg, Hallberg & Girton 2006, Comparison of entrainment in overflows simulated by z-coordinate, isopycnal and non-hydrostatic models (DOME), Ocean Modell. 11 | [10.1016/j.ocemod.2004.11.006](https://doi.org/10.1016/j.ocemod.2004.11.006) | `Legg_2006 - Comparison of Entrainment in Overflows Simulated by z-Coordinate Isopycnal and Non-Hydrostatic Models.pdf` | ✅ |
| ★ | Ezer & Mellor 2004, A generalized coordinate ocean model and a comparison of the bottom boundary layer dynamics in terrain-following and in z-level grids (definición original de DOME), Ocean Modell. 6 | [10.1016/s1463-5003(03)00026-x](https://doi.org/10.1016/s1463-5003(03)00026-x) | `Ezer_Mellor_2004 - A Generalized Coordinate Ocean Model and a Comparison of the Bottom Boundary Layer Dynamics.pdf` | ✅ |
| | Ezer 2004, Entrainment, diapycnal mixing and transport in three-dimensional bottom gravity current simulations using the Mellor–Yamada turbulence scheme, Ocean Modell. 9 | [10.1016/j.ocemod.2004.06.001](https://doi.org/10.1016/j.ocemod.2004.06.001) | `Ezer_2004 - Entrainment Diapycnal Mixing and Transport in Three-Dimensional Bottom Gravity Current Simulations.pdf` | ✅ |
| ★ | Klingbeil et al. 2014, Discrete variance decay in a finite-volume framework, Ocean Modell. | [10.1016/j.ocemod.2014.06.001](https://doi.org/10.1016/j.ocemod.2014.06.001) | `Klingbeil_2014 - Quantification of Spurious Dissipation and Mixing.pdf` | ✅ |
| ★ | Winters et al. 1995, Available potential energy and mixing in density-stratified fluids, JFM | [10.1017/S002211209500125X](https://doi.org/10.1017/S002211209500125X) | `Winters_1995 - Available Potential Energy and Mixing in Density-Stratified Fluids.pdf` | ✅ |
| ★ | Gibson et al. 2017, Horizontal and vertical contributions to spurious mixing in an ALE model, Ocean Modell. | [10.1016/j.ocemod.2017.09.008](https://doi.org/10.1016/j.ocemod.2017.09.008) | `Gibson_2017 - Attribution of Horizontal and Vertical Contributions to Spurious Mixing.pdf` | ✅ |
| | Ilicak 2016, Quantifying spatial distribution of spurious mixing, Ocean Modell. | [10.1016/j.ocemod.2016.11.002](https://doi.org/10.1016/j.ocemod.2016.11.002) | `Ilicak_2016 - Quantifying Spatial Distribution of Spurious Mixing.pdf` | ✅ |
| | Ilicak 2014, Energetics and mixing efficiency of lock-exchange flow, Ocean Modell. | [10.1016/j.ocemod.2014.08.003](https://doi.org/10.1016/j.ocemod.2014.08.003) | `Ilicak_2014 - Energetics and Mixing Efficiency of Lock-Exchange Flow.pdf` | ✅ |
| | Banerjee et al. 2024, Discrete variance decay analysis of spurious mixing, Ocean Modell. | [10.1016/j.ocemod.2024.102460](https://doi.org/10.1016/j.ocemod.2024.102460) | `Banerjee_2024 - Discrete Variance Decay Analysis of Spurious Mixing.pdf` | ✅ |
| | Schlichting et al. 2023, Physical and numerical mixing from salinity variance budgets, JAMES | [10.1029/2022MS003380](https://doi.org/10.1029/2022MS003380) | `Schlichting_2023 - Quantification of Physical and Numerical Mixing in a Coastal Ocean Model.pdf` | ✅ |
| | Schlichting et al. 2024, Numerical mixing suppresses submesoscale baroclinic instabilities, JAMES | [10.1029/2024MS004321](https://doi.org/10.1029/2024MS004321) | `Schlichting_2024 - Numerical Mixing Suppresses Submesoscale Baroclinic Instabilities.pdf` | ✅ |
| | Burchard 2012, Numerically induced mixing and dissipation in shallow water equations, GEM | [10.1007/s13137-012-0033-6](https://doi.org/10.1007/s13137-012-0033-6) | `Burchard_2012 - Numerically Induced Mixing and Dissipation in Shallow Water Equations.pdf` | ✅ |
| | Rennau & Burchard 2009, Numerically induced mixing in a coastal model application, Ocean Dyn. | [10.1007/s10236-009-0201-x](https://doi.org/10.1007/s10236-009-0201-x) | `Rennau_2009 - Quantitative Analysis of Numerically Induced Mixing in a Coastal Model.pdf` | ✅ |
| | Kalra et al. 2019, Physical vs numerical mixing with different tracer advection schemes, JMSE | [10.3390/jmse7100338](https://doi.org/10.3390/jmse7100338) | `Kalra_2019 - Comparison of Physical to Numerical Mixing.pdf` | ✅ |
| | Griffies et al. 2000, Spurious diapycnal mixing with advection in a z-coordinate model, MWR | [10.1175/1520-0493(2000)128<0538:SDMAWA>2.0.CO;2](https://doi.org/10.1175/1520-0493(2000)128%3C0538:SDMAWA%3E2.0.CO;2) | `Griffies_2000 - Spurious Diapycnal Mixing Associated with Advection.pdf` | ✅ |
| | Holmes et al. 2020, The geography of numerical mixing in global ocean models, JAMES | [10.1029/2020MS002333](https://doi.org/10.1029/2020MS002333) | `Holmes_2020 - The Geography of Numerical Mixing.pdf` | ✅ |
| | Garinet et al. 2024, Spurious numerical mixing under strong tidal forcing (Symphonie), GMD | [10.5194/gmd-17-6967-2024](https://doi.org/10.5194/gmd-17-6967-2024) | `Garinet_2024 - Spurious Numerical Mixing under Strong Tidal Forcing.pdf` | ✅ |
| | Schifano et al. 2025, Diapycnal mixing in a terrain-following coordinate model (CROCO), JAMES | [10.1029/2024MS004768](https://doi.org/10.1029/2024MS004768) | `Schifano_2025 - Diapycnal Mixing and Tracer Dispersion in a Terrain-Following Model.pdf` | ✅ |
| | Hiester et al. 2014, Spurious mixing in adaptive-mesh lock exchange, Ocean Modell. 73 | [10.1016/j.ocemod.2013.10.003](https://doi.org/10.1016/j.ocemod.2013.10.003) | `Hiester_2014 - Spurious Mixing in Adaptive Mesh Lock-Exchange.pdf` | ✅ |
| | Özgökmen et al. 2009, Reynolds-number dependence of lock-exchange mixing, Ocean Modell. | [10.1016/j.ocemod.2009.06.013](https://doi.org/10.1016/j.ocemod.2009.06.013) | `Ozgokmen_2009 - Reynolds Number Dependence of Mixing in a Lock-Exchange.pdf` | ✅ |
| | Hill et al. 2012, Controlling spurious diapycnal mixing in eddy-resolving z-models, Ocean Modell. | [10.1016/j.ocemod.2011.12.001](https://doi.org/10.1016/j.ocemod.2011.12.001) | `Hill_2012 - Controlling Spurious Diapycnal Mixing in Eddy-Resolving Height-Coordinate Models.pdf` | ✅ |
| | Drake et al. 2025, Water mass transformation budgets in FV generalized-coordinate models, JAMES | [10.1029/2024MS004383](https://doi.org/10.1029/2024MS004383) | `Drake_2025 - Water Mass Transformation Budgets in Finite-Volume GVC Models.pdf` | ✅ |

## 2. Coordenada vertical, ALE y HYCOM (etapas B y C)

Consensus (2026-10-11): dos búsquedas para el diseño ALE (`docs/superpowers/specs/`).

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| | Petersen et al. 2015, ALE vertical coordinate in MPAS-Ocean, Ocean Modell. | [10.1016/j.ocemod.2014.12.004](https://doi.org/10.1016/j.ocemod.2014.12.004) | `Petersen_2015 - Evaluation of the ALE Vertical Coordinate in MPAS-Ocean.pdf` | ✅ |
| | Megann et al. 2022, The z̃ time-filtered ALE coordinate in NEMO, JAMES | [10.1029/2022MS003056](https://doi.org/10.1029/2022MS003056) | `Megann_2022 - Assessment of the z-tilde Coordinate in a Global Eddy-Permitting Model.pdf` | ✅ |
| | Megann 2024, Numerical mixing in a tidally forced global model, Ocean Modell. | [10.1016/j.ocemod.2024.102329](https://doi.org/10.1016/j.ocemod.2024.102329) | `Megann_2024 - Quantifying Numerical Mixing in a Tidally Forced Global Model.pdf` | ✅ |
| | Chang et al. 2026, Vertical coordinates in regional MOM6 (NW Pacific), GMD | [10.5194/gmd-19-187-2026](https://doi.org/10.5194/gmd-19-187-2026) | `Chang_2026 - Assessing Vertical Coordinate Performance in Regional MOM6.pdf` | ✅ |
| | Bruciaferri et al. 2024, Localized general vertical coordinates, JAMES | [10.1029/2023MS003893](https://doi.org/10.1029/2023MS003893) | `Bruciaferri_2024 - Localized General Vertical Coordinates for Quasi-Eulerian Models.pdf` | ✅ |
| | Halliwell 2004, Vertical coordinate and mixing algorithms in HYCOM, Ocean Modell. | [10.1016/j.ocemod.2003.10.002](https://doi.org/10.1016/j.ocemod.2003.10.002) | `Halliwell_2004 - Evaluation of Vertical Coordinate and Vertical Mixing Algorithms in HYCOM.pdf` | ✅ |
| | Klingbeil et al. 2019, Reducing spurious diapycnal mixing in ocean models (book chapter) | [10.1007/978-3-030-05704-6_8](https://doi.org/10.1007/978-3-030-05704-6_8) | `Eden_Iske_2019 - Energy Transfers in Atmosphere and Ocean.pdf` (libro completo; capítulo de Klingbeil et al.) | ✅ |
| ★ | Griffies et al. 2020, A primer on the vertical Lagrangian-remap method in ocean models based on finite volume generalized vertical coordinates, JAMES | [10.1029/2019MS001954](https://doi.org/10.1029/2019ms001954) | `Griffies_2020 - A Primer on the Vertical Lagrangian-Remap Method in Ocean Models.pdf` | ✅ |
| ★ | White & Adcroft 2008, A high-order finite volume remapping scheme for nonuniform grids: PQM, J. Comput. Phys. | [10.1016/j.jcp.2008.04.026](https://doi.org/10.1016/j.jcp.2008.04.026) | `White_Adcroft_2008 - A High-Order Finite Volume Remapping Scheme for Nonuniform Grids PQM.pdf` | ✅ |
| ★ | White, Adcroft & Hallberg 2009, High-order regridding–remapping schemes for continuous isopycnal and generalized coordinates, J. Comput. Phys. | [10.1016/j.jcp.2009.08.016](https://doi.org/10.1016/j.jcp.2009.08.016) | `White_2009 - High-Order Regridding-Remapping Schemes for Continuous Isopycnal and Generalized Coordinates.pdf` | ✅ |
| ★ | Bleck 2002, An oceanic general circulation model framed in hybrid isopycnic-Cartesian coordinates, Ocean Modell. | [10.1016/S1463-5003(01)00012-9](https://doi.org/10.1016/s1463-5003(01)00012-9) | `Bleck_2002 - An Oceanic General Circulation Model Framed in Hybrid Isopycnic-Cartesian Coordinates.pdf` | ✅ |
| ★ | Adcroft et al. 2019, The GFDL global ocean and sea ice model OM4.0 (MOM6 hybrid z*–isopycnal), JAMES | [10.1029/2019MS001726](https://doi.org/10.1029/2019ms001726) | `Adcroft_2019 - The GFDL Global Ocean and Sea Ice Model OM4.0.pdf` | ✅ |
| | Adcroft & Hallberg 2006, On methods for solving the oceanic equations of motion in generalized vertical coordinates, Ocean Modell. | [10.1016/j.ocemod.2004.12.007](https://doi.org/10.1016/j.ocemod.2004.12.007) | `Adcroft_Hallberg_2006 - On Methods for Solving the Oceanic Equations of Motion in Generalized Vertical Coordinates.pdf` | ✅ |
| | Hofmeister, Burchard & Deleersnijder 2010, Non-uniform adaptive vertical grids for 3D ocean models, Ocean Modell. | [10.1016/j.ocemod.2009.12.003](https://doi.org/10.1016/j.ocemod.2009.12.003) | `Hofmeister_2010 - Non-Uniform Adaptive Vertical Grids for 3D Numerical Ocean Models.pdf` | ✅ |
| | Chen et al. 2024, Gradient-tracking generalized vertical coordinate in ROMS (ALE), J. Mar. Sci. Eng. | [10.3390/jmse12122249](https://doi.org/10.3390/jmse12122249) | `Chen_2024 - Gradient-Tracking Generalized Vertical Coordinate for Ocean Models ROMS.pdf` | ✅ |
| ★ | Hallberg & Adcroft 2009, Reconciling estimates of the free surface height in Lagrangian vertical coordinate ocean models, Ocean Modell. 29 (base de `zonal_flux_adjust` de MOM6: flujos de capa consistentes con el barotrópico) | [10.1016/j.ocemod.2009.02.008](https://doi.org/10.1016/j.ocemod.2009.02.008) | `Hallberg_Adcroft_2009 - Reconciling Estimates of the Free Surface Height in Lagrangian Vertical Coordinate Ocean Models.pdf` | ✅ |
| ★ | Higdon 2005, A two-level time-stepping method for layered ocean circulation models: further development and testing, J. Comput. Phys. (flujos de masa entre celdas gruesas y delgadas sobre topografía, capas que se desvanecen) | [10.1016/j.jcp.2004.12.011](https://doi.org/10.1016/j.jcp.2004.12.011) | `Higdon_2005 - A Two-Level Time-Stepping Method for Layered Ocean Circulation Models Further Development.pdf` | ✅ |
| | Higdon 2020, Discontinuous Galerkin methods for multi-layer ocean modeling: viscosity and thin layers, J. Comput. Phys. | [10.1016/j.jcp.2019.109018](https://doi.org/10.1016/j.jcp.2019.109018) | `Higdon_2020 - Discontinuous Galerkin Methods for Multi-Layer Ocean Modeling Viscosity and Thin Layers.pdf` | ✅ |
| | Adcroft & Campin 2004, Rescaled height coordinates for accurate representation of free-surface flows (z*), Ocean Modell. 7, 269 | [10.1016/j.ocemod.2003.09.003](https://doi.org/10.1016/j.ocemod.2003.09.003) | `Adcroft_Campin_2004 - Rescaled Height Coordinates for Accurate Representation of Free-Surface Flows.pdf` | ✅ |

## 3. Advección, pasos de tiempo y frentes (`core/dynamics/sigma_baroclinic.py`)

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| ★ | Shchepetkin & McWilliams 2005, ROMS split-explicit kernel, Ocean Modell. | [10.1016/j.ocemod.2004.08.002](https://doi.org/10.1016/j.ocemod.2004.08.002) | `Shchepetkin_McWilliams_2005 - The Regional Oceanic Modeling System ROMS.pdf` | ✅ |
| ★ | Jiang & Shu 1996, Efficient implementation of weighted ENO schemes, JCP | [10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130) | `Jiang_Shu_1996 - Efficient Implementation of Weighted ENO Schemes.pdf` | ✅ |
| ★ | Borges et al. 2008, An improved WENO scheme (WENO-Z), JCP | [10.1016/j.jcp.2007.11.038](https://doi.org/10.1016/j.jcp.2007.11.038) | `Borges_2008 - An Improved Weighted Essentially Non-Oscillatory Scheme.pdf` | ✅ |
| ★ | Ungarish et al. 2018, Internal jumps and fronts of gravity currents, JFM | [10.1017/jfm.2018.219](https://doi.org/10.1017/jfm.2018.219) | `Ungarish_2018 - Models of Internal Jumps and the Fronts of Gravity Currents.pdf` | ✅ |
| ★ | Borden & Meiburg 2013, Circulation-based models for Boussinesq gravity currents, Phys. Fluids | [10.1063/1.4825035](https://doi.org/10.1063/1.4825035) | `Borden_Meiburg_2013 - Circulation Based Models for Boussinesq Gravity Currents.pdf` | ✅ |
| ★ | Benjamin 1968, Gravity currents and related phenomena, JFM | [10.1017/S0022112068000133](https://doi.org/10.1017/S0022112068000133) [verificar] | `Benjamin_1968 - Gravity Currents and Related Phenomena.pdf` | ✅ |
| | Haidvogel et al. 2008, Ocean forecasting in terrain-following coordinates (ROMS), JCP | [10.1016/j.jcp.2007.06.016](https://doi.org/10.1016/j.jcp.2007.06.016) | `Haidvogel_2008 - Ocean Forecasting in Terrain-Following Coordinates.pdf` | ✅ |
| | Shchepetkin 2015, Adaptive Courant-number-dependent implicit vertical advection, Ocean Modell. | [10.1016/j.ocemod.2015.03.006](https://doi.org/10.1016/j.ocemod.2015.03.006) | `Shchepetkin_2015 - An Adaptive Courant-Number-Dependent Implicit Scheme for Vertical Advection.pdf` | ✅ |
| | Silvestri et al. 2024, WENO-based momentum advection, JAMES | [10.1029/2023MS004130](https://doi.org/10.1029/2023MS004130) | `Silvestri_2024 - A New WENO-Based Momentum Advection Scheme.pdf` | ✅ |
| | Xue & Lin 2001, Numerical equivalence of flux and advective forms, MWR | [10.1175/1520-0493(2001)129<0561:NEOAIF>2.0.CO;2](https://doi.org/10.1175/1520-0493(2001)129%3C0561:NEOAIF%3E2.0.CO;2) | `Xue_2001 - Numerical Equivalence of Advection in Flux and Advective Forms.pdf` | ✅ |
| | Cantero et al. 2007, On the front velocity of gravity currents, JFM | [10.1017/S0022112007005769](https://doi.org/10.1017/S0022112007005769) | `Cantero_2007 - On the Front Velocity of Gravity Currents.pdf` | ✅ |
| ★ | Zalesak 1979, Fully multidimensional flux-corrected transport algorithms for fluids, J. Comput. Phys. (FCT de la §4 del spec de robustez de la híbrida) | [10.1016/0021-9991(79)90051-2](https://doi.org/10.1016/0021-9991(79)90051-2) | `Zalesak_1979 - Fully Multidimensional Flux-Corrected Transport Algorithms for Fluids.pdf` | ✅ |

## 4. Mojado-secado, fricción y turbulencia (`wetting_drying.py`, `sigma_wd.py`, `gls.py`, `bottom_friction.py`)

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| ★ | Casulli & Cheng 1992, Semi-implicit 3-D shallow water flow, IJNMF | [10.1002/fld.1650150602](https://doi.org/10.1002/fld.1650150602) [verificar] | `Casulli_Cheng_1992 - Semi-Implicit Finite Difference Methods for 3D Shallow Water Flow.pdf` | ✅ |
| ★ | Casulli 2009, High-resolution wetting and drying (subgrid), IJNMF | [10.1002/fld.1896](https://doi.org/10.1002/fld.1896) | `Casulli_2009 - A High-Resolution Wetting and Drying Algorithm for Free-Surface Hydrodynamics.pdf` | ✅ |
| ★ | Stelling & Duinmeijer 2003, A staggered conservative scheme for every Froude number in rapidly varied shallow water flows, IJNMF 43:1329–1354 | [10.1002/fld.537](https://doi.org/10.1002/fld.537) (el DOI fld.595 de antes era otro artículo) | `Stelling_Duinmeijer_2003 - A Staggered Conservative Scheme for Every Froude Number.pdf` | ✅ |
| | Stelling & Zijlema 2003, Non-hydrostatic free-surface flow (SWASH-type), IJNMF 43:1–23 — etapa E | [10.1002/fld.595](https://doi.org/10.1002/fld.595) | `Stelling_Zijlema_2003 - An Accurate and Efficient Finite-Difference Algorithm for Non-Hydrostatic Free-Surface Flow.pdf` | ✅ |
| ★ | Thacker 1981, Some exact solutions to the nonlinear shallow-water wave equations, JFM | [10.1017/S0022112081001882](https://doi.org/10.1017/S0022112081001882) [verificar] | `Thacker_1981 - Some Exact Solutions to the Nonlinear Shallow-Water Wave Equations.pdf` | ✅ |
| ★ | Umlauf & Burchard 2003, A generic length-scale equation, J. Mar. Res. | [10.1357/002224003322005087](https://doi.org/10.1357/002224003322005087) [verificar] | `Umlauf_Burchard_2003 - A Generic Length-Scale Equation for Geophysical Turbulence Models.pdf` | ✅ |
| ★ | Warner et al. 2005, Four turbulence closures via GLS, Ocean Modell. | [10.1016/j.ocemod.2003.12.003](https://doi.org/10.1016/j.ocemod.2003.12.003) [verificar] | `Warner_2005 - Performance of Four Turbulence Closure Models Implemented Using GLS.pdf` | ✅ |
| ★ | Canuto et al. 2001, Ocean turbulence Part I, JPO | [10.1175/1520-0485(2001)031<1413:OTPIOP>2.0.CO;2](https://doi.org/10.1175/1520-0485(2001)031%3C1413:OTPIOP%3E2.0.CO;2) [verificar] | `Canuto_2001 - Ocean Turbulence Part I.pdf` | ✅ |
| ★ | Soulsby 1997, Dynamics of Marine Sands (libro, §3: C_d log-law) | ISBN 0-7277-2584-X [verificar] | `Soulsby_1997 - Dynamics of Marine Sands.pdf` | ⬜ |

## 5. Diferenciabilidad y adjuntos (etapa D)

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| | Koldunov et al. 2026, FESOM2-JAX: a differentiable shadow of FESOM2 | [10.48550/arXiv.2608.01546](https://doi.org/10.48550/arXiv.2608.01546) | `Koldunov_2026 - FESOM2-JAX.pdf` | ✅ |
| | Meunier et al. 2025, Towards fully differentiable neural ocean model with Veros | [10.48550/arXiv.2511.17427](https://doi.org/10.48550/arXiv.2511.17427) | `Meunier_2025 - Towards Fully Differentiable Neural Ocean Model with Veros.pdf` | ✅ |
| | Cardoso-Bihlo et al. 2026, AegirJAX: end-to-end differentiable coastal hydrodynamics | [10.48550/arXiv.2604.07129](https://doi.org/10.48550/arXiv.2604.07129) | `CardosoBihlo_2026 - A Solver-in-the-Loop Framework for Differentiable Coastal Hydrodynamics.pdf` | ✅ |
| | Moses et al. 2026, DJ4Earth: differentiable Earth system modeling (Enzyme/Reactant), JAMES | [10.1029/2025MS005615](https://doi.org/10.1029/2025MS005615) | `Moses_2026 - DJ4Earth Differentiable and Performance-Portable Earth System Modeling.pdf` | ✅ |
| | Gaikwad et al. 2025, MITgcm-AD v2 (Tapenade), FGCS | [10.1016/j.future.2024.107512](https://doi.org/10.1016/j.future.2024.107512) | `Gaikwad_2025 - MITgcm-AD v2.pdf` | ✅ |
| | Heimbach et al. 2005, Exact adjoint of the parallel MITgcm, FGCS | [10.1016/j.future.2004.11.010](https://doi.org/10.1016/j.future.2004.11.010) | `Heimbach_2005 - An Efficient Exact Adjoint of the Parallel MIT General Circulation Model.pdf` | ✅ |
| | Song et al. 2024, Adjoint-based gradients for implicit differentiable modeling, HESS | [10.5194/hess-28-3051-2024](https://doi.org/10.5194/hess-28-3051-2024) | `Song_2024 - Adjoint-Based Gradients for Implicit Differentiable Modeling.pdf` | ✅ |
| | Liu et al. 2026, JAX-AMG: differentiable sparse linear solver, SoftwareX | [10.1016/j.softx.2026.102966](https://doi.org/10.1016/j.softx.2026.102966) | `Liu_2026 - JAX-AMG.pdf` | ✅ |
| | Bezgin et al. 2023, JAX-Fluids, Comput. Phys. Commun. | [10.1016/j.cpc.2022.108527](https://doi.org/10.1016/j.cpc.2022.108527) | `Bezgin_2023 - JAX-FLUIDS.pdf` | ✅ |
| | Tber et al. 2007, Tangent and adjoint of OPA with Tapenade | [10.48550/arXiv.0711.4444](https://doi.org/10.48550/arXiv.0711.4444) | `Tber_2007 - Building the Tangent and Adjoint Codes of OPA with TAPENADE.pdf` | ✅ |

## 6. ML y emuladores (etapa F)

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| | Lupin-Jimenez et al. 2025, Regional ocean emulators, Gulf of Mexico | [10.48550/arXiv.2501.05058](https://doi.org/10.48550/arXiv.2501.05058) | `LupinJimenez_2025 - Simultaneous Emulation and Downscaling with Regional Ocean Emulators.pdf` | ✅ |
| | Zavala-Romero et al. 2025, CNNs for sea-surface DA in HYCOM, Gulf of Mexico, Ocean Sci. | [10.5194/os-21-113-2025](https://doi.org/10.5194/os-21-113-2025) | `ZavalaRomero_2025 - CNNs for Sea Surface Data Assimilation in Operational Ocean Models.pdf` | ✅ |
| | Chattopadhyay et al. 2024, OceanNet: neural-operator digital twin for regional oceans, Sci. Rep. | [10.1038/s41598-024-72145-0](https://doi.org/10.1038/s41598-024-72145-0) | `Chattopadhyay_2024 - OceanNet.pdf` | ✅ |
| | Dheeshjith et al. 2025, Samudra: AI global ocean emulator, GRL | [10.1029/2024GL114318](https://doi.org/10.1029/2024GL114318) | `Dheeshjith_2025 - Samudra.pdf` | ✅ |
| | Yuan et al. 2026, Samudra 2 | [10.48550/arXiv.2606.02610](https://doi.org/10.48550/arXiv.2606.02610) | `Yuan_2026 - Samudra 2.pdf` | ✅ |
| | Cui et al. 2025, WenHai: forecasting the eddying ocean, Nat. Commun. | [10.1038/s41467-025-57389-2](https://doi.org/10.1038/s41467-025-57389-2) | `Cui_2025 - Forecasting the Eddying Ocean with a Deep Neural Network.pdf` | ✅ |
| | Niu et al. 2026, FuXi-Ocean, npj Clim. Atmos. Sci. | [10.1038/s41612-026-01444-2](https://doi.org/10.1038/s41612-026-01444-2) | `Niu_2026 - FuXi-Ocean.pdf` | ✅ |
| | Holmberg et al. 2025, SeaCast: Mediterranean forecasting with graph DL, Sci. Rep. | [10.1038/s41598-025-31177-w](https://doi.org/10.1038/s41598-025-31177-w) | `Holmberg_2025 - Accurate Mediterranean Sea Forecasting via Graph-Based Deep Learning.pdf` | ✅ |
| | Espinosa et al. 2026, DLESyM-Ocean | [10.48550/arXiv.2608.11545](https://doi.org/10.48550/arXiv.2608.11545) | `Espinosa_2026 - DLESyM-Ocean.pdf` | ✅ |
| | Bolton & Zanna 2019, Deep learning for ocean data inference and subgrid parameterization, JAMES | [10.1029/2018MS001472](https://doi.org/10.1029/2018MS001472) | `Bolton_Zanna_2019 - Applications of Deep Learning to Ocean Data Inference and Subgrid Parameterization.pdf` | ✅ |
| | Sonnewald et al. 2021, Bridging observations, theory and simulation with ML, ERL | [10.1088/1748-9326/ac0eb0](https://doi.org/10.1088/1748-9326/ac0eb0) | `Sonnewald_2021 - Bridging Observations Theory and Numerical Simulation of the Ocean Using ML.pdf` | ✅ |

## 7. Preparación de dominio y gradiente de presión en σ (`diagnostics/domain_check.py`)

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| ★ | Beckmann & Haidvogel 1993, Numerical simulation of flow around a tall isolated seamount, JPO (rx0) | [10.1175/1520-0485(1993)023<1736:NSOFAA>2.0.CO;2](https://doi.org/10.1175/1520-0485(1993)023%3C1736:NSOFAA%3E2.0.CO;2) | `Beckmann_Haidvogel_1993 - Numerical Simulation of Flow around a Tall Isolated Seamount.pdf` | ✅ |
| ★ | Haney 1991, On the pressure gradient force over steep topography in sigma coordinate ocean models, JPO (rx1) | [10.1175/1520-0485(1991)021<0610:OTPGFO>2.0.CO;2](https://doi.org/10.1175/1520-0485(1991)021%3C0610:OTPGFO%3E2.0.CO;2) [verificar] | `Haney_1991 - On the Pressure Gradient Force over Steep Topography in Sigma Coordinate Models.pdf` | ✅ |
| ★ | Shchepetkin & McWilliams 2003, A method for computing horizontal pressure-gradient force in an oceanic model with a nonaligned vertical coordinate, JGR | [10.1029/2001JC001047](https://doi.org/10.1029/2001JC001047) | `Shchepetkin_McWilliams_2003 - Computing Horizontal Pressure-Gradient Force with a Nonaligned Vertical Coordinate.pdf` | ✅ |
| ✓ | Cushman-Roisin & Beckers 2011, §20.6 (error del gradiente de presión en σ, consistencia hidrostática) | libro | `CushmanRoisin_Beckers_2011 - Introduction to Geophysical Fluid Dynamics Physical and Numerical Aspects.pdf` | ✅ |
| ★ | Adcroft, Hallberg & Harrison 2008, A finite volume discretization of the pressure gradient force using analytic integration, Ocean Modell. | [10.1016/j.ocemod.2008.02.001](https://doi.org/10.1016/j.ocemod.2008.02.001) [verificar] | `Adcroft_2008 - A Finite Volume Discretization of the Pressure Gradient Force Using Analytic Integration.pdf` | ✅ |
| ★ | Song 1998, A general pressure gradient formulation for ocean models (density Jacobian), MWR | [10.1175/1520-0493(1998)126<3213:AGPGFF>2.0.CO;2](https://doi.org/10.1175/1520-0493(1998)126%3C3213:AGPGFF%3E2.0.CO;2) | `Song_1998 - A General Pressure Gradient Formulation for Ocean Models.pdf` | ✅ |
| ★ | Marsaleix et al. 2009, Low-order pressure gradient schemes: the seamount test revisited, Ocean Modell. | [10.1016/j.ocemod.2009.06.011](https://doi.org/10.1016/j.ocemod.2009.06.011) | `Marsaleix_2009 - Low-Order Pressure Gradient Schemes in Sigma Coordinate Models.pdf` | ✅ |
| | Berntsen et al. 2015, Internal pressure gradient errors in σ-models in fjords (balanced method), Ocean Modell. | [10.1016/j.ocemod.2015.05.009](https://doi.org/10.1016/j.ocemod.2015.05.009) | `Berntsen_2015 - Internal Pressure Gradient Errors in Sigma-Coordinate Ocean Models.pdf` | ✅ |
| | Engwirda et al. 2017, High-order finite-volume pressure gradient for layered models, Ocean Modell. | [10.1016/j.ocemod.2017.05.003](https://doi.org/10.1016/j.ocemod.2017.05.003) | `Engwirda_2017 - High-Order Accurate Finite-Volume Formulations for the Pressure Gradient Force.pdf` | ✅ |
| | Mellor, Oey & Ezer 1998, Sigma coordinate pressure gradient errors and the seamount problem, JAOT | [10.1175/1520-0426(1998)015<1122:SCPGEA>2.0.CO;2](https://doi.org/10.1175/1520-0426(1998)015%3C1122:SCPGEA%3E2.0.CO;2) | `Mellor_1998 - Sigma Coordinate Pressure Gradient Errors and the Seamount Problem.pdf` | ✅ |

## 8. Fronteras abiertas y forzamiento por viento (`core/dynamics/sigma_baroclinic.py`, etapa B, BayIW)

Consensus (2026-10-11): una búsqueda; la biblioteca local solo tenía menciones indirectas (Ramos-Musalem 2023, Haidvogel 2008, Chang 2026). Segunda búsqueda (2026-10-11, diseño ALE): fronteras con coordenadas generales y anidamiento; no hay literatura específica de OBC en ALE, la práctica está en el código de MOM6/HYCOM.

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| ★ | Orlanski 1976, A simple boundary condition for unbounded hyperbolic flows, J. Comput. Phys. | [10.1016/0021-9991(76)90023-1](https://doi.org/10.1016/0021-9991(76)90023-1) | `Orlanski_1976 - A Simple Boundary Condition for Unbounded Hyperbolic Flows.pdf` | ✅ |
| ★ | Chapman 1985, Numerical treatment of cross-shelf open boundaries in a barotropic coastal ocean model, JPO (esponja + Orlanski = la más transparente) | [10.1175/1520-0485(1985)015<1060:NTOCSO>2.0.CO;2](https://doi.org/10.1175/1520-0485(1985)015%3C1060:NTOCSO%3E2.0.CO;2) | `Chapman_1985 - Numerical Treatment of Cross-Shelf Open Boundaries.pdf` | ✅ (escaneado) |
| ★ | Marsaleix, Auclair & Estournel 2006, Considerations on open boundary conditions for regional and coastal ocean models, JAOT | [10.1175/jtech1930.1](https://doi.org/10.1175/jtech1930.1) | `Marsaleix_2006 - Considerations on Open Boundary Conditions for Regional and Coastal Ocean Models.pdf` | ✅ |
| ★ | Marchesiello, McWilliams & Shchepetkin 2001, Open boundary conditions for long-term integration of regional oceanic models, Ocean Modell. | [10.1016/S1463-5003(00)00013-5](https://doi.org/10.1016/S1463-5003(00)00013-5) | `Marchesiello_2001 - Open Boundary Conditions for Long-Term Integration of Regional Oceanic Models.pdf` | ✅ |
| ★ | Martinsen & Engedahl 1987, Implementation and testing of a lateral boundary scheme as an open boundary condition in a barotropic ocean model, Coastal Eng. (flow relaxation) | [10.1016/0378-3839(87)90028-7](https://doi.org/10.1016/0378-3839(87)90028-7) [verificar] | `Martinsen_Engedahl_1987 - Implementation and Testing of a Lateral Boundary Scheme.pdf` | ✅ |
| | Jensen 1998, Open boundary conditions in stratified ocean models, J. Mar. Syst. | [10.1016/S0924-7963(97)00023-7](https://doi.org/10.1016/S0924-7963(97)00023-7) | `Jensen_1998 - Open Boundary Conditions in Stratified Ocean Models.pdf` | ✅ |
| | Palma & Matano 1998, Passive open boundary conditions for a general circulation model: the barotropic mode, JGR | [10.1029/97JC02721](https://doi.org/10.1029/97JC02721) | `Palma_Matano_1998 - On the Implementation of Passive Open Boundary Conditions.pdf` | ✅ |
| | Marsaleix et al. 2009, Open boundary conditions for internal gravity wave modelling using polarization relations, Ocean Modell. (siguiente mejora posible) | [10.1016/j.ocemod.2009.02.010](https://doi.org/10.1016/j.ocemod.2009.02.010) | `Marsaleix_2009b - Open Boundary Conditions for Internal Gravity Wave Modelling Using Polarization Relations.pdf` | ✅ |
| | Siyanbola et al. 2023, Remote internal wave forcing of regional ocean simulations near the U.S. West Coast, Ocean Modell. (reflexión en esponjas hasta 73 %) | [10.1016/j.ocemod.2022.102154](https://doi.org/10.1016/j.ocemod.2022.102154) | `Siyanbola_2023 - Remote Internal Wave Forcing of Regional Ocean Simulations.pdf` | ✅ |
| | Lavelle & Thacker 2008, A pretty good sponge: dealing with open boundaries in limited-area ocean models, Ocean Modell. | [10.1016/j.ocemod.2007.10.002](https://doi.org/10.1016/j.ocemod.2007.10.002) | `Lavelle_Thacker_2008 - A Pretty Good Sponge.pdf` | ✅ |
| ★ | Song & Haidvogel 1994, A semi-implicit ocean circulation model using a generalized topography-following coordinate system, J. Comput. Phys. (función de estiramiento σ; fórmula verificada en Shchepetkin & McWilliams 2003, ec. 4.3) | [10.1006/jcph.1994.1189](https://doi.org/10.1006/jcph.1994.1189) | `Song_Haidvogel_1994 - A Semi-Implicit Ocean Circulation Model Using a Generalized Topography-Following Coordinate System.pdf` | ✅ (escaneado) |
| | Mason et al. 2010, Procedures for offline grid nesting in regional ocean models (ROMS; interpolation of parent data, volume correction), Ocean Modell. | [10.1016/j.ocemod.2010.05.007](https://doi.org/10.1016/j.ocemod.2010.05.007) | `Mason_2010 - Procedures for Offline Grid Nesting in Regional Ocean Models.pdf` | ✅ |
| | Klingbeil et al. 2018, The numerics of hydrostatic structured-grid coastal ocean models: state of the art (incl. OBC and nesting), Ocean Modell. | [10.1016/j.ocemod.2018.01.007](https://doi.org/10.1016/j.ocemod.2018.01.007) | `Klingbeil_2018 - The Numerics of Hydrostatic Structured-Grid Coastal Ocean Models.pdf` | ✅ |
| | Herzfeld et al. 2011, Realistic test cases for limited area ocean modelling (MOM4 OBC suite), Ocean Modell. | [10.1016/j.ocemod.2010.12.008](https://doi.org/10.1016/j.ocemod.2010.12.008) | `Herzfeld_2011 - Realistic Test Cases for Limited Area Ocean Modelling.pdf` | ✅ |
| | Oddo & Pinardi 2008, Lateral open boundary conditions for nested limited area models: a scale selective approach (generalized Flather), Ocean Modell. | [10.1016/j.ocemod.2007.08.001](https://doi.org/10.1016/j.ocemod.2007.08.001) | `Oddo_Pinardi_2008 - Lateral Open Boundary Conditions for Nested Limited Area Models.pdf` | ✅ |
| | Cailleau et al. 2008, Comparison of numerical methods for the open boundary of a Bay of Biscay model (clamped, radiation, nesting, coupling), Ocean Modell. | [10.1016/j.ocemod.2008.05.009](https://doi.org/10.1016/j.ocemod.2008.05.009) | `Cailleau_2008 - Comparison of Different Numerical Methods Used to Handle the Open Boundary.pdf` | ✅ |

## 9. Rendimiento y backends (`docs/design/performance.md`)

Consensus (2026-10-11): una búsqueda; FESOM2-JAX, DJ4Earth y Veros-JAX (Meunier 2025) ya estaban locales.

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| ★ | Häfner, Nuterman & Jochum 2021, Fast, cheap, and turbulent: global ocean modeling with GPU acceleration in Python, JAMES | [10.1029/2021MS002717](https://doi.org/10.1029/2021ms002717) | `Hafner_2021 - Fast Cheap and Turbulent Global Ocean Modeling with GPU Acceleration in Python.pdf` | ✅ |
| ★ | Häfner et al. 2018, Veros v0.1: a fast and versatile ocean simulator in pure Python, GMD | [10.5194/gmd-11-3299-2018](https://doi.org/10.5194/gmd-11-3299-2018) | `Hafner_2018 - Veros v0.1 a Fast and Versatile Ocean Simulator in Pure Python.pdf` | ✅ |
| | Bishnu et al. 2023, Julia CPU vs GPU and Julia-MPI vs Fortran-MPI: MPAS-Ocean, GMD | [10.5194/gmd-16-5539-2023](https://doi.org/10.5194/gmd-16-5539-2023) | `Bishnu_2023 - Comparing the Performance of Julia on CPUs versus GPUs.pdf` | ✅ |
| | Remy et al. 2026, Deep kernel fusion for ocean models on heterogeneous architectures (Poseidon), IJHPCA | [10.1177/10943420261456347](https://doi.org/10.1177/10943420261456347) | `Remy_2026 - Deep Kernel Fusion for Ocean Models on Heterogeneous Architectures.pdf` | ✅ |
| | Briand et al. 2025, Julia versus C++ Kokkos for performance-portable Cartesian CFD solvers, IJHPCA | [10.1177/10943420251341179](https://doi.org/10.1177/10943420251341179) | `Briand_2025 - Julia versus Cpp Kokkos for Performance Portable Cartesian CFD Solvers.pdf` | ✅ |
| | Kang et al. 2023, An implicit barotropic mode solver for MPAS-Ocean (ForTrilinos), IJHPCA | [10.1177/10943420231205601](https://doi.org/10.1177/10943420231205601) | `Kang_2023 - An Implicit Barotropic Mode Solver for MPAS-Ocean.pdf` | ✅ |
| | Lan et al. 2022, High-order multirate explicit time-stepping for the baroclinic–barotropic split, J. Comput. Phys. | [10.1016/j.jcp.2022.111050](https://doi.org/10.1016/j.jcp.2022.111050) | `Lan_2022 - High-Order Multirate Explicit Time-Stepping Schemes for Baroclinic-Barotropic Split Dynamics.pdf` | ✅ |
| | Silvestri et al. 2023, Oceananigans.jl: breakthrough resolution, memory and energy efficiency, arXiv | [10.48550/arxiv.2309.06662](https://doi.org/10.48550/arxiv.2309.06662) | `Silvestri_2023 - Oceananigans Breakthrough Resolution Memory and Energy Efficiency.pdf` | ✅ |
