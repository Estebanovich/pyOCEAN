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

| | Referencia | DOI | Archivo sugerido | Estado |
|---|---|---|---|---|
| | Petersen et al. 2015, ALE vertical coordinate in MPAS-Ocean, Ocean Modell. | [10.1016/j.ocemod.2014.12.004](https://doi.org/10.1016/j.ocemod.2014.12.004) | `Petersen_2015 - Evaluation of the ALE Vertical Coordinate in MPAS-Ocean.pdf` | ✅ |
| | Megann et al. 2022, The z̃ time-filtered ALE coordinate in NEMO, JAMES | [10.1029/2022MS003056](https://doi.org/10.1029/2022MS003056) | `Megann_2022 - Assessment of the z-tilde Coordinate in a Global Eddy-Permitting Model.pdf` | ✅ |
| | Megann 2024, Numerical mixing in a tidally forced global model, Ocean Modell. | [10.1016/j.ocemod.2024.102329](https://doi.org/10.1016/j.ocemod.2024.102329) | `Megann_2024 - Quantifying Numerical Mixing in a Tidally Forced Global Model.pdf` | ✅ |
| | Chang et al. 2026, Vertical coordinates in regional MOM6 (NW Pacific), GMD | [10.5194/gmd-19-187-2026](https://doi.org/10.5194/gmd-19-187-2026) | `Chang_2026 - Assessing Vertical Coordinate Performance in Regional MOM6.pdf` | ✅ |
| | Bruciaferri et al. 2024, Localized general vertical coordinates, JAMES | [10.1029/2023MS003893](https://doi.org/10.1029/2023MS003893) | `Bruciaferri_2024 - Localized General Vertical Coordinates for Quasi-Eulerian Models.pdf` | ✅ |
| | Halliwell 2004, Vertical coordinate and mixing algorithms in HYCOM, Ocean Modell. | [10.1016/j.ocemod.2003.10.002](https://doi.org/10.1016/j.ocemod.2003.10.002) | `Halliwell_2004 - Evaluation of Vertical Coordinate and Vertical Mixing Algorithms in HYCOM.pdf` | ✅ |
| | Klingbeil et al. 2019, Reducing spurious diapycnal mixing in ocean models (book chapter) | [10.1007/978-3-030-05704-6_8](https://doi.org/10.1007/978-3-030-05704-6_8) | `Eden_Iske_2019 - Energy Transfers in Atmosphere and Ocean.pdf` (libro completo; capítulo de Klingbeil et al.) | ✅ |

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
