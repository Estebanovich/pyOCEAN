# References

All literature cited in this manual, merged from the reference lists of each page. Entries marked **[verify]** were
written from memory or design notes and have not yet been checked against the paper; check them before citing in a
publication. Papers found through Consensus are also listed in `docs/literature/reading_list.md`.

## Literature

- Adcroft, A. & Campin, J.-M. (2004). Rescaled height coordinates for accurate representation of free-surface flows in ocean circulation models. *Ocean Modelling* 7, 269–284. doi:10.1016/j.ocemod.2003.09.003  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Vertical coordinates and ALE](theory/vertical-ale.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md)</small>
- Adcroft, A. et al. (2019). The GFDL global ocean and sea ice model OM4.0: model description and simulation features. *JAMES* 11, 3167–3211. doi:10.1029/2019MS001726  
    <small>Cited in: [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Vertical coordinates and ALE](theory/vertical-ale.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md), [Surface forcing and external data](user-guide/surface-forcing.md), [BayIW: bay internal waves and seiches against MITgcm and MOM6](validation/bayiw.md)</small>
- Adcroft, A., Hallberg, R. & Harrison, M. (2008). A finite volume discretization of the pressure gradient force using analytic integration. *Ocean Modell.* 22, 106–113. [10.1016/j.ocemod.2008.02.001](https://doi.org/10.1016/j.ocemod.2008.02.001) **[verify]**  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Limitations and roadmap](roadmap.md), [Pressure gradient](theory/pressure-gradient.md), [Configuration reference](user-guide/configuration-reference.md), [Analytical and exact solutions](validation/analytical.md)</small>
- Adcroft, A. & Hallberg, R. (2006). On methods for solving the oceanic equations of motion in generalized vertical coordinates. *Ocean Modelling* 11, 224–233. doi:10.1016/j.ocemod.2004.12.007  
    <small>Cited in: [Governing equations](theory/equations.md)</small>
- Arakawa, A. & Lamb, V.R. (1981). A potential enstrophy and energy conserving scheme for the shallow water equations. *Mon. Wea. Rev.* 109, 18–36. [verify] (the σ-mode docstring cites "Arakawa & Lamb 1977").  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Momentum equations](theory/momentum.md)</small>
- Arakawa, A. & Lamb, V.R. (1977). Computational design of the basic dynamical processes of the UCLA general circulation model. *Methods in Computational Physics* 17, 173–265.  
    <small>Cited in: [Data structures and conventions](architecture/data-structures.md), [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Grids and bathymetry](theory/grids.md)</small>
- Arbic, B.K., Wallcraft, A.J. & Metzger, E.J. (2010). Concurrent simulation of the eddying general circulation and tides in a global ocean model. *Ocean Modell.* 32, 175–187. doi:10.1016/j.ocemod.2010.01.007 **[verify]**  
    <small>Cited in: [Tides](theory/tides.md)</small>
- Beckmann, A. & Haidvogel, D.B. (1993). Numerical simulation of flow around a tall isolated seamount. *J. Phys. Oceanogr.* 23, 1736–1753. doi:[10.1175/1520-0485(1993)023<1736:NSOFAA>2.0.CO;2](https://doi.org/10.1175/1520-0485(1993)023%3C1736:NSOFAA%3E2.0.CO;2)  
    <small>Cited in: [Comparison with other models](comparison.md), [Grids and bathymetry](theory/grids.md), [Pressure gradient](theory/pressure-gradient.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md), [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md), [Regional setup: grid, bathymetry, initial and boundary data](user-guide/regional-setup.md), [Troubleshooting and FAQ](user-guide/troubleshooting.md), [Analytical and exact solutions](validation/analytical.md), [Idealised benchmarks](validation/idealised.md)</small>
- Benjamin, T. B. (1968). Gravity currents and related phenomena. *J. Fluid Mech.* 31, 209–248. [10.1017/S0022112068000133](https://doi.org/10.1017/S0022112068000133) **[verify]**  
    <small>Cited in: [Quick start: a first complete run](getting-started/quickstart.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md), [Analytical and exact solutions](validation/analytical.md), [Idealised benchmarks](validation/idealised.md), [Validation: strategy and scorecard](validation/index.md)</small>
- Berntsen, J., Xing, J. & Davies, A.M. (2015). Internal pressure gradient errors in σ-coordinate ocean models in high resolution fjord studies. *Ocean Modell.* doi:[10.1016/j.ocemod.2015.05.009](https://doi.org/10.1016/j.ocemod.2015.05.009)  
    <small>Cited in: [Pressure gradient](theory/pressure-gradient.md)</small>
- Bishnu, S. et al. (2023). Comparing the performance of Julia on CPUs versus GPUs and Julia-MPI versus Fortran-MPI: a case study with MPAS-Ocean. *GMD* 16, 5539. doi:10.5194/gmd-16-5539-2023  
    <small>Cited in: [Performance and parallelism](architecture/performance.md)</small>
- Blayo, E. & Debreu, L. (2005). Revisiting open boundary conditions from the point of view of characteristic variables. *Ocean Modell.* 9, 231–252. **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md)</small>
- Bleck, R. (2002). An oceanic general circulation model framed in hybrid isopycnic-Cartesian coordinates. *Ocean Modelling* 4, 55–88. doi:10.1016/S1463-5003(01)00012-9  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Vertical coordinates and ALE](theory/vertical-ale.md), [Component checks: forcing, tides tools, ALE columns, solvers](validation/components.md)</small>
- Bleck, R. & Smith, L.T. (1990). A wind-driven isopycnic coordinate model of the north and equatorial Atlantic Ocean. *J. Geophys. Res.* 95, 3273–3285. [verify]  
    <small>Cited in: [Momentum equations](theory/momentum.md), [Idealised benchmarks](validation/idealised.md)</small>
- Borden, Z. & Meiburg, E. (2013). Circulation based models for Boussinesq gravity currents. *Phys. Fluids* 25. [verify]  
    <small>Cited in: [Momentum equations](theory/momentum.md)</small>
- Borges, R., Carmona, M., Costa, B. & Don, W.S. (2008). An improved weighted essentially non-oscillatory scheme for hyperbolic conservation laws. *J. Comput. Phys.* 227, 3191–3211. doi:[10.1016/j.jcp.2007.11.038](https://doi.org/10.1016/j.jcp.2007.11.038)  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Tracer transport and spurious mixing](theory/tracers.md)</small>
- Bowden, K.F. (1978). Physical problems of the benthic boundary layer. *Geophys. Surv.* 3, 255–296. **[verify]**  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- Briand, L. et al. (2025). Julia vs C++ Kokkos for Cartesian CFD. *IJHPCA*. doi:10.1177/10943420251341179  
    <small>Cited in: [Performance and parallelism](architecture/performance.md)</small>
- Buck, A.L. (1981). New equations for computing vapor pressure and enhancement factor. *J. Appl. Meteor.* 20, 1527–1532. **[verify]**  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Burchard, H., Petersen, O. & Rippeth, T.P. (1998). Comparing the performance of the Mellor–Yamada and the κ–ε two-equation turbulence models. *J. Geophys. Res.* 103(C5), 10543–10554. **[verify]**  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- Burchard, H. & Rennau, H. (2008). Comparative quantification of physically and numerically induced mixing in ocean models. *Ocean Modell.* 20, 293–311. doi:[10.1016/j.ocemod.2007.10.003](https://doi.org/10.1016/j.ocemod.2007.10.003)  
    <small>Cited in: [Data structures and conventions](architecture/data-structures.md), [The time step, flow diagrams](architecture/time-step.md), [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Limitations and roadmap](roadmap.md), [Governing equations](theory/equations.md), [Tracer transport and spurious mixing](theory/tracers.md), [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md), [Idealised benchmarks](validation/idealised.md)</small>
- Campin, J.-M., Adcroft, A., Hill, C. & Marshall, J. (2004). Conservation of properties in a free-surface model. *Ocean Modell.* 6, 221–244.  
    <small>Cited in: [Governing equations](theory/equations.md), [Free surface and time stepping](theory/free-surface-time-stepping.md)</small>
- Canuto, V.M., Howard, A., Cheng, Y. & Dubovikov, M.S. (2001). Ocean turbulence. Part I: One-point closure model — momentum and heat vertical diffusivities. *J. Phys. Oceanogr.* 31, 1413–1426. [doi:10.1175/1520-0485(2001)031<1413:OTPIOP>2.0.CO;2](https://doi.org/10.1175/1520-0485(2001)031<1413:OTPIOP>2.0.CO;2) **[verify DOI]**  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- Carter, G.S. & Merrifield, M.A. (2007). Open boundary conditions for regional tidal simulations. *Ocean Modell.* 18, 194–209. [doi:10.1016/j.ocemod.2007.04.003](https://doi.org/10.1016/j.ocemod.2007.04.003) **[verify DOI]**  
    <small>Cited in: [Open boundaries and nesting](theory/open-boundaries.md)</small>
- Cartwright, D.E. (1982). Tidal analysis — a retrospect. In: *Time Series Methods in Hydrosciences*, Elsevier, 170–188.  
    <small>Cited in: [Tides](theory/tides.md)</small>
- Casulli, V. (2009). A high-resolution wetting and drying algorithm for free-surface hydrodynamics. *Int. J. Numer. Meth. Fluids* 60, 391–408. doi:[10.1002/fld.1896](https://doi.org/10.1002/fld.1896)  
    <small>Cited in: [Limitations and roadmap](roadmap.md), [Governing equations](theory/equations.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Wetting and drying](theory/wetting-drying.md), [Wetting and drying: Thacker and TIDAL_FLAT](validation/wetting-drying.md)</small>
- Casulli, V. (1999). A semi-implicit finite difference method for non-hydrostatic, free-surface flows. *Int. J. Numer. Meth. Fluids* 30, 425–440.  
    <small>Cited in: [Wetting and drying](theory/wetting-drying.md)</small>
- Casulli, V. & Cheng, R.T. (1992). Semi-implicit finite difference methods for three-dimensional shallow water flow. *Int. J. Numer. Meth. Fluids* 15, 629–648. [doi:10.1002/fld.1650150602](https://doi.org/10.1002/fld.1650150602) **[verify DOI]**  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Model overview in one page](getting-started/overview.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Turbulence closures and bottom friction](theory/turbulence-friction.md), [Wetting and drying](theory/wetting-drying.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md), [Wetting and drying: Thacker and TIDAL_FLAT](validation/wetting-drying.md)</small>
- Casulli, V. (1990). Semi-implicit finite difference methods for the two-dimensional shallow water equations. *J. Comput. Phys.* 86, 56–74. [doi:10.1016/0021-9991(90)90091-E](https://doi.org/10.1016/0021-9991(90)90091-E) **[verify DOI]**  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Wetting and drying](theory/wetting-drying.md), [Configuration reference](user-guide/configuration-reference.md), [Troubleshooting and FAQ](user-guide/troubleshooting.md)</small>
- Chapman, D.C. (1985). Numerical treatment of cross-shelf open boundaries in a barotropic coastal ocean model. *J. Phys. Oceanogr.* 15, 1060–1075. [doi:10.1175/1520-0485(1985)015<1060:NTOCSO>2.0.CO;2](https://doi.org/10.1175/1520-0485(1985)015<1060:NTOCSO>2.0.CO;2) **[verify DOI]**  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Open boundaries and nesting](theory/open-boundaries.md), [Idealised benchmarks](validation/idealised.md)</small>
- Cheng, Y. & Brutsaert, W. (2005). Flux-profile relationships for wind speed and temperature in the stable atmospheric boundary layer. *Boundary-Layer Meteorol.* 114, 519–538. doi:10.1007/s10546-004-1425-4 **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md)</small>
- Colella, P. & Woodward, P.R. (1984). The piecewise parabolic method (PPM) for gas-dynamical simulations. *J. Comput. Phys.* 54, 174–201.  
    <small>Cited in: [Tracer transport and spurious mixing](theory/tracers.md), [Vertical coordinates and ALE](theory/vertical-ale.md)</small>
- Davies, A.M. & Jones, J.E. (1988). Modelling turbulence in shallow sea regions. **[verify]**  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- Dukowicz, J.K. & Smith, R.D. (1994). Implicit free-surface method for the Bryan–Cox–Semtner ocean model. *J. Geophys. Res.* 99(C4), 7991–8014.  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Free surface and time stepping](theory/free-surface-time-stepping.md)</small>
- Durran, D.R. (2010). *Numerical Methods for Fluid Dynamics*, 2nd ed. Springer.  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Model overview in one page](getting-started/overview.md), [Free surface and time stepping](theory/free-surface-time-stepping.md)</small>
- ECMWF (2016). IFS Documentation Cy41r2, Part IV: Physical Processes, §7.2.1.  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Egbert, G.D. & Erofeeva, S.Y. (2002). Efficient inverse modeling of barotropic ocean tides. *J. Atmos. Oceanic Technol.* 19, 183–204. doi:10.1175/1520-0426(2002)019<0183:EIMOBO>2.0.CO;2 **[verify]**  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Tides](theory/tides.md), [Component checks: forcing, tides tools, ALE columns, solvers](validation/components.md), [Tides and sea level](validation/tides-sealevel.md)</small>
- Engwirda, D., Kelley, M. & Marshall, J. (2017). High-order accurate finite-volume formulations for the pressure gradient force in layered ocean models. *Ocean Modell.* [verify]  
    <small>Cited in: [Pressure gradient](theory/pressure-gradient.md)</small>
- Fairall, C.W., Bradley, E.F., Hare, J.E., Grachev, A.A. & Edson, J.B. (2003). Bulk parameterization of air–sea fluxes: updates and verification for the COARE algorithm. *J. Climate* 16, 571–591. doi:10.1175/1520-0442(2003)016<0571:BPOASF>2.0.CO;2 **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md), [Component checks: forcing, tides tools, ALE columns, solvers](validation/components.md)</small>
- Flather, R.A. (1976). A tidal model of the north-west European continental shelf. *Mém. Soc. R. Sci. Liège* 6(10), 141–164.  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Open boundaries and nesting](theory/open-boundaries.md), [Tides](theory/tides.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md)</small>
- Foreman, M.G.G., Henry, R.F., Walters, R.A. & Ballantyne, V.A. (1993). A finite element model for tides and resonance along the north coast of British Columbia. *J. Geophys. Res.* 98(C2), 2509–2531. doi:10.1029/92JC02470 **[verify]**  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Tides](theory/tides.md), [Tides and sea level](validation/tides-sealevel.md)</small>
- Foreman, M.G.G. (1977). Manual for tidal heights analysis and prediction. Pacific Marine Science Report 77-10.  
    <small>Cited in: [Tides](theory/tides.md)</small>
- Galperin, B., Kantha, L.H., Hassid, S. & Rosati, A. (1988). A quasi-equilibrium turbulent energy model for geophysical flows. *J. Atmos. Sci.* 45, 55–62. **[verify]**  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- GEBCO Compilation Group (2023). GEBCO 2023 Grid. doi:10.5285/f98b053b-0cbc-6c23-d9da-7de7d696a50d  
    <small>Cited in: [Grids and bathymetry](theory/grids.md), [Regional setup: grid, bathymetry, initial and boundary data](user-guide/regional-setup.md)</small>
- Gill, A. E. (1982). *Atmosphere–Ocean Dynamics*. Academic Press.  
    <small>Cited in: [Analytical and exact solutions](validation/analytical.md)</small>
- Griffies, S. M. et al. (2020). A primer on the vertical Lagrangian-remap method in ocean models based on finite volume generalized vertical coordinates. *JAMES* 12, e2019MS001954. doi:10.1029/2019MS001954  
    <small>Cited in: [Governing equations](theory/equations.md), [Vertical coordinates and ALE](theory/vertical-ale.md)</small>
- Haigh, I.D. et al. (2023). GESLA version 3. *Geosci. Data J.* 10, 293–314. doi:10.1002/gdj3.174 **[verify]**  
    <small>Cited in: [Tides and the 2-D tidal model](user-guide/tidal-model.md), [Tides and sea level](validation/tides-sealevel.md)</small>
- Hallberg, R. & Adcroft, A. (2009). Reconciling estimates of the free surface height in Lagrangian vertical coordinate ocean models with mode-split time stepping. *Ocean Modelling* 29, 15–26. doi:10.1016/j.ocemod.2009.02.008  
    <small>Cited in: [Vertical coordinates and ALE](theory/vertical-ale.md), [Idealised benchmarks](validation/idealised.md)</small>
- Hallberg, R. & Rhines, P. (1996). Buoyancy-driven circulation in an ocean basin with isopycnals intersecting the sloping boundary. *J. Phys. Oceanogr.* 26, 913–940. **[verify]**  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Momentum equations](theory/momentum.md), [Vertical coordinates and ALE](theory/vertical-ale.md), [Troubleshooting and FAQ](user-guide/troubleshooting.md), [Idealised benchmarks](validation/idealised.md)</small>
- Haney, R.L. (1991). On the pressure gradient force over steep topography in sigma coordinate ocean models. *J. Phys. Oceanogr.* 21, 610–619. doi:[10.1175/1520-0485(1991)021<0610:OTPGFO>2.0.CO;2](https://doi.org/10.1175/1520-0485(1991)021%3C0610:OTPGFO%3E2.0.CO;2) [verify]  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Grids and bathymetry](theory/grids.md), [Pressure gradient](theory/pressure-gradient.md), [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md), [Regional setup: grid, bathymetry, initial and boundary data](user-guide/regional-setup.md), [Troubleshooting and FAQ](user-guide/troubleshooting.md), [BayIW: bay internal waves and seiches against MITgcm and MOM6](validation/bayiw.md)</small>
- Hart-Davis, M. G. et al. (2022). TICON-3: tidal constants based on GESLA-3 sea-level records. PANGAEA, [10.1594/PANGAEA.951610](https://doi.org/10.1594/PANGAEA.951610)  
    <small>Cited in: [Tides and the 2-D tidal model](user-guide/tidal-model.md), [Tides and sea level](validation/tides-sealevel.md)</small>
- Hersbach, H. et al. (2020). The ERA5 global reanalysis. *Q. J. R. Meteorol. Soc.* 146, 1999–2049. doi:10.1002/qj.3803 **[verify]**  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Häfner, D., Nuterman, R. & Jochum, M. (2021). Fast, cheap, and turbulent: global ocean modeling with GPU acceleration in Python. *JAMES*. [10.1029/2021MS002717](https://doi.org/10.1029/2021ms002717)  
    <small>Cited in: [Performance and parallelism](architecture/performance.md), [Comparison with other models](comparison.md), [Limitations and roadmap](roadmap.md)</small>
- Higdon, R.L. (2005). A two-level time-stepping method for layered ocean circulation models: further development and testing. *J. Comput. Phys.* 206. doi:[10.1016/j.jcp.2004.12.011](https://doi.org/10.1016/j.jcp.2004.12.011)  
    <small>Cited in: [Momentum equations](theory/momentum.md), [Vertical coordinates and ALE](theory/vertical-ale.md), [Idealised benchmarks](validation/idealised.md)</small>
- Hofmeister, R., Burchard, H. & Beckers, J.-M. (2010). Non-uniform adaptive vertical grids for 3D numerical ocean models. *Ocean Modell.* 33, 70–86. **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md)</small>
- Ilicak, M., Adcroft, A.J., Griffies, S.M. & Hallberg, R.W. (2012). Spurious dianeutral mixing and the role of momentum closure. *Ocean Modell.* 45–46, 37–58. doi:[10.1016/j.ocemod.2011.10.003](https://doi.org/10.1016/j.ocemod.2011.10.003)  
    <small>Cited in: [Comparison with other models](comparison.md), [Quick start: a first complete run](getting-started/quickstart.md), [Limitations and roadmap](roadmap.md), [Governing equations](theory/equations.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Momentum equations](theory/momentum.md), [Tracer transport and spurious mixing](theory/tracers.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md), [Configuration reference](user-guide/configuration-reference.md), [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md), [Troubleshooting and FAQ](user-guide/troubleshooting.md), [Idealised benchmarks](validation/idealised.md), [Validation: strategy and scorecard](validation/index.md)</small>
- IOC, SCOR & IAPSO (2010). The international thermodynamic equation of seawater – 2010 (TEOS-10). Manuals and Guides 56, UNESCO.  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Jayne, S.R. & St. Laurent, L.C. (2001). Parameterizing tidal dissipation over rough topography. *Geophys. Res. Lett.* 28, 811–814. doi:10.1029/2000GL012044 **[verify]**  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Tides](theory/tides.md)</small>
- Jensen, T.G. (1998). Open boundary conditions in stratified ocean models. *J. Mar. Syst.* 16, 297–322. **[verify]**  
    <small>Cited in: [Open boundaries and nesting](theory/open-boundaries.md), [Idealised benchmarks](validation/idealised.md)</small>
- Jiang, G.-S. & Shu, C.-W. (1996). Efficient implementation of weighted ENO schemes. *J. Comput. Phys.* 126, 202–228. doi:[10.1006/jcph.1996.0130](https://doi.org/10.1006/jcph.1996.0130)  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Model overview in one page](getting-started/overview.md), [Tracer transport and spurious mixing](theory/tracers.md)</small>
- Kara, A.B., Hurlburt, H.E. & Wallcraft, A.J. (2005). Stability-dependent exchange coefficients for air–sea fluxes. *J. Atmos. Oceanic Technol.* 22, 1080–1094. doi:10.1175/JTECH1747.1 **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md), [Component checks: forcing, tides tools, ALE columns, solvers](validation/components.md)</small>
- Klingbeil, K., Mohammadi-Aragh, M., Gräwe, U. & Burchard, H. (2014). Quantification of spurious dissipation and mixing – discrete variance decay in a finite-volume framework. *Ocean Modell.* 81, 49–64. doi:[10.1016/j.ocemod.2014.06.001](https://doi.org/10.1016/j.ocemod.2014.06.001)  
    <small>Cited in: [Tracer transport and spurious mixing](theory/tracers.md), [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md)</small>
- Koldunov, N. et al. (2026). FESOM2-JAX v1.0. arXiv. [10.48550/arxiv.2608.01546](https://doi.org/10.48550/arxiv.2608.01546)  
    <small>Cited in: [Performance and parallelism](architecture/performance.md), [Limitations and roadmap](roadmap.md)</small>
- Large, W.G., McWilliams, J.C. & Doney, S.C. (1994). Oceanic vertical mixing: a review and a model with a nonlocal boundary layer parameterization. *Rev. Geophys.* 32(4), 363–403. [doi:10.1029/94RG01872](https://doi.org/10.1029/94RG01872)  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- Large, W.G. & Yeager, S.G. (2004). Diurnal to decadal global forcing for ocean and sea-ice models: the data sets and flux climatologies. NCAR Tech. Note NCAR/TN-460+STR. doi:10.5065/D6KK98Q6 **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md), [Component checks: forcing, tides tools, ALE columns, solvers](validation/components.md)</small>
- Large, W.G. & Yeager, S.G. (2009). The global climatology of an interannually varying air–sea flux data set. *Clim. Dyn.* 33, 341–364. doi:10.1007/s00382-008-0441-3 **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md), [Component checks: forcing, tides tools, ALE columns, solvers](validation/components.md)</small>
- Lavelle, J. W. & Thacker, W. C. (2008). A pretty good sponge: dealing with open boundaries in limited-area ocean models. *Ocean Modell.* 20, 270–292. **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md), [Open boundaries and nesting](theory/open-boundaries.md)</small>
- Lemarié, F., Debreu, L., Madec, G., Demange, J., Molines, J.-M. & Honnorat, M. (2015). Stability constraints for oceanic numerical models: implications for the formulation of time and space discretizations. *Ocean Modell.* 92, 124–148. **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md)</small>
- Leonard, B.P. (1979). A stable and accurate convective modelling procedure based on quadratic upstream interpolation. *Comput. Methods Appl. Mech. Eng.* 19, 59–98.  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Momentum equations](theory/momentum.md), [Tracer transport and spurious mixing](theory/tracers.md)</small>
- Lin, S.-J. (1994). A "consistent" method for the slope-limited piecewise linear advection scheme. *Mon. Wea. Rev.* (appendix B) **[verify]**  
    <small>Cited in: [Vertical coordinates and ALE](theory/vertical-ale.md)</small>
- Lowe, P.R. (1977). An approximating polynomial for the computation of saturation vapor pressure. *J. Appl. Meteor.* 16, 100–103. **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md)</small>
- Marchesiello, P., McWilliams, J.C. & Shchepetkin, A. (2001). Open boundary conditions for long-term integration of regional oceanic models. *Ocean Modell.* 3, 1–20. [doi:10.1016/S1463-5003(00)00013-5](https://doi.org/10.1016/S1463-5003(00)00013-5) **[verify DOI]**  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Open boundaries and nesting](theory/open-boundaries.md)</small>
- Marsaleix, P., Auclair, F. & Estournel, C. (2006). Considerations on open boundary conditions for regional and coastal ocean models. *J. Atmos. Oceanic Technol.* 23, 1604–1613. doi:[10.1175/jtech1930.1](https://doi.org/10.1175/jtech1930.1)  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Open boundaries and nesting](theory/open-boundaries.md), [Tracer transport and spurious mixing](theory/tracers.md)</small>
- Marsaleix, P., Auclair, F. & Estournel, C. (2009). Low-order pressure gradient schemes in sigma coordinate models: the seamount test revisited. *Ocean Modell.* 30, 169–177. doi:[10.1016/j.ocemod.2009.06.011](https://doi.org/10.1016/j.ocemod.2009.06.011)  
    <small>Cited in: [Pressure gradient](theory/pressure-gradient.md), [Idealised benchmarks](validation/idealised.md)</small>
- Marshall, J., Adcroft, A., Hill, C., Perelman, L. & Heisey, C. (1997). A finite-volume, incompressible Navier–Stokes model for studies of the ocean on parallel computers. *J. Geophys. Res.* 102(C3), 5753–5766.  
    <small>Cited in: [Data structures and conventions](architecture/data-structures.md), [The time step, flow diagrams](architecture/time-step.md), [Governing equations](theory/equations.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Grids and bathymetry](theory/grids.md), [Pressure gradient](theory/pressure-gradient.md)</small>
- Martinsen, E.A. & Engedahl, H. (1987). Implementation and testing of a lateral boundary scheme as an open boundary condition in a barotropic ocean model. *Coastal Eng.* 11, 603–627.  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Open boundaries and nesting](theory/open-boundaries.md)</small>
- Mellor, G.L. & Blumberg, A.F. (1985). Modeling vertical and horizontal diffusivities with the sigma coordinate system. *Mon. Wea. Rev.* 113, 1379–1383. [verify]  
    <small>Cited in: [Tracer transport and spurious mixing](theory/tracers.md)</small>
- Mellor, G. L., Ezer, T., Oey, L.-Y. (1994). The pressure gradient conundrum of sigma coordinate ocean models. *J. Atmos. Oceanic Technol.* 11. **[verify]**  
    <small>Cited in: [BayIW: bay internal waves and seiches against MITgcm and MOM6](validation/bayiw.md)</small>
- Metzger, E.J. et al. (2014). US Navy operational global ocean and Arctic ice prediction systems. *Oceanography* 27(3), 32–43.  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Orlanski, I. (1976). A simple boundary condition for unbounded hyperbolic flows. *J. Comput. Phys.* 21, 251–269.  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Model overview in one page](getting-started/overview.md), [Open boundaries and nesting](theory/open-boundaries.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md)</small>
- Palma, E.D. & Matano, R.P. (1998). On the implementation of passive open boundary conditions for a general circulation model: the barotropic mode. *J. Geophys. Res.* 103(C1), 1319–1341. **[verify]**  
    <small>Cited in: [Open boundaries and nesting](theory/open-boundaries.md), [Idealised benchmarks](validation/idealised.md)</small>
- Patankar, S.V. (1980). *Numerical Heat Transfer and Fluid Flow.* Hemisphere. **[verify]**  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- Paulson, C.A. & Simpson, J.J. (1977). Irradiance measurements in the upper ocean. *J. Phys. Oceanogr.* 7, 952–956. doi:10.1175/1520-0485(1977)007<0952:IMITUO>2.0.CO;2 **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md)</small>
- Paulson, C.A. (1970). The mathematical representation of wind speed and temperature profiles in the unstable atmospheric surface layer. *J. Appl. Meteor.* 9, 857–861. **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md)</small>
- Pawlowicz, R., Beardsley, B. & Lentz, S. (2002). Classical tidal harmonic analysis including error estimates in MATLAB using T_TIDE. *Comput. Geosci.* 28, 929–937. doi:10.1016/S0098-3004(02)00013-4 **[verify]**  
    <small>Cited in: [Tides](theory/tides.md)</small>
- Petersen, M. R. et al. (2015). Evaluation of the ALE vertical coordinate method in MPAS-Ocean. *Ocean Modell.* 86, 93–113. [10.1016/j.ocemod.2014.12.004](https://doi.org/10.1016/j.ocemod.2014.12.004)  
    <small>Cited in: [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md), [Idealised benchmarks](validation/idealised.md)</small>
- Press, W.H. et al. (2007). *Numerical Recipes*, 3rd ed., §3.6.  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Pérez, B. et al. (2013). Sea level data quality control with SELENE. ESEAS-RI / Puertos del Estado.  
    <small>Cited in: [Tides and the 2-D tidal model](user-guide/tidal-model.md)</small>
- Ray, R. D. (1998). Ocean self-attraction and loading in numerical tidal models. *Mar. Geod.* 21. **[verify]**  
    <small>Cited in: [Tides and sea level](validation/tides-sealevel.md)</small>
- Rodi, W. (1993). *Turbulence Models and Their Application in Hydraulics.* IAHR.  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- Sadourny, R. (1975). The dynamics of finite-difference models of the shallow-water equations. *J. Atmos. Sci.* 32, 680–689.  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Momentum equations](theory/momentum.md)</small>
- Schlichting, D. et al. (2023). *JAMES*. doi:[10.1029/2022MS003380](https://doi.org/10.1029/2022MS003380) [verify]  
    <small>Cited in: [Tracer transport and spurious mixing](theory/tracers.md), [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md)</small>
- Schureman, P. (1958). *Manual of Harmonic Analysis and Prediction of Tides.* US Coast and Geodetic Survey, Special Publication 98.  
    <small>Cited in: [Tides](theory/tides.md)</small>
- Shapiro, R. (1970). Smoothing, filtering, and boundary effects. *Rev. Geophys. Space Phys.* 8(2), 359–387.  
    <small>Cited in: [Grids and bathymetry](theory/grids.md), [Regional setup: grid, bathymetry, initial and boundary data](user-guide/regional-setup.md)</small>
- Shchepetkin, A.F. & McWilliams, J.C. (2003). A method for computing horizontal pressure-gradient force in an oceanic model with a nonaligned vertical coordinate. *J. Geophys. Res.* 108(C3). doi:[10.1029/2001JC001047](https://doi.org/10.1029/2001JC001047)  
    <small>Cited in: [Comparison with other models](comparison.md), [Grids and bathymetry](theory/grids.md), [Pressure gradient](theory/pressure-gradient.md), [Regional setup: grid, bathymetry, initial and boundary data](user-guide/regional-setup.md)</small>
- Shchepetkin, A.F. & McWilliams, J.C. (2005). The Regional Oceanic Modeling System (ROMS): a split-explicit, free-surface, topography-following-coordinate oceanic model. *Ocean Modell.* 9, 347–404. doi:[10.1016/j.ocemod.2004.08.002](https://doi.org/10.1016/j.ocemod.2004.08.002)  
    <small>Cited in: [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Momentum equations](theory/momentum.md)</small>
- Shu, C.-W. & Osher, S. (1988). Efficient implementation of essentially non-oscillatory shock-capturing schemes. *J. Comput. Phys.* 77, 439–471.  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Model overview in one page](getting-started/overview.md), [Free surface and time stepping](theory/free-surface-time-stepping.md), [Tracer transport and spurious mixing](theory/tracers.md)</small>
- Silvestri, S. et al. (2024). A new WENO-based momentum advection scheme for simulations of ocean mesoscale turbulence. *JAMES*. **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md)</small>
- Siyanbola, O. Q. et al. (2023). Remote internal wave forcing of regional ocean simulations near the U.S. West Coast. *Ocean Modell.* **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md), [Open boundaries and nesting](theory/open-boundaries.md)</small>
- Skamarock, W.C. et al. (2019). A Description of the Advanced Research WRF Model Version 4. NCAR/TN-556+STR.  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Song, Y. & Haidvogel, D. (1994). A semi-implicit ocean circulation model using a generalized topography-following coordinate system. *J. Comput. Phys.* 115, 228–244. doi:10.1006/jcph.1994.1189  
    <small>Cited in: [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Grids and bathymetry](theory/grids.md), [Building a model with `SigmaBaroclinic`](user-guide/building-a-model.md), [Configuration reference](user-guide/configuration-reference.md), [Regional setup: grid, bathymetry, initial and boundary data](user-guide/regional-setup.md)</small>
- Song, Y.T. (1998). A general pressure gradient formulation for ocean models. Part I: Scheme design and diagnostic analysis. *Mon. Wea. Rev.* 126, 3213–3230. doi:[10.1175/1520-0493(1998)126<3213:AGPGFF>2.0.CO;2](https://doi.org/10.1175/1520-0493(1998)126%3C3213:AGPGFF%3E2.0.CO;2)  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Pressure gradient](theory/pressure-gradient.md), [Configuration reference](user-guide/configuration-reference.md), [Analytical and exact solutions](validation/analytical.md)</small>
- Soulsby, R.L. (1997). *Dynamics of Marine Sands.* Thomas Telford, London, §3.  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Turbulence closures and bottom friction](theory/turbulence-friction.md), [Tides and the 2-D tidal model](user-guide/tidal-model.md), [Wetting and drying: Thacker and TIDAL_FLAT](validation/wetting-drying.md)</small>
- Stammer, D. et al. (2014). Accuracy assessment of global barotropic ocean tide models. *Rev. Geophys.* 52, 243–282. doi:10.1002/2014RG000450 **[verify]**  
    <small>Cited in: [Tides and the 2-D tidal model](user-guide/tidal-model.md)</small>
- Stelling, G. S. & Duinmeijer, S. P. A. (2003). A staggered conservative scheme for every Froude number in rapidly varied shallow water flows. *Int. J. Numer. Meth. Fluids* 43, 1329–1354. **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md), [Wetting and drying](theory/wetting-drying.md), [Troubleshooting and FAQ](user-guide/troubleshooting.md), [Wetting and drying: Thacker and TIDAL_FLAT](validation/wetting-drying.md)</small>
- Sud, Y.C. & Walker, G.K. (1997), as cited in HYCOM `thermf.F90` (salinity factor of the vapour pressure). **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md)</small>
- Sutterley, T.C. et al. pyTMD: Python-based tidal prediction software (test oracle only). **[verify]**  
    <small>Cited in: [Tides](theory/tides.md)</small>
- Thacker, W.C. (1981). Some exact solutions to the nonlinear shallow-water wave equations. *J. Fluid Mech.* 107, 499–508.  
    <small>Cited in: [Wetting and drying](theory/wetting-drying.md), [Troubleshooting and FAQ](user-guide/troubleshooting.md), [Analytical and exact solutions](validation/analytical.md), [Validation: strategy and scorecard](validation/index.md), [Wetting and drying: Thacker and TIDAL_FLAT](validation/wetting-drying.md)</small>
- Thakur, R., Arbic, B. K., Menemenlis, D. et al. (2022). Impact of vertical mixing parameterizations on internal gravity wave spectra in regional ocean models. *Geophys. Res. Lett.* **[verify]**  
    <small>Cited in: [Limitations and roadmap](roadmap.md)</small>
- Umlauf, L. & Burchard, H. (2003). A generic length-scale equation for geophysical turbulence models. *J. Mar. Res.* 61, 235–265. [doi:10.1357/002224003322005087](https://doi.org/10.1357/002224003322005087) **[verify DOI]**  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- UNESCO/IOC (2020). Quality control of in situ sea level observations. IOC Manuals and Guides 83.  
    <small>Cited in: [Tides and the 2-D tidal model](user-guide/tidal-model.md)</small>
- Ungarish, M. et al. (2018). [verify] (cited in `_flux_form_u_numpy` for front speeds).  
    <small>Cited in: [Momentum equations](theory/momentum.md)</small>
- Vallis, G. K. (2006). *Atmospheric and Oceanic Fluid Dynamics.* Cambridge University Press.  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Grids and bathymetry](theory/grids.md), [Momentum equations](theory/momentum.md), [Regional setup: grid, bathymetry, initial and boundary data](user-guide/regional-setup.md), [Tides and the 2-D tidal model](user-guide/tidal-model.md)</small>
- Wallace, J.M. & Hobbs, P.V. (2006). *Atmospheric Science*, 2nd ed., Academic Press, §3.1.  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md)</small>
- Wallcraft, A.J., Metzger, E.J. & Carroll, S.N. (2009). Software Design Description for HYCOM Version 2.2. NRL/MR/7320--09-9166.  
    <small>Cited in: [Surface forcing and external data](user-guide/surface-forcing.md)</small>
- Warner, J.C., Sherwood, C.R., Arango, H.G. & Signell, R.P. (2005). Performance of four turbulence closure models implemented using a generic length scale method. *Ocean Modell.* 8, 81–113. [doi:10.1016/j.ocemod.2003.12.003](https://doi.org/10.1016/j.ocemod.2003.12.003) **[verify DOI]**  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md)</small>
- White, L. & Adcroft, A. (2008). A high-order finite volume remapping scheme for nonuniform grids: the piecewise quartic method (PQM). *J. Comput. Phys.* 227, 7394–7422. doi:[10.1016/j.jcp.2008.04.026](https://doi.org/10.1016/j.jcp.2008.04.026)  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Limitations and roadmap](roadmap.md), [Pressure gradient](theory/pressure-gradient.md), [Vertical coordinates and ALE](theory/vertical-ale.md)</small>
- White, L., Adcroft, A. & Hallberg, R. (2009). High-order regridding–remapping schemes for continuous isopycnal and generalized coordinates in ocean models. *J. Comput. Phys.* 228, 8665–8692. doi:10.1016/j.jcp.2009.08.016  
    <small>Cited in: [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Vertical coordinates and ALE](theory/vertical-ale.md), [Configuration reference](user-guide/configuration-reference.md)</small>
- Winters, K.B., Lombard, P.N., Riley, J.J. & D'Asaro, E.A. (1995). Available potential energy and mixing in density-stratified fluids. *J. Fluid Mech.* 289, 115–128. doi:[10.1017/S002211209500125X](https://doi.org/10.1017/S002211209500125X)  
    <small>Cited in: [Model overview in one page](getting-started/overview.md), [Quick start: a first complete run](getting-started/quickstart.md), [Governing equations](theory/equations.md), [Tracer transport and spurious mixing](theory/tracers.md), [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md), [Idealised benchmarks](validation/idealised.md)</small>
- Yagüe, C., Viana, S., Maqueda, G. & Redondo, J.M. (2006). Influence of stability on the flux-profile relationships for wind speed, φm, and temperature, φh, for the stable atmospheric boundary layer. *Nonlin. Processes Geophys.* 13, 185–203. **[verify]**  
    <small>Cited in: [Air–sea fluxes](theory/air-sea-fluxes.md)</small>
- Zalesak, S.T. (1979). Fully multidimensional flux-corrected transport algorithms for fluids. *J. Comput. Phys.* 31, 335–362. doi:[10.1016/0021-9991(79)90051-2](https://doi.org/10.1016/0021-9991(79)90051-2)  
    <small>Cited in: [The time step, flow diagrams](architecture/time-step.md), [Comparison with other models](comparison.md), [Model overview in one page](getting-started/overview.md), [Governing equations](theory/equations.md), [Tracer transport and spurious mixing](theory/tracers.md), [Configuration reference](user-guide/configuration-reference.md), [Idealised benchmarks](validation/idealised.md)</small>

## Reference model source code

Read alongside the papers; cited as `file:line` in the pyOCEAN docstrings. Local copies are listed in `CLAUDE.md`.

- CROCO test cases `THACKER`, `TIDAL_FLAT` (`~/pyOCEAN_refs/croco/`, runs in `~/pyOCEAN_refs/croco_runs/`).  
    <small>Cited in: [Turbulence closures and bottom friction](theory/turbulence-friction.md), [Wetting and drying](theory/wetting-drying.md)</small>
- HYCOM source (`~/pyOCEAN/HYCOM-src`): `cnuity.F90`, `hybgen.F90`, `mod_momtum.F90`.  
    <small>Cited in: [Momentum equations](theory/momentum.md), [Vertical coordinates and ALE](theory/vertical-ale.md)</small>
- HYCOM-tools `topo/src/topo_smooth_skip.f` and `topo_zthin.f` (smoothing with a land buffer, thin-cell removal).  
    <small>Cited in: [Grids and bathymetry](theory/grids.md)</small>
- MITgcm `pkg/obcs` (`obcs_sponge.F`, `obcs_calc.F`), CROCO `OCEAN/u3dbc.F`, MOM6 `MOM_open_boundary.F90`, `MOM_continuity_PPM.F90` (reference implementations).  
    <small>Cited in: [Open boundaries and nesting](theory/open-boundaries.md)</small>
- MITgcm manual (2023), §2.10, partial-step topography (hFac).  
    <small>Cited in: [Grids and bathymetry](theory/grids.md)</small>
- MOM6 source, commit 0e9a561 (`~/pyOCEAN_refs/MOM6/src`): `ALE/MOM_remapping.F90`, `ALE/regrid_edge_values.F90`, `ALE/PLM_functions.F90`, `ALE/PPM_functions.F90`, `ALE/P1M_functions.F90`, `ALE/regrid_interp.F90`, `ALE/coord_zlike.F90`, `ALE/coord_sigma.F90`, `ALE/coord_hycom.F90`, `ALE/MOM_hybgen_regrid.F90`, `core/MOM_continuity_PPM.F90`, `parameterizations/vertical/MOM_vert_friction.F90`, `initialization/MOM_state_initialization.F90`.  
    <small>Cited in: [Momentum equations](theory/momentum.md), [Pressure gradient](theory/pressure-gradient.md), [Vertical coordinates and ALE](theory/vertical-ale.md)</small>
