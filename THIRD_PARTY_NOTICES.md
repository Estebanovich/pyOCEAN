# Third-party notices

pyOCEAN is distributed under the MIT License (`LICENSE`), with one exception:
`forcing/bulk_fms.py` is distributed under the LGPL-3.0-or-later (section 2).

Every scheme in pyOCEAN was written from its paper. Many were then checked against the
source code of a production ocean model, and some were ported line by line so the result
matches the compiled reference to round-off. Line-by-line ports are derivative works of
that code. This file lists the code they come from and its license. Docstrings cite the
exact `file:line` of the reference version.

Changes made to all ported code: translated from Fortran 77/90 to Python (NumPy, and
Numba where noted), vectorised over columns or points, and adapted to pyOCEAN's grid,
index order (k = 0 at the bed in σ/ALE) and data structures. Deliberate differences
from the original are documented in the docstrings.

---

## 1. MOM6 and CVMix: Apache License 2.0

**MOM6**, git `0e9a561`, https://github.com/mom-ocean/MOM6.
Copyright NOAA-GFDL and MOM6 contributors.
**CVMix**, as distributed in `MOM6/pkg/CVMix-src`, https://github.com/CVMix/CVMix-src.
Copyright CVMix contributors. Apache 2.0 with the LLVM exception.

License text: `LICENSES/Apache-2.0.txt`.

| pyOCEAN module | Ported from |
|----------------|-------------|
| `core/vertical/remap.py` | `MOM_remapping.F90`, `regrid_edge_values.F90`, `PPM_functions.F90`, `PLM_functions.F90` (remapping_core_h, PPM_H4) |
| `core/vertical/regrid.py` | `coord_zlike.F90`, `coord_sigma.F90`, `coord_hycom.F90`, `regrid_interp.F90` |
| `core/vertical/pgf.py`, `core/vertical/pgf_nonlinear.py` | `MOM_PressureForce_FV.F90`, `MOM_density_integrals.F90` (`int_density_dz_generic_plm`) |
| `core/vertical/ale.py` | `MOM_continuity_PPM.F90`, `MOM_vert_friction.F90` (upwind-biased viscous thickness) |
| `core/dynamics/vector_invariant.py` | `MOM_CoriolisAdv.F90` (Sadourny energy PV flux, BOUND_CORIOLIS, KE gradient), `MOM_vert_friction.F90` (`vertvisc_limit_vel`) |
| `core/dynamics/hor_visc.py` | `MOM_hor_visc.F90` (Laplacian and biharmonic Smagorinsky, BOUND_AH) |
| `core/dynamics/open_boundary.py` | `MOM_open_boundary.F90` (ORLANSKI_TAN, NUDGED_TAN, OBC vorticity and strain) |
| `core/components/bottom_drag.py`, `physics/bottom_friction.py` | `MOM_set_viscosity.F90` (bottom boundary layer drag) |
| `physics/eos/wright_full.py`, `physics/eos/roquet_rho.py` | `MOM_EOS_Wright_full.F90`, `MOM_EOS_Roquet_rho.F90` |
| `physics/eos/stratification.py` | `MOM_set_diffusivity.F90` (`find_N2`) |
| `physics/mixing/kappa_shear.py` | `MOM_kappa_shear.F90` (Jackson et al. 2008) |
| `physics/mixing/bbl_mixing.py` | `MOM_set_diffusivity.F90` (`add_LOTW_BBL_diffusivity`) |
| `physics/mixing/gls_merged.py` | `MOM_kappa_shear.F90` (`eliminate_massless` merge pattern) |
| `physics/mixing/kpp_cvmix.py` | CVMix `cvmix_kpp.F90`, MOM6 `MOM_CVMix_KPP.F90` |
| `physics/surface/shortwave.py` | `MOM_opacity.F90` (`absorbRemainingSW`), `MOM_diabatic_aux.F90` |
| `core/grid/supergrid.py` | `MOM_grid_initialize.F90` (metrics and angle from the supergrid) |

## 2. FMS and FMScoupler: GNU LGPL 3.0 or later

**FMS** 2025.02.01 (`sat_vapor_pres`) and **FMScoupler** 2025.02, git `5b700a4`
(`shared/surface_flux.F90`). https://github.com/NOAA-GFDL/FMS,
https://github.com/NOAA-GFDL/FMScoupler. Copyright NOAA-GFDL.

| pyOCEAN module | Ported from |
|----------------|-------------|
| `forcing/bulk_fms.py` (`bulk_ncar_fms`, `esat_fms`) | `surface_flux.F90` (`surface_flux_1d`, `ncar_ocean_fluxes`, `ncar_ocean_fluxes_multilevel`), `sat_vapor_pres_k.inc` (`compute_es_k`, `lookup_es_k`) |

This file is distributed under the **LGPL-3.0-or-later**, as its header states. It is a
separate module: `forcing/bulk.py` only imports and re-exports it. You may replace or
modify it under the LGPL. The rest of pyOCEAN stays under MIT. License texts:
`LICENSES/LGPL-3.0.txt` and `LICENSES/GPL-3.0.txt` (the LGPL is a set of additional
permissions on top of the GPL).

## 3. CROCO / ROMS: MIT/X-style

**CROCO** 2.1.3, git `ab63b58`, https://www.croco-ocean.org. The file headers say that
the routines from UCLA (Shchepetkin et al.) and Rutgers University (Arango et al.) are
under an MIT/X-style license. The CROCO-specific routines are under CeCILL-C (nesting),
and pyOCEAN ports none of them.

| pyOCEAN module | Ported from |
|----------------|-------------|
| `physics/mixing/gls.py` | `gls_mixing.F` (GLS k-ε, Canuto A stability functions) |
| `forcing/bulk_coare.py` | `bulk_flux.F` (COARE 3.0) |
| `physics/surface/shortwave.py` | `lmd_swfrac.F` (Jerlov profile) |
| `core/components/bottom_drag.py` | `get_vbc.F` (cross-velocity average for the bed speed) |

## 4. MITgcm: MIT License

**MITgcm**, https://github.com/MITgcm/MITgcm. Copyright (c) 2018 MITgcm Developers and
Contributors. License text: `LICENSES/MIT-MITgcm.txt`.

The implicit free surface (θ-method), the partial cells (hFac), the OBCS sponge profile
and the `delR`/`delX` grid conventions follow MITgcm. `forcing/bulk.py` (`bulk_ncar`)
follows `pkg/exf/exf_bulkformulae.F`. Most of this code was written from the papers and
checked against the MITgcm source; the docstrings cite the files.

## 5. HYCOM: MIT License

**HYCOM**, https://github.com/HYCOM. Copyright (c) 1999-present HYCOM Consortium and
contributors. License text: `LICENSES/MIT-HYCOM.txt`.

| pyOCEAN module | Ported from or following |
|----------------|--------------------------|
| `forcing/bulk.py` (`bulk_kara`) | `thermf.F90` (flxflg = 6), `mod_momtum.F90` (`cd_coarep`) |
| `physics/tides/drag.py` | `mod_barotp.F90` (`tiddrg` = 1) |
| `core/vertical/regrid.py` | `hybgen.F90` (massless-layer density fill) |
| `core/vertical/ale.py` | `mod_momtum.F90` (thin-layer blending), `cnuity` (barotropic flux correction) |
| `forcing/sources/hycom_ab.py`, `hycom_archive.py` | the `.a/.b` file formats (readers written independently) |

## 6. Followed but written independently

The following algorithms were read in the source of these projects and reimplemented.
No code was translated line by line.

- **CEFI-regional-MOM6** (git `36b8f49`, LGPL-3.0): the GloFAS river-mouth rule in
  `forcing/sources/glofas.py`, and the boundary and initial-state conventions in
  `forcing/ocean_init.py` and `forcing/interp.py`.
- **pyTMD** (MIT) and **OTPS** (TPXO): tidal constituent conventions in `forcing/tides.py`.
  pyTMD was used only as a test oracle.
- **GETM**: the TEF and mixing relation in `diagnostics/tef.py` come from Lorenz et al. (2021).
- **Veros** (GPL-3.0): used only as a reference for the architecture. No code was taken.

---

Corrections are welcome. If you maintain one of these projects and think an attribution
is missing or wrong, please open an issue.
