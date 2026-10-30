# pyOCEAN Model Manual

**pyOCEAN** is a regional and coastal ocean model written from scratch in Python. It solves the
hydrostatic primitive equations under the Boussinesq approximation on an Arakawa C grid, with a free
surface that can wet and dry, terrain-following σ-layers or an Arbitrary Lagrangian–Eulerian (ALE)
vertical coordinate (z\*, σ or a HYCOM-like hybrid isopycnal coordinate), open boundaries for nesting, and
the tooling around a regional model: bathymetry, forcing adapters (ERA5, WRF, HYCOM, TPXO), tides,
tide-gauge analysis, CF/Zarr output, bit-for-bit restarts and spurious-mixing diagnostics. It is not a
wrapper around HYCOM, MITgcm, ROMS or MOM6: every scheme is implemented from its paper, compared with the
source code of the production model that has it (cited as `file:line` in the docstrings), and validated
against analytic solutions, observations and those models.

The code is plain NumPy inside the time step, with the hot loops compiled by Numba; there is nothing to
compile by hand. It runs on a laptop (the development machine is an Apple M3 Pro).

## Goals

From the design document `docs/design/vision_roadmap.md` (approved 2026-10-11):

- **Be the best community regional and coastal model**, not a model tailored to one user. "Best" is
    measured on four axes: **accuracy** against truth (exact solutions, laboratory, observations — agreeing
    with other models is not enough), **cost** (time per simulated day at equal error), **robustness** (no
    blow-ups with real bathymetry, wetting–drying or open boundaries) and **usability and transparency**
    (no compilation, integrated domain preparation, every equation with its reference and its test,
    bit-for-bit reproducible results).
- **Fill gaps of the existing models**: no code combines ALE + wetting–drying + tides from the coast to the
    basin; 3-D baroclinic wetting–drying is open; spurious mixing is rarely diagnosed routinely; no
    differentiable 3-D regional model exists; open boundaries are tuned by hand.
- **Rule for every benchmark**: each difference is classified as (a) our bug, (b) missing physics,
    (c) a numerical choice or (d) an error of the reference model; only (a) and (b) change the model, and
    nothing is tuned to a single case.
- **Roadmap** (stages A–F): quality foundations (done), 3-D physics, the ALE hybrid coordinate,
    a differentiable JAX backend, internal waves and baroclinic tides, an operational and machine-learning
    cycle. See [Limitations and roadmap](roadmap.md).

pyOCEAN also has an educational purpose: to understand every component of a HYCOM-like model by
implementing it from first principles, and to improve specific physics, especially internal-wave dynamics.

## What exists

Maturity: **validated** = tested against an analytic solution, observations or another model, with the
result documented; **experimental** = implemented and unit-tested, validation partial or open;
**legacy** = phase-1 code kept with its tests, not extended.

| Capability | Where | Maturity | Evidence |
|---|---|---|---|
| Stratified σ-layer 3-D model (WENO5 tracers and momentum, density-Jacobian PGF, AB3, wetting–drying) | `SigmaBaroclinic` | validated | lock exchange = CROCO; seamount; BayIW vs MITgcm |
| ALE z\* coordinate (Lagrangian step, regrid, PPM_H4 remap, FV PGF, vector-invariant momentum) | `SigmaBaroclinic(vertical_coordinate="ale", regrid="zstar")` | validated | = MOM6 z\* (lock exchange, BayIW); seamount 0.08 cm s⁻¹; MITgcm internal wave |
| ALE hybrid (HYCOM1 rule, isopycnal interior, z\* floor) | `regrid="hybrid"` | validated in idealised cases, experimental in 3-D two-layer | hybrid lock exchange = MOM6; Ilicak benchmark; BayIW two-layer phase open |
| ALE σ | `regrid="sigma"` | validated | lock exchange (Benjamin ±10 %) |
| Spurious-mixing diagnostics (RPE, online variance decay `mix_num`) | `diagnostics/mixing.py`, `SigmaState.mix_num` | validated | exact variance budget; Ilicak et al. (2012) behaviour |
| Open boundaries: Flather + Orlanski + sponge, time-varying exterior (nesting) | `open_boundaries`, `open_boundary_state` | validated | mode-1 channel error 0.015; analytic wave ±5 %; BayIW nested |
| Wetting and drying, 2-D and 3-D σ | `WetDryFreeSurface`, `SigmaWetDry` | validated (Thacker front: strict `xfail`) | CROCO TIDAL_FLAT 2-D 1.1 cm, 3-D 1.2 cm |
| GLS k–ε turbulence closure (CROCO port, stratified), KPP (CVMix/MOM6 port), parabolic viscosity, log-law bed drag (full bed speed) | `physics/mixing/gls.py`, `physics/mixing/kpp_cvmix.py`, `parabolic.py`, `physics/bottom_friction.py`, `core/components/closure.py` | validated (σ and ALE) | CROCO TIDAL_FLAT 3-D; single columns vs CROCO and MOM6 (`validation/turbulence_columns/`); Lorenz ALE-σ + GLS ≈ σ + GLS |
| 2-D barotropic tides (TPXO boundaries, body tide, SAL, JSL drag) | `forcing/tides.py`, `ImplicitFreeSurface`, `validation/pacmex_tides/` | validated | 31 gauges: median RSS 1.44 cm vs TPXO10 1.79 cm |
| Tide prediction (OTPS/TPXO conventions), TPXO and HYCOM SAL readers | `forcing/tides.py`, `forcing/sources/tpxo.py` | validated | = pyTMD to round-off |
| Tide-gauge comparison and raw sea-level QC | `diagnostics/tide_gauges.py`, `sealevel_qc.py` | validated | TICON-3, IOC/SMN network |
| Bulk air–sea fluxes NCAR/CORE and Kara (HYCOM `flxflg=6`) | `forcing/bulk.py` | validated as a component; not yet coupled to the 3-D core | = compiled HYCOM to 1e-10 |
| Forcing adapters: ERA5, WRF, HYCOM z-level, HYCOM `.a/.b` and archives | `forcing/sources/` | validated on real and synthetic files | unit tests |
| Bathymetry pipeline and domain checker (rx0/rx1, pits, isolated water) | `core/grid/bathymetry.py`, `diagnostics/domain_check.py` | validated | found the Guaymas pit unaided |
| Bit-for-bit restarts | `core/restart.py` | validated | continuous = restarted run (σ, GLS, ALE) |
| CF-1.8 / SGRID output, Zarr or NetCDF4 | `diagnostics/output.py` | validated (σ geometry; no ALE layer thickness field yet) | unit tests |
| Blow-up post-mortem | `diagnostics/postmortem.py` | validated | unit tests |
| Numba kernels with NumPy oracles, CG + lagged-LU solver | throughout `core/` | validated | kernel = oracle tests |
| Salinity, TEOS-10, surface heat/freshwater fluxes, tides and KPP inside the 3-D core | — | not yet (planned, `docs/design/unified_core.md`) | |
| Phase-1 z-level model (TEOS-10/linear EOS, KPP, upwind tracers, AB2) | `core/state.py`, `core/timestepping/ab2.py`, … | legacy | 380 phase-1 tests |
| JAX tracer-advection prototype | `prototypes/` | experimental | benchmark only |

The numbers come from `CLAUDE.md`, `docs/design/vision_roadmap.md` §7 and the READMEs in `validation/`;
the [Validation](validation/index.md) chapter gives each one with its source.

## How this manual is organised

**Getting started**

- [Installation](getting-started/installation.md) — environment, dependencies, tests, building this manual, threads.
- [Quick start (first run)](getting-started/quickstart.md) — a lock exchange in σ, z\* and hybrid, line by line.
- [Model overview in one page](getting-started/overview.md) — equations, model lines, component diagram, every choice with its default and reference.

**User guide** — how to set up and run a model

- [Building a model (SigmaBaroclinic)](user-guide/building-a-model.md)
- [Configuration reference](user-guide/configuration-reference.md)
- [Regional setup (grid, bathymetry, initial and boundary data)](user-guide/regional-setup.md)
- [Surface forcing](user-guide/surface-forcing.md)
- [Tides and the 2-D tidal model](user-guide/tidal-model.md)
- [Output, restarts and diagnostics](user-guide/output-restart-diagnostics.md)
- [Troubleshooting and FAQ](user-guide/troubleshooting.md)

**Theory and numerics** — what the code solves and how

- [Governing equations](theory/equations.md)
- [Grids and bathymetry](theory/grids.md)
- [Vertical coordinates and ALE](theory/vertical-ale.md)
- [Momentum equations](theory/momentum.md)
- [Pressure gradient](theory/pressure-gradient.md)
- [Free surface and time stepping](theory/free-surface-time-stepping.md)
- [Tracer transport and spurious mixing](theory/tracers.md)
- [Turbulence closures and bottom friction](theory/turbulence-friction.md)
- [Open boundaries and nesting](theory/open-boundaries.md)
- [Wetting and drying](theory/wetting-drying.md)
- [Air–sea fluxes](theory/air-sea-fluxes.md)
- [Tides](theory/tides.md)

**Architecture** — how the code is organised

- [Code map](architecture/code-map.md) — every package and module, with the import graph.
- [The time step, flow diagrams](architecture/time-step.md) — the order of operations of `step`, σ and ALE.
- [Data structures and conventions](architecture/data-structures.md) — `SigmaState`, the grid Dataset, index order, units.
- [Performance and parallelism](architecture/performance.md) — Numba kernels, threads, solvers, measured costs.

**Validation** — how far each part can be trusted

- [Strategy and scorecard](validation/index.md)
- [Analytical and exact solutions](validation/analytical.md)
- [Idealised benchmarks (Ilicak, lock exchange, seamount, internal waves)](validation/idealised.md)
- [BayIW (bay internal waves and seiches) vs MITgcm and MOM6](validation/bayiw.md)
- [Tides and sea level (PAC_MEX, gauges, HYCOM-IOA)](validation/tides-sealevel.md)
- [Wetting and drying (Thacker, TIDAL_FLAT)](validation/wetting-drying.md)
- [Forcing and column checks (bulk, HYCOM, MOM6 columns)](validation/components.md)

**Context**

- [Comparison with other models](comparison.md)
- [Limitations and roadmap](roadmap.md)
- [References](references.md)

Conventions used throughout: code is cited as `path/to/file.py:LINE` with the function name (line numbers
drift; names do not); papers as Author (Year), with the full reference at the end of each page;
references not yet checked against the paper are marked **[verify]**. The design documents and most
validation READMEs in the repository are in Spanish; this manual is in English.

## Authorship, citation and contact

pyOCEAN is developed by **Esteban Cruz Isidro**, physical oceanographer, Universidad Nacional Autónoma de
México (UNAM) — Institute of Atmospheric Sciences and Climate Change (`README.md`). The code is released
under the MIT License (`LICENSE`, © 2026 Esteban Cruz Isidro).

**How to cite.** There is no paper or DOI for pyOCEAN yet. Until there is, cite the software as

> Cruz Isidro, E. (2026). *pyOCEAN: a regional ocean model written from scratch in Python* (software).
> UNAM, Institute of Atmospheric Sciences and Climate Change.

and cite the papers of the schemes you rely on (each page of this manual lists them).

**Contact:** estebanci96@gmail.com (`README.md`).
