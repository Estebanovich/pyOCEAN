# pyOCEAN

A modern, Python-native regional ocean model built from scratch, inspired by HYCOM and MITgcm.

**Author:** Esteban Cruz Isidro
**Institution:** UNAM — Institute of Atmospheric Sciences and Climate Change
**Contact:** estebanci96@gmail.com

---

## Motivation

HYCOM and MITgcm are powerful tools but are written in legacy Fortran, designed for HPC clusters, and difficult to modify or understand at a deep level. pyOCEAN is built from scratch in Python with the explicit goal of:

1. **Understanding** every equation and numerical scheme from the ground up
2. **Improving** specific physics (internal wave mixing, hybrid vertical coordinates, open boundary conditions)
3. **Running efficiently** on modern hardware including Apple Silicon (M-series)
4. **Integrating naturally** with the modern Python scientific stack (xarray, NumPy, Matplotlib)

This is not a wrapper around an existing model — every component is implemented and understood from first principles.

---

## Scientific Goals

- Implement a HYCOM-like hybrid vertical coordinate system (isopycnal + sigma + z-level) in clean Python
- Improve internal wave parameterization beyond standard KPP (based on M.Sc. research)
- Implement MITgcm-quality open boundary conditions (Orlanski radiation + sponge layers)
- Target regional ocean forecasting for the Gulf of Mexico
- Build native xarray diagnostics — no postprocessing conversion needed

---

## Model Physics

The model solves the **hydrostatic primitive equations** under the Boussinesq approximation:

```
Du/Dt - fv = -1/ρ₀ ∂p/∂x + ∂/∂z(Kv ∂u/∂z) + Fh(u)
Dv/Dt + fu = -1/ρ₀ ∂p/∂y + ∂/∂z(Kv ∂v/∂z) + Fh(v)
∂p/∂z     = -ρg          (hydrostatic balance)
∂u/∂x + ∂v/∂y + ∂w/∂z = 0  (continuity)
DT/Dt     = ∂/∂z(Kt ∂T/∂z) + Fh(T)
DS/Dt     = ∂/∂z(Ks ∂S/∂z) + Fh(S)
ρ         = ρ(T, S, p)    (equation of state — TEOS-10)
```

---

## Project Structure

```
pyOCEAN/
├── core/
│   ├── grid/              # Horizontal and vertical grid generation
│   ├── dynamics/          # Momentum, pressure, advection
│   ├── thermodynamics/    # T/S equations, equation of state
│   └── timestepping/      # Time integration schemes
├── physics/
│   ├── mixing/            # KPP, internal wave parameterization
│   ├── tides/             # Tidal forcing
│   └── eos/               # Equation of state (TEOS-10)
├── boundaries/
│   ├── obcs/              # Open boundary conditions (Orlanski + sponge)
│   └── surface/           # Atmospheric forcing (bulk formulae)
├── numerics/
│   ├── advection/         # WENO5, PPM schemes
│   ├── diffusion/         # Laplacian, biharmonic
│   └── solvers/           # Elliptic solver (barotropic pressure)
├── diagnostics/
│   ├── output.py          # Native xarray output
│   ├── energy.py          # Energy budget diagnostics
│   └── waves.py           # Internal wave diagnostics
├── forcing/
│   ├── era5.py            # ERA5 atmospheric forcing reader
│   └── bathymetry.py      # GEBCO bathymetry reader
├── configs/               # Model configuration files
├── tests/                 # Unit and integration tests
├── notebooks/             # Jupyter notebooks for validation and analysis
├── docs/                  # Documentation
├── environment.yml        # Conda environment
├── README.md
└── CLAUDE.md
```

---

## Current Status

**Phase 1 in progress** — building the z-level Cartesian core from scratch.

| Module | Description | Tests |
|--------|-------------|-------|
| `core/grid/horizontal.py` | Arakawa C-grid (Cartesian + spherical) | ✅ 28/28 |
| `core/grid/vertical.py` | z-levels with partial cells (hFac) | ✅ 37/37 |
| `core/state.py` | Prognostic variable container (u,v,w,T,S,η) | ✅ 52/52 |
| `physics/eos/` | TEOS-10 (gsw) + linear EOS | ✅ 36/36 |
| `core/dynamics/pressure.py` | Hydrostatic pressure + PGF + CG solver | ✅ 34/34 |
| `core/dynamics/momentum.py` | Momentum equations + Coriolis | pending |
| `core/timestepping/` | Adams–Bashforth 2 | pending |

**Next:** momentum equations (Coriolis + PGF) and Adams–Bashforth 2 timestepper.

Run the test suite: `pytest tests/`

---

## Development Phases

| Phase | Description | Status |
|-------|-------------|--------|
| 1 | Cartesian z-level idealized model (rectangular basin) | In progress |
| 2 | Real topography + open boundary conditions | Planned |
| 3 | Hybrid vertical coordinate (sigma + isopycnal + z) | Planned |
| 4 | Internal wave parameterization | Planned |
| 5 | Operational pipeline (Gulf of Mexico) | Planned |

---

## Tech Stack

| Component | Library |
|-----------|---------|
| Core numerics | NumPy |
| Performance (JIT) | Numba |
| Data structures & I/O | xarray, netCDF4, zarr |
| Visualization | Matplotlib, PyVista |
| GPU (Apple Silicon) | MLX / JAX |
| Testing | pytest |
| Environment | conda |

---

## Validation

Phase 1 results will be validated against existing MITgcm simulations:
- Internal wave and seiche dynamics in a rectangular bay (M.Sc. thesis configurations)
- BayIW_Rectan and BayIW_Rectan_2layer experiments

---

## References

- Bleck, R. (2002). An oceanic general circulation model framed in hybrid isopycnic-Cartesian coordinates. *Ocean Modelling*, 4, 55–88. (HYCOM)
- Marshall, J. et al. (1997). A finite-volume, incompressible Navier-Stokes model for studies of the ocean on parallel computers. *JGR Oceans*. (MITgcm)
- Large, W. G., McWilliams, J. C., & Doney, S. C. (1994). Oceanic vertical mixing: A review and a model with a nonlocal boundary layer parameterization. *Reviews of Geophysics*. (KPP)
- IOC, SCOR and IAPSO (2010). The international thermodynamic equation of seawater – 2010. (TEOS-10)
