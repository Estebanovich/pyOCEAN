# Installation

pyOCEAN is pure Python with Numba-compiled kernels: there is nothing to compile by hand, no Fortran and no
MPI. You need a Python environment with the scientific stack, a copy of the repository, and the right
thread settings. This page gives the environment that is known to work, a recipe to rebuild it, the list
of every dependency the code imports, how to run the tests, how to build this manual, and how to set the
threads.

## The environment that is known to work

All development, tests and validation runs of the repository use the **base conda environment
`/opt/anaconda3`** on an Apple M3 Pro (macOS, arm64). Its versions, as of 2026-10-14:

| Package | Version | Needed for |
|---|---|---|
| Python | 3.12.4 | everything |
| numpy | **1.26.4** (pinned) | everything |
| numba | 0.59.1 (llvmlite 0.42.0) | all compiled kernels of the σ/ALE core |
| scipy | 1.13.1 | sparse free-surface solvers (SuperLU, CG), interpolation, `ndimage` |
| xarray | 2023.6.0 | grids, forcing, output, restarts |
| netCDF4 | 1.7.1.post2 | NetCDF4 engine of restarts and output |
| zarr | 2.18.7 | Zarr output (`OutputWriter(fmt="zarr")`) |
| pandas | 2.2.2 | tide gauges, sea-level QC, WRF adapter |
| gsw | 3.6.19 | TEOS-10 (HYCOM z-level adapter, phase-1 EOS, θ/S_P ↔ Θ/S_A conversions and freezing point of the 3-D `eos="teos10"`; optional) |
| matplotlib | 3.8.4 | figures of the validation scripts and of this manual |
| pytest | 8.4.2 | the test suite |

!!! danger "numpy must stay at 1.26.4"
    numba 0.59.1 requires numpy < 1.27. A `pip install` of another package once upgraded numpy to 2.2.6
    silently and broke every Numba kernel. **Never let pip touch numpy**: install extra packages with
    `pip install --no-deps <package>==<version>` after checking their requirements, then run `pip check`.

!!! note "`environment.yml` holds these versions (since 2026-11-08)"
    The repository's `environment.yml` pins the versions above (plus `h5netcdf`, used for ERA5 files), so
    `conda env create -f environment.yml` gives the tested set; a conda dry-run on osx-arm64 solves it
    without conflicts. It has not yet been created on a clean machine; report any problem.

## A recipe for a new environment

The following conda command reproduces the tested versions in a separate environment. It is a recipe
derived from the working environment, not one that has been run on a clean machine; report any problem.

```bash
conda create -n pyocean -c conda-forge \
    python=3.12 numpy=1.26.4 numba=0.59.1 scipy=1.13 xarray=2023.6 \
    netcdf4=1.7 h5netcdf=1.6 "zarr>=2.18,<3" pandas=2.2 gsw=3.6 matplotlib=3.8 pytest
conda activate pyocean
python -c "import numpy, numba; print(numpy.__version__, numba.__version__)"   # 1.26.4 0.59.1
```

Then get the code (pyOCEAN is not packaged; there is no `pip install pyocean`):

```bash
git clone <repository URL> pyOCEAN
cd pyOCEAN
```

Scripts add the repository root to `sys.path` themselves (see the
[Quick start](quickstart.md)); in an interactive session either start Python from the repository root or
set `PYTHONPATH=/path/to/pyOCEAN`.

## Dependencies the code actually imports

From the import statements of `core/`, `physics/`, `numerics/`, `boundaries/`, `forcing/`,
`diagnostics/`, `scripts/` and `tests/` (validation and prototype scripts listed separately):

| Package | Required? | Imported by |
|---|---|---|
| numpy | **required** | everything |
| scipy | **required** | `core/dynamics/wetting_drying.py`, `free_surface.py`, `pressure.py` (sparse solvers); `core/grid/bathymetry.py`, `forcing/tides.py`, `physics/tides/drag.py` (interpolation); `core/grid/vertical.py` (`brentq`, local import); `diagnostics/domain_check.py` (`ndimage`) |
| xarray | **required** | grids, state I/O, forcing, output, restarts |
| numba | **required** for the σ/ALE core | `core/dynamics/sigma_baroclinic.py`, `sigma_wd.py`, `vector_invariant.py`, `nonlinear.py`, `core/vertical/*` |
| netCDF4 | required for restarts and NetCDF output (xarray engine `"netcdf4"`) | `core/restart.py`, `diagnostics/output.py`, `diagnostics/io.py`, forcing readers |
| zarr (< 3) | optional: only `OutputWriter(fmt="zarr")` (the default format of `OutputWriter`) | `diagnostics/output.py` through xarray |
| pandas | optional: tide gauges, sea-level QC, WRF times | `diagnostics/tide_gauges.py`, `diagnostics/sealevel_qc.py`, `forcing/sources/wrf.py` |
| gsw | optional (only for `eos="teos10"` and Θ/S_A conversions; `make_eos("teos10")` raises ImportError without it) | `physics/eos/convert.py` (imported inside the conversion and freezing functions), `forcing/sources/hycom_z.py` (top-level import), `physics/eos/__init__.py` (imported only for `eos="teos10"`) |
| matplotlib | optional: figures | validation scripts, manual figures (no core module imports it) |
| pytest | for the tests | `tests/` |
| jax 0.4.35 | optional, prototypes only | `prototypes/` |
| pyTMD, MITgcmutils | optional, only to regenerate references | `validation/tides/make_reference.py`, `validation/mitgcm_exf/compare.py` |

Not used anywhere, despite being mentioned in older documents: PyAMG, PyVista, cartopy, MLX. (dask is not imported by pyOCEAN, but `xarray.open_zarr` uses it, when installed, to read Zarr output lazily.)

## Running the tests

From the repository root:

```bash
python -m pytest tests/                 # or simply: pytest tests/
python -m pytest tests/test_sigma_ale.py -q   # one module
```

- 1350 tests are collected (2026-10-14). `CLAUDE.md` reports 1348 passing, 1 strict `xfail` (Thacker
    wetting–drying front) and 1 `xfail` (FV vs Jacobian PGF over the seamount).
- Benchmark tests **skip** when their external inputs are absent: the BayIW MITgcm runs on
    `/Volumes/ESTEBAN_MIT` (`test_bayiw_2d.py`, `test_bayiw_cut.py`, `test_bayiw_baroclinic.py`), the MITgcm
    `internal_wave` inputs (`test_internal_wave.py`), the compiled MOM6 column drivers
    (`test_vertical_vs_mom6.py`), the CROCO source (part of `test_ilicak_benchmark.py`), and netCDF4
    (`test_ilicak_benchmark.py`, `test_mom6_bayiw.py`).
- `tests/conftest.py` caps `NUMBA_NUM_THREADS` at 5 before any test imports Numba, and
    `tests/test_threads.py` checks the cap.
- The first run compiles the Numba kernels (they are cached in `__pycache__` afterwards), so it is slower.

!!! note "netCDF4 warning"
    netCDF4 in the base environment emits `RuntimeWarning: numpy.ndarray size changed, may indicate binary
    incompatibility`. It is harmless (a binary-build mismatch). If you run with `-W error`, filter it:
    `pytest -W error -W "ignore:numpy.ndarray size changed:RuntimeWarning" tests/`.

## Building this manual

The manual is built with MkDocs Material in a **separate virtual environment**, so the base environment's
numpy pin is never touched (`docs/requirements-docs.txt`):

```bash
python -m venv .venv-docs
.venv-docs/bin/pip install -r docs/requirements-docs.txt   # mkdocs 1.6.1, mkdocs-material 9.5.39, pymdown-extensions 10.11.2
.venv-docs/bin/mkdocs serve      # live preview at http://127.0.0.1:8000
.venv-docs/bin/mkdocs build      # static site in site/
```

The configuration is `mkdocs.yml` at the repository root (`docs_dir: docs/manual`). Math is rendered with
MathJax 3 (loaded from a CDN) and diagrams with Mermaid. Figures are produced by the scripts in
`docs/manual/figures/` and committed as PNG files under `docs/manual/img/`.

## Thread settings

The σ/ALE core parallelises its Numba kernels with threads; SciPy's SuperLU and the NumPy glue are
single-threaded. Set the threads **before** importing NumPy or Numba:

```python
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
    os.environ.setdefault(v, "1")          # BLAS/OpenMP: 1 (no gain from more)
os.environ.setdefault("NUMBA_NUM_THREADS", "5")   # Numba pool: at most the performance cores

from core.dynamics.sigma_baroclinic import set_threads
set_threads(5)                              # threads used by SigmaBaroclinic's kernels (default 1)
```

- `NUMBA_NUM_THREADS` fixes the size of Numba's thread pool at import time. Without it Numba uses every
    core (11 on the M3 Pro, including the 6 efficiency cores).
- `set_threads(n)` (`core/dynamics/sigma_baroclinic.py:166`) chooses how many of those threads the model
    uses; it is clipped to the pool size and **defaults to 1**. Results are bit-identical for any count.
- Kernels you call directly (outside `SigmaBaroclinic`) ignore `set_threads`; use
    `numba.set_num_threads(n)`.
- On the M3 Pro keep the **total** over all concurrent runs at or below the 5 performance cores (one run
    × 5 threads, or two runs × 2), and do not lower the process priority (`nice`, `taskpolicy`), which
    sends the work to the efficiency cores. There is no core-affinity API on macOS, so this is best effort.
- JAX (prototypes only) cannot be limited to the performance cores on this machine
    (`docs/design/performance.md` §4).

More in [Performance and parallelism](../architecture/performance.md).

## Optional external data

The model itself needs no external data. Some validation cases and tests read large inputs kept outside
the repository: the BayIW MITgcm runs (`/Volumes/ESTEBAN_MIT`), reference model sources and builds
(`~/pyOCEAN_refs/MOM6`, `croco`, `MITgcm`, `FMS`), the TPXO10 atlas (`~/pyOCEAN_refs/TPXO10_atlas_v2_nc`),
GEBCO extracts and the HYCOM source (`HYCOM-src/`, git-ignored). They are needed only to rerun those
benchmarks; see the [Validation](../validation/index.md) chapter.
