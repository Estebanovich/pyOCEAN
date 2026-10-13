# pyOCEAN vs MOM6 (and HYCOM cost) on BayIW — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Cheap runs of MOM6 (z\*, HYCOM1, HYBGEN) on the BayIW 2-D section (results) and the BayIW cut (cost), and HYCOM 2.3 (cost only, behind a feasibility gate), compared with pyOCEAN and MITgcm.

**Architecture:** `validation/mom6_bayiw/` generates MOM6 inputs (NetCDF) and namelists from `bayiw`/`section`/`cut`, runs the optimized MOM6 build, converts its output to the 2-D section `series.npz` format so `validation/bayiw_2d/compare.py` compares it unchanged, and parses the FMS clock table for cost. `validation/hycom_bayiw/` writes `.a/.b` inputs and a `blkdat.input`, builds HYCOM from a copied config, and parses its timers.

**Tech Stack:** Python 3.12 (NumPy, netCDF4, xarray), MOM6 + FMS (gfortran, MPICH, Homebrew), HYCOM 2.3; bash.

**Spec:** `docs/superpowers/specs/2026-10-13-mom6-bayiw-benchmark-design.md`

## Global Constraints

- Simulations yes, but considerably cheap (Esteban): cost runs 3 h simulated; section runs 60 h on 1 × 352; the 3-D 30-h MOM6 run only if the section does not settle the two-layer question; any run > 3× its estimate is reported before going on.
- ≤ 5 MPI processes / threads; never the efficiency cores; one heavy run at a time.
- MOM6, FMS and HYCOM sources are not modified; builds and runs live outside the repo (`~/pyOCEAN_refs/MOM6/.testing/build/repro`, `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/{mom6_bayiw,hycom_bayiw}/`).
- No package installs without asking; numpy stays 1.26.4.
- Tests run no simulation (synthetic inputs only).
- Never `git add` a directory; never commit `log_*.txt` or `*.npz`. Commit messages end with `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn`.
- MOM6 facts used below (subagent report, file:line in `~/pyOCEAN_refs/MOM6/src`): `REENTRANT_X` defaults True (`framework/MOM_domains.F90:189`) → set False; mosaic grid needs only `GRID_FILE` (supergrid `x,y(nyp,nxp)`, `dx(nyp,nx)`, `dy(ny,nxp)`, `area(ny,nx)`, half-cell lengths / quarter areas; `initialization/MOM_grid_initialize.F90:192-289`); f-plane `ROTATION="betaplane"`, `F_0`, `BETA=0`; z data init `INIT_LAYERS_FROM_Z_FILE`, `Z_INIT_ALE_REMAPPING`, `_FillValue` mandatory, source edges = midpoints of the centres (`MOM_state_initialization.F90:2674-2722`, `framework/MOM_horizontal_regridding.F90:438-480`); wind `WIND_CONFIG="file"`, `WIND_STAGGER="C"`, `WIND_FILE_DAYS_PER_RECORD=-72` (records piecewise constant, record k on [1200(k−1), 1200k) s; `config_src/drivers/solo_driver/MOM_surface_forcing.F90:1523-1566, 1946-1985`); linear EOS `RHO_REF_LINEAR_EOS`, `T_REF_LINEAR_EOS`, `DRHO_DT`, `DRHO_DS` (`equation_of_state/MOM_EOS.F90:1598-1633`); HYCOM1 `ALE_COORDINATE_CONFIG="HYBRID:hyb.nc,sigma2,dz"` (`ALE/MOM_regridding.F90:589-618`); HYBGEN targets `COORD_CONFIG="file"` (`initialization/MOM_coord_initialization.F90:459-470`); OBC segments `"J=0,I=0:N,FLATHER,ORLANSKI"` with a `_DATA` line each (`core/MOM_open_boundary.F90:1319-1326, 1795-1801`); clocks `clock_grain='ROUTINE'` in `&fms_nml`; `LAYOUT = 1, 5`.

## Review Focus

1. **Wind staggering and record calendar in MOM6:** `tauy` must sit on the north face of row j (drop the southernmost v-face row) and record k must cover [1200(k−1), 1200k) s, matching MITgcm's records at (k − ½)·1200 s only to within half a record — the comparison must state that offset. *Pinned in Task 2 (`test_mom6_wind_file_is_on_north_faces_and_records`).*
2. **z-data source edges:** MOM6 builds source edges as midpoints of the given centres; centres must reproduce the MITgcm interfaces, or the two-layer jump smears in the initial state. *Pinned in Task 2 (`test_ts_z_centres_reproduce_the_mitgcm_interfaces`).*
3. **Land and walls in the section:** columns 0 and 2 land (depth 0 ≤ `MASKING_DEPTH`), `REENTRANT_X=False`, south OBC only. *Pinned in Task 3 (`test_mom_input_section_has_walls_and_south_obc`).*
4. **Target densities in model units:** HYCOM1/HYBGEN targets are full linear-EOS densities (~997–999 kg m⁻³), the same values as `bayiw` targets. *Pinned in Task 2 (`test_coordinate_files_hold_the_bayiw_targets`).*
5. **MOM6 output → series:** η, T and V sampled on the MITgcm levels exactly as pyOCEAN's `record` (same `to_mitgcm_levels`, `mit_at_depth`). *Pinned in Task 4 (`test_series_from_mom6_matches_the_pyocean_record_layout`).*

---

### Task 1: Optimized MOM6 build and smoke run

**Files:**
- Create: `validation/mom6_bayiw/run_mom6.sh` (subcommand `build`)

- [ ] **Step 1: Script `build`**

```bash
#!/usr/bin/env bash
# MOM6 for the BayIW benchmark (spec 2026-10-13-mom6-bayiw-benchmark-design §2).
#   run_mom6.sh build                                   : optimized build (.testing build/repro, FCFLAGS -g -O2)
#   run_mom6.sh run <dir>                               : run the case in <dir> on 5 MPI ranks
set -euo pipefail
T=~/pyOCEAN_refs/MOM6/.testing
EXE=$T/build/repro/MOM6
if [[ "$1" == "build" ]]; then
  cd "$T" && make -j5 build/repro/MOM6 > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_bayiw/build_repro.log 2>&1
  ls -la "$EXE"; grep -m1 "^FCFLAGS" "$T/build/repro/Makefile"
elif [[ "$1" == "run" ]]; then
  cd "$2"
  mpirun -n 5 "$EXE" > stdout.txt 2>&1 &                       # MPICH may hang in MPI_Finalize after MOM6 ends
  pid=$!
  for _ in $(seq 1 3600); do
    grep -q "Total runtime" stdout.txt 2>/dev/null && break
    kill -0 $pid 2>/dev/null || break
    sleep 1
  done
  sleep 2; pkill -f "$EXE" 2>/dev/null || true
  grep -E "Total runtime|FATAL" stdout.txt | head -2 || tail -5 stdout.txt
fi
```

- [ ] **Step 2: Build** — `mkdir -p /Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_bayiw && bash validation/mom6_bayiw/run_mom6.sh build`. Expected: executable listed, FCFLAGS `-g -O2`. If `make build/repro/MOM6` needs the deps to be rebuilt with other flags, stop and ledger it (do not touch `build/symmetric`).
- [ ] **Step 3: Smoke** — copy `MOM6-examples/ocean_only/resting` (or `lock_exchange`) to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_bayiw/smoke`, set `LAYOUT = 1, 5` if its NJGLOBAL allows, else `mpirun -n 1`, run 10 steps (`DAYMAX` small). Expected: "Total runtime" in stdout.
- [ ] **Step 4: Commit** (`git add validation/mom6_bayiw/run_mom6.sh`).

---

### Task 2: MOM6 inputs from `bayiw`

**Files:**
- Create: `validation/mom6_bayiw/inputs.py`
- Test: `tests/test_mom6_bayiw.py`

**Interfaces:**
- Produces: `domain(kind, case, bay) -> dict(dx (nx,), dy (ny,), H (ny,nx) NaN land, T (50,ny,nx), wind (nrec, ny+1, nx) v-faces, x0, y0)` for `kind` `"section"` (nx = 3: wall | column i=280 | wall, as `section.write_mitgcm_inputs`) or `"cut"` (`cut.cut_inputs(case, bay, "py")`); `write_inputs(kind, case, bay, out_dir) -> None` writing `ocean_hgrid.nc`, `topog.nc`, `ts_z.nc`, `wind.nc`, `hyb.nc`, `layers.nc` in `out_dir/INPUT`; `source_centres() -> (50,) depths [m]`.

- [ ] **Step 1: Failing tests**

```python
import numpy as np
import pytest
netCDF4 = pytest.importorskip("netCDF4")


def test_ts_z_centres_reproduce_the_mitgcm_interfaces():
    from validation.bayiw_baroclinic import bayiw
    from validation.mom6_bayiw import inputs
    c = inputs.source_centres()
    edges = 0.5 * (c[1:] + c[:-1])                     # MOM6: midpoints of the centres are the source edges
    np.testing.assert_allclose(edges, -bayiw.Z_F[1:-1], atol=1e-9)
    assert (np.diff(c) > 0).all()


def test_mom6_inputs_match_bayiw_at_corners_and_stations(tmp_path):
    from validation.bayiw_2d import section
    from validation.mom6_bayiw import inputs
    d = inputs.domain("section", "2layer", "bay")
    inputs.write_inputs("section", "2layer", "bay", tmp_path)
    with netCDF4.Dataset(tmp_path / "INPUT" / "topog.nc") as ds:
        depth = ds["depth"][:]
    s = section.section_inputs("2layer", "bay")
    assert depth.shape == (352, 3) and (depth[:, 0] == 0).all() and (depth[:, 2] == 0).all()
    np.testing.assert_allclose(depth[:, 1], np.nan_to_num(s["H"][:, 0]))
    with netCDF4.Dataset(tmp_path / "INPUT" / "ocean_hgrid.nc") as ds:
        dxs = ds["dx"][:]
        assert ds.dimensions["nxp"].size == 7 and ds.dimensions["nyp"].size == 705
    np.testing.assert_allclose(dxs[0, 2] + dxs[0, 3], s["dx"][0])  # two half-cell lengths = the cell width
    with netCDF4.Dataset(tmp_path / "INPUT" / "ts_z.nc") as ds:
        T = ds["ptemp"][0]
        assert "_FillValue" in ds["ptemp"].ncattrs()
    j = section.SECTION_STATIONS["Omega"]
    np.testing.assert_allclose(T[:, j, 1], s["T"][:, j, 0])


def test_mom6_wind_file_is_on_north_faces_and_records(tmp_path):
    from validation.mom6_bayiw import inputs
    d = inputs.domain("section", "2layer", "bay")
    inputs.write_inputs("section", "2layer", "bay", tmp_path)
    with netCDF4.Dataset(tmp_path / "INPUT" / "wind.nc") as ds:
        ty = ds["STRESS_Y"][:]
        assert ds["STRESS_X"][:].max() == 0.0
    assert ty.shape == (d["wind"].shape[0], 352, 3)
    np.testing.assert_allclose(ty[7, :, 1], d["wind"][7, 1:, 1])   # row j = north face of cell j


def test_coordinate_files_hold_the_bayiw_targets(tmp_path):
    from validation.bayiw_baroclinic import bayiw
    from validation.mom6_bayiw import inputs
    inputs.write_inputs("section", "2layer", "bay", tmp_path)
    with netCDF4.Dataset(tmp_path / "INPUT" / "hyb.nc") as ds:
        sig, dz = ds["sigma2"][:], ds["dz"][:]
    np.testing.assert_allclose(dz, bayiw.hybrid_dz_top(25))
    assert sig.size == 26 and 996.0 < sig.min() < sig.max() < 1000.0 and (np.diff(sig) > 0).all()
    with netCDF4.Dataset(tmp_path / "INPUT" / "layers.nc") as ds:
        lay = ds["Layer"][:]
    np.testing.assert_allclose(lay, 0.5 * (sig[1:] + sig[:-1]))
```

- [ ] **Step 2: Run, expect FAIL** (`validation.mom6_bayiw` missing). `python -m pytest tests/test_mom6_bayiw.py -q`.

- [ ] **Step 3: Implement `validation/mom6_bayiw/__init__.py` (empty) and `inputs.py`:**

```python
"""
MOM6 inputs for the BayIW benchmark (spec 2026-10-13-mom6-bayiw-benchmark-design §2): mosaic supergrid,
topography, z-level temperature (centres whose midpoints are the MITgcm interfaces, so MOM6's source edges are
exact), file wind stress on the C grid and the HYCOM1 / HYBGEN coordinate files with the bayiw targets.
"""
from __future__ import annotations

from pathlib import Path

import netCDF4
import numpy as np

from validation.bayiw_2d import section
from validation.bayiw_baroclinic import bayiw
from validation.bayiw_cut import cut

FILL = -1.0e20


def source_centres() -> np.ndarray:
    """Centres c_k [m, positive down] with (c_k + c_{k+1})/2 = the MITgcm interfaces (MOM6 builds source edges
    as midpoints, MOM_horizontal_regridding.F90:454-480)."""
    zf = -bayiw.Z_F
    c = np.empty(zf.size - 1)
    c[0] = 0.5 * zf[1]
    for k in range(c.size - 1):
        c[k + 1] = 2.0 * zf[k + 1] - c[k]
    return c


def domain(kind: str, case: str, bay: str) -> dict:
    if kind == "section":
        s = section.section_inputs(case, bay)
        rep = lambda a: np.repeat(a, section.NX_MIT, axis=-1)                      # noqa: E731
        H = rep(s["H"]).astype(float)
        H[:, [0, 2]] = np.nan                                                      # land walls as MITgcm
        return dict(dx=np.repeat(s["dx"], 3), dy=s["dy"], H=H, T=rep(s["T"]), wind=rep(s["wind"]),
                    x0=s["x0"] - float(s["dx"][0]), y0=bayiw.Y0)
    if kind == "cut":
        d = cut.cut_inputs(case, bay, "py")
        return dict(dx=d["dx"], dy=d["dy"], H=d["H"], T=d["T"], wind=d["wind"], x0=d["x0"], y0=d["y0"])
    raise ValueError(f"unknown kind {kind!r}")
```

followed by the writers (one function each, NetCDF4 classic, float64):
  - `_write_hgrid(path, dx, dy, x0, y0)`: supergrid `nxp = 2nx+1`, `nyp = 2ny+1`; `x`, `y` labels in km (`x0`/1000 + cumulative half cells), `dx[j, 2i] = dx[j, 2i+1] = dx_i/2` on every supergrid row, `dy` likewise, `area = (dx_i/2)(dy_j/2)`, `angle_dx = 0`; units `"m"`, `"m2"`.
  - `_write_topog(path, H)`: `depth(ny, nx)` = `nan_to_num(H)` positive down.
  - `_write_ts(path, T)`: `ptemp(Time, depth, lat, lon)` with `depth = source_centres()`, `_FillValue = FILL` (NaN → FILL below the bed and on land), `salt` = 35.0 everywhere with the same fill.
  - `_write_wind(path, wind)`: `STRESS_X` zeros and `STRESS_Y(Time, lat, lon) = wind[:, 1:, :]` (row j = north face of cell j), `Time` in days = (k + ½)·1200/86400.
  - `_write_coords(path_hyb, path_layers, case, T, H)`: targets from `bayiw.centred_targets` (two-layer) or `bayiw.profile_targets` (else), with `rho = bayiw.RHO0 − bayiw.T_COEF (T − t_ref)` exactly as `bayiw.build_model`; `hyb.nc` vars `sigma2(interfaces=26)`, `dz(layers=25) = bayiw.hybrid_dz_top(25)`; `layers.nc` var `Layer(layers=25)` = midpoints of the targets (MOM6 HYBGEN layer densities).
  - `write_inputs(kind, case, bay, out_dir)` calls all of them into `out_dir/INPUT`.

- [ ] **Step 4: Run** `python -m pytest tests/test_mom6_bayiw.py -q` → 4 passed.
- [ ] **Step 5: Commit** (`git add validation/mom6_bayiw/__init__.py validation/mom6_bayiw/inputs.py tests/test_mom6_bayiw.py`).

---

### Task 3: MOM6 namelists

**Files:**
- Create: `validation/mom6_bayiw/config.py`
- Test: `tests/test_mom6_bayiw.py`

**Interfaces:**
- Produces: `write_case(kind, case, bay, coord, out_dir, hours, dt=120.0, layout=(1, 5)) -> Path` writing `MOM_input`, `MOM_override` (empty), `diag_table`, `input.nml` and the `INPUT/` files (Task 2), `RESTART/`; `coord` ∈ `"zstar" | "hycom1" | "hybgen"`.

- [ ] **Step 1: Failing tests**

```python
def test_mom_input_section_has_walls_and_south_obc(tmp_path):
    from validation.mom6_bayiw import config
    d = config.write_case("section", "2layer", "bay", "hycom1", tmp_path, hours=60.0)
    txt = (d / "MOM_input").read_text()
    for s in ('REENTRANT_X = False', 'GRID_CONFIG = "mosaic"', 'ROTATION = "betaplane"', 'BETA = 0.0',
              'EQN_OF_STATE = "LINEAR"', 'DRHO_DS = 0.0', 'KE_SCHEME = "KE_ARAKAWA"', 'KH = 100.0',
              'KHTR = 100.0', 'KV = 1.0E-05', 'KD = 1.0E-05', 'BOTTOMDRAGLAW = False', 'NOSLIP = False',
              'REGRIDDING_COORDINATE_MODE = "HYCOM1"', 'ALE_COORDINATE_CONFIG = "HYBRID:hyb.nc,sigma2,dz"',
              'REMAPPING_SCHEME = "PPM_H4"', 'Z_INIT_ALE_REMAPPING = True', 'WIND_FILE_DAYS_PER_RECORD = -72',
              'OBC_NUMBER_OF_SEGMENTS = 1', 'OBC_SEGMENT_001 = "J=0,I=0:N,FLATHER,ORLANSKI"', 'DT = 120.0',
              'LAYOUT = 1, 5', 'NIGLOBAL = 3', 'NJGLOBAL = 352', 'NK = 25'):
        assert s in txt, s
    assert "clock_grain = 'ROUTINE'" in (d / "input.nml").read_text()


def test_mom_input_cut_has_three_open_edges_and_zstar_levels(tmp_path):
    from validation.bayiw_baroclinic import bayiw
    from validation.mom6_bayiw import config
    d = config.write_case("cut", "2layer", "bay", "zstar", tmp_path, hours=3.0)
    txt = (d / "MOM_input").read_text()
    assert 'OBC_NUMBER_OF_SEGMENTS = 3' in txt and 'NK = 50' in txt and 'NIGLOBAL = 198' in txt
    assert 'REGRIDDING_COORDINATE_MODE = "ZSTAR"' in txt
    assert "ALE_RESOLUTION = " + ", ".join(f"{x:.6g}" for x in bayiw.DELR) in txt
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `config.py` that builds the `MOM_input` text from these blocks (values from the Global Constraints facts and the spec §2): domain (`NIGLOBAL`, `NJGLOBAL`, `NK`, `REENTRANT_X = False`, `REENTRANT_Y = False`, `LAYOUT`, `IO_LAYOUT = 1, 1`, `GRID_CONFIG = "mosaic"`, `GRID_FILE = "ocean_hgrid.nc"`, `INPUTDIR = "INPUT"`, `TOPO_CONFIG = "file"`, `TOPO_FILE = "topog.nc"`, `MINIMUM_DEPTH = 1.0`, `MASKING_DEPTH = 0.0`, `MAXIMUM_DEPTH = 599.9`, `ROTATION = "betaplane"`, `F_0 = 6.97E-05`, `BETA = 0.0`); EOS (`EQN_OF_STATE = "LINEAR"`, `RHO_0 = 999.8`, `RHO_REF_LINEAR_EOS = 999.8`, `T_REF_LINEAR_EOS = t_ref`, `DRHO_DT = -0.19996`, `DRHO_DS = 0.0`); coordinate (ZSTAR: `ALE_COORDINATE_CONFIG = "PARAM"` + `ALE_RESOLUTION`; HYCOM1: the HYBRID line; HYBGEN: `REGRIDDING_COORDINATE_MODE = "HYBGEN"`, `COORD_CONFIG = "file"`, `COORD_FILE = "layers.nc"`, `HYBGEN_DEEP_DZ_PR0FILE = <hybrid_dz_top(25)>`, `HYBGEN_SHALLOW_DZ_PR0FILE = <same>`; all: `USE_REGRIDDING = True`, `REMAPPING_SCHEME = "PPM_H4"`); init (`INIT_LAYERS_FROM_Z_FILE = True`, `TEMP_SALT_Z_INIT_FILE = "ts_z.nc"`, `Z_INIT_FILE_PTEMP_VAR = "ptemp"`, `Z_INIT_FILE_SALT_VAR = "salt"`, `Z_INIT_ALE_REMAPPING = True`, `Z_INIT_REMAPPING_SCHEME = "PPM_H4"`, for HYCOM1/HYBGEN also `Z_INIT_REMAP_GENERAL = True`, `Z_INIT_REMAP_FULL_COLUMN = True`); physics (`KE_SCHEME`, `LAPLACIAN = True`, `KH = 100.0`, `BIHARMONIC = False`, `NOSLIP = False`, `KV`, `KD`, `KHTR`, `BOTTOMDRAGLAW = False`, `HBBL = 1.0`, `HMIX_FIXED = 1.0`, `KV_ML_INVZ2 = 0.0`, `DIRECT_STRESS = False`, `BUOY_CONFIG = "zero"`); forcing (`WIND_CONFIG = "file"`, `WIND_FILE = "wind.nc"`, `WIND_STAGGER = "C"`, `WIND_FILE_DAYS_PER_RECORD = -72`); time (`DT = dt`, `DT_THERM = dt`, `DTBT = -0.98`, `TIMEUNIT = 3600.0`, `DAYMAX = hours`, `ENERGYSAVEDAYS = 0.25`); OBC (section: 1 south segment; cut: south, west `"I=0,J=N:0,FLATHER,ORLANSKI"`, east `"I=N,J=0:N,FLATHER,ORLANSKI"`, each with `_DATA = "U=value:0.0,V=value:0.0,SSH=value:0.0"`). `diag_table`: `"prog", 900, "seconds"` with `SSH`, `temp`, `v`, `h` instantaneous. `input.nml`: `&MOM_input_nml output_directory='./', input_filename='n', restart_input_dir='INPUT/', restart_output_dir='RESTART/', parameter_filename='MOM_input','MOM_override' /` and `&fms_nml clock_grain = 'ROUTINE' /`. `t_ref` = `bayiw.CASES[case]["t_ref"]`.
- [ ] **Step 4: Run** the two tests → pass; then one real **1-hour section run** (`hours=1`, hycom1) with `run_mom6.sh run <dir>` — Expected: "Total runtime"; if MOM6 stops with a FATAL about a parameter, fix the generator (ledger each fix as a ruling with the MOM6 file:line).
- [ ] **Step 5: Commit** (`git add validation/mom6_bayiw/config.py tests/test_mom6_bayiw.py`).

---

### Task 4: MOM6 output → section series

**Files:**
- Create: `validation/mom6_bayiw/read_mom6.py`
- Test: `tests/test_mom6_bayiw.py`

**Interfaces:**
- Produces: `section_series(run_dir, case, bay) -> dict` with the keys of `validation/bayiw_2d/run_pyocean.record` (`t_h, eta, T110, T250, V110, T_sec, V_sec, n_trunc, heat`) plus `rpe` (NaN, 2) and `meta_json`; `save_series(series, case, bay, coord)` → `section.RUNS/"pyocean"/f"{case}_{bay}_mom6{coord}"/series.npz` so `bayiw_2d/compare.report(case, "mom6" + coord, "orl")` works unchanged.

- [ ] **Step 1: Failing test** — a synthetic `prog.nc` (Time 3, zl 4, yh 352, xh 3, yq 353) with `h` = 4 layers of the section depth/4, `temp` known linear, `v` zeros, `SSH` known; `section_series` must return `T_sec` equal to `section.to_mitgcm_levels(h, temp, H)` on column 1, `eta` = SSH column 1, and the keys listed above.

```python
def test_series_from_mom6_matches_the_pyocean_record_layout(tmp_path):
    from validation.bayiw_2d import section
    from validation.mom6_bayiw import read_mom6
    H = np.nan_to_num(section.section_inputs("2layer", "bay")["H"][:, 0])
    nt, nk, ny = 3, 4, 352
    h = np.repeat((H / nk)[None, None, :, None], nk, 1).repeat(nt, 0).repeat(3, 3)
    temp = 15.0 + np.arange(nk)[None, :, None, None] * np.ones((nt, nk, ny, 3))        # surface first, as MOM6
    with netCDF4.Dataset(tmp_path / "prog.nc", "w") as ds:
        for n, s in (("Time", nt), ("zl", nk), ("yh", ny), ("xh", 3), ("yq", ny + 1)):
            ds.createDimension(n, s)
        ds.createVariable("Time", "f8", ("Time",))[:] = np.array([900.0, 1800.0, 2700.0]) / 86400.0
        ds["Time"].units = "days since 0001-01-01 00:00:00"
        ds.createVariable("h", "f8", ("Time", "zl", "yh", "xh"))[:] = h
        ds.createVariable("temp", "f8", ("Time", "zl", "yh", "xh"))[:] = temp
        ds.createVariable("v", "f8", ("Time", "zl", "yq", "xh"))[:] = 0.0
        ds.createVariable("SSH", "f8", ("Time", "yh", "xh"))[:] = 0.01
    s = read_mom6.section_series(tmp_path, "2layer", "bay")
    for k in ("t_h", "eta", "T110", "T250", "V110", "T_sec", "V_sec", "n_trunc", "heat", "rpe"):
        assert k in s, k
    np.testing.assert_allclose(s["t_h"], [0.25, 0.5, 0.75])
    exp = section.to_mitgcm_levels(h[0, ::-1, :, 1:2], temp[0, ::-1, :, 1:2], H[:, None])[::-1, :, 0]
    np.testing.assert_allclose(s["T_sec"][0], exp)
```

- [ ] **Step 2: Run, expect FAIL.**
- [ ] **Step 3: Implement** `section_series` reading `prog.nc` (netCDF4), flipping MOM6's surface-first layers to bed first, and building each record exactly as `run_pyocean.record` does (column 1; v-face depths `Hf` as there; `compare.mit_at_depth` for T110/T250/V110; `heat` = Σ h T × area; `n_trunc` = 0); `t_h` from `Time` in days × 24.
- [ ] **Step 4: Run** → pass. **Step 5: Commit.**

---

### Task 5: Section results runs and comparison

**Files:**
- Modify: `validation/mom6_bayiw/run_mom6.sh` (add `section <case> <bay> <coord>`: `config.write_case` into `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_bayiw/section/<case>_<bay>_<coord>`, run, `read_mom6.save_series`)
- Create: `validation/mom6_bayiw/README.md`

- [ ] **Step 1: Runs** (one after another; ~1–2 min each): `for c in 2layer real; do for b in bay nobay; do for z in zstar hycom1; do bash validation/mom6_bayiw/run_mom6.sh section $c $b $z; done; done; done` and `for b in bay nobay; do bash validation/mom6_bayiw/run_mom6.sh section 2layer $b hybgen; done`. Expected: 10 "Total runtime"; any run > 6 min is reported before going on.
- [ ] **Step 2: Compare** `python validation/bayiw_2d/compare.py <case> mom6<coord> orl` for each, saved to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/mom6_bayiw/compare_<case>_<coord>.txt`; and ζ (heat-based interface rise) for the two-layer bay case with the scratch-free function `validation/bayiw_cut/compare.heat_displacement` applied to `T_sec` (levels 20–40), MOM6 vs MITgcm vs pyOCEAN hybrid, at Seiche, Seiche2, Omega, j = 300, C.
- [ ] **Step 3: README (Spanish)** — configuration, the wind half-record offset (MOM6 piecewise records vs MITgcm interpolated), tables side by side (MITgcm reference; pyOCEAN z\*/hybrid; MOM6 z\*/HYCOM1/HYBGEN), the (a)–(d) classification and the two-layer reading rule of the spec §3. If MOM6 HYCOM1/HYBGEN resemble MITgcm where pyOCEAN does not, stop and report (class a candidate).
- [ ] **Step 4: Commit** (`run_mom6.sh`, `README.md`).

---

### Task 6: Cost on the cut (MOM6)

**Files:**
- Modify: `validation/mom6_bayiw/run_mom6.sh` (add `cost <coord> [dt]`), `validation/mom6_bayiw/read_mom6.py` (add `clock_table(stdout) -> dict[name, seconds]`)
- Test: `tests/test_mom6_bayiw.py` (parse a short synthetic "Tabulating mpp_clock statistics" block)

- [ ] **Step 1: Failing test** for `clock_table` with a 4-line synthetic block (`tmin tmax tavg tstd tfrac grain pemin pemax` columns, as FMS `mpp_comm_mpi.inc:212`) → dict of `tavg`.
- [ ] **Step 2: Implement; test passes.**
- [ ] **Step 3: Runs** — two-layer cut with bay, 3 h, Δt 120 s, for `zstar`, `hycom1`, `hybgen`; then the largest stable Δt for `hycom1` among 240, 360, 600 s (1 h each, stop at the first instability). Expected ≤ ~10 min total.
- [ ] **Step 4: Table in the README** — s per simulated hour for MOM6 (each coord, Δt 120 and the largest stable), pyOCEAN (`profile_final.txt`: 0.304 z\*, 0.160 hybrid s/step × 30) and MITgcm (4.4 s); the MOM6 clock breakdown next to pyOCEAN's `profile_step` breakdown, flagging pieces where pyOCEAN is > 2× MOM6. Note that FMS deps are built at `-O0` (I/O and halo exchanges).
- [ ] **Step 5: Commit.**

---

### Task 7: HYCOM cost (behind the gate)

**Files:**
- Create: `validation/hycom_bayiw/{__init__.py, inputs.py, blkdat.py, build_hycom.sh, run_hycom.sh, README.md}`
- Test: `tests/test_hycom_bayiw.py`

- [ ] **Step 1: Failing tests** — `.a/.b` writer round trip with `forcing.sources.hycom_ab` (a 7 × 5 field with land NaN → 2^100 data void, padding to 4096 words, `.b` min/max lines), and `blkdat.py` output containing `kdm`, 25 `sigma` lines increasing, `dp0k` = `bayiw.hybrid_dz_top(25)`, `baclin` dividing 21600, `cb = 0`, `lbflag = 0`, `relax = 1`.
- [ ] **Step 2: Implement** the writer and `blkdat.py` (`iversn = 23`, field order of `blkdat.F90`; the operational template `/Users/estebancruz/HYCOM/blkdat.input` as the line-format reference), inputs for the cut (`regional.grid` with constant `cori`, last row/column land, `regional.depth`, `relax.{temp,saln,intf,rmu}` with 12 identical months, `forcing.{tauewd,taunwd,radflx,shwflx,...}` with `span = 0`, wind day ≥ 1462, `patch.input` 5 × 1 tiles), all in Python; tests pass.
- [ ] **Step 3: Gate** (`build_hycom.sh`): copy `HYCOM-src` to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/hycom_bayiw/build/src`, copy `config/generic-gnu-relo_mpi` without `-mavx`, `make hycom ARCH=generic-gnu-relo TYPE=mpi CPP_EXTRAS="-DEOS_SIG0 -DEOS_7T -DMASSLESS_1MM"`; trivial run: resting basin, 10 steps, `mpiexec -n 5`. **If this is not working after ~1 h of work, stop, ledger the blocker and report to Esteban; skip Step 4.**
- [ ] **Step 4: Cost run** — the two-layer cut with bay, 3 h, 5 ranks; s per simulated hour and the `xctmrp` timer table into the README cost table.
- [ ] **Step 5: Commit** (explicit paths).

---

### Task 8: Optional 3-D run, docs

- [ ] **Step 1: Decide** from Task 5: if MOM6 HYCOM1/HYBGEN and pyOCEAN agree in the two-layer section (both close to MITgcm, or both off the same way), skip the 3-D run and ledger why. Otherwise: MOM6 HYCOM1 on the cut, two-layer with bay, 30 h (`run_mom6.sh cost hycom1` with `hours=30` and the `diag_table` station output), ζ at A, B, Omega vs MITgcm and pyOCEAN.
- [ ] **Step 2: Docs** — `validation/mom6_bayiw/README.md` final tables and "Qué debe mejorar pyOCEAN"; scorecard rows in `docs/design/vision_roadmap.md` §7 (MOM6 section results; cost vs MOM6/HYCOM/MITgcm); `CLAUDE.md` test count; memory `project_status.md`.
- [ ] **Step 3: Full suite** `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider` → all pass; commit the docs.
