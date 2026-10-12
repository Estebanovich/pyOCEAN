# BayIW 2-D Section (pyOCEAN vs MITgcm) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the BayIW bay-axis section (column i = 280: 1 × 352 × 50) in MITgcm and in pyOCEAN (z\* and hybrid) for the two-layer and linear cases, with and without the bay, for 60 h, and compare them with the spec §5 metrics.

**Architecture:**
- `validation/bayiw_2d/section.py` extracts the column from the MITgcm binaries (through `validation/bayiw_baroclinic/bayiw.py`), writes the MITgcm inputs and namelists, reads the MITgcm MNC output, and builds the pyOCEAN section model.
- `run_mitgcm.sh` builds MITgcm once with sNx = 1, sNy = 352 and runs a case.
- `run_pyocean.py` runs pyOCEAN.
- `compare.py` reuses the metric functions of `validation/bayiw_baroclinic/compare.py`.

**Tech Stack:** Python 3.12, NumPy, netCDF4, pytest; MITgcm (gfortran, optfile `darwin_arm64_gfortran`, no MPI).

**Spec:** `docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md` (configuration and metrics from `2026-10-12-bayiw-baroclinic-benchmark-design.md` §3, §6)

## Global Constraints

- **Cores:** ≤ 5 P-cores (`NUMBA_NUM_THREADS=5`); MITgcm uses 1 process. One heavy job at a time.
- **Outputs:** large outputs go to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2d/`. Never commit `*.npz`, `log_*.txt` or `pyOCEAN-literature-review-guide.docx`.
- **MITgcm source:** no changes (spec §3).
- **Commits:** one per task (Esteban: commits per phase). Messages end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn
  ```
- **Benchmark rules:** classify each difference (a)–(d); never tune; stop and report to Esteban on a class (a) failure.
- **MITgcm OBCS:** reproduce Esteban's 3-D settings: Orlanski CMAX 0.45, `cvelTimeScale` 250, 30-cell velocity sponge (τ 60 → 1200 s), `useOBCSbalance=.TRUE.` with `OBCS_balanceFacS = -1` (zero net transport through the south boundary). pyOCEAN uses Flather there. Classify that difference as (c).
- **Duration:** 60 h (stage 1 of the MITgcm runs: startTime 0, endTime 216 000 s, periodic wind).

## Review Focus

1. **Byte order and record layout of the MITgcm inputs** (big-endian float64; T as (Nr, Ny, Nx), wind as (nrec, Ny, Nx) with Nx = 1): a silent transpose would give a wrong but running model. *Pinned by Task 1 `test_mitgcm_inputs_round_trip_to_the_column`.*
2. **The section column matches station positions:** j indices of Seiche (345), Seiche2 (325), Omega (315), C (265), Gamma (215), with H in the bay 129–164 m. *Pinned by Task 1 `test_section_column_is_i280_with_the_bay`.*
3. **Reading MNC output with Nx = 1** (Xp1 = 2 u-faces; time variable `T`): U at the single interior face must not be misread. *Pinned by Task 2 `test_read_mitgcm_section_shapes` (skips until a run exists).*
4. **pyOCEAN with nx = 1:** west/east edges closed and no Coriolis corner issues; one step finite. *Pinned by Task 3 `test_section_model_steps`.*
5. **Time alignment:** MITgcm MNC dumps at t = 0, 900, … s; pyOCEAN at the same marks with Δt = 30 s. *Pinned by Task 4 `compare` using `np.interp` on hours and by the check in Task 5 Step 2.*

---

### Task 1: Section inputs and MITgcm namelists

**Files:**
- Create: `validation/bayiw_2d/section.py`
- Test: `tests/test_bayiw_2d.py`

**Interfaces:**
- Consumes: `validation.bayiw_baroclinic.bayiw` (`load_inputs`, `CASES`, `DELR`, `X0`, `Y0`, `F0`, `RHO0`, `T_COEF`, `STATIONS`, `RUNS`).
- Produces:
  - `I_SEC = 280`;
  - `section_inputs(case, bay) -> dict(dx (1,), dy (352,), H (352, 1), T (50, 352, 1), wind (180, 352, 1), x0 float)`;
  - `write_mitgcm_inputs(case, bay, run_dir: Path) -> None`, which writes `input/` binaries and the namelists `data`, `data.obcs`, `data.pkg`, `data.mnc`, `eedata`;
  - `SECTION_STATIONS` {name: j}.

- [ ] **Step 1: Write the failing tests** (`tests/test_bayiw_2d.py`)

```python
"""Tests of the BayIW 2-D section tools (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.bayiw_baroclinic import bayiw

HAVE = (bayiw.CASES["2layer"]["dir"] / "input").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="BayIW MITgcm inputs not mounted")


def test_section_column_is_i280_with_the_bay():
    from validation.bayiw_2d import section
    d = section.section_inputs("2layer", "bay")
    assert d["dx"].shape == (1,) and d["dy"].shape == (352,) and d["H"].shape == (352, 1)
    assert 120.0 < d["H"][345, 0] < 170.0 and abs(d["H"][200, 0] - 599.9) < 1e-9
    assert d["T"].shape == (50, 352, 1) and d["wind"].shape == (180, 352, 1)
    assert section.SECTION_STATIONS == {"Gamma": 215, "C": 265, "Omega": 315, "Seiche2": 325, "Seiche": 345}
    full = bayiw.load_inputs("2layer", "bay")
    np.testing.assert_array_equal(d["T"][:, :, 0], full[3][:, :, 280])


def test_mitgcm_inputs_round_trip_to_the_column(tmp_path):
    # review focus 1: big-endian float64, (Nr, Ny, Nx) and (nrec, Ny, Nx) with Nx = 1
    from validation.bayiw_2d import section
    section.write_mitgcm_inputs("2layer", "bay", tmp_path)
    d = section.section_inputs("2layer", "bay")
    T = np.fromfile(tmp_path / "input" / "theta.bin", ">f8").reshape(50, 352, 1)
    W = np.fromfile(tmp_path / "input" / "wind.bin", ">f8").reshape(180, 352, 1)
    B = np.fromfile(tmp_path / "input" / "bathy.bin", ">f8").reshape(352, 1)
    np.testing.assert_array_equal(T, d["T"])
    np.testing.assert_array_equal(W, d["wind"])
    np.testing.assert_array_equal(B, np.where(np.isfinite(d["H"]), -d["H"], 0.0))
    data = (tmp_path / "data").read_text()
    assert "endTime=216000." in data and "theta.bin" in data and "wind.bin" in data
    obcs = (tmp_path / "data.obcs").read_text()
    assert "OB_Jsouth = 1*1" in obcs and "OB_Iwest" not in obcs and "OBCS_balanceFacS=-1." in obcs
```

- [ ] **Step 2: Run them**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_2d.py -q`
Expected: FAIL with `ModuleNotFoundError: No module named 'validation.bayiw_2d'`.

- [ ] **Step 3: Implement `validation/bayiw_2d/section.py`**

```python
"""
BayIW 2-D section along the bay axis (column i = 280 of the MITgcm grid) for pyOCEAN and MITgcm (spec
docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md): inputs from Esteban's MITgcm binaries
(validation/bayiw_baroclinic/bayiw.py), MITgcm namelists of stage 1 (0–60 h, periodic wind) with only the
south boundary open, the MITgcm MNC reader and the pyOCEAN section model.
"""
from __future__ import annotations

import re
from pathlib import Path

import numpy as np

from validation.bayiw_baroclinic import bayiw

I_SEC = 280
SECTION_STATIONS = {"Gamma": 215, "C": 265, "Omega": 315, "Seiche2": 325, "Seiche": 345}
RUNS = bayiw.RUNS / "bayiw_2d"


def section_inputs(case: str, bay: str) -> dict:
    """
    The column i = 280: dx (1,), dy (352,), H (352, 1) NaN on land, T (50, 352, 1), wind (nrec, 352, 1), and
    x0, the x of its west edge [m] (xgOrigin + Σ dx[:280]).
    """
    dx, dy, H, T, wind = bayiw.load_inputs(case, bay)
    s = slice(I_SEC, I_SEC + 1)
    return dict(dx=dx[s].copy(), dy=dy.copy(), H=H[:, s].copy(), T=T[:, :, s].copy(), wind=wind[:, :, s].copy(),
                x0=bayiw.X0 + float(dx[:I_SEC].sum()))


def _write(path: Path, a: np.ndarray) -> None:
    np.ascontiguousarray(a, ">f8").tofile(path)


def write_mitgcm_inputs(case: str, bay: str, run_dir: Path) -> None:
    """
    MITgcm inputs and namelists of the section in run_dir: input/{dx,dy,bathy,theta,wind}.bin (big-endian
    float64) and data (from the case's run_expand/data.stage1 with the section files and endTime 216 000 s),
    data.obcs (south only: Orlanski CMAX 0.45, 30-cell velocity sponge, balanceFacS = −1 as Esteban's runs),
    data.pkg (OBCS, MNC; no diagnostics), data.mnc, eedata.
    """
    run_dir = Path(run_dir)
    (run_dir / "input").mkdir(parents=True, exist_ok=True)
    d = section_inputs(case, bay)
    _write(run_dir / "input" / "dx.bin", d["dx"])
    _write(run_dir / "input" / "dy.bin", d["dy"])
    _write(run_dir / "input" / "bathy.bin", np.where(np.isfinite(d["H"]), -d["H"], 0.0))
    _write(run_dir / "input" / "theta.bin", d["T"])
    _write(run_dir / "input" / "wind.bin", d["wind"])
    src = (bayiw.CASES[case]["dir"] / "run_expand" / "data.stage1").read_text()
    src = re.sub(r"delYFile *=.*", "delYFile = 'input/dy.bin',", src)
    src = re.sub(r"delXFile *=.*", "delXFile = 'input/dx.bin',", src)
    src = re.sub(r"bathyFile *=.*", "bathyFile='input/bathy.bin',", src)
    src = re.sub(r"meridWindFile *=.*", "meridWindFile='input/wind.bin',", src)
    src = re.sub(r"hydrogThetaFile *=.*", "hydrogThetaFile='input/theta.bin',", src)
    src = re.sub(r"endTime *=.*", "endTime=216000.,", src)
    src = re.sub(r"pChkptFreq *=.*", "pChkptFreq=0.,", src)
    (run_dir / "data").write_text(src)
    (run_dir / "data.obcs").write_text(
        " &OBCS_PARM01\n OB_Jsouth = 1*1,\n useOrlanskiSouth=.TRUE.,\n useOBCSsponge=.TRUE.,\n"
        " useLinearSponge=.FALSE.,\n OBCSsponge_Salt=.FALSE.,\n OBCSsponge_Theta=.FALSE.,\n useOBCSbalance=.TRUE.,\n"
        " useOBCSprescribe=.FALSE.,\n OBCS_balanceFacS=-1.,\n &\n &OBCS_PARM02\n cvelTimeScale=250.0,\n CMAX=0.45,\n &\n"
        " &OBCS_PARM03\n spongeThickness=30,\n Urelaxobcsinner=1200.E0,\n Vrelaxobcsinner=1200.E0,\n"
        " Urelaxobcsbound=60.E0,\n Vrelaxobcsbound=60.E0,\n &\n")
    (run_dir / "data.pkg").write_text(" &PACKAGES\n useMNC=.TRUE.,\n useOBCS=.TRUE.,\n &\n")
    (run_dir / "data.mnc").write_text(" &MNC_01\n mnc_use_outdir=.TRUE.,\n mnc_outdir_str='mnc_',\n monitor_mnc=.FALSE.,\n"
                                      " pickup_read_mnc=.FALSE.,\n pickup_write_mnc=.FALSE.,\n &\n")
    (run_dir / "eedata").write_text(" &EEPARMS\n nTx=1,\n nTy=1,\n &\n")
```

Also create an empty `validation/bayiw_2d/__init__.py`? No: the repo uses namespace packages (`validation.bayiw_baroclinic` has no `__init__.py`). Keep that.

- [ ] **Step 4: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_2d.py -q`
Expected: 2 passed.

If `data.stage1` of the linear case does not exist, list `run_expand/` of the linear directory, use the stage-1 file found there, and record a ruling.

- [ ] **Step 5: Commit**

```bash
git add validation/bayiw_2d/section.py tests/test_bayiw_2d.py
git commit -m "BayIW 2-D section: column inputs and MITgcm namelists (task 1)"
```

---

### Task 2: Build and run MITgcm for the section; MNC reader

**Files:**
- Create: `validation/bayiw_2d/run_mitgcm.sh`
- Modify: `validation/bayiw_2d/section.py` (add `read_mitgcm`)
- Test: `tests/test_bayiw_2d.py` (append)

**Interfaces:**
- Produces:
  - `run_mitgcm.sh build` (once) and `run_mitgcm.sh run <case> <bay>`;
  - `read_mitgcm(case, bay) -> dict(t_h (n,), eta (n, 352), T (n, 50, 352), U (n, 50, 352), V (n, 50, 353), zc (50,))`, sampled at x-index 0 (U at the single interior u-face is zero in a 1-cell channel; V is the along-axis velocity).

- [ ] **Step 1: Write the script**

```bash
#!/usr/bin/env bash
# BayIW 2-D section in MITgcm (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md §3).
#   run_mitgcm.sh build              : compile once (sNx = 1, sNy = 352, Nr = 50, one process)
#   run_mitgcm.sh run <case> <bay>   : write inputs (section.py) and run 60 h in RUNS/mitgcm/<case>_<bay>/
set -euo pipefail
ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
MIT=/Volumes/ESTEBAN_MIT/MITgcm
BASE=/Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2d/mitgcm
CODE_SRC=$MIT/verification/BayIW_Rectan_2layer/code
if [[ "$1" == "build" ]]; then
  mkdir -p "$BASE/code" "$BASE/build"
  cp "$CODE_SRC"/{packages.conf,OBCS_OPTIONS.h,DIAGNOSTICS_SIZE.h,MNC_SIZE.h} "$BASE/code/"
  sed -e 's/sNx = *[0-9]*,/sNx =   1,/' -e 's/sNy = *[0-9]*,/sNy = 352,/' \
      -e 's/nPx = *[0-9]*,/nPx =   1,/' -e 's/nPy = *[0-9]*,/nPy =   1,/' "$CODE_SRC/SIZE.h" > "$BASE/code/SIZE.h"
  grep -E "sNx|sNy|nPx|nPy|Nr  =" "$BASE/code/SIZE.h"
  cd "$BASE/build"
  "$MIT/tools/genmake2" -rootdir "$MIT" -mods ../code -optfile "$MIT/tools/build_options/darwin_arm64_gfortran" > genmake.log 2>&1
  make depend > make_depend.log 2>&1
  make -j 5 > make.log 2>&1
  ls -la mitgcmuv
elif [[ "$1" == "run" ]]; then
  RUN="$BASE/$2_$3"
  rm -rf "$RUN"; mkdir -p "$RUN"
  cd "$ROOT" && python -c "from pathlib import Path; from validation.bayiw_2d import section; section.write_mitgcm_inputs('$2', '$3', Path('$RUN'))"
  cd "$RUN"; cp "$BASE/build/mitgcmuv" .
  ./mitgcmuv > output.txt 2>&1
  grep -q "PROGRAM MAIN: Execution ended Normally" STDOUT.0000 && echo "OK $RUN"
fi
```

Make it executable: `chmod +x validation/bayiw_2d/run_mitgcm.sh`.

- [ ] **Step 2: Build**

Run: `bash validation/bayiw_2d/run_mitgcm.sh build`
Expected: `mitgcmuv` listed.

If the build fails:
- read `genmake.log` and `make.log`;
- try the optfile `darwin_arm64_gfortran_esteban`, and record a ruling;
- if `sNx = 1` with `OLx = 3` is rejected, set `OLx = 2` (as `verification/internal_wave` uses `OLy = 2` with `sNy = 1`), and record a ruling.

- [ ] **Step 3: Run the two-layer case with the bay**

Run: `bash validation/bayiw_2d/run_mitgcm.sh run 2layer bay`
Expected: `OK …/2layer_bay`, finishing in under a few minutes.

Then: `ls /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2d/mitgcm/2layer_bay/mnc_*/ | head`
Expected: `state.0000000000.t001.nc` and `grid.t001.nc`.

- [ ] **Step 4: Write the reader test** (append to `tests/test_bayiw_2d.py`)

```python
def test_read_mitgcm_section_shapes():
    # review focus 3: Nx = 1 (Xp1 = 2), time variable T, 241 dumps over 60 h
    from validation.bayiw_2d import section
    run = section.RUNS / "mitgcm" / "2layer_bay"
    if not run.exists():
        pytest.skip("MITgcm section run not done yet")
    d = section.read_mitgcm("2layer", "bay")
    n = d["t_h"].size
    assert abs(d["t_h"][-1] - 60.0) < 1e-6 and n == 241
    assert d["eta"].shape == (n, 352) and d["T"].shape == (n, 50, 352) and d["V"].shape == (n, 50, 353)
    np.testing.assert_allclose(d["T"][0, 0, 200], 22.767, atol=1e-3)
    assert np.isfinite(d["eta"]).all()
```

Run it: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_2d.py -q -k read_mitgcm`
Expected: FAIL with `AttributeError: module 'validation.bayiw_2d.section' has no attribute 'read_mitgcm'`.

- [ ] **Step 5: Implement `read_mitgcm`** (append to `section.py`)

```python
def read_mitgcm(case: str, bay: str) -> dict:
    """
    MITgcm MNC state dumps of a section run (one tile, t001): time [h], η (n, 352), T (n, 50, 352),
    U at the west face (n, 50, 352), V (n, 50, 353), level centres zc (50,) [m].
    """
    import glob

    import netCDF4
    run = RUNS / "mitgcm" / f"{case}_{bay}"
    files = sorted(glob.glob(str(run / "mnc_*" / "state.*.t001.nc")))
    if not files:
        raise FileNotFoundError(f"no MNC state files in {run}")
    t, eta, T, U, V = [], [], [], [], []
    for f in files:
        with netCDF4.Dataset(f) as ds:
            t.append(np.asarray(ds["T"][:], float))
            eta.append(np.asarray(ds["Eta"][:, :, 0], float))
            T.append(np.asarray(ds["Temp"][:, :, :, 0], float))
            U.append(np.asarray(ds["U"][:, :, :, 0], float))
            V.append(np.asarray(ds["V"][:, :, :, 0], float))
    order = np.argsort(np.concatenate(t))
    cat = lambda xs: np.concatenate(xs)[order]                                     # noqa: E731
    return dict(t_h=cat(t) / 3600.0, eta=cat(eta), T=cat(T), U=cat(U), V=cat(V), zc=bayiw.Z_C.copy())
```

- [ ] **Step 6: Run the test**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_2d.py -q`
Expected: 3 passed.

If the variable names or dimensions differ (for example `Eta` with a Z dimension), print `ncdump -h` of one state file, adapt the reader, and record a ruling.

- [ ] **Step 7: Run the other three MITgcm cases**

Run:
```bash
for c in 2layer linear; do for b in bay nobay; do bash validation/bayiw_2d/run_mitgcm.sh run $c $b; done; done
```
Expected: four `OK` lines. The two-layer bay case is re-run; that is harmless.

- [ ] **Step 8: Commit**

```bash
chmod +x validation/bayiw_2d/run_mitgcm.sh
git add validation/bayiw_2d/run_mitgcm.sh validation/bayiw_2d/section.py tests/test_bayiw_2d.py
git commit -m "BayIW 2-D section: MITgcm build/run script and MNC reader (task 2)"
```

---

### Task 3: pyOCEAN section model and runner

**Files:**
- Modify: `validation/bayiw_2d/section.py` (add `build_model`)
- Create: `validation/bayiw_2d/run_pyocean.py`
- Test: `tests/test_bayiw_2d.py` (append)

**Interfaces:**
- Consumes:
  - `section_inputs`;
  - `bayiw.{mitgcm_profile_at, centred_targets, profile_targets, N_LIGHT, DELR, RHO0, T_COEF, F0, X0, Y0, CASES, sample_at_depth, Z110, Z250, Z_C}`;
  - `validation.bayiw_seiche.run_pyocean.wind_at`.
- Produces:
  - `build_model(case, bay, coord="zstar", dt=30.0) -> (m, st, wind)`, with wind of shape (nrec, 352, 1);
  - `run_pyocean.py <case> <bay> --coord zstar|hybrid [--dt 30] [--hours 60]`, which writes `RUNS/pyocean/<case>_<bay>_<coord>/series.npz` with keys `t_h`, `eta` (n, 352), `T110`, `T250` (n, 352), `V110` (n, 352+1 faces), `T_sec` (n, 50, 352) on `Z_C`, `n_trunc`, `heat`, `rpe`.

- [ ] **Step 1: Write the failing test**

```python
def test_section_model_steps():
    # review focus 4: nx = 1 with closed west/east edges and open south
    from validation.bayiw_2d import section
    for coord in ("zstar", "hybrid"):
        m, st, wind = section.build_model("2layer", "bay", coord)
        assert st.h.shape[1:] == (352, 1)
        tau = np.zeros((353, 1))
        tau[:-1, :] = wind[3]
        for _ in range(3):
            st = m.step(st, tau_y=tau)
        assert np.isfinite(st.u).all() and np.isfinite(st.v).all() and st.n_trunc == 0
        np.testing.assert_allclose(st.h.sum(0), np.nan_to_num(m.H) + st.eta, rtol=1e-12, atol=1e-12)
```

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_2d.py -q -k section_model`
Expected: FAIL with `AttributeError: … has no attribute 'build_model'`.

- [ ] **Step 2: Implement `build_model`** (append to `section.py`)

```python
def build_model(case: str, bay: str, coord: str = "zstar", dt: float = 30.0):
    """
    pyOCEAN ALE model of the section with the MITgcm parameters (spec 3-D §3): z* on the 50 delR levels or
    hybrid with 25 layers (10 light targets); south open (Flather + Orlanski 0.45 + 30-cell velocity sponge),
    no mixing diagnostic.  Returns (model, initial state, wind records (nrec, 352, 1)).
    """
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid_stretched
    d = section_inputs(case, bay)
    hg = create_cartesian_grid_stretched(d["dx"], d["dy"], x0=d["x0"], y0=bayiw.Y0, f0=bayiw.F0)
    t_ref = bayiw.CASES[case]["t_ref"]
    if coord == "zstar":
        nz, params = bayiw.DELR.size, dict(dz_top=bayiw.DELR.copy())
    else:
        nz = 25
        rho = bayiw.RHO0 - bayiw.T_COEF * (d["T"][:, np.isfinite(d["H"])] - t_ref)
        tgt = (bayiw.centred_targets(float(rho.min()), float(rho.max()), nz, bayiw.N_LIGHT) if case == "2layer"
               else bayiw.profile_targets(rho, nz, bayiw.N_LIGHT))
        params = dict(dz_top=bayiw.DELR[:nz].copy(), rho_target=tgt)
    m = SigmaBaroclinic(hg, d["H"], dt=dt, n_layers=nz, rho0=bayiw.RHO0, t_coef=bayiw.T_COEF, t_ref=t_ref, z0=None,
                        closure="constant", nu_const=1e-5, ah=100.0, kh=100.0, kv=1e-5, pgf="fv", rho_ref="auto",
                        vertical_coordinate="ale", regrid=coord, regrid_params=params, open_boundaries=("south",),
                        orlanski_cmax=0.45, sponge_cells=30, sponge_tau=(1200.0, 60.0), sponge_tracer=False,
                        mix_diagnostic=False)
    z = m.layer_heights(np.zeros(d["H"].shape))
    st = m.initial_state(np.zeros(d["H"].shape), bayiw.mitgcm_profile_at(z, d["T"]))
    return m, st, d["wind"]
```

- [ ] **Step 3: Run the test**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_bayiw_2d.py -q`
Expected: 4 passed.

- [ ] **Step 4: Write `run_pyocean.py`**

```python
#!/usr/bin/env python
"""
BayIW 2-D section in pyOCEAN (spec docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md §4): every
900 s η, T at 110/250 m and V at 110 m along the section, and the T section on the MITgcm level centres.

Usage: NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py 2layer bay --coord zstar [--dt 30] [--hours 60]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from core.dynamics.sigma_baroclinic import set_threads  # noqa: E402
from validation.bayiw_2d import section  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_seiche.run_pyocean import wind_at  # noqa: E402


def record(m, st, buf: dict) -> None:
    zc = m.layer_heights(st.eta, st.h)[:, :, 0]
    h, T, bed = st.h[:, :, 0], st.T[:, :, 0], -np.asarray(m.H, float)[:, 0]
    buf["eta"].append(st.eta[:, 0].copy())
    buf["T110"].append(bayiw.sample_at_depth(zc, T, h, bayiw.Z110, bed))
    buf["T250"].append(bayiw.sample_at_depth(zc, T, h, bayiw.Z250, bed))
    zf = np.concatenate([zc[:, :1], 0.5 * (zc[:, 1:] + zc[:, :-1]), zc[:, -1:]], axis=1)      # v-faces
    hf = np.concatenate([h[:, :1], 0.5 * (h[:, 1:] + h[:, :-1]), h[:, -1:]], axis=1)
    bedf = np.concatenate([bed[:1], np.maximum(bed[1:], bed[:-1]), bed[-1:]])
    buf["V110"].append(bayiw.sample_at_depth(zf, st.v[:, :, 0], hf, bayiw.Z110, bedf))
    buf["T_sec"].append(np.array([bayiw.sample_at_depth(zc, T, h, z0, bed) for z0 in bayiw.Z_C]))
    buf["n_trunc"].append(st.n_trunc)
    buf["heat"].append(m.tracer_content(st))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("case", choices=tuple(bayiw.CASES))
    ap.add_argument("bay", choices=("bay", "nobay"))
    ap.add_argument("--coord", default="zstar", choices=("zstar", "hybrid"))
    ap.add_argument("--dt", type=float, default=30.0)
    ap.add_argument("--hours", type=float, default=60.0)
    a = ap.parse_args()
    out = section.RUNS / "pyocean" / f"{a.case}_{a.bay}_{a.coord}"
    out.mkdir(parents=True, exist_ok=True)
    set_threads(5)
    m, st, wind = section.build_model(a.case, a.bay, a.coord, a.dt)
    buf = {k: [] for k in ("t_h", "eta", "T110", "T250", "V110", "T_sec", "n_trunc", "heat")}
    buf["t_h"].append(0.0)
    record(m, st, buf)
    rpe = [m.rpe(st)]
    tau = np.zeros((353, 1))
    tic = time.time()
    for n in range(int(round(a.hours * 3600 / a.dt))):
        tau[:-1, :] = wind_at(n * a.dt, wind)
        st = m.step(st, tau_y=tau)
        if int((n + 1) * a.dt // 900.0) > int(n * a.dt // 900.0):
            buf["t_h"].append((n + 1) * a.dt / 3600.0)
            record(m, st, buf)
    rpe.append(m.rpe(st))
    np.savez(out / "series.npz", **{k: np.array(v) for k, v in buf.items()}, rpe=np.array(rpe),
             meta_json=json.dumps(vars(a)))
    print(f"{a.case} {a.bay} {a.coord}: {a.hours} h in {(time.time() - tic) / 60:.1f} min, trunc {st.n_trunc}")


if __name__ == "__main__":
    main()
```

- [ ] **Step 5: Smoke run** (1 h)

Run: `NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py 2layer bay --hours 1`
Expected: finishes in seconds, trunc 0. Load `series.npz`: `t_h` should be (5,) and `T_sec` (5, 50, 352).

- [ ] **Step 6: Commit**

```bash
git add validation/bayiw_2d/section.py validation/bayiw_2d/run_pyocean.py tests/test_bayiw_2d.py
git commit -m "BayIW 2-D section: pyOCEAN model and runner (task 3)"
```

---

### Task 4: Comparison

**Files:**
- Create: `validation/bayiw_2d/compare.py`

**Interfaces:**
- Consumes: `section.read_mitgcm`, the pyOCEAN `series.npz`; `validation.bayiw_baroclinic.compare.{dominant_period, rms, corr_lag, window, relative_field_error, interface_thickness}`; `bayiw.sample_at_depth`, `Z110`, `Z250`.
- Produces: `report(case, coord) -> list[str]`, which also prints.

- [ ] **Step 1: Implement**

```python
#!/usr/bin/env python
"""
Compare the BayIW 2-D section in pyOCEAN with MITgcm on the same section (spec
docs/superpowers/specs/2026-10-12-bayiw-2d-section-design.md §5).  MITgcm T and V are sampled at 110/250 m
from its level centres (z-levels: piecewise linear between centres, as pyOCEAN's sampling).

Usage: python validation/bayiw_2d/compare.py 2layer zstar
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from validation.bayiw_2d import section  # noqa: E402
from validation.bayiw_baroclinic import bayiw  # noqa: E402
from validation.bayiw_baroclinic.compare import (corr_lag, dominant_period, interface_thickness,  # noqa: E402
                                                 relative_field_error, rms, window)

IW = ("Seiche", "Seiche2", "Omega", "C")


def _flag(ok: bool) -> str:
    return "PASS" if ok else "FAIL"


def mit_at_depth(d: dict, H: np.ndarray, var: str, z0: float) -> np.ndarray:
    """MITgcm T (or V at the v-faces) at height z0 for every dump: (n, 352) or (n, 353); NaN below the bed."""
    X = d[var]
    zc = d["zc"][:, None] * np.ones((1, X.shape[2]))
    if var == "V":
        Hf = np.concatenate([H[:1], np.minimum(H[1:], H[:-1]), H[-1:]])
    else:
        Hf = H
    h = np.where(zc > -np.nan_to_num(Hf, nan=0.0)[None], 1.0, 0.0)          # levels above the bed (z-levels)
    return np.array([bayiw.sample_at_depth(zc, X[n], h, z0, -Hf) for n in range(X.shape[0])])


def report(case: str, coord: str) -> list[str]:
    L = [f"# BayIW 2-D section {case}, pyOCEAN {coord} vs MITgcm (same section)"]
    H = {b: section.section_inputs(case, b)["H"][:, 0] for b in ("bay", "nobay")}
    mit = {b: section.read_mitgcm(case, b) for b in ("bay", "nobay")}
    py = {b: dict(np.load(section.RUNS / "pyocean" / f"{case}_{b}_{coord}" / "series.npz")) for b in ("bay", "nobay")}
    tm, tp = mit["bay"]["t_h"], py["bay"]["t_h"]
    st = section.SECTION_STATIONS
    # seiche Δη, 0–27 h
    L.append("\n## Seiche Δη (0–27 h): period MIT/py [h], range MIT/py [mm], corr")
    for s in ("Seiche", "Seiche2"):
        j = st[s]
        dm = mit["bay"]["eta"][:, j] - mit["nobay"]["eta"][:, j]
        dp = np.interp(tm, tp, py["bay"]["eta"][:, j] - py["nobay"]["eta"][:, j])
        tw, xm = window(tm, dm, 0.0, 27.0)
        _, xp = window(tm, dp, 0.0, 27.0)
        Pm, Pp = dominant_period(tw, xm, 1.0, 10.0), dominant_period(tw, xp, 1.0, 10.0)
        Rm, Rp = np.ptp(xm), np.ptp(xp)
        c = corr_lag(tw, xm, xp, 0.0)[0]
        L.append(f"{s:8s} {Pm:5.2f} {Pp:5.2f} | {1e3 * Rm:6.2f} {1e3 * Rp:6.2f} | {c:5.2f} | "
                 f"{_flag(abs(Pp / Pm - 1) <= 0.02 and abs(Rp / Rm - 1) <= 0.10 and c >= 0.95)}")
    # internal waves: T anomaly at 110 and 250 m, 0–60 h
    for var, z0 in (("T110", bayiw.Z110), ("T250", bayiw.Z250)):
        L.append(f"\n## {var} anomaly (0–60 h): period MIT/py [h], rms py/MIT, corr, lag [h]")
        for b in ("bay", "nobay"):
            Xm = mit_at_depth(mit[b], H[b], "T", z0)
            for s, j in st.items():
                xm = Xm[:, j] - Xm[0, j]
                xp = np.interp(tm, tp, py[b][var][:, j] - py[b][var][0, j])
                if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                    continue
                Pm, Pp = dominant_period(tm, xm), dominant_period(tm, xp)
                ratio = rms(xp) / rms(xm)
                c0, lag, c = corr_lag(tm, xm, xp)
                ok = abs(Pp / Pm - 1) <= 0.05 and abs(ratio - 1) <= 0.20 and (c0 >= 0.8 if s in IW else True)
                L.append(f"{b:5s} {s:8s} {Pm:6.2f} {Pp:6.2f} | {ratio:5.2f} | {c0:5.2f} | {lag:+5.2f} | {_flag(ok)}")
    # bay effect ΔT110
    L.append("\n## Bay effect ΔT110 (bay − nobay), 0–60 h: rms py/MIT, corr")
    Xb, Xn = mit_at_depth(mit["bay"], H["bay"], "T", bayiw.Z110), mit_at_depth(mit["nobay"], H["nobay"], "T", bayiw.Z110)
    for s, j in st.items():
        dm = (Xb[:, j] - Xb[0, j]) - (Xn[:, j] - Xn[0, j])
        dp = np.interp(tm, tp, (py["bay"]["T110"][:, j] - py["bay"]["T110"][0, j])
                       - (py["nobay"]["T110"][:, j] - py["nobay"]["T110"][0, j]))
        if not np.isfinite(dm).all() or rms(dm) == 0:
            continue
        ratio, c = rms(dp) / rms(dm), corr_lag(tm, dm, dp, 0.0)[0]
        L.append(f"{s:8s} {ratio:5.2f} | {c:5.2f} | {_flag(1 / 3 <= ratio <= 3 and c > 0)}")
    # V at 110 m
    L.append("\n## V110 (0–60 h): rms py/MIT, corr   [bay | nobay]")
    for s, j in st.items():
        row = []
        for b in ("bay", "nobay"):
            xm = mit_at_depth(mit[b], H[b], "V", bayiw.Z110)[:, j]
            xp = np.interp(tm, tp, py[b]["V110"][:, j])
            if not (np.isfinite(xm).all() and np.isfinite(xp).all()) or rms(xm) == 0:
                row.append("   —   ")
                continue
            row.append(f"{rms(xp) / rms(xm):4.2f} {corr_lag(tm, xm, xp, 0.0)[0]:+4.2f}")
        L.append(f"{s:8s} " + " | ".join(row))
    # T section error and interface
    L.append("\n## T section anomaly (bay): relative rms error at 6, 24, 60 h"
             + ("; interface thickness MIT/py [m]" if case == "2layer" else ""))
    for h in (6.0, 24.0, 60.0):
        km, kp = int(np.argmin(np.abs(tm - h))), int(np.argmin(np.abs(tp - h)))
        Hb = H["bay"]
        below = bayiw.Z_C[:, None] < -np.nan_to_num(Hb, nan=0.0)[None]
        Tm = np.where(below, np.nan, mit["bay"]["T"][km]) - np.where(below, np.nan, mit["bay"]["T"][0])
        Tp = py["bay"]["T_sec"][kp] - py["bay"]["T_sec"][0]
        line = f"{h:4.0f} h  {relative_field_error(Tp, Tm):5.2f}"
        if case == "2layer":
            Tm_abs = np.where(below, np.nan, mit["bay"]["T"][km])
            line += (f" | {interface_thickness(Tm_abs, bayiw.Z_C, 18.52, 22.77):6.1f} "
                     f"{interface_thickness(py['bay']['T_sec'][kp], bayiw.Z_C, 18.52, 22.77):6.1f}")
        L.append(line)
    for b in ("bay", "nobay"):
        p = py[b]
        L.append(f"\n## Health py {b}: trunc {int(p['n_trunc'][-1])}, heat change {p['heat'][-1] / p['heat'][0] - 1:+.2e}, "
                 f"RPE gain {p['rpe'][-1] - p['rpe'][0]:.3e} J")
    print("\n".join(L))
    return L


if __name__ == "__main__":
    report(sys.argv[1], sys.argv[2])
```

- [ ] **Step 2: Check it on the existing runs**

Run the pyOCEAN two-layer z\* runs (≈1 min each), then compare:
```bash
NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py 2layer bay --coord zstar
NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py 2layer nobay --coord zstar
python validation/bayiw_2d/compare.py 2layer zstar
```
Expected: the tables print, with no exceptions and no NaN in the MITgcm columns at the stations above the bed.

Fix any sampling or alignment problem (review focus 5) before reading the PASS/FAIL. Record each fix as a ruling.

- [ ] **Step 3: Commit**

```bash
git add validation/bayiw_2d/compare.py
git commit -m "BayIW 2-D section: comparison with the spec metrics (task 4)"
```

---

### Task 5: Runs, classification and README

**Files:**
- Create: `validation/bayiw_2d/README.md`

- [ ] **Step 1: All pyOCEAN runs** (sequential; ≈1 min each)

```bash
for c in 2layer linear; do for b in bay nobay; do for z in zstar hybrid; do
  NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py $c $b --coord $z || exit 1
done; done; done
```
Expected: 8 lines with trunc 0.

- [ ] **Step 2: Compare all four**

Run:
```bash
for c in 2layer linear; do for z in zstar hybrid; do
  python validation/bayiw_2d/compare.py $c $z | tee /Volumes/ESTEBAN_MIT/pyOCEAN_runs/bayiw_2d/compare_${c}_${z}.txt
done; done
```

- [ ] **Step 3: Classify and write the README** (Spanish)

Write `validation/bayiw_2d/README.md`:
- purpose and links;
- configuration (spec §2–4) and the known (c) difference (MITgcm south net transport zero vs Flather);
- one table per case and coordinate;
- the classification (a)–(d) of every FAIL with its reasoning;
- z\* against hybrid against MITgcm;
- "Qué debe mejorar pyOCEAN".

If a FAIL is class (a): stop, diagnose (superpowers:systematic-debugging) and report to Esteban before going on.

- [ ] **Step 4: Commit**

```bash
git add validation/bayiw_2d/README.md
git commit -m "BayIW 2-D section: results vs MITgcm (task 5)"
```

---

### Task 6: Scorecard and documentation

**Files:**
- Modify: `docs/design/vision_roadmap.md` §7 (new row "BayIW sección 2-D"), `CLAUDE.md` (benchmark status), memory `project_status.md`.

- [ ] **Step 1: Write the scorecard row and `CLAUDE.md`**

Use the measured numbers from the README: seiche period/range/corr, and the T110 corr and amplitude ratio at the inner stations, for z\* and hybrid.

- [ ] **Step 2: Full suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider`
Expected: all pass, plus the existing xfails.

- [ ] **Step 3: Commit**

```bash
git add docs/design/vision_roadmap.md CLAUDE.md docs/superpowers/plans/2026-10-12-bayiw-2d-section.md
git commit -m "BayIW 2-D section: scorecard and docs (task 6)"
```
