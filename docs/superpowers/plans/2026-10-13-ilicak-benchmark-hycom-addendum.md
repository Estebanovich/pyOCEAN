# Four-model benchmark part A — HYCOM cost addendum (Implementation Plan)

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Measure HYCOM 2.3's cost on the Ilicak overflow (spec §A3) with the RELO build that already passes the gate, so the cost table of `validation/ilicak_benchmark/README.md` gets its fifth model.

**Architecture:** `validation/ilicak_benchmark/hycom/inputs.py` writes every file HYCOM opens for an idealised closed x–z run from `cases.get("overflow")`: `.a/.b` grid, depth and 12-month relax files (initial state through `iniflg=2`; `relax=0`, so nothing relaxes), `blkdat.input`, `patch.input`, `limits` and an empty `relax.weird`. `run_hycom.sh` runs the gate build on 1 and 5 MPI ranks. `timers.py` parses the second `xctmrp` block (time stepping only). Results are cost only (spec §A3): HYCOM's EOS, mixing and boundaries differ from the common physics.

**Tech Stack:** Python (NumPy), HYCOM 2.3 RELO MPI build (`/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom/build/hycom`, `CPP_EXTRAS="-DEOS_SIG0 -DEOS_7T"`), MPICH.

**Spec:** `docs/superpowers/specs/2026-10-13-ilicak-four-model-benchmark-design.md` §A3; HYCOM facts from a read of `~/pyOCEAN/HYCOM-src` (2026-10-13), file:line below.

## Global Constraints

- HYCOM source is not modified; inputs and runs live in `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/hycom/`.
- ≤ 5 MPI ranks, one run at a time, never the efficiency cores.
- Cost only: no HYCOM result enters the physics tables.
- Never commit outputs; commit messages end with the two attribution lines of the project.
- HYCOM facts used:
  - **`.a` files:** big-endian real*4, i fastest, each record padded to `4096·⌈idm·jdm/4096⌉` words, land = 2¹⁰⁰ (`mod_za_mp.h:99, 660, 1050`). This is the layout read by `forcing/sources/hycom_ab.py`. Each `.b` min/max must match its `.a` record to 1e-4 relative (`geopar.F90:175`).
  - **`regional.grid.b`:**
    - lines 1–3 are `idm`, `jdm`, `mapflg`;
    - then 18 records in this order: `plon, plat, qlon, qlat, ulon, ulat, vlon, vlat, pang, pscx, pscy, qscx, qscy, uscx, uscy, vscx, vscy, cori` (`geopar.F90:37-173`).
  - **`regional.depth.b`:** 5 title lines, then `min,max depth = a b` (`geopar.F90:208-253`).
  - **Closed domain:** the last column and the last row must be land (`bigrid.F90:26-95`).
  - **Initial state:** set with `iniflg=2` from `relax.{temp,saln,intf}.[ab]`.
    - `.b` files have 5 title lines, then one line per record: `name: month,layer,dens,range = MM KK σ min max`.
    - Records go month by month (`clmflg=12`: 12 months × kdm).
    - The intf values are in pressure units: `onem = 9806` per metre.
    - Code: `forfun.F90:2037-2085, 2741, 2867`; `inicon.F90:70-88`.
    - An empty `relax.weird` skips the dp0k sanity check (`forfun.F90:3713-3796`).
  - **`blkdat.input`:** each line is `value 'name6'`, in the order of `blkdat.F90` (listed in Task 2).
  - **`patch.input`:** fixed formats (`mod_xc_mp.h:2535, 2604-2649`). The RELO build reads idm, jdm from patch and kdm from blkdat.
  - **Timers:** second `xctmrp` block, lines `(3x,a6,'   calls =',i9,'   time =',f11.5,...)` (`mod_xc_mp.h:5436-5447`). `total` is the time-stepping wall time; `cnuity` calls = steps.
  - **7-term σ₀** (`stmt_fns.h:51-59, 317`): `sig(t,s) = c1 + c3 s + t (c2 + c5 s + t (c4 + c7 s + c6 t))`, with
    - c1 = −1.36471e-1, c2 = 4.68181e-2, c3 = 8.07004e-1, c4 = −7.45353e-3,
    - c5 = −2.94418e-3, c6 = 3.43570e-5, c7 = 3.48658e-5.

## Review Focus

1. **`.b` min/max against `.a` within 1e-4:** HYCOM stops at start-up otherwise. *Pinned in Task 1.*
2. **Land in the last column and row:** without it the domain counts as periodic and `bigrid` fails. *Pinned in Task 1.*
3. **blkdat line order:** one line out of place makes HYCOM read the wrong parameter, or stop. *Pinned in Task 2 by checking the ordered name list.*
4. **Time steps:** `baclin/batrop` must be even, `0.5·baclin/batrop` an integer, and `baclin` must divide 86400. *Pinned in Task 2.*
5. **Comparable cost:** HYCOM on 1 rank (like the others) and on 5; time-stepping block only. *Pinned in Task 3.*

---

### Task 1: Grid, depth and initial-state files

**Files:** Create `validation/ilicak_benchmark/hycom/__init__.py` (empty), `validation/ilicak_benchmark/hycom/inputs.py`; Test `tests/test_ilicak_benchmark.py` (append).

**Interfaces — Produces:**
- `inputs.SPVAL = 2.0**100`
- `inputs.sig0(T, S) -> np.ndarray`
- `inputs.write_ab(base: Path, records: list[tuple[str, np.ndarray]], header: list[str], fmt: Callable) -> None`
- `inputs.geometry() -> dict`: keys `idm`, `jdm`, `depth (jdm, idm)` (NaN on land), `T (jdm, idm)`, `S0`
- `inputs.write_grid(out: Path)`, `inputs.write_depth(out: Path)`, `inputs.write_relax(out: Path, kdm: int = 100)`

Domain: itdm = 202 and jtdm = 5. Columns 1 and 202 and rows 1 and 5 are land, which leaves 200 × 3 ocean cells with the overflow bed of `cases.py` in every ocean row. Δx = Δy = 1000 m, `pang = 0`, `cori = 0`, S = 35. The initial interfaces are the 100 z-levels of 20 m (cut at the bed). In every layer of a column, T is that column's T (10 or 20 °C).

- [ ] **Step 1: Failing tests**

```python
def test_hycom_sig0_matches_the_7_term_fit():
    from validation.ilicak_benchmark.hycom import inputs
    assert abs(float(inputs.sig0(10.0, 35.0)) - 26.957) < 2e-3            # stmt_fns.h:317 at (10 °C, 35)
    assert float(inputs.sig0(10.0, 35.0)) - float(inputs.sig0(20.0, 35.0)) > 1.9


def test_hycom_ab_round_trip_and_closed_domain(tmp_path):
    from forcing.sources import hycom_ab
    from validation.ilicak_benchmark import cases
    from validation.ilicak_benchmark.hycom import inputs
    inputs.write_grid(tmp_path)
    inputs.write_depth(tmp_path)
    g = inputs.geometry()
    d = hycom_ab.read_record(tmp_path / "regional.depth.b", 0, idm=g["idm"], jdm=g["jdm"])   # checks .b min/max
    assert np.isnan(d[:, 0]).all() and np.isnan(d[:, -1]).all() and np.isnan(d[0]).all() and np.isnan(d[-1]).all()
    np.testing.assert_allclose(d[2, 1:-1], cases.get("overflow").H[0], rtol=1e-6)        # float32
    lines = (tmp_path / "regional.grid.b").read_text().splitlines()
    assert lines[0].split()[0] == "202" and lines[1].split()[0] == "5" and len(lines) == 3 + 18
    for irec in range(18):                                     # read_record(check=True) compares .b min/max with .a
        hycom_ab.read_record(tmp_path / "regional.grid.b", irec, idm=g["idm"], jdm=g["jdm"])


def test_hycom_relax_files_hold_the_lock_columns(tmp_path):
    from forcing.sources import hycom_ab
    from validation.ilicak_benchmark.hycom import inputs
    inputs.write_relax(tmp_path, kdm=100)
    g = inputs.geometry()
    rd = lambda f, i: hycom_ab.read_record(tmp_path / f, i, idm=g["idm"], jdm=g["jdm"])          # noqa: E731
    T0 = rd("relax.temp.b", 0)
    assert np.isclose(T0[2, 1], 10.0) and np.isclose(T0[2, 100], 20.0)                 # x < 20 km cold
    rd("relax.temp.b", 1199)                                                            # 12 months × 100 layers
    assert np.allclose(np.nan_to_num(rd("relax.intf.b", 0)), 0.0) and np.isclose(rd("relax.intf.b", 1)[2, 1], 20.0 * 9806.0)
    head = (tmp_path / "relax.temp.b").read_text().splitlines()
    assert len(head) == 5 + 1200 and "= 01 01" in head[5].replace("  ", " ")
```

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k hycom` → Expected: FAIL (`No module named ...hycom.inputs`). The reader is `forcing.sources.hycom_ab.read_record(path_b, irec, idm, jdm, check=True)` (`hycom_ab.py:190`); if `read_b` cannot parse a `.b` line written here, adapt the writer's line format to the reader (`_parse_b_line`, `hycom_ab.py:112`) and ledger it.

- [ ] **Step 2: Implement `inputs.py`**

```python
"""
HYCOM 2.3 inputs of the Ilicak overflow cost case (spec 2026-10-13-ilicak-four-model-benchmark-design §A3), from
validation/ilicak_benchmark/cases.py: .a/.b grid and depth (mod_za_mp.h:99, 660, 1050; geopar.F90:37-253), the
initial state as 12-month relax files read with iniflg = 2 (forfun.F90:2037-2085, 2741; inicon.F90:70-88; relax = 0
so nothing relaxes), 7-term σ₀ of stmt_fns.h:51-59, 317 (built with -DEOS_SIG0 -DEOS_7T).
Closed domain: last column and row are land (bigrid.F90:26-95), first column and row too.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from validation.ilicak_benchmark import cases

SPVAL = 2.0 ** 100
ONEM = 9806.0                                            # Pa per metre (mod_cb_arrays.F90:844)
S0 = 35.0
C = (-1.36471e-1, 4.68181e-2, 8.07004e-1, -7.45353e-3, -2.94418e-3, 3.43570e-5, 3.48658e-5)


def sig0(T, S):
    """HYCOM 7-term σ₀ [kg m⁻³ − 1000] (stmt_fns.h:317)."""
    c1, c2, c3, c4, c5, c6, c7 = C
    T, S = np.asarray(T, float), np.asarray(S, float)
    return c1 + c3 * S + T * (c2 + c5 * S + T * (c4 + c7 * S + c6 * T))


def geometry() -> dict:
    c = cases.get("overflow")
    idm, jdm = c.nx + 2, 5
    depth = np.full((jdm, idm), np.nan)
    depth[1:-1, 1:-1] = c.H[0][None, :]
    T = np.full((jdm, idm), np.nan)
    T[1:-1, 1:-1] = np.where(c.x < 20e3, 10.0, 20.0)[None, :]
    return dict(idm=idm, jdm=jdm, depth=depth, T=T, S0=S0, dz=c.dz)


def _a(path: Path, arrays: list[np.ndarray]) -> list[tuple[float, float]]:
    """Write .a records (big-endian real*4, padded to 4096 words, NaN → 2^100); returns (min, max) of each."""
    out = []
    with open(path, "wb") as f:
        for a in arrays:
            a32 = np.where(np.isnan(a), SPVAL, a).astype(">f4").ravel()
            n = a32.size
            pad = (-n) % 4096
            np.concatenate([a32, np.full(pad, SPVAL, ">f4")]).tofile(f)
            v = a.astype(np.float32)[np.isfinite(a)]
            out.append((float(v.min()), float(v.max())))
    return out


def write_ab(base: Path, records: list[tuple[str, np.ndarray]], header: list[str], fmt) -> None:
    """base.a + base.b: header lines, then fmt(name, k, lo, hi) per record."""
    mm = _a(Path(f"{base}.a"), [r for _, r in records])
    lines = list(header) + [fmt(name, k, lo, hi) for k, ((name, _), (lo, hi)) in enumerate(zip(records, mm))]
    Path(f"{base}.b").write_text("\n".join(lines) + "\n")


def write_grid(out: Path) -> None:
    g = geometry()
    jdm, idm = g["jdm"], g["idm"]
    x = np.broadcast_to(np.arange(idm, dtype=float)[None, :] * 0.01, (jdm, idm))     # labels only (not metrics)
    y = np.broadcast_to(np.arange(jdm, dtype=float)[:, None] * 0.01, (jdm, idm))
    one = np.full((jdm, idm), 1000.0)
    recs = [("plon", x), ("plat", y), ("qlon", x), ("qlat", y), ("ulon", x), ("ulat", y), ("vlon", x), ("vlat", y),
            ("pang", np.zeros((jdm, idm)))] + [(n, one) for n in ("pscx", "pscy", "qscx", "qscy", "uscx", "uscy",
                                                                  "vscx", "vscy")] + [("cori", np.zeros((jdm, idm)))]
    write_ab(Path(out) / "regional.grid", recs,
             [f"{idm:5d}    'idm   ' = longitudinal array size", f"{jdm:5d}    'jdm   ' = latitudinal  array size",
              "    2    'mapflg' = map flag (uniform)"],
             lambda n, k, lo, hi: f"{n}:  min,max = {lo:15.5f} {hi:15.5f}")


def write_depth(out: Path) -> None:
    g = geometry()
    write_ab(Path(out) / "regional.depth", [("depth", g["depth"])],
             ["Ilicak overflow (pyOCEAN four-model benchmark)", "tanh shelf 500 -> 2000 m, Petersen 2015 eq. 24",
              "closed: land in the first/last column and row", "", ""],
             lambda n, k, lo, hi: f"min,max depth = {lo:10.3f} {hi:10.3f}")


def write_relax(out: Path, kdm: int = 100) -> None:
    """12 identical months of T, S and interface pressure (k = 1…kdm; interface k on top of layer k)."""
    g = geometry()
    zf = np.concatenate([[0.0], np.cumsum(g["dz"])])[:kdm]
    d = g["depth"]
    rho = sig0(np.nanmin(g["T"]), S0)
    tem, sal, itf = [], [], []
    for _ in range(12):
        for k in range(kdm):
            tem.append(("tem", g["T"]))
            sal.append(("sal", np.where(np.isfinite(d), S0, np.nan)))
            itf.append(("int", np.where(np.isfinite(d), np.minimum(zf[k], np.nan_to_num(d)) * ONEM, np.nan)))
    fmt = lambda n, i, lo, hi: (f"{n}: month,layer,dens,range = {i // kdm + 1:02d} {i % kdm + 1:02d} "   # noqa: E731
                                f"{rho:.3f} {lo:.6e} {hi:.6e}")
    head = ["Ilicak overflow initial state (iniflg = 2; relax = 0)", "", "", "", ""]
    for name, recs in (("relax.temp", tem), ("relax.saln", sal), ("relax.intf", itf)):
        write_ab(Path(out) / name, recs, head, fmt)
    (Path(out) / "relax.weird").write_text("")
```

Run: `python -m pytest tests/test_ilicak_benchmark.py -q -k hycom` → Expected: 3 passed.

- [ ] **Step 3: Commit** `validation/ilicak_benchmark/hycom/__init__.py`, `validation/ilicak_benchmark/hycom/inputs.py`, `tests/test_ilicak_benchmark.py`: "Ilicak benchmark: HYCOM grid, depth and relax inputs".

---

### Task 2: blkdat.input, patch.input, limits

**Files:** Create `validation/ilicak_benchmark/hycom/blkdat.py`; Test append.

**Interfaces — Produces:**
- `blkdat.blkdat(kdm: int, baclin: float, batrop: float, nu_h: float, sigma: np.ndarray, dp0k: np.ndarray) -> str`
- `blkdat.patch(npes: int, idm: int, jdm: int) -> str`
- `blkdat.limits(hours: float) -> str`
- `blkdat.write_case(out: Path, npes: int, baclin: float, hours: float, nu_h: float = 100.0, kdm: int = 100) -> Path`, which writes every input of Tasks 1–2 into `out`.

The blkdat lines, in `blkdat.F90` reading order (title × 4, then `value 'name6'`):

`iversn` 23, `iexpt` 10, `idm`, `jdm`, `itest` 0, `jtest` 0, `kdm`, `nhybrd` kdm, `nsigma` 0, `dp0k` × kdm (20 m each), `ds0k` × 1 (20 m), `dp00i` 1.0, `isotop` 5.0, `oneta0` 0.1, `saln0` 35, `locsig` 0, `kapref` 0, `thflag` 0, `thbase` 25.0, `vsigma` 0, `sigma` × kdm, `iniflg` 2, `jerlv0` 1, `yrflag` 0, `sshflg` 0, `dsurfq` 0, `diagfq` 0, `proffq` 0, `tilefq` 0, `meanfq` 0, `rstrfq` 0, `bnstfq` 0, `nestfq` 0, `cplifq` 0, `baclin`, `batrop`, `incflg` 0, `incstp` 1, `incupf` 0, `stfflg` 0, `stfrdt` 0, `stfrds` 0, `stfrdv` 0, `ra2fac` .125, `wbaro` .125, `btrlfr` 1, `btrmas` 0, `hybraf` 0, `hybrlx` 1.0, `hybiso` 0.01, `hybthn` 1.0, `hybthk` 0.0, `hybmap` 3, `hybflg` 0, `advflg` 0, `advtyp` 2, `momtyp` 2, `slip` 1.0, `visco2` 0, `visco4` 0, `facdf4` 0, `veldf2` ν_h/1000, `veldf4` 0, `thkdf2` 0, `thkdf4` 0, `temdf2` 0, `temdfc` 1.0, `vertmx` 0, `cbar` 0.05, `cb` 0, `cbtidc` 0, `drglim` 0, `thkbot` 10, `sigjmp` .02, `tmljmp` .2, `sefold` 30, `thkmls` 15, `thkmlt` 0, `thkriv` 0, `thkcdw` 3, `thkfrz` 20, `iceflg` 0, `tfrz_0` 0, `tfrz_s` -.054, `frzifq` -1, `ticegr` 0, `hicemn` .5, `hicemx` 10, `ishelf` 0, `ntracr` 0, `trcflg` 0, `mtracr` 0, `mstrcr` 0, `tsofrq` 1, `tofset` 0, `sofset` 0, `mlflag` 0, `pensol` 0, `dtrate` 999, `thkmin` 0, `dypflg` 0, `mixfrq` 1, `diapyc` 0, `rigr` .25, `ribc` .65, `rinfty` .7, `ricr` .45, `bldmin` 0, `bldmax` 1200, `cekman` .7, `cmonob` 1, `bblkpp` 0, `shinst` 0, `dbdiff` 0, `nonloc` 0, `botdiw` 0, `difout` 0, `difsmo` 0, `difm0` 5e-3, `difs0` 5e-3, `difmiw` 1e-4, `difsiw` 1e-5, `dsfmax` 1e-3, `rrho0` 1.9, `cs` 98.96, `cstar` 10, `cv` 0, `c11` 5, `hblflg` 2, `niter` 2, `langmr` 0, `fltflg` 0, `nfladv` 0, `nflsam` 0, `intpfl` 0, `iturbv` 0, `ismpfl` 0, `tbvar` 0, `tdecri` 0, `lbflag` 0, `lbmont` 0, `tidflg` 0, `tidein` 0, `tidcon` 0, `tidsal` 0, `tiddrg` 0, `thkdrg` 0, `drgscl` 0, `tidfm2` 0, `tidfs2` 0, `tidfk1` 0, `tidfo1` 0, `drgscm` 0, `drgscs` 0, `drgsck` 0, `drgsco` 0, `tidgen` 0, `tidrmp` 0, `tid_t0` 0, `tidnud` 0, `nudrmp` 0, `nud_t0` 0, `clmflg` 12, `wndflg` 0, `ocnscl` 0, `ustflg` 1, `flxflg` 0, `empflg` 0, `emptgt` 0, `empbal` 0, `dswflg` 0, `albflg` 0, `sssflg` 0, `sssbal` 0, `lwflag` 0, `sstflg` 0, `icmflg` 0, `prsbas` 0, `mslprf` 0, `stroff` 0, `flxoff` 0, `flxsmo` 0, `relax` 0, `trcrlx` 0, `priver` 0, `epmass` 0.

This order comes from the subagent read of `blkdat.F90`. The HYCOM run in Task 3 is the check: if HYCOM stops on a name, compare against `blkdat.F90`, fix the order and ledger it.

The `sigma` targets are evenly spaced σ₀ from `sig0(20, 35)` to `sig0(10, 35)`, with the two water masses at the centres of the first and last layers (the `cases.targets` rule in σ₀ units).

- [ ] **Step 1: Failing tests**

```python
def test_hycom_blkdat_order_and_time_steps():
    from validation.ilicak_benchmark.hycom import blkdat, inputs
    sig = np.linspace(24.5, 27.0, 100)
    txt = blkdat.blkdat(100, 10.0, 1.0, 100.0, sig, np.full(100, 20.0))
    names = [ln.split("'")[1] for ln in txt.splitlines()[4:]]
    assert names[:7] == ["iversn", "iexpt ", "idm   ", "jdm   ", "itest ", "jtest ", "kdm   "]
    assert names.count("dp0k  ") == 100 and names.count("sigma ") == 100 and names[-1] == "epmass"
    assert names.index("baclin") < names.index("batrop") < names.index("incflg")
    vals = {ln.split("'")[1]: ln.split()[0] for ln in txt.splitlines()[4:]}
    assert float(vals["veldf2"]) == 0.1 and vals["iniflg"] == "2" and vals["relax "] == "0"
    with pytest.raises(ValueError):
        blkdat.blkdat(100, 10.0, 3.0, 100.0, sig, np.full(100, 20.0))       # 10/3 not an even integer
    assert "    5     5     1   202     5    41     5     0" in blkdat.patch(5, 202, 5)
    assert blkdat.limits(12.0).split() == ["0.0", "0.5000000000"]
```

Run → FAIL (no module).

- [ ] **Step 2: Implement `blkdat.py`**

`blkdat()` builds the lines `f"  {value:<12} '{name:<6}' = "` in the order above. Integers print as integers; dp0k/sigma loop over k. It raises `ValueError` unless `(baclin / batrop) % 2 == 0` and `86400 % baclin == 0`.

`patch(npes, idm, jdm)` writes the fixed-format layout:
- `(/8i6/)` header with `npes npe mpe idm jdm ibig jbig nreg` = `npes npes 1 idm jdm ceil(idm/npes) jdm 0`;
- a blank line;
- `ispt(  1) =` and `iipe(  1) =` lines (`12x,8i6`): tiles split as evenly as possible, the larger ones first;
- a blank line;
- `jspt(  1) =      1` and `jjpe(  1) =` with jdm.

`limits(hours)` returns `f"0.0 {hours / 24:.10f}\n"`.

`write_case()` writes:
- the Task 1 files;
- `blkdat.input`, with `sigma` from `inputs.sig0` and `dp0k` = 20 m;
- `patch.input` and `limits`.

Run → 1 passed. Commit: "Ilicak benchmark: HYCOM blkdat, patch and limits".

---

### Task 3: Run, timers, cost table

**Files:** Create `validation/ilicak_benchmark/hycom/run_hycom.sh`, `validation/ilicak_benchmark/hycom/timers.py`. Modify `validation/ilicak_benchmark/hycom/README.md` and `validation/ilicak_benchmark/README.md` (cost table: HYCOM column). Test append.

**Interfaces — Produces:**
- `timers.parse(stdout: str) -> dict[str, float]`: name → time [s] of the **last** `timer statistics` block.
- `run_hycom.sh <npes> <baclin> <batrop> <hours>`: writes the case to `RUNS/hycom/run_np<npes>_dt<baclin>`, runs `mpirun -n <npes>` with the gate build, and prints `total`, `cnuity calls` and s per simulated hour. Like the MOM6 runner, it stops only its own ranks if MPI hangs in `MPI_Finalize`.

- [ ] **Step 1: Failing test**

```python
def test_hycom_timer_parser_reads_the_last_block():
    from validation.ilicak_benchmark.hycom import timers
    out = ("   timer statistics, processor    1 out of    1\n"
           "   total    calls =        1   time =    9.00000   time/call =    9.00000000\n"
           "   timer statistics, processor    1 out of    1\n"
           "   cnuity   calls =     4320   time =    1.50000   time/call =    0.00034722\n"
           "   total    calls =        1   time =    6.25000   time/call =    6.25000000\n")
    t = timers.parse(out)
    assert t["total"] == 6.25 and t["cnuity"] == 1.5 and t["cnuity_calls"] == 4320
```

- [ ] **Step 2: Implement** the parser (regex on `(\w+)\s+calls =\s*(\d+)\s+time =\s*([\d.]+)`, last block wins) and the runner.

- [ ] **Step 3: Runs (one at a time)**

  1. Smoke run: 1 rank, baclin 10, batrop 1, 1 h. On a blkdat or patch error, fix against the source (file:line) and ledger it.
  2. Cost: ladder baclin 10, 20, 40, 80, 160 with batrop 1, 2, 2, 2, 2, over 12 h at ν = 100. It is stable while the run ends "normal" and `summary_out` / stdout show finite fields. Run it on 1 rank, then on 5 ranks for the largest stable baclin.

  Expected: the case runs and the largest stable baclin and its s per simulated hour are recorded. If HYCOM cannot complete the smoke run after a ~1-h time box of fixes, stop and report.

- [ ] **Step 4: Docs.** In `hycom/README.md`, the outcome, timers and per-component split (`cnuity`, `tsadvc`, `momtum`, `barotp`, `hybgen`). In the main README cost table, add a HYCOM column, marked "cost only; EOS 7-term σ₀, no explicit mixing, hybgen". Commit: "Ilicak benchmark: HYCOM cost on the overflow".
