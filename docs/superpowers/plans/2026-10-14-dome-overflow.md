# DOME overflow benchmark (passive tracers, interior sponge, ALE bottom drag, exact inflow) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Run the DOME overflow (Legg et al. 2006, Case 1) in pyOCEAN (z\*, σ, hybrid), MOM6 ρ and MITgcm z at 10 and 5 km with one case definition. Measure plume descent, entrainment, mixing, robustness and cost. Close stage C's DOME row.

**Architecture:** Four model capabilities come first, each an option that is off by default, so every existing result stays bit-identical:
1. passive tracers carried in the ALE tracer stack;
2. an interior sponge applied after the regrid–remap;
3. a quadratic bottom drag in the ALE viscous step;
4. an exact prescribed transport at `clamped` open edges.

Then `validation/dome/` follows the `validation/ilicak_benchmark/` pattern: one `cases.py`, one writer and reader per model, common series, metrics, README.

**Tech Stack:** Python 3, NumPy 1.26.4, Numba (≤ 5 threads), xarray/netCDF4, pytest; MOM6 `repro` build; MITgcm (gfortran, `darwin_arm64_gfortran_esteban`).

**Spec:** `docs/superpowers/specs/2026-10-14-dome-overflow-design.md`. Read it first: case definition, criteria and decision points live there. Legg 2006 is in the local library (`python scripts/litsearch.py "DOME"`; `Legg_2006 - ….md`, §2 lines 144–300, tables 1–3, Figs. 15–17).

## Global Constraints

- **Threads:** ≤ 5 (`NUMBA_NUM_THREADS=5`, `set_threads(5)`). One heavy run at a time. Never the efficiency cores, never lowered priority.
- **After every MPI run** (MOM6), check for orphan ranks: `pgrep -fl "mpirun|orted|MOM6"` must be empty.
- **Outputs:**
  - outputs go to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/`;
  - never commit `log_*.txt`, `*.npz`, model outputs or `pyOCEAN-literature-review-guide.docx`;
  - `git add` takes explicit paths only, never directories;
  - the working tree also holds another session's files (`docs/manual/`, `mkdocs.yml`, `.gitignore`, `scripts/build_manual_pdf.sh`, `docs/requirements-docs.txt`): never stage them.
- **Commit messages** end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn
  ```
- **Reference models:** do not modify MOM6, FMS, MITgcm or CROCO sources. Configurations go only in run/build directories.
- **Packages:** no installs; NumPy stays 1.26.4.
- **Defaults stay as they are:** `n_tracers=0`, `sponge_rate=None`, `bottom_drag=None`, no `transport` in OBData. The existing suite (1348 passed + 2 xfail) and the fingerprints in `tests/test_sigma_ale.py::test_*fingerprint*` must stay bit-identical.
- **Docstrings:**
  - every equation has a LaTeX-style comment with its reference;
  - units in docstrings;
  - Spanish in README/docs, English in code;
  - match the comment density of the surrounding code.
- **`$WS`** is this plan's workspace (`.superpowers/sdd/2026-10-14-dome-overflow/`, from the skill's `sdd-workspace` script): long test output goes there; read only its tail.
- **Bounded runs:** 10 km runs take minutes; a 5 km run must take ≲ 1 h. If a run is slower, stop and report.

## Review Focus

1. **A model with `n_tracers > 0` whose open-boundary provider gives OBData without `tracers`:** the inflow tracer must be 0, not a crash and not T. Test in Task 1.
2. **Interior sponge rates on land cells (NaN bathymetry) or with `sponge_rate` of the wrong shape:** land must be ignored; a wrong shape must raise a ValueError naming the expected shape. Test in Task 2.
3. **Bottom drag in a column that is dry or thinner than `d_min`, or with all-vanished bed layers (z\* columns shallower than the deepest level):** r must be finite, with no NaN, and the stress must reach the lowest thick layer. Test in Task 3.
4. **Prescribed transport on edge faces whose boundary cell is land or dry:** those faces must carry no flux, and the remaining faces must still carry exactly their prescribed values. Test in Task 4.
5. **A restart written with tracers read into a model with `n_tracers=0`, or the other way round:** must refuse with a ValueError naming `n_tracers`. Test in Task 1.

---

### Task 1: Passive tracers in the ALE model (`n_tracers`)

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py`:
  - `SigmaState` (≈107–122);
  - `__init__` signature (1005–1020) and body after `_setup_ale` (≈1111);
  - `initial_state` (1190–1220) and `_initial_state_from_source` (1222–1265);
  - `tracer_content` (1296);
  - `_diffuse_h` / `_diffuse_h_numpy` (1578–1633);
  - `_open_faces` (1556–1576), `_rhs` (1803–1815), `_ext_stack` (1817–1820);
  - new `_exterior_tr`, after `_exterior` (1758);
  - `_step_ale` (2038–2081).
- Modify: `boundaries/obcs/__init__.py` (OBData, 124–152)
- Modify: `core/dynamics/open_boundary.py`:
  - `validate_state` (19–37);
  - `_source_columns` (64–77), `_source_columns_at` (105–113);
  - `_exterior_tracer_full`, `exterior_tracer` (80–135).
- Modify: `core/restart.py` (`_config` 54–78, `write_restart` 81–131, `read_restart` 134–184)
- Modify: `diagnostics/output.py` (`FIELDS` 34–43, `_record` 110–127)
- Test: `tests/test_sigma_ale.py`, `tests/test_restart.py`, `tests/test_output.py`

**Interfaces:**
- Produces:
  - `SigmaBaroclinic(..., n_tracers: int = 0)` and the attribute `model.n_tracers: int`.
  - `SigmaState.tr: np.ndarray | None`, shape (n_tracers, N, ny, nx), k = 0 at the bed.
  - `model.initial_state(eta, T, h_src=None, tr=None)`: `tr` is on the same grid as `T` (the source grid when `h_src` is given).
  - `model.tr_ext: np.ndarray | None`, the resting exterior kept in z with `h_ext`.
  - `OBData.tracers: np.ndarray | None`, shape (n_tracers, nz, n_bdy) on the OBData's `zw`.
  - `open_boundary.exterior_tracer(state, H, h, mask, scheme, open_edges=None, field=None)`: `field` None means T; an int i means `ob.tracers[i]`, or zeros when `ob.tracers` is None.
  - `model.tracer_content(st, power=1, tracer=None)`.
  - Output fields `tr0`, `tr1`, …

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sigma_ale.py`)

```python
# ---------------------------------------------------------------- passive tracers (DOME spec §1)
def _tracer_box(rule, ntr=1, open_=False, **kw):
    """Sloping 2-D channel with stratification and a sloshing surface; tracer 0 = T at t = 0."""
    nx, nz = 24, 8
    H = np.linspace(30.0, 12.0, nx)[None]
    params = dict(dz_top=np.full(nz, 30.0 / nz))
    if rule == "hybrid":
        params["rho_target"] = np.linspace(996.0, 999.5, nz + 1)
    extra = {}
    if open_:
        from boundaries.obcs import OBData
        zw = np.linspace(0.0, -30.0, 4)[:, None]

        def state(t):
            return {"west": OBData(side="west", T=np.full((3, 1), 14.0), u_normal=np.full((3, 1), 0.02), zw=zw,
                                   ubar=np.full(1, 0.02))}
        extra = dict(open_boundaries=("west",), open_boundary_mode={"west": "clamped"}, open_boundary_state=state)
    m = _ale_box(rule, nx=nx, nz=nz, bathy=H, params=params, dt=20.0, n_tracers=ntr, **extra, **kw)
    zc = m.layer_heights(np.zeros((1, nx)))
    T = 10.0 + 0.2 * (zc + 30.0)
    eta = 0.05 * np.cos(np.pi * np.arange(nx) / (nx - 1))[None]
    return m, m.initial_state(eta, T, tr=np.stack([T] * ntr) if ntr else None)


@pytest.mark.parametrize("rule", ["zstar", "sigma", "hybrid"])
def test_a_passive_tracer_started_as_T_stays_identical_to_T(rule):
    # oracle: same stack, same kernels, same remap — any special-casing of tracers breaks bit-identity
    m, st = _tracer_box(rule)
    for _ in range(40):
        st = m.step(st)
    assert st.tr.shape == (1,) + st.T.shape
    assert np.array_equal(st.tr[0], st.T)


def test_passive_tracer_is_conserved_and_stays_in_0_1_in_a_closed_box():
    m, st = _tracer_box("hybrid")
    x = np.arange(24)[None, None, :]
    st.tr = np.where((x > 8) & (x < 14), 1.0, 0.0) * np.ones_like(st.T)[None]
    c0 = m.tracer_content(st, tracer=0)
    for _ in range(60):
        st = m.step(st)
    assert abs(m.tracer_content(st, tracer=0) - c0) <= 1e-12 * abs(c0)
    assert st.tr.min() >= 0.0 and st.tr.max() <= 1.0


def test_inflow_tracer_is_zero_when_the_exterior_gives_no_tracers():
    m, st = _tracer_box("zstar", open_=True)
    st.tr = np.zeros_like(st.tr)
    for _ in range(30):
        st = m.step(st)
    assert np.abs(st.tr).max() == 0.0


def test_tracers_need_ale_and_matching_shapes():
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    with pytest.raises(ValueError, match="n_tracers"):
        SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=10.0, n_layers=4, z0=None, n_tracers=1)
    m, _ = _tracer_box("zstar", ntr=2)
    with pytest.raises(ValueError, match="tr"):
        m.initial_state(np.zeros((1, 24)), np.full((8, 1, 24), 10.0), tr=np.zeros((1, 8, 1, 24)))


def test_no_tracers_leaves_the_state_without_tr():
    m, st = _tracer_box("zstar", ntr=0)
    assert m.step(st).tr is None
```

Append to `tests/test_restart.py`:

```python
def _tracer_model(ntr):
    from tests.test_sigma_ale import _tracer_box
    return _tracer_box("hybrid", ntr=ntr)


def test_ale_restart_with_tracers_is_bit_for_bit(tmp_path):
    m, st0 = _tracer_model(1)
    st = st0
    for _ in range(25):
        st = m.step(st)
    m1, st1 = _tracer_model(1)
    for _ in range(10):
        st1 = m1.step(st1)
    write_restart(tmp_path / "rst.nc", m1, st1)
    m2, _ = _tracer_model(1)
    st2 = read_restart(tmp_path / "rst.nc", m2)
    for _ in range(15):
        st2 = m2.step(st2)
    assert np.array_equal(st2.tr, st.tr) and np.array_equal(st2.T, st.T)


def test_restart_refuses_another_tracer_count(tmp_path):
    m, st = _tracer_model(1)
    write_restart(tmp_path / "rst.nc", m, m.step(st))
    m0, _ = _tracer_model(0)
    with pytest.raises(ValueError, match="n_tracers"):
        read_restart(tmp_path / "rst.nc", m0)
```

Append to `tests/test_output.py`:

```python
def test_output_accepts_numbered_passive_tracers(tmp_path):
    from core.grid.horizontal import create_cartesian_grid
    from diagnostics.output import OutputWriter
    hg = create_cartesian_grid(nx=4, ny=3, dx=1.0, dy=1.0)
    w = OutputWriter(tmp_path / "o.nc", hg, np.full((3, 4), 10.0), n_layers=2)
    w.write(0.0, tr0=np.zeros((2, 3, 4)), tr1=np.ones((2, 3, 4)))
    w.close()
    import xarray as xr
    with xr.open_dataset(tmp_path / "o.nc") as ds:
        assert ds["tr1"].attrs["long_name"] == "passive tracer 1" and float(ds["tr1"].max()) == 1.0
```

- [ ] **Step 2: Run the tests to verify they fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k "tracer" tests/test_restart.py -k "tracer" tests/test_output.py -k tracers -q -p no:cacheprovider`
Expected: FAIL. The constructor rejects `n_tracers` ("unexpected keyword argument 'n_tracers'"), `tracer_content` has no `tracer` argument, and output says "unknown output field 'tr0'".

- [ ] **Step 3: Implement the model side**

In `SigmaState`, add the last field:

```python
    tr: np.ndarray | None = None                             # passive tracers (ntr, N, ny, nx), ALE only
```

In `__init__`, add the keyword `n_tracers: int = 0` after `tracer_limiter`, and append to the class docstring:

```
    n_tracers : ALE: number of passive tracers carried with T in the same advection stack (WENO5 + FCT with their own
        3-D bounds), remap, diffusion, open boundaries and sponges; no effect on density (spec 2026-10-14-dome §1)
```

At the end of `__init__`:

```python
        self.n_tracers = int(n_tracers)
        if self.n_tracers < 0 or (self.n_tracers and not self.ale):
            raise ValueError(f"n_tracers={n_tracers!r}: passive tracers need vertical_coordinate='ale' and n ≥ 0")
        self.tr_ext, self._tr_in = None, None
```

Add the helper below `initial_state`:

```python
    def _check_tracers(self, tr, shape: tuple) -> np.ndarray | None:
        """Passive tracers (n_tracers,) + shape (zeros when None), or None without tracers; ValueError otherwise."""
        if self.n_tracers == 0:
            if tr is not None:
                raise ValueError("tr given but the model has n_tracers=0")
            return None
        if tr is None:
            return np.zeros((self.n_tracers,) + tuple(shape))
        tr = np.array(tr, float)
        if tr.shape != (self.n_tracers,) + tuple(shape):
            raise ValueError(f"tr shape {tr.shape}, expected {(self.n_tracers,) + tuple(shape)}")
        return tr
```

In `initial_state`:
- the signature becomes `def initial_state(self, eta, T, h_src=None, tr=None) -> SigmaState:`;
- add to its docstring: "``tr``: passive tracers (n_tracers, ...) on the grid of T (default 0)";
- pass `tr` to `_initial_state_from_source(eta, T, h_src, tr)`;
- before building the state: `tr = self._check_tracers(tr, np.shape(T))`;
- build the state with `tr=tr`;
- replace the hybrid loop and the exterior copy with:

```python
        if self.ale and self.regrid_rule == "hybrid":
            for _ in range(self.init_iterations):
                fields = [st.T] + ([] if st.tr is None else list(st.tr))
                st.h, rem = self._regrid_remap(st.eta, st.h, fields, st.T)
                st.T = rem[0]
                st.tr = None if st.tr is None else np.stack(rem[1:])
        if self.ale:                                                    # exterior profile in z (spec §3.5)
            self.T_ext, self.h_ext = st.T.copy(), st.h.copy()
            self.tr_ext = None if st.tr is None else st.tr.copy()
```

In `_initial_state_from_source(self, eta, T_src, h_src, tr_src=None)`, after `T = from_source(h)`:

```python
        tr_src = self._check_tracers(tr_src, T_src.shape)
        tr = None if tr_src is None else np.stack(
            [np.where(wet[None], rm.remap_columns(h_src, f, h, self.remap_code, wet), 0.0) for f in tr_src])
```

Build `SigmaState(..., T=T, h=h, tr=tr)` and set `self.tr_ext = None if tr is None else tr.copy()`.

`tracer_content`:

```python
    def tracer_content(self, st: SigmaState, power: int = 1, tracer: int | None = None) -> float:
        """Σ V c^power [unit³ m³] of T (``tracer`` None) or of passive tracer ``tracer``."""
        V = self.layer_volumes(st.eta, st.h)
        c = st.T if tracer is None else st.tr[tracer]
        return float(np.sum(V * c**power))
```

`_diffuse_h(self, T, V, eta, h=None, anomaly: bool = True)` and `_diffuse_h_numpy` gain the same argument. Replace the `Tw` condition in both with:

```python
        if anomaly and self.rho_ref is not None and not isinstance(self.rho_ref, str) and self.t_coef != 0:
```

Document it: "anomaly: diffuse T − T_ref(z) (temperature); False for passive tracers, which have no reference profile".

Exterior stacks. Replace `_ext_stack` and use it in `_open_faces` and `_rhs`:

```python
    def _ext_stack(self, c: np.ndarray) -> np.ndarray:
        """
        Inflow values of the advected stack on open edges: T_in (and T_in² when the stack carries the mixing
        diagnostic) followed by the passive tracers' inflow values; ``c`` itself without open edges or exterior.
        """
        T_in = self._inflow_T()
        if T_in is None or not self.open:
            return c
        lead = c.shape[0] - (self.n_tracers if self.ale else 0)        # [T] or [T, T²] before the tracers
        parts = [T_in, T_in**2][:lead]
        if self.ale and self.n_tracers:
            parts += list(self._tr_in)
        return np.ascontiguousarray(np.stack(parts))
```

- In `_open_faces`: `T_in = self._inflow_T(); ext = np.stack([T_in, T_in**2])[: c.shape[0]]` becomes `ext = self._ext_stack(c)`.
- In `_rhs`: the line `ext = c if T_in is None ... [: c.shape[0]])` becomes `ext = self._ext_stack(c)`, and the now-unused `T_in` line goes.

New method after `_exterior`:

```python
    def _exterior_tr(self, h: np.ndarray) -> np.ndarray:
        """Exterior passive tracers (n_tracers, N, ny, nx) on layers h in the boundary/sponge columns: the
        provider's OBData.tracers where it owns the column (0 when it gives none), else the resting tr_ext in z."""
        numba.set_num_threads(_THREADS[0])
        rest = np.stack([exterior_on_layers(self.h_ext, f, h, self._ext_cols, self.remap_code) for f in self.tr_ext])
        if self._ext is None:
            return rest
        st, code = self._ext["state"], self.remap_code
        out = np.stack([obc.exterior_tracer(st, self.H, h, self._ext_cols, code, self.open, field=i)
                        for i in range(self.n_tracers)])
        return np.where(self._ext["own"][None, None], out, rest)
```

In `_step_ale`, replace the block from `self._T_in = ...` to `T_new = rem[0]` with:

```python
        self._T_in = self._exterior_T(st.h) if self.open else None        # inflow T on hⁿ (spec §3.5)
        ntr = self.n_tracers
        self._tr_in = self._exterior_tr(st.h) if (self.open and ntr) else None
        mixd = self.mix_diagnostic
        lead = 2 if mixd else 1
        parts = [st.T, st.T**2] if mixd else [st.T]
        stack = np.stack(parts + ([] if st.tr is None else list(st.tr)))
        C = (self._advect_fct(stack, V0, V1, Qx, Qy, ok) if self.tracer_limiter == "fct"
             else self._advect(stack, V0, V1, Qx, Qy, Fz, ok))
        mix = V1 * (C[1] - C[0] ** 2) / dt if mixd else None                # advection part
        T_lag = C[0]
        tr_lag = [C[lead + i] for i in range(ntr)]
        if self.kh > 0:
            T_lag = self._diffuse_h(T_lag, V1, eta_new, h_lag)
            tr_lag = [self._diffuse_h(c, V1, eta_new, h_lag, anomaly=False) for c in tr_lag]
        if self.kv > 0:
            dz = np.maximum(h_lag, DZ_FLOOR).reshape(n, -1)
            kvz, zero = np.full((n + 1, dz.shape[1]), self.kv), np.zeros(dz.shape[1])
            T_lag = vertical_implicit(T_lag.reshape(n, -1), dz, kvz, zero, dt).reshape(T_lag.shape)
            tr_lag = [vertical_implicit(c.reshape(n, -1), dz, kvz, zero, dt).reshape(c.shape) for c in tr_lag]
        if self._sp_c is not None and self.sponge_tracer:                 # tracer sponge toward the exterior on h†
            r = dt * self._sp_c[None]
            T_lag = (T_lag + r * self._exterior_T(h_lag)) / (1.0 + r)
            if ntr:
                ext_tr = self._exterior_tr(h_lag)
                tr_lag = [(c + r * e) / (1.0 + r) for c, e in zip(tr_lag, ext_tr)]
        fields = ([T_lag, T_lag**2] if mixd else [T_lag]) + tr_lag
        h_new, rem = self._regrid_remap(eta_new, h_lag, fields, T_lag)
        T_new = rem[0]
        tr_new = np.stack(rem[lead:]) if ntr else None
```

Pass `tr=tr_new` in the returned `SigmaState(...)`.

`OBData` (`boundaries/obcs/__init__.py`): add the field and its docstring entry:

```python
    tracers: np.ndarray | None = field(default=None)
```

```
    tracers : np.ndarray (ntr, nz, n_bdy) or None
        Passive tracers on the same ``zw`` grid (ALE ``n_tracers``); None = 0 in the exterior.
```

`core/dynamics/open_boundary.py`:
- In `validate_state`, after the eta/ubar loop, add:

  ```python
          if ob.tracers is not None and (np.ndim(ob.tracers) != 3 or np.shape(ob.tracers)[1:] != (nz, n)):
              raise ValueError(f"{edge}: tracers shape {np.shape(ob.tracers)}, expected (ntr, {nz}, {n})")
  ```

- Add a helper:

  ```python
  def _values(ob: OBData, field) -> np.ndarray:
      """The (nz, n) profile of T (``field`` None) or of passive tracer ``field`` (zeros when OBData has none)."""
      if field is None:
          return np.asarray(ob.T, float)
      return np.zeros(np.shape(ob.T)) if ob.tracers is None else np.asarray(ob.tracers, float)[field]
  ```

- In `_source_columns` and `_source_columns_at`, add a keyword `field=None` and replace `np.asarray(ob.T, float)[:, a][::-1]` with `_values(ob, field)[:, a][::-1]`.
- In `_exterior_tracer_full` and `exterior_tracer`, add the keyword `field=None` and pass it through to the source-columns call.
- Update both docstrings: "``field``: None = T, i = passive tracer i".

- [ ] **Step 4: Implement restart and output**

`core/restart.py`:
- `_config`, inside the ALE `cfg.update(...)`: add `n_tracers=int(getattr(model, "n_tracers", 0))`.
- `write_restart`, after the `h_ext` block:

```python
    if getattr(state, "tr", None) is not None:
        data["tr"] = xr.Variable(("tracer", "layer", "y", "x"), np.asarray(state.tr, np.float64),
                                 {"units": "1", "long_name": "passive tracers (ALE)"})
    if getattr(model, "tr_ext", None) is not None:
        data["tr_ext"] = xr.Variable(("tracer", "layer", "y", "x"), np.asarray(model.tr_ext, np.float64),
                                     {"units": "1", "long_name": "external passive tracers (kept in z with h_ext)"})
```

- `read_restart`: inside the ALE block, add `if "tr_ext" in ds: model.tr_ext = ds["tr_ext"].values.copy()`. In the returned `SigmaState(...)`, add `tr=ds["tr"].values.copy() if "tr" in ds else None`.

`diagnostics/output.py`, at the top of `_record`'s loop:

```python
            if name not in FIELDS and re.fullmatch(r"tr\d+", name):
                FIELDS[name] = ("t3", "1", None, f"passive tracer {int(name[2:])}")
```

Also add `import re`. Document the `trN` names in the class docstring.

- [ ] **Step 5: Run the new tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k "tracer" tests/test_restart.py tests/test_output.py -q -p no:cacheprovider`
Expected: PASS (all).

If the oracle fails for the hybrid, check that `fields` passes T_lag at index 0 and that the regrid uses `T_lag` only. The oracle must be exact; do not loosen it.

- [ ] **Step 6: Full suite, fingerprints unchanged**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider > $WS/suite_t1.txt 2>&1; tail -2 $WS/suite_t1.txt`
Expected: `1358 passed, 2 xfailed` (1348 + 10 new). The fingerprint tests pass unchanged.

- [ ] **Step 7: Commit**

```bash
git add core/dynamics/sigma_baroclinic.py boundaries/obcs/__init__.py core/dynamics/open_boundary.py core/restart.py diagnostics/output.py tests/test_sigma_ale.py tests/test_restart.py tests/test_output.py
git commit -m "ALE: passive tracers (n_tracers) in the advection stack, remap, open boundaries, sponge and restart

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 2: Interior sponge (`sponge_rate`, `sponge_T`, `sponge_zw`, `sponge_tr`)

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py`:
  - `__init__` (signature + body);
  - `_step_ale`, after `_regrid_remap`;
  - a new `_interior_sponge`.
- Modify: `core/restart.py` (`_config`)
- Test: `tests/test_sigma_ale.py`, `tests/test_restart.py`

**Interfaces:**
- Consumes: Task 1 (`st.tr`, `n_tracers`).
- Produces:
  - `SigmaBaroclinic(..., sponge_rate: np.ndarray | None = None, sponge_T: np.ndarray | None = None, sponge_zw: np.ndarray | None = None, sponge_tr: np.ndarray | None = None)`;
  - `sponge_rate` is (ny, nx) [s⁻¹];
  - `sponge_T` is (nz_s,) [°C] on the interfaces `sponge_zw` (nz_s+1,) [m], 0 at the surface, decreasing;
  - `sponge_tr` is (n_tracers, nz_s), default 0.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_sigma_ale.py`)

```python
# ---------------------------------------------------------------- interior sponge (DOME spec §2)
def _sponge_box(rate=1e-4, rule="zstar", **kw):
    nx, nz = 16, 6
    zw = np.linspace(0.0, -20.0, 5)
    target = np.full(4, 10.0)                                          # uniform: the remap of T* is exact
    r = np.zeros((1, nx))
    r[:, :4] = rate
    m = _ale_box(rule, nx=nx, nz=nz, depth=20.0, dt=60.0, sponge_rate=r, sponge_T=target, sponge_zw=zw, **kw)
    return m, m.initial_state(np.zeros((1, nx)), np.full((nz, 1, nx), 15.0))


def test_interior_sponge_relaxes_T_at_the_implicit_rate_and_leaves_the_rest_alone():
    m, st = _sponge_box()
    for _ in range(10):
        st = m.step(st)
    expected = 10.0 + 5.0 / (1.0 + 60.0 * 1e-4) ** 10                   # (1 + Δt r)⁻ⁿ toward T* = 10
    np.testing.assert_allclose(st.T[:, 0, 0], expected, rtol=1e-10)      # inside the band (neighbours alike)
    np.testing.assert_allclose(st.T[:, 0, 8:], 15.0, atol=1e-10)          # outside: WENO stencils see only 15


def test_interior_sponge_with_zero_rate_is_bit_identical_to_none():
    m0, s0 = _sponge_box(rate=0.0)
    m1 = _ale_box("zstar", nx=16, nz=6, depth=20.0, dt=60.0)
    s1 = m1.initial_state(np.zeros((1, 16)), np.full((6, 1, 16), 15.0))
    for _ in range(5):
        s0, s1 = m0.step(s0), m1.step(s1)
    assert np.array_equal(s0.T, s1.T) and np.array_equal(s0.h, s1.h)


def test_interior_sponge_ignores_land_and_checks_shapes():
    nx = 16
    H = np.full((1, nx), 20.0)
    H[0, :2] = np.nan
    r = np.full((1, nx), 1e-4)
    m = _ale_box("zstar", nx=nx, nz=6, bathy=H, dt=60.0, sponge_rate=r, sponge_T=np.full(4, 10.0),
                 sponge_zw=np.linspace(0.0, -20.0, 5))
    st = m.step(m.initial_state(np.zeros((1, nx)), np.full((6, 1, nx), 15.0)))
    assert np.isfinite(st.T[:, 0, 2:]).all()
    with pytest.raises(ValueError, match=r"sponge_rate shape"):
        _ale_box("zstar", nx=nx, nz=6, dt=60.0, sponge_rate=np.zeros((2, nx)), sponge_T=np.full(4, 10.0),
                 sponge_zw=np.linspace(0.0, -20.0, 5))


def test_interior_sponge_relaxes_passive_tracers_toward_their_target():
    m, st = _sponge_box(n_tracers=1)
    st.tr = np.ones_like(st.tr)
    for _ in range(10):
        st = m.step(st)
    np.testing.assert_allclose(st.tr[0, :, 0, 0], 1.0 / (1.0 + 60.0 * 1e-4) ** 10, rtol=1e-10)
    np.testing.assert_allclose(st.tr[0, :, 0, 8:], 1.0, atol=1e-10)
```

Append to `tests/test_restart.py`:

```python
def test_ale_restart_refuses_another_interior_sponge(tmp_path):
    from tests.test_sigma_ale import _sponge_box
    m, st = _sponge_box(rate=1e-4)
    write_restart(tmp_path / "rst.nc", m, m.step(st))
    m2, _ = _sponge_box(rate=2e-4)
    with pytest.raises(ValueError, match="sponge_rate"):
        read_restart(tmp_path / "rst.nc", m2)
```

- [ ] **Step 2: Run to verify they fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k interior_sponge tests/test_restart.py -k interior_sponge -q -p no:cacheprovider`
Expected: FAIL with "unexpected keyword argument 'sponge_rate'".

- [ ] **Step 3: Implement**

Add `sponge_rate=None, sponge_T=None, sponge_zw=None, sponge_tr=None` to the `__init__` keywords, and to the docstring:

```
    sponge_rate, sponge_T, sponge_zw, sponge_tr : ALE interior sponge (spec 2026-10-14-dome §2), independent of the open
        boundaries: after the regrid–remap of each step T ← (T + Δt r T*)/(1 + Δt r), implicit as MOM6 apply_sponge,
        with T* the profile sponge_T on the interfaces sponge_zw (surface first, [m] ≤ 0) remapped onto the current
        layers (core.vertical.ale.exterior_on_layers); passive tracers toward sponge_tr (default 0); r =
        sponge_rate (ny, nx) [s⁻¹], ignored on land; momentum is not damped (as MOM6 DOME_initialize_sponges)
```

At the end of `__init__` (after Task 1's lines):

```python
        self.sponge_rate = None
        if sponge_rate is not None:
            if not self.ale:
                raise ValueError("sponge_rate (interior sponge) needs vertical_coordinate='ale'")
            r = np.asarray(sponge_rate, float)
            if r.shape != self.H.shape:
                raise ValueError(f"sponge_rate shape {r.shape}, expected {self.H.shape}")
            zw = np.asarray(sponge_zw, float)
            Ts = np.asarray(sponge_T, float)
            if zw.ndim != 1 or Ts.shape != (zw.size - 1,) or not (np.diff(zw) < 0).all():
                raise ValueError("sponge_T (nz_s,) needs sponge_zw (nz_s + 1,) decreasing from the surface")
            self.sponge_rate = np.where(self.fs.active, np.nan_to_num(r), 0.0)
            self._sp_zw, self._sp_T = zw, Ts
            self._sp_tr = (np.zeros((self.n_tracers, Ts.size)) if sponge_tr is None
                           else np.asarray(sponge_tr, float).reshape(self.n_tracers, Ts.size))
            self._sp_cols = self.sponge_rate > 0
```

Add a method next to `_exterior_tr`:

```python
    def _interior_sponge(self, T: np.ndarray, tr: np.ndarray | None, eta: np.ndarray, h: np.ndarray):
        """
        Implicit relaxation toward the sponge profile on the layers h (spec 2026-10-14-dome §2):
            T ← (T + Δt r T*)/(1 + Δt r)   (MOM6 MOM_sponge.F90 apply_sponge, implicit form)
        T* is sponge_T on its z interfaces, stretched to each column's total and remapped (as the exterior profile).
        """
        cols = self._sp_cols
        H = np.nan_to_num(np.asarray(self.H, float))
        D = np.where(cols, H + eta, 0.0)
        z = np.maximum(self._sp_zw[:, None, None], -H[None])                # clipped to each bed, surface first
        z[-1] = -H
        hs = np.ascontiguousarray((z[:-1] - z[1:])[::-1] * np.ones((1,) + H.shape))   # bed first
        numba.set_num_threads(_THREADS[0])
        fill = lambda prof: np.ascontiguousarray(prof[::-1, None, None] * np.ones((1,) + H.shape))  # noqa: E731
        r = self.dt * self.sponge_rate[None]
        Tt = exterior_on_layers(hs, fill(self._sp_T), h, cols & (D > 0), self.remap_code)
        T = np.where(cols[None], (T + r * Tt) / (1.0 + r), T)
        if tr is not None:
            trt = [exterior_on_layers(hs, fill(p), h, cols & (D > 0), self.remap_code) for p in self._sp_tr]
            tr = np.stack([np.where(cols[None], (c + r * t) / (1.0 + r), c) for c, t in zip(tr, trt)])
        return T, tr
```

`exterior_on_layers` stretches the source column to the model column total (`core/vertical/ale.py:857-871`); keep that behaviour. Read its docstring first to confirm the argument order `(h_src, f_src, h, cols, scheme)`.

In `_step_ale`, right after `tr_new = ...` (Task 1), add:

```python
        if self.sponge_rate is not None:                               # interior sponge (spec 2026-10-14-dome §2)
            T_new, tr_new = self._interior_sponge(T_new, tr_new, eta_new, h_new)
```

The sponge goes after the remap, so the mixing diagnostic (which uses `rem[1]`) measures advection and remap only. Document this in the docstring line.

`core/restart.py` `_config` (ALE block):

```python
        sp = getattr(model, "sponge_rate", None)
        cfg.update(sponge_rate="none" if sp is None else _text(sp),
                   sponge_T="none" if sp is None else _text(model._sp_T),
                   sponge_zw="none" if sp is None else _text(model._sp_zw))
```

- [ ] **Step 4: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k interior_sponge tests/test_restart.py -q -p no:cacheprovider`
Expected: PASS.

The target is uniform on purpose: then the remap of T* is exact and the test checks the implicit rate only. If it fails, check that the sponge runs after `_regrid_remap`, and that `exterior_on_layers` gets the bed-first `hs`.

- [ ] **Step 5: Full suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider > $WS/suite_t2.txt 2>&1; tail -2 $WS/suite_t2.txt`
Expected: `1363 passed, 2 xfailed`.

- [ ] **Step 6: Commit**

```bash
git add core/dynamics/sigma_baroclinic.py core/restart.py tests/test_sigma_ale.py tests/test_restart.py
git commit -m "ALE: interior sponge (sponge_rate/T/zw/tr), implicit, independent of the open boundaries

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 3: Quadratic bottom drag in ALE (`bottom_drag="quadratic"`, `cd`, `drag_bg_vel`, `hbbl`)

**Files:**
- Modify: `core/dynamics/sigma_wd.py`:
  - `column_step` (187–253) gains `r_bed`;
  - `step` (266–305) gains `r_u`, `r_v`.
- Modify: `core/dynamics/sigma_baroclinic.py`:
  - `__init__` keywords and docstring;
  - a new `_bed_drag_rates`;
  - `_step_ale`, where `visc` is built (1997–2006).
- Modify: `core/restart.py` (`_config`)
- Test: `tests/test_sigma_wd.py`, `tests/test_sigma_ale.py`, `tests/test_restart.py`

**Interfaces:**
- Produces:
  - `SigmaWetDry.column_step(..., r_bed: np.ndarray | None = None)`: a given linear bed rate r [m s⁻¹] per column overrides the log-law;
  - `SigmaWetDry.step(..., r_u=None, r_v=None)`;
  - `SigmaBaroclinic(..., bottom_drag: str | None = None, cd: float = 2.5e-3, drag_bg_vel: float = 0.0, hbbl: float = 10.0)`;
  - `model._bed_drag_rates(u, v, hu, hv) -> (r_u, r_v)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sigma_ale.py`:

```python
# ---------------------------------------------------------------- quadratic bed drag in ALE (DOME spec §3)
def _drag_channel(rule="zstar", U0=0.5, cd=2e-3, nz=6, depth=50.0, **kw):
    """Homogeneous closed channel, uniform flow U0 at t = 0 (no pressure gradient: uniform η, uniform T)."""
    nx = 12
    m = _ale_box(rule, nx=nx, nz=nz, depth=depth, dt=60.0, bottom_drag="quadratic", cd=cd, nu=1e-2, **kw)
    st = m.initial_state(np.zeros((1, nx)), np.full((nz, 1, nx), 10.0))
    u = np.zeros_like(st.u)
    u[..., 1:-1] = U0
    st.u = u
    return m, st


def _ubar(st, i=6):
    h = 0.5 * (st.h[:, 0, i - 1] + st.h[:, 0, i])
    return float((st.u[:, 0, i] * h).sum() / h.sum())


def test_quadratic_drag_removes_the_bed_stress_from_the_depth_mean():
    # closed channel: walls make the flow slosh, so the drag is isolated by differencing against cd = 0;
    # one step: Δū = −Δt C_d |u_b| u_b / D (implicit factor 1/(1 + Δt r/D) ≈ 1 − 1e-6 here)
    a = _drag_channel(cd=0.0)
    b = _drag_channel(cd=2e-3)
    ua, ub = _ubar(a[0].step(a[1])), _ubar(b[0].step(b[1]))
    np.testing.assert_allclose(ub - ua, -60.0 * 2e-3 * 0.5 * 0.5 / 50.0, rtol=0.02)


def test_quadratic_drag_reaches_the_flow_above_vanished_bed_layers():
    # z* with 10 levels over 20 m in a 7-m column: the bed layers vanish; the drag must still act on the water
    nx, nz = 12, 10
    H = np.full((1, nx), 7.0)
    out = []
    for cd in (0.0, 2e-3):
        m = _ale_box("zstar", nx=nx, nz=nz, bathy=H, params=dict(dz_top=np.full(nz, 2.0)), dt=60.0,
                     bottom_drag="quadratic", cd=cd, nu=1e-2)
        st = m.initial_state(np.zeros((1, nx)), np.full((nz, 1, nx), 10.0))
        st.u = np.where(np.arange(nx + 1)[None, None] % nx == 0, 0.0, 0.3) * np.ones_like(st.u)
        for _ in range(20):
            st = m.step(st)
        assert np.isfinite(st.u).all()
        out.append(_ubar(st))
    expected = 20 * 60.0 * 2e-3 * 0.3**2 / 7.0                           # ≈ 0.031 m/s removed by the drag
    assert out[0] - out[1] > 0.5 * expected


def test_bottom_drag_off_is_bit_identical_and_background_velocity_alone_does_not_accelerate():
    m0 = _ale_box("zstar", nx=12, nz=6, depth=50.0, dt=60.0, nu=1e-2)
    m1, _ = _drag_channel(U0=0.0, drag_bg_vel=0.05)
    s0 = m0.initial_state(np.zeros((1, 12)), np.full((6, 1, 12), 10.0))
    s1 = m1.initial_state(np.zeros((1, 12)), np.full((6, 1, 12), 10.0))
    for _ in range(10):
        s0, s1 = m0.step(s0), m1.step(s1)
    assert np.array_equal(s1.u, s0.u)
    with pytest.raises(ValueError, match="bottom_drag"):
        _ale_box("zstar", nx=12, nz=6, depth=50.0, dt=60.0, bottom_drag="loglaw")


def test_bed_drag_rates_are_finite_on_empty_faces():
    # dry or fully vanished face columns: no bbl water → only the background velocity sets r
    m, st = _drag_channel(drag_bg_vel=0.05)
    ru, rv = m._bed_drag_rates(st.u, st.v, np.zeros_like(st.u), np.zeros_like(st.v))
    np.testing.assert_allclose(ru, 2e-3 * 0.05, rtol=1e-14)
    np.testing.assert_allclose(rv, 2e-3 * 0.05, rtol=1e-14)
```

Append to `tests/test_sigma_wd.py`:

```python
def test_column_step_with_a_given_bed_rate_matches_the_implicit_formula():
    from core.dynamics.sigma_wd import SigmaWetDry
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=3, ny=1, dx=100.0, dy=100.0)
    m = SigmaWetDry(hg, np.full((1, 3), 10.0), dt=10.0, n_layers=1, z0=None, closure="none")
    u = np.full((1, 4), 1.0)
    out = m.column_step(u, None, np.full(4, 10.0), r_bed=np.full(4, 0.01))
    np.testing.assert_allclose(out, 1.0 / (1.0 + 10.0 * 0.01 / 10.0), rtol=1e-14)   # (1 + Δt r / h)⁻¹
```

Append to `tests/test_restart.py`:

```python
def test_ale_restart_refuses_another_bottom_drag(tmp_path):
    from tests.test_sigma_ale import _drag_channel
    m, st = _drag_channel()
    write_restart(tmp_path / "rst.nc", m, m.step(st))
    m2, _ = _drag_channel(cd=3e-3)
    with pytest.raises(ValueError, match="cd"):
        read_restart(tmp_path / "rst.nc", m2)
```

- [ ] **Step 2: Run to verify they fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k "drag" tests/test_sigma_wd.py -k bed_rate tests/test_restart.py -k bottom_drag -q -p no:cacheprovider`
Expected: FAIL with "unexpected keyword argument 'bottom_drag'" / "'r_bed'".

- [ ] **Step 3: Implement in `sigma_wd.py`**

`column_step`:
- add the keyword `r_bed: np.ndarray | None = None`;
- add to the docstring: "r_bed : (...) linear bed-stress rate r = C_d |u_b| [m s⁻¹] given by the caller (ALE quadratic drag); overrides the log-law of ``z0``";
- replace the `if self.z0 is None:` block with:

```python
        if r_bed is not None:
            r = np.nan_to_num(np.asarray(r_bed, float).reshape(-1))
            ustar = np.zeros(D.size)
        elif self.z0 is None:
            r = np.zeros(D.size)
            ustar = np.zeros(D.size)
        else:
```

- the `closure == "none" and self.z0 is None` shortcut must also require `r_bed is None`:

```python
        if self.closure == "none" and self.z0 is None and r_bed is None:
```

`step`: add `r_u=None, r_v=None` to the signature and docstring ("linear bed rates at u-/v-faces [m s⁻¹], ALE quadratic drag"), and pass `r_bed=r_u` / `r_bed=r_v` to the two `column_step` calls.

- [ ] **Step 4: Implement in `sigma_baroclinic.py`**

Add the keywords `bottom_drag: str | None = None, cd: float = 2.5e-3, drag_bg_vel: float = 0.0, hbbl: float = 10.0`, and the docstring entry:

```
    bottom_drag, cd, drag_bg_vel, hbbl : ALE bed stress (spec 2026-10-14-dome §3): None (default, free slip) or
        'quadratic': τ_b/ρ₀ = −r u_b, r = C_d √(ū_bbl² + v̄_bbl² + u_bg²), with ū_bbl the thickness-weighted mean of the
        face column over the bottom ``hbbl`` metres (MOM6 HBBL, MOM_set_viscosity.F90) — a vanished bed layer does
        not set the drag alone; applied implicitly at the bed of the coupled viscous solve (the depth-mean correction
        δ of SigmaWetDry.column_step includes it).  Legg et al. 2006: C_d = 2e-3
```

At the end of `__init__`:

```python
        if bottom_drag not in (None, "quadratic"):
            raise ValueError(f"unknown bottom_drag {bottom_drag!r}; use None or 'quadratic'")
        if bottom_drag and not self.ale:
            raise ValueError("bottom_drag='quadratic' needs vertical_coordinate='ale' (σ uses z0, the log-law)")
        self.bottom_drag, self.cd, self.drag_bg_vel, self.hbbl = bottom_drag, float(cd), float(drag_bg_vel), float(hbbl)
```

New method:

```python
    def _bed_drag_rates(self, u: np.ndarray, v: np.ndarray, hu: np.ndarray, hv: np.ndarray):
        """
        Linear bed rates r = C_d √(ū_bbl² + v̄_bbl² + u_bg²) [m s⁻¹] at u- and v-faces (spec 2026-10-14-dome §3).
        ū_bbl: mean of the face column over its bottom hbbl metres, weights min(h_k, hbbl − Σ_{j<k} h_j)⁺ (k = 0 at
        the bed); the other component is the 4-point C-grid average of its own bbl means.
        """
        def bbl_mean(vel, h):
            below = np.concatenate([np.zeros((1,) + h.shape[1:]), np.cumsum(h, axis=0)[:-1]])
            w = np.clip(np.minimum(h, self.hbbl - below), 0.0, None)
            tot = w.sum(axis=0)
            return np.where(tot > 0, (w * vel).sum(axis=0) / np.where(tot > 0, tot, 1.0), 0.0)
        ub, vb = bbl_mean(u, hu), bbl_mean(v, hv)
        vc = 0.5 * (vb[:-1] + vb[1:])                       # v̄ at cell centres (ny, nx)
        uc = 0.5 * (ub[:, :-1] + ub[:, 1:])                 # ū at cell centres (ny, nx)
        v_at_u = np.zeros_like(ub)
        v_at_u[:, 1:-1] = 0.5 * (vc[:, :-1] + vc[:, 1:])
        u_at_v = np.zeros_like(vb)
        u_at_v[1:-1, :] = 0.5 * (uc[:-1, :] + uc[1:, :])
        bg2 = self.drag_bg_vel**2
        return (self.cd * np.sqrt(ub**2 + v_at_u**2 + bg2), self.cd * np.sqrt(vb**2 + u_at_v**2 + bg2))
```

In `_step_ale`, just before `eta_new, u_new, v_new = self.core.step(...)`:

```python
        if self.bottom_drag == "quadratic":                            # bed stress at uⁿ (spec 2026-10-14-dome §3)
            r_u, r_v = self._bed_drag_rates(st.u, st.v, hu, hv)
            visc.update(r_u=r_u, r_v=r_v)
            if "hu_mean" not in visc:                                  # δ needs the depth-mean column with drag
                visc.update(hu_mean=hu, hv_mean=hv)
```

`r_u` is (ny, nx+1) and `column_step` reshapes `r_bed` to `(-1,)` over the face columns, so the order matches `u.reshape(n, -1)`.

`core/restart.py` `_config` ALE block: `bottom_drag=str(getattr(model, "bottom_drag", None)), cd=model.cd, drag_bg_vel=model.drag_bg_vel, hbbl=model.hbbl`.

- [ ] **Step 5: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k drag tests/test_sigma_wd.py tests/test_restart.py -q -p no:cacheprovider`
Expected: PASS.

In the first test, Δū ≈ −6.0e-4 m/s; 2 % covers the implicit factor and the small difference in the free-surface response. If it is off by a factor, check that `hbbl = 10 m` covers the bottom layers (50 m / 6 = 8.3 m, so 2 layers), and that the δ correction runs, i.e. `hu_mean` is passed.

- [ ] **Step 6: Full suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider > $WS/suite_t3.txt 2>&1; tail -2 $WS/suite_t3.txt`
Expected: `1369 passed, 2 xfailed`.

- [ ] **Step 7: Commit**

```bash
git add core/dynamics/sigma_wd.py core/dynamics/sigma_baroclinic.py core/restart.py tests/test_sigma_wd.py tests/test_sigma_ale.py tests/test_restart.py
git commit -m "ALE: quadratic bottom drag over the bottom hbbl metres (bottom_drag, cd, drag_bg_vel, hbbl)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 4: Exact prescribed transport at open edges (`OBData.transport`)

**Files:**
- Modify: `boundaries/obcs/__init__.py` (OBData)
- Modify: `core/dynamics/open_boundary.py` (`validate_state`, `flather_targets` 195–209)
- Modify: `core/dynamics/wetting_drying.py` (`WetDryFreeSurface.step` 228–340)
- Modify: `core/dynamics/sigma_wd.py` (`SigmaWetDry.step`: pass `flather_q` through)
- Test: `tests/test_sigma_ale.py`, `tests/test_wetting_drying.py`

**Interfaces:**
- Produces:
  - `OBData.transport: np.ndarray | None`, shape (n_bdy,) [m³ s⁻¹ per face], in the model sign (positive toward +x/+y).
  - `flather_targets(...)` also returns `flather_q=(qu, qv)`: NaN where not prescribed.
  - `WetDryFreeSurface.step(..., flather_q=None)` and `SigmaWetDry.step(..., flather_q=None)`.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_sigma_ale.py`:

```python
# ---------------------------------------------------------------- prescribed inflow transport (DOME spec §4)
def _inflow_box(Q=-50.0, H=None):
    """Closed 2-D box (x, y = 3 rows) fed through its north edge: 'clamped' with an exact transport."""
    from boundaries.obcs import OBData
    nx, ny, nz = 6, 3, 4
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.full((ny, nx), 20.0) if H is None else H
    zw = np.linspace(0.0, -20.0, nz + 1)[:, None] * np.ones((1, nx))

    def state(t):
        q = np.full(nx, Q)
        return {"north": OBData(side="north", T=np.full((nz, nx), 9.0), u_normal=np.full((nz, nx), Q / 20000.0),
                                zw=zw, transport=q)}
    m = SigmaBaroclinic(hg, H, dt=30.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant",
                        nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 5.0)), open_boundaries=("north",),
                        open_boundary_mode={"north": "clamped"}, open_boundary_state=state)
    return m, m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 10.0))


def test_prescribed_transport_enters_exactly_and_the_volume_grows_by_it():
    m, st = _inflow_box()
    V0 = float(np.nansum((m.H + st.eta) * m.area))
    for _ in range(20):
        st = m.step(st)
        np.testing.assert_allclose(st.Qy[:, -1, :].sum(axis=0), -50.0, rtol=1e-12)
    V1 = float(np.nansum((m.H + st.eta) * m.area))
    np.testing.assert_allclose(V1 - V0, 20 * 30.0 * 6 * 50.0, rtol=1e-10)


def test_prescribed_transport_skips_faces_next_to_land():
    H = np.full((3, 6), 20.0)
    H[-1, :2] = np.nan
    m, st = _inflow_box(H=H)
    for _ in range(5):
        st = m.step(st)
    assert np.all(st.Qy[:, -1, :2] == 0.0)
    np.testing.assert_allclose(st.Qy[:, -1, 2:].sum(axis=0), -50.0, rtol=1e-12)
```

Append to `tests/test_wetting_drying.py`:

```python
def test_flather_face_with_a_prescribed_transport_has_no_radiation_term():
    from core.dynamics.wetting_drying import WetDryFreeSurface
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=3, ny=2, dx=100.0, dy=100.0)
    fl_v = np.zeros((3, 3), bool)
    fl_v[-1, :] = True
    fs = WetDryFreeSurface(hg, np.full((2, 3), 10.0), dt=10.0, theta=0.6, flather_v=fl_v)
    eta = np.full((2, 3), 0.3)                                    # η ≠ η_ext: plain Flather would drive a flux
    z = (np.zeros((2, 4)), np.zeros((3, 3)))
    q = (np.full((2, 4), np.nan), np.where(fl_v, -7.0, np.nan))
    fs.step(eta, np.zeros((2, 4)), np.zeros((3, 3)), np.zeros((2, 4)), np.zeros((3, 3)), flather_eta=z,
            flather_ubar=np.zeros((2, 4)), flather_vbar=np.zeros((3, 3)), flather_q=q)
    np.testing.assert_allclose(fs.last_open_Qv[-1, :], -7.0, rtol=1e-14)
```

- [ ] **Step 2: Run to verify they fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k prescribed_transport tests/test_wetting_drying.py -k prescribed -q -p no:cacheprovider`
Expected: FAIL. OBData gets an unexpected keyword 'transport', and `step()` an unexpected keyword 'flather_q'.

- [ ] **Step 3: Implement**

`OBData`: add `transport: np.ndarray | None = field(default=None)`, with the docstring entry "transport : (n_bdy,) prescribed normal volume flux per face [m³ s⁻¹], model sign; replaces Flather there (MOM6 SIMPLE segment): the free surface takes exactly this flux, with no radiation term".

`validate_state`: add `"transport"` to the `("eta", "ubar")` shape loop.

`flather_targets`:
- start from `qu, qv = np.full(shape_u, np.nan), np.full(shape_v, np.nan)`;
- for each edge with `ob.transport is not None`, fill its face line the same way as `ubar`;
- return `dict(..., flather_q=(qu, qv))`.

`WetDryFreeSurface.step(..., flather_q=None)`:
- docstring: "flather_q : (q_u, q_v) prescribed volume flux at Flather faces [m³ s⁻¹], NaN where Flather applies".
- In the Flather setup loop, after `ub = ...`:

```python
                qp = None if flather_q is None else np.asarray(flather_q[0] if fl["axis"] == "u" else flather_q[1])[fl["face"]]
                fixed = np.zeros(e_ext.shape, bool) if qp is None else np.isfinite(qp)
                c = np.where(open_ & ~fixed, fl["length"] * np.sqrt(g * D), 0.0)
                out_q = np.where(fixed, fl["sign"] * np.nan_to_num(qp), fl["length"] * D * fl["sign"] * ub)
                known = np.where(open_, out_q + c * ((1 - th) * eta[fl["cell"]] - e_ext), 0.0)
```

  (this replaces the existing `c = ...` and `known = ...` lines).
- Append `out_q` to `fl_state`: `fl_state.append((D, open_, c, e_ext, ub, out_q))`.
- In the post-solve loop, unpack `(D, open_, c, e_ext, ub, out_q)` and replace both occurrences of `fl["length"] * D * fl["sign"] * ub` with `out_q`.

`SigmaWetDry.step`: add `flather_q=None` and pass it to `self.fs.step(...)`.

`SigmaBaroclinic._flather_args` already returns `self._ext["flather"]`, which now carries `flather_q`; the resting case gives no key, which is fine.

- [ ] **Step 4: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k prescribed_transport tests/test_wetting_drying.py tests/test_free_surface.py -q -p no:cacheprovider`
Expected: PASS.

If the layer sum is not −50 to 1e-12, check `_open_face_transports`: it scales the layers to `last_open_Qv`, which is now exactly the prescribed q because `c = 0` there.

- [ ] **Step 5: Full suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider > $WS/suite_t4.txt 2>&1; tail -2 $WS/suite_t4.txt`
Expected: `1372 passed, 2 xfailed`.

- [ ] **Step 6: Commit**

```bash
git add boundaries/obcs/__init__.py core/dynamics/open_boundary.py core/dynamics/wetting_drying.py core/dynamics/sigma_wd.py tests/test_sigma_ale.py tests/test_wetting_drying.py
git commit -m "Open boundaries: exact prescribed transport (OBData.transport) instead of Flather on chosen faces

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 5: DOME case definition (`validation/dome/cases.py`)

**Files:**
- Create: `validation/dome/__init__.py` (empty)
- Create: `validation/dome/cases.py`
- Test: `tests/test_dome_benchmark.py`

**Interfaces:**
- Produces:
  - `cases.Case`, a dataclass with: `name, dx, nx, ny, nz, x, y, H, f0, dt, days, out_s, nu_h, nu_v, cd, hbbl`;
  - `cases.get(res_km: float) -> Case`;
  - `cases.T_ambient(z) -> T` [°C], with z ≤ 0 [m];
  - `cases.inflow_profile(case, nz_bdy=60) -> dict(zw (nz+1, nx), T (nz, nx), v (nz, nx), s (nz, nx), transport (nx,))`, on the north edge;
  - `cases.sponge_rate(case) -> (ny, nx)` [s⁻¹];
  - `cases.mom6_layer_transports() -> (25,)` [m³ s⁻¹];
  - `cases.targets(nz) -> (nz+1,)` density targets;
  - constants `RHO0 = 1031.0`, `G = 9.8`, `ALPHA = 0.2`, `T_REF = 0.0`, `DRHO = 2.0`, `H0 = 300.0`, `HE = 600.0`, `RI = 1/3`;
  - `run_dir(res_km, model, coord, tag="") -> Path` under `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/`.

- [ ] **Step 1: Write the failing tests** (`tests/test_dome_benchmark.py`)

```python
"""DOME benchmark pipeline (spec docs/superpowers/specs/2026-10-14-dome-overflow-design.md §5–§7)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.dome import cases


def test_bathymetry_matches_mom6_dome_initialize_topography():
    c = cases.get(10.0)
    assert (c.nx, c.ny) == (150, 70)
    j_shelf = np.searchsorted(c.y, 450e3)                            # mid-slope: 600 + 0.01·(600 − 450) km
    np.testing.assert_allclose(c.H[j_shelf, 0], 600.0 + 0.01 * (600e3 - c.y[j_shelf]), rtol=1e-12)
    assert np.all(c.H[c.y < 300e3] == 3600.0)
    north = c.y >= 600e3
    chan = (c.x > 1000e3) & (c.x < 1100e3)
    assert np.all(c.H[np.ix_(north, chan)] == 600.0)
    assert np.all(np.isnan(c.H[np.ix_(north, ~chan)]))


def test_inflow_transport_is_legg_and_close_to_mom6_by_density_class():
    c = cases.get(10.0)
    p = cases.inflow_profile(c)
    db0 = cases.G * cases.DRHO / cases.RHO0
    legg = db0 * cases.H0**2 / (2 * c.f0)                            # Legg 2006 §2: T_in = Δb₀ h₀² / (2f) ≈ 8.55 Sv
    assert abs(-p["transport"].sum() - legg) / legg < 0.05
    rho = cases.RHO0 - cases.ALPHA * (p["T"] - cases.T_REF)
    dz = -np.diff(p["zw"], axis=0)
    flux = -(p["v"] * dz * c.dx).sum(axis=1)                           # per row of the profile
    dense = (flux * (rho.mean(axis=1) > 1031.9)).sum()
    mom6 = cases.mom6_layer_transports()
    assert abs(dense - (-mom6[-1])) / (-mom6[-1]) < 0.25              # densest class: 7.45 Sv in MOM6 (diff. (c))


def test_inflow_profile_is_geostrophic_and_ri_bounded():
    c = cases.get(10.0)
    p = cases.inflow_profile(c)
    i = np.argmax(-p["transport"])                                    # the right-hand (west) wall of the channel
    assert p["s"][:, i].max() == 1.0 and p["s"][:, i].min() == 0.0
    assert np.all(np.diff(p["T"][:, i]) >= -1e-12)                    # stable: T increases upward (surface first ↓)
    assert np.all(p["v"] <= 0.0)                                       # southward into the domain


def test_sponge_profile_matches_mom6():
    c = cases.get(10.0)
    r = cases.sponge_rate(c)
    peak = 10.0 / 86400.0
    assert r[0, 0] == pytest.approx(peak) and r[0, 75] == 0.0
    i150 = np.argmin(np.abs(c.x - 150e3))                              # half way down the ramp
    assert r[0, i150] == pytest.approx(peak * 0.5, rel=0.1)
```

- [ ] **Step 2: Run to verify they fail**

Run: `python -m pytest tests/test_dome_benchmark.py -q -p no:cacheprovider`
Expected: FAIL with "No module named 'validation.dome'".

- [ ] **Step 3: Implement `validation/dome/cases.py`**

```python
"""
DOME overflow, Case 1 (Ref) of Legg, Hallberg & Girton (2006), Ocean Modell. 11, 69–97 (§2, eq. 3–8, tables 1–3), in
the geometry of MOM6's implementation (src/user/DOME_initialization.F90, Hallberg) — one definition for pyOCEAN,
MOM6 and MITgcm (spec docs/superpowers/specs/2026-10-14-dome-overflow-design.md §5).  x east, y north [m]; the
inflow enters through the north edge of a 100-km channel at 1000 < x < 1100 km and turns west along the slope.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

RUNS = Path("/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome")
RHO0, G = 1031.0, 9.8                       # MOM6 DOME MOM_input (RHO_0, G_EARTH)
ALPHA, T_REF = 0.2, 0.0                     # linear EOS ρ = ρ₀ − α (T − T_ref)
DRHO, H0, HE, HMAX, RI = 2.0, 300.0, 600.0, 3600.0, 1.0 / 3.0   # Legg tables 1–2, MOM6 DOME defaults
RHO_TOP = 1030.0                             # surface density (MOM6 LIGHTEST_DENSITY); bottom 1032


@dataclass
class Case:
    name: str
    dx: float
    nx: int
    ny: int
    nz: int
    x: np.ndarray
    y: np.ndarray
    H: np.ndarray
    f0: float
    dt: float
    days: float
    out_s: float
    nu_h: float
    nu_v: float
    cd: float
    hbbl: float


def T_of_rho(rho):
    """T [°C] of density ρ [kg m⁻³] with the linear EOS."""
    return T_REF + (RHO0 - np.asarray(rho, float)) / ALPHA


def T_ambient(z):
    """Ambient T at height z ≤ 0 [m]: linear density from 1030 at the surface to 1032 at 3600 m (N² const)."""
    return T_of_rho(RHO_TOP + DRHO * np.clip(-np.asarray(z, float), 0.0, HMAX) / HMAX)


def get(res_km: float) -> Case:
    """The case at dx = dy = res_km (10 or 5): MOM6 DOME_initialize_topography (DOME_initialization.F90:97-110)."""
    dx = float(res_km) * 1e3
    nx, ny = int(round(1500e3 / dx)), int(round(700e3 / dx))
    x = (np.arange(nx) + 0.5) * dx
    y = (np.arange(ny) + 0.5) * dx
    X, Y = np.meshgrid(x, y)
    H = np.minimum(HE + 0.01 * (600e3 - Y), HMAX)
    north = Y >= 600e3
    chan = (X > 1000e3) & (X < 1100e3)
    H = np.where(north, np.where(chan, HE, np.nan), H)
    nu_h = 50.0 * dx / 10e3                                        # Legg table 3: 50 m² s⁻¹ at 10 km, ∝ Δx
    return Case(name=f"dome{res_km:g}km", dx=dx, nx=nx, ny=ny, nz=25, x=x, y=y, H=H, f0=1e-4,
                dt=600.0 * dx / 10e3, days=50.0, out_s=86400.0, nu_h=nu_h, nu_v=0.5, cd=2e-3, hbbl=10.0)


def targets(nz: int = 25) -> np.ndarray:
    """Hybrid density targets at the interfaces of MOM6's classes R_k = 1030 + (k − 1)·2/24 (layer centres)."""
    R = RHO_TOP + np.arange(nz) * DRHO / (nz - 1)
    return np.concatenate([[R[0] - 0.5 * (R[1] - R[0])], 0.5 * (R[1:] + R[:-1]), [R[-1] + 0.5 * (R[1] - R[0])]])


def _F(zeta):
    """Legg eq. 8 transition: 1 above, 0 below, ζ/(Ri(ζ+1)) + ½ between (ζ = 0 at the interface, −1 at the bed)."""
    lo, hi = -RI / (2.0 + RI), RI / (2.0 - RI)
    mid = zeta / (RI * (zeta + 1.0)) + 0.5
    return np.where(zeta >= hi, 1.0, np.where(zeta <= lo, 0.0, mid))


def inflow_profile(case: Case, nz_bdy: int = 60) -> dict:
    """
    Inflow on the north edge (Legg eq. 3–7) on a uniform z grid of the 600-m channel, surface first: z depth
    positive down in the formulas (Legg's convention, spec §5), returned as heights zw ≤ 0.
        h = h₀ e^{−x_w/L_ρ},  L_ρ = √(Δb₀h₀)/f,  ζ = (h_e − z − h)/h
        b = min(b₀ − Δb₀(1 − F), b₀ − N² z),   v = −U₀ e^{−x_w/L_ρ}(1 − F),  U₀ = √(Δb₀h₀)
    x_w is the distance from the channel's west wall (the right-hand wall looking downstream, DOME_initialization.F90
    :450-457).  Buoyancy b = −g(ρ − ρ_top)/ρ₀ → T with the linear EOS; s = 1 where the water is denser than the
    ambient at the same depth.  Columns outside the channel: ambient T, v = 0, s = 0, transport 0.
    """
    db0 = G * DRHO / RHO0
    U0 = np.sqrt(db0 * H0)
    Lr = U0 / case.f0
    N2 = db0 / HMAX
    zw = -np.linspace(0.0, HE, nz_bdy + 1)[:, None] * np.ones((1, case.nx))
    zc = 0.5 * (zw[1:] + zw[:-1])
    depth = -zc
    xw = case.x - 1000e3
    chan = (case.x > 1000e3) & (case.x < 1100e3)
    h = H0 * np.exp(-np.clip(xw, 0.0, None) / Lr)
    zeta = (HE - depth - h[None]) / h[None]
    F = _F(zeta)
    b = np.minimum(-db0 * (1.0 - F), -N2 * depth)                     # b₀ = 0 at the surface
    rho = RHO_TOP - RHO0 * b / G
    T = np.where(chan[None], T_of_rho(rho), T_ambient(zc))
    v = np.where(chan[None], -U0 * np.exp(-np.clip(xw, 0.0, None) / Lr)[None] * (1.0 - F), 0.0)
    s = np.where(chan[None] & (b < -N2 * depth - 1e-12), 1.0, 0.0)
    dz = -np.diff(zw, axis=0)
    transport = (v * dz).sum(axis=0) * case.dx
    return dict(zw=zw, T=T, v=v, s=s, transport=transport)


def sponge_rate(case: Case) -> np.ndarray:
    """MOM6 DOME_initialize_sponges (DOME_initialization.F90:218-240): 10 day⁻¹ within 100 km of the W/E walls,
    linear to 0 at 200 km; Idamp = max(W, E); land 0."""
    peak = 10.0 / 86400.0
    def ramp(d):
        return np.where(d < 100e3, peak, np.where(d < 200e3, peak * (200e3 - d) / 100e3, 0.0))
    r = np.maximum(ramp(case.x), ramp(1500e3 - case.x))[None] * np.ones((case.ny, 1))
    return np.where(np.isfinite(case.H), r, 0.0)


def mom6_layer_transports(nz: int = 25) -> np.ndarray:
    """MOM6 DOME_set_OBC_data per-layer transports tr_k [m³ s⁻¹] (DOME_initialization.F90:402-446), Boussinesq."""
    gp = G * DRHO / RHO0
    c = np.sqrt(H0 * gp)
    tr0 = -H0 * c * 0.5 * c / 1e-4
    y = lambda r: (2 * RI * r + RI + 2) / (2 - RI)                     # noqa: E731
    out = np.empty(nz)
    for k in range(1, nz + 1):
        rst = -1.0 if k == 1 else -1.0 + (k - 1.5) / (nz - 1)
        rsb = 0.0 if k == nz else -1.0 + (k - 0.5) / (nz - 1)
        yt, yb = y(rst), y(rsb)
        t = tr0 * 2 / (RI * (2 - RI)) * ((np.log(yt) + 1) / yt - (np.log(yb) + 1) / yb)
        if k == nz:
            t += tr0 * 2 / (RI * (2 + RI)) * np.log((2 + RI) / (2 - RI))
        out[k - 1] = t
    return out


def run_dir(res_km: float, model: str, coord: str, tag: str = "") -> Path:
    """Output directory of one run."""
    return RUNS / f"{res_km:g}km" / f"{model}_{coord}{('_' + tag) if tag else ''}"
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dome_benchmark.py -q -p no:cacheprovider`
Expected: PASS.

If the transport test fails, print `-p["transport"].sum()/1e6`; it should be about 8.5 Sv. The class test's 25 % tolerance covers the (c) difference between Legg's F and MOM6's logarithmic profile. Ledger the measured ratio either way.

- [ ] **Step 5: Commit**

```bash
git add validation/dome/__init__.py validation/dome/cases.py tests/test_dome_benchmark.py
git commit -m "DOME: case definition (Legg 2006 Case 1 in MOM6's geometry): bathymetry, inflow, sponge, targets

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 6: pyOCEAN runner and common series (`validation/dome/run_pyocean.py`, `series.py`)

**Files:**
- Create: `validation/dome/series.py`
- Create: `validation/dome/run_pyocean.py`
- Test: `tests/test_dome_benchmark.py`

**Interfaces:**
- Consumes: Tasks 1–5.
- Produces:
  - `series.KEYS = ("t","x","y","dx","dy","H","h","T","s","u","v","eta")`; `series.save(path, s, meta)`; `series.load(path)`. Layers are surface first.
  - `run_pyocean.build(case, coord, dt=None) -> (model, state)`.
  - `run_pyocean.run(res_km, coord, dt=None, days=None, tag="", threads=5) -> Path`.

- [ ] **Step 1: Write the failing test** (append to `tests/test_dome_benchmark.py`)

```python
@pytest.mark.parametrize("coord", ["zstar", "sigma", "hybrid"])
def test_pyocean_dome_builds_and_takes_steps_with_inflow_tracer_and_sponge(coord):
    from validation.dome import run_pyocean
    c = cases.get(50.0)                                               # 30 × 14 coarse smoke grid
    m, st = run_pyocean.build(c, coord)
    for _ in range(6):
        st = m.step(st)
    assert np.isfinite(st.u).all() and st.tr.max() > 0.0               # dye entered through the channel
    q = st.Qy[:, -1, :].sum()
    np.testing.assert_allclose(q, cases.inflow_profile(c)["transport"].sum(), rtol=1e-10)
```

- [ ] **Step 2: Run to verify it fails**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_dome_benchmark.py -k builds -q -p no:cacheprovider`
Expected: FAIL with "No module named 'validation.dome.run_pyocean'".

- [ ] **Step 3: Implement**

`validation/dome/series.py`: a copy of `validation/ilicak_benchmark/series.py` with `KEYS` extended by `"s"` (shape `(nt, nz, ny, nx)`, like T) and its docstring pointing to the DOME spec §7.

`validation/dome/run_pyocean.py`:

```python
#!/usr/bin/env python
"""
pyOCEAN runs of the DOME benchmark (spec 2026-10-14-dome-overflow §5): ALE z*, σ or hybrid with the model defaults
(vector-invariant momentum, KE Arakawa, WENO5-Z + FCT, PPM_H4, PPM continuity, finite-volume PGF), the common physics
of the case (Laplacian ν_h, ν_v, no tracer diffusion, quadratic drag C_d over hbbl), one passive tracer s (the inflow
dye), the interior sponges of MOM6 DOME and the north inflow as a 'clamped' edge with its exact transport.

Usage: NUMBA_NUM_THREADS=5 python validation/dome/run_pyocean.py 10 zstar [--dt 600] [--days 50] [--tag x]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from boundaries.obcs import OBData  # noqa: E402
from core.dynamics.sigma_baroclinic import SigmaBaroclinic, set_threads  # noqa: E402
from core.grid.horizontal import create_cartesian_grid  # noqa: E402
from validation.dome import cases, series  # noqa: E402

N_WARM = 5


def build(case: cases.Case, coord: str, dt: float | None = None):
    """The ALE model of ``case`` in ``coord`` ∈ {'zstar', 'sigma', 'hybrid'} and its resting initial state."""
    hg = create_cartesian_grid(nx=case.nx, ny=case.ny, dx=case.dx, dy=case.dx, f0=case.f0)
    nz = case.nz
    dz = np.full(nz, cases.HMAX / nz)                                  # 25 × 144 m (Legg's MITgcm levels)
    if coord == "zstar":
        params = dict(dz_top=dz)
    elif coord == "sigma":
        params = {}
    elif coord == "hybrid":
        params = dict(dz_top=np.full(nz, 0.01), rho_target=cases.targets(nz))   # isopycnal interior (Ilicak)
    else:
        raise ValueError(f"unknown coord {coord!r}")
    prof = cases.inflow_profile(case)

    def state(t):
        return {"north": OBData(side="north", T=prof["T"], u_normal=prof["v"], zw=prof["zw"],
                                tracers=prof["s"][None], transport=prof["transport"])}
    zs = -np.linspace(0.0, cases.HMAX, 361)
    m = SigmaBaroclinic(hg, case.H.copy(), dt=case.dt if dt is None else dt, n_layers=nz, rho0=cases.RHO0,
                        t_coef=cases.ALPHA, t_ref=cases.T_REF, z0=None, closure="constant", nu_const=case.nu_v,
                        ah=case.nu_h, g=cases.G, pgf="fv", vertical_coordinate="ale", regrid=coord,
                        regrid_params=params, n_tracers=1, open_boundaries=("north",),
                        open_boundary_mode={"north": "clamped"}, open_boundary_state=state,
                        sponge_rate=cases.sponge_rate(case), sponge_T=cases.T_ambient(0.5 * (zs[1:] + zs[:-1])),
                        sponge_zw=zs, bottom_drag="quadratic", cd=case.cd, hbbl=case.hbbl)
    eta = np.zeros((case.ny, case.nx))
    if coord == "hybrid":                                             # T on 10-m levels as the source (MOM6 Z_INIT)
        zf = np.arange(0.0, cases.HMAX + 10.0, 10.0)[:, None, None]
        Hc = np.nan_to_num(case.H)[None]
        h_src = (np.minimum(zf[1:], Hc) - np.minimum(zf[:-1], Hc))[::-1]
        T_src = cases.T_ambient(-0.5 * (zf[1:] + zf[:-1]))[::-1] * np.ones_like(h_src)
        return m, m.initial_state(eta, T_src, h_src=h_src, tr=np.zeros((1,) + h_src.shape))
    zc = m.layer_heights(eta)
    return m, m.initial_state(eta, cases.T_ambient(zc))


def record(m, st) -> dict:
    """Series fields of ``st``, layers flipped to surface first, land NaN."""
    act = m.fs.active[None]
    return dict(h=st.h[::-1].copy(), T=np.where(act, st.T[::-1], np.nan), s=np.where(act, st.tr[0][::-1], np.nan),
                u=st.u[::-1].copy(), v=st.v[::-1].copy(), eta=st.eta.copy())


def run(res_km: float, coord: str, dt: float | None = None, days: float | None = None, tag: str = "",
        threads: int = 5) -> Path:
    """Run and write cases.run_dir(res_km, 'pyocean', coord, tag)/series.npz; returns its path."""
    set_threads(threads)
    c = cases.get(res_km)
    out = cases.run_dir(res_km, "pyocean", coord, tag) / "series.npz"
    out.unlink(missing_ok=True)
    dt = c.dt if dt is None else dt
    days = c.days if days is None else days
    m, st = build(c, coord, dt)
    n_steps = int(round(days * 86400.0 / dt))
    every = max(1, int(round(c.out_s / dt)))
    recs, times, tic = [record(m, st)], [0.0], None
    for n in range(1, n_steps + 1):
        if n == N_WARM + 1:
            tic = time.perf_counter()
        st = m.step(st)
        if n % every == 0 or n == n_steps:
            recs.append(record(m, st))
            times.append(n * dt)
    wall = (time.perf_counter() - tic) * n_steps / max(1, n_steps - N_WARM) if tic else float("nan")
    s = {k: np.array([r[k] for r in recs]) for k in ("h", "T", "s", "u", "v", "eta")}
    s.update(t=np.array(times), x=c.x, y=c.y, dx=np.full(c.nx, c.dx), dy=np.full(c.ny, c.dx), H=c.H)
    meta = dict(model="pyOCEAN", coord=coord, res_km=res_km, dt=dt, n_steps=n_steps, wall_s=wall, threads=threads,
                n_trunc=int(st.n_trunc), n_limited=int(st.n_limited))
    return series.save(out, s, meta)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("res_km", type=float)
    ap.add_argument("coord", choices=["zstar", "sigma", "hybrid"])
    ap.add_argument("--dt", type=float)
    ap.add_argument("--days", type=float)
    ap.add_argument("--tag", default="")
    ap.add_argument("--threads", type=int, default=5)
    a = ap.parse_args()
    print(run(a.res_km, a.coord, a.dt, a.days, a.tag, a.threads))


if __name__ == "__main__":
    main()
```

`cases.get(50.0)` must work: 1500/50 = 30 and 700/50 = 14, so the channel has 2 cells. The north edge's cells outside the channel are land (NaN), so their faces are closed.

- [ ] **Step 4: Run the smoke test**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_dome_benchmark.py -q -p no:cacheprovider`
Expected: PASS.

If the transport test fails because `_open_face_transports` scales with `last_open_Dv`, check that the provider's transport per face (`prof["transport"]`, in m³/s) matches `Qy[:, -1, :].sum(0)` face by face. The sum over faces must be exact.

- [ ] **Step 5: 10-km short run (2 days, z\*) and a Δt ladder**

Run, one at a time, ≤ 5 threads:

```bash
cd /Users/estebancruz/pyOCEAN; for dt in 1200 900 600; do NUMBA_NUM_THREADS=5 python validation/dome/run_pyocean.py 10 zstar --dt $dt --days 2 --tag dt$dt 2>&1 | tail -1; done
```

Expected: each prints a `series.npz` path, or fails with a truncation/blow-up message. Ledger the largest stable Δt per coordinate (repeat for `sigma` and `hybrid`) and the wall time per simulated day. If 10 km at the chosen Δt costs more than 10 min per 50 days, ledger it and continue. If 5 km is projected at more than 1 h, stop (Global Constraints).

- [ ] **Step 6: Commit**

```bash
git add validation/dome/series.py validation/dome/run_pyocean.py tests/test_dome_benchmark.py
git commit -m "DOME: pyOCEAN runner (inflow transport, dye, sponges, drag) and common series

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 7: Metrics (`validation/dome/metrics.py`)

**Files:**
- Create: `validation/dome/metrics.py`
- Test: `tests/test_dome_benchmark.py`

**Interfaces:**
- Consumes: `series.load`.
- Produces:
  - `path_Y(s, t0, t1) -> (x, Y)`: Legg eq. 18, time mean over [t0, t1] [s];
  - `plume_transport(s, t0, t1, thr=0.01) -> (x, T)`: Legg eq. 11, westward positive [m³ s⁻¹];
  - `entrainment(s, t0, t1, thr=0.01) -> (x, alpha_E)`: Legg eq. 12–13;
  - `final_density(s, t0, t1, x_max=800e3) -> float`: dye-weighted mean density west of x_max [kg m⁻³];
  - `centroid_depth(s, t) -> float` [m];
  - `ranges(s) -> dict(T=(min, max), s=(min, max))`;
  - `rpe_series(s) -> np.ndarray`, reusing `validation.ilicak_benchmark.metrics.hypsometry` and `diagnostics.mixing.reference_potential_energy` with the DOME EOS;
  - `inflow_by_class(s, t0, t1) -> (rho_edges, transport)`;
  - `thin_vs_thick(s, h_thin=0.01, h_thick=1.0) -> (max_thin, max_thick)`.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_dome_benchmark.py`)

```python
def _synthetic_series(nt=3):
    """A uniform-layer box with a dye slab at y ∈ [200, 300] km, 1 m/s westward flow inside it."""
    c = cases.get(50.0)
    nz = 4
    h = np.full((nt, nz, c.ny, c.nx), 900.0)
    s_ = np.zeros_like(h)
    in_slab = (c.y >= 200e3) & (c.y <= 300e3)
    s_[:, -1][:, in_slab, :] = 1.0                                     # bottom layer (surface-first index −1)
    u = np.zeros((nt, nz, c.ny, c.nx + 1))
    u[:, -1][:, in_slab, :] = -1.0
    T = np.full_like(h, cases.T_of_rho(1031.0))
    return dict(t=np.arange(nt) * 86400.0, x=c.x, y=c.y, dx=np.full(c.nx, c.dx), dy=np.full(c.ny, c.dx),
                H=np.full((c.ny, c.nx), 3600.0), h=h, T=T, s=s_, u=u, v=np.zeros((nt, nz, c.ny + 1, c.nx)),
                eta=np.zeros((nt, c.ny, c.nx)))


def test_metrics_on_a_synthetic_dye_slab():
    from validation.dome import metrics
    s = _synthetic_series()
    x, Y = metrics.path_Y(s, 0.0, 2 * 86400.0)
    np.testing.assert_allclose(Y, 250e3, rtol=0.05)                    # slab centre (cell-centred quadrature)
    x, T = metrics.plume_transport(s, 0.0, 2 * 86400.0)
    n_slab = ((s["y"] >= 200e3) & (s["y"] <= 300e3)).sum()
    np.testing.assert_allclose(T[1:-1], 1.0 * 900.0 * n_slab * s["dy"][0], rtol=1e-12)
    assert metrics.final_density(s, 0.0, 2 * 86400.0) == pytest.approx(1031.0)
    assert metrics.centroid_depth(s, 0.0) == pytest.approx(3600.0 - 450.0)
    assert metrics.ranges(s)["s"] == (0.0, 1.0)
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_dome_benchmark.py -k metrics -q -p no:cacheprovider`
Expected: FAIL with "No module named 'validation.dome.metrics'".

- [ ] **Step 3: Implement `validation/dome/metrics.py`**

```python
"""
Metrics of the DOME benchmark (spec 2026-10-14-dome-overflow §7; Legg et al. 2006 eq. 11–13, 18, Fig. 15), the same
functions for every model on the common series (validation/dome/series.py; layers surface first, h ≥ 0, land NaN).
x is along the slope (the plume turns west: transports westward positive), y across it.
"""
from __future__ import annotations

import numpy as np

from diagnostics.mixing import reference_potential_energy
from validation.dome import cases
from validation.ilicak_benchmark.metrics import hypsometry


def _window(s: dict, t0: float, t1: float) -> np.ndarray:
    return np.nonzero((s["t"] >= t0 - 1e-6) & (s["t"] <= t1 + 1e-6))[0]


def _vol(s: dict) -> np.ndarray:
    return np.nan_to_num(s["h"]) * (s["dy"][:, None] * s["dx"][None, :])[None, None]


def path_Y(s: dict, t0: float, t1: float):
    """Y(x) = ∫ y s dy dz / ∫ s dy dz (Legg eq. 18), mean over the records in [t0, t1]; NaN where no dye."""
    ys = []
    for n in _window(s, t0, t1):
        w = np.nan_to_num(s["s"][n]) * np.nan_to_num(s["h"][n])
        num = (w * s["y"][None, :, None]).sum(axis=(0, 1))
        den = w.sum(axis=(0, 1))
        ys.append(np.where(den > 0, num / np.where(den > 0, den, 1.0), np.nan))
    return s["x"], np.nanmean(ys, axis=0)


def plume_transport(s: dict, t0: float, t1: float, thr: float = 0.01):
    """T(x) = ∫_A (−u) dy dz, A = {s > thr} (Legg eq. 11), at cell centres (u averaged to them), time mean."""
    out = []
    for n in _window(s, t0, t1):
        uc = 0.5 * (s["u"][n][..., 1:] + s["u"][n][..., :-1])
        A = np.nan_to_num(s["s"][n]) > thr
        out.append((np.where(A, -uc * np.nan_to_num(s["h"][n]), 0.0) * s["dy"][None, :, None]).sum(axis=(0, 1)))
    return s["x"], np.mean(out, axis=0)


def entrainment(s: dict, t0: float, t1: float, thr: float = 0.01):
    """α_E(x) = (dT/dx)/(L Ū) (Legg eq. 12–13): L the length of the s = thr interface in (y, z), Ū = T/∫_A dy dz."""
    x, T = plume_transport(s, t0, t1, thr)
    Ls, As = [], []
    for n in _window(s, t0, t1):
        A = (np.nan_to_num(s["s"][n]) > thr)
        h = np.nan_to_num(s["h"][n])
        As.append((A * h * s["dy"][None, :, None]).sum(axis=(0, 1)))
        edge_y = (A[:, 1:] != A[:, :-1]) * 0.5 * (h[:, 1:] + h[:, :-1])          # vertical pieces of the interface
        edge_z = (A[1:] != A[:-1]) * s["dy"][None, :, None]                      # horizontal pieces
        Ls.append(edge_y.sum(axis=(0, 1)) + edge_z.sum(axis=(0, 1)))
    L, area = np.mean(Ls, axis=0), np.mean(As, axis=0)
    U = np.where(area > 0, T / np.where(area > 0, area, 1.0), np.nan)
    dTdx = -np.gradient(T, x)                                             # westward growth (x decreasing)
    return x, np.where((L > 0) & (U > 0), dTdx / (L * U), np.nan)


def final_density(s: dict, t0: float, t1: float, x_max: float = 800e3) -> float:
    """Dye-weighted mean density of the plume west of x_max (Legg Fig. 15 'final buoyancy'), time mean [kg m⁻³]."""
    vals = []
    for n in _window(s, t0, t1):
        w = np.nan_to_num(s["s"][n]) * _vol(s)[n] * (s["x"] < x_max)[None, None, :]
        rho = cases.RHO0 - cases.ALPHA * (np.nan_to_num(s["T"][n]) - cases.T_REF)
        vals.append((w * rho).sum() / w.sum())
    return float(np.mean(vals))


def centroid_depth(s: dict, t: float) -> float:
    """Dye-weighted mean depth of the layer centres at the record nearest t [m, positive down]."""
    n = int(np.argmin(np.abs(s["t"] - t)))
    h = np.nan_to_num(s["h"][n])
    zc = np.cumsum(h, axis=0) - 0.5 * h - np.nan_to_num(s["eta"][n])[None]
    w = np.nan_to_num(s["s"][n]) * _vol(s)[n]
    return float((w * zc).sum() / w.sum())


def ranges(s: dict) -> dict:
    """(min, max) of T and s over every wet cell of every record."""
    wet = np.nan_to_num(s["h"]) > 0
    return {k: (float(np.min(s[k][wet])), float(np.max(s[k][wet]))) for k in ("T", "s")}


def rpe_series(s: dict) -> np.ndarray:
    """RPE [J] of every record (Ilicak's exact hypsometry; validation/ilicak_benchmark/metrics.py)."""
    zA = hypsometry(s["H"], s["dx"], s["dy"])
    vol = _vol(s)
    rho = cases.RHO0 - cases.ALPHA * (s["T"] - cases.T_REF)
    return np.array([reference_potential_energy(np.where(vol[n] > 0, rho[n], np.nan), vol[n], hypsometry=zA)
                     for n in range(s["t"].size)])


def inflow_by_class(s: dict, t0: float, t1: float, n_classes: int = 25):
    """Southward transport through the north channel faces binned by density class (MOM6 R_k) [m³ s⁻¹]."""
    edges = cases.targets(n_classes)
    tot = np.zeros(n_classes)
    idx = _window(s, t0, t1)
    for n in idx:
        v = -s["v"][n][:, -1, :] * np.nan_to_num(s["h"][n][:, -1, :]) * s["dx"][None, :]
        rho = cases.RHO0 - cases.ALPHA * (np.nan_to_num(s["T"][n][:, -1, :]) - cases.T_REF)
        k = np.clip(np.searchsorted(edges, rho) - 1, 0, n_classes - 1)
        np.add.at(tot, k.ravel(), v.ravel())
    return edges, tot / max(1, idx.size)


def thin_vs_thick(s: dict, h_thin: float = 0.01, h_thick: float = 1.0):
    """Max |u| at u-faces whose thinner side is < h_thin and at those whose both sides are ≥ h_thick (4b/4d)."""
    h = np.nan_to_num(s["h"])
    hmin = np.minimum(h[..., 1:], h[..., :-1])
    u = np.abs(s["u"][..., 1:-1])
    thin, thick = u[hmin < h_thin], u[hmin >= h_thick]
    return float(thin.max()) if thin.size else 0.0, float(thick.max()) if thick.size else 0.0
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_dome_benchmark.py -q -p no:cacheprovider`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
git add validation/dome/metrics.py tests/test_dome_benchmark.py
git commit -m "DOME: metrics (Legg path, plume transport, entrainment, final density, RPE, inflow by class, thin faces)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 8: MOM6 ρ runs (`validation/dome/mom6.py`, `run_mom6.sh`)

**Files:**
- Create: `validation/dome/mom6.py`
- Create: `validation/dome/run_mom6.sh`
- Test: `tests/test_dome_benchmark.py`

**Interfaces:**
- Consumes: `cases`, `series`; the MOM6 build `~/pyOCEAN_refs/MOM6/.testing/build/repro/MOM6`; the DOME example inputs `~/pyOCEAN_refs/MOM6-examples/ocean_only/DOME/` (read only).
- Produces:
  - `mom6.mom_input(case, variant="common") -> str`, with `variant` ∈ {"common", "kv_low"};
  - `mom6.write_case(res_km, out_dir, variant)`;
  - `mom6.read(run_dir, res_km) -> series dict`, layers surface first, `T` from layer densities via `cases.T_of_rho`, `s` = tr_D1;
  - `mom6.wall_s(run_dir) -> float`.

- [ ] **Step 1: Write the failing test**

```python
def test_mom6_input_has_the_common_physics_and_the_dome_configs():
    from validation.dome import mom6
    txt = mom6.mom_input(cases.get(10.0), "common")
    for line in ('TOPO_CONFIG = "DOME"', 'OBC_USER_CONFIG = "DOME"', 'SPONGE_CONFIG = "DOME"', "USE_DOME_TRACER = True",
                 "LAPLACIAN = True", "KH = 50", "BIHARMONIC = False", "KV = 0.5", "KD = 0.0",
                 "USE_JACKSON_PARAM = False", "CDRAG = 0.002", "DRAG_BG_VEL = 0.0", "HBBL = 10.0", "DAYMAX = 50.0",
                 "NIGLOBAL = 150", "NJGLOBAL = 70", "LAYOUT = 1, 1"):
        assert line in txt, line
    assert "KV = 0.0001" in mom6.mom_input(cases.get(10.0), "kv_low")
    lines = mom6.override_lines(cases.get(10.0), "common")
    assert "#override KV = 0.5" in lines and "KH = 50" in lines and "LAPLACIAN = True" in lines
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_dome_benchmark.py -k mom6 -q -p no:cacheprovider`
Expected: FAIL (no module).

- [ ] **Step 3: Implement `validation/dome/mom6.py`**

`mom_input(case, variant)` returns the `MOM_input` of `MOM6-examples/ocean_only/DOME` (read the file and keep every line). Then it appends overrides; MOM6 takes the last assignment only through `MOM_override` with `#override`, so write the changes there. Concretely, `write_case` writes:
- `MOM_input`: the example's lines plus `LAYOUT = 1, 1` and `IO_LAYOUT = 1, 1`;
- `MOM_override`: the lines below;
- `mom_input()` returns the concatenation of both, which the test checks.

```python
# common physics (spec §5): Legg's z-model at 10 km; ν_h ∝ Δx.  The example's MOM_input sets NIGLOBAL, NJGLOBAL, DT,
# DT_THERM, DAYMAX, SMAGORINSKY_AH, AH_VEL_SCALE, KV, KD, USE_JACKSON_PARAM, CDRAG, DRAG_BG_VEL, HBBL (→ #override);
# LAPLACIAN, KH, BIHARMONIC are not in it (→ plain lines; MOM6 FATALs on an #override of an unset parameter).
SET_IN_EXAMPLE = {"NIGLOBAL", "NJGLOBAL", "DT", "DT_THERM", "DAYMAX", "SMAGORINSKY_AH", "AH_VEL_SCALE", "KV", "KD",
                  "USE_JACKSON_PARAM", "CDRAG", "DRAG_BG_VEL", "HBBL"}


def overrides(c: cases.Case, variant: str) -> dict:
    """Parameter → value of the run (plain, without #override)."""
    kv = {"common": c.nu_v, "kv_low": 1e-4}[variant]
    return {"NIGLOBAL": c.nx, "NJGLOBAL": c.ny, "DT": c.dt, "DT_THERM": c.dt, "DAYMAX": 50.0, "LAPLACIAN": "True",
            "KH": f"{c.nu_h:g}", "BIHARMONIC": "False", "SMAGORINSKY_AH": "False", "AH_VEL_SCALE": 0.0, "KV": kv,
            "KD": 0.0, "USE_JACKSON_PARAM": "False", "CDRAG": 0.002, "DRAG_BG_VEL": 0.0, "HBBL": 10.0}


def override_lines(c: cases.Case, variant: str) -> list[str]:
    """MOM_override lines: '#override ' only for parameters the example's MOM_input already sets."""
    return [("#override " if k in SET_IN_EXAMPLE else "") + f"{k} = {v}" for k, v in overrides(c, variant).items()]
```

`mom_input(case, variant)` returns the example's `MOM_input` text, then `LAYOUT = 1, 1`, then `override_lines(...)` with the `#override ` prefixes stripped, so the test sees plain `KH = 50`. Check `SET_IN_EXAMPLE` against the example's `MOM_input` with `grep -E "^(NIGLOBAL|...) *=" MOM_input` before the first run, and ledger any difference.

`diag_table`: daily snapshots of `h`, `e`, `u`, `v`, `tr_D1`:

```
"DOME pyOCEAN benchmark"
1 1 1 0 0 0
"prog", 1, "days", 1, "days", "Time",
"ocean_model", "h", "h", "prog", "all", .false., "none", 2
"ocean_model", "u", "u", "prog", "all", .false., "none", 2
"ocean_model", "v", "v", "prog", "all", .false., "none", 2
"ocean_model", "tr_D1", "tr_D1", "prog", "all", .false., "none", 2
"ocean_model", "SSH", "SSH", "prog", "all", .false., "none", 2
```

Write `input.nml` as `validation/ilicak_benchmark/mom6.py`'s `NML`, and `mkdir RESTART`.

`read(run_dir, res_km)`:
- opens `prog.nc`, with dims (time, zl, yh, xh) for h and tr_D1 and (time, zl, yh, xq) for u;
- MOM6 layer 1 is the surface, as in the series;
- drops MOM6's west q-point column if `xq` has nx+1 points (symmetric memory: the build is dynamic_symmetric, so u has nx+1 points and v ny+1);
- `T = cases.T_of_rho(R_k)` per layer, with R_k from `cases.targets` centres; layers with h < 1e-3 m get NaN;
- `s = tr_D1`; `H` from `ocean_geometry.nc` `D` (land NaN where D ≤ 1 m);
- prepends t = 0 from `MOM_IC.nc` if `prog.nc` has no t = 0 record (as `validation/ilicak_benchmark/mom6.py:read`).

Read `validation/ilicak_benchmark/mom6.py:read` (lines 128–153) and follow it.

`validation/dome/run_mom6.sh`: a copy of `validation/ilicak_benchmark/run_mom6.sh` with:
- `D=$(python -c "...cases.run_dir(RES, 'mom6', 'rho', TAG)")`;
- `mom6.write_case(RES, Path(D), VARIANT)`;
- `series.save(..., meta)` with `model='MOM6', coord='rho', res_km=RES, variant=VARIANT`;
- the usage line `run_mom6.sh <res_km> <variant> [tag]`;
- the same wait on "Total runtime" and the same cleanup of its own ranks.

- [ ] **Step 4: Run the test, then one 2-day MOM6 run at 10 km**

Run: `python -m pytest tests/test_dome_benchmark.py -k mom6 -q -p no:cacheprovider`
Expected: PASS.

Run: `bash validation/dome/run_mom6.sh 10 common short` with `DAYMAX` temporarily 2 via `MOM6_OVERRIDE='#override DAYMAX = 2.0'`, then `pgrep -fl "mpirun|orted|MOM6"`.
Expected: prints a `series.npz` path; no orphan processes.

If MOM6 FATALs on a parameter, fix the override list (plain vs `#override`) and ledger it.

- [ ] **Step 5: Commit**

```bash
git add validation/dome/mom6.py validation/dome/run_mom6.sh tests/test_dome_benchmark.py
git commit -m "DOME: MOM6 ρ writer/reader/runner with the common physics (Laplacian, KV 0.5, no Jackson, C_d 2e-3)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 9: MITgcm z runs and the Legg anchor (DECISION POINT 1)

**Files:**
- Create: `validation/dome/mitgcm.py`
- Create: `validation/dome/run_mitgcm.sh`
- Test: `tests/test_dome_benchmark.py`

**Interfaces:**
- Consumes: `cases`, `series`; MITgcm source `/Volumes/ESTEBAN_MIT/MITgcm` (read only); the build options `darwin_arm64_gfortran_esteban`.
- Produces:
  - `mitgcm.write_build(res_km, code_dir)`: SIZE.h, packages.conf with `gfd obcs ptracers rbcs diagnostics mnc`, and copies of `verification/dome/code/*_OPTIONS.h`;
  - `mitgcm.write_run(res_km, run_dir, variant)`, with `variant` ∈ {"legg", "common"};
  - `mitgcm.read(run_dir, res_km) -> series dict`;
  - `mitgcm.wall_s(run_dir)`.

- [ ] **Step 1: Write the failing test**

```python
def test_mitgcm_namelists_have_the_variant_physics():
    from validation.dome import mitgcm
    d = mitgcm.data_text(cases.get(10.0), "legg")
    for s in ("viscAh=50.", "viscAr=0.5", "bottomDragQuadratic=0.002", "no_slip_bottom=.TRUE.", "diffKhT=0.",
              "diffKrT=0.", "tempAdvScheme=77", "nonlinFreeSurf=4", "select_rStar=2", "f0=0.0001", "hFacMin=0.1"):
        assert s in d, s
    assert "no_slip_bottom=.FALSE." in mitgcm.data_text(cases.get(10.0), "common")
    o = mitgcm.obcs_text(cases.get(10.0))
    assert "useOBCSprescribe=.TRUE." in o and "OBNvFile='OBNv.bin'" in o and "OBNptrFile(1)='OBNs.bin'" in o
```

- [ ] **Step 2: Run to verify it fails**

Run: `python -m pytest tests/test_dome_benchmark.py -k mitgcm -q -p no:cacheprovider`
Expected: FAIL (no module).

- [ ] **Step 3: Implement**

Base `validation/dome/mitgcm.py` on `validation/ilicak_benchmark/mitgcm.py` (read it whole first), with these differences:

- **Grid:** 150 × 70 (or 300 × 140) plus one land column east and one land row south. Periodic exchanges need walls. The north edge is the OBCS row, so add the land row at the south: the bathymetry file has `ny + 1` rows, with row 0 land.
- **Levels:** `delR = 25*144.`, partial cells `hFacMin = 0.1`, z\* (`select_rStar=2`, `nonlinFreeSurf=4`, `exactConserv=.TRUE.`, NONLIN_FRSURF defined), because the sea level rises ~35 m.
- **EOS:** linear in T with `tAlpha = ALPHA/RHO0`, `rhoNil = rhoConst = 1031`, `tRef = 25*T_REF`, `gravity = 9.8`.
- **Advection:** `tempAdvScheme = 77` (Legg: flux-limited; check `pkg/generic_advdiff` for the 77 code = "2nd order flux limiter with SuperBee", and record it) and `PTRACERS_advScheme(1) = 77`.
- **Variant "legg":** `viscAh = nu_h`, `viscAr = nu_v`, `no_slip_bottom=.TRUE.`, `no_slip_sides=.FALSE.`, `bottomDragQuadratic=0.002`, `diffKhT = diffKrT = 0`.
- **Variant "common":** the same with `no_slip_bottom=.FALSE.`.
- **`data.obcs`:** `OB_Jnorth = Nx*-1` (north row; −1 means the last row, `obcs_readparms.F:73-85`), `useOBCSprescribe=.TRUE.`, `OBNvFile='OBNv.bin'`, `OBNtFile='OBNt.bin'`, `OBNptrFile(1)='OBNs.bin'` (`obcs_fields_load.F:306`).
  - The files are (Nr, Nx) big-endian float64 from `cases.inflow_profile(case, nz_bdy=25)` on MITgcm's 144-m levels. The channel is 600 m, so only the top ~5 levels are wet.
  - Remap `v`, `T` and `s` conservatively from the 60-level profile to the levels that intersect the 600-m column; the velocity is weighted so that Σ v dz dx equals `transport`.
  - Land columns get 0.
  - `useOBCSbalance=.FALSE.`: the inflow must be prescribed exactly.
- **`data.rbcs`:**
  - `RBCS_PARM01`: `relaxMaskFile(1)='rbcs_mask.bin'`, `relaxTFile='rbcs_T.bin'`, `tauRelaxT = 8640.` (1/(10 day⁻¹)), with the mask = `sponge_rate / peak` on all levels.
  - `RBCS_PARM02`: `useRBCpTrNum(1)=.TRUE.`, `tauRelaxPTR(1)=8640.`, `relaxPtracerFile(1)='rbcs_s.bin'` (zeros).
- **`data.ptracers`:** `PTRACERS_numInUse=1`, `PTRACERS_advScheme(1)=77`, `PTRACERS_diffKh(1)=0.`, `PTRACERS_diffKr(1)=0.`, `PTRACERS_initialFile(1)=' '` (zero).
- **Diagnostics:** daily snapshots of `THETA`, `UVEL`, `VVEL`, `hFactorC`, `TRAC01`, `ETAN`.
- **`read`:** as Ilicak's reader, plus `s` from `TRAC01`; drop the south land row and the east land column.

`run_mitgcm.sh`: Ilicak's script with `build <res_km>` and `run <res_km> <variant> [tag]`.

- [ ] **Step 4: Run the test and build**

Run: `python -m pytest tests/test_dome_benchmark.py -k mitgcm -q -p no:cacheprovider` (PASS), then `bash validation/dome/run_mitgcm.sh build 10`.
Expected: `OK …/mitgcmuv`. If the build fails, read `make.log`, fix the code-dir options, and ledger the fix.

- [ ] **Step 5: The Legg anchor at 10 km (DECISION POINT 1)**

Run: `bash validation/dome/run_mitgcm.sh run 10 legg`. Then compute, with `metrics.py`:
- `path_Y` and `plume_transport` over days 30–50;
- `final_density`, converted to final buoyancy b = −g(ρ − 1030)/ρ₀ as Legg plots.

Read Legg Figs. 15–17 from `~/Documents/TESIS_InternalWaves-2/Literatura/Legg_2006 - ….pdf`, opened as an image with Read, pages ~20–23: the 10-km z-coordinate final buoyancy (Fig. 15) and the mean path (Figs. 16–17). Write both values, with a reading tolerance, into `cases.py` as `LEGG_10KM = dict(final_buoyancy=..., path_y_at_x=...)`.

Expected: MITgcm "legg" within the reading tolerance (~10 %) of Legg's 10-km z values.
- **If not:** STOP and report to Esteban with the numbers. The case is mis-defined (spec, decision point 1).
- **If yes:** ledger the comparison and continue.

- [ ] **Step 6: Commit**

```bash
git add validation/dome/mitgcm.py validation/dome/run_mitgcm.sh validation/dome/cases.py tests/test_dome_benchmark.py
git commit -m "DOME: MITgcm z writer/reader/runner (OBCS prescribed inflow, ptracers dye, rbcs sponges); Legg anchor

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 10: Campaign, comparison, README and docs

**Files:**
- Create: `validation/dome/compare.py`
- Create: `validation/dome/README.md`
- Modify: `CLAUDE.md` (new rows for tracers, interior sponge, ALE drag, DOME; test counts)
- Modify: `docs/design/vision_roadmap.md` (§7 DOME row, stage C status)
- Modify: `validation/ilicak_benchmark/README.md` (only if DOME changes the status of items 4b/4d)

**Interfaces:**
- Consumes: Tasks 5–9.
- Produces: `compare.py <res_km>`, which prints Markdown tables of every metric for every run found under `cases.RUNS/<res>km/`.

- [ ] **Step 1: Runs, one at a time, ≤ 5 threads, outputs on the external disk**

10 km, 50 days:
- `run_pyocean.py 10 zstar`, `run_pyocean.py 10 sigma`, `run_pyocean.py 10 hybrid` (Δt from the Task 6 ladder);
- `run_mom6.sh 10 common`, `run_mom6.sh 10 kv_low`;
- `run_mitgcm.sh run 10 common` (the "legg" run exists from Task 9);
- `run_pyocean.py 10 hybrid --tag kv_low`, with `nu_v = 1e-4` through a `--nu-v` option; add it to `run_pyocean.py` in this step.

5 km, 50 days:
- pyOCEAN z\*, σ and hybrid;
- `run_mom6.sh 5 common`;
- `run_mitgcm.sh run 5 common`.

After each MOM6 run: `pgrep -fl "mpirun|orted|MOM6"` must be empty. Log each to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/dome/log_*.txt` (never committed).

If a pyOCEAN run stops or leaves its range: rerun with three 1e-10 T perturbations (seeds) before attributing the failure (spec, decision point 2), and ledger the outcome.

- [ ] **Step 2: `compare.py`**

`compare.py` loads every `series.npz` under `RUNS/<res>km/*/` and prints one table per resolution. Columns:
- model/coord;
- Y(x) at x = 900, 700 and 500 km;
- plume transport at x = 900 / 700 / 500 km;
- mean α_E between x = 1000 and 700 km;
- final density;
- centroid depth at days 10, 20, 30 and 50;
- RPE change over 50 days;
- T and s ranges;
- thin/thick max |u|;
- n_trunc, n_limited;
- Δt, s per simulated day.

A second table gives the inflow transport by density class (pyOCEAN vs MOM6). Write the output to `RUNS/compare_<res>km.txt`.

- [ ] **Step 3: Evaluate the spec criteria 1–9**

Write each criterion with its measured value into the ledger. Classify every difference (a)–(d):
- **(a)** our bug: fix it in this branch only if it is small and local, with a test. Otherwise it is decision point 3: a separate plan, and report.
- **(b)** missing physics (Smagorinsky, Ri mixing): goes to the README "what pyOCEAN should improve" list.
- **(c), (d):** explained in the README.

- [ ] **Step 4: `validation/dome/README.md` (Spanish, Ilicak style)**

Sections:
1. Configuración y diferencias por modelo (c), with the inflow, sponge, drag and EOS differences of spec §5–§6.
2. Tablas a 10 y 5 km (from `compare.py`).
3. Anclaje con Legg (Task 9).
4. Resultados por criterio.
5. Robustez (4b/4d) y costo.
6. "Qué debe mejorar pyOCEAN": numbered, each item with its class.

Figures go to `RUNS/figs/` (not the repo), referenced by path. They show the bottom dye at days 10, 20, 30 and 50, Y(x) and T(x) for every model.

- [ ] **Step 5: CLAUDE.md and roadmap**

- **`CLAUDE.md`:**
  - extend the ALE-mode row with `n_tracers`, the interior sponge (`sponge_rate`), `bottom_drag='quadratic'` and `OBData.transport`, one line each, with their tests;
  - add a "DOME overflow benchmark" row (`validation/dome/`, `tests/test_dome_benchmark.py`) with the headline numbers;
  - update the test counts (run `pytest --collect-only -q` per file) and the total.
- **`docs/design/vision_roadmap.md` §7:** a DOME row with criteria and status, and the stage C line (DOME ✅ / 🔶; split-explicit still open).

- [ ] **Step 6: Full suite and commit**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider > $WS/suite_t10.txt 2>&1; tail -2 $WS/suite_t10.txt`
Expected: all pass, 2 xfailed.

```bash
git add validation/dome/compare.py validation/dome/README.md validation/dome/run_pyocean.py CLAUDE.md docs/design/vision_roadmap.md
git commit -m "DOME: campaign at 10 and 5 km, comparison with MOM6 ρ, MITgcm z and Legg 2006; docs

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

Add `validation/ilicak_benchmark/README.md` to the commit only if Step 3 changed items 4b/4d.
