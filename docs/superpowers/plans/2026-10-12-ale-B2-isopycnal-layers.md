# ALE Plan B2 (B2c + B2b): isopycnal layer preservation and PPM thickness flux — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Stop the hybrid ALE rule from smearing sharp density steps (B2c: interfaces between isopycnally consistent layers stay put) and, if it sharpens the gravity-current nose, replace the donor-cell layer thickness flux by MOM6's PPM flux with a barotropic velocity adjustment (B2b).

**Architecture:** B2c is a post-step inside the per-column hybrid rule `regrid.hycom1_column` (Numba), switched by `regrid_params["keep_isopycnal"]` (default True). B2b adds a second layer-transport scheme in `core/vertical/ale.py` (`ppm_transport_x`), selected by a new `SigmaBaroclinic(..., thickness_flux="upwind" | "ppm")`; it is adopted as default only if the hybrid lock-exchange 1 m front falls within ±10 % of Benjamin (decision gate, Task 5).

**Tech Stack:** Python 3, NumPy 1.26.4, Numba (`@njit`, not disk-cached for kernels that call other modules), pytest.

**Spec:** `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` — §2 step 1 (transport, B2b), §3 "Conservación de las capas isopicnas" (B2c), §7 unit tests and integration item 6, §9 item 4b.

## Global Constraints

- Never use the 6 efficiency cores: run everything with `NUMBA_NUM_THREADS=5`; the model calls `set_threads(5)`-style `numba.set_num_threads(_THREADS[0])` before parallel kernels. One heavy run at a time.
- Do not install packages; numpy must stay 1.26.4.
- Commit only with Esteban's authorisation; never stage `pyOCEAN-literature-review-guide.docx`, `log_*.txt` or `*.npz`. Commit messages end with:
  `Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>` and `Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn`.
- Kernels in `regrid.py` that call `remap.py` must stay non-disk-cached (`@njit` without `cache=True`; `tests/test_vertical_regrid.py::test_kernels_calling_remap_are_not_disk_cached`).
- The σ path (`vertical_coordinate="sigma_eulerian"`) must stay bit-identical; ALE-σ and z\* results must not regress: z\* bay-mouth step ≤ 1e-10 m/s (now 3.5e-12), z\* seamount + FV + ρ_ref ≈ 0.11 cm/s at day 10.
- Do not overwrite `validation/seamount/out/summary_*.txt` with exploratory runs (use the scratchpad).
- Every scheme cites its source as `file:line` (MOM6 commit 0e9a561, HYCOM-src) in the docstring.
- Spec numbers to meet (§7 item 6): hybrid lock-exchange body (5 m) at Benjamin ±10 %; RPE gain < 0.2 × ALE-σ; with B2b the 1 m front within ±10 %, else B2b is not adopted and the xfail keeps the measured numbers.

## Review Focus

1. A column whose layers are all consistent except one thin inconsistent layer next to the bed (gravity-current head after mixing): the regridded interfaces must stay between the kept ones and the bed, no negative thickness, Σh = D.
2. `keep_isopycnal=True` with the shallow σ window (`d_hybrid > 0`): kept interfaces must still be clipped by the window (σ zone columns stay pure σ).
3. A column where the z\* floor pushes a kept interface down past a regridded one below it: order must be restored (monotone interfaces) before `enforce_min_thickness`.
4. B2b at faces next to land and on the domain edge: zero flux, no NaN from the PPM edge estimate with missing neighbours.
5. B2b with CFL > 0.5 in a vanished layer (current flooding a 1 mm layer): fluxes must not empty the donor cell (`lagrangian_thickness` must not raise).

---

### Task 1: Isopycnal layer preservation in the hybrid column rule (B2c)

**Files:**
- Modify: `core/vertical/regrid.py` (module docstring; `hycom1_column` at `:153`; new `_keep_consistent`; `_regrid_columns`; `regrid_columns`)
- Modify: `tests/test_vertical_regrid.py` (helper `_hybrid` at `:60`, `_window_case` at `:150`, tests at `:200` and `:255`; new tests)
- Modify: `validation/mom6_column/compare_hybgen.py:101` (pass the new argument)

**Interfaces:**
- Produces: `hycom1_column(H, eta, h_bed, rho_bed, rho_target, dz_top, max_h_top, h_min, keep, zw)` — `keep: bool` inserted before `zw`.
- Produces: `_keep_consistent(h_top, rho_top, targets, h_mass, x)` (njit) — overwrites interior `x` (depths from the surface, surface first) in place.
- Produces: `regrid_columns` reads `params.get("keep_isopycnal", True)`; `_regrid_columns(..., h_min, keep, wet, zw)`.

- [ ] **Step 1: Update the test helpers to the new signature (old behaviour = `keep=False`)**

In `tests/test_vertical_regrid.py` replace the helper at `:60` with:

```python
def _hybrid(H, eta, h, rho, targets, dz=DELR, max_h=None, h_min=rg.H_MIN, keep=False):
    n = h.size
    zw = np.empty(n + 1)
    mh = np.full(n, np.inf) if max_h is None else max_h
    rg.hycom1_column(H, eta, h, rho, targets, dz, mh, h_min, keep, zw)
    return zw
```

In `_window_case` (`:158`) change the call to
`rg.hycom1_column(H, 0.0, h, rho, targets, dz, np.full(n, np.inf), rg.H_MIN, False, hyb)`.
In `test_regrid_columns_hybrid_with_transition_matches_the_window` (`:207`) and `test_regrid_columns_hybrid_matches_single_column` (`:262`) add `keep_isopycnal=False` to `params` (they test the dispatch against the P1M single column).

In `validation/mom6_column/compare_hybgen.py:101` change the call to

```python
    rg.hycom1_column(c["H"], 0.0, c["h"][::-1].copy(), c["r"][::-1].copy(), tgt_iface, dz,
                     np.full(n, np.inf), H_MIN, keep, zw)
```

and give `hycom1(c, dz, tgt_iface, blend, keep=True)` the extra keyword argument (the model default).

- [ ] **Step 2: Write the failing B2c tests**

Append to `tests/test_vertical_regrid.py`:

```python
# ---------------------------------------------------------------- B2c: isopycnal layer preservation
def _consistent_column(n=10, H=200.0, seed=0):
    """Bed-first layers whose densities lie inside their own target brackets (surface-first targets)."""
    rng = np.random.default_rng(seed)
    targets = np.linspace(1020.0, 1030.0, n + 1)                        # surface first
    mid_top = 0.5 * (targets[:-1] + targets[1:])                        # layer j (surface first)
    h = rng.uniform(0.5, 1.5, n)
    h *= H / h.sum()
    return h, mid_top[::-1].copy(), targets                              # h, ρ bed first


def test_keep_isopycnal_leaves_consistent_layers_exactly_in_place():
    h, rho, targets = _consistent_column()
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=True)
    np.testing.assert_allclose(zw, -200.0 + np.concatenate([[0.0], np.cumsum(h)]), rtol=0, atol=1e-11)
    moved = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=False)
    assert np.abs(moved - zw).max() > 1.0                                # P1M would move them


def test_keep_isopycnal_regrids_only_the_interfaces_of_an_inconsistent_layer():
    h, rho, targets = _consistent_column()
    rho[4] = 1021.2                                                      # bed-first layer 4 out of its bracket
    zin = -200.0 + np.concatenate([[0.0], np.cumsum(h)])
    p1m = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=False)
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=True)
    other = [k for k in range(11) if k not in (4, 5)]
    np.testing.assert_allclose(zw[other], zin[other], rtol=0, atol=1e-11)
    for k in (4, 5):                                                     # P1M position, between the kept 3 and 6
        np.testing.assert_allclose(zw[k], np.clip(p1m[k], zin[3], zin[6]), rtol=0, atol=1e-11)
    assert (np.diff(zw) > 0).all()


def test_keep_isopycnal_counts_massless_layers_as_consistent():
    # a vanished layer with a stale density between two consistent thick ones does not unpin them
    h, rho, targets = _consistent_column()
    h[3] = rg.H_MIN
    rho[3] = 1029.9                                                      # far outside its bracket
    h *= 200.0 / h.sum()
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=np.full(10, 1e-6), keep=True)
    np.testing.assert_allclose(zw, -200.0 + np.concatenate([[0.0], np.cumsum(h)]), rtol=0, atol=1e-11)


def test_keep_isopycnal_still_applies_the_zstar_floor_and_keeps_order():
    # consistent layers squeezed against the surface: the z* floor pushes the kept interfaces
    # down; the column stays ordered and adds up to D
    h, rho, targets = _consistent_column()
    h[-3:] = [0.2, 0.2, 0.2]                                             # three thin top layers
    h *= 200.0 / h.sum()
    dz = np.full(10, 5.0)                                                # floor: 5 m per layer from the top
    zw = _hybrid(200.0, 0.0, h, rho, targets, dz=dz, keep=True)
    assert (np.diff(zw) > 0).all() and zw[0] == -200.0 and zw[-1] == 0.0
    np.testing.assert_allclose(zw[::-1][1:4], -np.cumsum(dz)[:3], atol=1e-11)


def test_regrid_columns_defaults_to_keep_isopycnal():
    h, rho, targets = _consistent_column()
    h3, r3 = h[:, None, None].repeat(2, 2), rho[:, None, None].repeat(2, 2)
    base = dict(dz_top=np.full(10, 1e-6), rho_target=targets)
    zw = rg.regrid_columns("hybrid", np.full((1, 2), 200.0), np.zeros((1, 2)), h3, r3, base, np.ones((1, 2), bool))
    np.testing.assert_allclose(zw[:, 0, 0], _hybrid(200.0, 0.0, h, rho, targets, dz=base["dz_top"], keep=True),
                               rtol=0, atol=1e-12)
    zw0 = rg.regrid_columns("hybrid", np.full((1, 2), 200.0), np.zeros((1, 2)), h3, r3,
                            dict(base, keep_isopycnal=False), np.ones((1, 2), bool))
    np.testing.assert_allclose(zw0[:, 0, 1], _hybrid(200.0, 0.0, h, rho, targets, dz=base["dz_top"], keep=False),
                               rtol=0, atol=1e-12)


def test_keep_isopycnal_interfaces_stay_inside_the_shallow_window():
    # review focus 2 (guard: passes before Task 1 and must still pass after): with q = 0.5 every
    # interface, kept or not, lies between L_k = q(−H) + (1−q)σ_k and U_k = q z*_k + (1−q)σ_k
    H = 7.5
    h, rho, targets = _consistent_column(H=H)
    params = dict(dz_top=np.full(10, 0.2), rho_target=targets, dsigma=np.full(10, 0.1), d_sigma=5.0, d_hybrid=10.0)
    zw = rg.regrid_columns("hybrid", np.full((1, 1), H), np.zeros((1, 1)), h[:, None, None], rho[:, None, None],
                           params, np.ones((1, 1), bool))[:, 0, 0]
    sig, zs = np.empty(11), np.empty(11)
    rg.sigma_column(H, 0.0, params["dsigma"], rg.H_MIN, sig)
    rg.zstar_column(H, 0.0, params["dz_top"], rg.H_MIN, zs)
    q = 0.5
    lo, hi = q * (-H) + (1 - q) * sig, q * zs + (1 - q) * sig
    assert ((zw >= lo - 1e-9) & (zw <= hi + 1e-9)).all()
    assert (np.diff(zw) > 0).all()
```

Append to `tests/test_sigma_ale.py` (after `test_hybrid_lock_exchange_fronts_leave_the_lock_at_the_benjamin_speed`), and delete `test_hybrid_lock_exchange_keeps_a_sharp_front_and_mixes_less_than_sigma`, which these two replace:

```python
def test_hybrid_lock_exchange_mixes_much_less_than_sigma():
    # spec §7 item 6: with isopycnal layer preservation the regrid no longer spreads the density
    # step at the nose (prototype: 2.5e7 J vs ALE-σ 3.2e8 J in 6 h)
    r, s = _lock_run("hybrid"), _lock_run("sigma")
    assert r["rpe_gain"] < 0.2 * s["rpe_gain"], (r["rpe_gain"], s["rpe_gain"])


@pytest.mark.xfail(strict=True, reason="open (ALE Plan B2b): donor-cell thickness flux lets a thin dense film "
                   "run ahead of the current")
def test_hybrid_lock_exchange_nose_moves_at_the_benjamin_speed():
    r = _lock_run("hybrid")
    for k in range(2):
        assert abs(r["speed"][1.0][k] - 1) < 0.10, r["speed"]
```

Note on `test_keep_isopycnal_still_applies_the_zstar_floor_and_keeps_order`: `_consistent_column` gives top layers with densities inside their brackets, so all interfaces are kept; the floor then pushes the top three down to 5, 10, 15 m (surface-first `zw[::-1][1:4]` = −5, −10, −15).

- [ ] **Step 3: Run the tests to see them fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_regrid.py -q`
Expected: the new tests and every test using `_hybrid`/`hycom1_column` FAIL with a Numba `TypingError`/`TypeError` about the number of arguments (signature not changed yet), except the window guard test.
Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k hybrid_lock -q`
Expected: `test_hybrid_lock_exchange_mixes_much_less_than_sigma` FAILS (≈ 3.1e8 J vs 3.2e8 J); the nose test xfails.

- [ ] **Step 4: Implement `_keep_consistent` and the new argument**

In `core/vertical/regrid.py`, add before `hycom1_column` (not disk-cached: it lives next to kernels that call `remap`):

```python
@njit
def _keep_consistent(h_top, rho_top, targets, h_mass, x):  # pragma: no cover
    """
    Isopycnal layer preservation (spec §3, Plan B2c).  Layer j (surface first, between the
    targets t_j and t_j+1) is consistent if it has no mass (h < h_mass) or t_j ≤ ρ_j ≤ t_j+1
    with its own density.  An interface between two consistent layers stays where the
    Lagrangian step left it; the others keep their P1M position, clipped between the nearest
    kept interfaces above and below so they never push a kept one.  x: depths from the surface
    (0 … D, surface first), overwritten in place.  Pure isopycnal layers have no spurious
    diffusion (White et al. 2009, figs. 11-12); hybgen moves only layers that left their target
    (Bleck 2002) — the principle, without porting hybgen.
    """
    n = h_top.size
    ok = np.empty(n, np.bool_)
    for j in range(n):
        ok[j] = h_top[j] < h_mass or (targets[j] <= rho_top[j] <= targets[j + 1])
    xg = np.empty(n + 1)
    xg[0] = 0.0
    for j in range(n):
        xg[j + 1] = xg[j] + h_top[j]
    kept = np.zeros(n + 1, np.bool_)
    kept[0] = True
    kept[n] = True
    for j in range(1, n):
        if ok[j - 1] and ok[j]:
            x[j] = xg[j]
            kept[j] = True
    lo = xg[0]
    for j in range(1, n):
        if kept[j]:
            lo = x[j]
        elif x[j] < lo:
            x[j] = lo
    hi = xg[n]
    for j in range(n - 1, 0, -1):
        if kept[j]:
            hi = x[j]
        elif x[j] > hi:
            x[j] = hi
```

Change `hycom1_column` to (only the signature, the saved raw densities and the call after `_p1m_h2_positions`; the z\* floor block also gets a final monotonic pass — review focus 3):

```python
@njit
def hycom1_column(H, eta, h_bed, rho_bed, rho_target, dz_top, max_h_top, h_min, keep, zw):  # pragma: no cover
    """Hybrid z*–isopycnal interfaces (bed first) — see the module docstring."""
    n = h_bed.size
    D = H + eta
    h_top = h_bed[::-1].copy()
    rho_own = rho_bed[::-1].copy()                           # each layer's own density (B2c test)
    rho = rho_own.copy()
    h_mass = 2.0 * h_min                                     # massless-layer fill (unchanged, hybgen.F90:904-925)
    last = -1.0
    for k in range(n):
        if h_top[k] >= h_mass:
            last = rho[k]
        elif last >= 0.0:
            rho[k] = last
    for k in range(n):                                       # top run without mass: from below
        if h_top[k] >= h_mass:
            break
    if k > 0 and h_top[k] >= h_mass:
        for kk in range(k):
            rho[kk] = rho[k]
    for k in range(n - 2, -1, -1):                           # monotonic (coord_hycom.F90:218)
        rho[k] = min(rho[k], rho[k + 1])
    x = np.empty(n + 1)
    _p1m_h2_positions(h_top, rho, rho_target, x)
    if keep:
        _keep_consistent(h_top, rho_own, rho_target, 2.0 * h_min, x)
    s = D / H                                                # z* floor (coord_hycom.F90:255-267)
    nominal = 0.0
    for k in range(1, n + 1):
        nominal += dz_top[k - 1] * s
        x[k] = min(max(x[k], nominal), D)
    for k in range(1, n):                                    # maximum layer thickness (:270-279)
        x[k] = min(x[k], x[k - 1] + max_h_top[k - 1])
    for k in range(1, n + 1):                                # the floor may push a kept interface past
        x[k] = max(x[k], x[k - 1])                           # a regridded one: restore the order
    x[n] = D
    for j in range(n + 1):
        zw[j] = eta - x[n - j]
    zw[0] = -H
    zw[n] = eta
    enforce_min_thickness(zw, h_min)
```

(The fill and monotonising loops are the current ones, now on `rho`; only `rho_own` is new.) Keep `hycom1_column` decorated `@njit` without `cache=True` as it is now.

In `_regrid_columns` add the argument `keep` after `h_min` and pass it: `hycom1_column(..., max_h_top, h_min, keep, col)`. In `regrid_columns` pass `bool(params.get("keep_isopycnal", True))` after the `h_min` argument, and add to its docstring: "keep_isopycnal (hybrid, default True): interfaces between isopycnally consistent layers stay put (spec §3, B2c); False = HYCOM1 of MOM6 (P1M everywhere)."

In the module docstring, after "and the minimum thickness.", add:
"With keep_isopycnal (default) an interface between two layers whose densities lie in their target brackets (or that have no mass) is not moved (Plan B2c; White et al. 2009 — pure isopycnal layers have no spurious diffusion)."  and add the reference line `White, L., Adcroft, A. & Hallberg, R. (2009). J. Comput. Phys. 228, 8665–8692.`

- [ ] **Step 5: Run the regrid tests and the MOM6 comparison**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_regrid.py tests/test_vertical_vs_mom6.py -q`
Expected: all pass (44 regrid + MOM6 comparison).
Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k hybrid_lock -q`
Expected: the mixing test PASSES (prototype 2.5e7 J); the nose test still xfails; the body test (`..._fronts_leave_the_lock_...`) may FAIL — that is Task 2.

- [ ] **Step 6: Re-run the hybgen comparison and record it**

Run: `NUMBA_NUM_THREADS=5 python validation/mom6_column/compare_hybgen.py` then `git diff validation/mom6_column/hybgen_comparison.txt`.
Expected: the table regenerates; columns whose layers start inside their brackets change least. Add one paragraph to `validation/mom6_column/README.md` (after the hybgen comparison section) with the new numbers of the "10 ciclos" columns and the sentence: "Con `keep_isopycnal` (Plan B2c, por defecto) las capas que ya están en su rango de densidad no se mueven; las diferencias con hybgen que quedan son las de las capas inconsistentes."

- [ ] **Step 7: Commit (with Esteban's authorisation)**

```bash
git add core/vertical/regrid.py tests/test_vertical_regrid.py tests/test_sigma_ale.py validation/mom6_column/compare_hybgen.py validation/mom6_column/hybgen_comparison.txt validation/mom6_column/README.md
git commit -m "Hybrid rule: keep interfaces between isopycnally consistent layers (ALE Plan B2c)"
```

---

### Task 2: B2c in the model — body speed of the hybrid current

**Files:**
- Modify: `tests/test_sigma_ale.py` (marker of `test_hybrid_lock_exchange_fronts_leave_the_lock_at_the_benjamin_speed`)
- Modify: `core/dynamics/sigma_baroclinic.py` (docstring of `regrid_params` only)

**Interfaces:**
- Consumes: `regrid_params["keep_isopycnal"]` (default True) from Task 1; `_lock_run(rule)` in `tests/test_sigma_ale.py` returns `dict(speed={1.0: (dense, light), 5.0: (dense, light)}, rpe_gain, heat, one_thick_layer, consistent)`.

- [ ] **Step 1: Measure the body speed with B2c**

Run: `NUMBA_NUM_THREADS=5 python -c "import tests.test_sigma_ale as t; print(t._lock_run('hybrid'))"`
Expected (prototype): body 0.77 / 0.91 U, i.e. `test_hybrid_lock_exchange_fronts_leave_the_lock_at_the_benjamin_speed` fails for the dense front.

- [ ] **Step 2: Run the body test**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k fronts_leave -q`
Expected: FAIL if the measured dense body speed is outside ±10 %; PASS otherwise (then skip Step 3).

- [ ] **Step 3: Record the body-speed result honestly**

If the body test fails as in the prototype, mark it (do not loosen the tolerance):

```python
@pytest.mark.xfail(strict=True, reason="ALE Plan B2c: with isopycnal layers preserved the dense body runs at "
                   "<measured> U (was 1.05 U with P1M everywhere); the film ahead carries the volume — pending B2b")
```

with `<measured>` replaced by the number read in Step 2. Ledger it as a ruling (spec §7 item 6 requires ±10 %; B2b, Task 3–5, is the planned fix; if B2b does not restore it, this is a decision for Esteban — stop and report).

- [ ] **Step 4: Document the parameter**

In the `SigmaBaroclinic` docstring entry for `regrid_params`, add: "hybrid: `keep_isopycnal` (default True) keeps interfaces between isopycnally consistent layers (spec §3, B2c)".

- [ ] **Step 5: Run the ALE suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_vertical_ale.py tests/test_vertical_regrid.py -q`
Expected: all pass except the strict xfails (now passing tests: mixing; xfail: nose; possibly body).

- [ ] **Step 6: Commit (with Esteban's authorisation)**

```bash
git add tests/test_sigma_ale.py core/dynamics/sigma_baroclinic.py
git commit -m "Hybrid lock exchange: RPE gain test for isopycnal layer preservation (ALE Plan B2c)"
```

---

### Task 3: PPM layer thickness flux with barotropic adjustment (B2b, column kernels)

Algorithm: MOM6 `src/core/MOM_continuity_PPM.F90` (commit 0e9a561), OM4 defaults (positive-definite limiter, `MONOTONIC_CONTINUITY=F`, `SIMPLE_2ND=F`, `UPWIND_1ST=F`, `CONT_PPM_BETTER_ITER=T`, `CONTINUITY_CFL_LIMIT=0.5`):
- edge values: `PPM_reconstruction_x` 2827-2942 (limited slope, Lin 1994 eq. B2, at 2899-2913; land neighbour → the cell itself at 2936-2937);
- positive-definite limiter `PPM_limit_pos` 3184-3226 with h_min = 2·Angstrom = 2e-10 m;
- swept-volume flux and marginal thickness `flux_elem` 1111-1202 (formulas at 1182-1183 and 1193-1194);
- uniform velocity correction `zonal_flux_adjust` 1368-1546 (Newton with bracketing bisection, ≤ 20 iterations, du bounded so no layer exceeds CFL 0.5, 773-874).

Deliberate differences, each a ruling to ledger:
- after the Newton iteration the remaining residual is shared like HYCOM (`cnuity.F90:585-620`), so Σ_k Q_k = Q_bt to round-off, because the model checks Σh = H + η to 1e-12 (MOM6 accepts its tolerance);
- the Courant number in the swept integral is capped at 1 (MOM6 has no guard; review focus 5);
- in z\*/hybrid the swept-mean thickness is capped by the clipped donor thickness where the clip is active (q < h_up), so water facing rock still does not move (B2a, spec §2).

**Files:**
- Modify: `core/vertical/ale.py` (new `ppm_edges_x`, `_swept_thickness`, `ppm_transport_x`; constant `H_PPM_MIN = 2e-10`; docstring paragraph)
- Test: `tests/test_vertical_ale.py`

**Interfaces:**
- Consumes: `face_transport_thickness(h, H, active, clip) -> (q_e, q_w)`, `upwind_transport_x(q_e, q_w, ut, L, Q_bt)` (B2a, existing).
- Produces: `ppm_edges_x(h: ndarray (N,ny,nx), active: ndarray (ny,nx) bool, h_min: float = H_PPM_MIN) -> (h_w, h_e)` each (N, ny, nx).
- Produces: `ppm_fluxes_x(h, ut, L, q_e, q_w, active, dt: float, dx_t: ndarray (ny,nx)) -> (F, dF)` — uncorrected PPM layer fluxes [m³ s⁻¹] and L·h_marg [m²], each (N, ny, nx+1).
- Produces: `ppm_transport_x(h, ut, L, Q_bt, q_e, q_w, active, dt: float, dx_t: ndarray (ny,nx), max_iter: int = 20) -> Q` (N, ny, nx+1) [m³ s⁻¹], Σ_k Q_k = Q_bt.

- [ ] **Step 1: Write the failing tests**

Append to `tests/test_vertical_ale.py`:

```python
# ---------------------------------------------------------------- B2b: PPM thickness flux (MOM6 continuity_PPM)
def _ppm_case(h, u, dt=10.0, dx=100.0, L=10.0, clip=False, H=None, active=None):
    n, ny, nx = h.shape
    active = np.ones((ny, nx), bool) if active is None else active
    H = h.sum(axis=0) if H is None else H
    qe, qw = ale.face_transport_thickness(h, H, active, clip)
    dx_t = np.full((ny, nx), dx)
    Lf = np.full((ny, nx + 1), L)
    return qe, qw, dx_t, Lf, active


def test_ppm_edges_are_exact_for_a_linear_thickness_and_positive_for_vanished_cells():
    x = np.arange(8) + 0.5
    h = (2.0 + 0.5 * x)[None, None, :]
    hw, he = ale.ppm_edges_x(h, np.ones((1, 8), bool))
    # cells whose neighbours have exact slopes (the edge cells have zero slope, MOM6 2899-2913)
    np.testing.assert_allclose(he[0, 0, 2:-2], 2.0 + 0.5 * (x[2:-2] + 0.5), rtol=1e-13)
    np.testing.assert_allclose(hw[0, 0, 2:-2], 2.0 + 0.5 * (x[2:-2] - 0.5), rtol=1e-13)
    rng = np.random.default_rng(1)
    h = rng.uniform(0.0, 5.0, (6, 3, 10))
    h[rng.uniform(size=h.shape) < 0.3] = 0.0                              # vanished cells
    hw, he = ale.ppm_edges_x(h, np.ones((3, 10), bool))
    assert hw.min() >= -1e-14 and he.min() >= -1e-14
    np.testing.assert_array_equal(hw[h == 0], 0.0)                       # flattened, nothing to export
    np.testing.assert_array_equal(he[h == 0], 0.0)


def test_ppm_transport_is_the_swept_mean_of_the_parabola():
    # linear thickness, u > 0: the swept volume is the last fraction C of the donor cell,
    # mean thickness = h_E − b·C·dx/2 (MOM6 flux_elem 1182)
    x = np.arange(8) + 0.5
    b = 0.5                                                              # per cell
    h = (2.0 + b * x)[None, None, :]
    u = np.zeros((1, 1, 9))
    u[0, 0, 4] = 3.0                                                     # C = 3·10/100 = 0.3
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    Qbt = np.zeros((1, 9))
    Qbt[0, 4] = Lf[0, 4] * 3.0 * (2.0 + b * 4.0 - b * 0.3 / 2)           # what the swept mean gives
    Q = ale.ppm_transport_x(h, u, Lf, Qbt, qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q[0, 0, 4], Qbt[0, 4], rtol=1e-12)
    assert (Q[0, 0, [0, 8]] == 0).all()


def test_ppm_transport_adds_up_to_the_free_surface_flux_with_a_uniform_correction():
    rng = np.random.default_rng(4)
    h = rng.uniform(0.2, 5.0, (6, 2, 7))
    h[2, :, 3] = 0.0
    u = rng.normal(0.0, 0.5, (6, 2, 8))
    u[..., [0, -1]] = 0.0
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    F0, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    Qbt = F0.sum(axis=0) + rng.normal(0.0, 5.0, (2, 8))                  # needs a correction du ≠ 0
    Qbt[:, [0, -1]] = 0.0
    Q = ale.ppm_transport_x(h, u, Lf, Qbt, qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q.sum(axis=0), Qbt, rtol=1e-12, atol=1e-12)
    assert np.isfinite(Q).all()


def test_ppm_transport_without_residual_keeps_the_uncorrected_fluxes():
    rng = np.random.default_rng(5)
    h = rng.uniform(0.2, 5.0, (4, 1, 6))
    u = rng.normal(0.0, 0.5, (4, 1, 7))
    u[..., [0, -1]] = 0.0
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    F0, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    Q = ale.ppm_transport_x(h, u, Lf, F0.sum(axis=0), qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q, F0, rtol=1e-12, atol=1e-12)


def test_ppm_transport_next_to_land_and_on_the_edge_is_zero_and_finite():
    # review focus 4
    h = np.full((3, 1, 5), 4.0)
    act = np.array([[True, True, False, True, True]])
    u = np.full((3, 1, 6), 0.2)
    qe, qw, dx_t, Lf, _ = _ppm_case(h, u, active=act)
    Q = ale.ppm_transport_x(h, u, Lf, np.zeros((1, 6)), qe, qw, act, 10.0, dx_t)
    assert np.isfinite(Q).all()
    assert (Q[:, 0, 2:4] == 0).all()


def test_ppm_transport_does_not_empty_a_vanished_donor_at_high_courant_number():
    # review focus 5: a 1 mm layer with C = 0.9 exports at most its own volume
    h = np.array([[[1e-3, 1e-3, 5.0, 5.0]], [[5.0, 5.0, 1e-3, 1e-3]]])
    u = np.zeros((2, 1, 5))
    u[0, 0, 2] = 9.0                                                     # C = 9·10/100 = 0.9
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    Q, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    assert Q[0, 0, 2] * 10.0 <= 1e-3 * 100.0 * 10.0 + 1e-12              # Q dt ≤ h A of the donor


def test_ppm_transport_respects_the_clip_at_the_shallower_bed():
    # z* layer facing rock: zero flux whatever the parabola says
    deep = np.full(6, 100.0)
    shallow = np.array([1e-3, 1e-3, 1e-3, 1e-3, 63.996, 100.0])
    h = np.stack([deep, deep, shallow, shallow], axis=1)[:, None, :]
    H = np.array([[600.0, 600.0, 164.0, 164.0]])
    u = np.zeros((6, 1, 5))
    u[:, 0, 2] = 0.1
    qe, qw, dx_t, Lf, act = _ppm_case(h, u, clip=True, H=H)
    Q, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q[:4, 0, 2], 0.0, atol=1e-12)
```

- [ ] **Step 2: Run them to see them fail**

Run: `python -m pytest tests/test_vertical_ale.py -q`
Expected: the 7 new tests FAIL with `AttributeError: module 'core.vertical.ale' has no attribute 'ppm_edges_x'` (or `ppm_fluxes_x`).

- [ ] **Step 3: Implement**

Append to `core/vertical/ale.py`:

```python
H_PPM_MIN = 2e-10      # [m] positive-definite floor of the PPM thickness (2 Angstrom, MOM_continuity_PPM.F90:534)


def ppm_edges_x(h: np.ndarray, active: np.ndarray, h_min: float = H_PPM_MIN) -> tuple[np.ndarray, np.ndarray]:
    """
    West and east edge values of the layer thickness in each cell [m] (MOM6 PPM_reconstruction_x,
    MOM_continuity_PPM.F90:2827-2942): limited slope (Lin 1994, eq. B2; 0 next to land),
    h_W = (h_i−1 + h_i)/2 + (s_i−1 − s_i)/6, a land or outside neighbour replaced by the cell
    itself, then the positive-definite limiter PPM_limit_pos (:3184-3226): a parabola whose
    minimum falls inside the cell is scaled so it stays ≥ h_min; cells with h ≤ h_min are flat.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    m = active.astype(float)[None]
    z = np.zeros_like(m[..., :1])
    mW = np.concatenate([z, m[..., :-1]], axis=-1)
    mE = np.concatenate([m[..., 1:], z], axis=-1)
    hm1 = mW * np.concatenate([h[..., :1], h[..., :-1]], axis=-1) + (1 - mW) * h
    hp1 = mE * np.concatenate([h[..., 1:], h[..., -1:]], axis=-1) + (1 - mE) * h
    slp = 0.5 * (hp1 - hm1)
    dmx = np.maximum(np.maximum(hp1, hm1), h) - h
    dmn = h - np.minimum(np.minimum(hp1, hm1), h)
    slp = np.sign(slp) * np.minimum(np.abs(slp), 2.0 * np.minimum(dmx, dmn))
    slp = np.where(mW * mE * m > 0, slp, 0.0)
    zs = np.zeros_like(slp[..., :1])
    slp_w = np.concatenate([zs, slp[..., :-1]], axis=-1)
    slp_e = np.concatenate([slp[..., 1:], zs], axis=-1)
    h_w = 0.5 * (hm1 + h) + (slp_w - slp) / 6.0
    h_e = 0.5 * (hp1 + h) + (slp - slp_e) / 6.0
    curv = 3.0 * ((h_w + h_e) - 2.0 * h)                             # PPM_limit_pos: only minima
    dh = h_e - h_w
    inside = (curv > 0) & (np.abs(dh) < curv)
    flat = inside & (h <= h_min)
    num = 12.0 * curv * (h - h_min)
    den = curv**2 + 3.0 * dh**2
    scale_ = inside & ~flat & (num < den)
    s = np.where(scale_, num / np.where(den > 0, den, 1.0), 1.0)
    h_w = np.where(flat, h, h + s * (h_w - h))
    h_e = np.where(flat, h, h + s * (h_e - h))
    return h_w, h_e


def _swept_thickness(h, h_w, h_e, u, dt, dx_t, q_e, q_w):
    """
    Mean thickness of the volume swept through each u-face in dt and its derivative factor
    (MOM6 flux_elem, MOM_continuity_PPM.F90:1182-1183, 1193-1194), with C = |u| dt/dx of the
    donor capped at 1 and, where the bed clip is active (q < h_donor), capped by q.
    Returns (mean, marginal) on the faces (N, ny, nx+1).
    """
    pad = lambda a: np.concatenate([a[..., :1], a, a[..., -1:]], axis=-1)   # noqa: E731
    hp, wp, ep = pad(h), pad(h_w), pad(h_e)
    dxp = np.concatenate([dx_t[:, :1], dx_t, dx_t[:, -1:]], axis=-1)[None]
    # face f: west cell = padded index f, east cell = padded index f+1
    hW, wW, eW, dW = hp[..., :-1], wp[..., :-1], ep[..., :-1], dxp[..., :-1]
    hE, wE, eE, dE = hp[..., 1:], wp[..., 1:], ep[..., 1:], dxp[..., 1:]
    pos = u >= 0
    C = np.minimum(np.abs(u) * dt / np.where(pos, dW, dE), 1.0)
    curv3 = np.where(pos, (wW + eW) - 2.0 * hW, (wE + eE) - 2.0 * hE)
    dh = np.where(pos, wW - eW, eE - wE)
    edge = np.where(pos, eW, wE)
    mean = edge + C * (0.5 * dh + curv3 * (C - 1.5))
    marg = edge + C * (dh + 3.0 * curv3 * (C - 1.0))
    q = np.where(pos, q_e, q_w)
    h_up = np.where(pos, hW, hE)
    cap = np.where(q < h_up, q, np.inf)                              # bed clip (z*, hybrid)
    return np.maximum(np.minimum(mean, cap), 0.0), np.maximum(np.minimum(marg, cap), 0.0)


def _open_faces(active: np.ndarray) -> np.ndarray:
    """u-faces with water on both sides (domain-edge faces: the inner cell), (1, ny, nx+1)."""
    aW = np.concatenate([active[:, :1], active], axis=-1)
    aE = np.concatenate([active, active[:, -1:]], axis=-1)
    return (aW & aE)[None]


def _ppm_fluxes(h, h_w, h_e, u, L, q_e, q_w, open_, dt, dx_t):
    mean, marg = _swept_thickness(h, h_w, h_e, u, dt, dx_t, q_e, q_w)
    return np.where(open_, L[None] * u * mean, 0.0), np.where(open_, L[None] * marg, 0.0)


def ppm_fluxes_x(h: np.ndarray, ut: np.ndarray, L: np.ndarray, q_e: np.ndarray, q_w: np.ndarray,
                 active: np.ndarray, dt: float, dx_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Uncorrected PPM layer fluxes F = L u h̄ [m³ s⁻¹] and dF/du = L h_marg [m²] at the u-faces
    (MOM6 flux_elem, MOM_continuity_PPM.F90:1111-1202), zero on faces without water on both sides.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    h_w, h_e = ppm_edges_x(h, active)
    return _ppm_fluxes(h, h_w, h_e, ut, L, q_e, q_w, _open_faces(active), dt, dx_t)


def ppm_transport_x(h: np.ndarray, ut: np.ndarray, L: np.ndarray, Q_bt: np.ndarray, q_e: np.ndarray,
                    q_w: np.ndarray, active: np.ndarray, dt: float, dx_t: np.ndarray,
                    max_iter: int = 20) -> np.ndarray:
    """
    Layer volume fluxes through the u-faces [m³ s⁻¹] with PPM thickness (MOM6 continuity_PPM,
    OM4 defaults) and a uniform velocity correction du per face column so that
    Σ_k L (u_k + du) h̄_k(u_k + du) = Q_bt (zonal_flux_adjust, MOM_continuity_PPM.F90:1368-1546:
    Newton with bracketing bisection; du keeps every layer under CFL ½, :773-874).  The
    remaining residual is shared like HYCOM (cnuity.F90:585-620) so Σ_k Q_k = Q_bt to
    round-off.  Only the fluxes change, not u (as MOM6).  For the v-faces call it on swapped axes.
    """
    h = np.where(active[None], np.nan_to_num(h), 0.0)
    h_w, h_e = ppm_edges_x(h, active)
    open_ = _open_faces(active)
    dxW = np.concatenate([dx_t[:, :1], dx_t], axis=-1)[None]
    dxE = np.concatenate([dx_t, dx_t[:, -1:]], axis=-1)[None]

    def fluxes(uu):
        return _ppm_fluxes(h, h_w, h_e, uu, L, q_e, q_w, open_, dt, dx_t)

    lim_w, lim_e = 0.5 * dxW / dt, 0.5 * dxE / dt                    # CFL ½ (CONTINUITY_CFL_LIMIT)
    du_max = np.maximum(np.minimum(lim_w[0], (lim_w - ut).min(axis=0)), 0.0)
    du_min = np.minimum(np.maximum(-lim_e[0], (-(lim_e + ut)).max(axis=0)), 0.0)
    du = np.zeros_like(Q_bt)
    F, dF = fluxes(ut)
    err = F.sum(axis=0) - Q_bt
    lo, hi = du_min.copy(), du_max.copy()
    for _ in range(max_iter):
        hi = np.where(err > 0, du, hi)
        lo = np.where(err < 0, du, lo)
        dsum = dF.sum(axis=0)
        step = np.where(dsum > 0, -err / np.where(dsum > 0, dsum, 1.0), 0.0)
        new = du + step
        new = np.where((step > 0) & (new >= hi), 0.5 * (du + hi), new)
        new = np.where((step < 0) & (new <= lo), 0.5 * (du + lo), new)
        du = new
        F, dF = fluxes(ut + du[None])
        err = F.sum(axis=0) - Q_bt
        if np.all(np.abs(err) <= 1e-12 * np.maximum(np.abs(Q_bt), 1.0)):
            break
    R = Q_bt - F.sum(axis=0)                                         # exact closure (HYCOM)
    qR = np.where((R >= 0)[None], q_e, q_w)
    alt = np.where((R >= 0)[None], q_w, q_e)
    qR = np.where((qR.sum(axis=0) > 0)[None], qR, alt)
    tot = qR.sum(axis=0)
    return np.where(open_, F + R[None] * qR / np.where(tot > 0, tot, 1.0)[None], 0.0)
```

Add to the module docstring, after the B2a transport bullet: "* PPM (Plan B2b, `ppm_transport_x`): MOM6 continuity_PPM — positive-definite PPM edge thicknesses, the mean thickness of the volume swept in Δt, and a uniform velocity correction per column so the layer fluxes add up to the free-surface flux (Adcroft et al. 2019, §A)."

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_vertical_ale.py -q`
Expected: 18 passed. If `test_ppm_transport_does_not_empty_a_vanished_donor_at_high_courant_number` fails, the swept mean exceeded h_donor/C: check the C cap and the limiter before touching the test.

- [ ] **Step 5: Commit (with Esteban's authorisation)**

```bash
git add core/vertical/ale.py tests/test_vertical_ale.py
git commit -m "ALE: PPM layer thickness flux with barotropic adjustment (MOM6 continuity_PPM, Plan B2b)"
```

---

### Task 4: `thickness_flux` option in the model

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (constructor signature and docstring near `:609-624`, `_setup_ale`, `_step_ale` transport block)
- Test: `tests/test_sigma_ale.py`

**Interfaces:**
- Consumes: `ppm_transport_x(...)` from Task 3; `self._transport_thickness(h)` (B2a) returning `((q_e, q_w), (q_n, q_s))`, the v-pair in swapped layout (N, nx, ny+1).
- Produces: `SigmaBaroclinic(..., thickness_flux: str = "upwind")`, accepted values `"upwind"`, `"ppm"`; stored as `self.thickness_flux`.

- [ ] **Step 1: Write the failing tests**

Add to `tests/test_sigma_ale.py` (configuration section: one more `pytest.param` row; plus two model tests):

In the `test_ale_configuration_errors` parametrize list add
`(dict(thickness_flux="weno"), "thickness_flux"),`.

```python
@pytest.mark.parametrize("rule", ["sigma", "zstar"])
def test_ppm_thickness_flux_keeps_uniform_tracer_and_column_consistency(rule):
    m = _ale_box(rule, thickness_flux="ppm")
    x = (np.arange(32) + 0.5) * 2000.0
    st = m.initial_state(0.5 * np.cos(np.pi * x / 64e3)[None, :], np.full((10, 1, 32), 7.0))
    c0 = m.tracer_content(st)
    for _ in range(200):
        st = m.step(st)
        np.testing.assert_allclose(st.h.sum(0), 20.0 + st.eta, rtol=1e-12)
        assert (st.h >= 0).all()
    np.testing.assert_allclose(st.T, 7.0, atol=1e-11)
    assert abs(m.tracer_content(st) / c0 - 1) < 1e-13


def test_ppm_thickness_flux_keeps_the_zstar_step_at_rest():
    m, st = _step_section("zstar", thickness_flux="ppm")
    for _ in range(720):
        st = m.step(st)
    assert np.abs(st.u).max() < 1e-10
```

and give `_step_section(coord, nz=30, **extra)` a `**extra` passed into the ALE `kw` dict (`kw.update(extra)` inside the `zstar` branch).

- [ ] **Step 2: Run them to see them fail**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -k "ppm or configuration" -q`
Expected: FAIL with `TypeError: ... unexpected keyword argument 'thickness_flux'`.

- [ ] **Step 3: Implement**

Constructor: add `thickness_flux: str = "upwind"` after `init_iterations` in the signature, document it ("ALE layer thickness at the faces: 'upwind' = donor cell (B2a), 'ppm' = MOM6 continuity_PPM with the barotropic adjustment (B2b)"), and pass it to `_setup_ale(...)`, which stores it after the other checks:

```python
        if thickness_flux not in ("upwind", "ppm"):
            raise ValueError(f"unknown thickness_flux {thickness_flux!r}; use 'upwind' or 'ppm'")
        self.thickness_flux = thickness_flux
```

(validate before the `if not self.ale: return` so a typo is caught in σ mode too; store `self.thickness_flux` there as well.)

In `_step_ale`, replace the two transport lines with:

```python
        if self.thickness_flux == "ppm":
            dx_t = np.asarray(self.hg["dx_t"].values, float)
            dy_t = np.asarray(self.hg["dy_t"].values, float)
            Qx = ppm_transport_x(st.h, utx, self.fs.Lu, Qbx, qe, qw, act, dt, dx_t)
            Qy = ppm_transport_x(st.h.swapaxes(1, 2), uty.swapaxes(1, 2), self.fs.Lv.T, Qby.T, qn, qs, act.T,
                                 dt, dy_t.T).swapaxes(1, 2)
        else:
            Qx = upwind_transport_x(qe, qw, utx, self.fs.Lu, Qbx)
            Qy = upwind_transport_x(qn, qs, uty.swapaxes(1, 2), self.fs.Lv.T, Qby.T).swapaxes(1, 2)
```

and import `ppm_transport_x` with the other `core.vertical.ale` names.

- [ ] **Step 4: Run the ALE tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_vertical_ale.py -q`
Expected: all pass (the strict xfails still xfail).

- [ ] **Step 5: Commit (with Esteban's authorisation)**

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py
git commit -m "ALE model: thickness_flux='ppm' option (Plan B2b)"
```

---

### Task 5: Decision gate — is B2b adopted?

**Files:**
- Modify: `tests/test_sigma_ale.py` (`_hybrid_lock`, `_lock_run` gain a `flux` argument; xfail markers per the outcome)
- Modify (if adopted): `core/dynamics/sigma_baroclinic.py` default `thickness_flux="ppm"`
- Modify: `docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md` §2 (record the measured numbers and the outcome)

**Interfaces:**
- Consumes: `thickness_flux` from Task 4; `_lock_run(rule, ...)`.
- Produces: `_hybrid_lock(nx=128, nz=10, depth=20.0, dt=20.0, flux="upwind")`, `_lock_run(rule, nx=128, nz=10, depth=20.0, dt=20.0, flux="upwind")` (cached per argument tuple).

- [ ] **Step 1: Thread the option through the lock helpers**

`_hybrid_lock(..., flux="upwind")` passes `thickness_flux=flux` to `_ale_box("hybrid", ...)`; `_lock_run(..., flux="upwind")` passes it to `_hybrid_lock` and to the σ `_ale_box`.

- [ ] **Step 2: Measure**

Run (one at a time, 5 threads):

```bash
NUMBA_NUM_THREADS=5 python -c "import tests.test_sigma_ale as t; print('upwind', t._lock_run('hybrid')); print('ppm', t._lock_run('hybrid', flux='ppm')); print('sigma ppm', t._lock_run('sigma', flux='ppm'))"
```

and, with a scratchpad copy of `validation/seamount/run_seamount.py`'s `setup` (do not write `validation/seamount/out/`), the z\* seamount with FV + ρ_ref and `thickness_flux="ppm"` for 10 days.
Expected to decide: hybrid `ppm` 1 m front within ±10 % for both currents, body (5 m) within ±10 %, RPE gain < 0.2 × σ; z\* seamount day 10 ≤ 0.15 cm/s; z\* step < 1e-10 m/s (Task 4 test).

- [ ] **Step 3a: If every criterion holds — adopt**

Set `thickness_flux: str = "ppm"` as the constructor default; in `_lock_run` make `flux="ppm"` the default; remove the strict xfail markers of `test_hybrid_lock_exchange_nose_moves_at_the_benjamin_speed` and (if Task 2 added one) of the body test; run `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q` and expect all pass with the remaining xfails (2: Thacker strict, FV vs Jacobian). Ledger `Ruling: B2b adopted — <numbers>`.

- [ ] **Step 3b: If any criterion fails — do not adopt**

Keep `"upwind"` as default; keep the `ppm` option only if it improved at least one measured number without worsening another (state which), otherwise revert Tasks 3–4 with `git revert` of their commits. Update the xfail reasons with the measured numbers. **Stop and report to Esteban**: spec §7 item 6 requires the body at ±10 %; the next design decision is his.

- [ ] **Step 4: Record in the spec**

In spec §2, under the "Plan B2b" bullet, add one line: "Medido el 2026-10-1x: frente 1 m … / …, cuerpo … / …, RPE … J; adoptado | no adoptado."

- [ ] **Step 5: Commit (with Esteban's authorisation)**

```bash
git add tests/test_sigma_ale.py core/dynamics/sigma_baroclinic.py docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md
git commit -m "ALE Plan B2b: decision on the PPM thickness flux (measured on the hybrid lock exchange)"
```

---

### Task 6: Documentation and scorecard

**Files:**
- Modify: `CLAUDE.md` (rows of `regrid.py`, `ale.py`, ALE mode; total test count)
- Modify: `docs/design/vision_roadmap.md` §7 (row "Lock exchange híbrido isopicno")
- Modify: memory `~/.claude/projects/-Users-estebancruz-pyOCEAN/memory/project_status.md`

- [ ] **Step 1: Run the full suite and note the count**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider | tail -1`
Expected: all pass; note "N passed, M xfailed".

- [ ] **Step 2: Update the docs with the measured numbers**

- `CLAUDE.md` regrid row: append "isopycnal layer preservation (`keep_isopycnal`, default): interfaces between layers inside their target brackets stay put (White et al. 2009; hybgen principle)"; update its test count.
- `CLAUDE.md` `ale.py` row: append the PPM flux (if kept) and update the count; ALE-mode row: hybrid lock numbers (body, 1 m front, RPE vs ALE-σ) and the xfail list; total line "N total tests passing (+M …)".
- Roadmap §7 row: new numbers and status (✅ if all of §7 item 6 holds, 🔶 otherwise with the open item).
- Memory: replace the B2 paragraph of "Current (2026-10-12)" with the outcome and the commit hashes.

- [ ] **Step 3: Commit (with Esteban's authorisation)**

```bash
git add CLAUDE.md docs/design/vision_roadmap.md
git commit -m "Docs: ALE Plan B2 results (isopycnal layer preservation, thickness flux)"
```
