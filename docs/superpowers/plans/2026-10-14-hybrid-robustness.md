# Hybrid robustness (thin-layer coupling, outflow limiter, FCT) Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Make the ALE model (above all the hybrid coordinate) robust in near-vanished layers. Three pieces:
1. layer thickness can never go negative;
2. near-vanished layers stay viscously tied to their neighbours;
3. tracers stay inside their local bounds.

With all three in place, the hybrid can use the time-centred baroclinic transport (θ_bc = ½) like z\* and σ.

**Architecture:**
- **Outflow limiter.** A pure function `ale.limit_outflow` scales the layer fluxes that would empty a cell and gives the cut flux back to other layers of the same face, so Σ_k Q_k still equals the free-surface flux. `_step_ale` calls it between building Qx/Qy and `lagrangian_thickness`.
- **Thin-layer coupling.** The implicit vertical viscosity gets a MOM6-style coupling thickness (harmonic mean of the two sides) through a new `hu_visc`/`hv_visc` path. Transport, PGF, remap and bed drag keep their current thicknesses.
- **FCT.** Zalesak (1979) is wrapped around the SSP-RK3 WENO5 advection of the ALE step, using face fluxes from a kernel extracted (bit for bit) from `_advect_div`.

All three are options:
- they default to "off" until Task 4;
- Task 4 switches the defaults and θ_bc;
- Task 5 validates against the spec's criteria;
- Task 6 documents.

**Tech Stack:** Python 3, NumPy 1.26.4, Numba (≤ 5 threads), pytest.

**Spec:** `docs/superpowers/specs/2026-10-14-hybrid-robustness-design.md` (read it first: criteria, diagnosis and decision points live there).

## Global Constraints

- **Threads:** ≤ 5 (`NUMBA_NUM_THREADS=5`). One heavy run at a time. Never the efficiency cores, and never lower the priority.
- **Outputs:**
  - large outputs go to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`, never to the repo;
  - never commit `log_*.txt`, `*.npz` or `pyOCEAN-literature-review-guide.docx`;
  - `git add` takes explicit paths only, never directories.
- **Commit messages** end with:
  ```
  Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
  Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn
  ```
- **Reference models:** do not modify MOM6, HYCOM, MITgcm or CROCO sources.
- **Packages:** no installs. NumPy stays 1.26.4.
- **Index order:** (k, j, i) with k = 0 at the bed. u-faces (N, ny, nx+1), v-faces (N, ny+1, nx).
- **New code:**
  - type hints, and docstrings that explain the physics;
  - every scheme cites its source as `file:line` (MOM6/HYCOM) or paper + equation;
  - Numba kernels keep a NumPy oracle and a test that the two are equal.
- **Stop and ask Esteban** (spec "Puntos de decisión") if:
  - (1) Task 0 shows a dominant term other than the one Task 2 fixes;
  - (2) the HYCOM blending (1B) seems needed;
  - (3) the T extrema originate in the remap;
  - (4) a default ("mom6", "fct") fails criteria 4–6.

## Review Focus

1. **Bed drag with thin bed layers.** `first_layer_cd(dz[0])` must keep using the mean bed-layer thickness when the coupling thickness is harmonic; otherwise log-law drag changes in z\* partial cells. *Pinned in Task 2* (`test_viscous_coupling_keeps_the_bed_drag_thickness`).
2. **Open boundaries.** FCT faces at open edges must equal the upstream (donor) flux, so the antidiffusive flux there is 0 and inflow T_ext is respected. The limiter must also handle fluxes at open faces. *Pinned in Tasks 1 and 3* (`test_limit_outflow_handles_open_edge_fluxes`, `test_fct_open_edge_faces_are_upstream`).
3. **Mixing diagnostic stack.** `mix_diagnostic=True` advects [T, T²]. FCT must limit each tracer independently, and the T² inflow value is T_ext². *Pinned in Task 3* (`test_fct_limits_each_tracer_of_the_stack`).
4. **Restart.** `tracer_limiter`, `thin_layer_coupling` and `n_limited` must survive a restart bit for bit; a mismatch must be refused. *Pinned in Tasks 1–4.*
5. **Columns with no water to give.** A face flux that no layer of the donor column can carry must raise a clear error naming the barotropic CFL, not loop forever or silently drop mass. *Pinned in Task 1* (`test_limit_outflow_raises_when_the_column_cannot_supply_the_flux`).

---

### Task 0: Diagnosis (scratch only; nothing enters the repo)

**Files:** scratch scripts in the session scratchpad. They are not committed.

**Interfaces:** none.

Goal (spec §0): find which tendency accelerates the near-vanished layers, and where the out-of-range T originates (advection or remap).

- [ ] **Step 1: Write the step-decomposition probe**

```python
"""Task 0 probe: per-term velocity change on faces with a side < 1 cm, and where T leaves its range.
Usage: python probe.py overflow 100 0.6 [every] | python probe.py iw 0.01 0.5 [every]"""
import sys
import numpy as np
sys.path.insert(0, "/Users/estebancruz/pyOCEAN")
import core.dynamics.sigma_baroclinic as sb
from core.dynamics import vector_invariant as vi
from core.dynamics.nonlinear import laplacian_3d
from validation.ilicak_benchmark import cases, run_pyocean

sb.set_threads(5)
c = cases.get(sys.argv[1]); nu = float(sys.argv[2])
m, st = run_pyocean.build(c, "hybrid", nu)
m.theta_bc = float(sys.argv[3])
every = int(sys.argv[4]) if len(sys.argv) > 4 else 60
lo, hi = float(np.nanmin(st.T)), float(np.nanmax(st.T))
rec = {}


def thin_mask(h):
    hm = np.minimum(h[..., 1:], h[..., :-1])
    msk = np.zeros(h.shape[:2] + (h.shape[2] + 1,), bool)
    msk[..., 1:-1] = hm < 0.01
    return msk


def terms(st):
    """Explicit tendencies split: PGF, PV flux (Coriolis + vorticity), KE gradient, Laplacian viscosity."""
    g = m._vi
    pgf_u, _ = m._baroclinic_pgf(st.eta, st.T, st.h)
    zeta, q = vi.vorticity_and_pv(st.u, st.v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"], m._fq, st.h,
                                  g["area_t"], m.fs.active)
    Qx, Qy = st.Qx, st.Qy
    if Qx is None:
        hu, hv = m._mean_face_thickness(st.h)
        Qx, Qy = m.fs.Lu[None] * st.u * hu, m.fs.Lv[None] * st.v * hv
    cu, _ = vi.coriolis_pv(q, m._fq[None] + zeta, Qx, Qy, st.u, st.v, g["dx_u"], g["dy_v"], bound=True)
    K = vi.kinetic_energy(st.u, st.v, g["area_u"], g["area_v"], g["area_t"], m.ke_scheme)
    kx, _ = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
    lap = np.zeros_like(st.u)
    if m.ah > 0:
        wet_u = np.isfinite(m.fs.H_u)
        lap = m.ah * laplacian_3d(st.u, wet_u, *m._lap[0]) if getattr(m, "_lap", None) else lap
    return dict(pgf=pgf_u, pv=cu, ke=kx, lap=lap)


core_step = m.core.step
def core_wrap(eta, u, v, G_u=None, **kw):
    out = core_step(eta, u, v, G_u=G_u, **kw)
    rec["u_in"], rec["G"], rec["u_core"] = u.copy(), G_u.copy(), out[1].copy()
    return out
m.core.step = core_wrap
remap = m._remap_faces
def remap_wrap(vel, *a, **k):
    out = remap(vel, *a, **k)
    if vel.shape == rec.get("u_core", np.empty(0)).shape:
        rec["u_before_remap"], rec["u_after_remap"] = vel.copy(), out.copy()
    return out
m._remap_faces = remap_wrap
adv = m._advect
def adv_wrap(C, V0, V1, *a):
    out = adv(C, V0, V1, *a)
    rec["T_adv"], rec["V1"] = out[0].copy(), V1.copy()
    return out
m._advect = adv_wrap

print("t[h]  rms Δu on thin faces per step: dt*pgf  dt*pv  dt*ke  dt*lap | implicit+bt | remap | T_out adv/remap")
try:
    for n in range(1, 10**7):
        msk = thin_mask(st.h) & (np.abs(st.u) > 0)
        tt = terms(st) if n % every == 0 else None
        st_new = m.step(st)
        if n % every == 0 and msk.any():
            r = lambda a: float(np.sqrt(np.mean(a[msk] ** 2)))          # noqa: E731
            expl = {k: r(m.dt * v) for k, v in tt.items()}
            impl = r(rec["u_core"] - (rec["u_in"] + m.dt * rec["G"]))
            rmp = r(rec["u_after_remap"] - rec["u_before_remap"])
            thick = rec["V1"] > 1e-3 * m.area[None]
            out_adv = int(((rec["T_adv"] < lo - 1e-6) | (rec["T_adv"] > hi + 1e-6))[thick].sum())
            okn = st_new.h > 1e-3
            out_rem = int(((st_new.T < lo - 1e-6) | (st_new.T > hi + 1e-6))[okn].sum())
            print(f"{n * m.dt / 3600:6.2f} " + " ".join(f"{expl[k]:.2e}" for k in ("pgf", "pv", "ke", "lap"))
                  + f" | {impl:.2e} | {rmp:.2e} | {out_adv}/{out_rem}  max|u| thin {np.abs(st.u[msk]).max():.2f}",
                  flush=True)
        st = st_new
except RuntimeError as e:
    print("stop:", str(e)[:120])
```

- [ ] **Step 2: Run both cases, one at a time**

Run:
```bash
cd /Users/estebancruz/pyOCEAN && python <scratch>/probe.py overflow 100 0.6 60 2>&1 | grep -v Warn
cd /Users/estebancruz/pyOCEAN && python <scratch>/probe.py iw 0.01 0.5 2880 2>&1 | grep -v Warn
```

Expected:
- the overflow stops at ~6 h ("layer emptied");
- the IW stops at about day 113 (step ~32567);
- one table row per `every` steps.

- [ ] **Step 3: Read the tables and record the outcome**

Write in the ledger, one line per case:
- the dominant contributor to |Δu| on thin faces during the growth (2–5.5 h in the overflow; the last 20 days in the IW): pgf, pv, ke, lap, implicit+bt or remap;
- whether T first leaves its range after advection (`out_adv` > 0) or only after the remap (`out_rem` > 0 while `out_adv` = 0).

**Decision rule:**

| Dominant term | T origin | Action |
|---|---|---|
| ke, pv, lap, implicit+bt or remap (the terms a viscous tie suppresses, because a tied layer cannot drift from its neighbours) | advection | Continue with Task 1. |
| pgf | any | Stop: decision point 1. Report the numbers to Esteban. |
| any | remap only | Stop: decision point 3. |

---

### Task 1: Outflow limiter (h ≥ 0 guaranteed)

**Files:**
- Modify: `core/vertical/ale.py` (new `limit_outflow` after `lagrangian_thickness`, ~line 225)
- Modify: `core/dynamics/sigma_baroclinic.py` (`SigmaState.n_limited`; `_step_ale` between Qx/Qy and `lagrangian_thickness`, ~line 1692; the returned state)
- Modify: `core/restart.py` (`n_limited` attribute, lines ~122 and ~179)
- Test: `tests/test_vertical_ale.py`, `tests/test_restart.py`

**Interfaces:**
- Produces: `ale.limit_outflow(h: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, area: np.ndarray, dt: float, active: np.ndarray, h_floor: float = H_PPM_MIN, max_pass: int = 10) -> tuple[np.ndarray, np.ndarray, int]`. It returns `(Qx, Qy, n_limited)`, where `n_limited` counts the (k, j, i) cells scaled over all passes. When nothing exceeds, it returns the input arrays themselves (same objects).
- Produces: `SigmaState.n_limited: int = 0`.

*Ruling for the executor, already decided:* the spec asks for a Numba kernel with a NumPy oracle. `limit_outflow` is NumPy-vectorised instead. When no cell exceeds, it costs one pass of ~6 array operations, and its cost is measured in Step 7. Add a kernel only if Step 7 shows > 2 % of the BayIW step, and ledger that.

- [ ] **Step 1: Write the failing tests** (append to `tests/test_vertical_ale.py`)

```python
# ---------------------------------------------------------------- outflow limiter (spec 2026-10-14-hybrid-robustness §2)
def _thin_divergent_cell():
    """One row, 3 cells × 3 layers; layer 1 of the middle cell is 1 mm and loses through both faces more than it
    holds (Courant 0.45 west + 0.52 east plus a barotropic correction, the IW failure of 2026-10-14)."""
    dx, dt = 5000.0, 300.0
    h = np.array([[[50.0, 50.0, 50.0]], [[0.08, 0.001, 49.9]], [[50.0, 50.0, 50.0]]])
    area = np.full((1, 3), dx * dx)
    Qx = np.zeros((3, 1, 4))
    Qx[1, 0, 1] = -0.459 * 0.001 * dx * dx / dt          # west face, flows west out of the middle cell
    Qx[1, 0, 2] = +0.553 * 0.001 * dx * dx / dt          # east face, flows east out of the middle cell
    Qx[0, 0, 1:3] = [0.2, 0.3]
    Qx[2, 0, 1:3] = [-0.4, 0.1]
    return h, Qx, np.zeros((3, 2, 3)), area, dt, np.ones((1, 3), bool)


def test_limit_outflow_keeps_the_thin_layer_non_negative_and_the_face_sums():
    h, Qx, Qy, area, dt, act = _thin_divergent_cell()
    Qx2, Qy2, n = ale.limit_outflow(h, Qx, Qy, area, dt, act)
    div = Qx2[..., 1:] - Qx2[..., :-1] + Qy2[:, 1:] - Qy2[:, :-1]
    h_new = h - dt * div / area[None]
    assert n >= 1 and h_new.min() >= 0.0
    np.testing.assert_allclose(Qx2.sum(0), Qx.sum(0), rtol=1e-12, atol=1e-12)       # Σ_k Q_k per face kept
    assert np.all(np.sign(Qx2[1]) * np.sign(Qx[1]) >= 0)                              # no direction reversal


def test_limit_outflow_is_a_no_op_when_nothing_empties():
    h, Qx, Qy, area, dt, act = _thin_divergent_cell()
    h = h.copy()
    h[1, 0, 1] = 1.0                                                                  # now holds enough
    Qx2, Qy2, n = ale.limit_outflow(h, Qx, Qy, area, dt, act)
    assert n == 0 and Qx2 is Qx and Qy2 is Qy


def test_limit_outflow_reduces_counter_flowing_layers_when_no_same_way_layer_has_room():
    # two layers; layer 0 empties eastwards, layer 1 flows westwards: the cut must come off layer 1's inflow
    dx, dt = 1000.0, 10.0
    h = np.array([[[1e-3, 10.0]], [[10.0, 10.0]]])
    area = np.full((1, 2), dx * dx)
    Qx = np.zeros((2, 1, 3))
    Qx[0, 0, 1] = 2e-3 * dx * dx / dt            # layer 0: twice its content out of cell 0
    Qx[1, 0, 1] = -500.0                         # layer 1: westwards into cell 0 (can absorb the ~100 m³/s cut)
    Qx2, _, n = ale.limit_outflow(h, Qx, np.zeros((2, 2, 2)), area, dt, np.ones((1, 2), bool))
    assert n == 1
    np.testing.assert_allclose(Qx2[0, 0, 1], 1e-3 * dx * dx / dt * (1 - 2e-7), rtol=1e-6)   # down to its content
    np.testing.assert_allclose(Qx2.sum(0), Qx.sum(0), rtol=1e-12)
    assert -500.0 < Qx2[1, 0, 1] < 0.0


def test_limit_outflow_handles_open_edge_fluxes():
    # outflow through an open west edge face (index 0) counts as outflow of cell 0
    dx, dt = 1000.0, 10.0
    h = np.array([[[1e-3, 10.0]], [[10.0, 10.0]]])
    area = np.full((1, 2), dx * dx)
    Qx = np.zeros((2, 1, 3))
    Qx[0, 0, 0] = -3e-3 * dx * dx / dt           # leaves cell 0 through the west edge (300 m³/s, holds 100)
    Qx[1, 0, 0] = 500.0                          # inflow of layer 1 through the edge absorbs the 200 m³/s cut
    Qx2, _, n = ale.limit_outflow(h, Qx, np.zeros((2, 2, 2)), area, dt, np.ones((1, 2), bool))
    h_new = h - dt * (Qx2[..., 1:] - Qx2[..., :-1]) / area[None]
    assert n >= 1 and h_new.min() >= 0.0
    np.testing.assert_allclose(Qx2.sum(0), Qx.sum(0), rtol=1e-12)


def test_limit_outflow_raises_when_the_column_cannot_supply_the_flux():
    dx, dt = 1000.0, 10.0
    h = np.array([[[1e-3, 10.0]], [[1e-3, 10.0]]])
    area = np.full((1, 2), dx * dx)
    Qx = np.zeros((2, 1, 3))
    Qx[:, 0, 1] = 150.0                          # 3000 m³ in 10 s eastwards from a column holding 2000 m³
    with pytest.raises(RuntimeError, match="barotropic CFL"):
        ale.limit_outflow(h, Qx, np.zeros((2, 2, 2)), area, dt, np.ones((1, 2), bool))
```

- [ ] **Step 2: Run them**

Run: `python -m pytest tests/test_vertical_ale.py -q -k limit_outflow`

Expected: 5 FAIL with `AttributeError: module 'core.vertical.ale' has no attribute 'limit_outflow'`.

- [ ] **Step 3: Implement `limit_outflow`** (insert in `core/vertical/ale.py` after `lagrangian_thickness`)

```python
def limit_outflow(h: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, area: np.ndarray, dt: float, active: np.ndarray,
                  h_floor: float = 2e-10, max_pass: int = 10) -> tuple[np.ndarray, np.ndarray, int]:
    """
    Outflow limiter of the Lagrangian step (spec 2026-10-14-hybrid-robustness §2): no layer of a cell may lose
    more than it holds in Δt,

        Δt O_k ≤ (h_k − h_floor)⁺ A ,   O_k = Σ_faces outgoing |Q_k|   (x and y together, unsplit),

    so h† = h − Δt ∇·Q_k / A ≥ 0 exactly.  A cell-layer that would exceed it has its outgoing fluxes scaled by
    r = (h_k − h_floor)⁺ A / (Δt O_k); the flux cut at each face goes back to the other layers of that face so
    Σ_k Q_k (the free-surface flux) is unchanged: first to layers flowing the same way, in proportion to the room
    left in their donor cell, then by reducing layers flowing the other way (in proportion to |Q|, never past 0),
    which only lowers the outflow of the neighbour.  Repeated until no cell exceeds (≤ max_pass).

    Why: the PPM swept-volume flux of a cell never exceeds its content for Courant ≤ ½ per face, but the uniform
    barotropic correction and the residual share can add the last few per cent (IW: 99.4 % + 1.2 %; overflow:
    90.6 % + 10 %, 2026-10-14).  MOM6 bounds that correction by the CFL and accepts a barotropic/baroclinic
    mismatch instead (MOM_continuity_PPM.F90:783-874, 1542-1544); here Σh = H + η stays exact.  HYCOM cnuity.F90
    has no such limit (FCT only) and stops below −10 m.  Velocities are not changed (as MOM6: only the fluxes).

    h (N, ny, nx) [m]; Qx (N, ny, nx+1), Qy (N, ny+1, nx) [m³ s⁻¹], edge faces included (open boundaries);
    area (ny, nx) [m²]; active (ny, nx).  Returns (Qx, Qy, n_limited); the inputs themselves when nothing exceeds.
    """
    A = np.where(active, area, 0.0)[None]
    room0 = np.maximum(h - h_floor, 0.0) * A / dt                      # what each cell-layer can give [m³ s⁻¹]

    def outflow(qx, qy):
        return (np.maximum(qx[..., 1:], 0.0) - np.minimum(qx[..., :-1], 0.0)
                + np.maximum(qy[:, 1:, :], 0.0) - np.minimum(qy[:, :-1, :], 0.0))

    O = outflow(Qx, Qy)
    over = active[None] & (O > room0)
    if not over.any():
        return Qx, Qy, 0
    Qx, Qy, n_lim = Qx.copy(), Qy.copy(), 0
    for _ in range(max_pass):
        n_lim += int(over.sum())
        r = np.where(over, room0 / np.where(O > 0, O, 1.0), 1.0)
        for Q, ax in ((Qx, 2), (Qy, 1)):
            one = np.ones_like(np.take(r, [0], axis=ax))
            r_lo = np.concatenate([one, r], axis=ax)                   # cell on the low side of each face
            r_hi = np.concatenate([r, one], axis=ax)                   # cell on the high side
            new = np.where(Q > 0, Q * r_lo, np.where(Q < 0, Q * r_hi, Q))
            delta = (Q - new).sum(axis=0)                              # net cut per face (sign = flow removed)
            Q[...] = new
            _give_back(Q, delta, room0 - outflow(Qx, Qy), ax)
        O = outflow(Qx, Qy)
        over = active[None] & (O > room0 * (1.0 + 1e-12))
        if not over.any():
            return Qx, Qy, n_lim
    k, j, i = (int(a[0]) for a in np.nonzero(over))
    raise RuntimeError(f"limit_outflow: layer {k} of cell (j={j}, i={i}) still loses more than it holds after "
                       f"{max_pass} passes (Δt = {dt:g} s): the column cannot supply the free-surface flux — "
                       f"barotropic CFL")


def _give_back(Q: np.ndarray, delta: np.ndarray, room: np.ndarray, ax: int) -> None:
    """
    Put the net cut ``delta`` (one value per face, the sign of the removed flow) back on the layers of each face,
    in place: first to layers already flowing that way (donor room shared in proportion), then by reducing layers
    flowing against it (in proportion to |Q|, never past 0).  Raises if neither can absorb it (barotropic CFL).
    ``room`` (N, ny, nx): what each donor cell-layer can still give [m³ s⁻¹]; ``ax`` the face axis of Q (2: x, 1: y).
    """
    zero = np.zeros_like(np.take(room, [0], axis=ax))
    room_lo = np.maximum(np.concatenate([zero, room], axis=ax), 0.0)   # donor of eastward/northward flow
    room_hi = np.maximum(np.concatenate([room, zero], axis=ax), 0.0)   # donor of westward/southward flow
    s = np.sign(delta)[None]
    same = (np.sign(Q) == s) & (Q != 0.0) & (s != 0.0)
    rm = np.where(same, np.where(s > 0, room_lo, room_hi), 0.0)
    tot = rm.sum(axis=0)
    need = np.abs(delta)
    give = np.minimum(need, tot)
    Q += s * rm * np.where(tot > 0, give / np.where(tot > 0, tot, 1.0), 0.0)[None]
    rest = need - give
    if not (rest > 0).any():
        return
    opp = (np.sign(Q) == -s) & (s != 0.0)
    cap = np.where(opp, np.abs(Q), 0.0).sum(axis=0)
    if (rest > cap * (1.0 + 1e-12)).any():
        idx = tuple(int(a[0]) for a in np.nonzero(rest > cap * (1.0 + 1e-12)))
        raise RuntimeError(f"limit_outflow: face {idx} cannot carry its free-surface flux with the water of its "
                           f"donor column — barotropic CFL (reduce Δt)")
    frac = np.where(cap > 0, rest / np.where(cap > 0, cap, 1.0), 0.0)[None]
    Q -= np.where(opp, Q * frac, 0.0)
```

- [ ] **Step 4: Run the tests**

Run: `python -m pytest tests/test_vertical_ale.py -q -k limit_outflow`

Expected: 5 passed. If `test_limit_outflow_reduces_counter_flowing_layers...` fails only on its first `assert_allclose` value, check the expected number against `room0` (it includes h_floor = 2e-10 m). Correct the test's expected value, not the code, and ledger it.

- [ ] **Step 5: Wire it into the ALE step**

In `core/dynamics/sigma_baroclinic.py`:
- add `limit_outflow` to the `from core.vertical.ale import (...)` list (line 95);
- add the field to `SigmaState` after `n_trunc`:

```python
    n_limited: int = 0                                       # cell-layers scaled by the outflow limiter so far (ALE)
```

In `_step_ale`, replace

```python
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        h_lag = lagrangian_thickness(st.h, divh, dt, self.area)
```

with

```python
        # outflow limiter: no layer loses more than it holds; Σ_k Q_k unchanged (core.vertical.ale.limit_outflow)
        Qx, Qy, n_lim = limit_outflow(st.h, Qx, Qy, self.area, dt, act)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        h_lag = lagrangian_thickness(st.h, divh, dt, self.area)
```

Then in the `return SigmaState(...)` of `_step_ale`, add `n_limited=st.n_limited + n_lim`.

In `core/restart.py`:
- add `"n_limited": int(state.n_limited)` to `attrs` (line ~122);
- add `n_limited=int(ds.attrs.get("n_limited", 0))` to the `SigmaState(...)` of `read_restart` (line ~179).

- [ ] **Step 6: Restart test** (append to `tests/test_restart.py`)

```python
def test_ale_restart_keeps_the_outflow_limiter_count(tmp_path):
    m = _ale_model("zstar")
    st = m.step(_initial(m))
    st.n_limited = 7
    write_restart(tmp_path / "rst.nc", m, st)
    assert read_restart(tmp_path / "rst.nc", _ale_model("zstar")).n_limited == 7
```

Run: `python -m pytest tests/test_restart.py -q`

Expected: all pass, including the new test.

- [ ] **Step 7: Bit-for-bit guard and cost**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_vertical_ale.py tests/test_restart.py -q`

Expected: all pass. In particular, the fingerprint tests `test_ale_results_unchanged_by_the_performance_kernels` pass, because nothing exceeds in those cases and the limiter returns the same arrays.

Then time it on the BayIW cut:
```bash
NUMBA_NUM_THREADS=5 python validation/bayiw_cut/profile_step.py
```
Read `profile_step.py` first for its arguments. Compare `limit_outflow`'s share with the total. If it is > 2 %, ledger it and stop to discuss a kernel.

- [ ] **Step 8: Probe the IW failure**

Rerun the Task 0 probe's IW case (`probe.py iw 0.01 0.5 2880`). Expected: no "layer emptied" stop. Either it completes 200 days, or it fails later with a different message (truncations), which Task 2 addresses. Ledger the outcome.

- [ ] **Step 9: Commit**

```bash
git add core/vertical/ale.py core/dynamics/sigma_baroclinic.py core/restart.py tests/test_vertical_ale.py tests/test_restart.py
git commit -m "ALE: outflow limiter keeps every layer thickness non-negative

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 2: Viscous coupling of near-vanished layers (option `thin_layer_coupling`)

**Files:**
- Modify: `core/vertical/ale.py` (new `viscous_face_thickness`)
- Modify: `core/dynamics/sigma_wd.py` (`column_step(..., dz_drag=None)` and `step(..., hu_drag=None, hv_drag=None)`)
- Modify: `core/dynamics/sigma_baroclinic.py` (constructor option; `_step_ale` call to `self.core.step`)
- Modify: `core/restart.py` (`_config`: `thin_layer_coupling`)
- Test: `tests/test_vertical_ale.py`, `tests/test_sigma_wd.py`, `tests/test_sigma_ale.py`, `tests/test_restart.py`

**Interfaces:**
- Consumes: Task 1 (nothing specific).
- Produces:
  - `ale.viscous_face_thickness(q_a: np.ndarray, q_b: np.ndarray, h_neglect: float = 1e-10) -> np.ndarray`;
  - `SigmaBaroclinic(..., thin_layer_coupling: str = "none")` with values `"none"` | `"mom6"` (the default changes in Task 4);
  - `SigmaWetDry.step(..., hu_drag=None, hv_drag=None)`.

- [ ] **Step 1: Read MOM6's coupling thickness**

Read `~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_vert_friction.F90` lines 1457–1532 and around 2236 (`find_coupling_coef`). Determine the default face thickness that enters the coupling a = Kv/(h_shear + …):
- the harmonic mean `2 h_L h_R/(h_L + h_R)`;
- the arithmetic mean;
- or a combination.

Write it in the ledger with file:line. If MOM6's default is not the harmonic mean, implement MOM6's formula in Step 3 instead of the harmonic mean, keep the name `viscous_face_thickness`, and ledger the ruling.

- [ ] **Step 2: Failing tests**

Append to `tests/test_vertical_ale.py`:

```python
def test_viscous_face_thickness_vanishes_with_either_side():
    qa = np.array([50.0, 1e-3, 0.0, 10.0])
    qb = np.array([50.0, 50.0, 50.0, 10.0])
    hv = ale.viscous_face_thickness(qa, qb)
    np.testing.assert_allclose(hv, [50.0, 2e-3, 0.0, 10.0], rtol=1e-6, atol=1e-9)
```

Append to `tests/test_sigma_wd.py`:

```python
def test_viscous_coupling_keeps_the_bed_drag_thickness():
    # the log-law bed drag C_d(dz_0) must use dz_drag when given, not the (thin) viscous thickness
    from core.dynamics.sigma_wd import SigmaWetDry
    from core.grid.horizontal import create_cartesian_grid
    hg = create_cartesian_grid(nx=3, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = SigmaWetDry(hg, np.full((1, 3), 20.0), dt=10.0, n_layers=3, z0=1e-3, closure="constant", nu_const=1e-4)
    u = np.ones((3, 1, 4))
    dz_visc = np.full((3, 1, 4), 6.0); dz_visc[0] = 1e-3          # thin viscous bed thickness
    dz_mean = np.full((3, 1, 4), 6.0)
    a = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_visc, dz_drag=dz_mean[0])
    b = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_visc)
    assert not np.allclose(a, b)                                   # dz_drag is used
    c = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_mean, dz_drag=dz_mean[0])
    d = m.column_step(u, None, np.full((1, 4), 18.0), dz=dz_mean)
    np.testing.assert_array_equal(c, d)                            # same as before when dz_drag = dz[0]
```

Append to `tests/test_sigma_ale.py`:

```python
def test_thin_layer_coupling_ties_a_vanished_layer_to_its_neighbours():
    # one column of the implicit viscosity with the coupling thickness of 'mom6': a 1 mm layer between two 10 m
    # layers takes their velocity in one step; momentum Σ dz u is conserved
    from core.dynamics.sigma_wd import vertical_implicit
    from core.vertical import ale
    hu_mean = np.array([[10.0], [25.0], [10.0]])                    # thin layer looks 25 m thick on the mean
    q_a = np.array([[10.0], [1e-3], [10.0]]); q_b = np.array([[10.0], [50.0], [10.0]])
    dz = np.maximum(ale.viscous_face_thickness(q_a, q_b), ale.DZ_FLOOR)
    u0 = np.array([[1.0], [5.0], [0.0]])
    nu = np.full((4, 1), 1e-4)
    u1 = vertical_implicit(u0, dz, nu, np.zeros(1), 10.0)
    assert abs(u1[1, 0] - 0.5 * (u1[0, 0] + u1[2, 0])) < 0.01 * abs(u0[1, 0] - 0.5)
    np.testing.assert_allclose((dz * u1).sum(), (dz * u0).sum(), rtol=1e-12)
    u_old = vertical_implicit(u0, hu_mean, nu, np.zeros(1), 10.0)   # the mean thickness barely couples it
    assert abs(u_old[1, 0] - u0[1, 0]) < 0.01


def test_thin_layer_coupling_option_is_validated():
    with pytest.raises(ValueError, match="thin_layer_coupling"):
        _ale_box("zstar", thin_layer_coupling="harmonic")
```

Append to `tests/test_restart.py`:

```python
def test_ale_restart_refuses_another_thin_layer_coupling(tmp_path):
    m = _ale_model("zstar")
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    other = _ale_model("zstar")
    other.thin_layer_coupling = "mom6" if m.thin_layer_coupling == "none" else "none"
    with pytest.raises(ValueError, match="thin_layer_coupling"):
        read_restart(tmp_path / "rst.nc", other)
```


Run: `python -m pytest tests/test_vertical_ale.py tests/test_sigma_wd.py tests/test_sigma_ale.py tests/test_restart.py -q -k "viscous or thin_layer_coupling"`

Expected: FAIL. `viscous_face_thickness` does not exist, and `column_step` has no `dz_drag`.

- [ ] **Step 3: Implement**

In `core/vertical/ale.py` (after `face_thickness`):

```python
def viscous_face_thickness(q_a: np.ndarray, q_b: np.ndarray, h_neglect: float = 1e-10) -> np.ndarray:
    """
    Layer thickness at a velocity face for the vertical-viscosity coupling only (spec 2026-10-14 §3): the harmonic
    mean 2 q_a q_b / (q_a + q_b + h_neglect) of the two sides' thicknesses [m] (MOM6 MOM_vert_friction.F90, see the
    ledger of plan 2026-10-14-hybrid-robustness Task 2 for the exact lines).  A layer that has vanished on one side
    gets a thickness → 0, so ν Δt / h² is large and the implicit viscosity ties its velocity to its neighbours;
    the mean (q_a + q_b)/2 would give it half the thick side and let it accelerate alone (overflow ν = 100:
    2 → 24 m/s on faces with a side < 1 cm, 2026-10-14).
    """
    return 2.0 * q_a * q_b / (q_a + q_b + h_neglect)
```

In `core/dynamics/sigma_wd.py`, `column_step`: add the parameter `dz_drag: np.ndarray | None = None` and document it ("bed-layer thickness for the log-law drag C_d(dz_drag) [m] (default dz[0])"). Replace `cd = first_layer_cd(dz[0], self.z0, cd_max=self.cd_max)` with

```python
            d0 = dz[0] if dz_drag is None else np.asarray(dz_drag, float).reshape(-1)
            cd = first_layer_cd(d0, self.z0, cd_max=self.cd_max)
```

In `step`: add `hu_drag: np.ndarray | None = None, hv_drag: np.ndarray | None = None` to the signature and the docstring, and pass `dz_drag=hu_drag` / `dz_drag=hv_drag` to the two `column_step` calls.

In `core/dynamics/sigma_baroclinic.py`:
- add the constructor parameter `thin_layer_coupling: str = "none"` after `theta_bc`;
- document it in the class docstring next to `theta_bc`;
- validate and store it right after `self.theta_bc = ...`:

```python
        if thin_layer_coupling not in ("none", "mom6"):
            raise ValueError(f"unknown thin_layer_coupling {thin_layer_coupling!r}; use 'none' or 'mom6'")
        self.thin_layer_coupling = thin_layer_coupling
```

In `_step_ale`, replace the `self.core.step(...)` call's `hu=np.maximum(hu, DZ_FLOOR), hv=np.maximum(hv, DZ_FLOOR)` with:

```python
        if self.thin_layer_coupling == "mom6":                         # viscous tie of vanished layers (spec §3)
            hu_v = viscous_face_thickness(qe, qw)
            hv_v = viscous_face_thickness(qn, qs).swapaxes(1, 2)
            visc = dict(hu=np.maximum(hu_v, DZ_FLOOR), hv=np.maximum(hv_v, DZ_FLOOR),
                        hu_drag=np.maximum(hu[0], DZ_FLOOR), hv_drag=np.maximum(hv[0], DZ_FLOOR))
        else:
            visc = dict(hu=np.maximum(hu, DZ_FLOOR), hv=np.maximum(hv, DZ_FLOOR))
        eta_new, u_new, v_new = self.core.step(st.eta, st.u, st.v, G_u=gu_ab, G_v=gv_ab, tau_u=tu, tau_v=tv,
                                               wu=wu, wv=wv, **visc, **self._flather_args())
```

Also add `viscous_face_thickness` to the import list. In `core/restart.py` `_config`, inside the ALE `cfg.update(...)`, add `thin_layer_coupling=model.thin_layer_coupling`.

- [ ] **Step 4: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_vertical_ale.py tests/test_sigma_wd.py tests/test_sigma_ale.py tests/test_restart.py -q`

Expected: all pass. The default is "none", so the fingerprints are unchanged.

- [ ] **Step 5: Check it against the Task 0 probe**

Rerun `probe.py overflow 100 0.6 60` and `probe.py overflow 1000 0.6 360`, with `m.thin_layer_coupling = "mom6"` set right after `build`.

Expected:
- With ν = 1000: max|u| on thin faces stays below max|u| on thick faces for the whole run (criterion 2).
- With ν = 100: ledger how far it gets and the thin-face max|u| at 5.5 h (it was 24 m/s). This is the "meta medida", not a gate.

If ν = 1000 still has thin faces faster than thick ones, stop: decision point 2 (HYCOM blending 1B) with Esteban.

- [ ] **Step 6: Commit**

```bash
git add core/vertical/ale.py core/dynamics/sigma_wd.py core/dynamics/sigma_baroclinic.py core/restart.py tests/test_vertical_ale.py tests/test_sigma_wd.py tests/test_sigma_ale.py tests/test_restart.py
git commit -m "ALE: optional viscous coupling of near-vanished layers (MOM6 harmonic face thickness)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 3: Bounded tracers: Zalesak FCT on the SSP-RK3 WENO5 advection (option `tracer_limiter`)

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py`:
  - new kernels `_advect_faces_h` (extracted from `_advect_div`) and `_fct_coeff_kernel`;
  - module-level `fct_coefficients_numpy`;
  - methods `_faces_h`, `_upwind_faces_h`, `_advect_fct`;
  - constructor option; the dispatch in `_step_ale`.
- Modify: `core/restart.py` (`_config`: `tracer_limiter`)
- Test: `tests/test_sigma_ale.py`, `tests/test_restart.py`

**Interfaces:**
- Consumes: Task 1's guarantee. With `limit_outflow`, the donor-cell update is a convex combination of the old value and the inflowing values.
- Produces:
  - `SigmaBaroclinic(..., tracer_limiter: str = "none")` with values `"none"` | `"fct"`. "fct" requires `vertical_coordinate="ale"`, `tracer_time="ssprk3"` and `tracer_scheme` in ("weno5", "weno5js").
  - `fct_coefficients_numpy(c0, cL, V1, Ax, Ay, ok, dt) -> (Cx, Cy)`.

- [ ] **Step 1: Extract the horizontal face kernel without changing any bit**

In `core/dynamics/sigma_baroclinic.py`, create `@njit(cache=True, parallel=True) def _advect_faces_h(c, Qx, Qy, ok, ext, open_w, open_e, open_s, open_n, z_variant)` returning `(fx, fy)`, with shapes `(nt, n, ny, nx+1)` and `(nt, n, ny+1, nx)`:
- its body is the `fx`/`fy` allocation plus the whole `for p in prange(nt * n):` loop of `_advect_div` (lines ~243–295), unchanged;
- add a docstring: "Horizontal face fluxes Q·c_face of `_advect_div` (WENO5 upwind faces, upstream values on open edges), returned instead of differenced — the FCT high-order flux (spec 2026-10-14 §4)."

In `_advect_div`, replace those lines with:

```python
    fx, fy = _advect_faces_h(c, Qx, Qy, ok, ext, open_w, open_e, open_s, open_n, z_variant)
```

The `fz` allocation, the `if vert:` loop and the final difference loop stay as they are.

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_sigma_baroclinic.py -q -k "unchanged or fingerprint or kernel"`

Expected: pass, because the fingerprints are bit-level. If any fails, the extraction changed arithmetic: undo and redo it.

- [ ] **Step 2: Failing tests** (append to `tests/test_sigma_ale.py`)

```python
# ---------------------------------------------------------------- FCT (spec 2026-10-14-hybrid-robustness §4)
def _step_box(nx=60, limiter="fct", T=None, u0=0.6, nz=4, **kw):
    """z* box with a uniform eastward layer velocity and a T step: one advection call of the ALE step."""
    m = _ale_box("zstar", nx=nx, nz=nz, dx=1000.0, dt=30.0, tracer_limiter=limiter, mix_diagnostic=False, **kw)
    st = m.initial_state(np.zeros((1, nx)), np.where(np.arange(nx) < nx // 2, 1.0, 0.0)[None, None, :].repeat(nz, 0)
                         if T is None else T)
    return m, st


def test_fct_keeps_a_step_inside_its_range_where_weno_overshoots():
    nx, dt, dx = 60, 30.0, 1000.0
    for limiter, bounded in (("fct", True), ("none", False)):
        m, st = _step_box(nx, limiter)
        V = m.layer_volumes(st.eta, st.h)
        Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = 0.9 * dx / dt * st.h[:, :, 1:] * dx    # Courant 0.9 eastwards
        Qx[:, :, -2:] = 0.0
        Qy = np.zeros(st.v.shape)
        div = Qx[..., 1:] - Qx[..., :-1]
        V1 = V - dt * div
        ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
        T = st.T[None]
        out = m._advect_fct(T, V, V1, Qx, Qy, ok) if limiter == "fct" else m._advect(T, V, V1, Qx, Qy,
                                                                                    np.zeros((m.n + 1, 1, nx)), ok)
        inside = (out.min() >= -1e-12) and (out.max() <= 1 + 1e-12)
        assert inside == bounded, (limiter, out.min(), out.max())


def test_fct_with_unit_coefficients_is_the_rk3_result():
    m, st = _step_box(40, "fct")
    V = m.layer_volumes(st.eta, st.h)
    rng = np.random.default_rng(0)
    Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = rng.uniform(-0.3, 0.3, Qx[:, :, 1:-1].shape) * 1000.0 * st.h[:, :, 1:]
    Qy = np.zeros(st.v.shape)
    V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
    ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
    T = (np.sin(np.arange(40) / 5.0)[None, None, None, :] + 2.0) * np.ones(st.T.shape)[None]
    ref = m._advect(T, V, V1, Qx, Qy, np.zeros((m.n + 1, 1, 40)), ok)
    got = m._advect_fct(T, V, V1, Qx, Qy, ok, unlimited=True)
    np.testing.assert_allclose(got, ref, rtol=0, atol=1e-12)


def test_fct_conserves_and_keeps_high_order_on_a_smooth_profile():
    nx = 80
    errs = {}
    for limiter in ("fct", "none"):
        m, st = _step_box(nx, limiter)
        x = np.arange(nx)
        T0 = (1.0 + 0.5 * np.sin(2 * np.pi * x / nx))[None, None, :].repeat(m.n, 0)
        st = m.initial_state(np.zeros((1, nx)), T0)
        V = m.layer_volumes(st.eta, st.h)
        Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = 0.5 * 1000.0 / m.dt * st.h[:, :, 1:] * 1000.0
        Qx[:, :, -1] = 0.0; Qx[:, :, 1] = 0.0
        V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
        ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
        out = (m._advect_fct(T0[None], V, V1, Qx, np.zeros(st.v.shape), ok) if limiter == "fct" else
               m._advect(T0[None], V, V1, Qx, np.zeros(st.v.shape), np.zeros((m.n + 1, 1, nx)), ok))[0]
        np.testing.assert_allclose((V1 * out).sum(), (V * T0).sum(), rtol=1e-12)
        exact = 1.0 + 0.5 * np.sin(2 * np.pi * (x - 0.5) / nx)                            # shifted by 0.5 cell
        errs[limiter] = np.abs(out[0, 0, 5:-5] - exact[5:-5]).max()
    assert errs["fct"] <= 1.5 * errs["none"], errs


def test_fct_bounds_a_thin_layer_receiving_other_water():
    m, st = _step_box(20, "fct")
    h = st.h.copy(); h[1, 0, 10] = 1e-3                              # 1 mm cell in layer 1
    T = np.ones_like(st.T); T[1, 0, 10] = 0.0; T[1, 0, :10] = 5.0
    V = m.layer_volumes(st.eta, h)
    Qx = np.zeros(st.u.shape); Qx[1, 0, 10] = 2.0; Qx[1, 0, 11] = 1.0                 # inflow 2, outflow 1 m³/s
    V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
    ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
    out = m._advect_fct(T[None], V, V1, Qx, np.zeros(st.v.shape), ok)[0]
    assert 0.0 - 1e-12 <= out[1, 0, 10] <= 5.0 + 1e-12


def test_fct_limits_each_tracer_of_the_stack():
    m, st = _step_box(30, "fct")
    V = m.layer_volumes(st.eta, st.h)
    Qx = np.zeros(st.u.shape); Qx[:, :, 1:-1] = 0.9 * 1000.0 / m.dt * st.h[:, :, 1:] * 1000.0; Qx[:, :, -2:] = 0.0
    V1 = V - m.dt * (Qx[..., 1:] - Qx[..., :-1])
    ok = np.broadcast_to(m.fs.active[None], V.shape) & (V1 > 0)
    C = np.stack([st.T, st.T ** 2 * 3.0])
    out = m._advect_fct(C, V, V1, Qx, np.zeros(st.v.shape), ok)
    for t, (lo, hi) in enumerate(((0.0, 1.0), (0.0, 3.0))):
        assert out[t].min() >= lo - 1e-12 and out[t].max() <= hi + 1e-12


def test_fct_open_edge_faces_are_upstream():
    m, st = _ale_open_box("zstar", tracer_limiter="fct")
    C = st.T[None]
    Qx = np.zeros(st.u.shape); Qx[:, :, 0] = 3.0; Qx[:, :, -1] = -2.0
    FH = m._faces_h(C, Qx, np.zeros(st.v.shape), np.ones(C.shape[1:], bool))[0]
    FL = m._upwind_faces_h(C, Qx, np.zeros(st.v.shape))[0]
    np.testing.assert_array_equal(FH[..., [0, -1]], FL[..., [0, -1]])


def test_fct_numba_coefficients_match_numpy():
    from core.dynamics.sigma_baroclinic import _fct_coeff_kernel, fct_coefficients_numpy
    rng = np.random.default_rng(3)
    nt, n, ny, nx = 2, 3, 4, 5
    c0 = rng.uniform(0, 1, (nt, n, ny, nx)); cL = rng.uniform(0, 1, (nt, n, ny, nx))
    V1 = rng.uniform(0.5, 2, (n, ny, nx)) * 1e6
    Ax = rng.normal(0, 1e4, (nt, n, ny, nx + 1)); Ax[..., [0, -1]] = 0
    Ay = rng.normal(0, 1e4, (nt, n, ny + 1, nx)); Ay[..., [0, -1], :] = 0
    ok = rng.uniform(0, 1, (n, ny, nx)) > 0.1
    ref = fct_coefficients_numpy(c0, cL, V1, Ax, Ay, ok, 30.0)
    got = _fct_coeff_kernel(c0, cL, V1, Ax, Ay, ok, 30.0)
    np.testing.assert_allclose(got[0], ref[0], rtol=0, atol=1e-14)
    np.testing.assert_allclose(got[1], ref[1], rtol=0, atol=1e-14)


def test_tracer_limiter_option_is_validated():
    with pytest.raises(ValueError, match="tracer_limiter"):
        _ale_box("zstar", tracer_limiter="clip")
    with pytest.raises(ValueError, match="tracer_limiter"):
        _ale_box("zstar", tracer_limiter="fct", tracer_time="pc")
```

Append to `tests/test_restart.py`:

```python
def test_ale_restart_refuses_another_tracer_limiter(tmp_path):
    m = _ale_model("zstar")
    write_restart(tmp_path / "rst.nc", m, m.step(_initial(m)))
    other = _ale_model("zstar")
    other.tracer_limiter = "fct" if m.tracer_limiter == "none" else "none"
    with pytest.raises(ValueError, match="tracer_limiter"):
        read_restart(tmp_path / "rst.nc", other)
```


Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_restart.py -q -k "fct or tracer_limiter"`

Expected: FAIL (`tracer_limiter` is an unknown keyword; `_advect_fct` does not exist).

- [ ] **Step 3: Implement the coefficients** (module level in `sigma_baroclinic.py`, after `_advect_div`)

```python
def fct_coefficients_numpy(c0: np.ndarray, cL: np.ndarray, V1: np.ndarray, Ax: np.ndarray, Ay: np.ndarray,
                           ok: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """
    Zalesak (1979, eqs. 7–13) flux-limiting coefficients C ∈ [0, 1] of the antidiffusive fluxes A = F_H − F_L
    (stack (nt, N, ny, nx); Ax (nt, N, ny, nx+1), Ay (nt, N, ny+1, nx) [tracer m³ s⁻¹]) so that
    V1 c = V1 c_L − Δt ∇·(C A) stays inside [c_min, c_max] = the extremes of c0 and c_L over the cell and its x, y
    neighbours of the same layer (eq. 15, ``ok`` cells only).  Cells not ``ok`` (dry, V1 = 0) get R± = 0, so the
    faces of a vanishing cell are upwind.  NumPy oracle of _fct_coeff_kernel.
    """
    big = np.inf
    okb = np.broadcast_to(ok, c0.shape)
    hi = np.where(okb, np.maximum(c0, cL), -big)
    lo = np.where(okb, np.minimum(c0, cL), big)

    def neigh(a, f):
        out = a.copy()
        out[..., 1:] = f(out[..., 1:], a[..., :-1]); out[..., :-1] = f(out[..., :-1], a[..., 1:])
        out[..., 1:, :] = f(out[..., 1:, :], a[..., :-1, :]); out[..., :-1, :] = f(out[..., :-1, :], a[..., 1:, :])
        return out

    cmax, cmin = neigh(hi, np.maximum), neigh(lo, np.minimum)
    p_in = dt * (np.maximum(Ax[..., :-1], 0) - np.minimum(Ax[..., 1:], 0)
                 + np.maximum(Ay[..., :-1, :], 0) - np.minimum(Ay[..., 1:, :], 0))
    p_out = dt * (np.maximum(Ax[..., 1:], 0) - np.minimum(Ax[..., :-1], 0)
                  + np.maximum(Ay[..., 1:, :], 0) - np.minimum(Ay[..., :-1, :], 0))
    q_in = np.maximum(V1[None] * (cmax - cL), 0.0)
    q_out = np.maximum(V1[None] * (cL - cmin), 0.0)
    r_in = np.where(okb & (p_in > 0), np.minimum(1.0, q_in / np.where(p_in > 0, p_in, 1.0)), 0.0)
    r_out = np.where(okb & (p_out > 0), np.minimum(1.0, q_out / np.where(p_out > 0, p_out, 1.0)), 0.0)
    Cx = np.zeros_like(Ax); Cy = np.zeros_like(Ay)
    Cx[..., 1:-1] = np.where(Ax[..., 1:-1] >= 0, np.minimum(r_in[..., 1:], r_out[..., :-1]),
                             np.minimum(r_in[..., :-1], r_out[..., 1:]))
    Cy[..., 1:-1, :] = np.where(Ay[..., 1:-1, :] >= 0, np.minimum(r_in[..., 1:, :], r_out[..., :-1, :]),
                                np.minimum(r_in[..., :-1, :], r_out[..., 1:, :]))
    return Cx, Cy


@njit(cache=True, parallel=True)
def _fct_coeff_kernel(c0, cL, V1, Ax, Ay, ok, dt):  # pragma: no cover
    """fct_coefficients_numpy with the same arithmetic, parallel over (tracer, layer)."""
    nt, n, ny, nx = c0.shape
    r_in = np.zeros((nt, n, ny, nx))
    r_out = np.zeros((nt, n, ny, nx))
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for i in range(nx):
                if not ok[k, j, i]:
                    continue
                cmax, cmin = -np.inf, np.inf
                for jj, ii in ((j, i), (j, i - 1), (j, i + 1), (j - 1, i), (j + 1, i)):
                    if 0 <= jj < ny and 0 <= ii < nx and ok[k, jj, ii]:
                        cmax = max(cmax, c0[t, k, jj, ii], cL[t, k, jj, ii])
                        cmin = min(cmin, c0[t, k, jj, ii], cL[t, k, jj, ii])
                p_in = dt * (max(Ax[t, k, j, i], 0.0) - min(Ax[t, k, j, i + 1], 0.0)
                             + max(Ay[t, k, j, i], 0.0) - min(Ay[t, k, j + 1, i], 0.0))
                p_out = dt * (max(Ax[t, k, j, i + 1], 0.0) - min(Ax[t, k, j, i], 0.0)
                              + max(Ay[t, k, j + 1, i], 0.0) - min(Ay[t, k, j, i], 0.0))
                q_in = max(V1[k, j, i] * (cmax - cL[t, k, j, i]), 0.0)
                q_out = max(V1[k, j, i] * (cL[t, k, j, i] - cmin), 0.0)
                r_in[t, k, j, i] = min(1.0, q_in / p_in) if p_in > 0.0 else 0.0
                r_out[t, k, j, i] = min(1.0, q_out / p_out) if p_out > 0.0 else 0.0
    Cx = np.zeros(Ax.shape)
    Cy = np.zeros(Ay.shape)
    for p in prange(nt * n):
        t, k = p // n, p % n
        for j in range(ny):
            for f in range(1, nx):
                if Ax[t, k, j, f] >= 0.0:
                    Cx[t, k, j, f] = min(r_in[t, k, j, f], r_out[t, k, j, f - 1])
                else:
                    Cx[t, k, j, f] = min(r_in[t, k, j, f - 1], r_out[t, k, j, f])
        for f in range(1, ny):
            for i in range(nx):
                if Ay[t, k, f, i] >= 0.0:
                    Cy[t, k, f, i] = min(r_in[t, k, f, i], r_out[t, k, f - 1, i])
                else:
                    Cy[t, k, f, i] = min(r_in[t, k, f - 1, i], r_out[t, k, f, i])
    return Cx, Cy
```

Note on `cmax`/`cmin` near land: the NumPy oracle's `neigh` uses ±inf for cells that are not ok, the same as the kernel's `ok` test. The two agree, and the Step 2 test checks it.

- [ ] **Step 4: Implement the methods and the option**

Constructor:
- add `tracer_limiter: str = "none"` after `thin_layer_coupling`;
- document it in the class docstring ("'fct': Zalesak 1979 flux-corrected transport around the SSP-RK3 WENO5 advection of the ALE step — tracers stay inside the local extremes of the old and the donor-cell values; spec 2026-10-14 §4");
- validate after `_setup_ale` (where `self.ale` is known):

```python
        if tracer_limiter not in ("none", "fct"):
            raise ValueError(f"unknown tracer_limiter {tracer_limiter!r}; use 'none' or 'fct'")
        if tracer_limiter == "fct" and not (self.ale and self.tracer_time == "ssprk3"
                                            and self.tracer_scheme in ("weno5", "weno5js")):
            raise ValueError("tracer_limiter='fct' needs vertical_coordinate='ale', tracer_time='ssprk3' and a "
                             "WENO5 tracer_scheme")
        self.tracer_limiter = tracer_limiter
```

Methods (next to `_advect`):

```python
    def _ext_stack(self, c: np.ndarray) -> np.ndarray:
        """Inflow values of the stack on open edges (T_in, T_in² for the mixing diagnostic), as _rhs."""
        T_in = self._inflow_T()
        return c if T_in is None or not self.open else np.ascontiguousarray(np.stack([T_in, T_in**2])[: c.shape[0]])

    def _faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, ok: np.ndarray):
        """WENO5 horizontal face fluxes (fx, fy) of the stack — the terms _advect_div differences (no ω in ALE)."""
        c = np.ascontiguousarray(c, dtype=float)
        numba.set_num_threads(_THREADS[0])
        return _advect_faces_h(c, np.ascontiguousarray(Qx, dtype=float), np.ascontiguousarray(Qy, dtype=float),
                               np.ascontiguousarray(np.broadcast_to(ok, c.shape[1:])), self._ext_stack(c),
                               "west" in self.open, "east" in self.open, "south" in self.open, "north" in self.open,
                               self.tracer_scheme == "weno5")

    def _upwind_faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray):
        """Donor-cell face fluxes Q c_up (the FCT low-order flux); open edges as _advect_div (ext on inflow)."""
        ext = self._ext_stack(c)
        fx = np.zeros(c.shape[:3] + (c.shape[3] + 1,))
        fy = np.zeros(c.shape[:2] + (c.shape[2] + 1, c.shape[3]))
        fx[..., 1:-1] = Qx[None, ..., 1:-1] * np.where(Qx[None, ..., 1:-1] >= 0, c[..., :-1], c[..., 1:])
        fy[..., 1:-1, :] = Qy[None, :, 1:-1, :] * np.where(Qy[None, :, 1:-1, :] >= 0, c[..., :-1, :], c[..., 1:, :])
        if "west" in self.open:
            fx[..., 0] = Qx[None, ..., 0] * np.where(Qx[None, ..., 0] > 0, ext[..., 0], c[..., 0])
        if "east" in self.open:
            fx[..., -1] = Qx[None, ..., -1] * np.where(Qx[None, ..., -1] < 0, ext[..., -1], c[..., -1])
        if "south" in self.open:
            fy[..., 0, :] = Qy[None, :, 0, :] * np.where(Qy[None, :, 0, :] > 0, ext[..., 0, :], c[..., 0, :])
        if "north" in self.open:
            fy[..., -1, :] = Qy[None, :, -1, :] * np.where(Qy[None, :, -1, :] < 0, ext[..., -1, :], c[..., -1, :])
        return fx, fy

    def _advect_fct(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                    ok: np.ndarray, unlimited: bool = False) -> np.ndarray:
        """
        SSP-RK3 WENO5 advection of the stack C with Zalesak (1979) FCT (spec 2026-10-14 §4).  The high-order flux
        of the step is the RK3 combination F_H = (F(c0) + F(c1))/6 + 2F(c2)/3, so V1 c_RK3 = V0 c0 − Δt ∇·F_H;
        the low-order flux is donor cell, V1 c_L = V0 c0 − Δt ∇·F_L, bounded because limit_outflow keeps every
        layer's outflow ≤ its content; the antidiffusive flux A = F_H − F_L is limited by
        fct_coefficients (_fct_coeff_kernel).  ``unlimited``: C = 1 (the RK3 result; test of the identity).
        HYCOM mod_tsadvc.F90:877-905 does the same around its own scheme.
        """
        dt = self.dt
        dV = V1 - V0
        div = lambda fx, fy: fx[..., 1:] - fx[..., :-1] + fy[..., 1:, :] - fy[..., :-1, :]   # noqa: E731
        safe = lambda q, v: np.where(v > 0, q / np.where(v > 0, v, 1.0), C)                  # noqa: E731
        q0 = V0 * C
        fx0, fy0 = self._faces_h(C, Qx, Qy, ok)
        q1 = q0 - dt * div(fx0, fy0)
        c1 = safe(q1, V0 + dV)
        fx1, fy1 = self._faces_h(c1, Qx, Qy, ok)
        q2 = 0.75 * q0 + 0.25 * (q1 - dt * div(fx1, fy1))
        c2 = safe(q2, V0 + 0.5 * dV)
        fx2, fy2 = self._faces_h(c2, Qx, Qy, ok)
        FHx = (fx0 + fx1) / 6.0 + 2.0 * fx2 / 3.0
        FHy = (fy0 + fy1) / 6.0 + 2.0 * fy2 / 3.0
        FLx, FLy = self._upwind_faces_h(C, Qx, Qy)
        Ax, Ay = FHx - FLx, FHy - FLy
        if unlimited:
            return safe(q0 - dt * div(FHx, FHy), V1)
        cL = safe(q0 - dt * div(FLx, FLy), V1)
        okc = np.ascontiguousarray(np.broadcast_to(ok, C.shape[1:]))
        numba.set_num_threads(_THREADS[0])
        Cx, Cy = _fct_coeff_kernel(np.ascontiguousarray(C, dtype=float), np.ascontiguousarray(cL),
                                   np.ascontiguousarray(V1, dtype=float), np.ascontiguousarray(Ax),
                                   np.ascontiguousarray(Ay), okc, float(dt))
        return safe(q0 - dt * div(FLx + Cx * Ax, FLy + Cy * Ay), V1)
```

In `_step_ale`, change the advection call:

```python
        stack = np.stack([st.T, st.T**2]) if mixd else st.T[None]
        C = (self._advect_fct(stack, V0, V1, Qx, Qy, ok) if self.tracer_limiter == "fct"
             else self._advect(stack, V0, V1, Qx, Qy, Fz, ok))
```

In `core/restart.py` `_config` (ALE block), add `tracer_limiter=model.tracer_limiter`.

- [ ] **Step 5: Run the tests**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py tests/test_restart.py -q`

Expected: all pass.

If `test_fct_with_unit_coefficients_is_the_rk3_result` fails above 1e-12, the RK3 flux combination does not match `_advect`'s stage arithmetic (`_rk_stage`). Check that the `safe` fallback uses C (as `_advect_numpy`) and that stage 1 divides by V0 + dV. Fix `_advect_fct`, not the test.

If `test_fct_keeps_a_step_inside_its_range_where_weno_overshoots` reports that "none" is also bounded, WENO5-Z does not overshoot at that Courant. Raise the step contrast (T = 0 → 10) and the Courant to 0.95 until "none" overshoots, then ledger the change.

- [ ] **Step 6: Overflow z\* new extrema**

Run the z\* overflow with ν = 1000 and FCT on, one run:
```bash
NUMBA_NUM_THREADS=5 python -c "
import sys; sys.argv=['x']
from validation.ilicak_benchmark import run_pyocean as r, cases, metrics, series
import core.dynamics.sigma_baroclinic as sb
orig = sb.SigmaBaroclinic.__init__
def init(self, *a, **k): k.setdefault('tracer_limiter', 'fct'); orig(self, *a, **k)
sb.SigmaBaroclinic.__init__ = init
p = r.run('overflow', 'zstar', 1000.0, tag='fct')
s, _ = series.load(p); print('t_excess', metrics.t_excess(s))"
```

Expected: `t_excess` ≤ 1e-6 °C (it was 4.74 °C), printed as `(value, volume fraction)`. Ledger it.

- [ ] **Step 7: Commit**

```bash
git add core/dynamics/sigma_baroclinic.py core/restart.py tests/test_sigma_ale.py tests/test_restart.py
git commit -m "ALE: Zalesak FCT around the SSP-RK3 WENO5 tracer advection (option tracer_limiter)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

---

### Task 4: New defaults (θ_bc = ½ everywhere, "mom6", "fct") and old fingerprints pinned

**Files:**
- Modify: `core/dynamics/sigma_baroclinic.py` (constructor defaults and docstrings)
- Modify: `tests/test_sigma_ale.py` (`_perf_cases` old-scheme pins; `test_baroclinic_theta_defaults_to_centred_except_hybrid` → `test_baroclinic_theta_defaults_to_centred`; new defaults test)
- Test: the whole suite

**Interfaces:**
- Consumes: the options of Tasks 2 and 3.
- Produces: the defaults `theta_bc=None` → ½ for every ALE coordinate; `thin_layer_coupling=None` → "mom6" in ALE, else "none"; `tracer_limiter=None` → "fct" in ALE with SSP-RK3 WENO5, else "none".

- [ ] **Step 1: Failing tests**

Replace the body of `test_baroclinic_theta_defaults_to_centred_except_hybrid` and rename it:

```python
def test_baroclinic_theta_defaults_to_centred():
    # θ_bc = ½ in every coordinate since the outflow limiter (spec 2026-10-14 §5); an explicit value always wins
    hyb = dict(dz_top=np.full(10, 0.01), rho_target=np.linspace(1000.0, 1001.0, 11))
    assert _ale_box("zstar").theta_bc == 0.5
    assert _ale_box("sigma").theta_bc == 0.5
    assert _ale_box("hybrid", params=hyb).theta_bc == 0.5
    assert _ale_box("hybrid", params=hyb, theta_bc=0.6).theta_bc == 0.6


def test_robustness_defaults():
    hg = create_cartesian_grid(nx=8, ny=1, dx=1000.0, dy=1000.0, f0=0.0)
    m = _ale_box("zstar")
    assert m.thin_layer_coupling == "mom6" and m.tracer_limiter == "fct"
    e = SigmaBaroclinic(hg, np.full((1, 8), 20.0), dt=30.0, z0=None, closure="constant")      # σ Eulerian
    assert e.thin_layer_coupling == "none" and e.tracer_limiter == "none"
    assert _ale_box("zstar", tracer_time="pc").tracer_limiter == "none"
```

In `_perf_cases`, change `old = dict(theta_bc=0.6)` to:

```python
    old = dict(theta_bc=0.6, thin_layer_coupling="none", tracer_limiter="none")
```

Extend the comment above it: "…and before the robustness defaults (spec 2026-10-14): coupling and FCT off".

In `test_internal_wave_is_not_damped_by_the_free_surface_theta`, change `("old", dict(theta_bc=0.6))` to `("old", dict(theta_bc=0.6, thin_layer_coupling="none", tracer_limiter="none"))`.

The restart guard needs no new test: `test_ale_restart_is_bit_for_bit` (`tests/test_restart.py:180`) is parametrised over zstar, sigma and hybrid (with open boundaries). With the new defaults it runs the limiter, the coupling and FCT, so it pins their restart bit for bit.

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/test_sigma_ale.py -q -k "defaults"`

Expected: FAIL. The hybrid θ_bc is still 0.6, and `thin_layer_coupling` is still "none".

- [ ] **Step 2: Implement the defaults**

In the constructor signature, use `theta_bc: float | None = None, thin_layer_coupling: str | None = None, tracer_limiter: str | None = None`. Resolve them where each is validated:

```python
        if theta_bc is None:    # automatic: time-centred u′ in every coordinate (spec 2026-10-14 §5)
            theta_bc = 0.5
```

(remove the hybrid special case)

```python
        if thin_layer_coupling is None:
            thin_layer_coupling = "mom6" if vertical_coordinate == "ale" else "none"
```

```python
        if tracer_limiter is None:
            tracer_limiter = ("fct" if self.ale and self.tracer_time == "ssprk3"
                              and self.tracer_scheme in ("weno5", "weno5js") else "none")
```

Update the docstrings of `theta_bc`, `thin_layer_coupling` and `tracer_limiter`: say what None resolves to, and why the hybrid no longer keeps θ (the outflow limiter).

- [ ] **Step 3: Run the whole suite**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider > <workspace>/suite_task4.txt 2>&1; tail -5 <workspace>/suite_task4.txt`

Expected: everything passes except tests whose tolerances depended on the old scheme. For each failure:
- **Bit-level references** (other JSON fingerprints): pin `thin_layer_coupling="none", tracer_limiter="none", theta_bc=0.6` in that test, with a comment like the `_perf_cases` one.
- **Physics tolerances** (lock-exchange fronts, RPE, seamount, BayIW-style checks): do NOT loosen them. Ledger the before/after values and stop: decision point 4 with Esteban.

- [ ] **Step 4: Commit**

```bash
git add core/dynamics/sigma_baroclinic.py tests/test_sigma_ale.py tests/test_restart.py
git commit -m "ALE defaults: theta_bc = 0.5 everywhere, thin-layer coupling and FCT on

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```

Add any other test files you changed in Step 3 explicitly to the `git add`.

---

### Task 5: Validation runs (spec §6, criteria 1–6)

**Files:** none in the repo. Outputs go to `/Volumes/ESTEBAN_MIT/pyOCEAN_runs/`. Log files are never committed.

**Interfaces:** consumes the Task 4 defaults.

All runs go one at a time, with `NUMBA_NUM_THREADS=5`. Record each result in the ledger as soon as it finishes.

- [ ] **Step 1: Ilicak results, pyOCEAN only**

```bash
cd /Users/estebancruz/pyOCEAN && NUMBA_NUM_THREADS=5 python -c "
from validation.ilicak_benchmark import campaign, cases
for c in ('lock', 'overflow', 'iw', 'seamount'):
    for nu in cases.get(c).nu_h:
        for model, coord in campaign.CONFIGS[c]:
            if model == 'pyocean':
                campaign._run(campaign.command(c, model, coord, nu))
" > /Volumes/ESTEBAN_MIT/pyOCEAN_runs/ilicak_benchmark/log_robustness.txt 2>&1
for c in lock overflow iw seamount; do python validation/ilicak_benchmark/compare.py $c; done
```

Expected, checked against the spec's criteria:
- **(1)** hybrid IW completes at all three ν, with `amp_ratio` 0.65–0.75 / 0.55–0.65 / 0.24–0.27;
- **(2)** hybrid overflow ν = 1000 completes;
- **(3)** no "layer emptied" in the log;
- **(4)** `t_excess` ≤ 1e-6 in every pyOCEAN row;
- **(5)** hybrid lock-exchange ΔRPE/RPE₀ ≤ 2× (1.24e-7 / 1.34e-7 / 1.5e-7) with fronts 0.96–1.04;
- **(5)** lock exchange z\* within ±5 % of MOM6 z\* RPE (6.75e-5 / MOM6 rows of the README).

Also record the overflow hybrid ν = 1 and 100 outcome (the measured goal).

- [ ] **Step 2: Cost ladder, overflow, pyOCEAN z\* and hybrid**

Read `campaign.py cost` (`COST_HOURS`, `stable`) and run its pyOCEAN entries for the overflow z\* and hybrid configurations only, one at a time. Record Δt_max and s/h. The README's z\* value is 80 s / 0.40 s/h.

- [ ] **Step 3: BayIW and internal wave against MITgcm**

```bash
for case in 2layer linear real; do for coord in zstar hybrid; do
  NUMBA_NUM_THREADS=5 python validation/bayiw_2d/run_pyocean.py $case bay --coord $coord
  python validation/bayiw_2d/compare.py $case $coord orl
done; done
for case in 2layer linear; do for coord in zstar hybrid; do
  NUMBA_NUM_THREADS=5 python validation/bayiw_cut/run_pyocean.py $case bay --coord $coord
  python validation/bayiw_cut/compare.py $case $coord radiate
done; done
NUMBA_NUM_THREADS=5 python validation/internal_wave/run_pyocean.py zstar --ke arakawa
python validation/internal_wave/compare.py zstar_arakawa
```

Before running, read each README for the validated numbers. They are also in the CLAUDE.md "ALE mode" row:
- 2-D: seiche ±3 %, linear z\*/hybrid internal waves 0.97–1.05;
- 3-D cut: z\* seiche −1.8 %/±1 %, linear 0.98–1.07, two-layer isotherm 1.01–1.21;
- internal_wave flat part: 0.91–1.01.

Expected: every metric is the same or closer to MITgcm. If one is worse beyond its documented range, that is decision point 4.

- [ ] **Step 4: Cost per step on BayIW (criterion 6)**

Read `validation/bayiw_baroclinic/README.md` and `validation/bayiw_cut/profile_step.py` for the timing commands. Time z\* and hybrid with the new defaults and with `thin_layer_coupling="none", tracer_limiter="none"`.

Expected: ≤ +15 % per step. If it is more, profile with the same script and record which piece costs it before deciding with Esteban.

- [ ] **Step 5: Ledger summary**

One ledger line per criterion (1–6 and the measured goal): pass or fail with numbers. Any fail is a decision point: stop and report.

---

### Task 6: Documentation

**Files:**
- Modify: `validation/ilicak_benchmark/README.md` (tables of pyOCEAN rows, the "Lectura y clasificación" points 6, 7, 8, 9, and "Qué debe mejorar" items 1–4)
- Modify: `docs/design/vision_roadmap.md` §7 (the Ilicak row; the ALE rows if BayIW changed)
- Modify: `CLAUDE.md` (the "ALE mode" and "ALE face geometry" rows, the four-model benchmark row, the test count)
- Modify: `core/vertical/ale.py` module docstring (add the limiter and the viscous thickness to the "three face thicknesses" list)

**Interfaces:** consumes the Task 5 numbers.

- [ ] **Step 1: README of the benchmark**

Regenerate the pyOCEAN rows with `compare.py` and replace them in the tables (lock, overflow, IW, seamount, cost). Rewrite:
- **Point 6:** hybrid T out of range is closed by FCT; give the new range.
- **Point 7:** hybrid overflow robustness, with the ν = 1/100 outcome.
- **Point 8:** WENO5-Z extrema in z\* are closed by FCT; give the new RPE (it should rise a little: Ilicak 2012).
- **Point 9:** the hybrid now uses θ_bc = ½; give its oscillation.
- **"Qué debe mejorar":** strike items 2 (WENO5-Z extrema), 3 (thin-layer T) and 4 (robustness), each with a one-line "corregido el 2026-10-xx" and the evidence. Keep anything still open as a new item.

Write in Spanish, matching the existing text.

- [ ] **Step 2: Scorecard, CLAUDE.md, docstrings**

- **`vision_roadmap.md` §7:** update the Ilicak row and the status emoji to match Task 5.
- **CLAUDE.md:**
  - add to the "ALE face geometry" row: outflow limiter `limit_outflow` (h ≥ 0 guaranteed, Σ_k Q_k kept), viscous face thickness;
  - add to the "ALE mode" row: `thin_layer_coupling`, `tracer_limiter` (FCT) and θ_bc = ½ in every coordinate;
  - update the test count from `python -m pytest tests/ -q | tail -1`.
- **`ale.py` module docstring:** add two bullets to the face-thickness list: viscous coupling (harmonic) and the outflow limiter.

- [ ] **Step 3: Full suite and commit**

Run: `NUMBA_NUM_THREADS=5 python -m pytest tests/ -q -p no:cacheprovider | tail -1`

Expected: all pass (count = previous + the tests added here).

```bash
git add validation/ilicak_benchmark/README.md docs/design/vision_roadmap.md CLAUDE.md core/vertical/ale.py
git commit -m "Docs: hybrid robustness results (limiter, thin-layer coupling, FCT)

Co-Authored-By: Claude Opus 5.5 (1M context) <noreply@anthropic.com>
Claude-Session: https://claude.ai/code/session_01JMVgnUEXCCiLrQ7cAF37qn"
```
