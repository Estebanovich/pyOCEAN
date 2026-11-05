"""
Term-by-term momentum budget of the ALE vector-invariant step (gs_drift d11 Phase 4), read-only.

The diagnostic attaches to ``Driver.phase_hook`` and to ``Driver.keep_terms`` of a ``SigmaBaroclinic`` model and,
every step, splits the change of the layer velocity at the u- and v-faces into the terms of the step [m s⁻²]:

    DUDT = PF + CA + gKE + HV + EXT + IMP + BDY + REM,                                                     (1)

    DUDT = (uⁿ⁺¹ − uⁿ)/Δt
    PF   baroclinic finite-volume pressure gradient (Adcroft et al. 2008; ``core.vertical.pgf_nonlinear``)
    CA   PV flux q·(vh, −uh) with BOUND_CORIOLIS (Sadourny 1975; MOM6 MOM_CoriolisAdv.F90), f included;
         ``CAf`` is the same kernel with ζ = 0 (f-only part; approximate under the bound, which is nonlinear)
    gKE  −∇K (Arakawa KE; the vector-invariant term is CA + gKE, ``Momentum.vector_invariant``)
    HV   horizontal viscosity (biharmonic or Laplacian Smagorinsky stress divergence, ``core.dynamics.hor_visc``)
    EXT  what the time combination adds outside the AB3 history: body tide g∇η_body, −∇p_a/ρ₀ (G_ab − Σ AB3 terms)
    IMP  (u_imp − uⁿ − Δt G_ab)/Δt of the implicit phase: free surface g∇η (θ-weighted), vertical viscosity, bed
         drag and wind.  Split as  BT = (Σ_k h_k IMP_k − τ_s/ρ₀ + τ_b/ρ₀)/D  (depth-uniform, the free-surface part)
         and  VV_k = IMP_k − BT  (viscosity, wind and bed; Σ_k h_k VV_k = τ_s/ρ₀ − τ_b/ρ₀),  with D = Σ_k h_k
    BDY  open-edge relaxation of phase 5 (sponge on u′, Orlanski; thin-bed rule)
    REM  phase 10: velocity remap to the new layers, thin-bed rule, CFL backstop

Each explicit term is evaluated at uⁿ, hⁿ and combined with its own AB3 history and the coefficients of the step
(``AB3Momentum.combine``, linear), so Σ of the AB3 terms equals the model's G_ab up to round-off and (1) closes to
round-off by construction.  The weights are the face thicknesses hⁿ of phase 1 (``ctx.hu``, ``ctx.hv``: the mean of
the bed-clipped transport thicknesses, the column of the free surface).

Daily records (``steps_per_record`` steps) saved as ``<out>/mb_<record>.npz``, means over the steps of the record:
  I_<X>_u, I_<X>_v     Σ_k h_k X_k                  (ny, nx+1), (ny+1, nx) [m² s⁻²]
  P_<X>_u, P_<X>_v     Σ_k h_k X_k over 0–z_part (fraction of each layer above z_part below the surface)
  K_<X>_u, K_<X>_v     Σ_k h_k uⁿ_k X_k             [m³ s⁻³] (KE work per unit face area)
  W_<X>_u, W_<X>_v     X_k in the window (N, j1−j0+1, i1−i0+2), (N, j1−j0+2, i1−i0+1), float32, unweighted
  with X in PF, CA, CAf, gKE, HV, EXT, IMP, BT, VV, BDY, REM, DUDT; for BED and WIND only I_ and K_ (columns);
  U_u, U_v (Σ h u), Pu_u, Pu_v (0–z_part), D_u, D_v (Σ h), Wu, Wv, Whu, Whv (window u, v, face thickness).
Index order (k, j, i), k = 0 at the bed; u-faces have nx+1 columns, v-faces ny+1 rows.

MOM6 equivalents (0e9a561): PF + BT ↔ PFu + u_BT_accel, CA + gKE ↔ CAu, gKE ↔ gKEu, CA − CAf ↔ rvxv, HV ↔ diffu,
VV ↔ du_dt_visc, DUDT ↔ dudt (MOM_dynamics_split_RK2.F90:963-975; MOM_diagnostics.F90:1559-1588).
"""
from __future__ import annotations

from pathlib import Path

import numba
import numpy as np

from core.dynamics import vector_invariant as vi
from core.model import threads

TERMS3 = ("PF", "CA", "CAf", "gKE", "HV", "EXT", "IMP", "BT", "VV", "BDY", "REM", "DUDT")
TERMS2 = ("BED", "WIND")
AB_TERMS = ("PF", "CA", "CAf", "gKE", "HV")


def partial_weights(h: np.ndarray, z_part: float) -> np.ndarray:
    """h_k times the fraction of layer k (k = 0 at the bed) that lies within z_part [m] of the surface [m]."""
    top = np.cumsum(h[::-1], axis=0)[::-1] - h                       # depth of the top of each layer
    return np.clip(np.minimum(h, z_part - top), 0.0, None)


class MomentumBudget:
    """
    Read-only momentum budget of an ALE vector-invariant ``SigmaBaroclinic`` model (module docstring).

    window : (j0, j1, i0, i1) inclusive T-cell indices of the 3-D window record (the u-faces i0 … i1+1, v-faces
             j0 … j1+1); z_part : depth [m] of the partial integrals; steps_per_record : steps per saved record;
    out : directory of the records (None: kept in ``records`` in memory, for tests).
    """

    def __init__(self, model, window: tuple[int, int, int, int], z_part: float = 1000.0,
                 steps_per_record: int = 144, out: Path | None = None) -> None:
        if not model.ale or model._momentum.form != "vector_invariant":
            raise ValueError("MomentumBudget needs vertical_coordinate='ale' and momentum='vector_invariant'")
        if model._momentum.per:
            raise ValueError("MomentumBudget: periodic axes are not supported")
        self.m, self.z_part, self.spr = model, float(z_part), int(steps_per_record)
        j0, j1, i0, i1 = window
        self.su = (slice(None), slice(j0, j1 + 1), slice(i0, i1 + 2))
        self.sv = (slice(None), slice(j0, j1 + 2), slice(i0, i1 + 1))
        self.window = window
        self.out = None if out is None else Path(out)
        if self.out is not None:
            self.out.mkdir(parents=True, exist_ok=True)
        self.records: list[dict] = []
        self.hist = {x: [] for x in AB_TERMS}
        self.n_rec, self.k = 0, 0
        self.acc: dict[str, np.ndarray] = {}
        self.max_resid = 0.0                                          # largest |(1)| / max term over the run
        d = model._driver
        d.keep_terms = True
        d.phase_hook = self.hook

    # ------------------------------------------------------------------ per phase
    def hook(self, name: str, ctx) -> None:
        """``Driver.phase_hook``: the handler of phase ``name`` (reads ``ctx``, writes only to this object)."""
        f = getattr(self, "_" + name, None)
        if f is not None:
            f(ctx)

    def _explicit_momentum(self, ctx) -> None:
        m, st = self.m, ctx.st
        mom = m._momentum
        t = m._driver.last_terms
        g = mom.vi_geom
        numba.set_num_threads(threads._THREADS[0])
        ke = "gudonov" if mom.ke_scheme == "godunov" else mom.ke_scheme
        K = vi.kinetic_energy(st.u, st.v, g["area_u"], g["area_v"], g["area_t"], ke)
        kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
        cu, cv = t["vector_invariant"]
        hu, hv = mom.mean_face_thickness(st.h)                       # the PV-flux transports of vector_invariant
        Qx, Qy = mom.fs.Lu[None] * st.u * hu, mom.fs.Lv[None] * st.v * hv
        nz, ny, nx = st.h.shape
        q_f = vi.pv_q(np.zeros((nz, ny + 1, nx + 1)), mom.fq, st.h, g["area_t"], mom.fs.active)
        fu, fv = vi.coriolis_pv(q_f, np.broadcast_to(mom.fq[None], q_f.shape).copy(), Qx, Qy, st.u, st.v,
                                g["dx_u"], g["dy_v"], bound=True)
        zero = (np.zeros_like(st.u), np.zeros_like(st.v))
        now = {"PF": t["pgf"], "CA": (cu - kx, cv - ky), "CAf": (fu, fv), "gKE": (kx, ky),
               "HV": t.get("hviscosity", zero)}
        self.ab = {}
        for x in AB_TERMS:
            au, av, self.hist[x] = m._time_stepper.combine(self.hist[x], *now[x])
            self.ab[x] = (au, av)

    def _time_combine(self, ctx) -> None:
        dt = self.m.dt
        s = [self.ab[x] for x in ("PF", "CA", "gKE", "HV")]
        self.ab["EXT"] = (ctx.gu_ab - sum(a[0] for a in s), ctx.gv_ab - sum(a[1] for a in s))
        self.us = (ctx.st.u + dt * ctx.gu_ab, ctx.st.v + dt * ctx.gv_ab)

    def _implicit_and_free_surface(self, ctx) -> None:
        m, st, dt = self.m, ctx.st, self.m.dt
        self.u_imp = (ctx.u_new.copy(), ctx.v_new.copy())        # phase 5 may change u_new in place
        imp = ((ctx.u_new - self.us[0]) / dt, (ctx.v_new - self.us[1]) / dt)
        hu, hv = ctx.hu, ctx.hv
        # bed stress τ_b/ρ₀ = Σ_k r_k u_k^{n+1} with the rates of the solve (quadratic: from uⁿ over hbbl, as
        # Driver._ale_implicit_and_free_surface; law of the wall: at the new velocities, as Driver._ale_bed_rates)
        if m.bottom_drag == "quadratic":
            r_u, r_v, f_u, f_v = m._bed_drag.rates(st.u, st.v, hu, hv)
            r_u, r_v = r_u[None] * f_u, r_v[None] * f_v
        elif m.core.z0 is not None:
            r_u, r_v = m._bed_drag.rates(ctx.u_new, ctx.v_new, hu, hv)
        else:
            r_u, r_v = np.zeros_like(hu), np.zeros_like(hv)
        bed = ((r_u * ctx.u_new).sum(axis=0), (r_v * ctx.v_new).sum(axis=0))
        self.bed_ke = ((r_u * ctx.u_new * st.u).sum(axis=0), (r_v * ctx.v_new * st.v).sum(axis=0))
        wind = (np.zeros(hu.shape[1:]) if ctx.tu is None else np.broadcast_to(np.nan_to_num(ctx.tu), hu.shape[1:]),
                np.zeros(hv.shape[1:]) if ctx.tv is None else np.broadcast_to(np.nan_to_num(ctx.tv), hv.shape[1:]))
        wet = (hu.sum(axis=0) > 0, hv.sum(axis=0) > 0)                # faces with water (D > 0)
        bed = (np.where(wet[0], bed[0], 0.0), np.where(wet[1], bed[1], 0.0))
        wind = (np.where(wet[0], wind[0], 0.0), np.where(wet[1], wind[1], 0.0))
        bt = []
        for c, h in ((0, hu), (1, hv)):
            D = h.sum(axis=0)
            col = (h * imp[c]).sum(axis=0) - wind[c] + bed[c]
            bt.append(np.where(D > 0, col / np.where(D > 0, D, 1.0), 0.0))
        self.imp, self.bed, self.wind = imp, bed, wind
        self.bt = (np.broadcast_to(bt[0][None], hu.shape), np.broadcast_to(bt[1][None], hv.shape))

    def _boundary_velocity(self, ctx) -> None:
        dt = self.m.dt
        self.u_bdy = (ctx.u_new.copy(), ctx.v_new.copy())
        self.bdy = ((ctx.u_new - self.u_imp[0]) / dt, (ctx.v_new - self.u_imp[1]) / dt)

    def _finalize(self, ctx) -> None:
        st, new, dt = ctx.st, ctx.new_state, self.m.dt
        rem = ((new.u - self.u_bdy[0]) / dt, (new.v - self.u_bdy[1]) / dt)
        dudt = ((new.u - st.u) / dt, (new.v - st.v) / dt)
        T = dict(self.ab)
        T.update(IMP=self.imp, BT=self.bt, VV=(self.imp[0] - self.bt[0], self.imp[1] - self.bt[1]), BDY=self.bdy,
                 REM=rem, DUDT=dudt)
        for c in (0, 1):                                             # closure of (1), relative to the largest term
            s = sum(T[x][c] for x in ("PF", "CA", "gKE", "HV", "EXT", "IMP", "BDY", "REM"))
            big = max(float(np.max(np.abs(T[x][c]))) for x in ("PF", "CA", "gKE", "HV", "IMP"))
            if big > 0:
                self.max_resid = max(self.max_resid, float(np.max(np.abs(dudt[c] - s))) / big)
        self._accumulate(ctx, T)

    # ------------------------------------------------------------------ records
    def _add(self, key: str, val: np.ndarray) -> None:
        a = self.acc.get(key)
        if a is None:
            self.acc[key] = np.array(val, dtype=float)
        else:
            a += val

    def _accumulate(self, ctx, T: dict) -> None:
        st = ctx.st
        hs = (ctx.hu, ctx.hv)
        ws = (partial_weights(ctx.hu, self.z_part), partial_weights(ctx.hv, self.z_part))
        vel = (st.u, st.v)
        win = (self.su, self.sv)
        for c, sfx in ((0, "_u"), (1, "_v")):
            h, w, u = hs[c], ws[c], vel[c]
            hu_ = h * u
            for x in TERMS3:
                X = T[x][c]
                self._add("I_" + x + sfx, (h * X).sum(axis=0))
                self._add("P_" + x + sfx, (w * X).sum(axis=0))
                self._add("K_" + x + sfx, (hu_ * X).sum(axis=0))
                self._add("W_" + x + sfx, X[win[c]])
            for x, col in (("BED", self.bed[c]), ("WIND", self.wind[c])):
                self._add("I_" + x + sfx, -col if x == "BED" else col)
            self._add("K_BED" + sfx, -self.bed_ke[c])
            self._add("K_WIND" + sfx, self.wind[c] * u[-1])
            self._add("U" + sfx, hu_.sum(axis=0))
            self._add("Pu" + sfx, (w * u).sum(axis=0))
            self._add("D" + sfx, h.sum(axis=0))
            self._add("W" + sfx[1], u[win[c]])
            self._add("Wh" + sfx[1], h[win[c]])
        self.k += 1
        if self.k == self.spr:
            self._flush(ctx.new_state.time)

    def _flush(self, time: float) -> None:
        rec = {k: (v / self.k).astype(np.float32 if k.startswith("W") else float) for k, v in self.acc.items()}
        rec["time_end"] = np.array(time)
        rec["n_steps"] = np.array(self.k)
        rec["window"] = np.array(self.window)
        rec["max_resid"] = np.array(self.max_resid)
        if self.out is None:
            self.records.append(rec)
        else:
            np.savez(self.out / f"mb_{self.n_rec:04d}.npz", **rec)
        self.n_rec += 1
        self.acc, self.k = {}, 0
