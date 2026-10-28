"""
Fixed-order driver of the 3-D core (N1, spec docs/superpowers/specs/2026-10-28-n1-component-core-design.md §4).

One time step is ten named phases, the same order in σ and ALE (phases 8–9 do nothing in σ):

 1 prepare                    exterior state at tⁿ⁺¹ (open boundaries); ALE: face geometry of hⁿ
 2 explicit_momentum          named explicit terms (PGF, Coriolis or vector-invariant, advection, viscosity, …)
 3 time_combine               AB3 (AB1, AB2 at start-up); τ/ρ₀, not extrapolated
 4 implicit_and_free_surface  implicit vertical viscosity + bed drag + wind, free surface, Flather, closure
 5 boundary_velocity          sponge on u′, Orlanski; ALE: thin bed layers, optional blending
 6 layer_transport            layer volume transports (σ: Fz from continuity; ALE: Lagrangian h†)
 7 tracers                    advection (SSP-RK3/FCT or 'pc'), mixing diagnostic, K_h, K_v, boundary sponge
 8 vertical_coordinate        ALE: regrid + remap (and the remap part of the mixing diagnostic)
 9 interior_sponge            ALE: interior sponge after the remap
10 finalize                   ALE: velocity remap in face-fraction space, CFL backstop; the new state

The order is the one of ``SigmaBaroclinic.step`` before N1 (bit for bit, tests/test_n1_golden.py) and follows the
reference models read for N1 (docs/design/n1_refs/): MOM6 step_MOM / MOM_dynamics_split_RK2 (dynamics, then
tracers on the accumulated transports, then ALE), CROCO step.F (predictor, barotropic, corrector, tracers) and
MITgcm forward_step (one fixed sequence, packages at fixed hook points).  Everything a phase hands to the next one
is a local of the step (``StepContext``), not model state; after the step the driver leaves the last exterior,
barotropic fluxes and layer fluxes on the model for inspection only.

``debug=True`` hashes the arrays alive at the end of every phase (``phase_hashes``), to find the first phase that
departs from a reference run; ``keep_terms=True`` keeps the named explicit momentum terms (``last_terms``).
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Any

import numba
import numpy as np

from core.components.momentum import sum_terms
from core.dynamics.sigma_wd import vertical_implicit
from core.model import threads
from core.vertical.ale import (DZ_FLOOR, blend_thin_layers, face_fractions, viscous_face_thickness,
                               zero_thin_bed_velocity)

PHASES = ("prepare", "explicit_momentum", "time_combine", "implicit_and_free_surface", "boundary_velocity",
          "layer_transport", "tracers", "vertical_coordinate", "interior_sponge", "finalize")


@dataclass
class StepContext:
    """What one phase hands to the next within a step."""
    st: Any
    tau_x: Any = None
    tau_y: Any = None
    vals: dict = field(default_factory=dict)

    def __getattr__(self, name: str):
        try:
            return self.__dict__["vals"][name]
        except KeyError:
            raise AttributeError(name) from None

    def set(self, **kw) -> None:
        self.vals.update(kw)


def _hash_arrays(vals: dict) -> str:
    h = hashlib.sha256()
    for k in sorted(vals):
        v = vals[k]
        if isinstance(v, np.ndarray):
            h.update(k.encode())
            h.update(np.ascontiguousarray(v).tobytes())
    return h.hexdigest()


class Driver:
    """Runs the ten phases with the components of a ``SigmaBaroclinic`` model (its facade)."""

    def __init__(self, model, *, debug: bool = False, keep_terms: bool = False, dt_dyn: float | None = None,
                 dt_tracer: float | None = None, dt_therm: float | None = None) -> None:
        self.m = model
        self.debug, self.keep_terms = debug, keep_terms
        self.dt_dyn = model.dt if dt_dyn is None else dt_dyn
        self.dt_tracer = model.dt if dt_tracer is None else dt_tracer
        self.dt_therm = model.dt if dt_therm is None else dt_therm
        self.phase_hashes: list[dict[str, str]] = []
        self.last_terms: dict | None = None

    # ------------------------------------------------------------------ driver
    def step(self, st, tau_x=None, tau_y=None):
        ctx = StepContext(st, tau_x, tau_y)
        hashes: dict[str, str] = {}
        ale = self.m.ale
        for name in PHASES:
            getattr(self, ("_ale_" if ale else "_sigma_") + name)(ctx)
            if self.debug:
                hashes[name] = _hash_arrays(ctx.vals)
        if self.debug:
            self.phase_hashes.append(hashes)
        m = self.m                                   # last-step records, for inspection only
        m._ext, m._fluxes, m._last_layer_fluxes = ctx.ext, ctx.fluxes, ctx.lf
        m._T_in = ctx.vals.get("T_in", m._T_in)
        m._tr_in = ctx.vals.get("tr_in", m._tr_in)
        return ctx.new_state

    def _inflow(self, stack: np.ndarray, T_in, tr_in):
        """Inflow stack of the advected ``stack`` (None: no open edge or no exterior → each stage's own values)."""
        m = self.m
        ext = m._ob.inflow_stack(stack, m.T_ext if T_in is None else T_in, tr_in)
        return None if ext is stack else ext

    # ------------------------------------------------------------------ phases shared by σ and ALE
    def _explicit_momentum(self, ctx: StepContext) -> None:
        terms, wet_u, wet_v = self.m._momentum_terms(ctx.st)
        if self.keep_terms:
            self.last_terms = {name: (tu, tv) for name, tu, tv, _ in terms}
        gu, gv = sum_terms(terms, wet_u, wet_v)
        ctx.set(gu=gu, gv=gv)

    def _time_combine(self, ctx: StepContext) -> None:
        m = self.m
        gu_ab, gv_ab, hist = m._time_stepper.combine(ctx.st.g_hist, ctx.gu, ctx.gv)
        tu = None if ctx.tau_x is None else np.asarray(ctx.tau_x, float) / m.rho0
        tv = None if ctx.tau_y is None else np.asarray(ctx.tau_y, float) / m.rho0
        ctx.set(gu_ab=gu_ab, gv_ab=gv_ab, hist=hist, tu=tu, tv=tv)

    _sigma_explicit_momentum = _ale_explicit_momentum = _explicit_momentum
    _sigma_time_combine = _ale_time_combine = _time_combine

    # ------------------------------------------------------------------ σ (Eulerian)
    def _sigma_prepare(self, ctx: StepContext) -> None:
        m, st = self.m, ctx.st
        ext, T_in = None, None
        if m.obc_state is not None:
            ds = m.ds[:, None, None]
            Hu_e, Hv_e = m._ob.edge_face_depths()
            ext = m._ob.evaluate(st.time + m.dt, Hu_e[None] * ds, Hv_e[None] * ds, None, None)
            D0 = np.maximum(np.nan_to_num(m.H + st.eta, nan=m.fs.d_min), m.fs.d_min)
            T_in = m._ob.exterior_T(D0[None] * ds, ext)
            ctx.set(T_in=T_in)
        ctx.set(ext=ext)

    def _sigma_implicit_and_free_surface(self, ctx: StepContext) -> None:
        m, st = self.m, ctx.st
        eta_new, u_new, v_new, fluxes, closure = m._free_surface.step(st.eta, st.u, st.v, ctx.gu_ab, ctx.gv_ab, ctx.tu,
                                                                      ctx.tv, st.closure, m._ob.flather_args(ctx.ext))
        ctx.set(eta_new=eta_new, u_new=u_new, v_new=v_new, fluxes=fluxes, closure=closure)

    def _sigma_boundary_velocity(self, ctx: StepContext) -> None:
        self.m._ob.relax_velocity(ctx.st.u, ctx.st.v, ctx.u_new, ctx.v_new, ctx.ext, ctx.fluxes)

    def _sigma_layer_transport(self, ctx: StepContext) -> None:
        m = self.m
        ctx.set(lf=m._layer_transport(ctx.st, ctx.eta_new, ctx.u_new, ctx.v_new, ctx.fluxes, m.theta_bc, m.open))

    def _sigma_tracers(self, ctx: StepContext) -> None:
        m, st, lf = self.m, ctx.st, ctx.lf
        dt, n = m.dt, m.n
        T_in_now = ctx.vals.get("T_in", m._T_in)                      # σ keeps the last provider value
        stack = np.stack([st.T, st.T**2])
        C = m._tracer_advection.advect(stack, lf.V0, lf.V1, lf.Qx, lf.Qy, lf.Fz, lf.ok,
                                       self._inflow(stack, T_in_now, None))
        T_new = C[0]
        mix = lf.V1 * (C[1] - C[0] ** 2) / dt
        if m.kh > 0:
            T_new = m._tracer_diffusion.diffuse_h(T_new, lf.V1, ctx.eta_new)
        if m.kv > 0:
            D1 = np.maximum(np.nan_to_num(m.H + ctx.eta_new, nan=m.fs.d_min), m.fs.d_min)
            dz = (D1[None] * m.ds[:, None, None]).reshape(n, -1)
            T_new = vertical_implicit(T_new.reshape(n, -1), dz, np.full((n + 1, dz.shape[1]), m.kv),
                                      np.zeros(dz.shape[1]), dt).reshape(T_new.shape)
        if m._sp_c is not None and m.sponge_tracer:
            r = dt * m._sp_c[None]
            T_new = (T_new + r * (m.T_ext if ctx.ext is None else T_in_now)) / (1.0 + r)
        ctx.set(T_new=T_new, mix=mix)

    def _sigma_vertical_coordinate(self, ctx: StepContext) -> None:
        pass

    def _sigma_interior_sponge(self, ctx: StepContext) -> None:
        pass

    def _sigma_finalize(self, ctx: StepContext) -> None:
        m, st, lf = self.m, ctx.st, ctx.lf
        omega = lf.Fz / np.where(m.area > 0, m.area, 1.0)[None]
        ctx.set(new_state=type(st)(eta=ctx.eta_new, u=ctx.u_new, v=ctx.v_new, T=ctx.T_new, g_hist=ctx.hist,
                                   omega=omega, mix_num=ctx.mix, Qx=lf.Qx, Qy=lf.Qy, Fz=lf.Fz, time=st.time + m.dt,
                                   closure=ctx.closure))

    # ------------------------------------------------------------------ ALE
    def _ale_prepare(self, ctx: StepContext) -> None:
        m, st = self.m, ctx.st
        numba.set_num_threads(threads._THREADS[0])                         # parallel ALE kernels (≤ 5 P-cores)
        (qe, qw), (qn, qs) = m._vertical.transport_thickness(st.h)
        hu, hv = 0.5 * (qe + qw), (0.5 * (qn + qs)).swapaxes(1, 2)
        wu, wv = face_fractions(hu), face_fractions(hv)
        ext = m._ob.evaluate(st.time + m.dt, hu, hv, wu, wv)
        ctx.set(qe=qe, qw=qw, qn=qn, qs=qs, hu=hu, hv=hv, wu=wu, wv=wv, ext=ext)

    def _ale_implicit_and_free_surface(self, ctx: StepContext) -> None:
        m, st = self.m, ctx.st
        qe, qw, qn, qs, hu, hv = ctx.qe, ctx.qw, ctx.qn, ctx.qs, ctx.hu, ctx.hv
        if m.thin_layer_coupling == "mom6":                            # viscous tie of vanished layers (spec §3)
            hu_v = viscous_face_thickness(qe, qw, st.u)                 # upwind-biased (MOM6, Hallberg & Rhines)
            hv_v = viscous_face_thickness(qn, qs, st.v.swapaxes(1, 2)).swapaxes(1, 2)
            visc = dict(hu=np.maximum(hu_v, DZ_FLOOR), hv=np.maximum(hv_v, DZ_FLOOR),
                        hu_drag=np.maximum(hu[0], DZ_FLOOR), hv_drag=np.maximum(hv[0], DZ_FLOOR),
                        hu_mean=hu, hv_mean=hv)                         # ū on the clipped face column
        else:
            visc = dict(hu=np.maximum(hu, DZ_FLOOR), hv=np.maximum(hv, DZ_FLOOR))
        if m.bottom_drag == "quadratic":                               # bed stress at uⁿ (spec 2026-10-14-dome §3)
            r_u, r_v, f_u, f_v = m._bed_drag.rates(st.u, st.v, hu, hv)
            visc.update(r_u=r_u[None] * f_u, r_v=r_v[None] * f_v)      # stress spread over hbbl
            if "hu_mean" not in visc:                                  # δ needs the depth-mean column with drag
                visc.update(hu_mean=hu, hv_mean=hv)
        eta_new, u_new, v_new, fluxes, _ = m._free_surface.step(st.eta, st.u, st.v, ctx.gu_ab, ctx.gv_ab, ctx.tu,
                                                                ctx.tv, st.closure, m._ob.flather_args(ctx.ext),
                                                                wu=ctx.wu, wv=ctx.wv, **visc)
        ctx.set(eta_new=eta_new, u_new=u_new, v_new=v_new, fluxes=fluxes)

    def _ale_boundary_velocity(self, ctx: StepContext) -> None:
        m, st = self.m, ctx.st
        wu, wv, hu, hv = ctx.wu, ctx.wv, ctx.hu, ctx.hv
        u_new, v_new = ctx.u_new, ctx.v_new
        m._ob.relax_velocity(st.u, st.v, u_new, v_new, ctx.ext, ctx.fluxes, wu, wv)   # sponge on u′, Orlanski
        u_new = zero_thin_bed_velocity(u_new, hu, wu, m.h_thin_bed)
        v_new = zero_thin_bed_velocity(v_new, hv, wv, m.h_thin_bed)
        if m.thin_layer_cutoff > 0.0:                                   # HYCOM blending of thin layers (spec §3, 1B)
            # on the upwind (donor) thickness, not the mean (Hallberg & Rhines 1996): a nose advancing from a thick
            # cell keeps its speed, a layer draining from where it has vanished moves with the layer above
            u_up = np.where(u_new >= 0, ctx.qe, ctx.qw)
            v_up = np.where(v_new.swapaxes(1, 2) >= 0, ctx.qn, ctx.qs).swapaxes(1, 2)
            u_new = blend_thin_layers(u_new, u_up, wu, m.thin_layer_cutoff)
            v_new = blend_thin_layers(v_new, v_up, wv, m.thin_layer_cutoff)
        ctx.set(u_new=u_new, v_new=v_new)

    def _ale_layer_transport(self, ctx: StepContext) -> None:
        m = self.m
        ctx.set(lf=m._layer_transport(ctx.st, ctx.eta_new, ctx.u_new, ctx.v_new, ctx.fluxes, m.theta_bc, m.open,
                                      ctx.wu, ctx.wv, ctx.qe, ctx.qw, ctx.qn, ctx.qs))

    def _ale_tracers(self, ctx: StepContext) -> None:
        m, st, lf, ext = self.m, ctx.st, ctx.lf, ctx.ext
        dt, n = m.dt, m.n
        h_lag, V1 = lf.h_dagger, lf.V1
        T_in = m._ob.exterior_T(st.h, ext) if m.open else None         # inflow T on hⁿ (spec §3.5)
        ntr = m.n_tracers
        tr_in = m._ob.exterior_tr(st.h, ext) if (m.open and ntr) else None
        mixd = m.mix_diagnostic
        lead = 2 if mixd else 1                                         # [T, T²] or [T], then the passive tracers
        parts = [st.T, st.T**2] if mixd else [st.T]
        stack = np.stack(parts + ([] if st.tr is None else list(st.tr)))
        inflow = self._inflow(stack, T_in, tr_in)
        adv = m._tracer_advection
        C = (adv.advect_fct(stack, lf.V0, V1, lf.Qx, lf.Qy, lf.ok, inflow=inflow) if m.tracer_limiter == "fct"
             else adv.advect(stack, lf.V0, V1, lf.Qx, lf.Qy, lf.Fz, lf.ok, inflow))
        mix = V1 * (C[1] - C[0] ** 2) / dt if mixd else None                # advection part
        T_lag = C[0]
        tr_lag = [C[lead + i] for i in range(ntr)]
        if m.kh > 0:
            dif = m._tracer_diffusion
            T_lag = dif.diffuse_h(T_lag, V1, ctx.eta_new, h_lag)
            tr_lag = [dif.diffuse_h(c, V1, ctx.eta_new, h_lag, anomaly=False) for c in tr_lag]
        if m.kv > 0:
            dz = np.maximum(h_lag, DZ_FLOOR).reshape(n, -1)
            kvz, zero = np.full((n + 1, dz.shape[1]), m.kv), np.zeros(dz.shape[1])
            T_lag = vertical_implicit(T_lag.reshape(n, -1), dz, kvz, zero, dt).reshape(T_lag.shape)
            tr_lag = [vertical_implicit(c.reshape(n, -1), dz, kvz, zero, dt).reshape(c.shape) for c in tr_lag]
        if m._sp_c is not None and m.sponge_tracer:                     # tracer sponge toward the exterior on h†
            r = dt * m._sp_c[None]
            T_lag = (T_lag + r * m._ob.exterior_T(h_lag, ext)) / (1.0 + r)
            if ntr:
                tr_lag = [(c + r * e) / (1.0 + r) for c, e in zip(tr_lag, m._ob.exterior_tr(h_lag, ext))]
        ctx.set(T_in=T_in, tr_in=tr_in, T_lag=T_lag, tr_lag=tr_lag, mix=mix, lead=lead)

    def _ale_vertical_coordinate(self, ctx: StepContext) -> None:
        m, dt = self.m, self.m.dt
        mixd, ntr, lead = m.mix_diagnostic, m.n_tracers, ctx.lead
        T_lag, tr_lag, eta_new = ctx.T_lag, ctx.tr_lag, ctx.eta_new
        fields = ([T_lag, T_lag**2] if mixd else [T_lag]) + tr_lag
        h_new, rem = m._vertical.regrid_remap(eta_new, ctx.lf.h_dagger, fields, T_lag)
        T_new = rem[0]
        tr_new = np.stack(rem[lead:]) if ntr else None
        mix = ctx.mix
        if mixd:                                                       # remap part, before the sponge (a forcing)
            mix = mix + m.layer_volumes(eta_new, h_new) * (rem[1] - T_new**2) / dt
        ctx.set(h_new=h_new, T_new=T_new, tr_new=tr_new, mix=mix)

    def _ale_interior_sponge(self, ctx: StepContext) -> None:
        m = self.m
        if m._interior_sponge_c is not None:                           # interior sponge (spec 2026-10-14-dome §2)
            T_new, tr_new = m._interior_sponge_c.apply(ctx.T_new, ctx.tr_new, ctx.eta_new, ctx.h_new)
            ctx.set(T_new=T_new, tr_new=tr_new)

    def _ale_finalize(self, ctx: StepContext) -> None:
        m, st, lf, vc = self.m, ctx.st, ctx.lf, self.m._vertical
        dt = m.dt
        hu_lag, hv_lag = vc.mean_face_thickness(lf.h_dagger)
        hu1, hv1 = vc.mean_face_thickness(ctx.h_new)
        wu1, wv1 = face_fractions(hu1), face_fractions(hv1)                # once: remap target and thin-bed rule
        u_new = zero_thin_bed_velocity(vc.remap_faces(ctx.u_new, hu_lag, hu1, wu1), hu1, wu1, m.h_thin_bed)
        v_new = zero_thin_bed_velocity(vc.remap_faces(ctx.v_new, hv_lag, hv1, wv1), hv1, wv1, m.h_thin_bed)
        u_new, v_new, n_trunc = vc.truncate(u_new, v_new, hu1, hv1, dt, st.n_trunc)
        ctx.set(new_state=type(st)(eta=ctx.eta_new, u=u_new, v=v_new, T=ctx.T_new, h=ctx.h_new, g_hist=ctx.hist,
                                   omega=None, mix_num=ctx.mix, Qx=lf.Qx, Qy=lf.Qy, Fz=lf.Fz, time=st.time + dt,
                                   n_trunc=n_trunc, n_limited=st.n_limited + lf.n_limited, tr=ctx.tr_new))
