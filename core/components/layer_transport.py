"""
Layer volume transports of the 3-D core (N1, role LAYER_TRANSPORT), moved unchanged from ``SigmaBaroclinic``.

σ (Eulerian): layer transports consistent with the free-surface flux, Q_k = L D Δσ_k u_k^θ with the depth mean
time-weighted by the free-surface θ and the baroclinic part by θ_bc (hybrid-robustness spec 2026-10-14 §5); at
open faces Δσ_k times the Flather flux plus the baroclinic part, so Σ_k Q_k is exactly the volume flux of the
continuity equation; the vertical flux Fz through the σ interfaces is diagnosed from layer continuity (cumulative
sum from the bed).

ALE: the Lagrangian step (no flux through the interfaces, Fz = 0) — transport velocities split θ/θ_bc
(``core.vertical.ale.transport_velocity``), the layer thickness flux PPM (MOM6 continuity_PPM) or donor cell, the
open-face fluxes, the outflow limiter (h ≥ 0, Σ_k Q_k kept) and the Lagrangian thickness h†.

Returns a ``LayerFluxes`` record used by the tracer step and carried as history (Qx, Qy, Fz).
Index order (k, j, i), k = 0 at the bed; Fz at the N+1 interfaces (0 = bed).
"""
from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from core.model.contract import BuildContext, Issue, Role
from core.vertical.ale import (lagrangian_thickness, limit_outflow, ppm_transport_x, transport_velocity,
                               upwind_transport_x)


@dataclass
class LayerFluxes:
    Qx: np.ndarray                   # layer volume transport at u-faces [m³ s⁻¹]
    Qy: np.ndarray                   # at v-faces
    Fz: np.ndarray                   # through the layer interfaces (N+1, ny, nx) [m³ s⁻¹]
    V0: np.ndarray                   # layer volumes before [m³]
    V1: np.ndarray                   # after (σ: new η; ALE: h†)
    ok: np.ndarray                   # wet cell-layers after the step
    h_dagger: np.ndarray | None = None   # ALE Lagrangian thickness h† [m]
    n_limited: int = 0               # ALE cell-layers scaled by the outflow limiter this step


class SigmaLayerTransport:
    name = "sigma_layer_transport"
    role = Role.LAYER_TRANSPORT
    reads = ("u", "v", "eta")
    writes = ("Qx", "Qy", "Fz")

    def __init__(self, *, fs, ds: np.ndarray, H: np.ndarray, dt: float, layer_volumes: Callable, ob) -> None:
        self.fs, self.ds, self.H, self.dt = fs, ds, H, dt
        self.layer_volumes, self.ob = layer_volumes, ob

    def __call__(self, st, eta_new: np.ndarray, u_new: np.ndarray, v_new: np.ndarray, fluxes, theta_bc: float,
                 open_: tuple[str, ...]) -> LayerFluxes:
        dt, th, n = self.dt, self.fs.theta, self.ds.size
        # layer transports consistent with the free-surface flux (fluxes.Du / Dv: upwind face depth)
        un = np.where(fluxes.wet_u[None], st.u, 0.0)
        vn = np.where(fluxes.wet_v[None], st.v, 0.0)
        # depth mean with the free-surface θ, baroclinic part with θ_bc (see transport_velocity)
        tb, ds = theta_bc, self.ds[:, None, None]
        ux = tb * u_new + (1 - tb) * un + ((th - tb) * ((ds * u_new).sum(0) - (ds * un).sum(0)))[None]
        vy = tb * v_new + (1 - tb) * vn + ((th - tb) * ((ds * v_new).sum(0) - (ds * vn).sum(0)))[None]
        Qx = (self.fs.Lu * fluxes.Du)[None] * ds * ux
        Qy = (self.fs.Lv * fluxes.Dv)[None] * ds * vy
        if open_:
            # open faces: Δσ_k × the free-surface (Flather) flux + the baroclinic part, so Σ_k Q_k is exactly
            # the volume flux of the continuity equation
            for Q, D, Qo, L, vo, vn1 in ((Qx, fluxes.open_Du, fluxes.open_Qu, self.fs.Lu, st.u, u_new),
                                         (Qy, fluxes.open_Dv, fluxes.open_Qv, self.fs.Lv, st.v, v_new)):
                po, pn = self.ob.baroclinic_part(vo)[1], self.ob.baroclinic_part(vn1)[1]
                ds = self.ds[:, None, None]
                Q[...] = np.where((D > 0)[None], ds * Qo[None] + (L * D)[None] * ds * (tb * pn + (1 - tb) * po), Q)
        V0, V1 = self.layer_volumes(st.eta), self.layer_volumes(eta_new)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        Fz = np.zeros((n + 1,) + self.H.shape)                              # volume flux through interface k (k=0 bed)
        Fz[1:] = -np.cumsum((V1 - V0) / dt + divh, axis=0)
        Fz[-1] = 0.0                                                        # surface: round-off residual only
        ok = (np.broadcast_to(self.fs.active[None], V1.shape)) & (V1 > 0)
        return LayerFluxes(Qx, Qy, Fz, V0, V1, ok)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}


class ALELayerTransport:
    name = "ale_layer_transport"
    role = Role.LAYER_TRANSPORT
    reads = ("u", "v", "eta", "h")
    writes = ("Qx", "Qy", "Fz", "h")

    def __init__(self, *, fs, H: np.ndarray, area: np.ndarray, dt: float, thickness_flux: str, vi_geom: dict,
                 layer_volumes: Callable, ob) -> None:
        self.fs, self.H, self.area, self.dt, self.thickness_flux = fs, H, area, dt, thickness_flux
        self.vi_geom, self.layer_volumes, self.ob = vi_geom, layer_volumes, ob

    def __call__(self, st, eta_new: np.ndarray, u_new: np.ndarray, v_new: np.ndarray, fluxes, theta_bc: float,
                 open_: tuple[str, ...], wu: np.ndarray, wv: np.ndarray, qe, qw, qn, qs) -> LayerFluxes:
        dt, th, n = self.dt, self.fs.theta, st.h.shape[0]
        act = self.fs.active
        # transport velocities and free-surface volume fluxes (core.vertical.ale.transport_velocity)
        tb = theta_bc
        utx, Qbx = transport_velocity(u_new, st.u, fluxes.wet_u, wu, self.fs.Lu, fluxes.Du, th, tb)
        uty, Qby = transport_velocity(v_new, st.v, fluxes.wet_v, wv, self.fs.Lv, fluxes.Dv, th, tb)
        if self.thickness_flux == "ppm":
            dx_t, dy_t = self.vi_geom["dx_t"], self.vi_geom["dy_t"]
            Qx = ppm_transport_x(st.h, utx, self.fs.Lu, Qbx, qe, qw, act, dt, dx_t)
            Qy = ppm_transport_x(st.h.swapaxes(1, 2), uty.swapaxes(1, 2), self.fs.Lv.T, Qby.T, qn, qs, act.T,
                                 dt, dy_t.T).swapaxes(1, 2)
        else:
            Qx = upwind_transport_x(qe, qw, utx, self.fs.Lu, Qbx)
            Qy = upwind_transport_x(qn, qs, uty.swapaxes(1, 2), self.fs.Lv.T, Qby.T).swapaxes(1, 2)
        if open_:                                                       # old velocity unmasked: the wet flag is False at edges
            self.ob.open_face_transports(Qx, Qy, st.h, st.u, st.v, u_new, v_new, fluxes)
        # outflow limiter: no layer loses more than it holds; Σ_k Q_k unchanged (core.vertical.ale.limit_outflow)
        Qx, Qy, n_lim = limit_outflow(st.h, Qx, Qy, self.area, dt, act)
        divh = Qx[..., 1:] - Qx[..., :-1] + Qy[:, 1:, :] - Qy[:, :-1, :]
        h_lag = lagrangian_thickness(st.h, divh, dt, self.area)
        V0, V1 = self.layer_volumes(st.eta, st.h), self.layer_volumes(eta_new, h_lag)
        Fz = np.zeros((n + 1,) + self.H.shape)                              # Lagrangian: no flux through interfaces
        ok = np.broadcast_to(act[None], V1.shape) & (V1 > 0)
        return LayerFluxes(Qx, Qy, Fz, V0, V1, ok, h_dagger=h_lag, n_limited=n_lim)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}
