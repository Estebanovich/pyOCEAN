"""
Open boundaries of the 3-D core (N1, role OPEN_BOUNDARY), moved unchanged from ``SigmaBaroclinic``.

* Flather (1976) radiation of the depth mean toward an exterior (η, ū) — targets for the free surface;
* flow-relaxation sponge on the baroclinic velocity u′ (Martinsen & Engedahl 1987), toward 0 or toward the
  exterior u′ₑₓₜ;
* Orlanski (1976) radiation of u′ at the open faces, implicit upstream form as ROMS u3dbc (Marchesiello et al.
  2001), with CROCO's in/out nudging (u3dbc.F:119-152) or a 'clamped' (Dirichlet) edge;
* inflow tracer values (upstream, Marsaleix et al. 2006) and the exterior profiles on the current layers — T, the
  salinity S (``salinity=True``, N2a: ``S_ext`` fixed by ``initial_state`` as ``T_ext``, ``OBData.S`` from a
  provider) and the passive tracers;
* open-face layer transports adding up to the Flather flux (``core.vertical.ale.open_face_transport``).

The exterior at a time t is **returned** by ``evaluate`` (a dict: provider state, Flather targets, exterior u′ on
the face layers, owned columns) and passed explicitly to the other methods; the component keeps no state that
changes from step to step (spec 2026-10-28 §3).  Its configuration (edges, modes, masks of the boundary columns
and faces, sponge rates, the exterior profiles fixed by ``initial_state``) is set by ``SigmaBaroclinic``, which
exposes it as properties of the same names (``CONFIG``).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

from dataclasses import replace as dc_replace

import numba
import numpy as np

from core.dynamics import open_boundary as obc
from physics.eos.convert import to_model_ts
from core.model import threads
from core.model.contract import BuildContext, Issue, Role
from core.vertical import remap as rm
from core.vertical.ale import baroclinic_split, exterior_on_layers, open_face_transport

# configuration shared with SigmaBaroclinic (exposed there as properties of the same name)
CONFIG = ("open", "obc_state", "obc_mode", "obc_tau_in", "obc_tau_out", "orlanski_cmax", "sponge_tracer",
          "theta_bc", "T_ext", "h_ext", "tr_ext", "S_ext", "_bc_mask", "_ext_cols", "_bc_u", "_bc_v", "_sp_c", "_sp_u",
          "_sp_v", "obc_tangential", "_tan_q")


def convert_exterior(provider, eos_ts_kind: str, ts_kind: str):
    """
    Wrap the time-varying exterior ``provider(t) -> {edge: OBData}`` so that its T and S reach the model in the EOS
    convention ``eos_ts_kind`` although the data are in ``ts_kind`` (TEOS-10 in 3-D, task 8).  The conversion
    (``physics.eos.convert.to_model_ts``: θ, S_P → Θ, S_A = (35.16504/35) S_P, or back) is done **once, when the exterior
    is received**, on every call at time t; the model never converts again.  T and S are (nz, n_bdy); a provider
    that gives T without S cannot be converted (Θ needs S_A) and raises.  Returns ``provider`` itself when the two
    conventions agree (no wrapper, bit for bit).
    """
    if ts_kind == eos_ts_kind:
        return provider

    def converted(t: float) -> dict:
        out = {}
        for edge, ob in provider(t).items():
            if ob.T is None:
                out[edge] = ob
                continue
            if ob.S is None:
                raise ValueError(f"{edge}: OBData has T but no S: the conversion {ts_kind!r} → {eos_ts_kind!r} "
                                 "needs both")
            T, S = to_model_ts(np.asarray(ob.T, float), np.asarray(ob.S, float), eos_ts_kind, ts_kind)
            out[edge] = dc_replace(ob, T=T, S=S)
        return out

    return converted


class OpenBoundary:
    name = "open_boundary"
    role = Role.OPEN_BOUNDARY
    reads = ("u", "v", "h", "Qx", "Qy")
    writes = ("u", "v", "Qx", "Qy")
    writes_inplace = ("u", "v", "Qx", "Qy")

    def __init__(self) -> None:
        self.open: tuple[str, ...] = ()
        self.obc_state = None
        self.obc_mode: dict = {}
        self.obc_tau_in = self.obc_tau_out = 0.0
        self.orlanski_cmax, self.sponge_tracer, self.theta_bc = 1.0, True, 0.5
        self.T_ext = self.h_ext = self.tr_ext = self.S_ext = None
        self._bc_mask = self._ext_cols = self._bc_u = self._bc_v = None
        self._sp_c = self._sp_u = self._sp_v = None
        self.dt, self.ds, self.fs, self.H = 0.0, None, None, None
        self.n_tracers, self.ale, self.remap_code = 0, False, rm.PPM_H4
        self.salinity = False
        self.obc_tangential, self._tan_q = "none", None        # MOM6 ORLANSKI_TAN + NUDGED_TAN (corners, 'mom6')

    @property
    def tangential(self) -> bool:
        """True when the tangential velocity of the 'radiating' edges is radiated and nudged (``obc_tangential``
        'mom6') and some corner qualifies."""
        return self.obc_tangential == "mom6" and self._tan_q is not None and bool(self._tan_q.any())

    def bind(self, *, dt: float, ds: np.ndarray, fs, H: np.ndarray, n_tracers: int, ale: bool, remap_code,
             salinity: bool = False) -> None:
        """Model geometry and sizes, fixed after construction."""
        self.dt, self.ds, self.fs, self.H = dt, ds, fs, H
        self.n_tracers, self.ale, self.remap_code = n_tracers, ale, remap_code
        self.salinity = bool(salinity)

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        """Exterior profiles fixed by ``initial_state`` (written to the restart)."""
        return {k: getattr(self, k) for k in ("T_ext", "h_ext", "tr_ext", "S_ext") if getattr(self, k) is not None}

    def baroclinic_part(self, vel: np.ndarray, w: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """
        Depth mean and baroclinic remainder u_k − ū: Σ Δσ_k u_k (σ, ``w`` None — kept as tensordot so
        σ runs stay bit-identical) or Σ w_k u_k with the ALE face fractions (core.vertical.ale.baroclinic_split).
        """
        if w is None:
            mean = np.tensordot(self.ds, vel, axes=(0, 0))
            return mean, vel - mean[None]
        return baroclinic_split(vel, w)

    def radiate(self, u_old: np.ndarray, u_new: np.ndarray, axis: int, edge: int,
                 w: np.ndarray | None = None, ext: np.ndarray | None = None, mode: str = "radiating",
                 fluxes=None) -> np.ndarray:
        """
        Orlanski (1976) radiation of the baroclinic normal velocity at one open edge, implicit
        upstream form as ROMS u3dbc (Marchesiello et al. 2001, normal part only):

            r = c Δt/Δx = (φ₁ⁿ − φ₁ⁿ⁺¹)(φ₁ⁿ − φ₂ⁿ) / (φ₁ⁿ − φ₂ⁿ)²,  clipped to [0, r_max]
            φ_Bⁿ⁺¹ = (φ_Bⁿ + r φ₁ⁿ⁺¹) / (1 + r)

        with B the boundary face and 1, 2 the next faces inwards (r = 0: incoming → clamped).
        The depth-mean part (Flather) is kept; u_new is modified in place.  ``w``: depth-mean weights
        (Δσ when None, the ALE face fractions of hⁿ for both levels; spec 2026-10-12-ale-open-boundaries §3.2).
        ``ext``: exterior u′ on the face layers (spec 2026-10-12-baroclinic-obc-internal-wave §2): 'radiating'
        nudges the radiated value toward it, u′_B ← (1 − α) u′_B + α u′ₑₓₜ with α = Δt/τ_in when flowing in
        (r = 0) and Δt/τ_out when out (CROCO u3dbc.F:119-127, 144-152); 'clamped' sets u′_B = u′ₑₓₜ.
        Returns r (N, n) along the edge (the phase speed of the tangential corners, ``obc_tangential``).
        """
        _, po = self.baroclinic_part(u_old, w)
        bar, pn = self.baroclinic_part(u_new, w)
        i1, i2 = (1, 2) if edge == 0 else (-2, -3)
        take = lambda a, i: np.take(a, i, axis=axis)                                  # noqa: E731
        grad = take(po, i1) - take(po, i2)
        cx = (take(po, i1) - take(pn, i1)) * grad
        r = np.clip(cx / np.maximum(grad**2, 1e-30), 0.0, self.orlanski_cmax)
        pb = (take(po, edge) + r * take(pn, i1)) / (1.0 + r)
        if ext is not None:
            pe = take(ext, edge)
            if mode == "clamped":
                pb = pe.copy()
            else:
                alpha = np.where(cx <= 0.0, self.dt / self.obc_tau_in, self.dt / self.obc_tau_out)
                pb = (1.0 - alpha) * pb + alpha * pe
        if w is None:
            pb -= np.tensordot(self.ds, pb, axes=(0, 0))[None]                        # keep Σ Δσ u′ = 0
        else:
            pb -= (take(w, edge) * pb).sum(axis=0)[None]                              # keep Σ w u′ = 0
        open_ = take((fluxes.open_Du if axis == 2 else fluxes.open_Dv)[None], edge) > 0
        sl = [slice(None)] * u_new.ndim
        sl[axis] = edge
        u_new[tuple(sl)] = np.where(open_, take(bar[None], edge) + pb, take(u_new, edge))
        return r

    def flather_args(self, ext: dict | None, tide: tuple | None = None, ib: tuple | None = None) -> dict:
        """
        Flather exterior for the free-surface step: the provider's (η, ū)ₑₓₜ, or a resting exterior (η = 0, ū = 0);
        {} if closed.  ``tide`` (η_u, η_v, ū_u, v̄_v) from ``BoundaryTides.flather_arrays`` (regional chain Task 5;
        zero away from the 'radiating' edges) is added to η and ū only — u′ₑₓₜ is not touched.  ``ib`` (η_u, η_v):
        the inverse barometer of the 'radiating' edges (regional chain Task 6, ``AtmosphericPressure.flather_eta``),
        added to η after the tide.  None: unchanged.
        """
        if not self.open:
            return {}
        if ext is not None:
            base = ext["flather"]
        else:
            zu, zv = np.zeros(self.fs.H_u.shape), np.zeros(self.fs.H_v.shape)
            base = dict(flather_eta=(zu, zv), flather_ubar=zu, flather_vbar=zv)
        if tide is None and ib is None:
            return base
        out = dict(base)
        if tide is not None:
            eu, ev, uu, vv = tide
            out.update(flather_eta=(base["flather_eta"][0] + eu, base["flather_eta"][1] + ev),
                       flather_ubar=base["flather_ubar"] + uu, flather_vbar=base["flather_vbar"] + vv)
        if ib is not None:
            out.update(flather_eta=(out["flather_eta"][0] + ib[0], out["flather_eta"][1] + ib[1]))
        return out

    def relax_velocity(self, u_old: np.ndarray, v_old: np.ndarray, u_new: np.ndarray, v_new: np.ndarray,
                       ext: dict | None, fluxes, wu: np.ndarray | None = None,
                       wv: np.ndarray | None = None) -> np.ndarray | None:
        """
        In place: flow-relaxation sponge on the baroclinic velocity (implicit; the depth mean, and so the
        free-surface fluxes, are untouched — barotropic waves leave through Flather), then Orlanski
        radiation at the open edges.  Weights wu, wv: Δσ (None) or the ALE face fractions (spec §3.2-3.3).
        Returns, with ``obc_tangential='mom6'``, the Orlanski ratio at the tangential corners (N, ny+1, nx+1) of the
        'radiating' edges (``obc.corner_phase_speed`` of the normal r, MOM_open_boundary.F90:3031-3036; 0 at the
        other corners), the next step's ``SigmaState.obc_rtan``; else None.
        """
        ext_u, ext_v = (None, None) if ext is None else (ext["up_u"], ext["up_v"])
        if self._sp_u is not None:
            for vel, rate, w, pe in ((u_new, self._sp_u, wu, ext_u), (v_new, self._sp_v, wv, ext_v)):
                bar, prime = self.baroclinic_part(vel, w)
                if pe is None:
                    vel[...] = bar[None] + prime / (1.0 + self.dt * rate[None])
                else:                                                     # toward the exterior u′
                    r = self.dt * rate[None]
                    vel[...] = bar[None] + (prime + r * pe) / (1.0 + r)
        rq = None
        if self.tangential:
            N, ny, nx1 = u_new.shape
            rq = np.zeros((N, ny + 1, nx1))
        for name, vel, old, axis, edge, w, pe in (("west", u_new, u_old, 2, 0, wu, ext_u),
                                                  ("east", u_new, u_old, 2, -1, wu, ext_u),
                                                  ("south", v_new, v_old, 1, 0, wv, ext_v),
                                                  ("north", v_new, v_old, 1, -1, wv, ext_v)):
            if name in self.open:
                r = self.radiate(old, vel, axis, edge, w, ext=pe, mode=self.obc_mode[name], fluxes=fluxes)
                if rq is not None and self.obc_mode[name] == "radiating":
                    line = {"west": np.s_[:, :, 0], "east": np.s_[:, :, -1], "south": np.s_[:, 0, :],
                            "north": np.s_[:, -1, :]}[name]
                    rq[line] = obc.corner_phase_speed(r)
        if rq is not None:
            rq = np.where(self._tan_q[None], rq, 0.0)
        return rq

    def evaluate(self, t: float, hu: np.ndarray, hv: np.ndarray, wu, wv) -> dict | None:
        """Exterior state at t (spec 2026-10-12-baroclinic-obc-internal-wave §1.4): Flather targets and u′ₑₓₜ on
        the face layers of the open faces and sponge band; None without a provider."""
        if self.obc_state is None:
            return None
        state = self.obc_state(t)
        obc.validate_state(state, self.open, self.H.shape, salinity=self.salinity)
        Hu, Hv = self.edge_face_depths()
        wu_ = self.ds[:, None, None] * np.ones(hu.shape) if wu is None else wu
        wv_ = self.ds[:, None, None] * np.ones(hv.shape) if wv is None else wv
        code = self.remap_code
        out = dict(t=t, state=state, flather=obc.flather_targets(state, Hu.shape, Hv.shape),
                   up_u=obc.exterior_velocity(state, "u", Hu, hu, self._bc_u, wu_, code, self.open),
                   up_v=obc.exterior_velocity(state, "v", Hv, hv, self._bc_v, wv_, code, self.open),
                   own=obc.owned_by_state(state, self.open, self.H.shape) & self._bc_mask)
        if self.tangential:                                         # exterior tangential velocity on the corners
            hq = obc.corner_layers(hu, hv, self._tan_q)
            out["ut_q"] = obc.exterior_tangential({e: ob for e, ob in state.items()
                                                   if self.obc_mode.get(e) == "radiating"}, hq, self._tan_q, code)
        return out

    def tangential_velocity(self, u: np.ndarray, v: np.ndarray, rtan: np.ndarray | None,
                            ext: dict | None) -> tuple[np.ndarray, np.ndarray]:
        """
        (corner flags (ny+1, nx+1), tangential velocity at those corners (N, ny+1, nx+1) [m s⁻¹]) of the
        'radiating' edges for the momentum terms of this step (MOM6 ORLANSKI_TAN + NUDGED_TAN,
        ``obc.tangential_velocity``): radiated from the current layer velocities u, v with the corner Orlanski
        ratio ``rtan`` of the last step (``SigmaState.obc_rtan``; None = 0, a start), nudged once toward the
        exterior ``ext['ut_q']`` (0 without a provider) with τ_in = ``obc_tau_in`` when r ≤ 0 and τ_out =
        ``obc_tau_out`` otherwise (MOM_open_boundary.F90:3045-3062).
        """
        N = u.shape[0]
        shape = (N,) + self._tan_q.shape
        r = np.zeros(shape) if rtan is None else rtan
        ut = ext["ut_q"] if ext is not None and "ut_q" in ext else np.zeros(shape)
        return self._tan_q, obc.tangential_velocity(u, v, r, ut, self._tan_q, self.dt, self.obc_tau_in,
                                                    self.obc_tau_out)

    def edge_face_depths(self) -> tuple[np.ndarray, np.ndarray]:
        """Rest depth at u- and v-faces [m], 0 on land; the open edge faces (NaN in H_u/H_v, outside the
        domain) take the depth of their boundary cell."""
        H = np.nan_to_num(np.asarray(self.H, float))
        Hu, Hv = np.nan_to_num(self.fs.H_u), np.nan_to_num(self.fs.H_v)
        for edge, (A, sl_f, sl_c) in (("west", (Hu, np.s_[:, 0], np.s_[:, 0])), ("east", (Hu, np.s_[:, -1], np.s_[:, -1])),
                                      ("south", (Hv, np.s_[0, :], np.s_[0, :])), ("north", (Hv, np.s_[-1, :], np.s_[-1, :]))):
            if edge in self.open:
                A[sl_f] = H[sl_c]
        return Hu, Hv

    def exterior_T(self, h: np.ndarray, ext: dict | None) -> np.ndarray:
        """Tₑₓₜ on layers h in the boundary/sponge columns: from the provider when set, else the z profile."""
        rest = self.exterior(h) if self.ale else self.T_ext
        if ext is None:
            return rest
        T = obc.exterior_tracer(ext["state"], self.H, h, self._ext_cols, self.remap_code,
                                self.open)
        return np.where(ext["own"][None], T, rest)              # edges without data: resting exterior

    def exterior(self, h: np.ndarray) -> np.ndarray:
        """
        Exterior T profile remapped onto thicknesses h in the boundary/sponge columns (core.vertical.ale).
        ``T_ext`` and ``h_ext`` go together (both set by ``initial_state``); replace them together.
        """
        if self.h_ext is None or self.T_ext is None or np.shape(self.h_ext) != np.shape(self.T_ext) \
                or np.shape(self.h_ext) != h.shape:
            raise ValueError("ALE open boundaries need the exterior profile T_ext and h_ext, set together and "
                             f"with the layer shape {h.shape} (call initial_state, or set both)")
        numba.set_num_threads(threads._THREADS[0])                                 # parallel remap kernel
        return exterior_on_layers(self.h_ext, self.T_ext, h, self._ext_cols, self.remap_code)

    def exterior_S(self, h: np.ndarray, ext: dict | None) -> np.ndarray:
        """
        Sₑₓₜ [g kg⁻¹] on layers h in the boundary/sponge columns, as ``exterior_T``: the provider's OBData.S where
        it owns the column, else the resting exterior ``S_ext`` (σ: as is; ALE: kept in z on ``h_ext`` and
        remapped onto h).
        """
        if self.S_ext is None:
            raise ValueError("salinity at open boundaries needs the exterior profile S_ext (call initial_state)")
        if self.ale:
            numba.set_num_threads(threads._THREADS[0])                             # parallel remap kernel
            rest = exterior_on_layers(self.h_ext, self.S_ext, h, self._ext_cols, self.remap_code)
        else:
            rest = self.S_ext
        if ext is None:
            return rest
        S = obc.exterior_tracer(ext["state"], self.H, h, self._ext_cols, self.remap_code, self.open, field="S")
        return np.where(ext["own"][None], S, rest)

    def exterior_tr(self, h: np.ndarray, ext: dict | None) -> np.ndarray:
        """Exterior passive tracers (n_tracers, N, ny, nx) on layers h in the boundary/sponge columns: the
        provider's OBData.tracers where it owns the column (0 when it gives none), else the resting tr_ext in z."""
        numba.set_num_threads(threads._THREADS[0])
        rest = np.stack([exterior_on_layers(self.h_ext, f, h, self._ext_cols, self.remap_code) for f in self.tr_ext])
        if ext is None:
            return rest
        st, code = ext["state"], self.remap_code
        out = np.stack([obc.exterior_tracer(st, self.H, h, self._ext_cols, code, self.open, field=i)
                        for i in range(self.n_tracers)])
        return np.where(ext["own"][None, None], out, rest)

    def open_face_transports(self, Qx: np.ndarray, Qy: np.ndarray, h: np.ndarray, un: np.ndarray, vn: np.ndarray,
                              u_new: np.ndarray, v_new: np.ndarray, fluxes) -> None:
        """
        In place: layer fluxes through the open faces with the interior thickness, adding up to the
        Flather flux of the free surface (spec §3.4; core.vertical.ale.open_face_transport).
        """
        th = self.fs.theta
        for edge, Q, D, Qo, L, hc, vo, vn1 in (
                ("west", Qx[..., 0], fluxes.open_Du[:, 0], fluxes.open_Qu[:, 0], self.fs.Lu[:, 0],
                 h[..., 0], un[..., 0], u_new[..., 0]),
                ("east", Qx[..., -1], fluxes.open_Du[:, -1], fluxes.open_Qu[:, -1], self.fs.Lu[:, -1],
                 h[..., -1], un[..., -1], u_new[..., -1]),
                ("south", Qy[:, 0, :], fluxes.open_Dv[0, :], fluxes.open_Qv[0, :], self.fs.Lv[0, :],
                 h[:, 0, :], vn[:, 0, :], v_new[:, 0, :]),
                ("north", Qy[:, -1, :], fluxes.open_Dv[-1, :], fluxes.open_Qv[-1, :], self.fs.Lv[-1, :],
                 h[:, -1, :], vn[:, -1, :], v_new[:, -1, :])):
            if edge in self.open:
                Q[...] = np.where((D > 0)[None], open_face_transport(hc, vn1, vo, Qo, L, th, self.theta_bc), Q)

    def inflow_stack(self, c: np.ndarray, T_in: np.ndarray | None, tr_in: np.ndarray | None,
                     S_in: np.ndarray | None = None, layout: dict[str, int] | None = None) -> np.ndarray:
        """
        Inflow values of the advected stack on open edges: T_in (and T_in² when the stack carries the mixing
        diagnostic) followed by the passive tracers' inflow values; ``c`` itself without open edges or exterior.
        ``layout`` (``core.model.driver._stack_layout``, N2a): the positions of T, T², S, S² and the tracers; the
        salinity stack ``[T, (T²), S, (S²), tr…]`` takes S_in (and S_in²) at its slots.  Without ``layout`` the
        stack is ``[T, (T²), tr…]`` as before N2a.
        """
        if T_in is None or not self.open:
            return c
        if layout is not None:
            out: list = [None] * layout["n"]
            out[layout["T"]] = T_in
            if "T2" in layout:
                out[layout["T2"]] = T_in**2
            if "S" in layout:
                out[layout["S"]] = S_in
            if "S2" in layout:
                out[layout["S2"]] = S_in**2
            if "tr" in layout:
                out[layout["tr"]:] = list(tr_in)
            return np.ascontiguousarray(np.stack(out))
        lead = c.shape[0] - (self.n_tracers if self.ale else 0)        # [T] or [T, T²] before the tracers
        parts = [T_in, T_in**2][:lead]
        if self.ale and self.n_tracers:
            parts += list(tr_in)
        return np.ascontiguousarray(np.stack(parts))
