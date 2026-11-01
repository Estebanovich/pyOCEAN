"""
Explicit momentum terms of the 3-D core (N1, role MOMENTUM), moved unchanged from ``SigmaBaroclinic._tendencies``,
``_coriolis``, ``_vector_invariant`` and ``_vertical_advection``.

``Momentum.terms`` returns the explicit tendencies at u- and v-faces [m s⁻²] as named terms, in the order in which
``sum_terms`` accumulates them (the order of the code before N1, which fixes the floating-point result):

1. "pgf"              baroclinic pressure-gradient force (``core.components.pgf``)
2. "coriolis"         energy-conserving C-grid Coriolis (Sadourny 1975; Arakawa & Lamb 1977), or
   "vector_invariant" PV flux q·(vh, −uh) with BOUND_CORIOLIS plus −∇K (Sadourny 1975; MOM6 MOM_CoriolisAdv.F90)
3. "advection"        flux-form momentum advection on the last step's layer transports (added), or the
                      advective form 'upwind3' / 'centred' (subtracted)
4. "hviscosity"       Laplacian viscosity A_h ∇²u along layers, or (``smagorinsky`` > 0) the thickness-weighted
                      Laplacian Smagorinsky stress divergence with ν = max(A_h, C Δ² |S|) (``core.dynamics.hor_visc``;
                      Smagorinsky 1963, Griffies & Hallberg 2000, MOM6 MOM_hor_visc.F90)
5. "vadvection"       σ vertical advection (ω/h) ∂u/∂k from the last step's ω (subtracted; advective form only)

This is the pattern of MOM6 (the stepper sums ``(CAu + PFu) + diffu`` in a fixed order,
MOM_dynamics_split_RK2.F90:575-581) and Oceananigans (one function per term, summed by the tendency kernel;
docs/design/n1_refs/).  Index order (k, j, i), k = 0 at the bed.

Periodic axes (N3 task 3, ``per``; ``core.model.periodic``): every term is computed on inputs padded with wrapped
halos and cropped (pad–crop), except the σ vertical advection, whose ω at the wrap faces is averaged natively.
"""
from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np

from core.dynamics import vector_invariant as vi
from core.dynamics.hor_visc import smag_geometry, smagorinsky
from core.dynamics.nonlinear import _spacings_u, _spacings_v, advection_u, advection_v, laplacian_3d
from core.model import threads
from core.model.contract import BuildContext, Issue, Role
from core.model.periodic import MOMENTUM_HALO, NONE
from core.vertical.ale import face_transport_thickness

Term = tuple[str, np.ndarray, np.ndarray, str]


def sum_terms(terms: list[Term], wet_u: np.ndarray, wet_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    G = Σ ± term in the given order (first term copied, the others added or subtracted in place), then zero on
    dry faces.  Reproduces the in-place accumulation of ``SigmaBaroclinic._tendencies`` bit for bit.
    """
    gu, gv = terms[0][1].copy(), terms[0][2].copy()
    for _, tu, tv, sign in terms[1:]:
        if sign == "+":
            gu += tu
            gv += tv
        else:
            gu -= tu
            gv -= tv
    return np.where(wet_u[None], gu, 0.0), np.where(wet_v[None], gv, 0.0)


def _flux_advection(u, v, Qx, Qy, Fz, V, wet_u, wet_v, lim: float = 0.0) -> tuple[np.ndarray, np.ndarray]:
    """Flux-form momentum advection at u- and v-faces (``_flux_form_u`` on u, and on v with x and y swapped)."""
    from core.dynamics.sigma_baroclinic import _flux_form_u
    au = _flux_form_u(u, Qx, Qy, Fz, V, wet_u[None], lim)
    swap = lambda a: np.swapaxes(a, 1, 2)                                       # noqa: E731
    av = swap(_flux_form_u(swap(v), swap(Qy), swap(Qx), swap(Fz), swap(V), swap(wet_v[None]), lim))
    return au, av


def _coriolis(u: np.ndarray, v: np.ndarray, fq: np.ndarray, dx_u: np.ndarray | None = None,
              dy_u: np.ndarray | None = None, dx_v: np.ndarray | None = None,
              dy_v: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
    """
    The energy-conserving C-grid Coriolis terms of ``Momentum.coriolis`` (interior faces; edge faces 0).

    Without metrics (a grid whose face lengths equal the centre distances, Δx_v = Δx_u and Δy_u = Δy_v, e.g. a
    uniform Cartesian grid) the unweighted mean ¼ Σ f_q v.  With metrics, the Sadourny (1975) energy form of MOM6
    with a uniform thickness (MOM_CoriolisAdv.F90:818-823, 1071-1076):

        C_u = ¼/Δx_u Σ₄ f_q (v Δx_v),        C_v = −¼/Δy_v Σ₄ f_q (u Δy_u)

    so that Σ A_u u C_u + Σ A_v v C_v = 0 with A_u = Δx_u Δy_u, A_v = Δx_v Δy_v (MOM6 areaCu, areaCv) on any
    orthogonal grid, and the discrete geostrophic state is steady (tests/test_curvilinear_invariance.py).
    """
    cu = np.zeros_like(u)
    cv = np.zeros_like(v)
    if dx_u is None:
        cu[..., 1:-1] = 0.25 * (fq[:-1, 1:-1] * (v[:, :-1, :-1] + v[:, :-1, 1:])
                                + fq[1:, 1:-1] * (v[:, 1:, :-1] + v[:, 1:, 1:]))
        cv[:, 1:-1, :] = -0.25 * (fq[1:-1, :-1] * (u[:, :-1, :-1] + u[:, 1:, :-1])
                                  + fq[1:-1, 1:] * (u[:, :-1, 1:] + u[:, 1:, 1:]))
        return cu, cv
    vt = v * dx_v[None]                                                 # v Δx_v (per unit thickness)
    ut = u * dy_u[None]                                                 # u Δy_u
    cu[..., 1:-1] = 0.25 * ((fq[1:, 1:-1] * (vt[:, 1:, 1:] + vt[:, 1:, :-1]))
                            + (fq[:-1, 1:-1] * (vt[:, :-1, :-1] + vt[:, :-1, 1:]))) / dx_u[None, :, 1:-1]
    cv[:, 1:-1, :] = -0.25 * ((fq[1:-1, 1:] * (ut[:, 1:, 1:] + ut[:, :-1, 1:]))
                              + (fq[1:-1, :-1] * (ut[:, :-1, :-1] + ut[:, 1:, :-1]))) / dy_v[None, 1:-1, :]
    return cu, cv


def coriolis_metrics(hg) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray] | None:
    """
    (Δx_u, Δy_u, Δx_v, Δy_v) for the metric-weighted Coriolis of ``_coriolis``, or None when every face length
    equals the centre distance across the neighbouring faces (Δx_v[J, i] = Δx_u[j, I] and Δy_u[j, I] = Δy_v[J, i]
    for the four (u, v) pairs of each corner), where the weights are exactly 1 and the unweighted form is used,
    bit for bit as before (uniform Cartesian grids).  Stretched, spherical and curvilinear grids get the weights
    (metric audit, docs/design/regional_refs/metric_audit.md).
    """
    dx_u, dy_u, dx_v, dy_v = (np.asarray(hg[k].values, float) for k in ("dx_u", "dy_u", "dx_v", "dy_v"))
    same = True
    for a in (dx_v[:-1], dx_v[1:]):                                     # the v-faces south and north of row j
        same &= bool(np.array_equal(a, dx_u[:, :-1]) and np.array_equal(a, dx_u[:, 1:]))
    for a in (dy_u[:, :-1], dy_u[:, 1:]):                               # the u-faces west and east of column i
        same &= bool(np.array_equal(a, dy_v[:-1]) and np.array_equal(a, dy_v[1:]))
    return None if same else (dx_u, dy_u, dx_v, dy_v)


class Momentum:
    """
    Explicit momentum terms.  ``form`` 'flux' (flux-form or advective-form advection plus energy-conserving
    Coriolis) or 'vector_invariant' (ALE only).  ``adv`` 'flux', 'upwind3' or 'centred' (flux form only),
    ``ah`` [m² s⁻¹] Laplacian viscosity, ``ke_scheme`` 'arakawa' or 'gudonov' (vector-invariant KE).
    ``smagorinsky`` C [-] (0 = off): Smagorinsky viscosity in place of the constant Laplacian, ``ah`` its
    background; ``open_u``/``open_v`` (ny, nx+1)/(ny+1, nx) bool, the open-boundary faces (no viscous tendency,
    free-slip corners).  After each call of ``terms`` with C > 0: ``last_nu_t`` (N, ny, nx) ν at h points
    [m² s⁻¹] and ``last_n_limited`` (points where the stability bound reduced ν this step).
    """
    name = "momentum"
    role = Role.MOMENTUM
    reads = ("u", "v", "eta", "T", "h", "Qx", "Qy", "Fz", "omega")
    writes = ()
    per = NONE                       # periodic axes (core.model.periodic), set by the model

    def __init__(self, *, form: str, adv: str, ah: float, ke_scheme: str, ale: bool, dt: float, hg, fq: np.ndarray,
                 fs, ds: np.ndarray, vi_geom: dict | None, mean_face_thickness: Callable,
                 layer_volumes: Callable, smagorinsky: float = 0.0, open_u: np.ndarray | None = None,
                 open_v: np.ndarray | None = None) -> None:
        self.form, self.adv, self.ah, self.ke_scheme, self.ale, self.dt = form, adv, ah, ke_scheme, ale, dt
        self.hg, self.fq, self.fs, self.ds, self.vi_geom = hg, fq, fs, ds, vi_geom
        self.mean_face_thickness, self.layer_volumes = mean_face_thickness, layer_volumes
        self._lap = None
        self.smagorinsky = float(smagorinsky)
        ny, nx = fs.H.shape
        self.open_u = np.zeros((ny, nx + 1), bool) if open_u is None else np.asarray(open_u, bool)
        self.open_v = np.zeros((ny + 1, nx), bool) if open_v is None else np.asarray(open_v, bool)
        self._smag_geom = None
        self._cor_metrics = coriolis_metrics(hg)                        # None: unweighted (uniform grid)
        self.last_nu_t: np.ndarray | None = None
        self.last_n_limited = 0

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}

    def coriolis(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Energy-conserving C-grid Coriolis terms (+f v̄ at u-faces, −f ū at v-faces):
        every adjacent (u, v) pair shares one cell corner q and is weighted by ¼ f_q (times the face-length ratio
        Δx_v/Δx_u or Δy_u/Δy_v on a non-uniform grid, ``_coriolis``), so the operator is antisymmetric and does no
        work (Sadourny 1975; Arakawa & Lamb 1977; MOM_CoriolisAdv.F90:818-823).  u (N, ny, nx+1), v (N, ny+1, nx).
        """
        met = self._cor_metrics
        if self.per:
            if met is None:
                return self.per.apply(_coriolis, (u, v, self.fq), ("u", "v", "q"), ("u", "v"), 1)
            return self.per.apply(_coriolis, (u, v, self.fq) + met, ("u", "v", "q", "u", "u", "v", "v"),
                                  ("u", "v"), 1)
        if met is None:
            return _coriolis(u, v, self.fq)
        return _coriolis(u, v, self.fq, *met)

    def _padded_vi_geometry(self) -> dict:
        """Vector-invariant geometry with periodic halos of 2 (corner area and mask rebuilt on the padded grid)."""
        if getattr(self, "_vi_pad", None) is None:
            p, g, H = self.per, self.vi_geom, 2
            out = {k: p.pad(g[k], kind, H) for k, kind in (("dx_u", "u"), ("dy_v", "v"), ("area_t", "c"),
                                                           ("area_u", "u"), ("area_v", "v"))}
            out["active"] = p.pad(self.fs.active, "c", H)
            out["area_q"] = (p.pad(g["area_q"], "q", H) if "area_q" in self.hg     # the grid's areaBu, wrapped
                             else vi.corner_area(out["dx_u"], out["dy_v"]))
            out["mask_q"] = vi.corner_mask(out["active"])
            out["fq"] = p.pad(self.fq, "q", H)
            self._vi_pad = out
        return self._vi_pad

    def vector_invariant(self, st: SigmaState) -> tuple[np.ndarray, np.ndarray]:
        """
        Vector-invariant momentum terms of the ALE step (spec 2026-10-12 §3): Sadourny (1975) energy
        PV flux q·(vh, −uh) with BOUND_CORIOLIS, minus ∇K (core.dynamics.vector_invariant).  The
        transports are those of the last Lagrangian step (as MOM6 uses the uh of its last continuity);
        the first step uses L u h̄.  Periodic: on halos of 2 (ζ and q at corners, then the faces).
        """
        Qx, Qy = st.Qx, st.Qy
        if Qx is None or Qy is None:
            hu, hv = self.mean_face_thickness(st.h)
            Qx, Qy = self.fs.Lu[None] * st.u * hu, self.fs.Lv[None] * st.v * hv
        if self.per:
            g = self._padded_vi_geometry()
            return self.per.apply(lambda u, v, h, qx, qy: self._vector_invariant(u, v, h, qx, qy, g, g["fq"],
                                                                                 g["active"]),
                                  (st.u, st.v, st.h, Qx, Qy), ("u", "v", "c", "u", "v"), ("u", "v"), 2)
        return self._vector_invariant(st.u, st.v, st.h, Qx, Qy, self.vi_geom, self.fq, self.fs.active)

    def _vector_invariant(self, u, v, h, Qx, Qy, g: dict, fq: np.ndarray, active: np.ndarray):
        """The terms of ``vector_invariant`` on the given arrays and geometry (unchanged arithmetic)."""
        numba.set_num_threads(threads._THREADS[0])                                 # vector-invariant kernels
        zeta, q = vi.vorticity_and_pv(u, v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"], fq, h, g["area_t"],
                                      active)
        cu, cv = vi.coriolis_pv(q, fq[None] + zeta, Qx, Qy, u, v, g["dx_u"], g["dy_v"], bound=True)
        ke = "gudonov" if self.ke_scheme == "godunov" else self.ke_scheme              # kernel spelling
        K = vi.kinetic_energy(u, v, g["area_u"], g["area_v"], g["area_t"], ke)
        kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
        return cu + kx, cv + ky

    def vertical_advection(self, vel: np.ndarray, omega_f: np.ndarray, h_f: np.ndarray) -> np.ndarray:
        """(ω/h) ∂u/∂k, centred: [ω_{k+½}(u_{k+1}−u_k) + ω_{k−½}(u_k−u_{k−1})] / (2h)."""
        out = np.zeros_like(vel)
        du = vel[1:] - vel[:-1]                                             # at interior interfaces
        out[:-1] += omega_f[1:-1] * du
        out[1:] += omega_f[1:-1] * du
        return out / (2.0 * np.maximum(h_f, 1e-6))

    def _unclipped_face_thickness(self, h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Unclipped land-masked mean layer thickness ½(h_L + h_R) at u- and v-faces [m] (MOM6 use_land_mask,
        MOM_hor_visc.F90:769-782): not cut at the shallower bed as the transport thickness of z*/hybrid is, so a
        layer below the shallower bed of a step is a wall for that layer, not a vanished layer (final review I1).
        Faces next to land are 0, domain-edge faces take the inner cell; periodic axes wrap (halo 1).
        """
        act, sw = self.fs.active, (lambda a: a.swapaxes(1, 2))         # noqa: E731
        H0 = np.zeros(act.shape)                                       # depth enters only the clipping
        if self.per:
            p = self.per
            hp, ap, Hp = p.pad(h, "c", 1), p.pad(act, "c", 1), p.pad(H0, "c", 1)
            hu = 0.5 * sum(face_transport_thickness(hp, Hp, ap, False))
            hv = 0.5 * sum(face_transport_thickness(sw(hp), Hp.T, ap.T, False))
            return p.crop(hu, "u", 1), p.crop(sw(hv), "v", 1)
        hu = 0.5 * sum(face_transport_thickness(h, H0, act, False))
        hv = 0.5 * sum(face_transport_thickness(sw(h), H0.T, act.T, False))
        return hu, sw(hv)

    def _smag_thickness(self, st, Du: np.ndarray, Dv: np.ndarray, wet_u: np.ndarray,
                        wet_v: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Layer thickness at h points and u-, v-faces [m] for the Smagorinsky operator: σ h = D Δσ_k with the cell
        total depth D = max(H + η, 0) and the face depths Du, Dv of the step; ALE h_t = h and the faces the
        unclipped mean ½(h_L + h_R) of the two cells at wet faces, 0 at land/dry faces (MOM6
        MOM_hor_visc.F90:769-782; not the bed-clipped ``mean_face_thickness``, whose ~0 faces below the shallower
        bed of a z* step zeroed ν and the background ah in the deep cell, final review I1).
        """
        if self.ale:
            hu, hv = self._unclipped_face_thickness(st.h)
            return st.h, np.where(wet_u[None], hu, 0.0), np.where(wet_v[None], hv, 0.0)
        ds = self.ds[:, None, None]
        D = np.maximum(np.nan_to_num(self.fs.H + st.eta, nan=0.0), 0.0)
        return (D[None] * ds, np.maximum(np.nan_to_num(Du, nan=0.0), 0.0)[None] * ds,
                np.maximum(np.nan_to_num(Dv, nan=0.0), 0.0)[None] * ds)

    def _smag_call(self, u, v, h_t, h_u, h_v, wet_u, wet_v, open_u, open_v, geom: dict):
        """
        ``hor_visc.smagorinsky`` on the given (possibly padded) arrays.  The corner mask is derived here, the only
        place: a corner is active when its four faces (u south/north, v west/east) are wet this step, so it follows
        wetting and drying (free slip, D_S = 0, at every other corner; MOM_hor_visc.F90:927).
        """
        mask_q = np.zeros((wet_v.shape[0], wet_u.shape[1]), bool)
        mask_q[1:-1, 1:-1] = wet_u[:-1, 1:-1] & wet_u[1:, 1:-1] & wet_v[1:-1, :-1] & wet_v[1:-1, 1:]
        fu, fv, nu_t, _, n_lim = smagorinsky(u, v, h_t, h_u, h_v, wet_u, wet_v, mask_q, open_u, open_v, geom,
                                             self.smagorinsky, self.ah, self.dt)
        self.last_n_limited = n_lim
        return fu, fv, nu_t

    def smagorinsky_terms(self, st, Du: np.ndarray, Dv: np.ndarray, wet_u: np.ndarray,
                          wet_v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Smagorinsky viscous tendencies at u- and v-faces [m s⁻²] of the state ``st`` (layer velocities at tⁿ,
        explicit in the AB3 tendency): F = ∇·(ν h D)/h, ν = max(A_h, C Δ² |S|) bounded for AB3
        (``core.dynamics.hor_visc``; Smagorinsky 1963, Griffies & Hallberg 2000, MOM_hor_visc.F90).  Periodic axes:
        pad–crop with halos of 2 (the stencil reaches the tension of the neighbour cell through the corner shear),
        geometry padded once.  Sets ``last_nu_t`` and ``last_n_limited``.
        """
        h_t, h_u, h_v = self._smag_thickness(st, Du, Dv, wet_u, wet_v)
        per = self.per
        if per:
            H = 2
            if self._smag_geom is None:
                self._smag_geom = smag_geometry(per.pad_dataset(self.hg, H))
            fu, fv, nu_t = per.apply(lambda *a: self._smag_call(*a, self._smag_geom),
                                     (st.u, st.v, h_t, h_u, h_v, wet_u, wet_v, self.open_u, self.open_v),
                                     ("u", "v", "c", "u", "v", "u", "v", "u", "v"), ("u", "v", "c"), H)
        else:
            if self._smag_geom is None:
                self._smag_geom = smag_geometry(self.hg)
            fu, fv, nu_t = self._smag_call(st.u, st.v, h_t, h_u, h_v, wet_u, wet_v, self.open_u, self.open_v,
                                           self._smag_geom)
        self.last_nu_t = nu_t
        return fu, fv

    def terms(self, st, pgf_xy: tuple[np.ndarray, np.ndarray], Du: np.ndarray, Dv: np.ndarray,
              wet_u: np.ndarray, wet_v: np.ndarray) -> list[Term]:
        """Named explicit terms in summation order (module docstring); ``pgf_xy`` from the PGF component."""
        n = st.u.shape[0]
        out: list[Term] = [("pgf", pgf_xy[0], pgf_xy[1], "+")]
        vinv = self.form == "vector_invariant"
        if vinv:
            cu, cv = self.vector_invariant(st)
            out.append(("vector_invariant", cu, cv, "+"))
        else:
            cu, cv = self.coriolis(st.u, st.v)
            out.append(("coriolis", cu, cv, "+"))
        per = self.per
        if self.adv == "flux" and not vinv:
            if st.Qx is not None:
                V = self.layer_volumes(st.eta, st.h)
                lim = self.dt if self.ale else 0.0                                      # vanished-cell limiter
                args = (st.u, st.v, st.Qx, st.Qy, st.Fz, V, wet_u, wet_v)
                if per:
                    au, av = per.apply(_flux_advection, args, ("u", "v", "u", "v", "c", "c", "u", "v"), ("u", "v"),
                                       MOMENTUM_HALO, lim=lim)
                else:
                    au, av = _flux_advection(*args, lim=lim)
                out.append(("advection", au, av, "+"))
        if self.adv != "flux" and not vinv:
            hg, H = self.hg, 2
            if per:                                                     # advective form on halos of 2
                if getattr(self, "_hg_pad", None) is None:
                    self._hg_pad = per.pad_dataset(self.hg, H)
                hg = self._hg_pad
            u_, v_, wu_, wv_ = st.u, st.v, wet_u, wet_v
            if per:
                u_, v_, wu_, wv_ = per.pad(st.u, "u", H), per.pad(st.v, "v", H), per.pad(wet_u, "u", H), \
                    per.pad(wet_v, "v", H)
            au = np.empty_like(u_)
            av = np.empty_like(v_)
            for k in range(n):
                au[k] = advection_u(u_[k], v_[k], hg, wu_, scheme=self.adv)
                av[k] = advection_v(u_[k], v_[k], hg, wv_, scheme=self.adv)
            if per:
                au, av = per.crop(au, "u", H), per.crop(av, "v", H)
            out.append(("advection", au, av, "-"))
        if self.smagorinsky > 0:                                        # Smagorinsky stress divergence (ah = floor)
            fu, fv = self.smagorinsky_terms(st, Du, Dv, wet_u, wet_v)
            out.append(("hviscosity", fu, fv, "+"))
        elif self.ah > 0:                                               # spacings built once (performance task 5.2)
            if self._lap is None:
                hg = per.pad_dataset(self.hg, 1) if per else self.hg   # periodic: spacings on halos of 1
                self._lap = (_spacings_u(hg), _spacings_v(hg))
            numba.set_num_threads(threads._THREADS[0])
            if per:
                lu = per.crop(laplacian_3d(per.pad(st.u, "u", 1), per.pad(wet_u, "u", 1), *self._lap[0]), "u", 1)
                lv = per.crop(laplacian_3d(per.pad(st.v, "v", 1), per.pad(wet_v, "v", 1), *self._lap[1]), "v", 1)
                out.append(("hviscosity", self.ah * lu, self.ah * lv, "+"))
            else:
                out.append(("hviscosity", self.ah * laplacian_3d(st.u, wet_u, *self._lap[0]),
                            self.ah * laplacian_3d(st.v, wet_v, *self._lap[1]), "+"))
        if st.omega is not None and self.adv != "flux":
            om_u = np.zeros((n + 1,) + Du.shape)
            om_u[:, :, 1:-1] = 0.5 * (st.omega[:, :, 1:] + st.omega[:, :, :-1])
            om_v = np.zeros((n + 1,) + Dv.shape)
            om_v[:, 1:-1, :] = 0.5 * (st.omega[:, 1:, :] + st.omega[:, :-1, :])
            if per.x:                                                   # ω at the wrap faces
                om_u[:, :, 0] = om_u[:, :, -1] = 0.5 * (st.omega[:, :, 0] + st.omega[:, :, -1])
            if per.y:
                om_v[:, 0, :] = om_v[:, -1, :] = 0.5 * (st.omega[:, 0, :] + st.omega[:, -1, :])
            out.append(("vadvection",
                        self.vertical_advection(st.u, om_u, np.nan_to_num(Du)[None] * self.ds[:, None, None]),
                        self.vertical_advection(st.v, om_v, np.nan_to_num(Dv)[None] * self.ds[:, None, None]), "-"))
        return out
