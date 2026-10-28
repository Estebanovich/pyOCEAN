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
4. "hviscosity"       Laplacian viscosity A_h ∇²u along layers
5. "vadvection"       σ vertical advection (ω/h) ∂u/∂k from the last step's ω (subtracted; advective form only)

This is the pattern of MOM6 (the stepper sums ``(CAu + PFu) + diffu`` in a fixed order,
MOM_dynamics_split_RK2.F90:575-581) and Oceananigans (one function per term, summed by the tendency kernel;
docs/design/n1_refs/).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np

from core.dynamics import vector_invariant as vi
from core.dynamics.nonlinear import _spacings_u, _spacings_v, advection_u, advection_v, laplacian_3d
from core.model import threads
from core.model.contract import BuildContext, Issue, Role

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


class Momentum:
    """
    Explicit momentum terms.  ``form`` 'flux' (flux-form or advective-form advection plus energy-conserving
    Coriolis) or 'vector_invariant' (ALE only).  ``adv`` 'flux', 'upwind3' or 'centred' (flux form only),
    ``ah`` [m² s⁻¹] Laplacian viscosity, ``ke_scheme`` 'arakawa' or 'gudonov' (vector-invariant KE).
    """
    name = "momentum"
    role = Role.MOMENTUM
    reads = ("u", "v", "eta", "T", "h", "Qx", "Qy", "Fz", "omega")
    writes = ()

    def __init__(self, *, form: str, adv: str, ah: float, ke_scheme: str, ale: bool, dt: float, hg, fq: np.ndarray,
                 fs, ds: np.ndarray, vi_geom: dict | None, mean_face_thickness: Callable,
                 layer_volumes: Callable) -> None:
        self.form, self.adv, self.ah, self.ke_scheme, self.ale, self.dt = form, adv, ah, ke_scheme, ale, dt
        self.hg, self.fq, self.fs, self.ds, self.vi_geom = hg, fq, fs, ds, vi_geom
        self.mean_face_thickness, self.layer_volumes = mean_face_thickness, layer_volumes
        self._lap = None

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}

    def coriolis(self, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Energy-conserving C-grid Coriolis terms (+f v̄ at u-faces, −f ū at v-faces):
        every adjacent (u, v) pair shares one cell corner q and is weighted by ¼ f_q,
        so the operator is antisymmetric and does no work (Sadourny 1975; Arakawa &
        Lamb 1977).  u (N, ny, nx+1), v (N, ny+1, nx).
        """
        fq = self.fq
        cu = np.zeros_like(u)
        cv = np.zeros_like(v)
        cu[..., 1:-1] = 0.25 * (fq[:-1, 1:-1] * (v[:, :-1, :-1] + v[:, :-1, 1:])
                                + fq[1:, 1:-1] * (v[:, 1:, :-1] + v[:, 1:, 1:]))
        cv[:, 1:-1, :] = -0.25 * (fq[1:-1, :-1] * (u[:, :-1, :-1] + u[:, 1:, :-1])
                                  + fq[1:-1, 1:] * (u[:, :-1, 1:] + u[:, 1:, 1:]))
        return cu, cv

    def vector_invariant(self, st: SigmaState) -> tuple[np.ndarray, np.ndarray]:
        """
        Vector-invariant momentum terms of the ALE step (spec 2026-10-12 §3): Sadourny (1975) energy
        PV flux q·(vh, −uh) with BOUND_CORIOLIS, minus ∇K (core.dynamics.vector_invariant).  The
        transports are those of the last Lagrangian step (as MOM6 uses the uh of its last continuity);
        the first step uses L u h̄.
        """
        g = self.vi_geom
        numba.set_num_threads(threads._THREADS[0])                                 # vector-invariant kernels
        zeta, q = vi.vorticity_and_pv(st.u, st.v, g["dx_u"], g["dy_v"], g["area_q"], g["mask_q"], self.fq, st.h,
                                      g["area_t"], self.fs.active)
        Qx, Qy = st.Qx, st.Qy
        if Qx is None or Qy is None:
            hu, hv = self.mean_face_thickness(st.h)
            Qx, Qy = self.fs.Lu[None] * st.u * hu, self.fs.Lv[None] * st.v * hv
        cu, cv = vi.coriolis_pv(q, self.fq[None] + zeta, Qx, Qy, st.u, st.v, g["dx_u"], g["dy_v"], bound=True)
        K = vi.kinetic_energy(st.u, st.v, g["area_u"], g["area_v"], g["area_t"], self.ke_scheme)
        kx, ky = vi.ke_gradient(K, g["dx_u"], g["dy_v"])
        return cu + kx, cv + ky

    def vertical_advection(self, vel: np.ndarray, omega_f: np.ndarray, h_f: np.ndarray) -> np.ndarray:
        """(ω/h) ∂u/∂k, centred: [ω_{k+½}(u_{k+1}−u_k) + ω_{k−½}(u_k−u_{k−1})] / (2h)."""
        out = np.zeros_like(vel)
        du = vel[1:] - vel[:-1]                                             # at interior interfaces
        out[:-1] += omega_f[1:-1] * du
        out[1:] += omega_f[1:-1] * du
        return out / (2.0 * np.maximum(h_f, 1e-6))

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
        if self.adv == "flux" and not vinv:
            if st.Qx is not None:
                from core.dynamics.sigma_baroclinic import _flux_form_u
                V = self.layer_volumes(st.eta, st.h)
                lim = self.dt if self.ale else 0.0                                      # vanished-cell limiter
                au = _flux_form_u(st.u, st.Qx, st.Qy, st.Fz, V, wet_u[None], lim)
                swap = lambda a: np.swapaxes(a, 1, 2)                                   # noqa: E731
                av = swap(_flux_form_u(swap(st.v), swap(st.Qy), swap(st.Qx), swap(st.Fz), swap(V),
                                       swap(wet_v[None]), lim))
                out.append(("advection", au, av, "+"))
        if self.adv != "flux" and not vinv:
            au = np.empty_like(st.u)
            av = np.empty_like(st.v)
            for k in range(n):
                au[k] = advection_u(st.u[k], st.v[k], self.hg, wet_u, scheme=self.adv)
                av[k] = advection_v(st.u[k], st.v[k], self.hg, wet_v, scheme=self.adv)
            out.append(("advection", au, av, "-"))
        if self.ah > 0:                                                 # spacings built once (performance task 5.2)
            if self._lap is None:
                self._lap = (_spacings_u(self.hg), _spacings_v(self.hg))
            numba.set_num_threads(threads._THREADS[0])
            out.append(("hviscosity", self.ah * laplacian_3d(st.u, wet_u, *self._lap[0]),
                        self.ah * laplacian_3d(st.v, wet_v, *self._lap[1]), "+"))
        if st.omega is not None and self.adv != "flux":
            om_u = np.zeros((n + 1,) + Du.shape)
            om_u[:, :, 1:-1] = 0.5 * (st.omega[:, :, 1:] + st.omega[:, :, :-1])
            om_v = np.zeros((n + 1,) + Dv.shape)
            om_v[:, 1:-1, :] = 0.5 * (st.omega[:, 1:, :] + st.omega[:, :-1, :])
            out.append(("vadvection",
                        self.vertical_advection(st.u, om_u, np.nan_to_num(Du)[None] * self.ds[:, None, None]),
                        self.vertical_advection(st.v, om_v, np.nan_to_num(Dv)[None] * self.ds[:, None, None]), "-"))
        return out
