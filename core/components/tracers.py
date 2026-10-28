"""
Tracer advection and horizontal diffusion of the 3-D core (N1, roles TRACER_ADVECTION and TRACER_DIFFUSION),
moved unchanged from ``SigmaBaroclinic``.

``TracerAdvection``: flux-form advection of a stack of tracers (T, T² for the mixing diagnostic, passive tracers)
with frozen layer transports, layer volumes going linearly from V0 to V1:

* SSP-RK3 (Shu & Osher 1988) with WENO5-Z/JS (Borges et al. 2008; Jiang & Shu 1996) or upwind-3/1 faces, in
  fused Numba kernels (``_rk_stage``, ``_advect_div``) or the NumPy reference path;
* 'pc': predictor with centred faces, corrector with upwind faces (CROCO step3d_t.F);
* ALE with ``limiter='fct'``: Zalesak (1979) FCT around the three RK3 stages (HYCOM mod_tsadvc.F90:877-905).

Open edges take the upstream value (Marsaleix et al. 2006): the boundary cell on outflow, the **inflow stack**
on inflow.  The inflow stack (exterior values of each advected field) is an argument of every method — None
means no open edge or no exterior, and each stage then uses its own values — so the component holds no state that
changes from step to step.  The compiled kernels stay in ``core.dynamics.sigma_baroclinic`` (looked up there at
call time).  Index order (ntr, k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

from collections.abc import Callable

import numba
import numpy as np

from core.model import threads
from core.model.contract import BuildContext, Issue, Role
from core.vertical.ale import face_thickness


class _Kernels:
    """The compiled tracer kernels of ``core.dynamics.sigma_baroclinic``, looked up at call time (that module
    imports this one, so a module-level import would be circular)."""

    def __getattr__(self, name: str):
        from core.dynamics import sigma_baroclinic
        return getattr(sigma_baroclinic, name)


sb = _Kernels()


class TracerAdvection:
    name = "tracer_advection"
    role = Role.TRACER_ADVECTION
    reads = ("T", "tr", "Qx", "Qy", "Fz")
    writes = ("T", "tr")

    def __init__(self, *, scheme: str, time: str, dt: float, ob) -> None:
        self.tracer_scheme, self.tracer_time, self.dt, self.ob = scheme, time, dt, ob

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}

    def faces(self, c: np.ndarray, flux: np.ndarray, ok: np.ndarray, axis: int, centred: bool = False) -> np.ndarray:
        if centred:
            return sb._centred_faces(c, ok, axis, n_out=flux.shape[axis])
        if self.tracer_scheme.startswith("weno5"):
            return sb._weno5_faces(c, flux, ok, axis, variant="js" if self.tracer_scheme == "weno5js" else "z")
        return sb._up3_faces(c, flux, ok, axis, order=1 if self.tracer_scheme == "upwind1" else 3)

    def open_faces(self, face: np.ndarray, c: np.ndarray, Q: np.ndarray, axis: int,
                   inflow: np.ndarray | None = None) -> np.ndarray:
        """
        Tracer values on the open-boundary faces: upstream (Marsaleix et al. 2006, advection
        equation for tracers) — the boundary cell when flowing out, the external value
        ``T_ext`` (and its square for the T² stack) when flowing in.
        """
        if not self.ob.open:
            return face
        ext = c if inflow is None else inflow
        face = np.array(face, copy=True)
        lo, hi = ("west", "east") if axis == 3 else ("south", "north")
        for edge, f_i, c_i, inflow_sign in ((lo, 0, 0, 1.0), (hi, -1, -1, -1.0)):
            if edge not in self.ob.open:
                continue
            sl_f = [slice(None)] * face.ndim
            sl_c = [slice(None)] * c.ndim
            sl_f[axis], sl_c[axis] = f_i, c_i
            q = np.take(Q, f_i, axis=axis - 1)
            face[tuple(sl_f)] = np.where(inflow_sign * q > 0, ext[tuple(sl_c)], c[tuple(sl_c)])
        return face

    def rhs_numpy(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, ok: np.ndarray,
                  centred: bool = False, inflow: np.ndarray | None = None) -> np.ndarray:
        """Advection RHS −∇·(Q c) of the stack c (ntr, N, ny, nx), vectorised reference path."""
        okb = np.broadcast_to(ok, c.shape)
        fx = Qx * self.open_faces(self.faces(c, np.broadcast_to(Qx, c.shape[:1] + Qx.shape), okb, 3, centred),
                                   c, Qx, 3, inflow)
        fy = Qy * self.open_faces(self.faces(c, np.broadcast_to(Qy, c.shape[:1] + Qy.shape), okb, 2, centred),
                                   c, Qy, 2, inflow)
        # vertical: Fz[j] is the volume flux through interface j (bottom of layer j; 0 = bed, N = surface)
        fzc = np.broadcast_to(Fz[:-1], c.shape)
        fz = np.zeros(c.shape[:1] + Fz.shape)
        fz[:, :-1] = Fz[:-1] * self.faces(c, fzc, okb, 1, centred)
        return -(fx[..., 1:] - fx[..., :-1] + fy[:, :, 1:, :] - fy[:, :, :-1, :] + fz[:, 1:] - fz[:, :-1])

    def rhs(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, Fz: np.ndarray, ok: np.ndarray,
            centred: bool = False, inflow: np.ndarray | None = None) -> np.ndarray:
        """Advection RHS: the fused compiled kernel for WENO5 upwind faces, else the reference path."""
        if centred or not self.tracer_scheme.startswith("weno5"):
            return self.rhs_numpy(c, Qx, Qy, Fz, ok, centred, inflow)
        c = np.ascontiguousarray(c, dtype=float)
        ext = c if inflow is None else inflow
        numba.set_num_threads(threads._THREADS[0])
        return sb._advect_div(c, np.ascontiguousarray(Qx, dtype=float), np.ascontiguousarray(Qy, dtype=float),
                           np.ascontiguousarray(Fz, dtype=float), np.ascontiguousarray(np.broadcast_to(ok, c.shape[1:])),
                           ext, "west" in self.ob.open, "east" in self.ob.open, "south" in self.ob.open,
                           "north" in self.ob.open, self.tracer_scheme == "weno5", bool(np.any(Fz != 0.0)))

    def faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, ok: np.ndarray, ext: np.ndarray | None = None,
                inflow: np.ndarray | None = None):
        """WENO5 horizontal face fluxes (fx, fy) of the stack — the terms _advect_div differences (no ω in ALE).
        ``ext``: the inflow values of _ext_stack, when the caller already has them."""
        c = np.ascontiguousarray(c, dtype=float)
        numba.set_num_threads(threads._THREADS[0])
        return sb._advect_faces_h(c, np.ascontiguousarray(Qx, dtype=float), np.ascontiguousarray(Qy, dtype=float),
                               np.ascontiguousarray(np.broadcast_to(ok, c.shape[1:])),
                               (c if inflow is None else inflow) if ext is None else ext,
                               "west" in self.ob.open, "east" in self.ob.open, "south" in self.ob.open, "north" in self.ob.open,
                               self.tracer_scheme == "weno5")

    def upwind_faces_h(self, c: np.ndarray, Qx: np.ndarray, Qy: np.ndarray, inflow: np.ndarray | None = None):
        """Donor-cell face fluxes Q c_up (the FCT low-order flux); open edges as _advect_div (ext on inflow)."""
        ext = c if inflow is None else inflow
        fx = np.zeros(c.shape[:3] + (c.shape[3] + 1,))
        fy = np.zeros(c.shape[:2] + (c.shape[2] + 1, c.shape[3]))
        fx[..., 1:-1] = Qx[None, ..., 1:-1] * np.where(Qx[None, ..., 1:-1] >= 0, c[..., :-1], c[..., 1:])
        fy[..., 1:-1, :] = Qy[None, :, 1:-1, :] * np.where(Qy[None, :, 1:-1, :] >= 0, c[..., :-1, :], c[..., 1:, :])
        if "west" in self.ob.open:
            fx[..., 0] = Qx[None, ..., 0] * np.where(Qx[None, ..., 0] > 0, ext[..., 0], c[..., 0])
        if "east" in self.ob.open:
            fx[..., -1] = Qx[None, ..., -1] * np.where(Qx[None, ..., -1] < 0, ext[..., -1], c[..., -1])
        if "south" in self.ob.open:
            fy[..., 0, :] = Qy[None, :, 0, :] * np.where(Qy[None, :, 0, :] > 0, ext[..., 0, :], c[..., 0, :])
        if "north" in self.ob.open:
            fy[..., -1, :] = Qy[None, :, -1, :] * np.where(Qy[None, :, -1, :] < 0, ext[..., -1, :], c[..., -1, :])
        return fx, fy

    def advect_fct(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                   ok: np.ndarray, unlimited: bool = False, inflow: np.ndarray | None = None) -> np.ndarray:
        """
        _advect_fct_numpy with its arithmetic in fused Numba kernels (_fct_stage, _fct_fluxes, _fct_coeff_kernel,
        _fct_final): the NumPy glue cost 23–36 % of the BayIW step (2026-10-14).  ``unlimited``: the NumPy path.
        """
        if unlimited:
            return self.advect_fct_numpy(C, V0, V1, Qx, Qy, ok, unlimited=True, inflow=inflow)
        dt = float(self.dt)
        C = np.ascontiguousarray(C, dtype=float)
        Qx, Qy = np.ascontiguousarray(Qx, dtype=float), np.ascontiguousarray(Qy, dtype=float)
        V0, V1 = np.ascontiguousarray(V0, dtype=float), np.ascontiguousarray(V1, dtype=float)
        dV = V1 - V0
        Va, Vb = np.ascontiguousarray(V0 + dV), np.ascontiguousarray(V0 + 0.5 * dV)
        okc = np.ascontiguousarray(np.broadcast_to(ok, C.shape[1:]))
        ext = C if inflow is None else inflow
        ext_s = None if ext is C else ext                                # no inflow data: each stage's own values
        ext = np.ascontiguousarray(ext, dtype=float)
        q0 = V0 * C
        q1, c1 = np.empty_like(C), np.empty_like(C)
        q2, c2 = np.empty_like(C), np.empty_like(C)
        fx0, fy0 = self.faces_h(C, Qx, Qy, okc, ext)
        numba.set_num_threads(threads._THREADS[0])
        sb._fct_stage(q0, q0, fx0, fy0, 0.0, 1.0, dt, Va, C, q1, c1)
        fx1, fy1 = self.faces_h(c1, Qx, Qy, okc, ext_s)
        sb._fct_stage(q0, q1, fx1, fy1, 0.75, 0.25, dt, Vb, C, q2, c2)
        fx2, fy2 = self.faces_h(c2, Qx, Qy, okc, ext_s)
        opn = self.ob.open
        FLx, FLy, Ax, Ay = sb._fct_fluxes(C, Qx, Qy, ext, "west" in opn, "east" in opn, "south" in opn, "north" in opn,
                                       fx0, fy0, fx1, fy1, fx2, fy2)
        qL, cL = q1, c1                                                  # stage buffers reused
        sb._fct_stage(q0, q0, FLx, FLy, 0.0, 1.0, dt, V1, C, qL, cL)
        Cx, Cy = sb._fct_coeff_kernel(C, cL, V1, Ax, Ay, okc, dt)
        return sb._fct_final(q0, FLx, FLy, Ax, Ay, Cx, Cy, dt, V1, C)

    def advect_fct_numpy(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                         ok: np.ndarray, unlimited: bool = False, inflow: np.ndarray | None = None) -> np.ndarray:
        """
        NumPy reference of _advect_fct (test oracle).
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
        fx0, fy0 = self.faces_h(C, Qx, Qy, ok, inflow=inflow)
        q1 = q0 - dt * div(fx0, fy0)
        c1 = safe(q1, V0 + dV)
        fx1, fy1 = self.faces_h(c1, Qx, Qy, ok, inflow=inflow)
        q2 = 0.75 * q0 + 0.25 * (q1 - dt * div(fx1, fy1))
        c2 = safe(q2, V0 + 0.5 * dV)
        fx2, fy2 = self.faces_h(c2, Qx, Qy, ok, inflow=inflow)
        FHx = (fx0 + fx1) / 6.0 + 2.0 * fx2 / 3.0
        FHy = (fy0 + fy1) / 6.0 + 2.0 * fy2 / 3.0
        FLx, FLy = self.upwind_faces_h(C, Qx, Qy, inflow)
        Ax, Ay = FHx - FLx, FHy - FLy
        if unlimited:
            return safe(q0 - dt * div(FHx, FHy), V1)
        cL = safe(q0 - dt * div(FLx, FLy), V1)
        okc = np.ascontiguousarray(np.broadcast_to(ok, C.shape[1:]))
        numba.set_num_threads(threads._THREADS[0])
        Cx, Cy = sb._fct_coeff_kernel(np.ascontiguousarray(C, dtype=float), np.ascontiguousarray(cL),
                                   np.ascontiguousarray(V1, dtype=float), np.ascontiguousarray(Ax),
                                   np.ascontiguousarray(Ay), okc, float(dt))
        return safe(q0 - dt * div(FLx + Cx * Ax, FLy + Cy * Ay), V1)

    def advect_numpy(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
                     Fz: np.ndarray, ok: np.ndarray, inflow: np.ndarray | None = None) -> np.ndarray:
        """
        NumPy reference of _advect (test oracle).
        SSP-RK3 flux-form advection of the tracer stack C (ntr,N,ny,nx) with frozen
        transports Qx, Qy (layer volume fluxes at faces) and Fz (N+1,ny,nx, through
        the interfaces); layer volumes go linearly from V0 to V1.
        """
        dt = self.dt
        rhs = lambda c, centred=False: self.rhs(c, Qx, Qy, Fz, ok, centred, inflow)          # noqa: E731

        dV = V1 - V0
        safe = lambda q, v: np.where(v > 0, q / np.where(v > 0, v, 1.0), C)     # noqa: E731
        q0 = V0 * C
        if self.tracer_time == "pc":
            # predictor: half step with centred 6th-order faces; corrector: full step from n with the
            # upwind (WENO) faces of the mid-step tracer — upwinding applied once (CROCO step3d_t.F)
            Vh = V0 + 0.5 * dV
            ch = safe(q0 + 0.5 * dt * rhs(C, centred=True), Vh)
            return safe(q0 + dt * rhs(ch), V1)
        q1 = q0 + dt * rhs(C)
        Va = V0 + dV
        c1 = safe(q1, Va)
        q2 = 0.75 * q0 + 0.25 * (q1 + dt * rhs(c1))
        Vb = V0 + 0.5 * dV
        c2 = safe(q2, Vb)
        q3 = q0 / 3.0 + 2.0 / 3.0 * (q2 + dt * rhs(c2))
        return safe(q3, V1)

    def advect(self, C: np.ndarray, V0: np.ndarray, V1: np.ndarray, Qx: np.ndarray, Qy: np.ndarray,
               Fz: np.ndarray, ok: np.ndarray, inflow: np.ndarray | None = None) -> np.ndarray:
        """
        SSP-RK3 flux-form advection of the tracer stack C (ntr,N,ny,nx) with frozen transports Qx, Qy and Fz;
        layer volumes go linearly from V0 to V1.  The stage combinations and the ``safe`` division run in the
        fused kernel _rk_stage with _advect_numpy's arithmetic (the reference); 'pc' keeps the NumPy path.
        """
        if self.tracer_time == "pc":
            return self.advect_numpy(C, V0, V1, Qx, Qy, Fz, ok, inflow)
        dt = self.dt
        C = np.ascontiguousarray(C, dtype=float)
        if np.shape(V0) != C.shape[1:] or np.shape(V1) != C.shape[1:]:     # the kernel does not check bounds
            raise ValueError(f"_advect: volume shape {np.shape(V0)}/{np.shape(V1)} vs tracer layers {C.shape[1:]}")
        dV = V1 - V0
        Va, Vb = np.ascontiguousarray(V0 + dV), np.ascontiguousarray(V0 + 0.5 * dV)
        q0 = V0 * C
        numba.set_num_threads(threads._THREADS[0])
        f = lambda a: np.ascontiguousarray(a, dtype=float).ravel()            # noqa: E731  1-D views
        q1, c1 = np.empty_like(C), np.empty_like(C)
        sb._rk_stage(f(q0), f(q0), f(self.rhs(C, Qx, Qy, Fz, ok, inflow=inflow)), dt, f(Va), f(C), 1, q1.ravel(), c1.ravel())
        q2, c2 = np.empty_like(C), np.empty_like(C)
        sb._rk_stage(f(q0), f(q1), f(self.rhs(c1, Qx, Qy, Fz, ok, inflow=inflow)), dt, f(Vb), f(C), 2, q2.ravel(), c2.ravel())
        q3, c3 = np.empty_like(C), np.empty_like(C)
        sb._rk_stage(f(q0), f(q2), f(self.rhs(c2, Qx, Qy, Fz, ok, inflow=inflow)), dt, f(V1), f(C), 3, q3.ravel(), c3.ravel())
        return c3


class TracerDiffusion:
    """
    Explicit Laplacian diffusion along the layers, K_h [m² s⁻¹] (MITgcm diffKhT; flux form, conservative, a
    uniform field stays uniform), of the anomaly T − T_ref(z) when a reference profile is set (Mellor & Blumberg
    1985 [verificar]).  σ: NumPy path; ALE: the Numba kernel with the same per-element arithmetic.  The implicit
    vertical diffusion K_v stays in the step of ``SigmaBaroclinic`` in N1.
    """
    name = "tracer_diffusion"
    role = Role.TRACER_DIFFUSION
    reads = ("T", "tr", "h", "eta")
    writes = ("T", "tr")

    def __init__(self, *, kh: float, dt: float, ds: np.ndarray, fs, eos, pgf, face_depths: Callable,
                 layer_heights: Callable) -> None:
        self.kh, self.dt, self.ds, self.fs, self.eos, self.pgf = kh, dt, ds, fs, eos, pgf
        self.face_depths, self.layer_heights = face_depths, layer_heights

    def check(self, ctx: BuildContext) -> list[Issue]:
        return []

    def state_fields(self) -> dict[str, np.ndarray]:
        return {}

    def diffuse_h(self, T: np.ndarray, V: np.ndarray, eta: np.ndarray, h: np.ndarray | None = None,
                   anomaly: bool = True) -> np.ndarray:
        """
        Explicit Laplacian diffusion along the layers (see _diffuse_h_numpy).  ALE: the Numba kernel
        _diffuse_ale_kernel with the same per-element arithmetic (performance task 5.6); σ: the NumPy version.
        """
        if h is None:
            return self.diffuse_h_numpy(T, V, eta, h, anomaly)
        Tw = T
        if anomaly and self.pgf.rho_ref is not None and not isinstance(self.pgf.rho_ref, str) and self.eos.t_coef != 0:
            Tw = T - (self.eos.t_ref + (self.eos.rho0 - self.pgf.rho_ref(self.layer_heights(eta, h))) / self.eos.t_coef)
        Du, Dv = self.face_depths(eta)
        okx = np.isfinite(self.fs.H_u) & np.isfinite(Du) & (Du > self.fs.d_min)
        oky = np.isfinite(self.fs.H_v) & np.isfinite(Dv) & (Dv > self.fs.d_min)
        act = self.fs.active
        c = np.ascontiguousarray
        out = np.empty(np.shape(T))
        numba.set_num_threads(threads._THREADS[0])
        sb._diffuse_ale_kernel(c(T, dtype=float), c(Tw, dtype=float), c(np.where(act[None], np.nan_to_num(h), 0.0)),
                            c(act), c(self.fs.Lu / self.fs.du), c(self.fs.Lv / self.fs.dv), c(okx), c(oky),
                            c(V, dtype=float), float(self.kh), float(self.dt), out)
        return out

    def diffuse_h_numpy(self, T: np.ndarray, V: np.ndarray, eta: np.ndarray, h: np.ndarray | None = None,
                         anomaly: bool = True) -> np.ndarray:
        """
        NumPy reference of _diffuse_h (test oracle; used for σ).
        Explicit Laplacian diffusion along σ-surfaces in flux form (MITgcm diffKhT):
            V_k ∂T/∂t = −Σ_faces F,   F_x = −K_h L_u h_{u,k} (T_i − T_{i−1}) / Δx_u
        with h_{u,k} = D_u Δσ_k (σ) or the face layer thickness min(h_L, h_R) (ALE); conservative, and a uniform
        T stays uniform.  With a reference profile the anomaly T − T_ref(z) is diffused instead
        (Mellor & Blumberg 1985 [verificar]), so a resting stratification is not mixed across
        sloping σ-surfaces (e.g. a shelf step).  ``anomaly`` False (passive tracers, which have no reference
        profile): the field itself is diffused.
        """
        Tw = T
        if anomaly and self.pgf.rho_ref is not None and not isinstance(self.pgf.rho_ref, str) and self.eos.t_coef != 0:
            Tw = T - (self.eos.t_ref + (self.eos.rho0 - self.pgf.rho_ref(self.layer_heights(eta, h))) / self.eos.t_coef)
        Du, Dv = self.face_depths(eta)
        okx = np.isfinite(self.fs.H_u) & np.isfinite(Du) & (Du > self.fs.d_min)
        oky = np.isfinite(self.fs.H_v) & np.isfinite(Dv) & (Dv > self.fs.d_min)
        ds = self.ds[:, None, None]
        Fx = np.zeros((T.shape[0],) + Du.shape)
        Fy = np.zeros((T.shape[0],) + Dv.shape)
        if h is None:
            Fx[..., 1:-1] = -self.kh * (self.fs.Lu * np.nan_to_num(Du) / self.fs.du)[None, :, 1:-1] * ds \
                * (Tw[..., 1:] - Tw[..., :-1])
            Fy[:, 1:-1, :] = -self.kh * (self.fs.Lv * np.nan_to_num(Dv) / self.fs.dv)[None, 1:-1, :] * ds \
                * (Tw[:, 1:, :] - Tw[:, :-1, :])
        else:                                                        # ALE: face layer thickness (min of the cells)
            hu, hv = face_thickness(h, self.fs.active)
            Fx[..., 1:-1] = -self.kh * (self.fs.Lu / self.fs.du)[None, :, 1:-1] * hu[..., 1:-1] \
                * (Tw[..., 1:] - Tw[..., :-1])
            Fy[:, 1:-1, :] = -self.kh * (self.fs.Lv / self.fs.dv)[None, 1:-1, :] * hv[:, 1:-1, :] \
                * (Tw[:, 1:, :] - Tw[:, :-1, :])
        Fx = np.where(okx[None], Fx, 0.0)
        Fy = np.where(oky[None], Fy, 0.0)
        div = Fx[..., 1:] - Fx[..., :-1] + Fy[:, 1:, :] - Fy[:, :-1, :]
        return np.where(V > 0, T - self.dt * div / np.where(V > 0, V, 1.0), T)
