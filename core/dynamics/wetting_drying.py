"""
Semi-implicit free surface with wetting and drying (barotropic, C grid).

Casulli (1990) θ-method with the *total* water depth at the faces:

    η_i^{n+1} = η_i^n − (Δt / A_i) Σ_f s_f L_f D_f [θ u_f^{n+1} + (1 − θ) u_f^n]
    u_f^{n+1} = u*_f − g θ Δt (η_R^{n+1} − η_L^{n+1}) / δ_f
    u*_f      = û_f − g (1 − θ) Δt (η_R^n − η_L^n) / δ_f

    D_f = max(0, H_f + η_up),   H_f = min(H_L, H_R)   (the higher bed),
    η_up = η of the upwind cell (u_f^n > 0: left, < 0: right, 0: the higher η)

A face with D_f ≤ D_min is dry: u_f = 0 and it carries no flux.  Substituting
u^{n+1} gives a symmetric positive-definite system for η^{n+1},

    A_i η_i + Σ_f c_f (η_i − η_nb) = A_i η_i^n − Δt Σ_f s_f L_f D_f [θ u*_f + (1 − θ) u_f^n],
    c_f = g θ² Δt² L_f D_f / δ_f ,

which stays SPD with dry cells (their row reduces to A_i η_i = A_i η_i^n).  The
flux form conserves volume exactly; the upwind total depth makes the scheme
mass-positive (Casulli & Cheng 1992; Casulli 2009), and a dry cell keeps its
surface on the bed (η = −H), so a lake at rest next to emerged land stays at
rest (well balanced).  The depth H may be negative (bed above datum: land
that can flood); NaN marks permanent walls.

Unlike ``ImplicitFreeSurface`` (fixed H, factorised once) the matrix changes
every step; it is assembled from face arrays and solved directly.

References
----------
Casulli, V. (1990). Semi-implicit finite difference methods for the two-
    dimensional shallow water equations. J. Comput. Phys. 86, 56–74.
Casulli, V. & Cheng, R.T. (1992). Semi-implicit finite difference methods for
    three-dimensional shallow water flow. Int. J. Numer. Methods Fluids 15, 629–648.
Casulli, V. (2009). A high-resolution wetting and drying algorithm for
    free-surface hydrodynamics. Int. J. Numer. Methods Fluids 60, 391–408.
Thacker, W.C. (1981). Some exact solutions to the nonlinear shallow-water wave
    equations. J. Fluid Mech. 107, 499–508.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.sparse.linalg import LinearOperator, cg, splu, spsolve

G = 9.81


class WetDryFreeSurface:
    """
    Parameters
    ----------
    h_grid : xr.Dataset
        Horizontal grid (dx_u, dy_u, dx_v, dy_v, area_t).
    bathy : (ny, nx)
        Still-water depth H [m] below the datum; negative above it (floodable
        land); NaN = permanent wall.
    dt : float
        Time step [s].
    theta : float
        Implicit weight, ½ ≤ θ ≤ 1.
    g : float
        Gravity [m s⁻²].
    d_min : float
        Face depth [m] below which a face is dry (CROCO ``Dcrit`` analogue).
    flather_u, flather_v : (ny, nx+1), (ny+1, nx) bool, optional
        Domain-edge faces with the Flather (1976) radiation condition, using the
        total depth of the adjacent cell D = H + η_c (closed while it is dry):
        outward transport Q = L D [ū_ext,out + √(g/D)(η_c − η_ext)], θ-weighted
        between n and n+1 as in ``ImplicitFreeSurface``.
    solver : {'direct', 'cg'}
        Sparse direct solve each step, or preconditioned conjugate gradients
        warm-started from η^n (the matrix changes every step with the total depth).
    cg_rtol : float
        Relative residual tolerance of CG.
    precond : {'lu', 'jacobi'}
        CG preconditioner: 'lu' (default) uses the exact LU factorisation of a
        lagged matrix — still the inverse of an SPD matrix close to the current
        one — refactorised when CG needs more than ``refactor_iterations``;
        'jacobi' is the diagonal.  On the 96 273-cell northern Gulf of
        California nest: 1–3 iterations and 7–19 ms per solve vs 97–117
        iterations and ~50 ms with Jacobi (factorisation 0.19 s).
    """

    def __init__(self, h_grid: xr.Dataset, bathy: np.ndarray, dt: float, theta: float = 0.6,
                 g: float = G, d_min: float = 0.05, flather_u: np.ndarray | None = None,
                 flather_v: np.ndarray | None = None, solver: str = "direct", cg_rtol: float = 1e-10,
                 precond: str = "lu", refactor_iterations: int = 5) -> None:
        if not 0.5 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0.5, 1], got {theta}")
        if solver not in ("direct", "cg"):
            raise ValueError(f"solver must be 'direct' or 'cg', got {solver!r}")
        if precond not in ("lu", "jacobi"):
            raise ValueError(f"precond must be 'lu' or 'jacobi', got {precond!r}")
        self.solver, self.cg_rtol = solver, cg_rtol
        # CG preconditioner: 'lu' = exact LU of a lagged matrix (refactorised when CG needs more than
        # ``refactor_iterations``); the matrix changes slowly with the total depth, so the lagged inverse
        # stays an excellent SPD preconditioner (1–3 iterations vs ~100 with Jacobi on the northern Gulf nest)
        self.precond, self.refactor_iterations = precond, refactor_iterations
        self._lu = None
        self._refactor = True
        self.lu_factorizations = 0
        self.cg_iterations = 0
        self.dt, self.theta, self.g, self.d_min = float(dt), float(theta), float(g), float(d_min)
        H = np.asarray(bathy, float)
        self.H = H
        self.active = np.isfinite(H)
        ny, nx = H.shape
        self.area = h_grid["area_t"].values
        self.Lu, self.du = h_grid["dy_u"].values, h_grid["dx_u"].values
        self.Lv, self.dv = h_grid["dx_v"].values, h_grid["dy_v"].values
        self.H_u = np.full((ny, nx + 1), np.nan)
        self.H_v = np.full((ny + 1, nx), np.nan)
        self.H_u[:, 1:-1] = np.fmin(H[:, :-1], H[:, 1:])                 # higher bed of the two
        self.H_u[:, 1:-1][~(self.active[:, :-1] & self.active[:, 1:])] = np.nan
        self.H_v[1:-1, :] = np.fmin(H[:-1, :], H[1:, :])
        self.H_v[1:-1, :][~(self.active[:-1, :] & self.active[1:, :])] = np.nan
        self.index = -np.ones((ny, nx), dtype=np.int64)
        self.index[self.active] = np.arange(self.active.sum())
        self.last_clip_volume = 0.0
        # Flather faces: (axis, face indices, adjacent cell indices, outward sign, face length)
        self._flather = []
        for mask, axis in ((flather_u, "u"), (flather_v, "v")):
            if mask is None:
                continue
            mask = np.asarray(mask, bool)
            inner = mask.copy()
            if axis == "u":
                inner[:, [0, -1]] = False
            else:
                inner[[0, -1], :] = False
            if inner.any():
                raise ValueError("Flather faces must lie on the domain edge")
            for edge in (0, -1):
                sel = np.zeros_like(mask)
                if axis == "u":
                    sel[:, edge] = mask[:, edge]
                    jj, ii = np.nonzero(sel)
                    cj, ci = jj, (ii if edge == 0 else ii - 1)
                    length = self.Lu[jj, ii]
                else:
                    sel[edge, :] = mask[edge, :]
                    jj, ii = np.nonzero(sel)
                    cj, ci = (jj if edge == 0 else jj - 1), ii
                    length = self.Lv[jj, ii]
                keep = self.active[cj, ci]
                if keep.any():
                    self._flather.append(dict(axis=axis, face=(jj[keep], ii[keep]), cell=(cj[keep], ci[keep]),
                                              sign=-1.0 if edge == 0 else 1.0, length=length[keep]))

    # ------------------------------------------------------------------
    def _face_depth(self, eta: np.ndarray, vel: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Upwind total depth and wet mask at the faces of ``axis`` (1: u, 0: v)."""
        Hf = self.H_u if axis == 1 else self.H_v
        lo = np.full(Hf.shape, np.nan)
        hi = np.full(Hf.shape, np.nan)
        if axis == 1:
            lo[:, 1:-1], hi[:, 1:-1] = eta[:, :-1], eta[:, 1:]
        else:
            lo[1:-1, :], hi[1:-1, :] = eta[:-1, :], eta[1:, :]
        up = np.where(vel > 0, lo, np.where(vel < 0, hi, np.fmax(lo, hi)))
        D = Hf + up
        wet = np.isfinite(D) & (D > self.d_min)
        return np.where(wet, D, 0.0), wet

    def _gradient(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.zeros(self.H_u.shape)
        gy = np.zeros(self.H_v.shape)
        gx[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / self.du[:, 1:-1]
        gy[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / self.dv[1:-1, :]
        return gx, gy

    def _divergence(self, fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
        return fu[:, 1:] - fu[:, :-1] + fv[1:, :] - fv[:-1, :]

    def _matrix(self, cu: np.ndarray, cv: np.ndarray, extra_diag: np.ndarray | None = None) -> sp.csr_matrix:
        n = int(self.active.sum())
        diag = self.area[self.active].copy()
        if extra_diag is not None:
            diag += extra_diag
        rows, cols, vals = [], [], []
        for c, a, b in ((cu[:, 1:-1], self.index[:, :-1], self.index[:, 1:]),
                        (cv[1:-1, :], self.index[:-1, :], self.index[1:, :])):
            m = (c > 0) & (a >= 0) & (b >= 0)
            ca, ia, ib = c[m], a[m], b[m]
            np.add.at(diag, ia, ca)
            np.add.at(diag, ib, ca)
            rows += [ia, ib]
            cols += [ib, ia]
            vals += [-ca, -ca]
        rows.append(np.arange(n))
        cols.append(np.arange(n))
        vals.append(diag)
        return sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))

    # ------------------------------------------------------------------
    def step(self, eta: np.ndarray, u_n: np.ndarray, v_n: np.ndarray, u_hat: np.ndarray,
             v_hat: np.ndarray, flather_eta=None, flather_ubar: np.ndarray | None = None,
             flather_vbar: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance η and the depth-mean velocities one step.

        u_n, v_n : (ny, nx+1), (ny+1, nx) velocities at time n [m s⁻¹]
        u_hat, v_hat : velocities after all other (explicit) tendencies
        flather_eta : external η at the Flather faces (array, or tuple (eta_u, eta_v))
        flather_ubar, flather_vbar : external depth-mean velocity there (model sign)
        Returns η^{n+1}, u^{n+1}, v^{n+1} (u = 0 on dry faces).
        """
        th, dt, g = self.theta, self.dt, self.g
        Du, wu = self._face_depth(eta, u_n, 1)
        Dv, wv = self._face_depth(eta, v_n, 0)
        gx, gy = self._gradient(eta)
        us = np.where(wu, u_hat - g * (1 - th) * dt * gx, 0.0)
        vs = np.where(wv, v_hat - g * (1 - th) * dt * gy, 0.0)
        un = np.where(wu, u_n, 0.0)
        vn = np.where(wv, v_n, 0.0)
        flux = self._divergence(self.Lu * Du * (th * us + (1 - th) * un),
                                self.Lv * Dv * (th * vs + (1 - th) * vn))
        rhs_full = self.area * eta - dt * flux
        extra = np.zeros(self.area.shape)
        fl_state = []
        if self._flather:
            if flather_eta is None:
                raise ValueError("Flather faces need flather_eta (and the external depth-mean velocity)")
            for fl in self._flather:
                e_all = (flather_eta[0] if fl["axis"] == "u" else flather_eta[1]) \
                    if isinstance(flather_eta, tuple) else flather_eta
                vel = flather_ubar if fl["axis"] == "u" else flather_vbar
                e_ext = np.asarray(e_all)[fl["face"]]
                ub = np.zeros_like(e_ext) if vel is None else np.asarray(vel)[fl["face"]]
                D = np.maximum(self.H[fl["cell"]] + eta[fl["cell"]], 0.0)
                open_ = D > self.d_min
                c = np.where(open_, fl["length"] * np.sqrt(g * D), 0.0)
                known = np.where(open_, fl["length"] * D * fl["sign"] * ub + c * ((1 - th) * eta[fl["cell"]] - e_ext), 0.0)
                np.add.at(rhs_full, fl["cell"], -dt * known)
                np.add.at(extra, fl["cell"], dt * th * c)
                fl_state.append((D, open_, c, e_ext, ub))
        rhs = rhs_full[self.active]
        cu = g * th**2 * dt**2 * self.Lu * Du / self.du
        cv = g * th**2 * dt**2 * self.Lv * Dv / self.dv
        A = self._matrix(np.where(wu, cu, 0.0), np.where(wv, cv, 0.0), extra[self.active])
        if self.solver == "direct":
            x = spsolve(A.tocsc(), rhs)
        else:
            if self.precond == "lu":
                if self._lu is None or self._refactor:
                    self._lu = splu(A.tocsc())
                    self.lu_factorizations += 1
                    self._refactor = False
                M = LinearOperator(A.shape, matvec=self._lu.solve)
            else:
                dinv = 1.0 / A.diagonal()
                M = LinearOperator(A.shape, matvec=lambda r: dinv * r)
            count = [0]

            def _cb(_):
                count[0] += 1
            x, info = cg(A, rhs, x0=eta[self.active], rtol=self.cg_rtol, atol=0.0, M=M, maxiter=10 * rhs.size,
                         callback=_cb)
            if info != 0:
                raise RuntimeError(f"wet-dry free-surface CG did not converge (info={info})")
            self.cg_iterations = count[0]
            if self.precond == "lu" and count[0] > self.refactor_iterations:
                self._refactor = True                       # refresh the lagged LU for the next step
        eta_new = np.array(eta, float, copy=True)
        eta_new[self.active] = x

        # positivity guard (should not trigger; reported for diagnostics)
        depth = self.H + eta_new
        neg = self.active & (depth < 0)
        self.last_clip_volume = float(np.sum(-depth[neg] * self.area[neg]))
        eta_new[neg] = -self.H[neg]

        gx1, gy1 = self._gradient(eta_new)
        self.last_wet_u, self.last_wet_v = wu, wv            # faces that carried flow in this step
        self.last_Du, self.last_Dv = Du, Dv                  # upwind total face depth used in the fluxes
        u_new = np.where(wu, us - g * th * dt * gx1, 0.0)
        v_new = np.where(wv, vs - g * th * dt * gy1, 0.0)
        # θ-weighted volume flux through the Flather faces as used by the continuity equation (model sign)
        # and the total depth there: the σ-layer transports must add up to exactly this flux
        self.last_open_Qu, self.last_open_Qv = np.zeros(self.H_u.shape), np.zeros(self.H_v.shape)
        self.last_open_Du, self.last_open_Dv = np.zeros(self.H_u.shape), np.zeros(self.H_v.shape)
        for fl, (D, open_, c, e_ext, ub) in zip(self._flather, fl_state):
            q = fl["length"] * D * fl["sign"] * ub + c * (eta_new[fl["cell"]] - e_ext)     # outward, at n+1
            vel = np.where(open_, fl["sign"] * q / np.where(open_, fl["length"] * D, 1.0), 0.0)
            (u_new if fl["axis"] == "u" else v_new)[fl["face"]] = vel
            q_th = fl["length"] * D * fl["sign"] * ub + c * ((1 - th) * eta[fl["cell"]] + th * eta_new[fl["cell"]] - e_ext)
            Q, Dd = (self.last_open_Qu, self.last_open_Du) if fl["axis"] == "u" else (self.last_open_Qv, self.last_open_Dv)
            Q[fl["face"]] = np.where(open_, fl["sign"] * q_th, 0.0)
            Dd[fl["face"]] = np.where(open_, D, 0.0)
        if self.solver == "cg":
            # exact conservation (MITgcm exactConserv): η^{n+1} from the divergence of the final θ-weighted
            # transports and Flather fluxes, so the CG tolerance does not leak into the volume and tracer budgets
            ui = np.where(wu, us - g * th * dt * gx1, 0.0)
            vi = np.where(wv, vs - g * th * dt * gy1, 0.0)
            div = self._divergence(self.Lu * Du * (th * ui + (1 - th) * un), self.Lv * Dv * (th * vi + (1 - th) * vn))
            for fl in self._flather:
                Q = self.last_open_Qu if fl["axis"] == "u" else self.last_open_Qv
                np.add.at(div, fl["cell"], fl["sign"] * Q[fl["face"]])            # outward flux of the cell
            eta_new = np.where(self.active, eta - dt * div / np.where(self.active, self.area, 1.0), eta_new)
        return eta_new, u_new, v_new
