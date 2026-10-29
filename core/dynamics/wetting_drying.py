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
    clamped_u, clamped_v : (ny, nx+1), (ny+1, nx) bool, optional
        Subset of the Flather faces whose adjacent cell has a **prescribed elevation** instead (Dirichlet η,
        "clamped" sea level; CROCO OBC_COM_ZSPECIFIED, OCEAN/zetabc.F:95-99, which sets the boundary ρ-point
        ζ(0, j) = ζ_bry and lets the face between it and the first interior point carry what the momentum equation
        gives; GETM's elevation boundary does the same on its boundary T-points; MOM6 has no SSH Dirichlet
        segment, only Flather / specified velocity, MOM_open_boundary.F90:238,1807,1848).  Here the boundary cell
        is a model cell: η^{n+1} there is the exterior η (``flather_eta``), the cell is eliminated from the
        implicit solve (known value moved to the rhs, the system stays SPD), and the flux through its domain-edge
        face is **diagnosed** from the cell's continuity, A (η^{n+1} − η^n)/Δt = Q_in − (outflow through its other
        faces)^θ + A fw, so the volume budget closes exactly.  A cell on two clamped faces shares the flux by face
        length.  None: no change (bit for bit).
    solver : {'direct', 'cg'}
        Sparse direct solve each step, or preconditioned conjugate gradients
        warm-started from η^n (the matrix changes every step with the total depth).
        With 'cg', η^{n+1} is recomputed from the divergence of the final transports
        (MITgcm exactConserv), so the CG tolerance does not leak into the volume budget;
        the positivity guard is applied to that final η.
    cg_rtol : float
        Relative residual tolerance of CG.
    precond : {'lu', 'jacobi'}
        CG preconditioner: 'lu' (default) uses the exact LU factorisation of a
        lagged matrix — still the inverse of an SPD matrix close to the current
        one — refactorised when CG needs more than ``refactor_iterations``;
        'jacobi' is the diagonal.  On the 96 273-cell northern Gulf of
        California nest: 1–3 iterations and 7–19 ms per solve vs 97–117
        iterations and ~50 ms with Jacobi (factorisation 0.19 s).
    periodic : core.model.periodic.Periodic, tuple of 'x'/'y', optional
        Periodic axes (N3 task 3).  With 'x' the u-faces 0 and nx are the same face, between cell nx−1 and
        cell 0: H_f = min(H_{nx−1}, H_0), the upwind η, the gradient (η_0 − η_{nx−1})/δ_f and one matrix
        coupling c_f between the two cells (the system stays SPD; with nx = 1 the coupling cancels).  Flather
        faces on a periodic edge are refused.  Default: none (bit for bit).
    """

    def __init__(self, h_grid: xr.Dataset, bathy: np.ndarray, dt: float, theta: float = 0.6,
                 g: float = G, d_min: float = 0.05, flather_u: np.ndarray | None = None,
                 flather_v: np.ndarray | None = None, solver: str = "direct", cg_rtol: float = 1e-10,
                 precond: str = "lu", refactor_iterations: int = 5, clamped_u: np.ndarray | None = None,
                 clamped_v: np.ndarray | None = None, periodic=None) -> None:
        from core.model.periodic import Periodic
        self.per = periodic if isinstance(periodic, Periodic) else Periodic(periodic or ())
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
        self._lu_matrix = None
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
        if self.per.x:                                       # the wrap face between cells nx−1 and 0
            hw = np.where(self.active[:, -1] & self.active[:, 0], np.fmin(H[:, -1], H[:, 0]), np.nan)
            self.H_u[:, 0] = self.H_u[:, -1] = hw
        if self.per.y:
            hw = np.where(self.active[-1, :] & self.active[0, :], np.fmin(H[-1, :], H[0, :]), np.nan)
            self.H_v[0, :] = self.H_v[-1, :] = hw
        for name, mask, ax in (("flather_u", flather_u, "x"), ("flather_v", flather_v, "y")):
            if mask is not None and ax in self.per.axes and np.asarray(mask, bool).any():
                raise ValueError(f"{name}: Flather (open) faces on the periodic axis {ax!r}")
        self.index = -np.ones((ny, nx), dtype=np.int64)
        self.index[self.active] = np.arange(self.active.sum())
        self.last_clip_volume = 0.0
        self.last_fw = None                                  # freshwater source of the last step [m s⁻¹] (N2a)
        # Flather faces: (axis, face indices, adjacent cell indices, outward sign, face length)
        self._flather = []
        for mask, clamp, axis in ((flather_u, clamped_u, "u"), (flather_v, clamped_v, "v")):
            if mask is None:
                if clamp is not None and np.asarray(clamp, bool).any():
                    raise ValueError("clamped-η faces must be Flather (open) faces")
                continue
            mask = np.asarray(mask, bool)
            clamp = np.zeros_like(mask) if clamp is None else np.asarray(clamp, bool)
            if (clamp & ~mask).any():
                raise ValueError("clamped-η faces must be Flather (open) faces")
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
                for cl in (False, True):                      # Flather faces, then clamped-η faces of this edge
                    k = keep & (clamp[jj, ii] == cl)
                    if k.any():
                        self._flather.append(dict(axis=axis, face=(jj[k], ii[k]), cell=(cj[k], ci[k]),
                                                  sign=-1.0 if edge == 0 else 1.0, length=length[k], clamped=cl))
        self._clamped = [fl for fl in self._flather if fl["clamped"]]

    # ------------------------------------------------------------------
    def _face_depth(self, eta: np.ndarray, vel: np.ndarray, axis: int) -> tuple[np.ndarray, np.ndarray]:
        """Upwind total depth and wet mask at the faces of ``axis`` (1: u, 0: v)."""
        Hf = self.H_u if axis == 1 else self.H_v
        lo = np.full(Hf.shape, np.nan)
        hi = np.full(Hf.shape, np.nan)
        if axis == 1:
            lo[:, 1:-1], hi[:, 1:-1] = eta[:, :-1], eta[:, 1:]
            if self.per.x:
                lo[:, 0] = lo[:, -1] = eta[:, -1]
                hi[:, 0] = hi[:, -1] = eta[:, 0]
        else:
            lo[1:-1, :], hi[1:-1, :] = eta[:-1, :], eta[1:, :]
            if self.per.y:
                lo[0, :] = lo[-1, :] = eta[-1, :]
                hi[0, :] = hi[-1, :] = eta[0, :]
        up = np.where(vel > 0, lo, np.where(vel < 0, hi, np.fmax(lo, hi)))
        D = Hf + up
        wet = np.isfinite(D) & (D > self.d_min)
        return np.where(wet, D, 0.0), wet

    def _gradient(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        gx = np.zeros(self.H_u.shape)
        gy = np.zeros(self.H_v.shape)
        gx[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / self.du[:, 1:-1]
        gy[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / self.dv[1:-1, :]
        if self.per.x:
            gx[:, 0] = gx[:, -1] = (eta[:, 0] - eta[:, -1]) / self.du[:, 0]
        if self.per.y:
            gy[0, :] = gy[-1, :] = (eta[0, :] - eta[-1, :]) / self.dv[0, :]
        return gx, gy

    def _divergence(self, fu: np.ndarray, fv: np.ndarray) -> np.ndarray:
        return fu[:, 1:] - fu[:, :-1] + fv[1:, :] - fv[:-1, :]

    def _matrix(self, cu: np.ndarray, cv: np.ndarray, extra_diag: np.ndarray | None = None) -> sp.csr_matrix:
        n = int(self.active.sum())
        diag = self.area[self.active].copy()
        if extra_diag is not None:
            diag += extra_diag
        rows, cols, vals = [], [], []
        pairs = [(cu[:, 1:-1], self.index[:, :-1], self.index[:, 1:]),
                 (cv[1:-1, :], self.index[:-1, :], self.index[1:, :])]
        if self.per.x:                                       # the wrap face, once (face nx is face 0)
            pairs.append((cu[:, :1], self.index[:, -1:], self.index[:, :1]))
        if self.per.y:
            pairs.append((cv[:1, :], self.index[-1:, :], self.index[:1, :]))
        for c, a, b in pairs:
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
    def preconditioner_state(self) -> dict | None:
        """
        The lagged-LU CG preconditioner as plain arrays, for bit-for-bit restarts: the CSC matrix it factorised
        (data, indices, indptr, shape) and whether the next step refactorises.  CG stops at an iterate that
        depends on the preconditioner, so a restart that refactorised the current matrix would drift at the
        CG tolerance.  None when there is no LU yet.
        """
        if self._lu_matrix is None:
            return None
        A = self._lu_matrix
        return dict(data=A.data.copy(), indices=A.indices.copy(), indptr=A.indptr.copy(), shape=A.shape,
                    refactor=bool(self._refactor))

    def set_preconditioner_state(self, state: dict | None) -> None:
        """Restore preconditioner_state: SuperLU of the same matrix gives the same factors."""
        if state is None:
            self._lu = self._lu_matrix = None
            self._refactor = True
            return
        self._lu_matrix = sp.csc_matrix((state["data"], state["indices"], state["indptr"]),
                                        shape=tuple(int(n) for n in state["shape"]))
        self._lu = splu(self._lu_matrix)
        self._refactor = bool(state["refactor"])

    def _clamped_fluxes(self, eta, eta_new, us, vs, un, vn, wu, wv, Du, Dv, fw_eff, fl_state, u_new, v_new) -> None:
        """
        Edge-face fluxes of the clamped-η cells, diagnosed from their continuity (class docstring): the inflow
        Q_in = A (η^{n+1} − η^n)/Δt + (θ-weighted outflow through the interior faces) + (Flather outflow) − A fw.
        Stored as the θ-weighted open flux (model sign) with the cell's depth at n, and as the face velocity.
        """
        th, dt, g = self.theta, self.dt, self.g
        gx1, gy1 = self._gradient(eta_new)
        ui = np.where(wu, us - g * th * dt * gx1, 0.0)
        vi = np.where(wv, vs - g * th * dt * gy1, 0.0)
        out = self._divergence(self.Lu * Du * (th * ui + (1 - th) * un), self.Lv * Dv * (th * vi + (1 - th) * vn))
        for fl in self._flather:                                           # Flather outflow of the cells
            if not fl["clamped"]:
                Q = self.last_open_Qu if fl["axis"] == "u" else self.last_open_Qv
                np.add.at(out, fl["cell"], fl["sign"] * Q[fl["face"]])
        q_in = self.area * (eta_new - eta) / dt + out
        if fw_eff is not None:
            q_in = q_in - self.area * fw_eff
        total_len = np.zeros(self.area.shape)
        for fl in self._clamped:
            np.add.at(total_len, fl["cell"], fl["length"])
        for fl, (D, _, _, _, _, _) in zip(self._flather, fl_state):
            if not fl["clamped"]:
                continue
            qin = q_in[fl["cell"]] * fl["length"] / total_len[fl["cell"]]
            Qm = -fl["sign"] * qin                                           # model sign (+x / +y)
            wet = D > self.d_min
            (u_new if fl["axis"] == "u" else v_new)[fl["face"]] = np.where(
                wet, Qm / np.where(wet, fl["length"] * D, 1.0), 0.0)
            if fl["axis"] == "u":
                Q, Dd = self.last_open_Qu, self.last_open_Du
            else:
                Q, Dd = self.last_open_Qv, self.last_open_Dv
            Q[fl["face"]] = Qm
            Dd[fl["face"]] = D

    def step(self, eta: np.ndarray, u_n: np.ndarray, v_n: np.ndarray, u_hat: np.ndarray,
             v_hat: np.ndarray, flather_eta=None, flather_ubar: np.ndarray | None = None,
             flather_vbar: np.ndarray | None = None, flather_q=None,
             fw: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance η and the depth-mean velocities one step.

        u_n, v_n : (ny, nx+1), (ny+1, nx) velocities at time n [m s⁻¹]
        u_hat, v_hat : velocities after all other (explicit) tendencies
        flather_eta : external η at the Flather faces (array, or tuple (eta_u, eta_v))
        flather_ubar, flather_vbar : external depth-mean velocity there (model sign)
        flather_q : (q_u, q_v) prescribed volume flux at Flather faces [m³ s⁻¹, model sign], NaN where Flather applies:
            those faces carry exactly q, with no radiation term (MOM6 SIMPLE segment; DOME inflow)
        fw : (ny, nx) surface freshwater flux P − E [m s⁻¹, positive into the ocean] (N2a), or None.  A real volume
            source (Huang 1993; MITgcm useRealFreshWaterFlux: integr_continuity.F:93,185, solve_for_pressure.F:147-156):
            A (ηⁿ⁺¹ − ηⁿ)/Δt = −∇·(D ū)^θ + A fw_eff, with fw_eff = fw on active cells with Hⁿ + ηⁿ > d_min and 0 on
            land and dry cells (no phantom volume).  The exact-conservation recompute of the CG path uses the same
            source.  fw_eff is kept as ``last_fw`` (None without fw).  fw None adds no operation (bit for bit).
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
        fw_eff = None
        if fw is not None:                                   # real freshwater volume source (N2a)
            fw_eff = np.where(self.active & (self.H + eta > self.d_min), np.asarray(fw, float), 0.0)
            rhs_full = rhs_full + dt * self.area * fw_eff
        self.last_fw = fw_eff
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
                if fl["clamped"]:                             # prescribed η: no radiation term (handled below)
                    fl_state.append((np.maximum(self.H[fl["cell"]] + eta[fl["cell"]], 0.0), None, None, e_ext,
                                     None, None))
                    continue
                ub = np.zeros_like(e_ext) if vel is None else np.asarray(vel)[fl["face"]]
                D = np.maximum(self.H[fl["cell"]] + eta[fl["cell"]], 0.0)
                open_ = D > self.d_min
                qp = None if flather_q is None else np.asarray(flather_q[0] if fl["axis"] == "u" else flather_q[1])[fl["face"]]
                fixed = np.zeros(e_ext.shape, bool) if qp is None else np.isfinite(qp)
                c = np.where(open_ & ~fixed, fl["length"] * np.sqrt(g * D), 0.0)
                out_q = fl["length"] * D * fl["sign"] * ub                                       # outward
                if qp is not None:
                    out_q = np.where(fixed, fl["sign"] * np.nan_to_num(qp), out_q)
                known = np.where(open_, out_q + c * ((1 - th) * eta[fl["cell"]] - e_ext), 0.0)
                np.add.at(rhs_full, fl["cell"], -dt * known)
                np.add.at(extra, fl["cell"], dt * th * c)
                fl_state.append((D, open_, c, e_ext, ub, out_q))
        rhs = rhs_full[self.active]
        cu = g * th**2 * dt**2 * self.Lu * Du / self.du
        cv = g * th**2 * dt**2 * self.Lv * Dv / self.dv
        A = self._matrix(np.where(wu, cu, 0.0), np.where(wv, cv, 0.0), extra[self.active])
        if self._clamped:
            # Dirichlet η at the clamped cells: x_b known, moved to the rhs; rows and columns replaced by the
            # identity (symmetric elimination keeps A SPD for CG)
            xb = np.zeros(A.shape[0])
            fix = np.zeros(A.shape[0], bool)
            for fl, st_ in zip(self._flather, fl_state):
                if fl["clamped"]:
                    idx = self.index[fl["cell"]]
                    xb[idx], fix[idx] = st_[3], True
            rhs = rhs - A @ xb
            rhs[fix] = xb[fix]
            P = sp.diags((~fix).astype(float))
            A = (P @ A @ P + sp.diags(fix.astype(float))).tocsr()
        if self.solver == "direct":
            x = spsolve(A.tocsc(), rhs)
        else:
            if self.precond == "lu":
                if self._lu is None or self._refactor:
                    self._lu_matrix = A.tocsc()                   # kept for restarts (preconditioner_state)
                    self._lu = splu(self._lu_matrix)
                    self.lu_factorizations += 1
                    self._refactor = False
                M = LinearOperator(A.shape, matvec=self._lu.solve)
            else:
                dinv = 1.0 / A.diagonal()
                M = LinearOperator(A.shape, matvec=lambda r: dinv * r)
            count = [0]

            def _cb(_):
                count[0] += 1
            x0 = eta[self.active]
            if self._clamped:
                x0 = np.where(fix, xb, x0)
            x, info = cg(A, rhs, x0=x0, rtol=self.cg_rtol, atol=0.0, M=M, maxiter=10 * rhs.size,
                         callback=_cb)
            if info != 0:
                raise RuntimeError(f"wet-dry free-surface CG did not converge (info={info})")
            self.cg_iterations = count[0]
            if self.precond == "lu" and count[0] > self.refactor_iterations:
                self._refactor = True                       # refresh the lagged LU for the next step
        if self._clamped:
            x[fix] = xb[fix]                                  # exact prescribed η (CG stops at its tolerance)
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
        for fl, (D, open_, c, e_ext, ub, out_q) in zip(self._flather, fl_state):
            if fl["clamped"]:
                continue
            q = out_q + c * (eta_new[fl["cell"]] - e_ext)                                  # outward, at n+1
            vel = np.where(open_, fl["sign"] * q / np.where(open_, fl["length"] * D, 1.0), 0.0)
            (u_new if fl["axis"] == "u" else v_new)[fl["face"]] = vel
            q_th = out_q + c * ((1 - th) * eta[fl["cell"]] + th * eta_new[fl["cell"]] - e_ext)
            Q, Dd = (self.last_open_Qu, self.last_open_Du) if fl["axis"] == "u" else (self.last_open_Qv, self.last_open_Dv)
            Q[fl["face"]] = np.where(open_, fl["sign"] * q_th, 0.0)
            Dd[fl["face"]] = np.where(open_, D, 0.0)
        if self._clamped:
            self._clamped_fluxes(eta, eta_new, us, vs, un, vn, wu, wv, Du, Dv, fw_eff, fl_state, u_new, v_new)
        if self.solver == "cg":
            # exact conservation (MITgcm exactConserv): η^{n+1} from the divergence of the final θ-weighted
            # transports and Flather fluxes, so the CG tolerance does not leak into the volume and tracer budgets
            ui = np.where(wu, us - g * th * dt * gx1, 0.0)
            vi = np.where(wv, vs - g * th * dt * gy1, 0.0)
            div = self._divergence(self.Lu * Du * (th * ui + (1 - th) * un), self.Lv * Dv * (th * vi + (1 - th) * vn))
            for fl in self._flather:
                Q = self.last_open_Qu if fl["axis"] == "u" else self.last_open_Qv
                np.add.at(div, fl["cell"], fl["sign"] * Q[fl["face"]])            # outward flux of the cell
            if fw_eff is None:
                eta_new = np.where(self.active, eta - dt * div / np.where(self.active, self.area, 1.0), eta_new)
            else:                                                                 # same source as the solve
                eta_new = np.where(self.active, eta - dt * div / np.where(self.active, self.area, 1.0) + dt * fw_eff,
                                   eta_new)
            # positivity guard on the final η (the recompute above undoes the clip applied to the CG solution)
            depth = self.H + eta_new
            neg = self.active & (depth < 0)
            self.last_clip_volume = float(np.sum(-depth[neg] * self.area[neg]))
            eta_new[neg] = -self.H[neg]
        return eta_new, u_new, v_new
