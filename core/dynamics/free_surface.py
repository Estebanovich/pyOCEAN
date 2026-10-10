"""
Implicit linear free surface (θ-method) for pyOCEAN — Phase 2.

Removes the external gravity-wave CFL limit Δt < Δx/√(gH) by treating the
surface pressure gradient and the barotropic divergence implicitly, as in
MITgcm (Marshall et al. 1997, §2; Campin et al. 2004).  With θ the implicit
weight (MITgcm ``implicSurfPress`` = ``implicDiv2DFlow`` = θ):

    u^{n+1} = û − gΔt [θ ∇η^{n+1} + (1 − θ) ∇η^n]                      (1)
    η^{n+1} = η^n − Δt ∇·{H [θ u^{n+1} + (1 − θ) u^n]}                  (2)

where û = u^n + Δt G^{n+½} is the velocity after all other (AB2) tendencies.
Substituting (1) into (2) with u** = û − gΔt(1 − θ)∇η^n gives the elliptic
problem for η^{n+1} (Dukowicz & Smith 1994, eq. 21, in finite-volume form):

    A_c η^{n+1} + gθ²Δt² Σ_f c_f (η_c − η_nb) =
        A_c η^n − Δt (1 − θ) D(u^n) − Δt θ D(u**)                      (3)

    c_f  = H_f Δy_f / Δx_f        face conductance        [m²]
    H_f  = Σ_k Δz_k hFac_{k,f}    rest depth at the face  [m]
    D(u) = Σ_faces ± Σ_k Δz_k hFac_{k,f} u_{k,f} Δy_f     net volume outflow [m³ s⁻¹]

A_c is the T-cell area.  The matrix of (3) is symmetric positive definite
(cell areas on the diagonal plus a weighted graph Laplacian), so it is
factorised once (or solved with preconditioned CG).  Then (1) gives u^{n+1}.

Boundaries
----------
* Closed faces (walls, land, domain edges) have H_f = 0: no flux and no
  coupling — the natural (Neumann) condition, so η is free on walls and
  seiche antinodes sit at the walls.
* Open faces (``open_u``/``open_v``) carry the prescribed normal velocity
  given in û; their flux enters the right-hand side and they receive no
  pressure correction.

θ = 1 (backward Euler, MITgcm default) damps gravity waves; θ = ½
(Crank–Nicolson) is neutral.  The free surface is linear (H at rest).

References
----------
Dukowicz, J.K. & Smith, R.D. (1994). Implicit free-surface method for the
    Bryan–Cox–Semtner ocean model. J. Geophys. Res. 99(C4), 7991–8014.
Marshall, J., Adcroft, A., Hill, C., Perelman, L. & Heisey, C. (1997).
    A finite-volume, incompressible Navier–Stokes model for studies of the
    ocean on parallel computers. J. Geophys. Res. 102(C3), 5753–5766.
Campin, J.-M., Adcroft, A., Hill, C. & Marshall, J. (2004). Conservation of
    properties in a free-surface model. Ocean Modelling 6, 221–244.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.sparse.linalg import LinearOperator, cg, factorized

G: float = 9.81   # m s⁻²


class ImplicitFreeSurface:
    """
    Assembled θ-method free-surface operator for a fixed grid and time step.

    Parameters
    ----------
    h_grid : xr.Dataset
        Horizontal grid (``dx_u``, ``dy_u``, ``dx_v``, ``dy_v``, ``area_t``).
    v_grid : xr.Dataset
        Vertical grid (``dz``).
    hfac : xr.Dataset
        Partial-cell fractions from ``core.grid.vertical.compute_hfac``
        (``hfac_t``, ``hfac_u``, ``hfac_v``).
    dt : float
        Time step [s].
    theta : float
        Implicit weight, ½ ≤ θ ≤ 1.  Default 1 (MITgcm default).
    g : float
        Gravitational acceleration [m s⁻²].
    open_u, open_v : np.ndarray of bool, optional
        Open faces, shapes (ny, nx+1) and (ny+1, nx), whose normal velocity is
        prescribed (taken from û unchanged).
    solver : {'direct', 'cg'}
        Sparse LU factorisation once, or Jacobi-preconditioned CG each step.
    cg_rtol : float
        Relative tolerance for 'cg' (MITgcm ``cg2dTargetResidual`` analogue).
    """

    def __init__(
        self,
        h_grid: xr.Dataset,
        v_grid: xr.Dataset,
        hfac: xr.Dataset,
        dt: float,
        theta: float = 1.0,
        g: float = G,
        open_u: np.ndarray | None = None,
        open_v: np.ndarray | None = None,
        solver: str = "direct",
        cg_rtol: float = 1e-12,
    ) -> None:
        if not 0.5 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0.5, 1] for stability, got {theta}")
        if solver not in ("direct", "cg"):
            raise ValueError(f"solver must be 'direct' or 'cg', got {solver!r}")
        self.dt, self.theta, self.g = float(dt), float(theta), float(g)
        self.solver, self.cg_rtol = solver, cg_rtol

        dz = v_grid["dz"].values
        self.hfac_u = hfac["hfac_u"].values
        self.hfac_v = hfac["hfac_v"].values
        self.wet_t = hfac["hfac_t"].values.sum(axis=0) > 0
        self.area = h_grid["area_t"].values
        self.dy_u, self.dx_u = h_grid["dy_u"].values, h_grid["dx_u"].values
        self.dx_v, self.dy_v = h_grid["dx_v"].values, h_grid["dy_v"].values
        ny, nx = self.area.shape

        # Layer thickness at faces, Δz_k hFac_{k,f}  [m]
        self.dzf_u = dz[:, None, None] * self.hfac_u
        self.dzf_v = dz[:, None, None] * self.hfac_v
        h_u, h_v = self.dzf_u.sum(axis=0), self.dzf_v.sum(axis=0)

        self.open_u = np.zeros((ny, nx + 1), bool) if open_u is None else np.asarray(open_u, bool)
        self.open_v = np.zeros((ny + 1, nx), bool) if open_v is None else np.asarray(open_v, bool)

        # Faces coupling two wet T-cells (interior, not open, wet)
        cu = np.zeros((ny, nx + 1), bool)
        cu[:, 1:-1] = self.wet_t[:, :-1] & self.wet_t[:, 1:] & (h_u[:, 1:-1] > 0)
        cv = np.zeros((ny + 1, nx), bool)
        cv[1:-1, :] = self.wet_t[:-1, :] & self.wet_t[1:, :] & (h_v[1:-1, :] > 0)
        self.coupled_u = cu & ~self.open_u
        self.coupled_v = cv & ~self.open_v

        cond_u = np.where(self.coupled_u, h_u * self.dy_u / self.dx_u, 0.0)
        cond_v = np.where(self.coupled_v, h_v * self.dx_v / self.dy_v, 0.0)

        # ---- Assemble SPD matrix of eq. (3) on wet cells -----------------
        index = -np.ones((ny, nx), int)
        index[self.wet_t] = np.arange(self.wet_t.sum())
        self._index = index
        a = self.g * self.theta ** 2 * self.dt ** 2
        rows, cols, vals = [], [], []

        def couple(c: np.ndarray, p: np.ndarray, q: np.ndarray) -> None:
            m = c > 0
            ip, iq, w = p[m], q[m], a * c[m]
            rows.extend([ip, iq, ip, iq])
            cols.extend([ip, iq, iq, ip])
            vals.extend([w, w, -w, -w])

        couple(cond_u[:, 1:-1], index[:, :-1], index[:, 1:])
        couple(cond_v[1:-1, :], index[:-1, :], index[1:, :])
        diag = np.arange(index.max() + 1)
        rows.append(diag)
        cols.append(diag)
        vals.append(self.area[self.wet_t])
        n = diag.size
        self.matrix = sp.csr_matrix(
            (np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(n, n))
        if solver == "direct":
            self._solve = factorized(self.matrix.tocsc())
        else:
            dinv = 1.0 / self.matrix.diagonal()
            self._precond = LinearOperator((n, n), matvec=lambda x: dinv * x)

    # ------------------------------------------------------------------
    def _grad(self, eta: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """∇η at coupled U- and V-faces [–]; zero elsewhere."""
        gx = np.zeros(self.hfac_u.shape[1:])
        gy = np.zeros(self.hfac_v.shape[1:])
        gx[:, 1:-1] = (eta[:, 1:] - eta[:, :-1]) / self.dx_u[:, 1:-1]
        gy[1:-1, :] = (eta[1:, :] - eta[:-1, :]) / self.dy_v[1:-1, :]
        return np.where(self.coupled_u, gx, 0.0), np.where(self.coupled_v, gy, 0.0)

    def volume_divergence(self, u: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Net volume outflow of each T-cell [m³ s⁻¹]:

            D = U_{i+1} − U_i + V_{j+1} − V_j,   U = Δy_u Σ_k Δz_k hFac_u u
        """
        U = self.dy_u * np.sum(self.dzf_u * u, axis=0)
        V = self.dx_v * np.sum(self.dzf_v * v, axis=0)
        return U[:, 1:] - U[:, :-1] + V[1:, :] - V[:-1, :]

    def step(
        self,
        eta: np.ndarray,
        u_n: np.ndarray,
        v_n: np.ndarray,
        u_hat: np.ndarray,
        v_hat: np.ndarray,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance η and correct the velocities with eqs. (1)–(3).

        Parameters
        ----------
        eta : (ny, nx)               η^n [m]
        u_n, v_n : 3-D               velocities at time n [m s⁻¹]
        u_hat, v_hat : 3-D           velocities after the other tendencies (û)

        Returns
        -------
        eta_new, u_new, v_new
            η^{n+1} (0 on land) and u^{n+1}, v^{n+1} (0 on dry faces).
        """
        th, dt, g = self.theta, self.dt, self.g
        gx, gy = self._grad(eta)
        u_ss = u_hat - g * dt * (1.0 - th) * gx[None]
        v_ss = v_hat - g * dt * (1.0 - th) * gy[None]

        rhs = (self.area * eta
               - dt * (1.0 - th) * self.volume_divergence(u_n, v_n)
               - dt * th * self.volume_divergence(u_ss, v_ss))[self.wet_t]
        if self.solver == "direct":
            x = self._solve(rhs)
        else:
            x, info = cg(self.matrix, rhs, x0=eta[self.wet_t], rtol=self.cg_rtol,
                         maxiter=10 * rhs.size, M=self._precond)
            if info != 0:
                raise RuntimeError(f"free-surface CG did not converge (info={info})")
        eta_new = np.zeros_like(eta, dtype=float)
        eta_new[self.wet_t] = x

        gx, gy = self._grad(eta_new)
        u_new = (u_ss - g * dt * th * gx[None]) * (self.hfac_u > 0)
        v_new = (v_ss - g * dt * th * gy[None]) * (self.hfac_v > 0)
        return eta_new, u_new, v_new
