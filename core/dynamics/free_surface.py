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

* Flather faces (``flather_u``/``flather_v``, domain edges only) radiate
  the barotropic mode with the Flather (1976) condition

      ū_out = ū_ext,out + √(g/H) (η_c − η_ext)                          (4)

  (ū_out positive outward, η_c the adjacent interior cell).  Applied
  θ-implicitly, its transport Δy H ū_out adds Δt θ Δy √(gH) to the
  diagonal of (3), so the matrix stays SPD.  External η, ū (e.g. tides +
  parent model) are given at each step; the baroclinic part of û is kept.

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
Flather, R.A. (1976). A tidal model of the north-west European continental
    shelf. Mém. Soc. R. Sci. Liège 6(10), 141–164.
Carter, G.S. & Merrifield, M.A. (2007). Open boundary conditions for
    regional tidal simulations. Ocean Modelling 18, 194–209.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp
import xarray as xr
from scipy.sparse.linalg import LinearOperator, cg, factorized

G: float = 9.81   # m s⁻²


def open_boundary_hfac(hfac: xr.Dataset, sides: list[str]) -> xr.Dataset:
    """
    Open the domain-edge faces of the given sides.

    ``core.grid.vertical.compute_hfac`` closes every domain edge (hFac = 0).
    For an open boundary the edge face takes the wet fraction of its
    adjacent T-cell, so that transport can cross it.

    Parameters
    ----------
    hfac : xr.Dataset
        Output of ``compute_hfac``.
    sides : list of {'west', 'east', 'south', 'north'}

    Returns
    -------
    xr.Dataset
        Copy with the edge ``hfac_u``/``hfac_v`` of the open sides filled.
    """
    out = hfac.copy(deep=True)
    hu, hv, ht = out["hfac_u"].values, out["hfac_v"].values, out["hfac_t"].values
    for side in sides:
        if side == "west":
            hu[:, :, 0] = ht[:, :, 0]
        elif side == "east":
            hu[:, :, -1] = ht[:, :, -1]
        elif side == "south":
            hv[:, 0, :] = ht[:, 0, :]
        elif side == "north":
            hv[:, -1, :] = ht[:, -1, :]
        else:
            raise ValueError(f"unknown side {side!r}")
    return out


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
    sal_beta : float
        Scalar self-attraction-and-loading factor β (HYCOM ``tidsal``): the
        surface pressure gradient uses g(1 − β)∇η.  Default 0.
    flather_u, flather_v : np.ndarray of bool, optional
        Domain-edge faces with the Flather radiation condition (4); their
        hFac must be open (``open_boundary_hfac``).
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
        flather_u: np.ndarray | None = None,
        flather_v: np.ndarray | None = None,
        sal_beta: float = 0.0,
        solver: str = "direct",
        cg_rtol: float = 1e-12,
    ) -> None:
        if not 0.5 <= theta <= 1.0:
            raise ValueError(f"theta must be in [0.5, 1] for stability, got {theta}")
        if solver not in ("direct", "cg"):
            raise ValueError(f"solver must be 'direct' or 'cg', got {solver!r}")
        if not 0.0 <= sal_beta < 1.0:
            raise ValueError(f"sal_beta must be in [0, 1), got {sal_beta}")
        self.dt, self.theta, self.g = float(dt), float(theta), float(g)
        self.g_pgf = self.g * (1.0 - sal_beta)       # gravity in the surface pressure gradient
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

        # Flather faces: (face mask, adjacent-cell index arrays, outward sign, Δy, H)
        self._flather = []
        for mask, axis in ((flather_u, "u"), (flather_v, "v")):
            if mask is None:
                continue
            mask = np.asarray(mask, bool)
            interior = mask.copy()
            if axis == "u":
                interior[:, [0, -1]] = False
            else:
                interior[[0, -1], :] = False
            if interior.any():
                raise ValueError("Flather faces must lie on the domain edge")
            h_f = h_u if axis == "u" else h_v
            if np.any(mask & (h_f <= 0)):
                raise ValueError("Flather faces must be wet: open them with open_boundary_hfac")
            for edge in (0, -1):
                sel = np.zeros_like(mask)
                if axis == "u":
                    sel[:, edge] = mask[:, edge]
                    jj, ii = np.nonzero(sel)
                    cj, ci = jj, (ii if edge == 0 else ii - 1)
                    length = self.dy_u[jj, ii]
                else:
                    sel[edge, :] = mask[edge, :]
                    jj, ii = np.nonzero(sel)
                    cj, ci = (jj if edge == 0 else jj - 1), ii
                    length = self.dx_v[jj, ii]
                if jj.size:
                    self._flather.append(dict(axis=axis, face=(jj, ii), cell=(cj, ci),
                                              sign=-1.0 if edge == 0 else 1.0,
                                              length=length, H=h_f[jj, ii]))
        self._flather_u = np.zeros((ny, nx + 1), bool) if flather_u is None else np.asarray(flather_u, bool)
        self._flather_v = np.zeros((ny + 1, nx), bool) if flather_v is None else np.asarray(flather_v, bool)

        cond_u = np.where(self.coupled_u, h_u * self.dy_u / self.dx_u, 0.0)
        cond_v = np.where(self.coupled_v, h_v * self.dx_v / self.dy_v, 0.0)

        # ---- Assemble SPD matrix of eq. (3) on wet cells -----------------
        index = -np.ones((ny, nx), int)
        index[self.wet_t] = np.arange(self.wet_t.sum())
        self._index = index
        a = self.g_pgf * self.theta ** 2 * self.dt ** 2
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
        for fl in self._flather:                     # implicit Flather: Δt θ Δy √(gH) on the diagonal
            rows.append(index[fl["cell"]])
            cols.append(index[fl["cell"]])
            vals.append(self.dt * self.theta * fl["length"] * np.sqrt(self.g * fl["H"]))
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
        flather_eta: np.ndarray | None = None,
        flather_ubar: np.ndarray | None = None,
        flather_vbar: np.ndarray | None = None,
        eta_tendency: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Advance η and correct the velocities with eqs. (1)–(4).

        Parameters
        ----------
        eta : (ny, nx)               η^n [m]
        u_n, v_n : 3-D               velocities at time n [m s⁻¹]
        u_hat, v_hat : 3-D           velocities after the other tendencies (û)
        flather_eta : (ny, nx+1) or (ny+1, nx) arrays, optional
            External η at the Flather faces [m] (one array per face type is
            accepted: pass a tuple (eta_u, eta_v) when both u- and v-faces
            are Flather).  Valid at mid-step for θ = ½.
        flather_ubar, flather_vbar : optional
            External depth-mean velocity at the Flather u-/v-faces [m s⁻¹],
            in model sign convention (positive east / north).
        eta_tendency : (ny, nx), optional
            Explicit source of η [m s⁻¹] added to the continuity equation, e.g.
            the nonlinear flux divergence −∇·(η u) (``core.dynamics.nonlinear``).

        Returns
        -------
        eta_new, u_new, v_new
            η^{n+1} (0 on land) and u^{n+1}, v^{n+1} (0 on dry faces).
        """
        th, dt, g, gp = self.theta, self.dt, self.g, self.g_pgf
        gx, gy = self._grad(eta)
        u_ss = u_hat - gp * dt * (1.0 - th) * gx[None]
        v_ss = v_hat - gp * dt * (1.0 - th) * gy[None]

        # Flather faces carry their own transport: drop them from D(·)
        fu, fv = self._flather_u[None], self._flather_v[None]
        rhs_full = (self.area * eta
                    - dt * (1.0 - th) * self.volume_divergence(np.where(fu, 0.0, u_n), np.where(fv, 0.0, v_n))
                    - dt * th * self.volume_divergence(np.where(fu, 0.0, u_ss), np.where(fv, 0.0, v_ss)))
        if eta_tendency is not None:
            rhs_full = rhs_full + dt * self.area * np.where(self.wet_t, eta_tendency, 0.0)
        ext = self._flather_external(flather_eta, flather_ubar, flather_vbar)
        for fl, (e_ext, ubar_ext) in zip(self._flather, ext):
            # Q = Δy H ū_ext,out + Δy √(gH) (η_c^θ − η_ext); the θ η^{n+1} part is in the matrix
            c = fl["length"] * np.sqrt(g * fl["H"])
            q_known = fl["length"] * fl["H"] * fl["sign"] * ubar_ext + c * ((1.0 - th) * eta[fl["cell"]] - e_ext)
            np.add.at(rhs_full, fl["cell"], -dt * q_known)
        rhs = rhs_full[self.wet_t]
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
        u_new = (u_ss - gp * dt * th * gx[None]) * (self.hfac_u > 0)
        v_new = (v_ss - gp * dt * th * gy[None]) * (self.hfac_v > 0)

        # Flather faces: depth mean from (4) at n+1, baroclinic part of û kept
        for fl, (e_ext, ubar_ext) in zip(self._flather, ext):
            ubar = ubar_ext + fl["sign"] * np.sqrt(g / fl["H"]) * (eta_new[fl["cell"]] - e_ext)
            target, dzf, src = (u_new, self.dzf_u, u_ss) if fl["axis"] == "u" else (v_new, self.dzf_v, v_ss)
            j, i = fl["face"]
            col = src[:, j, i]
            mean = np.sum(dzf[:, j, i] * col, axis=0) / fl["H"]
            target[:, j, i] = (col - mean + ubar) * (dzf[:, j, i] > 0)
        return eta_new, u_new, v_new

    def _flather_external(self, eta_ext, ubar_ext, vbar_ext) -> list[tuple[np.ndarray, np.ndarray]]:
        """External (η, ū_normal) at each Flather face group."""
        if not self._flather:
            return []
        if eta_ext is None:
            raise ValueError("Flather faces need flather_eta (and the external depth-mean velocity)")
        out = []
        for fl in self._flather:
            if isinstance(eta_ext, tuple):
                e = eta_ext[0] if fl["axis"] == "u" else eta_ext[1]
            else:
                e = eta_ext
            vel = ubar_ext if fl["axis"] == "u" else vbar_ext
            vel = np.zeros_like(e) if vel is None else vel
            out.append((e[fl["face"]], vel[fl["face"]]))
        return out
