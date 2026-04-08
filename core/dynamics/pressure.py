"""
Hydrostatic pressure and pressure gradient force — pyOCEAN Phase 1.

Implements the pressure computation chain for a z-coordinate Boussinesq model:

  1. ``hydrostatic_pressure`` — integrate ∂p/∂z = −ρg downward from the
     free surface to produce the 3-D pressure field p(k, j_t, i_t).

  2. ``pressure_gradient_x`` / ``pressure_gradient_y`` — second-order centred
     finite differences of p on the C-grid, yielding the pressure gradient
     force (PGF) at U- and V-faces respectively.

  3. ``surface_pressure_cg`` — optional 2-D conjugate-gradient (CG) solver for
     the implicit free-surface Helmholtz equation, needed when an implicit
     time-discretisation of the barotropic mode is desired (Phase 2+).
     In Phase 1 the free surface is advanced explicitly, so this function is
     provided for completeness and forward compatibility.

Hydrostatic pressure discretisation
------------------------------------
Following Marshall et al. (1997) and Vallis (2006 §2.4), we split the total
pressure into a barotropic and a baroclinic part:

    p(k) = p_bar(j,i) + p_hyd(k,j,i)

where the barotropic surface pressure is:

    p_bar = ρ₀ g η                                        [Pa]

and the baroclinic hydrostatic contribution at layer-interface k_f (counting
downward from k_f = 0 at the surface) is built by cumulative sum:

    p_face[0]   = 0
    p_face[k+1] = p_face[k] + ρ[k] g dz[k]              [Pa]    (eq. A1)

The pressure at the T-cell centre (layer k) is:

    p_centre[k] = p_bar + p_face[k] + 0.5 ρ[k] g dz[k]  [Pa]    (eq. A2)

This matches the MITgcm convention (Marshall et al. 1997, Appendix A).

Pressure gradient force
-----------------------
On an Arakawa C-grid, the PGF at U- and V-faces is:

    PGF_x[k, j, i+½] = −(1/ρ₀) (p[k,j,i+1] − p[k,j,i]) / Δx   [m/s²]
    PGF_y[k, j+½, i] = −(1/ρ₀) (p[k,j+1,i] − p[k,j,i]) / Δy   [m/s²]

These are second-order centred in space, consistent with the C-grid staggering
used in ``core/grid/horizontal.py``.

References
----------
Marshall et al. (1997) — A finite-volume, incompressible Navier-Stokes model
    for studies of the ocean on parallel computers. JGR 102(C3).
Vallis (2006) — Atmospheric and Oceanic Fluid Dynamics. §2.4, §14.1.
Durran (2010) — Numerical Methods for Fluid Dynamics. §4.2.
"""

from __future__ import annotations

import numpy as np
import xarray as xr
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import cg as sparse_cg

# ---------------------------------------------------------------------------
# Physical constants
# ---------------------------------------------------------------------------

G: float = 9.81      # m/s²   gravitational acceleration  (Vallis 2006 §1.2)
RHO_0: float = 1025.0  # kg/m³  Boussinesq reference density


# ---------------------------------------------------------------------------
# 1. Hydrostatic pressure
# ---------------------------------------------------------------------------

def hydrostatic_pressure(
    rho: xr.DataArray,
    eta: xr.DataArray,
    z_grid: xr.Dataset,
) -> xr.DataArray:
    """
    Compute the full hydrostatic pressure at T-cell centres.

    Integrates the hydrostatic balance downward from the free surface:

        ∂p/∂z = −ρ g                              (Vallis 2006, eq. 2.75)

    Discretised on z-levels (eqs. A1–A2 in the module docstring):

        p_face[0]   = 0
        p_face[k+1] = p_face[k] + ρ[k] · g · dz[k]
        p[k]        = ρ₀ g η + p_face[k] + ½ ρ[k] · g · dz[k]

    Parameters
    ----------
    rho : xr.DataArray, dims (k, j_t, i_t)
        In-situ density ρ [kg/m³] from ``physics.eos.compute_density``.
    eta : xr.DataArray, dims (j_t, i_t)
        Sea surface height η [m].
    z_grid : xr.Dataset
        Vertical grid providing ``dz`` [m] (positive layer thicknesses).

    Returns
    -------
    xr.DataArray, dims (k, j_t, i_t)
        Total hydrostatic pressure p [Pa] at T-cell centres.

    Notes
    -----
    * The surface pressure p_bar = ρ₀ g η must be included here, not added
      separately later, so that the pressure gradient correctly accounts for
      the free-surface slope.
    * The Boussinesq approximation means we use ρ₀ in the surface term but
      the full ρ in the baroclinic integral.  This is consistent with
      Marshall et al. (1997).

    Examples
    --------
    >>> p = hydrostatic_pressure(rho, state.eta, vgrid)
    >>> p.dims
    ('k', 'j_t', 'i_t')
    """
    rho_np = rho.values                    # (nz, ny, nx)
    eta_np = eta.values                    # (ny, nx)
    dz_np  = z_grid["dz"].values           # (nz,)  positive [m]

    nz, ny, nx = rho_np.shape

    # ---- Barotropic surface pressure p_bar = ρ₀ g η ----------------------
    # Shape: (1, ny, nx) for broadcasting
    p_bar = RHO_0 * G * eta_np[np.newaxis, :, :]   # (1, ny, nx) [Pa]

    # ---- Baroclinic cumulative integral from surface ----------------------
    # p_face[k]   = pressure at the top interface of layer k  (k=0: surface)
    # p_face[k+1] = p_face[k] + ρ[k] g dz[k]
    #
    # Build p_face (nz+1, ny, nx) by cumulative sum:
    #   p_face[0] = 0
    #   p_face[k] = Σ_{k'=0}^{k-1} ρ[k'] g dz[k']
    #
    # np.cumsum gives the running total shifted by one: use [:-1] for top faces.
    dz_3d = dz_np[:, np.newaxis, np.newaxis]          # (nz, 1, 1)
    layer_dp = rho_np * G * dz_3d                      # (nz, ny, nx) [Pa]

    # p_face[k] = cumsum up to (not including) k → prepend a zero row
    p_face_top = np.concatenate(
        [np.zeros((1, ny, nx)), np.cumsum(layer_dp, axis=0)[:-1, :, :]],
        axis=0,
    )                                                  # (nz, ny, nx)

    # ---- Pressure at layer centres (eq. A2) --------------------------------
    # p[k] = p_bar + p_face[k] + ½ ρ[k] g dz[k]
    p_np = p_bar + p_face_top + 0.5 * layer_dp        # (nz, ny, nx) [Pa]

    return xr.DataArray(
        p_np,
        dims=rho.dims,
        coords=rho.coords,
        attrs={
            "units": "Pa",
            "long_name": "Hydrostatic pressure",
            "description": "p_bar + baroclinic integral (Marshall et al. 1997, eq. A2)",
        },
    )


# ---------------------------------------------------------------------------
# 2. Pressure gradient force
# ---------------------------------------------------------------------------

def pressure_gradient_x(
    p: xr.DataArray,
    h_grid: xr.Dataset,
) -> xr.DataArray:
    """
    Zonal pressure gradient force at U-faces.

    Computes the acceleration term in the zonal momentum equation:

        PGF_x = −(1/ρ₀) ∂p/∂x                   (Vallis 2006, eq. 14.12a)

    Discretised as a second-order centred difference on the C-grid:

        PGF_x[k, j, i+½] = −(1/ρ₀) (p[k,j,i+1] − p[k,j,i]) / Δx

    where i+½ is the U-face between T-cells i and i+1.

    Parameters
    ----------
    p : xr.DataArray, dims (k, j_t, i_t)
        Hydrostatic pressure [Pa] at T-cell centres.
    h_grid : xr.Dataset
        Horizontal grid providing ``dx`` [m] (uniform or non-uniform).

    Returns
    -------
    xr.DataArray, dims (k, j_t, i_u)
        Zonal PGF [m/s²] at U-faces (interior faces only; boundary faces = 0).

    Notes
    -----
    The result array has shape (nz, ny, nx+1).  The two boundary columns
    (i_u = 0 and i_u = nx) are set to zero — they represent the domain walls
    where no pressure gradient is defined.  The dynamics module is responsible
    for applying the correct boundary condition (u = 0 at solid walls).
    """
    p_np   = p.values                       # (nz, ny, nx)
    # dx_u(j_t, i_u): distance between T-cell centres across each U-face [m]
    dx_u   = h_grid["dx_u"].values          # (ny, nx+1)

    nz, ny, nx = p_np.shape

    # PGF_x at interior U-faces: i_u = 1 … nx-1
    # Result shape: (nz, ny, nx+1); boundary faces (i_u=0 and i_u=nx) = 0
    pgf_x = np.zeros((nz, ny, nx + 1))

    # PGF_x[:, :, i_u] = -(p[:,:,i_u] - p[:,:,i_u-1]) / dx_u[:,i_u] / ρ₀
    # Interior slice i_u = 1…nx-1 maps to T-cell pairs (i_u-1, i_u):
    pgf_x[:, :, 1:nx] = (
        -(p_np[:, :, 1:] - p_np[:, :, :-1])
        / dx_u[np.newaxis, :, 1:nx]       # (1, ny, nx-1)
        / RHO_0
    )
    # Boundary faces i_u=0 (west) and i_u=nx (east) stay zero — solid wall.

    # ---- Coordinates for U-faces ------------------------------------------
    i_u_coords = h_grid.coords["i_u"].values    # (nx+1,)

    return xr.DataArray(
        pgf_x,
        dims=["k", "j_t", "i_u"],
        coords={"k": p.coords["k"], "j_t": p.coords["j_t"], "i_u": i_u_coords},
        attrs={
            "units": "m/s2",
            "long_name": "Zonal pressure gradient force",
            "description": "-(1/rho0) dp/dx at U-faces (Marshall et al. 1997)",
        },
    )


def pressure_gradient_y(
    p: xr.DataArray,
    h_grid: xr.Dataset,
) -> xr.DataArray:
    """
    Meridional pressure gradient force at V-faces.

    Computes the acceleration term in the meridional momentum equation:

        PGF_y = −(1/ρ₀) ∂p/∂y                   (Vallis 2006, eq. 14.12b)

    Discretised as a second-order centred difference on the C-grid:

        PGF_y[k, j+½, i] = −(1/ρ₀) (p[k,j+1,i] − p[k,j,i]) / Δy

    Parameters
    ----------
    p : xr.DataArray, dims (k, j_t, i_t)
        Hydrostatic pressure [Pa] at T-cell centres.
    h_grid : xr.Dataset
        Horizontal grid providing ``dy`` [m].

    Returns
    -------
    xr.DataArray, dims (k, j_v, i_t)
        Meridional PGF [m/s²] at V-faces (interior faces only; boundary = 0).

    Notes
    -----
    Analogous to ``pressure_gradient_x``.  The northern and southern boundary
    rows (j_v = 0 and j_v = ny) are set to zero.
    """
    p_np  = p.values                        # (nz, ny, nx)
    # dy_v(j_v, i_t): distance between T-cell centres across each V-face [m]
    dy_v  = h_grid["dy_v"].values           # (ny+1, nx)

    nz, ny, nx = p_np.shape

    pgf_y = np.zeros((nz, ny + 1, nx))

    # Interior faces j_v = 1…ny-1 (between T-cell rows j_v-1 and j_v):
    pgf_y[:, 1:ny, :] = (
        -(p_np[:, 1:, :] - p_np[:, :-1, :])
        / dy_v[np.newaxis, 1:ny, :]       # (1, ny-1, nx)
        / RHO_0
    )
    # Boundary faces j_v=0 (south) and j_v=ny (north) stay zero — solid wall.

    j_v_coords = h_grid.coords["j_v"].values    # (ny+1,)

    return xr.DataArray(
        pgf_y,
        dims=["k", "j_v", "i_t"],
        coords={"k": p.coords["k"], "j_v": j_v_coords, "i_t": p.coords["i_t"]},
        attrs={
            "units": "m/s2",
            "long_name": "Meridional pressure gradient force",
            "description": "-(1/rho0) dp/dy at V-faces (Marshall et al. 1997)",
        },
    )


# ---------------------------------------------------------------------------
# 3. 2-D surface pressure solver (implicit free surface)
# ---------------------------------------------------------------------------

def surface_pressure_cg(
    div_bar: np.ndarray,
    H: np.ndarray,
    dx: float,
    dy: float,
    dt: float,
    tol: float = 1e-8,
    maxiter: int = 1000,
) -> np.ndarray:
    """
    Solve the implicit free-surface Helmholtz equation for η^{n+1}.

    In the implicit (or semi-implicit) free-surface scheme, advancing η by
    one time step Δt requires solving (Dukowicz & Smith 1994, eq. 21):

        ∇·(H ∇η^{n+1}) − η^{n+1} / (g Δt²) = −η^n / (g Δt²) − ∇·(H ū^*)

    Rearranged with right-hand side RHS = div_bar (the depth-integrated
    barotropic divergence of the predictor velocity ū*):

        ∇·(H ∇η) − η / (g Δt²) = RHS          (Helmholtz equation)

    For a uniform depth H and uniform grid spacing (dx = dy), this reduces to
    a modified 2-D Poisson equation.  The system is assembled as a sparse
    linear system and solved by the conjugate gradient method.

    Parameters
    ----------
    div_bar : np.ndarray, shape (ny, nx)
        Depth-integrated barotropic divergence [m/s] of the predictor velocity.
        RHS of the Helmholtz equation (see above).
    H : np.ndarray, shape (ny, nx)
        Total water column depth H = h_bath + η [m].  For Phase 1 with a flat
        bottom this is constant.
    dx : float
        Zonal grid spacing [m].  (Uniform grid assumed in Phase 1.)
    dy : float
        Meridional grid spacing [m].
    dt : float
        Time step [s].
    tol : float, optional
        Convergence tolerance for the CG solver. Default 1e-8.
    maxiter : int, optional
        Maximum CG iterations. Default 1000.

    Returns
    -------
    np.ndarray, shape (ny, nx)
        Updated sea surface height η^{n+1} [m].

    Raises
    ------
    RuntimeError
        If the CG solver fails to converge within ``maxiter`` iterations.

    Notes
    -----
    * Dirichlet boundary conditions η = 0 are applied at all domain edges,
      consistent with a closed-basin configuration in Phase 1.
    * For Phase 1 validation (explicit free surface) this solver is not called
      in the main timestepper; it is provided for Phase 2+ where implicit time
      discretisation is needed to relax the CFL constraint on gravity waves.
    * The sparse assembly is O(ny × nx) and the CG iteration is O(N log N)
      per time step for well-conditioned systems.

    References
    ----------
    Dukowicz & Smith (1994) — Implicit free-surface method for the Bryan-Cox-
        Semtner ocean model. JGR 99(C4), 7991–8014.
    Marshall et al. (1997) — A finite-volume, incompressible Navier-Stokes
        model. JGR 102(C3), 5753–5766.
    """
    ny, nx = div_bar.shape
    N = ny * nx

    def idx(j, i):
        """Flat index for cell (j, i) in row-major order."""
        return j * nx + i

    # ---- Assemble sparse Helmholtz matrix A --------------------------------
    # For a cell (j, i) the stencil is:
    #
    #   A[p,p]     = −(H_e/dx² + H_w/dx² + H_n/dy² + H_s/dy²) − 1/(g dt²)
    #   A[p,p±1]   = +H_{e,w} / dx²      (zonal neighbours)
    #   A[p,p±nx]  = +H_{n,s} / dy²      (meridional neighbours)
    #
    # where H_{e,w,n,s} is the depth at the cell face (averaged from neighbours).
    # Dirichlet BC: boundary rows/cols fixed at 0 (η=0 on all walls).

    inv_g_dt2 = 1.0 / (G * dt ** 2)
    idx2_x = 1.0 / dx ** 2
    idx2_y = 1.0 / dy ** 2

    A = lil_matrix((N, N), dtype=float)
    b = np.zeros(N, dtype=float)

    for j in range(ny):
        for i in range(nx):
            p = idx(j, i)

            # Boundary cells: enforce η = 0 (Dirichlet)
            if j == 0 or j == ny - 1 or i == 0 or i == nx - 1:
                A[p, p] = 1.0
                b[p]    = 0.0
                continue

            # Face-averaged depths (arithmetic mean between neighbours)
            H_e = 0.5 * (H[j, i] + H[j, i + 1])
            H_w = 0.5 * (H[j, i] + H[j, i - 1])
            H_n = 0.5 * (H[j, i] + H[j + 1, i])
            H_s = 0.5 * (H[j, i] + H[j - 1, i])

            # Diagonal
            A[p, p] = -(H_e + H_w) * idx2_x - (H_n + H_s) * idx2_y - inv_g_dt2

            # Off-diagonal (zonal)
            A[p, idx(j, i + 1)] = H_e * idx2_x
            A[p, idx(j, i - 1)] = H_w * idx2_x

            # Off-diagonal (meridional)
            A[p, idx(j + 1, i)] = H_n * idx2_y
            A[p, idx(j - 1, i)] = H_s * idx2_y

            b[p] = div_bar[j, i]

    # ---- Solve with conjugate gradient ------------------------------------
    A_csr = A.tocsr()
    eta_flat, info = sparse_cg(A_csr, b, rtol=tol, maxiter=maxiter)

    if info != 0:
        raise RuntimeError(
            f"surface_pressure_cg did not converge (scipy.sparse.linalg.cg "
            f"info={info}, tol={tol}, maxiter={maxiter})."
        )

    return eta_flat.reshape(ny, nx)
