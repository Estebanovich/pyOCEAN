"""
Finite-volume pressure-gradient force (Adcroft, Hallberg & Harrison 2008), Boussinesq, linear
equation of state ρ = ρ₀ − α (T − T₀), on any layer geometry (σ, z*, hybrid).

The force on layer k at a u-face is the contour integral of the pressure around the layer's
vertical section between the two cell centres, divided by its area:

    PFu_k = −(1/(ρ₀ h̄ Δx)) ∮ p dz,   h̄ = (h_L + h_R)/2

evaluated with Boole's 5-point rule in the vertical and along the layer (exact for a linear EOS
and a piecewise-linear T).  Only the baroclinic part is returned: the ρ₀ g η part is the
free-surface gradient of the barotropic solver (the force is linear in p and ∮(η − z) dz gives
exactly −g δ_x η).

Reference profile.  With ρ = ρ_ref(z) + ρ′, the pressure of ρ_ref depends on z only, so its
contour integral vanishes exactly (∮ P_ref dz = 0) and only the anomaly ρ′ is integrated, plus the
column-constant surface term g η (ρ_ref(η/2) − ρ₀).  For the linear EOS, ρ′ = −α (T − T_ref(z))
with T_ref(z) = T₀ + (ρ₀ − ρ_ref(z))/α.  The PLM reconstruction (White & Adcroft 2008; MOM6
pressure_reconstruction_plm) is applied to the anomaly T′ = T − T_ref(z_c) at the layer centres,
not to T: the curvature of the mean stratification then never enters the reconstruction error,
and a column equal to the reference gives no force at all.  (Subtracting ρ_ref only at the
quadrature points, as a constant-ρ₀ code would, changes nothing: ∮ P_ref dz = 0 again.)  MOM6
reconstructs T itself and only subtracts a constant ρ₀.

Bottom mass weighting (MOM6 MASS_WEIGHT_IN_PRESSURE_GRADIENT with MASS_WEIGHT_IN_PGF_VANISHED_ONLY,
MOM_density_integrals.F90:652-685): when one of the two layers is vanished (h ≤ h_vanished) and a
layer lies wholly below the neighbour's bed, the corner values of T′ used along the layer are
weighted towards the thicker cell, so the meaningless T of a vanished layer does not enter the
force.  It is not applied between two non-vanished layers: there it would mix values from
different heights and break the exactness for linear stratification on steep σ-layers.

MOM6 source (commit 0e9a561, src/): core/MOM_PressureForce_FV.F90:1146-1198 (interfaces),
:1250-1260 (surface anomaly), :1337-1343 (pa), :1532-1558 (intx_pa), :1793-1802 (force),
:2124-2149 (mass-weighting options); core/MOM_density_integrals.F90:579-623 (column integrals),
:642-751 (along the layer), :652-685 (mass weighting); ALE/MOM_ALE.F90:1512-1596 and
ALE/PLM_functions.F90:22-183 (PLM).

References: Adcroft, A., Hallberg, R. & Harrison, M. (2008). A finite volume discretization of the
pressure gradient force using analytic integration. Ocean Modell. 22, 106–113.
White, L. & Adcroft, A. (2008). A high-order finite volume remapping scheme for nonuniform grids:
the piecewise quartic method (PQM). J. Comput. Phys. 227, 7394–7422.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.vertical.remap import H_NEGLECT, _plm_monotonized_slope, _plm_slope_wa

H_VANISHED = 0.01     # [m] layers this thin count as vanished for the mass weighting (= ale.H_THIN_BED)


# Not disk-cached: calls remap._plm_slope_wa / _plm_monotonized_slope (Numba's cache only checks this file).
@njit
def pressure_edges_plm(h, T, T_bot, T_top):  # pragma: no cover
    """
    Bottom and top edge values of T in each layer (bed first) for the pressure integrals: limited
    PLM in the interior (White & Adcroft 2008; MOM6 pressure_reconstruction_plm, MOM_ALE.F90:1512-1596)
    and unlimited h2 extrapolation in the bed and top layers (PLM_functions.F90:164-183).
    """
    n = h.size
    if n == 1:
        T_bot[0] = T[0]
        T_top[0] = T[0]
        return
    slp = np.zeros(n)
    for k in range(1, n - 1):
        slp[k] = _plm_slope_wa(h[k - 1], h[k], h[k + 1], H_NEGLECT, T[k - 1], T[k], T[k + 1])
    for k in range(1, n - 1):
        m = _plm_monotonized_slope(T[k - 1], T[k], T[k + 1], slp[k - 1], slp[k], slp[k + 1])
        T_bot[k] = T[k] - 0.5 * m
        T_top[k] = T[k] + 0.5 * m
    e = (T[1] * h[0] + T[0] * h[1]) / (h[0] + h[1] + H_NEGLECT)
    T_top[0] = e
    T_bot[0] = 2.0 * T[0] - e
    e = (T[n - 1] * h[n - 2] + T[n - 2] * h[n - 1]) / (h[n - 2] + h[n - 1] + H_NEGLECT)
    T_bot[n - 1] = e
    T_top[n - 1] = 2.0 * T[n - 1] - e


@njit(parallel=True)
def _edges_columns(h, T, T_bot, T_top):  # pragma: no cover
    n, ny, nx = h.shape
    for c in prange(ny * nx):
        j, i = c // nx, c % nx
        tb = np.empty(n)
        tt = np.empty(n)
        pressure_edges_plm(np.ascontiguousarray(h[:, j, i]), np.ascontiguousarray(T[:, j, i]), tb, tt)
        T_bot[:, j, i] = tb
        T_top[:, j, i] = tt


@njit(inline="always")
def _boole_mean(dz, t_top, t_bot, alpha):  # pragma: no cover
    """(B, r₀, …, r₄): Boole mean of ρ′ = −α T′ over a layer and its 5 point values, top to bottom."""
    r0 = -alpha * t_top
    r1 = -alpha * (3.0 * t_top + t_bot) / 4.0
    r2 = -alpha * (t_top + t_bot) / 2.0
    r3 = -alpha * (t_top + 3.0 * t_bot) / 4.0
    r4 = -alpha * t_bot
    b = (7.0 * (r0 + r4) + 32.0 * (r1 + r3) + 12.0 * r2) / 90.0
    return b, r0, r1, r2, r3, r4


@njit(parallel=True)
def _fv_pgf_x(e, h, Tt, Tb, dx, pa0, rho0, alpha, g, mass_weight, h_van, out):  # pragma: no cover
    """
    Force on the interior u-faces (columns i−1, i).  e (N+1, ny, nx), h, Tt, Tb (N, ny, nx):
    SURFACE FIRST (MOM6 order), Tt/Tb edge values of the anomaly T′; pa0 (ny, nx) surface
    pressure anomaly [Pa]; dx (ny, nx+1) [m]; out (N, ny, nx+1) [m s⁻²].
    """
    n, ny, nx = h.shape
    for j in prange(ny):
        for i in range(1, nx):
            L = i - 1
            R = i
            paL = pa0[j, L]
            paR = pa0[j, R]
            ixpa = 0.5 * (paL + paR)
            for k in range(n):
                hL = h[k, j, L]
                hR = h[k, j, R]
                bL, a0, a1, _, a3, a4 = _boole_mean(hL, Tt[k, j, L], Tb[k, j, L], alpha)
                bR, c0, c1, _, c3, c4 = _boole_mean(hR, Tt[k, j, R], Tb[k, j, R], alpha)
                dpaL = g * hL * bL
                dpaR = g * hR * bR
                izL = 0.5 * g * hL * hL * (bL - (16.0 * (a3 - a1) + 7.0 * (a4 - a0)) / 90.0)
                izR = 0.5 * g * hR * hR * (bR - (16.0 * (c3 - c1) + 7.0 * (c4 - c0)) / 90.0)
                ttl, tbl, ttr, tbr = Tt[k, j, L], Tb[k, j, L], Tt[k, j, R], Tb[k, j, R]
                if mass_weight and (hL <= h_van or hR <= h_van):
                    hw = max(0.0, e[n, j, L] - e[k, j, R], e[n, j, R] - e[k, j, L])
                    if hw > 0.0:
                        a = hL + H_NEGLECT
                        c = hR + H_NEGLECT
                        hw *= ((a - c) / (a + c)) ** 2
                        iD = 1.0 / (hw * (a + c) + a * c)
                        ttl, ttr = (hw * c * Tt[k, j, R] + (hw * a + a * c) * Tt[k, j, L]) * iD, \
                            (hw * a * Tt[k, j, L] + (hw * c + a * c) * Tt[k, j, R]) * iD
                        tbl, tbr = (hw * c * Tb[k, j, R] + (hw * a + a * c) * Tb[k, j, L]) * iD, \
                            (hw * a * Tb[k, j, L] + (hw * c + a * c) * Tb[k, j, R]) * iD
                I1 = 0.0
                I2 = 0.0
                I3 = 0.0
                for m in range(1, 4):
                    wl = (4 - m) / 4.0
                    wr = 1.0 - wl
                    dzx = wl * hL + wr * hR
                    bx = _boole_mean(dzx, wl * ttl + wr * ttr, wl * tbl + wr * tbr, alpha)[0]
                    if m == 1:
                        I1 = g * dzx * bx
                    elif m == 2:
                        I2 = g * dzx * bx
                    else:
                        I3 = g * dzx * bx
                ixdpa = (7.0 * (dpaL + dpaR) + 32.0 * (I1 + I3) + 12.0 * I2) / 90.0
                out[k, j, i] = ((paL * hL + izL) - (paR * hR + izR) + (hR - hL) * ixpa
                                - (e[k + 1, j, R] - e[k + 1, j, L]) * ixdpa) \
                    * 2.0 / (rho0 * dx[j, i] * (hL + hR + H_NEGLECT))
                paL += dpaL
                paR += dpaR
                ixpa += ixdpa


def fv_pgf(h: np.ndarray, H: np.ndarray, eta: np.ndarray, T: np.ndarray, du: np.ndarray, dv: np.ndarray,
           rho0: float, alpha: float, t0: float, rho_ref, g: float, mass_weight: bool = True,
           h_vanished: float = H_VANISHED) -> tuple[np.ndarray, np.ndarray]:
    """
    Baroclinic finite-volume pressure-gradient force at u- and v-faces [m s⁻²] (module docstring).

    h, T : (N, ny, nx) layer thickness [m] and temperature [°C], k = 0 at the bed
    H, eta : (ny, nx) bed depth and surface height [m] (land: any finite value, masked by the caller)
    du (ny, nx+1), dv (ny+1, nx) : centre-to-centre distances across the faces [m]
    rho_ref : reference profile, any function z [m] → ρ_ref(z) [kg m⁻³] (it is evaluated only at
        the layer centres and at η/2), or None for ρ_ref = ρ₀ (no profile)
    h_vanished : [m] thickness below which a layer counts as vanished for the mass weighting
    Returns gx (N, ny, nx+1), gy (N, ny+1, nx) with zeros on the domain-edge faces.
    """
    h = np.ascontiguousarray(h, float)
    n, ny, nx = h.shape
    eta = np.asarray(eta, float)
    ref = (lambda z: np.full(np.shape(z), float(rho0))) if rho_ref is None else rho_ref   # noqa: E731
    zw = -np.asarray(H, float)[None] + np.concatenate([np.zeros((1, ny, nx)), np.cumsum(h, 0)], 0)
    zc = 0.5 * (zw[1:] + zw[:-1])
    # T′ = T − T_ref(z_c), T_ref = T₀ + (ρ₀ − ρ_ref)/α: then ρ′ = −α T′ (linear EOS)
    Ta = np.ascontiguousarray(np.asarray(T, float) - (t0 + (rho0 - np.asarray(ref(zc), float)) / alpha))
    tb, tt = np.empty_like(h), np.empty_like(h)
    _edges_columns(h, Ta, tb, tt)
    hs, Tt, Tb = h[::-1].copy(), tt[::-1].copy(), tb[::-1].copy()          # surface first
    e = np.ascontiguousarray(zw[::-1])                                    # e[0] = η, e[N] = −H (:1195-1198)
    # surface anomaly g ∫_0^η (ρ_ref − ρ₀) dz ≈ g η (ρ_ref(η/2) − ρ₀): constant down each column
    pa0 = g * eta * (np.asarray(ref(0.5 * eta), float) - rho0)
    gx = np.zeros((n, ny, nx + 1))
    _fv_pgf_x(e, hs, Tt, Tb, np.ascontiguousarray(du, float), pa0, rho0, alpha, g, mass_weight, h_vanished, gx)
    sw = lambda a: np.swapaxes(a, -1, -2)                     # noqa: E731  strided view, no copy (perf part 2)
    gyT = np.zeros((n, nx, ny + 1))
    _fv_pgf_x(sw(e), sw(hs), sw(Tt), sw(Tb), sw(dv), sw(pa0), rho0, alpha, g, mass_weight, h_vanished, gyT)
    return gx[::-1].copy(), np.swapaxes(gyT, -1, -2)[::-1].copy()


def fv_pgf_rho(h: np.ndarray, H: np.ndarray, eta: np.ndarray, rho: np.ndarray, du: np.ndarray, dv: np.ndarray,
               rho0: float, rho_ref, g: float, mass_weight: bool = True,
               h_vanished: float = H_VANISHED) -> tuple[np.ndarray, np.ndarray]:
    """
    Baroclinic finite-volume pressure-gradient force [m s⁻²] from the layer densities ρ (N2a: ρ(T, S)).

    Same scheme as ``fv_pgf`` (Adcroft et al. 2008; MOM6 MOM_PressureForce_FV, module docstring), in density
    units: the anomaly ρ′ = ρ − ρ_ref(z_c) at the layer centres is reconstructed with PLM and integrated with
    Boole's rule, instead of T′ = T − T_ref(z) with ρ′ = −α T′.  It never divides by α (a salt-only EOS has
    α = 0) and holds for any EOS whose layer value ρ is given.  The kernel is ``_fv_pgf_x`` with α = −1, so its
    point values −α ρ′ are ρ′ exactly.  For ρ = ρ₀ − α (T − T₀) it equals ``fv_pgf`` to round-off, not bit for bit
    (the limiter and the quadrature see ρ′ instead of T′; tests/test_n2a_density.py).

    h, rho : (N, ny, nx) layer thickness [m] and density [kg m⁻³], k = 0 at the bed
    H, eta : (ny, nx) bed depth and surface height [m]
    du (ny, nx+1), dv (ny+1, nx) : centre-to-centre distances across the faces [m]
    rho_ref : callable z [m] → ρ_ref(z) [kg m⁻³], or None for ρ_ref = ρ₀
    Returns gx (N, ny, nx+1), gy (N, ny+1, nx) with zeros on the domain-edge faces.
    """
    h = np.ascontiguousarray(h, float)
    n, ny, nx = h.shape
    eta = np.asarray(eta, float)
    ref = (lambda z: np.full(np.shape(z), float(rho0))) if rho_ref is None else rho_ref   # noqa: E731
    zw = -np.asarray(H, float)[None] + np.concatenate([np.zeros((1, ny, nx)), np.cumsum(h, 0)], 0)
    zc = 0.5 * (zw[1:] + zw[:-1])
    ra = np.ascontiguousarray(np.asarray(rho, float) - np.asarray(ref(zc), float))     # ρ′ = ρ − ρ_ref(z_c)
    rb, rt = np.empty_like(h), np.empty_like(h)
    _edges_columns(h, ra, rb, rt)
    hs, Rt, Rb = h[::-1].copy(), rt[::-1].copy(), rb[::-1].copy()          # surface first
    e = np.ascontiguousarray(zw[::-1])
    pa0 = g * eta * (np.asarray(ref(0.5 * eta), float) - rho0)            # surface anomaly, as fv_pgf
    gx = np.zeros((n, ny, nx + 1))
    _fv_pgf_x(e, hs, Rt, Rb, np.ascontiguousarray(du, float), pa0, rho0, -1.0, g, mass_weight, h_vanished, gx)
    sw = lambda a: np.swapaxes(a, -1, -2)                     # noqa: E731
    gyT = np.zeros((n, nx, ny + 1))
    _fv_pgf_x(sw(e), sw(hs), sw(Rt), sw(Rb), sw(dv), sw(pa0), rho0, -1.0, g, mass_weight, h_vanished, gyT)
    return gx[::-1].copy(), np.swapaxes(gyT, -1, -2)[::-1].copy()
