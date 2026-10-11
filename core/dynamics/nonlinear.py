"""
Nonlinear shallow-water terms on the Arakawa C grid (explicit part).

Depth-integrated continuity with the total depth H + η,

    ∂η/∂t = −∇·((H + η) u) = −∇·(H u)  − ∇·(η u)                (Vallis 2006, eq. 3.13)
                              implicit      explicit (this module)

so the linear implicit free-surface operator (``ImplicitFreeSurface``) is kept
and the nonlinear volume flux enters as a source ``eta_tendency``.  Momentum
advection in advective form with centred differences,

    (u·∇)u = u ∂u/∂x + v ∂u/∂y ,   (u·∇)v = u ∂v/∂x + v ∂v/∂y,

second order (or third-order upwind-biased, ``scheme="upwind3"``), metric
(curvature) terms u v tanφ / R neglected (|u|/R ≪ f at
regional scales), and a Laplacian viscosity A_h ∇²u with free-slip walls to
control the grid-scale noise that centred advection with Adams–Bashforth 2
lets grow (Durran 2010, §3.3).  A term whose stencil reaches a dry face is set
to zero (no extrapolation into land).

References
----------
Vallis, G.K. (2006). Atmospheric and Oceanic Fluid Dynamics, ch. 3.
Durran, D.R. (2010). Numerical Methods for Fluid Dynamics, 2nd ed.
Leonard, B.P. (1979). A stable and accurate convective modelling procedure
    based on quadratic upstream interpolation. Comput. Methods Appl. Mech.
    Eng. 19, 59–98.
"""

from __future__ import annotations

import numpy as np
import xarray as xr


def cell_to_faces(field: np.ndarray, wet: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Cell values at u- and v-faces: mean of the adjacent wet cells (one cell at
    coasts and domain edges, 0 if none).  Returns (ny, nx+1) and (ny+1, nx).
    """
    f = np.where(wet, np.asarray(field, float), 0.0)
    w = wet.astype(float)
    ny, nx = f.shape
    s_u, n_u = np.zeros((ny, nx + 1)), np.zeros((ny, nx + 1))
    s_u[:, :-1] += f
    s_u[:, 1:] += f
    n_u[:, :-1] += w
    n_u[:, 1:] += w
    s_v, n_v = np.zeros((ny + 1, nx)), np.zeros((ny + 1, nx))
    s_v[:-1] += f
    s_v[1:] += f
    n_v[:-1] += w
    n_v[1:] += w
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(n_u > 0, s_u / n_u, 0.0), np.where(n_v > 0, s_v / n_v, 0.0)


def nonlinear_continuity(eta: np.ndarray, u: np.ndarray, v: np.ndarray, h_grid: xr.Dataset,
                         wet_t: np.ndarray) -> np.ndarray:
    """
    −∇·(η u) [m s⁻¹] in finite-volume form: face fluxes η_f u Δy (η_f the mean
    of the adjacent wet cells), divided by the cell area; 0 on land.  Exactly
    volume conserving in a closed basin.  u, v are depth-mean velocities on
    the faces (0 on dry faces).
    """
    eta_u, eta_v = cell_to_faces(eta, wet_t)
    fu = eta_u * u * h_grid["dy_u"].values
    fv = eta_v * v * h_grid["dx_v"].values
    div = (fu[:, 1:] - fu[:, :-1] + fv[1:, :] - fv[:-1, :]) / h_grid["area_t"].values
    return np.where(wet_t, -div, 0.0)


def _pair(f: np.ndarray, wet: np.ndarray, axis: int, k: int = 1) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Neighbours f[−k], f[+k] along ``axis`` and their wet masks (False beyond the edges)."""
    lo, hi = np.zeros_like(f), np.zeros_like(f)
    wlo, whi = np.zeros(f.shape, bool), np.zeros(f.shape, bool)
    sl = [slice(None)] * 2
    a, b = list(sl), list(sl)
    a[axis], b[axis] = slice(k, None), slice(None, -k)
    lo[tuple(a)], wlo[tuple(a)] = f[tuple(b)], wet[tuple(b)]
    hi[tuple(b)], whi[tuple(b)] = f[tuple(a)], wet[tuple(a)]
    return lo, hi, wlo, whi


def _v_at_u(v: np.ndarray) -> np.ndarray:
    out = np.zeros((v.shape[0] - 1, v.shape[1] + 1))
    out[:, 1:-1] = 0.25 * (v[:-1, :-1] + v[1:, :-1] + v[:-1, 1:] + v[1:, 1:])
    return out


def _u_at_v(u: np.ndarray) -> np.ndarray:
    out = np.zeros((u.shape[0] + 1, u.shape[1] - 1))
    out[1:-1, :] = 0.25 * (u[:-1, :-1] + u[:-1, 1:] + u[1:, :-1] + u[1:, 1:])
    return out


def _spacings_u(h_grid: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distances from a u-point to its W/E and S/N u neighbours."""
    dx_t, dy_u = h_grid["dx_t"].values, h_grid["dy_u"].values
    ny, nxu = dy_u.shape
    hw, he = np.full((ny, nxu), np.nan), np.full((ny, nxu), np.nan)
    hw[:, 1:] = dx_t                                   # u[i-1] → u[i] spans cell i-1
    he[:, :-1] = dx_t                                  # u[i] → u[i+1] spans cell i
    hs, hn = np.full((ny, nxu), np.nan), np.full((ny, nxu), np.nan)
    hs[1:] = 0.5 * (dy_u[:-1] + dy_u[1:])
    hn[:-1] = 0.5 * (dy_u[:-1] + dy_u[1:])
    return hw, he, hs, hn


def _spacings_v(h_grid: xr.Dataset) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Distances from a v-point to its W/E and S/N v neighbours."""
    dx_v, dy_t = h_grid["dx_v"].values, h_grid["dy_t"].values
    nyv, nx = dx_v.shape
    hw, he = np.full((nyv, nx), np.nan), np.full((nyv, nx), np.nan)
    hw[:, 1:] = 0.5 * (dx_v[:, :-1] + dx_v[:, 1:])
    he[:, :-1] = 0.5 * (dx_v[:, :-1] + dx_v[:, 1:])
    hs, hn = np.full((nyv, nx), np.nan), np.full((nyv, nx), np.nan)
    hs[1:] = dy_t                                      # v[j-1] → v[j] spans cell j-1
    hn[:-1] = dy_t
    return hw, he, hs, hn


def _centred(f, wet, axis, h_lo, h_hi):
    lo, hi, wlo, whi = _pair(f, wet, axis)
    ok = wet & wlo & whi
    with np.errstate(invalid="ignore"):
        return np.where(ok, (hi - lo) / (h_lo + h_hi), 0.0)


def _upwind3(f, wet, axis, c, h_lo, h_hi):
    """
    Third-order upwind-biased derivative of f along ``axis`` for advecting
    velocity c (Leonard 1979):
        c > 0:  (2 f₊₁ + 3 f₀ − 6 f₋₁ + f₋₂) / (6 h)
        c < 0:  (−f₊₂ + 6 f₊₁ − 3 f₀ − 2 f₋₁) / (6 h)
    = centred derivative + a hyperdiffusion |c| h³/12 ∂⁴f that damps the 2Δx
    mode.  Where the 4-point stencil reaches land or the edge (one-cell
    channels, coasts) it falls back to first-order upwind, which stays
    dissipative — a centred fallback would remove the damping exactly where
    flows are strongest.
    """
    m1, p1, wm1, wp1 = _pair(f, wet, axis, 1)
    m2, p2, wm2, wp2 = _pair(f, wet, axis, 2)
    with np.errstate(invalid="ignore"):
        h = 0.5 * (h_lo + h_hi)
        pos = (2 * p1 + 3 * f - 6 * m1 + m2) / (6 * h)
        neg = (-p2 + 6 * p1 - 3 * f - 2 * m1) / (6 * h)
        first_pos = (f - m1) / h_lo                    # first-order upwind, still dissipative
        first_neg = (p1 - f) / h_hi
    ok = wet & wm1 & wp1
    up = np.where(c >= 0, np.where(ok & wm2, pos, first_pos), np.where(ok & wp2, neg, first_neg))
    return np.where(ok, up, 0.0)


def _derivative(f, wet, axis, c, h_lo, h_hi, scheme):
    if scheme == "centred":
        return _centred(f, wet, axis, h_lo, h_hi)
    if scheme == "upwind3":
        return _upwind3(f, wet, axis, c, h_lo, h_hi)
    raise ValueError(f"unknown advection scheme {scheme!r} (use 'centred' or 'upwind3')")


def advection_u(u: np.ndarray, v: np.ndarray, h_grid: xr.Dataset, wet_u: np.ndarray,
                scheme: str = "centred") -> np.ndarray:
    """
    (u·∇)u at u-faces [m s⁻²]: u ∂u/∂x + v̄ ∂u/∂y, ``scheme`` 'centred'
    (second order, no damping of the 2Δx mode) or 'upwind3' (third-order
    upwind-biased, for strong flows through narrow passages).
    """
    hw, he, hs, hn = _spacings_u(h_grid)
    vu = _v_at_u(v)
    adv = u * _derivative(u, wet_u, 1, u, hw, he, scheme) + vu * _derivative(u, wet_u, 0, vu, hs, hn, scheme)
    return np.where(wet_u, adv, 0.0)


def advection_v(u: np.ndarray, v: np.ndarray, h_grid: xr.Dataset, wet_v: np.ndarray,
                scheme: str = "centred") -> np.ndarray:
    """(u·∇)v at v-faces [m s⁻²]: ū ∂v/∂x + v ∂v/∂y; ``scheme`` as in ``advection_u``."""
    hw, he, hs, hn = _spacings_v(h_grid)
    uv = _u_at_v(u)
    adv = uv * _derivative(v, wet_v, 1, uv, hw, he, scheme) + v * _derivative(v, wet_v, 0, v, hs, hn, scheme)
    return np.where(wet_v, adv, 0.0)


def _laplacian(f, wet, hw, he, hs, hn):
    out = np.zeros_like(f)
    for axis, h_lo, h_hi in ((1, hw, he), (0, hs, hn)):
        lo, hi, wlo, whi = _pair(f, wet, axis)
        lo = np.where(wlo, lo, f)                         # free slip: no gradient towards land
        hi = np.where(whi, hi, f)
        h_lo = np.where(np.isfinite(h_lo), h_lo, 1.0)
        h_hi = np.where(np.isfinite(h_hi), h_hi, 1.0)
        out += 2.0 * ((hi - f) / h_hi - (f - lo) / h_lo) / (h_lo + h_hi)
    return np.where(wet, out, 0.0)


def laplacian_u(u: np.ndarray, h_grid: xr.Dataset, wet_u: np.ndarray) -> np.ndarray:
    """∇²u at u-faces [m⁻¹ s⁻¹], free-slip at land/edges (non-uniform 3-point stencil)."""
    return _laplacian(u, wet_u, *_spacings_u(h_grid))


def laplacian_v(v: np.ndarray, h_grid: xr.Dataset, wet_v: np.ndarray) -> np.ndarray:
    """∇²v at v-faces [m⁻¹ s⁻¹], free-slip at land/edges."""
    return _laplacian(v, wet_v, *_spacings_v(h_grid))
