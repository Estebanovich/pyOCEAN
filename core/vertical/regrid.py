"""
Grid-generation (regridding) rules of the ALE vertical coordinate, per column
(spec docs/superpowers/specs/2026-10-11-ale-vertical-coordinate-design.md §3).

Convention: interface heights zw [m], bed first: zw[0] = −H, zw[N] = η (z = 0 at
rest).  No rule changes the column thickness D = H + η.  Minimum thickness
h_min,eff = min(h_min, D/N), inflated upward from the bed (MOM6 coord_zlike.F90:137-143,
coord_sigma.F90:76-81).

z*   Adcroft & Campin (2004): z = η + s z*, s = D/H, nominal levels Δz_k (surface first,
     e.g. MITgcm delR); levels below the bed collapse to h_min at the bed, the deepest
     active level is truncated — the partial cell of MITgcm.  MOM6 build_zstar_column.
σ    Fixed fractions Δσ_k (bed first): h_k = D Δσ_k.  MOM6 build_sigma_column.

Hybrid (HYCOM1 of MOM6, OM4; Adcroft et al. 2019): interface target densities
(surface first, increasing downward).  The layer densities are monotonized
(ρ_k = min(ρ_k, ρ_k+1) upward from the bed, coord_hycom.F90:218), reconstructed
P1M with H2 edges (edges bounded and averaged, P1M_functions.F90:29-61), and
each target is located by inverting the linear cell polynomial
(regrid_interp.F90:378-511).  Then: z* floor (an interface is never above its
z* position, coord_hycom.F90:255-267), maximum layer thickness (:270-279),
optional shallow-water σ transition by a window (window_shallow_sigma; spec §3)
and the minimum thickness.  With keep_isopycnal (default) an interface between two
layers whose densities lie in their target brackets (or that have no mass) is not
moved (Plan B2c; White et al. 2009 — pure isopycnal layers have no spurious diffusion).

References: Adcroft, A. & Campin, J.-M. (2004). Ocean Modell. 7, 269–284.
Adcroft, A. et al. (2019). JAMES 11, 3167–3211. doi:10.1029/2019MS001726
Bleck, R. (2002). Ocean Modell. 4, 55–88.
White, L., Adcroft, A. & Hallberg, R. (2009). J. Comput. Phys. 228, 8665–8692.
MOM6 source ~/pyOCEAN_refs/MOM6/src/ALE (commit 0e9a561).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.vertical.remap import _bound_edges

H_MIN = 1e-3     # [m] minimum layer thickness after regridding (MOM6 MIN_THICKNESS default)


@njit(cache=True)
def zstar_column(H, eta, dz_top, h_min, zw):  # pragma: no cover
    """z* interfaces (bed first) for depth H [m], surface η [m], nominal Δz surface first [m]."""
    n = dz_top.size
    D = H + eta
    hm = min(h_min, D / n)
    s = D / H
    e = np.empty(n + 1)                     # surface first, heights
    e[0] = eta
    for k in range(n):
        e[k + 1] = e[k] - s * dz_top[k]
    e[n] = -H
    for k in range(n - 1, -1, -1):
        if e[k] < e[k + 1] + hm:
            e[k] = e[k + 1] + hm
    for j in range(n + 1):
        zw[j] = e[n - j]


@njit(cache=True)
def sigma_column(H, eta, dsigma, h_min, zw):  # pragma: no cover
    """
    σ interfaces (bed first): fractions dsigma bed first, Σ = 1.  The interfaces are pinned
    at −H and η and then kept min(h_min, D/N) apart from the top and from the bed
    (enforce_min_thickness).  MOM6 build_sigma_column (coord_sigma.F90:76-81) only inflates
    from the bed and does not pin η; with stretched σ in very shallow water a bed-only sweep
    followed by the pin left a negative surface layer.
    """
    n = dsigma.size
    D = H + eta
    zw[0] = -H
    for j in range(n):
        zw[j + 1] = zw[j] + D * dsigma[j]
    zw[n] = eta
    enforce_min_thickness(zw, h_min)


def thickness(zw: np.ndarray) -> np.ndarray:
    """Layer thicknesses [m] from interface heights (first axis, bed first)."""
    return np.diff(zw, axis=0)


@njit(cache=True)
def enforce_min_thickness(zw, h_min):  # pragma: no cover
    """Keep layers ≥ min(h_min, D/N): clamp from the top, then inflate from the bed."""
    n = zw.size - 1
    hm = min(h_min, (zw[n] - zw[0]) / n)
    for j in range(n - 1, 0, -1):
        if zw[j] > zw[j + 1] - hm:
            zw[j] = zw[j + 1] - hm
    for j in range(1, n):
        if zw[j] < zw[j - 1] + hm:
            zw[j] = zw[j - 1] + hm


# Not disk-cached: calls remap._bound_edges, and Numba's cache only checks this file
# (a cached copy would go stale when remap.py changes).
@njit
def _p1m_h2_positions(h_top, rho_top, targets, x_new):  # pragma: no cover
    """
    Depths (positive down from the surface, 0 … D) of target interface densities, MOM6
    build_and_interpolate_grid with INTERPOLATION_SCHEME = P1M_H2 and no boundary
    extrapolation.  h_top, rho_top surface first; targets surface first (n+1).
    """
    n = h_top.size
    e = np.empty((n, 2))
    e[0, 0] = rho_top[0]
    e[n - 1, 1] = rho_top[n - 1]
    for k in range(1, n):                                    # H2 edges (regrid_edge_values.F90:168-194)
        if h_top[k - 1] + h_top[k] == 0.0:
            e[k, 0] = 0.5 * (rho_top[k - 1] + rho_top[k])
        else:
            e[k, 0] = (rho_top[k - 1] * h_top[k] + rho_top[k] * h_top[k - 1]) / (h_top[k - 1] + h_top[k])
        e[k - 1, 1] = e[k, 0]
    _bound_edges(h_top, rho_top, e)
    for k in range(n - 1):                                   # average_discontinuous_edge_values
        if e[k, 1] != e[k + 1, 0]:
            a = 0.5 * (e[k, 1] + e[k + 1, 0])
            e[k, 1] = a
            e[k + 1, 0] = a
    xg = np.empty(n + 1)
    xg[0] = 0.0
    for k in range(n):
        xg[k + 1] = xg[k] + h_top[k]
    for k in range(n + 1):                                   # fallback when no cell matches (NaN ρ):
        x_new[k] = xg[k]                                     # keep the input grid (MOM6 stops FATAL)
    for kk in range(1, n):
        t = targets[kk]
        if t <= e[0, 0]:
            x_new[kk] = xg[0]
            continue
        found = False
        for k in range(1, n):
            if t >= e[k - 1, 1] and t <= e[k, 0]:
                x_new[kk] = xg[k]
                found = True
                break
        if found:
            continue
        if t >= e[n - 1, 1]:
            x_new[kk] = xg[n]
            continue
        for k in range(n):
            if e[k, 0] < t < e[k, 1]:
                xi = (t - e[k, 0]) / (e[k, 1] - e[k, 0])
                x_new[kk] = xg[k] + min(max(xi, 0.0), 1.0) * h_top[k]
                break


# Not disk-cached: calls remap._bound_edges, and Numba's cache only checks this file
# (a cached copy would go stale when remap.py changes).
@njit
def _keep_consistent(h_top, rho_top, targets, h_mass, x):  # pragma: no cover
    """
    Isopycnal layer preservation (spec §3, Plan B2c).  Layer j (surface first, between the
    targets t_j and t_j+1) is consistent if it has no mass (h < h_mass) or t_j ≤ ρ_j ≤ t_j+1
    with its own density.  An interface between two consistent layers stays where the
    Lagrangian step left it; the others keep their P1M position, clipped between the nearest
    kept interfaces above and below so they never push a kept one.  x: depths from the surface
    (0 … D, surface first), overwritten in place.  Pure isopycnal layers have no spurious
    diffusion (White et al. 2009, figs. 11-12); hybgen moves only layers that left their target
    (Bleck 2002) — the principle, without porting hybgen.
    """
    n = h_top.size
    ok = np.empty(n, np.bool_)
    for j in range(n):
        ok[j] = h_top[j] < h_mass or (targets[j] <= rho_top[j] <= targets[j + 1])
    xg = np.empty(n + 1)
    xg[0] = 0.0
    for j in range(n):
        xg[j + 1] = xg[j] + h_top[j]
    kept = np.zeros(n + 1, np.bool_)
    kept[0] = True
    kept[n] = True
    for j in range(1, n):
        if ok[j - 1] and ok[j]:
            x[j] = xg[j]
            kept[j] = True
    lo = xg[0]
    for j in range(1, n):
        if kept[j]:
            lo = x[j]
        elif x[j] < lo:
            x[j] = lo
    hi = xg[n]
    for j in range(n - 1, 0, -1):
        if kept[j]:
            hi = x[j]
        elif x[j] > hi:
            x[j] = hi


@njit
def hycom1_column(H, eta, h_bed, rho_bed, rho_target, dz_top, max_h_top, h_min, keep, zw):  # pragma: no cover
    """Hybrid z*–isopycnal interfaces (bed first) — see the module docstring."""
    n = h_bed.size
    D = H + eta
    h_top = h_bed[::-1].copy()
    rho_own = rho_bed[::-1].copy()                           # each layer's own density (B2c test)
    rho = rho_own.copy()
    # massless layers (h < 2 h_min) keep a stale density, e.g. the light water of a bed layer that
    # a gravity current has covered; they take the density of the nearest layer with mass above
    # (below if there is none), as HYCOM fills them (hybgen.F90:904-925), so the monotonising
    # below does not lighten the water over them.  Only the density seen by the rule changes.
    h_mass = 2.0 * h_min
    last = -1.0
    for k in range(n):
        if h_top[k] >= h_mass:
            last = rho[k]
        elif last >= 0.0:
            rho[k] = last
    for k in range(n):                                       # top run without mass: from below
        if h_top[k] >= h_mass:
            break
    if k > 0 and h_top[k] >= h_mass:
        for kk in range(k):
            rho[kk] = rho[k]
    for k in range(n - 2, -1, -1):                           # monotonic (coord_hycom.F90:218)
        rho[k] = min(rho[k], rho[k + 1])
    x = np.empty(n + 1)
    _p1m_h2_positions(h_top, rho, rho_target, x)
    if keep:
        _keep_consistent(h_top, rho_own, rho_target, 2.0 * h_min, x)
    s = D / H                                                # z* floor (coord_hycom.F90:255-267)
    nominal = 0.0
    for k in range(1, n + 1):
        nominal += dz_top[k - 1] * s
        x[k] = min(max(x[k], nominal), D)
    for k in range(1, n):                                    # maximum layer thickness (:270-279)
        x[k] = min(x[k], x[k - 1] + max_h_top[k - 1])
    for k in range(1, n + 1):                                # the floor may push a kept interface past
        x[k] = max(x[k], x[k - 1])                           # a regridded one: restore the order
    x[n] = D
    for j in range(n + 1):
        zw[j] = eta - x[n - j]
    zw[0] = -H
    zw[n] = eta
    enforce_min_thickness(zw, h_min)


@njit(cache=True)
def transition_weight(D, d_sigma, d_hybrid):  # pragma: no cover
    """q = clip((D − d_σ)/(d_hyb − d_σ), 0, 1): 0 = pure σ (shallow), 1 = pure hybrid (deep)."""
    if d_hybrid <= d_sigma:
        return 1.0 if D > d_sigma else 0.0
    return min(max((D - d_sigma) / (d_hybrid - d_sigma), 0.0), 1.0)


@njit(cache=True)
def window_shallow_sigma(zw_hyb, zw_sig, zw_zs, H, D, d_sigma, d_hybrid, h_min, zw):  # pragma: no cover
    """
    Shallow-water σ transition by a window (spec §3, decision of 2026-10-11).  With
    q = transition_weight(D, d_σ, d_hyb), interface j keeps its hybrid position inside

        L_j = q (−H) + (1 − q) σ_j  ≤  z_j  ≤  U_j = q z*_j + (1 − q) σ_j

    and is clipped to the nearer bound otherwise; then the minimum thickness.  q = 0: pure σ
    (wetting–drying); q = 1: the hybrid grid unchanged; in between, the isopycnals that fit in
    the window are kept.  Continuous in D; interfaces never cross (monotone bounds and input);
    Σh = D.  HYCOM (MOM_hybgen_regrid.F90:590-620) instead blends the minimum thicknesses
    dp0k/ds0k with a weight that makes them add up to D, so between dsns and dpns its column
    is pure σ and follows no isopycnal.
    """
    n = zw.size - 1
    q = transition_weight(D, d_sigma, d_hybrid)
    if q >= 1.0:
        for j in range(n + 1):
            zw[j] = zw_hyb[j]
        return
    for j in range(n + 1):
        lo = q * (-H) + (1.0 - q) * zw_sig[j]
        hi = q * zw_zs[j] + (1.0 - q) * zw_sig[j]
        zw[j] = min(max(zw_hyb[j], lo), hi)
    zw[0] = -H
    zw[n] = zw_sig[n]
    enforce_min_thickness(zw, h_min)


def target_densities(rho_top: np.ndarray, n_layers: int, n_light: int = 0, d_rho_light: float = 0.01) -> np.ndarray:
    """
    Interface target densities (surface first) [kg m⁻³]: the first ``n_light`` lighter than any
    water, ρ_min − d_rho_light·(n_light … 1), so that the hybrid rule keeps those interfaces on
    their z* floor and the mixed layer stays resolved in z (OM4 style, Adcroft et al. 2019; see
    validation/mom6_column/README.md); the rest evenly spaced over the profile range.
    """
    if not 0 <= n_light < n_layers:
        raise ValueError(f"n_light must be in [0, {n_layers - 1}], got {n_light}")
    rho_top = np.asarray(rho_top, float)
    lo, hi = np.nanmin(rho_top), np.nanmax(rho_top)
    light = lo - d_rho_light * np.arange(n_light, 0, -1)
    return np.r_[light, np.linspace(lo, hi, n_layers + 1 - n_light)]


# Not disk-cached: calls remap._bound_edges, and Numba's cache only checks this file
# (a cached copy would go stale when remap.py changes).
@njit(parallel=True)
def _regrid_columns(code, H, eta, h, rho, dz_top, dsigma, rho_target, max_h_top,
                    d_sigma, d_hybrid, h_min, keep, wet, zw):  # pragma: no cover
    n, ny, nx = h.shape
    for c in prange(ny * nx):
        j = c // nx
        i = c % nx
        if not wet[j, i]:
            continue
        col = np.empty(n + 1)
        if code == 0:
            zstar_column(H[j, i], eta[j, i], dz_top, h_min, col)
        elif code == 1:
            sigma_column(H[j, i], eta[j, i], dsigma, h_min, col)
        else:
            q = transition_weight(H[j, i] + eta[j, i], d_sigma, d_hybrid) if d_hybrid > 0.0 else 1.0
            if d_hybrid > 0.0 and H[j, i] <= 0.0:            # flooded land: σ at any depth (no z* floor there)
                q = 0.0
            if q <= 0.0:                                     # σ zone: no hybrid grid needed (H may be ≤ 0)
                sigma_column(H[j, i], eta[j, i], dsigma, h_min, col)
            else:
                hycom1_column(H[j, i], eta[j, i], np.ascontiguousarray(h[:, j, i]),
                              np.ascontiguousarray(rho[:, j, i]), rho_target, dz_top, max_h_top, h_min, keep, col)
                if q < 1.0:
                    sig = np.empty(n + 1)
                    zs = np.empty(n + 1)
                    sigma_column(H[j, i], eta[j, i], dsigma, h_min, sig)
                    zstar_column(H[j, i], eta[j, i], dz_top, h_min, zs)
                    hyb = col.copy()
                    window_shallow_sigma(hyb, sig, zs, H[j, i], H[j, i] + eta[j, i], d_sigma, d_hybrid,
                                         h_min, col)
        for k in range(n + 1):
            zw[k, j, i] = col[k]


def regrid_columns(rule: str, H: np.ndarray, eta: np.ndarray, h: np.ndarray, rho: np.ndarray | None,
                   params: dict, wet: np.ndarray) -> np.ndarray:
    """
    Target interface heights zw (N+1, ny, nx) [m], bed first, for rule "zstar", "sigma" or "hybrid".
    params: dz_top (z* and hybrid floor), dsigma (σ and the shallow transition), rho_target,
    max_h_top, d_sigma, d_hybrid [m] (0 = no transition), h_min [m], keep_isopycnal (hybrid,
    default True): interfaces between isopycnally consistent layers stay put (spec §3, B2c);
    False = HYCOM1 of MOM6 (P1M everywhere).  Dry columns: NaN.
    Raises ValueError for a missing required input, a wet column with non-finite H or η, D ≤ 0,
    H ≤ 0 with z* (or with the hybrid rule without a σ zone, d_hybrid ≤ 0), or non-finite ρ (hybrid).
    """
    if rule not in ("zstar", "sigma", "hybrid"):
        raise ValueError(f"unknown rule {rule!r}; use 'zstar', 'sigma' or 'hybrid'")
    required = {"zstar": ("dz_top",), "sigma": ("dsigma",), "hybrid": ("dz_top", "rho_target")}[rule]
    if rule == "hybrid" and float(params.get("d_hybrid", 0.0)) > 0.0:
        required += ("dsigma",)
    for key in required:
        if key not in params:
            raise ValueError(f"regrid_columns({rule!r}) needs params[{key!r}]")
    if rule == "hybrid" and rho is None:
        raise ValueError("regrid_columns('hybrid') needs the layer densities rho")
    wet = np.asarray(wet, np.bool_)
    H = np.asarray(H, float)
    eta = np.asarray(eta, float)
    Hw, Dw = H[wet], (H + eta)[wet]
    if not (np.isfinite(Hw).all() and np.isfinite(Dw).all() and (Dw > 0).all()):
        raise ValueError("wet columns need finite H and D = H + eta > 0")
    if rule == "zstar" and not (Hw > 0).all():
        raise ValueError("z* needs H > 0 in every wet column (cells that can dry: sigma, or hybrid with d_hybrid > 0)")
    if rule == "hybrid":
        d_h = float(params.get("d_hybrid", 0.0))
        if d_h <= 0 and (Hw <= 0).any():
            raise ValueError("hybrid needs H > 0 unless it has a shallow σ zone (d_hybrid > 0); "
                             "columns with H ≤ 0 then stay σ at any depth")
    if rule == "hybrid" and not np.isfinite(np.asarray(rho, float)[:, wet]).all():
        raise ValueError("non-finite density in a wet column")
    code = {"zstar": 0, "sigma": 1, "hybrid": 2}[rule]
    n = h.shape[0]
    zw = np.full((n + 1,) + h.shape[1:], np.nan)
    f = lambda key, default: np.ascontiguousarray(params.get(key, default), float)  # noqa: E731
    _regrid_columns(code, np.nan_to_num(np.asarray(H, float)), np.asarray(eta, float), np.ascontiguousarray(h, float),
                    np.ascontiguousarray(h if rho is None else rho, float), f("dz_top", np.ones(n)),
                    f("dsigma", np.full(n, 1.0 / n)), f("rho_target", np.zeros(n + 1)),
                    f("max_h_top", np.full(n, np.inf)), float(params.get("d_sigma", 0.0)),
                    float(params.get("d_hybrid", 0.0)), float(params.get("h_min", H_MIN)),
                    bool(params.get("keep_isopycnal", True)), np.asarray(wet, np.bool_), zw)
    return zw
