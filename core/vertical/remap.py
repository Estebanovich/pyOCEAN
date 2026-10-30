"""
Conservative vertical remapping of one column between two grids with the same
total thickness (vertical Lagrangian-remap method; Griffies et al. 2020).

Each source cell gets a polynomial reconstruction p(ξ), ξ ∈ [0, 1] across the
cell, with the cell mean as its average:

    PCM     p = ū                                              (piecewise constant)
    PLM     p = u_L + (u_R − u_L) ξ        White & Adcroft (2008); MOM6 PLM_functions.F90
    PPM_H4  p = u_L + [4(ū−u_L) + 2(ū−u_R)] ξ + 3[(u_R−ū) + (u_L−ū)] ξ²
            with explicit 4th-order edge values; Colella & Woodward (1984) limiter;
            MOM6 PPM_functions.F90:23-120, regrid_edge_values.F90:215-350

The target means come from integrating p over the union of both grids.  In
every source cell, the thickest sub-cell takes the residual u0·h0 − Σ(other
sub-cells) (the "Hallberg trick", MOM6 MOM_remapping.F90:942-961), so the
column content Σ h u is conserved to round-off.  Top and bottom cells are PCM
(no extrapolation; MOM6 default REMAP_BOUNDARY_EXTRAP = False).  Short columns
reduce the order: 1 active cell → PCM, ≤ 3 → PLM.  Unlike MOM6 (MOM_remapping.F90:453-460,
which counts all n0 cells), the count is of active (h > 0) cells, so a z* column with ≤ 3
active cells among vanished ones uses PLM where MOM6 would use PPM_H4.  A column with no
active cell (dry, D = 0) keeps its values.

Columns can be in any consistent order (the model uses k = 0 at the bed).

References
----------
White, L. & Adcroft, A. (2008). J. Comput. Phys. 227, 7394–7422. doi:10.1016/j.jcp.2008.04.026
White, L., Adcroft, A. & Hallberg, R. (2009). J. Comput. Phys. 228, 8665–8692.
Colella, P. & Woodward, P. (1984). J. Comput. Phys. 54, 174–201.
Griffies, S.M. et al. (2020). JAMES 12, e2019MS001954.
MOM6 source: ~/pyOCEAN_refs/MOM6/src/ALE (commit 0e9a561).
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

PCM, PLM, PPM_H4 = 0, 1, 2
SCHEMES = {"pcm": PCM, "plm": PLM, "ppm_h4": PPM_H4}
H_NEGLECT = 1e-30          # [m] negligible thickness for cell reconstructions
H_NEGLECT_EDGE = 1e-10     # [m] negligible thickness for edge values


@njit(cache=True)
def _effective_scheme(n_active: int, scheme: int) -> int:
    """Lower the order on short columns, counting active cells (MOM6 counts n0, MOM_remapping.F90:453-460)."""
    if n_active <= 1:
        return PCM
    if n_active <= 3 and scheme > PLM:
        return PLM
    return scheme


_EPS = np.finfo(np.float64).eps


@njit(cache=True)
def _plm_slope_wa(h_l, h_c, h_r, h_neglect, u_l, u_c, u_r):
    """Limited PLM slope (difference across the cell), MOM6 PLM_functions.F90:22-68."""
    sigma_r = u_r - u_c
    sigma_l = u_c - u_l
    sigma_c = 2.0 * (u_r - u_l) * (h_c / (h_l + 2.0 * h_c + h_r + h_neglect))
    u_min = min(u_l, u_c, u_r)
    u_max = max(u_l, u_c, u_r)
    if sigma_l * sigma_r > 0.0:
        s = min(abs(sigma_c), 2.0 * min(u_c - u_min, u_max - u_c))
        slope = s if sigma_c >= 0.0 else -s
    else:
        slope = 0.0
    if u_c - 0.5 * abs(slope) < u_min or u_c + 0.5 * abs(slope) > u_max:
        slope = slope * (1.0 - _EPS)
    if abs(slope) < 1e-140:
        slope = 0.0
    return slope


@njit(cache=True)
def _plm_monotonized_slope(u_l, u_c, u_r, s_l, s_c, s_r):
    """Edge-monotonic PLM slope, MOM6 PLM_functions.F90:122-158."""
    almost_two = 2.0 * (1.0 - _EPS)
    e_r = u_l + 0.5 * s_l
    e_l = u_r - 0.5 * s_r
    slp = abs(s_c)
    edge = u_c - 0.5 * s_c
    if (edge - e_r) * (u_c - edge) < 0.0:
        edge = 0.5 * (edge + e_r)
        slp = min(slp, abs(edge - u_c) * almost_two)
    edge = u_c + 0.5 * s_c
    if (edge - u_c) * (e_l - edge) < 0.0:
        edge = 0.5 * (edge + e_l)
        slp = min(slp, abs(edge - u_c) * almost_two)
    return slp if s_c >= 0.0 else -slp


H_MIN_FRAC = 1e-5          # MOM6 regrid_edge_values.F90:27


@njit(cache=True)
def _end_values_h4(dz, u):
    """
    Edge values (C(0), C(dz[0])) of the cubic C(x) = C0 + C1 x + C2 x² + C3 x³, x from the
    column end, whose means over the first 4 cells equal u (White & Adcroft 2008, appendix).
    Closed form of MOM6 end_value_h4 (regrid_edge_values.F90:636-749): the 4×4 system is
    reduced analytically to sums of weighted differences u(k+1) − u(k) with no subtraction of
    thicknesses, and h3, h4 are raised to min_frac·h1 when h2+h3 or h3+h4 vanish (:680-681).
    A direct elimination is singular when vanished layers sit in the end stencil.
    dz : the 4 thicknesses from the end, > 0 [m];  u : the 4 cell means.
    """
    min_frac = 1.0e-6
    h1 = dz[0]
    h2 = dz[1]
    h3 = dz[2]
    h4 = dz[3]
    if (h2 + h3) < min_frac * h1:
        h3 = min_frac * h1 - h2
    if (h3 + h4) < min_frac * h1:
        h4 = min_frac * h1 - h3
    h12 = h1 + h2
    h23 = h2 + h3
    h34 = h3 + h4
    h123 = h12 + h3
    h234 = h2 + h34
    h1234 = h12 + h34
    I_denB3 = 1.0 / (h123 * h12 * h23)
    I_h12 = (h123 * h23) * I_denB3
    I_h23 = (h12 * h123) * I_denB3
    I_h123 = (h12 * h23) * I_denB3
    I_denom = 1.0 / (h1234 * (h234 * h34))
    I_h234 = (h1234 * h34) * I_denom
    I_h1234 = (h234 * h34) * I_denom

    w11 = -h1 * (I_h1234 + I_h123 + I_h12)
    w21 = h1 * h12 * (I_h234 * I_h1234 + I_h23 * (I_h234 + I_h123))
    w31 = -h1 * h12 * h123 * I_denom
    w12 = 2.0 * (I_h12 * (1.0 + (h1 + h12) * (I_h1234 + I_h123)) + h1 * I_h1234 * I_h123)
    w22 = -2.0 * ((h1 * h12 * I_h1234) * (I_h23 * (I_h234 + I_h123))
                  + (h1 + h12) * (I_h1234 * I_h234 + I_h23 * (I_h234 + I_h123)))
    w32 = 2.0 * ((h1 + h12) * h123 + h1 * h12) * I_denom
    w13 = -3.0 * I_h12 * I_h123 * (1.0 + I_h1234 * ((h1 + h12) + h123))
    w23 = 3.0 * I_h23 * (I_h123 + I_h1234 * ((h1 + h12) + h123) * (I_h123 + I_h234))
    w33 = -3.0 * ((h1 + h12) + h123) * I_denom
    w14 = 4.0 * I_h1234 * I_h123 * I_h12
    w24 = -4.0 * I_h1234 * (I_h23 * (I_h123 + I_h234))
    w34 = 4.0 * I_denom

    d1 = u[1] - u[0]
    d2 = u[2] - u[1]
    d3 = u[3] - u[2]
    c0 = ((u[0] + (w11 * d1)) + (w21 * d2)) + (w31 * d3)
    c1 = ((w12 * d1) + (w22 * d2)) + (w32 * d3)
    c2 = ((w13 * d1) + (w23 * d2)) + (w33 * d3)
    c3 = ((w14 * d1) + (w24 * d2)) + (w34 * d3)
    d = dz[0]                                    # MOM6 evaluates at the unmodified dz(1)
    return c0, c0 + d * (c1 + d * (c2 + d * c3))


@njit(cache=True)
def _edges_h4(h, u, h_neglect, e):
    """Explicit 4th-order edge values e[k] = (left, right), MOM6 regrid_edge_values.F90:215-350."""
    n = h.size
    for i in range(2, n - 1):
        h0 = h[i - 2]
        h1 = h[i - 1]
        h2 = h[i]
        h3 = h[i + 1]
        if h0 + h1 == 0.0 or h1 + h2 == 0.0 or h2 + h3 == 0.0:
            h_min = H_MIN_FRAC * max(h_neglect, (h0 + h1) + (h2 + h3))
            h0 = max(h_min, h0)
            h1 = max(h_min, h1)
            h2 = max(h_min, h2)
            h3 = max(h_min, h3)
        I_h12 = 1.0 / (h1 + h2)
        I_den_et2 = 1.0 / (((h0 + h1) + h2) * (h0 + h1))
        I_h012 = (h0 + h1) * I_den_et2
        I_den_et3 = 1.0 / ((h1 + (h2 + h3)) * (h2 + h3))
        I_h123 = (h2 + h3) * I_den_et3
        et1 = ((1.0 + (h1 * I_h012 + (h0 + h1) * I_h123)) * I_h12 * (h2 * (h2 + h3)) * u[i - 1]
               + (1.0 + (h2 * I_h123 + (h2 + h3) * I_h012)) * I_h12 * (h1 * (h0 + h1)) * u[i])
        et2 = (h1 * (h2 * (h2 + h3)) * I_den_et2) * (u[i - 1] - u[i - 2])
        et3 = (h2 * (h1 * (h0 + h1)) * I_den_et3) * (u[i] - u[i + 1])
        e[i, 0] = (et1 + (et2 + et3)) / ((h0 + h1) + (h2 + h3))
        e[i - 1, 1] = e[i, 0]
    dz = np.empty(4)
    ut = np.empty(4)
    for i in range(4):
        dz[i] = max(h_neglect, h[i])
        ut[i] = u[i]
    e[0, 0], e[0, 1] = _end_values_h4(dz, ut)
    e[1, 0] = e[0, 1]
    for i in range(4):
        dz[i] = max(h_neglect, h[n - 1 - i])
        ut[i] = u[n - 1 - i]
    e[n - 1, 1], e[n - 1, 0] = _end_values_h4(dz, ut)
    e[n - 2, 1] = e[n - 1, 0]


@njit(cache=True)
def _bound_edges(h, u, e):
    """MOM6 bound_edge_values, answer_date ≥ 2019 (regrid_edge_values.F90:40-103)."""
    n = h.size
    for k in range(n):
        km1 = max(0, k - 1)
        kp1 = min(k + 1, n - 1)
        slope_x_h = 0.0
        if (h[km1] + h[kp1]) + 2.0 * h[k] > 0.0:
            sl = u[k] - u[km1]
            sc = (u[kp1] - u[km1]) * (h[k] / ((h[km1] + h[kp1]) + 2.0 * h[k]))
            sr = u[kp1] - u[k]
            if sl * sr > 0.0:
                m = min(abs(sl), abs(sc), abs(sr))
                slope_x_h = m if sc >= 0.0 else -m
        sgn = 1.0 if slope_x_h >= 0.0 else -1.0
        if (u[km1] - e[k, 0]) * (e[k, 0] - u[k]) < 0.0:
            e[k, 0] = u[k] - sgn * min(abs(slope_x_h), abs(e[k, 0] - u[k]))
        if (u[kp1] - e[k, 1]) * (e[k, 1] - u[k]) < 0.0:
            e[k, 1] = u[k] + sgn * min(abs(slope_x_h), abs(e[k, 1] - u[k]))
        e[k, 0] = max(min(e[k, 0], max(u[km1], u[k])), min(u[km1], u[k]))
        e[k, 1] = max(min(e[k, 1], max(u[kp1], u[k])), min(u[kp1], u[k]))


@njit(cache=True)
def _check_discontinuous(u, e):
    """MOM6 check_discontinuous_edge_values (regrid_edge_values.F90:134-152)."""
    for k in range(u.size - 1):
        if (e[k + 1, 0] - e[k, 1]) * (u[k + 1] - u[k]) < 0.0:
            a = 0.5 * (e[k, 1] + e[k + 1, 0])
            a = max(min(a, max(u[k], u[k + 1])), min(u[k], u[k + 1]))
            e[k, 1] = a
            e[k + 1, 0] = a


@njit(cache=True)
def reconstruct(h, u, scheme, h_neglect, h_neglect_edge, coef):  # pragma: no cover
    """
    Polynomial coefficients coef[k] = (c0, c1, c2) of p(ξ) = c0 + c1 ξ + c2 ξ² in cell k [units of u].
    h : cell thicknesses [m];  u : cell means.  PLM: MOM6 PLM_reconstruction (PLM_functions.F90:186-262).
    """
    n = h.size
    for k in range(n):
        coef[k, 0] = u[k]
        coef[k, 1] = 0.0
        coef[k, 2] = 0.0
    if scheme == PCM or n < 3:
        return
    if scheme == PLM:
        slp = np.zeros(n)
        mslp = np.zeros(n)
        for k in range(1, n - 1):
            slp[k] = _plm_slope_wa(h[k - 1], h[k], h[k + 1], h_neglect, u[k - 1], u[k], u[k + 1])
        for k in range(1, n - 1):
            mslp[k] = _plm_monotonized_slope(u[k - 1], u[k], u[k + 1], slp[k - 1], slp[k], slp[k + 1])
        almost_one = 1.0 - _EPS
        for k in range(1, n - 1):
            u_l = u[k] - 0.5 * mslp[k]
            u_r = u[k] + 0.5 * mslp[k]
            coef[k, 0] = u_l
            coef[k, 1] = u_r - u_l
            edge = coef[k, 1] + coef[k, 0]
            # Fortran u(k+1) - 0.5*sign(mslp(k+1), slp(k+1)): |mslp| with the sign of slp
            e_r = u[k + 1] - 0.5 * (abs(mslp[k + 1]) if slp[k + 1] >= 0.0 else -abs(mslp[k + 1]))
            if (edge - u[k]) * (e_r - edge) < 0.0:
                coef[k, 1] = coef[k, 1] * almost_one
        return
    if scheme == PPM_H4 and n >= 4:
        e = np.empty((n, 2))
        _edges_h4(h, u, h_neglect_edge, e)
        _bound_edges(h, u, e)
        _check_discontinuous(u, e)
        for k in range(1, n - 1):                    # Colella & Woodward (1984) limiter
            u_l = u[k - 1]
            u_c = u[k]
            u_r = u[k + 1]
            el = e[k, 0]
            er = e[k, 1]
            if (u_r - u_c) * (u_c - u_l) <= 0.0:
                el = u_c
                er = u_c
            else:
                expr1 = 3.0 * (er - el) * ((u_c - el) + (u_c - er))
                expr2 = (er - el) * (er - el)
                if expr1 > expr2:
                    el = u_c + 2.0 * (u_c - er)
                    el = max(min(el, max(u_l, u_c)), min(u_l, u_c))
                elif expr1 < -expr2:
                    er = u_c + 2.0 * (u_c - el)
                    er = max(min(er, max(u_r, u_c)), min(u_r, u_c))
            if abs(er - el) < max(1e-60, _EPS * abs(u_c)):
                el = u_c
                er = u_c
            coef[k, 0] = el
            coef[k, 1] = 4.0 * (u_c - el) + 2.0 * (u_c - er)
            coef[k, 2] = 3.0 * ((er - u_c) + (el - u_c))


@njit(cache=True)
def _average(c0, c1, c2, xa, xb):
    """Mean of p(ξ) over [xa, xb] ⊂ [0, 1]; the point value when xb ≈ xa."""
    if xb - xa <= 1e-14:
        x = 0.5 * (xa + xb)
        return c0 + x * (c1 + x * c2)
    return c0 + 0.5 * c1 * (xa + xb) + c2 * (xa * xa + xa * xb + xb * xb) / 3.0


@njit(cache=True)
def remap_column(h_src, u_src, h_tgt, scheme, u_tgt, h_neglect=H_NEGLECT,
                 h_neglect_edge=H_NEGLECT_EDGE):  # pragma: no cover
    """
    Conservative remap of cell means u_src on thicknesses h_src [m] to thicknesses
    h_tgt [m] (same order; Σh_tgt = Σh_src up to round-off); result in u_tgt.
    """
    n0 = h_src.size
    n1 = h_tgt.size
    n_active = 0
    for k in range(n0):
        if h_src[k] > 0.0:
            n_active += 1
    coef = np.empty((n0, 3))
    reconstruct(h_src, u_src, _effective_scheme(n_active, scheme), h_neglect, h_neglect_edge, coef)

    # target interfaces (cumulative, starting at 0)
    t_edge = np.empty(n1 + 1)
    t_edge[0] = 0.0
    for j in range(n1):
        t_edge[j + 1] = t_edge[j] + h_tgt[j]
    content = np.zeros(n1)
    piece = np.empty(n1 + 1)
    piece_j = np.empty(n1 + 1, np.int64)

    a = 0.0                      # bottom of the current source cell
    j = 0
    last = -1                    # last source cell with h > 0
    for i in range(n0):
        hi = h_src[i]
        if hi <= 0.0:
            continue
        last = i
        b = a + hi
        npc = 0
        thick = -1.0
        ithick = 0
        while j < n1:
            lo = max(a, t_edge[j])
            hi_t = t_edge[j + 1] if j < n1 - 1 else max(t_edge[n1], b)   # last target absorbs a longer source
            up = min(b, hi_t)
            L = up - lo
            if L > 0.0:
                piece[npc] = L * _average(coef[i, 0], coef[i, 1], coef[i, 2], (lo - a) / hi, (up - a) / hi)
                piece_j[npc] = j
                if L > thick:
                    thick = L
                    ithick = npc
                npc += 1
            if hi_t <= b and j < n1 - 1:
                j += 1
            else:
                break
        if npc > 0:
            s = 0.0
            for p in range(npc):
                if p != ithick:
                    s += piece[p]
            piece[ithick] = u_src[i] * hi - s                      # Hallberg trick: exact cell content
            for p in range(npc):
                content[piece_j[p]] += piece[p]
        a = b

    # Target thicker than source (round-off, ±1–2 ε D): the extra thickness gets no content, so
    # the column content Σ h u is exact in both directions. MOM6 extends the last source cell with
    # its top edge value here (MOM_remapping.F90:939-940), which only ever adds content: a
    # one-signed +1.4 ε Σ h u per ALE step (N3 Lorenz A budget 1.5e-9; decision of 2026-10-30).

    if last < 0:                 # dry column (no source thickness): nothing to remap, keep the values
        mean = 0.0
        for i in range(n0):
            mean += u_src[i] / n0
        for jj in range(n1):
            u_tgt[jj] = u_src[jj] if n0 == n1 else mean
        return

    for jj in range(n1):
        if h_tgt[jj] > 0.0:
            u_tgt[jj] = content[jj] / h_tgt[jj]
        else:                                                     # vanished target: local value
            x = t_edge[jj]
            aa = 0.0
            val = u_src[last]
            for i in range(n0):
                if h_src[i] <= 0.0:
                    continue
                if x <= aa + h_src[i]:
                    xi = (x - aa) / h_src[i]
                    val = coef[i, 0] + xi * (coef[i, 1] + xi * coef[i, 2])
                    break
                aa += h_src[i]
            u_tgt[jj] = val


@njit(cache=True, parallel=True)
def _remap_columns(h_src, f_src, h_tgt, scheme, wet, out):  # pragma: no cover
    n0, ny, nx = h_src.shape
    n1 = h_tgt.shape[0]
    for c in prange(ny * nx):
        j = c // nx
        i = c % nx
        if not wet[j, i]:
            for k in range(n1):
                out[k, j, i] = f_src[k, j, i] if n0 == n1 else np.nan
            continue
        col = np.empty(n1)
        remap_column(np.ascontiguousarray(h_src[:, j, i]), np.ascontiguousarray(f_src[:, j, i]),
                     np.ascontiguousarray(h_tgt[:, j, i]), scheme, col)
        for k in range(n1):
            out[k, j, i] = col[k]


def remap_columns(h_src: np.ndarray, f_src: np.ndarray, h_tgt: np.ndarray, scheme: int,
                  wet: np.ndarray) -> np.ndarray:
    """
    Remap every wet column of f_src (N0, ny, nx) from thicknesses h_src to h_tgt (N1, ny, nx) [m].
    Dry columns: f_src copied when N0 == N1, else NaN.  Threads: numba.set_num_threads (≤ 5).
    """
    out = np.empty(h_tgt.shape)
    _remap_columns(np.ascontiguousarray(h_src, float), np.ascontiguousarray(f_src, float),
                   np.ascontiguousarray(h_tgt, float), int(scheme), np.asarray(wet, np.bool_), out)
    return out
