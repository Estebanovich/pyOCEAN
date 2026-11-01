"""
Finite-volume pressure-gradient force with a nonlinear equation of state (Adcroft, Hallberg &
Harrison 2008), Boussinesq, on any layer geometry: the MOM6 path ``int_density_dz_generic_plm`` +
``PressureForce_FV_Bouss`` (TEOS-10 in the 3-D core, task 3; spec goal B).

As in ``core.vertical.pgf``, the force on layer k at a u-face is the contour integral of the
pressure anomaly around the layer's section between the two cell centres,

    PFu_k = [ (pa_L h_L + ∫∫dpa_L) − (pa_R h_R + ∫∫dpa_R) + (h_R − h_L) ∫pa_top − (e_R − e_L)_bot ∫dpa ]
            · 2 / (ρ₀ Δx (h_L + h_R + h_neglect))                       (MOM_PressureForce_FV.F90:1796-1801)

but the density anomaly is evaluated pointwise from the equation of state at the quadrature points
of a 5 × 5 Boole rule in (x, z), with T and S reconstructed separately (MOM6 does not reconstruct
density; docs/design/teos10_refs/mom6_eos.md §4.0-4.c):

1. Interfaces from the bed up, e_N = −H, e_k = e_{k+1} + h_k (surface-first index, MOM6 order;
   MOM_PressureForce_FV.F90:1146-1148, 1195-1198).
2. T and S edge values by PLM, each on its own (TS_PLM_edge_values → ALE_PLM_edge_values,
   ALE/MOM_ALE.F90:1512-1596): PLM_slope_wa + PLM_monotonized_slope in the interior and
   PLM_extrapolate_slope in the top and bottom layers (BOUNDARY_EXTRAPOLATION_PRESSURE = True,
   the default; ALE/PLM_functions.F90:22-183), h_neglect = 1e-30 m (GV%H_subroundoff).
3. Per column and layer (MOM_density_integrals.F90:590-640): 5 points top to bottom with
   T_n = w_t T_t + w_b T_b, w_t = (5 − n)/4, and EOS pressure p_n = −g ρ₀ (e_top − Z_0p − (n−1) Δz/4)
   with Z_0p = 0 (mean sea level: η and p_atm are not in the EOS pressure, MOM6 default
   SSH_IN_EOS_PRESSURE_FOR_PGF = False, MOM_PressureForce_FV.F90:1262-1274);
       r_n = ρ(T_n, S_n, p_n) − ρ_ref     (MOM6 anomaly form inside the EOS, RHO_PGF_REF)
       dpa = g Δz B(r),   B(r) = (7(r₁+r₅) + 32(r₂+r₄) + 12 r₃)/90                      (Boole)
       ∫∫dpa = ½ g Δz² (B(r) − (16(r₄−r₂) + 7(r₅−r₁))/90).
4. Per face (:643-751): corner values of T and S, optionally mass weighted (5), three interior
   sub-columns at x-fractions ¼, ½, ¾ with thickness, top pressure, T and S linear in x and the 5
   points linear in z (15 EOS calls per face and layer); ∫dpa = Boole in x of g Δz_x B(r) with the
   two column values dpa_L, dpa_R at the ends.
5. Mass weighting (MOM_density_integrals.F90:652-685; MOM_PressureForce_FV.F90:2124-2149),
   ``mass_weight``:
       "vanished"  MASS_WEIGHT_IN_PRESSURE_GRADIENT = True, MASS_WEIGHT_IN_PGF_VANISHED_ONLY = True
                   with RESET_INTXPA_H_NONVANISHED = ``h_vanished`` (pyOCEAN's linear default);
       "full"      MASS_WEIGHT_IN_PRESSURE_GRADIENT = True, VANISHED_ONLY = False (NWA12);
       "none"      MASS_WEIGHT_IN_PRESSURE_GRADIENT = False.
   MASS_WEIGHT_IN_PRESSURE_GRADIENT_TOP is False in all three (NWA12).  The weight
   h_w = max(0, −D_L − e_R,top, −D_R − e_L,top) · ((h_L − h_R)/(h_L + h_R))² biases only the corner
   T and S towards the thicker cell; Δz and p stay linear in x.
6. Assembly (MOM_PressureForce_FV.F90:1338-1343, 1539-1556): pa_{k+1} = pa_k + dpa_k,
   ∫pa_{k+1} = ∫pa_k + ∫dpa_k, ∫pa_1 = ½(pa_L + pa_R).

Surface term.  pyOCEAN returns the baroclinic force; the barotropic solver supplies −g ∇η (as in
``core.vertical.pgf``), so the surface anomaly is pa_1 = g η (ρ_ref − ρ₀), not MOM6's g ρ_ref η
(MOM_PressureForce_FV.F90:1257-1259).  The two differ by exactly −g Δη/Δx at every layer.

Two modes (spec goal B).  Without a profile (``T_ref = S_ref = z_ref = None``) it is MOM6's method,
compared with compiled MOM6 to 1e-12 (tests/test_pgf_nonlinear.py; validation/mom6_column/
pgf_driver.F90).  With a profile T_ref(z), S_ref(z) (pyOCEAN only, no MOM6 counterpart), PLM acts on
the anomalies T′ = T − T_ref(z_c), S′ = S − S_ref(z_c) at the layer centres and each quadrature
point is evaluated as
       r_q = ρ(T_ref(z_q) + T′_q, S_ref(z_q) + S′_q, p_q) − ρ(T_ref(z_q), S_ref(z_q), p_q),
both terms in the anomaly form with ρ_ref = ``rho_pgf_ref``.  The pressure of ρ_ref(z) =
ρ(T_ref, S_ref, −g ρ₀ z) depends on z only, so its contour integral vanishes and a column equal to
the profile gives no force on any layers, partial bed cells and sloping hybrid layers included
(thermobaricity and the curvature of the mean stratification never enter the reconstruction
error).  The surface anomaly is then g η (ρ_ref(η/2) − ρ₀).  T_ref and S_ref are interpolated
linearly in z_ref and held constant beyond its ends.

Land and dry columns.  A column is inactive where ``wet`` is False (if given), H or η is not
finite, any h is not finite, or Σh = 0; its values are never read and the faces next to it get no
force.

References: Adcroft, A., Hallberg, R. & Harrison, M. (2008). A finite volume discretization of the
pressure gradient force using analytic integration. Ocean Modell. 22, 106–113.  Wright (1997) and
Roquet et al. (2015) for the equations of state (physics/eos/wright_full.py, roquet_rho.py).
MOM6 source git 0e9a561: core/MOM_density_integrals.F90:420-751, core/MOM_PressureForce_FV.F90
(PressureForce_FV_Bouss), ALE/MOM_ALE.F90:1512-1596, ALE/PLM_functions.F90.
"""

from __future__ import annotations

import numpy as np
from numba import njit, prange

from core.vertical.pgf import H_VANISHED
from core.vertical.remap import _EPS, H_NEGLECT, _plm_monotonized_slope, _plm_slope_wa
from physics.eos import roquet_rho, wright_full
from physics.eos.roquet_rho import rho_anom_elem as _roquet_anom
from physics.eos.wright_full import rho_anom_elem as _wright_anom

EOS_IDS = {"wright_full": 0, "teos10": 1}         # teos10 = MOM6 ROQUET_RHO (physics/eos/roquet_rho.py)
MASS_WEIGHT_MODES = ("vanished", "full", "none")
DZ_SUBROUNDOFF = 1e-30   # [m] MOM6 GV%dZ_subroundoff = 1e-20 max(Angstrom = 1e-10 m, 1e-17 m), MOM_verticalGrid.F90:220
Z_0P = 0.0               # [m] height of zero EOS pressure (G%meanSL)
C1_90 = 1.0 / 90.0       # Boole weights, as MOM6 multiplies by C1_90 (MOM_density_integrals.F90:617)


# Not disk-cached: calls jitted functions of other files (Numba's cache only checks this file).
@njit(inline="always")
def _anom(eos, T, S, p, rho_ref):  # pragma: no cover
    """ρ(T, S, p) − ρ_ref [kg m⁻³] in MOM6's anomaly form; eos 0 = WRIGHT_FULL, 1 = ROQUET_RHO."""
    if eos == 0:
        return _wright_anom(T, S, p, rho_ref)
    return _roquet_anom(T, S, p, rho_ref)


@njit(inline="always")
def _interp(z, zr, fr):  # pragma: no cover
    """Linear interpolation of fr(zr) at z (zr increasing), constant beyond the ends."""
    m = zr.size
    if z <= zr[0]:
        return fr[0]
    if z >= zr[m - 1]:
        return fr[m - 1]
    lo = 0
    hi = m - 1                                       # zr[lo] < z < zr[hi]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if zr[mid] <= z:
            lo = mid
        else:
            hi = mid
    return fr[lo] + (fr[lo + 1] - fr[lo]) * ((z - zr[lo]) / (zr[lo + 1] - zr[lo]))


@njit(inline="always")
def _extrapolate_slope(h_l, h_c, u_l, u_c):  # pragma: no cover
    """PLM_extrapolate_slope (ALE/PLM_functions.F90:164-183): 2 (u_c − edge), edge h-weighted."""
    hl = h_l + H_NEGLECT
    hc = h_c + H_NEGLECT
    left_edge = (u_l * hc + u_c * hl) / (hl + hc)
    return 2.0 * (u_c - left_edge)


@njit
def _plm_edges(h, Q, Qt, Qb, slp):  # pragma: no cover
    """
    Top and bottom edge values of Q in each layer of a SURFACE-FIRST column, ALE_PLM_edge_values
    with bdry_extrap = True (ALE/MOM_ALE.F90:1561-1593).  One layer: PCM (MOM6 needs ≥ 2).
    ``slp`` (n,) is scratch (the caller's per-row buffer, so no allocation per column).
    """
    n = h.size
    if n == 1:
        Qt[0] = Q[0]
        Qb[0] = Q[0]
        return
    slp[0] = 0.0
    slp[n - 1] = 0.0
    for k in range(1, n - 1):
        slp[k] = _plm_slope_wa(h[k - 1], h[k], h[k + 1], H_NEGLECT, Q[k - 1], Q[k], Q[k + 1])
    for k in range(1, n - 1):
        m = _plm_monotonized_slope(Q[k - 1], Q[k], Q[k + 1], slp[k - 1], slp[k], slp[k + 1])
        Qt[k] = Q[k] - 0.5 * m
        Qb[k] = Q[k] + 0.5 * m
    m = -_extrapolate_slope(h[1], h[0], Q[1], Q[0])
    Qt[0] = Q[0] - 0.5 * m
    Qb[0] = Q[0] + 0.5 * m
    m = _extrapolate_slope(h[n - 2], h[n - 1], Q[n - 2], Q[n - 1])
    Qt[n - 1] = Q[n - 1] - 0.5 * m
    Qb[n - 1] = Q[n - 1] + 0.5 * m


@njit(inline="always")
def _ref_ts(z, zr, Tr, Sr, lo):  # pragma: no cover
    """
    (T_ref(z), S_ref(z), lo): ``_interp`` of both profiles at z with one shared index search, started from the
    previous point's index ``lo`` (0 ≤ lo ≤ M − 2) and walked to the unique lo with zr[lo] ≤ z < zr[lo+1] that
    the bisection of ``_interp`` finds (zr strictly increasing), so the values are those of ``_interp`` bit for
    bit; the quadrature points of a column are visited top to bottom, so the walk is short.
    """
    m = zr.size
    if z <= zr[0]:
        return Tr[0], Sr[0], lo
    if z >= zr[m - 1]:
        return Tr[m - 1], Sr[m - 1], lo
    while zr[lo] > z:
        lo -= 1
    while zr[lo + 1] <= z:
        lo += 1
    f = (z - zr[lo]) / (zr[lo + 1] - zr[lo])
    return Tr[lo] + (Tr[lo + 1] - Tr[lo]) * f, Sr[lo] + (Sr[lo + 1] - Sr[lo]) * f, lo


@njit(inline="always")
def _point(eos, T, S, p, z, rho_ref, profile, zr, Tr, Sr, lo):  # pragma: no cover
    """(density anomaly at one quadrature point, profile index): ρ(T,S,p) − ρ_ref, or the profile form
    (T, S = T′, S′); ``lo`` is the index hint of ``_ref_ts``."""
    if profile:
        tr, sr, lo = _ref_ts(z, zr, Tr, Sr, lo)
        return _anom(eos, tr + T, sr + S, p, rho_ref) - _anom(eos, tr, sr, p, rho_ref), lo
    return _anom(eos, T, S, p, rho_ref), lo


@njit(parallel=True)
def _columns(H, hs, Ts, Ss, active, eos, rho0, rho_ref, g, profile, zr, Tr, Sr,
             e, Tt, Tb, St, Sb, dpa, intz):  # pragma: no cover
    """
    Column pass, surface first (k = 0 at the surface): interfaces e (N+1, ny, nx) [m], PLM edges of
    T, S (or T′, S′) and the layer integrals dpa [Pa] and ∫∫dpa [Pa m] (MOM_density_integrals.F90:590-640).
    Inactive columns (``active`` False) get dpa = ∫∫dpa = 0 without EOS calls: ``_faces`` never reads them.
    """
    n, ny, nx = hs.shape
    GxRho = g * rho0
    for j in prange(ny):
        hc = np.empty(n)
        q = np.empty(n)
        q2 = np.empty(n)
        qt = np.empty(n)
        qb = np.empty(n)
        slp = np.empty(n)
        r = np.empty(5)
        for i in range(nx):
            lo = zr.size - 2                                 # profile index hint, walked down the column
            e[n, j, i] = -H[j, i]
            for k in range(n - 1, -1, -1):
                e[k, j, i] = e[k + 1, j, i] + hs[k, j, i]
            for k in range(n):
                hc[k] = hs[k, j, i]
            if profile:                                      # anomalies T′, S′ at the layer centres
                for k in range(n):
                    zc = 0.5 * (e[k, j, i] + e[k + 1, j, i])
                    tr, sr, lo = _ref_ts(zc, zr, Tr, Sr, lo)
                    q[k] = Ts[k, j, i] - tr
                    q2[k] = Ss[k, j, i] - sr
                lo = zr.size - 2
            else:
                for k in range(n):
                    q[k] = Ts[k, j, i]
                    q2[k] = Ss[k, j, i]
            _plm_edges(hc, q, qt, qb, slp)
            for k in range(n):
                Tt[k, j, i] = qt[k]
                Tb[k, j, i] = qb[k]
            _plm_edges(hc, q2, qt, qb, slp)
            for k in range(n):
                St[k, j, i] = qt[k]
                Sb[k, j, i] = qb[k]
            if not active[j, i]:
                for k in range(n):
                    dpa[k, j, i] = 0.0
                    intz[k, j, i] = 0.0
                continue
            for k in range(n):
                dz = e[k, j, i] - e[k + 1, j, i]
                for m in range(5):
                    wt = 0.25 * (4 - m)
                    wb = 1.0 - wt
                    z = (e[k, j, i] - Z_0P) - 0.25 * m * dz
                    p = -GxRho * z
                    r[m], lo = _point(eos, wt * Tt[k, j, i] + wb * Tb[k, j, i], wt * St[k, j, i] + wb * Sb[k, j, i],
                                      p, z, rho_ref, profile, zr, Tr, Sr, lo)
                ra = C1_90 * (7.0 * (r[0] + r[4]) + 32.0 * (r[1] + r[3]) + 12.0 * r[2])
                dpa[k, j, i] = g * dz * ra
                intz[k, j, i] = 0.5 * g * (dz * dz) * (ra - C1_90 * (16.0 * (r[3] - r[1]) + 7.0 * (r[4] - r[0])))


@njit(inline="always")
def _weighted(hw, hl, hr, aL, aR):  # pragma: no cover
    """Mass-weighted corner values (left, right) of a field (MOM_density_integrals.F90:676-683)."""
    iD = 1.0 / (hw * (hr + hl) + hl * hr)
    return ((hw * hr) * aR + (hw * hl + hr * hl) * aL) * iD, ((hw * hl) * aL + (hw * hr + hr * hl) * aR) * iD


@njit(parallel=True)
def _faces(e, hs, Tt, Tb, St, Sb, dpa, intz, pa0, active, dx, eos, rho0, rho_ref, g, profile, zr, Tr, Sr,
           mw_toggle, nv_toggle, h_nv, out):  # pragma: no cover
    """
    Force on the interior u-faces (columns i−1, i), surface first; out (N, ny, nx+1) [m s⁻²] must be
    zero on entry (faces next to inactive columns are left at zero).  dx (ny, nx+1) [m].
    """
    n, ny, nx = hs.shape
    GxRho = g * rho0
    I_Rho0 = 1.0 / rho0
    for j in prange(ny):
        r = np.empty(5)
        lov = np.empty(3, np.int64)                          # profile index hints of the three sub-columns
        for i in range(1, nx):
            L = i - 1
            R = i
            if not (active[j, L] and active[j, R]):
                continue
            lov[:] = zr.size - 2
            paL = pa0[j, L]
            paR = pa0[j, R]
            ixpa = 0.5 * (paL + paR)
            idx = 1.0 / dx[j, i]
            for k in range(n):
                eLt = e[k, j, L]
                eLb = e[k + 1, j, L]
                eRt = e[k, j, R]
                eRb = e[k + 1, j, R]
                hw = mw_toggle * max(0.0, e[n, j, L] - eRt, e[n, j, R] - eLt)       # −D_L − e_R, −D_R − e_L
                if ((eLt - eLb) > h_nv) and ((eRt - eRb) > h_nv):
                    hw = nv_toggle * hw
                ttl = Tt[k, j, L]
                tbl = Tb[k, j, L]
                ttr = Tt[k, j, R]
                tbr = Tb[k, j, R]
                stl = St[k, j, L]
                sbl = Sb[k, j, L]
                str_ = St[k, j, R]
                sbr = Sb[k, j, R]
                if hw > 0.0:
                    hl = (eLt - eLb) + DZ_SUBROUNDOFF
                    hr = (eRt - eRb) + DZ_SUBROUNDOFF
                    q = (hl - hr) / (hl + hr)
                    hw = hw * (q * q)
                    ttl, ttr = _weighted(hw, hl, hr, Tt[k, j, L], Tt[k, j, R])
                    tbl, tbr = _weighted(hw, hl, hr, Tb[k, j, L], Tb[k, j, R])
                    stl, str_ = _weighted(hw, hl, hr, St[k, j, L], St[k, j, R])
                    sbl, sbr = _weighted(hw, hl, hr, Sb[k, j, L], Sb[k, j, R])
                I1 = 0.0
                I2 = 0.0
                I3 = 0.0
                for mm in range(3):                                # sub-columns at x = ¼, ½, ¾
                    wl = 0.25 * (3 - mm)
                    wr = 1.0 - wl
                    dzx = (wl * (eLt - eLb)) + (wr * (eRt - eRb))
                    t1 = (wl * ttl) + (wr * ttr)
                    t5 = (wl * tbl) + (wr * tbr)
                    s1 = (wl * stl) + (wr * str_)
                    s5 = (wl * sbl) + (wr * sbr)
                    z = (wl * (eLt - Z_0P)) + (wr * (eRt - Z_0P))
                    p = -GxRho * z
                    for m in range(5):
                        if m > 0:
                            p = p + GxRho * 0.25 * dzx
                            z = z - 0.25 * dzx
                        if m == 0:
                            tq = t1
                            sq = s1
                        elif m == 4:
                            tq = t5
                            sq = s5
                        else:
                            wt = 0.25 * (4 - m)
                            wb = 1.0 - wt
                            tq = wt * t1 + wb * t5
                            sq = wt * s1 + wb * s5
                        r[m], lov[mm] = _point(eos, tq, sq, p, z, rho_ref, profile, zr, Tr, Sr, lov[mm])
                    v = g * dzx * (C1_90 * (7.0 * (r[0] + r[4]) + 32.0 * (r[1] + r[3]) + 12.0 * r[2]))
                    if mm == 0:
                        I1 = v
                    elif mm == 1:
                        I2 = v
                    else:
                        I3 = v
                dpaL = dpa[k, j, L]
                dpaR = dpa[k, j, R]
                ixdpa = C1_90 * (7.0 * (dpaL + dpaR) + 32.0 * (I1 + I3) + 12.0 * I2)
                hL = hs[k, j, L]
                hR = hs[k, j, R]
                out[k, j, i] = (((paL * hL + intz[k, j, L]) - (paR * hR + intz[k, j, R]))
                                + ((hR - hL) * ixpa - (e[k + 1, j, R] - e[k + 1, j, L]) * ixdpa)) \
                    * ((2.0 * I_Rho0 * idx) / ((hL + hR) + H_NEGLECT))
                paL += dpaL
                paR += dpaR
                ixpa += ixdpa


def _layer_centres(h: np.ndarray, H: np.ndarray) -> np.ndarray:
    """Layer-centre heights z_c [m] (bed first) exactly as the kernels compute them: ½(e_k + e_{k+1})
    with the interfaces summed from the bed up (e_bed = −H)."""
    h = np.asarray(h, float)
    hs = h[::-1]
    e = np.empty((h.shape[0] + 1,) + h.shape[1:])
    e[-1] = -np.asarray(H, float)
    for k in range(h.shape[0] - 1, -1, -1):
        e[k] = e[k + 1] + hs[k]
    return (0.5 * (e[:-1] + e[1:]))[::-1].copy()


def _interp_profile_numpy(z: np.ndarray, zr: np.ndarray, fr: np.ndarray) -> np.ndarray:
    """NumPy oracle of ``_interp`` (same index search and the same formula)."""
    z = np.asarray(z, float)
    m = zr.size
    lo = np.clip(np.searchsorted(zr, z, side="right") - 1, 0, m - 2)
    v = fr[lo] + (fr[lo + 1] - fr[lo]) * ((z - zr[lo]) / (zr[lo + 1] - zr[lo]))
    return np.where(z <= zr[0], fr[0], np.where(z >= zr[m - 1], fr[m - 1], v))


def _prepare(h, H, eta, T, S, du, dv, eos_name, rho0, rho_pgf_ref, g, mass_weight, T_ref, S_ref, z_ref, wet):
    """Check the arguments; surface-first arrays with inactive columns zeroed; surface anomaly pa0."""
    if eos_name not in EOS_IDS:
        raise ValueError(f"eos_name must be one of {tuple(EOS_IDS)}, got {eos_name!r}")
    if mass_weight not in MASS_WEIGHT_MODES:
        raise ValueError(f"mass_weight must be one of {MASS_WEIGHT_MODES}, got {mass_weight!r}")
    given = [a is not None for a in (T_ref, S_ref, z_ref)]
    if any(given) and not all(given):
        raise ValueError("profile mode needs T_ref, S_ref and z_ref together (or none of them)")
    profile = all(given)
    if profile:
        zr = np.ascontiguousarray(z_ref, float)
        Tr = np.ascontiguousarray(T_ref, float)
        Sr = np.ascontiguousarray(S_ref, float)
        if zr.ndim != 1 or zr.size < 2 or Tr.shape != zr.shape or Sr.shape != zr.shape:
            raise ValueError("T_ref, S_ref, z_ref must be 1-D profiles of the same length ≥ 2")
        if not np.all(np.diff(zr) > 0.0):
            raise ValueError("z_ref must increase strictly (k = 0 at the bed)")
    else:
        zr = Tr = Sr = np.zeros(2)
    h = np.asarray(h, float)
    H = np.asarray(H, float)
    eta = np.asarray(eta, float)
    T = np.asarray(T, float)
    S = np.asarray(S, float)
    active = np.isfinite(H) & np.isfinite(eta) & np.all(np.isfinite(h), 0) & (h.sum(0) > 0.0)
    if wet is not None:
        active &= np.asarray(wet, bool)
    a3 = active[None]
    hs = np.ascontiguousarray(np.where(a3, h, 0.0)[::-1])
    Ts = np.ascontiguousarray(np.where(a3, T, 0.0)[::-1])
    Ss = np.ascontiguousarray(np.where(a3, S, 0.0)[::-1])
    H0 = np.where(active, H, 0.0)
    eta0 = np.where(active, eta, 0.0)
    eos = EOS_IDS[eos_name]
    if profile:          # g η (ρ_ref(η/2) − ρ₀), ρ_ref(z) = ρ(T_ref(z), S_ref(z), −g ρ₀ z)
        zh = 0.5 * eta0
        mod = wright_full if eos == 0 else roquet_rho
        d_ref = mod.density_anomaly_numpy(_interp_profile_numpy(zh, zr, Tr), _interp_profile_numpy(zh, zr, Sr),
                                          -(g * rho0) * zh, rho0)
        pa0 = g * eta0 * d_ref
    else:                # g η (ρ_ref − ρ₀): the −g ρ₀ ∇η part is the barotropic solver's
        pa0 = g * eta0 * (rho_pgf_ref - rho0)
    mw_toggle = 0.0 if mass_weight == "none" else 1.0
    nv_toggle = 0.0 if mass_weight == "vanished" else 1.0
    return dict(hs=hs, Ts=Ts, Ss=Ss, H=H0, pa0=np.ascontiguousarray(pa0), active=active, eos=eos,
                profile=profile, zr=zr, Tr=Tr, Sr=Sr, mw_toggle=mw_toggle, nv_toggle=nv_toggle,
                du=np.ascontiguousarray(du, float), dv=np.ascontiguousarray(dv, float))


def fv_pgf_nonlinear(h: np.ndarray, H: np.ndarray, eta: np.ndarray, T: np.ndarray, S: np.ndarray,
                     du: np.ndarray, dv: np.ndarray, *, eos_name: str, rho0: float, rho_pgf_ref: float, g: float,
                     mass_weight: str = "vanished", T_ref=None, S_ref=None, z_ref=None,
                     h_vanished: float = H_VANISHED, wet=None) -> tuple[np.ndarray, np.ndarray]:
    """
    Baroclinic finite-volume pressure-gradient force [m s⁻²] with a nonlinear EOS (module docstring).

    h, T, S : (N, ny, nx) layer thickness [m], temperature [°C] (θ for "wright_full", Θ for "teos10")
        and salinity (PSU for "wright_full", g kg⁻¹ S_A for "teos10"), k = 0 at the bed
    H, eta : (ny, nx) bed depth and surface height [m]
    du (ny, nx+1), dv (ny+1, nx) : centre-to-centre distances across the u- and v-faces [m]
    eos_name : "wright_full" (Wright 1997, MOM6 WRIGHT_FULL) or "teos10" (Roquet et al. 2015, MOM6 ROQUET_RHO)
    rho0 : Boussinesq density [kg m⁻³]: EOS pressure −g ρ₀ z and the 1/ρ₀ factor
    rho_pgf_ref : density subtracted inside the EOS [kg m⁻³] (MOM6 RHO_PGF_REF)
    g : gravitational acceleration [m s⁻²]
    mass_weight : "vanished", "full" or "none" (module docstring, point 5)
    T_ref, S_ref, z_ref : (M,) reference profiles at heights z_ref [m], increasing (k = 0 at the bed),
        for the profile mode; None (all three) for MOM6's method
    h_vanished : [m] thickness at or below which a layer counts as vanished ("vanished" mode)
    wet : optional (ny, nx) bool, False for dry or land columns (their faces get no force)
    Returns gx (N, ny, nx+1), gy (N, ny+1, nx), zero on the domain-edge faces.
    """
    a = _prepare(h, H, eta, T, S, du, dv, eos_name, rho0, rho_pgf_ref, g, mass_weight, T_ref, S_ref, z_ref, wet)
    hs = a["hs"]
    n, ny, nx = hs.shape
    e = np.empty((n + 1, ny, nx))
    Tt, Tb, St, Sb, dpa, intz = (np.empty((n, ny, nx)) for _ in range(6))
    _columns(a["H"], hs, a["Ts"], a["Ss"], a["active"], a["eos"], float(rho0), float(rho_pgf_ref), float(g), a["profile"],
             a["zr"], a["Tr"], a["Sr"], e, Tt, Tb, St, Sb, dpa, intz)
    rest = (a["eos"], float(rho0), float(rho_pgf_ref), float(g), a["profile"], a["zr"], a["Tr"], a["Sr"],
            a["mw_toggle"], a["nv_toggle"], float(h_vanished))
    gx = np.zeros((n, ny, nx + 1))
    _faces(e, hs, Tt, Tb, St, Sb, dpa, intz, a["pa0"], a["active"], a["du"], *rest, gx)
    sw = lambda x: np.swapaxes(x, -1, -2)                     # noqa: E731  strided view, no copy
    gyT = np.zeros((n, nx, ny + 1))
    _faces(sw(e), sw(hs), sw(Tt), sw(Tb), sw(St), sw(Sb), sw(dpa), sw(intz), sw(a["pa0"]), sw(a["active"]),
           sw(a["dv"]), *rest, gyT)
    return gx[::-1].copy(), np.swapaxes(gyT, -1, -2)[::-1].copy()


# ---------------------------------------------------------------------------------------------
# NumPy oracle (same operations in the same order, vectorised over columns and faces)
# ---------------------------------------------------------------------------------------------

def _slope_wa_numpy(h_l, h_c, h_r, u_l, u_c, u_r):
    """Vectorised remap._plm_slope_wa."""
    sigma_r = u_r - u_c
    sigma_l = u_c - u_l
    sigma_c = 2.0 * (u_r - u_l) * (h_c / (h_l + 2.0 * h_c + h_r + H_NEGLECT))
    u_min = np.minimum(np.minimum(u_l, u_c), u_r)
    u_max = np.maximum(np.maximum(u_l, u_c), u_r)
    s = np.minimum(np.abs(sigma_c), 2.0 * np.minimum(u_c - u_min, u_max - u_c))
    slope = np.where(sigma_l * sigma_r > 0.0, np.where(sigma_c >= 0.0, s, -s), 0.0)
    over = (u_c - 0.5 * np.abs(slope) < u_min) | (u_c + 0.5 * np.abs(slope) > u_max)
    slope = np.where(over, slope * (1.0 - _EPS), slope)
    return np.where(np.abs(slope) < 1e-140, 0.0, slope)


def _monotonized_numpy(u_l, u_c, u_r, s_l, s_c, s_r):
    """Vectorised remap._plm_monotonized_slope."""
    almost_two = 2.0 * (1.0 - _EPS)
    e_r = u_l + 0.5 * s_l
    e_l = u_r - 0.5 * s_r
    slp = np.abs(s_c)
    edge = u_c - 0.5 * s_c
    c = (edge - e_r) * (u_c - edge) < 0.0
    slp = np.where(c, np.minimum(slp, np.abs(0.5 * (edge + e_r) - u_c) * almost_two), slp)
    edge = u_c + 0.5 * s_c
    c = (edge - u_c) * (e_l - edge) < 0.0
    slp = np.where(c, np.minimum(slp, np.abs(0.5 * (edge + e_l) - u_c) * almost_two), slp)
    return np.where(s_c >= 0.0, slp, -slp)


def _extrapolate_numpy(h_l, h_c, u_l, u_c):
    hl = h_l + H_NEGLECT
    hc = h_c + H_NEGLECT
    return 2.0 * (u_c - (u_l * hc + u_c * hl) / (hl + hc))


def _plm_edges_numpy(h, Q):
    """Vectorised _plm_edges over columns: h, Q (N, ...) surface first → (Qt, Qb)."""
    n = h.shape[0]
    Qt = np.empty_like(Q)
    Qb = np.empty_like(Q)
    if n == 1:
        Qt[0] = Q[0]
        Qb[0] = Q[0]
        return Qt, Qb
    slp = np.zeros_like(Q)
    for k in range(1, n - 1):
        slp[k] = _slope_wa_numpy(h[k - 1], h[k], h[k + 1], Q[k - 1], Q[k], Q[k + 1])
    for k in range(1, n - 1):
        m = _monotonized_numpy(Q[k - 1], Q[k], Q[k + 1], slp[k - 1], slp[k], slp[k + 1])
        Qt[k] = Q[k] - 0.5 * m
        Qb[k] = Q[k] + 0.5 * m
    m = -_extrapolate_numpy(h[1], h[0], Q[1], Q[0])
    Qt[0] = Q[0] - 0.5 * m
    Qb[0] = Q[0] + 0.5 * m
    m = _extrapolate_numpy(h[n - 2], h[n - 1], Q[n - 2], Q[n - 1])
    Qt[n - 1] = Q[n - 1] - 0.5 * m
    Qb[n - 1] = Q[n - 1] + 0.5 * m
    return Qt, Qb


def _point_numpy(a, T, S, p, z, rho_ref):
    mod = wright_full if a["eos"] == 0 else roquet_rho
    if a["profile"]:
        tr = _interp_profile_numpy(z, a["zr"], a["Tr"])
        sr = _interp_profile_numpy(z, a["zr"], a["Sr"])
        return mod.density_anomaly_numpy(tr + T, sr + S, p, rho_ref) - mod.density_anomaly_numpy(tr, sr, p, rho_ref)
    return mod.density_anomaly_numpy(T, S, p, rho_ref)


def _faces_numpy(a, e, hs, Tt, Tb, St, Sb, dpa, intz, pa0, active, dx, rho0, rho_ref, g, h_nv):
    """Vectorised _faces: e, hs, … (…, ny, nx) surface first → force (N, ny, nx+1)."""
    n = hs.shape[0]
    GxRho = g * rho0
    I_Rho0 = 1.0 / rho0
    out = np.zeros(hs.shape[:2] + (hs.shape[2] + 1,))
    L = (slice(None), slice(None), slice(0, -1))
    R = (slice(None), slice(None), slice(1, None))
    ok = active[:, :-1] & active[:, 1:]
    idx = 1.0 / dx[:, 1:-1]
    paL = pa0[:, :-1]
    paR = pa0[:, 1:]
    ixpa = 0.5 * (paL + paR)
    with np.errstate(divide="ignore", invalid="ignore"):
        for k in range(n):
            eLt, eLb, eRt, eRb = e[k][L[1:]], e[k + 1][L[1:]], e[k][R[1:]], e[k + 1][R[1:]]
            hw = a["mw_toggle"] * np.maximum(np.maximum(0.0, e[n][L[1:]] - eRt), e[n][R[1:]] - eLt)
            nv = ((eLt - eLb) > h_nv) & ((eRt - eRb) > h_nv)
            hw = np.where(nv, a["nv_toggle"] * hw, hw)
            hl = (eLt - eLb) + DZ_SUBROUNDOFF
            hr = (eRt - eRb) + DZ_SUBROUNDOFF
            q = (hl - hr) / (hl + hr)
            hw2 = hw * (q * q)
            iD = 1.0 / (hw2 * (hr + hl) + hl * hr)
            use = hw > 0.0
            corners = []
            for f in (Tt, Tb, St, Sb):
                fL, fR = f[k][L[1:]], f[k][R[1:]]
                wl = ((hw2 * hr) * fR + (hw2 * hl + hr * hl) * fL) * iD
                wr = ((hw2 * hl) * fL + (hw2 * hr + hr * hl) * fR) * iD
                corners.append((np.where(use, wl, fL), np.where(use, wr, fR)))
            (ttl, ttr), (tbl, tbr), (stl, str_), (sbl, sbr) = corners
            Iv = []
            for mm in range(3):
                wl = 0.25 * (3 - mm)
                wr = 1.0 - wl
                dzx = (wl * (eLt - eLb)) + (wr * (eRt - eRb))
                t1 = (wl * ttl) + (wr * ttr)
                t5 = (wl * tbl) + (wr * tbr)
                s1 = (wl * stl) + (wr * str_)
                s5 = (wl * sbl) + (wr * sbr)
                z = (wl * (eLt - Z_0P)) + (wr * (eRt - Z_0P))
                p = -GxRho * z
                r = []
                for m in range(5):
                    if m > 0:
                        p = p + GxRho * 0.25 * dzx
                        z = z - 0.25 * dzx
                    if m == 0:
                        tq, sq = t1, s1
                    elif m == 4:
                        tq, sq = t5, s5
                    else:
                        wt = 0.25 * (4 - m)
                        wb = 1.0 - wt
                        tq = wt * t1 + wb * t5
                        sq = wt * s1 + wb * s5
                    r.append(_point_numpy(a, tq, sq, p, z, rho_ref))
                Iv.append(g * dzx * (C1_90 * (7.0 * (r[0] + r[4]) + 32.0 * (r[1] + r[3]) + 12.0 * r[2])))
            dpaL, dpaR = dpa[k][L[1:]], dpa[k][R[1:]]
            ixdpa = C1_90 * (7.0 * (dpaL + dpaR) + 32.0 * (Iv[0] + Iv[2]) + 12.0 * Iv[1])
            hL, hR = hs[k][L[1:]], hs[k][R[1:]]
            f = ((((paL * hL + intz[k][L[1:]]) - (paR * hR + intz[k][R[1:]]))
                  + ((hR - hL) * ixpa - (e[k + 1][R[1:]] - e[k + 1][L[1:]]) * ixdpa))
                 * ((2.0 * I_Rho0 * idx) / ((hL + hR) + H_NEGLECT)))
            out[k, :, 1:-1] = np.where(ok, f, 0.0)
            paL = paL + dpaL
            paR = paR + dpaR
            ixpa = ixpa + ixdpa
    return out


def _fv_pgf_nonlinear_numpy(h, H, eta, T, S, du, dv, *, eos_name, rho0, rho_pgf_ref, g, mass_weight="vanished",
                            T_ref=None, S_ref=None, z_ref=None, h_vanished=H_VANISHED, wet=None):
    """NumPy oracle of :func:`fv_pgf_nonlinear` (bit for bit; tests/test_pgf_nonlinear.py)."""
    a = _prepare(h, H, eta, T, S, du, dv, eos_name, rho0, rho_pgf_ref, g, mass_weight, T_ref, S_ref, z_ref, wet)
    hs = a["hs"]
    n = hs.shape[0]
    e = np.empty((n + 1,) + hs.shape[1:])
    e[n] = -a["H"]
    for k in range(n - 1, -1, -1):
        e[k] = e[k + 1] + hs[k]
    Tq, Sq = a["Ts"], a["Ss"]
    if a["profile"]:
        zc = 0.5 * (e[:-1] + e[1:])
        Tq = Tq - _interp_profile_numpy(zc, a["zr"], a["Tr"])
        Sq = Sq - _interp_profile_numpy(zc, a["zr"], a["Sr"])
    Tt, Tb = _plm_edges_numpy(hs, Tq)
    St, Sb = _plm_edges_numpy(hs, Sq)
    GxRho = g * rho0
    dpa = np.empty_like(hs)
    intz = np.empty_like(hs)
    for k in range(n):
        dz = e[k] - e[k + 1]
        r = []
        for m in range(5):
            wt = 0.25 * (4 - m)
            wb = 1.0 - wt
            z = (e[k] - Z_0P) - 0.25 * m * dz
            r.append(_point_numpy(a, wt * Tt[k] + wb * Tb[k], wt * St[k] + wb * Sb[k], -GxRho * z, z, rho_pgf_ref))
        ra = C1_90 * (7.0 * (r[0] + r[4]) + 32.0 * (r[1] + r[3]) + 12.0 * r[2])
        dpa[k] = g * dz * ra
        intz[k] = 0.5 * g * (dz * dz) * (ra - C1_90 * (16.0 * (r[3] - r[1]) + 7.0 * (r[4] - r[0])))
    args = (rho0, rho_pgf_ref, g, h_vanished)
    gx = _faces_numpy(a, e, hs, Tt, Tb, St, Sb, dpa, intz, a["pa0"], a["active"], a["du"], *args)
    sw = lambda x: np.swapaxes(x, -1, -2)                     # noqa: E731
    gyT = _faces_numpy(a, sw(e), sw(hs), sw(Tt), sw(Tb), sw(St), sw(Sb), sw(dpa), sw(intz), sw(a["pa0"]),
                       sw(a["active"]), sw(a["dv"]), *args)
    return gx[::-1].copy(), np.swapaxes(gyT, -1, -2)[::-1].copy()
