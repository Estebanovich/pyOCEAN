"""
Compiled GLS k-ε step on ALE columns with thin layers (N3 task 10a; carry from the task-5 review).

``GLSClosure._update_ale`` (core/components/closure.py) merges every layer thinner than ``h_massless`` with the
nearest layer with mass above it (below it at the top of the column), steps the k-ε GLS on the merged column and
interpolates the interface values back to the model interfaces (MOM6 ``eliminate_massless``,
MOM_kappa_shear.F90:254-307 and 355-363; class docstring of ``GLSClosure``).  Its NumPy form groups the columns by
the number of merged layers and builds one ``GLSKEpsilon`` per group every step; this module does the same per
column in one Numba kernel (``merged_columns``, threads over the columns), with the arithmetic of
``GLSKEpsilon.step`` / ``_step_stratified`` / ``_solve`` (CROCO gls_mixing.F port, ``physics.mixing.gls``) written
for one column of c merged layers.  The NumPy path is kept as the oracle
(``GLSClosure._update_ale_numpy``, ``tests/test_gls_ale.py::test_gls_merge_numba_matches_numpy``, rtol 1e-13).

Equations: as ``physics.mixing.gls`` (Umlauf & Burchard 2003, eqs. 11–12; Canuto et al. 2001 A; Patankar split of
B and of Burchard's 2002 P < 0; Galperin limit l ≤ 0.53 √(2k)/N).  Index order: layers k = 0 … N−1 from the bed,
interfaces 0 … N (0 = bed), columns last: (N, M) and (N+1, M).
"""
from __future__ import annotations

import numpy as np
from numba import njit, prange

from physics.mixing.gls import (BETA1, BETA2, BETA3M, BETA3P, E1, E2, E3, KAPPA, LIM_AM0, LIM_AM1, LIM_AM2, LIM_AM3,
                                LIM_AM4, LIM_AM5, LIM_AM6, N2_FLOOR, RM, RN, RP, SF_D0, SF_D1, SF_D2, SF_D3, SF_D4,
                                SF_D5, SF_N0, SF_N1, SF_N2, SF_NB0, SF_NB1, SF_NB2, SIG_K_INV, SIG_PSI_INV)


@njit(cache=True)
def _solve(var, sprod, dcoef, sig_inv, akv_old, hc, dt, flux_bot, flux_top, floor):
    """
    Implicit diffusion + Patankar sources at the interior interfaces 1 … c−1 of one merged column (the arithmetic
    of ``GLSKEpsilon._solve``: backward Euler, no coupling through the bed and surface layers, Thomas algorithm).
    var : (c+1,) old values; sprod, dcoef : (c−1,) explicit source and implicit sink rate; hc : (c,) layers [m].
    Returns the (c−1,) new interior values, floored.
    """
    c = hc.shape[0]
    m = c - 1
    fc = np.empty(c)
    for l in range(c):
        fc[l] = -0.5 * dt * sig_inv * (akv_old[l] + akv_old[l + 1]) / hc[l]
    fc[0] = 0.0
    fc[c - 1] = 0.0
    lo = np.empty(m)
    up = np.empty(m)
    diag = np.empty(m)
    rhs = np.empty(m)
    for i in range(m):
        h = 0.5 * (hc[i] + hc[i + 1])
        diag[i] = h * (1.0 + dt * dcoef[i]) - fc[i] - fc[i + 1]
        rhs[i] = h * (var[i + 1] + dt * sprod[i])
        lo[i] = fc[i]
        up[i] = fc[i + 1]
    rhs[0] += dt * flux_bot
    rhs[m - 1] += dt * flux_top
    lo[0] = 0.0
    up[m - 1] = 0.0
    cp = np.empty(m)
    dp = np.empty(m)
    cp[0] = up[0] / diag[0]
    dp[0] = rhs[0] / diag[0]
    for k in range(1, m):
        mm = diag[k] - lo[k] * cp[k - 1]
        cp[k] = up[k] / mm
        dp[k] = (rhs[k] - lo[k] * dp[k - 1]) / mm
    x = np.empty(m)
    x[m - 1] = dp[m - 1]
    for k in range(m - 2, -1, -1):
        x[k] = dp[k] - cp[k] * x[k + 1]
    for k in range(m):
        x[k] = max(x[k], floor)
    return x


@njit(cache=True)
def _column_step(hc, uc, vc, n2, prod_in, tk, ep, av, at, us2b, us2s, dt, stratified, energy,
                 cm0, tke_min, psi_min, eps_min, nu_min, kt_min, galp, z0s_min, chk, z0_bed, alpha_n_min,
                 tk_new, ep_new, av_new, at_new, last_prod):
    """
    One GLS k-ε step of a column of c layers (``GLSKEpsilon.step``, homogeneous, or ``_step_stratified``):
    hc, uc, vc : (c,) layer thickness [m] and velocities [m s⁻¹]; n2 : (c−1,) N² [s⁻²] (zeros if homogeneous);
    prod_in : (c−1,) Burchard P [m² s⁻³] (``energy``) or unused; tk, ep, av, at : (c+1,) old k, ψ = ε, ν, κ_t;
    us2b, us2s : bed and surface u*² [m² s⁻²].  Writes tk_new, ep_new, av_new, at_new (c+1,) and last_prod (c−1,).
    """
    c = hc.shape[0]
    m = c - 1
    shear2 = np.empty(m)
    kold = np.empty(m)
    psio = np.empty(m)
    diss = np.empty(m)
    P = np.empty(m)
    sink = np.zeros(m)
    for i in range(m):
        h = 0.5 * (hc[i] + hc[i + 1])
        shear2[i] = ((uc[i + 1] - uc[i]) ** 2 + (vc[i + 1] - vc[i]) ** 2) / h**2
        kold[i] = tk[i + 1]
        psio[i] = ep[i + 1]
        diss[i] = max(cm0**E1 * kold[i] ** E2 * psio[i] ** E3, eps_min)
        if energy:                                                   # Burchard (2002) eq. 16, Patankar split
            last_prod[i] = prod_in[i]
            P[i] = max(prod_in[i], 0.0)
            sink[i] = -min(prod_in[i], 0.0) / kold[i]
        else:                                                        # CROCO P = ν_old S² (gls_mixing.F:621)
            P[i] = av[i + 1] * shear2[i]
            last_prod[i] = P[i]
    # --- k: B = −κ_t N², Patankar when P + B ≤ 0 (gls_mixing.F:639-665)
    sprod = np.empty(m)
    dcoef = np.empty(m)
    for i in range(m):
        if stratified:
            buoy = -at[i + 1] * (max(n2[i], 0.0) + min(n2[i], 0.0))
            if buoy + P[i] > 0.0:
                sprod[i] = P[i] + buoy
                dcoef[i] = diss[i] / kold[i]
            else:
                sprod[i] = P[i]
                dcoef[i] = diss[i] / kold[i] - buoy / kold[i]
        else:
            sprod[i] = P[i]
            dcoef[i] = diss[i] / kold[i]
        if energy:
            dcoef[i] = dcoef[i] + sink[i]
    for i in range(c + 1):
        tk_new[i] = tk[i]
        ep_new[i] = ep[i]
    tk_new[0] = max(tke_min, us2b / cm0**2)
    tk_new[c] = max(tke_min, us2s / cm0**2)
    x = _solve(tk, sprod, dcoef, SIG_K_INV, av, hc, dt, 0.0, 0.0, tke_min)
    for i in range(m):
        tk_new[i + 1] = x[i]
    # --- ψ: log-layer flux at the bed and surface (Burchard et al. 1998)
    z0s = max(z0s_min, chk * us2s)
    ks = 0.5 * (tk_new[c - 1] + tk_new[c])
    ls = KAPPA * (0.5 * hc[c - 1] + z0s)
    flux_top = -RN * cm0 ** (RP + 1) * KAPPA * SIG_PSI_INV * ks ** (RM + 0.5) * ls**RN
    kb = 0.5 * (tk_new[1] + tk_new[0])
    lb = KAPPA * (0.5 * hc[0] + z0_bed)
    flux_bot = -RN * cm0 ** (RP + 1) * KAPPA * SIG_PSI_INV * kb ** (RM + 0.5) * lb**RN
    ep_new[c] = max(psi_min, cm0**RP * ls**RN * ks**RM)
    ep_new[0] = max(psi_min, cm0**RP * lb**RN * tk_new[0] ** RM)
    for i in range(m):
        sp = BETA1 * psio[i] / kold[i] * P[i]
        if stratified:
            bp = -at[i + 1] * (psio[i] / kold[i]) * (BETA3M * max(n2[i], 0.0) + BETA3P * min(n2[i], 0.0))
            if bp + sp > 0.0:
                sprod[i] = sp + bp
                dcoef[i] = BETA2 * diss[i] / kold[i]
            else:
                sprod[i] = sp
                dcoef[i] = BETA2 * diss[i] / kold[i] - bp / psio[i]
        else:
            sprod[i] = sp
            dcoef[i] = BETA2 * diss[i] / kold[i]
        if energy:
            dcoef[i] = dcoef[i] + BETA1 * sink[i]
    x = _solve(ep, sprod, dcoef, SIG_PSI_INV, av, hc, dt, flux_bot, flux_top, psi_min)
    # --- Galperin limit, α_N, α_M, stability functions, ν and κ_t (gls_mixing.F:825-905)
    for i in range(m):
        k = tk_new[i + 1]
        l_lim = galp * np.sqrt(2.0 * k) / np.sqrt(max(N2_FLOOR, n2[i]))
        psi = max(x[i], cm0**RP * l_lim**RN * k**RM)
        ep_new[i + 1] = psi
        eps = max(cm0**E1 * k**E2 * psi**E3, eps_min)
        cff = (k / eps) ** 2
        am = cff * shear2[i]
        an = min(max(0.73 * alpha_n_min, cff * n2[i]), 1e10)
        am_max = ((LIM_AM0 + LIM_AM1 * an + LIM_AM2 * an**2 + LIM_AM3 * an**3)
                  / (LIM_AM4 + LIM_AM5 * an + LIM_AM6 * an**2))
        am = min(am, am_max)
        denom = SF_D0 + SF_D1 * an + SF_D2 * am + SF_D3 * an * am + SF_D4 * an**2 + SF_D5 * am**2
        av_new[i + 1] = max((SF_N0 + SF_N1 * an + SF_N2 * am) / denom * k**2 / eps, nu_min)
        if stratified:
            at_new[i + 1] = max((SF_NB0 + SF_NB1 * an + SF_NB2 * am) / denom * k**2 / eps, kt_min)
    if c > 2:
        av_new[c] = max(1.5 * av_new[c - 1] - 0.5 * av_new[c - 2], nu_min)
        av_new[0] = max(1.5 * av_new[1] - 0.5 * av_new[2], nu_min)
    else:
        av_new[c] = av_new[c - 1]
        av_new[0] = av_new[1]
    if stratified:
        if c > 2:
            at_new[c] = max(1.5 * at_new[c - 1] - 0.5 * at_new[c - 2], kt_min)
            at_new[0] = max(1.5 * at_new[1] - 0.5 * at_new[2], kt_min)
        else:
            at_new[c] = at_new[c - 1]
            at_new[0] = at_new[1]


@njit(cache=True)
def _scatter(xc, kc, kf, c, out, j):
    """Merged interface values xc (c+1,) back to the model interfaces (N+1,) of column j (MOM6 kc, kf): the value
    of the merged interface an interface sits on, else the linear mix clipped to its two neighbours."""
    for i in range(kc.shape[0]):
        lo = xc[kc[i]]
        hi = xc[min(kc[i] + 1, c)]
        f = kf[i]
        if f == 0.0:
            out[i, j] = lo
        elif f == 1.0:
            out[i, j] = hi
        else:
            out[i, j] = min(max((1.0 - f) * lo + f * hi, min(lo, hi)), max(lo, hi))


@njit(cache=True, parallel=True)
def merged_columns(dz, uc, vc, rho, us2s, us2b, work, tke, eps, akv, akt, h_massless, dt, stratified, energy,
                   rho0, g, cm0, tke_min, psi_min, eps_min, nu_min, kt_min, galp, z0s_min, chk, z0_bed, alpha_n_min,
                   o_tke, o_eps, o_akv, o_akt, o_n2, o_prod):
    """
    GLS step of the columns of ``dz`` (N, M) [m] with thin layers (module docstring), one column per thread task.

    uc, vc, rho : (N, M) layer velocities [m s⁻¹] and density [kg m⁻³] (rho unused if not ``stratified``);
    us2s, us2b : (M,) surface and bed u*² [m² s⁻²]; work : (N−1, M) interface kinetic-energy loss [m³ s⁻³]
    (``energy``) or unused; tke, eps, akv, akt : (N+1, M) old state (akt unused if not ``stratified``).
    Outputs (N+1, M): o_tke, o_eps, o_akv, o_akt, o_n2 (N² of the merged column on the model interfaces, N2_FLOOR
    where c ≤ 1); o_prod (N−1, M) the P used (0 where c ≤ 1).  A column with no layer ≥ h_massless keeps its
    state; one with a single such layer goes to the floors (no interior interface).
    """
    n, mcol = dz.shape
    for j in prange(mcol):
        c = 0
        for k in range(n):
            if dz[k, j] >= h_massless:
                c += 1
        for i in range(n + 1):
            o_n2[i, j] = N2_FLOOR
        for i in range(n - 1):
            o_prod[i, j] = 0.0
        if c <= 1:
            for i in range(n + 1):
                if c == 0:                                           # dry or massless column: unchanged
                    o_tke[i, j], o_eps[i, j], o_akv[i, j] = tke[i, j], eps[i, j], akv[i, j]
                    if stratified:
                        o_akt[i, j] = akt[i, j]
                else:                                                # one layer with mass: floors
                    o_tke[i, j], o_eps[i, j], o_akv[i, j] = tke_min, psi_min, nu_min
                    if stratified:
                        o_akt[i, j] = kt_min
            continue
        # owner of each layer: nearest layer with mass above it (k' ≥ k), else the nearest below; merged index
        lay = np.empty(n, np.int64)
        nxt = -1
        for k in range(n - 1, -1, -1):
            if dz[k, j] >= h_massless:
                nxt = k
            lay[k] = nxt
        prv = -1
        rank = np.empty(n, np.int64)
        r = -1
        for k in range(n):
            if dz[k, j] >= h_massless:
                prv = k
                r += 1
            rank[k] = r
            if lay[k] < 0:
                lay[k] = prv
        for k in range(n):
            lay[k] = rank[lay[k]]
        hc = np.zeros(c)
        hu = np.zeros(c)
        hv = np.zeros(c)
        hr = np.zeros(c)
        for k in range(n):
            h = dz[k, j]
            hc[lay[k]] += h
            hu[lay[k]] += h * uc[k, j]
            hv[lay[k]] += h * vc[k, j]
            if stratified:
                hr[lay[k]] += h * rho[k, j]
        for q in range(c):
            hu[q] = hu[q] / hc[q]
            hv[q] = hv[q] / hc[q]
        # fractional height of each model interface inside its merged layer (MOM6 kc, kf)
        zc = np.empty(c + 1)
        zc[0] = 0.0
        for q in range(c):
            zc[q + 1] = zc[q] + hc[q]
        kc = np.empty(n + 1, np.int64)
        kf = np.empty(n + 1)
        zf = 0.0
        for i in range(n + 1):
            kc[i] = lay[i] if i < n else lay[n - 1]
            f = (zf - zc[kc[i]]) / hc[kc[i]]
            kf[i] = min(max(f, 0.0), 1.0)
            if i < n:
                zf = zf + dz[i, j]
        kf[0] = 0.0
        kf[n] = 1.0
        # merged interface q = bottom interface of merged layer q (its first model layer); q = c the surface
        first = np.empty(c + 1, np.int64)
        first[0] = 0
        for k in range(1, n):
            if lay[k] != lay[k - 1]:
                first[lay[k]] = k
        first[c] = n
        tk = np.empty(c + 1)
        ep = np.empty(c + 1)
        av = np.empty(c + 1)
        at = np.empty(c + 1)
        for q in range(c + 1):
            tk[q], ep[q], av[q] = tke[first[q], j], eps[first[q], j], akv[first[q], j]
            at[q] = akt[first[q], j] if stratified else 0.0
        n2c = np.zeros(c - 1)
        if stratified:
            for q in range(c - 1):
                n2c[q] = (g / rho0) * (hr[q] / hc[q] - hr[q + 1] / hc[q + 1]) / (0.5 * (hc[q] + hc[q + 1]))
        prod = np.zeros(c - 1)
        if energy:                                                   # energy split as the interpolation weights
            ec = np.zeros(c + 1)
            for i in range(1, n):
                ec[kc[i]] += (1.0 - kf[i]) * work[i - 1, j]
            for i in range(1, n):
                ec[min(kc[i] + 1, c)] += kf[i] * work[i - 1, j]
            for q in range(c - 1):
                prod[q] = ec[q + 1] / (0.5 * (hc[q] + hc[q + 1]))
        n2s = np.empty(c - 1)                                        # nan_to_num of the stratified branch
        for q in range(c - 1):
            x = n2c[q]
            n2s[q] = 0.0 if np.isnan(x) else x
        tk_n = np.empty(c + 1)
        ep_n = np.empty(c + 1)
        av_n = np.empty(c + 1)
        at_n = np.empty(c + 1)
        lp = np.empty(c - 1)
        _column_step(hc, hu, hv, n2s, prod, tk, ep, av, at, us2b[j], us2s[j], dt, stratified, energy,
                     cm0, tke_min, psi_min, eps_min, nu_min, kt_min, galp, z0s_min, chk, z0_bed, alpha_n_min,
                     tk_n, ep_n, av_n, at_n, lp)
        _scatter(tk_n, kc, kf, c, o_tke, j)
        _scatter(ep_n, kc, kf, c, o_eps, j)
        _scatter(av_n, kc, kf, c, o_akv, j)
        if stratified:
            _scatter(at_n, kc, kf, c, o_akt, j)
            n2p = np.empty(c + 1)
            n2p[0] = n2c[0]
            n2p[c] = n2c[c - 2]
            for q in range(c - 1):
                n2p[q + 1] = n2c[q]
            _scatter(n2p, kc, kf, c, o_n2, j)
        lpp = np.empty(c + 1)
        lpp[0] = lp[0]
        lpp[c] = lp[c - 2]
        for q in range(c - 1):
            lpp[q + 1] = lp[q]
        tmp = np.empty((n + 1, 1))
        _scatter(lpp, kc, kf, c, tmp, 0)
        for i in range(n - 1):
            o_prod[i, j] = tmp[i + 1, 0]
