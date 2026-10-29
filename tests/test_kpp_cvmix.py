"""
Tests for physics/mixing/kpp_cvmix.py (N3 task 6): CVMix-style KPP column.

Reference values of ``test_kpp_matches_cvmix_reg_test`` come from the CVMix
regression test ``~/pyOCEAN_refs/MOM6/pkg/CVMix-src/reg_tests/kpp`` (input.nl,
driver src/drivers/cvmix_kpp_drv.F90), built with gfortran from the local
CVMix-src (fce4221) and run on 2026-10-29; the same numbers are in the stored
baselines ``CVMix-src/baselines/kpp/test{1,4a-d,5}.nc``.  Test 3 values are
from the run's ``test3.out``, test 6 from its stdout.  Test 2 (MatchGradient
coefficients) and test 7 (Stokes-MOST) exercise options this module does not
implement.
"""

import numpy as np
import pytest

from physics.mixing.kpp_cvmix import (
    KPPParams,
    bulk_richardson,
    compute_kobl_depth,
    compute_obl_depth,
    kpp_coefficients,
    kpp_column,
    turbulent_scales,
    unresolved_shear,
)


def _col(x):
    return np.asarray(x, dtype=float)[:, None]


def _uniform_column(dz: float, depth: float):
    """Bed-up thickness and interface heights of a uniform column, plus surface-down centre depths."""
    n = int(round(depth / dz))
    dz = depth / n
    h = np.full((n, 1), dz)
    z_iface = (np.arange(n + 1) * dz - depth)[:, None]
    d_c = (np.arange(n) + 0.5) * dz        # surface-down centre depths
    return n, dz, h, z_iface, d_c


# ---------------------------------------------------------------------------
# CVMix regression test
# ---------------------------------------------------------------------------

def test_kpp_matches_cvmix_reg_test():
    """Reproduce CVMix reg_tests/kpp tests 1, 3, 4a-d, 5 and 6 to 1e-10."""
    # Test 1: OBL depth from a prescribed Ri_b, quadratic interpolation (input.nl kpp_col1_nml)
    zw = _col(-10.0 * np.arange(5))
    zt = 0.5 * (zw[:-1] + zw[1:])
    ri = _col([0.0, 0.1875, 1.5, 3.0])
    h, kobl = compute_obl_depth(ri, zw, zt, KPPParams(ri_crit=0.3, interp_type="quadratic"))
    assert h[0] == pytest.approx(17.436902811724739, rel=1e-10, abs=0)
    assert kobl[0] == 2.75

    # Test 3: 1/phi_m, 1/phi_s with vonkarman = surf_layer_ext = 1, sigma = h = u* = 1 (so zeta = B_f)
    p3 = KPPParams(vonkarman=1.0, surf_layer_ext=1.0)
    assert p3.a_m == pytest.approx(1.2573615701844176, rel=1e-14)
    assert p3.c_m == pytest.approx(8.3824104678961167, rel=1e-14)
    assert p3.a_s == pytest.approx(-28.861739379323623, rel=1e-14)
    assert p3.c_s == pytest.approx(98.954535014823847, rel=1e-14)
    table3 = [  # zeta, phi_m, phi_s  (test3.out rows 1, 51, 101, 181, 201, 221)
        (-0.20000000000000000e01, 0.38141479820561569e00, 0.18085501703062448e00),
        (-0.14999999999999996e01, 0.41659659001625915e00, 0.20298277264926820e00),
        (-0.99999999999999911e00, 0.46987003038591246e00, 0.24253562503633308e00),
        (-0.19999999999999840e00, 0.69853420565801072e00, 0.48795003647426805e00),
        (0.16410484082740595e-14, 0.10000000000000082e01, 0.10000000000000082e01),
        (0.20000000000000168e00, 0.20000000000000084e01, 0.20000000000000084e01),
    ]
    zeta = np.array([r[0] for r in table3])
    w_m, w_s = turbulent_scales(1.0, 1.0, zeta, 1.0, p3)
    np.testing.assert_allclose(1.0 / w_m, [r[1] for r in table3], rtol=1e-10)
    np.testing.assert_allclose(1.0 / w_s, [r[2] for r in table3], rtol=1e-10)

    # Test 4: K profiles, u* = 1, B_f = 100 (stable), incoming K = (1, 10, 5, 0, 0, 0),
    # h above (4a, 4c) and below (4b, 4d) the 3rd centre, with / without enhanced diffusion
    zw = _col(-5.0 * np.arange(6))
    zt = 0.5 * (zw[:-1] + zw[1:])
    old = _col([1.0, 10.0, 5.0, 0.0, 0.0, 0.0])
    expected4 = {
        (True, 11.25): [1.0, 0.00273134491423577, 1.2501613200589972, 0.0, 0.0, 0.0],
        (True, 13.75): [1.0, 2.9344831716373223e-03, 1.0779734099892206e-03,
                        2.1054168163852057e-05, 0.0, 0.0],
        (False, 11.25): [1.0, 2.7313449142357697e-03, 2.1850759313886132e-04, 0.0, 0.0, 0.0],
        (False, 13.75): [1.0, 0.00293448317163732, 0.00107797340998922, 0.0, 0.0, 0.0],
    }
    for (enh, obl), ref in expected4.items():
        obl_a = np.array([obl])
        kobl = compute_kobl_depth(zw, zt, obl_a)
        assert kobl[0] == (3.25 if obl == 11.25 else 3.75)
        km, ks, nl = kpp_coefficients(zw, zt, obl_a, kobl, np.array([1.0]), np.array([100.0]),
                                      KPPParams(enhance_diffusion=enh),
                                      kappa_m_in=old, kappa_s_in=old)
        np.testing.assert_allclose(km[:, 0], ref, rtol=1e-10, atol=1e-16)
        np.testing.assert_allclose(ks[:, 0], ref, rtol=1e-10, atol=1e-16)
        assert np.all(nl == 0.0)  # stable forcing: no nonlocal term

    # Test 5: bulk Ri of L94 Fig. C1 (h_m = 17 m, N = 0.01, linear interpolation, Cv = 1.5)
    p5 = KPPParams(cv=1.5, interp_type="linear")
    nlev, dz, hmix = 10, 5.0, 17.0
    zw = -dz * np.arange(nlev + 1)
    zt = 0.5 * (zw[:-1] + zw[1:])
    nsqr = 0.01 ** 2
    vslope = -0.1 / (nlev * dz - hmix)
    buoy = np.zeros(nlev)
    uvel = np.zeros(nlev)
    nif = np.zeros(nlev + 1)
    for kt in range(nlev):
        if zt[kt] >= -hmix or kt == 0:
            buoy[kt], uvel[kt] = nsqr, 0.1
        elif zw[kt] >= -hmix:
            buoy[kt] = -nsqr * (-zw[kt + 1] - hmix) ** 2 / (2 * dz) + nsqr
            uvel[kt] = vslope * (-zw[kt + 1] - hmix) ** 2 / (2 * dz) + 0.1
        else:
            buoy[kt] = nsqr - nsqr * (-zt[kt] - hmix)
            uvel[kt] = 0.1 + vslope * (-zt[kt] - hmix)
        if not (zt[kt] >= -hmix or kt == 0):
            nif[kt] = np.sqrt(-(buoy[kt] - buoy[kt - 1]) / dz)
    nif[nlev] = 0.01
    _, ws = turbulent_scales(0.0, -zt, buoy[0], 0.01, p5)
    vt2 = unresolved_shear(_col(zt), _col(ws), _col(np.sqrt(np.maximum(nif ** 2, 0.0))), p5)
    rib = bulk_richardson(_col(zt), _col(buoy[0] - buoy), _col((0.1 - uvel) ** 2), vt2, p5)
    ri_ref = [0.0, 0.0, 0.0, 0.50036100310406928, 2.7492222181497152, 4.6498619223254929,
              5.9957729369547081, 6.9400904214280787, 7.6130420877582772, 8.1038814329341502]
    np.testing.assert_allclose(rib[:, 0], ri_ref, rtol=1e-10, atol=1e-14)
    h5, _ = compute_obl_depth(rib, _col(zw), _col(zt), p5)
    assert h5[0] == pytest.approx(15.497835544126163, rel=1e-10, abs=0)

    # Test 6: velocity scales, neutral u* (6a) and free convection u* = 0 (6b)
    p6 = KPPParams(vonkarman=0.4)
    ustar = np.sqrt(0.2 / 1035.0)
    w_m, w_s = turbulent_scales(0.1, 6000.0, 0.0, ustar, p6)
    assert float(w_m) == pytest.approx(5.5603843748553280e-3, rel=1e-12)
    assert float(w_s) == pytest.approx(5.5603843748553280e-3, rel=1e-12)
    bf = (9.8 * 2.5e-4 / (1035.0 * 3992.0)) * -100.0
    assert bf == pytest.approx(-5.9297338638629921e-08, rel=1e-12)
    w_m, w_s = turbulent_scales(0.1, 6000.0, bf, 0.0, p6)
    assert float(w_m) == pytest.approx(1.9690879464369034e-2, rel=1e-12)
    assert float(w_s) == pytest.approx(4.4835364693284696e-2, rel=1e-12)


def test_kppparams_defaults_are_cvmix_mom6():
    """Defaults = CVMix as MOM6 calls it (cited in KPPParams); background mixing off."""
    p = KPPParams()
    assert (p.ri_crit, p.vonkarman, p.surf_layer_ext, p.cstar) == (0.3, 0.4, 0.1, 10.0)
    assert (p.zeta_m, p.zeta_s, p.min_vt2) == (-0.2, -1.0, 1e-10)
    assert p.cv is None and p.interp_type == "quadratic" and p.enhance_diffusion
    assert p.sw_method == "MXL_SW" and not p.compute_ekman and not p.compute_monin_obukhov
    assert p.kappa_bg == 0.0 and p.nu_bg == 0.0
    # MOM6 CS2 default 6.32739901508 (MOM_CVMix_KPP.F90:343-345); CS = 98.96 is c_s rounded
    assert p.nonlocal_coeff == pytest.approx(6.32739901508, rel=1e-10)
    assert p.c_s == pytest.approx(98.96, abs=0.006)
    assert p.vtc == pytest.approx(np.sqrt(0.2 / (p.c_s * 0.1)) / 0.16, rel=1e-14)
    with pytest.raises(ValueError):
        KPPParams(interp_type="spline")
    with pytest.raises(ValueError):
        KPPParams(sw_method="NONE")


# ---------------------------------------------------------------------------
# Column driver
# ---------------------------------------------------------------------------

def test_kpp_free_convection_ustar_zero():
    """
    Review Focus 5: u* = 0 with cooling gives finite h, K and nonlocal term
    (no division by u*³).  K_s inside h equals h w_s G(σ) with the
    free-convection scale w_s = κ (−c_s min(ε, σ) h κ B_f)^{1/3}
    (L94 eqs. 13, B1e; cvmix_kpp.F90:2590-2600), and a tiny u* tends to it.
    """
    p = KPPParams()
    n, dz, h, z_iface, d_c = _uniform_column(2.0, 100.0)
    n2 = 1e-5
    b_sd = -n2 * np.maximum(d_c - 30.0, 0.0)            # mixed 30 m over a linear thermocline
    b = b_sd[::-1, None]
    zeros = np.zeros((n, 1))
    bf = -1e-7
    res = kpp_column(h, z_iface, zeros, zeros, b, np.array([bf]), np.array([0.0]), None, None, p)
    hobl = res.obl_depth[0]
    for arr in (res.kappa_m, res.kappa_s, res.nonlocal_, res.bulk_ri):
        assert np.all(np.isfinite(arr))
    assert 30.0 < hobl < 100.0
    assert np.all(res.kappa_s >= 0.0) and np.all(res.kappa_m >= 0.0)
    assert res.nonlocal_[n, 0] == 0.0 and res.kappa_s[n, 0] == 0.0     # surface interface
    d_if = np.arange(n + 1) * dz                                       # surface-down interface depths
    inside = (d_if > 0) & (d_if < hobl - dz)                           # away from the enhanced interface
    nl_sd = res.nonlocal_[::-1, 0]
    ks_sd = res.kappa_s[::-1, 0]
    sig = d_if[inside] / hobl
    g = sig * (1 - sig) ** 2
    np.testing.assert_allclose(nl_sd[inside], p.nonlocal_coeff * g, rtol=1e-12)
    ws = p.vonkarman * (-p.c_s * np.minimum(p.surf_layer_ext, sig) * hobl * p.vonkarman * bf) ** (1 / 3)
    np.testing.assert_allclose(ks_sd[inside], hobl * ws * g, rtol=1e-12)
    assert np.all(nl_sd[d_if > hobl] == 0.0) and np.all(ks_sd[d_if > hobl + dz] == 0.0)

    res_eps = kpp_column(h, z_iface, zeros, zeros, b, np.array([bf]), np.array([1e-6]), None, None, p)
    assert res_eps.obl_depth[0] == pytest.approx(hobl, rel=1e-3)
    # u* = 0 and no buoyancy flux: no turbulence at all, still finite
    res0 = kpp_column(h, z_iface, zeros, zeros, b, np.array([0.0]), np.array([0.0]), None, None, p)
    assert np.all(res0.kappa_s == 0.0) and np.isfinite(res0.obl_depth[0])


def _hand_obl_l94(d_c, dz, b_sd, u_sd, ustar, p):
    """
    Large et al. (1994) eqs. (21)-(23) by hand, one column, uniform layers,
    B_f = 0 (so w_s = κ u*), CVMix constants, linear interpolation.
    """
    n = len(d_c)
    eps, kap, ric = p.surf_layer_ext, p.vonkarman, p.ri_crit
    c_s = 24.0 * np.sqrt(17.0)
    vtc = np.sqrt(0.2 / (c_s * eps)) / kap ** 2
    ri = np.zeros(n)
    for k in range(n):
        sl = eps * max(d_c[k], dz)                       # surface layer, at least eps * top layer
        acc_b = acc_u = acc_h = 0.0
        for j in range(n):
            part = min(max(sl - j * dz, 0.0), dz)
            acc_b += b_sd[j] * part
            acc_u += u_sd[j] * part
            acc_h += part
        br, ur = acc_b / acc_h, acc_u / acc_h
        n_below = np.sqrt(max((b_sd[k] - b_sd[k + 1]) / dz, 0.0)) if k < n - 1 else 0.0
        cv = 2.1 - 200.0 * n_below if n_below < 0.002 else 1.7
        vt2 = max(cv * vtc * d_c[k] * n_below * kap * ustar / ric, 1e-10)
        ri[k] = (1 - eps / 2) * d_c[k] * (br - b_sd[k]) / ((ur - u_sd[k]) ** 2 + vt2)
    k = next(i for i in range(n) if ri[i] > ric)
    d1, d2, r1, r2 = d_c[k - 1], d_c[k], ri[k - 1], ri[k]
    return d1 + (ric - r1) * (d2 - d1) / (r2 - r1)


def test_kpp_neutral_wind_obl_depth():
    """
    Linear stratification, wind (u*), no heat flux, no resolved shear: h from
    kpp_column equals the hand calculation of L94 eqs. (21)-(23) to 1e-10, and
    is within half a layer of the continuous solution of the same equations,
    Ri_b(d) = (1 − ε/2)² N d Ri_c / (C_v V_tc κ u*)  →  h = C_v V_tc κ u* / ((1 − ε/2)² N).
    """
    p = KPPParams(interp_type="linear")
    n, dz, h, z_iface, d_c = _uniform_column(1.0, 60.0)
    nfreq = 1e-3
    b_sd = -nfreq ** 2 * d_c
    u_sd = np.zeros(n)
    ustar = 0.02
    zeros = np.zeros((n, 1))
    res = kpp_column(h, z_iface, zeros, zeros, b_sd[::-1, None], np.array([0.0]),
                     np.array([ustar]), None, None, p)
    hand = _hand_obl_l94(d_c, dz, b_sd, u_sd, ustar, p)
    assert res.obl_depth[0] == pytest.approx(hand, rel=1e-10)
    cv = 2.1 - 200.0 * nfreq
    h_cont = cv * p.vtc * p.vonkarman * ustar / ((1 - p.surf_layer_ext / 2) ** 2 * nfreq)
    assert abs(res.obl_depth[0] - h_cont) < 0.5 * dz
    # Neutral forcing: the scales are κ u*, so K_m = K_s inside h
    np.testing.assert_allclose(res.kappa_m, res.kappa_s, rtol=1e-14)


def _fig_c1_column(dz, hm, interp):
    """
    L94 Fig. C1: B and U uniform down to h_m, then N = 0.01 s^-1 and U falling
    linearly to 0 at 50 m; exact layer averages; u* = 0.01, B_f = 0.
    """
    depth = 50.0
    n, dz, h, z_iface, d_c = _uniform_column(dz, depth)
    top = dz * np.arange(n)
    bot = top + dz
    ramp = 0.5 * (np.maximum(bot - hm, 0) ** 2 - np.maximum(top - hm, 0) ** 2) / dz
    b_sd = -1e-4 * ramp
    u_sd = 0.1 - 0.1 / (depth - hm) * ramp
    res = kpp_column(h, z_iface, u_sd[::-1, None], np.zeros((n, 1)), b_sd[::-1, None],
                     np.array([0.0]), np.array([0.01]), None, None, KPPParams(interp_type=interp))
    return res.obl_depth[0]


@pytest.mark.parametrize("interp", ["linear", "quadratic", "cubic"])
def test_kpp_resolution_large_appendix_c(interp):
    """
    Large et al. (1994) Appendix C, Fig. C1-C2: for mixed layers h_m = 8 ... 22 m
    the boundary-layer depth of discrete profiles h_d differs from the continuous
    one h_c by "much less than the layer thickness".  h_c is taken at
    Δz = 0.2 m (converged: Δz = 0.05 m gives the same to 1e-3 m) and the
    check uses the resolutions of the appendix's sensitivity runs, Δz = 2.0 and
    3.33 m, plus the 5 m of Fig. C1: |h_d − h_c| < Δz/2.  With linear
    interpolation L94 report h_d systematically below h_c.
    """
    for hm in (8.0, 11.0, 14.0, 17.0, 20.0, 22.0):
        h_c = _fig_c1_column(0.2, hm, interp)
        assert h_c == pytest.approx(_fig_c1_column(0.05, hm, interp), abs=1e-3)
        for dz in (2.0, 50.0 / 15.0, 5.0):
            h_d = _fig_c1_column(dz, hm, interp)
            assert abs(h_d - h_c) < 0.5 * dz, (hm, dz, h_d, h_c)
            if interp == "linear":
                assert h_d <= h_c + 1e-9


def test_kpp_column_orientation_and_stability():
    """
    k = 0 is the bed: a 20 m mixed layer at the TOP of a 200 m column with
    non-uniform layers gives h near 20 m; K and γ vanish at the surface
    interface (k = N) and at the bed; heating (B_f > 0) gives no nonlocal term;
    a column given upside down is rejected.
    """
    p = KPPParams()
    dz_sd = np.concatenate([np.full(10, 2.0), np.full(18, 10.0)])   # surface-down
    n = dz_sd.size
    d_bot = np.cumsum(dz_sd)
    d_c = d_bot - 0.5 * dz_sd
    b_sd = -1e-4 * np.maximum(d_c - 20.0, 0.0)
    h = dz_sd[::-1, None]
    z_iface = np.concatenate([[0.0], np.cumsum(h[:, 0])])[:, None] - d_bot[-1]
    zeros = np.zeros((n, 1))
    res = kpp_column(h, z_iface, zeros, zeros, b_sd[::-1, None], np.array([-1e-8]),
                     np.array([0.01]), None, None, p)
    assert 20.0 <= res.obl_depth[0] < 30.0
    assert res.kappa_s[n, 0] == 0.0 and res.kappa_s[0, 0] == 0.0
    assert res.nonlocal_[n, 0] == 0.0 and res.nonlocal_[0, 0] == 0.0
    assert res.kappa_s[n - 3, 0] > 0.0 and res.nonlocal_[n - 3, 0] > 0.0
    # model layer that holds h, from k_obl
    k_layer = n - int(np.floor(res.k_obl[0]))
    z_top = z_iface[k_layer + 1, 0]
    z_bed = z_iface[k_layer, 0]
    assert z_bed <= -res.obl_depth[0] <= z_top

    heated = kpp_column(h, z_iface, zeros, zeros, b_sd[::-1, None], np.array([1e-8]),
                        np.array([0.01]), None, None, p)
    assert np.all(heated.nonlocal_ == 0.0)
    assert heated.obl_depth[0] <= res.obl_depth[0]

    with pytest.raises(ValueError):
        kpp_column(h, z_iface[::-1], zeros, zeros, b_sd[::-1, None], np.array([-1e-8]),
                   np.array([0.01]), None, None, p)


def test_kpp_shortwave_in_boundary_layer_flux():
    """
    B_f used for K and γ: MXL_SW keeps only the shortwave absorbed above the
    bottom of the layer holding h (MOM_CVMix_KPP.F90:835-845, L94 eq. A3 with
    h_B = h); ALL_SW keeps all of it.  Extra columns are independent.
    """
    n, dz, h, z_iface, d_c = _uniform_column(2.0, 100.0)
    b_sd = -1e-5 * np.maximum(d_c - 30.0, 0.0)
    d_if = np.arange(n + 1) * dz
    frac_sd = 0.58 * np.exp(-d_if / 0.35) + 0.42 * np.exp(-d_if / 23.0)   # Jerlov I (L94 l.2067)
    frac = np.repeat(frac_sd[::-1, None], 2, axis=1)
    hh = np.repeat(h, 2, axis=1)
    zi = np.repeat(z_iface, 2, axis=1)
    zeros = np.zeros((n, 2))
    b = np.repeat(b_sd[::-1, None], 2, axis=1)
    bns = np.array([-2e-7, -2e-7])
    bsw = np.array([1e-7, 0.0])
    us = np.array([0.01, 0.01])
    res = kpp_column(hh, zi, zeros, zeros, b, bns, us, frac, None, KPPParams(), bflux_sw=bsw)
    kint = int(np.floor(res.k_obl[0]))
    assert res.bflux_obl[0] == pytest.approx(bns[0] + bsw[0] * (1.0 - frac_sd[kint]), rel=1e-14)
    assert res.bflux_obl[1] == bns[1]
    alone = kpp_column(h, z_iface, zeros[:, :1], zeros[:, :1], b[:, :1], bns[:1], us[:1],
                       frac[:, :1], None, KPPParams(), bflux_sw=bsw[:1])
    np.testing.assert_array_equal(alone.kappa_s[:, 0], res.kappa_s[:, 0])
    # sunlight makes the boundary layer shallower than with the non-solar cooling alone
    assert res.obl_depth[0] < res.obl_depth[1]
    all_sw = kpp_column(hh, zi, zeros, zeros, b, bns, us, frac, None,
                        KPPParams(sw_method="ALL_SW"), bflux_sw=bsw)
    assert all_sw.bflux_obl[0] == pytest.approx(bns[0] + bsw[0], rel=1e-14)


def test_kpp_vanished_layers_stay_finite():
    """
    Hybrid / z* columns with layers of 1e-10 m between 5 m layers (Review Focus 1
    for KPP): h, K and γ are finite, K ≥ 0, and h is that of the column without
    the vanished layers to within one layer thickness.
    """
    dz_thick = np.full(20, 5.0)
    d_c = np.cumsum(dz_thick) - 2.5
    b_thick = -1e-5 * np.maximum(d_c - 25.0, 0.0)
    u_thick = 0.05 * (d_c < 25.0)
    dz_v, b_v, u_v = [], [], []
    for k in range(20):
        reps = 2 if k in (1, 4, 6, 12) else 1
        for r in range(reps):
            dz_v.append(dz_thick[k] if r == 0 else 1e-10)
            b_v.append(b_thick[k])
            u_v.append(u_thick[k])
    out = []
    for dz_sd, b_sd, u_sd in ((dz_thick, b_thick, u_thick), (np.array(dz_v), np.array(b_v), np.array(u_v))):
        n = dz_sd.size
        h = dz_sd[::-1, None]
        zi = np.concatenate([[0.0], np.cumsum(h[:, 0])])[:, None] - dz_sd.sum()
        res = kpp_column(h, zi, u_sd[::-1, None], np.zeros((n, 1)), b_sd[::-1, None],
                         np.array([-5e-8]), np.array([0.008]), None, None, KPPParams())
        for arr in (res.kappa_m, res.kappa_s, res.nonlocal_, res.bulk_ri, res.obl_depth):
            assert np.all(np.isfinite(arr))
        assert np.all(res.kappa_s >= 0.0) and np.all(res.kappa_m >= 0.0)
        out.append(res)
    assert abs(out[0].obl_depth[0] - out[1].obl_depth[0]) < 5.0
    assert out[1].kappa_s.max() <= 1.01 * out[0].kappa_s.max() + 1e-12


def test_kpp_vanished_layers_at_crossing_and_uniform_column_no_warning():
    # N3 task 7 (Task 6 carry): two consecutive zero-thickness layers bracketing the Ri_c crossing put two centres
    # at the same height (zero spacing in the interpolation polynomial); a uniform column has a flat Ri_b ≡ 0
    # polynomial.  Neither may divide by zero (RuntimeWarning raises under -W error); the crossing is at the
    # vanished layers' height.
    import warnings
    hs = [2.0] * 5 + [0.0, 0.0] + [2.0] * 10                          # surface-down
    Ts = [15.0] * 5 + [15.0, 10.0] + [10.0] * 10
    h = np.array(hs[::-1])[:, None]
    n = h.shape[0]
    z = np.concatenate([[0.0], np.cumsum(h[:, 0])])[:, None]
    b = 9.81 * 2e-4 * np.array(Ts[::-1])[:, None]
    zero = np.zeros((n, 1))
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for interp in ("linear", "quadratic", "cubic"):
            r = kpp_column(h, z, zero, zero, b, np.zeros(1), np.full(1, 0.01), None, None,
                           KPPParams(interp_type=interp))
            assert r.obl_depth[0] == pytest.approx(10.0) and np.isfinite(r.kappa_s).all()
        for bf, us in ((-1e-7, 0.01), (0.0, 0.01), (1e-7, 0.0), (-1e-7, 0.0)):
            r = kpp_column(np.ones((4, 1)), np.arange(5.0)[:, None], np.zeros((4, 1)), np.zeros((4, 1)),
                           np.zeros((4, 1)), np.full(1, bf), np.full(1, us), None, None)
            assert np.isfinite(r.obl_depth).all() and np.isfinite(r.kappa_s).all()
