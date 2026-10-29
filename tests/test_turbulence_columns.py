"""
CROCO single-column cases in pyOCEAN (N3 tasks 8 and 9; validation/turbulence_columns/, spec
docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md criterion 1).

* the seven case definitions (grid, forcing, initial N²) equal CROCO's (docs/design/n2c_refs/croco_column_cases.md,
  values copied here from that table, each with its CROCO file:line), and the built model carries them;
* 6 h of Kato–Phillips in σ with the GLS: the mixed-layer depth (k < 1e-5) is within ±10 % of Price's law
  D_m = 1.05 u* N₀^(−1/2) t^(1/2) (Price 1979; Warner et al. 2005, eq. 60).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from validation.turbulence_columns import cases as C
from validation.turbulence_columns.compare import mixed_depth, price_depth

G = 9.81

# case: (N, depth, f0, rho0, N0² [s⁻²], u*² x [m² s⁻²], Q_ns, Q_sw peak [W m⁻²], body force at t = 0 [m s⁻²],
#        Zob [m], hours) — croco_column_cases.md §1-4 (param.h:78-93, ana_grid.F:119-137, ana_initial.F:568-599,
#        analytical.F:378-389, 553-557, 876-879, prsgrd.F:63-76, 455-462, TEST_CASES/croco.in.* l. 4, 44, 63)
CROCO = {
    "kato_phillips": (100, 50.0, 0.0, 1024.0, 1e-4, 1e-4, 0.0, 0.0, 0.0, 0.0, 30.0),
    "willis_deardorff": (50, 50.0, 0.0, 1024.0, G * 0.2 / 1024.0 * 0.1, 0.0, -100.0, 0.0, 0.0, 0.0, 72.0),
    "diurnal_cycle": (150, 150.0, 0.0, 1024.0, G * 0.2 / 1024.0 * 0.01, 0.0, -75.0, 264.91, 0.0, 0.0, 192.0),
    "forced_ekbbl": (40, 1500.0, 1e-4, 1.22, 0.0, 0.0, 0.0, 0.0, 1e-4 * 10.0, 0.1, 24.0),
    "forced_dbleek": (25, 25.0, 1e-4, 1024.0, 0.0, -6.25e-6, 0.0, 0.0, 1e-4 * 0.0015, 0.05, 24.0),
    "forced_nonrotbbl": (50, 5.0, 0.0, 1024.0, 0.0, 0.0, 0.0, 0.0, -2e-6 * 9.81, 0.01, 30.0),
    "forced_oscnonrotbbl": (50, 5.0, 0.0, 1024.0, 0.0, 0.0, 0.0, 0.0, -2e-6 * 9.81, 0.01, 60.0),
}


# initial T at z = 0 [°C]: ana_initial.F:568-599 (stratified), :1222 (default branch, uniform 20 °C)
T_SURF = {"kato_phillips": 16.0, "willis_deardorff": 22.0, "diurnal_cycle": 22.0, "forced_ekbbl": 20.0,
          "forced_dbleek": 20.0, "forced_nonrotbbl": 20.0, "forced_oscnonrotbbl": 20.0}


@pytest.mark.parametrize("name", C.NAMES)
def test_cases_match_croco_setup(name):
    n, depth, f0, rho0, n2, us2, qns, qsw, body, zob, hours = CROCO[name]
    c = C.CASES[name]
    assert (c.n, c.depth, c.rho0, c.zob, c.hours) == (n, depth, rho0, zob, hours)
    assert c.n_steps == round(hours * 3600 / 30.0) and C.DT == 30.0
    np.testing.assert_allclose(c.n2_initial, n2, rtol=1e-12, atol=0)
    assert c.T_surf == T_SURF[name]
    assert (c.ustar2_x, c.q_ns, c.q_sw_peak) == (us2, qns, qsw)
    np.testing.assert_allclose(C.body_force(c, 0.0), body, rtol=1e-12)
    assert C.shortwave(c, 0.0) == 0.0 and C.shortwave(c, 43200.0) == pytest.approx(qsw)   # night / local noon
    if name == "forced_oscnonrotbbl":                                 # M2 period 12.4 h (prsgrd.F:75)
        np.testing.assert_allclose(C.body_force(c, 6.2 * 3600.0), -body, rtol=1e-12)
    # the built model carries the case: 5 × 5 periodic, uniform σ, f, EOS, z0, initial N² and wind stress
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, st, frc = C.build(name, "sigma", "gls")
    assert m.H.shape == (5, 5) and np.all(m.H == depth) and m.core.n_layers == n and m.periodic.x and m.periodic.y
    np.testing.assert_allclose(m.layer_volumes(st.eta) / m.area[None], depth / n, rtol=1e-12)
    assert m.rho0 == rho0 and m.dt == 30.0 and m.core.z0 == (zob or None)
    np.testing.assert_allclose(m.hg["dx_t"].values, 2000.0)
    np.testing.assert_allclose(m.hg["f_q"].values, f0, rtol=1e-12, atol=0)
    dTdz = np.diff(st.T[:, 0, 0]) / (depth / n)
    np.testing.assert_allclose(G * (0.2 / rho0) * dTdz, n2, rtol=1e-10, atol=1e-18)
    np.testing.assert_allclose(frc.tau_x / rho0, us2, rtol=1e-12)


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
def test_body_force_enters_the_explicit_momentum(coord):
    # the harness's named term "body_force" (∂v/∂t += f U_G in FORCED_EKBBL) is summed with the other terms, in σ and
    # in ALE z* (the driver of both reaches the momentum through model._momentum_terms)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, st, frc = C.build("forced_ekbbl", coord, "gls")
    terms, wu, wv = m._momentum_terms(st)
    assert terms[-1][0] == "body_force"
    np.testing.assert_allclose(terms[-1][2], 1e-4 * 10.0, rtol=1e-12)
    st = frc.step(m, st)                                              # first step from rest: v grows by ≈ Δt f U_G
    assert 0.5 * 30.0 * 1e-3 < st.v[-1, 2, 2] <= 30.0 * 1e-3 * (1 + 1e-9)


def test_body_force_hook_fails_loudly(monkeypatch):
    # if the private hook stops reaching the summed terms (e.g. a refactor binds terms at construction), build raises
    monkeypatch.setattr(C, "_install_body_force", lambda model, case: None)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        with pytest.raises(RuntimeError, match="body force"):
            C.build("forced_nonrotbbl", "zstar", "gls")


def test_bottom_heat_flux_skips_vanished_bed_layer():
    # a vanished (< 1 mm) bed layer does not take Δt btflx / h₀: the flux goes to the lowest layer with mass, and the
    # column heat content changes by exactly Δt btflx (no other flux in this one-step check of the harness alone)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, st, frc = C.build("willis_deardorff", "zstar", "gls")

    class Stub:
        dt = m.dt

        def step(self, st, *a, **kw):
            return st
    st.h = st.h.copy()
    st.h[1] += st.h[0] - 1e-6                                         # bed layer 1 µm thick, total depth kept
    st.h[0] = 1e-6
    T0 = st.T.copy()
    frc.case = C.CASES["willis_deardorff"]
    new = C.ColumnForcing.step(frc, Stub(), st)
    dT = new.T - T0
    assert np.all(dT[0] == 0.0) and np.all(dT[2:] == 0.0)
    # ΔT is a difference of T ≈ 20 °C values: round-off 20 ε ≈ 4e-15 °C, i.e. 3e-9 of the 1.5e-6 °C step
    np.testing.assert_allclose(dT[1] * st.h[1], m.dt * frc.case.btflx, rtol=1e-8)


def test_mixed_depth_definition():
    zw = np.linspace(-10.0, 0.0, 11)                                  # bed first
    k = np.where(zw > -4.0, 1e-3, 1e-7)                               # k ≥ 1e-5 from −3 m up
    k[6] = 1e-5 * 10**0.5                                             # at z = −4 m: half a decade above 1e-5
    k[5] = 1e-5 * 10**-0.5                                            # at z = −5 m: half a decade below
    assert mixed_depth(zw, k) == pytest.approx(4.5)


def test_kato_phillips_price_law_short():
    """6 h of KATO_PHILIPS (σ, GLS k-ε Canuto A): D_m within ±10 % of 1.05 u* N₀^(−1/2) t^(1/2)."""
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, st, frc = C.build("kato_phillips", "sigma", "gls")
        for _ in range(720):
            st = frc.step(m, st)
    assert st.time == pytest.approx(6 * 3600.0)
    zw = -50.0 + np.arange(101) * 0.5
    dm = mixed_depth(zw, np.asarray(st.closure["tke"]).reshape(101, -1)[:, 0])
    price = price_depth(6 * 3600.0)                                   # 15.43 m
    assert abs(dm / price - 1.0) < 0.10, (dm, price)


# ------------------------------------------------------------------ MOM6/CVMix SCM columns (N3 task 9, criterion 2)
# case: (CONST_WIND_TAUX [Pa], GUST_CONST [Pa], SENSIBLE_HEAT_FLUX [W m⁻²], SCM_L2_DTDZ [K m⁻¹]) — the MOM_override
# of MOM6-examples/ocean_only/CVmix_SCM_tests/<case>/KPP (WIND_CONFIG/BUOY_CONFIG "zero" = 0); common_KPP/MOM_input:
# NK = 400, MAXIMUM_DEPTH = 400, NIGLOBAL = NJGLOBAL = 2, DT = 1200, DAYMAX = 5, F_0 = 1e-4, G_EARTH = 9.80616,
# RHO_0 = 1000, DRHO_DT = −0.255, C_P = 3992.10322329649, SCM_L2_TEMP = 15, INTERP_TYPE = "cubic"
MOM_OVERRIDE = {"wind_only": (0.1, 0.0, 0.0, 0.01), "cooling_only": (0.0, 0.0, -100.0, 0.01),
                "mech_only": (0.0, 0.05, 0.0, 0.01), "skin_warming_wind": (0.1, 0.0, 100.0, 0.0)}


@pytest.mark.parametrize("name", list(MOM_OVERRIDE))
def test_mom6_scm_cases_match_mom_override(name):
    from validation.turbulence_columns import mom6_scm as M
    taux, gust, q, dtdz = MOM_OVERRIDE[name]
    c = M.CASES[name]
    assert (c.taux, c.gust, c.q, c.dtdz) == (taux, gust, q, dtdz)
    assert (M.NK, M.DEPTH, M.NXY, M.DT, M.DAYS, M.F0, M.G, M.RHO0, M.DRHO_DT, M.CP) == (
        400, 400.0, 2, 1200.0, 5.0, 1e-4, 9.80616, 1000.0, -0.255, 3992.10322329649)
    assert c.n_steps == 360
    np.testing.assert_allclose(c.ustar, np.sqrt((taux + gust) / 1000.0), rtol=1e-15)
    # B_f = −(g/ρ₀) dρ/dT Q/(ρ₀ c_p): ∓6.2638e-8 m² s⁻³ at ∓100 W m⁻² (MOM6 KPP_buoyFlux of the runs)
    np.testing.assert_allclose(c.bflux, 9.80616 / 1000.0 * 0.255 * q / (1000.0 * 3992.10322329649), rtol=1e-14)
    for coord in M.COORDS:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            m, st, tx, ty, heat = M.build(name, coord)
        assert m.H.shape == (2, 2) and np.all(m.H == 400.0) and m.periodic.x and m.periodic.y
        assert m.core.n_layers == 400 and m.dt == 1200.0 and m.rho0 == 1000.0 and m.cp == 3992.10322329649
        assert m.core.g == 9.80616 and m.t_coef == 0.255 and m.kv == 0.0 and m.core.z0 is None
        assert m._closure.params.interp_type == "cubic" and m._closure.params.ri_crit == 0.3
        np.testing.assert_allclose(m.hg["f_q"].values, 1e-4, rtol=1e-12, atol=0)
        h = st.h if st.h is not None else m.layer_volumes(st.eta) / m.area[None]
        np.testing.assert_allclose(h, 1.0, rtol=1e-12)
        zc = -400.0 + np.arange(400) + 0.5
        np.testing.assert_allclose(st.T[:, 0, 0], np.maximum(4.0, 15.0 + dtdz * zc), rtol=0, atol=1e-12)
        np.testing.assert_allclose(tx, taux) and np.testing.assert_allclose(ty, 0.0)
        assert (heat is None) == (q == 0.0) and (heat is None or np.all(heat["Q_ns"] == q))
        if gust:                                                      # u*² = gust/ρ₀ reaches the KPP update
            st = m.step(st, tx, ty, heat=heat)
            M.check_gust(m, c)
            assert np.all(st.u == 0.0) and np.all(st.v == 0.0)       # the gust carries no momentum


def test_mld_003_definition():
    # MOM6 diagnoseMLDbyDensityDifference: Δρ = 0.255 (T_s − T) between centres, 0.03 kg m⁻³ interpolated
    from validation.turbulence_columns.mom6_scm import mld_003
    zw = -np.arange(11.0)[::-1]                                       # bed first, 1 m layers
    Ts = np.array([15.0, 15.0, 15.0, 14.9, 14.8, 14.0, 14.0, 14.0, 14.0, 14.0])
    T = Ts[::-1]                                                      # bed first
    # Δρ at the centres 0.5, 1.5, …: 0, 0, 0, 0.0255, 0.051 → crossing between 3.5 and 4.5 m
    a = (0.03 - 0.0255) / 0.0255
    assert mld_003(zw, T) == pytest.approx(3.5 + a)
    assert mld_003(zw, np.full(10, 15.0)) == pytest.approx(9.5)      # never reached: the deepest centre
