"""
CROCO single-column cases in pyOCEAN (N3 task 8; validation/turbulence_columns/, spec
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


def test_body_force_enters_the_explicit_momentum():
    # the harness's named term "body_force" (∂v/∂t += f U_G in FORCED_EKBBL) is summed with the other terms
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m, st, frc = C.build("forced_ekbbl", "sigma", "gls")
    terms, wu, wv = m._momentum_terms(st)
    assert terms[-1][0] == "body_force"
    np.testing.assert_allclose(terms[-1][2], 1e-4 * 10.0, rtol=1e-12)
    st = frc.step(m, st)                                              # first step from rest: v grows by ≈ Δt f U_G
    assert 0.5 * 30.0 * 1e-3 < st.v[-1, 2, 2] <= 30.0 * 1e-3 * (1 + 1e-9)


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
