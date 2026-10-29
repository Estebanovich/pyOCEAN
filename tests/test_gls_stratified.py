"""
N2a task 8c: buoyancy in the k-ε GLS closure (CROCO gls_mixing.F, GLS_KEPSILON + CANUTO_A) and the GLS tracer
diffusivity κ_t = c_μ′ k²/ε in the σ model (``gls_stratified=True``).

N² = −(g/ρ₀) ∂ρ/∂z from the model EOS, B = −κ_t N² in the k equation and β₃ B in the ψ (= ε) equation with CROCO's
Patankar split, α_N = (k/ε)² N² in the Canuto A stability functions, Galperin's l ≤ 0.53 √(2k)/N.

Targets: the steady-state Richardson number of the closure (Umlauf & Burchard 2003, eq. 28 and Table 8) and the
Kato & Phillips (1969) mixed-layer deepening as re-scaled by Price (1979), D = 1.05 u* (t/N₀)^½ (UB03 §3d).
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.components.eos import LinearEOS
from core.dynamics.sigma_wd import vertical_implicit
from physics.mixing.gls import (BETA1, BETA2, BETA3M, GLSKEpsilon, buoyancy_frequency2,
                                steady_state_richardson)
from tests import n1_cases as nc


def _shear_column(n: int = 30, m: int = 2, D: float = 15.0):
    """A sheared column: u linear in z plus a weak sine, bed stress from the bed velocity."""
    dz = np.full((n, m), D / n)
    z = (np.arange(n) + 0.5)[:, None] * D / n
    u = 0.02 * z + 0.05 * np.sin(np.pi * z / D) + 0.01 * np.arange(m)[None, :]
    v = 0.3 * u
    return u, v, dz


def test_unstratified_column_unchanged():
    """N² = 0 (uniform ρ): the stratified branch gives the pre-8c ν (≤ 1e-14 relative) over 200 steps."""
    u, v, dz = _shear_column()
    a = GLSKEpsilon(dz.shape[0], dz.shape[1], z0_bed=1e-3)
    b = GLSKEpsilon(dz.shape[0], dz.shape[1], z0_bed=1e-3, stratified=True)
    rho = np.full(dz.shape, 1025.0)
    n2 = buoyancy_frequency2(rho, dz, 1025.0)
    assert np.all(n2 == 0.0)
    for _ in range(200):
        us2 = 2.5e-3 * u[0] ** 2
        a.step(u, v, dz, us2, 30.0)
        b.step(u, v, dz, us2, 30.0, n2=n2)
    np.testing.assert_allclose(b.akv, a.akv, rtol=1e-14, atol=0)
    np.testing.assert_allclose(b.tke, a.tke, rtol=1e-14, atol=0)
    assert np.all(b.akt >= b.kt_min) and np.all(np.isfinite(b.akt))


def test_unstratified_model_unchanged():
    """
    In the σ model with uniform T and no wind (a 0.5 m seiche, turbulence seeded): gls_stratified True and False
    give the same ν and state bit for bit.  With wind they differ by design: the stratified branch also feeds the
    surface u*² to k and ψ (CROCO gls_mixing.F:556-562, 678, 696; ``test_wind_sets_the_surface_tke``).
    """
    warnings.simplefilter("ignore")
    out = []
    for strat in (False, True):
        m = nc._sigma(closure="gls", z0=1e-3, gls_stratified=strat)
        m.core.gls.tke[:], m.core.gls.eps[:] = 1e-4, 1e-7
        st = m.initial_state(10.0 * nc._eta(), np.full((nc.NZ, nc.NY, nc.NX), 12.0))
        for n in range(100):
            st = m.step(st)
        out.append((m.core.gls.akv.copy(), st.u.copy(), st.eta.copy(), st.T.copy()))
    assert out[0][0].max() > 1e-3                                   # turbulent, not at the floor
    for x, y in zip(out[0][:3], out[1][:3]):                         # ν, u, η
        np.testing.assert_allclose(y, x, rtol=1e-14, atol=1e-18)
    # T: uniform, but the stratified run diffuses it with κ_t (round-off only)
    np.testing.assert_allclose(out[1][3], out[0][3], rtol=1e-14, atol=0)


def test_wind_sets_the_surface_tke():
    """Stratified branch: k at the surface interface = u*²/c_μ⁰² with u*² = |τ_s|/ρ₀ (CROCO gls_mixing.F:678)."""
    warnings.simplefilter("ignore")
    m = nc._sigma(closure="gls", z0=1e-3)
    st = m.initial_state(nc._eta(), nc._T())
    tx, ty = nc._wind(True)(0)
    st = m.step(st, tau_x=tx, tau_y=ty)
    us2 = np.hypot(0.05, 0.02) / nc.RHO0
    np.testing.assert_allclose(m.core.gls.tke[-1], us2 / m.core.gls.cm0**2, rtol=1e-12)


def test_buoyancy_frequency_from_density():
    """N² = (g/ρ₀)(ρ_{k−1} − ρ_k)/h at the interior interfaces (CROCO rho_eos.F:404-411), k = 0 at the bed."""
    dz = np.array([[1.0], [2.0], [3.0]])
    rho = np.array([[1026.0], [1025.5], [1025.0]])
    n2 = buoyancy_frequency2(rho, dz, 1025.0, g=9.81)
    np.testing.assert_allclose(n2[:, 0], 9.81 / 1025.0 * np.array([0.5 / 1.5, 0.5 / 2.5]), rtol=1e-15)


def test_steady_richardson_of_the_closure_matches_umlauf_burchard():
    """
    UB03 eq. (28), Ri_st = (c_μ/c_μ′)(c_ψ2 − c_ψ1)/(c_ψ2 − c_ψ3) with the Canuto A stability functions in
    equilibrium (P + B = ε): with UB03 Table 8's k-ε (R87) value c_ψ3 = −0.629 the solver gives Ri_st = 0.25.
    """
    assert steady_state_richardson(-0.629) == pytest.approx(0.25, abs=0.005)
    ri = steady_state_richardson(BETA3M)                 # CROCO's β₃⁻ = −0.4 (gls_mixing.F:179)
    assert 0.25 < ri < 0.35


def test_steady_richardson_number():
    """
    Homogeneous shear S² and uniform N² (a tall column; the middle interface is far from the ends): k grows for
    Ri = N²/S² below Ri_st and decays above it; the Ri where k is stationary is the closure's Ri_st ± 10 %.
    """
    ri_st = steady_state_richardson(BETA3M)
    factors = np.array([0.7, 0.85, 1.15, 1.3])
    m, n, dz_, S = factors.size, 120, 1.0, 0.01
    dz = np.full((n, m), dz_)
    z = (np.arange(n) + 0.5)[:, None] * dz_
    u = S * z * np.ones((1, m))
    g = GLSKEpsilon(n, m, stratified=True)
    g.tke[:] = 1e-4
    g.eps[:] = 1e-4 / 300.0
    n2 = np.ones((n - 1, 1)) * (factors * ri_st * S**2)[None, :]
    dt, steps, mid = 5.0, 2400, n // 2
    lnk = []
    for _ in range(steps):
        g.step(u, np.zeros_like(u), dz, np.zeros(m), dt, n2=n2)
        lnk.append(np.log(g.tke[mid]))
    lnk = np.array(lnk)
    t = np.arange(steps) * dt
    late = slice(2 * steps // 3, None)
    rate = np.polyfit(t[late], lnk[late], 1)[0]             # d ln k / dt per column [s⁻¹]
    assert rate[0] > 0 and rate[1] > 0 and rate[2] < 0 and rate[3] < 0, rate
    root = np.interp(0.0, -rate[1:3], factors[1:3])          # linear zero crossing between 0.85 and 1.15
    assert root == pytest.approx(1.0, abs=0.10), (root, rate)


def _kato_phillips(hours: float = 30.0, ustar: float = 0.01, N2: float = 1e-4, D: float = 50.0, n: int = 100,
                   dt: float = 30.0):
    """1-D column: wind stress u*² on the top layer, no rotation, T-only linear stratification N₀² via the model
    EOS (α > 0, no S); momentum, then GLS, then T (CROCO step.F:982, 1094, 1100)."""
    eos = LinearEOS(rho0=1025.0, t_coef=0.2, t_ref=10.0)
    dz = np.full((n, 1), D / n)
    z = (np.arange(n) + 0.5)[:, None] * D / n                     # height above the bed
    T = 10.0 + N2 * eos.rho0 / (9.81 * eos.t_coef) * (z - D)       # N² = (g/ρ₀) α ∂T/∂z
    u = np.zeros((n, 1))
    g = GLSKEpsilon(n, 1, stratified=True)
    us2 = np.array([ustar**2])
    for _ in range(int(round(hours * 3600 / dt))):
        u = vertical_implicit(u, dz, g.akv, np.zeros(1), dt, tau_top=us2)
        n2 = buoyancy_frequency2(eos.density(T), dz, eos.rho0)
        g.step(u, np.zeros_like(u), dz, np.zeros(1), dt, ustar2_sfc=us2, n2=n2)
        T = vertical_implicit(T, dz, g.akt, np.zeros(1), dt)
    n2 = buoyancy_frequency2(eos.density(T), dz, eos.rho0)[:, 0]
    zw = np.arange(1, n) * D / n
    return D - zw[np.argmax(n2)], T, g


def test_kato_phillips_mixed_layer():
    """Mixed-layer depth (depth of the N² maximum at its base) after 30 h vs Price (1979) ± 15 %."""
    ustar, N2, t = 0.01, 1e-4, 30.0 * 3600.0
    mld, T, g = _kato_phillips(ustar=ustar, N2=N2)
    price = 1.05 * ustar * np.sqrt(t / np.sqrt(N2))                # 34.5 m
    assert mld == pytest.approx(price, rel=0.15), (mld, price)
    assert np.all(np.diff(T[:, 0]) >= -1e-10)                      # still stably stratified (warm on top)
    assert g.akt[-5, 0] > 1e-3 > g.akt[5, 0]                       # mixing in the layer, background below


def _sal_model(**kw):
    """σ + GLS with S (weak haline stratification, T uniform), turbulence seeded."""
    warnings.simplefilter("ignore")
    m = nc._sigma(closure="gls", z0=1e-3, salinity=True, s_coef=0.78, s_ref=30.0, variance_budget=True, **kw)
    m.core.gls.tke[:], m.core.gls.eps[:] = 1e-4, 1e-7
    k = np.arange(nc.NZ)[:, None, None]
    S = 30.0 - 0.2 * k / (nc.NZ - 1) - 0.1 * np.arange(nc.NX)[None, None, :] / nc.NX
    st = m.initial_state(10.0 * nc._eta(), np.full((nc.NZ, nc.NY, nc.NX), 12.0), S=S * np.ones((1, nc.NY, 1)))
    return m, st


def test_gls_tracer_diffusivity_used(monkeypatch):
    """
    σ + GLS with S: the vertical diffusion of T and S uses the closure's κ_t (not a constant); forcing κ_t
    constant changes S; the variance budget still closes.
    """
    import core.model.driver as drv
    calls = []
    orig = drv.vertical_implicit

    def spy(c, dz, nu, r_bed, dt, *a, **kw):
        calls.append(np.array(nu, copy=True))
        return orig(c, dz, nu, r_bed, dt, *a, **kw)

    m, st = _sal_model()
    assert m.gls_stratified and m.kv == 0.0
    monkeypatch.setattr(drv, "vertical_implicit", spy)
    wind = nc._wind(True)
    akt_max = 0.0
    for n in range(30):
        calls.clear()
        V0, S0, T0 = m.layer_volumes(st.eta, st.h), st.S, st.T
        tx, ty = wind(n)
        st = m.step(st, tau_x=tx, tau_y=ty)
        assert len(calls) == 2                                      # T and S
        akt = m.core.gls.akt
        for nu in calls:
            np.testing.assert_array_equal(nu, akt)
        V1, b = m.layer_volumes(st.eta, st.h), st.budget
        for c, c0, c1 in (("S", S0, st.S), ("T", T0, st.T)):
            lhs = (V1 * c1**2).sum() - (V0 * c0**2).sum()
            rhs = m.dt * (b[f"{c}_surf"].sum() - b[f"{c}_num_adv"].sum() - b[f"{c}_phy_v"].sum()
                          - b[f"{c}_phy_h"].sum() - b[f"{c}_sponge"].sum())
            assert abs(lhs - rhs) <= 1e-13 * (V0 * c0**2).sum(), (c, lhs, rhs)
        assert b["S_phy_v"].sum() > 0
        akt_max = max(akt_max, akt[1:-1].max())
    assert akt_max > 100 * m.core.gls.kt_min                           # κ_t from the closure, not a constant
    assert np.all(akt >= m.core.gls.kt_min)
    # κ_t forced constant: another S
    monkeypatch.setattr(drv, "vertical_implicit", orig)
    m2, st2 = _sal_model()
    m3, st3 = _sal_model()
    for n in range(30):
        tx, ty = wind(n)
        st2 = m2.step(st2, tau_x=tx, tau_y=ty)
        m3.core.gls.akt[:] = 1e-4
        st3 = m3.step(st3, tau_x=tx, tau_y=ty)
    assert np.abs(st2.S - st3.S).max() > 1e-8


def test_kv_is_the_gls_tracer_floor():
    """``kv`` > 0 with closure='gls' is the floor of κ_t (CROCO's nuws role); kv = 0 keeps CROCO's 1e-6."""
    m, _ = _sal_model()
    assert m.core.gls.kt_min == 1e-6
    m2, st = _sal_model(kv=1e-4)
    assert m2.core.gls.kt_min == 1e-4
    st = m2.step(st)
    assert m2.core.gls.akt.min() >= 1e-4


def test_gls_stratified_preflight_warns_without_gls():
    from core.model.contract import PyOceanConfigWarning
    with pytest.warns(PyOceanConfigWarning, match="gls_stratified"):
        nc._sigma(closure="constant", gls_stratified=False)      # a non-default value: 'set'


def test_constants():
    """CROCO GLS_KEPSILON β constants (gls_mixing.F:179)."""
    assert (BETA1, BETA2, BETA3M) == (1.44, 1.92, -0.4)
