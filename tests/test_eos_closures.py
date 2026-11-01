"""
TEOS-10 in the 3-D core, Task 6 (plan docs/superpowers/plans/2026-10-31-teos10-3d.md; spec
docs/superpowers/specs/2026-10-31-teos10-3d-design.md goal C, criterion 6, review focus 3): the N² and buoyancy of
the turbulence closures with a nonlinear EOS, as MOM6.

* GLS, Jackson and BBL: locally referenced N² (∂ρ/∂T, ∂ρ/∂S at the interface mean T, S and pressure;
  MOM_kappa_shear.F90:1128-1136, 1496-1500), ``physics.eos.stratification.n2_local``.
* KPP: density of both parts at the pressure of the top interface of layer k (MOM_CVMix_KPP.F90:1396-1436),
  ``kpp_delta_rho``; surface buoyancy flux with ∂ρ/∂T, ∂ρ/∂S of the top layer at p = 0.
* Reference: compiled MOM6 (``validation/mom6_column/eos_driver n2`` on the columns of ``eos_columns.n2_cases``,
  ``tests/data/mom6_n2_reference.npz``).
* Linear EOS: every closure bit for bit with and without the density context.

Index order (k, column), k = 0 at the bed.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest

from core.components.closure import (BBLAddedClosure, ConstantClosure, GLSClosure, KPPClosure, ShearAddedClosure,
                                     thin_layer_threshold)
from core.components.eos import make_eos
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import PyOceanConfigWarning
from core.vertical import remap as rm
from physics.eos.stratification import (DensityContext, buoyancy_jumps, interface_pressure, kpp_delta_rho, n2_local,
                                        surface_alpha_beta)
from physics.mixing import kpp_cvmix
from physics.mixing.bbl_mixing import (BBLParams, _merged_n2_local, lotw_bbl_columns, lotw_bbl_columns_numpy)
from physics.mixing.kappa_shear import (JacksonParams, _local_buoyancy, kappa_shear_columns,
                                        kappa_shear_columns_numpy)
from physics.mixing.kpp_cvmix import kpp_column

REF = np.load(Path(__file__).parent / "data" / "mom6_n2_reference.npz")
NAMES = [str(x) for x in REF["names"]]
EOS_NAMES = ("wright_full", "teos10")


def _ref_column(n: int):
    h, T, S, T_sl, S_sl = (REF[f"c{n}_{k}"] for k in ("h", "T", "S", "T_sl", "S_sl"))
    rho0, g = REF[f"c{n}_par"]
    return h, T, S, T_sl, S_sl, float(rho0), float(g)


# ---------------------------------------------------------------- against compiled MOM6
@pytest.mark.parametrize("eos_name", EOS_NAMES)
def test_n2_local_matches_mom6(eos_name):
    """Locally referenced N² of MOM6's kappa_shear on every column (z* with partial cells and 1e-10 m levels, σ,
    1 mm / 1e-10 m layers, bottom step, thermobaric pair): interface pressure bit for bit, N² to rtol 1e-12."""
    eos = make_eos(eos_name)
    for n, name in enumerate(NAMES):
        h, T, S, _, _, rho0, g = _ref_column(n)
        p = interface_pressure(h, g, rho0)
        assert np.array_equal(p[1:-1], REF[f"c{n}_{eos_name}_p_if"]), name
        n2 = n2_local(T, S, h, p, eos, rho0=rho0, g=g)
        np.testing.assert_allclose(n2, REF[f"c{n}_{eos_name}_n2"], rtol=1e-12, atol=0, err_msg=name)
        # the jump without the 1/Δz is the same buoyancy difference (Jackson's input)
        dz_int = 0.5 * (h[1:] + h[:-1])
        np.testing.assert_allclose(buoyancy_jumps(T, S, p, eos, rho0=rho0, g=g) / dz_int, n2, rtol=1e-12,
                                   atol=1e-300, err_msg=name)


@pytest.mark.parametrize("eos_name", EOS_NAMES)
def test_kpp_delta_rho_matches_mom6(eos_name):
    """KPP density differences at the top-interface pressure of layer k (MOM_CVMix_KPP.F90:1396-1436): ρ_k − ρ_sl and
    ρ_k − ρ_k+1 with one EOS call each, to rtol 1e-12."""
    eos = make_eos(eos_name)
    for n, name in enumerate(NAMES):
        h, T, S, T_sl, S_sl, rho0, g = _ref_column(n)
        p_top = interface_pressure(h, g, rho0)[1:]
        assert np.array_equal(p_top, REF[f"c{n}_{eos_name}_p_top"]), name
        d_sl, d_up = kpp_delta_rho(T, S, T_sl, S_sl, p_top, eos)
        np.testing.assert_allclose(d_sl, REF[f"c{n}_{eos_name}_drho_sl"], rtol=1e-12, atol=1e-13, err_msg=name)
        np.testing.assert_allclose(d_up, REF[f"c{n}_{eos_name}_drho_up"], rtol=1e-12, atol=1e-13, err_msg=name)


@pytest.mark.parametrize("eos_name", EOS_NAMES)
def test_kpp_column_receives_mom6_differences(eos_name, monkeypatch):
    """``kpp_column`` with T, S and a nonlinear EOS hands CVMix MOM6's ΔB(k) = (g/ρ₀)(ρ_k − ρ_sl) and local N
    (``bulk_richardson`` and ``unresolved_shear`` inputs captured), not differences of b."""
    eos = make_eos(eos_name)
    seen = {}
    real_ri, real_vt2 = kpp_cvmix.bulk_richardson, kpp_cvmix.unresolved_shear

    def ri(zt, delta_b, *a, **k):
        seen["delta_b"] = delta_b.copy()
        return real_ri(zt, delta_b, *a, **k)

    def vt2(zt, ws, n_if, *a, **k):
        seen["n_if"] = n_if.copy()
        return real_vt2(zt, ws, n_if, *a, **k)
    monkeypatch.setattr(kpp_cvmix, "bulk_richardson", ri)
    monkeypatch.setattr(kpp_cvmix, "unresolved_shear", vt2)
    n = NAMES.index("gulf_zstar_3")
    h, T, S, _, _, rho0, g = _ref_column(n)
    nz = h.size
    zw = np.concatenate([[0.0], np.cumsum(h)])[:, None]
    b_wrong = np.full((nz, 1), 7.0)                                    # must not be used
    kpp_column(h[:, None], zw, np.zeros((nz, 1)), np.zeros((nz, 1)), b_wrong, np.array([1e-8]), np.array([0.01]),
               None, None, T=T[:, None], S=S[:, None], eos=eos, rho0=rho0, g=g)
    go = g / rho0
    n2 = (go * REF[f"c{n}_{eos_name}_drho_up"][::-1]) / (0.5 * (h[::-1][:-1] + h[::-1][1:]) + kpp_cvmix.H_SUBROUNDOFF)
    np.testing.assert_allclose(seen["n_if"][1:nz, 0], np.sqrt(np.maximum(n2, 0.0)), rtol=1e-12, atol=0)
    assert seen["n_if"][1:nz].any()
    # ΔB: the same EOS calls with KPP's own surface-layer T, S (ΔB = 0 in the top layer, which is its surface layer)
    assert seen["delta_b"][0, 0] == 0.0 and np.abs(seen["delta_b"]).max() > 0


# ---------------------------------------------------------------- thermobaricity (review focus 3)
@pytest.mark.parametrize("eos_name", EOS_NAMES)
def test_thermobaric_sign(eos_name):
    """
    Cold and fresh water over warm and salty water (``eos_columns.thermobaric_columns``): the locally referenced N²
    is > 0 with the interface at 100 m and < 0 at 4000 m (MOM6 and pyOCEAN, both EOS), and the closures see that
    sign (GLS, BBL, Jackson).  The in-situ density difference of the deep pair, each layer at its own centre pressure,
    is stable-looking: the opposite sign, which is why the closures must not use it.
    """
    eos = make_eos(eos_name)
    signs = {"thermo_100": 1.0, "thermo_4000": -1.0}
    for name, sign in signs.items():
        n = NAMES.index(name)
        h, T, S, _, _, rho0, g = _ref_column(n)
        assert sign * REF[f"c{n}_{eos_name}_n2"][0] > 0                # compiled MOM6
        p = interface_pressure(h, g, rho0)
        n2 = n2_local(T, S, h, p, eos, rho0=rho0, g=g)
        assert sign * n2[0] > 0
        # GLS (ALE, direct step): the N² kept for the remap
        gls = GLSClosure(h.size, (1, 1), 30.0, stratified=True, rho0=rho0, g=g)
        gls.bind_ale(rm.PPM_H4)
        dens = _dens(eos, T[:, None], S[:, None], h[:, None], rho0, g)
        z = np.zeros((h.size, 1, 1))
        gls.update(z, z, h[:, None, None], dens.rho, np.zeros((1, 1)), np.full((1, 1), 1e-6), dens=dens)
        assert sign * gls._n2[1, 0] > 0 and gls._n2[1, 0] == n2[0]
        # BBL (merged-column N²; no thin layer) and Jackson (locally referenced buoyancy)
        lay = np.arange(h.size)
        code = 1 if eos_name == "wright_full" else 2
        assert _merged_n2_local(T, S, h, lay, h.copy(), h.size, rho0, g, code)[1] == n2[0]
        hb = np.zeros(h.size)
        _local_buoyancy(T.copy(), S.copy(), h.copy(), h.size, code, rho0, g, hb)
        assert sign * (hb[1] - hb[0]) > 0
        if name == "thermo_4000":                                      # in-situ difference: the wrong sign
            pc = 0.5 * (p[:-1] + p[1:])
            rho = eos.density(T, S, pc)
            assert rho[0] - rho[1] > 0                                 # lower layer denser in situ: "stable"


# ---------------------------------------------------------------- closures
def _dens(eos, T, S, h, rho0, g):
    """DensityContext of columns (N, M) (driver ``_closure_density`` form)."""
    above = np.cumsum(h[::-1], axis=0)[::-1]
    p = g * rho0 * (above - 0.5 * h)
    rho = eos.density(T, S) if eos.linear else eos.density(T, S, p)
    return DensityContext(eos, T, S, p, rho)


def _columns(n: int = 14, seed: int = 3):
    """Columns (N, M) with and without thin layers (as tests/test_gls_ale.py::_thin_columns), random sheared u, v
    and a stable, partly thermobaric T, S (S and T both decreasing upward in places)."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.5, 6.0, (n, 10))
    h[:3, 0] = 0.0
    h[-4:, 1] = 1e-12
    h[2:n:3, 2] = 1e-10
    h[5:9, 3] = 5e-4
    h[:, 4] = 0.0                                                     # dry
    h[1::2, 8] = 2e-4
    h[-3:, 9] = 0.01
    zc = np.cumsum(h, axis=0) - 0.5 * h
    u = 0.3 * np.tanh((zc - 15.0) / 6.0) + 0.05 * rng.standard_normal(h.shape)
    v = 0.1 * rng.standard_normal(h.shape)
    T = 4.0 + 0.4 * zc + 0.05 * rng.standard_normal(h.shape)
    S = 34.6 + 0.01 * zc + 0.005 * rng.standard_normal(h.shape)
    return h, u, v, T, S


def _linear():
    return make_eos("linear", 1025.0, t_coef=0.17, t_ref=10.0, s_coef=0.76, s_ref=35.0)


def _gls(n, m, ale):
    cl = GLSClosure(n, (1, m), 30.0, stratified=True, rho0=1025.0)
    if ale:
        cl.bind_ale(rm.PPM_H4)
    return cl


def _state(cl):
    g = cl.gls
    return [g.tke.copy(), g.eps.copy(), g.akv.copy(), g.akt.copy(), None if cl._n2 is None else cl._n2.copy()]


def test_closures_linear_unchanged():
    """GLS (σ and ALE with merged columns), KPP (with surface fluxes), Jackson and BBL with the linear EOS give, bit
    for bit, the same with the density context as without it."""
    h, u, v, T, S = _columns()
    n, m = h.shape
    eos = _linear()
    dens = _dens(eos, T, S, h, 1025.0, 9.81)
    rho = dens.rho
    sh = (n, 1, m)
    u3, v3, h3, r3 = (x.reshape(sh) for x in (u, v, h, rho))
    us2 = np.full((1, m), 1e-4)
    for ale in (False, True):
        hh = h3 if ale else np.where(h3 > 0, h3, 1.0)               # σ columns have no empty layer
        a, b = _gls(n, m, ale), _gls(n, m, ale)
        for _ in range(3):
            a.update(u3, v3, hh, r3, us2, 0.1 * us2)
            b.update(u3, v3, hh, r3, us2, 0.1 * us2, dens=dens)
        for x, y in zip(_state(a), _state(b)):
            assert (x is None and y is None) or np.array_equal(x, y)
    bflux = dict(Q_ns=np.full((1, m), -150.0), Q_sw=np.full((1, m), 300.0),
                 sw_frac_iface=np.linspace(0.0, 1.0, n + 1)[:, None, None] * np.ones((1, 1, m)),
                 fw=np.full((1, m), 1e-7), S_sfc=S[-1].reshape(1, m))
    a, b = (KPPClosure(n, (1, m), rho0=1025.0, t_coef=0.17, s_coef=0.76) for _ in range(2))
    a.update(u3, v3, h3, r3, us2, None, bflux=bflux)
    b.update(u3, v3, h3, r3, us2, None, bflux=bflux, dens=dens)
    for name in ("akv", "akt", "gamma", "obl_depth", "F_T", "F_S"):
        assert np.array_equal(getattr(a, name), getattr(b, name)), name
    f2 = np.full((1, m), 1e-8)
    a, b = (ShearAddedClosure(ConstantClosure(), n, (1, m), 60.0, f2, kv=1e-5, rho0=1025.0) for _ in range(2))
    a.update(u3, v3, h3, r3, None, None)
    b.update(u3, v3, h3, r3, None, None, dens=dens)
    assert np.array_equal(a.kd, b.kd) and a.kd.max() > 0
    a, b = (BBLAddedClosure(ConstantClosure(), n, (1, m), np.full((1, m), 1e-4), kv=1e-5, rho0=1025.0)
            for _ in range(2))
    E = np.full((1, m), 1e-6)
    a.update(u3, v3, h3, r3, None, np.full((1, m), 1e-4), bbl_energy=E)
    b.update(u3, v3, h3, r3, None, np.full((1, m), 1e-4), bbl_energy=E, dens=dens)
    assert np.array_equal(a.kd, b.kd) and a.kd.max() > 0


@pytest.mark.parametrize("eos_name", EOS_NAMES)
def test_gls_nonlinear_merged_kernel_matches_oracle(eos_name):
    """The compiled merged GLS step with the locally referenced N² (``merged_columns``, eos_code > 0) equals the NumPy
    oracle ``_update_ale_numpy`` from the same state (rtol 1e-13, as the linear test of tests/test_gls_ale.py); the
    direct step's N² is ``n2_local`` exactly."""
    h, u, v, T, S = _columns()
    n, m = h.shape
    eos = make_eos(eos_name)
    dens = _dens(eos, T, S, h, eos.rho0, 9.81)
    a, b = (GLSClosure(n, (1, m), 30.0, stratified=True, rho0=eos.rho0) for _ in range(2))
    for cl in (a, b):
        cl.bind_ale(rm.PPM_H4)
    rng = np.random.default_rng(5)
    for step in range(10):
        us2s, us2b = rng.uniform(0, 2e-4, (1, m)), rng.uniform(0, 5e-5, (1, m))
        b.gls.tke, b.gls.eps, b.gls.akv, b.gls.akt = (x.copy() for x in (a.gls.tke, a.gls.eps, a.gls.akv, a.gls.akt))
        a.update(u.reshape(n, 1, m), v.reshape(n, 1, m), h.reshape(n, 1, m), dens.rho.reshape(n, 1, m), us2s, us2b,
                 dens=dens)
        b._update_ale_numpy(u, v, h, dens.rho, us2s.ravel(), us2b.ravel(), None, dens=dens)
        for name in ("tke", "eps", "akv", "akt", "last_prod"):
            np.testing.assert_allclose(getattr(a.gls, name), getattr(b.gls, name), rtol=1e-13, atol=0,
                                       err_msg=f"{name} step {step}")
        np.testing.assert_allclose(a._n2, b._n2, rtol=1e-13, atol=0, err_msg=f"n2 step {step}")
    assert a.gls.akv.max() > 1e-4
    # a column without thin layers (5, 6, 7) takes the direct step: N² = n2_local on the model layers
    c = np.array([5, 6, 7])
    n2 = n2_local(T[:, c], S[:, c], h[:, c], interface_pressure(h[:, c], 9.81, eos.rho0), eos)
    assert np.array_equal(a._n2[1:-1, c], n2)


@pytest.mark.parametrize("eos_name", EOS_NAMES)
def test_bbl_and_jackson_nonlinear_kernels_match_oracles(eos_name):
    """BBL and Jackson with the locally referenced stratification: Numba kernels bit for bit with their NumPy oracles,
    and a column without thin layers gets ``n2_local`` exactly (BBL) or a buoyancy whose jumps are MOM6's (Jackson)."""
    h, u, v, T, S = _columns()
    n, m = h.shape
    eos = make_eos(eos_name)
    rho = _dens(eos, T, S, h, eos.rho0, 9.81).rho
    thr = thin_layer_threshold(h)
    ustar, absf, tke = np.full(m, 0.02), np.full(m, 1e-4), np.full(m, 2e-6)
    kw = dict(rho0=eos.rho0, g=9.81, T=T, S=S, eos=eos)
    p = BBLParams(use_omega=False)
    a = lotw_bbl_columns(h, np.full_like(rho, np.nan), ustar, absf, tke, thr, p, **kw)   # ρ unused
    b = lotw_bbl_columns_numpy(h, rho, ustar, absf, tke, thr, p, **kw)
    assert np.array_equal(a, b) and a.max() > 0
    c = 6
    lay = np.arange(n)
    n2 = n2_local(T[:, c], S[:, c], h[:, c], interface_pressure(h[:, c], 9.81, eos.rho0), eos)
    assert np.array_equal(_merged_n2_local(T[:, c], S[:, c], h[:, c], lay, h[:, c].copy(), n, eos.rho0, 9.81,
                                           1 if eos_name == "wright_full" else 2)[1:-1], n2)
    jp = JacksonParams()
    f2 = np.full(m, 1e-8)
    ka, qa = kappa_shear_columns(u, v, np.zeros_like(u), h, f2, 60.0, 1e-7, thr, jp, **kw)
    kb, qb = kappa_shear_columns_numpy(u, v, np.zeros_like(u), h, f2, 60.0, 1e-7, thr, jp, **kw)
    assert np.array_equal(ka, kb) and np.array_equal(qa, qb) and ka.max() > 0
    hb = np.zeros(n)
    _local_buoyancy(T[:, c].copy(), S[:, c].copy(), h[:, c].copy(), n, 1 if eos_name == "wright_full" else 2,
                    eos.rho0, 9.81, hb)
    jumps = buoyancy_jumps(T[:, c], S[:, c], interface_pressure(h[:, c], 9.81, eos.rho0), eos)
    np.testing.assert_allclose(np.diff(hb), jumps, rtol=1e-12, atol=1e-18)


@pytest.mark.parametrize("eos_name", EOS_NAMES)
def test_kpp_surface_buoyancy_flux(eos_name):
    """KPP surface buoyancy flux with a nonlinear EOS: ∂ρ/∂T, ∂ρ/∂S of the top layer at p = 0
    (MOM_forcing_type.F90:1138-1148), B_ns = −(g/ρ₀)(∂ρ/∂S F_S + ∂ρ/∂T F_T), B_sw = −(g/ρ₀) ∂ρ/∂T Q_sw/(ρ₀ c_p)."""
    eos = make_eos(eos_name)
    T_top, S_top = np.array([25.0, 2.0, 15.0]), np.array([36.0, 34.0, 0.0])
    dT, dS = surface_alpha_beta(T_top, S_top, eos)
    eT, eS = eos.density_derivs(T_top, S_top, np.zeros(3))
    assert np.array_equal(dT, eT) and np.array_equal(dS, eS) and (dT < 0).all() and (dS > 0).all()
    cl = KPPClosure(4, (1, 3), rho0=eos.rho0, cp=3991.86795711963)
    Q, Qsw, fw = np.array([-200.0, 50.0, 0.0]), np.array([400.0, 0.0, 100.0]), np.array([1e-7, -2e-7, 0.0])
    F_T, F_S, B_ns, B_sw = cl.buoyancy_fluxes(Q, Qsw, fw, S_top, drho=(dT, dS))
    go, rc = 9.81 / eos.rho0, eos.rho0 * 3991.86795711963
    assert np.array_equal(B_ns, -go * (dS * F_S + dT * F_T))
    assert np.array_equal(B_sw, -go * (dT * (Qsw / rc)))
    # equals the linear formula with α = −∂ρ/∂T, β = ∂ρ/∂S
    lin = KPPClosure(4, (1, 3), rho0=eos.rho0, cp=3991.86795711963)
    for i in range(3):
        lin.t_coef, lin.s_coef = -dT[i], dS[i]
        _, _, bn, bs = lin.buoyancy_fluxes(Q[i:i + 1], Qsw[i:i + 1], fw[i:i + 1], S_top[i:i + 1])
        np.testing.assert_allclose([B_ns[i], B_sw[i]], [bn[0], bs[0]], rtol=1e-14, atol=1e-25)


# ---------------------------------------------------------------- the model with a nonlinear EOS
NX, NY, NZ = 8, 6, 10
H = (40.0 + 5.0 * np.arange(NX))[None, :] * np.ones((NY, 1))
HMAX = float(H.max())


def _front():
    """Stable T, S with a front in x (tests/test_eos_model.py::_front)."""
    k = np.arange(NZ)[:, None, None] / (NZ - 1)
    i = np.arange(NX)[None, None, :] / NX
    return (10.0 + 4.0 * k + 1.0 * i) * np.ones((NZ, NY, NX)), (35.2 - 0.3 * k - 0.4 * i) * np.ones((NZ, NY, NX))


CASES = {
    "gls_sigma": dict(closure="gls", gls_stratified=True, z0=1e-3, pgf="jacobian"),
    "gls_hybrid": dict(closure="gls", gls_stratified=True, z0=1e-3, vertical_coordinate="ale", pgf="fv",
                       regrid="hybrid", regrid_params=dict(dz_top=np.full(NZ, HMAX / NZ),
                                                           rho_target=np.linspace(1025.8, 1027.6, NZ + 1))),
    "kpp": dict(closure="kpp", vertical_coordinate="ale", pgf="fv", regrid="zstar",
                regrid_params=dict(dz_top=np.full(NZ, HMAX / NZ))),
    "jackson": dict(closure="constant", shear_mixing="jackson", vertical_coordinate="ale", pgf="fv", regrid="zstar",
                    regrid_params=dict(dz_top=np.full(NZ, HMAX / NZ))),
    "bbl": dict(closure="constant", bbl_mixing="lotw", bottom_drag="quadratic", cd=2e-3, hbbl=5.0,
                vertical_coordinate="ale", pgf="fv", regrid="zstar", regrid_params=dict(dz_top=np.full(NZ, HMAX / NZ))),
}


@pytest.mark.parametrize("case", sorted(CASES))
def test_closures_run_nonlinear(case, monkeypatch):
    """Each closure takes 20 finite steps in the box of Task 4 with ``wright_full`` and a wind stress, and receives
    the nonlinear density context from the driver."""
    base = dict(nu_const=1e-4, kv=1e-5, ah=5.0, theta=0.6, d_min=0.05, salinity=True, eos="wright_full",
                rho_ref="auto")
    base.update(CASES[case])
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        m = SigmaBaroclinic(create_cartesian_grid(nx=NX, ny=NY, dx=2000.0, dy=2000.0, f0=1e-4), H, dt=60.0,
                            n_layers=NZ, **base)
    seen = []
    cl = m._closure
    real = cl.update

    def spy(*a, **k):
        seen.append(k.get("dens"))
        return real(*a, **k)
    monkeypatch.setattr(cl, "update", spy)
    T, S = _front()
    st = m.initial_state(np.zeros((NY, NX)), T, S=S)
    if case == "jackson":                                              # a shear with Ri < 1/4 to mix
        st.u = st.u + np.linspace(-0.8, 0.8, NZ)[:, None, None]
        st.u[:, :, 0] = st.u[:, :, -1] = 0.0
    tx, ty = np.full((NY, NX + 1), 2e-4), np.zeros((NY + 1, NX))
    for _ in range(20):
        st = m.step(st, tx, ty)
    for name in ("eta", "u", "v", "T", "S"):
        assert np.isfinite(getattr(st, name)).all(), name
    assert len(seen) == 20 and all(d is not None and not d.eos.linear for d in seen)
    kappa = cl.tracer_diffusivity()
    assert kappa is not None and np.isfinite(kappa).all() and kappa.max() > 1e-5
