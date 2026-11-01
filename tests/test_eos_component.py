"""EOS components, T/S conversions and freezing point (TEOS-10 in the 3-D core, task 2)."""
import gsw
import numpy as np
import pytest

from core.components.eos import EOS, LinearEOS, TEOS10EOS, WrightFullEOS, make_eos
from physics.eos import convert as cv


def test_linear_density_unchanged():
    e = LinearEOS(rho0=1025.0, t_coef=0.17, t_ref=10.0, s_coef=0.78, s_ref=35.0)
    T = np.linspace(2, 28, 7)
    S = np.linspace(34, 37, 7)
    assert np.array_equal(e.density(T, S), 1025.0 - 0.17 * (T - 10.0) + 0.78 * (S - 35.0))
    assert np.array_equal(e.density(T), 1025.0 - 0.17 * (T - 10.0))
    assert np.array_equal(e.density(T, S, p=np.full(7, 1e7)), e.density(T, S))


def test_linear_derivs():
    e = LinearEOS(t_coef=0.17, s_coef=0.78)
    dT, dS = e.density_derivs(np.zeros((2, 3)), np.zeros((2, 3)), np.zeros((2, 3)))
    assert dT.shape == dS.shape == (2, 3)
    assert np.all(dT == -0.17) and np.all(dS == 0.78)


def test_make_eos_names():
    for n, cls in (("linear", LinearEOS), ("wright_full", WrightFullEOS), ("teos10", TEOS10EOS)):
        e = make_eos(n, 1030.0)
        assert isinstance(e, cls) and e.name == n and e.rho0 == 1030.0 and isinstance(e, EOS)
    assert make_eos("linear", 1025.0, t_coef=0.3).t_coef == 0.3
    assert make_eos("teos10").ts_kind == "ct_sa" and make_eos("wright_full").ts_kind == "theta_sp"
    with pytest.raises(ValueError):
        make_eos("unesco")


def test_potential_density_sigma2():
    e = WrightFullEOS()
    assert e.potential_density(10.0, 35.0, 2e7) == e.density(10.0, 35.0, 2e7)
    assert e.density_anomaly(10.0, 35.0, 2e7, 1035.0) == pytest.approx(e.density(10.0, 35.0, 2e7) - 1035.0, abs=1e-9)


def test_roundtrip_ts():
    T = np.linspace(2, 28, 40)
    S = np.linspace(34, 37, 40)
    Tm, Sm = cv.to_model_ts(T, S, "ct_sa", "theta_sp")
    assert not np.allclose(Tm, T)
    T2, S2 = cv.from_model_ts(Tm, Sm, "ct_sa")
    assert np.max(abs(T2 - T)) < 1e-12 and np.max(abs(S2 - S)) < 1e-12
    Tb, Sb = cv.to_model_ts(Tm, Sm, "theta_sp", "ct_sa")
    assert np.max(abs(Tb - T)) < 1e-12 and np.max(abs(Sb - S)) < 1e-12
    assert cv.to_model_ts(T, S, "theta_sp", "theta_sp")[0] is T
    with pytest.raises(ValueError):
        cv.to_model_ts(T, S, "x", "theta_sp")


def test_sp_sa_zero():
    assert cv.sp_to_sa(0.0) == 0.0 and cv.sa_to_sp(0.0) == 0.0
    assert cv.sa_to_sp(cv.sp_to_sa(35.0)) == pytest.approx(35.0, abs=1e-13)
    assert cv.pressure_from_z(-100.0, 1035.0, 9.8) == pytest.approx(9.8 * 1035.0 * 100.0)


def test_freezing_points():
    SA = np.array([0.0, 35.16504, 38.0])
    p = np.array([0.0, 1e7, 5e7])
    assert np.array_equal(cv.freezing_teos10(SA, p), gsw.CT_freezing(SA, p * 1e-4, 0.0))
    S = np.array([0.0, 35.0, 20.0])
    assert cv.freezing_linear(S, p)[1] == (0.0 + -0.054 * 35.0) + -7.75e-8 * 1e7
    assert cv.freezing_linear(0.0, 0.0) == 0.0
    assert cv.freezing_linear(20.0, 2e7) == pytest.approx(-1.08 - 1.55)
    assert make_eos("teos10").freezing_point(SA, p)[1] < make_eos("wright_full").freezing_point(SA, p)[1] + 0.1
