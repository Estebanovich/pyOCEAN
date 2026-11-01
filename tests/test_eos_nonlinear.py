"""
Nonlinear equations of state of the 3-D core (TEOS-10 in the 3-D core, task 1):
Wright (1997) full range (``physics.eos.wright_full``) and Roquet et al. (2015)
density polynomial (``physics.eos.roquet_rho``) against compiled MOM6 (git 0e9a561).

The reference ``tests/data/mom6_eos_reference.npz`` is written by
``validation/mom6_column/build_reference.sh`` (``eos_driver`` + ``eos_columns.py``),
so these tests need no Fortran build.
"""

from __future__ import annotations

from pathlib import Path

import gsw
import numpy as np
import pytest

from physics.eos import roquet_rho, wright_full

REF = np.load(Path(__file__).parent / "data" / "mom6_eos_reference.npz")
T, S, P, RHO_REF = REF["T"], REF["S"], REF["p"], float(REF["rho_ref"])
MODULES = {"w": wright_full, "r": roquet_rho}

# Fit error of the Roquet et al. (2015) density polynomial against TEOS-10 (gsw.rho, the
# 75-term polynomial) over the oceanic cloud S >= 2: 1.1 x the measured maximum
# (ruling of the task-1 ledger).  Consistent with Roquet et al. (2015) Table 2: polyTEOS10-bsq
# against the exact Gibbs function is 1.83 +- 27.45e-3 kg m-3 over the whole cube
# (S_A 0-42, CT -2..40 C, 0-10000 dbar), 0.08 +- 0.32e-3 in the oceanic funnel.
ROQUET_FIT_MAX = 1.1 * 5.355e-2   # kg m-3; measured max 5.3546e-2 (test docstring)


@pytest.mark.parametrize("tag", ["w", "r"])
def test_coefficients_match_compiled(tag):
    """Every module-level coefficient equals the value the compiler folds, bit for bit."""
    mod = MODULES[tag]
    names = REF["coef_names"]
    vals = REF["coef_values"]
    own = [(n, v) for n, v in zip(names, vals) if hasattr(mod, str(n))]
    expected = 15 if tag == "w" else 132
    assert len(own) == expected
    bad = [(str(n), getattr(mod, str(n)), v) for n, v in own if getattr(mod, str(n)) != v]
    assert not bad, bad[:5]


def _check(tag):
    mod = MODULES[tag]
    rho = mod.density(T, S, P)
    anom = mod.density_anomaly(T, S, P, RHO_REF)
    dT, dS = mod.density_derivs(T, S, P)
    np.testing.assert_allclose(rho, REF[f"{tag}_rho"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(anom, REF[f"{tag}_rho_anom"], rtol=0, atol=1e-12)
    np.testing.assert_allclose(dT, REF[f"{tag}_drho_dT"], rtol=1e-12, atol=0)
    np.testing.assert_allclose(dS, REF[f"{tag}_drho_dS"], rtol=1e-12, atol=0)


def test_wright_full_matches_mom6():
    """WRIGHT_FULL: ρ, ρ − ρ_ref (anomaly form), ∂ρ/∂θ, ∂ρ/∂S as compiled MOM6."""
    _check("w")


def test_roquet_matches_mom6():
    """ROQUET_RHO: ρ, ρ − ρ_ref (anomaly form), ∂ρ/∂Θ, ∂ρ/∂S_A as compiled MOM6."""
    _check("r")


@pytest.mark.parametrize("tag", ["w", "r"])
def test_numba_equals_numpy(tag):
    """The Numba kernels equal the NumPy oracles bit for bit (also with a scalar p)."""
    mod = MODULES[tag]
    assert np.array_equal(mod.density(T, S, P), mod.density_numpy(T, S, P))
    assert np.array_equal(mod.density_anomaly(T, S, P, RHO_REF),
                          mod.density_anomaly_numpy(T, S, P, RHO_REF))
    for a, b in zip(mod.density_derivs(T, S, P), mod.density_derivs_numpy(T, S, P)):
        assert np.array_equal(a, b)
    # 2-D T, S with a scalar pressure
    T2, S2 = T[:2040].reshape(40, 51), S[:2040].reshape(40, 51)
    assert np.array_equal(mod.density(T2, S2, 2.0e7), mod.density_numpy(T2, S2, 2.0e7))
    assert mod.density(T2, S2, 2.0e7).shape == (40, 51)
    # scalar elementwise kernels used by later tasks
    i = 17
    assert mod.rho_anom_elem(T[i], S[i], P[i], RHO_REF) == mod.density_anomaly_numpy(T, S, P, RHO_REF)[i]
    dT, dS = mod.drho_elem(T[i], S[i], P[i])
    dTn, dSn = mod.density_derivs_numpy(T, S, P)
    assert dT == dTn[i] and dS == dSn[i]


def test_roquet_vs_gsw_fit_error():
    """Roquet's 52 + 6-term density polynomial against TEOS-10 (``gsw.rho``) over the
    oceanic cloud S_A ≥ 2 g/kg (T = Θ, p in dbar = p/1e4).

    Measured 2026-10-31 on the 2048-point cloud (2026 points with S ≥ 2): maximum
    |ρ_roquet − gsw.rho| = 5.3546e-2 kg m⁻³ (Θ 27.7 °C, S_A 3.0 g/kg, 5964 dbar), RMS
    7.6e-3, median 3.2e-4 kg m⁻³; 3.0e-2 for 30 < S_A < 40.  The largest differences are
    brackish water at abyssal pressure, outside the oceanic funnel that both polynomials
    (Roquet's 52 + 6 terms, GSW's 75-term specific volume) were fitted on."""
    m = S >= 2.0
    err = np.abs(roquet_rho.density(T[m], S[m], P[m]) - gsw.rho(S[m], T[m], P[m] / 1e4))
    assert err.max() <= ROQUET_FIT_MAX


@pytest.mark.parametrize("tag", ["w", "r"])
def test_fresh_water_finite(tag):
    """Review focus 1: S = 0 (rivers), T 0…30 °C, p 0…1e7 Pa gives finite ρ, anomaly
    and derivatives (Roquet's √(S + 32) offset; Wright is regular at S = 0)."""
    mod = MODULES[tag]
    Tg, Pg = np.meshgrid(np.linspace(0.0, 30.0, 31), np.linspace(0.0, 1.0e7, 11))
    Sg = np.zeros_like(Tg)
    outs = [mod.density(Tg, Sg, Pg), mod.density_anomaly(Tg, Sg, Pg, RHO_REF),
            *mod.density_derivs(Tg, Sg, Pg)]
    for o in outs:
        assert np.all(np.isfinite(o))
    rho = outs[0]
    assert 990.0 < rho.min() and rho.max() < 1010.0
