"""
TEOS-10 in the 3-D core, Task 5: the hybrid regrid follows the isopycnals of the potential density at p_ref
(σ₂ + 1000 for 2e7 Pa) with a nonlinear EOS, forced monotonic per column as MOM6 HYCOM1
(``coord_hycom.F90:214-219``); the linear EOS is unchanged.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError, PyOceanConfigWarning
from physics.eos.wright_full import density as wright_density

NX, NY, NZ = 8, 6, 10
DT = 60.0
H = (40.0 + 5.0 * np.arange(NX))[None, :] * np.ones((NY, 1))


def _model(eos: str, rho_target) -> SigmaBaroclinic:
    kw = dict(z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6, d_min=0.05, salinity=True, eos=eos,
              rho_ref="auto", vertical_coordinate="ale", pgf="fv", regrid="hybrid",
              regrid_params=dict(dz_top=np.full(NZ, H.max() / NZ), rho_target=rho_target))
    if eos == "linear":
        kw.update(s_coef=0.78, s_ref=35.0, t_ref=10.0)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        return SigmaBaroclinic(create_cartesian_grid(nx=NX, ny=NY, dx=2000.0, dy=2000.0, f0=1e-4), H, dt=DT,
                               n_layers=NZ, **kw)


def _front():
    k = np.arange(NZ)[:, None, None] / (NZ - 1)
    i = np.arange(NX)[None, None, :] / NX
    return ((10.0 + 4.0 * k + 1.0 * i) * np.ones((NZ, NY, NX)), (35.2 - 0.3 * k - 0.4 * i) * np.ones((NZ, NY, NX)))


def test_hybrid_uses_sigma2():
    m = _model("wright_full", np.linspace(1025.8, 1027.6, NZ + 1))
    T, S = _front()
    T[3, :, 4:] += 3.0                                              # cold-to-warm step, S front: not monotone
    rho = m._vertical.regrid_density(T, S)
    ref = np.fmin.accumulate(wright_density(T, S, 2e7), axis=0)
    assert np.array_equal(rho, ref)
    assert not np.allclose(rho, m._eos.density(T, S, 0.0))          # not the surface-referenced density


def test_monotonic_forcing():
    m = _model("teos10", np.linspace(1025.8, 1027.6, NZ + 1))
    T, S = _front()
    T[6] -= 6.0                                                     # injected inversion: dense layer above lighter ones
    raw = m._eos.potential_density(T, S, 2e7)
    assert (np.diff(raw, axis=0) > 0).any()                         # light layer (k+1) over denser (k) is stable...
    assert (np.diff(raw, axis=0) < 0).any()                         # ...and the injection makes an inversion
    rho = m._vertical.regrid_density(T, S)
    assert (np.diff(rho, axis=0) <= 0).all()                        # non-increasing upward = non-decreasing to the bed
    assert (rho <= raw).all() and np.array_equal(rho[0], raw[0])    # only lightened, bed layer untouched


def test_linear_regrid_density_unchanged():
    m = _model("linear", np.linspace(-0.5, 0.5, NZ + 1) + 1000.0)
    T, S = _front()
    T[6] -= 6.0
    assert np.array_equal(m._vertical.regrid_density(T, S), m._eos.density(T, S))


def test_rho_target_range_preflight():
    with pytest.raises(ConfigurationError, match="rho_target"):
        _model("wright_full", np.linspace(20.0, 30.0, NZ + 1))      # σ values, not ρ
    with pytest.raises(ConfigurationError, match="rho_target"):
        _model("teos10", np.linspace(1010.0, 1030.0, NZ + 1))
    _model("teos10", np.linspace(1025.0, 1037.0, NZ + 1))
    _model("linear", np.linspace(20.0, 30.0, NZ + 1))               # the linear EOS is not checked here
