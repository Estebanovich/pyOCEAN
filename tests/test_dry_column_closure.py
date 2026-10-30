"""
Final-review fix M1 (N3): a dry column (H + η ≤ d_min, also H ≤ 0) is left still by the turbulence closure — its
T is unchanged bit for bit over 10 steps with closure 'gls' and 'kpp' (σ and ALE-σ; z* cannot have H ≤ 0)
and the constant closure with kv > 0 (Review Focus 2 of
docs/superpowers/plans/2026-10-29-n3-turbulence-closures.md).  Before the fix the GLS κ_t floor (1e-6 m² s⁻¹)
homogenised the 2.5 mm layers of a dry column (≈ 2 °C in 10 steps).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid

NZ, H = 20, 50.0
BATHY = np.array([[H, H, 3.0, 0.02, -0.5]])                          # wet, wet, shallow, dry (D < d_min), dry bank


def _run(closure: str, ale: str | None, steps: int = 10):
    hg = create_cartesian_grid(nx=5, ny=1, dx=1000.0, dy=1000.0, f0=1e-4)
    kw = dict(rho0=1000.0, t_coef=0.2, closure=closure, theta=0.6, d_min=0.05)
    if closure == "gls":
        kw["z0"] = 1e-3
    if closure == "constant":
        kw.update(kv=1e-4, nu_const=1e-4)                               # a kv floor mixes like the GLS κ_t floor
    if ale is not None:
        kw.update(vertical_coordinate="ale", regrid=ale, pgf="fv",
                  regrid_params=dict(dz_top=np.full(NZ, H / NZ)) if ale == "zstar" else {})
        if closure == "gls":
            kw["hbbl"] = 1.0
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SigmaBaroclinic(hg, BATHY, dt=60.0, n_layers=NZ, **kw)
    z = (np.arange(NZ) + 0.5) * H / NZ
    T = (10.0 + 0.1 * z)[:, None, None] * np.ones((NZ, 1, 5))
    st = m.initial_state(np.maximum(0.0, -BATHY), T)
    T0 = st.T.copy()
    tx, ty = np.full((1, 6), 0.1), np.zeros((2, 5))
    for _ in range(steps):
        st = m.step(st, tx, ty)
    return st, T0


@pytest.mark.parametrize("closure,ale", [("gls", None), ("gls", "sigma"), ("constant", None),
                                         ("kpp", None), ("kpp", "sigma")])
def test_dry_column_tracers_left_still(closure, ale):
    st, T0 = _run(closure, ale)
    assert np.array_equal(st.T[:, :, 3:], T0[:, :, 3:]), np.abs(st.T[:, :, 3:] - T0[:, :, 3:]).max()
    assert np.isfinite(st.T).all()
    assert np.abs(st.T[:, :, :2] - T0[:, :, :2]).max() > 0.0                 # the wet columns are mixed
