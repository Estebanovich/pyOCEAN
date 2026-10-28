"""
N2a (plan docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md): real freshwater flux.

Task 4: the surface freshwater flux fw = P − E [m s⁻¹, positive into the ocean] is a real volume source in the
continuity equation of the free surface (Huang 1993; MITgcm useRealFreshWaterFlux, integr_continuity.F:93,185 and
solve_for_pressure.F:147-156),

    A (ηⁿ⁺¹ − ηⁿ) / Δt = −∇·(D ū)^θ + A fw,

masked to wet active cells (H + ηⁿ > d_min).  In ALE the top layer of the Lagrangian thickness absorbs fw Δt so
Σ_k h = H + η.  The tracer substep (S = 0 in the freshwater) is task 5; these tests check the volume only.
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.model.contract import ConfigurationError
from tests import n1_cases as nc

N_STEPS = 20


def _model(kind: str, H: np.ndarray | None = None):
    """Closed basin of the golden matrix (σ or ALE z*) with salinity, S₀ = 30, no wind."""
    kw = dict(salinity=True, s_coef=0.78, s_ref=30.0)
    if H is not None:
        from core.dynamics.sigma_baroclinic import SigmaBaroclinic
        base = dict(rho0=nc.RHO0, t_coef=nc.TCOEF, z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6,
                    d_min=0.05, **kw)
        m = SigmaBaroclinic(nc._grid(), H, dt=nc.DT, n_layers=nc.NZ, **base)
        eta = np.maximum(0.0, -np.nan_to_num(H))                       # at rest; dry banks at zero depth
        st = m.initial_state(eta, nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 30.0))
        return m, st
    m = nc._sigma(**kw) if kind == "sigma" else nc._ale("zstar", **kw)
    st = m.initial_state(nc._eta(), nc._T(), S=np.full((nc.NZ, nc.NY, nc.NX), 30.0))
    return m, st


def _volume(m, st) -> float:
    """Σ A (H + η) over the active cells [m³]."""
    act = m.fs.active
    D = np.maximum(m.H + st.eta, 0.0)
    return float(np.sum((m.area * D)[act]))


def _halves() -> np.ndarray:
    """fw = +2e-6 m s⁻¹ (P) on the west half, −2e-6 (E) on the east half."""
    fw = np.full((nc.NY, nc.NX), -2e-6)
    fw[:, : nc.NX // 2] = 2e-6
    return fw


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
@pytest.mark.parametrize("pattern", ["halves", "rain"])
def test_freshwater_changes_volume_exactly(kind, pattern):
    m, st = _model(kind)
    fw = _halves() if pattern == "halves" else np.full((nc.NY, nc.NX), 2e-6)
    act = m.fs.active
    source = m.dt * float(np.sum((m.area * fw)[act]))               # Δt Σ A fw per step [m³]
    V0 = _volume(m, st)
    for n in range(1, N_STEPS + 1):
        st = m.step(st, fw=fw)
        V = _volume(m, st)
        assert abs(V - (V0 + n * source)) <= 1e-13 * V0, (n, V - V0, n * source)
        if m.ale:
            colsum = st.h.sum(axis=0)
            np.testing.assert_allclose(colsum[act], (m.H + st.eta)[act], rtol=0, atol=1e-12)
    # the source acts where it falls: the west half rises relative to a run without fw
    m2, st2 = _model(kind)
    for _ in range(N_STEPS):
        st2 = m2.step(st2)
    deta = st.eta - st2.eta
    if pattern == "halves":
        assert deta[:, : nc.NX // 2].mean() > 0 > deta[:, nc.NX // 2:].mean()
    else:
        assert abs(_volume(m, st) - _volume(m2, st2) - N_STEPS * source) <= 1e-13 * V0


def test_fw_on_land_and_dry_cells_is_ignored():
    H = np.array(nc.H, float, copy=True)
    H[0, 0] = np.nan                                                   # land
    H[-1, -1] = -1.0                                                   # dry bank 1 m above the sea level
    m, st = _model("sigma", H=H)
    fw = np.zeros((nc.NY, nc.NX))
    fw[0, 0] = 5e-4
    fw[-1, -1] = 5e-4
    V0 = _volume(m, st)
    a, b = st, st
    for _ in range(5):
        a = m.step(a, fw=fw)
        b = m.step(b)
    assert abs(_volume(m, a) - V0) <= 1e-13 * V0
    assert np.array_equal(np.nan_to_num(a.eta, nan=-9.0), np.nan_to_num(b.eta, nan=-9.0))


def test_fw_needs_salinity():
    m = nc._sigma()
    st = m.initial_state(nc._eta(), nc._T())
    with pytest.raises(ConfigurationError, match="fw needs salinity=True"):
        m.step(st, fw=np.zeros((nc.NY, nc.NX)))


def test_fw_shape_is_checked():
    m, st = _model("zstar")
    with pytest.raises(ConfigurationError, match=r"fw must have shape \(6, 10\)"):
        m.step(st, fw=np.zeros((nc.NY, nc.NX + 1)))
