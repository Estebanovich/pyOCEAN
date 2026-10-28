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


# ---------------------------------------------------------------------------------------------------------------
# Task 5: the tracer substep of the freshwater (S = 0 in the freshwater, T of the top layer; MOM6
# applyBoundaryFluxesInOut, MOM_diabatic_aux.F90:1013-1117).  S, S² and passive tracers keep their content
# (c ← c V_adv / V_fw), T and T² keep their value.
# ---------------------------------------------------------------------------------------------------------------
T_UNI, S_UNI = 12.0, 30.0


def _uniform(kind: str, ntr: int = 0, **kw):
    """Closed basin with uniform T = 12 °C and S = 30 g kg⁻¹ (no baroclinic motion; the η bump still moves)."""
    sal = dict(salinity=True, s_coef=0.78, s_ref=30.0)
    sal.update(kw)
    if kind == "sigma":
        m = nc._sigma(**sal)
    else:
        m = nc._ale("zstar", n_tracers=ntr, **sal) if ntr else nc._ale("zstar", **sal)
    shape = (m.core.n_layers, nc.NY, nc.NX)
    init = dict(S=np.full(shape, S_UNI))
    if ntr:
        init["tr"] = np.ones((ntr,) + shape)
    st = m.initial_state(nc._eta(), np.full(shape, T_UNI), **init)
    return m, st


@pytest.mark.parametrize("kind", ["sigma", "zstar", "zstar_tracer"])
def test_evaporation_conserves_salt_and_concentrates(kind):
    ntr = 1 if kind == "zstar_tracer" else 0
    m, st = _uniform("sigma" if kind == "sigma" else "zstar", ntr=ntr)
    fw = np.full((nc.NY, nc.NX), -1e-6)                                # uniform evaporation E = 1e-6 m s⁻¹
    salt0, V0 = m.tracer_content(st, tracer="S"), _volume(m, st)
    tr0 = m.tracer_content(st, tracer=0) if ntr else None
    for _ in range(N_STEPS):
        st = m.step(st, fw=fw)
        assert abs(m.tracer_content(st, tracer="S") - salt0) <= 1e-13 * salt0
        if ntr:
            assert abs(m.tracer_content(st, tracer=0) - tr0) <= 1e-13 * tr0
        act = m.fs.active
        np.testing.assert_allclose(st.T[:, act], T_UNI, rtol=0, atol=1e-12)     # freshwater leaves at the top T
    V = _volume(m, st)
    assert V < V0
    S_mean = m.tracer_content(st, tracer="S") / float(np.sum(m.layer_volumes(st.eta, st.h)[:, m.fs.active]))
    np.testing.assert_allclose(S_mean, S_UNI * V0 / V, rtol=1e-13)
    assert st.S[-1].min() > S_UNI                                      # the surface water concentrates
    assert st.n_limited == 0                                           # thick top layers: nothing spilled


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
def test_precipitation_freshens_only_where_it_rains(kind):
    m, st = _uniform(kind)
    fw = np.zeros((nc.NY, nc.NX))
    east = slice(2 * nc.NX // 3, None)
    fw[:, east] = 5e-5                                                 # P on the east third
    st1 = m.step(st, fw=fw)
    dS = st1.S - S_UNI
    assert np.abs(dS[..., : 2 * nc.NX // 3]).max() <= 1e-12            # no change where it does not rain
    assert (dS[-1, :, east] < -1e-6).all()                             # fresher surface where it rains
    np.testing.assert_allclose(st1.T, T_UNI, rtol=0, atol=1e-12)
    # the salt is conserved under the rain too
    salt0 = m.tracer_content(st, tracer="S")
    assert abs(m.tracer_content(st1, tracer="S") - salt0) <= 1e-13 * salt0


def test_evaporation_deficit_moves_to_next_layer_and_is_counted():
    nz = nc.NZ
    # 2 mm top layer (dz_top[0] = surface).  Not 1 mm: with a nominal top layer equal to the regrid floor
    # H_MIN = 1 mm, zstar_column pushes the surface interface above η when D < H (Σ h ≠ H + η by ~2e-6 m), and the
    # remap then extends the column with the top value — a separate regrid issue (task-5 report).
    dz = np.full(nz, (nc.HMAX - 2e-3) / (nz - 1))
    dz[0] = 2e-3
    sal = dict(salinity=True, s_coef=0.78, s_ref=30.0)
    m = nc._ale("zstar", params=dict(dz_top=dz), **sal)
    shape = (nz, nc.NY, nc.NX)
    S0 = 30.0 + 0.5 * np.arange(nz)[:, None, None] / nz * np.ones(shape)   # S varies with depth: content counts
    st = m.initial_state(nc._eta(), np.full(shape, T_UNI), S=S0)
    assert st.h[-1].max() < 2.2e-3
    fw = np.full((nc.NY, nc.NX), -5e-5)                                # 3 mm per step: more than the top layer
    salt0 = m.tracer_content(st, tracer="S")
    for _ in range(5):
        st = m.step(st, fw=fw)
        assert st.h.min() >= 0.0
        assert abs(m.tracer_content(st, tracer="S") - salt0) <= 1e-13 * salt0
    assert st.n_limited > 0


def test_apply_freshwater_unit():
    """Content of S and passive tracers kept, T unchanged, var_surf = (V_fw S_new² − V_adv S_old²)/Δt."""
    from core.components.freshwater import apply_freshwater, freshwater_volumes
    rng = np.random.default_rng(1)
    n, ny, nx = 4, 4, 5
    lay = {"T": 0, "T2": 1, "S": 2, "tr": 3, "n": 4}
    T = 10 + rng.random((n, ny, nx))
    S = 30 + rng.random((n, ny, nx))
    stack = np.stack([T, T**2, S, rng.random((n, ny, nx))])
    V_adv = 1.0 + rng.random((n, ny, nx))
    dV = np.zeros((ny, nx))
    dV[0, :] = 0.2                                                     # precipitation
    dV[1, :] = -0.1                                                    # evaporation within the top layer
    dV[2, 0] = -(V_adv[-1, 2, 0] + 0.5)                                 # evaporation beyond the top layer
    floor = np.full((ny, nx), 1e-6)
    V_fw, n_lim = freshwater_volumes(V_adv, dV, floor)
    assert n_lim == 1
    np.testing.assert_allclose(V_fw.sum(0), V_adv.sum(0) + dV, rtol=1e-14)
    assert V_fw[-1, 2, 0] == pytest.approx(1e-6) and V_fw[-2, 2, 0] < V_adv[-2, 2, 0]
    out, n2, var_surf = apply_freshwater(stack.copy(), lay, V_adv, V_fw, dV, dt=10.0)
    assert n2 == 1
    np.testing.assert_array_equal(out[0], T)
    np.testing.assert_array_equal(out[1], T**2)
    np.testing.assert_allclose((V_fw * out[2]).sum(), (V_adv * S).sum(), rtol=1e-14)
    np.testing.assert_allclose((V_fw * out[3]).sum(), (V_adv * stack[3]).sum(), rtol=1e-14)
    np.testing.assert_allclose(var_surf, (V_fw * out[2] ** 2 - V_adv * S**2) / 10.0, rtol=1e-14, atol=1e-14)
    assert np.all(var_surf[:, 3:, :] == 0)                              # no freshwater: no term
