"""
N3 task 2 (plan docs/superpowers/plans/2026-10-29-n3-turbulence-closures.md): prescribed surface heat flux with
penetrating shortwave (Paulson & Simpson 1977 two-band Jerlov profile), ``physics/surface/shortwave.py`` and
``SigmaBaroclinic.step(..., heat={"Q_ns", "Q_sw"})``.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.model.contract import ConfigurationError
from physics.surface.shortwave import JERLOV, layer_heating, swfrac
from tests import n1_cases as nc

CP = 3985.0
SAL = dict(salinity=True, s_coef=0.78, s_ref=30.0)


def _heat(qns: float = -100.0, qsw: float = 200.0) -> dict:
    return {"Q_ns": np.full((nc.NY, nc.NX), qns), "Q_sw": np.full((nc.NY, nc.NX), qsw)}


T_REF = float(np.mean(nc._T()))


def _content(m, st) -> float:
    """ρ₀ c_p Σ (T − T_ref) V over the active cells [J] (T_ref only shrinks the round-off of the sum; the volume is
    conserved in the closed basin, so the change is that of Σ T V)."""
    V = m.layer_volumes(st.eta, st.h)
    act = np.broadcast_to(m.fs.active[None], V.shape)
    return float(m.rho0 * CP * np.sum(((st.T - T_REF) * V)[act]))


def _model(kind: str, **kw):
    m = nc._sigma(**kw) if kind == "sigma" else nc._ale("zstar", **kw)
    return m, m.initial_state(nc._eta(), nc._T())


def test_swfrac_type_I_values():
    assert swfrac(0.0) == 1.0
    assert swfrac(-1e4) < 1e-12
    ref = 0.58 * np.exp(-10 / 0.35) + 0.42 * np.exp(-10 / 23.0)
    assert abs(swfrac(-10.0) - ref) <= 1e-14
    for t in JERLOV:                                                   # monotone from 1 to 0 for every type
        f = swfrac(-np.linspace(0, 200, 50), t)
        assert f[0] == 1.0 and (np.diff(f) < 0).all() and f[-1] < 1e-3
    with pytest.raises(ValueError, match="unknown Jerlov"):
        swfrac(-1.0, "IV")


def test_layer_heating_sums_to_the_flux():
    h = np.full((5, 2, 3), 1.7)
    q, f = layer_heating(h, np.full((2, 3), -30.0), np.full((2, 3), 150.0))
    assert np.allclose(q.sum(0), 120.0, rtol=0, atol=1e-12)
    assert f.shape == (6, 2, 3) and (f[-1] == 1.0).all()
    assert (q[:-1] >= 0).all()                                          # shortwave only heats
    q0, f0 = layer_heating(h, np.full((2, 3), 10.0), None)              # no shortwave: fractions still (N+1, ny, nx)
    assert f0.shape == (6, 2, 3) and np.allclose(q0.sum(0), 10.0)


def test_vanished_layers_get_no_heat_and_nothing_is_lost():
    h = np.full((6, 1, 2), 3.0)
    h[4:, 0, 0] = 0.0                                                   # top two layers vanished in column 0
    h[0, 0, 1] = 0.0                                                    # bed layer vanished in column 1
    q, f = layer_heating(h, np.full((1, 2), -20.0), np.full((1, 2), 80.0))
    assert np.allclose(q.sum(0), 60.0, rtol=0, atol=1e-12)
    assert (q[4:, 0, 0] == 0).all() and q[3, 0, 0] != 0 and q[0, 0, 1] == 0


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
def test_heat_content_change_is_exact(kind):
    m, st = _model(kind)
    n, heat = 8, _heat(-1e5, 1e6)           # large enough that the ~100 J round-off drift of the sums is < 1e-12
    c0 = _content(m, st)
    for _ in range(n):
        st = m.step(st, heat=heat)
    area = float(np.sum(m.area[m.fs.active]))
    expected = (-1e5 + 1e6) * area * n * m.dt
    assert abs((_content(m, st) - c0) - expected) <= 1e-12 * abs(expected)


def test_heat_content_exact_with_vanished_top_layers():
    """Hybrid ALE: layers whose target density is absent are (nearly) massless; the flux must still all enter."""
    m = nc._ale("hybrid", nz=8)
    st = m.initial_state(nc._eta(), nc._T(8))
    assert (st.h < 1e-2).any()                                          # the hybrid grid has near-massless layers
    heat = _heat(-1e5, 1e6)
    c0 = _content(m, st)
    for _ in range(6):
        st = m.step(st, heat=heat)
        assert m._driver.last_bflux["sw_frac_iface"].shape == (9, nc.NY, nc.NX)
    area = float(np.sum(m.area[m.fs.active]))
    expected = 9e5 * area * 6 * m.dt
    assert abs((_content(m, st) - c0) - expected) <= 1e-12 * expected


def test_heat_zero_thickness_top_layer_direct():
    """Driver level: force h = 0 in the top layer of a wet column through layer_heating (see the unit test above)."""
    h = np.zeros((4, 1, 1)); h[:2] = 5.0
    q, _ = layer_heating(h, np.full((1, 1), 100.0), None)
    assert q[1, 0, 0] == 100.0 and q[2:].sum() == 0


def test_nonsolar_only_and_bottom_deposit():
    """Q_ns alone heats the top layer; the bed remainder Q_sw f_0 is spread over the column in proportion to h (MOM6
    absorbRemainingSW; it used to stay in layer 0, CROCO, final-review fix C1)."""
    m, st = _model("sigma")
    T0 = st.T.copy()
    a = m.step(st, heat={"Q_ns": np.full((nc.NY, nc.NX), 50.0)})
    dT = a.T - T0
    assert dT[-1].min() > 0 and np.abs(dT[:-1]).max() < 0.2 * dT[-1].min()      # top layer heated, rest advected only
    q, f = layer_heating(np.full((nc.NZ, 1, 1), 20.0 / nc.NZ), np.zeros((1, 1)), np.ones((1, 1)))
    band = f[1:, 0, 0] - f[:-1, 0, 0]
    np.testing.assert_allclose(q[:, 0, 0], band + f[0, 0, 0] / nc.NZ, rtol=1e-13)  # equal layers: f_0 / N each


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
def test_heat_on_land_and_dry_cells_is_ignored(kind):
    H = np.array(nc.H, float, copy=True)
    H[0, 0] = np.nan                                                    # land
    H[-1, -1] = -1.0                                                    # dry bank 1 m above the sea level
    base = dict(rho0=nc.RHO0, t_coef=nc.TCOEF, z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6,
                d_min=0.05)
    if kind == "zstar":
        base.update(vertical_coordinate="ale", regrid="sigma", regrid_params={}, pgf="fv")
    m = SigmaBaroclinic(nc._grid(), H, dt=nc.DT, n_layers=nc.NZ, **base)
    eta = np.maximum(0.0, -np.nan_to_num(H))
    st = m.initial_state(eta, nc._T())
    heat = {"Q_ns": np.zeros((nc.NY, nc.NX)), "Q_sw": np.zeros((nc.NY, nc.NX))}
    heat["Q_ns"][0, 0] = heat["Q_ns"][-1, -1] = 5e3
    heat["Q_sw"][0, 0] = heat["Q_sw"][-1, -1] = 5e3
    a, b = st, st
    for _ in range(5):
        a = m.step(a, heat=heat)
        b = m.step(b)
    for f in ("eta", "u", "v", "T"):
        assert np.array_equal(np.nan_to_num(getattr(a, f), nan=-9.0), np.nan_to_num(getattr(b, f), nan=-9.0)), f


def test_ctx_bflux_for_the_closure():
    m, st = _model("sigma")
    m.step(st)
    assert m._driver.last_bflux is None
    m.step(st, heat=_heat())
    bf = m._driver.last_bflux
    assert set(bf) == {"Q_ns", "Q_sw", "sw_frac_iface"} and bf["sw_frac_iface"].shape == (nc.NZ + 1, nc.NY, nc.NX)
    assert (bf["sw_frac_iface"][-1] == 1.0).all() and (bf["Q_sw"] == 200.0).all()


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
def test_variance_budget_closes_with_heat(kind):
    kw = dict(SAL, kh=50.0, kv=1e-3, variance_budget=True)
    m = nc._sigma(**kw) if kind == "sigma" else nc._ale("zstar", **kw)
    k = np.arange(nc.NZ)[:, None, None]
    S = 30.0 - 0.4 * k / (nc.NZ - 1) + 0 * nc._T()
    st = m.initial_state(nc._eta(), nc._T(), S=S)
    heat = _heat()
    for _ in range(6):
        V0 = m.layer_volumes(st.eta, st.h)
        T0 = st.T
        st = m.step(st, heat=heat)
        V1 = m.layer_volumes(st.eta, st.h)
        b = st.budget
        lhs = (V1 * st.T**2).sum() - (V0 * T0**2).sum()
        rhs = m.dt * (b["T_surf"].sum() - b["T_num_adv"].sum() - b["T_phy_v"].sum() - b["T_phy_h"].sum()
                      - b.get("T_num_remap", np.zeros(1)).sum() - b["T_sponge"].sum())
        assert abs(lhs - rhs) <= 5e-10 * (V0 * T0**2).sum(), (lhs, rhs)
        assert np.abs(b["T_surf"]).sum() > 0


def test_heat_validation():
    m, st = _model("sigma")
    with pytest.raises(ConfigurationError, match="heat must be a dict"):
        m.step(st, heat={"Q_sw": np.zeros((nc.NY, nc.NX))})
    with pytest.raises(ConfigurationError, match=r"heat\['Q_ns'\] must have shape"):
        m.step(st, heat={"Q_ns": np.zeros((2, 2))})
    with pytest.raises(ConfigurationError, match="unknown jerlov"):
        nc._sigma(jerlov="V")
    with pytest.raises(ConfigurationError, match="cp="):
        nc._sigma(cp=0.0)


@pytest.mark.parametrize("name", ["sigma_kh_kv", "zstar_vi_arakawa"])
def test_no_heat_is_bitwise(name):
    import json
    from pathlib import Path
    golden = json.loads((Path(__file__).resolve().parent / "data" / "n1_golden" / "hashes.json").read_text())
    got = nc.run_case(name)
    for step, ref in golden[name].items():
        assert got[int(step)] == ref


# ---------------------------------------------------------------- final-review fix C1 (MOM6 absorbRemainingSW)
def _sw_column(closure: str, ale: bool, kv: float = 0.0, steps: int = 240):
    """Doubly periodic 2×2, D = 10 m, 20 layers; in z* (dz_top = 20 × 2.5 m) the bed layers collapse to the regrid
    minimum (1 mm).  T = 10 °C, Q_sw = 200 W m⁻² only, Δt = 60 s (final review C1, probe7)."""
    from core.grid.horizontal import create_cartesian_grid
    nz = 20
    hg = create_cartesian_grid(nx=2, ny=2, dx=1000.0, dy=1000.0, f0=1e-4)
    kw = dict(rho0=1000.0, t_coef=0.2, closure=closure, theta=0.6, d_min=0.05, periodic=("x", "y"))
    if closure != "kpp":
        kw.update(kv=kv, nu_const=1e-4)
    if ale:
        kw.update(vertical_coordinate="ale", regrid="zstar", pgf="fv", regrid_params=dict(dz_top=np.full(nz, 2.5)))
    m = SigmaBaroclinic(hg, np.full((2, 2), 10.0), dt=60.0, n_layers=nz, **kw)
    st = m.initial_state(np.zeros((2, 2)), np.full((nz, 2, 2), 10.0))
    heat = {"Q_ns": np.zeros((2, 2)), "Q_sw": np.full((2, 2), 200.0)}
    for _ in range(steps):
        st = m.step(st, np.zeros((2, 3)), np.zeros((3, 2)), heat=heat)
    return m, st


@pytest.mark.parametrize("closure", ["kpp", "constant"])
def test_thin_bed_layer_does_not_take_the_bed_shortwave(closure):
    """The shortwave left at the bed is spread over the column (MOM6 absorbRemainingSW), so a 1 mm z* bed layer warms
    like its neighbours: no layer warmer than the σ column (whose layers are all D/N thick) by more than 0.05 °C (the
    bug gave 206.5 °C with KPP, σ about 10.4 °C)."""
    _, sig = _sw_column(closure, False)
    m, zs = _sw_column(closure, True)
    assert zs.h[0].max() < 2e-3                                         # the bed layer did collapse
    assert zs.T.max() <= sig.T.max() + 0.05, (zs.T.max(), sig.T.max())
    assert zs.T[0].max() <= zs.T[1:].max()
    V = m.layer_volumes(zs.eta, zs.h)                                   # heat content still exact (from 10 °C)
    gain = m.rho0 * m.cp * np.sum((zs.T - 10.0) * V)
    expected = 200.0 * float(np.sum(m.area)) * 240 * m.dt
    assert abs(gain - expected) <= 1e-10 * expected


def test_bed_remainder_is_spread_over_the_column():
    """Unit rule: q_k = Q_sw (f_{k+1} − f_k) + Q_sw f_0 w_k / Σ w, w_k = h_k (h ≥ 2 h_min_heat), 2 h_k − 2 h_min_heat
    (h_min_heat < h < 2 h_min_heat), 0 below (MOM6 MOM_opacity.F90:813-816, 846-860)."""
    from physics.surface.shortwave import H_MIN_HEAT
    h = np.full((5, 1, 1), 2.0)
    h[0] = 1e-3                                                         # 1 mm bed layer
    h[1] = 1.5 * H_MIN_HEAT                                             # partial weight
    h[2] = 0.5 * H_MIN_HEAT                                             # no heat at all
    q, f = layer_heating(h, np.zeros((1, 1)), np.full((1, 1), 100.0))
    assert abs(q.sum() - 100.0) <= 1e-12
    w = np.array([1e-3, 2 * 1.5 * H_MIN_HEAT - 2 * H_MIN_HEAT, 0.0, 2.0, 2.0])
    band = 100.0 * (f[1:, 0, 0] - f[:-1, 0, 0])
    band_thin = band[2]                                                 # absorbed in the no-heat layer: to the pool
    rem = 100.0 * f[0, 0, 0] + band_thin
    expect = np.where(np.arange(5) == 2, 0.0, band) + rem * w / w.sum()
    np.testing.assert_allclose(q[:, 0, 0], expect, rtol=1e-12, atol=1e-15)
    assert q[2, 0, 0] == 0.0
    # heating rate of the 1 mm bed layer = that of the thick layers' remainder share (no bed spike)
    assert q[0, 0, 0] / h[0, 0, 0] < 2 * q[3, 0, 0] / h[3, 0, 0]


def test_thin_top_layer_and_forcing_depth():
    """Q_ns: no heat to a layer at or below h_min_heat (= the vertical solvers' DZ_FLOOR); thinner than the forcing
    depth (1 mm, MOM6 MINIMUM_FORCING_DEPTH) a layer takes the fraction h/1 mm of what is left and passes the rest
    down (MOM6 MOM_diabatic_aux.F90:1101-1118)."""
    from core.vertical.ale import DZ_FLOOR
    from physics.surface.shortwave import FORCING_DEPTH, H_MIN_HEAT
    assert H_MIN_HEAT >= DZ_FLOOR and FORCING_DEPTH == 1e-3
    h = np.full((4, 1, 1), 3.0)
    h[-1] = 0.5e-6                                                      # top layer thinner than DZ_FLOOR
    q, _ = layer_heating(h, np.full((1, 1), -100.0), None)
    assert q[-1, 0, 0] == 0.0 and q[-2, 0, 0] == -100.0
    h[-1] = 0.25e-3                                                     # a quarter of the forcing depth
    h[-2] = 0.5e-3                                                      # half of it
    q, _ = layer_heating(h, np.full((1, 1), -100.0), None)
    np.testing.assert_allclose(q[:, 0, 0], [0.0, -100 * 0.75 * 0.5, -100 * 0.75 * 0.5, -25.0], rtol=1e-14)
