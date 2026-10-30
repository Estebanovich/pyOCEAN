"""
KPP in the 3-D model, ``closure='kpp'`` in σ and ALE (N3 task 7; spec docs/superpowers/specs/2026-10-29-n3-turbulence-
closures-design.md item 4 and decisions 6–7; ``core.components.closure.KPPClosure``, ``core.model.driver`` phase 7).

* the model's κ_s, ν and γ after one step equal ``kpp_column`` on the same inputs (rebuilt here independently: layer
  thicknesses, centre velocities, buoyancy, B_f from Q_ns and the absorbed shortwave, u* from τ) to 1e-12;
* the non-local term only redistributes: the column heat content changes by Δt A (Q_ns + Q_sw)/(ρ₀ c_p) to 1e-12,
  also with vanished layers and γ at the bed interface (Task 6 carry);
* dry and land columns are skipped (Review Focus 2); restarts are bit for bit and a restart of another closure is
  rejected (Review Focus 3); a fixed z* column equals σ to 1e-10; the variance budget closes with the non-local term
  folded into phy_v.

Index order (k, j, i), k = 0 at the bed; interface arrays (N+1, …), 0 = bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.components.closure import KPPClosure
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError, PyOceanConfigWarning
from core.restart import read_restart, write_restart
from physics.mixing.kpp_cvmix import KPPParams, kpp_column
from physics.surface.shortwave import layer_heating

NZ, H = 20, 50.0
RHO0, ALPHA, CP, G = 1000.0, 0.2, 3985.0, 9.81
TX, TY = np.full((2, 3), 0.1), np.full((3, 2), 0.05)                 # wind stress [N m⁻²]
QNS, QSW = -200.0, 80.0                                              # cooling + penetrating shortwave [W m⁻²]


def _column(ale: str | None, nz: int = NZ, **kw) -> SigmaBaroclinic:
    """Doubly periodic 2×2 column with closure 'kpp', σ (``ale`` None) or ALE with rule ``ale``."""
    hg = create_cartesian_grid(nx=2, ny=2, dx=1000.0, dy=1000.0, f0=1e-4)
    base = dict(rho0=RHO0, t_coef=ALPHA, closure="kpp", theta=0.6, d_min=0.05, periodic=("x", "y"), g=G)
    if ale is not None:
        base.update(vertical_coordinate="ale", regrid=ale, pgf="fv", regrid_params=dict(dz_top=np.full(nz, H / nz)))
    base.update(kw)
    return SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=nz, **base)


def _T(nz: int = NZ, shape=(2, 2)) -> np.ndarray:
    z = (np.arange(nz) + 0.5) * H / nz                                # height above the bed [m]
    return (10.0 + 0.1 * z)[:, None, None] * np.ones((nz,) + shape)   # N² ≈ 2e-4 s⁻²


def _heat(shape=(2, 2), qsw: float | None = QSW) -> dict:
    out = {"Q_ns": np.full(shape, QNS)}
    if qsw is not None:
        out["Q_sw"] = np.full(shape, qsw)
    return out


def _run(m: SigmaBaroclinic, steps: int, st=None, heat=None, **kw):
    st = m.initial_state(np.zeros((2, 2)), _T()) if st is None else st
    for _ in range(steps):
        st = m.step(st, TX, TY, heat=_heat() if heat is None else heat, **kw)
    return st


# ---------------------------------------------------------------- the model calls kpp_column with the right inputs
def test_kpp_column_model_matches_kpp_column():
    warnings.simplefilter("ignore")
    m = _column(None)
    st0 = _run(m, 30)                                                 # wind + cooling + shortwave: a mixed layer
    st = m.step(st0, TX, TY, heat=_heat())
    # the inputs of the last step, rebuilt independently: a horizontally uniform T is not changed by the advection
    hk = (m.layer_volumes(st.eta) / m.area[None]).reshape(NZ, -1)
    zw = np.concatenate([np.zeros((1, 4)), np.cumsum(hk, axis=0)])
    uc = (0.5 * (st.u[:, :, :-1] + st.u[:, :, 1:])).reshape(NZ, -1)
    vc = (0.5 * (st.v[:, :-1, :] + st.v[:, 1:, :])).reshape(NZ, -1)
    rho = RHO0 - ALPHA * st0.T                                        # linear EOS, t_ref = 0
    b = (-G * (rho - RHO0) / RHO0).reshape(NZ, -1)                    # −g(ρ − ρ₀)/ρ₀ (round-off as the model's)
    ustar = np.full(4, np.sqrt(np.hypot(0.1, 0.05) / RHO0))
    bns = np.full(4, G / RHO0 * ALPHA * QNS / (RHO0 * CP))
    bsw = np.full(4, G / RHO0 * ALPHA * QSW / (RHO0 * CP))
    _, frac = layer_heating(hk.reshape(NZ, 2, 2), np.full((2, 2), QNS), np.full((2, 2), QSW), "I")
    ref = kpp_column(hk, zw, uc, vc, b, bns, ustar, frac.reshape(NZ + 1, -1), None, KPPParams(), bflux_sw=bsw)
    cl = m._closure
    assert ref.obl_depth.min() > 2.0 and ref.nonlocal_.max() > 0.1   # a convective boundary layer with γ
    for got, want in ((cl.tracer_diffusivity(), ref.kappa_s), (st.closure["akv"], ref.kappa_m),
                      (cl.nonlocal_flux(), ref.nonlocal_)):
        want = want.copy()
        want[0] = want[-1] = 0.0                                      # no flux through the bed or the surface
        np.testing.assert_allclose(got.reshape(NZ + 1, -1), want, rtol=1e-12, atol=1e-12 * np.abs(want).max())
    np.testing.assert_allclose(cl.obl_depth, ref.obl_depth, rtol=1e-12)
    # and ν of this step is the viscosity of the next one (MOM6: Kv_KPP in the next dynamics step)
    nu_u, _ = cl.viscosity(st)
    np.testing.assert_allclose(nu_u[:, :, 1], st.closure["akv"].reshape(NZ + 1, 2, 2)[:, :, 0], rtol=1e-12)


# ---------------------------------------------------------------- the non-local term only redistributes
@pytest.mark.parametrize("ale", [None, "zstar"])
def test_kpp_nonlocal_conserves_heat(ale):
    warnings.simplefilter("ignore")
    m = _column(ale)
    st = m.initial_state(np.zeros((2, 2)), _T())
    for _ in range(30):
        V0 = m.layer_volumes(st.eta, st.h)
        c0 = (V0 * st.T).sum()
        st = m.step(st, TX, TY, heat=_heat())
        c1 = (m.layer_volumes(st.eta, st.h) * st.T).sum()
        dq = m.dt * m.area.sum() * (QNS + QSW) / (RHO0 * CP)
        assert abs((c1 - c0) - dq) <= 1e-12 * abs(c0), (c1 - c0, dq)
    assert m._closure.nonlocal_flux().max() > 0.1 and np.all(m._closure.F_T < 0)


def test_kpp_nonlocal_tendency_vanished_layers_and_bed():
    # unit level: γ with a value at the bed interface (h reaching the bed, Task 6) and vanished layers anywhere
    cl = KPPClosure(6, (1, 3))
    rng = np.random.default_rng(1)
    g = rng.uniform(0.0, 1.0, (7, 3))
    g[-1] = 0.0
    h = np.array([[0.0, 2.0, 0.0], [0.0, 1.0, 3.0], [1.0, 0.0, 0.0], [2.0, 0.0, 1.0], [0.0, 0.0, 2.0],
                  [1.0, 3.0, 0.0]])
    cl.gamma = g
    cl.gamma[0] = 0.0                                                 # what ``update`` enforces
    dT = cl.nonlocal_tendency(h.reshape(6, 1, 3), np.array([-1e-5, 2e-5, -3e-5]), 60.0).reshape(6, 3)
    assert np.isfinite(dT).all() and (dT[h == 0.0] == 0.0).all()
    content = (h * dT).sum(axis=0)
    np.testing.assert_allclose(content, 0.0, atol=1e-14 * np.abs(h * dT).sum())


def test_kpp_update_zeroes_bed_and_surface_interfaces(monkeypatch):
    # when h reaches the bed CVMix writes the enhanced K (and γ) at the bed interface (Task 6 carry): the closure
    # must not turn it into a flux through the bed.  kpp_column is replaced by one that fills every interface.
    import core.components.closure as closure_mod
    real = closure_mod.kpp_column

    def full(*a, **kw):
        r = real(*a, **kw)
        for x in (r.kappa_m, r.kappa_s, r.nonlocal_):
            x[:] = 0.5
        return r
    monkeypatch.setattr(closure_mod, "kpp_column", full)
    cl = KPPClosure(4, (1, 1))
    cl.update(np.zeros((4, 1, 1)), np.zeros((4, 1, 1)), np.full((4, 1, 1), 1.0), np.full((4, 1, 1), 1000.0),
              np.full((1, 1), 1e-4), np.zeros((1, 1)),
              bflux={"Q_ns": np.full((1, 1), -500.0), "Q_sw": None, "sw_frac_iface": None, "fw": None,
                     "S_sfc": None})
    for x in (cl.akv, cl.akt, cl.gamma):
        assert x[0, 0] == 0.0 and x[-1, 0] == 0.0 and (x[1:-1, 0] == 0.5).all()
    dT = cl.nonlocal_tendency(np.full((4, 1, 1), 1.0), np.array([-1e-5]), 60.0)
    assert abs(dT.sum()) < 1e-20 and dT[0, 0, 0] != 0.0               # conserved, the bed layer still gets its share


# ---------------------------------------------------------------- dry and land columns (Review Focus 2)
@pytest.mark.parametrize("ale", [None, "zstar"])
def test_kpp_dry_column_is_skipped(ale):
    warnings.simplefilter("ignore")
    hg = create_cartesian_grid(nx=4, ny=1, dx=1000.0, dy=1000.0, f0=1e-4)
    bathy = np.array([[H, H, 0.02, np.nan]])                          # wet, wet, dry (D < d_min), land
    kw = dict(rho0=RHO0, t_coef=ALPHA, closure="kpp", theta=0.6, d_min=0.05)
    if ale is not None:
        kw.update(vertical_coordinate="ale", regrid=ale, pgf="fv", regrid_params=dict(dz_top=np.full(NZ, H / NZ)))
    m = SigmaBaroclinic(hg, bathy, dt=60.0, n_layers=NZ, **kw)
    st = m.initial_state(np.zeros((1, 4)), _T(shape=(1, 4)))
    heat = _heat(shape=(1, 4))
    for _ in range(5):
        st = m.step(st, np.full((1, 5), 0.1), np.zeros((2, 4)), heat=heat)
    cl = m._closure
    for x in (cl.tracer_diffusivity(), cl.nonlocal_flux(), st.closure["akv"].reshape(NZ + 1, 1, 4)):
        assert np.isfinite(x).all()
        assert (x[:, 0, 2:] == 0.0).all()                             # dry and land: no K, no γ
    assert cl.tracer_diffusivity()[:, 0, :2].max() > 0.0 and cl.obl_depth[2:].tolist() == [0.0, 0.0]
    assert np.isfinite(st.T[:, 0, :3]).all() and np.isfinite(st.u).all()


# ---------------------------------------------------------------- restart (Review Focus 3)
@pytest.mark.parametrize("ale", [None, "zstar"])
def test_kpp_restart_bit_for_bit_and_wrong_closure_rejected(tmp_path, ale):
    warnings.simplefilter("ignore")
    st = _run(_column(ale), 8)
    m1 = _column(ale)
    st1 = _run(m1, 4)
    write_restart(tmp_path / "rst.nc", m1, st1)
    m2 = _column(ale)
    st2 = _run(m2, 4, read_restart(tmp_path / "rst.nc", m2))
    for name in ("eta", "u", "v", "T"):
        assert np.array_equal(getattr(st2, name), getattr(st, name)), name
    assert np.array_equal(st2.closure["akv"], st.closure["akv"])
    assert st.closure["akv"].max() > 1e-5
    # a KPP restart cannot start a GLS model, nor a GLS restart a KPP model; KPP parameters must match
    with pytest.raises(ValueError, match="closure='kpp' but the model uses closure='gls'"):
        read_restart(tmp_path / "rst.nc", _column(ale, closure="gls"))
    mg = _column(ale, closure="gls")
    write_restart(tmp_path / "gls.nc", mg, _run(mg, 1))
    with pytest.raises(ValueError, match="closure='gls' but the model uses closure='kpp'"):
        read_restart(tmp_path / "gls.nc", _column(ale))
    with pytest.raises(ValueError, match="kpp_interp_type"):
        read_restart(tmp_path / "rst.nc", _column(ale, kpp_params=KPPParams(interp_type="cubic")))


# ---------------------------------------------------------------- σ ≡ ALE z* with fixed layers
def test_kpp_sigma_vs_ale_zstar_column():
    warnings.simplefilter("ignore")
    sig = _run(_column(None), 100)
    ale = _run(_column("zstar"), 100)
    assert np.max(np.abs(ale.h - H / NZ)) < 1e-12                   # the z* layers did not move
    assert sig.closure["akv"].max() > 1e-3
    np.testing.assert_allclose(ale.closure["akv"], sig.closure["akv"], rtol=1e-10,
                               atol=1e-10 * sig.closure["akv"].max())
    np.testing.assert_allclose(ale.T, sig.T, rtol=1e-10)
    np.testing.assert_allclose(ale.u, sig.u, rtol=0, atol=1e-10 * np.max(np.abs(sig.u)))


def test_kpp_mixes_differently_from_constant_closure():
    warnings.simplefilter("ignore")
    kpp = _run(_column("zstar"), 50)
    const = _run(_column("zstar", closure="constant"), 50)
    assert np.max(np.abs(kpp.T - const.T)) > 1e-2


def test_kpp_hybrid_column_with_massless_layers():
    # hybrid column with two water masses and 12 density targets (most layers massless, h = 0 at the KPP call):
    # finite, no RuntimeWarning (two vanished layers at the Ri crossing, Task 6 carry), heat content exact
    nz = 12
    m0 = _column("zstar", nz=nz)
    rho_l, rho_d = m0._eos.density(np.array(14.0)), m0._eos.density(np.array(10.0))
    params = dict(dz_top=np.full(nz, 0.01), rho_target=np.linspace(rho_l - 1.0, rho_d + 1.0, nz + 1))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", UserWarning)                  # the hybrid floor warning
        m = _column("hybrid", nz=nz, regrid_params=params)
    T = np.where(np.arange(nz) < nz // 2, 10.0, 14.0)[:, None, None] * np.ones((nz, 2, 2))
    st = m.initial_state(np.zeros((2, 2)), T)
    assert (st.h < 1e-3).sum() >= 4 * 4
    with warnings.catch_warnings():
        warnings.simplefilter("error", RuntimeWarning)
        for _ in range(60):
            c0 = (m.layer_volumes(st.eta, st.h) * st.T).sum()
            st = m.step(st, TX, TY, heat=_heat())
            c1 = (m.layer_volumes(st.eta, st.h) * st.T).sum()
            dq = m.dt * m.area.sum() * (QNS + QSW) / (RHO0 * CP)
            assert abs((c1 - c0) - dq) <= 1e-12 * abs(c0)
    assert np.isfinite(st.T).all() and np.isfinite(st.u).all() and np.isfinite(st.closure["akv"]).all()
    assert (st.h < 1e-3).any() and m._closure.obl_depth.min() > 0.0


# ---------------------------------------------------------------- salinity, freshwater and the variance budget
@pytest.mark.parametrize("ale", [None, "zstar"])
def test_kpp_variance_budget_closes_with_freshwater(ale):
    # the non-local term is folded into {T,S}_phy_v; the S non-local flux is the dilution −S₁ (P − E)
    warnings.simplefilter("ignore")
    m = _column(ale, salinity=True, s_coef=0.78, s_ref=35.0, variance_budget=True)
    S = (35.0 - 0.02 * np.arange(NZ))[:, None, None] * np.ones((NZ, 2, 2))
    st = m.initial_state(np.zeros((2, 2)), _T(), S=S)
    fw = np.full((2, 2), -1e-6)                                       # evaporation: destabilising salt flux
    for _ in range(5):
        V0 = m.layer_volumes(st.eta, st.h)
        T0, S0 = st.T, st.S
        st = m.step(st, TX, TY, fw=fw, heat=_heat())
        V1 = m.layer_volumes(st.eta, st.h)
        b = st.budget
        for c, x0, x1 in (("T", T0, st.T), ("S", S0, st.S)):
            lhs = (V1 * x1**2).sum() - (V0 * x0**2).sum()
            rhs = m.dt * (b[f"{c}_surf"].sum() - b[f"{c}_num_adv"].sum() - b[f"{c}_phy_v"].sum()
                          - b[f"{c}_phy_h"].sum() - b.get(f"{c}_num_remap", np.zeros(1)).sum()
                          - b[f"{c}_sponge"].sum())
            assert abs(lhs - rhs) <= 5e-10 * (V0 * x0**2).sum(), (c, lhs, rhs)
        assert abs((V1 * st.S).sum() - (V0 * S0).sum()) <= 1e-12 * (V0 * S0).sum()   # salt content kept
    cl = m._closure
    assert np.all(cl.F_S > 0) and cl.nonlocal_flux().max() > 0.05     # S₁ (E) > 0: salt flux into the ocean


# ---------------------------------------------------------------- pre-flight
def test_kpp_preflight():
    with pytest.raises(ConfigurationError, match="kpp_params must be"):
        _column(None, kpp_params={"ri_crit": 0.3})
    with pytest.warns(PyOceanConfigWarning, match="kpp_params is ignored"):
        _column(None, closure="gls", kpp_params=KPPParams())
    with pytest.warns(PyOceanConfigWarning, match="kv=1e-05 is ignored with closure='kpp'"):
        _column("zstar", kv=1e-5)
    assert isinstance(_column("hybrid", regrid_params=dict(dz_top=np.full(NZ, H / NZ),
                                                          rho_target=np.linspace(996.0, 1000.0, NZ + 1)))._closure,
                      KPPClosure)


# ---------------------------------------------------------------- final-review fix I1 (thin layers)
def _probe8_column(thin: bool, stale_b: float = -1e-4):
    """Final review I1 / probe8: 100 m, 20 × 5 m layers, a 60 m mixed layer over a thermocline, u* = 0.01 m s⁻¹,
    B_f = −1e-8 m² s⁻³; with ``thin`` two 1e-10 m layers of buoyancy ``stale_b`` (Δρ ≈ 0.01 kg m⁻³) at ~20 m."""
    nt, dz, k = 20, 5.0, 16
    zc = (np.arange(nt) + 0.5) * dz
    b = np.where(zc < 40.0, -1e-3 * (40.0 - zc) / 40.0, 0.0)
    h = np.full(nt, dz)
    if thin:
        h = np.concatenate([h[:k], [1e-10, 1e-10], h[k:]])
        b = np.concatenate([b[:k], [stale_b, stale_b], b[k:]])
    n = h.size
    cl = KPPClosure(n, (1, 1), rho0=RHO0, g=G, t_coef=ALPHA, cp=CP)
    rho = (RHO0 - RHO0 * b / G)[:, None, None]
    Q_ns = np.full((1, 1), 1e-8 * RHO0 * RHO0 * CP / (G * ALPHA))   # B_ns = g α Q_ns/(ρ₀² c_p) = +1e-8 (destab.: −)
    bflux = {"Q_ns": -Q_ns, "Q_sw": None, "sw_frac_iface": None, "fw": None, "S_sfc": None}
    z = np.zeros((n, 1, 1))
    cl.update(z, z, h[:, None, None], rho, np.full((1, 1), 1e-4), np.zeros((1, 1)), bflux)
    return cl


@pytest.mark.parametrize("stale_b", [-1e-4, -3e-4, -1e-3])
def test_kpp_vanished_layer_does_not_set_the_boundary_layer(stale_b):
    """Two 1e-10 m layers of slightly denser water inside the mixed layer must not cut the boundary-layer depth
    (61 → 17.5 m before the fix) or κ_s (4×): thin layers are merged before the column call, as for GLS."""
    ref = _probe8_column(False)
    cl = _probe8_column(True, stale_b)
    assert ref.obl_depth[0] > 55.0
    assert abs(cl.obl_depth[0] - ref.obl_depth[0]) <= 0.03 * ref.obl_depth[0], (cl.obl_depth, ref.obl_depth)
    assert abs(cl.akt.max() - ref.akt.max()) <= 0.03 * ref.akt.max()
    k_thin = np.r_[0:16, 18:23]                                       # interfaces 16 and 18 bound the thin pair
    np.testing.assert_allclose(cl.akt[k_thin, 0], ref.akt[:, 0], rtol=0.03, atol=1e-12)
    assert np.isfinite(cl.akt).all() and np.isfinite(cl.gamma).all() and np.isfinite(cl.akv).all()


def test_kpp_merge_all_thin_or_single_merged_layer_is_finite():
    """A shallow column whose layers are all at the 1 mm threshold goes through unmerged; a column of one thick and
    several thin layers merges to one layer: finite, zero at the bed and the surface (final-review fix I1)."""
    for h in (np.full(20, 1e-3), np.r_[np.full(5, 1e-10), 2.0, np.full(4, 1e-10)]):
        n = h.size
        cl = KPPClosure(n, (1, 1), rho0=RHO0, g=G, t_coef=ALPHA, cp=CP)
        z = np.zeros((n, 1, 1))
        bflux = {"Q_ns": np.full((1, 1), -200.0), "Q_sw": None, "sw_frac_iface": None, "fw": None, "S_sfc": None}
        cl.update(z, z, h[:, None, None], np.full((n, 1, 1), RHO0), np.full((1, 1), 1e-4), np.zeros((1, 1)), bflux)
        for x in (cl.akv, cl.akt, cl.gamma):
            assert np.isfinite(x).all() and x[0, 0] == 0.0 and x[-1, 0] == 0.0
