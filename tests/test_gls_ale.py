"""
The stratified GLS k-ε closure in the ALE mode (N3 task 5; spec docs/superpowers/specs/2026-10-29-n3-turbulence-
closures-design.md §3 "GLS k-ε estratificada en ALE" and decisions 2–5; ``core.components.closure.GLSClosure``).

* a z* column whose layers do not move (doubly periodic 2×2, wind, initial N²) gives the σ GLS: k, ε, ν, κ_t after
  200 steps to rtol 1e-10 (the same ``GLSKEpsilon`` code on the same inputs; measured ≈ 5e-13);
* the remap of the interface state after a regrid conserves Σ w k (w the dual cells ½(h_k−1 + h_k)) to 1e-12 and
  keeps k ≥ 1e-10, ε ≥ 1e-12 (Review Focus 2);
* vanished (1e-10 m) layers between 5 m layers create no spike: ν, κ_t, k equal those of the column without them
  (MOM6 eliminate_massless merging; Review Focus 1); a hybrid column with massless layers stays finite;
* restarts of an ALE GLS run are bit for bit (Review Focus 3);
* ``gls_production='energy'`` (Burchard 2002 eq. 16): the kinetic energy the vertical viscosity removes in one step
  equals the integrated P that enters k, to 1e-10;
* Lorenz experiment A in ALE-σ with the GLS: 50 steps, finite, the s² budget closes step by step to ≤ 5e-10.

Index order (k, j, i), k = 0 at the bed; interface arrays (N+1, …), 0 = bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.components.closure import GLSClosure, dual_thickness
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.restart import read_restart, write_restart
from core.vertical import remap as rm

NZ, H = 20, 50.0
TX, TY = np.full((2, 3), 0.1), np.full((3, 2), 0.05)                 # uniform wind stress [N m⁻²]


def _column(ale: str | None, nz: int = NZ, f0: float = 1e-4, **kw) -> SigmaBaroclinic:
    """Doubly periodic 2×2 column, closure 'gls' (stratified), σ (``ale`` None) or ALE with rule ``ale``."""
    hg = create_cartesian_grid(nx=2, ny=2, dx=1000.0, dy=1000.0, f0=f0)
    base = dict(rho0=1000.0, t_coef=0.2, closure="gls", theta=0.6, d_min=0.05, periodic=("x", "y"))
    if ale is not None:
        base.update(vertical_coordinate="ale", regrid=ale, pgf="fv",
                    regrid_params=dict(dz_top=np.full(nz, H / nz)))
    base.update(kw)
    return SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=nz, **base)


def _start(m: SigmaBaroclinic, nz: int = NZ):
    z = (np.arange(nz) + 0.5) * H / nz                                # height above the bed [m]
    T = (10.0 + 0.1 * z)[:, None, None] * np.ones((nz, 2, 2))         # N² ≈ 2e-4 s⁻² (α = 0.2 kg m⁻³ K⁻¹)
    return m.initial_state(np.zeros((2, 2)), T)


def _run(m: SigmaBaroclinic, steps: int, st=None, tx=TX, ty=TY):
    st = _start(m) if st is None else st
    for _ in range(steps):
        st = m.step(st, tx, ty)
    return st


# ---------------------------------------------------------------- σ ≡ ALE with fixed layers
@pytest.mark.parametrize("production", ["croco", "energy"])
def test_gls_ale_column_matches_sigma(production):
    warnings.simplefilter("ignore")
    sig = _run(_column(None, gls_production=production), 200)
    ale = _run(_column("zstar", gls_production=production), 200)
    assert np.max(np.abs(ale.h - H / NZ)) < 1e-12                   # the z* layers did not move
    assert sig.closure["akv"].max() > 1e-3                           # a wind-mixed layer formed
    for k in ("tke", "eps", "akv", "akt"):
        ref = sig.closure[k]
        np.testing.assert_allclose(ale.closure[k], ref, rtol=1e-10, atol=1e-10 * np.max(np.abs(ref)), err_msg=k)
    np.testing.assert_allclose(ale.T, sig.T, rtol=1e-12)
    np.testing.assert_allclose(ale.u, sig.u, rtol=0, atol=1e-10 * np.max(np.abs(sig.u)))


def test_gls_ale_differs_from_constant_closure():
    # the closure acts: the T profile mixed by the GLS differs from the constant-ν/kv run by far more than 1e-10
    warnings.simplefilter("ignore")
    gls = _run(_column("zstar"), 100)
    const = _run(_column("zstar", closure="constant", kv=1e-5), 100)
    assert np.max(np.abs(gls.T - const.T)) > 1e-3


# ---------------------------------------------------------------- remap (Review Focus 2)
def _closure_with_state(n: int, shape=(3, 4), seed: int = 0) -> GLSClosure:
    rng = np.random.default_rng(seed)
    cl = GLSClosure(n, shape, 60.0, stratified=True)
    cl.bind_ale(rm.PPM_H4)
    m = shape[0] * shape[1]
    g = cl.gls
    g.tke = 10.0 ** rng.uniform(-10.0, -3.0, (n + 1, m))              # k ≥ tke_min = 1e-10, spiky
    g.eps = 10.0 ** rng.uniform(-12.0, -5.0, (n + 1, m))
    g.akv = 10.0 ** rng.uniform(-5.0, -2.0, (n + 1, m))
    g.akt = 10.0 ** rng.uniform(-6.0, -2.0, (n + 1, m))
    cl._n2 = rng.uniform(0.0, 1e-3, (n + 1, m))
    return cl


def _random_layers(rng, n: int, shape, total: np.ndarray, vanished: float = 0.3) -> np.ndarray:
    h = rng.uniform(0.0, 1.0, (n,) + shape)
    h[rng.uniform(size=h.shape) < vanished] = 0.0                     # vanished layers in source and target
    h[n // 2] += 0.1                                                  # never an empty column
    return h * (total / h.sum(axis=0))[None]


def test_gls_remap_conserves_column_tke():
    n, shape = 15, (3, 4)
    rng = np.random.default_rng(1)
    total = rng.uniform(5.0, 200.0, shape)
    for trial in range(5):
        cl = _closure_with_state(n, shape, seed=trial)
        hs, hd = _random_layers(rng, n, shape, total), _random_layers(rng, n, shape, total)
        ws, wd = dual_thickness(hs).reshape(n + 1, -1), dual_thickness(hd).reshape(n + 1, -1)
        before = (ws * cl.gls.tke).sum(axis=0)
        cl.remap(hs, hd)
        after = (wd * cl.gls.tke).sum(axis=0)
        np.testing.assert_allclose(after, before, rtol=1e-12)
        assert cl.gls.tke.min() >= 1e-10 and cl.gls.eps.min() >= 1e-12
        assert cl.gls.akv.min() >= cl.gls.nu_min and cl.gls.akt.min() >= cl.gls.kt_min
        assert np.isfinite(cl.gls.eps).all()


def test_gls_remap_identity_and_galperin():
    # an unchanged grid is not touched; after a real remap ε satisfies the Galperin limit with the remapped N²
    n, shape = 10, (2, 2)
    cl = _closure_with_state(n, shape)
    h = np.full((n,) + shape, 3.0)
    k0 = cl.gls.tke.copy()
    cl.remap(h, h.copy())
    assert np.array_equal(cl.gls.tke, k0)
    rng = np.random.default_rng(3)
    n2 = np.full((n + 1, 4), 1e-3)
    cl._n2 = n2
    cl.remap(h, _random_layers(rng, n, shape, np.full(shape, 3.0 * n), vanished=0.0))
    g = cl.gls
    k = g.tke[1:-1]
    lim = g.cm0**3 * k**1.5 / (g.galp * np.sqrt(2.0 * k) / np.sqrt(1e-3))
    assert np.all(g.eps[1:-1] >= lim * (1 - 1e-12))


# ---------------------------------------------------------------- vanished layers (Review Focus 1)
def _closure_column(n: int) -> GLSClosure:
    cl = GLSClosure(n, (1, 1), 60.0, stratified=True, rho0=1000.0)
    cl.bind_ale(rm.PPM_H4)
    return cl


def test_gls_vanished_layer_no_spike():
    n_thick, dz = 10, 5.0
    zc = (np.arange(n_thick) + 0.5) * dz
    u_t = 0.3 * np.tanh((zc - 30.0) / 5.0)                            # sheared, stably stratified column
    rho_t = 1025.0 - 0.02 * zc
    # the same column with a 1e-10 m layer between every pair of 5 m layers, carrying wild values
    n_v = 2 * n_thick - 1
    thick = np.arange(n_v) % 2 == 0
    h_v = np.where(thick, dz, 1e-10)
    u_v = np.where(thick, np.repeat(u_t, 2)[:n_v], 5.0 * (-1) ** np.arange(n_v))
    rho_v = np.where(thick, np.repeat(rho_t, 2)[:n_v], 1000.0 + 30.0 * (np.arange(n_v) % 4))
    a, b = _closure_column(n_thick), _closure_column(n_v)
    us2s, us2b = np.array([[1e-4]]), np.array([[2e-5]])
    for _ in range(60):
        a.update(u_t[:, None, None], 0 * u_t[:, None, None], np.full((n_thick, 1, 1), dz), rho_t[:, None, None],
                 us2s, us2b)
        b.update(u_v[:, None, None], 0 * u_v[:, None, None], h_v[:, None, None], rho_v[:, None, None], us2s, us2b)
    for name in ("akv", "akt", "tke", "eps"):
        va, vb = getattr(a.gls, name), getattr(b.gls, name)
        assert np.isfinite(vb).all(), name
        assert vb.max() <= 1.01 * va.max(), (name, vb.max(), va.max())
        # every interface of the vanished column sits on one of the 5 m column: same value
        idx = np.concatenate([[0], np.repeat(np.arange(1, n_thick), 2), [n_thick]])
        np.testing.assert_allclose(vb[:, 0], va[idx, 0], rtol=1e-6, err_msg=name)   # 1e-10 m merged
    assert a.gls.akv.max() > 1e-4                                     # turbulent, not at the floor


def test_gls_vanished_bed_and_top_layers():
    # z*-like massless bed layers and hybrid-like massless top layers: the bed distance is the first layer with mass
    n_thick, dz = 8, 5.0
    zc = (np.arange(n_thick) + 0.5) * dz
    u_t, rho_t = 0.2 * zc / zc[-1], 1025.0 - 0.01 * zc
    pad_b, pad_t = 3, 2
    n = n_thick + pad_b + pad_t
    h = np.concatenate([np.zeros(pad_b), np.full(n_thick, dz), np.full(pad_t, 1e-12)])
    u = np.concatenate([np.full(pad_b, 9.0), u_t, np.full(pad_t, -9.0)])
    rho = np.concatenate([np.full(pad_b, 1010.0), rho_t, np.full(pad_t, 1040.0)])
    a, b = _closure_column(n_thick), _closure_column(n)
    for _ in range(40):
        a.update(u_t[:, None, None], 0 * u_t[:, None, None], np.full((n_thick, 1, 1), dz), rho_t[:, None, None],
                 np.array([[5e-5]]), np.array([[1e-5]]))
        b.update(u[:, None, None], 0 * u[:, None, None], h[:, None, None], rho[:, None, None],
                 np.array([[5e-5]]), np.array([[1e-5]]))
    idx = np.concatenate([np.zeros(pad_b + 1, int), np.arange(1, n_thick), np.full(pad_t + 1, n_thick)])
    for name in ("akv", "akt", "tke", "eps"):
        np.testing.assert_allclose(getattr(b.gls, name)[:, 0], getattr(a.gls, name)[idx, 0], rtol=1e-8,
                                   err_msg=name)


@pytest.mark.parametrize("h_thin", [5e-3, 1e-2, 2e-2])
def test_gls_thin_layers_5mm_to_2cm_no_spike(h_thin):
    """
    The task-5 review probe (N3 task 9c): two adjacent thin layers of ``h_thin`` (5 mm–2 cm) between layers 4 and 5
    of the 10 × 5 m sheared, stratified column, with stale velocities ±2 cm/s and densities ∓0.05 kg m⁻³ off the
    mean of their neighbours.  With the 1 mm threshold they gave 25× ν and 100× k; the column-relative threshold
    (``thin_layer_threshold``: 1 % of D/N = 4.2 cm here) merges them, so ν and k stay ≤ 1.5× those of the column
    without them.
    """
    n_thick, dz = 10, 5.0
    zc = (np.arange(n_thick) + 0.5) * dz
    u_t = 0.3 * np.tanh((zc - 30.0) / 5.0)
    rho_t = 1025.0 - 0.02 * zc
    u_m, rho_m = 0.5 * (u_t[4] + u_t[5]), 0.5 * (rho_t[4] + rho_t[5])
    h = np.concatenate([np.full(5, dz), [h_thin, h_thin], np.full(5, dz)])
    u = np.concatenate([u_t[:5], [u_m + 0.02, u_m - 0.02], u_t[5:]])
    rho = np.concatenate([rho_t[:5], [rho_m - 0.05, rho_m + 0.05], rho_t[5:]])
    a, b = _closure_column(n_thick), _closure_column(n_thick + 2)
    us2s, us2b = np.array([[1e-4]]), np.array([[2e-5]])
    for _ in range(60):
        a.update(u_t[:, None, None], 0 * u_t[:, None, None], np.full((n_thick, 1, 1), dz), rho_t[:, None, None],
                 us2s, us2b)
        b.update(u[:, None, None], 0 * u[:, None, None], h[:, None, None], rho[:, None, None], us2s, us2b)
    for name in ("akv", "tke"):
        va, vb = getattr(a.gls, name), getattr(b.gls, name)
        assert np.isfinite(vb).all(), name
        assert vb.max() <= 1.5 * va.max(), (name, h_thin, vb.max() / va.max())
    assert a.gls.akv.max() > 1e-4                                     # turbulent, not at the floor


def test_gls_regrid_minimum_layers_merged_at_the_floor():
    """
    N3 task 9c fix 1 (review I1): a shallow z* column (5 m, 50 levels of 1 m) has 45 bed layers collapsed to the
    regrid minimum 1 mm, which come out at 1 mm + 3e-16 m (1.000000000000334 mm); the threshold sits at its 1 mm
    floor (1 % of D/N = 1 mm).
    A layer at the threshold is thin (MOM6 opens a cluster only for h > dz_massless, MOM_kappa_shear.F90:262), so
    the stack is merged and ν equals (to 5 %) that of the column with only the 5 thick layers.
    """
    from core.components.closure import thin_layer_threshold
    from core.vertical.regrid import zstar_column

    n, depth = 50, 5.0
    zw = np.empty(n + 1)
    zstar_column(depth, 0.0, np.full(n, 1.0), 1e-3, zw)
    h = np.diff(zw)
    thick = h > 0.5
    assert thick.sum() == 5 and (h[~thick] > 1e-3).all()             # round-off above the regrid minimum
    assert thin_layer_threshold(h[:, None])[0] == pytest.approx(1e-3, rel=1e-6)
    zc = np.cumsum(h) - 0.5 * h
    u = 0.3 * np.tanh((zc - 2.5) / 1.0) + np.where(thick, 0.0, 0.01 * (-1) ** np.arange(n))
    rho = 1025.0 - 0.02 * zc
    a, b = _closure_column(5), _closure_column(n)
    us2s, us2b = np.array([[1e-4]]), np.array([[2e-5]])
    for _ in range(60):
        a.update(u[thick, None, None], 0 * u[thick, None, None], h[thick, None, None], rho[thick, None, None],
                 us2s, us2b)
        b.update(u[:, None, None], 0 * u[:, None, None], h[:, None, None], rho[:, None, None], us2s, us2b)
    # interfaces nb … n of the full column sit on the thick-only column (the merged stack adds its 45 mm and its
    # velocities to the bed layer, so equal to a few %; unmerged, ν above the stack was 4e-4 against 0.17)
    nb = n - 5
    np.testing.assert_allclose(b.gls.akv[nb:, 0], a.gls.akv[:, 0], rtol=0.05)
    assert a.gls.akv.max() > 1e-3


def test_gls_active_layers_top_cluster():
    # N3 task 9c: which layers open a merged layer (k = 0 at the bed); threshold 4 cm, h_massless 1 mm
    from core.components.closure import active_layers, thin_layer_threshold

    h = np.array([[0.0, 5.0, 5.0, 5.0],                              # vanished bed layer
                  [5.0, 0.02, 5.0, 5.0],                             # interior thin layer
                  [5.0, 5.0, 0.01, 1e-12],                           # thin top layers: 2 cm vs 2e-12 m together
                  [5.0, 5.0, 0.01, 1e-12]])
    thr = np.full(4, 0.04)
    act = active_layers(h, thr, 1e-3)
    np.testing.assert_array_equal(act[:, 0], [False, True, True, True])
    np.testing.assert_array_equal(act[:, 1], [True, False, True, True])
    np.testing.assert_array_equal(act[:, 2], [True, True, True, False])   # the 2 cm top cluster stays its own layer
    np.testing.assert_array_equal(act[:, 3], [True, True, False, False])  # 1e-12 m top layers join the layer below
    # threshold: 1 % of D/N between 1 mm and 10 cm
    d = np.array([[0.05, 1.0, 100.0]]).repeat(10, axis=0)
    np.testing.assert_allclose(thin_layer_threshold(d), [1e-3, 1e-2, 0.1])


HYB_NZ = 12


def _hybrid_column(**kw):
    """Periodic hybrid column: two water masses (10 and 14 °C) and 12 density targets, so most layers are
    massless (the first periodic + hybrid model test, N3 task 3 carry)."""
    nz = HYB_NZ
    m0 = _column("zstar", nz=nz)
    rho_l, rho_d = m0._eos.density(np.array(14.0)), m0._eos.density(np.array(10.0))
    params = dict(dz_top=np.full(nz, 0.01), rho_target=np.linspace(rho_l - 1.0, rho_d + 1.0, nz + 1))
    m = _column("hybrid", nz=nz, regrid_params=params, **kw)
    T = np.where(np.arange(nz) < nz // 2, 10.0, 14.0)[:, None, None] * np.ones((nz, 2, 2))
    return m, m.initial_state(np.zeros((2, 2)), T)


def test_gls_hybrid_column_with_massless_layers_stays_finite():
    # 150 steps of wind on the hybrid column: the merged GLS columns and the remap after each regrid
    warnings.simplefilter("ignore")
    m, st = _hybrid_column()
    h0 = st.h.copy()
    assert (st.h < 1e-3).sum() >= 4 * 4                              # massless layers in every column
    st = _run(m, 150, st)
    assert (st.h < 1e-3).any() and np.max(np.abs(st.h - h0)) > 1e-2   # still massless layers; the grid moved
    for k in ("tke", "eps", "akv", "akt"):
        assert np.isfinite(st.closure[k]).all(), k
    g = m._closure.gls
    assert g.tke.min() >= g.tke_min and g.eps.min() >= g.psi_min
    assert st.closure["akv"].max() < 1.0 and np.isfinite(st.u).all() and np.isfinite(st.T).all()
    assert 10.0 - 1e-9 <= st.T.min() and st.T.max() <= 14.0 + 1e-9
    # the 1 cm z* floor at the top stays a merged surface layer of its own (MOM6 top cluster, N3 task 9c): the wind
    # gets a turbulent surface ν; merged into the 25 m layer below it got the floor 1e-5 and 8 m/s in the top 4 cm
    assert st.closure["akv"][-1].min() > 1e-4 and np.abs(st.u).max() < 1.0


# ---------------------------------------------------------------- restart (Review Focus 3)
@pytest.mark.parametrize("rule", ["zstar", "hybrid"])
def test_gls_ale_restart_bit_for_bit(tmp_path, rule):
    # z*: the all-layers path; hybrid (massless layers, moving grid): merged GLS columns and the remap
    warnings.simplefilter("ignore")

    def make(**kw):
        return _hybrid_column(**kw) if rule == "hybrid" else (lambda m: (m, _start(m)))(_column(rule, **kw))
    m, st = make()
    st = _run(m, 8, st)
    m1, st1 = make()
    st1 = _run(m1, 4, st1)
    write_restart(tmp_path / "rst.nc", m1, st1)
    m2, _ = make()
    st2 = read_restart(tmp_path / "rst.nc", m2)
    st2 = _run(m2, 4, st2)
    for name in ("eta", "u", "v", "T", "h"):
        assert np.array_equal(getattr(st2, name), getattr(st, name)), name
    for k in ("tke", "eps", "akv", "akt"):
        assert np.array_equal(st2.closure[k], st.closure[k]), k
    with pytest.raises(ValueError, match="gls_production"):
        read_restart(tmp_path / "rst.nc", make(gls_production="energy")[0])


# ---------------------------------------------------------------- energy-conserving production
@pytest.mark.parametrize("ale", [None, "zstar"])
def test_gls_production_energy_closes_energy(ale):
    """
    One step with no external source (no wind, free-slip bed, f = 0, horizontally uniform): the kinetic energy per
    unit area lost, ½ Σ_k h_k (uⁿ² − uⁿ⁺¹²) at a u- and a v-face, equals Δt Σ_interfaces P ½(h_k−1 + h_k), the P
    that enters the k equation (Burchard 2002 eq. 16).
    """
    warnings.simplefilter("ignore")
    m = _column(ale, f0=0.0, gls_production="energy")
    st = _run(m, 30)                                                  # wind-driven shear and turbulence
    h = st.h[:, 0, 0] if ale else np.full(NZ, H / NZ)
    st1 = m.step(st, None, None)
    ke = lambda s: 0.5 * np.sum(h * (s.u[:, 0, 0] ** 2 + s.v[:, 0, 0] ** 2))   # noqa: E731
    loss = ke(st) - ke(st1)
    prod = m._closure.gls.last_prod.reshape(NZ - 1, 2, 2)[:, 0, 0]
    gain = m.dt * np.sum(prod * 0.5 * (h[:-1] + h[1:]))
    assert loss > 1e-6 * ke(st)                                       # the viscosity does work
    assert abs(gain - loss) <= 1e-10 * loss, (gain, loss)
    # 'croco' is not energy-conserving: P = ν S² with the new shear differs
    m_c = _column(ale, f0=0.0)
    st_c = _run(m_c, 30)
    st_c1 = m_c.step(st_c, None, None)
    prod_c = m_c._closure.gls.last_prod.reshape(NZ - 1, 2, 2)[:, 0, 0]
    loss_c = ke(st_c) - ke(st_c1)
    assert abs(m.dt * np.sum(prod_c * 0.5 * (h[:-1] + h[1:])) - loss_c) > 1e-6 * loss_c


# ---------------------------------------------------------------- Lorenz experiment A, ALE-σ + GLS
def test_gls_ale_lorenz_short():
    from validation.lorenz_estuary import case

    warnings.simplefilter("ignore")
    x = (np.arange(case.NX) + 0.5) * case.DX
    S0 = (33.0 + 3.0 * np.tanh((x - 50e3) / 2e3)) * np.ones((case.NZ, case.NY, 1))
    m, st, forcing = case.build("A", "ale_sigma", S_init=S0, closure="gls", z0=case.Z0, kv=0.0,
                                gls_stratified=True, bottom_drag=None, hbbl=0.1)
    reg = m.tef_region_budget.mask
    worst = 0.0
    for n in range(50):
        c0 = (m.layer_volumes(st.eta, st.h) * st.S**2)[:, reg].sum()
        f0 = m.tef.salt2.sum()
        st = m.step(st, **forcing(n))
        b = st.budget
        src = b["S_surf"][:, reg].sum()
        for k in ("S_num_adv", "S_phy_v", "S_phy_h", "S_sponge", "S_num_remap"):
            src -= b[k][:, reg].sum() if k in b else 0.0
        lhs = (m.layer_volumes(st.eta, st.h) * st.S**2)[:, reg].sum() - c0
        worst = max(worst, abs(lhs - (m.dt * src + (m.tef.salt2.sum() - f0))) / c0)
    for k in ("tke", "eps", "akv", "akt"):
        assert np.isfinite(st.closure[k]).all(), k
    assert np.isfinite(st.S).all() and np.isfinite(st.u).all()
    assert st.closure["akt"].max() > 1e-5                             # the GLS mixes (tide + front)
    assert worst <= 5e-10, worst


def test_gls_production_preflight():
    from core.model.contract import ConfigurationError

    with pytest.raises(ConfigurationError, match="gls_production"):
        _column("zstar", gls_production="ad_hoc")
    with pytest.warns(Warning, match="gls_production"):
        _column("zstar", closure="constant", gls_production="energy")


# ---------------------------------------------------------------- compiled merge path (N3 task 10a)
def _thin_columns(n: int = 14, seed: int = 3):
    """Columns (N, M) with thin layers at the bed, the top, inside, everywhere (c = 0), all but one (c = 1), all but
    two (c = 2), plus two columns without thin layers and one whose 1 cm top layers form their own merged layer
    (N3 task 9c); random sheared, stably stratified u, v, ρ."""
    rng = np.random.default_rng(seed)
    h = rng.uniform(0.5, 6.0, (n, 13))
    h[:3, 0] = 0.0                                                   # z*-like vanished bed layers
    h[-4:, 1] = 1e-12                                                # hybrid-like massless top layers
    h[2:n:3, 2] = 1e-10                                              # vanished interior layers
    h[5:9, 3] = 5e-4                                                 # thin (< 1 mm) block
    h[:, 4] = 0.0                                                    # dry column: c = 0
    h[:, 5] = 1e-6
    h[7, 5] = 3.0                                                    # c = 1
    h[:, 6] = 1e-6
    h[[0, 9], 6] = 2.0                                               # c = 2
    h[0, 7], h[-1, 7], h[6, 7] = 1e-4, 9e-4, 0.0
    h[1::2, 8] = 2e-4
    h[[0, -1], 9] = 0.0
    h[-3:, 12] = 0.01                                                # thin top cluster (3 cm ≥ h_massless)
    zc = np.cumsum(h, axis=0) - 0.5 * h
    u = 0.3 * np.tanh((zc - 15.0) / 6.0) + 0.05 * rng.standard_normal(h.shape)
    v = 0.1 * rng.standard_normal(h.shape)
    rho = 1026.0 - 0.02 * zc + 0.01 * rng.standard_normal(h.shape)
    return h, u, v, rho


@pytest.mark.parametrize("stratified", [True, False])
@pytest.mark.parametrize("production", ["croco", "energy"])
def test_gls_merge_numba_matches_numpy(stratified, production):
    """
    The compiled merged-column step (``GLSClosure._update_ale`` → ``physics.mixing.gls_merged.merged_columns``)
    equals the NumPy oracle ``_update_ale_numpy`` (the task-5 code) step by step from the same state, to rtol 1e-13,
    in k, ε, ν, κ_t, the N² kept for the remap and the P used; columns without thin layers take the direct step.
    """
    n = 14
    h, u, v, rho = _thin_columns(n)
    m = h.shape[1]
    rng = np.random.default_rng(7)

    def make() -> GLSClosure:
        cl = GLSClosure(n, (1, m), 30.0, stratified=stratified, rho0=1000.0, production=production)
        cl.bind_ale(rm.PPM_H4)
        return cl
    a, b = make(), make()
    for step in range(25):
        us2s, us2b = rng.uniform(0, 2e-4, (1, m)), rng.uniform(0, 5e-5, (1, m))
        uu = u * (1 + 0.1 * np.sin(step))
        work = rng.uniform(-2e-7, 2e-6, (n - 1, 1, m)) if production == "energy" else None
        kw = dict(shear_work=work)
        b.gls.tke, b.gls.eps, b.gls.akv = a.gls.tke.copy(), a.gls.eps.copy(), a.gls.akv.copy()
        if stratified:
            b.gls.akt = a.gls.akt.copy()
        a.update(uu[:, None], v[:, None], h[:, None], rho[:, None], us2s, us2b, **kw)
        # oracle: the same inputs through the NumPy merge path
        dz, us2s_r = h, (us2s.ravel() if stratified else None)
        b._update_ale_numpy(uu, v, dz, rho if stratified else rho, us2s_r, us2b.ravel(),
                            None if work is None else work.reshape(n - 1, m))
        names = ("tke", "eps", "akv", "last_prod") + (("akt",) if stratified else ())
        for name in names:
            np.testing.assert_allclose(getattr(a.gls, name), getattr(b.gls, name), rtol=1e-13, atol=0,
                                       err_msg=f"{name} step {step}")
        if stratified:
            np.testing.assert_allclose(a._n2, b._n2, rtol=1e-13, atol=0, err_msg=f"n2 step {step}")
    assert a.gls.akv.max() > 1e-4                                     # turbulent, not only floors
    assert (a.gls.tke[:, 4] == a.gls.tke_min).all()                   # dry column unchanged from the floors
