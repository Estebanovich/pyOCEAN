"""
Drag-driven bottom-boundary-layer mixing in the 3-D model, ``bbl_mixing='lotw'`` (Task 4 of
docs/superpowers/plans/2026-10-30-bbl-drag-mixing.md; spec 2026-10-30-bbl-drag-mixing-design.md;
``core.components.closure.BBLAddedClosure``, port of MOM6 ``add_LOTW_BBL_diffusivity`` in
``physics.mixing.bbl_mixing``).

* off (None) is bit for bit the model without the argument (ALE hybrid and z*);
* a stratified column at rest (no drag work) equals the base closure bit for bit;
* the energy is pyOCEAN's drag work, C_d |U|³ for a uniform flow (spec "Energía física, no la de MOM6");
* a dense bottom current over a slope gets Kd_BBL > 0 near the bed and loses more T variance than the base;
* with Jackson: κ = κ_base + κ_shear + Kd_BBL and the viscosity is that of base + Jackson bit for bit;
* the pre-flight reports every configuration error at once;
* restarts are bit for bit (ALE hybrid, Jackson + BBL); dry columns stay still;
* periodic wrap faces and open edges in the face-to-centre energy (Review Focus 5);
* ``bbl_params`` assigned after construction reaches the closure and is checked.

Index order (k, j, i), k = 0 at the bed; interface arrays (N+1, …), 0 = bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.components.closure import BBLAddedClosure, ShearAddedClosure
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError
from core.model.driver import StepContext
from core.restart import read_restart, write_restart
from physics.mixing.bbl_mixing import BBLParams

NZ, H = 20, 50.0
RHO0, ALPHA = 1000.0, 0.2
CD = 2e-3
DRAG = dict(bottom_drag="quadratic", cd=CD, hbbl=10.0)


def _z(nz: int = NZ, h: float = H) -> np.ndarray:
    return (np.arange(nz) + 0.5) * h / nz                              # layer-centre heights above the bed [m]


def _T_column(nz: int = NZ, h: float = H) -> np.ndarray:
    """Dense bottom water: 2 °C colder below a 2 m tanh interface 8 m above the bed (N² ≈ 2e-3 s⁻²)."""
    return 10.0 + np.tanh((_z(nz, h) - 8.0) / 2.0)


def _model(ale: str = "zstar", shape=(2, 2), periodic=("x", "y"), nz: int = NZ, bathy=None, **kw) -> SigmaBaroclinic:
    ny, nx = shape
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=1e-4)
    bathy = np.full(shape, H) if bathy is None else bathy
    base = dict(rho0=RHO0, t_coef=ALPHA, closure="constant", kv=1e-5, nu_const=1e-4, theta=0.6, d_min=0.05,
                periodic=periodic, vertical_coordinate="ale", pgf="fv", **DRAG)
    hmax = float(np.max(bathy))
    if ale == "hybrid":
        rl, rd = RHO0 - ALPHA * 11.2, RHO0 - ALPHA * 8.8               # linear EOS ρ = ρ₀ − α T (t_ref 0)
        base.update(regrid="hybrid", regrid_params=dict(dz_top=np.full(nz, hmax / nz),
                                                        rho_target=np.linspace(rl, rd, nz + 1)))
    elif ale == "zstar":
        base.update(regrid="zstar", regrid_params=dict(dz_top=np.full(nz, hmax / nz)))
    else:
        base.update(regrid=ale)
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SigmaBaroclinic(hg, bathy, dt=60.0, n_layers=nz, **base)


def _start(m: SigmaBaroclinic, U: float = 0.3, along: str = "x"):
    ny, nx = m.H.shape
    T = np.stack([_T_column(m.n, float(m.H[j, i])) for j in range(ny) for i in range(nx)], axis=1)
    st = m.initial_state(np.zeros((ny, nx)), T.reshape(m.n, ny, nx))
    if along == "x":
        st.u = np.full((m.n, ny, nx + 1), U)
    else:
        st.v = np.full((m.n, ny + 1, nx), U)
        if "y" not in m.periodic.axes:
            st.v[:, 0] = st.v[:, -1] = 0.0
    return st


def _run(m: SigmaBaroclinic, steps: int, st=None, **kw):
    st = _start(m, **kw) if st is None else st
    ny, nx = m.H.shape
    tx, ty = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    for _ in range(steps):
        st = m.step(st, tx, ty)
    return st


def _same(a, b) -> None:
    for f in ("eta", "u", "v", "T", "h"):
        x, y = getattr(a, f), getattr(b, f)
        if x is None:
            assert y is None
        else:
            assert np.array_equal(x, y), (f, np.abs(x - y).max())


def _var(st, m) -> float:
    V = m.layer_volumes(st.eta, st.h)
    mean = (V * st.T).sum() / V.sum()
    return float((V * (st.T - mean) ** 2).sum())


# ---------------------------------------------------------------- off and no drag work
@pytest.mark.parametrize("ale", ["hybrid", "zstar"])
def test_off_is_bit_for_bit(ale):
    a = _run(_model(ale), 10)
    m = _model(ale, bbl_mixing=None)
    b = _run(m, 10)
    _same(a, b)
    assert not isinstance(m._closure, BBLAddedClosure)


@pytest.mark.parametrize("ale", ["hybrid", "zstar"])
def test_no_flow_no_mixing(ale):
    # stratified column at rest, drag_bg_vel = 0: no drag work, u* = 0, so Kd_BBL ≡ 0 and the run is the base's
    mb, mm = _model(ale, drag_bg_vel=0.0), _model(ale, drag_bg_vel=0.0, bbl_mixing="lotw")
    a, b = _run(mb, 10, U=0.0), _run(mm, 10, U=0.0)
    _same(a, b)
    assert isinstance(mm._closure, BBLAddedClosure)
    assert np.all(mm._closure.kd_bbl == 0.0) and np.all(mm._closure.energy == 0.0)


# ---------------------------------------------------------------- energy = drag work
def test_energy_is_drag_work():
    # uniform U in x, quadratic drag without background speed: per face Σ_k r f_k u_k² = C_d |U| U² Σ f_k = C_d |U|³
    U = 0.3
    m = _model("zstar", shape=(3, 4), periodic=("x",), bbl_mixing="lotw")
    st = _start(m, U=U)
    ny, nx = m.H.shape
    ctx = StepContext(st=st)
    m._driver._ale_prepare(ctx)
    r_u, r_v, f_u, f_v = m._bed_drag.rates(st.u, st.v, ctx.hu, ctx.hv)
    visc = dict(r_u=r_u[None] * f_u, r_v=r_v[None] * f_v)
    e = m._driver._ale_bed_energy(ctx, st.u, st.v, visc)
    assert e.shape == (ny, nx)
    np.testing.assert_allclose(e, CD * U**3, rtol=1e-12, atol=0)
    # in the model: the energy of a step is the drag work at the decelerated velocities, within ~1 % of C_d|U|³
    st = _run(m, 1, st)
    E = m._closure.energy
    assert E.shape == (ny, nx) and np.all(E > 0)
    np.testing.assert_allclose(E, CD * U**3, rtol=2e-2)
    # u* = √(|τ_b|/ρ₀) of the same drag: Kd_BBL > 0 in the BBL
    assert m._closure.kd_bbl[1:3].min() > 0.0                          # spent by the interface at 7.5 m (N² peak)


def test_energy_with_log_law():
    # law of the wall: the rates are recomputed at the new velocities, as for u*² (``_ale_bed_ustar2``)
    U = 0.3
    m = _model("zstar", shape=(2, 3), periodic=("x",), z0=1e-3, hbbl=10.0, bottom_drag=None, bbl_mixing="lotw")
    st = _start(m, U=U)
    ctx = StepContext(st=st)
    m._driver._ale_prepare(ctx)
    e = m._driver._ale_bed_energy(ctx, st.u, st.v, {})
    r_u, _ = m._bed_drag.rates(st.u, st.v, ctx.hu, ctx.hv)
    eu = (r_u * (st.u * st.u)).sum(axis=0)
    np.testing.assert_allclose(e, 0.5 * (eu[:, :-1] + eu[:, 1:]), rtol=1e-14, atol=0)
    # = C_d(d_b) |U|³ with the log-law C_d of a bed layer thicker than hbbl... here the z* bed layer is 2.5 m
    assert np.all(e > 0)
    st = _run(m, 2, st)
    assert np.all(m._closure.energy > 0) and m._closure.kd_bbl.max() > 0.0


# ---------------------------------------------------------------- the BBL mixes a dense bottom current
def test_bbl_mixes_dense_bottom_layer():
    # along-slope dense bottom current (v, periodic in y) over a gentle slope (H 48–54 m across x)
    bathy = np.tile(np.linspace(48.0, 54.0, 4), (2, 1))
    kw = dict(shape=(2, 4), periodic=("y",), bathy=bathy)
    mb, mm = _model("zstar", **kw), _model("zstar", bbl_mixing="lotw", **kw)
    a = _run(mb, 20, U=0.3, along="y")
    b = _run(mm, 1, U=0.3, along="y")
    kd = mm._closure.kd_bbl.copy()
    b = _run(mm, 19, b)
    assert np.isfinite(b.T).all() and np.isfinite(b.u).all()
    assert np.all(kd >= 0.0) and np.all(kd[0] == 0.0) and np.all(kd[-1] == 0.0)
    assert np.all(kd[1:6].max(axis=0) > 1e-3)                          # every column mixes near the bed (≤ 5 interfaces)
    assert mm._closure.kd_bbl.max() > 0.0
    assert _var(b, mm) < _var(a, mb) * (1.0 - 1e-3)


# ---------------------------------------------------------------- additive with Jackson
def test_adds_to_jackson():
    m = _model("zstar", shear_mixing="jackson", bbl_mixing="lotw")
    st = _run(m, 3, U=0.3)
    cl = m._closure
    assert isinstance(cl, BBLAddedClosure) and isinstance(cl.base, ShearAddedClosure)
    kb, ks = cl.kd_bbl, cl.base.kd_shear
    assert kb.max() > 1e-4
    np.testing.assert_allclose(cl.tracer_diffusivity(), m.kv + ks + kb, rtol=0, atol=1e-15)
    nu, nb = cl.viscosity(st), cl.base.viscosity(st)
    if nb is None:
        assert nu is None
    else:
        assert np.array_equal(nu[0], nb[0]) and np.array_equal(nu[1], nb[1])
    # Jackson attributes and parameters still reach the shear closure through the wrapper
    m.shear_prandtl = 2.0
    assert cl.base.prandtl == 2.0 and cl.prandtl == 2.0
    m.kv = 2e-5
    assert cl.kv == 2e-5 and cl.base.kv == 2e-5
    np.testing.assert_allclose(cl.tracer_diffusivity(), 2e-5 + ks + kb, rtol=0, atol=1e-15)


def test_kpp_base_adds():
    m = _model("zstar", closure="kpp", bbl_mixing="lotw")
    st = _run(m, 3, U=0.3)
    cl = m._closure
    kb = cl.kd_bbl
    assert kb.max() > 1e-4
    np.testing.assert_allclose(cl.tracer_diffusivity(), cl.base.tracer_diffusivity() + kb, rtol=0, atol=1e-15)
    nu, nb = cl.viscosity(st), cl.base.viscosity(st)
    assert np.array_equal(nu[0], nb[0]) and np.array_equal(nu[1], nb[1])
    assert cl.obl_depth is cl.base.obl_depth                           # helpers read KPP through the wrapper


# ---------------------------------------------------------------- pre-flight
def test_preflight_refuses_all_at_once():
    hg = create_cartesian_grid(nx=2, ny=2, dx=1000.0, dy=1000.0, f0=1e-4)
    with pytest.raises(ConfigurationError) as e:
        SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, closure="gls", bbl_mixing="lotw",
                        bbl_params=BBLParams(kd_max=0.0))
    msg = str(e.value)
    for part in ("closure='gls'", "vertical_coordinate='ale'", "bed drag", "bbl_params.kd_max"):
        assert part in msg, (part, msg)
    with pytest.raises(ConfigurationError) as e:
        SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, vertical_coordinate="ale", pgf="fv",
                        bottom_drag="quadratic", bbl_mixing="mom6_drag",
                        bbl_params=BBLParams(bbl_effic=1.5, max_decay=0.0, von_karman=float("nan"),
                                             kd_max=float("inf")))
    msg = str(e.value)
    for part in ("bbl_mixing='mom6_drag'", "bbl_params.bbl_effic", "bbl_params.max_decay",
                 "bbl_params.von_karman", "bbl_params.kd_max"):
        assert part in msg, (part, msg)
    with pytest.raises(ConfigurationError, match="closure='parabolic'"):
        SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, closure="parabolic", bbl_mixing="lotw")
    with pytest.raises(ConfigurationError, match="bbl_params must be"):
        SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, vertical_coordinate="ale", pgf="fv",
                        bottom_drag="quadratic", bbl_mixing="lotw", bbl_params={"bbl_effic": 0.2})
    # the defaults and the edge values are accepted; a log-law drag counts as bed drag
    _model("zstar", bbl_mixing="lotw", bbl_params=BBLParams(bbl_effic=0.0))
    _model("zstar", bbl_mixing="lotw", bbl_params=BBLParams(bbl_effic=1.0, use_omega=False))
    _model("zstar", bbl_mixing="lotw", bottom_drag=None, z0=1e-3)
    with pytest.warns(Warning, match="bbl_params is ignored"):
        SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, bbl_params=BBLParams())


# ---------------------------------------------------------------- restart
def test_restart_bit_for_bit_with_bbl(tmp_path):
    kw = dict(shear_mixing="jackson", bbl_mixing="lotw")
    st6 = _run(_model("hybrid", **kw), 6)
    m1 = _model("hybrid", **kw)
    st3 = _run(m1, 3)
    write_restart(tmp_path / "rst.nc", m1, st3)
    m2 = _model("hybrid", **kw)
    st6b = _run(m2, 3, read_restart(tmp_path / "rst.nc", m2))
    _same(st6, st6b)
    assert np.array_equal(st6.closure["shear_kv"], st6b.closure["shear_kv"])
    assert m1._closure.kd_bbl.max() > 0.0
    # another BBL configuration does not read it
    with pytest.raises(ValueError, match="bbl_mixing"):
        read_restart(tmp_path / "rst.nc", _model("hybrid", shear_mixing="jackson"))
    with pytest.raises(ValueError, match="bbl_"):
        read_restart(tmp_path / "rst.nc", _model("hybrid", bbl_params=BBLParams(bbl_effic=0.1), **kw))
    # constant base without Jackson: no closure state at all, still bit for bit
    st6 = _run(_model("hybrid", bbl_mixing="lotw"), 6)
    m1 = _model("hybrid", bbl_mixing="lotw")
    write_restart(tmp_path / "c.nc", m1, _run(m1, 3))
    m2 = _model("hybrid", bbl_mixing="lotw")
    _same(st6, _run(m2, 3, read_restart(tmp_path / "c.nc", m2)))


# ---------------------------------------------------------------- dry columns
BATHY_DRY = np.array([[H, H, 3.0, 0.02, -0.5]])                      # wet, wet, shallow, dry (D < d_min), dry bank


@pytest.mark.parametrize("closure", ["constant", "kpp"])
def test_dry_column_stays_still(closure):
    nz = 20
    hg = create_cartesian_grid(nx=5, ny=1, dx=1000.0, dy=1000.0, f0=1e-4)
    kw = dict(rho0=RHO0, t_coef=ALPHA, closure=closure, theta=0.6, d_min=0.05, vertical_coordinate="ale",
              regrid="sigma", pgf="fv", bbl_mixing="lotw", shear_mixing="jackson", drag_bg_vel=0.05, **DRAG)
    if closure == "constant":
        kw.update(kv=1e-4, nu_const=1e-4)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SigmaBaroclinic(hg, BATHY_DRY, dt=60.0, n_layers=nz, **kw)
    z = (np.arange(nz) + 0.5) * H / nz
    T = (10.0 + np.tanh((z - 8.0) / 2.0))[:, None, None] * np.ones((nz, 1, 5))
    st = m.initial_state(np.maximum(0.0, -BATHY_DRY), T)
    T0 = st.T.copy()
    st.u = np.zeros((nz, 1, 6))
    u = 0.3 * np.exp(-z / 5.0)
    st.u[:, 0, 1] = u - u.mean()                                       # bottom current without depth mean
    tx, ty = np.full((1, 6), 0.1), np.zeros((2, 5))
    for _ in range(10):
        st = m.step(st, tx, ty)
    assert np.array_equal(st.T[:, :, 3:], T0[:, :, 3:]), np.abs(st.T[:, :, 3:] - T0[:, :, 3:]).max()
    kd = m._closure.kd_bbl
    assert np.all(kd[:, :, 3:] == 0.0)
    assert np.isfinite(st.T).all() and np.isfinite(m._closure.energy).all()
    assert kd[:, :, :2].max() > 0.0                                    # the wet columns are mixed


# ---------------------------------------------------------------- periodic and open faces (Review Focus 5)
def test_periodic_and_open_energy():
    shape = (4, 3)
    m = _model("zstar", shape=shape, periodic=("x",), open_boundaries=("south", "north"), bbl_mixing="lotw",
               drag_bg_vel=0.05)
    ny, nx = shape
    st = _start(m, U=0.3)
    st.u = st.u * (1.0 + 0.2 * np.arange(ny))[None, :, None]           # rows differ
    st.v = 0.1 * np.ones((m.n, ny + 1, nx)) * (1.0 + 0.3 * np.arange(nx))[None, None, :]
    ctx = StepContext(st=st)
    m._driver._ale_prepare(ctx)
    r_u, r_v, f_u, f_v = m._bed_drag.rates(st.u, st.v, ctx.hu, ctx.hv)
    visc = dict(r_u=r_u[None] * f_u, r_v=r_v[None] * f_v)
    e = m._driver._ale_bed_energy(ctx, st.u, st.v, visc)
    eu = (visc["r_u"] * (st.u * st.u)).sum(axis=0)
    ev = (visc["r_v"] * (st.v * st.v)).sum(axis=0)
    assert np.array_equal(eu[:, 0], eu[:, -1])                         # wrap face stored twice, equal
    assert np.array_equal(e[:, 0], 0.5 * (eu[:, 0] + eu[:, 1]) + 0.5 * (ev[:-1, 0] + ev[1:, 0]))
    assert np.array_equal(e[:, -1], 0.5 * (eu[:, -2] + eu[:, 0]) + 0.5 * (ev[:-1, -1] + ev[1:, -1]))
    st = _run(m, 4, st)
    E = m._closure.energy
    assert np.isfinite(E).all() and np.all(E >= 0.0)
    assert np.all(E[0] > 0.0) and np.all(E[-1] > 0.0)                  # next to the open edges
    assert np.isfinite(st.T).all() and np.isfinite(st.u).all()
    assert np.isfinite(m._closure.kd_bbl).all() and np.all(m._closure.kd_bbl >= 0.0)


# ---------------------------------------------------------------- parameters after construction
def test_bbl_params_after_construction():
    m = _model("zstar", bbl_mixing="lotw")
    assert m._closure.bbl_params == BBLParams()
    new = BBLParams(bbl_effic=0.0)                                     # no energy for mixing: Kd_BBL = 0
    m.bbl_params = new
    assert m._closure.bbl_params is new
    _run(m, 2)
    assert np.all(m._closure.kd_bbl == 0.0)
    m.bbl_params = None
    assert m._closure.bbl_params == BBLParams()
    with pytest.raises(ConfigurationError, match="bbl_params.kd_max"):
        m.bbl_params = BBLParams(kd_max=-1.0)
    with pytest.raises(ConfigurationError, match="bbl_params must be"):
        m.bbl_params = 0.2
    assert m._closure.bbl_params == BBLParams()
