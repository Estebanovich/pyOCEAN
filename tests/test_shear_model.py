"""
Jackson et al. (2008) shear mixing in the 3-D model, ``shear_mixing='jackson'`` (Task 4 of
docs/superpowers/plans/2026-10-30-jackson-shear-mixing.md; ``core.components.closure.ShearAddedClosure``).

* off (None) is bit for bit the model without the argument, in σ and the ALE hybrid;
* a stratified column at rest gives κ_shear = 0 and the run equals the base closure bit for bit (Review Focus 2);
* a bottom current with Ri < 0.25 at its interface mixes T more than the base alone;
* Pr = 0 leaves the viscosity of the base; KPP + shear sums κ exactly;
* pre-flight reports the four configuration errors together;
* restarts are bit for bit (ALE hybrid) and a restart written without shear is refused;
* dry columns stay still bit for bit (Review Focus 3, N3 fix M1 pattern);
* periodic wrap faces are equal bit for bit and open-edge faces take only the inner cell (Review Focus 5).

Index order (k, j, i), k = 0 at the bed; interface arrays (N+1, …), 0 = bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.components.closure import ShearAddedClosure, centre_to_faces
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError
from core.restart import read_restart, write_restart
from physics.mixing.kappa_shear import JacksonParams

NZ, H = 25, 50.0
RHO0, ALPHA = 1000.0, 0.2


def _z(nz: int = NZ) -> np.ndarray:
    return (np.arange(nz) + 0.5) * H / nz                              # layer-centre heights above the bed [m]


def _T(shape, nz: int = NZ) -> np.ndarray:
    """Two water masses with a 3 m tanh interface at mid-depth: ΔT = 2 °C, N²_max ≈ 6.5e-4 s⁻²."""
    return (10.0 + np.tanh((_z(nz) - 0.5 * H) / 3.0))[:, None, None] * np.ones((nz,) + shape)


def _u_profile(nz: int = NZ, du: float = 0.5) -> np.ndarray:
    """Bottom current: du below the interface, 0 above (S²_max ≈ 6.9e-3 s⁻², Ri_min ≈ 0.09 < 0.25)."""
    return 0.5 * du * (1.0 - np.tanh((_z(nz) - 0.5 * H) / 3.0))


def _model(ale: str | None = None, shape=(2, 2), periodic=("x", "y"), nz: int = NZ, **kw) -> SigmaBaroclinic:
    ny, nx = shape
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=1e-4)
    base = dict(rho0=RHO0, t_coef=ALPHA, closure="constant", kv=1e-5, nu_const=1e-4, theta=0.6, d_min=0.05,
                periodic=periodic)
    if ale == "hybrid":
        m0 = _model("zstar", shape, periodic, nz)
        rl, rd = m0._eos.density(np.array(11.0)), m0._eos.density(np.array(9.0))
        base.update(vertical_coordinate="ale", regrid="hybrid", pgf="fv",
                    regrid_params=dict(dz_top=np.full(nz, H / nz), rho_target=np.linspace(rl - 0.2, rd + 0.2, nz + 1)))
    elif ale is not None:
        base.update(vertical_coordinate="ale", regrid=ale, pgf="fv", regrid_params=dict(dz_top=np.full(nz, H / nz)))
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SigmaBaroclinic(hg, np.full(shape, H), dt=60.0, n_layers=nz, **base)


def _start(m: SigmaBaroclinic, flow: bool = True, du: float = 0.5):
    shape = m.H.shape
    st = m.initial_state(np.zeros(shape), _T(shape, m.n))
    if flow:
        st.u = _u_profile(m.n, du)[:, None, None] * np.ones((m.n, shape[0], shape[1] + 1))
    return st


def _run(m: SigmaBaroclinic, steps: int, st=None, flow: bool = True, **kw):
    st = _start(m, flow) if st is None else st
    ny, nx = m.H.shape
    tx, ty = np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    for _ in range(steps):
        st = m.step(st, tx, ty, **kw)
    return st


def _same(a, b) -> None:
    for f in ("eta", "u", "v", "T", "h"):
        x, y = getattr(a, f), getattr(b, f)
        if x is None:
            assert y is None
        else:
            assert np.array_equal(x, y), (f, np.abs(x - y).max())


# ---------------------------------------------------------------- off and zero shear
@pytest.mark.parametrize("ale", [None, "hybrid"])
def test_off_is_bit_for_bit(ale):
    a = _run(_model(ale), 10)
    b = _run(_model(ale, shear_mixing=None), 10)
    _same(a, b)
    assert not isinstance(_model(ale, shear_mixing=None)._closure, ShearAddedClosure)


@pytest.mark.parametrize("ale", [None, "hybrid"])
def test_zero_shear_equals_base(ale):
    # stratified column at rest: S² = 0, so Jackson's κ is exactly 0 and nothing is added
    mb, ms = _model(ale), _model(ale, shear_mixing="jackson")
    a, b = _run(mb, 10, flow=False), _run(ms, 10, flow=False)
    _same(a, b)
    assert np.all(ms._closure.kd_shear == 0.0)


# ---------------------------------------------------------------- shear mixes the interface of a bottom current
@pytest.mark.parametrize("ale", [None, "hybrid"])
def test_shear_mixes_overflow_interface(ale):
    # Jackson mixes the shear layer within a few steps until Ri ≈ Ri_c (κ ~ 0.1 m² s⁻¹ at the first step, ~1e-5
    # after 20), so the first step is where κ is checked and the variance after 20 steps
    mb, ms = _model(ale), _model(ale, shear_mixing="jackson")
    a = _run(mb, 20)
    b = _run(ms, 1)
    kd = ms._closure.kd_shear.copy()
    b = _run(ms, 19, b)
    assert np.isfinite(b.T).all() and np.isfinite(b.u).all()
    if ale is None:                                                    # κ peaks at the interface (mid-depth)
        k_mid = NZ // 2
        assert kd[k_mid - 1:k_mid + 2].max() > 1e-2
        assert kd[k_mid - 1:k_mid + 2].min() > 10.0 * kd[2].max()
    assert kd.max() > 1e-2
    assert np.all(kd >= 0.0) and np.all(kd[0] == 0.0) and np.all(kd[-1] == 0.0)
    assert ms._closure.kd_shear.max() > 0.0                            # still mixing (Ri ≈ Ri_c) after 20 steps

    def var(st, m):
        V = m.layer_volumes(st.eta, st.h) if ale else m.layer_volumes(st.eta)
        mean = (V * st.T).sum() / V.sum()
        return (V * (st.T - mean) ** 2).sum()
    assert var(b, ms) < var(a, mb) * (1.0 - 1e-3)
    # ν_shear = Pr κ_shear with Pr = 1 (the default)
    assert np.array_equal(ms._closure.kv_shear, ms._closure.kd_shear)


def test_prandtl_zero_gives_no_viscosity():
    # constant base: the base has no ν of its own (None) and Pr = 0 adds none
    m = _model(None, shear_mixing="jackson", shear_prandtl=0.0)
    st = _run(m, 5)
    cl = m._closure
    assert cl.kd_shear.max() > 1e-4 and np.all(cl.kv_shear == 0.0)
    assert cl.viscosity(st) is None and cl.base.viscosity(st) is None
    # KPP base: ν is the KPP ν bit for bit
    m = _model(None, closure="kpp", shear_mixing="jackson", shear_prandtl=0.0)
    st = _run(m, 5)
    cl = m._closure
    assert cl.kd_shear.max() > 1e-4
    nu, nb = cl.viscosity(st), cl.base.viscosity(st)
    assert np.array_equal(nu[0], nb[0]) and np.array_equal(nu[1], nb[1])
    # Pr pushed after construction reaches the component
    m.shear_prandtl = 2.0
    assert cl.prandtl == 2.0


def test_kpp_plus_shear_adds():
    m = _model(None, closure="kpp", shear_mixing="jackson", shear_prandtl=0.7)
    st = _run(m, 5)
    cl = m._closure
    ks = cl.kd_shear
    assert ks.max() > 1e-4
    kt, kb = cl.tracer_diffusivity(), cl.base.tracer_diffusivity()
    np.testing.assert_allclose(kt, kb + ks, rtol=0, atol=1e-15)
    assert np.array_equal(cl.kv_shear, 0.7 * ks)
    nu, nb = cl.viscosity(st), cl.base.viscosity(st)
    su, sv = centre_to_faces(cl.kv_shear, m.periodic)
    np.testing.assert_allclose(nu[0], nb[0] + su, rtol=0, atol=1e-15)
    np.testing.assert_allclose(nu[1], nb[1] + sv, rtol=0, atol=1e-15)
    # constant base: κ = kv + κ_shear, ν = nu_const + ν_shear at the faces
    m = _model(None, shear_mixing="jackson")
    st = _run(m, 5)
    cl = m._closure
    np.testing.assert_allclose(cl.tracer_diffusivity(), m.kv + cl.kd_shear, rtol=0, atol=1e-15)
    su, sv = centre_to_faces(cl.kv_shear, m.periodic)
    nu = cl.viscosity(st)
    np.testing.assert_allclose(nu[0], m.core.nu_const + su, rtol=0, atol=1e-15)


# ---------------------------------------------------------------- pre-flight
def test_preflight_refuses_gls_and_bad_values():
    hg = create_cartesian_grid(nx=2, ny=2, dx=1000.0, dy=1000.0, f0=1e-4)
    with pytest.raises(ConfigurationError) as e:
        SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, closure="gls", z0=1e-3,
                        shear_mixing="lmd94", shear_prandtl=-1.0, jackson_params=JacksonParams(ri_crit=0.0))
    msg = str(e.value)
    for part in ("shear_mixing='lmd94'", "closure='gls'", "shear_prandtl", "ri_crit"):
        assert part in msg, (part, msg)
    for bad in (np.nan, np.inf):
        with pytest.raises(ConfigurationError, match="shear_prandtl"):
            SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, shear_mixing="jackson", shear_prandtl=bad)
    with pytest.raises(ConfigurationError, match="closure='parabolic'"):
        SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, closure="parabolic", shear_mixing="jackson")


# ---------------------------------------------------------------- restart
def test_restart_bit_for_bit_with_shear(tmp_path):
    st6 = _run(_model("hybrid", shear_mixing="jackson"), 6)
    m1 = _model("hybrid", shear_mixing="jackson")                      # fresh: initial_state takes the closure's
    st3 = _run(m1, 3)                                                  # current state, as for GLS and KPP
    write_restart(tmp_path / "rst.nc", m1, st3)
    m2 = _model("hybrid", shear_mixing="jackson")
    st6b = _run(m2, 3, read_restart(tmp_path / "rst.nc", m2))
    _same(st6, st6b)
    assert np.array_equal(st6.closure["shear_kv"], st6b.closure["shear_kv"])
    assert np.abs(st6.closure["shear_kv"]).max() > 1e-4
    # a restart of a model without shear mixing does not start one with it, and the other way round
    m0 = _model("hybrid")
    write_restart(tmp_path / "off.nc", m0, _run(m0, 1))
    with pytest.raises(ValueError, match="shear_mixing"):
        read_restart(tmp_path / "off.nc", _model("hybrid", shear_mixing="jackson"))
    with pytest.raises(ValueError, match="shear_mixing"):
        read_restart(tmp_path / "rst.nc", _model("hybrid"))
    with pytest.raises(ValueError, match="shear_prandtl"):
        read_restart(tmp_path / "rst.nc", _model("hybrid", shear_mixing="jackson", shear_prandtl=0.5))


# ---------------------------------------------------------------- dry columns (Review Focus 3)
BATHY_DRY = np.array([[H, H, 3.0, 0.02, -0.5]])                      # wet, wet, shallow, dry (D < d_min), dry bank


@pytest.mark.parametrize("closure,ale", [("constant", None), ("constant", "sigma"), ("kpp", None), ("kpp", "sigma")])
def test_dry_column_stays_still(closure, ale):
    nz = 20
    hg = create_cartesian_grid(nx=5, ny=1, dx=1000.0, dy=1000.0, f0=1e-4)
    kw = dict(rho0=RHO0, t_coef=ALPHA, closure=closure, theta=0.6, d_min=0.05, shear_mixing="jackson")
    if closure == "constant":
        kw.update(kv=1e-4, nu_const=1e-4)
    if ale is not None:
        kw.update(vertical_coordinate="ale", regrid=ale, pgf="fv")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        m = SigmaBaroclinic(hg, BATHY_DRY, dt=60.0, n_layers=nz, **kw)
    z = (np.arange(nz) + 0.5) * H / nz
    T = (10.0 + np.tanh((z - 0.5 * H) / 3.0))[:, None, None] * np.ones((nz, 1, 5))
    st = m.initial_state(np.maximum(0.0, -BATHY_DRY), T)
    T0 = st.T.copy()
    st.u = np.zeros((nz, 1, 6))
    # bottom current between the two deep cells, without depth mean: no barotropic wave reaches (and wets) the
    # shallow and dry cells, so the test sees the closure only
    u = _u_profile(nz)
    st.u[:, 0, 1] = u - u.mean()
    tx, ty = np.full((1, 6), 0.1), np.zeros((2, 5))
    for _ in range(10):
        st = m.step(st, tx, ty)
    assert np.array_equal(st.T[:, :, 3:], T0[:, :, 3:]), np.abs(st.T[:, :, 3:] - T0[:, :, 3:]).max()
    kd = m._closure.kd_shear
    assert np.all(kd[:, :, 3:] == 0.0) and np.all(m._closure.kv_shear[:, :, 3:] == 0.0)
    assert np.isfinite(st.T).all()
    assert kd[:, :, :2].max() > 1e-4                                   # the wet columns are mixed by shear


# ---------------------------------------------------------------- periodic and open faces (Review Focus 5)
def test_periodic_and_open_faces():
    shape = (4, 3)
    m = _model(None, shape=shape, periodic=("x",), open_boundaries=("south", "north"), shear_mixing="jackson")
    st = _start(m)
    st.u = st.u * (1.0 + 0.2 * np.arange(4))[None, :, None]           # row-dependent shear: rows differ
    st = _run(m, 4, st)
    cl = m._closure
    kv = cl.kv_shear
    assert kv.max() > 1e-4 and not np.array_equal(kv[:, 0], kv[:, 1])
    nu_u, nu_v = cl.viscosity(st)
    nc = m.core.nu_const
    assert np.array_equal(nu_u[:, :, 0], nu_u[:, :, -1])               # wrap faces stored twice, equal
    assert np.array_equal(nu_u[:, :, 0], nc + 0.5 * (kv[:, :, -1] + kv[:, :, 0]))
    assert np.array_equal(nu_v[:, 0], nc + kv[:, 0])                   # open edges: the inner cell only
    assert np.array_equal(nu_v[:, -1], nc + kv[:, -1])
    assert np.isfinite(st.T).all() and np.isfinite(st.u).all()


@pytest.mark.parametrize("ale", [None, "zstar"])
def test_none_base_gets_shear_viscosity(ale):
    # closure 'none' has no vertical viscosity of its own (the column solver skipped the solve); with shear mixing
    # the lagged ν_shear must reach it: the shear of the bottom current is reduced, ν_base = 0 at the faces
    a = _run(_model(ale, closure="none"), 6)
    m = _model(ale, closure="none", shear_mixing="jackson")
    b = _run(m, 6)
    nu_u, _ = m._closure.viscosity(b)
    su, _ = centre_to_faces(m._closure.kv_shear, m.periodic)
    assert np.array_equal(nu_u, 0.0 + su)
    shear = lambda st: np.abs(np.diff(st.u[:, 0, 0])).max()            # noqa: E731
    assert shear(b) < 0.9 * shear(a)


# ---------------------------------------------------------------- fix round 1 (review of Task 4)
@pytest.mark.parametrize("closure", ["kpp", "constant"])
def test_sigma_restart_bit_for_bit_with_shear(tmp_path, closure):
    # C1: KPP + shear crashed in write_restart (the KPP config block read the wrapper); M5: σ with a constant base
    st6 = _run(_model(None, closure=closure, shear_mixing="jackson"), 6)
    m1 = _model(None, closure=closure, shear_mixing="jackson")
    write_restart(tmp_path / "rst.nc", m1, _run(m1, 3))
    m2 = _model(None, closure=closure, shear_mixing="jackson")
    st6b = _run(m2, 3, read_restart(tmp_path / "rst.nc", m2))
    _same(st6, st6b)
    assert np.array_equal(st6.closure["shear_kv"], st6b.closure["shear_kv"])
    import xarray as xr
    with xr.open_dataset(tmp_path / "rst.nc") as ds:                  # M2: one fixed file name for the shear field
        assert "shear_kv" in ds and "kpp_shear_kv" not in ds
        if closure == "kpp":
            assert "kpp_akv" in ds and ds.attrs["config_kpp_cp"] == m1.cp


def test_nu_const_after_construction_reaches_shear():
    # M3: ν_base of a constant base is the live nu_const of the core
    m = _model(None, shear_mixing="jackson")
    st = _run(m, 3)
    m.core.nu_const = 3e-3
    su, _ = centre_to_faces(m._closure.kv_shear, m.periodic)
    nu_u, _ = m._closure.viscosity(st)
    assert np.array_equal(nu_u, 3e-3 + su)


def test_wrapper_forwards_base_attributes():
    # M4: helpers read KPP attributes through model._closure (validation/turbulence_columns/mom6_scm.py)
    m = _model(None, closure="kpp", shear_mixing="jackson")
    _run(m, 2)
    cl = m._closure
    assert cl.obl_depth is cl.base.obl_depth
    assert cl.akv is cl.base.akv
    with pytest.raises(AttributeError):
        cl.no_such_attribute
