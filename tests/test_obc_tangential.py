"""
MOM6 tangential open-boundary velocity (ORLANSKI_TAN + NUDGED_TAN) and the "computed" vorticity and shear strain
at the open-edge corners (``obc_tangential='mom6'``; reading notes docs/design/regional_refs/mom6_obc_tangential.md).

Pure functions first (corner set, radiated + nudged tangential velocity, corner vorticity, corner strain in the
Smagorinsky operator), then the model wiring (off = bit for bit, a jet entering through an open edge, a pulse
leaving without growth, restart, threads, pre-flight, exterior data).  Index order (k, j, i), k = 0 at the bed;
corners (N, ny+1, nx+1), corner [J, I] shared by cells (J−1, J) × (I−1, I).
"""
from __future__ import annotations

import numpy as np
import pytest

from core.dynamics import open_boundary as obc
from core.dynamics import vector_invariant as vi
from core.dynamics.hor_visc import open_corner_strain, smag_geometry, smagorinsky, smagorinsky_numpy
from core.grid.horizontal import create_cartesian_grid


# ------------------------------------------------------------------ corner set
def test_tangential_corners_between_two_open_wet_faces():
    ny, nx = 4, 5
    open_u = np.zeros((ny, nx + 1), bool)
    open_v = np.zeros((ny + 1, nx), bool)
    open_u[:, 0] = [True, True, False, True]                  # west: a land cell at j = 2
    open_v[0, :] = True                                       # south
    q = obc.tangential_corners(open_u, open_v, ("west", "south"))
    assert q.shape == (ny + 1, nx + 1)
    assert q[1, 0] and not q[2, 0] and not q[3, 0]           # J = 2, 3 touch the land face
    assert not q[0, 0] and not q[ny, 0]                       # outer corners of the domain stay closed
    assert q[0, 1:nx].all() and not q[0, nx]
    assert q.sum() == 1 + (nx - 1)
    assert not obc.tangential_corners(open_u, open_v, ("east",)).any()


# ------------------------------------------------------------------ corner phase speed and tangential velocity
def test_corner_phase_speed_is_the_mean_of_the_two_faces():
    r = np.array([[0.0, 0.4, 1.0]])                           # (N, n) at the three faces of an edge
    rc = obc.corner_phase_speed(r)
    np.testing.assert_allclose(rc, [[0.0, 0.2, 0.7, 1.0]])   # ends one-sided (MOM_open_boundary.F90:3031-3036)


def test_tangential_velocity_radiated_and_nudged_west_and_south():
    N, ny, nx, dt, tin, tout = 2, 3, 4, 600.0, 3 * 86400.0, 360 * 86400.0
    rng = np.random.default_rng(0)
    u = rng.standard_normal((N, ny, nx + 1))
    v = rng.standard_normal((N, ny + 1, nx))
    flags = np.zeros((ny + 1, nx + 1), bool)
    flags[1:-1, 0] = True
    flags[0, 1:-1] = True
    r = np.zeros((N, ny + 1, nx + 1))
    r[1, 1:-1, 0] = 0.5                                       # outgoing in layer 1 of the west edge
    ext = np.full((N, ny + 1, nx + 1), 0.3)
    vt = obc.tangential_velocity(u, v, r, ext, flags, dt, tin, tout)
    g_in, g_out = dt / (tin + dt), dt / (tout + dt)
    J = 2
    np.testing.assert_allclose(vt[0, J, 0], (1 - g_in) * v[0, J, 0] + g_in * 0.3, rtol=1e-14)
    rad = (v[1, J, 0] + 0.5 * v[1, J, 1]) / 1.5
    np.testing.assert_allclose(vt[1, J, 0], (1 - g_out) * rad + g_out * 0.3, rtol=1e-14)
    I = 2
    np.testing.assert_allclose(vt[0, 0, I], (1 - g_in) * u[0, 0, I] + g_in * 0.3, rtol=1e-14)
    assert np.all(vt[:, ~flags] == 0.0)


def test_tangential_velocity_east_and_north_use_the_last_interior_rows():
    N, ny, nx = 1, 3, 4
    u = np.arange(N * ny * (nx + 1), dtype=float).reshape(N, ny, nx + 1)
    v = 100.0 + np.arange(N * (ny + 1) * nx, dtype=float).reshape(N, ny + 1, nx)
    flags = np.zeros((ny + 1, nx + 1), bool)
    flags[1:-1, nx] = True
    flags[ny, 1:-1] = True
    r = np.ones((N, ny + 1, nx + 1))
    vt = obc.tangential_velocity(u, v, r, np.zeros_like(r), flags, 1.0, np.inf, np.inf)
    np.testing.assert_allclose(vt[0, 1, nx], 0.5 * (v[0, 1, nx - 1] + v[0, 1, nx - 2]))
    np.testing.assert_allclose(vt[0, ny, 2], 0.5 * (u[0, ny - 1, 2] + u[0, ny - 2, 2]))


# ------------------------------------------------------------------ corner vorticity (MOM_CoriolisAdv.F90:470-524)
def test_open_corner_vorticity_formulas():
    N, ny, nx = 1, 3, 4
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=800.0, dy=1000.0, f0=1e-4)
    dx_u, dy_v = hg["dx_u"].values, hg["dy_v"].values
    area_q = vi.corner_area(dx_u, dy_v)
    rng = np.random.default_rng(1)
    u = rng.standard_normal((N, ny, nx + 1))
    v = rng.standard_normal((N, ny + 1, nx))
    vt = rng.standard_normal((N, ny + 1, nx + 1))
    flags = obc.tangential_corners(np.ones((ny, nx + 1), bool) & (np.arange(nx + 1) % nx == 0)[None],
                                   np.ones((ny + 1, nx), bool) & (np.arange(ny + 1) % ny == 0)[:, None],
                                   ("west", "east", "south", "north"))
    z = vi.open_corner_vorticity(u, v, vt, flags, dx_u, dy_v, area_q)
    J, I = 1, 2
    w = (2 * (v[0, J, 0] - vt[0, J, 0]) * dy_v[J, 0] - (u[0, J, 0] * dx_u[J, 0] - u[0, J - 1, 0] * dx_u[J - 1, 0]))
    np.testing.assert_allclose(z[0, J, 0], w / area_q[J, 0], rtol=1e-13)
    e = (2 * (vt[0, J, nx] - v[0, J, nx - 1]) * dy_v[J, nx - 1]
         - (u[0, J, nx] * dx_u[J, nx] - u[0, J - 1, nx] * dx_u[J - 1, nx]))
    np.testing.assert_allclose(z[0, J, nx], e / area_q[J, nx], rtol=1e-13)
    s = (v[0, 0, I] * dy_v[0, I] - v[0, 0, I - 1] * dy_v[0, I - 1]) - 2 * (u[0, 0, I] - vt[0, 0, I]) * dx_u[0, I]
    np.testing.assert_allclose(z[0, 0, I], s / area_q[0, I], rtol=1e-13)
    n_ = ((v[0, ny, I] * dy_v[ny, I] - v[0, ny, I - 1] * dy_v[ny, I - 1])
          - 2 * (vt[0, ny, I] - u[0, ny - 1, I]) * dx_u[ny - 1, I])
    np.testing.assert_allclose(z[0, ny, I], n_ / area_q[ny, I], rtol=1e-13)
    assert np.all(z[:, ~flags] == 0.0)


# ------------------------------------------------------------------ corner strain (MOM_hor_visc.F90:795-836)
def _smag_setup(nx=5, ny=4, N=2, seed=2):
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=900.0, dy=1100.0, f0=1e-4)
    g = smag_geometry(hg)
    rng = np.random.default_rng(seed)
    u = 0.3 * rng.standard_normal((N, ny, nx + 1))
    v = 0.3 * rng.standard_normal((N, ny + 1, nx))
    h_t = 5.0 + rng.random((N, ny, nx))
    h_u = 5.0 + rng.random((N, ny, nx + 1))
    h_v = 5.0 + rng.random((N, ny + 1, nx))
    open_u = np.zeros((ny, nx + 1), bool)
    open_v = np.zeros((ny + 1, nx), bool)
    open_u[:, 0] = True
    open_v[0, :] = True
    wet_u = np.ones((ny, nx + 1), bool)
    wet_v = np.ones((ny + 1, nx), bool)
    wet_u[:, 0] = wet_u[:, -1] = False                        # outside-domain faces are not wet (NaN H_u)
    wet_v[0, :] = wet_v[-1, :] = False
    mask_q = np.zeros((ny + 1, nx + 1), bool)
    mask_q[1:-1, 1:-1] = wet_u[:-1, 1:-1] & wet_u[1:, 1:-1] & wet_v[1:-1, :-1] & wet_v[1:-1, 1:]
    flags = obc.tangential_corners(open_u, open_v, ("west", "south"))
    vt = 0.3 * rng.standard_normal((N, ny + 1, nx + 1))
    return dict(u=u, v=v, h_t=h_t, h_u=h_u, h_v=h_v, wet_u=wet_u, wet_v=wet_v, mask_q=mask_q, open_u=open_u,
                open_v=open_v, g=g, flags=flags, vt=vt)


def test_open_corner_strain_formulas():
    s = _smag_setup()
    g, u, v, vt = s["g"], s["u"], s["v"], s["vt"]
    ds, hq, hmin = open_corner_strain(u, v, vt, s["flags"], s["h_t"], s["h_u"], s["h_v"], g)
    J = 2
    dvdx = 2 * g["dy_dx_q"][J, 0] * (v[0, J, 0] - vt[0, J, 0]) * g["idy_v"][J, 0]
    dudy = g["dx_dy_q"][J, 0] * (u[0, J, 0] * g["idx_u"][J, 0] - u[0, J - 1, 0] * g["idx_u"][J - 1, 0])
    np.testing.assert_allclose(ds[0, J, 0], dvdx + dudy, rtol=1e-13)
    hus, hun, hv = s["h_t"][0, J - 1, 0], s["h_t"][0, J, 0], s["h_v"][0, J, 0]
    h2u, h2v = 4 * hus * hun, 4 * hv * hv
    np.testing.assert_allclose(hq[0, J, 0], 2 * h2u * h2v / ((h2u + h2v) * (hus + hun + 2 * hv)), rtol=1e-12)
    assert hmin[0, J, 0] == min(hus, hun, hv)
    I = 3
    dudy = 2 * g["dx_dy_q"][0, I] * (u[1, 0, I] - vt[1, 0, I]) * g["idx_u"][0, I]
    dvdx = g["dy_dx_q"][0, I] * (v[1, 0, I] * g["idy_v"][0, I] - v[1, 0, I - 1] * g["idy_v"][0, I - 1])
    np.testing.assert_allclose(ds[1, 0, I], dvdx + dudy, rtol=1e-13)
    assert np.all(ds[:, ~s["flags"]] == 0.0)


@pytest.mark.parametrize("threads", [1, 5])
def test_smagorinsky_with_open_corners_numba_equals_numpy(threads):
    from core.model.threads import _THREADS, set_threads
    s = _smag_setup()
    obc_args = open_corner_strain(s["u"], s["v"], s["vt"], s["flags"], s["h_t"], s["h_u"], s["h_v"], s["g"])
    args = (s["u"], s["v"], s["h_t"], s["h_u"], s["h_v"], s["wet_u"], s["wet_v"], s["mask_q"], s["open_u"],
            s["open_v"], s["g"], 0.15, 2.0, 60.0)
    old = _THREADS[0]
    try:
        set_threads(threads)
        a = smagorinsky(*args, obc_q=s["flags"], obc_strain=obc_args)
    finally:
        set_threads(old)
    b = smagorinsky_numpy(*args, obc_q=s["flags"], obc_strain=obc_args)
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(x, y)
    assert a[4] == b[4]
    nu_q = a[3]
    assert np.all(nu_q[:, s["flags"]] > 0.0)                 # the open-edge corners are active now
    fv_off = smagorinsky_numpy(*args)[1]
    assert not np.array_equal(a[1][:, 1:-1, 0], fv_off[:, 1:-1, 0])   # the v next to the west edge feels it
    np.testing.assert_array_equal(a[1][:, 2:, 2:], fv_off[:, 2:, 2:])  # away from the edges: unchanged


def test_smagorinsky_without_open_corners_is_unchanged():
    s = _smag_setup()
    args = (s["u"], s["v"], s["h_t"], s["h_u"], s["h_v"], s["wet_u"], s["wet_v"], s["mask_q"], s["open_u"],
            s["open_v"], s["g"], 0.15, 2.0, 60.0)
    a, b = smagorinsky(*args), smagorinsky(*args, obc_q=np.zeros_like(s["flags"]),
                                           obc_strain=tuple(np.zeros((2,) + s["flags"].shape) for _ in range(3)))
    for x, y in zip(a[:4], b[:4]):
        np.testing.assert_array_equal(x, y)


# ================================================================== model wiring
from boundaries.obcs import OBData                                                   # noqa: E402
from core.dynamics import sigma_baroclinic as sb                                     # noqa: E402
from core.dynamics.sigma_baroclinic import SigmaBaroclinic                           # noqa: E402
from core.model.contract import ConfigurationError                                   # noqa: E402

RHO0, TCOEF, G = 1000.0, 0.2, 9.81


def _box(obc_tangential="mom6", nx=12, ny=10, nz=3, f0=1e-4, open_=("west", "south", "east"), state=None, **kw):
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=2000.0, dy=2000.0, f0=f0)
    cfg = dict(dt=120.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None, closure="constant", nu_const=1e-4,
               vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, 30.0 / nz)),
               pgf="fv", momentum="vector_invariant", smagorinsky=0.15, open_boundaries=open_,
               open_boundary_state=state, obc_tangential=obc_tangential, g=G)
    cfg.update(kw)
    return SigmaBaroclinic(hg, np.full((ny, nx), 30.0), **cfg)


def test_preflight_obc_tangential():
    with pytest.raises(ConfigurationError, match="obc_tangential"):
        _box(obc_tangential="orlanski")
    with pytest.raises(ConfigurationError, match="obc_tangential"):
        _box(vertical_coordinate="sigma_eulerian", momentum=None, regrid_params=None, pgf="jacobian")
    with pytest.raises(ConfigurationError, match="obc_tangential"):
        _box(momentum="flux")
    with pytest.raises(ConfigurationError, match="obc_tangential"):
        _box(open_=("west",), periodic=("y",))
    m = _box(obc_tangential="none")
    assert m.obc_tangential == "none"
    assert _box().obc_tangential == "mom6"


def test_obdata_tangential_shape_and_time_interpolation():
    from forcing.boundary_time import interpolated_boundary
    n, nz = 4, 3
    zw = np.linspace(0.0, -30.0, nz + 1)[:, None].repeat(n, 1)
    mk = lambda a: OBData(side="west", T=np.full((nz, n), 7.0), u_normal=np.zeros((nz, n)), zw=zw,  # noqa: E731
                          u_tangential=np.full((nz, n + 1), a))
    obc.validate_state({"west": mk(1.0)}, ("west",), (n, 6))
    bad = mk(1.0)
    bad.u_tangential = np.zeros((nz, n))
    with pytest.raises(ValueError, match="u_tangential"):
        obc.validate_state({"west": bad}, ("west",), (n, 6))
    f = interpolated_boundary([0.0, 10.0], [{"west": mk(1.0)}, {"west": mk(3.0)}])
    np.testing.assert_allclose(f(5.0)["west"].u_tangential, 2.0)


def test_exterior_tangential_on_corner_layers():
    ny, nz = 4, 3
    flags = np.zeros((ny + 1, 6), bool)
    flags[1:-1, 0] = True
    hq = np.zeros((nz, ny + 1, 6))
    hq[:, 1:-1, 0] = 10.0
    zw = np.linspace(0.0, -30.0, 7)[:, None].repeat(ny, 1)                   # 6 data layers of 5 m
    zc = 0.5 * (zw[1:, 0] + zw[:-1, 0])
    ut = (1.0 + 0.01 * zc)[:, None].repeat(ny + 1, 1)                          # linear in z: exact layer means
    ob = OBData(side="west", T=np.zeros((6, ny)), u_normal=np.zeros((6, ny)), zw=zw, u_tangential=ut)
    out = obc.exterior_tangential({"west": ob}, hq, flags, 2)
    np.testing.assert_allclose(out[:, 2, 0], 1.0 + 0.01 * np.array([-25.0, -15.0, -5.0]), rtol=1e-12)
    assert np.all(out[:, ~flags] == 0.0)


def _advance(m, st, n):
    for _ in range(n):
        st = m.step(st)
    return st


def _random_flow(m, seed=3):
    st = m.initial_state(np.zeros(m.H.shape), np.full((m.n,) + m.H.shape, 10.0))
    rng = np.random.default_rng(seed)
    st.u = 0.05 * rng.standard_normal(st.u.shape)
    st.v = 0.05 * rng.standard_normal(st.v.shape)
    return st


def test_off_is_bitwise_and_on_changes_the_edges():
    sb.set_threads(1)
    a = _box(obc_tangential="none")
    hg = create_cartesian_grid(nx=12, ny=10, dx=2000.0, dy=2000.0, f0=1e-4)
    b = SigmaBaroclinic(hg, np.full((10, 12), 30.0), dt=120.0, n_layers=3, rho0=RHO0, t_coef=TCOEF, z0=None,
                        closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(3, 10.0)), pgf="fv", momentum="vector_invariant",
                        smagorinsky=0.15, open_boundaries=("west", "south", "east"), g=G)
    sa, sb_ = _advance(a, _random_flow(a), 5), _advance(b, _random_flow(b), 5)
    for f in ("u", "v", "eta", "h", "T"):
        np.testing.assert_array_equal(getattr(sa, f), getattr(sb_, f))
    assert sa.obc_rtan is None
    c = _box()
    sc = _advance(c, _random_flow(c), 5)
    assert sc.obc_rtan is not None and sc.obc_rtan.shape == (3, 11, 13)
    assert not np.array_equal(sc.v[:, :, 0], sa.v[:, :, 0])
    assert np.all(np.isfinite(sc.u)) and np.all(np.isfinite(sc.v))


def test_restart_bit_for_bit_with_tangential(tmp_path):
    from core.restart import read_restart, write_restart
    sb.set_threads(1)
    m = _box()
    ref = _advance(m, _random_flow(m), 6)
    m1 = _box()
    s1 = _advance(m1, _random_flow(m1), 3)
    assert np.any(s1.obc_rtan > 0)
    write_restart(tmp_path / "r.nc", m1, s1)
    m2 = _box()
    s2 = _advance(m2, read_restart(tmp_path / "r.nc", m2), 3)
    for f in ("u", "v", "eta", "h", "T", "obc_rtan"):
        np.testing.assert_array_equal(getattr(s2, f), getattr(ref, f))
    with pytest.raises(ValueError, match="obc_tangential"):
        read_restart(tmp_path / "r.nc", _box(obc_tangential="none"))


def test_one_thread_equals_five():
    out = []
    for t in (1, 5):
        sb.set_threads(t)
        m = _box()
        out.append(_advance(m, _random_flow(m), 4))
    sb.set_threads(1)
    for f in ("u", "v", "eta", "h", "T", "obc_rtan"):
        np.testing.assert_array_equal(getattr(out[0], f), getattr(out[1], f))


# ------------------------------------------------------------------ an x-uniform jet entering through the west edge
def _jet_model(mode: str, nx=40, ny=30, nz=4, D=200.0, dx=5000.0, f0=1e-4, U=0.5, L=20e3):
    """f-plane channel, walls south/north, 'radiating' west and east edges whose exterior is the same steady
    geostrophic jet u(y) = U exp(−(y − y₀)²/L²) (η in the model's own discrete balance at an interior column, the
    tangential exterior at rest); the initial state is the exact x-uniform jet."""
    y = (np.arange(ny) + 0.5) * dx
    u_y = U * np.exp(-((y - y.mean()) / L) ** 2)
    zw = np.repeat(np.linspace(0.0, -D - 1.0, 11)[:, None], ny, 1)

    def build(eta_y):
        def state(t):
            return {e: OBData(side=e, T=np.full((10, ny), 10.0), u_normal=np.repeat(u_y[None], 10, 0), zw=zw,
                              eta=eta_y.copy(), ubar=u_y.copy(), u_tangential=np.zeros((10, ny + 1)))
                    for e in ("west", "east")}
        hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=f0)
        return SigmaBaroclinic(hg, np.full((ny, nx), D), dt=300.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                               closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                               regrid_params=dict(dz_top=np.full(nz, D / nz)), pgf="fv", momentum="vector_invariant",
                               smagorinsky=0.15, open_boundaries=("west", "east"), open_boundary_state=state,
                               obc_tau_in=3 * 86400.0, obc_tau_out=360 * 86400.0, obc_tangential=mode, g=G)

    def start(m, eta_y):
        st = m.initial_state(np.repeat(eta_y[:, None], nx, 1), np.full((nz, ny, nx), 10.0))
        st.u[:] = u_y[None, :, None]
        return st

    eta = -(f0 / G) * np.cumsum(u_y) * dx
    for _ in range(3):                                          # discrete geostrophic η at the middle column
        m = build(eta - eta.mean())
        terms, _, _ = m._momentum_terms(start(m, eta - eta.mean()))
        cv = {n: tv for n, tu, tv, _ in terms}["vector_invariant"][0, :, nx // 2]
        eta = np.concatenate([[0.0], np.cumsum(cv[1:-1] * dx / G)])
    eta -= eta.mean()
    m = build(eta)
    return m, start(m, eta), u_y


def test_x_uniform_jet_has_no_edge_tendency_with_mom6():
    # With the computed corner vorticity/strain and the open-face velocity in the tension (MOM6, mask2dCu = 1 on
    # OBC faces), the x-uniform jet feels at the edge column exactly what it feels in the interior: every explicit
    # term at the first interior faces equals its value at the middle column.  With 'none' (ζ = 0, D_S = 0 at the
    # edge corners) the v tendency does not; the u tendency does since the open-face velocity enters the tension in
    # every mode (class-(a) fix of 2026-11-03, tests/test_hor_visc_open_faces.py; before it the open face was a
    # wall in the 'none' tension and u differed too).
    sb.set_threads(1)
    res = {}
    for mode in ("none", "mom6"):
        m, st, _ = _jet_model(mode)
        obc_tan = m._ob.tangential_velocity(st.u, st.v, None, None) if mode == "mom6" else None
        terms, _, _ = m._momentum_terms(st, obc_tan)
        gu = sum(tu for _, tu, _, _ in terms)
        gv = sum(tv for _, _, tv, _ in terms)
        nx = st.u.shape[2] - 1
        res[mode] = (np.abs(gu[:, :, 1] - gu[:, :, nx // 2]).max(), np.abs(gv[:, 1:-1, 0] - gv[:, 1:-1, nx // 2]).max())
    scale = 0.5 * 1e-4                                          # f U [m s⁻²]
    assert res["mom6"][0] < 1e-9 * scale and res["mom6"][1] < 1e-9 * scale, res
    assert res["none"][0] < 1e-9 * scale and res["none"][1] > 1e-2 * scale, res


def test_jet_entering_through_the_west_edge_keeps_its_profile():
    # Three inertial periods (52 h): with 'mom6' the jet next to the inflow edge stays within 2 % (RMS, first
    # interior u-faces) and 5 % (fifth faces) of the exterior profile; with 'none' it degrades by > 10 % / > 30 %
    # (measured 0.7 % / 2.9 % against 19 % / 49 %; spurious cross-jet v 0.22 m/s at the edge against 0.006; after
    # the open-face tension fix of 2026-11-03 'none' gives 13.7 % / 33.7 % and 0.15 m/s, the rest is ζ = D_S = 0).
    sb.set_threads(1)
    err = {}
    for mode in ("none", "mom6"):
        m, st, u_y = _jet_model(mode)
        for _ in range(int(round(3 * 2 * np.pi / 1e-4 / m.dt))):
            st = m.step(st)
        rel = lambda a: np.sqrt(np.mean((a - u_y) ** 2) / np.mean(u_y ** 2))   # noqa: E731
        err[mode] = (rel(st.u[-1, :, 1]), rel(st.u[-1, :, 5]), np.abs(st.v[:, :, 0]).max())
    assert err["mom6"][0] < 0.02 and err["mom6"][1] < 0.05, err
    assert err["none"][0] > 0.10 and err["none"][1] > 0.30, err
    assert err["mom6"][2] < 0.1 * err["none"][2], err


def test_pulse_leaves_through_the_open_edges_without_growth():
    # A Gaussian η pulse at rest radiates out through four 'radiating' edges (Flather, resting exterior): the
    # energy left after the wave has gone is < 1 % of the start and does not grow, and 'mom6' leaves the same
    # energy as 'none' to 1 %.
    sb.set_threads(1)
    nx = ny = 40
    nz, D, dx = 3, 100.0, 5000.0
    x = (np.arange(nx) + 0.5) * dx
    left = {}
    for mode in ("none", "mom6"):
        hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=1e-4)
        m = SigmaBaroclinic(hg, np.full((ny, nx), D), dt=120.0, n_layers=nz, rho0=RHO0, t_coef=TCOEF, z0=None,
                            closure="constant", nu_const=1e-4, vertical_coordinate="ale", regrid="zstar",
                            regrid_params=dict(dz_top=np.full(nz, D / nz)), pgf="fv", momentum="vector_invariant",
                            smagorinsky=0.15, open_boundaries=("west", "east", "south", "north"),
                            obc_tau_in=3 * 86400.0, obc_tau_out=360 * 86400.0, obc_tangential=mode, g=G)
        r2 = (x[None] - 0.3 * x.max()) ** 2 + (x[:, None] - 0.4 * x.max()) ** 2
        st = m.initial_state(0.2 * np.exp(-r2 / 20e3**2), np.full((nz, ny, nx), 10.0))
        E = []
        for k in range(900):
            st = m.step(st)
            if k % 50 == 0:
                E.append(0.5 * ((st.u**2).sum() + (st.v**2).sum()) * D / nz + 0.5 * G * (st.eta**2).sum())
        E = np.array(E)
        assert E[len(E) // 2:].max() < 0.01 * E[0], (mode, E)
        assert np.all(np.diff(E[len(E) // 2:]) <= 1e-9 * E[0]), (mode, E)
        left[mode] = E[-1]
    assert abs(left["mom6"] / left["none"] - 1.0) < 0.01, left


def test_provider_tangential_reaches_the_corners():
    ny, nx, nz = 10, 12, 3
    zw = np.repeat(np.linspace(0.0, -31.0, 6)[:, None], ny, 1)

    def state(t):
        return {"west": OBData(side="west", T=np.full((5, ny), 10.0), u_normal=np.zeros((5, ny)), zw=zw,
                               u_tangential=np.full((5, ny + 1), 0.2))}
    m = _box(open_=("west", "east"), state=state)
    st = m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 10.0))
    hu, hv = m.face_thickness(st.h)
    ext = m._ob.evaluate(0.0, hu, hv, None, None)
    q = m._ob._tan_q
    np.testing.assert_allclose(ext["ut_q"][:, q[:, 0], 0], 0.2, rtol=1e-12)
    assert np.all(ext["ut_q"][:, :, -1] == 0.0)                  # east: no data, resting exterior
    flags, vt = m._ob.tangential_velocity(st.u, st.v, None, ext)
    g = m.dt / (m.obc_tau_in + m.dt)
    np.testing.assert_allclose(vt[:, q[:, 0], 0], g * 0.2, rtol=1e-12)
