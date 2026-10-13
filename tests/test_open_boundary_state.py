"""Tests of the time-varying open-boundary exterior state (spec 2026-10-12-baroclinic-obc-internal-wave §1)."""
from __future__ import annotations

import numpy as np
import pytest

from boundaries.obcs import OBData


def _ob(val: float, n: int = 3, nz: int = 4) -> OBData:
    zw = np.linspace(0.0, -100.0, nz + 1)[:, None].repeat(n, 1)
    return OBData(side="west", T=np.full((nz, n), val), u_normal=np.full((nz, n), 2 * val), zw=zw,
                  eta=np.full(n, 0.1 * val), ubar=np.full(n, 0.5 * val))


def test_obdata_has_its_own_vertical_grid_and_ubar():
    ob = _ob(1.0)
    assert ob.zw.shape == (5, 3) and ob.ubar.shape == (3,)
    assert OBData(side="east").zw is None and OBData(side="east").ubar is None   # Phase-1 use unchanged


def test_interpolated_boundary_is_linear_in_time_and_constant_outside():
    from forcing.boundary_time import interpolated_boundary
    f = interpolated_boundary([0.0, 100.0], [{"west": _ob(1.0)}, {"west": _ob(3.0)}])
    mid = f(25.0)["west"]
    np.testing.assert_allclose(mid.T, 1.5)
    np.testing.assert_allclose(mid.u_normal, 3.0)
    np.testing.assert_allclose(mid.eta, 0.15)
    np.testing.assert_allclose(mid.ubar, 0.75)
    np.testing.assert_allclose(mid.zw, _ob(1.0).zw)
    np.testing.assert_allclose(f(-5.0)["west"].T, 1.0)
    np.testing.assert_allclose(f(500.0)["west"].T, 3.0)


def test_interpolated_boundary_rejects_inconsistent_states():
    from forcing.boundary_time import interpolated_boundary
    with pytest.raises(ValueError, match="increasing"):
        interpolated_boundary([10.0, 0.0], [{"west": _ob(1.0)}, {"west": _ob(2.0)}])
    with pytest.raises(ValueError, match="edges"):
        interpolated_boundary([0.0, 1.0], [{"west": _ob(1.0)}, {"east": _ob(2.0)}])


from core.dynamics import open_boundary as obc
from core.vertical import remap as rm


def _west(ny: int = 2, nz: int = 40, depth: float = 100.0, T=None, u=None) -> OBData:
    zw = np.linspace(0.0, -depth, nz + 1)[:, None].repeat(ny, 1)
    zc = 0.5 * (zw[1:] + zw[:-1])
    T = 10.0 + 0.1 * zc if T is None else T
    u = np.cos(np.pi * zc / depth) if u is None else u
    return OBData(side="west", T=T, u_normal=u, zw=zw)


def test_validate_state_errors():
    with pytest.raises(ValueError, match="not in open_boundaries"):
        obc.validate_state({"west": _west()}, ("east",), (2, 5))
    with pytest.raises(ValueError, match="shape"):
        obc.validate_state({"west": _west(ny=3)}, ("west",), (2, 5))           # edge length 2, data 3
    bad = _west()
    bad.zw = bad.zw[::-1]
    with pytest.raises(ValueError, match="decrease"):
        obc.validate_state({"west": bad}, ("west",), (2, 5))


def test_source_must_reach_the_edge_bed():
    H = np.full((2, 5), 150.0)                                          # deeper than the 100 m data
    h = np.full((5, 2, 5), 30.0)
    mask = np.zeros((2, 5), bool)
    mask[:, 0] = True
    with pytest.raises(ValueError, match="west.*bed"):
        obc.exterior_tracer({"west": _west()}, H, h, mask, rm.PPM_H4)


def test_exterior_tracer_on_layers_and_sponge_band():
    # linear T(z) on the data grid → exact layer means on any model layers; the sponge band takes the edge
    # profile, extended or clipped to each column's own bed (review focus 2)
    H = np.array([[100.0, 120.0, 75.0, 100.0, 100.0]] * 2)              # 75 m: a data interface
    nz = 7
    h = (H / nz)[None].repeat(nz, 0)
    mask = np.zeros((2, 5), bool)
    mask[:, :3] = True
    T = obc.exterior_tracer({"west": _west()}, H, h, mask, rm.PPM_H4)
    zc = -H[None] + np.cumsum(h, 0) - 0.5 * h
    inner = slice(1, -1)                                                # end cells are PCM (remap.py)
    np.testing.assert_allclose(T[inner, :, 0], (10.0 + 0.1 * zc)[inner, :, 0], atol=1e-10)
    np.testing.assert_allclose(T[inner, :, 2], (10.0 + 0.1 * zc)[inner, :, 2], atol=1e-10)   # clipped at 75 m
    assert np.isfinite(T[:, :, 1]).all() and (T[:, :, 3:] == 0).all()


def test_nearest_edge_wins_in_corners():
    H = np.full((4, 4), 100.0)
    h = np.full((2, 4, 4), 50.0)
    mask = np.ones((4, 4), bool)
    south = OBData(side="south", T=np.full((4, 4), 5.0), u_normal=np.zeros((4, 4)),
                   zw=np.linspace(0, -100, 5)[:, None].repeat(4, 1))
    west = _west(ny=4, nz=4, T=np.full((4, 4), 1.0), u=np.zeros((4, 4)))
    T = obc.exterior_tracer({"west": west, "south": south}, H, h, mask, rm.PPM_H4)
    assert (T[:, 3, 0] == 1.0).all() and (T[:, 0, 3] == 5.0).all()          # nearest edge
    assert (T[:, 0, 0] == 1.0).all()                                          # tie: first in EDGES order


def test_exterior_velocity_has_zero_weighted_mean():
    ny, nx, N = 2, 4, 6
    Hf = np.full((ny, nx + 1), 100.0)
    hf = (Hf / N)[None].repeat(N, 0)
    w = hf / hf.sum(0)
    mask = np.zeros((ny, nx + 1), bool)
    mask[:, :2] = True
    up = obc.exterior_velocity({"west": _west(u=_west().u_normal + 0.3)}, "u", Hf, hf, mask, w, rm.PPM_H4)
    np.testing.assert_allclose((w * up).sum(0)[:, :2], 0.0, atol=1e-14)        # the 0.3 depth mean is removed
    assert np.abs(up[:, :, 0]).max() > 0.5 and (up[:, :, 2:] == 0).all()


def test_flather_targets_at_the_open_faces():
    ob = _west()
    ob.eta, ob.ubar = np.array([0.1, 0.2]), np.array([0.01, 0.02])
    t = obc.flather_targets({"west": ob}, (2, 6), (3, 5))
    np.testing.assert_allclose(t["flather_eta"][0][:, 0], [0.1, 0.2])
    np.testing.assert_allclose(t["flather_ubar"][:, 0], [0.01, 0.02])
    assert (t["flather_eta"][0][:, 1:] == 0).all() and (t["flather_vbar"] == 0).all()


def test_points_nearer_an_open_edge_without_data_keep_the_resting_exterior():
    # open west and east, data for west only: the east half (nearer east) must not take the west profile
    H = np.full((1, 6), 100.0)
    h = np.full((5, 1, 6), 20.0)
    mask = np.ones((1, 6), bool)
    state = {"west": _west(ny=1)}
    T = obc.exterior_tracer(state, H, h, mask, rm.PPM_H4, open_edges=("west", "east"))
    own = obc.owned_by_state(state, ("west", "east"), (1, 6))
    assert own[0, :3].all() and not own[0, 3:].any()
    assert np.abs(T[:, 0, :3]).min() > 0 and (T[:, 0, 3:] == 0).all()
    Hf = np.full((1, 7), 100.0)
    hf = np.full((5, 1, 7), 20.0)
    up = obc.exterior_velocity(state, "u", Hf, hf, np.ones((1, 7), bool), hf / 100.0, rm.PPM_H4,
                               open_edges=("west", "east"))
    assert np.abs(up[:, 0, 0]).max() > 0 and (up[:, 0, 4:] == 0).all()


def test_tangential_faces_in_another_edges_sponge_band_get_no_exterior():
    # west and south open, data for west only: u-faces in the south band, far from the west edge, are nearest
    # the south edge (tangential to u) and must keep the resting exterior, not the west u_normal profile
    ny, nx, N = 8, 16, 5
    Hf = np.full((ny, nx + 1), 100.0)
    hf = np.full((N, ny, nx + 1), 20.0)
    up = obc.exterior_velocity({"west": _west(ny=ny)}, "u", Hf, hf, np.ones((ny, nx + 1), bool), hf / 100.0,
                               rm.PPM_H4, open_edges=("west", "south"))
    assert np.abs(up[:, 4, 0]).max() > 0                                   # west edge face: exterior profile
    assert (up[:, :2, nx // 2] == 0).all()                                 # south band, nearest south: rest


# ---------------------------------------------------------------- perf part 2, task 2: exterior only where used
def _nested_model(sponge_tracer=False):
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    from core.grid.horizontal import create_cartesian_grid
    ny, nx, n = 4, 20, 6
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=1000.0, dy=1000.0, f0=1e-4)
    zw = np.linspace(0.0, -50.0, 6)[:, None].repeat(ny, 1)

    def state(t):
        a = np.sin(2 * np.pi * t / 3600.0)
        return {"west": OBData(side="west", T=np.full((5, ny), 12.0 + a), u_normal=np.full((5, ny), 0.01 * a),
                               zw=zw, eta=np.full(ny, 0.02 * a), ubar=np.full(ny, 0.01 * a))}
    return SigmaBaroclinic(hg, np.full((ny, nx), 50.0), dt=30.0, n_layers=n, rho0=1000.0, t_coef=0.2, z0=None,
                           closure="constant", vertical_coordinate="ale", regrid="zstar",
                           regrid_params=dict(dz_top=np.full(n, 50.0 / n)), open_boundaries=("west", "east"),
                           sponge_cells=5, sponge_tau=(600.0, 60.0), sponge_tracer=sponge_tracer,
                           open_boundary_state=state)


def _rest(m):
    T = 10.0 + 0.1 * np.arange(m.n)[:, None, None] * np.ones((1,) + m.H.shape)
    return np.zeros(m.H.shape), T


def test_exterior_tracer_equals_the_full_domain_version_at_the_masked_columns():
    rng = np.random.default_rng(3)
    ny, nx, n = 6, 9, 5
    H = np.full((ny, nx), 100.0)
    h = np.full((n, ny, nx), 20.0) * rng.uniform(0.9, 1.1, (1, ny, nx))
    h *= H[None] / h.sum(0)
    zw = np.linspace(0.0, -100.0, 5)[:, None].repeat(ny, 1)
    st = {"west": OBData(side="west", T=rng.uniform(10, 20, (4, ny)), u_normal=np.zeros((4, ny)), zw=zw)}
    band = np.zeros((ny, nx), bool)
    band[:, :3] = True
    full = obc._exterior_tracer_full(st, H, h, band, rm.PPM_H4, ("west",))
    edge = np.zeros((ny, nx), bool)
    edge[:, 0] = True
    np.testing.assert_array_equal(obc.exterior_tracer(st, H, h, band, rm.PPM_H4, ("west",)), full)
    np.testing.assert_array_equal(obc.exterior_tracer(st, H, h, edge, rm.PPM_H4, ("west",))[:, :, 0], full[:, :, 0])


def test_exterior_tracer_band_kept_with_tracer_sponge():
    m = _nested_model(sponge_tracer=True)
    assert m._ext_cols.sum() == m._bc_mask.sum()
    m2 = _nested_model(sponge_tracer=False)
    assert m2._ext_cols.sum() < m2._bc_mask.sum() and m2._ext_cols[:, 0].all()


def test_nested_model_is_bit_identical_with_the_restricted_exterior():
    a, b = _nested_model(), _nested_model()
    b._ext_cols = b._bc_mask.copy()                                  # the old behaviour: the whole band
    sa, sb = a.initial_state(*_rest(a)), b.initial_state(*_rest(b))
    for _ in range(10):
        sa, sb = a.step(sa), b.step(sb)
    for k in ("eta", "u", "v", "T", "h"):
        assert np.array_equal(getattr(sa, k), getattr(sb, k)), k


def test_exterior_velocity_equals_the_full_domain_version():
    rng = np.random.default_rng(8)
    ny, nx, n = 5, 8, 4
    Hf = np.full((ny, nx + 1), 80.0)
    hf = rng.uniform(5.0, 30.0, (n, ny, nx + 1))
    hf *= Hf[None] / hf.sum(0)
    w = hf / hf.sum(0)
    zw = np.linspace(0.0, -80.0, 6)[:, None].repeat(ny, 1)
    st = {"west": OBData(side="west", T=np.full((5, ny), 10.0), u_normal=rng.normal(0, 0.1, (5, ny)), zw=zw)}
    band = np.zeros((ny, nx + 1), bool)
    band[:, :4] = True
    np.testing.assert_array_equal(obc.exterior_velocity(st, "u", Hf, hf, band, w, rm.PPM_H4, ("west",)),
                                  obc._exterior_velocity_full(st, "u", Hf, hf, band, w, rm.PPM_H4, ("west",)))
