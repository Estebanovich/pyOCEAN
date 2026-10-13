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
