"""
Tests for core/vertical/ale.py — layer geometry of the ALE coordinate on the C grid:
face thickness (thinner of the two cells, MITgcm hFacW), face fractions, zero velocity in
thin bed layers (HYCOM hybgen) and the Lagrangian continuity guard.
"""

from __future__ import annotations

import numpy as np
import pytest

from core.vertical import ale


def test_face_thickness_is_the_thinner_cell_and_fractions_are_dsigma_for_sigma_layers():
    ds = np.array([0.1, 0.2, 0.3, 0.4])
    D = np.array([[10.0, 30.0, 20.0]])
    h = D[None] * ds[:, None, None]
    hu, hv = ale.face_thickness(h, np.ones((1, 3), bool))
    np.testing.assert_allclose(hu[:, 0, 1], 10.0 * ds)                  # min of the two columns
    np.testing.assert_allclose(hu[:, 0, 0], h[:, 0, 0])                 # domain-edge face: inner cell
    np.testing.assert_allclose(hu[:, 0, 3], h[:, 0, 2])
    np.testing.assert_allclose(ale.face_fractions(hu), np.broadcast_to(ds[:, None, None], hu.shape), rtol=1e-14)
    assert hv.shape == (4, 2, 3)


def test_face_next_to_land_has_no_thickness():
    h = np.full((3, 1, 3), 5.0)
    active = np.array([[True, False, True]])
    hu, _ = ale.face_thickness(h, active)
    assert (hu[:, 0, 1] == 0).all() and (hu[:, 0, 2] == 0).all()
    np.testing.assert_allclose(ale.face_fractions(hu)[:, 0, 1], 1.0 / 3)  # defined, no NaN


def test_zstar_step_face_has_vanished_bed_layers():
    # z* across a 600 → 164 m step: layers below the shallow bed are 1 mm on one side, so the face
    # carries nothing there (MITgcm partial cells)
    deep = np.full(6, 100.0)                                             # 600 m, bed first
    shallow = np.array([1e-3, 1e-3, 1e-3, 63.997, 50.0, 50.0])           # 164 m, three collapsed layers
    h = np.stack([deep, shallow], axis=1)[:, None, :]
    hu, _ = ale.face_thickness(h, np.ones((1, 2), bool))
    np.testing.assert_allclose(hu[:3, 0, 1], 1e-3)
    assert ale.face_fractions(hu)[:3, 0, 1].max() < 1e-4


def test_zero_thin_bed_velocity_keeps_the_depth_mean():
    hf = np.array([1e-3, 5e-3, 2.0, 3.0, 5.0])[:, None, None]
    w = ale.face_fractions(hf)
    u = np.array([0.3, -0.2, 0.1, 0.05, 0.02])[:, None, None]
    out = ale.zero_thin_bed_velocity(u, hf, w)
    assert out[0] == 0 and out[1] == 0                                   # thin run from the bed
    np.testing.assert_allclose((w * out).sum(0), (w * u).sum(0), rtol=1e-14)
    np.testing.assert_allclose(np.diff(out[2:, 0, 0]), np.diff(u[2:, 0, 0]))   # shear above unchanged


def test_zero_thin_bed_velocity_ignores_thin_layers_above_a_thick_one_and_all_thin_columns():
    hf = np.array([2.0, 1e-3, 3.0])[:, None, None]
    u = np.array([0.1, 0.2, 0.3])[:, None, None]
    np.testing.assert_array_equal(ale.zero_thin_bed_velocity(u, hf, ale.face_fractions(hf)), u)
    hf = np.full((3, 1, 1), 1e-3)                                        # very shallow: all thin
    np.testing.assert_array_equal(ale.zero_thin_bed_velocity(u, hf, ale.face_fractions(hf)), u)


def test_lagrangian_thickness_updates_and_rejects_an_emptied_layer():
    h = np.full((2, 1, 1), 1.0)
    area = np.full((1, 1), 100.0)
    out = ale.lagrangian_thickness(h, np.array([10.0, -10.0])[:, None, None], 2.0, area)
    np.testing.assert_allclose(out[:, 0, 0], [0.8, 1.2])
    with pytest.raises(RuntimeError, match="layer 0"):
        ale.lagrangian_thickness(h, np.array([100.0, 0.0])[:, None, None], 2.0, area)
    tiny = ale.lagrangian_thickness(h, np.array([50.0 + 1e-11, 0.0])[:, None, None], 2.0, area)
    assert tiny[0, 0, 0] == 0.0                                          # round-off below zero → 0


def _q(h, H=None, clip=False, active=None):
    """Transport thicknesses (eastward, westward) at the u-faces of bed-first layers h."""
    H = h.sum(axis=0) if H is None else H
    active = np.ones(H.shape, bool) if active is None else active
    return ale.face_transport_thickness(h, H, active, clip)


def test_transport_thickness_without_clip_is_the_upwind_cell_and_its_mean_is_dsigma_for_sigma():
    ds = np.array([0.1, 0.2, 0.3, 0.4])
    D = np.array([[10.0, 30.0, 20.0]])
    h = D[None] * ds[:, None, None]
    qe, qw = _q(h)
    np.testing.assert_allclose(qe[:, 0, 1], h[:, 0, 0])
    np.testing.assert_allclose(qw[:, 0, 1], h[:, 0, 1])
    np.testing.assert_allclose(ale.face_fractions(0.5 * (qe + qw)), np.broadcast_to(ds[:, None, None], qe.shape),
                               rtol=1e-14)
    h = np.array([[[8.0, 0.0]], [[0.0, 8.0]]])                         # lock: dense left, light right
    qe, qw = _q(h, clip=True)                                            # flat bed: the clip changes nothing
    np.testing.assert_allclose(qe[:, 0, 1], [8.0, 0.0])
    np.testing.assert_allclose(qw[:, 0, 1], [0.0, 8.0])
    qe, qw = _q(h, active=np.array([[True, False]]))
    assert (qe[:, 0, 1:] == 0).all() and (qw[:, 0, 1:] == 0).all()     # land


def test_transport_thickness_clips_layers_below_the_shallower_bed():
    # z* across a 600 → 164 m step (HYCOM cnuity.F90:252-256): water facing rock does not move;
    # the partial layer keeps the part above the shallow bed (MITgcm hFacW = min)
    deep = np.full(6, 100.0)                                             # 600 m, bed first
    shallow = np.array([1e-3, 1e-3, 1e-3, 1e-3, 63.996, 100.0])          # 164 m, z* layers of 100 m
    h = np.stack([deep, shallow], axis=1)[:, None, :]
    H = np.array([[600.0, 164.0]])
    qe, qw = _q(h, H, clip=True)
    np.testing.assert_allclose(qe[:, 0, 1], [0, 0, 0, 0, 64.0, 100.0], atol=1e-9)
    np.testing.assert_allclose(qw[:, 0, 1], shallow)                    # shallow side: above its own bed
    hbar = 0.5 * (qe + qw)[:, 0, 1]
    assert hbar[:4].max() < 1e-3                                         # facing rock: thin, no velocity
    assert abs(hbar[4] - 64.0) < 1e-2                                    # partial layer: the min (hFacW)
    qe, _ = _q(h, H, clip=False)                                         # σ: no clip
    np.testing.assert_allclose(qe[:, 0, 1], deep)


def test_upwind_transport_takes_the_thickness_of_the_cell_the_water_comes_from():
    # lock exchange: dense layer (k=0) only in the left cell flows right, light layer only in the
    # right cell flows left; with min(h_L, h_R) both transports would vanish
    h = np.array([[[8.0, 0.0]], [[0.0, 8.0]]])
    qe, qw = _q(h, clip=True)
    ut = np.zeros((2, 1, 3))
    ut[:, 0, 1] = [0.2, -0.2]
    L = np.full((1, 3), 100.0)
    Qbt = np.zeros((1, 3))                                               # exchange flow, no net flux
    Q = ale.upwind_transport_x(qe, qw, ut, L, Qbt)
    np.testing.assert_allclose(Q[:, 0, 1], [100.0 * 0.2 * 8.0, -100.0 * 0.2 * 8.0])
    assert (Q[:, 0, [0, 2]] == 0).all()


def test_upwind_transport_adds_up_to_the_free_surface_flux_without_emptying_a_vanished_layer():
    # the layer fluxes are corrected to the barotropic flux of the free-surface solve (as HYCOM
    # cnuity.F90:585-620): the residual is shared in proportion to the layers of its upwind cell,
    # so a layer that is empty there receives none of it
    rng = np.random.default_rng(3)
    h = rng.uniform(0.5, 5.0, (5, 2, 4))
    h[1, :, 1] = 0.0
    qe, qw = _q(h)
    ut = rng.normal(0.0, 0.1, (5, 2, 5))
    ut[:, :, 2] = 0.0                                                    # the whole flux there is residual
    L = np.full((2, 5), 50.0)
    Qbt = rng.normal(0.0, 20.0, (2, 5))
    Qbt[:, 2] = np.abs(Qbt[:, 2]) + 1.0                                  # flows out of cell 1 eastward
    Q = ale.upwind_transport_x(qe, qw, ut, L, Qbt)
    np.testing.assert_allclose(Q.sum(0), Qbt, rtol=1e-12, atol=1e-12)
    assert (Q[1, :, 2] == 0).all()


def test_upwind_transport_is_zero_next_to_land():
    h = np.full((2, 1, 3), 4.0)
    qe, qw = _q(h, active=np.array([[True, False, True]]))
    ut = np.full((2, 1, 4), 0.1)
    Q = ale.upwind_transport_x(qe, qw, ut, np.full((1, 4), 10.0), np.full((1, 4), 8.0))
    assert (Q[:, 0, 1:3] == 0).all()


# ---------------------------------------------------------------- B2b: PPM thickness flux (MOM6 continuity_PPM)
def _ppm_case(h, u, dt=10.0, dx=100.0, L=10.0, clip=False, H=None, active=None):
    n, ny, nx = h.shape
    active = np.ones((ny, nx), bool) if active is None else active
    H = h.sum(axis=0) if H is None else H
    qe, qw = ale.face_transport_thickness(h, H, active, clip)
    dx_t = np.full((ny, nx), dx)
    Lf = np.full((ny, nx + 1), L)
    return qe, qw, dx_t, Lf, active


def test_ppm_edges_are_exact_for_a_linear_thickness_and_positive_for_vanished_cells():
    x = np.arange(8) + 0.5
    h = (2.0 + 0.5 * x)[None, None, :]
    hw, he = ale.ppm_edges_x(h, np.ones((1, 8), bool))
    # cells whose neighbours have exact slopes (the edge cells have zero slope, MOM6 2899-2913)
    np.testing.assert_allclose(he[0, 0, 2:-2], 2.0 + 0.5 * (x[2:-2] + 0.5), rtol=1e-13)
    np.testing.assert_allclose(hw[0, 0, 2:-2], 2.0 + 0.5 * (x[2:-2] - 0.5), rtol=1e-13)
    rng = np.random.default_rng(1)
    h = rng.uniform(0.0, 5.0, (6, 3, 10))
    h[rng.uniform(size=h.shape) < 0.3] = 0.0                              # vanished cells
    hw, he = ale.ppm_edges_x(h, np.ones((3, 10), bool))
    assert hw.min() >= -1e-14 and he.min() >= -1e-14
    np.testing.assert_array_equal(hw[h == 0], 0.0)                       # flattened, nothing to export
    np.testing.assert_array_equal(he[h == 0], 0.0)


def test_ppm_transport_is_the_swept_mean_of_the_parabola():
    # linear thickness, u > 0: the swept volume is the last fraction C of the donor cell,
    # mean thickness = h_E − b·C·dx/2 (MOM6 flux_elem 1182)
    x = np.arange(8) + 0.5
    b = 0.5                                                              # per cell
    h = (2.0 + b * x)[None, None, :]
    u = np.zeros((1, 1, 9))
    u[0, 0, 4] = 3.0                                                     # C = 3·10/100 = 0.3
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    Qbt = np.zeros((1, 9))
    Qbt[0, 4] = Lf[0, 4] * 3.0 * (2.0 + b * 4.0 - b * 0.3 / 2)           # what the swept mean gives
    Q = ale.ppm_transport_x(h, u, Lf, Qbt, qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q[0, 0, 4], Qbt[0, 4], rtol=1e-12)
    assert (Q[0, 0, [0, 8]] == 0).all()


def test_ppm_transport_adds_up_to_the_free_surface_flux_with_a_uniform_correction():
    rng = np.random.default_rng(4)
    h = rng.uniform(0.2, 5.0, (6, 2, 7))
    h[2, :, 3] = 0.0
    u = rng.normal(0.0, 0.5, (6, 2, 8))
    u[..., [0, -1]] = 0.0
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    F0, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    Qbt = F0.sum(axis=0) + rng.normal(0.0, 5.0, (2, 8))                  # needs a correction du ≠ 0
    Qbt[:, [0, -1]] = 0.0
    Q = ale.ppm_transport_x(h, u, Lf, Qbt, qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q.sum(axis=0), Qbt, rtol=1e-12, atol=1e-12)
    assert np.isfinite(Q).all()


def test_ppm_transport_without_residual_keeps_the_uncorrected_fluxes():
    rng = np.random.default_rng(5)
    h = rng.uniform(0.2, 5.0, (4, 1, 6))
    u = rng.normal(0.0, 0.5, (4, 1, 7))
    u[..., [0, -1]] = 0.0
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    F0, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    Q = ale.ppm_transport_x(h, u, Lf, F0.sum(axis=0), qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q, F0, rtol=1e-12, atol=1e-12)


def test_ppm_transport_next_to_land_and_on_the_edge_is_zero_and_finite():
    # review focus 4
    h = np.full((3, 1, 5), 4.0)
    act = np.array([[True, True, False, True, True]])
    u = np.full((3, 1, 6), 0.2)
    qe, qw, dx_t, Lf, _ = _ppm_case(h, u, active=act)
    Q = ale.ppm_transport_x(h, u, Lf, np.zeros((1, 6)), qe, qw, act, 10.0, dx_t)
    assert np.isfinite(Q).all()
    assert (Q[:, 0, 2:4] == 0).all()


def test_ppm_transport_does_not_empty_a_vanished_donor_at_high_courant_number():
    # review focus 5: a 1 mm layer with C = 0.9 exports at most its own volume
    h = np.array([[[1e-3, 1e-3, 5.0, 5.0]], [[5.0, 5.0, 1e-3, 1e-3]]])
    u = np.zeros((2, 1, 5))
    u[0, 0, 2] = 9.0                                                     # C = 9·10/100 = 0.9
    qe, qw, dx_t, Lf, act = _ppm_case(h, u)
    Q, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    assert Q[0, 0, 2] * 10.0 <= 1e-3 * 100.0 * 10.0 + 1e-12              # Q dt ≤ h A of the donor


def test_ppm_transport_respects_the_clip_at_the_shallower_bed():
    # z* layer facing rock: zero flux whatever the parabola says
    deep = np.full(6, 100.0)
    shallow = np.array([1e-3, 1e-3, 1e-3, 1e-3, 63.996, 100.0])
    h = np.stack([deep, deep, shallow, shallow], axis=1)[:, None, :]
    H = np.array([[600.0, 600.0, 164.0, 164.0]])
    u = np.zeros((6, 1, 5))
    u[:, 0, 2] = 0.1
    qe, qw, dx_t, Lf, act = _ppm_case(h, u, clip=True, H=H)
    Q, _ = ale.ppm_fluxes_x(h, u, Lf, qe, qw, act, 10.0, dx_t)
    np.testing.assert_allclose(Q[:4, 0, 2], 0.0, atol=1e-12)
