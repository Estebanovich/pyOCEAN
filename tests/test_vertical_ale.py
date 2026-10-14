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


# ---------------------------------------------------------------- Plan C: open boundaries (scope A)
def test_baroclinic_split_with_sigma_weights_matches_the_sigma_mean():
    rng = np.random.default_rng(11)
    ds = np.array([0.4, 0.3, 0.2, 0.1])
    u = rng.normal(0.0, 0.3, (4, 2, 5))
    mean, prime = ale.baroclinic_split(u, np.broadcast_to(ds[:, None, None], u.shape))
    np.testing.assert_allclose(mean, np.tensordot(ds, u, axes=(0, 0)), rtol=1e-14, atol=1e-16)
    np.testing.assert_allclose(prime, u - mean[None], rtol=0, atol=0)


def test_baroclinic_split_prime_has_zero_weighted_mean_with_vanished_layers():
    rng = np.random.default_rng(12)
    hf = rng.uniform(0.0, 5.0, (6, 3, 7))
    hf[2] = 0.0                                                          # a vanished layer
    hf[:, 0, 0] = 0.0                                                    # a face with no water
    w = ale.face_fractions(hf)
    u = rng.normal(0.0, 0.5, hf.shape)
    mean, prime = ale.baroclinic_split(u, w)
    np.testing.assert_allclose((w * prime).sum(axis=0), 0.0, atol=1e-15)
    np.testing.assert_allclose(mean[None] + prime, u, rtol=0, atol=1e-15)


def test_open_face_transport_adds_up_to_the_flather_flux():
    rng = np.random.default_rng(13)
    h = rng.uniform(0.1, 8.0, (5, 6))
    un, uo = rng.normal(0.0, 0.4, (5, 6)), rng.normal(0.0, 0.4, (5, 6))
    Qbt, L = rng.normal(0.0, 50.0, 6), np.full(6, 1000.0)
    Q = ale.open_face_transport(h, un, uo, Qbt, L, 0.6)
    np.testing.assert_allclose(Q.sum(axis=0), Qbt, rtol=1e-12, atol=1e-9)


def test_open_face_transport_vanished_layers_carry_nothing():
    # review focus 1: a vanished layer and a fully dry face column
    h = np.array([[0.0, 0.0], [3.0, 0.0], [0.0, 0.0], [2.0, 0.0]])
    un = np.full((4, 2), 0.3)
    uo = np.full((4, 2), -0.1)
    Q = ale.open_face_transport(h, un, uo, np.array([10.0, 0.0]), np.array([100.0, 100.0]), 0.6)
    assert np.isfinite(Q).all()
    assert (Q[[0, 2], 0] == 0).all() and (Q[:, 1] == 0).all()
    np.testing.assert_allclose(Q[:, 0].sum(), 10.0, rtol=1e-14)


def test_open_face_transport_reduces_to_the_sigma_formula():
    # h_k = D Δσ_k: Q_k = Δσ_k Q_bt + L D Δσ_k (θ u′ⁿ⁺¹ + (1 − θ) u′ⁿ), u′ with Δσ weights
    rng = np.random.default_rng(14)
    ds, D, L, th = np.array([0.1, 0.2, 0.3, 0.4]), np.array([20.0, 35.0, 50.0]), np.full(3, 500.0), 0.6
    un, uo = rng.normal(0.0, 0.3, (4, 3)), rng.normal(0.0, 0.3, (4, 3))
    Qbt = rng.normal(0.0, 100.0, 3)
    pn, po = un - np.tensordot(ds, un, axes=(0, 0)), uo - np.tensordot(ds, uo, axes=(0, 0))
    ref = ds[:, None] * Qbt[None] + (L * D)[None] * ds[:, None] * (th * pn + (1 - th) * po)
    Q = ale.open_face_transport(D[None] * ds[:, None], un, uo, Qbt, L, th)
    np.testing.assert_allclose(Q, ref, rtol=1e-12, atol=1e-10)


def test_exterior_on_layers_is_the_identity_on_the_same_grid():
    rng = np.random.default_rng(15)
    h = rng.uniform(0.5, 5.0, (6, 2, 3))
    T = rng.uniform(5.0, 20.0, h.shape)
    out = ale.exterior_on_layers(h, T, h.copy(), np.ones((2, 3), bool), 3)
    np.testing.assert_allclose(out, T, rtol=1e-12)


def test_exterior_on_layers_conserves_the_column_integral_after_scaling():
    # review focus 2: the target column holds H + η ≠ Σ h_ext; scaled by Σh / Σh_ext
    rng = np.random.default_rng(16)
    h_ext = rng.uniform(0.5, 5.0, (6, 1, 4))
    T = rng.uniform(5.0, 20.0, h_ext.shape)
    h = rng.uniform(0.5, 5.0, (6, 1, 4))
    mask = np.array([[True, True, False, True]])
    out = ale.exterior_on_layers(h_ext, T, h, mask, 3)
    scale = h.sum(0) / h_ext.sum(0)
    np.testing.assert_allclose((h * out).sum(0)[mask], (scale * (h_ext * T).sum(0))[mask], rtol=1e-12)
    np.testing.assert_array_equal(out[:, ~mask], T[:, ~mask])          # outside the mask: unchanged


def test_exterior_on_layers_gives_the_layer_means_of_a_linear_profile():
    # fine exterior layers with T linear in z; coarse targets get the exact layer means
    z_w = np.linspace(-100.0, 0.0, 21)
    h_ext = np.diff(z_w)[:, None, None] * np.ones((1, 1, 2))
    T = (10.0 + 0.05 * 0.5 * (z_w[1:] + z_w[:-1]))[:, None, None] * np.ones((1, 1, 2))
    zt = np.array([-100.0, -70.0, -50.0, -45.0, -10.0, 0.0])
    h = np.zeros((20, 1, 2))
    h[:5] = np.diff(zt)[:, None, None]                                   # 5 thick layers + 15 vanished at the top
    out = ale.exterior_on_layers(h_ext, T, h, np.ones((1, 2), bool), 3)
    np.testing.assert_allclose(out[:5, 0, 0], 10.0 + 0.05 * 0.5 * (zt[1:] + zt[:-1]), rtol=1e-10)


@pytest.mark.parametrize("clip", [True, False])
def test_transport_thickness_kernel_matches_numpy(clip):
    # review focus 1: domain-edge faces, land columns, vanished layers, a step in the bed
    rng = np.random.default_rng(31)
    n, ny, nx = 7, 5, 11
    H = rng.uniform(20.0, 600.0, (ny, nx))
    H[:, 4] = 150.0
    active = rng.uniform(size=(ny, nx)) > 0.15
    frac = rng.uniform(0.0, 1.0, (n, ny, nx))
    frac[:2] *= rng.uniform(size=(2, ny, nx)) > 0.5                        # vanished bed layers
    h = frac / frac.sum(0) * H[None]
    a = ale.face_transport_thickness(h, H, active, clip)
    b = ale.face_transport_thickness_numpy(h, H, active, clip)
    for x, y in zip(a, b):
        np.testing.assert_allclose(x, y, rtol=1e-14, atol=1e-14)


def _ppm_random(seed, n=6, ny=4, nx=9, clip=False, land=False, vanish=False):
    rng = np.random.default_rng(seed)
    H = rng.uniform(50.0, 300.0, (ny, nx)) if clip else np.full((ny, nx), 100.0)
    active = (rng.uniform(size=(ny, nx)) > 0.2) if land else np.ones((ny, nx), bool)
    frac = rng.uniform(0.05, 1.0, (n, ny, nx))
    if vanish:
        frac[rng.uniform(size=frac.shape) < 0.3] = 0.0
        frac[:, 0, 0] = 0.0
        frac[2, 0, 0] = 1.0
    h = frac / frac.sum(0) * H[None]
    ut = rng.normal(0.0, 0.3, (n, ny, nx + 1))
    qe, qw = ale.face_transport_thickness_numpy(h, H, active, clip)
    dx_t = rng.uniform(800.0, 1200.0, (ny, nx))
    L = rng.uniform(800.0, 1200.0, (ny, nx + 1))
    F0, _ = ale.ppm_fluxes_x(h, ut, L, qe, qw, active, 60.0, dx_t)
    Qbt = F0.sum(0) + rng.normal(0.0, 50.0, (ny, nx + 1))                 # needs a Newton correction
    return h, ut, L, Qbt, qe, qw, active, dx_t


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_ppm_kernel_matches_numpy(seed):
    args = _ppm_random(seed)
    a = ale.ppm_transport_x(*args[:7], 60.0, args[7])
    b = ale.ppm_transport_x_numpy(*args[:7], 60.0, args[7])
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)


@pytest.mark.parametrize("seed", [4, 5])
def test_ppm_kernel_matches_numpy_with_clip_and_vanished_layers(seed):
    # review focus 1–3: bed clip, vanished layers, land, edge faces; Newton with the global stop of the NumPy loop
    args = _ppm_random(seed, clip=True, land=True, vanish=True)
    a = ale.ppm_transport_x(*args[:7], 60.0, args[7])
    b = ale.ppm_transport_x_numpy(*args[:7], 60.0, args[7])
    np.testing.assert_allclose(a, b, rtol=1e-12, atol=1e-9)
    assert np.isfinite(a).all()


def test_ppm_edges_kernel_matches_numpy():
    args = _ppm_random(7, clip=True, land=True, vanish=True)
    h, active = args[0], args[6]
    hc = np.where(active[None], h, 0.0)
    hw, he = np.empty_like(hc), np.empty_like(hc)
    ale._ppm_edges_kernel(np.ascontiguousarray(hc), np.ascontiguousarray(active), ale.H_PPM_MIN, hw, he)
    rw, re = ale.ppm_edges_x(hc, active)
    np.testing.assert_allclose(hw, rw, rtol=1e-14, atol=1e-14)
    np.testing.assert_allclose(he, re, rtol=1e-14, atol=1e-14)


# ---------------------------------------------------------------- performance kernels (ALE performance task 5.4)
def _faces_random(seed=9, n=6, ny=5, nf=8):
    rng = np.random.default_rng(seed)
    hf = rng.uniform(0.0, 5.0, (n, ny, nf))
    hf[:3, 0, :] = 0.005                                                 # thin bed run
    hf[:, 1, 2] = 0.0                                                    # dry face
    hf[:, 2, 3] = 0.004                                                  # all thin
    hf[2, 3, :] = 0.0                                                    # vanished interior layer
    u = rng.normal(0.0, 0.3, (n, ny, nf))
    return hf, u


def test_face_fractions_kernel_matches_numpy():
    hf, _ = _faces_random()
    assert np.array_equal(ale.face_fractions(hf), ale.face_fractions_numpy(hf))


def test_baroclinic_split_kernel_matches_numpy():
    hf, u = _faces_random(10)
    w = ale.face_fractions_numpy(hf)
    a, b = ale.baroclinic_split(u, w), ale.baroclinic_split_numpy(u, w)
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_zero_thin_bed_kernel_matches_numpy():
    hf, u = _faces_random(11)
    w = ale.face_fractions_numpy(hf)
    assert np.array_equal(ale.zero_thin_bed_velocity(u, hf, w, 0.01), ale.zero_thin_bed_velocity_numpy(u, hf, w, 0.01))


@pytest.mark.parametrize("clip", [True, False])
def test_transport_thickness_kernel_reads_nan_land_and_swapped_views_as_numpy(clip):
    # perf part 2, task 4: NaN thickness/depth on land is read as 0 inside the kernel (no nan_to_num copies),
    # and the v-face call passes a swapped (strided) view
    rng = np.random.default_rng(32)
    n, ny, nx = 6, 7, 9
    H = rng.uniform(20.0, 300.0, (ny, nx))
    active = rng.uniform(size=(ny, nx)) > 0.2
    H[~active] = np.nan
    h = rng.uniform(0.0, 1.0, (n, ny, nx)) * np.nan_to_num(H)[None] / n
    h[:, ~active] = np.nan
    for args in ((h, H, active), (h.swapaxes(1, 2), H.T, active.T)):
        a = ale.face_transport_thickness(*args, clip)
        b = ale.face_transport_thickness_numpy(*args, clip)
        for x, y in zip(a, b):
            assert np.array_equal(x, y)


def test_transport_velocity_kernel_matches_numpy():
    # perf part 2, task 6: u^θ = θ u_new + (1−θ) u_old (0 on faces dry in the last step), Q_bt = L D Σ w u^θ
    rng = np.random.default_rng(7)
    n, a, b = 6, 5, 8
    u1, u0 = rng.normal(size=(n, a, b)), rng.normal(size=(n, a, b))
    wet = rng.random((a, b)) > 0.3
    w = rng.random((n, a, b))
    w /= w.sum(0)
    L, D = rng.uniform(500, 900, (a, b)), rng.uniform(0, 50, (a, b))
    ut, Qb = ale.transport_velocity(u1, u0, wet, w, L, D, 0.6)
    ut0, Qb0 = ale.transport_velocity_numpy(u1, u0, wet, w, L, D, 0.6)
    assert np.array_equal(ut, ut0) and np.array_equal(Qb, Qb0)


def test_transport_velocity_centres_the_baroclinic_part_only():
    # IW damping fix (2026-10-13): the barotropic part keeps the free-surface θ (its volume flux is the free-surface
    # flux), the baroclinic part u′ is time-centred with θ_bc: u^θ = θ_bc u¹ + (1−θ_bc) u⁰ + (θ−θ_bc)(ū¹ − ū⁰)
    rng = np.random.default_rng(11)
    n, a, b = 5, 3, 4
    u1, u0 = rng.normal(size=(n, a, b)), rng.normal(size=(n, a, b))
    wet = np.ones((a, b), bool)
    w = rng.random((n, a, b))
    w /= w.sum(0)
    L, D = np.full((a, b), 800.0), rng.uniform(10, 50, (a, b))
    ut, Qb = ale.transport_velocity_numpy(u1, u0, wet, w, L, D, 0.6, theta_bc=0.5)
    m1, m0 = (w * u1).sum(0), (w * u0).sum(0)
    np.testing.assert_allclose((w * ut).sum(0), 0.6 * m1 + 0.4 * m0, atol=1e-13)          # barotropic: θ
    np.testing.assert_allclose(ut - (w * ut).sum(0)[None], 0.5 * (u1 - m1[None]) + 0.5 * (u0 - m0[None]), atol=1e-13)
    _, Qb_old = ale.transport_velocity_numpy(u1, u0, wet, w, L, D, 0.6)                     # θ_bc = θ: old scheme
    np.testing.assert_allclose(Qb, Qb_old, rtol=1e-13)                                       # same volume flux
    ut_k, Qb_k = ale.transport_velocity(u1, u0, wet, w, L, D, 0.6, theta_bc=0.5)
    np.testing.assert_allclose(ut_k, ut, atol=1e-14) and np.testing.assert_allclose(Qb_k, Qb, rtol=1e-14)


def test_open_face_transport_centres_the_baroclinic_part():
    h = np.array([[2.0, 1.0], [3.0, 0.0], [5.0, 4.0]])
    un, uo = np.array([[0.3, 0.1], [0.1, 0.0], [-0.2, 0.2]]), np.array([[0.1, 0.0], [0.2, 0.0], [0.0, 0.1]])
    Qbt, L = np.array([10.0, 3.0]), np.array([100.0, 100.0])
    Q = ale.open_face_transport(h, un, uo, Qbt, L, 0.6, theta_bc=0.5)
    np.testing.assert_allclose(Q.sum(0), Qbt, rtol=1e-13)
    w = h / h.sum(0)
    pn, po = un - (w * un).sum(0), uo - (w * uo).sum(0)
    np.testing.assert_allclose(Q, w * Qbt + L * h * (0.5 * pn + 0.5 * po), rtol=1e-13)


# ---------------------------------------------------------------- outflow limiter (spec 2026-10-14-hybrid-robustness §2)
def _thin_divergent_cell():
    """One row, 3 cells × 3 layers; layer 1 of the middle cell is 1 mm and loses through both faces more than it
    holds (Courant 0.45 west + 0.52 east plus a barotropic correction, the IW failure of 2026-10-14)."""
    dx, dt = 5000.0, 300.0
    h = np.array([[[50.0, 50.0, 50.0]], [[0.08, 0.001, 49.9]], [[50.0, 50.0, 50.0]]])
    area = np.full((1, 3), dx * dx)
    Qx = np.zeros((3, 1, 4))
    Qx[1, 0, 1] = -0.459 * 0.001 * dx * dx / dt          # west face, flows west out of the middle cell
    Qx[1, 0, 2] = +0.553 * 0.001 * dx * dx / dt          # east face, flows east out of the middle cell
    Qx[0, 0, 1:3] = [0.2, 0.3]
    Qx[2, 0, 1:3] = [-0.4, 0.1]
    return h, Qx, np.zeros((3, 2, 3)), area, dt, np.ones((1, 3), bool)


def test_limit_outflow_keeps_the_thin_layer_non_negative_and_the_face_sums():
    h, Qx, Qy, area, dt, act = _thin_divergent_cell()
    Qx2, Qy2, n = ale.limit_outflow(h, Qx, Qy, area, dt, act)
    div = Qx2[..., 1:] - Qx2[..., :-1] + Qy2[:, 1:] - Qy2[:, :-1]
    h_new = h - dt * div / area[None]
    assert n >= 1 and h_new.min() >= 0.0
    np.testing.assert_allclose(Qx2.sum(0), Qx.sum(0), rtol=1e-12, atol=1e-12)       # Σ_k Q_k per face kept
    assert np.all(np.sign(Qx2[1]) * np.sign(Qx[1]) >= 0)                              # no direction reversal


def test_limit_outflow_is_a_no_op_when_nothing_empties():
    h, Qx, Qy, area, dt, act = _thin_divergent_cell()
    h = h.copy()
    h[1, 0, 1] = 1.0                                                                  # now holds enough
    Qx2, Qy2, n = ale.limit_outflow(h, Qx, Qy, area, dt, act)
    assert n == 0 and Qx2 is Qx and Qy2 is Qy


def test_limit_outflow_reduces_counter_flowing_layers_when_no_same_way_layer_has_room():
    # two layers; layer 0 empties eastwards, layer 1 flows westwards: the cut must come off layer 1's inflow
    dx, dt = 1000.0, 10.0
    h = np.array([[[1e-3, 10.0]], [[10.0, 10.0]]])
    area = np.full((1, 2), dx * dx)
    Qx = np.zeros((2, 1, 3))
    Qx[0, 0, 1] = 2e-3 * dx * dx / dt            # layer 0: twice its content out of cell 0
    Qx[1, 0, 1] = -500.0                         # layer 1: westwards into cell 0 (can absorb the ~100 m³/s cut)
    Qx2, _, n = ale.limit_outflow(h, Qx, np.zeros((2, 2, 2)), area, dt, np.ones((1, 2), bool))
    assert n == 1
    np.testing.assert_allclose(Qx2[0, 0, 1], 1e-3 * dx * dx / dt * (1 - 2e-7), rtol=1e-6)   # down to its content
    np.testing.assert_allclose(Qx2.sum(0), Qx.sum(0), rtol=1e-12)
    assert -500.0 < Qx2[1, 0, 1] < 0.0


def test_limit_outflow_handles_open_edge_fluxes():
    # outflow through an open west edge face (index 0) counts as outflow of cell 0
    dx, dt = 1000.0, 10.0
    h = np.array([[[1e-3, 10.0]], [[10.0, 10.0]]])
    area = np.full((1, 2), dx * dx)
    Qx = np.zeros((2, 1, 3))
    Qx[0, 0, 0] = -3e-3 * dx * dx / dt           # leaves cell 0 through the west edge (300 m³/s, holds 100)
    Qx[1, 0, 0] = 500.0                          # inflow of layer 1 through the edge absorbs the 200 m³/s cut
    Qx2, _, n = ale.limit_outflow(h, Qx, np.zeros((2, 2, 2)), area, dt, np.ones((1, 2), bool))
    h_new = h - dt * (Qx2[..., 1:] - Qx2[..., :-1]) / area[None]
    assert n >= 1 and h_new.min() >= 0.0
    np.testing.assert_allclose(Qx2.sum(0), Qx.sum(0), rtol=1e-12)


def test_limit_outflow_raises_when_the_column_cannot_supply_the_flux():
    dx, dt = 1000.0, 10.0
    h = np.array([[[1e-3, 10.0]], [[1e-3, 10.0]]])
    area = np.full((1, 2), dx * dx)
    Qx = np.zeros((2, 1, 3))
    Qx[:, 0, 1] = 150.0                          # 3000 m³ in 10 s eastwards from a column holding 2000 m³
    with pytest.raises(RuntimeError, match="barotropic CFL"):
        ale.limit_outflow(h, Qx, np.zeros((2, 2, 2)), area, dt, np.ones((1, 2), bool))


def test_outflow_check_kernel_matches_numpy():
    rng = np.random.default_rng(5)
    n, ny, nx = 4, 6, 7
    h = rng.uniform(0, 2, (n, ny, nx)); h[1, 2, 3] = 1e-4
    Qx = rng.normal(0, 50, (n, ny, nx + 1)); Qy = rng.normal(0, 50, (n, ny + 1, nx))
    area = np.full((ny, nx), 1e4); act = rng.uniform(0, 1, (ny, nx)) > 0.1
    for dt in (1.0, 30.0, 300.0):
        ref = bool((act[None] & (ale._outflow_numpy(Qx, Qy) > np.maximum(h - 2e-10, 0) * np.where(act, area, 0)[None]
                                                     / dt)).any())
        assert ale._outflow_exceeds(h, Qx, Qy, area, dt, act, 2e-10) == ref


def test_viscous_face_thickness_vanishes_with_either_side():
    qa = np.array([50.0, 1e-3, 0.0, 10.0])
    qb = np.array([50.0, 50.0, 50.0, 10.0])
    hv = ale.viscous_face_thickness(qa, qb)
    np.testing.assert_allclose(hv, [50.0, 2 * 1e-3 * 50 / 50.001, 0.0, 10.0], rtol=1e-9, atol=1e-12)


def test_viscous_face_thickness_is_upwind_biased_with_a_velocity():
    # MOM6 (MOM_vert_friction.F90:1519-1533; Hallberg & Rhines 1996): harmonic where water leaves the thin side
    # (u·(h_b − h_a) > 0), arithmetic where thick water advances over thin — the nose of a current is not held back
    q_a = np.array([1e-3, 1e-3, 50.0, 50.0])        # low-index (west/south) side
    q_b = np.array([50.0, 50.0, 1e-3, 1e-3])        # high-index side
    u = np.array([1.0, -1.0, 1.0, -1.0])
    hv = ale.viscous_face_thickness(q_a, q_b, u)
    harm = 2 * 1e-3 * 50 / 50.001
    np.testing.assert_allclose(hv, [harm, 25.0005, 25.0005, harm], rtol=1e-9)


def test_viscous_face_thickness_kernel_matches_numpy_on_strided_3d_arrays():
    # the ALE step passes (N, ny, nx+1) arrays and swapped views of the v faces: the fused kernel = the NumPy rule
    rng = np.random.default_rng(5)
    q_a = rng.uniform(0, 30, (6, 9, 7)); q_b = rng.uniform(0, 30, (6, 9, 7)); u = rng.normal(0, 1, (6, 9, 7))
    q_a[0, 0, 0] = 0.0; u[1, 2, 3] = 0.0
    for a, b, w in ((q_a, q_b, u), (q_a.swapaxes(1, 2), q_b.swapaxes(1, 2), u.swapaxes(1, 2))):
        harm = 2.0 * a * b / (a + b + 1e-10)
        ref = np.where(w * (b - a) > 0, harm, 0.5 * (a + b))
        np.testing.assert_allclose(ale.viscous_face_thickness(a, b, w), ref, rtol=1e-15, atol=0)
    assert ale._viscous_face_kernel is not None


def test_blend_thin_layers_follows_hycom():
    # HYCOM mod_momtum.F90:3055-3065: top-down, u_k ← (u_k q + u_above (c − q))/c, q = min(h, c); k = 0 at the bed
    c = 0.5
    hf = np.array([[[0.0]], [[0.25]], [[10.0]], [[0.0]]])        # bed: vanished, 25 cm, thick, top: vanished
    u = np.array([[[9.0]], [[4.0]], [[1.0]], [[7.0]]])
    w = hf / hf.sum(0)
    out = ale.blend_thin_layers(u, hf, w, c)
    raw = out - (out[2] - 1.0)                                     # remove the uniform Σ w u correction
    assert abs(raw[3, 0, 0] - 1.0) < 1e-14                         # thin top layer takes the layer below
    assert abs(raw[1, 0, 0] - (0.5 * 4.0 + 0.5 * 1.0)) < 1e-14     # half way: q = 0.25 of c = 0.5
    assert abs(raw[0, 0, 0] - raw[1, 0, 0]) < 1e-14                            # vanished bed layer: the (blended) layer above
    np.testing.assert_allclose((w * out).sum(0), (w * u).sum(0), rtol=1e-14)


def test_blend_thin_layers_is_a_no_op_for_thick_columns():
    hf = np.full((3, 2, 3), 2.0)
    u = np.random.default_rng(1).normal(size=(3, 2, 3))
    assert np.array_equal(ale.blend_thin_layers(u, hf, hf / hf.sum(0), 0.5), u)


def test_limit_outflow_solves_a_column_whose_same_way_layer_is_already_draining():
    # final review 2026-10-14: the cut of a 1 mm layer had nowhere to go at its face (the 10 m layer's room was taken
    # by its outflow through both faces, no counter-flow there) and the limiter stopped with "barotropic CFL",
    # although the column holds 10 m and loses 5
    h = np.array([[[1e-3, 1e-3, 1e-3]], [[10.0, 10.0, 10.0]], [[10.0, 1e-3, 10.0]]])
    Qx = np.zeros((3, 1, 4)); Qy = np.zeros((3, 2, 3))
    Qx[0, 0, 2] = 3e-3                                       # 1 mm layer leaves east with 3 mm in Δt
    Qx[1, 0, 1], Qx[1, 0, 2] = -4.9995, 4.9995               # 10 m layer leaves through both faces
    Qx[2, 0, 1] = 5.0                                        # the west cell's layer 2 flows in
    area, act = np.ones((1, 3)), np.ones((1, 3), bool)
    Qx2, Qy2, n = ale.limit_outflow(h, Qx, Qy, area, 1.0, act)
    np.testing.assert_allclose(Qx2.sum(0), Qx.sum(0), rtol=0, atol=1e-12)
    h_dag = h - (Qx2[..., 1:] - Qx2[..., :-1] + Qy2[:, 1:] - Qy2[:, :-1])
    assert h_dag.min() >= -1e-12 and n > 0
