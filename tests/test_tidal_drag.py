"""
Tests for physics.tides.drag — internal-tide (topographic wave) drag of
Jayne & St. Laurent (2001), as prepared and applied by HYCOM.

Preparation (HYCOM-examples GLBp0.08 relax/DRAG, wiki §8B):
    JSL [1/s] → bilinear to the model grid (isuba_field) → clip at 1/day
    (hycom_clip) → zero below 1/(10 days) (hycom_bandmask + hycom_void)
    → rh = JSL · depth [m/s], 0 on land (hycom_expr)            = tidal.rh
Use (mod_barotp.F90, tiddrg=1, thkdrg=0), implicit on the depth-mean flow:
    ū ← ū / (1 + Δt · drgscl · rh / H)
"""

from __future__ import annotations

import numpy as np
import pytest

from physics.tides.drag import (
    implicit_linear_drag,
    jsl_to_rh,
    regrid_bilinear,
    rh_at_faces,
    tidal_drag_rate,
)

DAY = 86400.0


class TestJSLToRh:

    def test_clip_band_and_land(self):
        jsl = np.array([[5e-5, 1.0e-5, 5e-7],
                        [2e-6, 0.0, 3e-6]])
        depth = np.array([[1000.0, 2000.0, 3000.0],
                          [500.0, 4000.0, np.nan]])                 # NaN = land
        rh = jsl_to_rh(jsl, depth)
        # 5e-5 clipped to 1.1574e-5 (1/day); 5e-7 < 1/(10 days) → 0
        np.testing.assert_allclose(rh[0], [1000.0 * 1.1574e-5, 2000.0 * 1.0e-5, 0.0], rtol=1e-12)
        np.testing.assert_allclose(rh[1], [500.0 * 2e-6, 0.0, 0.0])

    def test_rate_equals_jsl_where_in_band(self):
        # rh / H recovers JSL: the drag rate does not depend on the depth used
        jsl = np.full((2, 2), 4e-6)
        depth = np.array([[10.0, 100.0], [1000.0, 5000.0]])
        np.testing.assert_allclose(jsl_to_rh(jsl, depth) / depth, 4e-6)


class TestRegrid:

    def test_linear_field_reproduced(self):
        lon = np.arange(-180.0, 180.0, 0.08)
        lat = np.arange(-90.0, 90.01, 0.08)
        f = 2.0 + 0.01 * lon[None, :] - 0.03 * lat[:, None]
        lo, la = np.array([-100.013, -92.5]), np.array([15.031, 20.0])
        np.testing.assert_allclose(regrid_bilinear(f, lat, lon, la, lo), 2.0 + 0.01 * lo - 0.03 * la,
                                   rtol=1e-10)

    def test_plain_bilinear_like_isuba_field(self):
        # Land (zero) nodes are weighted like any other: no renormalisation
        lon = np.array([0.0, 1.0])
        lat = np.array([0.0, 1.0])
        f = np.array([[0.0, 4.0], [4.0, 4.0]])
        assert regrid_bilinear(f, lat, lon, np.array([0.5]), np.array([0.5]))[0] == pytest.approx(3.0)


class TestFaces:

    def test_mean_of_adjacent_wet_cells(self):
        rh = np.array([[1.0, 3.0, 0.0],
                       [2.0, 4.0, 6.0]])
        wet = np.array([[True, True, False],
                        [True, True, True]])
        ru, rv = rh_at_faces(rh, wet)
        assert ru.shape == (2, 4) and rv.shape == (3, 3)
        assert ru[0, 1] == pytest.approx(2.0)                       # between cells (0,0) and (0,1)
        assert ru[0, 2] == pytest.approx(3.0)                       # coast: only the wet neighbour
        assert ru[0, 0] == pytest.approx(1.0) and ru[1, 3] == pytest.approx(6.0)   # domain edges
        assert rv[1, 0] == pytest.approx(1.5) and rv[1, 2] == pytest.approx(6.0)
        assert ru[0, 3] == 0.0                                      # dry edge face


class TestDrag:

    def test_rate(self):
        r = tidal_drag_rate(np.array([0.02, 0.0, 0.05]), np.array([2000.0, 100.0, 0.0]), drgscl=0.25)
        np.testing.assert_allclose(r, [0.25 * 0.02 / 2000.0, 0.0, 0.0])

    def test_implicit_matches_hycom_scalar_tensor(self):
        # mod_barotp: q = 1/((1−d11)(1−d22) − d12 d21), ū_new = q(ū(1−d22) + v̄ d12) with
        # d11 = d22 = −Δt t11/H, d12 = d21 = 0
        u, dt, rate = np.array([0.3, -0.1]), 30.0, np.array([1e-4, 5e-6])
        d11 = -dt * rate
        hycom = 1.0 / ((1 - d11) * (1 - d11)) * (u * (1 - d11))
        np.testing.assert_allclose(implicit_linear_drag(u, rate, dt), hycom, rtol=1e-14)

    def test_unconditionally_stable_decay(self):
        u = np.array([1.0])
        for _ in range(100):
            u = implicit_linear_drag(u, np.array([1.0]), 10.0)          # rate·Δt = 10
        assert 0.0 < u[0] < 1e-100
