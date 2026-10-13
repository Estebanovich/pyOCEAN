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
