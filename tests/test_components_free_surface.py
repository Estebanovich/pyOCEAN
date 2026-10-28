"""N1 task 7: free-surface + vertical-viscosity wrapper, BarotropicFluxes record, quadratic bed drag."""
from __future__ import annotations

import warnings

import numpy as np

from core.components.bottom_drag import QuadraticBedDrag
from core.components.free_surface import BarotropicFluxes, FreeSurfaceVerticalViscosity
from tests.n1_cases import CASES


def test_free_surface_closed_has_no_open_fluxes():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["zstar_vi_arakawa"]()
    assert isinstance(m._free_surface, FreeSurfaceVerticalViscosity)
    st = m.step(st, *wind(0))
    f = m._fluxes
    assert isinstance(f, BarotropicFluxes)
    assert f.open_Qu is None and f.open_Du is None and f.open_Qv is None and f.open_Dv is None
    assert f.wet_u.shape == m.fs.H_u.shape and f.Du.shape == m.fs.H_u.shape


def test_free_surface_open_has_open_fluxes():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["zstar_obc_radiating"]()
    st = m.step(st, *wind(0))
    assert m._fluxes.open_Du is not None and (m._fluxes.open_Du[:, 0] > 0).all()


def test_quadratic_drag_component_is_what_the_model_uses():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["zstar_quadratic_drag"]()
    st = m.step(st, *wind(0))
    assert isinstance(m._bed_drag, QuadraticBedDrag) and m._bed_drag.cd == 2e-3 and m._bed_drag.hbbl == 5.0
    hu, hv = m._mean_face_thickness(st.h)
    a = m._bed_drag.rates(st.u, st.v, hu, hv)
    b = m._bed_drag_rates(st.u, st.v, hu, hv)
    assert all(np.array_equal(x, y) for x, y in zip(a, b))
