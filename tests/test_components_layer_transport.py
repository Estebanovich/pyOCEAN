"""N1 task 8: layer-transport component — fluxes consistent with the layer continuity."""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.components.layer_transport import ALELayerTransport, LayerFluxes, SigmaLayerTransport
from tests.n1_cases import CASES


@pytest.mark.parametrize("case, cls", [("sigma_kh_kv", SigmaLayerTransport), ("zstar_upwind_thick", ALELayerTransport),
                                       ("hybrid_default", ALELayerTransport)])
def test_layer_fluxes_close_the_layer_continuity(case, cls):
    warnings.simplefilter("ignore")
    m, st, wind = CASES[case]()
    assert isinstance(m._layer_transport, cls)
    for n in range(3):
        st = m.step(st, *wind(n))
    f = m._last_layer_fluxes
    assert isinstance(f, LayerFluxes)
    divh = f.Qx[..., 1:] - f.Qx[..., :-1] + f.Qy[:, 1:, :] - f.Qy[:, :-1, :]
    # V1 − V0 = −Δt (∇_h·Q + Fz_top − Fz_bottom) layer by layer (Fz = 0 in the Lagrangian ALE step)
    resid = (f.V1 - f.V0) + m.dt * (divh + f.Fz[1:] - f.Fz[:-1])
    scale = np.abs(f.V0).max()
    assert np.abs(resid[:-1]).max() <= 1e-12 * scale                 # σ: the surface interface holds round-off
    if f.h_dagger is not None:
        assert (f.h_dagger >= 0).all()
