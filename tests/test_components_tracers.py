"""N1 task 9: tracer advection and diffusion components — the inflow values are an argument, not model state."""
from __future__ import annotations

import warnings

import numpy as np

from core.components.tracers import TracerAdvection, TracerDiffusion
from tests.n1_cases import CASES


def test_tracer_advection_inflow_is_explicit():
    warnings.simplefilter("ignore")
    m, st, wind = CASES["zstar_obc_radiating"]()
    st = m.step(st, *wind(0))
    adv = m._tracer_advection
    assert isinstance(adv, TracerAdvection) and isinstance(m._tracer_diffusion, TracerDiffusion)
    lf = m._last_layer_fluxes
    stack = np.stack([st.T, st.T**2])
    T_in = np.full(st.T.shape, 30.0)
    a = adv.advect_fct(stack, lf.V0, lf.V1, lf.Qx, lf.Qy, lf.ok, inflow=np.stack([T_in, T_in**2]))
    b = adv.advect_fct(stack, lf.V0, lf.V1, lf.Qx, lf.Qy, lf.ok, inflow=np.stack([T_in - 10, (T_in - 10)**2]))
    diff = np.any(a != b, axis=(0, 1))                                    # (ny, nx) columns that differ
    assert diff.any() and diff[:, 0].any()                                 # the inflow reaches the west edge cells
    m._T_in = None                                                          # model scratch does not leak in
    c = adv.advect_fct(stack, lf.V0, lf.V1, lf.Qx, lf.Qy, lf.ok, inflow=np.stack([T_in, T_in**2]))
    assert np.array_equal(a, c)
