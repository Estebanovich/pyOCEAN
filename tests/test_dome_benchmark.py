"""DOME benchmark pipeline (spec docs/superpowers/specs/2026-10-14-dome-overflow-design.md §5–§7)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.dome import cases


def test_bathymetry_matches_mom6_dome_initialize_topography():
    c = cases.get(10.0)
    assert (c.nx, c.ny) == (150, 70)
    j_shelf = np.searchsorted(c.y, 450e3)                            # mid-slope: 600 + 0.01·(600 − 450) km
    np.testing.assert_allclose(c.H[j_shelf, 0], 600.0 + 0.01 * (600e3 - c.y[j_shelf]), rtol=1e-12)
    assert np.all(c.H[c.y < 300e3] == 3600.0)
    north = c.y >= 600e3
    chan = (c.x > 1000e3) & (c.x < 1100e3)
    assert np.all(c.H[np.ix_(north, chan)] == 600.0)
    assert np.all(np.isnan(c.H[np.ix_(north, ~chan)]))


def test_inflow_transport_is_legg_and_close_to_mom6_by_density_class():
    c = cases.get(10.0)
    p = cases.inflow_profile(c)
    db0 = cases.G * cases.DRHO / cases.RHO0
    legg = db0 * cases.H0**2 / (2 * c.f0)                            # Legg 2006 §2: T_in = Δb₀ h₀² / (2f) ≈ 8.55 Sv
    assert abs(-p["transport"].sum() - legg) / legg < 0.05
    rho = cases.RHO0 - cases.ALPHA * (p["T"] - cases.T_REF)
    dz = -np.diff(p["zw"], axis=0)
    flux = -p["v"] * dz * c.dx                                         # southward flux of every profile cell
    dense = flux[rho > 1031.9].sum()
    mom6 = cases.mom6_layer_transports()
    assert abs(dense - (-mom6[-1])) / (-mom6[-1]) < 0.25              # densest class: 7.45 Sv in MOM6 (diff. (c))


def test_inflow_profile_is_geostrophic_and_ri_bounded():
    c = cases.get(10.0)
    p = cases.inflow_profile(c)
    i = np.argmax(-p["transport"])                                    # the right-hand (west) wall of the channel
    assert p["s"][:, i].max() == 1.0 and p["s"][:, i].min() == 0.0
    assert np.all(np.diff(p["T"][:, i]) <= 1e-12)                     # stable: T decreases downward (surface first)
    assert np.all(p["v"] <= 0.0)                                       # southward into the domain


def test_sponge_profile_matches_mom6():
    c = cases.get(10.0)
    r = cases.sponge_rate(c)
    peak = 10.0 / 86400.0
    assert r[0, 0] == pytest.approx(peak) and r[0, 75] == 0.0
    i150 = np.argmin(np.abs(c.x - 150e3))                              # on the ramp (cell centre 145 km)
    assert r[0, i150] == pytest.approx(peak * (200e3 - c.x[i150]) / 100e3, rel=1e-12)


@pytest.mark.parametrize("coord", ["zstar", "sigma", "hybrid"])
def test_pyocean_dome_builds_and_takes_steps_with_inflow_tracer_and_sponge(coord):
    from validation.dome import run_pyocean
    c = cases.get(50.0)                                               # 30 × 14 coarse smoke grid
    m, st = run_pyocean.build(c, coord)
    for _ in range(6):
        st = m.step(st)
    assert np.isfinite(st.u).all() and st.tr.max() > 0.0               # dye entered through the channel
    q = st.Qy[:, -1, :].sum()
    np.testing.assert_allclose(q, cases.inflow_profile(c)["transport"].sum(), rtol=1e-10)
