"""
Tests for forcing.ocean_init — canonical ocean data → model initial state and
open-boundary values.

Pipeline: canonical ocean (any adapter: ESPC/GOFS z, HYCOM archive, CMEMS…)
→ sea-over-land fill → C-grid T/U/V points → model z levels → time
interpolation → ``ModelState`` (initial condition) and ``OBData`` per side.
Analytic fields linear in lon, lat and depth make every step exact.
"""

from __future__ import annotations

import numpy as np
import pytest
import xarray as xr

from boundaries.obcs import OBData, sponge_tendency
from core.grid.horizontal import create_spherical_grid
from core.grid.vertical import create_z_grid
from core.state import ModelState
from forcing.canonical import CANONICAL_VARS
from forcing.ocean_init import boundary_data, initial_state, prepare_ocean

NX, NY = 6, 5
LON0, LAT0, DL = -96.0, 19.0, 0.5
DEPTH = np.array([0.0, 10.0, 50.0, 100.0, 500.0])
TIMES = np.array(["2026-05-25T00", "2026-05-26T00"], dtype="datetime64[ns]")


def _canonical(land: bool = True, coast_inside: bool = False) -> xr.Dataset:
    lon = np.arange(LON0 - 1.0, LON0 + NX * DL + 1.01, 0.25)
    lat = np.arange(LAT0 - 1.0, LAT0 + NY * DL + 1.01, 0.25)
    T, Z, Y, X = np.meshgrid(np.arange(2), DEPTH, lat, lon, indexing="ij")
    temp = 28.0 - 0.02 * Z + 0.1 * (X - LON0) - 0.2 * (Y - LAT0) + 1.0 * T
    salt = 36.0 - 0.001 * Z + 0.0 * X
    u = 0.1 + 0.02 * (X - LON0) + 0.0 * Z + 0.05 * T
    v = -0.05 + 0.01 * (Y - LAT0) + 0.0 * Z
    ssh = 0.3 + 0.01 * (X[:, 0] - LON0) + 0.1 * T[:, 0]
    if land:                                        # a land block NW, outside the model domain
        for f in (temp, salt, u, v):
            f[:, :, -3:, :3] = np.nan
        ssh[:, -3:, :3] = np.nan
    if coast_inside:                                # land overlapping the model's SW corner
        inside = (lon[None, :] < LON0 + 0.6) & (lat[:, None] < LAT0 + 0.6)
        for f in (temp, salt, u, v):
            f[:, :, inside] = np.nan
        ssh[:, inside] = np.nan
    ds = xr.Dataset(
        {"temp": (("time", "depth", "lat", "lon"), temp), "salt": (("time", "depth", "lat", "lon"), salt),
         "u": (("time", "depth", "lat", "lon"), u), "v": (("time", "depth", "lat", "lon"), v),
         "ssh": (("time", "lat", "lon"), ssh)},
        coords={"time": TIMES, "depth": DEPTH, "lat": lat, "lon": lon})
    for name in ds.data_vars:
        ds[name].attrs["units"] = CANONICAL_VARS["ocean"][name]["units"]
    return ds


@pytest.fixture
def grids():
    hg = create_spherical_grid(nx=NX, ny=NY, dlon=DL, dlat=DL, lon0=LON0, lat0=LAT0)
    vg = create_z_grid(4, [10.0, 20.0, 70.0, 300.0])     # z_t = −5, −20, −65, −250
    return hg, vg


def _temp(lon, lat, z, t_days=0.0):
    return 28.0 + 0.02 * z + 0.1 * (lon - LON0) - 0.2 * (lat - LAT0) + t_days


# ===========================================================================

class TestPrepareOcean:

    def test_points_and_levels(self, grids):
        hg, vg = grids
        p = prepare_ocean(_canonical(), hg, vg)
        assert p["t"]["temp"].dims == ("time", "k", "j_t", "i_t")
        assert p["u"]["u"].dims == ("time", "k", "j_t", "i_u")
        assert p["v"]["v"].dims == ("time", "k", "j_v", "i_t")
        assert p["t"]["ssh"].dims == ("time", "j_t", "i_t")

    def test_temperature_exact_on_t_points(self, grids):
        hg, vg = grids
        p = prepare_ocean(_canonical(), hg, vg)
        lon, lat = hg.lon_t.values, hg.lat_t.values
        for k, z in enumerate(vg.z_t.values):
            np.testing.assert_allclose(p["t"]["temp"].isel(time=0, k=k), _temp(lon, lat, z), atol=1e-9)

    def test_u_evaluated_on_u_points(self, grids):
        hg, vg = grids
        p = prepare_ocean(_canonical(), hg, vg)
        np.testing.assert_allclose(p["u"]["u"].isel(time=0, k=0),
                                   0.1 + 0.02 * (hg.lon_u.values - LON0), atol=1e-9)

    def test_no_nan_after_land_fill(self, grids):
        hg, vg = grids
        p = prepare_ocean(_canonical(coast_inside=True), hg, vg)
        for pt in p.values():
            for name in pt.data_vars:
                assert not np.isnan(pt[name].values).any(), name


class TestInitialState:

    def test_state_fields(self, grids):
        hg, vg = grids
        st = initial_state(prepare_ocean(_canonical(), hg, vg), hg, vg, np.datetime64("2026-05-25T12"))
        assert isinstance(st, ModelState)
        lon, lat = hg.lon_t.values, hg.lat_t.values
        np.testing.assert_allclose(st.T.isel(k=1), _temp(lon, lat, -20.0, 0.5), atol=1e-9)
        np.testing.assert_allclose(st.u.isel(k=0), 0.1 + 0.02 * (hg.lon_u.values - LON0) + 0.025, atol=1e-9)
        assert np.all(st.w.values == 0.0)

    def test_ssh_domain_mean_removed_by_default(self, grids):
        # Product SSH has its own reference level (design doc §2.2)
        hg, vg = grids
        st = initial_state(prepare_ocean(_canonical(), hg, vg), hg, vg, TIMES[0])
        assert float(st.eta.mean()) == pytest.approx(0.0, abs=1e-12)

    def test_ssh_kept_when_requested(self, grids):
        hg, vg = grids
        st = initial_state(prepare_ocean(_canonical(), hg, vg), hg, vg, TIMES[0], remove_mean_ssh=False)
        np.testing.assert_allclose(st.eta, 0.3 + 0.01 * (hg.lon_t.values - LON0), atol=1e-9)


class TestBoundaryData:

    @pytest.mark.parametrize("side,t_idx,vel,vel_idx,n", [
        ("west", (slice(None), slice(None), 0), "u", (slice(None), slice(None), 0), NY),
        ("east", (slice(None), slice(None), -1), "u", (slice(None), slice(None), -1), NY),
        ("south", (slice(None), 0, slice(None)), "v", (slice(None), 0, slice(None)), NX),
        ("north", (slice(None), -1, slice(None)), "v", (slice(None), -1, slice(None)), NX),
    ])
    def test_values_match_state_edges(self, grids, side, t_idx, vel, vel_idx, n):
        hg, vg = grids
        prep = prepare_ocean(_canonical(), hg, vg)
        t = np.datetime64("2026-05-25T06")
        st = initial_state(prep, hg, vg, t, remove_mean_ssh=False)
        ob = boundary_data(prep, side, t, remove_mean_ssh=False)
        assert isinstance(ob, OBData) and ob.side == side
        assert ob.T.shape == (4, n) and ob.eta.shape == (n,)
        np.testing.assert_allclose(ob.T, st.T.values[t_idx], atol=1e-12)
        np.testing.assert_allclose(ob.S, st.S.values[t_idx], atol=1e-12)
        np.testing.assert_allclose(ob.u_normal, getattr(st, vel).values[vel_idx], atol=1e-12)
        eta_idx = t_idx[1:]
        np.testing.assert_allclose(ob.eta, st.eta.values[eta_idx], atol=1e-12)

    def test_unknown_side(self, grids):
        hg, vg = grids
        with pytest.raises(ValueError, match="side"):
            boundary_data(prepare_ocean(_canonical(), hg, vg), "up", TIMES[0])

    def test_drives_sponge(self, grids):
        hg, vg = grids
        prep = prepare_ocean(_canonical(), hg, vg)
        st = ModelState.from_grids(hg, vg, T_init=20.0, S_init=35.0)
        obs = [boundary_data(prep, s, TIMES[0]) for s in ("west", "south")]
        tend = sponge_tendency(st, hg, vg, obs, sponge_width=2, tau_sponge=86400.0)
        assert np.all(np.isfinite(tend.T.values))
        assert np.any(tend.T.values != 0.0)
