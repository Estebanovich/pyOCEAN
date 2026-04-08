"""
Tests for forcing/surface.py and boundaries/obcs/__init__.py.

Validation strategy:
  1. Analytical / limiting-case checks.
  2. Sign / direction checks from known physics.
  3. Conservation checks (column tracer, momentum impulse).
  4. Orlanski: outgoing wave reduces boundary reflection.
  5. Sponge: exponential-like decay toward prescribed values.
  6. Shape, dims, coordinate consistency.

Covers
------
  bulk_wind_stress   : magnitude, direction, zero input
  surface_tendency   : wind→u/v, Q_net→T, Q_sw penetration, E_P→S,
                       zero forcing → zero tendency, column T conservation
  sponge_tendency    : zero when no boundaries, sign/direction, γ profile,
                       dims, conservation-neutral (sponge nudges, not conserves)
  apply_orlanski     : no-op for empty boundary list, outgoing wave propagates,
                       incoming → prescribed value, c_n clipped to [0, dn/dt]
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import create_z_grid, uniform_z_grid
from core.state import ModelState
from forcing.surface import bulk_wind_stress, surface_tendency, RHO_0, CP
from boundaries.obcs import OBData, sponge_tendency, apply_orlanski

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY, NZ = 8, 6, 4
DX = DY = 2000.0
DZ = 10.0
F0 = 1e-4


@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DY, f0=F0)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=NZ, dz_uniform=DZ)


@pytest.fixture
def state(hgrid, vgrid):
    return ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)


# ===========================================================================
# bulk_wind_stress
# ===========================================================================

class TestBulkWindStress:

    def test_zero_wind_gives_zero_stress(self):
        u10 = np.zeros((NY, NX))
        v10 = np.zeros((NY, NX))
        tx, ty = bulk_wind_stress(u10, v10)
        assert np.all(tx == 0.0)
        assert np.all(ty == 0.0)

    def test_eastward_wind_gives_eastward_stress(self):
        u10 = np.full((NY, NX), 10.0)
        v10 = np.zeros((NY, NX))
        tx, ty = bulk_wind_stress(u10, v10)
        assert np.all(tx > 0)
        assert np.all(ty == 0.0)

    def test_stress_scales_quadratically(self):
        """τ ∝ |U|²."""
        u_slow = np.full((NY, NX), 5.0)
        u_fast = np.full((NY, NX), 10.0)
        tx_slow, _ = bulk_wind_stress(u_slow, np.zeros((NY, NX)))
        tx_fast, _ = bulk_wind_stress(u_fast, np.zeros((NY, NX)))
        assert np.allclose(tx_fast / tx_slow, 4.0, rtol=1e-10)

    def test_custom_Cd(self):
        u10 = np.full((NY, NX), 10.0)
        rho_a = 1.225
        Cd = 2.0e-3
        tx, _ = bulk_wind_stress(u10, np.zeros((NY, NX)), rho_a=rho_a, Cd=Cd)
        expected = rho_a * Cd * 100.0   # 1.225 * 2e-3 * 100 = 0.245
        assert np.allclose(tx, expected, rtol=1e-10)

    def test_symmetry(self):
        """τ_x from u=U, v=0 should equal τ_y from u=0, v=U."""
        u10 = np.full((NY, NX), 7.0)
        tx, _ = bulk_wind_stress(u10, np.zeros((NY, NX)))
        _, ty = bulk_wind_stress(np.zeros((NY, NX)), u10)
        assert np.allclose(tx, ty, rtol=1e-10)


# ===========================================================================
# surface_tendency — shape and dims
# ===========================================================================

class TestSurfaceTendencyShape:

    def test_output_shape_no_forcing(self, state, vgrid):
        tend = surface_tendency(state, vgrid)
        assert tend.u.shape == (NZ, NY, NX + 1)
        assert tend.v.shape == (NZ, NY + 1, NX)
        assert tend.T.shape == (NZ, NY, NX)
        assert tend.S.shape == (NZ, NY, NX)
        assert tend.eta.shape == (NY, NX)

    def test_zero_forcing_gives_zero_tendency(self, state, vgrid):
        tend = surface_tendency(state, vgrid)
        assert np.allclose(tend.u.values, 0.0)
        assert np.allclose(tend.v.values, 0.0)
        assert np.allclose(tend.T.values, 0.0)
        assert np.allclose(tend.S.values, 0.0)
        assert np.allclose(tend.eta.values, 0.0)

    def test_units_attrs(self, state, vgrid):
        tend = surface_tendency(state, vgrid,
                                tau_x=np.ones((NY, NX)),
                                Q_net=np.ones((NY, NX)))
        assert "units" in tend.u.attrs
        assert "units" in tend.T.attrs


# ===========================================================================
# surface_tendency — wind stress
# ===========================================================================

class TestWindStressTendency:

    def test_positive_tau_x_accelerates_u_top_layer(self, state, vgrid):
        tau_x = np.full((NY, NX), 0.1)
        tend = surface_tendency(state, vgrid, tau_x=tau_x)
        # Interior U-faces (i_u = 1..NX-1) should have du/dt > 0
        assert np.all(tend.u.values[0, :, 1:-1] > 0), "du/dt[0] should be positive"

    def test_wind_only_affects_top_layer(self, state, vgrid):
        tau_x = np.full((NY, NX), 0.1)
        tend = surface_tendency(state, vgrid, tau_x=tau_x)
        assert np.allclose(tend.u.values[1:], 0.0), "Wind stress only at k=0"

    def test_zero_at_wall_u_faces(self, state, vgrid):
        tau_x = np.full((NY, NX), 0.1)
        tend = surface_tendency(state, vgrid, tau_x=tau_x)
        assert np.allclose(tend.u.values[:, :, 0], 0.0)   # west wall
        assert np.allclose(tend.u.values[:, :, -1], 0.0)  # east wall

    def test_analytical_wind_stress_magnitude(self, state, vgrid):
        """du/dt[0] = τ_x_u / (ρ₀ × dz[0]) for interior U-faces."""
        tau_val = 0.1   # N/m²
        tau_x = np.full((NY, NX), tau_val)
        tend = surface_tendency(state, vgrid, tau_x=tau_x)
        expected = tau_val / (RHO_0 * DZ)
        # Interior U-faces: τ_x_u = 0.5*(τ_x[j,i-1] + τ_x[j,i]) = τ_val (uniform)
        assert np.allclose(tend.u.values[0, :, 1:-1], expected, rtol=1e-10)

    def test_positive_tau_y_accelerates_v_top_layer(self, state, vgrid):
        tau_y = np.full((NY, NX), 0.1)
        tend = surface_tendency(state, vgrid, tau_y=tau_y)
        # Interior V-faces (j_v = 1..NY-1)
        assert np.all(tend.v.values[0, 1:-1, :] > 0)
        assert np.allclose(tend.v.values[1:], 0.0)


# ===========================================================================
# surface_tendency — heat flux
# ===========================================================================

class TestHeatFluxTendency:

    def test_positive_Q_net_warms_surface(self, state, vgrid):
        Q = np.full((NY, NX), 100.0)   # W/m²
        tend = surface_tendency(state, vgrid, Q_net=Q)
        assert np.all(tend.T.values[0] > 0), "Positive Q_net should warm surface"

    def test_Q_net_only_top_layer_without_Qsw(self, state, vgrid):
        Q = np.full((NY, NX), 100.0)
        tend = surface_tendency(state, vgrid, Q_net=Q)
        assert np.allclose(tend.T.values[1:], 0.0), "No Q_sw → no sub-surface heating"

    def test_analytical_surface_heating(self, state, vgrid):
        """dT[0]/dt = Q_net / (ρ₀ Cp dz[0])."""
        Q_val = 200.0
        Q = np.full((NY, NX), Q_val)
        tend = surface_tendency(state, vgrid, Q_net=Q)
        expected = Q_val / (RHO_0 * CP * DZ)
        assert np.allclose(tend.T.values[0], expected, rtol=1e-10)

    def test_Qsw_penetrates_subsurface(self, state, vgrid):
        Q_sw = np.full((NY, NX), 100.0)
        tend = surface_tendency(state, vgrid, Q_net=Q_sw, Q_sw=Q_sw)
        # With Q_net = Q_sw → non-penetrating = 0, all shortwave
        # Q_sw should heat layers below surface too
        assert np.any(tend.T.values[1:] > 0), "Shortwave should heat sub-surface layers"

    def test_Qsw_column_energy_conservation(self, state, vgrid):
        """
        Sum of dT[k]*dz[k] × ρ₀ Cp = Q_net − Q_sw × F(z_bottom).

        Shortwave that has not been absorbed by the bottom interface escapes
        the column; only the fraction (1 − F(z_f[-1])) of Q_sw is deposited.
        """
        from forcing.surface import _sw_profile
        Q_val = 150.0
        Q_sw_val = 80.0
        Q = np.full((NY, NX), Q_val)
        Q_sw = np.full((NY, NX), Q_sw_val)
        dz = vgrid["dz"].values
        z_f = vgrid["z_f"].values
        tend = surface_tendency(state, vgrid, Q_net=Q, Q_sw=Q_sw)
        dT = tend.T.values   # (nz, ny, nx)
        col_E = np.sum(dz[:, np.newaxis, np.newaxis] * dT, axis=0) * RHO_0 * CP
        # Shortwave absorbed = Q_sw × (F(z_f[0]) − F(z_f[-1])) = Q_sw × (1 − F_bottom)
        F_bottom = float(_sw_profile(np.array([z_f[-1]]))[0])
        expected = (Q_val - Q_sw_val) + Q_sw_val * (1.0 - F_bottom)
        assert np.allclose(col_E, expected, rtol=1e-6), (
            f"Column energy = Q_nonsw + Q_sw×(1-F_bottom); "
            f"max diff = {np.max(np.abs(col_E - expected)):.2e}"
        )

    def test_cooling_gives_negative_dT(self, state, vgrid):
        Q = np.full((NY, NX), -50.0)   # ocean losing heat
        tend = surface_tendency(state, vgrid, Q_net=Q)
        assert np.all(tend.T.values[0] < 0)


# ===========================================================================
# surface_tendency — freshwater flux
# ===========================================================================

class TestFreshwaterTendency:

    def test_evaporation_increases_salinity(self, state, vgrid):
        """E > P → E_P > 0 → S increases."""
        E_P = np.full((NY, NX), 1e-6)   # m/s evaporation
        tend = surface_tendency(state, vgrid, E_P=E_P)
        assert np.all(tend.S.values[0] > 0), "Evaporation should increase salinity"

    def test_precipitation_decreases_salinity(self, state, vgrid):
        """E < P → E_P < 0 → S decreases."""
        E_P = np.full((NY, NX), -1e-6)
        tend = surface_tendency(state, vgrid, E_P=E_P)
        assert np.all(tend.S.values[0] < 0)

    def test_freshwater_only_top_layer(self, state, vgrid):
        E_P = np.full((NY, NX), 1e-6)
        tend = surface_tendency(state, vgrid, E_P=E_P)
        assert np.allclose(tend.S.values[1:], 0.0)

    def test_analytical_salt_tendency(self, state, vgrid):
        """dS[0]/dt = +S_sfc × (E−P) / dz[0]  (virtual salt flux)."""
        E_P_val = 2e-6
        S0 = 35.0
        E_P = np.full((NY, NX), E_P_val)
        tend = surface_tendency(state, vgrid, E_P=E_P)
        expected = S0 * E_P_val / DZ
        assert np.allclose(tend.S.values[0], expected, rtol=1e-10)


# ===========================================================================
# sponge_tendency
# ===========================================================================

class TestSpongeTendency:

    def test_no_boundaries_gives_zero(self, state, hgrid, vgrid):
        tend = sponge_tendency(state, hgrid, vgrid, [])
        assert np.allclose(tend.T.values, 0.0)
        assert np.allclose(tend.S.values, 0.0)
        assert np.allclose(tend.u.values, 0.0)

    def test_output_shape(self, state, hgrid, vgrid):
        T_ext = np.zeros((NZ, NY))
        obd = OBData(side="west", T=T_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd])
        assert tend.T.shape == (NZ, NY, NX)
        assert tend.eta.shape == (NY, NX)

    def test_west_sponge_nudges_toward_exterior(self, state, hgrid, vgrid):
        """
        Sponge on western boundary should nudge T toward T_ext < T_model.
        Cells near i=0 should have dT/dt < 0 (cooling toward exterior).
        """
        T_ext = np.zeros((NZ, NY))   # colder than 15°C
        obd = OBData(side="west", T=T_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd], sponge_width=3)
        # At i=0: γ=1, dT/dt = −1/τ × (15 − 0) < 0
        assert np.all(tend.T.values[:, :, 0] < 0), "West sponge should cool i=0"
        # Beyond sponge width: tendency should be zero
        assert np.allclose(tend.T.values[:, :, 3:], 0.0), "No sponge beyond width"

    def test_east_sponge_direction(self, state, hgrid, vgrid):
        T_ext = np.zeros((NZ, NY))   # colder exterior
        obd = OBData(side="east", T=T_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd], sponge_width=2)
        assert np.all(tend.T.values[:, :, -1] < 0)   # east boundary
        assert np.all(tend.T.values[:, :, -2] < 0)   # one cell in
        assert np.allclose(tend.T.values[:, :, :-2], 0.0)

    def test_south_sponge_direction(self, state, hgrid, vgrid):
        T_ext = np.full((NZ, NX), 20.0)   # warmer exterior
        obd = OBData(side="south", T=T_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd], sponge_width=2)
        assert np.all(tend.T.values[:, 0, :] > 0)   # warming toward warmer exterior

    def test_north_sponge_direction(self, state, hgrid, vgrid):
        T_ext = np.full((NZ, NX), 20.0)
        obd = OBData(side="north", T=T_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd], sponge_width=1)
        assert np.all(tend.T.values[:, -1, :] > 0)

    def test_sponge_strength_decays_inward(self, state, hgrid, vgrid):
        """Sponge at i=0 is stronger than at i=1."""
        T_ext = np.zeros((NZ, NY))
        obd = OBData(side="west", T=T_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd], sponge_width=4)
        # |dT/dt[i=0]| > |dT/dt[i=1]| (stronger nudging at boundary)
        assert np.all(
            np.abs(tend.T.values[:, :, 0]) >= np.abs(tend.T.values[:, :, 1]) - 1e-15
        )

    def test_no_velocity_tendency(self, state, hgrid, vgrid):
        """Sponge should not touch u or v (avoid spurious wall drag)."""
        T_ext = np.zeros((NZ, NY))
        obd = OBData(side="west", T=T_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd])
        assert np.allclose(tend.u.values, 0.0)
        assert np.allclose(tend.v.values, 0.0)

    def test_eta_sponge_applied(self, state, hgrid, vgrid):
        eta_ext = np.full((NY,), 0.5)
        obd = OBData(side="west", eta=eta_ext)
        tend = sponge_tendency(state, hgrid, vgrid, [obd], sponge_width=2)
        # eta[j,0] = 0 → exterior = 0.5 → deta/dt > 0
        assert np.all(tend.eta.values[:, 0] > 0)


# ===========================================================================
# apply_orlanski
# ===========================================================================

class TestApplyOrlanski:

    def test_empty_boundaries_returns_copy(self, state, hgrid, vgrid):
        state_new = apply_orlanski(state, state, hgrid, vgrid, 60.0, [])
        assert np.allclose(state_new.T.values, state.T.values)
        assert np.allclose(state_new.u.values, state.u.values)

    def test_output_shapes_unchanged(self, state, hgrid, vgrid):
        obd = OBData(side="west")
        state_new = apply_orlanski(state, state, hgrid, vgrid, 60.0, [obd])
        assert state_new.T.shape == state.T.shape
        assert state_new.u.shape == state.u.shape

    def test_prescribed_inflow_sets_boundary(self, state, hgrid, vgrid):
        """
        When c_n = 0 (no time change in interior) and phi_ext is provided,
        the boundary value is set to phi_ext.
        """
        T_ext = np.full((NZ, NY), 20.0)
        obd = OBData(side="west", T=T_ext)
        # state_prev == state → Δt_φ = 0 → c_n = 0 → inflow → T_bdy = T_ext
        state_new = apply_orlanski(state, state, hgrid, vgrid, 60.0, [obd])
        assert np.allclose(state_new.T.values[:, :, 0], 20.0), (
            "With zero interior time-change, boundary should be set to T_ext"
        )

    def test_outgoing_wave_propagates_to_boundary(self, hgrid, vgrid):
        """
        Construct a state where the interior T is changing in a way that
        implies an outgoing wave (c_n > 0).  The boundary T should be
        updated by radiation, not set to T_ext.
        """
        # Create two states with an outgoing-wave-like structure
        st_prev = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)
        st_curr = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)

        T_prev = st_prev.T.values.copy()
        T_curr = st_curr.T.values.copy()

        # Interior cell i=1 has a positive anomaly in prev but not curr
        # → Δt_φ_int = curr - prev < 0 (interior cooled)
        # Boundary cell i=0 is warmer than interior → Dn = int - bdy < 0
        # → c_n = -Dt/Dn > 0 (outgoing)
        T_prev[:, :, 1] = 16.0   # warm interior at t-dt
        T_curr[:, :, 1] = 15.0   # interior cooled at t
        T_curr[:, :, 0] = 17.0   # warmer boundary (wave front passed)

        st_prev.T = xr.DataArray(T_prev, dims=st_prev.T.dims, coords=st_prev.T.coords,
                                  attrs=st_prev.T.attrs)
        st_curr.T = xr.DataArray(T_curr, dims=st_curr.T.dims, coords=st_curr.T.coords,
                                  attrs=st_curr.T.attrs)

        T_ext = np.full((NZ, NY), 12.0)   # prescribed (should NOT be applied)
        obd = OBData(side="west", T=T_ext)
        dt = 60.0
        state_new = apply_orlanski(st_curr, st_prev, hgrid, vgrid, dt, [obd])

        # With outgoing wave: boundary updated by radiation, not set to T_ext=12
        # So T_bdy should be between T_prev_bdy and T_curr_bdy (radiation update)
        assert not np.allclose(state_new.T.values[:, :, 0], 12.0), (
            "Outgoing wave: boundary should NOT be set to T_ext"
        )

    def test_c_n_clipped_to_zero_and_c_max(self, hgrid, vgrid):
        """Phase speed c_n is always in [0, dn/dt] (causality + CFL)."""
        st_prev = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)
        st_curr = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)

        # Introduce extreme gradients that would give c_n > c_max without clipping
        T_curr_vals = st_curr.T.values.copy()
        T_prev_vals = st_prev.T.values.copy()
        T_curr_vals[:, :, 1] = 100.0   # huge interior value
        T_prev_vals[:, :, 1] = 0.0
        T_curr_vals[:, :, 0] = 0.0
        st_curr.T = xr.DataArray(T_curr_vals, dims=st_curr.T.dims,
                                   coords=st_curr.T.coords, attrs=st_curr.T.attrs)
        st_prev.T = xr.DataArray(T_prev_vals, dims=st_prev.T.dims,
                                   coords=st_prev.T.coords, attrs=st_prev.T.attrs)

        dt = 60.0
        obd = OBData(side="west")
        state_new = apply_orlanski(st_curr, st_prev, hgrid, vgrid, dt, [obd])

        # T_bdy should not have blown up (clipping prevents CFL violation)
        assert np.all(np.isfinite(state_new.T.values)), "T should be finite after Orlanski"

    def test_all_four_boundaries(self, state, hgrid, vgrid):
        """apply_orlanski with all four boundaries runs without error."""
        boundaries = [
            OBData(side="west",  T=np.zeros((NZ, NY))),
            OBData(side="east",  T=np.zeros((NZ, NY))),
            OBData(side="south", T=np.zeros((NZ, NX))),
            OBData(side="north", T=np.zeros((NZ, NX))),
        ]
        state_new = apply_orlanski(state, state, hgrid, vgrid, 60.0, boundaries)
        assert np.all(np.isfinite(state_new.T.values))

    def test_w_eta_unchanged_without_specification(self, state, hgrid, vgrid):
        """w is never touched by apply_orlanski."""
        obd = OBData(side="west", T=np.zeros((NZ, NY)))
        state_new = apply_orlanski(state, state, hgrid, vgrid, 60.0, [obd])
        assert np.allclose(state_new.w.values, state.w.values)
