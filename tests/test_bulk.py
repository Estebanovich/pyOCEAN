"""
Tests for forcing.bulk — air–sea turbulent and radiative fluxes.

Scheme under test: NCAR / CORE bulk formulae of Large & Yeager (2004, 2009),
the algorithm used by MITgcm pkg/exf and HYCOM flxflg=5.

Validation strategy (no external reference implementation is available
offline, so the tests check properties that any correct implementation of
the published equations must satisfy):
  * published coefficient values and functional forms (LY04 eqs 5–8, LY09 eq 11);
  * the exact neutral limit;
  * Monin–Obukhov log-profile consistency when the wind height ≠ 10 m;
  * stability ordering (unstable > neutral > stable transfer);
  * sign conventions and convergence of the fixed-count iteration.
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest
import xarray as xr

from core.grid.vertical import uniform_z_grid
from core.grid.horizontal import create_cartesian_grid
from core.state import ModelState
from forcing.bulk import (
    KAPPA,
    air_density,
    bulk_ncar,
    cd_neutral_ly04,
    compute_surface_fluxes,
    psi_h_ly04,
    psi_m_ly04,
    qsat_ly04,
    radiative_fluxes,
)
from forcing.surface import surface_tendency

T0K = 273.15


def _neutral_inputs(U: float = 8.0, sst: float = 25.0, rho: float = 1.22):
    """Inputs with zero air–sea temperature and humidity contrast at z = 10 m."""
    sst_k = sst + T0K
    q_air = 0.98 * qsat_ly04(np.array(sst_k), rho)
    return dict(
        u10=np.array(U), v10=np.array(0.0), t_air=np.array(sst_k), q_air=q_air,
        sst=np.array(sst), rho_air=rho, z_t=10.0, gamma=0.0,
    )


# ===========================================================================
# Stability functions (LY04 eq 8b, 8c)
# ===========================================================================

class TestStabilityFunctions:

    def test_zero_at_neutral(self):
        assert psi_m_ly04(np.array(0.0)) == pytest.approx(0.0, abs=1e-14)
        assert psi_h_ly04(np.array(0.0)) == pytest.approx(0.0, abs=1e-14)

    def test_stable_branch_is_minus_five_zeta(self):
        zeta = np.array([0.1, 0.5, 2.0])
        np.testing.assert_allclose(psi_m_ly04(zeta), -5.0 * zeta)
        np.testing.assert_allclose(psi_h_ly04(zeta), -5.0 * zeta)

    def test_unstable_branch_positive_and_heat_exceeds_momentum(self):
        # For ζ < 0 both ψ > 0 and ψ_h > ψ_m (Paulson 1970)
        zeta = np.array([-0.1, -1.0, -5.0])
        pm, ph = psi_m_ly04(zeta), psi_h_ly04(zeta)
        assert np.all(pm > 0.0)
        assert np.all(ph > pm)

    def test_unstable_value(self):
        # ζ = −1: X = 17^¼
        x = 17.0 ** 0.25
        pm = 2 * np.log((1 + x) / 2) + np.log((1 + x * x) / 2) - 2 * np.arctan(x) + np.pi / 2
        ph = 2 * np.log((1 + x * x) / 2)
        assert psi_m_ly04(np.array(-1.0)) == pytest.approx(pm)
        assert psi_h_ly04(np.array(-1.0)) == pytest.approx(ph)

    def test_continuous_at_zero(self):
        eps = 1e-8
        assert psi_m_ly04(np.array(-eps)) == pytest.approx(psi_m_ly04(np.array(eps)), abs=1e-6)
        assert psi_h_ly04(np.array(-eps)) == pytest.approx(psi_h_ly04(np.array(eps)), abs=1e-6)


# ===========================================================================
# Neutral drag coefficient (LY04 eq 6a; LY09 eq 11)
# ===========================================================================

class TestNeutralDrag:

    def test_value_at_10_m_per_s(self):
        # 1e-3 × (2.7/10 + 0.142 + 10/13.09)
        expected = 1e-3 * (0.27 + 0.142 + 10.0 / 13.09)
        assert cd_neutral_ly04(np.array(10.0)) == pytest.approx(expected, rel=1e-12)

    def test_minimum_near_6_m_per_s(self):
        # dC_dN/dU = 0 at U = sqrt(2.7 × 13.09) ≈ 5.94 m/s
        u = np.linspace(2.0, 15.0, 1301)
        assert u[np.argmin(cd_neutral_ly04(u))] == pytest.approx(np.sqrt(2.7 * 13.09), abs=0.02)

    def test_ly09_cap_above_33_m_per_s(self):
        np.testing.assert_allclose(cd_neutral_ly04(np.array([33.0, 40.0, 60.0]), high_wind=True), 2.34e-3)

    def test_ly09_nearly_continuous_at_33(self):
        below = cd_neutral_ly04(np.array(33.0 - 1e-9), high_wind=True)
        assert below == pytest.approx(2.34e-3, rel=2e-3)

    def test_ly09_identical_to_ly04_at_moderate_wind(self):
        # The U⁶ correction is 0.24 % of C_dN at 15 m/s and negligible below
        u = np.array([3.0, 8.0, 15.0])
        np.testing.assert_allclose(
            cd_neutral_ly04(u, high_wind=True), cd_neutral_ly04(u), rtol=3e-3
        )

    def test_ly04_unbounded_at_hurricane_wind(self):
        # Reason the LY09 cap exists (Gulf of Mexico hurricanes)
        assert cd_neutral_ly04(np.array(60.0)) > 4.5e-3


# ===========================================================================
# Thermodynamics
# ===========================================================================

class TestThermodynamics:

    @pytest.mark.parametrize("t_c", [0.0, 10.0, 20.0, 30.0])
    def test_qsat_close_to_bolton(self, t_c):
        # Bolton (1980) e_s with p = 1013.25 hPa.  LY04 eq 5 is a vapour density
        # divided by ρ_a, so it only agrees when evaluated with the actual air
        # density (with LY04's fixed 1.22 kg/m³ the error reaches ~6 % at 0 °C).
        es = 6.112 * np.exp(17.67 * t_c / (t_c + 243.5))
        q_ref = 0.622 * es / (1013.25 - 0.378 * es)
        rho = air_density(np.array(t_c + T0K), np.array(q_ref), np.array(101325.0))
        assert qsat_ly04(np.array(t_c + T0K), rho) == pytest.approx(q_ref, rel=0.025)

    def test_qsat_increases_with_temperature(self):
        q = qsat_ly04(np.array([280.0, 290.0, 300.0]), 1.22)
        assert np.all(np.diff(q) > 0)

    def test_dry_air_density_standard_atmosphere(self):
        assert air_density(np.array(288.15), np.array(0.0), np.array(101325.0)) == pytest.approx(1.225, rel=1e-3)

    def test_moist_air_lighter_than_dry(self):
        dry = air_density(np.array(300.0), np.array(0.0), np.array(101325.0))
        moist = air_density(np.array(300.0), np.array(0.02), np.array(101325.0))
        # 1 + 0.608 q  →  about 1.2 % lighter at q = 0.02
        assert moist / dry == pytest.approx(1.0 / (1.0 + 0.608 * 0.02), rel=1e-12)


# ===========================================================================
# bulk_ncar — neutral limit
# ===========================================================================

class TestBulkNcarNeutral:

    @pytest.mark.parametrize("U", [3.0, 8.0, 20.0])
    def test_coefficients_equal_neutral_values(self, U):
        f = bulk_ncar(**_neutral_inputs(U=U))
        cdn = cd_neutral_ly04(np.array(U))
        assert f.cd == pytest.approx(cdn, rel=1e-12)
        assert f.ce == pytest.approx(34.6e-3 * np.sqrt(cdn), rel=1e-12)
        assert f.ch == pytest.approx(32.7e-3 * np.sqrt(cdn), rel=1e-12)

    def test_no_heat_or_moisture_flux(self):
        f = bulk_ncar(**_neutral_inputs())
        assert f.q_sens == pytest.approx(0.0, abs=1e-10)
        assert f.q_lat == pytest.approx(0.0, abs=1e-10)
        assert f.evap == pytest.approx(0.0, abs=1e-16)

    def test_stress_is_quadratic_drag(self):
        U = 8.0
        f = bulk_ncar(**_neutral_inputs(U=U))
        assert f.tau_x == pytest.approx(1.22 * cd_neutral_ly04(np.array(U)) * U * U, rel=1e-12)
        assert f.tau_y == pytest.approx(0.0, abs=1e-15)

    def test_log_profile_consistency_for_20_m_wind(self):
        # Neutral log law: U(z) = u*/κ · ln(z/z0).  With wind measured at 20 m the
        # converged solution must satisfy U(20)/U10N = ln(20/z0)/ln(10/z0), with
        # z0 = 10·exp(−κ/√C_dN(U10N)).
        U20 = 12.0
        f = bulk_ncar(**{**_neutral_inputs(U=U20), "z_u": 20.0, "z_t": 20.0}, n_iter=40)
        ustar = np.sqrt(f.cd) * U20
        # u* = κ U10N / ln(10/z0) and √C_dN = κ / ln(10/z0)  ⇒  U10N = u*/√C_dN
        u10n = ustar / np.sqrt(cd_neutral_ly04(np.array(f.u10n)))
        assert u10n == pytest.approx(f.u10n, rel=1e-6)
        z0 = 10.0 * np.exp(-KAPPA / np.sqrt(cd_neutral_ly04(np.array(f.u10n))))
        assert U20 == pytest.approx(ustar / KAPPA * np.log(20.0 / z0), rel=1e-6)


# ===========================================================================
# bulk_ncar — stability, signs, conventions
# ===========================================================================

def _case(u=5.0, v=0.0, t_air_c=24.0, rh=0.8, sst=27.0, **kw):
    """Air at 2 m with relative humidity rh (w.r.t. LY04 qsat)."""
    t_k = t_air_c + T0K
    return bulk_ncar(
        u10=np.array(u), v10=np.array(v), t_air=np.array(t_k),
        q_air=rh * qsat_ly04(np.array(t_k), 1.22), sst=np.array(sst),
        rho_air=1.22, **kw,
    )


class TestBulkNcarPhysics:

    def test_cold_dry_air_over_warm_sea_cools_and_evaporates(self):
        # Northerly "norte" over the Gulf of Mexico
        f = _case(u=12.0, t_air_c=15.0, rh=0.5, sst=26.0)
        assert f.q_sens < 0.0
        assert f.q_lat < 0.0
        assert f.evap > 0.0

    def test_warm_moist_air_over_cold_sea_warms_ocean(self):
        f = _case(u=6.0, t_air_c=28.0, rh=0.98, sst=20.0)
        assert f.q_sens > 0.0
        assert f.q_lat > 0.0      # condensation (dew)
        assert f.evap < 0.0

    def test_latent_flux_equals_lv_times_evaporation(self):
        f = _case()
        assert f.q_lat == pytest.approx(-2.5e6 * 1000.0 * f.evap, rel=1e-12)

    def test_unstable_enhances_and_stable_suppresses_transfer(self):
        neutral = bulk_ncar(**_neutral_inputs(U=4.0, sst=25.0), z_u=10.0)
        unstable = _case(u=4.0, t_air_c=22.0, rh=0.8, sst=25.0, z_t=10.0, gamma=0.0)
        stable = _case(u=4.0, t_air_c=28.0, rh=0.8, sst=25.0, z_t=10.0, gamma=0.0)
        assert unstable.cd > neutral.cd > stable.cd
        assert unstable.ce > stable.ce

    def test_stress_parallel_to_relative_wind(self):
        f = _case(u=6.0, v=-8.0)
        assert f.tau_x / f.tau_y == pytest.approx(6.0 / -8.0, rel=1e-12)

    def test_relative_wind_removes_stress_when_air_moves_with_ocean(self):
        f = _case(u=0.3, v=0.2, u_ocean=np.array(0.3), v_ocean=np.array(0.2))
        assert f.tau_x == pytest.approx(0.0, abs=1e-15)
        assert f.tau_y == pytest.approx(0.0, abs=1e-15)

    def test_relative_wind_reduces_stress_for_following_current(self):
        absolute = _case(u=8.0)
        relative = _case(u=8.0, u_ocean=np.array(1.0))
        assert relative.tau_x < absolute.tau_x

    def test_calm_wind_gives_finite_fluxes(self):
        f = _case(u=0.0, v=0.0)
        for name in ("tau_x", "tau_y", "q_sens", "q_lat", "evap", "cd"):
            assert np.isfinite(getattr(f, name)), name
        assert f.q_sens != 0.0   # minimum wind speed keeps exchange alive

    def test_two_iterations_converged_within_two_percent(self):
        # LY04 §2.1: two iterations are sufficient
        for u, ta, sst in [(3.0, 22.0, 27.0), (8.0, 15.0, 26.0), (5.0, 29.0, 26.0)]:
            f2 = _case(u=u, t_air_c=ta, sst=sst, n_iter=2)
            f30 = _case(u=u, t_air_c=ta, sst=sst, n_iter=30)
            for name in ("tau_x", "q_sens", "q_lat"):
                assert getattr(f2, name) == pytest.approx(getattr(f30, name), rel=0.02), (u, name)

    def test_returned_zeta_satisfies_monin_obukhov_definition(self):
        # ζ = κ g z_u / u*² · (θ*/θ_v + q*/(q + 1/0.608))     (LY04 eq 8a)
        # with θ* = Q_S/(ρ c_pa u*) and q* = F_E/(ρ u*).  Converged run.
        rho, z_u = 1.22, 10.0
        t_k, q = 299.0, 0.012
        f = bulk_ncar(u10=np.array(4.0), v10=np.array(0.0), t_air=np.array(t_k), q_air=np.array(q),
                      sst=np.array(28.0), rho_air=rho, z_t=10.0, gamma=0.0, n_iter=60)
        tstar = f.q_sens / (rho * 1000.5 * f.ustar)
        qstar = (f.q_lat / 2.5e6) / (rho * f.ustar)
        zeta = KAPPA * 9.81 * z_u / f.ustar ** 2 * (tstar / (t_k * (1 + 0.608 * q)) + qstar / (q + 1 / 0.608))
        assert f.zeta == pytest.approx(zeta, rel=1e-8)

    def test_humidity_alone_makes_surface_layer_unstable(self):
        # Δθ = 0 but moist sea surface: water vapour is lighter → ζ < 0
        sst = 26.0
        t_k = sst + T0K
        f = bulk_ncar(u10=np.array(4.0), v10=np.array(0.0), t_air=np.array(t_k),
                      q_air=0.6 * qsat_ly04(np.array(t_k), 1.22), sst=np.array(sst),
                      rho_air=1.22, z_t=10.0, gamma=0.0)
        assert f.zeta < 0.0
        assert f.cd > cd_neutral_ly04(np.array(4.0))

    def test_2m_inputs_equivalent_to_profile_shifted_10m_inputs(self):
        # Similarity profile: θ(z_u) = θ(z_t) − θ*/κ [ln(z_t/z_u) + ψ_h(ζ_u) − ψ_h(ζ_t)]
        # Feeding the 10 m values implied by a converged 2 m run must give the same fluxes.
        rho, z_t, z_u = 1.22, 2.0, 10.0
        kw = dict(u10=np.array(5.0), v10=np.array(0.0), sst=np.array(27.0), rho_air=rho,
                  gamma=0.0, n_iter=60)
        t2, q2 = 296.0, 0.013
        f2 = bulk_ncar(t_air=np.array(t2), q_air=np.array(q2), z_t=z_t, **kw)
        tstar = f2.q_sens / (rho * 1000.5 * f2.ustar)
        qstar = (f2.q_lat / 2.5e6) / (rho * f2.ustar)
        shift = np.log(z_t / z_u) + psi_h_ly04(f2.zeta) - psi_h_ly04(f2.zeta * z_t / z_u)
        f10 = bulk_ncar(t_air=np.array(t2 - tstar / KAPPA * shift),
                        q_air=np.array(q2 - qstar / KAPPA * shift), z_t=z_u, **kw)
        for name in ("tau_x", "q_sens", "q_lat"):
            assert getattr(f10, name) == pytest.approx(getattr(f2, name), rel=1e-6), name

    def test_air_density_from_pressure_when_not_given(self):
        t_k = np.array(300.0)
        q = np.array(0.015)
        p = np.array(101000.0)
        kw = dict(u10=np.array(7.0), v10=np.array(0.0), t_air=t_k, q_air=q, sst=np.array(28.0))
        f_p = bulk_ncar(**kw, p=p)
        f_rho = bulk_ncar(**kw, rho_air=air_density(t_k + 0.0098 * 2.0, q, p))
        assert f_p.tau_x == pytest.approx(f_rho.tau_x, rel=1e-3)

    def test_requires_density_or_pressure(self):
        with pytest.raises(ValueError, match="rho_air.*p"):
            bulk_ncar(u10=np.array(5.0), v10=np.array(0.0), t_air=np.array(300.0),
                      q_air=np.array(0.015), sst=np.array(27.0))

    def test_works_on_2d_arrays(self):
        shape = (4, 5)
        f = bulk_ncar(
            u10=np.full(shape, 7.0), v10=np.full(shape, 1.0), t_air=np.full(shape, 296.0),
            q_air=np.full(shape, 0.014), sst=np.full(shape, 26.0), rho_air=1.2,
        )
        assert f.tau_x.shape == shape
        assert f.q_lat.shape == shape


# ===========================================================================
# Radiation
# ===========================================================================

class TestRadiativeFluxes:

    def test_shortwave_net_of_albedo(self):
        sw, _ = radiative_fluxes(np.array(800.0), np.array(400.0), np.array(25.0), albedo=0.066)
        assert sw == pytest.approx(800.0 * (1 - 0.066))

    def test_longwave_balance_point(self):
        sst = 25.0
        lw_down = 5.670374419e-8 * (sst + T0K) ** 4
        _, lw = radiative_fluxes(np.array(0.0), np.array(lw_down), np.array(sst))
        assert lw == pytest.approx(0.0, abs=1e-9)

    def test_longwave_uses_emissivity_on_both_terms(self):
        _, lw = radiative_fluxes(np.array(0.0), np.array(350.0), np.array(25.0), emissivity=0.97)
        expected = 0.97 * (350.0 - 5.670374419e-8 * (25.0 + T0K) ** 4)
        assert lw == pytest.approx(expected, rel=1e-12)


# ===========================================================================
# compute_surface_fluxes → surface_tendency
# ===========================================================================

def _atm_t_points(ny: int, nx: int, precip: float | None = 2e-7) -> xr.Dataset:
    dims = ("j_t", "i_t")
    vals = dict(u10=8.0, v10=-3.0, t2m=297.0, q2m=0.016, msl=101200.0, sw_down=600.0, lw_down=410.0)
    if precip is not None:
        vals["precip"] = precip
    return xr.Dataset({k: (dims, np.full((ny, nx), v)) for k, v in vals.items()})


class TestComputeSurfaceFluxes:

    def test_keys_match_surface_tendency_arguments(self):
        out = compute_surface_fluxes(_atm_t_points(3, 4), sst=np.full((3, 4), 27.0))
        params = set(inspect.signature(surface_tendency).parameters)
        assert set(out) <= params
        assert set(out) == {"tau_x", "tau_y", "Q_net", "Q_sw", "E_P"}

    def test_q_net_is_sum_of_components(self):
        atm = _atm_t_points(2, 2)
        sst = np.full((2, 2), 27.0)
        out = compute_surface_fluxes(atm, sst=sst, albedo=0.066, emissivity=0.97)
        sw, lw = radiative_fluxes(atm.sw_down.values, atm.lw_down.values, sst, albedo=0.066, emissivity=0.97)
        turb = bulk_ncar(atm.u10.values, atm.v10.values, atm.t2m.values, atm.q2m.values, sst, p=atm.msl.values)
        np.testing.assert_allclose(out["Q_net"], sw + lw + turb.q_sens + turb.q_lat, rtol=1e-12)
        np.testing.assert_allclose(out["Q_sw"], sw, rtol=1e-12)

    def test_e_minus_p(self):
        atm = _atm_t_points(2, 2, precip=2e-7)
        sst = np.full((2, 2), 27.0)
        out = compute_surface_fluxes(atm, sst=sst)
        turb = bulk_ncar(atm.u10.values, atm.v10.values, atm.t2m.values, atm.q2m.values, sst, p=atm.msl.values)
        np.testing.assert_allclose(out["E_P"], turb.evap - 2e-7, rtol=1e-12)

    def test_missing_precip_treated_as_zero(self):
        sst = np.full((2, 2), 27.0)
        dry = compute_surface_fluxes(_atm_t_points(2, 2, precip=None), sst=sst)
        wet = compute_surface_fluxes(_atm_t_points(2, 2, precip=1e-6), sst=sst)
        np.testing.assert_allclose(dry["E_P"] - wet["E_P"], 1e-6, rtol=1e-9)

    def test_unknown_scheme_rejected(self):
        with pytest.raises(ValueError, match="scheme"):
            compute_surface_fluxes(_atm_t_points(2, 2), sst=np.full((2, 2), 27.0), scheme="magic")

    def test_drives_surface_tendency(self):
        nx, ny = 5, 4
        hgrid = create_cartesian_grid(nx=nx, ny=ny, dx=2000.0, dy=2000.0, f0=1e-4)
        vgrid = uniform_z_grid(nz=3, dz_uniform=10.0)
        state = ModelState.from_grids(hgrid, vgrid, T_init=27.0, S_init=36.0)
        fluxes = compute_surface_fluxes(_atm_t_points(ny, nx), sst=state.T.values[0])
        tend = surface_tendency(state, vgrid, **fluxes)
        assert np.all(tend.u.values[0, :, 1:-1] > 0.0)    # eastward wind accelerates u
        assert np.all(np.isfinite(tend.T.values))
