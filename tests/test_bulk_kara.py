"""
Tests for forcing.bulk.bulk_kara — HYCOM flxflg=6 / wndflg=4 bulk fluxes.

The reference table ``tests/data/hycom_kara6_reference.csv`` was produced by
compiling the original HYCOM routines (``validation/hycom_kara/make_reference.sh``):
``cd_coarep`` from mod_momtum.F90 and the flxflg=6 block of thermf.F90, in
8-byte reals.  pyOCEAN must reproduce it to round-off, so a run forced like
HYCOM-IOA gets the same surface fluxes as HYCOM.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from forcing.bulk import albedo_large_yeager, bulk_kara, compute_surface_fluxes

REF = Path(__file__).parent / "data" / "hycom_kara6_reference.csv"


@pytest.fixture(scope="module")
def ref() -> dict[str, np.ndarray]:
    # Two '#' provenance lines precede the column header
    data = np.genfromtxt(REF, delimiter=",", skip_header=2, names=True)
    return {name: data[name] for name in data.dtype.names}


def _run(ref: dict[str, np.ndarray]):
    """Feed the HYCOM inputs to pyOCEAN in canonical units (K, specific humidity)."""
    r = ref["vapmix"]
    return bulk_kara(
        u10=ref["wndx"], v10=ref["wndy"],
        t_air=ref["airt_C"] + 273.15,          # HYCOM-IOA preprocessing: T2 − 273.15
        q_air=r / (1.0 + r),                    # canonical q → HYCOM mixing ratio
        sst=ref["sst_C"], p=ref["pair_Pa"], sss=ref["sss"],
    )


# ===========================================================================
# Exact agreement with compiled HYCOM code
# ===========================================================================

class TestAgainstHYCOM:

    def test_reference_table_present(self, ref):
        assert ref["cd"].size > 500

    def test_drag_coefficient(self, ref):
        np.testing.assert_allclose(_run(ref).cd, ref["cd"], rtol=1e-10)

    def test_latent_coefficient(self, ref):
        np.testing.assert_allclose(_run(ref).ce, ref["cl"], rtol=1e-10)

    def test_sensible_coefficient(self, ref):
        np.testing.assert_allclose(_run(ref).ch, ref["cs"], rtol=1e-10)

    def test_air_density(self, ref):
        np.testing.assert_allclose(_run(ref).rho_air, ref["rho_air"], rtol=1e-10)

    def test_latent_heat_flux(self, ref):
        # HYCOM 'evap' is latent heat flux into the atmosphere (W/m²)
        np.testing.assert_allclose(_run(ref).q_lat, -ref["latent_up_Wm2"], rtol=1e-9, atol=1e-9)

    def test_sensible_heat_flux(self, ref):
        np.testing.assert_allclose(_run(ref).q_sens, -ref["sensible_up_Wm2"], rtol=1e-9, atol=1e-9)

    def test_wind_stress(self, ref):
        f = _run(ref)
        np.testing.assert_allclose(f.tau_x, ref["taux"], rtol=1e-9, atol=1e-14)
        np.testing.assert_allclose(f.tau_y, ref["tauy"], rtol=1e-9, atol=1e-14)

    def test_evaporation_from_latent_heat(self, ref):
        # thermf.F90: emnp = evap·svref/evaplh, svref = 1/1000 m³/kg
        np.testing.assert_allclose(_run(ref).evap, ref["latent_up_Wm2"] / (2.47e6 * 1000.0), rtol=1e-9, atol=1e-18)


# ===========================================================================
# Behaviour beyond the reference table
# ===========================================================================

def _point(**kw):
    base = dict(u10=np.array(8.0), v10=np.array(0.0), t_air=np.array(297.0),
                q_air=np.array(0.015), sst=np.array(27.0), p=np.array(101300.0),
                sss=np.array(36.0))
    base.update(kw)
    return bulk_kara(**base)


class TestBulkKaraBehaviour:

    def test_relative_wind_equals_absolute_wind_minus_current(self):
        rel = _point(u10=np.array(8.0), v10=np.array(2.0), u_ocean=np.array(1.0), v_ocean=np.array(-0.5))
        shifted = _point(u10=np.array(7.0), v10=np.array(2.5))
        for name in ("tau_x", "tau_y", "q_sens", "q_lat", "cd", "ce"):
            assert getattr(rel, name) == pytest.approx(getattr(shifted, name), rel=1e-12), name

    def test_calm_gives_zero_turbulent_fluxes(self):
        # HYCOM has no gustiness or minimum speed in the flux formulas
        f = _point(u10=np.array(0.0), v10=np.array(0.0))
        assert f.tau_x == 0.0 and f.q_sens == 0.0 and f.q_lat == 0.0

    def test_fresher_surface_evaporates_more(self):
        # SSS depression of saturation vapour pressure (Sud & Walker 1997)
        assert _point(sss=np.array(0.0)).evap > _point(sss=np.array(36.0)).evap

    def test_requires_salinity(self):
        with pytest.raises(ValueError, match="sss"):
            bulk_kara(u10=np.array(5.0), v10=np.array(0.0), t_air=np.array(297.0),
                      q_air=np.array(0.015), sst=np.array(27.0), p=np.array(101300.0))

    def test_requires_pressure(self):
        with pytest.raises(ValueError, match="p"):
            bulk_kara(u10=np.array(5.0), v10=np.array(0.0), t_air=np.array(297.0),
                      q_air=np.array(0.015), sst=np.array(27.0), sss=np.array(36.0))

    def test_no_monin_obukhov_diagnostics(self):
        f = _point()
        assert f.zeta is None and f.u10n is None


# ===========================================================================
# Albedo and integration with compute_surface_fluxes
# ===========================================================================

class TestAlbedoAndIntegration:

    def test_large_yeager_albedo(self):
        # forfun.F90 albflg=2: 5.8 % at the equator, 8 % at the poles
        np.testing.assert_allclose(albedo_large_yeager(np.array([0.0, 90.0, -90.0])), [0.058, 0.08, 0.08])

    def test_compute_surface_fluxes_with_kara(self):
        dims = ("j_t", "i_t")
        vals = dict(u10=8.0, v10=-3.0, t2m=297.0, q2m=0.016, msl=101200.0, sw_down=600.0, lw_down=410.0)
        atm = xr.Dataset({k: (dims, np.full((2, 3), v)) for k, v in vals.items()})
        sst = np.full((2, 3), 27.0)
        sss = np.full((2, 3), 36.0)
        out = compute_surface_fluxes(atm, sst=sst, scheme="kara", sss=sss,
                                     albedo=albedo_large_yeager(np.full((2, 3), 20.0)), emissivity=1.0)
        turb = bulk_kara(atm.u10.values, atm.v10.values, atm.t2m.values, atm.q2m.values,
                         sst, p=atm.msl.values, sss=sss)
        np.testing.assert_allclose(out["tau_x"], turb.tau_x, rtol=1e-12)
        np.testing.assert_allclose(out["E_P"], turb.evap, rtol=1e-12)
