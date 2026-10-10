"""
Tests for the tidal potential (body tide) and self-attraction and loading (SAL).

Formulation follows HYCOM (mod_tides.F90, mod_barotp.F90): the surface
pressure gradient is −∇[g(1−β)η − g η_body] with

    η_body = Σ_c [α_c η_eq,c − η_SAL,c],   α_c = 1 + k_c − h_c (Foreman et al. 1993)
    η_eq,c = A_c f_c cos²φ cos(θ_c + 2λ)       semidiurnal
    η_eq,c = A_c f_c sin2φ cos(θ_c + λ)        diurnal

β is the scalar SAL factor (HYCOM ``tidsal``); η_SAL the per-constituent
SAL read from ``tidal.salReIm`` (HYCOM ``tidein=2``).
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from core.dynamics.free_surface import ImplicitFreeSurface
from core.grid.horizontal import create_cartesian_grid, create_spherical_grid
from core.grid.vertical import compute_hfac, create_z_grid
from core.state import ModelState
from core.timestepping.ab2 import ab2_step_implicit_fs
from forcing.tides import (
    CONSTITUENTS,
    EQ_AMPLITUDE,
    LOVE_ALPHA,
    body_force,
    body_tide,
    equilibrium_constants,
    predict,
    read_sal_hycom,
)

G = 9.81
T = np.array(["2026-10-10T00", "2026-10-10T03", "2026-10-10T07"], dtype="datetime64[ns]")


class TestEquilibriumTide:

    def test_love_factors_from_hycom(self):
        assert LOVE_ALPHA["m2"] == pytest.approx(1 + 0.302 - 0.609)
        assert LOVE_ALPHA["k1"] == pytest.approx(1 + 0.256 - 0.520)
        assert LOVE_ALPHA["o1"] == pytest.approx(1 + 0.298 - 0.603)
        assert LOVE_ALPHA["p1"] == pytest.approx(1 + 0.287 - 0.581)

    def test_semidiurnal_amplitude_cos2_latitude(self):
        lat = np.array([0.0, 30.0, 60.0, 90.0])
        hc = equilibrium_constants(np.zeros(4), lat, ["m2"])[:, 0]
        np.testing.assert_allclose(np.abs(hc), LOVE_ALPHA["m2"] * EQ_AMPLITUDE["m2"] * np.cos(np.deg2rad(lat)) ** 2,
                                   atol=1e-15)

    def test_diurnal_amplitude_sin2_latitude(self):
        lat = np.array([0.0, 45.0, 90.0, -45.0])
        hc = equilibrium_constants(np.zeros(4), lat, ["k1"])[:, 0]
        np.testing.assert_allclose(hc.real, LOVE_ALPHA["k1"] * EQ_AMPLITUDE["k1"] * np.sin(np.deg2rad(2 * lat)),
                                   atol=1e-15)

    def test_semidiurnal_antiphase_90_degrees_apart(self):
        # cos(θ + 2λ): λ = 90° shifts the M2 equilibrium tide by half a cycle
        hc = equilibrium_constants(np.array([0.0, 90.0]), np.array([10.0, 10.0]), ["m2"])
        eta = predict(T, hc, ["m2"])
        np.testing.assert_allclose(eta[:, 1], -eta[:, 0], atol=1e-14)

    def test_without_love_numbers(self):
        hc = equilibrium_constants(np.array([0.0]), np.array([0.0]), ["m2"], love=False)
        assert abs(hc[0, 0]) == pytest.approx(EQ_AMPLITUDE["m2"])


class TestBodyTide:

    def test_matches_prediction_of_equilibrium_constants(self):
        lon, lat = np.array([-100.0, -95.0]), np.array([15.0, 20.0])
        eta = body_tide(lon, lat, T, list(CONSTITUENTS))
        np.testing.assert_allclose(eta, predict(T, equilibrium_constants(lon, lat, list(CONSTITUENTS)),
                                                list(CONSTITUENTS)), atol=1e-15)

    def test_sal_subtracted(self):
        lon, lat = np.array([-100.0]), np.array([15.0])
        sal = equilibrium_constants(lon, lat, list(CONSTITUENTS))           # cancels exactly
        np.testing.assert_allclose(body_tide(lon, lat, T, list(CONSTITUENTS), sal=sal), 0.0, atol=1e-15)

    def test_ramp(self):
        lon, lat = np.array([-100.0]), np.array([15.0])
        full = body_tide(lon, lat, T, ["m2"])
        np.testing.assert_allclose(body_tide(lon, lat, T, ["m2"], ramp=0.25), 0.25 * full)


class TestBodyForce:

    def test_gradient_at_faces(self):
        hg = create_cartesian_grid(nx=5, ny=4, dx=1000.0, dy=2000.0, f0=0.0)
        eta_b = np.arange(5)[None, :] * 0.01 + np.arange(4)[:, None] * 0.02
        gx, gy = body_force(hg, eta_b)
        np.testing.assert_allclose(gx[:, 1:-1], G * 0.01 / 1000.0)
        np.testing.assert_allclose(gy[1:-1, :], G * 0.02 / 2000.0)
        assert np.all(gx[:, [0, -1]] == 0.0) and np.all(gy[[0, -1], :] == 0.0)


class TestScalarSAL:

    def test_seiche_frequency_uses_reduced_gravity(self):
        # SAL β reduces the restoring force: ω = (2√(g(1−β)H)/Δx) sin(kΔx/2)
        beta, nx, dx, H = 0.1, 40, 1000.0, 20.0
        hg = create_cartesian_grid(nx=nx, ny=3, dx=dx, dy=dx, f0=0.0)
        vg = create_z_grid(1, [H])
        hf = compute_hfac(vg, np.full((3, nx), H))
        dt = 200.0
        fs = ImplicitFreeSurface(hg, vg, hf, dt=dt, theta=0.5, sal_beta=beta)
        x = (np.arange(nx) + 0.5) * dx
        eta = 0.1 * np.cos(np.pi * x / (nx * dx))[None, :] * np.ones((3, 1))
        u, v = np.zeros(hf.hfac_u.shape), np.zeros(hf.hfac_v.shape)
        series = [eta[1, 0]]
        for _ in range(50):
            eta, u, v = fs.step(eta, u, v, u.copy(), v.copy())
            series.append(eta[1, 0])
        omega = 2 * np.sqrt(G * (1 - beta) * H) / dx * np.sin(np.pi / nx / 2)
        lam = (1 + 0.5j * omega * dt) / (1 - 0.5j * omega * dt)
        np.testing.assert_allclose(np.array(series) / series[0], np.cos(np.arange(51) * np.angle(lam)), atol=1e-9)

    def test_invalid_beta(self):
        hg = create_cartesian_grid(nx=4, ny=3, dx=1000.0, dy=1000.0, f0=0.0)
        vg = create_z_grid(1, [10.0])
        with pytest.raises(ValueError, match="sal_beta"):
            ImplicitFreeSurface(hg, vg, compute_hfac(vg, np.full((3, 4), 10.0)), dt=10.0, sal_beta=1.0)


class TestReadSAL:

    def test_hycom_salreim_record_order(self, tmp_path: Path):
        # tidal.salReIm: 16 records, Re then Im for m2, s2, k1, o1, n2, p1, k2, q1 (m)
        idm, jdm = 6, 4
        npad = (-idm * jdm) % 4096
        recs = []
        for k in range(8):
            recs += [np.full((jdm, idm), 0.01 * (k + 1)), np.full((jdm, idm), -0.001 * (k + 1))]
        with open(tmp_path / "tidal.salReIm.a", "wb") as f:
            for r in recs:
                r.astype(">f4").tofile(f)
                np.zeros(npad, ">f4").tofile(f)
        lines = ["SAL from TPXO10", "", "", "", f"i/jdm = {idm} {jdm}"]
        for k, c in enumerate(CONSTITUENTS):
            for part, r in (("Re", recs[2 * k]), ("Im", recs[2 * k + 1])):
                lines.append(f"  SAL-{c}{part}: range = {r.min():14.7E} {r.max():14.7E}")
        (tmp_path / "tidal.salReIm.b").write_text("\n".join(lines) + "\n")
        sal = read_sal_hycom(tmp_path / "tidal.salReIm.b")
        assert sal.shape == (jdm, idm, 8)
        np.testing.assert_allclose(sal[0, 0], [0.01 * (k + 1) - 0.001j * (k + 1) for k in range(8)], rtol=1e-6)


class TestBodyTideResponse:

    def test_small_equatorial_basin_follows_equilibrium_tide(self):
        # Basin ≪ tidal wavelength: η ≈ α η_eq − ⟨α η_eq⟩ (volume conserved)
        nx, ny = 40, 3
        hg = create_spherical_grid(nx=nx, ny=ny, dlon=0.1, dlat=0.1, lon0=-100.0, lat0=-0.15)
        vg = create_z_grid(1, [4000.0])
        hf = compute_hfac(vg, np.full((ny, nx), 4000.0))
        dt = 300.0
        fs = ImplicitFreeSurface(hg, vg, hf, dt=dt, theta=0.5)
        lon, lat = hg.lon_t.values, hg.lat_t.values
        st = ModelState.from_grids(hg, vg)
        t0 = np.datetime64("2026-10-10T00:00")
        clock = {"t": t0}

        def ramp(t):                                    # HYCOM tidrmp: 1 − exp(−5 t/τ), τ = 1 day
            return 1.0 - np.exp(-5.0 * ((t - t0) / np.timedelta64(1, "s")) / 86400.0)

        def tendency(s):
            tend = s * 0.0
            eta_b = body_tide(lon, lat, np.array([clock["t"]]), ["m2"], ramp=ramp(clock["t"]))[0]
            gx, gy = body_force(hg, eta_b)
            tend.u.values[:] = gx[None]
            tend.v.values[:] = gy[None]
            return tend

        prev, errs = None, []
        for n in range(int(3 * 86400 / dt)):
            clock["t"] = t0 + np.timedelta64(int((n + 0.5) * dt), "s")
            st, prev = ab2_step_implicit_fs(st, tendency, dt, fs, prev)
            if n * dt > 1.5 * 86400:
                t_now = t0 + np.timedelta64(int((n + 1) * dt), "s")
                eq = body_tide(lon, lat, np.array([t_now]), ["m2"], ramp=ramp(t_now))[0]
                errs.append(np.max(np.abs(st.eta.values - (eq - eq.mean()))))
        amp = LOVE_ALPHA["m2"] * EQ_AMPLITUDE["m2"]
        assert max(errs) < 0.05 * amp * np.deg2rad(4.0)       # ≪ the 4°-wide equilibrium tilt
