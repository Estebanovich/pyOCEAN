"""Tests of the MITgcm internal_wave benchmark tools (spec 2026-10-12-baroclinic-obc-internal-wave §3)."""
from __future__ import annotations

import numpy as np
import pytest

from validation.internal_wave import case

HAVE = (case.SRC / "input" / "delXvar").exists()
pytestmark = pytest.mark.skipif(not HAVE, reason="MITgcm internal_wave inputs not found")


def test_inputs_match_gendata():
    d = case.load_inputs()
    assert d["dx"].shape == (60,) and np.allclose(d["dx"], 13.3e3 / 61)
    assert d["H"][0] == 200.0 and abs(d["H"][-2] - 40.0) < 1e-9 and np.isnan(d["H"][-1])
    assert d["T"].shape == (20, 60) and np.allclose(d["T"][:, 0], d["tref"])
    np.testing.assert_allclose(np.diff(d["tref"]), -0.00509684, rtol=1e-4)


def test_obcs_west_reproduces_obcs_calc():
    # obcs_calc.F: OBWu = U0 * (cos(π rC/rF(Nr+1)) − mean) * sin(2πt/P) * cos(kx (3−2−0.5) Δx);
    # OBWt = tRef + U0 sin(π (k−½)/Nr) sin(2πt/P) sqrt(strat/(α g)) sqrt(ω² − f²)/ω
    d = case.load_inputs()
    t = 0.3 * case.PERIOD
    u, T = case.obcs_west(t, d["dz"])
    zc = -(np.arange(20) + 0.5) * 10.0
    vs = np.cos(np.pi * zc / -200.0)                                       # cos(mz π rC/rF(Nr+1))
    vs = vs - (vs * 10.0).sum() / -200.0                                    # − tmpsum (= Σ vs drF / rF(Nr+1))
    omega = 2 * np.pi / case.PERIOD
    kx = 2 * np.pi / 400.0 * np.sqrt(omega**2 / (1e-6 - omega**2))
    np.testing.assert_allclose(u, 0.024 * vs * np.sin(omega * t) * np.cos(kx * 0.5 * d["dx"][0]), rtol=1e-12)
    strat = 1e-6 / (9.81 * 2e-4)
    dT = 0.024 * np.sin(np.pi * (np.arange(1, 21) - 0.5) / 20) * np.sin(omega * t) \
        * np.sqrt(strat / (2e-4 * 9.81)) * omega / omega
    np.testing.assert_allclose(T, d["tref"] + dT, rtol=1e-12)
    assert abs((u * d["dz"]).sum()) < 1e-15                                 # zero depth mean


def test_mitgcm_depth_partial_cells():
    dz = np.full(20, 10.0)
    np.testing.assert_allclose(case.mitgcm_depth(np.array([200.0, 123.0, 101.5, 100.5, 40.0]), dz),
                               [200.0, 123.0, 102.0, 100.0, 40.0])          # hFac < 0.1 dropped, else ≥ 0.2


def test_read_mitgcm_shapes():
    if not (case.RUNS / "mitgcm" / "run").exists():
        pytest.skip("MITgcm internal_wave run not done yet")
    d = case.read_mitgcm()
    n = d["t"].size
    assert d["U"].shape == (n, 20, 61) and d["T"].shape == (n, 20, 60) and n > 200
    assert abs(d["t"][-1] - 10 * case.PERIOD) < 3600.0
    np.testing.assert_allclose(d["x"], np.cumsum(case.load_inputs()["dx"]) - 0.5 * case.load_inputs()["dx"])
    inp = case.load_inputs()
    np.testing.assert_allclose(np.nan_to_num(case.mitgcm_depth(inp["H"], inp["dz"])), d["depth"], atol=1e-9)


def test_harmonic_recovers_amplitude_and_phase():
    omega = 2 * np.pi / case.PERIOD
    t = np.arange(0, 4 * case.PERIOD, 1800.0)
    x = 0.3 * np.sin(omega * t - np.radians(40.0))[:, None] * np.ones((1, 3)) + 0.01
    amp, ph = case.harmonic(t, x, omega)
    np.testing.assert_allclose(amp, 0.3, rtol=1e-6)
    np.testing.assert_allclose(ph, 40.0, atol=1e-4)


def test_energy_flux_of_a_mode1_wave():
    # mode-1 wave: u = U cos(πz/H) cos θ, p′ = ρ0 (ω/k) U cos(πz/H) cos θ (f = 0, hydrostatic) →
    # mean flux ∫⟨p′u′⟩dz = ρ0 (ω/k) U² H/4.  p′ is computed from T and η as for the models.
    H, dz, N = 200.0, np.full(20, 10.0), 1e-3
    omega = 2 * np.pi / case.PERIOD
    k = np.pi * omega / (N * H)
    zc = -(np.arange(20) + 0.5) * 10.0
    t = np.arange(0, case.PERIOD, case.PERIOD / 200)
    th = omega * t[:, None]
    U = 0.02
    u = U * np.cos(np.pi * zc / H)[None] * np.cos(th)
    Tz = N**2 / (9.81 * 2e-4)
    zeta = (U * k * H / (np.pi * omega)) * np.sin(np.pi * zc / H)[None] * np.cos(th)    # ζ_t = w, w_z = −u_x
    T = Tz * zc[None] - Tz * zeta
    F = case.energy_flux(u[..., None], T[..., None], np.zeros((t.size, 1)), np.array([H]), dz, Tz * zc)
    np.testing.assert_allclose(F.mean(0), case.RHO0 * omega / k * U**2 * H / 4, rtol=0.03)


def test_build_model_domain_starts_at_mitgcm_column_2():
    m, st = case.build_model("zstar")
    assert st.h.shape == (20, 1, 59) and np.isnan(m.H[0, -1]) and m.H[0, 0] == 200.0


def test_cell_centred_u_treats_below_bed_faces_as_zero():
    # MITgcm writes U = 0 below a face's bed, pyOCEAN's level remap gives NaN there: both must centre alike
    from validation.internal_wave import compare
    U_mit = np.array([[[0.2, 0.4, 0.0]]])
    U_py = np.array([[[0.2, 0.4, np.nan]]])
    np.testing.assert_array_equal(compare.cell_u(U_mit), compare.cell_u(U_py))
    np.testing.assert_allclose(compare.cell_u(U_py), [[[0.3, 0.2]]])
