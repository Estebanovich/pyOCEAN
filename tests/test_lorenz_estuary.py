"""
N2a task 8 (plan docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md, spec §6): the idealized estuary of
Lorenz, Klingbeil & Burchard (2021, JPO 51, 1319-1333, doi:10.1175/jpo-d-20-0158.1, §4b, Table 1, Fig. 2) as a
pyOCEAN benchmark (validation/lorenz_estuary/).

1. The case reproduces Table 1: the area-integrated E and P, the river discharge and the M2 amplitude, with E and P
   where the caption of Fig. 2 puts them.
2. A short run of experiment A in both configurations closes the discrete s² budget of the region east of the
   transect step by step (task 6 + the TEF transect flux of task 7), and Lorenz's exact relation (eq. 32) equals the
   region's model mixing.
3. In ``sigma_gls`` the vertical diffusivity of S is the constant ``kv`` while GLS sets the momentum viscosity: GLS
   has no buoyancy term and does not act on tracers (N2a Review Focus 5, class (b) missing physics, pinned here).
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from diagnostics.tef import lorenz_mixing
from validation.lorenz_estuary import case

CONFIGS = ["sigma_gls", "ale_sigma"]


@pytest.mark.parametrize("exp", ["A", "B", "C", "D"])
def test_case_matches_table_1(exp):
    """Lorenz (2021) Table 1 and §4b: Qr, Q_evap, Q_precip [m³ s⁻¹] and the M2 amplitude [m], to 1e-12."""
    warnings.simplefilter("ignore")
    row = case.TABLE_1[exp]
    assert row == {"A": (0.0, -5.0, 0.0, 0.0), "B": (5.0, -5.0, 0.0, 0.5), "C": (0.0, -5.0, 5.0, 0.5),
                   "D": (5.0, -5.0, 5.0, 0.5)}[exp]
    Qr, Qe, Qp, amp = row
    m, st, forcing = case.build(exp, "sigma_gls")
    # geometry (§4b): 500 × 1 cells of 200 m, bed 15 m at the ocean (west) → 5 m at the river (east), 60 σ-layers
    assert (m.n, m.H.shape) == (60, (1, 500))
    np.testing.assert_allclose(m.ds, 1.0 / 60, rtol=1e-14)                 # uniform σ-layers
    np.testing.assert_allclose(m.area, 200.0 * 200.0, rtol=1e-14)
    assert m.H[0, 0] > m.H[0, -1] and np.all(np.diff(m.H[0]) < 0)
    np.testing.assert_allclose(np.diff(m.H[0]), -10.0 / 500, rtol=1e-12)
    np.testing.assert_allclose(m.H[0, [0, -1]], [15.0 - 0.01, 5.0 + 0.01], rtol=1e-12)
    np.testing.assert_allclose(st.S, 30.0)
    f = forcing(0)
    assert f["tau_x"] is None and f["tau_y"] is None
    fw = f["fw"]
    assert fw.shape == (1, 500)
    flux = fw * m.area                                                     # P − E per cell [m³ s⁻¹]
    assert abs(flux[flux < 0].sum() - Qe) <= 1e-12 * abs(Qe)
    assert abs(flux[flux > 0].sum() - Qp) <= 1e-12 * max(abs(Qp), 1.0)
    assert np.array_equal(forcing(12345)["fw"], fw)                         # constant in time
    # where E and P act (caption of Fig. 2): A inner half; B–D E in the middle third, P in the east third
    i = np.arange(500)
    if exp == "A":
        assert np.array_equal(fw[0] < 0, i >= 250) and not (fw > 0).any()
    else:
        assert (case.I_THIRD, case.I_TWO_THIRDS) == (167, 333)
        assert np.array_equal(fw[0] < 0, (i >= 167) & (i < 333))
        assert np.array_equal(fw[0] > 0, i >= 333 if Qp > 0 else np.zeros(500, bool))
    # the transect is the west edge of the evaporation zone (dashed lines of Fig. 2), outside the sponge
    face = case.TRANSECT[exp]
    assert face == (250 if exp == "A" else 167) and fw[0, face] < 0 and fw[0, face - 1] == 0
    reg = m.tef_region_budget.mask
    assert m.tef_face == face and m.tef.sign == 1 and reg[0, face:].all() and not reg[0, :face].any()
    # river (east, 'clamped', S = 0) and tide (west, prescribed η = a sin ωt, 'eta', S = 30)
    ext = [m.obc_state(t) for t in np.linspace(0.0, case.T_M2, 97)]
    assert all((e["west"].S == 30.0).all() for e in ext)
    assert m.obc_mode["west"] == "eta"
    case.check_identity_assumptions(m)                                      # kh = 0, no sponge in the region, …
    # model side: the ocean cell follows the prescribed tide (Lorenz l. 843-845), peak at T/4
    n_quarter = int(round(case.T_M2 / 4 / m.dt))
    eta0 = []
    for n in range(n_quarter + 2 if amp > 0 else 5):
        st = m.step(st, **forcing(n))
        eta0.append(st.eta[0, 0])
    assert abs(max(np.abs(eta0)) - amp) <= 1e-3 * max(amp, 1.0), (max(np.abs(eta0)), amp)
    if Qr > 0:
        assert m.open == ("west", "east") and m.obc_mode["east"] == "clamped"
        assert all(abs(-e["east"].transport.sum() - Qr) <= 1e-12 * Qr for e in ext)
        assert all((e["east"].S == 0.0).all() for e in ext)
    else:
        assert m.open == ("west",)


def _short_run(exp: str, config: str, S0: np.ndarray, n_step: int = 50) -> tuple[dict, dict, object]:
    """
    ``n_step`` steps of ``exp`` from S0.  Every step the s² content of the region east of the transect changes by
    Δt (S_surf − S_num_adv − S_phy_v − S_phy_h − S_num_remap − S_sponge) summed over the region plus the transect's
    s² flux (task 6 closure, with the open-boundary term supplied by the in-model TEF of task 7; the river carries
    S = S² = 0): to 100 ε Σ_region V S² (the round-off of one content difference).  Over the run, Lorenz's Me
    (eq. 32) equals the region's M_model_tot.  Its round-off floor is that of the storage terms, ε Σ V S²/T,
    amplified by the coefficients (s²)_in,out/(s_in − s_out) ~ s̄/|s_in − s_out| of eq. (32) (task 7 concern 7:
    relative to M it grows as 1/(M T)); the tolerance is 100 × that floor.  Returns the region inputs, the TEF bulk
    values and the model.
    """
    m, st, forcing = case.build(exp, config, S_init=S0)
    reg = m.tef_region_budget.mask
    eps = np.finfo(float).eps

    def s2_content(s) -> float:
        return (m.layer_volumes(s.eta, s.h) * s.S**2)[:, reg].sum()

    for n in range(n_step):
        c0 = s2_content(st)
        f0 = m.tef.salt2.sum()
        st = m.step(st, **forcing(n))
        b = st.budget
        terms = ((1, "S_surf"), (-1, "S_num_adv"), (-1, "S_phy_v"), (-1, "S_phy_h"), (-1, "S_sponge"))
        src = sum(sgn * b[k][:, reg].sum() for sgn, k in terms)
        src -= b["S_num_remap"][:, reg].sum() if "S_num_remap" in b else 0.0
        lhs = s2_content(st) - c0
        rhs = m.dt * src + (m.tef.salt2.sum() - f0)
        assert abs(lhs - rhs) <= 100 * eps * c0, (n, lhs, rhs)
    r = m.tef_region_budget.inputs()
    bulk = m.tef.bulk()
    assert r["M_sponge"] == 0.0
    assert abs(r["Qsurf"] - (case.TABLE_1[exp][1] + case.TABLE_1[exp][2])) <= 1e-12 * 5.0
    assert r["M_tot"] > 0 and r["M_phy"] > 0
    Me = lorenz_mixing(bulk, r["Qr"], r["Qsurf"], r["dV"], r["dS"], r["dS2"], r["Fs2_surf"])["Me"]
    T = n_step * m.dt
    s_bar, ds = max(bulk["s_in"], bulk["s_out"]), abs(bulk["s_in"] - bulk["s_out"])
    floor = eps * m.tef_region_budget.end[2] / T * s_bar / ds
    assert abs(Me - r["M_tot"]) <= 100 * floor, (Me, r["M_tot"], floor)
    out = case.summary(m)                                                    # the runner's summary agrees
    assert out["Me"] == Me and out["M_model_tot"] == r["M_tot"]
    return r, bulk, m


@pytest.mark.parametrize("config", CONFIGS)
def test_short_run_closes_budget_and_identity(config):
    """
    Experiment A, 50 steps (``_short_run``), from a salinity front at the transect, S = 33 + 3 tanh((x − 50 km)/
    2 km) uniform in depth, instead of the uniform start: the front collapses into an inverse exchange flow through
    the transect within the 50 steps (with S = 30 everywhere s_in = s_out and eq. 32 is 0/0; with a smooth
    x-gradient the barotropic adjustment carries every face-layer the same way in the first hour).  Measured: step
    closure ≤ 2.4 ε Σ V S², identity 0.04 and 0.7 of the floor in σ and ALE-σ.
    """
    warnings.simplefilter("ignore")
    x = (np.arange(case.NX) + 0.5) * case.DX
    S0 = (33.0 + 3.0 * np.tanh((x - 50e3) / 2e3)) * np.ones((case.NZ, case.NY, 1))
    r, bulk, _ = _short_run("A", config, S0)
    assert r["Qr"] == 0.0
    assert bulk["Q_in"] > 0 > bulk["Q_out"] and bulk["s_out"] > bulk["s_in"]   # inverse exchange, as in A


@pytest.mark.parametrize("config", CONFIGS)
def test_short_run_with_river_closes_budget_and_identity(config):
    """
    Experiment D (river, E, P, tide), 50 steps from a classical front at the transect, S = 25 − 5 tanh((x −
    33.4 km)/2 km): the region sees exactly the river's Qr = 5 m³ s⁻¹ through its 'clamped' east face, and the step
    closure and Lorenz's identity hold as in A (the river path of the region's budget).
    """
    warnings.simplefilter("ignore")
    x = (np.arange(case.NX) + 0.5) * case.DX
    S0 = (25.0 - 5.0 * np.tanh((x - case.I_THIRD * case.DX) / 2e3)) * np.ones((case.NZ, case.NY, 1))
    r, bulk, m = _short_run("D", config, S0)
    assert r["Qr"] == pytest.approx(5.0, rel=1e-12)
    assert bulk["Q_in"] > 0 > bulk["Q_out"] and bulk["s_in"] > bulk["s_out"]   # classical exchange


def test_gls_config_uses_constant_tracer_kv(monkeypatch):
    """
    sigma_gls: the implicit vertical diffusion of T and S uses the constant ``kv`` at every interface, while GLS
    sets a varying momentum viscosity (N2a Review Focus 5; Lorenz's GETM uses k-ε for tracers too — class (b)).
    """
    warnings.simplefilter("ignore")
    import core.model.driver as drv
    calls = []
    orig = drv.vertical_implicit

    def spy(c, dz, nu, r_bed, dt, *a, **kw):
        calls.append(np.array(nu, copy=True))
        return orig(c, dz, nu, r_bed, dt, *a, **kw)

    m, st, forcing = case.build("B", "sigma_gls")
    assert m.core.closure == "gls" and m.kv == case.KV > 0
    monkeypatch.setattr(drv, "vertical_implicit", spy)
    for n in range(40):
        calls.clear()
        st = m.step(st, **forcing(n))
    assert len(calls) == 2                                                  # T and S, σ tracer path
    for nu in calls:
        assert np.all(nu == case.KV)
    akv = m.core.gls.akv
    wet = akv[1:-1]
    assert wet.min() < case.KV < wet.max()                                  # GLS sets ν ≠ kv on momentum
