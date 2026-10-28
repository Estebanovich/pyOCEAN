"""
N2a task 6 (plan docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md, spec §5): exact discrete budget of
the tracer "variance" Σ V c² of S and T, operator by operator (``variance_budget=True``).

Discrete variance decay (Klingbeil et al. 2014, doi:10.1016/j.ocemod.2014.06.001; numerical advective mixing of
Burchard & Rennau 2008 with c² advected in the same stack): per step

    Σ V¹ c¹² − Σ V⁰ c⁰² = Δt Σ (c_surf − c_num_adv − c_phy_v − c_phy_h − c_num_remap − c_sponge)

to round-off in a closed basin.  Mixing terms > 0 destroy variance; the surface term (freshwater, Lorenz et al.
2021 eq. 43) is a source.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np
import pytest

from tests import n1_cases as nc

SAL = dict(salinity=True, s_coef=0.78, s_ref=30.0)
KEYS_SIGMA = {"num_adv", "phy_v", "phy_h", "surf", "sponge"}
KEYS_ALE = KEYS_SIGMA | {"num_remap"}


def _S(nz: int) -> np.ndarray:
    """Salinity with a horizontal and a vertical gradient (fresher on top and to the east) [g kg⁻¹]."""
    k = np.arange(nz)[:, None, None]
    i = np.arange(nc.NX)[None, None, :]
    j = np.arange(nc.NY)[None, :, None]
    return 30.0 - 0.4 * k / (nz - 1) - 0.3 * i / nc.NX + 0.05 * j / nc.NY


def _fw() -> np.ndarray:
    """E = 2e-5 m s⁻¹ on the west third, P = 3e-5 m s⁻¹ on the east third [m s⁻¹, positive into the ocean]."""
    fw = np.zeros((nc.NY, nc.NX))
    fw[:, : nc.NX // 3] = -2e-5
    fw[:, 2 * nc.NX // 3:] = 3e-5
    return fw


def _model(kind: str, budget: bool = True):
    kw = dict(SAL, kh=50.0, kv=1e-3)
    if budget:
        kw["variance_budget"] = True
    nz = 8 if kind == "hybrid" else nc.NZ
    if kind == "sigma":
        m = nc._sigma(**kw)
    elif kind == "zstar":
        m = nc._ale("zstar", **kw)
    elif kind == "zstar_nomixdiag":
        m = nc._ale("zstar", mix_diagnostic=False, **kw)
    elif kind == "zstar_sponge":
        m = nc._ale("zstar", sponge_S=np.array([29.0, 29.5, 30.0, 30.2]), **kw, **nc._sponge_kw())
    elif kind == "hybrid":
        # nominal top layers 38/8 = 4.75 m: far from the 1 mm regrid floor (task-5 report, concern 1)
        m = nc._ale("hybrid", nz=8, **kw)
    elif kind == "alesigma":
        m = nc._ale("sigma", **kw)
    else:
        raise ValueError(kind)
    st = m.initial_state(nc._eta(), nc._T(nz), S=_S(nz))
    return m, st


KINDS = ["sigma", "zstar", "zstar_nomixdiag", "zstar_sponge", "hybrid", "alesigma"]


@pytest.mark.parametrize("kind", KINDS)
def test_closed_basin_salinity_square_budget_closes(kind):
    m, st = _model(kind)
    fw = _fw()
    for _ in range(10):
        V0 = m.layer_volumes(st.eta, st.h)
        S0, T0 = st.S, st.T
        st = m.step(st, fw=fw)
        V1 = m.layer_volumes(st.eta, st.h)
        b = st.budget
        keys = KEYS_ALE if m.ale else KEYS_SIGMA
        assert set(b) == {f"{c}_{k}" for c in "ST" for k in keys}
        for c, c0, c1 in (("S", S0, st.S), ("T", T0, st.T)):
            for k in keys:
                assert b[f"{c}_{k}"].shape == st.T.shape
            lhs = (V1 * c1**2).sum() - (V0 * c0**2).sum()
            rhs = m.dt * (b[f"{c}_surf"].sum() - b[f"{c}_num_adv"].sum() - b[f"{c}_phy_v"].sum()
                          - b[f"{c}_phy_h"].sum() - b.get(f"{c}_num_remap", np.zeros(1)).sum()
                          - b[f"{c}_sponge"].sum())
            assert abs(lhs - rhs) <= 1e-12 * (V0 * c0**2).sum(), (c, lhs, rhs)
            # diffusion destroys variance; the freshwater is a nonzero source/sink of Σ V c²
            assert b[f"{c}_phy_v"].sum() > 0 and b[f"{c}_phy_h"].sum() > 0
            assert np.abs(b[f"{c}_surf"]).sum() > 0
    if kind == "zstar_sponge":
        assert np.abs(b["S_sponge"]).sum() > 0 and np.abs(b["T_sponge"]).sum() > 0
    else:
        assert not b["S_sponge"].any()


def test_salinity_surface_term_is_the_freshwater_substep():
    """S_surf is task 5's var_surf; zero away from the freshwater, and in the layers it does not touch."""
    m, st = _model("zstar")
    st = m.step(st, fw=_fw())
    b = st.budget
    mid = slice(nc.NX // 3, 2 * nc.NX // 3)
    assert not b["S_surf"][..., mid].any() and not b["T_surf"][..., mid].any()
    assert not b["S_surf"][:-1].any()                                     # thick top layers: no spill
    assert (b["S_surf"][-1, :, : nc.NX // 3] > 0).all()                    # evaporation concentrates: source
    assert (b["S_surf"][-1, :, 2 * nc.NX // 3:] < 0).all()                 # dilution by rain: sink of Σ V S²


@pytest.mark.parametrize("kind", ["sigma", "zstar", "hybrid"])
def test_budget_on_leaves_the_solution_bitwise(kind):
    m0, a = _model(kind, budget=False)
    m1, b = _model(kind, budget=True)
    fw = _fw()
    for _ in range(3):
        a, b = m0.step(a, fw=fw), m1.step(b, fw=fw)
    assert a.budget is None
    for f in ("eta", "u", "v", "T", "S", "h"):
        x, y = getattr(a, f), getattr(b, f)
        if x is None:
            assert y is None
            continue
        assert np.array_equal(x, y), f


def test_budget_off_is_bitwise():
    """With variance_budget=False (default) the golden matrix of N1 is unchanged (tests/test_n1_golden.py) and the
    state carries no budget."""
    import json
    from pathlib import Path
    golden = json.loads((Path(__file__).resolve().parent / "data" / "n1_golden" / "hashes.json").read_text())
    for name in ("sigma_kh_kv", "zstar_vi_arakawa", "zstar_nomixdiag", "hybrid_default"):
        got = nc.run_case(name)
        for step, ref in golden[name].items():
            assert got[int(step)] == ref, (name, step, nc.diff_fields(got[int(step)], ref))
    m = nc._ale("zstar", **SAL)
    assert m.variance_budget is False
    st = m.initial_state(nc._eta(), nc._T(), S=_S(nc.NZ))
    assert m.step(st).budget is None
