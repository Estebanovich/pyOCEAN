"""
N2a task 7 (plan docs/superpowers/plans/2026-10-28-n2a-salinity-freshwater.md, spec §6): TEF on a transect during
the run, with the step's effective face fluxes, and the exact mixing relation of Lorenz et al. (2021, JPO 51,
1319-1333, doi:10.1175/jpo-d-20-0158.1, eq. 32).

With the transect fluxes of volume, s and s² (the S² member of the tracer stack) taken from the advection that
updated the cells, the Knudsen relations (30)–(31) and the s² budget (28) of the region east of the transect hold
discretely, so

    Me = [s_out (s²)_in − s_in (s²)_out]/(s_in − s_out) (Qr − dV + Qsurf)
         + [(s²)_in − (s²)_out]/(s_in − s_out) dS − dS2 + F_surf^{s²}                      (Lorenz 2021, eq. 32)

equals the model's total mixing in the region, Σ (S_num_adv + S_phy_v + S_phy_h (+ S_num_remap)), to round-off.
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from boundaries.obcs import OBData
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from diagnostics.tef import TEFTransect, lorenz_mixing

NX, NY, NZ = 40, 1, 20
DX, DT, NSTEP = 200.0, 30.0, 200
H = np.linspace(15.0, 5.0, NX)[None, :]                     # bed 15 m at the ocean (west) → 5 m at the river
QR = 5.0                                                     # river discharge [m³ s⁻¹]
FACE = NX // 2                                               # transect: the middle u-face column
REGION = np.zeros((NY, NX), bool)
REGION[:, FACE:] = True                                      # the estuary: cells east of the transect
E_FLUX = -QR / (REGION.sum() * DX * DX)                      # fw = P − E [m s⁻¹]: E integrates to −5 m³ s⁻¹


def _state(t: float) -> dict:
    """West: ocean at S = 30, at rest; east: the river, S = 0, exact transport −Qr (into the domain)."""
    out = {}
    for edge, depth, S, q in (("west", H[0, 0], 30.0, None), ("east", H[0, -1], 0.0, -QR)):
        zw = np.linspace(0.0, -depth, 5)[:, None]
        u = np.zeros((4, NY)) if q is None else np.full((4, NY), q / (depth * DX))
        out[edge] = OBData(side=edge, T=np.full((4, NY), 10.0), S=np.full((4, NY), S), u_normal=u, zw=zw,
                           transport=None if q is None else np.full(NY, q))
    return out


def _model(config: str, **kw) -> SigmaBaroclinic:
    base = dict(rho0=1000.0, t_coef=0.0, salinity=True, s_coef=0.78, s_ref=30.0, z0=None, closure="constant",
                nu_const=1e-4, ah=1.0, theta=0.6, d_min=0.05, kv=1e-3, kh=0.0, variance_budget=True,
                open_boundaries=("west", "east"), open_boundary_mode={"east": "clamped"}, sponge_cells=2,
                open_boundary_state=_state, tef_face=FACE, tef_region=REGION)
    if config == "sigma_rk3":
        base.update(tracer_time="ssprk3")
    elif config == "sigma_pc":
        base.update(tracer_time="pc")
    elif config == "alesigma_fct":
        base.update(vertical_coordinate="ale", regrid="sigma", pgf="fv", tracer_limiter="fct")
    else:
        raise ValueError(config)
    base.update(kw)
    grid = create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DX, f0=0.0)
    return SigmaBaroclinic(grid, H, dt=DT, n_layers=NZ, **base)


def _S0() -> np.ndarray:
    """30 g kg⁻¹ at the ocean to 10 at the river end, uniform in depth: gravitational adjustment drives an
    exchange flow through the transect (salty inflow at depth, fresher outflow on top)."""
    return (30.0 - 20.0 * np.arange(NX) / (NX - 1))[None, None, :] * np.ones((NZ, NY, NX))


def _fw() -> np.ndarray:
    fw = np.zeros((NY, NX))
    fw[REGION] = E_FLUX
    return fw


CONFIGS = ["sigma_rk3", "sigma_pc", "alesigma_fct"]


@pytest.mark.parametrize("config", CONFIGS)
def test_lorenz_identity_holds_to_round_off(config):
    """
    Lorenz (2021) eq. (32) from the in-model TEF equals the region's total model mixing to 1e-10 (brief).  The
    round-off floor of the comparison is ~ε Σ_region V S² / (M Δt) (content differences over steps; measured
    0.2–6e-12 here, and 1e-11–5e-11 with Δt = 10 s and kv = 1e-4, scaling as 1/(M Δt)): Δt = 30 s and kv = 1e-3
    keep a margin of ≥ 15 below the tolerance.
    """
    warnings.simplefilter("ignore")
    m = _model(config)
    st = m.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), 10.0), S=_S0())
    fw = _fw()
    reg = REGION[None]

    def content(s):
        V = m.layer_volumes(s.eta, s.h)
        return (V * reg).sum(), (V * s.S * reg).sum(), (V * s.S**2 * reg).sum()

    c0 = content(st)
    M_tot = F_surf = M_sponge = 0.0
    for _ in range(NSTEP):
        st = m.step(st, fw=fw)
        b = st.budget
        M_tot += (b["S_num_adv"] + b["S_phy_v"] + b["S_phy_h"] + b.get("S_num_remap", 0.0))[:, REGION].sum()
        F_surf += b["S_surf"][:, REGION].sum()
        M_sponge += np.abs(b["S_sponge"][:, REGION]).sum()
    T = NSTEP * DT
    c1 = content(st)
    dV, dS, dS2 = ((b1 - b0) / T for b0, b1 in zip(c0, c1))
    M_tot, F_surf = M_tot / NSTEP, F_surf / NSTEP
    Qsurf = E_FLUX * REGION.sum() * DX * DX                                   # −5 m³ s⁻¹
    assert M_sponge == 0.0                                                     # no sponge cells in the region
    assert m.salinity and isinstance(m.tef, TEFTransect)
    bulk = m.tef.bulk()
    assert bulk["Q_in"] > 0 > bulk["Q_out"] and bulk["s_in"] > bulk["s_out"]   # an exchange flow through the transect
    # discrete Knudsen consistency: volume and salt of the region (Lorenz 2021 eqs. 26, 27)
    assert abs(dV - (bulk["Q_in"] + bulk["Q_out"] + QR + Qsurf)) <= 1e-10 * QR
    assert abs(dS - (bulk["Qs_in"] + bulk["Qs_out"])) <= 1e-10 * abs(bulk["Qs_in"])
    Me = lorenz_mixing(bulk, QR, Qsurf, dV, dS, dS2, F_surf)["Me"]
    assert M_tot > 0
    assert abs(Me - M_tot) <= 1e-10 * abs(M_tot), (Me, M_tot)
    # the model's region accumulator gives the same inputs
    r = m.tef_region_budget.inputs()
    assert r["Qr"] == pytest.approx(QR, rel=1e-12) and r["Qsurf"] == pytest.approx(Qsurf, rel=1e-12)
    for key, val in (("dV", dV), ("dS", dS), ("dS2", dS2), ("Fs2_surf", F_surf), ("M_tot", M_tot)):
        assert r[key] == pytest.approx(val, rel=1e-9, abs=1e-12 * abs(M_tot)), key
    Me2 = lorenz_mixing(bulk, r["Qr"], r["Qsurf"], r["dV"], r["dS"], r["dS2"], r["Fs2_surf"])["Me"]
    assert abs(Me2 - r["M_tot"]) <= 1e-10 * abs(r["M_tot"])


def test_tef_off_changes_nothing():
    """With tef_face=None (default) the step is the one of task 6, bit for bit, and the model has no TEF."""
    warnings.simplefilter("ignore")
    a = _model("sigma_rk3", tef_face=None, tef_region=None)
    b = _model("sigma_rk3")
    assert a.tef is None and a.tef_region_budget is None
    sa = a.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), 10.0), S=_S0())
    sb = b.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), 10.0), S=_S0())
    for _ in range(5):
        sa, sb = a.step(sa, fw=_fw()), b.step(sb, fw=_fw())
    for f in ("eta", "u", "v", "T", "S"):
        assert np.array_equal(getattr(sa, f), getattr(sb, f)), f


@pytest.mark.parametrize("scheme", ["rk3", "pc", "fct", "fct_numpy"])
def test_advect_returns_the_effective_face_flux(scheme):
    """The returned face flux is the one that updated the cells: V¹c¹ − V⁰c⁰ summed east of the face equals
    Δt × (flux through the face) when the east edge is closed."""
    warnings.simplefilter("ignore")
    kw = {"rk3": dict(), "pc": dict(tracer_time="pc"), "fct": dict(vertical_coordinate="ale", regrid="sigma",
                                                                    pgf="fv", tracer_limiter="fct"),
          "fct_numpy": dict(vertical_coordinate="ale", regrid="sigma", pgf="fv", tracer_limiter="fct")}[scheme]
    m = _model("sigma_rk3", open_boundaries=(), open_boundary_mode=None, open_boundary_state=None, sponge_cells=0,
               tef_face=None, tef_region=None, **kw)
    st = m.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), 10.0), S=_S0())
    for _ in range(20):
        st = m.step(st)
    rng = np.random.default_rng(0)
    # a front at the transect plus noise: the FCT limiter acts on the transect faces (C < 1 there)
    S = np.where(np.arange(NX) < FACE, 30.0, 10.0)[None, None, :] + rng.normal(0.0, 2.0, (NZ, NY, NX))
    C = np.stack([S, S**2])
    Qx = rng.normal(0.0, 0.5, (NZ, NY, NX + 1))
    Qx[..., [0, -1]] = 0.0
    Qy = np.zeros((NZ, NY + 1, NX))
    Fz = np.zeros((NZ + 1, NY, NX))
    V0 = m.layer_volumes(st.eta, st.h)
    V1 = V0 - DT * (Qx[..., 1:] - Qx[..., :-1])                       # continuity with Fz = 0
    ok = V0 > 0
    adv = m._tracer_advection
    if scheme == "fct":
        C1, fx = adv.advect_fct(C, V0, V1, Qx, Qy, ok, flux_face_i=FACE)
    elif scheme == "fct_numpy":
        C1, fx = adv.advect_fct_numpy(C, V0, V1, Qx, Qy, ok, flux_face_i=FACE)
    elif scheme == "pc":
        C1, fx = adv.advect_numpy(C, V0, V1, Qx, Qy, Fz, ok, flux_face_i=FACE)
    else:
        C1, fx = adv.advect(C, V0, V1, Qx, Qy, Fz, ok, flux_face_i=FACE)
    assert fx.shape == (2, NZ, NY)
    east = (V1 * C1)[..., FACE:].sum(axis=(1, 2, 3)) - (V0 * C)[..., FACE:].sum(axis=(1, 2, 3))
    scale = (V0 * C)[..., FACE:].sum(axis=(1, 2, 3))                  # round-off of the content difference
    assert (np.abs(east - DT * fx.sum(axis=(1, 2))) <= 1e-14 * scale).all(), (east, DT * fx.sum(axis=(1, 2)))
    # and without the argument the result is the same array, bit for bit
    C2 = (adv.advect_fct(C, V0, V1, Qx, Qy, ok) if scheme == "fct" else
          adv.advect_fct_numpy(C, V0, V1, Qx, Qy, ok) if scheme == "fct_numpy" else
          adv.advect_numpy(C, V0, V1, Qx, Qy, Fz, ok) if scheme == "pc" else adv.advect(C, V0, V1, Qx, Qy, Fz, ok))
    assert np.array_equal(C1, C2)
    if scheme.startswith("fct"):                                       # the limited flux, not F_H
        _, fH = adv.advect_fct_numpy(C, V0, V1, Qx, Qy, ok, unlimited=True, flux_face_i=FACE)
        assert not np.allclose(fx, fH)


def test_tef_bins_and_split():
    """Two face-layers at s = 31 (inflow) and 25 (outflow): bulk values, in/out by the sign of q(s) per bin
    (MacCready 2011), and the sign convention (+ into the region)."""
    tef = TEFTransect(face_i=3, sign=-1)
    Q = np.array([[-2.0], [1.5]])                                       # model x-direction; sign −1: into = −x
    s = np.array([[31.0], [25.0]])
    for _ in range(4):
        tef.add(Q, Q * s, Q * s**2, dt=10.0)
    b = tef.bulk()
    assert b["Q_in"] == pytest.approx(2.0) and b["Q_out"] == pytest.approx(-1.5)
    assert b["s_in"] == pytest.approx(31.0) and b["s_out"] == pytest.approx(25.0)
    assert b["s2_in"] == pytest.approx(961.0) and b["s2_out"] == pytest.approx(625.0)
    assert b["Q0"] == pytest.approx(0.5)
    assert tef.edges.shape == (251,) and tef.q.shape == (250,)


def test_lorenz_relations_reduce_as_in_the_paper():
    """With constancy ((s²)_in = s_in², (s²)_out = s_out²) eq. (32) reduces to (33); without storage to (34); both
    to (35), Mcp = s_in s_out (Qr + Qsurf) + F (Lorenz 2021)."""
    s_in, s_out, Qin = 30.0, 20.0, 10.0
    Qr, Qsurf, F = 3.0, -1.0, 400.0
    Qout = -(Qin + Qr + Qsurf)                                          # dV = 0
    bulk = dict(Q_in=Qin, Q_out=Qout, s_in=s_in, s_out=s_out, s2_in=s_in**2, s2_out=s_out**2)
    dS = Qin * s_in + Qout * s_out
    r = lorenz_mixing(bulk, Qr, Qsurf, 0.0, dS, 0.0, F)
    assert r["Me"] == pytest.approx(r["Mc"]) and r["Mp"] == pytest.approx(r["Mcp"])
    # with dS = 0 as well: all four agree and equal s_in s_out (Qr + Qsurf) + F
    Qin2 = s_out * (Qr + Qsurf) / (s_in - s_out)                         # Knudsen (30) with dV = dS = 0
    bulk2 = dict(bulk, Q_in=Qin2, Q_out=-(Qin2 + Qr + Qsurf))
    r2 = lorenz_mixing(bulk2, Qr, Qsurf, 0.0, 0.0, 0.0, F)
    for k in ("Me", "Mc", "Mp"):
        assert r2[k] == pytest.approx(s_in * s_out * (Qr + Qsurf) + F)
    assert r2["s_surf"] == pytest.approx(np.sqrt(F / 1.0))              # eq. (36): (−F/Qsurf)^½


def test_preflight_tef_arguments():
    """tef_face needs salinity, a valid u-face column; tef_region needs the variance budget; kh > 0 warns."""
    from core.model.contract import ConfigurationError
    with pytest.raises(ConfigurationError, match="tef_face needs salinity=True"):
        _model("sigma_rk3", salinity=False, s_coef=0.0, variance_budget=False, tef_region=None,
               open_boundary_state=None, open_boundaries=(), open_boundary_mode=None, sponge_cells=0)
    with pytest.raises(ConfigurationError, match="u-face column"):
        _model("sigma_rk3", tef_face=NX + 1)
    with pytest.raises(ConfigurationError, match="variance_budget=True"):
        _model("sigma_rk3", variance_budget=False)
    with pytest.raises(ConfigurationError, match="tef_sign"):
        _model("sigma_rk3", tef_sign=2)
    with pytest.warns(Warning, match="diffusive salt flux"):
        _model("sigma_rk3", kh=1.0)


def test_tef_without_budget_advects_S2():
    """tef_face alone (no variance budget) puts S² in the stack, so the transect's s² flux exists; the solution is
    the same as without TEF."""
    from core.model.driver import _stack_layout
    m = _model("sigma_rk3", variance_budget=False, tef_region=None)
    assert "S2" in _stack_layout(m)
    st = m.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), 10.0), S=_S0())
    for _ in range(3):
        st = m.step(st, fw=_fw())
    b = m.tef.bulk()
    assert b["Qs2_in"] > 0 and np.isfinite(b["s2_in"])
