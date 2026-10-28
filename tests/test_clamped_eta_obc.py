"""
Prescribed-elevation ("clamped η", Dirichlet) open boundary of the free surface: ``WetDryFreeSurface(clamped_u=…)``
and ``SigmaBaroclinic(open_boundary_mode={edge: 'eta'})`` (N2a task 8, controller ruling R4: Lorenz et al. 2021
prescribe the M2 elevation at the open boundary, l. 843-845, as GETM does; CROCO OBC_COM_ZSPECIFIED,
OCEAN/zetabc.F:95-99).

1. Linear standing wave: a frictionless closed channel of depth H forced by η = a sin ωt at its mouth cell has the
   amplitude a cos(k(L − x))/cos(kL) along it, k = ω/√(gH), L the distance from the mouth cell to the wall.
2. Exact volume: Σ A (ηⁿ⁺¹ − ηⁿ) = Δt (Σ open-face inflow + Σ A fw), with the diagnosed edge flux, in σ and ALE.
Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from boundaries.obcs import OBData
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.dynamics.wetting_drying import WetDryFreeSurface
from core.grid.horizontal import create_cartesian_grid

G = 9.81


@pytest.mark.parametrize("solver", ["direct", "cg"])
def test_clamped_mouth_gives_the_standing_wave(solver):
    """Head/mouth amplitude ratio 1/cos(kL) (frictionless, linear), from a harmonic fit at ω over cycles 4–12."""
    nx, dx, H, a = 120, 500.0, 10.0, 0.01
    T = 12.4206012 * 3600.0
    om = 2 * np.pi / T
    nstep = 240                                                     # steps per period
    dt = T / nstep
    hg = create_cartesian_grid(nx=nx, ny=1, dx=dx, dy=dx, f0=0.0)
    fl = np.zeros((1, nx + 1), bool)
    fl[:, 0] = True
    fs = WetDryFreeSurface(hg, np.full((1, nx), H), dt=dt, theta=0.5, flather_u=fl, clamped_u=fl, solver=solver)
    eta, u, v = np.zeros((1, nx)), np.zeros((1, nx + 1)), np.zeros((2, nx))
    zu, zv = np.zeros((1, nx + 1)), np.zeros((2, nx))
    t, rec = 0.0, []
    for n in range(12 * nstep):
        t += dt
        eu = np.full((1, nx + 1), a * np.sin(om * t))
        eta, u, v = fs.step(eta, u, v, u.copy(), v.copy(), flather_eta=(eu, zv), flather_ubar=zu, flather_vbar=zv)
        assert eta[0, 0] == pytest.approx(a * np.sin(om * t), abs=1e-15)
        if n >= 4 * nstep:
            rec.append((t, eta[0, 0], eta[0, -1]))
    rec = np.array(rec)
    X = np.stack([np.sin(om * rec[:, 0]), np.cos(om * rec[:, 0])], axis=1)
    amp = [np.hypot(*np.linalg.lstsq(X, rec[:, c], rcond=None)[0]) for c in (1, 2)]
    k = om / np.sqrt(G * H)
    L = (nx - 0.5) * dx                                             # mouth-cell centre to the closed wall
    expected = np.cos(k * 0.5 * dx) / np.cos(k * L)                 # at the last cell centre, x = L − dx/2
    assert amp[0] == pytest.approx(a, rel=1e-3)
    assert amp[1] / amp[0] == pytest.approx(expected, rel=0.03), (amp[1] / amp[0], expected)
    assert expected > 1.5                                            # a real amplification, not a flat response


def _channel(config: str, **kw) -> SigmaBaroclinic:
    """Sloping channel 40 × 1 × 10: west 'eta' edge (tide), east 'clamped' river 5 m³ s⁻¹, S on."""
    nx, nz = 40, 10
    H = np.linspace(15.0, 5.0, nx)[None, :]

    def state(t):
        out = {}
        for edge, depth, S, q, eta in (("west", H[0, 0], 30.0, None, 0.5 * np.sin(2 * np.pi * t / 44714.0)),
                                       ("east", H[0, -1], 0.0, -5.0, 0.0)):
            zw = np.linspace(0.0, -depth, 5)[:, None]
            u = np.zeros((4, 1)) if q is None else np.full((4, 1), q / (depth * 200.0))
            out[edge] = OBData(side=edge, T=np.full((4, 1), 10.0), S=np.full((4, 1), S), u_normal=u, zw=zw,
                               eta=np.full(1, eta), transport=None if q is None else np.full(1, q))
        return out
    base = dict(rho0=1025.0, t_coef=0.0, salinity=True, s_coef=0.78, s_ref=30.0, closure="constant", nu_const=1e-4,
                kv=1e-4, open_boundaries=("west", "east"), open_boundary_mode={"west": "eta", "east": "clamped"},
                open_boundary_state=state, sponge_cells=3)
    if config == "sigma":
        base.update(z0=1e-3)
    else:
        base.update(vertical_coordinate="ale", regrid="sigma", pgf="fv", z0=None, bottom_drag="quadratic")
    base.update(kw)
    return SigmaBaroclinic(create_cartesian_grid(nx=nx, ny=1, dx=200.0, dy=200.0, f0=0.0), H, dt=120.0,
                           n_layers=nz, **base)


@pytest.mark.parametrize("config", ["sigma", "ale"])
def test_clamped_edge_conserves_volume_exactly(config):
    """Σ V¹ − Σ V⁰ = Δt (Q_west − Q_east + Σ A fw) every step to 1e-13 of the total volume; the west cell keeps the
    exterior η; the layer transports at the open faces add up to the free-surface flux."""
    warnings.simplefilter("ignore")
    m = _channel(config)
    nx = m.H.shape[1]
    st = m.initial_state(np.zeros((1, nx)), np.full((10, 1, nx), 10.0), S=np.full((10, 1, nx), 30.0))
    fw = np.zeros((1, nx))
    fw[:, 10:25] = -3e-6
    fw[:, 30:] = 2e-6
    fs = m.core.fs
    for _ in range(60):
        V0 = m.layer_volumes(st.eta, st.h).sum()
        st = m.step(st, fw=fw)
        V1 = m.layer_volumes(st.eta, st.h).sum()
        Qw, Qe = fs.last_open_Qu[0, 0], fs.last_open_Qu[0, -1]
        src = Qw - Qe + (fs.last_fw * fs.area).sum()
        assert abs(V1 - V0 - m.dt * src) <= 1e-13 * V1, (V1 - V0, m.dt * src)
        assert st.eta[0, 0] == pytest.approx(0.5 * np.sin(2 * np.pi * st.time / 44714.0), abs=1e-12)
        assert Qe == pytest.approx(-5.0, rel=1e-12)
        assert st.Qx[:, 0, 0].sum() == pytest.approx(Qw, rel=1e-12, abs=1e-12)
    assert np.abs(st.u).max() > 0.01                                 # the tide is moving water


def test_eta_mode_unused_is_bit_for_bit():
    """Without an 'eta' edge nothing changes: same Flather list (plus the flag), no clamped faces."""
    hg = create_cartesian_grid(nx=10, ny=1, dx=100.0, dy=100.0, f0=0.0)
    fl = np.zeros((1, 11), bool)
    fl[:, 0] = True
    fs = WetDryFreeSurface(hg, np.full((1, 10), 5.0), dt=10.0, flather_u=fl)
    assert fs._clamped == [] and len(fs._flather) == 1 and not fs._flather[0]["clamped"]
    with pytest.raises(ValueError, match="Flather"):
        WetDryFreeSurface(hg, np.full((1, 10), 5.0), dt=10.0, clamped_u=fl)
