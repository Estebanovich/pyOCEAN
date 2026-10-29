"""
Law-of-the-wall bed drag with the full bed speed (N3 task 9b, class-(a) fix found by the FORCED_EKBBL column of
task 8).

The quadratic bed stress of a C-grid model is τ_x/ρ₀ = C_d |U| u at u-faces and τ_y/ρ₀ = C_d |U| v at v-faces with
|U| = √(u² + v̄²) (resp. √(v² + ū²)), the other component averaged from its four surrounding faces: CROCO
get_vbc.F:517-528, MOM6 MOM_set_viscosity.F90:684-688 (``set_v_at_u``, :1887-1929).  pyOCEAN used each face's own
component, C_d |u| u, which halves the stress on u when the bed flow is mostly along v.

* ``test_loglaw_drag_uses_full_speed``: a doubly periodic column with u = v = U: the bed rate of the implicit
  solve is r = C_d √(u² + v²) to 1e-14 (σ and ALE z*), and the closure's u*² is C_d (u² + v²) at the centres;
* ``test_loglaw_drag_unchanged_when_v_zero``: with v ≡ 0 the run is bit for bit the old one (hashes recorded at
  7b34f33, before the fix);
* ``test_cross_velocity_at_faces``: the 4-point average wraps on periodic axes and is 0 on closed edges.

Index order (k, j, i), k = 0 at the bed; u-faces (ny, nx+1), v-faces (ny+1, nx).
"""
from __future__ import annotations

import hashlib

import numpy as np
import pytest

import core.dynamics.sigma_wd as sigma_wd
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.driver import StepContext
from physics.bottom_friction import cross_velocity_at_faces, first_layer_cd

Z0, HBBL, NZ, H = 1e-3, 1.0, 5, 10.0


def _column(ale: bool, f0: float = 0.0, **kw) -> SigmaBaroclinic:
    hg = create_cartesian_grid(nx=2, ny=2, dx=1000.0, dy=1000.0, f0=f0)
    base = dict(rho0=1000.0, t_coef=0.2, z0=Z0, hbbl=HBBL, closure="constant", nu_const=5e-2, theta=0.6,
                d_min=0.05, periodic=("x", "y"))
    if ale:
        base.update(vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(NZ, H / NZ)),
                    pgf="fv")
    base.update(kw)
    return SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=NZ, **base)


@pytest.mark.parametrize("ale", [False, True], ids=["sigma", "zstar"])
def test_loglaw_drag_uses_full_speed(ale, monkeypatch):
    U = 0.3
    m = _column(ale)
    st = m.initial_state(np.zeros((2, 2)), np.full((NZ, 2, 2), 15.0))
    st.u[:] = U
    st.v[:] = U
    rates = []
    orig = sigma_wd.vertical_implicit

    def spy(u, dz, nu, r_bed, dt, tau_top=None):
        rates.append(np.array(r_bed))
        return orig(u, dz, nu, r_bed, dt, tau_top=tau_top)
    monkeypatch.setattr(sigma_wd, "vertical_implicit", spy)
    ustar2 = []
    drv = m._driver
    name = "_ale_implicit_and_free_surface" if ale else "_sigma_implicit_and_free_surface"
    phase = getattr(drv, name)

    def wrapped(ctx: StepContext) -> None:
        phase(ctx)
        ustar2.append((np.array(ctx.ustar2_bot), np.array(ctx.u_new), np.array(ctx.v_new)))
    monkeypatch.setattr(drv, name, wrapped)
    m.step(st, None, None)
    assert len(rates) == 2                                            # u- then v-faces
    cd = first_layer_cd(H / NZ, Z0)                                   # 2 m bed layer > hbbl: the layer alone
    want = cd * np.sqrt(U**2 + U**2)
    for r in rates:
        r0 = r[0] if r.ndim == 2 else r                              # ALE: per-layer rates, w_0 = 1
        assert np.max(np.abs(r0 / want - 1.0)) < 1e-14
        if r.ndim == 2:
            assert np.all(r[1:] == 0.0)
    # τ_x/ρ₀ = C_d √(u² + v²) u at the new velocities, averaged to the centres: u*² = C_d (u² + v²) here
    us2, u1, v1 = ustar2[0]
    u0, v0 = u1[0, 0, 0], v1[0, 0, 0]
    assert np.allclose(u1[0], u0, rtol=0, atol=1e-15) and np.allclose(v1[0], v0, rtol=0, atol=1e-15)
    assert np.max(np.abs(us2 / (cd * (u0**2 + v0**2)) - 1.0)) < 1e-14


def _hash(*arrays: np.ndarray) -> str:
    hsh = hashlib.sha256()
    for a in arrays:
        hsh.update(np.ascontiguousarray(a, dtype=np.float64).tobytes())
    return hsh.hexdigest()


# sha256 of (u, v, eta) after 30 steps, recorded at 7b34f33 (own-component log law) with v ≡ 0
V0_HASHES = {
    "sigma": "8e5494cd0f8309c1fbe726639645d2e9b63c269f6a16fc4da581dd79e728b5d2",
    "zstar": "5f9fa1e2d3bf09791aa1e844c80f2fa16f84836325ddb32de7df1e9c913f70a1",
}


@pytest.mark.parametrize("ale", [False, True], ids=["sigma", "zstar"])
def test_loglaw_drag_unchanged_when_v_zero(ale):
    m = _column(ale, closure="gls")                                   # f = 0: v stays exactly 0
    st = m.initial_state(np.zeros((2, 2)), np.full((NZ, 2, 2), 15.0))
    tx = np.full((2, 3), 0.2)
    for _ in range(30):
        st = m.step(st, tx, None)
    assert np.all(st.v == 0.0) and np.max(np.abs(st.u)) > 1e-2
    assert _hash(st.u, st.v, st.eta) == V0_HASHES["zstar" if ale else "sigma"]


def test_cross_velocity_at_faces():
    rng = np.random.default_rng(1)
    ny, nx = 3, 4
    ub = rng.standard_normal((ny, nx + 1))
    vb = rng.standard_normal((ny + 1, nx))
    v_at_u, u_at_v = cross_velocity_at_faces(ub, vb)
    # interior: the 4-point mean of CROCO get_vbc.F:517-518
    j, i = 1, 2
    assert v_at_u[j, i] == pytest.approx(0.25 * (vb[j, i] + vb[j + 1, i] + vb[j, i - 1] + vb[j + 1, i - 1]))
    assert u_at_v[j, i] == pytest.approx(0.25 * (ub[j, i] + ub[j, i + 1] + ub[j - 1, i] + ub[j - 1, i + 1]))
    assert np.all(v_at_u[:, [0, -1]] == 0.0) and np.all(u_at_v[[0, -1], :] == 0.0)    # closed edges
    # periodic: the wrap faces see cells n−1 and 0 (u-faces 0 and nx are the same face)
    ub[:, -1] = ub[:, 0]
    vb[-1, :] = vb[0, :]
    pv, pu = cross_velocity_at_faces(ub, vb, periodic_x=True, periodic_y=True)
    want = 0.25 * (vb[:-1, -1] + vb[1:, -1] + vb[:-1, 0] + vb[1:, 0])
    assert np.allclose(pv[:, 0], want) and np.array_equal(pv[:, 0], pv[:, -1])
    want = 0.25 * (ub[-1, :-1] + ub[-1, 1:] + ub[0, :-1] + ub[0, 1:])
    assert np.allclose(pu[0, :], want) and np.array_equal(pu[0, :], pu[-1, :])
