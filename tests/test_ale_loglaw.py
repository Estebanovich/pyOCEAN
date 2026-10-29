"""
Law-of-the-wall bed drag in the ALE mode and the bed u*² of the closure (N3 task 4; spec
docs/superpowers/specs/2026-10-29-n3-turbulence-closures-design.md §3 "Condición de fondo").

* a z* column whose layers do not move (doubly periodic 2×2, flat bed, uniform wind) with bed layers thicker than
  ``hbbl`` is σ's first-layer law exactly: u after 100 steps equal to σ with the same thicknesses to 1e-12;
* a vanished (1e-6 m) bed layer gives a finite C_d ≤ C_d(hbbl), the stress on the layers above, no NaN; a z* run
  over a slope (shallow columns have massless bed layers) stays finite;
* the bed rule is continuous in the bed-layer thickness at h_b = hbbl;
* ``ctx.ustar2_bot`` (ny, nx) is the closure's bed u*² in σ (the GLS input, unchanged) and in ALE;
* ``bottom_drag='quadratic'`` stays bit for bit (N1 golden ``zstar_quadratic_drag``), z0 and the quadratic drag
  together are a configuration error.

Index order (k, j, i), k = 0 at the bed; u-faces (ny, nx+1), v-faces (ny+1, nx).
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from core.components.bottom_drag import LogLawBedDrag
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.driver import StepContext
from physics.bottom_friction import first_layer_cd
from tests.n1_cases import diff_fields, run_case

ROOT = Path(__file__).resolve().parents[1]
Z0, HBBL = 1e-3, 1.0


def _column(ale: bool, nz: int = 5, H: float = 10.0, **kw) -> SigmaBaroclinic:
    hg = create_cartesian_grid(nx=2, ny=2, dx=1000.0, dy=1000.0, f0=1e-4)
    base = dict(rho0=1000.0, t_coef=0.2, z0=Z0, hbbl=HBBL, closure="constant", nu_const=5e-2, theta=0.6,
                d_min=0.05, periodic=("x", "y"))
    if ale:
        base.update(vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, H / nz)),
                    pgf="fv")
    base.update(kw)
    return SigmaBaroclinic(hg, np.full((2, 2), H), dt=60.0, n_layers=nz, **base)


def _run_column(m: SigmaBaroclinic, nz: int, steps: int = 100):
    st = m.initial_state(np.zeros((2, 2)), np.full((nz, 2, 2), 15.0))
    tx, ty = np.full((2, 3), 0.2), np.full((3, 2), 0.1)              # uniform wind stress [N m⁻²]
    for _ in range(steps):
        st = m.step(st, tx, ty)
    return st


def test_ale_loglaw_matches_sigma_column():
    nz = 5                                                            # 2 m layers > hbbl = 1 m: σ's first-layer law
    sig = _run_column(_column(False, nz), nz)
    ale = _run_column(_column(True, nz), nz)
    assert np.max(np.abs(ale.h - 2.0)) < 1e-12                        # the z* layers did not move
    scale = np.max(np.abs(sig.u))
    assert scale > 1e-2 and sig.u[0].max() < sig.u[-1].max()          # driven, sheared by the bed drag
    assert np.max(np.abs(ale.u - sig.u)) <= 1e-12 * scale
    assert np.max(np.abs(ale.v - sig.v)) <= 1e-12 * np.max(np.abs(sig.v))


def test_ale_without_drag_differs():
    # the comparison above is not trivial: without z0 the bed is free-slip and u differs by several per cent
    nz = 5
    sig = _run_column(_column(False, nz), nz)
    free = _run_column(_column(True, nz, z0=None), nz)
    assert np.max(np.abs(free.u - sig.u)) > 0.05 * np.max(np.abs(sig.u))


def test_ale_loglaw_vanished_bed_layer_finite():
    drag = LogLawBedDrag(Z0, hbbl=5.0)
    h = np.full((6, 3), 2.0)
    h[0] = [1e-6, 0.0, 3.0]                                           # vanished, massless, thick-but-thin-of-hbbl
    db, w, cd = drag.bed_layer(h)
    cd_hbbl = first_layer_cd(5.0, Z0)
    assert np.all(np.isfinite(cd)) and np.all(cd <= cd_hbbl * (1 + 1e-15))
    assert np.allclose(db, 5.0) and np.allclose(w.sum(axis=0), 1.0)
    assert w[0, 0] == 0.0 and w[0, 1] == 0.0                          # vanished layers carry no stress
    assert w[1, 0] == pytest.approx(2.0 / 5.0) and w[0, 2] == pytest.approx(3.0 / 5.0)
    u = np.linspace(0.1, 0.6, 6)[:, None] * np.ones((1, 3))
    u[0, 0] = 50.0                                                    # vanished-layer velocity noise: no effect
    r, _ = drag.rates(u, u, h, h)
    assert np.all(np.isfinite(r)) and r[0, 0] == 0.0
    # a dry face (no water) gets w_0 = 1 and the finite cd_max
    db0, w0, cd0 = drag.bed_layer(np.zeros((4, 1)))
    assert db0[0] == 0.0 and w0[0, 0] == 1.0 and np.isfinite(cd0[0])


def test_loglaw_bed_rule_continuous_at_hbbl():
    drag = LogLawBedDrag(Z0, hbbl=4.0)
    eps = 1e-9
    h = np.full((5, 2), 3.0)
    h[0] = [4.0 - eps, 4.0 + eps]
    db, w, cd = drag.bed_layer(h)
    assert abs(cd[0] - cd[1]) < 1e-9 and abs(w[0, 0] - w[0, 1]) < 1e-8
    thick = np.full((5, 1), 3.0)
    thick[0] = 8.0                                                    # thicker than hbbl: the layer alone, C_d(h_b)
    db, w, cd = drag.bed_layer(thick)
    assert db[0] == 8.0 and w[0, 0] == 1.0 and np.all(w[1:, 0] == 0.0)
    assert cd[0] == first_layer_cd(8.0, Z0)


def test_ale_loglaw_zstar_slope_finite():
    # z* over a slope: the shallow columns have massless bed layers (the reason z0 was blocked before N3)
    nx, nz = 12, 8
    hg = create_cartesian_grid(nx=nx, ny=1, dx=1000.0, dy=1000.0, f0=1e-4)
    H = np.linspace(4.0, 40.0, nx)[None]
    m = SigmaBaroclinic(hg, H, dt=30.0, n_layers=nz, rho0=1000.0, t_coef=0.2, z0=Z0, hbbl=2.0, closure="constant",
                        nu_const=1e-3, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 5.0)), pgf="fv")
    got = _capture_ustar2(m)
    st = m.initial_state(np.zeros((1, nx)), np.full((nz, 1, nx), 15.0))
    assert np.any(st.h[0] <= 1e-3)                                    # vanished (z* floor) bed layers exist
    tx = np.full((1, nx + 1), 0.2)
    for _ in range(60):
        st = m.step(st, tx, None)
    assert np.all(np.isfinite(st.u)) and np.all(np.isfinite(st.eta))
    assert len(got) == 60 and np.all(np.isfinite(got[-1])) and got[-1].max() > 0


def _capture_ustar2(m: SigmaBaroclinic) -> list:
    """Record ``ctx.ustar2_bot`` after phase 4 of every step."""
    got = []
    drv = m._driver
    name = "_ale_implicit_and_free_surface" if getattr(m, "ale", False) else "_sigma_implicit_and_free_surface"
    orig = getattr(drv, name)

    def wrapped(ctx: StepContext) -> None:
        orig(ctx)
        got.append(np.array(ctx.ustar2_bot))
    setattr(drv, name, wrapped)
    return got


@pytest.mark.parametrize("ale", [False, True], ids=["sigma", "zstar"])
def test_ustar2_bot_in_context(ale):
    nz = 5
    m = _column(ale, nz)
    got = _capture_ustar2(m)
    st = _run_column(m, nz, steps=20)
    assert len(got) == 20 and got[-1].shape == (2, 2)
    # the log-law stress of the bed layer at the new velocities: C_d(h_0) |u_0| u_0 averaged to the centres
    cd = first_layer_cd(2.0, Z0)
    tu = cd * np.abs(st.u[0]) * st.u[0]
    tv = cd * np.abs(st.v[0]) * st.v[0]
    want = np.hypot(0.5 * (tu[:, :-1] + tu[:, 1:]), 0.5 * (tv[:-1] + tv[1:]))
    # (ALE: the boundary relaxation of phase 5 does not touch a periodic column, st.u is u of phase 4)
    assert np.allclose(got[-1], want, rtol=1e-10, atol=0.0) and want.min() > 0


def test_ustar2_bot_zero_without_drag():
    m = _column(True, 5, z0=None)
    got = _capture_ustar2(m)
    _run_column(m, 5, steps=3)
    assert all(np.all(g == 0.0) for g in got)


def test_sigma_gls_gets_ctx_ustar2():
    # σ GLS: the closure receives exactly ctx.ustar2_bot (one source)
    m = _column(False, 5, closure="gls")
    seen = []
    orig = m._closure.update

    def upd(*a, **kw):
        seen.append(kw["ustar2_bot"])
        return orig(*a, **kw)
    m._closure.update = upd
    got = _capture_ustar2(m)
    _run_column(m, 5, steps=3)
    assert len(seen) == 3 and all(s is g or np.array_equal(s, g) for s, g in zip(seen, got))


def test_quadratic_drag_unchanged():
    golden = json.loads((ROOT / "tests" / "data" / "n1_golden" / "hashes.json").read_text())
    ref = golden["zstar_quadratic_drag"]
    got = run_case("zstar_quadratic_drag")
    for step, h in ref.items():
        assert got[int(step)] == h, diff_fields(got[int(step)], h)


def test_z0_with_quadratic_drag_is_an_error():
    with pytest.raises(ValueError, match="two bed drags"):
        _column(True, 5, bottom_drag="quadratic")
