#!/usr/bin/env python
"""
Reference arrays produced by the code **before** regional-chain Task 5 (commit 71e848c), so that the tests can
check that Task 5 changed nothing when its features are off (Task 5 review, M3 and M4):

* ``eight``: f, u of the eight HYCOM constituents at the mjd of ``tests/data/tide_otps_reference.npz``, their ω and
  astronomical arguments, and a 30-day prediction, from ``forcing/tides.py`` of the old commit;
* ``off``: a homogeneous z* channel with an open 'radiating' west edge stepped 5 times (η, u, v, T, h), and the 2-D
  ``WetDryFreeSurface`` with a Flather edge stepped 5 times (η, u), both built without any Task 5 argument.

Run in a worktree of the old commit, writing into this repository:

    git worktree add --detach /tmp/base 71e848c
    cp validation/tides/make_task5_base_reference.py /tmp/base/validation/tides/
    (cd /tmp/base && PYTHONPATH=/tmp/base python validation/tides/make_task5_base_reference.py --out <repo>/tests/data/task5_base_reference.npz)

2026-11-02: regenerated at 71e848c with the vector-invariant PV-flux transport fix (``git diff`` of
core/components/momentum.py, docs/design/theta_bc_inertial/root_cause.md) applied in the worktree, so the reference
is "the code before Task 5 with the fix"; only off3d η, u, v changed (≤ 4e-17).

``run_off_case`` and ``run_off_2d`` are imported by tests/test_boundary_tides_3d.py, so the test runs the same
configuration with the current code.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

CONSTITUENTS8 = ["m2", "s2", "k1", "o1", "n2", "p1", "k2", "q1"]


def _grid(nx: int = 12, ny: int = 4, dx: float = 2000.0):
    from core.grid.horizontal import create_cartesian_grid
    return create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=0.0)


def _eta0(nx: int = 12, ny: int = 4) -> np.ndarray:
    return 0.05 * np.cos(np.pi * (np.arange(nx) + 0.5) / nx)[None] * np.ones((ny, 1))


def run_off_case(**kw) -> dict[str, np.ndarray]:
    """z* channel 12 × 4 × 4, H 50 m, open 'radiating' west edge, 5 steps of 300 s; ``kw`` go to the model."""
    from core.dynamics.sigma_baroclinic import SigmaBaroclinic
    nx, ny, nz, dt = 12, 4, 4, 300.0
    H = np.full((ny, nx), 50.0)
    m = SigmaBaroclinic(_grid(nx, ny), H, dt=dt, n_layers=nz, rho0=1000.0, t_coef=0.2, z0=None, closure="constant",
                        nu_const=1e-4, ah=0.0, theta=0.6, d_min=0.05, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(nz, 50.0 / nz)), pgf="fv", open_boundaries=("west",), **kw)
    st = m.initial_state(_eta0(nx, ny), np.full((nz, ny, nx), 10.0))
    for _ in range(5):
        st = m.step(st)
    return {name: np.asarray(getattr(st, name)).copy() for name in ("eta", "u", "v", "T", "h")}


def run_off_2d(**kw) -> dict[str, np.ndarray]:
    """2-D ``WetDryFreeSurface`` on the same grid, Flather west edge to a resting exterior, 5 steps."""
    from core.dynamics.wetting_drying import WetDryFreeSurface
    nx, ny, dt = 12, 4, 300.0
    fl = np.zeros((ny, nx + 1), bool)
    fl[:, 0] = True
    fs = WetDryFreeSurface(_grid(nx, ny), np.full((ny, nx), 50.0), dt=dt, theta=0.6, flather_u=fl, **kw)
    e, u, v = _eta0(nx, ny), np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    for _ in range(5):
        e, u, v = fs.step(e, u, v, u, v, flather_eta=0.0 * u, flather_ubar=0.0 * u, flather_vbar=0.0 * v)
    return {"eta": e, "u": u}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--otps-ref", default=str(Path(__file__).resolve().parents[2] / "tests" / "data"
                                              / "tide_otps_reference.npz"))
    a = ap.parse_args()
    from forcing.tides import OMEGA, PHASE_1992, nodal_corrections, predict
    ref = np.load(a.otps_ref)
    f, u = nodal_corrections(ref["mjd"], CONSTITUENTS8)
    t = np.datetime64("1992-01-01T00:00", "ns") + (ref["t_days_1992"] * 86400e9).astype("timedelta64[ns]")
    eta = np.array([predict(t, hc, list(ref["constituents"])) for hc in ref["hc"]])
    out = dict(eight_names=np.array(CONSTITUENTS8), eight_f=f, eight_u=u,
               eight_omega=np.array([OMEGA[c] for c in CONSTITUENTS8]),
               eight_phase=np.array([PHASE_1992[c] for c in CONSTITUENTS8]), eight_eta=eta)
    out.update({f"off3d_{k}": v for k, v in run_off_case().items()})
    out.update({f"off2d_{k}": v for k, v in run_off_2d().items()})
    np.savez(a.out, **out)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
