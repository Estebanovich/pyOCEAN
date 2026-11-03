"""
Horizontal viscosity next to open boundaries (class-(a) fix of 2026-11-03).

The normal velocity on an open-boundary face is a real boundary value (Flather/Orlanski/clamped), so the
horizontal viscous operators must use it, as MOM6 does: MOM6 keeps mask2dCu = mask2dCv = 1 on OBC faces
(MOM_grid_initialize.F90:1270-1288), its tension loop reads u, v on every face with no OBC branch
(MOM_hor_visc.F90:708-715), the OBC face thickness is the boundary cell's (:857-880), and only the corner shear
strain depends on OBC_STRAIN_* (:790-842; ``obc_tangential='none'`` = OBC_STRAIN_ZERO, D_S = 0 at the open-edge
corners, unchanged).  Before the fix, with ``obc_tangential='none'``, the Smagorinsky tension treated the open
face as a wall (u = 0) and the constant-A_h Laplacian treated it as a free-slip land neighbour.

Checks: a uniform through-flow has no viscous tendency anywhere; a linear shear has the interior tendency next
to the open edges; x- and y-edges; σ and ALE (z*); open-face tendency stays 0.  Index order (k, j, i).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.dynamics import sigma_baroclinic as sb
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid

NX, NY, NZ, DX = 12, 8, 3, 2000.0


def _model(coord: str, edges: tuple[str, ...], smagorinsky: float, ah: float) -> SigmaBaroclinic:
    hg = create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DX, f0=0.0)
    cfg = dict(dt=120.0, n_layers=NZ, rho0=1000.0, t_coef=0.2, z0=None, closure="constant", nu_const=1e-4,
               smagorinsky=smagorinsky, ah=ah, open_boundaries=edges, g=9.81)
    if coord == "zstar":
        cfg.update(vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(NZ, 10.0)),
                   pgf="fv", momentum="vector_invariant")
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        return SigmaBaroclinic(hg, np.full((NY, NX), 30.0), **cfg)


def _hvisc(m: SigmaBaroclinic, u: np.ndarray, v: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """The 'hviscosity' term (F_u, F_v) [m s⁻²] of the state with layer velocities u, v."""
    sb.set_threads(1)
    st = m.initial_state(np.zeros((NY, NX)), np.full((NZ, NY, NX), 10.0))
    st.u[:] = u
    st.v[:] = v
    terms, _, _ = m._momentum_terms(st)
    (fu, fv), = [(t[1], t[2]) for t in terms if t[0] == "hviscosity"]
    return fu, fv


CASES = [("sigma", 0.15, 0.0), ("zstar", 0.15, 0.0), ("sigma", 0.0, 50.0), ("zstar", 0.0, 50.0)]


@pytest.mark.parametrize("coord,C,ah", CASES)
def test_uniform_through_flow_has_no_viscous_tendency(coord, C, ah):
    m = _model(coord, ("west", "east"), C, ah)
    fu, fv = _hvisc(m, 0.3, 0.0)
    assert np.abs(fu).max() < 1e-18 and np.abs(fv).max() < 1e-18
    m = _model(coord, ("south", "north"), C, ah)
    fu, fv = _hvisc(m, 0.0, -0.2)
    assert np.abs(fu).max() < 1e-18 and np.abs(fv).max() < 1e-18


@pytest.mark.parametrize("coord,C,ah", CASES)
def test_linear_strain_along_the_flow_matches_the_interior(coord, C, ah):
    """u = u0 + b x: constant tension (Smagorinsky ν uniform), ∇²u = 0, so F_u = 0 everywhere, edges included."""
    m = _model(coord, ("west", "east"), C, ah)
    x = np.arange(NX + 1) * DX
    fu, fv = _hvisc(m, (0.3 + 1e-5 * x)[None, None, :], 0.0)
    scale = (C * DX * DX * 1e-5 + ah) * 1e-5 / DX                 # size of a wrong edge tendency [m s⁻²]
    assert np.abs(fu).max() < 1e-9 * scale and np.abs(fv).max() < 1e-9 * scale
    m = _model(coord, ("south", "north"), C, ah)
    y = np.arange(NY + 1) * DX
    fu, fv = _hvisc(m, 0.0, (0.3 + 1e-5 * y)[None, :, None])
    assert np.abs(fu).max() < 1e-9 * scale and np.abs(fv).max() < 1e-9 * scale


@pytest.mark.parametrize("coord,C,ah", CASES)
def test_cross_shear_next_to_the_open_edge_equals_the_interior(coord, C, ah):
    """u = u0 + b y between walls (south, north), open west and east: the tendency at the faces next to the open
    edges equals the one at interior faces of the same row (only the walls distinguish rows)."""
    m = _model(coord, ("west", "east"), C, ah)
    yc = (np.arange(NY) + 0.5) * DX
    fu, _ = _hvisc(m, (0.3 + 1e-5 * yc)[None, :, None], 0.0)
    mid = fu[:, :, NX // 2]
    assert np.abs(mid).max() > 0.0                                     # rows next to the walls feel them
    np.testing.assert_allclose(fu[:, :, 1], mid, rtol=1e-12, atol=1e-24)
    np.testing.assert_allclose(fu[:, :, NX - 1], mid, rtol=1e-12, atol=1e-24)
    assert np.all(fu[:, :, 0] == 0.0) and np.all(fu[:, :, NX] == 0.0)  # no tendency on the open faces


def test_closed_box_is_unchanged_by_the_open_face_rule():
    """No open edge: the operator sees the same masks as before (bit for bit is checked by the N1 goldens)."""
    m = _model("zstar", (), 0.15, 0.0)
    fu, _ = _hvisc(m, 0.3, 0.0)
    assert np.abs(fu[:, :, 1]).max() > 0.0                              # walls: (u₁ − 0)/Δx tension is real there
