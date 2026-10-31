"""
Surface average of the bulk fluxes on the top layers only (N2b task 6, controller ruling R5): ``surface_average``
and the shared weights of ``SurfaceFluxes.forcing`` must be bit for bit with the full-column formula of Task 4,
kept here as the oracle, in z*, σ and hybrid columns, columns shallower than h_sfc, h_sfc = 0, land and dry
columns, and a hybrid column with a 1 mm top layer.  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.components import surface_fluxes as sfm
from core.components.surface_fluxes import _surface_weights, surface_average


def _oracle(h: np.ndarray, q: np.ndarray, h_sfc: float) -> np.ndarray:
    """The Task 4 formula (every layer weighted), verbatim."""
    h = np.asarray(h, float)
    q = np.asarray(q, float)
    D = h.sum(axis=0)
    if h_sfc == 0:
        return np.where(D > 0, q[-1], np.nan)
    below = np.cumsum(h[::-1], axis=0)[::-1]
    w = np.clip(np.minimum(below, h_sfc) - (below - h), 0.0, None)
    wsum = w.sum(axis=0)
    with np.errstate(invalid="ignore", divide="ignore"):
        return np.where(wsum > 0, (w * q).sum(axis=0) / np.where(wsum > 0, wsum, 1.0), np.nan)


def _columns(kind: str, rng: np.random.Generator, ny: int = 7, nx: int = 9) -> np.ndarray:
    """Layer thicknesses (N, ny, nx), bed first, with land (h ≡ 0) and very shallow columns."""
    D = rng.uniform(0.2, 300.0, (ny, nx))
    D[0, :3] = 0.0                                                     # land / dry: h ≡ 0
    D[1, 0], D[1, 1], D[1, 2] = 0.3, 0.999, 1.0                        # shallower than / equal to h_sfc
    if kind == "zstar":                                                # MITgcm-like levels, partial bed layer
        dz = np.r_[np.linspace(0.6, 20.0, 30)][::-1]                   # bed first: thick at the bed
        N = dz.size
        top_down = np.cumsum(dz[::-1])
        h = np.zeros((N, ny, nx))
        for j in range(ny):
            for i in range(nx):
                above = 0.0
                for m, z in enumerate(top_down):
                    k = N - 1 - m
                    h[k, j, i] = max(0.0, min(z, D[j, i]) - above)
                    above = min(z, D[j, i])
        return h
    if kind == "sigma":
        frac = rng.dirichlet(np.ones(20))
        return frac[:, None, None] * D[None]
    # hybrid: a 1 mm top layer, vanished (1e-10 and 0) layers, thick interior
    N = 12
    h = rng.uniform(0.5, 1.0, (N, ny, nx))
    h[3:6] = rng.choice([0.0, 1e-10, 1e-3], size=(3, ny, nx))
    h = h / h.sum(axis=0) * D
    h[-1] = 1e-3                                                       # 1 mm top layer
    h[-2, 2, :] = 0.0004                                               # thin second layer in one row
    h[:, 0, :3] = 0.0
    return h


@pytest.mark.parametrize("kind", ["zstar", "sigma", "hybrid"])
@pytest.mark.parametrize("h_sfc", [0.0, 1e-4, 0.5, 1.0, 1.0006, 10.0, 1000.0])
def test_surface_average_bit_for_bit(kind: str, h_sfc: float) -> None:
    rng = np.random.default_rng(7)
    h = _columns(kind, rng)
    q = rng.normal(15.0, 3.0, h.shape)
    np.testing.assert_array_equal(surface_average(h, q, h_sfc), _oracle(h, q, h_sfc))


def test_cut_at_an_interface_depth() -> None:
    """h_sfc exactly at the bottom of the top two layers (the margin edge) in every column."""
    h = np.broadcast_to(np.array([5.0, 3.0, 0.7, 0.3])[:, None, None], (4, 3, 4)).copy()
    q = np.arange(h.size, dtype=float).reshape(h.shape) * 0.37
    for h_sfc in (0.3, 1.0, 1.0 + 1e-7, 4.0, 9.0):
        np.testing.assert_array_equal(surface_average(h, q, h_sfc), _oracle(h, q, h_sfc))


def test_cut_uses_only_the_top_layers() -> None:
    """In deep z* columns only the layers within h_sfc are visited (the point of the cut)."""
    h = np.full((50, 4, 5), 2.0)
    h[:, 0, 0] = 0.0                                                   # land
    k0, w, ok, _ = _surface_weights(h, 1.0)
    assert k0 == 49 and w.shape[0] == 1 and not ok[0, 0] and ok[1:].all()
    k0, *_ = _surface_weights(h, 3.0)
    assert k0 == 48


@pytest.mark.parametrize("kind", ["zstar", "sigma", "hybrid"])
@pytest.mark.parametrize("h_uv", [None, 0.0, 5.0])
def test_forcing_averager_bit_for_bit(kind: str, h_uv) -> None:
    """``SurfaceFluxes._averager`` (shared weights, velocities centred on the top layers) = the oracle."""
    rng = np.random.default_rng(3)
    h = _columns(kind, rng)
    wet = h.sum(axis=0) > 0
    h = np.where(wet[None], h, 0.0)
    N, ny, nx = h.shape
    T, S = rng.normal(15.0, 3.0, h.shape), rng.normal(35.0, 0.5, h.shape)
    u, v = rng.normal(0.0, 0.2, (N, ny, nx + 1)), rng.normal(0.0, 0.2, (N, ny + 1, nx))
    h_sfc = 1.0
    d_uv = h_sfc if h_uv is None else h_uv
    avg = sfm.SurfaceFluxes._averager(h, wet)
    np.testing.assert_array_equal(avg(h_sfc, T), _oracle(h, T, h_sfc))
    np.testing.assert_array_equal(avg(h_sfc, S), _oracle(h, S, h_sfc))
    k0 = avg.k0(d_uv)
    uc_full, vc_full = 0.5 * (u[..., :-1] + u[..., 1:]), 0.5 * (v[:, :-1, :] + v[:, 1:, :])
    uc, vc = 0.5 * (u[k0:, :, :-1] + u[k0:, :, 1:]), 0.5 * (v[k0:, :-1, :] + v[k0:, 1:, :])
    np.testing.assert_array_equal(avg(d_uv, uc, top=True), _oracle(h, uc_full, d_uv))
    np.testing.assert_array_equal(avg(d_uv, vc, top=True), _oracle(h, vc_full, d_uv))


@pytest.mark.parametrize("kind", ["sigma", "zstar", "hybrid", "ale_sigma"])
def test_model_forcing_bit_for_bit(kind: str, monkeypatch) -> None:
    """``SurfaceFluxes.forcing`` on the bulk-test basins with land and a dry bank (z*: land only), after a few steps: the cut
    against the full column (``_SFC_CUT_MIN`` = ∞ weights every layer, the Task 4 formula)."""
    from tests import test_surface_fluxes_model as tsm
    m, st = tsm._basin(kind, H=tsm._land_dry_H(dry=kind != "zstar"))     # z* cannot dry
    for _ in range(3):
        st = m.step(st)
    sf = m._surface_fluxes
    t = st.time + m.dt / 2
    fast = sf.forcing(st, t)
    last_fast = dict(sf.last)
    monkeypatch.setattr(sfm, "_SFC_CUT_MIN", np.inf)
    full = sf.forcing(st, t)
    for a, b in zip(fast[:3], full[:3]):
        np.testing.assert_array_equal(a, b)
    for key in ("Q_ns", "Q_sw"):
        np.testing.assert_array_equal(fast[3][key], full[3][key])
    for key, val in sf.last.items():
        np.testing.assert_array_equal(last_fast[key], val)
