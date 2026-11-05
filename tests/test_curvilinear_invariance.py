"""
Metric audit for orthogonal curvilinear grids (regional chain, plan
docs/superpowers/plans/2026-11-01-regional-chain.md Task 3; audit table
docs/design/regional_refs/metric_audit.md).

Two kinds of test:

1. **Invariance** (criterion 2, 1e-10 relative): ``SigmaBaroclinic`` on a stretched Cartesian grid
   (``create_cartesian_grid_stretched``) and on the same grid written as a MOM6 supergrid and read with
   ``grid_from_supergrid`` must give the same state.  The supergrid path adds ``dx_q``, ``dy_q``, ``area_q`` and
   builds the areas from supergrid sub-cells, so these runs check that the model consumes a supergrid grid as the
   grid it describes.
2. **Exactness on a non-separable orthogonal grid** (the conformal map z = w + c w², every metric varies in i and j,
   dy_u ≠ dy_t, dx_v ≠ dx_t, area_t ≠ dx_t dy_t, area_q ≠ dx_u dy_v): discrete states whose answer is known on any
   grid if, and only if, each operator uses the MOM6 metric (a divergence-free flow is steady without forces; a
   discrete geostrophic state η = f/(gH) · mean₄(ψ) with transports from ψ is steady on an f-plane; a uniform
   tracer stays uniform and tracer content is conserved).  A separable stretched grid cannot tell dy_u from dy_t,
   so these are the tests that fail when a metric is swapped (mutation evidence in the task report).

Index order (k, j, i), k = 0 at the bed; u (N, ny, nx+1), v (N, ny+1, nx), corners (ny+1, nx+1).
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest
import xarray as xr

from core.dynamics import vector_invariant as vi
from core.dynamics.hor_visc import smag_geometry
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid_stretched
from core.grid.supergrid import grid_from_supergrid, supergrid_from_arrays
from core.model.contract import PyOceanConfigWarning

F0 = 1e-4                      # f-plane Coriolis parameter [s⁻¹]
RTOL = 1e-10                   # invariance tolerance (criterion 2)


# ---------------------------------------------------------------- grids
def _stretched_pair(nx: int = 11, ny: int = 9, seed: int = 3) -> tuple[xr.Dataset, xr.Dataset]:
    """
    A stretched Cartesian grid (random dx(i) ∈ [1, 2] km, dy(j) ∈ [0.8, 1.6] km) and the same grid as a planar
    MOM6 supergrid read by ``grid_from_supergrid`` (half-segments dx/2, dy/2; f-plane f = F0 set on both).
    """
    rng = np.random.default_rng(seed)
    dxs = 1_000.0 * (1.0 + rng.random(nx))
    dys = 800.0 * (1.0 + rng.random(ny))
    ref = create_cartesian_grid_stretched(dxs, dys, f0=F0)
    xn = np.empty(2 * nx + 1)
    xn[0::2], xn[1::2] = ref["x_u"].values[0], ref["x_t"].values[0]
    yn = np.empty(2 * ny + 1)
    yn[0::2], yn[1::2] = ref["y_v"].values[:, 0], ref["y_t"].values[:, 0]
    X, Y = np.meshgrid(xn, yn)
    hx, hy = np.repeat(dxs / 2.0, 2), np.repeat(dys / 2.0, 2)
    dx = np.broadcast_to(hx, (2 * ny + 1, 2 * nx)).copy()
    dy = np.broadcast_to(hy[:, None], (2 * ny, 2 * nx + 1)).copy()
    area = hy[:, None] * hx[None, :]
    sg = grid_from_supergrid(supergrid_from_arrays(X, Y, dx, dy, area, np.zeros_like(X), units="m"))
    return ref, _f_plane(sg)


def _f_plane(g: xr.Dataset, f0: float = F0) -> xr.Dataset:
    """A planar supergrid grid has no latitude (f = NaN): put it on an f-plane."""
    g = g.copy(deep=True)
    g["f_q"] = (g["f_q"].dims, np.full(g["f_q"].shape, f0))
    g["f_t"] = (g["f_t"].dims, np.full(g["f_t"].shape, f0))
    return g


def conformal_grid(nx: int = 12, ny: int = 10, d: float = 1_000.0, c: complex = 0.35 * np.exp(0.25j * np.pi),
                   f0: float = F0) -> xr.Dataset:
    """
    Orthogonal, non-separable planar grid: the image of a uniform (ξ, η) supergrid of spacing d/2 under the
    conformal map z = w + (c/L) w², L = nx d, w = ξ + iη [m], c complex so that the scale factor |1 + 2 c w/L|
    varies in both i and j (Δx_t and Δy_t by ≈ 70 % over the domain; neighbouring face/cell metrics by ≥ 0.2 %).
    Half-segment lengths are the chords between supergrid nodes, sub-cell areas the shoelace areas of the supergrid
    quadrilaterals, so the metrics are those a grid generator would write.
    """
    L = nx * d
    xi = np.arange(2 * nx + 1) * 0.5 * d
    et = np.arange(2 * ny + 1) * 0.5 * d
    W = xi[None, :] + 1j * et[:, None]
    Z = W + (c / L) * W * W
    X, Y = Z.real, Z.imag
    dx = np.abs(Z[:, 1:] - Z[:, :-1])
    dy = np.abs(Z[1:, :] - Z[:-1, :])
    a, b, cc, dd = Z[:-1, :-1], Z[:-1, 1:], Z[1:, 1:], Z[1:, :-1]
    area = 0.5 * np.abs(np.imag(np.conj(a) * b + np.conj(b) * cc + np.conj(cc) * dd + np.conj(dd) * a))
    ang = np.degrees(np.angle(1.0 + 2.0 * (c / L) * W))
    g = grid_from_supergrid(supergrid_from_arrays(X, Y, dx, dy, area, ang, units="m"), max_skew=2e-2)
    return _f_plane(g, f0)


# ---------------------------------------------------------------- models
def _model(grid: xr.Dataset, H: np.ndarray, coord: str, nz: int = 3, dt: float = 60.0, **kw) -> SigmaBaroclinic:
    """σ (flux form) or ALE z*/hybrid (vector invariant, FV PGF) with no closure-driven physics unless asked."""
    base = dict(z0=None, closure="none", ah=0.0, theta=0.6, d_min=0.05, t_coef=0.2, rho0=1000.0)
    if coord != "sigma":
        hmax = float(np.nanmax(H))
        base.update(vertical_coordinate="ale", regrid=coord, pgf="fv",
                    regrid_params=dict(dz_top=np.full(nz, hmax / nz)))
    if kw.get("eos", "linear") != "linear":                     # a nonlinear EOS takes its own ρ₀ and no α
        del base["t_coef"], base["rho0"]
    base.update(kw)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", PyOceanConfigWarning)
        return SigmaBaroclinic(grid, H, dt=dt, n_layers=nz, **base)


def _run(m: SigmaBaroclinic, st, n: int):
    for _ in range(n):
        st = m.step(st)
    return st


def _assert_same(a, b, names=("eta", "u", "v", "T", "h", "S", "tr"), rtol: float = RTOL) -> None:
    for name in names:
        x, y = getattr(a, name, None), getattr(b, name, None)
        if x is None and y is None:
            continue
        x, y = np.nan_to_num(np.asarray(x)), np.nan_to_num(np.asarray(y))
        scale = max(float(np.max(np.abs(x))), 1e-300)
        err = float(np.max(np.abs(x - y))) / scale
        assert err <= rtol, f"{name}: relative difference {err:.3e} > {rtol:g}"


def _psi_random(ny: int, nx: int, seed: int = 5) -> np.ndarray:
    """Stream function at the corners (ny+1, nx+1) [m³ s⁻¹ per unit amplitude], zero on the boundary (walls)."""
    rng = np.random.default_rng(seed)
    psi = np.zeros((ny + 1, nx + 1))
    psi[1:-1, 1:-1] = rng.standard_normal((ny - 1, nx - 1))
    return psi


def _transports(psi: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Divergence-free face transports from a corner stream function: Q_x = −δ_y ψ, Q_y = δ_x ψ [m³ s⁻¹]."""
    return -(psi[1:, :] - psi[:-1, :]), psi[:, 1:] - psi[:, :-1]


# ================================================================ 1. invariance (stretched Cartesian ≡ supergrid)
def _bump(g: xr.Dataset) -> np.ndarray:
    x, y = g["x_t"].values, g["y_t"].values
    x0, y0 = 0.5 * x.max(), 0.4 * y.max()
    return 0.05 * np.exp(-((x - x0) ** 2 + (y - y0) ** 2) / (3_000.0 ** 2))


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
def test_free_surface_wave_invariant(coord: str) -> None:
    """A free-surface bump (f = 0, Smagorinsky on): same state from the Cartesian and the supergrid grid."""
    ref, sg = _stretched_pair()
    out = []
    for g in (ref, sg):
        g = g.copy(deep=True)
        for name in ("f_q", "f_t"):
            g[name] = (g[name].dims, np.zeros(g[name].shape))
        H = np.full(g["dx_t"].shape, 50.0)
        m = _model(g, H, coord, smagorinsky=0.1, ah=1.0)
        st = m.initial_state(_bump(g), np.full((3,) + H.shape, 10.0))
        out.append(_run(m, st, 12))
    _assert_same(*out)


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
def test_geostrophic_adjustment_invariant(coord: str) -> None:
    """Geostrophic adjustment of a bump on an f-plane: same state from the two grids."""
    ref, sg = _stretched_pair()
    out = []
    for g in (ref, sg):
        H = np.full(g["dx_t"].shape, 50.0)
        m = _model(g, H, coord)
        st = m.initial_state(_bump(g), np.full((3,) + H.shape, 10.0))
        out.append(_run(m, st, 12))
    _assert_same(*out)


def test_tracer_advection_invariant() -> None:
    """WENO5 + FCT passive tracer and T in a flow set off by a bump over a slope (z*): same state."""
    ref, sg = _stretched_pair()
    out = []
    for g in (ref, sg):
        H = 30.0 + 20.0 * g["x_t"].values / g["x_t"].values.max()
        m = _model(g, H, "zstar", nz=4, n_tracers=1, tracer_limiter="fct")
        x = g["x_t"].values / g["x_t"].values.max()
        T = np.broadcast_to(10.0 + 2.0 * (x > 0.5), (4,) + H.shape).copy()
        tr = np.broadcast_to((x < 0.4).astype(float), (1, 4) + H.shape).copy()
        st = m.initial_state(_bump(g), T, tr=tr)
        out.append(_run(m, st, 12))
    _assert_same(*out)


@pytest.mark.parametrize("coord", ["zstar", "hybrid"])
def test_stratified_rest_invariant(coord: str) -> None:
    """Stratified water over a slope with the Wright EOS (z* and hybrid, FV PGF): same state from the two grids."""
    ref, sg = _stretched_pair()
    nz = 6
    out = []
    for g in (ref, sg):
        H = 40.0 + 30.0 * g["x_t"].values / g["x_t"].values.max()
        kw = dict(eos="wright_full", salinity=True, rho_ref="auto")
        params = dict(dz_top=np.full(nz, 70.0 / nz))
        if coord == "hybrid":
            params["rho_target"] = np.linspace(1025.8, 1027.6, nz + 1)
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", PyOceanConfigWarning)
            m = SigmaBaroclinic(g, H, dt=60.0, n_layers=nz, z0=None, closure="constant", nu_const=1e-4, ah=0.0,
                                theta=0.6, d_min=0.05, vertical_coordinate="ale", regrid=coord, pgf="fv",
                                regrid_params=params, **kw)
        k = np.arange(nz)[:, None, None] / (nz - 1)
        T = (8.0 + 6.0 * k) * np.ones((nz,) + H.shape)
        S = (35.2 - 0.4 * k) * np.ones((nz,) + H.shape)
        st = m.initial_state(np.zeros(H.shape), T, S=S)
        out.append(_run(m, st, 12))
    _assert_same(*out)


# ================================================================ 2. exactness on a non-separable orthogonal grid
def test_conformal_grid_is_non_separable() -> None:
    """The test grid distinguishes every metric the operators could confuse."""
    g = conformal_grid()
    dy_u, dy_t, dx_v, dx_t = (g[k].values for k in ("dy_u", "dy_t", "dx_v", "dx_t"))
    assert np.min(np.abs(dy_u[:, 1:] / dy_t - 1.0)) > 1e-3       # u-face length ≠ the cell's own height
    assert np.min(np.abs(dy_u[:, :-1] / dy_t - 1.0)) > 1e-3
    assert np.min(np.abs(dx_v[1:] / dx_t - 1.0)) > 1e-3
    assert np.min(np.abs(dx_v[:-1] / dx_t - 1.0)) > 1e-3
    assert np.max(np.abs(g["area_t"].values / (dx_t * dy_t) - 1.0)) > 1e-4
    aq = vi.corner_area(g["dx_u"].values, g["dy_v"].values)
    assert np.max(np.abs(g["area_q"].values[1:-1, 1:-1] / aq[1:-1, 1:-1] - 1.0)) > 1e-4
    assert np.ptp(dx_t) / dx_t.mean() > 0.5 and np.ptp(dy_t) / dy_t.mean() > 0.5


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_face_metrics_used(coord: str, axis: str) -> None:
    """
    Perturb only the u-face length dy_u (or the v-face length dx_v) at one face: the transport through that face
    changes in the same proportion.  Code that took the face length from dy_t, dy_v (dx_t, dx_u) would not see it.
    One face carries a small velocity (f = 0, so nothing else moves it in one step).
    """
    base = conformal_grid(f0=0.0)
    ny, nx = base["dx_t"].shape
    H = np.full((ny, nx), 20.0)
    j0, i0, fac = ny // 2, nx // 2, 1.3
    res = []
    for perturb in (False, True):
        g = base.copy(deep=True)
        name = "dy_u" if axis == "x" else "dx_v"
        if perturb:
            a = g[name].values.copy()
            a[j0, i0] *= fac
            g[name] = (g[name].dims, a)
        m = _model(g, H, coord, dt=0.01)                      # g H Δt²/Δx² ~ 2e-8: no free-surface feedback
        st = m.initial_state(np.zeros((ny, nx)), np.full((3, ny, nx), 10.0))
        if axis == "x":
            st.u[:, j0, i0] = 1e-6
        else:
            st.v[:, j0, i0] = 1e-6
        st = m.step(st)
        Q = st.Qx if axis == "x" else st.Qy
        res.append(Q[:, j0, i0].sum())
    np.testing.assert_allclose(res[1] / res[0], fac, rtol=1e-7)


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
def test_divergence_free_flow_steady(coord: str) -> None:
    """
    f = 0, flat η, layer transports from a corner stream function (discretely divergence-free with the u-face length
    dy_u and v-face length dx_v): nothing forces the flow, so η stays flat.  A continuity that used another length
    (dy_t, dy_v, …) or another area would move η by O(δ) of the scale below.
    """
    g = conformal_grid(f0=0.0)
    ny, nx = g["dx_t"].shape
    Hd, amp, nsteps, dt = 20.0, 1e-6, 5, 60.0
    H = np.full((ny, nx), Hd)
    m = _model(g, H, coord, dt=dt)
    st = m.initial_state(np.zeros((ny, nx)), np.full((3, ny, nx), 10.0))
    Qx, Qy = _transports(amp * _psi_random(ny, nx))
    st.u[:] = (Qx / (Hd * g["dy_u"].values))[None]
    st.v[:] = (Qy / (Hd * g["dx_v"].values))[None]
    st = _run(m, st, nsteps)
    scale = float(np.max(np.abs(Qx))) * nsteps * dt / float(g["area_t"].values.min())   # η of an O(1) divergence
    assert float(np.max(np.abs(st.eta))) <= 1e-6 * scale


GEO_CASES = [("sigma", "linear"), ("zstar", "linear"), ("zstar", "wright_full"), ("hybrid", "wright_full")]


def _geo_model(g: xr.Dataset, H: np.ndarray, coord: str, eos: str, dt: float, grav: float):
    """Model and resting (T, S) for the geostrophic test: ρ = ρ₀ at the surface, so η alone drives the flow."""
    ny, nx = H.shape
    if eos == "linear":
        return _model(g, H, coord, dt=dt, g=grav), np.zeros((3, ny, nx)), None     # T = t_ref: ρ = ρ₀
    from physics.eos.wright_full import density
    T, S = np.full((3, ny, nx), 10.0), np.full((3, ny, nx), 35.0)
    kw = dict(eos="wright_full", salinity=True, rho0=float(density(10.0, 35.0, 0.0)))
    if coord == "hybrid":                     # targets bracket the water: the hybrid keeps its z* floor layers
        kw["regrid_params"] = dict(dz_top=np.full(3, float(H.max()) / 3), rho_target=np.linspace(1025.0, 1030.0, 4))
    return _model(g, H, coord, dt=dt, g=grav, **kw), T, S


@pytest.mark.parametrize("coord,eos", GEO_CASES)
def test_geostrophic_balance_steady(coord: str, eos: str) -> None:
    """
    Discrete geostrophic state on an f-plane over a flat bed, on the non-separable grid: transports from a corner
    stream function ψ and η = f/(g H) · ¼ Σ₄ ψ(corners).  With the MOM6 (Sadourny 1975 energy) Coriolis weights,
    CAu = ¼/Δx_u Σ f (v Δx_v), and the PGF over Δx_u, this state is steady on any orthogonal grid; an unweighted
    or wrongly weighted Coriolis, or a PGF over another length, leaves an O(δ f u) imbalance.  Reduced g (0.1) keeps
    η well above the round-off of h in ALE at a Rossby number of 1e-6 (the vector-invariant ζ terms).
    """
    g = conformal_grid()
    ny, nx = g["dx_t"].shape
    Hd, grav, nsteps, dt = 10.0, 0.1, 5, 60.0
    H = np.full((ny, nx), Hd)
    m, T, S = _geo_model(g, H, coord, eos, dt, grav)
    psi = _psi_random(ny, nx)
    psi *= 1e-6 * F0 * 1_000.0 * 1_000.0 * Hd / np.max(np.abs(psi))          # u ~ Ro f Δx, Ro = 1e-6
    eta = F0 / (grav * Hd) * 0.25 * (psi[:-1, :-1] + psi[:-1, 1:] + psi[1:, :-1] + psi[1:, 1:])
    st = m.initial_state(eta, T, S=S)                          # surface ρ = ρ₀: no baroclinic force at rest
    Qx, Qy = _transports(psi)
    u0 = Qx / (Hd * g["dy_u"].values)
    v0 = Qy / (Hd * g["dx_v"].values)
    st.u[:] = u0[None]
    st.v[:] = v0[None]
    st = _run(m, st, nsteps)
    U = max(float(np.max(np.abs(u0))), float(np.max(np.abs(v0))))
    # A metric error δ (≥ 2e-3 on this grid) leaves an imbalance δ f U, i.e. ≥ 1e-5 U per step at f Δt = 6e-3;
    # the correct operators drift by the Rossby-number terms of the vector-invariant form (≈ 3e-9 U per step).
    for vel, ref in ((st.u, u0), (st.v, v0)):
        assert float(np.max(np.abs(vel - ref[None]))) <= 1e-7 * U * nsteps
    assert float(np.max(np.abs(st.eta - eta))) <= 1e-7 * float(np.max(np.abs(eta))) * nsteps


def test_tracer_uniform_and_conserved() -> None:
    """
    WENO5 + FCT on the non-separable grid, z* over a slope with a moving free surface: a uniform tracer stays
    uniform (tracer and continuity use the same face transports and cell areas) and the content Σ A_T h c is
    conserved to round-off.
    """
    g = conformal_grid()
    ny, nx = g["dx_t"].shape
    i = np.arange(nx)[None, :] / nx
    H = 20.0 + 10.0 * i * np.ones((ny, 1))
    m = _model(g, H, "zstar", nz=4, n_tracers=2, tracer_limiter="fct")
    x, y = g["x_t"].values, g["y_t"].values
    eta = 0.2 * np.exp(-((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / (3_000.0 ** 2))
    tr = np.empty((2, 4, ny, nx))
    tr[0] = 1.0
    tr[1] = (x > x.mean())[None]
    st = m.initial_state(eta, np.full((4, ny, nx), 10.0), tr=tr)
    A = g["area_t"].values
    c0 = float(np.sum(A[None] * st.h * st.tr[1]))
    st = _run(m, st, 20)
    assert float(np.max(np.abs(st.tr[0] - 1.0))) <= 1e-12
    c1 = float(np.sum(A[None] * st.h * st.tr[1]))
    assert abs(c1 - c0) <= 1e-12 * abs(c0)


def test_coriolis_does_no_work_on_curvilinear_grid() -> None:
    """
    The flux-form (σ) and vector-invariant Coriolis are energy neutral on any orthogonal grid:
    Σ A_u u C_u + Σ A_v v C_v = 0 with A_u = Δx_u Δy_u, A_v = Δx_v Δy_v (MOM6 areaCu, areaCv) for random u, v
    (flux form) and uniform h (vector invariant, BOUND_CORIOLIS off).
    """
    from core.components.momentum import Momentum
    g = conformal_grid()
    ny, nx = g["dx_t"].shape
    rng = np.random.default_rng(11)
    u = rng.standard_normal((2, ny, nx + 1))
    v = rng.standard_normal((2, ny + 1, nx))
    u[..., [0, -1]] = 0.0
    v[:, [0, -1], :] = 0.0
    m = _model(g, np.full((ny, nx), 20.0), "sigma")
    cu, cv = m._momentum.coriolis(u, v)
    Au, Av = g["area_u"].values, g["area_v"].values
    work = np.sum(Au * u * cu) + np.sum(Av * v * cv)
    scale = np.sum(Au * np.abs(u * cu)) + np.sum(Av * np.abs(v * cv))
    assert abs(work) <= 1e-13 * scale
    # vector invariant: q = f/h, Q = L u h
    h = 20.0
    q = np.full((2, ny + 1, nx + 1), F0 / h)
    Qx, Qy = g["dy_u"].values * u * h, g["dx_v"].values * v * h
    cu, cv = vi.coriolis_pv(q, q * h, Qx, Qy, u, v, g["dx_u"].values, g["dy_v"].values, bound=False)
    work = np.sum(Au * u * cu) + np.sum(Av * v * cv)
    scale = np.sum(Au * np.abs(u * cu)) + np.sum(Av * np.abs(v * cv))
    assert abs(work) <= 1e-13 * scale
    assert Momentum is not None


def test_supergrid_corner_metrics_used() -> None:
    """With a supergrid the vorticity and Smagorinsky operators take dx_q, dy_q, area_q (MOM6 dxBu, dyBu, areaBu)
    from the grid instead of rebuilding them from face metrics."""
    g = conformal_grid()
    m = _model(g, np.full(g["dx_t"].shape, 20.0), "zstar")
    np.testing.assert_array_equal(m._vi["area_q"], g["area_q"].values)
    sg = smag_geometry(g)
    for k in ("dx_q", "dy_q", "area_q"):
        np.testing.assert_array_equal(sg[k], g[k].values)


@pytest.mark.parametrize("coord,eos", [("sigma", "linear"), ("zstar", "linear"), ("zstar", "wright_full")])
@pytest.mark.parametrize("axis", ["x", "y"])
def test_pgf_metric_used(coord: str, eos: str, axis: str) -> None:
    """
    The baroclinic pressure-gradient force at a u-face is a difference over Δx_u (MOM6 IdxCu,
    MOM_PressureForce_FV.F90:869) and at a v-face over Δy_v (IdyCv, :872): perturbing only that metric at one face
    divides the force there by the same factor and leaves every other face unchanged.  Terms at the start of the
    first step (``keep_terms``), so nothing has moved yet.
    """
    base = conformal_grid()
    ny, nx = base["dx_t"].shape
    H = 20.0 + 10.0 * np.arange(nx)[None, :] / nx * np.ones((ny, 1))
    j0, i0, fac = ny // 2, nx // 2, 1.25
    name = "dx_u" if axis == "x" else "dy_v"
    k = np.arange(4)[:, None, None] / 3.0
    xx, yy = (base["x_t"].values, base["y_t"].values)
    T = 8.0 + 4.0 * k + 1.5 * np.sin(xx / 3_000.0)[None] + 1.0 * np.cos(yy / 2_500.0)[None]
    res = []
    for perturb in (False, True):
        g = base.copy(deep=True)
        if perturb:
            a = g[name].values.copy()
            a[j0, i0] *= fac
            g[name] = (g[name].dims, a)
        if eos == "linear":
            m = _model(g, H, coord, nz=4)
            st = m.initial_state(np.zeros((ny, nx)), T)
        else:
            m = _model(g, H, coord, nz=4, eos="wright_full", salinity=True)
            st = m.initial_state(np.zeros((ny, nx)), T, S=35.0 - 0.1 * T)
        m._driver.keep_terms = True
        m.step(st)
        res.append(m._driver.last_terms["pgf"][0 if axis == "x" else 1])
    r0, r1 = res
    assert np.max(np.abs(r0[:, j0, i0])) > 0.0
    np.testing.assert_allclose(r1[:, j0, i0] * fac, r0[:, j0, i0], rtol=1e-12)
    mask = np.ones(r0.shape[1:], bool)
    mask[j0, i0] = False
    np.testing.assert_array_equal(r1[:, mask], r0[:, mask])


# ================================================================ review fixes (I1, I2, M3, M4)
def _energy(m: SigmaBaroclinic, g: xr.Dataset, st, h_layer: float) -> float:
    """E = ½ g Σ A_T η² + ½ Σ_k h_k (Σ A_u u² + Σ A_v v²) [m⁵ s⁻²], A_u = Δx_u Δy_u, A_v = Δx_v Δy_v (MOM6 areaCu,
    areaCv): the energy of the linear free-surface system on a flat bed with equal layers."""
    At, Au, Av = (g[k].values for k in ("area_t", "area_u", "area_v"))
    kinetic = float(np.sum(Au * st.u ** 2) + np.sum(Av * st.v ** 2))
    return 0.5 * m.g * float(np.sum(At * st.eta ** 2)) + 0.5 * h_layer * kinetic


@pytest.mark.parametrize("coord", ["sigma", "zstar"])
def test_free_surface_wave_energy_on_curvilinear_grid(coord: str) -> None:
    """
    Free-surface gravity wave (f = 0, flat bed, no closure, A_h = 0) on the non-separable grid with θ = ½: the
    linear system is ∂u/∂t = −g δ_x η/Δx_u, ∂η/∂t = −(δ_x(H Δy_u u) + δ_y(H Δx_v v))/A_T, whose gradient and
    divergence are adjoint in the energy E (``_energy``) only with the MOM6 metrics, and Crank–Nicolson keeps E
    exactly.  The remaining drift is the O(η/H) nonlinearity of the free surface (2e-8 at η/H = 5e-8, 40 steps); a
    swapped face length or centre distance changes E at O(δ) ≥ 1e-4.
    """
    g = conformal_grid(f0=0.0)
    ny, nx = g["dx_t"].shape
    Hd = 20.0
    H = np.full((ny, nx), Hd)
    m = _model(g, H, coord, dt=30.0, theta=0.5)
    x, y = g["x_t"].values, g["y_t"].values
    eta = 1e-6 * np.exp(-((x - x.mean()) ** 2 + (y - y.mean()) ** 2) / (2_500.0 ** 2))
    st = m.initial_state(eta, np.zeros((3, ny, nx)))
    e0 = _energy(m, g, st, Hd / 3)
    worst = 0.0
    for _ in range(40):
        st = m.step(st)
        worst = max(worst, abs(_energy(m, g, st, Hd / 3) / e0 - 1.0))
    assert float(np.max(np.abs(st.u))) > 0.0
    assert worst <= 1e-6


@pytest.mark.parametrize("coord", ["zstar", "hybrid"])
def test_stratified_rest_on_curvilinear_grid(coord: str) -> None:
    """
    Stratified rest with the Wright EOS on the non-separable grid (flat bed, horizontally uniform T and S, z* and
    hybrid, FV PGF): the velocities stay at round-off.  The force is zero for any metric, so this checks only
    that the curvilinear grid introduces nothing (no swap can fail it; the swap-sensitive stratified checks are
    ``test_geostrophic_balance_steady[*-wright_full]`` and ``test_pgf_metric_used[*-wright_full]``).
    """
    g = conformal_grid()
    ny, nx = g["dx_t"].shape
    nz = 6
    H = np.full((ny, nx), 60.0)
    params = dict(dz_top=np.full(nz, 60.0 / nz))
    if coord == "hybrid":
        params["rho_target"] = np.linspace(1025.8, 1027.6, nz + 1)
    m = _model(g, H, coord, nz=nz, eos="wright_full", salinity=True, rho_ref="auto", closure="constant",
               nu_const=1e-4, regrid_params=params)
    k = np.arange(nz)[:, None, None] / (nz - 1)
    T = (8.0 + 6.0 * k) * np.ones((nz, ny, nx))
    S = (35.2 - 0.4 * k) * np.ones((nz, ny, nx))
    st = _run(m, m.initial_state(np.zeros((ny, nx)), T, S=S), 20)
    assert float(np.max(np.abs(st.u))) <= 1e-12 and float(np.max(np.abs(st.v))) <= 1e-12
    assert float(np.max(np.abs(st.eta))) <= 1e-12


def _smag_operator(g: xr.Dataset, nu: float = 50.0, h: float = 10.0):
    """The Smagorinsky stress divergence with constant ν (C = 1e-9 below A_h = ν, MOM6 control) and uniform h,
    as a linear map (u, v) ↦ (F_u, F_v) on one layer, all faces wet, walls at the domain edge."""
    from core.dynamics.hor_visc import smagorinsky
    ny, nx = g["dx_t"].shape
    geom = smag_geometry(g)
    wet_u = np.ones((ny, nx + 1), bool)
    wet_u[:, [0, -1]] = False
    wet_v = np.ones((ny + 1, nx), bool)
    wet_v[[0, -1], :] = False
    mask_q = np.zeros((ny + 1, nx + 1), bool)
    mask_q[1:-1, 1:-1] = True
    ou, ov = np.zeros((ny, nx + 1), bool), np.zeros((ny + 1, nx), bool)
    ht, hu, hv = np.full((1, ny, nx), h), np.full((1, ny, nx + 1), h), np.full((1, ny + 1, nx), h)

    def op(u, v):
        fu, fv, _, _, n_lim = smagorinsky(u[None], v[None], ht, hu, hv, wet_u, wet_v, mask_q, ou, ov, geom,
                                          1e-9, nu, 1.0)
        assert n_lim == 0
        return fu[0], fv[0]
    return op, wet_u, wet_v


def test_smagorinsky_self_adjoint_and_dissipative_on_curvilinear_grid() -> None:
    """
    MOM6's stress-tensor form (MOM_hor_visc.F90:707-722, 1757-1781) with constant ν and uniform h is the
    symmetric negative semi-definite operator −Σ ν h (D_T² Δx_T Δy_T + D_S² Δx_q Δy_q) in the inner product
    ⟨a, b⟩ = Σ A_u h a_u b_u + Σ A_v h a_v b_v (A_u = Δx_u Δy_u): ⟨a, L b⟩ = ⟨L a, b⟩ to round-off and ⟨a, L a⟩ ≤ 0,
    on any orthogonal grid.  A metric in the strain that differs from the one in the stress divergence (Δy_T for
    Δy_u, Δx_q² for Δy_q², A_u ≠ Δx_u Δy_u) breaks the symmetry.
    """
    g = conformal_grid()
    ny, nx = g["dx_t"].shape
    op, wet_u, wet_v = _smag_operator(g)
    Au, Av = g["area_u"].values, g["area_v"].values
    rng = np.random.default_rng(17)

    def rand():
        return np.where(wet_u, rng.standard_normal((ny, nx + 1)), 0.0), \
            np.where(wet_v, rng.standard_normal((ny + 1, nx)), 0.0)

    def dot(a, b):
        return float(np.sum(Au * a[0] * b[0]) + np.sum(Av * a[1] * b[1]))
    a, b = rand(), rand()
    La, Lb = op(*a), op(*b)
    scale = np.sqrt(abs(dot(a, La) * dot(b, Lb)))
    assert abs(dot(a, Lb) - dot(La, b)) <= 1e-12 * scale
    for x, Lx in ((a, La), (b, Lb)):
        assert dot(x, Lx) < 0.0


@pytest.mark.parametrize("periodic", [False, True])
def test_coriolis_does_no_work_on_stretched_grid(periodic: bool) -> None:
    """The weighted σ Coriolis is energy neutral on a stretched Cartesian grid (where ``coriolis_metrics`` switches
    the weights on), also through the periodic path (wrap faces stored twice, counted once)."""
    rng = np.random.default_rng(23)
    nx, ny = 11, 9
    g = create_cartesian_grid_stretched(1_000.0 * (1.0 + rng.random(nx)), 800.0 * (1.0 + rng.random(ny)), f0=F0)
    kw = dict(periodic=("x",)) if periodic else {}
    m = _model(g, np.full((ny, nx), 20.0), "sigma", **kw)
    assert m._momentum._cor_metrics is not None
    u = rng.standard_normal((2, ny, nx + 1))
    v = rng.standard_normal((2, ny + 1, nx))
    v[:, [0, -1], :] = 0.0
    if periodic:
        u[..., -1] = u[..., 0]
    else:
        u[..., [0, -1]] = 0.0
    cu, cv = m._momentum.coriolis(u, v)
    hg = m.hg                                   # the model's grid (wrap-face metrics when periodic)
    Au, Av = hg["area_u"].values, hg["area_v"].values
    if periodic:
        np.testing.assert_array_equal(cu[..., -1], cu[..., 0])
        Au = Au.copy()
        Au[:, -1] = 0.0                          # face nx is face 0
    work = np.sum(Au * u * cu) + np.sum(Av * v * cv)
    scale = np.sum(Au * np.abs(u * cu)) + np.sum(Av * np.abs(v * cv))
    assert abs(work) <= 1e-13 * scale


def _kh_model(g: xr.Dataset, kh: float = 50.0):
    ny, nx = g["dx_t"].shape
    return _model(g, np.full((ny, nx), 20.0), "zstar", t_coef=0.0, kh=kh, dt=60.0)


def test_tracer_diffusion_face_coefficient_on_curvilinear_grid() -> None:
    """
    Horizontal tracer diffusion at rest: one cell holds T = 1, the rest 0.  After one explicit step the heat that
    entered the east neighbour is Δt K_h (Δy_u/Δx_u) h (MOM_tracer_hor_diff.F90:278); perturbing only Δy_u of that
    face scales it in proportion, and the content Σ A_T h T is exact.
    """
    base = conformal_grid(f0=0.0)
    ny, nx = base["dx_t"].shape
    j0, i0, fac = ny // 2, nx // 2, 1.3
    got = []
    for perturb in (False, True):
        g = base.copy(deep=True)
        if perturb:
            a = g["dy_u"].values.copy()
            a[j0, i0 + 1] *= fac
            g["dy_u"] = (g["dy_u"].dims, a)
        m = _kh_model(g)
        T = np.zeros((3, ny, nx))
        T[:, j0, i0] = 1.0
        st0 = m.initial_state(np.zeros((ny, nx)), T)
        st = m.step(st0)
        A = g["area_t"].values
        c0 = float(np.sum(A[None] * st0.h * st0.T))
        c1 = float(np.sum(A[None] * st.h * st.T))
        assert abs(c1 - c0) <= 1e-13 * c0
        got.append(float(np.sum(A[j0, i0 + 1] * st.h[:, j0, i0 + 1] * st.T[:, j0, i0 + 1])))
    np.testing.assert_allclose(got[1] / got[0], fac, rtol=1e-10)


def test_tracer_diffusion_variance_on_curvilinear_grid() -> None:
    """K_h on a random T at rest: content Σ A_T h T exact and variance Σ A_T h T² non-increasing every step."""
    g = conformal_grid(f0=0.0)
    ny, nx = g["dx_t"].shape
    m = _kh_model(g)
    T = np.random.default_rng(29).random((3, ny, nx))
    st = m.initial_state(np.zeros((ny, nx)), T)
    A = g["area_t"].values[None]
    c0 = float(np.sum(A * st.h * st.T))
    var = float(np.sum(A * st.h * st.T ** 2))
    for _ in range(10):
        st = m.step(st)
        v1 = float(np.sum(A * st.h * st.T ** 2))
        assert v1 <= var * (1.0 + 1e-14)
        var = v1
    assert abs(float(np.sum(A * st.h * st.T)) - c0) <= 1e-13 * c0
