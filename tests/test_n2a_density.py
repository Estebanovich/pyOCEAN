"""
N2a task 3: the density ρ(T, S) in every consumer — PGF (Jacobian, two-term, finite volume via ``fv_pgf_rho``),
the hybrid regrid rule, ``rho_ref="auto"`` and the hybrid start-up; no division by α = ``t_coef`` on any salinity
path (spec docs/superpowers/specs/2026-10-28-n2a-salinity-freshwater-design.md).

EOS: ρ = ρ₀ − α (T − T₀) + β (S − S₀) (Vallis 2006 eq. 1.156).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings

import numpy as np
import pytest

from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import PyOceanConfigWarning
from core.vertical import pgf as vpgf
from tests import n1_cases as nc

RHO0, BETA, S0 = 1000.0, 0.78, 30.0


def _salt_kw(**kw) -> dict:
    """Salt-only density: α = 0, β = 0.78, S₀ = 30."""
    base = dict(rho0=RHO0, t_coef=0.0, salinity=True, s_coef=BETA, s_ref=S0)
    base.update(kw)
    return base


def _S_of_rho(rho: np.ndarray) -> np.ndarray:
    return S0 + (rho - RHO0) / BETA


# ---------------------------------------------------------------- PGF with S only
@pytest.mark.parametrize("scheme", ["jacobian", "fv"])
def test_salt_only_linear_density_pgf_exact_flat_bed(scheme):
    # ρ′ = a x + b z through S alone (α = 0): exact PGF_x = (g a/ρ₀) z at η = 0 (as test_sigma_baroclinic:176)
    hg = create_cartesian_grid(nx=10, ny=6, dx=1000.0, dy=1000.0, f0=0.0)
    H = np.full((6, 10), 30.0)
    nz = 8
    zc = -30.0 + 30.0 * (np.arange(nz) + 0.5) / nz
    xc = (np.arange(10) + 0.5) * 1000.0
    a, b = 1e-4, -0.02
    rho = RHO0 + a * xc[None, None, :] + b * zc[:, None, None] + 0 * np.ones((nz, 6, 10))
    S = _S_of_rho(rho)
    T = np.full(S.shape, 10.0)
    m = SigmaBaroclinic(hg, H, dt=10.0, n_layers=nz, pgf=scheme, **_salt_kw())
    gx, _ = m.pressure_gradient(np.zeros((6, 10)), T, S=S)
    exact = 9.81 * a / RHO0 * zc[:, None, None]
    np.testing.assert_allclose(gx[:, :, 1:-1], np.broadcast_to(exact, gx[:, :, 1:-1].shape), rtol=1e-10, atol=1e-16)


@pytest.mark.parametrize("scheme", ["jacobian", "fv"])
def test_salt_only_stratification_seamount_rest(scheme):
    # S linear in z over the seamount of test_sigma_baroclinic:165 (α = 0): no force at rest
    nx, ny, nz, dx = 24, 20, 12, 8000.0
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=1e-4)
    x = (np.arange(nx) + 0.5) * dx - nx * dx / 2
    y = (np.arange(ny) + 0.5) * dx - ny * dx / 2
    h = 4500.0 - 4050.0 * np.exp(-(x[None, :] ** 2 + y[:, None] ** 2) / 25e3**2)
    m = SigmaBaroclinic(hg, h, dt=600.0, n_layers=nz, z0=None, closure="constant", nu_const=1e-4, pgf=scheme,
                        **_salt_kw())
    z = m.layer_heights(np.zeros((ny, nx)))
    S = _S_of_rho(RHO0 - 0.001 * z)                                   # dρ/dz = −1e-3 kg m⁻⁴
    gx, gy = m.pressure_gradient(np.zeros((ny, nx)), np.full(z.shape, 10.0), S=S)
    assert max(np.abs(gx).max(), np.abs(gy).max()) < 1e-13


@pytest.mark.parametrize("with_ref", [False, True])
def test_fv_pgf_rho_equals_fv_pgf_for_T_only(with_ref):
    # ρ = ρ₀ − α (T − T₀): the density path gives the T path to round-off (distinct arithmetic, not bit for bit)
    rng = np.random.default_rng(3)
    ny, nx, n = 5, 7, 6
    alpha, t0 = 0.2, 4.0
    H = 30.0 + 10.0 * rng.uniform(size=(ny, nx))
    eta = 0.1 * rng.standard_normal((ny, nx))
    f = rng.uniform(0.2, 1.0, (n, ny, nx))
    f[rng.uniform(size=f.shape) < 0.15] = 1e-5                          # some vanished layers
    h = f / f.sum(0) * (H + eta)[None]
    zc = -H[None] + np.cumsum(h, 0) - 0.5 * h
    T = 10.0 - 0.1 * zc + 0.3 * rng.standard_normal((n, ny, nx))
    ref = (lambda z: RHO0 + 0.004 * (-np.asarray(z))) if with_ref else None           # noqa: E731
    du, dv = np.full((ny, nx + 1), 1000.0), np.full((ny + 1, nx), 1000.0)
    gx, gy = vpgf.fv_pgf(h, H, eta, T, du, dv, RHO0, alpha, t0, ref, 9.81)
    rx, ry = vpgf.fv_pgf_rho(h, H, eta, RHO0 - alpha * (T - t0), du, dv, RHO0, ref, 9.81)
    scale = max(np.abs(gx).max(), np.abs(gy).max())
    np.testing.assert_allclose(rx, gx, rtol=1e-12, atol=1e-12 * scale)
    np.testing.assert_allclose(ry, gy, rtol=1e-12, atol=1e-12 * scale)


# ---------------------------------------------------------------- α = 0 with a reference profile and K_h
@pytest.mark.parametrize("coord", ["sigma", "ale"])
def test_zero_t_coef_with_reference_profile_and_kh(coord):
    kw = _salt_kw(rho_ref="auto", kh=5.0, kv=1e-5)
    with pytest.warns(PyOceanConfigWarning, match="rho_ref"):
        m = nc._sigma(**kw) if coord == "sigma" else nc._ale("zstar", **kw)
    warnings.simplefilter("ignore")
    k = np.arange(nc.NZ)[:, None, None]
    S = (31.0 - 1.0 * k / (nc.NZ - 1) + 0.05 * np.arange(nc.NX)[None, None, :]) * np.ones((nc.NZ, nc.NY, nc.NX))
    st = m.initial_state(nc._eta(), np.full(S.shape, 10.0), S=S)
    assert callable(m.rho_ref)
    wind = nc._wind(True)
    for i in range(5):
        st = m.step(st, *wind(i))
    for f in (st.eta, st.u, st.v, st.T, st.S) + (() if st.h is None else (st.h,)):
        assert np.isfinite(f).all()
    assert not np.array_equal(st.S, S)


def test_zero_t_coef_without_salinity_ale_runs():
    # T-only α = 0 used to give 0/0 in fv_pgf (T_ref = T₀ + (ρ₀ − ρ_ref)/α): the component takes the density path
    warnings.simplefilter("ignore")
    m = nc._ale("zstar", t_coef=0.0)
    st = m.initial_state(nc._eta(), nc._T())
    wind = nc._wind(True)
    for i in range(3):
        st = m.step(st, *wind(i))
    assert np.isfinite(st.eta).all() and np.isfinite(st.u).all()


def test_salinity_rho_ref_kh_warns_only_with_both():
    for kw in (dict(rho_ref="auto", kh=0.0), dict(rho_ref=None, kh=5.0)):
        with warnings.catch_warnings():
            warnings.simplefilter("error", PyOceanConfigWarning)
            nc._sigma(**_salt_kw(**kw))


def test_rho_ref_auto_uses_salinity():
    warnings.simplefilter("ignore")
    m = nc._sigma(**_salt_kw(rho_ref="auto"))
    z = m.layer_heights(np.zeros((nc.NY, nc.NX)))
    S = _S_of_rho(RHO0 - 0.01 * z)
    m.initial_state(np.zeros((nc.NY, nc.NX)), np.full(S.shape, 10.0), S=S)
    zz = np.linspace(-15.0, -5.0, 5)
    np.testing.assert_allclose(m.rho_ref(zz), RHO0 - 0.01 * zz, rtol=1e-12)


def test_rpe_uses_salinity():
    # the RPE diagnostic sorts ρ(T, S): with α = 0 only S sets it
    warnings.simplefilter("ignore")
    from diagnostics.mixing import reference_potential_energy
    m = nc._sigma(**_salt_kw())
    k = np.arange(nc.NZ)[:, None, None]
    S = (31.0 - k / (nc.NZ - 1)) * np.ones((nc.NZ, nc.NY, nc.NX))
    st = m.initial_state(np.zeros((nc.NY, nc.NX)), np.full(S.shape, 10.0), S=S)
    V = m.layer_volumes(st.eta, st.h)
    exact = reference_potential_energy(m.density(st.T, S).ravel(), V.ravel(), area=float(m.area.sum()))
    assert m.rpe(st) == pytest.approx(exact, rel=1e-14)
    assert m.rpe(st) != pytest.approx(reference_potential_energy(m.density(st.T).ravel(), V.ravel(),
                                                                 area=float(m.area.sum())), rel=1e-9)


# ---------------------------------------------------------------- hybrid regrid with the salt density
def _hybrid_salt(keep: bool = True, **kw) -> tuple[SigmaBaroclinic, np.ndarray]:
    """
    Flat 30 m basin, 8 hybrid layers; ρ(z) = ρ₀ − c z through S (T constant); interface targets 3.5–4.5 m apart.
    ``keep`` False: plain HYCOM1 (P1M) without the isopycnal layer preservation, which would keep the initial
    σ-layers here (each one's density already lies inside its target bracket).
    """
    nz, D, c = 8, 30.0, 0.039
    H = np.full((nc.NY, nc.NX), D)
    # target depths, surface first; every interior target lies in an interior cell of the initial 3.75 m σ-layers
    # (a target inside the top or bed cell is placed by the end-cell extrapolation of the regrid, not exactly)
    depth_t = np.array([0.0, 4.5, 8.5, 12.0, 15.5, 19.0, 22.5, 26.0, 33.0])
    params = dict(dz_top=np.full(nz, 0.5), rho_target=RHO0 + c * depth_t, keep_isopycnal=keep)
    m = SigmaBaroclinic(nc._grid(), H, dt=nc.DT, n_layers=nz, z0=None, closure="constant", nu_const=1e-4, ah=5.0,
                        theta=0.6, d_min=0.05, vertical_coordinate="ale", regrid="hybrid", regrid_params=params,
                        pgf="fv", **_salt_kw(**kw))
    return m, depth_t[1:-1]


@pytest.mark.parametrize("remap", ["ppm_h4", "plm"])
def test_hybrid_regrid_uses_salt_density(remap):
    warnings.simplefilter("ignore")
    m, depth_t = _hybrid_salt(keep=False, remap_scheme=remap)
    eta = np.zeros((nc.NY, nc.NX))
    z = m.layer_heights(eta)                                           # initial σ-layers
    S = _S_of_rho(RHO0 - 0.039 * z)
    st = m.initial_state(eta, np.full(S.shape, 10.0), S=S)

    def interface_depths(st):
        return 30.0 - np.cumsum(st.h, axis=0)[:-1]                     # interfaces 1 … N−1, depth below z = 0

    got = interface_depths(st)[::-1]                                   # surface first: interfaces 1 … N−1
    np.testing.assert_allclose(got, np.broadcast_to(depth_t[:, None, None], got.shape), atol=0.01)
    for _ in range(5):                                                 # the step regrids with ρ(T, S) as well
        st = m.step(st)
    got = interface_depths(st)[::-1]
    np.testing.assert_allclose(got, np.broadcast_to(depth_t[:, None, None], got.shape), atol=0.01)


def test_hybrid_start_from_source_uses_salt_density():
    # initial_state(h_src=…): the one-off regrid on the source levels (rho_src) sees S
    warnings.simplefilter("ignore")
    m, depth_t = _hybrid_salt()
    nsrc = 60
    h_src = np.full((nsrc, nc.NY, nc.NX), 30.0 / nsrc)
    zc = -30.0 + np.cumsum(h_src, 0) - 0.5 * h_src
    S = _S_of_rho(RHO0 - 0.039 * zc)
    st = m.initial_state(np.zeros((nc.NY, nc.NX)), np.full(S.shape, 10.0), h_src=h_src, S=S)
    got = (30.0 - np.cumsum(st.h, axis=0)[:-1])[::-1]
    np.testing.assert_allclose(got, np.broadcast_to(depth_t[:, None, None], got.shape), atol=0.01)
