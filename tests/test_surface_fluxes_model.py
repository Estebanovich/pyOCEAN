"""
N2b task 4 (plan docs/superpowers/plans/2026-10-31-n2b-bulk-surface-fluxes.md, spec
docs/superpowers/specs/2026-10-31-n2b-bulk-surface-fluxes-design.md): bulk surface fluxes as a component of the 3-D
model, ``SigmaBaroclinic(..., surface_fluxes=BulkFluxes(atmosphere, scheme=...))``.

Every test uses a synthetic canonical atmosphere of a few hourly records (no files) on a regular lon/lat grid, taken
to the model T points by the real ``forcing.atmosphere.AtmosphereProvider`` (the Cartesian test grids get lon_t,
lat_t).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import xarray as xr

from core.components.surface_fluxes import BulkFluxes, surface_average
from core.dynamics import sigma_baroclinic as sb
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid
from core.model.contract import ConfigurationError
from core.restart import read_restart, write_restart
from forcing.atmosphere import AtmosphereProvider
from tests import n1_cases as nc

ORIGIN = np.datetime64("2017-07-01T00:00", "ns")
HOUR = 3600.0
SAL = dict(salinity=True, s_coef=0.78, s_ref=30.0)
CP = 3985.0


# ---------------------------------------------------------------------------------------------------------- fixtures
def _ll(hg: xr.Dataset) -> xr.Dataset:
    """The Cartesian test grid with lon_t, lat_t (1 km ≈ 0.01°) for the provider's bilinear step."""
    hg = hg.copy()
    hg["lon_t"] = -70.0 + hg["x_t"] / 1e5
    hg["lat_t"] = 40.0 + hg["y_t"] / 1e5
    return hg


def _source(times_s: np.ndarray, *, uniform: bool = False, **const) -> xr.Dataset:
    """
    Canonical atmosphere (time, lat, lon) on a 0.05° grid around the test grids.  Mildly varying in space and time
    (realistic summer values over the shelf), or spatially uniform; ``const`` overrides a field with a constant.
    """
    lat = np.arange(39.8, 40.61, 0.05)[::-1]
    lon = np.arange(-70.2, -69.39, 0.05)
    tt = np.asarray(times_s, float)
    time = ORIGIN + (tt * 1e9).astype("timedelta64[ns]")
    s = 0.0 if uniform else 1.0
    ph = np.sin(2 * np.pi * tt / 86400.0)[:, None, None]
    x = (lon[None, None, :] + 70.0) * s
    y = (lat[None, :, None] - 40.0) * s
    one = np.ones((tt.size, lat.size, lon.size))
    f = {"u10": 7.0 + 3.0 * x + 0.5 * ph, "v10": -3.0 + 2.0 * y - 0.3 * ph, "t2m": 293.0 + 2.0 * y + 0.4 * ph,
         "q2m": 0.011 + 0.002 * x, "msl": 101300.0 + 300.0 * y, "sw_down": 450.0 + 150.0 * ph + 20.0 * x,
         "lw_down": 380.0 + 10.0 * y, "precip": 3e-6 * (1.0 + x) * (1.0 + 0.5 * ph)}
    f.update(const)
    return xr.Dataset({k: (("time", "lat", "lon"), np.asarray(v, float) * one) for k, v in f.items()},
                      coords={"time": time, "lat": lat, "lon": lon})


def _atm(hg: xr.Dataset, t1: float = 4 * HOUR, t0: float = -HOUR, **kw) -> AtmosphereProvider:
    return AtmosphereProvider(_source(np.arange(t0, t1 + 1.0, HOUR), **kw), _ll(hg), ORIGIN)


def _bulk(hg: xr.Dataset, scheme: str = "ncar", atm_kw: dict | None = None, **kw) -> BulkFluxes:
    return BulkFluxes(_atm(hg, **(atm_kw or {})), scheme=scheme, **kw)


def _S(nz: int) -> np.ndarray:
    k = np.arange(nz)[:, None, None]
    return (30.5 - 0.5 * k / (nz - 1)) * np.ones((nz, nc.NY, nc.NX))


def _basin(kind: str, bulk: bool = True, scheme: str = "ncar", H: np.ndarray | None = None, **kw):
    """The golden-matrix basin (σ, z*, hybrid or ALE-σ) with salinity and, optionally, the bulk fluxes."""
    if bulk:
        kw["surface_fluxes"] = _bulk(nc._grid(), scheme)
    kw.update(SAL)
    nz = 8 if kind == "hybrid" else nc.NZ
    if H is None:
        m = nc._sigma(**kw) if kind == "sigma" else nc._ale(kind, nz=nz, **kw)
        st = m.initial_state(nc._eta(), nc._T(nz), S=_S(nz))
        return m, st
    base = dict(rho0=nc.RHO0, t_coef=nc.TCOEF, z0=None, closure="constant", nu_const=1e-4, ah=5.0, theta=0.6,
                d_min=0.05)
    if kind == "ale_sigma":
        base.update(vertical_coordinate="ale", regrid="sigma", regrid_params={}, pgf="fv")
    elif kind == "zstar":
        base.update(vertical_coordinate="ale", regrid="zstar", regrid_params=nc._zstar_params(nz), pgf="fv")
    elif kind == "hybrid":
        p = nc._hybrid_params(nz)
        p.update(d_sigma=3.0, d_hybrid=6.0)                             # shallow σ zone: cells with H ≤ 0 can dry
        base.update(vertical_coordinate="ale", regrid="hybrid", regrid_params=p, pgf="fv")
    base.update(kw)
    m = SigmaBaroclinic(nc._grid(), H, dt=nc.DT, n_layers=nz, **base)
    eta = np.maximum(0.0, -np.nan_to_num(H))                            # at rest; dry banks at zero depth
    st = m.initial_state(eta, nc._T(nz), S=_S(nz))
    return m, st


def _land_dry_H(dry: bool = True) -> np.ndarray:
    H = np.array(nc.H, float, copy=True)
    H[0, 0] = np.nan                                                    # land
    if dry:
        H[-1, -1] = -1.0                                                # dry bank 1 m above the sea level
    return H


def _column(kind: str, bulk: BulkFluxes | None, nz: int = 10, depth: float = 20.0, dt: float = 60.0,
            T0: float = 20.0, dz_top: np.ndarray | None = None, **kw):
    """Doubly periodic 3×3 column at rest, horizontally uniform (CROCO's single-column setup, N3 task 10)."""
    hg = create_cartesian_grid(nx=3, ny=3, dx=1000.0, dy=1000.0, f0=1e-4)
    base = dict(rho0=1000.0, t_coef=0.2, closure="constant", nu_const=1e-4, kv=1e-4, theta=0.6, d_min=0.05,
                periodic=("x", "y"), surface_fluxes=bulk, **SAL)
    dz = np.full(nz, depth / nz) if dz_top is None else dz_top
    if kind == "zstar":
        base.update(vertical_coordinate="ale", regrid="zstar", pgf="fv", regrid_params=dict(dz_top=dz))
    elif kind == "hybrid":
        rho_c = 1000.0 - 0.2 * T0
        base.update(vertical_coordinate="ale", regrid="hybrid", pgf="fv",
                    regrid_params=dict(dz_top=dz, rho_target=np.linspace(rho_c - 0.5, rho_c + 0.5, nz + 1)))
    base.update(kw)
    m = SigmaBaroclinic(hg, np.full((3, 3), depth), dt=dt, n_layers=nz, **base)
    st = m.initial_state(np.zeros((3, 3)), np.full((nz, 3, 3), T0), S=np.full((nz, 3, 3), 35.0))
    return m, st


def _col_grid():
    return create_cartesian_grid(nx=3, ny=3, dx=1000.0, dy=1000.0, f0=1e-4)


def _wet(m, st) -> np.ndarray:
    return m.fs.active & (m.fs.H + st.eta > m.fs.d_min) & (np.nan_to_num(m.area) > 0)


def _heat_content(m, st, T_ref: float) -> float:
    """ρ₀ c_p Σ (T − T_ref) V over the active cells [J], summed exactly (math.fsum)."""
    V = m.layer_volumes(st.eta, st.h)
    act = np.broadcast_to(m.fs.active[None], V.shape)
    return m.rho0 * CP * math.fsum(((st.T - T_ref) * V)[act].ravel())


def _volume(m, st) -> float:
    D = np.maximum(m.H + st.eta, 0.0)
    return math.fsum((m.area * D)[m.fs.active].ravel())


# ---------------------------------------------------------------------------------------------------------- tests
@pytest.mark.parametrize("kind", ["sigma", "zstar", "hybrid"])
def test_off_is_bit_for_bit(kind):
    """surface_fluxes=None is the model without the argument, bit for bit (prescribed wind on both)."""
    a, sa = _basin(kind, bulk=False, surface_fluxes=None)
    b, sb_ = _basin(kind, bulk=False)
    wind = nc._wind(True)
    for n in range(10):
        sa = a.step(sa, *wind(n))
        sb_ = b.step(sb_, *wind(n))
    assert nc.state_hashes(sa, a) == nc.state_hashes(sb_, b)
    assert a._driver.step.__func__ is type(a._driver).step              # no bulk wrapper on the default path


@pytest.mark.parametrize("kind", ["sigma", "zstar", "hybrid"])
@pytest.mark.parametrize("scheme", ["ncar", "ncar_ly04", "coare", "kara"])
def test_equals_prescribed(kind, scheme):
    """The bulk run is the run with the same τ, heat and fw passed by hand, bit for bit."""
    a, sa = _basin(kind, scheme=scheme)
    b, sb_ = _basin(kind, bulk=False)
    for _ in range(5):
        tx, ty, fw, heat = a._surface_fluxes.forcing(sa, sa.time + a.dt / 2)
        sa = a.step(sa)
        sb_ = b.step(sb_, tx, ty, fw=fw, heat=heat)
        assert tx.shape == (nc.NY, nc.NX + 1) and ty.shape == (nc.NY + 1, nc.NX)
    assert nc.state_hashes(sa, a) == nc.state_hashes(sb_, b)


@pytest.mark.parametrize("kind", ["sigma", "zstar", "hybrid"])
def test_heat_conservation(kind):
    """
    Δ(Σ ρ₀ c_p T V) = Σ (Q_ns + Q_sw) A Δt to ≤ 1e-13 of the change, with a land column (and a dry one where the
    coordinate can dry: z* refuses H ≤ 0 cells, so its basin has the land column only).  A strong shortwave (2e5
    W m⁻²) makes the change ≫ the round-off of the content sums.

    The bulk heat and stress of each step (the component evaluated on the state, as ``Driver._step_bulk`` does) drive
    the model without the freshwater flux: with fw the mass that enters or leaves carries the top-layer temperature
    (N2a, MOM6 applyBoundaryFluxesInOut), a heat-content change ρ₀ c_p Σ T ΔV that is not a surface heat flux and is
    not visible after the step.  ``test_equals_prescribed`` shows the bulk step is exactly this prescribed step with
    fw added, and ``test_freshwater_and_salt_conservation`` checks the fw part.
    """
    H = _land_dry_H(dry=kind != "zstar")
    m, st = _basin(kind, bulk=False, H=H)
    calc, _ = _basin(kind, bulk=False, H=H,
                     surface_fluxes=BulkFluxes(_atm(nc._grid(), sw_down=2e5), scheme="ncar"))
    T_ref = float(np.mean(st.T[np.isfinite(st.T)]))
    c0, expected = _heat_content(m, st, T_ref), 0.0
    A = np.nan_to_num(m.area)
    for _ in range(6):
        tx, ty, _fw, heat = calc._surface_fluxes.forcing(st, st.time + m.dt / 2)
        st = m.step(st, tx, ty, heat=heat)
        L = calc._surface_fluxes.last
        np.testing.assert_array_equal(heat["Q_ns"] + heat["Q_sw"], L["q_sens"] + L["q_lat"] + L["q_lw"] + L["q_sw"])
        expected += m.dt * math.fsum((A * (heat["Q_ns"] + heat["Q_sw"])).ravel())
    change = _heat_content(m, st, T_ref) - c0
    assert abs(change - expected) <= 1e-13 * abs(expected), (change, expected)


@pytest.mark.parametrize("kind", ["sigma", "zstar", "hybrid"])
def test_heat_conservation_with_freshwater(kind):
    """
    Review m1: the full bulk step (heat and fw together), one step from a uniform T₀.  The freshwater mass enters
    and leaves at the top-layer temperature (N2a; MOM6 applyBoundaryFluxesInOut, MOM_diabatic_aux.F90:1034-1045,
    ``Temp_in = T2d``), which is T₀ after the advection of a uniform field, so

        Δ Σ ρ₀ c_p (T − T_ref) V = Σ (Q_ns + Q_sw) A Δt + ρ₀ c_p Σ fw A Δt (T₀ − T_ref)

    to ≤ 1e-13 of the change (T_ref = T₀ − 1 keeps the advective term in the balance).  One step only: afterwards the
    top-layer T seen by the freshwater substep (after the advection) is not recoverable from the states.
    """
    H = _land_dry_H(dry=kind != "zstar")
    T0 = 12.0
    m, _ = _basin(kind, bulk=False, H=H,
                  surface_fluxes=BulkFluxes(_atm(nc._grid(), sw_down=2e5, precip=4e-4), scheme="ncar"))
    nz = m.core.n_layers
    st = m.initial_state(np.maximum(0.0, -np.nan_to_num(H)), np.full((nz, nc.NY, nc.NX), T0), S=_S(nz))
    T_ref = T0 - 1.0
    c0 = _heat_content(m, st, T_ref)
    st = m.step(st)
    L, A = m._surface_fluxes.last, np.nan_to_num(m.area)
    q = math.fsum((A * (L["q_sens"] + L["q_lat"] + L["q_lw"] + L["q_sw"])).ravel())
    adv = m.rho0 * CP * (T0 - T_ref) * math.fsum((A * L["fw"]).ravel())
    expected = m.dt * (q + adv)
    assert abs(m.dt * adv) > 1e-3 * abs(expected)                       # the advective term is in the balance
    change = _heat_content(m, st, T_ref) - c0
    assert abs(change - expected) <= 1e-13 * abs(expected), (change, expected)


@pytest.mark.parametrize("kind", ["sigma", "zstar", "hybrid"])
def test_freshwater_and_salt_conservation(kind):
    """Σ ΔV = Σ (P − E) A Δt at the N2a level (1e-13 of the volume) and the salt content constant (1e-13)."""
    H = _land_dry_H(dry=kind != "zstar")
    m, st = _basin(kind, H=H, surface_fluxes=BulkFluxes(_atm(nc._grid(), precip=4e-5), scheme="ncar"))
    V0, salt0, src = _volume(m, st), m.tracer_content(st, tracer="S"), 0.0
    A = np.nan_to_num(m.area)
    for _ in range(8):
        st = m.step(st)
        L = m._surface_fluxes.last
        np.testing.assert_array_equal(L["precip"] - L["evap"], L["fw"])
        src += m.dt * math.fsum((A * L["fw"]).ravel())
        assert abs(_volume(m, st) - (V0 + src)) <= 1e-13 * V0
        assert abs(m.tracer_content(st, tracer="S") - salt0) <= 1e-13 * salt0
    assert src > 0


@pytest.mark.filterwarnings("ignore:limit_outflow:RuntimeWarning")    # 40 m s⁻¹ over 2 mm layers, periodic: in = out
def test_evaporation_exceeds_thin_top_layer():
    """
    Review Focus 5: strong evaporation over a thin top layer of a hybrid column (dry, hot, 40 m s⁻¹ wind, Δt 1800 s:
    E Δt ≈ 3 mm > the 2 mm top layer).  The deficit spills to the layer below (``n_limited`` counts it), the volume
    and the salt are conserved.  The top layer is 2 mm, not 1 mm: a nominal top layer equal to the regrid floor
    (1 mm) makes the z* part of the column overshoot η (N2a task 5 report, test_n2a_freshwater).
    """
    nz = 6
    dz = np.full(nz, (20.0 - 2e-3) / (nz - 1))
    dz[0] = 2e-3
    bulk = BulkFluxes(_atm(_col_grid(), t1=10 * HOUR, uniform=True, u10=40.0, v10=0.0, q2m=0.0, t2m=308.0,
                           precip=0.0), scheme="ncar")
    m, st = _column("hybrid", bulk, nz=nz, dt=1800.0, T0=32.0, dz_top=dz, nu_const=1e-2, kv=1e-2)
    assert st.h[-1].max() < 2.2e-3
    V0, salt0 = _volume(m, st), m.tracer_content(st, tracer="S")
    src = 0.0
    for _ in range(4):
        st = m.step(st)
        src += m.dt * math.fsum((m.area * m._surface_fluxes.last["fw"]).ravel())
        assert st.h.min() >= 0.0
        assert abs(m.tracer_content(st, tracer="S") - salt0) <= 1e-13 * salt0
    assert -src / (4 * float(m.area.sum())) > 2e-3                     # E Δt per step above the top layer
    assert abs(_volume(m, st) - (V0 + src)) <= 1e-13 * V0
    assert st.n_limited > 0


def _column_run(kind: str, nz: int, dz, steps: int = 20, **kw):
    bulk = BulkFluxes(_atm(_col_grid(), uniform=True), scheme="ncar")
    m, st = _column(kind, bulk, nz=nz, dz_top=dz, **kw)
    for _ in range(steps):
        st = m.step(st)
    return m, st, {k: v.copy() for k, v in m._surface_fluxes.last.items()}


FLUX_KEYS = ("sst_bulk", "q_sens", "q_lat", "q_lw", "q_sw", "evap", "tau_x", "tau_y")


def test_sst_independent_of_coordinate(monkeypatch):
    """
    Criterion 4: a horizontally uniform column at rest under the same atmosphere gives the same SST, fluxes and T(z)
    in z* with fixed layers and in σ, to ≤ 1e-9 after 20 steps.  "Fixed layers" needs fw = 0 (with fw the free
    surface moves and the layers with it), so the component's fw is zeroed here; heat and stress are the bulk ones.
    """
    from core.components import surface_fluxes as sfm
    full = sfm.SurfaceFluxes.forcing

    def no_fw(self, st, t):
        tx, ty, fw, heat = full(self, st, t)
        return tx, ty, np.zeros_like(fw), heat
    monkeypatch.setattr(sfm.SurfaceFluxes, "forcing", no_fw)
    ms, ss, ls = _column_run("sigma", 10, None)
    mz, sz, lz = _column_run("zstar", 10, None)
    for k in FLUX_KEYS:
        np.testing.assert_allclose(lz[k], ls[k], rtol=1e-9, atol=0.0, err_msg=k)
    np.testing.assert_allclose(sz.T, ss.T, rtol=0, atol=1e-9)
    assert np.abs(sz.T - 20.0).max() > 1e-3                             # the column did heat or cool
    # hybrid: top layer 0.5 m over 1.95 m layers (its other targets are absent), against σ's 2 m layers.  The 1 m
    # mean makes the SST independent of the top-layer thickness, but a different grid resolves the surface heating
    # and the wind-driven shear differently, so the comparison is made where both are well mixed over the top
    # metres (κ = ν = 0.1 m² s⁻¹).  Residual structure over h = 2 m: ΔT ≈ |Q_ns| h/(ρ₀ c_p κ) ≈ 140·2/(1000·3985·0.1)
    # = 7e-4 °C and Δu ≈ τ h/(ρ₀ ν) ≈ 0.07·2/(1000·0.1) = 1.4e-3 m s⁻¹.  Hence |ΔSST| ≤ 7e-4 °C, heat fluxes within
    # ∂Q/∂SST·ΔSST ≈ 30 W m⁻² K⁻¹ · 7e-4 ≈ 0.02 W m⁻², evaporation within that latent flux / (ρ₀ L_v) ≈ 1e-11 m s⁻¹,
    # and τ ∝ |U₁₀ − u_s| (U₁₀ − u_s) within 2 Δu/|U₁₀| ≈ 4e-4 relative (measured: 4e-5 °C, ≤ 6e-4 W m⁻², 4e-5).
    k_mix = dict(kv=0.1, nu_const=0.1)
    _, _, ls = _column_run("sigma", 10, None, **k_mix)
    _, sh, lh = _column_run("hybrid", 11, np.r_[0.5, np.full(10, 19.5 / 10)], **k_mix)
    assert abs(sh.h[-1, 0, 0] - 0.5) < 1e-3                             # the hybrid kept its thin top layer
    np.testing.assert_allclose(lh["sst_bulk"], ls["sst_bulk"], rtol=0, atol=7e-4)
    for k in ("q_sens", "q_lat", "q_lw", "q_sw"):
        np.testing.assert_allclose(lh[k], ls[k], rtol=0, atol=0.02, err_msg=k)
    np.testing.assert_allclose(lh["evap"], ls["evap"], rtol=0, atol=1e-11)
    for k in ("tau_x", "tau_y"):
        np.testing.assert_allclose(lh[k], ls[k], rtol=4e-4, atol=0.0, err_msg=k)


def test_sst_with_freshwater_sigma_vs_zstar():
    """
    With the freshwater flux on, the layers follow the free surface and the two coordinates move the added volume
    differently: σ through the column by its vertical advection (WENO5 of ω), z* by the regrid + PPM_H4 remap.  Both
    errors scale with the curvature of T(z) × h², so T(z) differs at the 1e-6 °C level after 20 steps (measured
    1.3e-6; class (c), the schemes, not the SST definition) and the fluxes by ≤ 1e-5 relative.
    """
    _, ss, ls = _column_run("sigma", 10, None)
    _, sz, lz = _column_run("zstar", 10, None)
    np.testing.assert_allclose(sz.T, ss.T, rtol=0, atol=1e-5)
    for k in FLUX_KEYS:
        np.testing.assert_allclose(lz[k], ls[k], rtol=1e-4, atol=1e-5 * np.abs(ls[k]).max(), err_msg=k)
    assert np.abs(ls["evap"]).max() > 0


def test_surface_average_shallow_and_thin():
    """Review Focus 2: the h_sfc average over a shallow column, a thin top layer, and h_sfc = 0 (the top layer)."""
    # 0.4 m column, 4 layers: the whole column
    h = np.full((4, 1, 1), 0.1)
    q = np.arange(4.0)[:, None, None]
    assert surface_average(h, q, 1.0)[0, 0] == pytest.approx(1.5, abs=1e-15)
    # 1 mm top layer over a 5 m layer (and a 3 m bed layer): 1 m average = (0.001·10 + 0.999·20)/1
    h = np.array([3.0, 5.0, 1e-3])[:, None, None]
    q = np.array([30.0, 20.0, 10.0])[:, None, None]
    assert surface_average(h, q, 1.0)[0, 0] == pytest.approx((1e-3 * 10.0 + 0.999 * 20.0) / 1.0, rel=1e-14)
    # a 1 m window across two thick layers takes the top one only
    h2 = np.array([3.0, 2.0, 2.0])[:, None, None]
    assert surface_average(h2, q, 1.0)[0, 0] == 10.0
    # h_sfc = 0: the top layer as it is
    assert surface_average(h, q, 0.0)[0, 0] == 10.0
    # a dry (empty) column gives NaN
    assert np.isnan(surface_average(np.zeros((3, 1, 1)), q, 1.0)[0, 0])


@pytest.mark.parametrize("kind", ["sigma", "ale_sigma"])
def test_dry_and_land_get_nothing(kind):
    """A dry and a land column get no flux and stay still bit for bit (T, S and η as without forcing)."""
    H = _land_dry_H()
    a, sa = _basin(kind, H=H)
    b, sb_ = _basin(kind, bulk=False, H=H)
    for _ in range(5):
        sa = a.step(sa)
        sb_ = b.step(sb_)
    L = a._surface_fluxes.last
    for k in ("q_sens", "q_lat", "q_lw", "q_sw", "evap", "precip", "fw", "tau_x", "tau_y"):
        assert L[k][0, 0] == 0.0 and L[k][-1, -1] == 0.0, k
        assert np.abs(L[k]).max() > 0, k
    for f in ("T", "S"):
        for j, i in ((0, 0), (-1, -1)):
            x, y = getattr(sa, f)[:, j, i], getattr(sb_, f)[:, j, i]
            assert np.array_equal(np.nan_to_num(x, nan=-9.0), np.nan_to_num(y, nan=-9.0)), (f, j, i)
    assert sa.eta[-1, -1] == sb_.eta[-1, -1]


def test_coverage_ends_mid_run():
    """Review Focus 3: an atmosphere that ends inside the run stops the step that needs it, naming the date."""
    hg = nc._grid()
    times = np.arange(0.0, 241.0, 120.0)                                # records at 0, 2, 4 min: t ≤ 240 s
    bulk = BulkFluxes(AtmosphereProvider(_source(times), _ll(hg), ORIGIN), scheme="ncar")
    m = nc._sigma(surface_fluxes=bulk, **SAL)
    st = m.initial_state(nc._eta(), nc._T(), S=_S(nc.NZ))
    for _ in range(4):                                                  # t^{n+½} = 30, 90, 150, 210 s
        st = m.step(st)
    with pytest.raises(ValueError, match="2017-07-01T00:04:30"):       # step 5 needs t = 270 s
        m.step(st)


def test_preflight_refuses_all_at_once():
    """
    Every configuration error in one report; an empty coverage (one record: c0 = c1) is refused there.  Whether the
    coverage holds the run is a runtime check (ruling R4): an atmosphere starting after t = 0 constructs and the
    first step raises, naming the date and the coverage.
    """
    hg = nc._grid()
    single = AtmosphereProvider(_source(np.array([0.0])), _ll(hg), ORIGIN)                  # c0 = c1 = 0
    with pytest.raises(ConfigurationError) as e:
        nc._sigma(surface_fluxes=BulkFluxes(single, scheme="nope", albedo=2.0))
    msg = str(e.value)
    for part in ("unknown bulk scheme", "needs salinity=True", "empty or not finite", "albedo"):
        assert part in msg, (part, msg)
    late = AtmosphereProvider(_source(np.arange(HOUR, 4 * HOUR, HOUR)), _ll(hg), ORIGIN)   # starts at 1 h
    m = nc._sigma(surface_fluxes=BulkFluxes(late), **SAL)
    st = m.initial_state(nc._eta(), nc._T(), S=_S(nc.NZ))
    with pytest.raises(ValueError, match=r"2017-07-01T00:00:30.*coverage \[2017-07-01T01:00"):
        m.step(st)
    with pytest.raises(ConfigurationError) as e:
        nc._sigma(surface_fluxes=BulkFluxes(_atm(hg), h_sfc=-1.0, emissivity=0.0, z_q=10.0,
                                            scheme="ncar_ly04"), **SAL)
    msg = str(e.value)
    for part in ("h_sfc", "emissivity", "z_q"):
        assert part in msg, (part, msg)
    m, st = _basin("sigma")
    for kw in ({"tau_x": np.zeros((nc.NY, nc.NX + 1))}, {"heat": {"Q_ns": np.zeros((nc.NY, nc.NX))}},
               {"fw": np.zeros((nc.NY, nc.NX))}):
        with pytest.raises(ConfigurationError, match="surface_fluxes"):
            m.step(st, **kw)


def test_coverage_trimmed_to_the_requested_window():
    """Task 3 m2: a provider given the run window reports its coverage trimmed to it (the padded hour-ending
    record past t1 is not offered to the pre-flight) and refuses times outside it."""
    hg = nc._grid()
    p = AtmosphereProvider(_source(np.arange(-HOUR, 5 * HOUR, HOUR)), _ll(hg), ORIGIN,
                           window=(ORIGIN, ORIGIN + np.timedelta64(3, "h")))
    assert p.coverage() == (0.0, 3 * HOUR)
    assert p.origin == ORIGIN
    with pytest.raises(ValueError, match="2017-07-01T03:30"):
        p(3.5 * HOUR)


@pytest.mark.parametrize("kind", ["sigma", "hybrid"])
def test_restart_bit_for_bit_with_bulk(tmp_path, kind):
    sb.set_threads(1)
    m, st = _basin(kind)
    for _ in range(6):
        st = m.step(st)
    m1, s1 = _basin(kind)
    for _ in range(3):
        s1 = m1.step(s1)
    write_restart(tmp_path / "bulk.nc", m1, s1)
    m2, _ = _basin(kind)
    s2 = read_restart(tmp_path / "bulk.nc", m2)
    for _ in range(3):
        s2 = m2.step(s2)
    assert nc.state_hashes(s2, m2) == nc.state_hashes(st, m)
    m3, _ = _basin(kind, scheme="coare")                                # another scheme is refused
    with pytest.raises(ValueError, match="bulk_scheme"):
        read_restart(tmp_path / "bulk.nc", m3)
    m4, _ = _basin(kind, bulk=False)                                    # and so is a model without bulk
    with pytest.raises(ValueError, match="surface_fluxes"):
        read_restart(tmp_path / "bulk.nc", m4)


def test_restart_with_atmosphere_from_the_restart_date(tmp_path):
    """
    Review I1 / ruling R4: a continuation whose atmosphere covers only [T, …] (ERA5 loaded from the restart date,
    ``window`` trimming the padded record before it) constructs, reads the restart and steps bit for bit like the
    uninterrupted run with an atmosphere that covers everything.
    """
    sb.set_threads(1)
    hg, n0 = nc._grid(), 3
    T = n0 * nc.DT
    m, st = _basin("hybrid")
    for _ in range(2 * n0):
        st = m.step(st)
    m1, s1 = _basin("hybrid")
    for _ in range(n0):
        s1 = m1.step(s1)
    write_restart(tmp_path / "cont.nc", m1, s1)
    later = AtmosphereProvider(_source(np.arange(-HOUR, 4 * HOUR + 1.0, HOUR)), _ll(hg), ORIGIN,
                               window=(ORIGIN + np.timedelta64(int(T), "s"), ORIGIN + np.timedelta64(4, "h")))
    assert later.coverage()[0] == T > 0
    m2, _ = _basin("hybrid", bulk=False, surface_fluxes=BulkFluxes(later, scheme="ncar"))
    s2 = read_restart(tmp_path / "cont.nc", m2)
    for _ in range(n0):
        s2 = m2.step(s2)
    assert nc.state_hashes(s2, m2) == nc.state_hashes(st, m)


def test_threads_bit_for_bit():
    out = []
    for n in (1, 5):
        sb.set_threads(n)
        try:
            m, st = _basin("hybrid")
            for _ in range(5):
                st = m.step(st)
            out.append(nc.state_hashes(st, m))
        finally:
            sb.set_threads(1)
    assert out[0] == out[1]


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
def test_variance_budget_closes_with_bulk(kind):
    """
    N2b task 8 (ruling R6; open check of task 6): bulk fluxes together with ``variance_budget=True``.  The exact
    discrete budget of Σ V c² of S and T (Klingbeil et al. 2014; N2a ``tests/test_n2a_budget.py``) closes to the same
    1e-12 relative tolerance with the real freshwater flux and the heat flux of the bulk formulae in the basin.
    """
    sb.set_threads(1)
    m, st = _basin(kind, variance_budget=True, kh=50.0, kv=1e-3)
    keys = {"num_adv", "phy_v", "phy_h", "surf", "sponge"} | ({"num_remap"} if m.ale else set())
    for _ in range(6):
        V0 = m.layer_volumes(st.eta, st.h)
        S0, T0 = st.S, st.T
        st = m.step(st)
        V1 = m.layer_volumes(st.eta, st.h)
        b = st.budget
        assert set(b) == {f"{c}_{k}" for c in "ST" for k in keys}
        for c, c0, c1 in (("S", S0, st.S), ("T", T0, st.T)):
            lhs = (V1 * c1**2).sum() - (V0 * c0**2).sum()
            rhs = m.dt * (b[f"{c}_surf"].sum() - b[f"{c}_num_adv"].sum() - b[f"{c}_phy_v"].sum()
                          - b[f"{c}_phy_h"].sum() - b.get(f"{c}_num_remap", np.zeros(1)).sum()
                          - b[f"{c}_sponge"].sum())
            assert abs(lhs - rhs) <= 1e-12 * (V0 * c0**2).sum(), (kind, c, lhs, rhs)
            assert np.abs(b[f"{c}_surf"]).sum() > 0                        # the bulk fluxes are a real source
