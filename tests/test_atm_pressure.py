"""
Atmospheric pressure forcing of the 3-D core (regional chain Task 6, ``core.components.atm_pressure``; brief
.superpowers/sdd/2026-11-01-regional-chain/task-6-brief.md).

* static inverse barometer: closed basin (Cartesian σ and a spherical z* grid, where dx_u varies with latitude), p_a
  rising 1 hPa per 100 km, 10 damped days: η − η̄ = −(p_a − p̄_a)/(ρ₀ g) to ≤ 1 %, volume budget exact to 1e-13;
* inverse barometer at a 'radiating' edge: a channel open at its west end, p_a switched on at t = 0 (uniform with a
  fixed p̄_a, or a spatial step with the domain mean): with the edge η_IB the channel reaches the IB state, and the
  energy of the departure from it after 2 days is < 1 % of the run without the edge η_IB;
* off: bit for bit; pre-flight; restart bit for bit and the flag checked.

The atmosphere is a stub with the provider interface (``__call__(t)``, ``coverage()``, ``origin``) returning
T-point fields, calm, close to the SST and with a slight
longwave heating (a stable top layer: the constant closure has no convective adjustment, and a cooled top over
the p_a-dependent evaporation grew a hydrostatic overturning after ~10 days).  Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import numpy as np
import pytest

from core.components.surface_fluxes import BulkFluxes
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.grid.horizontal import create_cartesian_grid, create_spherical_grid
from core.model.contract import ConfigurationError
from core.restart import read_restart, write_restart

ORIGIN = np.datetime64("2017-07-01T00:00", "ns")
RHO0, G = 1000.0, 9.81
P0 = 101300.0
SAL = dict(salinity=True, s_coef=0.78, s_ref=35.0)


class _Atm:
    """Atmosphere stub on the model T points: calm air at the SST, sea-level pressure ``msl(t)`` (ny, nx) [Pa]."""

    def __init__(self, shape: tuple[int, int], msl, t1: float) -> None:
        self.shape, self.msl, self.t1 = shape, msl, float(t1)
        self.origin = ORIGIN

    def coverage(self) -> tuple[float, float]:
        return -3600.0, self.t1

    def __call__(self, t: float) -> dict[str, np.ndarray]:
        one = np.ones(self.shape)
        return dict(u10=0.0 * one, v10=0.0 * one, t2m=293.15 * one, q2m=0.0140 * one, msl=self.msl(t) * one,
                    sw_down=0.0 * one, lw_down=430.0 * one, precip=0.0 * one)


def _model(hg, H, dt, nz=2, kind="sigma", msl=None, t1=30 * 86400.0, **kw) -> SigmaBaroclinic:
    base = dict(rho0=RHO0, t_coef=0.2, t_ref=20.0, closure="constant", nu_const=1e-3, ah=0.0, theta=1.0,
                d_min=0.05, g=G, **SAL)
    if kind == "zstar":
        base.update(vertical_coordinate="ale", regrid="zstar", pgf="fv",
                    regrid_params=dict(dz_top=np.full(nz, float(np.nanmax(H)) / nz)))
    else:
        base.update(z0=1e-3)
    if msl is not None:
        base["surface_fluxes"] = BulkFluxes(_Atm(H.shape, msl, t1))
    base.update(kw)
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, **base)


def _start(m: SigmaBaroclinic, nz: int = 2):
    ny, nx = m.H.shape
    return m.initial_state(np.zeros((ny, nx)), np.full((nz, ny, nx), 20.0), S=np.full((nz, ny, nx), 35.0))


def _volume(m, eta) -> float:
    return float(np.nansum(np.nan_to_num(m.area) * eta))


# ---------------------------------------------------------------------------- static inverse barometer

def _ramp_field(hg, kind):
    """p_a rising 1 hPa per 100 km eastward (distance along x from the western cell centre)."""
    if kind == "zstar":                                       # spherical grid: x from the cumulative dx_t
        dx = np.asarray(hg["dx_t"].values, float)
        x = np.cumsum(dx, axis=1) - 0.5 * dx - 0.5 * dx[:, :1]
    else:
        x = np.asarray(hg["x_t"].values, float)
        x = x - x[:, :1]
    return P0 + 100.0 * x / 1e5


@pytest.mark.parametrize("kind", ["sigma", "zstar"])
def test_static_inverse_barometer(kind):
    """
    Closed 100 km basin, 50 m deep, f = 0, θ = 1 (damps the seiches), p_a = P0 + 1 hPa per 100 km: after 10 days
    η − η̄ = −(p_a − p̄_a)/(ρ₀ g) to ≤ 1 % of the IB range, and Σ A Δη = Σ Δt Σ A fw to 1e-13 of the volume.
    σ on a Cartesian grid; z* on a spherical grid (20–21.5°N, rows 0.5° apart), where dx_u differs between rows.
    """
    nx, ny, dt = 20, 3, 1200.0
    if kind == "zstar":
        hg = create_spherical_grid(nx=nx, ny=ny, dlon=0.05, dlat=0.5, lon0=-80.0, lat0=20.0)
        hg = hg.assign(f_t=hg["f_t"] * 0.0, f_q=hg["f_q"] * 0.0)
    else:
        hg = create_cartesian_grid(nx=nx, ny=ny, dx=5000.0, dy=5000.0, f0=0.0)
    H = np.full((ny, nx), 50.0)
    pa = _ramp_field(hg, kind)
    m = _model(hg, H, dt, kind=kind, msl=lambda t: pa, atm_pressure=True)
    st = _start(m)
    V0, fw_vol = _volume(m, st.eta), 0.0
    for _ in range(int(10 * 86400 / dt)):
        st = m.step(st)
        fw_vol += dt * float(np.nansum(np.nan_to_num(m.area) * np.nan_to_num(m._fluxes.fw)))
    A = np.nan_to_num(m.area)
    ib = -(pa - (A * pa).sum() / A.sum()) / (RHO0 * G)
    eta = st.eta - (A * st.eta).sum() / A.sum()
    err = np.abs(eta - ib).max() / np.abs(ib).max()
    Vtot = float((A * (H + st.eta)).sum())
    dv = abs(_volume(m, st.eta) - V0 - fw_vol) / Vtot
    print(f"{kind}: IB range {np.ptp(ib) * 100:.4f} cm, max |η − η_IB| / max|η_IB| = {err:.3e}, "
          f"volume budget {dv:.2e} of V, fw volume {fw_vol:.3e} m³, max|u| {np.abs(st.u).max():.2e} m/s")
    assert np.ptp(ib) > 0.009                                  # ≈ 1 cm across the basin
    assert err <= 0.01
    assert dv <= 1e-13
    np.testing.assert_array_equal(m.p_atm, pa)                 # registry diagnostic p_atm


# ---------------------------------------------------------------------------- radiating edge

def _channel(edge_ib: bool, case: str, days: float = 2.0):
    """
    50 km × 20 m channel open at the west ('radiating', resting exterior), f = 0, θ = 0.6.  'uniform': p_a = P0 + 1
    hPa everywhere from t = 0 with p̄_a = P0 (``atm_pressure_ref``); 'step': p_a = P0 + 1 hPa east of mid-channel,
    p̄_a the domain mean.  ``edge_ib`` False removes the edge η_IB (the interior force stays).  Returns the energy
    of the departure from the IB state, Σ A ½ g (η − η_IB)² + Σ_faces ½ D ū² A_f [J m⁻³ ρ₀⁻¹ · m³], η at the closed
    end and η_IB there, sampled every step.
    """
    nx, ny, dx, dt, D = 20, 3, 2500.0, 300.0, 20.0
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=0.0)
    H = np.full((ny, nx), D)
    if case == "uniform":
        pa = np.full((ny, nx), P0 + 100.0)
        kw = dict(atm_pressure_ref=P0)
    else:
        pa = np.where(np.arange(nx)[None, :] >= nx // 2, P0 + 100.0, P0) * np.ones((ny, 1))
        kw = {}
    m = _model(hg, H, dt, kind="zstar", msl=lambda t: pa, atm_pressure=True, theta=0.6,
               open_boundaries=("west",), **kw)
    if not edge_ib:
        m._atm_pressure.edges = ()
    st = _start(m)
    A = np.nan_to_num(m.area)
    pbar = P0 if case == "uniform" else (A * pa).sum() / A.sum()
    ib = -(pa - pbar) / (RHO0 * G)
    E = []
    for _ in range(int(days * 86400 / dt)):
        st = m.step(st)
        ubar = st.u.mean(axis=0)
        ke = 0.5 * D * (ubar**2).sum() * dx * dx
        pe = 0.5 * G * (A * (st.eta - ib)**2).sum()
        E.append(pe + ke)
    return np.array(E), st.eta[1, -1], ib[1, -1]


@pytest.mark.parametrize("case", ["uniform", "step"])
def test_ib_open_boundary_no_reflection(case):
    """With η_IB on the Flather exterior the channel settles in the IB state; without it, it cannot."""
    E_ib, eta_end, ib_end = _channel(True, case)
    E_no, eta_end_no, _ = _channel(False, case)
    ratio = E_ib[-1] / E_no[-1]
    print(f"{case}: E(2 d) with edge IB {E_ib[-1]:.3e}, without {E_no[-1]:.3e}, ratio {ratio:.2e}; η(closed end) "
          f"{eta_end * 100:.4f} cm vs η_IB {ib_end * 100:.4f} cm (without edge IB {eta_end_no * 100:.4f} cm)")
    assert abs(eta_end - ib_end) <= 0.01 * abs(ib_end)
    assert ratio < 0.01


# ---------------------------------------------------------------------------- off, pre-flight, restart

def _small(**kw):
    hg = create_cartesian_grid(nx=8, ny=4, dx=5000.0, dy=5000.0, f0=1e-4)
    H = np.full((4, 8), 30.0)
    pa = _ramp_field(hg, "sigma") + 50.0 * np.sin(np.arange(4))[:, None]
    return _model(hg, H, 600.0, msl=lambda t: pa * (1.0 + 1e-4 * np.sin(t / 3e4)), theta=0.6, **kw)


def test_off_bit_for_bit():
    """atm_pressure=False (explicit) is the model without the argument, phase by phase, and holds no p_a."""
    a = _small()
    b = _small(atm_pressure=False)
    assert a._atm_pressure is None and b._atm_pressure is None
    a._driver.debug = b._driver.debug = True
    sa, sb = _start(a), _start(b)
    for _ in range(5):
        sa, sb = a.step(sa), b.step(sb)
    assert a._driver.phase_hashes == b._driver.phase_hashes
    for f in ("eta", "u", "v", "T", "S"):
        np.testing.assert_array_equal(getattr(sa, f), getattr(sb, f))
    assert a.p_atm is None
    # the force is really on when asked: same set-up with atm_pressure moves η differently
    c = _small(atm_pressure=True)
    sc = _start(c)
    for _ in range(5):
        sc = c.step(sc)
    assert not np.array_equal(sc.eta, sa.eta)


def test_preflight_refuses_without_bulk():
    """atm_pressure=True without surface_fluxes, a reference in hPa and a non-bool flag are all reported at once."""
    hg = create_cartesian_grid(nx=8, ny=4, dx=5000.0, dy=5000.0, f0=1e-4)
    H = np.full((4, 8), 30.0)
    with pytest.raises(ConfigurationError) as e:
        _model(hg, H, 600.0, atm_pressure=True, atm_pressure_ref=1013.0)
    msg = str(e.value)
    assert "needs surface_fluxes" in msg and "atm_pressure_ref=1013.0" in msg
    with pytest.raises(ConfigurationError, match="True or False"):
        _model(hg, H, 600.0, atm_pressure="yes")
    with pytest.warns(UserWarning, match="atm_pressure_ref is ignored"):
        _model(hg, H, 600.0, atm_pressure_ref=1e5)


def test_restart_bit_for_bit(tmp_path):
    """6 steps = 3 + restart + 3, bit for bit, with p_a varying in space and time; the flag is checked."""
    m = _small(atm_pressure=True)
    st = _start(m)
    for _ in range(3):
        st = m.step(st)
    path = tmp_path / "r.nc"
    write_restart(path, m, st)
    for _ in range(3):
        st = m.step(st)
    m2 = _small(atm_pressure=True)
    st2 = read_restart(path, m2)
    for _ in range(3):
        st2 = m2.step(st2)
    for f in ("eta", "u", "v", "T", "S"):
        np.testing.assert_array_equal(getattr(st2, f), getattr(st, f))
    with pytest.raises(ValueError, match="atm_pressure"):
        read_restart(path, _small())
    with pytest.raises(ValueError, match="atm_pressure_ref"):
        read_restart(path, _small(atm_pressure=True, atm_pressure_ref=1e5))
