"""
Tides at the 3-D open boundary, body tide and scalar SAL in 3-D (regional chain Task 5,
``core.components.boundary_tides``; brief .superpowers/sdd/2026-11-01-regional-chain/task-5-brief.md).

* tidal channel: a straight homogeneous z* channel forced at its open west edge with an M2 η of 0.5 m; η at the
  closed end equals the 2-D ``WetDryFreeSurface`` channel forced with the same Flather exterior;
* body tide: closed homogeneous basin with only the equilibrium tide; the 3-D η equals the 2-D run with the
  same force at tⁿ + Δt/2;
* restart in the middle of the ramp, bit for bit; tides off bit for bit; pre-flight; rotation of TPXO transports
  to grid axes; scalar SAL g(1 − β)∇η in the free surface and in the Flather speed.

Index order (k, j, i), k = 0 at the bed.
"""
from __future__ import annotations

import warnings
from pathlib import Path

import numpy as np
import pytest
import xarray as xr

from core.components.boundary_tides import BoundaryTides, TidalPotential
from core.dynamics.sigma_baroclinic import SigmaBaroclinic
from core.dynamics.wetting_drying import WetDryFreeSurface
from core.grid.horizontal import create_cartesian_grid, create_spherical_grid
from core.model.contract import ConfigurationError
from core.restart import read_restart, write_restart
from forcing.tides import OMEGA

REF = np.datetime64("2026-01-01T00:00", "ns")
DEG = 111_195.0                                    # m per degree (lat/lon labels of the Cartesian channel)


def _channel_grid(nx: int, ny: int, dx: float):
    """Cartesian grid, f = 0, with lon/lat labels (-70 °E, 30 °N + metres / 111 km) for the tide interpolation."""
    hg = create_cartesian_grid(nx=nx, ny=ny, dx=dx, dy=dx, f0=0.0)
    add = {}
    for k in ("t", "u", "v"):
        add[f"lon_{k}"] = (hg[f"x_{k}"].dims, -70.0 + hg[f"x_{k}"].values / DEG)
        add[f"lat_{k}"] = (hg[f"y_{k}"].dims, 30.0 + hg[f"y_{k}"].values / DEG)
    return hg.assign(add)


def _constants(amp: float = 0.5, phase_deg: float = 40.0, U: complex | None = None, V: complex | None = None,
               names=("m2",)):
    """TPXO-like constants, uniform: h = amp e^{−iG} [m]; optional uniform transports U, V [m² s⁻¹]."""
    lat = np.arange(29.0, 31.01, 0.25)
    lon = np.arange(-71.0, -67.99, 0.25)
    nc = len(names)
    h = np.full((nc, lat.size, lon.size), amp * np.exp(-1j * np.deg2rad(phase_deg)))
    ds = xr.Dataset({"h": (("constituent", "lat", "lon"), h)},
                    coords={"constituent": list(names), "lat": lat, "lon": lon})
    if U is not None:
        ds["U"] = xr.DataArray(np.full((nc, lat.size, lon.size), U), dims=("constituent", "lat", "lon_u"),
                               coords={"lon_u": lon - 0.125})
        ds["V"] = xr.DataArray(np.full((nc, lat.size, lon.size), V), dims=("constituent", "lat_v", "lon"),
                               coords={"lat_v": lat - 0.125})
    return ds


def _channel_model(hg, H, dt, nz=4, **kw):
    base = dict(rho0=1000.0, t_coef=0.2, z0=None, closure="constant", nu_const=1e-4, ah=0.0, theta=0.6, d_min=0.05,
                vertical_coordinate="ale", regrid="zstar", regrid_params=dict(dz_top=np.full(nz, H.max() / nz)),
                pgf="fv", open_boundaries=("west",))
    base.update(kw)
    return SigmaBaroclinic(hg, H, dt=dt, n_layers=nz, **base)


def _harmonic(t: np.ndarray, x: np.ndarray, omega: float) -> tuple[float, float]:
    """Least-squares amplitude and phase [deg] of x = a + A cos(ω t − φ)."""
    M = np.column_stack([np.ones_like(t), np.cos(omega * t), np.sin(omega * t)])
    c = np.linalg.lstsq(M, x, rcond=None)[0]
    return float(np.hypot(c[1], c[2])), float(np.rad2deg(np.arctan2(c[2], c[1])))


# ---------------------------------------------------------------------------- tidal channel

def test_3d_tidal_channel():
    """
    100 km × 100 m channel, closed except the west edge, M2 η_tide = 0.5 m (ū_tide = 0: constants without
    transports).  3-D z* (4 layers, no drag, no Coriolis) vs the 2-D wetting–drying free surface driven by the same
    Flather exterior at tⁿ⁺¹ (the model's time level, as MOM6): η at the closed end within 2 % in amplitude and 2° in
    phase after the ramp.
    """
    nx, ny, dx, dt = 50, 3, 2000.0, 300.0
    hg = _channel_grid(nx, ny, dx)
    H = np.full((ny, nx), 100.0)
    bt = BoundaryTides(_constants(), ["m2"], REF, ramp_days=0.25)
    m = _channel_model(hg, H, dt, boundary_tides=bt)
    st = m.initial_state(np.zeros((ny, nx)), np.zeros((4, ny, nx)))     # T = T₀: ρ = ρ₀ (no (ρ − ρ₀) g∇η)
    fl_u = np.zeros((ny, nx + 1), bool)
    fl_u[:, 0] = True
    fs2 = WetDryFreeSurface(hg, H, dt=dt, theta=0.6, flather_u=fl_u, solver="direct")
    e2, u2, v2 = np.zeros((ny, nx)), np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    n_steps = int(2.5 * 86400 / dt)
    t, s3, s2 = [], [], []
    for n in range(n_steps):
        eu, ev, uu, vv = m._boundary_tides.flather_arrays((n + 1) * dt)
        e2, u2, v2 = fs2.step(e2, u2, v2, u2, v2, flather_eta=(eu, ev), flather_ubar=uu, flather_vbar=vv)
        st = m.step(st)
        t.append((n + 1) * dt)
        s3.append(st.eta[1, -1])
        s2.append(e2[1, -1])
    t, s3, s2 = np.array(t), np.array(s3), np.array(s2)
    late = t > t[-1] - 2 * 2 * np.pi / OMEGA["m2"]                       # last two M2 periods (ramp over at 6 h)
    a3, p3 = _harmonic(t[late], s3[late], OMEGA["m2"])
    a2, p2 = _harmonic(t[late], s2[late], OMEGA["m2"])
    print(f"closed-end M2: 3-D {a3:.5f} m {p3:.3f}°, 2-D {a2:.5f} m {p2:.3f}°, "
          f"ratio {a3 / a2:.5f}, Δφ {p3 - p2:.4f}°")
    assert a2 > 0.3                                                      # the tide entered the channel
    assert abs(a3 / a2 - 1.0) <= 0.02
    assert abs((p3 - p2 + 180.0) % 360.0 - 180.0) <= 2.0
    assert np.isfinite(st.u).all() and np.abs(st.u).max() < 1.0


# ---------------------------------------------------------------------------- body tide

def _basin_grid(nx=16, ny=3):
    hg = create_spherical_grid(nx=nx, ny=ny, dlon=0.25, dlat=0.25, lon0=-100.0, lat0=-0.375)
    return hg.assign(f_t=hg["f_t"] * 0.0, f_q=hg["f_q"] * 0.0)            # no Coriolis: the 2-D run has none


def test_tidal_potential_3d_equals_2d():
    """
    Closed homogeneous 4000 m basin (4° of longitude at the equator), only the M2 body tide (ramp ¼ day):
    the 3-D η equals the 2-D ``WetDryFreeSurface`` run whose û = uⁿ + Δt g∇η_body(tⁿ + Δt/2) to 1e-6 of max |η|.
    The linear ``ImplicitFreeSurface`` (fixed depth H; differs by O(η/H) ≈ 3e-6 relative) is checked to 1e-6 too.
    """
    from core.dynamics.free_surface import ImplicitFreeSurface
    from core.grid.vertical import compute_hfac, create_z_grid
    nx, ny, dt, Hd = 16, 3, 300.0, 4000.0
    hg = _basin_grid(nx, ny)
    H = np.full((ny, nx), Hd)
    tp = TidalPotential(["m2"], REF, ramp_days=0.25)
    m = SigmaBaroclinic(hg, H, dt=dt, n_layers=2, rho0=1000.0, t_coef=0.2, z0=None, closure="constant",
                        nu_const=1e-4, ah=0.0, theta=0.6, vertical_coordinate="ale", regrid="zstar",
                        regrid_params=dict(dz_top=np.full(2, Hd / 2)), pgf="fv", tidal_potential=tp)
    st = m.initial_state(np.zeros((ny, nx)), np.zeros((2, ny, nx)))     # T = T₀: ρ = ρ₀, no (ρ − ρ₀) g∇η
    fs2 = WetDryFreeSurface(hg, H, dt=dt, theta=0.6, solver="direct")
    vg = create_z_grid(1, [Hd])
    fsl = ImplicitFreeSurface(hg, vg, compute_hfac(vg, H), dt=dt, theta=0.6)
    e2, u2, v2 = np.zeros((ny, nx)), np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
    el, ul, vl = e2.copy(), u2.copy(), v2.copy()
    err, errl, scale = 0.0, 0.0, 0.0
    for n in range(int(1.0 * 86400 / dt)):
        fx, fy = m._tidal_potential.force(n * dt + 0.5 * dt)
        e2, u2, v2 = fs2.step(e2, u2, v2, u2 + dt * fx, v2 + dt * fy)
        el, ul, vl = fsl.step(el, ul, vl, ul + dt * fx, vl + dt * fy)
        st = m.step(st)
        err = max(err, float(np.abs(st.eta - e2).max()))
        errl = max(errl, float(np.abs(st.eta - el).max()))
        scale = max(scale, float(np.abs(e2).max()))
    print(f"body tide: max |η| {scale:.3e} m, 3-D − 2-D wet-dry {err / scale:.2e}, − linear {errl / scale:.2e}")
    assert scale > 1e-3                                                   # the basin responds (cm-scale tilt)
    assert err <= 1e-6 * scale
    assert errl <= 1e-6 * scale


# ---------------------------------------------------------------------------- restart, bit for bit

def _restart_model(hg, H, dt, ramp=1.0, beta=0.05):
    bt = BoundaryTides(_constants(U=30.0 - 5.0j, V=2.0 + 1.0j, names=("m2", "mf")), ["m2", "mf"], REF,
                       nodal_ref_date=np.datetime64("2026-07-02"), ramp_days=ramp)
    return _channel_model(hg, H, dt, nz=3, boundary_tides=bt, tidal_potential=True, sal_beta=beta, sal_flather=True,
                          sponge_cells=3)


def test_restart_during_ramp_bit_for_bit(tmp_path):
    nx, ny, dx, dt = 16, 4, 2000.0, 300.0
    hg = _channel_grid(nx, ny, dx)
    H = np.full((ny, nx), 60.0) - np.linspace(0, 20, nx)[None]
    m = _restart_model(hg, H, dt)
    st0 = m.initial_state(np.zeros((ny, nx)), np.full((3, ny, nx), 10.0))
    st = st0
    for _ in range(12):
        st = m.step(st)
    full = st
    st = st0
    m2 = _restart_model(hg, H, dt)
    st = m2.initial_state(np.zeros((ny, nx)), np.full((3, ny, nx), 10.0))
    for _ in range(6):
        st = m2.step(st)
    assert 0.0 < m2._boundary_tides.ramp(st.time) < 1.0                   # in the middle of the ramp
    write_restart(tmp_path / "r.nc", m2, st)
    m3 = _restart_model(hg, H, dt)
    st = read_restart(tmp_path / "r.nc", m3)
    for _ in range(6):
        st = m3.step(st)
    for name in ("eta", "u", "v", "T", "h"):
        np.testing.assert_array_equal(getattr(st, name), getattr(full, name), err_msg=name)
    assert np.abs(full.eta[:, 0]).max() > 1e-3                            # the tide is on at the edge
    m4 = _restart_model(hg, H, dt, ramp=2.0)                              # another ramp: refused
    with pytest.raises(ValueError, match="tides_ramp_days"):
        read_restart(tmp_path / "r.nc", m4)
    m5 = _restart_model(hg, H, dt, beta=0.04)
    with pytest.raises(ValueError, match="sal_beta"):
        read_restart(tmp_path / "r.nc", m5)


# ---------------------------------------------------------------------------- off = bit for bit

def test_tides_off_bit_for_bit():
    """
    With the Task 5 arguments at their off values (boundary_tides=None, tidal_potential=False, sal_beta=0,
    sal_flather=False) a z* channel with a 'radiating' edge and the 2-D WetDryFreeSurface with a Flather edge give,
    bit for bit, the arrays the code before Task 5 produced (commit 71e848c, tests/data/task5_base_reference.npz
    from validation/tides/make_task5_base_reference.py; regenerated 2026-11-02 at 71e848c with the PV-flux transport fix
    of core/components/momentum.py applied, docs/design/theta_bc_inertial/root_cause.md: off3d η, u, v moved by
    ≤ 4e-17).  The N1 goldens and the smoke hashes cover the rest.
    """
    from validation.tides.make_task5_base_reference import run_off_2d, run_off_case
    base = np.load(Path(__file__).parent / "data" / "task5_base_reference.npz")
    for kw in ({}, dict(boundary_tides=None, tidal_potential=False, sal_beta=0.0, sal_flather=False)):
        out = run_off_case(**kw)
        for name, a in out.items():
            np.testing.assert_array_equal(a, base[f"off3d_{name}"], err_msg=name)
    for kw in ({}, dict(sal_beta=0.0), dict(sal_beta=0.0, sal_flather=True)):
        out = run_off_2d(**kw)
        for name, a in out.items():
            np.testing.assert_array_equal(a, base[f"off2d_{name}"], err_msg=name)


# ---------------------------------------------------------------------------- scalar SAL

@pytest.mark.parametrize("flather", [False, True])
def test_sal_beta_is_reduced_gravity(flather):
    """g(1 − β)∇η everywhere in the solve: β with g equals β = 0 with g(1 − β), and also in the Flather speed
    when ``sal_flather`` (MOM6 TIDAL_SAL_FLATHER); without it the Flather speed keeps g."""
    nx, ny, dx, dt, beta = 20, 3, 1000.0, 200.0, 0.1
    hg = _channel_grid(nx, ny, dx)
    H = np.full((ny, nx), 20.0)
    fl = np.zeros((ny, nx + 1), bool)
    fl[:, 0] = True
    eta0 = 0.1 * np.cos(np.pi * (np.arange(nx) + 0.5) / nx)[None] * np.ones((ny, 1))

    def run(fs):
        e, u, v = eta0.copy(), np.zeros((ny, nx + 1)), np.zeros((ny + 1, nx))
        ext = np.zeros((ny, nx + 1))
        ext[:, 0] = 0.02
        for _ in range(40):
            e, u, v = fs.step(e, u, v, u, v, flather_eta=(ext, np.zeros((ny + 1, nx))), flather_ubar=0.0 * u,
                              flather_vbar=0.0 * v)
        return e
    g = 9.81
    a = run(WetDryFreeSurface(hg, H, dt=dt, theta=0.5, g=g, flather_u=fl, sal_beta=beta, sal_flather=flather))
    b = run(WetDryFreeSurface(hg, H, dt=dt, theta=0.5, g=g * (1 - beta), flather_u=fl))
    if flather:
        np.testing.assert_allclose(a, b, rtol=0, atol=1e-13)
    else:
        assert np.abs(a - b).max() > 1e-6                                 # Flather speed differs (√g vs √g(1−β))
    with pytest.raises(ValueError, match="sal_beta"):
        WetDryFreeSurface(hg, H, dt=dt, sal_beta=1.0)


# ---------------------------------------------------------------------------- transports, rotation

def test_transport_over_model_depth_and_rotation():
    """ū_tide = U/D with the model face depth; on a grid with cos_t/sin_t the (east, north) transports are
    rotated to grid axes with forcing.rotation.to_grid (u_x = cos α U + sin α V, v_y = −sin α U + cos α V)."""
    nx, ny, dx = 6, 5, 2000.0
    hg = _channel_grid(nx, ny, dx)
    U, V = 40.0 - 10.0j, 15.0 + 5.0j
    c = _constants(U=U, V=V)
    Hu, Hv = np.full((ny, nx + 1), 80.0), np.full((ny + 1, nx), 50.0)
    t = 12345.0
    plain = BoundaryTides(c, ["m2"], REF).bind(hg, ("west", "south"), Hu, Hv)
    alpha = np.deg2rad(30.0)
    rot_grid = hg.assign(cos_t=(hg["x_t"].dims, np.full((ny, nx), np.cos(alpha))),
                         sin_t=(hg["x_t"].dims, np.full((ny, nx), np.sin(alpha))))
    rot = BoundaryTides(c, ["m2"], REF).bind(rot_grid, ("west", "south"), Hu, Hv)
    mx = BoundaryTides(_constants(amp=1.0, phase_deg=0.0, U=1.0 + 0j, V=0j), ["m2"], REF)
    unit = mx.bind(hg, ("west",), Hu, Hv).at(t)["west"][1][0] * 80.0      # cos-wave of a unit transport
    unit_q = BoundaryTides(_constants(amp=1.0, phase_deg=0.0, U=1j, V=0j), ["m2"], REF).bind(
        hg, ("west",), Hu, Hv).at(t)["west"][1][0] * 80.0                 # the quadrature part (Im)

    def pred(q):                                                          # prediction of a complex transport
        return q.real * unit + q.imag * unit_q
    a = plain.at(t)
    np.testing.assert_allclose(a["west"][1], pred(U) / 80.0, rtol=1e-12)
    np.testing.assert_allclose(a["south"][1], pred(V) / 50.0, rtol=1e-12)
    b = rot.at(t)
    np.testing.assert_allclose(b["west"][1], pred(np.cos(alpha) * U + np.sin(alpha) * V) / 80.0, rtol=1e-12)
    np.testing.assert_allclose(b["south"][1], pred(-np.sin(alpha) * U + np.cos(alpha) * V) / 50.0, rtol=1e-12)
    np.testing.assert_allclose(b["west"][0], a["west"][0], rtol=0, atol=0)   # η is a scalar: not rotated


def test_velocity_depth_tpxo_uses_the_atlas_depth():
    """velocity_depth='tpxo': ū_tide = U/h_TPXO (the atlas's own velocity, independent of the model face depth);
    the default 'model' (U/D_model) is unchanged and needs no depths; 'tpxo' without hu/hv is refused."""
    nx, ny, dx = 6, 5, 2000.0
    hg = _channel_grid(nx, ny, dx)
    U, V = 40.0 - 10.0j, 15.0 + 5.0j
    c = _constants(U=U, V=V)
    ct = c.assign(hu=(("lat", "lon_u"), np.full((c.lat.size, c.lon_u.size), 25.0)),
                  hv=(("lat_v", "lon"), np.full((c.lat_v.size, c.lon.size), 20.0)))
    Hu, Hv = np.full((ny, nx + 1), 80.0), np.full((ny + 1, nx), 50.0)
    t = 4321.0
    model = BoundaryTides(c, ["m2"], REF).bind(hg, ("west", "south"), Hu, Hv).at(t)
    model_t = BoundaryTides(ct, ["m2"], REF).bind(hg, ("west", "south"), Hu, Hv).at(t)
    tpxo = BoundaryTides(ct, ["m2"], REF, velocity_depth="tpxo").bind(hg, ("west", "south"), Hu, Hv).at(t)
    for e in ("west", "south"):
        np.testing.assert_array_equal(model_t[e][1], model[e][1])           # extra depths change nothing
    np.testing.assert_allclose(tpxo["west"][1], model["west"][1] * 80.0 / 25.0, rtol=1e-12)
    np.testing.assert_allclose(tpxo["south"][1], model["south"][1] * 50.0 / 20.0, rtol=1e-12)
    np.testing.assert_array_equal(tpxo["west"][0], model["west"][0])
    assert BoundaryTides(c, ["m2"], REF, velocity_depth="tpxo").issues()
    assert "velocity_depth" not in BoundaryTides(c, ["m2"], REF).config()
    assert BoundaryTides(ct, ["m2"], REF, velocity_depth="tpxo").config()["velocity_depth"] == "tpxo"


def test_ramp_and_nodal_reference():
    c = _constants(names=("m2", "mf", "mm"))
    bt = BoundaryTides(c, ["m2", "mf", "mm"], REF, ramp_days=2.0)
    assert bt.ramp(0.0) == 0.0 and bt.ramp(86400.0) == 0.5 and bt.ramp(3 * 86400.0) == 1.0
    assert BoundaryTides(c, ["m2"], REF).ramp(-5.0) == 1.0
    Hu, Hv = np.full((3, 5), 50.0), np.full((4, 4), 50.0)
    hg = _channel_grid(4, 3, 2000.0)
    a = BoundaryTides(c, ["mf"], REF).bind(hg, ("west",), Hu, Hv)
    b = BoundaryTides(c, ["mf"], REF, nodal_ref_date=np.datetime64("1993-07-02")).bind(hg, ("west",), Hu, Hv)
    assert not np.allclose(a.at(1e6)["west"][0], b.at(1e6)["west"][0])  # fixed f, u differ from f(t), u(t)
    tp = TidalPotential.from_boundary_tides(BoundaryTides(c, ["m2", "mf", "mm"], REF, ramp_days=2.0))
    assert tp.constituents == ["m2", "mf", "mm"] and tp.ramp_days == 2.0


# ---------------------------------------------------------------------------- pre-flight

def test_preflight_refuses_tides_without_radiating_edge():
    hg = _channel_grid(6, 3, 2000.0)
    H = np.full((3, 6), 30.0)
    bt = BoundaryTides(_constants(), ["m2", "xx"], REF, ramp_days=-1.0)
    with pytest.raises(ConfigurationError) as e:                          # every error at once
        SigmaBaroclinic(hg, H, dt=60.0, n_layers=2, z0=None, closure="constant", boundary_tides=bt, sal_beta=1.5)
    msg = str(e.value)
    assert "'radiating'" in msg and "sal_beta" in msg and "ramp_days" in msg and "'xx'" in msg
    with pytest.raises(ConfigurationError, match="'radiating'"):
        SigmaBaroclinic(hg, H, dt=60.0, n_layers=2, z0=None, closure="constant",
                        boundary_tides=BoundaryTides(_constants(), ["m2"], REF), open_boundaries=("west",),
                        open_boundary_mode={"west": "eta"},
                        open_boundary_state=lambda t: {})
    with pytest.raises(ConfigurationError, match="tidal_potential=True"):
        SigmaBaroclinic(hg, H, dt=60.0, n_layers=2, z0=None, closure="constant", tidal_potential=True)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        SigmaBaroclinic(hg, H, dt=60.0, n_layers=2, z0=None, closure="constant", sal_flather=True)
    assert any("sal_flather" in str(x.message) for x in w)


# ---------------------------------------------------------------------------- incoming progressive wave (review I1)

def _wave_constants(axis: str, direction: int, amp: float, H: float, rotated: bool = False, flip: bool = False,
                    drop: bool = False):
    """
    TPXO-like constants of a progressive long wave (f = 0) travelling along ``axis`` ('lon' or 'lat') in
    ``direction`` ±1: h = a e^{−i(φ0 + k s)}, transport U = H ū = ±√(gH) h along the wave (east U for 'lon', north V for
    'lat'; with ``rotated`` the 'lon' wave's transport is given as north V, for a grid whose x axis points north).
    ``flip`` reverses the transport sign (a wave that would leave instead of enter); ``drop`` sets it to zero.
    """
    g = 9.81
    c = np.sqrt(g * H)
    k = OMEGA["m2"] / c                                                 # [rad m⁻¹]
    lat = np.arange(29.5, 32.01, 0.02)
    lon = np.arange(-70.5, -67.49, 0.02)
    LA, LO = np.meshgrid(lat, lon, indexing="ij")
    s = ((LO + 70.0) if axis == "lon" else (LA - 30.0)) * DEG           # distance along the wave [m]
    h = amp * np.exp(-1j * (np.deg2rad(30.0) + direction * k * s))
    q = (0.0 if drop else -1.0 if flip else 1.0) * direction * c * h                     # U = H ū = H √(g/H) η = √(gH) η [m² s⁻¹]
    zero = np.zeros_like(h)
    along_x = axis == "lon" and not rotated
    U, V = (q, zero) if along_x else (zero, q)
    ds = xr.Dataset({"h": (("constituent", "lat", "lon"), h[None])},
                    coords={"constituent": ["m2"], "lat": lat, "lon": lon})
    ds["U"] = xr.DataArray(U[None], dims=("constituent", "lat", "lon_u"), coords={"lon_u": lon})
    ds["V"] = xr.DataArray(V[None], dims=("constituent", "lat_v", "lon"), coords={"lat_v": lat})
    return ds


def _run_wave(axis: str, direction: int, rotated: bool = False, flip: bool = False, drop: bool = False):
    """
    200 km × 100 m z* channel along x ('lon') or y ('lat'), both ends 'radiating' with the boundary tide of a
    progressive M2 wave of 0.2 m (no Coriolis, no drag), 2 days; returns the fitted M2 amplitude and phase at
    every cell along the channel centre line (last M2 period).
    """
    n, dx, H, dt, amp = 50, 4000.0, 100.0, 300.0, 0.2
    nx, ny = (n, 3) if axis == "lon" else (3, n)
    hg = _channel_grid(nx, ny, dx)
    if rotated:                                                        # grid x points north: α = 90°
        hg = hg.assign(cos_t=(hg["x_t"].dims, np.zeros((ny, nx))), sin_t=(hg["x_t"].dims, np.ones((ny, nx))))
    edges = ("west", "east") if axis == "lon" else ("south", "north")
    bt = BoundaryTides(_wave_constants(axis, direction, amp, H, rotated, flip, drop), ["m2"], REF, ramp_days=0.25)
    m = _channel_model(hg, np.full((ny, nx), H), dt, nz=2, open_boundaries=edges, boundary_tides=bt)
    st = m.initial_state(np.zeros((ny, nx)), np.zeros((2, ny, nx)))
    t, rec = [], []
    for k in range(int(2.0 * 86400 / dt)):
        st = m.step(st)
        t.append((k + 1) * dt)
        rec.append(st.eta[1, :].copy() if axis == "lon" else st.eta[:, 1].copy())
    t, rec = np.array(t), np.array(rec)
    late = t > t[-1] - 2 * np.pi / OMEGA["m2"]
    fits = [_harmonic(t[late], rec[late, i], OMEGA["m2"]) for i in range(n)]
    from forcing.tides import nodal_corrections
    fm2 = nodal_corrections(np.array([bt.mjd(t[-1])]), ["m2"])[0][0, 0]   # nodal factor of M2 at the run date
    return np.array([f[0] for f in fits]), np.array([f[1] for f in fits]), fm2 * amp


@pytest.mark.parametrize("axis,direction,rotated", [("lon", 1, False),   # enters at the west edge
                                                    ("lon", -1, False),  # enters at the east edge
                                                    ("lat", -1, False),  # enters at the north edge
                                                    ("lon", 1, True)])   # east/north constants, grid rotated 90°
def test_incoming_progressive_wave(axis, direction, rotated):
    """
    A progressive long wave given as TPXO-like η and transport (U = √(gH) η along the wave) enters through the
    tidal Flather edge with its own amplitude and leaves through the other one: the interior M2 amplitude is the
    forced f_M2 · 0.2 m (f_M2 the nodal factor, ≈ 0.967 in 2026) within 3 % everywhere and the phase propagates in
    the wave's direction.  Wrong conventions must fail: with the transport sign reversed both tidal edges send the
    wave the other way (same amplitude, phase propagating backwards); with the transport dropped (ū_tide = 0) each
    edge injects half a wave from each side and the amplitude along the channel is far from the forced one.  Checks
    the sign of ū_tide on west, east, south/north edges and the rotation of (east, north) transports to grid axes
    (``forcing.rotation.to_grid``).
    """
    A, P, amp = _run_wave(axis, direction, rotated)
    ratio = A / amp
    print(f"{axis} dir {direction} rot {rotated}: amplitude/forced {ratio.min():.4f}–{ratio.max():.4f}")
    assert np.all(np.abs(ratio - 1.0) <= 0.03)
    dphi = np.diff(np.unwrap(np.deg2rad(P)))                           # phase lag grows along the wave
    assert np.all(direction * dphi > 0)
    _, Pf, _ = _run_wave(axis, direction, rotated, flip=True)
    assert np.all(direction * np.diff(np.unwrap(np.deg2rad(Pf))) < 0)  # the reversed wave
    Ad, _, _ = _run_wave(axis, direction, rotated, drop=True)
    print(f"  ū_tide dropped: amplitude/forced {Ad.min() / amp:.4f}–{Ad.max() / amp:.4f}")
    assert np.max(np.abs(Ad / amp - 1.0)) > 0.1
