"""
Tests of the Jackson, Hallberg & Legg (2008) shear-mixing column oracle
(``physics/mixing/kappa_shear.py``, port of MOM6 ``MOM_kappa_shear.F90``).

The columns are in the model orientation: layers k = 0 (bed) … N−1 (surface),
interfaces 0 (bed) … N (surface).  b = −g(ρ − ρ₀)/ρ₀ [m s⁻²].
"""

from __future__ import annotations

import numpy as np

from physics.mixing.kappa_shear import JacksonParams, kappa_shear_column

KAPPA0 = 1.0e-7  # background κ0 = KD_KAPPA_SHEAR_0 default floor [m² s⁻¹]


def _column(n: int = 100, depth: float = 200.0):
    """Uniform layers; returns dz [m] and layer-centre heights above the bed [m]."""
    dz = np.full(n, depth / n)
    zc = np.cumsum(dz) - 0.5 * dz
    return dz, zc


def _shear_layer(ri_min: float = 0.1, n0: float = 0.01, d: float = 5.0):
    """
    tanh shear layer centred at mid-depth with uniform stratification N0².

    u = (ΔU/2) tanh((z − z_c)/d), S_max = ΔU/(2d) at the centre, and
    Ri_min = N0²/S_max² = ``ri_min``.  Away from the layer S → 0 and Ri → ∞.
    """
    dz, zc = _column()
    z_mid = 0.5 * dz.sum()
    s_max = n0 / np.sqrt(ri_min)
    u = s_max * d * np.tanh((zc - z_mid) / d)
    v = np.zeros_like(u)
    b = n0**2 * zc  # N² = ∂b/∂z = N0²
    return u, v, b, dz, zc, z_mid


def test_supercritical_column_gives_zero() -> None:
    # N² = 1e-4 s⁻², S² = 2.5e-5 s⁻²: Ri = 4 ≥ Ri_c everywhere → no source.
    dz, zc = _column(40, 200.0)
    u = 0.005 * zc
    v = np.zeros_like(u)
    b = 1.0e-4 * zc
    kappa, q = kappa_shear_column(u, v, b, dz, f2=1.0e-8, dt=600.0, kappa0=KAPPA0,
                                  p=JacksonParams())
    assert kappa.shape == (41,) and q.shape == (41,)
    assert np.all(kappa == 0.0)
    assert np.isfinite(q).all()


def test_shear_layer_mixes_inside_only() -> None:
    u, v, b, dz, zc, z_mid = _shear_layer()
    kappa, q = kappa_shear_column(u, v, b, dz, f2=1.0e-8, dt=600.0, kappa0=KAPPA0,
                                  p=JacksonParams())
    zi = np.concatenate([[0.0], np.cumsum(dz)])
    centre = np.argmin(np.abs(zi - z_mid))
    assert kappa[centre] > 0.0
    assert kappa[0] == 0.0 and kappa[-1] == 0.0
    # Ri > 0.25 well away from the layer (|z − z_c| ≥ 40 m = 8 d): no mixing.
    far = np.abs(zi - z_mid) >= 40.0
    assert np.all(kappa[far] == 0.0)
    assert np.isfinite(kappa).all() and np.all(kappa >= 0.0)
    assert np.isfinite(q).all() and np.all(q > 0.0)


def test_unstable_column_is_finite() -> None:
    # Shear layer plus two interfaces with b decreasing upward (N² < 0 is
    # clipped to 0: Ri = 0 there, the strongest source of the scheme).
    u, v, b, dz, zc, z_mid = _shear_layer()
    k = int(np.argmin(np.abs(zc - z_mid)))
    b = b.copy()
    b[k + 1] = b[k] - 1.0e-3
    b[k + 2] = b[k + 1] - 1.0e-3
    kappa, q = kappa_shear_column(u, v, b, dz, f2=1.0e-8, dt=600.0, kappa0=KAPPA0,
                                  p=JacksonParams())
    assert np.isfinite(kappa).all() and np.isfinite(q).all()
    assert kappa.max() > 0.0
    assert kappa.max() < 10.0


def test_large_dt_terminates() -> None:
    # Hallberg (2000) effect: mixing erodes the shear inside the step, so the
    # Δt-averaged κ of a long step is not larger than that of a short one.
    u, v, b, dz, zc, z_mid = _shear_layer()
    k600, _ = kappa_shear_column(u, v, b, dz, f2=1.0e-8, dt=600.0, kappa0=KAPPA0,
                                 p=JacksonParams())
    k3600, q3600 = kappa_shear_column(u, v, b, dz, f2=1.0e-8, dt=3600.0, kappa0=KAPPA0,
                                      p=JacksonParams())
    assert np.isfinite(k3600).all() and np.isfinite(q3600).all()
    assert k3600.max() > 0.0
    assert k3600.max() <= 2.0 * k600.max()


def test_defaults_are_jhl_set_1() -> None:
    # Jackson et al. (2008) best-fit set 1 as MOM6's defaults
    # (MOM_kappa_shear.F90:2106-2112, 2159-2276).
    p = JacksonParams()
    assert (p.ri_crit, p.shearmix_rate, p.fri_curvature, p.kappa_buoy_scale,
            p.tke_n_decay, p.tke_shear_decay) == (0.25, 0.089, -0.97, 0.82, 0.24, 0.14)
    assert p.kd_seed == 1.0
    assert p.kd_trunc_frac == 0.01
    assert p.tol_err == 0.1
    assert p.max_rino_it == 50
    assert p.max_ks_it == 13
    assert p.max_kap_src_chg == 10.0
    assert p.tke_background == 0.0
    assert p.lz_rescale == 1.0
