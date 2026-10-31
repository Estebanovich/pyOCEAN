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


# ---------------------------------------------------------------- Numba kernel with thin-layer merging (task 3)
def _random_columns(m: int, n: int = 24, seed: int = 3):
    """
    ``m`` varied columns (N, M), k = 0 at the bed: stretched 0.5–4 m layers with a tanh shear layer (random
    position, width, Ri_min 0.03–0.3, a turning v) and N0², plus thin 1 mm–2 cm layers spliced in at random places
    (bed, interior, a surface cluster), some columns dry (all layers 1e-5 m) or with one layer of mass.  Returns
    u, v, b, dz (N, M), f2 (M,).
    """
    rng = np.random.default_rng(seed)
    u = np.zeros((n, m))
    v = np.zeros((n, m))
    b = np.zeros((n, m))
    dz = np.zeros((n, m))
    for c in range(m):
        h = rng.uniform(0.5, 4.0, n)
        kind = c % 8
        if kind in (1, 2, 3, 5):                       # thin layers anywhere (bed, interior)
            idx = rng.choice(n, size=rng.integers(1, 7), replace=False)
            h[idx] = rng.uniform(1e-3, 2e-2, idx.size)
        if kind in (2, 6):                             # a thin cluster at the surface (≥ 1 mm in total)
            h[-3:] = rng.uniform(1e-3, 2e-2, 3)
        if kind == 4:                                  # massless layers at the bed and the surface
            h[:2] = 1e-12
            h[-2:] = 1e-12
        if kind == 7 and c % 16 == 7:                  # dry column
            h[:] = 1e-5
        elif kind == 7:                                # one layer with mass
            h[:] = 1e-5
            h[n // 2] = 30.0
        zc = np.cumsum(h) - 0.5 * h
        d = h.sum()
        n0 = 10.0 ** rng.uniform(-2.5, -1.8)
        width = rng.uniform(3.0, 0.15 * d + 3.0)
        s_max = n0 / np.sqrt(rng.uniform(0.03, 0.3))
        zm = rng.uniform(0.2, 0.8) * d
        u[:, c] = s_max * width * np.tanh((zc - zm) / width)
        v[:, c] = 0.3 * s_max * width * np.tanh((zc - zm) / (2.0 * width)) + rng.normal(0.0, 1e-3, n)
        b[:, c] = n0**2 * zc + rng.normal(0.0, 1e-7, n)
        dz[:, c] = h
    f2 = rng.uniform(0.0, 1.0e-8, m)
    return u, v, b, dz, f2


def test_numba_matches_numpy_oracle() -> None:
    from core.components.closure import thin_layer_threshold
    from physics.mixing.kappa_shear import kappa_shear_columns, kappa_shear_columns_numpy

    u, v, b, dz, f2 = _random_columns(40)
    h_thin = thin_layer_threshold(dz)
    p = JacksonParams()
    k_nb, q_nb = kappa_shear_columns(u, v, b, dz, f2, 600.0, KAPPA0, h_thin, p)
    k_np, q_np = kappa_shear_columns_numpy(u, v, b, dz, f2, 600.0, KAPPA0, h_thin, p)
    assert k_nb.shape == (25, 40) and q_nb.shape == (25, 40)
    assert np.array_equal(k_nb, k_np)
    assert np.array_equal(q_nb, q_np)
    assert (k_np.max(axis=0) > 0.0).sum() >= 20       # most columns do mix
    assert np.isfinite(k_np).all() and np.isfinite(q_np).all() and (k_np >= 0.0).all()


def test_threads_bit_for_bit() -> None:
    import numba

    from core.components.closure import thin_layer_threshold
    from physics.mixing.kappa_shear import kappa_shear_columns

    u, v, b, dz, f2 = _random_columns(120, seed=11)
    h_thin = thin_layer_threshold(dz)
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        k1, q1 = kappa_shear_columns(u, v, b, dz, f2, 600.0, KAPPA0, h_thin, JacksonParams())
        numba.set_num_threads(min(5, numba.config.NUMBA_NUM_THREADS))
        k5, q5 = kappa_shear_columns(u, v, b, dz, f2, 600.0, KAPPA0, h_thin, JacksonParams())
    finally:
        numba.set_num_threads(old)
    assert np.array_equal(k1, k5) and np.array_equal(q1, q5)


def test_thin_layers_do_not_change_kappa() -> None:
    """
    Criterion 2 of the spec: thin layers spliced into a column leave κ unchanged.

    The thin layers are cut from the **bottom** of existing layers with the same u, v, b, so the merged column is
    the original one (to round-off: Σ h u / Σ h) and every original interface is an interface of the merged column
    (a thin layer joins the layer above it, MOM_kappa_shear.F90:256-265).  The comparison is therefore at the common
    interfaces themselves; a thin layer cut from the top of a layer would join the next layer up and move that
    interface by up to 2 cm, a real change of the column rather than of the merge.
    """
    from core.components.closure import thin_layer_threshold
    from physics.mixing.kappa_shear import kappa_shear_columns

    u, v, b, dz, zc, z_mid = _shear_layer(ri_min=0.15)
    n = dz.size
    cut = {0: 1e-3, 7: 5e-3, 31: 1.5e-2, 48: 1e-2, 49: 1.5e-3, 50: 1.5e-2, 63: 8e-3, 90: 1e-3}
    dz2, u2, v2, b2, common = [], [], [], [], [0]
    for k in range(n):
        if k in cut:
            for arr, x in ((dz2, cut[k]), (u2, u[k]), (v2, v[k]), (b2, b[k])):
                arr.append(x)
            for arr, x in ((dz2, dz[k] - cut[k]), (u2, u[k]), (v2, v[k]), (b2, b[k])):
                arr.append(x)
        else:
            for arr, x in ((dz2, dz[k]), (u2, u[k]), (v2, v[k]), (b2, b[k])):
                arr.append(x)
        common.append(len(dz2))
    cols = [np.array(x)[:, None] for x in (u2, v2, b2, dz2)]
    h_thin = thin_layer_threshold(cols[3])
    assert np.all(np.array(list(cut.values())) < h_thin[0])
    k_thin, q_thin = kappa_shear_columns(*cols, np.array([1e-8]), 600.0, KAPPA0, h_thin, JacksonParams())
    k_ref, q_ref = kappa_shear_column(u, v, b, dz, f2=1.0e-8, dt=600.0, kappa0=KAPPA0, p=JacksonParams())
    assert k_ref.max() > 1e-3
    np.testing.assert_allclose(k_thin[common, 0], k_ref, rtol=1e-6, atol=1e-6 * k_ref.max())
    np.testing.assert_allclose(q_thin[common, 0], q_ref, rtol=1e-6, atol=1e-6 * q_ref.max())


def test_dry_and_single_layer_columns() -> None:
    from core.components.closure import thin_layer_threshold
    from physics.mixing.kappa_shear import kappa_shear_columns, kappa_shear_columns_numpy

    n = 20
    zc = np.arange(n) + 0.5
    u = np.tile((0.05 * zc)[:, None], (1, 3))       # Ri = 1e-4 / 2.5e-3 = 0.04: would mix if the layers had mass
    v = np.zeros_like(u)
    b = np.tile((1e-4 * zc)[:, None], (1, 3))
    dz = np.full((n, 3), 1e-5)                       # column 0 dry (Σ dz = 2e-4 m ≤ 1 mm)
    dz[7, 1] = 50.0                                  # column 1: one layer with mass (the thin top stack < 1 mm)
    dz[:, 2] = 0.0                                   # column 2: zero thickness
    h_thin = thin_layer_threshold(dz)
    f2 = np.full(3, 1e-8)
    for fn in (kappa_shear_columns, kappa_shear_columns_numpy):
        k, q = fn(u, v, b, dz, f2, 600.0, KAPPA0, h_thin, JacksonParams())
        assert k.shape == (n + 1, 3) and q.shape == (n + 1, 3)
        assert np.all(k == 0.0) and np.all(q == 0.0)
