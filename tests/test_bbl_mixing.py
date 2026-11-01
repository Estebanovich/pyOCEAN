"""
Tests of the drag-driven bottom-boundary-layer mixing column oracle
(``physics/mixing/bbl_mixing.py``, port of MOM6 ``add_LOTW_BBL_diffusivity``,
MOM_set_diffusivity.F90:1606-1789).

Arrays follow pyOCEAN: layers k = 0 (bed) … N−1, interfaces 0 (bed) … N.
"""

from __future__ import annotations

import numpy as np

from core.grid.horizontal import OMEGA
import numba

from core.components.closure import thin_layer_threshold
from physics.mixing.bbl_mixing import (BBLParams, lotw_bbl_column, lotw_bbl_columns,
                                       lotw_bbl_columns_numpy, n2_bottom_bulk)


def _column(nz: int = 50, dz: float = 20.0, n2: float = 1.0e-6) -> tuple[np.ndarray, np.ndarray]:
    """Uniform column: nz layers of thickness dz [m], uniform N² [s⁻²]."""
    return np.full(nz, dz), np.full(nz + 1, n2)


def test_no_drag_gives_zero() -> None:
    """ustar = 0 and tke = 0 (no bed stress) → no BBL diffusivity anywhere."""
    dz, n2 = _column()
    kd = lotw_bbl_column(dz, n2, 0.0, 1.0e-4, 0.0, BBLParams())
    assert kd.shape == (dz.size + 1,)
    assert np.all(kd == 0.0)


def test_energy_bound() -> None:
    """The mixing work Σ Kd dz_int max(N², Ω²) never exceeds the TKE supplied."""
    dz, n2 = _column(n2=1.0e-6)
    tke = 5.0e-6
    kd = lotw_bbl_column(dz, n2, 0.01, 1.0e-4, tke, BBLParams())
    assert kd[0] == 0.0 and kd[-1] == 0.0
    assert np.all(kd >= 0.0)
    dz_int = np.zeros_like(kd)
    dz_int[1:-1] = 0.5 * (dz[:-1] + dz[1:])
    work = np.sum(kd * dz_int * np.maximum(n2, OMEGA * OMEGA))
    assert work > 0.0
    assert work <= tke * (1.0 + 1.0e-12)


def test_decays_with_height() -> None:
    """Kd near the bed is larger than Kd far above the scale u*/|f| (= 100 m here)."""
    dz, n2 = _column(nz=50, dz=20.0, n2=1.0e-6)
    ustar, absf = 0.01, 1.0e-4
    kd = lotw_bbl_column(dz, n2, ustar, absf, 5.0e-6, BBLParams())
    z_int = np.concatenate(([0.0], np.cumsum(dz)))
    far = (z_int > 3.0 * ustar / absf) & (z_int < z_int[-1])
    assert kd[1] > 0.0
    assert np.all(kd[1] > kd[far])
    assert np.all(kd[1:3].min() > kd[far].max())


def test_unstable_is_finite() -> None:
    """N² < 0 at two interfaces: N² is floored by Ω², Kd finite and ≤ kd_max.

    KD_MAX is not a cap of the law-of-the-wall Kd in MOM6 (it is used only where
    the mixing work is ≤ 0, SD:1769-1772); Kd_wall itself tends to κ u*²/|f|.
    u* = 5 mm/s keeps κ u*²/|f| ≈ 0.10 m² s⁻¹ and Kd_wall in this 100 m column
    ≤ 0.035 m² s⁻¹, so the bound below tests the unstable interfaces, not a cap.
    """
    dz, n2 = _column(nz=20, dz=5.0, n2=1.0e-6)
    n2[2] = -1.0e-5
    n2[3] = -3.0e-6
    p = BBLParams()
    kd = lotw_bbl_column(dz, n2, 0.005, 1.0e-4, 1.0e-4, p)
    assert np.all(np.isfinite(kd))
    assert np.all(kd >= 0.0)
    assert np.all(kd <= p.kd_max)
    assert kd[2] > 0.0 and kd[3] > 0.0
    # The Ω² floor makes the work > 0, so the unstable interfaces take the energy-bounded
    # law-of-the-wall Kd, not the KD_MAX branch (task-1 review: pins the floor).
    assert kd[2] < p.kd_max and kd[3] < p.kd_max
    # Without the Ω² floor, N² < 0 gives no TKE cost: MOM6 then sets Kd = KD_MAX.
    p0 = BBLParams(use_omega=False)
    kd0 = lotw_bbl_column(dz, n2, 0.005, 1.0e-4, 1.0e-4, p0)
    assert np.all(np.isfinite(kd0))
    assert kd0[2] == p0.kd_max and kd0[3] == p0.kd_max


def test_f_zero_uses_max_decay() -> None:
    """absf = 0: the decay scale is BBL_MIXING_MAX_DECAY (200 m), as for absf = u*/200.

    The TKE is chosen so small that the first interface takes all of it:
    Kd_1 = exp(−Idecay dz_0) tke / (dz_int N²), independent of the law of the wall,
    so it isolates the decay.
    """
    dz, n2 = _column(nz=30, dz=10.0, n2=1.0e-5)
    ustar, tke = 0.01, 1.0e-9
    p = BBLParams()
    kd_f0 = lotw_bbl_column(dz, n2, ustar, 0.0, tke, p)
    kd_fm = lotw_bbl_column(dz, n2, ustar, ustar / p.max_decay, tke, p)
    assert np.all(np.isfinite(kd_f0))
    expected = np.exp(-dz[0] / p.max_decay) * tke / (0.5 * (dz[0] + dz[1]) * n2[1])
    np.testing.assert_allclose(kd_f0[1], expected, rtol=1e-13)
    np.testing.assert_allclose(kd_f0, kd_fm, rtol=1e-13, atol=0.0)
    assert np.all(kd_f0[2:] == 0.0)


def test_defaults_are_om4() -> None:
    """BBLParams() = MOM6 / OM4 values (MOM_set_diffusivity.F90:2417-2524; OM4_025 MOM_input)."""
    p = BBLParams()
    assert p.bbl_effic == 0.2
    assert p.max_decay == 200.0
    assert p.von_karman == 0.41
    assert p.use_omega is True
    assert p.kd_max == 0.1


# ------------------------------------------------------------------ task 3: columns kernel with thin-layer merge
def _varied_columns(m: int = 40, n: int = 25, seed: int = 3) -> tuple[np.ndarray, ...]:
    """M varied columns (N, M): thick layers, 1 mm–2 cm thin layers at the bed, interior and top, dry columns,
    density inversions (N² < 0), u* = 0 and f = 0 columns."""
    rng = np.random.default_rng(seed)
    dz = rng.uniform(2.0, 60.0, (n, m))
    thin = rng.random((n, m)) < 0.25
    dz[thin] = rng.uniform(1.0e-3, 2.0e-2, thin.sum())
    dz[:3, 1] = 1.0e-3                       # thin bed stack
    dz[-4:, 2] = rng.uniform(1e-3, 2e-2, 4)  # thin top stack
    dz[:, 3] = 0.0                           # dry column
    dz[:, 4] = 1.0e-5                        # Σdz = 2.5e-4 m ≤ H_MASSLESS: dry
    dz[:-1, 5] = 1.0e-3                      # one thick layer only
    # layer densities: stable steps with 15 % inversions (N² < 0 interfaces, also across merged layers)
    step = rng.uniform(1.0e-4, 0.5, (n, m))
    neg = rng.random((n, m)) < 0.15
    step[neg] = -rng.uniform(1.0e-3, 0.5, neg.sum())
    rho = 1030.0 - np.cumsum(step, axis=0)    # k = 0 at the bed
    ustar = rng.uniform(1.0e-3, 3.0e-2, m)
    ustar[6] = 0.0
    absf = rng.uniform(0.0, 1.4e-4, m)
    absf[7] = 0.0
    tke = rng.uniform(1.0e-7, 1.0e-4, m)
    tke[8] = 0.0
    return dz, rho, ustar, absf, tke, thin_layer_threshold(dz)


def test_numba_matches_numpy_oracle() -> None:
    """Numba columns kernel = NumPy oracle (merge + ``lotw_bbl_column``) bit for bit, both N² floors."""
    dz, rho, ustar, absf, tke, h_thin = _varied_columns()
    for p in (BBLParams(), BBLParams(use_omega=False)):
        ref = lotw_bbl_columns_numpy(dz, rho, ustar, absf, tke, h_thin, p, rho0=1025.0, g=9.8)
        out = lotw_bbl_columns(dz, rho, ustar, absf, tke, h_thin, p, rho0=1025.0, g=9.8)
        assert out.shape == (dz.shape[0] + 1, dz.shape[1])
        assert np.all(np.isfinite(out))
        assert np.array_equal(out, ref)
        assert np.any(ref > 0.0)


def test_threads_bit_for_bit() -> None:
    """1 and 5 threads give the same Kd bit for bit (one column per task, no reduction across columns)."""
    dz, rho, ustar, absf, tke, h_thin = _varied_columns(m=200, seed=11)
    old = numba.get_num_threads()
    try:
        numba.set_num_threads(1)
        a = lotw_bbl_columns(dz, rho, ustar, absf, tke, h_thin, BBLParams())
        numba.set_num_threads(min(5, numba.config.NUMBA_NUM_THREADS))
        b = lotw_bbl_columns(dz, rho, ustar, absf, tke, h_thin, BBLParams())
    finally:
        numba.set_num_threads(old)
    assert np.array_equal(a, b)


def test_thin_bed_layers_do_not_change_kd() -> None:
    """1 mm layers split off the bottom of the bed layer and of interior layers (density of their parent) are
    merged into the thick layer above them: Kd at the interfaces of the original column is unchanged (rtol 1e-12),
    and no interface takes Kd = kd_max because of a vanishing layer (the N² = 0 across a 1 mm layer of the parent's
    density is not seen: N² comes from the merged densities, final review I1)."""
    nz = 20
    dz0 = np.full(nz, 10.0)
    rho0 = 1000.0 + 2.0e-3 * np.arange(nz)[::-1]       # N² = 1.962e-6 s⁻²
    split = (0, 5, 11)                       # layers whose bottom 1 mm becomes its own layer
    dz1, rho1, common = [], [], []
    for k in range(nz):
        common.append(len(dz1))              # interface at the bottom of original layer k
        if k in split:
            dz1 += [1.0e-3, dz0[k] - 1.0e-3]
            rho1 += [rho0[k], rho0[k]]       # the new interface: N² = 0, must not survive
        else:
            dz1.append(dz0[k])
            rho1.append(rho0[k])
    common.append(len(dz1))
    dz1, rho1 = np.array(dz1), np.array(rho1)
    for p in (BBLParams(), BBLParams(use_omega=False)):
        args = (0.01, 1.0e-4, 2.0e-5)
        k0 = lotw_bbl_columns(dz0[:, None], rho0[:, None], *(np.array([a]) for a in args),
                              thin_layer_threshold(dz0[:, None]), p)[:, 0]
        k1 = lotw_bbl_columns(dz1[:, None], rho1[:, None], *(np.array([a]) for a in args),
                              thin_layer_threshold(dz1[:, None]), p)[:, 0]
        assert np.any(k0 > 0.0)
        np.testing.assert_allclose(k1[common], k0, rtol=1e-12, atol=0.0)
        assert not np.any(k1 == p.kd_max)
        # without merging, the N² = 0 interfaces of the 1 mm layers would take kd_max (use_omega=False)
        if not p.use_omega:
            n21 = np.zeros(dz1.size + 1)
            n21[1:-1] = (9.81 / 1000.0) * (rho1[:-1] - rho1[1:]) / (0.5 * (dz1[:-1] + dz1[1:]))
            raw = lotw_bbl_column(dz1, n21, *args, p)
            assert np.any(raw == p.kd_max)


def test_dry_column_zero() -> None:
    """Σdz = 0 → Kd = 0 at every interface (oracle and kernel)."""
    dz = np.zeros((10, 2))
    dz[:, 1] = 5.0
    rho = 1025.0 - 5.0e-3 * np.arange(10)[:, None] * np.ones(2)   # layer densities (N² ≈ 1e-5 s⁻²)
    one = np.ones(2)
    for fn in (lotw_bbl_columns, lotw_bbl_columns_numpy):
        kd = fn(dz, rho, 0.01 * one, 1.0e-4 * one, 1.0e-5 * one, thin_layer_threshold(dz), BBLParams())
        assert np.all(kd[:, 0] == 0.0)
        assert np.any(kd[:, 1] > 0.0)


def test_columns_refuse_non_finite_inputs() -> None:
    """Task 3 review M2: a NaN u* (or any non-finite / negative input) raises instead of giving Kd = 0 silently."""
    import pytest
    dz = np.full((10, 3), 5.0)
    rho = 1025.0 - 5.0e-3 * np.arange(10)[:, None] * np.ones(3)   # layer densities
    ok = dict(ustar=np.full(3, 0.01), absf=np.full(3, 1e-4), tke=np.full(3, 1e-6), h_thin=np.full(3, 1e-3))
    lotw_bbl_columns(dz, rho, **ok)
    for name in ok:
        for bad in (np.nan, np.inf, -1e-3):
            kw = dict(ok)
            kw[name] = np.array(ok[name])
            kw[name][1] = bad
            with pytest.raises(ValueError, match=name):
                lotw_bbl_columns(dz, rho, **kw)
    rhob = rho.copy()
    rhob[4, 2] = np.nan
    with pytest.raises(ValueError, match="rho"):
        lotw_bbl_columns(dz, rhob, **ok)


def test_dry_columns_ignore_rho() -> None:
    """The kernel takes the whole model grid (no gather of the wet columns, cost fix of 2026-10-31): ρ of a dry or
    land column (Σdz ≤ h_massless) is never read, so a NaN there gives Kd = 0 and leaves the wet columns bit for
    bit; a NaN in a wet column still raises (task 3 review M2)."""
    import pytest
    dz = np.full((10, 4), 5.0)
    dz[:, 1] = 0.0                                                 # land
    dz[:, 3] = 5.0e-5                                              # Σdz = 0.5 mm: dry (≤ h_massless)
    rho = 1025.0 - 5.0e-3 * np.arange(10)[:, None] * np.ones(4)
    ok = dict(ustar=np.full(4, 0.01), absf=np.full(4, 1e-4), tke=np.full(4, 1e-6), h_thin=thin_layer_threshold(dz))
    ref = lotw_bbl_columns(dz, rho, **ok)
    rhob = rho.copy()
    rhob[:, 1] = np.nan
    rhob[3, 3] = np.inf
    kd = lotw_bbl_columns(dz, rhob, **ok)
    np.testing.assert_array_equal(kd, ref)
    assert np.all(kd[:, [1, 3]] == 0.0)
    rhob[2, 2] = np.nan
    with pytest.raises(ValueError, match="rho"):
        lotw_bbl_columns(dz, rhob, **ok)


def test_thin_layers_from_densities_do_not_change_kd() -> None:
    """Final review I1: starting from layer densities (``BBLAddedClosure.update``), 1 mm layers split off the bottom
    of layers 2 and 5 of a 20 × 10 m stratified column, with the density of their parent layer, leave Kd_BBL at the
    interfaces of the original column unchanged (rtol 1e-12).  N² at a merged interface must come from the merged
    (thickness-weighted) densities over the merged spacing, as GLS and Jackson; the model N² across the 1 mm layer
    (spacing ½(10 m + 1 mm)) doubled N² at interfaces 2 and 5 before the fix (interface 7: Kd 0.118 → 0)."""
    from core.components.closure import BBLAddedClosure, ConstantClosure
    nz, dz_lay, drho = 20, 10.0, 2.0e-3      # N² = (g/ρ₀) Δρ / Δz = 1.962e-6 s⁻²
    rho0 = 1000.0 + drho * np.arange(nz)[::-1]   # densest at the bed (k = 0)
    dz0 = np.full(nz, dz_lay)
    split = (2, 5)
    dz1, rho1, common = [], [], []
    for k in range(nz):
        common.append(len(dz1))             # interface at the bottom of original layer k
        if k in split:
            dz1 += [1.0e-3, dz_lay - 1.0e-3]
            rho1 += [rho0[k], rho0[k]]
        else:
            dz1.append(dz_lay)
            rho1.append(rho0[k])
    common.append(len(dz1))
    dz1, rho1 = np.array(dz1), np.array(rho1)

    def kd_of(dz: np.ndarray, rho: np.ndarray) -> np.ndarray:
        n = dz.size
        cl = BBLAddedClosure(ConstantClosure(), n, (1, 1), np.array([[1.0e-4]]), rho0=1000.0, g=9.81)
        z = np.zeros((n, 1, 1))
        cl.update(z, z, dz[:, None, None], rho[:, None, None], None, np.array([[1.0e-4]]),
                  bbl_energy=np.array([[1.0e-4]]))
        return cl.kd[:, 0].copy()

    k0 = kd_of(dz0, rho0)
    k1 = kd_of(dz1, rho1)
    assert np.count_nonzero(k0 > 0.0) >= 5
    np.testing.assert_allclose(k1[common], k0, rtol=1e-12, atol=0.0)


def test_no_thin_layers_bit_for_bit_with_column() -> None:
    """Final review I1: in columns without thin layers, the columns kernel from densities equals the unmerged
    ``lotw_bbl_column`` with the N² of ``buoyancy_frequency2`` bit for bit (the merged-density N² changes only
    columns that merge).  Both take MOM6's bottom N² (``n2_bottom_bulk``, fix of 2026-11-01)."""
    from physics.mixing.gls import buoyancy_frequency2
    rng = np.random.default_rng(5)
    n, m = 30, 60
    dz = rng.uniform(0.5, 60.0, (n, m))
    rho = 1028.0 - np.cumsum(rng.uniform(-0.05, 0.3, (n, m)), axis=0)
    n2 = np.zeros((n + 1, m))
    n2[1:n] = buoyancy_frequency2(rho, dz, 1025.0, 9.81)
    for j in range(m):
        n2_bottom_bulk(n2[:, j], dz[:, j], n)
    ustar, absf, tke = rng.uniform(1e-3, 3e-2, m), rng.uniform(0.0, 1.4e-4, m), rng.uniform(1e-7, 1e-4, m)
    h_thin = thin_layer_threshold(dz)
    assert np.all(dz >= h_thin)
    for p in (BBLParams(), BBLParams(use_omega=False)):
        kd = lotw_bbl_columns(dz, rho, ustar, absf, tke, h_thin, p, rho0=1025.0, g=9.81)
        ref = np.stack([lotw_bbl_column(dz[:, j], n2[:, j], ustar[j], absf[j], tke[j], p) for j in range(m)], 1)
        assert np.any(ref > 0.0)
        assert np.array_equal(kd, ref)


def test_n2_bottom_overwrite_measurement_helper():
    """The measurement helper of spec decision 8 (``validation/mom6_column/bbl_n2bot_effect.py``, not the model
    path): MOM6's N²_bot (MOM_set_diffusivity.F90:1202-1258, h_amp = 0) replaces interfaces 1 and 2 only, as the
    thickness-weighted mean of the clipped jumps; a uniform column is unchanged."""
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "validation" / "mom6_column"))
    from bbl_n2bot_effect import n2_bottom_overwrite

    dz = np.array([2.0, 4.0, 6.0, 8.0])
    n2 = np.array([0.0, 1e-5, -3e-5, 4e-5, 0.0])
    out = n2_bottom_overwrite(dz, n2)
    expect = (1e-5 * 3.0 + 0.0 * 5.0) / (3.0 + 5.0)
    assert out[1] == out[2] == expect
    assert np.array_equal(out[[0, 3, 4]], n2[[0, 3, 4]])
    np.testing.assert_allclose(n2_bottom_overwrite(np.full(6, 2.0), np.full(7, 3e-6)), np.full(7, 3e-6), rtol=1e-15)
    # the model's kernel (fix of 2026-11-01) is the same operation
    model = n2.copy()
    n2_bottom_bulk(model, dz, 4)
    assert np.array_equal(model, out)


def test_n2_bottom_bulk_short_columns_unchanged():
    """With fewer than three merged layers there is at most one interior interface: MOM6's bulk equals the local N²
    (MOM_set_diffusivity.F90:1219-1223 add no second interface), so the kernel leaves N² as it is."""
    for c in (1, 2):
        n2 = np.array([0.0, -2e-5, 0.0])[: c + 1].copy()
        before = n2.copy()
        n2_bottom_bulk(n2, np.full(c, 5.0), c)
        assert np.array_equal(n2, before)
