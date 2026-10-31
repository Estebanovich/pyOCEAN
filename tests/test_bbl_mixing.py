"""
Tests of the drag-driven bottom-boundary-layer mixing column oracle
(``physics/mixing/bbl_mixing.py``, port of MOM6 ``add_LOTW_BBL_diffusivity``,
MOM_set_diffusivity.F90:1606-1789).

Arrays follow pyOCEAN: layers k = 0 (bed) … N−1, interfaces 0 (bed) … N.
"""

from __future__ import annotations

import numpy as np

from core.grid.horizontal import OMEGA
from physics.mixing.bbl_mixing import BBLParams, lotw_bbl_column


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
