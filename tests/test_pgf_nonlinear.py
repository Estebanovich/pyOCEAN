"""
Nonlinear finite-volume pressure gradient (TEOS-10 in the 3-D core, task 3, criterion 3):
``core.vertical.pgf_nonlinear.fv_pgf_nonlinear`` against compiled MOM6
``int_density_dz_generic_plm`` + ``PressureForce_FV_Bouss`` (validation/mom6_column/pgf_driver.F90,
reference ``tests/data/mom6_pgf_plm_reference.npz``), the Numba kernel against its NumPy oracle,
the reference-profile mode (zero force, thermobaricity) and vanished / dry / land columns.
"""

from __future__ import annotations

from pathlib import Path

import numba
import numpy as np
import pytest

from core.vertical.pgf_nonlinear import (_fv_pgf_nonlinear_numpy, _interp_profile_numpy, _layer_centres,
                                         fv_pgf_nonlinear)

REF = Path(__file__).parent / "data" / "mom6_pgf_plm_reference.npz"
EOS = ("wright_full", "teos10")
MW = ("vanished", "full", "none")
# Gulf profile north of the Gulf Stream (validation/mom6_column/eos_columns.py GULF_*), bed first
Z_REF = -np.array([4000.0, 3000.0, 2000.0, 1500.0, 1000.0, 800.0, 600.0, 400.0, 200.0, 100.0, 50.0, 0.0])
T_REF = np.array([2.3, 2.6, 3.6, 4.0, 4.8, 5.5, 7.0, 9.5, 12.0, 13.5, 15.0, 18.0])
S_REF = np.array([34.91, 34.94, 34.96, 34.97, 34.98, 35.0, 35.0, 35.1, 35.4, 35.6, 35.6, 35.5])


def _case(d, n):
    """Inputs of reference case n as a one-row grid (ny = 1); u-faces 1..nx−1 = MOM6 faces 0..nx−2."""
    h, T, S = d[f"c{n}_h"][:, None, :], d[f"c{n}_T"][:, None, :], d[f"c{n}_S"][:, None, :]
    nx = h.shape[2]
    du = np.ones((1, nx + 1))
    du[0, 1:nx] = d[f"c{n}_dx"]
    rho0, rho_ref, g, h_nv = d[f"c{n}_par"]
    return dict(h=h, H=d[f"c{n}_H"][None, :], eta=d[f"c{n}_eta"][None, :], T=T, S=S, du=du,
                dv=np.full((2, nx), 1000.0)), dict(rho0=rho0, rho_pgf_ref=rho_ref, g=g, h_vanished=h_nv)


@pytest.fixture(scope="module")
def ref():
    return np.load(REF)


@pytest.mark.parametrize("mw", MW)
@pytest.mark.parametrize("eos", EOS)
def test_matches_mom6_plm(ref, eos, mw):
    """Force at the u-faces = compiled MOM6 (pa(1) in pyOCEAN's baroclinic convention) to 1e-12."""
    for n in range(len(ref["names"])):
        arr, par = _case(ref, n)
        gx, gy = fv_pgf_nonlinear(**arr, eos_name=eos, mass_weight=mw, **par)
        want = ref[f"c{n}_{eos}_{mw}_pfu_bc"]
        got = gx[:, 0, 1:-1]
        np.testing.assert_allclose(got, want, rtol=1e-12, atol=1e-12 * np.abs(want).max(),
                                   err_msg=f"case {ref['names'][n]}")
        assert np.all(gx[:, 0, [0, -1]] == 0.0)
        assert np.all(gy == 0.0)                                   # ny = 1: only domain-edge v-faces


def test_reference_convention(ref):
    """The driver's two assemblies differ by the barotropic term: pfu_bc = pfu_mom6 + g Δη / dx."""
    for n in range(len(ref["names"])):
        g = ref[f"c{n}_par"][2]
        bt = g * np.diff(ref[f"c{n}_eta"]) / ref[f"c{n}_dx"]
        for eos in EOS:
            a, b = ref[f"c{n}_{eos}_vanished_pfu_mom6"], ref[f"c{n}_{eos}_vanished_pfu_bc"]
            np.testing.assert_allclose(b - a, np.broadcast_to(bt, a.shape), rtol=1e-6, atol=1e-11)


def test_y_faces_are_x_faces_transposed(ref):
    """The v-face force is the u-face kernel on the transposed grid, bit for bit."""
    arr, par = _case(ref, 2)
    gx, _ = fv_pgf_nonlinear(**arr, eos_name="teos10", **par)
    tr = dict(h=np.swapaxes(arr["h"], 1, 2), H=arr["H"].T, eta=arr["eta"].T, T=np.swapaxes(arr["T"], 1, 2),
              S=np.swapaxes(arr["S"], 1, 2), du=arr["dv"].T, dv=arr["du"].T)
    _, gy = fv_pgf_nonlinear(**tr, eos_name="teos10", **par)
    np.testing.assert_array_equal(np.swapaxes(gy, 1, 2), gx)


def _grid_2d(ref):
    """A 3 × 6 grid from the bottom-step case, T and S perturbed differently in each row."""
    arr, par = _case(ref, 3)
    ny = 3
    rows = np.arange(ny)[None, :, None]
    out = dict(h=np.repeat(arr["h"], ny, 1), H=np.repeat(arr["H"], ny, 0), eta=np.repeat(arr["eta"], ny, 0),
               T=np.repeat(arr["T"], ny, 1) + 0.3 * rows * np.sin(np.arange(6))[None, None, :],
               S=np.repeat(arr["S"], ny, 1) - 0.05 * rows, du=np.repeat(arr["du"], ny, 0),
               dv=np.full((ny + 1, 6), 700.0))
    out["H"][1, 2] += 3.0
    return out, par


@pytest.mark.parametrize("profile", [False, True])
def test_numba_equals_numpy(ref, profile):
    """Numba kernel = NumPy oracle bit for bit, with 1 and 5 threads, every EOS and mass weighting."""
    arr, par = _grid_2d(ref)
    prof = dict(T_ref=T_REF, S_ref=S_REF, z_ref=Z_REF) if profile else {}
    old = numba.get_num_threads()
    try:
        for eos in EOS:
            for mw in MW:
                want = _fv_pgf_nonlinear_numpy(**arr, eos_name=eos, mass_weight=mw, **par, **prof)
                for nt in (1, min(5, numba.config.NUMBA_NUM_THREADS)):
                    numba.set_num_threads(nt)
                    got = fv_pgf_nonlinear(**arr, eos_name=eos, mass_weight=mw, **par, **prof)
                    np.testing.assert_array_equal(got[0], want[0])
                    np.testing.assert_array_equal(got[1], want[1])
                assert np.abs(want[0]).max() > 0.0 and np.abs(want[1]).max() > 0.0
    finally:
        numba.set_num_threads(old)


def test_numba_equals_numpy_profile_index_walk(ref):
    """The kernel's walked profile index (``_ref_ts``, cost brief 2026-11-01) gives ``_interp``'s values: Numba =
    NumPy oracle with identical bits (incl. the sign of zero), 1 and 5 threads, for a dense, unevenly spaced
    profile that covers only the middle of the columns (quadrature points above, inside and below it)."""
    arr, par = _grid_2d(ref)
    zc = _layer_centres(arr["h"], arr["H"])
    lo, hi = zc.min(), zc.max()
    rng = np.random.default_rng(3)
    z_ref = np.sort(lo + (hi - lo) * (0.2 + 0.6 * rng.random(40)))
    prof = dict(T_ref=np.interp(z_ref, Z_REF, T_REF) + 0.1 * rng.standard_normal(40),
                S_ref=np.interp(z_ref, Z_REF, S_REF) + 0.01 * rng.standard_normal(40), z_ref=z_ref)
    old = numba.get_num_threads()
    try:
        for eos in EOS:
            for mw in MW:
                want = _fv_pgf_nonlinear_numpy(**arr, eos_name=eos, mass_weight=mw, **par, **prof)
                for nt in (1, min(5, numba.config.NUMBA_NUM_THREADS)):
                    numba.set_num_threads(nt)
                    got = fv_pgf_nonlinear(**arr, eos_name=eos, mass_weight=mw, **par, **prof)
                    for g_, w_ in zip(got, want):
                        np.testing.assert_array_equal(g_.view(np.int64), w_.view(np.int64))
                assert np.abs(want[0]).max() > 0.0
    finally:
        numba.set_num_threads(old)


def _profile_columns(h, H):
    """T, S equal to the reference profile at the layer centres (the kernel's own z_c and interpolation)."""
    zc = _layer_centres(h, H)
    return _interp_profile_numpy(zc, Z_REF, T_REF), _interp_profile_numpy(zc, Z_REF, S_REF)


@pytest.mark.parametrize("mw", MW)
@pytest.mark.parametrize("eos", EOS)
def test_profile_mode_zero_force(eos, mw):
    """
    Criterion 4 (thermobaricity): T, S equal to their reference profiles give no force on z* layers
    with partial bed cells and on sloping hybrid layers over the 600 → 164 m step (η = 0, so the
    column-constant surface term vanishes too).
    """
    nx, nz = 6, 25
    H = np.array([[600.0, 587.5, 600.0, 164.0, 171.2, 164.0]])
    zw0 = -24.0 * np.arange(nz + 1)
    hz = np.empty((nz, 1, nx))
    for i in range(nx):                                            # z*: partial bed cells, 1e-10 m below
        thk = np.clip(np.minimum(zw0[:-1], 0.0) - np.maximum(zw0[1:], -H[0, i]), 0.0, None)[::-1]
        hz[:, 0, i] = np.where(thk > 0.0, thk, 1e-10)
    s = np.linspace(0.0, 1.0, nz + 1)
    hh = np.empty((nz, 1, nx))
    for i in range(nx):                                            # hybrid: σ near the bed, sloping interfaces
        zi = -H[0, i] * (1.0 - s) ** (1.0 + 0.3 * i)
        hh[:, 0, i] = np.diff(zi)
    hh[3, 0, 3] = 1e-3                                             # a thin and a vanished layer
    hh[7, 0, 4] = 1e-10
    for h in (hz, hh):
        T, S = _profile_columns(h, H)
        gx, gy = fv_pgf_nonlinear(h, H, np.zeros((1, nx)), T, S, np.full((1, nx + 1), 2000.0),
                                  np.full((2, nx), 2000.0), eos_name=eos, rho0=1035.0, rho_pgf_ref=1030.0,
                                  g=9.8, mass_weight=mw, T_ref=T_REF, S_ref=S_REF, z_ref=Z_REF)
        assert np.abs(gx).max() <= 1e-15 and np.abs(gy).max() <= 1e-15
        # the same columns without the profile feel the PLM error of the curved profile
        gx0, _ = fv_pgf_nonlinear(h, H, np.zeros((1, nx)), T, S, np.full((1, nx + 1), 2000.0),
                                  np.full((2, nx), 2000.0), eos_name=eos, rho0=1035.0, rho_pgf_ref=1030.0,
                                  g=9.8, mass_weight=mw)
        assert np.abs(gx0).max() > 1e-12


def test_vanished_and_dry_finite(ref):
    """
    Review focus 2: 1e-10 m layers and bed steps give a finite force; land columns (H = NaN, T = NaN,
    h = 0) and dry columns (wet = False) give none on their faces, and their values are never read.
    """
    arr, par = _grid_2d(ref)
    arr = {k: v.copy() for k, v in arr.items()}
    arr["H"][0, 4] = np.nan
    arr["T"][:, 0, 4] = np.nan
    arr["S"][:, 0, 4] = np.nan
    arr["eta"][0, 4] = np.nan
    arr["h"][:, 0, 4] = 0.0
    wet = np.ones((3, 6), bool)
    wet[2, 1] = False
    arr["T"][:, 2, 1] = 1e30                                       # garbage in the dry column
    for eos in EOS:
        for mw in MW:
            gx, gy = fv_pgf_nonlinear(**arr, eos_name=eos, mass_weight=mw, wet=wet, **par)
            assert np.all(np.isfinite(gx)) and np.all(np.isfinite(gy))
            assert np.all(gx[:, 0, 4:6] == 0.0) and np.all(gy[:, 0:2, 4] == 0.0)    # land (0, 4)
            assert np.all(gx[:, 2, 1:3] == 0.0) and np.all(gy[:, 2:4, 1] == 0.0)    # dry (2, 1)
            assert np.abs(gx[:, 1, 2:5]).max() > 0.0
            # land is recognised without the mask (non-finite H, h = 0)
            gx2, gy2 = fv_pgf_nonlinear(**{**arr, "T": np.where(np.isfinite(arr["T"]), arr["T"], np.nan)},
                                        eos_name=eos, mass_weight=mw, **par)
            assert np.all(np.isfinite(gx2)) and np.all(gx2[:, 0, 4:6] == 0.0)


def test_bad_arguments(ref):
    arr, par = _case(ref, 0)
    with pytest.raises(ValueError, match="eos_name"):
        fv_pgf_nonlinear(**arr, eos_name="linear", **par)
    with pytest.raises(ValueError, match="mass_weight"):
        fv_pgf_nonlinear(**arr, eos_name="teos10", mass_weight="top", **par)
    with pytest.raises(ValueError, match="T_ref"):
        fv_pgf_nonlinear(**arr, eos_name="teos10", T_ref=T_REF, **par)
    with pytest.raises(ValueError, match="z_ref"):
        fv_pgf_nonlinear(**arr, eos_name="teos10", T_ref=T_REF, S_ref=S_REF, z_ref=Z_REF[::-1], **par)
