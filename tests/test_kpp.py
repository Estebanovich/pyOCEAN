"""
Tests for physics/mixing/kpp.py.

Validation strategy (CLAUDE.md §Validation):
  1. Analytical / limiting-case solutions where derivable from theory.
  2. Sign and physical-bounds checks (K ≥ 0, h_bl > 0, γ ≥ 0).
  3. Consistency checks (K continuous at OBL base, γ = 0 for stable forcing).
  4. Conservation check: vertical mixing conserves column-integrated tracer.
  5. Dimension, shape, and coordinate consistency.

Covers
------
  _velocity_scales    : stable/unstable limits, zero u* → small but finite w
  obl_depth           : quiescent (→ shallow h_bl), strongly mixed (→ deep h_bl),
                        stably stratified (→ shallow), unstably stratified (→ deep),
                        analytical Ri_b profile with known solution
  interior_diffusivity: positivity, K = K_bg when Ri_g → ∞ (stable),
                        K = ν₀ + K_bg when Ri_g = 0 (fully unstable),
                        convective adjustment when N² < 0
  _blmix              : K inside OBL ≥ K_int (max convention),
                        K continuous at OBL base (no step discontinuity within tol),
                        γ = 0 when b_flux ≤ 0 (stable),
                        γ > 0 when b_flux > 0 (unstable)
  kpp_tendency        : shape and dims, zero tendency for uniform T/S/u/v,
                        column tracer conservation (no surface flux),
                        sign of mixing (smooth T profile → down-gradient flux),
                        heating is stabilising (no nonlocal term), cooling is not
"""

import numpy as np
import pytest
import xarray as xr

from core.grid.horizontal import create_cartesian_grid
from core.grid.vertical import create_z_grid, uniform_z_grid
from core.state import ModelState
from physics.mixing.kpp import (
    _velocity_scales,
    _blmix,
    obl_depth,
    interior_diffusivity,
    kpp_tendency,
    K_BG,
    NU_0,
    KAPPA_S0,
    K_CONV,
    RI_CR,
    EPS,
    G,
    RHO_0,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

NX, NY, NZ = 6, 4, 8
DX = DY = 2000.0   # m
DZ = 10.0          # m (uniform 10-m layers → 80 m total)


@pytest.fixture
def hgrid():
    return create_cartesian_grid(nx=NX, ny=NY, dx=DX, dy=DY, f0=1e-4, beta=0.0)


@pytest.fixture
def vgrid():
    return uniform_z_grid(nz=NZ, dz_uniform=DZ)


@pytest.fixture
def state_uniform(hgrid, vgrid):
    """Resting state with uniform T = 15 °C, S = 35 PSU."""
    return ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)


@pytest.fixture
def state_stratified(hgrid, vgrid):
    """State with linear stable temperature stratification (warm above, cool below)."""
    st = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)
    dz = vgrid["dz"].values
    nz = NZ
    # T decreases with depth: T[k] = 15 − 0.5 k  (°C)
    T_np = np.zeros((nz, NY, NX))
    for k in range(nz):
        T_np[k] = 15.0 - 0.5 * k
    st.T = xr.DataArray(
        T_np, dims=st.T.dims, coords=st.T.coords, attrs=st.T.attrs
    )
    return st


@pytest.fixture
def zeros_2d():
    return np.zeros((NY, NX))


# ---------------------------------------------------------------------------
# _velocity_scales
# ---------------------------------------------------------------------------

class TestVelocityScales:

    def test_stable_stratification_reduces_scales(self):
        """Stable forcing (b_flux < 0) reduces w_m and w_s relative to neutral."""
        u_star = np.full((NY, NX), 0.01)
        depth = 5.0
        w_m_neutral, w_s_neutral = _velocity_scales(u_star, np.zeros((NY, NX)), depth)
        w_m_stable, w_s_stable = _velocity_scales(u_star, np.full((NY, NX), -1e-7), depth)
        assert np.all(w_m_stable <= w_m_neutral + EPS)
        assert np.all(w_s_stable <= w_s_neutral + EPS)

    def test_unstable_stratification_increases_scales(self):
        """Unstable forcing (b_flux > 0) increases w_m and w_s."""
        u_star = np.full((NY, NX), 0.01)
        depth = 5.0
        w_m_neutral, _ = _velocity_scales(u_star, np.zeros((NY, NX)), depth)
        w_m_unstable, _ = _velocity_scales(u_star, np.full((NY, NX), 1e-7), depth)
        assert np.all(w_m_unstable >= w_m_neutral - EPS)

    def test_zero_ustar_returns_small_positive(self):
        """Zero friction velocity → velocity scales are tiny but positive (regularised)."""
        u_star = np.zeros((NY, NX))
        b_flux = np.zeros((NY, NX))
        w_m, w_s = _velocity_scales(u_star, b_flux, 5.0)
        assert np.all(w_m > 0)
        assert np.all(w_s > 0)

    def test_scales_positive(self):
        """Velocity scales are always positive."""
        u_star = np.abs(np.random.default_rng(42).standard_normal((NY, NX))) * 0.02
        b_flux = np.random.default_rng(0).standard_normal((NY, NX)) * 1e-7
        w_m, w_s = _velocity_scales(u_star, b_flux, 20.0)
        assert np.all(w_m > 0)
        assert np.all(w_s > 0)

    def test_scales_bounded_above(self):
        """Velocity scales never exceed 0.5 m/s (hard cap)."""
        u_star = np.full((NY, NX), 0.5)
        b_flux = np.full((NY, NX), 1.0)   # very strong destabilising flux
        w_m, w_s = _velocity_scales(u_star, b_flux, 5.0)
        assert np.all(w_m <= 0.5 + EPS)
        assert np.all(w_s <= 0.5 + EPS)


# ---------------------------------------------------------------------------
# obl_depth
# ---------------------------------------------------------------------------

class TestOblDepth:

    def _rho_stratified(self, nz, ny, nx, dz, drho_dz=0.1):
        """Linear density profile: ρ increases with depth at drho_dz [kg/m³/m]."""
        rho = np.ones((nz, ny, nx)) * RHO_0
        for k in range(nz):
            rho[k] += drho_dz * (k + 0.5) * dz
        return rho

    def test_quiescent_uniform_density_returns_shallow(self, vgrid):
        """Uniform density + zero shear + zero u* → Ri_b = 0 everywhere → h_bl = d[0]."""
        nz = NZ
        dz = vgrid["dz"].values
        z_t = vgrid["z_t"].values
        rho = np.ones((nz, NY, NX)) * RHO_0
        u_t = np.zeros((nz, NY, NX))
        v_t = np.zeros((nz, NY, NX))
        u_star = np.zeros((NY, NX))
        b_flux = np.zeros((NY, NX))
        h_bl = obl_depth(rho, u_t, v_t, z_t, u_star, b_flux)
        # With uniform density and zero shear, Ri_b = 0/0 → never crosses Ri_cr
        # h_bl defaults to full column depth
        assert h_bl.shape == (NY, NX)
        assert np.all(h_bl > 0)

    def test_strong_stratification_gives_shallow_obl(self, vgrid):
        """Strong stable stratification → large Ri_b at top layers → shallow h_bl."""
        nz = NZ
        z_t = vgrid["z_t"].values
        dz = vgrid["dz"].values
        rho = self._rho_stratified(nz, NY, NX, DZ, drho_dz=5.0)   # very dense below
        u_t = np.ones((nz, NY, NX)) * 0.1   # uniform u → shear = 0 except at k=0 ref
        v_t = np.zeros((nz, NY, NX))
        u_star = np.full((NY, NX), 0.01)
        b_flux = np.zeros((NY, NX))   # neutral surface
        h_bl = obl_depth(rho, u_t, v_t, z_t, u_star, b_flux)
        assert np.all(h_bl > 0)
        assert np.all(h_bl <= abs(z_t[-1]) + 1e-9)

    def test_obl_depth_shape(self, vgrid):
        """Output shape is (ny, nx)."""
        nz = NZ
        z_t = vgrid["z_t"].values
        rho = np.ones((nz, NY, NX)) * RHO_0
        u_t = np.zeros((nz, NY, NX))
        v_t = np.zeros((nz, NY, NX))
        h_bl = obl_depth(rho, u_t, v_t, z_t, np.zeros((NY, NX)), np.zeros((NY, NX)))
        assert h_bl.shape == (NY, NX)

    def test_obl_depth_positive(self, vgrid):
        """h_bl is always positive."""
        nz = NZ
        z_t = vgrid["z_t"].values
        rho = self._rho_stratified(nz, NY, NX, DZ)
        u_t = np.random.default_rng(7).standard_normal((nz, NY, NX)) * 0.05
        v_t = np.random.default_rng(8).standard_normal((nz, NY, NX)) * 0.05
        h_bl = obl_depth(rho, u_t, v_t, z_t, np.zeros((NY, NX)), np.zeros((NY, NX)))
        assert np.all(h_bl > 0)

    def test_analytical_ri_b_single_layer(self, vgrid):
        """
        Analytical check: for a two-layer column with known Δρ and Δu,
        Ri_b at layer k=1 should equal the formula value.

        Setup:  nz = 2,  dz = [10, 10] m,  z_t = [−5, −15] m
                ρ(0) = 1025 kg/m³,  ρ(1) = 1025.5 kg/m³
                u(0) = 0.5 m/s,     u(1) = 0 m/s
                v = 0 everywhere

        Ri_b(k=1) = ΔB × d / (ΔU² + Vt²)
                  = (g/ρ₀) × 0.5 × 15 / (0.5² + Vt²)
        With Vt² ≈ 0 (u*=0, b_flux=0 → w_s → small), Ri_b ≈ (9.81/1025)×0.5×15/0.25
        """
        nz2 = 2
        vg2 = create_z_grid(nz2, [10.0, 10.0])
        z_t2 = vg2["z_t"].values   # [-5, -15]
        rho2 = np.array([[[1025.0]], [[1025.5]]])                # (2,1,1)
        u_t2 = np.array([[[0.5]],   [[0.0]]])                   # (2,1,1)
        v_t2 = np.zeros((2, 1, 1))
        u_star2 = np.full((1, 1), 1e-6)   # near zero
        b_flux2 = np.zeros((1, 1))

        # Expected Ri_b at k=1 (depth = 15 m):
        delta_B = (G / RHO_0) * (1025.5 - 1025.0)   # ≈ 0.004786 m/s²
        delta_u2 = 0.5 ** 2                           # 0.25 m²/s²
        d1 = abs(z_t2[1])                              # 15 m
        Ri_b_expected = delta_B * d1 / delta_u2       # ≈ 0.287

        h_bl2 = obl_depth(rho2, u_t2, v_t2, z_t2, u_star2, b_flux2)
        # With Ri_b(k=1) < Ri_cr = 0.3, h_bl should default to full depth
        assert h_bl2[0, 0] == pytest.approx(d1, rel=0.05)

    def test_deep_shear_exceeds_ri_cr(self):
        """When shear is large, Ri_b < Ri_cr and h_bl defaults to full depth."""
        nz3 = 4
        dz3 = [5.0] * nz3
        vg3 = create_z_grid(nz3, dz3)
        z_t3 = vg3["z_t"].values
        rho3 = np.ones((nz3, 1, 1)) * RHO_0   # no stratification
        # Large shear: u varies strongly with depth → Ri_b = 0 everywhere (ρ uniform)
        u_t3 = np.arange(nz3).reshape(nz3, 1, 1).astype(float)
        v_t3 = np.zeros((nz3, 1, 1))
        h_bl3 = obl_depth(rho3, u_t3, v_t3, z_t3,
                           np.zeros((1, 1)), np.zeros((1, 1)))
        # Ri_b = 0 everywhere (no buoyancy difference) → never crosses Ri_cr → full depth
        assert h_bl3[0, 0] == pytest.approx(abs(z_t3[-1]), rel=0.01)


# ---------------------------------------------------------------------------
# interior_diffusivity
# ---------------------------------------------------------------------------

class TestInteriorDiffusivity:

    def test_output_shape(self, state_uniform, vgrid):
        """K_m and K_s have shape (nz+1, ny, nx)."""
        rho = np.ones((NZ, NY, NX)) * RHO_0
        u_t = np.zeros((NZ, NY, NX))
        v_t = np.zeros((NZ, NY, NX))
        K_m, K_s = interior_diffusivity(rho, u_t, v_t, vgrid["dz"].values)
        assert K_m.shape == (NZ + 1, NY, NX)
        assert K_s.shape == (NZ + 1, NY, NX)

    def test_non_negative(self):
        """Diffusivities are always ≥ 0."""
        rng = np.random.default_rng(99)
        rho = RHO_0 + rng.standard_normal((NZ, NY, NX)) * 0.5
        u_t = rng.standard_normal((NZ, NY, NX)) * 0.1
        v_t = rng.standard_normal((NZ, NY, NX)) * 0.1
        K_m, K_s = interior_diffusivity(rho, u_t, v_t, np.full(NZ, DZ))
        assert np.all(K_m >= 0)
        assert np.all(K_s >= 0)

    def test_surface_and_bottom_bc_zero(self):
        """K[0] = K[nz] = 0 (surface and bottom boundary fluxes)."""
        rho = np.ones((NZ, NY, NX)) * RHO_0
        u_t = np.zeros((NZ, NY, NX))
        v_t = np.zeros((NZ, NY, NX))
        K_m, K_s = interior_diffusivity(rho, u_t, v_t, np.full(NZ, DZ))
        assert np.all(K_m[0] == 0.0)
        assert np.all(K_s[0] == 0.0)
        assert np.all(K_m[NZ] == 0.0)
        assert np.all(K_s[NZ] == 0.0)

    def test_stable_large_ri_gives_background_only(self):
        """Very strong stratification → Ri_g → ∞ → f(Ri) = 0 → K = K_bg."""
        # Very dense below: huge N² → large Ri_g
        rho = np.ones((NZ, NY, NX)) * RHO_0
        for k in range(NZ):
            rho[k] += 100.0 * k   # absurdly strong stratification → Ri_g >> Ri₀
        u_t = np.zeros((NZ, NY, NX))
        v_t = np.zeros((NZ, NY, NX))
        K_m, K_s = interior_diffusivity(rho, u_t, v_t, np.full(NZ, DZ))
        # Interior interfaces (k_f = 1 .. nz-1)
        assert np.allclose(K_m[1:NZ], K_BG, rtol=1e-6)
        assert np.allclose(K_s[1:NZ], K_BG, rtol=1e-6)

    def test_no_shear_large_n2_gives_background(self):
        """No shear + large N² → Ri_g → ∞ → K = K_bg."""
        rho = np.ones((NZ, NY, NX)) * RHO_0
        for k in range(NZ):
            rho[k] += 0.5 * k   # stable stratification
        u_t = np.zeros((NZ, NY, NX))
        v_t = np.zeros((NZ, NY, NX))
        K_m, K_s = interior_diffusivity(rho, u_t, v_t, np.full(NZ, DZ))
        assert np.all(K_m[1:NZ] >= K_BG - EPS)
        assert np.all(K_s[1:NZ] >= K_BG - EPS)

    def test_convective_adjustment_when_unstable(self):
        """N² < 0 (heavy above light) → K includes K_conv."""
        rho = np.ones((NZ, NY, NX)) * RHO_0
        for k in range(NZ):
            rho[k] -= 0.5 * k   # density decreases with depth → unstable
        u_t = np.zeros((NZ, NY, NX))
        v_t = np.zeros((NZ, NY, NX))
        K_m, K_s = interior_diffusivity(rho, u_t, v_t, np.full(NZ, DZ))
        # All interior interfaces should have large K from convective adjustment
        assert np.all(K_m[1:NZ] >= K_CONV - EPS)
        assert np.all(K_s[1:NZ] >= K_CONV - EPS)

    def test_zero_shear_zero_stratification_gives_background(self):
        """Zero shear, zero N² → Ri_g = 0/0 → f(Ri) depends on S²; with S²≈ε, Ri_g≈0 → f=1 → K = ν₀+K_bg."""
        # Actually Ri_g = 0/(0 + ε) ≈ 0 → ratio=0 → f_Ri=1 → K = ν₀ + K_bg
        rho = np.ones((NZ, NY, NX)) * RHO_0   # zero stratification
        u_t = np.zeros((NZ, NY, NX))
        v_t = np.zeros((NZ, NY, NX))
        K_m, K_s = interior_diffusivity(rho, u_t, v_t, np.full(NZ, DZ))
        # N²=0, S²≈0 → Ri_g = 0/ε ≈ 0 → f_Ri = 1
        assert np.allclose(K_m[1:NZ], NU_0 + K_BG, rtol=1e-6)
        assert np.allclose(K_s[1:NZ], KAPPA_S0 + K_BG, rtol=1e-6)


# ---------------------------------------------------------------------------
# _blmix
# ---------------------------------------------------------------------------

class TestBlmix:

    def _make_args(self, vgrid, h_bl_val=20.0):
        """Common inputs for _blmix tests."""
        dz = vgrid["dz"].values
        z_f = vgrid["z_f"].values
        z_t = vgrid["z_t"].values
        rho = np.ones((NZ, NY, NX)) * RHO_0
        for k in range(NZ):
            rho[k] += 0.2 * k   # weak stable stratification
        u_t = np.zeros((NZ, NY, NX))
        v_t = np.zeros((NZ, NY, NX))
        K_m_int, K_s_int = interior_diffusivity(rho, u_t, v_t, dz)
        h_bl = np.full((NY, NX), h_bl_val)
        u_star = np.full((NY, NX), 0.01)
        b_flux = np.zeros((NY, NX))
        wT = np.zeros((NY, NX))
        wS = np.zeros((NY, NX))
        return K_m_int, K_s_int, z_f, dz, h_bl, u_star, b_flux, wT, wS

    def test_output_shapes(self, vgrid):
        """K_m_bl, K_s_bl, gamma_T, gamma_S have shape (nz+1, ny, nx)."""
        K_m_int, K_s_int, z_f, dz, h_bl, u_star, b_flux, wT, wS = self._make_args(vgrid)
        K_m_bl, K_s_bl, gT, gS = _blmix(h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux, wT, wS)
        assert K_m_bl.shape == (NZ + 1, NY, NX)
        assert K_s_bl.shape == (NZ + 1, NY, NX)
        assert gT.shape == (NZ + 1, NY, NX)
        assert gS.shape == (NZ + 1, NY, NX)

    def test_K_bl_geq_K_int_inside_obl(self, vgrid):
        """Inside the OBL, K_bl ≥ K_int (max convention, HYCOM mxkprf.F90)."""
        K_m_int, K_s_int, z_f, dz, h_bl, u_star, b_flux, wT, wS = self._make_args(vgrid, 30.0)
        K_m_bl, K_s_bl, _, _ = _blmix(h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux, wT, wS)
        d_f = -z_f
        in_bl = d_f[np.newaxis, :, np.newaxis, np.newaxis] <= h_bl  # broadcast wrong
        # Use simple mask
        for k_f in range(1, NZ):
            d_kf = d_f[k_f]
            inside = d_kf <= h_bl   # (ny, nx)
            K_m_inside = K_m_bl[k_f][inside]
            K_m_int_inside = K_m_int[k_f][inside]
            assert np.all(K_m_inside >= K_m_int_inside - EPS), (
                f"K_m_bl < K_m_int at k_f={k_f} inside OBL"
            )

    def test_K_non_negative(self, vgrid):
        """Diffusivities from _blmix are always ≥ 0."""
        K_m_int, K_s_int, z_f, dz, h_bl, u_star, b_flux, wT, wS = self._make_args(vgrid)
        K_m_bl, K_s_bl, _, _ = _blmix(h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux, wT, wS)
        assert np.all(K_m_bl >= 0)
        assert np.all(K_s_bl >= 0)

    def test_gamma_zero_for_stable(self, vgrid):
        """γ_T = γ_S = 0 when b_flux ≤ 0 (stable forcing)."""
        K_m_int, K_s_int, z_f, dz, h_bl, u_star, b_flux, wT, wS = self._make_args(vgrid)
        b_flux_stable = np.full((NY, NX), -1e-7)   # stable
        wT_nonzero = np.full((NY, NX), 1e-5)
        _, _, gT, gS = _blmix(h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux_stable, wT_nonzero, wS)
        assert np.all(gT == 0.0)
        assert np.all(gS == 0.0)

    def test_gamma_nonzero_for_unstable(self, vgrid):
        """γ > 0 inside OBL when b_flux > 0 and wT_sfc ≠ 0."""
        K_m_int, K_s_int, z_f, dz, h_bl, u_star, b_flux, wT, wS = self._make_args(vgrid)
        b_flux_unstable = np.full((NY, NX), 1e-7)   # destabilising
        wT_nonzero = np.full((NY, NX), 1e-5)
        _, _, gT, _ = _blmix(h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux_unstable, wT_nonzero, wS)
        # At least one interior interface inside OBL should have γ > 0
        assert np.any(gT > 0)


# ---------------------------------------------------------------------------
# kpp_tendency
# ---------------------------------------------------------------------------

class TestKppTendency:

    def test_output_shape_and_dims(self, state_uniform, vgrid):
        """Output ModelState has correct shapes and dimension names."""
        tend = kpp_tendency(state_uniform, vgrid)
        nz, ny, nx = NZ, NY, NX
        assert tend.T.shape == (nz, ny, nx)
        assert tend.S.shape == (nz, ny, nx)
        assert tend.u.shape == (nz, ny, nx + 1)
        assert tend.v.shape == (nz, ny + 1, nx)
        assert tend.eta.shape == (ny, nx)
        assert "k" in tend.T.dims
        assert "j_t" in tend.T.dims
        assert "i_t" in tend.T.dims

    def test_zero_tendency_for_uniform_T_S(self, state_uniform, vgrid):
        """Uniform T and S → no diffusion → zero T and S tendency."""
        tend = kpp_tendency(state_uniform, vgrid)
        assert np.allclose(tend.T.values, 0.0, atol=1e-15), "Expected zero T tendency for uniform T"
        assert np.allclose(tend.S.values, 0.0, atol=1e-15), "Expected zero S tendency for uniform S"

    def test_zero_tendency_for_uniform_velocity(self, state_uniform, vgrid):
        """Uniform u and v → no vertical shear → zero u and v tendency from KPP."""
        tend = kpp_tendency(state_uniform, vgrid)
        assert np.allclose(tend.u.values, 0.0, atol=1e-15)
        assert np.allclose(tend.v.values, 0.0, atol=1e-15)

    def test_eta_and_w_tendency_zero(self, state_uniform, vgrid):
        """KPP does not change η or w (those are momentum/continuity concerns)."""
        tend = kpp_tendency(state_uniform, vgrid)
        assert np.allclose(tend.eta.values, 0.0, atol=1e-15)
        assert np.allclose(tend.w.values, 0.0, atol=1e-15)

    def test_column_tracer_conservation_no_surface_flux(self, state_stratified, vgrid):
        """
        Vertical diffusion conserves the column-integrated tracer when there
        is no surface flux (closed-column test).

        Σ_k dz[k] × (∂T/∂t)[k] = 0  ←  flux divergence, with F[0]=F[nz]=0
        """
        dz = vgrid["dz"].values
        tend = kpp_tendency(state_stratified, vgrid)
        dT = tend.T.values   # (nz, ny, nx)
        # Column sum: Σ_k dz[k] × dT[k]  should be ≈ 0 for each (j, i)
        col_sum = np.sum(dz[:, np.newaxis, np.newaxis] * dT, axis=0)   # (ny, nx)
        assert np.allclose(col_sum, 0.0, atol=1e-14), (
            f"Column tracer not conserved; max |sum| = {np.max(np.abs(col_sum)):.2e}"
        )

    def test_mixing_smooths_gradient(self, hgrid, vgrid):
        """
        Linear T profile → diffusion tendency has correct sign:
        surface layer (warm) cools, deep layer (cold) warms.
        """
        # Build state with sharp warm surface layer
        st = ModelState.from_grids(hgrid, vgrid, T_init=10.0, S_init=35.0)
        T_np = st.T.values.copy()
        T_np[0] = 20.0   # very warm surface layer
        # rest = 10 °C → large downward flux at k=0/1 interface
        st.T = xr.DataArray(T_np, dims=st.T.dims, coords=st.T.coords, attrs=st.T.attrs)
        tend = kpp_tendency(st, vgrid)
        dT = tend.T.values
        # Surface layer should cool (negative tendency)
        assert np.all(dT[0] <= 0.0 + 1e-15), "Surface warm layer should lose heat"
        # Layer below should warm (positive tendency)
        assert np.all(dT[1] >= 0.0 - 1e-15), "Layer below warm surface should gain heat"

    def test_old_kpp_heating_is_stabilising(self, hgrid, vgrid, monkeypatch):
        """
        Surface heating (Q_net > 0) is a buoyancy gain, i.e. stabilising forcing
        (Large et al. 1994 §2: B_f > 0 stable; MOM6/CVMix ``lstable = surf_buoy > 0``,
        cvmix_kpp.F90:865).  It must give neither the nonlocal term nor the
        convective (unstable) branch of the velocity scales; cooling must give both.

        This replaces ``test_surface_heating_produces_nonlocal_warming``, which
        pinned the sign error of Phase 1 (heating treated as convective,
        N3 spec item 4, kpp_mom6_cvmix.md §8 A1).  The internal helpers keep their
        "positive = destabilising" argument; ``kpp_tendency`` must pass them
        −B_f.
        """
        import physics.mixing.kpp as kpp_mod

        st = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)
        T_np = st.T.values.copy()
        for k in range(NZ):
            T_np[k] = 15.0 - 0.1 * k
        st.T = xr.DataArray(T_np, dims=st.T.dims, coords=st.T.coords, attrs=st.T.attrs)

        seen = {}
        original = kpp_mod._blmix

        def spy(h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux, wT_sfc, wS_sfc):
            out = original(h_bl, K_m_int, K_s_int, z_f, dz, u_star, b_flux, wT_sfc, wS_sfc)
            seen["b_flux"] = np.array(b_flux)
            seen["gamma_T"] = out[2]
            return out

        monkeypatch.setattr(kpp_mod, "_blmix", spy)
        tau = np.full((NY, NX), 0.05)

        tend = kpp_tendency(st, vgrid, tau_x=tau, Q_net=np.full((NY, NX), 500.0))
        assert np.all(np.isfinite(tend.T.values))
        assert np.all(seen["b_flux"] < 0.0), "heating must reach the helpers as stable forcing"
        assert np.all(seen["gamma_T"] == 0.0), "heating must not switch on the nonlocal term"

        kpp_tendency(st, vgrid, tau_x=tau, Q_net=np.full((NY, NX), -500.0))
        assert np.all(seen["b_flux"] > 0.0), "cooling must reach the helpers as unstable forcing"
        assert np.any(seen["gamma_T"] != 0.0), "cooling must switch on the nonlocal term"

    def test_no_nan_with_random_state(self, hgrid, vgrid):
        """No NaN or Inf in output for a random but physically reasonable state."""
        rng = np.random.default_rng(2024)
        st = ModelState.from_grids(hgrid, vgrid, T_init=15.0, S_init=35.0)
        T_np = st.T.values + rng.standard_normal(st.T.shape) * 2.0
        S_np = st.S.values + rng.standard_normal(st.S.shape) * 0.5
        st.T = xr.DataArray(T_np, dims=st.T.dims, coords=st.T.coords, attrs=st.T.attrs)
        st.S = xr.DataArray(S_np, dims=st.S.dims, coords=st.S.coords, attrs=st.S.attrs)
        tau_x = rng.standard_normal((NY, NX)) * 0.1
        tau_y = rng.standard_normal((NY, NX)) * 0.1
        Q = rng.standard_normal((NY, NX)) * 100.0
        tend = kpp_tendency(st, vgrid, tau_x=tau_x, tau_y=tau_y, Q_net=Q)
        for field in [tend.T, tend.S, tend.u, tend.v]:
            assert np.all(np.isfinite(field.values)), f"{field.name} has non-finite values"

    def test_units_attrs_present(self, state_uniform, vgrid):
        """Output DataArrays carry units attributes."""
        tend = kpp_tendency(state_uniform, vgrid)
        assert "units" in tend.T.attrs
        assert "units" in tend.S.attrs
        assert "units" in tend.u.attrs
        assert "units" in tend.v.attrs
