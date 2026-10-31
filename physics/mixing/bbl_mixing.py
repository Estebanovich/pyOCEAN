"""
Drag-driven bottom-boundary-layer (BBL) diapycnal mixing: law-of-the-wall
diffusivity limited by the energy that bottom drag removes from the mean flow,
column version ported from MOM6 ``add_LOTW_BBL_diffusivity`` (BBL task 1).

This module is the **NumPy oracle**: a plain-Python, one-column port that keeps
MOM6's loop line by line, so that it can be compared with compiled MOM6 (task 2)
and with a Numba kernel (task 3).  Speed is irrelevant here.

Reference implementation (cited as ``file:line``, MOM6 git 0e9a561):
SD = ``~/pyOCEAN_refs/MOM6/src/parameterizations/vertical/MOM_set_diffusivity.F90``,
``add_LOTW_BBL_diffusivity`` SD:1606-1789, parameters SD:2417-2524; reference
read in ``docs/design/bbl_refs/mom6_bbl_mixing.md``.  This is the scheme of the
OM4 configurations (``USE_LOTW_BBL_DIFFUSIVITY = True``,
``ice_ocean_SIS2/OM4_025/MOM_input``:404-424, 661-689), the one of MOM6's two
BBL schemes that does not depend on layer target densities, so it works in σ,
z* and hybrid coordinates alike.  Background on energy-limited BBL mixing:
Legg et al. (2006), Ocean Modelling 11, 69–97, Appendix A,
https://doi.org/10.1016/j.ocemod.2004.11.006 (a different, single-layer form).

Physics (per column; heights in m, diffusivity Kd [m² s⁻¹], energy per unit
area and density TKE [m³ s⁻³])
------------------------------------------------------------------------
1. **Energy available**: TKE_col = BBL_EFFIC · E_drag (SD:1709), with E_drag the
   mean kinetic energy lost to bottom drag.  The caller passes TKE_col already
   multiplied by ``bbl_effic`` (argument ``tke``).
2. **Decay scale** (SD:1699-1700):

       I_decay = max(1/L_max, |f|/u*)  [m⁻¹],   L_max = ``max_decay``,

   so the e-folding height is min(u*/|f|, L_max); with f = 0 it is L_max.
3. **Upward sweep** over the interior interfaces, from the bed up (SD:1729-1787).
   Across each layer k the remaining energy decays ("Joule heating"):

       TKE_rem ← exp(−I_decay h_k) TKE_rem                         (SD:1741)

   At the upper interface of layer k, at height z_b above the bed and depth
   D − z below the surface (D = column depth), the law-of-the-wall
   diffusivity limited by rotation is (SD:1756-1757)

       Kd_wall = κ u*² z_b (D − z) / (u* D + |f| h_b (D − z)),

   κ = ``von_karman``, h_b = thickness between the interface and the bed
   (= z_b, Boussinesq).  Near the bed Kd_wall ≈ κ u* z_b; far from it
   Kd_wall → κ u*²/|f|.  Its mixing work across the interface is (SD:1762)

       W = Kd_wall dz_int max(N², N²_min),   N²_min = Ω² if ``use_omega`` else 0,

   dz_int = ½(h_k + h_{k+1}).  If W > 0, Kd is cut so that it spends at most
   the energy left (SD:1765-1767): Kd = Kd_wall · min(W, TKE_rem)/W and
   TKE_rem ← TKE_rem − min(W, TKE_rem).  If W ≤ 0 (N² ≤ 0 without the Ω²
   floor, or dz_int = 0), Kd = ``kd_max`` while energy remains, 0 otherwise
   (SD:1769-1775), and no energy is spent.  KD_MAX is **not** a cap on the
  law-of-the-wall Kd: with enough energy Kd → κ u*²/|f| (e.g. 1.6 m² s⁻¹ for
  u* = 2 cm/s, |f| = 1e-4 s⁻¹), as in MOM6.
4. Kd = 0 at the bed and the surface (MOM6 never touches K = 1, nz + 1).  MOM6
   adds Kd to the net interface and layer diffusivities (SD:1782-1783); the
   oracle returns the BBL part only (MOM6's diagnostic ``Kd_BBL``, SD:1785).
   No viscosity is produced and there is no Prandtl number (MOM6 neither).

Decisions of this port (Claude, 2026-10-30; see the task-1 report)
------------------------------------------------------------------
* Boussinesq, heights in m: MOM6's thickness h [H] and height change dz [Z]
  are the same array ``dz`` (H = Z = m), so h_bot = z_bot and the decay uses dz.
* Only the newer arithmetic (``LOTW_BBL_ANSWER_DATE`` > 20240630, MOM6's default
  through ``DEFAULT_ANSWER_DATE`` = 99991231, SD:2347-2349, 2466-2471): D − z is
  the cumulative sum of the layer thicknesses from the surface down, started at
  ``GV%dz_subroundoff`` = 1e-20 · max(ANGSTROM, 1e-17 m) = 1e-30 m
  (``MOM_verticalGrid.F90``:156-158, 220), and D is that sum at the bed.
* No tidal u* (``fluxes%ustar_tidal``), tidal BBL dissipation
  (``fluxes%BBL_tidal_dis``) or Rayleigh-drag energy (``visc%Ray_u``,
  SD:1732-1738): pyOCEAN has none of them; the caller folds any extra energy
  into ``tke``.
* u* = 0 returns Kd = 0 at once.  MOM6 would set Kd = KD_MAX at every interior
  interface when u* = 0 but TKE_col > 0 (Kd_wall = 0 → W = 0 → the KD_MAX
  branch); that input only arises in MOM6 from tidal dissipation without drag,
  which pyOCEAN lacks.  With u* > 0 and tke = 0 the MOM6 path already gives 0.
* ``kd_max`` must be > 0 (OM4: 0.1 m² s⁻¹); MOM6's default KD_MAX = −1 would
  make the W ≤ 0 branch return a negative Kd.  The pre-flight check belongs to
  the caller (task 4).
* Every layer must be thicker than zero: thin layers are merged by the caller
  before the call (as for GLS and Jackson), because dz_int = 0 sends an
  interface to the ``kd_max`` branch.
* ``x*x`` instead of ``x**2``: libm ``pow`` may differ by 1 ulp from Fortran's
  ``**2`` (seen in the kappa_shear port), and task 2 compares at 1e-10.

Conventions of this module
--------------------------
* ``lotw_bbl_columns`` (Numba, threads over columns) and its NumPy oracle
  ``lotw_bbl_columns_numpy`` apply the thin-layer merge of the ALE closures
  (section "Columns with thin-layer merge" below) and then the column scheme.
* ``lotw_bbl_column`` takes model arrays: layers k = 0 (bed) … N−1 (surface),
  interfaces 0 (bed) … N (surface).
* Internally the arrays follow MOM6: **1-based, top-down** (layer 1 = surface,
  interface 1 = surface, interface nz + 1 = bed), length nz + 2 with index 0
  unused, so every line keeps the indices of the Fortran it ports.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
from numba import njit, prange

from core.grid.horizontal import OMEGA

# GV%dz_subroundoff = 1e-20 max(ANGSTROM = 1e-10 m, 1e-17 m) [m]
# (MOM_verticalGrid.F90:156-158, 220).
DZ_SUBROUNDOFF: float = 1.0e-30


@dataclass(frozen=True)
class BBLParams:
    """
    Parameters of the drag-driven BBL mixing; defaults = MOM6 / OM4
    (SD:2417-2524; ``OM4_025/MOM_input``:404-424, 661-689).

    Attributes
    ----------
    bbl_effic : float
        Fraction of the drag energy loss used for mixing [nondim]; BBL_EFFIC
        0.20.  Used by the caller to form ``tke``, not by ``lotw_bbl_column``.
    max_decay : float
        Maximum decay scale of the TKE with height [m]; BBL_MIXING_MAX_DECAY 200.
    von_karman : float
        κ of the law of the wall [nondim]; VON_KARMAN_BBL 0.41.
    use_omega : bool
        Floor N² by Ω² in the mixing work; LOTW_BBL_USE_OMEGA True.
    kd_max : float
        Diffusivity where mixing costs no energy [m² s⁻¹]; KD_MAX 0.1 (OM4;
        MOM6's default −1 is not allowed here).
    """

    bbl_effic: float = 0.2      # BBL_EFFIC
    max_decay: float = 200.0    # BBL_MIXING_MAX_DECAY [m]
    von_karman: float = 0.41    # VON_KARMAN_BBL
    use_omega: bool = True      # LOTW_BBL_USE_OMEGA
    kd_max: float = 0.1         # KD_MAX [m2 s-1] (OM4)


def lotw_bbl_column(dz: np.ndarray, n2: np.ndarray, ustar: float, absf: float,
                    tke: float, p: BBLParams = BBLParams()) -> np.ndarray:
    """
    Energy-limited law-of-the-wall BBL diffusivity of one column
    (MOM6 ``add_LOTW_BBL_diffusivity``, SD:1606-1789; physics in the module
    docstring).

    Parameters
    ----------
    dz : ndarray, shape (N,)
        Layer thicknesses [m], k = 0 at the bed; all > 0 (thin layers merged).
    n2 : ndarray, shape (N + 1,)
        Buoyancy frequency squared N² [s⁻²] at the interfaces, 0 = bed.
        Only the interior interfaces 1 … N−1 are used.
    ustar : float
        Bottom friction velocity u* [m s⁻¹] (MOM6 ``visc%ustar_BBL``).
    absf : float
        |f|, absolute Coriolis parameter at the column [s⁻¹].
    tke : float
        TKE_col = BBL_EFFIC × drag energy loss [m³ s⁻³] (SD:1709).
    p : BBLParams
        Parameters.

    Returns
    -------
    kd : ndarray, shape (N + 1,)
        BBL diffusivity [m² s⁻¹] at the interfaces, 0 = bed; kd[0] = kd[N] = 0.
    """
    dz = np.asarray(dz, dtype=np.float64)
    n2 = np.asarray(n2, dtype=np.float64)
    nz = dz.size
    if n2.size != nz + 1:
        raise ValueError(f"n2 must have N + 1 = {nz + 1} interfaces, got {n2.size}")
    kd = np.zeros(nz + 1)
    if not (ustar > 0.0):   # deviation: see module docstring (u* = 0 → no mixing)
        return kd

    # --- MOM6 layout: 1-based, top-down -------------------------------------
    dz_m = np.zeros(nz + 2)          # dz(i,k), k = 1 (surface) … nz (bed)
    dz_m[1:nz + 1] = dz[::-1]
    n2_m = np.zeros(nz + 2)          # N2_int(i,K), K = 1 (surface) … nz+1 (bed)
    n2_m[1:nz + 2] = n2[::-1]
    kd_m = np.zeros(nz + 2)          # Kd_BBL(i,j,K)

    n2_min = 0.0                                        # SD:1666-1667
    if p.use_omega:
        n2_min = OMEGA * OMEGA
    i_max_decay = 0.0                                   # SD:2441-2442
    if p.max_decay > 0.0:
        i_max_decay = 1.0 / p.max_decay

    ustar2 = ustar * ustar                              # SD:1685
    # Decay scale min(u*/|f|, max_decay)                  SD:1699-1700
    idecay = i_max_decay
    if (ustar > 0.0) and (absf > i_max_decay * ustar):
        idecay = absf / ustar

    tke_remaining = tke                                 # SD:1709-1711
    # LOTW_BBL_ANSWER_DATE > 20240630                     SD:1712-1717
    dz_above = np.zeros(nz + 2)
    dz_above[1] = DZ_SUBROUNDOFF
    for K in range(2, nz + 2):
        dz_above[K] = dz_above[K - 1] + dz_m[K - 1]
    total_depth = dz_above[nz + 1]
    ustar_d = ustar * total_depth                       # SD:1721
    h_bot = 0.0
    z_bot = 0.0
    # MOM6's Kd_lower (SD:1724, 1784) only builds the layer diffusivity Kd_lay; the
    # oracle returns interface Kd only, so it is not carried (task-1 review, F841).

    for K in range(nz, 1, -1):                          # SD:1728
        k = K
        dz_int = 0.5 * (dz_m[k - 1] + dz_m[k])          # SD:1729

        tke_remaining = math.exp(-idecay * dz_m[k]) * tke_remaining   # SD:1741

        z_bot = z_bot + dz_m[k]                         # SD:1743
        h_bot = h_bot + dz_m[k]                         # SD:1744
        d_minus_z = dz_above[K]                         # SD:1746

        # Law of the wall limited by rotation          SD:1753-1758
        if ustar_d + absf * (h_bot * d_minus_z) == 0.0:
            kd_wall = 0.0
        else:
            kd_wall = ((p.von_karman * ustar2) * (z_bot * d_minus_z)) \
                / (ustar_d + absf * (h_bot * d_minus_z))

        # Work of Kd_wall across the interface          SD:1762
        tke_kd_wall = kd_wall * dz_int * max(n2_m[K], n2_min)

        # Bound Kd by the energy left                   SD:1765-1775
        if tke_kd_wall > 0.0:
            tke_consumed = min(tke_kd_wall, tke_remaining)
            kd_wall = (tke_consumed / tke_kd_wall) * kd_wall
        else:
            if tke_remaining > 0.0:
                kd_wall = p.kd_max
            else:
                kd_wall = 0.0
            tke_consumed = 0.0

        tke_remaining = tke_remaining - tke_consumed    # SD:1779
        kd_m[K] = kd_wall                               # SD:1782, 1785

    kd[:] = kd_m[1:nz + 2][::-1]
    return kd


# ------------------------------------------------------------------------ Columns with thin-layer merge (BBL task 3)
# A layer thinner than h_thin (``core.components.closure.thin_layer_threshold``) is merged before the column scheme,
# exactly as for GLS and Jackson (``core.components.closure.active_layers`` + ``kappa_shear._merge_map``): it joins
# the nearest open layer above it (k' ≥ k), else the nearest below (a top thin stack < h_massless); the merged
# thickness is the sum of the members.
#
# The merge and the interpolation of Kd back onto the model interfaces (below) are **pyOCEAN's choice**, the same
# one made for the ALE GLS and for Jackson; **MOM6's LOTW does no merge**: ``add_LOTW_BBL_diffusivity`` sweeps every
# interface of the column, vanished layers included (SD:1728-1787).  MOM6 merges massless layers only inside its
# shear scheme (``KAPPA_SHEAR_ELIM_MASSLESS``, MOM_kappa_shear.F90:254-307), not here.  Why pyOCEAN merges: a 1 mm
# layer gives dz_int ≈ 0.5 mm at its interfaces, so the mixing work Kd dz_int N² is nearly zero there and, with
# N² ≤ 0 and no Ω² floor, the interface takes KD_MAX (SD:1769-1772).  The unmerged kernel is the one compared with
# compiled MOM6 (task 2, spec "Decisiones de Claude" 6); the merge is tested by the property that a column with
# near-empty layers gives the Kd of the column without them (spec criterion 2).
#
# * N² of the merged column is computed from the **merged densities** (final review I1, class a), as the ALE GLS
#   (``core.components.closure``, ``buoyancy_frequency2(merge(h ρ) / hc, hc, …)``) and Jackson
#   (``kappa_shear``, Σb·h / Σh): ρ of a merged layer is the thickness-weighted mean Σ h_k ρ_k / Σ h_k of its members
#   with h_k > 0, and N²_q = (g/ρ₀)(ρ_{q−1} − ρ_q) / (½(hc_{q−1} + hc_q)) at the interior merged interfaces
#   (``buoyancy_frequency2``, CROCO rho_eos.F:404-411).  The mean is summed as an anomaly about the density ρ_r of
#   the lowest member with h > 0, ρ_q = ρ_r + Σ h_k (ρ_k − ρ_r) / hc_q: the same mean, but a single member gives
#   ρ_r exactly (so a column without thin layers is bit for bit the unmerged ``lotw_bbl_column`` with the N² of
#   ``buoyancy_frequency2``) and members of equal density give it exactly too (Σhρ/Σh loses ~ε ρ/Δρ ≈ 1e-10 of N²
#   for ρ ≈ 1000, Δρ ≈ 2e-3 kg m⁻³).  The earlier version reused the model N² at
#   the bottom of the lowest member, taken over the unmerged spacing ½(h_{k−1} + h_k): a 1 mm layer on top of a 10 m
#   one doubled N² there (1.962e-6 → 3.924e-6) and changed Kd by up to 100 % (interface 7 of
#   tests/test_bbl_mixing.py::test_thin_layers_from_densities_do_not_change_kd: 0.118 → 0).
# * Kd back on the model interfaces: an interface at the bottom of a merged layer takes that merged interface's Kd;
#   an interface inside a merged layer takes the linear interpolation in height between the merged interfaces below
#   and above (weights kf of ``_merge_map``, as Jackson's κ), so the bed interface 0 stays 0 and bed thin layers get
#   values between 0 and the first merged interior Kd.
# * A dry column (Σdz ≤ h_massless), a column with no thick layer or with fewer than two merged layers (no interior
#   interface) gives Kd = 0.  u* = 0 gives 0 (``lotw_bbl_column``).
#
# The Numba column ``_lotw_column_nb`` is a line-by-line copy of ``lotw_bbl_column`` (same operation order, ``x*x``,
# math.exp), so ``lotw_bbl_columns`` is bit for bit with ``lotw_bbl_columns_numpy`` at any thread count
# (tests/test_bbl_mixing.py::test_numba_matches_numpy_oracle, ::test_threads_bit_for_bit).

H_MASSLESS: float = 1.0e-3  # [m] dry-column limit / least total of a surface thin cluster (= closure.H_MASSLESS)


def _merged_n2(rho: np.ndarray, dz: np.ndarray, lay: np.ndarray, hc: np.ndarray, c: int, rho0: float,
               g: float) -> np.ndarray:
    """
    N² (c+1,) [s⁻²] at the interfaces of the merged column from the layer densities (section header):
    ρ_q = ρ_r + Σ h_k (ρ_k − ρ_r) / hc_q over the members with h_k > 0, ρ_r = ρ of the lowest such member,
    N²_q = (g/ρ₀)(ρ_{q−1} − ρ_q) / (½(hc_{q−1} + hc_q)), q = 1 … c−1; 0 at the bed and the surface (unused).
    rho, dz : (N,) model column [kg m⁻³], [m];  lay : (N,) merged layer of each model layer;  hc : (c,) [m].
    """
    rref = np.zeros(c)
    has = np.zeros(c, np.bool_)
    asum = np.zeros(c)
    for k in range(dz.shape[0]):
        if dz[k] > 0.0:
            q = lay[k]
            if not has[q]:
                rref[q] = rho[k]
                has[q] = True
            asum[q] += dz[k] * (rho[k] - rref[q])
    rc = rref + asum / hc
    out = np.zeros(c + 1)
    for q in range(1, c):
        out[q] = (g / rho0) * (rc[q - 1] - rc[q]) / (0.5 * (hc[q - 1] + hc[q]))
    return out


def lotw_bbl_columns_numpy(dz: np.ndarray, rho: np.ndarray, ustar: np.ndarray, absf: np.ndarray, tke: np.ndarray,
                           h_thin: np.ndarray, p: BBLParams = BBLParams(),
                           h_massless: float = H_MASSLESS, rho0: float = 1000.0, g: float = 9.81) -> np.ndarray:
    """
    Energy-limited law-of-the-wall BBL Kd of M columns with thin-layer merging — the **NumPy oracle** of
    ``lotw_bbl_columns`` (merge of the section header, then ``lotw_bbl_column`` per merged column).

    Parameters
    ----------
    dz : (N, M) layer thicknesses [m] ≥ 0, k = 0 at the bed.
    rho : (N, M) layer density [kg m⁻³] (the model EOS); N² is computed on the merged column (section header).
    ustar, absf, tke : (M,) u* [m s⁻¹], |f| [s⁻¹], TKE_col = BBL_EFFIC × drag energy loss [m³ s⁻³].
    h_thin : (M,) thin-layer threshold [m] (``core.components.closure.thin_layer_threshold``).
    p : BBLParams; h_massless : [m] dry-column limit and least total of a surface thin cluster.
    rho0, g : ρ₀ [kg m⁻³] and g [m s⁻²] of N² = −(g/ρ₀) ∂ρ/∂z.

    Returns
    -------
    kd : (N+1, M) BBL diffusivity [m² s⁻¹] at the model interfaces (0 = bed).
    """
    from core.components.closure import active_layers     # lazy: closure will import this module (task 4)
    from physics.mixing.kappa_shear import _merge_map

    dz = np.asarray(dz, float)
    rho = np.asarray(rho, float)
    n, m = dz.shape
    ustar, absf, tke, h_thin = (np.broadcast_to(np.asarray(x, float), (m,)) for x in (ustar, absf, tke, h_thin))
    kd = np.zeros((n + 1, m))
    act = active_layers(dz, h_thin, h_massless)
    for j in range(m):
        dzc = dz[:, j]
        dry = 0.0
        for k in range(n):
            dry += dzc[k]
        if dry <= h_massless or not act[:, j].any():
            continue
        lay, kc, kf, c = _merge_map(dzc, act[:, j])
        if c < 2:
            continue
        hc = np.zeros(c)
        for k in range(n):
            hc[lay[k]] += dzc[k]
        n2c = _merged_n2(rho[:, j], dzc, lay, hc, c, float(rho0), float(g))
        kd_c = lotw_bbl_column(hc, n2c, float(ustar[j]), float(absf[j]), float(tke[j]), p)
        for i in range(n + 1):
            lo, hi = kd_c[kc[i]], kd_c[min(kc[i] + 1, c)]
            f = kf[i]
            if f == 0.0:
                kd[i, j] = lo
            elif f == 1.0:
                kd[i, j] = hi
            else:
                kd[i, j] = (1.0 - f) * lo + f * hi
    return kd


@njit(cache=True)
def _lotw_column_nb(dz, n2, nz, ustar, absf, tke, n2_min, i_max_decay, von_karman, kd_max, kd, dz_above):
    """
    ``lotw_bbl_column`` on the merged column dz (nz,), n2 (nz+1,) (k = 0 at the bed), Kd written to kd (nz+1,)
    (zeroed by the caller); dz_above (≥ nz+1,) is scratch.  Line-by-line copy of the oracle (SD:1606-1789) in model (bottom-up) indices:
    MOM6 index K ↔ model interface nz + 1 − K, layer k ↔ model layer nz − k.
    """
    if not (ustar > 0.0):
        return
    ustar2 = ustar * ustar                              # SD:1685
    idecay = i_max_decay                                # SD:1699-1700
    if (ustar > 0.0) and (absf > i_max_decay * ustar):
        idecay = absf / ustar
    tke_remaining = tke                                 # SD:1709-1711
    # dz_above[K] (MOM6) = DZ_SUBROUNDOFF + Σ of the layers above interface K, summed from the surface down
    dz_above[nz] = DZ_SUBROUNDOFF
    for i in range(nz - 1, -1, -1):
        dz_above[i] = dz_above[i + 1] + dz[i]
    total_depth = dz_above[0]
    ustar_d = ustar * total_depth                       # SD:1721
    h_bot = 0.0
    z_bot = 0.0
    for i in range(1, nz):                              # SD:1728: K = nz … 2  ↔  model interface i = 1 … nz−1
        dz_int = 0.5 * (dz[i] + dz[i - 1])              # SD:1729: 0.5*(dz(k-1) + dz(k)), k-1 above = model i
        tke_remaining = math.exp(-idecay * dz[i - 1]) * tke_remaining   # SD:1741
        z_bot = z_bot + dz[i - 1]                       # SD:1743
        h_bot = h_bot + dz[i - 1]                       # SD:1744
        d_minus_z = dz_above[i]                         # SD:1746
        if ustar_d + absf * (h_bot * d_minus_z) == 0.0:   # SD:1753-1758
            kd_wall = 0.0
        else:
            kd_wall = ((von_karman * ustar2) * (z_bot * d_minus_z)) \
                / (ustar_d + absf * (h_bot * d_minus_z))
        tke_kd_wall = kd_wall * dz_int * max(n2[i], n2_min)   # SD:1762
        if tke_kd_wall > 0.0:                           # SD:1765-1775
            tke_consumed = min(tke_kd_wall, tke_remaining)
            kd_wall = (tke_consumed / tke_kd_wall) * kd_wall
        else:
            if tke_remaining > 0.0:
                kd_wall = kd_max
            else:
                kd_wall = 0.0
            tke_consumed = 0.0
        tke_remaining = tke_remaining - tke_consumed    # SD:1779
        kd[i] = kd_wall                                 # SD:1782, 1785


COL_BLOCK: int = 64  # columns per thread task of ``_columns_nb`` (scratch arrays allocated once per block)


@njit(cache=True, parallel=True)
def _columns_nb(dz, rho, ustar, absf, tke, h_thin, h_massless, n2_min, i_max_decay, von_karman, kd_max, kd_out,
                rho0, g, bad):
    """
    Thin-layer merge + ``_lotw_column_nb`` of every column of dz (N, M), COL_BLOCK columns per thread task.  The
    scratch arrays are allocated once per block and cleared per column (same arithmetic as one allocation per
    column; cost fix of 2026-10-31).  Dry columns (Σdz ≤ h_massless) are skipped before ρ is read; a wet column
    with a non-finite ρ is flagged in bad (M,) and left at Kd = 0 (the caller raises).
    """
    n, mcol = dz.shape
    nblk = (mcol + COL_BLOCK - 1) // COL_BLOCK
    for b in prange(nblk):
        act = np.zeros(n, np.bool_)
        lay = np.empty(n, np.int64)
        rank = np.empty(n, np.int64)
        hc = np.zeros(n)
        kc = np.empty(n + 1, np.int64)
        kf = np.empty(n + 1)
        rref = np.zeros(n)
        has = np.zeros(n, np.bool_)
        asum = np.zeros(n)
        rc = np.empty(n)
        n2c = np.zeros(n + 1)
        kd_c = np.zeros(n + 1)
        dz_above = np.empty(n + 1)
        for j in range(b * COL_BLOCK, min(mcol, (b + 1) * COL_BLOCK)):
            dry = 0.0
            for k in range(n):
                dry += dz[k, j]
            if dry <= h_massless:
                continue
            finite = True
            for k in range(n):
                if not math.isfinite(rho[k, j]):
                    finite = False
            if not finite:
                bad[j] = 1
                continue
            # ``core.components.closure.active_layers``
            act[:] = False
            kt = -1
            for k in range(n):
                if dz[k, j] >= h_thin[j]:
                    act[k] = True
                    kt = k
            if 0 <= kt < n - 1:
                top = 0.0
                for k in range(n):
                    if k > kt:
                        top += dz[k, j]
                if top >= h_massless:
                    act[kt + 1] = True
            if kt < 0:
                continue
            # ``kappa_shear._merge_map``
            nxt = -1
            for k in range(n - 1, -1, -1):
                if act[k]:
                    nxt = k
                lay[k] = nxt
            prv = -1
            r = -1
            for k in range(n):
                if act[k]:
                    prv = k
                    r += 1
                rank[k] = r
                if lay[k] < 0:
                    lay[k] = prv
            for k in range(n):
                lay[k] = rank[lay[k]]
            c = r + 1
            if c < 2:
                continue
            hc[:c] = 0.0
            for k in range(n):
                hc[lay[k]] += dz[k, j]
            d_in = 0.0
            for i in range(n):
                kc[i] = lay[i]
                if i == 0 or lay[i] != lay[i - 1]:
                    kf[i] = 0.0
                    d_in = dz[i, j]
                else:
                    kf[i] = d_in / hc[lay[i]]
                    d_in = d_in + dz[i, j]
            kc[n] = lay[n - 1]
            kf[n] = 1.0
            # N² from the merged densities (``_merged_n2``)
            rref[:c] = 0.0
            has[:c] = False
            asum[:c] = 0.0
            for k in range(n):
                if dz[k, j] > 0.0:
                    q = lay[k]
                    if not has[q]:
                        rref[q] = rho[k, j]
                        has[q] = True
                    asum[q] += dz[k, j] * (rho[k, j] - rref[q])
            for q in range(c):
                rc[q] = rref[q] + asum[q] / hc[q]
            n2c[:c + 1] = 0.0
            for q in range(1, c):
                n2c[q] = (g / rho0) * (rc[q - 1] - rc[q]) / (0.5 * (hc[q - 1] + hc[q]))
            kd_c[:c + 1] = 0.0
            _lotw_column_nb(hc, n2c, c, ustar[j], absf[j], tke[j], n2_min, i_max_decay, von_karman, kd_max, kd_c,
                            dz_above)
            for i in range(n + 1):
                q = kc[i]
                q1 = q + 1
                if q1 > c:
                    q1 = c
                lo = kd_c[q]
                hi = kd_c[q1]
                f = kf[i]
                if f == 0.0:
                    kd_out[i, j] = lo
                elif f == 1.0:
                    kd_out[i, j] = hi
                else:
                    kd_out[i, j] = (1.0 - f) * lo + f * hi


def lotw_bbl_columns(dz: np.ndarray, rho: np.ndarray, ustar: np.ndarray, absf: np.ndarray, tke: np.ndarray,
                     h_thin: np.ndarray, p: BBLParams = BBLParams(), h_massless: float = H_MASSLESS,
                     rho0: float = 1000.0, g: float = 9.81) -> np.ndarray:
    """
    Energy-limited law-of-the-wall BBL Kd [m² s⁻¹] of M columns with thin-layer merging, compiled (Numba, threads
    over the columns); bit for bit with ``lotw_bbl_columns_numpy`` (same arguments and return: dz (N, M) with
    k = 0 at the bed, rho (N, M) layer density [kg m⁻³], ustar/absf/tke/h_thin (M,) or scalars, rho0/g of
    N² = −(g/ρ₀) ∂ρ/∂z on the merged column; Kd (N+1, M) at the model interfaces).
    """
    dz = np.ascontiguousarray(dz, dtype=float)
    if dz.ndim != 2:
        raise ValueError("lotw_bbl_columns: dz must be (N, M)")
    n, m = dz.shape
    if not np.isfinite(dz).all() or (dz < 0.0).any():
        raise ValueError("lotw_bbl_columns: dz must be finite and >= 0 (dry columns: zero thicknesses)")
    rho = np.ascontiguousarray(rho, dtype=float)
    if rho.shape != (n, m):
        raise ValueError(f"lotw_bbl_columns: rho must have shape {(n, m)}, got {rho.shape}")
    cols = []
    for name, x in (("ustar", ustar), ("absf", absf), ("tke", tke), ("h_thin", h_thin)):
        arr = np.asarray(x, float)
        if arr.ndim != 0 and arr.shape != (m,):
            raise ValueError(f"lotw_bbl_columns: {name} must be a scalar or have shape ({m},), got {arr.shape}")
        # a NaN u* would fail ``ustar > 0`` in the kernel and give Kd = 0 silently (task 3 review M2)
        if not np.isfinite(arr).all() or (arr < 0.0).any():
            raise ValueError(f"lotw_bbl_columns: {name} must be finite and >= 0")
        cols.append(np.ascontiguousarray(np.broadcast_to(arr, (m,))))
    ustar, absf, tke, h_thin = cols
    n2_min = OMEGA * OMEGA if p.use_omega else 0.0                       # SD:1666-1667
    i_max_decay = 1.0 / p.max_decay if p.max_decay > 0.0 else 0.0       # SD:2441-2442
    kd = np.zeros((n + 1, m))
    bad = np.zeros(m, np.int8)
    _columns_nb(dz, rho, ustar, absf, tke, h_thin, float(h_massless), float(n2_min), float(i_max_decay),
                float(p.von_karman), float(p.kd_max), kd, float(rho0), float(g), bad)
    if bad.any():                                                       # ρ is read only in wet columns
        raise ValueError(f"lotw_bbl_columns: rho must be finite in the wet columns ({int(bad.sum())} are not)")
    return kd
