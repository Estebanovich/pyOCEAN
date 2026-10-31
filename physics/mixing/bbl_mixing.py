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
    kd_lower = 0.0                                      # SD:1724

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
        kd_lower = kd_wall                              # SD:1784 (Kd_lay only)

    kd[:] = kd_m[1:nz + 2][::-1]
    return kd
